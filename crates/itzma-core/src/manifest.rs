//! The namespace manifest and its compare-and-swap update.
//!
//! The manifest is the one mutable object in a namespace. It records the
//! schema, which segment files are live, and how much of the WAL has been
//! folded into segments. Everything else in a namespace is written once and
//! never modified.
//!
//! It also carries the namespace's WAL and segment format levels, which is how
//! a newer build tells older ones to stay away (see [`Manifest::wal_format`]
//! and [`Manifest::segment_format`]).
//!
//! Two operations change it, and both rely on the object store to arbitrate
//! between concurrent writers:
//!
//! - [`create`] writes it with put-if-absent, so exactly one caller creates a
//!   namespace.
//! - [`update`] and [`update_if`] rewrite it with a conditional put tied to
//!   the version that was read, so concurrent updates cannot silently
//!   overwrite each other. Compaction publishes segments through
//!   [`update_if`].

use std::collections::BTreeMap;

use bytes::Bytes;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt, PutMode, PutOptions, UpdateVersion};
use serde::{Deserialize, Serialize};

use crate::document::Schema;
use crate::error::{ItzmaError, Result};
use crate::{segment, wal};

/// Version stamped into every manifest. [`load`] rejects any other value with
/// [`ItzmaError::UnsupportedFormat`].
pub const FORMAT_VERSION: u32 = 1;

/// How many read-modify-write rounds [`update`] tries before giving up with
/// [`ItzmaError::ManifestContention`].
pub const MAX_UPDATE_ATTEMPTS: u32 = 32;

/// The contents of `<namespace>/manifest/current.json`.
///
/// # JSON form
///
/// ```json
/// {
///   "format_version": 1,
///   "schema": {"dim": 2, "metric": "l2"},
///   "live_segments": [],
///   "wal_folded_through": 0,
///   "wal_format": 5,
///   "segment_format": 2,
///   "indexes": {}
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    /// Encoding version of this object. See [`FORMAT_VERSION`].
    pub format_version: u32,
    /// Vector shape and metric, fixed when the namespace was created.
    pub schema: Schema,
    /// Ids of the live segments, oldest first. Each id names the WAL range
    /// the segment covers (see [`segment`]); the ranges are
    /// contiguous, start at 1, and end at `wal_folded_through`. Empty until
    /// the first compaction.
    pub live_segments: Vec<String>,
    /// Highest WAL sequence number already folded into segments. WAL objects
    /// up to and including this seq are ignored on read, and the next append
    /// starts at `wal_folded_through + 1`. Zero means nothing has been folded.
    pub wal_folded_through: u64,
    /// The highest WAL object format this namespace may contain. A build
    /// whose own [`wal::FORMAT_VERSION`] is lower refuses the namespace in
    /// [`load`] with [`ItzmaError::WalFormatTooNew`], before it reads any WAL
    /// object it could not decode.
    ///
    /// Only a newer build raises this, once every build that touches the
    /// namespace can read the new format, with
    /// [`Namespace::upgrade_wal_format`](crate::Namespace::upgrade_wal_format).
    /// A keyed write needs level 3.
    pub wal_format: u32,
    /// The newest segment format compaction may write into this namespace, and
    /// the highest one it may contain. A build whose own
    /// [`segment::FORMAT_VERSION`] is lower refuses the namespace in [`load`]
    /// with [`ItzmaError::SegmentFormatTooNew`].
    ///
    /// A build that reads version 3 also reads version 2, so the level only
    /// decides what a fold or merge writes: at 2 it writes version 2, at 3
    /// version 3. It is raised, once every build that touches the namespace
    /// reads the new version, with
    /// [`Namespace::upgrade_segment_format`](crate::Namespace::upgrade_segment_format).
    pub segment_format: u32,
    /// The index built for each live segment that has one, by segment id.
    /// A segment that is not listed is read whole, as before. A
    /// manifest without the field has none. Derived data: compaction drops an
    /// entry when it retires the segment, and nothing else depends on it.
    #[serde(default)]
    pub indexes: BTreeMap<String, IndexEntry>,
}

/// Which index object belongs to a segment: the object is
/// `segments/<id>/index-<generation>.bin` and was built by the algorithm
/// `kind`.
///
/// JSON form: `{"gen": 1, "kind": "sq8"}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexEntry {
    /// Starts at 1 and goes up by one each time the segment is re-indexed, so a
    /// rebuild never overwrites the object a reader may be using. (`gen` is a
    /// reserved word in Rust 2024, hence the rename.)
    #[serde(rename = "gen")]
    pub generation: u32,
    /// The algorithm's name, `sq8` for the only one built so far. A name this
    /// build does not know is left alone, never guessed at.
    pub kind: String,
}

impl Manifest {
    /// A fresh manifest for a new namespace: current format version, no
    /// segments, nothing folded, the newest WAL format this build writes and
    /// the segment level new namespaces start at ([`segment::DEFAULT_FORMAT`]).
    pub fn new(schema: Schema) -> Self {
        Self {
            format_version: FORMAT_VERSION,
            schema,
            live_segments: Vec::new(),
            wal_folded_through: 0,
            wal_format: wal::FORMAT_VERSION,
            segment_format: segment::DEFAULT_FORMAT,
            indexes: BTreeMap::new(),
        }
    }
}

/// Object key of a namespace's manifest: `<namespace>/manifest/current.json`.
///
/// Does not validate `namespace`; that is [`Namespace`](crate::Namespace)'s job.
pub fn manifest_path(namespace: &str) -> Path {
    Path::from(format!("{namespace}/manifest/current.json"))
}

/// Serialize to JSON bytes, tagging any failure with a label instead of a key
/// since the object has not been written yet.
fn encode(manifest: &Manifest) -> Result<Bytes> {
    serde_json::to_vec(manifest)
        .map(Bytes::from)
        .map_err(|source| ItzmaError::Serde {
            path: "manifest".into(),
            source,
        })
}

/// Write the manifest for a new namespace, failing if one already exists.
///
/// Uses `PutMode::Create` (put-if-absent), which the store performs
/// atomically. When several callers race to create the same id, exactly one
/// succeeds and the rest see [`ItzmaError::NamespaceExists`].
///
/// # Errors
///
/// - [`ItzmaError::NamespaceExists`] if the manifest is already there.
/// - [`ItzmaError::Storage`] for any other backend failure.
///
/// # Examples
///
/// ```
/// # #[tokio::main]
/// # async fn main() -> itzma_core::Result<()> {
/// use itzma_core::manifest::{self, Manifest};
/// use itzma_core::{ItzmaError, Metric, Schema};
/// use object_store::memory::InMemory;
///
/// let store = InMemory::new();
/// let m = Manifest::new(Schema::new(3, Metric::Cosine));
///
/// manifest::create(&store, "docs", &m).await?;
/// let again = manifest::create(&store, "docs", &m).await;
/// assert!(matches!(again, Err(ItzmaError::NamespaceExists(_))));
/// # Ok(())
/// # }
/// ```
pub async fn create(store: &dyn ObjectStore, namespace: &str, manifest: &Manifest) -> Result<()> {
    let opts = PutOptions::from(PutMode::Create);
    match store
        .put_opts(&manifest_path(namespace), encode(manifest)?.into(), opts)
        .await
    {
        Ok(_) => Ok(()),
        Err(object_store::Error::AlreadyExists { .. }) => {
            Err(ItzmaError::NamespaceExists(namespace.to_string()))
        }
        Err(e) => Err(e.into()),
    }
}

/// Read and decode the manifest, returning it with the version token needed
/// to update it conditionally.
///
/// The returned [`UpdateVersion`] carries the object's ETag and version id,
/// whichever the backend provides. Hand it to `PutMode::Update` to say "only
/// write if nobody changed this since I read it".
///
/// # Errors
///
/// - [`ItzmaError::NamespaceNotFound`] if there is no manifest.
/// - [`ItzmaError::Serde`] if the object is not valid manifest JSON. The error
///   names the object key.
/// - [`ItzmaError::UnsupportedFormat`] if `format_version` is not
///   [`FORMAT_VERSION`].
/// - [`ItzmaError::WalFormatTooNew`] if `wal_format` is above what this build
///   reads. Every operation on a namespace starts here, so this is the one
///   place an older build is turned away, and it has written nothing yet.
/// - [`ItzmaError::SegmentFormatTooNew`] if `segment_format` is above what this
///   build reads. Checked after the WAL level, and turned away the same way.
/// - [`ItzmaError::Storage`] for any other backend failure.
pub async fn load(store: &dyn ObjectStore, namespace: &str) -> Result<(Manifest, UpdateVersion)> {
    let path = manifest_path(namespace);
    let result = match store.get(&path).await {
        Ok(r) => r,
        Err(object_store::Error::NotFound { .. }) => {
            return Err(ItzmaError::NamespaceNotFound(namespace.to_string()));
        }
        Err(e) => return Err(e.into()),
    };
    let version = UpdateVersion {
        e_tag: result.meta.e_tag.clone(),
        version: result.meta.version.clone(),
    };
    let bytes = result.bytes().await?;
    let manifest: Manifest =
        serde_json::from_slice(&bytes).map_err(|source| ItzmaError::Serde {
            path: path.to_string(),
            source,
        })?;
    if manifest.format_version != FORMAT_VERSION {
        return Err(ItzmaError::UnsupportedFormat {
            version: manifest.format_version,
        });
    }
    if manifest.wal_format > wal::FORMAT_VERSION {
        return Err(ItzmaError::WalFormatTooNew {
            found: manifest.wal_format,
            supported: wal::FORMAT_VERSION,
        });
    }
    if manifest.segment_format > segment::FORMAT_VERSION {
        return Err(ItzmaError::SegmentFormatTooNew {
            found: manifest.segment_format,
            supported: segment::FORMAT_VERSION,
        });
    }
    Ok((manifest, version))
}

/// Atomically change the manifest, unless the closure declines.
///
/// Like [`update`], each round loads the current manifest and applies `f`, and
/// writes the result back with `PutMode::Update` tied to the version that was
/// read. A lost race restarts the round from a fresh read. The closure may
/// therefore run several times and **must be re-runnable**.
///
/// The closure returns `true` to publish its change and `false` to decline.
/// Declining writes nothing and returns `Ok(None)`. Compaction uses this to
/// say "publish my segment only if the manifest still looks the way it did
/// when I planned it".
///
/// # Errors
///
/// Same as [`update`]: [`ItzmaError::ManifestContention`] after 32 lost
/// rounds, anything [`load`] can return, and [`ItzmaError::Storage`] for
/// backend failures other than a lost race.
///
/// # Examples
///
/// ```
/// # #[tokio::main]
/// # async fn main() -> itzma_core::Result<()> {
/// use itzma_core::manifest::{self, Manifest};
/// use itzma_core::{Metric, Schema};
/// use object_store::memory::InMemory;
///
/// let store = InMemory::new();
/// manifest::create(&store, "docs", &Manifest::new(Schema::new(3, Metric::L2))).await?;
///
/// // Publish only if nothing has been folded yet.
/// let first = manifest::update_if(&store, "docs", |m| {
///     if m.wal_folded_through != 0 {
///         return false;
///     }
///     m.wal_folded_through = 4;
///     true
/// })
/// .await?;
/// assert_eq!(first.map(|m| m.wal_folded_through), Some(4));
///
/// // The same call now declines.
/// let second = manifest::update_if(&store, "docs", |m| m.wal_folded_through == 0).await?;
/// assert!(second.is_none());
/// # Ok(())
/// # }
/// ```
pub async fn update_if<F>(
    store: &dyn ObjectStore,
    namespace: &str,
    mut f: F,
) -> Result<Option<Manifest>>
where
    F: FnMut(&mut Manifest) -> bool,
{
    let path = manifest_path(namespace);
    for _ in 0..MAX_UPDATE_ATTEMPTS {
        let (mut manifest, version) = load(store, namespace).await?;
        if !f(&mut manifest) {
            return Ok(None);
        }
        let opts = PutOptions::from(PutMode::Update(version));
        match store.put_opts(&path, encode(&manifest)?.into(), opts).await {
            Ok(_) => return Ok(Some(manifest)),
            Err(object_store::Error::Precondition { .. }) => continue,
            Err(e) => return Err(e.into()),
        }
    }
    Err(ItzmaError::ManifestContention {
        attempts: MAX_UPDATE_ATTEMPTS,
    })
}

/// Atomically change the manifest by read, modify, conditional write.
///
/// Each round loads the current manifest, applies `f` to it, and writes the
/// result back with `PutMode::Update` tied to the version that was read. If
/// another writer got in first the store answers with a precondition failure,
/// and the round restarts from a fresh read, so `f` runs again on the new
/// contents. That means `f` may be called several times and **must be
/// re-runnable**: derive the change from the manifest it is given, and do not
/// carry state between calls.
///
/// The winning manifest is returned. After [`MAX_UPDATE_ATTEMPTS`] lost rounds
/// the call gives up. Use [`update_if`] when the change should be abandoned
/// if the manifest is no longer what the caller expected.
///
/// # Errors
///
/// - [`ItzmaError::ManifestContention`] after 32 lost rounds.
/// - Anything [`load`] can return.
/// - [`ItzmaError::Storage`] if the backend rejects the write for any reason
///   other than a lost race. `LocalFileSystem` does not implement conditional
///   update, so on that backend this is always a `NotImplemented` storage
///   error. Use `InMemory` or a cloud store.
///
/// # Examples
///
/// ```
/// # #[tokio::main]
/// # async fn main() -> itzma_core::Result<()> {
/// use itzma_core::manifest::{self, Manifest};
/// use itzma_core::{Metric, Schema};
/// use object_store::memory::InMemory;
///
/// let store = InMemory::new();
/// let m = Manifest::new(Schema::new(3, Metric::L2));
/// manifest::create(&store, "docs", &m).await?;
///
/// let updated = manifest::update(&store, "docs", |m| m.wal_folded_through += 1).await?;
/// assert_eq!(updated.wal_folded_through, 1);
/// # Ok(())
/// # }
/// ```
pub async fn update<F>(store: &dyn ObjectStore, namespace: &str, mut f: F) -> Result<Manifest>
where
    F: FnMut(&mut Manifest),
{
    let outcome = update_if(store, namespace, |m| {
        f(m);
        true
    })
    .await?;
    match outcome {
        Some(manifest) => Ok(manifest),
        None => unreachable!("the closure above always accepts"),
    }
}
