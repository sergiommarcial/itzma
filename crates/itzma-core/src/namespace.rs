//! The [`Namespace`] handle: the public entry point for reading and writing.
//!
//! A namespace is an isolated collection of documents that share one schema.
//! On object storage it is just a key prefix, `<id>/`, holding a manifest and a
//! WAL (see [`manifest`] and [`wal`]). A handle itself holds nothing but the
//! schema and a sequence hint, so any number of handles, in any number of
//! processes, can point at the same namespace — the exception is
//! [`Namespace::with_cache`], which opts the query path into reusing a
//! caller-supplied [`CacheStore`](crate::CacheStore) across calls; the
//! manifest is never cached, so this changes nothing about write-then-query
//! visibility.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use object_store::ObjectStore;

use crate::budget::{DEFAULT_READ_BUDGET, ReadBudget};
use crate::columns::DocColumns;
use crate::compaction::{self, Compaction, CompactionOptions};
use crate::decoded::{DecodedCache, DecodedRef};
use crate::document;
use crate::document::{Document, Schema, SparseVector, WalOp};
use crate::error::{ItzmaError, Result};
use crate::fetch::FetchPolicy;
use crate::filter::{self, FilterOptions, FilterPage};
use crate::gc::{self, GcOptions, GcReport};
use crate::index;
use crate::indexer::{self, IndexOptions, IndexReport, MIN_SEGMENT_FORMAT, Outcome};
use crate::maintain::{self, MaintenanceOptions, MaintenanceReport};
use crate::manifest::{self, Manifest};
use crate::predicate::Predicate;
use crate::query::{self, Builtin, QueryOptions};
use crate::recover::{self, RecoverReport};
use crate::schema_object;
use crate::search::Hit;
use crate::segment;
use crate::sparse_query::{self, SparseQueryOptions};
use crate::stream::{self, StreamError, StreamOptions, StreamReport};
use crate::view;
use crate::wal::{self, WriteKey};
use crate::wal_codec;

/// A handle to one namespace on an object store.
///
/// Get one with [`Namespace::create`] (new namespace) or [`Namespace::open`]
/// (existing). Handles are cheap: they hold the store, the id, the schema and
/// a sequence hint. Methods take `&self`, so a handle can be shared across
/// tasks, and separate handles on the same namespace are safe to use
/// concurrently. Writers are sequenced by the object store, not by the
/// handle.
///
/// # Consistency
///
/// - A write is durable once `upsert` or `delete` returns.
/// - A `query` that starts after a write returned sees that write, even from a
///   different handle or process (read-your-writes).
/// - Readers see a consistent prefix of the log: all of a batch or none of it,
///   and never a later batch without an earlier one.
/// - Two writers racing get distinct seqs and are ordered by seq. If both
///   touch the same id, the higher seq wins.
///
/// # Examples
///
/// ```
/// # #[tokio::main]
/// # async fn main() -> itzma_core::Result<()> {
/// use std::sync::Arc;
///
/// use itzma_core::{Document, Metric, Namespace, Schema};
/// use object_store::memory::InMemory;
///
/// let store = Arc::new(InMemory::new());
/// let ns = Namespace::create(store, "docs", Schema::new(2, Metric::L2)).await?;
///
/// ns.upsert(vec![
///     Document { id: "near".into(), vector: vec![1.0, 0.0], attrs: None, sparse: None },
///     Document { id: "far".into(), vector: vec![9.0, 9.0], attrs: None, sparse: None },
/// ])
/// .await?;
///
/// let hits = ns.query(&[0.0, 0.0], 1).await?;
/// assert_eq!(hits[0].id, "near");
/// # Ok(())
/// # }
/// ```
pub struct Namespace {
    store: Arc<dyn ObjectStore>,
    /// The store the query path reads segment, index, and WAL-tail bytes
    /// through. Defaults to the same value as `store` — set it with
    /// [`with_cache`](Self::with_cache) to let repeated queries reuse a
    /// [`CacheStore`](crate::CacheStore). The manifest is never read through
    /// this field; every manifest load always uses `store`, so write-then-
    /// query visibility is unaffected either way (see `with_cache`'s doc).
    query_store: Arc<dyn ObjectStore>,
    id: String,
    schema: Schema,
    /// Where the next append starts probing. A stale hint costs one extra
    /// listing on collision, never correctness.
    next_seq: AtomicU64,
    /// The namespace's `wal_format` as this handle last saw it. Only keyed
    /// writes look at it, and it only ever goes up.
    wal_level: AtomicU32,
    /// The most bytes one query or one compaction run may read.
    read_budget: u64,
    /// How ranged reads turn "these pieces of this object" into requests.
    /// Defaults to [`FetchPolicy::object_store`]; set it with
    /// [`with_fetch_policy`](Self::with_fetch_policy).
    fetch_policy: FetchPolicy,
    /// Parsed structures kept between queries, set with
    /// [`with_decoded_cache`](Self::with_decoded_cache).
    decoded_cache: Option<Arc<DecodedCache>>,
    /// The last manifest and folded WAL tail the query path read.
    views: view::Views,
}

/// Ids become key prefixes, so they are restricted to characters that cannot
/// introduce a `/` or otherwise collide with another namespace's keys.
pub(crate) fn validate_id(id: &str) -> Result<()> {
    let ok = (1..=64).contains(&id.len())
        && id
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-');
    if ok {
        Ok(())
    } else {
        Err(ItzmaError::InvalidNamespaceId(id.to_string()))
    }
}

/// How many times `query` reads against a manifest before it gives up, when
/// the segments it needs keep vanishing because the manifest keeps moving.
const MAX_READ_ATTEMPTS: u32 = 3;

impl Namespace {
    /// Create a new, empty namespace and return a handle to it.
    ///
    /// Writes the manifest with put-if-absent. If several callers race to
    /// create the same id, exactly one succeeds. The schema is fixed from this
    /// point on, and is written a second time to `schema.json`, an object that
    /// is never changed, so losing the manifest does not lose it.
    ///
    /// The manifest is written first and decides whether the namespace exists.
    /// If the call fails after that, before the schema object is written, the
    /// namespace exists: `open` it, which writes the object. A create that
    /// finds a schema object for the id holding a different schema is refused
    /// with [`ItzmaError::NamespaceExists`] and writes nothing.
    ///
    /// Creation is explicit, unlike the implicit create-on-first-write,
    /// because the dimension and metric have to be declared up front.
    ///
    /// # Errors
    ///
    /// - [`ItzmaError::InvalidNamespaceId`] unless `id` matches
    ///   `[a-z0-9_-]{1,64}`.
    /// - [`ItzmaError::InvalidSchema`] if `schema.dim == 0`, or the attribute
    ///   schema has more than 256 fields, a field name outside
    ///   `[A-Za-z0-9_]{1,64}`, or is free-form.
    /// - [`ItzmaError::NamespaceExists`] if the id is taken.
    /// - [`ItzmaError::UnsupportedFormat`] or [`ItzmaError::Serde`] if the id
    ///   has a schema object this build cannot read.
    /// - [`ItzmaError::Storage`] for any backend failure.
    pub async fn create(store: Arc<dyn ObjectStore>, id: &str, schema: Schema) -> Result<Self> {
        validate_id(id)?;
        if schema.dim == 0 {
            return Err(ItzmaError::InvalidSchema("dim must be at least 1"));
        }
        schema.attrs.check_declaration()?;
        if let Some(existing) = schema_object::read(&*store, id).await? {
            if existing != schema {
                return Err(ItzmaError::NamespaceExists(id.to_string()));
            }
        }
        let manifest = Manifest::new(schema.clone());
        manifest::create(&*store, id, &manifest).await?;
        schema_object::write(&*store, id, &schema).await?;
        Ok(Self::from_manifest(store, id, &manifest))
    }

    /// Open an existing namespace and return a handle to it.
    ///
    /// Reads the manifest to learn the schema and where the WAL resumes. The
    /// handle keeps no other state from this read: each later `query` loads
    /// the manifest again.
    ///
    /// Then it reads `schema.json` and checks it holds the manifest's schema.
    /// If the object is missing, `open` writes it from the manifest, which
    /// finishes a create that stopped between its two writes and covers a
    /// namespace made before the object existed. That makes `open`
    /// a write, once, for such a namespace, so it needs write access then.
    ///
    /// # Errors
    ///
    /// - [`ItzmaError::InvalidNamespaceId`] for a malformed id.
    /// - [`ItzmaError::NamespaceNotFound`] if no such namespace exists.
    /// - [`ItzmaError::CorruptManifest`] if `schema.json` holds a different
    ///   schema from the manifest. Nothing is changed.
    /// - [`ItzmaError::UnsupportedFormat`] if the manifest was written by an
    ///   incompatible version.
    /// - [`ItzmaError::WalFormatTooNew`] if a newer build raised the
    ///   namespace's WAL format past what this build reads. Every
    ///   method that loads the manifest returns it too.
    /// - [`ItzmaError::Serde`] / [`ItzmaError::Storage`] on a corrupt object
    ///   or backend failure.
    pub async fn open(store: Arc<dyn ObjectStore>, id: &str) -> Result<Self> {
        validate_id(id)?;
        let (manifest, _version) = manifest::load(&*store, id).await?;
        // A namespace with no attribute schema comes from a format this
        // build does not read.
        if manifest.schema.attrs.is_free_form() {
            return Err(ItzmaError::UnsupportedFormat { version: 1 });
        }
        schema_object::ensure(&*store, id, &manifest.schema).await?;
        Ok(Self::from_manifest(store, id, &manifest))
    }

    /// Write a new manifest for a namespace that has lost its own, and return a
    /// handle to it.
    ///
    /// The schema comes from `schema.json`. The live segments are a chain from
    /// seq 1 built from the segment names, taking at each step the segment that
    /// reaches furthest and decodes, and the WAL past its end is left as the
    /// tail. A segment that is missing or does not decode only shortens the
    /// chain: the WAL from there on is read instead, and the next
    /// [`compact`](Self::compact) rebuilds the segment. The new manifest carries
    /// the newest WAL format level this build writes, and as segment level the
    /// newest version any segment object of the namespace was written in, or
    /// [`segment::DEFAULT_FORMAT`] if that is higher (a raise with no segment
    /// written since leaves no trace, so it comes back at the default). A build
    /// that reads less refuses the namespace. Recovering asserts that every
    /// build that touches it reads what this one does. Writers are not held up,
    /// since they never read the manifest.
    ///
    /// Only the manifest is written, and only with put-if-absent. WAL objects,
    /// segments and `schema.json` are read and left as they are, and a segment
    /// the chain does not use is not deleted (that is [`gc`](Self::gc)'s job).
    /// Segments are checked by decoding them one at a time, under the default
    /// read budget, so a run reads every live segment once.
    ///
    /// # Errors
    ///
    /// - [`ItzmaError::InvalidNamespaceId`] for a malformed id.
    /// - [`ItzmaError::ManifestExists`] if the namespace has a manifest, or
    ///   another recovery wrote one first. Nothing is changed.
    /// - [`ItzmaError::Unrecoverable`] if `schema.json` is missing, or a range
    ///   of seqs has no readable copy (no segment that decodes covers it, and
    ///   its WAL objects are missing or fenced). The error names what to restore
    ///   from a replica, a versioned bucket or a backup, and a rerun can
    ///   succeed once it is back. Nothing is written.
    /// - [`ItzmaError::UnsupportedFormat`] if a segment is of a format this
    ///   build does not read. That is a newer build's data, not damage, so the
    ///   run stops and does not skip it.
    /// - [`ItzmaError::ReadBudgetExceeded`] if a single segment is over the
    ///   default read budget.
    /// - [`ItzmaError::Storage`] for any backend failure.
    pub async fn recover(store: Arc<dyn ObjectStore>, id: &str) -> Result<(Self, RecoverReport)> {
        validate_id(id)?;
        let (manifest, report) = recover::rebuild(&*store, id).await?;
        Ok((Self::from_manifest(store, id, &manifest), report))
    }

    /// Everything `upsert` and `upsert_keyed` check about one document before
    /// any storage call: the id, the vector, the sparse vector and the
    /// attributes.
    fn validate_doc(&self, doc: &Document) -> Result<()> {
        doc.validate_id()?;
        self.schema.validate(&doc.vector)?;
        doc.validate_sparse()?;
        self.schema.attrs.validate(doc.attrs.as_ref())
    }

    /// Build a handle from a loaded manifest, seeding the sequence hint just
    /// past the folded prefix.
    fn from_manifest(store: Arc<dyn ObjectStore>, id: &str, manifest: &Manifest) -> Self {
        Self {
            query_store: store.clone(),
            store,
            id: id.to_string(),
            schema: manifest.schema.clone(),
            next_seq: AtomicU64::new(manifest.wal_folded_through + 1),
            wal_level: AtomicU32::new(manifest.wal_format),
            read_budget: DEFAULT_READ_BUDGET,
            fetch_policy: FetchPolicy::object_store(),
            decoded_cache: None,
            views: view::Views::new(),
        }
    }

    /// Set the most bytes one [`query`](Self::query) or one
    /// [`compact`](Self::compact) run may read, and return the handle. The
    /// default is [`DEFAULT_READ_BUDGET`], 1 GiB. The budget belongs to this
    /// handle: it is not stored, and another handle on the same namespace
    /// keeps its own.
    ///
    /// Every segment and WAL object a run reads is counted before its body
    /// is, and a run that would go over stops with
    /// [`ItzmaError::ReadBudgetExceeded`] and writes nothing. The count is
    /// stored bytes (a compressed WAL object counts as its uncompressed
    /// size), a proxy for memory that does not see the extra weight of
    /// attributes once decoded. Since a query or a merge holds every live
    /// document in memory, this is also the largest namespace either can
    /// handle.
    pub fn with_read_budget(mut self, bytes: u64) -> Self {
        self.read_budget = bytes;
        self
    }

    /// Let the query path (`query`, `query_with`) read segment, index, and WAL-
    /// tail bytes through `cached` instead of the store this handle was opened
    /// with, and return the handle. Pass a [`CacheStore`](crate::CacheStore)
    /// wrapping the same backing store to let repeated queries reuse warm
    /// RAM/NVMe bytes instead of re-fetching from the object store every time.
    ///
    /// The manifest is never affected: every query still loads it fresh from
    /// the store this handle was opened with, on every call, so a write is
    /// visible to the very next query exactly as it is without this method —
    /// `cached` only ever serves segment, index, and WAL-tail reads. Compaction,
    /// GC, index builds, and `inspect` are also unaffected; they always use the
    /// store this handle was opened with, never `cached`. The manifest is
    /// excluded so a write is always visible to the next query.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use itzma_core::{CacheOptions, CacheStore, Namespace, Schema, Metric};
    /// use object_store::memory::InMemory;
    ///
    /// # #[tokio::main]
    /// # async fn main() {
    /// let store: Arc<dyn object_store::ObjectStore> = Arc::new(InMemory::new());
    /// let ns = Namespace::create(store.clone(), "docs", Schema::new(4, Metric::L2))
    ///     .await
    ///     .unwrap();
    /// let cache = CacheStore::new(store, CacheOptions::default()).unwrap();
    /// let ns = ns.with_cache(Arc::new(cache));
    /// # let _ = ns;
    /// # }
    /// ```
    pub fn with_cache(mut self, cached: Arc<dyn ObjectStore>) -> Self {
        self.query_store = cached;
        self
    }

    /// Keep what the query path parses (index heads and partitions, segment
    /// directories, Bloom filters, id, attribute and tombstone blocks, sparse
    /// term directories, and whole segments the flat path decodes) in `cache`
    /// between queries, and return the handle.
    ///
    /// A structure found in the cache is neither fetched nor charged to the
    /// read budget, so a warm query can read nothing from the store but the
    /// manifest and the WAL listing (the handle keeps its folded tail itself).
    /// One cache can be
    /// shared by many handles, on one store or several. Compaction, GC, index
    /// builds, `inspect` and `recover` never use it.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use itzma_core::{DecodedCache, DecodedCacheOptions, Metric, Namespace, Schema};
    /// use object_store::memory::InMemory;
    ///
    /// # #[tokio::main]
    /// # async fn main() -> itzma_core::Result<()> {
    /// let cache = DecodedCache::new(DecodedCacheOptions::default());
    /// let ns = Namespace::create(Arc::new(InMemory::new()), "docs", Schema::new(4, Metric::L2))
    ///     .await?
    ///     .with_decoded_cache(cache);
    /// let hits = ns.query(&[0.0; 4], 10).await?;
    /// assert!(hits.is_empty());
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn with_decoded_cache(mut self, cache: Arc<DecodedCache>) -> Self {
        self.decoded_cache = Some(cache);
        self
    }

    /// The decoded cache bound to the query store, if the handle has one.
    fn decoded(&self) -> Option<DecodedRef> {
        self.decoded_cache
            .as_ref()
            .map(|cache| cache.handle(&self.query_store))
    }

    /// Set how ranged reads turn wanted byte ranges into requests,
    /// and return the handle. The default, [`FetchPolicy::object_store`], is
    /// tuned for S3 and similar stores; pass [`FetchPolicy::local`] for an
    /// `InMemory` or `LocalFileSystem` store, where a request costs about as
    /// much as the bytes it reads.
    pub fn with_fetch_policy(mut self, policy: FetchPolicy) -> Self {
        self.fetch_policy = policy;
        self
    }

    /// The namespace id this handle was created or opened with.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The schema this namespace was created with. Read from the manifest at
    /// `create` or `open` time; it never changes afterwards.
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Insert documents, replacing any that already have the same id.
    ///
    /// The whole slice is committed as one WAL entry, so it becomes visible
    /// atomically: readers see all of it or none of it. Every vector is
    /// checked against the schema first, and one bad document rejects the
    /// entire call with nothing written.
    ///
    /// If the same id appears twice in `docs`, the later one wins. Returns the
    /// WAL sequence number the batch was committed at.
    ///
    /// # Errors
    ///
    /// - [`ItzmaError::EmptyBatch`] if `docs` is empty.
    /// - [`ItzmaError::InvalidId`] if a document's `id` is empty or longer than
    ///   [`document::MAX_ID_BYTES`].
    /// - [`ItzmaError::DimensionMismatch`] / [`ItzmaError::InvalidVector`] for
    ///   a bad vector (see [`Schema::validate`]).
    /// - [`ItzmaError::UnknownAttr`], [`ItzmaError::AttrTypeMismatch`] and
    ///   [`ItzmaError::InvalidAttrs`] if a document's `attrs` break the
    ///   namespace's attribute schema. One bad document rejects the
    ///   whole batch. A namespace made before typed attributes checks nothing.
    /// - [`ItzmaError::WriteContention`] if 32 sequence numbers in a row were
    ///   taken by other writers. Nothing was written; retrying is safe.
    /// - [`ItzmaError::Storage`] for any backend failure.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[tokio::main]
    /// # async fn main() -> itzma_core::Result<()> {
    /// use std::sync::Arc;
    ///
    /// use itzma_core::{Document, Metric, Namespace, Schema};
    /// use object_store::memory::InMemory;
    ///
    /// let store = Arc::new(InMemory::new());
    /// let ns = Namespace::create(store, "docs", Schema::new(1, Metric::L2)).await?;
    ///
    /// let doc = |v: f32| Document { id: "a".into(), vector: vec![v], attrs: None, sparse: None };
    /// assert_eq!(ns.upsert(vec![doc(1.0)]).await?, 1);
    /// assert_eq!(ns.upsert(vec![doc(5.0)]).await?, 2); // replaces "a"
    ///
    /// let hits = ns.query(&[5.0], 10).await?;
    /// assert_eq!(hits.len(), 1);
    /// assert_eq!(hits[0].distance, 0.0);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn upsert(&self, docs: Vec<Document>) -> Result<u64> {
        for doc in &docs {
            self.validate_doc(doc)?;
        }
        self.append(docs.into_iter().map(WalOp::Upsert).collect())
            .await
    }

    /// Remove documents by id.
    ///
    /// Like `upsert`, the whole call is one atomic WAL entry. Ids that do not
    /// exist are ignored, so deleting is idempotent. Returns the WAL sequence
    /// number the batch was committed at.
    ///
    /// The documents are hidden from queries immediately, but their bytes are
    /// not erased. They stay in the WAL until [`gc`](Self::gc) fences its
    /// objects, which is off unless asked for, and in any segment that
    /// holds them until that segment is superseded and collected.
    ///
    /// # Errors
    ///
    /// - [`ItzmaError::EmptyBatch`] if `ids` is empty.
    /// - [`ItzmaError::InvalidId`] if an id is empty or longer than
    ///   [`document::MAX_ID_BYTES`].
    /// - [`ItzmaError::WriteContention`] / [`ItzmaError::Storage`] as for
    ///   [`upsert`](Self::upsert).
    ///
    /// # Examples
    ///
    /// ```
    /// # #[tokio::main]
    /// # async fn main() -> itzma_core::Result<()> {
    /// use std::sync::Arc;
    ///
    /// use itzma_core::{Document, Metric, Namespace, Schema};
    /// use object_store::memory::InMemory;
    ///
    /// let store = Arc::new(InMemory::new());
    /// let ns = Namespace::create(store, "docs", Schema::new(1, Metric::L2)).await?;
    /// ns.upsert(vec![Document { id: "a".into(), vector: vec![1.0], attrs: None, sparse: None }]).await?;
    ///
    /// ns.delete(vec!["a".into()]).await?;
    /// assert!(ns.query(&[1.0], 10).await?.is_empty());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn delete(&self, ids: Vec<String>) -> Result<u64> {
        for id in &ids {
            document::validate_id(id)?;
        }
        self.append(ids.into_iter().map(WalOp::Delete).collect())
            .await
    }

    /// Insert documents under an idempotency key, so that retrying the call
    /// does not write the batch twice.
    ///
    /// Behaves like [`upsert`](Self::upsert), and the batch is committed as one
    /// WAL entry. What is different is the key, in `write`: a first attempt is
    /// [`WriteKey::new`], and a call that may repeat an earlier one is
    /// [`WriteKey::retry`], which looks for the key in the WAL objects from a
    /// seq the caller names before it writes anything. A found key returns the
    /// seq of the original and writes nothing. The rules, and what they do not
    /// cover, are on [`WriteKey`].
    ///
    /// The namespace must be at WAL level 3, which every namespace created by
    /// this build is. One created by an older build has to be raised first with
    /// [`upgrade_wal_format`](Self::upgrade_wal_format). A handle reads the
    /// manifest for the level once, and only if it does not already know the
    /// level is high enough. Unkeyed writes never read it.
    ///
    /// # Errors
    ///
    /// As for [`upsert`](Self::upsert), plus:
    ///
    /// - [`ItzmaError::InvalidKey`] unless the key is 1 to 128 bytes.
    /// - [`ItzmaError::WalFormatTooOld`] if the namespace is below level 3.
    /// - [`ItzmaError::KeyConflict`] if the key was used for a batch of a
    ///   different size.
    /// - [`ItzmaError::WalFormatTooNew`] if the manifest read for the level
    ///   shows a format this build cannot read.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[tokio::main]
    /// # async fn main() -> itzma_core::Result<()> {
    /// use std::sync::Arc;
    ///
    /// use itzma_core::{Document, Metric, Namespace, Schema, WriteKey};
    /// use object_store::memory::InMemory;
    ///
    /// let store = Arc::new(InMemory::new());
    /// let ns = Namespace::create(store, "docs", Schema::new(1, Metric::L2)).await?;
    /// let batch = || vec![Document { id: "a".into(), vector: vec![1.0], attrs: None, sparse: None }];
    ///
    /// let seq = ns.upsert_keyed(&WriteKey::new("load-7/batch-1"), batch()).await?;
    ///
    /// // The acknowledgement never arrived, so the caller tries again from
    /// // the last seq it saw committed (here nothing, so 1).
    /// let retry = WriteKey::retry("load-7/batch-1", 1);
    /// assert_eq!(ns.upsert_keyed(&retry, batch()).await?, seq);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn upsert_keyed(&self, write: &WriteKey, docs: Vec<Document>) -> Result<u64> {
        for doc in &docs {
            self.validate_doc(doc)?;
        }
        self.append_keyed(write, docs.into_iter().map(WalOp::Upsert).collect())
            .await
    }

    /// Insert a columnar batch, one row a document: the same WAL
    /// object, byte for byte, that [`upsert`](Self::upsert) writes for the
    /// same documents, without building them. The batch is checked with
    /// [`DocColumns::validate`] first, and one bad row rejects the whole call
    /// with nothing written.
    ///
    /// # Errors
    ///
    /// The errors of [`upsert`](Self::upsert). An input with one defect gets
    /// exactly `upsert`'s error; with several, the first one reported can
    /// differ (see [`DocColumns::validate`]). A batch with a sparse column
    /// needs WAL level 5 ([`ItzmaError::WalFormatTooOld`]).
    ///
    /// # Examples
    ///
    /// ```
    /// # #[tokio::main]
    /// # async fn main() -> itzma_core::Result<()> {
    /// use std::sync::Arc;
    ///
    /// use itzma_core::{DocColumns, Metric, Namespace, Schema};
    /// use object_store::memory::InMemory;
    ///
    /// let ns = Namespace::create(Arc::new(InMemory::new()), "docs", Schema::new(2, Metric::L2)).await?;
    /// // Ids "a" and "b", vectors [1, 0] and [0, 1].
    /// let batch = DocColumns::new(b"ab".to_vec(), vec![0, 1, 2], vec![1.0, 0.0, 0.0, 1.0], 2)?;
    /// assert_eq!(ns.upsert_columns(&batch).await?, 1);
    /// assert_eq!(ns.query(&[1.0, 0.0], 1).await?[0].id, "a");
    /// # Ok(())
    /// # }
    /// ```
    pub async fn upsert_columns(&self, cols: &DocColumns) -> Result<u64> {
        cols.validate(&self.schema)?;
        let bytes = wal_codec::encode_columns(cols, wal::MAX_BODY_BYTES)?;
        self.append_bytes(bytes, cols.has_sparse()).await
    }

    /// Load a stream of columnar batches in bounded memory.
    ///
    /// The input is cut into chunks of about
    /// [`StreamOptions::target_bytes`] of encoded body, wherever its own
    /// batches happen to end, so the same rows always make the same chunks.
    /// Each chunk is checked as [`upsert_columns`](Self::upsert_columns)
    /// checks a batch and encoded on the compute pool while the previous one
    /// is being written; exactly one PUT is in flight, so the chunks land in
    /// order and a crash can never leave a hole between them. At most
    /// [`StreamOptions::depth`] encoded chunks wait for the writer, so memory
    /// is about `(depth + 2) * target_bytes` however long the input is.
    ///
    /// Each chunk is its own WAL object, made visible on its own: a reader
    /// can see the first chunks of a load before the rest.
    ///
    /// The batches of one stream must have the same columns: the same `dim`,
    /// the same attribute names, each attribute column of the same kind (an
    /// all-null [`AttrColumn::Null`](crate::AttrColumn::Null) fits any kind; a number column cannot
    /// switch between integers and floats from one batch to the next, which
    /// `upsert` of documents would allow), and all a sparse column or none.
    /// Otherwise the stream stops with [`ItzmaError::InvalidColumns`].
    ///
    /// # Errors
    ///
    /// A [`StreamError`]: the error, and in
    /// [`landed`](StreamError::landed) what was written before it, whole and
    /// in order. Chunks the encoder had already queued when a later chunk
    /// failed its check are written first. The errors are those of
    /// `upsert_columns`, plus an error from the input itself,
    /// [`ItzmaError::EmptyBatch`] for an input with no rows, and
    /// [`ItzmaError::InvalidConfig`] for bad options (before the input is
    /// polled). To resume an unkeyed load, feed the input again from row
    /// [`landed.rows`](StreamReport::rows).
    ///
    /// # Examples
    ///
    /// ```
    /// # #[tokio::main]
    /// # async fn main() -> itzma_core::Result<()> {
    /// use std::sync::Arc;
    ///
    /// use itzma_core::{DocColumns, Metric, Namespace, Schema, StreamOptions};
    /// use object_store::memory::InMemory;
    ///
    /// let ns = Namespace::create(Arc::new(InMemory::new()), "docs", Schema::new(1, Metric::L2)).await?;
    /// let batches = vec![
    ///     DocColumns::new(b"a".to_vec(), vec![0, 1], vec![1.0], 1),
    ///     DocColumns::new(b"b".to_vec(), vec![0, 1], vec![2.0], 1),
    /// ];
    /// let report = ns
    ///     .upsert_stream(futures::stream::iter(batches), &StreamOptions::default())
    ///     .await
    ///     .map_err(|e| e.error)?;
    /// assert_eq!((report.rows, report.objects), (2, 1));
    /// # Ok(())
    /// # }
    /// ```
    pub async fn upsert_stream<S>(
        &self,
        input: S,
        options: &StreamOptions,
    ) -> std::result::Result<StreamReport, StreamError>
    where
        S: futures::Stream<Item = Result<DocColumns>> + Send,
    {
        stream::run(self, input, options, None).await
    }

    /// [`upsert_stream`](Self::upsert_stream) under an idempotency key: chunk
    /// `n` is written under `<key>#<n>`, so the key can be at most 107 bytes
    /// (room for `#` and any chunk number). Those chunk keys share the key
    /// space of [`upsert_keyed`](Self::upsert_keyed): do not give a plain
    /// keyed write a key of the form `<stream key>#<number>`, or the two can
    /// be taken for each other.
    /// A retry with [`WriteKey::retry`] and the same input and
    /// `target_bytes` cuts the same chunks, finds the ones that landed by
    /// their keys and writes only the rest, however the input is
    /// batched the second time. Its `search_from` must be at or before the
    /// first chunk the first attempt may have written: the failed attempt's
    /// [`landed.first_seq`](StreamReport::first_seq), or, when that is `None`
    /// (nothing was acknowledged, though the first put may have landed), the
    /// namespace's next seq before the first attempt. The search reads the
    /// headers from there once; each later chunk is looked for only past the
    /// one before it.
    ///
    /// # Errors
    ///
    /// Those of `upsert_stream` and [`upsert_keyed`](Self::upsert_keyed);
    /// [`ItzmaError::InvalidKey`] for a key over 107 bytes, before the input
    /// is polled.
    pub async fn upsert_stream_keyed<S>(
        &self,
        write: &WriteKey,
        input: S,
        options: &StreamOptions,
    ) -> std::result::Result<StreamReport, StreamError>
    where
        S: futures::Stream<Item = Result<DocColumns>> + Send,
    {
        stream::run(self, input, options, Some(write)).await
    }

    /// [`upsert_columns`](Self::upsert_columns) under an idempotency key, with
    /// the rules of [`upsert_keyed`](Self::upsert_keyed).
    ///
    /// # Errors
    ///
    /// Those of `upsert_columns` and `upsert_keyed`.
    pub async fn upsert_columns_keyed(&self, write: &WriteKey, cols: &DocColumns) -> Result<u64> {
        write.validate()?;
        cols.validate(&self.schema)?;
        let object = wal_codec::encode_columns_keyed(cols, wal::MAX_BODY_BYTES, &write.key)?;
        self.append_bytes_keyed(write, object, cols.has_sparse())
            .await
    }

    /// Remove documents by id under an idempotency key. The keyed form of
    /// [`delete`](Self::delete), with the rules and errors of
    /// [`upsert_keyed`](Self::upsert_keyed).
    ///
    /// A delete is where a replay does the most harm, since it can undo a
    /// newer upsert of the same id. With a key, the retry finds the delete
    /// already in the log and leaves it alone.
    pub async fn delete_keyed(&self, write: &WriteKey, ids: Vec<String>) -> Result<u64> {
        for id in &ids {
            document::validate_id(id)?;
        }
        self.append_keyed(write, ids.into_iter().map(WalOp::Delete).collect())
            .await
    }

    /// Raise the namespace's WAL format to at least `level`, so that writes
    /// which need it are allowed.
    ///
    /// One compare-and-swap on the manifest. It only goes up: a level at or
    /// below the current one changes nothing and writes nothing, and two
    /// callers racing both succeed. Level 3 is what [`upsert_keyed`](Self::upsert_keyed)
    /// needs, level 4 is what [`gc`](Self::gc) needs to fence WAL objects, and
    /// level 5 is what a sparse-carrying [`upsert`](Self::upsert) or
    /// [`upsert_keyed`](Self::upsert_keyed) needs (WAL object
    /// versions 4 and 5). A namespace created by this build starts at 5.
    ///
    /// **Raise it only once every build that reads or compacts this namespace
    /// reads the new format** (version 3 for level 3, fences for level 4, and
    /// versions 4/5's sparse column for level 5). From the next manifest read, a build that does not is
    /// refused with [`ItzmaError::WalFormatTooNew`] (the gate), which is what
    /// keeps it away, but it cannot be undone: there is no way to lower the
    /// level. Writers that already hold a handle keep writing what they wrote,
    /// which stays readable.
    ///
    /// Needs conditional update on the store, which `LocalFileSystem` does not
    /// implement.
    ///
    /// # Errors
    ///
    /// - [`ItzmaError::WalFormatTooNew`] if `level` is above what this build
    ///   reads ([`wal::FORMAT_VERSION`]).
    /// - [`ItzmaError::ManifestContention`] after 32 lost rounds.
    /// - Anything [`manifest::load`] can return.
    pub async fn upgrade_wal_format(&self, level: u32) -> Result<()> {
        if level > wal::FORMAT_VERSION {
            return Err(ItzmaError::WalFormatTooNew {
                found: level,
                supported: wal::FORMAT_VERSION,
            });
        }
        manifest::update_if(&*self.store, &self.id, |m| {
            if m.wal_format >= level {
                return false;
            }
            m.wal_format = level;
            true
        })
        .await?;
        // Raised by this call or already there: either way the level is at
        // least this.
        self.wal_level.fetch_max(level, Ordering::Relaxed);
        Ok(())
    }

    /// Raise the namespace's segment format to at least `level`, so that
    /// compaction writes segments in that version.
    ///
    /// One compare-and-swap on the manifest. It only goes up: a level at or
    /// below the current one changes nothing and writes nothing, and two
    /// callers racing both succeed. Level 2 is the block-addressable layout.
    /// Level 3 stores a typed namespace's attributes as columns with
    /// a zone map per block, and a free-form namespace's as level 2 does. A
    /// namespace created by this build starts at
    /// [`segment::DEFAULT_FORMAT`], and one made before the levels existed is
    /// at 1 and stays there until this is called.
    ///
    /// Nothing is rewritten. The segments already there keep their version, and
    /// each later fold or merge writes the new one, so a namespace can hold
    /// both. **Raise it only once every build that reads or compacts this
    /// namespace reads the new version.** From the next manifest read, a build
    /// that does not is refused with [`ItzmaError::SegmentFormatTooNew`] (the
    /// gate), but the raise cannot be undone: there is no way to lower the
    /// level.
    ///
    /// A compaction that stopped after writing a segment at the old level and
    /// is rerun after the raise builds different bytes for the same key, and
    /// [`compact`](Self::compact) reports [`ItzmaError::CorruptSegment`] rather
    /// than publish over it. Nothing is published. Delete that
    /// orphan segment object (it is listed as an orphan by
    /// [`inspect::report`](crate::inspect::report)) and rerun.
    ///
    /// Needs conditional update on the store, which `LocalFileSystem` does not
    /// implement.
    ///
    /// # Errors
    ///
    /// - [`ItzmaError::SegmentFormatTooNew`] if `level` is above what this
    ///   build reads ([`segment::FORMAT_VERSION`]).
    /// - [`ItzmaError::ManifestContention`] after 32 lost rounds.
    /// - Anything [`manifest::load`] can return.
    pub async fn upgrade_segment_format(&self, level: u32) -> Result<()> {
        if level > segment::FORMAT_VERSION {
            return Err(ItzmaError::SegmentFormatTooNew {
                found: level,
                supported: segment::FORMAT_VERSION,
            });
        }
        manifest::update_if(&*self.store, &self.id, |m| {
            if m.segment_format >= level {
                return false;
            }
            m.segment_format = level;
            true
        })
        .await?;
        Ok(())
    }

    /// Fold the un-compacted WAL into a segment, or merge segments when there
    /// are too many, and publish the result.
    ///
    /// What happens depends on the state:
    ///
    /// - WAL tail empty and at most `max_segments` segments: nothing to do.
    /// - Adding one more segment would exceed `max_segments`: every segment
    ///   and the tail are merged into one base segment.
    /// - Otherwise: the tail becomes a new segment beside the existing ones.
    ///
    /// The segment is written with put-if-absent, then listed in the manifest
    /// by one compare-and-swap. Writers are never blocked and never see the
    /// manifest change. Queries running during a compaction are unaffected.
    /// Any process may call this, as often as it likes, concurrently: racing
    /// compactors build the same object, one wins the swap, and the rest get
    /// [`Compaction::Superseded`]. A run that crashes before publishing leaves
    /// an orphan segment that the next run verifies against the bytes it
    /// builds and reuses. An orphan that differs makes `compact` fail with
    /// [`ItzmaError::CorruptSegment`] and publish nothing.
    ///
    /// The swap also drops the manifest's index entries (see
    /// [`index`](Self::index)) of any segment the run retires, so a merge clears
    /// them, a fold keeps them, and the segment written starts without one.
    ///
    /// Compaction never deletes anything: not the WAL objects that were
    /// folded, not superseded segments, not orphans. Superseded segments and
    /// dead orphans are removed by a separate call to [`gc`](Self::gc), and
    /// folded WAL objects are kept until that call fences them. If a
    /// segment the run needs disappears, or a WAL object it reads has become a
    /// fence, because a compaction and a garbage collection ran after it
    /// loaded the manifest, the run reloads the manifest and returns
    /// [`Compaction::Superseded`].
    ///
    /// Publishing needs conditional update on the store, which
    /// `LocalFileSystem` does not implement. There a run fails at the publish
    /// step with a storage error after writing its segment.
    ///
    /// # Errors
    ///
    /// - [`ItzmaError::ManifestContention`] after 32 lost publish rounds.
    /// - [`ItzmaError::CorruptManifest`] / [`ItzmaError::CorruptSegment`] if
    ///   the manifest or a segment it lists is damaged.
    /// - [`ItzmaError::CorruptSegment`] also when an object already sits at
    ///   the new segment's key with different bytes.
    /// - [`ItzmaError::WalGap`] if the log has a hole.
    /// - [`ItzmaError::InvalidConfig`] when a merge is due and
    ///   `options.max_segment_bytes` is above the handle's read budget, before
    ///   any segment is read or anything written. A fold does not need the
    ///   check, so a handle with a small query budget still folds.
    /// - [`ItzmaError::ReadBudgetExceeded`] if the tail, or one merge group,
    ///   is more than the read budget: the tail and each group get a budget
    ///   of their own, so a namespace bigger than the budget still
    ///   merges. Nothing is published; groups written before the refused one
    ///   stay as orphans that a rerun reuses or `gc` collects.
    /// - [`ItzmaError::DimensionMismatch`] if a WAL document's vector does not
    ///   match the schema (only possible for a hand-edited WAL).
    /// - [`ItzmaError::UnsupportedFormat`], [`ItzmaError::Serde`] or
    ///   [`ItzmaError::Storage`] for unreadable objects or backend failures.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[tokio::main]
    /// # async fn main() -> itzma_core::Result<()> {
    /// use std::sync::Arc;
    ///
    /// use itzma_core::{Compaction, CompactionOptions, Document, Metric, Namespace, Schema};
    /// use object_store::memory::InMemory;
    ///
    /// let store = Arc::new(InMemory::new());
    /// let ns = Namespace::create(store, "docs", Schema::new(1, Metric::L2)).await?;
    /// ns.upsert(vec![Document { id: "a".into(), vector: vec![1.0], attrs: None, sparse: None }]).await?;
    ///
    /// let options = CompactionOptions::default();
    /// let outcome = ns.compact(&options).await?;
    /// assert!(matches!(outcome, Compaction::Published { folded_through: 1, .. }));
    /// assert_eq!(ns.query(&[1.0], 1).await?[0].id, "a");
    ///
    /// // Nothing new has been written, so there is nothing to fold.
    /// assert_eq!(ns.compact(&options).await?, Compaction::Nothing);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn compact(&self, options: &CompactionOptions) -> Result<Compaction> {
        let manifest = self.load_manifest().await?;
        let outcome = compaction::run(
            &*self.store,
            &self.id,
            self.schema.clone(),
            manifest.clone(),
            options,
            self.read_budget,
            &self.fetch_policy,
        )
        .await;
        match outcome {
            // A segment this run needed is gone, or a WAL object it read is now
            // a fence. If the manifest has moved, a merge or a fold and a
            // garbage collection ran after this run planned, which is the same
            // as losing any other race.
            Err(err) if is_stale_read(&err) => {
                self.reload_if_moved(&manifest, err).await?;
                Ok(Compaction::Superseded)
            }
            other => other,
        }
    }

    /// Build and publish an index for each live segment that has none.
    /// An index lets a query search a segment without reading it whole
    /// (see [`query`](Self::query)); a segment with no index is read whole, as it
    /// always was, so a namespace never has to be indexed.
    ///
    /// Segments are taken one at a time. For each one that is version 2 and has
    /// at least [`IndexOptions::min_rows`] documents, the data object is read
    /// (under a read budget of its own, so a namespace is not limited by the sum
    /// of its segments, but one segment must fit in it), the index object
    /// `segments/<id>/index-<generation>.bin` is written
    /// with put-if-absent, and one
    /// manifest compare-and-swap lists it. Until that swap the object is an
    /// orphan that changes nothing, and a run that stops there leaves one that the
    /// next run checks and reuses. Any valid index of the segment is accepted, so
    /// two indexers racing on the same segment both succeed and one is counted as
    /// [`already_indexed`](IndexReport::already_indexed).
    ///
    /// A fold or merge (`compact`) never indexes: it leaves the segment it writes
    /// without one, and a merge also drops the entries of the segments it retires.
    /// Call `index` again afterwards. [`IndexOptions::rebuild`] writes generation
    /// `n + 1` for a segment that already has generation `n`, and the old object
    /// is left for [`gc`](Self::gc), which removes index objects that no manifest
    /// names once they are `min_age` old.
    ///
    /// Needs conditional update on the store, like `compact`.
    ///
    /// # Errors
    ///
    /// - [`ItzmaError::SegmentFormatTooOld`] if the namespace is below segment
    ///   format 2. Nothing is touched.
    /// - [`ItzmaError::ReadBudgetExceeded`] if one segment is bigger than the
    ///   read budget.
    /// - [`ItzmaError::CorruptSegment`] if a segment is damaged. A segment a
    ///   compaction and a garbage collection removed after this call loaded the
    ///   manifest is counted as [`superseded`](IndexReport::superseded) instead.
    /// - [`ItzmaError::CorruptIndex`] if an object already sits at the new
    ///   index's key and does not fit the segment. Delete it and rerun.
    /// - [`ItzmaError::ManifestContention`] after 32 lost publish rounds.
    /// - Anything [`manifest::load`] can return.
    ///
    /// A failure part way leaves the segments before it indexed. Call again to
    /// finish.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[tokio::main]
    /// # async fn main() -> itzma_core::Result<()> {
    /// use std::sync::Arc;
    ///
    /// use itzma_core::{CompactionOptions, Document, IndexOptions, Metric, Namespace, Schema};
    /// use object_store::memory::InMemory;
    ///
    /// let store = Arc::new(InMemory::new());
    /// let ns = Namespace::create(store, "docs", Schema::new(2, Metric::L2)).await?;
    /// ns.upsert(vec![Document { id: "a".into(), vector: vec![1.0, 0.0], attrs: None, sparse: None }]).await?;
    /// ns.compact(&CompactionOptions::default()).await?;
    ///
    /// // One document is under the default minimum, so nothing is indexed.
    /// let report = ns.index(&IndexOptions::default()).await?;
    /// assert_eq!((report.indexed, report.skipped_small), (0, 1));
    ///
    /// // A minimum of zero indexes it.
    /// let report = ns.index(&IndexOptions::default().min_rows(0)).await?;
    /// assert_eq!(report.indexed, 1);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn index(&self, options: &IndexOptions) -> Result<IndexReport> {
        options.validate()?;
        let manifest = self.load_manifest().await?;
        if manifest.segment_format < MIN_SEGMENT_FORMAT {
            return Err(ItzmaError::SegmentFormatTooOld {
                have: manifest.segment_format,
                need: MIN_SEGMENT_FORMAT,
            });
        }
        let mut report = IndexReport::default();
        for id in &manifest.live_segments {
            let outcome = indexer::index_segment(
                &*self.store,
                &self.id,
                self.schema.dim,
                self.schema.metric,
                id,
                manifest.indexes.get(id),
                options,
                self.read_budget,
                &self.fetch_policy,
            )
            .await;
            let outcome = match outcome {
                // The segment is gone because a merge and a collection ran
                // after the manifest was loaded. If the manifest has not
                // moved, it is really missing and that is the error.
                Err(err) if is_stale_read(&err) => {
                    self.reload_if_moved(&manifest, err).await?;
                    Outcome::Superseded
                }
                other => other?,
            };
            report.record(outcome);
        }
        Ok(report)
    }

    /// Delete the segments that a merge or a later fold has superseded, and
    /// the orphans that can no longer be reused (phase A). With
    /// [`GcOptions::fence_wal`] it also replaces folded WAL objects with fences
    /// (phase B).
    ///
    /// A segment is dead when it is not in the manifest, its last seq is at or
    /// below `wal_folded_through`, and it is not `1-<wal_folded_through>` (the
    /// id a merge over an empty tail would produce and reuse). Only dead
    /// segments are deleted, and only when both they and the live segment that
    /// replaced them are at least [`GcOptions::min_age`] old. Live segments,
    /// orphans past the fold point and the manifest are never touched. With
    /// [`GcOptions::dry_run`] nothing is deleted or fenced.
    ///
    /// **Fencing** is off unless asked for. A fence is an 8-byte object that
    /// takes the place of a WAL object at the same key, so put-if-absent still
    /// refuses a writer whose hint is stale, and the bytes of the old batch are
    /// gone. An object is fenced when its seq is at or below
    /// `wal_folded_through`, and both it and the live segment that holds its seq
    /// are at least `min_age` old. **After that the segments are the only copy of
    /// what those objects held.** A damaged segment can no longer be rebuilt from
    /// the WAL, and neither can a manifest reset to fold nothing. The namespace
    /// must be at WAL format level 4 (see
    /// [`upgrade_wal_format`](Self::upgrade_wal_format)), and every build that
    /// reads it must know fences, which the gate enforces. A fence has no
    /// idempotency key, so a keyed retry can find its original only until the
    /// object is fenced, at least `min_age` after the write.
    ///
    /// GC is a separate call because deleting has its own cost and failures,
    /// and nothing hosts a background job yet. It reads the manifest and never
    /// writes it, so it needs no conditional update and also works on
    /// `LocalFileSystem`. It may run at any time, from any number of processes
    /// at once, beside writers, compactors and queries. A query that loaded
    /// the manifest before a merge and reads after GC reloads and finishes on
    /// the new one, and so does a compaction (see [`query`](Self::query) and
    /// [`compact`](Self::compact)). That covers a fence in the tail as well as a
    /// missing segment. Roll out readers with that behavior before running GC.
    /// A fencing run lists the WAL only past a small hint object,
    /// `gc/fenced-through`, that it keeps up to date, so the listing
    /// does not grow with the fenced part of the log.
    ///
    /// Inside a live segment's directory it also deletes index objects
    /// that the manifest does not name: an older generation that a
    /// rebuild replaced, or an orphan of a build that never published. Such an
    /// object goes once it and the index that replaced it are `min_age` old (an
    /// orphan with no replacement needs only its own age). The index the
    /// manifest names is never deleted.
    ///
    /// GC deletes segments, so a document's old bytes stay in the store until
    /// the segments that held them are collected, and in the WAL until its
    /// objects are fenced.
    ///
    /// # Errors
    ///
    /// - [`ItzmaError::CorruptManifest`] if the manifest's segment list breaks
    ///   its invariant.
    /// - [`ItzmaError::WalFormatTooOld`] if `fence_wal` is set and the namespace
    ///   is below level 4. Nothing is changed.
    /// - [`ItzmaError::UnsupportedFormat`], [`ItzmaError::Serde`] or
    ///   [`ItzmaError::Storage`] for an unreadable manifest or a backend
    ///   failure. A failure part way leaves some dead segments deleted and
    ///   others not, and some objects fenced and others whole; rerunning
    ///   finishes the job.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[tokio::main]
    /// # async fn main() -> itzma_core::Result<()> {
    /// use std::sync::Arc;
    /// use std::time::Duration;
    ///
    /// use itzma_core::{CompactionOptions, Document, GcOptions, Metric, Namespace, Schema};
    /// use object_store::memory::InMemory;
    ///
    /// let store = Arc::new(InMemory::new());
    /// let ns = Namespace::create(store, "docs", Schema::new(1, Metric::L2)).await?;
    /// let doc = |id: &str| Document { id: id.into(), vector: vec![1.0], attrs: None, sparse: None };
    ///
    /// // Two folds, then a merge that replaces both segments.
    /// let tight = CompactionOptions { max_segments: 1, max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES };
    /// ns.upsert(vec![doc("a")]).await?;
    /// ns.compact(&tight).await?;
    /// ns.upsert(vec![doc("b")]).await?;
    /// ns.compact(&tight).await?;
    ///
    /// // The default keeps anything younger than a day.
    /// assert_eq!(ns.gc(&GcOptions::default()).await?.segments_deleted, 0);
    ///
    /// let now = GcOptions { min_age: Duration::ZERO, ..GcOptions::default() };
    /// assert_eq!(ns.gc(&now).await?.segments_deleted, 1);
    /// assert_eq!(ns.query(&[1.0], 10).await?.len(), 2);
    ///
    /// // The WAL is only touched when asked. Both objects are folded now.
    /// assert_eq!(ns.gc(&now).await?.wal_objects_fenced, 0);
    /// let fencing = GcOptions { fence_wal: true, ..now };
    /// assert_eq!(ns.gc(&fencing).await?.wal_objects_fenced, 2);
    /// assert_eq!(ns.query(&[1.0], 10).await?.len(), 2);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn gc(&self, options: &GcOptions) -> Result<GcReport> {
        let manifest = self.load_manifest().await?;
        gc::run(&*self.store, &self.id, &manifest, options).await
    }

    /// Compact if the WAL tail is long enough, then collect, and report what
    /// happened. The caller decides when to call it, from a cron job
    /// or a loop. The library starts no task of its own.
    ///
    /// It lists the WAL tail past `wal_folded_through` (sizes and write times,
    /// no GETs) and runs [`compact`](Self::compact) with
    /// [`MaintenanceOptions::compaction`] if the tail is not empty and any
    /// [`CompactTrigger`](crate::CompactTrigger) threshold is crossed, or if
    /// there are more live segments than `max_segments`. Then, if
    /// [`MaintenanceOptions::gc`] is set, it runs [`gc`](Self::gc), which
    /// happens whether or not a compaction was due. The compaction goes first,
    /// so the segments it replaced are candidates at once.
    ///
    /// It is safe from any number of processes at once, because `compact` and
    /// `gc` are. A compaction that another process won reports
    /// [`Compaction::Superseded`]. A call that stopped part way is finished by
    /// calling it again. It reads the manifest twice, since `compact` loads its
    /// own.
    ///
    /// # Errors
    ///
    /// Those of [`compact`](Self::compact) and [`gc`](Self::gc), including
    /// [`ItzmaError::WalFormatTooNew`] and [`ItzmaError::WalFormatTooOld`]
    /// (`fence_wal` below level 4). A failure in the compaction stops the call
    /// before the collection.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[tokio::main]
    /// # async fn main() -> itzma_core::Result<()> {
    /// use std::sync::Arc;
    ///
    /// use itzma_core::{CompactTrigger, Compaction, Document, MaintenanceOptions, Metric, Namespace, Schema};
    /// use object_store::memory::InMemory;
    ///
    /// let store = Arc::new(InMemory::new());
    /// let ns = Namespace::create(store, "docs", Schema::new(1, Metric::L2)).await?;
    /// let doc = |id: &str| Document { id: id.into(), vector: vec![1.0], attrs: None, sparse: None };
    ///
    /// // Fold once the tail has two objects.
    /// let options = MaintenanceOptions {
    ///     compact_when: CompactTrigger { tail_objects: 2, ..CompactTrigger::default() },
    ///     ..MaintenanceOptions::default()
    /// };
    ///
    /// ns.upsert(vec![doc("a")]).await?;
    /// assert_eq!(ns.maintain(&options).await?.compaction, None);
    ///
    /// ns.upsert(vec![doc("b")]).await?;
    /// let report = ns.maintain(&options).await?;
    /// assert_eq!(report.tail_objects, 2);
    /// assert!(matches!(report.compaction, Some(Compaction::Published { folded_through: 2, .. })));
    /// # Ok(())
    /// # }
    /// ```
    pub async fn maintain(&self, options: &MaintenanceOptions) -> Result<MaintenanceReport> {
        let manifest = self.load_manifest().await?;
        let objects =
            gc::list_wal_after(&*self.store, &self.id, manifest.wal_folded_through).await?;
        let tail = maintain::Tail::of(&objects);
        let compaction = if maintain::due(
            &options.compact_when,
            tail,
            manifest.live_segments.len(),
            options.compaction.max_segments,
            gc::now_millis(),
        ) {
            Some(self.compact(&options.compaction).await?)
        } else {
            None
        };
        let gc = match &options.gc {
            Some(gc_options) => Some(self.gc(gc_options).await?),
            None => None,
        };
        Ok(MaintenanceReport {
            tail_objects: tail.objects,
            tail_bytes: tail.bytes,
            compaction,
            gc,
        })
    }

    /// Commit ops through [`wal::append`], then advance the hint past the seq
    /// that was won. `fetch_max` keeps the hint from moving backwards when
    /// several tasks share a handle. A batch that carries a sparse vector
    /// needs the namespace at [`wal::SPARSE_FORMAT_VERSION`] first,
    /// checked before anything is written; a plain batch never reads the
    /// level.
    async fn append(&self, ops: Vec<WalOp>) -> Result<u64> {
        if has_sparse(&ops) {
            self.ensure_wal_level(wal::SPARSE_FORMAT_VERSION).await?;
        }
        let hint = self.next_seq.load(Ordering::Relaxed);
        let seq = wal::append(&*self.store, &self.id, hint, ops).await?;
        self.next_seq.fetch_max(seq + 1, Ordering::Relaxed);
        Ok(seq)
    }

    /// Commit an object `wal_codec` already encoded, with [`append`]'s level
    /// check (`sparse`: the object has a sparse column) and hint bookkeeping.
    pub(crate) async fn append_bytes(&self, bytes: bytes::Bytes, sparse: bool) -> Result<u64> {
        if sparse {
            self.ensure_wal_level(wal::SPARSE_FORMAT_VERSION).await?;
        }
        let hint = self.next_seq.load(Ordering::Relaxed);
        let seq = wal::put_encoded(&*self.store, &self.id, hint, bytes).await?;
        self.next_seq.fetch_max(seq + 1, Ordering::Relaxed);
        Ok(seq)
    }

    /// [`append_bytes`](Self::append_bytes) for a keyed object, with
    /// [`append_keyed`](Self::append_keyed)'s checks.
    pub(crate) async fn append_bytes_keyed(
        &self,
        write: &WriteKey,
        object: wal_codec::Keyed,
        sparse: bool,
    ) -> Result<u64> {
        write.validate()?;
        self.ensure_wal_level(wal::KEYED_FORMAT_VERSION).await?;
        if sparse {
            self.ensure_wal_level(wal::SPARSE_FORMAT_VERSION).await?;
        }
        let hint = self.next_seq.load(Ordering::Relaxed);
        let seq = wal::put_encoded_keyed(&*self.store, &self.id, hint, write, object).await?;
        self.next_seq.fetch_max(seq + 1, Ordering::Relaxed);
        Ok(seq)
    }

    /// Commit ops under a key through [`wal::append_keyed`], after checking the
    /// key and the namespace's level (and, for a sparse-carrying batch, the
    /// sparse level too).
    async fn append_keyed(&self, write: &WriteKey, ops: Vec<WalOp>) -> Result<u64> {
        write.validate()?;
        if ops.is_empty() {
            return Err(ItzmaError::EmptyBatch);
        }
        self.ensure_wal_level(wal::KEYED_FORMAT_VERSION).await?;
        if has_sparse(&ops) {
            self.ensure_wal_level(wal::SPARSE_FORMAT_VERSION).await?;
        }
        let hint = self.next_seq.load(Ordering::Relaxed);
        let seq = wal::append_keyed(&*self.store, &self.id, hint, write, ops).await?;
        self.next_seq.fetch_max(seq + 1, Ordering::Relaxed);
        Ok(seq)
    }

    /// Check that the namespace's WAL format is at least `need`. A handle
    /// that already knows it is does not read anything. Otherwise the manifest
    /// is read once, because another process may have raised the level since
    /// this handle opened.
    async fn ensure_wal_level(&self, need: u32) -> Result<()> {
        if self.wal_level.load(Ordering::Relaxed) >= need {
            return Ok(());
        }
        let (manifest, _version) = manifest::load(&*self.store, &self.id).await?;
        self.wal_level
            .fetch_max(manifest.wal_format, Ordering::Relaxed);
        if manifest.wal_format >= need {
            Ok(())
        } else {
            Err(ItzmaError::WalFormatTooOld {
                have: manifest.wal_format,
                need,
            })
        }
    }

    /// Find the `top_k` documents closest to `vector`, closest first.
    ///
    /// Steps:
    ///
    /// 1. Validate `vector` against the schema.
    /// 2. Load the manifest (one GET, no cache) and check that its segment
    ///    list is contiguous and ends at `wal_folded_through`.
    /// 3. In parallel, fetch and decode the live segments (16 at a time) and
    ///    read the WAL entries after `wal_folded_through`, checking they are
    ///    contiguous. Each entry is folded into a layer of its own as it
    ///    arrives. Every object is charged to the read budget before its body
    ///    is read.
    /// 4. Scan each source for its closest documents and drop any that a
    ///    newer source holds as a document or a tombstone, so the last op on
    ///    an id wins, including deletes.
    /// 5. Keep the best `top_k` across sources (as [`crate::search::flat_top_k`]
    ///    would over the replayed documents).
    ///
    /// The metric comes from the schema. Distances follow the "lower is closer"
    /// convention documented on [`crate::search::distance`], and ties break by id.
    /// If fewer than `top_k` documents exist, all of them are returned; if
    /// `top_k` is 0 the result is empty.
    ///
    /// A query started after a compaction published sees the same documents as
    /// one started before it. A query racing a compaction works from the
    /// manifest it loaded. If garbage collection has since deleted a segment
    /// that manifest lists, the query loads the manifest again and reads
    /// against the new one, three attempts in all. A segment that is missing
    /// from an unchanged manifest is damage, and the query fails.
    ///
    /// # Cost
    ///
    /// Every call fetches all live segments and the un-compacted WAL tail and
    /// scans every document, so cost is O(live documents) plus one request
    /// per segment and per tail entry. Run [`compact`](Self::compact) to keep
    /// the tail short. Caching and indexes are what remove the rest.
    ///
    /// A query holds every live document in memory, so it reads at most the
    /// handle's read budget (see [`with_read_budget`](Self::with_read_budget)).
    ///
    /// # Errors
    ///
    /// - [`ItzmaError::DimensionMismatch`] / [`ItzmaError::InvalidVector`] for
    ///   a bad query vector.
    /// - [`ItzmaError::CorruptManifest`] if the segment list breaks its
    ///   invariant.
    /// - [`ItzmaError::CorruptSegment`] if a listed segment is missing from
    ///   the current manifest, damaged, or has the wrong dimension.
    /// - [`ItzmaError::WalGap`] if the log has a hole.
    /// - [`ItzmaError::ReadBudgetExceeded`] if the live segments and the tail
    ///   add up to more than the read budget.
    /// - [`ItzmaError::UnsupportedFormat`], [`ItzmaError::Serde`] or
    ///   [`ItzmaError::Storage`] for unreadable objects or backend failures.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[tokio::main]
    /// # async fn main() -> itzma_core::Result<()> {
    /// use std::sync::Arc;
    ///
    /// use itzma_core::{AttrSchema, AttrType, Document, Metric, Namespace, Schema};
    /// use object_store::memory::InMemory;
    ///
    /// let store = Arc::new(InMemory::new());
    /// let schema = Schema::new(2, Metric::Cosine)
    ///     .with_attrs(AttrSchema::new().field("label", AttrType::Text));
    /// let ns = Namespace::create(store, "docs", schema).await?;
    /// let doc = |id: &str, v: [f32; 2]| Document {
    ///     id: id.into(),
    ///     vector: v.to_vec(),
    ///     attrs: Some(serde_json::json!({ "label": id })),
    ///     sparse: None,
    /// };
    /// ns.upsert(vec![doc("east", [1.0, 0.0]), doc("north", [0.0, 1.0])]).await?;
    ///
    /// let hits = ns.query(&[2.0, 0.1], 2).await?;
    /// assert_eq!(hits[0].id, "east");
    /// assert_eq!(hits[0].attrs, Some(serde_json::json!({ "label": "east" })));
    /// assert!(hits[0].distance < hits[1].distance);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn query(&self, vector: &[f32], top_k: usize) -> Result<Vec<Hit>> {
        self.query_with(vector, top_k, &QueryOptions::default())
            .await
    }

    /// [`query`](Self::query) with [`QueryOptions`]: how many candidates an
    /// index proposes per hit wanted ([`QueryOptions::rerank`]), how many of its
    /// partitions to search ([`QueryOptions::nprobe`]), and whether to ignore
    /// indexes and scan flat ([`QueryOptions::exact`]).
    ///
    /// [`QueryOptions::filter`] keeps only documents whose attributes match a
    /// [`Predicate`]. It is checked against the attribute schema
    /// before any read: an undeclared field is [`ItzmaError::UnknownAttr`], an
    /// operator that does not fit its field is [`ItzmaError::AttrTypeMismatch`],
    /// and a namespace with no attribute schema refuses it with
    /// [`ItzmaError::InvalidQuery`]. The answer is the closest `top_k` live
    /// documents that match, and has fewer only when fewer match. The newest
    /// version of an id is the one tested. A selective filter widens the search
    /// until it has enough matches or has read every row, bounded by the read
    /// budget.
    ///
    /// With no live segment holding an index this build can search (an empty
    /// namespace, one whose segments are not indexed, or `exact: true`), this is
    /// exactly the flat scan `query` has always done, and the options change
    /// nothing else.
    ///
    /// Otherwise a live segment with an index is read through it: its algorithm
    /// proposes `rerank * top_k` candidate rows by approximate distance (every
    /// row if `rerank` is `None`, which makes the answer equal the flat scan's),
    /// each is checked against every newer source (a segment or the WAL tail)
    /// and dropped if one holds it as a document or a tombstone, and what
    /// survives is rescored against its exact vector. A segment left short of
    /// `top_k` live candidates by shadowing is asked again for more before the
    /// next source is read. Segments without an index, and the WAL tail, are
    /// read whole and scanned flat, like today. The merge across sources keeps
    /// the closest `top_k` by exact distance, and only then are their attributes
    /// fetched.
    ///
    /// # Errors
    ///
    /// - [`ItzmaError::InvalidQuery`] for a `rerank` or `nprobe` of zero.
    /// - Everything [`query`](Self::query) can return.
    /// - [`ItzmaError::CorruptIndex`] for a damaged index, or one that proposed
    ///   a row past its segment. Never falls back to a flat scan; retry with
    ///   `exact: true`, or `index` with `rebuild` to replace it.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[tokio::main]
    /// # async fn main() -> itzma_core::Result<()> {
    /// use std::sync::Arc;
    ///
    /// use itzma_core::{Document, Metric, Namespace, QueryOptions, Schema};
    /// use object_store::memory::InMemory;
    ///
    /// let store = Arc::new(InMemory::new());
    /// let ns = Namespace::create(store, "docs", Schema::new(1, Metric::L2)).await?;
    /// ns.upsert(vec![Document { id: "a".into(), vector: vec![1.0], attrs: None, sparse: None }]).await?;
    ///
    /// let exact = QueryOptions::default().exact(true);
    /// assert_eq!(ns.query(&[0.0], 1).await?, ns.query_with(&[0.0], 1, &exact).await?);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn query_with(
        &self,
        vector: &[f32],
        top_k: usize,
        options: &QueryOptions,
    ) -> Result<Vec<Hit>> {
        self.schema.validate(vector)?;
        options.validate()?;
        self.check_filter(&options.filter)?;
        let mut budget = ReadBudget::new(self.read_budget);
        let mut plan = self.views.plan(&self.reader(&budget)).await?;
        let mut attempt = 1;
        loop {
            let outcome = self
                .query_once(vector, top_k, options, &plan, &budget)
                .await;
            match outcome {
                Err(err) if is_stale_read(&err) && attempt < MAX_READ_ATTEMPTS => {
                    budget = ReadBudget::new(self.read_budget);
                    plan = self.views.replan(&self.reader(&budget), &plan, err).await?;
                    attempt += 1;
                }
                done => return done,
            }
        }
    }

    /// One attempt at a query against `manifest`, through the engine in
    /// `query.rs` (which reads every segment flat for `exact` or a namespace
    /// with no index it can search).
    async fn query_once(
        &self,
        vector: &[f32],
        top_k: usize,
        options: &QueryOptions,
        plan: &view::Plan,
        budget: &ReadBudget,
    ) -> Result<Vec<Hit>> {
        let manifest = &*plan.manifest;
        let reader = self.reader(budget);
        let tail = self.views.tail(&reader, plan);
        let decoded = self.decoded();
        let q = query::Query {
            store: &*self.query_store,
            namespace: &self.id,
            dim: self.schema.dim,
            metric: self.schema.metric,
            vector,
            top_k,
            options,
            budget,
            policy: &self.fetch_policy,
            registry: &Builtin,
            decoded: decoded.as_ref(),
        };
        query::run(&q, manifest, tail).await
    }

    /// Find the `top_k` documents whose sparse vectors score highest against
    /// `query`, best first.
    ///
    /// `query` is a set of `(term id, weight)` pairs in the same vocabulary the
    /// documents' [`sparse`](Document::sparse) vectors use. A document's score
    /// is the dot product over the terms it shares with the query, an exact
    /// sum with no approximation, so there is no rerank step and nothing to
    /// tune. A document that shares no term with the query, or has no sparse
    /// vector, is never a hit. A term repeated in `query` counts as the sum of
    /// its weights.
    ///
    /// Hits are [`Hit`]s like [`query`](Self::query)'s, and follow the same
    /// convention: `distance` is lower for a closer match, so it is the score
    /// negated. Ties break by id.
    ///
    /// [`SparseQueryOptions::filter`] keeps only documents whose attributes
    /// match a [`Predicate`], with the same checks and meaning as
    /// [`QueryOptions::filter`].
    ///
    /// Shadowing works as it does for [`query_with`](Self::query_with): the
    /// newest version of an id decides, so an overwrite or a delete hides the
    /// older document's sparse vector. The segments and the WAL tail are read
    /// whole and scored directly. Reads go through the
    /// handle's cache if [`with_cache`](Self::with_cache) set one, and are
    /// bounded by the read budget.
    ///
    /// # Errors
    ///
    /// - [`ItzmaError::InvalidQuery`] for a `top_k` of zero, an empty `query`
    ///   or a non-finite weight in it, before any storage call.
    /// - [`ItzmaError::CorruptSegment`] for a damaged `sparse.bin`.
    /// - [`ItzmaError::ReadBudgetExceeded`] and the other errors of
    ///   [`query`](Self::query).
    ///
    /// # Examples
    ///
    /// ```
    /// # #[tokio::main]
    /// # async fn main() -> itzma_core::Result<()> {
    /// use std::sync::Arc;
    ///
    /// use itzma_core::{Document, Metric, Namespace, Schema, SparseQueryOptions};
    /// use object_store::memory::InMemory;
    ///
    /// let store = Arc::new(InMemory::new());
    /// let ns = Namespace::create(store, "docs", Schema::new(1, Metric::L2)).await?;
    /// let doc = |id: &str, terms: Vec<(u32, f32)>| Document {
    ///     id: id.into(),
    ///     vector: vec![0.0],
    ///     attrs: None,
    ///     sparse: Some(terms),
    /// };
    /// ns.upsert(vec![doc("cats", vec![(1, 2.0), (2, 1.0)]), doc("dogs", vec![(3, 2.0)])]).await?;
    ///
    /// let hits = ns.query_sparse(&vec![(1, 1.0), (2, 1.0)], 5, &SparseQueryOptions::default()).await?;
    /// assert_eq!(hits.len(), 1);
    /// assert_eq!(hits[0].id, "cats");
    /// assert_eq!(hits[0].distance, -3.0);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn query_sparse(
        &self,
        query: &SparseVector,
        top_k: usize,
        options: &SparseQueryOptions,
    ) -> Result<Vec<Hit>> {
        sparse_query::validate(query, top_k)?;
        self.check_filter(&options.filter)?;
        let mut budget = ReadBudget::new(self.read_budget);
        let mut plan = self.views.plan(&self.reader(&budget)).await?;
        let mut attempt = 1;
        loop {
            let q = sparse_query::Query {
                store: &*self.query_store,
                namespace: &self.id,
                dim: self.schema.dim,
                query,
                top_k,
                filter: options.filter.as_ref(),
                budget: &budget,
                policy: &self.fetch_policy,
            };
            let reader = self.reader(&budget);
            let tail = self.views.tail(&reader, &plan);
            match sparse_query::run(&q, &plan.manifest, tail).await {
                Err(err) if is_stale_read(&err) && attempt < MAX_READ_ATTEMPTS => {
                    budget = ReadBudget::new(self.read_budget);
                    plan = self.views.replan(&self.reader(&budget), &plan, err).await?;
                    attempt += 1;
                }
                done => return done,
            }
        }
    }

    /// Return the documents whose attributes match `predicate`, in ascending id
    /// order, one page at a time.
    ///
    /// No vector is involved and none is read: a [`Row`](crate::Row) is an id
    /// and its attributes. The predicate is checked against the namespace's
    /// attribute schema first, so a field that is not declared or an operator
    /// that does not fit its type is an error and never an empty result. A
    /// comparison against a null field is false and [`Predicate::Not`] flips it
    /// (see [`Predicate`] for the difference from SQL).
    ///
    /// The newest version of an id decides it, as for every query: an upsert
    /// that no longer matches hides the older copy that did, and a delete hides
    /// it too. Ids are compared in byte order. A page holds at most
    /// [`FilterOptions::limit`] rows, and [`FilterPage::next`] is the id to pass
    /// as [`FilterOptions::after`] for the following one. Paging is not a
    /// snapshot: a write or a compaction between two pages can change what the
    /// second sees, and each page is consistent on its own.
    ///
    /// The cost is linear in the ids and attributes read, and a page stops once
    /// it has `limit` matches, so a loose filter is cheap and a selective one
    /// reads most of the namespace. Reads are bounded by the handle's read
    /// budget and go through its cache if [`with_cache`](Self::with_cache) set
    /// one. A scan that would go over the budget stops with
    /// [`ItzmaError::ReadBudgetExceeded`] and returns no partial page and no
    /// cursor, so a selective filter on a namespace bigger than the budget
    /// cannot be paged through until the budget is raised.
    ///
    /// # Errors
    ///
    /// - [`ItzmaError::InvalidQuery`] for a `limit` of zero or above 10,000, a
    ///   malformed predicate (see [`Predicate`]), or a namespace made before
    ///   typed attributes, which has nothing to filter on.
    /// - [`ItzmaError::UnknownAttr`] and [`ItzmaError::AttrTypeMismatch`] for a
    ///   predicate that does not fit the schema.
    /// - [`ItzmaError::ReadBudgetExceeded`] and the other errors of
    ///   [`query`](Self::query).
    ///
    /// # Examples
    ///
    /// ```
    /// # #[tokio::main]
    /// # async fn main() -> itzma_core::Result<()> {
    /// use std::sync::Arc;
    ///
    /// use itzma_core::{
    ///     AttrSchema, AttrType, Compare, Document, FilterOptions, Metric, Namespace,
    ///     Predicate, Schema,
    /// };
    /// use object_store::memory::InMemory;
    ///
    /// let schema = Schema::new(1, Metric::L2)
    ///     .with_attrs(AttrSchema::new().field("price", AttrType::Number));
    /// let ns = Namespace::create(Arc::new(InMemory::new()), "shop", schema).await?;
    /// let item = |id: &str, price: f64| Document {
    ///     id: id.into(),
    ///     vector: vec![0.0],
    ///     attrs: Some(serde_json::json!({ "price": price })),
    ///     sparse: None,
    /// };
    /// ns.upsert(vec![item("a", 5.0), item("b", 50.0), item("c", 12.0)]).await?;
    ///
    /// let cheap = Predicate::number("price", Compare::Lt, 20.0);
    /// let page = ns.filter(&cheap, &FilterOptions::default()).await?;
    /// let ids: Vec<&str> = page.rows.iter().map(|r| r.id.as_str()).collect();
    /// assert_eq!(ids, ["a", "c"]);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn filter(
        &self,
        predicate: &Predicate,
        options: &FilterOptions,
    ) -> Result<FilterPage> {
        options.validate()?;
        if self.schema.attrs.is_free_form() {
            return Err(ItzmaError::InvalidQuery(
                "this namespace has no attribute schema, so there is nothing to filter on",
            ));
        }
        predicate.check(&self.schema.attrs)?;
        let mut budget = ReadBudget::new(self.read_budget);
        let mut plan = self.views.plan(&self.reader(&budget)).await?;
        let mut attempt = 1;
        let decoded = self.decoded();
        loop {
            let scan = filter::Scan {
                store: &*self.query_store,
                namespace: &self.id,
                dim: self.schema.dim,
                predicate,
                limit: options.limit,
                after: options.after.as_deref(),
                budget: &budget,
                policy: &self.fetch_policy,
                decoded: decoded.as_ref(),
            };
            let reader = self.reader(&budget);
            let tail = self.views.tail(&reader, &plan);
            match filter::run(&scan, &plan.manifest, tail).await {
                Err(err) if is_stale_read(&err) && attempt < MAX_READ_ATTEMPTS => {
                    budget = ReadBudget::new(self.read_budget);
                    plan = self.views.replan(&self.reader(&budget), &plan, err).await?;
                    attempt += 1;
                }
                done => return done,
            }
        }
    }

    /// Refuse a filter that cannot run, before any storage call.
    fn check_filter(&self, filter: &Option<Predicate>) -> Result<()> {
        let Some(predicate) = filter else {
            return Ok(());
        };
        if self.schema.attrs.is_free_form() {
            return Err(ItzmaError::InvalidQuery(
                "this namespace has no attribute schema, so there is nothing to filter on",
            ));
        }
        predicate.check(&self.schema.attrs)
    }

    /// Where this handle's queries read their snapshot from.
    fn reader<'a>(&'a self, budget: &'a ReadBudget) -> view::Reader<'a> {
        view::Reader {
            store: &*self.store,
            query_store: &*self.query_store,
            namespace: &self.id,
            budget,
        }
    }

    /// `err` came from reading against `stale`. If it says a segment is gone
    /// or a WAL object is a fence, and the manifest has changed since, the
    /// caller was only behind (a compaction and a garbage collection ran after
    /// it loaded): return the current manifest to try again. Any other error,
    /// or an unchanged manifest, means the object really is missing or the fence
    /// really is misplaced, and `err` comes back as it was.
    async fn reload_if_moved(&self, stale: &Manifest, err: ItzmaError) -> Result<Manifest> {
        if !is_stale_read(&err) {
            return Err(err);
        }
        let fresh = self.load_manifest().await?;
        if fresh == *stale { Err(err) } else { Ok(fresh) }
    }

    /// Load the manifest and check its segment list against the invariant, so
    /// every reader fails the same way on a damaged manifest.
    async fn load_manifest(&self) -> Result<Manifest> {
        view::load_checked(&*self.store, &self.id).await
    }
}

/// Garbage collection removes segments and index objects and fences WAL
/// objects, so a reader that loaded its manifest before a collection
/// can meet any of the three. All of them mean "reload and look again" before
/// they mean damage.
pub(crate) fn is_stale_read(err: &ItzmaError) -> bool {
    segment::is_missing(err) || index::is_missing(err) || wal_codec::is_fenced(err)
}

/// Whether any upsert in `ops` carries a sparse vector — such a
/// batch needs [`wal::SPARSE_FORMAT_VERSION`] before it is written.
fn has_sparse(ops: &[WalOp]) -> bool {
    ops.iter()
        .any(|op| matches!(op, WalOp::Upsert(doc) if doc.sparse.is_some()))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use object_store::memory::InMemory;

    use super::*;
    use crate::document::Metric;

    #[test]
    fn fetch_policy_defaults_to_object_store_and_with_fetch_policy_overrides_it() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let manifest = Manifest::new(Schema::new(1, Metric::L2));
        let ns = Namespace::from_manifest(store, "docs", &manifest);
        assert_eq!(ns.fetch_policy, FetchPolicy::object_store());
        let ns = ns.with_fetch_policy(FetchPolicy::local());
        assert_eq!(ns.fetch_policy, FetchPolicy::local());
    }
}
