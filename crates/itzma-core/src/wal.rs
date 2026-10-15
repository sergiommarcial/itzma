//! The write-ahead log, sequenced by put-if-absent.
//!
//! Each write batch becomes one object, `<namespace>/wal/<seq>.json`, where
//! `<seq>` is a `u64` zero-padded to 20 digits so that listing in key order is
//! listing in sequence order. Sequence numbers start at 1. The object's
//! content is binary (`wal_codec.rs`). The `.json` suffix is kept so
//! that every writer fights for the same key.
//!
//! There is no coordinator and no lock. A writer claims a sequence number by
//! creating that key with `PutMode::Create`, which the object store performs
//! atomically: of any number of writers racing for the same key, exactly one
//! wins. A loser learns which seqs are taken by listing, jumps past them, and
//! tries again. Once the put returns, the batch is durable.
//!
//! **Invariant:** a writer only tries seq `n + 1` after it knows `n` exists
//! (it wrote `n`, or it saw `n` in a listing). So the log is always a
//! contiguous run with no holes, and [`read_log`] treats a hole as corruption.

use bytes::Bytes;
use futures::stream::{self, BoxStream, StreamExt, TryStreamExt};
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt, PutMode, PutOptions, PutPayload};
use serde::{Deserialize, Serialize};

use crate::budget::ReadBudget;
use crate::cpu::{self, Cost, Lane};
use crate::document::WalOp;
use crate::error::{ItzmaError, Result};
use crate::wal_codec::{self, Peek};

/// The newest `wal_format` level this build reads, and so the highest a
/// namespace can hold before this build refuses it. [`read_log`]
/// reads WAL object versions 2 (binary) and 3 (binary with an
/// idempotency key), and rejects any other with
/// [`ItzmaError::UnsupportedFormat`]. A decoded [`WalEntry`] carries the
/// version of the object it came from. Level 4 adds the fence marker that
/// garbage collection leaves at the key of an old object, which is
/// not a batch and never comes out of [`read_log`] except as an error.
///
/// [`append`] writes version 2 for an ordinary batch. A keyed write produces
/// version 3, needing a namespace at level [`KEYED_FORMAT_VERSION`].
pub const FORMAT_VERSION: u32 = FENCE_FORMAT_VERSION;

/// The WAL format of an object with an idempotency key, and the level a
/// namespace's `wal_format` must have reached before a keyed write is allowed.
pub const KEYED_FORMAT_VERSION: u32 = 3;

/// The level a namespace's `wal_format` must have reached before
/// [`Namespace::gc`](crate::Namespace::gc) may fence WAL objects.
pub const FENCE_FORMAT_VERSION: u32 = 4;

/// How many sequence numbers [`append`] tries before giving up with
/// [`ItzmaError::WriteContention`].
pub const MAX_APPEND_ATTEMPTS: u32 = 32;

/// The largest batch [`append`] accepts, measured as the uncompressed body of
/// the object: 64 MiB, about 21,000 vectors of 768 dimensions.
///
/// This is a writer policy under the format's own ceiling (a u32 `body_len`,
/// which readers accept), so it can change without a format version. It sits
/// well under the default read budget ([`DEFAULT_READ_BUDGET`](crate::DEFAULT_READ_BUDGET))
/// so that one batch can never be too big for a reader to load. It
/// was 1 GiB before that. Objects written under the old limit are still
/// read. A bigger batch is refused with [`ItzmaError::BatchTooLarge`] before
/// anything is built or written.
pub const MAX_BODY_BYTES: usize = 64 << 20;

/// A listing that shows a gap is re-listed this many times before erroring,
/// because a multi-page listing can race a writer.
const MAX_LIST_ATTEMPTS: u32 = 3;
const FETCH_CONCURRENCY: usize = 16;

/// The contents of one WAL object: a batch of ops that was committed together.
///
/// This is the decoded shape of one object, whichever version it came from:
/// `format_version` is 2, or 3 for an object with a key.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WalEntry {
    /// Encoding version of the object this came from: 2, or 3 with a key.
    pub format_version: u32,
    /// The idempotency key the batch was written with. Only version 3 objects
    /// have one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    /// The batch, in the order the caller supplied it. Never empty.
    pub ops: Vec<WalOp>,
}

/// A caller-chosen name for one write, so that retrying it does not append the
/// batch twice.
///
/// The key goes into the header of the WAL object. Nothing can tell from the
/// key alone whether it was used before, so a retry says where to look:
///
/// - [`WriteKey::new`] is a first attempt. The batch is written as it is, with
///   no lookup, one put.
/// - [`WriteKey::retry`] may be a retry. Before writing, the writer reads the
///   headers of the WAL objects from `search_from` to the end, and if one
///   carries the key it returns that object's seq and writes nothing.
///   `search_from` has to be at or before the original's seq. The last seq the
///   caller saw committed, plus one, is always safe. `1` is safe too and reads
///   the whole log.
///
/// Either way, a put that runs into the writer's own object at the seq it is
/// trying (the store's own retry after a lost acknowledgement does this)
/// recognises it and returns that seq.
///
/// A key names one batch. Using it for a different batch is
/// [`ItzmaError::KeyConflict`] when the two bodies differ in length, and
/// silently drops the second batch when they do not, so derive keys from the
/// batch, for example `<job>/<batch number>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteKey {
    /// 1 to 128 bytes. Checked when the write is made, not here.
    pub key: String,
    /// Where a retry starts looking for an earlier attempt. `None` skips the
    /// lookup.
    pub search_from: Option<u64>,
}

impl WriteKey {
    /// A first attempt: no lookup, one put.
    pub fn new(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            search_from: None,
        }
    }

    /// A possible retry: look for the key in the WAL objects from
    /// `search_from` before writing.
    pub fn retry(key: impl Into<String>, search_from: u64) -> Self {
        Self {
            key: key.into(),
            search_from: Some(search_from),
        }
    }
}

impl WriteKey {
    /// The key checks that need no storage, so a caller can run them first.
    pub(crate) fn validate(&self) -> Result<()> {
        wal_codec::validate_key(&self.key).map(|_| ())
    }
}

/// The key and body length of the batch being written, which is what an
/// earlier object has to match to count as this write.
#[derive(Clone, Copy)]
struct Ours<'a> {
    key: &'a str,
    body_len: u32,
}

impl Ours<'_> {
    /// `found` carries this write's key at `seq`. It is this write if the body
    /// has the same length, and a reused key if not.
    fn claim(self, seq: u64, found: &Peek) -> Result<u64> {
        if found.body_len == self.body_len {
            Ok(seq)
        } else {
            Err(ItzmaError::KeyConflict {
                key: self.key.to_string(),
            })
        }
    }
}

/// Key prefix under which a namespace's WAL objects live: `<namespace>/wal`.
///
/// Does not validate `namespace`; that is [`Namespace`](crate::Namespace)'s job.
pub fn wal_prefix(namespace: &str) -> Path {
    Path::from(format!("{namespace}/wal"))
}

/// Object key for one WAL entry: `<namespace>/wal/<seq, 20 digits>.json`.
///
/// ```
/// use itzma_core::wal::wal_path;
///
/// assert_eq!(
///     wal_path("docs", 7).as_ref(),
///     "docs/wal/00000000000000000007.json"
/// );
/// ```
pub fn wal_path(namespace: &str, seq: u64) -> Path {
    Path::from(format!("{namespace}/wal/{seq:020}.json"))
}

/// Recover the sequence number from a WAL key, or `None` for anything that is
/// not `<digits>.json`. Foreign files under `wal/` are skipped, not fatal.
pub(crate) fn parse_seq(path: &Path) -> Option<u64> {
    path.filename()?.strip_suffix(".json")?.parse().ok()
}

/// Commit a batch of ops as the next WAL entry and return its sequence number.
///
/// `hint` is where to start probing. Callers keep it at "one past the last seq
/// I know about", which is right in the common single-writer case and costs
/// one put. If that seq is taken, the put fails with `AlreadyExists`; the
/// function then lists the WAL, jumps to one past the highest seq it can see,
/// and tries again. A stale hint therefore costs an extra listing, never
/// correctness. A `hint` of 0 is treated as 1, since seq 0 is reserved to mean
/// "nothing folded".
///
/// The batch is written in the binary format (`wal_codec`). The returned seq is
/// durable: the object exists in the store when this returns. Ops are written
/// as given; validating them against a schema is the caller's job, except that
/// the format stores the upserts' vectors as one matrix, so they must all have
/// the same length.
///
/// Storage errors other than a lost race are returned immediately, not
/// retried. If the call fails after a put whose response was lost (a network
/// error, say), the entry may in fact exist, so a blind retry can apply the
/// batch twice. Upserts and deletes are idempotent by id, so replaying in
/// order is safe, but a replay can overwrite a newer write. [`append_keyed`]
/// is the write whose retry is a no-op.
///
/// # Errors
///
/// - [`ItzmaError::EmptyBatch`] if `ops` is empty.
/// - [`ItzmaError::BatchTooLarge`] if the batch is over [`MAX_BODY_BYTES`].
/// - [`ItzmaError::DimensionMismatch`] if the upserts' vectors differ in
///   length.
/// - [`ItzmaError::Serde`] if an attribute cannot be serialized.
/// - [`ItzmaError::WriteContention`] after 32 lost races.
/// - [`ItzmaError::Storage`] for any other backend failure.
///
/// # Examples
///
/// ```
/// # #[tokio::main]
/// # async fn main() -> itzma_core::Result<()> {
/// use itzma_core::wal;
/// use itzma_core::{Document, WalOp};
/// use object_store::memory::InMemory;
///
/// let store = InMemory::new();
/// let op = |id: &str| WalOp::Upsert(Document {
///     id: id.into(),
///     vector: vec![1.0],
///     attrs: None,
/// });
///
/// assert_eq!(wal::append(&store, "docs", 1, vec![op("a")]).await?, 1);
/// // A stale hint lands on the next free seq anyway.
/// assert_eq!(wal::append(&store, "docs", 1, vec![op("b")]).await?, 2);
/// # Ok(())
/// # }
/// ```
pub async fn append(
    store: &dyn ObjectStore,
    namespace: &str,
    hint: u64,
    ops: Vec<WalOp>,
) -> Result<u64> {
    append_limited(store, namespace, hint, ops, MAX_BODY_BYTES).await
}

/// [`append`] with the size limit as a parameter, so a test can hit it with a
/// tiny batch.
async fn append_limited(
    store: &dyn ObjectStore,
    namespace: &str,
    hint: u64,
    ops: Vec<WalOp>,
    max_body: usize,
) -> Result<u64> {
    if ops.is_empty() {
        return Err(ItzmaError::EmptyBatch);
    }
    let bytes = wal_codec::encode(&ops, max_body)?;
    put_encoded(store, namespace, hint, bytes).await
}

/// Put an object [`wal_codec`] already encoded at the first free seq from
/// `hint` on, as [`append`] does after encoding.
pub(crate) async fn put_encoded(
    store: &dyn ObjectStore,
    namespace: &str,
    hint: u64,
    bytes: Bytes,
) -> Result<u64> {
    put_next(store, namespace, hint, bytes, None).await
}

/// Commit a batch under an idempotency key and return the sequence number it
/// is at. The retry rules are on [`WriteKey`].
///
/// The batch is written as a version 3 object, which a namespace
/// only takes once its `wal_format` has been raised to
/// [`KEYED_FORMAT_VERSION`]. This function does not check that, because it
/// does not read the manifest: [`Namespace::upsert_keyed`](crate::Namespace::upsert_keyed)
/// does. Called directly on a store, it writes version 3 objects into any log.
///
/// `hint` works as for [`append`]. On a `PutMode::Create` collision the writer
/// reads the header of the object it ran into, and if that object has this
/// key, the write is already there and its seq is returned.
///
/// # Errors
///
/// As for [`append`], plus:
///
/// - [`ItzmaError::InvalidKey`] unless the key is 1 to 128 bytes.
/// - [`ItzmaError::KeyConflict`] if an object with this key holds a batch of a
///   different size. Nothing is written.
/// - [`ItzmaError::CorruptWal`] or [`ItzmaError::UnsupportedFormat`] if a
///   header the search reads is damaged.
///
/// # Examples
///
/// ```
/// # #[tokio::main]
/// # async fn main() -> itzma_core::Result<()> {
/// use itzma_core::wal::{self, WriteKey};
/// use itzma_core::{Document, WalOp};
/// use object_store::memory::InMemory;
///
/// let store = InMemory::new();
/// let batch = || vec![WalOp::Upsert(Document {
///     id: "a".into(),
///     vector: vec![1.0],
///     attrs: None,
/// })];
///
/// let seq = wal::append_keyed(&store, "docs", 1, &WriteKey::new("load-7/batch-1"), batch()).await?;
///
/// // The acknowledgement was lost, so the caller tries again from seq 1.
/// let retry = WriteKey::retry("load-7/batch-1", 1);
/// assert_eq!(wal::append_keyed(&store, "docs", 1, &retry, batch()).await?, seq);
/// assert_eq!(wal::read_log(&store, "docs", 0).await?.len(), 1);
/// # Ok(())
/// # }
/// ```
pub async fn append_keyed(
    store: &dyn ObjectStore,
    namespace: &str,
    hint: u64,
    write: &WriteKey,
    ops: Vec<WalOp>,
) -> Result<u64> {
    if ops.is_empty() {
        return Err(ItzmaError::EmptyBatch);
    }
    let object = wal_codec::encode_keyed(&ops, MAX_BODY_BYTES, &write.key)?;
    put_encoded_keyed(store, namespace, hint, write, object).await
}

/// Put a keyed object [`wal_codec`] already encoded, with [`append_keyed`]'s
/// search for an earlier attempt and its collision rule.
pub(crate) async fn put_encoded_keyed(
    store: &dyn ObjectStore,
    namespace: &str,
    hint: u64,
    write: &WriteKey,
    object: wal_codec::Keyed,
) -> Result<u64> {
    let ours = Ours {
        key: &write.key,
        body_len: object.body_len,
    };
    if let Some(from) = write.search_from {
        if let Some(seq) = find_key(store, namespace, from, ours).await? {
            return Ok(seq);
        }
    }
    put_next(store, namespace, hint, object.bytes, Some(ours)).await
}

/// Put `bytes` at the first free seq from `hint` on. With `ours`, a collision
/// with an object that is this write returns its seq instead of moving on.
async fn put_next(
    store: &dyn ObjectStore,
    namespace: &str,
    hint: u64,
    bytes: Bytes,
    ours: Option<Ours<'_>>,
) -> Result<u64> {
    // Seq 0 is reserved: listings start strictly after `wal_folded_through`.
    let mut seq = hint.max(1);
    for _ in 0..MAX_APPEND_ATTEMPTS {
        let opts = PutOptions::from(PutMode::Create);
        let payload = PutPayload::from(bytes.clone());
        match store
            .put_opts(&wal_path(namespace, seq), payload, opts)
            .await
        {
            Ok(_) => return Ok(seq),
            Err(object_store::Error::AlreadyExists { .. }) => {
                if let Some(ours) = ours {
                    if let Some(found) = own_object_at(store, namespace, seq, ours).await? {
                        return Ok(found);
                    }
                }
                seq = next_free_after(store, namespace, seq).await?;
            }
            Err(e) => return Err(e.into()),
        }
    }
    Err(ItzmaError::WriteContention {
        attempts: MAX_APPEND_ATTEMPTS,
    })
}

/// The seq of the object at `seq` if it carries `ours`'s key.
async fn own_object_at(
    store: &dyn ObjectStore,
    namespace: &str,
    seq: u64,
    ours: Ours<'_>,
) -> Result<Option<u64>> {
    match peek_at(store, &wal_path(namespace, seq)).await? {
        Some(found) if found.key == ours.key => ours.claim(seq, &found).map(Some),
        _ => Ok(None),
    }
}

/// The lowest seq at or after `from` whose object carries `ours`'s key.
/// Lists the WAL past `from` and reads each object's header, 16 at a time.
async fn find_key(
    store: &dyn ObjectStore,
    namespace: &str,
    from: u64,
    ours: Ours<'_>,
) -> Result<Option<u64>> {
    let prefix = wal_prefix(namespace);
    let offset = wal_path(namespace, from.saturating_sub(1));
    let mut found: Vec<(u64, Path)> = store
        .list_with_offset(Some(&prefix), &offset)
        .try_filter_map(|m| async move { Ok(parse_seq(&m.location).map(|seq| (seq, m.location))) })
        .try_collect()
        .await?;
    found.sort_by_key(|(seq, _)| *seq);

    let mut heads = stream::iter(found)
        .map(|(seq, path)| async move { Ok::<_, ItzmaError>((seq, peek_at(store, &path).await?)) })
        .buffered(FETCH_CONCURRENCY);
    while let Some((seq, head)) = heads.try_next().await? {
        if let Some(found) = head {
            if found.key == ours.key {
                return ours.claim(seq, &found).map(Some);
            }
        }
    }
    Ok(None)
}

/// The key and body length in an object's header, from a ranged GET of its
/// first bytes. `None` for an object that has no key.
async fn peek_at(store: &dyn ObjectStore, path: &Path) -> Result<Option<Peek>> {
    let head = store
        .get_range(path, 0..wal_codec::HEADER_PEEK_LEN as u64)
        .await?;
    wal_codec::peek_key(path.as_ref(), &head)
}

/// One past the highest seq at or after `collided`. Lists with an offset, so
/// only entries newer than the collision are fetched from the store.
async fn next_free_after(store: &dyn ObjectStore, namespace: &str, collided: u64) -> Result<u64> {
    let prefix = wal_prefix(namespace);
    let metas: Vec<_> = store
        .list_with_offset(Some(&prefix), &wal_path(namespace, collided))
        .try_collect()
        .await?;
    let highest = metas
        .iter()
        .filter_map(|m| parse_seq(&m.location))
        .max()
        .unwrap_or(collided)
        .max(collided);
    Ok(highest + 1)
}

/// Read every WAL entry after `folded_through`, in sequence order.
///
/// Lists the WAL past `folded_through` (exclusive), checks the seqs form an
/// unbroken run starting at `folded_through + 1`, then fetches and decodes the
/// objects with up to 16 requests in flight. The result pairs each seq with
/// its entry and is empty for a namespace that has no un-folded writes.
///
/// A listing is not a snapshot: on a store that pages results, a writer can
/// commit between pages and make a healthy log look like it has a hole. So a
/// listing that shows a gap is repeated, up to 3 times in total, before the
/// gap is believed.
///
/// Reads are consistent with writes that have already returned: object-store
/// listings are strongly consistent, so an entry that [`append`] has returned
/// for will appear.
///
/// # Errors
///
/// - [`ItzmaError::WalGap`] if the seqs are not contiguous.
/// - [`ItzmaError::UnsupportedFormat`] if an object's version is neither 2
///   nor 3.
/// - [`ItzmaError::CorruptWal`] if an object is damaged or is not a WAL
///   object at all (bad magic). The error names
///   the object key. So does a fence inside the range, since it holds
///   no batch: a namespace whose old objects were fenced can be read only from
///   its fold point on.
/// - [`ItzmaError::Storage`] for any backend failure.
///
/// # Examples
///
/// ```
/// # #[tokio::main]
/// # async fn main() -> itzma_core::Result<()> {
/// use itzma_core::WalOp;
/// use itzma_core::wal;
/// use object_store::memory::InMemory;
///
/// let store = InMemory::new();
/// let ops = vec![WalOp::Delete("a".into())];
/// wal::append(&store, "docs", 1, ops.clone()).await?;
///
/// let log = wal::read_log(&store, "docs", 0).await?;
/// assert_eq!(log.len(), 1);
/// assert_eq!(log[0].0, 1);
/// assert_eq!(log[0].1.ops, ops);
///
/// // Entries at or below `folded_through` are skipped.
/// assert!(wal::read_log(&store, "docs", 1).await?.is_empty());
/// # Ok(())
/// # }
/// ```
pub async fn read_log(
    store: &dyn ObjectStore,
    namespace: &str,
    folded_through: u64,
) -> Result<Vec<(u64, WalEntry)>> {
    let budget = ReadBudget::unlimited();
    stream_log(store, namespace, folded_through, &budget, Lane::Maintenance)
        .await?
        .try_collect()
        .await
}

/// List the WAL past `folded_through` and check it is one unbroken run, then
/// return its entries as a stream in seq order, fetched 16 at a time. The
/// caller folds each entry as it arrives, so the tail is never held twice.
/// Every object is charged to `budget` before its body is read.
///
/// The listing and the gap check are [`read_log`]'s, including the repeat
/// listings before a gap is believed. Errors from fetching or decoding an
/// object come out of the stream.
pub(crate) async fn stream_log<'a>(
    store: &'a dyn ObjectStore,
    namespace: &str,
    folded_through: u64,
    budget: &'a ReadBudget,
    lane: Lane,
) -> Result<BoxStream<'a, Result<(u64, WalEntry)>>> {
    let found = list_tail(store, namespace, folded_through).await?;
    Ok(
        stream::iter(found)
            .map(move |(seq, path)| async move {
                Ok((seq, fetch_one(store, &path, budget, lane).await?))
            })
            .buffered(FETCH_CONCURRENCY)
            .boxed(),
    )
}

async fn list_tail(
    store: &dyn ObjectStore,
    namespace: &str,
    folded_through: u64,
) -> Result<Vec<(u64, Path)>> {
    let prefix = wal_prefix(namespace);
    let offset = wal_path(namespace, folded_through);
    let mut gap = (0, 0);
    for _ in 0..MAX_LIST_ATTEMPTS {
        let mut found: Vec<(u64, Path)> = store
            .list_with_offset(Some(&prefix), &offset)
            .try_filter_map(
                |m| async move { Ok(parse_seq(&m.location).map(|seq| (seq, m.location))) },
            )
            .try_collect()
            .await?;
        found.sort_by_key(|(seq, _)| *seq);

        let mismatch = found
            .iter()
            .zip(folded_through + 1..)
            .find(|((seq, _), want)| seq != want)
            .map(|((seq, _), want)| (want, *seq));
        match mismatch {
            None => return Ok(found),
            Some(g) => gap = g,
        }
    }
    Err(ItzmaError::WalGap {
        expected: gap.0,
        found: gap.1,
    })
}

/// GET and decode one object. The charge is made as soon as the header is in
/// hand, from the first bytes of the same response, so a bigger object than
/// the budget allows is not downloaded to the end. A binary object is charged
/// the larger of its size and its declared `body_len` (a compressed object
/// decodes to `body_len`).
async fn fetch_one(
    store: &dyn ObjectStore,
    path: &Path,
    budget: &ReadBudget,
    lane: Lane,
) -> Result<WalEntry> {
    let got = store.get(path).await?;
    let size = got.meta.size;
    let mut chunks = got.into_stream();
    let mut bytes: Vec<u8> = Vec::new();
    let mut charged = false;
    while let Some(chunk) = chunks.try_next().await? {
        bytes.extend_from_slice(&chunk);
        if !charged && bytes.len() >= wal_codec::HEADER_LEN {
            charge(budget, path, size, &bytes)?;
            charged = true;
        }
    }
    if !charged {
        charge(budget, path, size, &bytes)?;
    }
    let cost = decode_cost(path.as_ref(), &bytes);
    let path = path.clone();
    cpu::run(lane, cost, move || decode_object(&path, &bytes)).await
}

/// What decoding `bytes` costs: a compressed object decodes to its declared
/// `body_len`, which can be hundreds of times its stored size.
fn decode_cost(path: &str, bytes: &[u8]) -> Cost {
    let declared = wal_codec::declared_body_len(path, bytes)
        .ok()
        .flatten()
        .and_then(|n| usize::try_from(n).ok())
        .unwrap_or(0);
    Cost::Bytes(bytes.len().max(declared))
}

fn charge(budget: &ReadBudget, path: &Path, size: u64, prefix: &[u8]) -> Result<()> {
    let declared = wal_codec::declared_body_len(path.as_ref(), prefix)?.unwrap_or(0);
    budget.charge(path.as_ref(), size.max(declared))
}

/// Version 2 and 3 objects start with the `ITZW` magic, and a fence with `ITZF`,
/// which is not a batch and comes back as [`wal_codec::fenced`]. Anything
/// else is [`ItzmaError::CorruptWal`] ("bad magic").
fn decode_object(path: &Path, bytes: &[u8]) -> Result<WalEntry> {
    if wal_codec::is_fence(bytes) {
        wal_codec::check_fence(path.as_ref(), bytes)?;
        return Err(wal_codec::fenced(path.as_ref()));
    }
    wal_codec::decode(path.as_ref(), bytes)
}

#[cfg(test)]
mod tests {
    use futures::TryStreamExt;
    use object_store::memory::InMemory;

    use super::*;
    use crate::document::Document;

    fn op(id: &str) -> WalOp {
        WalOp::Upsert(Document {
            id: id.into(),
            vector: vec![1.0, 2.0],
            attrs: None,
        })
    }

    #[test]
    fn a_compressed_object_costs_what_it_decodes_to() {
        // Ten thousand copies of one document: ZSTD shrinks the body a lot.
        let ops: Vec<WalOp> = (0..10_000).map(|i| op(&format!("d{i:05}"))).collect();
        let bytes = wal_codec::encode(&ops, usize::MAX).unwrap();
        let body = wal_codec::declared_body_len("p", &bytes).unwrap().unwrap();
        assert!(body > bytes.len() as u64 * 4, "the body compressed");
        let cost = decode_cost("p", &bytes).units();
        assert_eq!(cost, crate::cpu::Cost::Bytes(body as usize).units());
    }

    #[tokio::test]
    async fn a_batch_over_the_limit_is_refused_and_the_store_is_untouched() {
        let store = InMemory::new();
        append(&store, "ns", 1, vec![op("a")]).await.unwrap();
        let before: Vec<_> = store.list(None).try_collect().await.unwrap();

        let err = append_limited(&store, "ns", 2, vec![op("b")], 8)
            .await
            .unwrap_err();
        assert!(
            matches!(err, ItzmaError::BatchTooLarge { max: 8, .. }),
            "{err:?}"
        );

        let after: Vec<_> = store.list(None).try_collect().await.unwrap();
        assert_eq!(before, after);
    }

    #[test]
    fn the_build_reads_fences_and_says_so() {
        assert_eq!(FENCE_FORMAT_VERSION, 4);
        assert_eq!(FORMAT_VERSION, FENCE_FORMAT_VERSION);
    }

    #[test]
    fn a_fence_decodes_to_the_fenced_error() {
        let path = wal_path("ns", 1);
        let err = decode_object(&path, &wal_codec::fence()).unwrap_err();
        assert!(wal_codec::is_fenced(&err), "{err:?}");
    }

    #[test]
    fn a_damaged_fence_is_not_the_fenced_error() {
        let path = wal_path("ns", 1);
        let mut longer = wal_codec::fence().to_vec();
        longer.push(0);
        let err = decode_object(&path, &longer).unwrap_err();
        assert!(matches!(err, ItzmaError::CorruptWal { .. }), "{err:?}");
        assert!(!wal_codec::is_fenced(&err));
    }

    #[tokio::test]
    async fn read_log_stops_at_a_fence_inside_its_range() {
        let store = InMemory::new();
        append(&store, "ns", 1, vec![op("a")]).await.unwrap();
        append(&store, "ns", 2, vec![op("b")]).await.unwrap();
        store
            .put(&wal_path("ns", 1), wal_codec::fence().into())
            .await
            .unwrap();

        let err = read_log(&store, "ns", 0).await.unwrap_err();
        assert!(wal_codec::is_fenced(&err), "{err:?}");
        // From the fold point on, the fence is not in range.
        let tail = read_log(&store, "ns", 1).await.unwrap();
        assert_eq!(tail.len(), 1);
        assert_eq!(tail[0].0, 2);
    }
}
