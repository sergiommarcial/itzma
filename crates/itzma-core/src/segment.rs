//! Segments: immutable binary snapshots of a layer.
//!
//! A segment is one object, `<namespace>/segments/<lo>-<hi>/data.bin`, holding
//! the net effect of WAL entries `lo` through `hi`. The id carries the range,
//! so the manifest can be checked for contiguity from the ids alone, and the
//! bytes are a pure function of the range: any compactor building the same
//! range produces the same object, which is what lets a rerun or a racing
//! compactor reuse an existing one.
//!
//! Three formats exist, and [`decode`] reads all of them. Version 1 is a flat
//! run of sections. Version 2 (in the private `v2` module) holds the
//! same content in blocks with a directory and a Bloom filter at the end, so a
//! reader can fetch pieces of it. Version 3 (the same module with
//! `columns` and `zone`) stores a typed namespace's attributes as columns per
//! block with a zone map beside them. [`encode_with_schema`] writes any of
//! them as compaction does, and the namespace's `segment_format` level decides
//! which one compaction writes.
//!
//! Ids and tombstones are sorted, strictly ascending, and never overlap.
//! [`decode`] checks every length against the bytes that remain before
//! allocating and never panics.

mod attr_rows;
mod columns;
mod ranged;
mod v2;
mod view;
mod writer;
mod zone;

pub(crate) use ranged::{DocWindow, Held, RangedSegment};
pub(crate) use view::SegmentView;
pub(crate) use writer::{AttrIn, SegmentWriter, attr_json, encode_attr_block};

use bytes::Bytes;
use futures::stream::{self, StreamExt, TryStreamExt};
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt, PutMode, PutOptions, PutPayload, UpdateVersion};

use crate::attrs::AttrSchema;
use crate::budget::ReadBudget;
use crate::cpu::{self, Cost, Lane};
use crate::decoded::{DecodedRef, What};
use crate::document::{Document, Schema};
use crate::error::{ItzmaError, Result};
use crate::fetch::FetchPolicy;
use crate::layer::Layer;

/// The newest segment version this build reads, and the highest level
/// [`Namespace::upgrade_segment_format`](crate::Namespace::upgrade_segment_format)
/// accepts. [`decode`] rejects a version above it (and 0) with
/// [`ItzmaError::UnsupportedFormat`]. A change to an encoding must add a
/// version, because correctness relies on two builders producing identical
/// bytes.
pub const FORMAT_VERSION: u32 = 3;

/// The segment level a namespace created by this build starts at, which can
/// trail [`FORMAT_VERSION`]: a new version is readable first and written by
/// default only once it pays off (version 3 is opt-in until its zone
/// maps are used).
pub const DEFAULT_FORMAT: u32 = 2;

const MAGIC: [u8; 4] = *b"ITZS";
/// Magic, version, dim, doc count, tombstone count.
const FETCH_CONCURRENCY: usize = 16;

/// The id of the segment covering WAL seqs `lo` through `hi`: both numbers
/// zero-padded to 20 digits, so ids sort in range order.
///
/// ```
/// use itzma_core::segment::segment_id;
///
/// assert_eq!(segment_id(1, 10), "00000000000000000001-00000000000000000010");
/// ```
pub fn segment_id(lo: u64, hi: u64) -> String {
    format!("{lo:020}-{hi:020}")
}

/// Recover `(lo, hi)` from a segment id, or `None` unless the id is exactly
/// what [`segment_id`] produces and `1 <= lo <= hi`.
pub fn parse_segment_id(id: &str) -> Option<(u64, u64)> {
    let (lo, hi) = id.split_once('-')?;
    let (lo, hi) = (lo.parse::<u64>().ok()?, hi.parse::<u64>().ok()?);
    ((1..=hi).contains(&lo) && segment_id(lo, hi) == id).then_some((lo, hi))
}

/// Object key of a segment: `<namespace>/segments/<id>/data.bin`. The
/// directory leaves room for index files beside the data later.
///
/// Does not validate `namespace`; that is [`Namespace`](crate::Namespace)'s job.
pub fn segment_path(namespace: &str, id: &str) -> Path {
    Path::from(format!("{namespace}/segments/{id}/data.bin"))
}

/// The file names under one segment's prefix: its data object, its sidecars,
/// and every derived index object, named or not.
pub(crate) async fn object_names(
    store: &dyn ObjectStore,
    namespace: &str,
    id: &str,
) -> Result<Vec<String>> {
    let prefix = Path::from(format!("{namespace}/segments/{id}"));
    let metas: Vec<_> = store
        .list(Some(&prefix))
        .try_collect()
        .await
        .map_err(ItzmaError::from)?;
    Ok(metas
        .iter()
        .filter_map(|meta| meta.location.filename().map(str::to_string))
        .collect())
}

/// The highest generation of one kind the manifest names for the segment or
/// any object under its prefix carries, 0 if none.
pub(crate) fn generation_floor(
    named: Option<u32>,
    names: &[String],
    generation_of: fn(&str) -> Option<u32>,
) -> u32 {
    names
        .iter()
        .filter_map(|name| generation_of(name))
        .chain(named)
        .max()
        .unwrap_or(0)
}

/// Key of a segment's generation high-water mark: the highest generation any
/// derived object of the segment has claimed, in decimal.
pub(crate) fn generation_mark_path(namespace: &str, id: &str) -> Path {
    Path::from(format!("{namespace}/segments/{id}/generation"))
}

/// The mark never holds more than a u32 in decimal.
const MAX_MARK_BYTES: u64 = 10;

/// Claim the generation a new derived object of the segment takes: one above
/// `floor` and above the segment's high-water mark, which is raised to it
/// before the caller writes anything. A generation a reader may still hold is
/// then never reused, even after `gc` collected the highest object
/// and a lost manifest stopped naming it. The three index kinds
/// share the mark, so one kind's numbers can skip.
///
/// # Errors
///
/// [`ItzmaError::CorruptIndex`] for a mark that is not a decimal u32 (reading
/// it low could reuse a number), [`ItzmaError::ManifestContention`] after 32
/// lost races on the mark, and [`ItzmaError::Storage`] for anything else.
pub(crate) async fn claim_generation(
    store: &dyn ObjectStore,
    namespace: &str,
    id: &str,
    floor: u32,
) -> Result<u32> {
    let path = generation_mark_path(namespace, id);
    for _ in 0..crate::manifest::MAX_UPDATE_ATTEMPTS {
        let (mark, version) = read_generation_mark(store, &path).await?;
        let generation = floor.max(mark).saturating_add(1);
        let mode = match version {
            Some(version) => PutMode::Update(version),
            None => PutMode::Create,
        };
        let body = Bytes::from(generation.to_string());
        match store
            .put_opts(&path, body.into(), PutOptions::from(mode))
            .await
        {
            Ok(_) => return Ok(generation),
            Err(
                object_store::Error::Precondition { .. }
                | object_store::Error::AlreadyExists { .. },
            ) => continue,
            Err(e) => return Err(e.into()),
        }
    }
    Err(ItzmaError::ManifestContention {
        attempts: crate::manifest::MAX_UPDATE_ATTEMPTS,
    })
}

/// The mark and the version to tie a raise to, or `(0, None)` when there is
/// no mark yet (a segment never indexed, or indexed before the mark existed,
/// whose objects the listing still covers).
async fn read_generation_mark(
    store: &dyn ObjectStore,
    path: &Path,
) -> Result<(u32, Option<UpdateVersion>)> {
    let result = match store.get(path).await {
        Ok(result) => result,
        Err(object_store::Error::NotFound { .. }) => return Ok((0, None)),
        Err(e) => return Err(e.into()),
    };
    let corrupt = |reason: &str| ItzmaError::CorruptIndex {
        path: path.to_string(),
        reason: format!("generation mark {reason}"),
    };
    if result.meta.size > MAX_MARK_BYTES {
        return Err(corrupt("is too long"));
    }
    let version = UpdateVersion {
        e_tag: result.meta.e_tag.clone(),
        version: result.meta.version.clone(),
    };
    let bytes = result.bytes().await?;
    let mark = std::str::from_utf8(&bytes)
        .ok()
        .filter(|text| text.bytes().all(|b| b.is_ascii_digit()))
        .and_then(|text| text.parse::<u32>().ok())
        .ok_or_else(|| corrupt("is not a decimal number"))?;
    Ok((mark, Some(version)))
}

/// Check the manifest invariant on its segment list: every id parses, the
/// first range starts at 1, each next range starts one past the previous end,
/// and the last range ends at `folded_through`. An empty list requires
/// `folded_through == 0`.
///
/// # Errors
///
/// [`ItzmaError::CorruptManifest`] describing the first violation.
pub fn validate_chain(ids: &[String], folded_through: u64) -> Result<()> {
    let corrupt = |reason: String| ItzmaError::CorruptManifest(reason);
    let mut next_lo = 1u64;
    for id in ids {
        let (lo, hi) = parse_segment_id(id)
            .ok_or_else(|| corrupt(format!("segment id {id:?} is malformed")))?;
        if lo != next_lo {
            return Err(corrupt(format!(
                "segment {id} starts at {lo}, expected {next_lo}"
            )));
        }
        next_lo = hi
            .checked_add(1)
            .ok_or_else(|| corrupt(format!("segment {id} ends at the largest seq")))?;
    }
    let covered = next_lo - 1;
    if covered != folded_through {
        return Err(corrupt(format!(
            "segments cover through {covered} but wal_folded_through is {folded_through}"
        )));
    }
    Ok(())
}

/// Encode a layer as a segment object of the version a new namespace writes
/// ([`DEFAULT_FORMAT`]) for vectors of length `dim`. See [`encode_as`].
pub fn encode(layer: &Layer, dim: usize) -> Result<Bytes> {
    encode_as(layer, dim, DEFAULT_FORMAT)
}

/// Encode a layer as a segment object of `version` (1, 2 or 3) for vectors of
/// length `dim`, with attributes stored as JSON. At version 3 that is the
/// layout of a free-form namespace; a typed namespace's segments are what
/// [`encode_with_schema`] writes.
///
/// The result depends only on the layer's contents, not on the order things
/// were inserted. The caller decides whether tombstones belong: a base segment
/// (range starting at 1) should have had them dropped first.
///
/// # Errors
///
/// - [`ItzmaError::UnsupportedFormat`] for any other `version`.
/// - [`ItzmaError::DimensionMismatch`] if any document's vector is not `dim`
///   long, so a hand-edited WAL cannot produce a corrupt segment.
/// - [`ItzmaError::Serde`] / [`ItzmaError::CorruptSegment`] if an attribute
///   cannot be serialized or a field is longer than 4 GiB.
pub fn encode_as(layer: &Layer, dim: usize, version: u32) -> Result<Bytes> {
    Ok(encode_parts(layer, dim, version, &AttrSchema::free_form())?.into())
}

/// Encode a layer as the segment object `compact` writes at `version` for a
/// namespace with `schema`: at version 3 a typed namespace's attributes become
/// columns and zone maps. Versions 1 and 2 ignore the attribute
/// schema, so this equals [`encode_as`] there.
///
/// # Errors
///
/// As [`encode_as`], plus [`ItzmaError::CorruptSegment`] at version 3 for a
/// document whose attributes the schema does not allow, which a strict write
/// never stores.
pub fn encode_with_schema(layer: &Layer, schema: &Schema, version: u32) -> Result<Bytes> {
    Ok(encode_parts(layer, schema.dim, version, &schema.attrs)?.into())
}

/// [`encode_as`] without joining the chunks: what compaction writes.
pub(crate) fn encode_parts(
    layer: &Layer,
    dim: usize,
    version: u32,
    attrs: &AttrSchema,
) -> Result<PutPayload> {
    let mut writer = SegmentWriter::new(dim, version, attrs)?;
    let docs = layer.sorted_docs();
    let columnar = version >= 3 && !attrs.is_free_form();
    if columnar {
        // Column and zone blocks a block per chunk, in parallel on a pool
        // thread, handed to the writer in order.
        let fields = columns::fields_of(attrs);
        let blocks = cpu::chunked(&docs, v2::BLOCK_ROWS, |chunk| {
            let rows: Vec<Option<&serde_json::Value>> =
                chunk.iter().map(|doc| doc.attrs.as_ref()).collect();
            encode_attr_block(&fields, &rows)
        });
        writer.attr_blocks(blocks.into_iter().collect::<Result<_>>()?)?;
    }
    // JSON per document, a block per chunk in parallel on a pool thread,
    // unless the writer takes the values themselves; the chunks
    // come back in order, so the bytes are the same either way.
    let mut json: Vec<Vec<u8>> = Vec::new();
    if !columnar {
        json.reserve(docs.len());
        for chunk in cpu::chunked(&docs, v2::BLOCK_ROWS, |chunk| {
            chunk
                .iter()
                .map(|doc| attr_json(doc.attrs.as_ref()))
                .collect::<Result<Vec<_>>>()
        }) {
            json.extend(chunk?);
        }
    }
    let tombstones = layer.sorted_tombstones();
    let mut dead = tombstones.iter().peekable();
    for (i, doc) in docs.iter().enumerate() {
        while let Some(id) = dead.next_if(|id| **id < doc.id.as_str()) {
            writer.tombstone(id)?;
        }
        let attrs = if columnar {
            AttrIn::Encoded
        } else {
            AttrIn::Json(&json[i])
        };
        writer.doc(&doc.id, &doc.vector, attrs)?;
    }
    for id in dead {
        writer.tombstone(id)?;
    }
    writer.finish()
}

/// The vector length as the u32 the formats store, after checking that every
/// document has that length.
#[cfg(test)]
fn check_dim(docs: &[&Document], dim: usize) -> Result<u32> {
    if let Some(bad) = docs.iter().find(|d| d.vector.len() != dim) {
        return Err(ItzmaError::DimensionMismatch {
            expected: dim,
            got: bad.vector.len(),
        });
    }
    u32::try_from(dim).map_err(|_| too_long())
}

fn put_bytes(buf: &mut Vec<u8>, bytes: &[u8]) -> Result<()> {
    let len = u32::try_from(bytes.len()).map_err(|_| too_long())?;
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(bytes);
    Ok(())
}

fn too_long() -> ItzmaError {
    ItzmaError::CorruptSegment {
        path: "(encoding)".into(),
        reason: "a field is longer than 4 GiB".into(),
    }
}

/// The `reason` of the error for a segment the store does not have. Callers
/// recognise it through [`is_missing`], never by text.
const MISSING: &str = "missing from the store";

/// The error for a segment the store does not have: one the manifest lists, or
/// one that was there a moment ago.
pub(crate) fn missing(path: &Path) -> ItzmaError {
    ItzmaError::CorruptSegment {
        path: path.to_string(),
        reason: MISSING.into(),
    }
}

/// Whether `err` is what [`missing`] builds. Garbage collection deletes
/// segments, so a missing one may only mean the caller read a stale manifest.
/// Callers reload the manifest before treating it as damage.
pub(crate) fn is_missing(err: &ItzmaError) -> bool {
    matches!(err, ItzmaError::CorruptSegment { reason, .. } if reason == MISSING)
}

/// Decode a segment object (version 2 or 3) for a namespace whose vectors
/// have length `dim`. `path` is only used to label errors.
///
/// # Errors
///
/// - [`ItzmaError::UnsupportedFormat`] for a `format_version` other than 2 or 3.
/// - [`ItzmaError::CorruptSegment`] for anything else wrong with the bytes:
///   truncation, bad magic, a dimension different from `dim`, counts larger
///   than the object can hold, ids or tombstones out of order, an id that is
///   both a document and a tombstone, invalid UTF-8 or JSON, trailing bytes,
///   and for version 2 a directory, block, offset or Bloom filter that does
///   not match the content.
pub fn decode(path: &str, bytes: &[u8], dim: usize) -> Result<Layer> {
    let mut r = Reader::new(path, bytes);
    if r.take(4)? != MAGIC {
        return Err(r.corrupt("bad magic"));
    }
    match r.u32()? {
        version @ (2 | 3) => v2::decode(r, version, dim),
        version => Err(ItzmaError::UnsupportedFormat { version }),
    }
}

/// Whether two id-sorted lists share an entry. One pass over both.
fn overlaps(docs: &[Document], tombstones: &[String]) -> bool {
    let (mut i, mut j) = (0, 0);
    while i < docs.len() && j < tombstones.len() {
        match docs[i].id.cmp(&tombstones[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => return true,
        }
    }
    false
}

/// A cursor over the object's bytes. Every read is bounds-checked and
/// failures carry the object's key.
struct Reader<'a> {
    path: &'a str,
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(path: &'a str, buf: &'a [u8]) -> Self {
        Reader { path, buf, pos: 0 }
    }

    fn corrupt(&self, reason: impl Into<String>) -> ItzmaError {
        ItzmaError::CorruptSegment {
            path: self.path.to_string(),
            reason: reason.into(),
        }
    }

    fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let out = self
            .buf
            .get(self.pos..)
            .and_then(|rest| rest.get(..n))
            .ok_or_else(|| self.corrupt("object is truncated"))?;
        self.pos += n;
        Ok(out)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let bytes = self.take(N)?;
        <[u8; N]>::try_from(bytes).map_err(|_| self.corrupt("object is truncated"))
    }

    fn u8(&mut self) -> Result<u8> {
        let [byte] = self.array::<1>()?;
        Ok(byte)
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.array()?))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn string(&mut self, what: &str) -> Result<String> {
        let len = self.u32()? as usize;
        let raw = self.take(len)?;
        std::str::from_utf8(raw)
            .map(str::to_owned)
            .map_err(|_| self.corrupt(format!("{what} is not valid UTF-8")))
    }
}

/// A dense `rows x dim` matrix of `f32`, row-major, one allocation for the
/// whole thing. `RangedSegment::vectors_of`
/// returns this instead of `Vec<Vec<f32>>` (one heap allocation per row) so
/// a rerank walks contiguous memory and the distance kernels see
/// one slice per row of the same buffer. A scan hands one out per batch.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub(crate) rows: usize,
    pub(crate) dim: usize,
    pub(crate) values: Vec<f32>,
}

impl Matrix {
    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Values per row.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Every value, row after row.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Row `i`'s `dim` values.
    ///
    /// # Panics
    ///
    /// If `i >= self.rows()`.
    pub fn row(&self, i: usize) -> &[f32] {
        let start = i * self.dim;
        &self.values[start..start + self.dim]
    }

    /// The values, row after row, without a copy.
    pub fn into_values(self) -> Vec<f32> {
        self.values
    }
}

/// Fetch one segment object whole, charging its bytes to `budget` before
/// they are read. Above `policy`'s `split_above`, the object is
/// fetched as parallel planner parts instead of one connection.
pub(crate) async fn fetch_bytes(
    store: &dyn ObjectStore,
    path: &Path,
    budget: &ReadBudget,
    policy: &FetchPolicy,
) -> Result<Bytes> {
    match crate::fetch::whole_object(store, budget, policy, path).await {
        Ok(bytes) => Ok(bytes),
        Err(ItzmaError::Storage(object_store::Error::NotFound { .. })) => Err(missing(path)),
        Err(e) => Err(e),
    }
}

/// Fetch one segment and decode it on `lane`.
pub(crate) async fn fetch(
    store: &dyn ObjectStore,
    namespace: &str,
    id: &str,
    dim: usize,
    budget: &ReadBudget,
    policy: &FetchPolicy,
    lane: Lane,
) -> Result<Layer> {
    let path = segment_path(namespace, id);
    let bytes = fetch_bytes(store, &path, budget, policy).await?;
    decode_on(lane, path, bytes, dim).await
}

/// [`decode`] on `lane`'s compute pool when the object is big enough to be
/// worth a hand-off.
pub(crate) async fn decode_on(lane: Lane, path: Path, bytes: Bytes, dim: usize) -> Result<Layer> {
    cpu::run(lane, Cost::Bytes(bytes.len()), move || {
        decode(path.as_ref(), &bytes, dim)
    })
    .await
}

/// [`encode_parts`] on `lane`'s compute pool when the layer is big enough to
/// be worth a hand-off. The layer comes back with the parts, so it
/// is dropped on the caller's thread, not the pool's.
pub(crate) async fn encode_parts_on(
    lane: Lane,
    layer: Layer,
    dim: usize,
    version: u32,
    attrs: AttrSchema,
) -> Result<(Layer, PutPayload)> {
    let cost = Cost::Bytes(layer.doc_count().saturating_mul(dim * 4 + 64));
    cpu::run(lane, cost, move || {
        let data = encode_parts(&layer, dim, version, &attrs)?;
        Ok((layer, data))
    })
    .await
}

/// What the first bytes of a segment object say, from a 20-byte ranged read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Peek {
    pub(crate) version: u32,
    pub(crate) dim: u32,
    pub(crate) doc_count: u64,
}

/// Magic, version, dim, doc count: the part of the header both versions share.
const PEEK_LEN: u64 = 4 + 4 + 4 + 8;

/// Read the version, dimension and document count of a segment without its
/// body, charged to `budget`.
///
/// # Errors
///
/// [`ItzmaError::CorruptSegment`] if the object is missing (see [`is_missing`])
/// or does not start with a segment header, [`ItzmaError::UnsupportedFormat`]
/// for a version other than 1 or 2.
pub(crate) async fn peek(
    store: &dyn ObjectStore,
    path: &Path,
    budget: &ReadBudget,
) -> Result<Peek> {
    budget.charge(path.as_ref(), PEEK_LEN)?;
    let bytes = match store.get_range(path, 0..PEEK_LEN).await {
        Ok(bytes) => bytes,
        Err(object_store::Error::NotFound { .. }) => return Err(missing(path)),
        Err(e) => return Err(e.into()),
    };
    let mut r = Reader::new(path.as_ref(), &bytes);
    if r.take(4)? != MAGIC {
        return Err(r.corrupt("bad magic"));
    }
    let version = r.u32()?;
    if !(2..=FORMAT_VERSION).contains(&version) {
        return Err(ItzmaError::UnsupportedFormat { version });
    }
    Ok(Peek {
        version,
        dim: r.u32()?,
        doc_count: r.u64()?,
    })
}

/// Fetch and decode the listed segments, up to 16 requests in flight, and
/// overlay them oldest to newest into one layer. Each segment is charged to
/// `budget` before its body is read.
///
/// A listed segment that the store does not have is reported as
/// [`ItzmaError::CorruptSegment`] (see [`is_missing`]). It means damage only
/// if the manifest is current: garbage collection may have deleted the segment
/// after the caller loaded an older manifest.
pub(crate) async fn read_all(
    store: &dyn ObjectStore,
    namespace: &str,
    ids: &[String],
    dim: usize,
    budget: &ReadBudget,
    policy: &FetchPolicy,
    lane: Lane,
) -> Result<Layer> {
    // Owned ids: a closure over `&String` is inferred as higher-ranked, and
    // rustc then cannot prove the stream `Send`, so `Namespace::query` and
    // `compact` could not be moved into `tokio::spawn`.
    stream::iter(ids.iter().cloned())
        .map(|id| async move { fetch(store, namespace, &id, dim, budget, policy, lane).await })
        .buffered(FETCH_CONCURRENCY)
        .try_fold(Layer::new(), |mut acc, segment| async move {
            acc.overlay(segment);
            Ok::<_, ItzmaError>(acc)
        })
        .await
}

/// Fetch a segment whole and decode it as a view, on `lane`'s pool.
async fn load_flat(
    store: &dyn ObjectStore,
    path: Path,
    dim: usize,
    budget: &ReadBudget,
    policy: &FetchPolicy,
    lane: Lane,
) -> Result<(std::sync::Arc<SegmentView>, u64)> {
    let bytes = fetch_bytes(store, &path, budget, policy).await?;
    let view = cpu::run(lane, Cost::Bytes(bytes.len()), move || {
        SegmentView::decode(path.as_ref(), bytes, dim)
    })
    .await?;
    // Weighed by what it holds, which is not the object's size.
    let weight = view.weight();
    Ok((std::sync::Arc::new(view), weight))
}

/// The segments `ids` of `namespace`, in order, for a flat scan, through
/// `decoded` when the handle has one.
///
/// # Errors
///
/// [`missing`] for a segment that is gone, [`ItzmaError::CorruptSegment`],
/// and [`ItzmaError::ReadBudgetExceeded`] for a miss over the budget.
#[allow(clippy::too_many_arguments)] // the handle's read settings
pub(crate) async fn read_flat(
    store: &dyn ObjectStore,
    namespace: &str,
    ids: &[String],
    dim: usize,
    budget: &ReadBudget,
    policy: &FetchPolicy,
    lane: Lane,
    decoded: Option<&DecodedRef>,
) -> Result<Vec<std::sync::Arc<SegmentView>>> {
    // Owned ids: a closure over `&String` is inferred as higher-ranked, and
    // rustc then cannot prove the stream `Send` (see `read_all_uncached`).
    stream::iter(ids.iter().cloned())
        .map(|id| async move {
            let path = segment_path(namespace, &id);
            let Some(decoded) = decoded else {
                return Ok(load_flat(store, path, dim, budget, policy, lane).await?.0);
            };
            let view = decoded
                .get_or_load(&path, What::View, || {
                    load_flat(store, path.clone(), dim, budget, policy, lane)
                })
                .await?;
            Ok(std::sync::Arc::clone(&*view))
        })
        .buffered(FETCH_CONCURRENCY)
        .try_collect()
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::Lane;
    use crate::document::{Document, WalOp};
    use crate::error::ItzmaError;

    #[tokio::test]
    async fn read_flat_gives_views_and_refuses_version_1() {
        let store = object_store::memory::InMemory::new();
        let mut layer = Layer::new();
        for i in 0..3 {
            layer.apply(WalOp::Upsert(Document {
                id: format!("d{i}"),
                vector: vec![i as f32; 2],
                attrs: None,
            }));
        }
        let ids = [segment_id(1, 1)];
        let bytes = encode_as(&layer, 2, 2).unwrap();
        store
            .put(&segment_path("ns", &ids[0]), bytes.clone().into())
            .await
            .unwrap();
        let budget = ReadBudget::new(1 << 20);
        let policy = FetchPolicy::local();
        let got = read_flat(&store, "ns", &ids, 2, &budget, &policy, Lane::Query, None)
            .await
            .unwrap();
        assert_eq!(got[0].rows(), 3);
        assert_eq!(got[0].vector(2), &[2.0, 2.0]);

        // The same object labelled version 1, which this build does not read.
        let mut v1 = bytes.to_vec();
        v1[4..8].copy_from_slice(&1u32.to_le_bytes());
        store
            .put(&segment_path("ns", &ids[0]), v1.into())
            .await
            .unwrap();
        let err = read_flat(&store, "ns", &ids, 2, &budget, &policy, Lane::Query, None)
            .await
            .err()
            .unwrap();
        assert!(
            matches!(err, ItzmaError::UnsupportedFormat { version: 1 }),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn a_missing_segment_is_missing_for_read_flat() {
        let store = object_store::memory::InMemory::new();
        let budget = ReadBudget::new(1 << 20);
        let policy = FetchPolicy::local();
        let ids = [segment_id(1, 1)];
        let err = read_flat(&store, "ns", &ids, 2, &budget, &policy, Lane::Query, None)
            .await
            .err()
            .unwrap();
        assert!(is_missing(&err), "{err:?}");
    }

    /// `n` documents of `dim` values, every third with attributes, and ten
    /// tombstones: past the compute pool's inline threshold at `n = 20_000`.
    fn big_layer(n: usize, dim: usize) -> Layer {
        let mut layer = Layer::new();
        for i in 0..n {
            layer.apply(WalOp::Upsert(Document {
                id: format!("d{i:06}"),
                vector: (0..dim).map(|d| ((i * 31 + d * 7) % 101) as f32).collect(),
                attrs: (i % 3 == 0).then(|| serde_json::json!({ "n": i, "tag": "x" })),
            }));
        }
        for i in 0..10 {
            layer.apply(WalOp::Delete(format!("gone{i}")));
        }
        layer
    }

    #[tokio::test]
    async fn decode_on_a_lane_equals_decode() {
        let layer = big_layer(20_000, 32);
        for version in [2, 3] {
            let bytes = encode_as(&layer, 32, version).unwrap();
            let direct = decode("p", &bytes, 32).unwrap();
            let pooled = decode_on(Lane::Query, Path::from("p"), bytes, 32)
                .await
                .unwrap();
            assert_eq!(pooled, direct, "version {version}");
        }
    }

    #[tokio::test]
    async fn encode_on_a_lane_gives_the_same_bytes_and_the_layer_back() {
        let layer = big_layer(20_000, 32);
        for version in [2, 3] {
            let direct = encode_as(&layer, 32, version).unwrap();
            let (back, data) = encode_parts_on(
                Lane::Maintenance,
                layer.clone(),
                32,
                version,
                AttrSchema::free_form(),
            )
            .await
            .unwrap();
            assert_eq!(Bytes::from(data), direct, "version {version}");
            assert_eq!(back, layer);
        }
    }

    #[test]
    fn matrix_row_slices_out_the_right_values_in_order() {
        let m = Matrix {
            rows: 3,
            dim: 2,
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        assert_eq!(m.row(0), &[1.0, 2.0]);
        assert_eq!(m.row(1), &[3.0, 4.0]);
        assert_eq!(m.row(2), &[5.0, 6.0]);
    }

    fn up(id: &str, v: &[f32], attrs: Option<serde_json::Value>) -> WalOp {
        WalOp::Upsert(Document {
            id: id.into(),
            vector: v.to_vec(),
            attrs,
        })
    }

    fn layer(ops: Vec<WalOp>) -> Layer {
        let mut l = Layer::new();
        for op in ops {
            l.apply(op);
        }
        l
    }

    fn sample() -> Layer {
        layer(vec![
            up(
                "b",
                &[1.0, 2.0],
                Some(serde_json::json!({"k": [1, 2], "s": "x"})),
            ),
            up("a", &[-0.5, 0.25], None),
            up(
                "ü-日本",
                &[3.0, 4.0],
                Some(serde_json::json!("plain string")),
            ),
            WalOp::Delete("y".into()),
            WalOp::Delete("x".into()),
        ])
    }

    /// A version 2 object, the format the tests below decode.
    fn v2(layer: &Layer, dim: usize) -> Result<Bytes> {
        encode_as(layer, dim, 2)
    }

    fn corrupt(err: ItzmaError) -> String {
        match err {
            ItzmaError::CorruptSegment { reason, .. } => reason,
            other => panic!("expected CorruptSegment, got {other:?}"),
        }
    }

    #[test]
    fn roundtrip_preserves_docs_attrs_and_tombstones() {
        let original = sample();
        let bytes = v2(&original, 2).unwrap();
        assert_eq!(decode("p", &bytes, 2).unwrap(), original);
    }

    #[test]
    fn json_null_attrs_survive_the_roundtrip() {
        let original = layer(vec![up("a", &[1.0], Some(serde_json::Value::Null))]);
        let bytes = v2(&original, 1).unwrap();
        let back = decode("p", &bytes, 1).unwrap();
        assert_eq!(back.sorted_docs()[0].attrs, Some(serde_json::Value::Null));
    }

    #[test]
    fn bytes_do_not_depend_on_insertion_order() {
        let one = layer(vec![
            up("a", &[1.0], None),
            up("b", &[2.0], None),
            WalOp::Delete("z".into()),
        ]);
        let two = layer(vec![
            WalOp::Delete("z".into()),
            up("b", &[2.0], None),
            up("a", &[1.0], None),
        ]);
        assert_eq!(v2(&one, 1).unwrap(), v2(&two, 1).unwrap());
    }

    #[test]
    fn encode_rejects_a_vector_of_the_wrong_length() {
        let l = layer(vec![up("a", &[1.0, 2.0, 3.0], None)]);
        let err = v2(&l, 2).unwrap_err();
        assert!(matches!(
            err,
            ItzmaError::DimensionMismatch {
                expected: 2,
                got: 3
            }
        ));
    }

    #[test]
    fn every_truncation_is_an_error_and_none_panics() {
        let bytes = v2(&sample(), 2).unwrap();
        for cut in 0..bytes.len() {
            assert!(decode("p", &bytes[..cut], 2).is_err(), "cut at {cut}");
        }
    }

    #[test]
    fn decode_rejects_bad_magic() {
        let mut bytes = v2(&sample(), 2).unwrap().to_vec();
        bytes[0] = b'X';
        assert!(corrupt(decode("p", &bytes, 2).unwrap_err()).contains("magic"));
    }

    #[test]
    fn decode_reports_an_unknown_version() {
        // 4: the first version this build does not read.
        for version in [0, 4, u32::MAX] {
            let mut bytes = v2(&sample(), 2).unwrap().to_vec();
            bytes[4..8].copy_from_slice(&version.to_le_bytes());
            assert!(
                matches!(
                    decode("p", &bytes, 2).unwrap_err(),
                    ItzmaError::UnsupportedFormat { version: v } if v == version
                ),
                "version {version}"
            );
        }
    }

    #[test]
    fn decode_rejects_a_dimension_that_differs_from_the_schema() {
        let bytes = v2(&sample(), 2).unwrap();
        assert!(corrupt(decode("p", &bytes, 3).unwrap_err()).contains("dimension"));
    }

    #[test]
    fn encode_writes_the_default_version() {
        let bytes = encode(&sample(), 2).unwrap();
        assert_eq!(bytes[4..8], DEFAULT_FORMAT.to_le_bytes());
        assert_eq!((DEFAULT_FORMAT, FORMAT_VERSION), (2, 3));
    }

    #[test]
    fn encode_as_writes_the_version_asked_for() {
        for version in [2, 3] {
            let bytes = encode_as(&sample(), 2, version).unwrap();
            assert_eq!(bytes[4..8], version.to_le_bytes());
        }
    }

    #[test]
    fn encode_as_refuses_a_version_it_does_not_write() {
        for version in [0, 4, u32::MAX] {
            let err = encode_as(&sample(), 2, version).unwrap_err();
            assert!(
                matches!(err, ItzmaError::UnsupportedFormat { version: v } if v == version),
                "{err:?}"
            );
        }
    }

    #[test]
    fn a_short_object_is_corrupt_not_unsupported() {
        for cut in 0..8 {
            let bytes = encode(&sample(), 2).unwrap();
            let err = decode("p", &bytes[..cut], 2).unwrap_err();
            assert!(matches!(err, ItzmaError::CorruptSegment { .. }), "{cut}");
        }
    }

    #[test]
    fn segment_ids_are_zero_padded_ranges() {
        assert_eq!(
            segment_id(1, 10),
            "00000000000000000001-00000000000000000010"
        );
        assert_eq!(
            segment_path("docs", &segment_id(1, 10)).as_ref(),
            "docs/segments/00000000000000000001-00000000000000000010/data.bin"
        );
    }

    #[test]
    fn parse_segment_id_accepts_only_the_canonical_form() {
        assert_eq!(parse_segment_id(&segment_id(3, 9)), Some((3, 9)));
        assert_eq!(parse_segment_id(&segment_id(5, 5)), Some((5, 5)));
        for bad in [
            "",
            "1-2",
            "00000000000000000001",
            "+0000000000000000001-00000000000000000002",
            "00000000000000000000-00000000000000000002",
            "00000000000000000003-00000000000000000002",
            "00000000000000000001-00000000000000000002-x",
            "0000000000000000000a-00000000000000000002",
        ] {
            assert_eq!(parse_segment_id(bad), None, "{bad:?}");
        }
    }

    fn ids(ranges: &[(u64, u64)]) -> Vec<String> {
        ranges.iter().map(|(lo, hi)| segment_id(*lo, *hi)).collect()
    }

    fn manifest_reason(err: ItzmaError) -> String {
        match err {
            ItzmaError::CorruptManifest(reason) => reason,
            other => panic!("expected CorruptManifest, got {other:?}"),
        }
    }

    #[test]
    fn validate_chain_accepts_a_contiguous_run() {
        assert!(validate_chain(&[], 0).is_ok());
        assert!(validate_chain(&ids(&[(1, 3)]), 3).is_ok());
        assert!(validate_chain(&ids(&[(1, 3), (4, 4), (5, 9)]), 9).is_ok());
    }

    #[test]
    fn validate_chain_rejects_every_broken_shape() {
        let cases = [
            (ids(&[(1, 3), (5, 6)]), 6, "starts at 5"), // gap
            (ids(&[(1, 3), (3, 4)]), 4, "starts at 3"), // overlap
            (ids(&[(2, 3)]), 3, "starts at 2"),         // does not begin at 1
            (ids(&[(1, 3)]), 4, "folded"),              // last hi differs from folded
            (vec![], 5, "folded"),                      // folded with no segments
            (vec!["nonsense".to_string()], 0, "malformed"),
        ];
        for (list, folded, want) in cases {
            let reason = manifest_reason(validate_chain(&list, folded).unwrap_err());
            assert!(reason.contains(want), "{list:?} folded {folded}: {reason}");
        }
    }

    #[test]
    fn validate_chain_does_not_overflow_on_the_largest_range() {
        assert!(validate_chain(&ids(&[(1, u64::MAX)]), u64::MAX).is_err());
    }

    async fn peek_of(bytes: &[u8]) -> (Result<Peek>, ReadBudget) {
        use object_store::memory::InMemory;
        let store = InMemory::new();
        let path = Path::from("docs/segments/x/data.bin");
        store
            .put(&path, Bytes::copy_from_slice(bytes).into())
            .await
            .unwrap();
        let budget = ReadBudget::new(1_000);
        (peek(&store, &path, &budget).await, budget)
    }

    #[tokio::test]
    async fn a_peek_reads_the_version_dimension_and_count_of_either_version() {
        let l = sample();
        let docs = l.doc_count() as u64;
        for version in [2, 3] {
            let bytes = encode_as(&l, 2, version).unwrap();
            let (got, _) = peek_of(&bytes).await;
            assert_eq!(
                got.unwrap(),
                Peek {
                    version,
                    dim: 2,
                    doc_count: docs
                }
            );
        }
    }

    #[tokio::test]
    async fn a_peek_is_charged_its_twenty_bytes_and_no_more() {
        use object_store::memory::InMemory;
        let bytes = encode(&sample(), 2).unwrap();
        let store = InMemory::new();
        let path = Path::from("docs/segments/x/data.bin");
        store.put(&path, bytes.into()).await.unwrap();
        peek(&store, &path, &ReadBudget::new(20)).await.unwrap();
        let err = peek(&store, &path, &ReadBudget::new(19)).await.unwrap_err();
        assert!(
            matches!(err, ItzmaError::ReadBudgetExceeded { .. }),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn a_peek_refuses_what_is_not_a_segment() {
        let good = encode(&sample(), 2).unwrap().to_vec();
        let mut bad_magic = good.clone();
        bad_magic[0] = b'X';
        assert!(corrupt(peek_of(&bad_magic).await.0.unwrap_err()).contains("magic"));

        let mut future = good.clone();
        future[4..8].copy_from_slice(&4u32.to_le_bytes());
        assert!(matches!(
            peek_of(&future).await.0,
            Err(ItzmaError::UnsupportedFormat { version: 4 })
        ));
        let mut zero = good.clone();
        zero[4..8].copy_from_slice(&0u32.to_le_bytes());
        assert!(matches!(
            peek_of(&zero).await.0,
            Err(ItzmaError::UnsupportedFormat { version: 0 })
        ));

        for cut in 0..20 {
            assert!(peek_of(&good[..cut]).await.0.is_err(), "cut at {cut}");
        }
    }

    #[tokio::test]
    async fn a_peek_of_a_missing_object_is_missing() {
        use object_store::memory::InMemory;
        let path = Path::from("docs/segments/x/data.bin");
        let err = peek(&InMemory::new(), &path, &ReadBudget::new(100))
            .await
            .unwrap_err();
        assert!(is_missing(&err), "{err:?}");
    }
}
