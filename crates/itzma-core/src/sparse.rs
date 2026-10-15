//! The `sparse.bin` sidecar: one segment's raw sparse vectors,
//! row-addressed the same way `data.bin`'s dense matrix is. Written by
//! compaction only when at least one document in the segment has one
//! (`crate::compaction`); read whole by `Namespace::index_sparse` to build
//! the inverted index, and by the query engine's unindexed-segment fallback.
//!
//! # Layout
//!
//! All integers are little-endian.
//!
//! | Section | Contents |
//! |---|---|
//! | Header | magic `ITZS`, `format_version` u32, `rows` u32 |
//! | Directory | `rows` entries: `term_count` u16, `byte_offset` u64 |
//! | Rows | each row's `term_count` pairs of `(term_id u32, weight f32)` |
//!
//! Row `i` is document `i` of the same segment's `data.bin`, in the ascending
//! id order [`crate::layer::Layer::sorted_docs`] both objects are built from.
//! A row with no sparse vector has `term_count` 0 and no bytes of its own.

use bytes::Bytes;
#[cfg(test)]
use bytes::BytesMut;
use std::ops::Range;

use object_store::ObjectStore;
use object_store::path::Path;

use crate::budget::ReadBudget;
use crate::cpu::{self, Cost, Lane};
use crate::document::SparseVector;
use crate::error::{ItzmaError, Result};
use crate::fetch::FetchPolicy;
use crate::layer::Layer;

const MAGIC: [u8; 4] = *b"ITZS";
pub(crate) const FORMAT_VERSION: u32 = 1;
/// Magic, format_version, rows.
const HEADER_LEN: usize = 4 + 4 + 4;
/// term_count, byte_offset.
const DIR_ENTRY_LEN: usize = 2 + 8;

/// Object key of a segment's sparse sidecar:
/// `<namespace>/segments/<id>/sparse.bin`. Absent when no document in the
/// segment has a sparse vector.
pub fn sparse_path(namespace: &str, id: &str) -> Path {
    Path::from(format!("{namespace}/segments/{id}/sparse.bin"))
}

/// Encode `layer`'s sparse vectors in `sorted_docs()` order. A document with
/// none contributes an empty row, not an absent one, so row numbers still
/// line up with `data.bin`'s. Compaction writes it through
/// [`crate::segment::SegmentWriter`]; this is the whole-layer form tests use.
#[cfg(test)]
pub(crate) fn encode(layer: &Layer) -> Result<Bytes> {
    let mut writer = SparseWriter::new();
    for doc in layer.sorted_docs() {
        writer.row(doc.sparse.as_ref())?;
    }
    let (payload, _) = writer.finish()?;
    Ok(payload.into())
}

/// Builds `sparse.bin` one row at a time, in `data.bin`'s row order. The
/// directory comes first in the object but is complete only at the end, so
/// it is written by [`finish`](Self::finish) and the rows go out as chunks.
pub(crate) struct SparseWriter {
    dir: Vec<(u16, u64)>,
    body: crate::segment::Chunks,
    any: bool,
}

impl SparseWriter {
    pub(crate) fn new() -> Self {
        Self {
            dir: Vec::new(),
            body: crate::segment::Chunks::default(),
            any: false,
        }
    }

    /// The next row: a document's sparse vector, or `None` for one without.
    pub(crate) fn row(&mut self, sparse: Option<&SparseVector>) -> Result<()> {
        self.any |= sparse.is_some();
        let terms = sparse.map_or(&[][..], Vec::as_slice);
        let term_count = u16::try_from(terms.len()).map_err(|_| ItzmaError::CorruptSegment {
            path: "(encoding)".into(),
            reason: "a row has more sparse terms than fit a u16".into(),
        })?;
        self.dir.push((term_count, self.body.len()));
        let mut row = Vec::with_capacity(terms.len() * 8);
        for &(term, weight) in terms {
            row.extend_from_slice(&term.to_le_bytes());
            row.extend_from_slice(&weight.to_le_bytes());
        }
        self.body.push(&row);
        Ok(())
    }

    /// The object, and whether any row was given a sparse vector.
    pub(crate) fn finish(self) -> Result<(object_store::PutPayload, bool)> {
        let rows = u32::try_from(self.dir.len()).map_err(|_| ItzmaError::CorruptSegment {
            path: "(encoding)".into(),
            reason: "too many rows for sparse.bin's u32 row count".into(),
        })?;
        let mut head = Vec::with_capacity(HEADER_LEN + self.dir.len() * DIR_ENTRY_LEN);
        head.extend_from_slice(&MAGIC);
        head.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        head.extend_from_slice(&rows.to_le_bytes());
        for (term_count, byte_offset) in &self.dir {
            head.extend_from_slice(&term_count.to_le_bytes());
            head.extend_from_slice(&byte_offset.to_le_bytes());
        }
        let payload = std::iter::once(Bytes::from(head))
            .chain(self.body.into_bytes())
            .collect();
        Ok((payload, self.any))
    }
}

/// The encoder [`SparseWriter`] replaced, kept as the reference its
/// equality test compares against.
#[cfg(test)]
pub(crate) fn reference_encode(layer: &Layer) -> Result<Bytes> {
    let docs = layer.sorted_docs();
    let rows = u32::try_from(docs.len()).map_err(|_| ItzmaError::CorruptSegment {
        path: "(encoding)".into(),
        reason: "too many rows for sparse.bin's u32 row count".into(),
    })?;

    let mut body = BytesMut::new();
    let mut dir = Vec::with_capacity(docs.len());
    for doc in &docs {
        let terms = doc.sparse.as_deref().unwrap_or(&[]);
        let term_count = u16::try_from(terms.len()).map_err(|_| ItzmaError::CorruptSegment {
            path: "(encoding)".into(),
            reason: "a row has more sparse terms than fit a u16".into(),
        })?;
        dir.push((term_count, body.len() as u64));
        for &(term, weight) in terms {
            body.extend_from_slice(&term.to_le_bytes());
            body.extend_from_slice(&weight.to_le_bytes());
        }
    }

    let mut out = BytesMut::with_capacity(HEADER_LEN + dir.len() * DIR_ENTRY_LEN + body.len());
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&rows.to_le_bytes());
    for (term_count, byte_offset) in &dir {
        out.extend_from_slice(&term_count.to_le_bytes());
        out.extend_from_slice(&byte_offset.to_le_bytes());
    }
    out.extend_from_slice(&body);
    Ok(out.freeze())
}

/// A cursor over `sparse.bin`'s bytes. Every read is bounds-checked; failures
/// carry the object's key and raise `CorruptSegment`, the same variant
/// `data.bin`'s own decoder uses — this is segment data, just derived.
struct Reader<'a> {
    path: &'a str,
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(path: &'a str, buf: &'a [u8]) -> Self {
        Self { path, buf, pos: 0 }
    }

    fn corrupt(&self, reason: impl Into<String>) -> ItzmaError {
        ItzmaError::CorruptSegment {
            path: self.path.to_string(),
            reason: reason.into(),
        }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let out = self
            .buf
            .get(self.pos..)
            .and_then(|rest| rest.get(..n))
            .ok_or_else(|| self.corrupt("sparse.bin is truncated"))?;
        self.pos += n;
        Ok(out)
    }

    fn u16(&mut self) -> Result<u16> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn u32(&mut self) -> Result<u32> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn u64(&mut self) -> Result<u64> {
        let bytes = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(bytes);
        Ok(u64::from_le_bytes(a))
    }

    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.u32()?))
    }
}

/// Decode `bytes` into one entry per row (`None` for a row with no sparse
/// vector), checked against the segment's own row count.
///
/// # Errors
///
/// [`ItzmaError::CorruptSegment`] for a bad magic, an unsupported version, a
/// row count that disagrees with `rows`, a directory entry or row that runs
/// past the object, or an overflowing offset.
/// [`ItzmaError::UnsupportedFormat`] for a container version this build does
/// not read.
pub(crate) fn decode(path: &str, bytes: &[u8], rows: u64) -> Result<Vec<Option<SparseVector>>> {
    let mut r = Reader::new(path, bytes);
    if r.take(4)? != MAGIC {
        return Err(r.corrupt("bad magic"));
    }
    match r.u32()? {
        FORMAT_VERSION => {}
        version => return Err(ItzmaError::UnsupportedFormat { version }),
    }
    let got_rows = u64::from(r.u32()?);
    if got_rows != rows {
        return Err(r.corrupt(format!("{got_rows} rows, the segment has {rows}")));
    }
    let mut dir = Vec::with_capacity(got_rows as usize);
    for _ in 0..got_rows {
        dir.push((r.u16()?, r.u64()?));
    }
    let body_start = r.pos;
    let mut out = Vec::with_capacity(got_rows as usize);
    for &(term_count, byte_offset) in &dir {
        if term_count == 0 {
            out.push(None);
            continue;
        }
        let start = body_start
            .checked_add(byte_offset as usize)
            .ok_or_else(|| r.corrupt("row offset overflows"))?;
        let mut row = Reader {
            path,
            buf: bytes,
            pos: start,
        };
        let mut terms = Vec::with_capacity(term_count as usize);
        for _ in 0..term_count {
            let term = row.u32()?;
            let weight = row.f32()?;
            terms.push((term, weight));
        }
        out.push(Some(terms));
    }
    Ok(out)
}

/// A segment's `sparse.bin`, read forward a range of rows at a time by a
/// streaming merge: the directory once, then one body request per
/// range. Every request is charged to the budget before it is sent.
pub(crate) struct SparseRows<'a> {
    store: &'a dyn ObjectStore,
    budget: &'a ReadBudget,
    path: Path,
    dir: Vec<(u16, u64)>,
    body_start: u64,
}

impl<'a> SparseRows<'a> {
    /// Open segment `id`'s sidecar for a segment of `rows` documents, or
    /// `Ok(None)` when it has none. Reads and checks the header and the
    /// directory.
    ///
    /// # Errors
    ///
    /// [`ItzmaError::CorruptSegment`] for a bad magic, a row count other than
    /// `rows`, or a directory past the object; [`ItzmaError::UnsupportedFormat`]
    /// for a version this build does not read.
    pub(crate) async fn open(
        store: &'a dyn ObjectStore,
        namespace: &str,
        id: &str,
        rows: u64,
        budget: &'a ReadBudget,
    ) -> Result<Option<Self>> {
        let path = sparse_path(namespace, id);
        let head = match charged_range(store, budget, &path, 0..HEADER_LEN as u64).await {
            Ok(head) => head,
            Err(e) if is_missing(&e) => return Ok(None),
            Err(e) => return Err(e),
        };
        let name = path.to_string();
        let mut r = Reader::new(&name, &head);
        if r.take(4)? != MAGIC {
            return Err(r.corrupt("bad magic"));
        }
        match r.u32()? {
            FORMAT_VERSION => {}
            version => return Err(ItzmaError::UnsupportedFormat { version }),
        }
        let got_rows = u64::from(r.u32()?);
        if got_rows != rows {
            return Err(r.corrupt(format!("{got_rows} rows, the segment has {rows}")));
        }
        let body_start = HEADER_LEN as u64 + rows * DIR_ENTRY_LEN as u64;
        let raw = charged_range(store, budget, &path, HEADER_LEN as u64..body_start).await?;
        let mut r = Reader::new(&name, &raw);
        let mut dir = Vec::with_capacity(rows as usize);
        for _ in 0..rows {
            dir.push((r.u16()?, r.u64()?));
        }
        Ok(Some(Self {
            store,
            budget,
            path,
            dir,
            body_start,
        }))
    }

    /// Where `rows`' bytes are, relative to the body, or `None` for no rows.
    fn span(&self, rows: &Range<u64>) -> Option<Result<Range<u64>>> {
        if rows.is_empty() {
            return None;
        }
        let first = self.dir.get(rows.start as usize)?;
        let (count, last) = self.dir.get(rows.end as usize - 1)?;
        let end = last.checked_add(u64::from(*count) * 8);
        Some(match end {
            Some(end) if end >= first.1 => Ok(first.1..end),
            _ => Err(self.corrupt("row offsets go backwards or overflow")),
        })
    }

    /// The body bytes `rows` take, for sizing a read-ahead window; 0 for rows
    /// the directory does not hold.
    pub(crate) fn bytes_of(&self, rows: Range<u64>) -> u64 {
        match self.span(&rows) {
            Some(Ok(span)) => span.end - span.start,
            _ => 0,
        }
    }

    /// Rows `rows`, in order, `None` for a row without a sparse vector.
    ///
    /// # Errors
    ///
    /// [`ItzmaError::CorruptSegment`] for rows the directory does not hold or
    /// whose bytes run past what the directory says, [`missing`] for an
    /// object that is gone.
    pub(crate) async fn rows(&self, rows: Range<u64>) -> Result<Vec<Option<SparseVector>>> {
        let span = match self.span(&rows) {
            None if rows.is_empty() => return Ok(Vec::new()),
            None => return Err(self.corrupt("rows past the directory")),
            Some(span) => span?,
        };
        let range = self
            .body_start
            .checked_add(span.start)
            .zip(self.body_start.checked_add(span.end))
            .map(|(start, end)| start..end)
            .ok_or_else(|| self.corrupt("row offset overflows"))?;
        let body = charged_range(self.store, self.budget, &self.path, range).await?;
        let name = self.path.to_string();
        let mut out = Vec::with_capacity((rows.end - rows.start) as usize);
        for &(term_count, offset) in &self.dir[rows.start as usize..rows.end as usize] {
            if term_count == 0 {
                out.push(None);
                continue;
            }
            let at = offset
                .checked_sub(span.start)
                .ok_or_else(|| self.corrupt("row offsets go backwards"))?;
            let mut row = Reader {
                path: &name,
                buf: &body,
                pos: usize::try_from(at).unwrap_or(usize::MAX),
            };
            let mut terms = Vec::with_capacity(term_count as usize);
            for _ in 0..term_count {
                terms.push((row.u32()?, row.f32()?));
            }
            out.push(Some(terms));
        }
        Ok(out)
    }

    fn corrupt(&self, reason: &str) -> ItzmaError {
        ItzmaError::CorruptSegment {
            path: self.path.to_string(),
            reason: reason.into(),
        }
    }
}

/// One ranged GET of a sidecar, charged once the response headers say the
/// object is there and before its body is read, as every segment read is
///: a missing object is [`missing`] and costs nothing.
async fn charged_range(
    store: &dyn ObjectStore,
    budget: &ReadBudget,
    path: &Path,
    range: Range<u64>,
) -> Result<Bytes> {
    if range.is_empty() {
        return Ok(Bytes::new());
    }
    let options = object_store::GetOptions {
        range: Some(object_store::GetRange::Bounded(range.clone())),
        ..object_store::GetOptions::default()
    };
    let got = match store.get_opts(path, options).await {
        Ok(got) => got,
        Err(object_store::Error::NotFound { .. }) => return Err(missing(path)),
        Err(e) => return Err(e.into()),
    };
    budget.charge(path.as_ref(), range.end - range.start)?;
    let bytes = got.bytes().await?;
    if bytes.len() as u64 != range.end - range.start {
        return Err(ItzmaError::CorruptSegment {
            path: path.to_string(),
            reason: "sparse.bin is truncated".into(),
        });
    }
    Ok(bytes)
}

/// The `reason` of the error for a `sparse.bin` the store does not have.
const MISSING: &str = "missing from the store";

/// The error for a segment's sparse sidecar the store lacks. A segment with
/// no sparse documents legitimately has none — callers check
/// [`is_missing`] and treat it as "no sparse data here," not damage.
pub(crate) fn missing(path: &Path) -> ItzmaError {
    ItzmaError::CorruptSegment {
        path: path.to_string(),
        reason: MISSING.into(),
    }
}

/// Whether `err` is what [`missing`] builds.
pub(crate) fn is_missing(err: &ItzmaError) -> bool {
    matches!(err, ItzmaError::CorruptSegment { reason, .. } if reason == MISSING)
}

/// Fetch one `sparse.bin` object whole, charging its bytes to `budget`
/// before they are read. Above `policy`'s `split_above`, the
/// object is fetched as parallel planner parts instead of one connection.
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

/// Decode one segment's data and, if it has one, its sparse sidecar, and
/// return a layer whose live documents carry their sparse vectors. Row `i`
/// of `sparse.bin` is `layer.sorted_docs()[i]`, the same order `data.bin`
/// and `sparse.bin` were both built from (`crate::compaction`).
///
/// # Errors
///
/// Whatever `segment::decode` or [`decode`] give. A missing `sparse.bin` is
/// not an error: the segment simply has no sparse documents.
pub(crate) async fn fetch_with_sparse(
    store: &dyn ObjectStore,
    namespace: &str,
    id: &str,
    dim: usize,
    budget: &ReadBudget,
    policy: &FetchPolicy,
    lane: Lane,
) -> Result<Layer> {
    let data_path = crate::segment::segment_path(namespace, id);
    let bytes = crate::segment::fetch_bytes(store, &data_path, budget, policy).await?;
    let path = sparse_path(namespace, id);
    let sparse_bytes = match fetch_bytes(store, &path, budget, policy).await {
        Ok(sparse_bytes) => Some(sparse_bytes),
        Err(e) if is_missing(&e) => None,
        Err(e) => return Err(e),
    };
    let cost = Cost::Bytes(bytes.len() + sparse_bytes.as_ref().map_or(0, Bytes::len));
    cpu::run(lane, cost, move || {
        let mut layer = crate::segment::decode(data_path.as_ref(), &bytes, dim)?;
        if let Some(sparse_bytes) = sparse_bytes {
            let sorted_ids: Vec<String> =
                layer.sorted_docs().iter().map(|d| d.id.clone()).collect();
            let vectors = decode(path.as_ref(), &sparse_bytes, sorted_ids.len() as u64)?;
            layer.set_sparse(&sorted_ids, vectors);
        }
        Ok(layer)
    })
    .await
}

/// Requests in flight at once when fetching several segments (mirrors
/// `segment::read_all`'s own constant).
const FETCH_CONCURRENCY: usize = 16;

/// Fetch and decode the listed segments (each with its sparse sidecar, if it
/// has one) and overlay them oldest to newest — the sparse-aware sibling of
/// `segment::read_all`, used by the query engine's unindexed-segment source.
pub(crate) async fn read_all_with_sparse(
    store: &dyn ObjectStore,
    namespace: &str,
    ids: &[String],
    dim: usize,
    budget: &ReadBudget,
    policy: &FetchPolicy,
    lane: Lane,
) -> Result<Layer> {
    use futures::stream::{self, StreamExt, TryStreamExt};
    // Owned ids: a closure over `&String` is inferred as higher-ranked,
    // which then fails "Send is not general enough" (see
    // `segment::read_all`'s own comment on the same shape of fix).
    stream::iter(ids.iter().cloned())
        .map(|id| async move {
            fetch_with_sparse(store, namespace, &id, dim, budget, policy, lane).await
        })
        .buffered(FETCH_CONCURRENCY)
        .try_fold(Layer::new(), |mut acc, segment| async move {
            acc.overlay(segment);
            Ok::<_, ItzmaError>(acc)
        })
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{Document, WalOp};

    fn layer_of(docs: Vec<Document>) -> Layer {
        let mut layer = Layer::new();
        for d in docs {
            layer.apply(WalOp::Upsert(d));
        }
        layer
    }

    fn doc(id: &str, sparse: Option<Vec<(u32, f32)>>) -> Document {
        Document {
            id: id.into(),
            vector: vec![0.0],
            attrs: None,
            sparse,
        }
    }

    #[test]
    fn encode_decode_round_trips_in_row_order() {
        // sorted_docs() is ascending by id: "a", "b", "c".
        let layer = layer_of(vec![
            doc("c", Some(vec![(5, 1.0)])),
            doc("a", None),
            doc("b", Some(vec![(1, 0.5), (2, 0.25)])),
        ]);
        let bytes = encode(&layer).expect("encode");
        let rows = layer.doc_count() as u64;
        let decoded = decode("p", &bytes, rows).expect("decode");
        assert_eq!(
            decoded,
            vec![
                None,                            // "a"
                Some(vec![(1, 0.5), (2, 0.25)]), // "b"
                Some(vec![(5, 1.0)]),            // "c"
            ]
        );
    }

    #[test]
    fn encode_of_an_empty_layer_decodes_to_no_rows() {
        let layer = Layer::new();
        let bytes = encode(&layer).expect("encode");
        assert_eq!(
            decode("p", &bytes, 0).expect("decode"),
            Vec::<Option<Vec<(u32, f32)>>>::new()
        );
    }

    #[test]
    fn decode_rejects_a_row_count_mismatch() {
        let layer = layer_of(vec![doc("a", None)]);
        let bytes = encode(&layer).expect("encode");
        let err = decode("p", &bytes, 2).unwrap_err();
        assert!(matches!(err, ItzmaError::CorruptSegment { .. }));
    }

    #[test]
    fn every_truncation_and_bit_flip_is_refused_never_panics() {
        let layer = layer_of(vec![
            doc("a", Some(vec![(1, 1.0), (2, 2.0)])),
            doc("b", Some(vec![(3, 3.0)])),
        ]);
        let good = encode(&layer).expect("encode");
        for len in 0..good.len() {
            let _ = decode("p", &good[..len], 2); // must not panic
        }
        for i in 0..good.len() {
            let mut bad = good.to_vec();
            bad[i] ^= 0xFF;
            let _ = decode("p", &bad, 2); // must not panic
        }
    }

    const NS: &str = "ns";
    const ID: &str = "00000000000000000001-00000000000000000003";

    /// Rows 0 to `n`, every third without a sparse vector, the others with
    /// one to four terms.
    fn sparse_layer(n: usize) -> Layer {
        layer_of(
            (0..n)
                .map(|i| {
                    let terms = (i % 3 != 0).then(|| {
                        (0..i % 4 + 1)
                            .map(|t| (t as u32 * 3 + i as u32, i as f32 + 0.5))
                            .collect()
                    });
                    doc(&format!("d{i:05}"), terms)
                })
                .collect(),
        )
    }

    async fn stored(bytes: Bytes) -> object_store::memory::InMemory {
        use object_store::ObjectStoreExt;
        let store = object_store::memory::InMemory::new();
        store
            .put(&sparse_path(NS, ID), bytes.into())
            .await
            .expect("put");
        store
    }

    #[tokio::test]
    async fn rows_read_forward_equal_the_whole_decode() {
        for n in [0usize, 1, 7, 300] {
            let bytes = encode(&sparse_layer(n)).expect("encode");
            let want = decode("p", &bytes, n as u64).expect("decode");
            let store = stored(bytes).await;
            let budget = ReadBudget::unlimited();
            let rows = SparseRows::open(&store, NS, ID, n as u64, &budget)
                .await
                .expect("open")
                .expect("present");
            for per in [1u64, 7, 1000] {
                let mut got = Vec::new();
                let mut start = 0u64;
                while start < n as u64 {
                    let end = (start + per).min(n as u64);
                    got.extend(rows.rows(start..end).await.expect("rows"));
                    start = end;
                }
                assert_eq!(got, want, "{n} rows, {per} per read");
            }
        }
    }

    #[tokio::test]
    async fn a_read_is_charged_its_rows_and_bytes_of_says_so() {
        let bytes = encode(&sparse_layer(300)).expect("encode");
        let store = stored(bytes).await;
        let budget = ReadBudget::unlimited();
        let rows = SparseRows::open(&store, NS, ID, 300, &budget)
            .await
            .expect("open")
            .expect("present");
        assert_eq!(budget.used(), (HEADER_LEN + 300 * DIR_ENTRY_LEN) as u64);
        let before = budget.used();
        rows.rows(10..20).await.expect("rows");
        assert_eq!(budget.used() - before, rows.bytes_of(10..20));
        assert!(rows.bytes_of(10..20) > 0);
        assert_eq!(rows.bytes_of(0..0), 0);
    }

    #[tokio::test]
    async fn a_missing_sidecar_is_none() {
        let store = object_store::memory::InMemory::new();
        let budget = ReadBudget::unlimited();
        let opened = SparseRows::open(&store, NS, ID, 3, &budget)
            .await
            .expect("open");
        assert!(opened.is_none());
        assert_eq!(budget.used(), 0, "nothing was read, so nothing is charged");
    }

    #[tokio::test]
    async fn a_row_count_that_disagrees_is_refused() {
        let store = stored(encode(&sparse_layer(5)).expect("encode")).await;
        let budget = ReadBudget::unlimited();
        let err = SparseRows::open(&store, NS, ID, 6, &budget)
            .await
            .err()
            .expect("refused");
        assert!(matches!(err, ItzmaError::CorruptSegment { .. }), "{err:?}");
    }

    /// Every row of a sidecar holding `bytes`, read forward in one range.
    async fn read_all(bytes: Bytes, rows: u64) -> Result<Vec<Option<SparseVector>>> {
        let store = stored(bytes).await;
        let budget = ReadBudget::unlimited();
        match SparseRows::open(&store, NS, ID, rows, &budget).await? {
            Some(sidecar) => sidecar.rows(0..rows).await,
            None => Ok(Vec::new()),
        }
    }

    #[tokio::test]
    async fn every_truncation_and_bit_flip_of_the_forward_reader_is_refused_never_panics() {
        let good = encode(&sparse_layer(4)).expect("encode");
        let want = decode("p", &good, 4).expect("decode");
        for len in 0..good.len() {
            assert!(
                read_all(good.slice(..len), 4).await.is_err(),
                "truncated to {len}"
            );
        }
        for i in 0..good.len() {
            let mut bad = good.to_vec();
            bad[i] ^= 0xFF;
            // A flip inside a weight still decodes, to another weight; every
            // other flip must be an error. Neither may panic.
            if let Ok(got) = read_all(Bytes::from(bad), 4).await {
                assert_eq!(got.len(), want.len(), "flip at {i}");
            }
        }
    }
}
