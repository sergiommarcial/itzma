//! The index object: a structure derived from one segment's data
//! object, that a query searches before it reads any exact vector.
//!
//! Every kind of index shares a 36-byte prefix, and the rest of the object is
//! the kind's own. The prefix says which kind it is and which segment it was
//! built for (its row count and the size of its data object), so a reader can
//! tell an index that does not belong to a segment without reading further. A
//! reader takes the prefix, then the head (everything before the first
//! partition), then only the partitions it needs.
//!
//! An index only decides speed and recall, never what is correct, so nothing
//! compares two builds byte for byte. The decoders check every length against
//! the bytes that remain before allocating and never panic.
//!
//! # Prefix
//!
//! All integers are little-endian.
//!
//! | Field | Contents |
//! |---|---|
//! | magic | `ITZI` |
//! | `format_version` | u32, 1 |
//! | `kind` | u32, 1 is SQ8 |
//! | `dim` | u32 |
//! | `rows` | u32, the segment's document count |
//! | `data_size` | u64, the size of the segment's data object |
//! | `head_len` | u64, everything before the first partition, the prefix included |

pub(crate) mod hnsw;
pub(crate) mod kmeans;
pub(crate) mod sq8;

use std::sync::Arc;

use bytes::Bytes;
use futures::future::BoxFuture;
use object_store::path::Path;
use object_store::{GetOptions, GetRange, ObjectStore};

use crate::budget::ReadBudget;
use crate::decoded::{DecodedRef, What};
use crate::document::Metric;
use crate::error::{ItzmaError, Result};
use crate::fetch::FetchPolicy;

const MAGIC: [u8; 4] = *b"ITZI";
/// The version of the container. A kind versions its own head and partitions
/// through its `kind` id.
pub(crate) const FORMAT_VERSION: u32 = 1;
/// Magic, version, kind, dim, rows, data size, head length.
pub(crate) const PREFIX_LEN: usize = 4 + 4 + 4 + 4 + 4 + 8 + 8;

/// Which algorithm an index object belongs to. The id is stored in the object
/// and the name in the manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    /// Scalar quantization to 8 bits per dimension, scanned exhaustively
    /// inside the partitions it probes (`sq8`).
    Sq8,
    /// Segmented HNSW: a graph over SQ8-quantized codes, traversed directly
    /// out of the cached object bytes (`hnsw`).
    Hnsw,
}

impl Kind {
    pub(crate) fn id(self) -> u32 {
        match self {
            Kind::Sq8 => 1,
            Kind::Hnsw => 2,
        }
    }

    pub(crate) fn from_id(id: u32) -> Option<Kind> {
        match id {
            1 => Some(Kind::Sq8),
            2 => Some(Kind::Hnsw),
            _ => None,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Kind::Sq8 => "sq8",
            Kind::Hnsw => "hnsw",
        }
    }

    pub(crate) fn from_name(name: &str) -> Option<Kind> {
        match name {
            "sq8" => Some(Kind::Sq8),
            "hnsw" => Some(Kind::Hnsw),
            _ => None,
        }
    }

    /// The code that searches an index of this kind.
    pub(crate) fn algorithm(self) -> &'static dyn Algorithm {
        match self {
            Kind::Sq8 => &sq8::Sq8,
            Kind::Hnsw => &hnsw::Hnsw,
        }
    }
}

/// One candidate row of a segment with the distance the algorithm gave it,
/// which may be approximate. Lower is closer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Candidate {
    pub(crate) row: u32,
    pub(crate) distance: f32,
}

/// What an algorithm gets to search one segment's index.
///
/// `Copy`, since cursors pass this by value and need to rebuild
/// a tweaked copy (a different `want`) on every internal re-search.
#[derive(Clone, Copy)]
pub(crate) struct Probe<'a> {
    pub(crate) store: &'a dyn ObjectStore,
    /// Every byte the search reads is charged to this.
    pub(crate) budget: &'a ReadBudget,
    /// How ranged reads turn wanted byte ranges into requests.
    pub(crate) policy: &'a FetchPolicy,
    /// The index object's key.
    pub(crate) path: &'a Path,
    /// The segment's dimension, document count and data object size, which the
    /// index object has to agree with.
    pub(crate) dim: usize,
    pub(crate) rows: u64,
    pub(crate) data_size: u64,
    pub(crate) metric: Metric,
    pub(crate) query: &'a [f32],
    /// How many candidates the engine wants at most, `usize::MAX` for every row.
    pub(crate) want: usize,
    /// How many partitions to probe, for a kind that has them, or `None` for
    /// the kind's own choice.
    pub(crate) nprobe: Option<usize>,
    /// Where parsed heads and partitions are kept between queries.
    /// A hit is neither fetched nor charged to `budget`.
    pub(crate) decoded: Option<&'a DecodedRef>,
}

/// The half of an index kind that runs at query time (the other half, building,
/// is CPU only and lives with the kind). The engine reads no index bytes itself:
/// it asks the kind for candidate rows and does the exact rerank, the shadow
/// check across segments and the attributes for whatever comes back, so a kind
/// cannot break any of them. A kind may return fewer than `want` rows, which
/// says it has no more, and the engine does not trust the rows it returns
/// (see `query.rs`).
pub(crate) trait Algorithm: Send + Sync {
    /// Up to `probe.want` candidate rows of the segment, closest first by the
    /// kind's own distance.
    ///
    /// # Errors
    ///
    /// [`ItzmaError::CorruptIndex`] if the object is damaged or does not fit the
    /// segment (see [`is_missing`] for one that is gone), and
    /// [`ItzmaError::ReadBudgetExceeded`] before a fetch that would go over.
    fn search<'a>(&'a self, probe: &'a Probe<'_>) -> BoxFuture<'a, Result<Vec<Candidate>>>;

    /// A lower bound on the exact distance of a row this algorithm scored at
    /// `approx`, or `None` when the kind cannot promise one for this probe's
    /// metric. The default is always `None`: a kind that does not
    /// override this gets none of the pruning, never a wrong one.
    fn lower_bound(&self, probe: &Probe<'_>, approx: f32) -> Option<f32> {
        let _ = (probe, approx);
        None
    }

    /// A cursor over this index. The default wraps [`search`](Algorithm::search)
    /// with the same widening and every-row fallback the engine used before
    /// cursors existed: the first call searches for `n`, and a
    /// call that wants more than has been scanned so far re-searches with a
    /// bigger total (times [`DEFAULT_CURSOR_WIDEN`], then every row), never
    /// returning a row already given out.
    fn cursor<'a>(&'a self, probe: Probe<'a>) -> BoxFuture<'a, Result<Box<dyn Cursor + 'a>>> {
        Box::pin(async move { Ok(Box::new(DefaultCursor::new(self, probe)) as Box<dyn Cursor>) })
    }
}

/// What a batch of fresh candidates from a [`Cursor`] carries.
#[derive(Debug)]
pub(crate) struct Batch {
    pub(crate) candidates: Vec<Candidate>,
    /// A lower bound on the exact distance of every row not yet returned,
    /// or `None` if the kind cannot give one. `Some(f32::INFINITY)` once
    /// nothing is left (equivalent to `exhausted`, but explicit so a caller
    /// checking only the bound still stops).
    pub(crate) rest_lower_bound: Option<f32>,
    /// Every row of the segment has now been returned, across all calls.
    pub(crate) exhausted: bool,
}

/// Candidates of one segment's index, pulled in batches. No row is ever
/// returned twice across calls on the same cursor.
pub(crate) trait Cursor: Send {
    /// Up to `n` more candidates, sorted by distance within this batch,
    /// none repeated from an earlier call. Not necessarily monotonic
    /// *across* calls: a kind that groups rows (SQ8's partitions, visited
    /// nearest-centroid-first) can have a later batch contain a row closer
    /// than one an earlier batch already returned, since a centroid being
    /// closer does not bound every row in its group. True cross-batch
    /// best-first order is the partition bounds, not built here.
    fn next<'a>(&'a mut self, n: usize) -> BoxFuture<'a, Result<Batch>>;
}

/// `found`, sorted by row, plus every row of the segment it lacks, farthest
/// last. Asking an index for every row is the widening loop's last resort, and
/// a graph can leave a row that no search reaches. The distance of a filler
/// row is never used: the rescore against the exact vector replaces it.
pub(crate) fn with_every_row(found: Vec<Candidate>, rows: u64) -> Vec<Candidate> {
    let last = u32::try_from(rows).unwrap_or(u32::MAX);
    let mut known = found.into_iter().peekable();
    (0..last)
        .map(|row| {
            known.next_if(|c| c.row == row).unwrap_or(Candidate {
                row,
                distance: f32::INFINITY,
            })
        })
        .collect()
}

/// How much bigger each internal re-search is, when a caller's `next(n)`
/// wants more new rows than are buffered. Same constant and reasoning as
/// the pre-cursor engine's `WIDEN`.
const DEFAULT_CURSOR_WIDEN: usize = 8;

/// Wraps [`Algorithm::search`] to behave like a [`Cursor`], for any kind
/// that has not implemented one of its own. Keeps every row `search` has
/// returned so far, in distance order, and the set of rows already handed
/// out, so `next` never repeats one.
///
/// Generic over `A` (rather than holding `&'a dyn Algorithm` directly) so
/// the trait's default `cursor` method — which must stay callable through
/// `&dyn Algorithm`, since that is exactly how the engine calls it — never
/// needs to coerce `&Self` into a trait object inside its own generic body.
/// `A: ?Sized` lets `A` be a concrete kind (when a caller invokes
/// `cursor` directly on, say, `Sq8`) or `dyn Algorithm` itself (when called
/// through a trait object); either way `DefaultCursor<'a, A>` is `Sized`,
/// since it only ever stores a *reference* to `A`, never `A` itself.
struct DefaultCursor<'a, A: Algorithm + ?Sized> {
    algorithm: &'a A,
    probe: Probe<'a>,
    /// Every row `search` has returned so far, sorted by distance, deduped,
    /// with gaps filled in once `total` reaches every row (today's
    /// `with_every_row`).
    buffered: Vec<Candidate>,
    /// Rows already handed back by `next`.
    returned: std::collections::HashSet<u32>,
    /// The running total asked of `search` so far; grows by
    /// `DEFAULT_CURSOR_WIDEN` or jumps to every row.
    total: usize,
    exhausted: bool,
}

impl<'a, A: Algorithm + ?Sized> DefaultCursor<'a, A> {
    fn new(algorithm: &'a A, probe: Probe<'a>) -> Self {
        let total = probe.want;
        DefaultCursor {
            algorithm,
            probe,
            buffered: Vec::new(),
            returned: std::collections::HashSet::new(),
            total,
            exhausted: false,
        }
    }

    /// Re-search with a bigger total and refresh `buffered`, applying the
    /// same checks `Indexed::propose` used to: no row past the segment, no
    /// duplicate, filled in to every row once `total` saturates.
    async fn refill(&mut self) -> Result<()> {
        let mut probe = self.probe;
        probe.want = self.total;
        let name = probe.path.as_ref().to_string();
        let rows = probe.rows;
        let mut found = self.algorithm.search(&probe).await?;
        if let Some(bad) = found.iter().find(|c| u64::from(c.row) >= rows) {
            return Err(corrupt(
                &name,
                format!("the index proposed row {}, the segment has {rows}", bad.row),
            ));
        }
        found.sort_unstable_by_key(|c| c.row);
        found.dedup_by_key(|c| c.row);
        // Compared against `self.total` (what this round actually asked
        // for), not `rows` (the whole segment): a kind that returns
        // exactly `want` rows whenever it can, which is almost always far
        // below `rows`, so comparing against `rows` here declared every
        // such round "saturated" and jumped straight to `usize::MAX`
        // (every row) on the very next widening, skipping the intended
        // `DEFAULT_CURSOR_WIDEN` growth steps entirely. SQ8 has its own
        // cursor and never takes this path; a kind that returns exactly
        // `want` rows would.
        let saturated = found.len() < self.total && self.total != usize::MAX;
        if self.total == usize::MAX && (found.len() as u64) < rows {
            found = with_every_row(found, rows);
        }
        found.sort_by(|a, b| a.distance.total_cmp(&b.distance).then(a.row.cmp(&b.row)));
        self.exhausted = self.total == usize::MAX || found.len() as u64 >= rows;
        self.buffered = found;
        if saturated {
            self.total = usize::MAX; // a partial probe never finds more with a bigger ask
        } else if !self.exhausted {
            self.total = self.total.saturating_mul(DEFAULT_CURSOR_WIDEN);
        }
        Ok(())
    }
}

impl<A: Algorithm + ?Sized> Cursor for DefaultCursor<'_, A> {
    fn next<'a>(&'a mut self, n: usize) -> BoxFuture<'a, Result<Batch>> {
        Box::pin(async move {
            loop {
                let fresh: Vec<Candidate> = self
                    .buffered
                    .iter()
                    .filter(|c| !self.returned.contains(&c.row))
                    .take(n)
                    .copied()
                    .collect();
                if fresh.len() >= n || self.exhausted {
                    for c in &fresh {
                        self.returned.insert(c.row);
                    }
                    // `self.exhausted` means `buffered` holds every row of the
                    // segment — not that every row has been handed to the
                    // caller. `with_every_row` can fill `buffered` to the full
                    // row count in one `refill`, long before `next` has
                    // drained it: a caller that stopped widening on seeing
                    // `self.exhausted` alone would miss every row still
                    // sitting unreturned in `buffered` (this crashed
                    // `a_selective_filter_on_a_partially_probed_clustered_index_still_finds_every_match`
                    // in `tests/filtered_query.rs`, since most matches sat in
                    // partitions the single probe never read, parked in
                    // `buffered` as filler rows this batch never reached).
                    let exhausted = self.exhausted && self.returned.len() as u64 >= self.probe.rows;
                    return Ok(Batch {
                        candidates: fresh,
                        rest_lower_bound: None,
                        exhausted,
                    });
                }
                self.refill().await?;
            }
        })
    }
}

/// Prefix every index object is written with, regardless of kind — the kind
/// itself is read from the object's own header (see [`Prefix`]), never
/// guessed from the name.
const OBJECT_PREFIX: &str = "index-";
const OBJECT_SUFFIX: &str = ".bin";

/// Object key of an index: `<namespace>/segments/<id>/index-<generation>.bin`.
/// The kind inside is read from the object's own header ([`Prefix::kind`]),
/// never from the name — the name says nothing about which algorithm built it.
pub(crate) fn index_path(namespace: &str, id: &str, generation: u32) -> Path {
    Path::from(format!(
        "{namespace}/segments/{id}/{OBJECT_PREFIX}{generation}{OBJECT_SUFFIX}"
    ))
}

/// The generation in an object name inside a segment directory, or `None`
/// for any other name (`data.bin`, or a name that is not exactly what
/// [`index_path`] produces).
pub(crate) fn generation_of(file_name: &str) -> Option<u32> {
    let digits = file_name
        .strip_prefix(OBJECT_PREFIX)?
        .strip_suffix(OBJECT_SUFFIX)?;
    let generation: u32 = digits.parse().ok()?;
    (generation >= 1 && generation.to_string() == digits).then_some(generation)
}

pub(crate) fn corrupt(path: &str, reason: impl Into<String>) -> ItzmaError {
    ItzmaError::CorruptIndex {
        path: path.to_string(),
        reason: reason.into(),
    }
}

/// The `reason` of the error for an index object the store does not have.
/// Callers recognise it through [`is_missing`], never by text.
const MISSING: &str = "missing from the store";

/// The error for an index object the manifest names and the store lacks. Like a
/// missing segment it may only mean the caller read a stale manifest: a rebuild
/// and a garbage collection can have replaced the generation since.
pub(crate) fn missing(path: &Path) -> ItzmaError {
    corrupt(path.as_ref(), MISSING)
}

/// Whether `err` is what [`missing`] builds.
pub(crate) fn is_missing(err: &ItzmaError) -> bool {
    matches!(err, ItzmaError::CorruptIndex { reason, .. } if reason == MISSING)
}

/// How many bytes the first request for an index asks for. The prefix and the
/// head of a segment of a few thousand dimensions fit, so opening is one round
/// trip for most.
const HEAD_GUESS: u64 = 32 * 1024;

/// The prefix and head of an index object, checked against the segment.
#[derive(Debug)]
pub(crate) struct Opened {
    pub(crate) prefix: Prefix,
    /// The bytes between the prefix and `head_len`: the kind's own head.
    pub(crate) body: Bytes,
    /// The size of the whole object.
    pub(crate) size: u64,
}

/// Read the prefix and the head of an index of `kind`, and check that they
/// belong to the segment the probe is about. One request when the head fits the
/// guess, two otherwise. Nothing is sized from the object before the object's
/// real size has been compared with it.
///
/// # Errors
///
/// [`ItzmaError::CorruptIndex`] for an object that is missing (see
/// [`is_missing`]), too short, of another kind, built for another segment, or
/// whose head runs past its end. [`ItzmaError::UnsupportedFormat`] for a
/// container version this build does not read.
///
/// With a decoded cache in the probe, a head opened before is served from it
/// with no request and no charge.
pub(crate) async fn open(probe: &Probe<'_>, kind: Kind) -> Result<Arc<Opened>> {
    match probe.decoded {
        Some(decoded) => {
            decoded
                .get_or_load(probe.path, What::IndexHead, || async {
                    let mut opened = open_uncached(probe, kind).await?;
                    // Its own buffer, so the kept head does not pin the whole
                    // first read it was sliced from.
                    opened.body = Bytes::copy_from_slice(&opened.body);
                    let weight = opened.body.len() as u64;
                    Ok((opened, weight))
                })
                .await
        }
        None => Ok(Arc::new(open_uncached(probe, kind).await?)),
    }
}

/// [`open`] with no cache: always fetched and charged.
pub(crate) async fn open_uncached(probe: &Probe<'_>, kind: Kind) -> Result<Opened> {
    let name = probe.path.as_ref();
    probe.budget.charge(name, HEAD_GUESS)?;
    let options = GetOptions {
        range: Some(GetRange::Bounded(0..HEAD_GUESS)),
        ..GetOptions::default()
    };
    let got = match probe.store.get_opts(probe.path, options).await {
        Ok(got) => got,
        Err(object_store::Error::NotFound { .. }) => return Err(missing(probe.path)),
        Err(e) => return Err(e.into()),
    };
    let size = got.meta.size;
    let first = got.bytes().await?;

    let prefix = Prefix::decode(name, &first)?;
    if prefix.kind != kind {
        return Err(corrupt(
            name,
            format!(
                "the manifest names an index of kind {}, the object is of kind {}",
                kind.name(),
                prefix.kind.name()
            ),
        ));
    }
    prefix.check_segment(name, probe.dim, probe.rows, probe.data_size)?;
    if prefix.head_len > size {
        return Err(corrupt(name, "the head runs past the end of the object"));
    }
    let head_len = prefix.head_len;
    let body = if head_len <= first.len() as u64 {
        first.slice(PREFIX_LEN..head_len as usize)
    } else {
        #[allow(clippy::single_range_in_vec_init)] // one wanted range: the rest of the head
        let wanted = [first.len() as u64..head_len];
        let planned = crate::fetch::plan(probe.policy, &wanted, size);
        let slices = crate::fetch::execute(probe.store, probe.budget, probe.path, planned)
            .await
            .map_err(|e| match e {
                ItzmaError::Storage(object_store::Error::NotFound { .. }) => missing(probe.path),
                other => other,
            })?;
        Bytes::from([&first[PREFIX_LEN..], &slices[0][..]].concat())
    };
    Ok(Opened { prefix, body, size })
}

/// The fixed part at the front of every index object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Prefix {
    pub(crate) kind: Kind,
    pub(crate) dim: u32,
    pub(crate) rows: u32,
    pub(crate) data_size: u64,
    pub(crate) head_len: u64,
}

impl Prefix {
    pub(crate) fn encode(&self) -> [u8; PREFIX_LEN] {
        let mut out = [0u8; PREFIX_LEN];
        let mut at = 0;
        let mut put = |bytes: &[u8]| {
            out[at..at + bytes.len()].copy_from_slice(bytes);
            at += bytes.len();
        };
        put(&MAGIC);
        put(&FORMAT_VERSION.to_le_bytes());
        put(&self.kind.id().to_le_bytes());
        put(&self.dim.to_le_bytes());
        put(&self.rows.to_le_bytes());
        put(&self.data_size.to_le_bytes());
        put(&self.head_len.to_le_bytes());
        out
    }

    /// Read the prefix from the first bytes of an object.
    ///
    /// # Errors
    ///
    /// [`ItzmaError::UnsupportedFormat`] for a container version this build
    /// does not read. [`ItzmaError::CorruptIndex`] for anything else wrong: too
    /// few bytes, bad magic, an unknown kind, a zero dimension, or a head
    /// length shorter than the prefix.
    pub(crate) fn decode(path: &str, bytes: &[u8]) -> Result<Prefix> {
        let mut r = Reader::new(path, bytes);
        if r.take(4)? != MAGIC {
            return Err(corrupt(path, "bad magic"));
        }
        match r.u32()? {
            FORMAT_VERSION => {}
            version => return Err(ItzmaError::UnsupportedFormat { version }),
        }
        let kind_id = r.u32()?;
        let kind = Kind::from_id(kind_id)
            .ok_or_else(|| corrupt(path, format!("unknown index kind {kind_id}")))?;
        let dim = r.u32()?;
        let rows = r.u32()?;
        let data_size = r.u64()?;
        let head_len = r.u64()?;
        if dim == 0 {
            return Err(corrupt(path, "dimension is 0"));
        }
        if head_len < PREFIX_LEN as u64 {
            return Err(corrupt(path, "head is shorter than the prefix"));
        }
        Ok(Prefix {
            kind,
            dim,
            rows,
            data_size,
            head_len,
        })
    }

    /// Check that the index was built for a segment with this dimension, row
    /// count and data object size.
    pub(crate) fn check_segment(
        &self,
        path: &str,
        dim: usize,
        rows: u64,
        data_size: u64,
    ) -> Result<()> {
        if usize::try_from(self.dim).ok() != Some(dim) {
            return Err(corrupt(
                path,
                format!("dimension {}, the segment has {dim}", self.dim),
            ));
        }
        if u64::from(self.rows) != rows {
            return Err(corrupt(
                path,
                format!("{} rows, the segment has {rows}", self.rows),
            ));
        }
        if self.data_size != data_size {
            return Err(corrupt(
                path,
                format!(
                    "built for a data object of {} bytes, the segment's is {data_size}",
                    self.data_size
                ),
            ));
        }
        Ok(())
    }
}

/// A cursor over bytes that never reads past the end.
pub(crate) struct Reader<'a> {
    path: &'a str,
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(path: &'a str, buf: &'a [u8]) -> Self {
        Reader { path, buf, pos: 0 }
    }

    #[cfg(test)]
    pub(crate) fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    pub(crate) fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let out = self
            .buf
            .get(self.pos..)
            .and_then(|rest| rest.get(..n))
            .ok_or_else(|| corrupt(self.path, "object is truncated"))?;
        self.pos += n;
        Ok(out)
    }

    pub(crate) fn u16(&mut self) -> Result<u16> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    pub(crate) fn u32(&mut self) -> Result<u32> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    pub(crate) fn u64(&mut self) -> Result<u64> {
        let bytes = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(bytes);
        Ok(u64::from_le_bytes(a))
    }

    pub(crate) fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.u32()?))
    }

    /// How many bytes have been consumed so far. Used where a decoder needs
    /// to know exactly where one variable-length record ended and the next
    /// begins (the HNSW node-directory chain check), the same role
    /// `head_len` plays for SQ8's fixed-size partition table.
    pub(crate) fn pos(&self) -> usize {
        self.pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::ObjectStoreExt;

    fn prefix() -> Prefix {
        Prefix {
            kind: Kind::Sq8,
            dim: 3,
            rows: 10,
            data_size: 1234,
            head_len: 100,
        }
    }

    #[test]
    fn the_prefix_is_fixed_bytes() {
        let bytes = prefix().encode();
        assert_eq!(bytes.len(), PREFIX_LEN);
        assert_eq!(PREFIX_LEN, 36);
        let mut want = Vec::new();
        want.extend_from_slice(b"ITZI");
        want.extend_from_slice(&1u32.to_le_bytes());
        want.extend_from_slice(&1u32.to_le_bytes());
        want.extend_from_slice(&3u32.to_le_bytes());
        want.extend_from_slice(&10u32.to_le_bytes());
        want.extend_from_slice(&1234u64.to_le_bytes());
        want.extend_from_slice(&100u64.to_le_bytes());
        assert_eq!(bytes.to_vec(), want);
    }

    #[test]
    fn the_prefix_round_trips() {
        let p = prefix();
        assert_eq!(Prefix::decode("p", &p.encode()).expect("decode"), p);
    }

    #[test]
    fn kinds_have_a_stable_id_and_name() {
        assert_eq!(Kind::Sq8.id(), 1);
        assert_eq!(Kind::Sq8.name(), "sq8");
        assert_eq!(Kind::from_id(1), Some(Kind::Sq8));
        assert_eq!(Kind::from_name("sq8"), Some(Kind::Sq8));
        assert_eq!(Kind::Hnsw.id(), 2);
        assert_eq!(Kind::Hnsw.name(), "hnsw");
        assert_eq!(Kind::from_id(2), Some(Kind::Hnsw));
        assert_eq!(Kind::from_name("hnsw"), Some(Kind::Hnsw));
        assert_eq!(Kind::from_name("graph"), None);
        assert_eq!(Kind::from_id(0), None);
        assert_eq!(Kind::from_id(3), None);
        assert_eq!(Kind::from_name("ivf"), None);
    }

    fn reason(err: ItzmaError) -> String {
        match err {
            ItzmaError::CorruptIndex { reason, .. } => reason,
            other => format!("{other:?}"),
        }
    }

    #[test]
    fn a_bad_prefix_is_refused_with_the_reason() {
        let good = prefix().encode();
        let patched = |at: usize, bytes: &[u8]| {
            let mut b = good.to_vec();
            b[at..at + bytes.len()].copy_from_slice(bytes);
            b
        };
        let cases: [(Vec<u8>, &str); 5] = [
            (patched(0, b"ITZX"), "bad magic"),
            (patched(8, &7u32.to_le_bytes()), "unknown index kind 7"),
            (patched(12, &0u32.to_le_bytes()), "dimension is 0"),
            (
                patched(28, &35u64.to_le_bytes()),
                "head is shorter than the prefix",
            ),
            (good[..PREFIX_LEN - 1].to_vec(), "truncated"),
        ];
        for (bytes, want) in cases {
            let err = Prefix::decode("p", &bytes).expect_err(want);
            assert!(reason(err).contains(want), "{want}");
        }
        let newer = patched(4, &2u32.to_le_bytes());
        assert!(matches!(
            Prefix::decode("p", &newer),
            Err(ItzmaError::UnsupportedFormat { version: 2 })
        ));
    }

    #[test]
    fn the_prefix_is_checked_against_its_segment() {
        let p = prefix();
        p.check_segment("p", 3, 10, 1234).expect("matches");
        for (dim, rows, size, want) in [
            (4, 10, 1234, "dimension"),
            (3, 11, 1234, "rows"),
            (3, 10, 1235, "data object"),
        ] {
            let err = p.check_segment("p", dim, rows, size).expect_err(want);
            assert!(reason(err).contains(want), "{want}");
        }
    }

    #[test]
    fn an_index_lives_beside_its_data_object() {
        assert_eq!(
            index_path("docs", "0001-0002", 3).as_ref(),
            "docs/segments/0001-0002/index-3.bin"
        );
    }

    #[test]
    fn only_the_exact_object_name_has_a_generation() {
        assert_eq!(generation_of("index-1.bin"), Some(1));
        assert_eq!(generation_of("index-42.bin"), Some(42));
        for name in [
            "ivf-1.bin",
            "data.bin",
            "index-0.bin",
            "index-01.bin",
            "index-.bin",
            "index-1",
            "index-1.bin.tmp",
            "index--1.bin",
            "index-+1.bin",
            "INDEX-1.bin",
            "index-99999999999.bin",
            "ivf-0.bin",
            "ivf-01.bin",
            "ivf-.bin",
            "ivf-1",
            "ivf-1.bin.tmp",
            "IVF-1.bin",
        ] {
            assert_eq!(generation_of(name), None, "{name}");
        }
    }

    #[test]
    fn a_reader_never_goes_past_the_end() {
        let mut r = Reader::new("p", &[1, 2, 3]);
        assert!(r.u32().is_err());
        assert_eq!(r.remaining(), 3);
        assert_eq!(r.take(3).expect("take"), [1, 2, 3]);
        assert!(r.take(1).is_err());
    }

    /// An in-memory store that counts every read, so a test can say how
    /// many requests a call made. Not shared with the similarly-named type
    /// in other modules — see `segment/ranged.rs`'s doc comment on why.
    #[derive(Debug, Default)]
    struct CountingStore {
        inner: object_store::memory::InMemory,
        served: std::sync::atomic::AtomicUsize,
        /// Deletes `.0` from `inner` just before serving its `.1`-th GET
        /// (1-based), standing in for a `gc` race landing between two of a
        /// caller's own requests.
        delete_on_nth_get: std::sync::Mutex<Option<(Path, usize)>>,
    }

    impl CountingStore {
        fn requests(&self) -> usize {
            self.served.load(std::sync::atomic::Ordering::SeqCst)
        }

        fn delete_on_nth_get(&self, path: Path, n: usize) {
            *self.delete_on_nth_get.lock().unwrap() = Some((path, n));
        }
    }

    impl std::fmt::Display for CountingStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "CountingStore")
        }
    }

    #[async_trait::async_trait]
    impl ObjectStore for CountingStore {
        async fn put_opts(
            &self,
            location: &Path,
            payload: object_store::PutPayload,
            opts: object_store::PutOptions,
        ) -> object_store::Result<object_store::PutResult> {
            self.inner.put_opts(location, payload, opts).await
        }

        async fn put_multipart_opts(
            &self,
            location: &Path,
            opts: object_store::PutMultipartOptions,
        ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
            self.inner.put_multipart_opts(location, opts).await
        }

        async fn get_opts(
            &self,
            location: &Path,
            options: GetOptions,
        ) -> object_store::Result<object_store::GetResult> {
            let n = self
                .served
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                + 1;
            let should_delete = {
                let guard = self.delete_on_nth_get.lock().unwrap();
                matches!(&*guard, Some((path, at)) if *at == n && path == location)
            };
            if should_delete {
                let _ = self.inner.delete(location).await;
            }
            self.inner.get_opts(location, options).await
        }

        fn delete_stream(
            &self,
            locations: futures::stream::BoxStream<'static, object_store::Result<Path>>,
        ) -> futures::stream::BoxStream<'static, object_store::Result<Path>> {
            self.inner.delete_stream(locations)
        }

        fn list(
            &self,
            prefix: Option<&Path>,
        ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>>
        {
            self.inner.list(prefix)
        }

        async fn list_with_delimiter(
            &self,
            prefix: Option<&Path>,
        ) -> object_store::Result<object_store::ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }

        async fn copy_opts(
            &self,
            from: &Path,
            to: &Path,
            options: object_store::CopyOptions,
        ) -> object_store::Result<()> {
            self.inner.copy_opts(from, to, options).await
        }
    }

    /// A head bigger than `HEAD_GUESS` (an index whose kind keeps most of
    /// its object in the head). Under a
    /// splitting policy, the rest of it is fetched as parallel planner
    /// parts instead of one big `get_range`.
    #[tokio::test]
    async fn a_head_bigger_than_the_guess_is_split_into_parts_under_a_splitting_policy() {
        use object_store::ObjectStoreExt as _;

        let dim = 4u32;
        let rows = 10u32;
        let head_len = HEAD_GUESS * 3; // forces a second request
        // Distinct, non-zero bytes so a shuffled or truncated reassembly
        // would actually show up in the body comparison below.
        let mut bytes: Vec<u8> = (0u64..head_len + 100).map(|i| (i % 251) as u8).collect();
        let prefix = Prefix {
            kind: Kind::Sq8,
            dim,
            rows,
            data_size: bytes.len() as u64,
            head_len,
        };
        bytes[..PREFIX_LEN].copy_from_slice(&prefix.encode());

        let store = CountingStore::default();
        let path = Path::from("idx.bin");
        store.put(&path, bytes.clone().into()).await.expect("put");

        let budget = ReadBudget::unlimited();
        let splitting = FetchPolicy {
            split_above: HEAD_GUESS,
            part_size: HEAD_GUESS,
            ..FetchPolicy::object_store()
        };
        let probe = Probe {
            store: &store,
            budget: &budget,
            policy: &splitting,
            path: &path,
            dim: dim as usize,
            rows: rows as u64,
            data_size: bytes.len() as u64,
            metric: Metric::L2,
            query: &[0.0; 4],
            want: 1,
            nprobe: None,
            decoded: None,
        };
        let opened = open(&probe, Kind::Sq8).await.expect("open");
        assert_eq!(
            &opened.body[..],
            &bytes[PREFIX_LEN..head_len as usize],
            "the reassembled head must match the stored bytes exactly, in order"
        );
        assert!(
            store.requests() > 2,
            "the remainder of the head was split into more than one part"
        );
    }

    /// `open`'s first request translates a missing object directly
    /// (matched on the raw `object_store::Error`). Its second request —
    /// the rest of a head bigger than `HEAD_GUESS`, fetched through
    /// `fetch::plan`/`execute` — must translate one too: a `gc` race that
    /// deletes the object between the two requests is exactly what
    /// `is_stale_read` exists to retry, not a raw storage error.
    #[tokio::test]
    async fn a_missing_object_found_only_on_opens_second_request_is_still_reported_as_missing() {
        use object_store::ObjectStoreExt as _;

        let dim = 4u32;
        let rows = 10u32;
        let head_len = HEAD_GUESS * 3;
        let mut bytes = vec![0u8; head_len as usize + 100];
        let prefix = Prefix {
            kind: Kind::Sq8,
            dim,
            rows,
            data_size: bytes.len() as u64,
            head_len,
        };
        bytes[..PREFIX_LEN].copy_from_slice(&prefix.encode());

        let store = CountingStore::default();
        let path = Path::from("idx.bin");
        store.put(&path, bytes.clone().into()).await.expect("put");
        store.delete_on_nth_get(path.clone(), 2);

        let budget = ReadBudget::unlimited();
        let policy = FetchPolicy::object_store();
        let probe = Probe {
            store: &store,
            budget: &budget,
            policy: &policy,
            path: &path,
            dim: dim as usize,
            rows: rows as u64,
            data_size: bytes.len() as u64,
            metric: Metric::L2,
            query: &[0.0; 4],
            want: 1,
            nprobe: None,
            decoded: None,
        };
        let err = open(&probe, Kind::Sq8).await.expect_err("missing");
        assert!(is_missing(&err), "{err:?}");
    }
}

/// `DefaultCursor` reproducing today's pre-cursor widening behavior, for any
/// kind that does not implement its own `Cursor`.
#[cfg(test)]
mod default_cursor_tests {
    use super::*;
    use crate::budget::ReadBudget;
    use futures::FutureExt;
    use object_store::memory::InMemory;
    use object_store::path::Path;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Returns exactly the rows it is built with, once per call, counting
    /// calls — the same shape as `query.rs`'s pre-cursor `Counting` test
    /// stub, moved here since `DefaultCursor` is what now owns the widening
    /// logic it is testing.
    struct Counting {
        rows: Vec<Candidate>,
        calls: AtomicUsize,
    }

    impl Algorithm for Counting {
        fn search<'a>(&'a self, probe: &'a Probe<'_>) -> BoxFuture<'a, Result<Vec<Candidate>>> {
            let _ = probe;
            self.calls.fetch_add(1, Ordering::SeqCst);
            let rows = self.rows.clone();
            async move { Ok(rows) }.boxed()
        }
    }

    #[tokio::test]
    async fn default_cursor_grows_like_todays_widening_then_fills_every_row() {
        let store = InMemory::new();
        let budget = ReadBudget::unlimited();
        let policy = FetchPolicy::local();
        let path = Path::from("x");
        let algo = Counting {
            rows: vec![Candidate {
                row: 0,
                distance: 0.0,
            }],
            calls: AtomicUsize::new(0),
        };
        let probe = Probe {
            store: &store,
            budget: &budget,
            policy: &policy,
            path: &path,
            dim: 2,
            rows: 3,
            data_size: 0,
            metric: crate::document::Metric::L2,
            query: &[0.0, 0.0],
            want: 1,
            nprobe: None,
            decoded: None,
        };
        let mut cursor = algo.cursor(probe).await.expect("cursor");
        // First batch: only what's been scanned so far (1 row).
        let batch1 = cursor.next(1).await.expect("next");
        assert_eq!(batch1.candidates.len(), 1);
        assert!(!batch1.exhausted);
        // Ask for more than the algorithm has given so far: it must widen
        // (ask `search` again with a bigger total) and hand back new rows
        // only (rows 1 and 2, filled in since the algorithm never offers
        // them — the "every row" fallback from today's `with_every_row`).
        let batch2 = cursor.next(5).await.expect("next");
        assert_eq!(batch2.candidates.len(), 2, "rows 1 and 2, not row 0 again");
        assert!(batch2.exhausted);
    }

    /// Regression for `a_selective_filter_on_a_partially_probed_clustered_index_still_finds_every_match`
    /// (`tests/filtered_query.rs`): an algorithm that always returns the same
    /// small set of rows (as a fixed `nprobe` does, probing the same nearest
    /// partition every call) forces `with_every_row`'s fill-in on the first
    /// widening, which fills `buffered` with every row of the segment at
    /// once. `self.exhausted` (internal: "buffered holds everything") must
    /// not leak into `Batch.exhausted` until every one of those rows has
    /// actually been handed out across `next` calls — otherwise a caller
    /// stops widening with most of the segment never returned.
    #[tokio::test]
    async fn exhausted_is_not_reported_until_every_filled_in_row_has_been_returned() {
        struct AlwaysTheSameTwo;
        impl Algorithm for AlwaysTheSameTwo {
            fn search<'a>(&'a self, probe: &'a Probe<'_>) -> BoxFuture<'a, Result<Vec<Candidate>>> {
                let _ = probe;
                async move {
                    Ok(vec![
                        Candidate {
                            row: 0,
                            distance: 0.0,
                        },
                        Candidate {
                            row: 1,
                            distance: 1.0,
                        },
                    ])
                }
                .boxed()
            }
        }

        let store = InMemory::new();
        let budget = ReadBudget::unlimited();
        let policy = FetchPolicy::local();
        let path = Path::from("x");
        let algo = AlwaysTheSameTwo;
        let probe = Probe {
            store: &store,
            budget: &budget,
            policy: &policy,
            path: &path,
            dim: 2,
            rows: 10, // far more rows than the algorithm ever proposes
            data_size: 0,
            metric: crate::document::Metric::L2,
            query: &[0.0, 0.0],
            want: 2,
            nprobe: None,
            decoded: None,
        };
        let mut cursor = algo.cursor(probe).await.expect("cursor");

        let batch1 = cursor.next(2).await.expect("next");
        assert_eq!(batch1.candidates.len(), 2, "the two real rows");
        assert!(
            !batch1.exhausted,
            "8 filled-in rows still sit in buffered, unreturned"
        );

        let mut seen: std::collections::HashSet<u32> =
            batch1.candidates.iter().map(|c| c.row).collect();
        let mut exhausted = batch1.exhausted;
        while !exhausted {
            let batch = cursor.next(2).await.expect("next");
            for c in &batch.candidates {
                assert!(seen.insert(c.row), "row {} returned twice", c.row);
            }
            exhausted = batch.exhausted;
        }
        assert_eq!(seen.len(), 10, "every row of the segment, eventually");
    }

    /// Regression: `refill`'s "saturated" check
    /// must compare `found.len()` against what this round actually asked
    /// for (`self.total`), not against the whole segment's row count. A
    /// kind that returns exactly `want` rows whenever it can — nearly
    /// always far below the segment's total — so comparing against the
    /// segment size declared almost every round "saturated" and jumped
    /// straight to `usize::MAX` (every row) on the very next widening,
    /// skipping the intended `DEFAULT_CURSOR_WIDEN` growth entirely.
    #[tokio::test]
    async fn default_cursor_grows_by_widen_before_jumping_to_every_row() {
        /// Returns exactly `probe.want` rows whenever the segment has that
        /// many: never "saturated" by the whole-row-count
        /// comparison this test is pinning against. Records every `want`
        /// it was asked for, so the test can check the growth *sequence*
        /// directly instead of inferring it from candidates that happen
        /// to come out the same either way.
        struct ReturnsExactlyWhatWasAsked {
            wants_seen: std::sync::Mutex<Vec<usize>>,
        }
        impl Algorithm for ReturnsExactlyWhatWasAsked {
            fn search<'a>(&'a self, probe: &'a Probe<'_>) -> BoxFuture<'a, Result<Vec<Candidate>>> {
                self.wants_seen.lock().unwrap().push(probe.want);
                let n = (probe.want as u64).min(probe.rows);
                let found = (0..n)
                    .map(|row| Candidate {
                        row: row as u32,
                        distance: row as f32,
                    })
                    .collect();
                async move { Ok(found) }.boxed()
            }
        }

        let store = InMemory::new();
        let budget = ReadBudget::unlimited();
        let policy = FetchPolicy::local();
        let path = Path::from("x");
        let algo = ReturnsExactlyWhatWasAsked {
            wants_seen: std::sync::Mutex::new(Vec::new()),
        };
        let probe = Probe {
            store: &store,
            budget: &budget,
            policy: &policy,
            path: &path,
            dim: 2,
            rows: 100_000, // far bigger than any `want` this test asks for
            data_size: 0,
            metric: crate::document::Metric::L2,
            query: &[0.0, 0.0],
            want: 1,
            nprobe: None,
            decoded: None,
        };
        let mut cursor = algo.cursor(probe).await.expect("cursor");

        let batch1 = cursor.next(1).await.expect("next");
        assert_eq!(batch1.candidates.len(), 1);

        // Ask for more than has been scanned so far: if `refill` wrongly
        // declared round 1 "saturated" (comparing `found.len()` against
        // the 100,000-row segment instead of against the 1 actually
        // asked for), `want` would jump straight to `usize::MAX` on this
        // very next call. It must instead grow by `DEFAULT_CURSOR_WIDEN`
        // (8x) each round: 1, then 8, then 64 — never `usize::MAX` this
        // early.
        let _ = cursor.next(9).await.expect("next");
        assert_eq!(
            *algo.wants_seen.lock().unwrap(),
            vec![1, 8, 64],
            "must widen by 8x each round, not jump to usize::MAX \
             (every row of a 100,000-row segment) this early"
        );
    }
}
