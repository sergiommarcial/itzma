//! The SQ8 index kind: scalar quantization to one byte per
//! dimension.
//!
//! Each dimension has its own range, from the smallest value in the segment to
//! the largest, split into 255 steps. A row is stored as one byte per
//! dimension and reconstructed as `min + code * scale`, which is off by at most
//! half a step in each dimension. A query stays in f32 and is compared with the
//! reconstruction on the fly (an asymmetric distance), so the scan reads a
//! quarter of the bytes of the exact vectors. The result is approximate, and the
//! exact rerank against the data object is what makes it right.
//!
//! An index has one or more partitions. With one (`nlist = 1`) every row is in
//! it and a query scans every code. With more, a seeded k-means (see
//! [`kmeans`]) puts each row in the partition of its nearest centroid, and a
//! query reads only the `nprobe` partitions whose centroids are nearest to it
//!. The format is the same either way: `nlist`, the centroids and a
//! partition table.
//!
//! # Layout after the prefix
//!
//! All integers are little-endian. `head_len` in the prefix is the end of the
//! table.
//!
//! | Section | Contents |
//! |---|---|
//! | `nlist` | u32, at least 1 |
//! | parameters | per dimension, its minimum then its scale, both f32 |
//! | centroids | `nlist * dim` f32 |
//! | partition table | per partition, its row count u32 and its byte offset u64 |
//! | partitions | per partition, the row numbers as u32 (strictly ascending, each below `rows`), then `count * dim` u8 codes |

use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};
use std::ops::Range;
use std::sync::Arc;

use bytes::Bytes;
use futures::future::BoxFuture;

use crate::cpu::{self, Cost, Lane};

use super::{
    Algorithm, Batch, Candidate, Cursor, Kind, PREFIX_LEN, Prefix, Probe, Reader, corrupt, missing,
    open,
};
use crate::decoded::What;
use crate::document::Metric;
use crate::error::{ItzmaError, Result};
use crate::quant::Sq8Query;
use crate::search;

use super::kmeans;

/// Rows whose distances [`Partition::scan`] computes before selecting from
/// them: small enough to stay in L1, large enough that the loop is long.
const SCAN_CHUNK: usize = 256;

/// A table entry: the row count and the byte offset.
const ENTRY_LEN: u64 = 4 + 8;

/// What the head of an SQ8 index says. Small: it is `8 * dim` bytes of
/// parameters, the centroids and the table, so a query reads it whole.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Head {
    dim: usize,
    rows: u32,
    codec: crate::quant::Sq8Codec,
    centroids: Vec<f32>,
    partitions: Vec<Entry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Entry {
    rows: u32,
    offset: u64,
}

impl Entry {
    /// How many bytes the partition takes: its row numbers and its codes.
    fn len(self, dim: usize) -> Option<u64> {
        u64::from(self.rows).checked_mul(4 + dim as u64)
    }
}

/// Below this many rows an index has one partition. A recall sweep held
/// recall at 0.987 or more from 100,000 rows up, and at 20,000 GloVe-25 fell to
/// 0.967 for a saving of only 2.4x in bytes, since a scan of that many codes is
/// already small.
const AUTO_MIN_ROWS: usize = 100_000;

/// How many partitions an index has when the caller does not say: one for a
/// small segment, and about the square root of the rows above that. The recall
/// sweep found no gain in more (4x the square root took six times as
/// long to build for the same recall and bytes per query) and a loss in fewer.
pub(crate) fn default_nlist(rows: usize) -> usize {
    if rows < AUTO_MIN_ROWS {
        return 1;
    }
    ((rows as f64).sqrt().round() as usize).clamp(1, rows)
}

/// How many partitions a query probes when it does not say: twice the square
/// root of `nlist`. A recall sweep put recall@10 at 0.96 to 0.98 at the
/// square root itself and at 0.99 or more at twice it, on SIFT1M and GloVe-25.
pub(crate) fn default_nprobe(nlist: usize) -> usize {
    (2.0 * (nlist as f64).sqrt()).ceil() as usize
}

/// Build the index object for a segment of `rows` vectors of length `dim`,
/// given as one row-major matrix, whose data object is `data_size` bytes, with
/// `nlist` partitions (clamped to `1..=rows`). One partition has no clustering,
/// and more run a seeded k-means under `metric` (see [`kmeans`]). The bytes
/// depend only on the input.
///
/// # Errors
///
/// [`ItzmaError::CorruptIndex`] (labelled `"(encoding)"`) if the matrix does
/// not have `rows * dim` values, holds a value that is not finite, or `rows`
/// does not fit the format's u32.
pub(crate) fn build(
    matrix: &[f32],
    dim: usize,
    rows: usize,
    data_size: u64,
    nlist: usize,
    metric: Metric,
) -> Result<Bytes> {
    let label = "(encoding)";
    let dim32 = u32::try_from(dim)
        .ok()
        .filter(|d| *d > 0)
        .ok_or_else(|| corrupt(label, "dimension must be between 1 and u32::MAX"))?;
    let rows32 = u32::try_from(rows)
        .map_err(|_| corrupt(label, "a segment of 2^32 rows or more cannot be indexed"))?;
    if rows.checked_mul(dim) != Some(matrix.len()) {
        return Err(corrupt(label, "the matrix is not rows * dim values"));
    }
    if matrix.iter().any(|x| !x.is_finite()) {
        return Err(corrupt(label, "a vector holds a value that is not finite"));
    }

    let codec = crate::quant::Sq8Codec::fit(matrix, dim)?;
    let nlist = nlist.clamp(1, rows.max(1));

    // One partition keeps the mean of the raw vectors as its centroid and every
    // row in order. More cluster the rows and give each cluster its own.
    let (centroids, members): (Vec<f32>, Vec<Vec<u32>>) = if nlist == 1 {
        (mean_of(matrix, dim, rows), vec![(0..rows32).collect()])
    } else {
        let clustering = kmeans::cluster(matrix, dim, rows, nlist, metric);
        let mut members = vec![Vec::new(); nlist];
        // Rows are visited in order, so each partition's list is ascending.
        for (row, cluster) in clustering.assignment.iter().enumerate() {
            members[*cluster as usize].push(row as u32);
        }
        (clustering.centroids, members)
    };
    let nlist32 = members.len() as u32;

    let head_len = PREFIX_LEN as u64 + head_body_len(dim as u64, u64::from(nlist32));
    let prefix = Prefix {
        kind: Kind::Sq8,
        dim: dim32,
        rows: rows32,
        data_size,
        head_len,
    };

    let mut out = Vec::with_capacity(head_len as usize + rows * (4 + dim));
    out.extend_from_slice(&prefix.encode());
    out.extend_from_slice(&nlist32.to_le_bytes());
    for d in 0..dim {
        out.extend_from_slice(&codec.min()[d].to_le_bytes());
        out.extend_from_slice(&codec.scale()[d].to_le_bytes());
    }
    for x in &centroids {
        out.extend_from_slice(&x.to_le_bytes());
    }
    let mut offset = head_len;
    for list in &members {
        let count = list.len() as u32;
        out.extend_from_slice(&count.to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
        offset += u64::from(count) * (4 + dim as u64);
    }
    for list in &members {
        for row in list {
            out.extend_from_slice(&row.to_le_bytes());
        }
        // In chunks, in parallel on a pool thread, appended in order.
        for codes in cpu::chunked(list, kmeans::ASSIGN_CHUNK, |rows| {
            let mut codes = Vec::with_capacity(rows.len() * dim);
            for row in rows {
                let at = *row as usize * dim;
                codes.extend(codec.encode(&matrix[at..at + dim]));
            }
            codes
        }) {
            out.extend(codes);
        }
    }
    Ok(Bytes::from(out))
}

/// The mean of the rows, zero for no rows.
fn mean_of(matrix: &[f32], dim: usize, rows: usize) -> Vec<f32> {
    let mut sum = vec![0f64; dim];
    for row in matrix.chunks_exact(dim) {
        for (d, x) in row.iter().enumerate() {
            sum[d] += f64::from(*x);
        }
    }
    sum.iter()
        .map(|s| {
            if rows == 0 {
                0.0
            } else {
                (s / rows as f64) as f32
            }
        })
        .collect()
}

/// The bytes of the head after the prefix, for `nlist` partitions.
fn head_body_len(dim: u64, nlist: u64) -> u64 {
    // Saturating, since a forged `nlist` and `dim` must not overflow before the
    // length is compared with the bytes.
    let centroids = dim.saturating_mul(nlist).saturating_mul(4);
    4u64.saturating_add(dim.saturating_mul(8))
        .saturating_add(centroids)
        .saturating_add(ENTRY_LEN.saturating_mul(nlist))
}

impl Head {
    /// Read the head from `body`, the bytes between the prefix and `head_len`.
    /// Nothing is allocated before the bytes are known to hold what `nlist` and
    /// `dim` say, so a forged count costs nothing.
    ///
    /// # Errors
    ///
    /// [`ItzmaError::CorruptIndex`] if the length disagrees with `nlist` and
    /// `dim`, a parameter or centroid is not finite, a scale is negative, the
    /// partition table does not start at `head_len` or does not chain, or the
    /// row counts do not add up to `rows`.
    pub(crate) fn parse(path: &str, prefix: &Prefix, body: &[u8]) -> Result<Head> {
        #[cfg(test)]
        HEAD_PARSES.with(|count| count.set(count.get() + 1));
        let dim = prefix.dim as usize;
        let mut r = Reader::new(path, body);
        let nlist = r.u32()?;
        if nlist == 0 {
            return Err(corrupt(path, "nlist is 0"));
        }
        let want = head_body_len(u64::from(prefix.dim), u64::from(nlist));
        if want != body.len() as u64 {
            return Err(corrupt(
                path,
                "the head's length disagrees with nlist and the dimension",
            ));
        }

        let mut min = Vec::with_capacity(dim);
        let mut scale = Vec::with_capacity(dim);
        for _ in 0..dim {
            let (lo, step) = (r.f32()?, r.f32()?);
            if !lo.is_finite() || !step.is_finite() || step < 0.0 {
                return Err(corrupt(
                    path,
                    "a parameter is not finite, or a scale is negative",
                ));
            }
            min.push(lo);
            scale.push(step);
        }
        let mut centroids = Vec::with_capacity(dim * nlist as usize);
        for _ in 0..dim * nlist as usize {
            let x = r.f32()?;
            if !x.is_finite() {
                return Err(corrupt(path, "a centroid is not finite"));
            }
            centroids.push(x);
        }

        let mut partitions = Vec::with_capacity(nlist as usize);
        let mut next = prefix.head_len;
        let mut total = 0u64;
        for _ in 0..nlist {
            let entry = Entry {
                rows: r.u32()?,
                offset: r.u64()?,
            };
            if entry.offset != next {
                return Err(corrupt(
                    path,
                    "partitions do not follow each other from the end of the head",
                ));
            }
            next = entry
                .len(dim)
                .and_then(|len| next.checked_add(len))
                .ok_or_else(|| corrupt(path, "a partition runs past the largest offset"))?;
            total += u64::from(entry.rows);
            partitions.push(entry);
        }
        if total != u64::from(prefix.rows) {
            return Err(corrupt(
                path,
                "the partitions' row counts do not add up to the segment's",
            ));
        }
        Ok(Head {
            dim,
            rows: prefix.rows,
            codec: crate::quant::Sq8Codec::from_parts(min, scale),
            centroids,
            partitions,
        })
    }

    /// About how many bytes the parsed head holds, for the decoded cache.
    fn weight(&self) -> u64 {
        (self.dim * 8 + self.centroids.len() * 4 + self.partitions.len() * 16) as u64
    }

    /// How many partitions there are.
    pub(crate) fn nlist(&self) -> usize {
        self.partitions.len()
    }

    /// This index's L2 error bound, a thin wrapper over the
    /// codec's own, for a caller (`Sq8Cursor`) that only has the `Head`.
    pub(crate) fn codec_l2_error_bound(&self) -> f32 {
        self.codec.l2_error_bound()
    }

    /// The centroid of partition `i`.
    pub(crate) fn centroid(&self, i: usize) -> Option<&[f32]> {
        self.centroids.get(i * self.dim..(i + 1) * self.dim)
    }

    /// The partitions to scan for `query`: the `nprobe` whose centroids are
    /// closest under `metric` (at least one), nearest first. A head with no more
    /// partitions than that gives them all, in order.
    fn nearest(&self, metric: Metric, query: &[f32], nprobe: usize) -> Vec<usize> {
        let probes = nprobe.max(1);
        if self.nlist() <= probes {
            return (0..self.nlist()).collect();
        }
        let mut ranked: Vec<(f32, usize)> = (0..self.nlist())
            .filter_map(|i| Some((search::distance(metric, query, self.centroid(i)?), i)))
            .collect();
        ranked.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        ranked.into_iter().take(probes).map(|(_, i)| i).collect()
    }

    /// Where partition `i` is in the object, for a ranged read.
    pub(crate) fn partition_range(&self, i: usize) -> Option<Range<u64>> {
        let entry = self.partitions.get(i)?;
        Some(entry.offset..entry.offset.checked_add(entry.len(self.dim)?)?)
    }

    /// Where the last partition ends, which is the size of the whole object.
    fn end(&self) -> u64 {
        self.partition_range(self.nlist().saturating_sub(1))
            .map_or(0, |range| range.end)
    }

    /// Reconstruct a code for tests and for comparison with exact vectors.
    #[cfg(test)]
    fn dequantize(&self, code: &[u8]) -> Vec<f32> {
        self.codec.decode(code)
    }

    /// Prepare `query` for scanning with this head's parameters.
    ///
    /// # Errors
    ///
    /// [`ItzmaError::DimensionMismatch`] if the query is not `dim` long.
    pub(crate) fn prepare(&self, metric: Metric, query: &[f32]) -> Result<Sq8Query> {
        if query.len() != self.dim {
            return Err(ItzmaError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        Ok(self.codec.prepare(metric, query))
    }
}

#[cfg(test)]
thread_local! {
    /// How many heads this thread has parsed, so a test can tell a cached head
    /// from a parsed one.
    pub(crate) static HEAD_PARSES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// A parsed head and the size of its object, checked against each other: what
/// a search keeps of an index between its reads, and what the decoded cache
/// keeps between searches.
#[derive(Debug)]
pub(crate) struct OpenedHead {
    head: Head,
    size: u64,
}

/// The head of the probe's index, from the decoded cache if it has it (no
/// request, no charge), else read and parsed.
async fn open_head(probe: &Probe<'_>) -> Result<Arc<OpenedHead>> {
    let load = || async {
        let name = probe.path.as_ref();
        let opened = open(probe, Kind::Sq8).await?;
        let head = Head::parse(name, &opened.prefix, &opened.body)?;
        if head.end() != opened.size {
            return Err(corrupt(
                name,
                "the object does not end where its last partition does",
            ));
        }
        let weight = head.weight();
        Ok((
            OpenedHead {
                head,
                size: opened.size,
            },
            weight,
        ))
    };
    match probe.decoded {
        Some(decoded) => decoded.get_or_load(probe.path, What::Sq8Head, load).await,
        None => Ok(Arc::new(load().await?.0)),
    }
}

/// Partitions `wanted` of the probe's index, in that order. The ones the
/// decoded cache does not have are fetched together through the planner
///, so nearby partitions can share a request, and each is charged
/// before it is sent.
async fn read_partitions(
    probe: &Probe<'_>,
    opened: &OpenedHead,
    wanted: &[usize],
) -> Result<Vec<Arc<Partition>>> {
    let name = probe.path.as_ref();
    let load = |positions: Vec<usize>| async move {
        let picked: Vec<usize> = positions.iter().map(|&at| wanted[at]).collect();
        let ranges: Vec<Range<u64>> = picked
            .iter()
            .map(|&i| {
                opened
                    .head
                    .partition_range(i)
                    .ok_or_else(|| corrupt(name, format!("there is no partition {i}")))
            })
            .collect::<Result<_>>()?;
        let planned = crate::fetch::plan(probe.policy, &ranges, opened.size);
        let slices = crate::fetch::execute(probe.store, probe.budget, probe.path, planned)
            .await
            .map_err(|e| match e {
                ItzmaError::Storage(object_store::Error::NotFound { .. }) => missing(probe.path),
                other => other,
            })?;
        picked
            .iter()
            .zip(slices)
            .map(|(&i, slice)| {
                // A kept partition gets a buffer of its own: a view would pin
                // the whole merged fetch it came from, beyond its weight.
                let slice = match probe.decoded {
                    Some(_) => Bytes::copy_from_slice(&slice),
                    None => slice,
                };
                let partition = Partition::decode(name, &opened.head, i, slice)?;
                let weight = partition.weight();
                Ok((partition, weight))
            })
            .collect::<Result<Vec<_>>>()
    };
    match probe.decoded {
        Some(decoded) => {
            let whats: Vec<What> = wanted.iter().map(|&i| What::Partition(i as u32)).collect();
            decoded.get_or_load_many(probe.path, &whats, load).await
        }
        None => Ok(load((0..wanted.len()).collect())
            .await?
            .into_iter()
            .map(|(partition, _)| Arc::new(partition))
            .collect()),
    }
}

/// One partition, read in place: views of the bytes it was decoded from, the
/// row numbers (u32 each) and their codes, so a cached partition
/// costs its bytes once and a scan copies nothing.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Partition {
    dim: usize,
    count: usize,
    rows: Bytes,
    codes: Bytes,
}

/// The row number at the front of `bytes`, which is 4 bytes or more.
fn row_at(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

impl Partition {
    /// Read partition `index` of `head` from `bytes`, which must be exactly its
    /// range in the object. The row numbers are checked here, once.
    ///
    /// # Errors
    ///
    /// [`ItzmaError::CorruptIndex`] if there is no such partition, the length is
    /// not `count * (4 + dim)`, or the row numbers are not strictly ascending
    /// and below the segment's row count.
    pub(crate) fn decode(path: &str, head: &Head, index: usize, bytes: Bytes) -> Result<Self> {
        let entry = head
            .partitions
            .get(index)
            .ok_or_else(|| corrupt(path, format!("there is no partition {index}")))?;
        if entry.len(head.dim) != Some(bytes.len() as u64) {
            return Err(corrupt(
                path,
                format!("partition {index} is not the length its table entry says"),
            ));
        }
        let count = entry.rows as usize;
        let rows = bytes.slice(..count * 4);
        let codes = bytes.slice(count * 4..);
        let mut previous: Option<u32> = None;
        for row in rows.chunks_exact(4).map(row_at) {
            if previous.is_some_and(|p| p >= row) {
                return Err(corrupt(
                    path,
                    format!("partition {index} has row numbers out of order"),
                ));
            }
            previous = Some(row);
        }
        if previous.is_some_and(|last| last >= head.rows) {
            return Err(corrupt(
                path,
                format!("partition {index} has a row number past the segment's rows"),
            ));
        }
        Ok(Partition {
            dim: head.dim,
            count,
            rows,
            codes,
        })
    }

    /// How many rows the partition holds.
    pub(crate) fn len(&self) -> usize {
        self.count
    }

    /// The row numbers, decoded.
    #[cfg(test)]
    fn row_numbers(&self) -> Vec<u32> {
        self.rows.chunks_exact(4).map(row_at).collect()
    }

    /// How many bytes the partition keeps alive, for the decoded cache.
    pub(crate) fn weight(&self) -> u64 {
        (self.rows.len() + self.codes.len()) as u64
    }

    /// The `want` closest rows to the query by approximate distance, closest
    /// first, ties by row number.
    ///
    /// A partition over [`SPLIT_ROWS`] rows is scanned in ranges of that many,
    /// in parallel on a pool thread; each range keeps its closest
    /// `want`, and the merge sorts by (distance, row), so the answer is the
    /// one a single pass gives.
    pub(crate) fn scan(&self, query: &Sq8Query, want: usize) -> Vec<Candidate> {
        if want == 0 {
            return Vec::new();
        }
        if !self.worth_splitting() {
            return self.scan_range(query, want, 0..self.len());
        }
        let mut found: Vec<Candidate> = self
            .split(|range| self.scan_range(query, want, range))
            .into_iter()
            .flatten()
            .collect();
        found.sort_by(|a, b| a.distance.total_cmp(&b.distance).then(a.row.cmp(&b.row)));
        found.truncate(want);
        found
    }

    /// [`scan`](Self::scan) over the rows `range` only: distances computed
    /// [`SCAN_CHUNK`] rows at a time into a buffer, a loop with no branches,
    /// and then selected: a row enters the heap only when it beats the
    /// current worst.
    fn scan_range(&self, query: &Sq8Query, want: usize, range: Range<usize>) -> Vec<Candidate> {
        let mut heap: BinaryHeap<Ranked> = BinaryHeap::with_capacity(want.min(range.len()) + 1);
        let mut distances: Vec<f32> = Vec::with_capacity(SCAN_CHUNK);
        let (rows, codes) = self.slices(range);
        for (rows, codes) in rows
            .chunks(SCAN_CHUNK * 4)
            .zip(codes.chunks(SCAN_CHUNK * self.dim))
        {
            distances.clear();
            distances.extend(
                codes
                    .chunks_exact(self.dim)
                    .map(|code| query.distance(code)),
            );
            for (row, distance) in rows.chunks_exact(4).map(row_at).zip(&distances) {
                let ranked = Ranked(Candidate {
                    row,
                    distance: *distance,
                });
                if heap.len() < want {
                    heap.push(ranked);
                } else if heap.peek().is_some_and(|worst| ranked < *worst) {
                    heap.pop();
                    heap.push(ranked);
                }
            }
        }
        heap.into_sorted_vec().into_iter().map(|r| r.0).collect()
    }

    /// Every row's approximate distance, in row order, no selection and no
    /// sort — `O(partition.len())`, not `O(partition.len() * log
    /// partition.len())`. For a caller that is going to keep every row
    /// regardless (`Sq8Cursor::fetch_wave`, which must retain a whole
    /// probed partition since it is read at most once ever), `scan`'s
    /// bounded heap earns nothing: asking it for `want = self.len()` makes
    /// every row "beat the current worst" and pushes it through the heap
    /// anyway, which costs the full `O(n log n)` it exists to avoid. Split
    /// like [`scan`](Self::scan) and joined in row order.
    pub(crate) fn scan_all(&self, query: &Sq8Query) -> Vec<Candidate> {
        if !self.worth_splitting() {
            return self.scan_all_range(query, 0..self.len());
        }
        let parts = self.split(|range| self.scan_all_range(query, range));
        let mut out = Vec::with_capacity(self.len());
        for part in parts {
            out.extend(part);
        }
        out
    }

    fn scan_all_range(&self, query: &Sq8Query, range: Range<usize>) -> Vec<Candidate> {
        let mut out = Vec::with_capacity(range.len());
        let mut distances: Vec<f32> = Vec::with_capacity(SCAN_CHUNK);
        let (rows, codes) = self.slices(range);
        for (rows, codes) in rows
            .chunks(SCAN_CHUNK * 4)
            .zip(codes.chunks(SCAN_CHUNK * self.dim))
        {
            distances.clear();
            distances.extend(
                codes
                    .chunks_exact(self.dim)
                    .map(|code| query.distance(code)),
            );
            out.extend(
                rows.chunks_exact(4)
                    .map(row_at)
                    .zip(&distances)
                    .map(|(row, &distance)| Candidate { row, distance }),
            );
        }
        out
    }

    /// Split only a partition bigger than one range, and only on a pool
    /// thread: run inline, the ranges would go one after another and the
    /// split would cost a copy for nothing.
    fn worth_splitting(&self) -> bool {
        self.len() > SPLIT_ROWS && rayon::current_thread_index().is_some()
    }

    /// The row numbers and codes of rows `range`.
    fn slices(&self, range: Range<usize>) -> (&[u8], &[u8]) {
        (
            &self.rows[range.start * 4..range.end * 4],
            &self.codes[range.start * self.dim..range.end * self.dim],
        )
    }

    /// `f` over the partition's rows in ranges of [`SPLIT_ROWS`], results in
    /// range order.
    fn split<R: Send>(&self, f: impl Fn(Range<usize>) -> R + Sync) -> Vec<R> {
        let starts: Vec<usize> = (0..self.len()).step_by(SPLIT_ROWS).collect();
        cpu::chunked(&starts, 1, |one| {
            let start = one.first().copied().unwrap_or(0);
            f(start..(start + SPLIT_ROWS).min(self.len()))
        })
    }
}

/// Rows per range when one partition is split across pool threads: a
/// constant, so the answer never depends on the thread count.
const SPLIT_ROWS: usize = 16_384;

/// Heap entry ordered by (distance, row), so the worst candidate is on top.
#[derive(Debug, Clone, Copy)]
struct Ranked(Candidate);

impl PartialEq for Ranked {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Ranked {}

impl PartialOrd for Ranked {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Ranked {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .distance
            .total_cmp(&other.0.distance)
            .then_with(|| self.0.row.cmp(&other.0.row))
    }
}

/// The SQ8 kind, as the query engine calls it.
pub(crate) struct Sq8;

impl Algorithm for Sq8 {
    fn search<'a>(&'a self, probe: &'a Probe<'_>) -> BoxFuture<'a, Result<Vec<Candidate>>> {
        Box::pin(scan_object(probe))
    }

    // `Sq8` stays on the trait's default `lower_bound` (always `None`): the
    // method's `(probe, approx)` signature has no cheap way to reach the
    // already-parsed `Head`/`Sq8Query` a real bound needs, short of
    // re-fetching and re-parsing the index object per candidate, which would
    // cost more than the pruning saves. `sq8_lower_bound` below is what the
    // cursor calls directly instead, once it holds those parsed once per
    // source-open.

    fn cursor<'a>(&'a self, probe: Probe<'a>) -> BoxFuture<'a, Result<Box<dyn Cursor + 'a>>> {
        Box::pin(async move { Ok(Box::new(Sq8Cursor::open(probe).await?) as Box<dyn Cursor>) })
    }
}

/// SQ8's own cursor: opens the index once, ranks every partition
/// by centroid distance (not only the `nprobe` nearest), and fetches/scans
/// them in waves of `nprobe` as the engine asks for more, sorted by distance
/// within each returned batch. Waves are visited nearest-centroid-first, but
/// that only orders *partitions*, not individual rows, so a later wave can
/// still hold a row closer than one an earlier batch already returned —
/// true cross-wave best-first order needs per-partition distance bounds
///, not built here. Replaces, for SQ8 specifically, the default
/// cursor's "re-run `search` with a bigger total" behavior: a wave is
/// fetched and scanned exactly once, ever.
struct Sq8Cursor<'a> {
    probe: Probe<'a>,
    /// The head and the index object's own size (not `probe.data_size`, the
    /// segment's), which the fetch planner must never merge or
    /// align a wave's partition ranges past.
    opened: Arc<OpenedHead>,
    prepared: Arc<Sq8Query>,
    l2_error: f32,
    dot_error: f32,
    /// Partition indices, nearest centroid first, not yet fetched.
    remaining_partitions: VecDeque<usize>,
    nprobe: usize,
    /// Not-yet-returned candidates, in no particular order between calls to
    /// `next`. Rows leave this vector the moment they are handed to a
    /// caller (drained, not just counted past): centroid order ranks
    /// *partitions*, not individual rows, so a later wave can still contain
    /// a row closer than one a caller already consumed from an earlier
    /// wave. Kept unsorted on purpose — a wave's rows are appended as found,
    /// and `next` selects (not sorts) only the prefix it is about to drain,
    /// so a caller that stays within one small `top_k` never pays to order
    /// rows it will never ask for (the earlier version sorted the whole
    /// buffer after every wave, `O(n log n)` regardless of how much of it
    /// any call ever drained — a real, measured regression).
    buffered: Vec<Candidate>,
}

impl<'a> Sq8Cursor<'a> {
    async fn open(probe: Probe<'a>) -> Result<Self> {
        let opened = open_head(&probe).await?;
        let head = &opened.head;
        let prepared = Arc::new(head.prepare(probe.metric, probe.query)?);
        let l2_error = head.codec_l2_error_bound();
        let dot_error = prepared.dot_error_bound();
        let nprobe = probe.nprobe.unwrap_or_else(|| default_nprobe(head.nlist()));
        // Not `head.nearest(metric, query, head.nlist())`: when `nprobe >=
        // nlist`, `Head::nearest` takes its "give me all of them" shortcut
        // and returns partitions in plain index order, not ranked — correct
        // for `scan_object`, which immediately re-sorts for coalescing and
        // doesn't care about probe order since it scans every one of them
        // regardless. The cursor's waves *do* care: the whole point of
        // waves is to probe the nearest partitions first, so every
        // partition is ranked here directly (this crashed recall to ~22%
        // the first time, caught by
        // `the_default_nprobe_on_a_clustered_index_keeps_recall`).
        let mut ranked: Vec<(f32, usize)> = (0..head.nlist())
            .filter_map(|i| {
                Some((
                    search::distance(probe.metric, probe.query, head.centroid(i)?),
                    i,
                ))
            })
            .collect();
        ranked.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let remaining_partitions: VecDeque<usize> = ranked.into_iter().map(|(_, i)| i).collect();
        Ok(Sq8Cursor {
            probe,
            opened,
            prepared,
            l2_error,
            dot_error,
            remaining_partitions,
            nprobe,
            buffered: Vec::new(),
        })
    }

    /// Fetch and scan the next wave of up to `nprobe` partitions, planned
    /// through the shared fetch planner (nearby partitions share
    /// a request under a merging policy, not just ones that exactly
    /// touch), merging their scored rows into `buffered`.
    async fn fetch_wave(&mut self) -> Result<()> {
        let mut wave: Vec<usize> = Vec::new();
        for _ in 0..self.nprobe {
            match self.remaining_partitions.pop_front() {
                Some(i) => wave.push(i),
                None => break,
            }
        }
        if wave.is_empty() {
            return Ok(());
        }
        let partitions = read_partitions(&self.probe, &self.opened, &wave).await?;
        let scanned = scan_wave_on_pool(partitions, self.prepared.clone(), None).await;
        if self.buffered.is_empty() {
            self.buffered = scanned;
        } else {
            self.buffered.extend(scanned);
        }
        Ok(())
    }
}

impl Cursor for Sq8Cursor<'_> {
    fn next<'a>(&'a mut self, n: usize) -> BoxFuture<'a, Result<Batch>> {
        Box::pin(async move {
            while self.buffered.len() < n && !self.remaining_partitions.is_empty() {
                self.fetch_wave().await?;
            }
            let take = n.min(self.buffered.len());
            let by_distance_then_row = |a: &Candidate, b: &Candidate| {
                a.distance.total_cmp(&b.distance).then(a.row.cmp(&b.row))
            };
            // `buffered` is not kept sorted (see its doc comment): select
            // just the `take` closest rows to the front — `O(len)`, not
            // `O(len log len)` — then sort only that small prefix. Nothing
            // past `take` needs an order yet, and may never.
            if take > 0 && take < self.buffered.len() {
                self.buffered
                    .select_nth_unstable_by(take - 1, by_distance_then_row);
            }
            self.buffered[..take].sort_by(by_distance_then_row);
            let candidates: Vec<Candidate> = self.buffered.drain(0..take).collect();
            let exhausted = self.remaining_partitions.is_empty() && self.buffered.is_empty();
            // While any partition is still unscanned, `rest_lower_bound`
            // must be `None`, never a bound derived only from `buffered`:
            // a plain `sq8` index (not the `sq8-ball`) has no
            // per-partition distance bound, so an unscanned partition
            // could hold a row closer than everything already buffered.
            // Reporting a bound derived from an arbitrary (not necessarily
            // closest) remaining candidate as "the rest" would let the
            // engine stop this source early and drop a real match. Found
            // by a linear scan, not `buffered.first()` — `buffered` is
            // unsorted, so its first element is not reliably its closest
            // ("an sq8 index uses 0 for unscanned partitions,
            // meaning 'no bound'").
            let rest_lower_bound = if exhausted {
                Some(f32::INFINITY)
            } else if !self.remaining_partitions.is_empty() {
                None
            } else {
                self.buffered
                    .iter()
                    .min_by(|a, b| a.distance.total_cmp(&b.distance))
                    .and_then(|c| {
                        sq8_lower_bound(
                            self.probe.metric,
                            self.l2_error,
                            self.dot_error,
                            c.distance,
                        )
                    })
            };
            Ok(Batch {
                candidates,
                rest_lower_bound,
                exhausted,
            })
        })
    }
}

/// A lower bound on the exact distance of a row SQ8 scored at `approx` under
/// `metric`, given the error terms a scan already has in hand — or `None`
/// for cosine, which needs the reconstructed row's own norm (not carried by
/// `approx` alone) and is not built yet.
pub(crate) fn sq8_lower_bound(
    metric: Metric,
    l2_error: f32,
    dot_error: f32,
    approx: f32,
) -> Option<f32> {
    match metric {
        Metric::L2 => Some((approx.sqrt() - l2_error).max(0.0).powi(2)),
        Metric::Dot => Some(approx - dot_error),
        Metric::Cosine => None,
    }
}

/// Read the head, then each probed partition, and return the closest rows by
/// approximate distance. Each read is charged before it is sent.
async fn scan_object(probe: &Probe<'_>) -> Result<Vec<Candidate>> {
    let opened = open_head(probe).await?;
    let head = &opened.head;
    let prepared = head.prepare(probe.metric, probe.query)?;

    // Probed partitions, planned: nearby partitions can share a
    // request under a merging policy, not only ones that exactly touch.
    let nprobe = probe.nprobe.unwrap_or_else(|| default_nprobe(head.nlist()));
    let mut wanted_partitions = head.nearest(probe.metric, probe.query, nprobe);
    wanted_partitions.sort_unstable();

    let partitions = read_partitions(probe, &opened, &wanted_partitions).await?;
    let mut found = scan_wave_on_pool(partitions, Arc::new(prepared), Some(probe.want)).await;
    found.sort_by(|a, b| a.distance.total_cmp(&b.distance).then(a.row.cmp(&b.row)));
    found.truncate(probe.want);
    Ok(found)
}

/// Scan a wave of partitions, in their order: every row of each (`want:
/// None`) or each one's closest `want`. On a pool thread the partitions are
/// scanned in parallel ([`cpu::chunked`]), results still in their order.
fn scan_wave(
    partitions: &[Arc<Partition>],
    query: &Sq8Query,
    want: Option<usize>,
) -> Vec<Candidate> {
    let mut scanned = cpu::chunked(partitions, 1, |one| {
        one.first().map_or_else(Vec::new, |p| match want {
            None => p.scan_all(query),
            Some(w) => p.scan(query, w),
        })
    });
    // Moved, not copied, when there is one partition: a whole-segment wave
    // can be hundreds of thousands of candidates.
    if scanned.len() == 1 {
        return scanned.pop().unwrap_or_default();
    }
    let mut out = Vec::with_capacity(scanned.iter().map(Vec::len).sum());
    for part in scanned {
        out.extend(part);
    }
    out
}

/// [`scan_wave`] on the query lane, or inline when the wave is small
///: one hand-off per wave, not one per partition.
async fn scan_wave_on_pool(
    partitions: Vec<Arc<Partition>>,
    query: Arc<Sq8Query>,
    want: Option<usize>,
) -> Vec<Candidate> {
    let rows = partitions.iter().map(|p| p.len()).sum();
    let dim = partitions.first().map_or(0, |p| p.dim);
    cpu::run(Lane::Query, Cost::Distances { rows, dim }, move || {
        scan_wave(&partitions, &query, want)
    })
    .await
}

/// A whole index object, decoded, for tests and for `inspect`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Index {
    pub(crate) head: Head,
    pub(crate) partitions: Vec<Partition>,
}

impl Index {
    /// Decode a whole object built for a segment with this dimension, row count
    /// and data size, checking everything: the prefix against the segment, the
    /// head, every partition, and that the object ends where the last one does.
    pub(crate) fn decode(
        path: &str,
        bytes: &[u8],
        dim: usize,
        rows: u64,
        data_size: u64,
    ) -> Result<Index> {
        let prefix = Prefix::decode(path, bytes)?;
        prefix.check_segment(path, dim, rows, data_size)?;
        if prefix.kind != Kind::Sq8 {
            return Err(corrupt(path, "not an SQ8 index"));
        }
        let head_end = usize::try_from(prefix.head_len)
            .ok()
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| corrupt(path, "the head runs past the end of the object"))?;
        let head = Head::parse(path, &prefix, &bytes[PREFIX_LEN..head_end])?;
        if head.end() != bytes.len() as u64 {
            return Err(corrupt(
                path,
                "the object does not end where its last partition does",
            ));
        }
        let owned = Bytes::copy_from_slice(bytes);
        let partitions = (0..head.nlist())
            .map(|i| {
                let range = head
                    .partition_range(i)
                    .ok_or_else(|| corrupt(path, "a partition has no range"))?;
                let (start, end) = (range.start as usize, range.end as usize);
                if end > owned.len() || start > end {
                    return Err(corrupt(path, "a partition runs past the end of the object"));
                }
                Partition::decode(path, &head, i, owned.slice(start..end))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Index { head, partitions })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::is_missing;
    use crate::search::distance;
    use object_store::ObjectStoreExt;

    const PATH: &str = "idx";

    /// Deterministic values in [-1, 1).
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }

        fn matrix(&mut self, rows: usize, dim: usize) -> Vec<f32> {
            (0..rows * dim).map(|_| self.next()).collect()
        }
    }

    fn built(rows: usize, dim: usize, seed: u64) -> (Vec<f32>, Bytes) {
        let matrix = Lcg(seed).matrix(rows, dim);
        let bytes = build(&matrix, dim, rows, 999, 1, Metric::L2).expect("build");
        (matrix, bytes)
    }

    fn decoded(bytes: &[u8], dim: usize, rows: u64) -> Result<Index> {
        Index::decode(PATH, bytes, dim, rows, 999)
    }

    fn reason(err: ItzmaError) -> String {
        match err {
            ItzmaError::CorruptIndex { reason, .. } => reason,
            other => format!("{other:?}"),
        }
    }

    #[test]
    fn lower_bound_is_never_above_the_f64_reference_distance_for_l2_and_dot() {
        let (matrix, bytes) = built(300, 16, 5);
        let decoded = decoded(&bytes, 16, 300).expect("decode");
        let query = Lcg(999).matrix(1, 16);
        let l2_error = decoded.head.codec.l2_error_bound();
        for metric in [Metric::L2, Metric::Dot] {
            let prepared = decoded.head.prepare(metric, &query).expect("prepare");
            let dot_error = prepared.dot_error_bound();
            for (row, chunk) in matrix.chunks_exact(16).enumerate() {
                let code = &decoded.partitions[0].codes[row * 16..(row + 1) * 16];
                let approx = prepared.distance(code);
                let exact = distance(metric, &query, chunk);
                let lower = sq8_lower_bound(metric, l2_error, dot_error, approx).unwrap();
                assert!(
                    lower <= exact + 1e-2,
                    "{metric:?} row {row}: lower {lower} above exact {exact}"
                );
            }
        }
    }

    #[test]
    fn cosine_has_no_lower_bound_yet() {
        assert_eq!(sq8_lower_bound(Metric::Cosine, 0.1, 0.1, 0.5), None);
    }

    /// An in-memory store that counts every read, so a test can say how
    /// many requests a call made. A smaller cousin of `segment/ranged.rs`'s
    /// own `CountingStore` (not shared across modules in this codebase, see
    /// that file's doc comment) — these tests only need the count, not the
    /// range or byte size of each request.
    #[derive(Debug, Default)]
    struct CountingStore {
        inner: object_store::memory::InMemory,
        served: std::sync::atomic::AtomicUsize,
        /// Deletes `.0` from `inner` just before serving its `.1`-th GET
        /// (1-based), standing in for a `gc` race landing between two of a
        /// caller's own requests.
        delete_on_nth_get: std::sync::Mutex<Option<(object_store::path::Path, usize)>>,
    }

    impl CountingStore {
        fn reset(&self) {
            self.served.store(0, std::sync::atomic::Ordering::SeqCst);
        }

        fn requests(&self) -> usize {
            self.served.load(std::sync::atomic::Ordering::SeqCst)
        }

        fn delete_on_nth_get(&self, path: object_store::path::Path, n: usize) {
            *self.delete_on_nth_get.lock().unwrap() = Some((path, n));
        }
    }

    impl std::fmt::Display for CountingStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "CountingStore")
        }
    }

    #[async_trait::async_trait]
    impl object_store::ObjectStore for CountingStore {
        async fn put_opts(
            &self,
            location: &object_store::path::Path,
            payload: object_store::PutPayload,
            opts: object_store::PutOptions,
        ) -> object_store::Result<object_store::PutResult> {
            self.inner.put_opts(location, payload, opts).await
        }

        async fn put_multipart_opts(
            &self,
            location: &object_store::path::Path,
            opts: object_store::PutMultipartOptions,
        ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
            self.inner.put_multipart_opts(location, opts).await
        }

        async fn get_opts(
            &self,
            location: &object_store::path::Path,
            options: object_store::GetOptions,
        ) -> object_store::Result<object_store::GetResult> {
            let n = self.served.load(std::sync::atomic::Ordering::SeqCst) + 1;
            let should_delete = {
                let guard = self.delete_on_nth_get.lock().unwrap();
                matches!(&*guard, Some((path, at)) if *at == n && path == location)
            };
            if should_delete {
                let _ = self.inner.delete(location).await;
            }
            let got = self.inner.get_opts(location, options).await?;
            self.served
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(got)
        }

        fn delete_stream(
            &self,
            locations: futures::stream::BoxStream<
                'static,
                object_store::Result<object_store::path::Path>,
            >,
        ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::path::Path>>
        {
            self.inner.delete_stream(locations)
        }

        fn list(
            &self,
            prefix: Option<&object_store::path::Path>,
        ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>>
        {
            self.inner.list(prefix)
        }

        async fn list_with_delimiter(
            &self,
            prefix: Option<&object_store::path::Path>,
        ) -> object_store::Result<object_store::ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }

        async fn copy_opts(
            &self,
            from: &object_store::path::Path,
            to: &object_store::path::Path,
            options: object_store::CopyOptions,
        ) -> object_store::Result<()> {
            self.inner.copy_opts(from, to, options).await
        }
    }

    #[tokio::test]
    async fn the_cursor_returns_every_row_exactly_once_closest_first_within_each_wave() {
        use object_store::ObjectStoreExt as _;
        use object_store::memory::InMemory;
        use object_store::path::Path;

        // `built` always builds with `nlist = 1` (one partition, everything
        // in it), so a real multi-wave exercise needs `clustered_object`'s
        // explicit `nlist` instead.
        let (_matrix, bytes) = clustered_object(500, 16, 8, Metric::L2);
        let store = InMemory::new();
        let path = Path::from("idx");
        store.put(&path, bytes.into()).await.expect("put");
        let budget = crate::budget::ReadBudget::unlimited();
        let policy = crate::fetch::FetchPolicy::local();
        let query = Lcg(42).matrix(1, 16);
        let probe = Probe {
            store: &store,
            budget: &budget,
            policy: &policy,
            path: &path,
            dim: 16,
            rows: 500,
            data_size: 999,
            metric: Metric::L2,
            query: &query,
            want: 10,
            nprobe: Some(1),
            decoded: None,
        };
        let mut cursor = Sq8.cursor(probe).await.expect("cursor");
        let mut seen = std::collections::HashSet::new();
        loop {
            let batch = cursor.next(50).await.expect("next");
            // Sorted within this batch — but not necessarily monotonic
            // *across* batches: waves are visited nearest-centroid-first,
            // and a centroid being closer does not mean every row in its
            // partition is closer than every row of a farther partition's.
            // True cross-wave best-first order needs per-partition
            // distance bounds, which this index does not have.
            let mut last_distance = f32::NEG_INFINITY;
            for c in &batch.candidates {
                assert!(seen.insert(c.row), "row {} returned twice", c.row);
                assert!(
                    c.distance >= last_distance - 1e-6,
                    "not closest-first within the batch"
                );
                last_distance = c.distance;
            }
            if batch.exhausted {
                break;
            }
        }
        assert_eq!(seen.len(), 500, "every row returned exactly once");
    }

    #[tokio::test]
    async fn the_index_head_is_fetched_once_however_many_next_calls() {
        use object_store::ObjectStoreExt as _;
        use object_store::path::Path;

        let (_matrix, bytes) = clustered_object(2_000, 16, 8, Metric::L2); // several waves' worth
        let store = CountingStore::default();
        let path = Path::from("idx");
        store.inner.put(&path, bytes.into()).await.expect("put");
        let budget = crate::budget::ReadBudget::unlimited();
        let policy = crate::fetch::FetchPolicy::local();
        let query = Lcg(7).matrix(1, 16);
        let probe = Probe {
            store: &store,
            budget: &budget,
            policy: &policy,
            path: &path,
            dim: 16,
            rows: 2_000,
            data_size: 999,
            metric: Metric::L2,
            query: &query,
            want: 10,
            nprobe: Some(1),
            decoded: None,
        };
        let mut cursor = Sq8.cursor(probe).await.expect("cursor");
        let _ = cursor.next(10).await.expect("next");
        let after_first = store.requests();
        let _ = cursor.next(2_000).await.expect("next"); // forces every remaining wave
        // One request served the head (`open`'s single GET up to `HEAD_GUESS`,
        // or two if the head overran the guess) plus one per wave's partition
        // fetch — the head itself must not be re-fetched by a later wave.
        assert!(
            store.requests() > after_first,
            "later waves must still fetch their own partitions"
        );
        // Rebuild a fresh cursor against the same store and drain it in many
        // small `next` calls instead of two big ones: the total request count
        // must be the same either way, proving the head was never re-read.
        store.reset();
        let mut cursor2 = Sq8.cursor(probe).await.expect("cursor");
        loop {
            let batch = cursor2.next(50).await.expect("next");
            if batch.exhausted {
                break;
            }
        }
        let many_small_calls = store.requests();
        store.reset();
        let mut cursor3 = Sq8.cursor(probe).await.expect("cursor");
        loop {
            let batch = cursor3.next(2_000).await.expect("next");
            if batch.exhausted {
                break;
            }
        }
        assert_eq!(
            many_small_calls,
            store.requests(),
            "the head and each partition are each fetched once, regardless of how next() is called"
        );
    }

    /// Regression: `nprobe >= nlist` makes `Head::nearest` take its "give
    /// me all of them" shortcut, returning partitions in plain index order
    /// — the cursor's own ranking of *every* partition must not reuse that
    /// call, or waves probe arbitrary partitions instead of nearest-first
    /// (this crashed `tests/query.rs`'s
    /// `the_default_nprobe_on_a_clustered_index_keeps_recall` to ~22%
    /// recall, since its default `nprobe` of 8 over `nlist(16)` partitions
    /// never actually hits this exact `>=` case — the real culprit was
    /// calling `head.nearest(.., head.nlist())` to rank *all* partitions
    /// for the cursor's own book-keeping, which always trips the shortcut
    /// by construction).
    #[tokio::test]
    async fn every_partition_is_ranked_by_centroid_distance_even_when_nprobe_covers_them_all() {
        let (_, bytes) = clustered_object(400, 8, 5, Metric::L2);
        let decoded_index = decoded(&bytes, 8, 400).expect("decode");
        let store = object_store::memory::InMemory::new();
        let path = object_store::path::Path::from("idx");
        use object_store::ObjectStoreExt as _;
        store.put(&path, bytes.into()).await.expect("put");
        let budget = crate::budget::ReadBudget::unlimited();
        let policy = crate::fetch::FetchPolicy::local();
        let query = Lcg(55).matrix(1, 8);

        let mut want: Vec<(f32, usize)> = (0..decoded_index.head.nlist())
            .filter_map(|i| {
                Some((
                    distance(Metric::L2, &query, decoded_index.head.centroid(i)?),
                    i,
                ))
            })
            .collect();
        want.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let want_order: Vec<usize> = want.into_iter().map(|(_, i)| i).collect();

        // nprobe (10) >= nlist (5): exactly the shortcut-triggering case.
        let probe = Probe {
            store: &store,
            budget: &budget,
            policy: &policy,
            path: &path,
            dim: 8,
            rows: 400,
            data_size: 999,
            metric: Metric::L2,
            query: &query,
            want: 1,
            nprobe: Some(10),
            decoded: None,
        };
        let cursor = Sq8Cursor::open(probe).await.expect("open");
        let got_order: Vec<usize> = cursor.remaining_partitions.iter().copied().collect();
        assert_eq!(
            got_order, want_order,
            "partitions must be ranked by centroid distance, not left in index order"
        );
    }

    /// Regression: `rest_lower_bound` must be `None` while any partition is
    /// still unscanned, never a bound derived only from `buffered`. A plain
    /// `sq8` index has no per-partition distance bound, so an unscanned
    /// partition could hold anything, including a row closer than every
    /// row already buffered. Reporting `buffered.first()`'s bound as "the
    /// rest" lets the engine stop a source early and drop a real match
    /// hiding in a partition it never got to.
    #[tokio::test]
    async fn rest_lower_bound_is_none_while_any_partition_is_still_unscanned() {
        use object_store::ObjectStoreExt as _;

        // nprobe(1) over several partitions: the first wave alone holds far
        // more than `nprobe`'s rows will be asked for in one `next` call,
        // so `buffered` stays non-empty *and* `remaining_partitions` stays
        // non-empty after the first `next`, the exact condition this bug
        // needs to show up in.
        let (_, bytes) = clustered_object(2_000, 8, 8, Metric::L2);
        let store = object_store::memory::InMemory::new();
        let path = object_store::path::Path::from("idx");
        store.put(&path, bytes.into()).await.expect("put");
        let budget = crate::budget::ReadBudget::unlimited();
        let policy = crate::fetch::FetchPolicy::local();
        let query = Lcg(17).matrix(1, 8);
        let probe = Probe {
            store: &store,
            budget: &budget,
            policy: &policy,
            path: &path,
            dim: 8,
            rows: 2_000,
            data_size: 999,
            metric: Metric::L2,
            query: &query,
            want: 1,
            nprobe: Some(1),
            decoded: None,
        };
        let mut cursor = Sq8Cursor::open(probe).await.expect("open");
        let batch = cursor.next(10).await.expect("next");
        assert!(!batch.exhausted, "7 of 8 partitions are still unprobed");
        assert_eq!(
            batch.rest_lower_bound, None,
            "a plain sq8 index has no bound on what an unscanned partition might hold"
        );
    }

    #[test]
    fn rest_lower_bound_is_never_above_the_exact_distance_of_any_unreturned_row() {
        let (matrix, bytes) = built(600, 12, 11);
        let decoded = decoded(&bytes, 12, 600).expect("decode");
        for metric in [Metric::L2, Metric::Dot] {
            let query = Lcg(321).matrix(1, 12);
            let prepared = decoded.head.prepare(metric, &query).expect("prepare");
            let l2_error = decoded.head.codec_l2_error_bound();
            let dot_error = prepared.dot_error_bound();
            // Scan every partition in full (nprobe = nlist) to get every row's
            // approximate distance, then split at an arbitrary point: the rows
            // "not yet returned" are everything past it.
            let mut all: Vec<Candidate> = Vec::new();
            for i in 0..decoded.head.nlist() {
                all.extend(decoded.partitions[i].scan(&prepared, decoded.partitions[i].len()));
            }
            all.sort_by(|a, b| a.distance.total_cmp(&b.distance));
            let split = all.len() / 2;
            for candidate in &all[split..] {
                let lower =
                    sq8_lower_bound(metric, l2_error, dot_error, candidate.distance).unwrap();
                let exact_vector =
                    &matrix[candidate.row as usize * 12..candidate.row as usize * 12 + 12];
                let exact = distance(metric, &query, exact_vector);
                assert!(
                    lower <= exact + 1e-2,
                    "{metric:?} row {}: lower {lower} above exact {exact}",
                    candidate.row
                );
            }
        }
    }

    #[test]
    fn the_bytes_are_fixed_for_a_tiny_input() {
        let matrix = [0.0f32, 10.0, 255.0, 20.0];
        let bytes = build(&matrix, 2, 2, 999, 1, Metric::L2).expect("build");

        let head_len = 36u64 + 4 + 16 + 8 + 12;
        let mut want = Vec::new();
        want.extend_from_slice(b"ITZI");
        for n in [1u32, 1, 2, 2] {
            want.extend_from_slice(&n.to_le_bytes());
        }
        want.extend_from_slice(&999u64.to_le_bytes());
        want.extend_from_slice(&head_len.to_le_bytes());
        want.extend_from_slice(&1u32.to_le_bytes());
        for x in [0.0f32, 1.0, 10.0, 10.0 / 255.0, 127.5, 15.0] {
            want.extend_from_slice(&x.to_le_bytes());
        }
        want.extend_from_slice(&2u32.to_le_bytes());
        want.extend_from_slice(&head_len.to_le_bytes());
        for row in [0u32, 1] {
            want.extend_from_slice(&row.to_le_bytes());
        }
        want.extend_from_slice(&[0, 0, 255, 255]);
        assert_eq!(bytes.to_vec(), want);
    }

    #[test]
    fn the_same_input_gives_the_same_bytes() {
        assert_eq!(built(200, 8, 3).1, built(200, 8, 3).1);
        assert_ne!(built(200, 8, 3).1, built(200, 8, 4).1);
    }

    #[test]
    fn a_round_trip_is_within_half_a_step_and_uses_the_full_range() {
        let (rows, dim) = (500, 8);
        let (matrix, bytes) = built(rows, dim, 1);
        let index = decoded(&bytes, dim, rows as u64).expect("decode");
        assert_eq!(index.head.nlist(), 1);
        assert_eq!(index.partitions[0].len(), rows);
        for (r, code) in index.partitions[0].codes.chunks_exact(dim).enumerate() {
            let back = index.head.dequantize(code);
            for d in 0..dim {
                let step = index.head.codec.scale()[d];
                let err = (back[d] - matrix[r * dim + d]).abs();
                assert!(
                    err <= step / 2.0 + 1e-6,
                    "row {r} dim {d}: {err} > {step}/2"
                );
            }
        }
        for d in 0..dim {
            let column: Vec<u8> = index.partitions[0]
                .codes
                .chunks_exact(dim)
                .map(|c| c[d])
                .collect();
            assert!(column.contains(&0) && column.contains(&255), "dim {d}");
        }
        assert_eq!(
            index.partitions[0].row_numbers(),
            (0..rows as u32).collect::<Vec<_>>()
        );
    }

    #[test]
    fn constant_dimensions_one_row_and_no_rows_build() {
        // A constant column has scale 0 and reconstructs exactly.
        let matrix = [5.0f32, 1.0, 5.0, 2.0, 5.0, 3.0];
        let bytes = build(&matrix, 2, 3, 999, 1, Metric::L2).expect("build");
        let index = decoded(&bytes, 2, 3).expect("decode");
        assert_eq!(index.head.codec.scale()[0], 0.0);
        assert_eq!(index.head.dequantize(&[0, 0])[0], 5.0);

        let one = build(&[1.0, 2.0, 3.0], 3, 1, 999, 1, Metric::L2).expect("build");
        let index = decoded(&one, 3, 1).expect("decode");
        assert_eq!(index.head.codec.scale(), [0.0, 0.0, 0.0]);
        assert_eq!(
            index.head.dequantize(&index.partitions[0].codes),
            [1.0, 2.0, 3.0]
        );

        let none = build(&[], 3, 0, 999, 1, Metric::L2).expect("build");
        let index = decoded(&none, 3, 0).expect("decode");
        assert_eq!(index.partitions[0].len(), 0);
        let head = &index.head;
        let prepared = head.prepare(Metric::L2, &[0.0, 0.0, 0.0]).expect("prepare");
        assert!(index.partitions[0].scan(&prepared, 5).is_empty());
    }

    #[test]
    fn the_builder_refuses_bad_input() {
        for (matrix, dim, rows, want) in [
            (vec![1.0f32, 2.0, 3.0], 2usize, 2usize, "rows * dim"),
            (vec![1.0, f32::NAN], 2, 1, "not finite"),
            (vec![f32::INFINITY, 1.0], 2, 1, "not finite"),
            (vec![], 0, 0, "dimension"),
        ] {
            let err = build(&matrix, dim, rows, 1, 1, Metric::L2).expect_err(want);
            assert!(reason(err).contains(want), "{want}");
        }
    }

    fn metrics() -> [Metric; 3] {
        [Metric::L2, Metric::Dot, Metric::Cosine]
    }

    /// The scan distance is the reference distance to the reconstructed vector.
    #[test]
    fn the_kernel_matches_the_reference_on_the_reconstruction() {
        let (rows, dim) = (300, 16);
        let (_, bytes) = built(rows, dim, 5);
        let index = decoded(&bytes, dim, rows as u64).expect("decode");
        let query = Lcg(77).matrix(1, dim);
        for metric in metrics() {
            let prepared = index.head.prepare(metric, &query).expect("prepare");
            for code in index.partitions[0].codes.chunks_exact(dim) {
                let want = distance(metric, &query, &index.head.dequantize(code));
                let got = prepared.distance(code);
                assert!(
                    (got - want).abs() <= 1e-4 * (1.0 + want.abs()),
                    "{metric:?}: {got} vs {want}"
                );
            }
        }
    }

    /// And it is close to the distance to the original, within what half a step
    /// per dimension can move it.
    #[test]
    fn the_kernel_is_close_to_the_exact_distance() {
        let (rows, dim) = (300, 16);
        let (matrix, bytes) = built(rows, dim, 6);
        let index = decoded(&bytes, dim, rows as u64).expect("decode");
        let query = Lcg(78).matrix(1, dim);
        for metric in metrics() {
            let prepared = index.head.prepare(metric, &query).expect("prepare");
            for (r, code) in index.partitions[0].codes.chunks_exact(dim).enumerate() {
                let exact = distance(metric, &query, &matrix[r * dim..(r + 1) * dim]);
                let got = prepared.distance(code);
                let bound: f32 = (0..dim)
                    .map(|d| {
                        let e = index.head.codec.scale()[d] / 2.0 + 1e-6;
                        let x = matrix[r * dim + d];
                        match metric {
                            Metric::L2 => 2.0 * (query[d] - x).abs() * e + e * e + 2.0 * e * e,
                            Metric::Dot => query[d].abs() * e,
                            Metric::Cosine => 0.02 / dim as f32,
                        }
                    })
                    .sum();
                assert!(
                    (got - exact).abs() <= bound + 1e-4,
                    "{metric:?} row {r}: {got} vs {exact}, bound {bound}"
                );
            }
        }
    }

    #[test]
    fn a_zero_vector_is_at_cosine_distance_one() {
        // The second column is all zero, so its rows reconstruct to zero.
        let matrix = [0.0f32, 0.0, 0.0, 0.0];
        let bytes = build(&matrix, 2, 2, 999, 1, Metric::L2).expect("build");
        let index = decoded(&bytes, 2, 2).expect("decode");
        let q = index
            .head
            .prepare(Metric::Cosine, &[1.0, 2.0])
            .expect("prepare");
        assert_eq!(q.distance(&[0, 0]), 1.0);
        let zero = index
            .head
            .prepare(Metric::Cosine, &[0.0, 0.0])
            .expect("prepare");
        assert_eq!(zero.distance(&[0, 0]), 1.0);
    }

    #[test]
    fn a_query_of_the_wrong_length_is_refused() {
        let (_, bytes) = built(10, 4, 1);
        let index = decoded(&bytes, 4, 10).expect("decode");
        let err = index.head.prepare(Metric::L2, &[1.0]).expect_err("refused");
        assert!(matches!(
            err,
            ItzmaError::DimensionMismatch {
                expected: 4,
                got: 1
            }
        ));
    }

    /// The scan works 256 rows at a time; identical rows on both sides of that
    /// edge still come back lowest row first.
    #[test]
    fn ties_across_the_chunk_edge_break_by_row() {
        let (rows, dim) = (600, 4);
        let matrix = vec![0.5f32; rows * dim];
        let bytes = build(&matrix, dim, rows, 999, 1, Metric::L2).expect("build");
        let index = decoded(&bytes, dim, rows as u64).expect("decode");
        let prepared = index.head.prepare(Metric::L2, &[0.0; 4]).expect("prepare");
        for want in [1, 3, 255, 256, 257, 300, 599, 600, 1000] {
            let got: Vec<u32> = index.partitions[0]
                .scan(&prepared, want)
                .iter()
                .map(|c| c.row)
                .collect();
            let expected: Vec<u32> = (0..want.min(rows) as u32).collect();
            assert_eq!(got, expected, "want {want}");
        }
    }

    #[test]
    fn a_scan_keeps_the_closest_rows_and_breaks_ties_by_row() {
        let (rows, dim) = (400, 6);
        let (_, bytes) = built(rows, dim, 9);
        let index = decoded(&bytes, dim, rows as u64).expect("decode");
        let query = Lcg(5).matrix(1, dim);
        for metric in metrics() {
            let prepared = index.head.prepare(metric, &query).expect("prepare");
            let mut all: Vec<Candidate> = index.partitions[0]
                .codes
                .chunks_exact(dim)
                .enumerate()
                .map(|(row, code)| Candidate {
                    row: row as u32,
                    distance: prepared.distance(code),
                })
                .collect();
            all.sort_by(|a, b| a.distance.total_cmp(&b.distance).then(a.row.cmp(&b.row)));
            for want in [0, 1, 10, 399, 400, 1000] {
                let got = index.partitions[0].scan(&prepared, want);
                assert_eq!(got, all[..want.min(rows)], "{metric:?} want {want}");
            }
        }

        // Identical rows tie, and the lower row number wins.
        let same = [1.0f32, 1.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0];
        let bytes = build(&same, 2, 4, 999, 1, Metric::L2).expect("build");
        let index = decoded(&bytes, 2, 4).expect("decode");
        let prepared = index
            .head
            .prepare(Metric::L2, &[1.0, 1.0])
            .expect("prepare");
        let got = index.partitions[0].scan(&prepared, 2);
        assert_eq!(got.iter().map(|c| c.row).collect::<Vec<_>>(), [0, 2]);
    }

    /// Patch `bytes` at `at`.
    fn patched(bytes: &Bytes, at: usize, with: &[u8]) -> Vec<u8> {
        let mut b = bytes.to_vec();
        b[at..at + with.len()].copy_from_slice(with);
        b
    }

    /// Offsets in a two-row, two-dimension object.
    const NLIST_AT: usize = 36;
    const PARAMS_AT: usize = 40;
    const CENTROID_AT: usize = PARAMS_AT + 16;
    const TABLE_AT: usize = CENTROID_AT + 8;
    const ROWS_AT: usize = TABLE_AT + 12;

    fn tiny() -> Bytes {
        build(&[0.0, 10.0, 255.0, 20.0], 2, 2, 999, 1, Metric::L2).expect("build")
    }

    #[test]
    fn a_segment_mismatch_is_refused() {
        let bytes = tiny();
        decoded(&bytes, 2, 2).expect("matches");
        for (dim, rows, size, want) in [
            (3, 2, 999, "dimension"),
            (2, 3, 999, "rows, the segment has"),
            (2, 2, 1000, "data object"),
        ] {
            let err = Index::decode(PATH, &bytes, dim, rows, size).expect_err(want);
            assert!(reason(err).contains(want), "{want}");
        }
    }

    #[test]
    fn each_forged_object_is_refused_for_its_reason() {
        let bytes = tiny();
        let head_len = TABLE_AT as u64 + 12;
        let nan = f32::NAN.to_le_bytes();
        let cases: Vec<(Vec<u8>, &str)> = vec![
            (
                patched(&bytes, 28, &(head_len + 1000).to_le_bytes()),
                "runs past the end",
            ),
            (
                patched(&bytes, 28, &(head_len - 1).to_le_bytes()),
                "disagrees",
            ),
            (patched(&bytes, NLIST_AT, &0u32.to_le_bytes()), "nlist is 0"),
            (
                patched(&bytes, NLIST_AT, &u32::MAX.to_le_bytes()),
                "disagrees",
            ),
            (patched(&bytes, NLIST_AT, &2u32.to_le_bytes()), "disagrees"),
            (patched(&bytes, PARAMS_AT, &nan), "not finite"),
            (
                patched(&bytes, PARAMS_AT + 4, &(-1.0f32).to_le_bytes()),
                "negative",
            ),
            (
                patched(&bytes, PARAMS_AT + 8, &f32::INFINITY.to_le_bytes()),
                "not finite",
            ),
            (patched(&bytes, CENTROID_AT, &nan), "centroid"),
            (
                patched(&bytes, TABLE_AT + 4, &(head_len + 4).to_le_bytes()),
                "do not follow",
            ),
            (patched(&bytes, TABLE_AT, &1u32.to_le_bytes()), "add up"),
            (
                patched(&bytes, ROWS_AT, &1u32.to_le_bytes()),
                "out of order",
            ),
            (
                patched(&bytes, ROWS_AT + 4, &0u32.to_le_bytes()),
                "out of order",
            ),
            (
                patched(&bytes, ROWS_AT + 4, &9u32.to_le_bytes()),
                "past the segment's rows",
            ),
            (
                patched(&bytes, ROWS_AT + 4, &2u32.to_le_bytes()),
                "past the segment's rows",
            ),
            (
                {
                    let mut b = bytes.to_vec();
                    b.push(0);
                    b
                },
                "does not end",
            ),
            (bytes[..bytes.len() - 1].to_vec(), "does not end"),
        ];
        for (forged, want) in cases {
            let err = decoded(&forged, 2, 2).expect_err(want);
            assert!(reason(err).contains(want), "expected {want:?}");
        }
    }

    #[test]
    fn a_partition_of_the_wrong_length_is_refused() {
        let bytes = tiny();
        let index = decoded(&bytes, 2, 2).expect("decode");
        let slice = bytes.slice(ROWS_AT..);
        Partition::decode(PATH, &index.head, 0, slice.clone()).expect("right length");
        for bad in [slice.slice(..slice.len() - 1), Bytes::new()] {
            let err = Partition::decode(PATH, &index.head, 0, bad).expect_err("short");
            assert!(reason(err).contains("length"));
        }
        let err = Partition::decode(PATH, &index.head, 1, slice).expect_err("no such");
        assert!(reason(err).contains("no partition 1"));
    }

    /// a partition is read in place. Its codes and row numbers are
    /// views of the bytes it was decoded from, not copies, so a cached
    /// partition costs its bytes once.
    #[test]
    fn a_partition_reads_its_codes_in_place() {
        let (_, bytes) = built(300, 8, 3);
        let index = decoded(&bytes, 8, 300).expect("decode");
        let range = index.head.partition_range(0).expect("one partition");
        let slice = bytes.slice(range.start as usize..range.end as usize);
        let partition = Partition::decode(PATH, &index.head, 0, slice.clone()).expect("decode");
        let within = slice.as_ptr_range();
        assert!(
            within.contains(&partition.codes.as_ptr()),
            "codes were copied"
        );
        assert!(
            within.contains(&partition.rows.as_ptr()),
            "rows were copied"
        );
        assert_eq!(partition.row_numbers(), (0..300).collect::<Vec<u32>>());
    }

    #[test]
    fn every_truncation_is_refused() {
        let bytes = built(30, 3, 2).1;
        for len in 0..bytes.len() {
            assert!(decoded(&bytes[..len], 3, 30).is_err(), "{len} bytes");
        }
    }

    /// A flipped bit never panics. One in the prefix, `nlist` or the table
    /// always shows: a flip in a float or a code is a different valid index.
    #[test]
    fn a_flipped_bit_never_panics_and_structure_flips_show() {
        let bytes = tiny();
        let structural = (0..PARAMS_AT).chain(TABLE_AT..ROWS_AT);
        for i in structural {
            for bit in 0..8 {
                let mut b = bytes.to_vec();
                b[i] ^= 1 << bit;
                assert!(decoded(&b, 2, 2).is_err(), "byte {i} bit {bit}");
            }
        }
        for i in 0..bytes.len() {
            for bit in 0..8 {
                let mut b = bytes.to_vec();
                b[i] ^= 1 << bit;
                let _ = decoded(&b, 2, 2);
            }
        }
    }

    /// The format already carries several partitions, which clustering will
    /// use: two partitions built by hand parse, and a table that does not
    /// chain is refused.
    #[test]
    fn a_two_partition_object_parses() {
        let dim = 1u32;
        let head_len = 36u64 + 4 + 8 + 8 + 24;
        let prefix = Prefix {
            kind: Kind::Sq8,
            dim,
            rows: 4,
            data_size: 999,
            head_len,
        };
        let mut b = prefix.encode().to_vec();
        b.extend_from_slice(&2u32.to_le_bytes());
        for x in [0.0f32, 1.0, 0.5, 2.5] {
            b.extend_from_slice(&x.to_le_bytes());
        }
        let second = head_len + 2 * 5;
        for (rows, offset) in [(2u32, head_len), (2, second)] {
            b.extend_from_slice(&rows.to_le_bytes());
            b.extend_from_slice(&offset.to_le_bytes());
        }
        for row in [0u32, 2, 1, 3] {
            // Row numbers, then the codes, for each partition in turn.
            b.extend_from_slice(&row.to_le_bytes());
            if row == 2 || row == 3 {
                b.extend_from_slice(&[10, 20]);
            }
        }
        let index = Index::decode(PATH, &b, 1, 4, 999).expect("two partitions");
        assert_eq!(index.head.nlist(), 2);
        assert_eq!(index.head.centroid(1), Some(&[2.5f32][..]));
        assert_eq!(index.head.partition_range(1), Some(second..second + 10));
        assert_eq!(index.partitions[1].row_numbers(), [1, 3]);

        let mut gap = b.clone();
        let table = (head_len - 24) as usize;
        gap[table + 12 + 4..table + 12 + 12].copy_from_slice(&(second + 1).to_le_bytes());
        let err = Index::decode(PATH, &gap, 1, 4, 999).expect_err("gap");
        assert!(reason(err).contains("do not follow"));
    }

    /// A clustered object for `metric`, and its matrix.
    fn clustered_object(
        rows: usize,
        dim: usize,
        nlist: usize,
        metric: Metric,
    ) -> (Vec<f32>, Bytes) {
        let matrix = Lcg(3).matrix(rows, dim);
        let bytes = build(&matrix, dim, rows, 999, nlist, metric).expect("build");
        (matrix, bytes)
    }

    #[test]
    fn one_big_partition_split_across_threads_equals_the_sequential_scan() {
        // 40,000 identical rows: every distance ties, across every split edge.
        let (dim, rows) = (8usize, 40_000usize);
        let matrix = vec![0.25f32; rows * dim];
        let bytes = build(&matrix, dim, rows, 999, 1, Metric::L2).expect("build");
        let index = decoded(&bytes, dim, rows as u64).expect("decode");
        let prepared = index.head.prepare(Metric::L2, &[0.0; 8]).expect("prepare");
        let partition = &index.partitions[0];
        let outside = (partition.scan(&prepared, 10), partition.scan_all(&prepared));
        assert_eq!(
            outside.0.iter().map(|c| c.row).collect::<Vec<_>>(),
            (0..10).collect::<Vec<u32>>()
        );
        for threads in [2, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let inside =
                pool.install(|| (partition.scan(&prepared, 10), partition.scan_all(&prepared)));
            assert_eq!(inside, outside, "{threads} threads");
        }
    }

    #[tokio::test]
    async fn a_wave_scanned_on_the_pool_equals_the_inline_scan() {
        // 25,000 rows at dimension 64 is past the inline threshold.
        let (dim, rows) = (64usize, 25_000usize);
        let (matrix, bytes) = clustered_object(rows, dim, 4, Metric::L2);
        let index = decoded(&bytes, dim, rows as u64).expect("decode");
        let prepared = index
            .head
            .prepare(Metric::L2, &matrix[..dim])
            .expect("prepare");
        let partitions: Vec<Arc<Partition>> =
            index.partitions.iter().cloned().map(Arc::new).collect();
        let every: Vec<Candidate> = partitions
            .iter()
            .flat_map(|p| p.scan_all(&prepared))
            .collect();
        let closest: Vec<Candidate> = partitions
            .iter()
            .flat_map(|p| p.scan(&prepared, 10))
            .collect();
        let prepared = Arc::new(prepared);
        assert_eq!(
            scan_wave_on_pool(partitions.clone(), prepared.clone(), None).await,
            every
        );
        assert_eq!(
            scan_wave_on_pool(partitions, prepared, Some(10)).await,
            closest
        );
    }

    #[test]
    fn a_clustered_object_decodes_and_every_row_is_in_exactly_one_partition() {
        let (dim, rows, nlist) = (4usize, 200usize, 8usize);
        for metric in [Metric::L2, Metric::Cosine, Metric::Dot] {
            let (_, bytes) = clustered_object(rows, dim, nlist, metric);
            let index = decoded(&bytes, dim, rows as u64).expect("decode");
            assert_eq!(index.head.nlist(), nlist, "{metric:?}");
            let mut seen = vec![false; rows];
            for p in &index.partitions {
                assert!(
                    p.row_numbers().windows(2).all(|w| w[0] < w[1]),
                    "ascending, {metric:?}"
                );
                for r in &p.row_numbers() {
                    assert!(
                        !std::mem::replace(&mut seen[*r as usize], true),
                        "row {r} twice"
                    );
                }
            }
            assert!(
                seen.iter().all(|s| *s),
                "every row is in a partition, {metric:?}"
            );
        }
    }

    #[test]
    fn a_query_probes_the_partition_that_holds_a_stored_vector() {
        let (dim, rows, nlist) = (4usize, 200usize, 6usize);
        // Not `Dot`: it clusters under L2 and probes by the dot product, so a stored
        // vector is not always nearest its own centroid.
        for metric in [Metric::L2, Metric::Cosine] {
            let (matrix, bytes) = clustered_object(rows, dim, nlist, metric);
            let index = decoded(&bytes, dim, rows as u64).expect("decode");
            let mut agree = 0;
            for row in 0..rows {
                let query = &matrix[row * dim..(row + 1) * dim];
                let probed = index.head.nearest(metric, query, 1)[0];
                if index.partitions[probed]
                    .row_numbers()
                    .contains(&(row as u32))
                {
                    agree += 1;
                }
            }
            // A vector is nearest its own centroid almost always: assignment used
            // L2 (or direction), probing uses the metric, and the two agree here.
            assert!(
                agree * 10 >= rows * 9,
                "{agree}/{rows} probed their own partition, {metric:?}"
            );
        }
    }

    #[test]
    fn every_truncation_of_a_clustered_object_is_refused_and_no_flip_panics() {
        let (dim, rows) = (3usize, 40usize);
        let (_, bytes) = clustered_object(rows, dim, 4, Metric::L2);
        for len in 0..bytes.len() {
            assert!(
                decoded(&bytes[..len], dim, rows as u64).is_err(),
                "{len} bytes"
            );
        }
        for i in 0..bytes.len() {
            for bit in 0..8 {
                let mut b = bytes.to_vec();
                b[i] ^= 1 << bit;
                let _ = decoded(&b, dim, rows as u64);
            }
        }
    }

    #[test]
    fn more_partitions_than_rows_is_clamped_and_one_row_builds() {
        let (_, bytes) = clustered_object(5, 2, 50, Metric::L2);
        assert_eq!(decoded(&bytes, 2, 5).expect("decode").head.nlist(), 5);
        let one = build(&[1.0, 2.0], 2, 1, 999, 8, Metric::L2).expect("build");
        assert_eq!(decoded(&one, 2, 1).expect("decode").head.nlist(), 1);
    }

    /// A segment that existed when the index object was opened but is gone
    /// by the time `scan_object`'s own planner-backed partition fetch runs
    /// (a `gc` race) must still report `is_missing`, the same way every
    /// other reader in this crate does — that's what `Namespace`'s
    /// stale-read retry keys on.
    #[tokio::test]
    async fn a_segment_deleted_between_open_and_the_partition_fetch_is_still_reported_as_missing() {
        use object_store::ObjectStoreExt as _;
        use object_store::path::Path;

        let (dim, rows) = (4usize, 20usize);
        let (_matrix, bytes) = built(rows, dim, 1);
        let store = CountingStore::default();
        let path = Path::from("idx.bin");
        store.put(&path, bytes.clone().into()).await.expect("put");
        // Index objects this small open in one request; the next GET is
        // scan_object's own partition fetch.
        store.delete_on_nth_get(path.clone(), 2);

        let budget = crate::budget::ReadBudget::unlimited();
        let policy = crate::fetch::FetchPolicy::local();
        let query = vec![0.0f32; dim];
        let probe = Probe {
            store: &store,
            budget: &budget,
            policy: &policy,
            path: &path,
            dim,
            rows: rows as u64,
            data_size: 999,
            metric: Metric::L2,
            query: &query,
            want: 5,
            nprobe: None,
            decoded: None,
        };
        let err = Sq8.search(&probe).await.expect_err("missing");
        assert!(is_missing(&err), "{err:?}");
    }

    /// `scan_object` on a clustered object in a store: the answer is what a scan
    /// of exactly the partitions `Head::nearest` picks gives, run partition by
    /// partition, and a bigger `nprobe` reads more bytes. This is what breaks if
    /// the probe, the coalescing of adjacent partitions or the offsets inside a
    /// run go wrong. (A query through the engine hides such a bug: the widening
    /// loop's last resort asks for every row.)
    #[tokio::test]
    async fn a_cached_head_and_its_partitions_are_parsed_once_and_never_charged_again() {
        use object_store::ObjectStore;
        use object_store::ObjectStoreExt as _;
        use object_store::memory::InMemory;
        use object_store::path::Path;
        use std::sync::Arc;

        let (dim, rows, nlist) = (4usize, 400usize, 8usize);
        let (_, bytes) = clustered_object(rows, dim, nlist, Metric::L2);
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("idx.bin");
        store.put(&path, bytes.clone().into()).await.expect("put");
        let cache = crate::decoded::DecodedCache::new(Default::default());
        let decoded = cache.handle(&store);
        let query = Lcg(9).matrix(1, dim);
        let policy = crate::fetch::FetchPolicy::local();

        let mut answers = Vec::new();
        let mut charged = Vec::new();
        let parses_before = HEAD_PARSES.with(std::cell::Cell::get);
        for _ in 0..2 {
            let budget = crate::budget::ReadBudget::unlimited();
            let probe = Probe {
                store: &*store,
                budget: &budget,
                policy: &policy,
                path: &path,
                dim,
                rows: rows as u64,
                data_size: 999,
                metric: Metric::L2,
                query: &query,
                want: 25,
                nprobe: Some(3),
                decoded: Some(&decoded),
            };
            answers.push(Sq8.search(&probe).await.expect("search"));
            let mut cursor = Sq8.cursor(probe).await.expect("cursor");
            answers.push(cursor.next(25).await.expect("next").candidates);
            charged.push(budget.used());
        }
        assert_eq!(HEAD_PARSES.with(std::cell::Cell::get) - parses_before, 1);
        assert!(charged[0] > 0);
        assert_eq!(charged[1], 0, "a warm search reads and charges nothing");
        assert_eq!(answers[0], answers[2]);
        assert_eq!(answers[1], answers[3]);
    }

    #[tokio::test]
    async fn scan_object_reads_the_probed_partitions_and_only_those() {
        use object_store::ObjectStoreExt as _;
        use object_store::memory::InMemory;
        use object_store::path::Path;

        let (dim, rows, nlist) = (4usize, 400usize, 8usize);
        for metric in [Metric::L2, Metric::Cosine] {
            let (_, bytes) = clustered_object(rows, dim, nlist, metric);
            let store = InMemory::new();
            let path = Path::from("idx.bin");
            store.put(&path, bytes.clone().into()).await.expect("put");
            let index = decoded(&bytes, dim, rows as u64).expect("decode");
            let query = Lcg(9).matrix(1, dim);
            let prepared = index.head.prepare(metric, &query).expect("prepare");

            let mut charged = Vec::new();
            for nprobe in [1, 2, 3, 5, 8] {
                let budget = crate::budget::ReadBudget::unlimited();
                let policy = crate::fetch::FetchPolicy::local();
                let probe = Probe {
                    store: &store,
                    budget: &budget,
                    policy: &policy,
                    path: &path,
                    dim,
                    rows: rows as u64,
                    data_size: 999,
                    metric,
                    query: &query,
                    want: 25,
                    nprobe: Some(nprobe),
                    decoded: None,
                };
                let got = Sq8.search(&probe).await.expect("search");

                let mut want: Vec<Candidate> = index
                    .head
                    .nearest(metric, &query, nprobe)
                    .into_iter()
                    .flat_map(|i| index.partitions[i].scan(&prepared, 25))
                    .collect();
                want.sort_by(|a, b| a.distance.total_cmp(&b.distance).then(a.row.cmp(&b.row)));
                want.truncate(25);
                assert_eq!(got, want, "{metric:?}, nprobe {nprobe}");
                charged.push(budget.used());
            }
            assert!(
                charged.windows(2).all(|w| w[0] < w[1]),
                "more partitions must cost more bytes: {charged:?}"
            );
        }
    }

    /// Partitions that are near each other but do not exactly touch share a
    /// request under a merging policy, not just ones whose bytes are
    /// literally adjacent in the index object — the same planner
    /// `RangedSegment` uses now does this fetch too, instead of
    /// `scan_object`'s own touching-only merge rule.
    #[tokio::test]
    async fn nearby_but_not_touching_partitions_share_a_request_under_a_merging_policy() {
        use object_store::ObjectStoreExt as _;
        use object_store::path::Path;

        let (dim, rows, nlist) = (4usize, 400usize, 8usize);
        let (_, bytes) = clustered_object(rows, dim, nlist, Metric::L2);
        let store = CountingStore::default();
        let path = Path::from("idx.bin");
        store.put(&path, bytes.clone().into()).await.expect("put");
        let query = Lcg(2).matrix(1, dim);
        // Pinned, not just asserted by its effect: if clustering or seeding
        // ever changes what this query's 4 nearest partitions are, this
        // test must fail loudly here rather than silently stop exercising
        // the non-contiguous case it exists for.
        let index = decoded(&bytes, dim, rows as u64).expect("decode");
        let mut selected = index.head.nearest(Metric::L2, &query, 4);
        selected.sort_unstable();
        assert_eq!(
            selected,
            vec![0, 2, 4, 5],
            "this seed's 4 nearest partitions must stay non-contiguous \
             (gaps at 1 and 3) for this test to mean anything"
        );

        let budget = crate::budget::ReadBudget::unlimited();
        let wide = crate::fetch::FetchPolicy {
            min_gap: 1024 * 1024,
            max_gap: 1024 * 1024,
            ..crate::fetch::FetchPolicy::local()
        };
        let probe = Probe {
            store: &store,
            budget: &budget,
            policy: &wide,
            path: &path,
            dim,
            rows: rows as u64,
            data_size: 999,
            metric: Metric::L2,
            query: &query,
            want: 25,
            nprobe: Some(4), // a subset, so some probed partitions are not touching
            decoded: None,
        };
        store.reset();
        Sq8.search(&probe).await.expect("search");
        assert_eq!(
            store.requests(),
            2,
            "header/head guess plus one merged partitions request, even though \
             the probed partitions are not all touching"
        );
    }

    #[test]
    fn the_default_rules() {
        assert_eq!(default_nlist(0), 1);
        assert_eq!(
            default_nlist(99_999),
            1,
            "below the threshold, one partition"
        );
        assert_eq!(default_nlist(100_000), 316);
        assert_eq!(default_nlist(1_000_000), 1_000);
        assert_eq!(default_nprobe(1), 2);
        assert_eq!(default_nprobe(1_000), 64);
        assert_eq!(default_nprobe(4), 4);
    }
}
