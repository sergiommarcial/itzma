//! Segmented HNSW: a graph over SQ8-quantized codes, stored so a
//! query can traverse it directly out of already-fetched bytes — no decoded
//! `Graph` struct, matching what `CacheStore` actually caches (raw
//! bytes, not structures).
//!
//! Unlike SQ8, which fetches its head once and then a ranged GET per probed
//! partition, this format's `head_len` (the common prefix's field) covers the
//! *entire* object: there is nothing past it. [`super::open`]'s existing
//! "fetch up to `head_len`, extend past the guess if short" logic therefore
//! fetches the whole graph in one cold touch for free, with no HNSW-specific
//! fetch code at all.
//!
//! # Layout after the prefix
//!
//! All integers are little-endian. `head_len` in the prefix equals the size
//! of the whole object.
//!
//! | Section | Contents |
//! |---|---|
//! | `entry_point` | u32, a row number (unused, 0, when `rows` is 0) |
//! | `max_layer` | u32, the entry point's top layer (unused, 0, when `rows` is 0) |
//! | quantizer | per dimension, its minimum then its scale, both f32 (the same shape SQ8's head uses, [`crate::quant::Sq8Codec`]) |
//! | node directory | per row, its top layer (u8) and the byte offset of its node record, relative to the start of this body (the bytes right after the 36-byte common prefix) |
//! | node records | per row, in row order: its SQ8 code (`dim` bytes), then per layer `0..=top_layer`, a neighbor count (u16) and that many row numbers (u32), capped at `2 * M` for layer 0 and `M` above it |

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use bytes::Bytes;
use futures::future::BoxFuture;

use super::{Algorithm, Candidate, Kind, PREFIX_LEN, Prefix, Probe, Reader, corrupt, open};
use crate::cpu::{self, Cost, Lane};
use crate::document::Metric;
use crate::error::Result;
use crate::quant::{Sq8Codec, Sq8Query};

/// Max neighbors per node above layer 0. Layer 0 gets `2 * M`, the standard
/// HNSW asymmetry (denser at the base, where most of the graph's connectivity
/// has to live). Fixed for v1: no real per-namespace tuning need has shown up
/// yet, and exposing it is a small follow-up if one does.
const M: usize = 16;
const M0: usize = 2 * M;
/// Candidate list size while inserting. Larger builds a better graph, more
/// slowly. Fixed for v1, same reasoning as `M`.
pub(crate) const EF_CONSTRUCTION: usize = 200;

/// The bytes of the fixed part of the body (before the node records), for
/// `rows` rows of dimension `dim`: `entry_point` (4) + `max_layer` (4) +
/// quantizer (`dim * 8`) + directory (`rows * 9`).
fn directory_end(dim: u64, rows: u64) -> u64 {
    // Saturating: a forged `rows` or `dim` must not overflow before the
    // length is compared with the bytes that remain, matching sq8.rs's
    // `head_body_len`.
    4u64.saturating_add(4)
        .saturating_add(dim.saturating_mul(8))
        .saturating_add(rows.saturating_mul(9))
}

/// One row's byte offset (relative to the start of the body) and top layer,
/// before its node record is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DirEntry {
    top_layer: u8,
    offset: u64,
}

/// One row's neighbor lists, one per layer it appears in (`neighbors[0]` is
/// always present).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NodeBuild {
    pub(crate) top_layer: u8,
    pub(crate) neighbors: Vec<Vec<u32>>,
}

/// A graph ready to be written: the entry point, the quantizer, every row's
/// code, and every row's neighbor lists. `build` (added alongside
/// `IndexOptions::kind`) produces one of these by running the HNSW
/// construction algorithm; [`encode`] only knows how to turn it into bytes.
pub(crate) struct BuiltGraph {
    pub(crate) entry_point: u32,
    pub(crate) codec: Sq8Codec,
    pub(crate) codes: Vec<Vec<u8>>,
    pub(crate) nodes: Vec<NodeBuild>,
}

/// Write `built` as an index object for a segment of `rows` rows of `dim`
/// dimensions whose data object is `data_size` bytes. Node records are
/// written in row order, back to back with no gaps, so the directory's
/// offsets chain exactly like SQ8's partition table does.
///
/// # Errors
///
/// [`crate::error::ItzmaError::CorruptIndex`] (labelled `"(encoding)"`) if a
/// neighbor list at layer 0 exceeds `2 * M` entries, or above it exceeds `M`
/// — a build that produced more than its own cap is a bug worth catching here
/// rather than shipping an object no reader can safely trust its own cap
/// against.
pub(crate) fn encode(built: &BuiltGraph, dim: usize, rows: usize, data_size: u64) -> Result<Bytes> {
    let label = "(encoding)";
    let dim32 = u32::try_from(dim)
        .ok()
        .filter(|d| *d > 0)
        .ok_or_else(|| corrupt(label, "dimension must be between 1 and u32::MAX"))?;
    let rows32 = u32::try_from(rows)
        .map_err(|_| corrupt(label, "a segment of 2^32 rows or more cannot be indexed"))?;
    for node in &built.nodes {
        for (layer, neighbors) in node.neighbors.iter().enumerate() {
            let cap = if layer == 0 { M0 } else { M };
            if neighbors.len() > cap {
                return Err(corrupt(
                    label,
                    format!(
                        "layer {layer} has {} neighbors, more than the cap {cap}",
                        neighbors.len()
                    ),
                ));
            }
        }
    }

    let dir_end = directory_end(u64::from(dim32), u64::from(rows32));
    let mut node_bytes: Vec<Vec<u8>> = Vec::with_capacity(rows);
    for (row, node) in built.nodes.iter().enumerate() {
        let mut b = Vec::new();
        b.extend_from_slice(&built.codes[row]);
        for neighbors in &node.neighbors {
            b.extend_from_slice(&(neighbors.len() as u16).to_le_bytes());
            for n in neighbors {
                b.extend_from_slice(&n.to_le_bytes());
            }
        }
        node_bytes.push(b);
    }
    let head_len =
        PREFIX_LEN as u64 + dir_end + node_bytes.iter().map(|b| b.len() as u64).sum::<u64>();
    let prefix = Prefix {
        kind: Kind::Hnsw,
        dim: dim32,
        rows: rows32,
        data_size,
        head_len,
    };
    // The entry point is always the globally highest-layer node (that is how
    // HNSW picks one), so its own directory entry's top layer is the graph's
    // max_layer.
    let max_layer = built
        .nodes
        .get(built.entry_point as usize)
        .map_or(0u8, |n| n.top_layer);

    let mut out = Vec::with_capacity(head_len as usize);
    out.extend_from_slice(&prefix.encode());
    out.extend_from_slice(&built.entry_point.to_le_bytes());
    out.extend_from_slice(&u32::from(max_layer).to_le_bytes());
    for d in 0..dim {
        out.extend_from_slice(&built.codec.min()[d].to_le_bytes());
        out.extend_from_slice(&built.codec.scale()[d].to_le_bytes());
    }
    let mut offset = dir_end;
    for (node, bytes) in built.nodes.iter().zip(&node_bytes) {
        out.push(node.top_layer);
        out.extend_from_slice(&offset.to_le_bytes());
        offset += bytes.len() as u64;
    }
    for bytes in &node_bytes {
        out.extend_from_slice(bytes);
    }
    Ok(Bytes::from(out))
}

/// Deterministic LCG for level assignment, the same constants and shape as
/// `benches/core.rs`'s: no `rand` dependency.
struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64).max(f64::MIN_POSITIVE)
    }
}

/// In-progress node during construction: neighbor lists, one `Vec` per layer.
struct BuildNode {
    neighbors: Vec<Vec<u32>>,
}
impl BuildNode {
    fn top_layer(&self) -> usize {
        self.neighbors.len() - 1
    }
}

/// Scratch reused across `build_search_layer` calls within one `build` run,
/// so inserting `n` rows does not allocate a fresh `HashSet`/`BinaryHeap` per
/// row (profiling found this reuse the biggest win in construction).
#[derive(Default)]
struct BuildScratch {
    visited: HashSet<u32>,
    candidates: BinaryHeap<std::cmp::Reverse<Scored>>,
    results: BinaryHeap<Scored>,
}
impl BuildScratch {
    fn clear(&mut self) {
        self.visited.clear();
        self.candidates.clear();
        self.results.clear();
    }
}

/// In-memory layer search used only during construction (unlike the
/// query-time [`search_layer`], which reads nodes lazily out of bytes, this
/// one reads straight out of `nodes`/`codes`, since construction holds the
/// whole graph in memory already). Distance is computed through the codec
/// against the codes already assigned, so construction quality comes from the same
/// quantized comparisons a real query will use, not from a temporarily more
/// accurate distance.
#[allow(clippy::too_many_arguments)]
fn build_search_layer(
    codec: &Sq8Codec,
    metric: Metric,
    codes: &[Vec<u8>],
    nodes: &[BuildNode],
    scratch: &mut BuildScratch,
    query: &[f32],
    entry_points: &[u32],
    ef: usize,
    layer: usize,
) -> Vec<Scored> {
    scratch.clear();
    let BuildScratch {
        visited,
        candidates,
        results,
    } = scratch;
    let prepared = codec.prepare(metric, query);
    let dist = |row: u32| prepared.distance(&codes[row as usize]);

    visited.extend(entry_points.iter().copied());
    for &row in entry_points {
        let scored = Scored {
            distance: dist(row),
            row,
        };
        candidates.push(std::cmp::Reverse(scored));
        results.push(scored);
    }
    while let Some(std::cmp::Reverse(current)) = candidates.pop() {
        if results.len() >= ef {
            if let Some(farthest) = results.peek() {
                if current.distance > farthest.distance {
                    break;
                }
            }
        }
        let Some(node) = nodes.get(current.row as usize) else {
            continue;
        };
        let Some(neighbors) = node.neighbors.get(layer) else {
            continue;
        };
        for &n in neighbors {
            if !visited.insert(n) {
                continue;
            }
            let distance = dist(n);
            let better_than_worst =
                results.len() < ef || results.peek().is_some_and(|f| distance < f.distance);
            if better_than_worst {
                let scored = Scored { distance, row: n };
                candidates.push(std::cmp::Reverse(scored));
                results.push(scored);
                if results.len() > ef {
                    results.pop();
                }
            }
        }
    }
    let mut out: Vec<Scored> = results.iter().copied().collect();
    out.sort();
    out
}

/// Keep the `max` closest of `candidate_rows` to `target_row`, by distance to
/// each other (not to the original query being inserted) — the paper's
/// "simple" neighbor selection heuristic.
fn select_neighbors(
    codec: &Sq8Codec,
    metric: Metric,
    codes: &[Vec<u8>],
    target_row: u32,
    candidate_rows: &[u32],
    max: usize,
) -> Vec<u32> {
    let target = codec.prepare(metric, &codec.decode(&codes[target_row as usize]));
    let mut scored: Vec<(f32, u32)> = candidate_rows
        .iter()
        .map(|&row| (target.distance(&codes[row as usize]), row))
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    scored.into_iter().take(max).map(|(_, row)| row).collect()
}

/// Build the index object for a segment of `rows` vectors of length `dim`,
/// row-major in `matrix`, whose data object is `data_size` bytes, under
/// `metric` (construction quality is metric-dependent: neighbor selection
/// compares candidates by the segment's real distance function). Inserts
/// once per row in row order, with a fixed seed so the same input always builds the same bytes (matching
/// every other index builder in this crate — nothing here needs the
/// segment's content to influence the seed, since an index is derived data
/// compared for fitness, never byte for byte).
///
/// # Errors
///
/// [`crate::error::ItzmaError::CorruptIndex`] (labelled `"(encoding)"`), same
/// cases as [`super::sq8::build`]: a matrix that is not `rows * dim` values,
/// a non-finite value, or `rows` too big for a `u32`.
pub(crate) fn build(
    matrix: &[f32],
    dim: usize,
    rows: usize,
    data_size: u64,
    metric: Metric,
) -> Result<Bytes> {
    let label = "(encoding)";
    if dim == 0 || u32::try_from(dim).is_err() {
        return Err(corrupt(label, "dimension must be between 1 and u32::MAX"));
    }
    if rows.checked_mul(dim) != Some(matrix.len()) {
        return Err(corrupt(label, "the matrix is not rows * dim values"));
    }
    if matrix.iter().any(|x| !x.is_finite()) {
        return Err(corrupt(label, "a vector holds a value that is not finite"));
    }
    u32::try_from(rows)
        .map_err(|_| corrupt(label, "a segment of 2^32 rows or more cannot be indexed"))?;

    let codec = Sq8Codec::fit(matrix, dim)?;
    let codes: Vec<Vec<u8>> = matrix.chunks_exact(dim).map(|r| codec.encode(r)).collect();

    let mut nodes: Vec<BuildNode> = Vec::with_capacity(rows);
    let mut entry_point: Option<u32> = None;
    let mut rng = Lcg(0x5eed);
    let m_l = 1.0 / (M.max(2) as f64).ln();
    let mut scratch = BuildScratch::default();

    for row in 0..rows as u32 {
        let level = (-rng.next_f64().ln() * m_l).floor() as usize;
        let Some(mut ep) = entry_point else {
            nodes.push(BuildNode {
                neighbors: vec![Vec::new(); level + 1],
            });
            entry_point = Some(row);
            continue;
        };
        // The row's exact vector, reconstructed from its own code, is what
        // it is compared against while being connected — the graph only
        // ever compares quantized codes, at build time and at query time
        // alike (see the module doc and `build_search_layer`'s own doc).
        let query = codec.decode(&codes[row as usize]);
        let top_layer = nodes[ep as usize].top_layer();
        for layer in (level + 1..=top_layer).rev() {
            let found = build_search_layer(
                &codec,
                metric,
                &codes,
                &nodes,
                &mut scratch,
                &query,
                &[ep],
                1,
                layer,
            );
            if let Some(nearest) = found.first() {
                ep = nearest.row;
            }
        }
        nodes.push(BuildNode {
            neighbors: vec![Vec::new(); level + 1],
        });
        let mut entry_points = vec![ep];
        for layer in (0..=level.min(top_layer)).rev() {
            let found = build_search_layer(
                &codec,
                metric,
                &codes,
                &nodes,
                &mut scratch,
                &query,
                &entry_points,
                EF_CONSTRUCTION,
                layer,
            );
            let max = if layer == 0 { M0 } else { M };
            let candidate_rows: Vec<u32> = found.iter().map(|s| s.row).collect();
            let selected = select_neighbors(&codec, metric, &codes, row, &candidate_rows, max);

            for &neighbor in &selected {
                nodes[row as usize].neighbors[layer].push(neighbor);
                nodes[neighbor as usize].neighbors[layer].push(row);
                if nodes[neighbor as usize].neighbors[layer].len() > max {
                    let over_max = nodes[neighbor as usize].neighbors[layer].clone();
                    nodes[neighbor as usize].neighbors[layer] =
                        select_neighbors(&codec, metric, &codes, neighbor, &over_max, max);
                }
            }
            entry_points = selected;
        }
        if level > top_layer {
            entry_point = Some(row);
        }
    }

    let built = BuiltGraph {
        entry_point: entry_point.unwrap_or(0),
        codec,
        codes,
        nodes: nodes
            .into_iter()
            .map(|n| NodeBuild {
                top_layer: n.top_layer() as u8,
                neighbors: n.neighbors,
            })
            .collect(),
    };
    encode(&built, dim, rows, data_size)
}

/// The fixed header plus the directory, parsed on every query the same way
/// SQ8's `Head::parse` is. Cheap: O(rows), no I/O beyond what `open` already
/// fetched.
struct Header {
    entry_point: u32,
    max_layer: u32,
    codec: Sq8Codec,
    directory: Vec<DirEntry>,
}

impl Header {
    /// # Errors
    ///
    /// [`crate::error::ItzmaError::CorruptIndex`] if the fixed part of the
    /// body runs past the object, a quantizer parameter is not finite or has
    /// a negative scale, the entry point is past `rows`, a row's layer is
    /// above `max_layer`, or the entry point's own layer disagrees with
    /// `max_layer`.
    fn parse(path: &str, prefix: &Prefix, body: &[u8]) -> Result<Header> {
        let dim = prefix.dim as usize;
        let mut r = Reader::new(path, body);
        let entry_point = r.u32()?;
        let max_layer = r.u32()?;

        // Bounds-checked before anything sized by `dim` or `rows` is
        // allocated: a bit-flipped `dim`/`rows` must error here, not attempt
        // a multi-gigabyte `Vec::with_capacity`.
        let want = directory_end(u64::from(prefix.dim), u64::from(prefix.rows));
        if want > body.len() as u64 {
            return Err(corrupt(
                path,
                "the fixed header runs past the end of the object",
            ));
        }
        if prefix.rows > 0 && entry_point >= prefix.rows {
            return Err(corrupt(path, "entry point is past the segment's rows"));
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
        let mut directory = Vec::with_capacity(prefix.rows as usize);
        for _ in 0..prefix.rows {
            let top_layer = r.take(1)?[0];
            let offset = r.u64()?;
            if u32::from(top_layer) > max_layer {
                return Err(corrupt(
                    path,
                    "a row's layer is above the graph's max layer",
                ));
            }
            directory.push(DirEntry { top_layer, offset });
        }
        if prefix.rows > 0 {
            let entry_top_layer = u32::from(directory[entry_point as usize].top_layer);
            if entry_top_layer != max_layer {
                return Err(corrupt(
                    path,
                    "the entry point's own layer disagrees with max_layer",
                ));
            }
        }
        Ok(Header {
            entry_point,
            max_layer,
            codec: Sq8Codec::from_parts(min, scale),
            directory,
        })
    }
}

/// One row, decoded: its code and its neighbor lists.
#[derive(Debug, Clone, PartialEq)]
struct DecodedNode {
    code: Vec<u8>,
    neighbors: Vec<Vec<u32>>,
}

/// Read one row's node record starting at `entry.offset` into `body`. Bounds
/// are enforced by [`Reader::take`] (never allocates from a length field
/// before checking it against the bytes that remain). Returns the node and
/// how many bytes its record took, so a whole-object walk (`Index::decode`)
/// can check the directory chains without a separately stored length.
fn read_node(
    path: &str,
    body: &[u8],
    dim: usize,
    entry: DirEntry,
    rows: u32,
) -> Result<(DecodedNode, usize)> {
    let start = usize::try_from(entry.offset)
        .ok()
        .filter(|s| *s <= body.len())
        .ok_or_else(|| corrupt(path, "a node's offset runs past the object"))?;
    let mut r = Reader::new(path, &body[start..]);
    let code = r.take(dim)?.to_vec();
    let mut neighbors = Vec::with_capacity(entry.top_layer as usize + 1);
    for layer in 0..=entry.top_layer {
        let cap = if layer == 0 { M0 } else { M };
        let count = r.u16()? as usize;
        if count > cap {
            return Err(corrupt(
                path,
                format!("layer {layer} has {count} neighbors, more than the cap {cap}"),
            ));
        }
        let mut layer_neighbors = Vec::with_capacity(count);
        for _ in 0..count {
            let n = r.u32()?;
            if n >= rows {
                return Err(corrupt(path, "a neighbor row is past the segment's rows"));
            }
            layer_neighbors.push(n);
        }
        neighbors.push(layer_neighbors);
    }
    Ok((DecodedNode { code, neighbors }, r.pos()))
}

/// A whole index object, decoded, for tests and for the build-reuse check
/// (`indexer.rs`'s `put_or_reuse`). Walks the directory in row order and
/// checks the chain, the same discipline `sq8::Head::parse` applies to its
/// partition table.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Index {
    pub(crate) entry_point: u32,
    pub(crate) max_layer: u32,
    codec: Sq8Codec,
    #[allow(dead_code)] // kept for symmetry with Header; not read by any caller yet
    directory: Vec<DirEntry>,
    nodes: Vec<DecodedNode>,
}

impl Index {
    /// # Errors
    ///
    /// [`crate::error::ItzmaError::CorruptIndex`] for a segment mismatch, a
    /// bad quantizer parameter, a directory entry past the object, a layer's
    /// neighbor count over its cap, a neighbor row past `rows`, a directory
    /// that does not chain from the fixed header to the object's exact end,
    /// or an entry point/max layer that disagree with the directory.
    pub(crate) fn decode(
        path: &str,
        bytes: &[u8],
        dim: usize,
        rows: u64,
        data_size: u64,
    ) -> Result<Index> {
        let prefix = Prefix::decode(path, bytes)?;
        prefix.check_segment(path, dim, rows, data_size)?;
        if prefix.kind != Kind::Hnsw {
            return Err(corrupt(path, "not an HNSW index"));
        }
        let body_end = usize::try_from(prefix.head_len)
            .ok()
            .filter(|end| *end == bytes.len())
            .ok_or_else(|| corrupt(path, "head_len must equal the whole object's size"))?;
        let body = &bytes[PREFIX_LEN..body_end];
        let header = Header::parse(path, &prefix, body)?;

        let mut cursor = directory_end(u64::from(prefix.dim), u64::from(prefix.rows));
        let mut nodes = Vec::with_capacity(header.directory.len());
        for (row, entry) in header.directory.iter().enumerate() {
            if entry.offset != cursor {
                return Err(corrupt(
                    path,
                    format!("row {row}'s node does not follow the previous one"),
                ));
            }
            let (node, consumed) = read_node(path, body, dim, *entry, prefix.rows)?;
            cursor += consumed as u64;
            nodes.push(node);
        }
        if cursor != body.len() as u64 {
            return Err(corrupt(
                path,
                "the object does not end where its last node does",
            ));
        }
        Ok(Index {
            entry_point: header.entry_point,
            max_layer: header.max_layer,
            codec: header.codec,
            directory: header.directory,
            nodes,
        })
    }

    /// Reconstruct a code, for tests.
    #[cfg(test)]
    fn dequantize(&self, code: &[u8]) -> Vec<f32> {
        self.codec.decode(code)
    }
}

/// One candidate found while traversing a layer.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Scored {
    distance: f32,
    row: u32,
}
impl Eq for Scored {}
impl Ord for Scored {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then(self.row.cmp(&other.row))
    }
}
impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Fetch and cache row `row`'s node, if not already cached.
fn cache_node(
    cache: &mut HashMap<u32, DecodedNode>,
    path: &str,
    body: &[u8],
    dim: usize,
    header: &Header,
    row: u32,
) -> Result<()> {
    if cache.contains_key(&row) {
        return Ok(());
    }
    let entry = *header
        .directory
        .get(row as usize)
        .ok_or_else(|| corrupt(path, "a neighbor row is past the segment's rows"))?;
    let (node, _) = read_node(path, body, dim, entry, header.directory.len() as u32)?;
    cache.insert(row, node);
    Ok(())
}

/// Greedy best-first search of one layer from `entry_points`, keeping the
/// closest `ef` nodes found, reading nodes lazily out of `body` (via
/// [`read_node`], through `cache` so a hop already visited this query is
/// never re-parsed) instead of an in-memory `Vec<Node>`, and propagating
/// errors instead of assuming well-formed input.
#[allow(clippy::too_many_arguments)]
fn search_layer(
    path: &str,
    body: &[u8],
    header: &Header,
    dim: usize,
    query: &Sq8Query,
    entry_points: &[u32],
    ef: usize,
    layer: u8,
    cache: &mut HashMap<u32, DecodedNode>,
) -> Result<Vec<Scored>> {
    let mut visited: HashSet<u32> = entry_points.iter().copied().collect();
    let mut candidates: BinaryHeap<std::cmp::Reverse<Scored>> = BinaryHeap::new();
    let mut results: BinaryHeap<Scored> = BinaryHeap::new();

    for &row in entry_points {
        cache_node(cache, path, body, dim, header, row)?;
        let distance = query.distance(&cache[&row].code);
        let scored = Scored { distance, row };
        candidates.push(std::cmp::Reverse(scored));
        results.push(scored);
    }

    while let Some(std::cmp::Reverse(current)) = candidates.pop() {
        if results.len() >= ef {
            if let Some(farthest) = results.peek() {
                if current.distance > farthest.distance {
                    break;
                }
            }
        }
        cache_node(cache, path, body, dim, header, current.row)?;
        let neighbors: Vec<u32> = match cache[&current.row].neighbors.get(layer as usize) {
            Some(n) => n.clone(),
            None => continue,
        };
        for n in neighbors {
            if !visited.insert(n) {
                continue;
            }
            cache_node(cache, path, body, dim, header, n)?;
            let distance = query.distance(&cache[&n].code);
            let better_than_worst =
                results.len() < ef || results.peek().is_some_and(|f| distance < f.distance);
            if better_than_worst {
                let scored = Scored { distance, row: n };
                candidates.push(std::cmp::Reverse(scored));
                results.push(scored);
                if results.len() > ef {
                    results.pop();
                }
            }
        }
    }
    let mut out: Vec<Scored> = results.into_vec();
    out.sort();
    Ok(out)
}

/// The HNSW kind, as the query engine calls it.
pub(crate) struct Hnsw;

impl Algorithm for Hnsw {
    fn search<'a>(&'a self, probe: &'a Probe<'_>) -> BoxFuture<'a, Result<Vec<Candidate>>> {
        Box::pin(scan_object(probe))
    }
}

/// Fetch the whole object (one GET cold, cached bytes warm — see the module
/// doc), then traverse: greedy single-best descent from the top layer down
/// to layer 1, then a beam search of `ef` at layer 0. `nprobe` is
/// reinterpreted as an `ef_search` multiplier: `ef = want *
/// nprobe.max(1)`, so the default `nprobe: 1` searches just enough to fill
/// `want`, and a caller asking for a more thorough (slower) search raises it
/// — the same "bigger means more work" shape SQ8's partition probe has.
async fn scan_object(probe: &Probe<'_>) -> Result<Vec<Candidate>> {
    let name = probe.path.to_string();
    let opened = open(probe, Kind::Hnsw).await?;
    if opened.prefix.rows == 0 {
        return Ok(Vec::new());
    }
    let header = Header::parse(&name, &opened.prefix, &opened.body)?;
    let prepared = header.codec.prepare(probe.metric, probe.query);
    let (dim, want) = (probe.dim, probe.want);
    let ef = want.saturating_mul(probe.nprobe.unwrap_or(1).max(1)).max(1);
    let body = opened.body.clone();
    // About `ef` nodes visited at layer 0, each scoring up to `M0` neighbors:
    // an estimate of the work, not a bound.
    let cost = Cost::Distances {
        rows: ef.saturating_mul(M0),
        dim,
    };
    cpu::run(Lane::Query, cost, move || {
        traverse(&name, &body, &header, dim, &prepared, want, ef)
    })
    .await
}

/// Greedy single-best descent from the top layer down to layer 1, then a beam
/// search of `ef` at layer 0, keeping the closest `want`.
fn traverse(
    name: &str,
    body: &[u8],
    header: &Header,
    dim: usize,
    prepared: &Sq8Query,
    want: usize,
    ef: usize,
) -> Result<Vec<Candidate>> {
    let mut cache: HashMap<u32, DecodedNode> = HashMap::new();
    let mut ep = header.entry_point;

    // `max_layer` is guaranteed <= 255 by Header::parse's own consistency
    // check (it must equal the entry point's directory-recorded top layer,
    // a u8), so this cast never truncates a real value.
    for layer in (1..=header.max_layer).rev() {
        let found = search_layer(
            name,
            body,
            header,
            dim,
            prepared,
            &[ep],
            1,
            layer as u8,
            &mut cache,
        )?;
        if let Some(nearest) = found.first() {
            ep = nearest.row;
        }
    }
    let found = search_layer(name, body, header, dim, prepared, &[ep], ef, 0, &mut cache)?;
    Ok(found
        .into_iter()
        .take(want)
        .map(|s| Candidate {
            row: s.row,
            distance: s.distance,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ItzmaError;

    /// A hand-built 4-row, 2-dimension graph on a line: 0.0, 1.0, 2.0, 3.0.
    /// Row 1 is the entry point (top layer 1), fully connected at layer 0.
    fn tiny() -> BuiltGraph {
        let matrix = vec![0.0f32, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0];
        let codec = Sq8Codec::fit(&matrix, 2).expect("fit");
        let codes: Vec<Vec<u8>> = matrix.chunks_exact(2).map(|r| codec.encode(r)).collect();
        let nodes = vec![
            NodeBuild {
                top_layer: 0,
                neighbors: vec![vec![1]],
            },
            NodeBuild {
                top_layer: 1,
                neighbors: vec![vec![0, 2], vec![]],
            },
            NodeBuild {
                top_layer: 0,
                neighbors: vec![vec![1, 3]],
            },
            NodeBuild {
                top_layer: 0,
                neighbors: vec![vec![2]],
            },
        ];
        BuiltGraph {
            entry_point: 1,
            codec,
            codes,
            nodes,
        }
    }

    fn reason(err: ItzmaError) -> String {
        match err {
            ItzmaError::CorruptIndex { reason, .. } => reason,
            other => format!("{other:?}"),
        }
    }

    #[test]
    fn a_hand_built_graph_round_trips() {
        let built = tiny();
        let bytes = encode(&built, 2, 4, 999).expect("encode");
        let index = Index::decode("idx", &bytes, 2, 4, 999).expect("decode");
        assert_eq!(index.entry_point, 1);
        assert_eq!(index.max_layer, 1);
        assert_eq!(index.nodes.len(), 4);
        assert_eq!(index.nodes[1].neighbors[0], vec![0, 2]);
        assert_eq!(index.dequantize(&index.nodes[0].code)[0], 0.0);
    }

    #[tokio::test]
    async fn search_finds_the_true_nearest_on_a_well_connected_line() {
        let built = tiny();
        let bytes = encode(&built, 2, 4, 999).expect("encode");
        let store = object_store::memory::InMemory::new();
        let path = object_store::path::Path::from("idx.bin");
        {
            use object_store::ObjectStoreExt as _;
            store.put(&path, bytes.into()).await.unwrap();
        }
        let budget = crate::budget::ReadBudget::unlimited();
        let policy = crate::fetch::FetchPolicy::local();
        let probe = Probe {
            store: &store,
            budget: &budget,
            policy: &policy,
            path: &path,
            dim: 2,
            rows: 4,
            data_size: 999,
            metric: Metric::L2,
            query: &[0.0, 0.0],
            want: 2,
            nprobe: Some(1),
            decoded: None,
        };
        let found = Hnsw.search(&probe).await.expect("search");
        let rows: Vec<u32> = found.iter().map(|c| c.row).collect();
        assert_eq!(rows, vec![0, 1], "closest to (0,0) should be rows 0 and 1");
    }

    #[tokio::test]
    async fn the_pool_traversal_equals_the_direct_one() {
        let (rows, dim) = (2_000usize, 16usize);
        let matrix: Vec<f32> = (0..rows * dim)
            .map(|i| ((i * 7919) % 1000) as f32 / 1000.0)
            .collect();
        let bytes = build(&matrix, dim, rows, 999, Metric::L2).expect("build");
        let store = object_store::memory::InMemory::new();
        let path = object_store::path::Path::from("idx.bin");
        {
            use object_store::ObjectStoreExt as _;
            store.put(&path, bytes.clone().into()).await.unwrap();
        }
        let budget = crate::budget::ReadBudget::unlimited();
        let policy = crate::fetch::FetchPolicy::local();
        let query = &matrix[..dim];
        // `ef` = 10 * 400: past the inline threshold, so the search runs on
        // the query lane.
        let probe = Probe {
            store: &store,
            budget: &budget,
            policy: &policy,
            path: &path,
            dim,
            rows: rows as u64,
            data_size: 999,
            metric: Metric::L2,
            query,
            want: 10,
            nprobe: Some(400),
            decoded: None,
        };
        let searched = Hnsw.search(&probe).await.expect("search");
        let prefix = Prefix::decode("idx", &bytes).expect("prefix");
        let body = bytes.slice(PREFIX_LEN..);
        let header = Header::parse("idx", &prefix, &body).expect("header");
        let prepared = header.codec.prepare(Metric::L2, query);
        let direct = traverse("idx", &body, &header, dim, &prepared, 10, 4_000).expect("traverse");
        assert_eq!(searched, direct);
        assert_eq!(direct.len(), 10);
    }

    #[tokio::test]
    async fn an_empty_segment_finds_nothing() {
        let built = BuiltGraph {
            entry_point: 0,
            codec: Sq8Codec::fit(&[], 2).unwrap(),
            codes: Vec::new(),
            nodes: Vec::new(),
        };
        let bytes = encode(&built, 2, 0, 999).expect("encode");
        let store = object_store::memory::InMemory::new();
        let path = object_store::path::Path::from("idx.bin");
        {
            use object_store::ObjectStoreExt as _;
            store.put(&path, bytes.into()).await.unwrap();
        }
        let budget = crate::budget::ReadBudget::unlimited();
        let policy = crate::fetch::FetchPolicy::local();
        let probe = Probe {
            store: &store,
            budget: &budget,
            policy: &policy,
            path: &path,
            dim: 2,
            rows: 0,
            data_size: 999,
            metric: Metric::L2,
            query: &[0.0, 0.0],
            want: 5,
            nprobe: Some(1),
            decoded: None,
        };
        let found = Hnsw.search(&probe).await.expect("search");
        assert!(found.is_empty());
    }

    #[test]
    fn every_truncation_is_refused() {
        let built = tiny();
        let bytes = encode(&built, 2, 4, 999).expect("encode");
        for len in 0..bytes.len() {
            assert!(
                Index::decode("idx", &bytes[..len], 2, 4, 999).is_err(),
                "{len} bytes"
            );
        }
    }

    #[test]
    fn a_flipped_bit_never_panics() {
        let built = tiny();
        let bytes = encode(&built, 2, 4, 999).expect("encode");
        for i in 0..bytes.len() {
            for bit in 0..8 {
                let mut b = bytes.to_vec();
                b[i] ^= 1 << bit;
                let _ = Index::decode("idx", &b, 2, 4, 999);
            }
        }
    }

    #[test]
    fn a_neighbor_count_over_the_cap_is_refused() {
        let mut built = tiny();
        built.nodes[1].neighbors[0] = (0..(M0 as u32 + 1)).collect();
        let err = encode(&built, 2, 4, 999).expect_err("over cap");
        assert!(reason(err).contains("more than the cap"));
    }

    #[test]
    fn a_segment_mismatch_is_refused() {
        let built = tiny();
        let bytes = encode(&built, 2, 4, 999).expect("encode");
        Index::decode("idx", &bytes, 2, 4, 999).expect("matches");
        for (dim, rows, size, want) in [
            (3, 4, 999, "dimension"),
            (2, 5, 999, "rows, the segment has"),
            (2, 4, 1000, "data object"),
        ] {
            let err = Index::decode("idx", &bytes, dim, rows, size).expect_err(want);
            assert!(reason(err).contains(want), "{want}");
        }
    }

    #[test]
    fn a_forged_neighbor_row_past_the_segment_is_refused() {
        let built = tiny();
        let mut bytes = encode(&built, 2, 4, 999).expect("encode").to_vec();
        // Row 1's node starts right after row 0's (dim 2 code + a u16 count(1)
        // + 1 neighbor u32 = 2 + 2 + 4 = 8 bytes). Its own first layer-0
        // neighbor sits after its own code (2) and count (2).
        let dir_end = directory_end(2, 4) as usize;
        let row0_len = 2 + 2 + 4; // code + count + one neighbor at layer 0
        let row1_offset = PREFIX_LEN + dir_end + row0_len;
        let neighbor_at = row1_offset + 2 /* code */ + 2 /* count */;
        bytes[neighbor_at..neighbor_at + 4].copy_from_slice(&9u32.to_le_bytes());
        let err = Index::decode("idx", &bytes, 2, 4, 999).expect_err("past rows");
        assert!(reason(err).contains("past the segment's rows"));
    }
}
