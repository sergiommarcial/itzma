//! Building the index of one segment and publishing it:
//! [`Namespace::index`](crate::Namespace::index) indexes each live segment in
//! turn.
//!
//! An index is derived data. It is written with put-if-absent under a name that
//! carries its generation, so a build never overwrites an object a reader may
//! be using, and it becomes part of the namespace when one manifest
//! compare-and-swap lists it. Until then it is an orphan, and a crash between
//! the two costs nothing: the next run finds the object, checks it fits the
//! segment, and publishes it. Nothing byte-compares two builds, since any valid
//! index is correct (unlike a segment).

use bytes::Bytes;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt, PutMode, PutOptions};
use serde::{Deserialize, Serialize};

use crate::budget::ReadBudget;
use crate::cpu::{self, Cost, Lane};
use crate::document::Metric;
use crate::error::{ItzmaError, Result};
use crate::fetch::FetchPolicy;
use crate::index::{self, Kind, sq8};
use crate::manifest::{self, IndexEntry};
use crate::segment;

/// The default for [`IndexOptions::min_rows`].
pub const DEFAULT_MIN_ROWS: usize = 4096;

/// The segment format level `index` needs: an index is built for version 2.
pub(crate) const MIN_SEGMENT_FORMAT: u32 = 2;

/// Settings for [`Namespace::index`](crate::Namespace::index).
///
/// New fields will be added, so build one from `default()` and the setters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct IndexOptions {
    /// A segment with fewer documents than this is not indexed and is read
    /// whole, since a flat read of a small segment costs about what the index
    /// head does. Zero indexes everything, an empty segment included.
    pub min_rows: usize,
    /// Build a new generation for a segment that already has an index, instead
    /// of leaving it alone. The old object stays until
    /// [`gc`](crate::Namespace::gc) collects it.
    pub rebuild: bool,
    /// Which algorithm to build. Different segments may use
    /// different kinds; rebuilding a segment to a different kind replaces its
    /// entry like any other rebuild. Defaults to [`IndexKind::Sq8`].
    pub kind: IndexKind,
    /// How many partitions an SQ8 index has. `None` chooses from the
    /// segment's row count: one partition for a small segment and more for a big
    /// one. A query then reads only the few partitions nearest to it. `Some(0)` is
    /// [`ItzmaError::InvalidQuery`], and a number
    /// above the segment's rows is cut to it.
    pub nlist: Option<usize>,
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self {
            min_rows: DEFAULT_MIN_ROWS,
            rebuild: false,
            kind: IndexKind::default(),
            nlist: None,
        }
    }
}

impl IndexOptions {
    /// Set [`min_rows`](Self::min_rows).
    #[must_use]
    pub fn min_rows(mut self, min_rows: usize) -> Self {
        self.min_rows = min_rows;
        self
    }

    /// Set [`rebuild`](Self::rebuild).
    #[must_use]
    pub fn rebuild(mut self, rebuild: bool) -> Self {
        self.rebuild = rebuild;
        self
    }

    /// Set [`nlist`](Self::nlist).
    #[must_use]
    pub fn nlist(mut self, nlist: Option<usize>) -> Self {
        self.nlist = nlist;
        self
    }

    /// Refuse options that cannot be run, before any storage call.
    pub(crate) fn validate(&self) -> Result<()> {
        if self.nlist == Some(0) {
            return Err(ItzmaError::InvalidQuery("nlist must be at least 1"));
        }
        Ok(())
    }

    /// Set [`kind`](Self::kind).
    #[must_use]
    pub fn kind(mut self, kind: IndexKind) -> Self {
        self.kind = kind;
        self
    }
}

/// The number of partitions [`IndexOptions::nlist`] chooses when it is left at
/// `None`: one below a threshold and about the square root of the rows above
/// it. A caller building a namespace's own report of what an
/// automatic build chose can call this before or after `index` runs; `index`
/// itself uses the same rule.
#[must_use]
pub fn default_nlist(rows: usize) -> usize {
    sq8::default_nlist(rows)
}

/// The number of partitions a query probes when
/// [`QueryOptions::nprobe`](crate::QueryOptions::nprobe) is left at `None`, for
/// an SQ8 index with `nlist` partitions.
#[must_use]
pub fn default_nprobe(nlist: usize) -> usize {
    sq8::default_nprobe(nlist)
}

/// Which algorithm builds a segment's index. New variants may be
/// added, so match on this non-exhaustively.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum IndexKind {
    /// Scalar quantization to 8 bits per dimension. Near-zero build
    /// cost, the same cost every query pays whether the segment is warm or
    /// cold.
    #[default]
    Sq8,
}

impl IndexKind {
    fn as_kind(self) -> Kind {
        match self {
            IndexKind::Sq8 => Kind::Sq8,
        }
    }
}

/// What a call to [`Namespace::index`](crate::Namespace::index) did, one count
/// per live segment it looked at. Every segment lands in exactly one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct IndexReport {
    /// Segments this run published an index for.
    pub indexed: usize,
    /// Segments left alone because they hold fewer than
    /// [`IndexOptions::min_rows`] documents.
    pub skipped_small: usize,
    /// Segments that already had an index (and `rebuild` was not set), or that
    /// another indexer published while this run built its own.
    pub already_indexed: usize,
    /// Segments a compaction retired while this run was working on them. Their
    /// index object, if it was written, is left for [`gc`](crate::Namespace::gc).
    pub superseded: usize,
}

/// What became of one segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Outcome {
    Indexed,
    SkippedSmall,
    AlreadyIndexed,
    Superseded,
}

impl IndexReport {
    pub(crate) fn record(&mut self, outcome: Outcome) {
        let count = match outcome {
            Outcome::Indexed => &mut self.indexed,
            Outcome::SkippedSmall => &mut self.skipped_small,
            Outcome::AlreadyIndexed => &mut self.already_indexed,
            Outcome::Superseded => &mut self.superseded,
        };
        *count += 1;
    }
}

/// Index the live segment `id`, whose current entry in the manifest is `have`.
///
/// Reads 20 bytes to decide whether the segment is worth it, and only then the
/// whole data object, one segment at a time and under a budget of its own, so a
/// namespace of many segments is not bounded by their sum. A segment bigger than
/// `read_budget` cannot be indexed.
///
/// # Errors
///
/// Whatever reading the segment gives (a missing object is
/// [`segment::is_missing`], which the caller turns into `Superseded` when the
/// manifest has moved), [`ItzmaError::CorruptIndex`] if an object already at the
/// new key does not fit the segment, and the errors of a manifest update.
#[allow(clippy::too_many_arguments)] // the metric joined dim as a per-segment parameter
pub(crate) async fn index_segment(
    store: &dyn ObjectStore,
    namespace: &str,
    dim: usize,
    metric: Metric,
    id: &str,
    have: Option<&IndexEntry>,
    options: &IndexOptions,
    read_budget: u64,
    policy: &FetchPolicy,
) -> Result<Outcome> {
    if have.is_some() && !options.rebuild {
        return Ok(Outcome::AlreadyIndexed);
    }
    let budget = ReadBudget::new(read_budget);
    let data_path = segment::segment_path(namespace, id);

    let head = segment::peek(store, &data_path, &budget).await?;
    if usize::try_from(head.doc_count).is_ok_and(|rows| rows < options.min_rows) {
        return Ok(Outcome::SkippedSmall);
    }

    let bytes = segment::fetch_bytes(store, &data_path, &budget, policy).await?;
    let data_size = bytes.len() as u64;
    let layer = segment::decode_on(Lane::Maintenance, data_path.clone(), bytes, dim).await?;
    let docs = layer.sorted_docs();
    let matrix: Vec<f32> = docs.iter().flat_map(|d| d.vector.iter().copied()).collect();
    let rows = docs.len();

    let names = segment::object_names(store, namespace, id).await?;
    let floor = segment::generation_floor(
        have.map(|entry| entry.generation),
        &names,
        index::generation_of,
    );
    let generation = segment::claim_generation(store, namespace, id, floor).await?;
    let path = index::index_path(namespace, id, generation);
    let kind = options.kind.as_kind();
    let nlist = options.nlist.unwrap_or_else(|| sq8::default_nlist(rows));
    let cost = build_cost(kind, rows, nlist, dim);
    let built = cpu::run(Lane::Maintenance, cost, move || match kind {
        Kind::Sq8 => sq8::build(&matrix, dim, rows, data_size, nlist, metric),
    })
    .await?;
    put_or_reuse(store, &path, built, kind, dim, rows as u64, data_size).await?;

    publish(store, namespace, id, have, generation, kind).await
}

/// How much work a build is, for the compute pool's hand-off: SQ8
/// compares every row with every centroid.
fn build_cost(kind: Kind, rows: usize, nlist: usize, dim: usize) -> Cost {
    let per_row = match kind {
        Kind::Sq8 => nlist.max(1),
    };
    Cost::Distances {
        rows: rows.saturating_mul(per_row),
        dim,
    }
}

/// Write the index with put-if-absent. An object already at the key is an
/// orphan of an earlier run or the winner of a race, and it is used if it is a
/// valid index of this segment, whoever built it and however it was built.
async fn put_or_reuse(
    store: &dyn ObjectStore,
    path: &Path,
    built: Bytes,
    kind: Kind,
    dim: usize,
    rows: u64,
    data_size: u64,
) -> Result<()> {
    match store
        .put_opts(path, built.into(), PutOptions::from(PutMode::Create))
        .await
    {
        Ok(_) => Ok(()),
        Err(object_store::Error::AlreadyExists { .. }) => {
            let existing = store.get(path).await?.bytes().await?;
            let checked = match kind {
                Kind::Sq8 => {
                    sq8::Index::decode(path.as_ref(), &existing, dim, rows, data_size).map(|_| ())
                }
            };
            checked.map_err(|err| match err {
                ItzmaError::CorruptIndex { path, reason } => ItzmaError::CorruptIndex {
                    path,
                    reason: format!(
                        "{reason}; an object already at this key does not fit the segment, \
                         delete it and rerun"
                    ),
                },
                other => other,
            })
        }
        Err(e) => Err(e.into()),
    }
}

/// List the index in the manifest, if the manifest still looks the way it did
/// when the run planned: the segment live and its entry unchanged.
async fn publish(
    store: &dyn ObjectStore,
    namespace: &str,
    id: &str,
    have: Option<&IndexEntry>,
    generation: u32,
    kind: Kind,
) -> Result<Outcome> {
    let mut lost = Outcome::AlreadyIndexed;
    let published = manifest::update_if(store, namespace, |m| {
        if !m.live_segments.iter().any(|live| live == id) {
            lost = Outcome::Superseded;
            return false;
        }
        if m.indexes.get(id) != have {
            lost = Outcome::AlreadyIndexed;
            return false;
        }
        m.indexes.insert(
            id.to_string(),
            IndexEntry {
                generation,
                kind: kind.name().to_string(),
            },
        );
        true
    })
    .await?;
    Ok(published.map_or(lost, |_| Outcome::Indexed))
}

#[cfg(test)]
mod tests {

    #[test]
    fn every_build_worth_indexing_leaves_the_tokio_worker() {
        use crate::cpu::INLINE_BELOW;
        // SQ8 with clustering: every row against every centroid, five times.
        let cost = build_cost(Kind::Sq8, 100_000, 316, 25);
        assert!(cost.units() >= INLINE_BELOW);
    }

    use super::*;

    #[test]
    fn options_default_to_sq8() {
        assert_eq!(IndexOptions::default().kind, IndexKind::Sq8);
        let opts = IndexOptions::default().kind(IndexKind::Sq8);
        assert_eq!(opts.kind, IndexKind::Sq8);
    }
}
