//! Building the sparse inverted index of one segment and publishing it
//!: [`Namespace::index_sparse`](crate::Namespace::index_sparse).
//! Mirrors `indexer.rs`'s shape for the dense index — put-if-absent, then
//! one manifest compare-and-swap — with one algorithm, so there is no `kind`.

use std::collections::BTreeMap;

use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt, PutMode, PutOptions};
use serde::{Deserialize, Serialize};

use crate::budget::ReadBudget;
use crate::error::{ItzmaError, Result};
use crate::fetch::FetchPolicy;
use crate::manifest::{self, SparseIndexEntry};
use crate::segment;
use crate::sparse;
use crate::sparse_index;

/// Settings for [`Namespace::index_sparse`](crate::Namespace::index_sparse).
///
/// New fields will be added, so build one from `default()` and the setters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct IndexSparseOptions {
    /// Build a new generation for a segment that already has a sparse
    /// index, instead of leaving it alone. The old object stays until
    /// [`gc`](crate::Namespace::gc) collects it.
    pub rebuild: bool,
}

impl IndexSparseOptions {
    /// Set [`rebuild`](Self::rebuild).
    #[must_use]
    pub fn rebuild(mut self, rebuild: bool) -> Self {
        self.rebuild = rebuild;
        self
    }
}

/// What a call to [`Namespace::index_sparse`](crate::Namespace::index_sparse)
/// did, one count per live segment it looked at. Every segment lands in
/// exactly one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct IndexSparseReport {
    /// Segments this run published a sparse index for.
    pub indexed: usize,
    /// Segments left alone because they have no `sparse.bin` (no document in
    /// them carries a sparse vector).
    pub skipped_no_sparse: usize,
    /// Segments that already had a sparse index (and `rebuild` was not set),
    /// or that another indexer published while this run built its own.
    pub already_indexed: usize,
    /// Segments a compaction retired while this run was working on them.
    pub superseded: usize,
}

/// What became of one segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Outcome {
    Indexed,
    SkippedNoSparse,
    AlreadyIndexed,
    Superseded,
}

impl IndexSparseReport {
    pub(crate) fn record(&mut self, outcome: Outcome) {
        let count = match outcome {
            Outcome::Indexed => &mut self.indexed,
            Outcome::SkippedNoSparse => &mut self.skipped_no_sparse,
            Outcome::AlreadyIndexed => &mut self.already_indexed,
            Outcome::Superseded => &mut self.superseded,
        };
        *count += 1;
    }
}

/// Index the live segment `id`'s sparse sidecar, whose current entry in the
/// manifest is `have`.
///
/// # Errors
///
/// Whatever reading `sparse.bin` gives ([`ItzmaError::CorruptSegment`]; a
/// missing object is [`sparse::is_missing`], turned into
/// [`Outcome::SkippedNoSparse`] here, not an error — a segment legitimately
/// has none), [`ItzmaError::CorruptSparseIndex`] if an object already at the
/// new key does not fit the segment, and the errors of a manifest update.
pub(crate) async fn index_segment(
    store: &dyn ObjectStore,
    namespace: &str,
    id: &str,
    have: Option<&SparseIndexEntry>,
    options: &IndexSparseOptions,
    read_budget: u64,
    policy: &FetchPolicy,
) -> Result<Outcome> {
    if have.is_some() && !options.rebuild {
        return Ok(Outcome::AlreadyIndexed);
    }
    let budget = ReadBudget::new(read_budget);
    let data_path = segment::segment_path(namespace, id);
    let head = segment::peek(store, &data_path, &budget).await?;
    let rows = u32::try_from(head.doc_count).map_err(|_| ItzmaError::CorruptSegment {
        path: data_path.to_string(),
        reason: "too many rows for a sparse index's u32 row count".into(),
    })?;

    let sparse_path = sparse::sparse_path(namespace, id);
    let bytes = match sparse::fetch_bytes(store, &sparse_path, &budget, policy).await {
        Ok(bytes) => bytes,
        Err(e) if sparse::is_missing(&e) => return Ok(Outcome::SkippedNoSparse),
        Err(e) => return Err(e),
    };
    let rows_decoded = sparse::decode(sparse_path.as_ref(), &bytes, u64::from(rows))?;

    let mut postings: BTreeMap<u32, Vec<(u32, f32)>> = BTreeMap::new();
    for (row, terms) in rows_decoded.into_iter().enumerate() {
        let Some(terms) = terms else { continue };
        for (term, weight) in terms {
            postings.entry(term).or_default().push((row as u32, weight));
        }
    }

    let names = segment::object_names(store, namespace, id).await?;
    let floor = segment::generation_floor(
        have.map(|entry| entry.generation),
        &names,
        sparse_index::generation_of,
    );
    let generation = segment::claim_generation(store, namespace, id, floor).await?;
    let path = sparse_index::sparse_index_path(namespace, id, generation);
    let built = sparse_index::build(rows, &postings);
    put_or_reuse(store, &path, built, u64::from(rows)).await?;

    publish(store, namespace, id, have, generation).await
}

/// Write the sparse index with put-if-absent. An object already at the key is
/// an orphan of an earlier run or the winner of a race, and it is used if it
/// checks out as a valid sparse index of this segment.
async fn put_or_reuse(
    store: &dyn ObjectStore,
    path: &Path,
    built: bytes::Bytes,
    rows: u64,
) -> Result<()> {
    match store
        .put_opts(path, built.into(), PutOptions::from(PutMode::Create))
        .await
    {
        Ok(_) => Ok(()),
        Err(object_store::Error::AlreadyExists { .. }) => {
            let existing = store.get(path).await?.bytes().await?;
            sparse_index::open_bytes_for_check(path.as_ref(), &existing, rows).map_err(|err| {
                match err {
                    ItzmaError::CorruptSparseIndex { path, reason } => {
                        ItzmaError::CorruptSparseIndex {
                            path,
                            reason: format!(
                                "{reason}; an object already at this key does not fit the \
                                 segment, delete it and rerun"
                            ),
                        }
                    }
                    other => other,
                }
            })
        }
        Err(e) => Err(e.into()),
    }
}

/// List the sparse index in the manifest, if the manifest still looks the way
/// it did when the run planned: the segment live and its entry unchanged.
async fn publish(
    store: &dyn ObjectStore,
    namespace: &str,
    id: &str,
    have: Option<&SparseIndexEntry>,
    generation: u32,
) -> Result<Outcome> {
    let mut lost = Outcome::AlreadyIndexed;
    let published = manifest::update_if(store, namespace, |m| {
        if !m.live_segments.iter().any(|live| live == id) {
            lost = Outcome::Superseded;
            return false;
        }
        if m.sparse_indexes.get(id) != have {
            lost = Outcome::AlreadyIndexed;
            return false;
        }
        m.sparse_indexes
            .insert(id.to_string(), SparseIndexEntry { generation });
        true
    })
    .await?;
    Ok(published.map_or(lost, |_| Outcome::Indexed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebuild_setter_changes_the_default() {
        assert!(!IndexSparseOptions::default().rebuild);
        assert!(IndexSparseOptions::default().rebuild(true).rebuild);
    }
}
