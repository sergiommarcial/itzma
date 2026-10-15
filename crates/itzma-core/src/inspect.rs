//! A read-only look at one namespace: what is stored and what looks wrong.
//!
//! [`report`] loads the manifest and `schema.json` and lists `wal/` and
//! `segments/`. It never writes, and unlike [`Namespace::open`] it does not
//! heal a missing schema object, so it is safe to point at a namespace someone
//! else is running, or at one that is damaged. Nothing it finds is repaired:
//! a problem is a line in [`Report::problems`].
//!
//! The listings come after the manifest, as in [`gc`], so a segment
//! or a WAL object written in between is past the fold point and never looks
//! like damage. A namespace that is being written to can still show a passing
//! problem, such as a gap while a compaction's segment is half listed. Run it
//! again before believing it.
//!
//! It reads no object bodies apart from the manifest and the schema, so its
//! cost is two listings and two small GETs however large the namespace is. That
//! also means it cannot tell a WAL object's format version, or a segment's
//! document count, or whether an index object decodes: it shows which index the
//! manifest names for each segment and whether that object is there, and a
//! query is what finds a damaged one. A WAL object of exactly 8 bytes is a
//! fence, since every real object is longer.
//!
//! [`Namespace::open`]: crate::Namespace::open

use std::collections::{BTreeMap, HashSet};
use std::fmt;

use futures::TryStreamExt;
use object_store::path::Path;
use object_store::{ObjectMeta, ObjectStore};

use crate::document::Schema;
use crate::error::{ItzmaError, Result};
use crate::gc::{self, Found};
use crate::manifest::{self, IndexEntry, Manifest};
use crate::namespace::validate_id;
use crate::{schema_object, segment, wal_codec};

/// What [`report`] found in one namespace.
#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    /// The namespace id that was inspected.
    pub namespace: String,
    /// The schema in `schema.json`, or `None` if the object is missing or
    /// unreadable (the second is also a problem).
    pub schema: Option<Schema>,
    /// The manifest, or `None` if it is missing or unreadable (also a problem).
    pub manifest: Option<Manifest>,
    /// The WAL objects listed under `wal/`.
    pub wal: WalReport,
    /// Every segment directory listed under `segments/`, oldest range first.
    pub segments: Vec<SegmentReport>,
    /// What looks wrong, one sentence each. Empty means nothing did.
    pub problems: Vec<String>,
}

/// A summary of the WAL objects in a namespace.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct WalReport {
    /// How many WAL objects there are, fences included.
    pub objects: usize,
    /// The lowest seq listed.
    pub first: Option<u64>,
    /// The highest seq listed.
    pub last: Option<u64>,
    /// Total stored size of the objects.
    pub bytes: u64,
    /// How many are fences (8 bytes), left by garbage collection.
    pub fenced: usize,
    /// How many are past the manifest's fold point, so a query still reads
    /// them. Without a manifest, all of them.
    pub tail: usize,
    /// Runs of missing seqs as inclusive `(from, to)` pairs, from 1 up to the
    /// higher of the last seq listed and the fold point.
    pub gaps: Vec<(u64, u64)>,
}

/// One directory under `segments/`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentReport {
    /// The segment id, `<lo>-<hi>` (see [`segment`]).
    pub id: String,
    /// First WAL seq the segment covers.
    pub lo: u64,
    /// Last WAL seq the segment covers.
    pub hi: u64,
    /// Total stored size of the objects in the directory.
    pub bytes: u64,
    /// Whether the manifest lists it, and if not, whether [`Namespace::gc`]
    /// would take it.
    ///
    /// [`Namespace::gc`]: crate::Namespace::gc
    pub state: SegmentState,
    /// The index the manifest names for this segment, if it names one.
    pub index: Option<IndexInfo>,
    /// Generations of index objects in the directory that the manifest does
    /// not name, oldest first. [`Namespace::gc`](crate::Namespace::gc) deletes
    /// them of a live segment once they are old enough.
    pub stray_indexes: Vec<u32>,
    /// Whether the directory has a `sparse.bin`, so some document in the
    /// segment carries a sparse vector.
    pub has_sparse_data: bool,
}

/// An index the manifest names, and whether its object is there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexInfo {
    /// The generation, as `index-<generation>.bin`.
    pub generation: u32,
    /// The algorithm's name in the manifest, such as `sq8`.
    pub kind: String,
    /// The object's stored size, or `None` if it is not in the directory (also
    /// a problem).
    pub bytes: Option<u64>,
}

/// How a segment on disk stands against the manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentState {
    /// The manifest lists it.
    Live,
    /// Not live, and superseded: garbage collection deletes it once it is old
    /// enough. The rule is the one [`Namespace::gc`](crate::Namespace::gc)
    /// uses.
    Dead,
    /// Not live and not superseded. A compaction that has not published yet
    /// looks like this, and so does one that stopped before it did. It is also
    /// what GC leaves alone at `1-<folded>`.
    Orphan,
}

impl fmt::Display for SegmentState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(match self {
            Self::Live => "live",
            Self::Dead => "dead",
            Self::Orphan => "orphan",
        })
    }
}

/// Inspect the namespace `namespace` in `store`. See the module docs.
///
/// A manifest or schema object that is missing or cannot be decoded is
/// reported in [`Report::problems`], not returned as an error, so the rest of
/// the namespace still shows.
///
/// # Errors
///
/// - [`ItzmaError::InvalidNamespaceId`] for a malformed id.
/// - [`ItzmaError::Storage`] if the store fails.
///
/// # Examples
///
/// ```
/// # #[tokio::main]
/// # async fn main() -> itzma_core::Result<()> {
/// use std::sync::Arc;
///
/// use itzma_core::{Metric, Namespace, Schema};
/// use object_store::memory::InMemory;
///
/// let store = Arc::new(InMemory::new());
/// Namespace::create(store.clone(), "docs", Schema::new(2, Metric::L2)).await?;
///
/// let report = itzma_core::inspect::report(&*store, "docs").await?;
/// assert!(report.problems.is_empty());
/// assert_eq!(report.wal.objects, 0);
/// # Ok(())
/// # }
/// ```
pub async fn report(store: &dyn ObjectStore, namespace: &str) -> Result<Report> {
    validate_id(namespace)?;
    let mut problems = Vec::new();

    let manifest = match manifest::load(store, namespace).await {
        Ok((manifest, _)) => Some(manifest),
        Err(e) => {
            problems.push(unreadable("manifest", e)?);
            None
        }
    };
    let schema = match schema_object::read(store, namespace).await {
        Ok(schema) => schema,
        Err(e) => {
            problems.push(unreadable("schema.json", e)?);
            None
        }
    };
    if let (Some(manifest), Some(schema)) = (&manifest, &schema) {
        if &manifest.schema != schema {
            problems.push(format!(
                "schema.json holds {schema:?} but the manifest holds {:?}",
                manifest.schema
            ));
        }
    }

    let prefix = Path::from(format!("{namespace}/segments"));
    let listing: Vec<ObjectMeta> = store.list(Some(&prefix)).try_collect().await?;
    let found = gc::group(&prefix, listing);
    let wal_objects = gc::list_wal(store, namespace).await?;

    let folded = manifest.as_ref().map_or(0, |m| m.wal_folded_through);
    let wal = summarize_wal(
        wal_objects.iter().map(|o| (o.seq, o.size)).collect(),
        folded,
    );
    for &(from, to) in &wal.gaps {
        problems.push(if from == to {
            format!("WAL seq {from} is missing")
        } else {
            format!("WAL seqs {from} through {to} are missing")
        });
    }

    let mut segments = Vec::new();
    if let Some(manifest) = &manifest {
        if let Err(e) = segment::validate_chain(&manifest.live_segments, folded) {
            problems.push(e.to_string());
        }
        for id in &manifest.live_segments {
            let data = segment::segment_path(namespace, id);
            let there = found
                .get(id)
                .is_some_and(|f| f.objects.iter().any(|(path, _)| *path == data));
            if !there {
                problems.push(format!("live segment {id} has no data object"));
            }
        }
        let live: HashSet<&str> = manifest.live_segments.iter().map(String::as_str).collect();
        segments = found
            .iter()
            .map(|(id, f)| segment_report(id, f, folded, &live, &manifest.indexes))
            .collect();
        for report in &segments {
            if let Some(index) = &report.index {
                if index.bytes.is_none() {
                    problems.push(format!(
                        "segment {} is listed with index generation {} but has no such index object",
                        report.id, index.generation
                    ));
                }
            }
        }
        for id in manifest.indexes.keys() {
            if !live.contains(id.as_str()) {
                problems.push(format!(
                    "the manifest lists an index for {id}, which is not a live segment"
                ));
            }
        }
    } else {
        segments.extend(
            found
                .iter()
                .map(|(id, f)| segment_report(id, f, 0, &HashSet::new(), &BTreeMap::new())),
        );
    }

    Ok(Report {
        namespace: namespace.to_string(),
        schema,
        manifest,
        wal,
        segments,
        problems,
    })
}

/// A missing or undecodable object becomes a problem line. A failure of the
/// store itself is not something to describe, so it is returned.
fn unreadable(what: &str, err: ItzmaError) -> Result<String> {
    match err {
        ItzmaError::NamespaceNotFound(_) => Ok(format!("{what} is missing")),
        ItzmaError::Serde { .. }
        | ItzmaError::UnsupportedFormat { .. }
        | ItzmaError::WalFormatTooNew { .. }
        | ItzmaError::SegmentFormatTooNew { .. } => Ok(format!("{what} cannot be read: {err}")),
        other => Err(other),
    }
}

fn segment_report(
    id: &str,
    found: &Found,
    folded: u64,
    live: &HashSet<&str>,
    indexes: &BTreeMap<String, IndexEntry>,
) -> SegmentReport {
    let state = if live.contains(id) {
        SegmentState::Live
    } else if gc::is_dead(folded, live, id, found) {
        SegmentState::Dead
    } else {
        SegmentState::Orphan
    };
    let named = indexes.get(id).filter(|_| state == SegmentState::Live);
    let index = named.map(|entry| IndexInfo {
        generation: entry.generation,
        kind: entry.kind.clone(),
        bytes: found
            .indexes
            .iter()
            .find(|object| object.generation == entry.generation)
            .map(|object| object.size),
    });
    let mut stray_indexes: Vec<u32> = found
        .indexes
        .iter()
        .map(|object| object.generation)
        .filter(|generation| Some(*generation) != named.map(|entry| entry.generation))
        .collect();
    stray_indexes.sort_unstable();
    let has_sparse_data = found
        .objects
        .iter()
        .any(|(path, _)| path.filename() == Some("sparse.bin"));
    SegmentReport {
        id: id.to_string(),
        lo: found.lo,
        hi: found.hi,
        bytes: found.objects.iter().map(|(_, size)| size).sum(),
        state,
        index,
        stray_indexes,
        has_sparse_data,
    }
}

/// Summarize `(seq, size)` pairs against the fold point. Pure: no storage.
fn summarize_wal(mut objects: Vec<(u64, u64)>, folded: u64) -> WalReport {
    objects.sort_unstable();
    let mut gaps = Vec::new();
    let mut prev = 0u64;
    for &(seq, _) in &objects {
        if seq > prev.saturating_add(1) {
            gaps.push((prev + 1, seq - 1));
        }
        prev = prev.max(seq);
    }
    if folded > prev {
        gaps.push((prev + 1, folded));
    }
    WalReport {
        objects: objects.len(),
        first: objects.first().map(|&(seq, _)| seq),
        last: objects.last().map(|&(seq, _)| seq),
        bytes: objects.iter().map(|&(_, size)| size).sum(),
        fenced: objects
            .iter()
            .filter(|&&(_, size)| size == wal_codec::FENCE_LEN as u64)
            .count(),
        tail: objects.iter().filter(|&&(seq, _)| seq > folded).count(),
        gaps,
    }
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "namespace {}", self.namespace)?;
        match &self.schema {
            Some(schema) => writeln!(f, "schema.json  dim {}, {:?}", schema.dim, schema.metric)?,
            None => writeln!(f, "schema.json  none")?,
        }
        match &self.manifest {
            Some(m) => writeln!(
                f,
                "manifest     wal_format {}, segment_format {}, folded through {}, {} live segments",
                m.wal_format,
                m.segment_format,
                m.wal_folded_through,
                m.live_segments.len()
            )?,
            None => writeln!(f, "manifest     none")?,
        }
        let wal = &self.wal;
        write!(f, "wal          {} objects", wal.objects)?;
        if let (Some(first), Some(last)) = (wal.first, wal.last) {
            write!(f, ", seq {first} through {last}")?;
        }
        writeln!(
            f,
            ", {} bytes, {} fenced, {} in the tail",
            wal.bytes, wal.fenced, wal.tail
        )?;
        writeln!(f, "segments     {} on disk", self.segments.len())?;
        for s in &self.segments {
            write!(f, "  {}  {:<6}  {} bytes", s.id, s.state, s.bytes)?;
            if let Some(index) = &s.index {
                match index.bytes {
                    Some(bytes) => {
                        write!(
                            f,
                            "  index {} {} {bytes} bytes",
                            index.generation, index.kind
                        )?;
                    }
                    None => write!(f, "  index {} {} missing", index.generation, index.kind)?,
                }
            }
            if !s.stray_indexes.is_empty() {
                write!(f, "  unnamed indexes {:?}", s.stray_indexes)?;
            }
            if s.has_sparse_data {
                write!(f, "  sparse")?;
            }
            writeln!(f)?;
        }
        if self.problems.is_empty() {
            writeln!(f, "problems     none")
        } else {
            writeln!(f, "problems     {}", self.problems.len())?;
            for problem in &self.problems {
                writeln!(f, "  {problem}")?;
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seqs(list: &[(u64, u64)], folded: u64) -> WalReport {
        summarize_wal(list.to_vec(), folded)
    }

    #[test]
    fn an_empty_wal_has_no_gaps() {
        assert_eq!(seqs(&[], 0), WalReport::default());
    }

    #[test]
    fn a_contiguous_run_has_no_gaps() {
        let r = seqs(&[(1, 100), (2, 50), (3, 8)], 2);
        assert_eq!(r.objects, 3);
        assert_eq!((r.first, r.last), (Some(1), Some(3)));
        assert_eq!(r.bytes, 158);
        assert_eq!(r.fenced, 1);
        assert_eq!(r.tail, 1);
        assert!(r.gaps.is_empty());
    }

    #[test]
    fn the_input_order_does_not_matter() {
        let r = seqs(&[(3, 20), (1, 20), (2, 20)], 0);
        assert!(r.gaps.is_empty());
        assert_eq!((r.first, r.last), (Some(1), Some(3)));
    }

    #[test]
    fn a_missing_start_and_a_missing_middle_are_gaps() {
        let r = seqs(&[(3, 20), (4, 20), (8, 20)], 0);
        assert_eq!(r.gaps, vec![(1, 2), (5, 7)]);
    }

    #[test]
    fn a_fold_point_past_the_last_object_is_a_gap() {
        let r = seqs(&[(1, 20), (2, 20)], 5);
        assert_eq!(r.gaps, vec![(3, 5)]);
        assert_eq!(r.tail, 0);
    }

    #[test]
    fn a_fold_point_at_the_last_object_is_not() {
        assert!(seqs(&[(1, 20), (2, 20)], 2).gaps.is_empty());
    }

    #[test]
    fn a_seq_listed_twice_is_not_a_gap() {
        let r = seqs(&[(1, 20), (1, 20), (2, 20)], 0);
        assert!(r.gaps.is_empty());
    }
}
