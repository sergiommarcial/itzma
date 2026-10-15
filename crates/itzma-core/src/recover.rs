//! Rebuilding a lost manifest.
//!
//! The manifest is the one object whose loss loses a namespace: it says which
//! segments are live and how far the WAL is folded. Everything it says can be
//! read back from the store. `schema.json` holds the schema, a segment's name
//! is the WAL range it covers, and the WAL listing shows where the log ends and
//! which objects have been replaced by fences.
//!
//! [`Namespace::recover`](crate::Namespace::recover) builds a chain of segments
//! from seq 1, reads the WAL past its end as the tail, and writes the manifest
//! with put-if-absent. It changes nothing else, and it never overwrites a
//! manifest. Segments are pure functions of their WAL range, so any chain that
//! covers the same seqs holds the same data, and the choice between a merged
//! segment and the deltas it replaced does not change what a query returns.
//!
//! What it cannot do is invent data. A range that no readable segment covers
//! and whose WAL objects are missing or fenced has no copy left in the store,
//! and the run stops with [`ItzmaError::Unrecoverable`], naming the objects to
//! restore. There is no way to publish a manifest with a hole in it.

use std::collections::BTreeMap;

use futures::stream::TryStreamExt;
use object_store::path::Path;
use object_store::{ObjectMeta, ObjectStore, ObjectStoreExt};

use crate::budget::{DEFAULT_READ_BUDGET, ReadBudget};
use crate::cpu::Lane;
use crate::error::{ItzmaError, Result};
use crate::manifest::{self, Manifest};
use crate::{gc, schema_object, segment, wal, wal_codec};

/// How many keys an [`ItzmaError::Unrecoverable`] lists. A gap can be long, and
/// the range in the reason says how long.
const MAX_KEYS: usize = 8;

/// What [`Namespace::recover`](crate::Namespace::recover) wrote.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoverReport {
    /// The segment ids the new manifest lists, oldest first.
    pub live_segments: Vec<String>,
    /// The new manifest's `wal_folded_through`: where the chain ends.
    pub folded_through: u64,
    /// WAL objects past the fold point, which readers will fold in as the tail.
    pub tail_objects: u64,
    /// Segment directories that are not in the chain: dead segments that `gc`
    /// has not collected, or segments that did not decode. They are left alone.
    pub unused_segments: usize,
}

/// Where the WAL stands past the end of the chain.
#[derive(Debug, PartialEq, Eq)]
struct Tail {
    /// The last seq of the unbroken run of real (unfenced) objects that starts
    /// right after the chain. Equal to the fold point when there is none.
    end: u64,
    /// The first and last seq of what is missing, if anything that belongs to
    /// the log lies past a break in it.
    gap: Option<(u64, u64)>,
}

/// Find out whether the log past `folded` is whole.
///
/// `wal` maps each WAL seq to its object's size, and `segments` are the ranges
/// of every segment that has a data object, usable or not. Something "lies
/// beyond" the run when a WAL object (real or fence) or a segment reaches past
/// its end, and then the first seq after the run has no copy anywhere.
fn examine_tail(folded: u64, wal: &BTreeMap<u64, u64>, segments: &[(u64, u64)]) -> Tail {
    let real = |seq: u64| {
        wal.get(&seq)
            .is_some_and(|size| *size != wal_codec::FENCE_LEN as u64)
    };
    let mut end = folded;
    while end < u64::MAX && real(end + 1) {
        end += 1;
    }
    let known_end = wal
        .keys()
        .next_back()
        .copied()
        .into_iter()
        .chain(segments.iter().map(|&(_, hi)| hi))
        .max()
        .unwrap_or(0);
    if known_end <= end {
        return Tail { end, gap: None };
    }
    let from = end + 1;
    let to = (from + 1..=known_end)
        .find(|seq| real(*seq))
        .map_or(known_end, |seq| seq - 1);
    Tail {
        end,
        gap: Some((from, to)),
    }
}

/// Read the schema, the segment names and the WAL listing, build the manifest
/// and write it. The caller has validated `namespace`.
pub(crate) async fn rebuild(
    store: &dyn ObjectStore,
    namespace: &str,
) -> Result<(Manifest, RecoverReport)> {
    match store.head(&manifest::manifest_path(namespace)).await {
        Ok(_) => return Err(ItzmaError::ManifestExists(namespace.to_string())),
        Err(object_store::Error::NotFound { .. }) => {}
        Err(e) => return Err(e.into()),
    }
    let schema = schema_object::read(store, namespace)
        .await?
        .ok_or_else(|| ItzmaError::Unrecoverable {
            reason: "schema.json is missing, so the dimension and metric are unknown".into(),
            keys: vec![schema_object::schema_path(namespace).to_string()],
        })?;

    let prefix = Path::from(format!("{namespace}/segments"));
    let listing: Vec<ObjectMeta> = store.list(Some(&prefix)).try_collect().await?;
    let found = gc::group(&prefix, listing);
    let wal: BTreeMap<u64, u64> = gc::list_wal(store, namespace)
        .await?
        .into_iter()
        .map(|object| (object.seq, object.size))
        .collect();

    // A segment directory without its data object is not a segment.
    let with_data: Vec<(u64, u64, String)> = found
        .iter()
        .filter(|(id, dir)| {
            let data = segment::segment_path(namespace, id);
            dir.objects.iter().any(|(path, _)| *path == data)
        })
        .map(|(id, dir)| (dir.lo, dir.hi, id.clone()))
        .collect();

    let (chain, failed) = choose_chain(store, namespace, schema.dim, &with_data).await?;
    let folded = chain.last().map_or(0, |id| {
        segment::parse_segment_id(id).map_or(0, |(_, hi)| hi)
    });
    let ranges: Vec<(u64, u64)> = with_data.iter().map(|(lo, hi, _)| (*lo, *hi)).collect();
    let tail = examine_tail(folded, &wal, &ranges);
    if let Some((from, to)) = tail.gap {
        return Err(gap_error(namespace, from, to, &failed));
    }

    segment::validate_chain(&chain, folded)?;
    let mut manifest = Manifest::new(schema.clone());
    manifest.segment_format = segment_level(store, namespace, &with_data).await?;
    manifest.live_segments = chain.clone();
    manifest.wal_folded_through = folded;
    match manifest::create(store, namespace, &manifest).await {
        Ok(()) => {}
        Err(ItzmaError::NamespaceExists(id)) => return Err(ItzmaError::ManifestExists(id)),
        Err(e) => return Err(e),
    }
    let report = RecoverReport {
        unused_segments: found.len().saturating_sub(chain.len()),
        live_segments: chain,
        folded_through: folded,
        tail_objects: tail.end - folded,
    };
    Ok((manifest, report))
}

/// Build the chain from seq 1: at each step the segment that starts at the
/// next seq, and of those the one that reaches furthest and decodes. Returns the
/// chain and the ids that were tried and did not decode.
///
/// A segment that is missing or damaged only ends the chain early, and the WAL
/// from there on is read as the tail. Any other failure (a format this build
/// does not read, a store error, the read budget) is not damage and is
/// returned.
async fn choose_chain(
    store: &dyn ObjectStore,
    namespace: &str,
    dim: usize,
    with_data: &[(u64, u64, String)],
) -> Result<(Vec<String>, Vec<String>)> {
    let mut by_lo: BTreeMap<u64, Vec<(u64, &str)>> = BTreeMap::new();
    for (lo, hi, id) in with_data {
        by_lo.entry(*lo).or_default().push((*hi, id.as_str()));
    }
    for options in by_lo.values_mut() {
        options.sort_by_key(|option| std::cmp::Reverse(option.0));
    }

    let mut chain = Vec::new();
    let mut failed = Vec::new();
    let mut next = 1u64;
    while let Some(options) = by_lo.get(&next) {
        let mut extended = false;
        for &(hi, id) in options {
            if decodes(store, namespace, dim, id).await? {
                chain.push(id.to_string());
                extended = true;
                match hi.checked_add(1) {
                    Some(after) => next = after,
                    None => return Ok((chain, failed)),
                }
                break;
            }
            failed.push(id.to_string());
        }
        if !extended {
            break;
        }
    }
    Ok((chain, failed))
}

/// The segment level to record: a new namespace's, or the newest version any
/// segment object under the namespace was written in, unused ones
/// included, since the run that wrote one was at that level and a rerun of its
/// range at a lower one would refuse it. A namespace raised with no segment
/// written since comes back at the default level, which nothing records.
async fn segment_level(
    store: &dyn ObjectStore,
    namespace: &str,
    with_data: &[(u64, u64, String)],
) -> Result<u32> {
    let budget = ReadBudget::new(DEFAULT_READ_BUDGET);
    let mut level = segment::DEFAULT_FORMAT;
    for (_, _, id) in with_data {
        let path = segment::segment_path(namespace, id);
        match segment::peek(store, &path, &budget).await {
            Ok(peek) => level = level.max(peek.version),
            // Damage or a vanished object says nothing about the level.
            Err(ItzmaError::CorruptSegment { .. }) => {}
            Err(e) => return Err(e),
        }
    }
    Ok(level)
}

/// Whether the segment is there and decodes for `dim`, one at a time under the
/// default read budget.
async fn decodes(store: &dyn ObjectStore, namespace: &str, dim: usize, id: &str) -> Result<bool> {
    let budget = ReadBudget::new(DEFAULT_READ_BUDGET);
    // recover has no caller preference to read yet (it predates the handle
    // it returns); object_store() is the safe default, same as a fresh
    // Namespace::open's.
    let policy = crate::fetch::FetchPolicy::object_store();
    let ids = [id.to_string()];
    match segment::read_all(
        store,
        namespace,
        &ids,
        dim,
        &budget,
        &policy,
        Lane::Maintenance,
    )
    .await
    {
        Ok(_) => Ok(true),
        Err(ItzmaError::CorruptSegment { .. }) => Ok(false),
        Err(e) => Err(e),
    }
}

fn gap_error(namespace: &str, from: u64, to: u64, failed: &[String]) -> ItzmaError {
    let (span, verb) = if from == to {
        (format!("seq {from}"), "has")
    } else {
        (format!("seqs {from} through {to}"), "have")
    };
    let mut keys: Vec<String> = failed
        .iter()
        .filter(|id| segment::parse_segment_id(id).is_some_and(|(lo, hi)| lo <= to && hi >= from))
        .map(|id| segment::segment_path(namespace, id).to_string())
        .collect();
    keys.extend(
        (from..=to)
            .take(MAX_KEYS)
            .map(|seq| wal::wal_path(namespace, seq).to_string()),
    );
    keys.truncate(MAX_KEYS);
    ItzmaError::Unrecoverable {
        reason: format!(
            "{span} {verb} no readable copy: no segment that decodes covers the range, and the \
             WAL is missing or fenced there"
        ),
        keys,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn log(sizes: &[(u64, u64)]) -> BTreeMap<u64, u64> {
        sizes.iter().copied().collect()
    }

    const OBJ: u64 = 100;
    const FENCE: u64 = wal_codec::FENCE_LEN as u64;

    #[test]
    fn an_empty_store_has_no_tail_and_no_gap() {
        assert_eq!(examine_tail(0, &log(&[]), &[]), Tail { end: 0, gap: None });
    }

    #[test]
    fn a_contiguous_tail_ends_at_its_last_object() {
        let wal = log(&[(1, OBJ), (2, OBJ), (3, OBJ), (4, OBJ)]);
        assert_eq!(examine_tail(2, &wal, &[]), Tail { end: 4, gap: None });
    }

    #[test]
    fn fences_at_or_below_the_fold_point_are_normal() {
        let wal = log(&[(1, FENCE), (2, FENCE), (3, OBJ)]);
        assert_eq!(examine_tail(2, &wal, &[(1, 2)]), Tail { end: 3, gap: None });
    }

    #[test]
    fn a_missing_object_in_the_middle_is_a_gap() {
        let wal = log(&[(1, OBJ), (2, OBJ), (4, OBJ)]);
        assert_eq!(
            examine_tail(0, &wal, &[]),
            Tail {
                end: 2,
                gap: Some((3, 3))
            }
        );
    }

    #[test]
    fn a_fence_past_the_chain_is_a_gap_that_runs_to_the_next_real_object() {
        let wal = log(&[(1, OBJ), (2, FENCE), (3, FENCE), (4, OBJ)]);
        assert_eq!(
            examine_tail(1, &wal, &[]),
            Tail {
                end: 1,
                gap: Some((2, 3))
            }
        );
    }

    #[test]
    fn a_segment_that_reaches_past_the_log_is_a_gap_when_nothing_covers_it() {
        // The chain ended at 2, the WAL stops there, and a segment 3-5 exists
        // but was not usable.
        let wal = log(&[(1, FENCE), (2, FENCE)]);
        assert_eq!(
            examine_tail(2, &wal, &[(1, 2), (3, 5)]),
            Tail {
                end: 2,
                gap: Some((3, 5))
            }
        );
    }

    #[test]
    fn segments_inside_the_chain_do_not_make_a_gap() {
        let wal = log(&[(1, FENCE), (2, FENCE), (3, FENCE)]);
        assert_eq!(
            examine_tail(3, &wal, &[(1, 3), (1, 1), (2, 2)]),
            Tail { end: 3, gap: None }
        );
    }

    #[test]
    fn the_error_names_the_range_and_the_keys_to_restore() {
        let err = gap_error(
            "docs",
            2,
            3,
            &[segment::segment_id(2, 2), segment::segment_id(9, 9)],
        );
        let ItzmaError::Unrecoverable { reason, keys } = err else {
            panic!("unexpected error");
        };
        assert!(
            reason.contains("seqs 2 through 3 have no readable copy"),
            "{reason}"
        );
        assert_eq!(
            keys,
            [
                segment::segment_path("docs", &segment::segment_id(2, 2)).to_string(),
                wal::wal_path("docs", 2).to_string(),
                wal::wal_path("docs", 3).to_string(),
            ]
        );
    }

    #[test]
    fn a_long_gap_lists_only_a_few_keys() {
        let ItzmaError::Unrecoverable { keys, .. } = gap_error("docs", 1, 500, &[]) else {
            panic!("unexpected error");
        };
        assert_eq!(keys.len(), MAX_KEYS);
    }

    #[test]
    fn a_one_seq_gap_reads_naturally() {
        let ItzmaError::Unrecoverable { reason, .. } = gap_error("docs", 7, 7, &[]) else {
            panic!("unexpected error");
        };
        assert!(reason.starts_with("seq 7 has no readable copy"), "{reason}");
    }
}
