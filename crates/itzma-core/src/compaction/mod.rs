//! Compaction: fold the WAL tail into a segment, or merge the segments.
//!
//! One run is: load the manifest, read the WAL tail, decide what to do, build
//! the new segment, write it with put-if-absent, then publish it with one
//! manifest compare-and-swap. Writers never touch the manifest, so they never
//! conflict with a compactor. Two compactors that plan the same work build the
//! same object (a segment is a pure function of its WAL range); at most one
//! publish wins and the other reports [`Compaction::Superseded`].
//!
//! Compaction never deletes anything. A run that dies before publishing
//! leaves an orphan segment that the next run verifies and reuses. Segments
//! that a run supersedes are removed later, by an explicit
//! [`Namespace::gc`](crate::Namespace::gc).

use std::collections::HashSet;

use futures::TryStreamExt;
use object_store::path::Path;
use object_store::{ObjectMeta, ObjectStore, ObjectStoreExt, PutMode, PutOptions, PutPayload};

use crate::budget::ReadBudget;
use crate::cpu::{self, Cost, Lane};
use crate::document::Schema;
use crate::error::{ItzmaError, Result};
use crate::fetch::FetchPolicy;
use crate::gc;
use crate::layer;
use crate::manifest::{self, Manifest};
use crate::segment::SegmentWriter;
use crate::{segment, wal};

mod merge;

/// The default for [`CompactionOptions::max_segments`].
pub const DEFAULT_MAX_SEGMENTS: usize = 8;

/// The default for [`CompactionOptions::max_segment_bytes`]: 128 MiB,
/// comfortably under the 1 GiB default read budget so several
/// segments plus a WAL tail can coexist in one query's budget.
pub const DEFAULT_MAX_SEGMENT_BYTES: u64 = 128 * 1024 * 1024;

/// Settings for [`Namespace::compact`](crate::Namespace::compact).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactionOptions {
    /// The most live segments a namespace may end up with. When one more
    /// fold would exceed it, every segment is merged into a single base
    /// instead. Zero is treated as one.
    pub max_segments: usize,
    /// The most stored bytes one segment's `data.bin` may hold. A merge
    /// produces as many size-balanced, contiguous segments as this cap
    /// requires, instead of collapsing everything into one. Only the data object's size counts, never an
    /// index object beside it. A merge needs it at or under the handle's
    /// read budget, since it reads each group under a budget of its own;
    /// see [`validate`](Self::validate).
    pub max_segment_bytes: u64,
}

impl CompactionOptions {
    /// Check the options against the read budget of the handle that will run
    /// a merge with them. [`Namespace::compact`](crate::Namespace::compact)
    /// calls this once it knows a merge is due, before any segment is read; a
    /// fold does not need it.
    ///
    /// # Errors
    ///
    /// [`ItzmaError::InvalidConfig`] when `max_segment_bytes` is above
    /// `read_budget`: a merge group that size could never be read.
    ///
    /// Passing is necessary, not sufficient: a group can exceed the cap, so
    /// keep the budget well above the cap (the defaults are 1 GiB and
    /// 128 MiB).
    pub fn validate(&self, read_budget: u64) -> Result<()> {
        if self.max_segment_bytes > read_budget {
            return Err(ItzmaError::InvalidConfig(
                "max_segment_bytes is above the read budget, so a merge group could never be read",
            ));
        }
        Ok(())
    }
}

impl Default for CompactionOptions {
    fn default() -> Self {
        Self {
            max_segments: DEFAULT_MAX_SEGMENTS,
            max_segment_bytes: DEFAULT_MAX_SEGMENT_BYTES,
        }
    }
}

/// What a call to [`Namespace::compact`](crate::Namespace::compact) did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Compaction {
    /// The WAL tail was empty and the segment count was within the limit.
    Nothing,
    /// A segment was written and the manifest now lists it.
    Published {
        /// Ids of the new segments, oldest first. Exactly one unless the
        /// live segments and tail together exceeded `max_segment_bytes`, in
        /// which case several balanced, contiguous segments are produced
        /// (the segment size cap and merge policy).
        segments: Vec<String>,
        /// The new `wal_folded_through`.
        folded_through: u64,
        /// How many existing segments were merged into it: 0 for a fold.
        merged_segments: usize,
        /// Live documents in the new segment.
        docs: usize,
        /// Tombstones in the new segment. Always 0 for a base segment.
        tombstones: usize,
    },
    /// Another compactor changed the manifest first, so this run published
    /// nothing. The segment it wrote is either an orphan or the very object
    /// the winner published.
    Superseded,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Plan {
    Nothing,
    /// Turn the WAL tail into a new segment beside the existing ones.
    Fold,
    /// Merge into this many size-balanced, contiguous segments.
    Merge {
        target_count: usize,
    },
}

fn plan(
    segments: usize,
    tail_is_empty: bool,
    max_segments: usize,
    oversized: bool,
    total_bytes: u64,
    max_segment_bytes: u64,
) -> Plan {
    let max = max_segments.max(1);
    if tail_is_empty && segments <= max && !oversized {
        Plan::Nothing
    } else if segments + 1 > max || oversized {
        let cap = max_segment_bytes.max(1);
        let target_count = usize::try_from(total_bytes.div_ceil(cap))
            .unwrap_or(usize::MAX)
            .max(1);
        Plan::Merge { target_count }
    } else {
        Plan::Fold
    }
}

/// Partition `sizes` (one weight per unit, in order) into `target_count`
/// contiguous groups whose boundaries land as close as possible to `i *
/// total / target_count` for each cut `i`. Pure: no I/O, so two compactors
/// planning the same live segments and tail compute identical groups
/// (the "same range, same bytes" determinism). Returns the exclusive
/// end index of each group; the last entry always equals `sizes.len()`.
/// `target_count` is clamped to `1..=sizes.len().max(1)` — asking for more
/// groups than there are units is impossible, and zero units always
/// produces one (empty) group.
fn partition(sizes: &[u64], target_count: usize) -> Vec<usize> {
    let n = sizes.len();
    let target_count = target_count.clamp(1, n.max(1));
    if n == 0 || target_count <= 1 {
        return vec![n];
    }
    let total: u64 = sizes.iter().sum();
    let mut prefix = Vec::with_capacity(n);
    let mut running = 0u64;
    for &s in sizes {
        running += s;
        prefix.push(running);
    }

    let mut cuts = Vec::with_capacity(target_count);
    let mut start = 0usize;
    for i in 1..target_count {
        let remaining_groups_after = target_count - i;
        let max_idx = n - remaining_groups_after;
        let ideal = (u128::from(total) * i as u128 / target_count as u128) as u64;
        let mut best = start + 1;
        let mut best_diff = prefix[start].abs_diff(ideal);
        for idx in (start + 2)..=max_idx {
            let diff = prefix[idx - 1].abs_diff(ideal);
            if diff < best_diff {
                best_diff = diff;
                best = idx;
            }
        }
        cuts.push(best);
        start = best;
    }
    cuts.push(n);
    cuts
}

/// The `data.bin` object's size in one segment's directory, from a listing
/// (`gc::group`'s output), or 0 if the listing does not show it (a stale
/// read against a manifest a merge or a GC has since moved past — the actual
/// content fetch later either succeeds against a reloaded manifest or
/// reports the same staleness `Namespace::compact` already handles, so this
/// helper only needs to be a reasonable planning estimate, never a source of
/// truth).
fn data_size(found: &std::collections::BTreeMap<String, gc::Found>, id: &str) -> u64 {
    found
        .get(id)
        .and_then(|f| {
            f.objects
                .iter()
                .find(|(path, _)| path.filename() == Some("data.bin"))
        })
        .map(|(_, size)| *size)
        .unwrap_or(0)
}

/// One compaction run against a manifest the caller has already loaded and
/// validated. See the module docs and
/// [`Namespace::compact`](crate::Namespace::compact).
pub(crate) async fn run(
    store: &dyn ObjectStore,
    namespace: &str,
    schema: Schema,
    manifest: Manifest,
    options: &CompactionOptions,
    read_budget: u64,
    policy: &FetchPolicy,
) -> Result<Compaction> {
    let folded = manifest.wal_folded_through;
    let live = manifest.live_segments;

    let segments_prefix = Path::from(format!("{namespace}/segments"));
    let listing: Vec<ObjectMeta> = store.list(Some(&segments_prefix)).try_collect().await?;
    let found = gc::group(&segments_prefix, listing);
    let wal_objects = gc::list_wal_after(store, namespace, folded).await?;
    let tail_size_estimate: u64 = wal_objects.iter().map(|o| o.size).sum();

    // The cap is at least one byte, as `plan` takes it: a cap of zero would
    // call an empty namespace oversized and plan a merge of nothing.
    let cap = options.max_segment_bytes.max(1);
    let oversized = live.iter().any(|id| data_size(&found, id) >= cap) || tail_size_estimate >= cap;

    // The tail and each merge group get a budget of their own: the
    // budget bounds what one step holds, not how big the namespace is.
    let tail_budget = ReadBudget::new(read_budget);
    let (tail, last) = layer::fold_log(
        wal::stream_log(store, namespace, folded, &tail_budget, Lane::Maintenance).await?,
    )
    .await?;
    let newest = last.unwrap_or(folded);

    // Units in WAL order: each live segment, then the tail if it is
    // non-empty. `Namespace::compact` already ran `segment::validate_chain`
    // on this manifest before calling here (`load_manifest`), so every live
    // id is guaranteed to parse.
    struct Unit {
        lo: u64,
        hi: u64,
        size: u64,
    }
    let mut units: Vec<Unit> = live
        .iter()
        .map(|id| {
            let (lo, hi) = segment::parse_segment_id(id)
                .expect("manifest invariant already validated by load_manifest");
            Unit {
                lo,
                hi,
                size: data_size(&found, id),
            }
        })
        .collect();
    if let Some(last_seq) = last {
        units.push(Unit {
            lo: folded + 1,
            hi: last_seq,
            size: tail_size_estimate,
        });
    }
    let total_bytes: u64 = units.iter().map(|u| u.size).sum();

    let plan = plan(
        live.len(),
        last.is_none(),
        options.max_segments,
        oversized,
        total_bytes,
        options.max_segment_bytes,
    );

    match plan {
        Plan::Nothing => Ok(Compaction::Nothing),
        Plan::Fold => {
            let lo = folded + 1;
            let id = segment::segment_id(lo, newest);
            let mut layer = tail;
            if lo == 1 {
                layer.drop_tombstones();
            }
            let (docs, tombstones) = (layer.doc_count(), layer.tombstone_count());
            let (_layer, data) = segment::encode_parts_on(
                Lane::Maintenance,
                layer,
                schema.dim,
                manifest.segment_format,
                schema.attrs.clone(),
            )
            .await?;
            write_segment(store, namespace, &id, data).await?;

            let published = manifest::update_if(store, namespace, |m| {
                if m.wal_folded_through != folded || m.live_segments != live {
                    return false;
                }
                m.live_segments.push(id.clone());
                m.wal_folded_through = newest;
                true
            })
            .await?;
            Ok(match published {
                Some(_) => Compaction::Published {
                    segments: vec![id],
                    folded_through: newest,
                    merged_segments: 0,
                    docs,
                    tombstones,
                },
                None => Compaction::Superseded,
            })
        }
        Plan::Merge { target_count } => {
            options.validate(read_budget)?;
            let sizes: Vec<u64> = units.iter().map(|u| u.size).collect();
            let cuts = partition(&sizes, target_count);
            let merged_segments = live.len();

            let mut new_live = Vec::with_capacity(cuts.len());
            let mut docs = 0usize;
            let mut tombstones = 0usize;
            let mut tail_left = Some(tail);
            let mut start = 0usize;
            for &end in &cuts {
                let live_end = end.min(live.len());
                let seg_ids = &live[start.min(live.len())..live_end];
                let group_lo = units[start].lo;
                let group_hi = units[end - 1].hi;

                // Each input is read forward a window at a time and the
                // output written as it is merged. Inputs are opened
                // by index, not over `&String`s, for the `Send` reason in
                // `segment::read_all`.
                let group_budget = ReadBudget::new(read_budget);
                let mut inputs = futures::future::try_join_all((0..seg_ids.len()).map(|i| {
                    merge::Input::segment(
                        store,
                        namespace,
                        &seg_ids[i],
                        schema.dim,
                        &group_budget,
                        policy,
                        data_size(&found, &seg_ids[i]),
                        merge::MERGE_WINDOW,
                    )
                }))
                .await?;
                if end == units.len() {
                    if let Some(t) = tail_left.take() {
                        inputs.push(merge::Input::layer(t));
                    }
                }
                let writer =
                    SegmentWriter::new(schema.dim, manifest.segment_format, &schema.attrs)?;
                let writer = merge::merge_into(inputs, group_lo == 1, writer).await?;
                docs += writer.docs();
                tombstones += writer.tombstones();
                let id = segment::segment_id(group_lo, group_hi);
                let data = finish_on(writer).await?;
                write_segment(store, namespace, &id, data).await?;
                new_live.push(id);
                start = end;
            }

            let published = manifest::update_if(store, namespace, |m| {
                if m.wal_folded_through != folded || m.live_segments != live {
                    return false;
                }
                m.live_segments = new_live.clone();
                m.wal_folded_through = newest;
                // An index belongs to the segment it was built for, so a
                // merge takes the entries of the segments it retires with
                // it. Every new segment starts without one.
                let live_set: HashSet<&str> = m.live_segments.iter().map(String::as_str).collect();
                m.indexes.retain(|id, _| live_set.contains(id.as_str()));
                true
            })
            .await?;
            Ok(match published {
                Some(_) => Compaction::Published {
                    segments: new_live,
                    folded_through: newest,
                    merged_segments,
                    docs,
                    tombstones,
                },
                None => Compaction::Superseded,
            })
        }
    }
}

/// [`SegmentWriter::finish`] on the maintenance lane: it builds the Bloom
/// filter and the directory over every row.
async fn finish_on(writer: SegmentWriter) -> Result<PutPayload> {
    let cost = Cost::Bytes(writer.docs() * 64 + writer.tombstones() * 16);
    cpu::run(Lane::Maintenance, cost, move || writer.finish()).await
}

/// Bytes compared per request when an object is already at a key.
const COMPARE_PIECE: usize = 8 * 1024 * 1024;

/// Put the segment at its final key with put-if-absent. An object already at
/// that key is not trusted: it must be byte-identical to what this run built,
/// or it is damaged (or came from an encoder that disagrees), and publishing
/// it would poison the namespace. It is never overwritten.
async fn write_segment(
    store: &dyn ObjectStore,
    namespace: &str,
    id: &str,
    data: PutPayload,
) -> Result<()> {
    let path = segment::segment_path(namespace, id);
    put_or_reuse(
        store,
        &path,
        data,
        segment::missing,
        "an object already at this key differs from the segment this range produces; if the \
         manifest does not list it, delete it and rerun",
    )
    .await
}

/// Put `payload` at `path` with put-if-absent, or accept an object already
/// there only if it holds the same bytes, compared a piece at a time so the
/// existing object is never held whole beside the new one. A dead
/// orphan can be deleted by garbage collection between the refused put and
/// the compare; that is `missing`, which `Namespace::compact` treats as a
/// stale read.
async fn put_or_reuse(
    store: &dyn ObjectStore,
    path: &Path,
    payload: PutPayload,
    missing: fn(&Path) -> ItzmaError,
    differs: &str,
) -> Result<()> {
    match store
        .put_opts(path, payload.clone(), PutOptions::from(PutMode::Create))
        .await
    {
        Ok(_) => Ok(()),
        Err(object_store::Error::AlreadyExists { .. }) => {
            match same_object(store, path, &payload).await {
                Ok(true) => Ok(()),
                Ok(false) => Err(ItzmaError::CorruptSegment {
                    path: path.to_string(),
                    reason: differs.into(),
                }),
                Err(object_store::Error::NotFound { .. }) => Err(missing(path)),
                Err(e) => Err(e.into()),
            }
        }
        Err(e) => Err(e.into()),
    }
}

/// Whether the object at `path` holds exactly `payload`'s bytes.
async fn same_object(
    store: &dyn ObjectStore,
    path: &Path,
    payload: &PutPayload,
) -> object_store::Result<bool> {
    if store.head(path).await?.size != payload.content_length() as u64 {
        return Ok(false);
    }
    let mut at = 0u64;
    for chunk in payload.iter() {
        for piece in chunk.chunks(COMPARE_PIECE) {
            let end = at + piece.len() as u64;
            if store.get_range(path, at..end).await?.as_ref() != piece {
                return Ok(false);
            }
            at = end;
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oversized_forces_a_merge_even_under_max_segments() {
        // One segment, well under max_segments (8), but already over the
        // cap: must merge, not sit there.
        assert_eq!(
            plan(1, true, 8, true, 200, 100),
            Plan::Merge { target_count: 2 }
        );
        assert_eq!(plan(1, true, 8, false, 50, 100), Plan::Nothing);
    }

    #[test]
    fn an_oversized_tail_also_forces_a_merge() {
        // tail_is_empty is false here to represent a non-empty tail that is
        // itself already over the cap; `oversized` is computed by the
        // caller from the tail size estimate, `plan` itself just trusts the
        // bool.
        assert_eq!(
            plan(1, false, 8, true, 150, 100),
            Plan::Merge { target_count: 2 }
        );
    }

    #[test]
    fn the_existing_count_trigger_is_unchanged() {
        assert_eq!(
            plan(8, false, 8, false, 100, 100),
            Plan::Merge { target_count: 1 }
        ); // 8 + 1 > 8
        assert_eq!(plan(7, false, 8, false, 100, 100), Plan::Fold);
        assert_eq!(plan(3, true, 8, false, 100, 100), Plan::Nothing);
    }

    #[test]
    fn a_cap_above_the_read_budget_is_refused() {
        let options = CompactionOptions {
            max_segments: 8,
            max_segment_bytes: 1000,
        };
        assert!(options.validate(1000).is_ok());
        assert!(matches!(
            options.validate(999),
            Err(ItzmaError::InvalidConfig(_))
        ));
    }

    #[test]
    fn target_count_is_the_ceiling_of_total_over_cap() {
        assert_eq!(
            plan(9, false, 8, false, 257, 100),
            Plan::Merge { target_count: 3 }
        );
    }
}

#[cfg(test)]
mod partition_tests {
    use super::partition;

    #[test]
    fn below_the_cap_makes_one_group() {
        assert_eq!(partition(&[10, 10, 10], 1), vec![3]);
    }

    #[test]
    fn four_equal_sizes_split_evenly_in_two() {
        assert_eq!(partition(&[10, 10, 10, 10], 2), vec![2, 4]);
    }

    #[test]
    fn an_already_oversized_unit_gets_its_own_group() {
        // One huge unit, three tiny ones: the balanced cut isolates the
        // huge one rather than diluting it into a mixed group.
        assert_eq!(partition(&[100, 1, 1, 1], 2), vec![1, 4]);
    }

    #[test]
    fn target_count_is_clamped_to_the_unit_count() {
        assert_eq!(partition(&[10, 10], 5), vec![1, 2]);
    }

    #[test]
    fn zero_units_returns_one_empty_group() {
        assert_eq!(partition(&[], 3), vec![0]);
    }

    #[test]
    fn every_group_gets_at_least_one_unit() {
        for target in 1..=6 {
            let cuts = partition(&[1, 1, 1, 1, 1, 1], target);
            assert_eq!(cuts.last(), Some(&6));
            let mut prev = 0;
            for &c in &cuts {
                assert!(c > prev, "group must not be empty: {cuts:?}");
                prev = c;
            }
        }
    }
}
