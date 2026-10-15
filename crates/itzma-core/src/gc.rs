//! Garbage collection: dead segments, and optionally old WAL objects.
//!
//! GC is an explicit call, [`Namespace::gc`](crate::Namespace::gc). It loads the
//! manifest, lists `segments/` once, works out which segment directories are
//! dead, and deletes them. It never writes the manifest, so it needs only
//! load, list, put and delete, and any number of collectors can run at once.
//!
//! A segment `lo-hi` is dead when it is not live, `hi` is at or below
//! `wal_folded_through`, and it is not `1-<wal_folded_through>`. The last
//! exclusion protects an orphan that a merge over an empty tail would produce
//! and reuse. Every id a compactor makes starts at `folded + 1` (a fold) or at 1
//! (a merge), and the fold point only moves forward, so a dead id never becomes
//! live again.
//!
//! It also deletes the index objects inside live segments' directories
//! that the manifest does not name: an older generation that a rebuild replaced,
//! or an orphan of a build that never published. Like a segment, such an object
//! goes only once it and the index that replaced it are at least `min_age` old.
//! The whole directory of a dead segment goes with the segment, `sparse.bin`
//! included: it has one generation and lives and dies with its segment.
//!
//! With [`GcOptions::fence_wal`] it also lists `wal/` and replaces WAL objects
//! that a segment has absorbed with an 8-byte fence at the same key, so that
//! put-if-absent still refuses a writer with a stale hint. An object is fenced
//! only when its seq is at or below the fold point and both it and the live
//! segment that holds its seq are at least `min_age` old.
//!
//! A fencing run does not list the whole WAL. It reads a small hint object,
//! `<namespace>/gc/fenced-through`, which says how far an unbroken run of fences
//! reaches from seq 1, lists only past that, and writes the hint again once its
//! fences are in place. The hint is not a source of truth (see
//! [`fenced_through_path`]): a low or missing one costs a longer listing, and
//! nothing reads it but `gc`.

use std::collections::{BTreeMap, HashSet};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures::stream::{self, TryStreamExt};
use object_store::path::Path;
use object_store::{ObjectMeta, ObjectStore, ObjectStoreExt};

use crate::error::{ItzmaError, Result};
use crate::index;
use crate::manifest::Manifest;
use crate::{segment, wal, wal_codec};

/// The default for [`GcOptions::min_age`]: 24 hours.
pub const DEFAULT_MIN_AGE: Duration = Duration::from_secs(24 * 60 * 60);

/// How many deletes, or fences, run at once.
const CONCURRENCY: usize = 16;

/// The longest an object can be and still count as a hint. A seq is at most 20
/// digits, so a longer one is not a hint and is read as 0 without being read.
const HINT_READ_BYTES: u64 = 32;

/// Object key of the fence hint: `<namespace>/gc/fenced-through`. It holds one
/// seq as decimal text: every WAL object at or below it was a fence when a
/// fencing run wrote it. Only [`Namespace::gc`](crate::Namespace::gc) reads or
/// writes it, and only with `fence_wal`.
///
/// It is a hint and not a fact. A fence never turns back into an object, so the
/// value cannot go stale by itself. A missing, unreadable or too low value makes
/// the next run list more than it needs to, and two collectors may write a lower
/// value over a higher one, which costs the same. A value that is too high (only
/// a hand-written one can be) leaves objects below it unfenced, which wastes
/// space and loses nothing. Nothing else uses it: `recover` works out
/// the same fact from the WAL listing.
///
/// Does not validate `namespace`; that is [`Namespace`](crate::Namespace)'s job.
pub fn fenced_through_path(namespace: &str) -> Path {
    Path::from(format!("{namespace}/gc/fenced-through"))
}

/// Settings for [`Namespace::gc`](crate::Namespace::gc).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GcOptions {
    /// How old a segment must be before it is deleted, and how old the live
    /// segment that replaced it must be. It gives a slow reader time to finish
    /// and an operator time to notice a mistake. Readers do not depend on it:
    /// one that finds a segment gone reloads the manifest and reads again.
    pub min_age: Duration,
    /// Work out what would be deleted or fenced, report it, and change nothing.
    pub dry_run: bool,
    /// Also replace old WAL objects with fences (phase B). Off by
    /// default, because a fenced range can no longer be rebuilt from the WAL:
    /// the segments become the only copy of what it held.
    ///
    /// The namespace must be at WAL format level 4
    /// ([`FENCE_FORMAT_VERSION`](crate::wal::FENCE_FORMAT_VERSION)), or the call
    /// fails with [`ItzmaError::WalFormatTooOld`] and changes nothing.
    pub fence_wal: bool,
}

impl Default for GcOptions {
    fn default() -> Self {
        Self {
            min_age: DEFAULT_MIN_AGE,
            dry_run: false,
            fence_wal: false,
        }
    }
}

/// What a call to [`Namespace::gc`](crate::Namespace::gc) did.
///
/// The counts are what the run set out to delete and did not fail on. Deleting
/// an object that is already gone is not an error and is still counted, so two
/// collectors running at once can both count the same segment. With
/// [`GcOptions::dry_run`] they are what a real run would have counted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GcReport {
    /// Dead segments removed, each with every object under its directory.
    pub segments_deleted: usize,
    /// Total size of the objects removed.
    pub bytes_freed: u64,
    /// Dead segments left for a later run because they, or the live segment
    /// that replaced them, are younger than `min_age`.
    pub segments_too_young: usize,
    /// WAL objects replaced by a fence. Zero unless [`GcOptions::fence_wal`] is
    /// set. An object that is already a fence is not counted.
    pub wal_objects_fenced: usize,
    /// What the fenced WAL objects held, less the 8 bytes each fence keeps.
    pub wal_bytes_freed: u64,
    /// WAL objects left for a later run because they, or the live segment that
    /// holds their seq, are younger than `min_age`.
    pub wal_too_young: usize,
    /// Index objects removed from segments that are still live: the ones the
    /// manifest does not name, an older generation that a rebuild replaced or
    /// an orphan of a build that never published. The index objects
    /// inside a dead segment's directory go with the segment and are counted
    /// there. Their size is part of `bytes_freed`.
    pub indexes_deleted: usize,
    /// Index objects left for a later run because they, or the index that
    /// replaced them, are younger than `min_age`.
    pub indexes_too_young: usize,
}

/// One `segments/<id>/` directory, as the listing shows it.
pub(crate) struct Found {
    pub(crate) lo: u64,
    pub(crate) hi: u64,
    /// Write time of the newest object in the directory, in unix milliseconds.
    pub(crate) modified_ms: i64,
    /// Every object in the directory with its size.
    pub(crate) objects: Vec<(Path, u64)>,
    /// The index objects among them (`index-<generation>.bin`).
    pub(crate) indexes: Vec<FoundIndex>,
}

/// One index object in a segment directory.
pub(crate) struct FoundIndex {
    pub(crate) generation: u32,
    pub(crate) path: Path,
    pub(crate) size: u64,
    pub(crate) modified_ms: i64,
}

/// What to do about the segments found.
struct Plan {
    delete: Vec<Found>,
    too_young: usize,
}

/// The age rule: how old something must be, against a clock passed in so the
/// rule can be tested.
#[derive(Clone, Copy)]
struct Age {
    now_ms: i64,
    min_age_ms: i64,
}

impl Age {
    fn new(now_ms: i64, min_age: Duration) -> Self {
        Self {
            now_ms,
            min_age_ms: i64::try_from(min_age.as_millis()).unwrap_or(i64::MAX),
        }
    }

    /// A write time in the future (clock skew) counts as age zero.
    fn enough(self, modified_ms: i64) -> bool {
        self.now_ms.saturating_sub(modified_ms).max(0) >= self.min_age_ms
    }
}

/// The live segments, oldest first, with each one's last seq and write time.
/// The write time is `None` when the listing does not show the segment, which
/// counts as too young. Whatever holds a seq is the live segment that replaced
/// it, so its write time stands in for the moment the seq stopped being needed.
struct Covering(Vec<(u64, Option<i64>)>);

impl Covering {
    fn new(manifest: &Manifest, found: &BTreeMap<String, Found>) -> Self {
        Self(
            manifest
                .live_segments
                .iter()
                .filter_map(|id| {
                    let (_, hi) = segment::parse_segment_id(id)?;
                    Some((hi, found.get(id).map(|f| f.modified_ms)))
                })
                .collect(),
        )
    }

    /// When the live segment that holds `seq` was written.
    fn modified_ms(&self, seq: u64) -> Option<i64> {
        let at = self.0.partition_point(|(hi, _)| *hi < seq);
        self.0.get(at).and_then(|(_, modified_ms)| *modified_ms)
    }
}

/// Group a listing of `segments/` by directory. Objects that sit directly
/// under the prefix, and directories whose name is not a segment id, are not
/// ours and are skipped.
pub(crate) fn group(prefix: &Path, listing: Vec<ObjectMeta>) -> BTreeMap<String, Found> {
    let mut found: BTreeMap<String, Found> = BTreeMap::new();
    for meta in listing {
        let id = {
            let Some(mut parts) = meta.location.prefix_match(prefix) else {
                continue;
            };
            let Some(dir) = parts.next() else { continue };
            if parts.next().is_none() {
                continue;
            }
            dir.as_ref().to_string()
        };
        let Some((lo, hi)) = segment::parse_segment_id(&id) else {
            continue;
        };
        let entry = found.entry(id).or_insert_with(|| Found {
            lo,
            hi,
            modified_ms: i64::MIN,
            objects: Vec::new(),
            indexes: Vec::new(),
        });
        let modified_ms = meta.last_modified.timestamp_millis();
        entry.modified_ms = entry.modified_ms.max(modified_ms);
        if let Some(generation) = meta.location.filename().and_then(index::generation_of) {
            entry.indexes.push(FoundIndex {
                generation,
                path: meta.location.clone(),
                size: meta.size,
                modified_ms,
            });
        }
        entry.objects.push((meta.location, meta.size));
    }
    found
}

/// A segment is dead when it is not live, its range ends at or below the fold
/// point, and it is not `1-<folded>`, which an orphan of a merge over an empty
/// tail would be and a later run would reuse. One rule for `gc` and for
/// [`inspect`](crate::inspect), so what a report calls dead is what a run takes.
pub(crate) fn is_dead(folded: u64, live: &HashSet<&str>, id: &str, segment: &Found) -> bool {
    !live.contains(id) && segment.hi <= folded && !(segment.lo == 1 && segment.hi == folded)
}

/// Decide which segments to delete. Pure: no storage.
fn plan(
    manifest: &Manifest,
    found: BTreeMap<String, Found>,
    covering: &Covering,
    age: Age,
) -> Plan {
    let folded = manifest.wal_folded_through;
    let live_ids: HashSet<&str> = manifest.live_segments.iter().map(String::as_str).collect();

    let mut delete = Vec::new();
    let mut too_young = 0;
    for (id, segment) in found {
        if !is_dead(folded, &live_ids, &id, &segment) {
            continue;
        }
        if age.enough(segment.modified_ms)
            && covering
                .modified_ms(segment.hi)
                .is_some_and(|m| age.enough(m))
        {
            delete.push(segment);
        } else {
            too_young += 1;
        }
    }
    Plan { delete, too_young }
}

/// What to do about the index objects of live segments.
#[derive(Default)]
struct IndexPlan {
    delete: Vec<(Path, u64)>,
    too_young: usize,
}

/// Decide which index objects of live segments to delete. Pure: no storage.
///
/// An index object is dead when the manifest does not name it: an older
/// generation, a newer one that was never published, or one for a segment the
/// manifest lists no index for. Like a segment (see [`plan`]) it is deleted only
/// when it and the index that replaced it are both `min_age` old. An object with
/// no replacement, an orphan, needs only its own age, and one whose replacement
/// the listing does not show counts as too young. The directory of a segment
/// that is not live is not looked at here: it is [`plan`]'s, whole.
fn plan_indexes(manifest: &Manifest, found: &BTreeMap<String, Found>, age: Age) -> IndexPlan {
    plan_generations(
        manifest,
        found,
        age,
        |id| manifest.indexes.get(id).map(|entry| entry.generation),
        |dir| &dir.indexes,
    )
}

/// The rule of [`plan_indexes`], for whichever kind of generational object
/// `named` and `objects` pick out.
fn plan_generations(
    manifest: &Manifest,
    found: &BTreeMap<String, Found>,
    age: Age,
    named: impl Fn(&str) -> Option<u32>,
    objects: impl Fn(&Found) -> &[FoundIndex],
) -> IndexPlan {
    let mut plan = IndexPlan::default();
    for id in &manifest.live_segments {
        let Some(dir) = found.get(id) else { continue };
        let named = named(id);
        let present = objects(dir);
        let replacement_ready = match named {
            None => true,
            Some(generation) => present
                .iter()
                .find(|index| index.generation == generation)
                .is_some_and(|index| age.enough(index.modified_ms)),
        };
        for index in present {
            if Some(index.generation) == named {
                continue;
            }
            if replacement_ready && age.enough(index.modified_ms) {
                plan.delete.push((index.path.clone(), index.size));
            } else {
                plan.too_young += 1;
            }
        }
    }
    plan
}

/// One `wal/<seq>.json` object, as the listing shows it.
pub(crate) struct WalObject {
    pub(crate) seq: u64,
    pub(crate) path: Path,
    pub(crate) size: u64,
    pub(crate) modified_ms: i64,
}

/// What to do about the WAL objects found.
#[derive(Default)]
struct WalPlan {
    fence: Vec<WalObject>,
    too_young: usize,
}

/// Decide which WAL objects to fence. Pure: no storage. Objects past the fold
/// point are not ours to touch, and an object the size of a fence already is
/// one (every real object is longer), so neither is counted.
fn plan_wal(
    manifest: &Manifest,
    covering: &Covering,
    objects: Vec<WalObject>,
    age: Age,
) -> WalPlan {
    let mut plan = WalPlan::default();
    for object in objects {
        if object.seq > manifest.wal_folded_through || object.size == wal_codec::FENCE_LEN as u64 {
            continue;
        }
        if age.enough(object.modified_ms)
            && covering
                .modified_ms(object.seq)
                .is_some_and(|m| age.enough(m))
        {
            plan.fence.push(object);
        } else {
            plan.too_young += 1;
        }
    }
    plan
}

pub(crate) fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
}

/// One collection run against a manifest the caller has already loaded and
/// validated. See the module docs and
/// [`Namespace::gc`](crate::Namespace::gc).
pub(crate) async fn run(
    store: &dyn ObjectStore,
    namespace: &str,
    manifest: &Manifest,
    options: &GcOptions,
) -> Result<GcReport> {
    if options.fence_wal && manifest.wal_format < wal::FENCE_FORMAT_VERSION {
        return Err(ItzmaError::WalFormatTooOld {
            have: manifest.wal_format,
            need: wal::FENCE_FORMAT_VERSION,
        });
    }
    // The listings come after the manifest on purpose: a segment written in
    // between has `hi` past the manifest's fold point, so it is not dead, and
    // a WAL object written in between has a seq past it too.
    let prefix = Path::from(format!("{namespace}/segments"));
    let listing: Vec<ObjectMeta> = store.list(Some(&prefix)).try_collect().await?;
    let found = group(&prefix, listing);
    let covering = Covering::new(manifest, &found);
    let age = Age::new(now_millis(), options.min_age);
    let indexes = plan_indexes(manifest, &found, age);
    let Plan { delete, too_young } = plan(manifest, found, &covering, age);
    let hint = if options.fence_wal {
        read_hint(store, namespace).await?
    } else {
        0
    };
    let (wal_plan, reach) = if options.fence_wal {
        let objects = list_wal_after(store, namespace, hint).await?;
        let listed: Vec<(u64, u64)> = objects.iter().map(|o| (o.seq, o.size)).collect();
        let plan = plan_wal(manifest, &covering, objects, age);
        let reach = fenced_reach(hint, &listed, &plan.fence);
        (plan, reach)
    } else {
        (WalPlan::default(), hint)
    };

    let report = GcReport {
        segments_deleted: delete.len(),
        bytes_freed: delete
            .iter()
            .flat_map(|found| &found.objects)
            .map(|(_, size)| size)
            .chain(indexes.delete.iter().map(|(_, size)| size))
            .sum(),
        segments_too_young: too_young,
        wal_objects_fenced: wal_plan.fence.len(),
        wal_bytes_freed: wal_plan
            .fence
            .iter()
            .map(|object| object.size.saturating_sub(wal_codec::FENCE_LEN as u64))
            .sum(),
        wal_too_young: wal_plan.too_young,
        indexes_deleted: indexes.delete.len(),
        indexes_too_young: indexes.too_young,
    };
    if !options.dry_run {
        let paths: Vec<Path> = delete
            .into_iter()
            .flat_map(|found| found.objects)
            .chain(indexes.delete)
            .map(|(path, _)| path)
            .collect();
        delete_all(store, paths).await?;
        fence_all(store, wal_plan.fence).await?;
        if reach > hint {
            write_hint(store, namespace, reach).await?;
        }
    }
    Ok(report)
}

/// List `wal/` with sizes and write times. Keys that are not a seq are not
/// ours and are skipped.
pub(crate) async fn list_wal(store: &dyn ObjectStore, namespace: &str) -> Result<Vec<WalObject>> {
    list_wal_after(store, namespace, 0).await
}

/// [`list_wal`] for the objects with a seq above `after`. Zero lists them all.
pub(crate) async fn list_wal_after(
    store: &dyn ObjectStore,
    namespace: &str,
    after: u64,
) -> Result<Vec<WalObject>> {
    let prefix = wal::wal_prefix(namespace);
    let listing: Vec<ObjectMeta> = if after == 0 {
        store.list(Some(&prefix)).try_collect().await?
    } else {
        store
            .list_with_offset(Some(&prefix), &wal::wal_path(namespace, after))
            .try_collect()
            .await?
    };
    Ok(listing
        .into_iter()
        .filter_map(|meta| {
            Some(WalObject {
                seq: wal::parse_seq(&meta.location)?,
                modified_ms: meta.last_modified.timestamp_millis(),
                size: meta.size,
                path: meta.location,
            })
        })
        .collect())
}

/// The seq in the fence hint, or 0 if there is no hint or it does not parse. A
/// bad hint is never an error, because starting from the top is always right.
async fn read_hint(store: &dyn ObjectStore, namespace: &str) -> Result<u64> {
    match store.get(&fenced_through_path(namespace)).await {
        // The size comes with the response, so an object that cannot be a hint
        // is not downloaded.
        Ok(got) if got.meta.size > HINT_READ_BYTES => Ok(0),
        Ok(got) => Ok(parse_hint(&got.bytes().await?)),
        Err(object_store::Error::NotFound { .. }) => Ok(0),
        Err(e) => Err(e.into()),
    }
}

fn parse_hint(bytes: &[u8]) -> u64 {
    std::str::from_utf8(bytes)
        .ok()
        .and_then(|text| text.trim().parse().ok())
        .unwrap_or(0)
}

/// A plain overwrite. Two collectors racing may leave the lower value, which
/// only makes the next listing longer.
async fn write_hint(store: &dyn ObjectStore, namespace: &str, seq: u64) -> Result<()> {
    store
        .put(&fenced_through_path(namespace), seq.to_string().into())
        .await?;
    Ok(())
}

/// How far the unbroken run of fences reaches from `from` once the objects in
/// `fenced` have been written. `listed` is every object the run saw, with its
/// size, and nothing past a gap, a young object or an unfolded one counts.
fn fenced_reach(from: u64, listed: &[(u64, u64)], fenced: &[WalObject]) -> u64 {
    let planned: HashSet<u64> = fenced.iter().map(|object| object.seq).collect();
    let mut sorted = listed.to_vec();
    sorted.sort_unstable();
    let mut reach = from;
    for (seq, size) in sorted {
        let fence = size == wal_codec::FENCE_LEN as u64 || planned.contains(&seq);
        if seq != reach + 1 || !fence {
            break;
        }
        reach = seq;
    }
    reach
}

/// Overwrite each object with a fence, 16 at a time. The first failure ends the
/// run, and a rerun finishes the job: an object already fenced is skipped, and
/// two collectors that fence the same object write the same bytes.
async fn fence_all(store: &dyn ObjectStore, objects: Vec<WalObject>) -> Result<()> {
    stream::iter(objects.into_iter().map(Ok::<_, ItzmaError>))
        .try_for_each_concurrent(CONCURRENCY, |object| async move {
            store.put(&object.path, wal_codec::fence().into()).await?;
            Ok(())
        })
        .await
}

/// Delete every path, 16 at a time. The first failure ends the run, and a rerun
/// finishes the job, since deleting is idempotent.
async fn delete_all(store: &dyn ObjectStore, paths: Vec<Path>) -> Result<()> {
    stream::iter(paths.into_iter().map(Ok::<_, ItzmaError>))
        .try_for_each_concurrent(
            CONCURRENCY,
            |path| async move { delete_one(store, &path).await },
        )
        .await
}

/// Delete one object. One already gone, say from another collector, is fine.
async fn delete_one(store: &dyn ObjectStore, path: &Path) -> Result<()> {
    match store.delete(path).await {
        Ok(()) | Err(object_store::Error::NotFound { .. }) => Ok(()),
        Err(e) => Err(e.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{Metric, Schema};

    fn to_fence(seq: u64) -> WalObject {
        WalObject {
            seq,
            path: wal::wal_path("docs", seq),
            size: 100,
            modified_ms: 0,
        }
    }

    const FENCE_SIZE: u64 = wal_codec::FENCE_LEN as u64;

    #[test]
    fn a_hint_is_decimal_text_and_anything_else_is_zero() {
        assert_eq!(parse_hint(b"1205"), 1205);
        assert_eq!(parse_hint(b" 7\n"), 7);
        for bad in [
            &b""[..],
            b"x",
            b"-1",
            b"1.5",
            b"\xff",
            b"99999999999999999999999",
        ] {
            assert_eq!(parse_hint(bad), 0, "{bad:?}");
        }
    }

    #[test]
    fn the_reach_follows_an_unbroken_run_of_fences() {
        let listed = [(1, FENCE_SIZE), (2, FENCE_SIZE), (3, 100), (4, 100)];
        // 3 is being fenced now, 4 is not.
        assert_eq!(fenced_reach(0, &listed, &[to_fence(3)]), 3);
        // Nothing is being fenced, so the run stops at the first real object.
        assert_eq!(fenced_reach(0, &listed, &[]), 2);
    }

    #[test]
    fn the_reach_stops_at_a_gap_and_starts_from_the_hint() {
        let listed = [(6, FENCE_SIZE), (7, FENCE_SIZE), (9, FENCE_SIZE)];
        assert_eq!(fenced_reach(5, &listed, &[]), 7);
        // The listing is in any order.
        let shuffled = [(9, FENCE_SIZE), (7, FENCE_SIZE), (6, FENCE_SIZE)];
        assert_eq!(fenced_reach(5, &shuffled, &[]), 7);
        // Objects the listing did not show are not assumed to be fences.
        assert_eq!(fenced_reach(5, &[], &[]), 5);
        assert_eq!(fenced_reach(5, &[(8, FENCE_SIZE)], &[]), 5);
    }

    #[test]
    fn the_reach_never_falls_below_the_hint() {
        assert_eq!(fenced_reach(10, &[(11, 100)], &[]), 10);
    }

    const HOUR_MS: i64 = 60 * 60 * 1000;
    const NOW: i64 = 1_000 * HOUR_MS;

    fn found(lo: u64, hi: u64, age_hours: i64) -> (String, Found) {
        let id = segment::segment_id(lo, hi);
        let path = segment::segment_path("docs", &id);
        (
            id,
            Found {
                lo,
                hi,
                modified_ms: NOW - age_hours * HOUR_MS,
                objects: vec![(path, 100)],
                indexes: Vec::new(),
            },
        )
    }

    fn manifest(live: &[(u64, u64)], folded: u64) -> Manifest {
        let mut m = Manifest::new(Schema::new(1, Metric::L2));
        m.live_segments = live
            .iter()
            .map(|(lo, hi)| segment::segment_id(*lo, *hi))
            .collect();
        m.wal_folded_through = folded;
        m
    }

    /// Ids the plan would delete, sorted.
    fn deleted(plan: &Plan) -> Vec<String> {
        let mut ids: Vec<String> = plan
            .delete
            .iter()
            .map(|f| segment::segment_id(f.lo, f.hi))
            .collect();
        ids.sort();
        ids
    }

    fn ids(pairs: &[(u64, u64)]) -> Vec<String> {
        pairs
            .iter()
            .map(|(lo, hi)| segment::segment_id(*lo, *hi))
            .collect()
    }

    fn plan_for(m: &Manifest, segments: &[(u64, u64, i64)], min_age_hours: u64) -> Plan {
        let listing: BTreeMap<String, Found> = segments
            .iter()
            .map(|(lo, hi, age)| found(*lo, *hi, *age))
            .collect();
        let covering = Covering::new(m, &listing);
        plan(
            m,
            listing,
            &covering,
            Age::new(NOW, Duration::from_secs(min_age_hours * 3600)),
        )
    }

    fn wal_object(seq: u64, size: u64, age_hours: i64) -> WalObject {
        WalObject {
            seq,
            path: wal::wal_path("docs", seq),
            size,
            modified_ms: NOW - age_hours * HOUR_MS,
        }
    }

    /// The seqs the plan would fence.
    fn fenced(plan: &WalPlan) -> Vec<u64> {
        plan.fence.iter().map(|o| o.seq).collect()
    }

    fn wal_plan_for(
        m: &Manifest,
        segments: &[(u64, u64, i64)],
        objects: Vec<WalObject>,
        min_age_hours: u64,
    ) -> WalPlan {
        let listing: BTreeMap<String, Found> = segments
            .iter()
            .map(|(lo, hi, age)| found(*lo, *hi, *age))
            .collect();
        let covering = Covering::new(m, &listing);
        plan_wal(
            m,
            &covering,
            objects,
            Age::new(NOW, Duration::from_secs(min_age_hours * 3600)),
        )
    }

    #[test]
    fn a_live_segment_is_never_dead() {
        let m = manifest(&[(1, 3)], 3);
        let plan = plan_for(&m, &[(1, 3, 500)], 0);
        assert!(plan.delete.is_empty());
        assert_eq!(plan.too_young, 0);
    }

    #[test]
    fn a_superseded_segment_is_dead() {
        let m = manifest(&[(1, 3)], 3);
        let plan = plan_for(
            &m,
            &[(1, 1, 500), (2, 2, 500), (3, 3, 500), (1, 3, 500)],
            24,
        );
        assert_eq!(deleted(&plan), ids(&[(1, 1), (2, 2), (3, 3)]));
    }

    #[test]
    fn a_fold_that_a_merge_replaced_is_dead() {
        // Live: a base 1-4 and a delta 5-6. The delta 5-5 that 5-6 replaced is dead.
        let m = manifest(&[(1, 4), (5, 6)], 6);
        let plan = plan_for(&m, &[(1, 4, 500), (5, 5, 500), (5, 6, 500)], 24);
        assert_eq!(deleted(&plan), ids(&[(5, 5)]));
    }

    #[test]
    fn an_orphan_past_the_fold_point_is_kept() {
        let m = manifest(&[(1, 3)], 3);
        let plan = plan_for(&m, &[(1, 3, 500), (4, 5, 500), (1, 5, 500)], 0);
        assert!(plan.delete.is_empty());
        assert_eq!(plan.too_young, 0, "kept is not the same as too young");
    }

    #[test]
    fn an_orphan_that_a_merge_over_an_empty_tail_would_reuse_is_kept() {
        // Live 1-1 and 2-2, folded through 2. A merge would make 1-2.
        let m = manifest(&[(1, 1), (2, 2)], 2);
        let plan = plan_for(&m, &[(1, 1, 500), (2, 2, 500), (1, 2, 500)], 0);
        assert!(plan.delete.is_empty());
    }

    #[test]
    fn an_orphan_at_or_below_the_fold_point_is_dead() {
        let m = manifest(&[(1, 3)], 3);
        let plan = plan_for(&m, &[(1, 3, 500), (2, 3, 500), (2, 2, 500)], 24);
        assert_eq!(deleted(&plan), ids(&[(2, 2), (2, 3)]));
    }

    #[test]
    fn a_dead_segment_needs_its_own_age_and_its_coverers() {
        let m = manifest(&[(1, 3)], 3);
        // 1-1 is old but the merge that replaced it is an hour old.
        let plan = plan_for(&m, &[(1, 1, 500), (1, 3, 1)], 24);
        assert!(plan.delete.is_empty());
        assert_eq!(plan.too_young, 1);

        // The merge is old but the dead segment was written an hour ago.
        let plan = plan_for(&m, &[(2, 2, 1), (1, 3, 500)], 24);
        assert!(plan.delete.is_empty());
        assert_eq!(plan.too_young, 1);

        // Both old enough.
        let plan = plan_for(&m, &[(2, 2, 30), (1, 3, 25)], 24);
        assert_eq!(deleted(&plan), ids(&[(2, 2)]));
    }

    #[test]
    fn the_covering_segment_is_the_one_that_holds_the_dead_segments_end() {
        // Live 1-4 (old) and 5-6 (young). 3-4 is covered by 1-4, not by 5-6.
        let m = manifest(&[(1, 4), (5, 6)], 6);
        let plan = plan_for(&m, &[(1, 4, 500), (5, 6, 1), (3, 4, 500), (5, 5, 500)], 24);
        assert_eq!(deleted(&plan), ids(&[(3, 4)]));
        assert_eq!(plan.too_young, 1, "5-5 waits for 5-6");
    }

    #[test]
    fn a_zero_min_age_deletes_regardless_of_age() {
        let m = manifest(&[(1, 3)], 3);
        let plan = plan_for(&m, &[(2, 2, 0), (1, 3, 0)], 0);
        assert_eq!(deleted(&plan), ids(&[(2, 2)]));
    }

    #[test]
    fn a_write_time_in_the_future_counts_as_brand_new() {
        // The store's clock runs ahead of ours.
        let m = manifest(&[(1, 3)], 3);
        let plan = plan_for(&m, &[(2, 2, -5), (1, 3, 500)], 24);
        assert!(plan.delete.is_empty());
        assert_eq!(plan.too_young, 1);
        let plan = plan_for(&m, &[(2, 2, -5), (1, 3, -5)], 0);
        assert_eq!(deleted(&plan), ids(&[(2, 2)]));
    }

    #[test]
    fn a_live_segment_the_listing_does_not_show_counts_as_too_young() {
        let m = manifest(&[(1, 3)], 3);
        let plan = plan_for(&m, &[(2, 2, 500)], 24);
        assert!(plan.delete.is_empty());
        assert_eq!(plan.too_young, 1);
    }

    #[test]
    fn nothing_is_dead_before_the_first_fold() {
        let m = manifest(&[], 0);
        let plan = plan_for(&m, &[(1, 2, 500), (3, 3, 500)], 0);
        assert!(plan.delete.is_empty());
    }

    #[test]
    fn folded_and_old_wal_objects_are_fenced() {
        let m = manifest(&[(1, 3)], 3);
        let objects = (1..=3).map(|seq| wal_object(seq, 500, 500)).collect();
        let plan = wal_plan_for(&m, &[(1, 3, 500)], objects, 24);
        assert_eq!(fenced(&plan), [1, 2, 3]);
        assert_eq!(plan.too_young, 0);
    }

    #[test]
    fn a_wal_object_past_the_fold_point_is_never_fenced() {
        let m = manifest(&[(1, 3)], 3);
        let objects = (1..=5).map(|seq| wal_object(seq, 500, 500)).collect();
        let plan = wal_plan_for(&m, &[(1, 3, 500)], objects, 0);
        assert_eq!(fenced(&plan), [1, 2, 3]);
        assert_eq!(plan.too_young, 0, "left alone is not the same as too young");
    }

    #[test]
    fn a_fence_is_skipped_and_not_counted() {
        let m = manifest(&[(1, 3)], 3);
        let objects = vec![
            wal_object(1, wal_codec::FENCE_LEN as u64, 500),
            wal_object(2, wal_codec::FENCE_LEN as u64, 0),
            wal_object(3, 500, 500),
        ];
        let plan = wal_plan_for(&m, &[(1, 3, 500)], objects, 24);
        assert_eq!(fenced(&plan), [3]);
        assert_eq!(plan.too_young, 0);
    }

    #[test]
    fn a_wal_object_needs_its_own_age_and_its_covering_segments() {
        let m = manifest(&[(1, 3)], 3);
        // The object is an hour old, the segment that holds it is old.
        let plan = wal_plan_for(&m, &[(1, 3, 500)], vec![wal_object(2, 500, 1)], 24);
        assert!(plan.fence.is_empty());
        assert_eq!(plan.too_young, 1);

        // The object is old, the segment that holds it is an hour old.
        let plan = wal_plan_for(&m, &[(1, 3, 1)], vec![wal_object(2, 500, 500)], 24);
        assert!(plan.fence.is_empty());
        assert_eq!(plan.too_young, 1);

        let plan = wal_plan_for(&m, &[(1, 3, 25)], vec![wal_object(2, 500, 30)], 24);
        assert_eq!(fenced(&plan), [2]);
    }

    #[test]
    fn the_segment_that_holds_the_seq_is_the_one_that_counts_for_a_wal_object() {
        // Live 1-4 (old) and 5-6 (young). Seq 3 is held by 1-4, seq 5 by 5-6.
        let m = manifest(&[(1, 4), (5, 6)], 6);
        let objects = vec![wal_object(3, 500, 500), wal_object(5, 500, 500)];
        let plan = wal_plan_for(&m, &[(1, 4, 500), (5, 6, 1)], objects, 24);
        assert_eq!(fenced(&plan), [3]);
        assert_eq!(plan.too_young, 1);
    }

    #[test]
    fn a_covering_segment_the_listing_does_not_show_counts_as_too_young_for_wal() {
        let m = manifest(&[(1, 3)], 3);
        let plan = wal_plan_for(&m, &[], vec![wal_object(2, 500, 500)], 24);
        assert!(plan.fence.is_empty());
        assert_eq!(plan.too_young, 1);
    }

    #[test]
    fn a_zero_min_age_fences_regardless_of_age() {
        let m = manifest(&[(1, 3)], 3);
        let plan = wal_plan_for(&m, &[(1, 3, 0)], vec![wal_object(2, 500, 0)], 0);
        assert_eq!(fenced(&plan), [2]);
    }

    #[test]
    fn a_wal_write_time_in_the_future_counts_as_brand_new() {
        let m = manifest(&[(1, 3)], 3);
        let plan = wal_plan_for(&m, &[(1, 3, 500)], vec![wal_object(2, 500, -5)], 24);
        assert!(plan.fence.is_empty());
        assert_eq!(plan.too_young, 1);
    }

    #[test]
    fn nothing_is_fenced_before_the_first_fold() {
        let m = manifest(&[], 0);
        let objects = (1..=3).map(|seq| wal_object(seq, 500, 500)).collect();
        let plan = wal_plan_for(&m, &[], objects, 0);
        assert!(plan.fence.is_empty());
        assert_eq!(plan.too_young, 0);
    }

    /// A live segment `1-1` with the given index objects `(generation, age in
    /// hours)`, and a manifest that names `named` for it, if any.
    fn plan_indexes_for(
        objects: &[(u32, i64)],
        named: Option<u32>,
        min_age_hours: u64,
    ) -> IndexPlan {
        let mut m = manifest(&[(1, 1)], 1);
        let (id, mut dir) = found(1, 1, 100);
        dir.indexes = objects
            .iter()
            .map(|(generation, age)| FoundIndex {
                generation: *generation,
                path: index::index_path("docs", &id, *generation),
                size: 10,
                modified_ms: NOW - age * HOUR_MS,
            })
            .collect();
        if let Some(generation) = named {
            m.indexes.insert(
                id.clone(),
                crate::manifest::IndexEntry {
                    generation,
                    kind: "sq8".into(),
                },
            );
        }
        let listing = BTreeMap::from([(id, dir)]);
        plan_indexes(
            &m,
            &listing,
            Age::new(NOW, Duration::from_secs(min_age_hours * 3600)),
        )
    }

    fn deleted_generations(plan: &IndexPlan) -> Vec<String> {
        plan.delete
            .iter()
            .map(|(path, _)| path.filename().unwrap_or_default().to_string())
            .collect()
    }

    #[test]
    fn an_index_the_manifest_names_is_never_deleted() {
        let plan = plan_indexes_for(&[(1, 1_000)], Some(1), 0);
        assert!(plan.delete.is_empty());
        assert_eq!(plan.too_young, 0);
    }

    #[test]
    fn a_replaced_generation_needs_its_own_age_and_its_replacements() {
        // 1 is old, 2 replaced it. Both old enough: 1 goes.
        let plan = plan_indexes_for(&[(1, 30), (2, 25)], Some(2), 24);
        assert_eq!(deleted_generations(&plan), ["index-1.bin"]);
        assert_eq!(plan.too_young, 0);
        // The replacement is young: the old one waits.
        let plan = plan_indexes_for(&[(1, 30), (2, 1)], Some(2), 24);
        assert!(plan.delete.is_empty());
        assert_eq!(plan.too_young, 1);
        // The old one is young itself.
        let plan = plan_indexes_for(&[(1, 1), (2, 30)], Some(2), 24);
        assert!(plan.delete.is_empty());
        assert_eq!(plan.too_young, 1);
    }

    #[test]
    fn an_orphan_waits_only_for_its_own_age() {
        // Nothing named: the segment has no index, whatever is there is stray.
        let plan = plan_indexes_for(&[(1, 30)], None, 24);
        assert_eq!(deleted_generations(&plan), ["index-1.bin"]);
        let plan = plan_indexes_for(&[(1, 1)], None, 24);
        assert!(plan.delete.is_empty());
        assert_eq!(plan.too_young, 1);
    }

    #[test]
    fn a_newer_generation_that_was_never_published_is_an_orphan() {
        // Generation 2 is named. 3 is a rebuild that stopped before it published.
        let plan = plan_indexes_for(&[(2, 30), (3, 30)], Some(2), 24);
        assert_eq!(deleted_generations(&plan), ["index-3.bin"]);
        let plan = plan_indexes_for(&[(2, 30), (3, 1)], Some(2), 24);
        assert!(plan.delete.is_empty());
        assert_eq!(plan.too_young, 1);
    }

    #[test]
    fn a_replacement_the_listing_does_not_show_counts_as_too_young() {
        let plan = plan_indexes_for(&[(1, 1_000)], Some(2), 24);
        assert!(plan.delete.is_empty());
        assert_eq!(plan.too_young, 1);
    }

    #[test]
    fn a_zero_min_age_deletes_every_unnamed_index() {
        let plan = plan_indexes_for(&[(1, 0), (2, 0), (3, 0)], Some(2), 0);
        assert_eq!(deleted_generations(&plan), ["index-1.bin", "index-3.bin"]);
    }

    #[test]
    fn a_segment_that_is_not_live_is_not_this_plans() {
        // Its directory is the segment plan's, deleted whole.
        let m = manifest(&[(1, 2)], 2);
        let (id, mut dir) = found(1, 1, 100);
        dir.indexes = vec![FoundIndex {
            generation: 1,
            path: index::index_path("docs", &id, 1),
            size: 10,
            modified_ms: NOW - 1_000 * HOUR_MS,
        }];
        let listing = BTreeMap::from([(id, dir)]);
        let plan = plan_indexes(&m, &listing, Age::new(NOW, Duration::ZERO));
        assert!(plan.delete.is_empty());
        assert_eq!(plan.too_young, 0);
    }

    #[tokio::test]
    async fn the_listing_finds_index_objects_by_name_only() {
        use object_store::memory::InMemory;
        let store = InMemory::new();
        let id = segment::segment_id(1, 1);
        for name in [
            "data.bin",
            "index-2.bin",
            "index-01.bin",
            "index-1.bin.tmp",
            "notes.txt",
        ] {
            let path = Path::from(format!("docs/segments/{id}/{name}"));
            store
                .put(&path, bytes::Bytes::from_static(b"seven!!").into())
                .await
                .expect("put");
        }
        let prefix = Path::from("docs/segments");
        let listing: Vec<ObjectMeta> = store.list(Some(&prefix)).try_collect().await.expect("list");

        let found = group(&prefix, listing);
        let dir = &found[&id];
        assert_eq!(dir.objects.len(), 5, "everything is still in the directory");
        let mut generations: Vec<u32> = dir.indexes.iter().map(|i| i.generation).collect();
        generations.sort_unstable();
        assert_eq!(
            generations,
            [2],
            "one valid generation, malformed names ignored"
        );
    }
}
