//! Building and publishing indexes: `Namespace::index`, the manifest's
//! `indexes`, and what compaction, garbage collection, `inspect` and `recover`
//! do with them.
//!
//! `assert_consistent` checks every index the manifest names against its
//! segment (a hand-written reading of the prefix), so each test that publishes
//! one ends there.

mod common;

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use common::faulty::{Fault, FaultyStore, PutKind};
use common::{Lcg, assert_consistent, assert_untouched, cas_backends, doc, schema, snapshot};
use futures::TryStreamExt;
use itzma_core::inspect::{self, SegmentState};
use itzma_core::manifest::{self, IndexEntry};
use itzma_core::{
    Compaction, CompactionOptions, GcOptions, GcReport, IndexKind, IndexOptions, IndexReport,
    ItzmaError, Manifest, Metric, Namespace, QueryOptions, segment,
};
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt};

const DIM: usize = 4;
const L2: Metric = Metric::L2;
const ROOMY: CompactionOptions = CompactionOptions {
    max_segments: 8,
    max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
};
const TIGHT: CompactionOptions = CompactionOptions {
    max_segments: 1,
    max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
};

/// Collect anything, however young.
const NOW: GcOptions = GcOptions {
    min_age: Duration::ZERO,
    dry_run: false,
    fence_wal: false,
};

/// Index every segment, however small.
/// Every segment, as SQ8: the tests here check SQ8's objects and sizes.
fn every() -> IndexOptions {
    IndexOptions::default().min_rows(0).kind(IndexKind::Sq8)
}

/// Index every segment, however small, as HNSW.
fn hnsw_every() -> IndexOptions {
    IndexOptions::default().min_rows(0).kind(IndexKind::Hnsw)
}

fn dynamic(store: &Arc<FaultyStore>) -> Arc<dyn ObjectStore> {
    store.clone()
}

async fn create(store: Arc<dyn ObjectStore>) -> Namespace {
    Namespace::create(store, "docs", schema(DIM, L2))
        .await
        .unwrap()
}

/// `n` documents named `<prefix>0000`, `<prefix>0001`, and so on, with random
/// vectors.
fn docs_of(prefix: &str, seed: u64, n: usize) -> Vec<itzma_core::Document> {
    let mut rng = Lcg(seed);
    (0..n)
        .map(|i| doc(&format!("{prefix}{i:04}"), &rng.vector(DIM)))
        .collect()
}

/// `n` documents, written as one batch.
async fn write(ns: &Namespace, prefix: &str, seed: u64, n: usize) {
    ns.upsert(docs_of(prefix, seed, n)).await.unwrap();
}

/// Two folds of `n` documents each, so the manifest lists `1-1` and `2-2`.
async fn two_segments(store: Arc<dyn ObjectStore>, n: usize) -> Namespace {
    let ns = create(store).await;
    write(&ns, "a", 1, n).await;
    ns.compact(&ROOMY).await.unwrap();
    write(&ns, "b", 2, n).await;
    ns.compact(&ROOMY).await.unwrap();
    ns
}

async fn manifest_of(store: &dyn ObjectStore) -> Manifest {
    manifest::load(store, "docs").await.unwrap().0
}

fn seg(lo: u64, hi: u64) -> String {
    segment::segment_id(lo, hi)
}

fn index_obj(id: &str, generation: u32) -> Path {
    Path::from(format!("docs/segments/{id}/index-{generation}.bin"))
}

/// Every index object in the store, by key.
async fn index_keys(store: &dyn ObjectStore) -> Vec<String> {
    snapshot(store)
        .await
        .into_keys()
        .filter(|key| key.contains("/index-"))
        .collect()
}

async fn hits(ns: &Namespace) -> Vec<String> {
    ns.query(&Lcg(7).vector(DIM), 1_000)
        .await
        .unwrap()
        .into_iter()
        .map(|hit| hit.id)
        .collect()
}

fn entry(generation: u32) -> IndexEntry {
    IndexEntry {
        generation,
        kind: "sq8".into(),
    }
}

fn report(indexed: usize, skipped_small: usize, already: usize) -> IndexReport {
    let mut r = IndexReport::default();
    r.indexed = indexed;
    r.skipped_small = skipped_small;
    r.already_indexed = already;
    r
}

#[tokio::test]
async fn an_hnsw_index_is_built_and_published() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = create(b.store.clone()).await;
        write(&ns, "a", 1, 200).await;
        ns.compact(&ROOMY).await.unwrap();

        let report = ns.index(&hnsw_every()).await.unwrap();
        assert_eq!(report.indexed, 1, "{report:?}");
        let m = manifest_of(&*b.store).await;
        assert_eq!(m.indexes[&seg(1, 1)].kind, "hnsw");

        let got = hits(&ns).await;
        let exact = ns
            .query_with(
                &Lcg(7).vector(DIM),
                1_000,
                &QueryOptions::default().exact(true),
            )
            .await
            .unwrap()
            .into_iter()
            .map(|hit| hit.id)
            .collect::<Vec<_>>();
        assert_eq!(
            got.len(),
            exact.len(),
            "an approximate index should not lose or invent documents"
        );
        assert_consistent(b.store.clone(), DIM, L2).await;
    }
}

#[tokio::test]
async fn an_index_is_built_and_published() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = two_segments(b.store.clone(), 30).await;
        let expected = hits(&ns).await;

        let outcome = ns.index(&every()).await.unwrap();
        assert_eq!(outcome, report(2, 0, 0));

        let m = manifest_of(&*b.store).await;
        assert_eq!(m.indexes.len(), 2);
        assert_eq!(m.indexes[&seg(1, 1)], entry(1));
        assert_eq!(m.indexes[&seg(2, 2)], entry(1));
        assert_eq!(
            index_keys(&*b.store).await,
            [
                index_obj(&seg(1, 1), 1).to_string(),
                index_obj(&seg(2, 2), 1).to_string()
            ]
        );

        assert_eq!(hits(&ns).await, expected, "an index changes no answer");
        assert_consistent(b.store.clone(), DIM, L2).await;
    }
}

/// With no kind named, `index` builds HNSW.
#[tokio::test]
async fn the_default_kind_is_hnsw() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = two_segments(b.store.clone(), 30).await;
        let expected = hits(&ns).await;

        let outcome = ns
            .index(&IndexOptions::default().min_rows(0))
            .await
            .unwrap();
        assert_eq!(outcome, report(2, 0, 0));

        let m = manifest_of(&*b.store).await;
        assert_eq!(m.indexes[&seg(1, 1)].kind, "hnsw");
        assert_eq!(m.indexes[&seg(2, 2)].kind, "hnsw");
        assert_eq!(hits(&ns).await, expected, "an index changes no answer");
        assert_consistent(b.store.clone(), DIM, L2).await;
    }
}

#[tokio::test]
async fn a_rerun_does_nothing() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let faulty = FaultyStore::new(b.store.clone());
        let ns = two_segments(dynamic(&faulty), 30).await;
        ns.index(&every()).await.unwrap();
        let before = snapshot(&*b.store).await;

        faulty.reset_ops();
        let again = ns.index(&every()).await.unwrap();
        assert_eq!(again, report(0, 0, 2));
        assert_eq!(faulty.ops(), 1, "the manifest read is all a rerun costs");
        assert_untouched(&*b.store, &before).await;
    }
}

#[tokio::test]
async fn a_segment_under_the_minimum_is_left_flat() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = two_segments(b.store.clone(), 20).await;
        let before = snapshot(&*b.store).await;

        let by_default = ns.index(&IndexOptions::default()).await.unwrap();
        assert_eq!(by_default, report(0, 2, 0), "20 is under 4096");
        let just_over = ns
            .index(&IndexOptions::default().min_rows(21))
            .await
            .unwrap();
        assert_eq!(just_over, report(0, 2, 0));
        assert_untouched(&*b.store, &before).await;
        assert!(index_keys(&*b.store).await.is_empty());

        let exactly = ns
            .index(&IndexOptions::default().min_rows(20))
            .await
            .unwrap();
        assert_eq!(exactly, report(2, 0, 0), "the minimum itself is indexed");
    }
}

#[tokio::test]
async fn a_segment_with_no_documents_is_indexed_when_asked() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = create(b.store.clone()).await;
        ns.upsert(vec![doc("x", &[1.0; DIM])]).await.unwrap();
        ns.compact(&ROOMY).await.unwrap();
        ns.delete(vec!["x".into()]).await.unwrap();
        ns.compact(&ROOMY).await.unwrap();

        assert_eq!(ns.index(&every()).await.unwrap(), report(2, 0, 0));
        assert_consistent(b.store.clone(), DIM, L2).await;
    }
}

#[tokio::test]
async fn a_namespace_below_segment_level_2_is_refused_untouched() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = two_segments(b.store.clone(), 30).await;
        manifest::update(&*b.store, "docs", |m| m.segment_format = 1)
            .await
            .unwrap();
        let before = snapshot(&*b.store).await;

        let err = ns.index(&every()).await.err();
        assert!(
            matches!(
                err,
                Some(ItzmaError::SegmentFormatTooOld { have: 1, need: 2 })
            ),
            "{err:?}"
        );
        assert_untouched(&*b.store, &before).await;
    }
}

/// A run that stops before it lists the index leaves an orphan, and the next run
/// uses it and does not write another.
#[tokio::test]
async fn an_orphan_from_a_stopped_run_is_left_for_gc_and_not_reused() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let faulty = FaultyStore::new(b.store.clone());
        let ns = two_segments(dynamic(&faulty), 30).await;

        faulty.fail_puts(
            "manifest/current.json",
            PutKind::Update,
            Fault::Generic,
            Some(1),
        );
        assert!(ns.index(&every()).await.is_err());
        faulty.clear();
        assert!(manifest_of(&*b.store).await.indexes.is_empty());
        let orphan = b
            .store
            .get(&index_obj(&seg(1, 1), 1))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();

        let outcome = ns.index(&every()).await.unwrap();
        assert_eq!(outcome, report(2, 0, 0));
        // The orphan's generation is taken, so the build names the next one.
        assert_eq!(
            manifest_of(&*b.store).await.indexes[&seg(1, 1)].generation,
            2
        );
        let kept = b
            .store
            .get(&index_obj(&seg(1, 1), 1))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(kept, orphan, "the orphan is left for gc, never overwritten");
        assert_consistent(b.store.clone(), DIM, L2).await;
    }
}

/// the rule: a generation number is never reused. After a manifest
/// loss (`recover` leaves `indexes` empty) and a collection of the old
/// generation, the next build must name a generation above every one the
/// segment ever had. Reusing generation 1 would let a process still holding
/// the old generation 1 read partition offsets from a different head.
#[tokio::test]
async fn a_generation_is_never_reused_after_recovery_and_collection() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = two_segments(b.store.clone(), 30).await;
        ns.index(&every()).await.unwrap();
        ns.index(&every().rebuild(true)).await.unwrap();
        assert_eq!(
            manifest_of(&*b.store).await.indexes[&seg(1, 1)].generation,
            2
        );

        // What gc does once generation 1 is old enough: it is no longer named.
        b.store.delete(&index_obj(&seg(1, 1), 1)).await.unwrap();
        b.store
            .delete(&Path::from("docs/manifest/current.json"))
            .await
            .unwrap();
        let (recovered, _) = Namespace::recover(b.store.clone(), "docs").await.unwrap();
        assert!(manifest_of(&*b.store).await.indexes.is_empty());

        recovered.index(&every()).await.unwrap();
        let entry = &manifest_of(&*b.store).await.indexes[&seg(1, 1)];
        assert_eq!(
            entry.generation, 3,
            "generation 2 still exists, so 3 is the next name, not 1"
        );
        assert_consistent(b.store.clone(), DIM, L2).await;
    }
}

/// Issue #35: the highest generation can be collected too (an orphan of a
/// stopped rebuild, or the named one once a lost manifest stops naming it).
/// The listing then cannot see its number, so the segment's high-water mark
/// must keep it from coming back.
#[tokio::test]
async fn a_collected_highest_generation_is_never_reused() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = two_segments(b.store.clone(), 30).await;
        ns.index(&every()).await.unwrap();
        ns.index(&every().rebuild(true)).await.unwrap();

        b.store.delete(&index_obj(&seg(1, 1), 2)).await.unwrap();
        b.store
            .delete(&Path::from("docs/manifest/current.json"))
            .await
            .unwrap();
        let (recovered, _) = Namespace::recover(b.store.clone(), "docs").await.unwrap();

        recovered.index(&every()).await.unwrap();
        let entry = &manifest_of(&*b.store).await.indexes[&seg(1, 1)];
        assert_eq!(
            entry.generation, 3,
            "generation 2 was claimed once, so it is never claimed again"
        );
        assert_consistent(b.store.clone(), DIM, L2).await;
    }
}

fn mark_of(id: &str) -> Path {
    Path::from(format!("docs/segments/{id}/generation"))
}

async fn read_mark(store: &dyn ObjectStore, id: &str) -> String {
    let bytes = store.get(&mark_of(id)).await.unwrap().bytes().await;
    String::from_utf8(bytes.unwrap().to_vec()).unwrap()
}

#[tokio::test]
async fn a_build_raises_the_mark_to_the_generation_it_claims() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = two_segments(b.store.clone(), 30).await;
        ns.index(&every()).await.unwrap();
        assert_eq!(read_mark(&*b.store, &seg(1, 1)).await, "1");
        ns.index(&every().rebuild(true)).await.unwrap();
        assert_eq!(read_mark(&*b.store, &seg(1, 1)).await, "2");
        assert_eq!(read_mark(&*b.store, &seg(2, 2)).await, "2");
    }
}

/// A mark above every object wins (the objects it counted were collected), and
/// a listing above the mark wins (objects from before the mark existed).
#[tokio::test]
async fn the_higher_of_the_mark_and_the_listing_decides() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = two_segments(b.store.clone(), 30).await;
        b.store
            .put(&mark_of(&seg(1, 1)), Bytes::from("7").into())
            .await
            .unwrap();
        b.store
            .put(&index_obj(&seg(2, 2), 4), Bytes::from("orphan").into())
            .await
            .unwrap();
        b.store
            .put(&mark_of(&seg(2, 2)), Bytes::from("2").into())
            .await
            .unwrap();

        ns.index(&every()).await.unwrap();
        let m = manifest_of(&*b.store).await;
        assert_eq!(m.indexes[&seg(1, 1)].generation, 8);
        assert_eq!(m.indexes[&seg(2, 2)].generation, 5);
        assert_eq!(read_mark(&*b.store, &seg(2, 2)).await, "5");
    }
}

/// Reading a damaged mark low could reuse a number, so it stops the build
/// before any index object is written.
#[tokio::test]
async fn a_garbled_mark_is_corrupt_and_nothing_is_written() {
    for garbled in ["", "x1", "-3", "4294967296", "12345678901"] {
        let b = cas_backends().swap_remove(0);
        let ns = two_segments(b.store.clone(), 30).await;
        b.store
            .put(&mark_of(&seg(1, 1)), Bytes::from(garbled).into())
            .await
            .unwrap();
        let before = snapshot(&*b.store).await;

        let err = ns.index(&every()).await.unwrap_err();
        assert!(
            matches!(&err, ItzmaError::CorruptIndex { path, .. } if path.ends_with("/generation")),
            "{garbled:?}: {err:?}"
        );
        assert_untouched(&*b.store, &before).await;
    }
}

/// A lost race on the mark (another build claimed between the read and the
/// put) is retried from a fresh read.
#[tokio::test]
async fn a_lost_race_on_the_mark_is_retried() {
    for (kind, fault, preset) in [
        (PutKind::Create, Fault::AlreadyExists, None),
        (PutKind::Update, Fault::Precondition, Some("3")),
    ] {
        let b = cas_backends().swap_remove(0);
        let faulty = FaultyStore::new(b.store.clone());
        let ns = two_segments(dynamic(&faulty), 30).await;
        if let Some(preset) = preset {
            b.store
                .put(&mark_of(&seg(1, 1)), Bytes::from(preset).into())
                .await
                .unwrap();
        }
        faulty.fail_puts("/generation", kind, fault, Some(1));
        ns.index(&every()).await.unwrap();
        faulty.clear();
        let expected = preset.map_or(1, |_| 4);
        assert_eq!(
            manifest_of(&*b.store).await.indexes[&seg(1, 1)].generation,
            expected,
            "{kind:?}"
        );
        assert_consistent(b.store.clone(), DIM, L2).await;
    }
}

#[tokio::test]
async fn endless_races_on_the_mark_give_up_with_contention() {
    let b = cas_backends().swap_remove(0);
    let faulty = FaultyStore::new(b.store.clone());
    let ns = two_segments(dynamic(&faulty), 30).await;
    faulty.fail_puts("/generation", PutKind::Create, Fault::AlreadyExists, None);
    let err = ns.index(&every()).await.unwrap_err();
    assert!(
        matches!(err, ItzmaError::ManifestContention { attempts: 32 }),
        "{err:?}"
    );
    assert!(index_keys(&*b.store).await.is_empty());
}

/// The state the generation rule is for: segment `1-1` indexed at generation 1
/// then rebuilt at 2, both objects collected and the manifest lost, then
/// recovered (`recover` leaves `indexes` empty). Only the segment's
/// high-water mark still knows that 1 and 2 were taken.
async fn recovered_after_a_rebuild_and_collection(store: Arc<dyn ObjectStore>) -> Namespace {
    let ns = two_segments(store.clone(), 30).await;
    ns.index(&every()).await.unwrap();
    ns.index(&every().rebuild(true)).await.unwrap();
    store.delete(&index_obj(&seg(1, 1), 1)).await.unwrap();
    store.delete(&index_obj(&seg(1, 1), 2)).await.unwrap();
    store
        .delete(&Path::from("docs/manifest/current.json"))
        .await
        .unwrap();
    Namespace::recover(store, "docs").await.unwrap().0
}

/// Every generation of `id`'s index objects that exists, named or not.
async fn generations(store: &dyn ObjectStore, id: &str) -> Vec<u32> {
    let prefix = format!("docs/segments/{id}/index-");
    snapshot(store)
        .await
        .into_keys()
        .filter_map(|key| {
            key.strip_prefix(&prefix)?
                .strip_suffix(".bin")?
                .parse::<u32>()
                .ok()
        })
        .collect()
}

/// Fault every operation of a build after a manifest loss and a collection of
/// every generation, as a plain error and as a lost ack. Whatever
/// the fault left behind, generations 1 and 2 never come back, the rerun names
/// the generation the mark holds, and the answers are the same.
#[tokio::test]
async fn a_stopped_build_after_recovery_never_reuses_a_generation() {
    for ix in 0..cas_backends().len() {
        let counted = FaultyStore::new(cas_backends().swap_remove(ix).store);
        let ns = recovered_after_a_rebuild_and_collection(dynamic(&counted)).await;
        counted.reset_ops();
        ns.index(&every()).await.unwrap();
        let total = counted.ops();
        assert!(
            total >= 10,
            "two segments are ten or so operations, not {total}"
        );

        for fault in [Fault::Generic, Fault::LostAck] {
            for n in 0..total {
                eprintln!("backend {ix}: {fault:?} at operation {n} of {total}");
                let store = FaultyStore::new(cas_backends().swap_remove(ix).store);
                let ns = recovered_after_a_rebuild_and_collection(dynamic(&store)).await;
                let expected = hits(&ns).await;

                store.fail_nth_op(n, fault);
                let stopped = ns.index(&every()).await;
                store.clear();
                assert!(stopped.is_err(), "a fault inside index must show");
                drop(ns);

                assert_consistent(dynamic(&store), DIM, L2).await;
                let ns = Namespace::open(dynamic(&store), "docs").await.unwrap();
                // A stopped run may already have published this segment. Then
                // the rerun keeps that entry.
                let published = manifest_of(&*store)
                    .await
                    .indexes
                    .get(&seg(1, 1))
                    .map(|entry| entry.generation);
                ns.index(&every()).await.unwrap();

                let named = manifest_of(&*store).await.indexes[&seg(1, 1)].generation;
                if let Some(generation) = published {
                    assert_eq!(named, generation, "a published index is not rebuilt");
                }
                assert!(named > 2, "1 and 2 were claimed before, not {named}");
                let mark: u32 = read_mark(&*store, &seg(1, 1)).await.parse().unwrap();
                assert_eq!(named, mark, "the named generation is the last claimed");
                let existing = generations(&*store, &seg(1, 1)).await;
                assert!(
                    existing.iter().all(|g| *g > 2 && *g <= mark),
                    "only claimed generations above the collected ones exist: {existing:?}"
                );
                assert_eq!(hits(&ns).await, expected);
                assert_consistent(dynamic(&store), DIM, L2).await;
            }
        }
    }
}

/// Two indexers of one segment: the one that loses the put-if-absent takes the
/// winner's object, and the manifest lists it once.
#[tokio::test]
async fn two_indexers_at_once_both_succeed() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let faulty = FaultyStore::new(b.store.clone());
        let ns = two_segments(dynamic(&faulty), 30).await;
        let other = Namespace::open(b.store.clone(), "docs").await.unwrap();

        // The other indexer runs to the end between this one's first read of
        // the segment and its write, so this one finds the object in its way.
        faulty.before_get(&seg(1, 1), async move {
            let done = other.index(&every()).await.unwrap();
            assert_eq!(done.indexed, 2);
        });
        let outcome = ns.index(&every()).await.unwrap();
        assert_eq!(outcome, report(0, 0, 2), "both were the other's");

        let m = manifest_of(&*b.store).await;
        assert_eq!(m.indexes.len(), 2);
        // This indexer listed the segment after the other had published, so it
        // wrote generation 2 and lost the publish. Its two objects are orphans
        // for gc; the manifest names the other's generation 1.
        assert_eq!(m.indexes[&seg(1, 1)].generation, 1);
        assert_eq!(index_keys(&*b.store).await.len(), 4);
        assert_consistent(b.store.clone(), DIM, L2).await;
    }
}

#[tokio::test]
async fn two_indexers_started_together_finish_with_one_index_each() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = two_segments(b.store.clone(), 30).await;
        let other = Namespace::open(b.store.clone(), "docs").await.unwrap();

        let options = every();
        let (one, two) = tokio::join!(ns.index(&options), other.index(&options));
        let (one, two) = (one.unwrap(), two.unwrap());
        assert_eq!(one.indexed + two.indexed, 2, "{one:?} {two:?}");
        assert_eq!(one.already_indexed + two.already_indexed + 2, 4);

        assert_eq!(manifest_of(&*b.store).await.indexes.len(), 2);
        // Per segment, either both listed before either wrote and the loser
        // reused the winner's object, or the loser listed after the winner
        // wrote and left its own generation as an orphan. Which one
        // depends on timing, so each segment holds one or two objects, and gc
        // brings it back to the named one.
        let keys = index_keys(&*b.store).await.len();
        assert!((2..=4).contains(&keys), "{keys} index objects");
        assert_consistent(b.store.clone(), DIM, L2).await;
        ns.gc(&NOW).await.unwrap();
        assert_eq!(index_keys(&*b.store).await.len(), 2);
        assert_consistent(b.store.clone(), DIM, L2).await;
    }
}

/// Any object already under a segment's prefix counts as a generation in use,
/// so a build steps over it: the object is neither read nor overwritten, and
/// the new index takes the next generation. A damaged or foreign object is
/// gc's to delete, not this build's to refuse.
#[tokio::test]
async fn an_object_in_the_way_is_stepped_over_and_never_overwritten() {
    // A valid index of a different segment: 25 rows, not 30.
    let elsewhere = common::memory();
    let other = two_segments(elsewhere.clone(), 25).await;
    other.index(&every()).await.unwrap();
    let foreign = elsewhere
        .get(&index_obj(&seg(1, 1), 1))
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();

    let plants: [(&str, Bytes); 3] = [
        ("truncated", Bytes::from_static(b"ITZI")),
        (
            "not an index",
            Bytes::from_static(b"this is not an index at all, nor close"),
        ),
        ("another segment's", foreign),
    ];
    for (what, bytes) in plants {
        // A fresh namespace per plant: the first build would change the state
        // the others start from.
        let store = common::memory();
        let ns = two_segments(store.clone(), 30).await;
        store
            .put(&index_obj(&seg(1, 1), 1), bytes.clone().into())
            .await
            .unwrap();

        assert_eq!(ns.index(&every()).await.unwrap(), report(2, 0, 0), "{what}");
        let kept = store
            .get(&index_obj(&seg(1, 1), 1))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(kept, bytes, "{what}: the planted object is left as it was");
        assert_eq!(
            manifest_of(&*store).await.indexes[&seg(1, 1)].generation,
            2,
            "{what}"
        );
        assert_consistent(store.clone(), DIM, L2).await;
    }
}

#[tokio::test]
async fn a_fold_keeps_the_indexes_and_leaves_its_own_segment_without_one() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = two_segments(b.store.clone(), 30).await;
        ns.index(&every()).await.unwrap();

        write(&ns, "c", 3, 30).await;
        assert!(matches!(
            ns.compact(&ROOMY).await.unwrap(),
            Compaction::Published { .. }
        ));
        let m = manifest_of(&*b.store).await;
        assert_eq!(m.live_segments.len(), 3);
        assert_eq!(m.indexes.len(), 2, "a fold retires nothing");

        assert_eq!(ns.index(&every()).await.unwrap(), report(1, 0, 2));
        assert_consistent(b.store.clone(), DIM, L2).await;
    }
}

/// A merge retires the segments, so it drops their entries in the same swap.
/// The objects stay until collected, and their directories go whole.
#[tokio::test]
async fn a_merge_drops_the_entries_and_the_old_indexes_become_garbage() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = two_segments(b.store.clone(), 30).await;
        ns.index(&every()).await.unwrap();
        let expected = hits(&ns).await;

        let merged = ns.compact(&TIGHT).await.unwrap();
        assert!(matches!(
            merged,
            Compaction::Published {
                merged_segments: 2,
                ..
            }
        ));
        let m = manifest_of(&*b.store).await;
        assert!(m.indexes.is_empty(), "{:?}", m.indexes);
        assert_eq!(index_keys(&*b.store).await.len(), 2, "still stored");
        assert_consistent(b.store.clone(), DIM, L2).await;

        let freed = ns.gc(&NOW).await.unwrap();
        assert_eq!(freed.segments_deleted, 2);
        assert!(index_keys(&*b.store).await.is_empty());
        assert_eq!(hits(&ns).await, expected);

        assert_eq!(ns.index(&every()).await.unwrap(), report(1, 0, 0));
        assert_consistent(b.store.clone(), DIM, L2).await;
    }
}

#[tokio::test]
async fn a_rebuild_publishes_the_next_generation() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = two_segments(b.store.clone(), 30).await;
        ns.index(&every()).await.unwrap();

        let again = ns.index(&every().rebuild(true)).await.unwrap();
        assert_eq!(again, report(2, 0, 0));
        let m = manifest_of(&*b.store).await;
        assert_eq!(m.indexes[&seg(1, 1)], entry(2));
        assert_eq!(m.indexes[&seg(2, 2)], entry(2));
        assert_eq!(index_keys(&*b.store).await.len(), 4, "the old ones stay");
        assert_consistent(b.store.clone(), DIM, L2).await;

        // Without `rebuild` the segments are left as they are.
        assert_eq!(ns.index(&every()).await.unwrap(), report(0, 0, 2));

        let freed = ns.gc(&NOW).await.unwrap();
        assert_eq!(freed.indexes_deleted, 2);
        assert_eq!(freed.segments_deleted, 0);
        assert_eq!(
            index_keys(&*b.store).await,
            [
                index_obj(&seg(1, 1), 2).to_string(),
                index_obj(&seg(2, 2), 2).to_string()
            ]
        );
        assert_consistent(b.store.clone(), DIM, L2).await;
    }
}

#[tokio::test]
async fn a_rebuild_can_switch_kind() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = two_segments(b.store.clone(), 30).await;
        ns.index(&every()).await.unwrap(); // both segments, sq8
        let before = manifest_of(&*b.store).await;
        assert_eq!(before.indexes[&seg(1, 1)].kind, "sq8");

        let report = ns
            .index(&every().rebuild(true).kind(itzma_core::IndexKind::Hnsw))
            .await
            .unwrap();
        assert_eq!(report.indexed, 2, "{report:?}");
        let after = manifest_of(&*b.store).await;
        assert_eq!(after.indexes[&seg(1, 1)].kind, "hnsw");
        assert_eq!(
            after.indexes[&seg(1, 1)].generation,
            before.indexes[&seg(1, 1)].generation + 1
        );
        assert_consistent(b.store.clone(), DIM, L2).await;
    }
}

/// Two rebuilds at once end at generation 2, not 3.
#[tokio::test]
async fn two_rebuilds_at_once_publish_one_generation() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let faulty = FaultyStore::new(b.store.clone());
        let ns = two_segments(dynamic(&faulty), 30).await;
        ns.index(&every()).await.unwrap();
        let other = Namespace::open(b.store.clone(), "docs").await.unwrap();

        faulty.before_get(&seg(1, 1), async move {
            other.index(&every().rebuild(true)).await.unwrap();
        });
        // Both of this run's builds find the other's objects in the way and its
        // entries in the manifest, so neither is this run's to count.
        let outcome = ns.index(&every().rebuild(true)).await.unwrap();
        assert_eq!(outcome, report(0, 0, 2));

        let m = manifest_of(&*b.store).await;
        assert_eq!(m.indexes[&seg(1, 1)], entry(2));
        assert_eq!(m.indexes[&seg(2, 2)], entry(2));
        assert_consistent(b.store.clone(), DIM, L2).await;
    }
}

#[tokio::test]
async fn gc_never_deletes_an_index_the_manifest_names() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = two_segments(b.store.clone(), 30).await;
        ns.index(&every()).await.unwrap();
        let before = snapshot(&*b.store).await;

        assert_eq!(ns.gc(&NOW).await.unwrap(), GcReport::default());
        assert_eq!(snapshot(&*b.store).await, before);

        // After a rebuild the older generation is unnamed and would go, and a dry
        // run says so without deleting it.
        ns.index(&every().rebuild(true)).await.unwrap();
        let dry = ns
            .gc(&GcOptions {
                dry_run: true,
                ..NOW
            })
            .await
            .unwrap();
        assert_eq!((dry.indexes_deleted, dry.indexes_too_young), (2, 0));
        assert_eq!(
            index_keys(&*b.store).await.len(),
            4,
            "a dry run deletes nothing"
        );
    }
}

#[tokio::test]
async fn gc_takes_an_orphan_index_in_a_live_directory() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let faulty = FaultyStore::new(b.store.clone());
        let ns = two_segments(dynamic(&faulty), 30).await;
        faulty.fail_puts(
            "manifest/current.json",
            PutKind::Update,
            Fault::Generic,
            Some(1),
        );
        assert!(ns.index(&every()).await.is_err());
        faulty.clear();
        assert_eq!(index_keys(&*b.store).await.len(), 1);

        let freed = ns.gc(&NOW).await.unwrap();
        assert_eq!((freed.indexes_deleted, freed.segments_deleted), (1, 0));
        assert!(freed.bytes_freed > 0);
        assert!(index_keys(&*b.store).await.is_empty());
        assert_consistent(b.store.clone(), DIM, L2).await;
    }
}

/// The age rules on the real clock: an orphan waits `min_age`, and a replaced
/// generation waits until the one that replaced it is `min_age` old too.
async fn gc_ages_indexes(store: Arc<dyn ObjectStore>) {
    let faulty = FaultyStore::new(store.clone());
    let ns = two_segments(dynamic(&faulty), 30).await;
    let aged = GcOptions {
        min_age: Duration::from_secs(2),
        ..NOW
    };

    // An orphan of segment 1, brand new: kept.
    faulty.fail_puts(
        "manifest/current.json",
        PutKind::Update,
        Fault::Generic,
        Some(1),
    );
    assert!(ns.index(&every()).await.is_err());
    faulty.clear();
    let young = ns.gc(&aged).await.unwrap();
    assert_eq!((young.indexes_deleted, young.indexes_too_young), (0, 1));
    assert_eq!(index_keys(&*store).await.len(), 1);

    // Past the age: taken. (S3 keeps whole seconds, so wait one more.)
    tokio::time::sleep(Duration::from_secs(3)).await;
    let old = ns.gc(&aged).await.unwrap();
    assert_eq!((old.indexes_deleted, old.indexes_too_young), (1, 0));

    // The first build is old when the rebuild is published, and stays until
    // the rebuild is old as well. Segment 1's collected orphan took generation
    // 1, so its builds are 2 and 3 (a number is never reused).
    ns.index(&every()).await.unwrap();
    tokio::time::sleep(Duration::from_secs(3)).await;
    ns.index(&every().rebuild(true)).await.unwrap();
    let replaced = ns.gc(&aged).await.unwrap();
    assert_eq!(
        (replaced.indexes_deleted, replaced.indexes_too_young),
        (0, 2),
        "the replacement is young"
    );
    tokio::time::sleep(Duration::from_secs(3)).await;
    let done = ns.gc(&aged).await.unwrap();
    assert_eq!((done.indexes_deleted, done.indexes_too_young), (2, 0));
    assert_eq!(
        index_keys(&*store).await,
        [
            index_obj(&seg(1, 1), 3).to_string(),
            index_obj(&seg(2, 2), 2).to_string()
        ]
    );
    assert_consistent(store, DIM, L2).await;
}

#[tokio::test]
async fn gc_waits_for_the_age_of_an_index_and_of_its_replacement() {
    let runs = cas_backends().into_iter().map(|b| {
        eprintln!("backend: {}", b.name);
        async move {
            gc_ages_indexes(b.store.clone()).await;
            drop(b);
        }
    });
    futures::future::join_all(runs).await;
}

/// The manifest is behind: a merge and a collection ran after this call loaded
/// it. Segments that are gone are counted, not errors, and what was already
/// published stays.
#[tokio::test]
async fn segments_merged_away_during_a_run_are_counted_not_failed() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let faulty = FaultyStore::new(b.store.clone());
        let ns = two_segments(dynamic(&faulty), 30).await;

        let store = b.store.clone();
        faulty.before_get(&seg(2, 2), async move {
            merge_and_delete_old(store).await;
        });
        let outcome = ns.index(&every()).await.unwrap();
        assert_eq!(
            (outcome.indexed, outcome.superseded),
            (1, 1),
            "segment 1 was published before the merge, 2 was gone after it"
        );
        assert!(
            manifest_of(&*b.store).await.indexes.is_empty(),
            "the merge pruned it"
        );
        assert_consistent(b.store.clone(), DIM, L2).await;
    }
}

#[tokio::test]
async fn a_merge_before_the_first_read_supersedes_every_segment() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let faulty = FaultyStore::new(b.store.clone());
        let ns = two_segments(dynamic(&faulty), 30).await;

        let store = b.store.clone();
        faulty.before_get(&seg(1, 1), async move {
            merge_and_delete_old(store).await;
        });
        let outcome = ns.index(&every()).await.unwrap();
        assert_eq!((outcome.indexed, outcome.superseded), (0, 2));
        assert!(index_keys(&*b.store).await.is_empty());
        assert_consistent(b.store.clone(), DIM, L2).await;
    }
}

/// The merge lands after the run has read the segment but before it publishes:
/// the swap declines, since the segment is no longer live, and the object it
/// wrote is garbage in a dead directory.
#[tokio::test]
async fn a_merge_between_the_write_and_the_publish_declines_the_publish() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let faulty = FaultyStore::new(b.store.clone());
        let ns = two_segments(dynamic(&faulty), 30).await;

        let other = Namespace::open(b.store.clone(), "docs").await.unwrap();
        faulty.before_get(&seg(1, 1), async move {
            other.compact(&TIGHT).await.unwrap();
        });
        let outcome = ns.index(&every()).await.unwrap();
        assert_eq!((outcome.indexed, outcome.superseded), (0, 2));
        assert!(manifest_of(&*b.store).await.indexes.is_empty());
        assert_eq!(
            index_keys(&*b.store).await.len(),
            2,
            "written, never listed"
        );

        let freed = ns.gc(&NOW).await.unwrap();
        assert_eq!(freed.segments_deleted, 2);
        assert!(
            index_keys(&*b.store).await.is_empty(),
            "the directories go whole"
        );
        assert_consistent(b.store.clone(), DIM, L2).await;
    }
}

/// Retire both segments as a merge followed by a collection would.
async fn merge_and_delete_old(store: Arc<dyn ObjectStore>) {
    let ns = Namespace::open(store.clone(), "docs").await.unwrap();
    ns.compact(&TIGHT).await.unwrap();
    for id in [seg(1, 1), seg(2, 2)] {
        store
            .delete(&segment::segment_path("docs", &id))
            .await
            .unwrap();
    }
}

/// A segment that is gone while the manifest still lists it is damage, and it
/// comes back as one.
#[tokio::test]
async fn a_listed_segment_that_is_really_missing_is_an_error() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = two_segments(b.store.clone(), 30).await;
        b.store
            .delete(&segment::segment_path("docs", &seg(1, 1)))
            .await
            .unwrap();
        let before = snapshot(&*b.store).await;

        let err = ns.index(&every()).await.err();
        assert!(
            matches!(err, Some(ItzmaError::CorruptSegment { .. })),
            "{err:?}"
        );
        assert_eq!(snapshot(&*b.store).await, before);
    }
}

/// Each segment is read under a budget of its own: a namespace is not limited by
/// the sum of its segments, but one segment has to fit.
#[tokio::test]
async fn the_read_budget_is_per_segment() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        two_segments(b.store.clone(), 30).await;
        let size = snapshot(&*b.store)
            .await
            .iter()
            .filter(|(key, _)| key.ends_with("/data.bin"))
            .map(|(_, bytes)| bytes.len() as u64)
            .max()
            .unwrap();
        // The 20-byte look at the header, then the whole object.
        let exact = size + 20;

        let tight = Namespace::open(b.store.clone(), "docs")
            .await
            .unwrap()
            .with_read_budget(exact - 1);
        let before = snapshot(&*b.store).await;
        let err = tight.index(&every()).await.err();
        assert!(
            matches!(err, Some(ItzmaError::ReadBudgetExceeded { .. })),
            "{err:?}"
        );
        assert_eq!(snapshot(&*b.store).await, before);

        let fits = Namespace::open(b.store.clone(), "docs")
            .await
            .unwrap()
            .with_read_budget(exact);
        assert_eq!(
            fits.index(&every()).await.unwrap(),
            report(2, 0, 0),
            "two segments each at the budget"
        );
        assert_consistent(b.store.clone(), DIM, L2).await;
    }
}

#[tokio::test]
async fn inspect_shows_which_index_the_manifest_names() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = two_segments(b.store.clone(), 30).await;
        let plain = inspect::report(&*b.store, "docs").await.unwrap();
        assert!(
            plain
                .segments
                .iter()
                .all(|s| s.index.is_none() && s.stray_indexes.is_empty())
        );

        ns.index(&every()).await.unwrap();
        let indexed = inspect::report(&*b.store, "docs").await.unwrap();
        assert!(indexed.problems.is_empty(), "{:?}", indexed.problems);
        let first = &indexed.segments[0];
        let info = first.index.as_ref().expect("index");
        assert_eq!((info.generation, info.kind.as_str()), (1, "sq8"));
        let stored = b.store.head(&index_obj(&seg(1, 1), 1)).await.unwrap().size;
        assert_eq!(info.bytes, Some(stored));
        assert!(indexed.to_string().contains("index 1 sq8"), "{indexed}");

        ns.index(&every().rebuild(true)).await.unwrap();
        let rebuilt = inspect::report(&*b.store, "docs").await.unwrap();
        assert_eq!(
            rebuilt.segments[0].index.as_ref().map(|i| i.generation),
            Some(2)
        );
        assert_eq!(rebuilt.segments[0].stray_indexes, [1]);
        assert!(rebuilt.problems.is_empty());
        assert!(
            rebuilt.to_string().contains("unnamed indexes [1]"),
            "{rebuilt}"
        );
    }
}

#[tokio::test]
async fn inspect_shows_an_hnsw_index_the_same_shape_as_sq8() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = two_segments(b.store.clone(), 30).await;
        ns.index(&every().kind(itzma_core::IndexKind::Hnsw))
            .await
            .unwrap();
        let report = inspect::report(&*b.store, "docs").await.unwrap();
        let seg1 = report
            .segments
            .iter()
            .find(|s| s.id == seg(1, 1))
            .expect("segment 1");
        let index = seg1.index.as_ref().expect("named index");
        assert_eq!(index.kind, "hnsw");
        assert!(index.bytes.is_some());
    }
}

#[tokio::test]
async fn inspect_reports_a_named_index_that_is_not_there() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = two_segments(b.store.clone(), 30).await;
        ns.index(&every()).await.unwrap();
        b.store.delete(&index_obj(&seg(2, 2), 1)).await.unwrap();

        let r = inspect::report(&*b.store, "docs").await.unwrap();
        assert_eq!(r.problems.len(), 1, "{:?}", r.problems);
        assert!(r.problems[0].contains(&seg(2, 2)) && r.problems[0].contains("index"));
        assert_eq!(r.segments[1].index.as_ref().map(|i| i.bytes), Some(None));
        assert!(r.to_string().contains("missing"), "{r}");
    }
}

#[tokio::test]
async fn inspect_reports_an_index_for_a_segment_that_is_not_live() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        two_segments(b.store.clone(), 30).await;
        manifest::update(&*b.store, "docs", |m| {
            m.indexes.insert(seg(9, 9), entry(1));
        })
        .await
        .unwrap();

        let r = inspect::report(&*b.store, "docs").await.unwrap();
        assert_eq!(r.problems.len(), 1, "{:?}", r.problems);
        assert!(r.problems[0].contains(&seg(9, 9)), "{:?}", r.problems);
        assert!(r.segments.iter().all(|s| s.state == SegmentState::Live));
    }
}

#[tokio::test]
async fn a_manifest_without_the_field_has_no_indexes() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = create(b.store.clone()).await;
        drop(ns);
        // The manifest as a build from before `indexes` wrote it.
        let old = br#"{"format_version":1,"schema":{"dim":4,"metric":"l2","attrs":{}},"live_segments":[],"wal_folded_through":0,"wal_format":4,"segment_format":2}"#;
        b.store
            .put(
                &manifest::manifest_path("docs"),
                Bytes::from_static(old).into(),
            )
            .await
            .unwrap();

        let m = manifest_of(&*b.store).await;
        assert!(m.indexes.is_empty());
        let ns = Namespace::open(b.store.clone(), "docs").await.unwrap();
        ns.upsert(vec![doc("a", &[1.0; DIM])]).await.unwrap();
        assert_eq!(hits(&ns).await, ["a"]);
    }
}

#[test]
fn an_entry_is_written_as_gen_and_kind() {
    let text = serde_json::to_string(&entry(3)).unwrap();
    assert_eq!(text, r#"{"gen":3,"kind":"sq8"}"#);
    let back: IndexEntry = serde_json::from_str(&text).unwrap();
    assert_eq!(back, entry(3));
}

#[test]
fn the_options_and_the_report_are_plain_data() {
    let options = IndexOptions::default().min_rows(9).rebuild(true);
    let text = serde_json::to_string(&options).unwrap();
    assert_eq!(
        serde_json::from_str::<IndexOptions>(&text).unwrap(),
        options
    );
    assert_eq!(
        serde_json::from_str::<IndexOptions>("{}").unwrap(),
        IndexOptions::default(),
        "a missing field is the default"
    );
    assert_eq!(IndexOptions::default().min_rows, 4096);
    assert!(!IndexOptions::default().rebuild);

    let outcome = report(1, 2, 4);
    let text = serde_json::to_string(&outcome).unwrap();
    assert_eq!(serde_json::from_str::<IndexReport>(&text).unwrap(), outcome);
}

/// Fault every operation of a run that indexes two segments, as a plain error
/// and as a lost ack. Afterwards the namespace is consistent and a rerun
/// finishes what the first left.
#[tokio::test]
async fn a_stopped_run_is_finished_by_the_next_one() {
    for ix in 0..cas_backends().len() {
        let counted = FaultyStore::new(cas_backends().swap_remove(ix).store);
        let ns = two_segments(dynamic(&counted), 30).await;
        counted.reset_ops();
        ns.index(&every()).await.unwrap();
        let total = counted.ops();
        assert!(
            total >= 10,
            "two segments are ten or so operations, not {total}"
        );

        for fault in [Fault::Generic, Fault::LostAck] {
            for n in 0..total {
                eprintln!("backend {ix}: {fault:?} at operation {n} of {total}");
                let store = FaultyStore::new(cas_backends().swap_remove(ix).store);
                let ns = two_segments(dynamic(&store), 30).await;
                let expected = hits(&ns).await;

                store.fail_nth_op(n, fault);
                let stopped = ns.index(&every()).await;
                store.clear();
                assert!(stopped.is_err(), "a fault inside index must show");
                drop(ns);

                assert_consistent(dynamic(&store), DIM, L2).await;
                let ns = Namespace::open(dynamic(&store), "docs").await.unwrap();
                let finished = ns.index(&every()).await.unwrap();
                assert_eq!(
                    finished.indexed + finished.already_indexed,
                    2,
                    "{finished:?}"
                );
                assert_eq!(manifest_of(&*store).await.indexes.len(), 2);
                assert_eq!(hits(&ns).await, expected);
                assert_consistent(dynamic(&store), DIM, L2).await;

                assert_eq!(ns.index(&every()).await.unwrap(), report(0, 0, 2));
            }
        }
    }
}

/// The same sweep, over a clustered build: a fault is always an error, and a
/// rerun finishes what the first left, over the same data an unclustered build
/// already covers above.
#[tokio::test]
async fn a_stopped_clustered_build_is_finished_by_the_next_one() {
    let clustered = || IndexOptions::default().min_rows(0).nlist(Some(6));
    for ix in 0..cas_backends().len() {
        let counted = FaultyStore::new(cas_backends().swap_remove(ix).store);
        let ns = two_segments(dynamic(&counted), 60).await;
        counted.reset_ops();
        ns.index(&clustered()).await.unwrap();
        let total = counted.ops();
        assert!(
            total >= 10,
            "two segments cost more than a handful of operations, not {total}"
        );

        for fault in [Fault::Generic, Fault::LostAck] {
            for n in 0..total {
                eprintln!("backend {ix}: clustered {fault:?} at operation {n} of {total}");
                let store = FaultyStore::new(cas_backends().swap_remove(ix).store);
                let ns = two_segments(dynamic(&store), 60).await;
                let expected = hits(&ns).await;

                store.fail_nth_op(n, fault);
                let stopped = ns.index(&clustered()).await;
                store.clear();
                assert!(
                    stopped.is_err(),
                    "a fault inside a clustered index must show"
                );
                drop(ns);

                assert_consistent(dynamic(&store), DIM, L2).await;
                let ns = Namespace::open(dynamic(&store), "docs").await.unwrap();
                let finished = ns.index(&clustered()).await.unwrap();
                assert_eq!(
                    finished.indexed + finished.already_indexed,
                    2,
                    "{finished:?}"
                );
                assert_eq!(manifest_of(&*store).await.indexes.len(), 2);
                assert_eq!(hits(&ns).await, expected);
                assert_consistent(dynamic(&store), DIM, L2).await;

                assert_eq!(ns.index(&clustered()).await.unwrap(), report(0, 0, 2));
            }
        }
    }
}

#[tokio::test]
async fn the_index_future_can_be_spawned() {
    let store = common::memory();
    let ns = two_segments(store, 30).await;
    let handle = tokio::spawn(async move { ns.index(&every()).await });
    assert_eq!(handle.await.unwrap().unwrap().indexed, 2);
}

/// Recovery does not restore what is derived: the indexes come back only when
/// `index` is run, and it takes the objects that are still there.
#[tokio::test]
async fn recovery_leaves_the_indexes_unlisted_and_index_takes_them_back() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = two_segments(b.store.clone(), 30).await;
        ns.index(&every()).await.unwrap();
        let objects = index_keys(&*b.store).await;
        drop(ns);
        b.store
            .delete(&manifest::manifest_path("docs"))
            .await
            .unwrap();

        let (ns, _) = Namespace::recover(b.store.clone(), "docs").await.unwrap();
        assert!(manifest_of(&*b.store).await.indexes.is_empty());
        assert_eq!(
            index_keys(&*b.store).await,
            objects,
            "recovery deletes nothing"
        );
        assert_consistent(b.store.clone(), DIM, L2).await;

        let kept: Vec<(String, Bytes)> = {
            let mut all = Vec::new();
            for key in &objects {
                let bytes = b
                    .store
                    .get(&Path::from(key.as_str()))
                    .await
                    .unwrap()
                    .bytes()
                    .await
                    .unwrap();
                all.push((key.clone(), bytes));
            }
            all
        };
        assert_eq!(ns.index(&every()).await.unwrap(), report(2, 0, 0));
        for (key, bytes) in kept {
            let now = b
                .store
                .get(&Path::from(key.as_str()))
                .await
                .unwrap()
                .bytes()
                .await
                .unwrap();
            assert_eq!(now, bytes, "{key} was reused, not rewritten");
        }
        assert_consistent(b.store.clone(), DIM, L2).await;
    }
}

#[tokio::test]
async fn index_objects_are_listed_beside_their_data() {
    let store = common::memory();
    let ns = two_segments(store.clone(), 30).await;
    ns.index(&every()).await.unwrap();
    let prefix = Path::from(format!("docs/segments/{}", seg(1, 1)));
    let mut names: Vec<String> = store
        .list(Some(&prefix))
        .map_ok(|m| m.location.filename().unwrap().to_string())
        .try_collect()
        .await
        .unwrap();
    names.sort();
    assert_eq!(names, ["data.bin", "generation", "index-1.bin"]);
}

/// A manifest can name an index kind this build does not read (another build
/// may add kinds over time). That segment is searched flat, with the exact
/// answer, and is never an error.
#[tokio::test]
async fn a_segment_whose_index_kind_is_unknown_is_searched_flat() {
    for backend in cas_backends() {
        let store = backend.store.clone();
        let ns = two_segments(store.clone(), 300).await;
        ns.index(&every()).await.unwrap();
        let probe = Lcg(7).vector(DIM);
        let exact = ns
            .query_with(&probe, 20, &QueryOptions::default().exact(true))
            .await
            .unwrap();

        let path = manifest::manifest_path("docs");
        let raw = store.get(&path).await.unwrap().bytes().await.unwrap();
        let mut json: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        let indexes = json["indexes"].as_object_mut().unwrap();
        assert_eq!(indexes.len(), 2, "{}", backend.name);
        for entry in indexes.values_mut() {
            entry["kind"] = serde_json::Value::from("graph");
        }
        store
            .put(&path, serde_json::to_vec(&json).unwrap().into())
            .await
            .unwrap();

        let reopened = Namespace::open(store.clone(), "docs").await.unwrap();
        let hits = reopened
            .query_with(&probe, 20, &QueryOptions::default())
            .await
            .unwrap();
        assert_eq!(hits, exact, "{}", backend.name);
    }
}
