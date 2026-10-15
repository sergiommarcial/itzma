//! Garbage collection (phase A), and what readers and compactors do
//! when a segment they planned to read has been deleted.

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::faulty::{Fault, FaultyStore, PutKind};
use common::{
    Backend, Snapshot, assert_consistent, backends, cas_backends, doc, publish, put_segment,
    schema, snapshot,
};
use futures::TryStreamExt;
use itzma_core::{
    Compaction, CompactionOptions, GcOptions, GcReport, Hit, ItzmaError, Manifest, Metric,
    Namespace, WalOp, manifest, segment,
};
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt};

const ROOMY: CompactionOptions = CompactionOptions {
    max_segments: 8,
    max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
};
const TIGHT: CompactionOptions = CompactionOptions {
    max_segments: 1,
    max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
};

fn ids(hits: &[Hit]) -> Vec<&str> {
    hits.iter().map(|h| h.id.as_str()).collect()
}

/// A namespace whose two writes are each folded into their own segment, so
/// the manifest lists `1-1` and `2-2`.
async fn two_segments(store: Arc<dyn ObjectStore>) -> Namespace {
    let ns = Namespace::create(store, "docs", schema(1, Metric::L2))
        .await
        .unwrap();
    for (id, x) in [("a", 1.0), ("b", 2.0)] {
        ns.upsert(vec![doc(id, &[x])]).await.unwrap();
        ns.compact(&ROOMY).await.unwrap();
    }
    ns
}

/// What another process does between a reader's manifest read and its segment
/// read: merge the two segments into one and delete the old ones, as a merge
/// followed by a garbage collection would.
async fn merge_and_delete_old(store: Arc<dyn ObjectStore>) {
    let ns = Namespace::open(store.clone(), "docs").await.unwrap();
    let merged = ns.compact(&TIGHT).await.unwrap();
    assert!(matches!(
        merged,
        Compaction::Published {
            merged_segments: 2,
            ..
        }
    ));
    for id in [segment::segment_id(1, 1), segment::segment_id(2, 2)] {
        store
            .delete(&segment::segment_path("docs", &id))
            .await
            .unwrap();
    }
}

#[tokio::test]
async fn a_reader_that_loses_its_segments_to_a_merge_reads_again() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let faulty = FaultyStore::new(b.store.clone());
        let store: Arc<dyn ObjectStore> = faulty.clone();
        let ns = two_segments(store).await;

        faulty.before_get(
            &segment::segment_id(1, 1),
            merge_and_delete_old(b.store.clone()),
        );
        let hits = ns.query(&[0.0], 10).await.unwrap();
        assert_eq!(ids(&hits), ["a", "b"]);
    }
}

#[tokio::test]
async fn a_reader_gives_up_after_three_attempts() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let faulty = FaultyStore::new(b.store.clone());
        let store: Arc<dyn ObjectStore> = faulty.clone();
        let ns = Namespace::create(store, "docs", schema(1, Metric::L2))
            .await
            .unwrap();

        // Manifests that list a base segment nobody wrote: 1-1, and each time
        // the reader looks for one, another process moves the manifest on to
        // the next, so the reader is always behind.
        let id = |hi: u64| segment::segment_id(1, hi);
        publish(&*b.store, "docs", &[id(1).as_str()], 1).await;
        for hi in 1..=3u64 {
            let next = id(hi + 1);
            let raw = b.store.clone();
            faulty.before_get(&id(hi), async move {
                publish(&*raw, "docs", &[next.as_str()], hi + 1).await;
            });
        }

        match ns.query(&[0.0], 1).await.unwrap_err() {
            ItzmaError::CorruptSegment { path, .. } => {
                assert!(path.contains(&id(3)), "the third attempt failed: {path}");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}

#[tokio::test]
async fn a_merge_that_loses_its_segments_to_another_merge_is_superseded() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let faulty = FaultyStore::new(b.store.clone());
        let store: Arc<dyn ObjectStore> = faulty.clone();
        let ns = two_segments(store).await;

        faulty.before_get(
            &segment::segment_id(1, 1),
            merge_and_delete_old(b.store.clone()),
        );
        assert_eq!(ns.compact(&TIGHT).await.unwrap(), Compaction::Superseded);

        let (m, _) = manifest::load(&*b.store, "docs").await.unwrap();
        assert_eq!(m.live_segments, [segment::segment_id(1, 2)]);
    }
}

#[tokio::test]
async fn a_compactor_whose_orphan_is_deleted_under_it_is_superseded() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let faulty = FaultyStore::new(b.store.clone());
        let store: Arc<dyn ObjectStore> = faulty.clone();
        let ns = Namespace::create(store, "docs", schema(1, Metric::L2))
            .await
            .unwrap();
        ns.upsert(vec![doc("a", &[1.0])]).await.unwrap();
        ns.upsert(vec![doc("b", &[2.0])]).await.unwrap();

        // A first run writes segment 1-2 and dies before it publishes.
        faulty.fail_puts("manifest", PutKind::Update, Fault::Generic, Some(1));
        assert!(ns.compact(&ROOMY).await.is_err());
        faulty.clear();
        let orphan = segment::segment_id(1, 2);
        assert!(
            b.store
                .get(&segment::segment_path("docs", &orphan))
                .await
                .is_ok()
        );

        // A second run plans the same fold. Before it checks the orphan's
        // bytes, another process publishes a fold past it (so the orphan is
        // dead) and the orphan is deleted.
        let raw = b.store.clone();
        let orphan_path = segment::segment_path("docs", &orphan);
        faulty.before_get(&orphan, async move {
            let other = Namespace::open(raw.clone(), "docs").await.unwrap();
            other.upsert(vec![doc("c", &[3.0])]).await.unwrap();
            other.compact(&ROOMY).await.unwrap();
            raw.delete(&orphan_path).await.unwrap();
        });
        assert_eq!(ns.compact(&ROOMY).await.unwrap(), Compaction::Superseded);

        let hits = ns.query(&[0.0], 10).await.unwrap();
        assert_eq!(ids(&hits), ["a", "b", "c"]);
    }
}
const DIM: usize = 2;
const L2: Metric = Metric::L2;

/// Collect anything, however young.
const NOW: GcOptions = GcOptions {
    min_age: Duration::ZERO,
    dry_run: false,
    fence_wal: false,
};

fn dynamic(store: &Arc<FaultyStore>) -> Arc<dyn ObjectStore> {
    store.clone()
}

/// A fresh store on backend `ix` of `cas_backends()`, wrapped so faults can be
/// armed on it later.
fn fresh(ix: usize) -> Arc<FaultyStore> {
    FaultyStore::new(cas_backends().swap_remove(ix).store)
}

fn seg(lo: u64, hi: u64) -> String {
    segment::segment_id(lo, hi)
}

/// Three writes, each folded into its own segment, then a merge into `1-3`.
/// The manifest lists only `1-3`, and `1-1`, `2-2` and `3-3` are dead.
async fn prepare(store: Arc<dyn ObjectStore>) -> Namespace {
    let ns = Namespace::create(store, "docs", schema(DIM, L2))
        .await
        .unwrap();
    for (id, x) in [("a", 1.0), ("b", 2.0), ("c", 3.0)] {
        ns.upsert(vec![doc(id, &[x, 0.0])]).await.unwrap();
        ns.compact(&ROOMY).await.unwrap();
    }
    let merged = ns
        .compact(&CompactionOptions {
            max_segments: 2,
            max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
        })
        .await
        .unwrap();
    assert!(matches!(
        merged,
        Compaction::Published {
            merged_segments: 3,
            ..
        }
    ));
    ns
}

const DEAD: [(u64, u64); 3] = [(1, 1), (2, 2), (3, 3)];

/// The segment directories in the store, sorted.
async fn segment_ids(store: &dyn ObjectStore) -> Vec<String> {
    let prefix = Path::from("docs/segments");
    let listing: Vec<_> = store.list(Some(&prefix)).try_collect().await.unwrap();
    let mut ids: Vec<String> = listing
        .into_iter()
        .filter_map(|meta| {
            let mut parts = meta.location.prefix_match(&prefix)?;
            parts.next().map(|part| part.as_ref().to_string())
        })
        .collect();
    ids.sort();
    ids.dedup();
    ids
}

/// Compared with `before`, only objects of the three dead segments may be gone,
/// and nothing new or changed may appear.
async fn assert_only_dead_segments_gone(store: &dyn ObjectStore, before: &Snapshot) {
    let after = snapshot(store).await;
    let dead: Vec<String> = DEAD.iter().map(|(lo, hi)| seg(*lo, *hi)).collect();
    for (key, bytes) in before {
        if after.get(key) == Some(bytes) {
            continue;
        }
        assert!(
            dead.iter().any(|id| key.contains(id.as_str())),
            "{key} changed or vanished"
        );
    }
    for key in after.keys() {
        assert!(before.contains_key(key), "gc created {key}");
    }
}

async fn size_of_dead(before: &Snapshot) -> u64 {
    let dead: Vec<String> = DEAD.iter().map(|(lo, hi)| seg(*lo, *hi)).collect();
    before
        .iter()
        .filter(|(key, _)| dead.iter().any(|id| key.contains(id.as_str())))
        .map(|(_, bytes)| bytes.len() as u64)
        .sum()
}

#[tokio::test]
async fn gc_deletes_superseded_segments_and_nothing_else() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = prepare(b.store.clone()).await;
        let before = snapshot(&*b.store).await;

        let report = ns.gc(&NOW).await.unwrap();
        assert_eq!(
            report,
            GcReport {
                segments_deleted: 3,
                bytes_freed: size_of_dead(&before).await,
                segments_too_young: 0,
                ..GcReport::default()
            }
        );
        assert_eq!(segment_ids(&*b.store).await, [seg(1, 3)]);
        assert_only_dead_segments_gone(&*b.store, &before).await;

        let docs = assert_consistent(b.store.clone(), DIM, L2).await;
        assert_eq!(docs.len(), 3);
    }
}

#[tokio::test]
async fn gc_deletes_every_superseded_segment_of_a_multi_output_merge() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", schema(DIM, L2))
            .await
            .unwrap();
        for i in 0..6u32 {
            let docs: Vec<_> = (0..20u32)
                .map(|j| doc(&format!("g{i}-{j:03}"), &[i as f32, j as f32]))
                .collect();
            ns.upsert(docs).await.unwrap();
            ns.compact(&CompactionOptions {
                max_segments: 8,
                max_segment_bytes: 2_000,
            })
            .await
            .unwrap();
        }
        let (before, _) = manifest::load(&*b.store, "docs").await.unwrap();
        let before = before.live_segments;

        let outcome = ns
            .compact(&CompactionOptions {
                max_segments: 1,
                max_segment_bytes: 2_000,
            })
            .await
            .unwrap();
        let Compaction::Published {
            segments: after, ..
        } = outcome
        else {
            panic!("expected Published, got {outcome:?}");
        };
        assert!(!after.is_empty());

        let freed = ns.gc(&NOW).await.unwrap();
        assert_eq!(freed.segments_deleted, before.len());
        for id in &before {
            assert!(
                b.store
                    .get(&segment::segment_path("docs", id))
                    .await
                    .is_err(),
                "superseded segment {id} should be gone"
            );
        }
        for id in &after {
            assert!(
                b.store
                    .get(&segment::segment_path("docs", id))
                    .await
                    .is_ok(),
                "live segment {id} should still be there"
            );
        }
    }
}

#[tokio::test]
async fn a_second_run_finds_nothing() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = prepare(b.store.clone()).await;
        ns.gc(&NOW).await.unwrap();
        assert_eq!(ns.gc(&NOW).await.unwrap(), GcReport::default());
    }
}

#[tokio::test]
async fn a_dry_run_deletes_nothing_and_reports_what_a_real_run_would() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = prepare(b.store.clone()).await;
        let before = snapshot(&*b.store).await;

        let dry = ns
            .gc(&GcOptions {
                dry_run: true,
                ..NOW
            })
            .await
            .unwrap();
        assert_eq!(snapshot(&*b.store).await, before);

        assert_eq!(ns.gc(&NOW).await.unwrap(), dry);
        assert_eq!(dry.segments_deleted, 3);
    }
}

#[tokio::test]
async fn the_default_min_age_leaves_fresh_garbage_alone() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = prepare(b.store.clone()).await;
        let before = snapshot(&*b.store).await;

        let report = ns.gc(&GcOptions::default()).await.unwrap();
        assert_eq!(
            report,
            GcReport {
                segments_deleted: 0,
                bytes_freed: 0,
                segments_too_young: 3,
                ..GcReport::default()
            }
        );
        assert_eq!(snapshot(&*b.store).await, before);
    }
}

/// `min_age` against a real clock. Every other test uses a zero age or the
/// 24 hour default, so none of them sits between "too young" and "old enough".
/// The store stamps each object when it is written, and S3 keeps whole
/// seconds, so the wait is a full second longer than the age.
const AGE: Duration = Duration::from_secs(3);
const WAIT: Duration = Duration::from_secs(4);

/// Three deltas are written, wait, and only then merged. The deltas are old
/// but the segment that replaces them is not, so they stay until it has aged.
async fn dead_segments_wait_for_their_replacement_to_age(b: Backend) {
    eprintln!("backend: {}", b.name);
    let ns = Namespace::create(b.store.clone(), "docs", schema(DIM, L2))
        .await
        .unwrap();
    for (id, x) in [("a", 1.0), ("b", 2.0), ("c", 3.0)] {
        ns.upsert(vec![doc(id, &[x, 0.0])]).await.unwrap();
        ns.compact(&ROOMY).await.unwrap();
    }
    tokio::time::sleep(WAIT).await;
    let merged = ns
        .compact(&CompactionOptions {
            max_segments: 2,
            max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
        })
        .await
        .unwrap();
    assert!(matches!(merged, Compaction::Published { .. }), "{merged:?}");

    let options = GcOptions {
        min_age: AGE,
        ..NOW
    };
    let before = snapshot(&*b.store).await;
    let report = ns.gc(&options).await.unwrap();
    assert_eq!(
        report,
        GcReport {
            segments_too_young: 3,
            ..GcReport::default()
        }
    );
    assert_eq!(snapshot(&*b.store).await, before);

    tokio::time::sleep(WAIT).await;
    let report = ns.gc(&options).await.unwrap();
    assert_eq!(report.segments_deleted, 3);
    assert_eq!(report.segments_too_young, 0);
    assert_eq!(segment_ids(&*b.store).await, [seg(1, 3)]);
    assert_eq!(
        ids(&ns.query(&[0.0, 0.0], 10).await.unwrap()),
        ["a", "b", "c"]
    );
    assert_consistent(b.store.clone(), DIM, L2).await;
}

#[tokio::test]
async fn garbage_is_kept_until_it_and_its_replacement_are_old_enough() {
    futures::future::join_all(
        cas_backends()
            .into_iter()
            .map(dead_segments_wait_for_their_replacement_to_age),
    )
    .await;
}

#[tokio::test]
async fn orphans_are_judged_by_the_fold_point() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = prepare(b.store.clone()).await;
        // Folded through 3, live 1-3. 2-3 is an orphan at the fold point and
        // is dead. 4-5 is past it and a later fold may reuse it. The two
        // directories below are not segments at all.
        put_segment(
            &*b.store,
            "docs",
            2,
            3,
            DIM,
            vec![WalOp::Delete("x".into())],
        )
        .await;
        put_segment(
            &*b.store,
            "docs",
            4,
            5,
            DIM,
            vec![WalOp::Delete("y".into())],
        )
        .await;
        for stray in ["docs/segments/notes/data.bin", "docs/segments/loose-file"] {
            b.store
                .put(&Path::from(stray), "keep me".into())
                .await
                .unwrap();
        }

        let report = ns.gc(&NOW).await.unwrap();
        assert_eq!(report.segments_deleted, 4, "1-1, 2-2, 3-3 and 2-3");
        assert_eq!(
            segment_ids(&*b.store).await,
            [seg(1, 3), seg(4, 5), "loose-file".into(), "notes".into()]
        );
        assert_consistent(b.store.clone(), DIM, L2).await;
    }
}

#[tokio::test]
async fn an_orphan_a_merge_over_an_empty_tail_would_reuse_survives_and_is_reused() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let faulty = FaultyStore::new(b.store.clone());
        let ns = two_segments(dynamic(&faulty)).await;

        // The merge writes 1-2, the same id as the range the manifest already
        // covers, and dies before it publishes.
        faulty.fail_puts("manifest", PutKind::Update, Fault::Generic, Some(1));
        assert!(ns.compact(&TIGHT).await.is_err());
        faulty.clear();
        assert_eq!(
            segment_ids(&*b.store).await,
            [seg(1, 1), seg(1, 2), seg(2, 2)]
        );

        // Nothing is dead: 1-1 and 2-2 are live and 1-2 is spoken for.
        assert_eq!(ns.gc(&NOW).await.unwrap(), GcReport::default());
        assert_eq!(segment_ids(&*b.store).await.len(), 3);

        // The rerun reuses the orphan (its bytes are checked) and publishes.
        let outcome = ns.compact(&TIGHT).await.unwrap();
        assert!(
            matches!(outcome, Compaction::Published { ref segments, .. } if segments == &[seg(1, 2)])
        );
        assert_eq!(ns.gc(&NOW).await.unwrap().segments_deleted, 2);
        assert_eq!(segment_ids(&*b.store).await, [seg(1, 2)]);
    }
}

/// Overwrite the manifest with a plain put, which every backend has. It is how
/// the tests reach a state on `LocalFileSystem`, which has no conditional
/// update.
async fn overwrite_manifest(store: &dyn ObjectStore, live: &[&str], folded: u64) {
    let mut m = Manifest::new(schema(1, L2));
    m.live_segments = live.iter().map(|id| id.to_string()).collect();
    m.wal_folded_through = folded;
    store
        .put(
            &manifest::manifest_path("docs"),
            serde_json::to_vec(&m).unwrap().into(),
        )
        .await
        .unwrap();
}

#[tokio::test]
async fn gc_runs_on_a_store_without_conditional_update() {
    for b in backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", schema(1, L2))
            .await
            .unwrap();
        let up = |id: &str, x: f32| WalOp::Upsert(doc(id, &[x]));
        put_segment(&*b.store, "docs", 1, 1, 1, vec![up("a", 1.0)]).await;
        let merged =
            put_segment(&*b.store, "docs", 1, 2, 1, vec![up("a", 1.0), up("b", 2.0)]).await;
        overwrite_manifest(&*b.store, &[merged.as_str()], 2).await;

        let report = ns.gc(&NOW).await.unwrap();
        assert_eq!(report.segments_deleted, 1);
        assert_eq!(segment_ids(&*b.store).await, [merged]);
    }
}

#[tokio::test]
async fn a_namespace_with_no_segments_has_nothing_to_collect() {
    for b in backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", schema(1, L2))
            .await
            .unwrap();
        ns.upsert(vec![doc("a", &[1.0])]).await.unwrap();
        assert_eq!(ns.gc(&NOW).await.unwrap(), GcReport::default());
    }
}

#[tokio::test]
async fn gc_on_a_damaged_manifest_fails_before_it_deletes_anything() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = prepare(b.store.clone()).await;
        // The last live segment does not reach the fold point.
        overwrite_manifest(&*b.store, &[seg(1, 2).as_str()], 3).await;
        let before = snapshot(&*b.store).await;
        let err = ns.gc(&NOW).await.unwrap_err();
        assert!(matches!(err, ItzmaError::CorruptManifest(_)), "{err:?}");
        assert_eq!(snapshot(&*b.store).await, before);
    }
}

/// Several collectors at once, every store call yielding so their deletes
/// interleave. All succeed, and the end state is the same as after one.
#[tokio::test]
async fn collectors_running_together_agree() {
    for ix in 0..cas_backends().len() {
        let store = fresh(ix);
        let ns = prepare(dynamic(&store)).await;
        store.yield_each_op(true);
        drop(ns);

        let mut tasks = Vec::new();
        for _ in 0..3 {
            let store = dynamic(&store);
            tasks.push(tokio::spawn(async move {
                let ns = Namespace::open(store, "docs").await.unwrap();
                ns.gc(&NOW).await.unwrap()
            }));
        }
        let mut counted = 0;
        for task in tasks {
            counted += task.await.unwrap().segments_deleted;
        }
        assert!(
            counted >= 3,
            "someone must have deleted all three: {counted}"
        );

        store.yield_each_op(false);
        assert_eq!(segment_ids(&*store).await, [seg(1, 3)]);
        assert_consistent(dynamic(&store), DIM, L2).await;
    }
}

/// Run `gc` with a fault at every store operation it makes, once as a plain
/// error and once as a lost ack. After each: with faults off and a new handle
/// (a restart), the store is consistent, only dead segments have gone, and a
/// rerun finishes the job and then finds nothing more to do.
#[tokio::test]
async fn gc_survives_a_fault_at_every_step() {
    for ix in 0..cas_backends().len() {
        let store = fresh(ix);
        let ns = prepare(dynamic(&store)).await;
        store.reset_ops();
        ns.gc(&NOW).await.unwrap();
        let total = store.ops();
        assert!(
            total >= 5,
            "the manifest, a listing and three deletes: {total}"
        );

        for fault in [Fault::Generic, Fault::LostAck] {
            for n in 0..total {
                eprintln!("backend {ix}: {fault:?} at operation {n} of {total}");
                let store = fresh(ix);
                let ns = prepare(dynamic(&store)).await;
                let before = snapshot(&*store).await;

                store.fail_nth_op(n, fault);
                let _ = ns.gc(&NOW).await;
                store.clear();
                drop(ns);

                assert_eq!(assert_consistent(dynamic(&store), DIM, L2).await.len(), 3);
                assert_only_dead_segments_gone(&*store, &before).await;

                let ns = Namespace::open(dynamic(&store), "docs").await.unwrap();
                ns.gc(&NOW).await.unwrap();
                assert_eq!(segment_ids(&*store).await, [seg(1, 3)]);
                assert_only_dead_segments_gone(&*store, &before).await;
                assert_eq!(ns.gc(&NOW).await.unwrap(), GcReport::default());
                assert_consistent(dynamic(&store), DIM, L2).await;
            }
        }
    }
}

/// Three folds are live. One task merges them, another collects repeatedly,
/// and readers query in a loop, every store call yielding so the steps
/// interleave. The merge is the only thing that moves the manifest, so a
/// reader can lose its segments at most once and every query must succeed with
/// all three documents.
#[tokio::test]
async fn readers_survive_a_merge_and_gc_running_beside_them() {
    for ix in 0..cas_backends().len() {
        let store = fresh(ix);
        let ns = Namespace::create(dynamic(&store), "docs", schema(DIM, L2))
            .await
            .unwrap();
        for (id, x) in [("a", 1.0), ("b", 2.0), ("c", 3.0)] {
            ns.upsert(vec![doc(id, &[x, 0.0])]).await.unwrap();
            ns.compact(&ROOMY).await.unwrap();
        }
        drop(ns);
        store.yield_each_op(true);

        let mut tasks = Vec::new();
        for _ in 0..3 {
            let store = dynamic(&store);
            tasks.push(tokio::spawn(async move {
                let ns = Namespace::open(store, "docs").await.unwrap();
                for _ in 0..15 {
                    let hits = ns.query(&[0.0, 0.0], 10).await.unwrap();
                    assert_eq!(ids(&hits), ["a", "b", "c"]);
                }
            }));
        }
        let merger = dynamic(&store);
        tasks.push(tokio::spawn(async move {
            let ns = Namespace::open(merger, "docs").await.unwrap();
            ns.compact(&CompactionOptions {
                max_segments: 2,
                max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
            })
            .await
            .unwrap();
        }));
        let collector = dynamic(&store);
        tasks.push(tokio::spawn(async move {
            let ns = Namespace::open(collector, "docs").await.unwrap();
            for _ in 0..5 {
                ns.gc(&NOW).await.unwrap();
            }
        }));
        for task in tasks {
            task.await.unwrap();
        }

        store.yield_each_op(false);
        let ns = Namespace::open(dynamic(&store), "docs").await.unwrap();
        ns.gc(&NOW).await.unwrap();
        assert_eq!(segment_ids(&*store).await, [seg(1, 3)]);
        assert_consistent(dynamic(&store), DIM, L2).await;
    }
}

#[tokio::test]
async fn a_merge_whose_input_is_collected_while_it_reads_is_superseded() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let faulty = FaultyStore::new(b.store.clone());
        let store: Arc<dyn ObjectStore> = faulty.clone();
        let ns = Namespace::create(store, "docs", schema(1, Metric::L2))
            .await
            .unwrap();
        for (n, id) in ["a", "b"].into_iter().enumerate() {
            let mut d = doc(id, &[n as f32]);
            d.sparse = Some(vec![(1, 1.0)]);
            ns.upsert(vec![d]).await.unwrap();
            ns.compact(&ROOMY).await.unwrap();
        }

        // Another merge publishes and 1-1 is collected while this run reads
        // its inputs. These segments are small, so they are read whole and
        // the read is already done: this run's publish finds the manifest
        // moved. The next test covers a segment read in windows.
        let raw = b.store.clone();
        faulty.before_get(
            &format!("{}/sparse.bin", segment::segment_id(1, 1)),
            async move {
                let other = Namespace::open(raw.clone(), "docs").await.unwrap();
                assert!(matches!(
                    other.compact(&TIGHT).await.unwrap(),
                    Compaction::Published { .. }
                ));
                let one = segment::segment_path("docs", &segment::segment_id(1, 1));
                raw.delete(&one).await.unwrap();
            },
        );
        assert_eq!(ns.compact(&TIGHT).await.unwrap(), Compaction::Superseded);
    }
}

#[tokio::test]
async fn a_segment_collected_before_its_first_window_is_a_stale_read() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let faulty = FaultyStore::new(b.store.clone());
        let store: Arc<dyn ObjectStore> = faulty.clone();
        // Segments over 1 MiB, so the merge reads them in windows.
        let dim = 256;
        let ns = Namespace::create(store, "docs", schema(dim, Metric::L2))
            .await
            .unwrap();
        for s in 0..2u32 {
            let docs = (0..1200u32)
                .map(|i| {
                    let mut d = doc(&format!("s{s}-{i:05}"), &vec![i as f32; dim]);
                    d.sparse = Some(vec![(i, 1.0)]);
                    d
                })
                .collect();
            ns.upsert(docs).await.unwrap();
            ns.compact(&ROOMY).await.unwrap();
        }

        // 1-1's sidecar is opened after its data object and before its first
        // window: the merge and collection land in between, so the window's
        // read finds the object gone.
        let raw = b.store.clone();
        faulty.before_get(
            &format!("{}/sparse.bin", segment::segment_id(1, 1)),
            async move {
                let other = Namespace::open(raw.clone(), "docs").await.unwrap();
                assert!(matches!(
                    other.compact(&TIGHT).await.unwrap(),
                    Compaction::Published { .. }
                ));
                let one = segment::segment_path("docs", &segment::segment_id(1, 1));
                raw.delete(&one).await.unwrap();
            },
        );
        assert_eq!(ns.compact(&TIGHT).await.unwrap(), Compaction::Superseded);
    }
}
