//! `Namespace::maintain`: the decision, the order of the two halves,
//! and what happens with many callers or a fault part way.

mod common;

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use common::faulty::{Fault, FaultyStore};
use common::{assert_consistent, cas_backends, doc, schema, snapshot, wal_listing};
use futures::TryStreamExt;
use itzma_core::{
    CompactTrigger, Compaction, CompactionOptions, GcOptions, GcReport, Hit, ItzmaError,
    MaintenanceOptions, Metric, Namespace, manifest, segment,
};
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt};

const DIM: usize = 2;
const L2: Metric = Metric::L2;
const ROOMY: CompactionOptions = CompactionOptions {
    max_segments: 8,
    max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
};
const TIGHT: CompactionOptions = CompactionOptions {
    max_segments: 1,
    max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
};
const QUERY: [f32; DIM] = [0.0, 0.0];

/// Collect anything, however young.
const NOW: GcOptions = GcOptions {
    min_age: Duration::ZERO,
    dry_run: false,
    fence_wal: false,
};
const FENCING: GcOptions = GcOptions {
    fence_wal: true,
    ..NOW
};

/// Every trigger off.
const NEVER: CompactTrigger = CompactTrigger {
    tail_objects: usize::MAX,
    tail_bytes: u64::MAX,
    tail_age: None,
};

fn options(trigger: CompactTrigger, gc: Option<GcOptions>) -> MaintenanceOptions {
    MaintenanceOptions {
        compact_when: trigger,
        compaction: ROOMY,
        gc,
    }
}

fn dynamic(store: &Arc<FaultyStore>) -> Arc<dyn ObjectStore> {
    store.clone()
}

async fn create(store: Arc<dyn ObjectStore>) -> Namespace {
    Namespace::create(store, "docs", schema(DIM, L2))
        .await
        .unwrap()
}

async fn write_one(ns: &Namespace, i: usize) {
    ns.upsert(vec![doc(&format!("d{i}"), &[i as f32, 0.0])])
        .await
        .unwrap();
}

async fn write(ns: &Namespace, n: usize) {
    for i in 0..n {
        write_one(ns, i).await;
    }
}

/// `n` writes, each folded into its own segment.
async fn deltas(ns: &Namespace, n: usize) {
    for i in 0..n {
        write_one(ns, i).await;
        ns.compact(&ROOMY).await.unwrap();
    }
}

async fn hits(ns: &Namespace) -> Vec<Hit> {
    ns.query(&QUERY, 1_000).await.unwrap()
}

async fn segment_ids(store: &dyn ObjectStore) -> Vec<String> {
    let prefix = Path::from("docs/segments");
    let listing: Vec<_> = store.list(Some(&prefix)).try_collect().await.unwrap();
    let mut ids: Vec<String> = listing
        .iter()
        .filter_map(|m| m.location.prefix_match(&prefix)?.next())
        .map(|part| part.as_ref().to_string())
        .collect();
    ids.dedup();
    ids
}

async fn tail_bytes(store: &dyn ObjectStore) -> u64 {
    wal_listing(store).await.iter().map(|(_, size)| size).sum()
}

fn published(compaction: &Option<Compaction>) -> bool {
    matches!(compaction, Some(Compaction::Published { .. }))
}

#[tokio::test]
async fn nothing_is_due_under_every_threshold() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = create(b.store.clone()).await;
        write(&ns, 3).await;
        let before = snapshot(&*b.store).await;

        let report = ns.maintain(&MaintenanceOptions::default()).await.unwrap();
        assert_eq!(report.tail_objects, 3);
        assert_eq!(report.tail_bytes, tail_bytes(&*b.store).await);
        assert_eq!(report.compaction, None);
        assert_eq!(report.gc, Some(GcReport::default()));
        assert_eq!(snapshot(&*b.store).await, before);
    }
}

#[tokio::test]
async fn the_object_count_decides_at_its_edge() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = create(b.store.clone()).await;
        let at_three = options(
            CompactTrigger {
                tail_objects: 3,
                ..NEVER
            },
            None,
        );
        write(&ns, 2).await;
        assert_eq!(ns.maintain(&at_three).await.unwrap().compaction, None);
        write_one(&ns, 2).await;
        let report = ns.maintain(&at_three).await.unwrap();
        assert!(
            matches!(
                report.compaction,
                Some(Compaction::Published {
                    folded_through: 3,
                    ..
                })
            ),
            "{report:?}"
        );
    }
}

#[tokio::test]
async fn the_byte_count_decides_at_its_edge() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = create(b.store.clone()).await;
        write(&ns, 2).await;
        let total = tail_bytes(&*b.store).await;

        let over = options(
            CompactTrigger {
                tail_bytes: total + 1,
                ..NEVER
            },
            None,
        );
        assert_eq!(ns.maintain(&over).await.unwrap().compaction, None);
        let at = options(
            CompactTrigger {
                tail_bytes: total,
                ..NEVER
            },
            None,
        );
        assert!(published(&ns.maintain(&at).await.unwrap().compaction));
    }
}

/// The age is the store's write time against the clock, so this waits. The wait
/// is a full second past the age because S3 keeps whole seconds.
async fn a_slow_tail_folds_once_it_is_old_enough(b: common::Backend) {
    eprintln!("backend: {}", b.name);
    let ns = create(b.store.clone()).await;
    write(&ns, 2).await;
    let aged = options(
        CompactTrigger {
            tail_age: Some(Duration::from_secs(3)),
            ..NEVER
        },
        None,
    );
    assert_eq!(ns.maintain(&aged).await.unwrap().compaction, None);

    tokio::time::sleep(Duration::from_secs(4)).await;
    assert!(published(&ns.maintain(&aged).await.unwrap().compaction));
    assert_consistent(b.store.clone(), DIM, L2).await;
}

#[tokio::test]
async fn the_age_decides_on_the_real_clock() {
    futures::future::join_all(
        cas_backends()
            .into_iter()
            .map(a_slow_tail_folds_once_it_is_old_enough),
    )
    .await;
}

#[tokio::test]
async fn too_many_segments_are_merged_with_no_tail_at_all() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = create(b.store.clone()).await;
        deltas(&ns, 3).await;
        let expected = hits(&ns).await;
        let merge_over_two = MaintenanceOptions {
            compaction: CompactionOptions {
                max_segments: 2,
                max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
            },
            ..options(NEVER, None)
        };

        let report = ns.maintain(&merge_over_two).await.unwrap();
        assert_eq!(report.tail_objects, 0);
        assert!(
            matches!(
                report.compaction,
                Some(Compaction::Published {
                    merged_segments: 3,
                    ..
                })
            ),
            "{report:?}"
        );
        assert_eq!(hits(&ns).await, expected);

        // One segment is within the limit, and nothing is due.
        assert_eq!(ns.maintain(&merge_over_two).await.unwrap().compaction, None);
    }
}

#[tokio::test]
async fn an_empty_tail_with_few_segments_changes_nothing() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = create(b.store.clone()).await;
        deltas(&ns, 2).await;
        let before = snapshot(&*b.store).await;
        let zero = CompactTrigger {
            tail_objects: 0,
            tail_bytes: 0,
            tail_age: Some(Duration::ZERO),
        };

        let report = ns.maintain(&options(zero, Some(NOW))).await.unwrap();
        assert_eq!(report.compaction, None, "even at zero, an empty tail waits");
        assert_eq!(snapshot(&*b.store).await, before);
    }
}

/// The merge makes the deltas dead, and the collection in the same call takes
/// them, which it could only do if it ran second.
#[tokio::test]
async fn the_collection_runs_after_the_compaction() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = create(b.store.clone()).await;
        deltas(&ns, 3).await;
        let merge_over_two = MaintenanceOptions {
            compaction: CompactionOptions {
                max_segments: 2,
                max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
            },
            ..options(NEVER, Some(NOW))
        };

        let report = ns.maintain(&merge_over_two).await.unwrap();
        assert!(published(&report.compaction), "{report:?}");
        assert_eq!(report.gc.unwrap().segments_deleted, 3);
        assert_eq!(segment_ids(&*b.store).await, [segment::segment_id(1, 3)]);
        assert_consistent(b.store.clone(), DIM, L2).await;
    }
}

#[tokio::test]
async fn the_collection_runs_when_no_compaction_is_due() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = create(b.store.clone()).await;
        deltas(&ns, 3).await;
        ns.compact(&TIGHT).await.unwrap();

        let report = ns.maintain(&options(NEVER, Some(NOW))).await.unwrap();
        assert_eq!(report.compaction, None);
        assert_eq!(report.gc.unwrap().segments_deleted, 3);
    }
}

#[tokio::test]
async fn no_gc_options_means_no_collection() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = create(b.store.clone()).await;
        deltas(&ns, 3).await;
        ns.compact(&TIGHT).await.unwrap();
        let before = snapshot(&*b.store).await;

        let report = ns.maintain(&options(NEVER, None)).await.unwrap();
        assert_eq!(report.gc, None);
        assert_eq!(snapshot(&*b.store).await, before);
    }
}

#[tokio::test]
async fn a_dry_run_and_fencing_reach_the_collection() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = create(b.store.clone()).await;
        deltas(&ns, 3).await;
        let before = snapshot(&*b.store).await;

        let dry = GcOptions {
            dry_run: true,
            ..FENCING
        };
        let report = ns.maintain(&options(NEVER, Some(dry))).await.unwrap();
        assert_eq!(report.gc.unwrap().wal_objects_fenced, 3);
        assert_eq!(snapshot(&*b.store).await, before);

        let report = ns.maintain(&options(NEVER, Some(FENCING))).await.unwrap();
        assert_eq!(report.gc.unwrap().wal_objects_fenced, 3);
        assert_eq!(hits(&ns).await.len(), 3);
        assert_consistent(b.store.clone(), DIM, L2).await;
    }
}

#[tokio::test]
async fn a_second_call_straight_after_finds_nothing_to_do() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = create(b.store.clone()).await;
        write(&ns, 4).await;
        let every_write = options(
            CompactTrigger {
                tail_objects: 1,
                ..NEVER
            },
            Some(NOW),
        );

        let first = ns.maintain(&every_write).await.unwrap();
        assert!(published(&first.compaction));
        let settled = snapshot(&*b.store).await;

        let second = ns.maintain(&every_write).await.unwrap();
        assert_eq!(second.tail_objects, 0);
        assert_eq!(second.compaction, None);
        assert_eq!(second.gc, Some(GcReport::default()));
        assert_eq!(snapshot(&*b.store).await, settled);
    }
}

#[tokio::test]
async fn two_maintainers_at_once_leave_a_consistent_store() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let first = create(b.store.clone()).await;
        let second = Namespace::open(b.store.clone(), "docs").await.unwrap();
        write(&first, 5).await;
        let expected = hits(&first).await;
        let every_write = options(
            CompactTrigger {
                tail_objects: 1,
                ..NEVER
            },
            Some(NOW),
        );

        let (a, c) = tokio::join!(first.maintain(&every_write), second.maintain(&every_write));
        let (a, c) = (a.unwrap(), c.unwrap());
        let wins = [&a, &c]
            .iter()
            .filter(|report| published(&report.compaction))
            .count();
        assert_eq!(wins, 1, "{a:?} {c:?}");

        assert_eq!(hits(&first).await, expected);
        assert_consistent(b.store.clone(), DIM, L2).await;
    }
}

/// A namespace above the gate is refused before anything is read or written.
#[tokio::test]
async fn a_namespace_above_the_gate_is_refused_and_left_alone() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = create(b.store.clone()).await;
        write(&ns, 3).await;
        let (mut m, _) = manifest::load(&*b.store, "docs").await.unwrap();
        m.wal_format = 99;
        b.store
            .put(
                &manifest::manifest_path("docs"),
                Bytes::from(serde_json::to_vec(&m).unwrap()).into(),
            )
            .await
            .unwrap();
        let before = snapshot(&*b.store).await;

        let err = ns.maintain(&MaintenanceOptions::default()).await.err();
        assert!(
            matches!(err, Some(ItzmaError::WalFormatTooNew { .. })),
            "{err:?}"
        );
        assert_eq!(snapshot(&*b.store).await, before);
    }
}

/// Fencing below level 4 is the collection's error, and it comes back as is.
#[tokio::test]
async fn the_collections_errors_pass_through() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = create(b.store.clone()).await;
        write(&ns, 2).await;
        manifest::update(&*b.store, "docs", |m| m.wal_format = 3)
            .await
            .unwrap();
        let before = snapshot(&*b.store).await;

        let err = ns.maintain(&options(NEVER, Some(FENCING))).await.err();
        assert!(
            matches!(err, Some(ItzmaError::WalFormatTooOld { have: 3, need: 4 })),
            "{err:?}"
        );
        assert_eq!(snapshot(&*b.store).await, before);
    }
}

/// Three deltas and a tail of two, with a policy that merges everything and then
/// fences it: a call that does all three kinds of work.
fn busy() -> MaintenanceOptions {
    MaintenanceOptions {
        compact_when: CompactTrigger {
            tail_objects: 1,
            ..NEVER
        },
        compaction: CompactionOptions {
            max_segments: 2,
            max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
        },
        gc: Some(FENCING),
    }
}

async fn busy_namespace(store: Arc<dyn ObjectStore>) -> Namespace {
    let ns = create(store).await;
    deltas(&ns, 3).await;
    for i in 3..5 {
        write_one(&ns, i).await;
    }
    ns
}

/// Fault every operation of a call that compacts, merges, collects and fences,
/// as a plain error and as a lost ack. After each, calling it again has to
/// finish, and the namespace has to answer as it did before.
#[tokio::test]
async fn a_stopped_call_is_finished_by_the_next_one() {
    for ix in 0..cas_backends().len() {
        let counted = FaultyStore::new(cas_backends().swap_remove(ix).store);
        let ns = busy_namespace(dynamic(&counted)).await;
        counted.reset_ops();
        let report = ns.maintain(&busy()).await.unwrap();
        assert!(published(&report.compaction), "{report:?}");
        let total = counted.ops();
        assert!(total > 10, "a call that does everything is many operations");

        for fault in [Fault::Generic, Fault::LostAck] {
            for n in 0..total {
                eprintln!("backend {ix}: {fault:?} at operation {n} of {total}");
                let store = FaultyStore::new(cas_backends().swap_remove(ix).store);
                let ns = busy_namespace(dynamic(&store)).await;
                let expected = hits(&ns).await;
                store.reset_ops();

                store.fail_nth_op(n, fault);
                let stopped = ns.maintain(&busy()).await;
                store.clear();
                assert!(stopped.is_err(), "a fault inside maintain must show");
                drop(ns);

                assert_consistent(dynamic(&store), DIM, L2).await;
                let ns = Namespace::open(dynamic(&store), "docs").await.unwrap();
                ns.maintain(&busy()).await.unwrap();
                assert_eq!(hits(&ns).await, expected);
                assert_consistent(dynamic(&store), DIM, L2).await;

                let settled = ns.maintain(&busy()).await.unwrap();
                assert_eq!(settled.compaction, None);
                assert_eq!(settled.gc, Some(GcReport::default()));
            }
        }
    }
}

#[tokio::test]
async fn the_maintain_future_can_be_spawned() {
    let store = common::memory();
    let ns = create(store).await;
    write(&ns, 2).await;
    let handle = tokio::spawn(async move { ns.maintain(&MaintenanceOptions::default()).await });
    assert!(handle.await.unwrap().is_ok());
}
