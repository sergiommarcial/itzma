//! Building and publishing the sparse inverted index:
//! `Namespace::index_sparse`, the manifest's `sparse_indexes`, and what
//! compaction does with them.

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::faulty::{Fault, FaultyStore};
use common::{assert_consistent, cas_backends, doc, schema};
use futures::TryStreamExt;
use itzma_core::{
    CompactionOptions, Document, GcOptions, IndexSparseOptions, Metric, Namespace,
    SparseQueryOptions, manifest, segment,
};
use object_store::{ObjectStore, ObjectStoreExt};

const TIGHT: CompactionOptions = CompactionOptions {
    max_segments: 1,
    max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
};

fn sparse_doc(id: &str, terms: &[(u32, f32)]) -> Document {
    Document {
        id: id.into(),
        vector: vec![1.0],
        attrs: None,
        sparse: Some(terms.to_vec()),
    }
}

async fn create(store: std::sync::Arc<dyn ObjectStore>) -> Namespace {
    Namespace::create(store, "docs", schema(1, Metric::L2))
        .await
        .expect("create")
}

#[tokio::test]
async fn a_segment_with_no_sparse_data_is_skipped() {
    for backend in cas_backends() {
        let ns = create(backend.store.clone()).await;
        ns.upsert(vec![doc("a", &[1.0])]).await.expect("upsert");
        ns.compact(&CompactionOptions::default())
            .await
            .expect("compact");

        let report = ns
            .index_sparse(&IndexSparseOptions::default())
            .await
            .expect("index_sparse");
        assert_eq!(
            (report.indexed, report.skipped_no_sparse),
            (0, 1),
            "{}",
            backend.name
        );
    }
}

#[tokio::test]
async fn a_segment_with_sparse_data_is_indexed_and_published() {
    for backend in cas_backends() {
        let ns = create(backend.store.clone()).await;
        ns.upsert(vec![
            sparse_doc("a", &[(3, 0.5)]),
            sparse_doc("b", &[(9, 1.0)]),
        ])
        .await
        .expect("upsert");
        ns.compact(&CompactionOptions::default())
            .await
            .expect("compact");

        let report = ns
            .index_sparse(&IndexSparseOptions::default())
            .await
            .expect("index_sparse");
        assert_eq!(report.indexed, 1, "{}", backend.name);

        let (m, _) = manifest::load(&*backend.store, "docs")
            .await
            .expect("manifest");
        let id = segment::segment_id(1, 1);
        assert_eq!(m.sparse_indexes.get(&id).map(|e| e.generation), Some(1));

        let object = object_store::path::Path::from(format!("docs/segments/{id}/sparse-idx-1.bin"));
        assert!(
            backend.store.get(&object).await.is_ok(),
            "{}: the index object must exist",
            backend.name
        );
    }
}

#[tokio::test]
async fn rerunning_without_rebuild_reports_already_indexed() {
    let store = common::memory();
    let ns = create(store).await;
    ns.upsert(vec![sparse_doc("a", &[(3, 0.5)])])
        .await
        .expect("upsert");
    ns.compact(&CompactionOptions::default())
        .await
        .expect("compact");
    ns.index_sparse(&IndexSparseOptions::default())
        .await
        .expect("first index");

    let report = ns
        .index_sparse(&IndexSparseOptions::default())
        .await
        .expect("second index");
    assert_eq!((report.indexed, report.already_indexed), (0, 1));
}

#[tokio::test]
async fn rebuild_bumps_the_generation_and_leaves_the_old_object() {
    let store = common::memory();
    let ns = create(store.clone()).await;
    ns.upsert(vec![sparse_doc("a", &[(3, 0.5)])])
        .await
        .expect("upsert");
    ns.compact(&CompactionOptions::default())
        .await
        .expect("compact");
    ns.index_sparse(&IndexSparseOptions::default())
        .await
        .expect("first index");

    let report = ns
        .index_sparse(&IndexSparseOptions::default().rebuild(true))
        .await
        .expect("rebuild");
    assert_eq!(report.indexed, 1);

    let (m, _) = manifest::load(&*store, "docs").await.expect("manifest");
    let id = segment::segment_id(1, 1);
    assert_eq!(m.sparse_indexes.get(&id).map(|e| e.generation), Some(2));

    for generation in [1, 2] {
        let object = object_store::path::Path::from(format!(
            "docs/segments/{id}/sparse-idx-{generation}.bin"
        ));
        assert!(
            store.get(&object).await.is_ok(),
            "generation {generation} must still be there"
        );
    }
}

/// Issue #35: with every generation collected and the manifest lost, the
/// segment's high-water mark keeps a rebuild from reusing a number.
#[tokio::test]
async fn a_collected_generation_is_never_reused_after_recovery() {
    for backend in cas_backends() {
        let store = backend.store.clone();
        let ns = create(store.clone()).await;
        ns.upsert(vec![sparse_doc("a", &[(3, 0.5)])])
            .await
            .expect("upsert");
        ns.compact(&CompactionOptions::default())
            .await
            .expect("compact");
        ns.index_sparse(&IndexSparseOptions::default())
            .await
            .expect("index");
        ns.index_sparse(&IndexSparseOptions::default().rebuild(true))
            .await
            .expect("rebuild");

        let id = segment::segment_id(1, 1);
        for generation in [1, 2] {
            store
                .delete(&object_store::path::Path::from(format!(
                    "docs/segments/{id}/sparse-idx-{generation}.bin"
                )))
                .await
                .expect("collect");
        }
        store
            .delete(&object_store::path::Path::from(
                "docs/manifest/current.json",
            ))
            .await
            .expect("lose the manifest");
        let (recovered, _) = Namespace::recover(store.clone(), "docs")
            .await
            .expect("recover");

        recovered
            .index_sparse(&IndexSparseOptions::default())
            .await
            .expect("index after recovery");
        let (m, _) = manifest::load(&*store, "docs").await.expect("manifest");
        assert_eq!(
            m.sparse_indexes.get(&id).map(|e| e.generation),
            Some(3),
            "{}",
            backend.name
        );
    }
}

#[tokio::test]
async fn a_merge_drops_the_sparse_index_entry_of_a_retired_segment() {
    let store = common::memory();
    let ns = create(store.clone()).await;
    ns.upsert(vec![sparse_doc("a", &[(3, 0.5)])])
        .await
        .expect("upsert");
    ns.compact(&TIGHT).await.expect("first fold");
    ns.index_sparse(&IndexSparseOptions::default())
        .await
        .expect("index_sparse");
    let (before, _) = manifest::load(&*store, "docs").await.expect("manifest");
    assert_eq!(before.sparse_indexes.len(), 1);

    // A second fold with max_segments 1 has to merge, which retires the
    // first segment and, with it, its sparse index entry.
    ns.upsert(vec![doc("b", &[2.0])]).await.expect("upsert");
    ns.compact(&TIGHT).await.expect("merge");

    let (after, _) = manifest::load(&*store, "docs").await.expect("manifest");
    assert!(
        after.sparse_indexes.is_empty(),
        "the merged-away segment's entry must be pruned: {:?}",
        after.sparse_indexes
    );
}

/// An object already under the segment's prefix takes its generation, so a
/// build steps over it: the object is neither read nor overwritten, and the
/// index takes the next generation. A damaged one is gc's to delete.
#[tokio::test]
async fn an_unfit_object_under_the_prefix_is_stepped_over() {
    let store = common::memory();
    let ns = create(store.clone()).await;
    ns.upsert(vec![sparse_doc("a", &[(3, 0.5)])])
        .await
        .expect("upsert");
    ns.compact(&CompactionOptions::default())
        .await
        .expect("compact");

    let id = segment::segment_id(1, 1);
    let object = object_store::path::Path::from(format!("docs/segments/{id}/sparse-idx-1.bin"));
    let planted = bytes::Bytes::from_static(b"not an index");
    store
        .put(&object, planted.clone().into())
        .await
        .expect("plant");

    ns.index_sparse(&IndexSparseOptions::default())
        .await
        .expect("index_sparse");
    let kept = store
        .get(&object)
        .await
        .expect("get")
        .bytes()
        .await
        .expect("bytes");
    assert_eq!(kept, planted, "the planted object is left as it was");
    let (m, _) = manifest::load(&*store, "docs").await.expect("manifest");
    assert_eq!(m.sparse_indexes[&id].generation, 2);
}

#[tokio::test]
async fn a_valid_orphan_left_by_a_stopped_run_is_left_for_gc_and_not_reused() {
    let store = common::memory();
    let ns = create(store.clone()).await;
    ns.upsert(vec![sparse_doc("a", &[(3, 0.5)])])
        .await
        .expect("upsert");
    ns.compact(&CompactionOptions::default())
        .await
        .expect("compact");
    ns.index_sparse(&IndexSparseOptions::default())
        .await
        .expect("first index");

    // Drop the manifest entry by hand, leaving the object: what a run that
    // stopped between its put and its publish looks like from the next run.
    let id = segment::segment_id(1, 1);
    manifest::update(&*store, "docs", |m| {
        m.sparse_indexes.remove(&id);
    })
    .await
    .expect("remove entry");

    let report = ns
        .index_sparse(&IndexSparseOptions::default())
        .await
        .expect("rerun");
    assert_eq!(report.indexed, 1);
    // The orphan's generation is taken: the rerun names the next one and
    // leaves the orphan for gc.
    let (m, _) = manifest::load(&*store, "docs").await.expect("manifest");
    assert_eq!(m.sparse_indexes[&id].generation, 2);
}

/// A live segment whose sparse index was rebuilt, so generation 1 is a stray
/// next to the named generation 2.
async fn with_a_stray_generation(store: std::sync::Arc<dyn ObjectStore>) -> Namespace {
    let ns = create(store).await;
    ns.upsert(vec![sparse_doc("a", &[(3, 0.5)])])
        .await
        .expect("upsert");
    ns.compact(&CompactionOptions::default())
        .await
        .expect("compact");
    ns.index_sparse(&IndexSparseOptions::default())
        .await
        .expect("index");
    ns.index_sparse(&IndexSparseOptions::default().rebuild(true))
        .await
        .expect("rebuild");
    ns
}

#[tokio::test]
async fn gc_collects_a_stray_sparse_generation_and_keeps_the_named_one() {
    for backend in cas_backends() {
        let ns = with_a_stray_generation(backend.store.clone()).await;
        let now = GcOptions {
            min_age: Duration::ZERO,
            ..GcOptions::default()
        };

        let dry = ns
            .gc(&GcOptions {
                dry_run: true,
                ..now
            })
            .await
            .expect("dry run");
        assert_eq!(
            (dry.sparse_indexes_deleted, dry.sparse_indexes_too_young),
            (1, 0),
            "{}",
            backend.name
        );
        assert!(dry.bytes_freed > 0);

        let freed = ns.gc(&now).await.expect("gc");
        assert_eq!(freed.sparse_indexes_deleted, 1, "{}", backend.name);
        assert_eq!(freed.segments_deleted, 0);

        let (m, _) = manifest::load(&*backend.store, "docs").await.expect("load");
        let id = &m.live_segments[0];
        let named = m.sparse_indexes[id].generation;
        assert_eq!(named, 2);
        backend
            .store
            .head(&object_store::path::Path::from(format!(
                "docs/segments/{id}/sparse-idx-2.bin"
            )))
            .await
            .expect("the named index survives");
        let again = ns.gc(&now).await.expect("second gc");
        assert_eq!(again.sparse_indexes_deleted, 0, "{}", backend.name);
    }
}

#[tokio::test]
async fn gc_leaves_a_young_stray_sparse_generation_for_later() {
    for backend in cas_backends() {
        let ns = with_a_stray_generation(backend.store.clone()).await;
        let report = ns.gc(&GcOptions::default()).await.expect("gc");
        assert_eq!(
            (
                report.sparse_indexes_deleted,
                report.sparse_indexes_too_young
            ),
            (0, 1),
            "{}",
            backend.name
        );
    }
}

// ---------------------------------------------------------------------------
// Faults and races
// ---------------------------------------------------------------------------

/// Collect anything, however young.
const NOW: GcOptions = GcOptions {
    min_age: Duration::ZERO,
    dry_run: false,
    fence_wal: false,
};

fn dynamic(store: &Arc<FaultyStore>) -> Arc<dyn ObjectStore> {
    store.clone()
}

fn seg(lo: u64, hi: u64) -> String {
    segment::segment_id(lo, hi)
}

/// Two live segments, `1-1` and `2-2`, six documents each, none indexed.
async fn two_sparse_segments(store: Arc<dyn ObjectStore>) -> Namespace {
    let ns = create(store).await;
    for first in [0u32, 6] {
        let docs = (first..first + 6)
            .map(|i| {
                sparse_doc(
                    &format!("d{i:02}"),
                    &[(i % 4, 1.0 + i as f32 * 0.5), (9, 0.25)],
                )
            })
            .collect();
        ns.upsert(docs).await.expect("upsert");
        ns.compact(&CompactionOptions::default())
            .await
            .expect("compact");
    }
    ns
}

async fn answer(ns: &Namespace) -> Vec<String> {
    ns.query_sparse(&vec![(1, 1.0), (9, 1.0)], 5, &SparseQueryOptions::default())
        .await
        .expect("query_sparse")
        .into_iter()
        .map(|hit| hit.id)
        .collect()
}

async fn sparse_index_keys(store: &dyn ObjectStore) -> Vec<String> {
    let prefix = object_store::path::Path::from("docs/segments");
    let metas: Vec<_> = store.list(Some(&prefix)).try_collect().await.expect("list");
    metas
        .into_iter()
        .filter(|m| {
            m.location
                .filename()
                .is_some_and(|f| f.starts_with("sparse-idx-"))
        })
        .map(|m| m.location.to_string())
        .collect()
}

async fn sparse_indexes_of(store: &dyn ObjectStore) -> usize {
    manifest::load(store, "docs")
        .await
        .expect("load")
        .0
        .sparse_indexes
        .len()
}

/// Retire both segments as a merge followed by a collection would.
async fn merge_and_delete_old(store: Arc<dyn ObjectStore>) {
    let ns = Namespace::open(store.clone(), "docs").await.expect("open");
    ns.compact(&TIGHT).await.expect("merge");
    for id in [seg(1, 1), seg(2, 2)] {
        store
            .delete(&segment::segment_path("docs", &id))
            .await
            .expect("delete");
    }
}

/// Fault every operation of a run that indexes two segments, as a plain error
/// and as a lost ack. Afterwards the namespace answers as before and a rerun
/// finishes what the first left.
#[tokio::test]
async fn a_stopped_index_sparse_run_is_finished_by_the_next_one() {
    for ix in 0..cas_backends().len() {
        let counted = FaultyStore::new(cas_backends().swap_remove(ix).store);
        let ns = two_sparse_segments(dynamic(&counted)).await;
        counted.reset_ops();
        ns.index_sparse(&IndexSparseOptions::default())
            .await
            .expect("index_sparse");
        let total = counted.ops();
        assert!(
            total >= 10,
            "two segments are ten or so operations, not {total}"
        );

        for fault in [Fault::Generic, Fault::LostAck] {
            for n in 0..total {
                eprintln!("backend {ix}: {fault:?} at operation {n} of {total}");
                let store = FaultyStore::new(cas_backends().swap_remove(ix).store);
                let ns = two_sparse_segments(dynamic(&store)).await;
                let expected = answer(&ns).await;

                store.fail_nth_op(n, fault);
                let stopped = ns.index_sparse(&IndexSparseOptions::default()).await;
                store.clear();
                assert!(stopped.is_err(), "a fault inside index_sparse must show");
                assert_eq!(
                    answer(&ns).await,
                    expected,
                    "the stopped run changed answers"
                );
                drop(ns);

                assert_consistent(dynamic(&store), 1, Metric::L2).await;
                let ns = Namespace::open(dynamic(&store), "docs")
                    .await
                    .expect("open");
                let finished = ns
                    .index_sparse(&IndexSparseOptions::default())
                    .await
                    .expect("rerun");
                assert_eq!(
                    finished.indexed + finished.already_indexed,
                    2,
                    "{finished:?}"
                );
                assert_eq!(sparse_indexes_of(&*store).await, 2);
                assert_eq!(answer(&ns).await, expected);

                let again = ns
                    .index_sparse(&IndexSparseOptions::default())
                    .await
                    .expect("noop");
                assert_eq!((again.indexed, again.already_indexed), (0, 2));
            }
        }
    }
}

/// A query never writes, so one namespace serves the whole sweep. A fault at
/// any operation is an error or the right answer, never a wrong one, and the
/// next query is right again.
#[tokio::test]
async fn a_query_sparse_faulted_at_every_operation_is_an_error_or_correct() {
    for ix in 0..cas_backends().len() {
        for indexed in [false, true] {
            let store = FaultyStore::new(cas_backends().swap_remove(ix).store);
            let ns = two_sparse_segments(dynamic(&store)).await;
            if indexed {
                ns.index_sparse(&IndexSparseOptions::default())
                    .await
                    .expect("index_sparse");
            }
            let expected = answer(&ns).await;
            store.reset_ops();
            assert_eq!(answer(&ns).await, expected);
            let total = store.ops();
            assert!(
                total >= 3,
                "a query is at least a few operations, not {total}"
            );

            for fault in [Fault::Generic, Fault::LostAck] {
                for n in 0..total {
                    store.fail_nth_op(n, fault);
                    let result = ns
                        .query_sparse(&vec![(1, 1.0), (9, 1.0)], 5, &SparseQueryOptions::default())
                        .await;
                    store.clear();
                    if let Ok(hits) = result {
                        let got: Vec<String> = hits.into_iter().map(|h| h.id).collect();
                        assert_eq!(got, expected, "{fault:?} at {n}, indexed {indexed}");
                    }
                    assert_eq!(answer(&ns).await, expected);
                }
            }
        }
    }
}

/// The merge lands before the run has read anything: both segments are gone
/// and the run counts them as superseded.
#[tokio::test]
async fn a_merge_before_the_first_read_supersedes_every_segment() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let faulty = FaultyStore::new(b.store.clone());
        let ns = two_sparse_segments(dynamic(&faulty)).await;

        let store = b.store.clone();
        faulty.before_get(&seg(1, 1), async move {
            merge_and_delete_old(store).await;
        });
        let outcome = ns
            .index_sparse(&IndexSparseOptions::default())
            .await
            .expect("index_sparse");
        assert_eq!((outcome.indexed, outcome.superseded), (0, 2));
        assert!(sparse_index_keys(&*b.store).await.is_empty());
    }
}

/// The merge lands after the run has read the segment but before it publishes:
/// the swap declines, and the object it wrote sits in a dead directory until
/// `gc` takes the directory whole.
#[tokio::test]
async fn a_merge_between_the_write_and_the_publish_declines_the_publish() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let faulty = FaultyStore::new(b.store.clone());
        let ns = two_sparse_segments(dynamic(&faulty)).await;

        let other = Namespace::open(b.store.clone(), "docs")
            .await
            .expect("open");
        faulty.before_get(&seg(1, 1), async move {
            other.compact(&TIGHT).await.expect("merge");
        });
        let outcome = ns
            .index_sparse(&IndexSparseOptions::default())
            .await
            .expect("index_sparse");
        assert_eq!((outcome.indexed, outcome.superseded), (0, 2));
        assert_eq!(sparse_indexes_of(&*b.store).await, 0);
        assert_eq!(
            sparse_index_keys(&*b.store).await.len(),
            2,
            "written, never listed"
        );

        let freed = ns.gc(&NOW).await.expect("gc");
        assert_eq!(freed.segments_deleted, 2);
        assert!(sparse_index_keys(&*b.store).await.is_empty());
    }
}

/// Another process rebuilds the sparse index and collects the old generation
/// after a query loaded its manifest but before it fetched the index: the query
/// reloads and answers from the new generation.
#[tokio::test]
async fn a_query_whose_sparse_index_generation_was_collected_reloads_and_answers() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let faulty = FaultyStore::new(b.store.clone());
        let ns = two_sparse_segments(dynamic(&faulty)).await;
        ns.index_sparse(&IndexSparseOptions::default())
            .await
            .expect("index_sparse");
        let expected = answer(&ns).await;

        let other = Namespace::open(b.store.clone(), "docs")
            .await
            .expect("open");
        faulty.before_get("sparse-idx-1.bin", async move {
            other
                .index_sparse(&IndexSparseOptions::default().rebuild(true))
                .await
                .expect("rebuild");
            other.gc(&NOW).await.expect("gc");
        });
        assert_eq!(answer(&ns).await, expected, "{}", b.name);
        let m = manifest::load(&*b.store, "docs").await.expect("load").0;
        assert!(m.sparse_indexes.values().all(|e| e.generation == 2));
    }
}

/// A merge and a collection retire an indexed segment under a query.
#[tokio::test]
async fn a_query_whose_indexed_segments_were_merged_away_reloads_and_answers() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let faulty = FaultyStore::new(b.store.clone());
        let ns = two_sparse_segments(dynamic(&faulty)).await;
        ns.index_sparse(&IndexSparseOptions::default())
            .await
            .expect("index_sparse");
        let expected = answer(&ns).await;

        let store = b.store.clone();
        faulty.before_get(&seg(1, 1), async move {
            merge_and_delete_old(store).await;
        });
        assert_eq!(answer(&ns).await, expected, "{}", b.name);
    }
}
