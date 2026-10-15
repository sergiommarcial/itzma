//! `inspect::report`: what it shows for healthy and damaged namespaces, and
//! that it never changes anything.

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::faulty::{Fault, FaultyStore};
use common::{backends, cas_backends, doc, publish, put_segment, schema, snapshot};
use itzma_core::inspect::{self, Report, SegmentState};
use itzma_core::{
    CompactionOptions, GcOptions, ItzmaError, Metric, Namespace, WalOp, manifest, schema_object,
    segment, wal,
};
use object_store::{ObjectStore, ObjectStoreExt};

const DIM: usize = 1;
const ROOMY: CompactionOptions = CompactionOptions {
    max_segments: 8,
    max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
};
const TIGHT: CompactionOptions = CompactionOptions {
    max_segments: 1,
    max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
};
const NOW: GcOptions = GcOptions {
    min_age: Duration::ZERO,
    dry_run: false,
    fence_wal: false,
};

async fn create(store: Arc<dyn ObjectStore>) -> Namespace {
    Namespace::create(store, "docs", schema(DIM, Metric::L2))
        .await
        .unwrap()
}

async fn write(ns: &Namespace, id: &str, x: f32) {
    ns.upsert(vec![doc(id, &[x])]).await.unwrap();
}

async fn report(store: &dyn ObjectStore) -> Report {
    inspect::report(store, "docs").await.unwrap()
}

fn states(report: &Report, state: SegmentState) -> usize {
    report.segments.iter().filter(|s| s.state == state).count()
}

fn has_problem(report: &Report, needle: &str) -> bool {
    report.problems.iter().any(|p| p.contains(needle))
}

#[tokio::test]
async fn a_new_namespace_is_healthy_and_empty() {
    for b in backends() {
        create(b.store.clone()).await;
        let r = report(&*b.store).await;

        assert_eq!(r.namespace, "docs");
        assert_eq!(r.schema, Some(schema(DIM, Metric::L2)), "{}", b.name);
        let m = r.manifest.expect("manifest");
        assert_eq!((m.wal_folded_through, m.live_segments.len()), (0, 0));
        assert_eq!(m.wal_format, wal::FORMAT_VERSION);
        assert_eq!(m.segment_format, segment::DEFAULT_FORMAT);
        assert_eq!(r.wal.objects, 0);
        assert_eq!((r.wal.first, r.wal.last), (None, None));
        assert!(r.segments.is_empty());
        assert!(r.problems.is_empty(), "{}: {:?}", b.name, r.problems);
    }
}

#[tokio::test]
async fn an_unfolded_tail_is_counted() {
    for b in backends() {
        let ns = create(b.store.clone()).await;
        for (id, x) in [("a", 1.0), ("b", 2.0), ("c", 3.0)] {
            write(&ns, id, x).await;
        }
        let r = report(&*b.store).await;

        assert_eq!(r.wal.objects, 3, "{}", b.name);
        assert_eq!((r.wal.first, r.wal.last), (Some(1), Some(3)));
        assert_eq!((r.wal.tail, r.wal.fenced), (3, 0));
        assert!(r.wal.bytes > 0);
        assert!(r.problems.is_empty(), "{}: {:?}", b.name, r.problems);
    }
}

#[tokio::test]
async fn folded_and_merged_segments_show_as_live_and_dead() {
    for b in cas_backends() {
        let ns = create(b.store.clone()).await;
        for (id, x) in [("a", 1.0), ("b", 2.0), ("c", 3.0)] {
            write(&ns, id, x).await;
            ns.compact(&ROOMY).await.unwrap();
        }
        // Three folds, then a merge of all of them into `1-3`.
        ns.compact(&TIGHT).await.unwrap();
        let (m, _) = manifest::load(&*b.store, "docs").await.unwrap();
        let r = report(&*b.store).await;

        assert_eq!(m.live_segments, vec![segment::segment_id(1, 3)]);
        assert_eq!(states(&r, SegmentState::Live), 1, "{}", b.name);
        assert_eq!(states(&r, SegmentState::Dead), 3);
        assert_eq!(states(&r, SegmentState::Orphan), 0);
        assert_eq!(r.wal.tail, 0);
        assert!(r.problems.is_empty(), "{}: {:?}", b.name, r.problems);
        assert!(r.segments.windows(2).all(|w| w[0].id < w[1].id));
        assert!(r.segments.iter().all(|s| s.bytes > 0));
    }
}

/// The report and `gc` share one rule, so what the first calls dead is exactly
/// what the second deletes.
#[tokio::test]
async fn dead_is_what_gc_deletes() {
    for b in cas_backends() {
        let ns = create(b.store.clone()).await;
        for (id, x) in [("a", 1.0), ("b", 2.0), ("c", 3.0), ("d", 4.0)] {
            write(&ns, id, x).await;
            ns.compact(&ROOMY).await.unwrap();
        }
        ns.compact(&TIGHT).await.unwrap();
        write(&ns, "e", 5.0).await;
        ns.compact(&ROOMY).await.unwrap();

        let before = report(&*b.store).await;
        let dead = states(&before, SegmentState::Dead);
        assert!(dead > 0, "{}: nothing to collect", b.name);

        let gc = ns.gc(&NOW).await.unwrap();
        assert_eq!(gc.segments_deleted, dead, "{}", b.name);

        let after = report(&*b.store).await;
        assert_eq!(states(&after, SegmentState::Dead), 0);
        assert_eq!(
            after.segments.len(),
            before.segments.len() - dead,
            "{}",
            b.name
        );
        assert!(
            after.problems.is_empty(),
            "{}: {:?}",
            b.name,
            after.problems
        );
    }
}

#[tokio::test]
async fn fences_are_counted() {
    for b in cas_backends() {
        let ns = create(b.store.clone()).await;
        for (id, x) in [("a", 1.0), ("b", 2.0), ("c", 3.0)] {
            write(&ns, id, x).await;
        }
        ns.compact(&ROOMY).await.unwrap();
        write(&ns, "d", 4.0).await;

        let run = ns
            .gc(&GcOptions {
                fence_wal: true,
                ..NOW
            })
            .await
            .unwrap();
        let r = report(&*b.store).await;

        assert_eq!(run.wal_objects_fenced, 3, "{}", b.name);
        assert_eq!(r.wal.fenced, 3);
        assert_eq!((r.wal.objects, r.wal.tail), (4, 1));
        assert!(r.problems.is_empty(), "{}: {:?}", b.name, r.problems);
    }
}

#[tokio::test]
async fn an_unpublished_segment_is_an_orphan_and_not_a_problem() {
    for b in backends() {
        let ns = create(b.store.clone()).await;
        write(&ns, "a", 1.0).await;
        put_segment(
            &*b.store,
            "docs",
            5,
            9,
            DIM,
            vec![WalOp::Upsert(doc("z", &[9.0]))],
        )
        .await;
        let r = report(&*b.store).await;

        assert_eq!(states(&r, SegmentState::Orphan), 1, "{}", b.name);
        assert!(r.problems.is_empty(), "{}: {:?}", b.name, r.problems);
    }
}

#[tokio::test]
async fn a_gap_in_the_wal_is_reported() {
    for b in backends() {
        let ns = create(b.store.clone()).await;
        for (id, x) in [("a", 1.0), ("b", 2.0), ("c", 3.0), ("d", 4.0)] {
            write(&ns, id, x).await;
        }
        b.store.delete(&wal::wal_path("docs", 2)).await.unwrap();
        b.store.delete(&wal::wal_path("docs", 3)).await.unwrap();
        let r = report(&*b.store).await;

        assert_eq!(r.wal.gaps, vec![(2, 3)], "{}", b.name);
        assert!(has_problem(&r, "WAL seqs 2 through 3 are missing"));

        b.store.delete(&wal::wal_path("docs", 4)).await.unwrap();
        b.store.delete(&wal::wal_path("docs", 1)).await.unwrap();
        write(&ns, "e", 5.0).await;
        // The WAL is now 5 alone: the hint puts the write at seq 5.
        let r = report(&*b.store).await;
        assert_eq!(r.wal.gaps, vec![(1, 4)], "{}", b.name);
        assert!(has_problem(&r, "WAL seqs 1 through 4 are missing"));
    }
}

#[tokio::test]
async fn a_fold_point_past_the_wal_is_reported() {
    for b in cas_backends() {
        let ns = create(b.store.clone()).await;
        write(&ns, "a", 1.0).await;
        write(&ns, "b", 2.0).await;
        ns.compact(&ROOMY).await.unwrap();
        b.store.delete(&wal::wal_path("docs", 2)).await.unwrap();
        let r = report(&*b.store).await;

        assert_eq!(r.wal.gaps, vec![(2, 2)], "{}", b.name);
        assert!(has_problem(&r, "WAL seq 2 is missing"));
    }
}

#[tokio::test]
async fn a_live_segment_with_no_data_is_reported() {
    for b in cas_backends() {
        let ns = create(b.store.clone()).await;
        write(&ns, "a", 1.0).await;
        ns.compact(&ROOMY).await.unwrap();
        let id = segment::segment_id(1, 1);
        b.store
            .delete(&segment::segment_path("docs", &id))
            .await
            .unwrap();
        let r = report(&*b.store).await;

        assert!(
            has_problem(&r, &format!("live segment {id} has no data object")),
            "{}: {:?}",
            b.name,
            r.problems
        );
    }
}

#[tokio::test]
async fn a_broken_segment_chain_is_reported() {
    for b in cas_backends() {
        let ns = create(b.store.clone()).await;
        write(&ns, "a", 1.0).await;
        write(&ns, "b", 2.0).await;
        let id = put_segment(
            &*b.store,
            "docs",
            1,
            2,
            DIM,
            vec![WalOp::Upsert(doc("a", &[1.0]))],
        )
        .await;
        // The segment covers 1 through 2 and the manifest claims only 1.
        publish(&*b.store, "docs", &[&id], 1).await;
        let r = report(&*b.store).await;

        assert!(
            has_problem(&r, "wal_folded_through"),
            "{}: {:?}",
            b.name,
            r.problems
        );
        assert_eq!(states(&r, SegmentState::Live), 1);
    }
}

#[tokio::test]
async fn a_missing_manifest_still_shows_the_rest() {
    for b in backends() {
        let ns = create(b.store.clone()).await;
        write(&ns, "a", 1.0).await;
        b.store
            .delete(&manifest::manifest_path("docs"))
            .await
            .unwrap();
        let r = report(&*b.store).await;

        assert!(r.manifest.is_none());
        assert!(has_problem(&r, "manifest is missing"), "{}", b.name);
        assert_eq!(r.schema, Some(schema(DIM, Metric::L2)));
        assert_eq!((r.wal.objects, r.wal.tail), (1, 1));
    }
}

#[tokio::test]
async fn a_namespace_that_does_not_exist_reports_its_manifest_missing() {
    for b in backends() {
        let r = inspect::report(&*b.store, "nope").await.unwrap();

        assert!(r.manifest.is_none() && r.schema.is_none(), "{}", b.name);
        assert_eq!(r.problems, vec!["manifest is missing".to_string()]);
    }
}

#[tokio::test]
async fn an_unreadable_manifest_is_a_problem_and_the_wal_still_lists() {
    for b in backends() {
        let ns = create(b.store.clone()).await;
        write(&ns, "a", 1.0).await;
        b.store
            .put(&manifest::manifest_path("docs"), "not json".into())
            .await
            .unwrap();
        let r = report(&*b.store).await;

        assert!(r.manifest.is_none());
        assert!(has_problem(&r, "manifest cannot be read"), "{}", b.name);
        assert_eq!(r.wal.objects, 1);
    }
}

#[tokio::test]
async fn a_manifest_from_a_newer_build_is_a_problem_not_an_error() {
    for b in backends() {
        let ns = create(b.store.clone()).await;
        write(&ns, "a", 1.0).await;
        let (mut m, _) = manifest::load(&*b.store, "docs").await.unwrap();
        m.wal_format = wal::FORMAT_VERSION + 1;
        b.store
            .put(
                &manifest::manifest_path("docs"),
                serde_json::to_vec(&m).unwrap().into(),
            )
            .await
            .unwrap();
        let r = report(&*b.store).await;

        assert!(has_problem(&r, "manifest cannot be read"), "{}", b.name);
        assert_eq!(r.wal.objects, 1);
    }
}

#[tokio::test]
async fn a_manifest_from_a_newer_segment_format_is_a_problem_not_an_error() {
    for b in backends() {
        let ns = create(b.store.clone()).await;
        write(&ns, "a", 1.0).await;
        let (mut m, _) = manifest::load(&*b.store, "docs").await.unwrap();
        m.segment_format = segment::FORMAT_VERSION + 1;
        b.store
            .put(
                &manifest::manifest_path("docs"),
                serde_json::to_vec(&m).unwrap().into(),
            )
            .await
            .unwrap();
        let r = report(&*b.store).await;

        assert!(r.manifest.is_none());
        assert!(has_problem(&r, "manifest cannot be read"), "{}", b.name);
        assert!(
            has_problem(&r, "segment format"),
            "{}: {:?}",
            b.name,
            r.problems
        );
        assert_eq!(r.wal.objects, 1);
    }
}

#[tokio::test]
async fn the_manifest_line_shows_both_levels() {
    for b in backends() {
        create(b.store.clone()).await;
        let text = report(&*b.store).await.to_string();
        let expected = format!(
            "wal_format {}, segment_format {}",
            wal::FORMAT_VERSION,
            segment::DEFAULT_FORMAT
        );
        assert!(text.contains(&expected), "{}: {text}", b.name);
    }
}

#[tokio::test]
async fn a_schema_that_differs_from_the_manifest_is_reported() {
    for b in backends() {
        create(b.store.clone()).await;
        b.store
            .put(
                &schema_object::schema_path("docs"),
                br#"{"format_version":2,"dim":3,"metric":"l2","attrs":{}}"#
                    .as_slice()
                    .into(),
            )
            .await
            .unwrap();
        let r = report(&*b.store).await;

        assert_eq!(r.schema, Some(schema(3, Metric::L2)), "{}", b.name);
        assert!(has_problem(&r, "schema.json holds"), "{:?}", r.problems);
    }
}

/// `Namespace::open` writes a missing `schema.json`. A report must not.
#[tokio::test]
async fn a_missing_schema_object_is_shown_and_not_healed() {
    for b in backends() {
        create(b.store.clone()).await;
        b.store
            .delete(&schema_object::schema_path("docs"))
            .await
            .unwrap();
        let before = snapshot(&*b.store).await;
        let r = report(&*b.store).await;

        assert!(r.schema.is_none(), "{}", b.name);
        assert!(r.problems.is_empty(), "{}: {:?}", b.name, r.problems);
        assert_eq!(snapshot(&*b.store).await, before);
    }
}

/// Every kind of state at once, and nothing in the store moves, the manifest
/// and the schema object included.
#[tokio::test]
async fn a_report_changes_nothing() {
    for b in cas_backends() {
        let ns = create(b.store.clone()).await;
        for (id, x) in [("a", 1.0), ("b", 2.0), ("c", 3.0)] {
            write(&ns, id, x).await;
            ns.compact(&ROOMY).await.unwrap();
        }
        ns.compact(&TIGHT).await.unwrap();
        write(&ns, "d", 4.0).await;
        ns.gc(&GcOptions {
            fence_wal: true,
            ..NOW
        })
        .await
        .unwrap();
        let before = snapshot(&*b.store).await;

        report(&*b.store).await;
        assert_eq!(snapshot(&*b.store).await, before, "{}", b.name);
    }
}

#[tokio::test]
async fn a_malformed_id_is_refused_before_any_storage_call() {
    let faulty = FaultyStore::new(common::memory());
    let err = inspect::report(&*faulty, "Not Valid").await.unwrap_err();

    assert!(matches!(err, ItzmaError::InvalidNamespaceId(_)));
    assert_eq!(faulty.ops(), 0);
}

/// A store that fails is not described as a problem with the namespace: the
/// call errors, whichever operation it was.
#[tokio::test]
async fn a_store_failure_is_an_error_at_every_operation() {
    let inner = common::memory();
    let ns = create(inner.clone()).await;
    write(&ns, "a", 1.0).await;
    ns.compact(&ROOMY).await.unwrap();
    let faulty = FaultyStore::new(inner);

    faulty.reset_ops();
    report(&*faulty).await;
    let ops = faulty.ops();
    assert!(ops >= 4, "manifest, schema and two listings, at least");

    for n in 0..ops {
        faulty.fail_nth_op(n, Fault::Generic);
        let err = inspect::report(&*faulty, "docs").await.unwrap_err();
        assert!(matches!(err, ItzmaError::Storage(_)), "op {n}: {err:?}");
        faulty.clear();
    }
    assert!(report(&*faulty).await.problems.is_empty());
}

#[tokio::test]
async fn the_display_names_each_part_and_each_problem() {
    for b in cas_backends() {
        let ns = create(b.store.clone()).await;
        write(&ns, "a", 1.0).await;
        ns.compact(&ROOMY).await.unwrap();
        let healthy = report(&*b.store).await.to_string();

        assert!(healthy.contains("namespace docs"), "{healthy}");
        assert!(healthy.contains("dim 1, L2"), "{healthy}");
        assert!(healthy.contains("folded through 1, 1 live segments"));
        assert!(healthy.contains(&format!("{}  live", segment::segment_id(1, 1))));
        assert!(healthy.contains("problems     none"));

        b.store.delete(&wal::wal_path("docs", 1)).await.unwrap();
        let broken = report(&*b.store).await.to_string();
        assert!(broken.contains("problems     1"), "{broken}");
        assert!(broken.contains("WAL seq 1 is missing"));
    }
}
