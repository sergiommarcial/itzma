//! Rebuilding a lost manifest.
//!
//! The model is the namespace itself before the manifest is deleted: a
//! recovery has to give back the same manifest, or one that answers every query
//! the same way. `assert_consistent` then checks the result against the WAL and
//! the segments independently.

mod common;

use std::sync::Arc;

use bytes::Bytes;
use common::faulty::{Fault, FaultyStore};
use common::{
    Snapshot, assert_consistent, assert_untouched, cas_backends, doc, schema, snapshot, wal_listing,
};
use itzma_core::{
    CompactionOptions, Document, GcOptions, Hit, IndexSparseOptions, ItzmaError, Manifest, Metric,
    Namespace, SparseQueryOptions, manifest, segment, wal,
};
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
    min_age: std::time::Duration::ZERO,
    dry_run: false,
    fence_wal: false,
};
const FENCING: GcOptions = GcOptions {
    fence_wal: true,
    ..NOW
};

fn dynamic(store: &Arc<FaultyStore>) -> Arc<dyn ObjectStore> {
    store.clone()
}

/// The five operations every stage is made of, so an overwrite and a delete
/// sit in different segments.
async fn op(ns: &Namespace, n: usize) {
    match n {
        1 => ns.upsert(vec![doc("a", &[1.0, 0.0])]).await.unwrap(),
        2 => ns.upsert(vec![doc("b", &[2.0, 0.0])]).await.unwrap(),
        3 => ns.upsert(vec![doc("a", &[5.0, 0.0])]).await.unwrap(),
        4 => ns.delete(vec!["b".into()]).await.unwrap(),
        5 => ns.upsert(vec![doc("c", &[3.0, 0.0])]).await.unwrap(),
        _ => unreachable!(),
    };
}

#[derive(Debug, Clone, Copy)]
enum Stage {
    /// Created, nothing written.
    Created,
    /// Five writes, nothing folded.
    Unfolded,
    /// Three writes folded into one segment, two in the tail.
    OneFold,
    /// A fold after every write: five delta segments.
    Deltas,
    /// The five deltas merged into `1-5`, and the deltas still in the store.
    MergedWithDeadDeltas,
    /// The same after a collection deleted the deltas.
    Collected,
    /// Merged, collected and fenced, then two more writes.
    Fenced,
    /// A version 2 segment, a version 3 segment and a tail.
    Mixed,
}

const STAGES: [Stage; 8] = [
    Stage::Created,
    Stage::Unfolded,
    Stage::OneFold,
    Stage::Deltas,
    Stage::MergedWithDeadDeltas,
    Stage::Collected,
    Stage::Fenced,
    Stage::Mixed,
];

async fn build(store: Arc<dyn ObjectStore>, stage: Stage) -> Namespace {
    let ns = Namespace::create(store.clone(), "docs", schema(DIM, L2))
        .await
        .unwrap();
    match stage {
        Stage::Created => {}
        Stage::Unfolded => {
            for n in 1..=5 {
                op(&ns, n).await;
            }
        }
        Stage::OneFold => {
            for n in 1..=3 {
                op(&ns, n).await;
            }
            ns.compact(&ROOMY).await.unwrap();
            for n in 4..=5 {
                op(&ns, n).await;
            }
        }
        Stage::Deltas => deltas(&ns).await,
        Stage::MergedWithDeadDeltas => {
            deltas(&ns).await;
            ns.compact(&TIGHT).await.unwrap();
        }
        Stage::Collected => {
            deltas(&ns).await;
            ns.compact(&TIGHT).await.unwrap();
            ns.gc(&NOW).await.unwrap();
        }
        Stage::Fenced => {
            deltas(&ns).await;
            ns.compact(&TIGHT).await.unwrap();
            ns.gc(&FENCING).await.unwrap();
            ns.upsert(vec![doc("d", &[4.0, 0.0])]).await.unwrap();
            ns.delete(vec!["c".into()]).await.unwrap();
        }
        Stage::Mixed => {
            op(&ns, 1).await;
            op(&ns, 2).await;
            ns.compact(&ROOMY).await.unwrap();
            ns.upgrade_segment_format(3).await.unwrap();
            op(&ns, 3).await;
            op(&ns, 4).await;
            ns.compact(&ROOMY).await.unwrap();
            op(&ns, 5).await;
        }
    }
    ns
}

async fn deltas(ns: &Namespace) {
    for n in 1..=5 {
        op(ns, n).await;
        ns.compact(&ROOMY).await.unwrap();
    }
}

async fn manifest_of(store: &dyn ObjectStore) -> Manifest {
    manifest::load(store, "docs").await.unwrap().0
}

async fn lose_manifest(store: &dyn ObjectStore) {
    store
        .delete(&manifest::manifest_path("docs"))
        .await
        .unwrap();
}

async fn hits(ns: &Namespace) -> Vec<Hit> {
    ns.query(&QUERY, 1_000).await.unwrap()
}

fn segment_key(lo: u64, hi: u64) -> object_store::path::Path {
    segment::segment_path("docs", &segment::segment_id(lo, hi))
}

async fn bytes_at(store: &dyn ObjectStore, path: &object_store::path::Path) -> Bytes {
    store.get(path).await.unwrap().bytes().await.unwrap()
}

fn segment_dirs(snap: &Snapshot) -> usize {
    snap.keys()
        .filter(|key| key.contains("/segments/") && key.ends_with("/data.bin"))
        .count()
}

/// The manifest is the only thing a recovery adds.
async fn assert_only_the_manifest_was_added(store: &dyn ObjectStore, before: &Snapshot) {
    assert_untouched(store, before).await;
    let after = snapshot(store).await;
    let added: Vec<&String> = after.keys().filter(|k| !before.contains_key(*k)).collect();
    assert_eq!(added, ["docs/manifest/current.json"]);
}

#[tokio::test]
async fn a_lost_manifest_is_rebuilt_at_every_stage() {
    for stage in STAGES {
        for b in cas_backends() {
            eprintln!("backend: {}, stage {stage:?}", b.name);
            let ns = build(b.store.clone(), stage).await;
            let expected = hits(&ns).await;
            let before = manifest_of(&*b.store).await;
            lose_manifest(&*b.store).await;
            let lost = snapshot(&*b.store).await;
            let err = Namespace::open(b.store.clone(), "docs").await.err();
            assert!(matches!(err, Some(ItzmaError::NamespaceNotFound(_))));

            let (recovered, report) = Namespace::recover(b.store.clone(), "docs").await.unwrap();
            assert_eq!(manifest_of(&*b.store).await, before);
            assert_eq!(report.live_segments, before.live_segments);
            assert_eq!(report.folded_through, before.wal_folded_through);
            let wal_len = wal_listing(&*b.store).await.len() as u64;
            assert_eq!(report.tail_objects, wal_len - before.wal_folded_through);
            assert_eq!(
                report.unused_segments,
                segment_dirs(&lost) - before.live_segments.len()
            );
            assert_only_the_manifest_was_added(&*b.store, &lost).await;

            assert_eq!(hits(&recovered).await, expected);
            let fresh = Namespace::open(b.store.clone(), "docs").await.unwrap();
            assert_eq!(hits(&fresh).await, expected);
            assert_consistent(b.store.clone(), DIM, L2).await;

            // The recovered namespace takes writes, folds and collections.
            let seq = recovered.upsert(vec![doc("z", &[9.0, 0.0])]).await.unwrap();
            assert_eq!(seq, wal_len + 1);
            recovered.compact(&ROOMY).await.unwrap();
            recovered.gc(&NOW).await.unwrap();
            assert_eq!(hits(&recovered).await.len(), expected.len() + 1);
            assert_consistent(b.store.clone(), DIM, L2).await;
        }
    }
}

#[tokio::test]
async fn a_manifest_that_exists_is_never_replaced() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        build(b.store.clone(), Stage::Deltas).await;
        let before = snapshot(&*b.store).await;

        let err = Namespace::recover(b.store.clone(), "docs").await.err();
        assert!(
            matches!(err, Some(ItzmaError::ManifestExists(_))),
            "{err:?}"
        );
        assert_eq!(snapshot(&*b.store).await, before);
    }
}

#[tokio::test]
async fn a_malformed_id_is_refused_before_the_store_is_read() {
    let store = FaultyStore::new(common::memory());
    let err = Namespace::recover(dynamic(&store), "Bad/Id").await.err();
    assert!(matches!(err, Some(ItzmaError::InvalidNamespaceId(_))));
    assert_eq!(store.ops(), 0);
}

#[tokio::test]
async fn without_schema_json_nothing_is_rebuilt() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        build(b.store.clone(), Stage::OneFold).await;
        lose_manifest(&*b.store).await;
        b.store
            .delete(&itzma_core::schema_object::schema_path("docs"))
            .await
            .unwrap();
        let before = snapshot(&*b.store).await;

        let err = Namespace::recover(b.store.clone(), "docs").await.err();
        let Some(ItzmaError::Unrecoverable { reason, keys }) = err else {
            panic!("unexpected {err:?}");
        };
        assert!(reason.contains("schema.json"), "{reason}");
        assert_eq!(keys, ["docs/schema.json"]);
        assert_eq!(snapshot(&*b.store).await, before);
    }
}

/// A segment that does not decode, with the WAL behind it intact, costs nothing
/// but the fold: the chain ends before it and the WAL is the tail.
#[tokio::test]
async fn a_damaged_segment_with_its_wal_intact_shortens_the_chain() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = build(b.store.clone(), Stage::Deltas).await;
        let expected = hits(&ns).await;
        b.store
            .put(&segment_key(2, 2), Bytes::from_static(b"garbage").into())
            .await
            .unwrap();
        lose_manifest(&*b.store).await;

        let (recovered, report) = Namespace::recover(b.store.clone(), "docs").await.unwrap();
        assert_eq!(report.live_segments, [segment::segment_id(1, 1)]);
        assert_eq!(report.folded_through, 1);
        assert_eq!(report.tail_objects, 4);
        assert_eq!(report.unused_segments, 4);
        assert_eq!(hits(&recovered).await, expected);
        assert_consistent(b.store.clone(), DIM, L2).await;

        // The next fold rebuilds the range, and the collector clears the bad
        // object with the other dead deltas.
        recovered.compact(&ROOMY).await.unwrap();
        assert_eq!(hits(&recovered).await, expected);
        assert_eq!(recovered.gc(&NOW).await.unwrap().segments_deleted, 4);
        assert_consistent(b.store.clone(), DIM, L2).await;
    }
}

/// A merged segment that is damaged while the deltas it replaced are still
/// there: the chain falls back to the deltas.
#[tokio::test]
async fn a_damaged_merge_falls_back_to_the_deltas_it_replaced() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = build(b.store.clone(), Stage::MergedWithDeadDeltas).await;
        let expected = hits(&ns).await;
        b.store
            .put(&segment_key(1, 5), Bytes::from_static(b"garbage").into())
            .await
            .unwrap();
        lose_manifest(&*b.store).await;

        let (recovered, report) = Namespace::recover(b.store.clone(), "docs").await.unwrap();
        let deltas: Vec<String> = (1..=5).map(|n| segment::segment_id(n, n)).collect();
        assert_eq!(report.live_segments, deltas);
        assert_eq!(report.folded_through, 5);
        assert_eq!(hits(&recovered).await, expected);
        assert_consistent(b.store.clone(), DIM, L2).await;

        // A merge would run into the bad object at its key. It has to go first.
        b.store.delete(&segment_key(1, 5)).await.unwrap();
        recovered.compact(&TIGHT).await.unwrap();
        assert_eq!(hits(&recovered).await, expected);
        assert_consistent(b.store.clone(), DIM, L2).await;
    }
}

/// Both copies of a range are gone: the segment is damaged and the collector
/// has fenced its WAL objects.
#[tokio::test]
async fn a_damaged_segment_whose_wal_is_fenced_is_unrecoverable_until_restored() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", schema(DIM, L2))
            .await
            .unwrap();
        for n in [1, 2, 4] {
            op(&ns, n).await;
            ns.compact(&ROOMY).await.unwrap();
        }
        ns.gc(&FENCING).await.unwrap();
        let expected = hits(&ns).await;
        let original = bytes_at(&*b.store, &segment_key(2, 2)).await;
        b.store
            .put(&segment_key(2, 2), Bytes::from_static(b"garbage").into())
            .await
            .unwrap();
        lose_manifest(&*b.store).await;
        let before = snapshot(&*b.store).await;

        let err = Namespace::recover(b.store.clone(), "docs").await.err();
        let Some(ItzmaError::Unrecoverable { reason, keys }) = err else {
            panic!("unexpected {err:?}");
        };
        assert!(reason.contains("seqs 2 through 3"), "{reason}");
        assert!(keys.contains(&segment_key(2, 2).to_string()), "{keys:?}");
        assert!(
            keys.contains(&wal::wal_path("docs", 2).to_string()),
            "{keys:?}"
        );
        assert_eq!(snapshot(&*b.store).await, before, "nothing may be written");

        // Restore what the error named, and the rerun works.
        b.store
            .put(&segment_key(2, 2), original.into())
            .await
            .unwrap();
        let (recovered, report) = Namespace::recover(b.store.clone(), "docs").await.unwrap();
        assert_eq!(report.folded_through, 3);
        assert_eq!(hits(&recovered).await, expected);
        assert_consistent(b.store.clone(), DIM, L2).await;
    }
}

/// A WAL object that is gone past the end of the chain, with nothing that
/// covers it.
#[tokio::test]
async fn a_missing_wal_object_past_the_chain_is_unrecoverable_until_restored() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", schema(DIM, L2))
            .await
            .unwrap();
        op(&ns, 1).await;
        op(&ns, 2).await;
        ns.compact(&ROOMY).await.unwrap();
        op(&ns, 3).await;
        op(&ns, 4).await;
        op(&ns, 5).await;
        let expected = hits(&ns).await;
        let path = wal::wal_path("docs", 4);
        let original = bytes_at(&*b.store, &path).await;
        b.store.delete(&path).await.unwrap();
        lose_manifest(&*b.store).await;
        let before = snapshot(&*b.store).await;

        let err = Namespace::recover(b.store.clone(), "docs").await.err();
        let Some(ItzmaError::Unrecoverable { reason, keys }) = err else {
            panic!("unexpected {err:?}");
        };
        assert!(reason.starts_with("seq 4 has no readable copy"), "{reason}");
        assert_eq!(keys, [path.to_string()]);
        assert_eq!(snapshot(&*b.store).await, before);

        b.store.put(&path, original.into()).await.unwrap();
        let (recovered, report) = Namespace::recover(b.store.clone(), "docs").await.unwrap();
        assert_eq!(report.folded_through, 2);
        assert_eq!(report.tail_objects, 3);
        assert_eq!(hits(&recovered).await, expected);
    }
}

/// A fold that was written and never published leaves a segment no manifest
/// lists. It is a pure function of its range, so recovery adopts it.
#[tokio::test]
async fn an_unpublished_segment_is_adopted() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = build(b.store.clone(), Stage::Unfolded).await;
        let expected = hits(&ns).await;
        ns.compact(&ROOMY).await.unwrap();
        manifest::update(&*b.store, "docs", |m| {
            m.live_segments.clear();
            m.wal_folded_through = 0;
        })
        .await
        .unwrap();
        lose_manifest(&*b.store).await;

        let (recovered, report) = Namespace::recover(b.store.clone(), "docs").await.unwrap();
        assert_eq!(report.live_segments, [segment::segment_id(1, 5)]);
        assert_eq!(report.folded_through, 5);
        assert_eq!(report.tail_objects, 0);
        assert_eq!(hits(&recovered).await, expected);
        assert_consistent(b.store.clone(), DIM, L2).await;
    }
}

/// The sparse vectors sit in `sparse.bin` beside the data object, so a
/// recovered manifest keeps them queryable. Like `indexes`, `sparse_indexes` comes
/// back empty and a query scans the segment's sparse data until `index_sparse`
/// runs again.
#[tokio::test]
async fn a_recovered_namespace_keeps_its_sparse_vectors_and_names_no_sparse_index() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", schema(DIM, L2))
            .await
            .unwrap();
        ns.upsert(vec![Document {
            id: "a".into(),
            vector: vec![0.0, 0.0],
            attrs: None,
            sparse: Some(vec![(3, 2.0)]),
        }])
        .await
        .unwrap();
        ns.compact(&ROOMY).await.unwrap();
        ns.index_sparse(&IndexSparseOptions::default())
            .await
            .unwrap();
        assert_eq!(manifest_of(&*b.store).await.sparse_indexes.len(), 1);
        lose_manifest(&*b.store).await;

        let (recovered, _) = Namespace::recover(b.store.clone(), "docs").await.unwrap();
        assert!(manifest_of(&*b.store).await.sparse_indexes.is_empty());
        let found = recovered
            .query_sparse(&vec![(3, 1.0)], 5, &SparseQueryOptions::default())
            .await
            .unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].id, "a");
    }
}

/// A directory with an index file and no data object is not a segment, and a
/// range that only it names is not a gap.
#[tokio::test]
async fn a_segment_directory_without_data_is_ignored() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = build(b.store.clone(), Stage::Deltas).await;
        let expected = hits(&ns).await;
        let before = manifest_of(&*b.store).await;
        lose_manifest(&*b.store).await;
        let stray = format!("docs/segments/{}/ivf-1.bin", segment::segment_id(1, 9));
        b.store
            .put(
                &object_store::path::Path::from(stray.as_str()),
                Bytes::from_static(b"index").into(),
            )
            .await
            .unwrap();

        let (recovered, _) = Namespace::recover(b.store.clone(), "docs").await.unwrap();
        assert_eq!(manifest_of(&*b.store).await, before);
        assert_eq!(hits(&recovered).await, expected);
    }
}

/// A segment of a format this build does not read is a newer build's data, and
/// skipping it would make it look like damage.
#[tokio::test]
async fn a_segment_of_a_newer_format_stops_the_run() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        build(b.store.clone(), Stage::OneFold).await;
        let path = segment_key(1, 3);
        let mut bytes = bytes_at(&*b.store, &path).await.to_vec();
        bytes[4..8].copy_from_slice(&99u32.to_le_bytes());
        b.store.put(&path, Bytes::from(bytes).into()).await.unwrap();
        lose_manifest(&*b.store).await;
        let before = snapshot(&*b.store).await;

        let err = Namespace::recover(b.store.clone(), "docs").await.err();
        assert!(
            matches!(err, Some(ItzmaError::UnsupportedFormat { .. })),
            "{err:?}"
        );
        assert_eq!(snapshot(&*b.store).await, before);
    }
}

#[tokio::test]
async fn two_recoveries_at_once_write_one_manifest() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = build(b.store.clone(), Stage::Deltas).await;
        let expected = hits(&ns).await;
        let before = manifest_of(&*b.store).await;
        lose_manifest(&*b.store).await;

        let (first, second) = tokio::join!(
            Namespace::recover(b.store.clone(), "docs"),
            Namespace::recover(b.store.clone(), "docs")
        );
        let results = [first.map(|_| ()), second.map(|_| ())];
        let ok = results.iter().filter(|r| r.is_ok()).count();
        assert_eq!(ok, 1, "{results:?}");
        assert!(
            results
                .iter()
                .any(|r| matches!(r, Err(ItzmaError::ManifestExists(_))))
        );

        assert_eq!(manifest_of(&*b.store).await, before);
        let fresh = Namespace::open(b.store.clone(), "docs").await.unwrap();
        assert_eq!(hits(&fresh).await, expected);
    }
}

/// Fault every operation of a recovery, as a plain error and as a lost ack.
/// After each, either the manifest is there (the put landed and only the answer
/// was lost) or a second recovery finishes the job. Both end in the manifest
/// the namespace had before.
#[tokio::test]
async fn a_stopped_recovery_ends_with_the_same_manifest() {
    for ix in 0..cas_backends().len() {
        let counted = FaultyStore::new(cas_backends().swap_remove(ix).store);
        build(counted.clone(), Stage::Fenced).await;
        lose_manifest(&*counted).await;
        counted.reset_ops();
        Namespace::recover(dynamic(&counted), "docs").await.unwrap();
        let total = counted.ops();
        assert!(total > 5, "a recovery is more than a few operations");

        for fault in [Fault::Generic, Fault::LostAck] {
            for n in 0..total {
                eprintln!("backend {ix}: {fault:?} at operation {n} of {total}");
                let backend = cas_backends().swap_remove(ix);
                let store = FaultyStore::new(backend.store.clone());
                let ns = build(dynamic(&store), Stage::Fenced).await;
                let expected = hits(&ns).await;
                let before = manifest_of(&*store).await;
                lose_manifest(&*store).await;
                store.reset_ops();

                store.fail_nth_op(n, fault);
                let stopped = Namespace::recover(dynamic(&store), "docs").await;
                store.clear();
                assert!(stopped.is_err(), "a fault inside recover must show");

                match Namespace::open(dynamic(&store), "docs").await {
                    Ok(_) => {}
                    Err(ItzmaError::NamespaceNotFound(_)) => {
                        Namespace::recover(dynamic(&store), "docs").await.unwrap();
                    }
                    Err(other) => panic!("{other:?}"),
                }

                assert_eq!(manifest_of(&*store).await, before);
                let fresh = Namespace::open(dynamic(&store), "docs").await.unwrap();
                assert_eq!(hits(&fresh).await, expected);
                assert_consistent(dynamic(&store), DIM, L2).await;
            }
        }
    }
}

#[tokio::test]
async fn the_recover_future_can_be_spawned() {
    let store = common::memory();
    build(store.clone(), Stage::Deltas).await;
    lose_manifest(&*store).await;
    let handle = tokio::spawn(Namespace::recover(store.clone(), "docs"));
    assert!(handle.await.unwrap().is_ok());
}

/// A namespace raised to segment format 3 holds version 3
/// segments; a recovered manifest must keep level 3, or later folds write
/// version 2 again and a build that reads only version 2 passes the gate and
/// fails on the segments instead.
#[tokio::test]
async fn a_namespace_at_level_3_is_recovered_at_level_3() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", common::schema(DIM, L2))
            .await
            .unwrap();
        ns.upsert(vec![doc("a", &[1.0, 0.0])]).await.unwrap();
        ns.compact(&ROOMY).await.unwrap();
        ns.upgrade_segment_format(3).await.unwrap();
        ns.upsert(vec![doc("b", &[2.0, 0.0])]).await.unwrap();
        ns.compact(&ROOMY).await.unwrap();
        assert_eq!(common::live_segment_versions(&*b.store).await, [2, 3]);
        let before = manifest_of(&*b.store).await;
        lose_manifest(&*b.store).await;

        let (recovered, _) = Namespace::recover(b.store.clone(), "docs").await.unwrap();
        assert_eq!(manifest_of(&*b.store).await, before);
        assert_eq!(manifest_of(&*b.store).await.segment_format, 3);
        recovered.upsert(vec![doc("c", &[3.0, 0.0])]).await.unwrap();
        recovered.compact(&ROOMY).await.unwrap();
        assert_eq!(common::live_segment_versions(&*b.store).await, [2, 3, 3]);
        assert_consistent(b.store.clone(), DIM, L2).await;
    }
}

/// An unpublished version 3 object counts too: the run that wrote it was at
/// level 3, and a rerun of its range at level 2 would refuse it.
#[tokio::test]
async fn an_unused_version_3_segment_keeps_the_level_at_3() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", common::schema(DIM, L2))
            .await
            .unwrap();
        ns.upsert(vec![doc("a", &[1.0, 0.0])]).await.unwrap();
        let mut layer = itzma_core::layer::Layer::new();
        layer.apply(itzma_core::WalOp::Upsert(doc("a", &[1.0, 0.0])));
        b.store
            .put(
                &segment_key(1, 1),
                segment::encode_as(&layer, DIM, 3).unwrap().into(),
            )
            .await
            .unwrap();
        lose_manifest(&*b.store).await;
        Namespace::recover(b.store.clone(), "docs").await.unwrap();
        assert_eq!(manifest_of(&*b.store).await.segment_format, 3);
    }
}
