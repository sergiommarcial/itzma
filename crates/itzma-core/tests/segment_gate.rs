//! The segment format gate: a namespace whose
//! `segment_format` is above what this build reads is refused by every
//! operation, and compaction writes the segment version the namespace's level
//! allows.

mod common;

use std::sync::Arc;

use common::{assert_consistent, cas_backends, doc, naive_top_k, schema, snapshot};
use itzma_core::layer::Layer;
use itzma_core::segment::FORMAT_VERSION;
use itzma_core::{
    Compaction, CompactionOptions, Document, GcOptions, ItzmaError, Metric, Namespace, WalOp,
    manifest, segment, wal,
};
use object_store::{ObjectStore, ObjectStoreExt};

const DIM: usize = 2;
const ROOMY: CompactionOptions = CompactionOptions {
    max_segments: 8,
    max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
};
const TIGHT: CompactionOptions = CompactionOptions {
    max_segments: 2,
    max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
};

/// Rewrite the manifest object as raw JSON, so a level the build refuses can
/// be written without going through `manifest::load`. A plain put, so it
/// works on every backend.
async fn edit_manifest(
    store: &dyn ObjectStore,
    change: impl FnOnce(&mut serde_json::Map<String, serde_json::Value>),
) {
    let path = manifest::manifest_path("docs");
    let raw = store.get(&path).await.unwrap().bytes().await.unwrap();
    let mut json: serde_json::Value = serde_json::from_slice(&raw).unwrap();
    change(json.as_object_mut().unwrap());
    store
        .put(&path, serde_json::to_vec(&json).unwrap().into())
        .await
        .unwrap();
}

async fn set_level(store: &dyn ObjectStore, level: u32) {
    edit_manifest(store, |m| {
        m.insert("segment_format".into(), level.into());
    })
    .await;
}

/// The manifest as a build from before this level wrote it.
async fn remove_level(store: &dyn ObjectStore) {
    edit_manifest(store, |m| {
        m.remove("segment_format");
    })
    .await;
}

async fn level(store: &dyn ObjectStore) -> u32 {
    manifest::load(store, "docs")
        .await
        .unwrap()
        .0
        .segment_format
}

/// The format version stamped in each live segment, oldest first.
async fn live_versions(store: &dyn ObjectStore) -> Vec<u32> {
    let (m, _) = manifest::load(store, "docs").await.unwrap();
    let mut versions = Vec::new();
    for id in &m.live_segments {
        let bytes = store
            .get(&segment::segment_path("docs", id))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        versions.push(u32::from_le_bytes(bytes[4..8].try_into().unwrap()));
    }
    versions
}

fn assert_too_new(err: ItzmaError) {
    assert!(
        matches!(
            err,
            ItzmaError::SegmentFormatTooNew { found, supported }
                if found == FORMAT_VERSION + 1 && supported == FORMAT_VERSION
        ),
        "unexpected error: {err:?}"
    );
}

fn point(i: usize) -> [f32; DIM] {
    [i as f32, (i * 2) as f32]
}

async fn create(store: Arc<dyn ObjectStore>) -> Namespace {
    Namespace::create(store, "docs", schema(DIM, Metric::L2))
        .await
        .unwrap()
}

/// Upsert `ids` and fold them into a segment.
async fn fold(ns: &Namespace, ids: std::ops::Range<usize>, options: &CompactionOptions) {
    ns.upsert(docs_of(ids)).await.unwrap();
    let outcome = ns.compact(options).await.unwrap();
    assert!(
        matches!(outcome, Compaction::Published { .. }),
        "{outcome:?}"
    );
}

fn docs_of(ids: std::ops::Range<usize>) -> Vec<Document> {
    ids.map(|i| doc(&format!("d{i}"), &point(i))).collect()
}

/// A query for every document equals a naive scan of what was written.
async fn assert_query_matches(ns: &Namespace, docs: &[Document]) {
    let query = [1.5, 0.5];
    let got: Vec<String> = ns
        .query(&query, docs.len())
        .await
        .unwrap()
        .into_iter()
        .map(|hit| hit.id)
        .collect();
    assert_eq!(got, naive_top_k(Metric::L2, &query, docs, docs.len()));
}

#[tokio::test]
async fn a_new_namespace_starts_at_the_default_level_not_the_newest() {
    for b in cas_backends() {
        let ns = create(b.store.clone()).await;
        assert_eq!(level(&*b.store).await, segment::DEFAULT_FORMAT);
        const { assert!(segment::DEFAULT_FORMAT <= FORMAT_VERSION) };
        fold(&ns, 0..3, &ROOMY).await;
        assert_eq!(live_versions(&*b.store).await, [segment::DEFAULT_FORMAT]);
    }
}

#[tokio::test]
async fn a_new_manifest_object_stores_the_default_level() {
    for b in common::backends() {
        eprintln!("backend: {}", b.name);
        create(b.store.clone()).await;
        assert_eq!(level(&*b.store).await, segment::DEFAULT_FORMAT);
        let raw = b
            .store
            .get(&manifest::manifest_path("docs"))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        assert_eq!(json["segment_format"], segment::DEFAULT_FORMAT);
    }
}

#[tokio::test]
async fn open_query_and_gc_are_refused_and_write_nothing() {
    for b in common::backends() {
        eprintln!("backend: {}", b.name);
        let earlier = create(b.store.clone()).await;
        earlier.upsert(vec![doc("a", &[0.0, 0.0])]).await.unwrap();
        set_level(&*b.store, FORMAT_VERSION + 1).await;
        let before = snapshot(&*b.store).await;

        let opened = Namespace::open(b.store.clone(), "docs").await;
        assert_too_new(opened.err().unwrap());
        assert_too_new(earlier.query(&[0.0, 0.0], 3).await.unwrap_err());
        assert_too_new(earlier.gc(&GcOptions::default()).await.unwrap_err());

        assert_eq!(snapshot(&*b.store).await, before, "the store changed");
    }
}

#[tokio::test]
async fn compact_upgrade_and_manifest_updates_are_refused_and_write_nothing() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let earlier = create(b.store.clone()).await;
        fold(&earlier, 0..3, &ROOMY).await;
        earlier
            .upsert(vec![doc("tail", &[9.0, 9.0])])
            .await
            .unwrap();
        set_level(&*b.store, FORMAT_VERSION + 1).await;
        let before = snapshot(&*b.store).await;

        assert_too_new(earlier.compact(&ROOMY).await.unwrap_err());
        assert_too_new(earlier.upgrade_segment_format(2).await.unwrap_err());
        let update = manifest::update(&*b.store, "docs", |m| m.wal_folded_through += 1).await;
        assert_too_new(update.unwrap_err());

        assert_eq!(snapshot(&*b.store).await, before, "the store changed");
    }
}

/// The gate is on the manifest, so a handle opened before the level moved finds
/// out on its next read instead of as a decode error.
#[tokio::test]
async fn a_handle_opened_before_the_level_moved_finds_out_on_its_next_read() {
    for b in common::backends() {
        eprintln!("backend: {}", b.name);
        let ns = create(b.store.clone()).await;
        ns.upsert(vec![doc("a", &[0.0, 0.0])]).await.unwrap();
        assert_eq!(ns.query(&[0.0, 0.0], 3).await.unwrap().len(), 1);
        set_level(&*b.store, FORMAT_VERSION + 1).await;
        assert_too_new(ns.query(&[0.0, 0.0], 3).await.unwrap_err());
    }
}

/// Both gates trip on one manifest: the WAL check comes first, so what an
/// operator sees is the older, more basic incompatibility.
#[tokio::test]
async fn the_wal_gate_is_checked_before_the_segment_gate() {
    for b in common::backends() {
        eprintln!("backend: {}", b.name);
        create(b.store.clone()).await;
        set_level(&*b.store, FORMAT_VERSION + 1).await;
        edit_manifest(&*b.store, |m| {
            m.insert("wal_format".into(), (wal::FORMAT_VERSION + 1).into());
        })
        .await;
        let err = Namespace::open(b.store.clone(), "docs")
            .await
            .err()
            .unwrap();
        assert!(matches!(err, ItzmaError::WalFormatTooNew { .. }), "{err:?}");
    }
}

#[tokio::test]
async fn a_manifest_without_the_field_is_refused() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        create(b.store.clone()).await;
        remove_level(&*b.store).await;
        let err = Namespace::open(b.store.clone(), "docs")
            .await
            .err()
            .unwrap();
        assert!(matches!(err, ItzmaError::Serde { .. }), "{err:?}");
    }
}

#[tokio::test]
async fn a_namespace_created_by_this_build_folds_to_the_default_version() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = create(b.store.clone()).await;
        fold(&ns, 0..4, &ROOMY).await;
        assert_eq!(live_versions(&*b.store).await, [segment::DEFAULT_FORMAT]);
    }
}

#[tokio::test]
async fn the_raise_is_monotonic() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = create(b.store.clone()).await;

        ns.upgrade_segment_format(3).await.unwrap();
        assert_eq!(level(&*b.store).await, 3);

        // Lower and equal levels are no-ops that write nothing.
        let before = snapshot(&*b.store).await;
        ns.upgrade_segment_format(2).await.unwrap();
        ns.upgrade_segment_format(0).await.unwrap();
        ns.upgrade_segment_format(3).await.unwrap();
        assert_eq!(snapshot(&*b.store).await, before, "the store changed");

        // Above what the build reads is refused, and writes nothing.
        assert_too_new(
            ns.upgrade_segment_format(FORMAT_VERSION + 1)
                .await
                .unwrap_err(),
        );
        assert_eq!(snapshot(&*b.store).await, before, "the store changed");
        assert_eq!(level(&*b.store).await, 3);
    }
}

#[tokio::test]
async fn two_raises_racing_both_succeed() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let first = create(b.store.clone()).await;
        let second = Namespace::open(b.store.clone(), "docs").await.unwrap();

        let a = tokio::spawn(async move { first.upgrade_segment_format(3).await });
        let c = tokio::spawn(async move { second.upgrade_segment_format(3).await });
        a.await.unwrap().unwrap();
        c.await.unwrap().unwrap();

        assert_eq!(level(&*b.store).await, 3);
    }
}

#[tokio::test]
async fn at_level_3_folds_write_version_3_beside_older_segments() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = create(b.store.clone()).await;
        fold(&ns, 0..4, &ROOMY).await;
        ns.upgrade_segment_format(3).await.unwrap();
        fold(&ns, 4..8, &ROOMY).await;
        assert_eq!(live_versions(&*b.store).await, [2, 3]);
        assert_query_matches(&ns, &docs_of(0..8)).await;
        assert_consistent(b.store.clone(), DIM, Metric::L2).await;
    }
}

#[tokio::test]
async fn a_merge_at_level_3_rewrites_version_2_segments_as_version_3() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = create(b.store.clone()).await;
        fold(&ns, 0..3, &ROOMY).await;
        fold(&ns, 3..6, &ROOMY).await;
        ns.upgrade_segment_format(3).await.unwrap();
        fold(&ns, 6..9, &TIGHT).await;
        let versions = live_versions(&*b.store).await;
        assert!(versions.iter().all(|v| *v == 3), "{versions:?}");
        assert_query_matches(&ns, &docs_of(0..9)).await;
        assert_consistent(b.store.clone(), DIM, Metric::L2).await;
    }
}

#[tokio::test]
async fn an_orphan_version_2_segment_blocks_its_range_after_the_raise_to_3() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = create(b.store.clone()).await;
        ns.upsert(vec![doc("a", &[0.0, 0.0]), doc("b", &[1.0, 1.0])])
            .await
            .unwrap();

        // What a dead run at level 2 left behind: the fold of seq 1 as
        // version 2, never published.
        let mut layer = Layer::new();
        layer.apply(WalOp::Upsert(doc("a", &[0.0, 0.0])));
        layer.apply(WalOp::Upsert(doc("b", &[1.0, 1.0])));
        layer.drop_tombstones();
        let orphan_path = segment::segment_path("docs", &segment::segment_id(1, 1));
        b.store
            .put(
                &orphan_path,
                segment::encode_as(&layer, DIM, 2).unwrap().into(),
            )
            .await
            .unwrap();

        ns.upgrade_segment_format(3).await.unwrap();
        let before = snapshot(&*b.store).await;

        let err = ns.compact(&ROOMY).await.unwrap_err();
        assert!(matches!(err, ItzmaError::CorruptSegment { .. }), "{err:?}");
        assert_eq!(snapshot(&*b.store).await, before, "nothing is published");

        b.store.delete(&orphan_path).await.unwrap();
        let outcome = ns.compact(&ROOMY).await.unwrap();
        assert!(
            matches!(outcome, Compaction::Published { .. }),
            "{outcome:?}"
        );
        assert_eq!(live_versions(&*b.store).await, [3]);
        assert_query_matches(&ns, &[doc("a", &[0.0, 0.0]), doc("b", &[1.0, 1.0])]).await;
        assert_consistent(b.store.clone(), DIM, Metric::L2).await;
    }
}
