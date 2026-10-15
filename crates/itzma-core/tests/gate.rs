//! The manifest gate: a namespace whose `wal_format` is above what
//! this build reads is refused by every operation, and nothing is written.

mod common;

use std::sync::Arc;

use common::{doc, schema, snapshot};
use itzma_core::manifest;
use itzma_core::wal::FORMAT_VERSION;
use itzma_core::{CompactionOptions, GcOptions, ItzmaError, Metric, Namespace};
use object_store::{ObjectStore, ObjectStoreExt};

/// What a newer build does when it upgrades a namespace: rewrite the manifest
/// with a higher `wal_format`. A plain put, so it works on every backend.
async fn raise_level(store: &dyn ObjectStore, namespace: &str) {
    let (mut m, _) = manifest::load(store, namespace).await.unwrap();
    m.wal_format = FORMAT_VERSION + 1;
    store
        .put(
            &manifest::manifest_path(namespace),
            serde_json::to_vec(&m).unwrap().into(),
        )
        .await
        .unwrap();
}

fn assert_too_new(err: ItzmaError) {
    assert!(
        matches!(
            err,
            ItzmaError::WalFormatTooNew { found, supported }
                if found == FORMAT_VERSION + 1 && supported == FORMAT_VERSION
        ),
        "unexpected error: {err:?}"
    );
}

async fn prepared(store: Arc<dyn ObjectStore>) -> Namespace {
    let ns = Namespace::create(store, "docs", schema(2, Metric::L2))
        .await
        .unwrap();
    ns.upsert(vec![doc("a", &[0.0, 0.0]), doc("b", &[1.0, 1.0])])
        .await
        .unwrap();
    ns.upsert(vec![doc("c", &[2.0, 2.0])]).await.unwrap();
    ns
}

#[tokio::test]
async fn open_query_and_gc_are_refused_and_write_nothing() {
    for b in common::backends() {
        eprintln!("backend: {}", b.name);
        let earlier = prepared(b.store.clone()).await;
        raise_level(&*b.store, "docs").await;
        let before = snapshot(&*b.store).await;

        let opened = Namespace::open(b.store.clone(), "docs").await;
        assert_too_new(opened.err().unwrap());
        assert_too_new(earlier.query(&[0.0, 0.0], 3).await.unwrap_err());
        let gc = earlier.gc(&GcOptions::default()).await;
        assert_too_new(gc.unwrap_err());

        assert_eq!(snapshot(&*b.store).await, before, "the store changed");
    }
}

#[tokio::test]
async fn compact_and_manifest_updates_are_refused_and_write_nothing() {
    for b in common::cas_backends() {
        eprintln!("backend: {}", b.name);
        let earlier = prepared(b.store.clone()).await;
        earlier
            .compact(&CompactionOptions::default())
            .await
            .unwrap();
        earlier.upsert(vec![doc("d", &[3.0, 3.0])]).await.unwrap();
        raise_level(&*b.store, "docs").await;
        let before = snapshot(&*b.store).await;

        let compact = earlier.compact(&CompactionOptions::default()).await;
        assert_too_new(compact.unwrap_err());
        let update = manifest::update(&*b.store, "docs", |m| m.wal_folded_through += 1).await;
        assert_too_new(update.unwrap_err());

        assert_eq!(snapshot(&*b.store).await, before, "the store changed");
    }
}

/// The gate is on the manifest, not on any one call, so a handle that opened
/// before the level moved learns about it on its next read rather than as a
/// decode error on an object it cannot parse.
#[tokio::test]
async fn a_handle_opened_before_the_level_moved_finds_out_on_its_next_read() {
    for b in common::backends() {
        eprintln!("backend: {}", b.name);
        let ns = prepared(b.store.clone()).await;
        assert_eq!(ns.query(&[0.0, 0.0], 3).await.unwrap().len(), 3);
        raise_level(&*b.store, "docs").await;
        assert_too_new(ns.query(&[0.0, 0.0], 3).await.unwrap_err());
    }
}
