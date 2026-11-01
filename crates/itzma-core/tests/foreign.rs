//! Objects this build did not write: another build's magic numbers in place
//! of this one's, or a manifest at a WAL format level above what this build
//! reads. Every read path must refuse them cleanly, never panic, and leave
//! the store as it was.

mod common;

use std::time::Duration;

use common::{assert_untouched, cas_backends, doc, schema, snapshot};
use futures::TryStreamExt;
use itzma_core::{
    AttrSchema, AttrType, CompactionOptions, FilterOptions, GcOptions, IndexOptions, ItzmaError,
    Metric, Namespace, Predicate, Schema, inspect, manifest,
};
use object_store::{ObjectStore, ObjectStoreExt};

const DIM: usize = 2;

fn typed() -> Schema {
    schema(DIM, Metric::L2).with_attrs(AttrSchema::new().field("n", AttrType::Number))
}

/// A namespace with an indexed segment and a one-object WAL tail.
async fn built(store: std::sync::Arc<dyn ObjectStore>) -> Namespace {
    let ns = Namespace::create(store, "docs", typed()).await.unwrap();
    let docs = (0..20)
        .map(|i| {
            let mut d = doc(&format!("d{i:02}"), &[i as f32, 1.0]);
            d.attrs = Some(serde_json::json!({ "n": i }));
            d
        })
        .collect();
    ns.upsert(docs).await.unwrap();
    ns.compact(&CompactionOptions::default()).await.unwrap();
    ns.index(&IndexOptions::default().min_rows(0))
        .await
        .unwrap();
    ns.upsert(vec![doc("tail", &[0.5, 0.5])]).await.unwrap();
    ns
}

/// The first object under `prefix` whose name ends with `suffix`.
async fn object(store: &dyn ObjectStore, prefix: &str, suffix: &str) -> object_store::path::Path {
    let listed: Vec<_> = store
        .list(Some(&prefix.into()))
        .try_collect()
        .await
        .unwrap();
    listed
        .into_iter()
        .map(|meta| meta.location)
        .find(|path| path.as_ref().ends_with(suffix))
        .unwrap_or_else(|| panic!("no object ending in {suffix} under {prefix}"))
}

/// Replace the first four bytes of `path` with `magic`, another build's.
async fn relabel(store: &dyn ObjectStore, path: &object_store::path::Path, magic: [u8; 4]) {
    let mut bytes = store
        .get(path)
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap()
        .to_vec();
    bytes[..4].copy_from_slice(&magic);
    store.put(path, bytes.into()).await.unwrap();
}

/// Every read path refuses or reports, and the store is unchanged.
async fn every_read_refuses(store: &dyn ObjectStore, ns: &Namespace, what: &str) {
    let before = snapshot(store).await;
    let query = ns.query(&[0.0, 0.0], 5).await;
    assert!(query.is_err(), "{what}: query {query:?}");
    let filter = ns
        .filter(
            &Predicate::number("n", itzma_core::Compare::Lt, 100.0),
            &FilterOptions::default(),
        )
        .await;
    assert!(filter.is_err(), "{what}: filter {filter:?}");
    // `inspect` reports what it finds; it only has to come back.
    let _ = inspect::report(store, "docs").await;
    let dry = GcOptions {
        min_age: Duration::ZERO,
        dry_run: true,
        fence_wal: false,
    };
    let _ = ns.gc(&dry).await;
    assert_untouched(store, &before).await;
}

#[tokio::test]
async fn another_builds_wal_magic_is_refused() {
    for b in cas_backends() {
        let ns = built(b.store.clone()).await;
        let tail = object(&*b.store, "docs/wal", "00000000000000000002.json").await;
        relabel(&*b.store, &tail, *b"CALW").await;
        every_read_refuses(&*b.store, &ns, b.name).await;
        let err = ns
            .compact(&CompactionOptions::default())
            .await
            .err()
            .unwrap();
        assert!(matches!(err, ItzmaError::CorruptWal { .. }), "{err:?}");
    }
}

#[tokio::test]
async fn another_builds_fence_magic_is_refused() {
    for b in cas_backends() {
        let ns = built(b.store.clone()).await;
        let tail = object(&*b.store, "docs/wal", "00000000000000000002.json").await;
        let mut fence = vec![b'C', b'A', b'L', b'F'];
        fence.extend_from_slice(&1u32.to_le_bytes());
        b.store.put(&tail, fence.into()).await.unwrap();
        every_read_refuses(&*b.store, &ns, b.name).await;
    }
}

#[tokio::test]
async fn another_builds_segment_magic_is_refused() {
    for b in cas_backends() {
        let ns = built(b.store.clone()).await;
        let data = object(&*b.store, "docs/segments", "/data.bin").await;
        relabel(&*b.store, &data, *b"CALS").await;
        every_read_refuses(&*b.store, &ns, b.name).await;
    }
}

#[tokio::test]
async fn another_builds_index_magic_is_refused() {
    for b in cas_backends() {
        let ns = built(b.store.clone()).await;
        let index = object(&*b.store, "docs/segments", "/index-1.bin").await;
        relabel(&*b.store, &index, *b"CALI").await;
        let before = snapshot(&*b.store).await;
        let err = ns.query(&[0.0, 0.0], 5).await.err().unwrap();
        assert!(matches!(err, ItzmaError::CorruptIndex { .. }), "{err:?}");
        assert_untouched(&*b.store, &before).await;
    }
}

#[tokio::test]
async fn a_manifest_at_a_wal_level_this_build_does_not_read_is_refused() {
    for b in cas_backends() {
        built(b.store.clone()).await;
        manifest::update(&*b.store, "docs", |m| m.wal_format = 6)
            .await
            .unwrap();
        let before = snapshot(&*b.store).await;
        let err = Namespace::open(b.store.clone(), "docs")
            .await
            .err()
            .unwrap();
        assert!(
            matches!(err, ItzmaError::WalFormatTooNew { found: 6, .. }),
            "{err:?}"
        );
        assert_untouched(&*b.store, &before).await;
    }
}
