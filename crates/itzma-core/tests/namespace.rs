mod common;

use std::sync::Arc;

use itzma_core::manifest::{self, FORMAT_VERSION};
use itzma_core::{ItzmaError, Metric, Namespace};
use object_store::ObjectStoreExt;

use common::{backends, doc, schema};

fn ids(hits: &[itzma_core::Hit]) -> Vec<&str> {
    hits.iter().map(|h| h.id.as_str()).collect()
}

#[tokio::test]
async fn create_upsert_query_roundtrip() {
    for b in backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", schema(2, Metric::L2))
            .await
            .unwrap();
        ns.upsert(vec![doc("near", &[1.0, 0.0]), doc("far", &[9.0, 9.0])])
            .await
            .unwrap();
        let hits = ns.query(&[0.0, 0.0], 1).await.unwrap();
        assert_eq!(ids(&hits), ["near"]);
    }
}

#[tokio::test]
async fn attrs_come_back_on_hits() {
    let store = common::memory();
    let with_title = schema(2, Metric::L2)
        .with_attrs(itzma_core::AttrSchema::new().field("title", itzma_core::AttrType::Text));
    let ns = Namespace::create(store, "docs", with_title).await.unwrap();
    let mut d = doc("a", &[1.0, 1.0]);
    d.attrs = Some(serde_json::json!({"title": "hello"}));
    ns.upsert(vec![d]).await.unwrap();
    let hits = ns.query(&[1.0, 1.0], 1).await.unwrap();
    assert_eq!(hits[0].attrs, Some(serde_json::json!({"title": "hello"})));
}

#[tokio::test]
async fn upsert_replaces_and_delete_removes_across_wal_objects() {
    for b in backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", schema(2, Metric::L2))
            .await
            .unwrap();
        ns.upsert(vec![doc("a", &[100.0, 100.0]), doc("b", &[1.0, 1.0])])
            .await
            .unwrap();
        // Replace "a" so it becomes the nearest to the origin.
        ns.upsert(vec![doc("a", &[0.5, 0.5])]).await.unwrap();
        let hits = ns.query(&[0.0, 0.0], 2).await.unwrap();
        assert_eq!(ids(&hits), ["a", "b"]);
        assert_eq!(hits.len(), 2, "replace must not duplicate the id");

        ns.delete(vec!["a".into()]).await.unwrap();
        let hits = ns.query(&[0.0, 0.0], 5).await.unwrap();
        assert_eq!(ids(&hits), ["b"]);
    }
}

#[tokio::test]
async fn upsert_after_delete_brings_the_id_back() {
    let store = common::memory();
    let ns = Namespace::create(store, "docs", schema(1, Metric::L2))
        .await
        .unwrap();
    ns.upsert(vec![doc("a", &[1.0])]).await.unwrap();
    ns.delete(vec!["a".into()]).await.unwrap();
    ns.upsert(vec![doc("a", &[2.0])]).await.unwrap();
    let hits = ns.query(&[0.0], 5).await.unwrap();
    assert_eq!(ids(&hits), ["a"]);
    assert_eq!(hits[0].distance, 4.0);
}

#[tokio::test]
async fn read_your_writes_on_a_separate_handle() {
    for b in backends() {
        eprintln!("backend: {}", b.name);
        let writer = Namespace::create(b.store.clone(), "docs", schema(1, Metric::L2))
            .await
            .unwrap();
        let reader = Namespace::open(b.store.clone(), "docs").await.unwrap();
        for i in 0..5 {
            let id = format!("d{i}");
            writer.upsert(vec![doc(&id, &[i as f32])]).await.unwrap();
            let hits = reader.query(&[i as f32], 1).await.unwrap();
            assert_eq!(hits[0].id, id, "write {i} not visible right after append");
        }
    }
}

#[tokio::test]
async fn empty_namespace_queries_return_nothing() {
    let store = common::memory();
    let ns = Namespace::create(store, "docs", schema(2, Metric::Cosine))
        .await
        .unwrap();
    assert!(ns.query(&[1.0, 0.0], 3).await.unwrap().is_empty());
}

#[tokio::test]
async fn create_twice_and_open_missing() {
    let store = common::memory();
    Namespace::create(store.clone(), "docs", schema(2, Metric::L2))
        .await
        .unwrap();
    let err = Namespace::create(store.clone(), "docs", schema(2, Metric::L2))
        .await
        .err()
        .unwrap();
    assert!(matches!(err, ItzmaError::NamespaceExists(_)));
    let err = Namespace::open(store, "other").await.err().unwrap();
    assert!(matches!(err, ItzmaError::NamespaceNotFound(_)));
}

#[tokio::test]
async fn open_recovers_schema() {
    let store = common::memory();
    Namespace::create(store.clone(), "docs", schema(4, Metric::Dot))
        .await
        .unwrap();
    let ns = Namespace::open(store, "docs").await.unwrap();
    assert_eq!(ns.schema(), &schema(4, Metric::Dot));
    assert_eq!(ns.id(), "docs");
}

#[tokio::test]
async fn namespaces_are_isolated() {
    let store = common::memory();
    let a = Namespace::create(store.clone(), "a", schema(1, Metric::L2))
        .await
        .unwrap();
    let b = Namespace::create(store, "b", schema(1, Metric::L2))
        .await
        .unwrap();
    a.upsert(vec![doc("only-in-a", &[1.0])]).await.unwrap();
    assert!(b.query(&[1.0], 5).await.unwrap().is_empty());
}

#[tokio::test]
async fn invalid_namespace_ids_are_rejected() {
    let store = common::memory();
    let too_long = "a".repeat(65);
    for bad in [
        "",
        "Upper",
        "has/slash",
        "has space",
        "dot.dot",
        too_long.as_str(),
    ] {
        let err = Namespace::create(store.clone(), bad, schema(2, Metric::L2))
            .await
            .err()
            .unwrap();
        assert!(matches!(err, ItzmaError::InvalidNamespaceId(_)), "{bad:?}");
    }
    assert!(
        Namespace::create(store, &"a".repeat(64), schema(2, Metric::L2))
            .await
            .is_ok()
    );
}

#[tokio::test]
async fn zero_dim_schema_is_rejected() {
    let store = common::memory();
    let err = Namespace::create(store, "docs", schema(0, Metric::L2))
        .await
        .err()
        .unwrap();
    assert!(matches!(err, ItzmaError::InvalidSchema(_)));
}

#[tokio::test]
async fn bad_writes_are_rejected_before_touching_storage() {
    let store = common::memory();
    let ns = Namespace::create(store.clone(), "docs", schema(2, Metric::Cosine))
        .await
        .unwrap();

    let err = ns.upsert(vec![doc("a", &[1.0])]).await.unwrap_err();
    assert!(matches!(
        err,
        ItzmaError::DimensionMismatch {
            expected: 2,
            got: 1
        }
    ));
    let err = ns
        .upsert(vec![doc("a", &[1.0, f32::NAN])])
        .await
        .unwrap_err();
    assert!(matches!(err, ItzmaError::InvalidVector(_)));
    let err = ns.upsert(vec![doc("a", &[0.0, 0.0])]).await.unwrap_err();
    assert!(matches!(err, ItzmaError::InvalidVector(_)));
    let err = ns.upsert(vec![]).await.unwrap_err();
    assert!(matches!(err, ItzmaError::EmptyBatch));
    let err = ns.delete(vec![]).await.unwrap_err();
    assert!(matches!(err, ItzmaError::EmptyBatch));

    // One bad doc poisons the whole batch: nothing was written.
    let err = ns
        .upsert(vec![doc("ok", &[1.0, 1.0]), doc("bad", &[1.0])])
        .await
        .unwrap_err();
    assert!(matches!(err, ItzmaError::DimensionMismatch { .. }));
    assert!(ns.query(&[1.0, 1.0], 5).await.unwrap().is_empty());
}

#[tokio::test]
async fn bad_ids_are_rejected_before_touching_storage() {
    let store = common::memory();
    let ns = Namespace::create(store.clone(), "docs", schema(2, Metric::L2))
        .await
        .unwrap();

    let err = ns.upsert(vec![doc("", &[1.0, 1.0])]).await.unwrap_err();
    assert!(matches!(err, ItzmaError::InvalidId(_)));

    let too_long = "a".repeat(itzma_core::document::MAX_ID_BYTES + 1);
    let err = ns
        .upsert(vec![doc(&too_long, &[1.0, 1.0])])
        .await
        .unwrap_err();
    assert!(matches!(err, ItzmaError::InvalidId(_)));

    let err = ns.delete(vec!["".into()]).await.unwrap_err();
    assert!(matches!(err, ItzmaError::InvalidId(_)));
    let err = ns.delete(vec![too_long]).await.unwrap_err();
    assert!(matches!(err, ItzmaError::InvalidId(_)));

    // A good id ahead of a bad one still poisons the whole delete batch.
    ns.upsert(vec![doc("ok", &[1.0, 1.0])]).await.unwrap();
    let err = ns.delete(vec!["ok".into(), "".into()]).await.unwrap_err();
    assert!(matches!(err, ItzmaError::InvalidId(_)));
    assert_eq!(ids(&ns.query(&[1.0, 1.0], 5).await.unwrap()), ["ok"]);
}

#[tokio::test]
async fn bad_queries_are_rejected() {
    let store = common::memory();
    let ns = Namespace::create(store, "docs", schema(2, Metric::Cosine))
        .await
        .unwrap();
    assert!(matches!(
        ns.query(&[1.0], 1).await.unwrap_err(),
        ItzmaError::DimensionMismatch { .. }
    ));
    assert!(matches!(
        ns.query(&[0.0, 0.0], 1).await.unwrap_err(),
        ItzmaError::InvalidVector(_)
    ));
}

#[tokio::test]
async fn open_rejects_unknown_manifest_format() {
    let store = common::memory();
    Namespace::create(store.clone(), "docs", schema(2, Metric::L2))
        .await
        .unwrap();
    let bad = format!(
        r#"{{"format_version": {}, "schema": {{"dim": 2, "metric": "l2", "attrs": {{}}}}, "live_segments": [], "wal_folded_through": 0, "wal_format": 4, "segment_format": 2}}"#,
        FORMAT_VERSION + 1
    );
    store
        .put(&manifest::manifest_path("docs"), bad.into())
        .await
        .unwrap();
    let err = Namespace::open(store, "docs").await.err().unwrap();
    assert!(matches!(err, ItzmaError::UnsupportedFormat { .. }));
}

#[tokio::test]
async fn a_corrupt_wal_object_is_an_error_naming_the_key() {
    let store = common::memory();
    let ns = Namespace::create(store.clone(), "docs", schema(1, Metric::L2))
        .await
        .unwrap();
    ns.upsert(vec![doc("a", &[1.0])]).await.unwrap();
    store
        .put(&itzma_core::wal::wal_path("docs", 1), "not json".into())
        .await
        .unwrap();
    let err = ns.query(&[1.0], 1).await.unwrap_err();
    match err {
        ItzmaError::CorruptWal { path, .. } => assert!(path.contains("wal/00000000000000000001")),
        other => panic!("unexpected error: {other:?}"),
    }
}

#[tokio::test]
async fn top_k_larger_than_corpus_returns_everything() {
    let store: Arc<dyn object_store::ObjectStore> = common::memory();
    let ns = Namespace::create(store, "docs", schema(1, Metric::L2))
        .await
        .unwrap();
    ns.upsert(vec![doc("a", &[1.0]), doc("b", &[2.0])])
        .await
        .unwrap();
    assert_eq!(ns.query(&[0.0], 100).await.unwrap().len(), 2);
    assert!(ns.query(&[0.0], 0).await.unwrap().is_empty());
}

/// A new namespace's manifest carries exactly the fields this build knows,
/// so nothing it does not support is written into stored metadata.
#[tokio::test]
async fn a_new_manifest_has_exactly_the_known_fields() {
    let store = common::memory();
    itzma_core::Namespace::create(
        store.clone(),
        "docs",
        itzma_core::Schema::new(2, itzma_core::Metric::L2),
    )
    .await
    .unwrap();
    let raw =
        object_store::ObjectStoreExt::get(&*store, &itzma_core::manifest::manifest_path("docs"))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&raw).unwrap();
    let mut keys: Vec<&str> = json
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        [
            "format_version",
            "indexes",
            "live_segments",
            "schema",
            "segment_format",
            "sparse_indexes",
            "wal_folded_through",
            "wal_format"
        ]
    );
    assert_eq!(json["wal_format"], 5);
}
