//! The schema object: written once at create, checked and healed at
//! open, and never looked at by anything else.

mod common;

use std::sync::Arc;

use bytes::Bytes;
use common::faulty::{Fault, FaultyStore};
use common::{assert_consistent, backends, cas_backends, doc, schema, snapshot};
use itzma_core::manifest::{self, Manifest};
use itzma_core::schema_object::schema_path;
use itzma_core::wal::FORMAT_VERSION;
use itzma_core::{
    AttrSchema, AttrType, CompactionOptions, GcOptions, ItzmaError, Metric, Namespace, Schema,
};
use object_store::{ObjectStore, ObjectStoreExt};

const JSON: &[u8] = br#"{"format_version":2,"dim":3,"metric":"cosine","attrs":{}}"#;

fn cosine3() -> Schema {
    schema(3, Metric::Cosine)
}

fn dynamic(store: &Arc<FaultyStore>) -> Arc<dyn ObjectStore> {
    store.clone()
}

async fn schema_bytes(store: &dyn ObjectStore) -> Option<Bytes> {
    match store.get(&schema_path("docs")).await {
        Ok(r) => Some(r.bytes().await.unwrap()),
        Err(object_store::Error::NotFound { .. }) => None,
        Err(e) => panic!("{e}"),
    }
}

/// A namespace as an older build left it: a manifest and no schema object.
async fn manifest_only(store: &dyn ObjectStore, schema: Schema) {
    manifest::create(store, "docs", &Manifest::new(schema))
        .await
        .unwrap();
}

#[tokio::test]
async fn create_writes_the_schema_object() {
    for b in backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", cosine3())
            .await
            .unwrap();
        assert_eq!(ns.schema(), &cosine3());
        assert_eq!(schema_bytes(&*b.store).await.as_deref(), Some(JSON));
        let (m, _) = manifest::load(&*b.store, "docs").await.unwrap();
        assert_eq!(m.schema, cosine3());
    }
}

#[tokio::test]
async fn create_costs_a_read_and_two_puts_and_open_two_reads() {
    let store = FaultyStore::new(common::memory());
    store.reset_ops();
    Namespace::create(dynamic(&store), "docs", cosine3())
        .await
        .unwrap();
    assert_eq!(store.ops(), 3, "schema check, manifest put, schema put");

    store.reset_ops();
    Namespace::open(dynamic(&store), "docs").await.unwrap();
    assert_eq!(store.ops(), 2, "manifest, schema");
}

#[tokio::test]
async fn a_second_create_is_refused_and_changes_nothing() {
    for b in backends() {
        eprintln!("backend: {}", b.name);
        Namespace::create(b.store.clone(), "docs", cosine3())
            .await
            .unwrap();
        let before = snapshot(&*b.store).await;
        for other in [cosine3(), schema(4, Metric::L2)] {
            let err = Namespace::create(b.store.clone(), "docs", other)
                .await
                .err()
                .unwrap();
            assert!(matches!(err, ItzmaError::NamespaceExists(_)), "{err:?}");
        }
        assert_eq!(snapshot(&*b.store).await, before);
    }
}

#[tokio::test]
async fn racing_creates_leave_one_winner_and_an_object_that_matches_it() {
    for b in backends() {
        eprintln!("backend: {}", b.name);
        let tasks: Vec<_> = (1..=8usize)
            .map(|dim| {
                let store = b.store.clone();
                tokio::spawn(async move {
                    Namespace::create(store, "docs", schema(dim, Metric::L2)).await
                })
            })
            .collect();
        let mut winners = Vec::new();
        for task in tasks {
            match task.await.unwrap() {
                Ok(ns) => winners.push(ns.schema().clone()),
                Err(ItzmaError::NamespaceExists(_)) => {}
                Err(other) => panic!("{other:?}"),
            }
        }
        assert_eq!(winners.len(), 1, "exactly one create wins");
        let opened = Namespace::open(b.store.clone(), "docs").await.unwrap();
        assert_eq!(opened.schema(), &winners[0]);
        let expected = format!(
            r#"{{"format_version":2,"dim":{},"metric":"l2","attrs":{{}}}}"#,
            winners[0].dim
        );
        assert_eq!(
            schema_bytes(&*b.store).await.as_deref(),
            Some(expected.as_bytes())
        );
    }
}

#[tokio::test]
async fn open_writes_a_missing_object_once() {
    for b in backends() {
        eprintln!("backend: {}", b.name);
        manifest_only(&*b.store, cosine3()).await;
        assert_eq!(schema_bytes(&*b.store).await, None);

        let ns = Namespace::open(b.store.clone(), "docs").await.unwrap();
        assert_eq!(ns.schema(), &cosine3());
        assert_eq!(schema_bytes(&*b.store).await.as_deref(), Some(JSON));

        let healed = snapshot(&*b.store).await;
        Namespace::open(b.store.clone(), "docs").await.unwrap();
        assert_eq!(snapshot(&*b.store).await, healed, "the second open writes");
    }
}

#[tokio::test]
async fn open_refuses_a_manifest_that_disagrees_with_the_object() {
    for b in backends() {
        eprintln!("backend: {}", b.name);
        Namespace::create(b.store.clone(), "docs", cosine3())
            .await
            .unwrap();
        // A manifest that lost its schema in a bad restore and came back with
        // another one. A plain put works on every backend.
        let (mut m, _) = manifest::load(&*b.store, "docs").await.unwrap();
        m.schema = schema(4, Metric::L2);
        b.store
            .put(
                &manifest::manifest_path("docs"),
                serde_json::to_vec(&m).unwrap().into(),
            )
            .await
            .unwrap();
        let before = snapshot(&*b.store).await;

        let err = Namespace::open(b.store.clone(), "docs")
            .await
            .err()
            .unwrap();
        assert!(matches!(err, ItzmaError::CorruptManifest(_)), "{err:?}");
        assert_eq!(snapshot(&*b.store).await, before);
    }
}

#[tokio::test]
async fn an_object_without_a_manifest_reserves_its_id_for_its_own_schema() {
    for b in backends() {
        eprintln!("backend: {}", b.name);
        Namespace::create(b.store.clone(), "docs", cosine3())
            .await
            .unwrap();
        b.store
            .delete(&manifest::manifest_path("docs"))
            .await
            .unwrap();

        let err = Namespace::open(b.store.clone(), "docs")
            .await
            .err()
            .unwrap();
        assert!(matches!(err, ItzmaError::NamespaceNotFound(_)), "{err:?}");

        // A different schema does not get the id, and writes no manifest.
        let err = Namespace::create(b.store.clone(), "docs", schema(4, Metric::L2))
            .await
            .err()
            .unwrap();
        assert!(matches!(err, ItzmaError::NamespaceExists(_)), "{err:?}");
        let err = Namespace::open(b.store.clone(), "docs")
            .await
            .err()
            .unwrap();
        assert!(matches!(err, ItzmaError::NamespaceNotFound(_)), "{err:?}");
        assert_eq!(schema_bytes(&*b.store).await.as_deref(), Some(JSON));

        // The same schema adopts the object.
        Namespace::create(b.store.clone(), "docs", cosine3())
            .await
            .unwrap();
        let ns = Namespace::open(b.store.clone(), "docs").await.unwrap();
        assert_eq!(ns.schema(), &cosine3());
        assert_eq!(schema_bytes(&*b.store).await.as_deref(), Some(JSON));
    }
}

#[tokio::test]
async fn a_refused_namespace_is_left_untouched_by_open() {
    for b in backends() {
        eprintln!("backend: {}", b.name);
        manifest_only(&*b.store, cosine3()).await;
        let (mut m, _) = manifest::load(&*b.store, "docs").await.unwrap();
        m.wal_format = FORMAT_VERSION + 1;
        b.store
            .put(
                &manifest::manifest_path("docs"),
                serde_json::to_vec(&m).unwrap().into(),
            )
            .await
            .unwrap();
        let before = snapshot(&*b.store).await;

        let err = Namespace::open(b.store.clone(), "docs")
            .await
            .err()
            .unwrap();
        assert!(matches!(err, ItzmaError::WalFormatTooNew { .. }), "{err:?}");
        assert_eq!(snapshot(&*b.store).await, before, "open healed a refusal");
    }
}

#[tokio::test]
async fn only_open_looks_at_the_object() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", schema(2, Metric::L2))
            .await
            .unwrap();
        b.store.delete(&schema_path("docs")).await.unwrap();

        ns.upsert(vec![doc("a", &[1.0, 0.0])]).await.unwrap();
        ns.delete(vec!["a".into()]).await.unwrap();
        ns.upsert(vec![doc("b", &[0.0, 1.0])]).await.unwrap();
        assert_eq!(ns.query(&[0.0, 1.0], 1).await.unwrap()[0].id, "b");
        ns.compact(&CompactionOptions::default()).await.unwrap();
        ns.gc(&GcOptions::default()).await.unwrap();

        assert_eq!(
            schema_bytes(&*b.store).await,
            None,
            "a write, query, compaction or gc wrote or needed the object"
        );
    }
}

#[tokio::test]
async fn create_and_open_futures_are_send() {
    let store = common::memory();
    let s = store.clone();
    let ns = tokio::spawn(async move { Namespace::create(s, "docs", cosine3()).await })
        .await
        .unwrap()
        .unwrap();
    drop(ns);
    tokio::spawn(async move { Namespace::open(store, "docs").await })
        .await
        .unwrap()
        .unwrap();
}

/// Fault every operation of a create, as a plain error and as a lost ack. Then
/// open, or create again if the manifest never landed. The object must end up
/// there and equal, and the namespace must be sound.
#[tokio::test]
async fn a_stopped_create_ends_with_the_object_in_place() {
    for ix in 0..backends().len() {
        let clean = backends().swap_remove(ix);
        let counted = FaultyStore::new(clean.store.clone());
        counted.reset_ops();
        Namespace::create(dynamic(&counted), "docs", cosine3())
            .await
            .unwrap();
        let total = counted.ops();
        assert_eq!(total, 3);

        for fault in [Fault::Generic, Fault::LostAck] {
            for n in 0..total {
                eprintln!("backend {ix}: {fault:?} at operation {n} of {total}");
                let backend = backends().swap_remove(ix);
                let store = FaultyStore::new(backend.store.clone());

                store.fail_nth_op(n, fault);
                let created = Namespace::create(dynamic(&store), "docs", cosine3()).await;
                store.clear();
                assert!(created.is_err(), "a fault inside create must show");

                match Namespace::open(dynamic(&store), "docs").await {
                    Ok(_) => {}
                    Err(ItzmaError::NamespaceNotFound(_)) => {
                        Namespace::create(dynamic(&store), "docs", cosine3())
                            .await
                            .unwrap();
                    }
                    Err(other) => panic!("{other:?}"),
                }

                assert_eq!(schema_bytes(&*store).await.as_deref(), Some(JSON));
                let again = Namespace::create(dynamic(&store), "docs", cosine3())
                    .await
                    .err()
                    .unwrap();
                assert!(matches!(again, ItzmaError::NamespaceExists(_)), "{again:?}");
                assert_consistent(dynamic(&store), 3, Metric::Cosine).await;
            }
        }
    }
}

/// Fault every operation of an open that has to write the object.
#[tokio::test]
async fn a_stopped_heal_is_finished_by_the_next_open() {
    for ix in 0..backends().len() {
        let clean = backends().swap_remove(ix);
        manifest_only(&*clean.store, cosine3()).await;
        let counted = FaultyStore::new(clean.store.clone());
        counted.reset_ops();
        Namespace::open(dynamic(&counted), "docs").await.unwrap();
        let total = counted.ops();
        assert_eq!(total, 3, "manifest, schema, schema put");

        for fault in [Fault::Generic, Fault::LostAck] {
            for n in 0..total {
                eprintln!("backend {ix}: {fault:?} at operation {n} of {total}");
                let backend = backends().swap_remove(ix);
                manifest_only(&*backend.store, cosine3()).await;
                let store = FaultyStore::new(backend.store.clone());

                store.fail_nth_op(n, fault);
                let opened = Namespace::open(dynamic(&store), "docs").await;
                store.clear();
                assert!(opened.is_err(), "a fault inside a healing open must show");

                Namespace::open(dynamic(&store), "docs").await.unwrap();
                assert_eq!(schema_bytes(&*store).await.as_deref(), Some(JSON));
                assert_consistent(dynamic(&store), 3, Metric::Cosine).await;
            }
        }
    }
}

fn typed() -> Schema {
    schema(3, Metric::Cosine).with_attrs(
        AttrSchema::new()
            .field("price", AttrType::Number)
            .field("tags", AttrType::TextSet),
    )
}

#[tokio::test]
async fn a_typed_schema_reaches_the_manifest_the_object_and_a_reopened_handle() {
    for b in backends() {
        eprintln!("backend: {}", b.name);
        Namespace::create(b.store.clone(), "docs", typed())
            .await
            .unwrap();
        let bytes = schema_bytes(&*b.store).await.unwrap();
        assert_eq!(
            std::str::from_utf8(&bytes).unwrap(),
            r#"{"format_version":2,"dim":3,"metric":"cosine","attrs":{"price":"number","tags":"set<string>"}}"#
        );
        let (m, _) = manifest::load(&*b.store, "docs").await.unwrap();
        assert_eq!(m.schema, typed());
        let ns = Namespace::open(b.store.clone(), "docs").await.unwrap();
        assert_eq!(ns.schema(), &typed());
    }
}

#[tokio::test]
async fn create_refuses_a_bad_attribute_declaration_and_writes_nothing() {
    let store = common::memory();
    let long = "x".repeat(65);
    let mut too_many = AttrSchema::new();
    for i in 0..=itzma_core::attrs::MAX_ATTR_FIELDS {
        too_many = too_many.field(format!("f{i}"), AttrType::Bool);
    }
    let bad = [
        AttrSchema::new().field("", AttrType::Bool),
        AttrSchema::new().field("has space", AttrType::Bool),
        AttrSchema::new().field(long, AttrType::Bool),
        too_many,
    ];
    for attrs in bad {
        let err = Namespace::create(
            store.clone(),
            "docs",
            schema(3, Metric::L2).with_attrs(attrs),
        )
        .await
        .map(|_| ())
        .unwrap_err();
        assert!(matches!(err, ItzmaError::InvalidSchema(_)), "{err:?}");
    }
    assert!(snapshot(&*store).await.is_empty());
}

#[tokio::test]
async fn create_refuses_a_free_form_schema() {
    let store = common::memory();
    let free: Schema = serde_json::from_str(r#"{"dim":3,"metric":"l2"}"#).unwrap();
    let err = Namespace::create(store.clone(), "docs", free)
        .await
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(err, ItzmaError::InvalidSchema(_)), "{err:?}");
    assert!(snapshot(&*store).await.is_empty());
}

#[tokio::test]
async fn create_with_a_different_attribute_schema_is_namespace_exists() {
    let store = common::memory();
    Namespace::create(store.clone(), "docs", typed())
        .await
        .unwrap();
    let other =
        schema(3, Metric::Cosine).with_attrs(AttrSchema::new().field("price", AttrType::Text));
    let err = Namespace::create(store, "docs", other)
        .await
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(err, ItzmaError::NamespaceExists(_)), "{err:?}");
}

#[tokio::test]
async fn open_heals_a_missing_typed_schema_object_with_its_attrs() {
    let store = common::memory();
    manifest_only(&*store, typed()).await;
    Namespace::open(store.clone(), "docs").await.unwrap();
    let bytes = schema_bytes(&*store).await.unwrap();
    assert!(
        std::str::from_utf8(&bytes)
            .unwrap()
            .contains(r#""attrs":{"price":"number""#)
    );
}

#[tokio::test]
async fn open_refuses_a_schema_object_whose_attrs_differ_from_the_manifest() {
    let store = common::memory();
    manifest_only(&*store, typed()).await;
    store
        .put(
            &schema_path("docs"),
            Bytes::from_static(br#"{"format_version":2,"dim":3,"metric":"cosine","attrs":{}}"#)
                .into(),
        )
        .await
        .unwrap();
    let err = Namespace::open(store, "docs")
        .await
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(err, ItzmaError::CorruptManifest(_)), "{err:?}");
}

#[tokio::test]
async fn open_refuses_a_schema_object_from_a_newer_format() {
    let store = common::memory();
    manifest_only(&*store, typed()).await;
    store
        .put(
            &schema_path("docs"),
            Bytes::from_static(br#"{"format_version":3,"dim":3,"metric":"cosine","attrs":{}}"#)
                .into(),
        )
        .await
        .unwrap();
    let err = Namespace::open(store, "docs")
        .await
        .map(|_| ())
        .unwrap_err();
    assert!(
        matches!(err, ItzmaError::UnsupportedFormat { version: 3 }),
        "{err:?}"
    );
}

#[tokio::test]
async fn recover_carries_the_attribute_schema_into_the_new_manifest() {
    let store = common::memory();
    Namespace::create(store.clone(), "docs", typed())
        .await
        .unwrap();
    store
        .delete(&manifest::manifest_path("docs"))
        .await
        .unwrap();
    let (ns, _report) = Namespace::recover(store.clone(), "docs").await.unwrap();
    assert_eq!(ns.schema(), &typed());
    let (m, _) = manifest::load(&*store, "docs").await.unwrap();
    assert_eq!(m.schema, typed());
}
