//! Strict attribute writes: every `attrs` is checked against the
//! namespace's declared schema before anything is written.

mod common;

use common::{backends, doc, schema, snapshot};
use itzma_core::wal::WriteKey;
use itzma_core::{AttrSchema, AttrType, Document, ItzmaError, Metric, Namespace};
use serde_json::{Value, json};

fn typed() -> itzma_core::Schema {
    schema(2, Metric::L2).with_attrs(
        AttrSchema::new()
            .field("active", AttrType::Bool)
            .field("price", AttrType::Number)
            .field("tags", AttrType::TextSet),
    )
}

/// A bad `attrs` value and a check of the error it must produce.
type BadCase = (Value, fn(&ItzmaError) -> bool);

fn with_attrs(id: &str, attrs: Option<Value>) -> Document {
    let mut d = doc(id, &[1.0, 1.0]);
    d.attrs = attrs;
    d
}

#[tokio::test]
async fn a_valid_document_is_written_and_read_back_as_stored() {
    for b in backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", typed())
            .await
            .unwrap();
        let attrs = json!({"active": true, "price": 9.5, "tags": ["b", "a", "a"]});
        ns.upsert(vec![with_attrs("a", Some(attrs.clone()))])
            .await
            .unwrap();
        let hits = ns.query(&[1.0, 1.0], 1).await.unwrap();
        assert_eq!(
            hits[0].attrs,
            Some(attrs),
            "the stored array is returned as written"
        );
    }
}

#[tokio::test]
async fn each_kind_of_bad_attrs_is_refused_and_nothing_is_written() {
    let store = common::memory();
    let ns = Namespace::create(store.clone(), "docs", typed())
        .await
        .unwrap();
    let before = snapshot(&*store).await;
    let cases: Vec<BadCase> = vec![
        (json!({"colour": "red"}), |e| {
            matches!(e, ItzmaError::UnknownAttr { .. })
        }),
        (json!({"price": "9"}), |e| {
            matches!(e, ItzmaError::AttrTypeMismatch { .. })
        }),
        (json!({"tags": [1]}), |e| {
            matches!(e, ItzmaError::AttrTypeMismatch { .. })
        }),
        (json!(null), |e| matches!(e, ItzmaError::InvalidAttrs(_))),
        (json!([1]), |e| matches!(e, ItzmaError::InvalidAttrs(_))),
    ];
    for (attrs, is_expected) in cases {
        let err = ns
            .upsert(vec![with_attrs("a", Some(attrs.clone()))])
            .await
            .unwrap_err();
        assert!(is_expected(&err), "{attrs}: {err:?}");
    }
    common::assert_untouched(&*store, &before).await;
}

#[tokio::test]
async fn one_bad_document_rejects_the_whole_batch() {
    let store = common::memory();
    let ns = Namespace::create(store.clone(), "docs", typed())
        .await
        .unwrap();
    let before = snapshot(&*store).await;
    let err = ns
        .upsert(vec![
            with_attrs("good", Some(json!({"price": 1}))),
            with_attrs("bad", Some(json!({"price": "x"}))),
        ])
        .await
        .unwrap_err();
    assert!(matches!(err, ItzmaError::AttrTypeMismatch { .. }));
    common::assert_untouched(&*store, &before).await;
}

#[tokio::test]
async fn null_values_an_empty_object_and_no_attrs_are_accepted() {
    let ns = Namespace::create(common::memory(), "docs", typed())
        .await
        .unwrap();
    ns.upsert(vec![
        with_attrs("a", None),
        with_attrs("b", Some(json!({}))),
        with_attrs("c", Some(json!({"price": null, "tags": []}))),
    ])
    .await
    .unwrap();
}

#[tokio::test]
async fn a_namespace_with_an_empty_schema_takes_no_attributes() {
    let ns = Namespace::create(common::memory(), "docs", schema(2, Metric::L2))
        .await
        .unwrap();
    ns.upsert(vec![with_attrs("a", None)]).await.unwrap();
    let err = ns
        .upsert(vec![with_attrs("b", Some(json!({"x": 1})))])
        .await
        .unwrap_err();
    assert!(matches!(err, ItzmaError::UnknownAttr { .. }));
}

#[tokio::test]
async fn keyed_writes_are_checked_too() {
    let store = common::memory();
    let ns = Namespace::create(store.clone(), "docs", typed())
        .await
        .unwrap();
    let before = snapshot(&*store).await;
    let err = ns
        .upsert_keyed(
            &WriteKey::new("k"),
            vec![with_attrs("a", Some(json!({"price": "x"})))],
        )
        .await
        .unwrap_err();
    assert!(matches!(err, ItzmaError::AttrTypeMismatch { .. }));
    common::assert_untouched(&*store, &before).await;
    ns.upsert_keyed(
        &WriteKey::new("k"),
        vec![with_attrs("a", Some(json!({"price": 1})))],
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn a_namespace_without_an_attribute_schema_is_refused() {
    let store = common::memory();
    common::legacy_namespace(&*store).await;
    let err = Namespace::open(store.clone(), "docs").await.err().unwrap();
    assert!(
        matches!(err, ItzmaError::UnsupportedFormat { version: 1 }),
        "{err:?}"
    );
}
