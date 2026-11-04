//! GCS-specific behavior the conformance suite does not cover.

mod common;

use std::collections::BTreeSet;
use std::ops::Range;

use futures::TryStreamExt;
use google_cloud_storage::client::{Storage, StorageControl};
use itzma_store_gcs::{GcsOptions, GcsStore};
use object_store::path::Path;
use object_store::{
    Attribute, Attributes, Error, GetOptions, GetRange, ObjectStore, ObjectStoreExt, PutOptions,
    PutPayload,
};

#[tokio::test(flavor = "multi_thread")]
async fn round_trip() {
    let Some(store) = common::gcs().await else {
        return;
    };
    let key = Path::from("hello");
    store
        .put(&key, PutPayload::from_static(b"hello"))
        .await
        .expect("put");
    let body = store
        .get(&key)
        .await
        .expect("get")
        .bytes()
        .await
        .expect("body");
    assert_eq!(body.as_ref(), b"hello");
    let meta = store.head(&key).await.expect("head");
    assert_eq!(meta.size, 5);
    let generation: i64 = meta
        .version
        .as_deref()
        .expect("head reports a generation")
        .parse()
        .expect("generation is a number");
    assert!(generation > 0, "generation {generation}");
}

#[tokio::test(flavor = "multi_thread")]
async fn odd_names_round_trip() {
    let Some(store) = common::gcs().await else {
        return;
    };
    let names = ["a b/c.bin", "pct/100%.bin", "ü/ñ.bin"];
    for name in names {
        let key = Path::from(name);
        store
            .put(&key, PutPayload::from(name.as_bytes().to_vec()))
            .await
            .expect("put");
        let body = store
            .get(&key)
            .await
            .expect("get")
            .bytes()
            .await
            .expect("body");
        assert_eq!(body.as_ref(), name.as_bytes(), "round trip of {name:?}");
    }
    let listed: BTreeSet<String> = store
        .list(None)
        .map_ok(|m| m.location.to_string())
        .try_collect()
        .await
        .expect("list");
    let expected: BTreeSet<String> = names.iter().map(|n| Path::from(*n).to_string()).collect();
    assert_eq!(listed, expected, "listing returns the names as written");
}

#[tokio::test(flavor = "multi_thread")]
async fn get_preconditions_refused() {
    let Some(store) = common::gcs().await else {
        return;
    };
    let key = Path::from("preconditions");
    store
        .put(&key, PutPayload::from_static(b"x"))
        .await
        .expect("put");
    let if_match = GetOptions {
        if_match: Some("x".into()),
        ..GetOptions::default()
    };
    let result = store.get_opts(&key, if_match).await;
    assert!(
        matches!(result, Err(Error::NotImplemented { .. })),
        "if_match must be refused, got {:?}",
        result.map(|r| r.meta)
    );
    let version = GetOptions {
        version: Some("1".into()),
        ..GetOptions::default()
    };
    let result = store.get_opts(&key, version).await;
    assert!(
        matches!(result, Err(Error::NotImplemented { .. })),
        "version must be refused, got {:?}",
        result.map(|r| r.meta)
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn put_attributes_refused() {
    let Some(store) = common::gcs().await else {
        return;
    };
    let mut attributes = Attributes::new();
    attributes.insert(Attribute::ContentType, "text/plain".into());
    let options = PutOptions {
        attributes,
        ..PutOptions::default()
    };
    let result = store
        .put_opts(
            &Path::from("attributes"),
            PutPayload::from_static(b"x"),
            options,
        )
        .await;
    assert!(
        matches!(result, Err(Error::NotImplemented { .. })),
        "put attributes must be refused, got {result:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn list_root_finds_everything() {
    let Some(store) = common::gcs().await else {
        return;
    };
    for key in ["x/1", "y/2"] {
        store
            .put(&Path::from(key), PutPayload::from_static(b"1"))
            .await
            .expect("put");
    }
    let listed: BTreeSet<String> = store
        .list(None)
        .map_ok(|m| m.location.to_string())
        .try_collect()
        .await
        .expect("list");
    let expected = BTreeSet::from(["x/1".to_string(), "y/2".to_string()]);
    assert_eq!(listed, expected, "list(None) is the whole store");
}

#[tokio::test(flavor = "multi_thread")]
async fn empty_and_inverted_ranges_refused() {
    let Some(store) = common::gcs().await else {
        return;
    };
    let key = Path::from("ranges");
    store
        .put(&key, PutPayload::from_static(b"0123456789"))
        .await
        .expect("put");
    // Built field by field: an inverted range literal is what clippy refuses.
    for range in [3..3, Range { start: 5, end: 2 }] {
        let options = GetOptions {
            range: Some(GetRange::Bounded(range.clone())),
            ..GetOptions::default()
        };
        let result = store.get_opts(&key, options).await;
        assert!(
            matches!(result, Err(Error::Generic { .. })),
            "range {range:?} must be refused, got {:?}",
            result.map(|r| r.range)
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn root_listing_reaches_the_bucket_root() {
    let Some(store) = common::bare().await else {
        return;
    };
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let key = Path::from(format!("root-{}-{nanos}", std::process::id()));
    store
        .put(&key, PutPayload::from_static(b"1"))
        .await
        .expect("put");
    let listed: Vec<Path> = store
        .list(None)
        .map_ok(|m| m.location)
        .try_collect()
        .await
        .expect("list");
    assert!(
        listed.contains(&key),
        "list(None) on the bare store finds {key}"
    );
    store.delete(&key).await.expect("delete");
}

#[tokio::test(flavor = "multi_thread")]
async fn one_endpoint_without_the_other_refused() {
    for options in [
        GcsOptions::default()
            .endpoint("http://127.0.0.1:1")
            .anonymous(true),
        GcsOptions::default()
            .control_endpoint("http://127.0.0.1:1")
            .anonymous(true),
    ] {
        let result = GcsStore::connect("b", options).await;
        let message = result.expect_err("refused").to_string();
        assert!(
            message.contains("endpoint") && message.contains("control_endpoint"),
            "{message}"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn copy_conflict_names_target() {
    let Some(store) = common::gcs().await else {
        return;
    };
    let (src, taken) = (Path::from("copy-src"), Path::from("copy-taken"));
    for key in [&src, &taken] {
        store
            .put(key, PutPayload::from_static(b"x"))
            .await
            .expect("put");
    }
    match store.copy_if_not_exists(&src, &taken).await {
        Err(Error::AlreadyExists { path, .. }) => {
            assert!(path.ends_with("copy-taken"), "names the target, got {path}")
        }
        other => panic!("expected AlreadyExists, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn overwrite_retries_a_transient_failure() {
    let Some((endpoint, control, bucket)) = common::settings() else {
        return;
    };
    // storage-testbench fails the next upload once with a 503 for requests
    // carrying this test's id.
    let id = common::retry_test(&endpoint, "storage.objects.insert", "return-503").await;
    let credentials = google_cloud_auth::credentials::anonymous::Builder::new().build();
    let storage = Storage::builder()
        .with_endpoint(&endpoint)
        .with_credentials(credentials.clone())
        .with_custom_header(
            http::HeaderName::from_static("x-retry-test-id"),
            http::HeaderValue::from_str(&id).expect("header value"),
        )
        .build()
        .await
        .expect("storage");
    let control = StorageControl::builder()
        .with_endpoint(&control)
        .with_credentials(credentials)
        .build()
        .await
        .expect("control");
    let store = GcsStore::new(storage, control, bucket);
    let key = Path::from(format!("retry-{id}"));
    store
        .put(&key, PutPayload::from_static(b"once"))
        .await
        .expect("an overwrite is retried after a 503");
    store.delete(&key).await.expect("delete");
}
