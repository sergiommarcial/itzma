//! Azure-specific behavior the conformance suite does not cover.

mod common;

use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use futures::TryStreamExt;
use itzma_store_azure::{AzureCredential, AzureOptions, AzureStore};
use object_store::path::Path;
use object_store::{
    Attribute, Attributes, Error, GetOptions, GetRange, ObjectStore, ObjectStoreExt, PutOptions,
    PutPayload,
};

#[tokio::test(flavor = "multi_thread")]
async fn round_trip() {
    let Some(store) = common::azure().await else {
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
    assert!(meta.e_tag.is_some(), "head reports an etag");
}

#[tokio::test(flavor = "multi_thread")]
async fn token_over_http_refused() {
    let options = AzureOptions::default()
        .endpoint("http://127.0.0.1:10000/devstoreaccount1")
        .credential(AzureCredential::ManagedIdentity);
    let message = AzureStore::connect("devstoreaccount1", "c", options)
        .await
        .expect_err("refused")
        .to_string();
    assert!(
        message.contains("https") && message.contains("SAS"),
        "{message}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn errors_hide_the_sas() {
    let Some(store) = common::azure().await else {
        return;
    };
    let sas = common::test_sas();
    let signature = sas
        .split('&')
        .find_map(|pair| pair.strip_prefix("sig="))
        .expect("sig");
    let err = store
        .get(&Path::from("missing"))
        .await
        .expect_err("missing");
    let text = format!("{err} {err:?}");
    assert!(
        !text.contains(signature),
        "the error carries the SAS signature: {text}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn odd_names_round_trip() {
    let Some(store) = common::azure().await else {
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
    let Some(store) = common::azure().await else {
        return;
    };
    let key = Path::from("preconditions");
    store
        .put(&key, PutPayload::from_static(b"x"))
        .await
        .expect("put");
    for options in [
        GetOptions {
            if_match: Some("x".into()),
            ..GetOptions::default()
        },
        GetOptions {
            version: Some("1".into()),
            ..GetOptions::default()
        },
    ] {
        let result = store.get_opts(&key, options).await;
        assert!(
            matches!(result, Err(Error::NotImplemented { .. })),
            "a get precondition must be refused, got {:?}",
            result.map(|r| r.meta)
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn put_attributes_refused() {
    let Some(store) = common::azure().await else {
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
async fn empty_and_inverted_ranges_refused() {
    let Some(store) = common::azure().await else {
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
async fn reads_cost_one_request_and_a_suffix_two() {
    let requests = Arc::new(common::Requests::default());
    let Some(store) = common::counted(requests.clone()) else {
        return;
    };
    let key = Path::from(format!("count-{}", std::process::id()));
    store
        .put(&key, PutPayload::from_static(b"0123456789"))
        .await
        .expect("put");
    let cost = |options: GetOptions| {
        let (store, key, requests) = (&store, key.clone(), requests.clone());
        async move {
            let before = requests.0.load(Ordering::SeqCst);
            let got = store.get_opts(&key, options).await.expect("get");
            let range = got.range.clone();
            got.bytes().await.expect("body");
            (requests.0.load(Ordering::SeqCst) - before, range)
        }
    };
    assert_eq!(cost(GetOptions::default()).await, (1, 0..10), "a whole get");
    let bounded = GetOptions {
        range: Some(GetRange::Bounded(2..5)),
        ..GetOptions::default()
    };
    assert_eq!(cost(bounded).await, (1, 2..5), "a bounded get");
    let suffix = GetOptions {
        range: Some(GetRange::Suffix(3)),
        ..GetOptions::default()
    };
    assert_eq!(
        cost(suffix).await,
        (2, 7..10),
        "a suffix get: head, then the range"
    );
    store.delete(&key).await.expect("delete");
}

#[tokio::test(flavor = "multi_thread")]
async fn list_root_finds_everything() {
    let Some(store) = common::azure().await else {
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
async fn root_listing_reaches_the_container_root() {
    let Some(store) = common::bare().await else {
        return;
    };
    let key = Path::from(format!("root-{}", std::process::id()));
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

fn unique(name: &str) -> Path {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    Path::from(format!("{name}-{}-{nanos}", std::process::id()))
}

#[tokio::test(flavor = "multi_thread")]
async fn complete_retried_after_a_failed_commit_keeps_the_parts() {
    let fail = Arc::new(common::FailOnce {
        needle: "comp=blocklist",
        done: Default::default(),
    });
    let Some(store) = common::with_policy(fail) else {
        return;
    };
    let key = unique("retry-commit");
    let mut upload = store.put_multipart(&key).await.expect("start");
    upload
        .put_part(PutPayload::from_static(b"first "))
        .await
        .expect("part");
    upload
        .put_part(PutPayload::from_static(b"second"))
        .await
        .expect("part");
    assert!(
        upload.complete().await.is_err(),
        "the injected commit failure"
    );
    upload.complete().await.expect("the retried commit");
    let body = store
        .get(&key)
        .await
        .expect("get")
        .bytes()
        .await
        .expect("body");
    assert_eq!(
        body.as_ref(),
        b"first second",
        "the retry committed the parts"
    );
    store.delete(&key).await.expect("delete");
}

#[tokio::test(flavor = "multi_thread")]
async fn concurrent_multipart_uploads_do_not_mix() {
    let Some(store) = common::azure().await else {
        return;
    };
    let key = Path::from("mixed");
    let mut a = store.put_multipart(&key).await.expect("start a");
    let mut b = store.put_multipart(&key).await.expect("start b");
    a.put_part(PutPayload::from_static(b"a0"))
        .await
        .expect("a0");
    b.put_part(PutPayload::from_static(b"b0"))
        .await
        .expect("b0");
    a.put_part(PutPayload::from_static(b"a1"))
        .await
        .expect("a1");
    b.put_part(PutPayload::from_static(b"b1"))
        .await
        .expect("b1");
    b.complete().await.expect("complete b");
    // B's commit discards every uncommitted block on the blob, so A's commit
    // may fail. What must never happen is A reporting success while the blob
    // holds B's bytes (or a mix), as it did when both used the same block ids.
    let a_done = a.complete().await;
    let body = store
        .get(&key)
        .await
        .expect("get")
        .bytes()
        .await
        .expect("body");
    match a_done {
        Ok(_) => assert_eq!(body.as_ref(), b"a0a1", "A committed, so the blob is A's"),
        Err(_) => assert_eq!(body.as_ref(), b"b0b1", "A failed, so the blob is still B's"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_large_put_after_an_aborted_upload_works() {
    let Some(store) = common::azure().await else {
        return;
    };
    let key = Path::from("after-abort");
    let mut upload = store.put_multipart(&key).await.expect("start");
    upload
        .put_part(PutPayload::from_static(b"left behind"))
        .await
        .expect("part");
    upload.abort().await.expect("abort");
    // Above 4 MiB the SDK stages its own blocks, which must fit beside ours.
    let big = vec![7u8; 5 * 1024 * 1024];
    store
        .put(&key, PutPayload::from(big.clone()))
        .await
        .expect("a large put after an abort");
    let body = store
        .get(&key)
        .await
        .expect("get")
        .bytes()
        .await
        .expect("body");
    assert!(body.as_ref() == big.as_slice(), "large body round trip");
}

#[tokio::test(flavor = "multi_thread")]
async fn large_writes_keep_their_conditions() {
    let Some(store) = common::azure().await else {
        return;
    };
    // Above 4 MiB the SDK stages blocks; the condition rides on the commit.
    let big = PutPayload::from(vec![1u8; 9 * 1024 * 1024]);
    let key = Path::from("large-conditions");
    let put = store
        .put_opts(
            &key,
            big.clone(),
            PutOptions::from(object_store::PutMode::Create),
        )
        .await
        .expect("create");
    let again = store
        .put_opts(
            &key,
            big.clone(),
            PutOptions::from(object_store::PutMode::Create),
        )
        .await;
    assert!(
        matches!(again, Err(Error::AlreadyExists { .. })),
        "a large put-if-absent on an existing key, got {again:?}"
    );
    let version = object_store::UpdateVersion {
        e_tag: put.e_tag.clone(),
        version: None,
    };
    store
        .put_opts(
            &key,
            big.clone(),
            PutOptions::from(object_store::PutMode::Update(version.clone())),
        )
        .await
        .expect("update");
    let stale = store
        .put_opts(
            &key,
            big,
            PutOptions::from(object_store::PutMode::Update(version)),
        )
        .await;
    assert!(
        matches!(stale, Err(Error::Precondition { .. })),
        "a large stale update, got {stale:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_suffix_get_is_pinned_to_the_etag_it_sized() {
    let seen = Arc::new(common::IfMatch::default());
    let Some(store) = common::with_policy(seen.clone()) else {
        return;
    };
    let key = unique("pinned");
    let put = store
        .put(&key, PutPayload::from_static(b"0123456789"))
        .await
        .expect("put");
    seen.0.lock().expect("lock").clear();
    let options = GetOptions {
        range: Some(GetRange::Suffix(3)),
        ..GetOptions::default()
    };
    store
        .get_opts(&key, options)
        .await
        .expect("get")
        .bytes()
        .await
        .expect("body");
    let gets = seen.0.lock().expect("lock").clone();
    assert_eq!(gets, [put.e_tag], "the ranged GET carries the HEAD's ETag");
    store.delete(&key).await.expect("delete");
}

#[tokio::test(flavor = "multi_thread")]
async fn an_endpoint_with_a_query_is_refused_without_echoing_it() {
    let options = AzureOptions::default()
        .endpoint("https://acct.blob.core.windows.net/?sv=1&sig=SECRET123")
        .credential(AzureCredential::Anonymous);
    let message = AzureStore::connect("acct", "c", options)
        .await
        .expect_err("refused")
        .to_string();
    assert!(message.contains("query"), "{message}");
    assert!(!message.contains("SECRET123"), "{message}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_container_url_with_a_trailing_slash_works() {
    let Some((endpoint, container)) = common::settings() else {
        return;
    };
    let url =
        azure_core::http::Url::parse(&format!("{endpoint}/{container}/?{}", common::test_sas()))
            .expect("url");
    let client = azure_storage_blob::BlobContainerClient::new(url, None, None).expect("client");
    let store = AzureStore::new(client);
    assert_eq!(store.to_string(), format!("AzureStore(az://{container})"));
    let key = unique("slash");
    store
        .put(&key, PutPayload::from_static(b"x"))
        .await
        .expect("put");
    assert_eq!(store.head(&key).await.expect("head").size, 1);
    store.delete(&key).await.expect("delete");
}
