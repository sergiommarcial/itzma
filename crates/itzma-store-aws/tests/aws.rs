//! S3-specific behavior the conformance suite does not cover.

mod common;

use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::Arc;

use futures::{StreamExt, TryStreamExt};
use itzma_store_aws::{S3Options, S3Store};
use object_store::path::Path;
use object_store::{
    Attribute, Attributes, Error, GetOptions, GetRange, ObjectStore, ObjectStoreExt, PutMode,
    PutOptions, PutPayload, UpdateVersion,
};

#[tokio::test(flavor = "multi_thread")]
async fn round_trip() {
    let Some(store) = common::aws().await else {
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
async fn http_needs_allow_http() {
    let options = S3Options::default()
        .env(false)
        .endpoint("http://127.0.0.1:9000")
        .access_key_id("k")
        .secret_access_key("s");
    let message = S3Store::connect("b", options)
        .await
        .expect_err("refused")
        .to_string();
    assert!(message.contains("allow_http"), "{message}");
}

#[test]
fn debug_hides_secrets() {
    let options = S3Options::default()
        .access_key_id("AKIDEXAMPLE")
        .secret_access_key("wJalrXUtnFEMI")
        .session_token("FwoGZXIvYXdzE");
    let text = format!("{options:?}");
    for secret in ["AKIDEXAMPLE", "wJalrXUtnFEMI", "FwoGZXIvYXdzE"] {
        assert!(!text.contains(secret), "{text}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn odd_names_round_trip() {
    let Some(store) = common::aws().await else {
        return;
    };
    let names = ["a b/c.bin", "pct/100%.bin", "ü/ñ.bin", "plus/a+b.bin"];
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
    let Some(store) = common::aws().await else {
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
    let Some(store) = common::aws().await else {
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
    let requests = Arc::new(common::Requests::default());
    let Some(store) = common::counted(requests.clone()) else {
        return;
    };
    let key = Path::from(format!("ranges-{}", std::process::id()));
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
    assert!(requests.sent().is_empty(), "refused before any request");
}

#[tokio::test(flavor = "multi_thread")]
async fn update_of_a_missing_key_is_a_precondition() {
    let Some(store) = common::aws().await else {
        return;
    };
    let version = UpdateVersion {
        e_tag: Some("\"0123456789abcdef0123456789abcdef\"".into()),
        version: None,
    };
    let result = store
        .put_opts(
            &Path::from("never-written"),
            PutPayload::from_static(b"x"),
            PutOptions::from(PutMode::Update(version)),
        )
        .await;
    assert!(
        matches!(result, Err(Error::Precondition { .. })),
        "an update of a missing key must be Precondition, got {result:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn reads_cost_one_request() {
    let requests = Arc::new(common::Requests::default());
    let Some(store) = common::counted(requests.clone()) else {
        return;
    };
    let key = Path::from(format!("count-{}", std::process::id()));
    store
        .put(&key, PutPayload::from_static(b"0123456789"))
        .await
        .expect("put");
    let cost = |range: Option<GetRange>| {
        let (store, key, requests) = (&store, key.clone(), requests.clone());
        async move {
            let before = requests.sent().len();
            let options = GetOptions {
                range,
                ..GetOptions::default()
            };
            let got = store.get_opts(&key, options).await.expect("get");
            let range = got.range.clone();
            got.bytes().await.expect("body");
            (requests.sent()[before..].to_vec(), range)
        }
    };
    let get = vec!["GET".to_string()];
    assert_eq!(cost(None).await, (get.clone(), 0..10), "a whole get");
    assert_eq!(
        cost(Some(GetRange::Bounded(2..5))).await,
        (get.clone(), 2..5),
        "a bounded get"
    );
    assert_eq!(
        cost(Some(GetRange::Suffix(3))).await,
        (get, 7..10),
        "a suffix get is one GET, no HEAD"
    );
    store.delete(&key).await.expect("delete");
}

#[tokio::test(flavor = "multi_thread")]
async fn list_root_finds_everything() {
    let Some(store) = common::aws().await else {
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
async fn root_listing_reaches_the_bucket_root() {
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

#[tokio::test(flavor = "multi_thread")]
async fn listing_crosses_a_page() {
    // S3 pages at 1,000 keys, so 1,050 really takes two pages.
    let Some(store) = common::aws().await else {
        return;
    };
    let keys: BTreeSet<String> = (0..1050).map(|i| format!("many/{i:04}")).collect();
    futures::stream::iter(keys.iter().cloned())
        .map(|key| {
            let store = store.clone();
            async move {
                store
                    .put(&Path::from(key), PutPayload::from_static(b"1"))
                    .await
            }
        })
        .buffer_unordered(32)
        .try_collect::<Vec<_>>()
        .await
        .expect("puts");
    let listed: BTreeSet<String> = store
        .list(Some(&Path::from("many")))
        .map_ok(|m| m.location.to_string())
        .try_collect()
        .await
        .expect("list");
    assert_eq!(listed, keys, "every key across the page boundary");
    let after: Vec<String> = store
        .list_with_offset(Some(&Path::from("many")), &Path::from("many/0999"))
        .map_ok(|m| m.location.to_string())
        .try_collect()
        .await
        .expect("list with offset");
    assert_eq!(after.len(), 50, "keys strictly after the offset");
}

#[tokio::test(flavor = "multi_thread")]
async fn empty_multipart_upload_reads_empty() {
    // S3 refuses to complete an upload with no parts.
    let Some(store) = common::aws().await else {
        return;
    };
    let key = Path::from("empty-multipart");
    let mut upload = store.put_multipart(&key).await.expect("start");
    upload.complete().await.expect("complete with no parts");
    assert_eq!(store.head(&key).await.expect("head").size, 0);
    let body = store
        .get(&key)
        .await
        .expect("get")
        .bytes()
        .await
        .expect("body");
    assert!(body.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn abort_leaves_no_object() {
    let Some(store) = common::aws().await else {
        return;
    };
    let key = Path::from("aborted");
    let mut upload = store.put_multipart(&key).await.expect("start");
    upload
        .put_part(PutPayload::from_static(b"left behind"))
        .await
        .expect("part");
    upload.abort().await.expect("abort");
    let head = store.head(&key).await;
    assert!(
        matches!(head, Err(Error::NotFound { .. })),
        "an aborted upload leaves nothing, got {head:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn concurrent_multipart_uploads_do_not_mix() {
    // Parts other than the last must be 5 MiB on S3, so one part each.
    let Some(store) = common::aws().await else {
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
    b.complete().await.expect("complete b");
    a.complete().await.expect("complete a");
    let body = store
        .get(&key)
        .await
        .expect("get")
        .bytes()
        .await
        .expect("body");
    assert_eq!(body.as_ref(), b"a0", "the last completed upload, whole");
}

#[tokio::test(flavor = "multi_thread")]
async fn errors_hide_the_access_key() {
    let Some((endpoint, bucket)) = common::settings() else {
        return;
    };
    let options = common::options(&endpoint).access_key_id("itzma-leak-check");
    let store = S3Store::connect(&bucket, options).await.expect("s3 store");
    let err = store
        .put(&Path::from("leak"), PutPayload::from_static(b"1"))
        .await
        .expect_err("an unknown key is refused");
    let text = format!("{err} {err:?}");
    assert!(text.contains("InvalidAccessKeyId"), "{text}");
    assert!(
        !text.contains("itzma-leak-check"),
        "the error carries the access key id: {text}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn odd_names_copy() {
    let Some(store) = common::aws().await else {
        return;
    };
    for name in ["pct/100%.bin", "plus/a+b.bin", "a b/c.bin", "ü/ñ.bin"] {
        let from = Path::from(name);
        let body = name.as_bytes().to_vec();
        store
            .put(&from, PutPayload::from(body.clone()))
            .await
            .expect("put");
        for (to, copy) in [
            (Path::from(format!("copied/{name}")), "copy"),
            (Path::from(format!("absent/{name}")), "copy_if_not_exists"),
        ] {
            match copy {
                "copy" => store.copy(&from, &to).await.expect("copy"),
                _ => store
                    .copy_if_not_exists(&from, &to)
                    .await
                    .expect("copy if absent"),
            }
            let got = store
                .get(&to)
                .await
                .expect("get")
                .bytes()
                .await
                .expect("body");
            assert_eq!(got.as_ref(), body.as_slice(), "{copy} of {name:?}");
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn complete_retried_after_a_failed_empty_part() {
    let Some(store) = common::with_interceptor(common::FailFirstPart::default()) else {
        return;
    };
    let key = Path::from(format!("retried-empty-{}", std::process::id()));
    let mut upload = store.put_multipart(&key).await.expect("start");
    upload
        .complete()
        .await
        .expect_err("the empty part's upload fails once");
    upload.complete().await.expect("a retried complete works");
    assert_eq!(store.head(&key).await.expect("head").size, 0);
    store.delete(&key).await.expect("delete");
}

async fn body(store: &dyn ObjectStore, key: &Path) -> Vec<u8> {
    store
        .get(key)
        .await
        .expect("get")
        .bytes()
        .await
        .expect("body")
        .to_vec()
}

#[tokio::test(flavor = "multi_thread")]
async fn put_if_absent_whose_reply_was_lost_succeeds() {
    // The first attempt lands and its reply is lost, so the SDK's retry meets
    // the object that attempt wrote.
    let inject = common::Inject::new(common::conditional_put, 503, common::SLOW_DOWN);
    let Some(store) = common::with_interceptor(inject) else {
        return;
    };
    let key = Path::from(format!("{}/lost-create", common::fresh_prefix()));
    let put = store
        .put_opts(
            &key,
            PutPayload::from_static(b"ours"),
            PutMode::Create.into(),
        )
        .await;
    let read = body(&store, &key).await;
    let head = store.head(&key).await.expect("head");
    store.delete(&key).await.expect("delete");
    let put = put.expect("the retry finds its own write");
    assert_eq!(read, b"ours");
    assert_eq!(put.e_tag, head.e_tag, "the ETag of the object written");
}

#[tokio::test(flavor = "multi_thread")]
async fn retried_put_if_absent_still_refuses_another_writers_object() {
    let inject = common::Inject::new(common::conditional_put, 503, common::SLOW_DOWN);
    let Some(store) = common::with_interceptor(inject) else {
        return;
    };
    let key = Path::from(format!("{}/taken", common::fresh_prefix()));
    store
        .put(&key, PutPayload::from_static(b"theirs"))
        .await
        .expect("put");
    let put = store
        .put_opts(
            &key,
            PutPayload::from_static(b"ours"),
            PutMode::Create.into(),
        )
        .await;
    let read = body(&store, &key).await;
    store.delete(&key).await.expect("delete");
    assert!(
        matches!(put, Err(Error::AlreadyExists { .. })),
        "another writer's object is taken, got {put:?}"
    );
    assert_eq!(read, b"theirs");
}

#[tokio::test(flavor = "multi_thread")]
async fn update_whose_reply_was_lost_succeeds() {
    let inject = common::Inject::new(common::conditional_put, 503, common::SLOW_DOWN);
    let Some(store) = common::with_interceptor(inject) else {
        return;
    };
    let key = Path::from(format!("{}/lost-update", common::fresh_prefix()));
    let first = store
        .put(&key, PutPayload::from_static(b"v1"))
        .await
        .expect("put");
    let version = UpdateVersion {
        e_tag: first.e_tag,
        version: None,
    };
    let put = store
        .put_opts(
            &key,
            PutPayload::from_static(b"v2"),
            PutMode::Update(version).into(),
        )
        .await;
    let read = body(&store, &key).await;
    store.delete(&key).await.expect("delete");
    put.expect("the retry finds its own write");
    assert_eq!(read, b"v2");
}

#[tokio::test(flavor = "multi_thread")]
async fn conditional_conflict_is_retried() {
    // S3 answers 409 ConditionalRequestConflict while another conditional
    // write or a delete of the key is in flight, and says to try again.
    let prefix = common::fresh_prefix();
    let key = format!("{prefix}/conflict");
    let decoy = format!("{prefix}/decoy");
    let inject =
        common::Inject::new(common::conditional_put, 409, common::CONFLICT).decoy(&key, &decoy);
    let Some(store) = common::with_interceptor(inject) else {
        return;
    };
    let key = Path::from(key);
    let put = store
        .put_opts(
            &key,
            PutPayload::from_static(b"ours"),
            PutMode::Create.into(),
        )
        .await;
    let read = store.get(&key).await;
    for k in [&key, &Path::from(decoy)] {
        store.delete(k).await.expect("delete");
    }
    put.expect("a conflict is retried, not reported as taken");
    let read = read.expect("get").bytes().await.expect("body");
    assert_eq!(read.as_ref(), b"ours");
}

#[tokio::test(flavor = "multi_thread")]
async fn empty_copy_if_absent_works_when_the_part_copy_fails() {
    // Real S3 may refuse to copy a zero-byte object as a part.
    let inject = common::Inject::new(common::part_copy, 400, common::BAD_COPY);
    let Some(store) = common::with_interceptor(inject) else {
        return;
    };
    let prefix = common::fresh_prefix();
    let (from, to) = (
        Path::from(format!("{prefix}/empty")),
        Path::from(format!("{prefix}/empty-copy")),
    );
    store.put(&from, PutPayload::new()).await.expect("put");
    let copied = store.copy_if_not_exists(&from, &to).await;
    let read = store.get(&to).await;
    store.delete(&from).await.expect("delete");
    let _ = store.delete(&to).await;
    copied.expect("an empty object is copied without a part copy");
    assert!(read.expect("get").bytes().await.expect("body").is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn failed_part_copy_of_a_full_object_is_an_error() {
    let inject = common::Inject::new(common::part_copy, 400, common::BAD_COPY);
    let Some(store) = common::with_interceptor(inject) else {
        return;
    };
    let prefix = common::fresh_prefix();
    let (from, to) = (
        Path::from(format!("{prefix}/full")),
        Path::from(format!("{prefix}/full-copy")),
    );
    store
        .put(&from, PutPayload::from_static(b"bytes"))
        .await
        .expect("put");
    let copied = store.copy_if_not_exists(&from, &to).await;
    let head = store.head(&to).await;
    store.delete(&from).await.expect("delete");
    assert!(copied.is_err(), "the copy failed, got {copied:?}");
    assert!(matches!(head, Err(Error::NotFound { .. })), "{head:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn small_parts_go_up_as_one() {
    // S3 needs 5 MiB in every part but the last, so smaller ones are held
    // until they add up or the upload completes.
    let requests = Arc::new(common::Requests::default());
    let Some(store) = common::counted(requests.clone()) else {
        return;
    };
    let key = Path::from(format!("{}/small-parts", common::fresh_prefix()));
    let mut upload = store.put_multipart(&key).await.expect("start");
    for part in [&b"one "[..], b"two ", b"three"] {
        upload
            .put_part(PutPayload::from(part.to_vec()))
            .await
            .expect("part");
    }
    let completed = upload.complete().await;
    let read = store.get(&key).await;
    let _ = store.delete(&key).await;
    completed.expect("complete");
    let read = read.expect("get").bytes().await.expect("body");
    assert_eq!(read.as_ref(), b"one two three");
    let parts = requests.sent().iter().filter(|m| *m == "PUT").count();
    assert_eq!(parts, 1, "one part upload");
}

#[tokio::test(flavor = "multi_thread")]
async fn folder_markers_are_not_objects() {
    // The S3 console's "Create folder" writes an empty `name/` key.
    let (Some(store), Some((client, bucket))) = (common::bare().await, common::client()) else {
        return;
    };
    let prefix = common::fresh_prefix();
    for key in [format!("{prefix}/"), format!("{prefix}/dir/")] {
        client
            .put_object()
            .bucket(&bucket)
            .key(key)
            .send()
            .await
            .expect("marker");
    }
    let file = Path::from(format!("{prefix}/dir/file"));
    store
        .put(&file, PutPayload::from_static(b"1"))
        .await
        .expect("put");
    let base = Path::from(prefix.clone());
    let dir = Path::from(format!("{prefix}/dir"));
    let listed: Vec<String> = store
        .list(Some(&base))
        .map_ok(|m| m.location.to_string())
        .try_collect()
        .await
        .expect("list");
    let top = store.list_with_delimiter(Some(&base)).await.expect("top");
    let inner = store.list_with_delimiter(Some(&dir)).await.expect("inner");
    for key in [format!("{prefix}/"), format!("{prefix}/dir/")] {
        client
            .delete_object()
            .bucket(&bucket)
            .key(key)
            .send()
            .await
            .expect("delete marker");
    }
    store.delete(&file).await.expect("delete");
    assert_eq!(listed, [file.to_string()]);
    assert!(top.objects.is_empty(), "{:?}", top.objects);
    assert_eq!(top.common_prefixes, [dir]);
    let inner: Vec<_> = inner.objects.iter().map(|m| m.location.clone()).collect();
    assert_eq!(inner, [file]);
}
