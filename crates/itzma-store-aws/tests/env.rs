//! `env: false` reads nothing from the environment. In its own binary: it
//! sets process-wide variables that would leak into other tests.

mod common;

use itzma_store_aws::S3Store;
use object_store::path::Path;
use object_store::prefix::PrefixStore;
use object_store::{ObjectStoreExt, PutPayload};

#[tokio::test(flavor = "multi_thread")]
async fn env_false_reads_nothing_from_the_environment() {
    let Some((endpoint, bucket)) = common::settings() else {
        return;
    };
    // SAFETY: the only test in this binary, set before any thread reads them.
    unsafe {
        std::env::set_var("AWS_ACCESS_KEY_ID", "junk");
        std::env::set_var("AWS_SECRET_ACCESS_KEY", "junk");
        std::env::set_var("AWS_SESSION_TOKEN", "junk");
        std::env::set_var("AWS_ENDPOINT_URL", "http://127.0.0.1:1");
        std::env::set_var("AWS_ENDPOINT_URL_S3", "http://127.0.0.1:1");
        std::env::set_var("AWS_REGION", "junk-1");
        std::env::set_var("AWS_PROFILE", "junk");
    }
    let store = S3Store::connect(&bucket, common::options(&endpoint))
        .await
        .expect("s3 store");
    let store = PrefixStore::new(store, common::fresh_prefix());
    let key = Path::from("env");
    store
        .put(&key, PutPayload::from_static(b"own keys"))
        .await
        .expect("put with the options' own keys");
    assert_eq!(store.head(&key).await.expect("head").size, 8);

    // The control: with `env` and no keys of its own, the store signs with
    // the junk variables above, so they really were in reach.
    let options = itzma_store_aws::S3Options::default()
        .endpoint(&endpoint)
        .allow_http(true)
        .region("us-east-1");
    let store = S3Store::connect(&bucket, options).await.expect("s3 store");
    let store = PrefixStore::new(store, common::fresh_prefix());
    let err = store
        .put(&key, PutPayload::from_static(b"junk keys"))
        .await
        .expect_err("env: true must sign with AWS_* keys");
    assert!(err.to_string().contains("InvalidAccessKeyId"), "{err}");
}
