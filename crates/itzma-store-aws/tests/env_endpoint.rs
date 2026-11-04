//! With `env`, an endpoint from the environment gets the same checks and
//! path-style addressing as one in the options, under the names
//! `object_store`'s `from_env` read too. In its own binary: it changes
//! process-wide variables.

mod common;

use itzma_store_aws::S3Store;
use object_store::path::Path;
use object_store::prefix::PrefixStore;
use object_store::{ObjectStoreExt, PutPayload};

#[tokio::test(flavor = "multi_thread")]
async fn endpoints_from_the_environment() {
    let Some((endpoint, bucket)) = common::settings() else {
        return;
    };
    let keys_only = || {
        let mut options = common::options(&endpoint).env(true).allow_http(false);
        options.endpoint = None;
        options
    };
    let set = |pairs: &[(&str, Option<&str>)]| {
        // SAFETY: the only test in this binary; no other thread reads these.
        unsafe {
            for (name, value) in pairs {
                match value {
                    Some(v) => std::env::set_var(name, v),
                    None => std::env::remove_var(name),
                }
            }
        }
    };

    // object_store's names: AWS_ENDPOINT and AWS_ALLOW_HTTP.
    set(&[
        ("AWS_ENDPOINT", Some(endpoint.as_str())),
        ("AWS_ALLOW_HTTP", Some("true")),
        ("AWS_ENDPOINT_URL", None),
        ("AWS_ENDPOINT_URL_S3", None),
    ]);
    let store = S3Store::connect(&bucket, keys_only())
        .await
        .expect("s3 store");
    let store = PrefixStore::new(store, common::fresh_prefix());
    store
        .put(&Path::from("e"), PutPayload::from_static(b"1"))
        .await
        .expect("put through AWS_ENDPOINT, path-style");

    // The SDK's own name, path-style too.
    set(&[
        ("AWS_ENDPOINT", None),
        ("AWS_ENDPOINT_URL", Some(endpoint.as_str())),
    ]);
    let store = S3Store::connect(&bucket, keys_only())
        .await
        .expect("s3 store");
    let store = PrefixStore::new(store, common::fresh_prefix());
    store
        .put(&Path::from("e"), PutPayload::from_static(b"1"))
        .await
        .expect("put through AWS_ENDPOINT_URL, path-style");

    // An http endpoint from the environment still needs allow_http.
    set(&[("AWS_ALLOW_HTTP", None)]);
    let message = S3Store::connect(&bucket, keys_only())
        .await
        .expect_err("refused")
        .to_string();
    assert!(message.contains("allow_http"), "{message}");
}
