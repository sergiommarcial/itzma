//! With `env` and no region anywhere, `connect` falls back to us-east-1
//! without asking the instance metadata service, which off EC2 means waiting
//! on connect timeouts. In its own binary: it changes process-wide variables.

mod common;

use std::time::{Duration, Instant};

use itzma_store_aws::S3Store;
use object_store::path::Path;
use object_store::prefix::PrefixStore;
use object_store::{ObjectStoreExt, PutPayload};

#[tokio::test(flavor = "multi_thread")]
async fn no_region_falls_back_without_imds() {
    let Some((endpoint, bucket)) = common::settings() else {
        return;
    };
    let nowhere = std::env::temp_dir().join("itzma-no-such-aws-file");
    // SAFETY: the only test in this binary, set before any thread reads them.
    unsafe {
        for name in [
            "AWS_REGION",
            "AWS_DEFAULT_REGION",
            "AWS_PROFILE",
            "AWS_EC2_METADATA_DISABLED",
        ] {
            std::env::remove_var(name);
        }
        std::env::set_var("AWS_CONFIG_FILE", &nowhere);
        std::env::set_var("AWS_SHARED_CREDENTIALS_FILE", &nowhere);
    }
    let mut options = common::options(&endpoint).env(true);
    options.region = None;
    let started = Instant::now();
    let store = S3Store::connect(&bucket, options).await.expect("s3 store");
    let took = started.elapsed();
    assert!(
        took < Duration::from_millis(500),
        "connect took {took:?}, as if it waited on IMDS"
    );
    let store = PrefixStore::new(store, common::fresh_prefix());
    store
        .put(&Path::from("r"), PutPayload::from_static(b"1"))
        .await
        .expect("put in the fallback region");
}
