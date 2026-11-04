#![allow(dead_code)] // each test binary uses a subset of these helpers

//! The test store: storage-testbench from `make gcs-up`, anonymous, each call
//! under its own key prefix.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use itzma_store_gcs::{GcsOptions, GcsStore};
use object_store::ObjectStore;
use object_store::prefix::PrefixStore;

/// The testbench's two endpoints and bucket, or `None` unless
/// `ITZMA_TEST_GCS_ENDPOINT` is set.
pub fn settings() -> Option<(String, String, String)> {
    let var = |name: &str| std::env::var(name).ok().filter(|v| !v.is_empty());
    Some((
        var("ITZMA_TEST_GCS_ENDPOINT")?,
        var("ITZMA_TEST_GCS_CONTROL_ENDPOINT").expect("ITZMA_TEST_GCS_CONTROL_ENDPOINT"),
        var("ITZMA_TEST_GCS_BUCKET").unwrap_or_else(|| "itzma-test".into()),
    ))
}

/// The store over the whole bucket, with no prefix: what a `gs://bucket` URL
/// gets. The bucket is shared, so tests on it check containment only.
pub async fn bare() -> Option<GcsStore> {
    let (endpoint, control, bucket) = settings()?;
    let options = GcsOptions::default()
        .endpoint(endpoint)
        .control_endpoint(control)
        .anonymous(true);
    Some(
        GcsStore::connect(&bucket, options)
            .await
            .expect("gcs store"),
    )
}

/// Registers a storage-testbench retry test that injects `failure` once into
/// `method`, and returns its id for the `x-retry-test-id` header.
pub async fn retry_test(endpoint: &str, method: &str, failure: &str) -> String {
    let body = format!(r#"{{"instructions":{{"{method}":["{failure}"]}}}}"#);
    let output = std::process::Command::new("curl")
        .args([
            "-sf",
            "-X",
            "POST",
            "-H",
            "Content-Type: application/json",
            "-d",
            &body,
        ])
        .arg(format!("{endpoint}/retry_test"))
        .output()
        .expect("curl");
    let text = String::from_utf8(output.stdout).expect("utf-8");
    let start = text.find("\"id\"").expect("retry test id") + 4;
    let rest = &text[start..];
    let open = rest.find('"').expect("id value") + 1;
    let close = rest[open..].find('"').expect("id end") + open;
    rest[open..close].to_string()
}

/// `None` unless `ITZMA_TEST_GCS_ENDPOINT` is set.
pub async fn gcs() -> Option<Arc<dyn ObjectStore>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let var = |name: &str| std::env::var(name).ok().filter(|v| !v.is_empty());
    let endpoint = var("ITZMA_TEST_GCS_ENDPOINT")?;
    let options = GcsOptions::default()
        .endpoint(endpoint)
        .control_endpoint(
            var("ITZMA_TEST_GCS_CONTROL_ENDPOINT").expect("ITZMA_TEST_GCS_CONTROL_ENDPOINT"),
        )
        .anonymous(true);
    let bucket = var("ITZMA_TEST_GCS_BUCKET").unwrap_or_else(|| "itzma-test".into());
    let store = GcsStore::connect(&bucket, options)
        .await
        .expect("gcs store");
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let prefix = format!(
        "conformance/{}-{nanos}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    );
    Some(Arc::new(PrefixStore::new(store, prefix)))
}
