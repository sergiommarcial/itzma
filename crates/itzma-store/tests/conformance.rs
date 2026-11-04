//! The storage contract, checked against the stores Itzma already trusts.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use object_store::ObjectStore;
use object_store::aws::{AmazonS3Builder, S3CopyIfNotExists};
use object_store::memory::InMemory;
use object_store::prefix::PrefixStore;

async fn memory() -> Option<Arc<dyn ObjectStore>> {
    Some(Arc::new(InMemory::new()))
}

/// MinIO from `compose.yaml`, or `None` when `ITZMA_TEST_S3_ENDPOINT` is unset.
/// Each call gets its own key prefix so checks never see each other's keys.
async fn s3() -> Option<Arc<dyn ObjectStore>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let endpoint = std::env::var("ITZMA_TEST_S3_ENDPOINT")
        .ok()
        .filter(|v| !v.is_empty())?;
    let var = |name: &str, default: &str| std::env::var(name).unwrap_or_else(|_| default.into());
    let store = AmazonS3Builder::new()
        .with_endpoint(endpoint)
        .with_allow_http(true)
        .with_region("us-east-1")
        .with_bucket_name(var("ITZMA_TEST_S3_BUCKET", "itzma-test"))
        .with_access_key_id(var("ITZMA_TEST_S3_ACCESS_KEY", "itzma"))
        .with_secret_access_key(var("ITZMA_TEST_S3_SECRET_KEY", "itzma-test-secret"))
        .with_copy_if_not_exists(S3CopyIfNotExists::Multipart)
        .build()
        .expect("s3 store");
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

itzma_store::conformance_tests!(memory, memory);
itzma_store::conformance_tests!(s3, s3);
