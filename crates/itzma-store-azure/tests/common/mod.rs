#![allow(dead_code)] // each test binary uses a subset of these helpers

//! The test store: Azurite from `make azure-up`, reached over HTTP with an
//! account SAS signed from Azurite's published development key.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use azure_core::credentials::Secret;
use azure_core::http::Url;
use itzma_store_azure::{AzureCredential, AzureOptions, AzureStore};
use object_store::ObjectStore;
use object_store::prefix::PrefixStore;

pub const ACCOUNT: &str = "devstoreaccount1";
/// Azurite's published development key; not a secret.
pub const KEY: &str =
    "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==";
const SAS_VERSION: &str = "2025-01-05";

/// An account SAS for the blob service (resources `sco`, permissions
/// `racwdl`), as `scripts/azurite_sas.py` builds it.
pub fn sas(account: &str, key: &str, expiry: &str) -> String {
    let (permissions, services, resources, protocols) = ("racwdl", "b", "sco", "https,http");
    let to_sign = [
        account,
        permissions,
        services,
        resources,
        "",
        expiry,
        "",
        protocols,
        SAS_VERSION,
        "",
        "",
    ]
    .join("\n");
    let signature =
        azure_core::hmac::hmac_sha256(&to_sign, &Secret::new(key.to_string())).expect("signature");
    let mut url = Url::parse("http://sas/").expect("url");
    url.query_pairs_mut()
        .append_pair("sv", SAS_VERSION)
        .append_pair("ss", services)
        .append_pair("srt", resources)
        .append_pair("sp", permissions)
        .append_pair("se", expiry)
        .append_pair("spr", protocols)
        .append_pair("sig", &signature);
    url.query().expect("query").to_string()
}

pub fn test_sas() -> String {
    sas(ACCOUNT, KEY, "2099-01-01T00:00:00Z")
}

/// The account endpoint and container, or `None` unless
/// `ITZMA_TEST_AZURE_ENDPOINT` is set.
pub fn settings() -> Option<(String, String)> {
    let var = |name: &str| std::env::var(name).ok().filter(|v| !v.is_empty());
    Some((
        var("ITZMA_TEST_AZURE_ENDPOINT")?,
        var("ITZMA_TEST_AZURE_CONTAINER").unwrap_or_else(|| "itzma-test".into()),
    ))
}

/// The store over the whole container: what an `az://container` URL gets.
/// The container is shared, so tests on it check containment only.
pub async fn bare() -> Option<AzureStore> {
    let (endpoint, container) = settings()?;
    let options = AzureOptions::default()
        .endpoint(endpoint)
        .credential(AzureCredential::Sas(test_sas()));
    Some(
        AzureStore::connect(ACCOUNT, &container, options)
            .await
            .expect("azure store"),
    )
}

/// Counts every HTTP request the SDK sends, for tests that pin how many a
/// call costs.
#[derive(Debug, Default)]
pub struct Requests(pub std::sync::atomic::AtomicUsize);

#[async_trait::async_trait]
impl azure_core::http::policies::Policy for Requests {
    async fn send(
        &self,
        ctx: &azure_core::http::Context,
        request: &mut azure_core::http::Request,
        next: &[Arc<dyn azure_core::http::policies::Policy>],
    ) -> azure_core::http::policies::PolicyResult {
        self.0.fetch_add(1, Ordering::SeqCst);
        next[0].send(ctx, request, &next[1..]).await
    }
}

/// Fails the first request whose URL contains `needle`, once, before the
/// SDK's retry policy sees it, as a commit that times out for good would.
#[derive(Debug)]
pub struct FailOnce {
    pub needle: &'static str,
    pub done: std::sync::atomic::AtomicBool,
}

#[async_trait::async_trait]
impl azure_core::http::policies::Policy for FailOnce {
    async fn send(
        &self,
        ctx: &azure_core::http::Context,
        request: &mut azure_core::http::Request,
        next: &[Arc<dyn azure_core::http::policies::Policy>],
    ) -> azure_core::http::policies::PolicyResult {
        if request.url().as_str().contains(self.needle) && !self.done.swap(true, Ordering::SeqCst) {
            return Err(azure_core::Error::with_message(
                azure_core::error::ErrorKind::Other,
                "injected failure",
            ));
        }
        next[0].send(ctx, request, &next[1..]).await
    }
}

/// Records the `If-Match` header of every GET the SDK sends.
#[derive(Debug, Default)]
pub struct IfMatch(pub std::sync::Mutex<Vec<Option<String>>>);

#[async_trait::async_trait]
impl azure_core::http::policies::Policy for IfMatch {
    async fn send(
        &self,
        ctx: &azure_core::http::Context,
        request: &mut azure_core::http::Request,
        next: &[Arc<dyn azure_core::http::policies::Policy>],
    ) -> azure_core::http::policies::PolicyResult {
        if request.method() == azure_core::http::Method::Get {
            let value = request
                .headers()
                .get_optional_str(&azure_core::http::headers::HeaderName::from_static(
                    "if-match",
                ))
                .map(str::to_string);
            self.0.lock().expect("lock").push(value);
        }
        next[0].send(ctx, request, &next[1..]).await
    }
}

/// A store over the container with `policy` in its client's pipeline.
pub fn with_policy(policy: Arc<dyn azure_core::http::policies::Policy>) -> Option<AzureStore> {
    let (endpoint, container) = settings()?;
    let url = Url::parse(&format!("{endpoint}/{container}?{}", test_sas())).expect("url");
    let mut options = azure_storage_blob::BlobContainerClientOptions::default();
    options.client_options.per_call_policies.push(policy);
    let client =
        azure_storage_blob::BlobContainerClient::new(url, None, Some(options)).expect("client");
    Some(AzureStore::new(client))
}

/// A store over the container whose client counts its requests in `requests`.
pub fn counted(requests: Arc<Requests>) -> Option<AzureStore> {
    let (endpoint, container) = settings()?;
    let url = Url::parse(&format!("{endpoint}/{container}?{}", test_sas())).expect("url");
    let mut options = azure_storage_blob::BlobContainerClientOptions::default();
    options.client_options.per_call_policies.push(requests);
    let client =
        azure_storage_blob::BlobContainerClient::new(url, None, Some(options)).expect("client");
    Some(AzureStore::new(client))
}

/// `None` unless `ITZMA_TEST_AZURE_ENDPOINT` is set; each call under its own
/// key prefix.
pub async fn azure() -> Option<Arc<dyn ObjectStore>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let store = bare().await?;
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
