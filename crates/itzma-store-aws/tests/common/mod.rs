#![allow(dead_code)] // each test binary uses a subset of these helpers

//! The test store: MinIO from `make s3-up`, with the throwaway keys in
//! `compose.yaml`. Built with `env: false`, so no `AWS_*` variable or
//! `~/.aws` profile on the machine is ever used.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use itzma_store_aws::{S3Options, S3Store};
use object_store::ObjectStore;
use object_store::prefix::PrefixStore;

fn var(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

/// The endpoint and bucket, or `None` unless `ITZMA_TEST_S3_ENDPOINT` is set.
pub fn settings() -> Option<(String, String)> {
    Some((
        var("ITZMA_TEST_S3_ENDPOINT")?,
        var("ITZMA_TEST_S3_BUCKET").unwrap_or_else(|| "itzma-test".into()),
    ))
}

/// The options the test store connects with.
pub fn options(endpoint: &str) -> S3Options {
    S3Options::default()
        .env(false)
        .endpoint(endpoint)
        .allow_http(true)
        .region("us-east-1")
        .access_key_id(var("ITZMA_TEST_S3_ACCESS_KEY").unwrap_or_else(|| "itzma".into()))
        .secret_access_key(
            var("ITZMA_TEST_S3_SECRET_KEY").unwrap_or_else(|| "itzma-test-secret".into()),
        )
}

/// The store over the whole bucket: what an `s3://bucket` URL gets. The
/// bucket is shared, so tests on it check containment only.
pub async fn bare() -> Option<S3Store> {
    let (endpoint, bucket) = settings()?;
    Some(
        S3Store::connect(&bucket, options(&endpoint))
            .await
            .expect("s3 store"),
    )
}

/// A key prefix no other test or run uses.
pub fn fresh_prefix() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    format!(
        "conformance/{}-{nanos}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}

/// `None` unless `ITZMA_TEST_S3_ENDPOINT` is set; each call under its own
/// key prefix.
pub async fn aws() -> Option<Arc<dyn ObjectStore>> {
    let store = bare().await?;
    Some(Arc::new(PrefixStore::new(store, fresh_prefix())))
}

/// Records the HTTP method of every request the SDK sends, for tests that pin
/// what a call costs.
#[derive(Debug, Default)]
pub struct Requests(pub std::sync::Mutex<Vec<String>>);

impl Requests {
    pub fn sent(&self) -> Vec<String> {
        self.0.lock().expect("lock").clone()
    }
}

#[derive(Debug)]
struct Recorder(Arc<Requests>);

impl aws_sdk_s3::config::Intercept for Recorder {
    fn name(&self) -> &'static str {
        "Recorder"
    }

    fn read_before_transmit(
        &self,
        context: &aws_sdk_s3::config::interceptors::BeforeTransmitInterceptorContextRef<'_>,
        _components: &aws_sdk_s3::config::RuntimeComponents,
        _config: &mut aws_sdk_s3::config::ConfigBag,
    ) -> Result<(), aws_sdk_s3::error::BoxError> {
        let method = context.request().method().to_string();
        self.0.0.lock().expect("lock").push(method);
        Ok(())
    }
}

/// Fails the first part upload (`?partNumber=`), once, before it is sent, as
/// a part that times out for good would.
#[derive(Debug, Default)]
pub struct FailFirstPart(pub std::sync::atomic::AtomicBool);

impl aws_sdk_s3::config::Intercept for FailFirstPart {
    fn name(&self) -> &'static str {
        "FailFirstPart"
    }

    fn read_before_transmit(
        &self,
        context: &aws_sdk_s3::config::interceptors::BeforeTransmitInterceptorContextRef<'_>,
        _components: &aws_sdk_s3::config::RuntimeComponents,
        _config: &mut aws_sdk_s3::config::ConfigBag,
    ) -> Result<(), aws_sdk_s3::error::BoxError> {
        let part = context.request().uri().contains("partNumber=");
        if part && !self.0.swap(true, Ordering::SeqCst) {
            return Err("injected failure".into());
        }
        Ok(())
    }
}

/// A client config for the test store, without the bucket.
fn config(endpoint: &str) -> aws_sdk_s3::config::Builder {
    let credentials = aws_sdk_s3::config::Credentials::new(
        var("ITZMA_TEST_S3_ACCESS_KEY").unwrap_or_else(|| "itzma".into()),
        var("ITZMA_TEST_S3_SECRET_KEY").unwrap_or_else(|| "itzma-test-secret".into()),
        None,
        None,
        "test",
    );
    aws_sdk_s3::Config::builder()
        .behavior_version_latest()
        .region(aws_sdk_s3::config::Region::new("us-east-1"))
        .credentials_provider(credentials)
        .endpoint_url(endpoint)
        .force_path_style(true)
}

/// A plain SDK client and the bucket, for writing keys `Path` can't name.
pub fn client() -> Option<(aws_sdk_s3::Client, String)> {
    let (endpoint, bucket) = settings()?;
    Some((
        aws_sdk_s3::Client::from_conf(config(&endpoint).build()),
        bucket,
    ))
}

/// A store over the bucket whose client runs `interceptor` on every request.
pub fn with_interceptor(
    interceptor: impl aws_sdk_s3::config::Intercept + 'static,
) -> Option<S3Store> {
    let (endpoint, bucket) = settings()?;
    let config = config(&endpoint).interceptor(interceptor).build();
    Some(S3Store::new(aws_sdk_s3::Client::from_conf(config), bucket))
}

/// The body S3 sends with a 409 when two conditional writes overlap.
pub const CONFLICT: &str = "<Error><Code>ConditionalRequestConflict</Code>\
    <Message>A conflicting conditional operation is currently in progress \
    against this resource. Try again.</Message></Error>";

/// A 503 the SDK retries, standing in for a reply lost on the way back.
pub const SLOW_DOWN: &str =
    "<Error><Code>SlowDown</Code><Message>Please reduce your request rate.</Message></Error>";

/// Any part copy's failure; the adapter must not depend on its code.
pub const BAD_COPY: &str =
    "<Error><Code>InvalidRequest</Code><Message>The copy failed.</Message></Error>";

/// A PUT of a whole object with `If-None-Match` or `If-Match`.
pub fn conditional_put(request: &aws_sdk_s3::config::http::HttpRequest) -> bool {
    let headers = request.headers();
    request.method() == "PUT"
        && !request.uri().contains("partNumber=")
        && (headers.contains_key("if-none-match") || headers.contains_key("if-match"))
}

/// An `UploadPartCopy`.
pub fn part_copy(request: &aws_sdk_s3::config::http::HttpRequest) -> bool {
    request.uri().contains("partNumber=") && request.headers().contains_key("x-amz-copy-source")
}

/// Replaces the server's reply to the first request `pick` chooses with
/// `status` and `body`. With a decoy, that request is first sent to another
/// key, so the server never applies it where the test looks.
#[derive(Debug)]
pub struct Inject {
    pick: fn(&aws_sdk_s3::config::http::HttpRequest) -> bool,
    status: u16,
    body: &'static str,
    decoy: Option<(String, String)>,
    /// 0 waiting, 1 the picked request is in flight, 2 done.
    state: std::sync::atomic::AtomicU8,
}

impl Inject {
    pub fn new(
        pick: fn(&aws_sdk_s3::config::http::HttpRequest) -> bool,
        status: u16,
        body: &'static str,
    ) -> Self {
        Self {
            pick,
            status,
            body,
            decoy: None,
            state: Default::default(),
        }
    }

    /// Sends the picked request to `decoy` in place of `key`.
    pub fn decoy(mut self, key: &str, decoy: &str) -> Self {
        self.decoy = Some((key.into(), decoy.into()));
        self
    }
}

impl aws_sdk_s3::config::Intercept for Inject {
    fn name(&self) -> &'static str {
        "Inject"
    }

    fn modify_before_signing(
        &self,
        context: &mut aws_sdk_s3::config::interceptors::BeforeTransmitInterceptorContextMut<'_>,
        _components: &aws_sdk_s3::config::RuntimeComponents,
        _config: &mut aws_sdk_s3::config::ConfigBag,
    ) -> Result<(), aws_sdk_s3::error::BoxError> {
        let request = context.request_mut();
        if !(self.pick)(request) || self.state.load(Ordering::SeqCst) != 0 {
            return Ok(());
        }
        self.state.store(1, Ordering::SeqCst);
        if let Some((key, decoy)) = &self.decoy {
            let uri = request.uri().replacen(key.as_str(), decoy, 1);
            request.set_uri(uri)?;
        }
        Ok(())
    }

    fn modify_before_deserialization(
        &self,
        context: &mut aws_sdk_s3::config::interceptors::BeforeDeserializationInterceptorContextMut<
            '_,
        >,
        _components: &aws_sdk_s3::config::RuntimeComponents,
        _config: &mut aws_sdk_s3::config::ConfigBag,
    ) -> Result<(), aws_sdk_s3::error::BoxError> {
        if self
            .state
            .compare_exchange(1, 2, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            *context.response_mut() = aws_sdk_s3::config::http::HttpResponse::new(
                self.status.try_into()?,
                aws_sdk_s3::primitives::SdkBody::from(self.body),
            );
        }
        Ok(())
    }
}

/// A store over the bucket whose client records its requests in `requests`.
pub fn counted(requests: Arc<Requests>) -> Option<S3Store> {
    with_interceptor(Recorder(requests))
}
