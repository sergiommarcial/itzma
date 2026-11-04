use std::fmt;
use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use aws_config::BehaviorVersion;
use aws_config::environment::EnvironmentVariableRegionProvider;
use aws_config::meta::region::RegionProviderChain;
use aws_config::profile::ProfileFileRegionProvider;
use aws_config::timeout::TimeoutConfig;
use aws_sdk_s3::Client;
use aws_sdk_s3::config::{
    Credentials, Region, RequestChecksumCalculation, ResponseChecksumValidation,
};
use aws_sdk_s3::operation::complete_multipart_upload::CompleteMultipartUploadError;
use aws_sdk_s3::operation::put_object::PutObjectError;
use aws_sdk_s3::primitives::{ByteStream, DateTime as AwsDateTime};
use aws_sdk_s3::types::{CompletedMultipartUpload, CompletedPart};
use bytes::Bytes;
use chrono::{DateTime, Utc};
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use object_store::path::Path;
use object_store::{
    CopyMode, CopyOptions, GetOptions, GetRange, GetResult, GetResultPayload, ListResult,
    MultipartUpload, ObjectMeta, ObjectStore, PutMode, PutMultipartOptions, PutOptions, PutPayload,
    PutResult, Result,
};

use crate::errors::{self, Write};
use crate::retry::{self, Attempts};
use crate::upload::Upload;

/// The region when neither the options nor (with `env`) the environment name
/// one, as `object_store`'s S3 client does.
const DEFAULT_REGION: &str = "us-east-1";

const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// How long a read may wait for the next bytes of a response.
const READ_TIMEOUT: Duration = Duration::from_secs(30);

/// Settings for [`S3Store::connect`], named after the Python bridge's `s3://`
/// options so both read the same.
#[non_exhaustive]
#[derive(Clone)]
pub struct S3Options {
    /// The bucket's region.
    pub region: Option<String>,
    /// An S3-compatible endpoint, such as MinIO's. Requests then use path-style
    /// addressing (`<endpoint>/<bucket>/<key>`).
    pub endpoint: Option<String>,
    /// A static access key id; goes with `secret_access_key`.
    pub access_key_id: Option<String>,
    /// A static secret access key; goes with `access_key_id`.
    pub secret_access_key: Option<String>,
    /// A session token for temporary static keys.
    pub session_token: Option<String>,
    /// Allows an `http://` endpoint. Off by default, so plain HTTP is a choice.
    pub allow_http: bool,
    /// Reads the AWS environment: `AWS_*` variables, `~/.aws` profiles, web
    /// identity, the container and instance credentials (`aws-config`'s
    /// default chain), and an endpoint from `AWS_ENDPOINT_URL_S3`,
    /// `AWS_ENDPOINT_URL` or `AWS_ENDPOINT` with `AWS_ALLOW_HTTP`. The region
    /// comes from the environment or a profile, never the instance metadata
    /// service; else us-east-1. The options above override all of it. Without
    /// it only the options are used, and with no keys the instance's role
    /// (IMDS, which still reads its own `AWS_EC2_METADATA_*` settings).
    pub env: bool,
}

impl Default for S3Options {
    fn default() -> Self {
        Self {
            region: None,
            endpoint: None,
            access_key_id: None,
            secret_access_key: None,
            session_token: None,
            allow_http: false,
            env: true,
        }
    }
}

impl fmt::Debug for S3Options {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let set = |v: &Option<String>| v.as_ref().map(|_| "..");
        f.debug_struct("S3Options")
            .field("region", &self.region)
            .field("endpoint", &self.endpoint)
            .field("access_key_id", &set(&self.access_key_id))
            .field("secret_access_key", &set(&self.secret_access_key))
            .field("session_token", &set(&self.session_token))
            .field("allow_http", &self.allow_http)
            .field("env", &self.env)
            .finish()
    }
}

impl S3Options {
    /// Sets [`region`](Self::region).
    pub fn region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Sets [`endpoint`](Self::endpoint).
    pub fn endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }

    /// Sets [`access_key_id`](Self::access_key_id).
    pub fn access_key_id(mut self, key: impl Into<String>) -> Self {
        self.access_key_id = Some(key.into());
        self
    }

    /// Sets [`secret_access_key`](Self::secret_access_key).
    pub fn secret_access_key(mut self, key: impl Into<String>) -> Self {
        self.secret_access_key = Some(key.into());
        self
    }

    /// Sets [`session_token`](Self::session_token).
    pub fn session_token(mut self, token: impl Into<String>) -> Self {
        self.session_token = Some(token.into());
        self
    }

    /// Sets [`allow_http`](Self::allow_http).
    pub fn allow_http(mut self, allow: bool) -> Self {
        self.allow_http = allow;
        self
    }

    /// Sets [`env`](Self::env).
    pub fn env(mut self, env: bool) -> Self {
        self.env = env;
        self
    }
}

/// An [`ObjectStore`] over one S3 bucket through the official SDK.
pub struct S3Store {
    client: Arc<Client>,
    bucket: String,
}

impl fmt::Debug for S3Store {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("S3Store")
            .field("bucket", &self.bucket)
            .finish()
    }
}

impl fmt::Display for S3Store {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "S3Store(s3://{})", self.bucket)
    }
}

impl S3Store {
    /// A store over `bucket` through `client`, built by the caller with
    /// whatever credentials and settings they need.
    pub fn new(client: Client, bucket: impl Into<String>) -> Self {
        Self {
            client: Arc::new(client),
            bucket: bucket.into(),
        }
    }

    /// Builds a client from `options`, then a store over `bucket`.
    ///
    /// # Errors
    ///
    /// `Error::Generic` for an `http` endpoint without `allow_http`, an
    /// endpoint that isn't a URL, or only one of the two static keys.
    pub async fn connect(bucket: &str, options: S3Options) -> Result<Self> {
        // With `env`, an endpoint and `allow_http` may come from the
        // environment, under the SDK's names and the ones `object_store`'s
        // `from_env` read; either way they get the same checks.
        let env = |name: &str| {
            options
                .env
                .then(|| std::env::var(name).ok().filter(|v| !v.is_empty()))
                .flatten()
        };
        let endpoint = options
            .endpoint
            .clone()
            .or_else(|| env("AWS_ENDPOINT_URL_S3"))
            .or_else(|| env("AWS_ENDPOINT_URL"))
            .or_else(|| env("AWS_ENDPOINT"));
        let allow_http = options.allow_http
            || env("AWS_ALLOW_HTTP").is_some_and(|v| v.eq_ignore_ascii_case("true"));
        if let Some(endpoint) = &endpoint {
            let scheme = endpoint.split_once("://").map(|(s, _)| s);
            match scheme {
                Some("https") => {}
                Some("http") if allow_http => {}
                Some("http") => {
                    return Err(errors::generic(
                        "the endpoint is http; set allow_http to use plain HTTP",
                    ));
                }
                _ => {
                    return Err(errors::generic(format!(
                        "bad endpoint {endpoint:?}: expected an http:// or https:// URL"
                    )));
                }
            }
        }
        let credentials = match (&options.access_key_id, &options.secret_access_key) {
            (Some(id), Some(secret)) => Some(Credentials::new(
                id,
                secret,
                options.session_token.clone(),
                None,
                "itzma",
            )),
            (None, None) => None,
            _ => {
                return Err(errors::generic(
                    "access_key_id and secret_access_key go together",
                ));
            }
        };
        let mut config = if options.env {
            let mut loader = aws_config::defaults(BehaviorVersion::latest());
            loader = match &options.region {
                Some(region) => loader.region(Region::new(region.clone())),
                // The default chain ends at the instance metadata service,
                // which off EC2 means waiting on connect timeouts, and on EC2
                // would move a store `object_store` kept in us-east-1.
                None => loader.region(
                    RegionProviderChain::first_try(EnvironmentVariableRegionProvider::new())
                        .or_else(ProfileFileRegionProvider::new())
                        .or_else(Region::new(DEFAULT_REGION)),
                ),
            };
            if let Some(credentials) = credentials {
                loader = loader.credentials_provider(credentials);
            }
            aws_sdk_s3::config::Builder::from(&loader.load().await)
        } else {
            // Nothing from the environment or `~/.aws`: the SDK's own builder
            // reads neither, unlike `aws_config::defaults`. Only the IMDS
            // client, used when there are no keys, reads its own
            // `AWS_EC2_METADATA_*` settings.
            let mut config = aws_sdk_s3::config::Builder::new()
                .behavior_version(BehaviorVersion::latest())
                .region(Region::new(
                    options
                        .region
                        .clone()
                        .unwrap_or_else(|| DEFAULT_REGION.into()),
                ));
            config = match credentials {
                Some(credentials) => config.credentials_provider(credentials),
                None => config.credentials_provider(
                    aws_config::imds::credentials::ImdsCredentialsProvider::builder().build(),
                ),
            };
            config
        };
        // The SDK sets only a connect timeout; a server that accepts and then
        // never answers would hang a call. `object_store` gave up after 30 s.
        config = config.timeout_config(
            TimeoutConfig::builder()
                .connect_timeout(CONNECT_TIMEOUT)
                .read_timeout(READ_TIMEOUT)
                .build(),
        );
        if let Some(endpoint) = &endpoint {
            // S3-compatible stores take path-style addressing, and not all of
            // them take the checksums the SDK adds for S3 by default.
            config = config
                .endpoint_url(endpoint)
                .force_path_style(true)
                .request_checksum_calculation(RequestChecksumCalculation::WhenRequired)
                .response_checksum_validation(ResponseChecksumValidation::WhenRequired);
        }
        Ok(Self::new(Client::from_conf(config.build()), bucket))
    }

    /// Every object under `prefix`, page by page, after `offset` if given.
    fn list_from(
        &self,
        prefix: Option<&Path>,
        offset: Option<Path>,
    ) -> BoxStream<'static, Result<ObjectMeta>> {
        let (client, bucket) = (self.client.clone(), self.bucket.clone());
        let prefix = key_prefix(prefix);
        let start = offset.clone();
        let fetch = move |token: Option<String>| {
            let (client, bucket, prefix) = (client.clone(), bucket.clone(), prefix.clone());
            let start = start.as_ref().map(|s| s.to_string());
            async move {
                let page = client
                    .list_objects_v2()
                    .bucket(bucket)
                    .set_prefix((!prefix.is_empty()).then(|| prefix.clone()))
                    .set_continuation_token(token)
                    .set_start_after(start)
                    .send()
                    .await
                    .map_err(|e| errors::map(e, &prefix, Write::None))?;
                let objects = page
                    .contents()
                    .iter()
                    .filter_map(|o| meta_of(o).transpose())
                    .collect::<Result<_>>()?;
                Ok(Page {
                    objects,
                    next: page.next_continuation_token().map(str::to_string),
                })
            }
        };
        paged(fetch, offset)
    }

    /// A copy that fails with `AlreadyExists` when `to` exists: one part
    /// copied server-side into a multipart upload, which completes with
    /// `If-None-Match: *` (`CopyObject` takes no such condition). This is
    /// `object_store`'s `S3CopyIfNotExists::Multipart`.
    async fn copy_if_absent(&self, source: &str, from: &Path, to: &Path) -> Result<()> {
        let key = to.as_ref();
        let created = self
            .client
            .create_multipart_upload()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .map_err(|e| errors::map(e, key, Write::None))?;
        let upload_id = created
            .upload_id()
            .ok_or_else(|| errors::generic(format!("no upload id for {key}")))?
            .to_string();
        let part = self
            .client
            .upload_part_copy()
            .bucket(&self.bucket)
            .key(key)
            .upload_id(&upload_id)
            .part_number(1)
            .copy_source(source)
            .send()
            .await;
        let part = match part {
            Ok(part) => part,
            Err(e) => {
                let err = errors::map(e, from.as_ref(), Write::None);
                self.abort(key, &upload_id).await;
                // S3 may refuse a zero-byte source as a part. An empty object
                // needs no copy: put-if-absent writes the same thing.
                return match self.head_object(from).await {
                    Ok(meta) if meta.size == 0 => self
                        .put_opts(to, PutPayload::new(), PutMode::Create.into())
                        .await
                        .map(|_| ()),
                    _ => Err(err),
                };
            }
        };
        let result = async {
            let e_tag = part
                .copy_part_result()
                .and_then(|r| r.e_tag())
                .ok_or_else(|| errors::generic(format!("the copy of {from} has no ETag")))?;
            let completed = CompletedMultipartUpload::builder()
                .parts(CompletedPart::builder().part_number(1).e_tag(e_tag).build())
                .build();
            self.client
                .complete_multipart_upload()
                .bucket(&self.bucket)
                .key(key)
                .upload_id(&upload_id)
                .multipart_upload(completed)
                .if_none_match("*")
                .customize()
                .config_override(retry::conditional::<CompleteMultipartUploadError>())
                .send()
                .await
                .map_err(|e| errors::map(e, key, Write::Create))?;
            Ok(())
        }
        .await;
        if result.is_err() {
            self.abort(key, &upload_id).await;
        }
        result
    }

    /// Best effort: a failed abort leaves the parts to the bucket's lifecycle
    /// rule, and the caller's own error is the one that matters.
    async fn abort(&self, key: &str, upload_id: &str) {
        let _ = self
            .client
            .abort_multipart_upload()
            .bucket(&self.bucket)
            .key(key)
            .upload_id(upload_id)
            .send()
            .await;
    }

    async fn head_object(&self, location: &Path) -> Result<ObjectMeta> {
        let path = location.as_ref();
        let head = self
            .client
            .head_object()
            .bucket(&self.bucket)
            .key(path)
            .send()
            .await
            .map_err(|e| errors::map(e, path, Write::None))?;
        Ok(ObjectMeta {
            location: location.clone(),
            last_modified: timestamp(head.last_modified()),
            size: size(head.content_length()),
            e_tag: head.e_tag().map(str::to_string),
            version: head.version_id().map(str::to_string),
        })
    }
}

/// One page of a listing and the token for the next, if any.
struct Page {
    objects: Vec<ObjectMeta>,
    next: Option<String>,
}

/// A listing as one stream, calling `fetch` with each page's token until a
/// page comes back without one. Keys at or before `offset` are dropped here
/// too, in case a store ignores `start_after`.
fn paged<F, Fut>(fetch: F, offset: Option<Path>) -> BoxStream<'static, Result<ObjectMeta>>
where
    F: Fn(Option<String>) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<Page>> + Send + 'static,
{
    let fetch = Arc::new(fetch);
    // `Some(token)` is the next page to read; `None` means the last was read.
    let pages = futures::stream::try_unfold(Some(None), move |token| {
        let (fetch, offset) = (fetch.clone(), offset.clone());
        async move {
            let Some(token) = token else {
                return Ok::<_, object_store::Error>(None);
            };
            let page = fetch(token).await?;
            let metas: Vec<Result<ObjectMeta>> = page
                .objects
                .into_iter()
                .filter(|meta| match &offset {
                    Some(offset) => meta.location.as_ref() > offset.as_ref(),
                    None => true,
                })
                .map(Ok)
                .collect();
            let next = page.next.filter(|t| !t.is_empty()).map(Some);
            Ok(Some((futures::stream::iter(metas), next)))
        }
    });
    pages.try_flatten().boxed()
}

/// A listed object as an `ObjectMeta`; `None` for one with no key, or for a
/// folder marker (`name/`, as the S3 console's "Create folder" writes), which
/// `Path` would read as an object named like its own prefix.
fn meta_of(object: &aws_sdk_s3::types::Object) -> Result<Option<ObjectMeta>> {
    let Some(key) = object.key().filter(|k| !k.ends_with('/')) else {
        return Ok(None);
    };
    Ok(Some(ObjectMeta {
        location: parse(key)?,
        last_modified: timestamp(object.last_modified()),
        size: size(object.size()),
        e_tag: object.e_tag().map(str::to_string),
        version: None,
    }))
}

/// The key prefix for an `object_store` one: whole path segments, so `a`
/// becomes `a/` and never matches `ab/1`; the root is no prefix at all.
fn key_prefix(prefix: Option<&Path>) -> String {
    match prefix {
        Some(p) if !p.as_ref().is_empty() => format!("{}/", p.as_ref()),
        _ => String::new(),
    }
}

/// A listed key back to a path. Keys were written from `Path`'s own encoding,
/// so they are parsed, not encoded again.
fn parse(key: &str) -> Result<Path> {
    Path::parse(key).map_err(|e| errors::generic(e.to_string()))
}

/// Every byte but the unreserved ones and `/`, which `x-amz-copy-source`
/// takes as is.
const COPY_SOURCE: &percent_encoding::AsciiSet = &percent_encoding::NON_ALPHANUMERIC
    .remove(b'-')
    .remove(b'.')
    .remove(b'_')
    .remove(b'~')
    .remove(b'/');

/// `bucket/key` for `x-amz-copy-source`, URL-encoded as S3 expects.
fn copy_source(bucket: &str, from: &Path) -> String {
    let key = percent_encoding::utf8_percent_encode(from.as_ref(), COPY_SOURCE);
    format!("{bucket}/{key}")
}

/// `bytes a-b/total` as the range `a..b+1` and the object's size.
fn content_range(header: &str) -> Option<(Range<u64>, u64)> {
    let (span, total) = header.strip_prefix("bytes ")?.split_once('/')?;
    let (start, end) = span.split_once('-')?;
    let (start, end, total) = (
        start.parse().ok()?,
        end.parse::<u64>().ok()?,
        total.parse().ok()?,
    );
    Some((start..end + 1, total))
}

fn timestamp(t: Option<&AwsDateTime>) -> DateTime<Utc> {
    t.and_then(|t| DateTime::<Utc>::from_timestamp(t.secs(), t.subsec_nanos()))
        .unwrap_or_default()
}

fn size(length: Option<i64>) -> u64 {
    length.and_then(|n| u64::try_from(n).ok()).unwrap_or(0)
}

/// A body as a stream of chunks, fused: `bytes()` polls again after the last
/// chunk, and an empty object's body ends at once.
fn body_stream(body: ByteStream, path: &str) -> BoxStream<'static, Result<Bytes>> {
    let path = path.to_string();
    futures::stream::unfold(body, |mut body| async move {
        body.next().await.map(|chunk| (chunk, body))
    })
    .map(move |chunk| {
        chunk.map_err(|e| object_store::Error::Generic {
            store: "S3",
            source: format!("reading {path}: {e}").into(),
        })
    })
    .fuse()
    .boxed()
}

#[async_trait]
impl ObjectStore for S3Store {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> Result<PutResult> {
        if !opts.tags.encoded().is_empty() || !opts.attributes.is_empty() {
            return Err(errors::not_implemented("put_opts with tags or attributes"));
        }
        let path = location.as_ref();
        let body = Bytes::from(payload);
        let request = self
            .client
            .put_object()
            .bucket(&self.bucket)
            .key(path)
            .body(ByteStream::from(body.clone()));
        let (request, kind) = match opts.mode {
            PutMode::Overwrite => (request, Write::None),
            PutMode::Create => (request.if_none_match("*"), Write::Create),
            PutMode::Update(version) => {
                // Versions here are ETags; without one the write can't be
                // conditional, so it is refused rather than sent blind.
                let e_tag = version
                    .e_tag
                    .ok_or_else(|| object_store::Error::Precondition {
                        path: path.into(),
                        source: "an update needs the object's ETag".into(),
                    })?;
                (request.if_match(e_tag), Write::Update)
            }
        };
        let attempts = Attempts::default();
        let sent = match kind {
            Write::None => request.send().await,
            _ => {
                request
                    .customize()
                    .config_override(retry::conditional::<PutObjectError>())
                    .interceptor(attempts.clone())
                    .send()
                    .await
            }
        };
        let response = match sent {
            Ok(response) => response,
            Err(e) => {
                let err = errors::map(e, path, kind);
                // A retry fails its own condition when the attempt before it
                // landed and only the reply was lost. The object then holds
                // this body; another writer's identical bytes look the same.
                let lost = matches!(
                    err,
                    object_store::Error::AlreadyExists { .. }
                        | object_store::Error::Precondition { .. }
                );
                if lost
                    && attempts.retried()
                    && let Ok(meta) = self.head_object(location).await
                    && meta
                        .e_tag
                        .as_deref()
                        .is_some_and(|t| retry::is_e_tag_of(t, &body))
                {
                    return Ok(PutResult {
                        e_tag: meta.e_tag,
                        version: meta.version,
                        extensions: Default::default(),
                    });
                }
                return Err(err);
            }
        };
        Ok(PutResult {
            e_tag: response.e_tag().map(str::to_string),
            version: response.version_id().map(str::to_string),
            extensions: Default::default(),
        })
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> Result<Box<dyn MultipartUpload>> {
        if !opts.tags.encoded().is_empty() || !opts.attributes.is_empty() {
            return Err(errors::not_implemented(
                "put_multipart_opts with tags or attributes",
            ));
        }
        let key = location.to_string();
        let created = self
            .client
            .create_multipart_upload()
            .bucket(&self.bucket)
            .key(&key)
            .send()
            .await
            .map_err(|e| errors::map(e, &key, Write::None))?;
        let upload_id = created
            .upload_id()
            .ok_or_else(|| errors::generic(format!("no upload id for {key}")))?
            .to_string();
        Ok(Box::new(Upload {
            client: self.client.clone(),
            bucket: self.bucket.clone(),
            key,
            upload_id,
            parts: Default::default(),
            pending: Default::default(),
            last: None,
        }))
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> Result<GetResult> {
        // A cache treats a get without these as safe to keep, so dropping one
        // silently would be wrong; Itzma itself never sends them.
        if options.if_match.is_some()
            || options.if_none_match.is_some()
            || options.if_modified_since.is_some()
            || options.if_unmodified_since.is_some()
            || options.version.is_some()
        {
            return Err(errors::not_implemented(
                "get_opts with preconditions or a version",
            ));
        }
        if let Some(range) = &options.range {
            range
                .is_valid()
                .map_err(|e| errors::generic(e.to_string()))?;
        }
        if options.head {
            let meta = self.head_object(location).await?;
            return Ok(GetResult {
                payload: GetResultPayload::Stream(futures::stream::empty().boxed()),
                range: 0..meta.size,
                meta,
                attributes: Default::default(),
                extensions: Default::default(),
            });
        }
        let path = location.as_ref();
        let header = match &options.range {
            None => None,
            Some(GetRange::Bounded(r)) => Some(format!("bytes={}-{}", r.start, r.end - 1)),
            Some(GetRange::Offset(o)) => Some(format!("bytes={o}-")),
            // `bytes=-0` is not a range S3 serves; an empty suffix is empty.
            Some(GetRange::Suffix(0)) => {
                let meta = self.head_object(location).await?;
                return Ok(GetResult {
                    payload: GetResultPayload::Stream(futures::stream::empty().boxed()),
                    range: meta.size..meta.size,
                    meta,
                    attributes: Default::default(),
                    extensions: Default::default(),
                });
            }
            Some(GetRange::Suffix(n)) => Some(format!("bytes=-{n}")),
        };
        let response = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(path)
            .set_range(header)
            .send()
            .await
            .map_err(|e| errors::map(e, path, Write::None))?;
        let length = size(response.content_length());
        // A ranged response says where it starts and the object's size in
        // `Content-Range` (`bytes a-b/total`); a whole one is all of it.
        let (range, total) = match response.content_range().and_then(content_range) {
            Some((range, total)) => (range, total),
            None => (0..length, length),
        };
        let meta = ObjectMeta {
            location: location.clone(),
            last_modified: timestamp(response.last_modified()),
            size: total,
            e_tag: response.e_tag().map(str::to_string),
            version: response.version_id().map(str::to_string),
        };
        Ok(GetResult {
            payload: GetResultPayload::Stream(body_stream(response.body, path)),
            range,
            meta,
            attributes: Default::default(),
            extensions: Default::default(),
        })
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, Result<Path>>,
    ) -> BoxStream<'static, Result<Path>> {
        let (client, bucket) = (self.client.clone(), self.bucket.clone());
        locations
            .map(move |location| {
                let (client, bucket) = (client.clone(), bucket.clone());
                async move {
                    let location = location?;
                    client
                        .delete_object()
                        .bucket(bucket)
                        .key(location.as_ref())
                        .send()
                        .await
                        .map_err(|e| errors::map(e, location.as_ref(), Write::None))?;
                    Ok(location)
                }
            })
            .buffered(10)
            .boxed()
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, Result<ObjectMeta>> {
        self.list_from(prefix, None)
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, Result<ObjectMeta>> {
        self.list_from(prefix, Some(offset.clone()))
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> Result<ListResult> {
        let base = key_prefix(prefix);
        let mut objects = Vec::new();
        let mut common_prefixes = Vec::new();
        let mut token = None;
        loop {
            let page = self
                .client
                .list_objects_v2()
                .bucket(&self.bucket)
                .set_prefix((!base.is_empty()).then(|| base.clone()))
                .delimiter("/")
                .set_continuation_token(token)
                .send()
                .await
                .map_err(|e| errors::map(e, &base, Write::None))?;
            for object in page.contents() {
                objects.extend(meta_of(object)?);
            }
            for common in page.common_prefixes() {
                if let Some(p) = common.prefix() {
                    common_prefixes.push(parse(p.trim_end_matches('/'))?);
                }
            }
            token = page
                .next_continuation_token()
                .filter(|t| !t.is_empty())
                .map(str::to_string);
            if token.is_none() {
                break;
            }
        }
        Ok(ListResult {
            common_prefixes,
            objects,
            extensions: Default::default(),
        })
    }

    async fn copy_opts(&self, from: &Path, to: &Path, options: CopyOptions) -> Result<()> {
        let source = copy_source(&self.bucket, from);
        match options.mode {
            CopyMode::Overwrite => {
                self.client
                    .copy_object()
                    .bucket(&self.bucket)
                    .key(to.as_ref())
                    .copy_source(source)
                    .send()
                    .await
                    .map_err(|e| errors::map(e, from.as_ref(), Write::None))?;
                Ok(())
            }
            CopyMode::Create => self.copy_if_absent(&source, from, to).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::TryStreamExt;

    fn page(names: &[&str], next: Option<&str>) -> Page {
        Page {
            objects: names
                .iter()
                .map(|n| ObjectMeta {
                    location: Path::from(*n),
                    last_modified: Default::default(),
                    size: 0,
                    e_tag: None,
                    version: None,
                })
                .collect(),
            next: next.map(str::to_string),
        }
    }

    /// Three pages chained by continuation tokens. It ignores `start_after`,
    /// as a store that doesn't know the parameter would.
    async fn fetch(token: Option<String>) -> Result<Page> {
        Ok(match token.as_deref() {
            None => page(&["p/1", "p/2"], Some("t1")),
            Some("t1") => page(&["p/3", "p/4"], Some("t2")),
            Some("t2") => page(&["p/5"], None),
            Some(other) => panic!("unexpected token {other}"),
        })
    }

    async fn names(offset: Option<Path>) -> Vec<String> {
        paged(fetch, offset)
            .map_ok(|m| m.location.to_string())
            .try_collect()
            .await
            .expect("listing")
    }

    #[tokio::test]
    async fn paged_follows_every_token() {
        assert_eq!(names(None).await, ["p/1", "p/2", "p/3", "p/4", "p/5"]);
    }

    #[tokio::test]
    async fn paged_offset_inside_a_later_page() {
        assert_eq!(names(Some(Path::from("p/3"))).await, ["p/4", "p/5"]);
    }
}
