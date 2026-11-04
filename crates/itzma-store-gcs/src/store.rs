use std::fmt;
use std::ops::Range;

use async_trait::async_trait;
use bytes::Bytes;
use chrono::{DateTime, Utc};
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use google_cloud_storage::client::{Storage, StorageControl};
use google_cloud_storage::model::{ListObjectsResponse, Object};
use google_cloud_storage::model_ext::ReadRange;
use object_store::path::Path;
use object_store::{
    CopyMode, CopyOptions, GetOptions, GetRange, GetResult, GetResultPayload, ListResult,
    MultipartUpload, ObjectMeta, ObjectStore, PutMode, PutMultipartOptions, PutOptions, PutPayload,
    PutResult, Result, UpdateVersion,
};

use crate::errors::{self, Write};
use crate::upload::Buffered;

/// Settings for [`GcsStore::connect`]. Everything left unset uses the SDK's
/// defaults: the production endpoints and the default credentials, which on
/// GKE and Cloud Run come from Workload Identity.
#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub struct GcsOptions {
    /// The JSON API endpoint, for an emulator such as storage-testbench.
    pub endpoint: Option<String>,
    /// The gRPC endpoint for `StorageControl` (listings, deletes, copies).
    pub control_endpoint: Option<String>,
    /// Send no credentials. For an emulator only.
    pub anonymous: bool,
}

impl GcsOptions {
    /// Sets [`endpoint`](Self::endpoint).
    pub fn endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }

    /// Sets [`control_endpoint`](Self::control_endpoint).
    pub fn control_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.control_endpoint = Some(endpoint.into());
        self
    }

    /// Sets [`anonymous`](Self::anonymous).
    pub fn anonymous(mut self, anonymous: bool) -> Self {
        self.anonymous = anonymous;
        self
    }
}

/// An [`ObjectStore`] over one GCS bucket through the official SDK.
///
/// `get` reports the object's size and generation but not its etag or
/// modification time (the download response doesn't carry the object's own
/// etag); `head` and listings report all four.
pub struct GcsStore {
    storage: Storage,
    control: StorageControl,
    bucket: String,
    resource: String,
}

impl fmt::Debug for GcsStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GcsStore")
            .field("bucket", &self.bucket)
            .finish()
    }
}

impl fmt::Display for GcsStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GcsStore(gs://{})", self.bucket)
    }
}

impl GcsStore {
    /// A store over `bucket` using clients the caller built, so every SDK
    /// setting (credentials, retries, endpoints) is the caller's.
    pub fn new(storage: Storage, control: StorageControl, bucket: impl Into<String>) -> Self {
        let bucket = bucket.into();
        let resource = format!("projects/_/buckets/{bucket}");
        Self {
            storage,
            control,
            bucket,
            resource,
        }
    }

    /// Builds both clients from `options`, then a store over `bucket`.
    ///
    /// # Errors
    ///
    /// `Error::Generic` if only one of the two endpoints is set, or if a client
    /// cannot be built, for example when no default credentials are found and
    /// `anonymous` is not set.
    pub async fn connect(bucket: &str, options: GcsOptions) -> Result<Self> {
        // One without the other would send reads and writes to one place and
        // listings, deletes and copies to production.
        if options.endpoint.is_some() != options.control_endpoint.is_some() {
            return Err(object_store::Error::Generic {
                store: "GCS",
                source: "set endpoint and control_endpoint together, or neither".into(),
            });
        }
        let mut storage = Storage::builder();
        let mut control = StorageControl::builder();
        if let Some(endpoint) = &options.endpoint {
            storage = storage.with_endpoint(endpoint);
        }
        if let Some(endpoint) = &options.control_endpoint {
            control = control.with_endpoint(endpoint);
        }
        if options.anonymous {
            let credentials = google_cloud_auth::credentials::anonymous::Builder::new().build();
            storage = storage.with_credentials(credentials.clone());
            control = control.with_credentials(credentials);
        }
        let generic = |e: Box<dyn std::error::Error + Send + Sync>| object_store::Error::Generic {
            store: "GCS",
            source: e,
        };
        let storage = storage.build().await.map_err(|e| generic(Box::new(e)))?;
        let control = control.build().await.map_err(|e| generic(Box::new(e)))?;
        Ok(Self::new(storage, control, bucket))
    }

    fn meta(location: &Path, object: &Object) -> ObjectMeta {
        let last_modified = object
            .update_time
            .as_ref()
            .and_then(|t| DateTime::<Utc>::from_timestamp(t.seconds(), t.nanos().max(0) as u32))
            .unwrap_or_default();
        ObjectMeta {
            location: location.clone(),
            last_modified,
            size: object.size.max(0) as u64,
            e_tag: (!object.etag.is_empty()).then(|| object.etag.clone()),
            version: Some(object.generation.to_string()),
        }
    }

    /// Every object under `prefix`, page by page, after `offset` if given.
    fn list_from(
        &self,
        prefix: Option<&Path>,
        offset: Option<Path>,
    ) -> BoxStream<'static, Result<ObjectMeta>> {
        let control = self.control.clone();
        let resource = self.resource.clone();
        let prefix = gcs_prefix(prefix);
        let start = offset.clone();
        let fetch = move |token: String| {
            let (control, resource, prefix, start) = (
                control.clone(),
                resource.clone(),
                prefix.clone(),
                start.clone(),
            );
            async move {
                let mut request = control
                    .list_objects()
                    .set_parent(resource)
                    .set_prefix(prefix.clone())
                    .set_page_token(token);
                if let Some(start) = &start {
                    request = request.set_lexicographic_start(start.as_ref());
                }
                request
                    .send()
                    .await
                    .map_err(|e| errors::map(e, &prefix, Write::None))
            }
        };
        paged(fetch, offset)
    }

    async fn head_object(&self, location: &Path) -> Result<ObjectMeta> {
        let object = self
            .control
            .get_object()
            .set_bucket(&self.resource)
            .set_object(location.as_ref())
            .send()
            .await
            .map_err(|e| errors::map(e, location.as_ref(), Write::None))?;
        Ok(Self::meta(location, &object))
    }
}

/// A listing as one stream, calling `fetch` with each page's token until a
/// page comes back without one. GCS starts a listing at the offset itself,
/// and `object_store` lists strictly after it, so that entry is dropped.
fn paged<F, Fut>(fetch: F, offset: Option<Path>) -> BoxStream<'static, Result<ObjectMeta>>
where
    F: Fn(String) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<ListObjectsResponse>> + Send + 'static,
{
    let fetch = std::sync::Arc::new(fetch);
    // `Some(token)` is the next page to read; `None` means the last one was read.
    let pages = futures::stream::try_unfold(Some(String::new()), move |token| {
        let (fetch, offset) = (fetch.clone(), offset.clone());
        async move {
            let Some(token) = token else {
                return Ok::<_, object_store::Error>(None);
            };
            let page = fetch(token).await?;
            let metas: Vec<Result<ObjectMeta>> = page
                .objects
                .iter()
                .map(|object| Ok(GcsStore::meta(&parse(&object.name)?, object)))
                .filter(|meta| match (meta, &offset) {
                    (Ok(meta), Some(offset)) => meta.location != *offset,
                    _ => true,
                })
                .collect();
            let next = (!page.next_page_token.is_empty()).then_some(page.next_page_token);
            Ok(Some((futures::stream::iter(metas), next)))
        }
    });
    pages.try_flatten().boxed()
}

/// The GCS prefix for an `object_store` one: whole path segments, so `a`
/// becomes `a/` and never matches `ab/1`; the root is no prefix at all.
fn gcs_prefix(prefix: Option<&Path>) -> String {
    match prefix {
        Some(p) if !p.as_ref().is_empty() => format!("{}/", p.as_ref()),
        _ => String::new(),
    }
}

/// A listed name back to a path. Names were written from `Path`'s own
/// encoding, so they are parsed, not encoded again.
fn parse(name: &str) -> Result<Path> {
    Path::parse(name).map_err(|e| object_store::Error::Generic {
        store: "GCS",
        source: Box::new(e),
    })
}

/// The generation an update must match. Only `version` carries it: an
/// e-tag alone, or nothing, cannot make the write conditional.
fn generation(version: &UpdateVersion, path: &str) -> Result<i64> {
    version
        .version
        .as_deref()
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| object_store::Error::Precondition {
            path: path.into(),
            source: "an update needs the object's generation as its version".into(),
        })
}

/// Where a ranged read starts and ends in an object of `size` bytes.
fn resolved(range: &Option<GetRange>, size: u64) -> Range<u64> {
    match range {
        None => 0..size,
        Some(GetRange::Bounded(r)) => r.start..r.end.min(size),
        Some(GetRange::Offset(o)) => *o..size,
        Some(GetRange::Suffix(n)) => size.saturating_sub(*n)..size,
    }
}

#[async_trait]
impl ObjectStore for GcsStore {
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
        let write = self
            .storage
            .write_object(&self.resource, path, Bytes::from(payload));
        let (write, kind) = match opts.mode {
            // The SDK retries only conditional uploads by default. Writing the
            // same bytes twice to one key is harmless, so overwrites retry too.
            PutMode::Overwrite => (write.with_idempotency(true), Write::None),
            PutMode::Create => (write.set_if_generation_match(0), Write::Create),
            PutMode::Update(version) => {
                let generation = generation(&version, path)?;
                (write.set_if_generation_match(generation), Write::Update)
            }
        };
        let object = write
            .send_buffered()
            .await
            .map_err(|e| errors::map(e, path, kind))?;
        Ok(PutResult {
            e_tag: (!object.etag.is_empty()).then(|| object.etag.clone()),
            version: Some(object.generation.to_string()),
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
        Ok(Box::new(Buffered {
            storage: self.storage.clone(),
            resource: self.resource.clone(),
            name: location.to_string(),
            parts: Vec::new(),
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
        // The SDK reads `segment(a, 0)` as "to the end", so an empty or
        // inverted range must be refused here, as object_store's stores do.
        if let Some(range) = &options.range {
            range.is_valid().map_err(|e| object_store::Error::Generic {
                store: "GCS",
                source: Box::new(e),
            })?;
        }
        let mut read = self.storage.read_object(&self.resource, location.as_ref());
        match &options.range {
            None => {}
            Some(GetRange::Bounded(r)) => {
                read =
                    read.set_read_range(ReadRange::segment(r.start, r.end.saturating_sub(r.start)))
            }
            Some(GetRange::Offset(o)) => read = read.set_read_range(ReadRange::offset(*o)),
            Some(GetRange::Suffix(n)) => read = read.set_read_range(ReadRange::tail(*n)),
        }
        let response = read
            .send()
            .await
            .map_err(|e| errors::map(e, location.as_ref(), Write::None))?;
        let highlights = response.object();
        let size = highlights.size.max(0) as u64;
        let path = location.to_string();
        // `into_stream` needs the SDK's `unstable-stream` feature; `next` is stable.
        // Fused because `bytes()` polls again after the last chunk, and an
        // `unfold` that has ended panics if polled.
        let stream = futures::stream::unfold(response, |mut response| async move {
            let chunk = response.next().await?;
            Some((chunk, response))
        })
        .fuse()
        .map_err(move |e| errors::map(e, &path, Write::None))
        .boxed();
        Ok(GetResult {
            payload: GetResultPayload::Stream(stream),
            range: resolved(&options.range, size),
            meta: ObjectMeta {
                location: location.clone(),
                last_modified: DateTime::<Utc>::default(),
                size,
                e_tag: None,
                version: Some(highlights.generation.to_string()),
            },
            attributes: Default::default(),
            extensions: Default::default(),
        })
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, Result<Path>>,
    ) -> BoxStream<'static, Result<Path>> {
        let control = self.control.clone();
        let resource = self.resource.clone();
        locations
            .map(move |location| {
                let control = control.clone();
                let resource = resource.clone();
                async move {
                    let location = location?;
                    control
                        .delete_object()
                        .set_bucket(resource)
                        .set_object(location.as_ref())
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
        let mut objects = Vec::new();
        let mut common_prefixes = Vec::new();
        let mut token = String::new();
        loop {
            let page = self
                .control
                .list_objects()
                .set_parent(&self.resource)
                .set_prefix(gcs_prefix(prefix))
                .set_delimiter("/")
                .set_page_token(token)
                .send()
                .await
                .map_err(|e| errors::map(e, &gcs_prefix(prefix), Write::None))?;
            for object in &page.objects {
                objects.push(Self::meta(&parse(&object.name)?, object));
            }
            for name in &page.prefixes {
                common_prefixes.push(parse(name.trim_end_matches('/'))?);
            }
            if page.next_page_token.is_empty() {
                return Ok(ListResult {
                    common_prefixes,
                    objects,
                    extensions: Default::default(),
                });
            }
            token = page.next_page_token;
        }
    }

    async fn copy_opts(&self, from: &Path, to: &Path, options: CopyOptions) -> Result<()> {
        let create = matches!(options.mode, CopyMode::Create);
        let kind = if create { Write::Create } else { Write::None };
        // A rewrite may take several calls for a large object; each returns a
        // token to continue with until it reports `done`.
        let mut token = String::new();
        loop {
            let mut request = self
                .control
                .rewrite_object()
                .set_source_bucket(&self.resource)
                .set_source_object(from.as_ref())
                .set_destination_bucket(&self.resource)
                .set_destination_name(to.as_ref())
                .set_rewrite_token(token);
            if create {
                request = request.set_if_generation_match(0);
            }
            let response = request.send().await.map_err(|e| {
                let err = errors::map(e, from.as_ref(), kind);
                // The conflict is at the target; a missing source stays named.
                match err {
                    object_store::Error::AlreadyExists { source, .. } => {
                        object_store::Error::AlreadyExists {
                            path: to.to_string(),
                            source,
                        }
                    }
                    other => other,
                }
            })?;
            if response.done {
                return Ok(());
            }
            token = response.rewrite_token;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn page(names: &[&str], next: &str) -> ListObjectsResponse {
        ListObjectsResponse::new()
            .set_objects(names.iter().map(|n| Object::new().set_name(*n)))
            .set_next_page_token(next)
    }

    /// Three pages chained by tokens, as real GCS returns a long listing.
    async fn fetch(token: String) -> Result<ListObjectsResponse> {
        Ok(match token.as_str() {
            "" => page(&["p/1", "p/2"], "t1"),
            "t1" => page(&["p/3"], "t2"),
            "t2" => page(&["p/4", "p/5"], ""),
            other => panic!("unexpected token {other}"),
        })
    }

    #[tokio::test]
    async fn paged_follows_every_token() {
        let names: Vec<String> = paged(fetch, None)
            .map_ok(|m| m.location.to_string())
            .try_collect()
            .await
            .expect("listing");
        assert_eq!(names, ["p/1", "p/2", "p/3", "p/4", "p/5"]);
    }

    #[tokio::test]
    async fn paged_drops_the_offset_itself() {
        let names: Vec<String> = paged(fetch, Some(Path::from("p/3")))
            .map_ok(|m| m.location.to_string())
            .try_collect()
            .await
            .expect("listing");
        assert_eq!(names, ["p/1", "p/2", "p/4", "p/5"]);
    }
}
