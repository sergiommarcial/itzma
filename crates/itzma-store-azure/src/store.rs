use std::fmt;
use std::num::NonZero;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use azure_core::credentials::TokenCredential;
use azure_core::http::headers::{HeaderName, Headers};
use azure_core::http::{Etag, NoFormat, RequestContent, Url};
use azure_identity::{
    DeveloperToolsCredential, ManagedIdentityCredential, WorkloadIdentityCredential,
};
use azure_storage_blob::BlobContainerClient;
use azure_storage_blob::models::{
    BlobClientDownloadOptions, BlobClientGetPropertiesResultHeaders,
    BlobContainerClientListBlobsOptions, BlobItem, BlockBlobClientUploadOptions, HttpRange,
    ListBlobsIncludeItem, ListBlobsResponse,
};
use bytes::Bytes;
use chrono::{DateTime, Utc};
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use object_store::path::Path;
use object_store::{
    CopyMode, CopyOptions, GetOptions, GetRange, GetResult, GetResultPayload, ListResult,
    MultipartUpload, ObjectMeta, ObjectStore, ObjectStoreExt, PutMode, PutMultipartOptions,
    PutOptions, PutPayload, PutResult, Result,
};

use crate::errors::{self, Write};
use crate::upload::Staged;

/// How [`AzureStore::connect`] authenticates. The SDK has no default chain,
/// so the choice is explicit. Token credentials need an `https` endpoint.
#[non_exhaustive]
#[derive(Clone, Default)]
pub enum AzureCredential {
    /// The managed identity of the VM, App Service or AKS node.
    #[default]
    ManagedIdentity,
    /// AKS workload identity (a federated token in the pod's environment).
    WorkloadIdentity,
    /// The developer's `az login` or `azd auth login`.
    DeveloperTools,
    /// A SAS token, sent in the URL's query (with or without a leading `?`).
    Sas(String),
    /// No credentials, for a public container or an emulator.
    Anonymous,
}

impl fmt::Debug for AzureCredential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::ManagedIdentity => "ManagedIdentity",
            Self::WorkloadIdentity => "WorkloadIdentity",
            Self::DeveloperTools => "DeveloperTools",
            Self::Sas(_) => "Sas(..)",
            Self::Anonymous => "Anonymous",
        })
    }
}

/// Settings for [`AzureStore::connect`].
#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub struct AzureOptions {
    /// The account's blob endpoint; `https://<account>.blob.core.windows.net`
    /// when unset. Azurite's is `http://127.0.0.1:10000/devstoreaccount1`.
    pub endpoint: Option<String>,
    /// How to authenticate.
    pub credential: AzureCredential,
}

impl AzureOptions {
    /// Sets [`endpoint`](Self::endpoint).
    pub fn endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }

    /// Sets [`credential`](Self::credential).
    pub fn credential(mut self, credential: AzureCredential) -> Self {
        self.credential = credential;
        self
    }
}

/// An [`ObjectStore`] over one Azure Blob Storage container through the
/// official SDK.
pub struct AzureStore {
    container: Arc<BlobContainerClient>,
    name: String,
}

impl fmt::Debug for AzureStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AzureStore")
            .field("container", &self.name)
            .finish()
    }
}

impl fmt::Display for AzureStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AzureStore(az://{})", self.name)
    }
}

impl AzureStore {
    /// A store over the container `container` points at, built by the caller
    /// with whatever credential and options they need.
    pub fn new(container: BlobContainerClient) -> Self {
        let name = container
            .url()
            .path_segments()
            .and_then(|mut s| s.rfind(|segment| !segment.is_empty()).map(str::to_string))
            .unwrap_or_default();
        Self {
            container: Arc::new(container),
            name,
        }
    }

    /// Builds the container client for `account` and `container` from
    /// `options`, then a store over it.
    ///
    /// # Errors
    ///
    /// `Error::Generic` for a token credential over `http` (the SDK sends
    /// tokens over https only), a bad endpoint, or a credential that cannot be
    /// built.
    pub async fn connect(account: &str, container: &str, options: AzureOptions) -> Result<Self> {
        let endpoint = options
            .endpoint
            .clone()
            .unwrap_or_else(|| format!("https://{account}.blob.core.windows.net"));
        // A pasted "Blob SAS URL" would put the container inside its query;
        // the SAS goes in `credential` instead.
        let account_url = Url::parse(&endpoint).map_err(|e| {
            errors::generic(errors::redact(&format!("bad endpoint {endpoint:?}: {e}")))
        })?;
        if account_url.query().is_some() || account_url.fragment().is_some() {
            return Err(errors::generic(
                "the endpoint has a query or fragment; pass a SAS as the credential, not in the endpoint",
            ));
        }
        let mut url = Url::parse(&format!("{}/{container}", endpoint.trim_end_matches('/')))
            .map_err(|e| {
                errors::generic(errors::redact(&format!("bad endpoint {endpoint:?}: {e}")))
            })?;
        let token: Option<Arc<dyn TokenCredential>> = match &options.credential {
            AzureCredential::Sas(sas) => {
                url.set_query(Some(sas.trim_start_matches('?')));
                None
            }
            AzureCredential::Anonymous => None,
            AzureCredential::ManagedIdentity => Some(
                ManagedIdentityCredential::new(None).map_err(|e| errors::generic(e.to_string()))?,
            ),
            AzureCredential::WorkloadIdentity => Some(
                WorkloadIdentityCredential::new(None)
                    .map_err(|e| errors::generic(e.to_string()))?,
            ),
            AzureCredential::DeveloperTools => Some(
                DeveloperToolsCredential::new(None).map_err(|e| errors::generic(e.to_string()))?,
            ),
        };
        if token.is_some() && url.scheme() != "https" {
            return Err(errors::generic(
                "a token credential needs an https endpoint; use a SAS token or anonymous access for http (Azurite)",
            ));
        }
        let client = BlobContainerClient::new(url, token, None)
            .map_err(|e| errors::generic(errors::redact(&e.to_string())))?;
        Ok(Self::new(client))
    }

    /// Every object under `prefix`, page by page, after `offset` if given.
    fn list_from(
        &self,
        prefix: Option<&Path>,
        offset: Option<Path>,
    ) -> BoxStream<'static, Result<ObjectMeta>> {
        let container = self.container.clone();
        let prefix = blob_prefix(prefix);
        let start = offset.clone();
        let fetch = move |marker: String| {
            let (container, prefix, start) = (container.clone(), prefix.clone(), start.clone());
            async move {
                let options = BlobContainerClientListBlobsOptions {
                    // Metadata marks the directories a hierarchical-namespace
                    // account lists as blobs.
                    include: Some(vec![ListBlobsIncludeItem::Metadata]),
                    prefix: (!prefix.is_empty()).then(|| prefix.clone()),
                    marker: (!marker.is_empty()).then_some(marker),
                    start_from: start.map(|s| s.to_string()),
                    ..Default::default()
                };
                let first = container
                    .list_blobs(Some(options))
                    .map_err(|e| errors::map(e, &prefix, Write::None))?
                    .into_pages()
                    .try_next()
                    .await
                    .map_err(|e| errors::map(e, &prefix, Write::None))?;
                match first {
                    Some(page) => page
                        .into_model()
                        .map_err(|e| errors::map(e, &prefix, Write::None)),
                    None => Ok(ListBlobsResponse::default()),
                }
            }
        };
        paged(fetch, offset)
    }

    async fn head_object(&self, location: &Path) -> Result<ObjectMeta> {
        let path = location.as_ref();
        let response = self
            .container
            .blob_client(path)
            .get_properties(None)
            .await
            .map_err(|e| errors::map(e, path, Write::None))?;
        let size = response
            .content_length()
            .map_err(|e| errors::map(e, path, Write::None))?
            .unwrap_or(0);
        let e_tag = response
            .etag()
            .map_err(|e| errors::map(e, path, Write::None))?
            .map(|e| e.to_string());
        let last_modified = response
            .last_modified()
            .map_err(|e| errors::map(e, path, Write::None))?
            .and_then(|t| DateTime::<Utc>::from_timestamp(t.unix_timestamp(), t.nanosecond()))
            .unwrap_or_default();
        Ok(ObjectMeta {
            location: location.clone(),
            last_modified,
            size,
            e_tag,
            version: None,
        })
    }
}

/// A listing as one stream, calling `fetch` with each page's marker until a
/// page comes back without one. Keys at or before `offset` are dropped here
/// too: `start_from` is inclusive, and a service that ignores it returns
/// earlier keys.
fn paged<F, Fut>(fetch: F, offset: Option<Path>) -> BoxStream<'static, Result<ObjectMeta>>
where
    F: Fn(String) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<ListBlobsResponse>> + Send + 'static,
{
    let fetch = Arc::new(fetch);
    // `Some(marker)` is the next page to read; `None` means the last was read.
    let pages = futures::stream::try_unfold(Some(String::new()), move |marker| {
        let (fetch, offset) = (fetch.clone(), offset.clone());
        async move {
            let Some(marker) = marker else {
                return Ok::<_, object_store::Error>(None);
            };
            let page = fetch(marker).await?;
            let metas: Vec<Result<ObjectMeta>> = page
                .blob_items
                .iter()
                .filter_map(|item| meta_of(item).transpose())
                .filter(|meta| match (meta, &offset) {
                    (Ok(meta), Some(offset)) => meta.location.as_ref() > offset.as_ref(),
                    _ => true,
                })
                .collect();
            let next = page.next_marker.filter(|m| !m.is_empty());
            Ok(Some((futures::stream::iter(metas), next)))
        }
    });
    pages.try_flatten().boxed()
}

/// A listed blob as an `ObjectMeta`; `None` for an item with no name.
fn meta_of(item: &BlobItem) -> Result<Option<ObjectMeta>> {
    let Some(name) = &item.name else {
        return Ok(None);
    };
    let folder = item
        .metadata
        .as_ref()
        .and_then(|m| m.values.as_ref())
        .and_then(|v| v.get("hdi_isfolder"))
        .is_some_and(|v| v.eq_ignore_ascii_case("true"));
    if folder {
        return Ok(None);
    }
    let properties = item.properties.as_ref();
    Ok(Some(ObjectMeta {
        location: parse(name)?,
        last_modified: properties
            .and_then(|p| p.last_modified)
            .and_then(|t| DateTime::<Utc>::from_timestamp(t.unix_timestamp(), t.nanosecond()))
            .unwrap_or_default(),
        size: properties.and_then(|p| p.content_length).unwrap_or(0),
        e_tag: properties
            .and_then(|p| p.etag.as_ref())
            .map(|e| e.to_string()),
        version: None,
    }))
}

/// The blob prefix for an `object_store` one: whole path segments, so `a`
/// becomes `a/` and never matches `ab/1`; the root is no prefix at all.
fn blob_prefix(prefix: Option<&Path>) -> String {
    match prefix {
        Some(p) if !p.as_ref().is_empty() => format!("{}/", p.as_ref()),
        _ => String::new(),
    }
}

/// A listed name back to a path. Names were written from `Path`'s own
/// encoding, so they are parsed, not encoded again.
fn parse(name: &str) -> Result<Path> {
    Path::parse(name).map_err(|e| errors::generic(e.to_string()))
}

/// The object's size from a ranged response's `Content-Range`
/// (`bytes a-b/total`), or `None` for a whole-object response.
fn total_size(headers: &Headers) -> Option<u64> {
    headers
        .get_optional_str(&HeaderName::from_static("content-range"))?
        .rsplit_once('/')?
        .1
        .parse()
        .ok()
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
impl ObjectStore for AzureStore {
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
        let mut options = BlockBlobClientUploadOptions {
            blob_content_type: Some("application/octet-stream".into()),
            ..Default::default()
        };
        let kind = match opts.mode {
            PutMode::Overwrite => Write::None,
            PutMode::Create => {
                options.if_none_match = Some(Etag::from("*"));
                Write::Create
            }
            PutMode::Update(version) => {
                // Azure versions are ETags; without one the write can't be
                // conditional, so it is refused rather than sent blind.
                let e_tag = version
                    .e_tag
                    .ok_or_else(|| object_store::Error::Precondition {
                        path: path.into(),
                        source: "an update needs the object's ETag".into(),
                    })?;
                options.if_match = Some(Etag::from(e_tag));
                Write::Update
            }
        };
        let response = self
            .container
            .blob_client(path)
            .upload(
                <RequestContent<Bytes, NoFormat> as From<Bytes>>::from(Bytes::from(payload)),
                Some(options),
            )
            .await
            .map_err(|e| errors::map(e, path, kind))?;
        Ok(PutResult {
            e_tag: response.etag.map(|e| e.to_string()),
            version: None,
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
        Ok(Box::new(Staged {
            container: self.container.clone(),
            name: location.to_string(),
            nonce: crate::upload::nonce(),
            blocks: Vec::new(),
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
        // `x-ms-range` has no suffix form, so a suffix read learns the size
        // first; every other read takes the size from its own response.
        let (http_range, known) = match &options.range {
            None => (None, None),
            Some(GetRange::Bounded(r)) => (Some(HttpRange::from(r.clone())), None),
            Some(GetRange::Offset(o)) => (Some(HttpRange::from(*o..)), None),
            Some(GetRange::Suffix(n)) => {
                let meta = self.head_object(location).await?;
                let start = meta.size.saturating_sub(*n);
                if start == meta.size {
                    return Ok(GetResult {
                        payload: GetResultPayload::Stream(futures::stream::empty().boxed()),
                        range: start..start,
                        meta,
                        attributes: Default::default(),
                        extensions: Default::default(),
                    });
                }
                (Some(HttpRange::from(start..)), Some(meta))
            }
        };
        let download = BlobClientDownloadOptions {
            range: http_range,
            // A suffix read is sized by the HEAD before it; pinning the GET to
            // that ETag makes a replacement in between a failure, not a read
            // whose range belongs to another object.
            if_match: known.as_ref().and_then(|m| m.e_tag.clone()).map(Etag::from),
            // Large reads fan out in 4 MiB partitions; the SDK's default scales
            // with cores, which buffers hundreds of MiB on a large VM.
            parallel: NonZero::new(4),
            ..Default::default()
        };
        let response = self
            .container
            .blob_client(path)
            .download(Some(download))
            .await
            .map_err(|e| errors::map(e, path, Write::None))?;
        let meta = match known {
            Some(meta) => meta,
            None => {
                let size = total_size(&response.headers)
                    .or(response.properties.content_length)
                    .unwrap_or(0);
                ObjectMeta {
                    location: location.clone(),
                    last_modified: response
                        .properties
                        .last_modified
                        .and_then(|t| {
                            DateTime::<Utc>::from_timestamp(t.unix_timestamp(), t.nanosecond())
                        })
                        .unwrap_or_default(),
                    size,
                    e_tag: response.properties.etag.as_ref().map(|e| e.to_string()),
                    version: None,
                }
            }
        };
        let range = resolved(&options.range, meta.size);
        let owned = path.to_string();
        let stream = response
            .body
            .map_err(move |e| errors::map(e, &owned, Write::None))
            .boxed();
        Ok(GetResult {
            payload: GetResultPayload::Stream(stream),
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
        let container = self.container.clone();
        locations
            .map(move |location| {
                let container = container.clone();
                async move {
                    let location = location?;
                    container
                        .blob_client(location.as_ref())
                        .delete(None)
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
        // The SDK lists flat only, so the next path segment is split off here:
        // a name with another `/` after the prefix is a common prefix.
        let base = blob_prefix(prefix);
        let mut objects = Vec::new();
        let mut prefixes = std::collections::BTreeSet::new();
        let mut all = self.list_from(prefix, None);
        while let Some(meta) = all.try_next().await? {
            let name = meta.location.as_ref();
            let rest = name.strip_prefix(base.as_str()).unwrap_or(name);
            match rest.split_once('/') {
                Some((first, _)) => {
                    prefixes.insert(format!("{base}{first}"));
                }
                None => objects.push(meta),
            }
        }
        let common_prefixes = prefixes.iter().map(|p| parse(p)).collect::<Result<_>>()?;
        Ok(ListResult {
            common_prefixes,
            objects,
            extensions: Default::default(),
        })
    }

    async fn copy_opts(&self, from: &Path, to: &Path, options: CopyOptions) -> Result<()> {
        // Through the client: a server-side copy needs the source authorized
        // on its own under a token credential.
        let body = self.get(from).await?.bytes().await?;
        let mode = match options.mode {
            CopyMode::Create => PutMode::Create,
            CopyMode::Overwrite => PutMode::Overwrite,
        };
        self.put_opts(to, PutPayload::from(body), PutOptions::from(mode))
            .await
            .map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use azure_storage_blob::models::BlobMetadata;

    fn page(names: &[&str], next: Option<&str>) -> ListBlobsResponse {
        // The SDK's models are non-exhaustive: built from `Default`.
        let mut page = ListBlobsResponse::default();
        page.blob_items = names
            .iter()
            .map(|n| {
                let mut item = BlobItem::default();
                item.name = Some(n.to_string());
                item
            })
            .collect();
        page.next_marker = next.map(str::to_string);
        page
    }

    /// Three pages chained by markers. It ignores `start_from`, as a service
    /// that doesn't know the parameter would.
    async fn fetch(marker: String) -> Result<ListBlobsResponse> {
        Ok(match marker.as_str() {
            "" => page(&["p/1", "p/2"], Some("m1")),
            "m1" => page(&["p/3", "p/4"], Some("m2")),
            "m2" => page(&["p/5"], None),
            other => panic!("unexpected marker {other}"),
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
    async fn paged_follows_every_marker() {
        assert_eq!(names(None).await, ["p/1", "p/2", "p/3", "p/4", "p/5"]);
    }

    #[tokio::test]
    async fn paged_skips_directory_entries() {
        // A hierarchical-namespace account lists its directories as blobs
        // marked `hdi_isfolder`; they are not objects.
        async fn fetch(_marker: String) -> Result<ListBlobsResponse> {
            let mut page = page(&["ns/wal/1", "ns/wal"], None);
            let mut metadata = BlobMetadata::default();
            metadata.values = Some([("hdi_isfolder".to_string(), "true".to_string())].into());
            page.blob_items[1].metadata = Some(metadata);
            Ok(page)
        }
        let names: Vec<String> = paged(fetch, None)
            .map_ok(|m| m.location.to_string())
            .try_collect()
            .await
            .expect("listing");
        assert_eq!(names, ["ns/wal/1"]);
    }

    #[tokio::test]
    async fn paged_offset_inside_a_later_page() {
        assert_eq!(names(Some(Path::from("p/3"))).await, ["p/4", "p/5"]);
    }
}
