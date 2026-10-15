//! A store that counts every read, so a test can see how many requests and
//! bytes a call actually sends over the wire, and to which objects.

use std::fmt;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use async_trait::async_trait;
use futures::stream::BoxStream;
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult,
};

/// An `InMemory` store that counts every `get`: how many, how many bytes, and
/// which paths.
#[derive(Debug, Default)]
pub struct CountingStore {
    inner: object_store::memory::InMemory,
    requests: AtomicUsize,
    bytes: AtomicU64,
    paths: Mutex<Vec<String>>,
    lists: AtomicUsize,
    copying: AtomicBool,
}

impl CountingStore {
    /// A counting layer over an *existing* `InMemory` (which shares its data
    /// on clone), so two independent counters can see the same objects and a
    /// test can tell which of two wrapping layers a request reached.
    pub fn wrapping(inner: object_store::memory::InMemory) -> Self {
        Self {
            inner,
            ..Self::default()
        }
    }

    /// Hand every body back as fresh copies of 64 KiB, as a network store's
    /// response arrives, instead of one view of the stored object: a memory
    /// test then sees the bytes a read keeps alive.
    pub fn copying(self) -> Self {
        self.copying.store(true, Ordering::SeqCst);
        self
    }

    pub fn reset(&self) {
        self.requests.store(0, Ordering::SeqCst);
        self.bytes.store(0, Ordering::SeqCst);
        self.paths.lock().unwrap().clear();
        self.lists.store(0, Ordering::SeqCst);
    }

    pub fn requests(&self) -> usize {
        self.requests.load(Ordering::SeqCst)
    }

    pub fn bytes(&self) -> u64 {
        self.bytes.load(Ordering::SeqCst)
    }

    /// How many listings (any kind) since the last reset.
    pub fn lists(&self) -> usize {
        self.lists.load(Ordering::SeqCst)
    }

    /// The paths of every `get` since the last reset, in order.
    pub fn paths(&self) -> Vec<String> {
        self.paths.lock().unwrap().clone()
    }
}

impl fmt::Display for CountingStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CountingStore")
    }
}

#[async_trait]
impl ObjectStore for CountingStore {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(
        &self,
        location: &Path,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        let got = self.inner.get_opts(location, options).await?;
        self.requests.fetch_add(1, Ordering::SeqCst);
        self.bytes
            .fetch_add(got.range.end - got.range.start, Ordering::SeqCst);
        self.paths.lock().unwrap().push(location.to_string());
        if !self.copying.load(Ordering::SeqCst) {
            return Ok(got);
        }
        let (meta, range, attributes, extensions) = (
            got.meta.clone(),
            got.range.clone(),
            got.attributes.clone(),
            got.extensions.clone(),
        );
        const PIECE: usize = 64 * 1024;
        let body = got.bytes().await?;
        let pieces = (0..body.len()).step_by(PIECE).map(move |at| {
            Ok(bytes::Bytes::copy_from_slice(
                &body[at..(at + PIECE).min(body.len())],
            ))
        });
        Ok(GetResult {
            payload: object_store::GetResultPayload::Stream(Box::pin(futures::stream::iter(
                pieces,
            ))),
            meta,
            range,
            attributes,
            extensions,
        })
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.lists.fetch_add(1, Ordering::SeqCst);
        self.inner.list(prefix)
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.lists.fetch_add(1, Ordering::SeqCst);
        self.inner.list_with_offset(prefix, offset)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> object_store::Result<ListResult> {
        self.lists.fetch_add(1, Ordering::SeqCst);
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &Path,
        to: &Path,
        options: CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, options).await
    }
}
