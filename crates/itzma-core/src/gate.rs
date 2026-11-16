//! IO admission control: a store wrapper that bounds the reads in flight by
//! count and by bytes, so nested fan-outs and several queries on one process
//! cannot open an unbounded number of requests.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetRange, GetResult, GetResultPayload, ListResult, MultipartUpload,
    ObjectMeta, ObjectStore, PutMultipartOptions, PutOptions, PutPayload, PutResult, Result,
};
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;

use crate::error::ItzmaError;

/// Default [`GateOptions::max_bytes`] of a query gate: 64 MiB in flight.
pub const DEFAULT_GATE_MAX_BYTES: u64 = 64 << 20;
/// Default [`GateOptions::max_requests`]: 256 reads in flight.
pub const DEFAULT_GATE_MAX_REQUESTS: usize = 256;

/// The limits of one [`GatedStore`].
///
/// `Default` is a query gate; [`maintenance`](Self::maintenance) is the
/// smaller gate for compaction, index builds and GC, so they never take the
/// permits queries need. The numbers are starting points, not measurements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct GateOptions {
    /// Bytes that may be in flight at once, 1 to `u32::MAX`. A read larger
    /// than this is admitted alone.
    pub max_bytes: u64,
    /// Reads that may be in flight at once, at least 1.
    pub max_requests: usize,
}

impl Default for GateOptions {
    fn default() -> Self {
        Self {
            max_bytes: DEFAULT_GATE_MAX_BYTES,
            max_requests: DEFAULT_GATE_MAX_REQUESTS,
        }
    }
}

impl GateOptions {
    /// A maintenance gate: a quarter of the default bytes, the same requests.
    pub fn maintenance() -> Self {
        Self {
            max_bytes: DEFAULT_GATE_MAX_BYTES / 4,
            max_requests: DEFAULT_GATE_MAX_REQUESTS,
        }
    }

    /// Set [`max_bytes`](Self::max_bytes).
    pub fn max_bytes(mut self, max_bytes: u64) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    /// Set [`max_requests`](Self::max_requests).
    pub fn max_requests(mut self, max_requests: usize) -> Self {
        self.max_requests = max_requests;
        self
    }
}

/// An [`ObjectStore`] wrapper that admits reads only while the bytes and the
/// requests in flight stay under its [`GateOptions`].
///
/// Every `get_opts` (and so `get`, `get_range`, `get_ranges` and `head`)
/// takes a permit before the inner call and keeps it until the whole body has
/// been read, then returns the bytes as a one-item stream. A read of known
/// length is charged its length up front; a whole object or open range is
/// charged the length its response carries, once the headers arrive. Waiters are served
/// in order. Puts, deletes, copies and listings are not gated. A permit is
/// never held while waiting on another, so nested fan-outs cannot deadlock,
/// and a dropped or failed read gives its permit back.
///
/// Put it under any [`CacheStore`](crate::CacheStore), so a cache hit takes no
/// permit, and give maintenance its own, smaller gate:
///
/// ```
/// use std::sync::Arc;
/// use itzma_core::{CacheOptions, CacheStore, GateOptions, GatedStore, Metric, Namespace, Schema};
/// use object_store::{ObjectStore, memory::InMemory};
///
/// # #[tokio::main]
/// # async fn main() -> itzma_core::Result<()> {
/// let s3: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
/// let queries = Arc::new(GatedStore::new(s3.clone(), GateOptions::default())?);
/// let maintenance = Arc::new(GatedStore::new(s3, GateOptions::maintenance())?);
/// let cache = CacheStore::new(queries, CacheOptions::default())?;
/// let ns = Namespace::create(maintenance, "docs", Schema::new(4, Metric::L2))
///     .await?
///     .with_cache(Arc::new(cache));
/// assert!(ns.query(&[0.0; 4], 10).await?.is_empty());
/// # Ok(())
/// # }
/// ```
pub struct GatedStore {
    inner: Arc<dyn ObjectStore>,
    requests: Semaphore,
    bytes: Semaphore,
    max_bytes: u32,
}

impl GatedStore {
    /// Wrap `inner` in a gate with `options`.
    ///
    /// # Errors
    ///
    /// [`ItzmaError::InvalidConfig`] when `max_requests` is 0 or above
    /// `tokio::sync::Semaphore::MAX_PERMITS`, or `max_bytes` is 0 or above
    /// `u32::MAX`.
    pub fn new(inner: Arc<dyn ObjectStore>, options: GateOptions) -> crate::Result<Self> {
        if options.max_requests == 0 || options.max_requests > Semaphore::MAX_PERMITS {
            return Err(ItzmaError::InvalidConfig(
                "a gate's max_requests must be at least 1 and fit a semaphore",
            ));
        }
        let max_bytes = u32::try_from(options.max_bytes)
            .ok()
            .filter(|&b| b > 0)
            .ok_or(ItzmaError::InvalidConfig(
                "a gate's max_bytes must be between 1 and 4 GiB",
            ))?;
        Ok(Self {
            inner,
            requests: Semaphore::new(options.max_requests),
            bytes: Semaphore::new(max_bytes as usize),
            max_bytes,
        })
    }

    /// The bytes a read is charged before it is sent: its range's length,
    /// nothing for a `head`, or `None` when the length is unknown until the
    /// headers arrive (a whole object, an open range), which is then charged
    /// by [`clamp`](Self::clamp) of the length the response carries.
    fn weight(&self, options: &GetOptions) -> Option<u32> {
        if options.head {
            return Some(0);
        }
        match &options.range {
            Some(GetRange::Bounded(r)) => Some(self.clamp(r.end.saturating_sub(r.start))),
            Some(GetRange::Suffix(n)) => Some(self.clamp(*n)),
            Some(GetRange::Offset(_)) | None => None,
        }
    }

    /// Never more than the gate holds, so the largest read is admitted alone,
    /// not never.
    fn clamp(&self, bytes: u64) -> u32 {
        u32::try_from(bytes.min(u64::from(self.max_bytes))).unwrap_or(self.max_bytes)
    }
}

/// The gate's semaphores are never closed; this only names the impossible.
fn closed(_: tokio::sync::AcquireError) -> object_store::Error {
    object_store::Error::Generic {
        store: "GatedStore",
        source: "the gate's semaphore was closed".into(),
    }
}

impl fmt::Debug for GatedStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GatedStore")
            .field("inner", &self.inner)
            .field("max_bytes", &self.max_bytes)
            .finish_non_exhaustive()
    }
}

impl fmt::Display for GatedStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GatedStore({})", self.inner)
    }
}

#[async_trait]
impl ObjectStore for GatedStore {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> Result<PutResult> {
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> Result<GetResult> {
        let weight = self.weight(&options);
        let head = options.head;
        let _request = self.requests.acquire().await.map_err(closed)?;
        // A known length is charged before the request goes out. An unknown
        // one is charged once, by the length the headers carry, before the
        // body is read: charging a guess up front made every small manifest
        // and WAL GET weigh megabytes. Each read takes its bytes in one
        // acquire, never in two parts, so two reads cannot each hold half.
        let mut bytes_permit = match weight {
            Some(weight) => Some(self.bytes.acquire_many(weight).await.map_err(closed)?),
            None => None,
        };
        let result = self.inner.get_opts(location, options).await?;
        if head {
            return Ok(result);
        }
        if bytes_permit.is_none() {
            let len = self.clamp(result.range.end.saturating_sub(result.range.start));
            bytes_permit = Some(self.bytes.acquire_many(len).await.map_err(closed)?);
        }
        let _bytes = bytes_permit;
        let meta = result.meta.clone();
        let range = result.range.clone();
        let attributes = result.attributes.clone();
        let bytes = result.bytes().await?;
        Ok(GetResult {
            payload: GetResultPayload::Stream(
                futures::stream::once(async move { Ok(bytes) }).boxed(),
            ),
            meta,
            range,
            attributes,
            extensions: Default::default(),
        })
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, Result<Path>>,
    ) -> BoxStream<'static, Result<Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, Result<ObjectMeta>> {
        self.inner.list(prefix)
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, Result<ObjectMeta>> {
        self.inner.list_with_offset(prefix, offset)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> Result<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(&self, from: &Path, to: &Path, options: CopyOptions) -> Result<()> {
        self.inner.copy_opts(from, to, options).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

    use futures::FutureExt;
    use object_store::ObjectStoreExt;
    use object_store::memory::InMemory;
    use tokio::sync::Notify;

    /// An InMemory store that records how many reads, and how many bytes,
    /// are inside `get_opts` at once, and can park reads until released.
    #[derive(Debug, Default)]
    struct Recording {
        inner: InMemory,
        in_flight: AtomicUsize,
        max_in_flight: AtomicUsize,
        bytes_in_flight: AtomicU64,
        max_bytes_in_flight: AtomicU64,
        park: AtomicBool,
        parked: Notify,
        entered: Notify,
    }

    impl fmt::Display for Recording {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "Recording")
        }
    }

    #[async_trait]
    impl ObjectStore for Recording {
        async fn put_opts(&self, l: &Path, p: PutPayload, o: PutOptions) -> Result<PutResult> {
            self.inner.put_opts(l, p, o).await
        }
        async fn put_multipart_opts(
            &self,
            l: &Path,
            o: PutMultipartOptions,
        ) -> Result<Box<dyn MultipartUpload>> {
            self.inner.put_multipart_opts(l, o).await
        }
        async fn get_opts(&self, l: &Path, o: GetOptions) -> Result<GetResult> {
            let len = match &o.range {
                Some(GetRange::Bounded(r)) => r.end - r.start,
                _ => 0,
            };
            let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_in_flight.fetch_max(now, Ordering::SeqCst);
            let bytes = self.bytes_in_flight.fetch_add(len, Ordering::SeqCst) + len;
            self.max_bytes_in_flight.fetch_max(bytes, Ordering::SeqCst);
            // Decided before `entered` fires: a test that clears `park` once
            // it hears a read entered must not unpark that same read.
            let parking = self.park.load(Ordering::SeqCst);
            self.entered.notify_one();
            if parking {
                self.parked.notified().await;
            }
            for _ in 0..4 {
                tokio::task::yield_now().await;
            }
            let got = self.inner.get_opts(l, o).await;
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            self.bytes_in_flight.fetch_sub(len, Ordering::SeqCst);
            got
        }
        fn delete_stream(
            &self,
            l: BoxStream<'static, Result<Path>>,
        ) -> BoxStream<'static, Result<Path>> {
            self.inner.delete_stream(l)
        }
        fn list(&self, p: Option<&Path>) -> BoxStream<'static, Result<ObjectMeta>> {
            self.inner.list(p)
        }
        async fn list_with_delimiter(&self, p: Option<&Path>) -> Result<ListResult> {
            self.inner.list_with_delimiter(p).await
        }
        async fn copy_opts(&self, f: &Path, t: &Path, o: CopyOptions) -> Result<()> {
            self.inner.copy_opts(f, t, o).await
        }
    }

    fn obj() -> Path {
        Path::from("obj")
    }

    async fn setup(size: usize, options: GateOptions) -> (Arc<Recording>, Arc<GatedStore>) {
        let inner = Arc::new(Recording::default());
        inner
            .put(&obj(), vec![7u8; size].into())
            .await
            .expect("put");
        let gate = GatedStore::new(inner.clone(), options).expect("gate");
        (inner, Arc::new(gate))
    }

    async fn read_all(gate: &Arc<GatedStore>, ranges: Vec<std::ops::Range<u64>>) {
        let reads = ranges.into_iter().map(|r| {
            let gate = gate.clone();
            tokio::spawn(async move { gate.get_range(&obj(), r).await.expect("read") })
        });
        for read in futures::future::join_all(reads).await {
            read.expect("join");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn never_more_requests_in_flight_than_the_gate_allows() {
        let (inner, gate) = setup(1000, GateOptions::default().max_requests(3)).await;
        read_all(&gate, (0..50).map(|i| i * 10..i * 10 + 10).collect()).await;
        let max = inner.max_in_flight.load(Ordering::SeqCst);
        assert!(max <= 3, "max in flight {max}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn never_more_bytes_in_flight_than_the_gate_allows() {
        let (inner, gate) = setup(4000, GateOptions::default().max_bytes(100)).await;
        read_all(&gate, (0..50).map(|i| i * 40..i * 40 + 40).collect()).await;
        let max = inner.max_bytes_in_flight.load(Ordering::SeqCst);
        assert!(max <= 100, "max bytes in flight {max}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_read_bigger_than_the_gate_is_admitted_alone() {
        let (inner, gate) = setup(5000, GateOptions::default().max_bytes(100)).await;
        let mut ranges: Vec<_> = (0..20).map(|i| i * 10..i * 10 + 10).collect();
        ranges.insert(10, 0..1000);
        read_all(&gate, ranges).await;
        assert_eq!(inner.max_bytes_in_flight.load(Ordering::SeqCst), 1000);
        assert_eq!(gate.bytes.available_permits(), 100);
    }

    #[tokio::test]
    async fn the_body_is_read_before_the_permit_is_released() {
        let (_inner, gate) = setup(100, GateOptions::default().max_requests(1)).await;
        let got = gate
            .get_opts(&obj(), GetOptions::default())
            .await
            .expect("get");
        assert_eq!(
            gate.requests.available_permits(),
            1,
            "permit held past the call"
        );
        assert_eq!(got.bytes().await.expect("body").len(), 100);
    }

    #[tokio::test]
    async fn a_failed_read_releases_its_permits() {
        let (_inner, gate) = setup(10, GateOptions::default().max_requests(1).max_bytes(10)).await;
        let missing = gate.get(&Path::from("missing")).await;
        assert!(matches!(missing, Err(object_store::Error::NotFound { .. })));
        assert_eq!(gate.requests.available_permits(), 1);
        assert_eq!(gate.bytes.available_permits(), 10);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_dropped_read_releases_its_permits() {
        let (inner, gate) = setup(10, GateOptions::default().max_requests(1).max_bytes(10)).await;
        inner.park.store(true, Ordering::SeqCst);
        let held = {
            let gate = gate.clone();
            tokio::spawn(async move { gate.get_range(&obj(), 0..10).await })
        };
        inner.entered.notified().await;
        assert_eq!(gate.requests.available_permits(), 0);
        held.abort();
        let _ = held.await;
        assert_eq!(gate.requests.available_permits(), 1);
        assert_eq!(gate.bytes.available_permits(), 10);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_small_read_queued_behind_a_full_gate_runs_once_it_frees() {
        let (inner, gate) = setup(100, GateOptions::default().max_bytes(10)).await;
        inner.park.store(true, Ordering::SeqCst);
        let big = {
            let gate = gate.clone();
            tokio::spawn(async move { gate.get_range(&obj(), 0..100).await })
        };
        inner.entered.notified().await;
        inner.park.store(false, Ordering::SeqCst);
        let small = {
            let gate = gate.clone();
            tokio::spawn(async move { gate.get_range(&obj(), 0..1).await })
        };
        // The big read is the only one inside the store; however long the
        // small one is given, it must not get in while the gate is full.
        for _ in 0..1_000 {
            tokio::task::yield_now().await;
        }
        assert_eq!(
            inner.in_flight.load(Ordering::SeqCst),
            1,
            "a full gate let a read past"
        );
        assert!(!small.is_finished());
        inner.parked.notify_one();
        assert_eq!(big.await.expect("join").expect("big").len(), 100);
        assert_eq!(small.await.expect("join").expect("small").len(), 1);
    }

    #[tokio::test]
    async fn a_whole_object_bigger_than_the_gate_is_read_whole() {
        let (_inner, gate) = setup(64 * 1024, GateOptions::default().max_bytes(1024)).await;
        let got = gate
            .get(&obj())
            .await
            .expect("get")
            .bytes()
            .await
            .expect("body");
        assert_eq!(got.len(), 64 * 1024);
    }

    #[tokio::test]
    async fn a_local_file_body_is_read_inside_the_permit() {
        let dir = tempfile::tempdir().expect("tempdir");
        let local = object_store::local::LocalFileSystem::new_with_prefix(dir.path()).expect("fs");
        local.put(&obj(), vec![1u8; 300].into()).await.expect("put");
        let gate =
            GatedStore::new(Arc::new(local), GateOptions::default().max_requests(1)).expect("gate");
        let got = gate
            .get_opts(&obj(), GetOptions::default())
            .await
            .expect("get");
        assert_eq!(gate.requests.available_permits(), 1);
        assert_eq!(got.bytes().await.expect("body").len(), 300);
        assert_eq!(gate.head(&obj()).await.expect("head").size, 300);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn writes_deletes_copies_and_listings_bypass_the_gate() {
        let (inner, gate) = setup(10, GateOptions::default().max_requests(1)).await;
        inner.park.store(true, Ordering::SeqCst);
        let held = {
            let gate = gate.clone();
            tokio::spawn(async move { gate.get_range(&obj(), 0..1).await })
        };
        inner.entered.notified().await;
        let other = Path::from("other");
        assert!(gate.put(&other, vec![1u8].into()).now_or_never().is_some());
        assert!(
            gate.copy(&other, &Path::from("copy"))
                .now_or_never()
                .is_some()
        );
        assert!(gate.delete(&other).now_or_never().is_some());
        assert!(gate.list_with_delimiter(None).now_or_never().is_some());
        let listed: Vec<_> = gate.list(None).collect().now_or_never().expect("list");
        assert_eq!(listed.len(), 2);
        inner.parked.notify_one();
        held.await.expect("join").expect("read");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn small_reads_of_unknown_length_run_together() {
        // A manifest or WAL GET has no range; weighing it 8 MiB up front let
        // only max_bytes / 8 MiB of them run at once (final review).
        let (inner, gate) = setup(100, GateOptions::default().max_bytes(1000)).await;
        inner.park.store(true, Ordering::SeqCst);
        let reads: Vec<_> = (0..5)
            .map(|_| {
                let gate = gate.clone();
                tokio::spawn(async move { gate.get(&obj()).await?.bytes().await })
            })
            .collect();
        // A deadline, not a count of yields: on a loaded runner the spawned
        // reads can take longer to start than any fixed number of yields.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while inner.in_flight.load(Ordering::SeqCst) < 5 {
            assert!(
                std::time::Instant::now() < deadline,
                "the reads never ran together"
            );
            tokio::task::yield_now().await;
        }
        assert_eq!(inner.in_flight.load(Ordering::SeqCst), 5);
        // A read counts itself in flight before it waits on `parked`, and
        // `notify_waiters` wakes only those already waiting, so keep waking
        // until every read is done.
        while !reads.iter().all(|r| r.is_finished()) {
            assert!(
                std::time::Instant::now() < deadline,
                "a parked read never woke"
            );
            inner.parked.notify_waiters();
            tokio::task::yield_now().await;
        }
        for read in reads {
            assert_eq!(read.await.expect("join").expect("read").len(), 100);
        }
        assert_eq!(gate.bytes.available_permits(), 1000);
    }

    #[test]
    fn weights_follow_the_range() {
        let gate = GatedStore::new(
            Arc::new(InMemory::new()),
            GateOptions::default().max_bytes(16 << 20),
        )
        .expect("gate");
        let with = |range: Option<GetRange>| GetOptions::new().with_range(range);
        assert_eq!(
            gate.weight(&with(Some(GetRange::Bounded(10..30)))),
            Some(20)
        );
        assert_eq!(gate.weight(&with(Some(GetRange::Suffix(64)))), Some(64));
        assert_eq!(gate.weight(&with(Some(GetRange::Offset(5)))), None);
        assert_eq!(gate.weight(&with(None)), None);
        assert_eq!(gate.weight(&GetOptions::new().with_head(true)), Some(0));
        let small = GatedStore::new(
            Arc::new(InMemory::new()),
            GateOptions::default().max_bytes(100),
        )
        .expect("gate");
        assert_eq!(small.clamp(8 << 20), 100, "clamped to the gate");
        assert_eq!(
            small.weight(&with(Some(GetRange::Bounded(0..1000)))),
            Some(100)
        );
    }

    #[test]
    fn options_out_of_range_are_refused() {
        let store = || -> Arc<dyn ObjectStore> { Arc::new(InMemory::new()) };
        for options in [
            GateOptions::default().max_requests(0),
            GateOptions::default().max_bytes(0),
            GateOptions::default().max_bytes(u64::from(u32::MAX) + 1),
            GateOptions::default().max_requests(Semaphore::MAX_PERMITS + 1),
        ] {
            assert!(matches!(
                GatedStore::new(store(), options),
                Err(ItzmaError::InvalidConfig(_))
            ));
        }
        assert_eq!(
            GateOptions::maintenance().max_bytes,
            DEFAULT_GATE_MAX_BYTES / 4
        );
    }
}
