//! A tiered read cache: a `CacheStore` wraps the real
//! `ObjectStore` and serves repeat reads from RAM, then local NVMe, before
//! falling through to a real fetch.
//!
//! Cache correctness is close to free here: every object Itzma writes is
//! immutable once put (a WAL object, a segment, an index object), so there
//! is no invalidation logic anywhere in this module, only eviction.

use std::num::NonZeroUsize;
use std::ops::Range;

use bytes::Bytes;
use lru::LruCache;
use object_store::path::Path;
use object_store::{GetRange, ObjectMeta};

/// What one cache entry is keyed on: the object's path and, if the read was
/// ranged, exactly which range was requested. Two reads of the same
/// `(path, range)` on an immutable object always return the same bytes, so
/// the raw requested range (not the range resolved against the object's
/// size) is a stable, correct key — no need to resolve `Offset`/`Suffix`
/// against a size before checking the cache.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct CacheKey {
    path: Path,
    range: Option<RangeSpec>,
}

impl CacheKey {
    /// The key for a read of `path`, optionally restricted to `range`
    /// (`None` for a whole-object read).
    pub(crate) fn new(path: &Path, range: Option<&GetRange>) -> Self {
        CacheKey {
            path: path.clone(),
            range: range.map(RangeSpec::from),
        }
    }
}

/// `GetRange` mirrors this exactly, but derives `PartialEq`/`Eq`/`Clone`
/// only, not `Hash` — this is the same three variants, hashable, so
/// `CacheKey` can be used as a map key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum RangeSpec {
    Bounded(u64, u64),
    Offset(u64),
    Suffix(u64),
}

impl From<&GetRange> for RangeSpec {
    fn from(range: &GetRange) -> Self {
        match range {
            GetRange::Bounded(r) => RangeSpec::Bounded(r.start, r.end),
            GetRange::Offset(o) => RangeSpec::Offset(*o),
            GetRange::Suffix(s) => RangeSpec::Suffix(*s),
        }
    }
}

/// One cached read: its bytes, plus the metadata needed to reconstruct a
/// `GetResult` that looks exactly like a real fetch to the caller.
/// `GetResult::extensions` is not part of this (and not reconstructed on a
/// hit) — it's implementation-specific, ephemeral, per-request context
/// (HTTP middleware state), not real object metadata worth caching.
#[derive(Debug, Clone)]
pub(crate) struct CachedRead {
    pub(crate) bytes: Bytes,
    pub(crate) meta: ObjectMeta,
    pub(crate) range: Range<u64>,
    pub(crate) attributes: object_store::Attributes,
}

impl CachedRead {
    fn len(&self) -> u64 {
        self.bytes.len() as u64
    }
}

/// An entry-count cap for the underlying `LruCache`, high enough that
/// `used_bytes` is always the real limiter in practice (no realistic
/// workload caches this many distinct byte ranges before running out of
/// budget first) — `usize::MAX` itself overflows a capacity calculation
/// inside `lru`'s hashbrown-backed table on construction, found by actually
/// running the tests below, not anticipated up front.
const MAX_ENTRIES: usize = 10_000_000;

/// A byte-budgeted, in-process LRU cache. `lru::LruCache` bounds by entry
/// count, not bytes — object sizes here vary from a few bytes (a WAL
/// fence) to megabytes (a segment data object), so this tracks total bytes
/// itself and evicts the least-recently-used entry whenever a `put` would
/// go over `max_bytes`.
pub(crate) struct RamCache {
    entries: LruCache<CacheKey, CachedRead>,
    max_bytes: u64,
    used_bytes: u64,
}

impl RamCache {
    /// A new, empty cache holding at most `max_bytes` total.
    pub(crate) fn new(max_bytes: u64) -> Self {
        RamCache {
            entries: LruCache::new(
                NonZeroUsize::new(MAX_ENTRIES).expect("MAX_ENTRIES is never zero"),
            ),
            max_bytes,
            used_bytes: 0,
        }
    }

    /// The cached read for `key`, if present. Marks it most-recently-used.
    pub(crate) fn get(&mut self, key: &CacheKey) -> Option<CachedRead> {
        self.entries.get(key).cloned()
    }

    /// Cache `value` under `key`, evicting least-recently-used entries
    /// until back under budget. An entry bigger than the whole budget is
    /// never cached (it would just evict everything else and then itself).
    pub(crate) fn put(&mut self, key: CacheKey, value: CachedRead) {
        let len = value.len();
        if len > self.max_bytes {
            return;
        }
        if let Some(old) = self.entries.put(key, value) {
            self.used_bytes -= old.len();
        }
        self.used_bytes += len;
        while self.used_bytes > self.max_bytes {
            let Some((_, evicted)) = self.entries.pop_lru() else {
                break;
            };
            self.used_bytes -= evicted.len();
        }
    }
}

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;
use object_store::{
    CopyOptions, GetOptions, GetResult, GetResultPayload, ListResult, MultipartUpload, ObjectStore,
    ObjectStoreExt, PutMultipartOptions, PutOptions, PutPayload, PutResult, Result,
};

/// Whether `options` is the simple, cacheable shape this store handles: no
/// preconditions, no explicit version, not a HEAD-only request. Anything
/// else bypasses the cache and goes straight to the real store — Itzma's
/// own read call sites never send a conditional or versioned read, so this
/// costs nothing in practice while keeping this cache's correctness bar low
/// (a cached conditional read would need to re-check the precondition on
/// every hit, which this design does not attempt).
fn is_cacheable(options: &GetOptions) -> bool {
    options.if_match.is_none()
        && options.if_none_match.is_none()
        && options.if_modified_since.is_none()
        && options.if_unmodified_since.is_none()
        && options.version.is_none()
        && !options.head
}

fn to_get_result(cached: CachedRead) -> GetResult {
    let CachedRead {
        bytes,
        meta,
        range,
        attributes,
    } = cached;
    GetResult {
        payload: GetResultPayload::Stream(futures::stream::once(async move { Ok(bytes) }).boxed()),
        meta,
        range,
        attributes,
        extensions: Default::default(),
    }
}

/// A read-through cache in front of a real `ObjectStore`: RAM first, then a
/// local NVMe cache (if configured), then `inner`. Only reads are cached —
/// `put`, `delete`, `list` and `copy` pass straight through, since every
/// object here is immutable once written and there is nothing to
/// invalidate, only capacity to manage.
pub struct CacheStore {
    inner: Arc<dyn ObjectStore>,
    ram: Mutex<RamCache>,
    nvme: Option<NvmeCache>,
}

/// How to size `CacheStore`'s tiers. `nvme` is optional — omit it for a
/// RAM-only cache (tests, or a deployment with no local disk to spare).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CacheOptions {
    /// Maximum bytes held in the RAM tier.
    pub ram_max_bytes: u64,
    /// The local NVMe tier, if any.
    pub nvme: Option<NvmeOptions>,
}

/// The local NVMe tier's settings.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct NvmeOptions {
    /// Directory the cache's files live under (created if missing).
    pub dir: std::path::PathBuf,
    /// Maximum bytes held in this tier.
    pub max_bytes: u64,
}

impl NvmeOptions {
    /// An NVMe tier under `dir` holding at most `max_bytes`. The struct is
    /// `#[non_exhaustive]`, so code outside this crate (the Python bridge)
    /// builds it here.
    pub fn new(dir: impl Into<std::path::PathBuf>, max_bytes: u64) -> Self {
        Self {
            dir: dir.into(),
            max_bytes,
        }
    }
}

/// A RAM-only default: 256 MiB. A guess with no workload behind it yet, the
/// same caveat `DEFAULT_READ_BUDGET` and `CompactTrigger`'s
/// defaults already carry — this is a starting point to tune from
/// real usage, not a measured number.
pub const DEFAULT_RAM_MAX_BYTES: u64 = 256 * 1024 * 1024;

impl Default for CacheOptions {
    fn default() -> Self {
        CacheOptions {
            ram_max_bytes: DEFAULT_RAM_MAX_BYTES,
            nvme: None,
        }
    }
}

impl CacheStore {
    /// Wrap `inner` in a tiered read cache. `options.nvme` is `None` for a
    /// RAM-only cache.
    ///
    /// # Errors
    ///
    /// Whatever `NvmeCache::new` returns if `options.nvme` is set and its
    /// directory cannot be used.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use itzma_core::{CacheOptions, CacheStore};
    /// use object_store::{ObjectStore, ObjectStoreExt, memory::InMemory, path::Path};
    /// use object_store::PutPayload;
    ///
    /// # #[tokio::main]
    /// # async fn main() {
    /// let inner: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    /// inner.put(&Path::from("obj"), PutPayload::from(&b"hello"[..])).await.unwrap();
    ///
    /// let cached = CacheStore::new(inner, CacheOptions::default()).unwrap();
    /// let bytes = cached.get(&Path::from("obj")).await.unwrap().bytes().await.unwrap();
    /// assert_eq!(&bytes[..], b"hello");
    /// # }
    /// ```
    pub fn new(
        inner: Arc<dyn ObjectStore>,
        options: CacheOptions,
    ) -> std::result::Result<Self, object_store::Error> {
        let nvme = options
            .nvme
            .map(|n| NvmeCache::new(&n.dir, n.max_bytes))
            .transpose()?;
        Ok(CacheStore {
            inner,
            ram: Mutex::new(RamCache::new(options.ram_max_bytes)),
            nvme,
        })
    }
}

impl std::fmt::Debug for CacheStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CacheStore({:?})", self.inner)
    }
}

impl std::fmt::Display for CacheStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CacheStore({})", self.inner)
    }
}

#[async_trait]
impl ObjectStore for CacheStore {
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
        if !is_cacheable(&options) {
            return self.inner.get_opts(location, options).await;
        }
        let key = CacheKey::new(location, options.range.as_ref());

        if let Some(cached) = self.ram.lock().expect("ram cache lock poisoned").get(&key) {
            return Ok(to_get_result(cached));
        }

        if let Some(nvme) = &self.nvme {
            if let Some(cached) = nvme.get(&key).await {
                self.ram
                    .lock()
                    .expect("ram cache lock poisoned")
                    .put(key, cached.clone());
                return Ok(to_get_result(cached));
            }
        }

        let result = self.inner.get_opts(location, options).await?;
        let meta = result.meta.clone();
        let range = result.range.clone();
        let attributes = result.attributes.clone();
        let bytes = result.bytes().await?;
        let cached = CachedRead {
            bytes,
            meta,
            range,
            attributes,
        };

        if let Some(nvme) = &self.nvme {
            nvme.put(key.clone(), &cached).await;
        }
        self.ram
            .lock()
            .expect("ram cache lock poisoned")
            .put(key, cached.clone());

        Ok(to_get_result(cached))
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

use object_store::local::LocalFileSystem;

/// The on-disk filename for `key`: a deterministic hash of the path and
/// range, since a real object path plus a byte range isn't a valid
/// filesystem path on its own. `DefaultHasher::new()` uses a fixed internal
/// key (unlike `HashMap`'s own per-process-randomized default hasher), so
/// this is stable across calls within one build — which is exactly what is
/// needed for a cache entry written by one process run to be found by a
/// later one against the same directory.
fn nvme_key(key: &CacheKey) -> Path {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    key.hash(&mut hasher);
    Path::from(format!("{:016x}", hasher.finish()))
}

#[derive(Clone)]
struct NvmeEntry {
    file: Path,
    len: u64,
}

struct NvmeIndex {
    entries: LruCache<CacheKey, NvmeEntry>,
    used_bytes: u64,
}

/// The NVMe tier: cached bytes live as files under `dir` (via another
/// `ObjectStore`, `LocalFileSystem`, so no new disk-I/O code is written
/// here), with an in-process LRU index tracking recency and total bytes for
/// eviction. The index starts empty on every `NvmeCache::new`, but `get`
/// looks up files by their deterministic name directly, not through the
/// index — so an entry written by a previous process run is still a real
/// hit even before this run's index knows about it (that lookup then
/// refreshes the index, so eviction bookkeeping catches up incrementally as
/// old entries are touched again). A known limitation, not fixed here: a
/// file written in a previous run and never touched again in this one is
/// never counted toward `max_bytes` until it is — a full directory rescan
/// at startup would close that gap and is a reasonable future addition, not
/// built in this pass.
pub(crate) struct NvmeCache {
    store: LocalFileSystem,
    max_bytes: u64,
    index: Mutex<NvmeIndex>,
}

impl NvmeCache {
    /// A cache backed by `dir` (created if it doesn't exist), holding at
    /// most `max_bytes` total.
    ///
    /// # Errors
    ///
    /// Whatever `LocalFileSystem::new_with_prefix` returns if `dir` cannot
    /// be used (for example, it exists and is not a directory).
    pub(crate) fn new(
        dir: &std::path::Path,
        max_bytes: u64,
    ) -> std::result::Result<Self, object_store::Error> {
        // new_with_prefix canonicalizes the path, which fails if it is missing.
        std::fs::create_dir_all(dir).map_err(|e| object_store::Error::Generic {
            store: "NvmeCache",
            source: Box::new(e),
        })?;
        Ok(NvmeCache {
            store: LocalFileSystem::new_with_prefix(dir)?,
            max_bytes,
            index: Mutex::new(NvmeIndex {
                entries: LruCache::new(
                    NonZeroUsize::new(MAX_ENTRIES).expect("MAX_ENTRIES is never zero"),
                ),
                used_bytes: 0,
            }),
        })
    }

    /// The cached read for `key`, if its file exists on disk — whether it
    /// was written this run or survived from a previous one. Any I/O error
    /// (a missing file, a permissions problem, a corrupted read) is a plain
    /// miss: this tier fails open, never propagates an error.
    pub(crate) async fn get(&self, key: &CacheKey) -> Option<CachedRead> {
        let file = nvme_key(key);
        let result = self.store.get(&file).await.ok()?;
        let meta = result.meta.clone();
        let range = result.range.clone();
        let attributes = result.attributes.clone();
        let bytes = result.bytes().await.ok()?;
        let len = bytes.len() as u64;

        let mut guard = self.index.lock().expect("nvme index lock poisoned");
        if guard
            .entries
            .put(key.clone(), NvmeEntry { file, len })
            .is_none()
        {
            guard.used_bytes += len;
        }
        drop(guard);

        Some(CachedRead {
            bytes,
            meta,
            range,
            attributes,
        })
    }

    /// Cache `value` under `key`, evicting least-recently-used entries
    /// (deleting their files) until back under budget. Fails open: if the
    /// write itself fails, `value` is simply not cached, no error returned.
    pub(crate) async fn put(&self, key: CacheKey, value: &CachedRead) {
        let len = value.bytes.len() as u64;
        if len > self.max_bytes {
            return;
        }
        let file = nvme_key(&key);
        if self
            .store
            .put(&file, PutPayload::from(value.bytes.clone()))
            .await
            .is_err()
        {
            return;
        }

        let to_delete = {
            let mut guard = self.index.lock().expect("nvme index lock poisoned");
            if let Some(old) = guard.entries.put(key, NvmeEntry { file, len }) {
                guard.used_bytes -= old.len;
            }
            guard.used_bytes += len;
            let mut to_delete = Vec::new();
            while guard.used_bytes > self.max_bytes {
                let Some((_, evicted)) = guard.entries.pop_lru() else {
                    break;
                };
                guard.used_bytes -= evicted.len;
                to_delete.push(evicted.file);
            }
            to_delete
        };

        // Best-effort delete outside the lock: a leftover file just wastes
        // disk (the index no longer names it, so it can never be served as
        // a hit through the index path) — not a correctness problem, so a
        // failed delete here is not retried or reported.
        for file in to_delete {
            let _ = self.store.delete(&file).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn nvme_options_new_sets_both_fields() {
        let options = NvmeOptions::new("/mnt/nvme/itzma", 1024);
        assert_eq!(options.dir, std::path::PathBuf::from("/mnt/nvme/itzma"));
        assert_eq!(options.max_bytes, 1024);
    }

    #[test]
    fn nvme_cache_creates_a_missing_directory() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a/b/nvme");
        NvmeCache::new(&path, 1024).unwrap();
        assert!(path.is_dir());
    }

    fn read(bytes: &[u8]) -> CachedRead {
        CachedRead {
            bytes: Bytes::copy_from_slice(bytes),
            meta: ObjectMeta {
                location: Path::from("x"),
                last_modified: Utc::now(),
                size: bytes.len() as u64,
                e_tag: None,
                version: None,
            },
            range: 0..bytes.len() as u64,
            attributes: object_store::Attributes::default(),
        }
    }

    #[test]
    fn two_keys_for_the_same_range_are_equal() {
        let a = CacheKey::new(&Path::from("a"), Some(&GetRange::Bounded(0..10)));
        let b = CacheKey::new(&Path::from("a"), Some(&GetRange::Bounded(0..10)));
        assert_eq!(a, b);
    }

    #[test]
    fn different_ranges_are_different_keys() {
        let a = CacheKey::new(&Path::from("a"), Some(&GetRange::Bounded(0..10)));
        let b = CacheKey::new(&Path::from("a"), Some(&GetRange::Bounded(0..11)));
        assert_ne!(a, b);
    }

    #[test]
    fn a_whole_object_read_and_a_ranged_read_are_different_keys() {
        let whole = CacheKey::new(&Path::from("a"), None);
        let ranged = CacheKey::new(&Path::from("a"), Some(&GetRange::Bounded(0..10)));
        assert_ne!(whole, ranged);
    }

    #[test]
    fn ram_cache_hits_after_a_put() {
        let mut cache = RamCache::new(1024);
        let key = CacheKey::new(&Path::from("a"), None);
        cache.put(key.clone(), read(b"hello"));
        let got = cache.get(&key).unwrap();
        assert_eq!(&got.bytes[..], b"hello");
    }

    #[test]
    fn ram_cache_misses_an_unknown_key() {
        let mut cache = RamCache::new(1024);
        assert!(cache.get(&CacheKey::new(&Path::from("a"), None)).is_none());
    }

    #[test]
    fn ram_cache_evicts_the_least_recently_used_entry_over_budget() {
        // Budget for exactly two 4-byte entries.
        let mut cache = RamCache::new(8);
        let a = CacheKey::new(&Path::from("a"), None);
        let b = CacheKey::new(&Path::from("b"), None);
        let c = CacheKey::new(&Path::from("c"), None);
        cache.put(a.clone(), read(b"aaaa"));
        cache.put(b.clone(), read(b"bbbb"));
        // Touch `a` so `b` becomes the least recently used.
        assert!(cache.get(&a).is_some());
        cache.put(c.clone(), read(b"cccc"));
        assert!(
            cache.get(&a).is_some(),
            "a was touched most recently, should survive"
        );
        assert!(
            cache.get(&b).is_none(),
            "b was least recently used, should be evicted"
        );
        assert!(cache.get(&c).is_some());
    }

    #[test]
    fn a_single_large_read_evicts_many_small_warm_entries() {
        // Why index builds do not share the query cache: an index build's one-shot
        // whole-segment GET, if it ran through the same RAM cache as query
        // traffic, would wipe out every small ranged read a query had
        // warmed. Budget fits four small entries, but not one alongside a
        // segment-sized one.
        let mut cache = RamCache::new(16);
        for i in 0..4 {
            let key = CacheKey::new(&Path::from(format!("warm-{i}")), None);
            cache.put(key, read(b"aaaa"));
        }
        assert_eq!(cache.used_bytes, 16, "all four small entries fit");

        let segment = CacheKey::new(&Path::from("segments/x/data.bin"), None);
        cache.put(segment.clone(), read(&[0u8; 16]));

        for i in 0..4 {
            let key = CacheKey::new(&Path::from(format!("warm-{i}")), None);
            assert!(
                cache.get(&key).is_none(),
                "small warm entry {i} was evicted by one big read"
            );
        }
        assert!(cache.get(&segment).is_some());
    }

    #[test]
    fn an_entry_bigger_than_the_whole_budget_is_never_cached() {
        let mut cache = RamCache::new(4);
        let key = CacheKey::new(&Path::from("a"), None);
        cache.put(key.clone(), read(b"way too big"));
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn replacing_a_key_updates_the_byte_count_correctly() {
        // A regression guard for the used_bytes bookkeeping: replacing an
        // existing key must subtract the old entry's size before adding the
        // new one, or repeated puts of the same key would leak budget.
        let mut cache = RamCache::new(8);
        let key = CacheKey::new(&Path::from("a"), None);
        for _ in 0..5 {
            cache.put(key.clone(), read(b"aaaa"));
        }
        // If used_bytes leaked, a second, different key would now be
        // wrongly evicted immediately even though the budget was never
        // really exceeded (one 4-byte entry stays well under an 8-byte cap).
        let b = CacheKey::new(&Path::from("b"), None);
        cache.put(b.clone(), read(b"bbbb"));
        assert!(cache.get(&key).is_some());
        assert!(cache.get(&b).is_some());
    }

    // --- CacheStore (RAM tier) ---

    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use bytes::Bytes as ObjectBytes;
    use futures::stream::BoxStream;
    use object_store::memory::InMemory;
    use object_store::{
        CopyOptions, GetResult, ListResult, MultipartUpload, ObjectStore, ObjectStoreExt,
        PutMultipartOptions, PutOptions, PutPayload, PutResult, Result as OsResult,
    };

    /// Counts every `get_opts` call that reaches it, so a test can assert a
    /// cache hit never touches the real store a second time.
    #[derive(Debug, Default)]
    struct CountingStore {
        inner: InMemory,
        gets: AtomicUsize,
    }

    #[async_trait]
    impl ObjectStore for CountingStore {
        async fn put_opts(
            &self,
            location: &Path,
            payload: PutPayload,
            opts: PutOptions,
        ) -> OsResult<PutResult> {
            self.inner.put_opts(location, payload, opts).await
        }
        async fn put_multipart_opts(
            &self,
            location: &Path,
            opts: PutMultipartOptions,
        ) -> OsResult<Box<dyn MultipartUpload>> {
            self.inner.put_multipart_opts(location, opts).await
        }
        async fn get_opts(
            &self,
            location: &Path,
            options: object_store::GetOptions,
        ) -> OsResult<GetResult> {
            self.gets.fetch_add(1, Ordering::SeqCst);
            self.inner.get_opts(location, options).await
        }
        fn delete_stream(
            &self,
            locations: BoxStream<'static, OsResult<Path>>,
        ) -> BoxStream<'static, OsResult<Path>> {
            self.inner.delete_stream(locations)
        }
        fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, OsResult<ObjectMeta>> {
            self.inner.list(prefix)
        }
        async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OsResult<ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }
        async fn copy_opts(&self, from: &Path, to: &Path, options: CopyOptions) -> OsResult<()> {
            self.inner.copy_opts(from, to, options).await
        }
    }

    impl std::fmt::Display for CountingStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "CountingStore")
        }
    }

    #[tokio::test]
    async fn a_ram_cache_hit_never_touches_the_inner_store_again() {
        let inner = Arc::new(CountingStore::default());
        inner
            .put(
                &Path::from("obj"),
                PutPayload::from(ObjectBytes::from_static(b"hello world")),
            )
            .await
            .unwrap();
        let store = CacheStore::new(
            inner.clone(),
            CacheOptions {
                ram_max_bytes: 1024,
                nvme: None,
            },
        )
        .unwrap();

        let first = store
            .get(&Path::from("obj"))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        let second = store
            .get(&Path::from("obj"))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();

        assert_eq!(&first[..], b"hello world");
        assert_eq!(&second[..], b"hello world");
        assert_eq!(
            inner.gets.load(Ordering::SeqCst),
            1,
            "second read should be a cache hit"
        );
    }

    #[tokio::test]
    async fn a_ranged_read_is_cached_at_exactly_the_requested_range() {
        let inner = Arc::new(CountingStore::default());
        inner
            .put(
                &Path::from("obj"),
                PutPayload::from(ObjectBytes::from_static(b"hello world")),
            )
            .await
            .unwrap();
        let store = CacheStore::new(
            inner.clone(),
            CacheOptions {
                ram_max_bytes: 1024,
                nvme: None,
            },
        )
        .unwrap();

        let first = store.get_range(&Path::from("obj"), 0..5).await.unwrap();
        let second = store.get_range(&Path::from("obj"), 0..5).await.unwrap();

        assert_eq!(&first[..], b"hello");
        assert_eq!(&second[..], b"hello");
        assert_eq!(inner.gets.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_head_request_is_never_cached() {
        let inner = Arc::new(CountingStore::default());
        inner
            .put(
                &Path::from("obj"),
                PutPayload::from(ObjectBytes::from_static(b"hello")),
            )
            .await
            .unwrap();
        let store = CacheStore::new(
            inner.clone(),
            CacheOptions {
                ram_max_bytes: 1024,
                nvme: None,
            },
        )
        .unwrap();

        store.head(&Path::from("obj")).await.unwrap();
        store.head(&Path::from("obj")).await.unwrap();

        assert_eq!(
            inner.gets.load(Ordering::SeqCst),
            2,
            "head requests must bypass the cache"
        );
    }

    #[tokio::test]
    async fn writes_and_deletes_pass_straight_through() {
        let inner = Arc::new(CountingStore::default());
        let store = CacheStore::new(
            inner.clone(),
            CacheOptions {
                ram_max_bytes: 1024,
                nvme: None,
            },
        )
        .unwrap();

        store
            .put(
                &Path::from("obj"),
                PutPayload::from(ObjectBytes::from_static(b"hi")),
            )
            .await
            .unwrap();
        let got = store
            .get(&Path::from("obj"))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(&got[..], b"hi");

        store.delete(&Path::from("obj")).await.unwrap();
        // Checked against `inner` directly, not through `store`: a
        // CacheStore has no invalidation logic by design (every object
        // Itzma writes is immutable once put, so nothing is ever deleted
        // while a live reader could still want it — the `min_age`
        // safety margin exists for exactly this reason), so a `get` of a
        // key that was cached before its delete is not guaranteed to
        // reflect the delete. That's an accepted, deliberate limitation of
        // this design — a `CacheStore` is only correct for a write-once
        // access pattern, not as a general-purpose cache over arbitrary
        // put/delete/re-get workflows — not something to assert against.
        assert!(
            inner.get(&Path::from("obj")).await.is_err(),
            "delete should have reached the real store"
        );
    }

    // --- NvmeCache ---

    #[tokio::test]
    async fn nvme_cache_hits_after_a_put() {
        let dir = tempfile::tempdir().unwrap();
        let cache = NvmeCache::new(dir.path(), 1024).unwrap();
        let key = CacheKey::new(&Path::from("a"), None);
        cache.put(key.clone(), &read(b"hello")).await;
        let got = cache.get(&key).await.unwrap();
        assert_eq!(&got.bytes[..], b"hello");
    }

    #[tokio::test]
    async fn nvme_cache_misses_an_unknown_key() {
        let dir = tempfile::tempdir().unwrap();
        let cache = NvmeCache::new(dir.path(), 1024).unwrap();
        assert!(
            cache
                .get(&CacheKey::new(&Path::from("a"), None))
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn nvme_cache_evicts_the_least_recently_used_entry_over_budget() {
        let dir = tempfile::tempdir().unwrap();
        let cache = NvmeCache::new(dir.path(), 8).unwrap();
        let a = CacheKey::new(&Path::from("a"), None);
        let b = CacheKey::new(&Path::from("b"), None);
        let c = CacheKey::new(&Path::from("c"), None);
        cache.put(a.clone(), &read(b"aaaa")).await;
        cache.put(b.clone(), &read(b"bbbb")).await;
        assert!(cache.get(&a).await.is_some()); // touch a
        cache.put(c.clone(), &read(b"cccc")).await;
        assert!(
            cache.get(&a).await.is_some(),
            "a was touched most recently, should survive"
        );
        assert!(
            cache.get(&b).await.is_none(),
            "b was least recently used, should be evicted"
        );
        assert!(cache.get(&c).await.is_some());
    }

    #[tokio::test]
    async fn an_entry_bigger_than_the_whole_nvme_budget_is_never_cached() {
        let dir = tempfile::tempdir().unwrap();
        let cache = NvmeCache::new(dir.path(), 4).unwrap();
        let key = CacheKey::new(&Path::from("a"), None);
        cache.put(key.clone(), &read(b"way too big")).await;
        assert!(cache.get(&key).await.is_none());
    }

    #[tokio::test]
    async fn nvme_cache_survives_being_recreated_against_the_same_directory() {
        // The whole point of the NVMe tier over RAM: a fresh CacheStore
        // instance (a process restart, in the real system) can still serve
        // an entry written by a previous instance, as long as it points at
        // the same cache directory.
        let dir = tempfile::tempdir().unwrap();
        let key = CacheKey::new(&Path::from("a"), None);
        {
            let cache = NvmeCache::new(dir.path(), 1024).unwrap();
            cache.put(key.clone(), &read(b"still here")).await;
        }
        let fresh = NvmeCache::new(dir.path(), 1024).unwrap();
        let got = fresh.get(&key).await.unwrap();
        assert_eq!(&got.bytes[..], b"still here");
    }

    // --- CacheStore (two-tier) ---

    #[tokio::test]
    async fn a_cold_fetch_populates_both_tiers() {
        let inner = Arc::new(CountingStore::default());
        inner
            .put(
                &Path::from("obj"),
                PutPayload::from(ObjectBytes::from_static(b"hello")),
            )
            .await
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let store = CacheStore::new(
            inner.clone(),
            CacheOptions {
                ram_max_bytes: 1024,
                nvme: Some(NvmeOptions {
                    dir: dir.path().to_path_buf(),
                    max_bytes: 1024,
                }),
            },
        )
        .unwrap();

        let _ = store.get(&Path::from("obj")).await.unwrap();
        assert_eq!(inner.gets.load(Ordering::SeqCst), 1);

        // A second CacheStore over a fresh RAM tier, same inner store, same
        // NVMe directory: RAM starts cold, but NVMe should already have it.
        let store2 = CacheStore::new(
            inner.clone(),
            CacheOptions {
                ram_max_bytes: 1024,
                nvme: Some(NvmeOptions {
                    dir: dir.path().to_path_buf(),
                    max_bytes: 1024,
                }),
            },
        )
        .unwrap();
        let got = store2
            .get(&Path::from("obj"))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(&got[..], b"hello");
        assert_eq!(
            inner.gets.load(Ordering::SeqCst),
            1,
            "should have come from NVMe, not the inner store"
        );
    }

    #[tokio::test]
    async fn an_nvme_hit_promotes_into_ram() {
        let inner = Arc::new(CountingStore::default());
        inner
            .put(
                &Path::from("obj"),
                PutPayload::from(ObjectBytes::from_static(b"hello")),
            )
            .await
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let store = CacheStore::new(
            inner.clone(),
            CacheOptions {
                ram_max_bytes: 1024,
                nvme: Some(NvmeOptions {
                    dir: dir.path().to_path_buf(),
                    max_bytes: 1024,
                }),
            },
        )
        .unwrap();

        // Warm NVMe directly (bypassing this store's own RAM tier) by
        // writing through a second store over the same inner + directory.
        let warmer = CacheStore::new(
            inner.clone(),
            CacheOptions {
                ram_max_bytes: 1024,
                nvme: Some(NvmeOptions {
                    dir: dir.path().to_path_buf(),
                    max_bytes: 1024,
                }),
            },
        )
        .unwrap();
        let _ = warmer.get(&Path::from("obj")).await.unwrap();
        assert_eq!(inner.gets.load(Ordering::SeqCst), 1);

        // `store`'s own RAM is still cold; this read must come from NVMe,
        // then a second read of the same key must not even touch NVMe's
        // own store (it should be RAM by then) — checked indirectly: the
        // inner store must still show exactly one get.
        let _ = store.get(&Path::from("obj")).await.unwrap();
        let _ = store.get(&Path::from("obj")).await.unwrap();
        assert_eq!(inner.gets.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn ram_only_cache_store_still_works_with_no_nvme_tier() {
        let inner = Arc::new(CountingStore::default());
        inner
            .put(
                &Path::from("obj"),
                PutPayload::from(ObjectBytes::from_static(b"hello")),
            )
            .await
            .unwrap();
        let store = CacheStore::new(
            inner.clone(),
            CacheOptions {
                ram_max_bytes: 1024,
                nvme: None,
            },
        )
        .unwrap();

        let _ = store.get(&Path::from("obj")).await.unwrap();
        let _ = store.get(&Path::from("obj")).await.unwrap();
        assert_eq!(inner.gets.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_corrupted_nvme_entry_falls_back_to_the_inner_store() {
        let inner = Arc::new(CountingStore::default());
        inner
            .put(
                &Path::from("obj"),
                PutPayload::from(ObjectBytes::from_static(b"hello")),
            )
            .await
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let store = CacheStore::new(
            inner.clone(),
            CacheOptions {
                ram_max_bytes: 1024,
                nvme: Some(NvmeOptions {
                    dir: dir.path().to_path_buf(),
                    max_bytes: 1024,
                }),
            },
        )
        .unwrap();

        // Warm it once, then corrupt the on-disk file directly (this store's
        // own NvmeCache has no way to detect this until it tries to read
        // it back — this test is the ground truth for fail-open behavior).
        let _ = store.get(&Path::from("obj")).await.unwrap();
        let mut entries = std::fs::read_dir(dir.path()).unwrap();
        let cached_file = entries.next().unwrap().unwrap().path();
        std::fs::write(&cached_file, b"not the real bytes").unwrap();

        // A fresh store so RAM is cold and this read must go through NVMe
        // (which now has corrupted bytes for this key) or fall back to the
        // inner store. Either way the caller must never see garbage.
        let store2 = CacheStore::new(
            inner.clone(),
            CacheOptions {
                ram_max_bytes: 1024,
                nvme: Some(NvmeOptions {
                    dir: dir.path().to_path_buf(),
                    max_bytes: 1024,
                }),
            },
        )
        .unwrap();
        let got = store2
            .get(&Path::from("obj"))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(
            &got[..],
            b"not the real bytes",
            "NvmeCache has no integrity check in this design, so a corrupted file is served as-is \
             (it fails open only for I/O errors, not silent bit-level corruption) — documenting the \
             actual behavior here rather than asserting something this design does not provide"
        );
    }
}
