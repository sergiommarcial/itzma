//! The decoded-structure cache.
//!
//! [`CacheStore`](crate::CacheStore) keeps bytes, so a warm query still parses
//! every head, partition, block and directory it touches. This keeps what the
//! parse produced, keyed by the object's path and the structure, so a warm
//! query reads nothing from the store and decodes nothing. It needs no
//! invalidation: an object's bytes never change while it exists, and index
//! generations are never reused.
//!
//! Loads are single-flight. The first caller of a missing key marks it pending
//! and runs its own load. Callers that find it pending wait, then look again:
//! a value is a hit, and a load that failed or was dropped leaves nothing
//! behind, so the next caller loads it itself with its own read budget. No
//! error is ever shared between callers, so [`ItzmaError`] needs no `Clone`.
//!
//! One LRU bounded by weight in bytes, not shards: a decoded `Layer` of a whole
//! segment can be larger than any fixed share of the limit would be.

use std::any::Any;
use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Mutex, PoisonError, Weak};

use futures::channel::oneshot;
use futures::future::{FutureExt, Shared};
use lru::LruCache;
use object_store::ObjectStore;
use object_store::path::Path;

use crate::error::{ItzmaError, Result};

/// The limit a [`DecodedCache`] has when the caller does not say: 256 MiB.
pub const DEFAULT_DECODED_MAX_BYTES: u64 = 256 * 1024 * 1024;

/// How a [`DecodedCache`] is sized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct DecodedCacheOptions {
    /// The most bytes of decoded structures kept, by each structure's own
    /// estimate of its size. A structure bigger than this is used and not
    /// kept. Default [`DEFAULT_DECODED_MAX_BYTES`].
    pub max_bytes: u64,
}

impl Default for DecodedCacheOptions {
    fn default() -> Self {
        Self {
            max_bytes: DEFAULT_DECODED_MAX_BYTES,
        }
    }
}

impl DecodedCacheOptions {
    /// Set [`max_bytes`](Self::max_bytes).
    #[must_use]
    pub fn max_bytes(mut self, max_bytes: u64) -> Self {
        self.max_bytes = max_bytes;
        self
    }
}

/// Which structure of an object a cache entry holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum What {
    /// A segment's header, trailer and directory (`RangedSegment::open`).
    SegmentMeta,
    /// A segment's Bloom filter.
    Bloom,
    /// One block of a segment's ids, attributes or tombstones.
    IdBlock(usize),
    AttrBlock(usize),
    TombstoneBlock(usize),
    /// An index object's prefix and head bytes (`index::open`), any kind.
    IndexHead,
    /// An SQ8 head, parsed.
    Sq8Head,
    /// One SQ8 partition, parsed.
    Partition(u32),
    /// A whole segment for the query's flat path, as a `SegmentView`.
    View,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Key {
    store: usize,
    path: Path,
    what: What,
}

type Value = Arc<dyn Any + Send + Sync>;
type Pending = Shared<oneshot::Receiver<()>>;

struct Inner {
    ready: LruCache<Key, (Value, u64)>,
    pending: HashMap<Key, Pending>,
    used: u64,
    /// The stores whose handles have used this cache, by address, so a store
    /// dropped and another allocated at the same address cannot see its
    /// entries.
    stores: HashMap<usize, Weak<dyn ObjectStore>>,
}

/// Parsed index heads, partitions, segment directories, blocks and decoded
/// segments, shared by every handle given it with
/// [`Namespace::with_decoded_cache`](crate::Namespace::with_decoded_cache).
///
/// A hit costs no store call and is not charged to the read budget; a miss is
/// read and charged as it would be without the cache. Only the query path uses
/// it: `index`, `compact`, `gc`, `inspect` and `recover` never do. Entries are
/// keyed by store as well as path, so one cache can serve handles over several
/// stores.
pub struct DecodedCache {
    max_bytes: u64,
    inner: Mutex<Inner>,
}

impl std::fmt::Debug for DecodedCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.lock();
        f.debug_struct("DecodedCache")
            .field("max_bytes", &self.max_bytes)
            .field("used", &inner.used)
            .field("entries", &inner.ready.len())
            .finish()
    }
}

impl DecodedCache {
    /// An empty cache.
    #[must_use]
    pub fn new(options: DecodedCacheOptions) -> Arc<Self> {
        Arc::new(Self {
            max_bytes: options.max_bytes,
            inner: Mutex::new(Inner {
                ready: LruCache::unbounded(),
                pending: HashMap::new(),
                used: 0,
                stores: HashMap::new(),
            }),
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// A store's identity while it lives: the address of its data.
    pub(crate) fn store_id(store: &Arc<dyn ObjectStore>) -> usize {
        Arc::as_ptr(store).cast::<()>() as usize
    }

    /// A handle for reads from `store`.
    pub(crate) fn handle(self: &Arc<Self>, store: &Arc<dyn ObjectStore>) -> DecodedRef {
        let id = Self::store_id(store);
        let mut inner = self.lock();
        let known = inner.stores.get(&id).map(|weak| weak.strong_count() > 0);
        if known == Some(false) {
            Self::purge(&mut inner, id);
        }
        inner.stores.insert(id, Arc::downgrade(store));
        DecodedRef {
            cache: self.clone(),
            store: id,
        }
    }

    /// Drop the entries of the store at `id` if it is gone.
    #[cfg(test)]
    pub(crate) fn forget_dead(&self, id: usize) {
        let mut inner = self.lock();
        if inner.stores.get(&id).is_some_and(|w| w.strong_count() == 0) {
            Self::purge(&mut inner, id);
        }
    }

    #[cfg(test)]
    pub(crate) fn entries_of(&self, id: usize) -> usize {
        self.lock()
            .ready
            .iter()
            .filter(|(k, _)| k.store == id)
            .count()
    }

    fn purge(inner: &mut Inner, id: usize) {
        let dead: Vec<Key> = inner
            .ready
            .iter()
            .filter(|(key, _)| key.store == id)
            .map(|(key, _)| key.clone())
            .collect();
        for key in dead {
            if let Some((_, weight)) = inner.ready.pop(&key) {
                inner.used -= weight;
            }
        }
        inner.stores.remove(&id);
    }

    fn insert(&self, inner: &mut Inner, key: Key, value: Value, weight: u64) {
        if weight > self.max_bytes {
            return;
        }
        if let Some((_, old)) = inner.ready.put(key, (value, weight)) {
            inner.used -= old;
        }
        inner.used += weight;
        while inner.used > self.max_bytes {
            match inner.ready.pop_lru() {
                Some((_, (_, w))) => inner.used -= w,
                None => break,
            }
        }
    }
}

/// Removes the keys a caller claimed but did not fill, and wakes their
/// waiters, however the load ended: an error, success, or a dropped future.
struct Claim {
    cache: Arc<DecodedCache>,
    keys: Vec<Key>,
    _wake: Vec<oneshot::Sender<()>>,
}

impl Drop for Claim {
    fn drop(&mut self) {
        let mut inner = self.cache.lock();
        for key in &self.keys {
            inner.pending.remove(key);
        }
    }
}

/// A [`DecodedCache`] bound to one store, as a query holds it.
#[derive(Clone)]
pub(crate) struct DecodedRef {
    cache: Arc<DecodedCache>,
    store: usize,
}

impl std::fmt::Debug for DecodedRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecodedRef")
            .field("store", &self.store)
            .finish()
    }
}

impl DecodedRef {
    fn key(&self, path: &Path, what: What) -> Key {
        Key {
            store: self.store,
            path: path.clone(),
            what,
        }
    }

    /// The structure `what` of `path`, from the cache or from `load`, which
    /// gives the value and its weight in bytes and runs at most once per call.
    pub(crate) async fn get_or_load<T, F, Fut>(
        &self,
        path: &Path,
        what: What,
        load: F,
    ) -> Result<Arc<T>>
    where
        T: Send + Sync + 'static,
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<(T, u64)>>,
    {
        let mut load = Some(load);
        let mut got = self
            .get_or_load_many(path, &[what], |_| {
                let load = load.take();
                async move {
                    match load {
                        Some(load) => Ok(vec![load().await?]),
                        // Only the first claim loads: a second would mean the
                        // first one's value vanished, which `get_or_load_many`
                        // never lets happen within one call.
                        None => Err(ItzmaError::InvalidQuery("a cache load ran twice")),
                    }
                }
            })
            .await?;
        got.pop()
            .ok_or(ItzmaError::InvalidQuery("a cache load gave no value"))
    }

    /// The structures `whats` of `path`, in order. `load` is given the
    /// positions in `whats` that are neither cached nor being loaded by another
    /// caller, and returns their values and weights in that order, so it can
    /// fetch them together. It runs once per round that finds such positions.
    pub(crate) async fn get_or_load_many<T, F, Fut>(
        &self,
        path: &Path,
        whats: &[What],
        mut load: F,
    ) -> Result<Vec<Arc<T>>>
    where
        T: Send + Sync + 'static,
        F: FnMut(Vec<usize>) -> Fut,
        Fut: Future<Output = Result<Vec<(T, u64)>>>,
    {
        let keys: Vec<Key> = whats.iter().map(|w| self.key(path, *w)).collect();
        let mut out: Vec<Option<Arc<T>>> = vec![None; keys.len()];
        loop {
            let mut claimed = Vec::new();
            let mut senders = Vec::new();
            let mut waits = Vec::new();
            {
                let mut inner = self.cache.lock();
                for (at, key) in keys.iter().enumerate() {
                    if out[at].is_some() {
                        continue;
                    }
                    let hit = inner.ready.get(key).map(|(value, _)| value.clone());
                    if let Some(typed) = hit.and_then(|value| value.downcast::<T>().ok()) {
                        out[at] = Some(typed);
                    } else if let Some(pending) = inner.pending.get(key) {
                        waits.push(pending.clone());
                    } else {
                        let (tx, rx) = oneshot::channel();
                        inner.pending.insert(key.clone(), rx.shared());
                        senders.push(tx);
                        claimed.push(at);
                    }
                }
            }
            if claimed.is_empty() && waits.is_empty() {
                break;
            }
            if !claimed.is_empty() {
                let claim = Claim {
                    cache: self.cache.clone(),
                    keys: claimed.iter().map(|at| keys[*at].clone()).collect(),
                    _wake: senders,
                };
                let values = load(claimed.clone()).await?;
                if values.len() != claimed.len() {
                    return Err(ItzmaError::InvalidQuery(
                        "a cache load gave the wrong number of values",
                    ));
                }
                let mut inner = self.cache.lock();
                for (at, (value, weight)) in claimed.iter().zip(values) {
                    let value = Arc::new(value);
                    let shared: Value = value.clone();
                    self.cache
                        .insert(&mut inner, keys[*at].clone(), shared, weight);
                    out[*at] = Some(value);
                }
                drop(inner);
                drop(claim);
            }
            for wait in waits {
                let _ = wait.await;
            }
        }
        Ok(out.into_iter().flatten().collect())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use futures::channel::oneshot;
    use futures::future::FutureExt;
    use object_store::ObjectStore;
    use object_store::memory::InMemory;
    use object_store::path::Path;

    use super::*;
    use crate::error::ItzmaError;

    fn store() -> Arc<dyn ObjectStore> {
        Arc::new(InMemory::new())
    }

    fn cache(max_bytes: u64) -> Arc<DecodedCache> {
        DecodedCache::new(DecodedCacheOptions::default().max_bytes(max_bytes))
    }

    fn path(name: &str) -> Path {
        Path::from(name)
    }

    /// A load that counts its calls and gives `value` with weight `weight`.
    async fn counted(
        calls: &AtomicUsize,
        value: u64,
        weight: u64,
    ) -> crate::error::Result<(u64, u64)> {
        calls.fetch_add(1, Ordering::SeqCst);
        Ok((value, weight))
    }

    #[tokio::test]
    async fn two_concurrent_gets_of_one_key_load_once() {
        let s = store();
        let handle = cache(1 << 20).handle(&s);
        let calls = AtomicUsize::new(0);
        let (open, gate) = oneshot::channel::<()>();
        let gate = gate.shared();
        let a = path("a");

        let first = handle.get_or_load(&a, What::IndexHead, || async {
            calls.fetch_add(1, Ordering::SeqCst);
            let _ = gate.clone().await;
            Ok::<_, ItzmaError>((7u64, 8))
        });
        let second = handle.get_or_load(&a, What::IndexHead, || async {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok::<_, ItzmaError>((9u64, 8))
        });
        let release = async {
            tokio::task::yield_now().await;
            let _ = open.send(());
        };
        let (a, b, ()) = futures::join!(first, second, release);
        assert_eq!(calls.load(Ordering::SeqCst), 1, "the waiter must not load");
        assert!(Arc::ptr_eq(&a.unwrap(), &b.unwrap()));
    }

    #[tokio::test]
    async fn a_hit_does_not_load_and_moves_the_entry_to_the_front() {
        let s = store();
        let handle = cache(100).handle(&s);
        let calls = AtomicUsize::new(0);
        let get = |name: &'static str, weight: u64| {
            let handle = handle.clone();
            let calls = &calls;
            async move {
                handle
                    .get_or_load(&path(name), What::IndexHead, || counted(calls, 1, weight))
                    .await
                    .unwrap()
            }
        };
        get("a", 60).await;
        get("b", 30).await;
        get("a", 60).await;
        assert_eq!(calls.load(Ordering::SeqCst), 2, "the third get is a hit");

        // 60 + 30 + 30 is over 100: the least recently used, `b`, goes.
        get("c", 30).await;
        get("a", 60).await;
        assert_eq!(calls.load(Ordering::SeqCst), 3, "a stayed");
        get("b", 30).await;
        assert_eq!(calls.load(Ordering::SeqCst), 4, "b was evicted");
    }

    #[tokio::test]
    async fn a_value_bigger_than_the_cache_is_returned_and_not_kept() {
        let s = store();
        let handle = cache(10).handle(&s);
        let calls = AtomicUsize::new(0);
        for _ in 0..2 {
            let got = handle
                .get_or_load(&path("big"), What::View, || counted(&calls, 5, 11))
                .await
                .unwrap();
            assert_eq!(*got, 5);
        }
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn a_failed_load_is_not_kept_and_the_next_caller_loads_again() {
        let s = store();
        let handle = cache(1 << 20).handle(&s);
        let failed = handle
            .get_or_load(&path("a"), What::Bloom, || async {
                Err::<(u64, u64), _>(ItzmaError::EmptyBatch)
            })
            .await;
        assert!(matches!(failed, Err(ItzmaError::EmptyBatch)), "{failed:?}");
        let calls = AtomicUsize::new(0);
        let got = handle
            .get_or_load(&path("a"), What::Bloom, || counted(&calls, 3, 1))
            .await
            .unwrap();
        assert_eq!((*got, calls.load(Ordering::SeqCst)), (3, 1));
    }

    /// A waiter whose loader fails, or is dropped mid-load, loads itself.
    #[tokio::test]
    async fn a_waiter_loads_itself_when_the_loader_fails_or_is_dropped() {
        for drop_the_loader in [false, true] {
            let s = store();
            let handle = cache(1 << 20).handle(&s);
            let (open, gate) = oneshot::channel::<()>();
            let gate = gate.shared();
            let a = path("a");
            let mut first = handle
                .get_or_load(&a, What::IndexHead, || async {
                    let _ = gate.clone().await;
                    Err::<(u64, u64), _>(ItzmaError::EmptyBatch)
                })
                .boxed();
            // The first caller claims the key and stops at the gate.
            assert!(futures::poll!(&mut first).is_pending());
            let calls = AtomicUsize::new(0);
            let second = handle.get_or_load(&a, What::IndexHead, || counted(&calls, 4, 1));

            // `second` is polled first and waits; then the loader fails or
            // goes away.
            let driver = async move {
                if drop_the_loader {
                    drop(first);
                } else {
                    let _ = open.send(());
                    assert!(first.await.is_err());
                }
            };
            let (got, ()) = futures::join!(second, driver);
            assert_eq!(*got.unwrap(), 4, "drop: {drop_the_loader}");
            assert_eq!(calls.load(Ordering::SeqCst), 1);
        }
    }

    #[tokio::test]
    async fn many_loads_only_the_missing_keys_in_one_call() {
        let s = store();
        let handle = cache(1 << 20).handle(&s);
        let p = path("idx");
        handle
            .get_or_load(&p, What::Partition(1), || async {
                Ok::<_, ItzmaError>((10u64, 1))
            })
            .await
            .unwrap();
        let asked = std::sync::Mutex::new(Vec::new());
        let got = handle
            .get_or_load_many(
                &p,
                &[What::Partition(0), What::Partition(1), What::Partition(2)],
                |missing: Vec<usize>| {
                    asked.lock().unwrap().push(missing.clone());
                    async move { Ok(missing.iter().map(|i| (*i as u64 * 100, 1)).collect()) }
                },
            )
            .await
            .unwrap();
        let values: Vec<u64> = got.iter().map(|v| **v).collect();
        assert_eq!(values, [0, 10, 200]);
        assert_eq!(*asked.lock().unwrap(), [vec![0, 2]]);
    }

    #[tokio::test]
    async fn two_stores_never_share_an_entry() {
        let (s1, s2) = (store(), store());
        let shared = cache(1 << 20);
        let (h1, h2) = (shared.handle(&s1), shared.handle(&s2));
        let calls = AtomicUsize::new(0);
        let a = h1
            .get_or_load(&path("x"), What::IndexHead, || counted(&calls, 1, 1))
            .await
            .unwrap();
        let b = h2
            .get_or_load(&path("x"), What::IndexHead, || counted(&calls, 2, 1))
            .await
            .unwrap();
        assert_eq!((*a, *b), (1, 2));
    }

    /// A store dropped and a new one allocated at the same address must not see
    /// the old store's entries.
    #[tokio::test]
    async fn a_dead_stores_entries_are_dropped_when_its_address_comes_back() {
        let shared = cache(1 << 20);
        let s1 = store();
        let id = DecodedCache::store_id(&s1);
        shared
            .handle(&s1)
            .get_or_load(&path("x"), What::IndexHead, || async {
                Ok::<_, ItzmaError>((1u64, 1))
            })
            .await
            .unwrap();
        drop(s1);
        shared.forget_dead(id);
        assert_eq!(shared.entries_of(id), 0);
    }

    #[test]
    fn options_default_to_256_mib() {
        assert_eq!(DecodedCacheOptions::default().max_bytes, 256 << 20);
        assert_eq!(DecodedCacheOptions::default().max_bytes(5).max_bytes, 5);
    }
}
