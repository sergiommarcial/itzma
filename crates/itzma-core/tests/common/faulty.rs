//! A store wrapper for failure testing: it can fail puts by key (for the
//! retry caps and the crash window between writing a segment and publishing
//! it), fail or lose the Nth operation of any kind (for crash-at-every-step
//! sweeps), and yield to the scheduler on every operation (so a task can be
//! aborted between two store calls, which is what a killed process looks
//! like: each call is atomic). Deletes count and can be faulted like puts, and
//! `before_get` runs a hook before a read, to stand in for another process
//! acting between two of a caller's store calls.

use std::fmt;
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::FutureExt;
use futures::future::BoxFuture;
use futures::stream::{self, BoxStream, StreamExt};
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMode, PutMultipartOptions, PutOptions, PutPayload, PutResult, Result,
};

/// Which puts a rule applies to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PutKind {
    /// Put-if-absent (`PutMode::Create`).
    Create,
    /// Conditional update (`PutMode::Update`).
    Update,
    /// Any put.
    Any,
}

impl PutKind {
    fn matches(self, mode: &PutMode) -> bool {
        matches!(
            (self, mode),
            (PutKind::Any, _)
                | (PutKind::Create, PutMode::Create)
                | (PutKind::Update, PutMode::Update(_))
        )
    }
}

/// The failure a rule injects.
#[derive(Debug, Clone, Copy)]
pub enum Fault {
    /// The store says the key already exists.
    AlreadyExists,
    /// The store says a precondition failed (a lost compare-and-swap).
    Precondition,
    /// An unclassified backend error, which the library must not retry. The
    /// operation is not applied.
    Generic,
    /// A lost acknowledgement: a put is applied, then the caller is told it
    /// failed (a timeout after the server committed). On anything that is not
    /// a put it behaves like `Generic`.
    LostAck,
}

impl Fault {
    fn error(self, path: &Path) -> object_store::Error {
        let source: Box<dyn std::error::Error + Send + Sync> = "injected fault".into();
        let path = path.to_string();
        match self {
            Fault::AlreadyExists => object_store::Error::AlreadyExists { path, source },
            Fault::Precondition => object_store::Error::Precondition { path, source },
            Fault::Generic | Fault::LostAck => object_store::Error::Generic {
                store: "faulty",
                source,
            },
        }
    }
}

#[derive(Debug)]
struct Rule {
    needle: String,
    kind: PutKind,
    fault: Fault,
    remaining: Option<usize>,
}

/// Work to run before a read of a matching key, once.
struct Hook {
    needle: String,
    action: BoxFuture<'static, ()>,
}

impl fmt::Debug for Hook {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Hook")
            .field("needle", &self.needle)
            .finish_non_exhaustive()
    }
}

/// Wraps a store and fails operations on demand. Without a rule or a plan,
/// everything goes straight through.
#[derive(Debug)]
pub struct FaultyStore {
    inner: Arc<dyn ObjectStore>,
    rules: Mutex<Vec<Rule>>,
    ops: AtomicUsize,
    nth: Mutex<Option<(usize, Fault)>>,
    yields: AtomicBool,
    hooks: Mutex<Vec<Hook>>,
}

impl FaultyStore {
    pub fn new(inner: Arc<dyn ObjectStore>) -> Arc<Self> {
        Arc::new(Self {
            inner,
            rules: Mutex::new(Vec::new()),
            ops: AtomicUsize::new(0),
            nth: Mutex::new(None),
            yields: AtomicBool::new(false),
            hooks: Mutex::new(Vec::new()),
        })
    }

    /// Fail puts whose key contains `needle` and whose mode matches `kind`.
    /// `times` limits how many puts fail; `None` fails every match.
    pub fn fail_puts(&self, needle: &str, kind: PutKind, fault: Fault, times: Option<usize>) {
        self.rules.lock().expect("rules lock").push(Rule {
            needle: needle.to_string(),
            kind,
            fault,
            remaining: times,
        });
    }

    /// Fail the `n`th store operation from now (0 is the next one), once.
    /// Every kind counts: puts, gets (heads included) and listings. Restarts
    /// the operation count and replaces any earlier plan.
    pub fn fail_nth_op(&self, n: usize, fault: Fault) {
        self.reset_ops();
        *self.nth.lock().expect("nth lock") = Some((n, fault));
    }

    /// Store operations seen since the last `reset_ops` or `fail_nth_op`.
    pub fn ops(&self) -> usize {
        self.ops.load(Ordering::SeqCst)
    }

    /// Restart the operation count at zero.
    pub fn reset_ops(&self) {
        self.ops.store(0, Ordering::SeqCst);
    }

    /// Make every operation yield to the scheduler first, so tasks interleave
    /// and can be aborted between two store calls.
    pub fn yield_each_op(&self, on: bool) {
        self.yields.store(on, Ordering::SeqCst);
    }

    /// Run `action` once, just before the first read (`get` or `head`) of a
    /// key that contains `needle`. It stands in for another process acting
    /// between two of this caller's store calls, such as a merge and a
    /// garbage collection landing after a reader loaded the manifest. Give
    /// it the wrapped store, not this one, so its own calls are not counted
    /// or faulted.
    pub fn before_get(&self, needle: &str, action: impl Future<Output = ()> + Send + 'static) {
        self.hooks.lock().expect("hooks lock").push(Hook {
            needle: needle.to_string(),
            action: action.boxed(),
        });
    }

    async fn run_hook(&self, location: &Path) {
        let key: &str = location.as_ref();
        let hook = {
            let mut hooks = self.hooks.lock().expect("hooks lock");
            let at = hooks.iter().position(|h| key.contains(&h.needle));
            at.map(|i| hooks.remove(i))
        };
        if let Some(hook) = hook {
            hook.action.await;
        }
    }

    /// Remove every rule, hook and planned fault.
    pub fn clear(&self) {
        self.rules.lock().expect("rules lock").clear();
        self.hooks.lock().expect("hooks lock").clear();
        *self.nth.lock().expect("nth lock") = None;
    }

    /// Count one operation and say whether the plan fails it.
    fn tick(&self) -> Option<Fault> {
        let index = self.ops.fetch_add(1, Ordering::SeqCst);
        let mut nth = self.nth.lock().expect("nth lock");
        match *nth {
            Some((n, fault)) if n == index => {
                *nth = None;
                Some(fault)
            }
            _ => None,
        }
    }

    fn injected(&self, path: &Path, mode: &PutMode) -> Option<Fault> {
        let key: &str = path.as_ref();
        let mut rules = self.rules.lock().expect("rules lock");
        let rule = rules
            .iter_mut()
            .find(|r| r.remaining != Some(0) && r.kind.matches(mode) && key.contains(&r.needle))?;
        if let Some(n) = rule.remaining.as_mut() {
            *n -= 1;
        }
        Some(rule.fault)
    }

    async fn pause(&self) {
        if self.yields.load(Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }
    }

    /// Count a listing, apply the plan to it, and yield first if asked.
    fn gate(
        &self,
        prefix: Option<&Path>,
        listing: BoxStream<'static, Result<ObjectMeta>>,
    ) -> BoxStream<'static, Result<ObjectMeta>> {
        if let Some(fault) = self.tick() {
            let err = fault.error(&prefix.cloned().unwrap_or_default());
            return stream::once(async move { Err(err) }).boxed();
        }
        if self.yields.load(Ordering::SeqCst) {
            let pause = stream::once(tokio::task::yield_now())
                .filter_map(|()| async { None::<Result<ObjectMeta>> });
            return pause.chain(listing).boxed();
        }
        listing
    }
}

impl fmt::Display for FaultyStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FaultyStore({})", self.inner)
    }
}

#[async_trait]
impl ObjectStore for FaultyStore {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> Result<PutResult> {
        self.pause().await;
        let fault = match self.tick() {
            Some(fault) => Some(fault),
            None => self.injected(location, &opts.mode),
        };
        match fault {
            None => self.inner.put_opts(location, payload, opts).await,
            Some(Fault::LostAck) => {
                self.inner.put_opts(location, payload, opts).await?;
                Err(Fault::LostAck.error(location))
            }
            Some(fault) => Err(fault.error(location)),
        }
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> Result<GetResult> {
        self.pause().await;
        self.run_hook(location).await;
        if let Some(fault) = self.tick() {
            return Err(fault.error(location));
        }
        self.inner.get_opts(location, options).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, Result<Path>>,
    ) -> BoxStream<'static, Result<Path>> {
        let deleted = match self.tick() {
            None => self.inner.delete_stream(locations),
            Some(Fault::LostAck) => {
                let applied = self.inner.delete_stream(locations);
                stream::once(async move {
                    applied.collect::<Vec<_>>().await;
                    Err(Fault::LostAck.error(&Path::default()))
                })
                .boxed()
            }
            Some(fault) => stream::once(async move { Err(fault.error(&Path::default())) }).boxed(),
        };
        if self.yields.load(Ordering::SeqCst) {
            let pause = stream::once(tokio::task::yield_now())
                .filter_map(|()| async { None::<Result<Path>> });
            return pause.chain(deleted).boxed();
        }
        deleted
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, Result<ObjectMeta>> {
        self.gate(prefix, self.inner.list(prefix))
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, Result<ObjectMeta>> {
        self.gate(prefix, self.inner.list_with_offset(prefix, offset))
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> Result<ListResult> {
        self.pause().await;
        if let Some(fault) = self.tick() {
            return Err(fault.error(&prefix.cloned().unwrap_or_default()));
        }
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(&self, from: &Path, to: &Path, options: CopyOptions) -> Result<()> {
        self.inner.copy_opts(from, to, options).await
    }
}
