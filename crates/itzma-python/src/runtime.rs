//! One tokio runtime per process.
//!
//! A tokio runtime does not survive `fork()`: its worker threads are not
//! copied. So the runtime is keyed by pid. The first call from a new process
//! builds a fresh one and leaks the old one, since dropping a runtime whose
//! threads are gone would wait on them forever. Handles remember the pid they
//! were opened in and refuse calls from any other (`fork_error`).
//!
//! `pyo3_async_runtimes::tokio` keeps one global runtime forever, so it cannot
//! follow a fork. `ItzmaRuntime` implements that crate's generic traits over
//! this module's runtime instead; it is a copy of the crate's tokio module,
//! with `current()` in place of its global.
//!
//! Every task that bridge starts is counted until it ends (`InFlight`), and
//! an `atexit` hook waits for the count to reach zero (`drain`). Without it
//! a task can still be inside Python, setting an asyncio future's result,
//! when the main thread finishes and the interpreter finalizes under it,
//! which aborts the process (`PyGILState_Release: thread state ... must be
//! current when releasing`).

use std::cell::OnceCell;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use pyo3::prelude::*;
use pyo3_async_runtimes::TaskLocals;
use pyo3_async_runtimes::generic::{ContextExt, JoinError, Runtime as GenericRuntime};
use tokio::runtime::Runtime;

use crate::error;

/// Worker threads for the next runtime built; 0 keeps tokio's default.
static WORKERS: AtomicUsize = AtomicUsize::new(0);

/// A runtime and the process it was built in. Leaked, so a reference to one
/// is valid forever, even after a newer one replaces it.
struct Published {
    pid: u32,
    runtime: &'static Runtime,
    /// Bridge tasks running on `runtime`. Kept per runtime, so a forked
    /// child never waits on the parent's tasks, whose threads it lacks.
    in_flight: AtomicUsize,
}

/// The runtime for the latest process that asked. No lock guards it, not even
/// while a runtime is built: a fork can copy a held lock into the child
/// without the thread that would release it, and the child's first call would
/// wait forever. Builders race instead, and the loser shuts its runtime down.
static PUBLISHED: AtomicPtr<Published> = AtomicPtr::new(std::ptr::null_mut());

fn published_for(pid: u32) -> Option<&'static Published> {
    let ptr = PUBLISHED.load(Ordering::Acquire);
    // SAFETY: `PUBLISHED` only ever holds null or a pointer from
    // `Box::leak`, which is never freed.
    let published = unsafe { ptr.as_ref() }?;
    (published.pid == pid).then_some(published)
}

/// Set the worker thread count. Only before the first call in this process.
pub(crate) fn configure(workers: usize) -> Result<(), String> {
    configure_for(std::process::id(), workers)
}

fn configure_for(pid: u32, workers: usize) -> Result<(), String> {
    if workers == 0 {
        return Err("worker_threads must be at least 1".into());
    }
    let refused =
        || Err("the runtime is already running; call configure before the first Itzma call".into());
    if published_for(pid).is_some() {
        return refused();
    }
    WORKERS.store(workers, Ordering::SeqCst);
    // A build that began before the store may have read the old count: it
    // marked itself started first, so this refuses rather than report a
    // setting nobody will use.
    if STARTED.load(Ordering::SeqCst) == pid {
        return refused();
    }
    Ok(())
}

/// The pid that last began building a runtime, set before the build reads
/// `WORKERS` (see `configure_for`).
static STARTED: AtomicU32 = AtomicU32::new(0);

/// The worker count the next runtime will be built with (0: tokio's default).
#[cfg(test)]
pub(crate) fn configured_workers() -> usize {
    WORKERS.load(Ordering::SeqCst)
}

/// This process's runtime, built on first use and again after a fork.
pub(crate) fn current() -> Result<&'static Runtime, String> {
    current_for(std::process::id()).map(|published| published.runtime)
}

fn current_for(pid: u32) -> Result<&'static Published, String> {
    loop {
        let seen = PUBLISHED.load(Ordering::Acquire);
        if let Some(published) = published_for(pid) {
            return Ok(published);
        }
        STARTED.store(pid, Ordering::SeqCst);
        let workers = WORKERS.load(Ordering::SeqCst);
        #[cfg(test)]
        tests::pause_build_of(pid);
        let runtime = Box::into_raw(Box::new(build(workers)?));
        // SAFETY: `runtime` came from `Box::into_raw` just above.
        let published = Box::into_raw(Box::new(Published {
            pid,
            runtime: unsafe { &*runtime },
            in_flight: AtomicUsize::new(0),
        }));
        match PUBLISHED.compare_exchange(seen, published, Ordering::AcqRel, Ordering::Acquire) {
            // Both are now published and never freed; what `seen` pointed at
            // is leaked on purpose, since a handle may still use it.
            // SAFETY: as above.
            Ok(_) => return Ok(unsafe { &*published }),
            Err(_) => {
                // Another thread published first; neither box was shared.
                // SAFETY: both came from `Box::into_raw` above and were not
                // published.
                drop(unsafe { Box::from_raw(published) });
                unsafe { Box::from_raw(runtime) }.shutdown_background();
            }
        }
    }
}

fn build(workers: usize) -> Result<Runtime, String> {
    let mut builder = tokio::runtime::Builder::new_multi_thread();
    builder.enable_all().thread_name("itzma");
    if workers > 0 {
        builder.worker_threads(workers);
    }
    builder
        .build()
        .map_err(|e| format!("cannot start the tokio runtime: {e}"))
}

/// Why a handle opened in process `opened` may not be used in `now`.
pub(crate) fn fork_error(opened: u32, now: u32) -> Option<String> {
    (opened != now).then(|| {
        format!(
            "this Itzma handle was opened in process {opened} and used in process {now}, \
             after a fork; open namespaces in each worker process"
        )
    })
}

/// Run `fut` to completion on this process's runtime with the GIL released.
pub(crate) fn block_on<F, T>(py: Python<'_>, fut: F) -> PyResult<T>
where
    F: Future<Output = PyResult<T>> + Send,
    T: Send,
{
    let runtime = current().map_err(error::itzma_error)?;
    py.detach(|| runtime.block_on(fut))
}

/// Hand `fut` to the caller's asyncio loop as an awaitable.
pub(crate) fn spawn<'py, F, T>(py: Python<'py>, fut: F) -> PyResult<Bound<'py, PyAny>>
where
    F: Future<Output = PyResult<T>> + Send + 'static,
    T: for<'a> IntoPyObject<'a> + Send + 'static,
{
    current().map_err(error::itzma_error)?;
    pyo3_async_runtimes::generic::future_into_py::<ItzmaRuntime, _, T>(py, fut)
}

/// The task counters are per process, so the tests that read them run one at
/// a time.
#[cfg(test)]
pub(crate) static COUNTER_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Wait, up to `limit`, until no bridge task of this process's runtime is
/// running. False if some still are. Call it with the GIL released, since
/// the tasks it waits on may need it to finish.
pub(crate) fn drain(limit: Duration) -> bool {
    let Some(published) = published_for(std::process::id()) else {
        return true;
    };
    let deadline = Instant::now() + limit;
    while published.in_flight.load(Ordering::SeqCst) > 0 {
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    true
}

/// One running bridge task, counted from before it is spawned until it is
/// dropped, which is after any `Python::attach` inside it has returned.
struct InFlight(&'static AtomicUsize);

impl InFlight {
    fn enter(count: &'static AtomicUsize) -> Self {
        count.fetch_add(1, Ordering::SeqCst);
        Self(count)
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

pub(crate) struct ItzmaRuntime;

pub(crate) struct Join(tokio::task::JoinError);

impl JoinError for Join {
    fn is_panic(&self) -> bool {
        self.0.is_panic()
    }

    fn into_panic(self) -> Box<dyn std::any::Any + Send + 'static> {
        self.0.into_panic()
    }
}

type Handle = Pin<Box<dyn Future<Output = Result<(), Join>> + Send>>;

tokio::task_local! {
    static TASK_LOCALS: OnceCell<TaskLocals>;
}

impl GenericRuntime for ItzmaRuntime {
    type JoinError = Join;
    type JoinHandle = Handle;

    fn spawn<F>(fut: F) -> Handle
    where
        F: Future<Output = ()> + Send + 'static,
    {
        match current_for(std::process::id()) {
            Ok(published) => {
                let task = InFlight::enter(&published.in_flight);
                let handle = published.runtime.spawn(async move {
                    let _task = task;
                    fut.await
                });
                Box::pin(async move { handle.await.map_err(Join) })
            }
            // `spawn` above checked `current()` in this same process, so this
            // only happens if the lock was poisoned in between.
            Err(_) => Box::pin(async { Ok(()) }),
        }
    }

    fn spawn_blocking<F>(f: F) -> Handle
    where
        F: FnOnce() + Send + 'static,
    {
        match current_for(std::process::id()) {
            Ok(published) => {
                let task = InFlight::enter(&published.in_flight);
                let handle = published.runtime.spawn_blocking(move || {
                    let _task = task;
                    f()
                });
                Box::pin(async move { handle.await.map_err(Join) })
            }
            Err(_) => Box::pin(async { Ok(()) }),
        }
    }
}

impl ContextExt for ItzmaRuntime {
    fn scope<F, R>(locals: TaskLocals, fut: F) -> Pin<Box<dyn Future<Output = R> + Send>>
    where
        F: Future<Output = R> + Send + 'static,
    {
        let cell = OnceCell::new();
        let _ = cell.set(locals);
        Box::pin(TASK_LOCALS.scope(cell, fut))
    }

    fn get_task_locals() -> Option<TaskLocals> {
        TASK_LOCALS
            .try_with(|cell| cell.get().cloned())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;

    use super::*;

    /// A pid whose runtime build stops before the runtime is made until
    /// `RELEASE` is set: a build a fork left half done.
    static PAUSED_PID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    static RELEASE: AtomicBool = AtomicBool::new(false);

    /// The pause hook is shared, so the tests that use it run one at a time.
    static PAUSING: std::sync::Mutex<()> = std::sync::Mutex::new(());

    pub(super) fn pause_build_of(pid: u32) {
        if PAUSED_PID.load(Ordering::SeqCst) == pid {
            while !RELEASE.load(Ordering::SeqCst) {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }

    #[test]
    fn a_configure_racing_the_first_build_is_never_ok_and_ignored() {
        let _one = PAUSING.lock().unwrap_or_else(|e| e.into_inner());
        let racing = u32::MAX - 5;
        PAUSED_PID.store(racing, Ordering::SeqCst);
        RELEASE.store(false, Ordering::SeqCst);
        // The build has read the worker count and is about to make the runtime.
        let build = std::thread::spawn(move || {
            current_for(racing)
                .expect("runtime")
                .runtime
                .metrics()
                .num_workers()
        });
        std::thread::sleep(std::time::Duration::from_millis(50));
        let configured = configure_for(racing, 3);
        RELEASE.store(true, Ordering::SeqCst);
        let workers = build.join().expect("build");
        PAUSED_PID.store(0, Ordering::SeqCst);
        WORKERS.store(0, Ordering::SeqCst);
        let _ = current();
        if configured.is_ok() {
            assert_eq!(workers, 3, "configure said Ok but the runtime ignored it");
        }
    }

    #[test]
    fn a_build_left_half_done_never_blocks_another_pid() {
        let _one = PAUSING.lock().unwrap_or_else(|e| e.into_inner());
        let (stuck, fresh) = (u32::MAX - 3, u32::MAX - 4);
        PAUSED_PID.store(stuck, Ordering::SeqCst);
        RELEASE.store(false, Ordering::SeqCst);
        let parent = std::thread::spawn(move || {
            let _ = current_for(stuck);
        });
        std::thread::sleep(std::time::Duration::from_millis(50));
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(current_for(fresh).is_ok());
        });
        let got = rx.recv_timeout(std::time::Duration::from_secs(10));
        RELEASE.store(true, Ordering::SeqCst);
        parent
            .join()
            .expect("the paused build finishes once released");
        // Give the real process its runtime back for the other tests.
        let _ = current();
        assert_eq!(got, Ok(true), "the child's build waited on the parent's");
    }

    #[test]
    fn drain_waits_for_running_bridge_tasks() {
        let _counter = COUNTER_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        // Dropping a handle leaves its task running.
        drop(ItzmaRuntime::spawn_blocking(move || {
            let _ = rx.recv();
        }));
        drop(ItzmaRuntime::spawn(async {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }));
        assert!(
            !drain(Duration::from_millis(20)),
            "drain returned while a task was running"
        );
        tx.send(()).expect("the blocking task is waiting");
        assert!(drain(Duration::from_secs(10)), "the tasks never finished");
    }

    #[test]
    fn same_process_is_not_a_fork() {
        assert_eq!(fork_error(42, 42), None);
    }

    #[test]
    fn another_process_is_a_fork() {
        let message = fork_error(42, 43).expect("a different pid is a fork");
        assert!(message.contains("fork"), "{message}");
        assert!(
            message.contains("42") && message.contains("43"),
            "{message}"
        );
    }

    #[test]
    fn current_is_built_once_per_process() {
        let a = current().expect("runtime") as *const Runtime;
        let b = current().expect("runtime") as *const Runtime;
        assert_eq!(a, b);
    }

    #[test]
    fn configure_refuses_zero_workers() {
        let err = configure(0).expect_err("zero workers");
        assert!(err.contains("at least 1"), "{err}");
    }

    #[test]
    fn configure_after_start_is_refused() {
        current().expect("runtime");
        let err = configure(2).expect_err("already running");
        assert!(err.contains("already running"), "{err}");
    }
}
