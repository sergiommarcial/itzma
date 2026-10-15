//! Where CPU work runs.
//!
//! Two private rayon pools: the query lane (every core) and the maintenance
//! lane (a quarter of them), so an index build or a merge cannot take the cores
//! queries need. Work below [`INLINE_BELOW`] runs on the caller's thread, since
//! a hand-off costs more than it.
//!
//! The pools are keyed by process id. A fork copies no threads, so a pool from
//! the parent would wait forever; the first call in a new process builds new
//! pools and leaks the old ones, as `itzma-python`'s runtime does for tokio.

use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicUsize, Ordering};

use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::error::{ItzmaError, Result};

/// Which pool a piece of work belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Lane {
    /// Queries, filters and scans: every core.
    Query,
    /// Compaction, index builds, maintenance and recovery: a quarter of them.
    Maintenance,
}

/// A rough size of a piece of work, to decide whether to hand it off.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Cost {
    /// Distance computations of `dim` values each.
    Distances { rows: usize, dim: usize },
    /// Bytes to decode or encode (for a compressed object, what it decodes to).
    Bytes(usize),
}

impl Cost {
    pub(crate) fn units(self) -> usize {
        match self {
            Cost::Distances { rows, dim } => rows.saturating_mul(dim),
            // A segment decodes at about 0.4 ns per byte (`measure_handoff`),
            // so three units a byte puts the hand-off near 400 KB.
            Cost::Bytes(n) => n.saturating_mul(3),
        }
    }
}

/// Work under this many units stays on the caller. Measured (`measure_handoff`
/// below, release, Apple M-series): a hand-off costs 8 µs back to back and
/// 14 µs when the pool sleeps, and 64,000 units (500 rows at dimension 128)
/// take about 7 µs. Work is handed off only when
/// it is worth ten hand-offs, about 140 µs. Measured on one machine only;
/// Linux x86 is
pub(crate) const INLINE_BELOW: usize = 1_250_000;

/// Thread counts for this process's two compute pools.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct CpuThreads {
    /// Threads of the query lane: queries, filters and scans.
    pub query: usize,
    /// Threads of the maintenance lane: compaction, index builds, recovery.
    pub maintenance: usize,
}

impl Default for CpuThreads {
    /// Every core for queries, a quarter of them (at least one) for maintenance.
    fn default() -> Self {
        let cores = std::thread::available_parallelism().map_or(1, usize::from);
        CpuThreads {
            query: cores,
            maintenance: (cores / 4).max(1),
        }
    }
}

impl CpuThreads {
    /// Set the query lane's thread count.
    pub fn query(mut self, n: usize) -> Self {
        self.query = n;
        self
    }

    /// Set the maintenance lane's thread count.
    pub fn maintenance(mut self, n: usize) -> Self {
        self.maintenance = n;
        self
    }
}

/// The pid that last began building pools, set before the build reads the
/// thread counts, so a `configure_cpu` that stores counts after that read sees
/// it and refuses instead of reporting success for counts nobody will use.
static STARTED: AtomicU32 = AtomicU32::new(0);

/// 0 means "not configured": use `CpuThreads::default()`.
static QUERY_THREADS: AtomicUsize = AtomicUsize::new(0);
static MAINTENANCE_THREADS: AtomicUsize = AtomicUsize::new(0);

/// Set the thread counts of this process's compute pools. The pools belong to
/// the process, not to a [`Namespace`](crate::Namespace), so this is called
/// once, before the first query or build. A forked child has no pools yet and
/// may call it again.
///
/// # Errors
///
/// [`ItzmaError::InvalidConfig`] for a count of 0, or when this process's
/// pools already exist.
///
/// ```
/// use itzma_core::{CpuThreads, configure_cpu};
///
/// // In a process that has not run any Itzma work yet.
/// configure_cpu(CpuThreads::default().query(4).maintenance(1)).unwrap();
/// ```
pub fn configure_cpu(threads: CpuThreads) -> Result<()> {
    configure_cpu_for(std::process::id(), threads)
}

fn configure_cpu_for(pid: u32, threads: CpuThreads) -> Result<()> {
    if threads.query == 0 || threads.maintenance == 0 {
        return Err(ItzmaError::InvalidConfig(
            "each compute lane needs at least one thread",
        ));
    }
    let refused = || {
        Err(ItzmaError::InvalidConfig(
            "the compute pools are already running; configure them before the first call",
        ))
    };
    if published_for(pid).is_some() {
        return refused();
    }
    QUERY_THREADS.store(threads.query, Ordering::SeqCst);
    MAINTENANCE_THREADS.store(threads.maintenance, Ordering::SeqCst);
    // A build that began before the stores may have read the old counts.
    // Either it marked itself started first and this refuses, or it reads
    // the counts just stored: never an `Ok` that the pools ignore.
    if STARTED.load(Ordering::SeqCst) == pid {
        return refused();
    }
    Ok(())
}

/// This process's two pools.
pub(crate) struct Pools {
    pid: u32,
    query: ThreadPool,
    maintenance: ThreadPool,
}

/// The pools of the latest process that asked. No lock guards it, not even
/// while pools are built: a fork can copy a held lock into the child without
/// the thread that would release it, and the child's first hand-off would wait
/// forever. Builders race instead, and the loser drops what it built.
static PUBLISHED: AtomicPtr<Pools> = AtomicPtr::new(std::ptr::null_mut());

fn published_for(pid: u32) -> Option<&'static Pools> {
    let ptr = PUBLISHED.load(Ordering::Acquire);
    // SAFETY: `PUBLISHED` only ever holds null or a pointer from `Box::leak`,
    // which is never freed.
    let pools = unsafe { ptr.as_ref() }?;
    (pools.pid == pid).then_some(pools)
}

fn pools() -> &'static Pools {
    pools_for(std::process::id())
}

fn pools_for(pid: u32) -> &'static Pools {
    loop {
        let seen = PUBLISHED.load(Ordering::Acquire);
        if let Some(pools) = published_for(pid) {
            return pools;
        }
        STARTED.store(pid, Ordering::SeqCst);
        let defaults = CpuThreads::default();
        let pick = |set: &AtomicUsize, default: usize| match set.load(Ordering::SeqCst) {
            0 => default,
            n => n,
        };
        let (query, maintenance) = (
            pick(&QUERY_THREADS, defaults.query),
            pick(&MAINTENANCE_THREADS, defaults.maintenance),
        );
        #[cfg(test)]
        tests::pause_build_of(pid);
        let built = Box::new(Pools {
            pid,
            query: build("itzma-query", query),
            maintenance: build("itzma-maint", maintenance),
        });
        let ours = Box::into_raw(built);
        match PUBLISHED.compare_exchange(seen, ours, Ordering::AcqRel, Ordering::Acquire) {
            // What `seen` pointed at, if anything, is leaked on purpose: another
            // thread may still hold a reference to it.
            // SAFETY: `ours` came from `Box::into_raw` just above and is now
            // published, so it is never freed.
            Ok(_) => return unsafe { &*ours },
            Err(_) => {
                // Another thread published first. `ours` was never shared, so
                // it is dropped here (its threads exit), and the loop reads
                // what won.
                // SAFETY: `ours` came from `Box::into_raw` above and was not
                // published.
                drop(unsafe { Box::from_raw(ours) });
            }
        }
    }
}

fn build(prefix: &'static str, threads: usize) -> ThreadPool {
    let pool = |threads: usize| {
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(move |i| format!("{prefix}-{i}"))
            .build()
    };
    // A process at its thread limit gets one thread; if even that cannot
    // start, no work can run at all and there is no caller to tell but this.
    pool(threads)
        .or_else(|_| pool(1))
        .unwrap_or_else(|e| panic!("cannot start a compute thread: {e}"))
}

/// Run `f` on `lane`'s pool and wait for it, or inline when `cost` is below
/// [`INLINE_BELOW`]. A panic in `f` resumes on the caller. Dropping the
/// returned future does not stop `f`; its result is thrown away.
pub(crate) async fn run<F, T>(lane: Lane, cost: Cost, f: F) -> T
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    if cost.units() < INLINE_BELOW {
        return f();
    }
    let pools = pools();
    let pool = match lane {
        Lane::Query => &pools.query,
        Lane::Maintenance => &pools.maintenance,
    };
    let (send, receive) = futures::channel::oneshot::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let job_cancel = cancel.clone();
    pool.spawn(move || {
        let result = with_cancel(&job_cancel, || panic::catch_unwind(AssertUnwindSafe(f)));
        // The receiver is gone when the caller was dropped; nothing to do.
        let _ = send.send(result);
    });
    // Dropping this future before the answer comes marks the job cancelled,
    // so `chunked` starts no more chunks for it.
    let _cancel_on_drop = CancelOnDrop(cancel);
    match receive.await {
        Ok(Ok(value)) => value,
        Ok(Err(payload)) => panic::resume_unwind(payload),
        // The closure always sends, even after a panic, so a dropped sender
        // means its thread was killed from outside.
        Err(_) => panic!("a compute thread exited without an answer"),
    }
}

/// Documents per chunk of a flat scan: a constant, never derived
/// from the thread count, so a result does not depend on the machine.
pub(crate) const FLAT_CHUNK: usize = 16_384;

/// `f` over `items` in chunks of `chunk`, results in chunk order. Parallel only
/// on a pool thread (inside [`run`]); anywhere else the same chunks run one
/// after another, so the answer is the same either way and rayon's global pool
/// is never started.
pub(crate) fn chunked<I, R, F>(items: &[I], chunk: usize, f: F) -> Vec<R>
where
    I: Sync,
    R: Send,
    F: Fn(&[I]) -> R + Sync,
{
    use rayon::prelude::*;
    let chunk = chunk.max(1);
    if rayon::current_thread_index().is_none() {
        return items.chunks(chunk).map(f).collect();
    }
    // Chunks run on other pool threads, which do not see this thread's job;
    // each takes the job's cancel flag along, so a nested `chunked` sees it too.
    let cancel = CANCEL.with(|c| c.borrow().clone());
    items
        .par_chunks(chunk)
        .map(|one| match &cancel {
            None => f(one),
            Some(cancel) => {
                stop_if_cancelled(cancel);
                with_cancel(cancel, || f(one))
            }
        })
        .collect()
}

/// The payload a cancelled job unwinds with. Nobody sees it: the caller that
/// would have is gone.
struct Cancelled;

thread_local! {
    /// The cancel flag of the `run` job this pool thread is working for.
    static CANCEL: std::cell::RefCell<Option<Arc<AtomicBool>>> =
        const { std::cell::RefCell::new(None) };
}

/// Run `f` with `cancel` as this thread's job flag, then put back what was
/// there (a thread can work for several jobs in turn).
fn with_cancel<T>(cancel: &Arc<AtomicBool>, f: impl FnOnce() -> T) -> T {
    let previous = CANCEL.with(|c| c.replace(Some(cancel.clone())));
    let out = panic::catch_unwind(AssertUnwindSafe(f));
    CANCEL.with(|c| *c.borrow_mut() = previous);
    out.unwrap_or_else(|payload| panic::resume_unwind(payload))
}

/// Stop the job here if its caller left. `resume_unwind` skips the panic
/// hook, so a cancelled job prints nothing.
fn stop_if_cancelled(cancel: &AtomicBool) {
    if cancel.load(Ordering::Relaxed) {
        panic::resume_unwind(Box::new(Cancelled));
    }
}

/// Marks a job cancelled when the future waiting for it is dropped. Set also
/// after a normal answer, when nothing reads it any more.
struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// The lane of the pool thread this runs on, if any.
#[cfg(test)]
pub(crate) fn on_pool_thread() -> Option<Lane> {
    let name = std::thread::current().name()?.to_owned();
    if name.starts_with("itzma-query-") {
        Some(Lane::Query)
    } else if name.starts_with("itzma-maint-") {
        Some(Lane::Maintenance)
    } else {
        None
    }
}

// The `PID_SWAP` read guard is held across awaits on purpose: each test has its
// own runtime, readers share the lock, and the one writer is a sync test.
#[cfg(test)]
#[allow(clippy::await_holding_lock)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
    use std::sync::{Arc, RwLock};

    const BIG: Cost = Cost::Distances {
        rows: 1_000_000,
        dim: 128,
    };
    const SMALL: Cost = Cost::Distances { rows: 10, dim: 8 };

    /// A pid whose pool build stops (inside `pools_for`, before the pools
    /// are made) until `RELEASE` is set: a build a fork left half done.
    static PAUSED_PID: AtomicU32 = AtomicU32::new(0);
    static RELEASE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

    pub(super) fn pause_build_of(pid: u32) {
        if PAUSED_PID.load(Ordering::SeqCst) == pid {
            while !RELEASE.load(Ordering::SeqCst) {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }

    #[test]
    fn a_configure_racing_the_first_build_is_never_ok_and_ignored() {
        let _g = PID_SWAP.write().unwrap_or_else(|e| e.into_inner());
        let racing = u32::MAX - 5;
        PAUSED_PID.store(racing, Ordering::SeqCst);
        RELEASE.store(false, Ordering::SeqCst);
        // The build has read its thread counts and is about to make the pools.
        let build = std::thread::spawn(move || pools_for(racing).query.current_num_threads());
        std::thread::sleep(std::time::Duration::from_millis(50));
        let configured = configure_cpu_for(racing, CpuThreads::default().query(3));
        RELEASE.store(true, Ordering::SeqCst);
        let threads = build.join().expect("build");
        PAUSED_PID.store(0, Ordering::SeqCst);
        QUERY_THREADS.store(0, Ordering::SeqCst);
        MAINTENANCE_THREADS.store(0, Ordering::SeqCst);
        let _ = pools();
        if configured.is_ok() {
            assert_eq!(threads, 3, "configure said Ok but the pools ignored it");
        }
    }

    #[test]
    fn a_build_left_half_done_never_blocks_another_pid() {
        let _g = PID_SWAP.write().unwrap_or_else(|e| e.into_inner());
        let (stuck, fresh) = (u32::MAX - 3, u32::MAX - 4);
        PAUSED_PID.store(stuck, Ordering::SeqCst);
        RELEASE.store(false, Ordering::SeqCst);
        let parent = std::thread::spawn(move || {
            pools_for(stuck);
        });
        std::thread::sleep(std::time::Duration::from_millis(50));
        let (send, receive) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = send.send(pools_for(fresh).pid);
        });
        let got = receive.recv_timeout(std::time::Duration::from_secs(10));
        RELEASE.store(true, Ordering::SeqCst);
        parent
            .join()
            .expect("the paused build finishes once released");
        PAUSED_PID.store(0, Ordering::SeqCst);
        let _ = pools();
        assert_eq!(
            got.ok(),
            Some(fresh),
            "the child's build waited on the parent's"
        );
    }

    /// `another_pid_gets_new_pools` repoints the published pools at a fake
    /// pid; the other tests must not run on two pools across that swap.
    static PID_SWAP: RwLock<()> = RwLock::new(());

    fn shared() -> std::sync::RwLockReadGuard<'static, ()> {
        PID_SWAP.read().unwrap_or_else(|e| e.into_inner())
    }

    #[tokio::test]
    async fn big_work_runs_on_its_lane() {
        let _g = shared();
        assert_eq!(
            run(Lane::Query, BIG, on_pool_thread).await,
            Some(Lane::Query)
        );
        assert_eq!(
            run(Lane::Maintenance, BIG, on_pool_thread).await,
            Some(Lane::Maintenance)
        );
    }

    #[test]
    fn decoding_a_few_hundred_kilobytes_is_worth_a_hand_off() {
        // Measured in `measure_handoff`: a segment decodes at about 0.4 ns
        // per byte, so ten hand-offs (about 140 µs) is about 400 KB.
        assert!(Cost::Bytes(1 << 20).units() >= INLINE_BELOW);
        assert!(Cost::Bytes(512 << 10).units() >= INLINE_BELOW);
        assert!(Cost::Bytes(64 << 10).units() < INLINE_BELOW);
    }

    #[tokio::test]
    async fn small_work_runs_inline() {
        let _g = shared();
        assert_eq!(run(Lane::Query, SMALL, on_pool_thread).await, None);
    }

    #[tokio::test]
    async fn maintenance_never_uses_more_threads_than_its_lane() {
        let _g = shared();
        let lane = pools().maintenance.current_num_threads();
        let busy = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let runs = (0..lane * 4).map(|_| {
            let (busy, peak) = (busy.clone(), peak.clone());
            run(Lane::Maintenance, BIG, move || {
                let now = busy.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                std::thread::sleep(std::time::Duration::from_millis(5));
                busy.fetch_sub(1, Ordering::SeqCst);
            })
        });
        futures::future::join_all(runs).await;
        let peak = peak.load(Ordering::SeqCst);
        assert!(peak >= 1 && peak <= lane, "peak {peak}, lane {lane}");
    }

    #[tokio::test]
    async fn a_panic_resumes_on_the_caller_and_the_pool_survives() {
        let _g = shared();
        let caught = tokio::spawn(run(Lane::Query, BIG, || -> u32 { panic!("kernel bug") }))
            .await
            .expect_err("the panic comes back");
        let payload = caught.into_panic();
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"kernel bug"));
        assert_eq!(run(Lane::Query, BIG, || 7).await, 7);
    }

    #[tokio::test]
    async fn a_dropped_caller_stops_its_chunked_work_early() {
        let _g = shared();
        let items: Vec<u32> = (0..5_000).collect();
        let done = Arc::new(AtomicUsize::new(0));
        let counted = done.clone();
        // 5,000 chunks of a millisecond: about half a second on a 10-core
        // lane if nothing stops it.
        let job = run(Lane::Query, BIG, move || {
            chunked(&items, 1, |_| {
                std::thread::sleep(std::time::Duration::from_millis(1));
                counted.fetch_add(1, Ordering::SeqCst);
            })
            .len()
        });
        let timed_out = tokio::time::timeout(std::time::Duration::from_millis(20), job).await;
        assert!(timed_out.is_err());
        tokio::time::sleep(std::time::Duration::from_millis(1_500)).await;
        let ran = done.load(Ordering::SeqCst);
        assert!(
            ran < 2_500,
            "{ran} of 5,000 chunks ran after the caller left"
        );
        assert_eq!(
            run(Lane::Query, BIG, || 1).await,
            1,
            "the pool still serves"
        );
    }

    #[tokio::test]
    async fn dropping_the_caller_does_not_hang_the_pool() {
        let _g = shared();
        let slow = run(Lane::Query, BIG, || {
            std::thread::sleep(std::time::Duration::from_millis(50));
        });
        let timed_out = tokio::time::timeout(std::time::Duration::from_millis(5), slow).await;
        assert!(timed_out.is_err());
        assert_eq!(run(Lane::Query, BIG, || 1).await, 1);
    }

    #[test]
    fn another_pid_gets_new_pools() {
        let _g = PID_SWAP.write().unwrap_or_else(|e| e.into_inner());
        let here: *const Pools = pools_for(u32::MAX - 1);
        assert_eq!(pools_for(u32::MAX - 1) as *const Pools, here);
        assert_ne!(pools_for(u32::MAX - 2) as *const Pools, here);
        // Hand the real process its pools back before the lock is released.
        let _ = pools();
    }

    #[test]
    fn configure_refuses_zero_and_a_running_pool() {
        let _g = shared();
        assert!(matches!(
            configure_cpu(CpuThreads::default().query(0)),
            Err(ItzmaError::InvalidConfig(_))
        ));
        assert!(matches!(
            configure_cpu(CpuThreads::default().maintenance(0)),
            Err(ItzmaError::InvalidConfig(_))
        ));
        let _ = pools();
        assert!(matches!(
            configure_cpu(CpuThreads::default()),
            Err(ItzmaError::InvalidConfig(_))
        ));
    }

    fn sums(items: &[f64]) -> Vec<f64> {
        chunked(items, 1000, |c| c.iter().sum::<f64>())
    }

    #[test]
    fn chunked_is_the_same_on_every_thread_count() {
        let items: Vec<f64> = (0..10_007).map(|i| (i as f64).sin() * 1e-3).collect();
        let sequential = sums(&items);
        assert_eq!(sequential.len(), 11);
        for threads in [1, 2, 8] {
            let pool = ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|i| format!("itzma-query-{i}"))
                .build()
                .expect("pool");
            assert_eq!(
                pool.install(|| sums(&items)),
                sequential,
                "{threads} threads"
            );
        }
    }

    #[test]
    fn chunked_keeps_chunk_order() {
        let items: Vec<u32> = (0..5000).collect();
        let pool = ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("pool");
        let firsts = pool.install(|| chunked(&items, 1024, |c| (c.len(), c[0])));
        assert_eq!(
            firsts,
            vec![
                (1024, 0),
                (1024, 1024),
                (1024, 2048),
                (1024, 3072),
                (904, 4096)
            ]
        );
    }

    #[test]
    fn chunked_runs_in_parallel_only_on_a_pool_thread() {
        let items = [0u8; 64];
        let outside = chunked(&items, 1, |_| rayon::current_thread_index());
        assert!(
            outside.iter().all(Option::is_none),
            "never on rayon's global pool"
        );
        let pool = ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("pool");
        let inside = pool.install(|| chunked(&items, 1, |_| rayon::current_thread_index()));
        assert!(inside.iter().all(Option::is_some));
    }

    /// `cargo test -p itzma-core --release --lib cpu::tests::measure_handoff -- --ignored --nocapture`
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore]
    async fn measure_handoff() {
        let _g = shared();
        let rounds: u32 = 2_000;
        let start = std::time::Instant::now();
        for _ in 0..rounds {
            std::hint::black_box(run(Lane::Query, BIG, || 1u32).await);
        }
        println!("hand-off, back to back: {:?}", start.elapsed() / rounds);
        let mut cold = std::time::Duration::ZERO;
        for _ in 0..200 {
            std::thread::sleep(std::time::Duration::from_millis(2));
            let start = std::time::Instant::now();
            std::hint::black_box(run(Lane::Query, BIG, || 1u32).await);
            cold += start.elapsed();
        }
        println!("hand-off, pool asleep: {:?}", cold / 200);
        let q = vec![0.5f32; 128];
        let rows: Vec<f32> = (0..100_000 * 128)
            .map(|i| (i % 977) as f32 * 1e-3)
            .collect();
        let work = move || {
            let mut s = 0f32;
            for row in rows.chunks_exact(128) {
                s += crate::search::distance(crate::document::Metric::L2, &q, row);
            }
            s
        };
        let work = std::sync::Arc::new(work);
        for _ in 0..3 {
            let w = work.clone();
            let start = std::time::Instant::now();
            std::hint::black_box(w());
            let inline = start.elapsed();
            let w = work.clone();
            let start = std::time::Instant::now();
            std::hint::black_box(run(Lane::Query, BIG, move || w()).await);
            println!(
                "100k x 128 inline {inline:?}, on the pool {:?}",
                start.elapsed()
            );
        }
        let small_rows: Vec<f32> = (0..500 * 128).map(|i| i as f32 * 1e-4).collect();
        let q = vec![0.5f32; 128];
        let start = std::time::Instant::now();
        for _ in 0..200 {
            for row in small_rows.chunks_exact(128) {
                std::hint::black_box(crate::search::distance(
                    crate::document::Metric::L2,
                    &q,
                    row,
                ));
            }
        }
        println!("500 x 128 inline: {:?}", start.elapsed() / 200);

        // Decode cost per byte, for `Cost::Bytes`: a segment of 4,000
        // documents at dimension 128, every third with attributes, and the
        // same documents as one WAL object.
        let mut layer = crate::layer::Layer::new();
        let mut ops = Vec::new();
        for i in 0..4_000usize {
            let doc = crate::document::Document {
                id: format!("d{i:06}"),
                vector: (0..128).map(|d| ((i * 31 + d) % 97) as f32).collect(),
                attrs: (i % 3 == 0).then(|| serde_json::json!({ "n": i, "tag": "x" })),
            };
            ops.push(crate::document::WalOp::Upsert(doc.clone()));
            layer.apply(crate::document::WalOp::Upsert(doc));
        }
        let segment = crate::segment::encode_as(&layer, 128, 2).expect("encode");
        let start = std::time::Instant::now();
        for _ in 0..20 {
            std::hint::black_box(crate::segment::decode("p", &segment, 128).expect("decode"));
        }
        let per = start.elapsed() / 20;
        println!(
            "segment decode: {} bytes in {per:?}, {:.1} ns per byte",
            segment.len(),
            per.as_nanos() as f64 / segment.len() as f64
        );
        let wal = crate::wal_codec::encode(&ops, usize::MAX).expect("encode");
        let start = std::time::Instant::now();
        for _ in 0..20 {
            std::hint::black_box(crate::wal_codec::decode("p", &wal).expect("decode"));
        }
        let per = start.elapsed() / 20;
        println!(
            "wal decode: {} bytes in {per:?}, {:.1} ns per byte",
            wal.len(),
            per.as_nanos() as f64 / wal.len() as f64
        );
    }
}
