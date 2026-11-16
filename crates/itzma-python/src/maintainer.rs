//! A background loop that keeps a namespace compacted and indexed.
//!
//! The library starts no task on its own; this one runs when Python asks,
//! explicitly with `start_maintenance` or by default when a handle opens. A
//! loop given a lease maintains only while it holds it. It holds a weak
//! reference, so it ends when the handle does.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use itzma_core::{
    IndexOptions, IndexSparseOptions, ItzmaError, MaintenanceLease, MaintenanceOptions, Namespace,
    Role,
};
use pyo3::prelude::*;
use serde_json::Value;
use tokio::sync::Notify;
use tokio::task::{AbortHandle, JoinHandle};
use tokio::time::MissedTickBehavior;

use crate::{reports, runtime};

pub(crate) struct Plan {
    pub(crate) every: Duration,
    pub(crate) maintain: MaintenanceOptions,
    pub(crate) index: Option<(IndexOptions, IndexSparseOptions)>,
}

#[derive(Default)]
struct State {
    runs: AtomicU64,
    errors: AtomicU64,
    leader: AtomicBool,
    last_error: Mutex<Option<String>>,
    last_report: Mutex<Option<String>>,
}

impl State {
    /// Record a finished run, or a lease error, which counts like a failed run.
    fn record(&self, (report, error): (Option<String>, Option<String>)) {
        if let Some(report) = report {
            set(&self.last_report, report);
        }
        if let Some(message) = error {
            log_warning("maintenance run failed: %s", &message);
            self.errors.fetch_add(1, Ordering::SeqCst);
            set(&self.last_error, message);
        }
    }
}

fn set(slot: &Mutex<Option<String>>, value: String) {
    if let Ok(mut slot) = slot.lock() {
        *slot = Some(value);
    }
}

fn get(slot: &Mutex<Option<String>>) -> Option<String> {
    slot.lock().ok().and_then(|s| s.clone())
}

#[pyclass(frozen, name = "Maintainer", module = "itzma._itzma")]
pub(crate) struct PyMaintainer {
    state: Arc<State>,
    task: AbortHandle,
    /// Asks a leased loop to release its lease and end. An unleased loop is
    /// aborted instead.
    stop: Arc<Notify>,
    leased: bool,
    /// The process whose runtime runs the task.
    pid: u32,
}

/// Start a loop. With a lease it ticks every `renew` and maintains only while
/// leader, the first run `every` after it becomes leader. Without one it runs
/// every `every`, the first run at once.
pub(crate) fn start(
    ns: Weak<Namespace>,
    plan: Plan,
    lease: Option<MaintenanceLease>,
) -> Result<PyMaintainer, String> {
    let runtime = runtime::current()?;
    let state = Arc::new(State::default());
    let stop = Arc::new(Notify::new());
    let leased = lease.is_some();
    let task = match lease {
        Some(lease) => runtime.spawn(run_leased(
            ns,
            Arc::new(plan),
            lease,
            state.clone(),
            stop.clone(),
        )),
        None => {
            state.leader.store(true, Ordering::SeqCst);
            runtime.spawn(run(ns, plan, state.clone()))
        }
    };
    Ok(PyMaintainer {
        state,
        task: task.abort_handle(),
        stop,
        leased,
        pid: std::process::id(),
    })
}

async fn run(handle: Weak<Namespace>, plan: Plan, state: Arc<State>) {
    let mut ticks = tokio::time::interval(plan.every);
    ticks.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        ticks.tick().await;
        let Some(ns) = handle.upgrade() else { break };
        state.record(tick(&ns, &plan).await);
        state.runs.fetch_add(1, Ordering::SeqCst);
    }
}

/// A store that cannot hold the lease, or credentials that may not write it.
/// Trying again every tick would only log the same thing forever.
fn lease_is_hopeless(e: &ItzmaError) -> bool {
    matches!(
        e,
        ItzmaError::Storage(
            object_store::Error::NotImplemented { .. }
                | object_store::Error::PermissionDenied { .. }
                | object_store::Error::Unauthenticated { .. }
        )
    )
}

async fn run_leased(
    handle: Weak<Namespace>,
    plan: Arc<Plan>,
    mut lease: MaintenanceLease,
    state: Arc<State>,
    stop: Arc<Notify>,
) {
    let mut ticks = tokio::time::interval(lease.options().renew);
    ticks.set_missed_tick_behavior(MissedTickBehavior::Delay);
    // When this loop last became leader or last started a run.
    let mut since: Option<Instant> = None;
    let mut run: Option<JoinHandle<(Option<String>, Option<String>)>> = None;
    loop {
        tokio::select! {
            () = stop.notified() => break,
            done = async { run.as_mut().expect("guarded").await }, if run.is_some() => {
                run = None;
                if let Ok(outcome) = done {
                    state.record(outcome);
                    state.runs.fetch_add(1, Ordering::SeqCst);
                }
            }
            _ = ticks.tick() => {
                let Some(ns) = handle.upgrade() else { break };
                let role = lease.tick(Instant::now()).await;
                state.leader.store(lease.role() == Role::Leader, Ordering::SeqCst);
                match role {
                    Ok(Role::Leader) => {
                        let now = Instant::now();
                        let from = *since.get_or_insert(now);
                        if run.is_none() && now.duration_since(from) >= plan.every {
                            since = Some(now);
                            let plan = plan.clone();
                            run = Some(tokio::spawn(async move { tick(&ns, &plan).await }));
                        }
                    }
                    Ok(Role::Follower) => since = None,
                    Err(e) if lease_is_hopeless(&e) => {
                        let message = e.to_string();
                        log_warning("maintenance stopped: %s", &message);
                        state.errors.fetch_add(1, Ordering::SeqCst);
                        set(&state.last_error, message);
                        break;
                    }
                    Err(e) => {
                        since = None;
                        state.record((None, Some(e.to_string())));
                    }
                }
            }
        }
    }
    if let Some(run) = run {
        run.abort();
    }
    let _ = lease.release().await;
}

/// One run: the report, if `maintain` succeeded, and the error, if anything
/// failed. A failed index build keeps the report of the maintenance that
/// did run, since it may have published a compaction.
async fn tick(ns: &Namespace, plan: &Plan) -> (Option<String>, Option<String>) {
    let maintained = ns
        .maintain(&plan.maintain)
        .await
        .map(|r| reports::maintenance(&r))
        .map_err(|e| e.to_string());
    let indexed = match (&maintained, &plan.index) {
        (Ok(_), Some((index, sparse))) => Some(index_both(ns, index, sparse).await),
        _ => None,
    };
    outcome(maintained, indexed)
}

async fn index_both(
    ns: &Namespace,
    index: &IndexOptions,
    sparse: &IndexSparseOptions,
) -> Result<(Value, Value), String> {
    let dense = ns.index(index).await.map_err(|e| e.to_string())?;
    let sparse = ns.index_sparse(sparse).await.map_err(|e| e.to_string())?;
    Ok((
        serde_json::to_value(dense).unwrap_or_default(),
        serde_json::to_value(sparse).unwrap_or_default(),
    ))
}

fn outcome(
    maintained: Result<Value, String>,
    indexed: Option<Result<(Value, Value), String>>,
) -> (Option<String>, Option<String>) {
    let mut report = match maintained {
        Ok(report) => report,
        Err(e) => return (None, Some(e)),
    };
    let error = match indexed {
        Some(Ok((dense, sparse))) => {
            report["index"] = dense;
            report["index_sparse"] = sparse;
            None
        }
        Some(Err(e)) => Some(e),
        None => None,
    };
    (Some(report.to_string()), error)
}

/// Log through Python's `logging`, logger `itzma.maintenance`. `try_attach`
/// skips it during interpreter shutdown instead of blocking or panicking.
fn log_warning(format: &str, message: &str) {
    let _ = Python::try_attach(|py| -> PyResult<()> {
        py.import("logging")?
            .call_method1("getLogger", ("itzma.maintenance",))?
            .call_method1("warning", (format, message))?;
        Ok(())
    });
}

impl PyMaintainer {
    /// `stop`, called from process `now`. A forked child does nothing: the
    /// task belongs to the parent's runtime, whose threads the child does not
    /// have, so aborting it would touch a scheduler nobody runs.
    fn stop_in(&self, now: u32) {
        if now != self.pid {
            return;
        }
        if self.leased {
            // A stored permit, so a stop between two waits is not lost.
            self.stop.notify_one();
        } else {
            self.task.abort();
        }
    }
}

#[pymethods]
impl PyMaintainer {
    fn stop(&self) {
        self.stop_in(std::process::id());
    }

    #[getter]
    fn running(&self) -> bool {
        !self.task.is_finished()
    }

    /// `"leader"` while this loop holds the lease, else `"follower"`. A loop
    /// without a lease is always the leader.
    #[getter]
    fn role(&self) -> &'static str {
        if self.state.leader.load(Ordering::SeqCst) {
            "leader"
        } else {
            "follower"
        }
    }

    #[getter]
    fn runs(&self) -> u64 {
        self.state.runs.load(Ordering::SeqCst)
    }

    #[getter]
    fn errors(&self) -> u64 {
        self.state.errors.load(Ordering::SeqCst)
    }

    #[getter]
    fn last_error(&self) -> Option<String> {
        get(&self.state.last_error)
    }

    #[getter]
    fn last_report(&self) -> Option<String> {
        get(&self.state.last_report)
    }
}

#[cfg(test)]
mod tests {
    use itzma_core::{LeaseOptions, Metric, Schema};
    use object_store::ObjectStore;
    use object_store::local::LocalFileSystem;
    use object_store::memory::InMemory;
    use object_store::throttle::{ThrottleConfig, ThrottledStore};
    use serde_json::json;

    use super::*;

    #[test]
    fn a_failed_maintain_is_an_error_and_no_report() {
        assert_eq!(
            outcome(Err("boom".into()), None),
            (None, Some("boom".into()))
        );
    }

    #[test]
    fn index_reports_join_the_maintain_report() {
        let (report, error) = outcome(
            Ok(json!({"tail_objects": 1})),
            Some(Ok((json!({"indexed": 1}), json!({"indexed": 0})))),
        );
        assert_eq!(error, None);
        let report: serde_json::Value =
            serde_json::from_str(&report.expect("report")).expect("json");
        assert_eq!(report["index"]["indexed"], 1);
        assert_eq!(report["index_sparse"]["indexed"], 0);
    }

    #[test]
    fn a_stop_in_a_forked_child_leaves_the_task_alone() {
        let runtime = runtime::current().expect("runtime");
        let ns = runtime
            .block_on(Namespace::create(
                Arc::new(InMemory::new()),
                "docs",
                Schema::new(2, Metric::L2),
            ))
            .expect("create");
        let ns = Arc::new(ns);
        let plan = Plan {
            every: Duration::from_secs(3600),
            maintain: MaintenanceOptions::default(),
            index: None,
        };
        let maintainer = start(Arc::downgrade(&ns), plan, None).expect("start");
        maintainer.stop_in(std::process::id().wrapping_add(1));
        std::thread::sleep(Duration::from_millis(50));
        assert!(maintainer.running(), "a child's stop aborted the task");
        maintainer.stop();
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while maintainer.running() {
            assert!(
                std::time::Instant::now() < deadline,
                "the task kept running"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    const RENEW: Duration = Duration::from_millis(50);
    const TTL: Duration = Duration::from_millis(200);
    const EVERY: Duration = Duration::from_millis(100);

    fn create(store: &Arc<dyn ObjectStore>) {
        let runtime = runtime::current().expect("runtime");
        runtime
            .block_on(Namespace::create(
                store.clone(),
                "docs",
                Schema::new(2, Metric::L2),
            ))
            .expect("create");
    }

    /// A handle on `docs` with a leased loop: renew 50 ms, ttl 200 ms, every
    /// 100 ms, maintenance thresholds left at the core's defaults.
    fn leased(store: &Arc<dyn ObjectStore>) -> (Arc<Namespace>, PyMaintainer) {
        let runtime = runtime::current().expect("runtime");
        let ns = Arc::new(
            runtime
                .block_on(Namespace::open(store.clone(), "docs"))
                .expect("open"),
        );
        let lease = ns
            .maintenance_lease(LeaseOptions {
                renew: RENEW,
                ttl: TTL,
            })
            .expect("lease");
        let plan = Plan {
            every: EVERY,
            maintain: MaintenanceOptions::default(),
            index: None,
        };
        let maintainer = start(Arc::downgrade(&ns), plan, Some(lease)).expect("start");
        (ns, maintainer)
    }

    fn wait_for(what: &str, done: impl Fn() -> bool) {
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while !done() {
            assert!(std::time::Instant::now() < deadline, "timed out: {what}");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// Every listing takes 600 ms, three times the ttl, so a `maintain` run
    /// (which lists the WAL tail and the segments) outlasts the lease. The
    /// lease itself only gets and puts, so it stays quick.
    fn slow_lists() -> Arc<dyn ObjectStore> {
        Arc::new(ThrottledStore::new(
            InMemory::new(),
            ThrottleConfig {
                wait_list_per_call: Duration::from_millis(600),
                ..ThrottleConfig::default()
            },
        ))
    }

    #[test]
    fn a_leased_loop_waits_every_before_its_first_run() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        create(&store);
        let (_ns, m) = leased(&store);
        wait_for("the claim", || m.role() == "leader");
        assert_eq!(m.runs(), 0, "ran before every had passed");
        wait_for("a run", || m.runs() >= 1);
        m.stop();
    }

    #[test]
    fn two_leased_loops_have_one_leader() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        create(&store);
        let (_a, a) = leased(&store);
        let (_b, b) = leased(&store);
        std::thread::sleep(Duration::from_millis(300));
        let roles = [a.role(), b.role()];
        assert_eq!(
            roles.iter().filter(|r| **r == "leader").count(),
            1,
            "{roles:?}"
        );
        let (leader, follower) = if a.role() == "leader" {
            (&a, &b)
        } else {
            (&b, &a)
        };
        wait_for("the leader's run", || leader.runs() > 0);
        assert_eq!(follower.runs(), 0);
        a.stop();
        b.stop();
    }

    #[test]
    fn a_long_run_keeps_the_lease() {
        let store = slow_lists();
        create(&store);
        let (_a, a) = leased(&store);
        wait_for("the claim", || a.role() == "leader");
        std::thread::sleep(EVERY + RENEW);
        let (_b, b) = leased(&store);
        let until = std::time::Instant::now() + TTL * 4;
        while std::time::Instant::now() < until {
            assert_eq!(b.role(), "follower", "took over a busy leader");
            assert_eq!(a.role(), "leader");
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(a.runs(), 0, "the run was not long after all");
        a.stop();
        b.stop();
    }

    #[test]
    fn stop_during_a_run_releases_the_lease() {
        let store = slow_lists();
        create(&store);
        let (_a, a) = leased(&store);
        wait_for("the claim", || a.role() == "leader");
        std::thread::sleep(EVERY + RENEW);
        let (_b, b) = leased(&store);
        std::thread::sleep(RENEW * 2);
        assert_eq!(a.runs(), 0, "no run in flight");
        let stopped = std::time::Instant::now();
        a.stop();
        wait_for("the handover", || b.role() == "leader");
        assert!(
            stopped.elapsed() < TTL,
            "waited out the ttl instead of a release"
        );
        wait_for("the stopped loop to end", || !a.running());
        b.stop();
    }

    #[test]
    fn not_implemented_ends_the_loop() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store: Arc<dyn ObjectStore> =
            Arc::new(LocalFileSystem::new_with_prefix(dir.path()).expect("local"));
        create(&store);
        let (_ns, m) = leased(&store);
        wait_for("the loop to end", || !m.running());
        assert_eq!(m.errors(), 1);
        assert!(m.last_error().is_some());
    }

    #[test]
    fn a_failed_index_keeps_the_maintain_report() {
        let (report, error) = outcome(Ok(json!({"tail_objects": 1})), Some(Err("no index".into())));
        assert_eq!(error, Some("no index".into()));
        let report: serde_json::Value =
            serde_json::from_str(&report.expect("report")).expect("json");
        assert_eq!(report["tail_objects"], 1);
    }
}
