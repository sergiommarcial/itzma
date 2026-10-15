//! An optional background loop that keeps a namespace compacted and indexed.
//!
//! The library starts no task on its own; this one runs only when
//! Python asks. It holds a weak reference, so it ends when the handle does.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use itzma_core::{IndexOptions, MaintenanceOptions, Namespace};
use pyo3::prelude::*;
use serde_json::Value;
use tokio::task::AbortHandle;
use tokio::time::MissedTickBehavior;

use crate::{reports, runtime};

pub(crate) struct Plan {
    pub(crate) every: Duration,
    pub(crate) maintain: MaintenanceOptions,
    pub(crate) index: Option<IndexOptions>,
}

#[derive(Default)]
struct State {
    runs: AtomicU64,
    errors: AtomicU64,
    last_error: Mutex<Option<String>>,
    last_report: Mutex<Option<String>>,
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
    /// The process whose runtime runs the task.
    pid: u32,
}

pub(crate) fn start(ns: Weak<Namespace>, plan: Plan) -> Result<PyMaintainer, String> {
    let runtime = runtime::current()?;
    let state = Arc::new(State::default());
    let task = runtime.spawn(run(ns, plan, state.clone())).abort_handle();
    Ok(PyMaintainer {
        state,
        task,
        pid: std::process::id(),
    })
}

async fn run(handle: Weak<Namespace>, plan: Plan, state: Arc<State>) {
    let mut ticks = tokio::time::interval(plan.every);
    ticks.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        ticks.tick().await;
        let Some(ns) = handle.upgrade() else { break };
        let (report, error) = tick(&ns, &plan).await;
        if let Some(report) = report {
            set(&state.last_report, report);
        }
        if let Some(message) = error {
            log_warning(&message);
            state.errors.fetch_add(1, Ordering::SeqCst);
            set(&state.last_error, message);
        }
        state.runs.fetch_add(1, Ordering::SeqCst);
    }
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
        (Ok(_), Some(index)) => Some(
            ns.index(index)
                .await
                .map(|r| serde_json::to_value(r).unwrap_or_default())
                .map_err(|e| e.to_string()),
        ),
        _ => None,
    };
    outcome(maintained, indexed)
}

fn outcome(
    maintained: Result<Value, String>,
    indexed: Option<Result<Value, String>>,
) -> (Option<String>, Option<String>) {
    let mut report = match maintained {
        Ok(report) => report,
        Err(e) => return (None, Some(e)),
    };
    let error = match indexed {
        Some(Ok(index)) => {
            report["index"] = index;
            None
        }
        Some(Err(e)) => Some(e),
        None => None,
    };
    (Some(report.to_string()), error)
}

/// Log through Python's `logging`, logger `itzma.maintenance`. `try_attach`
/// skips it during interpreter shutdown instead of blocking or panicking.
fn log_warning(message: &str) {
    let _ = Python::try_attach(|py| -> PyResult<()> {
        py.import("logging")?
            .call_method1("getLogger", ("itzma.maintenance",))?
            .call_method1("warning", ("maintenance run failed: %s", message))?;
        Ok(())
    });
}

impl PyMaintainer {
    /// `stop`, called from process `now`. A forked child does nothing: the
    /// task belongs to the parent's runtime, whose threads the child does not
    /// have, so aborting it would touch a scheduler nobody runs.
    fn stop_in(&self, now: u32) {
        if now == self.pid {
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
    fn the_index_report_joins_the_maintain_report() {
        let (report, error) = outcome(
            Ok(json!({"tail_objects": 1})),
            Some(Ok(json!({"indexed": 1}))),
        );
        assert_eq!(error, None);
        let report: serde_json::Value =
            serde_json::from_str(&report.expect("report")).expect("json");
        assert_eq!(report["index"]["indexed"], 1);
    }

    #[test]
    fn a_stop_in_a_forked_child_leaves_the_task_alone() {
        use itzma_core::{Metric, Schema};
        use object_store::memory::InMemory;

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
        let maintainer = start(Arc::downgrade(&ns), plan).expect("start");
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

    #[test]
    fn a_failed_index_keeps_the_maintain_report() {
        let (report, error) = outcome(Ok(json!({"tail_objects": 1})), Some(Err("no index".into())));
        assert_eq!(error, Some("no index".into()));
        let report: serde_json::Value =
            serde_json::from_str(&report.expect("report")).expect("json");
        assert_eq!(report["tail_objects"], 1);
    }
}
