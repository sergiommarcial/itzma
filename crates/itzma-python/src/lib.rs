//! Python bindings for itzma-core, built by maturin as `itzma._itzma`.
//!
//! This layer only converts values, releases the GIL and runs futures on one
//! tokio runtime per process. The Python package in
//! `py-itzma/src/itzma` owns the API: names, defaults, docs and types.
#![cfg_attr(not(test), deny(clippy::unwrap_used))]

mod convert;
mod error;
mod maintainer;
mod namespace;
mod options;
mod predicate;
mod reports;
mod results;
mod runtime;
mod schema;
mod store;
mod write;

use pyo3::prelude::*;

/// Set the runtime's worker thread count, and the compute pools' thread
/// counts. Each only before it starts: the runtime at the first
/// Itzma call, the pools at the first piece of work big enough to need them.
/// Every argument is checked before any is applied.
#[pyfunction]
#[pyo3(signature = (*, worker_threads=None, cpu_threads=None, maintenance_cpu_threads=None))]
fn configure(
    worker_threads: Option<usize>,
    cpu_threads: Option<usize>,
    maintenance_cpu_threads: Option<usize>,
) -> PyResult<()> {
    configure_all(worker_threads, cpu_threads, maintenance_cpu_threads).map_err(|r| match r {
        Refusal::Runtime(message) => error::itzma_error(message),
        Refusal::Core(e) => error::to_py(e),
    })
}

/// Why `configure` refused.
#[derive(Debug)]
enum Refusal {
    Runtime(String),
    Core(itzma_core::ItzmaError),
}

/// `configure` without Python. Zero counts are refused before anything is
/// applied. The runtime goes first: the pools only start after it does, so
/// if it is still configurable the pools are too, and a refusal never leaves
/// half a call applied.
fn configure_all(
    worker_threads: Option<usize>,
    cpu_threads: Option<usize>,
    maintenance_cpu_threads: Option<usize>,
) -> Result<(), Refusal> {
    if worker_threads == Some(0) {
        return Err(Refusal::Runtime("worker_threads must be at least 1".into()));
    }
    if cpu_threads == Some(0) || maintenance_cpu_threads == Some(0) {
        return Err(Refusal::Core(itzma_core::ItzmaError::InvalidConfig(
            "each compute lane needs at least one thread",
        )));
    }
    if let Some(workers) = worker_threads {
        runtime::configure(workers).map_err(Refusal::Runtime)?;
    }
    if cpu_threads.is_some() || maintenance_cpu_threads.is_some() {
        let mut threads = itzma_core::CpuThreads::default();
        if let Some(n) = cpu_threads {
            threads = threads.query(n);
        }
        if let Some(n) = maintenance_cpu_threads {
            threads = threads.maintenance(n);
        }
        itzma_core::configure_cpu(threads).map_err(Refusal::Core)?;
    }
    Ok(())
}

/// How long the exit hook waits for bridge tasks to leave Python.
const EXIT_WAIT: std::time::Duration = std::time::Duration::from_secs(5);

/// Registered with `atexit`, which runs before the interpreter finalizes:
/// waits, with the GIL released, for tasks still setting asyncio results
/// (see `runtime`).
#[pyfunction]
fn drain_at_exit(py: Python<'_>) {
    py.detach(|| runtime::drain(EXIT_WAIT));
}

#[pymodule]
fn _itzma(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    let hook = wrap_pyfunction!(drain_at_exit, m)?;
    m.py().import("atexit")?.call_method1("register", (hook,))?;
    m.add_function(wrap_pyfunction!(configure, m)?)?;
    m.add_function(wrap_pyfunction!(error::exception_names, m)?)?;
    m.add_function(wrap_pyfunction!(options::default_options, m)?)?;
    m.add_class::<store::PyStore>()?;
    m.add_class::<namespace::PyNamespace>()?;
    m.add_class::<convert::docs::PyBatch>()?;
    m.add_class::<results::PyHits>()?;
    m.add_class::<results::PyRows>()?;
    m.add_class::<maintainer::PyMaintainer>()?;
    m.add_function(wrap_pyfunction!(convert::docs::batch_from_rows, m)?)?;
    m.add_function(wrap_pyfunction!(convert::docs::batch_from_columns, m)?)?;
    m.add_function(wrap_pyfunction!(convert::arrow::stream_from_arrow, m)?)?;
    m.add_class::<convert::arrow::PyArrowStream>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_refused_argument_applies_nothing() {
        let before = runtime::configured_workers();
        let err = configure_all(Some(before + 7), Some(0), None).expect_err("zero cpu threads");
        assert!(matches!(err, Refusal::Core(_)), "{err:?}");
        assert_eq!(
            runtime::configured_workers(),
            before,
            "worker_threads was applied"
        );
        let err = configure_all(Some(0), Some(2), None).expect_err("zero workers");
        assert!(matches!(err, Refusal::Runtime(_)), "{err:?}");
    }
}
