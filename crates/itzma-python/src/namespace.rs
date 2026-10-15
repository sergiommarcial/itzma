//! The `Namespace` class. Every operation has a blocking form and an
//! `_async` form that returns an awaitable; both run the same core future.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use itzma_core::{
    CacheOptions, CacheStore, DEFAULT_RAM_MAX_BYTES, FetchPolicy, Namespace, NvmeOptions, Schema,
};
use object_store::ObjectStore;
use pyo3::prelude::*;

use std::future::Future;
use std::time::Duration;

use itzma_core::{
    CompactionOptions, FilterOptions, GcOptions, IndexOptions, MaintenanceOptions, QueryOptions,
    SparseQueryOptions, StreamOptions, WriteKey,
};

use crate::convert::arrow::PyArrowStream;
use crate::convert::docs::PyBatch;
use crate::convert::layout::ArrowLayout;
use crate::convert::vectors;
use crate::results::{PyHits, PyRows};
use crate::store::PyStore;
use crate::write::{self, Op};
use crate::{error, maintainer, options, predicate, reports, runtime, schema};

#[pyclass(frozen, name = "Namespace", module = "itzma._itzma")]
pub(crate) struct PyNamespace {
    pub(crate) inner: Arc<Namespace>,
    pub(crate) layout: Arc<ArrowLayout>,
    pid: u32,
}

impl PyNamespace {
    fn new(inner: Namespace) -> Self {
        let layout = Arc::new(ArrowLayout::new(inner.schema()));
        Self {
            inner: Arc::new(inner),
            layout,
            pid: std::process::id(),
        }
    }

    /// The core handle, refused in a process other than the one that opened it.
    pub(crate) fn handle(&self) -> PyResult<Arc<Namespace>> {
        match runtime::fork_error(self.pid, std::process::id()) {
            Some(message) => Err(error::itzma_error(message)),
            None => Ok(self.inner.clone()),
        }
    }

    /// Run a maintenance-style call to completion; the report comes back as JSON.
    fn report<Fut>(
        &self,
        py: Python<'_>,
        call: impl FnOnce(Arc<Namespace>) -> Fut,
    ) -> PyResult<String>
    where
        Fut: Future<Output = PyResult<String>> + Send,
    {
        let ns = self.handle()?;
        runtime::block_on(py, call(ns))
    }

    /// The awaitable form of `report`.
    fn report_async<'py, Fut>(
        &self,
        py: Python<'py>,
        call: impl FnOnce(Arc<Namespace>) -> Fut,
    ) -> PyResult<Bound<'py, PyAny>>
    where
        Fut: Future<Output = PyResult<String>> + Send + 'static,
    {
        let ns = self.handle()?;
        runtime::spawn(py, call(ns))
    }
}

/// The cache tiers, or `None` when the caller asked for no cache.
pub(crate) fn cache_options(
    ram_bytes: Option<u64>,
    nvme_dir: Option<PathBuf>,
    nvme_bytes: Option<u64>,
) -> Result<Option<CacheOptions>, String> {
    let nvme = match (nvme_dir, nvme_bytes) {
        (Some(dir), Some(max_bytes)) => Some(NvmeOptions::new(dir, max_bytes)),
        (None, None) => None,
        _ => return Err("the NVMe cache needs both nvme_dir and nvme_bytes".into()),
    };
    if ram_bytes.is_none() && nvme.is_none() {
        return Ok(None);
    }
    let mut options = CacheOptions::default();
    options.ram_max_bytes = ram_bytes.unwrap_or(DEFAULT_RAM_MAX_BYTES);
    options.nvme = nvme;
    Ok(Some(options))
}

/// The fetch policy, or `None` when the caller set none of its
/// six fields — in which case `Namespace`'s own default (`object_store()`)
/// stands, so a caller that never heard of this setting is unaffected.
#[allow(clippy::too_many_arguments)]
fn fetch_policy_options(
    min_gap: Option<u64>,
    max_gap: Option<u64>,
    gap_divisor: Option<u64>,
    split_above: Option<u64>,
    part_size: Option<u64>,
    align: Option<u64>,
) -> Option<FetchPolicy> {
    let any = min_gap.is_some()
        || max_gap.is_some()
        || gap_divisor.is_some()
        || split_above.is_some()
        || part_size.is_some()
        || align.is_some();
    if !any {
        return None;
    }
    // FetchPolicy is #[non_exhaustive]: no struct literal (even with every
    // field named, even `..base`) works outside itzma-core, so a tweak of
    // one field over the default goes through its setters instead.
    let mut policy = FetchPolicy::object_store();
    if let Some(v) = min_gap {
        policy = policy.min_gap(v);
    }
    if let Some(v) = max_gap {
        policy = policy.max_gap(v);
    }
    if let Some(v) = gap_divisor {
        policy = policy.gap_divisor(v);
    }
    if let Some(v) = split_above {
        policy = policy.split_above(v);
    }
    if let Some(v) = part_size {
        policy = policy.part_size(v);
    }
    if let Some(v) = align {
        policy = policy.align(v);
    }
    Some(policy)
}

/// Everything `create` and `open` need, gathered while the GIL is held.
struct Opening {
    store: Arc<dyn ObjectStore>,
    id: String,
    schema: Option<Schema>,
    read_budget: Option<u64>,
    cache: Option<CacheOptions>,
    fetch: Option<FetchPolicy>,
}

impl Opening {
    async fn run(self) -> PyResult<PyNamespace> {
        let ns = match self.schema {
            Some(schema) => Namespace::create(self.store.clone(), &self.id, schema).await,
            None => Namespace::open(self.store.clone(), &self.id).await,
        }
        .map_err(error::to_py)?;
        let ns = match self.read_budget {
            Some(bytes) => ns.with_read_budget(bytes),
            None => ns,
        };
        let ns = match self.fetch {
            Some(policy) => ns.with_fetch_policy(policy),
            None => ns,
        };
        let ns = match self.cache {
            Some(options) => {
                let cached = CacheStore::new(self.store, options)
                    .map_err(|e| error::to_py(itzma_core::ItzmaError::Storage(e)))?;
                ns.with_cache(Arc::new(cached))
            }
            None => ns,
        };
        Ok(PyNamespace::new(ns))
    }
}

/// The per-handle settings `create` and `open` share.
struct Settings {
    read_budget: Option<u64>,
    cache_ram_bytes: Option<u64>,
    cache_nvme_dir: Option<PathBuf>,
    cache_nvme_bytes: Option<u64>,
    fetch_min_gap: Option<u64>,
    fetch_max_gap: Option<u64>,
    fetch_gap_divisor: Option<u64>,
    fetch_split_above: Option<u64>,
    fetch_part_size: Option<u64>,
    fetch_align: Option<u64>,
}

fn opening(
    store: &PyStore,
    id: String,
    schema: Option<Schema>,
    settings: Settings,
) -> PyResult<Opening> {
    let cache = cache_options(
        settings.cache_ram_bytes,
        settings.cache_nvme_dir,
        settings.cache_nvme_bytes,
    )
    .map_err(error::input)?;
    let fetch = fetch_policy_options(
        settings.fetch_min_gap,
        settings.fetch_max_gap,
        settings.fetch_gap_divisor,
        settings.fetch_split_above,
        settings.fetch_part_size,
        settings.fetch_align,
    );
    Ok(Opening {
        store: store.handle()?,
        id,
        schema,
        read_budget: settings.read_budget,
        cache,
        fetch,
    })
}

/// An upsert gathered while the GIL is held, run after it is released:
/// documents from rows or numpy go through the split loop, an Arrow stream
/// through `upsert_stream`.
struct Upsert {
    ns: Arc<Namespace>,
    input: UpsertInput,
    key: Option<WriteKey>,
    options: StreamOptions,
}

enum UpsertInput {
    Documents(Vec<itzma_core::Document>),
    Arrow(Box<dyn arrow_array::RecordBatchReader + Send>),
}

impl Upsert {
    async fn run(self) -> PyResult<u64> {
        match self.input {
            UpsertInput::Documents(docs) => write::write(self.ns, Op::Upsert(docs), self.key).await,
            UpsertInput::Arrow(reader) => {
                write::write_stream(self.ns, reader, self.key, self.options).await
            }
        }
    }
}

impl PyNamespace {
    fn upsert_of(
        &self,
        batch: &Bound<'_, PyAny>,
        key: Option<String>,
        retry_from: Option<u64>,
        target_bytes: Option<usize>,
        depth: Option<usize>,
    ) -> PyResult<Upsert> {
        let input = if let Ok(batch) = batch.cast::<PyBatch>() {
            UpsertInput::Documents(batch.get().take()?)
        } else if let Ok(stream) = batch.cast::<PyArrowStream>() {
            UpsertInput::Arrow(stream.get().take()?)
        } else {
            return Err(error::input(
                "upsert takes a Batch or an ArrowStream".into(),
            ));
        };
        let mut options = StreamOptions::default();
        if let Some(bytes) = target_bytes {
            options = options.target_bytes(bytes);
        }
        if let Some(depth) = depth {
            options = options.depth(depth);
        }
        // Checked for every input, so a bad value is never silently ignored
        // on a write that happens not to stream.
        options.validate().map_err(error::to_py)?;
        Ok(Upsert {
            ns: self.handle()?,
            input,
            key: write::write_key(key, retry_from)?,
            options,
        })
    }
}

#[pymethods]
impl PyNamespace {
    #[staticmethod]
    #[pyo3(signature = (store, id, dim, metric, attrs, *, read_budget=None,
                        cache_ram_bytes=None, cache_nvme_dir=None, cache_nvme_bytes=None,
                        fetch_min_gap=None, fetch_max_gap=None, fetch_gap_divisor=None,
                        fetch_split_above=None, fetch_part_size=None, fetch_align=None))]
    #[allow(clippy::too_many_arguments)]
    fn create(
        py: Python<'_>,
        store: &PyStore,
        id: String,
        dim: usize,
        metric: &str,
        attrs: BTreeMap<String, String>,
        read_budget: Option<u64>,
        cache_ram_bytes: Option<u64>,
        cache_nvme_dir: Option<PathBuf>,
        cache_nvme_bytes: Option<u64>,
        fetch_min_gap: Option<u64>,
        fetch_max_gap: Option<u64>,
        fetch_gap_divisor: Option<u64>,
        fetch_split_above: Option<u64>,
        fetch_part_size: Option<u64>,
        fetch_align: Option<u64>,
    ) -> PyResult<Self> {
        let schema = schema::from_args(dim, metric, &attrs).map_err(error::input)?;
        let settings = Settings {
            read_budget,
            cache_ram_bytes,
            cache_nvme_dir,
            cache_nvme_bytes,
            fetch_min_gap,
            fetch_max_gap,
            fetch_gap_divisor,
            fetch_split_above,
            fetch_part_size,
            fetch_align,
        };
        let open = opening(store, id, Some(schema), settings)?;
        runtime::block_on(py, open.run())
    }

    #[staticmethod]
    #[pyo3(signature = (store, id, dim, metric, attrs, *, read_budget=None,
                        cache_ram_bytes=None, cache_nvme_dir=None, cache_nvme_bytes=None,
                        fetch_min_gap=None, fetch_max_gap=None, fetch_gap_divisor=None,
                        fetch_split_above=None, fetch_part_size=None, fetch_align=None))]
    #[allow(clippy::too_many_arguments)]
    fn create_async<'py>(
        py: Python<'py>,
        store: &PyStore,
        id: String,
        dim: usize,
        metric: &str,
        attrs: BTreeMap<String, String>,
        read_budget: Option<u64>,
        cache_ram_bytes: Option<u64>,
        cache_nvme_dir: Option<PathBuf>,
        cache_nvme_bytes: Option<u64>,
        fetch_min_gap: Option<u64>,
        fetch_max_gap: Option<u64>,
        fetch_gap_divisor: Option<u64>,
        fetch_split_above: Option<u64>,
        fetch_part_size: Option<u64>,
        fetch_align: Option<u64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let schema = schema::from_args(dim, metric, &attrs).map_err(error::input)?;
        let settings = Settings {
            read_budget,
            cache_ram_bytes,
            cache_nvme_dir,
            cache_nvme_bytes,
            fetch_min_gap,
            fetch_max_gap,
            fetch_gap_divisor,
            fetch_split_above,
            fetch_part_size,
            fetch_align,
        };
        let open = opening(store, id, Some(schema), settings)?;
        runtime::spawn(py, open.run())
    }

    #[staticmethod]
    #[pyo3(signature = (store, id, *, read_budget=None,
                        cache_ram_bytes=None, cache_nvme_dir=None, cache_nvme_bytes=None,
                        fetch_min_gap=None, fetch_max_gap=None, fetch_gap_divisor=None,
                        fetch_split_above=None, fetch_part_size=None, fetch_align=None))]
    #[allow(clippy::too_many_arguments)]
    fn open(
        py: Python<'_>,
        store: &PyStore,
        id: String,
        read_budget: Option<u64>,
        cache_ram_bytes: Option<u64>,
        cache_nvme_dir: Option<PathBuf>,
        cache_nvme_bytes: Option<u64>,
        fetch_min_gap: Option<u64>,
        fetch_max_gap: Option<u64>,
        fetch_gap_divisor: Option<u64>,
        fetch_split_above: Option<u64>,
        fetch_part_size: Option<u64>,
        fetch_align: Option<u64>,
    ) -> PyResult<Self> {
        let settings = Settings {
            read_budget,
            cache_ram_bytes,
            cache_nvme_dir,
            cache_nvme_bytes,
            fetch_min_gap,
            fetch_max_gap,
            fetch_gap_divisor,
            fetch_split_above,
            fetch_part_size,
            fetch_align,
        };
        let open = opening(store, id, None, settings)?;
        runtime::block_on(py, open.run())
    }

    #[staticmethod]
    #[pyo3(signature = (store, id, *, read_budget=None,
                        cache_ram_bytes=None, cache_nvme_dir=None, cache_nvme_bytes=None,
                        fetch_min_gap=None, fetch_max_gap=None, fetch_gap_divisor=None,
                        fetch_split_above=None, fetch_part_size=None, fetch_align=None))]
    #[allow(clippy::too_many_arguments)]
    fn open_async<'py>(
        py: Python<'py>,
        store: &PyStore,
        id: String,
        read_budget: Option<u64>,
        cache_ram_bytes: Option<u64>,
        cache_nvme_dir: Option<PathBuf>,
        cache_nvme_bytes: Option<u64>,
        fetch_min_gap: Option<u64>,
        fetch_max_gap: Option<u64>,
        fetch_gap_divisor: Option<u64>,
        fetch_split_above: Option<u64>,
        fetch_part_size: Option<u64>,
        fetch_align: Option<u64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let settings = Settings {
            read_budget,
            cache_ram_bytes,
            cache_nvme_dir,
            cache_nvme_bytes,
            fetch_min_gap,
            fetch_max_gap,
            fetch_gap_divisor,
            fetch_split_above,
            fetch_part_size,
            fetch_align,
        };
        let open = opening(store, id, None, settings)?;
        runtime::spawn(py, open.run())
    }

    #[getter]
    fn id(&self) -> String {
        self.inner.id().to_string()
    }

    fn schema_json(&self) -> PyResult<String> {
        Ok(schema::to_json(self.handle()?.schema()))
    }

    #[pyo3(signature = (batch, *, key=None, retry_from=None, target_bytes=None, depth=None))]
    fn upsert(
        &self,
        py: Python<'_>,
        batch: &Bound<'_, PyAny>,
        key: Option<String>,
        retry_from: Option<u64>,
        target_bytes: Option<usize>,
        depth: Option<usize>,
    ) -> PyResult<u64> {
        let upsert = self.upsert_of(batch, key, retry_from, target_bytes, depth)?;
        runtime::block_on(py, upsert.run())
    }

    #[pyo3(signature = (batch, *, key=None, retry_from=None, target_bytes=None, depth=None))]
    fn upsert_async<'py>(
        &self,
        py: Python<'py>,
        batch: &Bound<'py, PyAny>,
        key: Option<String>,
        retry_from: Option<u64>,
        target_bytes: Option<usize>,
        depth: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let upsert = self.upsert_of(batch, key, retry_from, target_bytes, depth)?;
        runtime::spawn(py, upsert.run())
    }

    #[pyo3(signature = (ids, *, key=None, retry_from=None))]
    fn delete(
        &self,
        py: Python<'_>,
        ids: Vec<String>,
        key: Option<String>,
        retry_from: Option<u64>,
    ) -> PyResult<u64> {
        let (ns, key) = (self.handle()?, write::write_key(key, retry_from)?);
        runtime::block_on(py, write::write(ns, Op::Delete(ids), key))
    }

    #[pyo3(signature = (ids, *, key=None, retry_from=None))]
    fn delete_async<'py>(
        &self,
        py: Python<'py>,
        ids: Vec<String>,
        key: Option<String>,
        retry_from: Option<u64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (ns, key) = (self.handle()?, write::write_key(key, retry_from)?);
        runtime::spawn(py, write::write(ns, Op::Delete(ids), key))
    }

    fn query(
        &self,
        py: Python<'_>,
        vector: &Bound<'_, PyAny>,
        top_k: usize,
        options: &str,
    ) -> PyResult<PyHits> {
        let (ns, vector) = (self.handle()?, vectors::vector(vector)?);
        let options: QueryOptions = options::merge(options).map_err(error::input)?;
        runtime::block_on(
            py,
            query_call(ns, vector, top_k, options, self.layout.clone()),
        )
    }

    fn query_async<'py>(
        &self,
        py: Python<'py>,
        vector: &Bound<'_, PyAny>,
        top_k: usize,
        options: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (ns, vector) = (self.handle()?, vectors::vector(vector)?);
        let options: QueryOptions = options::merge(options).map_err(error::input)?;
        runtime::spawn(
            py,
            query_call(ns, vector, top_k, options, self.layout.clone()),
        )
    }

    fn query_sparse(
        &self,
        py: Python<'_>,
        terms: &Bound<'_, PyAny>,
        top_k: usize,
        options: &str,
    ) -> PyResult<PyHits> {
        let (ns, terms) = (self.handle()?, vectors::sparse(terms)?);
        let options: SparseQueryOptions = options::merge(options).map_err(error::input)?;
        runtime::block_on(
            py,
            query_sparse_call(ns, terms, top_k, options, self.layout.clone()),
        )
    }

    fn query_sparse_async<'py>(
        &self,
        py: Python<'py>,
        terms: &Bound<'_, PyAny>,
        top_k: usize,
        options: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (ns, terms) = (self.handle()?, vectors::sparse(terms)?);
        let options: SparseQueryOptions = options::merge(options).map_err(error::input)?;
        runtime::spawn(
            py,
            query_sparse_call(ns, terms, top_k, options, self.layout.clone()),
        )
    }

    fn filter(&self, py: Python<'_>, predicate: &str, options: &str) -> PyResult<PyRows> {
        let ns = self.handle()?;
        let predicate = predicate::parse(predicate).map_err(error::input)?;
        let options: FilterOptions = options::merge(options).map_err(error::input)?;
        runtime::block_on(py, filter_call(ns, predicate, options, self.layout.clone()))
    }

    fn filter_async<'py>(
        &self,
        py: Python<'py>,
        predicate: &str,
        options: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let ns = self.handle()?;
        let predicate = predicate::parse(predicate).map_err(error::input)?;
        let options: FilterOptions = options::merge(options).map_err(error::input)?;
        runtime::spawn(py, filter_call(ns, predicate, options, self.layout.clone()))
    }

    fn index(&self, py: Python<'_>, options: &str) -> PyResult<String> {
        let o: IndexOptions = options::merge(options).map_err(error::input)?;
        self.report(py, |ns| index_call(ns, o))
    }

    fn index_async<'py>(&self, py: Python<'py>, options: &str) -> PyResult<Bound<'py, PyAny>> {
        let o: IndexOptions = options::merge(options).map_err(error::input)?;
        self.report_async(py, |ns| index_call(ns, o))
    }

    fn compact(&self, py: Python<'_>, options: &str) -> PyResult<String> {
        let o = options::compaction(options).map_err(error::input)?;
        self.report(py, |ns| compact_call(ns, o))
    }

    fn compact_async<'py>(&self, py: Python<'py>, options: &str) -> PyResult<Bound<'py, PyAny>> {
        let o = options::compaction(options).map_err(error::input)?;
        self.report_async(py, |ns| compact_call(ns, o))
    }

    fn gc(&self, py: Python<'_>, options: &str) -> PyResult<String> {
        let o = options::gc(options).map_err(error::input)?;
        self.report(py, |ns| gc_call(ns, o))
    }

    fn gc_async<'py>(&self, py: Python<'py>, options: &str) -> PyResult<Bound<'py, PyAny>> {
        let o = options::gc(options).map_err(error::input)?;
        self.report_async(py, |ns| gc_call(ns, o))
    }

    fn maintain(&self, py: Python<'_>, options: &str) -> PyResult<String> {
        let o = options::maintenance(options).map_err(error::input)?;
        self.report(py, |ns| maintain_call(ns, o))
    }

    fn maintain_async<'py>(&self, py: Python<'py>, options: &str) -> PyResult<Bound<'py, PyAny>> {
        let o = options::maintenance(options).map_err(error::input)?;
        self.report_async(py, |ns| maintain_call(ns, o))
    }

    #[pyo3(signature = (every_secs, maintain, index=None))]
    fn start_maintenance(
        &self,
        every_secs: f64,
        maintain: &str,
        index: Option<&str>,
    ) -> PyResult<maintainer::PyMaintainer> {
        self.handle()?;
        let every = Duration::try_from_secs_f64(every_secs)
            .ok()
            .filter(|d| !d.is_zero())
            .ok_or_else(|| error::input("every must be more than zero seconds".into()))?;
        let maintain = options::maintenance(maintain).map_err(error::input)?;
        let index = index
            .map(|i| options::merge::<IndexOptions>(i).map_err(error::input))
            .transpose()?;
        let plan = maintainer::Plan {
            every,
            maintain,
            index,
        };
        maintainer::start(Arc::downgrade(&self.inner), plan).map_err(error::itzma_error)
    }
}

async fn index_call(ns: Arc<Namespace>, o: IndexOptions) -> PyResult<String> {
    let r = ns.index(&o).await.map_err(error::to_py)?;
    Ok(serde_json::to_string(&r).unwrap_or_default())
}

async fn compact_call(ns: Arc<Namespace>, o: CompactionOptions) -> PyResult<String> {
    let r = ns.compact(&o).await.map_err(error::to_py)?;
    Ok(reports::compaction(&r).to_string())
}

async fn gc_call(ns: Arc<Namespace>, o: GcOptions) -> PyResult<String> {
    let r = ns.gc(&o).await.map_err(error::to_py)?;
    Ok(reports::gc(&r).to_string())
}

async fn maintain_call(ns: Arc<Namespace>, o: MaintenanceOptions) -> PyResult<String> {
    let r = ns.maintain(&o).await.map_err(error::to_py)?;
    Ok(reports::maintenance(&r).to_string())
}

async fn query_call(
    ns: Arc<Namespace>,
    vector: Vec<f32>,
    top_k: usize,
    options: QueryOptions,
    layout: Arc<ArrowLayout>,
) -> PyResult<PyHits> {
    let hits = ns
        .query_with(&vector, top_k, &options)
        .await
        .map_err(error::to_py)?;
    Ok(PyHits { hits, layout })
}

async fn query_sparse_call(
    ns: Arc<Namespace>,
    terms: Vec<(u32, f32)>,
    top_k: usize,
    options: SparseQueryOptions,
    layout: Arc<ArrowLayout>,
) -> PyResult<PyHits> {
    let hits = ns
        .query_sparse(&terms, top_k, &options)
        .await
        .map_err(error::to_py)?;
    Ok(PyHits { hits, layout })
}

async fn filter_call(
    ns: Arc<Namespace>,
    predicate: itzma_core::Predicate,
    options: FilterOptions,
    layout: Arc<ArrowLayout>,
) -> PyResult<PyRows> {
    let page = ns
        .filter(&predicate, &options)
        .await
        .map_err(error::to_py)?;
    Ok(PyRows {
        rows: page.rows,
        next: page.next,
        layout,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_options_cases() {
        assert!(cache_options(None, None, None).expect("none").is_none());
        let ram = cache_options(Some(10), None, None)
            .expect("ram")
            .expect("some");
        assert_eq!(ram.ram_max_bytes, 10);
        assert!(ram.nvme.is_none());
        let nvme = cache_options(None, Some("/x".into()), Some(5))
            .expect("nvme")
            .expect("some");
        assert_eq!(nvme.ram_max_bytes, DEFAULT_RAM_MAX_BYTES);
        assert_eq!(nvme.nvme.expect("nvme").max_bytes, 5);
        assert!(cache_options(None, Some("/x".into()), None).is_err());
        assert!(cache_options(None, None, Some(5)).is_err());
    }
}
