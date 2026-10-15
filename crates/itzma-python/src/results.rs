//! Query results, kept in Rust until Python asks for them.

use std::sync::Arc;

use itzma_core::{Hit, Row};
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3_arrow::PyTable;

use crate::convert::json;
use crate::convert::layout::ArrowLayout;
use crate::error;

fn table(batch: arrow_array::RecordBatch) -> PyResult<PyTable> {
    let schema = batch.schema();
    PyTable::try_new(vec![batch], schema)
}

#[pyclass(frozen, name = "Hits", module = "itzma._itzma")]
pub(crate) struct PyHits {
    pub(crate) hits: Vec<Hit>,
    pub(crate) layout: Arc<ArrowLayout>,
}

#[pymethods]
impl PyHits {
    fn __len__(&self) -> usize {
        self.hits.len()
    }

    fn ids(&self) -> Vec<String> {
        self.hits.iter().map(|h| h.id.clone()).collect()
    }

    fn distances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_vec(py, self.hits.iter().map(|h| h.distance).collect())
    }

    fn rows<'py>(&self, py: Python<'py>) -> PyResult<Vec<(String, f32, Bound<'py, PyAny>)>> {
        self.hits
            .iter()
            .map(|h| {
                let attrs = match &h.attrs {
                    Some(a) => json::to_py(py, a)?,
                    None => py.None().into_bound(py),
                };
                Ok((h.id.clone(), h.distance, attrs))
            })
            .collect()
    }

    fn to_arrow(&self) -> PyResult<PyTable> {
        table(self.layout.hits(&self.hits).map_err(error::itzma_error)?)
    }
}

#[pyclass(frozen, name = "Rows", module = "itzma._itzma")]
pub(crate) struct PyRows {
    pub(crate) rows: Vec<Row>,
    pub(crate) next: Option<String>,
    pub(crate) layout: Arc<ArrowLayout>,
}

#[pymethods]
impl PyRows {
    fn __len__(&self) -> usize {
        self.rows.len()
    }

    #[getter]
    fn next(&self) -> Option<String> {
        self.next.clone()
    }

    fn rows<'py>(&self, py: Python<'py>) -> PyResult<Vec<(String, Bound<'py, PyAny>)>> {
        self.rows
            .iter()
            .map(|r| {
                let attrs = match &r.attrs {
                    Some(a) => json::to_py(py, a)?,
                    None => py.None().into_bound(py),
                };
                Ok((r.id.clone(), attrs))
            })
            .collect()
    }

    fn to_arrow(&self) -> PyResult<PyTable> {
        table(self.layout.rows(&self.rows).map_err(error::itzma_error)?)
    }
}
