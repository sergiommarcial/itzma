//! A `Batch`: documents converted while the GIL is held, written later.

use std::sync::Mutex;

use itzma_core::Document;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::convert::{json, vectors};
use crate::error;

#[pyclass(frozen, name = "Batch", module = "itzma._itzma")]
pub(crate) struct PyBatch {
    docs: Mutex<Option<Vec<Document>>>,
    len: usize,
}

impl PyBatch {
    pub(crate) fn from_docs(docs: Vec<Document>) -> Self {
        let len = docs.len();
        Self {
            docs: Mutex::new(Some(docs)),
            len,
        }
    }

    /// The documents, once. A batch is consumed by the write that takes it.
    pub(crate) fn take(&self) -> PyResult<Vec<Document>> {
        let mut docs = self
            .docs
            .lock()
            .map_err(|_| error::itzma_error("batch lock poisoned".into()))?;
        docs.take()
            .ok_or_else(|| error::itzma_error("this batch was already written".into()))
    }
}

#[pymethods]
impl PyBatch {
    fn __len__(&self) -> usize {
        self.len
    }
}

const ROW_KEYS: [&str; 3] = ["id", "vector", "attrs"];

fn optional_attrs(value: Option<Bound<'_, PyAny>>) -> PyResult<Option<serde_json::Value>> {
    match value {
        Some(v) if !v.is_none() => Ok(Some(json::to_json(&v)?)),
        _ => Ok(None),
    }
}

/// Documents from a list of dicts with `id`, `vector` and optional `attrs`.
#[pyfunction]
pub(crate) fn batch_from_rows(rows: &Bound<'_, PyList>) -> PyResult<PyBatch> {
    let mut docs = Vec::with_capacity(rows.len());
    for (i, row) in rows.iter().enumerate() {
        let row = row
            .cast::<PyDict>()
            .map_err(|_| error::input(format!("row {i}: expected a dict with id and vector")))?;
        for key in row.keys() {
            let key: String = key.extract()?;
            if !ROW_KEYS.contains(&key.as_str()) {
                return Err(error::input(format!(
                    "row {i}: unknown key {key:?}; a row has id, vector and attrs"
                )));
            }
        }
        let id: String = row
            .get_item("id")?
            .ok_or_else(|| error::input(format!("row {i}: no id")))?
            .extract()
            .map_err(|_| error::input(format!("row {i}: id must be a string")))?;
        let vector = vectors::vector(
            &row.get_item("vector")?
                .ok_or_else(|| error::input(format!("row {i}: no vector")))?,
        )?;
        let attrs = optional_attrs(row.get_item("attrs")?)?;
        docs.push(Document { id, vector, attrs });
    }
    Ok(PyBatch::from_docs(docs))
}

/// Documents from columns: ids, a 2-D float32 matrix, and an optional list
/// of attribute dicts of the same length.
#[pyfunction]
#[pyo3(signature = (ids, vectors, attrs=None))]
pub(crate) fn batch_from_columns(
    ids: Vec<String>,
    vectors: &Bound<'_, PyAny>,
    attrs: Option<&Bound<'_, PyList>>,
) -> PyResult<PyBatch> {
    let rows = vectors::matrix(vectors)?;
    let n = ids.len();
    let same = |what: &str, len: usize| {
        if len == n {
            Ok(())
        } else {
            Err(error::input(format!("{n} ids but {len} {what}")))
        }
    };
    same("vectors", rows.len())?;
    if let Some(a) = attrs {
        same("attrs", a.len())?;
    }
    let mut docs = Vec::with_capacity(n);
    for (i, (id, vector)) in ids.into_iter().zip(rows).enumerate() {
        let doc_attrs = attrs.map(|a| a.get_item(i)).transpose()?;
        docs.push(Document {
            id,
            vector,
            attrs: optional_attrs(doc_attrs)?,
        });
    }
    Ok(PyBatch::from_docs(docs))
}
