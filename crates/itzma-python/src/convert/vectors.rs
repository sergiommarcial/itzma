//! Vectors from lists, numpy arrays and Arrow arrays.

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::error;

/// One dense vector: a float32 numpy array (any stride) or a list of
/// numbers. The Python side turns other numpy dtypes into float32 first.
pub(crate) fn vector(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    if let Ok(array) = obj.extract::<PyReadonlyArray1<'_, f32>>() {
        return Ok(array.as_array().to_vec());
    }
    if let Ok(values) = obj.extract::<Vec<f32>>() {
        return Ok(values);
    }
    if obj.hasattr("__arrow_c_array__")? {
        let array: pyo3_arrow::PyArray = obj.extract()?;
        return crate::convert::arrow::float32_values(array.array()).map_err(error::input);
    }
    Err(error::input(
        "a vector must be a list of numbers, a 1-D float32 numpy array or an Arrow float array"
            .into(),
    ))
}

/// Rows of a 2-D float32 numpy array (any stride).
pub(crate) fn matrix(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f32>>> {
    let array = obj
        .extract::<PyReadonlyArray2<'_, f32>>()
        .map_err(|_| error::input("vectors must be a 2-D float32 numpy array".into()))?;
    Ok(array
        .as_array()
        .rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect())
}
