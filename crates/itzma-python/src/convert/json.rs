//! Python values to `serde_json::Value` and back, for attributes.

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyFrozenSet, PyInt, PyList, PySet, PyString, PyTuple};
use serde_json::{Map, Number, Value};

use crate::error;

/// A Python value as JSON. numpy scalars (`np.int64`, `np.float32`,
/// `np.bool_`) and arrays are turned into plain values with `.tolist()` first.
pub(crate) fn to_json(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(b) = obj.cast::<PyBool>() {
        return Ok(Value::Bool(b.is_true()));
    }
    if obj.is_instance_of::<PyInt>() {
        if let Ok(i) = obj.extract::<i64>() {
            return Ok(Value::from(i));
        }
        if let Ok(u) = obj.extract::<u64>() {
            return Ok(Value::from(u));
        }
        return finite(obj.extract::<f64>()?);
    }
    if obj.is_instance_of::<PyFloat>() {
        return finite(obj.extract::<f64>()?);
    }
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(Value::String(s.to_str()?.to_owned()));
    }
    if let Ok(d) = obj.cast::<PyDict>() {
        let mut map = Map::with_capacity(d.len());
        for (k, v) in d.iter() {
            let key = k
                .cast::<PyString>()
                .map_err(|_| error::input("attribute names must be strings".into()))?
                .to_str()?
                .to_owned();
            map.insert(key, to_json(&v)?);
        }
        return Ok(Value::Object(map));
    }
    if obj.is_instance_of::<PyList>()
        || obj.is_instance_of::<PyTuple>()
        || obj.is_instance_of::<PySet>()
        || obj.is_instance_of::<PyFrozenSet>()
    {
        let mut items = Vec::new();
        for item in obj.try_iter()? {
            items.push(to_json(&item?)?);
        }
        return Ok(Value::Array(items));
    }
    // numpy scalars and arrays: `tolist` gives plain Python values for both.
    if obj.hasattr("dtype")? && obj.hasattr("tolist")? {
        return to_json(&obj.call_method0("tolist")?);
    }
    Err(error::input(format!(
        "cannot store a {} as an attribute value",
        obj.get_type().name()?
    )))
}

fn finite(v: f64) -> PyResult<Value> {
    Number::from_f64(v)
        .map(Value::Number)
        .ok_or_else(|| error::input(format!("attribute numbers must be finite, got {v}")))
}

/// JSON back to plain Python values.
pub(crate) fn to_py<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match value {
        Value::Null => py.None().into_bound(py),
        Value::Bool(b) => b.into_pyobject(py)?.to_owned().into_any(),
        Value::Number(n) => match (n.as_i64(), n.as_u64()) {
            (Some(i), _) => i.into_pyobject(py)?.into_any(),
            (None, Some(u)) => u.into_pyobject(py)?.into_any(),
            _ => n.as_f64().unwrap_or(f64::NAN).into_pyobject(py)?.into_any(),
        },
        Value::String(s) => PyString::new(py, s).into_any(),
        Value::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(to_py(py, item)?)?;
            }
            list.into_any()
        }
        Value::Object(map) => {
            let dict = PyDict::new(py);
            for (k, v) in map {
                dict.set_item(k, to_py(py, v)?)?;
            }
            dict.into_any()
        }
    })
}
