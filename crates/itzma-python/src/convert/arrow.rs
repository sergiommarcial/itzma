//! Arrow record batches to documents.
//!
//! Columns: `id` (any string type), `vector` (a list, large list or fixed
//! size list of floats), and every other column an attribute. Types are normalized
//! with `arrow_cast::cast`, so Polars' `Utf8View` and `LargeList` work too.

use std::sync::{Arc, Mutex};

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type, Int64Type};
use arrow_array::{Array, ArrayRef, ListArray, RecordBatch, RecordBatchReader, StringArray};
use arrow_cast::cast;
use arrow_schema::{DataType, Field};
#[cfg(test)]
use itzma_core::Document;
use itzma_core::{AttrColumn as Column, DocColumns};
use pyo3::prelude::*;
use pyo3_arrow::PyRecordBatchReader;
#[cfg(test)]
use serde_json::{Map, Number, Value};

use crate::error;

fn list_of(inner: DataType) -> DataType {
    DataType::List(Arc::new(Field::new_list_field(inner, true)))
}

fn cast_to(name: &str, col: &ArrayRef, to: &DataType) -> Result<ArrayRef, String> {
    cast(col, to).map_err(|e| format!("column {name:?}: {e}"))
}

#[cfg(test)]
fn number(v: f64) -> Result<Value, String> {
    Number::from_f64(v)
        .map(Value::Number)
        .ok_or_else(|| format!("attribute numbers must be finite, got {v}"))
}

enum AttrColumn {
    Null,
    Bool(ArrayRef),
    Int(ArrayRef),
    Float(ArrayRef),
    Text(ArrayRef),
    TextList(ArrayRef),
    IntList(ArrayRef),
    NumberList(ArrayRef),
}

fn is_string(t: &DataType) -> bool {
    matches!(t, DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View)
}

fn is_int_that_fits_i64(t: &DataType) -> bool {
    t.is_integer() && *t != DataType::UInt64
}

fn is_number(t: &DataType) -> bool {
    t.is_integer() || t.is_floating()
}

impl AttrColumn {
    fn new(name: &str, col: &ArrayRef) -> Result<Self, String> {
        Ok(match col.data_type() {
            DataType::Null => AttrColumn::Null,
            DataType::Boolean => AttrColumn::Bool(col.clone()),
            DataType::UInt64 => AttrColumn::Float(cast_to(name, col, &DataType::Float64)?),
            t if t.is_integer() => AttrColumn::Int(cast_to(name, col, &DataType::Int64)?),
            t if t.is_floating() => AttrColumn::Float(cast_to(name, col, &DataType::Float64)?),
            t if is_string(t) => AttrColumn::Text(cast_to(name, col, &DataType::Utf8)?),
            // Pandas categories and Polars' Categorical arrive dictionary-encoded.
            DataType::Dictionary(_, values) if is_string(values) => {
                AttrColumn::Text(cast_to(name, col, &DataType::Utf8)?)
            }
            DataType::List(f) | DataType::LargeList(f) | DataType::FixedSizeList(f, _)
                if is_int_that_fits_i64(f.data_type()) =>
            {
                AttrColumn::IntList(cast_to(name, col, &list_of(DataType::Int64))?)
            }
            DataType::List(f) | DataType::LargeList(f) | DataType::FixedSizeList(f, _)
                if is_string(f.data_type()) =>
            {
                AttrColumn::TextList(cast_to(name, col, &list_of(DataType::Utf8))?)
            }
            DataType::List(f) | DataType::LargeList(f) | DataType::FixedSizeList(f, _)
                if is_number(f.data_type()) =>
            {
                AttrColumn::NumberList(cast_to(name, col, &list_of(DataType::Float64))?)
            }
            other => {
                return Err(format!(
                    "column {name:?}: type {other} cannot be an attribute; use bool, a number, \
                     a string, or a list of strings or numbers"
                ));
            }
        })
    }

    #[cfg(test)]
    fn value(&self, row: usize) -> Result<Value, String> {
        Ok(match self {
            AttrColumn::Null => Value::Null,
            AttrColumn::Bool(a) if a.is_null(row) => Value::Null,
            AttrColumn::Bool(a) => Value::Bool(a.as_boolean().value(row)),
            AttrColumn::Int(a) if a.is_null(row) => Value::Null,
            AttrColumn::Int(a) => Value::from(a.as_primitive::<Int64Type>().value(row)),
            AttrColumn::Float(a) if a.is_null(row) => Value::Null,
            AttrColumn::Float(a) => number(a.as_primitive::<Float64Type>().value(row))?,
            AttrColumn::Text(a) if a.is_null(row) => Value::Null,
            AttrColumn::Text(a) => Value::String(a.as_string::<i32>().value(row).to_owned()),
            AttrColumn::TextList(a) if a.is_null(row) => Value::Null,
            AttrColumn::TextList(a) => {
                let items = a.as_list::<i32>().value(row);
                let items = items.as_string::<i32>();
                Value::Array(
                    (0..items.len())
                        .map(|i| {
                            if items.is_null(i) {
                                Value::Null
                            } else {
                                Value::String(items.value(i).to_owned())
                            }
                        })
                        .collect(),
                )
            }
            AttrColumn::IntList(a) if a.is_null(row) => Value::Null,
            AttrColumn::IntList(a) => {
                let items = a.as_list::<i32>().value(row);
                let items = items.as_primitive::<Int64Type>();
                Value::Array(
                    (0..items.len())
                        .map(|i| {
                            if items.is_null(i) {
                                Value::Null
                            } else {
                                Value::from(items.value(i))
                            }
                        })
                        .collect(),
                )
            }
            AttrColumn::NumberList(a) if a.is_null(row) => Value::Null,
            AttrColumn::NumberList(a) => {
                let items = a.as_list::<i32>().value(row);
                let items = items.as_primitive::<Float64Type>();
                let mut out = Vec::with_capacity(items.len());
                for i in 0..items.len() {
                    out.push(if items.is_null(i) {
                        Value::Null
                    } else {
                        number(items.value(i))?
                    });
                }
                Value::Array(out)
            }
        })
    }
}

/// Which rows (or items) of `array` are not null.
fn validity(array: &dyn Array) -> Vec<bool> {
    (0..array.len()).map(|i| array.is_valid(i)).collect()
}

/// Offsets rebased to start at 0, as a sliced Arrow array's may not.
fn rebased(offsets: &[i32]) -> Vec<u32> {
    let first = offsets.first().copied().unwrap_or(0);
    offsets.iter().map(|&o| (o - first) as u32).collect()
}

/// A string array's bytes and offsets, copied once.
fn string_parts(strings: &StringArray) -> (Vec<u8>, Vec<u32>) {
    let offsets = strings.value_offsets();
    let start = offsets.first().copied().unwrap_or(0) as usize;
    let end = offsets.last().copied().unwrap_or(0) as usize;
    (strings.value_data()[start..end].to_vec(), rebased(offsets))
}

/// The items of a list array's rows, as one array starting at the first.
fn list_items(list: &ListArray) -> ArrayRef {
    let offsets = list.value_offsets();
    let start = offsets.first().copied().unwrap_or(0) as usize;
    let end = offsets.last().copied().unwrap_or(0) as usize;
    list.values().slice(start, end - start)
}

impl AttrColumn {
    /// The column as `itzma_core` takes it: buffers copied once, with the
    /// same kinds `value` gives (integers stay integers, `UInt64` is a float,
    /// null items stay null). Floats are not checked here: the core refuses a
    /// NaN as it checks the batch.
    fn column(&self, rows: usize) -> Column {
        match self {
            AttrColumn::Null => Column::Null { rows },
            AttrColumn::Bool(a) => {
                let b = a.as_boolean();
                Column::Bool {
                    valid: validity(b),
                    values: (0..rows).map(|i| b.value(i)).collect(),
                }
            }
            AttrColumn::Int(a) => {
                let p = a.as_primitive::<Int64Type>();
                Column::I64 {
                    valid: validity(p),
                    values: p.values().to_vec(),
                }
            }
            AttrColumn::Float(a) => {
                let p = a.as_primitive::<Float64Type>();
                Column::F64 {
                    valid: validity(p),
                    values: p.values().to_vec(),
                }
            }
            AttrColumn::Text(a) => {
                let strings = a.as_string::<i32>();
                let (bytes, offsets) = string_parts(strings);
                Column::Text {
                    valid: validity(strings),
                    bytes,
                    offsets,
                }
            }
            AttrColumn::IntList(a) => {
                let list = a.as_list::<i32>();
                let items = list_items(list);
                let items = items.as_primitive::<Int64Type>();
                Column::I64Set {
                    valid: validity(list),
                    offsets: rebased(list.value_offsets()),
                    items_valid: validity(items),
                    values: items.values().to_vec(),
                }
            }
            AttrColumn::NumberList(a) => {
                let list = a.as_list::<i32>();
                let items = list_items(list);
                let items = items.as_primitive::<Float64Type>();
                Column::F64Set {
                    valid: validity(list),
                    offsets: rebased(list.value_offsets()),
                    items_valid: validity(items),
                    values: items.values().to_vec(),
                }
            }
            AttrColumn::TextList(a) => {
                let list = a.as_list::<i32>();
                let items = list_items(list);
                let items = items.as_string::<i32>();
                let (bytes, byte_offsets) = string_parts(items);
                Column::TextSet {
                    valid: validity(list),
                    offsets: rebased(list.value_offsets()),
                    items_valid: validity(items),
                    bytes,
                    byte_offsets,
                }
            }
        }
    }
}

/// A record batch as one columnar batch: the same columns, the
/// same kinds and the same refusals as the row-by-row conversion it
/// replaced (`documents`, kept in the tests as the reference), with each
/// buffer copied once instead of a `Document` per row. One more refusal: a
/// row whose vector length differs from the first row's, since a batch has
/// one `dim`.
pub(crate) fn columns_from_batch(batch: &RecordBatch) -> Result<DocColumns, String> {
    let rows = batch.num_rows();
    let ids = batch
        .column_by_name("id")
        .ok_or("an Arrow table needs an \"id\" column")?;
    let ids = cast_to("id", ids, &DataType::Utf8)?;
    let ids = ids.as_string::<i32>();
    let vectors = batch
        .column_by_name("vector")
        .ok_or("an Arrow table needs a \"vector\" column")?;
    let vectors = cast_to("vector", vectors, &list_of(DataType::Float32))?;
    let vectors = vectors.as_list::<i32>();
    let schema = batch.schema();
    let mut attrs = Vec::new();
    for (field, column) in schema.fields().iter().zip(batch.columns()) {
        if !matches!(field.name().as_str(), "id" | "vector") {
            attrs.push((field.name().clone(), AttrColumn::new(field.name(), column)?));
        }
    }

    let offsets = vectors.value_offsets();
    let dim = match rows {
        0 => 0,
        _ => (offsets[1] - offsets[0]) as usize,
    };
    let items = list_items(vectors);
    let items = items.as_primitive::<Float32Type>();
    for row in 0..rows {
        if ids.is_null(row) {
            return Err(format!("row {row}: id is null"));
        }
        if vectors.is_null(row) {
            return Err(format!("row {row}: vector is null"));
        }
        let (start, end) = (
            (offsets[row] - offsets[0]) as usize,
            (offsets[row + 1] - offsets[0]) as usize,
        );
        if (start..end).any(|i| items.is_null(i)) {
            return Err(format!("row {row}: the vector has a null element"));
        }
        if end - start != dim {
            return Err(format!(
                "row {row}: the vector has {} values, but the first row's has {dim}",
                end - start
            ));
        }
    }
    let (id_bytes, id_offsets) = string_parts(ids);
    let mut cols = DocColumns::new(id_bytes, id_offsets, items.values().to_vec(), dim)
        .map_err(|e| e.to_string())?;
    for (name, column) in &attrs {
        cols = cols
            .with_attr(name.clone(), column.column(rows))
            .map_err(|e| format!("column {name:?}: {e}"))?;
    }
    Ok(cols)
}

#[cfg(test)]
pub(crate) fn documents(batch: &RecordBatch) -> Result<Vec<Document>, String> {
    let ids = batch
        .column_by_name("id")
        .ok_or("an Arrow table needs an \"id\" column")?;
    let ids = cast_to("id", ids, &DataType::Utf8)?;
    let ids = ids.as_string::<i32>();
    let vectors = batch
        .column_by_name("vector")
        .ok_or("an Arrow table needs a \"vector\" column")?;
    let vectors = cast_to("vector", vectors, &list_of(DataType::Float32))?;
    let vectors = vectors.as_list::<i32>();
    let schema = batch.schema();
    let mut attrs = Vec::new();
    for (field, column) in schema.fields().iter().zip(batch.columns()) {
        if !matches!(field.name().as_str(), "id" | "vector") {
            attrs.push((field.name().clone(), AttrColumn::new(field.name(), column)?));
        }
    }
    let mut docs = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        if ids.is_null(row) {
            return Err(format!("row {row}: id is null"));
        }
        if vectors.is_null(row) {
            return Err(format!("row {row}: vector is null"));
        }
        let values = vectors.value(row);
        let values = values.as_primitive::<Float32Type>();
        if values.null_count() > 0 {
            return Err(format!("row {row}: the vector has a null element"));
        }
        let doc_attrs = if attrs.is_empty() {
            None
        } else {
            let mut map = Map::with_capacity(attrs.len());
            for (name, column) in &attrs {
                map.insert(name.clone(), column.value(row)?);
            }
            Some(Value::Object(map))
        };
        docs.push(Document {
            id: ids.value(row).to_owned(),
            vector: values.values().to_vec(),
            attrs: doc_attrs,
        });
    }
    Ok(docs)
}

/// An Arrow stream waiting to be written: anything with
/// `__arrow_c_stream__`, read one record batch at a time by the write that
/// takes it, never whole.
#[pyclass(frozen, name = "ArrowStream", module = "itzma._itzma")]
pub(crate) struct PyArrowStream {
    reader: Mutex<Option<Box<dyn RecordBatchReader + Send>>>,
}

impl PyArrowStream {
    /// The reader, once. A stream is consumed by the write that takes it.
    pub(crate) fn take(&self) -> PyResult<Box<dyn RecordBatchReader + Send>> {
        self.reader
            .lock()
            .map_err(|_| error::itzma_error("stream lock poisoned".into()))?
            .take()
            .ok_or_else(|| error::itzma_error("this stream was already written".into()))
    }
}

/// Wrap anything with `__arrow_c_stream__` for a streaming write.
#[pyfunction]
pub(crate) fn stream_from_arrow(data: PyRecordBatchReader) -> PyResult<PyArrowStream> {
    Ok(PyArrowStream {
        reader: Mutex::new(Some(data.into_reader()?)),
    })
}

/// A query vector from an Arrow float array.
pub(crate) fn float32_values(array: &ArrayRef) -> Result<Vec<f32>, String> {
    let array = cast_to("vector", array, &DataType::Float32)?;
    if array.null_count() > 0 {
        return Err("the query vector has a null element".into());
    }
    Ok(array.as_primitive::<Float32Type>().values().to_vec())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::builder::{
        FixedSizeListBuilder, Float32Builder, Float64Builder, LargeListBuilder, ListBuilder,
        StringBuilder,
    };
    use arrow_array::{
        ArrayRef, BooleanArray, Date32Array, Float64Array, Int64Array, NullArray, RecordBatch,
        StringArray, StringViewArray,
    };
    use serde_json::json;

    use super::*;

    fn fixed_vectors(rows: &[[f32; 2]]) -> ArrayRef {
        let mut b = FixedSizeListBuilder::new(Float32Builder::new(), 2);
        for row in rows {
            b.values().append_slice(row);
            b.append(true);
        }
        Arc::new(b.finish())
    }

    fn batch(columns: Vec<(&str, ArrayRef)>) -> RecordBatch {
        RecordBatch::try_from_iter(columns).expect("batch")
    }

    /// Every attribute kind, nulls in each: three rows.
    fn rich() -> RecordBatch {
        use arrow_array::builder::{BooleanBuilder, Int64Builder};
        use arrow_array::types::Int32Type;
        use arrow_array::{DictionaryArray, UInt64Array};

        let ids: ArrayRef = Arc::new(StringArray::from(vec!["a", "b\"é", "c"]));
        let vectors = fixed_vectors(&[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]);
        let mut flag = BooleanBuilder::new();
        flag.append_option(Some(true));
        flag.append_null();
        flag.append_option(Some(false));
        let flag: ArrayRef = Arc::new(flag.finish());
        let year: ArrayRef = Arc::new(Int64Array::from(vec![Some(2020), None, Some(-1)]));
        let big: ArrayRef = Arc::new(UInt64Array::from(vec![1u64 << 60, 5, 0]));
        let score: ArrayRef = Arc::new(Float64Array::from(vec![Some(1.5), Some(5.0), None]));
        let mut labels = ListBuilder::new(StringBuilder::new());
        labels.values().append_value("red");
        labels.values().append_value("é\n");
        labels.append(true);
        labels.append(false);
        labels.append(true);
        let labels: ArrayRef = Arc::new(labels.finish());
        let tag: DictionaryArray<Int32Type> = vec!["news", "blog", "news"].into_iter().collect();
        let tag: ArrayRef = Arc::new(tag);
        let mut years = ListBuilder::new(Int64Builder::new());
        years.values().append_slice(&[2020, 2021]);
        years.append(true);
        years.append(true);
        years.values().append_value(7);
        years.append(true);
        let years: ArrayRef = Arc::new(years.finish());
        let mut nums = ListBuilder::new(Float64Builder::new());
        nums.values().append_slice(&[0.5, 2.0]);
        nums.append(true);
        nums.append(false);
        nums.values().append_value(3.0);
        nums.append(true);
        let nums: ArrayRef = Arc::new(nums.finish());
        let nothing: ArrayRef = Arc::new(NullArray::new(3));

        batch(vec![
            ("id", ids),
            ("vector", vectors),
            ("flag", flag),
            ("year", year),
            ("big", big),
            ("score", score),
            ("labels", labels),
            ("tag", tag),
            ("years", years),
            ("nums", nums),
            ("nothing", nothing),
        ])
    }

    /// The WAL object a write puts at seq 1 of a fresh namespace.
    fn first_object(write: impl AsyncFnOnce(&itzma_core::Namespace)) -> Vec<u8> {
        use itzma_core::{AttrSchema, AttrType, Metric, Namespace, Schema};
        use object_store::ObjectStoreExt;

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        runtime.block_on(async {
            let store = Arc::new(object_store::memory::InMemory::new());
            let schema = Schema::new(2, Metric::L2).with_attrs(
                AttrSchema::new()
                    .field("flag", AttrType::Bool)
                    .field("year", AttrType::Number)
                    .field("big", AttrType::Number)
                    .field("score", AttrType::Number)
                    .field("labels", AttrType::TextSet)
                    .field("tag", AttrType::Text)
                    .field("years", AttrType::NumberSet)
                    .field("nums", AttrType::NumberSet)
                    .field("nothing", AttrType::Text),
            );
            let ns = Namespace::create(store.clone(), "docs", schema)
                .await
                .expect("create");
            write(&ns).await;
            store
                .get(&itzma_core::wal::wal_path("docs", 1))
                .await
                .expect("object")
                .bytes()
                .await
                .expect("bytes")
                .to_vec()
        })
    }

    #[test]
    fn columns_write_what_documents_write() {
        let full = rich();
        for (label, batch) in [("whole", full.clone()), ("sliced", full.slice(1, 2))] {
            let by_rows = first_object(async |ns| {
                ns.upsert(documents(&batch).expect("docs"))
                    .await
                    .expect("upsert");
            });
            let by_columns = first_object(async |ns| {
                let cols = columns_from_batch(&batch).expect("columns");
                ns.upsert_columns(&cols).await.expect("upsert_columns");
            });
            assert_eq!(by_rows, by_columns, "{label}");
        }
    }

    #[test]
    fn columns_refuse_what_documents_refuse() {
        let ids: ArrayRef = Arc::new(StringArray::from(vec![Some("a"), None]));
        let err = columns_from_batch(&batch(vec![
            ("id", ids),
            ("vector", fixed_vectors(&[[1.0, 0.0], [0.0, 1.0]])),
        ]))
        .expect_err("null id");
        assert_eq!(err, "row 1: id is null");

        let ids: ArrayRef = Arc::new(StringArray::from(vec!["a", "b"]));
        let mut vectors = ListBuilder::new(Float32Builder::new());
        vectors.values().append_slice(&[1.0, 2.0]);
        vectors.append(true);
        vectors.append(false);
        let vectors: ArrayRef = Arc::new(vectors.finish());
        let err = columns_from_batch(&batch(vec![("id", ids.clone()), ("vector", vectors)]))
            .expect_err("null vector");
        assert_eq!(err, "row 1: vector is null");

        let mut vectors = ListBuilder::new(Float32Builder::new());
        vectors.values().append_slice(&[1.0, 2.0]);
        vectors.append(true);
        vectors.values().append_value(1.0);
        vectors.values().append_null();
        vectors.append(true);
        let vectors: ArrayRef = Arc::new(vectors.finish());
        let err = columns_from_batch(&batch(vec![("id", ids.clone()), ("vector", vectors)]))
            .expect_err("null element");
        assert_eq!(err, "row 1: the vector has a null element");

        let mut vectors = ListBuilder::new(Float32Builder::new());
        vectors.values().append_slice(&[1.0, 2.0]);
        vectors.append(true);
        vectors.values().append_slice(&[1.0, 2.0, 3.0]);
        vectors.append(true);
        let vectors: ArrayRef = Arc::new(vectors.finish());
        let err = columns_from_batch(&batch(vec![("id", ids.clone()), ("vector", vectors)]))
            .expect_err("ragged");
        assert!(err.starts_with("row 1: the vector has 3 values"), "{err}");

        let when: ArrayRef = Arc::new(Date32Array::from(vec![1, 2]));
        let err = columns_from_batch(&batch(vec![
            ("id", ids),
            ("vector", fixed_vectors(&[[1.0, 0.0], [0.0, 1.0]])),
            ("when", when),
        ]))
        .expect_err("a date column");
        assert!(err.contains("cannot be an attribute"), "{err}");
    }

    #[test]
    fn plain_table() {
        let ids: ArrayRef = Arc::new(StringArray::from(vec!["a", "b"]));
        let docs = documents(&batch(vec![
            ("id", ids),
            ("vector", fixed_vectors(&[[1.0, 2.0], [3.0, 4.0]])),
        ]))
        .expect("docs");
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[1].id, "b");
        assert_eq!(docs[1].vector, vec![3.0, 4.0]);
        assert_eq!(docs[0].attrs, None);
    }

    #[test]
    fn polars_types() {
        let ids: ArrayRef = Arc::new(StringViewArray::from(vec!["a"]));
        let mut vectors = LargeListBuilder::new(Float64Builder::new());
        vectors.values().append_slice(&[0.5, 1.5]);
        vectors.append(true);
        let vectors: ArrayRef = Arc::new(vectors.finish());
        let tag: ArrayRef = Arc::new(StringViewArray::from(vec!["x"]));
        let docs =
            documents(&batch(vec![("id", ids), ("vector", vectors), ("tag", tag)])).expect("docs");
        assert_eq!(docs[0].vector, vec![0.5, 1.5]);
        assert_eq!(docs[0].attrs, Some(json!({"tag": "x"})));
    }

    #[test]
    fn attribute_columns() {
        let ids: ArrayRef = Arc::new(StringArray::from(vec!["a", "b"]));
        let flag: ArrayRef = Arc::new(BooleanArray::from(vec![Some(true), None]));
        let year: ArrayRef = Arc::new(Int64Array::from(vec![2020, 2021]));
        let score: ArrayRef = Arc::new(Float64Array::from(vec![1.5, 2.5]));
        let mut labels = ListBuilder::new(StringBuilder::new());
        labels.values().append_value("red");
        labels.append(true);
        labels.append(false);
        let labels: ArrayRef = Arc::new(labels.finish());
        let nothing: ArrayRef = Arc::new(NullArray::new(2));
        let docs = documents(&batch(vec![
            ("id", ids),
            ("vector", fixed_vectors(&[[1.0, 0.0], [0.0, 1.0]])),
            ("flag", flag),
            ("year", year),
            ("score", score),
            ("labels", labels),
            ("nothing", nothing),
        ]))
        .expect("docs");
        assert_eq!(
            docs[0].attrs,
            Some(
                json!({"flag": true, "year": 2020, "score": 1.5, "labels": ["red"], "nothing": null})
            )
        );
        assert_eq!(
            docs[1].attrs,
            Some(
                json!({"flag": null, "year": 2021, "score": 2.5, "labels": null, "nothing": null})
            )
        );
    }

    #[test]
    fn refusals() {
        let ids: ArrayRef = Arc::new(StringArray::from(vec![Some("a"), None]));
        let err = documents(&batch(vec![
            ("id", ids),
            ("vector", fixed_vectors(&[[1.0, 0.0], [0.0, 1.0]])),
        ]))
        .expect_err("null id");
        assert!(err.contains("row 1") && err.contains("id"), "{err}");

        let ids: ArrayRef = Arc::new(StringArray::from(vec!["a"]));
        assert!(
            documents(&batch(vec![("id", ids.clone())]))
                .expect_err("no vector")
                .contains("vector")
        );
        assert!(
            documents(&batch(vec![("vector", fixed_vectors(&[[1.0, 0.0]]))]))
                .expect_err("no id")
                .contains("id")
        );

        let when: ArrayRef = Arc::new(Date32Array::from(vec![1]));
        let err = documents(&batch(vec![
            ("id", ids.clone()),
            ("vector", fixed_vectors(&[[1.0, 0.0]])),
            ("when", when),
        ]))
        .expect_err("date");
        assert!(err.contains("when"), "{err}");

        let nan: ArrayRef = Arc::new(Float64Array::from(vec![f64::NAN]));
        let err = documents(&batch(vec![
            ("id", ids),
            ("vector", fixed_vectors(&[[1.0, 0.0]])),
            ("x", nan),
        ]))
        .expect_err("nan");
        assert!(err.contains("finite"), "{err}");
    }

    #[test]
    fn float_arrays_to_a_query_vector() {
        let values: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 2.0]));
        assert_eq!(float32_values(&values).expect("values"), vec![1.0, 2.0]);
        let nulls: ArrayRef = Arc::new(Float64Array::from(vec![Some(1.0), None]));
        assert!(float32_values(&nulls).is_err());
    }

    #[test]
    fn integer_lists_stay_integers_and_dictionary_strings_work() {
        use arrow_array::DictionaryArray;
        use arrow_array::builder::Int64Builder;
        use arrow_array::types::Int32Type;

        let ids: ArrayRef = Arc::new(StringArray::from(vec!["a", "b"]));
        let mut years = ListBuilder::new(Int64Builder::new());
        years.values().append_slice(&[2020, 2021]);
        years.append(true);
        years.append(true);
        let years: ArrayRef = Arc::new(years.finish());
        let tag: DictionaryArray<Int32Type> = vec!["news", "blog"].into_iter().collect();
        let tag: ArrayRef = Arc::new(tag);
        let docs = documents(&batch(vec![
            ("id", ids),
            ("vector", fixed_vectors(&[[1.0, 0.0], [0.0, 1.0]])),
            ("years", years),
            ("tag", tag),
        ]))
        .expect("docs");
        assert_eq!(
            docs[0].attrs,
            Some(json!({"years": [2020, 2021], "tag": "news"}))
        );
        assert_eq!(docs[1].attrs, Some(json!({"years": [], "tag": "blog"})));
    }
}
