//! `DocColumns`: a columnar batch is checked, and written, exactly
//! as `upsert` checks and writes the same documents.

mod common;

use std::sync::Arc;

use itzma_core::{
    AttrColumn, AttrSchema, AttrType, DocColumns, Document, Metric, Namespace, Schema,
};
use object_store::ObjectStore;
use serde_json::{Map, Number, Value};

/// A small batch described once, built twice: as columns through the public
/// constructors, and as documents by this file's own conversion, so the code
/// under test never builds its own reference.
#[derive(Clone)]
struct Case {
    ids: Vec<String>,
    vectors: Vec<Vec<f32>>,
    attrs: Vec<(String, AttrColumn)>,
}

impl Case {
    fn columns(&self) -> DocColumns {
        let mut bytes = Vec::new();
        let mut offsets = vec![0u32];
        for id in &self.ids {
            bytes.extend_from_slice(id.as_bytes());
            offsets.push(bytes.len() as u32);
        }
        let dim = self.vectors.first().map_or(0, Vec::len);
        let flat: Vec<f32> = self.vectors.iter().flatten().copied().collect();
        let mut cols = DocColumns::new(bytes, offsets, flat, dim).expect("well formed");
        for (name, column) in &self.attrs {
            cols = cols
                .with_attr(name.clone(), column.clone())
                .expect("well formed");
        }
        cols
    }

    fn documents(&self) -> Vec<Document> {
        (0..self.ids.len())
            .map(|row| Document {
                id: self.ids[row].clone(),
                vector: self.vectors[row].clone(),
                attrs: (!self.attrs.is_empty()).then(|| {
                    let mut map = Map::new();
                    for (name, column) in &self.attrs {
                        map.insert(name.clone(), json_of(column, row));
                    }
                    Value::Object(map)
                }),
            })
            .collect()
    }
}

fn float(f: f64) -> Value {
    Number::from_f64(f).map_or(Value::Null, Value::Number)
}

fn str_at(bytes: &[u8], offsets: &[u32], at: usize) -> String {
    String::from_utf8(bytes[offsets[at] as usize..offsets[at + 1] as usize].to_vec()).unwrap()
}

/// What `upsert` would be given for this row of `column`. A NaN becomes
/// `null` here, since JSON has none; the cases that use NaN compare errors,
/// which come from the column side's own check.
fn json_of(column: &AttrColumn, row: usize) -> Value {
    let items = |offsets: &[u32]| offsets[row] as usize..offsets[row + 1] as usize;
    match column {
        AttrColumn::Null { .. } => Value::Null,
        AttrColumn::Bool { valid, values } if valid[row] => Value::Bool(values[row]),
        AttrColumn::I64 { valid, values } if valid[row] => Value::from(values[row]),
        AttrColumn::F64 { valid, values } if valid[row] => float(values[row]),
        AttrColumn::Text {
            valid,
            bytes,
            offsets,
        } if valid[row] => Value::String(str_at(bytes, offsets, row)),
        AttrColumn::I64Set {
            valid,
            offsets,
            items_valid,
            values,
        } if valid[row] => Value::Array(
            items(offsets)
                .map(|i| {
                    if items_valid[i] {
                        Value::from(values[i])
                    } else {
                        Value::Null
                    }
                })
                .collect(),
        ),
        AttrColumn::F64Set {
            valid,
            offsets,
            items_valid,
            values,
        } if valid[row] => Value::Array(
            items(offsets)
                .map(|i| {
                    if items_valid[i] {
                        float(values[i])
                    } else {
                        Value::Null
                    }
                })
                .collect(),
        ),
        AttrColumn::TextSet {
            valid,
            offsets,
            items_valid,
            bytes,
            byte_offsets,
        } if valid[row] => Value::Array(
            items(offsets)
                .map(|i| {
                    if items_valid[i] {
                        Value::String(str_at(bytes, byte_offsets, i))
                    } else {
                        Value::Null
                    }
                })
                .collect(),
        ),
        _ => Value::Null,
    }
}

fn text(values: &[Option<&str>]) -> AttrColumn {
    let mut bytes = Vec::new();
    let mut offsets = vec![0u32];
    for v in values {
        bytes.extend_from_slice(v.unwrap_or("").as_bytes());
        offsets.push(bytes.len() as u32);
    }
    AttrColumn::Text {
        valid: values.iter().map(Option::is_some).collect(),
        bytes,
        offsets,
    }
}

fn text_set(rows: &[Option<Vec<Option<&str>>>]) -> AttrColumn {
    let (mut valid, mut offsets, mut items_valid, mut bytes, mut byte_offsets) =
        (vec![], vec![0u32], vec![], vec![], vec![0u32]);
    for row in rows {
        valid.push(row.is_some());
        for item in row.iter().flatten() {
            items_valid.push(item.is_some());
            bytes.extend_from_slice(item.unwrap_or("").as_bytes());
            byte_offsets.push(bytes.len() as u32);
        }
        offsets.push(items_valid.len() as u32);
    }
    AttrColumn::TextSet {
        valid,
        offsets,
        items_valid,
        bytes,
        byte_offsets,
    }
}

fn f64_set(rows: &[Option<Vec<Option<f64>>>]) -> AttrColumn {
    let (mut valid, mut offsets, mut items_valid, mut values) =
        (vec![], vec![0u32], vec![], vec![]);
    for row in rows {
        valid.push(row.is_some());
        for item in row.iter().flatten() {
            items_valid.push(item.is_some());
            values.push(item.unwrap_or(0.0));
        }
        offsets.push(items_valid.len() as u32);
    }
    AttrColumn::F64Set {
        valid,
        offsets,
        items_valid,
        values,
    }
}

fn typed() -> Schema {
    Schema::new(2, Metric::Cosine).with_attrs(
        AttrSchema::new()
            .field("on", AttrType::Bool)
            .field("price", AttrType::Number)
            .field("name", AttrType::Text)
            .field("tags", AttrType::TextSet)
            .field("scores", AttrType::NumberSet),
    )
}

/// Two good rows using every field.
fn good() -> Case {
    Case {
        ids: vec!["a".into(), "b".into()],
        vectors: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        attrs: vec![
            (
                "on".into(),
                AttrColumn::Bool {
                    valid: vec![true, false],
                    values: vec![true, false],
                },
            ),
            (
                "price".into(),
                AttrColumn::I64 {
                    valid: vec![true, true],
                    values: vec![5, -3],
                },
            ),
            ("name".into(), text(&[Some("x"), None])),
            (
                "tags".into(),
                text_set(&[Some(vec![Some("t")]), Some(vec![])]),
            ),
            ("scores".into(), f64_set(&[Some(vec![Some(1.5)]), None])),
        ],
    }
}

fn set_attr(case: &mut Case, name: &str, column: AttrColumn) {
    case.attrs.retain(|(n, _)| n != name);
    case.attrs.push((name.into(), column));
}

/// Every single defect, each alone in an otherwise good batch.
fn defects() -> Vec<(&'static str, Case)> {
    let mut out = Vec::new();
    let mut with = |label: &'static str, edit: &dyn Fn(&mut Case)| {
        let mut case = good();
        edit(&mut case);
        out.push((label, case));
    };
    with("empty id", &|c| c.ids[1] = String::new());
    with("id of 513 bytes", &|c| c.ids[1] = "x".repeat(513));
    with("NaN in a vector", &|c| c.vectors[1][0] = f32::NAN);
    with("cosine zero vector", &|c| c.vectors[1] = vec![0.0, 0.0]);
    with("wrong dim", &|c| {
        c.vectors = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]
    });
    with("undeclared column", &|c| {
        set_attr(c, "colour", text(&[Some("red"), Some("blue")]));
    });
    with("integers under a string field", &|c| {
        set_attr(
            c,
            "name",
            AttrColumn::I64 {
                valid: vec![false, true],
                values: vec![0, 7],
            },
        );
    });
    with("a null item in a typed set", &|c| {
        set_attr(
            c,
            "tags",
            text_set(&[Some(vec![Some("t")]), Some(vec![None])]),
        );
    });
    with("a set of 1,025 items", &|c| {
        let many: Vec<Option<&str>> = (0..1_025).map(|_| Some("t")).collect();
        set_attr(c, "tags", text_set(&[None, Some(many)]));
    });
    with("strings in a number set", &|c| {
        set_attr(c, "scores", text_set(&[Some(vec![Some("1")]), None]));
    });
    with("a NaN number", &|c| {
        set_attr(
            c,
            "price",
            AttrColumn::F64 {
                valid: vec![true, true],
                values: vec![1.0, f64::NAN],
            },
        );
    });
    out
}

async fn typed_namespace(store: Arc<dyn ObjectStore>) -> Namespace {
    Namespace::create(store, "docs", typed()).await.unwrap()
}

#[tokio::test]
async fn a_good_batch_passes() {
    let ns = typed_namespace(common::memory()).await;
    good().columns().validate(ns.schema()).unwrap();
    ns.upsert(good().documents()).await.unwrap();
}

#[tokio::test]
async fn each_single_defect_gives_upserts_error() {
    let ns = typed_namespace(common::memory()).await;
    for (label, case) in defects() {
        let from_columns = case.columns().validate(ns.schema()).unwrap_err();
        // A NaN attribute cannot even be given to `upsert` (JSON has no NaN):
        // the bridge refused it before it reached the core, and the column
        // check refuses it the same way the core refuses a bad attribute.
        if label == "a NaN number" {
            assert!(
                matches!(from_columns, itzma_core::ItzmaError::InvalidAttrs(_)),
                "{label}: {from_columns:?}"
            );
            continue;
        }
        let from_documents = ns.upsert(case.documents()).await.unwrap_err();
        assert_eq!(
            format!("{from_columns:?}"),
            format!("{from_documents:?}"),
            "{label}"
        );
    }
}

#[tokio::test]
async fn an_all_null_column_fits_any_type() {
    let ns = typed_namespace(common::memory()).await;
    let mut case = good();
    set_attr(
        &mut case,
        "name",
        AttrColumn::I64 {
            valid: vec![false, false],
            values: vec![1, 2],
        },
    );
    set_attr(&mut case, "on", AttrColumn::Null { rows: 2 });
    case.columns().validate(ns.schema()).unwrap();
    ns.upsert(case.documents()).await.unwrap();
}

#[tokio::test]
async fn an_empty_batch_is_refused() {
    let ns = typed_namespace(common::memory()).await;
    let empty = DocColumns::new(Vec::new(), vec![0], Vec::new(), 2).unwrap();
    assert!(matches!(
        empty.validate(ns.schema()),
        Err(itzma_core::ItzmaError::EmptyBatch)
    ));
}

async fn wal_object(store: &dyn ObjectStore, seq: u64) -> bytes::Bytes {
    use object_store::ObjectStoreExt;
    store
        .get(&itzma_core::wal::wal_path("docs", seq))
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap()
}

/// A namespace as an older build left it: put back at WAL level 2.
async fn at_level_2(store: Arc<dyn ObjectStore>) -> Namespace {
    use object_store::ObjectStoreExt;
    typed_namespace(store.clone()).await;
    let (mut m, _) = itzma_core::manifest::load(&*store, "docs").await.unwrap();
    m.wal_format = 2;
    store
        .put(
            &itzma_core::manifest::manifest_path("docs"),
            serde_json::to_vec(&m).unwrap().into(),
        )
        .await
        .unwrap();
    Namespace::open(store, "docs").await.unwrap()
}

#[tokio::test]
async fn upsert_columns_writes_what_upsert_writes() {
    let (rows_store, cols_store) = (common::memory(), common::memory());
    let by_rows = typed_namespace(rows_store.clone()).await;
    let by_cols = typed_namespace(cols_store.clone()).await;
    assert_eq!(by_rows.upsert(good().documents()).await.unwrap(), 1);
    assert_eq!(by_cols.upsert_columns(&good().columns()).await.unwrap(), 1);
    assert_eq!(
        wal_object(&*rows_store, 1).await,
        wal_object(&*cols_store, 1).await
    );
    let probe = [1.0, 0.2];
    assert_eq!(
        by_rows.query(&probe, 5).await.unwrap(),
        by_cols.query(&probe, 5).await.unwrap()
    );
}

#[tokio::test]
async fn a_bad_batch_writes_nothing() {
    let store = common::memory();
    let ns = typed_namespace(store.clone()).await;
    let before = common::snapshot(&*store).await;
    for (label, case) in defects() {
        assert!(ns.upsert_columns(&case.columns()).await.is_err(), "{label}");
    }
    common::assert_untouched(&*store, &before).await;
}

#[tokio::test]
async fn a_keyed_batch_needs_level_3() {
    let store = common::memory();
    let ns = at_level_2(store.clone()).await;
    let before = common::snapshot(&*store).await;
    let err = ns
        .upsert_columns_keyed(&itzma_core::WriteKey::new("k"), &good().columns())
        .await
        .unwrap_err();
    assert!(
        matches!(err, itzma_core::ItzmaError::WalFormatTooOld { .. }),
        "{err:?}"
    );
    common::assert_untouched(&*store, &before).await;
    assert_eq!(ns.upsert_columns(&good().columns()).await.unwrap(), 1);
}

#[tokio::test]
async fn a_keyed_retry_lands_once() {
    let store = common::memory();
    let ns = typed_namespace(store.clone()).await;
    let seq = ns
        .upsert_columns_keyed(&itzma_core::WriteKey::new("k"), &good().columns())
        .await
        .unwrap();
    let again = ns
        .upsert_columns_keyed(&itzma_core::WriteKey::retry("k", 1), &good().columns())
        .await
        .unwrap();
    assert_eq!(seq, again);
    assert_eq!(
        itzma_core::wal::read_log(&*store, "docs", 0)
            .await
            .unwrap()
            .len(),
        1
    );
}
