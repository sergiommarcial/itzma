//! Arrow out: one typed layout per namespace for every table the bridge
//! returns (query hits and filter pages).

use std::sync::Arc;

use arrow_array::builder::{BooleanBuilder, Float64Builder, ListBuilder, StringBuilder};
use arrow_array::{ArrayRef, Float32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema, SchemaRef};
use itzma_core::{AttrType, Hit, Row, Schema};
use serde_json::Value;

/// Column names the layout itself uses; a field with one of them is renamed.
const BUILT_IN: [&str; 2] = ["id", "distance"];

/// One declared attribute field and the Arrow column it becomes.
#[derive(Debug, Clone)]
struct AttrCol {
    field: String,
    name: String,
    ty: AttrType,
}

/// How a namespace's documents look as Arrow: built once per handle from its
/// schema, so every table the bridge returns has the same columns.
#[derive(Debug)]
pub(crate) struct ArrowLayout {
    attrs: Vec<AttrCol>,
}

fn arrow_name(field: &str, schema: &Schema) -> String {
    let mut name = field.to_string();
    while BUILT_IN.contains(&name.as_str()) || (name != field && schema.attrs.get(&name).is_some())
    {
        name = format!("attr_{name}");
    }
    name
}

fn list_of(inner: DataType) -> DataType {
    DataType::List(Arc::new(Field::new_list_field(inner, true)))
}

fn attr_type(ty: AttrType) -> DataType {
    match ty {
        AttrType::Bool => DataType::Boolean,
        AttrType::Number => DataType::Float64,
        AttrType::Text => DataType::Utf8,
        AttrType::TextSet => list_of(DataType::Utf8),
        AttrType::NumberSet => list_of(DataType::Float64),
    }
}

impl ArrowLayout {
    pub(crate) fn new(schema: &Schema) -> Self {
        let attrs = schema
            .attrs
            .fields()
            .map(|(field, ty)| AttrCol {
                field: field.to_string(),
                name: arrow_name(field, schema),
                ty,
            })
            .collect();
        Self { attrs }
    }

    fn schema(&self, distance: bool) -> SchemaRef {
        let mut fields = vec![Field::new("id", DataType::Utf8, false)];
        if distance {
            fields.push(Field::new("distance", DataType::Float32, false));
        }
        for c in &self.attrs {
            fields.push(Field::new(&c.name, attr_type(c.ty), true));
        }
        Arc::new(ArrowSchema::new(fields))
    }
}

fn field<'a>(attrs: Option<&'a Value>, key: &str) -> Option<&'a Value> {
    attrs.and_then(|a| a.get(key))
}

fn attr_array<'a>(ty: AttrType, values: impl Iterator<Item = Option<&'a Value>>) -> ArrayRef {
    match ty {
        AttrType::Bool => {
            let mut b = BooleanBuilder::new();
            values.for_each(|v| b.append_option(v.and_then(Value::as_bool)));
            Arc::new(b.finish())
        }
        AttrType::Number => {
            let mut b = Float64Builder::new();
            values
                .for_each(|v| b.append_option(v.filter(|v| v.is_number()).and_then(Value::as_f64)));
            Arc::new(b.finish())
        }
        AttrType::Text => {
            let mut b = StringBuilder::new();
            values.for_each(|v| b.append_option(v.and_then(Value::as_str)));
            Arc::new(b.finish())
        }
        AttrType::TextSet => {
            let mut b = ListBuilder::new(StringBuilder::new());
            for v in values {
                let items: Option<Vec<&str>> = v
                    .and_then(Value::as_array)
                    .and_then(|a| a.iter().map(Value::as_str).collect());
                match items {
                    Some(items) => {
                        items.iter().for_each(|s| b.values().append_value(s));
                        b.append(true);
                    }
                    None => b.append(false),
                }
            }
            Arc::new(b.finish())
        }
        AttrType::NumberSet => {
            let mut b = ListBuilder::new(Float64Builder::new());
            for v in values {
                let items: Option<Vec<f64>> = v.and_then(Value::as_array).and_then(|a| {
                    a.iter()
                        .map(|x| x.is_number().then(|| x.as_f64()).flatten())
                        .collect()
                });
                match items {
                    Some(items) => {
                        b.values().append_slice(&items);
                        b.append(true);
                    }
                    None => b.append(false),
                }
            }
            Arc::new(b.finish())
        }
    }
}

impl ArrowLayout {
    fn build(
        &self,
        ids: Vec<String>,
        distances: Option<Vec<f32>>,
        attrs: Vec<Option<Value>>,
    ) -> Result<RecordBatch, String> {
        let schema = self.schema(distances.is_some());
        let mut arrays: Vec<ArrayRef> = vec![Arc::new(StringArray::from(ids))];
        if let Some(d) = distances {
            arrays.push(Arc::new(Float32Array::from(d)));
        }
        for c in &self.attrs {
            arrays.push(attr_array(
                c.ty,
                attrs.iter().map(|a| field(a.as_ref(), &c.field)),
            ));
        }
        RecordBatch::try_new(schema, arrays).map_err(|e| e.to_string())
    }

    pub(crate) fn hits(&self, hits: &[Hit]) -> Result<RecordBatch, String> {
        self.build(
            hits.iter().map(|h| h.id.clone()).collect(),
            Some(hits.iter().map(|h| h.distance).collect()),
            hits.iter().map(|h| h.attrs.clone()).collect(),
        )
    }

    pub(crate) fn rows(&self, rows: &[Row]) -> Result<RecordBatch, String> {
        self.build(
            rows.iter().map(|r| r.id.clone()).collect(),
            None,
            rows.iter().map(|r| r.attrs.clone()).collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::Array;
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Float32Type, Float64Type};
    use arrow_schema::DataType;
    use itzma_core::{AttrSchema, AttrType, Hit, Metric, Row, Schema};
    use serde_json::{Value, json};

    use super::*;

    fn schema() -> Schema {
        Schema::new(2, Metric::L2).with_attrs(
            AttrSchema::new()
                .field("flag", AttrType::Bool)
                .field("year", AttrType::Number)
                .field("tag", AttrType::Text)
                .field("labels", AttrType::TextSet)
                .field("scores", AttrType::NumberSet),
        )
    }

    fn names(batch: &RecordBatch) -> Vec<String> {
        batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect()
    }

    fn rows(attrs: Vec<Option<Value>>) -> Vec<Row> {
        attrs
            .into_iter()
            .enumerate()
            .map(|(i, attrs)| Row {
                id: format!("d{i}"),
                attrs,
            })
            .collect()
    }

    #[test]
    fn columns_and_types() {
        let layout = ArrowLayout::new(&schema());
        let batch = layout
            .rows(&rows(vec![Some(json!({"year": 2020}))]))
            .expect("rows");
        assert_eq!(
            names(&batch),
            ["id", "flag", "labels", "scores", "tag", "year"],
            "only the id and the attributes, no vector column"
        );
        let types: Vec<DataType> = batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.data_type().clone())
            .collect();
        assert_eq!(types[0], DataType::Utf8);
        assert_eq!(types[1], DataType::Boolean);
        assert!(matches!(&types[2], DataType::List(f) if f.data_type() == &DataType::Utf8));
        assert!(matches!(&types[3], DataType::List(f) if f.data_type() == &DataType::Float64));
        assert_eq!(types[4], DataType::Utf8);
        assert_eq!(types[5], DataType::Float64);
    }

    #[test]
    fn values_nulls_and_wrong_types() {
        let layout = ArrowLayout::new(&schema());
        let batch = layout
            .rows(&rows(vec![
                Some(
                    json!({"flag": true, "year": 2020, "tag": "a", "labels": ["x", "y"], "scores": [1.5]}),
                ),
                Some(
                    json!({"flag": null, "year": "2020", "tag": 7, "labels": ["x", 3], "scores": "no"}),
                ),
                None,
            ]))
            .expect("rows");
        let col = |n: &str| batch.column_by_name(n).expect(n).clone();
        assert!(col("flag").as_boolean().value(0));
        assert_eq!(col("year").as_primitive::<Float64Type>().value(0), 2020.0);
        assert_eq!(col("tag").as_string::<i32>().value(0), "a");
        let labels = col("labels");
        let first = labels.as_list::<i32>().value(0);
        assert_eq!(first.as_string::<i32>().value(1), "y");
        for name in ["flag", "year", "tag", "labels", "scores"] {
            assert!(col(name).is_null(1), "{name}: a wrong type is null");
            assert!(col(name).is_null(2), "{name}: no attributes is null");
        }
    }

    #[test]
    fn collisions_are_prefixed() {
        let schema = Schema::new(2, Metric::L2).with_attrs(
            AttrSchema::new()
                .field("id", AttrType::Text)
                .field("attr_id", AttrType::Text)
                .field("distance", AttrType::Number),
        );
        let layout = ArrowLayout::new(&schema);
        let out = layout
            .hits(&[Hit {
                id: "a".into(),
                distance: 0.5,
                attrs: Some(json!({"id": "x", "attr_id": "y", "distance": 2.0})),
            }])
            .expect("hits");
        assert_eq!(
            names(&out),
            ["id", "distance", "attr_id", "attr_distance", "attr_attr_id"]
        );
        let col = |n: &str| out.column_by_name(n).expect(n).clone();
        assert_eq!(col("attr_attr_id").as_string::<i32>().value(0), "x");
        assert_eq!(col("attr_id").as_string::<i32>().value(0), "y");
    }

    #[test]
    fn hits_carry_distance_and_typed_attrs() {
        let layout = ArrowLayout::new(&schema());
        let hits = vec![Hit {
            id: "a".into(),
            distance: 0.5,
            attrs: Some(json!({"year": 1.0})),
        }];
        let out = layout.hits(&hits).expect("hits");
        assert_eq!(
            names(&out),
            ["id", "distance", "flag", "labels", "scores", "tag", "year"]
        );
        assert_eq!(out.column(1).as_primitive::<Float32Type>().value(0), 0.5);
    }
}
