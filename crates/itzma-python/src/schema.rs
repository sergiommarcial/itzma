//! Schemas across the boundary: Python passes names, the core wants types.

use std::collections::BTreeMap;

use itzma_core::{AttrSchema, AttrType, Metric, Schema};
use serde_json::Value;

/// A typed schema. `attrs` maps field names to the type names `schema.json`
/// uses (`bool`, `number`, `string`, `set<string>`, `set<number>`).
pub(crate) fn from_args(
    dim: usize,
    metric: &str,
    attrs: &BTreeMap<String, String>,
) -> Result<Schema, String> {
    let metric: Metric = serde_json::from_value(Value::String(metric.into()))
        .map_err(|_| format!("unknown metric {metric:?}: expected cosine, l2 or dot"))?;
    let mut fields = AttrSchema::new();
    for (name, ty) in attrs {
        let ty: AttrType = serde_json::from_value(Value::String(ty.clone())).map_err(|_| {
            format!(
                "attribute {name:?}: unknown type {ty:?}: expected bool, number, string, \
                 set<string> or set<number>"
            )
        })?;
        fields = fields.field(name.clone(), ty);
    }
    Ok(Schema::new(dim, metric).with_attrs(fields))
}

/// The schema as `schema.json` writes it. A free-form namespace (from before
/// typed attributes) has no `attrs` key.
pub(crate) fn to_json(schema: &Schema) -> String {
    serde_json::to_string(schema).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use itzma_core::{AttrType, Metric};

    use super::*;

    #[test]
    fn builds_a_typed_schema() {
        let attrs = BTreeMap::from([
            ("tag".to_string(), "string".to_string()),
            ("labels".to_string(), "set<string>".to_string()),
        ]);
        let schema = from_args(4, "cosine", &attrs).expect("schema");
        assert_eq!(schema.dim, 4);
        assert_eq!(schema.metric, Metric::Cosine);
        assert_eq!(schema.attrs.get("tag"), Some(AttrType::Text));
        assert_eq!(schema.attrs.get("labels"), Some(AttrType::TextSet));
    }

    #[test]
    fn no_attrs_is_typed_and_empty() {
        let schema = from_args(2, "l2", &BTreeMap::new()).expect("schema");
        let json: serde_json::Value = serde_json::from_str(&to_json(&schema)).expect("json");
        assert_eq!(json["attrs"], serde_json::json!({}));
    }

    #[test]
    fn refuses_unknown_names() {
        assert!(
            from_args(2, "euclid", &BTreeMap::new())
                .unwrap_err()
                .contains("euclid")
        );
        let attrs = BTreeMap::from([("x".to_string(), "int".to_string())]);
        assert!(from_args(2, "l2", &attrs).unwrap_err().contains("\"x\""));
    }
}
