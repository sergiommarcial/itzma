//! Typed attributes: the declared type of each field, and the check
//! a document's `attrs` has to pass to be written.

//!
//! A namespace declares its fields when it is created and never changes them,
//! so a stored value always has the type the schema names. A field that is
//! missing or `null` is "null", and every field may be.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{ItzmaError, Result};

/// The most fields one namespace may declare.
pub const MAX_ATTR_FIELDS: usize = 256;
/// The longest field name, in bytes.
pub const MAX_FIELD_NAME_LEN: usize = 64;
/// The most elements one set value may hold.
pub const MAX_SET_ELEMENTS: usize = 1_024;

/// The type of one attribute field.
///
/// A set is a JSON array in the stored document. Order and duplicates do not
/// count when a filter matches it, and the stored array is returned as written.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttrType {
    /// `true` or `false`.
    #[serde(rename = "bool")]
    Bool,
    /// A finite JSON number, compared as `f64`.
    #[serde(rename = "number")]
    Number,
    /// A string.
    #[serde(rename = "string")]
    Text,
    /// An array of strings, matched as a set.
    #[serde(rename = "set<string>")]
    TextSet,
    /// An array of numbers, matched as a set.
    #[serde(rename = "set<number>")]
    NumberSet,
}

impl AttrType {
    /// The name the schema object uses for this type, such as `set<string>`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::Number => "number",
            Self::Text => "string",
            Self::TextSet => "set<string>",
            Self::NumberSet => "set<number>",
        }
    }
}

/// The name of a JSON value's kind, for error messages.
fn kind_of(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// The attribute fields of a namespace and their types.
///
/// Built once, at [`Namespace::create`](crate::Namespace::create), and as
/// immutable afterwards as the dimension and the metric. An empty schema means
/// the namespace takes no attributes at all.
///
/// A namespace made before typed attributes existed has no schema and stays
/// free-form. That state is private to the crate: nothing public builds it, so
/// nothing new can be created free-form.
///
/// # Examples
///
/// ```
/// use itzma_core::{AttrSchema, AttrType};
///
/// let attrs = AttrSchema::new()
///     .field("price", AttrType::Number)
///     .field("active", AttrType::Bool)
///     .field("tags", AttrType::TextSet);
/// assert_eq!(attrs.get("price"), Some(AttrType::Number));
/// assert_eq!(attrs.get("colour"), None);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AttrSchema {
    /// `None` is the schemaless state.
    fields: Option<BTreeMap<String, AttrType>>,
}

impl AttrSchema {
    /// An empty schema: a namespace that takes no attributes. Add fields with
    /// [`field`](Self::field).
    // Not `Default`: the default state of a schema is free-form, which is
    // private, and an empty typed schema is a different thing.
    #[allow(clippy::new_without_default)]
    pub const fn new() -> Self {
        Self {
            fields: Some(BTreeMap::new()),
        }
    }

    /// Declare a field, replacing an earlier declaration of the same name.
    /// Names and counts are checked by `create`.
    #[must_use]
    pub fn field(mut self, name: impl Into<String>, ty: AttrType) -> Self {
        self.fields
            .get_or_insert_with(BTreeMap::new)
            .insert(name.into(), ty);
        self
    }

    /// The declared type of `name`, if it is declared.
    pub fn get(&self, name: &str) -> Option<AttrType> {
        self.fields.as_ref()?.get(name).copied()
    }

    /// Every declared field and its type, in name order.
    pub fn fields(&self) -> impl Iterator<Item = (&str, AttrType)> {
        self.fields
            .iter()
            .flatten()
            .map(|(name, ty)| (name.as_str(), *ty))
    }

    /// The schemaless state: no declared fields, attributes
    /// unchecked, `filter` refused.
    pub(crate) fn free_form() -> Self {
        Self { fields: None }
    }

    /// True for the schemaless state: attributes are free-form JSON,
    /// unchecked. No namespace this build opens is in it; encoders use it.
    pub(crate) fn is_free_form(&self) -> bool {
        self.fields.is_none()
    }

    /// Check the declaration itself, at `create`: not free-form, few enough
    /// fields, well-formed names.
    ///
    /// # Errors
    ///
    /// [`ItzmaError::InvalidSchema`].
    pub(crate) fn check_declaration(&self) -> Result<()> {
        let Some(fields) = &self.fields else {
            return Err(ItzmaError::InvalidSchema(
                "a new namespace must declare its attribute schema",
            ));
        };
        if fields.len() > MAX_ATTR_FIELDS {
            return Err(ItzmaError::InvalidSchema("too many attribute fields"));
        }
        let well_formed = |name: &String| {
            (1..=MAX_FIELD_NAME_LEN).contains(&name.len())
                && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
        };
        if !fields.keys().all(well_formed) {
            return Err(ItzmaError::InvalidSchema(
                "attribute field names must match [A-Za-z0-9_]{1,64}",
            ));
        }
        Ok(())
    }

    /// Check a document's `attrs` against the schema. A free-form schema
    /// accepts anything.
    ///
    /// # Errors
    ///
    /// - [`ItzmaError::InvalidAttrs`] if `attrs` is not a JSON object, or a set
    ///   has more than [`MAX_SET_ELEMENTS`] elements.
    /// - [`ItzmaError::UnknownAttr`] for a key the schema does not declare.
    /// - [`ItzmaError::AttrTypeMismatch`] for a value of the wrong type.
    pub(crate) fn validate(&self, attrs: Option<&Value>) -> Result<()> {
        let Some(fields) = &self.fields else {
            return Ok(());
        };
        let Some(attrs) = attrs else {
            return Ok(());
        };
        let Value::Object(map) = attrs else {
            return Err(ItzmaError::InvalidAttrs("attrs must be a JSON object"));
        };
        for (name, value) in map {
            let ty = fields.get(name).ok_or_else(|| ItzmaError::UnknownAttr {
                field: name.clone(),
            })?;
            check_value(name, *ty, value)?;
        }
        Ok(())
    }
}

/// One value against its declared type. `null` is always fine.
fn check_value(field: &str, ty: AttrType, value: &Value) -> Result<()> {
    let fits = match (ty, value) {
        (_, Value::Null)
        | (AttrType::Bool, Value::Bool(_))
        | (AttrType::Number, Value::Number(_))
        | (AttrType::Text, Value::String(_)) => true,
        (AttrType::TextSet, Value::Array(items)) => {
            check_len(items)?;
            items.iter().all(Value::is_string)
        }
        (AttrType::NumberSet, Value::Array(items)) => {
            check_len(items)?;
            items.iter().all(Value::is_number)
        }
        _ => false,
    };
    if fits {
        Ok(())
    } else {
        Err(ItzmaError::AttrTypeMismatch {
            field: field.to_string(),
            expected: ty.name(),
            found: kind_of(value),
        })
    }
}

fn check_len(items: &[Value]) -> Result<()> {
    if items.len() > MAX_SET_ELEMENTS {
        return Err(ItzmaError::InvalidAttrs("a set has too many elements"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn schema() -> AttrSchema {
        AttrSchema::new()
            .field("active", AttrType::Bool)
            .field("price", AttrType::Number)
            .field("name", AttrType::Text)
            .field("tags", AttrType::TextSet)
            .field("scores", AttrType::NumberSet)
    }

    fn check(attrs: Value) -> Result<()> {
        schema().validate(Some(&attrs))
    }

    #[test]
    fn a_good_document_passes() {
        check(json!({
            "active": true, "price": 9.5, "name": "x",
            "tags": ["a", "b", "a"], "scores": [1, 2.5]
        }))
        .expect("valid");
    }

    #[test]
    fn none_an_empty_object_null_values_and_empty_sets_pass() {
        assert!(schema().validate(None).is_ok());
        check(json!({})).expect("empty object");
        check(json!({"active": null, "price": null})).expect("nulls");
        check(json!({"tags": [], "scores": []})).expect("empty sets");
    }

    #[test]
    fn attrs_must_be_an_object() {
        for bad in [json!(null), json!(3), json!("x"), json!([1])] {
            assert!(matches!(
                check(bad),
                Err(ItzmaError::InvalidAttrs("attrs must be a JSON object"))
            ));
        }
    }

    #[test]
    fn an_undeclared_field_is_refused_by_name() {
        match check(json!({"colour": "red"})) {
            Err(ItzmaError::UnknownAttr { field }) => assert_eq!(field, "colour"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn each_wrong_type_is_a_mismatch_naming_both_sides() {
        let cases = [
            (json!({"active": "yes"}), "active", "bool", "string"),
            (json!({"price": "9"}), "price", "number", "string"),
            (json!({"name": 4}), "name", "string", "number"),
            (json!({"tags": "a"}), "tags", "set<string>", "string"),
            (json!({"tags": [1]}), "tags", "set<string>", "array"),
            (json!({"scores": ["a"]}), "scores", "set<number>", "array"),
            (
                json!({"scores": {"a": 1}}),
                "scores",
                "set<number>",
                "object",
            ),
        ];
        for (attrs, field, expected, found) in cases {
            match check(attrs.clone()) {
                Err(ItzmaError::AttrTypeMismatch {
                    field: f,
                    expected: e,
                    found: g,
                }) => assert_eq!((f.as_str(), e, g), (field, expected, found), "{attrs}"),
                other => panic!("{attrs}: {other:?}"),
            }
        }
    }

    #[test]
    fn a_set_over_the_limit_is_refused_and_one_at_it_passes() {
        let at: Vec<String> = (0..MAX_SET_ELEMENTS).map(|i| i.to_string()).collect();
        check(json!({ "tags": at })).expect("at the limit");
        let over: Vec<String> = (0..=MAX_SET_ELEMENTS).map(|i| i.to_string()).collect();
        assert!(matches!(
            check(json!({ "tags": over })),
            Err(ItzmaError::InvalidAttrs(_))
        ));
    }

    #[test]
    fn a_free_form_schema_accepts_anything() {
        let free = AttrSchema::free_form();
        assert!(free.is_free_form());
        free.validate(Some(&json!({"anything": [1, {"x": null}]})))
            .expect("free-form");
        free.validate(Some(&json!(3))).expect("even a non-object");
    }

    #[test]
    fn the_declaration_is_checked() {
        assert!(schema().check_declaration().is_ok());
        assert!(AttrSchema::new().check_declaration().is_ok());
        assert!(AttrSchema::free_form().check_declaration().is_err());
        for bad in ["", "has space", "dash-ed", "ünï", &"x".repeat(65)] {
            let s = AttrSchema::new().field(bad, AttrType::Bool);
            assert!(s.check_declaration().is_err(), "{bad:?}");
        }
        let ok = AttrSchema::new().field("x".repeat(64), AttrType::Bool);
        assert!(ok.check_declaration().is_ok());
        let many = (0..=MAX_ATTR_FIELDS).fold(AttrSchema::new(), |s, i| {
            s.field(format!("f{i}"), AttrType::Bool)
        });
        assert!(many.check_declaration().is_err());
    }

    #[test]
    fn the_wire_form_is_a_map_of_type_names() {
        let json = serde_json::to_string(&schema()).expect("serialize");
        assert_eq!(
            json,
            r#"{"active":"bool","name":"string","price":"number","scores":"set<number>","tags":"set<string>"}"#
        );
        let back: AttrSchema = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, schema());
        assert_eq!(
            serde_json::to_string(&AttrSchema::new()).expect("serialize"),
            "{}"
        );
    }

    #[test]
    fn a_schema_serializes_as_bare_type_names() {
        let s = AttrSchema::new()
            .field("price", AttrType::Number)
            .field("tags", AttrType::TextSet);
        assert_eq!(
            serde_json::to_string(&s).unwrap(),
            r#"{"price":"number","tags":"set<string>"}"#
        );
    }

    #[test]
    fn a_field_declared_as_an_object_is_refused() {
        let parsed: std::result::Result<AttrSchema, _> =
            serde_json::from_str(r#"{"price":{"type":"number","indexed":true}}"#);
        assert!(parsed.is_err());
    }

    #[test]
    fn a_bare_type_name_parses() {
        let back: AttrSchema = serde_json::from_str(r#"{"price":"number"}"#).unwrap();
        assert_eq!(back.get("price"), Some(AttrType::Number));
    }
}
