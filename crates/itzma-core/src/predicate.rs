//! Filter predicates.

//!
//! A typed tree, checked against the namespace's attribute schema before any
//! read, then evaluated against each document's stored attributes.
//!
//! Logic is two-valued. A comparison against a null field is false, and
//! [`Predicate::Not`] flips it, so `not (price > 10)` matches a document with no
//! price. SQL would call that unknown and leave the document out. Use
//! [`Predicate::IsNull`] when the difference matters.
//!
//! A stored value that does not fit its field's declared type is treated as
//! null by the typed operators (`Number`, `Text`, `IsTrue`, `Contains` and the
//! rest) but is still a value to `IsNull` and `IsNotNull`. This build cannot
//! write one, so it only comes from a hand-made object.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::attrs::{AttrSchema, AttrType};
use crate::error::{ItzmaError, Result};

/// How a number field is compared with a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Compare {
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
    /// Less than.
    Lt,
    /// Less than or equal.
    Le,
    /// Greater than.
    Gt,
    /// Greater than or equal.
    Ge,
}

impl Compare {
    // Exact equality is the semantics: a stored number equals a value when the
    // two `f64`s are the same.
    #[allow(clippy::float_cmp)]
    fn apply(self, field: f64, value: f64) -> bool {
        match self {
            Self::Eq => field == value,
            Self::Ne => field != value,
            Self::Lt => field < value,
            Self::Le => field <= value,
            Self::Gt => field > value,
            Self::Ge => field >= value,
        }
    }
}

/// How a string field is compared with a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TextCompare {
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
    /// The field begins with the value. An empty value matches every string.
    StartsWith,
}

/// One element of a set, in a predicate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AttrValue {
    /// An element of a `set<number>`.
    Number(f64),
    /// An element of a `set<string>`.
    Text(String),
}

/// A condition over a document's attributes.
///
/// Fields are named as the namespace's [`AttrSchema`] declares them, and an
/// operator has to fit the field's type: a wrong one is refused by
/// [`Namespace::filter`](crate::Namespace::filter) before any read, never an
/// empty result.
///
/// The serialized form is derived from the type and is not a promised format.
///
/// # Examples
///
/// ```
/// use itzma_core::{Compare, Predicate};
///
/// // active is true and price is under 20
/// let p = Predicate::and(vec![
///     Predicate::is_true("active"),
///     Predicate::number("price", Compare::Lt, 20.0),
/// ]);
/// # let _ = p;
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Predicate {
    /// Every condition holds. At least one is required.
    And(Vec<Predicate>),
    /// At least one condition holds. At least one is required.
    Or(Vec<Predicate>),
    /// The condition does not hold. A null field makes most conditions false,
    /// so this matches it.
    Not(Box<Predicate>),
    /// The field is missing or `null`. Any type.
    IsNull(String),
    /// The field has a value. Any type.
    IsNotNull(String),
    /// A `bool` field is `true`.
    IsTrue(String),
    /// A `bool` field is `false`.
    IsFalse(String),
    /// A `number` field compared with a value.
    Number {
        /// The field.
        field: String,
        /// The comparison.
        op: Compare,
        /// The value, finite.
        value: f64,
    },
    /// A `number` field equals one of the values.
    NumberIn {
        /// The field.
        field: String,
        /// The values, at least one, all finite.
        values: Vec<f64>,
    },
    /// A `number` field is between `low` and `high`, both included.
    Between {
        /// The field.
        field: String,
        /// The lower bound, finite.
        low: f64,
        /// The upper bound, finite and not below `low`.
        high: f64,
    },
    /// A `string` field compared with a value.
    Text {
        /// The field.
        field: String,
        /// The comparison.
        op: TextCompare,
        /// The value.
        value: String,
    },
    /// A `string` field equals one of the values.
    TextIn {
        /// The field.
        field: String,
        /// The values, at least one.
        values: Vec<String>,
    },
    /// A set field holds the element.
    Contains {
        /// The field.
        field: String,
        /// The element. Its kind must be the set's element type.
        value: AttrValue,
    },
    /// A set field holds at least one of the elements.
    ContainsAny {
        /// The field.
        field: String,
        /// The elements, at least one.
        values: Vec<AttrValue>,
    },
    /// A set field holds every one of the elements.
    ContainsAll {
        /// The field.
        field: String,
        /// The elements, at least one.
        values: Vec<AttrValue>,
    },
    /// A set field is present and has no elements. A null field is not empty.
    IsEmpty(String),
}

impl Predicate {
    /// A `bool` field is `true`.
    pub fn is_true(field: impl Into<String>) -> Self {
        Self::IsTrue(field.into())
    }

    /// A `bool` field is `false`.
    pub fn is_false(field: impl Into<String>) -> Self {
        Self::IsFalse(field.into())
    }

    /// A `number` field compared with `value`.
    pub fn number(field: impl Into<String>, op: Compare, value: f64) -> Self {
        Self::Number {
            field: field.into(),
            op,
            value,
        }
    }

    /// A `string` field compared with `value`.
    pub fn text(field: impl Into<String>, op: TextCompare, value: impl Into<String>) -> Self {
        Self::Text {
            field: field.into(),
            op,
            value: value.into(),
        }
    }

    /// Every condition holds.
    pub fn and(items: Vec<Predicate>) -> Self {
        Self::And(items)
    }

    /// At least one condition holds.
    pub fn or(items: Vec<Predicate>) -> Self {
        Self::Or(items)
    }

    /// The condition does not hold. Named `negate` because `not` is
    /// [`std::ops::Not`]'s method.
    pub fn negate(inner: Predicate) -> Self {
        Self::Not(Box::new(inner))
    }

    /// Check the predicate against the schema, without touching storage.
    ///
    /// # Errors
    ///
    /// - [`ItzmaError::UnknownAttr`] for a field the schema does not declare.
    /// - [`ItzmaError::AttrTypeMismatch`] for an operator that does not fit the
    ///   field's type (`expected` is what the operator needs).
    /// - [`ItzmaError::InvalidQuery`] for an empty `And`, `Or`, `NumberIn`,
    ///   `TextIn`, `ContainsAny` or `ContainsAll`, a number that is not finite,
    ///   or a `Between` whose `low` is above its `high`.
    pub(crate) fn check(&self, schema: &AttrSchema) -> Result<()> {
        match self {
            Self::And(items) | Self::Or(items) => {
                if items.is_empty() {
                    return Err(ItzmaError::InvalidQuery(
                        "and and or need at least one condition",
                    ));
                }
                items.iter().try_for_each(|item| item.check(schema))
            }
            Self::Not(inner) => inner.check(schema),
            Self::IsNull(field) | Self::IsNotNull(field) => declared(schema, field).map(|_| ()),
            Self::IsTrue(field) | Self::IsFalse(field) => expect(schema, field, AttrType::Bool),
            Self::Number { field, value, .. } => {
                expect(schema, field, AttrType::Number)?;
                finite(*value)
            }
            Self::NumberIn { field, values } => {
                expect(schema, field, AttrType::Number)?;
                non_empty(values.len())?;
                values.iter().try_for_each(|v| finite(*v))
            }
            Self::Between { field, low, high } => {
                expect(schema, field, AttrType::Number)?;
                finite(*low)?;
                finite(*high)?;
                if low > high {
                    return Err(ItzmaError::InvalidQuery("between: low is above high"));
                }
                Ok(())
            }
            Self::Text { field, .. } => expect(schema, field, AttrType::Text),
            Self::TextIn { field, values } => {
                expect(schema, field, AttrType::Text)?;
                non_empty(values.len())
            }
            Self::Contains { field, value } => elements(schema, field, std::slice::from_ref(value)),
            Self::ContainsAny { field, values } | Self::ContainsAll { field, values } => {
                non_empty(values.len())?;
                elements(schema, field, values)
            }
            Self::IsEmpty(field) => {
                let ty = declared(schema, field)?;
                if matches!(ty, AttrType::TextSet | AttrType::NumberSet) {
                    Ok(())
                } else {
                    Err(not_a_set(field, ty))
                }
            }
        }
    }

    /// Whether a document with these stored attributes matches. Only right for
    /// a predicate that passed [`check`](Self::check). A stored value that does
    /// not fit its field's type counts as null: this build cannot write one, so
    /// it can only come from a hand-made object, and failing a whole scan on it
    /// would be worse than skipping it.
    pub(crate) fn matches(&self, attrs: Option<&Value>) -> bool {
        match self {
            Self::And(items) => items.iter().all(|item| item.matches(attrs)),
            Self::Or(items) => items.iter().any(|item| item.matches(attrs)),
            Self::Not(inner) => !inner.matches(attrs),
            Self::IsNull(field) => value_of(attrs, field).is_none(),
            Self::IsNotNull(field) => value_of(attrs, field).is_some(),
            Self::IsTrue(field) => value_of(attrs, field) == Some(&Value::Bool(true)),
            Self::IsFalse(field) => value_of(attrs, field) == Some(&Value::Bool(false)),
            Self::Number { field, op, value } => value_of(attrs, field)
                .and_then(Value::as_f64)
                .is_some_and(|x| op.apply(x, *value)),
            Self::NumberIn { field, values } => value_of(attrs, field)
                .and_then(Value::as_f64)
                .is_some_and(|x| values.contains(&x)),
            Self::Between { field, low, high } => value_of(attrs, field)
                .and_then(Value::as_f64)
                .is_some_and(|x| *low <= x && x <= *high),
            Self::Text { field, op, value } => value_of(attrs, field)
                .and_then(Value::as_str)
                .is_some_and(|s| match op {
                    TextCompare::Eq => s == value,
                    TextCompare::Ne => s != value,
                    TextCompare::StartsWith => s.starts_with(value.as_str()),
                }),
            Self::TextIn { field, values } => value_of(attrs, field)
                .and_then(Value::as_str)
                .is_some_and(|s| values.iter().any(|v| v == s)),
            Self::Contains { field, value } => {
                set_of(attrs, field).is_some_and(|items| holds(items, value))
            }
            Self::ContainsAny { field, values } => {
                set_of(attrs, field).is_some_and(|items| values.iter().any(|v| holds(items, v)))
            }
            Self::ContainsAll { field, values } => {
                set_of(attrs, field).is_some_and(|items| values.iter().all(|v| holds(items, v)))
            }
            Self::IsEmpty(field) => set_of(attrs, field).is_some_and(<[Value]>::is_empty),
        }
    }
}

/// The field's value, or `None` if the attributes have none or it is `null`.
fn value_of<'a>(attrs: Option<&'a Value>, field: &str) -> Option<&'a Value> {
    attrs?.get(field).filter(|v| !v.is_null())
}

fn set_of<'a>(attrs: Option<&'a Value>, field: &str) -> Option<&'a [Value]> {
    value_of(attrs, field)?.as_array().map(Vec::as_slice)
}

fn holds(items: &[Value], value: &AttrValue) -> bool {
    match value {
        AttrValue::Number(n) => items.iter().any(|v| v.as_f64() == Some(*n)),
        AttrValue::Text(s) => items.iter().any(|v| v.as_str() == Some(s.as_str())),
    }
}

fn declared(schema: &AttrSchema, field: &str) -> Result<AttrType> {
    schema.get(field).ok_or_else(|| ItzmaError::UnknownAttr {
        field: field.to_string(),
    })
}

fn expect(schema: &AttrSchema, field: &str, want: AttrType) -> Result<()> {
    let found = declared(schema, field)?;
    if found == want {
        return Ok(());
    }
    Err(ItzmaError::AttrTypeMismatch {
        field: field.to_string(),
        expected: want.name(),
        found: found.name(),
    })
}

fn not_a_set(field: &str, found: AttrType) -> ItzmaError {
    ItzmaError::AttrTypeMismatch {
        field: field.to_string(),
        expected: "set<string> or set<number>",
        found: found.name(),
    }
}

/// `values` against the element type of the set field.
fn elements(schema: &AttrSchema, field: &str, values: &[AttrValue]) -> Result<()> {
    let ty = declared(schema, field)?;
    let element = match ty {
        AttrType::TextSet => "string",
        AttrType::NumberSet => "number",
        other => return Err(not_a_set(field, other)),
    };
    for value in values {
        let found = match (ty, value) {
            (AttrType::TextSet, AttrValue::Text(_)) => continue,
            (AttrType::NumberSet, AttrValue::Number(n)) => {
                finite(*n)?;
                continue;
            }
            (_, AttrValue::Number(_)) => "number",
            (_, AttrValue::Text(_)) => "string",
        };
        return Err(ItzmaError::AttrTypeMismatch {
            field: field.to_string(),
            expected: element,
            found,
        });
    }
    Ok(())
}

fn finite(value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ItzmaError::InvalidQuery(
            "a number in a predicate must be finite",
        ))
    }
}

fn non_empty(len: usize) -> Result<()> {
    if len == 0 {
        return Err(ItzmaError::InvalidQuery(
            "a list in a predicate needs at least one value",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use Compare::*;

    fn schema() -> AttrSchema {
        AttrSchema::new()
            .field("active", AttrType::Bool)
            .field("price", AttrType::Number)
            .field("name", AttrType::Text)
            .field("tags", AttrType::TextSet)
            .field("scores", AttrType::NumberSet)
    }

    fn text(s: &str) -> AttrValue {
        AttrValue::Text(s.into())
    }

    fn num(n: f64) -> AttrValue {
        AttrValue::Number(n)
    }

    fn ok(p: &Predicate) {
        p.check(&schema()).expect("valid predicate");
    }

    fn eval(p: &Predicate, attrs: Value) -> bool {
        ok(p);
        p.matches(Some(&attrs))
    }

    #[test]
    fn bool_operators_and_null() {
        let t = Predicate::is_true("active");
        let f = Predicate::is_false("active");
        assert!(eval(&t, json!({"active": true})));
        assert!(!eval(&t, json!({"active": false})));
        assert!(eval(&f, json!({"active": false})));
        for null in [json!({}), json!({"active": null})] {
            assert!(
                !eval(&t, null.clone()) && !eval(&f, null),
                "null is neither"
            );
        }
        assert!(!t.matches(None) && !f.matches(None));
    }

    #[test]
    fn number_comparators_at_the_edges() {
        let attrs = json!({"price": 10});
        for (op, at, want) in [
            (Eq, 10.0, true),
            (Eq, 11.0, false),
            (Ne, 10.0, false),
            (Ne, 11.0, true),
            (Lt, 10.0, false),
            (Lt, 10.5, true),
            (Le, 10.0, true),
            (Le, 9.9, false),
            (Gt, 10.0, false),
            (Gt, 9.9, true),
            (Ge, 10.0, true),
            (Ge, 10.1, false),
        ] {
            let p = Predicate::number("price", op, at);
            assert_eq!(eval(&p, attrs.clone()), want, "{op:?} {at}");
        }
    }

    #[test]
    fn number_comparison_against_null_is_false_for_every_operator_including_ne() {
        for op in [Eq, Ne, Lt, Le, Gt, Ge] {
            let p = Predicate::number("price", op, 1.0);
            assert!(!eval(&p, json!({})), "{op:?}");
            assert!(!eval(&p, json!({"price": null})), "{op:?}");
        }
    }

    #[test]
    fn negative_zero_equals_zero_and_big_numbers_compare() {
        let zero = Predicate::number("price", Eq, 0.0);
        assert!(eval(&zero, json!({"price": -0.0})));
        let big = Predicate::number("price", Gt, 1e300);
        assert!(eval(&big, json!({"price": 1.5e300})));
        // Integers above 2^53 lose precision: the two differ as integers and
        // are equal as f64.
        let p = Predicate::number("price", Eq, 9_007_199_254_740_993.0);
        assert!(eval(&p, json!({"price": 9_007_199_254_740_992_u64})));
    }

    #[test]
    fn number_in_and_between_include_their_bounds() {
        let inn = Predicate::NumberIn {
            field: "price".into(),
            values: vec![1.0, 5.0],
        };
        assert!(eval(&inn, json!({"price": 5})));
        assert!(!eval(&inn, json!({"price": 4})));
        assert!(!eval(&inn, json!({})));
        let between = Predicate::Between {
            field: "price".into(),
            low: 2.0,
            high: 4.0,
        };
        for (x, want) in [
            (1.9, false),
            (2.0, true),
            (3.0, true),
            (4.0, true),
            (4.1, false),
        ] {
            assert_eq!(eval(&between, json!({ "price": x })), want, "{x}");
        }
        assert!(!eval(&between, json!({})));
    }

    #[test]
    fn text_operators() {
        let eq = Predicate::text("name", TextCompare::Eq, "abc");
        let ne = Predicate::text("name", TextCompare::Ne, "abc");
        let starts = Predicate::text("name", TextCompare::StartsWith, "ab");
        assert!(eval(&eq, json!({"name": "abc"})) && !eval(&eq, json!({"name": "abd"})));
        assert!(!eval(&ne, json!({"name": "abc"})) && eval(&ne, json!({"name": "abd"})));
        assert!(eval(&starts, json!({"name": "abc"})) && !eval(&starts, json!({"name": "b"})));
        assert!(!eval(&eq, json!({})) && !eval(&ne, json!({})) && !eval(&starts, json!({})));
        let empty = Predicate::text("name", TextCompare::StartsWith, "");
        assert!(eval(&empty, json!({"name": ""})) && eval(&empty, json!({"name": "x"})));
        assert!(
            !eval(&empty, json!({})),
            "an empty prefix still needs a value"
        );
        let inn = Predicate::TextIn {
            field: "name".into(),
            values: vec!["a".into(), "b".into()],
        };
        assert!(eval(&inn, json!({"name": "b"})) && !eval(&inn, json!({"name": "c"})));
    }

    #[test]
    fn set_operators_treat_the_array_as_a_set() {
        let attrs = json!({"tags": ["a", "b", "a"], "scores": [1, 2.5]});
        let has = |field: &str, value| Predicate::Contains {
            field: field.into(),
            value,
        };
        assert!(eval(&has("tags", text("a")), attrs.clone()));
        assert!(!eval(&has("tags", text("z")), attrs.clone()));
        assert!(eval(&has("scores", num(2.5)), attrs.clone()));
        let any = Predicate::ContainsAny {
            field: "tags".into(),
            values: vec![text("z"), text("b")],
        };
        let all = Predicate::ContainsAll {
            field: "tags".into(),
            values: vec![text("a"), text("b")],
        };
        let all_missing = Predicate::ContainsAll {
            field: "tags".into(),
            values: vec![text("a"), text("z")],
        };
        assert!(
            eval(&any, attrs.clone()) && eval(&all, attrs.clone()) && !eval(&all_missing, attrs)
        );
    }

    #[test]
    fn empty_sets_and_null_sets() {
        let empty = Predicate::IsEmpty("tags".into());
        assert!(eval(&empty, json!({"tags": []})));
        assert!(!eval(&empty, json!({"tags": ["a"]})));
        assert!(!eval(&empty, json!({})), "a null set is not empty");
        assert!(!eval(&empty, json!({"tags": null})));
        let has = Predicate::Contains {
            field: "tags".into(),
            value: text("a"),
        };
        assert!(!eval(&has, json!({})) && !eval(&has, json!({"tags": []})));
        let all_of_one = Predicate::ContainsAll {
            field: "tags".into(),
            values: vec![text("a")],
        };
        assert!(!eval(&all_of_one, json!({})), "a null set contains nothing");
    }

    #[test]
    fn is_null_and_is_not_null() {
        let null = Predicate::IsNull("price".into());
        let not_null = Predicate::IsNotNull("price".into());
        for missing in [json!({}), json!({"price": null})] {
            assert!(eval(&null, missing.clone()) && !eval(&not_null, missing));
        }
        assert!(!eval(&null, json!({"price": 0})) && eval(&not_null, json!({"price": 0})));
        assert!(null.matches(None) && !not_null.matches(None));
        let empty_set = Predicate::IsNull("tags".into());
        assert!(
            !eval(&empty_set, json!({"tags": []})),
            "an empty set is not null"
        );
    }

    #[test]
    fn and_or_not_and_the_two_valued_null_rule() {
        let cheap = Predicate::number("price", Lt, 10.0);
        let active = Predicate::is_true("active");
        let both = Predicate::and(vec![cheap.clone(), active.clone()]);
        let either = Predicate::or(vec![cheap.clone(), active.clone()]);
        let attrs = json!({"price": 5, "active": false});
        assert!(!eval(&both, attrs.clone()) && eval(&either, attrs));
        // not (price < 10) matches a document with no price: two-valued logic.
        let not_cheap = Predicate::negate(cheap);
        assert!(eval(&not_cheap, json!({})));
        assert!(eval(&not_cheap, json!({"price": 50})));
        assert!(!eval(&not_cheap, json!({"price": 5})));
    }

    #[test]
    fn a_stored_value_of_the_wrong_type_counts_as_null() {
        assert!(!eval(
            &Predicate::number("price", Gt, 0.0),
            json!({"price": "9"})
        ));
        assert!(!eval(
            &Predicate::is_true("active"),
            json!({"active": "yes"})
        ));
        assert!(!eval(
            &Predicate::Contains {
                field: "tags".into(),
                value: text("a")
            },
            json!({"tags": "a"})
        ));
        // Only the typed operators see it as null: the value is there, so
        // `IsNull` is false and `IsNotNull` is true.
        assert!(!eval(
            &Predicate::IsNull("price".into()),
            json!({"price": "9"})
        ));
        assert!(eval(
            &Predicate::IsNotNull("price".into()),
            json!({"price": "9"})
        ));
    }

    #[test]
    fn check_refuses_unknown_fields_and_wrong_operators() {
        let s = schema();
        assert!(matches!(
            Predicate::is_true("nope").check(&s),
            Err(ItzmaError::UnknownAttr { field }) if field == "nope"
        ));
        let wrong = [
            Predicate::is_true("price"),
            Predicate::number("name", Gt, 1.0),
            Predicate::text("price", TextCompare::Eq, "x"),
            Predicate::IsEmpty("name".into()),
            Predicate::Contains {
                field: "price".into(),
                value: num(1.0),
            },
            Predicate::Contains {
                field: "tags".into(),
                value: num(1.0),
            },
            Predicate::Contains {
                field: "scores".into(),
                value: text("a"),
            },
            Predicate::ContainsAny {
                field: "tags".into(),
                values: vec![text("a"), num(1.0)],
            },
        ];
        for p in wrong {
            assert!(
                matches!(p.check(&s), Err(ItzmaError::AttrTypeMismatch { .. })),
                "{p:?}"
            );
        }
    }

    #[test]
    fn check_refuses_malformed_predicates() {
        let s = schema();
        let bad = [
            Predicate::And(vec![]),
            Predicate::Or(vec![]),
            Predicate::number("price", Eq, f64::NAN),
            Predicate::number("price", Eq, f64::INFINITY),
            Predicate::NumberIn {
                field: "price".into(),
                values: vec![],
            },
            Predicate::NumberIn {
                field: "price".into(),
                values: vec![f64::NAN],
            },
            Predicate::Between {
                field: "price".into(),
                low: 2.0,
                high: 1.0,
            },
            Predicate::TextIn {
                field: "name".into(),
                values: vec![],
            },
            Predicate::ContainsAny {
                field: "tags".into(),
                values: vec![],
            },
            Predicate::ContainsAll {
                field: "tags".into(),
                values: vec![],
            },
            Predicate::negate(Predicate::And(vec![])),
        ];
        for p in bad {
            assert!(
                matches!(p.check(&s), Err(ItzmaError::InvalidQuery(_))),
                "{p:?}"
            );
        }
        // A bad condition deep in the tree is found.
        let deep = Predicate::and(vec![
            Predicate::is_true("active"),
            Predicate::is_true("nope"),
        ]);
        assert!(matches!(
            deep.check(&s),
            Err(ItzmaError::UnknownAttr { .. })
        ));
        // low == high is a valid single-point range.
        ok(&Predicate::Between {
            field: "price".into(),
            low: 1.0,
            high: 1.0,
        });
    }
}
