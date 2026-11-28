//! Row evaluation over a block's typed columns.
//!
//! [`Predicate::on_columns`] answers [`Predicate::matches`] for every row of
//! a [`ColumnBlock`] at once, with no JSON parse: a number is compared as
//! stored, a dictionary string is tested once per distinct entry and each row
//! looks its code up. `matches` is the reference, so a null is a missing
//! field, a comparison against it is false and `Not` flips.

use std::collections::BTreeSet;

use super::{AttrValue, Predicate, TextCompare};

use crate::columnar::Bitmap;
use crate::segment::{Column, ColumnBlock, Values};

impl Predicate {
    /// Every field the predicate names, added to `out`: the columns
    /// [`on_columns`](Self::on_columns) reads.
    pub(crate) fn fields_into<'a>(&'a self, out: &mut BTreeSet<&'a str>) {
        match self {
            Self::And(items) | Self::Or(items) => {
                items.iter().for_each(|item| item.fields_into(out));
            }
            Self::Not(inner) => inner.fields_into(out),
            Self::IsNull(field)
            | Self::IsNotNull(field)
            | Self::IsTrue(field)
            | Self::IsFalse(field)
            | Self::IsEmpty(field) => {
                out.insert(field);
            }
            Self::Number { field, .. }
            | Self::NumberIn { field, .. }
            | Self::Between { field, .. }
            | Self::Text { field, .. }
            | Self::TextIn { field, .. }
            | Self::Contains { field, .. }
            | Self::ContainsAny { field, .. }
            | Self::ContainsAll { field, .. } => {
                out.insert(field);
            }
        }
    }

    /// The rows of `within` that match, in a bitmap of the block's rows.
    /// Right for a predicate that passed [`check`](Self::check).
    pub(crate) fn on_columns(&self, block: &ColumnBlock, within: &Bitmap) -> Bitmap {
        match self {
            Self::And(items) => {
                let mut rows = within.clone();
                for item in items {
                    if rows.is_empty() {
                        break;
                    }
                    rows = item.on_columns(block, &rows);
                }
                rows
            }
            Self::Or(items) => {
                let mut found = Bitmap::new(block.rows());
                let mut left = within.clone();
                for item in items {
                    if left.is_empty() {
                        break;
                    }
                    let hit = item.on_columns(block, &left);
                    left.subtract(&hit);
                    found.union_with(&hit);
                }
                found
            }
            Self::Not(inner) => {
                let mut rows = within.clone();
                rows.subtract(&inner.on_columns(block, within));
                rows
            }
            Self::IsNull(field) => match column(block, field) {
                Some(c) => each_row(block, within, |i| !c.valued.get(i)),
                None => within.clone(),
            },
            Self::IsNotNull(field) => with(block, field, |c| {
                each_row(block, within, |i| c.valued.get(i))
            }),
            Self::IsTrue(field) => with(block, field, |c| match &c.values {
                Values::Bool(bits) => each_row(block, within, |i| bits.get(i)),
                _ => Bitmap::new(block.rows()),
            }),
            Self::IsFalse(field) => with(block, field, |c| match &c.values {
                Values::Bool(bits) => each_row(block, within, |i| c.valued.get(i) && !bits.get(i)),
                _ => Bitmap::new(block.rows()),
            }),
            Self::Number { field, op, value } => {
                numbers(block, within, field, |x| op.apply(x, *value))
            }
            Self::NumberIn { field, values } => {
                numbers(block, within, field, |x| values.contains(&x))
            }
            Self::Between { field, low, high } => {
                numbers(block, within, field, |x| *low <= x && x <= *high)
            }
            Self::Text { field, op, value } => texts(block, within, field, |s| match op {
                TextCompare::Eq => s == value,
                TextCompare::Ne => s != value,
                TextCompare::StartsWith => s.starts_with(value.as_str()),
            }),
            Self::TextIn { field, values } => {
                texts(block, within, field, |s| values.iter().any(|v| v == s))
            }
            Self::Contains { field, value } => sets(
                block,
                within,
                field,
                std::slice::from_ref(value),
                Quantifier::All,
            ),
            Self::ContainsAny { field, values } => {
                sets(block, within, field, values, Quantifier::Any)
            }
            Self::ContainsAll { field, values } => {
                sets(block, within, field, values, Quantifier::All)
            }
            Self::IsEmpty(field) => with(block, field, |c| match &c.values {
                Values::NumberSet { ends, .. } | Values::TextSet { ends, .. } => {
                    each_valued(block, within, c, |j| ends[j] == ends[j + 1])
                }
                _ => Bitmap::new(block.rows()),
            }),
        }
    }
}

/// How a set predicate combines its elements.
#[derive(Clone, Copy)]
enum Quantifier {
    Any,
    All,
}

fn column<'a>(block: &'a ColumnBlock, field: &str) -> Option<&'a Column> {
    let at = block.fields().iter().position(|f| f.name == field)?;
    debug_assert!(
        block.is_loaded(at),
        "{field} was not fetched for this block"
    );
    block.columns().get(at)
}

/// `f` over the field's column, or no row at all when the block has none: a
/// missing field is null, and every operator but `IsNull` is false on null.
fn with(block: &ColumnBlock, field: &str, f: impl FnOnce(&Column) -> Bitmap) -> Bitmap {
    column(block, field).map_or_else(|| Bitmap::new(block.rows()), f)
}

/// The rows of `within` where `test(row)` holds.
fn each_row(block: &ColumnBlock, within: &Bitmap, mut test: impl FnMut(usize) -> bool) -> Bitmap {
    let mut out = Bitmap::new(block.rows());
    for i in 0..block.rows() {
        if within.get(i) && test(i) {
            out.set(i);
        }
    }
    out
}

/// The rows of `within` the column has a value for where `test(j)` holds,
/// `j` being the value's position among the column's values. Every valued
/// row advances `j`, in or out of `within`.
fn each_valued(
    block: &ColumnBlock,
    within: &Bitmap,
    column: &Column,
    mut test: impl FnMut(usize) -> bool,
) -> Bitmap {
    let mut out = Bitmap::new(block.rows());
    let mut j = 0;
    for i in 0..block.rows() {
        if column.valued.get(i) {
            if within.get(i) && test(j) {
                out.set(i);
            }
            j += 1;
        }
    }
    out
}

fn numbers(
    block: &ColumnBlock,
    within: &Bitmap,
    field: &str,
    test: impl Fn(f64) -> bool,
) -> Bitmap {
    with(block, field, |c| match &c.values {
        Values::Number(n) => each_valued(block, within, c, |j| test(n.as_f64(j))),
        _ => Bitmap::new(block.rows()),
    })
}

fn texts(block: &ColumnBlock, within: &Bitmap, field: &str, test: impl Fn(&str) -> bool) -> Bitmap {
    with(block, field, |c| match &c.values {
        Values::Text(t) => {
            let hits = t.matching(test);
            each_valued(block, within, c, |j| hits[j])
        }
        _ => Bitmap::new(block.rows()),
    })
}

/// The set field holds `wanted`: any of them or all of them. An element of
/// the other kind than the set's never matches, as in `Predicate::matches`.
#[allow(clippy::float_cmp)] // exact equality is the semantics, as in `Compare::apply`
fn sets(
    block: &ColumnBlock,
    within: &Bitmap,
    field: &str,
    wanted: &[AttrValue],
    how: Quantifier,
) -> Bitmap {
    with(block, field, |c| match &c.values {
        Values::TextSet { ends, items } => {
            // One pass over the elements per wanted value: `held[k][e]` is
            // whether element `e` equals wanted value `k`.
            let held: Vec<Vec<bool>> = wanted
                .iter()
                .map(|w| match w {
                    AttrValue::Text(s) => items.matching(|x| x == s),
                    AttrValue::Number(_) => vec![false; items.len()],
                })
                .collect();
            set_rows(block, within, c, ends, &held, how)
        }
        Values::NumberSet { ends, items } => {
            let held: Vec<Vec<bool>> = wanted
                .iter()
                .map(|w| match w {
                    AttrValue::Number(n) => {
                        (0..items.len()).map(|e| items.as_f64(e) == *n).collect()
                    }
                    AttrValue::Text(_) => vec![false; items.len()],
                })
                .collect();
            set_rows(block, within, c, ends, &held, how)
        }
        _ => Bitmap::new(block.rows()),
    })
}

fn set_rows(
    block: &ColumnBlock,
    within: &Bitmap,
    column: &Column,
    ends: &[u32],
    held: &[Vec<bool>],
    how: Quantifier,
) -> Bitmap {
    let has =
        |hits: &Vec<bool>, j: usize| hits[ends[j] as usize..ends[j + 1] as usize].contains(&true);
    each_valued(block, within, column, |j| match how {
        Quantifier::Any => held.iter().any(|hits| has(hits, j)),
        Quantifier::All => held.iter().all(|hits| has(hits, j)),
    })
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::super::block::tests::{Rng, predicate, row, schema, words};
    use super::*;
    use crate::attrs::{AttrSchema, AttrType};
    use crate::predicate::{Compare, TextCompare};
    use crate::segment::{Texts, Values, fields_of};

    fn check(
        p: &Predicate,
        rows: &[Option<Value>],
        block: &ColumnBlock,
        within: &Bitmap,
        why: &str,
    ) {
        let got = p.on_columns(block, within);
        assert_eq!(got.len(), rows.len(), "{why}: length\n{p:?}");
        for (i, r) in rows.iter().enumerate() {
            let want = within.get(i) && p.matches(r.as_ref());
            assert_eq!(got.get(i), want, "{why}: row {i}\n{p:?}\n{rows:?}");
        }
    }

    fn mask(rng: &mut Rng, rows: usize) -> Bitmap {
        let mut within = Bitmap::new(rows);
        for i in 0..rows {
            if !rng.chance(3) {
                within.set(i);
            }
        }
        within
    }

    /// `on_columns` is `matches` row by row, for every operator and type,
    /// over a random mask of rows.
    #[test]
    fn on_columns_equals_matches_row_by_row() {
        let words = words();
        let fields = fields_of(&schema());
        let mut rng = Rng(11);
        for case in 0..3000 {
            let n = if case % 50 == 0 { 256 } else { rng.below(13) };
            let rows: Vec<Option<Value>> = (0..n).map(|_| row(&mut rng, &words)).collect();
            let block = ColumnBlock::from_rows(&fields, &rows).unwrap();
            let all = Bitmap::full(n);
            let some = mask(&mut rng, n);
            for _ in 0..12 {
                let p = predicate(&mut rng, &words, 2);
                check(&p, &rows, &block, &all, &format!("case {case} full"));
                check(&p, &rows, &block, &some, &format!("case {case} masked"));
                check(
                    &p,
                    &rows,
                    &block,
                    &Bitmap::new(n),
                    &format!("case {case} empty"),
                );
            }
        }
    }

    /// Dictionary and plain strings answer alike on both sides of the size
    /// edge where the encoder flips from one to the other.
    #[test]
    fn dictionary_and_plain_strings_answer_alike() {
        let schema = AttrSchema::new()
            .field("t", AttrType::Text)
            .field("tags", AttrType::TextSet);
        let fields = fields_of(&schema);
        let (mut dict, mut plain) = (0, 0);
        for distinct in [1, 2, 3, 8, 40, 128, 200, 256] {
            for width in [1, 6, 30] {
                let name = |k: usize| format!("{k:0>width$}", width = width);
                let rows: Vec<Option<Value>> = (0..256)
                    .map(|i| {
                        Some(json!({
                            "t": name(i * 7 % distinct),
                            "tags": [name(i % distinct), name((i + 1) % distinct)],
                        }))
                    })
                    .collect();
                let block = ColumnBlock::from_rows(&fields, &rows).unwrap();
                for column in block.columns() {
                    match &column.values {
                        Values::Text(Texts::Dict { .. })
                        | Values::TextSet {
                            items: Texts::Dict { .. },
                            ..
                        } => dict += 1,
                        _ => plain += 1,
                    }
                }
                let pivot = name(distinct / 2);
                let ps = [
                    Predicate::text("t", TextCompare::Eq, pivot.clone()),
                    Predicate::text("t", TextCompare::Ne, pivot.clone()),
                    Predicate::text("t", TextCompare::StartsWith, &pivot[..1]),
                    Predicate::TextIn {
                        field: "t".into(),
                        values: vec![pivot.clone(), name(0), "none".into()],
                    },
                    Predicate::Contains {
                        field: "tags".into(),
                        value: crate::predicate::AttrValue::Text(pivot.clone()),
                    },
                    Predicate::ContainsAny {
                        field: "tags".into(),
                        values: vec![
                            crate::predicate::AttrValue::Text(pivot.clone()),
                            crate::predicate::AttrValue::Text(name(0)),
                        ],
                    },
                    Predicate::ContainsAll {
                        field: "tags".into(),
                        values: vec![
                            crate::predicate::AttrValue::Text(pivot.clone()),
                            crate::predicate::AttrValue::Text(name(1 % distinct)),
                        ],
                    },
                ];
                for p in &ps {
                    check(
                        p,
                        &rows,
                        &block,
                        &Bitmap::full(256),
                        &format!("{distinct} distinct, width {width}"),
                    );
                }
            }
        }
        assert!(
            dict > 0 && plain > 0,
            "both encodings must be reached: {dict} dict, {plain} plain"
        );
        let _ = Compare::Eq;
    }
}
