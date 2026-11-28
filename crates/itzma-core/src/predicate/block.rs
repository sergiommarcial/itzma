//! What a version 3 block's zone statistics say about a predicate.
//!
//! [`Predicate::on_block`] answers one of three things for a whole block:
//! no row can match, every row matches, or the rows must be looked at. It
//! follows [`Predicate::matches`] exactly, so a comparison against a null is
//! false and `Not` flips: a block of nulls is `Never` for `price > 10` and
//! `Always` for `not (price > 10)`.
//!
//! The verdict is conservative. Anything the statistics cannot prove is
//! `Maybe`, including a field the block has no entry for.

use super::{AttrValue, Compare, Predicate, TextCompare};
use crate::segment::{Field, TextZone, ZoneEntry, ZoneStats};

/// What a block's statistics prove about a predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tri {
    /// No row of the block matches.
    Never,
    /// Every row of the block matches.
    Always,
    /// Some rows may match; look at them.
    Maybe,
}

impl Tri {
    fn flip(self) -> Self {
        match self {
            Self::Never => Self::Always,
            Self::Always => Self::Never,
            Self::Maybe => Self::Maybe,
        }
    }
}

/// One block's statistics: its fields, one entry each, and its row count.
#[derive(Clone, Copy)]
pub(crate) struct BlockZones<'a> {
    pub(crate) fields: &'a [Field],
    pub(crate) entries: &'a [ZoneEntry],
    pub(crate) rows: usize,
}

impl<'a> BlockZones<'a> {
    fn entry(&self, field: &str) -> Option<&'a ZoneEntry> {
        let at = self.fields.iter().position(|f| f.name == field)?;
        self.entries.get(at)
    }
}

impl Predicate {
    /// Whether no row, every row or some rows of the block can match. Right
    /// for a predicate that passed [`check`](Self::check).
    pub(crate) fn on_block(&self, block: &BlockZones<'_>) -> Tri {
        match self {
            Self::And(items) => {
                let mut all = Tri::Always;
                for item in items {
                    match item.on_block(block) {
                        Tri::Never => return Tri::Never,
                        Tri::Maybe => all = Tri::Maybe,
                        Tri::Always => {}
                    }
                }
                all
            }
            Self::Or(items) => {
                let mut none = Tri::Never;
                for item in items {
                    match item.on_block(block) {
                        Tri::Always => return Tri::Always,
                        Tri::Maybe => none = Tri::Maybe,
                        Tri::Never => {}
                    }
                }
                none
            }
            Self::Not(inner) => inner.on_block(block).flip(),
            Self::IsNull(field) => with(block, field, |e| {
                nulls_verdict(usize::from(e.nulls), block.rows)
            }),
            Self::IsNotNull(field) => with(block, field, |e| {
                nulls_verdict(usize::from(e.nulls), block.rows).flip()
            }),
            Self::IsTrue(field) => with(block, field, |e| match e.stats {
                ZoneStats::Bool { trues } => count_verdict(usize::from(trues), block.rows),
                _ => Tri::Maybe,
            }),
            Self::IsFalse(field) => with(block, field, |e| match e.stats {
                ZoneStats::Bool { trues } => {
                    let falses = block
                        .rows
                        .saturating_sub(usize::from(e.nulls))
                        .saturating_sub(usize::from(trues));
                    count_verdict(falses, block.rows)
                }
                _ => Tri::Maybe,
            }),
            Self::Number { field, op, value } => with(block, field, |e| match e.stats {
                ZoneStats::Number(range) => {
                    number_verdict(e, block.rows, range, |lo, hi, whole| {
                        compare_verdict(*op, *value, lo, hi, whole)
                    })
                }
                _ => Tri::Maybe,
            }),
            Self::NumberIn { field, values } => with(block, field, |e| match e.stats {
                ZoneStats::Number(range) => {
                    number_verdict(e, block.rows, range, |lo, hi, whole| {
                        if values.iter().all(|v| *v < lo || *v > hi) {
                            Tri::Never
                        } else if whole && lo == hi && values.contains(&lo) {
                            Tri::Always
                        } else {
                            Tri::Maybe
                        }
                    })
                }
                _ => Tri::Maybe,
            }),
            Self::Between { field, low, high } => with(block, field, |e| match e.stats {
                ZoneStats::Number(range) => {
                    number_verdict(e, block.rows, range, |lo, hi, whole| {
                        if *high < lo || *low > hi {
                            Tri::Never
                        } else if whole && *low <= lo && hi <= *high {
                            Tri::Always
                        } else {
                            Tri::Maybe
                        }
                    })
                }
                _ => Tri::Maybe,
            }),
            Self::Text { field, op, value } => with(block, field, |e| match &e.stats {
                ZoneStats::Text(range) => {
                    text_verdict(e, block.rows, range.as_ref(), |zone, whole| {
                        text_compare(*op, value, zone, whole)
                    })
                }
                _ => Tri::Maybe,
            }),
            Self::TextIn { field, values } => with(block, field, |e| match &e.stats {
                ZoneStats::Text(range) => {
                    text_verdict(e, block.rows, range.as_ref(), |zone, whole| {
                        if values.iter().all(|v| text_misses(v, zone)) {
                            Tri::Never
                        } else if whole && zone.is_single() && values.contains(&zone.min) {
                            Tri::Always
                        } else {
                            Tri::Maybe
                        }
                    })
                }
                _ => Tri::Maybe,
            }),
            Self::Contains { field, value } => with(block, field, |e| {
                set_verdict(e, |range| {
                    if range.misses(value) {
                        Tri::Never
                    } else {
                        Tri::Maybe
                    }
                })
            }),
            Self::ContainsAny { field, values } => with(block, field, |e| {
                set_verdict(e, |range| {
                    if values.iter().all(|v| range.misses(v)) {
                        Tri::Never
                    } else {
                        Tri::Maybe
                    }
                })
            }),
            Self::ContainsAll { field, values } => with(block, field, |e| {
                set_verdict(e, |range| {
                    if values.iter().any(|v| range.misses(v)) {
                        Tri::Never
                    } else {
                        Tri::Maybe
                    }
                })
            }),
            Self::IsEmpty(field) => with(block, field, |e| match e.stats {
                ZoneStats::NumberSet { empties, .. } | ZoneStats::TextSet { empties, .. } => {
                    count_verdict(usize::from(empties), block.rows)
                }
                _ => Tri::Maybe,
            }),
        }
    }
}

/// `f` over the field's entry, or `Maybe` when the block has none.
fn with(block: &BlockZones<'_>, field: &str, f: impl FnOnce(&ZoneEntry) -> Tri) -> Tri {
    block.entry(field).map_or(Tri::Maybe, f)
}

/// `count` of `rows` rows has the property: none, all or some.
fn count_verdict(count: usize, rows: usize) -> Tri {
    if count == 0 {
        Tri::Never
    } else if count == rows {
        Tri::Always
    } else {
        Tri::Maybe
    }
}

/// Whether a row is null, from the block's null count.
fn nulls_verdict(nulls: usize, rows: usize) -> Tri {
    count_verdict(nulls, rows)
}

/// A number column's verdict: `Never` when no row has a value, else `f` with
/// the range and whether every row has one.
fn number_verdict(
    entry: &ZoneEntry,
    rows: usize,
    range: Option<(f64, f64)>,
    f: impl FnOnce(f64, f64, bool) -> Tri,
) -> Tri {
    match range {
        None => Tri::Never,
        Some((lo, hi)) => f(lo, hi, entry.nulls == 0 && rows > 0),
    }
}

// Exact comparison against the stored bounds is the semantics, as in
// `Compare::apply`.
#[allow(clippy::float_cmp)]
fn compare_verdict(op: Compare, value: f64, lo: f64, hi: f64, whole: bool) -> Tri {
    let (never, always) = match op {
        Compare::Eq => (value < lo || value > hi, whole && lo == hi && lo == value),
        Compare::Ne => (lo == hi && lo == value, whole && (value < lo || value > hi)),
        Compare::Lt => (lo >= value, whole && hi < value),
        Compare::Le => (lo > value, whole && hi <= value),
        Compare::Gt => (hi <= value, whole && lo > value),
        Compare::Ge => (hi < value, whole && lo >= value),
    };
    if never {
        Tri::Never
    } else if always {
        Tri::Always
    } else {
        Tri::Maybe
    }
}

/// A text column's verdict, as [`number_verdict`].
fn text_verdict(
    entry: &ZoneEntry,
    rows: usize,
    range: Option<&TextZone>,
    f: impl FnOnce(&TextZone, bool) -> Tri,
) -> Tri {
    match range {
        None => Tri::Never,
        Some(zone) => f(zone, entry.nulls == 0 && rows > 0),
    }
}

impl TextZone {
    /// Every value is the one string `min`: the bounds meet and the maximum
    /// is not a cut prefix of something longer.
    fn is_single(&self) -> bool {
        !self.max_cut && self.min == self.max
    }

    /// Every value is below `s`: a whole maximum under it, or a cut one that
    /// `s` passes without sharing its prefix.
    fn all_below(&self, s: &str) -> bool {
        if self.max_cut {
            self.max.as_str() < s && !s.starts_with(self.max.as_str())
        } else {
            self.max.as_str() < s
        }
    }

    /// Every value is above `s`, and none of them starts with it (a cut
    /// minimum is a prefix of the real one, so this holds for the real one).
    fn all_above(&self, s: &str) -> bool {
        self.min.as_str() > s && !self.min.starts_with(s)
    }
}

/// `s` cannot be a value of the column: it sorts outside the range.
fn text_misses(s: &str, zone: &TextZone) -> bool {
    s < zone.min.as_str() || zone.all_below(s)
}

fn text_compare(op: TextCompare, value: &str, zone: &TextZone, whole: bool) -> Tri {
    match op {
        TextCompare::Eq => {
            if text_misses(value, zone) {
                Tri::Never
            } else if whole && zone.is_single() && zone.min == value {
                Tri::Always
            } else {
                Tri::Maybe
            }
        }
        TextCompare::Ne => {
            if zone.is_single() && zone.min == value {
                Tri::Never
            } else if whole && text_misses(value, zone) {
                Tri::Always
            } else {
                Tri::Maybe
            }
        }
        TextCompare::StartsWith => {
            if value.is_empty() {
                if whole { Tri::Always } else { Tri::Maybe }
            } else if zone.all_below(value) || zone.all_above(value) {
                Tri::Never
            } else if whole && zone.min.starts_with(value) && zone.max.starts_with(value) {
                Tri::Always
            } else {
                Tri::Maybe
            }
        }
    }
}

/// A set column's element range, number or text.
enum SetRange<'a> {
    Numbers(Option<(f64, f64)>),
    Texts(Option<&'a TextZone>),
}

impl SetRange<'_> {
    /// The element cannot be in any row's set. A value of the wrong kind is
    /// not claimed to miss.
    fn misses(&self, value: &AttrValue) -> bool {
        match (self, value) {
            (Self::Numbers(None), AttrValue::Number(_))
            | (Self::Texts(None), AttrValue::Text(_)) => true,
            (Self::Numbers(Some((lo, hi))), AttrValue::Number(n)) => n < lo || n > hi,
            (Self::Texts(Some(zone)), AttrValue::Text(s)) => text_misses(s, zone),
            _ => false,
        }
    }
}

/// A set column's verdict: `Never` when no row has an element, else `f` over
/// the element range.
fn set_verdict(entry: &ZoneEntry, f: impl FnOnce(&SetRange<'_>) -> Tri) -> Tri {
    match &entry.stats {
        ZoneStats::NumberSet { range, .. } => f(&SetRange::Numbers(*range)),
        ZoneStats::TextSet { range, .. } => f(&SetRange::Texts(range.as_ref())),
        _ => Tri::Maybe,
    }
}

#[cfg(test)]
pub(super) mod tests {
    use serde_json::{Value, json};

    use super::*;
    use crate::attrs::{AttrSchema, AttrType};
    use crate::segment::{ColumnBlock, decode_zone, encode_zone, fields_of};

    pub(crate) fn schema() -> AttrSchema {
        AttrSchema::new()
            .field("b", AttrType::Bool)
            .field("n", AttrType::Number)
            .field("t", AttrType::Text)
            .field("tags", AttrType::TextSet)
            .field("nums", AttrType::NumberSet)
    }

    /// A small deterministic generator, so a failure replays.
    pub(crate) struct Rng(pub(crate) u64);

    impl Rng {
        pub(crate) fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 33
        }
        pub(crate) fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
        pub(crate) fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
            &items[self.below(items.len())]
        }
        pub(crate) fn chance(&mut self, one_in: usize) -> bool {
            self.below(one_in) == 0
        }
    }

    const NUMBERS: [f64; 7] = [-2.0, 0.0, 1.0, 2.0, 3.5, 10.0, 1e9];

    /// Strings that share prefixes, one over the zone's cut length and one
    /// multi-byte, so a cut `max` and a prefix test are both reached.
    pub(crate) fn words() -> Vec<String> {
        vec![
            String::new(),
            "a".into(),
            "ab".into(),
            "abc".into(),
            "b".into(),
            "é".into(),
            "a".repeat(40),
            format!("{}b", "a".repeat(40)),
            format!("{}c", "a".repeat(40)),
            "é".repeat(20),
            "z".into(),
        ]
    }

    pub(crate) fn row(rng: &mut Rng, words: &[String]) -> Option<Value> {
        if rng.chance(8) {
            return None;
        }
        let mut o = serde_json::Map::new();
        if !rng.chance(3) {
            o.insert("b".into(), json!(rng.chance(2)));
        }
        match rng.below(4) {
            0 => {}
            1 => {
                o.insert("n".into(), Value::Null);
            }
            _ => {
                let n = *rng.pick(&NUMBERS);
                // `-0.0` and `0.0` are one value.
                o.insert("n".into(), json!(if rng.chance(5) { -0.0 } else { n }));
            }
        }
        if !rng.chance(3) {
            o.insert("t".into(), json!(rng.pick(words)));
        }
        if !rng.chance(3) {
            let items: Vec<&String> = (0..rng.below(3)).map(|_| rng.pick(words)).collect();
            o.insert("tags".into(), json!(items));
        }
        if !rng.chance(3) {
            let items: Vec<f64> = (0..rng.below(3)).map(|_| *rng.pick(&NUMBERS)).collect();
            o.insert("nums".into(), json!(items));
        }
        Some(Value::Object(o))
    }

    fn number_value(rng: &mut Rng) -> f64 {
        if rng.chance(3) {
            rng.below(14) as f64 / 2.0 - 3.0
        } else {
            *rng.pick(&NUMBERS)
        }
    }

    type Element<'a> = Box<dyn FnMut(&mut Rng) -> AttrValue + 'a>;

    fn leaf(rng: &mut Rng, words: &[String]) -> Predicate {
        use Compare::*;
        let ops = [Eq, Ne, Lt, Le, Gt, Ge];
        let text_ops = [TextCompare::Eq, TextCompare::Ne, TextCompare::StartsWith];
        let attr_text = |rng: &mut Rng| AttrValue::Text(rng.pick(words).clone());
        let attr_num = |rng: &mut Rng| AttrValue::Number(number_value(rng));
        let set_number = rng.chance(2);
        let (set, mut element): (&str, Element<'_>) = if set_number {
            ("nums", Box::new(attr_num))
        } else {
            ("tags", Box::new(attr_text))
        };
        match rng.below(15) {
            0 => Predicate::IsNull(rng.pick(&["b", "n", "t", "tags", "nums"]).to_string()),
            1 => Predicate::IsNotNull(rng.pick(&["b", "n", "t", "tags", "nums"]).to_string()),
            2 => Predicate::IsTrue("b".into()),
            3 => Predicate::IsFalse("b".into()),
            4 | 5 => Predicate::Number {
                field: "n".into(),
                op: *rng.pick(&ops),
                value: number_value(rng),
            },
            6 => Predicate::NumberIn {
                field: "n".into(),
                values: (0..=rng.below(3)).map(|_| number_value(rng)).collect(),
            },
            7 => {
                let (a, b) = (number_value(rng), number_value(rng));
                Predicate::Between {
                    field: "n".into(),
                    low: a.min(b),
                    high: a.max(b),
                }
            }
            8 | 9 => Predicate::Text {
                field: "t".into(),
                op: *rng.pick(&text_ops),
                value: rng.pick(words).clone(),
            },
            10 => Predicate::TextIn {
                field: "t".into(),
                values: (0..=rng.below(3))
                    .map(|_| rng.pick(words).clone())
                    .collect(),
            },
            11 => Predicate::Contains {
                field: set.into(),
                value: element(rng),
            },
            12 => Predicate::ContainsAny {
                field: set.into(),
                values: (0..=rng.below(2)).map(|_| element(rng)).collect(),
            },
            13 => Predicate::ContainsAll {
                field: set.into(),
                values: (0..=rng.below(2)).map(|_| element(rng)).collect(),
            },
            _ => Predicate::IsEmpty(set.into()),
        }
    }

    pub(crate) fn predicate(rng: &mut Rng, words: &[String], depth: usize) -> Predicate {
        if depth == 0 || rng.chance(2) {
            return leaf(rng, words);
        }
        let kids = |rng: &mut Rng| -> Vec<Predicate> {
            (0..=rng.below(2) + 1)
                .map(|_| predicate(rng, words, depth - 1))
                .collect()
        };
        match rng.below(3) {
            0 => Predicate::And(kids(rng)),
            1 => Predicate::Or(kids(rng)),
            _ => Predicate::Not(Box::new(predicate(rng, words, depth - 1))),
        }
    }

    /// A verdict never contradicts the rows: `Never` means no row matches and
    /// `Always` that every row does. The test also counts the verdicts it saw,
    /// so a predicate that only ever says `Maybe` cannot pass it.
    #[test]
    fn a_block_verdict_never_contradicts_the_rows_it_describes() {
        let words = words();
        let schema = schema();
        let fields = fields_of(&schema);
        let mut rng = Rng(7);
        let (mut never, mut always, mut maybe) = (0, 0, 0);
        for case in 0..3000 {
            let rows: Vec<Option<Value>> =
                (0..=rng.below(12)).map(|_| row(&mut rng, &words)).collect();
            let block = ColumnBlock::from_rows(&fields, &rows).unwrap();
            let mut bytes = Vec::new();
            let ranges = block.encode(&mut bytes);
            let mut zone = Vec::new();
            encode_zone(&block, &ranges, &mut zone);
            let entries = decode_zone("seg", &zone, &fields, rows.len()).unwrap();
            let zones = BlockZones {
                fields: &fields,
                entries: &entries,
                rows: rows.len(),
            };
            for _ in 0..12 {
                let p = predicate(&mut rng, &words, 2);
                let matching = rows.iter().filter(|r| p.matches(r.as_ref())).count();
                match p.on_block(&zones) {
                    Tri::Never => {
                        never += 1;
                        assert_eq!(
                            matching, 0,
                            "case {case}: Never, but a row matches\n{p:?}\n{rows:?}"
                        );
                    }
                    Tri::Always => {
                        always += 1;
                        assert_eq!(
                            matching,
                            rows.len(),
                            "case {case}: Always, but a row does not match\n{p:?}\n{rows:?}"
                        );
                    }
                    Tri::Maybe => maybe += 1,
                }
            }
        }
        assert!(
            never > 2000 && always > 500 && maybe > 2000,
            "{never} {always} {maybe}"
        );
    }

    fn verdict(rows: &[Option<Value>], p: &Predicate) -> Tri {
        let schema = schema();
        let fields = fields_of(&schema);
        let block = ColumnBlock::from_rows(&fields, rows).unwrap();
        let mut bytes = Vec::new();
        let ranges = block.encode(&mut bytes);
        let mut zone = Vec::new();
        encode_zone(&block, &ranges, &mut zone);
        let entries = decode_zone("seg", &zone, &fields, rows.len()).unwrap();
        p.on_block(&BlockZones {
            fields: &fields,
            entries: &entries,
            rows: rows.len(),
        })
    }

    fn number(op: Compare, value: f64) -> Predicate {
        Predicate::Number {
            field: "n".into(),
            op,
            value,
        }
    }

    #[test]
    fn a_block_of_nulls_is_never_for_a_comparison_and_always_for_its_negation() {
        let rows = [Some(json!({"n": null})), None, Some(json!({}))];
        let gt = number(Compare::Gt, 10.0);
        assert_eq!(verdict(&rows, &gt), Tri::Never);
        assert_eq!(verdict(&rows, &Predicate::Not(Box::new(gt))), Tri::Always);
        assert_eq!(verdict(&rows, &Predicate::IsNull("n".into())), Tri::Always);
        assert_eq!(
            verdict(&rows, &Predicate::IsNotNull("n".into())),
            Tri::Never
        );
    }

    #[test]
    fn number_ranges_decide_the_edges() {
        let rows = [Some(json!({"n": 2})), Some(json!({"n": 5}))];
        assert_eq!(verdict(&rows, &number(Compare::Gt, 5.0)), Tri::Never);
        assert_eq!(verdict(&rows, &number(Compare::Ge, 5.0)), Tri::Maybe);
        assert_eq!(verdict(&rows, &number(Compare::Ge, 2.0)), Tri::Always);
        assert_eq!(verdict(&rows, &number(Compare::Lt, 2.0)), Tri::Never);
        assert_eq!(verdict(&rows, &number(Compare::Eq, 3.0)), Tri::Maybe);
        assert_eq!(verdict(&rows, &number(Compare::Ne, 9.0)), Tri::Always);
        // A null row stops a comparison from being `Always`.
        let with_null = [Some(json!({"n": 2})), None];
        assert_eq!(verdict(&with_null, &number(Compare::Ge, 2.0)), Tri::Maybe);
        // Both zeros are one point.
        let zero = [Some(json!({"n": -0.0})), Some(json!({"n": 0.0}))];
        assert_eq!(verdict(&zero, &number(Compare::Eq, 0.0)), Tri::Always);
    }

    #[test]
    fn a_cut_text_maximum_is_never_proof_of_an_equal_value() {
        let long = "a".repeat(40);
        let longer = format!("{long}b");
        let rows = [Some(json!({"t": long})), Some(json!({"t": longer}))];
        let eq = |v: &str| Predicate::Text {
            field: "t".into(),
            op: TextCompare::Eq,
            value: v.into(),
        };
        // The stored maximum is a 32 byte prefix: a value that starts with it
        // may still be in the block.
        assert_eq!(
            verdict(&rows, &eq(&format!("{}c", "a".repeat(40)))),
            Tri::Maybe
        );
        assert_eq!(verdict(&rows, &eq("b")), Tri::Never);
        assert_eq!(verdict(&rows, &eq("")), Tri::Never);
        let starts = Predicate::Text {
            field: "t".into(),
            op: TextCompare::StartsWith,
            value: "aaa".into(),
        };
        assert_eq!(verdict(&rows, &starts), Tri::Always);
    }

    #[test]
    fn a_field_the_block_has_no_entry_for_is_maybe() {
        let rows = [Some(json!({"n": 2}))];
        assert_eq!(
            verdict(&rows, &Predicate::IsNull("nope".into())),
            Tri::Maybe
        );
        // A wrong-typed leaf is not claimed to be anything.
        assert_eq!(verdict(&rows, &Predicate::IsTrue("n".into())), Tri::Maybe);
    }
}
