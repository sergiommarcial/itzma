//! Zone maps for version 3 attribute blocks: per block and field,
//! the column's byte range and its statistics. Written and checked from 5a,
//! read from 5b.

use std::cmp::Ordering;

use super::columns::{ColumnBlock, Numbers, Texts, Values, tag};

/// Longest `min` or `max` a string zone keeps.
pub(crate) const ZONE_STRING_BYTES: usize = 32;

/// The zone block of a free-form (JSON) attribute block: no entries.
pub(crate) fn encode_empty_zone(out: &mut Vec<u8>) {
    out.extend_from_slice(&0u32.to_le_bytes());
}

/// Append `block`'s zone block, its columns at `ranges`, to `out`.
pub(crate) fn encode_zone(block: &ColumnBlock, ranges: &[(u32, u32)], out: &mut Vec<u8>) {
    let mut entries = Vec::new();
    for ((field, column), (offset, len)) in block.fields().iter().zip(block.columns()).zip(ranges) {
        entries.push(tag(field.ty));
        entries.extend_from_slice(&offset.to_le_bytes());
        entries.extend_from_slice(&len.to_le_bytes());
        let nulls = (block.rows() - column.valued.count()) as u16;
        entries.extend_from_slice(&nulls.to_le_bytes());
        match &column.values {
            Values::Bool(bits) => {
                entries.extend_from_slice(&(bits.count() as u16).to_le_bytes());
            }
            Values::Number(numbers) => put_numbers(&mut entries, numbers),
            Values::Text(texts) => put_texts(&mut entries, texts),
            Values::NumberSet { ends, items } => {
                entries.extend_from_slice(&empties(ends).to_le_bytes());
                put_numbers(&mut entries, items);
            }
            Values::TextSet { ends, items } => {
                entries.extend_from_slice(&empties(ends).to_le_bytes());
                put_texts(&mut entries, items);
            }
        }
    }
    // A block of at most 256 rows and 256 fields stays far below 4 GiB.
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    out.extend_from_slice(&entries);
}

fn empties(ends: &[u32]) -> u16 {
    ends.windows(2).filter(|p| p[0] == p[1]).count() as u16
}

/// The value a predicate compares, with `-0.0` as `0.0` so both zeros are one
/// point (`attr_index::Value::number` does the same).
fn comparable(x: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x }
}

fn put_numbers(out: &mut Vec<u8>, numbers: &Numbers) {
    let mut values = (0..numbers.len()).map(|j| comparable(numbers.as_f64(j)));
    let Some(first) = values.next() else { return };
    let (min, max) = values.fold((first, first), |(lo, hi), x| {
        (
            if x.total_cmp(&lo) == Ordering::Less {
                x
            } else {
                lo
            },
            if x.total_cmp(&hi) == Ordering::Greater {
                x
            } else {
                hi
            },
        )
    });
    out.extend_from_slice(&min.to_le_bytes());
    out.extend_from_slice(&max.to_le_bytes());
}

/// The longest prefix of `s` of at most [`ZONE_STRING_BYTES`] bytes that ends
/// on a character boundary.
fn cut(s: &str) -> &str {
    if s.len() <= ZONE_STRING_BYTES {
        return s;
    }
    let mut end = ZONE_STRING_BYTES;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

fn put_texts(out: &mut Vec<u8>, texts: &Texts) {
    let mut values = (0..texts.len()).map(|j| texts.get(j));
    let Some(first) = values.next() else { return };
    let (min, max) = values.fold((first, first), |(lo, hi), s| (lo.min(s), hi.max(s)));
    let (min_cut, max_cut) = (cut(min), cut(max));
    out.push(u8::from(max_cut.len() < max.len()));
    for s in [min_cut, max_cut] {
        out.push(s.len() as u8);
        out.extend_from_slice(s.as_bytes());
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;
    use crate::attrs::{AttrSchema, AttrType};
    use crate::segment::columns::{ColumnBlock, fields_of};

    fn zone_of(schema: AttrSchema, rows: &[Option<Value>]) -> Vec<u8> {
        let block = ColumnBlock::from_rows(&fields_of(&schema), rows).unwrap();
        let mut bytes = Vec::new();
        let ranges = block.encode(&mut bytes);
        let mut zone = Vec::new();
        encode_zone(&block, &ranges, &mut zone);
        zone
    }

    fn le16(v: u16) -> Vec<u8> {
        v.to_le_bytes().to_vec()
    }
    fn le32(v: u32) -> Vec<u8> {
        v.to_le_bytes().to_vec()
    }
    fn le64f(v: f64) -> Vec<u8> {
        v.to_le_bytes().to_vec()
    }

    #[test]
    fn a_number_column_keeps_nulls_min_and_max_with_negative_zero_as_zero() {
        let zone = zone_of(
            AttrSchema::new().field("n", AttrType::Number),
            &[
                Some(json!({"n": 3})),
                Some(json!({"n": -0.0})),
                None,
                Some(json!({"n": null})),
            ],
        );
        // Block: u32 count, objects (1 byte), then the column at offset 5:
        // present, valued (1 byte each), 2 kinds, 2 values = 20 bytes.
        let entry = [vec![1], le32(5), le32(20), le16(2), le64f(0.0), le64f(3.0)].concat();
        assert_eq!(zone, [le32(entry.len() as u32), entry].concat());
    }

    #[test]
    fn a_null_only_column_has_no_min_or_max() {
        let zone = zone_of(
            AttrSchema::new().field("n", AttrType::Number),
            &[Some(json!({"n": null})), None],
        );
        assert_eq!(zone[4], 1);
        assert_eq!(&zone[13..15], &le16(2)[..]);
        assert_eq!(zone.len(), 15);
    }

    #[test]
    fn bools_count_trues_and_nulls() {
        let zone = zone_of(
            AttrSchema::new().field("b", AttrType::Bool),
            &[
                Some(json!({"b": true})),
                Some(json!({"b": false})),
                Some(json!({})),
                Some(json!({"b": true})),
            ],
        );
        assert_eq!(&zone[13..], &[le16(1), le16(2)].concat()[..]);
    }

    #[test]
    fn strings_are_cut_on_a_character_boundary_and_a_cut_max_is_flagged() {
        let long = format!("{}é", "a".repeat(31)); // 33 bytes, é straddles 32
        let zone = zone_of(
            AttrSchema::new().field("s", AttrType::Text),
            &[Some(json!({"s": "b"})), Some(json!({"s": long}))],
        );
        let stats = &zone[13..];
        // nulls 0, flag 1 (max cut), min "a…é" cut to 31 bytes, max "b".
        assert_eq!(&stats[..2], &le16(0)[..]);
        assert_eq!(stats[2], 0, "the max, \"b\", is not cut");
        assert_eq!(stats[3] as usize, 31, "the min is cut before é");
        let max_at = 4 + 31;
        assert_eq!(&stats[max_at..], &[1, b'b'][..]);
        let zone = zone_of(
            AttrSchema::new().field("s", AttrType::Text),
            &[Some(json!({"s": long}))],
        );
        assert_eq!(zone[13 + 2], 1, "a single value that is cut is a cut max");
    }

    #[test]
    fn sets_count_empties_and_range_over_every_element() {
        let zone = zone_of(
            AttrSchema::new().field("ns", AttrType::NumberSet),
            &[
                Some(json!({"ns": []})),
                Some(json!({"ns": [4, -2]})),
                Some(json!({"ns": null})),
            ],
        );
        assert_eq!(
            &zone[13..],
            &[le16(1), le16(1), le64f(-2.0), le64f(4.0)].concat()[..]
        );
    }

    #[test]
    fn an_empty_zone_is_a_zero_length() {
        let mut out = Vec::new();
        encode_empty_zone(&mut out);
        assert_eq!(out, le32(0));
    }

    #[test]
    fn a_block_with_no_fields_has_an_empty_zone() {
        assert_eq!(
            zone_of(AttrSchema::new(), &[None, Some(json!({}))]),
            le32(0)
        );
    }
}
