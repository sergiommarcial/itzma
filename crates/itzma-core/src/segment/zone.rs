//! Zone maps for version 3 attribute blocks: per block and field,
//! the column's byte range and its statistics. Written and checked from 5a,
//! read from 5b.

use std::cmp::Ordering;

use super::Reader;
use super::columns::{ColumnBlock, Field, Numbers, Texts, Values, tag};
use super::v2::corrupt;
use crate::attrs::AttrType;
use crate::error::Result;

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
/// point.
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

/// A text column's range in one block: the least and greatest value, each
/// cut to [`ZONE_STRING_BYTES`]. A cut `max` means "at most this prefix
/// followed by anything".
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TextZone {
    pub(crate) min: String,
    pub(crate) max: String,
    /// `max` was cut, so a value can sort above it while starting with it.
    pub(crate) max_cut: bool,
}

/// What one column's zone entry says besides its null count. A range is
/// `None` when the column holds no value (no element, for a set).
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ZoneStats {
    /// Rows that are `true`; `false` is the valued rest.
    Bool {
        trues: u16,
    },
    Number(Option<(f64, f64)>),
    Text(Option<TextZone>),
    /// Valued rows holding no element, and the range over every element.
    NumberSet {
        empties: u16,
        range: Option<(f64, f64)>,
    },
    TextSet {
        empties: u16,
        range: Option<TextZone>,
    },
}

/// One field's entry in one zone block: where its column sits in the
/// attribute block, how many rows have no value, and its statistics.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ZoneEntry {
    /// The column's byte offset in the attribute block.
    pub(crate) offset: u32,
    /// The column's length in bytes.
    pub(crate) len: u32,
    /// Rows that are absent or `null`, which a predicate treats as null.
    pub(crate) nulls: u16,
    pub(crate) stats: ZoneStats,
}

impl ZoneEntry {
    /// Heap and inline bytes this entry holds, for the decoded cache.
    pub(crate) fn weight(&self) -> u64 {
        let text = |t: &Option<TextZone>| t.as_ref().map_or(0, |t| t.min.len() + t.max.len());
        let heap = match &self.stats {
            ZoneStats::Text(t) | ZoneStats::TextSet { range: t, .. } => text(t),
            _ => 0,
        };
        (std::mem::size_of::<Self>() + heap) as u64
    }
}

/// Every field's entry of one zone block of `rows` rows, in field order, as
/// [`encode_zone`] wrote them. The statistics are read as written: the whole
/// decoder recomputes and compares them, and a ranged read trusts them as it
/// trusts the Bloom filter. A statistic that cannot be true (more nulls than
/// rows, a range out of order, a bool count above the valued rows, a box off
/// the globe) is corrupt.
///
/// # Errors
///
/// [`crate::ItzmaError::CorruptSegment`] for a truncated or inconsistent
/// block.
pub(crate) fn decode_zone(
    path: &str,
    block: &[u8],
    fields: &[Field],
    rows: usize,
) -> Result<Vec<ZoneEntry>> {
    let mut r = Reader::new(path, block);
    let len = r.u32()? as usize;
    if len != r.remaining() {
        return Err(corrupt(
            path,
            "a zone block's length does not match its bytes",
        ));
    }
    let mut entries = Vec::with_capacity(fields.len());
    for f in fields {
        if r.u8()? != tag(f.ty) {
            return Err(corrupt(path, "a zone entry's tag is not its field's"));
        }
        let (offset, len) = (r.u32()?, r.u32()?);
        let nulls = r.u16()?;
        let valued = rows
            .checked_sub(usize::from(nulls))
            .ok_or_else(|| corrupt(path, "a zone counts more nulls than rows"))?;
        let stats = match f.ty {
            AttrType::Bool => {
                let trues = r.u16()?;
                if usize::from(trues) > valued {
                    return Err(corrupt(path, "a zone counts more trues than valued rows"));
                }
                ZoneStats::Bool { trues }
            }
            AttrType::Number => ZoneStats::Number(read_numbers(&mut r, path, valued > 0)?),
            AttrType::Text => ZoneStats::Text(read_texts(&mut r, path, valued > 0)?),
            AttrType::NumberSet | AttrType::TextSet => {
                let empties = r.u16()?;
                let filled = valued
                    .checked_sub(usize::from(empties))
                    .ok_or_else(|| corrupt(path, "a zone counts more empty sets than rows"))?;
                if f.ty == AttrType::NumberSet {
                    let range = read_numbers(&mut r, path, filled > 0)?;
                    ZoneStats::NumberSet { empties, range }
                } else {
                    let range = read_texts(&mut r, path, filled > 0)?;
                    ZoneStats::TextSet { empties, range }
                }
            }
        };
        entries.push(ZoneEntry {
            offset,
            len,
            nulls,
            stats,
        });
    }
    if r.remaining() != 0 {
        return Err(corrupt(path, "a zone block has bytes after its last entry"));
    }
    Ok(entries)
}

fn read_numbers(r: &mut Reader<'_>, path: &str, present: bool) -> Result<Option<(f64, f64)>> {
    if !present {
        return Ok(None);
    }
    let (min, max) = (r.f64()?, r.f64()?);
    // A NaN has no order, so it is refused here too.
    if !matches!(
        min.partial_cmp(&max),
        Some(Ordering::Less | Ordering::Equal)
    ) {
        return Err(corrupt(path, "a number zone's range is not ordered"));
    }
    Ok(Some((min, max)))
}

fn read_texts(r: &mut Reader<'_>, path: &str, present: bool) -> Result<Option<TextZone>> {
    if !present {
        return Ok(None);
    }
    let flags = r.u8()?;
    if flags > 1 {
        return Err(corrupt(path, "a string zone has unknown flag bits"));
    }
    let part = |r: &mut Reader<'_>| -> Result<String> {
        let len = usize::from(r.u8()?);
        if len > ZONE_STRING_BYTES {
            return Err(corrupt(path, "a string zone bound is over the cut length"));
        }
        std::str::from_utf8(r.take(len)?)
            .map(str::to_owned)
            .map_err(|_| corrupt(path, "a string zone bound is not UTF-8"))
    };
    let (min, max) = (part(r)?, part(r)?);
    if min > max && flags == 0 {
        return Err(corrupt(path, "a string zone's range is not ordered"));
    }
    Ok(Some(TextZone {
        min,
        max,
        max_cut: flags == 1,
    }))
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;
    use crate::attrs::{AttrSchema, AttrType};
    use crate::error::ItzmaError;
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

    fn mixed() -> AttrSchema {
        AttrSchema::new()
            .field("a_bool", AttrType::Bool)
            .field("b_num", AttrType::Number)
            .field("c_text", AttrType::Text)
            .field("d_tags", AttrType::TextSet)
            .field("e_nums", AttrType::NumberSet)
            .field("z_text", AttrType::Text)
    }

    fn mixed_rows() -> Vec<Option<Value>> {
        vec![
            Some(
                json!({"a_bool": true, "b_num": 1, "c_text": "x", "d_tags": ["p", "q"], "e_nums": [3],
                "z_text": "k"}),
            ),
            Some(json!({"b_num": null, "d_tags": [], "e_nums": []})),
            None,
            Some(json!({"c_text": "yy"})),
        ]
    }

    fn zone_and_ranges(schema: &AttrSchema, rows: &[Option<Value>]) -> (Vec<u8>, Vec<(u32, u32)>) {
        let block = ColumnBlock::from_rows(&fields_of(schema), rows).unwrap();
        let mut bytes = Vec::new();
        let ranges = block.encode(&mut bytes);
        let mut zone = Vec::new();
        encode_zone(&block, &ranges, &mut zone);
        (zone, ranges)
    }

    fn texts(min: &str, max: &str, max_cut: bool) -> Option<TextZone> {
        Some(TextZone {
            min: min.into(),
            max: max.into(),
            max_cut,
        })
    }

    #[test]
    fn a_zone_block_decodes_every_field_with_the_stats_the_encoder_wrote() {
        let schema = mixed();
        let rows = mixed_rows();
        let (zone, ranges) = zone_and_ranges(&schema, &rows);
        let entries = decode_zone("seg", &zone, &fields_of(&schema), rows.len()).unwrap();
        let stats: Vec<(u16, ZoneStats)> =
            entries.iter().map(|e| (e.nulls, e.stats.clone())).collect();
        assert_eq!(
            stats,
            vec![
                (3, ZoneStats::Bool { trues: 1 }),
                (3, ZoneStats::Number(Some((1.0, 1.0)))),
                (2, ZoneStats::Text(texts("x", "yy", false))),
                (
                    2,
                    ZoneStats::TextSet {
                        empties: 1,
                        range: texts("p", "q", false)
                    }
                ),
                (
                    2,
                    ZoneStats::NumberSet {
                        empties: 1,
                        range: Some((3.0, 3.0))
                    }
                ),
                (3, ZoneStats::Text(texts("k", "k", false))),
            ]
        );
        let located: Vec<(u32, u32)> = entries.iter().map(|e| (e.offset, e.len)).collect();
        assert_eq!(located, ranges);
    }

    #[test]
    fn a_column_with_no_values_has_no_range_and_a_cut_max_is_flagged() {
        let schema = AttrSchema::new()
            .field("n", AttrType::Number)
            .field("t", AttrType::Text);
        let long = "é".repeat(40);
        let rows = vec![Some(json!({"t": "a"})), Some(json!({"t": long}))];
        let (zone, _) = zone_and_ranges(&schema, &rows);
        let entries = decode_zone("seg", &zone, &fields_of(&schema), rows.len()).unwrap();
        assert_eq!(
            (entries[0].nulls, entries[0].stats.clone()),
            (2, ZoneStats::Number(None))
        );
        let ZoneStats::Text(Some(text)) = &entries[1].stats else {
            panic!("a text range")
        };
        assert!(text.max_cut && text.max.len() <= ZONE_STRING_BYTES && long.starts_with(&text.max));
        assert_eq!(text.min, "a");
    }

    #[test]
    fn a_stat_that_cannot_be_true_is_corrupt() {
        let schema = AttrSchema::new().field("n", AttrType::Number);
        let rows = [Some(json!({"n": 1})), Some(json!({"n": 5}))];
        let fields = fields_of(&schema);
        let (zone, _) = zone_and_ranges(&schema, &rows);
        // min above max, then a NaN.
        for (min, max) in [(9.0, 5.0), (f64::NAN, 5.0), (1.0, f64::NAN)] {
            let mut bad = zone.clone();
            let at = bad.len() - 16;
            bad[at..at + 8].copy_from_slice(&min.to_le_bytes());
            bad[at + 8..].copy_from_slice(&max.to_le_bytes());
            assert!(
                matches!(
                    decode_zone("seg", &bad, &fields, 2),
                    Err(ItzmaError::CorruptSegment { .. })
                ),
                "{min} {max}"
            );
        }
        let flags = AttrSchema::new().field("b", AttrType::Bool);
        let (mut bad, _) = zone_and_ranges(&flags, &[Some(json!({"b": true})), None]);
        let at = bad.len() - 2;
        bad[at..].copy_from_slice(&le16(9));
        assert!(matches!(
            decode_zone("seg", &bad, &fields_of(&flags), 2),
            Err(ItzmaError::CorruptSegment { .. })
        ));
    }

    #[test]
    fn every_truncation_and_bit_flip_of_a_whole_zone_errors_or_decodes_never_panics() {
        let schema = mixed();
        let rows = mixed_rows();
        let fields = fields_of(&schema);
        let (zone, _) = zone_and_ranges(&schema, &rows);
        for cut in 0..zone.len() {
            assert!(
                decode_zone("seg", &zone[..cut], &fields, rows.len()).is_err(),
                "a zone cut to {cut} bytes decoded"
            );
        }
        for byte in 0..zone.len() {
            for bit in 0..8 {
                let mut bad = zone.clone();
                bad[byte] ^= 1 << bit;
                let _ = decode_zone("seg", &bad, &fields, rows.len());
            }
        }
    }
}
