//! A version 2 or 3 segment held for a flat scan: its ids in one
//! buffer, its vectors as one `f32` matrix decoded once, and its attributes as
//! JSON rows copied out (version 2, or a free-form namespace at 3) or as the
//! decoded column blocks (version 3), turned into a value one row at
//! a time when a filter or a hit needs it.
//! It keeps none of the fetched object, so its weight in the decoded cache is
//! what it holds ([`SegmentView::weight`]).
//!
//! The checks are `v2::decode`'s, except two. The Bloom filter is not rebuilt
//! and compared: a view never reads it (`holds` is a binary search), and
//! `RangedSegment` already trusts it unchecked. Attribute JSON is not parsed
//! here: damage there is `CorruptSegment` from [`SegmentView::attrs`]. Column
//! blocks are decoded and checked, but the zone section is not compared with
//! them, the trust `RangedSegment` gives it too.

use bytes::Bytes;

use super::columns::ColumnBlock;
use super::{MAGIC, Reader, v2};
use crate::columnar::IdColumn;
use crate::error::Result;
use crate::layer::SortedIds;

pub(crate) struct SegmentView {
    path: String,
    dim: usize,
    ids: IdColumn,
    tombstones: IdColumn,
    matrix: Vec<f32>,
    attrs: ViewAttrs,
}

/// A view's attributes, copied out of the object so the view does not keep
/// the whole object alive.
enum ViewAttrs {
    /// Every row's JSON one after another, and where each ends; an empty row
    /// is no attributes.
    Json { text: Vec<u8>, ends: Vec<usize> },
    /// One decoded column block per id block.
    Columns(Vec<ColumnBlock>),
}

impl SegmentView {
    /// Decode `bytes`, a whole version 2 or 3 object.
    ///
    /// # Errors
    ///
    /// [`ItzmaError::CorruptSegment`](crate::error::ItzmaError::CorruptSegment)
    /// for anything `v2::decode` refuses apart from the Bloom filter and
    /// attribute JSON.
    /// [`ItzmaError::UnsupportedFormat`](crate::error::ItzmaError::UnsupportedFormat)
    /// for a version other than 2 or 3.
    pub(crate) fn decode(path: &str, bytes: Bytes, dim: usize) -> Result<Self> {
        let mut r = Reader::new(path, &bytes);
        if r.take(4)? != MAGIC {
            return Err(r.corrupt("bad magic"));
        }
        let version = r.u32()?;
        if !matches!(version, 2 | 3) {
            return Err(crate::error::ItzmaError::UnsupportedFormat { version });
        }
        let parts = v2::parts(r, version, dim)?;
        let too_long =
            |rows: &[&[u8]]| rows.iter().map(|row| row.len()).sum::<usize>() > u32::MAX as usize;
        if too_long(&parts.ids) || too_long(&parts.tombstones) {
            return Err(v2::corrupt(path, "ids are longer than a view holds"));
        }
        let ids = column(path, &parts.ids, "id")?;
        let tombstones = column(path, &parts.tombstones, "tombstone")?;
        if !ids.is_strictly_ascending() {
            return Err(v2::corrupt(path, "ids are not in strictly ascending order"));
        }
        if !tombstones.is_strictly_ascending() {
            return Err(v2::corrupt(
                path,
                "tombstones are not in strictly ascending order",
            ));
        }
        if overlap(&ids, &tombstones) {
            return Err(v2::corrupt(
                path,
                "an id is both a document and a tombstone",
            ));
        }
        let matrix = parts
            .matrix
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let attrs = match &parts.attrs {
            v2::AttrSection::Rows(rows) => {
                let mut text = Vec::with_capacity(rows.iter().map(|row| row.len()).sum());
                let mut ends = Vec::with_capacity(rows.len());
                for row in rows {
                    text.extend_from_slice(row);
                    ends.push(text.len());
                }
                ViewAttrs::Json { text, ends }
            }
            v2::AttrSection::Columns { fields, blocks } => {
                let docs = ids.len();
                let columns = blocks
                    .iter()
                    .enumerate()
                    .map(|(i, block)| {
                        let rows = (docs - i * v2::BLOCK_ROWS).min(v2::BLOCK_ROWS);
                        ColumnBlock::decode(path, fields, block, rows).map(|(c, _)| c)
                    })
                    .collect::<Result<_>>()?;
                ViewAttrs::Columns(columns)
            }
        };
        Ok(Self {
            path: path.to_owned(),
            dim,
            ids,
            tombstones,
            matrix,
            attrs,
        })
    }

    /// Documents in the segment.
    pub(crate) fn rows(&self) -> usize {
        self.ids.len()
    }

    /// The bytes the view holds, its weight in the decoded cache: the
    /// matrix, the ids and tombstones, the attribute rows and their offsets.
    pub(crate) fn weight(&self) -> u64 {
        let held = self.matrix.len() * 4
            + self.ids.heap_bytes()
            + self.tombstones.heap_bytes()
            + match &self.attrs {
                ViewAttrs::Json { text, ends } => {
                    text.len() + ends.len() * std::mem::size_of::<usize>()
                }
                ViewAttrs::Columns(blocks) => {
                    blocks.iter().map(|b| b.weight() as usize).sum::<usize>()
                }
            };
        held as u64
    }

    pub(crate) fn id(&self, row: usize) -> &str {
        self.ids.get(row)
    }

    pub(crate) fn vector(&self, row: usize) -> &[f32] {
        &self.matrix[row * self.dim..(row + 1) * self.dim]
    }

    /// Row `row`'s attributes, parsed now.
    ///
    /// # Errors
    ///
    /// [`ItzmaError::CorruptSegment`](crate::error::ItzmaError::CorruptSegment)
    /// if the stored JSON does not parse.
    pub(crate) fn attrs(&self, row: usize) -> Result<Option<serde_json::Value>> {
        let (text, ends) = match &self.attrs {
            ViewAttrs::Json { text, ends } => (text, ends),
            ViewAttrs::Columns(blocks) => {
                return Ok(blocks[row / v2::BLOCK_ROWS].row(row % v2::BLOCK_ROWS));
            }
        };
        let start = if row == 0 { 0 } else { ends[row - 1] };
        let raw = &text[start..ends[row]];
        if raw.is_empty() {
            return Ok(None);
        }
        serde_json::from_slice(raw)
            .map(Some)
            .map_err(|e| v2::corrupt(&self.path, format!("attributes are not valid JSON: {e}")))
    }

    /// Whether the segment holds `id` as a document or a tombstone.
    pub(crate) fn holds(&self, id: &str) -> bool {
        self.ids.contains(id) || self.tombstones.contains(id)
    }

    pub(crate) fn sorted_ids(&self) -> SortedIds<'_> {
        SortedIds {
            docs: &self.ids,
            tombstones: &self.tombstones,
        }
    }
}

fn column(path: &str, rows: &[&[u8]], what: &str) -> Result<IdColumn> {
    let ids = rows
        .iter()
        .map(|row| {
            std::str::from_utf8(row)
                .map_err(|_| v2::corrupt(path, format!("{what} is not valid UTF-8")))
        })
        .collect::<Result<Vec<&str>>>()?;
    Ok(IdColumn::from_sorted(ids))
}

/// Whether two ascending columns share an id. One pass over both.
fn overlap(a: &IdColumn, b: &IdColumn) -> bool {
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a.get(i).cmp(b.get(j)) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => return true,
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use serde_json::json;

    use super::*;
    use crate::document::{Document, WalOp};
    use crate::error::ItzmaError;
    use crate::layer::Layer;
    use crate::segment::{decode as decode_whole, encode_as};

    const DIM: usize = 3;

    fn sample(n: usize, tombstones: usize) -> Layer {
        let mut layer = Layer::new();
        for i in 0..n {
            layer.apply(WalOp::Upsert(Document {
                id: format!("d{i:05}"),
                vector: vec![i as f32, -(i as f32), 0.5],
                attrs: (i % 3 != 0).then(|| json!({"n": i, "tag": format!("t{}", i % 7)})),
            }));
        }
        for i in 0..tombstones {
            layer.apply(WalOp::Delete(format!("t{i:05}")));
        }
        layer
    }

    fn bytes_of(layer: &Layer) -> Bytes {
        encode_as(layer, DIM, 2).unwrap()
    }

    fn assert_agrees(view: &SegmentView, whole: &Layer) {
        let docs = whole.sorted_docs();
        assert_eq!(view.rows(), docs.len());
        for (row, doc) in docs.iter().enumerate() {
            assert_eq!(view.id(row), doc.id);
            assert_eq!(view.vector(row), doc.vector.as_slice());
            assert_eq!(view.attrs(row).unwrap(), doc.attrs);
            assert!(view.holds(&doc.id));
        }
        for id in whole.sorted_tombstones() {
            assert!(view.holds(id));
        }
        assert!(!view.holds("absent"));
        let ids = view.sorted_ids();
        assert_eq!(ids.docs.len(), docs.len());
        assert_eq!(ids.tombstones.len(), whole.tombstone_count());
    }

    #[test]
    fn a_view_agrees_with_the_whole_decoder_at_block_edges() {
        for (n, t) in [
            (0, 0),
            (0, 3),
            (1, 0),
            (255, 1),
            (256, 256),
            (257, 513),
            (1000, 7),
        ] {
            let layer = sample(n, t);
            let bytes = bytes_of(&layer);
            let view = SegmentView::decode("p", bytes.clone(), DIM).unwrap();
            assert_agrees(&view, &decode_whole("p", &bytes, DIM).unwrap());
        }
    }

    /// A view's weight in the decoded cache is what it holds: the decoded
    /// matrix, the ids and the attribute rows, not the object it came from,
    /// which it does not keep.
    #[test]
    fn a_views_weight_is_what_it_holds() {
        let layer = sample(1000, 7);
        let bytes = bytes_of(&layer);
        let view = SegmentView::decode("p", bytes.clone(), DIM).unwrap();
        let ids: usize = layer.sorted_docs().iter().map(|d| d.id.len()).sum();
        let attrs: usize = layer
            .sorted_docs()
            .iter()
            .filter_map(|d| d.attrs.as_ref())
            .map(|a| serde_json::to_vec(a).unwrap().len())
            .sum();
        let floor = (1000 * DIM * 4 + ids + attrs) as u64;
        assert!(view.weight() >= floor, "{} < {floor}", view.weight());
        // Offsets per row on top of that, nothing like a second object.
        assert!(
            view.weight() < floor + 1000 * 24 + 7 * 16,
            "{} for a floor of {floor}",
            view.weight()
        );
        drop(bytes);
        assert_agrees(&view, &decode_whole("p", &bytes_of(&layer), DIM).unwrap());
    }

    #[test]
    fn a_version_1_object_is_not_a_view() {
        let mut bytes = encode_as(&sample(5, 1), DIM, 2).unwrap().to_vec();
        bytes[4..8].copy_from_slice(&1u32.to_le_bytes());
        let err = SegmentView::decode("p", bytes.into(), DIM).err().unwrap();
        assert!(
            matches!(
                err,
                ItzmaError::CorruptSegment { .. } | ItzmaError::UnsupportedFormat { .. }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn a_wrong_dimension_is_refused() {
        let bytes = bytes_of(&sample(5, 0));
        assert!(SegmentView::decode("p", bytes, DIM + 1).is_err());
    }

    #[test]
    fn every_truncation_is_refused() {
        let bytes = bytes_of(&sample(300, 5));
        for len in 0..bytes.len() {
            assert!(
                SegmentView::decode("p", bytes.slice(..len), DIM).is_err(),
                "len {len}"
            );
        }
    }

    /// Where the Bloom filter sits: the directory's second and third u64.
    fn bloom_range(bytes: &[u8]) -> std::ops::Range<usize> {
        let u64_at = |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap()) as usize;
        let directory = u64_at(bytes.len() - 20);
        let start = u64_at(directory + 8);
        start..start + u64_at(directory + 16)
    }

    /// A flipped bit outside the Bloom filter is refused, or the view still
    /// agrees with the whole decoder, or (attribute JSON) the whole decoder
    /// refuses it and the view refuses that row when it is read, or (an id
    /// that stays valid and sorted) only the Bloom comparison would have seen
    /// it. Never a panic.
    #[test]
    fn every_bit_flip_outside_the_bloom_filter_is_caught_or_harmless() {
        let original = bytes_of(&sample(40, 3));
        let bloom = bloom_range(&original);
        for byte in 0..original.len() {
            if bloom.contains(&byte) {
                // Never read by a view: `holds` is a binary search.
                continue;
            }
            for bit in 0..8 {
                let mut flipped = original.to_vec();
                flipped[byte] ^= 1 << bit;
                let flipped = Bytes::from(flipped);
                let Ok(view) = SegmentView::decode("p", flipped.clone(), DIM) else {
                    continue;
                };
                match decode_whole("p", &flipped, DIM) {
                    Ok(whole) => assert_agrees(&view, &whole),
                    // An id byte changed and stayed valid and in order: only
                    // the Bloom comparison sees it, and a view skips that the
                    // way `RangedSegment` does.
                    Err(ItzmaError::CorruptSegment { reason, .. })
                        if reason == "the Bloom filter does not match the ids" => {}
                    Err(_) => assert!(
                        (0..view.rows()).any(|row| view.attrs(row).is_err()),
                        "byte {byte} bit {bit}: the whole decoder refused, the view read every row"
                    ),
                }
            }
        }
    }

    #[test]
    fn damaged_attribute_json_fails_on_read_not_at_decode() {
        let mut layer = Layer::new();
        for (id, attrs) in [("a", json!({"k": "vvvv"})), ("b", json!({"k": 1}))] {
            layer.apply(WalOp::Upsert(Document {
                id: id.into(),
                vector: vec![0.0; DIM],
                attrs: Some(attrs),
            }));
        }
        let mut bytes = bytes_of(&layer).to_vec();
        let at = bytes.windows(4).position(|w| w == b"vvvv").unwrap();
        // The quote before the value becomes a brace.
        bytes[at - 1] = b'{';
        let view = SegmentView::decode("p", Bytes::from(bytes), DIM).unwrap();
        assert!(matches!(
            view.attrs(0),
            Err(ItzmaError::CorruptSegment { .. })
        ));
        assert_eq!(view.attrs(1).unwrap(), Some(json!({"k": 1})));
    }

    /// A tombstone equal to a document id, forged by renaming in place
    /// ("d00001" to "t00000" keeps every length): the overlap check, the
    /// order check or the directory's first ids must catch it.
    #[test]
    fn a_document_that_is_also_a_tombstone_is_refused() {
        let mut bytes = bytes_of(&sample(3, 1)).to_vec();
        let at = bytes.windows(6).position(|w| w == b"d00002").unwrap();
        bytes[at..at + 6].copy_from_slice(b"t00000");
        let err = SegmentView::decode("p", Bytes::from(bytes), DIM)
            .err()
            .unwrap();
        let ItzmaError::CorruptSegment { reason, .. } = err else {
            panic!("{err:?}")
        };
        assert!(
            reason.contains("both") || reason.contains("ascending"),
            "{reason}"
        );
    }

    #[test]
    fn a_version_3_view_agrees_with_the_whole_decoder() {
        let schema = crate::document::Schema::new(DIM, crate::Metric::L2).with_attrs(
            crate::attrs::AttrSchema::new()
                .field("n", crate::attrs::AttrType::Number)
                .field("s", crate::attrs::AttrType::Text),
        );
        let mut layer = Layer::new();
        for i in 0..600usize {
            let attrs = match i % 6 {
                0 => None,
                1 => Some(json!({})),
                _ => Some(json!({"n": i, "s": format!("v{}", i % 3)})),
            };
            layer.apply(WalOp::Upsert(Document {
                id: format!("d{i:04}"),
                vector: vec![i as f32; DIM],
                attrs,
            }));
        }
        layer.apply(WalOp::Delete("e".into()));
        let bytes = crate::segment::encode_with_schema(&layer, &schema, 3).unwrap();
        let view = SegmentView::decode("t", bytes, DIM).unwrap();
        let docs = layer.sorted_docs();
        assert_eq!(view.rows(), docs.len());
        for (row, doc) in docs.iter().enumerate() {
            assert_eq!(view.id(row), doc.id);
            assert_eq!(view.vector(row), doc.vector.as_slice());
            assert_eq!(view.attrs(row).unwrap(), doc.attrs);
        }
        assert!(view.holds("e"));
        assert!(view.weight() > (600 * DIM * 4) as u64);
    }
}
