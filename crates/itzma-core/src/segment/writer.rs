//! The segment encoder: one writer, fed documents and tombstones in
//! id order, for both the whole-layer path ([`super::encode_parts`]) and a
//! streaming merge. The output is a list of chunks in file order, so a large
//! segment never needs one contiguous buffer, and it goes out as one PUT.
//!
//! The bytes are the ones `encode_as` always wrote: the equality
//! test below compares them with the encoders this one replaced.

use bytes::Bytes;
use object_store::PutPayload;

use super::columns::{ColumnBlock, Fields, fields_of, tag};
use super::v2::{BLOCK_ROWS, END_MAGIC, HEADER_LEN};
use super::zone::{encode_empty_zone, encode_zone};
use super::{MAGIC, too_long};
use crate::attrs::AttrSchema;
use crate::bloom::Bloom;
use crate::document::SparseVector;
use crate::error::{ItzmaError, Result};
use crate::sparse::SparseWriter;

/// Most bytes one chunk of a section holds before the next one starts. A
/// chunk can pass it by the one row or block that crossed it.
pub(crate) const SECTION_CHUNK: usize = 8 * 1024 * 1024;

/// One section of an object being written: sealed chunks and an open tail.
#[derive(Default)]
pub(crate) struct Chunks {
    done: Vec<Bytes>,
    open: Vec<u8>,
    len: u64,
}

impl Chunks {
    pub(crate) fn push(&mut self, bytes: &[u8]) {
        self.open.extend_from_slice(bytes);
        self.len += bytes.len() as u64;
        if self.open.len() >= SECTION_CHUNK {
            self.seal();
        }
    }

    /// Little-endian `f32`s, written straight into the open chunk.
    fn push_f32s(&mut self, values: &[f32]) {
        self.open.reserve(values.len() * 4);
        for x in values {
            self.open.extend_from_slice(&x.to_le_bytes());
        }
        self.len += values.len() as u64 * 4;
        if self.open.len() >= SECTION_CHUNK {
            self.seal();
        }
    }

    /// Bytes pushed so far.
    pub(crate) fn len(&self) -> u64 {
        self.len
    }

    fn seal(&mut self) {
        if !self.open.is_empty() {
            self.done.push(Bytes::from(std::mem::take(&mut self.open)));
        }
    }

    pub(crate) fn into_bytes(mut self) -> Vec<Bytes> {
        self.seal();
        self.done
    }
}

/// A version 2 section of rows in blocks of [`BLOCK_ROWS`]: a block is a u32
/// row count, `count + 1` u32 end offsets starting at 0, then the rows.
#[derive(Default)]
struct Blocks {
    section: Chunks,
    /// Where each block starts, relative to the section.
    starts: Vec<u64>,
    /// The first row of each block, kept for sections whose directory lists it.
    firsts: Vec<Vec<u8>>,
    keep_firsts: bool,
    data: Vec<u8>,
    ends: Vec<u32>,
}

impl Blocks {
    fn with_firsts() -> Self {
        Self {
            keep_firsts: true,
            ..Self::default()
        }
    }

    fn row(&mut self, bytes: &[u8]) -> Result<()> {
        if self.ends.is_empty() && self.keep_firsts {
            self.firsts.push(bytes.to_vec());
        }
        self.data.extend_from_slice(bytes);
        self.ends
            .push(u32::try_from(self.data.len()).map_err(|_| too_long())?);
        if self.ends.len() == BLOCK_ROWS {
            self.close();
        }
        Ok(())
    }

    fn close(&mut self) {
        if self.ends.is_empty() {
            return;
        }
        self.starts.push(self.section.len());
        let mut head = Vec::with_capacity(8 + self.ends.len() * 4);
        head.extend_from_slice(&(self.ends.len() as u32).to_le_bytes());
        head.extend_from_slice(&0u32.to_le_bytes());
        for end in &self.ends {
            head.extend_from_slice(&end.to_le_bytes());
        }
        self.section.push(&head);
        self.section.push(&self.data);
        self.data.clear();
        self.ends.clear();
    }

    /// The section's chunks, and the `n + 1` absolute offsets of its blocks
    /// for a section that starts at `base`.
    fn finish(mut self, base: u64) -> (Chunks, Vec<u64>, Vec<Vec<u8>>) {
        self.close();
        let mut offsets: Vec<u64> = self.starts.iter().map(|s| base + s).collect();
        offsets.push(base + self.section.len());
        (self.section, offsets, self.firsts)
    }
}

/// One document's attributes as a caller has them.
pub(crate) enum AttrIn<'a> {
    /// As stored: JSON, empty for none.
    Json(&'a [u8]),
    /// As a value the caller is done with, so a columnar writer keeps it
    /// without a copy (a merge's documents).
    Owned(Option<serde_json::Value>),
    /// Already in an attribute block given to
    /// [`SegmentWriter::attr_blocks`] (a typed version 3 writer only).
    Encoded,
}

/// One version 3 attribute block encoded ahead of the writer, so a
/// whole-layer fold can encode its blocks in parallel: the column block, its
/// zone block, and how many documents it holds.
pub(crate) struct EncodedAttrs {
    block: Vec<u8>,
    zone: Vec<u8>,
    rows: usize,
}

/// Encode one block of rows' attributes as columns and its zone block.
///
/// # Errors
///
/// [`ItzmaError::CorruptSegment`] at `(encoding)` for attributes the fields
/// do not allow.
pub(crate) fn encode_attr_block(
    fields: &Fields,
    rows: &[Option<&serde_json::Value>],
) -> Result<EncodedAttrs> {
    let columns = ColumnBlock::from_refs(fields, rows)?;
    let mut block = Vec::new();
    let ranges = columns.encode(&mut block);
    let mut zone = Vec::new();
    encode_zone(&columns, &ranges, &mut zone);
    Ok(EncodedAttrs {
        block,
        zone,
        rows: rows.len(),
    })
}

fn refuse(reason: &str) -> ItzmaError {
    ItzmaError::CorruptSegment {
        path: "(encoding)".into(),
        reason: reason.into(),
    }
}

/// Where a blocked segment's attributes go, by layout.
// One per segment being written, so the bigger variant costs nothing.
#[allow(clippy::large_enum_variant)]
enum AttrSink {
    /// JSON rows in blocks: version 2, and version 3 for a free-form namespace.
    Json(Blocks),
    /// Version 3 typed: rows buffered until their block closes,
    /// then written as columns with a zone block beside them.
    Columns {
        fields: Fields,
        pending: Vec<Option<serde_json::Value>>,
        /// Blocks encoded ahead, used in order instead of `pending`.
        ahead: std::collections::VecDeque<EncodedAttrs>,
        /// Rows of the current block given as [`AttrIn::Encoded`].
        encoded_rows: usize,
        section: Chunks,
        starts: Vec<u64>,
        zone: Chunks,
        zone_starts: Vec<u64>,
    },
}

/// A finished attribute section: its chunks, its `n + 1` absolute block
/// offsets, and for columns its zone section and that section's `n` block
/// starts, relative to the section.
struct FinishedAttrs {
    section: Chunks,
    offsets: Vec<u64>,
    zone: Option<(Chunks, Vec<u64>)>,
}

impl AttrSink {
    fn row(&mut self, attrs: AttrIn<'_>) -> Result<()> {
        match self {
            Self::Json(blocks) => match attrs {
                AttrIn::Json(bytes) => blocks.row(bytes),
                AttrIn::Owned(value) => blocks.row(&attr_json(value.as_ref())?),
                AttrIn::Encoded => Err(refuse("encoded attributes need a typed version 3 writer")),
            },
            Self::Columns {
                pending,
                encoded_rows,
                ..
            } => {
                let value = match attrs {
                    AttrIn::Encoded if pending.is_empty() => {
                        *encoded_rows += 1;
                        if *encoded_rows == BLOCK_ROWS {
                            self.close()?;
                        }
                        return Ok(());
                    }
                    AttrIn::Encoded => None,
                    _ if *encoded_rows > 0 => None,
                    AttrIn::Json(bytes) => Some(super::ranged::decode_attrs("(encoding)", bytes)?),
                    AttrIn::Owned(value) => Some(value),
                };
                let Some(value) = value else {
                    return Err(refuse(
                        "encoded and plain attributes are mixed in one block",
                    ));
                };
                pending.push(value);
                if pending.len() == BLOCK_ROWS {
                    self.close()?;
                }
                Ok(())
            }
        }
    }

    /// Write the buffered rows of a columnar sink as one block.
    fn close(&mut self) -> Result<()> {
        let Self::Columns {
            fields,
            pending,
            ahead,
            encoded_rows,
            section,
            starts,
            zone,
            zone_starts,
        } = self
        else {
            return Ok(());
        };
        let encoded = if *encoded_rows > 0 {
            let next = ahead
                .pop_front()
                .ok_or_else(|| refuse("no encoded attribute block is left"))?;
            if next.rows != *encoded_rows {
                return Err(refuse(
                    "an encoded attribute block does not hold its documents' rows",
                ));
            }
            *encoded_rows = 0;
            next
        } else if pending.is_empty() {
            return Ok(());
        } else {
            let rows: Vec<Option<&serde_json::Value>> =
                pending.iter().map(Option::as_ref).collect();
            let next = encode_attr_block(fields, &rows)?;
            pending.clear();
            next
        };
        starts.push(section.len());
        section.push(&encoded.block);
        zone_starts.push(zone.len());
        zone.push(&encoded.zone);
        Ok(())
    }

    fn finish(mut self, base: u64) -> Result<FinishedAttrs> {
        self.close()?;
        if matches!(&self, Self::Columns { ahead, .. } if !ahead.is_empty()) {
            return Err(refuse("an encoded attribute block is left over"));
        }
        Ok(match self {
            Self::Json(blocks) => {
                let (section, offsets, _) = blocks.finish(base);
                FinishedAttrs {
                    section,
                    offsets,
                    zone: None,
                }
            }
            Self::Columns {
                section,
                starts,
                zone,
                zone_starts,
                ..
            } => {
                let mut offsets: Vec<u64> = starts.iter().map(|s| base + s).collect();
                offsets.push(base + section.len());
                FinishedAttrs {
                    section,
                    offsets,
                    zone: Some((zone, zone_starts)),
                }
            }
        })
    }
}

// One writer exists per segment being written, so the size of the bigger
// variant costs nothing worth a box.
#[allow(clippy::large_enum_variant)]
enum Layout {
    /// Versions 2 and 3: rows in blocks, a Bloom filter and a directory.
    Blocked {
        ids: Blocks,
        attrs: AttrSink,
        tombstones: Blocks,
        /// Every id and tombstone, for the Bloom filter, whose size depends
        /// on how many there are and so is built at the end: the bytes of
        /// all of them, and where each one ends.
        keys: Vec<u8>,
        key_ends: Vec<usize>,
    },
}

/// Encodes one segment object (version 2 or 3) and its `sparse.bin` from
/// documents and tombstones given in strictly ascending id order.
pub(crate) struct SegmentWriter {
    dim: usize,
    dim32: u32,
    version: u32,
    layout: Layout,
    vectors: Chunks,
    docs: u64,
    tombstones: u64,
    /// The last id written, kept in one reused buffer, once there is one.
    last: String,
    any: bool,
    sparse: SparseWriter,
}

impl SegmentWriter {
    /// A writer for `version` 2 or 3 with vectors of length `dim`. At
    /// version 3 a typed `attrs` schema stores attributes as columns;
    /// version 2, and a free-form schema, store JSON.
    ///
    /// # Errors
    ///
    /// [`ItzmaError::UnsupportedFormat`] for any other version.
    pub(crate) fn new(dim: usize, version: u32, attrs: &AttrSchema) -> Result<Self> {
        let layout = match version {
            2 | 3 => Layout::Blocked {
                ids: Blocks::with_firsts(),
                attrs: if version == 3 && !attrs.is_free_form() {
                    AttrSink::Columns {
                        fields: fields_of(attrs),
                        pending: Vec::with_capacity(BLOCK_ROWS),
                        ahead: std::collections::VecDeque::new(),
                        encoded_rows: 0,
                        section: Chunks::default(),
                        starts: Vec::new(),
                        zone: Chunks::default(),
                        zone_starts: Vec::new(),
                    }
                } else {
                    AttrSink::Json(Blocks::default())
                },
                tombstones: Blocks::with_firsts(),
                keys: Vec::new(),
                key_ends: Vec::new(),
            },
            version => return Err(ItzmaError::UnsupportedFormat { version }),
        };
        Ok(Self {
            dim,
            dim32: u32::try_from(dim).map_err(|_| too_long())?,
            version,
            layout,
            vectors: Chunks::default(),
            docs: 0,
            tombstones: 0,
            last: String::new(),
            any: false,
            sparse: SparseWriter::new(),
        })
    }

    /// Ids across documents and tombstones must rise strictly.
    fn next_id(&mut self, id: &str) -> Result<()> {
        if self.any && id <= self.last.as_str() {
            return Err(ItzmaError::CorruptSegment {
                path: "(encoding)".into(),
                reason: "ids are not in strictly ascending order".into(),
            });
        }
        self.last.clear();
        self.last.push_str(id);
        self.any = true;
        Ok(())
    }

    /// A live document with its attributes.
    ///
    /// # Errors
    ///
    /// [`ItzmaError::DimensionMismatch`] for a vector of another length,
    /// [`ItzmaError::CorruptSegment`] for an id at or below the last one, a
    /// field longer than 4 GiB, or at version 3 attributes the schema does
    /// not allow.
    pub(crate) fn doc(
        &mut self,
        id: &str,
        vector: &[f32],
        attrs: AttrIn<'_>,
        sparse: Option<&SparseVector>,
    ) -> Result<()> {
        if vector.len() != self.dim {
            return Err(ItzmaError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        self.next_id(id)?;
        match &mut self.layout {
            Layout::Blocked {
                ids,
                attrs: a,
                keys,
                key_ends,
                ..
            } => {
                ids.row(id.as_bytes())?;
                a.row(attrs)?;
                keys.extend_from_slice(id.as_bytes());
                key_ends.push(keys.len());
            }
        }
        self.vectors.push_f32s(vector);
        self.sparse.row(sparse)?;
        self.docs += 1;
        Ok(())
    }

    /// A tombstone.
    ///
    /// # Errors
    ///
    /// [`ItzmaError::CorruptSegment`] for an id at or below the last one.
    pub(crate) fn tombstone(&mut self, id: &str) -> Result<()> {
        self.next_id(id)?;
        match &mut self.layout {
            Layout::Blocked {
                tombstones,
                keys,
                key_ends,
                ..
            } => {
                tombstones.row(id.as_bytes())?;
                keys.extend_from_slice(id.as_bytes());
                key_ends.push(keys.len());
            }
        }
        self.tombstones += 1;
        Ok(())
    }

    /// Hand a typed version 3 writer the attribute blocks of the documents to
    /// come, encoded ahead in order (one per [`BLOCK_ROWS`] documents, the
    /// last one shorter); each of those documents is then written with
    /// [`AttrIn::Encoded`]. The bytes are the ones the writer would encode.
    ///
    /// # Errors
    ///
    /// [`ItzmaError::CorruptSegment`] at `(encoding)` for any other writer or
    /// once a document is written; a block that does not hold its documents'
    /// rows, or one left over, fails the write that finds it.
    pub(crate) fn attr_blocks(&mut self, blocks: Vec<EncodedAttrs>) -> Result<()> {
        match &mut self.layout {
            Layout::Blocked {
                attrs: AttrSink::Columns { ahead, .. },
                ..
            } if self.docs == 0 => {
                ahead.extend(blocks);
                Ok(())
            }
            _ => Err(refuse(
                "encoded attribute blocks need a fresh typed version 3 writer",
            )),
        }
    }

    /// Documents written so far.
    pub(crate) fn docs(&self) -> usize {
        self.docs as usize
    }

    /// Tombstones written so far.
    pub(crate) fn tombstones(&self) -> usize {
        self.tombstones as usize
    }

    /// The data object in file order, and its `sparse.bin` if any document
    /// was given a sparse vector.
    ///
    /// # Errors
    ///
    /// [`ItzmaError::CorruptSegment`] for a first id longer than 4 GiB or more
    /// rows than `sparse.bin` can count.
    pub(crate) fn finish(self) -> Result<(PutPayload, Option<PutPayload>)> {
        let (sparse, any) = self.sparse.finish()?;
        let mut header = Vec::with_capacity(HEADER_LEN);
        header.extend_from_slice(&MAGIC);
        header.extend_from_slice(&self.version.to_le_bytes());
        header.extend_from_slice(&self.dim32.to_le_bytes());
        header.extend_from_slice(&self.docs.to_le_bytes());
        header.extend_from_slice(&self.tombstones.to_le_bytes());
        let mut parts = Vec::new();
        match self.layout {
            Layout::Blocked {
                ids,
                attrs,
                tombstones,
                keys,
                key_ends,
            } => {
                header.extend_from_slice(&(BLOCK_ROWS as u32).to_le_bytes());
                parts.push(Bytes::from(header));
                // Each section's end, and so the next one's start, is known
                // only once its last block is closed.
                let (ids, id_offsets, id_firsts) = ids.finish(HEADER_LEN as u64);
                let blocks = id_offsets.len() - 1;
                let vectors_offset = HEADER_LEN as u64 + ids.len();
                let columnar = matches!(attrs, AttrSink::Columns { .. });
                let field_table = match &attrs {
                    AttrSink::Columns { fields, .. } => Some(fields.clone()),
                    AttrSink::Json(_) => None,
                };
                let attrs = attrs.finish(vectors_offset + self.vectors.len())?;
                let tombstones_base = vectors_offset + self.vectors.len() + attrs.section.len();
                let (tombstones, tombstone_offsets, tombstone_firsts) =
                    tombstones.finish(tombstones_base);
                let bloom_offset = tombstones_base + tombstones.len();
                for section in [ids, self.vectors, attrs.section, tombstones] {
                    parts.extend(section.into_bytes());
                }

                let mut start = 0;
                let keys: Vec<&[u8]> = key_ends
                    .iter()
                    .map(|&end| {
                        let key = &keys[start..end];
                        start = end;
                        key
                    })
                    .collect();
                let mut tail = Bloom::build(keys.into_iter()).to_bytes();
                let bloom_len = tail.len() as u64;

                // Version 3: the zone section, one block per id block.
                let zone_base = bloom_offset + bloom_len;
                let mut zone_offsets = Vec::new();
                if self.version >= 3 {
                    match attrs.zone {
                        Some((zone, starts)) => {
                            zone_offsets.extend(starts.iter().map(|s| zone_base + s));
                            zone_offsets.push(zone_base + zone.len());
                            for chunk in zone.into_bytes() {
                                tail.extend_from_slice(&chunk);
                            }
                        }
                        None => {
                            for _ in 0..blocks {
                                zone_offsets.push(bloom_offset + tail.len() as u64);
                                encode_empty_zone(&mut tail);
                            }
                            zone_offsets.push(bloom_offset + tail.len() as u64);
                        }
                    }
                }
                debug_assert!(!columnar || zone_offsets.len() == blocks + 1);

                let directory_offset = bloom_offset + tail.len() as u64;
                tail.extend_from_slice(&vectors_offset.to_le_bytes());
                tail.extend_from_slice(&bloom_offset.to_le_bytes());
                tail.extend_from_slice(&bloom_len.to_le_bytes());
                put_offsets(&mut tail, &id_offsets);
                for first in &id_firsts {
                    super::put_bytes(&mut tail, first)?;
                }
                put_offsets(&mut tail, &attrs.offsets);
                put_offsets(&mut tail, &tombstone_offsets);
                for first in &tombstone_firsts {
                    super::put_bytes(&mut tail, first)?;
                }
                if self.version >= 3 {
                    match &field_table {
                        Some(fields) => {
                            tail.push(1);
                            put_field_table(&mut tail, fields)?;
                        }
                        None => tail.push(0),
                    }
                    put_offsets(&mut tail, &zone_offsets);
                }
                let directory_len = bloom_offset + tail.len() as u64 - directory_offset;
                tail.extend_from_slice(&directory_offset.to_le_bytes());
                tail.extend_from_slice(&directory_len.to_le_bytes());
                tail.extend_from_slice(&END_MAGIC);
                parts.push(Bytes::from(tail));
            }
        }
        Ok((parts.into_iter().collect(), any.then_some(sparse)))
    }
}

fn put_offsets(buf: &mut Vec<u8>, offsets: &[u64]) {
    for offset in offsets {
        buf.extend_from_slice(&offset.to_le_bytes());
    }
}

/// A version 3 directory's field table: u16 count, then per field a u8 name
/// length, the name and its type's tag.
fn put_field_table(out: &mut Vec<u8>, fields: &Fields) -> Result<()> {
    let count = u16::try_from(fields.len()).map_err(|_| too_long())?;
    out.extend_from_slice(&count.to_le_bytes());
    for field in fields.iter() {
        // Field names are at most 64 bytes.
        let len = u8::try_from(field.name.len()).map_err(|_| too_long())?;
        out.push(len);
        out.extend_from_slice(field.name.as_bytes());
        out.push(tag(field.ty));
    }
    Ok(())
}

/// One document's attributes as the formats store them: JSON, empty for none.
pub(crate) fn attr_json(attrs: Option<&serde_json::Value>) -> Result<Vec<u8>> {
    match attrs {
        None => Ok(Vec::new()),
        Some(value) => serde_json::to_vec(value).map_err(|source| ItzmaError::Serde {
            path: "segment".into(),
            source,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{Document, WalOp};
    use crate::layer::Layer;
    use crate::segment::encode_parts;

    /// xorshift64, so every run sees the same cases without a `rand` dependency.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n.max(1)
        }

        fn f32(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u32 << 23) as f32 - 1.0
        }
    }

    fn id(rng: &mut Rng, n: u64) -> String {
        let prefixes = ["", "ü-", "日本/", "a", "zz"];
        let prefix = prefixes[rng.below(prefixes.len() as u64) as usize];
        let pad = rng.below(30) as usize;
        format!("{prefix}{n:0pad$}")
    }

    fn attrs(rng: &mut Rng) -> Option<serde_json::Value> {
        match rng.below(6) {
            0 => None,
            1 => Some(serde_json::Value::Null),
            2 => Some(serde_json::json!({ "n": rng.below(1000), "f": 5.0, "tiny": 1e-7 })),
            3 => Some(serde_json::json!({ "nested": { "list": [1, "two", null] }, "s": "x" })),
            4 => Some(serde_json::json!("plain")),
            _ => Some(serde_json::json!(f64::from(rng.f32()))),
        }
    }

    /// A random layer: documents, tombstones interleaved with them, sparse
    /// vectors on some, every block count from none to several.
    fn layer(rng: &mut Rng, dim: usize) -> Layer {
        let mut l = Layer::new();
        let docs = match rng.below(5) {
            0 => 0,
            1 => 256,
            2 => 257,
            _ => rng.below(1_100),
        };
        for n in 0..docs {
            let sparse = (rng.below(3) == 0).then(|| {
                (0..rng.below(5) + 1)
                    .map(|t| (t as u32 * 7 + 1, rng.f32()))
                    .collect()
            });
            l.apply(WalOp::Upsert(Document {
                id: id(rng, n),
                vector: (0..dim).map(|_| rng.f32()).collect(),
                attrs: attrs(rng),
                sparse,
            }));
        }
        for _ in 0..rng.below(300) {
            let n = rng.below(docs.max(1) * 2);
            l.apply(WalOp::Delete(id(rng, n)));
        }
        l
    }

    fn join(payload: &PutPayload) -> Vec<u8> {
        payload.iter().flat_map(|b| b.iter().copied()).collect()
    }

    /// Equal bytes, or a panic naming the first byte that differs.
    fn same(got: &[u8], want: &[u8], what: &str) {
        if let Some(at) = got.iter().zip(want).position(|(a, b)| a != b) {
            panic!("{what}: first difference at byte {at} of {}", want.len());
        }
        assert_eq!(got.len(), want.len(), "{what}: lengths differ");
    }

    fn reference(layer: &Layer, dim: usize) -> Vec<u8> {
        crate::segment::v2::reference_encode(layer, dim)
            .unwrap()
            .to_vec()
    }

    #[test]
    fn writes_what_the_old_encoders_wrote() {
        let mut rng = Rng(0x0037_5eed);
        for case in 0..300 {
            let dim = rng.below(9) as usize + 1;
            let layer = layer(&mut rng, dim);
            let (data, sparse) = encode_parts(&layer, dim, 2, &AttrSchema::free_form()).unwrap();
            same(
                &join(&data),
                &reference(&layer, dim),
                &format!("data, case {case}"),
            );
            let want_sparse = layer.sorted_docs().iter().any(|d| d.sparse.is_some());
            assert_eq!(sparse.is_some(), want_sparse, "case {case}");
            if let Some(s) = sparse {
                same(
                    &join(&s),
                    &crate::sparse::reference_encode(&layer).unwrap(),
                    &format!("sparse, case {case}"),
                );
            }
        }
    }

    #[test]
    fn an_empty_writer_is_an_empty_layer() {
        let (data, sparse) = SegmentWriter::new(3, 2, &AttrSchema::free_form())
            .unwrap()
            .finish()
            .unwrap();
        assert_eq!(join(&data), reference(&Layer::new(), 3));
        assert!(sparse.is_none());
    }

    #[test]
    fn ids_out_of_order_are_refused() {
        {
            let mut w = SegmentWriter::new(1, 2, &AttrSchema::free_form()).unwrap();
            w.doc("b", &[0.0], AttrIn::Json(b""), None).unwrap();
            assert!(matches!(
                w.doc("a", &[0.0], AttrIn::Json(b""), None),
                Err(ItzmaError::CorruptSegment { .. })
            ));
            assert!(matches!(
                w.tombstone("b"),
                Err(ItzmaError::CorruptSegment { .. })
            ));
            w.tombstone("c").unwrap();
            assert!(matches!(
                w.doc("c", &[0.0], AttrIn::Json(b""), None),
                Err(ItzmaError::CorruptSegment { .. })
            ));
        }
    }

    #[test]
    fn a_vector_of_another_length_is_refused() {
        let mut w = SegmentWriter::new(2, 2, &AttrSchema::free_form()).unwrap();
        assert!(matches!(
            w.doc("a", &[0.0], AttrIn::Json(b""), None),
            Err(ItzmaError::DimensionMismatch {
                expected: 2,
                got: 1
            })
        ));
    }

    #[test]
    fn an_unknown_version_is_refused() {
        for version in [0, 4] {
            assert!(matches!(
                SegmentWriter::new(2, version, &AttrSchema::free_form()),
                Err(ItzmaError::UnsupportedFormat { .. })
            ));
        }
    }

    #[test]
    fn a_section_is_split_into_chunks_of_bounded_size() {
        let dim = 1024;
        let mut w = SegmentWriter::new(dim, 2, &AttrSchema::free_form()).unwrap();
        let rows = 2 * SECTION_CHUNK / (dim * 4) + 10;
        for n in 0..rows {
            w.doc(&format!("{n:08}"), &vec![0.5; dim], AttrIn::Json(b""), None)
                .unwrap();
        }
        let (data, _) = w.finish().unwrap();
        assert!(data.iter().count() >= 3, "the vectors are several chunks");
        assert!(data.iter().all(|c| c.len() <= SECTION_CHUNK + dim * 4));
    }

    use crate::attrs::{AttrSchema, AttrType};
    use crate::document::Schema;
    use crate::segment::{decode, encode_with_schema};

    fn typed() -> Schema {
        Schema::new(3, crate::Metric::L2).with_attrs(
            AttrSchema::new()
                .field("b", AttrType::Bool)
                .field("n", AttrType::Number)
                .field("s", AttrType::Text)
                .field("ss", AttrType::TextSet)
                .field("ns", AttrType::NumberSet),
        )
    }

    /// A layer of `docs` documents with attributes of every type (some none,
    /// some `{}`, some nulls) and `tombstones` tombstones.
    fn typed_layer(rng: &mut Rng, docs: usize, tombstones: usize) -> Layer {
        let mut layer = Layer::new();
        for i in 0..docs {
            // Built outside `json!`, which would read `[..]` as an array literal.
            let ss: Vec<&str> = ["a", "b"][..(rng.next() % 3) as usize].to_vec();
            let attrs = match rng.next() % 5 {
                0 => None,
                1 => Some(serde_json::json!({})),
                2 => Some(serde_json::json!({"n": null, "s": "x"})),
                _ => Some(serde_json::json!({
                    "b": rng.next().is_multiple_of(2),
                    "n": (rng.next() % 1000) as f64 / 4.0,
                    "s": format!("v{}", rng.next() % 7),
                    "ss": ss,
                    "ns": [rng.next() % 5, 2],
                })),
            };
            layer.apply(WalOp::Upsert(Document {
                id: format!("d{i:05}"),
                vector: vec![i as f32, 0.5, -1.0],
                attrs,
                sparse: None,
            }));
        }
        for i in 0..tombstones {
            layer.apply(WalOp::Delete(format!("t{i:05}")));
        }
        layer
    }

    #[test]
    fn a_version_3_segment_decodes_to_the_layer_it_was_built_from() {
        let mut rng = Rng(11);
        for (docs, tombstones) in [(0, 0), (1, 0), (255, 3), (256, 256), (257, 1), (700, 40)] {
            let layer = typed_layer(&mut rng, docs, tombstones);
            let bytes = encode_with_schema(&layer, &typed(), 3).unwrap();
            assert_eq!(bytes[4..8], 3u32.to_le_bytes());
            let back = decode("t", &bytes, 3).unwrap();
            assert_eq!(back.sorted_docs(), layer.sorted_docs(), "{docs} docs");
            assert_eq!(back.sorted_tombstones(), layer.sorted_tombstones());
        }
    }

    #[test]
    fn json_and_owned_inputs_write_the_same_bytes() {
        let layer = typed_layer(&mut Rng(5), 600, 10);
        let fields = &typed().attrs;
        for version in [2, 3] {
            let write = |how: u8| {
                let mut w = SegmentWriter::new(3, version, fields).unwrap();
                let mut dead = layer.sorted_tombstones().into_iter().peekable();
                for doc in layer.sorted_docs() {
                    while let Some(t) = dead.next_if(|t| *t < doc.id.as_str()) {
                        w.tombstone(t).unwrap();
                    }
                    let json = attr_json(doc.attrs.as_ref()).unwrap();
                    let attrs = match how {
                        0 => AttrIn::Json(&json),
                        _ => AttrIn::Owned(doc.attrs.clone()),
                    };
                    w.doc(&doc.id, &doc.vector, attrs, None).unwrap();
                }
                for t in dead {
                    w.tombstone(t).unwrap();
                }
                Bytes::from(w.finish().unwrap().0)
            };
            assert_eq!(write(0), write(1), "version {version}");
        }
    }

    #[test]
    fn json_and_value_inputs_write_the_same_version_3_bytes() {
        let layer = typed_layer(&mut Rng(5), 600, 10);
        let fields = &typed().attrs;
        let write = |as_json: bool| {
            let mut w = SegmentWriter::new(3, 3, fields).unwrap();
            let mut dead = layer.sorted_tombstones().into_iter().peekable();
            for doc in layer.sorted_docs() {
                while let Some(t) = dead.next_if(|t| *t < doc.id.as_str()) {
                    w.tombstone(t).unwrap();
                }
                let json = attr_json(doc.attrs.as_ref()).unwrap();
                let attrs = if as_json {
                    AttrIn::Json(&json)
                } else {
                    AttrIn::Owned(doc.attrs.clone())
                };
                w.doc(&doc.id, &doc.vector, attrs, None).unwrap();
            }
            for t in dead {
                w.tombstone(t).unwrap();
            }
            let (data, _) = w.finish().unwrap();
            Bytes::from(data)
        };
        assert_eq!(write(true), write(false));
    }

    #[test]
    fn version_2_bytes_are_unchanged_by_the_schema() {
        let layer = typed_layer(&mut Rng(3), 300, 5);
        let typed_bytes = encode_with_schema(&layer, &typed(), 2).unwrap();
        let free = crate::segment::encode_as(&layer, 3, 2).unwrap();
        assert_eq!(typed_bytes, free);
        assert_eq!(free, super::super::v2::reference_encode(&layer, 3).unwrap());
    }

    #[test]
    fn a_typed_namespace_with_no_fields_round_trips() {
        let schema = Schema::new(3, crate::Metric::L2).with_attrs(AttrSchema::new());
        let mut layer = Layer::new();
        for (i, attrs) in [None, Some(serde_json::json!({}))].into_iter().enumerate() {
            layer.apply(WalOp::Upsert(Document {
                id: format!("d{i}"),
                vector: vec![0.0; 3],
                attrs,
                sparse: None,
            }));
        }
        let bytes = encode_with_schema(&layer, &schema, 3).unwrap();
        assert_eq!(
            decode("t", &bytes, 3).unwrap().sorted_docs(),
            layer.sorted_docs()
        );
    }

    #[test]
    fn a_free_form_layer_writes_json_blocks_at_version_3() {
        let layer = typed_layer(&mut Rng(9), 300, 2);
        let bytes = crate::segment::encode_as(&layer, 3, 3).unwrap();
        assert_eq!(
            decode("t", &bytes, 3).unwrap().sorted_docs(),
            layer.sorted_docs()
        );
        // Undeclared keys are fine in a free-form namespace.
        let mut odd = Layer::new();
        odd.apply(WalOp::Upsert(Document {
            id: "a".into(),
            vector: vec![0.0; 3],
            attrs: Some(serde_json::json!({"anything": [1, "x", null]})),
            sparse: None,
        }));
        let bytes = crate::segment::encode_as(&odd, 3, 3).unwrap();
        assert_eq!(
            decode("t", &bytes, 3).unwrap().sorted_docs(),
            odd.sorted_docs()
        );
    }

    #[test]
    fn a_typed_layer_with_an_undeclared_key_is_refused_at_encoding() {
        let mut layer = Layer::new();
        layer.apply(WalOp::Upsert(Document {
            id: "a".into(),
            vector: vec![0.0; 3],
            attrs: Some(serde_json::json!({"zz": 1})),
            sparse: None,
        }));
        let err = encode_with_schema(&layer, &typed(), 3).unwrap_err();
        assert!(matches!(err, ItzmaError::CorruptSegment { ref path, .. } if path == "(encoding)"));
    }

    /// Feed `layer` to a version 3 writer one value at a time, the reference
    /// the precomputed path must match byte for byte.
    fn one_at_a_time(layer: &Layer, schema: &Schema) -> Bytes {
        let mut w = SegmentWriter::new(schema.dim, 3, &schema.attrs).unwrap();
        let mut dead = layer.sorted_tombstones().into_iter().peekable();
        for doc in layer.sorted_docs() {
            while let Some(t) = dead.next_if(|t| *t < doc.id.as_str()) {
                w.tombstone(t).unwrap();
            }
            w.doc(&doc.id, &doc.vector, AttrIn::Owned(doc.attrs.clone()), None)
                .unwrap();
        }
        for t in dead {
            w.tombstone(t).unwrap();
        }
        Bytes::from(w.finish().unwrap().0)
    }

    #[test]
    fn attribute_blocks_encoded_ahead_write_the_same_bytes() {
        let mut rng = Rng(17);
        for (docs, tombstones) in [(0, 0), (1, 0), (255, 2), (256, 0), (257, 3), (700, 9)] {
            let layer = typed_layer(&mut rng, docs, tombstones);
            let want = one_at_a_time(&layer, &typed());
            // `encode_parts` encodes the blocks ahead, a chunk per block.
            let got = encode_with_schema(&layer, &typed(), 3).unwrap();
            same(&got, &want, &format!("{docs} docs"));
        }
    }

    #[test]
    fn encoded_blocks_that_do_not_fit_the_documents_are_refused() {
        let schema = typed();
        let fields = fields_of(&schema.attrs);
        let block = |n: usize| {
            let rows: Vec<Option<&serde_json::Value>> = vec![None; n];
            encode_attr_block(&fields, &rows).unwrap()
        };
        let feed = |blocks: Vec<EncodedAttrs>, docs: usize| -> Result<()> {
            let mut w = SegmentWriter::new(3, 3, &schema.attrs)?;
            w.attr_blocks(blocks)?;
            for i in 0..docs {
                w.doc(&format!("d{i:04}"), &[0.0; 3], AttrIn::Encoded, None)?;
            }
            w.finish().map(|_| ())
        };
        // Right: one block of 3 rows for 3 documents.
        feed(vec![block(3)], 3).unwrap();
        for (blocks, docs) in [
            (vec![block(2)], 3),           // a block too short
            (vec![block(3)], 2),           // a block too long
            (vec![block(3), block(1)], 3), // a block left over
            (vec![], 1),                   // no block at all
        ] {
            let err = feed(blocks, docs).unwrap_err();
            assert!(
                matches!(err, ItzmaError::CorruptSegment { ref path, .. } if path == "(encoding)"),
                "{err:?}"
            );
        }
        // Encoded blocks are for a typed version 3 writer only.
        let mut v2 = SegmentWriter::new(3, 2, &schema.attrs).unwrap();
        assert!(v2.attr_blocks(vec![block(1)]).is_err());
    }
}
