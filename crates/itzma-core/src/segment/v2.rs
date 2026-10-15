//! Segment data object, format versions 2 and 3.
//!
//! Version 3 is version 2 with typed attribute columns per block (layout 1,
//! `columns.rs`; a free-form namespace keeps the JSON blocks, layout 0), a
//! zone section between the Bloom filter and the directory (`zone.rs`), and a
//! directory that adds the layout byte, the field table for layout 1, and the
//! zone block offsets. Everything below holds for both versions otherwise.
//!
//! The same content as version 1, laid out so that a reader can fetch pieces
//! of it: ids, attributes and tombstones sit in blocks of 256 rows, the
//! vectors are one matrix addressed by row number, a Bloom filter answers "is
//! this id here?" without a read, and a directory at the end says where
//! everything is. Nothing is compressed and no float is computed, so the bytes
//! are a pure function of the layer and the byte-for-byte reuse holds.
//!
//! # Layout
//!
//! All integers are little-endian.
//!
//! | Section | Contents |
//! |---|---|
//! | Header, 32 bytes | magic `ITZS`, `format_version` u32 (2), `dim` u32, `doc_count` u64, `tombstone_count` u64, `block_rows` u32 (256) |
//! | Ids | `ceil(doc_count / 256)` blocks |
//! | Vectors | `doc_count * dim` `f32`, row-major, in id order |
//! | Attributes | as many blocks as ids, one JSON value per row, empty for none |
//! | Tombstones | `ceil(tombstone_count / 256)` blocks |
//! | Bloom filter | over every id and every tombstone together |
//! | Directory | `vectors_offset`, `bloom_offset`, `bloom_len` (u64 each); the id block offsets (`n + 1` u64, the last is where the section ends) and the first id of each id block (u32 length, bytes); the attribute block offsets (`n + 1` u64); the tombstone block offsets (`m + 1` u64) and the first id of each tombstone block |
//! | Trailer, 20 bytes | directory offset u64, directory length u64, magic `ITZE` |
//!
//! A block is a u32 row count, `count + 1` u32 offsets (the first is 0, the
//! last is the length of the data), then the concatenated rows. `n` and `m`
//! are the block counts and come from the header, so the directory has no
//! count of its own to forge.
//!
//! [`decode`] checks that the sections touch with no gaps, that every block
//! stands on its own, that the directory agrees with the blocks, that order
//! holds inside and across blocks, and that the Bloom filter is the one the
//! content would produce. It checks every length against the bytes that
//! remain before allocating, and never panics.

#[cfg(test)]
use bytes::Bytes;

use super::columns::{ColumnBlock, Field, Fields, type_of_tag};
use super::zone::{encode_empty_zone, encode_zone};
#[cfg(test)]
use super::{MAGIC, check_dim, put_bytes, too_long};
use super::{Reader, overlaps};
use crate::bloom::Bloom;
use crate::document::Document;
use crate::error::{ItzmaError, Result};
use crate::layer::Layer;

pub(super) const VERSION: u32 = 2;
/// Rows per block. A constant of the version, so the layout is a function of
/// the counts alone.
pub(crate) const BLOCK_ROWS: usize = 256;
/// Magic, version, dim, doc count, tombstone count, block rows.
pub(super) const HEADER_LEN: usize = 4 + 4 + 4 + 8 + 8 + 4;
/// Directory offset, directory length, magic.
pub(super) const TRAILER_LEN: usize = 8 + 8 + 4;
pub(super) const END_MAGIC: [u8; 4] = *b"ITZE";

pub(super) fn corrupt(path: &str, reason: impl Into<String>) -> ItzmaError {
    ItzmaError::CorruptSegment {
        path: path.to_string(),
        reason: reason.into(),
    }
}

/// A Bloom filter over ids and tombstones together, the same on both sides of
/// the codec.
fn bloom_of<'a>(keys: impl Iterator<Item = &'a str>) -> Bloom {
    let keys: Vec<&[u8]> = keys.map(str::as_bytes).collect();
    Bloom::build(keys.into_iter())
}

/// The encoder `SegmentWriter` replaced, kept as the reference its equality
/// test compares against.
#[cfg(test)]
pub(super) fn reference_encode(layer: &Layer, dim: usize) -> Result<Bytes> {
    let docs = layer.sorted_docs();
    let tombstones = layer.sorted_tombstones();
    let dim32 = check_dim(&docs, dim)?;

    let mut buf = Vec::with_capacity(HEADER_LEN + docs.len() * (dim * 4 + 16));
    buf.extend_from_slice(&MAGIC);
    buf.extend_from_slice(&VERSION.to_le_bytes());
    buf.extend_from_slice(&dim32.to_le_bytes());
    buf.extend_from_slice(&(docs.len() as u64).to_le_bytes());
    buf.extend_from_slice(&(tombstones.len() as u64).to_le_bytes());
    buf.extend_from_slice(&(BLOCK_ROWS as u32).to_le_bytes());

    let ids: Vec<&[u8]> = docs.iter().map(|d| d.id.as_bytes()).collect();
    let id_offsets = put_blocks(&mut buf, &ids)?;

    let vectors_offset = buf.len() as u64;
    for doc in &docs {
        for x in &doc.vector {
            buf.extend_from_slice(&x.to_le_bytes());
        }
    }

    // A block of documents per chunk, in parallel on a pool thread;
    // the chunks come back in order, so the bytes are the same either way.
    let mut attrs: Vec<Vec<u8>> = Vec::with_capacity(docs.len());
    for chunk in crate::cpu::chunked(&docs, BLOCK_ROWS, |chunk| {
        chunk
            .iter()
            .map(|doc| match &doc.attrs {
                None => Ok(Vec::new()),
                Some(value) => serde_json::to_vec(value).map_err(|source| ItzmaError::Serde {
                    path: "segment".into(),
                    source,
                }),
            })
            .collect::<Result<Vec<_>>>()
    }) {
        attrs.extend(chunk?);
    }
    let attr_offsets = put_blocks(&mut buf, &attrs)?;

    let dead: Vec<&[u8]> = tombstones.iter().map(|t| t.as_bytes()).collect();
    let tombstone_offsets = put_blocks(&mut buf, &dead)?;

    let bloom_offset = buf.len() as u64;
    let bloom = bloom_of(
        docs.iter()
            .map(|d| d.id.as_str())
            .chain(tombstones.iter().copied()),
    )
    .to_bytes();
    buf.extend_from_slice(&bloom);

    let directory_offset = buf.len() as u64;
    buf.extend_from_slice(&vectors_offset.to_le_bytes());
    buf.extend_from_slice(&bloom_offset.to_le_bytes());
    buf.extend_from_slice(&(bloom.len() as u64).to_le_bytes());
    put_offsets(&mut buf, &id_offsets);
    for first in ids.iter().step_by(BLOCK_ROWS) {
        put_bytes(&mut buf, first)?;
    }
    put_offsets(&mut buf, &attr_offsets);
    put_offsets(&mut buf, &tombstone_offsets);
    for first in dead.iter().step_by(BLOCK_ROWS) {
        put_bytes(&mut buf, first)?;
    }
    let directory_len = buf.len() as u64 - directory_offset;

    buf.extend_from_slice(&directory_offset.to_le_bytes());
    buf.extend_from_slice(&directory_len.to_le_bytes());
    buf.extend_from_slice(&END_MAGIC);
    Ok(Bytes::from(buf))
}

#[cfg(test)]
/// Write `rows` as blocks of `BLOCK_ROWS`. Returns where each block starts,
/// then where the section ends, so `n + 1` offsets for `n` blocks.
fn put_blocks<R: AsRef<[u8]>>(buf: &mut Vec<u8>, rows: &[R]) -> Result<Vec<u64>> {
    let mut offsets = Vec::with_capacity(rows.len().div_ceil(BLOCK_ROWS) + 1);
    for block in rows.chunks(BLOCK_ROWS) {
        offsets.push(buf.len() as u64);
        buf.extend_from_slice(&(block.len() as u32).to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        let mut end = 0usize;
        for row in block {
            end += row.as_ref().len();
            let end32 = u32::try_from(end).map_err(|_| too_long())?;
            buf.extend_from_slice(&end32.to_le_bytes());
        }
        for row in block {
            buf.extend_from_slice(row.as_ref());
        }
    }
    offsets.push(buf.len() as u64);
    Ok(offsets)
}

#[cfg(test)]
fn put_offsets(buf: &mut Vec<u8>, offsets: &[u64]) {
    for offset in offsets {
        buf.extend_from_slice(&offset.to_le_bytes());
    }
}

/// How a segment stores attributes. Version 2 is always `Json`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum AttrLayout {
    /// One JSON value per row (version 2, and a free-form namespace at 3).
    Json,
    /// Typed columns per block, for these fields in name order.
    Columns(Fields),
}

/// What the directory says, checked against nothing yet.
pub(super) struct Directory {
    pub(super) version: u32,
    pub(super) vectors_offset: u64,
    pub(super) bloom_offset: u64,
    pub(super) bloom_len: u64,
    pub(super) id_offsets: Vec<u64>,
    pub(super) id_firsts: Vec<String>,
    pub(super) attr_offsets: Vec<u64>,
    pub(super) tombstone_offsets: Vec<u64>,
    pub(super) tombstone_firsts: Vec<String>,
    /// How the attribute blocks are laid out.
    pub(crate) attrs: AttrLayout,
    /// Where each zone block starts, then where the section ends; empty at
    /// version 2.
    pub(super) zone_offsets: Vec<u64>,
}

impl Directory {
    /// Parse the directory bytes for an object with these counts. The block
    /// counts come from `doc_count` and `tombstone_count`, and the bytes must
    /// hold at least what those need before anything is allocated, so a forged
    /// count costs nothing.
    pub(super) fn parse(
        path: &str,
        bytes: &[u8],
        version: u32,
        doc_count: u64,
        tombstone_count: u64,
    ) -> Result<Self> {
        let blocks = |rows: u64| rows.div_ceil(BLOCK_ROWS as u64);
        let (n, m) = (blocks(doc_count), blocks(tombstone_count));
        // Three u64, then per section its `blocks + 1` offsets, and per first
        // id at least its length.
        let least = (3 * 8u64)
            .saturating_add((n + 1) * 8 + n * 4)
            .saturating_add((n + 1) * 8)
            .saturating_add((m + 1) * 8 + m * 4)
            // Version 3: the layout byte and the zone offsets.
            .saturating_add(if version >= 3 { 1 + (n + 1) * 8 } else { 0 });
        if least > bytes.len() as u64 {
            return Err(corrupt(path, "directory is too short for the counts"));
        }
        // Both fit in the directory, so they fit in usize.
        let (n, m) = (n as usize, m as usize);

        let mut r = Reader::new(path, bytes);
        let vectors_offset = r.u64()?;
        let bloom_offset = r.u64()?;
        let bloom_len = r.u64()?;
        let id_offsets = offsets(&mut r, n)?;
        let id_firsts = firsts(&mut r, n)?;
        let attr_offsets = offsets(&mut r, n)?;
        let tombstone_offsets = offsets(&mut r, m)?;
        let tombstone_firsts = firsts(&mut r, m)?;
        let (attrs, zone_offsets) = if version >= 3 {
            let attrs = match r.u8()? {
                0 => AttrLayout::Json,
                1 => AttrLayout::Columns(field_table(&mut r)?),
                other => {
                    return Err(corrupt(
                        path,
                        format!("attribute layout {other} is unknown"),
                    ));
                }
            };
            (attrs, offsets(&mut r, n)?)
        } else {
            (AttrLayout::Json, Vec::new())
        };
        if r.remaining() != 0 {
            return Err(corrupt(path, "directory has trailing bytes"));
        }
        Ok(Directory {
            version,
            attrs,
            zone_offsets,
            vectors_offset,
            bloom_offset,
            bloom_len,
            id_offsets,
            id_firsts,
            attr_offsets,
            tombstone_offsets,
            tombstone_firsts,
        })
    }

    /// The sections must follow each other exactly, from the end of the
    /// header to the start of the directory, with every block offset
    /// ascending inside its section.
    pub(super) fn check_layout(
        &self,
        path: &str,
        matrix_len: u64,
        directory_offset: u64,
    ) -> Result<()> {
        // A block holds at least a count and one offset, and an offset list
        // that goes backwards would make a slice that does not exist.
        for (what, list) in [
            ("id", &self.id_offsets),
            ("attribute", &self.attr_offsets),
            ("tombstone", &self.tombstone_offsets),
            ("zone", &self.zone_offsets),
        ] {
            if !list.is_sorted() {
                return Err(corrupt(
                    path,
                    format!("{what} block offsets are not ascending"),
                ));
            }
        }
        let (ids_start, ids_end) = ends(&self.id_offsets);
        let (attrs_start, attrs_end) = ends(&self.attr_offsets);
        let (tombstones_start, tombstones_end) = ends(&self.tombstone_offsets);
        let joins = [
            (
                ids_start == HEADER_LEN as u64,
                "ids do not start after the header",
            ),
            (
                ids_end == self.vectors_offset,
                "ids do not end where the vectors start",
            ),
            (
                self.vectors_offset.checked_add(matrix_len) == Some(attrs_start),
                "the vectors do not end where the attributes start",
            ),
            (
                attrs_end == tombstones_start,
                "attributes do not end where the tombstones start",
            ),
            (
                tombstones_end == self.bloom_offset,
                "tombstones do not end where the Bloom filter starts",
            ),
        ];
        let bloom_end = self.bloom_offset.checked_add(self.bloom_len);
        let tail_joins: [(bool, &str); 2] = if self.version >= 3 {
            let (zone_start, zone_end) = ends(&self.zone_offsets);
            [
                (
                    bloom_end == Some(zone_start),
                    "the Bloom filter does not end where the zone section starts",
                ),
                (
                    zone_end == directory_offset,
                    "the zone section does not end where the directory starts",
                ),
            ]
        } else {
            [
                (
                    bloom_end == Some(directory_offset),
                    "the Bloom filter does not end where the directory starts",
                ),
                (true, ""),
            ]
        };
        match joins.iter().chain(&tail_joins).find(|(ok, _)| !ok) {
            Some((_, reason)) => Err(corrupt(path, *reason)),
            None => Ok(()),
        }
    }
}

/// The first and the last of a nonempty offset list. The list always has
/// `blocks + 1` entries, so it is never empty.
fn ends(offsets: &[u64]) -> (u64, u64) {
    (
        offsets.first().copied().unwrap_or(0),
        offsets.last().copied().unwrap_or(0),
    )
}

/// A version 3 directory's field table: u16 count, then per field a u8 name
/// length, the name and a tag byte, names strictly ascending.
fn field_table(r: &mut Reader<'_>) -> Result<Fields> {
    let count = r.u16()? as usize;
    let mut fields: Vec<Field> = Vec::with_capacity(count.min(r.remaining() / 2));
    for _ in 0..count {
        let len = r.u8()? as usize;
        let name = std::str::from_utf8(r.take(len)?)
            .map_err(|_| r.corrupt("a field name is not valid UTF-8"))?
            .to_owned();
        let tag = r.u8()?;
        let ty =
            type_of_tag(tag).ok_or_else(|| r.corrupt(format!("field tag {tag} is unknown")))?;
        if fields.last().is_some_and(|f| f.name >= name) {
            return Err(r.corrupt("field names are not strictly ascending"));
        }
        fields.push(Field { name, ty });
    }
    Ok(fields.into())
}

fn offsets(r: &mut Reader<'_>, blocks: usize) -> Result<Vec<u64>> {
    (0..=blocks).map(|_| r.u64()).collect()
}

fn firsts(r: &mut Reader<'_>, blocks: usize) -> Result<Vec<String>> {
    (0..blocks).map(|_| r.string("first id")).collect()
}

/// A version 2 object's sections, every block split into rows and checked
/// against the directory, nothing converted yet. [`decode`] and the segment
/// view both start here.
pub(super) struct Parts<'a> {
    pub(super) ids: Vec<&'a [u8]>,
    pub(super) matrix: &'a [u8],
    pub(super) attrs: AttrSection<'a>,
    pub(super) tombstones: Vec<&'a [u8]>,
    pub(super) bloom: &'a [u8],
    /// One zone block per id block, version 3 only.
    pub(super) zone: Vec<&'a [u8]>,
}

/// The attribute section of an object, split but not decoded.
pub(super) enum AttrSection<'a> {
    /// One JSON value per row, empty for none.
    Rows(Vec<&'a [u8]>),
    /// One columnar block per id block.
    Columns {
        fields: Fields,
        blocks: Vec<&'a [u8]>,
    },
}

/// Parse a version 2 or 3 object into its [`Parts`], `r` being just past
/// the version.
pub(super) fn parts<'a>(mut r: Reader<'a>, version: u32, dim: usize) -> Result<Parts<'a>> {
    let (path, bytes) = (r.path, r.buf);
    if bytes.len() < HEADER_LEN + TRAILER_LEN {
        return Err(r.corrupt("object is shorter than a header and a trailer"));
    }
    let seg_dim = r.u32()?;
    if usize::try_from(seg_dim).ok() != Some(dim) {
        return Err(r.corrupt(format!("dimension {seg_dim}, namespace has {dim}")));
    }
    let doc_count = r.u64()?;
    let tombstone_count = r.u64()?;
    let block_rows = r.u32()?;
    if block_rows as usize != BLOCK_ROWS {
        return Err(r.corrupt(format!("block_rows is {block_rows}, expected {BLOCK_ROWS}")));
    }

    // A document needs its two row offsets and its vector, and a tombstone its
    // row offset, so counts the object cannot hold are refused before anything
    // is sized from them.
    let per_doc = (dim as u64).saturating_mul(4).saturating_add(8);
    let needed = doc_count
        .saturating_mul(per_doc)
        .saturating_add(tombstone_count.saturating_mul(4));
    if needed > bytes.len() as u64 {
        return Err(r.corrupt("counts exceed the object size"));
    }
    // Both counts are bounded by the object's length, so they fit in usize.
    let (docs_n, tombstones_n) = (doc_count as usize, tombstone_count as usize);

    let body_end = bytes.len() - TRAILER_LEN;
    let mut trailer = Reader::new(
        path,
        slice(path, bytes, body_end as u64, bytes.len() as u64)?,
    );
    let directory_offset = trailer.u64()?;
    let directory_len = trailer.u64()?;
    if trailer.take(4)? != END_MAGIC {
        return Err(r.corrupt("bad trailer magic"));
    }
    if directory_offset.checked_add(directory_len) != Some(body_end as u64) {
        return Err(r.corrupt("directory does not end where the trailer starts"));
    }
    let directory = Directory::parse(
        path,
        slice(path, bytes, directory_offset, body_end as u64)?,
        version,
        doc_count,
        tombstone_count,
    )?;
    directory.check_layout(path, doc_count * dim as u64 * 4, directory_offset)?;

    let ids = read_blocks(path, bytes, &directory.id_offsets, docs_n, "id")?;
    check_firsts(path, &ids, &directory.id_firsts, "id")?;
    let matrix = slice(
        path,
        bytes,
        directory.vectors_offset,
        directory.attr_offsets.first().copied().unwrap_or(0),
    )?;
    let attrs = match &directory.attrs {
        AttrLayout::Json => AttrSection::Rows(read_blocks(
            path,
            bytes,
            &directory.attr_offsets,
            docs_n,
            "attribute",
        )?),
        AttrLayout::Columns(fields) => AttrSection::Columns {
            fields: fields.clone(),
            blocks: directory
                .attr_offsets
                .windows(2)
                .map(|pair| slice(path, bytes, pair[0], pair[1]))
                .collect::<Result<_>>()?,
        },
    };
    let zone = directory
        .zone_offsets
        .windows(2)
        .map(|pair| slice(path, bytes, pair[0], pair[1]))
        .collect::<Result<_>>()?;
    let tombstones = read_blocks(
        path,
        bytes,
        &directory.tombstone_offsets,
        tombstones_n,
        "tombstone",
    )?;
    check_firsts(path, &tombstones, &directory.tombstone_firsts, "tombstone")?;
    let bloom = slice(
        path,
        bytes,
        directory.bloom_offset,
        directory.bloom_offset + directory.bloom_len,
    )?;
    Ok(Parts {
        ids,
        matrix,
        attrs,
        tombstones,
        bloom,
        zone,
    })
}

/// Decode the rest of a version 2 or 3 object, `r` being just past the
/// version. At version 3 every zone block must be the one its attribute block
/// gives, so a damaged zone map is caught here.
pub(super) fn decode(r: Reader<'_>, version: u32, dim: usize) -> Result<Layer> {
    let path = r.path;
    let parts = parts(r, version, dim)?;
    let docs_n = parts.ids.len();
    let ids = strings(path, &parts.ids, "id")?;
    let mut floats = parts
        .matrix
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    let parsed = match &parts.attrs {
        AttrSection::Rows(rows) => {
            // Parsed a block per chunk, in parallel on a pool thread.
            // The first error in row order is the one returned.
            let mut parsed: Vec<Option<serde_json::Value>> = Vec::with_capacity(docs_n);
            for chunk in crate::cpu::chunked(rows, BLOCK_ROWS, |chunk| {
                chunk
                    .iter()
                    .map(|raw| {
                        if raw.is_empty() {
                            return Ok(None);
                        }
                        serde_json::from_slice(raw).map(Some).map_err(|e| {
                            corrupt(path, format!("attributes are not valid JSON: {e}"))
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            }) {
                parsed.extend(chunk?);
            }
            let mut empty = Vec::new();
            encode_empty_zone(&mut empty);
            if parts.zone.iter().any(|zone| *zone != empty.as_slice()) {
                return Err(corrupt(
                    path,
                    "a zone block does not match its attribute block",
                ));
            }
            parsed
        }
        AttrSection::Columns { fields, blocks } => {
            let mut parsed = Vec::with_capacity(docs_n);
            for (i, (block, zone)) in blocks.iter().zip(&parts.zone).enumerate() {
                let rows = (docs_n - i * BLOCK_ROWS).min(BLOCK_ROWS);
                let (columns, ranges) = ColumnBlock::decode(path, fields, block, rows)?;
                let mut expected = Vec::new();
                encode_zone(&columns, &ranges, &mut expected);
                if *zone != expected.as_slice() {
                    return Err(corrupt(
                        path,
                        "a zone block does not match its attribute block",
                    ));
                }
                parsed.extend((0..rows).map(|row| columns.row(row)));
            }
            parsed
        }
    };

    let mut docs = Vec::with_capacity(docs_n);
    for (id, attrs) in ids.into_iter().zip(parsed) {
        let vector: Vec<f32> = floats.by_ref().take(dim).collect();
        docs.push(Document { id, vector, attrs });
    }

    let tombstones = strings(path, &parts.tombstones, "tombstone")?;

    // Across the whole list, so a block that starts before the one above it
    // is caught as well as two ids the wrong way round inside one.
    if !docs.is_sorted_by(|a, b| a.id < b.id) {
        return Err(corrupt(path, "ids are not in strictly ascending order"));
    }
    if !tombstones.is_sorted_by(|a, b| a < b) {
        return Err(corrupt(
            path,
            "tombstones are not in strictly ascending order",
        ));
    }
    if overlaps(&docs, &tombstones) {
        return Err(corrupt(path, "an id is both a document and a tombstone"));
    }

    let stored = Bloom::from_bytes(parts.bloom)
        .ok_or_else(|| corrupt(path, "the Bloom filter is malformed"))?;
    let rebuilt = bloom_of(
        docs.iter()
            .map(|d| d.id.as_str())
            .chain(tombstones.iter().map(String::as_str)),
    );
    if stored != rebuilt {
        return Err(corrupt(path, "the Bloom filter does not match the ids"));
    }
    Ok(Layer::from_parts(docs, tombstones))
}

/// `bytes[start..end]`, or `CorruptSegment` if it is not inside the object.
fn slice<'a>(path: &str, bytes: &'a [u8], start: u64, end: u64) -> Result<&'a [u8]> {
    let range = usize::try_from(start).ok().zip(usize::try_from(end).ok());
    range
        .and_then(|(start, end)| bytes.get(start..end))
        .ok_or_else(|| corrupt(path, "an offset is outside the object"))
}

/// Every row of one section, in order. `rows` is how many the section holds,
/// which fixes how many each block must hold: 256, and the remainder in the
/// last one.
fn read_blocks<'a>(
    path: &str,
    bytes: &'a [u8],
    offsets: &[u64],
    rows: usize,
    what: &str,
) -> Result<Vec<&'a [u8]>> {
    let mut out = Vec::with_capacity(rows);
    for (i, pair) in offsets.windows(2).enumerate() {
        let expected = (rows - i * BLOCK_ROWS).min(BLOCK_ROWS);
        let block = slice(path, bytes, pair[0], pair[1])?;
        read_block(
            path,
            block,
            expected,
            &format!("{what} block {i}"),
            &mut out,
        )?;
    }
    Ok(out)
}

/// Split one block into its rows, adding them to `out`.
pub(super) fn read_block<'a>(
    path: &str,
    block: &'a [u8],
    expected: usize,
    what: &str,
    out: &mut Vec<&'a [u8]>,
) -> Result<()> {
    if block.len() < 8 {
        return Err(corrupt(path, format!("{what} is shorter than 8 bytes")));
    }
    let mut r = Reader::new(path, block);
    let count = r.u32()? as usize;
    if count != expected {
        return Err(corrupt(
            path,
            format!("{what} holds {count} rows, expected {expected}"),
        ));
    }
    // At most 256 offsets, and the block has to hold them.
    let offsets = (0..=count)
        .map(|_| r.u32().map(|o| o as usize))
        .collect::<Result<Vec<_>>>()?;
    let data = block.get(r.pos..).unwrap_or_default();
    let adds_up =
        offsets.first() == Some(&0) && offsets.is_sorted() && offsets.last() == Some(&data.len());
    if !adds_up {
        return Err(corrupt(
            path,
            format!("{what} has offsets that do not add up"),
        ));
    }
    for pair in offsets.windows(2) {
        out.push(slice(path, data, pair[0] as u64, pair[1] as u64)?);
    }
    Ok(())
}

/// The directory keeps the first id of every block, and it must be the
/// block's own.
fn check_firsts(path: &str, rows: &[&[u8]], firsts: &[String], what: &str) -> Result<()> {
    for (i, first) in firsts.iter().enumerate() {
        if rows.get(i * BLOCK_ROWS) != Some(&first.as_bytes()) {
            return Err(corrupt(
                path,
                format!("the directory's first {what} of block {i} is not the block's"),
            ));
        }
    }
    Ok(())
}

fn strings(path: &str, rows: &[&[u8]], what: &str) -> Result<Vec<String>> {
    rows.iter()
        .map(|row| {
            std::str::from_utf8(row)
                .map(str::to_owned)
                .map_err(|_| corrupt(path, format!("{what} is not valid UTF-8")))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::document::WalOp;
    use crate::segment::{decode as decode_any, encode_as};

    fn encode(layer: &Layer, dim: usize) -> Result<Bytes> {
        encode_as(layer, dim, VERSION)
    }

    fn up(id: &str, v: &[f32], attrs: Option<serde_json::Value>) -> WalOp {
        WalOp::Upsert(Document {
            id: id.into(),
            vector: v.to_vec(),
            attrs,
        })
    }

    fn layer(ops: Vec<WalOp>) -> Layer {
        let mut l = Layer::new();
        for op in ops {
            l.apply(op);
        }
        l
    }

    /// `n` documents of `dim` values, each with attributes that differ.
    fn attr_heavy_layer(n: usize, dim: usize) -> Layer {
        layer(
            (0..n)
                .map(|i| {
                    let attrs =
                        (i % 5 != 0).then(|| json!({ "n": i, "tag": format!("t{}", i % 13) }));
                    up(&format!("d{i:06}"), &vec![i as f32; dim], attrs)
                })
                .collect(),
        )
    }

    #[test]
    fn chunked_attribute_work_gives_the_same_bytes_in_and_out_of_a_pool() {
        let original = attr_heavy_layer(BLOCK_ROWS * 9 + 3, 4);
        let outside = encode(&original, 4).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .thread_name(|i| format!("itzma-maint-{i}"))
            .build()
            .unwrap();
        let inside = pool.install(|| encode(&original, 4)).unwrap();
        assert_eq!(inside, outside);
        let back_outside = decode_any("p", &outside, 4).unwrap();
        let back_inside = pool.install(|| decode_any("p", &outside, 4)).unwrap();
        assert_eq!(back_inside, back_outside);
        assert_eq!(back_outside, original);
    }

    fn sample() -> Layer {
        layer(vec![
            up("b", &[1.0, 2.0], Some(json!({"k": [1, 2], "s": "x"}))),
            up("a", &[-0.5, 0.25], None),
            up("ü-日本", &[3.0, 4.0], Some(json!("plain string"))),
            WalOp::Delete("y".into()),
            WalOp::Delete("x".into()),
        ])
    }

    /// Documents `d0000` upward and tombstones `t0000` upward, all ids five
    /// bytes long, with a mix of no attributes, an object and JSON null.
    fn numbered(docs: usize, tombstones: usize) -> Layer {
        let mut l = Layer::new();
        for i in 0..docs {
            let attrs = match i % 3 {
                0 => None,
                1 => Some(json!({ "i": i })),
                _ => Some(serde_json::Value::Null),
            };
            l.apply(up(&format!("d{i:04}"), &[i as f32, 0.5], attrs));
        }
        for i in 0..tombstones {
            l.apply(WalOp::Delete(format!("t{i:04}")));
        }
        l
    }

    fn enc(l: &Layer) -> Vec<u8> {
        encode(l, 2).unwrap().to_vec()
    }

    fn dec(bytes: &[u8]) -> Result<Layer> {
        decode_any("p", bytes, 2)
    }

    fn reason(result: Result<Layer>) -> String {
        match result.unwrap_err() {
            ItzmaError::CorruptSegment { reason, .. } => reason,
            other => panic!("expected CorruptSegment, got {other:?}"),
        }
    }

    fn rejects(bytes: &[u8], needle: &str) {
        let got = reason(dec(bytes));
        assert!(got.contains(needle), "wanted {needle:?}, got {got:?}");
    }

    fn u32_at(bytes: &[u8], pos: usize) -> u32 {
        u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap())
    }

    fn u64_at(bytes: &[u8], pos: usize) -> u64 {
        u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap())
    }

    fn set_u32(bytes: &mut [u8], pos: usize, value: u32) {
        bytes[pos..pos + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn set_u64(bytes: &mut [u8], pos: usize, value: u64) {
        bytes[pos..pos + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn bump(bytes: &mut [u8], pos: usize, by: u64) {
        let value = u64_at(bytes, pos) + by;
        set_u64(bytes, pos, value);
    }

    /// Where things sit in an object whose ids are all five bytes long, so a
    /// test can patch one field.
    struct At {
        dir: usize,
        n: usize,
        m: usize,
    }

    impl At {
        fn of(bytes: &[u8]) -> At {
            let blocks = |rows: u64| rows.div_ceil(BLOCK_ROWS as u64) as usize;
            At {
                dir: u64_at(bytes, bytes.len() - TRAILER_LEN) as usize,
                n: blocks(u64_at(bytes, 12)),
                m: blocks(u64_at(bytes, 20)),
            }
        }

        fn vectors_offset(&self) -> usize {
            self.dir
        }

        fn bloom_offset(&self) -> usize {
            self.dir + 8
        }

        fn bloom_len(&self) -> usize {
            self.dir + 16
        }

        fn id_offset(&self, i: usize) -> usize {
            self.dir + 24 + 8 * i
        }

        /// The first byte of the i-th first id in the directory.
        fn id_first(&self, i: usize) -> usize {
            self.id_offset(self.n + 1) + 9 * i + 4
        }

        fn attr_offset(&self, i: usize) -> usize {
            self.id_offset(self.n + 1) + 9 * self.n + 8 * i
        }

        fn tombstone_offset(&self, i: usize) -> usize {
            self.attr_offset(self.n + 1) + 8 * i
        }

        fn tombstone_first(&self, i: usize) -> usize {
            self.tombstone_offset(self.m + 1) + 9 * i + 4
        }
    }

    /// Where the rows of an id or tombstone block start, given its offset
    /// entry in the directory and how many rows it holds.
    fn rows_at(bytes: &[u8], offset_entry: usize, count: usize) -> usize {
        u64_at(bytes, offset_entry) as usize + 4 + 4 * (count + 1)
    }

    fn directory(bytes: &[u8]) -> Directory {
        let at = At::of(bytes);
        let body = &bytes[at.dir..bytes.len() - TRAILER_LEN];
        Directory::parse("p", body, VERSION, u64_at(bytes, 12), u64_at(bytes, 20)).unwrap()
    }

    #[test]
    fn round_trips_layers_of_every_shape() {
        for original in [
            Layer::new(),
            sample(),
            numbered(3, 2),
            numbered(255, 255),
            numbered(256, 256),
            numbered(257, 257),
            numbered(300, 300),
            numbered(0, 600),
            numbered(600, 0),
        ] {
            let bytes = enc(&original);
            assert_eq!(dec(&bytes).unwrap(), original);
        }
    }

    #[test]
    fn json_null_attributes_survive() {
        let original = layer(vec![up("a", &[1.0, 2.0], Some(serde_json::Value::Null))]);
        let back = dec(&enc(&original)).unwrap();
        assert_eq!(back.sorted_docs()[0].attrs, Some(serde_json::Value::Null));
    }

    #[test]
    fn an_empty_layer_is_header_bloom_directory_and_trailer() {
        // 32 header, 16 for a one-word filter, 48 of directory (three fields
        // and one offset for each of the three sections), 20 trailer.
        assert_eq!(enc(&Layer::new()).len(), 32 + 16 + 48 + 20);
    }

    #[test]
    fn bytes_do_not_depend_on_insertion_order() {
        let one = layer(vec![
            up("a", &[1.0, 0.0], None),
            up("b", &[2.0, 0.0], None),
            WalOp::Delete("z".into()),
        ]);
        let two = layer(vec![
            WalOp::Delete("z".into()),
            up("b", &[2.0, 0.0], None),
            up("a", &[1.0, 0.0], None),
        ]);
        assert_eq!(enc(&one), enc(&two));
    }

    #[test]
    fn a_full_block_is_one_block_and_one_more_row_makes_two() {
        let cases = [
            (0, 0),
            (1, 1),
            (255, 255),
            (256, 256),
            (257, 257),
            (300, 300),
        ];
        for (docs, tombstones) in cases {
            let d = directory(&enc(&numbered(docs, tombstones)));
            assert_eq!(d.id_offsets.len(), docs.div_ceil(256) + 1, "{docs} docs");
            assert_eq!(d.attr_offsets.len(), d.id_offsets.len());
            assert_eq!(d.id_firsts.len(), docs.div_ceil(256));
            assert_eq!(d.tombstone_offsets.len(), tombstones.div_ceil(256) + 1);
            assert_eq!(d.tombstone_firsts.len(), tombstones.div_ceil(256));
        }
        let d = directory(&enc(&numbered(300, 300)));
        assert_eq!(d.id_firsts, ["d0000", "d0256"]);
        assert_eq!(d.tombstone_firsts, ["t0000", "t0256"]);
    }

    #[test]
    fn layout_matches_adr_018_byte_for_byte() {
        let l = layer(vec![up("a", &[1.0], None), WalOp::Delete("b".into())]);
        let mut want = Vec::new();
        want.extend_from_slice(b"ITZS");
        want.extend_from_slice(&2u32.to_le_bytes()); // version
        want.extend_from_slice(&1u32.to_le_bytes()); // dim
        want.extend_from_slice(&1u64.to_le_bytes()); // doc_count
        want.extend_from_slice(&1u64.to_le_bytes()); // tombstone_count
        want.extend_from_slice(&256u32.to_le_bytes()); // block_rows
        // ids block, at 32: count, offsets 0 and 1, then the id
        want.extend_from_slice(&1u32.to_le_bytes());
        want.extend_from_slice(&0u32.to_le_bytes());
        want.extend_from_slice(&1u32.to_le_bytes());
        want.extend_from_slice(b"a");
        // vectors, at 45
        want.extend_from_slice(&1.0f32.to_le_bytes());
        // attributes block, at 49: one row, no bytes
        want.extend_from_slice(&1u32.to_le_bytes());
        want.extend_from_slice(&0u32.to_le_bytes());
        want.extend_from_slice(&0u32.to_le_bytes());
        // tombstones block, at 61
        want.extend_from_slice(&1u32.to_le_bytes());
        want.extend_from_slice(&0u32.to_le_bytes());
        want.extend_from_slice(&1u32.to_le_bytes());
        want.extend_from_slice(b"b");
        // Bloom filter, at 74: seven probes, one word (the word bloom.rs pins
        // for these two keys)
        want.extend_from_slice(&7u32.to_le_bytes());
        want.extend_from_slice(&1u32.to_le_bytes());
        want.extend_from_slice(&0x4002_0030_0180_1fc0u64.to_le_bytes());
        // directory, at 90
        want.extend_from_slice(&45u64.to_le_bytes()); // vectors_offset
        want.extend_from_slice(&74u64.to_le_bytes()); // bloom_offset
        want.extend_from_slice(&16u64.to_le_bytes()); // bloom_len
        want.extend_from_slice(&32u64.to_le_bytes()); // id offsets
        want.extend_from_slice(&45u64.to_le_bytes());
        want.extend_from_slice(&1u32.to_le_bytes()); // first id of block 0
        want.extend_from_slice(b"a");
        want.extend_from_slice(&49u64.to_le_bytes()); // attribute offsets
        want.extend_from_slice(&61u64.to_le_bytes());
        want.extend_from_slice(&61u64.to_le_bytes()); // tombstone offsets
        want.extend_from_slice(&74u64.to_le_bytes());
        want.extend_from_slice(&1u32.to_le_bytes()); // first tombstone of block 0
        want.extend_from_slice(b"b");
        // trailer
        want.extend_from_slice(&90u64.to_le_bytes());
        want.extend_from_slice(&82u64.to_le_bytes());
        want.extend_from_slice(b"ITZE");
        assert_eq!(encode(&l, 1).unwrap().to_vec(), want);
    }

    #[test]
    fn encode_rejects_a_vector_of_the_wrong_length() {
        let l = layer(vec![up("a", &[1.0, 2.0, 3.0], None)]);
        assert!(matches!(
            encode(&l, 2).unwrap_err(),
            ItzmaError::DimensionMismatch {
                expected: 2,
                got: 3
            }
        ));
    }

    #[test]
    fn every_truncation_is_an_error_and_none_panics() {
        for original in [sample(), numbered(300, 300)] {
            let bytes = enc(&original);
            for cut in 0..bytes.len() {
                assert!(dec(&bytes[..cut]).is_err(), "cut at {cut}");
            }
        }
    }

    #[test]
    fn a_flipped_bit_never_panics_and_never_gets_past_the_framing() {
        // Every bit of a small object. The header, the Bloom filter, the
        // directory and the trailer are all cross-checked, so damage there
        // must be an error. Damage inside a vector or a row may decode.
        let bytes = enc(&sample());
        let at = u64_at(&bytes, bytes.len() - TRAILER_LEN) as usize;
        let bloom = u64_at(&bytes, at + 8) as usize;
        for bit in 0..bytes.len() * 8 {
            let mut hurt = bytes.clone();
            hurt[bit / 8] ^= 1 << (bit % 8);
            let checked = bit / 8 < HEADER_LEN || bit / 8 >= bloom;
            if checked {
                assert!(dec(&hurt).is_err(), "bit {bit} of byte {}", bit / 8);
            } else {
                let _ = dec(&hurt);
            }
        }
    }

    #[test]
    fn a_flipped_bit_in_the_framing_of_a_two_block_object_is_an_error() {
        let bytes = enc(&numbered(300, 300));
        let bloom = u64_at(&bytes, At::of(&bytes).bloom_offset()) as usize;
        let framing = (0..HEADER_LEN).chain(bloom..bytes.len());
        for byte in framing {
            for bit in 0..8 {
                let mut hurt = bytes.clone();
                hurt[byte] ^= 1 << bit;
                assert!(dec(&hurt).is_err(), "bit {bit} of byte {byte}");
            }
        }
    }

    #[test]
    fn a_short_object_is_refused() {
        let bytes = enc(&sample());
        rejects(&bytes[..HEADER_LEN + TRAILER_LEN - 1], "shorter");
    }

    #[test]
    fn a_dimension_other_than_the_namespaces_is_refused() {
        let bytes = enc(&sample());
        let got = reason(decode_any("p", &bytes, 3));
        assert!(got.contains("dimension"), "{got}");
    }

    #[test]
    fn a_block_size_other_than_256_is_refused() {
        let mut bytes = enc(&sample());
        set_u32(&mut bytes, 28, 128);
        rejects(&bytes, "block_rows");
    }

    #[test]
    fn counts_the_object_cannot_hold_are_refused() {
        for pos in [12, 20] {
            let mut bytes = enc(&sample());
            set_u64(&mut bytes, pos, u64::MAX);
            rejects(&bytes, "counts");
            let mut bytes = enc(&sample());
            set_u64(&mut bytes, pos, 1 << 40);
            rejects(&bytes, "counts");
        }
    }

    #[test]
    fn a_directory_too_short_for_its_counts_is_refused() {
        // 600 tombstones make three blocks. Claiming 1500 passes the size
        // bound (4 bytes each) but needs seven blocks of directory.
        let mut bytes = enc(&numbered(0, 600));
        set_u64(&mut bytes, 20, 1500);
        rejects(&bytes, "directory is too short");
    }

    #[test]
    fn a_bad_trailer_is_refused() {
        let mut bytes = enc(&sample());
        *bytes.last_mut().unwrap() = b'X';
        rejects(&bytes, "trailer magic");

        let mut bytes = enc(&sample());
        bytes.push(0);
        rejects(&bytes, "trailer magic");

        let mut bytes = enc(&sample());
        let len_at = bytes.len() - 16;
        bump(&mut bytes, len_at, 1);
        rejects(&bytes, "directory does not end");

        let mut bytes = enc(&sample());
        let offset_at = bytes.len() - TRAILER_LEN;
        bump(&mut bytes, offset_at, 1);
        rejects(&bytes, "directory does not end");
    }

    #[test]
    fn block_offsets_that_go_backwards_are_refused() {
        let mut bytes = enc(&numbered(300, 300));
        let at = At::of(&bytes);
        set_u64(&mut bytes, at.id_offset(1), 0);
        rejects(&bytes, "id block offsets are not ascending");

        let mut bytes = enc(&numbered(300, 300));
        set_u64(&mut bytes, at.attr_offset(1), 0);
        rejects(&bytes, "attribute block offsets are not ascending");

        let mut bytes = enc(&numbered(300, 300));
        set_u64(&mut bytes, at.tombstone_offset(1), 0);
        rejects(&bytes, "tombstone block offsets are not ascending");
    }

    #[test]
    fn a_block_shorter_than_its_header_is_refused() {
        let mut bytes = enc(&numbered(300, 300));
        let at = At::of(&bytes);
        let start = u64_at(&bytes, at.id_offset(0));
        set_u64(&mut bytes, at.id_offset(1), start + 4);
        rejects(&bytes, "shorter than 8 bytes");
    }

    #[test]
    fn sections_that_do_not_touch_are_refused() {
        let bytes = enc(&numbered(300, 300));
        let at = At::of(&bytes);
        let cases: [(usize, i64, &str); 6] = [
            (at.id_offset(0), 1, "ids do not start after the header"),
            (
                at.vectors_offset(),
                1,
                "ids do not end where the vectors start",
            ),
            (
                at.attr_offset(0),
                4,
                "vectors do not end where the attributes start",
            ),
            (
                at.tombstone_offset(0),
                4,
                "attributes do not end where the tombstones start",
            ),
            (
                at.bloom_offset(),
                8,
                "tombstones do not end where the Bloom filter starts",
            ),
            (
                at.bloom_len(),
                8,
                "Bloom filter does not end where the directory starts",
            ),
        ];
        for (pos, by, want) in cases {
            let mut hurt = bytes.clone();
            bump(&mut hurt, pos, by as u64);
            rejects(&hurt, want);
        }
    }

    #[test]
    fn a_block_with_the_wrong_row_count_is_refused() {
        let bytes = enc(&numbered(300, 300));
        let at = At::of(&bytes);
        for (entry, was, want) in [
            (at.id_offset(0), 256, "holds 255 rows, expected 256"),
            (at.id_offset(1), 44, "holds 43 rows, expected 44"),
            (at.attr_offset(1), 44, "holds 43 rows, expected 44"),
            (at.tombstone_offset(0), 256, "holds 255 rows, expected 256"),
        ] {
            let mut hurt = bytes.clone();
            let block = u64_at(&hurt, entry) as usize;
            assert_eq!(u32_at(&hurt, block), was);
            set_u32(&mut hurt, block, was - 1);
            rejects(&hurt, want);
        }
    }

    #[test]
    fn block_offsets_that_do_not_add_up_are_refused() {
        let bytes = enc(&numbered(300, 300));
        let at = At::of(&bytes);
        let block = u64_at(&bytes, at.id_offset(0)) as usize;

        let mut hurt = bytes.clone();
        set_u32(&mut hurt, block + 4, 1); // the first offset is not 0
        rejects(&hurt, "offsets that do not add up");

        let mut hurt = bytes.clone();
        set_u32(&mut hurt, block + 4 + 4 * 2, 1); // goes backwards
        rejects(&hurt, "offsets that do not add up");

        let mut hurt = bytes.clone();
        let last = block + 4 + 4 * 256;
        let now = u32_at(&hurt, last);
        set_u32(&mut hurt, last, now + 1); // longer than the data
        rejects(&hurt, "offsets that do not add up");
    }

    #[test]
    fn an_id_or_tombstone_that_is_not_utf8_is_refused() {
        let bytes = enc(&numbered(10, 10));
        let at = At::of(&bytes);
        let ids = rows_at(&bytes, at.id_offset(0), 10);
        let mut hurt = bytes.clone();
        hurt[ids + 5 * 4] = 0xff;
        rejects(&hurt, "id is not valid UTF-8");

        let tombstones = rows_at(&bytes, at.tombstone_offset(0), 10);
        let mut hurt = bytes.clone();
        hurt[tombstones + 5 * 4] = 0xff;
        rejects(&hurt, "tombstone is not valid UTF-8");
    }

    #[test]
    fn attributes_that_are_not_json_are_refused() {
        let one = layer(vec![up("a", &[1.0, 1.0], Some(json!({"k": 1})))]);
        let mut bytes = enc(&one);
        let bloom = u64_at(&bytes, At::of(&bytes).bloom_offset()) as usize;
        bytes[bloom - 1] = b'x'; // the closing brace, last byte before the Bloom filter
        rejects(&bytes, "attributes are not valid JSON");
    }

    #[test]
    fn a_first_id_that_is_not_the_blocks_is_refused() {
        let bytes = enc(&numbered(300, 300));
        let at = At::of(&bytes);
        for (pos, want) in [
            (at.id_first(1), "first id of block 1"),
            (at.tombstone_first(1), "first tombstone of block 1"),
        ] {
            let mut hurt = bytes.clone();
            hurt[pos] = b'e';
            rejects(&hurt, want);
        }
    }

    #[test]
    fn ids_out_of_order_inside_a_block_are_refused() {
        let bytes = enc(&numbered(300, 300));
        let at = At::of(&bytes);
        for (entry, want) in [
            (at.id_offset(0), "ids are not in strictly ascending order"),
            (
                at.tombstone_offset(0),
                "tombstones are not in strictly ascending order",
            ),
        ] {
            let rows = rows_at(&bytes, entry, 256);
            let mut hurt = bytes.clone();
            // rows 1 and 2, so the block's first id still matches the directory
            let (a, b) = (rows + 5, rows + 10);
            for i in 0..5 {
                hurt.swap(a + i, b + i);
            }
            rejects(&hurt, want);
        }
    }

    #[test]
    fn ids_out_of_order_across_two_blocks_are_refused() {
        let bytes = enc(&numbered(300, 300));
        let at = At::of(&bytes);
        let cases = [
            (at.id_offset(1), at.id_first(1), "d0000", "ids are not in"),
            (
                at.tombstone_offset(1),
                at.tombstone_first(1),
                "t0000",
                "tombstones are not in",
            ),
        ];
        for (entry, copy, smaller, want) in cases {
            let mut hurt = bytes.clone();
            let first = rows_at(&hurt, entry, 44);
            // The second block starts with an id below the first block's last
            // one. The directory's copy is patched too, so only the order
            // check can catch it.
            hurt[first..first + 5].copy_from_slice(smaller.as_bytes());
            hurt[copy..copy + 5].copy_from_slice(smaller.as_bytes());
            rejects(&hurt, want);
        }
    }

    #[test]
    fn an_id_that_is_also_a_tombstone_is_refused() {
        let l = layer(vec![
            up("a", &[1.0, 1.0], None),
            up("c", &[1.0, 1.0], None),
            WalOp::Delete("b".into()),
        ]);
        let mut bytes = enc(&l);
        let bloom = u64_at(&bytes, At::of(&bytes).bloom_offset()) as usize;
        bytes[bloom - 1] = b'c'; // the tombstone row, last byte before the Bloom filter
        let copy = bytes.len() - TRAILER_LEN - 1; // its copy, last byte of the directory
        bytes[copy] = b'c';
        rejects(&bytes, "both a document and a tombstone");
    }

    #[test]
    fn a_damaged_bloom_filter_is_refused() {
        let bytes = enc(&numbered(10, 10));
        let at = At::of(&bytes);
        let bloom = u64_at(&bytes, at.bloom_offset()) as usize;

        let mut hurt = bytes.clone();
        hurt[bloom] ^= 1; // the probe count
        rejects(&hurt, "Bloom filter is malformed");

        // The right size, the wrong keys.
        let other: Vec<String> = (0..20).map(|i| format!("x{i:04}")).collect();
        let filter = bloom_of(other.iter().map(String::as_str)).to_bytes();
        let len = u64_at(&bytes, at.bloom_len()) as usize;
        assert_eq!(filter.len(), len);
        let mut hurt = bytes.clone();
        hurt[bloom..bloom + len].copy_from_slice(&filter);
        rejects(&hurt, "Bloom filter does not match the ids");
    }

    fn v3_schema() -> crate::document::Schema {
        crate::document::Schema::new(2, crate::Metric::L2).with_attrs(
            crate::attrs::AttrSchema::new()
                .field("n", crate::attrs::AttrType::Number)
                .field("s", crate::attrs::AttrType::Text),
        )
    }

    fn v3_object() -> Bytes {
        let mut l = Layer::new();
        for i in 0..300 {
            let attrs = json!({"n": i, "s": format!("v{}", i % 3)});
            l.apply(up(&format!("d{i:04}"), &[i as f32, 1.0], Some(attrs)));
        }
        l.apply(WalOp::Delete("t".into()));
        crate::segment::encode_with_schema(&l, &v3_schema(), 3).unwrap()
    }

    #[test]
    fn every_truncation_of_a_version_3_object_is_refused() {
        let bytes = v3_object();
        for cut in 0..bytes.len() {
            assert!(decode_any("t", &bytes[..cut], 2).is_err(), "cut at {cut}");
        }
    }

    #[test]
    fn every_flipped_bit_of_a_version_3_object_is_refused_or_canonical() {
        let bytes = v3_object();
        let n = bytes.len();
        let dir_off = u64::from_le_bytes(bytes[n - 20..n - 12].try_into().unwrap()) as usize;
        let dir = Directory::parse("t", &bytes[dir_off..n - TRAILER_LEN], 3, 300, 1).unwrap();
        // The field names: a flipped name that stays valid and ascending
        // decodes as another field, as a flipped key in a JSON row does.
        let table = n - TRAILER_LEN - dir.zone_offsets.len() * 8 - 8;
        let names = [table + 3, table + 6];
        for at in 0..n * 8 {
            let mut flipped = bytes.to_vec();
            flipped[at / 8] ^= 1 << (at % 8);
            // Nothing may panic, and what decodes must be exactly what the
            // encoder writes for that content, so a damaged object never
            // passes as another.
            let Ok(layer) = decode_any("t", &flipped, 2) else {
                continue;
            };
            match crate::segment::encode_with_schema(&layer, &v3_schema(), 3) {
                Ok(again) => assert_eq!(again.as_ref(), flipped.as_slice(), "bit {at}"),
                Err(_) => assert!(names.contains(&(at / 8)), "bit {at}"),
            }
        }
    }

    /// One edit per directory or zone check, at a position read off the
    /// object's own directory.
    #[test]
    fn forged_version_3_objects_are_refused_with_their_reason() {
        let bytes = v3_object();
        let n = bytes.len();
        let dir_end = n - TRAILER_LEN;
        let dir_off = u64::from_le_bytes(bytes[n - 20..n - 12].try_into().unwrap()) as usize;
        let dir = Directory::parse("t", &bytes[dir_off..dir_end], 3, 300, 1).unwrap();
        // Fields "n" and "s": count u16, then (len 1, "n", tag 1), (len 1, "s", tag 2),
        // then the zone offsets up to the directory's end.
        let table = dir_end - dir.zone_offsets.len() * 8 - 8;
        assert_eq!(&bytes[table..table + 8], &[2, 0, 1, b'n', 1, 1, b's', 2]);
        let zone_start = dir.zone_offsets[0] as usize;
        let cases: [(usize, u8, &str); 4] = [
            (table - 1, 7, "attribute layout 7"),
            (table + 3, b't', "strictly ascending"),
            (table + 4, 9, "field tag 9"),
            // The first zone entry's tag, just past the block's u32 length.
            (zone_start + 4, 3, "zone block"),
        ];
        for (at, value, want) in cases {
            let mut b = bytes.to_vec();
            b[at] = value;
            let got = match decode_any("t", &b, 2) {
                Err(ItzmaError::CorruptSegment { reason, .. }) => reason,
                other => panic!("{want}: {other:?}"),
            };
            assert!(got.contains(want), "{want:?} not in {got:?}");
        }
    }
}
