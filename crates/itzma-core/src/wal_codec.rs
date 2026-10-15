//! Encoding and decoding of one WAL object, format versions 2 and 3,
//! and the fence that garbage collection leaves
//! in place of an old one.
//!
//! Bytes in, bytes out: nothing here touches the object store. The layout is
//! a 13-byte header followed by a payload:
//!
//! | Field | Contents |
//! |---|---|
//! | Magic | `ITZW` |
//! | `format_version` | u32, 2 or 3 |
//! | Codec | u8: 0 payload is the body, 1 payload is a ZSTD frame of the body |
//! | `body_len` | u32, the body's length once decompressed |
//!
//! Version 3 is written for keyed batches. It has one more field after
//! `body_len`, the caller's idempotency key: a `key_len` byte (1 to 128) and
//! that many UTF-8 bytes. Everything after it is exactly the version 2 payload.
//! The key sits in the header, uncompressed, so a writer looking for an earlier
//! attempt can read it with a ranged GET of [`HEADER_PEEK_LEN`] bytes.
//!
//! The body is a fixed list of columns, all integers little-endian: counts
//! (`op_count`, `upsert_count`, `dim`), one kind byte per op, the ids, the
//! vectors as four byte planes, and the attributes as length-prefixed JSON.
//! Operation order and duplicates are kept, because the last op on an id in a
//! batch wins. A new column would be a new format version. Versions 4 and 5
//! are assigned elsewhere and are not read here
//! ([`ItzmaError::UnsupportedFormat`]).
//!
//! A fence is not a batch. It is 8 bytes, the magic `ITZF` and a u32 version
//! (1), and it stands at the key of a WAL object that garbage collection
//! removed. Put-if-absent still fails there, which is all it is for. It
//! carries no key and no ops, so decoding one is an error that says so
//! ([`fenced`]) and the reader decides what that means.

use std::borrow::Cow;
use std::io::Read;

use bytes::Bytes;

use crate::columns::{AttrColumn, DocColumns};
use crate::document::{Document, WalOp};
use crate::error::{ItzmaError, Result};
use crate::wal::WalEntry;

const MAGIC: [u8; 4] = *b"ITZW";
const FENCE_MAGIC: [u8; 4] = *b"ITZF";
const FENCE_VERSION: u32 = 1;
/// The size of a fence. Every real object is longer than its 13-byte header,
/// so a listing can tell a fence by size alone.
pub(crate) const FENCE_LEN: usize = 8;
const V2: u32 = 2;
const V3: u32 = 3;
/// The fixed part of a binary header: magic, version, codec and `body_len`.
pub(crate) const HEADER_LEN: usize = 13;
/// The longest idempotency key, in bytes.
pub(crate) const MAX_KEY_LEN: usize = 128;
/// How many bytes from the start of an object always hold its whole header,
/// key included.
pub(crate) const HEADER_PEEK_LEN: usize = HEADER_LEN + 1 + MAX_KEY_LEN;
pub(crate) const COUNTS_LEN: usize = 12;
const CODEC_RAW: u8 = 0;
const CODEC_ZSTD: u8 = 1;
/// Level 1, not ZSTD's default 3: the same size or smaller on SIFT and
/// GloVe batches and 30 to 50% faster to write. A frame decodes the same
/// whatever level wrote it.
const ZSTD_LEVEL: i32 = 1;
const KIND_UPSERT: u8 = 0;
const KIND_DELETE: u8 = 1;

/// Whether an object starts with the WAL magic.
pub(crate) fn is_binary(bytes: &[u8]) -> bool {
    bytes.starts_with(&MAGIC)
}

/// The fence object: `ITZF` and the version, 8 bytes.
pub(crate) fn fence() -> Bytes {
    let mut out = Vec::with_capacity(FENCE_LEN);
    out.extend_from_slice(&FENCE_MAGIC);
    out.extend_from_slice(&FENCE_VERSION.to_le_bytes());
    Bytes::from(out)
}

/// Whether an object starts with the fence magic. [`check_fence`] says whether
/// it is a good one.
pub(crate) fn is_fence(bytes: &[u8]) -> bool {
    bytes.starts_with(&FENCE_MAGIC)
}

/// Check that `bytes` are exactly a fence of a version this build knows.
///
/// # Errors
///
/// - [`ItzmaError::CorruptWal`] if the magic is wrong or the length is not
///   [`FENCE_LEN`].
/// - [`ItzmaError::UnsupportedFormat`] for a fence version other than 1.
pub(crate) fn check_fence(path: &str, bytes: &[u8]) -> Result<()> {
    if !is_fence(bytes) {
        return Err(corrupt(path, "bad fence magic"));
    }
    if bytes.len() != FENCE_LEN {
        return Err(corrupt(path, "a fence is 8 bytes"));
    }
    let mut head = Reader::new(path, bytes);
    head.take(FENCE_MAGIC.len())?;
    match head.u32()? {
        FENCE_VERSION => Ok(()),
        version => Err(ItzmaError::UnsupportedFormat { version }),
    }
}

/// The `reason` of the error for a fence found where a batch was expected.
/// Callers recognise it through [`is_fenced`], never by text.
const FENCED: &str = "a fence stands where a batch should be";

/// The error for reading a fence as a batch. A reader that loaded its manifest
/// before garbage collection ran can meet one, and reloads. Against
/// a current manifest it is damage, since only objects at or below the fold
/// point are ever fenced.
pub(crate) fn fenced(path: &str) -> ItzmaError {
    corrupt(path, FENCED)
}

/// Whether `err` is what [`fenced`] builds.
pub(crate) fn is_fenced(err: &ItzmaError) -> bool {
    matches!(err, ItzmaError::CorruptWal { reason, .. } if reason == FENCED)
}

/// A keyed object and the `body_len` in its header, which a later attempt
/// compares to tell a retry from a reused key.
#[derive(Debug)]
pub(crate) struct Keyed {
    pub(crate) bytes: Bytes,
    pub(crate) body_len: u32,
}

/// What a header read of a keyed object yields.
#[derive(Debug)]
pub(crate) struct Peek {
    pub(crate) key: String,
    pub(crate) body_len: u32,
}

/// Check an idempotency key and return its length as the header stores it.
///
/// # Errors
///
/// [`ItzmaError::InvalidKey`] unless the key is 1 to [`MAX_KEY_LEN`] bytes.
pub(crate) fn validate_key(key: &str) -> Result<u8> {
    if key.is_empty() || key.len() > MAX_KEY_LEN {
        return Err(ItzmaError::InvalidKey("must be 1 to 128 bytes"));
    }
    u8::try_from(key.len()).map_err(|_| ItzmaError::InvalidKey("must be 1 to 128 bytes"))
}

/// Encode a batch as a version 2 object.
///
/// The exact body size is worked out before the vector planes are allocated,
/// so a batch over `max_body` builds nothing. The body is compressed with
/// ZSTD and the frame is kept only when it is strictly smaller.
///
/// # Errors
///
/// - [`ItzmaError::DimensionMismatch`] if the upserts do not all have vectors
///   of the same length.
/// - [`ItzmaError::BatchTooLarge`] if the body would exceed `max_body`.
/// - [`ItzmaError::Serde`] if an attribute cannot be serialized.
/// - [`ItzmaError::CorruptWal`] if a length does not fit in a u32 or ZSTD
///   fails (neither can happen under the 64 MiB writer limit).
pub(crate) fn encode(ops: &[WalOp], max_body: usize) -> Result<Bytes> {
    Ok(encode_with(ops, max_body, None)?.bytes)
}

/// Encode a batch as a version 3 object carrying `key`. Same body, same size
/// rules and same errors as [`encode`], plus [`ItzmaError::InvalidKey`].
pub(crate) fn encode_keyed(ops: &[WalOp], max_body: usize, key: &str) -> Result<Keyed> {
    validate_key(key)?;
    encode_with(ops, max_body, Some(key))
}

fn encode_with(ops: &[WalOp], max_body: usize, key: Option<&str>) -> Result<Keyed> {
    let (upserts, dim) = shape(ops)?;
    let attrs = attribute_json(ops)?;
    let size = body_size(ops, upserts, dim, &attrs);
    if size > max_body as u64 {
        return Err(ItzmaError::BatchTooLarge {
            bytes: size,
            max: max_body as u64,
        });
    }

    let body = build_body(ops, upserts, dim, &attrs, size)?;
    frame(body, key)
}

/// Compress `body` when that makes it smaller, and put the header (and the
/// key, for a keyed object) in front: version 3 with a key, else 2.
fn frame(body: Vec<u8>, key: Option<&str>) -> Result<Keyed> {
    let body_len = len32(body.len())?;
    let compressed = zstd::bulk::compress(&body, ZSTD_LEVEL)
        .map_err(|e| encoding_error(format!("ZSTD failed: {e}")))?;
    let (codec, payload) = if compressed.len() < body.len() {
        (CODEC_ZSTD, compressed)
    } else {
        (CODEC_RAW, body)
    };

    let version = if key.is_some() { V3 } else { V2 };
    let key_len = key.map_or(0, |k| 1 + k.len());
    let mut out = Vec::with_capacity(HEADER_LEN + key_len + payload.len());
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&version.to_le_bytes());
    out.push(codec);
    out.extend_from_slice(&body_len.to_le_bytes());
    if let Some(key) = key {
        out.push(validate_key(key)?);
        out.extend_from_slice(key.as_bytes());
    }
    out.extend_from_slice(&payload);
    Ok(Keyed {
        bytes: Bytes::from(out),
        body_len,
    })
}

/// Encode a columnar batch as the version 2 object that [`encode`]
/// writes for the same documents, byte for byte: every row an upsert,
/// attributes written as JSON straight from the columns. Same size rule and
/// errors as [`encode`] (the batch's shape was checked when it was built).
pub(crate) fn encode_columns(cols: &DocColumns, max_body: usize) -> Result<Bytes> {
    Ok(encode_columns_with(cols, max_body, None)?.bytes)
}

/// [`encode_columns`] as a version 3 object carrying `key`, as
/// [`encode_keyed`] writes it.
pub(crate) fn encode_columns_keyed(cols: &DocColumns, max_body: usize, key: &str) -> Result<Keyed> {
    validate_key(key)?;
    encode_columns_with(cols, max_body, Some(key))
}

/// The exact body length [`encode_columns`] would write for `cols`.
// The tests check it against `body_size`; the stream's chunker sums
// `row_body_bytes` instead, row by row.
#[cfg(test)]
pub(crate) fn columns_body_size(cols: &DocColumns) -> u64 {
    let mut scratch = Vec::new();
    let attrs: u64 = (0..cols.rows())
        .map(|row| {
            scratch.clear();
            write_row_attrs(cols, row, &mut scratch);
            4 + scratch.len() as u64
        })
        .sum();
    columns_size(cols, attrs)
}

/// What row `row` adds to a body: its kind byte, its id, its vector and its
/// attribute entry. A body is [`COUNTS_LEN`] plus this over its rows, so the
/// stream's chunker can cut a batch at a size without encoding it.
pub(crate) fn row_body_bytes(cols: &DocColumns, row: usize, scratch: &mut Vec<u8>) -> u64 {
    scratch.clear();
    write_row_attrs(cols, row, scratch);
    1 + 4 + cols.id(row).len() as u64 + cols.dim() as u64 * 4 + 4 + scratch.len() as u64
}

/// The body length given the attribute column's total length (length
/// prefixes included).
fn columns_size(cols: &DocColumns, attrs: u64) -> u64 {
    let rows = cols.rows() as u64;
    let ids: u64 = (0..cols.rows())
        .map(|row| 4 + cols.id(row).len() as u64)
        .sum();
    let vectors = (cols.vectors().len() as u64).saturating_mul(4);
    COUNTS_LEN as u64 + rows + ids + vectors + attrs
}

fn encode_columns_with(cols: &DocColumns, max_body: usize, key: Option<&str>) -> Result<Keyed> {
    let rows = cols.rows();
    // Every row's attribute entry, one after the other, and where each ends.
    let mut attrs = Vec::new();
    let mut attr_ends = Vec::with_capacity(rows);
    for row in 0..rows {
        write_row_attrs(cols, row, &mut attrs);
        attr_ends.push(attrs.len());
    }
    let size = columns_size(cols, 4 * rows as u64 + attrs.len() as u64);
    if size > max_body as u64 {
        return Err(ItzmaError::BatchTooLarge {
            bytes: size,
            max: max_body as u64,
        });
    }

    let mut body = Vec::with_capacity(usize::try_from(size).unwrap_or(0));
    // A batch of no upserts has no vectors and `encode` writes dim 0.
    let dim = if rows == 0 { 0 } else { cols.dim() };
    body.extend_from_slice(&len32(rows)?.to_le_bytes());
    body.extend_from_slice(&len32(rows)?.to_le_bytes());
    body.extend_from_slice(&len32(dim)?.to_le_bytes());
    body.resize(body.len() + rows, KIND_UPSERT);
    for row in 0..rows {
        put_bytes(&mut body, cols.id(row).as_bytes())?;
    }

    // The vectors are row-major already, which is the order `build_body`
    // walks them in.
    let values = cols.vectors();
    let plane_len = values.len();
    let start = body.len();
    body.resize(start + plane_len * 4, 0);
    for (i, x) in values.iter().enumerate() {
        for (k, byte) in x.to_le_bytes().into_iter().enumerate() {
            body[start + k * plane_len + i] = byte;
        }
    }

    let mut from = 0;
    for end in attr_ends {
        put_bytes(&mut body, &attrs[from..end])?;
        from = end;
    }
    frame(body, key)
}

/// Row `row`'s attributes as the JSON `serde_json::to_vec` writes for the
/// same object: nothing when the batch has no attribute columns, otherwise
/// every column in name order (the batch keeps them sorted, as
/// `serde_json`'s map does), `null` for a row that is not valid.
fn write_row_attrs(cols: &DocColumns, row: usize, out: &mut Vec<u8>) {
    let columns = cols.attrs();
    if columns.is_empty() {
        return;
    }
    out.push(b'{');
    for (at, (name, column)) in columns.iter().enumerate() {
        if at > 0 {
            out.push(b',');
        }
        json_str(out, name);
        out.push(b':');
        write_value(column, row, out);
    }
    out.push(b'}');
}

fn write_value(column: &AttrColumn, row: usize, out: &mut Vec<u8>) {
    if column.is_null(row) {
        out.extend_from_slice(b"null");
        return;
    }
    match column {
        AttrColumn::Null { .. } => out.extend_from_slice(b"null"),
        AttrColumn::Bool { values, .. } => {
            out.extend_from_slice(if values[row] { b"true" } else { b"false" });
        }
        AttrColumn::I64 { values, .. } => json_number(out, &values[row]),
        AttrColumn::F64 { values, .. } => json_number(out, &values[row]),
        AttrColumn::Text { bytes, offsets, .. } => {
            json_str(out, AttrColumn::text(bytes, offsets, row));
        }
        AttrColumn::I64Set {
            offsets,
            items_valid,
            values,
            ..
        } => json_array(
            out,
            AttrColumn::items(offsets, row),
            items_valid,
            |out, i| {
                json_number(out, &values[i]);
            },
        ),
        AttrColumn::F64Set {
            offsets,
            items_valid,
            values,
            ..
        } => json_array(
            out,
            AttrColumn::items(offsets, row),
            items_valid,
            |out, i| {
                json_number(out, &values[i]);
            },
        ),
        AttrColumn::TextSet {
            offsets,
            items_valid,
            bytes,
            byte_offsets,
            ..
        } => json_array(
            out,
            AttrColumn::items(offsets, row),
            items_valid,
            |out, i| {
                json_str(out, AttrColumn::text(bytes, byte_offsets, i));
            },
        ),
    }
}

fn json_array(
    out: &mut Vec<u8>,
    items: std::ops::Range<usize>,
    items_valid: &[bool],
    mut item: impl FnMut(&mut Vec<u8>, usize),
) {
    out.push(b'[');
    for (at, i) in items.enumerate() {
        if at > 0 {
            out.push(b',');
        }
        if items_valid[i] {
            item(out, i);
        } else {
            out.extend_from_slice(b"null");
        }
    }
    out.push(b']');
}

/// A string through `serde_json`'s own escaping, so the bytes match a
/// `Value::String` written by `serde_json::to_vec`. Writing to a `Vec`
/// cannot fail.
fn json_str(out: &mut Vec<u8>, s: &str) {
    let _ = serde_json::to_writer(&mut *out, s);
}

/// A number through `serde_json`: `5` for an integer, `5.0` for a float, as
/// a `Value::Number` built from either writes it. A non-finite float would
/// come out as `null`; the batch's checks refuse one first.
fn json_number<T: serde::Serialize>(out: &mut Vec<u8>, n: &T) {
    let _ = serde_json::to_writer(&mut *out, n);
}

/// The number of upserts and the shared vector length (0 with no upserts).
fn shape(ops: &[WalOp]) -> Result<(usize, usize)> {
    let mut dim = None;
    let mut upserts = 0;
    for op in ops {
        if let WalOp::Upsert(doc) = op {
            upserts += 1;
            let expected = *dim.get_or_insert(doc.vector.len());
            if doc.vector.len() != expected {
                return Err(ItzmaError::DimensionMismatch {
                    expected,
                    got: doc.vector.len(),
                });
            }
        }
    }
    Ok((upserts, dim.unwrap_or(0)))
}

/// Each upsert's attributes as JSON bytes, in upsert order. `None` is empty.
fn attribute_json(ops: &[WalOp]) -> Result<Vec<Vec<u8>>> {
    ops.iter()
        .filter_map(|op| match op {
            WalOp::Upsert(doc) => Some(doc),
            WalOp::Delete(_) => None,
        })
        .map(|doc| match &doc.attrs {
            None => Ok(Vec::new()),
            Some(value) => serde_json::to_vec(value).map_err(|source| ItzmaError::Serde {
                path: "wal entry".into(),
                source,
            }),
        })
        .collect()
}

fn op_id(op: &WalOp) -> &str {
    match op {
        WalOp::Upsert(doc) => &doc.id,
        WalOp::Delete(id) => id,
    }
}

/// The exact length of the body, without building it.
fn body_size(ops: &[WalOp], upserts: usize, dim: usize, attrs: &[Vec<u8>]) -> u64 {
    let ids: u64 = ops.iter().map(|op| 4 + op_id(op).len() as u64).sum();
    let vectors = (upserts as u64)
        .saturating_mul(dim as u64)
        .saturating_mul(4);
    let attrs: u64 = attrs.iter().map(|a| 4 + a.len() as u64).sum();
    COUNTS_LEN as u64 + ops.len() as u64 + ids + vectors + attrs
}

fn build_body(
    ops: &[WalOp],
    upserts: usize,
    dim: usize,
    attrs: &[Vec<u8>],
    size: u64,
) -> Result<Vec<u8>> {
    let mut body = Vec::with_capacity(usize::try_from(size).unwrap_or(0));
    body.extend_from_slice(&len32(ops.len())?.to_le_bytes());
    body.extend_from_slice(&len32(upserts)?.to_le_bytes());
    body.extend_from_slice(&len32(dim)?.to_le_bytes());
    for op in ops {
        body.push(match op {
            WalOp::Upsert(_) => KIND_UPSERT,
            WalOp::Delete(_) => KIND_DELETE,
        });
    }
    for op in ops {
        put_bytes(&mut body, op_id(op).as_bytes())?;
    }

    // Byte-stream-split: plane k holds byte k of every float. Same-position
    // bytes compress far better together than interleaved.
    let plane_len = upserts * dim;
    let start = body.len();
    body.resize(start + plane_len * 4, 0);
    let mut i = 0;
    for op in ops {
        if let WalOp::Upsert(doc) = op {
            for x in &doc.vector {
                for (k, byte) in x.to_le_bytes().into_iter().enumerate() {
                    body[start + k * plane_len + i] = byte;
                }
                i += 1;
            }
        }
    }

    for json in attrs {
        put_bytes(&mut body, json)?;
    }
    Ok(body)
}

fn put_bytes(buf: &mut Vec<u8>, bytes: &[u8]) -> Result<()> {
    buf.extend_from_slice(&len32(bytes.len())?.to_le_bytes());
    buf.extend_from_slice(bytes);
    Ok(())
}

fn len32(n: usize) -> Result<u32> {
    u32::try_from(n).map_err(|_| encoding_error("a length does not fit in a u32"))
}

fn encoding_error(reason: impl Into<String>) -> ItzmaError {
    ItzmaError::CorruptWal {
        path: "(encoding)".into(),
        reason: reason.into(),
    }
}

fn corrupt(path: &str, reason: impl Into<String>) -> ItzmaError {
    ItzmaError::CorruptWal {
        path: path.to_string(),
        reason: reason.into(),
    }
}

/// The fixed fields at the start of a binary object.
struct Header {
    version: u32,
    codec: u8,
    body_len: u32,
    key: Option<String>,
}

/// Read the header and leave the reader at the start of the payload.
fn read_header(head: &mut Reader<'_>) -> Result<Header> {
    if head.take(MAGIC.len())? != MAGIC {
        return Err(head.corrupt("bad magic"));
    }
    let version = head.u32()?;
    if !matches!(version, V2 | V3) {
        return Err(ItzmaError::UnsupportedFormat { version });
    }
    let codec = head.u8()?;
    let body_len = head.u32()?;
    let key = if version == V3 {
        let len = usize::from(head.u8()?);
        if len == 0 || len > MAX_KEY_LEN {
            return Err(head.corrupt("key length is outside 1 to 128"));
        }
        let raw = head.take(len)?;
        let key = std::str::from_utf8(raw)
            .map_err(|_| head.corrupt("key is not valid UTF-8"))?
            .to_owned();
        Some(key)
    } else {
        None
    };
    Ok(Header {
        version,
        codec,
        body_len,
        key,
    })
}

/// Read the idempotency key from the start of an object, without decoding the
/// body. `prefix` needs to be at least the header, which the first
/// [`HEADER_PEEK_LEN`] bytes always are. Returns `None` for version 2, which
/// carries no key, and for bytes that are not a WAL object.
///
/// # Errors
///
/// - [`ItzmaError::UnsupportedFormat`] if the header names a version this
///   build does not read.
/// - [`ItzmaError::CorruptWal`] if the header is cut short or damaged.
pub(crate) fn peek_key(path: &str, prefix: &[u8]) -> Result<Option<Peek>> {
    if !is_binary(prefix) {
        return Ok(None);
    }
    let header = read_header(&mut Reader::new(path, prefix))?;
    Ok(header.key.map(|key| Peek {
        key,
        body_len: header.body_len,
    }))
}

/// The `body_len` a binary object's header declares, read from its first
/// [`HEADER_LEN`] bytes, so a reader can decide whether it may afford the
/// object before it has the rest. `None` for bytes without the WAL magic,
/// which [`decode`] then refuses. Does not check the key or the payload.
///
/// # Errors
///
/// - [`ItzmaError::UnsupportedFormat`] for a version other than 2 or 3.
/// - [`ItzmaError::CorruptWal`] if the prefix is cut short.
pub(crate) fn declared_body_len(path: &str, prefix: &[u8]) -> Result<Option<u64>> {
    if !is_binary(prefix) {
        return Ok(None);
    }
    let mut head = Reader::new(path, prefix);
    head.take(MAGIC.len())?;
    let version = head.u32()?;
    if !matches!(version, V2 | V3) {
        return Err(ItzmaError::UnsupportedFormat { version });
    }
    head.u8()?;
    Ok(Some(u64::from(head.u32()?)))
}

/// Decode a version 2 or 3 object. `path` only labels errors.
///
/// # Errors
///
/// - [`ItzmaError::UnsupportedFormat`] if the header names a version other
///   than 2 or 3.
/// - [`ItzmaError::CorruptWal`] for anything else wrong with the bytes.
pub(crate) fn decode(path: &str, bytes: &[u8]) -> Result<WalEntry> {
    let mut head = Reader::new(path, bytes);
    let Header {
        version,
        codec,
        body_len,
        key,
    } = read_header(&mut head)?;
    let body_len = usize::try_from(body_len)
        .map_err(|_| corrupt(path, "body_len does not fit in memory on this target"))?;
    let payload = head.rest();

    let body = match codec {
        CODEC_RAW => {
            if payload.len() != body_len {
                return Err(corrupt(path, "payload length differs from body_len"));
            }
            Cow::Borrowed(payload)
        }
        CODEC_ZSTD => Cow::Owned(inflate(path, payload, body_len)?),
        other => return Err(corrupt(path, format!("unknown codec {other}"))),
    };
    parse_body(path, &body, version, key)
}

/// Decompress a frame that must hold exactly `body_len` bytes. Nothing is
/// allocated from a claimed length: the frame's recorded size must match the
/// header first, and the output is read in growing chunks and stopped one byte
/// past `body_len`.
fn inflate(path: &str, frame: &[u8], body_len: usize) -> Result<Vec<u8>> {
    match zstd::zstd_safe::get_frame_content_size(frame) {
        Ok(Some(size)) if size == body_len as u64 => {}
        Ok(Some(size)) => {
            return Err(corrupt(
                path,
                format!("frame holds {size} bytes, header says {body_len}"),
            ));
        }
        Ok(None) => return Err(corrupt(path, "frame does not record its size")),
        Err(_) => return Err(corrupt(path, "payload is not a ZSTD frame")),
    }
    let mut body = Vec::new();
    zstd::stream::read::Decoder::new(frame)
        .and_then(|decoder| decoder.take(body_len as u64 + 1).read_to_end(&mut body))
        .map_err(|e| corrupt(path, format!("ZSTD failed: {e}")))?;
    if body.len() != body_len {
        return Err(corrupt(path, "decompressed length differs from body_len"));
    }
    Ok(body)
}

fn parse_body(path: &str, body: &[u8], version: u32, key: Option<String>) -> Result<WalEntry> {
    let mut r = Reader::new(path, body);
    let op_count = r.count()?;
    let upsert_count = r.count()?;
    let dim = r.count()?;
    if upsert_count > op_count {
        return Err(r.corrupt("upsert_count exceeds op_count"));
    }
    if upsert_count == 0 && dim != 0 {
        return Err(r.corrupt("dim is set but there are no upserts"));
    }
    // Every op needs a kind byte and an id length, every upsert its vector
    // and an attribute length. Refusing counts the rest of the body cannot
    // hold stops a forged header from forcing a large allocation. The sums
    // saturate so they cannot wrap.
    let needed = (op_count as u64).saturating_mul(5).saturating_add(
        (upsert_count as u64).saturating_mul((dim as u64).saturating_mul(4).saturating_add(4)),
    );
    if needed > r.remaining() as u64 {
        return Err(r.corrupt("counts exceed the body size"));
    }

    let kinds = r.take(op_count)?;
    if let Some(bad) = kinds.iter().find(|k| **k > KIND_DELETE) {
        return Err(r.corrupt(format!("unknown op kind {bad}")));
    }
    if kinds.iter().filter(|k| **k == KIND_UPSERT).count() != upsert_count {
        return Err(r.corrupt("kinds disagree with upsert_count"));
    }
    let mut ids = Vec::with_capacity(op_count);
    for _ in 0..op_count {
        ids.push(r.string("id")?);
    }

    // The check above bounds upsert_count * dim * 4 by the body length, so
    // these products cannot overflow.
    let plane_len = upsert_count * dim;
    let planes = [
        r.take(plane_len)?,
        r.take(plane_len)?,
        r.take(plane_len)?,
        r.take(plane_len)?,
    ];
    let vector_of = |upsert: usize| -> Vec<f32> {
        (upsert * dim..(upsert + 1) * dim)
            .map(|i| f32::from_le_bytes([planes[0][i], planes[1][i], planes[2][i], planes[3][i]]))
            .collect()
    };

    let mut attrs = Vec::with_capacity(upsert_count);
    for _ in 0..upsert_count {
        let len = r.count()?;
        attrs.push(if len == 0 {
            None
        } else {
            let raw = r.take(len)?;
            let value = serde_json::from_slice(raw)
                .map_err(|e| r.corrupt(format!("attributes are not valid JSON: {e}")))?;
            Some(value)
        });
    }

    r.finish()?;

    let mut attrs = attrs.into_iter();
    let mut upsert = 0;
    let mut ops = Vec::with_capacity(op_count);
    for (kind, id) in kinds.iter().zip(ids) {
        if *kind == KIND_UPSERT {
            ops.push(WalOp::Upsert(Document {
                id,
                vector: vector_of(upsert),
                attrs: attrs.next().flatten(),
            }));
            upsert += 1;
        } else {
            ops.push(WalOp::Delete(id));
        }
    }
    Ok(WalEntry {
        format_version: version,
        key,
        ops,
    })
}

/// A cursor over an object's bytes. Every read is bounds-checked and failures
/// carry the object's key.
struct Reader<'a> {
    path: &'a str,
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(path: &'a str, buf: &'a [u8]) -> Self {
        Reader { path, buf, pos: 0 }
    }

    fn corrupt(&self, reason: impl Into<String>) -> ItzmaError {
        corrupt(self.path, reason)
    }

    fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let out = self
            .buf
            .get(self.pos..)
            .and_then(|rest| rest.get(..n))
            .ok_or_else(|| self.corrupt("object is truncated"))?;
        self.pos += n;
        Ok(out)
    }

    fn rest(&mut self) -> &'a [u8] {
        let out = self.buf.get(self.pos..).unwrap_or_default();
        self.pos = self.buf.len();
        out
    }

    fn u8(&mut self) -> Result<u8> {
        let bytes = self.take(1)?;
        bytes
            .first()
            .copied()
            .ok_or_else(|| self.corrupt("object is truncated"))
    }

    fn u32(&mut self) -> Result<u32> {
        let bytes = self.take(4)?;
        <[u8; 4]>::try_from(bytes)
            .map(u32::from_le_bytes)
            .map_err(|_| self.corrupt("object is truncated"))
    }

    /// A u32 used as a count or length. A u32 always fits in a `usize` on the
    /// targets this crate supports, and `try_from` says so without a cast.
    fn count(&mut self) -> Result<usize> {
        let n = self.u32()?;
        usize::try_from(n).map_err(|_| self.corrupt("count does not fit in memory on this target"))
    }

    fn string(&mut self, what: &str) -> Result<String> {
        let len = self.count()?;
        let raw = self.take(len)?;
        std::str::from_utf8(raw)
            .map(str::to_owned)
            .map_err(|_| self.corrupt(format!("{what} is not valid UTF-8")))
    }

    fn finish(&self) -> Result<()> {
        if self.remaining() == 0 {
            Ok(())
        } else {
            Err(self.corrupt("trailing bytes after the last column"))
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    const BIG: usize = 1 << 30;

    fn up(id: &str, v: &[f32], attrs: Option<serde_json::Value>) -> WalOp {
        WalOp::Upsert(Document {
            id: id.into(),
            vector: v.to_vec(),
            attrs,
        })
    }

    fn del(id: &str) -> WalOp {
        WalOp::Delete(id.into())
    }

    /// A header followed by `payload`, for hand-made objects.
    fn header(codec: u8, body_len: u32, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"ITZW");
        out.extend_from_slice(&2u32.to_le_bytes());
        out.push(codec);
        out.extend_from_slice(&body_len.to_le_bytes());
        out.extend_from_slice(payload);
        out
    }

    /// The body of `[up("a", [1.0, 2.0]), del("b")]`, written by hand from
    /// the layout table at the top of this module.
    fn sample_body() -> Vec<u8> {
        let mut body = Vec::new();
        body.extend_from_slice(&2u32.to_le_bytes()); // op_count
        body.extend_from_slice(&1u32.to_le_bytes()); // upsert_count
        body.extend_from_slice(&2u32.to_le_bytes()); // dim
        body.extend_from_slice(&[0, 1]); // kinds: upsert, delete
        body.extend_from_slice(&1u32.to_le_bytes());
        body.extend_from_slice(b"a");
        body.extend_from_slice(&1u32.to_le_bytes());
        body.extend_from_slice(b"b");
        // 1.0 is 00 00 80 3f and 2.0 is 00 00 00 40 in little endian. Byte
        // plane k holds byte k of each float: 00 00 | 00 00 | 80 00 | 3f 40.
        body.extend_from_slice(&[0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0x3f, 0x40]);
        body.extend_from_slice(&0u32.to_le_bytes()); // no attributes
        body
    }

    fn sample_ops() -> Vec<WalOp> {
        vec![up("a", &[1.0, 2.0], None), del("b")]
    }

    fn roundtrip(ops: &[WalOp]) -> WalEntry {
        let bytes = encode(ops, BIG).unwrap();
        decode("wal/test", &bytes).unwrap()
    }

    #[test]
    fn layout_matches_adr_012() {
        let body = sample_body();
        let expected = header(0, body.len() as u32, &body);
        let got = encode(&sample_ops(), BIG).unwrap();
        assert_eq!(got.as_ref(), expected.as_slice());
        assert!(is_binary(&got));
    }

    #[test]
    fn roundtrip_keeps_order_duplicates_and_attributes() {
        let ops = vec![
            up("a", &[1.0, 2.0], None),
            del("a"),
            up("a", &[3.0, 4.0], Some(json!(null))),
            del("b"),
            up("c", &[5.0, 6.0], Some(json!({"k": [1, 2], "s": "x"}))),
        ];
        let entry = roundtrip(&ops);
        assert_eq!(entry.format_version, 2);
        assert_eq!(entry.ops, ops);
    }

    #[test]
    fn a_plain_batch_is_version_2() {
        let ops = vec![up("a", &[1.0], None), del("b")];
        assert_eq!(roundtrip(&ops).format_version, 2);
    }

    #[test]
    fn versions_4_and_5_are_not_read_here() {
        let ops = vec![up("a", &[1.0], None)];
        for version in [4u32, 5] {
            let mut bytes = encode(&ops, BIG).unwrap().to_vec();
            bytes[4..8].copy_from_slice(&version.to_le_bytes());
            assert!(matches!(
                decode("p", &bytes),
                Err(ItzmaError::UnsupportedFormat { version: v }) if v == version
            ));
            assert!(matches!(
                declared_body_len("p", &bytes),
                Err(ItzmaError::UnsupportedFormat { .. })
            ));
        }
    }

    #[test]
    fn a_delete_only_batch_has_no_dimension() {
        let ops = vec![del("a"), del("b")];
        assert_eq!(roundtrip(&ops).ops, ops);
    }

    #[test]
    fn floats_survive_bit_for_bit() {
        let odd = [
            -0.0,
            f32::MIN_POSITIVE / 2.0,
            f32::MAX,
            f32::NEG_INFINITY,
            f32::from_bits(0x7fc0_0001),
        ];
        let entry = roundtrip(&[up("a", &odd, None)]);
        let WalOp::Upsert(doc) = &entry.ops[0] else {
            panic!("expected an upsert");
        };
        let got: Vec<u32> = doc.vector.iter().map(|x| x.to_bits()).collect();
        let want: Vec<u32> = odd.iter().map(|x| x.to_bits()).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn unicode_ids_roundtrip() {
        let ops = vec![up("naïve-日本-🦀", &[1.0], None), del("é")];
        assert_eq!(roundtrip(&ops).ops, ops);
    }

    fn compressible() -> Vec<WalOp> {
        (0..200)
            .map(|i| {
                up(
                    &format!("doc-{i:04}"),
                    &[0.0; 32],
                    Some(json!({"lang": "en", "year": 2024})),
                )
            })
            .collect()
    }

    #[test]
    fn a_compressible_batch_is_stored_as_a_zstd_frame() {
        let ops = compressible();
        let bytes = encode(&ops, BIG).unwrap();
        assert_eq!(bytes[8], 1, "codec byte");
        // The body is about 34 KB.
        assert!(bytes.len() < 4_000, "got {} bytes", bytes.len());
        assert_eq!(decode("wal/test", &bytes).unwrap().ops, ops);
    }

    #[test]
    fn a_tiny_batch_stays_raw_because_zstd_would_not_shrink_it() {
        let bytes = encode(&sample_ops(), BIG).unwrap();
        assert_eq!(bytes[8], 0, "codec byte");
    }

    #[test]
    fn the_writer_limit_counts_the_exact_body_size() {
        // The hand-written body is 36 bytes.
        assert_eq!(sample_body().len(), 36);
        let err = encode(&sample_ops(), 35).unwrap_err();
        assert!(
            matches!(err, ItzmaError::BatchTooLarge { bytes: 36, max: 35 }),
            "unexpected error: {err:?}"
        );
        assert!(encode(&sample_ops(), 36).is_ok());
    }

    #[test]
    fn vectors_of_different_lengths_are_refused() {
        let ops = vec![up("a", &[1.0, 2.0], None), up("b", &[1.0], None)];
        let err = encode(&ops, BIG).unwrap_err();
        assert!(matches!(
            err,
            ItzmaError::DimensionMismatch {
                expected: 2,
                got: 1
            }
        ));
    }

    #[test]
    fn a_version_1_reader_cannot_parse_a_version_2_object() {
        // Compatibility canary: this is why the rollout rule exists. If a v2
        // object ever parses as v1 JSON, the rule and the ADR need a rethink.
        let bytes = encode(&sample_ops(), BIG).unwrap();
        assert!(serde_json::from_slice::<WalEntry>(&bytes).is_err());
    }

    #[test]
    fn the_object_is_far_smaller_than_json() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let ops: Vec<WalOp> = (0..100)
            .map(|i| {
                let v: Vec<f32> = (0..128).map(|_| next()).collect();
                up(&format!("doc-{i:06}"), &v, Some(json!({"n": i})))
            })
            .collect();
        let v2 = encode(&ops, BIG).unwrap();
        let v1 = serde_json::to_vec(&WalEntry {
            format_version: 1,
            key: None,
            ops,
        })
        .unwrap();
        assert!(
            v2.len() * 2 < v1.len(),
            "v2 {} bytes, v1 {}",
            v2.len(),
            v1.len()
        );
    }

    fn reason(err: ItzmaError) -> String {
        match err {
            ItzmaError::CorruptWal { reason, .. } => reason,
            other => panic!("expected CorruptWal, got {other:?}"),
        }
    }

    fn raw_object(body: &[u8]) -> Vec<u8> {
        header(0, body.len() as u32, body)
    }

    #[test]
    fn every_truncation_is_an_error_and_none_panics() {
        for object in [
            encode(&sample_ops(), BIG).unwrap(),
            encode(&compressible(), BIG).unwrap(),
            keyed(&sample_ops()),
            keyed(&compressible()),
        ] {
            // Shorter than the magic is not a version 2 object at all.
            for len in MAGIC.len()..object.len() {
                assert!(
                    decode("wal/test", &object[..len]).is_err(),
                    "a {len}-byte prefix decoded"
                );
            }
        }
    }

    #[test]
    fn a_flipped_bit_never_panics() {
        for object in [
            encode(&sample_ops(), BIG).unwrap(),
            encode(&compressible(), BIG).unwrap(),
            keyed(&sample_ops()),
            keyed(&compressible()),
        ] {
            for i in 0..object.len() {
                for bit in 0..8 {
                    let mut damaged = object.to_vec();
                    damaged[i] ^= 1 << bit;
                    // Ok or Err are both fine: a flipped bit inside an id may
                    // still be a valid id. The point is that it returns.
                    let _ = decode("wal/test", &damaged);
                    let _ = peek_key("wal/test", &damaged);
                }
            }
        }
    }

    #[test]
    fn an_unknown_version_is_unsupported_format() {
        let mut object = encode(&sample_ops(), BIG).unwrap().to_vec();
        object[4..8].copy_from_slice(&6u32.to_le_bytes());
        let err = decode("wal/test", &object).unwrap_err();
        assert!(matches!(err, ItzmaError::UnsupportedFormat { version: 6 }));
    }

    #[test]
    fn an_unknown_codec_is_corrupt() {
        let object = header(2, 0, &[]);
        assert_eq!(
            reason(decode("wal/test", &object).unwrap_err()),
            "unknown codec 2"
        );
    }

    #[test]
    fn a_raw_payload_must_match_body_len() {
        let body = sample_body();
        let longer = header(0, body.len() as u32 - 1, &body);
        let shorter = header(0, body.len() as u32 + 1, &body);
        for object in [longer, shorter] {
            let why = reason(decode("wal/test", &object).unwrap_err());
            assert!(why.contains("body_len"), "{why}");
        }
    }

    #[test]
    fn a_huge_claimed_body_len_is_refused_without_allocating() {
        let object = header(0, u32::MAX, &[0; 16]);
        assert!(decode("wal/test", &object).is_err());
    }

    #[test]
    fn a_zstd_frame_that_disagrees_with_body_len_is_refused() {
        // A megabyte of zeros is a few dozen bytes of ZSTD. The header claims
        // a small body, then a huge one; neither may be believed.
        let frame = zstd::bulk::compress(&vec![0u8; 1 << 20], 3).unwrap();
        for claimed in [100, u32::MAX] {
            let object = header(1, claimed, &frame);
            let why = reason(decode("wal/test", &object).unwrap_err());
            assert!(why.contains("header says"), "{why}");
        }
    }

    #[test]
    fn a_zstd_frame_without_a_recorded_size_is_refused() {
        let frame = zstd::encode_all(sample_body().as_slice(), 3).unwrap();
        let object = header(1, sample_body().len() as u32, &frame);
        let why = reason(decode("wal/test", &object).unwrap_err());
        assert!(why.contains("does not record its size"), "{why}");
    }

    #[test]
    fn a_payload_that_is_not_a_zstd_frame_is_refused() {
        let object = header(1, 36, &[0xde; 40]);
        assert!(decode("wal/test", &object).is_err());
    }

    /// Counts, kinds and the rest of a body, for forging bad ones.
    fn body(counts: [u32; 3], rest: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        for c in counts {
            out.extend_from_slice(&c.to_le_bytes());
        }
        out.extend_from_slice(rest);
        out
    }

    fn cat(parts: &[&[u8]]) -> Vec<u8> {
        parts.concat()
    }

    fn id(s: &[u8]) -> Vec<u8> {
        let mut out = (s.len() as u32).to_le_bytes().to_vec();
        out.extend_from_slice(s);
        out
    }

    #[test]
    fn a_body_that_contradicts_itself_is_refused() {
        let mut trailing = sample_body();
        trailing.push(0);
        let cases: Vec<(&str, Vec<u8>, &str)> = vec![
            (
                "an op count the body cannot hold",
                body([u32::MAX, 0, 0], &[]),
                "counts exceed the body size",
            ),
            (
                "more upserts than ops",
                body([1, 2, 1], &[0; 32]),
                "upsert_count exceeds op_count",
            ),
            (
                "a dimension with no upserts",
                body([1, 0, 5], &cat(&[&[1], &id(b"a")])),
                "dim is set but there are no upserts",
            ),
            (
                "an unknown op kind",
                body([1, 0, 0], &cat(&[&[2], &id(b"a")])),
                "unknown op kind 2",
            ),
            (
                "kinds that do not add up to upsert_count",
                body([1, 1, 0], &cat(&[&[1], &id(b"a"), &[0; 4]])),
                "kinds disagree with upsert_count",
            ),
            (
                "trailing bytes",
                trailing,
                "trailing bytes after the last column",
            ),
            (
                "an id that is not UTF-8",
                body([1, 0, 0], &cat(&[&[1], &id(&[0xff])])),
                "id is not valid UTF-8",
            ),
            (
                "attributes that are not JSON",
                body(
                    [1, 1, 0],
                    &cat(&[&[0], &id(b"a"), &1u32.to_le_bytes(), b"{"]),
                ),
                "attributes are not valid JSON",
            ),
        ];
        for (name, body, expected) in cases {
            let err = decode("wal/test", &raw_object(&body)).unwrap_err();
            let why = reason(err);
            assert!(why.contains(expected), "{name}: got {why:?}");
        }
    }

    const KEY: &str = "job-7/batch-42";

    fn keyed(ops: &[WalOp]) -> Bytes {
        encode_keyed(ops, BIG, KEY).unwrap().bytes
    }

    /// A version 3 header followed by `payload`, for hand-made objects. The
    /// key length byte is given separately so a test can lie about it.
    fn header_v3(key_len: u8, key: &[u8], body_len: u32, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"ITZW");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.push(0);
        out.extend_from_slice(&body_len.to_le_bytes());
        out.push(key_len);
        out.extend_from_slice(key);
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn a_keyed_batch_is_version_3_and_carries_its_key() {
        let ops = vec![up("a", &[1.0, 2.0], Some(json!({"k": 1}))), del("b")];
        let entry = decode("wal/test", &keyed(&ops)).unwrap();
        assert_eq!(entry.format_version, 3);
        assert_eq!(entry.key.as_deref(), Some(KEY));
        assert_eq!(entry.ops, ops);

        let plain = roundtrip(&ops);
        assert_eq!(plain.format_version, 2);
        assert_eq!(plain.key, None);
    }

    #[test]
    fn a_keyed_object_is_the_version_2_object_with_a_key_in_the_header() {
        let ops = sample_ops();
        let plain = encode(&ops, BIG).unwrap();
        let object = encode_keyed(&ops, BIG, KEY).unwrap();
        let bytes = object.bytes;
        let key_end = 14 + KEY.len();

        assert_eq!(&bytes[..4], b"ITZW");
        assert_eq!(bytes[4..8], 3u32.to_le_bytes());
        assert_eq!(bytes[8..13], plain[8..13], "codec and body_len");
        assert_eq!(usize::from(bytes[13]), KEY.len());
        assert_eq!(&bytes[14..key_end], KEY.as_bytes());
        assert_eq!(&bytes[key_end..], &plain[13..], "the payload is the same");
        assert_eq!(object.body_len.to_le_bytes(), plain[9..13]);
    }

    #[test]
    fn the_key_is_read_from_the_first_142_bytes_alone() {
        let object = encode_keyed(&compressible(), BIG, KEY).unwrap();
        assert!(object.bytes.len() > HEADER_PEEK_LEN);
        let peek = peek_key("wal/test", &object.bytes[..HEADER_PEEK_LEN])
            .unwrap()
            .unwrap();
        assert_eq!(peek.key, KEY);
        assert_eq!(peek.body_len, object.body_len);

        // The longest key still fits in the peek.
        let longest = "k".repeat(MAX_KEY_LEN);
        let object = encode_keyed(&compressible(), BIG, &longest).unwrap();
        let peek = peek_key("wal/test", &object.bytes[..HEADER_PEEK_LEN])
            .unwrap()
            .unwrap();
        assert_eq!(peek.key, longest);
    }

    #[test]
    fn peek_finds_no_key_in_json_and_version_2_objects() {
        let v2 = encode(&sample_ops(), BIG).unwrap();
        assert!(peek_key("wal/test", &v2).unwrap().is_none());
        let v1 = br#"{"format_version": 1, "ops": []}"#;
        assert!(peek_key("wal/test", v1).unwrap().is_none());
    }

    #[test]
    fn peek_refuses_a_header_that_stops_early_or_names_a_newer_version() {
        let bytes = keyed(&sample_ops());
        let header_end = 14 + KEY.len();
        for len in MAGIC.len()..header_end {
            assert!(
                peek_key("wal/test", &bytes[..len]).is_err(),
                "a {len}-byte prefix peeked"
            );
        }
        assert!(peek_key("wal/test", &bytes[..header_end]).is_ok());

        let mut newer = bytes.to_vec();
        newer[4..8].copy_from_slice(&6u32.to_le_bytes());
        let err = peek_key("wal/test", &newer).unwrap_err();
        assert!(matches!(err, ItzmaError::UnsupportedFormat { version: 6 }));
    }

    #[test]
    fn a_key_must_be_1_to_128_bytes() {
        for bad in ["".to_string(), "k".repeat(MAX_KEY_LEN + 1)] {
            let err = encode_keyed(&sample_ops(), BIG, &bad).unwrap_err();
            assert!(matches!(err, ItzmaError::InvalidKey(_)), "{err:?}");
        }
        assert!(encode_keyed(&sample_ops(), BIG, "k").is_ok());
        assert!(encode_keyed(&sample_ops(), BIG, &"k".repeat(MAX_KEY_LEN)).is_ok());
        // The limit counts bytes, not characters.
        let wide = "🦀".repeat(MAX_KEY_LEN / 4 + 1);
        assert!(encode_keyed(&sample_ops(), BIG, &wide).is_err());
    }

    #[test]
    fn a_forged_key_is_refused() {
        let body = sample_body();
        let cases: Vec<(&str, Vec<u8>, &str)> = vec![
            (
                "a zero-length key",
                header_v3(0, b"", body.len() as u32, &body),
                "key length",
            ),
            (
                "a key over the limit",
                header_v3(129, &[b'k'; 129], body.len() as u32, &body),
                "key length",
            ),
            (
                "a key that is not UTF-8",
                header_v3(2, &[0xff, 0xfe], body.len() as u32, &body),
                "key is not valid UTF-8",
            ),
            (
                "a key that runs past the end",
                header_v3(9, b"ab", 0, &[]),
                "truncated",
            ),
        ];
        for (name, object, expected) in cases {
            let why = reason(decode("wal/test", &object).unwrap_err());
            assert!(why.contains(expected), "{name}: got {why:?}");
        }
    }

    #[test]
    fn a_keyed_batch_counts_toward_the_writer_limit_by_its_body_only() {
        let err = encode_keyed(&sample_ops(), 35, KEY).unwrap_err();
        assert!(matches!(
            err,
            ItzmaError::BatchTooLarge { bytes: 36, max: 35 }
        ));
        assert!(encode_keyed(&sample_ops(), 36, KEY).is_ok());
    }

    #[test]
    fn errors_name_the_object_key() {
        let err = decode("ns/wal/00000000000000000007.json", &header(2, 0, &[])).unwrap_err();
        match err {
            ItzmaError::CorruptWal { path, .. } => {
                assert_eq!(path, "ns/wal/00000000000000000007.json");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn the_declared_body_len_is_read_from_the_first_13_bytes() {
        let v2 = encode(&compressible(), BIG).unwrap();
        let want = u32::from_le_bytes(v2[9..13].try_into().unwrap());
        assert_eq!(
            declared_body_len("wal/test", &v2[..HEADER_LEN]).unwrap(),
            Some(u64::from(want))
        );

        let v3 = encode_keyed(&compressible(), BIG, KEY).unwrap();
        assert_eq!(
            declared_body_len("wal/test", &v3.bytes[..HEADER_LEN]).unwrap(),
            Some(u64::from(v3.body_len))
        );
    }

    #[test]
    fn a_json_object_declares_nothing() {
        let v1 = br#"{"format_version": 1, "ops": []}"#;
        assert_eq!(declared_body_len("wal/test", v1).unwrap(), None);
    }

    #[test]
    fn a_binary_prefix_cut_short_is_corrupt() {
        let v2 = encode(&sample_ops(), BIG).unwrap();
        for len in 4..HEADER_LEN {
            let err = declared_body_len("wal/test", &v2[..len]).unwrap_err();
            assert!(
                matches!(err, ItzmaError::CorruptWal { .. }),
                "{len}: {err:?}"
            );
        }
    }

    #[test]
    fn an_unknown_version_is_unsupported_when_the_length_is_asked_for() {
        let mut bytes = header(0, 5, b"12345");
        bytes[4..8].copy_from_slice(&9u32.to_le_bytes());
        let err = declared_body_len("wal/test", &bytes).unwrap_err();
        assert!(
            matches!(err, ItzmaError::UnsupportedFormat { version: 9 }),
            "{err:?}"
        );
    }

    #[test]
    fn a_fence_is_the_magic_and_version_1() {
        assert_eq!(fence().as_ref(), b"ITZF\x01\x00\x00\x00");
        assert_eq!(fence().len(), FENCE_LEN);
    }

    #[test]
    fn only_the_fence_magic_is_a_fence() {
        assert!(is_fence(&fence()));
        assert!(!is_fence(&encode(&sample_ops(), BIG).unwrap()));
        assert!(!is_fence(br#"{"format_version": 1, "ops": []}"#));
        assert!(!is_fence(b""));
        assert!(!is_fence(b"ITZ"));
    }

    #[test]
    fn a_fence_and_no_other_bytes_check_out() {
        assert!(check_fence("wal/test", &fence()).is_ok());

        let mut longer = fence().to_vec();
        longer.push(0);
        assert!(matches!(
            check_fence("wal/test", &longer),
            Err(ItzmaError::CorruptWal { .. })
        ));

        let mut newer = fence().to_vec();
        newer[4..8].copy_from_slice(&2u32.to_le_bytes());
        assert!(matches!(
            check_fence("wal/test", &newer),
            Err(ItzmaError::UnsupportedFormat { version: 2 })
        ));
    }

    #[test]
    fn every_truncation_and_bit_flip_of_a_fence_is_an_error() {
        let fence = fence();
        for len in 0..fence.len() {
            assert!(
                check_fence("wal/test", &fence[..len]).is_err(),
                "a {len}-byte prefix passed"
            );
        }
        for i in 0..fence.len() {
            for bit in 0..8 {
                let mut damaged = fence.to_vec();
                damaged[i] ^= 1 << bit;
                assert!(
                    check_fence("wal/test", &damaged).is_err(),
                    "byte {i} bit {bit} passed"
                );
            }
        }
    }

    #[test]
    fn the_fenced_error_is_recognised_and_other_errors_are_not() {
        let err = fenced("wal/test");
        assert!(is_fenced(&err));
        assert!(matches!(&err, ItzmaError::CorruptWal { path, .. } if path == "wal/test"));
        assert!(!is_fenced(&corrupt("wal/test", "bad magic")));
        assert!(!is_fenced(&ItzmaError::EmptyBatch));
    }
}

/// `encode_columns` against `encode`: the same batch, given as
/// columns and as documents built here from the same raw cells, must give
/// the same bytes.
#[cfg(test)]
mod column_tests {
    use serde_json::{Map, Number, Value};

    use super::*;
    use crate::columns::{AttrColumn, DocColumns};

    const BIG: usize = 1 << 30;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }

        fn text(&mut self, min: u64, max: u64) -> String {
            const CHARS: [&str; 10] = ["a", "Z", "\"", "\\", "é", "\n", "😀", " ", "0", "\u{1}"];
            let len = min + self.below(max - min + 1);
            (0..len)
                .map(|_| CHARS[self.below(CHARS.len() as u64) as usize])
                .collect()
        }

        fn int(&mut self) -> i64 {
            match self.below(4) {
                0 => i64::MIN,
                1 => i64::MAX,
                _ => self.below(2_000) as i64 - 1_000,
            }
        }

        fn float(&mut self) -> f64 {
            if self.below(2) == 0 {
                self.below(100) as f64
            } else {
                (self.below(1_000_000) as f64 - 500_000.0) / 7.0
            }
        }
    }

    #[derive(Clone, Copy)]
    enum Kind {
        Null,
        Bool,
        I64,
        F64,
        Text,
        I64Set,
        F64Set,
        TextSet,
    }

    const KINDS: [Kind; 8] = [
        Kind::Null,
        Kind::Bool,
        Kind::I64,
        Kind::F64,
        Kind::Text,
        Kind::I64Set,
        Kind::F64Set,
        Kind::TextSet,
    ];

    #[derive(Clone)]
    enum Cell {
        Null,
        Bool(bool),
        I64(i64),
        F64(f64),
        Text(String),
        I64Set(Vec<Option<i64>>),
        F64Set(Vec<Option<f64>>),
        TextSet(Vec<Option<String>>),
    }

    fn cell(rng: &mut Rng, kind: Kind) -> Cell {
        let items = rng.below(5) as usize;
        let null_item = |rng: &mut Rng| rng.below(4) == 0;
        match kind {
            Kind::Null => Cell::Null,
            Kind::Bool => Cell::Bool(rng.below(2) == 0),
            Kind::I64 => Cell::I64(rng.int()),
            Kind::F64 => Cell::F64(rng.float()),
            Kind::Text => Cell::Text(rng.text(0, 8)),
            Kind::I64Set => Cell::I64Set(
                (0..items)
                    .map(|_| (!null_item(rng)).then(|| rng.int()))
                    .collect(),
            ),
            Kind::F64Set => Cell::F64Set(
                (0..items)
                    .map(|_| (!null_item(rng)).then(|| rng.float()))
                    .collect(),
            ),
            Kind::TextSet => Cell::TextSet(
                (0..items)
                    .map(|_| (!null_item(rng)).then(|| rng.text(0, 5)))
                    .collect(),
            ),
        }
    }

    fn json(cell: &Cell) -> Value {
        let float = |f: f64| Value::Number(Number::from_f64(f).unwrap());
        match cell {
            Cell::Null => Value::Null,
            Cell::Bool(b) => Value::Bool(*b),
            Cell::I64(i) => Value::from(*i),
            Cell::F64(f) => float(*f),
            Cell::Text(s) => Value::String(s.clone()),
            Cell::I64Set(items) => Value::Array(
                items
                    .iter()
                    .map(|i| i.map_or(Value::Null, Value::from))
                    .collect(),
            ),
            Cell::F64Set(items) => {
                Value::Array(items.iter().map(|f| f.map_or(Value::Null, float)).collect())
            }
            Cell::TextSet(items) => Value::Array(
                items
                    .iter()
                    .map(|s| s.clone().map_or(Value::Null, Value::String))
                    .collect(),
            ),
        }
    }

    /// The column for `cells`, a `Null` cell being a row that is not valid.
    fn column(kind: Kind, cells: &[Cell]) -> AttrColumn {
        let valid: Vec<bool> = cells.iter().map(|c| !matches!(c, Cell::Null)).collect();
        let mut offsets = vec![0u32];
        let mut items_valid = Vec::new();
        let mut bytes = Vec::new();
        let mut byte_offsets = vec![0u32];
        let mut ints = Vec::new();
        let mut floats = Vec::new();
        let mut bools = Vec::new();
        for c in cells {
            match c {
                // A null row still takes a slot in a scalar column's values;
                // a set or text column gives it no items or bytes.
                Cell::Null => match kind {
                    Kind::Bool => bools.push(false),
                    Kind::I64 => ints.push(0),
                    Kind::F64 => floats.push(0.0),
                    _ => {}
                },
                Cell::Bool(b) => bools.push(*b),
                Cell::I64(i) => ints.push(*i),
                Cell::F64(f) => floats.push(*f),
                Cell::Text(s) => bytes.extend_from_slice(s.as_bytes()),
                Cell::I64Set(items) => {
                    for i in items {
                        items_valid.push(i.is_some());
                        ints.push(i.unwrap_or(0));
                    }
                }
                Cell::F64Set(items) => {
                    for f in items {
                        items_valid.push(f.is_some());
                        floats.push(f.unwrap_or(0.0));
                    }
                }
                Cell::TextSet(items) => {
                    for s in items {
                        items_valid.push(s.is_some());
                        bytes.extend_from_slice(s.as_deref().unwrap_or("").as_bytes());
                        byte_offsets.push(bytes.len() as u32);
                    }
                }
            }
            offsets.push(match kind {
                Kind::Text => bytes.len() as u32,
                _ => items_valid.len() as u32,
            });
        }
        match kind {
            Kind::Null => AttrColumn::Null { rows: cells.len() },
            Kind::Bool => AttrColumn::Bool {
                valid,
                values: bools,
            },
            Kind::I64 => AttrColumn::I64 {
                valid,
                values: ints,
            },
            Kind::F64 => AttrColumn::F64 {
                valid,
                values: floats,
            },
            Kind::Text => AttrColumn::Text {
                valid,
                bytes,
                offsets,
            },
            Kind::I64Set => AttrColumn::I64Set {
                valid,
                offsets,
                items_valid,
                values: ints,
            },
            Kind::F64Set => AttrColumn::F64Set {
                valid,
                offsets,
                items_valid,
                values: floats,
            },
            Kind::TextSet => AttrColumn::TextSet {
                valid,
                offsets,
                items_valid,
                bytes,
                byte_offsets,
            },
        }
    }

    /// One random batch, as columns and as the equivalent upserts.
    fn case(seed: u64) -> (DocColumns, Vec<WalOp>) {
        let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
        let rows = [1, 2, 17, 300][rng.below(4) as usize];
        let dim = [1, 3, 8][rng.below(3) as usize];
        let ids: Vec<String> = (0..rows).map(|_| rng.text(1, 40)).collect();
        let vectors: Vec<f32> = (0..rows * dim)
            .map(|_| (rng.below(20_000) as f32 - 10_000.0) / 3.0)
            .collect();

        let mut names = Vec::new();
        let mut columns: Vec<(Kind, Vec<Cell>)> = Vec::new();
        for _ in 0..rng.below(7) {
            let name = rng.text(1, 6);
            if names.contains(&name) {
                continue;
            }
            let kind = KINDS[rng.below(KINDS.len() as u64) as usize];
            // All valid, all null, or mixed.
            let mode = rng.below(3);
            let cells = (0..rows)
                .map(|_| {
                    let null = mode == 1 || (mode == 2 && rng.below(3) == 0);
                    if null {
                        Cell::Null
                    } else {
                        cell(&mut rng, kind)
                    }
                })
                .collect();
            names.push(name);
            columns.push((kind, cells));
        }
        let mut id_bytes = Vec::new();
        let mut id_offsets = vec![0u32];
        for id in &ids {
            id_bytes.extend_from_slice(id.as_bytes());
            id_offsets.push(id_bytes.len() as u32);
        }
        let mut cols = DocColumns::new(id_bytes, id_offsets, vectors.clone(), dim).unwrap();
        for (name, (kind, cells)) in names.iter().zip(&columns) {
            cols = cols.with_attr(name.clone(), column(*kind, cells)).unwrap();
        }
        let ops = (0..rows)
            .map(|row| {
                WalOp::Upsert(Document {
                    id: ids[row].clone(),
                    vector: vectors[row * dim..(row + 1) * dim].to_vec(),
                    attrs: (!columns.is_empty()).then(|| {
                        let mut map = Map::new();
                        for (name, (_, cells)) in names.iter().zip(&columns) {
                            map.insert(name.clone(), json(&cells[row]));
                        }
                        Value::Object(map)
                    }),
                })
            })
            .collect();
        (cols, ops)
    }

    #[test]
    fn encode_columns_equals_encode() {
        for seed in 0..500 {
            let (cols, ops) = case(seed);
            assert_eq!(
                encode_columns(&cols, BIG).unwrap(),
                encode(&ops, BIG).unwrap(),
                "seed {seed}"
            );
            assert_eq!(
                encode_columns_keyed(&cols, BIG, "k").unwrap().bytes,
                encode_keyed(&ops, BIG, "k").unwrap().bytes,
                "seed {seed}, keyed"
            );
        }
    }

    #[test]
    fn the_size_limit_is_at_the_same_byte() {
        for seed in 0..50 {
            let (cols, ops) = case(seed);
            let size = columns_body_size(&cols);
            let (upserts, dim) = shape(&ops).unwrap();
            let attrs = attribute_json(&ops).unwrap();
            assert_eq!(size, body_size(&ops, upserts, dim, &attrs), "seed {seed}");
            let at = usize::try_from(size).unwrap();
            assert!(matches!(
                encode_columns(&cols, at - 1),
                Err(ItzmaError::BatchTooLarge { bytes, max }) if bytes == size && max == size - 1
            ));
            assert!(encode_columns(&cols, at).is_ok());
        }
    }
}
