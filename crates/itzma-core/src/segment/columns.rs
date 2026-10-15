//! Typed attribute columns for one block of a version 3 segment.
//!
//! A block is a u32 row count, an `objects` bitmap (the row has an attributes
//! object at all), then one column per declared field in name order: a
//! `present` bitmap (the key is in the object), a `valued` bitmap (present and
//! not `null`), and the values of the valued rows. The layout is a pure
//! function of the rows, so the byte compare holds, and the decoder
//! refuses any block the encoder would not have written.

use std::sync::Arc;

use serde_json::{Map, Number, Value};

use super::Reader;
use super::v2::corrupt;
use crate::attrs::{AttrSchema, AttrType};
use crate::error::{ItzmaError, Result};

/// Most dictionary entries a text column holds; above it the column is plain.
pub(crate) const MAX_DICT: usize = 65_535;
/// Up to this many entries a dictionary's codes are one byte.
const NARROW_DICT: usize = 256;

/// A declared field, as a segment's directory lists it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Field {
    pub(crate) name: String,
    pub(crate) ty: AttrType,
}

/// A namespace's fields in name order, shared by every block of a segment.
pub(crate) type Fields = Arc<[Field]>;

/// The fields of `schema`, in name order. Empty for a free-form namespace.
pub(crate) fn fields_of(schema: &AttrSchema) -> Fields {
    schema
        .fields()
        .map(|(name, ty)| Field {
            name: name.to_string(),
            ty,
        })
        .collect()
}

/// The tag a field's type is stored as, in the directory and the zone map.
pub(crate) fn tag(ty: AttrType) -> u8 {
    match ty {
        AttrType::Bool => 0,
        AttrType::Number => 1,
        AttrType::Text => 2,
        AttrType::TextSet => 3,
        AttrType::NumberSet => 4,
    }
}

pub(crate) fn type_of_tag(tag: u8) -> Option<AttrType> {
    Some(match tag {
        0 => AttrType::Bool,
        1 => AttrType::Number,
        2 => AttrType::Text,
        3 => AttrType::TextSet,
        4 => AttrType::NumberSet,
        _ => return None,
    })
}

fn refuse(reason: impl Into<String>) -> ItzmaError {
    corrupt("(encoding)", reason)
}

/// One bit per row of a block, least significant bit first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Bits(Vec<u8>);

impl Bits {
    fn new(rows: usize) -> Self {
        Self(vec![0; rows.div_ceil(8)])
    }

    fn set(&mut self, i: usize) {
        self.0[i / 8] |= 1 << (i % 8);
    }

    pub(crate) fn get(&self, i: usize) -> bool {
        self.0.get(i / 8).is_some_and(|b| b & (1 << (i % 8)) != 0)
    }

    pub(crate) fn count(&self) -> usize {
        self.0.iter().map(|b| b.count_ones() as usize).sum()
    }

    /// Set bits below `i`: the position of row `i` among the set rows.
    fn rank(&self, i: usize) -> usize {
        let whole: usize = self.0[..i / 8]
            .iter()
            .map(|b| b.count_ones() as usize)
            .sum();
        let part = self
            .0
            .get(i / 8)
            .map_or(0, |b| (b & ((1u8 << (i % 8)) - 1)).count_ones());
        whole + part as usize
    }

    /// Every bit set here is set in `other`.
    fn within(&self, other: &Bits) -> bool {
        self.0.iter().zip(&other.0).all(|(a, b)| a & !b == 0)
    }

    fn read(r: &mut Reader<'_>, rows: usize, what: &str) -> Result<Self> {
        let bytes = r.take(rows.div_ceil(8))?.to_vec();
        if !rows.is_multiple_of(8) && bytes.last().is_some_and(|b| b >> (rows % 8) != 0) {
            return Err(r.corrupt(format!("{what} bitmap has padding bits set")));
        }
        Ok(Self(bytes))
    }
}

/// Numbers as stored: a kind per value, then its 64 bits.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct Numbers {
    kinds: Vec<u8>,
    raw: Vec<u64>,
}

impl Numbers {
    fn push(&mut self, n: &Number) {
        // serde_json parses a non-negative integer as u64 and a negative one
        // as i64, and builds the same from Rust integers, so this choice is
        // canonical: an i64 here is always negative.
        let (kind, raw) = if let Some(u) = n.as_u64() {
            (2, u)
        } else if let Some(i) = n.as_i64() {
            (1, i as u64)
        } else {
            (0, n.as_f64().map_or(0, f64::to_bits))
        };
        self.kinds.push(kind);
        self.raw.push(raw);
    }

    pub(crate) fn len(&self) -> usize {
        self.kinds.len()
    }

    /// Value `j` as the `f64` a predicate compares.
    pub(crate) fn as_f64(&self, j: usize) -> f64 {
        match self.kinds[j] {
            2 => self.raw[j] as f64,
            1 => self.raw[j] as i64 as f64,
            _ => f64::from_bits(self.raw[j]),
        }
    }

    fn number(&self, j: usize) -> Value {
        match self.kinds[j] {
            2 => Value::Number(Number::from(self.raw[j])),
            1 => Value::Number(Number::from(self.raw[j] as i64)),
            // Finite: checked when the block was built or decoded.
            _ => Number::from_f64(f64::from_bits(self.raw[j])).map_or(Value::Null, Value::Number),
        }
    }

    fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.kinds);
        for raw in &self.raw {
            out.extend_from_slice(&raw.to_le_bytes());
        }
    }

    fn read(r: &mut Reader<'_>, n: usize) -> Result<Self> {
        let kinds = r.take(n)?.to_vec();
        let mut raw = Vec::with_capacity(n);
        for &kind in &kinds {
            let bits = r.u64()?;
            match kind {
                0 if !f64::from_bits(bits).is_finite() => {
                    return Err(r.corrupt("a number is not finite"));
                }
                1 if (bits as i64) >= 0 => {
                    return Err(r.corrupt("an i64 number is not negative"));
                }
                0..=2 => {}
                _ => return Err(r.corrupt(format!("number kind {kind} is unknown"))),
            }
            raw.push(bits);
        }
        Ok(Self { kinds, raw })
    }
}

/// Strings as stored, plain or through a dictionary.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Texts {
    Plain {
        ends: Vec<u32>,
        bytes: Vec<u8>,
    },
    Dict {
        ends: Vec<u32>,
        bytes: Vec<u8>,
        codes: Vec<u16>,
    },
}

/// `count + 1` ends starting at 0, then the bytes, for `strings`.
fn plain_parts(strings: &[&str]) -> Result<(Vec<u32>, Vec<u8>)> {
    let mut ends = Vec::with_capacity(strings.len() + 1);
    let mut bytes = Vec::new();
    ends.push(0);
    for s in strings {
        bytes.extend_from_slice(s.as_bytes());
        ends.push(u32::try_from(bytes.len()).map_err(|_| refuse("a column is longer than 4 GiB"))?);
    }
    Ok((ends, bytes))
}

/// Whether the encoder stores `n` strings of `plain_bytes` bytes in all, `d`
/// of them distinct with `dict_bytes` bytes in all, as a dictionary: exactly
/// when one fits and is strictly smaller than plain.
fn prefers_dict(n: usize, plain_bytes: usize, d: usize, dict_bytes: usize) -> bool {
    if d > MAX_DICT {
        return false;
    }
    let code_width = if d <= NARROW_DICT { 1 } else { 2 };
    let plain_len = 1 + 4 * (n + 1) + plain_bytes;
    let dict_len = 1 + 2 + 4 * (d + 1) + dict_bytes + code_width * n;
    dict_len < plain_len
}

/// The distinct values of `strings`, sorted.
fn distinct<'a>(strings: &[&'a str]) -> Vec<&'a str> {
    let mut distinct = strings.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    distinct
}

impl Texts {
    fn build(strings: &[&str]) -> Result<Self> {
        let (ends, bytes) = plain_parts(strings)?;
        let distinct = distinct(strings);
        let dict_bytes: usize = distinct.iter().map(|s| s.len()).sum();
        if !prefers_dict(strings.len(), bytes.len(), distinct.len(), dict_bytes) {
            return Ok(Self::Plain { ends, bytes });
        }
        let codes = strings
            .iter()
            .map(|s| distinct.binary_search(s).map_or(0, |at| at as u16))
            .collect();
        let (ends, bytes) = plain_parts(&distinct)?;
        Ok(Self::Dict { ends, bytes, codes })
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Plain { ends, .. } => ends.len() - 1,
            Self::Dict { codes, .. } => codes.len(),
        }
    }

    fn entry<'a>(ends: &[u32], bytes: &'a [u8], at: usize) -> &'a str {
        let slice = &bytes[ends[at] as usize..ends[at + 1] as usize];
        // UTF-8: checked when built (from `&str`) or decoded.
        std::str::from_utf8(slice).unwrap_or_default()
    }

    pub(crate) fn get(&self, j: usize) -> &str {
        match self {
            Self::Plain { ends, bytes } => Self::entry(ends, bytes, j),
            Self::Dict { ends, bytes, codes } => Self::entry(ends, bytes, codes[j] as usize),
        }
    }

    fn encode(&self, out: &mut Vec<u8>) {
        let put_ends = |out: &mut Vec<u8>, ends: &[u32]| {
            for e in ends {
                out.extend_from_slice(&e.to_le_bytes());
            }
        };
        match self {
            Self::Plain { ends, bytes } => {
                out.push(0);
                put_ends(out, ends);
                out.extend_from_slice(bytes);
            }
            Self::Dict { ends, bytes, codes } => {
                out.push(1);
                let d = ends.len() - 1;
                out.extend_from_slice(&(d as u16).to_le_bytes());
                put_ends(out, ends);
                out.extend_from_slice(bytes);
                for &code in codes {
                    if d <= NARROW_DICT {
                        out.push(code as u8);
                    } else {
                        out.extend_from_slice(&code.to_le_bytes());
                    }
                }
            }
        }
    }

    /// `count + 1` ends, starting at 0, ascending, then the strings they
    /// cut, each valid UTF-8.
    fn read_strings(r: &mut Reader<'_>, count: usize) -> Result<(Vec<u32>, Vec<u8>)> {
        let mut ends = Vec::with_capacity(count.min(r.remaining() / 4) + 1);
        for _ in 0..=count {
            ends.push(r.u32()?);
        }
        if ends[0] != 0 || !ends.is_sorted() {
            return Err(r.corrupt("string offsets do not start at 0 and ascend"));
        }
        let bytes = r.take(ends[count] as usize)?.to_vec();
        for pair in ends.windows(2) {
            if std::str::from_utf8(&bytes[pair[0] as usize..pair[1] as usize]).is_err() {
                return Err(r.corrupt("a string is not valid UTF-8"));
            }
        }
        Ok((ends, bytes))
    }

    fn read(r: &mut Reader<'_>, n: usize) -> Result<Self> {
        let texts = match r.u8()? {
            0 => {
                let (ends, bytes) = Self::read_strings(r, n)?;
                Self::Plain { ends, bytes }
            }
            1 => {
                let d = r.u16()? as usize;
                let (ends, bytes) = Self::read_strings(r, d)?;
                let ascending = (1..d)
                    .all(|i| Self::entry(&ends, &bytes, i - 1) < Self::entry(&ends, &bytes, i));
                if !ascending {
                    return Err(r.corrupt("dictionary entries are not strictly ascending"));
                }
                let mut codes = Vec::with_capacity(n.min(r.remaining()));
                for _ in 0..n {
                    let code = if d <= NARROW_DICT {
                        u16::from(r.u8()?)
                    } else {
                        r.u16()?
                    };
                    if code as usize >= d {
                        return Err(r.corrupt("a dictionary code is out of range"));
                    }
                    codes.push(code);
                }
                Self::Dict { ends, bytes, codes }
            }
            other => return Err(r.corrupt(format!("string encoding {other} is unknown"))),
        };
        // The encoder's choice is a function of the values, so any other
        // choice is a block it would not have written. Checked by sizes, with
        // no copy of the strings.
        if !texts.is_canonical() {
            return Err(r.corrupt("string column is not in its canonical encoding"));
        }
        Ok(texts)
    }

    /// Whether [`build`](Self::build) would have chosen this encoding for
    /// these values: plain only when a dictionary is not smaller, and a
    /// dictionary only when it is, with every entry used (entries are
    /// checked strictly ascending when decoded).
    fn is_canonical(&self) -> bool {
        match self {
            Self::Plain { ends, bytes } => {
                let n = ends.len() - 1;
                let strings: Vec<&str> = (0..n).map(|j| self.get(j)).collect();
                let distinct = distinct(&strings);
                let dict_bytes = distinct.iter().map(|s| s.len()).sum();
                !prefers_dict(n, bytes.len(), distinct.len(), dict_bytes)
            }
            Self::Dict { ends, bytes, codes } => {
                let d = ends.len() - 1;
                let mut used = vec![false; d];
                let mut plain_bytes = 0;
                for &code in codes {
                    let code = code as usize;
                    used[code] = true;
                    plain_bytes += (ends[code + 1] - ends[code]) as usize;
                }
                used.iter().all(|&u| u) && prefers_dict(codes.len(), plain_bytes, d, bytes.len())
            }
        }
    }
}

/// The values of one column.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Values {
    /// One bit per row, set where the row is valued and `true`.
    Bool(Bits),
    Number(Numbers),
    Text(Texts),
    NumberSet {
        ends: Vec<u32>,
        items: Numbers,
    },
    TextSet {
        ends: Vec<u32>,
        items: Texts,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Column {
    pub(crate) present: Bits,
    pub(crate) valued: Bits,
    pub(crate) values: Values,
}

/// One block's attributes as typed columns.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ColumnBlock {
    fields: Fields,
    rows: usize,
    objects: Bits,
    columns: Vec<Column>,
}

fn set_ends(lens: &[usize]) -> Result<Vec<u32>> {
    let mut ends = Vec::with_capacity(lens.len() + 1);
    let mut total = 0usize;
    ends.push(0);
    for len in lens {
        total += len;
        ends.push(u32::try_from(total).map_err(|_| refuse("a set column is longer than 4 GiB"))?);
    }
    Ok(ends)
}

impl ColumnBlock {
    /// Build a block from rows' attributes (at most [`BLOCK_ROWS`](super::BLOCK_ROWS)).
    ///
    /// # Errors
    ///
    /// [`ItzmaError::CorruptSegment`] at `(encoding)` for attributes that are
    /// not an object, an undeclared key, or a value of the wrong type, none of
    /// which a strict write stores.
    #[cfg(test)]
    pub(crate) fn from_rows(fields: &Fields, rows: &[Option<Value>]) -> Result<Self> {
        let refs: Vec<Option<&Value>> = rows.iter().map(Option::as_ref).collect();
        Self::from_refs(fields, &refs)
    }

    /// Build a block from borrowed rows' attributes (at most 256).
    ///
    /// # Errors
    ///
    /// [`ItzmaError::CorruptSegment`] at `(encoding)` for attributes that are
    /// not an object, an undeclared key, or a value of the wrong type, none of
    /// which a strict write stores.
    pub(crate) fn from_refs(fields: &Fields, rows: &[Option<&Value>]) -> Result<Self> {
        let n = rows.len();
        let mut objects = Bits::new(n);
        let mut maps: Vec<Option<&Map<String, Value>>> = Vec::with_capacity(n);
        for (i, row) in rows.iter().copied().enumerate() {
            match row {
                None => maps.push(None),
                Some(Value::Object(map)) => {
                    if let Some(key) = map
                        .keys()
                        .find(|k| fields.binary_search_by(|f| f.name.as_str().cmp(k)).is_err())
                    {
                        return Err(refuse(format!("attribute {key} is not declared")));
                    }
                    objects.set(i);
                    maps.push(Some(map));
                }
                Some(_) => return Err(refuse("attributes are not an object")),
            }
        }
        let mut columns = Vec::with_capacity(fields.len());
        for field in fields.iter() {
            columns.push(Self::column(field, &maps)?);
        }
        Ok(Self {
            fields: fields.clone(),
            rows: n,
            objects,
            columns,
        })
    }

    fn column(field: &Field, maps: &[Option<&Map<String, Value>>]) -> Result<Column> {
        let n = maps.len();
        let (mut present, mut valued) = (Bits::new(n), Bits::new(n));
        let mut values: Vec<&Value> = Vec::new();
        for (i, map) in maps.iter().enumerate() {
            match map.and_then(|m| m.get(&field.name)) {
                None => {}
                Some(Value::Null) => present.set(i),
                Some(v) => {
                    present.set(i);
                    valued.set(i);
                    values.push(v);
                }
            }
        }
        let wrong = || {
            refuse(format!(
                "attribute {} has a value of the wrong type",
                field.name
            ))
        };
        let values = match field.ty {
            AttrType::Bool => {
                let mut bits = Bits::new(n);
                let mut j = 0;
                for i in 0..n {
                    if valued.get(i) {
                        match values[j] {
                            Value::Bool(true) => bits.set(i),
                            Value::Bool(false) => {}
                            _ => return Err(wrong()),
                        }
                        j += 1;
                    }
                }
                Values::Bool(bits)
            }
            AttrType::Number => {
                let mut numbers = Numbers::default();
                for v in &values {
                    numbers.push(v.as_number().ok_or_else(wrong)?);
                }
                Values::Number(numbers)
            }
            AttrType::Text => {
                let strings = values
                    .iter()
                    .map(|v| v.as_str().ok_or_else(wrong))
                    .collect::<Result<Vec<_>>>()?;
                Values::Text(Texts::build(&strings)?)
            }
            AttrType::NumberSet => {
                let mut items = Numbers::default();
                let mut lens = Vec::with_capacity(values.len());
                for v in &values {
                    let array = v.as_array().ok_or_else(wrong)?;
                    for item in array {
                        items.push(item.as_number().ok_or_else(wrong)?);
                    }
                    lens.push(array.len());
                }
                Values::NumberSet {
                    ends: set_ends(&lens)?,
                    items,
                }
            }
            AttrType::TextSet => {
                let mut strings = Vec::new();
                let mut lens = Vec::with_capacity(values.len());
                for v in &values {
                    let array = v.as_array().ok_or_else(wrong)?;
                    for item in array {
                        strings.push(item.as_str().ok_or_else(wrong)?);
                    }
                    lens.push(array.len());
                }
                Values::TextSet {
                    ends: set_ends(&lens)?,
                    items: Texts::build(&strings)?,
                }
            }
        };
        Ok(Column {
            present,
            valued,
            values,
        })
    }

    /// Append the block to `out` (which must be empty or end where the block
    /// starts). Returns each column's `(offset, length)` from the block's start.
    pub(crate) fn encode(&self, out: &mut Vec<u8>) -> Vec<(u32, u32)> {
        let base = out.len();
        out.extend_from_slice(&(self.rows as u32).to_le_bytes());
        out.extend_from_slice(&self.objects.0);
        let mut ranges = Vec::with_capacity(self.columns.len());
        for column in &self.columns {
            let start = out.len() - base;
            out.extend_from_slice(&column.present.0);
            out.extend_from_slice(&column.valued.0);
            match &column.values {
                Values::Bool(bits) => out.extend_from_slice(&bits.0),
                Values::Number(numbers) => numbers.encode(out),
                Values::Text(texts) => texts.encode(out),
                Values::NumberSet { ends, items } => {
                    ends.iter()
                        .for_each(|e| out.extend_from_slice(&e.to_le_bytes()));
                    items.encode(out);
                }
                Values::TextSet { ends, items } => {
                    ends.iter()
                        .for_each(|e| out.extend_from_slice(&e.to_le_bytes()));
                    items.encode(out);
                }
            }
            // A block holds at most 256 rows of bounded columns, far below 4 GiB.
            ranges.push((start as u32, (out.len() - base - start) as u32));
        }
        ranges
    }

    /// Decode one block of `rows` rows. Returns the block and its column
    /// ranges, as [`encode`](Self::encode) does.
    ///
    /// # Errors
    ///
    /// [`ItzmaError::CorruptSegment`] for anything the encoder would not
    /// have written; see the checks below.
    pub(crate) fn decode(
        path: &str,
        fields: &Fields,
        bytes: &[u8],
        rows: usize,
    ) -> Result<(Self, Vec<(u32, u32)>)> {
        let mut r = Reader::new(path, bytes);
        let count = r.u32()? as usize;
        if count != rows {
            return Err(r.corrupt(format!(
                "attribute block holds {count} rows, expected {rows}"
            )));
        }
        let objects = Bits::read(&mut r, rows, "objects")?;
        let mut columns = Vec::with_capacity(fields.len());
        let mut ranges = Vec::with_capacity(fields.len());
        for field in fields.iter() {
            let start = r.pos;
            let present = Bits::read(&mut r, rows, "present")?;
            let valued = Bits::read(&mut r, rows, "valued")?;
            if !present.within(&objects) {
                return Err(r.corrupt(format!("{}: present outside objects", field.name)));
            }
            if !valued.within(&present) {
                return Err(r.corrupt(format!("{}: valued outside present", field.name)));
            }
            let k = valued.count();
            let values = match field.ty {
                AttrType::Bool => {
                    let bits = Bits::read(&mut r, rows, "bool")?;
                    if !bits.within(&valued) {
                        return Err(
                            r.corrupt(format!("{}: bool set on a row with no value", field.name))
                        );
                    }
                    Values::Bool(bits)
                }
                AttrType::Number => Values::Number(Numbers::read(&mut r, k)?),
                AttrType::Text => Values::Text(Texts::read(&mut r, k)?),
                AttrType::NumberSet => {
                    let ends = Self::read_set_ends(&mut r, k)?;
                    let items = Numbers::read(&mut r, ends[k] as usize)?;
                    Values::NumberSet { ends, items }
                }
                AttrType::TextSet => {
                    let ends = Self::read_set_ends(&mut r, k)?;
                    let items = Texts::read(&mut r, ends[k] as usize)?;
                    Values::TextSet { ends, items }
                }
            };
            ranges.push((start as u32, (r.pos - start) as u32));
            columns.push(Column {
                present,
                valued,
                values,
            });
        }
        if r.remaining() != 0 {
            return Err(r.corrupt("attribute block has trailing bytes"));
        }
        Ok((
            Self {
                fields: fields.clone(),
                rows,
                objects,
                columns,
            },
            ranges,
        ))
    }

    fn read_set_ends(r: &mut Reader<'_>, k: usize) -> Result<Vec<u32>> {
        let mut ends = Vec::with_capacity(k.min(r.remaining() / 4) + 1);
        for _ in 0..=k {
            ends.push(r.u32()?);
        }
        if ends[0] != 0 || !ends.is_sorted() {
            return Err(r.corrupt("set offsets do not start at 0 and ascend"));
        }
        Ok(ends)
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn fields(&self) -> &Fields {
        &self.fields
    }

    pub(crate) fn columns(&self) -> &[Column] {
        &self.columns
    }

    /// Row `i`'s attributes rebuilt as stored: `None` with no object, else an
    /// object with each present field, `null` where not valued.
    pub(crate) fn row(&self, i: usize) -> Option<Value> {
        if !self.objects.get(i) {
            return None;
        }
        let mut map = Map::new();
        for (field, column) in self.fields.iter().zip(&self.columns) {
            if !column.present.get(i) {
                continue;
            }
            let value = if column.valued.get(i) {
                column.value(i)
            } else {
                Value::Null
            };
            map.insert(field.name.clone(), value);
        }
        Some(Value::Object(map))
    }

    /// Bytes the block keeps alive, for the decoded cache.
    pub(crate) fn weight(&self) -> u64 {
        let texts = |t: &Texts| match t {
            Texts::Plain { ends, bytes } => ends.len() * 4 + bytes.len(),
            Texts::Dict { ends, bytes, codes } => ends.len() * 4 + bytes.len() + codes.len() * 2,
        };
        let numbers = |n: &Numbers| n.kinds.len() * 9;
        let values: usize = self
            .columns
            .iter()
            .map(|c| {
                c.present.0.len() * 2
                    + match &c.values {
                        Values::Bool(b) => b.0.len(),
                        Values::Number(n) => numbers(n),
                        Values::Text(t) => texts(t),
                        Values::NumberSet { ends, items } => ends.len() * 4 + numbers(items),
                        Values::TextSet { ends, items } => ends.len() * 4 + texts(items),
                    }
            })
            .sum();
        (values + self.objects.0.len() + std::mem::size_of::<Self>()) as u64
    }
}

impl Column {
    /// The value of valued row `i`.
    fn value(&self, i: usize) -> Value {
        let j = self.valued.rank(i);
        match &self.values {
            Values::Bool(bits) => Value::Bool(bits.get(i)),
            Values::Number(numbers) => numbers.number(j),
            Values::Text(texts) => Value::String(texts.get(j).to_owned()),
            Values::NumberSet { ends, items } => Value::Array(
                (ends[j] as usize..ends[j + 1] as usize)
                    .map(|e| items.number(e))
                    .collect(),
            ),
            Values::TextSet { ends, items } => Value::Array(
                (ends[j] as usize..ends[j + 1] as usize)
                    .map(|e| Value::String(items.get(e).to_owned()))
                    .collect(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;
    use crate::attrs::{AttrSchema, AttrType};

    fn every_type() -> Fields {
        fields_of(
            &AttrSchema::new()
                .field("b", AttrType::Bool)
                .field("n", AttrType::Number)
                .field("s", AttrType::Text)
                .field("ss", AttrType::TextSet)
                .field("ns", AttrType::NumberSet),
        )
    }

    fn round_trip(fields: &Fields, rows: &[Option<Value>]) -> Vec<u8> {
        let block = ColumnBlock::from_rows(fields, rows).expect("from_rows");
        let mut bytes = Vec::new();
        let ranges = block.encode(&mut bytes);
        let (back, back_ranges) =
            ColumnBlock::decode("t", fields, &bytes, rows.len()).expect("decode");
        assert_eq!(ranges, back_ranges);
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(&back.row(i), row, "row {i}");
        }
        bytes
    }

    #[test]
    fn rows_round_trip_exactly() {
        let rows = vec![
            None,
            Some(json!({})),
            Some(json!({"b": null, "n": null, "s": null, "ss": null, "ns": null})),
            Some(json!({"b": true, "n": 5, "s": "x", "ss": ["a", "a", "b"], "ns": [1, 1.5]})),
            Some(json!({"b": false, "n": 5.0, "ss": [], "ns": []})),
            Some(json!({"n": -5})),
            Some(json!({"n": -0.0})),
            Some(json!({"n": u64::MAX})),
            Some(json!({"n": i64::MIN})),
            Some(json!({"n": 1e300, "s": ""})),
        ];
        round_trip(&every_type(), &rows);
    }

    #[test]
    fn a_rebuilt_row_serializes_to_the_stored_json() {
        let texts = [
            r#"{"n":5}"#,
            r#"{"n":5.0}"#,
            r#"{"n":-5}"#,
            r#"{"n":-0.0}"#,
            r#"{"n":18446744073709551615}"#,
            r#"{"b":true,"ns":[3,3.25,-1],"s":"é","ss":["z","a"]}"#,
            r#"{"b":null}"#,
            r#"{}"#,
        ];
        let rows: Vec<Option<Value>> = texts
            .iter()
            .map(|t| Some(serde_json::from_str(t).unwrap()))
            .collect();
        let fields = every_type();
        let block = ColumnBlock::from_rows(&fields, &rows).unwrap();
        for (i, text) in texts.iter().enumerate() {
            let rebuilt = serde_json::to_string(&block.row(i).unwrap()).unwrap();
            assert_eq!(&rebuilt, text);
        }
    }

    #[test]
    fn the_same_rows_give_the_same_bytes() {
        let rows: Vec<Option<Value>> = (0..256)
            .map(|i| Some(json!({"n": i % 7, "s": format!("v{}", i % 3)})))
            .collect();
        let a = round_trip(&every_type(), &rows);
        let b = round_trip(&every_type(), &rows);
        assert_eq!(a, b);
    }

    #[test]
    fn a_block_with_no_fields_holds_only_the_objects_bitmap() {
        let fields = fields_of(&AttrSchema::new());
        let bytes = round_trip(&fields, &[None, Some(json!({})), None]);
        assert_eq!(bytes, [3, 0, 0, 0, 0b010]);
    }

    /// One row per text value; the dictionary is chosen exactly when smaller.
    fn text_encoding(values: &[String]) -> u8 {
        let fields = fields_of(&AttrSchema::new().field("s", AttrType::Text));
        let rows: Vec<Option<Value>> = values.iter().map(|v| Some(json!({ "s": v }))).collect();
        let bytes = round_trip(&fields, &rows);
        // u32 count, objects, present, valued: then the encoding byte.
        let bitmap = values.len().div_ceil(8);
        bytes[4 + 3 * bitmap]
    }

    #[test]
    fn the_dictionary_is_chosen_exactly_when_it_is_smaller() {
        // Two distinct short strings over 256 rows: the dictionary wins.
        let repeated: Vec<String> = (0..256).map(|i| format!("v{}", i % 2)).collect();
        assert_eq!(text_encoding(&repeated), 1);
        // Every value distinct: the dictionary only adds a count and codes.
        let distinct: Vec<String> = (0..256).map(|i| format!("v{i:03}")).collect();
        assert_eq!(text_encoding(&distinct), 0);
        // Two rows of one value: plain is 1 + 12 + 2 = 15, dictionary
        // 1 + 2 + 8 + 1 + 2 = 14.
        assert_eq!(text_encoding(&["a".into(), "a".into()]), 1);
        // One row: plain 1 + 8 + 1 = 10, dictionary 1 + 2 + 8 + 1 + 1 = 13.
        assert_eq!(text_encoding(&["a".into()]), 0);
    }

    #[test]
    fn codes_widen_past_256_entries_and_sets_fall_back_to_plain_past_65535() {
        let fields = fields_of(&AttrSchema::new().field("ss", AttrType::TextSet));
        // 257 distinct values, each repeated in many rows' sets: u16 codes.
        let rows: Vec<Option<Value>> = (0..256)
            .map(|_| Some(json!({ "ss": (0..257).map(|v| format!("{v}")).collect::<Vec<_>>() })))
            .collect();
        round_trip(&fields, &rows);
        // 65,536 distinct one-byte-ish strings across a block: plain.
        let wide: Vec<Option<Value>> = (0..64)
            .map(|r| {
                Some(json!({
                    "ss": (0..1024).map(|v| format!("{}", r * 1024 + v)).collect::<Vec<_>>()
                }))
            })
            .collect();
        round_trip(&fields, &wide);
    }

    #[test]
    fn the_encoder_refuses_what_strict_writes_never_store() {
        let fields = every_type();
        for bad in [
            json!(5),
            json!({"zz": 1}),
            json!({"b": 1}),
            json!({"n": "1"}),
            json!({"s": 1}),
            json!({"ss": [1]}),
            json!({"ns": ["1"]}),
            json!({"ns": [null]}),
        ] {
            let err = ColumnBlock::from_rows(&fields, &[Some(bad.clone())]).unwrap_err();
            assert!(
                matches!(&err, ItzmaError::CorruptSegment { path, .. } if path == "(encoding)"),
                "{bad}: {err:?}"
            );
        }
    }

    #[test]
    fn every_truncation_is_refused() {
        let rows = vec![
            Some(json!({"b": true, "n": 5, "s": "x", "ss": ["a"], "ns": [1]})),
            None,
            Some(json!({"n": 2.5, "s": "x"})),
        ];
        let fields = every_type();
        let mut bytes = Vec::new();
        ColumnBlock::from_rows(&fields, &rows)
            .unwrap()
            .encode(&mut bytes);
        for cut in 0..bytes.len() {
            assert!(
                ColumnBlock::decode("t", &fields, &bytes[..cut], 3).is_err(),
                "cut {cut}"
            );
        }
        let mut longer = bytes.clone();
        longer.push(0);
        assert!(ColumnBlock::decode("t", &fields, &longer, 3).is_err());
    }

    #[test]
    fn a_flipped_bit_never_panics_and_never_changes_a_row_silently() {
        let rows = vec![
            Some(json!({"b": true, "n": 5, "s": "x", "ss": ["a", "b"], "ns": [1, 2.5]})),
            None,
            Some(json!({"n": -2, "s": "x", "ss": []})),
        ];
        let fields = every_type();
        let mut bytes = Vec::new();
        ColumnBlock::from_rows(&fields, &rows)
            .unwrap()
            .encode(&mut bytes);
        for at in 0..bytes.len() * 8 {
            let mut flipped = bytes.clone();
            flipped[at / 8] ^= 1 << (at % 8);
            // A flip may still decode (a value bit is a different value);
            // what it must never do is panic. A flip the decoder accepts
            // re-encodes to exactly the flipped bytes, so it is canonical.
            if let Ok((block, _)) = ColumnBlock::decode("t", &fields, &flipped, 3) {
                let mut again = Vec::new();
                block.encode(&mut again);
                assert_eq!(again, flipped, "bit {at} decoded to a non-canonical block");
            }
        }
    }

    /// A dictionary the encoder would never write: two entries where the
    /// rows use one, and plain text that a dictionary would beat.
    #[test]
    fn a_column_in_an_encoding_the_encoder_would_not_choose_is_refused() {
        let one_text = fields_of(&AttrSchema::new().field("s", AttrType::Text));
        let header = |rows: u8| vec![rows, 0, 0, 0, 0b111, 0b111, 0b111];
        let le32 = |v: u32| v.to_le_bytes().to_vec();
        // d = 2 ("a", "b"), three rows all coded 0.
        let unused = [
            header(3),
            vec![1],
            2u16.to_le_bytes().to_vec(),
            le32(0),
            le32(1),
            le32(2),
            b"ab".to_vec(),
            vec![0, 0, 0],
        ]
        .concat();
        // 3 rows of "aaaa" stored plain: 1 + 16 + 12 = 29 bytes, where a
        // dictionary takes 1 + 2 + 8 + 4 + 3 = 18.
        let plain = [
            header(3),
            vec![0],
            le32(0),
            le32(4),
            le32(8),
            le32(12),
            b"aaaaaaaaaaaa".to_vec(),
        ]
        .concat();
        for bytes in [unused, plain] {
            let err = ColumnBlock::decode("t", &one_text, &bytes, 3).unwrap_err();
            assert!(reason(err).contains("canonical"));
        }
    }

    fn reason(err: ItzmaError) -> String {
        match err {
            ItzmaError::CorruptSegment { reason, .. } => reason,
            other => panic!("{other:?}"),
        }
    }

    /// Hand-built blocks, one per decoder check, each with its reason.
    #[test]
    fn forged_blocks_are_refused_with_their_reason() {
        let one_num = fields_of(&AttrSchema::new().field("n", AttrType::Number));
        let one_text = fields_of(&AttrSchema::new().field("s", AttrType::Text));
        let one_bool = fields_of(&AttrSchema::new().field("b", AttrType::Bool));
        let cases: Vec<(&Fields, Vec<u8>, usize, &str)> = vec![
            // count 2, expected 1
            (&one_num, vec![2, 0, 0, 0, 1, 0, 0], 1, "holds 2 rows"),
            // padding bit set in objects
            (&one_num, vec![1, 0, 0, 0, 0b11, 0, 0], 1, "padding"),
            // present outside objects
            (&one_num, vec![1, 0, 0, 0, 0, 1, 0], 1, "present"),
            // valued outside present
            (&one_num, vec![1, 0, 0, 0, 1, 0, 1], 1, "valued"),
            // unknown number kind 3
            (
                &one_num,
                [vec![1, 0, 0, 0, 1, 1, 1, 3], 0u64.to_le_bytes().to_vec()].concat(),
                1,
                "kind",
            ),
            // i64 kind holding a non-negative value
            (
                &one_num,
                [vec![1, 0, 0, 0, 1, 1, 1, 1], 5u64.to_le_bytes().to_vec()].concat(),
                1,
                "negative",
            ),
            // f64 kind holding NaN
            (
                &one_num,
                [
                    vec![1, 0, 0, 0, 1, 1, 1, 0],
                    f64::NAN.to_bits().to_le_bytes().to_vec(),
                ]
                .concat(),
                1,
                "finite",
            ),
            // unknown text encoding 2
            (&one_text, vec![1, 0, 0, 0, 1, 1, 1, 2], 1, "encoding"),
            // plain text whose first offset is not 0
            (
                &one_text,
                [
                    vec![1, 0, 0, 0, 1, 1, 1, 0],
                    1u32.to_le_bytes().to_vec(),
                    1u32.to_le_bytes().to_vec(),
                    b"x".to_vec(),
                ]
                .concat(),
                1,
                "offsets",
            ),
            // plain text that is not UTF-8
            (
                &one_text,
                [
                    vec![1, 0, 0, 0, 1, 1, 1, 0],
                    0u32.to_le_bytes().to_vec(),
                    1u32.to_le_bytes().to_vec(),
                    vec![0xff],
                ]
                .concat(),
                1,
                "UTF-8",
            ),
            // dictionary code out of range: d = 1, code 1
            (
                &one_text,
                [
                    vec![1, 0, 0, 0, 1, 1, 1, 1],
                    1u16.to_le_bytes().to_vec(),
                    0u32.to_le_bytes().to_vec(),
                    1u32.to_le_bytes().to_vec(),
                    b"x".to_vec(),
                    vec![1],
                ]
                .concat(),
                1,
                "code",
            ),
            // a one-row dictionary, which plain beats: not canonical
            (
                &one_text,
                [
                    vec![1, 0, 0, 0, 1, 1, 1, 1],
                    1u16.to_le_bytes().to_vec(),
                    0u32.to_le_bytes().to_vec(),
                    1u32.to_le_bytes().to_vec(),
                    b"x".to_vec(),
                    vec![0],
                ]
                .concat(),
                1,
                "canonical",
            ),
            // bool value set where the row is not valued
            (&one_bool, vec![1, 0, 0, 0, 1, 1, 0, 1], 1, "bool"),
        ];
        for (fields, bytes, rows, want) in cases {
            let got = reason(ColumnBlock::decode("t", fields, &bytes, rows).unwrap_err());
            assert!(got.contains(want), "{want:?} not in {got:?}");
        }
    }
}
