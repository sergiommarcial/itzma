//! A batch of upserts held as columns.
//!
//! An Arrow table arrives as columns and the WAL stores columns, so a load
//! that turned every row into a [`Document`](crate::Document) and back would
//! copy every vector three times and build a JSON tree per row for nothing.
//! [`DocColumns`] keeps the batch as flat buffers: ids as one UTF-8 buffer and
//! offsets, vectors as one row-major `f32` slice and one typed column per
//! attribute. The constructors check the shape
//! (offsets, lengths, UTF-8), and [`DocColumns::validate`] checks the batch
//! against a namespace's schema the way `upsert` checks documents, column by
//! column.

use crate::attrs::{AttrType, MAX_SET_ELEMENTS};
use crate::document::{self, Schema};
use crate::error::{ItzmaError, Result};

/// A batch of upserts as columns. Build it with [`DocColumns::new`], then
/// [`with_attr`](Self::with_attr); the constructors are the only way in, so every batch is well formed.
///
/// Every row has every attribute column: a column cannot leave a field out
/// of one row, only make it `null` there (`valid` false). A batch with no
/// attribute columns writes no attributes at all, as a document with
/// `attrs: None` does.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct DocColumns {
    ids: Vec<u8>,
    id_offsets: Vec<u32>,
    vectors: Vec<f32>,
    dim: usize,
    /// Sorted by name, in byte order: the order `serde_json` writes an
    /// object's keys in, which keeps the WAL bytes equal to `upsert`'s.
    attrs: Vec<(String, AttrColumn)>,
}

/// One attribute column of a [`DocColumns`] batch.
///
/// `valid[row]` false means the field is `null` on that row, and whatever
/// the value buffers hold there is ignored. A set's `offsets` (one more than
/// the rows) cut its items into rows, and `items_valid` says which items are
/// not `null`. Numbers keep their kind, so `5` stays an integer and `5.0` a
/// float, as `serde_json` writes them.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum AttrColumn {
    /// A column that is `null` on every one of `rows` rows (Arrow's `Null`
    /// type).
    Null {
        /// The number of rows.
        rows: usize,
    },
    /// Booleans.
    Bool {
        /// Which rows are not `null`.
        valid: Vec<bool>,
        /// One value a row.
        values: Vec<bool>,
    },
    /// Integers.
    I64 {
        /// Which rows are not `null`.
        valid: Vec<bool>,
        /// One value a row.
        values: Vec<i64>,
    },
    /// Floating-point numbers. A valid value must be finite.
    F64 {
        /// Which rows are not `null`.
        valid: Vec<bool>,
        /// One value a row.
        values: Vec<f64>,
    },
    /// Strings, as one UTF-8 buffer cut by `offsets`.
    Text {
        /// Which rows are not `null`.
        valid: Vec<bool>,
        /// Every row's bytes, one after the other.
        bytes: Vec<u8>,
        /// `rows + 1` offsets into `bytes`.
        offsets: Vec<u32>,
    },
    /// Sets of integers.
    I64Set {
        /// Which rows are not `null`.
        valid: Vec<bool>,
        /// `rows + 1` offsets into the items.
        offsets: Vec<u32>,
        /// Which items are not `null`.
        items_valid: Vec<bool>,
        /// One value an item.
        values: Vec<i64>,
    },
    /// Sets of floating-point numbers. A valid item must be finite.
    F64Set {
        /// Which rows are not `null`.
        valid: Vec<bool>,
        /// `rows + 1` offsets into the items.
        offsets: Vec<u32>,
        /// Which items are not `null`.
        items_valid: Vec<bool>,
        /// One value an item.
        values: Vec<f64>,
    },
    /// Sets of strings.
    TextSet {
        /// Which rows are not `null`.
        valid: Vec<bool>,
        /// `rows + 1` offsets into the items.
        offsets: Vec<u32>,
        /// Which items are not `null`.
        items_valid: Vec<bool>,
        /// Every item's bytes, one after the other.
        bytes: Vec<u8>,
        /// One more than the items: offsets into `bytes`.
        byte_offsets: Vec<u32>,
    },
}

fn malformed<T>(reason: &'static str) -> Result<T> {
    Err(ItzmaError::InvalidColumns(reason))
}

/// Whether `offsets` cut `end` items or bytes into `count` runs: `count + 1`
/// entries, from 0, never going down, ending at `end`.
fn offsets_fit(offsets: &[u32], count: usize, end: usize) -> bool {
    offsets.len() == count + 1
        && offsets.first() == Some(&0)
        && offsets.windows(2).all(|w| w[0] <= w[1])
        && offsets.last().map(|&last| last as usize) == Some(end)
}

/// Whether every run `offsets` cuts out of `bytes` is UTF-8 on its own.
fn runs_are_utf8(bytes: &[u8], offsets: &[u32]) -> bool {
    offsets
        .windows(2)
        .all(|w| std::str::from_utf8(&bytes[w[0] as usize..w[1] as usize]).is_ok())
}

/// The run `at` of a buffer whose runs were checked to be UTF-8.
fn run_str<'a>(bytes: &'a [u8], offsets: &[u32], at: usize) -> &'a str {
    let run = &bytes[offsets[at] as usize..offsets[at + 1] as usize];
    std::str::from_utf8(run).unwrap_or_default()
}

/// Keep `offsets[..=at]` and return the rest, rebased to start at 0, with
/// the position in the buffer where the cut falls.
fn split_offsets(offsets: &mut Vec<u32>, at: usize) -> (Vec<u32>, usize) {
    let cut = offsets[at];
    let tail = offsets[at..].iter().map(|o| o - cut).collect();
    offsets.truncate(at + 1);
    (tail, cut as usize)
}

/// Add `other`'s runs after these: its offsets shifted past our last one.
fn append_offsets(offsets: &mut Vec<u32>, other: &[u32]) -> Result<()> {
    let base = offsets.last().copied().unwrap_or(0);
    for &o in other.iter().skip(1) {
        let shifted = base.checked_add(o).ok_or(ItzmaError::InvalidColumns(
            "a batch is too big for its offsets",
        ))?;
        offsets.push(shifted);
    }
    Ok(())
}

fn run_range(offsets: &[u32], at: usize) -> std::ops::Range<usize> {
    offsets[at] as usize..offsets[at + 1] as usize
}

impl DocColumns {
    /// A batch of `id_offsets.len() - 1` rows with no attributes: row `i`'s
    /// id is `ids[id_offsets[i]..id_offsets[i + 1]]`
    /// and its vector `vectors[i * dim..(i + 1) * dim]`.
    ///
    /// # Errors
    ///
    /// [`ItzmaError::InvalidColumns`] if the offsets do not start at 0, go
    /// down or do not end at `ids.len()`, an id is not UTF-8 on its own, or
    /// `vectors` is not `rows * dim` long.
    pub fn new(ids: Vec<u8>, id_offsets: Vec<u32>, vectors: Vec<f32>, dim: usize) -> Result<Self> {
        let rows = id_offsets.len().saturating_sub(1);
        if !offsets_fit(&id_offsets, rows, ids.len()) {
            return malformed("id offsets must run from 0 to the end of the ids, never going down");
        }
        if !runs_are_utf8(&ids, &id_offsets) {
            return malformed("every id must be UTF-8");
        }
        if Some(vectors.len()) != rows.checked_mul(dim) {
            return malformed("vectors must hold rows times dim values");
        }
        Ok(Self {
            ids,
            id_offsets,
            vectors,
            dim,
            attrs: Vec::new(),
        })
    }

    /// Add an attribute column named `name`.
    ///
    /// # Errors
    ///
    /// [`ItzmaError::InvalidColumns`] if the name is empty or already used,
    /// or the column does not have one entry a row (see [`AttrColumn`]), or
    /// its offsets or text are malformed.
    pub fn with_attr(mut self, name: impl Into<String>, column: AttrColumn) -> Result<Self> {
        let name = name.into();
        if name.is_empty() {
            return malformed("an attribute column needs a name");
        }
        column.check_shape(self.rows())?;
        match self.attrs.binary_search_by(|(n, _)| n.as_str().cmp(&name)) {
            Ok(_) => malformed("two attribute columns have the same name"),
            Err(at) => {
                self.attrs.insert(at, (name, column));
                Ok(self)
            }
        }
    }

    /// The number of rows.
    pub fn rows(&self) -> usize {
        self.id_offsets.len().saturating_sub(1)
    }

    /// The length of every vector.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Check the batch against a namespace's schema, without writing: the
    /// checks [`Namespace::upsert`](crate::Namespace::upsert) runs on each
    /// document, run column by column. An input with one defect gets exactly
    /// the error `upsert` gives for the same documents; with several, the
    /// first one reported can differ, since every id is checked before any
    /// vector, every vector before any attribute, and so on.
    ///
    /// # Errors
    ///
    /// - [`ItzmaError::EmptyBatch`] for a batch of no rows.
    /// - [`ItzmaError::InvalidId`] for an empty id or one longer than
    ///   [`MAX_ID_BYTES`](crate::document::MAX_ID_BYTES).
    /// - [`ItzmaError::DimensionMismatch`] and [`ItzmaError::InvalidVector`]
    ///   for vectors, as [`Schema::validate`] decides.
    /// - [`ItzmaError::UnknownAttr`], [`ItzmaError::AttrTypeMismatch`] and
    ///   [`ItzmaError::InvalidAttrs`] for attributes. A free-form namespace
    ///   checks only that every number is finite, since JSON has no NaN.
    pub fn validate(&self, schema: &Schema) -> Result<()> {
        let rows = self.rows();
        if rows == 0 {
            return Err(ItzmaError::EmptyBatch);
        }
        for row in 0..rows {
            document::validate_id(self.id(row))?;
        }
        if self.dim != schema.dim {
            return Err(ItzmaError::DimensionMismatch {
                expected: schema.dim,
                got: self.dim,
            });
        }
        for row in 0..rows {
            schema.validate(self.vector(row))?;
        }
        for (name, column) in &self.attrs {
            if !schema.attrs.is_free_form() {
                let Some(ty) = schema.attrs.get(name) else {
                    return Err(ItzmaError::UnknownAttr {
                        field: name.clone(),
                    });
                };
                column.check_type(name, ty, rows)?;
            }
            column.check_finite()?;
        }
        Ok(())
    }

    pub(crate) fn id(&self, row: usize) -> &str {
        run_str(&self.ids, &self.id_offsets, row)
    }

    pub(crate) fn vector(&self, row: usize) -> &[f32] {
        &self.vectors[row * self.dim..(row + 1) * self.dim]
    }

    pub(crate) fn vectors(&self) -> &[f32] {
        &self.vectors
    }

    pub(crate) fn attrs(&self) -> &[(String, AttrColumn)] {
        &self.attrs
    }

    /// Add `other`'s rows after these, for the stream's chunker. Both must
    /// have the same `dim` and the same attribute names; a column that is
    /// [`AttrColumn::Null`] on one side takes the other side's kind.
    ///
    /// # Errors
    ///
    /// [`ItzmaError::InvalidColumns`] if the batches do not share their
    /// columns, or the result would not fit its `u32` offsets.
    pub(crate) fn append(&mut self, other: DocColumns) -> Result<()> {
        if self.rows() == 0 && self.attrs.is_empty() {
            *self = other;
            return Ok(());
        }
        let same_names = self.attrs.len() == other.attrs.len()
            && self
                .attrs
                .iter()
                .zip(&other.attrs)
                .all(|((a, _), (b, _))| a == b);
        if self.dim != other.dim || !same_names {
            return malformed("batches in one stream must have the same columns");
        }
        append_offsets(&mut self.id_offsets, &other.id_offsets)?;
        self.ids.extend_from_slice(&other.ids);
        self.vectors.extend_from_slice(&other.vectors);
        for ((_, mine), (_, theirs)) in self.attrs.iter_mut().zip(other.attrs) {
            mine.append(theirs)?;
        }
        Ok(())
    }

    /// Split off rows `at..` into a batch of their own and keep `..at`.
    pub(crate) fn split_off(&mut self, at: usize) -> DocColumns {
        let (id_offsets, cut) = split_offsets(&mut self.id_offsets, at);
        DocColumns {
            ids: self.ids.split_off(cut),
            id_offsets,
            vectors: self.vectors.split_off(at * self.dim),
            dim: self.dim,
            attrs: self
                .attrs
                .iter_mut()
                .map(|(name, column)| (name.clone(), column.split_off(at)))
                .collect(),
        }
    }

    /// Give back spare capacity, as a `split_off` leaves the head holding
    /// all of it.
    pub(crate) fn shrink_to_fit(&mut self) {
        self.ids.shrink_to_fit();
        self.id_offsets.shrink_to_fit();
        self.vectors.shrink_to_fit();
        for (_, column) in &mut self.attrs {
            column.shrink_to_fit();
        }
    }

    /// The batch without the column `name`, for tests that build a variant.
    #[cfg(test)]
    pub(crate) fn without_attr(mut self, name: &str) -> Self {
        self.attrs.retain(|(n, _)| n != name);
        self
    }
}

impl AttrColumn {
    /// Check that the column has `rows` rows and well-formed offsets and text.
    fn check_shape(&self, rows: usize) -> Result<()> {
        let ok = match self {
            AttrColumn::Null { rows: n } => *n == rows,
            AttrColumn::Bool { valid, values } => valid.len() == rows && values.len() == rows,
            AttrColumn::I64 { valid, values } => valid.len() == rows && values.len() == rows,
            AttrColumn::F64 { valid, values } => valid.len() == rows && values.len() == rows,
            AttrColumn::Text {
                valid,
                bytes,
                offsets,
            } => {
                if !(valid.len() == rows && offsets_fit(offsets, rows, bytes.len())) {
                    false
                } else if !runs_are_utf8(bytes, offsets) {
                    return malformed("text must be UTF-8");
                } else {
                    true
                }
            }
            AttrColumn::I64Set {
                valid,
                offsets,
                items_valid,
                values,
            } => set_fits(valid, offsets, items_valid, values.len(), rows),
            AttrColumn::F64Set {
                valid,
                offsets,
                items_valid,
                values,
            } => set_fits(valid, offsets, items_valid, values.len(), rows),
            AttrColumn::TextSet {
                valid,
                offsets,
                items_valid,
                bytes,
                byte_offsets,
            } => {
                let items = items_valid.len();
                if !(set_fits(valid, offsets, items_valid, items, rows)
                    && offsets_fit(byte_offsets, items, bytes.len()))
                {
                    false
                } else if !runs_are_utf8(bytes, byte_offsets) {
                    return malformed("text must be UTF-8");
                } else {
                    true
                }
            }
        };
        if ok {
            Ok(())
        } else {
            malformed("an attribute column must have one entry a row, with offsets that fit")
        }
    }

    /// Check every non-null row against the declared type `ty`, the way
    /// `attrs::check_value` checks a JSON value: a set's length first, then
    /// its items, and a `null` item never fits a typed set.
    fn check_type(&self, field: &str, ty: AttrType, rows: usize) -> Result<()> {
        let mismatch = |found: &'static str| {
            Err(ItzmaError::AttrTypeMismatch {
                field: field.to_string(),
                expected: ty.name(),
                found,
            })
        };
        let (kind, item_ty) = match self {
            AttrColumn::Null { .. } => return Ok(()),
            AttrColumn::Bool { .. } => ("bool", AttrType::Bool),
            AttrColumn::I64 { .. } | AttrColumn::F64 { .. } => ("number", AttrType::Number),
            AttrColumn::Text { .. } => ("string", AttrType::Text),
            AttrColumn::I64Set { .. } | AttrColumn::F64Set { .. } => ("array", AttrType::NumberSet),
            AttrColumn::TextSet { .. } => ("array", AttrType::TextSet),
        };
        for row in (0..rows).filter(|&row| !self.is_null(row)) {
            let Some((offsets, items_valid)) = self.set_parts() else {
                if ty != item_ty {
                    return mismatch(kind);
                }
                continue;
            };
            if !matches!(ty, AttrType::TextSet | AttrType::NumberSet) {
                return mismatch(kind);
            }
            let items = run_range(offsets, row);
            if items.len() > MAX_SET_ELEMENTS {
                return Err(ItzmaError::InvalidAttrs("a set has too many elements"));
            }
            let all_valid = items_valid[items.clone()].iter().all(|&v| v);
            if !(items.is_empty() || (all_valid && ty == item_ty)) {
                return mismatch(kind);
            }
        }
        Ok(())
    }

    /// Every non-null number is finite: JSON has no NaN or infinity.
    fn check_finite(&self) -> Result<()> {
        let finite = match self {
            AttrColumn::F64 { valid, values } => values
                .iter()
                .zip(valid)
                .all(|(v, &ok)| !ok || v.is_finite()),
            AttrColumn::F64Set {
                valid,
                offsets,
                items_valid,
                values,
            } => (0..valid.len()).filter(|&row| valid[row]).all(|row| {
                run_range(offsets, row).all(|i| !items_valid[i] || values[i].is_finite())
            }),
            _ => true,
        };
        if finite {
            Ok(())
        } else {
            Err(ItzmaError::InvalidAttrs("attribute numbers must be finite"))
        }
    }

    /// A set column's offsets and item validity.
    fn set_parts(&self) -> Option<(&[u32], &[bool])> {
        match self {
            AttrColumn::I64Set {
                offsets,
                items_valid,
                ..
            }
            | AttrColumn::F64Set {
                offsets,
                items_valid,
                ..
            }
            | AttrColumn::TextSet {
                offsets,
                items_valid,
                ..
            } => Some((offsets, items_valid)),
            _ => None,
        }
    }

    /// The same kind of column, `null` on each of `rows` rows.
    fn empty_like(&self, rows: usize) -> AttrColumn {
        let valid = vec![false; rows];
        let offsets = vec![0u32; rows + 1];
        match self {
            AttrColumn::Null { .. } => AttrColumn::Null { rows },
            AttrColumn::Bool { .. } => AttrColumn::Bool {
                valid,
                values: vec![false; rows],
            },
            AttrColumn::I64 { .. } => AttrColumn::I64 {
                valid,
                values: vec![0; rows],
            },
            AttrColumn::F64 { .. } => AttrColumn::F64 {
                valid,
                values: vec![0.0; rows],
            },
            AttrColumn::Text { .. } => AttrColumn::Text {
                valid,
                bytes: Vec::new(),
                offsets,
            },
            AttrColumn::I64Set { .. } => AttrColumn::I64Set {
                valid,
                offsets,
                items_valid: Vec::new(),
                values: Vec::new(),
            },
            AttrColumn::F64Set { .. } => AttrColumn::F64Set {
                valid,
                offsets,
                items_valid: Vec::new(),
                values: Vec::new(),
            },
            AttrColumn::TextSet { .. } => AttrColumn::TextSet {
                valid,
                offsets,
                items_valid: Vec::new(),
                bytes: Vec::new(),
                byte_offsets: vec![0],
            },
        }
    }

    /// Add `other`'s rows after these. A `Null` column on either side takes
    /// the other's kind, its rows `null`.
    fn append(&mut self, other: AttrColumn) -> Result<()> {
        let other = match (&*self, other) {
            (AttrColumn::Null { rows }, AttrColumn::Null { rows: more }) => {
                *self = AttrColumn::Null { rows: rows + more };
                return Ok(());
            }
            (AttrColumn::Null { rows }, other) => {
                *self = other.empty_like(*rows);
                other
            }
            (_, AttrColumn::Null { rows }) => self.empty_like(rows),
            (_, other) => other,
        };
        match (self, other) {
            (
                AttrColumn::Bool { valid, values },
                AttrColumn::Bool {
                    valid: v,
                    values: x,
                },
            ) => {
                valid.extend(v);
                values.extend(x);
            }
            (
                AttrColumn::I64 { valid, values },
                AttrColumn::I64 {
                    valid: v,
                    values: x,
                },
            ) => {
                valid.extend(v);
                values.extend(x);
            }
            (
                AttrColumn::F64 { valid, values },
                AttrColumn::F64 {
                    valid: v,
                    values: x,
                },
            ) => {
                valid.extend(v);
                values.extend(x);
            }
            (
                AttrColumn::Text {
                    valid,
                    bytes,
                    offsets,
                },
                AttrColumn::Text {
                    valid: v,
                    bytes: b,
                    offsets: o,
                },
            ) => {
                append_offsets(offsets, &o)?;
                valid.extend(v);
                bytes.extend(b);
            }
            (
                AttrColumn::I64Set {
                    valid,
                    offsets,
                    items_valid,
                    values,
                },
                AttrColumn::I64Set {
                    valid: v,
                    offsets: o,
                    items_valid: iv,
                    values: x,
                },
            ) => {
                append_offsets(offsets, &o)?;
                valid.extend(v);
                items_valid.extend(iv);
                values.extend(x);
            }
            (
                AttrColumn::F64Set {
                    valid,
                    offsets,
                    items_valid,
                    values,
                },
                AttrColumn::F64Set {
                    valid: v,
                    offsets: o,
                    items_valid: iv,
                    values: x,
                },
            ) => {
                append_offsets(offsets, &o)?;
                valid.extend(v);
                items_valid.extend(iv);
                values.extend(x);
            }
            (
                AttrColumn::TextSet {
                    valid,
                    offsets,
                    items_valid,
                    bytes,
                    byte_offsets,
                },
                AttrColumn::TextSet {
                    valid: v,
                    offsets: o,
                    items_valid: iv,
                    bytes: b,
                    byte_offsets: bo,
                },
            ) => {
                append_offsets(offsets, &o)?;
                append_offsets(byte_offsets, &bo)?;
                valid.extend(v);
                items_valid.extend(iv);
                bytes.extend(b);
            }
            _ => {
                return malformed(
                    "batches in one stream must give each attribute column the same kind: a number column cannot switch between integers and floats",
                );
            }
        }
        Ok(())
    }

    fn shrink_to_fit(&mut self) {
        match self {
            AttrColumn::Null { .. } => {}
            AttrColumn::Bool { valid, values } => {
                valid.shrink_to_fit();
                values.shrink_to_fit();
            }
            AttrColumn::I64 { valid, values } => {
                valid.shrink_to_fit();
                values.shrink_to_fit();
            }
            AttrColumn::F64 { valid, values } => {
                valid.shrink_to_fit();
                values.shrink_to_fit();
            }
            AttrColumn::Text {
                valid,
                bytes,
                offsets,
            } => {
                valid.shrink_to_fit();
                bytes.shrink_to_fit();
                offsets.shrink_to_fit();
            }
            AttrColumn::I64Set {
                valid,
                offsets,
                items_valid,
                values,
            } => {
                valid.shrink_to_fit();
                offsets.shrink_to_fit();
                items_valid.shrink_to_fit();
                values.shrink_to_fit();
            }
            AttrColumn::F64Set {
                valid,
                offsets,
                items_valid,
                values,
            } => {
                valid.shrink_to_fit();
                offsets.shrink_to_fit();
                items_valid.shrink_to_fit();
                values.shrink_to_fit();
            }
            AttrColumn::TextSet {
                valid,
                offsets,
                items_valid,
                bytes,
                byte_offsets,
            } => {
                valid.shrink_to_fit();
                offsets.shrink_to_fit();
                items_valid.shrink_to_fit();
                bytes.shrink_to_fit();
                byte_offsets.shrink_to_fit();
            }
        }
    }

    /// Split off rows `at..` and keep `..at`.
    fn split_off(&mut self, at: usize) -> AttrColumn {
        match self {
            AttrColumn::Null { rows } => {
                let tail = *rows - at;
                *rows = at;
                AttrColumn::Null { rows: tail }
            }
            AttrColumn::Bool { valid, values } => AttrColumn::Bool {
                valid: valid.split_off(at),
                values: values.split_off(at),
            },
            AttrColumn::I64 { valid, values } => AttrColumn::I64 {
                valid: valid.split_off(at),
                values: values.split_off(at),
            },
            AttrColumn::F64 { valid, values } => AttrColumn::F64 {
                valid: valid.split_off(at),
                values: values.split_off(at),
            },
            AttrColumn::Text {
                valid,
                bytes,
                offsets,
            } => {
                let (offsets, cut) = split_offsets(offsets, at);
                AttrColumn::Text {
                    valid: valid.split_off(at),
                    bytes: bytes.split_off(cut),
                    offsets,
                }
            }
            AttrColumn::I64Set {
                valid,
                offsets,
                items_valid,
                values,
            } => {
                let (offsets, cut) = split_offsets(offsets, at);
                AttrColumn::I64Set {
                    valid: valid.split_off(at),
                    offsets,
                    items_valid: items_valid.split_off(cut),
                    values: values.split_off(cut),
                }
            }
            AttrColumn::F64Set {
                valid,
                offsets,
                items_valid,
                values,
            } => {
                let (offsets, cut) = split_offsets(offsets, at);
                AttrColumn::F64Set {
                    valid: valid.split_off(at),
                    offsets,
                    items_valid: items_valid.split_off(cut),
                    values: values.split_off(cut),
                }
            }
            AttrColumn::TextSet {
                valid,
                offsets,
                items_valid,
                bytes,
                byte_offsets,
            } => {
                let (offsets, cut) = split_offsets(offsets, at);
                let (byte_offsets, byte_cut) = split_offsets(byte_offsets, cut);
                AttrColumn::TextSet {
                    valid: valid.split_off(at),
                    offsets,
                    items_valid: items_valid.split_off(cut),
                    bytes: bytes.split_off(byte_cut),
                    byte_offsets,
                }
            }
        }
    }

    /// Whether row `row` is `null`.
    pub(crate) fn is_null(&self, row: usize) -> bool {
        match self {
            AttrColumn::Null { .. } => true,
            AttrColumn::Bool { valid, .. }
            | AttrColumn::I64 { valid, .. }
            | AttrColumn::F64 { valid, .. }
            | AttrColumn::Text { valid, .. }
            | AttrColumn::I64Set { valid, .. }
            | AttrColumn::F64Set { valid, .. }
            | AttrColumn::TextSet { valid, .. } => !valid[row],
        }
    }

    /// Row `row`'s string, for a `Text` column.
    pub(crate) fn text<'a>(bytes: &'a [u8], offsets: &[u32], row: usize) -> &'a str {
        run_str(bytes, offsets, row)
    }

    /// The item range of row `row` in a set column's items.
    pub(crate) fn items(offsets: &[u32], row: usize) -> std::ops::Range<usize> {
        run_range(offsets, row)
    }
}

fn set_fits(
    valid: &[bool],
    offsets: &[u32],
    items_valid: &[bool],
    values: usize,
    rows: usize,
) -> bool {
    valid.len() == rows
        && offsets_fit(offsets, rows, items_valid.len())
        && values == items_valid.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ItzmaError;

    fn ids(list: &[&str]) -> (Vec<u8>, Vec<u32>) {
        let mut bytes = Vec::new();
        let mut offsets = vec![0];
        for id in list {
            bytes.extend_from_slice(id.as_bytes());
            offsets.push(bytes.len() as u32);
        }
        (bytes, offsets)
    }

    fn two_rows() -> DocColumns {
        let (bytes, offsets) = ids(&["a", "bc"]);
        DocColumns::new(bytes, offsets, vec![1.0, 2.0, 3.0, 4.0], 2).unwrap()
    }

    fn malformed<T: std::fmt::Debug>(result: Result<T>) {
        assert!(
            matches!(result, Err(ItzmaError::InvalidColumns(_))),
            "{result:?}"
        );
    }

    #[test]
    fn id_offsets_must_start_at_zero() {
        malformed(DocColumns::new(b"ab".to_vec(), vec![1, 2], vec![0.0], 1));
    }

    #[test]
    fn id_offsets_must_not_decrease() {
        malformed(DocColumns::new(
            b"ab".to_vec(),
            vec![0, 2, 1],
            vec![0.0, 0.0],
            1,
        ));
    }

    #[test]
    fn id_offsets_must_end_at_the_buffer() {
        malformed(DocColumns::new(b"abc".to_vec(), vec![0, 2], vec![0.0], 1));
    }

    #[test]
    fn an_id_cannot_split_a_character() {
        // "é" is two bytes.
        malformed(DocColumns::new(
            "é".as_bytes().to_vec(),
            vec![0, 1, 2],
            vec![0.0, 0.0],
            1,
        ));
    }

    #[test]
    fn vectors_must_be_rows_times_dim() {
        let (bytes, offsets) = ids(&["a", "b"]);
        malformed(DocColumns::new(bytes, offsets, vec![1.0, 2.0, 3.0], 2));
    }

    #[test]
    fn a_column_must_mark_every_row() {
        malformed(two_rows().with_attr(
            "price",
            AttrColumn::F64 {
                valid: vec![true],
                values: vec![1.0, 2.0],
            },
        ));
        malformed(two_rows().with_attr("n", AttrColumn::Null { rows: 3 }));
    }

    #[test]
    fn text_must_be_utf8() {
        malformed(two_rows().with_attr(
            "name",
            AttrColumn::Text {
                valid: vec![true, true],
                bytes: vec![0xff, b'a'],
                offsets: vec![0, 1, 2],
            },
        ));
    }

    #[test]
    fn set_items_must_line_up() {
        malformed(two_rows().with_attr(
            "tags",
            AttrColumn::I64Set {
                valid: vec![true, true],
                offsets: vec![0, 1, 3],
                items_valid: vec![true, true],
                values: vec![1, 2, 3],
            },
        ));
    }

    #[test]
    fn names_are_not_empty_and_not_repeated() {
        let price = || AttrColumn::Null { rows: 2 };
        malformed(two_rows().with_attr("", price()));
        malformed(
            two_rows()
                .with_attr("price", price())
                .unwrap()
                .with_attr("price", price()),
        );
    }

    #[test]
    fn a_good_batch_round_trips_its_accessors() {
        let cols = two_rows()
            .with_attr(
                "z",
                AttrColumn::F64 {
                    valid: vec![true, false],
                    values: vec![1.5, 0.0],
                },
            )
            .unwrap()
            .with_attr("a", AttrColumn::Null { rows: 2 })
            .unwrap();
        assert_eq!((cols.rows(), cols.dim()), (2, 2));
        assert_eq!((cols.id(0), cols.id(1)), ("a", "bc"));
        assert_eq!(cols.vector(1), &[3.0, 4.0]);
        let names: Vec<&str> = cols.attrs().iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["a", "z"]);
    }
}
