//! One attribute block as a reader holds it, JSON rows (version 2, and a
//! free-form namespace at version 3) or typed columns (version 3), and one
//! row's attributes left as stored until a caller needs the value.

use std::ops::Range;
use std::sync::Arc;

use bytes::Bytes;
use serde_json::Value;

use super::columns::ColumnBlock;
use super::ranged::decode_attrs;
use super::v2::{AttrLayout, read_block};
use crate::error::Result;

/// Bytes one column row is counted as when a cost estimate needs a size.
const COLUMN_ROW_COST: usize = 32;

#[derive(Debug, Clone)]
pub(crate) enum AttrRows {
    Json {
        block: Bytes,
        spans: Vec<Range<u32>>,
    },
    Columns(Arc<ColumnBlock>),
}

/// One row's attributes as stored.
#[derive(Debug, Clone)]
pub(crate) enum Stored {
    /// JSON, empty for none, a view of its block.
    Json(Bytes),
    /// Row `.1` of a column block.
    Column(Arc<ColumnBlock>, u16),
}

impl Stored {
    /// The row's attributes.
    ///
    /// # Errors
    ///
    /// [`ItzmaError::CorruptSegment`](crate::ItzmaError::CorruptSegment) at
    /// `path` for JSON that does not parse.
    pub(crate) fn value(&self, path: &str) -> Result<Option<Value>> {
        match self {
            Self::Json(raw) => decode_attrs(path, raw),
            Self::Column(block, row) => Ok(block.row(*row as usize)),
        }
    }
}

impl AttrRows {
    /// Parse one fetched attribute block of `rows` rows. `block` is kept
    /// (JSON) as given, so a caller that caches the result passes a copy, not
    /// a view of a larger response.
    pub(crate) fn parse(
        path: &str,
        layout: &AttrLayout,
        block: Bytes,
        rows: usize,
        what: &str,
    ) -> Result<Self> {
        match layout {
            AttrLayout::Json => {
                let mut parsed = Vec::with_capacity(rows);
                read_block(path, &block, rows, what, &mut parsed)?;
                let base = block.as_ptr() as usize;
                // A block is far below 4 GiB, so its offsets fit a u32.
                let spans = parsed
                    .iter()
                    .map(|row| {
                        let start = row.as_ptr() as usize - base;
                        start as u32..(start + row.len()) as u32
                    })
                    .collect();
                Ok(Self::Json { block, spans })
            }
            AttrLayout::Columns(fields) => {
                let (columns, _) = ColumnBlock::decode(path, fields, &block, rows)?;
                Ok(Self::Columns(Arc::new(columns)))
            }
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Json { spans, .. } => spans.len(),
            Self::Columns(block) => block.rows(),
        }
    }

    /// Row `i` as stored.
    ///
    /// # Panics
    ///
    /// If `i >= self.len()`.
    pub(crate) fn stored(&self, i: usize) -> Stored {
        match self {
            Self::Json { block, spans } => {
                let span = &spans[i];
                Stored::Json(block.slice(span.start as usize..span.end as usize))
            }
            Self::Columns(block) => {
                assert!(i < block.rows(), "row {i} is past the block");
                // A block holds at most 256 rows.
                Stored::Column(block.clone(), i as u16)
            }
        }
    }

    pub(crate) fn value(&self, path: &str, i: usize) -> Result<Option<Value>> {
        self.stored(i).value(path)
    }

    /// What the block keeps alive, for the decoded cache.
    pub(crate) fn weight(&self) -> u64 {
        match self {
            Self::Json { block, spans } => (block.len() + spans.len() * 8) as u64,
            Self::Columns(block) => block.weight(),
        }
    }

    /// Stored bytes behind the rows, for cost estimates.
    pub(crate) fn stored_bytes(&self) -> usize {
        match self {
            Self::Json { block, .. } => block.len(),
            Self::Columns(block) => block.rows() * COLUMN_ROW_COST,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ItzmaError;

    #[test]
    #[should_panic(expected = "past the block")]
    fn a_column_row_past_the_block_panics_like_a_json_one() {
        let fields = crate::segment::columns::fields_of(&crate::attrs::AttrSchema::new());
        let block = ColumnBlock::from_rows(&fields, &[None, None]).unwrap();
        AttrRows::Columns(Arc::new(block)).stored(2);
    }

    #[test]
    fn a_json_row_is_none_when_empty_its_value_otherwise_and_corrupt_when_bad() {
        assert_eq!(Stored::Json(Bytes::new()).value("p").unwrap(), None);
        assert_eq!(
            Stored::Json(Bytes::from_static(br#"{"a":1}"#))
                .value("p")
                .unwrap(),
            Some(serde_json::json!({"a": 1}))
        );
        let err = Stored::Json(Bytes::from_static(b"{"))
            .value("p")
            .unwrap_err();
        assert!(
            matches!(err, ItzmaError::CorruptSegment { ref path, .. } if path == "p"),
            "{err:?}"
        );
    }
}
