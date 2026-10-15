//! Column types the columnar flat source shares: a segment's sorted
//! ids in one buffer, and a bitmap over its rows.

use std::cmp::Ordering;

/// Ids in ascending order, in one `String`, each a slice of it. Built once per
/// segment, so a lookup is a binary search with no allocation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct IdColumn {
    text: String,
    /// Where each id ends in `text`; id `i` starts where `i - 1` ends.
    ends: Vec<u32>,
}

impl IdColumn {
    /// The ids, in the order given. Whether they are sorted is
    /// [`is_strictly_ascending`](Self::is_strictly_ascending)'s question.
    pub(crate) fn from_sorted<'a>(ids: impl IntoIterator<Item = &'a str>) -> Self {
        // The segment view refuses an id section over u32::MAX bytes before
        // it builds a column, so `push` never saturates here.
        let mut column = Self::default();
        for id in ids {
            column.push(id);
        }
        column
    }

    /// Add `id` after the last. Whether the column stays sorted is
    /// [`is_strictly_ascending`](Self::is_strictly_ascending)'s question.
    pub(crate) fn push(&mut self, id: &str) {
        self.text.push_str(id);
        // Callers build a column from one segment's section or window, which
        // is refused or bounded far below u32::MAX bytes first.
        self.ends
            .push(u32::try_from(self.text.len()).unwrap_or(u32::MAX));
    }

    pub(crate) fn len(&self) -> usize {
        self.ends.len()
    }

    /// Id `row`.
    ///
    /// # Panics
    ///
    /// If `row >= self.len()`.
    pub(crate) fn get(&self, row: usize) -> &str {
        let start = if row == 0 {
            0
        } else {
            self.ends[row - 1] as usize
        };
        &self.text[start..self.ends[row] as usize]
    }

    /// The bytes the column holds: its text and its offsets.
    pub(crate) fn heap_bytes(&self) -> usize {
        self.text.len() + self.ends.len() * 4
    }

    pub(crate) fn contains(&self, id: &str) -> bool {
        let (mut lo, mut hi) = (0, self.len());
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match self.get(mid).cmp(id) {
                Ordering::Less => lo = mid + 1,
                Ordering::Greater => hi = mid,
                Ordering::Equal => return true,
            }
        }
        false
    }

    pub(crate) fn is_strictly_ascending(&self) -> bool {
        (1..self.len()).all(|row| self.get(row - 1) < self.get(row))
    }
}

/// One bit per row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Bitmap {
    words: Vec<u64>,
    len: usize,
}

impl Bitmap {
    /// `len` bits, all clear.
    pub(crate) fn new(len: usize) -> Self {
        Self {
            words: vec![0; len.div_ceil(64)],
            len,
        }
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn set(&mut self, i: usize) {
        self.words[i / 64] |= 1 << (i % 64);
    }

    pub(crate) fn get(&self, i: usize) -> bool {
        self.words[i / 64] & (1 << (i % 64)) != 0
    }

    #[cfg(test)]
    pub(crate) fn count(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_id_column_gives_back_its_ids_and_finds_them() {
        let ids = ["a", "bb", "c\u{e9}", "zz"];
        let column = IdColumn::from_sorted(ids);
        assert_eq!(column.len(), 4);
        for (row, id) in ids.iter().enumerate() {
            assert_eq!(column.get(row), *id);
            assert!(column.contains(id));
        }
        assert!(!column.contains("b"));
        assert!(!column.contains(""));
        assert!(column.is_strictly_ascending());
        assert!(!IdColumn::from_sorted(["b", "a"]).is_strictly_ascending());
        assert!(!IdColumn::from_sorted(["a", "a"]).is_strictly_ascending());
        assert_eq!(IdColumn::from_sorted([]).len(), 0);
    }

    #[test]
    fn a_bitmap_sets_and_counts_across_words() {
        let mut bits = Bitmap::new(130);
        for i in [0, 63, 64, 129] {
            bits.set(i);
        }
        assert_eq!(bits.len(), 130);
        assert_eq!(bits.count(), 4);
        assert!(bits.get(63) && bits.get(64) && !bits.get(65));
    }
}
