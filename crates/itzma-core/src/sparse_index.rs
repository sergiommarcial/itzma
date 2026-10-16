//! The sparse inverted index: a term directory and postings
//! blocks, derived from one segment's `sparse.bin`. Any valid rebuild is
//! correct — nothing byte-compares two builds, same rule as the dense index
//! (`crate::index`).
//!
//! # Layout
//!
//! All integers are little-endian.
//!
//! | Section | Contents |
//! |---|---|
//! | Header | magic `ITZP`, `format_version` u32, `rows` u32, `term_count` u32 |
//! | Term directory | `term_count` entries, sorted by `term_id`: `term_id` u32, `doc_freq` u32, `postings_offset` u64, `postings_len` u32 |
//! | Postings | for each term, `doc_freq` pairs of `(row u32, weight f32)`, sorted by row |
//!
//! `postings_offset` is relative to the end of the term directory. A query
//! reads every matched term's postings in full; there is no skip pruning.

use std::collections::BTreeMap;
use std::ops::Range;

use bytes::{Bytes, BytesMut};
use object_store::path::Path;
use object_store::{GetOptions, GetRange, ObjectStore, ObjectStoreExt};

use crate::budget::ReadBudget;
use crate::error::{ItzmaError, Result};

const MAGIC: [u8; 4] = *b"ITZP";
pub(crate) const FORMAT_VERSION: u32 = 1;
/// Magic, format_version, rows, term_count.
const HEADER_LEN: usize = 4 + 4 + 4 + 4;
/// term_id, doc_freq, postings_offset, postings_len.
const DIR_ENTRY_LEN: usize = 4 + 4 + 8 + 4;
/// row, weight.
const POSTING_LEN: usize = 4 + 4;

const OBJECT_PREFIX: &str = "sparse-idx-";
const OBJECT_SUFFIX: &str = ".bin";

/// Object key of a segment's sparse inverted index:
/// `<namespace>/segments/<id>/sparse-idx-<generation>.bin`.
pub(crate) fn sparse_index_path(namespace: &str, id: &str, generation: u32) -> Path {
    Path::from(format!(
        "{namespace}/segments/{id}/{OBJECT_PREFIX}{generation}{OBJECT_SUFFIX}"
    ))
}

/// The generation in a `sparse-idx-<n>.bin` name, or `None` for anything
/// else (mirrors `crate::index::generation_of`, one prefix, no legacy name).
pub(crate) fn generation_of(file_name: &str) -> Option<u32> {
    let digits = file_name
        .strip_prefix(OBJECT_PREFIX)?
        .strip_suffix(OBJECT_SUFFIX)?;
    let generation: u32 = digits.parse().ok()?;
    (generation >= 1 && generation.to_string() == digits).then_some(generation)
}

fn corrupt(path: &str, reason: impl Into<String>) -> ItzmaError {
    ItzmaError::CorruptSparseIndex {
        path: path.to_string(),
        reason: reason.into(),
    }
}

const MISSING: &str = "missing from the store";

/// The error for a sparse index the store does not have.
pub(crate) fn missing(path: &Path) -> ItzmaError {
    corrupt(path.as_ref(), MISSING)
}

/// Whether `err` is what [`missing`] builds.
pub(crate) fn is_missing(err: &ItzmaError) -> bool {
    matches!(err, ItzmaError::CorruptSparseIndex { reason, .. } if reason == MISSING)
}

/// The fixed part at the front of the object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Prefix {
    pub(crate) rows: u32,
    pub(crate) term_count: u32,
}

/// One term directory entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Entry {
    pub(crate) doc_freq: u32,
    postings_offset: u64,
    postings_len: u32,
}

/// The whole term directory, sorted by term id, fetched once per query per
/// segment.
#[derive(Debug)]
pub(crate) struct TermDirectory {
    entries: Vec<(u32, Entry)>,
    /// Byte offset, from the start of the object, where postings begin —
    /// `HEADER_LEN + term_count * DIR_ENTRY_LEN`.
    postings_base: u64,
}

impl TermDirectory {
    /// About how many bytes the directory holds, for the decoded cache.
    pub(crate) fn weight(&self) -> u64 {
        (self.entries.len() * std::mem::size_of::<(u32, Entry)>()) as u64
    }

    /// The entry for `term_id`, if this segment's index has one.
    pub(crate) fn find(&self, term_id: u32) -> Option<&Entry> {
        self.entries
            .binary_search_by_key(&term_id, |(id, _)| *id)
            .ok()
            .map(|at| &self.entries[at].1)
    }

    /// Absolute byte range of `entry`'s postings within the object.
    fn range_of(&self, entry: &Entry) -> Range<u64> {
        let start = self.postings_base + entry.postings_offset;
        start..start + u64::from(entry.postings_len)
    }
}

/// A cursor over the object's bytes, bounds-checked, raising
/// `CorruptSparseIndex`.
struct Reader<'a> {
    path: &'a str,
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(path: &'a str, buf: &'a [u8]) -> Self {
        Self { path, buf, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let out = self
            .buf
            .get(self.pos..)
            .and_then(|rest| rest.get(..n))
            .ok_or_else(|| corrupt(self.path, "object is truncated"))?;
        self.pos += n;
        Ok(out)
    }

    fn u32(&mut self) -> Result<u32> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn u64(&mut self) -> Result<u64> {
        let bytes = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(bytes);
        Ok(u64::from_le_bytes(a))
    }

    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.u32()?))
    }
}

/// Build a whole `sparse-idx-<gen>.bin` object from `postings`: term id to
/// `(row, weight)` pairs, already sorted by row within each term. Used by
/// `Namespace::index_sparse`.
pub(crate) fn build(rows: u32, postings: &BTreeMap<u32, Vec<(u32, f32)>>) -> Bytes {
    let term_count = postings.len() as u32;
    let mut body = BytesMut::new();
    let mut dir = Vec::with_capacity(postings.len());
    for (&term_id, rows_for_term) in postings {
        let offset = body.len() as u64;
        for &(row, weight) in rows_for_term {
            body.extend_from_slice(&row.to_le_bytes());
            body.extend_from_slice(&weight.to_le_bytes());
        }
        dir.push((
            term_id,
            rows_for_term.len() as u32,
            offset,
            (rows_for_term.len() * POSTING_LEN) as u32,
        ));
    }

    let mut out = BytesMut::with_capacity(HEADER_LEN + dir.len() * DIR_ENTRY_LEN + body.len());
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&rows.to_le_bytes());
    out.extend_from_slice(&term_count.to_le_bytes());
    for (term_id, doc_freq, offset, len) in &dir {
        out.extend_from_slice(&term_id.to_le_bytes());
        out.extend_from_slice(&doc_freq.to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
        out.extend_from_slice(&len.to_le_bytes());
    }
    out.extend_from_slice(&body);
    out.freeze()
}

/// How many bytes the first request asks for: the prefix plus a term
/// directory of up to this many entries fits in one round trip for most
/// segments (mirrors `index::HEAD_GUESS`).
const HEAD_GUESS: u64 = 64 * 1024;

/// Fetch and check the prefix and the whole term directory of a sparse index,
/// against the segment's row count.
///
/// # Errors
///
/// [`ItzmaError::CorruptSparseIndex`] for a missing object (see
/// [`is_missing`]), bad magic, a row count that disagrees with `rows`, or a
/// directory entry that runs past the object.
/// [`ItzmaError::UnsupportedFormat`] for a container version this build does
/// not read.
pub(crate) async fn open(
    store: &dyn ObjectStore,
    budget: &ReadBudget,
    path: &Path,
    rows: u64,
) -> Result<(Prefix, TermDirectory)> {
    let name = path.as_ref();
    budget.charge(name, HEAD_GUESS)?;
    let options = GetOptions {
        range: Some(GetRange::Bounded(0..HEAD_GUESS)),
        ..GetOptions::default()
    };
    let got = match store.get_opts(path, options).await {
        Ok(got) => got,
        Err(object_store::Error::NotFound { .. }) => return Err(missing(path)),
        Err(e) => return Err(e.into()),
    };
    let size = got.meta.size;
    let first = got.bytes().await?;

    let mut r = Reader::new(name, &first);
    if r.take(4)? != MAGIC {
        return Err(corrupt(name, "bad magic"));
    }
    match r.u32()? {
        FORMAT_VERSION => {}
        version => return Err(ItzmaError::UnsupportedFormat { version }),
    }
    let got_rows = u64::from(r.u32()?);
    if got_rows != rows {
        return Err(corrupt(
            name,
            format!("{got_rows} rows, the segment has {rows}"),
        ));
    }
    let term_count = r.u32()?;
    let dir_len = term_count as u64 * DIR_ENTRY_LEN as u64;
    let dir_end = HEADER_LEN as u64 + dir_len;
    if dir_end > size {
        return Err(corrupt(
            name,
            "the term directory runs past the end of the object",
        ));
    }
    let dir_bytes = if dir_end <= first.len() as u64 {
        first.slice(HEADER_LEN..dir_end as usize)
    } else {
        let rest = get_range(store, budget, path, first.len() as u64..dir_end).await?;
        Bytes::from([&first[HEADER_LEN..], &rest[..]].concat())
    };

    let mut dr = Reader::new(name, &dir_bytes);
    let mut entries = Vec::with_capacity(term_count as usize);
    let mut last_term: Option<u32> = None;
    for _ in 0..term_count {
        let term_id = dr.u32()?;
        if let Some(last) = last_term {
            if term_id <= last {
                return Err(corrupt(
                    name,
                    "term directory is not sorted by ascending term id",
                ));
            }
        }
        last_term = Some(term_id);
        let doc_freq = dr.u32()?;
        let postings_offset = dr.u64()?;
        let postings_len = dr.u32()?;
        let end = dir_end
            .checked_add(postings_offset)
            .and_then(|s| s.checked_add(u64::from(postings_len)))
            .ok_or_else(|| corrupt(name, "a postings range overflows"))?;
        if end > size {
            return Err(corrupt(
                name,
                format!("term {term_id}'s postings run past the end of the object"),
            ));
        }
        if u64::from(doc_freq) * POSTING_LEN as u64 != u64::from(postings_len) {
            return Err(corrupt(
                name,
                format!("term {term_id}'s doc_freq disagrees with its postings length"),
            ));
        }
        entries.push((
            term_id,
            Entry {
                doc_freq,
                postings_offset,
                postings_len,
            },
        ));
    }
    Ok((
        Prefix {
            rows: got_rows as u32,
            term_count,
        },
        TermDirectory {
            entries,
            postings_base: dir_end,
        },
    ))
}

/// One charged `GET` of `range`.
async fn get_range(
    store: &dyn ObjectStore,
    budget: &ReadBudget,
    path: &Path,
    range: Range<u64>,
) -> Result<Bytes> {
    if range.end <= range.start {
        return Ok(Bytes::new());
    }
    budget.charge(path.as_ref(), range.end - range.start)?;
    match store.get_range(path, range).await {
        Ok(bytes) => Ok(bytes),
        Err(object_store::Error::NotFound { .. }) => Err(missing(path)),
        Err(e) => Err(e.into()),
    }
}

/// Fetch and decode one term's postings: `(row, weight)` pairs, sorted by
/// row, exactly `entry.doc_freq` of them.
pub(crate) async fn postings_of(
    store: &dyn ObjectStore,
    budget: &ReadBudget,
    path: &Path,
    dir: &TermDirectory,
    entry: &Entry,
) -> Result<Vec<(u32, f32)>> {
    let bytes = get_range(store, budget, path, dir.range_of(entry)).await?;
    let mut r = Reader::new(path.as_ref(), &bytes);
    let mut out = Vec::with_capacity(entry.doc_freq as usize);
    let mut last_row: Option<u32> = None;
    for _ in 0..entry.doc_freq {
        let row = r.u32()?;
        if let Some(last) = last_row {
            if row <= last {
                return Err(corrupt(
                    path.as_ref(),
                    "postings are not sorted by ascending row",
                ));
            }
        }
        last_row = Some(row);
        let weight = r.f32()?;
        out.push((row, weight));
    }
    Ok(out)
}

/// Check that `bytes` decodes as a valid sparse index for a segment of
/// `rows` rows, without a store round trip (the bytes are already in
/// memory — used when an object already sits at the put-if-absent key).
/// Checks the prefix only, the same depth as a stray generation gets before
/// anything actually queries it.
pub(crate) fn open_bytes_for_check(path: &str, bytes: &[u8], rows: u64) -> Result<()> {
    let mut r = Reader::new(path, bytes);
    if r.take(4)? != MAGIC {
        return Err(corrupt(path, "bad magic"));
    }
    match r.u32()? {
        FORMAT_VERSION => {}
        version => return Err(ItzmaError::UnsupportedFormat { version }),
    }
    let got_rows = u64::from(r.u32()?);
    if got_rows != rows {
        return Err(corrupt(
            path,
            format!("{got_rows} rows, the segment has {rows}"),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> BTreeMap<u32, Vec<(u32, f32)>> {
        let mut postings = BTreeMap::new();
        postings.insert(3, vec![(0, 0.5), (2, 0.25)]); // term 3 -> rows 0, 2
        postings.insert(9, vec![(1, 1.0)]); // term 9 -> row 1
        postings
    }

    #[tokio::test]
    async fn build_open_and_postings_round_trip() {
        let bytes = build(3, &sample());
        let store = object_store::memory::InMemory::new();
        let path = Path::from("p");
        store.put(&path, bytes.into()).await.expect("put");
        let budget = ReadBudget::unlimited();

        let (prefix, dir) = open(&store, &budget, &path, 3).await.expect("open");
        assert_eq!((prefix.rows, prefix.term_count), (3, 2));

        let entry = dir.find(3).expect("term 3 present");
        assert_eq!(
            postings_of(&store, &budget, &path, &dir, entry)
                .await
                .expect("postings"),
            vec![(0, 0.5), (2, 0.25)]
        );
        let entry = dir.find(9).expect("term 9 present");
        assert_eq!(
            postings_of(&store, &budget, &path, &dir, entry)
                .await
                .expect("postings"),
            vec![(1, 1.0)]
        );
        assert!(dir.find(404).is_none());
    }

    #[tokio::test]
    async fn open_rejects_a_row_count_mismatch() {
        let bytes = build(3, &sample());
        let store = object_store::memory::InMemory::new();
        let path = Path::from("p");
        store.put(&path, bytes.into()).await.expect("put");
        let budget = ReadBudget::unlimited();

        let err = open(&store, &budget, &path, 4).await.unwrap_err();
        assert!(matches!(err, ItzmaError::CorruptSparseIndex { .. }));
    }

    #[tokio::test]
    async fn every_truncation_and_bit_flip_is_refused_never_panics() {
        let bytes = build(3, &sample());
        let store = object_store::memory::InMemory::new();
        let budget = ReadBudget::unlimited();
        for len in 0..bytes.len() {
            let path = Path::from(format!("p{len}"));
            store
                .put(&path, bytes.slice(..len).into())
                .await
                .expect("put");
            let _ = open(&store, &budget, &path, 3).await; // must not panic
        }
        for i in 0..bytes.len() {
            let mut bad = bytes.to_vec();
            bad[i] ^= 0xFF;
            let path = Path::from(format!("b{i}"));
            store.put(&path, bad.into()).await.expect("put");
            let _ = open(&store, &budget, &path, 3).await; // must not panic
        }
    }

    #[test]
    fn only_the_exact_object_name_has_a_generation() {
        assert_eq!(generation_of("sparse-idx-1.bin"), Some(1));
        assert_eq!(generation_of("sparse-idx-42.bin"), Some(42));
        for name in [
            "sparse.bin",
            "data.bin",
            "index-1.bin",
            "sparse-idx-0.bin",
            "sparse-idx-.bin",
            "sparse-idx-1",
        ] {
            assert_eq!(generation_of(name), None, "{name}");
        }
    }

    #[test]
    fn a_sparse_index_lives_beside_its_segment() {
        assert_eq!(
            sparse_index_path("docs", "0001-0002", 3).as_ref(),
            "docs/segments/0001-0002/sparse-idx-3.bin"
        );
    }
}
