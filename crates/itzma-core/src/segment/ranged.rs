//! Ranged reads of a version 2 or 3 segment.
//!
//! A whole read decodes the object into a [`Layer`](crate::layer::Layer). This
//! reads only what a question needs: the ids of a few rows, their vectors, their
//! attributes, or whether an id is in the segment at all. Opening costs one
//! suffix GET, which normally holds the trailer and the whole directory, and one
//! GET of the 32-byte header, sent together. After that a call fetches only
//! the blocks or rows it touches.
//!
//! Every fetch charges its length to the [`ReadBudget`] before it is sent, so
//! a query over indexed segments is bounded by what it touches, not by the
//! namespace. The whole decoder in `v2.rs` checks a segment against
//! itself, including that the Bloom filter is the one its ids would produce.
//! This reader cannot afford to read everything, so it checks the header, the
//! trailer, the directory and the layout, and each block it reads on its own.
//! It trusts the Bloom filter only as far as it parses. A damaged filter can hide
//! an id, so `inspect` and a whole decode remain the checks for that.
//!
//! It can also be scanned block by block, for the filter query.
//!
//! Only version 2 and 3 objects can be read this way. A version 3 object's
//! attribute blocks are typed columns, held as [`AttrRows`] and
//! rebuilt into JSON values only for the rows a caller asks for. Its zone
//! section is not read here.

use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use std::sync::{Arc, Mutex as StdMutex, OnceLock, PoisonError};

use bytes::Bytes;
use futures::lock::Mutex;
use object_store::path::Path;
use object_store::{GetOptions, GetRange, ObjectStore, ObjectStoreExt};

use super::attr_rows::{AttrRows, Stored};
use super::v2::{
    AttrLayout, BLOCK_ROWS, Directory, END_MAGIC, HEADER_LEN, TRAILER_LEN, VERSION, corrupt,
    read_block,
};
use super::{MAGIC, Matrix, Reader, missing, segment_path};
use crate::bloom::Bloom;
use crate::budget::ReadBudget;
use crate::decoded::{DecodedRef, What};
use crate::error::{ItzmaError, Result};
use crate::fetch::{self, FetchPolicy};

/// How many bytes the opening suffix GET asks for. The directory of a segment
/// of about 200 thousand rows fits, so opening is one round trip for most.
const TAIL_GUESS: u64 = 64 * 1024;

/// What a segment holds for one id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Held {
    /// The id is a document of the segment.
    Doc,
    /// The id is a tombstone of the segment.
    Tombstone,
    /// Neither.
    No,
}

/// The rows of one block, shared between the cache and the callers.
type Block = Arc<Vec<Vec<u8>>>;

/// What opening a version 2 or 3 segment learns from its header, trailer and
/// directory: all a reader needs before it fetches any block, and what the
/// decoded cache keeps.
#[derive(Clone)]
pub(crate) struct Meta {
    doc_count: u64,
    tombstone_count: u64,
    data_size: u64,
    directory: Arc<Directory>,
}

/// A section read as blocks of row bytes and kept by block: attribute blocks
/// go through `AttrRows` instead, under their own key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Section {
    Id,
    Tombstone,
}

impl Section {
    /// The decoded cache's key for block `block` of this section.
    fn key(self, block: usize) -> What {
        match self {
            Self::Id => What::IdBlock(block),
            Self::Tombstone => What::TombstoneBlock(block),
        }
    }
}

impl std::fmt::Display for Section {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Id => "id",
            Self::Tombstone => "tombstone",
        })
    }
}

/// What a parsed block keeps alive, for the decoded cache.
fn block_weight(rows: &[Vec<u8>]) -> u64 {
    rows.iter()
        .map(|row| row.len() as u64 + std::mem::size_of::<Vec<u8>>() as u64)
        .sum()
}

/// An open version 2 or 3 segment, read in pieces.
pub(crate) struct RangedSegment<'a> {
    store: &'a dyn ObjectStore,
    budget: &'a ReadBudget,
    policy: &'a FetchPolicy,
    /// Where the segment's parsed structures are kept between queries, if the
    /// handle has a decoded cache. Then a block or the Bloom filter comes from
    /// there, and the per-segment `blocks` map is not used.
    decoded: Option<&'a DecodedRef>,
    path: Path,
    dim: usize,
    doc_count: u64,
    tombstone_count: u64,
    /// The size of the whole object, which an index is built against.
    data_size: u64,
    directory: Arc<Directory>,
    /// Fetched on the first `holds`, since a query over the newest segment
    /// never asks.
    bloom: OnceLock<Arc<Bloom>>,
    /// Lets one caller fetch the filter while the others wait for it, so a
    /// burst of `holds` calls costs one request and not one each.
    loading: Mutex<()>,
    /// The id and attribute blocks read so far, kept for as long as the segment
    /// is open. A query asks a source again for more rows when shadowing or a
    /// filter leaves it short, and the blocks the earlier rounds read are then
    /// served from here instead of being fetched and charged again. Bounded by
    /// the read budget, which every fetch is charged to.
    blocks: StdMutex<HashMap<(Section, usize), Block>>,
    /// The attribute blocks read so far, kept like `blocks`.
    attr_blocks: StdMutex<HashMap<usize, Arc<AttrRows>>>,
}

/// One charged `GET` of `range`. An object the store does not have is
/// [`missing`], like every other read of a segment.
async fn get(
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

/// The last `n` bytes, with where they start and how big the object is. The
/// charge is `n`, an upper bound, since the object may be smaller.
async fn get_suffix(
    store: &dyn ObjectStore,
    budget: &ReadBudget,
    path: &Path,
    n: u64,
) -> Result<(Bytes, u64, u64)> {
    budget.charge(path.as_ref(), n)?;
    let options = GetOptions {
        range: Some(GetRange::Suffix(n)),
        ..GetOptions::default()
    };
    let got = match store.get_opts(path, options).await {
        Ok(got) => got,
        Err(object_store::Error::NotFound { .. }) => return Err(missing(path)),
        Err(e) => return Err(e.into()),
    };
    let (start, size) = (got.range.start, got.meta.size);
    Ok((got.bytes().await?, start, size))
}

fn decode_id(name: &str, bytes: Vec<u8>) -> Result<String> {
    String::from_utf8(bytes).map_err(|_| corrupt(name, "id is not valid UTF-8"))
}

pub(crate) fn decode_attrs(name: &str, bytes: &[u8]) -> Result<Option<serde_json::Value>> {
    if bytes.is_empty() {
        return Ok(None);
    }
    serde_json::from_slice(bytes)
        .map(Some)
        .map_err(|e| corrupt(name, format!("attributes are not valid JSON: {e}")))
}

fn blocks(rows: u64) -> usize {
    usize::try_from(rows.div_ceil(BLOCK_ROWS as u64)).unwrap_or(usize::MAX)
}

fn first_block_after(firsts: &[String], after: &str) -> usize {
    firsts
        .partition_point(|first| first.as_str() <= after)
        .saturating_sub(1)
}

impl<'a> RangedSegment<'a> {
    /// Open the segment `id` of `namespace` for vectors of length `dim`.
    ///
    /// # Errors
    ///
    /// [`ItzmaError::CorruptSegment`] if the object is missing (see
    /// [`is_missing`](super::is_missing)), is not version 2 or 3, has another
    /// dimension, or has a header, trailer or directory that disagree with
    /// each other or with its size. [`ItzmaError::ReadBudgetExceeded`] before
    /// the fetch that would go over. [`ItzmaError::Storage`] for a store failure.
    pub(crate) async fn open(
        store: &'a dyn ObjectStore,
        namespace: &str,
        id: &str,
        dim: usize,
        budget: &'a ReadBudget,
        policy: &'a FetchPolicy,
        decoded: Option<&'a DecodedRef>,
    ) -> Result<Self> {
        Self::open_guessing(
            store, namespace, id, dim, budget, policy, decoded, TAIL_GUESS,
        )
        .await
    }

    /// [`open`](Self::open) with the suffix guess named, so a test can make it
    /// too short.
    #[allow(clippy::too_many_arguments)] // the decoded cache joined the handle's read settings
    async fn open_guessing(
        store: &'a dyn ObjectStore,
        namespace: &str,
        id: &str,
        dim: usize,
        budget: &'a ReadBudget,
        policy: &'a FetchPolicy,
        decoded: Option<&'a DecodedRef>,
        guess: u64,
    ) -> Result<Self> {
        let path = segment_path(namespace, id);
        let meta = match decoded {
            Some(decoded) => {
                let load = || read_meta(store, budget, &path, dim, guess);
                Meta::clone(&*decoded.get_or_load(&path, What::SegmentMeta, load).await?)
            }
            None => read_meta(store, budget, &path, dim, guess).await?.0,
        };
        Ok(Self {
            store,
            budget,
            policy,
            decoded,
            path,
            dim,
            doc_count: meta.doc_count,
            tombstone_count: meta.tombstone_count,
            data_size: meta.data_size,
            directory: meta.directory,
            bloom: OnceLock::new(),
            loading: Mutex::new(()),
            blocks: StdMutex::new(HashMap::new()),
            attr_blocks: StdMutex::new(HashMap::new()),
        })
    }
}

/// Read and check a segment's header, trailer and directory, with the weight
/// of what was parsed for the decoded cache.
async fn read_meta(
    store: &dyn ObjectStore,
    budget: &ReadBudget,
    path: &Path,
    dim: usize,
    guess: u64,
) -> Result<(Meta, u64)> {
    {
        let name = path.to_string();
        let (head, (tail, tail_start, size)) = futures::try_join!(
            get(store, budget, path, 0..HEADER_LEN as u64),
            get_suffix(store, budget, path, guess),
        )?;

        // The version is read before the size is judged, so an object of a
        // version this build does not read is reported as that, however short.
        if head.len() >= 8 && head[..4] == MAGIC {
            let version = u32::from_le_bytes([head[4], head[5], head[6], head[7]]);
            if !matches!(version, VERSION | 3) {
                return Err(ItzmaError::UnsupportedFormat { version });
            }
        }
        if size < (HEADER_LEN + TRAILER_LEN) as u64 {
            return Err(corrupt(
                &name,
                "object is shorter than a header and a trailer",
            ));
        }
        let mut r = Reader::new(&name, &head);
        if r.take(4)? != MAGIC {
            return Err(corrupt(&name, "bad magic"));
        }
        let version = match r.u32()? {
            v @ (VERSION | 3) => v,
            version => return Err(ItzmaError::UnsupportedFormat { version }),
        };
        let seg_dim = r.u32()?;
        if usize::try_from(seg_dim).ok() != Some(dim) {
            return Err(corrupt(
                &name,
                format!("dimension {seg_dim}, namespace has {dim}"),
            ));
        }
        let doc_count = r.u64()?;
        let tombstone_count = r.u64()?;
        let block_rows = r.u32()?;
        if block_rows as usize != BLOCK_ROWS {
            return Err(corrupt(
                &name,
                format!("block_rows is {block_rows}, expected {BLOCK_ROWS}"),
            ));
        }
        // The same bound the whole decoder uses: counts the object cannot
        // hold are refused before anything is sized from them.
        let per_doc = (dim as u64).saturating_mul(4).saturating_add(8);
        let needed = doc_count
            .saturating_mul(per_doc)
            .saturating_add(tombstone_count.saturating_mul(4));
        if needed > size {
            return Err(corrupt(&name, "counts exceed the object size"));
        }

        // The suffix must be the end of the object, and hold the trailer.
        if tail_start.checked_add(tail.len() as u64) != Some(size) || tail.len() < TRAILER_LEN {
            return Err(corrupt(&name, "the end of the object could not be read"));
        }
        let body_end = size - TRAILER_LEN as u64;
        let mut trailer = Reader::new(&name, &tail[tail.len() - TRAILER_LEN..]);
        let directory_offset = trailer.u64()?;
        let directory_len = trailer.u64()?;
        if trailer.take(4)? != END_MAGIC {
            return Err(corrupt(&name, "bad trailer magic"));
        }
        if directory_offset.checked_add(directory_len) != Some(body_end) {
            return Err(corrupt(
                &name,
                "directory does not end where the trailer starts",
            ));
        }
        let directory_bytes = if directory_offset >= tail_start {
            let from = (directory_offset - tail_start) as usize;
            let to = (body_end - tail_start) as usize;
            tail.slice(from..to)
        } else {
            get(store, budget, path, directory_offset..body_end).await?
        };
        let directory =
            Directory::parse(&name, &directory_bytes, version, doc_count, tombstone_count)?;
        let matrix_len = doc_count
            .checked_mul(dim as u64 * 4)
            .ok_or_else(|| corrupt(&name, "counts exceed the object size"))?;
        directory.check_layout(&name, matrix_len, directory_offset)?;

        let weight = directory_bytes.len() as u64;
        Ok((
            Meta {
                doc_count,
                tombstone_count,
                data_size: size,
                directory: Arc::new(directory),
            },
            weight,
        ))
    }
}

impl<'a> RangedSegment<'a> {
    /// How many documents the segment holds.
    pub(crate) fn doc_count(&self) -> u64 {
        self.doc_count
    }

    /// The size of the whole data object in bytes.
    pub(crate) fn data_size(&self) -> u64 {
        self.data_size
    }

    fn name(&self) -> String {
        self.path.to_string()
    }

    /// `fetch::execute` has no domain-specific "missing" error of its own
    /// (it is shared by every reader in this crate, each with a different
    /// one), so a `NotFound` it surfaces as a plain `ItzmaError::Storage`
    /// is translated here into this module's own `missing`, the same way
    /// `get`'s direct `object_store::Error::NotFound` match already is —
    /// otherwise a segment deleted between `open` and a later `vectors_of`/
    /// `rows_of` call stops being recognized by `is_stale_read`, and a
    /// `gc` race that should retry against a reloaded manifest instead
    /// surfaces as a raw storage error.
    fn translate_missing(&self, err: ItzmaError) -> ItzmaError {
        match err {
            ItzmaError::Storage(object_store::Error::NotFound { .. }) => missing(&self.path),
            other => other,
        }
    }

    async fn get(&self, range: Range<u64>) -> Result<Bytes> {
        get(self.store, self.budget, &self.path, range).await
    }

    /// The rows of one block of a section, as owned bytes.
    async fn block(
        &self,
        offsets: &[u64],
        total: u64,
        what: Section,
        block: usize,
    ) -> Result<Vec<Vec<u8>>> {
        let range = offsets.get(block).zip(offsets.get(block + 1));
        let Some((&start, &end)) = range else {
            return Err(corrupt(
                &self.name(),
                format!("{what} block {block} does not exist"),
            ));
        };
        let bytes = self.get(start..end).await?;
        let expected = usize::try_from(total)
            .unwrap_or(usize::MAX)
            .saturating_sub(block * BLOCK_ROWS)
            .min(BLOCK_ROWS);
        let mut rows = Vec::with_capacity(expected);
        read_block(
            &self.name(),
            &bytes,
            expected,
            &format!("{what} block {block}"),
            &mut rows,
        )?;
        Ok(rows.into_iter().map(<[u8]>::to_vec).collect())
    }

    /// One block of a section, from a cache if an earlier call read it.
    async fn cached_block(
        &self,
        offsets: &[u64],
        total: u64,
        what: Section,
        block: usize,
    ) -> Result<Block> {
        self.blocks_of(offsets, total, what, &[block])
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| corrupt(&self.name(), format!("{what} block {block} is missing")))
    }

    fn held_block(&self, key: (Section, usize)) -> Option<Block> {
        self.blocks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&key)
            .cloned()
    }

    /// Blocks `wanted` of a section, in that order: from the decoded cache
    /// when the handle has one, else from this segment's own map.
    /// The rest are fetched together through the planner, so adjacent ones can
    /// share a request under a merging policy.
    async fn blocks_of(
        &self,
        offsets: &[u64],
        total: u64,
        what: Section,
        wanted: &[usize],
    ) -> Result<Vec<Block>> {
        if let Some(decoded) = self.decoded {
            let whats: Vec<What> = wanted.iter().map(|&b| what.key(b)).collect();
            return decoded
                .get_or_load_many(&self.path, &whats, |positions: Vec<usize>| {
                    let picked: Vec<usize> = positions.iter().map(|&at| wanted[at]).collect();
                    async move {
                        let fetched = self.fetch_blocks(offsets, total, what, &picked).await?;
                        Ok(fetched
                            .into_iter()
                            .map(|rows| {
                                let weight = block_weight(&rows);
                                (rows, weight)
                            })
                            .collect())
                    }
                })
                .await;
        }
        let missing: Vec<usize> = wanted
            .iter()
            .copied()
            .filter(|&b| self.held_block((what, b)).is_none())
            .collect();
        if !missing.is_empty() {
            let fetched = self.fetch_blocks(offsets, total, what, &missing).await?;
            let mut held = self.blocks.lock().unwrap_or_else(PoisonError::into_inner);
            for (&block, rows) in missing.iter().zip(fetched) {
                held.insert((what, block), Arc::new(rows));
            }
        }
        wanted
            .iter()
            .map(|&b| {
                self.held_block((what, b))
                    .ok_or_else(|| corrupt(&self.name(), format!("{what} block {b} is missing")))
            })
            .collect()
    }

    /// Fetch and parse blocks `wanted` of a section, in that order, through
    /// the planner. Each is charged before it is sent.
    async fn fetch_blocks(
        &self,
        offsets: &[u64],
        total: u64,
        what: Section,
        wanted: &[usize],
    ) -> Result<Vec<Vec<Vec<u8>>>> {
        let wanted_ranges: Vec<Range<u64>> = wanted
            .iter()
            .map(|&b| {
                offsets
                    .get(b)
                    .zip(offsets.get(b + 1))
                    .map(|(&s, &e)| s..e)
                    .ok_or_else(|| {
                        corrupt(&self.name(), format!("{what} block {b} does not exist"))
                    })
            })
            .collect::<Result<_>>()?;
        let planned = fetch::plan(self.policy, &wanted_ranges, self.data_size);
        let slices = fetch::execute(self.store, self.budget, &self.path, planned)
            .await
            .map_err(|e| self.translate_missing(e))?;
        wanted
            .iter()
            .zip(&slices)
            .map(|(&block, bytes)| {
                let expected = usize::try_from(total)
                    .unwrap_or(usize::MAX)
                    .saturating_sub(block * BLOCK_ROWS)
                    .min(BLOCK_ROWS);
                let mut parsed = Vec::with_capacity(expected);
                read_block(
                    &self.name(),
                    bytes,
                    expected,
                    &format!("{what} block {block}"),
                    &mut parsed,
                )?;
                Ok(parsed.into_iter().map(<[u8]>::to_vec).collect())
            })
            .collect()
    }

    /// The listed rows of a block section, in the order asked, fetching each
    /// block once, and not at all if an earlier call already did.
    async fn rows_of(&self, offsets: &[u64], what: Section, rows: &[u32]) -> Result<Vec<Vec<u8>>> {
        self.check_rows(rows)?;
        let mut wanted_blocks: Vec<usize> = rows.iter().map(|r| *r as usize / BLOCK_ROWS).collect();
        wanted_blocks.sort_unstable();
        wanted_blocks.dedup();
        let blocks = self
            .blocks_of(offsets, self.doc_count, what, &wanted_blocks)
            .await?;
        let fetched: BTreeMap<usize, Block> = wanted_blocks.into_iter().zip(blocks).collect();
        rows.iter()
            .map(|row| {
                let row = *row as usize;
                fetched
                    .get(&(row / BLOCK_ROWS))
                    .and_then(|block| block.get(row % BLOCK_ROWS))
                    .cloned()
                    .ok_or_else(|| corrupt(&self.name(), format!("{what} row {row} is missing")))
            })
            .collect()
    }

    fn check_rows(&self, rows: &[u32]) -> Result<()> {
        match rows.iter().find(|row| u64::from(**row) >= self.doc_count) {
            Some(row) => Err(corrupt(
                &self.name(),
                format!("row {row} is out of range for {} rows", self.doc_count),
            )),
            None => Ok(()),
        }
    }

    /// The ids of the listed rows, in the order asked.
    pub(crate) async fn ids_of(&self, rows: &[u32]) -> Result<Vec<String>> {
        self.rows_of(&self.directory.id_offsets, Section::Id, rows)
            .await?
            .into_iter()
            .map(|bytes| decode_id(&self.name(), bytes))
            .collect()
    }

    /// The attributes of the listed rows, in the order asked. `None` for a row
    /// that has none.
    pub(crate) async fn attrs_of(&self, rows: &[u32]) -> Result<Vec<Option<serde_json::Value>>> {
        self.check_rows(rows)?;
        let mut wanted: Vec<usize> = rows.iter().map(|r| *r as usize / BLOCK_ROWS).collect();
        wanted.sort_unstable();
        wanted.dedup();
        let blocks = self.attr_blocks_of(&wanted, true).await?;
        let fetched: BTreeMap<usize, Arc<AttrRows>> = wanted.into_iter().zip(blocks).collect();
        let name = self.name();
        rows.iter()
            .map(|row| {
                let row = *row as usize;
                let block = fetched
                    .get(&(row / BLOCK_ROWS))
                    .filter(|b| row % BLOCK_ROWS < b.len())
                    .ok_or_else(|| corrupt(&name, format!("attribute row {row} is missing")))?;
                block.value(&name, row % BLOCK_ROWS)
            })
            .collect()
    }

    /// Attribute blocks `wanted`, in that order: from the decoded cache when
    /// the handle has one, else from this segment's own map when
    /// `keep`, else fetched and not kept (a scan holds one block at a time).
    async fn attr_blocks_of(&self, wanted: &[usize], keep: bool) -> Result<Vec<Arc<AttrRows>>> {
        if let Some(decoded) = self.decoded {
            let whats: Vec<What> = wanted.iter().map(|&b| What::AttrBlock(b)).collect();
            return decoded
                .get_or_load_many(&self.path, &whats, |positions: Vec<usize>| {
                    let picked: Vec<usize> = positions.iter().map(|&at| wanted[at]).collect();
                    async move {
                        let fetched = self.fetch_attr_blocks(&picked, true).await?;
                        Ok(fetched
                            .into_iter()
                            .map(|rows| {
                                let weight = rows.weight();
                                (rows, weight)
                            })
                            .collect())
                    }
                })
                .await;
        }
        if !keep {
            let fetched = self.fetch_attr_blocks(wanted, false).await?;
            return Ok(fetched.into_iter().map(Arc::new).collect());
        }
        let held = |b: usize| {
            self.attr_blocks
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .get(&b)
                .cloned()
        };
        let missing: Vec<usize> = wanted
            .iter()
            .copied()
            .filter(|&b| held(b).is_none())
            .collect();
        if !missing.is_empty() {
            let fetched = self.fetch_attr_blocks(&missing, true).await?;
            let mut map = self
                .attr_blocks
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            for (&block, rows) in missing.iter().zip(fetched) {
                map.insert(block, Arc::new(rows));
            }
        }
        wanted
            .iter()
            .map(|&b| {
                held(b)
                    .ok_or_else(|| corrupt(&self.name(), format!("attribute block {b} is missing")))
            })
            .collect()
    }

    /// Fetch and parse attribute blocks `wanted`, in that order, through the
    /// planner, each charged before it is sent. With `copy`, a JSON block is
    /// copied out of the response, so a kept block does not hold a larger
    /// merged response alive.
    async fn fetch_attr_blocks(&self, wanted: &[usize], copy: bool) -> Result<Vec<AttrRows>> {
        let offsets = &self.directory.attr_offsets;
        let ranges: Vec<Range<u64>> = wanted
            .iter()
            .map(|&b| {
                offsets
                    .get(b)
                    .zip(offsets.get(b + 1))
                    .map(|(&s, &e)| s..e)
                    .ok_or_else(|| {
                        corrupt(&self.name(), format!("attribute block {b} does not exist"))
                    })
            })
            .collect::<Result<_>>()?;
        let fetched = self.fetch_planned(&ranges).await?;
        let name = self.name();
        wanted
            .iter()
            .zip(fetched)
            .map(|(&block, bytes)| {
                // Only JSON rows keep the fetched bytes; columns are decoded
                // into owned vectors, so copying them would be waste.
                let json = matches!(self.directory.attrs, AttrLayout::Json);
                let bytes = if copy && json {
                    Bytes::copy_from_slice(&bytes)
                } else {
                    bytes
                };
                AttrRows::parse(
                    &name,
                    &self.directory.attrs,
                    bytes,
                    Self::rows_in(self.doc_count, block) as usize,
                    &format!("attribute block {block}"),
                )
            })
            .collect()
    }

    /// How many id and attribute blocks the segment has.
    pub(crate) fn doc_block_count(&self) -> usize {
        blocks(self.doc_count)
    }

    /// How many tombstone blocks the segment has.
    pub(crate) fn tombstone_block_count(&self) -> usize {
        blocks(self.tombstone_count)
    }

    /// The first id block that can hold an id above `after`. Blocks before it
    /// are all at or below `after`.
    pub(crate) fn first_doc_block_after(&self, after: &str) -> usize {
        first_block_after(&self.directory.id_firsts, after)
    }

    /// The same for the tombstone blocks.
    pub(crate) fn first_tombstone_block_after(&self, after: &str) -> usize {
        first_block_after(&self.directory.tombstone_firsts, after)
    }

    /// The ids and attributes of the documents of one block, in id order. Two
    /// requests, sent together. The vector matrix is not touched.
    ///
    /// # Errors
    ///
    /// [`ItzmaError::CorruptSegment`] for a block that does not exist or a
    /// block whose ids or attributes do not decode.
    pub(crate) async fn doc_block(
        &self,
        block: usize,
    ) -> Result<Vec<(String, Option<serde_json::Value>)>> {
        let d = &self.directory;
        let ids_fetch = async {
            if self.decoded.is_some() {
                let ids = self
                    .cached_block(&d.id_offsets, self.doc_count, Section::Id, block)
                    .await?;
                Ok(Vec::clone(&ids))
            } else {
                // Without a decoded cache a scan does not keep the blocks it
                // passes, so a filter over a whole namespace holds one at a
                // time.
                self.block(&d.id_offsets, self.doc_count, Section::Id, block)
                    .await
            }
        };
        let wanted = [block];
        let (ids, attrs) = futures::try_join!(ids_fetch, self.attr_blocks_of(&wanted, false))?;
        let attrs = attrs
            .into_iter()
            .next()
            .ok_or_else(|| corrupt(&self.name(), format!("attribute block {block} is missing")))?;
        if ids.len() != attrs.len() {
            return Err(corrupt(
                &self.name(),
                "an id block and its attribute block differ in length",
            ));
        }
        let name = self.name();
        ids.into_iter()
            .enumerate()
            .map(|(i, id)| Ok((decode_id(&name, id)?, attrs.value(&name, i)?)))
            .collect()
    }

    /// The ids of one tombstone block, in order.
    ///
    /// # Errors
    ///
    /// [`ItzmaError::CorruptSegment`] as for [`doc_block`](Self::doc_block).
    pub(crate) async fn tombstone_block(&self, block: usize) -> Result<Vec<String>> {
        let d = &self.directory;
        let name = self.name();
        let rows = if self.decoded.is_some() {
            let held = self
                .cached_block(
                    &d.tombstone_offsets,
                    self.tombstone_count,
                    Section::Tombstone,
                    block,
                )
                .await?;
            Vec::clone(&held)
        } else {
            self.block(
                &d.tombstone_offsets,
                self.tombstone_count,
                Section::Tombstone,
                block,
            )
            .await?
        };
        rows.into_iter()
            .map(|bytes| decode_id(&name, bytes))
            .collect()
    }

    /// A [`ItzmaError::CorruptSegment`] for this segment's object.
    pub(crate) fn corrupt(&self, reason: &str) -> ItzmaError {
        corrupt(&self.name(), reason)
    }

    /// The exact vectors of the listed rows, in the order asked, as one
    /// dense matrix. Nearby rows share a request under the handle's
    /// [`FetchPolicy`], through the shared fetch planner.
    ///
    /// The rerank's path: its rows are few and are its answer, so they are
    /// fetched whole and decoded once, with none of
    /// [`vectors_into`](Self::vectors_into)'s per-chunk bookkeeping.
    pub(crate) async fn vectors_of(&self, rows: &[u32]) -> Result<Matrix> {
        self.check_rows(rows)?;
        if rows.is_empty() {
            return Ok(Matrix {
                rows: 0,
                dim: self.dim,
                values: Vec::new(),
            });
        }
        let row_len = self.dim * 4;
        let planned = fetch::plan(self.policy, &self.vector_ranges(rows), self.data_size);
        let slices = fetch::execute(self.store, self.budget, &self.path, planned)
            .await
            .map_err(|e| self.translate_missing(e))?;
        let mut values = Vec::with_capacity(rows.len() * self.dim);
        for slice in &slices {
            let raw = slice
                .get(..row_len)
                .ok_or_else(|| corrupt(&self.name(), "a vector row is truncated"))?;
            values.extend(
                raw.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
            );
        }
        Ok(Matrix {
            rows: rows.len(),
            dim: self.dim,
            values,
        })
    }

    /// The byte range of each of `rows`' vectors, in the order given.
    fn vector_ranges(&self, rows: &[u32]) -> Vec<Range<u64>> {
        let row_len = self.dim as u64 * 4;
        rows.iter()
            .map(|&row| {
                let start = self.directory.vectors_offset + u64::from(row) * row_len;
                start..start + row_len
            })
            .collect()
    }

    /// Whether the segment holds `id` as a document, as a tombstone, or not at
    /// all. The Bloom filter answers "not at all" without a request. On a maybe
    /// it reads the one id block that could hold the id, and then the one
    /// tombstone block, if the id was not a document.
    pub(crate) async fn holds(&self, id: &str) -> Result<Held> {
        if !self.bloom().await?.might_contain(id.as_bytes()) {
            return Ok(Held::No);
        }
        let d = &self.directory;
        if self
            .contains(&d.id_offsets, &d.id_firsts, self.doc_count, Section::Id, id)
            .await?
        {
            return Ok(Held::Doc);
        }
        if self
            .contains(
                &d.tombstone_offsets,
                &d.tombstone_firsts,
                self.tombstone_count,
                Section::Tombstone,
                id,
            )
            .await?
        {
            return Ok(Held::Tombstone);
        }
        Ok(Held::No)
    }

    /// Whether the block of `firsts` that could hold `id` does.
    async fn contains(
        &self,
        offsets: &[u64],
        firsts: &[String],
        total: u64,
        what: Section,
        id: &str,
    ) -> Result<bool> {
        let block = match firsts.partition_point(|first| first.as_str() <= id) {
            0 => return Ok(false),
            n => n - 1,
        };
        let rows = self.cached_block(offsets, total, what, block).await?;
        Ok(rows
            .binary_search_by(|row| row.as_slice().cmp(id.as_bytes()))
            .is_ok())
    }

    async fn bloom(&self) -> Result<&Bloom> {
        if let Some(bloom) = self.bloom.get() {
            return Ok(bloom);
        }
        let _loading = self.loading.lock().await;
        if let Some(bloom) = self.bloom.get() {
            return Ok(bloom);
        }
        let d = &self.directory;
        let load = || async {
            let bytes = self
                .get(d.bloom_offset..d.bloom_offset.saturating_add(d.bloom_len))
                .await?;
            let bloom = Bloom::from_bytes(&bytes)
                .ok_or_else(|| corrupt(&self.name(), "the Bloom filter is malformed"))?;
            Ok((bloom, bytes.len() as u64))
        };
        let bloom = match self.decoded {
            Some(decoded) => decoded.get_or_load(&self.path, What::Bloom, load).await?,
            None => Arc::new(load().await?.0),
        };
        Ok(self.bloom.get_or_init(|| bloom).as_ref())
    }
}

/// The documents of a run of id blocks, read forward by a streaming merge
/// and never kept by the segment.
#[derive(Debug)]
pub(crate) struct DocWindow {
    pub(crate) ids: Vec<String>,
    /// Each document's attributes as stored, views of the fetched blocks.
    pub(crate) attrs: WindowAttrs,
    pub(crate) vectors: Matrix,
}

/// The attributes of a window's rows: its attribute blocks as read, one per
/// id block, JSON rows as views of the fetched bytes or decoded columns.
#[derive(Debug, Default)]
pub(crate) struct WindowAttrs {
    blocks: Vec<AttrRows>,
}

impl WindowAttrs {
    /// Row `row` of the window as stored. A window starts at a block edge,
    /// so the row's block is `row / BLOCK_ROWS`.
    ///
    /// # Panics
    ///
    /// If `row` is past the window.
    pub(crate) fn get(&self, row: usize) -> Stored {
        self.blocks[row / BLOCK_ROWS].stored(row % BLOCK_ROWS)
    }

    /// Stored bytes behind the window's attributes, for cost estimates.
    pub(crate) fn stored_bytes(&self) -> usize {
        self.blocks.iter().map(AttrRows::stored_bytes).sum()
    }
}

/// What [`RangedSegment::window_sections`] fetched.
struct WindowSections {
    rows: usize,
    ids: Vec<Bytes>,
    attrs: Vec<Bytes>,
    vectors: Option<Bytes>,
}

/// The bytes block `block` spans in a section's offset list, 0 past its end.
fn span(offsets: &[u64], block: usize) -> u64 {
    match (offsets.get(block), offsets.get(block + 1)) {
        (Some(start), Some(end)) => end.saturating_sub(*start),
        _ => 0,
    }
}

/// Strictly ascending, as every section of a segment is.
fn check_ascending(name: &str, ids: &[String], what: &str) -> Result<()> {
    if ids.windows(2).all(|pair| pair[0] < pair[1]) {
        Ok(())
    } else {
        Err(corrupt(
            name,
            format!("{what} are not in strictly ascending order"),
        ))
    }
}

impl RangedSegment<'_> {
    /// Rows in block `block` of a section of `total` rows.
    fn rows_in(total: u64, block: usize) -> u64 {
        total
            .saturating_sub(block as u64 * BLOCK_ROWS as u64)
            .min(BLOCK_ROWS as u64)
    }

    /// Stored bytes of id block `block`, its attribute block and its vector
    /// rows: what [`doc_window`](Self::doc_window) reads for it.
    pub(crate) fn doc_window_bytes(&self, block: usize) -> u64 {
        let d = &self.directory;
        span(&d.id_offsets, block)
            + span(&d.attr_offsets, block)
            + Self::rows_in(self.doc_count, block) * self.dim as u64 * 4
    }

    /// Stored bytes of tombstone block `block`.
    pub(crate) fn tombstone_window_bytes(&self, block: usize) -> u64 {
        span(&self.directory.tombstone_offsets, block)
    }

    /// Id blocks `blocks`, their attribute blocks and their vector rows,
    /// fetched together through the planner and charged to the budget. Unlike
    /// every other read here nothing is kept, by the segment or the decoded
    /// cache, so a merge that reads a segment forward holds one window of it.
    ///
    /// # Errors
    ///
    /// [`ItzmaError::CorruptSegment`] for a block that does not exist, does
    /// not parse, or whose ids are out of order, and [`missing`] for an object
    /// that is gone.
    pub(crate) async fn doc_window(&self, blocks: Range<usize>) -> Result<DocWindow> {
        let (ids, attrs, vectors) = self.window(blocks, true).await?;
        let vectors = vectors.unwrap_or(Matrix {
            rows: 0,
            dim: self.dim,
            values: Vec::new(),
        });
        Ok(DocWindow {
            ids,
            attrs,
            vectors,
        })
    }

    /// The fetched bytes of id blocks `blocks`, their attribute blocks and,
    /// when `vectors` is set, their vector rows: one slice per block, and one
    /// slice for the rows.
    async fn window_sections(
        &self,
        blocks: &Range<usize>,
        vectors: bool,
    ) -> Result<WindowSections> {
        if blocks.end > self.doc_block_count() {
            return Err(corrupt(
                &self.name(),
                format!("id block {} does not exist", blocks.end - 1),
            ));
        }
        let first_row = blocks.start as u64 * BLOCK_ROWS as u64;
        let end_row = (blocks.end as u64 * BLOCK_ROWS as u64).min(self.doc_count);
        let d = &self.directory;
        let row_len = self.dim as u64 * 4;
        let range_of = |offsets: &[u64], block: usize| offsets[block]..offsets[block + 1];
        let ids_wanted: Vec<Range<u64>> =
            blocks.clone().map(|b| range_of(&d.id_offsets, b)).collect();
        let attrs_wanted: Vec<Range<u64>> = blocks
            .clone()
            .map(|b| range_of(&d.attr_offsets, b))
            .collect();
        let vectors_start = d.vectors_offset + first_row * row_len;
        let vector_range = vectors_start..d.vectors_offset + end_row * row_len;
        let vector_fetch = async {
            if vectors {
                self.fetch_planned(std::slice::from_ref(&vector_range))
                    .await
                    .map(Some)
            } else {
                Ok(None)
            }
        };
        // One plan per section: planned together, the planner could read
        // across the gap between two sections, which is the rest of the
        // vector matrix and can be megabytes.
        let (ids, attrs, vectors) = futures::try_join!(
            self.fetch_planned(&ids_wanted),
            self.fetch_planned(&attrs_wanted),
            vector_fetch,
        )?;
        Ok(WindowSections {
            rows: (end_row - first_row) as usize,
            ids,
            attrs,
            vectors: vectors.map(|mut v| v.swap_remove(0)),
        })
    }

    /// Id blocks `blocks` and their attribute blocks, and their vector rows
    /// when `vectors` is set, each row its own `String` and `Bytes`.
    async fn window(
        &self,
        blocks: Range<usize>,
        vectors: bool,
    ) -> Result<(Vec<String>, WindowAttrs, Option<Matrix>)> {
        if blocks.is_empty() {
            return Ok((Vec::new(), WindowAttrs::default(), None));
        }
        let name = self.name();
        let sections = self.window_sections(&blocks, vectors).await?;
        let mut ids = Vec::with_capacity(sections.rows);
        let mut attrs = WindowAttrs {
            blocks: Vec::with_capacity(blocks.len()),
        };
        for (i, block) in blocks.enumerate() {
            let expected = Self::rows_in(self.doc_count, block) as usize;
            let mut rows = Vec::with_capacity(expected);
            read_block(
                &name,
                &sections.ids[i],
                expected,
                &format!("id block {block}"),
                &mut rows,
            )?;
            for row in rows {
                ids.push(decode_id(&name, row.to_vec())?);
            }
            attrs.blocks.push(AttrRows::parse(
                &name,
                &self.directory.attrs,
                sections.attrs[i].clone(),
                expected,
                &format!("attribute block {block}"),
            )?);
        }
        check_ascending(&name, &ids, "ids")?;
        let Some(raw) = sections.vectors else {
            return Ok((ids, attrs, None));
        };
        let row_len = self.dim * 4;
        if raw.len() != sections.rows * row_len {
            return Err(corrupt(&name, "a vector row is truncated"));
        }
        let values = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let matrix = Matrix {
            rows: ids.len(),
            dim: self.dim,
            values,
        };
        Ok((ids, attrs, Some(matrix)))
    }

    /// `wanted`, through the planner, charged, in the order given.
    async fn fetch_planned(&self, wanted: &[Range<u64>]) -> Result<Vec<Bytes>> {
        let planned = fetch::plan(self.policy, wanted, self.data_size);
        fetch::execute(self.store, self.budget, &self.path, planned)
            .await
            .map_err(|e| self.translate_missing(e))
    }

    /// Tombstone blocks `blocks`, in order, read and charged like
    /// [`doc_window`](Self::doc_window) and likewise never kept.
    ///
    /// # Errors
    ///
    /// As for [`doc_window`](Self::doc_window).
    pub(crate) async fn tombstone_window(&self, blocks: Range<usize>) -> Result<Vec<String>> {
        if blocks.is_empty() {
            return Ok(Vec::new());
        }
        let wanted: Vec<usize> = blocks.collect();
        let fetched = self
            .fetch_blocks(
                &self.directory.tombstone_offsets,
                self.tombstone_count,
                Section::Tombstone,
                &wanted,
            )
            .await?;
        let name = self.name();
        let ids = fetched
            .into_iter()
            .flatten()
            .map(|row| decode_id(&name, row))
            .collect::<Result<Vec<_>>>()?;
        check_ascending(&name, &ids, "tombstones")?;
        Ok(ids)
    }

    /// Blocks this segment keeps for later calls.
    #[cfg(test)]
    fn held_blocks(&self) -> usize {
        self.blocks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
            + self
                .attr_blocks
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .len()
    }
}

#[cfg(test)]
mod tests {
    use std::fmt;
    use std::sync::Mutex;

    use async_trait::async_trait;
    use futures::stream::BoxStream;
    use object_store::memory::InMemory;
    use object_store::{
        CopyOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, PutMultipartOptions,
        PutOptions, PutPayload, PutResult,
    };
    use serde_json::json;

    use super::*;
    use crate::document::{Document, WalOp};
    use crate::layer::Layer;
    use crate::segment::{encode_as, is_missing};

    /// One `get` the store served: what was asked and how many bytes came back.
    #[derive(Debug, Clone)]
    struct Served {
        range: Option<GetRange>,
        bytes: u64,
    }

    /// An in-memory store that logs every read, so a test can say how many
    /// requests a call made and how many bytes they carried.
    #[derive(Debug, Default)]
    struct CountingStore {
        inner: InMemory,
        served: Mutex<Vec<Served>>,
        /// Hand every body back in pieces of this many bytes, as a real
        /// store's response arrives, instead of in one.
        chunk: Option<usize>,
    }

    impl CountingStore {
        fn served(&self) -> Vec<Served> {
            self.served.lock().expect("log").clone()
        }

        fn reset(&self) {
            self.served.lock().expect("log").clear();
        }

        fn requests(&self) -> usize {
            self.served().len()
        }

        fn bytes(&self) -> u64 {
            self.served().iter().map(|s| s.bytes).sum()
        }
    }

    impl fmt::Display for CountingStore {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "CountingStore")
        }
    }

    #[async_trait]
    impl ObjectStore for CountingStore {
        async fn put_opts(
            &self,
            location: &Path,
            payload: PutPayload,
            opts: PutOptions,
        ) -> object_store::Result<PutResult> {
            self.inner.put_opts(location, payload, opts).await
        }

        async fn put_multipart_opts(
            &self,
            location: &Path,
            opts: PutMultipartOptions,
        ) -> object_store::Result<Box<dyn MultipartUpload>> {
            self.inner.put_multipart_opts(location, opts).await
        }

        async fn get_opts(
            &self,
            location: &Path,
            options: GetOptions,
        ) -> object_store::Result<GetResult> {
            let range = options.range.clone();
            let got = self.inner.get_opts(location, options).await?;
            self.served.lock().expect("log").push(Served {
                range,
                bytes: got.range.end - got.range.start,
            });
            let Some(chunk) = self.chunk else {
                return Ok(got);
            };
            let (meta, range, attributes, extensions) = (
                got.meta.clone(),
                got.range.clone(),
                got.attributes.clone(),
                got.extensions.clone(),
            );
            let body = got.bytes().await?;
            let pieces: Vec<object_store::Result<Bytes>> = (0..body.len())
                .step_by(chunk)
                .map(|at| Ok(body.slice(at..(at + chunk).min(body.len()))))
                .collect();
            Ok(GetResult {
                payload: object_store::GetResultPayload::Stream(Box::pin(futures::stream::iter(
                    pieces,
                ))),
                meta,
                range,
                attributes,
                extensions,
            })
        }

        fn delete_stream(
            &self,
            locations: BoxStream<'static, object_store::Result<Path>>,
        ) -> BoxStream<'static, object_store::Result<Path>> {
            self.inner.delete_stream(locations)
        }

        fn list(
            &self,
            prefix: Option<&Path>,
        ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
            self.inner.list(prefix)
        }

        async fn list_with_delimiter(
            &self,
            prefix: Option<&Path>,
        ) -> object_store::Result<ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }

        async fn copy_opts(
            &self,
            from: &Path,
            to: &Path,
            options: CopyOptions,
        ) -> object_store::Result<()> {
            self.inner.copy_opts(from, to, options).await
        }
    }

    const NS: &str = "ns";
    const ID: &str = "00000000000000000001-00000000000000000009";

    fn layer_of(docs: usize, tombstones: usize, dim: usize) -> Layer {
        let mut layer = Layer::new();
        for i in 0..docs {
            layer.apply(WalOp::Upsert(Document {
                id: format!("d{i:05}"),
                vector: (0..dim).map(|k| (i * 7 + k) as f32 * 0.5).collect(),
                attrs: (i % 3 == 0).then(|| json!({ "n": i, "tag": "x" })),
            }));
        }
        for i in 0..tombstones {
            layer.apply(WalOp::Delete(format!("t{i:05}")));
        }
        layer
    }

    /// A store holding one version 2 segment, and its bytes.
    async fn setup(docs: usize, tombstones: usize, dim: usize) -> (CountingStore, Bytes) {
        let bytes = encode_as(&layer_of(docs, tombstones, dim), dim, 2).expect("encode");
        put(bytes.clone()).await
    }

    async fn put(bytes: Bytes) -> (CountingStore, Bytes) {
        let store = CountingStore::default();
        store
            .put(&segment_path(NS, ID), bytes.clone().into())
            .await
            .expect("put");
        (store, bytes)
    }

    async fn open<'a>(
        store: &'a CountingStore,
        dim: usize,
        budget: &'a ReadBudget,
        policy: &'a FetchPolicy,
    ) -> Result<RangedSegment<'a>> {
        RangedSegment::open(store, NS, ID, dim, budget, policy, None).await
    }

    fn reason(err: &ItzmaError) -> String {
        match err {
            ItzmaError::CorruptSegment { reason, .. } => reason.clone(),
            other => format!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn every_call_agrees_with_the_whole_decoder() {
        let dim = 3;
        let counts = [
            (0, 0),
            (1, 0),
            (0, 1),
            (255, 255),
            (256, 256),
            (257, 257),
            (300, 1),
            (513, 600),
        ];
        for (docs, tombstones) in counts {
            let (store, bytes) = setup(docs, tombstones, dim).await;
            let layer = crate::segment::decode("p", &bytes, dim).expect("decode");
            let budget = ReadBudget::unlimited();
            let policy = FetchPolicy::local();
            let seg = open(&store, dim, &budget, &policy).await.expect("open");
            assert_eq!(seg.doc_count(), docs as u64);

            let all: Vec<u32> = (0..docs as u32).collect();
            let want = layer.sorted_docs();
            let ids: Vec<&str> = want.iter().map(|d| d.id.as_str()).collect();
            assert_eq!(seg.ids_of(&all).await.expect("ids"), ids, "{docs} docs");
            let vectors: Vec<&[f32]> = want.iter().map(|d| d.vector.as_slice()).collect();
            let got = seg.vectors_of(&all).await.expect("vectors");
            let got_rows: Vec<&[f32]> = (0..got.rows).map(|i| got.row(i)).collect();
            assert_eq!(got_rows, vectors, "{docs} docs");
            let attrs: Vec<&Option<serde_json::Value>> = want.iter().map(|d| &d.attrs).collect();
            let got = seg.attrs_of(&all).await.expect("attrs");
            assert_eq!(got.iter().collect::<Vec<_>>(), attrs, "{docs} docs");

            // Any order, and a row twice.
            let mixed: Vec<u32> = all
                .iter()
                .rev()
                .chain(all.iter().take(2))
                .copied()
                .collect();
            let want_mixed: Vec<&str> = mixed.iter().map(|r| ids[*r as usize]).collect();
            assert_eq!(seg.ids_of(&mixed).await.expect("ids"), want_mixed);

            for doc in &want {
                assert_eq!(seg.holds(&doc.id).await.expect("holds"), Held::Doc);
            }
            for dead in layer.sorted_tombstones() {
                assert_eq!(seg.holds(dead).await.expect("holds"), Held::Tombstone);
            }
            for i in 0..50 {
                let absent = format!("zz-absent-{i}");
                assert_eq!(seg.holds(&absent).await.expect("holds"), Held::No);
            }
        }
    }

    #[tokio::test]
    async fn an_empty_request_makes_no_request() {
        let (store, _) = setup(10, 0, 2).await;
        let budget = ReadBudget::unlimited();
        let policy = FetchPolicy::local();
        let seg = open(&store, 2, &budget, &policy).await.expect("open");
        store.reset();
        assert!(seg.ids_of(&[]).await.expect("ids").is_empty());
        assert_eq!(seg.vectors_of(&[]).await.expect("vectors").rows, 0);
        assert_eq!(store.requests(), 0);
    }

    #[tokio::test]
    async fn opening_is_two_requests_and_the_suffix_holds_a_small_directory() {
        let (store, bytes) = setup(600, 10, 4).await;
        let budget = ReadBudget::unlimited();
        let policy = FetchPolicy::local();
        open(&store, 4, &budget, &policy).await.expect("open");
        // The header, and the tail, which here is the whole object.
        assert_eq!(store.requests(), 2);
        assert_eq!(store.bytes(), HEADER_LEN as u64 + bytes.len() as u64);
    }

    #[tokio::test]
    async fn a_short_guess_is_followed_by_the_exact_directory() {
        let (store, bytes) = setup(600, 10, 4).await;
        let directory_len = u64::from_le_bytes(
            bytes[bytes.len() - TRAILER_LEN + 8..bytes.len() - TRAILER_LEN + 16]
                .try_into()
                .expect("length"),
        );
        let budget = ReadBudget::unlimited();
        let policy = FetchPolicy::local();
        let seg = RangedSegment::open_guessing(&store, NS, ID, 4, &budget, &policy, None, 64)
            .await
            .expect("open");
        let served = store.served();
        assert_eq!(served.len(), 3, "header, short tail, exact directory");
        assert_eq!(served[2].bytes, directory_len);
        assert!(matches!(served[2].range, Some(GetRange::Bounded(_))));
        assert_eq!(seg.ids_of(&[599]).await.expect("ids"), ["d00599"]);
    }

    #[tokio::test]
    async fn ids_and_attributes_fetch_each_block_once() {
        let (store, _) = setup(1000, 0, 2).await;
        let budget = ReadBudget::unlimited();
        let policy = FetchPolicy::local();
        let seg = open(&store, 2, &budget, &policy).await.expect("open");
        store.reset();
        seg.ids_of(&[0, 1, 2, 300, 301, 2]).await.expect("ids");
        // Blocks 0 and 1 are adjacent (zero gap between them, same section),
        // so the planner merges them into one request even under
        // the tight local() policy — a strict improvement over fetching
        // each wanted block on its own regardless of how close it is to
        // its neighbor, which is what this test used to pin.
        assert_eq!(store.requests(), 1, "blocks 0 and 1 merge, being adjacent");
        store.reset();
        seg.attrs_of(&[0, 3, 700, 999]).await.expect("attrs");
        // Block 1's own bytes sit between blocks 0 and 2, but this
        // segment's attributes are small enough (most rows have none) that
        // even that whole gap is still within local()'s min_gap, so all
        // four wanted rows' blocks merge into one request.
        assert_eq!(store.requests(), 1, "blocks 0, 2 and 3 all merge");
    }

    #[tokio::test]
    async fn adjacent_blocks_share_a_request_under_a_merging_policy() {
        let (store, _) = setup(2000, 0, 2).await; // 8 id blocks of 256 rows each
        let budget = ReadBudget::unlimited();
        // A policy whose gap is wide enough to span the gap between two
        // neighboring blocks' byte ranges, even though they are not the
        // same block and a tight policy fetches each separately.
        let wide = FetchPolicy {
            min_gap: 1024 * 1024,
            max_gap: 1024 * 1024,
            ..FetchPolicy::local()
        };
        let seg = open(&store, 2, &budget, &wide).await.expect("open");
        store.reset();
        // Blocks 0 and 1 (rows 0 and 300 land in different id blocks).
        seg.ids_of(&[0, 300]).await.expect("ids");
        assert_eq!(store.requests(), 1, "both id blocks fetched in one request");
    }

    /// The widening loop of a query asks a segment again for more rows, and
    /// the blocks it already has must not be fetched or charged twice.
    #[tokio::test]
    async fn a_block_read_before_is_neither_fetched_nor_charged_again() {
        let (store, _) = setup(1000, 0, 2).await;
        let budget = ReadBudget::unlimited();
        let policy = FetchPolicy::local();
        let seg = open(&store, 2, &budget, &policy).await.expect("open");
        seg.ids_of(&[0, 300]).await.expect("ids");
        seg.attrs_of(&[0, 300]).await.expect("attrs");

        store.reset();
        let charged = budget.used();
        let ids = seg.ids_of(&[1, 301, 0]).await.expect("ids");
        let attrs = seg.attrs_of(&[2, 302]).await.expect("attrs");
        assert_eq!((ids.len(), attrs.len()), (3, 2));
        assert_eq!(store.requests(), 0, "both blocks of each section are held");
        assert_eq!(budget.used(), charged, "and nothing more is charged");

        seg.ids_of(&[700]).await.expect("ids");
        assert_eq!(store.requests(), 1, "a block not read before is fetched");
    }

    /// The shadow check asks `holds` once per candidate, and every candidate in
    /// a block would otherwise fetch that block again.
    #[tokio::test]
    async fn holds_reads_a_block_once_however_many_ids_it_is_asked_about() {
        let (store, _) = setup(1000, 3, 2).await;
        let budget = ReadBudget::unlimited();
        let policy = FetchPolicy::local();
        let seg = open(&store, 2, &budget, &policy).await.expect("open");
        assert_eq!(seg.holds("d00001").await.expect("holds"), Held::Doc);
        store.reset();
        for n in [2, 3, 200, 255] {
            let id = format!("d{n:05}");
            assert_eq!(seg.holds(&id).await.expect("holds"), Held::Doc, "{id}");
        }
        assert_eq!(store.requests(), 0, "block 0 is held, and so is the filter");
    }

    #[tokio::test]
    async fn near_rows_share_a_request_and_far_rows_do_not() {
        let dim = 4;
        let row = dim as u64 * 4;
        let (store, _) = setup(1000, 0, dim).await;
        let budget = ReadBudget::unlimited();
        let policy = FetchPolicy::local();
        let seg = open(&store, dim, &budget, &policy).await.expect("open");
        let cases: [(&[u32], usize, u64); 6] = [
            (&[10, 11, 12], 1, 3 * row),
            (&[10, 900], 2, 2 * row),
            (&[10, 12], 1, 3 * row),
            (&[10, 10], 1, row),
            // Exactly the coalescing distance apart, then one row further.
            (
                &[10, 10 + 1 + (policy.min_gap / row) as u32],
                1,
                (2 + policy.min_gap / row) * row,
            ),
            (&[10, 11 + 1 + (policy.min_gap / row) as u32], 2, 2 * row),
        ];
        for (rows, requests, bytes) in cases {
            store.reset();
            seg.vectors_of(rows).await.expect("vectors");
            assert_eq!(store.requests(), requests, "{rows:?}");
            assert_eq!(store.bytes(), bytes, "{rows:?}");
        }
    }

    #[tokio::test]
    async fn vectors_come_back_in_the_order_asked() {
        let dim = 4;
        let (store, bytes) = setup(1000, 0, dim).await;
        let layer = crate::segment::decode("p", &bytes, dim).expect("decode");
        let docs = layer.sorted_docs();
        let budget = ReadBudget::unlimited();
        let policy = FetchPolicy::local();
        let seg = open(&store, dim, &budget, &policy).await.expect("open");
        let got = seg.vectors_of(&[900, 10, 11, 900]).await.expect("vectors");
        let want = [900usize, 10, 11, 900].map(|r| docs[r].vector.clone());
        let got_rows: Vec<&[f32]> = (0..got.rows).map(|i| got.row(i)).collect();
        let want_rows: Vec<&[f32]> = want.iter().map(|v| v.as_slice()).collect();
        assert_eq!(got_rows, want_rows);
    }

    /// Ids the segment's filter rejects, found through a second handle so the
    /// handle a test measures has not fetched its filter yet.
    async fn rejected_ids(store: &CountingStore, dim: usize, want: usize) -> Vec<String> {
        let budget = ReadBudget::unlimited();
        let policy = FetchPolicy::local();
        let other = open(store, dim, &budget, &policy).await.expect("open");
        let bloom = other.bloom().await.expect("bloom");
        (0..)
            .map(|i| format!("q-{i}"))
            .filter(|id| !bloom.might_contain(id.as_bytes()))
            .take(want)
            .collect()
    }

    #[tokio::test]
    async fn the_filter_answers_no_without_a_request() {
        let (store, _) = setup(1000, 100, 2).await;
        let absent = rejected_ids(&store, 2, 2).await;
        let budget = ReadBudget::unlimited();
        let policy = FetchPolicy::local();
        let seg = open(&store, 2, &budget, &policy).await.expect("open");

        // The first question fetches the filter, and nothing else.
        store.reset();
        assert_eq!(seg.holds(&absent[0]).await.expect("holds"), Held::No);
        assert_eq!(store.requests(), 1);
        // The next one is answered from memory.
        store.reset();
        assert_eq!(seg.holds(&absent[1]).await.expect("holds"), Held::No);
        assert_eq!(store.requests(), 0);

        // A document costs one block, a tombstone the id block and its own.
        assert_eq!(seg.holds("d00500").await.expect("holds"), Held::Doc);
        assert_eq!(store.requests(), 1);
        store.reset();
        assert_eq!(seg.holds("t00050").await.expect("holds"), Held::Tombstone);
        assert_eq!(store.requests(), 2);
    }

    #[tokio::test]
    async fn the_filter_is_not_fetched_by_reading_rows() {
        let (store, _) = setup(1000, 0, 2).await;
        let budget = ReadBudget::unlimited();
        let policy = FetchPolicy::local();
        let seg = open(&store, 2, &budget, &policy).await.expect("open");
        store.reset();
        seg.ids_of(&[5]).await.expect("ids");
        seg.vectors_of(&[5]).await.expect("vectors");
        seg.attrs_of(&[5]).await.expect("attrs");
        assert_eq!(store.requests(), 3, "one block, one row, one block");
    }

    /// A fetch is charged before it is sent, and at its length.
    #[tokio::test]
    async fn a_read_that_would_go_over_the_budget_is_never_sent() {
        let dim = 4;
        let row = dim as u64 * 4;
        let guess = 4096u64;
        let (store, _) = setup(1000, 0, dim).await;

        let policy = FetchPolicy::local();
        let tiny = ReadBudget::new(10);
        let err = open(&store, dim, &tiny, &policy)
            .await
            .err()
            .expect("refused");
        assert!(
            matches!(err, ItzmaError::ReadBudgetExceeded { .. }),
            "{err:?}"
        );
        assert_eq!(store.requests(), 0);

        // Room for the open and exactly two vector rows.
        let budget = ReadBudget::new(HEADER_LEN as u64 + guess + 2 * row);
        let seg = RangedSegment::open_guessing(&store, NS, ID, dim, &budget, &policy, None, guess)
            .await
            .expect("open");
        seg.vectors_of(&[0]).await.expect("first row");
        seg.vectors_of(&[500]).await.expect("second row");
        let before = store.requests();
        let err = seg.vectors_of(&[900]).await.expect_err("refused");
        assert!(
            matches!(err, ItzmaError::ReadBudgetExceeded { .. }),
            "{err:?}"
        );
        assert_eq!(store.requests(), before, "nothing was sent");
    }

    /// Two far-apart rows under a policy wide enough to merge them cost far
    /// more bytes merged (the whole span between them) than fetched
    /// separately. A budget between those two totals must still succeed:
    /// the planner's own fallback falls back to the unmerged
    /// plan rather than fail a read merging would have refused.
    #[tokio::test]
    async fn merging_that_would_exceed_the_budget_falls_back_to_the_unmerged_plan() {
        let dim = 4;
        let row = dim as u64 * 4;
        let (store, _) = setup(1000, 0, dim).await;

        // Learn the opening cost under an unlimited budget first.
        let unlimited = ReadBudget::unlimited();
        let local = FetchPolicy::local();
        open(&store, dim, &unlimited, &local).await.expect("open");
        let open_cost = unlimited.used();

        // Wide enough to merge rows 0 and 900 (and everything between) into
        // one request; merging all of it costs the whole 900-row span,
        // unmerged costs just the two rows themselves.
        let wide = FetchPolicy {
            min_gap: 1 << 20,
            max_gap: 1 << 20,
            ..FetchPolicy::local()
        };
        let merged_cost = 900 * row + row; // the whole 0..=900 span, inclusive
        let unmerged_cost = 2 * row;
        assert!(
            unmerged_cost < merged_cost,
            "the two totals must actually differ for this test to mean anything"
        );

        let tight = ReadBudget::new(open_cost + unmerged_cost);
        let seg = RangedSegment::open(&store, NS, ID, dim, &tight, &wide, None)
            .await
            .expect("open");
        seg.vectors_of(&[0, 900])
            .await
            .expect("the fallback must let this fit");

        // One byte short of even the unmerged floor genuinely refuses.
        let too_tight = ReadBudget::new(open_cost + unmerged_cost - 1);
        let seg = RangedSegment::open(&store, NS, ID, dim, &too_tight, &wide, None)
            .await
            .expect("open");
        let err = seg.vectors_of(&[0, 900]).await.expect_err("refused");
        assert!(
            matches!(err, ItzmaError::ReadBudgetExceeded { .. }),
            "{err:?}"
        );
    }

    /// A segment that existed when `open` ran but is gone by the time a
    /// later call reads it (a merge plus `gc` raced ahead) must still
    /// report `is_missing`, the same way `open` itself does — that's what
    /// `Namespace`'s stale-read retry (`is_stale_read`) keys on. Covers
    /// every reader that goes through the planner: `vectors_of` (a single
    /// request) and `ids_of` (through `rows_of`'s block fetch).
    #[tokio::test]
    async fn a_segment_deleted_after_open_is_still_reported_as_missing_on_a_later_read() {
        let (store, _) = setup(10, 0, 2).await;
        let budget = ReadBudget::unlimited();
        let policy = FetchPolicy::local();
        let seg = open(&store, 2, &budget, &policy).await.expect("open");
        store.delete(&segment_path(NS, ID)).await.expect("delete");

        let err = seg.vectors_of(&[0]).await.expect_err("missing");
        assert!(is_missing(&err), "vectors_of: {err:?}");
        let err = seg.ids_of(&[0]).await.expect_err("missing");
        assert!(is_missing(&err), "ids_of: {err:?}");
    }

    #[tokio::test]
    async fn a_missing_segment_is_reported_as_missing() {
        let store = CountingStore::default();
        let budget = ReadBudget::unlimited();
        let policy = FetchPolicy::local();
        let err = open(&store, 2, &budget, &policy)
            .await
            .err()
            .expect("missing");
        assert!(is_missing(&err), "{err:?}");
    }

    #[tokio::test]
    async fn refusals() {
        let dim = 3;
        // A version 1 object, which this build does not read.
        let mut v1 = encode_as(&layer_of(5, 0, dim), dim, 2)
            .expect("encode")
            .to_vec();
        v1[4..8].copy_from_slice(&1u32.to_le_bytes());
        let (store, _) = put(v1.into()).await;
        let budget = ReadBudget::unlimited();
        let policy = FetchPolicy::local();
        let err = open(&store, dim, &budget, &policy).await.err().expect("v1");
        assert!(
            matches!(err, ItzmaError::UnsupportedFormat { version: 1 }),
            "{err:?}"
        );

        // Another dimension.
        let (store, bytes) = setup(5, 0, dim).await;
        let err = open(&store, dim + 1, &budget, &policy)
            .await
            .err()
            .expect("dim");
        assert!(reason(&err).contains("dimension"), "{err:?}");

        // A version this build does not know.
        let mut newer = bytes.to_vec();
        newer[4..8].copy_from_slice(&9u32.to_le_bytes());
        let (store, _) = put(Bytes::from(newer)).await;
        let err = open(&store, dim, &budget, &policy)
            .await
            .err()
            .expect("newer");
        assert!(
            matches!(err, ItzmaError::UnsupportedFormat { version: 9 }),
            "{err:?}"
        );

        // A row past the end.
        let (store, _) = setup(5, 0, dim).await;
        let seg = open(&store, dim, &budget, &policy).await.expect("open");
        for err in [
            seg.ids_of(&[5]).await.err(),
            seg.vectors_of(&[5]).await.err(),
            seg.attrs_of(&[0, 5]).await.err(),
        ] {
            assert!(reason(&err.expect("refused")).contains("out of range"));
        }
    }

    /// Every strict prefix of an object is refused.
    #[tokio::test]
    async fn a_truncated_object_is_refused() {
        let dim = 2;
        let (_, bytes) = setup(20, 3, dim).await;
        let budget = ReadBudget::unlimited();
        let policy = FetchPolicy::local();
        for len in 0..bytes.len() {
            let (store, _) = put(bytes.slice(..len)).await;
            assert!(
                open(&store, dim, &budget, &policy).await.is_err(),
                "a prefix of {len} bytes opened"
            );
        }
    }

    /// A flipped bit may go unseen (the ranged reader does not read every
    /// byte), but it never panics: the open and every call return a result.
    #[tokio::test]
    async fn a_flipped_bit_never_panics() {
        let dim = 2;
        let (_, bytes) = setup(20, 3, dim).await;
        let budget = ReadBudget::unlimited();
        let policy = FetchPolicy::local();
        let all: Vec<u32> = (0..20).collect();
        for i in 0..bytes.len() {
            for bit in [0u8, 7] {
                let mut damaged = bytes.to_vec();
                damaged[i] ^= 1 << bit;
                let (store, _) = put(Bytes::from(damaged)).await;
                let Ok(seg) = open(&store, dim, &budget, &policy).await else {
                    continue;
                };
                let _ = seg.ids_of(&all).await;
                let _ = seg.vectors_of(&all).await;
                let _ = seg.attrs_of(&all).await;
                for id in ["d00000", "d00019", "t00001", "nope"] {
                    let _ = seg.holds(id).await;
                }
            }
        }
    }

    /// The futures can be moved to another task, since queries are spawned.
    #[tokio::test]
    async fn the_futures_are_send() {
        fn send<T: Send>(_: &T) {}
        let (store, _) = setup(5, 1, 2).await;
        let budget = ReadBudget::unlimited();
        let policy = FetchPolicy::local();
        let opening = RangedSegment::open(&store, NS, ID, 2, &budget, &policy, None);
        send(&opening);
        let seg = opening.await.expect("open");
        send(&seg.ids_of(&[0]));
        send(&seg.vectors_of(&[0]));
        send(&seg.attrs_of(&[0]));
        send(&seg.holds("d00000"));
    }

    async fn stored(docs: usize, tombstones: usize, version: u32) -> (InMemory, Layer) {
        let layer = layer_of(docs, tombstones, 4);
        let store = InMemory::new();
        store
            .put(
                &segment_path(NS, ID),
                encode_as(&layer, 4, version).expect("encode").into(),
            )
            .await
            .expect("put");
        (store, layer)
    }

    #[tokio::test]
    async fn doc_and_tombstone_blocks_equal_the_whole_decode_at_every_block() {
        // 700 rows is three doc blocks (256, 256, 188), 300 tombstones two.
        let (store, layer) = stored(700, 300, 2).await;
        let budget = ReadBudget::new(u64::MAX);
        let policy = FetchPolicy::local();
        let segment = RangedSegment::open(&store, NS, ID, 4, &budget, &policy, None)
            .await
            .expect("open");
        assert_eq!(segment.doc_block_count(), 3);
        assert_eq!(segment.tombstone_block_count(), 2);

        let mut docs = Vec::new();
        for block in 0..segment.doc_block_count() {
            docs.extend(segment.doc_block(block).await.expect("doc block"));
        }
        let want: Vec<_> = layer
            .sorted_docs()
            .into_iter()
            .map(|d| (d.id.clone(), d.attrs.clone()))
            .collect();
        assert_eq!(docs, want);

        let mut tombstones = Vec::new();
        for block in 0..segment.tombstone_block_count() {
            tombstones.extend(
                segment
                    .tombstone_block(block)
                    .await
                    .expect("tombstone block"),
            );
        }
        assert_eq!(tombstones, layer.sorted_tombstones());
    }

    #[tokio::test]
    async fn a_doc_block_is_two_requests_and_a_tombstone_block_one() {
        let (store, _) = stored(700, 300, 2).await;
        let counting = CountingStore {
            inner: store,
            ..Default::default()
        };
        let budget = ReadBudget::new(u64::MAX);
        let policy = FetchPolicy::local();
        let segment = RangedSegment::open(&counting, NS, ID, 4, &budget, &policy, None)
            .await
            .expect("open");
        counting.reset();
        segment.doc_block(1).await.expect("doc block");
        assert_eq!(
            counting.requests(),
            2,
            "the id block and the attribute block"
        );
        counting.reset();
        segment.tombstone_block(1).await.expect("tombstone block");
        assert_eq!(counting.requests(), 1);
    }

    #[tokio::test]
    async fn the_first_block_after_a_cursor_never_skips_a_row() {
        let (store, layer) = stored(700, 300, 2).await;
        let budget = ReadBudget::new(u64::MAX);
        let policy = FetchPolicy::local();
        let segment = RangedSegment::open(&store, NS, ID, 4, &budget, &policy, None)
            .await
            .expect("open");
        let ids: Vec<String> = layer.sorted_docs().iter().map(|d| d.id.clone()).collect();
        // Before everything, on each block's first and last id, and past the end.
        for after in [
            "", "a", "d00000", "d00255", "d00256", "d00511", "d00699", "d99999",
        ] {
            let from = segment.first_doc_block_after(after);
            let mut got = Vec::new();
            for block in from..segment.doc_block_count() {
                got.extend(segment.doc_block(block).await.expect("block"));
            }
            let want_tail: Vec<&String> = ids.iter().filter(|id| id.as_str() > after).collect();
            let got_tail: Vec<&String> = got
                .iter()
                .map(|(id, _)| id)
                .filter(|id| id.as_str() > after)
                .collect();
            assert_eq!(got_tail, want_tail, "after {after:?}");
        }
    }

    #[tokio::test]
    async fn a_block_past_the_end_is_corrupt_segment() {
        let (store, _) = stored(10, 0, 2).await;
        let budget = ReadBudget::new(u64::MAX);
        let policy = FetchPolicy::local();
        let segment = RangedSegment::open(&store, NS, ID, 4, &budget, &policy, None)
            .await
            .expect("open");
        assert!(matches!(
            segment.doc_block(1).await,
            Err(ItzmaError::CorruptSegment { .. })
        ));
    }

    /// Every id block of `seg` read forward in windows of `per` blocks,
    /// concatenated, and every tombstone block likewise.
    async fn read_forward(
        seg: &RangedSegment<'_>,
        per: usize,
    ) -> (Vec<String>, Vec<Vec<u8>>, Vec<f32>, Vec<String>) {
        let (mut ids, mut attrs, mut vectors, mut dead) = (vec![], vec![], vec![], vec![]);
        for start in (0..seg.doc_block_count()).step_by(per) {
            let end = (start + per).min(seg.doc_block_count());
            let window = seg.doc_window(start..end).await.expect("doc window");
            assert_eq!(window.vectors.rows, window.ids.len());
            attrs.extend((0..window.ids.len()).map(|i| {
                let value = window.attrs.get(i).value("t").expect("attributes");
                crate::segment::attr_json(value.as_ref()).expect("json")
            }));
            ids.extend(window.ids);
            vectors.extend(window.vectors.values);
        }
        for start in (0..seg.tombstone_block_count()).step_by(per) {
            let end = (start + per).min(seg.tombstone_block_count());
            let window = seg.tombstone_window(start..end).await.expect("tombstones");
            dead.extend(window);
        }
        (ids, attrs, vectors, dead)
    }

    #[tokio::test]
    async fn forward_windows_cover_every_row_at_block_edges() {
        let dim = 3;
        for docs in [0, 1, 255, 256, 257, 512, 1000] {
            for tombstones in [0, 1, 256, 300] {
                let layer = layer_of(docs, tombstones, dim);
                let (store, _) = setup(docs, tombstones, dim).await;
                let budget = ReadBudget::unlimited();
                let policy = FetchPolicy::local();
                let seg = open(&store, dim, &budget, &policy).await.expect("open");
                let want_docs = layer.sorted_docs();
                for per in [1, 2, usize::MAX / 2] {
                    let (ids, attrs, vectors, dead) = read_forward(&seg, per).await;
                    let case = format!("{docs} docs, {tombstones} tombstones, {per} per window");
                    let want_ids: Vec<&str> = want_docs.iter().map(|d| d.id.as_str()).collect();
                    assert_eq!(ids, want_ids, "{case}");
                    let want_attrs: Vec<Vec<u8>> = want_docs
                        .iter()
                        .map(|d| crate::segment::attr_json(d.attrs.as_ref()).unwrap())
                        .collect();
                    assert_eq!(attrs, want_attrs, "{case}");
                    let want_vectors: Vec<f32> =
                        want_docs.iter().flat_map(|d| d.vector.clone()).collect();
                    assert_eq!(vectors, want_vectors, "{case}");
                    assert_eq!(dead, layer.sorted_tombstones(), "{case}");
                }
                assert_eq!(seg.held_blocks(), 0, "a window is never kept");
            }
        }
    }

    #[tokio::test]
    async fn a_window_is_charged_what_it_covers_and_no_more() {
        let dim = 4;
        let (store, bytes) = setup(1000, 300, dim).await;
        let budget = ReadBudget::unlimited();
        let policy = FetchPolicy::local();
        let seg = open(&store, dim, &budget, &policy).await.expect("open");
        let opened = budget.used();
        store.reset();
        let covered: u64 = (0..2).map(|b| seg.doc_window_bytes(b)).sum();
        seg.doc_window(0..2).await.expect("window");
        // Each section is planned on its own, so the planner never reads
        // across the gap between two sections: under `local()`, which does
        // not align, a window is charged exactly what it covers.
        assert_eq!(budget.used() - opened, covered);
        assert_eq!(store.requests(), 3, "one request per section");
        assert!(covered < bytes.len() as u64 / 2);
        let before = budget.used();
        let dead: u64 = (0..2).map(|b| seg.tombstone_window_bytes(b)).sum();
        seg.tombstone_window(0..2).await.expect("tombstones");
        assert_eq!(budget.used() - before, dead);
    }

    #[tokio::test]
    async fn a_window_of_a_collected_segment_is_missing() {
        let dim = 2;
        let (store, _) = setup(600, 0, dim).await;
        let budget = ReadBudget::unlimited();
        let policy = FetchPolicy::local();
        let seg = open(&store, dim, &budget, &policy).await.expect("open");
        store.delete(&segment_path(NS, ID)).await.expect("delete");
        let err = seg.doc_window(0..1).await.unwrap_err();
        assert!(is_missing(&err), "{err:?}");
        let err = seg.tombstone_window(0..0).await;
        assert!(err.is_ok(), "an empty window reads nothing");
    }

    fn typed_schema() -> crate::document::Schema {
        crate::document::Schema::new(4, crate::Metric::L2).with_attrs(
            crate::attrs::AttrSchema::new()
                .field("n", crate::attrs::AttrType::Number)
                .field("s", crate::attrs::AttrType::Text),
        )
    }

    /// `docs` documents with typed attributes (every fifth none, every
    /// seventh `{}`), as `compact` writes them at `version`.
    fn typed_layer(docs: usize, tombstones: usize) -> Layer {
        let mut layer = layer_of(docs, tombstones, 4);
        let now: Vec<Document> = layer.sorted_docs().into_iter().cloned().collect();
        for (n, doc) in now.into_iter().enumerate() {
            let attrs = match n % 35 {
                x if x % 5 == 0 => None,
                x if x % 7 == 0 => Some(json!({})),
                _ => Some(json!({"n": n, "s": format!("v{}", n % 4)})),
            };
            layer.apply(WalOp::Upsert(Document { attrs, ..doc }));
        }
        layer
    }

    async fn stored_typed(docs: usize, tombstones: usize, version: u32) -> (CountingStore, Layer) {
        let layer = typed_layer(docs, tombstones);
        let bytes =
            crate::segment::encode_with_schema(&layer, &typed_schema(), version).expect("encode");
        (put(bytes).await.0, layer)
    }

    #[tokio::test]
    async fn a_version_3_segment_agrees_with_the_whole_decoder_at_every_block_edge() {
        for docs in [1, 255, 256, 257, 513] {
            let (store, layer) = stored_typed(docs, 3, 3).await;
            let budget = ReadBudget::unlimited();
            let policy = FetchPolicy::local();
            let seg = open(&store, 4, &budget, &policy).await.expect("open");
            let want: Vec<Document> = layer.sorted_docs().into_iter().cloned().collect();
            let rows: Vec<u32> = (0..docs as u32).rev().collect();
            let attrs = seg.attrs_of(&rows).await.expect("attrs");
            for (row, got) in rows.iter().zip(attrs) {
                assert_eq!(got, want[*row as usize].attrs, "row {row} of {docs}");
            }
            for block in 0..seg.doc_block_count() {
                let got = seg.doc_block(block).await.expect("doc block");
                for (i, (id, attrs)) in got.into_iter().enumerate() {
                    let doc = &want[block * BLOCK_ROWS + i];
                    assert_eq!((id, attrs), (doc.id.clone(), doc.attrs.clone()));
                }
            }
            let window = seg
                .doc_window(0..seg.doc_block_count())
                .await
                .expect("window");
            for (i, doc) in want.iter().enumerate() {
                assert_eq!(window.attrs.get(i).value("t").expect("value"), doc.attrs);
            }
        }
    }

    #[tokio::test]
    async fn a_version_3_attribute_block_is_fetched_and_charged_once() {
        let (store, _) = stored_typed(600, 0, 3).await;
        let budget = ReadBudget::unlimited();
        let policy = FetchPolicy::local();
        let seg = open(&store, 4, &budget, &policy).await.expect("open");
        store.reset();
        seg.attrs_of(&[3, 7, 300]).await.expect("attrs");
        let first = store.requests();
        assert!(first > 0);
        seg.attrs_of(&[4, 301]).await.expect("attrs");
        assert_eq!(store.requests(), first, "blocks 0 and 1 are held");
    }

    #[tokio::test]
    async fn a_truncated_or_flipped_version_3_object_never_panics() {
        let layer = typed_layer(300, 2);
        let bytes = crate::segment::encode_with_schema(&layer, &typed_schema(), 3).expect("encode");
        let budget = ReadBudget::unlimited();
        let policy = FetchPolicy::local();
        for cut in [0, 8, 32, bytes.len() / 2, bytes.len() - 21, bytes.len() - 1] {
            let (store, _) = put(bytes.slice(..cut)).await;
            assert!(
                open(&store, 4, &budget, &policy).await.is_err(),
                "cut {cut}"
            );
        }
        for at in (0..bytes.len() * 8).step_by(7) {
            let mut flipped = bytes.to_vec();
            flipped[at / 8] ^= 1 << (at % 8);
            let (store, _) = put(Bytes::from(flipped)).await;
            if let Ok(seg) = open(&store, 4, &budget, &policy).await {
                let rows: Vec<u32> = (0..300).collect();
                let _ = seg.attrs_of(&rows).await;
                let _ = seg.doc_window(0..seg.doc_block_count()).await;
            }
        }
    }

    #[test]
    fn each_section_has_its_own_cache_key_and_name() {
        assert_eq!(Section::Id.key(3), What::IdBlock(3));
        assert_eq!(Section::Tombstone.key(3), What::TombstoneBlock(3));
        assert_eq!(
            format!("{} {}", Section::Id, Section::Tombstone),
            "id tombstone"
        );
    }
}
