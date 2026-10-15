//! The k-way merge over a namespace's sources in id order.
//!
//! The sources are the live segments, oldest to newest, and the WAL tail as one
//! more. Each yields its documents and tombstones in id order, starting after
//! the cursor. For the smallest id across the sources, the newest source that
//! holds it decides: a document is the winner, a tombstone drops the id, and
//! every older copy is skipped. So shadowing costs nothing here, since the
//! merge sees every copy of an id side by side. The fronts are kept in
//! [`NewestWins`], so the smallest id costs `log(sources)`.
//!
//! A version 2 or 3 segment is read through `RangedSegment`, one 256-row block of
//! ids and attributes (and of tombstones) at a time, fetched when the merge
//! needs it, and the vector matrix is never read. A run of version 1 segments
//! has no blocks and is decoded whole.

use std::collections::VecDeque;
use std::future::Future;
use std::sync::Arc;

use futures::future::try_join_all;
use futures::stream::{self, StreamExt, TryStreamExt};
use object_store::ObjectStore;
use serde_json::Value;

use crate::budget::ReadBudget;
use crate::decoded::DecodedRef;
use crate::error::Result;
use crate::fetch::FetchPolicy;
use crate::layer::{Layer, NewestWins};
use crate::segment::RangedSegment;

/// Requests in flight at once when the segments are opened.
const FETCH_CONCURRENCY: usize = 16;

/// What one source holds for one id.
pub(crate) enum Body {
    Doc(Option<Value>),
    Tombstone,
}

pub(crate) struct Entry {
    pub(crate) id: String,
    pub(crate) body: Body,
}

/// A live document the merge decided on: the newest copy of its id.
pub(crate) struct Winner {
    pub(crate) id: String,
    pub(crate) attrs: Option<Value>,
}

/// What a merge reads, against one manifest.
pub(crate) struct Open<'a> {
    pub(crate) store: &'a dyn ObjectStore,
    pub(crate) namespace: &'a str,
    pub(crate) dim: usize,
    pub(crate) budget: &'a ReadBudget,
    pub(crate) policy: &'a FetchPolicy,
    /// Parsed structures kept between queries, if the handle has a
    /// decoded cache.
    pub(crate) decoded: Option<&'a DecodedRef>,
    pub(crate) after: Option<&'a str>,
}

/// A source of entries in strictly ascending id order.
pub(crate) enum Source<'a> {
    /// A version 2 or 3 segment, read a block at a time.
    Segment(Box<SegmentSource<'a>>),
    /// A version 1 run or the tail, in memory.
    Layer { entries: VecDeque<Entry> },
}

/// One side of a segment (its documents, or its tombstones): the next block to
/// fetch and what is buffered from the last one.
pub(crate) struct Side {
    next_block: usize,
    blocks: usize,
    buffer: VecDeque<Entry>,
    /// The last id this side produced, to check the segment really is sorted.
    last: Option<String>,
}

impl Side {
    /// Its buffer is empty and it has a block left.
    fn ran_dry(&self) -> bool {
        self.buffer.is_empty() && self.next_block < self.blocks
    }
}

pub(crate) struct SegmentSource<'a> {
    segment: RangedSegment<'a>,
    docs: Side,
    tombstones: Side,
    after: Option<String>,
}

impl Source<'_> {
    /// The next entry's id, without taking it.
    fn front_id(&self) -> Option<&str> {
        match self {
            Self::Layer { entries, .. } => entries.front().map(|e| e.id.as_str()),
            Self::Segment(s) => match (front_id(&s.docs), front_id(&s.tombstones)) {
                (Some(d), Some(t)) => Some(d.min(t)),
                (d, t) => d.or(t),
            },
        }
    }

    fn pop(&mut self) -> Option<Entry> {
        match self {
            Self::Layer { entries, .. } => entries.pop_front(),
            Self::Segment(s) => {
                let doc_first = match (front_id(&s.docs), front_id(&s.tombstones)) {
                    (Some(d), Some(t)) => d <= t,
                    (Some(_), None) => true,
                    (None, _) => false,
                };
                let side = if doc_first {
                    &mut s.docs
                } else {
                    &mut s.tombstones
                };
                side.buffer.pop_front()
            }
        }
    }

    /// Fetch the next block of any side that ran dry and still has blocks.
    /// Nothing is fetched ahead.
    async fn fill(&mut self) -> Result<()> {
        let Self::Segment(s) = self else {
            return Ok(());
        };
        let after = s.after.clone();
        while s.docs.ran_dry() {
            let block = s.docs.next_block;
            s.docs.next_block += 1;
            let rows = s.segment.doc_block(block).await?;
            let entries = rows.into_iter().map(|(id, attrs)| Entry {
                id,
                body: Body::Doc(attrs),
            });
            push_sorted(&s.segment, &mut s.docs, entries, after.as_deref())?;
        }
        while s.tombstones.ran_dry() {
            let block = s.tombstones.next_block;
            s.tombstones.next_block += 1;
            let ids = s.segment.tombstone_block(block).await?;
            let entries = ids.into_iter().map(|id| Entry {
                id,
                body: Body::Tombstone,
            });
            push_sorted(&s.segment, &mut s.tombstones, entries, after.as_deref())?;
        }
        Ok(())
    }

    /// Whether a side of a segment emptied its buffer and may have more to
    /// read: the only case [`fill`](Self::fill) has anything to do.
    fn ran_dry(&self) -> bool {
        match self {
            Self::Segment(s) => s.docs.ran_dry() || s.tombstones.ran_dry(),
            Self::Layer { .. } => false,
        }
    }
}

/// The id at the front of a side's buffer.
fn front_id(side: &Side) -> Option<&str> {
    side.buffer.front().map(|e| e.id.as_str())
}

/// Buffer `entries` (one block, in order) that come after the cursor, checking
/// that ids only go up. The ranged reads trust the directory, and the whole
/// decoder that would catch an unsorted segment is not on this path.
fn push_sorted(
    segment: &RangedSegment<'_>,
    side: &mut Side,
    entries: impl Iterator<Item = Entry>,
    after: Option<&str>,
) -> Result<()> {
    for entry in entries {
        if side
            .last
            .as_deref()
            .is_some_and(|last| entry.id.as_str() <= last)
        {
            return Err(segment.corrupt("ids are not in ascending order"));
        }
        side.last = Some(entry.id.clone());
        if after.is_none_or(|after| entry.id.as_str() > after) {
            side.buffer.push_back(entry);
        }
    }
    Ok(())
}

/// The sources of one namespace, merged in id order.
pub(crate) struct Merge<'a> {
    sources: Vec<Source<'a>>,
    heads: NewestWins<String>,
}

impl<'a> Merge<'a> {
    /// Open every live segment in `ids` and the tail as sources and fill each
    /// one's first block. The tail is read alongside the segment opens.
    ///
    /// # Errors
    ///
    /// The errors of reading segments and the tail (a missing segment, or a
    /// fenced WAL object, is what the caller reloads the manifest for), and
    /// [`ItzmaError::ReadBudgetExceeded`](crate::ItzmaError::ReadBudgetExceeded).
    pub(crate) async fn open(
        open: &Open<'a>,
        ids: &'a [String],
        tail: impl Future<Output = Result<Arc<Layer>>>,
    ) -> Result<Self> {
        let (mut sources, tail) = futures::try_join!(open_sources(open, ids), tail)?;
        if !tail.is_empty() {
            let entries = layer_entries(&tail, open.after);
            sources.push(Source::Layer { entries });
        }
        // Every source's first block is read at the same time.
        try_join_all(sources.iter_mut().map(Source::fill)).await?;
        let mut merge = Self {
            heads: NewestWins::with_capacity(sources.len()),
            sources,
        };
        merge.push_heads();
        Ok(merge)
    }

    /// A merge over sources already filled.
    #[cfg(test)]
    pub(crate) fn from_sources(sources: Vec<Source<'a>>) -> Self {
        let mut merge = Self {
            heads: NewestWins::with_capacity(sources.len()),
            sources,
        };
        merge.push_heads();
        merge
    }

    fn push_heads(&mut self) {
        if self.sources.len() == 1 {
            return;
        }
        for (at, source) in self.sources.iter().enumerate() {
            if let Some(front) = source.front_id() {
                self.heads.push(at, front.to_owned());
            }
        }
    }

    /// The next live document in id order, after shadowing and tombstones.
    /// The caller tests any predicate.
    ///
    /// # Errors
    ///
    /// As for [`open`](Self::open), for the blocks read on the way.
    pub(crate) async fn next(&mut self) -> Result<Option<Winner>> {
        if self.sources.len() == 1 {
            return self.next_alone().await;
        }
        loop {
            let Some((id, winner)) = self.heads.pop() else {
                return Ok(None);
            };
            let entry = self.take(winner).await?;
            while let Some(older) = self.heads.pop_older(&id) {
                self.take(older).await?;
            }
            if let Some(Entry {
                id,
                body: Body::Doc(attrs),
                ..
            }) = entry
            {
                return Ok(Some(Winner { id, attrs }));
            }
        }
    }

    /// [`next`](Self::next) over a single source, which has nothing to
    /// shadow: its entries in order, without the heap. A one-segment
    /// namespace with no tail is common, and the heap's push, pop and id
    /// clone per row are most of what the merge costs there.
    async fn next_alone(&mut self) -> Result<Option<Winner>> {
        loop {
            let Some(entry) = self.sources[0].pop() else {
                return Ok(None);
            };
            if self.sources[0].ran_dry() {
                self.sources[0].fill().await?;
            }
            if let Body::Doc(attrs) = entry.body {
                return Ok(Some(Winner {
                    id: entry.id,
                    attrs,
                }));
            }
        }
    }

    /// Take source `at`'s front, refill it, and push its new front. `None` only
    /// if the heap named a source with nothing buffered, which `from_sources`
    /// and this function never let happen; the caller skips it rather than add
    /// an error for an impossible state.
    async fn take(&mut self, at: usize) -> Result<Option<Entry>> {
        let entry = self.sources[at].pop();
        // Most pops leave the buffer non-empty: skip building a refill future
        // for those, which is most of the per-row cost of the merge.
        if self.sources[at].ran_dry() {
            self.sources[at].fill().await?;
        }
        if let Some(front) = self.sources[at].front_id() {
            self.heads.push(at, front.to_owned());
        }
        Ok(entry)
    }
}

/// Every live segment as a source, in order, opened in parallel.
async fn open_sources<'a>(open: &Open<'a>, ids: &'a [String]) -> Result<Vec<Source<'a>>> {
    // Indexed by position, not mapped over borrowed values: see
    // `query::open_sources` for why a closure whose parameter borrows is not
    // provably `Send`.
    let opened: Vec<RangedSegment<'a>> = stream::iter(0..ids.len())
        .map(|at| open_one(open, &ids[at]))
        .buffered(FETCH_CONCURRENCY)
        .try_collect()
        .await?;
    Ok(opened
        .into_iter()
        .map(|segment| segment_source(segment, open.after))
        .collect())
}

/// Segment `id` opened for ranged reads.
async fn open_one<'a>(open: &Open<'a>, id: &str) -> Result<RangedSegment<'a>> {
    RangedSegment::open(
        open.store,
        open.namespace,
        id,
        open.dim,
        open.budget,
        open.policy,
        open.decoded,
    )
    .await
}

/// A version 2 or 3 segment read from its first block past `after`.
fn segment_source<'a>(segment: RangedSegment<'a>, after: Option<&str>) -> Source<'a> {
    let (doc_start, tombstone_start) = match after {
        Some(after) => (
            segment.first_doc_block_after(after),
            segment.first_tombstone_block_after(after),
        ),
        None => (0, 0),
    };
    let docs = Side {
        next_block: doc_start,
        blocks: segment.doc_block_count(),
        buffer: VecDeque::new(),
        last: None,
    };
    let tombstones = Side {
        next_block: tombstone_start,
        blocks: segment.tombstone_block_count(),
        buffer: VecDeque::new(),
        last: None,
    };
    Source::Segment(Box::new(SegmentSource {
        segment,
        docs,
        tombstones,
        after: after.map(str::to_owned),
    }))
}

/// A layer's documents and tombstones after the cursor, merged in id order.
fn layer_entries(layer: &Layer, after: Option<&str>) -> VecDeque<Entry> {
    let past = |id: &str| after.is_none_or(|after| id > after);
    let mut docs = layer
        .sorted_docs()
        .into_iter()
        .filter(|d| past(&d.id))
        .peekable();
    let mut tombstones = layer
        .sorted_tombstones()
        .into_iter()
        .filter(|id| past(id))
        .peekable();
    let mut entries = VecDeque::new();
    loop {
        let take_doc = match (docs.peek(), tombstones.peek()) {
            (Some(d), Some(t)) => d.id.as_str() <= *t,
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => break,
        };
        if take_doc {
            if let Some(d) = docs.next() {
                entries.push_back(Entry {
                    id: d.id.clone(),
                    body: Body::Doc(d.attrs.clone()),
                });
            }
        } else if let Some(id) = tombstones.next() {
            entries.push_back(Entry {
                id: id.to_owned(),
                body: Body::Tombstone,
            });
        }
    }
    entries
}

#[cfg(test)]
mod tests {
    use object_store::ObjectStoreExt;
    use object_store::memory::InMemory;

    use super::*;
    use crate::document::{Document, WalOp};
    use crate::segment;

    /// The linear merge `filter::run` used before the heap, kept as the
    /// reference: the smallest front id, the newest holder decides, every
    /// older copy is dropped.
    fn reference_merge(mut sources: Vec<VecDeque<Entry>>) -> Vec<String> {
        let mut out = Vec::new();
        loop {
            let Some(smallest) = sources
                .iter()
                .filter_map(VecDeque::front)
                .map(|e| e.id.clone())
                .min()
            else {
                return out;
            };
            let mut decided = false;
            for at in (0..sources.len()).rev() {
                if sources[at].front().is_none_or(|e| e.id != smallest) {
                    continue;
                }
                let Some(entry) = sources[at].pop_front() else {
                    continue;
                };
                if decided {
                    continue;
                }
                decided = true;
                if matches!(entry.body, Body::Doc(_)) {
                    out.push(entry.id);
                }
            }
        }
    }

    fn random_sources(seed: u64) -> Vec<VecDeque<Entry>> {
        let mut x = seed;
        let mut next = move |n: u64| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (x >> 33) % n
        };
        (0..1 + next(6))
            .map(|_| {
                let mut ids: Vec<u64> = (0..next(40)).map(|_| next(60)).collect();
                ids.sort_unstable();
                ids.dedup();
                ids.into_iter()
                    .map(|n| Entry {
                        id: format!("k{n:03}"),
                        body: match next(4) {
                            0 => Body::Tombstone,
                            _ => Body::Doc(Some(serde_json::json!({ "n": n }))),
                        },
                    })
                    .collect()
            })
            .collect()
    }

    fn copy(entries: &VecDeque<Entry>) -> VecDeque<Entry> {
        entries
            .iter()
            .map(|e| Entry {
                id: e.id.clone(),
                body: match &e.body {
                    Body::Doc(attrs) => Body::Doc(attrs.clone()),
                    Body::Tombstone => Body::Tombstone,
                },
            })
            .collect()
    }

    #[tokio::test]
    async fn the_heap_merge_equals_the_linear_merge() {
        for seed in 0..500 {
            let sources = random_sources(seed);
            let want = reference_merge(sources.iter().map(copy).collect());
            let mut merge = Merge::from_sources(
                sources
                    .into_iter()
                    .map(|entries| Source::Layer { entries })
                    .collect(),
            );
            let mut got = Vec::new();
            while let Some(winner) = merge.next().await.expect("next") {
                got.push(winner.id);
            }
            assert_eq!(got, want, "seed {seed}");
        }
    }

    fn sample_layer(docs: usize, tombstones: usize) -> Layer {
        let mut layer = Layer::new();
        for i in 0..docs {
            layer.apply(WalOp::Upsert(Document {
                id: format!("d{i:05}"),
                vector: vec![i as f32, 1.0],
                attrs: (i % 3 != 0).then(|| serde_json::json!({ "n": i })),
            }));
        }
        for i in 0..tombstones {
            layer.apply(WalOp::Delete(format!("d{:05}", i * 7 + docs)));
        }
        layer
    }

    /// Every winner of a merge over one stored segment.
    async fn winners(store: &InMemory, after: Option<&str>) -> Vec<(String, Option<Value>)> {
        let budget = ReadBudget::unlimited();
        let policy = FetchPolicy::local();
        let open = Open {
            store,
            namespace: "ns",
            dim: 2,
            budget: &budget,
            policy: &policy,
            decoded: None,
            after,
        };
        let ids = vec!["seg".to_string()];
        let mut merge = Merge::open(&open, &ids, async { Ok(Arc::new(Layer::new())) })
            .await
            .expect("open");
        let mut out = Vec::new();
        while let Some(winner) = merge.next().await.expect("next") {
            out.push((winner.id, winner.attrs));
        }
        out
    }

    #[tokio::test]
    async fn page_blocks_read_every_document_past_the_cursor() {
        let store = InMemory::new();
        let layer = sample_layer(1000, 50);
        let bytes = segment::encode_as(&layer, 2, 2).expect("encode");
        store
            .put(&segment::segment_path("ns", "seg"), bytes.into())
            .await
            .expect("put");
        for after in [None, Some("d00600"), Some("d00999"), Some("e")] {
            let want: Vec<(String, Option<Value>)> = layer
                .sorted_docs()
                .into_iter()
                .filter(|d| after.is_none_or(|after| d.id.as_str() > after))
                .map(|d| (d.id.clone(), d.attrs.clone()))
                .collect();
            assert_eq!(winners(&store, after).await, want, "after {after:?}");
        }
    }
}
