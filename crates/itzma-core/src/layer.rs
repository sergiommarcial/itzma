//! Layers: the net effect of applying operations in order.
//!
//! A [`Layer`] is a map of live documents plus a set of tombstones, the ids
//! that this layer deletes. Queries build one from segments and the WAL tail,
//! and compaction builds one to freeze into a segment. Both go through the
//! same code, so they cannot disagree about what an upsert or a delete means.
//!
//! The two halves never overlap: an id is either a live document or a
//! tombstone. A tombstone matters only when something older might hold the
//! id, so a layer that starts at the very beginning of the log drops them
//! (see [`Layer::drop_tombstones`]).

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use futures::stream::{BoxStream, TryStreamExt};

use crate::columnar::{Bitmap, IdColumn};
use crate::document::{Document, WalOp};
use crate::error::Result;
use crate::wal::WalEntry;

/// Live documents and tombstones produced by applying operations in order.
///
/// # Examples
///
/// ```
/// use itzma_core::layer::Layer;
/// use itzma_core::{Document, WalOp};
///
/// let doc = |id: &str| Document { id: id.into(), vector: vec![1.0], attrs: None };
///
/// let mut older = Layer::new();
/// older.apply(WalOp::Upsert(doc("a")));
///
/// let mut newer = Layer::new();
/// newer.apply(WalOp::Delete("a".into()));
///
/// older.overlay(newer);
/// assert_eq!(older.doc_count(), 0);
/// assert_eq!(older.sorted_tombstones(), ["a"]);
/// ```
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Layer {
    docs: HashMap<String, Document>,
    tombstones: HashSet<String>,
}

impl Layer {
    /// An empty layer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply one operation. An upsert replaces the document and clears any
    /// tombstone for its id. A delete removes the document and records a
    /// tombstone, even when the id is not in this layer, because it may be
    /// hiding a document in an older one.
    pub fn apply(&mut self, op: WalOp) {
        match op {
            WalOp::Upsert(doc) => {
                self.tombstones.remove(&doc.id);
                self.docs.insert(doc.id.clone(), doc);
            }
            WalOp::Delete(id) => {
                self.docs.remove(&id);
                self.tombstones.insert(id);
            }
        }
    }

    /// Apply every operation of one WAL entry, in order.
    pub fn apply_entry(&mut self, entry: WalEntry) {
        for op in entry.ops {
            self.apply(op);
        }
    }

    /// Apply every operation of every entry, in log order.
    pub fn apply_log(&mut self, log: Vec<(u64, WalEntry)>) {
        for (_seq, entry) in log {
            self.apply_entry(entry);
        }
    }

    /// Put a newer layer on top of this one: its tombstones remove documents
    /// here, then its documents replace or add. Folding a range of operations
    /// in two halves and overlaying the results equals replaying the whole
    /// range.
    pub fn overlay(&mut self, newer: Layer) {
        for id in newer.tombstones {
            self.docs.remove(&id);
            self.tombstones.insert(id);
        }
        for (id, doc) in newer.docs {
            self.tombstones.remove(&id);
            self.docs.insert(id, doc);
        }
    }

    /// Forget every tombstone. Only right for a layer with nothing older
    /// beneath it, which is a segment that starts at WAL seq 1.
    pub fn drop_tombstones(&mut self) {
        self.tombstones.clear();
    }

    /// Whether the layer holds neither a document nor a tombstone.
    pub(crate) fn is_empty(&self) -> bool {
        self.docs.is_empty() && self.tombstones.is_empty()
    }

    /// Whether this layer decides `id`: it holds it as a document or as a
    /// tombstone. Either one hides the id in every older layer.
    pub(crate) fn holds(&self, id: &str) -> bool {
        self.docs.contains_key(id) || self.tombstones.contains(id)
    }

    /// The live documents, borrowed, in no particular order, but the same
    /// order on every call while the layer is unchanged. The compute pool
    /// relies on that: it scores `docs()` in one pass and the caller pairs the
    /// scores with a second pass. Keep it true if the storage
    /// changes.
    pub(crate) fn docs(&self) -> impl Iterator<Item = &Document> {
        self.docs.values()
    }

    /// Number of live documents.
    pub fn doc_count(&self) -> usize {
        self.docs.len()
    }

    /// Number of tombstones.
    pub fn tombstone_count(&self) -> usize {
        self.tombstones.len()
    }

    /// Live documents in ascending id order. Sorting is what makes an
    /// encoded segment a pure function of the layer's contents.
    pub fn sorted_docs(&self) -> Vec<&Document> {
        let mut docs: Vec<&Document> = self.docs.values().collect();
        docs.sort_unstable_by(|a, b| a.id.cmp(&b.id));
        docs
    }

    /// Tombstoned ids in ascending order.
    pub fn sorted_tombstones(&self) -> Vec<&str> {
        let mut ids: Vec<&str> = self.tombstones.iter().map(String::as_str).collect();
        ids.sort_unstable();
        ids
    }

    /// Consume the layer, yielding its live documents in no particular order.
    pub fn into_docs(self) -> impl Iterator<Item = Document> {
        self.docs.into_values()
    }

    /// Live documents and tombstones, each in ascending id order, owned: what
    /// a streaming merge reads a whole input as.
    pub(crate) fn into_sorted_parts(self) -> (Vec<Document>, Vec<String>) {
        let mut docs: Vec<Document> = self.docs.into_values().collect();
        docs.sort_unstable_by(|a, b| a.id.cmp(&b.id));
        let mut tombstones: Vec<String> = self.tombstones.into_iter().collect();
        tombstones.sort_unstable();
        (docs, tombstones)
    }

    /// Build a layer from decoded parts. The caller has checked that the two
    /// lists are disjoint.
    pub(crate) fn from_parts(docs: Vec<Document>, tombstones: Vec<String>) -> Self {
        Self {
            docs: docs.into_iter().map(|d| (d.id.clone(), d)).collect(),
            tombstones: tombstones.into_iter().collect(),
        }
    }
}

/// Fold a stream of log entries, in order, into a layer of their own, one
/// entry at a time, and return it with the seq of the last entry (`None` for
/// an empty stream). Overlaying the result on older layers equals replaying
/// the entries on top of them (see [`Layer::overlay`]).
pub(crate) async fn fold_log(
    mut log: BoxStream<'_, Result<(u64, WalEntry)>>,
) -> Result<(Layer, Option<u64>)> {
    let mut layer = Layer::new();
    let mut last = None;
    while let Some((seq, entry)) = log.try_next().await? {
        layer.apply_entry(entry);
        last = Some(seq);
    }
    Ok((layer, last))
}

/// The heads of several sources sorted by id, sources numbered oldest first,
/// ordered so the smallest id comes out first and, for one id, the newest
/// source holding it. The one form of [`Layer::overlay`]'s rule for sorted
/// data: compaction's streaming merge and the flat source's survivors drive it
///. The caller pushes each source's next id after taking
/// one, and checks that a source's ids rise.
pub(crate) struct NewestWins<K> {
    heap: BinaryHeap<Reverse<(K, Reverse<usize>)>>,
}

impl<K: Ord> NewestWins<K> {
    pub(crate) fn with_capacity(sources: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(sources),
        }
    }

    /// `source`'s next id.
    pub(crate) fn push(&mut self, source: usize, id: K) {
        self.heap.push(Reverse((id, Reverse(source))));
    }

    /// The smallest id any source holds, and the newest source holding it:
    /// the one that decides the id.
    pub(crate) fn pop(&mut self) -> Option<(K, usize)> {
        self.heap
            .pop()
            .map(|Reverse((id, Reverse(source)))| (id, source))
    }

    /// Another source whose head is also `id`, newest first: one the id's
    /// winner shadows. `None` once no source's head is `id`.
    pub(crate) fn pop_older(&mut self, id: &K) -> Option<usize> {
        match self.heap.peek() {
            Some(Reverse((next, _))) if next == id => {
                self.heap.pop().map(|Reverse((_, Reverse(source)))| source)
            }
            _ => None,
        }
    }
}

/// One sorted source of [`survivors`]: its document ids and its tombstones,
/// each strictly ascending, the two disjoint.
pub(crate) struct SortedIds<'a> {
    pub(crate) docs: &'a IdColumn,
    pub(crate) tombstones: &'a IdColumn,
}

impl<'a> SortedIds<'a> {
    /// The smaller of the next document and the next tombstone at `at`, and
    /// whether it is a document.
    fn head(&self, (doc, tombstone): (usize, usize)) -> Option<(&'a str, bool)> {
        let docs: &'a IdColumn = self.docs;
        let tombstones: &'a IdColumn = self.tombstones;
        let d = (doc < docs.len()).then(|| docs.get(doc));
        let t = (tombstone < tombstones.len()).then(|| tombstones.get(tombstone));
        match (d, t) {
            (Some(d), Some(t)) if t < d => Some((t, false)),
            (Some(d), _) => Some((d, true)),
            (None, Some(t)) => Some((t, false)),
            (None, None) => None,
        }
    }
}

/// For sources sorted by id, oldest first, which document rows survive: one
/// bitmap per source, a row set when no newer source holds its id as a
/// document or a tombstone. [`Layer::overlay`]'s answer without building the
/// map.
pub(crate) fn survivors(sources: &[SortedIds<'_>]) -> Vec<Bitmap> {
    let mut live: Vec<Bitmap> = sources.iter().map(|s| Bitmap::new(s.docs.len())).collect();
    let mut at = vec![(0usize, 0usize); sources.len()];
    let mut wins = NewestWins::with_capacity(sources.len());
    for (s, source) in sources.iter().enumerate() {
        if let Some((id, _)) = source.head(at[s]) {
            wins.push(s, id);
        }
    }
    while let Some((id, winner)) = wins.pop() {
        if let Some((_, true)) = sources[winner].head(at[winner]) {
            live[winner].set(at[winner].0);
        }
        step(sources, winner, &mut at, &mut wins);
        while let Some(older) = wins.pop_older(&id) {
            step(sources, older, &mut at, &mut wins);
        }
    }
    live
}

/// Move source `s` past its head and push its next id.
fn step<'a>(
    sources: &[SortedIds<'a>],
    s: usize,
    at: &mut [(usize, usize)],
    wins: &mut NewestWins<&'a str>,
) {
    match sources[s].head(at[s]) {
        Some((_, true)) => at[s].0 += 1,
        Some((_, false)) => at[s].1 += 1,
        None => return,
    }
    if let Some((id, _)) = sources[s].head(at[s]) {
        wins.push(s, id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn docs_come_in_the_same_order_on_every_call() {
        let mut layer = Layer::new();
        for i in 0..5_000 {
            layer.apply(WalOp::Upsert(Document {
                id: format!("d{i}"),
                vector: vec![i as f32],
                attrs: None,
            }));
        }
        let first: Vec<&str> = layer.docs().map(|d| d.id.as_str()).collect();
        let shared = std::sync::Arc::new(layer.clone());
        let second: Vec<&str> = shared.docs().map(|d| d.id.as_str()).collect();
        let again: Vec<&str> = shared.docs().map(|d| d.id.as_str()).collect();
        assert_eq!(first.len(), 5_000);
        assert_eq!(second, again);
        assert_eq!(
            layer.docs().map(|d| d.id.as_str()).collect::<Vec<_>>(),
            first
        );
    }
    use crate::document::{Document, WalOp};

    fn up(id: &str, v: f32) -> WalOp {
        WalOp::Upsert(Document {
            id: id.into(),
            vector: vec![v],
            attrs: None,
        })
    }

    fn del(id: &str) -> WalOp {
        WalOp::Delete(id.into())
    }

    fn replay(ops: impl IntoIterator<Item = WalOp>) -> Layer {
        let mut layer = Layer::new();
        for op in ops {
            layer.apply(op);
        }
        layer
    }

    #[test]
    fn upsert_inserts_and_replaces() {
        let layer = replay([up("a", 1.0), up("a", 2.0)]);
        assert_eq!(layer.doc_count(), 1);
        assert_eq!(layer.docs["a"].vector, vec![2.0]);
        assert_eq!(layer.tombstone_count(), 0);
    }

    #[test]
    fn delete_removes_the_doc_and_leaves_a_tombstone() {
        let layer = replay([up("a", 1.0), del("a")]);
        assert_eq!(layer.doc_count(), 0);
        assert_eq!(layer.sorted_tombstones(), ["a"]);
    }

    #[test]
    fn delete_of_an_unknown_id_still_leaves_a_tombstone() {
        // It may be hiding a document in an older segment.
        let layer = replay([del("ghost")]);
        assert_eq!(layer.sorted_tombstones(), ["ghost"]);
    }

    #[test]
    fn upsert_after_delete_clears_the_tombstone() {
        let layer = replay([del("a"), up("a", 3.0)]);
        assert_eq!(layer.doc_count(), 1);
        assert_eq!(layer.tombstone_count(), 0);
    }

    #[test]
    fn drop_tombstones_keeps_the_docs() {
        let mut layer = replay([up("a", 1.0), del("b")]);
        layer.drop_tombstones();
        assert_eq!(layer.doc_count(), 1);
        assert_eq!(layer.tombstone_count(), 0);
    }

    #[test]
    fn sorted_views_are_in_id_order() {
        let layer = replay([up("c", 1.0), up("a", 1.0), up("b", 1.0), del("z"), del("y")]);
        let ids: Vec<&str> = layer.sorted_docs().iter().map(|d| d.id.as_str()).collect();
        assert_eq!(ids, ["a", "b", "c"]);
        assert_eq!(layer.sorted_tombstones(), ["y", "z"]);
    }

    #[test]
    fn overlay_newer_delete_hides_an_older_doc() {
        let mut older = replay([up("a", 1.0)]);
        older.overlay(replay([del("a")]));
        assert_eq!(older.doc_count(), 0);
        assert_eq!(older.sorted_tombstones(), ["a"]);
    }

    #[test]
    fn overlay_newer_upsert_beats_an_older_tombstone() {
        let mut older = replay([del("a")]);
        older.overlay(replay([up("a", 5.0)]));
        assert_eq!(older.docs["a"].vector, vec![5.0]);
        assert_eq!(older.tombstone_count(), 0);
    }

    /// Folding a range in two halves must equal replaying it whole. This is
    /// what makes a merge of segments produce the same bytes as a fold of the
    /// WAL, so it is checked for every op sequence up to length 5 over two ids.
    #[test]
    fn overlay_matches_replay_for_every_short_sequence() {
        for len in 0..=5u32 {
            for code in 0..4u32.pow(len) {
                let ops: Vec<WalOp> = (0..len)
                    .map(|i| {
                        let v = i as f32;
                        match (code / 4u32.pow(i)) % 4 {
                            0 => up("a", v),
                            1 => up("b", v),
                            2 => del("a"),
                            _ => del("b"),
                        }
                    })
                    .collect();
                let whole = replay(ops.clone());
                for split in 0..=ops.len() {
                    let mut left = replay(ops[..split].to_vec());
                    left.overlay(replay(ops[split..].to_vec()));
                    assert_eq!(left, whole, "ops {ops:?} split at {split}");
                }
            }
        }
    }

    #[test]
    fn newest_wins_pops_the_smallest_id_and_the_newest_source_first() {
        let mut wins = NewestWins::with_capacity(3);
        wins.push(0, "b");
        wins.push(1, "a");
        wins.push(2, "b");
        assert_eq!(wins.pop(), Some(("a", 1)));
        assert_eq!(wins.pop_older(&"a"), None);
        assert_eq!(wins.pop(), Some(("b", 2)));
        assert_eq!(wins.pop_older(&"b"), Some(0));
        assert_eq!(wins.pop_older(&"b"), None);
        assert_eq!(wins.pop(), None);
    }

    #[test]
    fn pop_older_takes_every_older_holder_newest_first() {
        let mut wins = NewestWins::with_capacity(4);
        for source in [0, 3, 1, 2] {
            wins.push(source, "x");
        }
        wins.push(1, "y");
        assert_eq!(wins.pop(), Some(("x", 3)));
        assert_eq!(wins.pop_older(&"x"), Some(2));
        assert_eq!(wins.pop_older(&"x"), Some(1));
        assert_eq!(wins.pop_older(&"x"), Some(0));
        assert_eq!(wins.pop_older(&"x"), None);
        assert_eq!(wins.pop(), Some(("y", 1)));
    }

    /// `survivors` over sorted runs gives what `Layer::overlay` gives: a row
    /// survives exactly when its source is the newest holding the id, and the
    /// overlay's documents are exactly the surviving rows.
    #[test]
    fn survivors_equal_overlay_on_random_runs() {
        use crate::columnar::IdColumn;
        let mut state = 0x5eed_0043_u64;
        let mut next = move |n: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % n
        };
        for case in 0..300 {
            let sources = 1 + next(5) as usize;
            // Up to 600 ids, so sources span several 256-row blocks.
            let ids = 1 + next(600);
            let layers: Vec<Layer> = (0..sources)
                .map(|s| {
                    // Some sources empty.
                    let ops = next(700) as usize;
                    replay(
                        (0..ops)
                            .map(|i| {
                                let id = format!("i{:04}", next(ids));
                                match next(4) {
                                    0 => del(&id),
                                    _ => up(&id, (s * 1000 + i) as f32),
                                }
                            })
                            .collect::<Vec<_>>(),
                    )
                })
                .collect();
            let docs: Vec<IdColumn> = layers
                .iter()
                .map(|l| IdColumn::from_sorted(l.sorted_docs().iter().map(|d| d.id.as_str())))
                .collect();
            let tombstones: Vec<IdColumn> = layers
                .iter()
                .map(|l| IdColumn::from_sorted(l.sorted_tombstones()))
                .collect();
            let sorted: Vec<SortedIds<'_>> = docs
                .iter()
                .zip(&tombstones)
                .map(|(docs, tombstones)| SortedIds { docs, tombstones })
                .collect();

            let live = survivors(&sorted);

            let mut overlaid = Layer::new();
            for layer in layers.iter().cloned() {
                overlaid.overlay(layer);
            }
            let mut from_bitmaps: Vec<(String, f32)> = Vec::new();
            for (s, layer) in layers.iter().enumerate() {
                assert_eq!(live[s].len(), layer.doc_count(), "case {case}");
                for (row, doc) in layer.sorted_docs().into_iter().enumerate() {
                    if live[s].get(row) {
                        from_bitmaps.push((doc.id.clone(), doc.vector[0]));
                    }
                }
            }
            from_bitmaps.sort_by(|a, b| a.0.cmp(&b.0));
            let expected: Vec<(String, f32)> = overlaid
                .sorted_docs()
                .into_iter()
                .map(|d| (d.id.clone(), d.vector[0]))
                .collect();
            assert_eq!(from_bitmaps, expected, "case {case}");
        }
    }

    #[test]
    fn survivors_of_no_sources_and_of_empty_sources_are_empty() {
        use crate::columnar::IdColumn;
        assert!(survivors(&[]).is_empty());
        let empty = IdColumn::default();
        let live = survivors(&[SortedIds {
            docs: &empty,
            tombstones: &empty,
        }]);
        assert_eq!(live.len(), 1);
        assert_eq!(live[0].len(), 0);
    }

    /// The same property over longer logs and more ids, which is what an
    /// incremental tail leans on: it folds a tail in two steps,
    /// split wherever the last query happened to stop.
    #[test]
    fn folding_long_logs_in_two_steps_equals_folding_in_one() {
        let mut state = 0x9e37_79b9_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..200 {
            let ops: Vec<WalOp> = (0..12)
                .map(|i| {
                    let id = ["a", "b", "c", "d"][(next() % 4) as usize];
                    if next() % 3 == 0 {
                        del(id)
                    } else {
                        up(id, i as f32)
                    }
                })
                .collect();
            let whole = replay(ops.clone());
            for split in 0..=ops.len() {
                let mut left = replay(ops[..split].to_vec());
                left.overlay(replay(ops[split..].to_vec()));
                assert_eq!(left, whole, "ops {ops:?} split at {split}");
            }
        }
    }
}
