//! The k-way merge of a streaming merge.
//!
//! Every input is sorted by id: a version 2 or 3 segment read forward a window at
//! a time, a version 1 segment or the WAL tail as a whole `Layer`. For each
//! id the newest input's entry wins, a document or a tombstone, which is
//! `Layer::overlay`'s rule, and the winners go to a [`SegmentWriter`] in id
//! order. Nothing needs to be held but each input's current window and the
//! output.

use std::collections::VecDeque;
use std::iter::Peekable;

use object_store::ObjectStore;

use crate::budget::ReadBudget;
use crate::cpu::{self, Cost, Lane};
use crate::document::Document;
use crate::error::{ItzmaError, Result};
use crate::fetch::FetchPolicy;
use crate::layer::{Layer, NewestWins};
use crate::segment::{self, AttrIn, RangedSegment, SegmentWriter};

/// Bytes one input reads ahead (a guess, measured only on `InMemory` so
/// far). A window is at least one block, so a block
/// bigger than this is read on its own.
pub(crate) const MERGE_WINDOW: u64 = 8 * 1024 * 1024;

/// A segment at or below this size (or the window, if smaller) is read whole:
/// one request charged its exact size, where opening it for ranged reads
/// charges a 64 KiB suffix guess up front. Small enough that holding it
/// decoded costs little.
const WHOLE_BELOW: u64 = 1024 * 1024;

/// Entries handed to the writer per trip to the compute pool.
const WRITE_BATCH: usize = 4096;

/// What one input holds for one id.
pub(crate) enum Entry {
    Doc(Document),
    Tombstone(String),
}

impl Entry {
    fn id(&self) -> &str {
        match self {
            Entry::Doc(doc) => &doc.id,
            Entry::Tombstone(id) => id,
        }
    }

    /// Roughly what writing it costs, for the compute pool's threshold.
    fn weight(&self) -> usize {
        match self {
            Entry::Doc(doc) => doc.vector.len() * 4 + doc.id.len() + 64,
            Entry::Tombstone(id) => id.len(),
        }
    }
}

fn corrupt(label: &str, reason: &str) -> ItzmaError {
    ItzmaError::CorruptSegment {
        path: label.to_owned(),
        reason: reason.to_owned(),
    }
}

/// The next of two sorted runs, documents and tombstones, by id. One id in
/// both is damage: a segment never holds one.
fn smaller(label: &str, doc: Option<&str>, tombstone: Option<&str>) -> Result<Option<bool>> {
    match (doc, tombstone) {
        (None, None) => Ok(None),
        (Some(_), None) => Ok(Some(true)),
        (None, Some(_)) => Ok(Some(false)),
        (Some(d), Some(t)) if d == t => {
            Err(corrupt(label, "an id is both a document and a tombstone"))
        }
        (Some(d), Some(t)) => Ok(Some(d < t)),
    }
}

/// An input held whole: a version 1 segment or the WAL tail.
pub(crate) struct WholeInput {
    label: String,
    docs: Peekable<std::vec::IntoIter<Document>>,
    tombstones: Peekable<std::vec::IntoIter<String>>,
}

impl WholeInput {
    fn new(label: String, layer: Layer) -> Self {
        let (docs, tombstones) = layer.into_sorted_parts();
        Self::from_parts_unchecked(&label, docs, tombstones)
    }

    /// Parts taken as given, in the order given. Tests use it to forge an
    /// input out of order.
    pub(crate) fn from_parts_unchecked(
        label: &str,
        docs: Vec<Document>,
        tombstones: Vec<String>,
    ) -> Self {
        Self {
            label: label.to_owned(),
            docs: docs.into_iter().peekable(),
            tombstones: tombstones.into_iter().peekable(),
        }
    }

    fn next(&mut self) -> Result<Option<Entry>> {
        let doc = self.docs.peek().map(|d| d.id.as_str());
        let tombstone = self.tombstones.peek().map(String::as_str);
        Ok(match smaller(&self.label, doc, tombstone)? {
            None => None,
            Some(true) => self.docs.next().map(Entry::Doc),
            Some(false) => self.tombstones.next().map(Entry::Tombstone),
        })
    }
}

/// A version 2 or 3 segment read forward, one window of documents and one of
/// tombstones at a time.
pub(crate) struct RangedInput<'a> {
    label: String,
    segment: RangedSegment<'a>,
    window: u64,
    next_doc_block: usize,
    next_tombstone_block: usize,
    docs: VecDeque<Document>,
    tombstones: VecDeque<String>,
}

impl RangedInput<'_> {
    /// Blocks from `start` whose bytes fit the window, at least one.
    fn window_end(&self, start: usize, count: usize, bytes_of: impl Fn(usize) -> u64) -> usize {
        let mut end = start;
        let mut bytes = 0u64;
        while end < count {
            let next = bytes_of(end);
            if end > start && bytes.saturating_add(next) > self.window {
                break;
            }
            bytes = bytes.saturating_add(next);
            end += 1;
        }
        end
    }

    /// The next window, fetched only once the last one is drained; fetching
    /// it in the background is
    async fn refill_docs(&mut self) -> Result<()> {
        let count = self.segment.doc_block_count();
        let start = self.next_doc_block;
        if !self.docs.is_empty() || start >= count {
            return Ok(());
        }
        let end = self.window_end(start, count, |b| self.segment.doc_window_bytes(b));
        let window = self.segment.doc_window(start..end).await?;
        let label = self.label.clone();
        let cost = Cost::Bytes(window.vectors.values.len() * 4 + window.attrs.stored_bytes());
        let docs = cpu::run(Lane::Maintenance, cost, move || documents(&label, window)).await?;
        // The queue is empty here, so the decoded window becomes it as is.
        self.docs = VecDeque::from(docs);
        self.next_doc_block = end;
        Ok(())
    }

    async fn refill_tombstones(&mut self) -> Result<()> {
        let count = self.segment.tombstone_block_count();
        let start = self.next_tombstone_block;
        if !self.tombstones.is_empty() || start >= count {
            return Ok(());
        }
        let end = self.window_end(start, count, |b| self.segment.tombstone_window_bytes(b));
        self.tombstones = VecDeque::from(self.segment.tombstone_window(start..end).await?);
        self.next_tombstone_block = end;
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<Entry>> {
        self.refill_docs().await?;
        self.refill_tombstones().await?;
        let doc = self.docs.front().map(|d| d.id.as_str());
        let tombstone = self.tombstones.front().map(String::as_str);
        Ok(match smaller(&self.label, doc, tombstone)? {
            None => None,
            Some(true) => self.docs.pop_front().map(Entry::Doc),
            Some(false) => self.tombstones.pop_front().map(Entry::Tombstone),
        })
    }
}

/// A window's rows as documents, attributes parsed from their JSON.
fn documents(label: &str, window: segment::DocWindow) -> Result<Vec<Document>> {
    let dim = window.vectors.dim;
    window
        .ids
        .into_iter()
        .enumerate()
        .map(|(i, id)| {
            let attrs = window.attrs.get(i).value(label)?;
            Ok(Document {
                id,
                vector: window.vectors.values[i * dim..(i + 1) * dim].to_vec(),
                attrs,
            })
        })
        .collect()
}

/// One sorted input of a merge. A merge holds a handful, so the size of the
/// bigger variant costs nothing worth a box.
#[allow(clippy::large_enum_variant)]
pub(crate) enum Input<'a> {
    /// A version 2 or 3 segment, read forward.
    Ranged(RangedInput<'a>),
    /// A version 1 segment or the WAL tail, decoded whole.
    Whole(WholeInput),
}

impl<'a> Input<'a> {
    /// Open segment `id` of `namespace`, whose `data.bin` the caller's
    /// listing says is `size` bytes: read forward when it is version 2 and
    /// bigger than [`WHOLE_BELOW`] (or `window`, if smaller), else decoded
    /// whole. Everything is charged to `budget`, and
    /// nothing goes through the decoded cache.
    ///
    /// # Errors
    ///
    /// What opening or reading the segment gives: [`segment::missing`] for one
    /// that is gone, [`ItzmaError::CorruptSegment`], a budget refusal.
    #[allow(clippy::too_many_arguments)] // the handle's read settings and the listing's size
    pub(crate) async fn segment(
        store: &'a dyn ObjectStore,
        namespace: &str,
        id: &str,
        dim: usize,
        budget: &'a ReadBudget,
        policy: &'a FetchPolicy,
        size: u64,
        window: u64,
    ) -> Result<Input<'a>> {
        let path = segment::segment_path(namespace, id);
        let label = path.to_string();
        if size > window.min(WHOLE_BELOW) {
            let segment =
                RangedSegment::open(store, namespace, id, dim, budget, policy, None).await?;
            return Ok(Input::Ranged(RangedInput {
                label,
                segment,
                window,
                next_doc_block: 0,
                next_tombstone_block: 0,
                docs: VecDeque::new(),
                tombstones: VecDeque::new(),
            }));
        }
        let layer =
            segment::fetch(store, namespace, id, dim, budget, policy, Lane::Maintenance).await?;
        Ok(Input::Whole(WholeInput::new(label, layer)))
    }

    /// The WAL tail, or any layer, as an input.
    pub(crate) fn layer(layer: Layer) -> Input<'a> {
        Input::Whole(WholeInput::new("the WAL tail".into(), layer))
    }

    fn label(&self) -> &str {
        match self {
            Input::Ranged(input) => &input.label,
            Input::Whole(input) => &input.label,
        }
    }

    /// The next entry in id order, or `None` once drained.
    pub(crate) async fn next(&mut self) -> Result<Option<Entry>> {
        match self {
            Input::Ranged(input) => input.next().await,
            Input::Whole(input) => input.next(),
        }
    }
}

/// Take input `at`'s next entry into `heads` and `wins`, checking that the
/// input's ids rise strictly.
async fn advance(
    inputs: &mut [Input<'_>],
    heads: &mut [Option<Entry>],
    wins: &mut NewestWins<String>,
    at: usize,
    last: Option<&str>,
) -> Result<()> {
    let Some(entry) = inputs[at].next().await? else {
        return Ok(());
    };
    if last.is_some_and(|last| entry.id() <= last) {
        return Err(corrupt(
            inputs[at].label(),
            "ids are not in strictly ascending order",
        ));
    }
    wins.push(at, entry.id().to_owned());
    heads[at] = Some(entry);
    Ok(())
}

/// Merge `inputs`, oldest first, into `writer`: for each id the newest
/// input's entry, tombstones left out when `drop_tombstones` (a group that
/// starts at WAL seq 1). The writer is fed on the maintenance lane in batches.
///
/// # Errors
///
/// What reading an input gives, [`ItzmaError::CorruptSegment`] for an input
/// out of order, and what the writer gives.
pub(crate) async fn merge_into(
    mut inputs: Vec<Input<'_>>,
    drop_tombstones: bool,
    mut writer: SegmentWriter,
) -> Result<SegmentWriter> {
    let mut heads: Vec<Option<Entry>> = inputs.iter().map(|_| None).collect();
    let mut wins = NewestWins::with_capacity(inputs.len());
    for at in 0..inputs.len() {
        advance(&mut inputs, &mut heads, &mut wins, at, None).await?;
    }
    let mut batch = Vec::with_capacity(WRITE_BATCH);
    let mut batch_weight = 0;
    while let Some((id, at)) = wins.pop() {
        let winner = heads[at].take();
        advance(&mut inputs, &mut heads, &mut wins, at, Some(&id)).await?;
        while let Some(older) = wins.pop_older(&id) {
            heads[older] = None;
            advance(&mut inputs, &mut heads, &mut wins, older, Some(&id)).await?;
        }
        match winner {
            Some(Entry::Tombstone(_)) if drop_tombstones => {}
            Some(entry) => {
                batch_weight += entry.weight();
                batch.push(entry);
            }
            None => {}
        }
        if batch.len() == WRITE_BATCH {
            writer = write(writer, std::mem::take(&mut batch), batch_weight).await?;
            batch_weight = 0;
        }
    }
    if !batch.is_empty() {
        writer = write(writer, batch, batch_weight).await?;
    }
    Ok(writer)
}

/// Feed `batch` to `writer` on the maintenance lane and hand it back.
async fn write(
    mut writer: SegmentWriter,
    batch: Vec<Entry>,
    weight: usize,
) -> Result<SegmentWriter> {
    cpu::run(Lane::Maintenance, Cost::Bytes(weight), move || {
        for entry in batch {
            match entry {
                Entry::Doc(doc) => writer.doc(&doc.id, &doc.vector, AttrIn::Owned(doc.attrs))?,
                Entry::Tombstone(id) => writer.tombstone(&id)?,
            }
        }
        Ok(writer)
    })
    .await
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use object_store::memory::InMemory;
    use object_store::{ObjectStore, ObjectStoreExt};

    use super::*;
    use crate::attrs::AttrSchema;
    use crate::document::WalOp;
    use crate::segment::{encode_as, encode_parts, segment_id, segment_path};

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n.max(1)
        }
    }

    const DIM: usize = 3;

    /// One input: upserts and deletes over a shared id space, so ids repeat
    /// across inputs, deletes hit ids a later input inserts again, and some
    /// ids are only ever tombstones.
    fn input(rng: &mut Rng, space: u64) -> Layer {
        let mut l = Layer::new();
        for _ in 0..rng.below(space + 1) {
            let n = rng.below(space);
            let id = format!("id{n:04}");
            if rng.below(4) == 0 {
                l.apply(WalOp::Delete(id));
            } else {
                let v = rng.below(1000) as f32;
                l.apply(WalOp::Upsert(Document {
                    id,
                    vector: vec![v, -v, 0.5],
                    attrs: (rng.below(2) == 0).then(|| serde_json::json!({ "v": v, "s": "x" })),
                }));
            }
        }
        l
    }

    /// Cases of 1 to 6 inputs, oldest first, and whether tombstones go.
    fn cases(count: usize) -> Vec<(Vec<Layer>, bool)> {
        let mut rng = Rng(0x00ab_0037);
        (0..count)
            .map(|_| {
                let space = rng.below(600) + 1;
                let inputs = (0..rng.below(6) + 1)
                    .map(|_| input(&mut rng, space))
                    .collect();
                (inputs, rng.below(2) == 0)
            })
            .collect()
    }

    fn expected(inputs: &[Layer], drop: bool) -> Vec<u8> {
        let mut all = Layer::new();
        for layer in inputs {
            all.overlay(layer.clone());
        }
        if drop {
            all.drop_tombstones();
        }
        Bytes::from(encode_parts(&all, DIM, 2, &AttrSchema::free_form()).unwrap()).to_vec()
    }

    async fn merged(inputs: Vec<Input<'_>>, drop: bool) -> Result<Vec<u8>> {
        let writer = merge_into(
            inputs,
            drop,
            SegmentWriter::new(DIM, 2, &AttrSchema::free_form())?,
        )
        .await?;
        Ok(Bytes::from(writer.finish()?).to_vec())
    }

    #[tokio::test]
    async fn merge_equals_overlay_on_random_inputs() {
        for (case, (inputs, drop)) in cases(500).into_iter().enumerate() {
            let want = expected(&inputs, drop);
            let got = merged(inputs.into_iter().map(Input::layer).collect(), drop)
                .await
                .unwrap();
            assert!(got == want, "case {case}");
        }
    }

    /// Each input as a segment of `version`.
    async fn store_of(inputs: &[Layer], versions: &[u32]) -> InMemory {
        let store = InMemory::new();
        for (i, layer) in inputs.iter().enumerate() {
            let id = segment_id(i as u64 + 1, i as u64 + 1);
            let version = versions[i % versions.len()];
            let data = encode_as(layer, DIM, version).unwrap();
            store
                .put(&segment_path("ns", &id), data.into())
                .await
                .unwrap();
        }
        store
    }

    async fn open_all<'a>(
        store: &'a dyn ObjectStore,
        count: usize,
        budget: &'a ReadBudget,
        policy: &'a FetchPolicy,
        window: u64,
    ) -> Vec<Input<'a>> {
        let mut inputs = Vec::new();
        for i in 0..count {
            let id = segment_id(i as u64 + 1, i as u64 + 1);
            let size = store.head(&segment_path("ns", &id)).await.unwrap().size;
            let input = Input::segment(store, "ns", &id, DIM, budget, policy, size, window);
            inputs.push(input.await.unwrap());
        }
        inputs
    }

    async fn ranged_cases(versions: &[u32], window: u64, count: usize) {
        for (case, (inputs, drop)) in cases(count).into_iter().enumerate() {
            let want = expected(&inputs, drop);
            let store = store_of(&inputs, versions).await;
            let budget = ReadBudget::unlimited();
            let policy = FetchPolicy::local();
            let opened = open_all(&store, inputs.len(), &budget, &policy, window).await;
            let got = merged(opened, drop).await.unwrap();
            assert!(
                got == want,
                "case {case}, versions {versions:?}, window {window}"
            );
        }
    }

    /// A window smaller than any segment here, so every version 2 input is
    /// read forward rather than whole.
    const SMALL_WINDOW: u64 = 64;

    #[tokio::test]
    async fn ranged_inputs_merge_the_same_as_layers() {
        ranged_cases(&[2], SMALL_WINDOW, 120).await;
    }

    #[tokio::test]
    async fn small_segments_read_whole_merge_the_same() {
        ranged_cases(&[2], MERGE_WINDOW, 60).await;
    }

    #[tokio::test]
    async fn a_one_block_window_merges_the_same() {
        ranged_cases(&[2], 1, 60).await;
    }

    #[tokio::test]
    async fn a_window_bounds_what_one_refill_reads() {
        let mut layer = Layer::new();
        for n in 0..2_000 {
            layer.apply(WalOp::Upsert(Document {
                id: format!("d{n:05}"),
                vector: vec![n as f32; DIM],
                attrs: Some(serde_json::json!({ "n": n })),
            }));
        }
        for n in 0..700 {
            layer.apply(WalOp::Delete(format!("t{n:05}")));
        }
        let store = store_of(std::slice::from_ref(&layer), &[2]).await;
        let budget = ReadBudget::unlimited();
        let policy = FetchPolicy::local();
        let window = 16 * 1024;
        let mut input = open_all(&store, 1, &budget, &policy, window)
            .await
            .remove(0);
        let mut largest = 0;
        let mut entries = 0;
        loop {
            let before = budget.used();
            let Some(_) = input.next().await.unwrap() else {
                break;
            };
            entries += 1;
            largest = largest.max(budget.used() - before);
        }
        assert_eq!(entries, 2_700);
        // One window, plus one block that crossed it, plus the planner's
        // gaps (64 KiB each under `local()`, at most two per window).
        let block = 256 * (DIM as u64 * 4 + 64);
        assert!(
            largest <= window + block + 2 * 64 * 1024,
            "a refill read {largest} bytes"
        );
        let stored = encode_as(&layer, DIM, 2).unwrap().len() as u64;
        assert!(
            budget.used() <= stored + 64 * 1024 * 4,
            "read {}",
            budget.used()
        );
    }

    #[tokio::test]
    async fn a_segment_collected_after_it_was_opened_is_missing() {
        let mut layer = Layer::new();
        for n in 0..600 {
            layer.apply(WalOp::Upsert(Document {
                id: format!("d{n:04}"),
                vector: vec![n as f32; DIM],
                attrs: None,
            }));
        }
        let store = store_of(std::slice::from_ref(&layer), &[2]).await;
        let budget = ReadBudget::unlimited();
        let policy = FetchPolicy::local();
        let mut input = open_all(&store, 1, &budget, &policy, SMALL_WINDOW)
            .await
            .remove(0);
        assert!(matches!(input, Input::Ranged(_)));
        store
            .delete(&segment_path("ns", &segment_id(1, 1)))
            .await
            .unwrap();
        let err = input.next().await.err().expect("the window is gone");
        assert!(segment::is_missing(&err), "{err:?}");
    }

    #[tokio::test]
    async fn an_input_out_of_order_is_refused() {
        let docs: Vec<Document> = ["b", "a"]
            .iter()
            .map(|id| Document {
                id: (*id).into(),
                vector: vec![0.0; DIM],
                attrs: None,
            })
            .collect();
        let input = Input::Whole(WholeInput::from_parts_unchecked("forged", docs, vec![]));
        let err = merged(vec![input], false).await.unwrap_err();
        assert!(
            matches!(&err, ItzmaError::CorruptSegment { path, .. } if path == "forged"),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn an_id_both_document_and_tombstone_in_one_input_is_refused() {
        let docs = vec![Document {
            id: "a".into(),
            vector: vec![0.0; DIM],
            attrs: None,
        }];
        let input = Input::Whole(WholeInput::from_parts_unchecked(
            "forged",
            docs,
            vec!["a".into()],
        ));
        let err = merged(vec![input], false).await.unwrap_err();
        assert!(
            matches!(&err, ItzmaError::CorruptSegment { reason, .. }
                if reason.contains("both a document and a tombstone")),
            "{err:?}"
        );
    }
}
