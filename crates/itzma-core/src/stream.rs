//! Streaming columnar ingest: the chunker that cuts a stream of
//! batches into WAL-sized chunks.

use bytes::Bytes;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};

use crate::columns::DocColumns;
use crate::cpu::{self, Cost, Lane};
use crate::document::Schema;
use crate::error::{ItzmaError, Result};
use crate::namespace::Namespace;
use crate::wal::{self, WriteKey};
use crate::wal_codec::{self, COUNTS_LEN};

/// The default [`StreamOptions::target_bytes`]: half the WAL's 64 MiB body
/// limit, so a chunk's own estimate never pushes it over.
pub const DEFAULT_STREAM_TARGET_BYTES: usize = wal::MAX_BODY_BYTES / 2;

/// The default [`StreamOptions::depth`].
pub const DEFAULT_STREAM_DEPTH: usize = 2;

/// The longest key a keyed stream takes: each chunk's key is `<key>#<n>`,
/// a WAL key is at most 128 bytes, and `n` can take 20 digits.
const MAX_STREAM_KEY: usize = 107;

/// How [`Namespace::upsert_stream`] cuts and queues a stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct StreamOptions {
    /// About how many encoded body bytes each WAL object holds: 1 to
    /// [`wal::MAX_BODY_BYTES`]. A single row bigger than this is an object of
    /// its own. Default [`DEFAULT_STREAM_TARGET_BYTES`].
    pub target_bytes: usize,
    /// How many encoded chunks may wait for the writer, at least 1. Memory
    /// is about `(depth + 2) * target_bytes`, whatever the input's size.
    /// Default [`DEFAULT_STREAM_DEPTH`].
    pub depth: usize,
}

impl Default for StreamOptions {
    fn default() -> Self {
        Self {
            target_bytes: DEFAULT_STREAM_TARGET_BYTES,
            depth: DEFAULT_STREAM_DEPTH,
        }
    }
}

impl StreamOptions {
    /// Set [`target_bytes`](Self::target_bytes).
    #[must_use]
    pub fn target_bytes(mut self, bytes: usize) -> Self {
        self.target_bytes = bytes;
        self
    }

    /// Set [`depth`](Self::depth).
    #[must_use]
    pub fn depth(mut self, depth: usize) -> Self {
        self.depth = depth;
        self
    }

    /// Check the options, as [`Namespace::upsert_stream`] does before it
    /// reads its input.
    ///
    /// # Errors
    ///
    /// [`ItzmaError::InvalidConfig`] for a `target_bytes` of 0 or above
    /// [`wal::MAX_BODY_BYTES`], or a `depth` of 0.
    pub fn validate(&self) -> Result<()> {
        if self.target_bytes == 0 || self.target_bytes > wal::MAX_BODY_BYTES {
            return Err(ItzmaError::InvalidConfig(
                "target_bytes must be 1 to the WAL's 64 MiB body limit",
            ));
        }
        if self.depth == 0 {
            return Err(ItzmaError::InvalidConfig("depth must be at least 1"));
        }
        Ok(())
    }
}

/// What a stream wrote: the first and last seq it landed at, and how many
/// rows and objects. On an error, [`StreamError::landed`] is this for what
/// landed before it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct StreamReport {
    /// The seq of the first object written, `None` if none was.
    pub first_seq: Option<u64>,
    /// The seq of the last object written, `None` if none was.
    pub last_seq: Option<u64>,
    /// The rows written. After an error, an unkeyed load resumes by feeding
    /// its input again from this row.
    pub rows: u64,
    /// The WAL objects written.
    pub objects: u64,
}

/// A stream that stopped: the error, and what landed before it. Every
/// object in [`landed`](Self::landed) is whole and in the WAL, one after the
/// other.
#[derive(Debug)]
pub struct StreamError {
    /// What was written before the error.
    pub landed: StreamReport,
    /// Why the stream stopped.
    pub error: ItzmaError,
}

impl std::fmt::Display for StreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} (after {} rows in {} objects)",
            self.error, self.landed.rows, self.landed.objects
        )
    }
}

impl std::error::Error for StreamError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// An encoded chunk on its way to the writer.
struct Encoded {
    object: Object,
    rows: u64,
}

enum Object {
    Plain(Bytes),
    /// The chunk's own key, `<key>#<n>`, and its object.
    Keyed(String, wal_codec::Keyed),
}

/// Run a stream: an encode stage and a write stage joined in one future,
/// with a channel of `depth` encoded chunks between them. The write stage
/// puts one object at a time; when the channel is full the encode stage
/// waits, and the input is not polled.
pub(crate) async fn run<S>(
    ns: &Namespace,
    input: S,
    options: &StreamOptions,
    key: Option<&WriteKey>,
) -> std::result::Result<StreamReport, StreamError>
where
    S: Stream<Item = Result<DocColumns>> + Send,
{
    let refuse = |error| StreamError {
        landed: StreamReport::default(),
        error,
    };
    options.validate().map_err(refuse)?;
    if let Some(write) = key {
        write.validate().map_err(refuse)?;
        if write.key.len() > MAX_STREAM_KEY {
            return Err(refuse(ItzmaError::InvalidKey(
                "a stream key must be 1 to 107 bytes, leaving room for the chunk number",
            )));
        }
    }
    let (tx, rx) = tokio::sync::mpsc::channel(options.depth);
    let (encoded, (landed, written)) = futures::join!(
        encode_stage(ns.schema(), input, options.target_bytes, key, tx),
        write_stage(ns, rx, key.and_then(|write| write.search_from)),
    );
    // A write error is always about an earlier chunk than an encode error:
    // once the writer stops, the encoder's next send fails and it stops too.
    match (written, encoded) {
        (Err(error), _) | (Ok(()), Err(Some(error))) => Err(StreamError { landed, error }),
        _ => Ok(landed),
    }
}

/// Cut the input into chunks, check and encode each, and send it on. `Err(None)`
/// means the writer stopped first, which is its error to report.
async fn encode_stage<S>(
    schema: &Schema,
    input: S,
    target: usize,
    key: Option<&WriteKey>,
    tx: tokio::sync::mpsc::Sender<Encoded>,
) -> std::result::Result<(), Option<ItzmaError>>
where
    S: Stream<Item = Result<DocColumns>> + Send,
{
    let mut input = std::pin::pin!(input);
    let mut chunker = Chunker::new(target as u64);
    let mut chunks = 0u64;
    loop {
        // Wait for the next batch or for the writer to stop, whichever comes
        // first: a writer that failed must not wait on a source that may
        // take long, or forever, to fill another chunk.
        let next = {
            let closed = std::pin::pin!(tx.closed());
            match futures::future::select(input.next(), closed).await {
                futures::future::Either::Left((next, _)) => next,
                futures::future::Either::Right(_) => return Err(None),
            }
        };
        let Some(batch) = next else {
            break;
        };
        for chunk in chunker.push(batch?)? {
            send(&tx, encode(schema, chunk, key, chunks).await?).await?;
            chunks += 1;
        }
    }
    if let Some(chunk) = chunker.finish() {
        send(&tx, encode(schema, chunk, key, chunks).await?).await?;
        chunks += 1;
    }
    if chunks == 0 {
        return Err(Some(ItzmaError::EmptyBatch));
    }
    Ok(())
}

async fn send(
    tx: &tokio::sync::mpsc::Sender<Encoded>,
    encoded: Encoded,
) -> std::result::Result<(), Option<ItzmaError>> {
    tx.send(encoded).await.map_err(|_| None)
}

/// Check one chunk and encode it on the compute pool; a keyed stream's
/// chunk `n` is keyed `<key>#<n>`.
async fn encode(
    schema: &Schema,
    chunk: DocColumns,
    key: Option<&WriteKey>,
    n: u64,
) -> Result<Encoded> {
    let rows = chunk.rows() as u64;
    // A rough size: the vectors and a little for each row's id and
    // attributes. It only decides whether to hand the work off.
    let cost = Cost::Bytes(chunk.vectors().len() * 4 + chunk.rows() * 64);
    let encode_key = key.map(|write| format!("{}#{n}", write.key));
    let schema = schema.clone();
    // The checks run on the pool too: a chunk's per-row checks (cosine norms
    // among them) are as much work as its encoding.
    let encoded = cpu::run(Lane::Query, cost, move || {
        chunk.validate(&schema)?;
        match encode_key {
            None => wal_codec::encode_columns(&chunk, wal::MAX_BODY_BYTES).map(Object::Plain),
            Some(key) => wal_codec::encode_columns_keyed(&chunk, wal::MAX_BODY_BYTES, &key)
                .map(|object| Object::Keyed(key, object)),
        }
    })
    .await?;
    Ok(Encoded {
        object: encoded,
        rows,
    })
}

/// Put each encoded chunk, one at a time, and count what landed. Stops at
/// the first error, which closes the channel and so stops the encoder.
///
/// A keyed retry looks for each chunk from `search_from` on, and moves that
/// point to just past each chunk it lands or finds: a stream's chunks are in
/// the WAL in order, so chunk `n + 1` is never before chunk `n`. The first
/// chunk's search covers the WAL from where the caller says the first
/// attempt started; every later one only what follows the chunk before it.
async fn write_stage(
    ns: &Namespace,
    mut rx: tokio::sync::mpsc::Receiver<Encoded>,
    mut search_from: Option<u64>,
) -> (StreamReport, Result<()>) {
    let mut report = StreamReport::default();
    while let Some(chunk) = rx.recv().await {
        let landed = match chunk.object {
            Object::Plain(bytes) => ns.append_bytes(bytes).await,
            Object::Keyed(key, object) => {
                let write = WriteKey { key, search_from };
                ns.append_bytes_keyed(&write, object).await
            }
        };
        match landed {
            Ok(seq) => {
                search_from = search_from.map(|_| seq + 1);
                report.first_seq.get_or_insert(seq);
                report.last_seq = Some(seq);
                report.rows += chunk.rows;
                report.objects += 1;
            }
            Err(error) => return (report, Err(error)),
        }
    }
    (report, Ok(()))
}

/// Cuts a stream of batches into chunks of about `target` encoded bytes.
///
/// Rows are added one at a time, and a row that would take the chunk past
/// `target` starts the next chunk, unless the chunk is still empty (a row
/// bigger than `target` is a chunk of its own). So where the chunks fall
/// depends only on the rows, never on how the input was batched, which a
/// keyed stream needs: a retry must cut the same chunks to find them by key.
pub(crate) struct Chunker {
    target: u64,
    pending: Option<DocColumns>,
    /// The body bytes `pending` would encode to, counts included.
    bytes: u64,
    scratch: Vec<u8>,
}

impl Chunker {
    pub(crate) fn new(target: u64) -> Self {
        Self {
            target,
            pending: None,
            bytes: COUNTS_LEN as u64,
            scratch: Vec::new(),
        }
    }

    /// Take `cols` in and return every chunk it completed, in order.
    ///
    /// # Errors
    ///
    /// [`ItzmaError::InvalidColumns`](crate::ItzmaError::InvalidColumns) if
    /// `cols` does not have the columns of the batches before it.
    pub(crate) fn push(&mut self, mut cols: DocColumns) -> Result<Vec<DocColumns>> {
        let mut rows_in_chunk = self.pending.as_ref().map_or(0, DocColumns::rows);
        let mut cuts = Vec::new();
        for row in 0..cols.rows() {
            let size = wal_codec::row_body_bytes(&cols, row, &mut self.scratch);
            if rows_in_chunk > 0 && self.bytes + size > self.target {
                cuts.push(row);
                self.bytes = COUNTS_LEN as u64;
                rows_in_chunk = 0;
            }
            self.bytes += size;
            rows_in_chunk += 1;
        }
        // Cut from the back: each `split_off` then copies only its own piece,
        // so a batch many times the target is copied about once. Cutting
        // from the front copied the whole rest of the batch for every chunk
        // and left each chunk holding the batch's capacity.
        let mut pieces = Vec::with_capacity(cuts.len());
        for &cut in cuts.iter().rev() {
            pieces.push(cols.split_off(cut));
        }
        pieces.reverse();
        cols.shrink_to_fit();
        let mut done = Vec::with_capacity(cuts.len());
        if cols.rows() > 0 {
            self.add(cols)?;
        }
        for piece in pieces {
            done.extend(self.pending.take());
            self.pending = Some(piece);
        }
        Ok(done)
    }

    /// The last, partly filled chunk, if any rows are left.
    pub(crate) fn finish(self) -> Option<DocColumns> {
        self.pending
    }

    fn add(&mut self, cols: DocColumns) -> Result<()> {
        match &mut self.pending {
            Some(pending) => pending.append(cols),
            None => {
                self.pending = Some(cols);
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::columns::{AttrColumn, DocColumns};
    use crate::wal_codec::columns_body_size;

    /// Rows `from..to` of a fixed table: ids and a text attribute of varied
    /// lengths, so rows have different sizes.
    fn rows(from: usize, to: usize) -> DocColumns {
        let (mut ids, mut id_offsets) = (Vec::new(), vec![0u32]);
        let (mut text, mut text_offsets) = (Vec::new(), vec![0u32]);
        let mut vectors = Vec::new();
        for i in from..to {
            ids.extend_from_slice(format!("id-{i}-{}", "x".repeat(i % 37)).as_bytes());
            id_offsets.push(ids.len() as u32);
            text.extend_from_slice("t".repeat((i * 7) % 300).as_bytes());
            text_offsets.push(text.len() as u32);
            vectors.extend_from_slice(&[i as f32, 1.0, 2.0, 3.0]);
        }
        let n = to - from;
        DocColumns::new(ids, id_offsets, vectors, 4)
            .unwrap()
            .with_attr(
                "note",
                AttrColumn::Text {
                    valid: (from..to).map(|i| i % 5 != 0).collect(),
                    bytes: text,
                    offsets: text_offsets,
                },
            )
            .unwrap()
            .with_attr("n", AttrColumn::Null { rows: n })
            .unwrap()
    }

    /// Feed `batches` through a chunker and return each chunk's ids.
    fn chunk(target: u64, batches: Vec<DocColumns>) -> Vec<Vec<String>> {
        let mut chunker = Chunker::new(target);
        let mut out = Vec::new();
        for batch in batches {
            out.extend(chunker.push(batch).unwrap());
        }
        out.extend(chunker.finish());
        out.iter()
            .map(|c| (0..c.rows()).map(|r| c.id(r).to_string()).collect())
            .collect()
    }

    const TOTAL: usize = 1_000;
    const TARGET: u64 = 10 * 1024;

    #[test]
    fn chunks_do_not_depend_on_how_the_input_was_batched() {
        let whole = chunk(TARGET, vec![rows(0, TOTAL)]);
        let singles = chunk(TARGET, (0..TOTAL).map(|i| rows(i, i + 1)).collect());
        let mut state = 7u64;
        let mut ragged = Vec::new();
        let mut at = 0;
        while at < TOTAL {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let next = (at + 1 + (state >> 33) as usize % 90).min(TOTAL);
            ragged.push(rows(at, next));
            at = next;
        }
        let ragged = chunk(TARGET, ragged);
        assert!(whole.len() > 5, "{} chunks", whole.len());
        assert_eq!(whole, singles);
        assert_eq!(whole, ragged);
        let all: Vec<String> = whole.concat();
        assert_eq!(all.len(), TOTAL);
    }

    #[test]
    fn every_chunk_stays_under_the_target_unless_it_is_one_row() {
        let mut chunker = Chunker::new(TARGET);
        let mut chunks = chunker.push(rows(0, TOTAL)).unwrap();
        chunks.extend(chunker.finish());
        for c in &chunks {
            assert!(
                columns_body_size(c) <= TARGET || c.rows() == 1,
                "{} bytes in {} rows",
                columns_body_size(c),
                c.rows()
            );
        }
    }

    #[test]
    fn a_row_bigger_than_the_target_is_its_own_chunk() {
        let mut chunker = Chunker::new(64);
        let mut chunks = chunker.push(rows(0, 3)).unwrap();
        chunks.extend(chunker.finish());
        assert_eq!(
            chunks.iter().map(DocColumns::rows).collect::<Vec<_>>(),
            [1, 1, 1]
        );
    }

    #[test]
    fn empty_batches_and_an_empty_stream_make_no_chunk() {
        let mut chunker = Chunker::new(TARGET);
        assert!(chunker.push(rows(0, 0)).unwrap().is_empty());
        assert!(chunker.finish().is_none());
    }

    #[test]
    fn append_rejects_other_columns() {
        let mut a = rows(0, 2);
        let other_dim = DocColumns::new(b"z".to_vec(), vec![0, 1], vec![1.0], 1).unwrap();
        assert!(a.append(other_dim).is_err());
        let fewer_columns = DocColumns::new(b"z".to_vec(), vec![0, 1], vec![1.0; 4], 4).unwrap();
        assert!(a.append(fewer_columns).is_err());
    }

    #[test]
    fn a_null_column_then_a_typed_one_appends() {
        let typed = |from: usize, to: usize| {
            let n = to - from;
            let mut c = rows(from, to);
            c = c
                .without_attr("n")
                .with_attr(
                    "n",
                    AttrColumn::I64 {
                        valid: vec![true; n],
                        values: (from..to).map(|i| i as i64).collect(),
                    },
                )
                .unwrap();
            c
        };
        let mut nulls_first = rows(0, 2);
        nulls_first.append(typed(2, 4)).unwrap();
        let mut typed_first = typed(0, 2);
        typed_first.append(rows(2, 4)).unwrap();
        let kinds = |c: &DocColumns| {
            c.attrs()
                .iter()
                .find(|(name, _)| name == "n")
                .map(|(_, col)| (0..4).map(|r| col.is_null(r)).collect::<Vec<_>>())
                .unwrap()
        };
        assert_eq!(kinds(&nulls_first), [true, true, false, false]);
        assert_eq!(kinds(&typed_first), [false, false, true, true]);
    }

    #[test]
    fn split_off_then_append_restores_the_batch() {
        let whole = rows(0, 50);
        for at in [0, 1, 17, 49, 50] {
            let mut head = whole.clone();
            let tail = head.split_off(at);
            assert_eq!((head.rows(), tail.rows()), (at, 50 - at));
            head.append(tail).unwrap();
            assert_eq!(head, whole, "split at {at}");
        }
    }
}
