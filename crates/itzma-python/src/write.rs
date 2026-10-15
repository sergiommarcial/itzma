//! Writes: splitting big unkeyed loads into WAL-sized batches.

use std::sync::Arc;

use arrow_array::RecordBatchReader;
use itzma_core::{DocColumns, Document, ItzmaError, Namespace, StreamOptions, WriteKey};
use pyo3::prelude::*;

use crate::convert::arrow;
use crate::error;

/// Half the WAL's 64 MiB body limit, so encoding overhead never pushes a
/// batch over it (`wal::MAX_BODY_BYTES`).
pub(crate) const BATCH_TARGET_BYTES: usize = itzma_core::wal::MAX_BODY_BYTES / 2;

/// A rough encoded size: what dominates a WAL batch.
pub(crate) fn doc_size(doc: &Document) -> usize {
    let attrs = doc.attrs.as_ref().map_or(0, |a| a.to_string().len());
    doc.id.len() + doc.vector.len() * 4 + attrs + 16
}

/// Split `items` into runs of at most `target` bytes. An empty input stays
/// one empty batch, so the core reports `EmptyBatch` as it would unsplit.
pub(crate) fn chunks<T>(items: Vec<T>, size: impl Fn(&T) -> usize, target: usize) -> Vec<Vec<T>> {
    let mut out = Vec::new();
    let mut current = Vec::new();
    let mut bytes = 0;
    for item in items {
        let item_bytes = size(&item);
        if !current.is_empty() && bytes + item_bytes > target {
            out.push(std::mem::take(&mut current));
            bytes = 0;
        }
        bytes += item_bytes;
        current.push(item);
    }
    if !current.is_empty() || out.is_empty() {
        out.push(current);
    }
    out
}

fn precheck(ns: &Namespace, docs: &[Document]) -> itzma_core::Result<()> {
    for doc in docs {
        ns.schema().validate(&doc.vector)?;
    }
    Ok(())
}

pub(crate) enum Op {
    Upsert(Vec<Document>),
    Delete(Vec<String>),
}

/// The idempotency key, if any. `retry_from` without a key is refused.
pub(crate) fn write_key(
    key: Option<String>,
    retry_from: Option<u64>,
) -> PyResult<Option<WriteKey>> {
    match (key, retry_from) {
        (Some(key), Some(from)) => Ok(Some(WriteKey::retry(key, from))),
        (Some(key), None) => Ok(Some(WriteKey::new(key))),
        (None, Some(_)) => Err(error::input("retry_from needs a key".into())),
        (None, None) => Ok(None),
    }
}

/// Write `op`. A keyed write is one batch, since a key names one batch.
/// An unkeyed one is split, and a failure part way raises with
/// `last_seq` set to the last batch that landed.
pub(crate) async fn write(ns: Arc<Namespace>, op: Op, key: Option<WriteKey>) -> PyResult<u64> {
    if let Some(key) = key {
        let result = match op {
            Op::Upsert(docs) => ns.upsert_keyed(&key, docs).await,
            Op::Delete(ids) => ns.delete_keyed(&key, ids).await,
        };
        return result.map_err(|e| error::with_last_seq(error::to_py(e), None));
    }
    if let Op::Upsert(docs) = &op {
        // Check every vector before the first batch is sent, so bad input
        // anywhere in a split load writes nothing. Attributes are still
        // checked per batch by the core, whose attribute check is private.
        precheck(&ns, docs).map_err(|e| error::with_last_seq(error::to_py(e), None))?;
    }
    let batches: Vec<Op> = match op {
        Op::Upsert(docs) => chunks(docs, doc_size, BATCH_TARGET_BYTES)
            .into_iter()
            .map(Op::Upsert)
            .collect(),
        Op::Delete(ids) => chunks(ids, |id| id.len() + 16, BATCH_TARGET_BYTES)
            .into_iter()
            .map(Op::Delete)
            .collect(),
    };
    let mut last = None;
    for batch in batches {
        let result = match batch {
            Op::Upsert(docs) => ns.upsert(docs).await,
            Op::Delete(ids) => ns.delete(ids).await,
        };
        match result {
            Ok(seq) => last = Some(seq),
            Err(e) => return Err(error::with_last_seq(error::to_py(e), last)),
        }
    }
    last.ok_or_else(|| error::itzma_error("a write produced no batch".into()))
}

/// What `columns_from_batch` refused, kept for the error the caller sees:
/// the core only learns that the input failed.
type Refusal = Arc<std::sync::Mutex<Option<String>>>;

/// The reason the reader sends the core in place of a batch it could not
/// convert. Only an error with exactly this reason is the reader's own.
const CONVERSION_FAILED: &str = "the Arrow input could not be converted";

/// The bridge's message for `error`, when `error` is the reader's own
/// refusal. The core can also refuse a batch with `InvalidColumns` for its
/// own reasons while the reader, a batch ahead, has stored a message about
/// a later one; that message must not replace the core's error.
fn refusal_for(error: &ItzmaError, stored: Option<String>) -> Option<String> {
    match error {
        ItzmaError::InvalidColumns(reason) if *reason == CONVERSION_FAILED => stored,
        _ => None,
    }
}

/// Write an Arrow stream with `upsert_stream`. A blocking thread
/// reads the record batches (the Arrow C stream takes the GIL itself when it
/// calls into Python; this thread holds none) and converts each to columns,
/// one batch ahead of the writer at most. It stops when the write stops,
/// since its next send then fails. Returns the last seq written; a failure
/// carries `last_seq` and `rows_written`.
pub(crate) async fn write_stream(
    ns: Arc<Namespace>,
    reader: Box<dyn RecordBatchReader + Send>,
    key: Option<WriteKey>,
    options: StreamOptions,
) -> PyResult<u64> {
    let (tx, rx) = tokio::sync::mpsc::channel::<itzma_core::Result<DocColumns>>(1);
    let refusal = Refusal::default();
    let refused = refusal.clone();
    // Detached: it ends on its own once the receiver is gone.
    drop(tokio::task::spawn_blocking(move || {
        for batch in reader {
            let converted = batch
                .map_err(|e| format!("reading the Arrow stream: {e}"))
                .and_then(|batch| arrow::columns_from_batch(&batch));
            let item = converted.map_err(|message| {
                if let Ok(mut slot) = refused.lock() {
                    *slot = Some(message);
                }
                ItzmaError::InvalidColumns(CONVERSION_FAILED)
            });
            let last = item.is_err();
            if tx.blocking_send(item).is_err() || last {
                break;
            }
        }
    }));
    let input = futures::stream::unfold(rx, |mut rx| async {
        rx.recv().await.map(|item| (item, rx))
    });
    let result = match &key {
        Some(key) => ns.upsert_stream_keyed(key, input, &options).await,
        None => ns.upsert_stream(input, &options).await,
    };
    match result {
        Ok(report) => report
            .last_seq
            .ok_or_else(|| error::itzma_error("a write produced no batch".into())),
        Err(failed) => {
            let stored = refusal.lock().ok().and_then(|mut slot| slot.take());
            let err = match refusal_for(&failed.error, stored) {
                Some(message) => error::input(message),
                None => error::to_py(failed.error),
            };
            Err(error::with_progress(
                err,
                failed.landed.first_seq,
                failed.landed.last_seq,
                failed.landed.rows,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use itzma_core::Document;

    use super::*;

    #[test]
    fn only_the_bridges_own_refusal_uses_its_message() {
        let message = || Some("row 3: id is null".to_string());
        let own = ItzmaError::InvalidColumns(CONVERSION_FAILED);
        assert_eq!(refusal_for(&own, message()), message());
        // The core refusing a batch for its own reason keeps its own error,
        // even if the reader had stored a message for a later batch.
        let core = ItzmaError::InvalidColumns("batches in one stream must have the same columns");
        assert_eq!(refusal_for(&core, message()), None);
        assert_eq!(refusal_for(&ItzmaError::EmptyBatch, message()), None);
    }

    #[test]
    fn empty_input_stays_one_empty_batch() {
        let out: Vec<Vec<u8>> = chunks(Vec::new(), |_| 1, 10);
        assert_eq!(out, vec![Vec::<u8>::new()]);
    }

    #[test]
    fn splits_at_the_target() {
        let out = chunks(vec![4, 4, 4, 4, 4], |n| *n, 10);
        assert_eq!(out, vec![vec![4, 4], vec![4, 4], vec![4]]);
    }

    #[test]
    fn an_item_over_the_target_goes_alone() {
        let out = chunks(vec![1, 50, 1], |n| *n, 10);
        assert_eq!(out, vec![vec![1], vec![50], vec![1]]);
    }

    #[test]
    fn doc_size_counts_every_part() {
        let doc = Document {
            id: "abc".into(),
            vector: vec![0.0; 4],
            attrs: Some(serde_json::json!({"a": 1})),
        };
        assert_eq!(doc_size(&doc), 3 + 16 + r#"{"a":1}"#.len() + 16);
    }
}
