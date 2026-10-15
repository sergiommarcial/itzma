//! `Namespace::upsert_stream`: a stream of columnar batches lands
//! as WAL-sized chunks, in order, one PUT at a time, and a stopped stream can
//! be resumed without losing or doubling a chunk.

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use common::faulty::{Fault, FaultyStore};
use common::{assert_consistent, cas_backends};
use futures::stream::{self, BoxStream, StreamExt};
use itzma_core::{
    AttrColumn, AttrSchema, AttrType, DocColumns, Document, ItzmaError, Metric, Namespace, Schema,
    StreamOptions, WriteKey,
};
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult,
};
use serde_json::json;

const DIM: usize = 4;

fn schema() -> Schema {
    Schema::new(DIM, Metric::L2).with_attrs(
        AttrSchema::new()
            .field("price", AttrType::Number)
            .field("tag", AttrType::Text),
    )
}

fn vector(i: usize) -> Vec<f32> {
    let x = i as f32;
    vec![x.sin(), x.cos(), (x * 0.3).sin(), 1.0]
}

fn tag(i: usize) -> String {
    format!("tag-{}", i % 13)
}

/// Rows `from..to` as one columnar batch. Row `i` has id `d{i}`, a price
/// (null on every seventh row) and a tag.
fn batch(from: usize, to: usize) -> DocColumns {
    let (mut ids, mut id_offsets) = (Vec::new(), vec![0u32]);
    let (mut tags, mut tag_offsets) = (Vec::new(), vec![0u32]);
    let mut vectors = Vec::new();
    for i in from..to {
        ids.extend_from_slice(format!("d{i}").as_bytes());
        id_offsets.push(ids.len() as u32);
        tags.extend_from_slice(tag(i).as_bytes());
        tag_offsets.push(tags.len() as u32);
        vectors.extend(vector(i));
    }
    DocColumns::new(ids, id_offsets, vectors, DIM)
        .unwrap()
        .with_attr(
            "price",
            AttrColumn::F64 {
                valid: (from..to).map(|i| i % 7 != 0).collect(),
                values: (from..to).map(|i| i as f64 / 4.0).collect(),
            },
        )
        .unwrap()
        .with_attr(
            "tag",
            AttrColumn::Text {
                valid: vec![true; to - from],
                bytes: tags,
                offsets: tag_offsets,
            },
        )
        .unwrap()
}

/// What a naive replay holds after rows `0..n`.
fn model(n: usize) -> Vec<Document> {
    (0..n)
        .map(|i| Document {
            id: format!("d{i}"),
            vector: vector(i),
            attrs: Some(json!({
                "price": if i % 7 == 0 { serde_json::Value::Null } else { json!(i as f64 / 4.0) },
                "tag": tag(i),
            })),
        })
        .collect()
}

/// Rows `from..to` in batches of `size`.
fn input(
    from: usize,
    to: usize,
    size: usize,
) -> BoxStream<'static, itzma_core::Result<DocColumns>> {
    stream::iter((from..to).step_by(size))
        .map(move |at| Ok(batch(at, (at + size).min(to))))
        .boxed()
}

fn small(target: usize) -> StreamOptions {
    StreamOptions::default().target_bytes(target)
}

async fn create(store: Arc<dyn ObjectStore>) -> Namespace {
    Namespace::create(store, "docs", schema()).await.unwrap()
}

/// Every stored document, sorted by id, for comparing with the model.
fn sorted(mut docs: Vec<Document>) -> Vec<Document> {
    docs.sort_by(|a, b| a.id.cmp(&b.id));
    docs
}

#[tokio::test]
async fn many_small_batches_land_as_few_objects_in_order() {
    let store = common::memory();
    let ns = create(store.clone()).await;
    let report = ns
        .upsert_stream(input(0, 2_000, 5), &small(64 * 1024))
        .await
        .unwrap();
    assert_eq!(report.rows, 2_000);
    assert!(report.objects > 1 && report.objects < 400, "{report:?}");
    assert_eq!(report.first_seq, Some(1));
    assert_eq!(report.last_seq, Some(report.objects));
    let log = itzma_core::wal::read_log(&*store, "docs", 0).await.unwrap();
    assert_eq!(log.len() as u64, report.objects);
    let stored = assert_consistent(store, DIM, Metric::L2).await;
    assert_eq!(sorted(stored), sorted(model(2_000)));
}

/// A store that sleeps in every put and records how many were ever in
/// flight at once.
#[derive(Debug)]
struct SlowPuts {
    inner: Arc<dyn ObjectStore>,
    in_flight: AtomicUsize,
    most: AtomicUsize,
}

impl std::fmt::Display for SlowPuts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SlowPuts")
    }
}

#[async_trait]
impl ObjectStore for SlowPuts {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.most.fetch_max(now, Ordering::SeqCst);
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        let result = self.inner.put_opts(location, payload, opts).await;
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        result
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
        self.inner.get_opts(location, options).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner.list(prefix)
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner.list_with_offset(prefix, offset)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> object_store::Result<ListResult> {
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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn one_put_in_flight() {
    let slow = Arc::new(SlowPuts {
        inner: common::memory(),
        in_flight: AtomicUsize::new(0),
        most: AtomicUsize::new(0),
    });
    let ns = create(slow.clone()).await;
    slow.most.store(0, Ordering::SeqCst);
    let report = ns
        .upsert_stream(input(0, 3_000, 50), &small(8 * 1024).depth(4))
        .await
        .unwrap();
    assert!(report.objects >= 20, "{report:?}");
    assert_eq!(slow.most.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn a_failed_check_mid_stream_keeps_the_chunks_before_it() {
    let store = common::memory();
    let ns = create(store.clone()).await;
    let bad_row = 3_000;
    let rows = stream::iter((0..5_000).step_by(100))
        .map(move |at| {
            let mut cols = batch(at, at + 100);
            if (at..at + 100).contains(&bad_row) {
                cols = poisoned(at, bad_row);
            }
            Ok(cols)
        })
        .boxed();
    let err = ns.upsert_stream(rows, &small(16 * 1024)).await.unwrap_err();
    assert!(matches!(err.error, ItzmaError::InvalidVector(_)), "{err:?}");
    let landed = err.landed.rows as usize;
    assert!(landed > 0 && landed <= bad_row, "{err:?}");
    let stored = assert_consistent(store, DIM, Metric::L2).await;
    assert_eq!(sorted(stored), sorted(model(landed)));
}

/// Rows `at..at + 100` with a NaN in row `bad`'s vector.
fn poisoned(at: usize, bad: usize) -> DocColumns {
    let (mut ids, mut id_offsets) = (Vec::new(), vec![0u32]);
    let (mut tags, mut tag_offsets) = (Vec::new(), vec![0u32]);
    let mut vectors = Vec::new();
    for i in at..at + 100 {
        ids.extend_from_slice(format!("d{i}").as_bytes());
        id_offsets.push(ids.len() as u32);
        tags.extend_from_slice(tag(i).as_bytes());
        tag_offsets.push(tags.len() as u32);
        let mut v = vector(i);
        if i == bad {
            v[0] = f32::NAN;
        }
        vectors.extend(v);
    }
    DocColumns::new(ids, id_offsets, vectors, DIM)
        .unwrap()
        .with_attr(
            "price",
            AttrColumn::F64 {
                valid: (at..at + 100).map(|i| i % 7 != 0).collect(),
                values: (at..at + 100).map(|i| i as f64 / 4.0).collect(),
            },
        )
        .unwrap()
        .with_attr(
            "tag",
            AttrColumn::Text {
                valid: vec![true; 100],
                bytes: tags,
                offsets: tag_offsets,
            },
        )
        .unwrap()
}

#[tokio::test]
async fn an_input_error_stops_the_stream() {
    let store = common::memory();
    let ns = create(store.clone()).await;
    let rows = input(0, 500, 50)
        .chain(stream::iter([Err(ItzmaError::InvalidColumns(
            "from the source",
        ))]))
        .chain(input(500, 1_000, 50))
        .boxed();
    let err = ns.upsert_stream(rows, &small(8 * 1024)).await.unwrap_err();
    assert!(
        matches!(err.error, ItzmaError::InvalidColumns(_)),
        "{err:?}"
    );
    assert!(err.landed.rows <= 500, "{err:?}");
    let stored = assert_consistent(store, DIM, Metric::L2).await;
    assert_eq!(stored.len() as u64, err.landed.rows);
}

#[tokio::test]
async fn an_empty_stream_is_an_empty_batch() {
    let ns = create(common::memory()).await;
    let err = ns
        .upsert_stream(input(0, 0, 10), &StreamOptions::default())
        .await
        .unwrap_err();
    assert!(matches!(err.error, ItzmaError::EmptyBatch), "{err:?}");
    assert_eq!(err.landed, Default::default());
}

#[tokio::test]
async fn bad_options_are_refused_before_the_input_is_polled() {
    let ns = create(common::memory()).await;
    let never = || {
        stream::poll_fn(
            |_| -> std::task::Poll<Option<itzma_core::Result<DocColumns>>> {
                panic!("the input was polled")
            },
        )
    };
    for options in [
        StreamOptions::default().target_bytes(0),
        StreamOptions::default().target_bytes(itzma_core::wal::MAX_BODY_BYTES + 1),
        StreamOptions::default().depth(0),
    ] {
        let err = ns.upsert_stream(never(), &options).await.unwrap_err();
        assert!(matches!(err.error, ItzmaError::InvalidConfig(_)), "{err:?}");
    }
    // 107 bytes leaves room for `#` and any chunk number (20 digits).
    let long = WriteKey::new("k".repeat(108));
    let err = ns
        .upsert_stream_keyed(&long, never(), &StreamOptions::default())
        .await
        .unwrap_err();
    assert!(matches!(err.error, ItzmaError::InvalidKey(_)), "{err:?}");
}

#[tokio::test]
async fn the_stream_future_is_send() {
    let ns = Arc::new(create(common::memory()).await);
    let task = tokio::spawn({
        let ns = ns.clone();
        async move {
            ns.upsert_stream(input(0, 100, 10), &StreamOptions::default())
                .await
        }
    });
    assert_eq!(task.await.unwrap().unwrap().rows, 100);
}

const KEY: &str = "load/7";
const TOTAL: usize = 1_000;
const TARGET: usize = 6 * 1024;

fn dynamic(store: &Arc<FaultyStore>) -> Arc<dyn ObjectStore> {
    store.clone()
}

async fn chunk_keys(store: &dyn ObjectStore) -> Vec<String> {
    itzma_core::wal::read_log(store, "docs", 0)
        .await
        .unwrap()
        .into_iter()
        .filter_map(|(_, entry)| entry.key)
        .collect()
}

/// A keyed stream stopped at any operation, plain error or lost
/// acknowledgement, then retried with the same key from where the first
/// attempt started: every chunk lands once.
#[tokio::test]
async fn a_keyed_stream_stopped_at_any_operation_lands_each_chunk_once() {
    for ix in 0..cas_backends().len() {
        let clean = FaultyStore::new(cas_backends().swap_remove(ix).store);
        let ns = create(dynamic(&clean)).await;
        clean.reset_ops();
        let report = ns
            .upsert_stream_keyed(&WriteKey::new(KEY), input(0, TOTAL, 40), &small(TARGET))
            .await
            .unwrap();
        let total = clean.ops();
        let want = chunk_keys(&*clean).await;
        assert_eq!(want.len() as u64, report.objects);
        assert!(report.objects >= 4, "{report:?}");

        for fault in [Fault::Generic, Fault::LostAck] {
            for n in 0..total {
                eprintln!("backend {ix}: {fault:?} at operation {n} of {total}");
                let store = FaultyStore::new(cas_backends().swap_remove(ix).store);
                let ns = create(dynamic(&store)).await;
                store.fail_nth_op(n, fault);
                let first = ns
                    .upsert_stream_keyed(&WriteKey::new(KEY), input(0, TOTAL, 40), &small(TARGET))
                    .await;
                store.clear();
                // At or before the first chunk the attempt may have written.
                let from = match &first {
                    Ok(report) => report.first_seq.unwrap_or(1),
                    Err(err) => err.landed.first_seq.unwrap_or(1),
                };
                ns.upsert_stream_keyed(
                    &WriteKey::retry(KEY, from),
                    input(0, TOTAL, 40),
                    &small(TARGET),
                )
                .await
                .unwrap();
                assert_eq!(chunk_keys(&*store).await, want, "{fault:?} at {n}");
                let stored = assert_consistent(dynamic(&store), DIM, Metric::L2).await;
                assert_eq!(sorted(stored), sorted(model(TOTAL)));
            }
        }
    }
}

#[tokio::test]
async fn a_keyed_stream_resumed_with_other_batching_lands_each_chunk_once() {
    let store = FaultyStore::new(common::memory());
    let ns = create(dynamic(&store)).await;
    // Fail the third put (no reads happen in between on a fresh namespace).
    store.fail_nth_op(2, Fault::LostAck);
    let err = ns
        .upsert_stream_keyed(&WriteKey::new(KEY), input(0, TOTAL, 7), &small(TARGET))
        .await
        .unwrap_err();
    store.clear();
    let from = err.landed.first_seq.unwrap_or(1);
    let report = ns
        .upsert_stream_keyed(
            &WriteKey::retry(KEY, from),
            input(0, TOTAL, 13),
            &small(TARGET),
        )
        .await
        .unwrap();
    let keys = chunk_keys(&*store).await;
    let want: Vec<String> = (0..report.objects).map(|n| format!("{KEY}#{n}")).collect();
    assert_eq!(keys, want);
    let stored = assert_consistent(dynamic(&store), DIM, Metric::L2).await;
    assert_eq!(sorted(stored), sorted(model(TOTAL)));
}

#[tokio::test]
async fn an_unkeyed_stream_resumed_from_rows_written_matches_one_clean_run() {
    let clean = common::memory();
    create(clean.clone())
        .await
        .upsert_stream(input(0, TOTAL, 25), &small(TARGET))
        .await
        .unwrap();
    let want = sorted(assert_consistent(clean, DIM, Metric::L2).await);

    let store = FaultyStore::new(common::memory());
    let ns = create(dynamic(&store)).await;
    store.fail_nth_op(2, Fault::Generic);
    let err = ns
        .upsert_stream(input(0, TOTAL, 25), &small(TARGET))
        .await
        .unwrap_err();
    store.clear();
    let done = err.landed.rows as usize;
    assert!(done > 0 && done < TOTAL, "{err:?}");
    ns.upsert_stream(input(done, TOTAL, 25), &small(TARGET))
        .await
        .unwrap();
    assert_eq!(
        sorted(assert_consistent(dynamic(&store), DIM, Metric::L2).await),
        want
    );
}

/// A failed PUT stops the stream at once, even when the source has nothing
/// more to give: the encoder must not wait for another chunk's worth of
/// input before it learns the writer stopped.
#[tokio::test]
async fn a_failed_put_returns_even_if_the_source_stalls() {
    let store = FaultyStore::new(common::memory());
    let ns = create(dynamic(&store)).await;
    store.fail_nth_op(0, Fault::Generic);
    // About one and a half chunks, then nothing: the first chunk is sent and
    // its put fails, and the rest never fills a second chunk.
    let stalls = input(0, 100, 100).chain(stream::pending()).boxed();
    let outcome = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        ns.upsert_stream(stalls, &small(4 * 1024)),
    )
    .await
    .expect("the stream must return once its writer failed");
    let err = outcome.unwrap_err();
    assert!(matches!(err.error, ItzmaError::Storage(_)), "{err:?}");
    assert_eq!(err.landed.rows, 0);
}

/// The longest stream key still fits every chunk key, whatever the chunk
/// number: `<key>#<n>` with `n` up to `u64::MAX` is at most 128 bytes.
#[tokio::test]
async fn the_longest_stream_key_is_107_bytes() {
    let ns = create(common::memory()).await;
    let report = ns
        .upsert_stream_keyed(
            &WriteKey::new("k".repeat(107)),
            input(0, 10, 5),
            &StreamOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(report.rows, 10);
}
