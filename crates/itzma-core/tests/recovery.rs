//! Failure and recovery: a compaction or a data load stopped at any point,
//! then restarted.

mod common;

use std::collections::{BTreeMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::faulty::{Fault, FaultyStore, PutKind};
use common::{Lcg, assert_consistent, assert_untouched, cas_backends, doc, schema, snapshot};
use futures::TryStreamExt;
use itzma_core::wal;
use itzma_core::{
    Compaction, CompactionOptions, Document, GcOptions, Hit, ItzmaError, Metric, Namespace,
    manifest, segment,
};
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt};

const DIM: usize = 2;
const L2: Metric = Metric::L2;

fn dynamic(store: &Arc<FaultyStore>) -> Arc<dyn ObjectStore> {
    store.clone()
}

/// A fresh store on backend `ix` of `cas_backends()` (memory first, then S3
/// when configured), wrapped so faults can be armed on it later.
fn fresh(ix: usize) -> Arc<FaultyStore> {
    FaultyStore::new(cas_backends().swap_remove(ix).store)
}

fn ids(hits: &[Hit]) -> Vec<&str> {
    let mut ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
    ids.sort();
    ids
}

async fn wal_len(store: &dyn ObjectStore) -> usize {
    wal::read_log(store, "docs", 0).await.unwrap().len()
}

/// One WAL object's worth of writes.
#[derive(Clone)]
enum Batch {
    Put(Vec<Document>),
    Del(Vec<String>),
}

impl Batch {
    async fn write(&self, ns: &Namespace) -> itzma_core::Result<()> {
        match self {
            Batch::Put(docs) => ns.upsert(docs.clone()).await.map(|_| ()),
            Batch::Del(ids) => ns.delete(ids.clone()).await.map(|_| ()),
        }
    }

    /// The obviously-correct effect: last write wins per id.
    fn apply(&self, model: &mut BTreeMap<String, Document>) {
        match self {
            Batch::Put(docs) => {
                for d in docs {
                    model.insert(d.id.clone(), d.clone());
                }
            }
            Batch::Del(ids) => {
                for id in ids {
                    model.remove(id);
                }
            }
        }
    }
}

fn put(docs: &[(&str, f32)]) -> Batch {
    Batch::Put(docs.iter().map(|(id, x)| doc(id, &[*x, 0.0])).collect())
}

/// [`put`] with typed attributes, for [`Scenario::MergeV3`].
fn put_typed(docs: &[(&str, f32)]) -> Batch {
    Batch::Put(
        docs.iter()
            .map(|(id, x)| {
                let mut d = doc(id, &[*x, 0.0]);
                d.attrs = Some(serde_json::json!({ "n": x, "tag": id }));
                d
            })
            .collect(),
    )
}

fn del(ids: &[&str]) -> Batch {
    Batch::Del(ids.iter().map(|id| id.to_string()).collect())
}

fn values(model: &BTreeMap<String, Document>) -> Vec<Document> {
    model.values().cloned().collect()
}

async fn write_all(ns: &Namespace, model: &mut BTreeMap<String, Document>, batches: &[Batch]) {
    for batch in batches {
        batch.write(ns).await.unwrap();
        batch.apply(model);
    }
}

/// Rerun compaction until it has nothing left to do. A fold or a merge is
/// followed by `Nothing`, so four runs is slack.
async fn settle(ns: &Namespace, opts: &CompactionOptions) {
    for _ in 0..4 {
        if ns.compact(opts).await.unwrap() == Compaction::Nothing {
            return;
        }
    }
    panic!("compaction did not converge");
}

#[derive(Clone, Copy, Debug)]
enum Scenario {
    /// Four batches in the WAL, no segment yet: the fold makes a base.
    Fold,
    /// A base exists, two more batches (one a delete) follow: the fold makes a
    /// delta that keeps a tombstone.
    Delta,
    /// Two segments exist and `max_segments` is 2: the next run is a merge.
    Merge,
    /// As `Merge` with typed attributes, the first segment at format 2 and the
    /// namespace raised to 3 after it, so the merge reads both
    /// versions and writes version 3.
    MergeV3,
}

/// History that leaves compaction one interesting decision to make. Returns
/// the namespace, a model of what is committed, and the options to compact
/// with.
async fn prepare(
    store: Arc<dyn ObjectStore>,
    scenario: Scenario,
) -> (Namespace, BTreeMap<String, Document>, CompactionOptions) {
    let schema = match scenario {
        Scenario::MergeV3 => schema(DIM, L2).with_attrs(
            itzma_core::AttrSchema::new()
                .field("n", itzma_core::AttrType::Number)
                .field("tag", itzma_core::AttrType::Text),
        ),
        _ => schema(DIM, L2),
    };
    let ns = Namespace::create(store, "docs", schema).await.unwrap();
    let mut model = BTreeMap::new();
    let first = put(&[("a", 1.0), ("b", 2.0), ("c", 3.0)]);
    let second = del(&["b"]);
    let third = put(&[("c", 4.0), ("d", 5.0)]);
    let fourth = del(&["a"]);
    let roomy = CompactionOptions {
        max_segments: 8,
        max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
    };
    let tight = CompactionOptions {
        max_segments: 2,
        max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
    };
    let opts = match scenario {
        Scenario::Fold => {
            write_all(&ns, &mut model, &[first, second, third, fourth]).await;
            roomy
        }
        Scenario::Delta => {
            write_all(&ns, &mut model, &[first, second]).await;
            ns.compact(&roomy).await.unwrap();
            write_all(&ns, &mut model, &[third, fourth]).await;
            roomy
        }
        Scenario::Merge => {
            write_all(&ns, &mut model, &[first]).await;
            ns.compact(&tight).await.unwrap();
            write_all(&ns, &mut model, &[second]).await;
            ns.compact(&tight).await.unwrap();
            write_all(&ns, &mut model, &[third, fourth]).await;
            tight
        }
        Scenario::MergeV3 => {
            let first = put_typed(&[("a", 1.0), ("b", 2.0), ("c", 3.0)]);
            let third = put_typed(&[("c", 4.0), ("d", 5.0)]);
            write_all(&ns, &mut model, &[first]).await;
            ns.compact(&tight).await.unwrap();
            ns.upgrade_segment_format(3).await.unwrap();
            write_all(&ns, &mut model, &[second]).await;
            ns.compact(&tight).await.unwrap();
            write_all(&ns, &mut model, &[third, fourth]).await;
            tight
        }
    };
    (ns, model, opts)
}

/// Run `compact` with a fault at every store operation it makes, once per
/// operation and fault. After each run, with faults off and a new handle as
/// after a restart: the store is consistent and holds exactly the committed
/// documents, nothing that existed before was touched, and a rerun converges.
async fn sweep(scenario: Scenario) {
    for ix in 0..cas_backends().len() {
        let store = fresh(ix);
        let (ns, _, opts) = prepare(dynamic(&store), scenario).await;
        store.reset_ops();
        ns.compact(&opts).await.unwrap();
        let total = store.ops();
        assert!(total > 0);

        for fault in [Fault::Generic, Fault::LostAck] {
            for n in 0..total {
                eprintln!("{scenario:?}, backend {ix}: {fault:?} at operation {n} of {total}");
                let store = fresh(ix);
                let (ns, model, opts) = prepare(dynamic(&store), scenario).await;
                let before = snapshot(&*store).await;

                store.fail_nth_op(n, fault);
                let _ = ns.compact(&opts).await;
                store.clear();
                drop(ns);

                let docs = assert_consistent(dynamic(&store), DIM, L2).await;
                assert_eq!(docs, values(&model));
                assert_untouched(&*store, &before).await;

                let ns = Namespace::open(dynamic(&store), "docs").await.unwrap();
                settle(&ns, &opts).await;
                assert_eq!(
                    assert_consistent(dynamic(&store), DIM, L2).await,
                    values(&model)
                );
                let (m, _) = manifest::load(&*store, "docs").await.unwrap();
                assert!(m.live_segments.len() <= opts.max_segments);
                assert_untouched(&*store, &before).await;
            }
        }
    }
}

#[tokio::test]
async fn a_fold_survives_a_fault_at_every_step() {
    sweep(Scenario::Fold).await;
}

#[tokio::test]
async fn a_delta_fold_survives_a_fault_at_every_step() {
    sweep(Scenario::Delta).await;
}

#[tokio::test]
async fn a_merge_survives_a_fault_at_every_step() {
    sweep(Scenario::Merge).await;
}

#[tokio::test]
async fn a_merge_into_version_3_survives_a_fault_at_every_step() {
    sweep(Scenario::MergeV3).await;
}

/// Crash, restart, crash again while recovering, restart. In memory only:
/// the run count is quadratic in the operations per compaction.
#[tokio::test]
async fn a_second_fault_while_recovering_changes_nothing() {
    let scenario = Scenario::Delta;
    let store = fresh(0);
    let (ns, _, opts) = prepare(dynamic(&store), scenario).await;
    store.reset_ops();
    ns.compact(&opts).await.unwrap();
    let total = store.ops();

    for fault in [Fault::Generic, Fault::LostAck] {
        for first in 0..total {
            for second in 0..total {
                eprintln!("{fault:?}: first at {first}, second at {second} of {total}");
                let store = fresh(0);
                let (ns, model, opts) = prepare(dynamic(&store), scenario).await;
                let before = snapshot(&*store).await;

                store.fail_nth_op(first, fault);
                let _ = ns.compact(&opts).await;
                drop(ns);
                let ns = Namespace::open(dynamic(&store), "docs").await.unwrap();
                store.fail_nth_op(second, fault);
                let _ = ns.compact(&opts).await;
                store.clear();
                drop(ns);

                assert_eq!(
                    assert_consistent(dynamic(&store), DIM, L2).await,
                    values(&model)
                );
                let ns = Namespace::open(dynamic(&store), "docs").await.unwrap();
                settle(&ns, &opts).await;
                assert_eq!(
                    assert_consistent(dynamic(&store), DIM, L2).await,
                    values(&model)
                );
                assert_untouched(&*store, &before).await;
            }
        }
    }
}

const CHUNKS: usize = 6;
const PER_CHUNK: usize = 4;

fn chunk_of(prefix: &str, chunk: usize) -> Vec<Document> {
    (0..PER_CHUNK)
        .map(|i| {
            doc(
                &format!("{prefix}-{}", chunk * PER_CHUNK + i),
                &[chunk as f32, i as f32],
            )
        })
        .collect()
}

/// What is in the namespace when the load starts: a base and a delta with a
/// tombstone, so the load has published data it must leave alone.
async fn existing(
    store: Arc<dyn ObjectStore>,
    opts: &CompactionOptions,
) -> (Namespace, BTreeMap<String, Document>) {
    let ns = Namespace::create(store, "docs", schema(DIM, L2))
        .await
        .unwrap();
    let mut model = BTreeMap::new();
    let first = [
        Batch::Put(chunk_of("old", 0)),
        Batch::Put(chunk_of("old", 1)),
    ];
    write_all(&ns, &mut model, &first).await;
    ns.compact(opts).await.unwrap();
    write_all(
        &ns,
        &mut model,
        &[del(&["old-1"]), put(&[("old-extra", 7.0)])],
    )
    .await;
    ns.compact(opts).await.unwrap();
    (ns, model)
}

/// The load: chunked upserts that also replace an existing document, then
/// deletes of one of its own documents and of an existing one.
fn load_batches() -> Vec<Batch> {
    (0..CHUNKS)
        .map(|chunk| match chunk {
            1 => {
                let mut docs = chunk_of("new", chunk);
                docs.push(doc("old-3", &[99.0, 0.0]));
                Batch::Put(docs)
            }
            2 | 5 => Batch::Del(vec![
                format!("new-{}", (chunk - 1) * PER_CHUNK),
                "old-0".to_string(),
            ]),
            _ => Batch::Put(chunk_of("new", chunk)),
        })
        .collect()
}

/// Run the load from batch `start`, compacting after every second batch as a
/// real loader would. Stops at the first error, as a killed process would.
/// Returns the checkpoint: the first batch not yet acknowledged.
async fn run_load(
    ns: &Namespace,
    batches: &[Batch],
    start: usize,
    opts: &CompactionOptions,
) -> usize {
    for (i, batch) in batches.iter().enumerate().skip(start) {
        if batch.write(ns).await.is_err() {
            return i;
        }
        if i % 2 == 1 && ns.compact(opts).await.is_err() {
            return i + 1;
        }
    }
    batches.len()
}

#[derive(Clone, Copy, Debug)]
enum Restart {
    /// Replay the whole load. Upserts are idempotent, so this is safe.
    FromStart,
    /// Continue after the last batch that was acknowledged.
    FromCheckpoint,
}

/// A load is stopped at every store operation (and with a lost acknowledgement
/// at every operation), the process "restarts", and the load is run again
/// both ways. Between the stop and the restart the store holds exactly the
/// batches that landed. After the restart it holds the whole load. Nothing
/// that existed before the load is ever touched.
#[tokio::test]
async fn a_load_stopped_at_any_step_can_be_restarted() {
    let batches = load_batches();
    let opts = CompactionOptions {
        max_segments: 3,
        max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
    };
    for ix in 0..cas_backends().len() {
        let store = fresh(ix);
        let (ns, _) = existing(dynamic(&store), &opts).await;
        store.reset_ops();
        assert_eq!(run_load(&ns, &batches, 0, &opts).await, batches.len());
        let total = store.ops();

        for fault in [Fault::Generic, Fault::LostAck] {
            for n in 0..total {
                for restart in [Restart::FromStart, Restart::FromCheckpoint] {
                    eprintln!("backend {ix}: {fault:?} at operation {n} of {total}, {restart:?}");
                    let store = fresh(ix);
                    let (ns, existing_model) = existing(dynamic(&store), &opts).await;
                    let existing_wal = wal_len(&*store).await;
                    let before = snapshot(&*store).await;

                    store.fail_nth_op(n, fault);
                    let checkpoint = run_load(&ns, &batches, 0, &opts).await;
                    store.clear();
                    drop(ns);

                    // Each batch is one WAL object, so the WAL says how many
                    // landed: the acknowledged ones, plus one whose
                    // acknowledgement was lost.
                    let landed = wal_len(&*store).await - existing_wal;
                    assert!(
                        landed == checkpoint || landed == checkpoint + 1,
                        "{landed} landed, {checkpoint} acknowledged"
                    );
                    let mut model = existing_model.clone();
                    for batch in &batches[..landed] {
                        batch.apply(&mut model);
                    }
                    assert_eq!(
                        assert_consistent(dynamic(&store), DIM, L2).await,
                        values(&model)
                    );
                    assert_untouched(&*store, &before).await;

                    let ns = Namespace::open(dynamic(&store), "docs").await.unwrap();
                    let from = match restart {
                        Restart::FromStart => 0,
                        Restart::FromCheckpoint => checkpoint,
                    };
                    assert_eq!(run_load(&ns, &batches, from, &opts).await, batches.len());
                    settle(&ns, &opts).await;

                    let mut full = existing_model;
                    for batch in &batches {
                        batch.apply(&mut full);
                    }
                    assert_eq!(
                        assert_consistent(dynamic(&store), DIM, L2).await,
                        values(&full)
                    );
                    assert_untouched(&*store, &before).await;
                }
            }
        }
    }
}

/// A namespace whose last act was a merge: one base `[1, N]` and an empty
/// tail. Returns the namespace, the model, the options, and `N`.
async fn merged_history(
    store: Arc<dyn ObjectStore>,
) -> (
    Namespace,
    BTreeMap<String, Document>,
    CompactionOptions,
    u64,
) {
    let (ns, model, opts) = prepare(store.clone(), Scenario::Merge).await;
    assert!(matches!(
        ns.compact(&opts).await.unwrap(),
        Compaction::Published {
            merged_segments: 2,
            ..
        }
    ));
    let n = wal_len(&*store).await as u64;
    (ns, model, opts, n)
}

async fn reset_manifest(store: &dyn ObjectStore) {
    manifest::update(store, "docs", |m| {
        m.live_segments.clear();
        m.wal_folded_through = 0;
    })
    .await
    .unwrap();
}

async fn wipe_segments(store: &dyn ObjectStore) {
    let prefix = Path::from("docs/segments");
    let objects: Vec<_> = store.list(Some(&prefix)).try_collect().await.unwrap();
    for object in objects {
        store.delete(&object.location).await.unwrap();
    }
}

/// The WAL is the source of truth. Reset the manifest to "nothing folded" and
/// the same documents come back, and the merged base that is still in the
/// store is byte-identical to what a fold of the same range builds, or the
/// rerun would refuse it.
#[tokio::test]
async fn resetting_the_manifest_rebuilds_the_same_state_from_the_wal() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let (ns, model, opts, n) = merged_history(b.store.clone()).await;
        let base = segment::segment_id(1, n);
        reset_manifest(&*b.store).await;

        assert_eq!(
            assert_consistent(b.store.clone(), DIM, L2).await,
            values(&model)
        );
        match ns.compact(&opts).await.unwrap() {
            Compaction::Published {
                segments,
                merged_segments: 0,
                ..
            } => assert_eq!(segments, vec![base.clone()]),
            other => panic!("unexpected {other:?}"),
        }
        assert_eq!(
            assert_consistent(b.store.clone(), DIM, L2).await,
            values(&model)
        );
    }
}

/// Segments are derived data. Wipe them all: queries fail loudly while the
/// manifest still lists them, and work again after the manifest is reset.
#[tokio::test]
async fn segments_are_disposable() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let (ns, model, opts, _) = merged_history(b.store.clone()).await;
        wipe_segments(&*b.store).await;

        let err = ns.query(&[0.0, 0.0], 10).await.unwrap_err();
        assert!(matches!(err, ItzmaError::CorruptSegment { .. }), "{err:?}");

        reset_manifest(&*b.store).await;
        assert_eq!(
            assert_consistent(b.store.clone(), DIM, L2).await,
            values(&model)
        );
        assert!(matches!(
            ns.compact(&opts).await.unwrap(),
            Compaction::Published { .. }
        ));
        assert_eq!(
            assert_consistent(b.store.clone(), DIM, L2).await,
            values(&model)
        );
    }
}

/// Damage in place: queries never return a partial answer. A manifest reset
/// makes them work at once. Compacting that range again is refused until the
/// damaged object is deleted.
#[tokio::test]
async fn a_truncated_segment_fails_loudly_and_recovery_needs_two_steps() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let (ns, model, opts, n) = merged_history(b.store.clone()).await;
        let key = segment::segment_path("docs", &segment::segment_id(1, n));
        let bytes = b.store.get(&key).await.unwrap().bytes().await.unwrap();
        b.store
            .put(&key, bytes.slice(..bytes.len() / 2).into())
            .await
            .unwrap();

        let err = ns.query(&[0.0, 0.0], 10).await.unwrap_err();
        assert!(matches!(err, ItzmaError::CorruptSegment { .. }), "{err:?}");

        reset_manifest(&*b.store).await;
        let hits = ns.query(&[0.0, 0.0], 10).await.unwrap();
        assert_eq!(hits.len(), model.len());

        let err = ns.compact(&opts).await.unwrap_err();
        assert!(matches!(err, ItzmaError::CorruptSegment { .. }), "{err:?}");

        b.store.delete(&key).await.unwrap();
        assert!(matches!(
            ns.compact(&opts).await.unwrap(),
            Compaction::Published { .. }
        ));
        assert_eq!(
            assert_consistent(b.store.clone(), DIM, L2).await,
            values(&model)
        );
    }
}

/// a put that lands but is reported as failed makes the caller's
/// retry append the batch again. For an upsert that changes nothing.
#[tokio::test]
async fn a_lost_ack_makes_a_retry_append_the_batch_twice() {
    let inner = Arc::new(object_store::memory::InMemory::new());
    let store = FaultyStore::new(inner.clone());
    let ns = Namespace::create(store.clone(), "docs", schema(1, L2))
        .await
        .unwrap();
    store.fail_puts("/wal/", PutKind::Create, Fault::LostAck, Some(1));

    let batch = vec![doc("a", &[1.0])];
    assert!(ns.upsert(batch.clone()).await.is_err());
    assert_eq!(ns.upsert(batch).await.unwrap(), 2);

    assert_eq!(wal_len(&*inner).await, 2);
    assert_eq!(ids(&ns.query(&[0.0], 5).await.unwrap()), ["a"]);
}

/// Pins the at-least-once contract, including its sharp edge: a
/// replayed batch can overwrite a newer value another writer stored in the
/// meantime. If idempotency keys ever land, this test changes on purpose.
#[tokio::test]
async fn at_least_once_a_replayed_batch_can_overwrite_a_newer_write() {
    let store = FaultyStore::new(Arc::new(object_store::memory::InMemory::new()));
    let ns = Namespace::create(store.clone(), "docs", schema(1, L2))
        .await
        .unwrap();
    let other = Namespace::open(store.clone(), "docs").await.unwrap();

    store.fail_puts("/wal/", PutKind::Create, Fault::LostAck, Some(1));
    assert!(ns.upsert(vec![doc("x", &[1.0])]).await.is_err());
    other.upsert(vec![doc("x", &[2.0])]).await.unwrap();
    ns.upsert(vec![doc("x", &[1.0])]).await.unwrap();

    let hits = ns.query(&[0.0], 1).await.unwrap();
    assert_eq!(hits[0].id, "x");
    assert_eq!(hits[0].distance, 1.0, "the replay won: x is 1.0, not 2.0");
}

/// One writer, two compactors and a collector run, are killed at a seeded random point,
/// and are started again, over and over. Every store operation yields, so a
/// kill lands between two store calls, which is what killing a process looks
/// like (each call is atomic). Two compactors make the manifest swap race. The collector deletes
/// segments the compactors have superseded, at any age, so a compactor can find
/// a segment it needs gone.
/// After every round the store is consistent and every acknowledged write is
/// present. Deterministic on the current-thread runtime with the memory store.
#[tokio::test]
async fn killing_writers_and_compactors_at_random_points_never_hurts() {
    let opts = CompactionOptions {
        max_segments: 2,
        max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
    };
    for ix in 0..cas_backends().len() {
        let store = fresh(ix);
        store.yield_each_op(true);
        Namespace::create(dynamic(&store), "docs", schema(DIM, L2))
            .await
            .unwrap();
        let acknowledged = Arc::new(Mutex::new(Vec::<String>::new()));
        let mut rng = Lcg(2024);

        for round in 0..30 {
            let mut tasks = Vec::new();
            for role in 0..4 {
                let (store, acknowledged) = (dynamic(&store), acknowledged.clone());
                tasks.push(tokio::spawn(async move {
                    let ns = Namespace::open(store, "docs").await.unwrap();
                    for i in 0..=u32::MAX {
                        match role {
                            0 => {
                                let id = format!("r{round}-{i}");
                                ns.upsert(vec![doc(&id, &[round as f32, i as f32])])
                                    .await
                                    .unwrap();
                                acknowledged.lock().unwrap().push(id);
                            }
                            3 => {
                                let now = GcOptions {
                                    min_age: Duration::ZERO,
                                    dry_run: false,
                                    fence_wal: false,
                                };
                                ns.gc(&now).await.unwrap();
                            }
                            _ => {
                                ns.compact(&opts).await.unwrap();
                            }
                        }
                    }
                }));
            }

            for _ in 0..rng.below(80) {
                tokio::task::yield_now().await;
            }
            for task in &tasks {
                task.abort();
            }
            for task in tasks {
                if let Err(e) = task.await {
                    assert!(e.is_cancelled(), "round {round}: a task failed: {e}");
                }
            }

            eprintln!("backend {ix}: round {round}");
            let present: HashSet<String> = assert_consistent(dynamic(&store), DIM, L2)
                .await
                .into_iter()
                .map(|d| d.id)
                .collect();
            for id in acknowledged.lock().unwrap().iter() {
                assert!(present.contains(id), "acknowledged write {id} is gone");
            }
        }

        let ns = Namespace::open(dynamic(&store), "docs").await.unwrap();
        settle(&ns, &opts).await;
        assert_consistent(dynamic(&store), DIM, L2).await;
    }
}
