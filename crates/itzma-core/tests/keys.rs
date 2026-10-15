//! Idempotency keys through the `Namespace` API, and the manifest
//! gate's upgrade step that keyed writes depend on.

mod common;

use std::collections::BTreeMap;
use std::sync::Arc;

use common::faulty::{Fault, FaultyStore};
use common::{assert_consistent, cas_backends, doc, schema, snapshot};
use itzma_core::manifest;
use itzma_core::segment;
use itzma_core::wal::{self, FORMAT_VERSION, KEYED_FORMAT_VERSION, SPARSE_FORMAT_VERSION};
use itzma_core::{CompactionOptions, Document, ItzmaError, Metric, Namespace, WriteKey};
use object_store::{ObjectStore, ObjectStoreExt};

/// Put the manifest's `wal_format` at `level`, whatever it was. A plain put, so
/// it works on every backend. This is how a namespace from before keyed writes
/// looks.
async fn set_level(store: &dyn ObjectStore, namespace: &str, level: u32) {
    let (mut m, _) = manifest::load(store, namespace).await.unwrap();
    m.wal_format = level;
    store
        .put(
            &manifest::manifest_path(namespace),
            serde_json::to_vec(&m).unwrap().into(),
        )
        .await
        .unwrap();
}

async fn level(store: &dyn ObjectStore) -> u32 {
    manifest::load(store, "docs").await.unwrap().0.wal_format
}

/// A namespace as an older build left it: created, then put back at level 2.
async fn at_level_2(store: Arc<dyn ObjectStore>) -> Namespace {
    Namespace::create(store.clone(), "docs", schema(2, Metric::L2))
        .await
        .unwrap();
    set_level(&*store, "docs", 2).await;
    Namespace::open(store, "docs").await.unwrap()
}

/// A namespace at the fencing level (4) but below the sparse level (5): the
/// state a namespace an older build (one that knows
/// fencing but not sparse vectors) created is in.
async fn at_level_4(store: Arc<dyn ObjectStore>) -> Namespace {
    Namespace::create(store.clone(), "docs", schema(2, Metric::L2))
        .await
        .unwrap();
    set_level(&*store, "docs", 4).await;
    Namespace::open(store, "docs").await.unwrap()
}

fn too_old(err: ItzmaError) {
    assert!(
        matches!(err, ItzmaError::WalFormatTooOld { have: 2, need } if need == KEYED_FORMAT_VERSION),
        "unexpected error: {err:?}"
    );
}

fn too_old_for_sparse(err: ItzmaError) {
    assert!(
        matches!(err, ItzmaError::WalFormatTooOld { have: 4, need } if need == SPARSE_FORMAT_VERSION),
        "unexpected error: {err:?}"
    );
}

#[tokio::test]
async fn a_namespace_below_level_3_refuses_keyed_writes_and_writes_nothing() {
    for b in common::backends() {
        eprintln!("backend: {}", b.name);
        let ns = at_level_2(b.store.clone()).await;
        let before = snapshot(&*b.store).await;

        let write = WriteKey::new("k");
        too_old(
            ns.upsert_keyed(&write, vec![doc("a", &[0.0, 0.0])])
                .await
                .unwrap_err(),
        );
        too_old(ns.delete_keyed(&write, vec!["a".into()]).await.unwrap_err());

        assert_eq!(snapshot(&*b.store).await, before, "the store changed");
    }
}

fn sparse_doc(id: &str, v: &[f32], terms: &[(u32, f32)]) -> Document {
    Document {
        id: id.into(),
        vector: v.to_vec(),
        attrs: None,
        sparse: Some(terms.to_vec()),
    }
}

#[tokio::test]
async fn a_namespace_below_level_5_refuses_a_sparse_upsert_and_writes_nothing() {
    for b in common::backends() {
        eprintln!("backend: {}", b.name);
        let ns = at_level_4(b.store.clone()).await;
        let before = snapshot(&*b.store).await;

        too_old_for_sparse(
            ns.upsert(vec![sparse_doc("a", &[0.0, 0.0], &[(3, 0.5)])])
                .await
                .unwrap_err(),
        );
        too_old_for_sparse(
            ns.upsert_keyed(
                &WriteKey::new("k"),
                vec![sparse_doc("a", &[0.0, 0.0], &[(3, 0.5)])],
            )
            .await
            .unwrap_err(),
        );

        assert_eq!(snapshot(&*b.store).await, before, "the store changed");
    }
}

#[tokio::test]
async fn a_plain_upsert_is_unaffected_by_being_below_the_sparse_level() {
    let ns = at_level_4(common::memory()).await;
    // No sparse vector in this batch, so the sparse gate never applies, even
    // though the namespace is below level 5.
    ns.upsert(vec![doc("a", &[0.0, 0.0])]).await.unwrap();
}

#[tokio::test]
async fn upgrading_to_the_sparse_level_allows_a_sparse_upsert() {
    for b in common::cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = at_level_4(b.store.clone()).await;
        ns.upgrade_wal_format(SPARSE_FORMAT_VERSION).await.unwrap();

        ns.upsert(vec![sparse_doc("a", &[0.0, 0.0], &[(3, 0.5)])])
            .await
            .unwrap();
        assert_eq!(level(&*b.store).await, SPARSE_FORMAT_VERSION);
    }
}

#[tokio::test]
async fn a_bad_key_is_refused_before_the_manifest_is_read() {
    let store = FaultyStore::new(common::memory());
    let ns = at_level_2(store.clone()).await;
    store.reset_ops();

    let err = ns
        .upsert_keyed(&WriteKey::new(""), vec![doc("a", &[0.0, 0.0])])
        .await
        .unwrap_err();
    assert!(matches!(err, ItzmaError::InvalidKey(_)), "{err:?}");
    let err = ns
        .upsert_keyed(&WriteKey::new("k"), vec![])
        .await
        .unwrap_err();
    assert!(matches!(err, ItzmaError::EmptyBatch), "{err:?}");
    let err = ns
        .upsert_keyed(&WriteKey::new("k"), vec![doc("a", &[0.0])])
        .await
        .unwrap_err();
    assert!(
        matches!(err, ItzmaError::DimensionMismatch { .. }),
        "{err:?}"
    );
    assert_eq!(store.ops(), 0, "a storage call was made");
}

/// Writers do not read the manifest. Keyed writes only do
/// when the handle does not yet know the level is high enough.
#[tokio::test]
async fn unkeyed_writes_still_cost_one_put_and_read_no_manifest() {
    let store = FaultyStore::new(common::memory());
    let ns = at_level_2(store.clone()).await;
    store.reset_ops();

    ns.upsert(vec![doc("a", &[0.0, 0.0])]).await.unwrap();
    assert_eq!(store.ops(), 1);
    ns.delete(vec!["a".into()]).await.unwrap();
    assert_eq!(store.ops(), 2);
}

#[tokio::test]
async fn upgrading_raises_the_level_once_and_never_lowers_it() {
    for b in common::cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = at_level_2(b.store.clone()).await;

        ns.upgrade_wal_format(KEYED_FORMAT_VERSION).await.unwrap();
        assert_eq!(level(&*b.store).await, KEYED_FORMAT_VERSION);

        // Asking again, or for less, changes nothing and writes nothing.
        let raised = snapshot(&*b.store).await;
        ns.upgrade_wal_format(KEYED_FORMAT_VERSION).await.unwrap();
        ns.upgrade_wal_format(2).await.unwrap();
        assert_eq!(snapshot(&*b.store).await, raised);

        // A level this build cannot read is refused, and nothing is written.
        let err = ns.upgrade_wal_format(FORMAT_VERSION + 1).await.unwrap_err();
        assert!(
            matches!(
                err,
                ItzmaError::WalFormatTooNew { found, supported }
                    if found == FORMAT_VERSION + 1 && supported == FORMAT_VERSION
            ),
            "{err:?}"
        );
        assert_eq!(snapshot(&*b.store).await, raised);
    }
}

#[tokio::test]
async fn after_the_upgrade_the_same_handle_writes_keyed_batches_in_one_put() {
    for b in common::cas_backends() {
        eprintln!("backend: {}", b.name);
        let store = FaultyStore::new(b.store.clone());
        let ns = at_level_2(store.clone()).await;
        ns.upgrade_wal_format(KEYED_FORMAT_VERSION).await.unwrap();
        store.reset_ops();

        let seq = ns
            .upsert_keyed(&WriteKey::new("k"), vec![doc("a", &[0.0, 0.0])])
            .await
            .unwrap();
        assert_eq!(seq, 1);
        assert_eq!(store.ops(), 1, "the handle already knows the level");
    }
}

/// Another process raised the level after this handle opened. The handle finds
/// out with one manifest read on its next keyed write, and not again.
#[tokio::test]
async fn a_handle_opened_before_the_upgrade_learns_the_level_with_one_read() {
    for b in common::cas_backends() {
        eprintln!("backend: {}", b.name);
        let store = FaultyStore::new(b.store.clone());
        let stale = at_level_2(store.clone()).await;
        let other = Namespace::open(b.store.clone(), "docs").await.unwrap();
        other
            .upgrade_wal_format(KEYED_FORMAT_VERSION)
            .await
            .unwrap();
        store.reset_ops();

        stale
            .upsert_keyed(&WriteKey::new("one"), vec![doc("a", &[0.0, 0.0])])
            .await
            .unwrap();
        assert_eq!(store.ops(), 2, "one manifest read, one put");
        stale
            .upsert_keyed(&WriteKey::new("two"), vec![doc("b", &[1.0, 1.0])])
            .await
            .unwrap();
        assert_eq!(store.ops(), 3, "the level is cached now");
    }
}

#[tokio::test]
async fn two_upgrades_racing_both_succeed() {
    for b in common::cas_backends() {
        eprintln!("backend: {}", b.name);
        let first = at_level_2(b.store.clone()).await;
        let second = Namespace::open(b.store.clone(), "docs").await.unwrap();

        let (a, c) = tokio::join!(
            first.upgrade_wal_format(KEYED_FORMAT_VERSION),
            second.upgrade_wal_format(KEYED_FORMAT_VERSION)
        );
        a.unwrap();
        c.unwrap();
        assert_eq!(level(&*b.store).await, KEYED_FORMAT_VERSION);
    }
}

#[tokio::test]
async fn a_new_namespace_takes_keyed_writes_at_once() {
    for b in common::backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", schema(2, Metric::L2))
            .await
            .unwrap();
        assert_eq!(level(&*b.store).await, FORMAT_VERSION);

        let seq = ns
            .upsert_keyed(&WriteKey::new("k"), vec![doc("a", &[0.0, 0.0])])
            .await
            .unwrap();
        assert_eq!(seq, 1);
        let log = wal::read_log(&*b.store, "docs", 0).await.unwrap();
        assert_eq!(log[0].1.format_version, 3);
        assert_eq!(log[0].1.key.as_deref(), Some("k"));
    }
}

#[tokio::test]
async fn queries_see_keyed_and_unkeyed_writes_alike() {
    for b in common::backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", schema(2, Metric::L2))
            .await
            .unwrap();
        ns.upsert(vec![doc("a", &[0.0, 0.0])]).await.unwrap();
        ns.upsert_keyed(&WriteKey::new("k1"), vec![doc("b", &[1.0, 1.0])])
            .await
            .unwrap();
        ns.upsert(vec![doc("c", &[2.0, 2.0])]).await.unwrap();
        ns.delete_keyed(&WriteKey::new("k2"), vec!["a".into()])
            .await
            .unwrap();

        let ids: Vec<String> = ns
            .query(&[0.0, 0.0], 10)
            .await
            .unwrap()
            .into_iter()
            .map(|h| h.id)
            .collect();
        assert_eq!(ids, ["b", "c"]);
    }
}

/// The harm of at-least-once writes: a replayed delete undoing a newer write. With a
/// key, the retry finds the delete already there and does nothing.
#[tokio::test]
async fn a_retried_delete_does_not_undo_a_later_upsert() {
    for b in common::backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", schema(2, Metric::L2))
            .await
            .unwrap();
        ns.upsert(vec![doc("a", &[0.0, 0.0])]).await.unwrap();
        let deleted = ns
            .delete_keyed(&WriteKey::new("drop-a"), vec!["a".into()])
            .await
            .unwrap();
        ns.upsert(vec![doc("a", &[5.0, 5.0])]).await.unwrap();
        let before = snapshot(&*b.store).await;

        let again = ns
            .delete_keyed(&WriteKey::retry("drop-a", 1), vec!["a".into()])
            .await
            .unwrap();
        assert_eq!(again, deleted);
        assert_eq!(snapshot(&*b.store).await, before);
        assert_eq!(ns.query(&[5.0, 5.0], 1).await.unwrap()[0].id, "a");
    }
}

/// Keys live only in the WAL header. A segment is built from the operations, so
/// keyed and unkeyed logs of the same operations fold into the same bytes.
/// The documented gap: two submits of one key that run at the same
/// time can both land. Here `b` collides with an unkeyed object at seq 1 and
/// reads its header. Before it lists the log, `a` (whose hint is past that
/// object) puts the same key at seq 2. `b` then lists, sees seq 2, and puts at
/// seq 3, so the batch is in the log twice. Readers see it once because an
/// upsert is idempotent by id, and a retry afterwards finds the first copy.
#[tokio::test]
async fn two_submits_of_one_key_at_once_can_both_land() {
    for backend in cas_backends() {
        eprintln!("backend: {}", backend.name);
        let raw = backend.store.clone();
        let a = Namespace::create(raw.clone(), "docs", schema(2, Metric::L2))
            .await
            .unwrap();
        a.upsert(vec![doc("x", &[0.0, 0.0])]).await.unwrap();
        let store = FaultyStore::new(raw.clone());
        let b = Namespace::open(dynamic(&store), "docs").await.unwrap();

        store.before_get("00000000000000000001.json", async move {
            let seq = a
                .upsert_keyed(&WriteKey::new("k"), vec![doc("d", &[1.0, 1.0])])
                .await
                .unwrap();
            assert_eq!(seq, 2);
        });
        let seq = b
            .upsert_keyed(&WriteKey::new("k"), vec![doc("d", &[1.0, 1.0])])
            .await
            .unwrap();
        assert_eq!(
            seq, 3,
            "b listed after a had landed and did not see the key"
        );

        let log = wal::read_log(&*raw, "docs", 0).await.unwrap();
        let keys: Vec<Option<&str>> = log.iter().map(|(_, batch)| batch.key.as_deref()).collect();
        assert_eq!(keys, [None, Some("k"), Some("k")]);
        assert_eq!(b.query(&[1.0, 1.0], 10).await.unwrap().len(), 2, "x and d");
        assert_consistent(raw.clone(), 2, Metric::L2).await;

        // A retry from seq 1 finds the first copy and writes nothing.
        let again = b
            .upsert_keyed(&WriteKey::retry("k", 1), vec![doc("d", &[1.0, 1.0])])
            .await
            .unwrap();
        assert_eq!(again, 2);
        assert_eq!(wal::read_log(&*raw, "docs", 0).await.unwrap().len(), 3);
    }
}

#[tokio::test]
async fn a_log_with_keyed_objects_folds_into_the_same_segment_as_an_unkeyed_one() {
    for b in common::cas_backends() {
        eprintln!("backend: {}", b.name);
        let keyed = Namespace::create(b.store.clone(), "docs", schema(2, Metric::L2))
            .await
            .unwrap();
        let plain_store = common::memory();
        let plain = Namespace::create(plain_store.clone(), "docs", schema(2, Metric::L2))
            .await
            .unwrap();

        keyed
            .upsert_keyed(
                &WriteKey::new("k1"),
                vec![doc("a", &[0.0, 0.0]), doc("b", &[1.0, 1.0])],
            )
            .await
            .unwrap();
        keyed
            .delete_keyed(&WriteKey::new("k2"), vec!["a".into()])
            .await
            .unwrap();
        plain
            .upsert(vec![doc("a", &[0.0, 0.0]), doc("b", &[1.0, 1.0])])
            .await
            .unwrap();
        plain.delete(vec!["a".into()]).await.unwrap();

        let options = CompactionOptions::default();
        keyed.compact(&options).await.unwrap();
        plain.compact(&options).await.unwrap();

        // The WALs differ on purpose, so only the segment is compared.
        let path = segment::segment_path("docs", &segment::segment_id(1, 2));
        let a = b.store.get(&path).await.unwrap().bytes().await.unwrap();
        let c = plain_store.get(&path).await.unwrap().bytes().await.unwrap();
        assert_eq!(a, c);
    }
}

// A keyed load stopped at any operation and resumed. This is what the keys are
// for: a lost acknowledgement, a crash, a restart, and the batch that may or
// may not have landed must end up in the log exactly once.

const BATCHES: usize = 4;
const DIM: usize = 2;

fn dynamic(store: &Arc<FaultyStore>) -> Arc<dyn ObjectStore> {
    store.clone()
}

fn fresh(ix: usize) -> Arc<FaultyStore> {
    FaultyStore::new(cas_backends().swap_remove(ix).store)
}

fn key(i: usize) -> String {
    format!("load/{i}")
}

/// Batch `i`. They all overwrite "shared", so the order they land in shows.
fn batch(i: usize) -> Vec<Document> {
    let n = i as f32;
    vec![
        doc(&format!("b{i}-x"), &[n, 0.0]),
        doc(&format!("b{i}-y"), &[n, 1.0]),
        doc("shared", &[n, 2.0]),
    ]
}

/// What a loader keeps: how many batches were acknowledged, and the seq of the
/// last one. The seed write is seq 1.
#[derive(Clone, Copy)]
struct Progress {
    done: usize,
    last_seq: u64,
}

const START: Progress = Progress {
    done: 0,
    last_seq: 1,
};

/// Write the remaining batches, each under its own key, and stop at the first
/// error. A resumed load says every write may be a retry, from just past the
/// last seq it saw acknowledged.
async fn load(ns: &Namespace, from: Progress, resumed: bool) -> Progress {
    let mut progress = from;
    while progress.done < BATCHES {
        let i = progress.done;
        let write = if resumed {
            WriteKey::retry(key(i), progress.last_seq + 1)
        } else {
            WriteKey::new(key(i))
        };
        match ns.upsert_keyed(&write, batch(i)).await {
            Ok(seq) => {
                progress.done += 1;
                progress.last_seq = seq;
            }
            Err(_) => break,
        }
    }
    progress
}

/// A namespace with one unkeyed write, folded, so the load starts past a
/// segment.
async fn seeded(store: Arc<dyn ObjectStore>) -> Namespace {
    let ns = Namespace::create(store, "docs", schema(DIM, Metric::L2))
        .await
        .unwrap();
    ns.upsert(vec![doc("seed", &[9.0, 9.0])]).await.unwrap();
    ns.compact(&CompactionOptions::default()).await.unwrap();
    ns
}

/// What a query has to return once every batch is in: the seed, then each
/// batch once, in order.
fn expected(other: bool) -> Vec<Document> {
    let mut model = BTreeMap::new();
    let mut put = |d: Document| model.insert(d.id.clone(), d);
    put(doc("seed", &[9.0, 9.0]));
    for i in 0..BATCHES {
        for d in batch(i) {
            put(d);
        }
    }
    if other {
        put(doc("other", &[7.0, 7.0]));
    }
    model.into_values().collect()
}

async fn assert_each_key_once(store: &dyn ObjectStore) {
    let log = wal::read_log(store, "docs", 0).await.unwrap();
    for i in 0..BATCHES {
        let copies = log
            .iter()
            .filter(|(_, entry)| entry.key.as_deref() == Some(key(i).as_str()))
            .count();
        assert_eq!(copies, 1, "batch {i} is in the log {copies} times");
    }
}

#[derive(Clone, Copy, Debug)]
enum Between {
    /// Nothing happens between the failure and the restart.
    Nothing,
    /// A compaction folds the log, so the original is no longer in the tail.
    Compaction,
    /// Another writer appends, so the restarted handle's hint is stale.
    OtherWriter,
}

#[tokio::test]
async fn a_keyed_load_stopped_at_any_operation_lands_every_batch_once() {
    for ix in 0..cas_backends().len() {
        let clean = fresh(ix);
        let ns = seeded(dynamic(&clean)).await;
        clean.reset_ops();
        assert_eq!(load(&ns, START, false).await.done, BATCHES);
        let total = clean.ops();
        assert!(total > 0);

        for between in [Between::Nothing, Between::Compaction, Between::OtherWriter] {
            for fault in [Fault::Generic, Fault::LostAck] {
                for n in 0..total {
                    eprintln!("backend {ix}: {fault:?} at operation {n} of {total}, {between:?}");
                    let store = fresh(ix);
                    let ns = seeded(dynamic(&store)).await;

                    store.fail_nth_op(n, fault);
                    let progress = load(&ns, START, false).await;
                    store.clear();
                    drop(ns);

                    let resumed = Namespace::open(dynamic(&store), "docs").await.unwrap();
                    match between {
                        Between::Nothing => {}
                        Between::Compaction => {
                            let opts = CompactionOptions::default();
                            resumed.compact(&opts).await.unwrap();
                        }
                        Between::OtherWriter => {
                            let other = Namespace::open(dynamic(&store), "docs").await.unwrap();
                            other.upsert(vec![doc("other", &[7.0, 7.0])]).await.unwrap();
                        }
                    }
                    let finished = load(&resumed, progress, true).await;
                    assert_eq!(finished.done, BATCHES);

                    assert_each_key_once(&*store).await;
                    let docs = assert_consistent(dynamic(&store), DIM, Metric::L2).await;
                    assert_eq!(docs, expected(matches!(between, Between::OtherWriter)));
                }
            }
        }
    }
}

/// The retry can fail too. Two failures, one after the other, and a third
/// clean run still leaves every key once.
#[tokio::test]
async fn a_resumed_load_that_fails_again_still_lands_every_batch_once() {
    let clean = fresh(0);
    let ns = seeded(dynamic(&clean)).await;
    clean.reset_ops();
    assert_eq!(load(&ns, START, false).await.done, BATCHES);
    let total = clean.ops();

    // A resumed run reads before it writes, so it is longer than a first one.
    // Run one from the start over the finished log for the longest case.
    clean.reset_ops();
    assert_eq!(load(&ns, START, true).await.done, BATCHES);
    let resumed_total = clean.ops();
    assert!(resumed_total > total);

    for fault in [Fault::Generic, Fault::LostAck] {
        for first_at in 0..total {
            // A run that scans a log with no match reads more than the
            // measured one, which finds its keys early. A fault past the end
            // of a run never fires, so overshooting costs only time.
            for second_at in 0..4 * resumed_total {
                eprintln!("{fault:?}: first at {first_at}, second at {second_at}");
                let store = fresh(0);
                let ns = seeded(dynamic(&store)).await;

                store.fail_nth_op(first_at, fault);
                let progress = load(&ns, START, false).await;
                drop(ns);

                let ns = Namespace::open(dynamic(&store), "docs").await.unwrap();
                store.fail_nth_op(second_at, fault);
                let progress = load(&ns, progress, true).await;
                store.clear();
                drop(ns);

                let ns = Namespace::open(dynamic(&store), "docs").await.unwrap();
                let finished = load(&ns, progress, true).await;
                assert_eq!(finished.done, BATCHES);

                assert_each_key_once(&*store).await;
                let docs = assert_consistent(dynamic(&store), DIM, Metric::L2).await;
                assert_eq!(docs, expected(false));
            }
        }
    }
}

/// The futures of the new methods have to be `Send`, so callers can spawn
/// them. A regression is a compile error here.
#[tokio::test]
async fn the_keyed_and_upgrade_futures_can_be_spawned() {
    let ns = Arc::new(at_level_2(common::memory()).await);

    let upgrade = {
        let ns = ns.clone();
        tokio::spawn(async move { ns.upgrade_wal_format(KEYED_FORMAT_VERSION).await })
    };
    upgrade.await.unwrap().unwrap();

    let write = {
        let ns = ns.clone();
        tokio::spawn(async move {
            ns.upsert_keyed(&WriteKey::new("k"), vec![doc("a", &[0.0, 0.0])])
                .await?;
            ns.delete_keyed(&WriteKey::retry("d", 1), vec!["a".into()])
                .await
        })
    };
    assert_eq!(write.await.unwrap().unwrap(), 2);
}

/// The version of the WAL object at `seq`, from its header.
async fn object_version(store: &dyn ObjectStore, seq: u64) -> u32 {
    let bytes = store
        .get(&wal::wal_path("docs", seq))
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    assert_eq!(&bytes[..4], b"ITZW");
    u32::from_le_bytes(bytes[4..8].try_into().unwrap())
}

/// A batch is written in the oldest version that can hold it: version 3 only
/// when it carries a key.
#[tokio::test]
async fn a_batch_is_written_in_the_oldest_version_that_holds_it() {
    let store = common::memory();
    let ns = Namespace::create(store.clone(), "docs", schema(2, Metric::L2))
        .await
        .unwrap();
    let plain = || vec![doc("a", &[0.0, 0.0])];
    let sparse = || vec![sparse_doc("a", &[0.0, 0.0], &[(3, 0.5)])];

    ns.upsert(plain()).await.unwrap();
    ns.upsert_keyed(&WriteKey::new("k1"), plain())
        .await
        .unwrap();
    ns.upsert(sparse()).await.unwrap();
    ns.upsert_keyed(&WriteKey::new("k2"), sparse())
        .await
        .unwrap();
    ns.delete(vec!["a".into()]).await.unwrap();

    let versions = [
        object_version(&*store, 1).await,
        object_version(&*store, 2).await,
        object_version(&*store, 3).await,
        object_version(&*store, 4).await,
        object_version(&*store, 5).await,
    ];
    assert_eq!(versions, [2, 3, 4, 5, 2]);
}

/// A sparse vector with no terms is still a sparse field, so the batch takes
/// the sparse version and the gate that goes with it.
#[tokio::test]
async fn an_empty_sparse_vector_still_needs_the_sparse_level() {
    let ns = at_level_4(common::memory()).await;
    let err = ns
        .upsert(vec![sparse_doc("a", &[0.0, 0.0], &[])])
        .await
        .unwrap_err();
    too_old_for_sparse(err);
}

/// The batch size limit counts the sparse column, and a refused batch writes
/// nothing.
#[tokio::test]
async fn a_batch_made_too_big_by_its_sparse_column_is_refused_and_writes_nothing() {
    let store = common::memory();
    let ns = Namespace::create(store.clone(), "docs", schema(2, Metric::L2))
        .await
        .unwrap();
    let before = snapshot(&*store).await;
    let terms: Vec<(u32, f32)> = (0..itzma_core::document::MAX_SPARSE_TERMS as u32)
        .map(|t| (t, 1.0))
        .collect();
    // 65,535 pairs of 8 bytes is about 512 KiB a document, so 140 pass 64 MiB.
    let docs: Vec<Document> = (0..140)
        .map(|i| sparse_doc(&format!("d{i:03}"), &[0.0, 0.0], &terms))
        .collect();
    let err = ns.upsert(docs).await.unwrap_err();
    assert!(matches!(err, ItzmaError::BatchTooLarge { .. }), "{err:?}");
    assert_eq!(snapshot(&*store).await, before);
}
