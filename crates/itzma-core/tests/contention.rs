mod common;

use std::sync::Arc;

use common::faulty::{Fault, FaultyStore, PutKind};
use common::{doc, schema};
use futures::TryStreamExt;
use itzma_core::manifest::{self, Manifest};
use itzma_core::wal::wal_prefix;
use itzma_core::{ItzmaError, Metric, Namespace};
use object_store::memory::InMemory;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt};

fn faulty() -> (Arc<FaultyStore>, Arc<InMemory>) {
    let inner = Arc::new(InMemory::new());
    (FaultyStore::new(inner.clone()), inner)
}

#[tokio::test]
async fn append_gives_up_after_32_lost_races_and_writes_nothing() {
    let (store, inner) = faulty();
    let ns = Namespace::create(store.clone(), "docs", schema(1, Metric::L2))
        .await
        .unwrap();
    store.fail_puts("/wal/", PutKind::Create, Fault::AlreadyExists, None);

    let err = ns.upsert(vec![doc("a", &[1.0])]).await.unwrap_err();
    assert!(
        matches!(err, ItzmaError::WriteContention { attempts: 32 }),
        "unexpected error: {err:?}"
    );
    let listed: Vec<_> = inner
        .list(Some(&wal_prefix("docs")))
        .try_collect()
        .await
        .unwrap();
    assert!(
        listed.is_empty(),
        "a failed append must leave nothing behind"
    );

    // Once the store behaves, the same handle writes at seq 1.
    store.clear();
    assert_eq!(ns.upsert(vec![doc("a", &[1.0])]).await.unwrap(), 1);
}

#[tokio::test]
async fn manifest_update_gives_up_after_32_lost_rounds() {
    let (store, _inner) = faulty();
    manifest::create(&*store, "docs", &Manifest::new(schema(1, Metric::L2)))
        .await
        .unwrap();
    store.fail_puts(
        "manifest/current.json",
        PutKind::Update,
        Fault::Precondition,
        None,
    );

    let mut runs = 0;
    let err = manifest::update(&*store, "docs", |m| {
        runs += 1;
        m.wal_folded_through = 9;
    })
    .await
    .unwrap_err();
    assert!(
        matches!(err, ItzmaError::ManifestContention { attempts: 32 }),
        "unexpected error: {err:?}"
    );
    assert_eq!(runs, 32, "the closure runs once per round");

    store.clear();
    let (loaded, _) = manifest::load(&*store, "docs").await.unwrap();
    assert_eq!(loaded.wal_folded_through, 0);
}

#[tokio::test]
async fn faults_only_hit_the_puts_they_name() {
    let (store, _inner) = faulty();
    let ns = Namespace::create(store.clone(), "docs", schema(1, Metric::L2))
        .await
        .unwrap();

    // A rule for conditional updates does not touch WAL creates.
    store.fail_puts("/wal/", PutKind::Update, Fault::Precondition, None);
    assert_eq!(ns.upsert(vec![doc("a", &[1.0])]).await.unwrap(), 1);
    store.clear();

    // A one-shot generic failure is returned as a storage error, not retried,
    // and the next call works.
    store.fail_puts("/wal/", PutKind::Create, Fault::Generic, Some(1));
    let err = ns.upsert(vec![doc("b", &[2.0])]).await.unwrap_err();
    assert!(matches!(err, ItzmaError::Storage(_)), "unexpected: {err:?}");
    assert_eq!(ns.upsert(vec![doc("b", &[2.0])]).await.unwrap(), 2);
}

#[tokio::test]
async fn a_lost_ack_applies_the_put_and_still_errors() {
    let (store, inner) = faulty();
    let ns = Namespace::create(store.clone(), "docs", schema(1, Metric::L2))
        .await
        .unwrap();
    store.fail_puts("/wal/", PutKind::Create, Fault::LostAck, Some(1));

    let err = ns.upsert(vec![doc("a", &[1.0])]).await.unwrap_err();
    assert!(matches!(err, ItzmaError::Storage(_)), "unexpected: {err:?}");

    let wal: Vec<_> = inner
        .list(Some(&wal_prefix("docs")))
        .try_collect()
        .await
        .unwrap();
    assert_eq!(
        wal.len(),
        1,
        "the put landed although the caller saw an error"
    );
}

#[tokio::test]
async fn fail_nth_op_fails_exactly_that_operation_of_any_kind() {
    let (store, _inner) = faulty();
    let ns = Namespace::create(store.clone(), "docs", schema(1, Metric::L2))
        .await
        .unwrap();
    ns.upsert(vec![doc("a", &[1.0])]).await.unwrap();

    // One query is a manifest get, a listing and a WAL get at least.
    store.reset_ops();
    ns.query(&[0.0], 1).await.unwrap();
    let total = store.ops();
    assert!(total >= 3, "only {total} operations counted");

    // A fresh handle per run: a handle keeps the tail it folded, so
    // a second query on `ns` would not fetch the WAL object again.
    for n in 0..total {
        let cold = Namespace::open(store.clone(), "docs").await.unwrap();
        store.fail_nth_op(n, Fault::Generic);
        assert!(
            cold.query(&[0.0], 1).await.is_err(),
            "operation {n} was not failed"
        );
        store.clear();
    }
    ns.query(&[0.0], 1).await.unwrap();
}

#[tokio::test]
async fn a_planned_fault_on_a_delete_leaves_the_object() {
    let (store, inner) = faulty();
    let path = Path::from("a/b");
    inner.put(&path, "x".into()).await.unwrap();

    store.fail_nth_op(0, Fault::Generic);
    assert!(store.delete(&path).await.is_err());
    assert!(
        inner.get(&path).await.is_ok(),
        "a failed delete must not apply"
    );

    // The plan was spent, so the next delete goes through.
    store.delete(&path).await.unwrap();
    assert!(inner.get(&path).await.is_err());
}

#[tokio::test]
async fn a_lost_ack_on_a_delete_applies_it_and_still_errors() {
    let (store, inner) = faulty();
    let path = Path::from("a/b");
    inner.put(&path, "x".into()).await.unwrap();

    store.fail_nth_op(0, Fault::LostAck);
    assert!(store.delete(&path).await.is_err());
    assert!(
        inner.get(&path).await.is_err(),
        "the delete must have applied"
    );
}

#[tokio::test]
async fn a_hook_runs_once_before_the_first_read_of_its_key() {
    let (store, inner) = faulty();
    let path = Path::from("a/b");
    inner.put(&path, "x".into()).await.unwrap();

    // The hook deletes the object, so the read it precedes finds nothing.
    let hook_store = inner.clone();
    let hook_path = path.clone();
    store.before_get("a/b", async move {
        hook_store.delete(&hook_path).await.unwrap();
    });
    let other = Path::from("c/d");
    inner.put(&other, "y".into()).await.unwrap();
    store.get(&other).await.unwrap();
    assert!(
        inner.get(&path).await.is_ok(),
        "a read of another key must not fire it"
    );

    assert!(store.get(&path).await.is_err());

    // Spent: putting the object back and reading again runs no hook.
    inner.put(&path, "x".into()).await.unwrap();
    store.get(&path).await.unwrap();
}
