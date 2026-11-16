//! A handle that opted in to a staleness bound with
//! `Namespace::with_staleness`.

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::counting::CountingStore;
use common::{backends, cas_backends, doc, schema};
use itzma_core::{CompactionOptions, GcOptions, Metric, Namespace, WriteKey};
use object_store::ObjectStore;

const DIM: usize = 2;
const HOUR: Duration = Duration::from_secs(3600);

async fn create(store: Arc<dyn ObjectStore>) -> Namespace {
    Namespace::create(store, "docs", schema(DIM, Metric::L2))
        .await
        .unwrap()
}

async fn open(store: Arc<dyn ObjectStore>, bound: Duration) -> Namespace {
    Namespace::open(store, "docs")
        .await
        .unwrap()
        .with_staleness(bound)
}

/// Every id a query finds, sorted.
async fn found(ns: &Namespace) -> Vec<String> {
    let mut ids: Vec<String> = ns
        .query(&[0.0; DIM], 1000)
        .await
        .unwrap()
        .into_iter()
        .map(|hit| hit.id)
        .collect();
    ids.sort();
    ids
}

#[tokio::test]
async fn a_young_view_sends_no_request() {
    let counting = Arc::new(CountingStore::default());
    let store: Arc<dyn ObjectStore> = counting.clone();
    create(store.clone())
        .await
        .upsert(vec![doc("a", &[1.0, 0.0])])
        .await
        .unwrap();
    let ns = open(store, HOUR).await;
    assert_eq!(found(&ns).await, ["a"]);
    counting.reset();
    assert_eq!(found(&ns).await, ["a"]);
    assert_eq!((counting.requests(), counting.lists()), (0, 0));
}

#[tokio::test]
async fn another_writers_write_shows_up_once_the_bound_passes() {
    // Real clock: a bound of one second, waited out once for every backend.
    let bound = Duration::from_secs(1);
    let all = backends();
    let mut readers = Vec::new();
    for b in &all {
        let writer = create(b.store.clone()).await;
        let reader = open(b.store.clone(), bound).await;
        assert!(found(&reader).await.is_empty());
        writer.upsert(vec![doc("w", &[0.0, 1.0])]).await.unwrap();
        assert!(
            found(&reader).await.is_empty(),
            "{}: visible inside the bound",
            b.name
        );
        readers.push(reader);
    }
    tokio::time::sleep(bound + Duration::from_secs(1)).await;
    for (b, reader) in all.iter().zip(&readers) {
        assert_eq!(found(reader).await, ["w"], "{}", b.name);
    }
}

#[tokio::test]
async fn own_writes_are_visible_at_once_in_every_form() {
    for b in backends() {
        eprintln!("backend: {}", b.name);
        create(b.store.clone()).await;
        let ns = open(b.store.clone(), HOUR).await;
        assert!(found(&ns).await.is_empty());
        ns.upsert(vec![doc("a", &[1.0, 0.0])]).await.unwrap();
        assert_eq!(found(&ns).await, ["a"]);
        ns.upsert_keyed(&WriteKey::new("k1"), vec![doc("b", &[0.0, 1.0])])
            .await
            .unwrap();
        assert_eq!(found(&ns).await, ["a", "b"]);
        ns.delete(vec!["a".into()]).await.unwrap();
        assert_eq!(found(&ns).await, ["b"]);
        ns.delete_keyed(&WriteKey::new("k2"), vec!["b".into()])
            .await
            .unwrap();
        assert!(found(&ns).await.is_empty());
    }
}

#[tokio::test]
async fn an_own_write_that_lost_a_seq_race_is_visible_at_once() {
    for b in backends() {
        eprintln!("backend: {}", b.name);
        let other = create(b.store.clone()).await;
        let ns = open(b.store.clone(), HOUR).await;
        assert!(found(&ns).await.is_empty());
        // The other handle takes seq 1, so this handle's hint collides and
        // its write lands at 2.
        other
            .upsert(vec![doc("theirs", &[0.0, 1.0])])
            .await
            .unwrap();
        let seq = ns.upsert(vec![doc("mine", &[1.0, 0.0])]).await.unwrap();
        assert_eq!(seq, 2);
        assert_eq!(found(&ns).await, ["mine", "theirs"]);
    }
}

#[tokio::test]
async fn a_merge_and_gc_under_a_stale_view_is_a_stale_read_that_recovers() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let writer = create(b.store.clone()).await;
        for i in 0..3 {
            writer
                .upsert(vec![doc(&format!("d{i}"), &[i as f32, 0.0])])
                .await
                .unwrap();
            writer.compact(&CompactionOptions::default()).await.unwrap();
        }
        let reader = open(b.store.clone(), HOUR).await;
        assert_eq!(found(&reader).await, ["d0", "d1", "d2"]);
        // Merge the segments the reader's cached manifest names, then
        // collect them at once.
        let tight = CompactionOptions {
            max_segments: 1,
            ..CompactionOptions::default()
        };
        writer.compact(&tight).await.unwrap();
        let report = writer
            .gc(&GcOptions {
                min_age: Duration::ZERO,
                ..GcOptions::default()
            })
            .await
            .unwrap();
        assert!(report.segments_deleted > 0, "{report:?}");
        assert_eq!(found(&reader).await, ["d0", "d1", "d2"]);
        // The reload republished the snapshot: no second miss, and still
        // right.
        assert_eq!(found(&reader).await, ["d0", "d1", "d2"]);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn many_queries_past_the_bound_share_one_refresh() {
    let counting = Arc::new(CountingStore::default());
    let store: Arc<dyn ObjectStore> = counting.clone();
    create(store.clone())
        .await
        .upsert(vec![doc("a", &[1.0, 0.0])])
        .await
        .unwrap();
    let bound = Duration::from_millis(200);
    let ns = Arc::new(open(store, bound).await);
    found(&ns).await;
    tokio::time::sleep(bound * 2).await;
    counting.reset();
    let tasks: Vec<_> = (0..16)
        .map(|_| {
            let ns = ns.clone();
            tokio::spawn(async move { found(&ns).await })
        })
        .collect();
    for task in tasks {
        assert_eq!(task.await.unwrap(), ["a"]);
    }
    let manifests = counting
        .paths()
        .iter()
        .filter(|p| p.contains("manifest"))
        .count();
    assert_eq!(manifests, 1);
    assert_eq!(counting.lists(), 1);
}

#[tokio::test]
async fn a_strict_handle_is_unchanged() {
    for b in backends() {
        eprintln!("backend: {}", b.name);
        let writer = create(b.store.clone()).await;
        let reader = Namespace::open(b.store.clone(), "docs").await.unwrap();
        assert!(found(&reader).await.is_empty());
        writer.upsert(vec![doc("w", &[0.0, 1.0])]).await.unwrap();
        assert_eq!(found(&reader).await, ["w"]);
    }
}

/// The handle's own write was folded and fenced (min_age 0) before its next
/// query: folding it over the cached manifest meets the fence, which is a
/// stale read that reloads, never an error.
#[tokio::test]
async fn an_own_write_fenced_under_a_young_view_reloads() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        create(b.store.clone()).await;
        let ns = open(b.store.clone(), HOUR).await;
        assert!(found(&ns).await.is_empty());
        ns.upsert(vec![doc("a", &[1.0, 0.0])]).await.unwrap();
        ns.compact(&CompactionOptions::default()).await.unwrap();
        let report = ns
            .gc(&GcOptions {
                min_age: Duration::ZERO,
                fence_wal: true,
                ..GcOptions::default()
            })
            .await
            .unwrap();
        assert!(report.wal_objects_fenced > 0, "{report:?}");
        assert_eq!(found(&ns).await, ["a"]);
    }
}

/// After a stale read under a bounded view recovers, the reload is the new
/// snapshot: the next query within the bound reads no manifest and lists
/// nothing.
#[tokio::test]
async fn the_snapshot_a_stale_read_reloads_is_kept() {
    let counting = Arc::new(CountingStore::default());
    let store: Arc<dyn ObjectStore> = counting.clone();
    let writer = create(store.clone()).await;
    for i in 0..3 {
        writer
            .upsert(vec![doc(&format!("d{i}"), &[i as f32, 0.0])])
            .await
            .unwrap();
        writer.compact(&CompactionOptions::default()).await.unwrap();
    }
    let reader = open(store, HOUR).await;
    assert_eq!(found(&reader).await, ["d0", "d1", "d2"]);
    let tight = CompactionOptions {
        max_segments: 1,
        ..CompactionOptions::default()
    };
    writer.compact(&tight).await.unwrap();
    writer
        .gc(&GcOptions {
            min_age: Duration::ZERO,
            ..GcOptions::default()
        })
        .await
        .unwrap();
    assert_eq!(found(&reader).await, ["d0", "d1", "d2"]);
    counting.reset();
    assert_eq!(found(&reader).await, ["d0", "d1", "d2"]);
    // Segments are read as always (no decoded cache here); the manifest and
    // the WAL listing are not.
    let manifests = counting
        .paths()
        .iter()
        .filter(|p| p.contains("manifest"))
        .count();
    assert_eq!(
        (counting.lists(), manifests),
        (0, 0),
        "the reloaded snapshot was not kept"
    );
}
