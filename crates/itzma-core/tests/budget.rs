//! The read budget: a query or a compaction run refuses to read
//! more than the handle's budget, says which object went over, and leaves the
//! store as it was.

mod common;

use std::sync::Arc;

use common::{backends, cas_backends, doc, schema, snapshot};
use itzma_core::wal::wal_path;
use itzma_core::{
    Compaction, CompactionOptions, FetchPolicy, ItzmaError, Metric, Namespace, segment,
};
use object_store::{ObjectStore, ObjectStoreExt};

const DIM: usize = 4;

// Every exact byte/request-count assertion in this file pins un-merged,
// un-aligned reads, so every handle this file builds opts into
// FetchPolicy::local() rather than the object_store() default.
async fn create(store: &Arc<dyn ObjectStore>) -> Namespace {
    Namespace::create(store.clone(), "docs", schema(DIM, Metric::L2))
        .await
        .unwrap()
        .with_fetch_policy(FetchPolicy::local())
}

/// A fresh handle with the given budget.
async fn with_budget(store: &Arc<dyn ObjectStore>, bytes: u64) -> Namespace {
    Namespace::open(store.clone(), "docs")
        .await
        .unwrap()
        .with_read_budget(bytes)
        .with_fetch_policy(FetchPolicy::local())
}

/// One batch of `n` documents, ids `<start>..<start + n>`. The vectors differ
/// from document to document, so the batch does not compress well.
async fn batch(ns: &Namespace, start: usize, n: usize) {
    let docs = (start..start + n)
        .map(|i| {
            let f = i as f32;
            doc(&format!("d{i:04}"), &[f, f * 0.5 + 1.0, 7.0 - f, f * f])
        })
        .collect();
    ns.upsert(docs).await.unwrap();
}

/// One batch of `n` documents with the same vector, which ZSTD squeezes.
async fn flat_batch(ns: &Namespace, start: usize, n: usize) {
    let docs = (start..start + n)
        .map(|i| doc(&format!("d{i:04}"), &[1.0, 1.0, 1.0, 1.0]))
        .collect();
    ns.upsert(docs).await.unwrap();
}

async fn stored_size(store: &dyn ObjectStore, path: &object_store::path::Path) -> u64 {
    store.head(path).await.unwrap().size
}

/// What a read charges for the WAL object at `seq`: the larger of its size
/// and the `body_len` in its header.
async fn wal_charge(store: &dyn ObjectStore, seq: u64) -> u64 {
    let path = wal_path("docs", seq);
    let bytes = store.get(&path).await.unwrap().bytes().await.unwrap();
    let body_len = u64::from(u32::from_le_bytes(bytes[9..13].try_into().unwrap()));
    body_len.max(bytes.len() as u64)
}

async fn segment_size(store: &dyn ObjectStore, lo: u64, hi: u64) -> u64 {
    let id = segment::segment_id(lo, hi);
    stored_size(store, &segment::segment_path("docs", &id)).await
}

/// Options whose segment cap fits a handle with a budget of `bytes`, as
/// `Namespace::compact` requires.
fn fits(bytes: u64) -> CompactionOptions {
    CompactionOptions {
        max_segment_bytes: bytes,
        ..CompactionOptions::default()
    }
}

fn over(err: ItzmaError) -> (String, u64, u64) {
    match err {
        ItzmaError::ReadBudgetExceeded { path, needed, max } => (path, needed, max),
        other => panic!("expected ReadBudgetExceeded, got {other:?}"),
    }
}

#[tokio::test]
async fn a_segment_over_budget_is_refused_and_the_store_is_left_alone() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = create(&b.store).await;
        batch(&ns, 0, 20).await;
        ns.compact(&CompactionOptions::default()).await.unwrap();
        let size = segment_size(&*b.store, 1, 1).await;
        let before = snapshot(&*b.store).await;

        let (path, needed, max) = over(
            with_budget(&b.store, size - 1)
                .await
                .query(&[0.0; DIM], 3)
                .await
                .unwrap_err(),
        );
        assert!(path.contains("segments/"), "{path}");
        assert_eq!((needed, max), (size, size - 1));

        let hits = with_budget(&b.store, size)
            .await
            .query(&[0.0; DIM], 3)
            .await
            .unwrap();
        assert_eq!(hits.len(), 3);
        assert_eq!(snapshot(&*b.store).await, before);
    }
}

/// The edge of the budget for a namespace whose whole tail is the batch that
/// `write` puts there: one byte less fails, exactly enough passes.
async fn assert_edge_of_a_single_batch<F, Fut>(write: F) -> (u64, u64)
where
    F: Fn(Namespace) -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    let mut sizes = (0, 0);
    for b in backends() {
        eprintln!("backend: {}", b.name);
        write(create(&b.store).await).await;
        let charge = wal_charge(&*b.store, 1).await;
        sizes = (charge, stored_size(&*b.store, &wal_path("docs", 1)).await);

        let (path, needed, _) = over(
            with_budget(&b.store, charge - 1)
                .await
                .query(&[1.0; DIM], 1)
                .await
                .unwrap_err(),
        );
        assert!(path.contains("wal/"), "{path}");
        assert_eq!(needed, charge);
        with_budget(&b.store, charge)
            .await
            .query(&[1.0; DIM], 1)
            .await
            .unwrap();
    }
    sizes
}

#[tokio::test]
async fn a_compressed_wal_object_is_charged_its_body_len() {
    let (charge, size) =
        assert_edge_of_a_single_batch(|ns| async move { flat_batch(&ns, 0, 200).await }).await;
    assert!(
        charge > size,
        "the flat batch should compress: {charge} vs {size}"
    );
}

#[tokio::test]
async fn a_raw_wal_object_is_charged_its_size_header_included() {
    let (charge, size) =
        assert_edge_of_a_single_batch(|ns| async move { batch(&ns, 1000, 1).await }).await;
    assert_eq!(charge, size);
}

#[tokio::test]
async fn the_budget_is_a_running_total_over_the_whole_tail() {
    for b in backends() {
        eprintln!("backend: {}", b.name);
        let ns = create(&b.store).await;
        let mut total = 0;
        for seq in 1..=5u64 {
            batch(&ns, seq as usize * 10, 5).await;
            total += wal_charge(&*b.store, seq).await;
        }
        let each = wal_charge(&*b.store, 1).await;
        assert!(each < total, "every batch fits alone");

        let (_, needed, _) = over(
            with_budget(&b.store, total - 1)
                .await
                .query(&[0.0; DIM], 1)
                .await
                .unwrap_err(),
        );
        assert_eq!(needed, total);
        let hits = with_budget(&b.store, total)
            .await
            .query(&[0.0; DIM], 100)
            .await
            .unwrap();
        assert_eq!(hits.len(), 25);
    }
}

#[tokio::test]
async fn segments_and_tail_share_one_budget() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = create(&b.store).await;
        batch(&ns, 0, 20).await;
        ns.compact(&CompactionOptions::default()).await.unwrap();
        batch(&ns, 100, 5).await;
        batch(&ns, 200, 5).await;
        let total = segment_size(&*b.store, 1, 1).await
            + wal_charge(&*b.store, 2).await
            + wal_charge(&*b.store, 3).await;

        over(
            with_budget(&b.store, total - 1)
                .await
                .query(&[0.0; DIM], 1)
                .await
                .unwrap_err(),
        );
        let hits = with_budget(&b.store, total)
            .await
            .query(&[0.0; DIM], 100)
            .await
            .unwrap();
        assert_eq!(hits.len(), 30);
    }
}

#[tokio::test]
async fn a_fold_over_budget_publishes_nothing_and_writes_nothing() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = create(&b.store).await;
        batch(&ns, 0, 10).await;
        batch(&ns, 10, 10).await;
        let total = wal_charge(&*b.store, 1).await + wal_charge(&*b.store, 2).await;
        let before = snapshot(&*b.store).await;

        let strict = with_budget(&b.store, total - 1).await;
        over(strict.compact(&fits(total - 1)).await.unwrap_err());
        assert_eq!(snapshot(&*b.store).await, before);

        let enough = with_budget(&b.store, total).await;
        let outcome = enough.compact(&fits(total)).await.unwrap();
        assert!(matches!(
            outcome,
            Compaction::Published {
                folded_through: 2,
                ..
            }
        ));
    }
}

/// The tail and each merge group are charged against a budget of their own
///, so a merge's edge is the bigger of the two, not their sum.
#[tokio::test]
async fn a_merge_over_budget_publishes_nothing_and_writes_nothing() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = create(&b.store).await;
        batch(&ns, 0, 10).await;
        ns.compact(&CompactionOptions::default()).await.unwrap();
        batch(&ns, 10, 10).await;
        let edge = segment_size(&*b.store, 1, 1)
            .await
            .max(wal_charge(&*b.store, 2).await);
        let before = snapshot(&*b.store).await;
        let one = |bytes| CompactionOptions {
            max_segments: 1,
            ..fits(bytes)
        };

        let strict = with_budget(&b.store, edge - 1).await;
        over(strict.compact(&one(edge - 1)).await.unwrap_err());
        assert_eq!(snapshot(&*b.store).await, before);

        let enough = with_budget(&b.store, edge).await;
        let outcome = enough.compact(&one(edge)).await.unwrap();
        assert!(matches!(
            outcome,
            Compaction::Published {
                merged_segments: 1,
                folded_through: 2,
                ..
            }
        ));
    }
}

#[tokio::test]
async fn the_budget_is_per_group_not_per_run() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = create(&b.store).await;
        let roomy = CompactionOptions {
            max_segments: 4,
            ..CompactionOptions::default()
        };
        for s in 0..3 {
            batch(&ns, s * 20, 20).await;
            ns.compact(&roomy).await.unwrap();
        }
        batch(&ns, 60, 5).await;
        let mut edge = wal_charge(&*b.store, 4).await;
        let mut sum = 0;
        for s in 1..=3 {
            let size = segment_size(&*b.store, s, s).await;
            edge = edge.max(size);
            sum += size;
        }
        assert!(sum > edge, "the run as a whole must not fit");

        let tight = with_budget(&b.store, edge).await;
        let one = CompactionOptions {
            max_segments: 1,
            ..fits(edge)
        };
        let outcome = tight.compact(&one).await.unwrap();
        assert!(matches!(
            outcome,
            Compaction::Published {
                merged_segments: 3,
                folded_through: 4,
                ..
            }
        ));
    }
}

#[tokio::test]
async fn nothing_to_read_costs_nothing() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        create(&b.store).await;
        let ns = with_budget(&b.store, 0).await;
        assert!(ns.query(&[0.0; DIM], 5).await.unwrap().is_empty());
        assert_eq!(ns.compact(&fits(0)).await.unwrap(), Compaction::Nothing);

        batch(&ns, 0, 1).await;
        over(ns.query(&[0.0; DIM], 5).await.unwrap_err());
    }
}

#[tokio::test]
async fn the_budget_belongs_to_the_handle() {
    let store = common::memory();
    let ns = create(&store).await;
    batch(&ns, 0, 3).await;

    let strict = with_budget(&store, 1).await;
    over(strict.query(&[0.0; DIM], 1).await.unwrap_err());

    let other = Namespace::open(store.clone(), "docs").await.unwrap();
    assert_eq!(other.query(&[0.0; DIM], 1).await.unwrap().len(), 1);
    assert_eq!(ns.query(&[0.0; DIM], 1).await.unwrap().len(), 1);
}

#[tokio::test]
async fn a_budgeted_handle_can_still_be_moved_into_a_task() {
    let store = common::memory();
    let ns = create(&store).await;
    batch(&ns, 0, 3).await;
    let ns = with_budget(&store, 1 << 20).await;
    let hits = tokio::spawn(async move { ns.query(&[0.0; DIM], 2).await })
        .await
        .unwrap()
        .unwrap();
    assert_eq!(hits.len(), 2);
}

/// a decoded-cache hit fetches nothing and is not charged, so a warm
/// query runs under a budget smaller than the structures it uses. A miss is
/// charged as it is without the cache.
#[tokio::test]
async fn a_decoded_cache_hit_is_not_charged_and_a_miss_is() {
    use itzma_core::{DecodedCache, DecodedCacheOptions, QueryOptions};

    let store = common::memory();
    let ns = create(&store).await;
    batch(&ns, 0, 50).await;
    ns.compact(&CompactionOptions::default()).await.unwrap();
    let cache = DecodedCache::new(DecodedCacheOptions::default());
    let exact = QueryOptions::default().exact(true);
    let probe = [0.5; DIM];

    let tiny = with_budget(&store, 1)
        .await
        .with_decoded_cache(cache.clone());
    let cold = tiny.query_with(&probe, 5, &exact).await;
    assert!(
        matches!(cold, Err(ItzmaError::ReadBudgetExceeded { .. })),
        "a miss is charged: {cold:?}"
    );

    let roomy = with_budget(&store, u64::MAX)
        .await
        .with_decoded_cache(cache);
    let expected = roomy.query_with(&probe, 5, &exact).await.unwrap();
    let warm = tiny.query_with(&probe, 5, &exact).await.unwrap();
    assert_eq!(warm, expected, "a hit is not charged");
}
