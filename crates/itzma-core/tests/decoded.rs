//! The decoded-structure cache: `Namespace::with_decoded_cache`.
//!
//! A warm query reads the manifest and nothing else it already parsed, answers
//! what a handle without the cache answers, and a handle without the cache is
//! unchanged.

mod common;

use std::sync::Arc;

use common::Lcg;
use common::counting::CountingStore;
use common::faulty::FaultyStore;
use futures::future::{BoxFuture, FutureExt};
use itzma_core::{
    AttrSchema, AttrType, CompactionOptions, DecodedCache, DecodedCacheOptions, Document,
    FilterOptions, Hit, IndexKind, IndexOptions, IndexSparseOptions, Metric, Namespace, Predicate,
    QueryOptions, Schema, SparseQueryOptions,
};
use object_store::ObjectStore;

const DIM: usize = 8;
const MANIFEST: &str = "docs/manifest/current.json";
const ROOMY: CompactionOptions = CompactionOptions {
    max_segments: 8,
    max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
};

fn typed() -> Schema {
    Schema::new(DIM, Metric::L2).with_attrs(AttrSchema::new().field("active", AttrType::Bool))
}

fn document(rng: &mut Lcg, i: usize) -> Document {
    Document {
        id: format!("d{i:05}"),
        vector: rng.vector(DIM),
        attrs: Some(serde_json::json!({ "active": i.is_multiple_of(3) })),
        sparse: Some(vec![((i % 50) as u32, 1.0), (100 + (i % 7) as u32, 0.5)]),
    }
}

/// Two indexed segments, the newer one overwriting and deleting some of the
/// older one's ids, so a query runs shadow checks (Bloom filter, id and
/// tombstone blocks) as well as the index.
///
/// Segments are written at format `level` (3), raised before the
/// first fold.
async fn built_at(store: Arc<dyn ObjectStore>, kind: IndexKind, level: u32) -> Namespace {
    let ns = Namespace::create(store, "docs", typed()).await.unwrap();
    ns.upgrade_segment_format(level).await.unwrap();
    let mut rng = Lcg(3);
    ns.upsert((0..2_000).map(|i| document(&mut rng, i)).collect())
        .await
        .unwrap();
    ns.compact(&ROOMY).await.unwrap();
    ns.upsert((0..600).map(|i| document(&mut rng, i * 3)).collect())
        .await
        .unwrap();
    ns.delete((0..200).map(|i| format!("d{:05}", i * 3 + 1)).collect())
        .await
        .unwrap();
    ns.compact(&ROOMY).await.unwrap();
    ns.index(&IndexOptions::default().min_rows(0).kind(kind))
        .await
        .unwrap();
    ns.index_sparse(&IndexSparseOptions::default())
        .await
        .unwrap();
    ns
}

/// A counting store and a namespace built on it, plus a handle on the same
/// store with a decoded cache.
async fn setup(kind: IndexKind) -> (Arc<CountingStore>, Namespace, Namespace) {
    setup_at(kind, 2).await
}

/// [`setup`] with segments at format `level`.
async fn setup_at(kind: IndexKind, level: u32) -> (Arc<CountingStore>, Namespace, Namespace) {
    let counting = Arc::new(CountingStore::default());
    let store: Arc<dyn ObjectStore> = counting.clone();
    let plain = built_at(store.clone(), kind, level).await;
    let cached = Namespace::open(store, "docs")
        .await
        .unwrap()
        .with_decoded_cache(DecodedCache::new(DecodedCacheOptions::default()));
    (counting, plain, cached)
}

fn query() -> Vec<f32> {
    Lcg(7).vector(DIM)
}

/// Run `call` cold then warm on the cached handle, and once on the plain one.
/// Returns the warm run's GET paths, and checks the three answers agree.
async fn warm_paths<T, F>(
    counting: &CountingStore,
    plain: &Namespace,
    cached: &Namespace,
    call: F,
) -> Vec<String>
where
    T: PartialEq + std::fmt::Debug,
    F: for<'a> Fn(&'a Namespace) -> BoxFuture<'a, T>,
{
    let expected = call(plain).await;
    let cold = call(cached).await;
    counting.reset();
    let warm = call(cached).await;
    let paths = counting.paths();
    assert_eq!(cold, expected, "a cold cached answer differs");
    assert_eq!(warm, expected, "a warm cached answer differs");
    paths
}

/// What a warm indexed query still reads: the manifest, then only the exact
/// vectors of its shortlist for the rerank, which the cache does not keep (each
/// query wants other rows). No index object, directory, filter or block.
fn assert_only_rerank_vectors(paths: &[String]) {
    assert_eq!(paths[0], MANIFEST);
    assert!(
        paths[1..].iter().all(|p| p.ends_with("/data.bin")),
        "a warm query read an index or a sidecar: {paths:?}"
    );
}

async fn dense(ns: &Namespace) -> Vec<Hit> {
    ns.query(&query(), 10).await.unwrap()
}

#[tokio::test]
async fn a_warm_indexed_query_reads_only_the_manifest_and_rerank_vectors() {
    for kind in [IndexKind::Sq8, IndexKind::Hnsw] {
        let (counting, plain, cached) = setup(kind).await;
        let paths = warm_paths(&counting, &plain, &cached, |ns| dense(ns).boxed()).await;
        assert_only_rerank_vectors(&paths);
    }
}

#[tokio::test]
async fn a_warm_filtered_indexed_query_reads_only_the_manifest_and_rerank_vectors() {
    let (counting, plain, cached) = setup(IndexKind::Sq8).await;
    let options = QueryOptions::default().filter(Some(Predicate::is_true("active")));
    let paths = warm_paths(&counting, &plain, &cached, |ns| {
        let options = options.clone();
        async move { ns.query_with(&query(), 10, &options).await.unwrap() }.boxed()
    })
    .await;
    assert_only_rerank_vectors(&paths);
}

/// Version 3 attribute blocks are `AttrRows` in the cache, held as
/// columns: warm, a filtered query and a filter read nothing more.
#[tokio::test]
async fn a_warm_filtered_query_and_filter_over_version_3_read_nothing_more() {
    let (counting, plain, cached) = setup_at(IndexKind::Sq8, 3).await;
    let versions = common::live_segment_versions(&*counting).await;
    assert!(versions.iter().all(|v| *v == 3), "{versions:?}");
    let options = QueryOptions::default().filter(Some(Predicate::is_true("active")));
    let paths = warm_paths(&counting, &plain, &cached, |ns| {
        let options = options.clone();
        async move { ns.query_with(&query(), 10, &options).await.unwrap() }.boxed()
    })
    .await;
    assert_only_rerank_vectors(&paths);
    let paths = warm_paths(&counting, &plain, &cached, |ns| {
        async move {
            ns.filter(&Predicate::is_true("active"), &FilterOptions::default())
                .await
                .unwrap()
        }
        .boxed()
    })
    .await;
    assert_eq!(paths, [MANIFEST]);
    let exact = QueryOptions::default().exact(true);
    let paths = warm_paths(&counting, &plain, &cached, |ns| {
        let exact = exact.clone();
        async move { ns.query_with(&query(), 10, &exact).await.unwrap() }.boxed()
    })
    .await;
    assert_eq!(paths, [MANIFEST]);
}

#[tokio::test]
async fn a_warm_flat_query_reads_only_the_manifest() {
    let (counting, plain, cached) = setup(IndexKind::Sq8).await;
    let exact = QueryOptions::default().exact(true);
    let paths = warm_paths(&counting, &plain, &cached, |ns| {
        let exact = exact.clone();
        async move { ns.query_with(&query(), 10, &exact).await.unwrap() }.boxed()
    })
    .await;
    assert_eq!(paths, [MANIFEST]);
}

#[tokio::test]
async fn a_warm_filter_reads_only_the_manifest() {
    let (counting, plain, cached) = setup(IndexKind::Sq8).await;
    let paths = warm_paths(&counting, &plain, &cached, |ns| {
        async move {
            ns.filter(&Predicate::is_true("active"), &FilterOptions::default())
                .await
                .unwrap()
        }
        .boxed()
    })
    .await;
    assert_eq!(paths, [MANIFEST]);
}

/// The term directory is cached; the postings of the query's own terms are
/// not, so a warm sparse query reads those
/// and nothing of the segments.
#[tokio::test]
async fn a_warm_sparse_query_reads_only_the_manifest_and_postings() {
    let (counting, plain, cached) = setup(IndexKind::Sq8).await;
    let terms = vec![(3u32, 1.0f32), (101, 2.0)];
    let paths = warm_paths(&counting, &plain, &cached, |ns| {
        let terms = terms.clone();
        async move {
            ns.query_sparse(&terms, 10, &SparseQueryOptions::default())
                .await
                .unwrap()
        }
        .boxed()
    })
    .await;
    assert_eq!(paths[0], MANIFEST);
    assert!(
        paths[1..].iter().all(|p| p.contains("/sparse-idx-")),
        "only postings after the manifest: {paths:?}"
    );
}

#[tokio::test]
async fn a_handle_without_the_cache_reads_the_same_on_every_query() {
    let (counting, plain, _) = setup(IndexKind::Sq8).await;
    dense(&plain).await;
    counting.reset();
    dense(&plain).await;
    // Sorted: segments and indexes are fetched concurrently.
    let mut first = counting.paths();
    first.sort();
    counting.reset();
    dense(&plain).await;
    let mut second = counting.paths();
    second.sort();
    assert_eq!(second, first);
    assert!(first.len() > 1);
}

/// Sixteen queries on a cold cache at once read each structure once: the GETs
/// of one cold query, plus what each other query reads even when warm (its
/// manifest and its rerank vectors).
#[tokio::test]
async fn concurrent_cold_queries_share_one_load_per_structure() {
    let (counting, _, _) = setup(IndexKind::Sq8).await;
    // Every store call yields first, so the queries really interleave
    // (`InMemory` alone would let each one finish before the next starts).
    let yielding = FaultyStore::new(counting.clone());
    yielding.yield_each_op(true);
    let store: Arc<dyn ObjectStore> = yielding;
    let open_cached = || async {
        Namespace::open(store.clone(), "docs")
            .await
            .unwrap()
            .with_decoded_cache(DecodedCache::new(DecodedCacheOptions::default()))
    };

    let one = open_cached().await;
    counting.reset();
    dense(&one).await;
    let single = counting.requests();
    counting.reset();
    dense(&one).await;
    let warm = counting.requests();

    let shared = open_cached().await;
    counting.reset();
    let answers = futures::future::join_all((0..16).map(|_| dense(&shared))).await;
    assert!(answers.windows(2).all(|w| w[0] == w[1]));
    assert_eq!(
        counting.requests(),
        single + 15 * warm,
        "{:?}",
        counting.paths()
    );
}

/// An unindexed namespace's warm flat query reads nothing but the manifest:
/// its segments are views kept by the decoded cache.
#[tokio::test]
async fn a_warm_flat_query_on_an_unindexed_namespace_reads_only_the_manifest() {
    let counting = Arc::new(CountingStore::default());
    let store: Arc<dyn ObjectStore> = counting.clone();
    let ns = Namespace::create(store, "docs", typed())
        .await
        .unwrap()
        .with_decoded_cache(DecodedCache::new(DecodedCacheOptions::default()));
    let mut rng = Lcg(5);
    ns.upsert((0..500).map(|i| document(&mut rng, i)).collect())
        .await
        .unwrap();
    ns.compact(&ROOMY).await.unwrap();
    ns.upsert((0..100).map(|i| document(&mut rng, i * 2)).collect())
        .await
        .unwrap();
    ns.compact(&ROOMY).await.unwrap();
    let cold = ns.query(&query(), 10).await.unwrap();
    counting.reset();
    let warm = ns.query(&query(), 10).await.unwrap();
    assert_eq!(warm, cold);
    assert_eq!(counting.paths(), [MANIFEST]);
}
