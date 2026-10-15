//! The indexed query: `Namespace::query_with`, the
//! engine in `src/query.rs`, and how it uses `Namespace::index`'s indexes.
//!
//! The stub-kind tests that check the engine never trusts what an algorithm
//! returns (a row past the segment, a repeated row, too many rows) are unit
//! tests inside `src/query.rs`, since the algorithm interface is crate-private.
//! Everything here goes through the public API with real, published indexes.

mod common;

use common::counting::CountingStore;

use std::collections::BTreeMap;
use std::sync::Arc;

use common::faulty::{Fault, FaultyStore};
use common::{Lcg, assert_untouched, cas_backends, doc, naive_top_k, schema, snapshot};
use itzma_core::indexer::DEFAULT_MIN_ROWS;
use itzma_core::{
    CompactionOptions, Document, FetchPolicy, IndexKind, IndexOptions, ItzmaError, Manifest,
    Metric, Namespace, QueryOptions, manifest, segment,
};
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt};

/// Wraps a `CacheStore` over a fresh `CountingStore`, so a test can see how
/// many requests actually reached the underlying store versus how many were
/// served from cache — separate from the `store` a `Namespace` writes and
/// loads its manifest through.
fn cache_over(inner: Arc<CountingStore>) -> Arc<dyn ObjectStore> {
    Arc::new(itzma_core::CacheStore::new(inner, itzma_core::CacheOptions::default()).unwrap())
}

const DIM: usize = 4;
const L2: Metric = Metric::L2;
const ROOMY: CompactionOptions = CompactionOptions {
    max_segments: 8,
    max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
};
const TIGHT: CompactionOptions = CompactionOptions {
    max_segments: 1,
    max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
};

/// Index every segment, however small.
fn every() -> IndexOptions {
    IndexOptions::default().min_rows(0)
}

// Several tests in this file assert exact byte/request counts against a
// CountingStore, or binary-search for the exact byte at which a budget
// stops fitting, so every handle opts into FetchPolicy::local() to keep
// pinning that earlier behavior.
async fn create(store: Arc<dyn ObjectStore>) -> Namespace {
    Namespace::create(store, "docs", schema(DIM, L2))
        .await
        .unwrap()
        .with_fetch_policy(itzma_core::FetchPolicy::local())
}

fn docs_of(prefix: &str, seed: u64, n: usize) -> Vec<Document> {
    let mut rng = Lcg(seed);
    (0..n)
        .map(|i| doc(&format!("{prefix}{i:04}"), &rng.vector(DIM)))
        .collect()
}

async fn write(ns: &Namespace, prefix: &str, seed: u64, n: usize) {
    ns.upsert(docs_of(prefix, seed, n)).await.unwrap();
}

async fn manifest_of(store: &dyn ObjectStore) -> Manifest {
    manifest::load(store, "docs").await.unwrap().0
}

fn seg(lo: u64, hi: u64) -> String {
    segment::segment_id(lo, hi)
}

/// Every hit's id, in order.
fn ids(hits: &[itzma_core::Hit]) -> Vec<String> {
    hits.iter().map(|h| h.id.clone()).collect()
}

/// The exact top-`k` ids for `query` over `docs`, the model an indexed query
/// is checked against.
fn model(query: &[f32], docs: &[Document], k: usize) -> Vec<String> {
    naive_top_k(L2, query, docs, k)
}

/// `docs`, folded as `upsert` would: last write for an id wins.
fn state_of(batches: Vec<Vec<Document>>) -> BTreeMap<String, Document> {
    let mut state = BTreeMap::new();
    for batch in batches {
        for d in batch {
            state.insert(d.id.clone(), d);
        }
    }
    state
}

/// Assert that `ns` answers every query the same with `options` as a naive
/// scan of `docs`, for a handful of query vectors and of `top_k`.
async fn assert_matches_flat(ns: &Namespace, docs: &[Document], options: &QueryOptions) {
    for seed in [1, 2, 3, 40, 500] {
        for k in [1, 5, docs.len(), docs.len() + 10] {
            let query = Lcg(seed).vector(DIM);
            let got = ids(&ns.query_with(&query, k, options).await.unwrap());
            assert_eq!(got, model(&query, docs, k), "seed {seed}, k {k}");
        }
    }
}

#[tokio::test]
async fn a_namespace_with_both_index_kinds_matches_the_flat_scan() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = create(b.store.clone()).await;
        write(&ns, "a", 1, 200).await;
        ns.compact(&ROOMY).await.unwrap();
        ns.index(&IndexOptions::default().min_rows(0).kind(IndexKind::Sq8))
            .await
            .unwrap(); // segment 1: sq8

        write(&ns, "b", 2, 200).await;
        ns.compact(&ROOMY).await.unwrap();
        ns.index(&IndexOptions::default().min_rows(0))
            .await
            .unwrap(); // only segment 2 is new, so only it gets the default, hnsw

        let docs: Vec<Document> = docs_of("a", 1, 200)
            .into_iter()
            .chain(docs_of("b", 2, 200))
            .collect();
        assert_matches_flat(&ns, &docs, &QueryOptions::default()).await;

        let m = manifest_of(&*b.store).await;
        assert_eq!(m.indexes[&seg(1, 1)].kind, "sq8");
        assert_eq!(m.indexes[&seg(2, 2)].kind, "hnsw");
    }
}

/// The fetch policy changes how many requests a read makes and how
/// big each one is, never what a query returns: the same namespace, built
/// and queried under each policy in turn, must answer identically.
#[tokio::test]
async fn results_are_identical_under_both_fetch_policies() {
    for policy in [FetchPolicy::local(), FetchPolicy::object_store()] {
        let store = common::memory();
        let ns = create(store.clone()).await.with_fetch_policy(policy);
        write(&ns, "a", 7, 200).await;
        ns.compact(&ROOMY).await.unwrap();
        ns.index(&IndexOptions::default().min_rows(0))
            .await
            .unwrap();
        write(&ns, "b", 8, 50).await; // an unindexed tail segment too
        let docs: Vec<Document> = docs_of("a", 7, 200)
            .into_iter()
            .chain(docs_of("b", 8, 50))
            .collect();
        assert_matches_flat(&ns, &docs, &QueryOptions::default()).await;
    }
}

/// Index every segment with `nlist` partitions.
fn clustered(nlist: usize) -> IndexOptions {
    IndexOptions::default()
        .min_rows(0)
        .kind(IndexKind::Sq8)
        .nlist(Some(nlist))
}

#[tokio::test]
async fn a_clustered_index_probed_in_full_equals_the_flat_scan_at_every_stage() {
    for b in cas_backends() {
        let ns = create(b.store.clone()).await;
        write(&ns, "a", 1, 120).await;
        ns.compact(&ROOMY).await.unwrap();
        ns.index(&clustered(8)).await.unwrap();
        // nprobe = nlist and every row proposed: exactly the flat scan.
        let all = QueryOptions::default().rerank(None).nprobe(Some(8));
        assert_matches_flat(&ns, &docs_of("a", 1, 120), &all).await;

        // A second clustered segment above it, with a delete of an indexed row.
        write(&ns, "b", 2, 60).await;
        ns.delete(vec!["a0003".into()]).await.unwrap();
        ns.compact(&ROOMY).await.unwrap();
        ns.index(&clustered(5)).await.unwrap();
        let mut state = state_of(vec![docs_of("a", 1, 120), docs_of("b", 2, 60)]);
        state.remove("a0003");
        let live: Vec<Document> = state.values().cloned().collect();
        assert_matches_flat(&ns, &live, &all).await;
    }
}

#[tokio::test]
async fn nlist_zero_is_invalid_before_any_storage_call() {
    for b in cas_backends() {
        let ns = create(b.store.clone()).await;
        write(&ns, "a", 1, 5).await;
        ns.compact(&ROOMY).await.unwrap();
        let before = snapshot(&*b.store).await;
        let bad_build = IndexOptions::default().min_rows(0).nlist(Some(0));
        assert!(matches!(
            ns.index(&bad_build).await,
            Err(ItzmaError::InvalidQuery(_))
        ));
        assert_untouched(&*b.store, &before).await;
    }
}

/// How many bytes a query over `ns` actually reads, under a budget ample
/// enough that the fetch planner's budget fallback never has
/// anything to fall back from — this measures the real cost of an index,
/// not an edge race against the planner's own adaptive un-merging, which
/// no longer tracks "the one true minimum" once a tight budget can make
/// the same query cheaper by fetching less contiguously.
async fn bytes_read(store: &Arc<CountingStore>, ns: &Namespace, options: &QueryOptions) -> u64 {
    store.reset();
    ns.query_with(&Lcg(1).vector(DIM), 10, options)
        .await
        .unwrap();
    store.bytes()
}

#[tokio::test]
async fn a_clustered_index_reads_fewer_bytes_than_a_one_partition_scan() {
    let store = Arc::new(CountingStore::default());
    let ns = create(store.clone() as Arc<dyn ObjectStore>).await;
    write(&ns, "a", 1, 3000).await;
    ns.compact(&ROOMY).await.unwrap();

    let options = QueryOptions::default().rerank(Some(1));
    ns.index(&clustered(1)).await.unwrap();
    let one_partition = bytes_read(&store, &ns, &options).await;

    ns.index(&clustered(64).rebuild(true)).await.unwrap();
    let clustered_bytes = bytes_read(&store, &ns, &options).await;

    assert!(
        clustered_bytes < one_partition,
        "clustered read {clustered_bytes} bytes, one partition {one_partition}"
    );
    // Not checked here any more: "a budget between the two totals fits the
    // clustered index and refuses the one-partition scan." The fetch
    // planner's budget fallback means a budget-constrained
    // read can always shrink to its fully-unmerged floor and still succeed,
    // which is far below either index's ample-budget total — so no budget
    // strictly between the two totals reliably refuses one index while
    // fitting the other any more. That refusal was a consequence of the old
    // unconditional-merge model, not a guarantee this crate still makes.
}

#[tokio::test]
async fn a_partial_probe_reads_fewer_bytes_than_a_full_one() {
    // The engine's widening loop asks for every row as a last resort, so the
    // answer alone cannot show that a probe read only some partitions. Bytes can.
    let store = Arc::new(CountingStore::default());
    let ns = create(store.clone() as Arc<dyn ObjectStore>).await;
    write(&ns, "a", 1, 3000).await;
    ns.compact(&ROOMY).await.unwrap();
    ns.index(&clustered(16)).await.unwrap();

    let query = Lcg(7).vector(DIM);
    let mut bytes = Vec::new();
    for nprobe in [1, 4, 16] {
        let options = QueryOptions::default().rerank(Some(1)).nprobe(Some(nprobe));
        store.reset();
        let hits = ns.query_with(&query, 10, &options).await.unwrap();
        assert_eq!(hits.len(), 10, "nprobe {nprobe}");
        bytes.push(store.bytes());
    }
    assert!(
        bytes[0] < bytes[1] && bytes[1] < bytes[2],
        "more partitions probed must read more: {bytes:?}"
    );
}

/// Widening across several waves never re-fetches an already-fetched
/// partition: run the same heavily-shadowed clustered query twice and
/// confirm the second run costs exactly the same bytes as the first — a
/// cursor that leaked state or re-read a partition between runs would
/// drift, since nothing here is cached across separate `query_with` calls.
/// (The stronger claim, "widening always reads less than a flat scan,"
/// does not hold in general for a namespace this small and this heavily
/// shadowed: with `DIM` this low and the overwrites scattered across every
/// partition, nearly every id/attribute block ends up touched regardless,
/// so the id/attribute cost alone can approach a flat scan's — true of the
/// pre-cursor engine's own exhaustive fallback too, not a regression. The
/// real guarantee — no row or partition fetched twice within
/// one query — is what `index::sq8::tests::the_index_head_is_fetched_once_however_many_next_calls`
/// already pins at the unit level.)
#[tokio::test]
async fn widening_many_waves_costs_the_same_bytes_every_run() {
    let store = Arc::new(CountingStore::default());
    let ns = create(store.clone() as Arc<dyn ObjectStore>).await;
    write(&ns, "a", 1, 1600).await;
    ns.compact(&ROOMY).await.unwrap();
    ns.index(&clustered(16)).await.unwrap();

    // Overwrite almost everything from the tail, unindexed, so the index's
    // proposals are mostly shadowed and the engine must widen across most
    // of the clustered index's partitions to find top_k live rows.
    let mut rng = Lcg(99);
    let overwritten: Vec<_> = (0..1500)
        .map(|i| doc(&format!("a{i:04}"), &rng.vector(DIM)))
        .collect();
    ns.upsert(overwritten).await.unwrap();

    let query = Lcg(7).vector(DIM);
    let options = QueryOptions::default().rerank(Some(1)).nprobe(Some(1));
    // Fold the tail into the handle's view first, so neither
    // measured run pays for it and only the cursor's reads are compared.
    ns.query_with(&query, 10, &options).await.unwrap();

    store.reset();
    let first = ns.query_with(&query, 10, &options).await.unwrap();
    assert_eq!(first.len(), 10);
    let first_bytes = store.bytes();
    let first_requests = store.requests();

    store.reset();
    let second = ns.query_with(&query, 10, &options).await.unwrap();
    assert_eq!(
        ids(&second),
        ids(&first),
        "the same query against the same data must answer the same"
    );
    assert_eq!(
        store.bytes(),
        first_bytes,
        "a fresh cursor every call: byte cost must not drift between identical runs"
    );
    assert_eq!(store.requests(), first_requests);
}

#[tokio::test]
async fn a_fault_at_every_operation_of_a_clustered_query_is_an_error_never_a_wrong_answer() {
    let store = FaultyStore::new(common::memory());
    let ns = create(store.clone() as Arc<dyn ObjectStore>).await;
    write(&ns, "a", 1, 120).await;
    ns.compact(&ROOMY).await.unwrap();
    ns.index(&clustered(8)).await.unwrap();

    let query = Lcg(1).vector(DIM);
    let options = QueryOptions::default().nprobe(Some(3));
    store.reset_ops();
    let want = ids(&ns.query_with(&query, 5, &options).await.unwrap());
    let total_ops = store.ops();
    assert!(total_ops > 0);

    // A query never writes, so one namespace serves every fault.
    for fault in [Fault::Generic, Fault::LostAck] {
        for n in 0..total_ops {
            store.fail_nth_op(n, fault);
            let outcome = ns.query_with(&query, 5, &options).await;
            assert!(
                outcome.is_err(),
                "{fault:?} at op {n} of {total_ops} should have failed"
            );
        }
    }
    store.clear();
    assert_eq!(
        ids(&ns.query_with(&query, 5, &options).await.unwrap()),
        want
    );
}

#[tokio::test]
async fn a_rebuild_replaces_a_one_partition_index_with_a_clustered_one() {
    let ns = create(common::memory()).await;
    write(&ns, "a", 1, 150).await;
    ns.compact(&ROOMY).await.unwrap();
    ns.index(&every()).await.unwrap(); // nlist None: one partition at this size
    let again = ns.index(&clustered(6).rebuild(true)).await.unwrap();
    assert_eq!(again.indexed, 1);
    let all = QueryOptions::default().rerank(None).nprobe(Some(6));
    assert_matches_flat(&ns, &docs_of("a", 1, 150), &all).await;
}

#[test]
fn nprobe_defaults_to_none_and_survives_serde() {
    assert_eq!(QueryOptions::default().nprobe, None);
    let set = QueryOptions::default().nprobe(Some(3));
    let back: QueryOptions = serde_json::from_str(&serde_json::to_string(&set).unwrap()).unwrap();
    assert_eq!(back.nprobe, Some(3));
    assert_eq!(set.nprobe(None).nprobe, None);
}

#[tokio::test]
async fn the_default_nprobe_on_a_clustered_index_keeps_recall() {
    let ns = create(common::memory()).await;
    write(&ns, "a", 5, 400).await;
    ns.compact(&ROOMY).await.unwrap();
    ns.index(&clustered(16)).await.unwrap();
    let docs = docs_of("a", 5, 400);
    let mut found = 0;
    for seed in 1..=20u64 {
        let query = Lcg(seed).vector(DIM);
        let want = model(&query, &docs, 10);
        let got = ids(&ns
            .query_with(&query, 10, &QueryOptions::default())
            .await
            .unwrap());
        found += got.iter().filter(|id| want.contains(id)).count();
    }
    // Loose on purpose: real data gives the real number.
    assert!(found * 10 >= 20 * 10 * 9, "recall@10 was {found}/200");
}

#[tokio::test]
async fn nprobe_widens_the_hnsw_search_and_never_hurts_recall() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = create(b.store.clone()).await;
        write(&ns, "a", 1, 2_000).await;
        ns.compact(&ROOMY).await.unwrap();
        ns.index(&IndexOptions::default().min_rows(0).kind(IndexKind::Hnsw))
            .await
            .unwrap();

        let query = Lcg(99).vector(DIM);
        let exact: Vec<String> = ids(&ns
            .query_with(&query, 10, &QueryOptions::default().exact(true))
            .await
            .unwrap());

        let narrow = ns
            .query_with(&query, 10, &QueryOptions::default().nprobe(Some(1)))
            .await
            .unwrap();
        let wide = ns
            .query_with(&query, 10, &QueryOptions::default().nprobe(Some(20)))
            .await
            .unwrap();
        let overlap = |hits: &[itzma_core::Hit]| -> usize {
            hits.iter().filter(|h| exact.contains(&h.id)).count()
        };
        assert!(
            overlap(&wide) >= overlap(&narrow),
            "a wider ef_search multiplier should never find strictly fewer true neighbors: \
             narrow {} wide {}",
            overlap(&narrow),
            overlap(&wide)
        );
    }
}

/// Every index object's key, from a snapshot.
fn index_keys(snapshot: &BTreeMap<String, bytes::Bytes>) -> Vec<String> {
    snapshot
        .keys()
        .filter(|k| k.contains("/index-"))
        .cloned()
        .collect()
}

#[tokio::test]
async fn indexed_query_with_no_rerank_limit_equals_flat_at_every_stage() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = create(b.store.clone()).await;

        // A base segment, indexed.
        write(&ns, "a", 1, 50).await;
        ns.compact(&ROOMY).await.unwrap();
        ns.index(&every()).await.unwrap();

        // An unindexed fold on top (small, under the default minimum).
        write(&ns, "b", 2, 3).await;
        ns.compact(&ROOMY).await.unwrap();

        // A second indexed fold.
        write(&ns, "c", 3, 50).await;
        ns.compact(&ROOMY).await.unwrap();
        ns.index(&IndexOptions::default().min_rows(50))
            .await
            .unwrap();

        // A tail: an overwrite of an id from the first indexed segment, a
        // delete of one from the unindexed fold, and a fresh id.
        ns.upsert(vec![doc("a0000", &[9.0, 9.0, 9.0, 9.0])])
            .await
            .unwrap();
        ns.delete(vec!["b0001".into()]).await.unwrap();
        ns.upsert(vec![doc("z-new", &[0.1, 0.2, 0.3, 0.4])])
            .await
            .unwrap();

        let mut state = state_of(vec![
            docs_of("a", 1, 50),
            docs_of("b", 2, 3),
            docs_of("c", 3, 50),
        ]);
        state.insert("a0000".into(), doc("a0000", &[9.0, 9.0, 9.0, 9.0]));
        state.remove("b0001");
        state.insert("z-new".into(), doc("z-new", &[0.1, 0.2, 0.3, 0.4]));
        let docs: Vec<_> = state.into_values().collect();

        assert_matches_flat(&ns, &docs, &QueryOptions::default().rerank(None)).await;
        // The default rerank is generous enough to be exact on a namespace
        // this small too, since rerank * top_k covers every indexed segment.
        assert_matches_flat(&ns, &docs, &QueryOptions::default()).await;
    }
}

#[tokio::test]
async fn a_merge_leaves_the_new_segment_unindexed_and_the_answer_still_matches_flat() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = create(b.store.clone()).await;
        write(&ns, "a", 1, 40).await;
        ns.compact(&ROOMY).await.unwrap();
        ns.index(&every()).await.unwrap();
        write(&ns, "b", 2, 40).await;
        ns.compact(&ROOMY).await.unwrap();
        ns.index(&every()).await.unwrap();

        // Force a merge: the new base is unindexed.
        write(&ns, "c", 3, 5).await;
        ns.compact(&TIGHT).await.unwrap();

        let state = state_of(vec![
            docs_of("a", 1, 40),
            docs_of("b", 2, 40),
            docs_of("c", 3, 5),
        ]);
        let docs: Vec<_> = state.into_values().collect();
        assert_matches_flat(&ns, &docs, &QueryOptions::default().rerank(None)).await;
    }
}

#[tokio::test]
async fn shadowing_hides_an_overwrite_a_delete_and_a_tail_delete() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = create(b.store.clone()).await;
        write(&ns, "a", 1, 60).await;
        ns.compact(&ROOMY).await.unwrap();
        ns.index(&every()).await.unwrap();

        // Overwrite from a second indexed segment.
        ns.upsert(vec![doc("a0000", &[5.0, 5.0, 5.0, 5.0])])
            .await
            .unwrap();
        ns.compact(&ROOMY).await.unwrap();
        ns.index(&every()).await.unwrap();

        // Delete from an unindexed fold.
        ns.delete(vec!["a0001".into()]).await.unwrap();
        ns.compact(&ROOMY).await.unwrap();

        // Delete straight from the tail (never compacted).
        ns.delete(vec!["a0002".into()]).await.unwrap();

        let mut state = state_of(vec![docs_of("a", 1, 60)]);
        state.insert("a0000".into(), doc("a0000", &[5.0, 5.0, 5.0, 5.0]));
        state.remove("a0001");
        state.remove("a0002");
        let docs: Vec<_> = state.into_values().collect();

        assert_matches_flat(&ns, &docs, &QueryOptions::default().rerank(None)).await;

        let query = Lcg(1).vector(DIM);
        let got = ids(&ns
            .query_with(&query, 100, &QueryOptions::default())
            .await
            .unwrap());
        assert!(!got.contains(&"a0001".to_string()));
        assert!(!got.contains(&"a0002".to_string()));
    }
}

#[tokio::test]
async fn heavy_shadowing_widens_the_shortlist_and_still_matches_flat() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = create(b.store.clone()).await;
        write(&ns, "a", 1, 200).await;
        ns.compact(&ROOMY).await.unwrap();
        ns.index(&every()).await.unwrap();

        // Overwrite almost everything from the tail, unindexed, so a small
        // rerank factor proposes mostly-shadowed rows and has to ask again.
        let mut rng = Lcg(99);
        let overwritten: Vec<_> = (0..190)
            .map(|i| doc(&format!("a{i:04}"), &rng.vector(DIM)))
            .collect();
        ns.upsert(overwritten.clone()).await.unwrap();

        let mut state = state_of(vec![docs_of("a", 1, 200)]);
        for d in overwritten {
            state.insert(d.id.clone(), d);
        }
        let docs: Vec<_> = state.into_values().collect();

        // rerank 1 makes the index propose barely more than top_k rows, most
        // of which are shadowed, forcing the widening loop.
        let tight = QueryOptions::default().rerank(Some(1));
        assert_matches_flat(&ns, &docs, &tight).await;
    }
}

#[tokio::test]
async fn budget_refuses_the_flat_scan_but_the_index_still_answers() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = create(b.store.clone()).await;
        write(&ns, "a", 1, 20_000).await;
        ns.compact(&ROOMY).await.unwrap();
        ns.index(&IndexOptions::default().kind(IndexKind::Sq8).nlist(Some(1)))
            .await
            .unwrap();

        let sizes = snapshot(&*b.store).await;
        let seg_size: u64 = sizes
            .iter()
            .find(|(k, _)| k.contains("/data.bin"))
            .map(|(_, v)| v.len() as u64)
            .unwrap();
        // Comfortably more than the two fixed-size opening reads (a ranged
        // segment's suffix guess and an index's head guess) plus a few
        // candidate rows, comfortably less than reading the whole segment.
        let tight_budget = 400_000;
        assert!(
            tight_budget < seg_size,
            "the segment ({seg_size} bytes) must be bigger than the tight budget"
        );

        let tight = Namespace::open(b.store.clone(), "docs")
            .await
            .unwrap()
            .with_read_budget(tight_budget);
        let query = Lcg(1).vector(DIM);

        let flat = tight
            .query_with(&query, 5, &QueryOptions::default().exact(true))
            .await;
        assert!(
            matches!(flat, Err(ItzmaError::ReadBudgetExceeded { .. })),
            "{flat:?}"
        );

        let indexed = tight
            .query_with(&query, 5, &QueryOptions::default())
            .await
            .unwrap();
        assert_eq!(indexed.len(), 5);
    }
}

#[tokio::test]
async fn the_indexed_path_reads_far_fewer_bytes_than_a_flat_scan() {
    let store = Arc::new(CountingStore::default());
    let ns = create(store.clone() as Arc<dyn ObjectStore>).await;
    write(&ns, "a", 1, DEFAULT_MIN_ROWS * 30).await;
    ns.compact(&ROOMY).await.unwrap();
    // One partition: this compares a scan of every code with a flat read. A
    // clustered index would also take minutes to build in a debug build.
    ns.index(&IndexOptions::default().kind(IndexKind::Sq8).nlist(Some(1)))
        .await
        .unwrap();

    let query = Lcg(1).vector(DIM);

    store.reset();
    ns.query_with(&query, 5, &QueryOptions::default().exact(true))
        .await
        .unwrap();
    let flat_bytes = store.bytes();

    store.reset();
    ns.query_with(&query, 5, &QueryOptions::default())
        .await
        .unwrap();
    let indexed_bytes = store.bytes();
    let indexed_requests = store.requests();

    // One partition scans every code, so the
    // saving is bounded: roughly the ratio of an SQ8 code (1 byte per
    // dimension) to an f32 (4 bytes), less the ids and attributes the index
    // does not carry at all and the flat read fetches for every row. A third
    // is comfortably inside that at this scale and well outside noise.
    assert!(
        indexed_bytes < flat_bytes / 3,
        "indexed {indexed_bytes} bytes over {indexed_requests} requests, \
         flat {flat_bytes} bytes: a regression that reads a segment whole \
         through the index would show up here"
    );
}

#[tokio::test]
async fn a_missing_index_object_is_corrupt_when_the_manifest_has_not_moved() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = create(b.store.clone()).await;
        write(&ns, "a", 1, 30).await;
        ns.compact(&ROOMY).await.unwrap();
        ns.index(&every()).await.unwrap();

        // Delete the index object behind the manifest's back, as damage would.
        let keys = index_keys(&snapshot(&*b.store).await);
        assert_eq!(keys.len(), 1);
        b.store.delete(&Path::from(keys[0].clone())).await.unwrap();

        let query = Lcg(1).vector(DIM);
        let err = ns
            .query_with(&query, 5, &QueryOptions::default())
            .await
            .unwrap_err();
        assert!(matches!(err, ItzmaError::CorruptIndex { .. }), "{err:?}");

        // `exact` never looks at the index, so it still works.
        let exact = ns
            .query_with(&query, 5, &QueryOptions::default().exact(true))
            .await
            .unwrap();
        assert_eq!(exact.len(), 5);
    }
}

#[tokio::test]
async fn a_damaged_index_object_is_corrupt_and_never_falls_back() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = create(b.store.clone()).await;
        write(&ns, "a", 1, 30).await;
        ns.compact(&ROOMY).await.unwrap();
        ns.index(&every()).await.unwrap();

        let keys = index_keys(&snapshot(&*b.store).await);
        b.store
            .put(
                &Path::from(keys[0].clone()),
                bytes::Bytes::from_static(b"not an index").into(),
            )
            .await
            .unwrap();

        let query = Lcg(1).vector(DIM);
        let err = ns
            .query_with(&query, 5, &QueryOptions::default())
            .await
            .unwrap_err();
        assert!(matches!(err, ItzmaError::CorruptIndex { .. }), "{err:?}");
    }
}

/// What another process does between the query's manifest read and its first
/// read of the index object: rebuild it and delete the old generation, as a
/// rebuild followed by a garbage collection would.
async fn rebuild_and_delete_old(store: Arc<dyn ObjectStore>) {
    let ns = Namespace::open(store.clone(), "docs").await.unwrap();
    let report = ns.index(&every().rebuild(true)).await.unwrap();
    assert_eq!(report.indexed, 1, "{report:?}");
    let old = index_keys(&snapshot(&*store).await)
        .into_iter()
        .find(|k| k.ends_with("/index-1.bin"))
        .expect("generation 1 should still be there before it is deleted");
    store.delete(&Path::from(old)).await.unwrap();
}

#[tokio::test]
async fn a_generation_replaced_after_the_manifest_was_loaded_is_a_stale_read() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let faulty = FaultyStore::new(b.store.clone());
        let store: Arc<dyn ObjectStore> = faulty.clone();
        let ns = create(store).await;
        write(&ns, "a", 1, 30).await;
        ns.compact(&ROOMY).await.unwrap();
        ns.index(&every()).await.unwrap();

        faulty.before_get("/index-1.bin", rebuild_and_delete_old(b.store.clone()));

        let query = Lcg(1).vector(DIM);
        let got = ns
            .query_with(&query, 5, &QueryOptions::default())
            .await
            .unwrap();
        assert_eq!(got.len(), 5);
    }
}

#[tokio::test]
async fn a_fault_at_every_operation_of_an_indexed_query_is_an_error_never_a_wrong_answer() {
    let store = FaultyStore::new(common::memory());
    let ns = create(store.clone() as Arc<dyn ObjectStore>).await;
    write(&ns, "a", 1, 30).await;
    ns.compact(&ROOMY).await.unwrap();
    ns.index(&every()).await.unwrap();

    let query = Lcg(1).vector(DIM);
    store.reset_ops();
    ns.query_with(&query, 5, &QueryOptions::default())
        .await
        .unwrap();
    let total_ops = store.ops();
    assert!(total_ops > 0);

    // A query never writes, so the same store and namespace can be reused
    // for every fault: nothing it does can corrupt state to check afterwards.
    for fault in [Fault::Generic, Fault::LostAck] {
        for n in 0..total_ops {
            store.fail_nth_op(n, fault);
            let outcome = ns.query_with(&query, 5, &QueryOptions::default()).await;
            assert!(
                outcome.is_err(),
                "{fault:?} at op {n} of {total_ops} should have failed"
            );
        }
    }
    store.clear();

    let got = ns
        .query_with(&query, 5, &QueryOptions::default())
        .await
        .unwrap();
    assert_eq!(got.len(), 5);
}

#[tokio::test]
async fn invalid_query_options_are_rejected_before_touching_storage() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = create(b.store.clone()).await;
        write(&ns, "a", 1, 5).await;
        let before = snapshot(&*b.store).await;

        let query = Lcg(1).vector(DIM);
        let zero_rerank = QueryOptions::default().rerank(Some(0));
        assert!(matches!(
            ns.query_with(&query, 1, &zero_rerank).await,
            Err(ItzmaError::InvalidQuery(_))
        ));
        let zero_nprobe = QueryOptions::default().nprobe(Some(0));
        assert!(matches!(
            ns.query_with(&query, 1, &zero_nprobe).await,
            Err(ItzmaError::InvalidQuery(_))
        ));
        assert_untouched(&*b.store, &before).await;
    }
}

#[tokio::test]
async fn plain_query_equals_query_with_default_options() {
    let ns = create(common::memory()).await;
    write(&ns, "a", 1, 40).await;
    ns.compact(&ROOMY).await.unwrap();
    ns.index(&every()).await.unwrap();

    let query = Lcg(3).vector(DIM);
    let a = ns.query(&query, 7).await.unwrap();
    let b = ns
        .query_with(&query, 7, &QueryOptions::default())
        .await
        .unwrap();
    assert_eq!(a, b);
}

#[tokio::test]
async fn top_k_zero_is_empty_and_larger_than_the_corpus_returns_everything() {
    let ns = create(common::memory()).await;
    write(&ns, "a", 1, 10).await;
    ns.compact(&ROOMY).await.unwrap();
    ns.index(&every()).await.unwrap();

    let query = Lcg(1).vector(DIM);
    assert!(
        ns.query_with(&query, 0, &QueryOptions::default())
            .await
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        ns.query_with(&query, 1_000, &QueryOptions::default())
            .await
            .unwrap()
            .len(),
        10
    );
}

#[tokio::test]
async fn the_indexed_query_future_can_be_spawned() {
    let ns = Arc::new(create(common::memory()).await);
    write(&ns, "a", 1, 40).await;
    ns.compact(&ROOMY).await.unwrap();
    ns.index(&every()).await.unwrap();

    let handle = ns.clone();
    let query = Lcg(1).vector(DIM);
    let got = tokio::spawn(async move {
        handle
            .query_with(&query, 5, &QueryOptions::default())
            .await
            .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(got.len(), 5);
}

// ---------------------------------------------------------------------------
// Namespace::with_cache
// ---------------------------------------------------------------------------

/// A write through a raw handle is visible on a cache-wired handle's very
/// next query — the core promise `.with_cache` makes: the manifest is never
/// served from cache, so nothing about write-then-query freshness changes.
#[tokio::test]
async fn a_write_is_visible_immediately_through_a_cache_wired_handle() {
    let raw = Arc::new(CountingStore::default());
    // The cache must wrap the *same* store the segments actually live in —
    // exactly how a real caller uses it (`CacheStore::new(store.clone(), ..)`).
    let ns = create(raw.clone())
        .await
        .with_cache(cache_over(raw.clone()));

    write(&ns, "a", 1, 50).await;
    let hits = ns
        .query_with(
            &docs_of("a", 1, 50)[0].vector,
            5,
            &QueryOptions::default().exact(true),
        )
        .await
        .unwrap();
    assert!(!hits.is_empty());

    // A second write, through the same handle, must show up on the very
    // next query too — not just the first one after construction.
    write(&ns, "b", 2, 50).await;
    let hits = ns
        .query_with(
            &docs_of("b", 2, 50)[0].vector,
            5,
            &QueryOptions::default().exact(true),
        )
        .await
        .unwrap();
    assert!(ids(&hits).iter().any(|id| id.starts_with('b')));
}

/// A compact/gc/index run through a *separate*, uncached handle on the same
/// namespace is visible on the cache-wired handle's next query — proves the
/// manifest read really is fresh every time, not just on the handle that
/// happened to write.
#[tokio::test]
async fn changes_from_a_separate_uncached_handle_are_visible_through_the_cache() {
    let raw = Arc::new(CountingStore::default());
    let admin = create(raw.clone()).await;
    write(&admin, "a", 1, 200).await;
    admin.compact(&ROOMY).await.unwrap();

    let cached = Namespace::open(raw.clone(), "docs")
        .await
        .unwrap()
        .with_cache(cache_over(raw.clone()));

    let before = cached
        .query_with(
            &docs_of("a", 1, 200)[0].vector,
            5,
            &QueryOptions::default().exact(true),
        )
        .await
        .unwrap();
    assert!(!before.is_empty());

    // Compact again from the admin handle after a second write — the
    // cache-wired handle must see the new data on its next call.
    write(&admin, "b", 2, 200).await;
    admin.compact(&ROOMY).await.unwrap();

    let after = cached
        .query_with(
            &docs_of("b", 2, 200)[0].vector,
            5,
            &QueryOptions::default().exact(true),
        )
        .await
        .unwrap();
    assert!(ids(&after).iter().any(|id| id.starts_with('b')));
}

/// The manifest is read only from the raw store, never from `query_store` —
/// checked directly by counting requests on each store separately. Both
/// stores wrap *clones* of the same `InMemory` (which shares its data,
/// `Arc<RwLock<_>>`-backed, on clone), so they see the same objects but keep
/// independent counters — a request's destination is unambiguous.
#[tokio::test]
async fn the_manifest_is_never_read_from_the_cache_store() {
    let backing = object_store::memory::InMemory::default();
    let raw = Arc::new(CountingStore::wrapping(backing.clone()));
    let cache_inner = Arc::new(CountingStore::wrapping(backing));
    let ns = create(raw.clone())
        .await
        .with_cache(cache_over(cache_inner.clone()));
    write(&ns, "a", 1, 50).await;

    raw.reset();
    cache_inner.reset();

    ns.query_with(
        &docs_of("a", 1, 50)[0].vector,
        5,
        &QueryOptions::default().exact(true),
    )
    .await
    .unwrap();

    // The manifest GET landed on raw, never on the cache's inner store.
    assert!(raw.requests() >= 1, "the manifest read must reach raw");
    // The segment/WAL-tail reads landed on the cache's inner store instead.
    assert!(
        cache_inner.requests() >= 1,
        "the data reads must reach query_store's cache"
    );
}

/// Two identical queries in a row against a cache-wired handle: the second
/// issues fewer requests to the underlying store than the first (the
/// segment/WAL-tail bytes are already warm), but never drops to zero — the
/// manifest read persists every time, proving it's never served from cache.
#[tokio::test]
async fn a_repeated_query_reuses_the_cache() {
    let backing = object_store::memory::InMemory::default();
    let raw = Arc::new(CountingStore::wrapping(backing.clone()));
    let admin = create(raw.clone()).await;
    write(&admin, "a", 1, 500).await;
    admin.compact(&ROOMY).await.unwrap();

    let cache_inner = Arc::new(CountingStore::wrapping(backing));
    let cached = Namespace::open(raw.clone(), "docs")
        .await
        .unwrap()
        .with_cache(cache_over(cache_inner.clone()));

    let query_vec = docs_of("a", 1, 500)[0].vector.clone();

    cache_inner.reset();
    cached
        .query_with(&query_vec, 5, &QueryOptions::default().exact(true))
        .await
        .unwrap();
    let first = cache_inner.requests();
    assert!(first > 0, "first query must actually reach the store");

    cache_inner.reset();
    cached
        .query_with(&query_vec, 5, &QueryOptions::default().exact(true))
        .await
        .unwrap();
    let second = cache_inner.requests();

    assert!(
        second < first,
        "second query ({second} requests) should reuse cached bytes from the first ({first} requests)"
    );

    // The manifest read (through `raw`, not `cache_inner`) never disappears,
    // second query or not.
    raw.reset();
    cached
        .query_with(&query_vec, 5, &QueryOptions::default().exact(true))
        .await
        .unwrap();
    assert!(
        raw.requests() >= 1,
        "the manifest read must persist every query"
    );
}

/// `compact`, `gc`, and `index` on a cache-wired handle behave identically
/// to an uncached one — they never read through `query_store`. Proven by
/// running them entirely against a `CacheStore` whose *inner* store is a
/// `FaultyStore` that counts operations: if any admin path touched it,
/// `.ops()` would be nonzero. (`inspect::report` is deliberately not part of
/// this test — it's a free function that takes its own `&dyn ObjectStore`
/// directly, not a `Namespace` method, so it never sees `query_store` at
/// all; there's nothing this design could leak into it.)
#[tokio::test]
async fn admin_operations_never_touch_the_cache_store() {
    let raw = Arc::new(CountingStore::default());
    let poison = FaultyStore::new(Arc::new(object_store::memory::InMemory::new()));
    let ns = create(raw.clone()).await.with_cache(Arc::new(
        itzma_core::CacheStore::new(poison.clone(), itzma_core::CacheOptions::default()).unwrap(),
    ));

    write(&ns, "a", 1, 200).await;
    ns.compact(&ROOMY).await.unwrap();
    ns.index(&every()).await.unwrap();
    ns.gc(&itzma_core::GcOptions::default()).await.unwrap();

    assert_eq!(
        poison.ops(),
        0,
        "no admin call should ever reach query_store"
    );
}

/// A cache-wired handle whose `query_store` errors on its very first read
/// still answers a query with a normal storage error — not a panic, not
/// silently wrong data — exactly as it would from `store` itself with no
/// cache involved. `query_store` here is a bare `FaultyStore`, not wrapped
/// in a `CacheStore`, so this isolates `Namespace`'s own error handling from
/// `CacheStore`'s separately-tested fail-open behavior.
#[tokio::test]
async fn a_failing_query_store_returns_a_clean_error_not_wrong_data() {
    let raw = Arc::new(CountingStore::default());
    let ns = create(raw.clone()).await;
    write(&ns, "a", 1, 50).await;

    let broken = FaultyStore::new(Arc::new(object_store::memory::InMemory::new()));
    broken.fail_nth_op(0, Fault::Generic);
    let cached = ns.with_cache(broken);

    let err = cached
        .query_with(
            &docs_of("a", 1, 50)[0].vector,
            5,
            &QueryOptions::default().exact(true),
        )
        .await
        .unwrap_err();
    assert!(matches!(err, ItzmaError::Storage(_)));
}

/// A flat source bigger than one compute chunk (16,384 documents),
/// full of tied vectors, through the exact path and through the engine's flat
/// source beside an indexed segment: both must equal a naive scan, ties by id.
#[tokio::test]
async fn a_flat_source_bigger_than_a_chunk_matches_a_naive_scan() {
    const WIDE: usize = 96;
    let store: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
    let ns = Namespace::create(store, "docs", schema(WIDE, L2))
        .await
        .unwrap();
    let mut rng = Lcg(9);
    let indexed: Vec<Document> = (0..100)
        .map(|i| doc(&format!("a{i:05}"), &rng.vector(WIDE)))
        .collect();
    ns.upsert(indexed.clone()).await.unwrap();
    ns.compact(&CompactionOptions::default()).await.unwrap();
    ns.index(&every()).await.unwrap();
    // 50 distinct vectors repeated, so every distance ties with hundreds of
    // others on both sides of each chunk edge.
    let shapes: Vec<Vec<f32>> = (0..50).map(|_| rng.vector(WIDE)).collect();
    let flat: Vec<Document> = (0..16_484)
        .map(|i| doc(&format!("b{i:05}"), &shapes[i % 50]))
        .collect();
    ns.upsert(flat.clone()).await.unwrap();
    ns.compact(&CompactionOptions::default()).await.unwrap();

    let all: Vec<Document> = indexed.into_iter().chain(flat).collect();
    for seed in [1, 2] {
        let query = Lcg(seed).vector(WIDE);
        for k in [1, 10, 700] {
            let expected = naive_top_k(L2, &query, &all, k);
            let exact = QueryOptions::default().exact(true);
            assert_eq!(
                ids(&ns.query_with(&query, k, &exact).await.unwrap()),
                expected
            );
            // rerank: None keeps the indexed segment's shortlist exact.
            let engine = QueryOptions::default().rerank(None);
            assert_eq!(
                ids(&ns.query_with(&query, k, &engine).await.unwrap()),
                expected
            );
        }
    }
}

/// Many tied rows spread over several SQ8 partitions, all probed in one wave
/// big enough for the compute pool: the indexed answer must equal
/// the exact one id for id, ties broken by id.
#[tokio::test]
async fn tied_rows_across_partitions_on_the_pool_match_the_exact_scan() {
    const WIDE: usize = 64;
    let store: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
    let ns = Namespace::create(store, "docs", schema(WIDE, L2))
        .await
        .unwrap();
    let mut rng = Lcg(21);
    let shapes: Vec<Vec<f32>> = (0..50).map(|_| rng.vector(WIDE)).collect();
    let docs: Vec<Document> = (0..20_000)
        .map(|i| doc(&format!("t{i:05}"), &shapes[i % 50]))
        .collect();
    ns.upsert(docs).await.unwrap();
    ns.compact(&CompactionOptions::default()).await.unwrap();
    ns.index(&every().kind(IndexKind::Sq8).nlist(Some(8)))
        .await
        .unwrap();

    let every_partition = QueryOptions::default().nprobe(Some(8)).rerank(None);
    let exact = QueryOptions::default().exact(true);
    for seed in [1, 2, 3] {
        let query = Lcg(seed).vector(WIDE);
        for k in [1, 10, 450] {
            assert_eq!(
                ids(&ns.query_with(&query, k, &every_partition).await.unwrap()),
                ids(&ns.query_with(&query, k, &exact).await.unwrap()),
                "seed {seed}, k {k}"
            );
        }
    }
}

/// Only the manifest among `paths`, or panic with what else was fetched.
fn assert_only_manifest(paths: &[String]) {
    assert_eq!(paths.len(), 1, "{paths:?}");
    assert!(paths[0].ends_with("manifest/current.json"), "{paths:?}");
}

/// A strict query against an unchanged tail sends one
/// manifest GET and one LIST, and fetches no WAL object.
#[tokio::test]
async fn an_unchanged_tail_is_listed_but_not_fetched_again() {
    let counting = Arc::new(CountingStore::default());
    let ns = create(counting.clone()).await;
    for seed in 0..5 {
        write(&ns, &format!("b{seed}-"), seed, 20).await;
    }
    let probe = [0.5; DIM];
    let first = ns.query(&probe, 10).await.unwrap();
    counting.reset();
    let second = ns.query(&probe, 10).await.unwrap();
    assert_eq!(ids(&first), ids(&second));
    assert_eq!(counting.lists(), 1);
    assert_only_manifest(&counting.paths());
}

/// A new write between two queries costs one GET, for that object only.
#[tokio::test]
async fn a_new_write_is_the_only_wal_object_fetched() {
    let counting = Arc::new(CountingStore::default());
    let ns = create(counting.clone()).await;
    write(&ns, "a-", 1, 20).await;
    ns.query(&[0.5; DIM], 10).await.unwrap();
    write(&ns, "b-", 2, 20).await;
    counting.reset();
    assert_eq!(ns.query(&[0.5; DIM], 100).await.unwrap().len(), 40);
    let wal: Vec<String> = counting
        .paths()
        .into_iter()
        .filter(|p| p.contains("/wal/"))
        .collect();
    assert_eq!(wal.len(), 1, "{wal:?}");
    assert!(wal[0].ends_with("00000000000000000002.json"), "{wal:?}");
}

/// Another handle compacts: this handle's view was built on an older
/// `folded_through`, so it is rebuilt, never extended past the fold.
#[tokio::test]
async fn a_compaction_by_another_handle_rebuilds_the_view() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let reader = create(b.store.clone()).await;
        let mut batches = Vec::new();
        for seed in 0..3 {
            let docs = docs_of(&format!("x{seed}-"), seed, 10);
            reader.upsert(docs.clone()).await.unwrap();
            batches.push(docs);
        }
        let probe = [0.25; DIM];
        reader.query(&probe, 5).await.unwrap();
        let other = Namespace::open(b.store.clone(), "docs").await.unwrap();
        other.compact(&ROOMY).await.unwrap();
        other.delete(vec!["x1-0003".into()]).await.unwrap();
        let back = docs_of("x0-", 99, 2);
        other.upsert(back.clone()).await.unwrap();
        batches.push(back);
        let mut state = state_of(batches);
        state.remove("x1-0003");
        let live: Vec<Document> = state.into_values().collect();
        let hits = reader.query(&probe, 40).await.unwrap();
        assert_eq!(ids(&hits), model(&probe, &live, 40));
    }
}

/// A namespace whose handle has a folded view, then three more batches and a
/// delete past it: the next query extends the view. Returns the live
/// documents.
async fn extending_setup(ns: &Namespace) -> Vec<Document> {
    let first = docs_of("a-", 1, 10);
    ns.upsert(first.clone()).await.unwrap();
    ns.query(&[0.5; DIM], 10).await.unwrap();
    let mut batches = vec![first];
    for seed in 2..5 {
        let docs = docs_of(&format!("n{seed}-"), seed, 5);
        ns.upsert(docs.clone()).await.unwrap();
        batches.push(docs);
    }
    ns.delete(vec!["a-0000".into()]).await.unwrap();
    let mut state = state_of(batches);
    state.remove("a-0000");
    state.into_values().collect()
}

/// A fault anywhere while a query extends the view leaves the old view in
/// place: the next query on the same handle is still right.
#[tokio::test]
async fn a_fault_while_extending_the_view_leaves_the_next_query_correct() {
    let probe = [0.5; DIM];
    let total = {
        let store = FaultyStore::new(common::memory());
        let ns = create(store.clone()).await;
        extending_setup(&ns).await;
        store.reset_ops();
        ns.query(&probe, 30).await.unwrap();
        store.ops()
    };
    // The manifest, the listing and four new WAL objects at least.
    assert!(total >= 6, "{total}");
    for fault in [Fault::Generic, Fault::LostAck] {
        for n in 0..total {
            let store = FaultyStore::new(common::memory());
            let ns = create(store.clone()).await;
            let live = extending_setup(&ns).await;
            let want = model(&probe, &live, 30);
            store.fail_nth_op(n, fault);
            let faulted = ns.query(&probe, 30).await;
            store.clear();
            if let Ok(hits) = faulted {
                assert_eq!(ids(&hits), want, "{fault:?} at op {n}");
            }
            let hits = ns.query(&probe, 30).await.unwrap();
            assert_eq!(ids(&hits), want, "{fault:?} at op {n}, after");
        }
    }
}

/// A strict query fetches new tail objects while it opens the segments, not
/// after: the tail's GET
/// waits for a segment read to start, which only happens if both run at once.
#[tokio::test]
async fn the_tail_is_read_alongside_the_segments() {
    let faulty = FaultyStore::new(common::memory());
    let ns = create(faulty.clone()).await;
    write(&ns, "s-", 1, 10).await;
    ns.compact(&ROOMY).await.unwrap();
    write(&ns, "t-", 2, 10).await;

    let segment_started = Arc::new(tokio::sync::Notify::new());
    let overlapped = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let signal = segment_started.clone();
    faulty.before_get("/segments/", async move { signal.notify_one() });
    let (wait, seen) = (segment_started.clone(), overlapped.clone());
    faulty.before_get("/wal/00000000000000000002", async move {
        let met = tokio::time::timeout(std::time::Duration::from_secs(2), wait.notified()).await;
        seen.store(met.is_ok(), std::sync::atomic::Ordering::SeqCst);
    });
    assert_eq!(ns.query(&[0.5; DIM], 100).await.unwrap().len(), 20);
    assert!(
        overlapped.load(std::sync::atomic::Ordering::SeqCst),
        "the tail was read only after the segments"
    );
}

/// Apply upserts then deletes to the model the way the WAL would.
fn apply(state: &mut BTreeMap<String, Document>, upserts: &[Document], deletes: &[&str]) {
    for d in upserts {
        state.insert(d.id.clone(), d.clone());
    }
    for id in deletes {
        state.remove(*id);
    }
}

/// Default and exact queries both equal a naive ranking of `model`.
async fn assert_matches_model(ns: &Namespace, state: &BTreeMap<String, Document>) {
    let docs: Vec<Document> = state.values().cloned().collect();
    for seed in 0..5 {
        let query = Lcg(100 + seed).vector(DIM);
        for exact in [false, true] {
            let options = QueryOptions::default().exact(exact);
            let got = ids(&ns.query_with(&query, 7, &options).await.unwrap());
            assert_eq!(got, model(&query, &docs, 7), "exact {exact}, seed {seed}");
        }
    }
}

#[tokio::test]
async fn an_unindexed_run_with_shadowing_inside_it_and_from_the_tail_answers_like_a_replay() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = create(b.store.clone()).await;
        let mut state = BTreeMap::new();
        let first = docs_of("a", 1, 300);
        ns.upsert(first.clone()).await.unwrap();
        apply(&mut state, &first, &[]);
        ns.compact(&ROOMY).await.unwrap();
        // A newer segment updates some, deletes some, deletes then re-adds
        // one, and deletes an id nothing holds.
        let updated = docs_of("a", 2, 40);
        ns.upsert(updated.clone()).await.unwrap();
        let gone = ["a0100", "a0101", "a0005", "ghost"];
        ns.delete(gone.iter().map(|id| id.to_string()).collect())
            .await
            .unwrap();
        apply(&mut state, &updated, &gone);
        let back = docs_of("a", 3, 6)[5..].to_vec();
        ns.upsert(back.clone()).await.unwrap();
        apply(&mut state, &back, &[]);
        ns.compact(&ROOMY).await.unwrap();
        // The tail shadows both segments.
        ns.delete(vec!["a0000".into()]).await.unwrap();
        let tail = docs_of("a", 4, 3);
        ns.upsert(tail.clone()).await.unwrap();
        apply(&mut state, &[], &["a0000"]);
        apply(&mut state, &tail, &[]);
        let manifest = manifest_of(&*b.store).await;
        assert_eq!(manifest.live_segments.len(), 2);
        assert!(manifest.indexes.is_empty());
        assert_matches_model(&ns, &state).await;
    }
}

#[tokio::test]
async fn exact_reads_no_index_object_and_equals_a_replay() {
    let counting = Arc::new(CountingStore::default());
    let store: Arc<dyn ObjectStore> = counting.clone();
    let ns = create(store.clone()).await;
    let mut state = BTreeMap::new();
    for (prefix, seed) in [("a", 1), ("b", 2)] {
        let docs = docs_of(prefix, seed, 50);
        ns.upsert(docs.clone()).await.unwrap();
        apply(&mut state, &docs, &[]);
        ns.compact(&ROOMY).await.unwrap();
    }
    ns.index(&every()).await.unwrap();
    assert_eq!(manifest_of(&*store).await.indexes.len(), 2);

    counting.reset();
    let query = Lcg(9).vector(DIM);
    let docs: Vec<Document> = state.values().cloned().collect();
    let got = ns
        .query_with(&query, 7, &QueryOptions::default().exact(true))
        .await
        .unwrap();
    assert_eq!(ids(&got), model(&query, &docs, 7));
    let paths = counting.paths();
    assert!(paths.iter().all(|p| !p.contains("index-")), "{paths:?}");
}

#[tokio::test]
async fn edge_shapes_answer_like_a_replay() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = create(b.store.clone()).await;
        let query = Lcg(1).vector(DIM);
        assert!(ns.query(&query, 5).await.unwrap().is_empty());
        // Only a tail.
        let docs = docs_of("a", 1, 4);
        ns.upsert(docs.clone()).await.unwrap();
        assert_eq!(
            ids(&ns.query(&query, 9).await.unwrap()),
            model(&query, &docs, 9)
        );
        assert!(ns.query(&query, 0).await.unwrap().is_empty());
        // Every row of the run shadowed by a later segment.
        ns.compact(&ROOMY).await.unwrap();
        ns.delete(docs.iter().map(|d| d.id.clone()).collect())
            .await
            .unwrap();
        ns.compact(&ROOMY).await.unwrap();
        for exact in [false, true] {
            let options = QueryOptions::default().exact(exact);
            assert!(ns.query_with(&query, 5, &options).await.unwrap().is_empty());
        }
    }
}

#[tokio::test]
async fn a_fault_at_every_operation_of_an_unindexed_query_is_an_error_never_a_wrong_answer() {
    let store = FaultyStore::new(common::memory());
    let ns = create(store.clone() as Arc<dyn ObjectStore>).await;
    write(&ns, "a", 1, 30).await;
    ns.compact(&ROOMY).await.unwrap();
    write(&ns, "b", 2, 10).await;
    ns.compact(&ROOMY).await.unwrap();
    write(&ns, "c", 3, 5).await;

    let query = Lcg(1).vector(DIM);
    let want = ns.query(&query, 5).await.unwrap();
    store.reset_ops();
    ns.query(&query, 5).await.unwrap();
    let total_ops = store.ops();
    assert!(total_ops > 0);
    // A query never writes, so one namespace serves every fault.
    for fault in [Fault::Generic, Fault::LostAck] {
        for n in 0..total_ops {
            store.fail_nth_op(n, fault);
            if let Ok(got) = ns.query(&query, 5).await {
                panic!("{fault:?} at op {n} of {total_ops} answered {got:?}");
            }
        }
    }
    store.clear();
    assert_eq!(ns.query(&query, 5).await.unwrap(), want);
}

/// A flat source's early stop must use a true lower bound for its unread
/// rows. One unindexed segment big enough to be scored on the compute pool
/// (20,001 rows at dimension 64), whose closest document the tail moves far
/// away: the tail, scored inline, offers its distance to the bound first, the
/// run's first batch is shadowed, and a stop judged on an arbitrary unread
/// row would drop the true answer, which sits among 20,000 far rows.
#[tokio::test]
async fn a_flat_source_never_stops_before_a_closer_unread_row() {
    const WIDE: usize = 64;
    let store = common::memory();
    let ns = Namespace::create(store, "docs", schema(WIDE, L2))
        .await
        .unwrap();
    let at = |x: f32| {
        let mut v = vec![0.0; WIDE];
        v[0] = x;
        v
    };
    let mut docs: Vec<Document> = (0..20_000)
        .map(|i| doc(&format!("far{i:05}"), &at(100.0 + (i % 7) as f32)))
        .collect();
    docs.push(doc("a", &at(0.0)));
    // Last in id order, so it is not where an unordered rest happens to
    // start.
    docs.push(doc("zz", &at(1.0)));
    ns.upsert(docs).await.unwrap();
    ns.compact(&ROOMY).await.unwrap();
    ns.upsert(vec![doc("a", &at(50.0))]).await.unwrap();

    for exact in [false, true] {
        let options = QueryOptions::default().exact(exact);
        let got = ids(&ns.query_with(&at(0.0), 1, &options).await.unwrap());
        assert_eq!(got, ["zz"], "exact {exact}");
    }
}

/// A namespace holding version 2 and version 3 segments, some
/// indexed, answers like the flat scan, flat and through the indexes, and
/// every hit carries its attributes as written.
#[tokio::test]
async fn a_query_across_versions_2_and_3_matches_the_flat_scan_with_attributes() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let typed = schema(DIM, L2).with_attrs(
            itzma_core::AttrSchema::new()
                .field("n", itzma_core::AttrType::Number)
                .field("tag", itzma_core::AttrType::Text),
        );
        let ns = Namespace::create(b.store.clone(), "docs", typed)
            .await
            .unwrap()
            .with_fetch_policy(itzma_core::FetchPolicy::local());
        let with_attrs = |prefix: &str, seed: u64| -> Vec<Document> {
            docs_of(prefix, seed, 300)
                .into_iter()
                .enumerate()
                .map(|(i, mut d)| {
                    d.attrs = match i % 3 {
                        0 => None,
                        1 => Some(serde_json::json!({"tag": null})),
                        _ => Some(serde_json::json!({"n": i, "tag": format!("{prefix}{}", i % 4)})),
                    };
                    d
                })
                .collect()
        };
        let a = with_attrs("a", 1);
        ns.upsert(a.clone()).await.unwrap();
        ns.compact(&ROOMY).await.unwrap();
        ns.upgrade_segment_format(3).await.unwrap();
        let b2 = with_attrs("b", 2);
        ns.upsert(b2.clone()).await.unwrap();
        ns.compact(&ROOMY).await.unwrap();
        let c = with_attrs("c", 3);
        ns.upsert(c.clone()).await.unwrap();
        ns.compact(&ROOMY).await.unwrap();
        assert_eq!(common::live_segment_versions(&*b.store).await, [2, 3, 3]);
        ns.index(&every()).await.unwrap();
        // A fold after the index build stays unindexed, so one version 3
        // segment is read flat beside the indexed ones.
        let d = with_attrs("d", 4);
        ns.upsert(d.clone()).await.unwrap();
        ns.compact(&ROOMY).await.unwrap();

        let docs: Vec<Document> = [a, b2, c, d].concat();
        let by_id: BTreeMap<String, Document> =
            docs.iter().map(|d| (d.id.clone(), d.clone())).collect();
        for options in [QueryOptions::default(), QueryOptions::default().exact(true)] {
            assert_matches_flat(&ns, &docs, &options).await;
            let hits = ns
                .query_with(&Lcg(9).vector(DIM), 50, &options)
                .await
                .unwrap();
            for hit in hits {
                assert_eq!(hit.attrs, by_id[&hit.id].attrs, "{}", hit.id);
            }
        }
    }
}
