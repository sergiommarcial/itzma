//! Filtering vector queries: the answer equals a naive
//! filter-then-rank over the replayed writes, at every stage a namespace goes
//! through.

mod common;

use common::counting::CountingStore;

use std::collections::BTreeMap;
use std::sync::Arc;

use common::faulty::{Fault, FaultyStore};
use common::{Lcg, cas_backends, doc, naive_top_k};
use itzma_core::{
    AttrSchema, AttrType, CompactionOptions, Compare, Document, IndexOptions, ItzmaError, Metric,
    Namespace, Predicate, QueryOptions, Schema,
};
use object_store::{ObjectStore, ObjectStoreExt};
use serde_json::json;

const DIM: usize = 4;
const ROOMY: CompactionOptions = CompactionOptions {
    max_segments: 8,
    max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
};
const TIGHT: CompactionOptions = CompactionOptions {
    max_segments: 1,
    max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
};

fn typed() -> Schema {
    Schema::new(DIM, Metric::L2).with_attrs(
        AttrSchema::new()
            .field("price", AttrType::Number)
            .field("active", AttrType::Bool),
    )
}

async fn create(store: Arc<dyn ObjectStore>) -> Namespace {
    Namespace::create(store, "docs", typed()).await.unwrap()
}

/// A document whose price is `price` and whose vector comes from `rng`.
fn item(id: &str, price: f64, rng: &mut Lcg) -> Document {
    let mut d = doc(id, &rng.vector(DIM));
    d.attrs = Some(json!({"price": price, "active": price < 5.0}));
    d
}

fn cheap() -> Predicate {
    Predicate::number("price", Compare::Lt, 3.0)
}

fn is_cheap(d: &Document) -> bool {
    d.attrs
        .as_ref()
        .and_then(|a| a["price"].as_f64())
        .is_some_and(|p| p < 3.0)
}

/// Fold writes as `upsert` does: the last write for an id wins.
fn state_of(batches: &[Vec<Document>]) -> BTreeMap<String, Document> {
    let mut state = BTreeMap::new();
    for d in batches.iter().flatten() {
        state.insert(d.id.clone(), d.clone());
    }
    state
}

fn ids(hits: &[itzma_core::Hit]) -> Vec<String> {
    hits.iter().map(|h| h.id.clone()).collect()
}

/// `ns` answers every query like a naive filter-then-rank of `state`, on the
/// flat path, through the indexes and with every row proposed.
async fn assert_matches_naive(
    ns: &Namespace,
    state: &BTreeMap<String, Document>,
    keep: fn(&Document) -> bool,
    filter: &Predicate,
) {
    let matching: Vec<Document> = state.values().filter(|d| keep(d)).cloned().collect();
    let variants = [
        QueryOptions::default(),
        QueryOptions::default().rerank(None),
        QueryOptions::default().exact(true),
    ];
    for options in variants {
        let options = options.filter(Some(filter.clone()));
        for seed in [1, 2, 3, 40] {
            for k in [1, 5, state.len() + 10] {
                let query = Lcg(seed).vector(DIM);
                let got = ids(&ns.query_with(&query, k, &options).await.unwrap());
                let want = naive_top_k(Metric::L2, &query, &matching, k);
                assert_eq!(got, want, "seed {seed}, k {k}, {options:?}");
            }
        }
    }
}

#[tokio::test]
async fn a_filtered_flat_query_returns_only_matching_documents() {
    for backend in cas_backends() {
        let ns = create(backend.store.clone()).await;
        let mut rng = Lcg(7);
        let batch: Vec<Document> = (0..40)
            .map(|i| item(&format!("d{i:03}"), f64::from(i % 10), &mut rng))
            .collect();
        ns.upsert(batch.clone()).await.unwrap();
        let state = state_of(&[batch]);
        assert_matches_naive(&ns, &state, is_cheap, &cheap()).await;
    }
}

#[tokio::test]
async fn a_predicate_that_does_not_fit_the_schema_is_refused_before_storage() {
    for backend in cas_backends() {
        let ns = create(backend.store.clone()).await;
        let q = [0.0; DIM];
        let unknown = Predicate::number("missing", Compare::Lt, 1.0);
        let wrong_type = Predicate::is_true("price");
        for bad in [unknown, wrong_type] {
            let options = QueryOptions::default().filter(Some(bad));
            let err = ns.query_with(&q, 3, &options).await.unwrap_err();
            assert!(
                matches!(
                    err,
                    ItzmaError::UnknownAttr { .. } | ItzmaError::AttrTypeMismatch { .. }
                ),
                "{err:?}"
            );
        }
    }
}

#[test]
fn the_filter_option_survives_a_serde_round_trip() {
    let options = QueryOptions::default().filter(Some(cheap()));
    let json = serde_json::to_string(&options).unwrap();
    let back: QueryOptions = serde_json::from_str(&json).unwrap();
    assert_eq!(back, options);
    assert_eq!(QueryOptions::default().filter, None);
}

/// Index every segment, however small.
fn every() -> IndexOptions {
    IndexOptions::default().min_rows(0)
}

/// With `raise_to_3`, the namespace moves to segment format 3
/// after its first fold, so version 2 and 3 segments are searched together.
async fn filtered_stages(raise_to_3: bool) {
    for backend in cas_backends() {
        let ns = create(backend.store.clone()).await;
        let mut rng = Lcg(11);
        let first: Vec<Document> = (0..60)
            .map(|i| item(&format!("d{i:03}"), f64::from(i % 10), &mut rng))
            .collect();
        let mut batches = vec![first.clone()];
        ns.upsert(first).await.unwrap();

        assert_matches_naive(&ns, &state_of(&batches), is_cheap, &cheap()).await;

        ns.compact(&ROOMY).await.unwrap();
        if raise_to_3 {
            ns.upgrade_segment_format(3).await.unwrap();
        }
        assert_matches_naive(&ns, &state_of(&batches), is_cheap, &cheap()).await;

        ns.index(&every()).await.unwrap();
        assert_matches_naive(&ns, &state_of(&batches), is_cheap, &cheap()).await;

        // Overwrites move documents across the predicate in both directions:
        // d005 (price 5) becomes cheap, d001 (price 1) stops being cheap. d002
        // is deleted. All three live in the indexed segment, so the tail and the
        // shadow check decide them.
        let second = vec![
            item("d005", 1.0, &mut rng),
            item("d001", 9.0, &mut rng),
            item("n000", 0.0, &mut rng),
        ];
        ns.upsert(second.clone()).await.unwrap();
        ns.delete(vec!["d002".into()]).await.unwrap();
        batches.push(second);
        let mut state = state_of(&batches);
        state.remove("d002");
        assert_matches_naive(&ns, &state, is_cheap, &cheap()).await;

        // An unindexed segment above an indexed one, then two indexed segments
        // with a delete from the tail above them.
        ns.compact(&ROOMY).await.unwrap();
        assert_matches_naive(&ns, &state, is_cheap, &cheap()).await;
        ns.index(&every()).await.unwrap();
        assert_matches_naive(&ns, &state, is_cheap, &cheap()).await;
        ns.delete(vec!["d003".into(), "n000".into()]).await.unwrap();
        state.remove("d003");
        state.remove("n000");
        assert_matches_naive(&ns, &state, is_cheap, &cheap()).await;

        ns.compact(&TIGHT).await.unwrap();
        assert_matches_naive(&ns, &state, is_cheap, &cheap()).await;

        ns.index(&every()).await.unwrap();
        assert_matches_naive(&ns, &state, is_cheap, &cheap()).await;
        if raise_to_3 {
            let versions = common::live_segment_versions(&*backend.store).await;
            assert!(versions.contains(&3), "{versions:?}");
        }
    }
}

#[tokio::test]
async fn a_filtered_query_equals_filter_then_rank_at_every_stage() {
    filtered_stages(false).await;
}

#[tokio::test]
async fn a_filtered_query_equals_filter_then_rank_across_versions_2_and_3() {
    filtered_stages(true).await;
}

#[tokio::test]
async fn an_old_version_that_matches_does_not_resurface_when_the_new_one_does_not() {
    for backend in cas_backends() {
        let ns = create(backend.store.clone()).await;
        let mut rng = Lcg(5);
        let query = rng.vector(DIM);
        let mut near = doc("near", &query);
        near.attrs = Some(json!({"price": 1.0}));
        let mut far = doc("far", &[9.0, 9.0, 9.0, 9.0]);
        far.attrs = Some(json!({"price": 1.0}));
        ns.upsert(vec![near, far]).await.unwrap();
        ns.compact(&ROOMY).await.unwrap();
        ns.index(&every()).await.unwrap();

        // The newest `near` costs 8, so it no longer matches. The indexed copy
        // that does match must stay hidden.
        let mut newer = doc("near", &query);
        newer.attrs = Some(json!({"price": 8.0}));
        ns.upsert(vec![newer]).await.unwrap();

        let options = QueryOptions::default().filter(Some(cheap()));
        let got = ids(&ns.query_with(&query, 5, &options).await.unwrap());
        assert_eq!(got, ["far"], "{}", backend.name);
    }
}

#[tokio::test]
async fn a_filter_matching_fewer_than_top_k_returns_exactly_those() {
    for backend in cas_backends() {
        let ns = create(backend.store.clone()).await;
        let mut rng = Lcg(3);
        let batch: Vec<Document> = (0..300)
            .map(|i| {
                let price = if i % 100 == 7 { 0.0 } else { 100.0 };
                item(&format!("d{i:04}"), price, &mut rng)
            })
            .collect();
        ns.upsert(batch.clone()).await.unwrap();
        ns.compact(&ROOMY).await.unwrap();
        ns.index(&every()).await.unwrap();

        let query = Lcg(9).vector(DIM);
        let options = QueryOptions::default().filter(Some(cheap()));
        let got = ids(&ns.query_with(&query, 10, &options).await.unwrap());
        let matching: Vec<Document> = batch.into_iter().filter(is_cheap).collect();
        assert_eq!(got, naive_top_k(Metric::L2, &query, &matching, 10));
        assert_eq!(got.len(), 3, "{}", backend.name);
    }
}

#[tokio::test]
async fn a_filter_that_matches_nothing_gives_an_empty_answer() {
    for backend in cas_backends() {
        let ns = create(backend.store.clone()).await;
        let mut rng = Lcg(2);
        let batch = (0..50)
            .map(|i| item(&format!("d{i:02}"), 100.0, &mut rng))
            .collect();
        ns.upsert(batch).await.unwrap();
        ns.compact(&ROOMY).await.unwrap();
        ns.index(&every()).await.unwrap();
        let options = QueryOptions::default().filter(Some(cheap()));
        let hits = ns.query_with(&[0.0; DIM], 5, &options).await.unwrap();
        assert!(hits.is_empty(), "{}", backend.name);
    }
}

#[tokio::test]
async fn hits_of_a_filtered_indexed_query_carry_their_attributes() {
    for backend in cas_backends() {
        let ns = create(backend.store.clone()).await;
        let mut rng = Lcg(4);
        let batch = (0..30)
            .map(|i| item(&format!("d{i:02}"), f64::from(i % 10), &mut rng))
            .collect();
        ns.upsert(batch).await.unwrap();
        ns.compact(&ROOMY).await.unwrap();
        ns.index(&every()).await.unwrap();
        let options = QueryOptions::default().filter(Some(cheap()));
        let hits = ns.query_with(&[0.5; DIM], 5, &options).await.unwrap();
        assert!(!hits.is_empty());
        for hit in hits {
            let price = hit.attrs.as_ref().and_then(|a| a["price"].as_f64());
            assert!(price.is_some_and(|p| p < 3.0), "{hit:?}");
        }
    }
}

/// A namespace of indexed segments behind a faulty store.
async fn indexed_namespace(store: Arc<dyn ObjectStore>) -> Namespace {
    let ns = create(store).await;
    let batch: Vec<Document> = (0..200u32)
        .map(|i| {
            item(
                &format!("d{i:04}"),
                f64::from(i % 10),
                &mut Lcg(u64::from(i) + 1),
            )
        })
        .collect();
    ns.upsert(batch).await.unwrap();
    ns.compact(&ROOMY).await.unwrap();
    ns.index(&every()).await.unwrap();
    ns
}

#[tokio::test]
async fn a_fault_at_every_operation_of_a_filtered_query_is_an_error_never_a_wrong_answer() {
    let store = FaultyStore::new(common::memory());
    let ns = indexed_namespace(store.clone() as Arc<dyn ObjectStore>).await;
    let dense = QueryOptions::default().rerank(None).filter(Some(cheap()));
    let query = Lcg(1).vector(DIM);

    store.reset_ops();
    let want_dense = ids(&ns.query_with(&query, 5, &dense).await.unwrap());
    let dense_ops = store.ops();
    assert!(dense_ops > 0);

    // A query never writes, so one store and namespace serve every fault.
    for fault in [Fault::Generic, Fault::LostAck] {
        for n in 0..dense_ops {
            store.fail_nth_op(n, fault);
            let outcome = ns.query_with(&query, 5, &dense).await;
            assert!(outcome.is_err(), "dense {fault:?} at op {n} of {dense_ops}");
        }
    }
    store.clear();
    assert_eq!(
        ids(&ns.query_with(&query, 5, &dense).await.unwrap()),
        want_dense
    );
}

#[tokio::test]
async fn a_selective_filter_over_a_small_budget_is_refused_and_a_big_one_answers() {
    let store = common::memory();
    let ns = indexed_namespace(store.clone()).await;
    let dense = QueryOptions::default()
        .rerank(None)
        .filter(Some(Predicate::number("price", Compare::Eq, 7.0)));
    let query = Lcg(1).vector(DIM);
    let want = ids(&ns.query_with(&query, 500, &dense).await.unwrap());
    assert_eq!(want.len(), 20);

    // Below some budget the scan cannot afford the ids and attributes it must
    // read. Above it the answer is the same as unbudgeted.
    let mut budget = 64;
    let mut refused = 0;
    loop {
        let small = Namespace::open(store.clone(), "docs")
            .await
            .unwrap()
            .with_read_budget(budget);
        match small.query_with(&query, 500, &dense).await {
            Err(ItzmaError::ReadBudgetExceeded { .. }) => refused += 1,
            Err(other) => panic!("budget {budget}: {other:?}"),
            Ok(hits) => {
                assert_eq!(ids(&hits), want, "budget {budget}");
                break;
            }
        }
        budget *= 2;
        assert!(budget < 1 << 30, "never fit");
    }
    assert!(refused > 0, "the first budget was already enough");
}

#[tokio::test]
async fn a_refused_filter_costs_no_storage_call() {
    let faulty = FaultyStore::new(common::memory());
    let as_store: Arc<dyn ObjectStore> = faulty.clone();
    let ns = create(as_store.clone()).await;

    let unknown = Predicate::number("missing", Compare::Lt, 1.0);
    let wrong_type = Predicate::is_true("price");
    faulty.reset_ops();
    for bad in [unknown, wrong_type] {
        let dense = QueryOptions::default().filter(Some(bad));
        assert!(ns.query_with(&[0.0; DIM], 3, &dense).await.is_err());
    }
    assert_eq!(faulty.ops(), 0, "no storage call for a refused filter");
}

#[tokio::test]
async fn filtered_query_futures_can_be_spawned() {
    let ns = Arc::new(indexed_namespace(common::memory()).await);
    let dense = QueryOptions::default().filter(Some(cheap()));

    let handle = ns.clone();
    let dense_hits = tokio::spawn(async move {
        handle
            .query_with(&Lcg(1).vector(DIM), 5, &dense)
            .await
            .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(dense_hits.len(), 5);
}

#[tokio::test]
async fn a_selective_filter_on_a_partially_probed_clustered_index_still_finds_every_match() {
    // Most matches sit in partitions the first probe never reads, so stopping at
    // "the index gave less than I asked for" would return a short answer.
    for backend in cas_backends() {
        let ns = create(backend.store.clone()).await;
        let mut rng = Lcg(21);
        let batch: Vec<Document> = (0..400)
            .map(|i| item(&format!("d{i:04}"), f64::from(i % 20), &mut rng))
            .collect();
        ns.upsert(batch.clone()).await.unwrap();
        ns.compact(&ROOMY).await.unwrap();
        ns.index(&IndexOptions::default().min_rows(0).nlist(Some(16)))
            .await
            .unwrap();

        let only_seven = Predicate::number("price", Compare::Eq, 7.0);
        let matching: Vec<Document> = batch
            .iter()
            .filter(|d| d.attrs.as_ref().is_some_and(|a| a["price"] == 7.0))
            .cloned()
            .collect();
        assert_eq!(matching.len(), 20);
        let options = QueryOptions::default()
            .nprobe(Some(1))
            .filter(Some(only_seven));
        for seed in [1, 2, 3, 4] {
            let query = Lcg(seed).vector(DIM);
            let got = ids(&ns.query_with(&query, 15, &options).await.unwrap());
            let want = naive_top_k(Metric::L2, &query, &matching, 15);
            assert_eq!(got, want, "seed {seed}, {}", backend.name);
        }
    }
}

/// An indexed segment, then an unindexed version 2 segment whose attribute
/// JSON is damaged on one row: a filtered query reads that row's attributes
/// and must fail with `CorruptSegment`, never answer.
#[tokio::test]
async fn a_filtered_query_over_damaged_attribute_json_is_corrupt_segment() {
    let store = common::memory();
    let ns = create(store.clone()).await;
    let mut rng = Lcg(3);
    let older: Vec<Document> = (0..20)
        .map(|i| item(&format!("a{i:03}"), f64::from(i % 10), &mut rng))
        .collect();
    ns.upsert(older).await.unwrap();
    ns.compact(&ROOMY).await.unwrap();
    ns.index(&IndexOptions::default().min_rows(0))
        .await
        .unwrap();
    let newer: Vec<Document> = (0..5)
        .map(|i| item(&format!("b{i:03}"), 1.0, &mut rng))
        .collect();
    ns.upsert(newer).await.unwrap();
    ns.compact(&ROOMY).await.unwrap();

    let manifest = itzma_core::manifest::load(&*store, "docs").await.unwrap().0;
    assert_eq!(manifest.indexes.len(), 1);
    let newest = manifest.live_segments.last().unwrap();
    let path = itzma_core::segment::segment_path("docs", newest);
    let mut bytes = store
        .get(&path)
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap()
        .to_vec();
    let at = bytes.windows(8).position(|w| w == b"\"active\"").unwrap();
    bytes[at] = b'{';
    store.put(&path, bytes.into()).await.unwrap();

    let options = QueryOptions::default().filter(Some(cheap()));
    let err = ns.query_with(&[0.0; DIM], 3, &options).await.unwrap_err();
    assert!(matches!(err, ItzmaError::CorruptSegment { .. }), "{err:?}");
}

/// Attributes are parsed for the final hits only. An unindexed segment's
/// closest row has damaged attribute JSON, but the tail holds a closer
/// document: the unfiltered top 1 is the tail's, and the damaged row, never a
/// hit, must not fail the query.
#[tokio::test]
async fn damaged_attribute_json_on_a_row_that_is_not_a_hit_does_not_fail_the_query() {
    let store = common::memory();
    let ns = create(store.clone()).await;
    let point = |x: f32| [x, 0.0, 0.0, 0.0];
    let mut damaged = doc("near", &point(1.0));
    damaged.attrs = Some(json!({"price": 7.25, "active": false}));
    let mut far = doc("far", &point(9.0));
    far.attrs = Some(json!({"price": 1.0, "active": true}));
    ns.upsert(vec![damaged, far]).await.unwrap();
    ns.compact(&ROOMY).await.unwrap();

    let manifest = itzma_core::manifest::load(&*store, "docs").await.unwrap().0;
    let path = itzma_core::segment::segment_path("docs", &manifest.live_segments[0]);
    let mut bytes = store
        .get(&path)
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap()
        .to_vec();
    let at = bytes.windows(4).position(|w| w == b"7.25").unwrap();
    bytes[at] = b'{';
    store.put(&path, bytes.into()).await.unwrap();

    let mut best = doc("best", &point(0.0));
    best.attrs = Some(json!({"price": 2.0, "active": true}));
    ns.upsert(vec![best]).await.unwrap();

    for exact in [false, true] {
        let options = QueryOptions::default().exact(exact);
        let hits = ns.query_with(&point(0.0), 1, &options).await.unwrap();
        assert_eq!(ids(&hits), ["best"], "exact {exact}");
        assert_eq!(hits[0].attrs, Some(json!({"price": 2.0, "active": true})));
    }
    // A query whose hits include the damaged row still fails.
    let err = ns.query(&point(0.0), 2).await.unwrap_err();
    assert!(matches!(err, ItzmaError::CorruptSegment { .. }), "{err:?}");
}

/// `exact` with a filter on a namespace whose every segment is indexed reads
/// no index object and answers like a naive filter-then-rank.
#[tokio::test]
async fn exact_with_a_filter_reads_no_index_object() {
    let counting = Arc::new(CountingStore::default());
    let store: Arc<dyn ObjectStore> = counting.clone();
    let ns = create(store.clone()).await;
    let mut rng = Lcg(11);
    let mut batches = Vec::new();
    for prefix in ["a", "b"] {
        let batch: Vec<Document> = (0..40)
            .map(|i| item(&format!("{prefix}{i:03}"), f64::from(i % 10), &mut rng))
            .collect();
        ns.upsert(batch.clone()).await.unwrap();
        ns.compact(&ROOMY).await.unwrap();
        batches.push(batch);
    }
    ns.index(&IndexOptions::default().min_rows(0))
        .await
        .unwrap();
    let manifest = itzma_core::manifest::load(&*store, "docs").await.unwrap().0;
    assert_eq!(manifest.indexes.len(), manifest.live_segments.len());

    let state = state_of(&batches);
    let matching: Vec<Document> = state.values().filter(|d| is_cheap(d)).cloned().collect();
    let options = QueryOptions::default().exact(true).filter(Some(cheap()));
    for seed in [1, 2, 3] {
        let query = Lcg(seed).vector(DIM);
        counting.reset();
        let got = ids(&ns.query_with(&query, 7, &options).await.unwrap());
        assert_eq!(
            got,
            naive_top_k(Metric::L2, &query, &matching, 7),
            "seed {seed}"
        );
        let paths = counting.paths();
        assert!(
            paths
                .iter()
                .all(|p| !p.contains("index-") && !p.contains("ivf-")),
            "{paths:?}"
        );
    }
}
