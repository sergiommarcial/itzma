//! `Namespace::query_sparse`: equality with a naive scan at every
//! stage a namespace goes through (tail only, folded, merged, mixed), and
//! shadowing by overwrites and deletes.

mod common;

use common::counting::CountingStore;

use std::collections::BTreeMap;
use std::sync::Arc;

use common::{Lcg, cas_backends, doc, schema};
use itzma_core::{
    CompactionOptions, Document, ItzmaError, Metric, Namespace, SparseQueryOptions, SparseVector,
};
use object_store::ObjectStore;

const ROOMY: CompactionOptions = CompactionOptions {
    max_segments: 8,
    max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
};
const TIGHT: CompactionOptions = CompactionOptions {
    max_segments: 2,
    max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
};

fn sdoc(id: &str, terms: &[(u32, f32)]) -> Document {
    Document {
        id: id.into(),
        vector: vec![1.0],
        attrs: None,
        sparse: Some(terms.to_vec()),
    }
}

async fn create(store: Arc<dyn ObjectStore>) -> Namespace {
    let with_title = schema(1, Metric::L2)
        .with_attrs(itzma_core::AttrSchema::new().field("title", itzma_core::AttrType::Text));
    Namespace::create(store, "docs", with_title)
        .await
        .expect("create")
}

async fn ids(ns: &Namespace, query: &SparseVector, k: usize) -> Vec<String> {
    ns.query_sparse(query, k, &SparseQueryOptions::default())
        .await
        .expect("query_sparse")
        .into_iter()
        .map(|hit| hit.id)
        .collect()
}

/// What a naive scan of `model` returns: every document sharing a term, best
/// score first, ties by id.
fn naive(
    model: &BTreeMap<String, Option<SparseVector>>,
    query: &SparseVector,
    k: usize,
) -> Vec<String> {
    let mut scored: Vec<(f32, &str)> = model
        .iter()
        .filter_map(|(id, terms)| {
            let terms = terms.as_ref()?;
            let mut shared = false;
            let mut score = 0.0;
            for (qt, qw) in query {
                for (t, w) in terms {
                    if t == qt {
                        shared = true;
                        score += qw * w;
                    }
                }
            }
            shared.then_some((-score, id.as_str()))
        })
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(b.1)));
    scored
        .into_iter()
        .take(k)
        .map(|(_, id)| id.to_string())
        .collect()
}

#[tokio::test]
async fn the_same_answer_in_the_tail_and_folded() {
    for backend in cas_backends() {
        let ns = create(backend.store.clone()).await;
        ns.upsert(vec![
            sdoc("a", &[(3, 1.0), (9, 0.5)]),
            sdoc("b", &[(3, 0.5)]),
            sdoc("c", &[(9, 2.0)]),
            doc("d", &[1.0]), // no sparse vector at all
        ])
        .await
        .expect("upsert");

        let query: SparseVector = vec![(3, 1.0), (9, 1.0)];
        let want = vec!["c".to_string(), "a".to_string(), "b".to_string()];

        assert_eq!(ids(&ns, &query, 10).await, want, "{} tail", backend.name);
        ns.compact(&ROOMY).await.expect("compact");
        assert_eq!(ids(&ns, &query, 10).await, want, "{} folded", backend.name);
    }
}

#[tokio::test]
async fn hits_carry_the_negated_score_and_the_attributes() {
    let ns = create(common::memory()).await;
    let mut d = sdoc("a", &[(1, 2.0), (2, 1.0)]);
    d.attrs = Some(serde_json::json!({"title": "cats"}));
    ns.upsert(vec![d]).await.expect("upsert");
    ns.compact(&ROOMY).await.expect("compact");

    let hits = ns
        .query_sparse(&vec![(1, 1.5), (2, 1.0)], 1, &SparseQueryOptions::default())
        .await
        .expect("query_sparse");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].distance, -4.0);
    assert_eq!(hits[0].attrs, Some(serde_json::json!({"title": "cats"})));
}

#[tokio::test]
async fn a_repeated_query_term_counts_as_the_sum_of_its_weights() {
    let ns = create(common::memory()).await;
    ns.upsert(vec![sdoc("a", &[(1, 2.0)])])
        .await
        .expect("upsert");
    let hits = ns
        .query_sparse(&vec![(1, 1.0), (1, 0.5)], 1, &SparseQueryOptions::default())
        .await
        .expect("query_sparse");
    assert_eq!(hits[0].distance, -3.0);
}

#[tokio::test]
async fn top_k_truncates_and_ties_break_by_id() {
    let ns = create(common::memory()).await;
    ns.upsert(vec![
        sdoc("c", &[(1, 1.0)]),
        sdoc("a", &[(1, 1.0)]),
        sdoc("b", &[(1, 1.0)]),
    ])
    .await
    .expect("upsert");
    assert_eq!(ids(&ns, &vec![(1, 1.0)], 2).await, ["a", "b"]);
}

#[tokio::test]
async fn an_overwrite_or_a_delete_shadows_an_older_match() {
    for backend in cas_backends() {
        let ns = create(backend.store.clone()).await;
        ns.upsert(vec![sdoc("a", &[(3, 1.0)]), sdoc("b", &[(3, 0.5)])])
            .await
            .expect("upsert");
        ns.compact(&ROOMY).await.expect("compact");
        let query: SparseVector = vec![(3, 1.0)];
        assert_eq!(ids(&ns, &query, 10).await, ["a", "b"]);

        // Overwritten by a document with no sparse vector: it must vanish.
        ns.upsert(vec![doc("a", &[2.0])]).await.expect("overwrite");
        assert_eq!(ids(&ns, &query, 10).await, ["b"], "{} tail", backend.name);

        // Deleted in the tail.
        ns.delete(vec!["b".into()]).await.expect("delete");
        assert!(ids(&ns, &query, 10).await.is_empty(), "{}", backend.name);

        // The same once those writes are folded into a newer, unindexed segment.
        ns.compact(&ROOMY).await.expect("fold");
        assert!(
            ids(&ns, &query, 10).await.is_empty(),
            "{} folded",
            backend.name
        );

        // A new sparse vector for "a" replaces the old one, with the new weight.
        ns.upsert(vec![sdoc("a", &[(3, 4.0)])])
            .await
            .expect("upsert");
        let hits = ns
            .query_sparse(&query, 10, &SparseQueryOptions::default())
            .await
            .expect("query_sparse");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].distance, -4.0, "{}", backend.name);
    }
}

#[tokio::test]
async fn a_mix_of_segments_and_tail_sources_agrees_with_a_naive_scan() {
    let ns = create(common::memory()).await;
    let mut model: BTreeMap<String, Option<SparseVector>> = BTreeMap::new();
    let mut put = |id: &str, terms: &[(u32, f32)]| {
        model.insert(id.to_string(), Some(terms.to_vec()));
        sdoc(id, terms)
    };

    ns.upsert(vec![put("a", &[(1, 1.0)]), put("b", &[(1, 2.0), (2, 1.0)])])
        .await
        .expect("upsert");
    ns.compact(&ROOMY).await.expect("first fold");

    ns.upsert(vec![put("c", &[(2, 3.0)]), put("a", &[(2, 0.5)])])
        .await
        .expect("upsert");
    ns.compact(&ROOMY)
        .await
        .expect("second fold, left unindexed");

    ns.upsert(vec![put("d", &[(1, 0.25), (2, 0.25)])])
        .await
        .expect("tail");

    let query: SparseVector = vec![(1, 1.0), (2, 1.0)];
    assert_eq!(ids(&ns, &query, 10).await, naive(&model, &query, 10));
    assert_eq!(ids(&ns, &query, 2).await, naive(&model, &query, 2));
}

/// With `raise_to_3`, the namespace moves to segment format 3
/// partway, so later folds and merges write version 3 beside version 2.
async fn random_history(raise_to_3: bool) {
    for backend in cas_backends() {
        let ns = create(backend.store.clone()).await;
        let mut rng = Lcg(7);
        let mut model: BTreeMap<String, Option<SparseVector>> = BTreeMap::new();

        for step in 0..60 {
            if raise_to_3 && step == 20 {
                ns.upgrade_segment_format(3).await.expect("raise");
            }
            match rng.below(6) {
                0..=2 => {
                    let docs: Vec<Document> = (0..1 + rng.below(3))
                        .map(|_| {
                            let id = format!("d{:02}", rng.below(12));
                            if rng.below(4) == 0 {
                                doc(&id, &[1.0])
                            } else {
                                let mut terms: SparseVector = Vec::new();
                                for _ in 0..1 + rng.below(4) {
                                    let term = rng.below(10) as u32;
                                    if !terms.iter().any(|(t, _)| *t == term) {
                                        // Quarter steps, so sums are exact in f32 and
                                        // the order they are added in cannot matter.
                                        terms.push((term, (1 + rng.below(8)) as f32 * 0.25));
                                    }
                                }
                                sdoc(&id, &terms)
                            }
                        })
                        .collect();
                    for d in &docs {
                        model.insert(d.id.clone(), d.sparse.clone());
                    }
                    ns.upsert(docs).await.expect("upsert");
                }
                3 => {
                    let id = format!("d{:02}", rng.below(12));
                    model.remove(&id);
                    ns.delete(vec![id]).await.expect("delete");
                }
                4 => {
                    let opts = if rng.below(2) == 0 { ROOMY } else { TIGHT };
                    ns.compact(&opts).await.expect("compact");
                }
                _ => {}
            }

            let query: SparseVector = (0..1 + rng.below(3))
                .map(|_| (rng.below(10) as u32, (1 + rng.below(4)) as f32 * 0.5))
                .collect();
            // Duplicate query terms sum, so merge them the same way in the model.
            let mut merged: BTreeMap<u32, f32> = BTreeMap::new();
            for (t, w) in &query {
                *merged.entry(*t).or_insert(0.0) += w;
            }
            let merged: SparseVector = merged.into_iter().collect();

            for k in [1, 3, 100] {
                assert_eq!(
                    ids(&ns, &query, k).await,
                    naive(&model, &merged, k),
                    "{}: step {step}, k {k}, query {query:?}\nmodel {model:?}\nhits {:?}",
                    backend.name,
                    ns.query_sparse(&query, 100, &SparseQueryOptions::default())
                        .await
                        .unwrap()
                        .iter()
                        .map(|h| (h.id.clone(), h.distance))
                        .collect::<Vec<_>>()
                );
            }
        }
        if raise_to_3 {
            let versions = common::live_segment_versions(&*backend.store).await;
            assert!(versions.contains(&3), "{versions:?}");
        }
    }
}

#[tokio::test]
async fn a_random_history_agrees_with_a_naive_scan_at_every_step() {
    random_history(false).await;
}

#[tokio::test]
async fn a_random_history_across_versions_2_and_3_agrees_with_a_naive_scan() {
    random_history(true).await;
}

#[tokio::test]
async fn an_invalid_query_is_refused_before_any_storage_call() {
    let ns = create(common::memory()).await;
    for (query, k) in [
        (vec![(1, 1.0)], 0),
        (Vec::new(), 1),
        (vec![(1, f32::NAN)], 1),
    ] {
        let err = ns
            .query_sparse(&query, k, &SparseQueryOptions::default())
            .await
            .unwrap_err();
        assert!(matches!(err, ItzmaError::InvalidQuery(_)), "{err:?}");
    }
}

#[tokio::test]
async fn an_upsert_with_a_bad_sparse_vector_is_refused_and_writes_nothing() {
    let store = common::memory();
    let ns = create(store.clone()).await;
    let before = common::snapshot(&*store).await;
    let too_many: SparseVector = (0..=itzma_core::document::MAX_SPARSE_TERMS as u32)
        .map(|t| (t, 1.0))
        .collect();
    for terms in [vec![(1, 1.0), (1, 2.0)], vec![(1, f32::NAN)], too_many] {
        let err = ns.upsert(vec![sdoc("a", &terms)]).await.unwrap_err();
        assert!(matches!(err, ItzmaError::InvalidVector(_)), "{err:?}");
    }
    assert_eq!(common::snapshot(&*store).await, before);
}

/// A second sparse query over an unchanged tail lists the WAL
/// once and fetches nothing but the manifest.
#[tokio::test]
async fn an_unchanged_tail_is_listed_but_not_fetched_again() {
    let counting = Arc::new(CountingStore::default());
    let ns = create(counting.clone()).await;
    for i in 0..3u32 {
        ns.upsert(vec![sdoc(&format!("d{i}"), &[(i, 1.0), (7, 0.5)])])
            .await
            .unwrap();
    }
    let query: SparseVector = vec![(7, 1.0)];
    let first = ids(&ns, &query, 10).await;
    assert_eq!(first.len(), 3);
    counting.reset();
    assert_eq!(ids(&ns, &query, 10).await, first);
    assert_eq!(counting.lists(), 1);
    let paths = counting.paths();
    assert_eq!(paths.len(), 1, "{paths:?}");
    assert!(paths[0].ends_with("manifest/current.json"), "{paths:?}");
}
