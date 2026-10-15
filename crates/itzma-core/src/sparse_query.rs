//! The sparse query engine behind
//! [`Namespace::query_sparse`](crate::Namespace::query_sparse).
//!
//! The namespace is a list of *sources*, oldest to newest: the live segments
//! folded into one in-memory layer and scored directly, and the WAL tail,
//! kept folded by the handle between queries and always its own, newest
//! source. Each source proposes every document that shares a term with the
//! query, and the engine drops any proposal a newer source holds as a document
//! or a tombstone, merges the sources and keeps the closest `top_k`.
//!
//! There is no rerank. A sparse dot product over stored weights is exact, so
//! shadowing can only shrink the set of real matches.

use std::collections::HashMap;
use std::sync::Arc;

use futures::future::try_join_all;
use futures::stream::{self, StreamExt, TryStreamExt};
use object_store::ObjectStore;
use serde::{Deserialize, Serialize};

use crate::budget::ReadBudget;
use crate::cpu::{self, Cost, Lane};
use crate::document::SparseVector;
use crate::error::{ItzmaError, Result};
use crate::fetch::FetchPolicy;
use crate::layer::Layer;
use crate::manifest::Manifest;
use crate::predicate::Predicate;
use crate::search::Hit;
use crate::sparse;

/// Requests in flight at once when a step fans out over segments or ids.
const FETCH_CONCURRENCY: usize = 16;

/// Settings for [`Namespace::query_sparse`](crate::Namespace::query_sparse).
///
/// A sparse dot product is exact, so there is no rerank or exactness knob the
/// way [`QueryOptions`](crate::QueryOptions) needs. The only field is a filter.
/// Build one from `default()` and the setter.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct SparseQueryOptions {
    /// Keep only documents whose attributes match this predicate, as
    /// [`QueryOptions::filter`](crate::QueryOptions::filter) does.
    pub filter: Option<Predicate>,
}

impl SparseQueryOptions {
    /// Set [`filter`](Self::filter).
    #[must_use]
    pub fn filter(mut self, filter: Option<Predicate>) -> Self {
        self.filter = filter;
        self
    }
}

/// Refuse a query that cannot be run, before any storage call.
///
/// # Errors
///
/// [`ItzmaError::InvalidQuery`] for a `top_k` of zero, an empty query vector,
/// or a non-finite weight in it.
pub(crate) fn validate(query: &SparseVector, top_k: usize) -> Result<()> {
    if top_k == 0 {
        return Err(ItzmaError::InvalidQuery("top_k must be at least 1"));
    }
    if query.is_empty() {
        return Err(ItzmaError::InvalidQuery(
            "the sparse query vector must not be empty",
        ));
    }
    if query.iter().any(|(_, weight)| !weight.is_finite()) {
        return Err(ItzmaError::InvalidQuery(
            "the sparse query vector has a non-finite weight",
        ));
    }
    Ok(())
}

/// The query's weight for each distinct term. A term repeated in the query
/// counts as the sum of its weights, which is what the dot product with the
/// summed vector is.
fn weights_of(query: &SparseVector) -> HashMap<u32, f32> {
    let mut weights: HashMap<u32, f32> = HashMap::with_capacity(query.len());
    for &(term, weight) in query {
        *weights.entry(term).or_insert(0.0) += weight;
    }
    weights
}

/// One sparse query against one manifest.
pub(crate) struct Query<'a> {
    pub(crate) store: &'a dyn ObjectStore,
    pub(crate) namespace: &'a str,
    pub(crate) dim: usize,
    pub(crate) query: &'a SparseVector,
    pub(crate) top_k: usize,
    pub(crate) filter: Option<&'a Predicate>,
    pub(crate) budget: &'a ReadBudget,
    pub(crate) policy: &'a FetchPolicy,
}

/// What the manifest's live segments turn into.
enum Step<'a> {
    /// The live segments, oldest first.
    Flat(&'a [String]),
}

/// A place a document can come from, newest last.
enum Source {
    /// The live segments overlaid into one layer, or the WAL tail;
    /// either way its documents carry their sparse vectors.
    Flat(Arc<Layer>),
}

/// A hit's attributes, already in memory.
enum Attrs {
    Have(Option<serde_json::Value>),
}

/// A live candidate with its score, negated so lower is closer like every
/// other distance in the crate.
struct Scored {
    distance: f32,
    id: String,
    attrs: Attrs,
}

/// The live segments as one step: overlaying them gives their net effect,
/// and nothing between them needs a separate shadow check.
fn plan<'a>(manifest: &'a Manifest) -> Vec<Step<'a>> {
    let live = &manifest.live_segments;
    if live.is_empty() {
        Vec::new()
    } else {
        vec![Step::Flat(live)]
    }
}

/// Run the query over `manifest`.
///
/// # Errors
///
/// The errors of reading segments, sparse sidecars and the tail (a missing
/// segment or a fenced WAL object is what the caller reloads the manifest
/// for), and [`ItzmaError::ReadBudgetExceeded`].
pub(crate) async fn run<'a>(
    q: &Query<'a>,
    manifest: &'a Manifest,
    tail: impl Future<Output = Result<Arc<Layer>>>,
) -> Result<Vec<Hit>> {
    let steps = plan(manifest);
    // The tail is read alongside the segments and is its own, newest source,
    // shared through the handle's view, as in the dense engine.
    let (mut sources, tail) = futures::try_join!(open_sources(q, steps), tail)?;
    if !tail.is_empty() {
        sources.push(Source::Flat(tail));
    }

    let weights = weights_of(q.query);
    let sources = &sources;
    let weights = &weights;
    let mut live: Vec<Scored> =
        try_join_all((0..sources.len()).map(|at| search_source(q, sources, at, weights)))
            .await?
            .into_iter()
            .flatten()
            .collect();
    live.sort_by(|a, b| {
        a.distance
            .total_cmp(&b.distance)
            .then_with(|| a.id.cmp(&b.id))
    });
    live.truncate(q.top_k);
    finish(live)
}

async fn open_sources<'a>(q: &Query<'a>, steps: Vec<Step<'a>>) -> Result<Vec<Source>> {
    // Indexed by position, not mapped over the owned `Step` values: a closure
    // whose parameter itself borrows `'a` is inferred as higher-ranked, which
    // then fails to prove the stream `Send` (see `query::open_sources`).
    stream::iter(0..steps.len())
        .map(|at| open_source(q, &steps[at]))
        .buffered(FETCH_CONCURRENCY)
        .try_collect()
        .await
}

async fn open_source<'a>(q: &Query<'a>, step: &Step<'a>) -> Result<Source> {
    match *step {
        Step::Flat(ids) => Ok(Source::Flat(Arc::new(
            sparse::read_all_with_sparse(
                q.store,
                q.namespace,
                ids,
                q.dim,
                q.budget,
                q.policy,
                Lane::Query,
            )
            .await?,
        ))),
    }
}

async fn search_source(
    q: &Query<'_>,
    sources: &[Source],
    at: usize,
    weights: &HashMap<u32, f32>,
) -> Result<Vec<Scored>> {
    match &sources[at] {
        Source::Flat(layer) => evaluate_flat(q, sources, at, layer, weights).await,
    }
}

/// The negated score of every document of `layer` against the query's
/// `weights`, in `layer.docs()` order: `None` where `filter` drops the
/// document or it shares no term with the query. Chunked so a pool thread
/// splits it across cores, with the same result either way.
fn flat_scores(
    layer: &Layer,
    weights: &HashMap<u32, f32>,
    filter: Option<&Predicate>,
) -> Vec<Option<f32>> {
    let docs: Vec<&crate::document::Document> = layer.docs().collect();
    let chunks = cpu::chunked(&docs, cpu::FLAT_CHUNK, |chunk| {
        chunk
            .iter()
            .map(|doc| {
                if !filter.is_none_or(|p| p.matches(doc.attrs.as_ref())) {
                    return None;
                }
                let terms = doc.sparse.as_ref()?;
                let mut shared = false;
                let mut score = 0.0f32;
                for (term, doc_weight) in terms {
                    if let Some(weight) = weights.get(term) {
                        shared = true;
                        score += weight * doc_weight;
                    }
                }
                shared.then_some(-score)
            })
            .collect::<Vec<_>>()
    });
    let mut out = Vec::with_capacity(docs.len());
    for chunk in chunks {
        out.extend(chunk);
    }
    out
}

/// The same for a layer held in memory: score each live document against its
/// own `sparse` field, which `sparse::read_all_with_sparse` (or the WAL)
/// already filled in. A document sharing no term with the query is not a
/// candidate.
async fn evaluate_flat(
    q: &Query<'_>,
    sources: &[Source],
    at: usize,
    layer: &Arc<Layer>,
    weights: &HashMap<u32, f32>,
) -> Result<Vec<Scored>> {
    // One lookup per stored term: the work is the layer's total term count.
    let terms: usize = layer
        .docs()
        .map(|doc| doc.sparse.as_ref().map_or(0, Vec::len))
        .sum();
    let (owned, weights, filter) = (layer.clone(), weights.clone(), q.filter.cloned());
    let scores = cpu::run(
        Lane::Query,
        Cost::Distances {
            rows: terms,
            dim: 1,
        },
        move || flat_scores(&owned, &weights, filter.as_ref()),
    )
    .await;
    // `Layer::docs` keeps its order while the layer is unchanged (its
    // contract), so this second pass lines up with the one on the pool.
    debug_assert_eq!(scores.len(), layer.doc_count());
    let scored: Vec<(f32, &crate::document::Document)> = layer
        .docs()
        .zip(scores)
        .filter_map(|(doc, score)| score.map(|s| (s, doc)))
        .collect();
    let ids: Vec<String> = scored.iter().map(|(_, doc)| doc.id.clone()).collect();
    let live = unshadowed(sources, at, &ids).await?;
    Ok(scored
        .into_iter()
        .zip(live)
        .filter(|(_, live)| *live)
        .map(|((distance, doc), _)| Scored {
            distance,
            id: doc.id.clone(),
            attrs: Attrs::Have(doc.attrs.clone()),
        })
        .collect())
}

/// For each of `ids`, proposed by source `at`: whether it is still the live
/// version, that is, no newer source holds the id as a document or a
/// tombstone.
async fn unshadowed(sources: &[Source], at: usize, ids: &[String]) -> Result<Vec<bool>> {
    // Owned ids: a closure over `&String` is inferred as higher-ranked, which
    // makes the stream not provably `Send` (see `segment::read_all`).
    stream::iter(ids.iter().cloned())
        .map(|id| async move { is_live(sources, at, &id).await })
        .buffered(FETCH_CONCURRENCY)
        .try_collect()
        .await
}

async fn is_live(sources: &[Source], at: usize, id: &str) -> Result<bool> {
    let newer = &sources[at + 1..];
    // Layers in memory first: they cost no request.
    if newer
        .iter()
        .any(|source| matches!(source, Source::Flat(layer) if layer.holds(id)))
    {
        return Ok(false);
    }
    Ok(true)
}

/// Build the hits.
fn finish(live: Vec<Scored>) -> Result<Vec<Hit>> {
    Ok(live
        .into_iter()
        .map(|hit| Hit {
            attrs: match hit.attrs {
                Attrs::Have(value) => value,
            },
            id: hit.id,
            distance: hit.distance,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{Document, WalOp};

    #[tokio::test]
    async fn sparse_flat_scores_on_the_pool_equal_the_inline_ones() {
        // 20,000 documents of 100 terms: past the inline threshold.
        let mut layer = Layer::new();
        for i in 0..20_000u32 {
            layer.apply(WalOp::Upsert(Document {
                id: format!("d{i:06}"),
                vector: vec![0.0],
                attrs: Some(serde_json::json!({ "active": i % 3 == 0 })),
                sparse: Some(
                    (0..100)
                        .map(|t| ((i + t * 7) % 500, 1.0 + (t % 5) as f32))
                        .collect(),
                ),
            }));
        }
        let layer = Arc::new(layer);
        let weights: HashMap<u32, f32> = [(3, 1.0), (10, 0.5), (499, 2.0)].into();
        for filter in [None, Some(Predicate::is_true("active"))] {
            let inline = flat_scores(&layer, &weights, filter.as_ref());
            assert_eq!(inline.len(), 20_000);
            assert!(inline.iter().any(Option::is_some) && inline.iter().any(Option::is_none));
            let (l, w, f) = (layer.clone(), weights.clone(), filter.clone());
            let pooled = cpu::run(
                Lane::Query,
                Cost::Distances {
                    rows: 2_000_000,
                    dim: 1,
                },
                move || flat_scores(&l, &w, f.as_ref()),
            )
            .await;
            assert_eq!(pooled, inline);
        }
    }

    #[test]
    fn validate_refuses_what_cannot_run() {
        let ok: SparseVector = vec![(1, 1.0)];
        assert!(validate(&ok, 1).is_ok());
        for (query, top_k) in [
            (ok.clone(), 0),
            (Vec::new(), 1),
            (vec![(1, f32::NAN)], 1),
            (vec![(1, f32::INFINITY)], 1),
        ] {
            assert!(
                matches!(validate(&query, top_k), Err(ItzmaError::InvalidQuery(_))),
                "{query:?} / {top_k}"
            );
        }
    }

    #[test]
    fn a_repeated_query_term_counts_as_the_sum_of_its_weights() {
        let weights = weights_of(&vec![(3, 1.0), (5, 2.0), (3, 0.5)]);
        assert_eq!(weights.len(), 2);
        assert_eq!(weights[&3], 1.5);
        assert_eq!(weights[&5], 2.0);
    }
}
