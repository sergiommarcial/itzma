//! The sparse query engine behind
//! [`Namespace::query_sparse`](crate::Namespace::query_sparse).
//!
//! The namespace is a list of *sources*, oldest to newest, planned the way the
//! dense engine plans them (`query.rs`) but grouped by the manifest's
//! `sparse_indexes` instead of `indexes`. A live segment with a sparse index is
//! an indexed source: its term directory is fetched once, and only the
//! postings of the query's own terms after that. Every run of segments without
//! one is folded into one in-memory layer and scored directly, and so is the
//! WAL tail, kept folded by the handle between queries and always
//! its own, newest source. Each source proposes every document that shares a term with the
//! query, and the engine drops any proposal a newer source holds as a document
//! or a tombstone, merges the sources and keeps the closest `top_k`.
//!
//! An indexed source reuses [`RangedSegment`], built for `data.bin`, for the
//! three things it needs a segment for: row to id, the shadow check, and the
//! attributes of the final hits. A sparse index only adds the term lookup.
//!
//! There is no rerank and no re-reading of postings. A sparse dot product over
//! stored weights is exact, and the full postings scan of every query term
//! returns every row that shares one, so shadowing can only shrink the set of
//! real matches. The one loop is local: an indexed source ranks its rows by
//! score and resolves ids and the shadow check down that ranking only until
//! `top_k` live rows are in hand, so an unshadowed query reads about `top_k`
//! ids, not one per matching row.

use std::collections::HashMap;
use std::sync::Arc;

use futures::future::try_join_all;
use futures::stream::{self, StreamExt, TryStreamExt};
use object_store::ObjectStore;
use object_store::path::Path;
use serde::{Deserialize, Serialize};

use crate::budget::ReadBudget;
use crate::cpu::{self, Cost, Lane};
use crate::decoded::{DecodedRef, What};
use crate::document::SparseVector;
use crate::error::{ItzmaError, Result};
use crate::fetch::FetchPolicy;
use crate::layer::Layer;
use crate::manifest::Manifest;
use crate::predicate::Predicate;
use crate::search::Hit;
use crate::segment::{Held, RangedSegment};
use crate::sparse;
use crate::sparse_index::{self, TermDirectory};

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
    /// Parsed structures kept between queries, if the handle has a
    /// decoded cache.
    pub(crate) decoded: Option<&'a DecodedRef>,
}

/// What the manifest's live segments turn into.
enum Step<'a> {
    /// A segment with a sparse index.
    Indexed { id: &'a str, generation: u32 },
    /// A run of consecutive segments without one.
    Flat(&'a [String]),
}

/// A place a document can come from, newest last.
enum Source<'a> {
    Indexed(Box<Indexed<'a>>),
    /// A run of unindexed segments overlaid into one layer, or the WAL tail;
    /// either way its documents carry their sparse vectors.
    Flat(Arc<Layer>),
}

struct Indexed<'a> {
    segment: RangedSegment<'a>,
    index_path: Path,
    dir: Arc<TermDirectory>,
}

/// Where a hit's attributes are: already in memory, or in the data object.
enum Attrs {
    Have(Option<serde_json::Value>),
    Row(u32),
}

/// A live candidate with its score, negated so lower is closer like every
/// other distance in the crate.
struct Scored {
    distance: f32,
    id: String,
    source: usize,
    attrs: Attrs,
}

/// Group the live segments into steps, oldest first. Consecutive segments
/// without a sparse index stay together, since overlaying them gives their
/// net effect and nothing between them needs a separate shadow check.
fn plan<'a>(manifest: &'a Manifest) -> Vec<Step<'a>> {
    let live = &manifest.live_segments;
    let mut steps = Vec::new();
    let mut run_start = 0;
    for (at, id) in live.iter().enumerate() {
        if let Some(entry) = manifest.sparse_indexes.get(id) {
            if run_start < at {
                steps.push(Step::Flat(&live[run_start..at]));
            }
            steps.push(Step::Indexed {
                id,
                generation: entry.generation,
            });
            run_start = at + 1;
        }
    }
    if run_start < live.len() {
        steps.push(Step::Flat(&live[run_start..]));
    }
    steps
}

/// Run the query over `manifest`.
///
/// # Errors
///
/// The errors of reading segments, sparse sidecars, sparse indexes and the
/// tail (a missing segment or index, or a fenced WAL object, is what the
/// caller reloads the manifest for), [`ItzmaError::CorruptSparseIndex`] for a
/// damaged index or a row it made up, and [`ItzmaError::ReadBudgetExceeded`].
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
    finish(sources, live).await
}

async fn open_sources<'a>(q: &Query<'a>, steps: Vec<Step<'a>>) -> Result<Vec<Source<'a>>> {
    // Indexed by position, not mapped over the owned `Step` values: a closure
    // whose parameter itself borrows `'a` is inferred as higher-ranked, which
    // then fails to prove the stream `Send` (see `query::open_sources`).
    stream::iter(0..steps.len())
        .map(|at| open_source(q, &steps[at]))
        .buffered(FETCH_CONCURRENCY)
        .try_collect()
        .await
}

async fn open_source<'a>(q: &Query<'a>, step: &Step<'a>) -> Result<Source<'a>> {
    match *step {
        Step::Indexed { id, generation } => {
            let segment = RangedSegment::open(
                q.store,
                q.namespace,
                id,
                q.dim,
                q.budget,
                q.policy,
                q.decoded,
            )
            .await?;
            let index_path = sparse_index::sparse_index_path(q.namespace, id, generation);
            let rows = segment.doc_count();
            let load = || async {
                let (_prefix, dir) =
                    sparse_index::open(q.store, q.budget, &index_path, rows).await?;
                let weight = dir.weight();
                Ok((dir, weight))
            };
            let dir = match q.decoded {
                Some(decoded) => {
                    decoded
                        .get_or_load(&index_path, What::SparseDirectory, load)
                        .await?
                }
                None => Arc::new(load().await?.0),
            };
            Ok(Source::Indexed(Box::new(Indexed {
                segment,
                index_path,
                dir,
            })))
        }
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
    sources: &[Source<'_>],
    at: usize,
    weights: &HashMap<u32, f32>,
) -> Result<Vec<Scored>> {
    match &sources[at] {
        Source::Indexed(source) => evaluate_indexed(q, sources, at, source, weights).await,
        Source::Flat(layer) => evaluate_flat(q, sources, at, layer, weights).await,
    }
}

/// Score every row the postings of the query's terms name, then keep the best
/// live ones: at least `top_k`, plus any that tie with the last. One ranged read
/// per matched term, and ids are resolved only as far down the ranking as needed.
async fn evaluate_indexed(
    q: &Query<'_>,
    sources: &[Source<'_>],
    at: usize,
    source: &Indexed<'_>,
    weights: &HashMap<u32, f32>,
) -> Result<Vec<Scored>> {
    let mut postings: Vec<(f32, Vec<(u32, f32)>)> = Vec::new();
    for (&term, &weight) in weights {
        let Some(entry) = source.dir.find(term) else {
            continue;
        };
        let found =
            sparse_index::postings_of(q.store, q.budget, &source.index_path, &source.dir, entry)
                .await?;
        postings.push((weight, found));
    }
    let rows_in_segment = source.segment.doc_count();
    let path = source.index_path.to_string();
    let cost = Cost::Distances {
        rows: postings.iter().map(|(_, p)| p.len()).sum(),
        dim: 1,
    };
    let ranked = cpu::run(Lane::Query, cost, move || {
        score_postings(postings, rows_in_segment, &path)
    })
    .await?;

    // Resolve ids and check shadowing down the ranking, in growing chunks, and
    // stop once `top_k` live rows that match are in hand and the next row
    // scores strictly less than the last of them. A row tied with it can still
    // win on id, so ties are all kept. With no newer source and no filter
    // nothing is dropped, and this reads the ids of about `top_k` rows, not of
    // every row a term names. A filter can drop most rows, so it reads further
    // down the ranking.
    let mut kept: Vec<Scored> = Vec::new();
    let mut next = 0;
    let mut chunk = q.top_k;
    while next < ranked.len() {
        if kept.len() >= q.top_k && ranked[next].1 < -kept[q.top_k - 1].distance {
            break;
        }
        let end = (next + chunk).min(ranked.len());
        let rows: Vec<u32> = ranked[next..end].iter().map(|(row, _)| *row).collect();
        let ids = source.segment.ids_of(&rows).await?;
        let live = unshadowed(sources, at, &ids).await?;
        let survivors: Vec<(u32, f32, String)> = ranked[next..end]
            .iter()
            .zip(ids)
            .zip(live)
            .filter_map(|(((row, score), id), live)| live.then_some((*row, *score, id)))
            .collect();
        // The attributes are fetched once per chunk for the live rows only, and
        // stay on the hit so `finish` does not fetch them again.
        let fetched = match q.filter {
            None => None,
            Some(_) => {
                let rows: Vec<u32> = survivors.iter().map(|(row, _, _)| *row).collect();
                Some(source.segment.attrs_of(&rows).await?)
            }
        };
        for (n, (row, score, id)) in survivors.into_iter().enumerate() {
            let attrs = match (&fetched, q.filter) {
                (Some(fetched), Some(predicate)) => {
                    let value = fetched[n].clone();
                    if !predicate.matches(value.as_ref()) {
                        continue;
                    }
                    Attrs::Have(value)
                }
                _ => Attrs::Row(row),
            };
            kept.push(Scored {
                distance: -score,
                id,
                source: at,
                attrs,
            });
        }
        next = end;
        chunk = chunk.saturating_mul(2);
    }
    Ok(kept)
}

/// Sum `weight * doc_weight` per row over every term's postings, in the
/// order given, and rank the rows: highest score first, row order among
/// equals so a run is repeatable.
///
/// # Errors
///
/// [`ItzmaError::CorruptSparseIndex`] for a row at or past `rows_in_segment`.
fn score_postings(
    postings: Vec<(f32, Vec<(u32, f32)>)>,
    rows_in_segment: u64,
    path: &str,
) -> Result<Vec<(u32, f32)>> {
    let mut scores: HashMap<u32, f32> = HashMap::new();
    for (weight, found) in postings {
        for (row, doc_weight) in found {
            if u64::from(row) >= rows_in_segment {
                return Err(ItzmaError::CorruptSparseIndex {
                    path: path.to_string(),
                    reason: format!(
                        "the index named row {row}, the segment has {rows_in_segment} rows"
                    ),
                });
            }
            *scores.entry(row).or_insert(0.0) += weight * doc_weight;
        }
    }
    let mut ranked: Vec<(u32, f32)> = scores.into_iter().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    Ok(ranked)
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
    sources: &[Source<'_>],
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
            source: at,
            attrs: Attrs::Have(doc.attrs.clone()),
        })
        .collect())
}

/// For each of `ids`, proposed by source `at`: whether it is still the live
/// version, that is, no newer source holds the id as a document or a
/// tombstone.
async fn unshadowed(sources: &[Source<'_>], at: usize, ids: &[String]) -> Result<Vec<bool>> {
    // Owned ids: a closure over `&String` is inferred as higher-ranked, which
    // makes the stream not provably `Send` (see `segment::read_all`).
    stream::iter(ids.iter().cloned())
        .map(|id| async move { is_live(sources, at, &id).await })
        .buffered(FETCH_CONCURRENCY)
        .try_collect()
        .await
}

async fn is_live(sources: &[Source<'_>], at: usize, id: &str) -> Result<bool> {
    let newer = &sources[at + 1..];
    // Layers in memory first: they cost no request.
    if newer
        .iter()
        .any(|source| matches!(source, Source::Flat(layer) if layer.holds(id)))
    {
        return Ok(false);
    }
    for source in newer {
        if let Source::Indexed(newer) = source {
            if newer.segment.holds(id).await? != Held::No {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// Fetch the attributes the final hits still lack, one request per block, and
/// build the hits.
async fn finish(sources: &[Source<'_>], live: Vec<Scored>) -> Result<Vec<Hit>> {
    let mut wanted: HashMap<usize, Vec<u32>> = HashMap::new();
    for hit in &live {
        if let Attrs::Row(row) = hit.attrs {
            wanted.entry(hit.source).or_default().push(row);
        }
    }
    let fetched = try_join_all(
        wanted
            .into_iter()
            .map(|(at, rows)| fetch_attrs(sources, at, rows)),
    )
    .await?;
    let mut attrs: HashMap<(usize, u32), Option<serde_json::Value>> = HashMap::new();
    for (at, rows, values) in fetched {
        attrs.extend(rows.into_iter().zip(values).map(|(row, v)| ((at, row), v)));
    }
    Ok(live
        .into_iter()
        .map(|hit| Hit {
            attrs: match hit.attrs {
                Attrs::Have(value) => value,
                Attrs::Row(row) => attrs.remove(&(hit.source, row)).flatten(),
            },
            id: hit.id,
            distance: hit.distance,
        })
        .collect())
}

async fn fetch_attrs(
    sources: &[Source<'_>],
    at: usize,
    rows: Vec<u32>,
) -> Result<(usize, Vec<u32>, Vec<Option<serde_json::Value>>)> {
    let Source::Indexed(source) = &sources[at] else {
        return Ok((at, rows, Vec::new()));
    };
    let values = source.segment.attrs_of(&rows).await?;
    Ok((at, rows, values))
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

    #[tokio::test]
    async fn scored_postings_on_the_pool_equal_the_inline_ones() {
        let postings: Vec<(f32, Vec<(u32, f32)>)> = (0..30u32)
            .map(|t| {
                (
                    1.0 + t as f32 / 10.0,
                    (0..50_000)
                        .step_by(1 + t as usize)
                        .map(|r| (r, 0.5))
                        .collect(),
                )
            })
            .collect();
        let inline = score_postings(postings.clone(), 50_000, "p").expect("score");
        let p = postings.clone();
        let pooled = cpu::run(
            Lane::Query,
            Cost::Distances {
                rows: 2_000_000,
                dim: 1,
            },
            move || score_postings(p, 50_000, "p"),
        )
        .await
        .expect("score");
        assert_eq!(pooled, inline);
        assert!(
            inline
                .windows(2)
                .all(|w| w[0].1 > w[1].1 || (w[0].1 == w[1].1 && w[0].0 < w[1].0))
        );
        let err = score_postings(postings, 10, "p").expect_err("row past the segment");
        assert!(matches!(err, ItzmaError::CorruptSparseIndex { .. }));
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
