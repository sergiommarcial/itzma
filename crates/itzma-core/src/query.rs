//! The query engine: everything about a query that is the same
//! whichever index algorithm a segment has.
//!
//! The engine treats the namespace as a list of *sources*, oldest to newest.
//! Each live segment that the manifest gives an index (of a kind this build
//! knows) is an indexed source, read in pieces. Every run of segments without
//! one is folded into one in-memory layer and scanned flat, and so is the WAL
//! tail, which the handle keeps folded between queries and which is
//! always its own, newest source. A source proposes its closest rows, and the engine does the rest:
//!
//! 1. drop every proposal that a newer source holds as a document or as a
//!    tombstone (an id's newest holder decides it, so this is what an overwrite
//!    or a delete means),
//! 2. fetch the exact vectors of what is left and rescore them, which is the
//!    rerank, so the answer never carries an approximate distance,
//! 3. merge the sources, keep the closest `top_k` and fetch their attributes.
//!
//! A source that ends up with fewer than `top_k` live proposals while it has more
//! rows to give pulls another, bigger batch from its cursor, so
//! shadowing costs requests and never an answer. A source stops widening once it
//! has `top_k` live rows, its cursor is exhausted, or the shared top-k bound
//! proves the rest of it cannot beat what other sources have already
//! found. An index algorithm only proposes rows through a
//! [`Cursor`](crate::index::Cursor) (see [`index::Algorithm`](crate::index)), and
//! its output is checked: a row past the segment is `CorruptIndex`, and a row
//! repeated across batches is too.
//!
//! Every byte read is charged to one [`ReadBudget`] for the whole query, so a
//! namespace bigger than the budget can be queried through its indexes while
//! reading it whole is refused.

use std::collections::HashMap;
use std::sync::Arc;

use futures::future::try_join_all;
use futures::stream::{self, StreamExt, TryStreamExt};
use object_store::ObjectStore;
use object_store::path::Path;
use serde::{Deserialize, Serialize};

use crate::budget::ReadBudget;
use crate::cpu::{self, Cost, Lane};
use crate::decoded::DecodedRef;
use crate::document::Metric;
use crate::error::{ItzmaError, Result};
use crate::fetch::FetchPolicy;
use crate::index::{self, Algorithm, Candidate, Probe};
use crate::layer::Layer;
use crate::manifest::Manifest;
use crate::predicate::Predicate;
use crate::search::{self, Hit};
use crate::segment::{self, Held, RangedSegment, SegmentView};

/// The default for [`QueryOptions::rerank`], a guess until the recall harness
/// has measured one.
pub const DEFAULT_RERANK: usize = 4;

/// Requests in flight at once when a step fans out over blocks or ids.
const FETCH_CONCURRENCY: usize = 16;
/// How much bigger each batch a source is asked for grows, when it is still
/// short of `top_k` live rows and the bound has not ruled out the rest.
const WIDEN: usize = 8;

/// Settings for [`Namespace::query_with`](crate::Namespace::query_with).
///
/// New fields may be added, so build one from `default()` and the setters.
///
/// # Examples
///
/// ```
/// use itzma_core::QueryOptions;
///
/// let every_row = QueryOptions::default().rerank(None);
/// assert_eq!(every_row.rerank, None);
/// assert_eq!(QueryOptions::default().rerank, Some(4));
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct QueryOptions {
    /// How many candidates an index gives for each hit wanted: an indexed
    /// segment proposes `rerank * top_k` rows by approximate distance, and those
    /// are rescored against the exact vectors. Bigger is slower and closer to the
    /// exact answer. `None` proposes every row, which makes the answer equal the
    /// flat scan's. Must be at least 1.
    pub rerank: Option<usize>,
    /// How many partitions of an index to search. `None` lets the index
    /// choose from how many partitions it has. A bigger value costs more and
    /// finds more true neighbors. Must be at least 1.
    pub nprobe: Option<usize>,
    /// Ignore every index and scan flat. The ground truth for measuring recall,
    /// and a way through a damaged index.
    pub exact: bool,
    /// Keep only documents whose attributes match this predicate.
    /// Checked against the namespace's attribute schema before any read, so an
    /// undeclared field or an operator that does not fit its type is an error.
    /// The result has fewer than `top_k` hits only when fewer live documents
    /// match. A very selective filter can read the whole namespace, bounded by
    /// the read budget.
    pub filter: Option<Predicate>,
}

impl Default for QueryOptions {
    fn default() -> Self {
        Self {
            rerank: Some(DEFAULT_RERANK),
            nprobe: None,
            exact: false,
            filter: None,
        }
    }
}

impl QueryOptions {
    /// Set [`rerank`](Self::rerank).
    #[must_use]
    pub fn rerank(mut self, rerank: Option<usize>) -> Self {
        self.rerank = rerank;
        self
    }

    /// Set [`nprobe`](Self::nprobe).
    #[must_use]
    pub fn nprobe(mut self, nprobe: Option<usize>) -> Self {
        self.nprobe = nprobe;
        self
    }

    /// Set [`exact`](Self::exact).
    #[must_use]
    pub fn exact(mut self, exact: bool) -> Self {
        self.exact = exact;
        self
    }

    /// Set [`filter`](Self::filter).
    #[must_use]
    pub fn filter(mut self, filter: Option<Predicate>) -> Self {
        self.filter = filter;
        self
    }

    /// Refuse options that cannot be run, before any storage call.
    pub(crate) fn validate(&self) -> Result<()> {
        if self.rerank == Some(0) {
            return Err(ItzmaError::InvalidQuery("rerank must be at least 1"));
        }
        if self.nprobe == Some(0) {
            return Err(ItzmaError::InvalidQuery("nprobe must be at least 1"));
        }
        Ok(())
    }
}

/// Which code searches an index of a given kind name. A manifest may name a kind
/// this build does not have (a newer build published it), and the engine then
/// reads that segment flat like any unindexed one: an index is derived data, so
/// not knowing one costs speed and never a right answer.
pub(crate) trait Registry: Sync {
    fn algorithm(&self, kind: &str) -> Option<&dyn Algorithm>;
}

/// The kinds this build has.
pub(crate) struct Builtin;

impl Registry for Builtin {
    fn algorithm(&self, kind: &str) -> Option<&dyn Algorithm> {
        index::Kind::from_name(kind).map(index::Kind::algorithm)
    }
}

/// Whether any live segment has an index this build can search. Without one
/// every segment is one flat step.
fn uses_indexes(manifest: &Manifest, registry: &dyn Registry) -> bool {
    manifest.live_segments.iter().any(|id| {
        manifest
            .indexes
            .get(id)
            .is_some_and(|entry| registry.algorithm(&entry.kind).is_some())
    })
}

/// One query against one manifest.
pub(crate) struct Query<'a> {
    pub(crate) store: &'a dyn ObjectStore,
    pub(crate) namespace: &'a str,
    pub(crate) dim: usize,
    pub(crate) metric: Metric,
    pub(crate) vector: &'a [f32],
    pub(crate) top_k: usize,
    pub(crate) options: &'a QueryOptions,
    pub(crate) budget: &'a ReadBudget,
    pub(crate) policy: &'a FetchPolicy,
    pub(crate) registry: &'a dyn Registry,
    /// Parsed structures kept between queries, if the handle has a
    /// decoded cache.
    pub(crate) decoded: Option<&'a DecodedRef>,
}

impl Query<'_> {
    /// The predicate the query filters by, if any.
    fn filter(&self) -> Option<&Predicate> {
        self.options.filter.as_ref()
    }
}

/// What the manifest's live segments turn into.
enum Step<'a> {
    /// A segment with an index this build can search.
    Indexed {
        id: &'a str,
        generation: u32,
        algorithm: &'a dyn Algorithm,
    },
    /// A run of consecutive segments without one.
    Flat(&'a [String]),
}

/// A place a document can come from, newest last.
enum Source<'a> {
    Indexed(Box<Indexed<'a>>),
    /// A run of consecutive unindexed version 2 or 3 segments.
    Columnar(Run),
    /// A run of consecutive version 1 segments overlaid into one layer, or
    /// the WAL tail.
    Flat(Arc<Layer>),
}

/// Consecutive unindexed version 2 or 3 segments, oldest first.
struct Run {
    views: Vec<Arc<SegmentView>>,
}

impl Run {
    /// Whether any segment of the run holds `id` as a document or a
    /// tombstone: either hides it in every older source.
    fn holds(&self, id: &str) -> bool {
        self.views.iter().any(|view| view.holds(id))
    }
}

struct Indexed<'a> {
    segment: RangedSegment<'a>,
    algorithm: &'a dyn Algorithm,
    index_path: Path,
}

/// Where a hit's attributes are: already in memory, in the data object, or
/// in a view of a columnar run.
enum Attrs {
    Have(Option<serde_json::Value>),
    Row(u32),
    /// A row of one view of a columnar run, parsed only if it is a final hit.
    InView {
        view: usize,
        row: usize,
    },
}

/// A live candidate with its exact distance.
struct Scored {
    distance: f32,
    id: String,
    source: usize,
    attrs: Attrs,
}

/// Group the live segments into steps, oldest first. Consecutive unindexed
/// segments stay together, since overlaying them gives their net effect and
/// nothing between them needs a separate shadow check.
fn plan<'a>(q: &Query<'a>, manifest: &'a Manifest) -> Vec<Step<'a>> {
    let live = &manifest.live_segments;
    // The exact path, and a namespace with nothing this build can search:
    // every segment is read flat, through the same engine.
    if q.options.exact || !uses_indexes(manifest, q.registry) {
        return if live.is_empty() {
            Vec::new()
        } else {
            vec![Step::Flat(live)]
        };
    }
    let registry: &'a dyn Registry = q.registry;
    let mut steps = Vec::new();
    let mut run_start = 0;
    for (at, id) in live.iter().enumerate() {
        let known = manifest.indexes.get(id).and_then(|entry| {
            registry
                .algorithm(&entry.kind)
                .map(|algorithm| (entry.generation, algorithm))
        });
        if let Some((generation, algorithm)) = known {
            if run_start < at {
                steps.push(Step::Flat(&live[run_start..at]));
            }
            steps.push(Step::Indexed {
                id,
                generation,
                algorithm,
            });
            run_start = at + 1;
        }
    }
    if run_start < live.len() {
        steps.push(Step::Flat(&live[run_start..]));
    }
    steps
}

/// Run the query over `manifest`, reading indexed segments in pieces and the
/// rest whole.
///
/// # Errors
///
/// The errors of reading segments, indexes and the tail (a missing segment or
/// index, or a fenced WAL object, is what the caller reloads the manifest for),
/// [`ItzmaError::CorruptIndex`] for a damaged index or a row an index made up,
/// and [`ItzmaError::ReadBudgetExceeded`].
pub(crate) async fn run<'a>(
    q: &Query<'a>,
    manifest: &'a Manifest,
    tail: impl Future<Output = Result<Arc<Layer>>>,
) -> Result<Vec<Hit>> {
    if q.top_k == 0 {
        return Ok(Vec::new());
    }
    // A fresh bound per attempt, never reused across a stale-read retry: a
    // bound built from the old manifest's sources could come from a
    // document the new manifest shadows.
    let bound = crate::bound::Bound::new(q.top_k);
    let steps = plan(q, manifest);
    // The tail is read alongside the segments, and it is always its own,
    // newest source: the handle's view shares it between queries,
    // and shadowing between it and the newest unindexed run goes through
    // `Layer::holds`, which costs nothing.
    let (mut sources, tail) = futures::try_join!(open_sources(q, steps), tail)?;
    if !tail.is_empty() {
        sources.push(Source::Flat(tail));
    }

    let sources = &sources;
    let mut live: Vec<Scored> =
        try_join_all((0..sources.len()).map(|at| search_source(q, sources, at, &bound)))
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
    // whose parameter type itself borrows `'a` is inferred as higher-ranked
    // (for any lifetime), which then fails to prove the stream `Send` even
    // though every call in fact shares the same `'a` (see `segment::read_all`
    // for the same shape of fix with an owned id instead of a borrowed one).
    let opened: Vec<Vec<Source<'a>>> = stream::iter(0..steps.len())
        .map(|at| open_source(q, &steps[at]))
        .buffered(FETCH_CONCURRENCY)
        .try_collect()
        .await?;
    Ok(opened.into_iter().flatten().collect())
}

/// The sources one step opens to: one for an indexed segment; for a run of
/// unindexed segments, a columnar run per stretch of version 2 or 3 segments and
/// a layer per stretch of version 1 ones, in manifest order.
async fn open_source<'a>(q: &Query<'a>, step: &Step<'a>) -> Result<Vec<Source<'a>>> {
    match *step {
        Step::Indexed {
            id,
            generation,
            algorithm,
        } => {
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
            Ok(vec![Source::Indexed(Box::new(Indexed {
                segment,
                algorithm,
                index_path: index::index_path(q.namespace, id, generation),
            }))])
        }
        Step::Flat(ids) => {
            let views = segment::read_flat(
                q.store,
                q.namespace,
                ids,
                q.dim,
                q.budget,
                q.policy,
                Lane::Query,
                q.decoded,
            )
            .await?;
            if views.is_empty() {
                return Ok(Vec::new());
            }
            Ok(vec![Source::Columnar(Run { views })])
        }
    }
}

/// The live candidates of source `at`.
async fn search_source(
    q: &Query<'_>,
    sources: &[Source<'_>],
    at: usize,
    bound: &crate::bound::Bound,
) -> Result<Vec<Scored>> {
    match &sources[at] {
        Source::Indexed(source) => search_indexed(q, sources, at, source, bound).await,
        Source::Columnar(run) => search_columnar(q, sources, at, run, bound).await,
        Source::Flat(layer) => search_flat(q, sources, at, layer, bound).await,
    }
}

/// Pull candidates from the index's cursor in growing batches, dropping
/// those the bound proves cannot matter, running the shadow check and the
/// predicate on new rows only, and stopping as soon as `top_k` live rows are
/// in hand, the cursor is exhausted, or the rest of the cursor cannot beat
/// the bound — replaces the old `Round`/`MAX_WIDENINGS`
/// re-ask-from-scratch loop.
async fn search_indexed(
    q: &Query<'_>,
    sources: &[Source<'_>],
    at: usize,
    source: &Indexed<'_>,
    bound: &crate::bound::Bound,
) -> Result<Vec<Scored>> {
    let rows = source.segment.doc_count();
    let probe = Probe {
        store: q.store,
        budget: q.budget,
        policy: q.policy,
        path: &source.index_path,
        dim: q.dim,
        rows,
        data_size: source.segment.data_size(),
        metric: q.metric,
        query: q.vector,
        want: match q.options.rerank {
            Some(rerank) => rerank.saturating_mul(q.top_k),
            None => usize::MAX,
        },
        nprobe: q.options.nprobe,
        decoded: q.decoded,
    };
    let mut cursor = source.algorithm.cursor(probe).await?;
    let mut n = probe.want;
    let mut live: Vec<Scored> = Vec::new();
    let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
    loop {
        let batch = cursor.next(n).await?;
        for c in &batch.candidates {
            if u64::from(c.row) >= rows {
                return Err(index::corrupt(
                    source.index_path.as_ref(),
                    format!("the index proposed row {}, the segment has {rows}", c.row),
                ));
            }
            if !seen.insert(c.row) {
                return Err(index::corrupt(
                    source.index_path.as_ref(),
                    format!("the index proposed row {} more than once", c.row),
                ));
            }
        }
        let candidates: Vec<Candidate> = batch
            .candidates
            .into_iter()
            .filter(|c| {
                source
                    .algorithm
                    .lower_bound(&probe, c.distance)
                    .is_none_or(|lb| !bound.beats(lb))
            })
            .collect();
        let scored = evaluate_candidates(q, sources, at, source, candidates).await?;
        bound.offer(&scored.iter().map(|s| s.distance).collect::<Vec<_>>());
        live.extend(scored);
        if live.len() >= q.top_k || batch.exhausted {
            return Ok(live);
        }
        if let Some(rest) = batch.rest_lower_bound {
            if bound.beats(rest) {
                return Ok(live);
            }
        }
        n = n.saturating_mul(WIDEN);
    }
}

/// The shadow check, the predicate, and the exact rerank for one batch of
/// fresh candidates — the body `evaluate_indexed` used to run once per
/// widening round, now run once per cursor batch on new rows only.
async fn evaluate_candidates(
    q: &Query<'_>,
    sources: &[Source<'_>],
    at: usize,
    source: &Indexed<'_>,
    candidates: Vec<Candidate>,
) -> Result<Vec<Scored>> {
    let rows: Vec<u32> = candidates.iter().map(|c| c.row).collect();
    let ids = source.segment.ids_of(&rows).await?;
    let live = unshadowed(sources, at, &ids).await?;

    let kept: Vec<(u32, String)> = rows
        .into_iter()
        .zip(ids)
        .zip(live)
        .filter_map(|(pair, live)| live.then_some(pair))
        .collect();
    // Test the predicate before fetching vectors, so a selective filter does
    // not pay for rows it throws away. The attributes fetched here stay on the
    // hit and `finish` does not fetch them again.
    let (kept, attrs): (Vec<(u32, String)>, Vec<Attrs>) = match q.filter() {
        None => {
            let attrs = kept.iter().map(|(row, _)| Attrs::Row(*row)).collect();
            (kept, attrs)
        }
        Some(predicate) => {
            let rows: Vec<u32> = kept.iter().map(|(row, _)| *row).collect();
            let fetched = source.segment.attrs_of(&rows).await?;
            kept.into_iter()
                .zip(fetched)
                .filter(|(_, attrs)| predicate.matches(attrs.as_ref()))
                .map(|(pair, attrs)| (pair, Attrs::Have(attrs)))
                .unzip()
        }
    };
    let kept_rows: Vec<u32> = kept.iter().map(|(row, _)| *row).collect();
    let vectors = source.segment.vectors_of(&kept_rows).await?;
    Ok(kept
        .into_iter()
        .enumerate()
        .zip(attrs)
        .map(|((i, (_, id)), attrs)| Scored {
            distance: search::distance(q.metric, q.vector, vectors.row(i)),
            id,
            source: at,
            attrs,
        })
        .collect())
}

/// The same shape for a layer held in memory, whose distances are already
/// exact: sort every filtered document once, then slice growing batches off
/// the front, stopping early the same three ways `search_indexed` does.
/// Order used to rank `(distance, doc)` pairs in [`search_flat`]: closest
/// first, ties by id.
fn by_distance_then_id(
    a: &(f32, &crate::document::Document),
    b: &(f32, &crate::document::Document),
) -> std::cmp::Ordering {
    a.0.total_cmp(&b.0).then_with(|| a.1.id.cmp(&b.1.id))
}

async fn search_flat(
    q: &Query<'_>,
    sources: &[Source<'_>],
    at: usize,
    layer: &Arc<Layer>,
    bound: &crate::bound::Bound,
) -> Result<Vec<Scored>> {
    // Distances computed once; each widening round *partitions* the
    // not-yet-returned suffix (`select_nth_unstable_by`, O(remaining)) for
    // just the rows this round newly needs, then sorts only that small new
    // slice — never a full O(N log N) sort of the whole filtered layer up
    // front. A full sort here was a real, measured regression (up to +280%
    // at 20k rows on an unindexed/flat query), since the common case needs
    // only `top_k` rows, not every row in order.
    let (owned, query, filter, metric) = (
        layer.clone(),
        q.vector.to_vec(),
        q.filter().cloned(),
        q.metric,
    );
    let cost = Cost::Distances {
        rows: layer.doc_count(),
        dim: q.vector.len(),
    };
    let distances = cpu::run(Lane::Query, cost, move || {
        flat_distances(&owned, metric, &query, filter.as_ref())
    })
    .await;
    // `Layer::docs` keeps its order while the layer is unchanged (its
    // contract), so this second pass lines up with the one on the pool.
    debug_assert_eq!(distances.len(), layer.doc_count());
    let mut scored: Vec<(f32, &crate::document::Document)> = layer
        .docs()
        .zip(distances)
        .filter_map(|(doc, distance)| distance.map(|d| (d, doc)))
        .collect();

    let mut live: Vec<Scored> = Vec::new();
    let mut returned = 0usize;
    let mut n = q.top_k;
    loop {
        let take = n.min(scored.len());
        if take > returned {
            scored[returned..].select_nth_unstable_by(take - returned - 1, by_distance_then_id);
            scored[returned..take].sort_by(by_distance_then_id);
        }
        let batch = &scored[returned..take];
        let ids: Vec<String> = batch.iter().map(|(_, doc)| doc.id.clone()).collect();
        let shadow = unshadowed(sources, at, &ids).await?;
        let scored_batch: Vec<Scored> = batch
            .iter()
            .zip(shadow)
            .filter(|(_, live)| *live)
            .map(|((distance, doc), _)| Scored {
                distance: *distance,
                id: doc.id.clone(),
                source: at,
                attrs: Attrs::Have(doc.attrs.clone()),
            })
            .collect();
        bound.offer(&scored_batch.iter().map(|s| s.distance).collect::<Vec<_>>());
        returned = take;
        live.extend(scored_batch);
        let exhausted = returned >= scored.len();
        if live.len() >= q.top_k || exhausted {
            return Ok(live);
        }
        // The rows past `returned` are in no order after the partial select;
        // the batch's largest distance is a true lower bound for all of them.
        let rest_lower_bound = scored[returned - 1].0;
        if bound.beats(rest_lower_bound) {
            return Ok(live);
        }
        n = n.saturating_mul(WIDEN);
    }
}

/// Rows per span of a columnar scan handed to one pool task: the same
/// constant as the layer path, so neither depends on the thread count.
const SPAN: usize = cpu::FLAT_CHUNK;

/// The distance to every surviving row of `views` that `filter` keeps, as
/// (distance, view, row), views in order and rows ascending in each.
/// Survivors first, then the predicate (attributes parsed only for
/// survivors), then the kernel on the row's slice of the matrix.
///
/// # Errors
///
/// [`ItzmaError::CorruptSegment`] for attribute JSON that does not parse.
pub(crate) fn columnar_distances(
    views: &[Arc<SegmentView>],
    metric: Metric,
    query: &[f32],
    filter: Option<&Predicate>,
) -> Result<Vec<(f32, usize, usize)>> {
    // One segment shadows nothing of its own: every row survives.
    let live = if views.len() == 1 {
        None
    } else {
        let sorted: Vec<crate::layer::SortedIds<'_>> =
            views.iter().map(|view| view.sorted_ids()).collect();
        Some(crate::layer::survivors(&sorted))
    };
    let spans: Vec<(usize, usize, usize)> = views
        .iter()
        .enumerate()
        .flat_map(|(v, view)| {
            (0..view.rows())
                .step_by(SPAN)
                .map(move |start| (v, start, (start + SPAN).min(view.rows())))
        })
        .collect();
    let chunks = cpu::chunked(&spans, 1, |chunk| {
        let mut out = Vec::new();
        for &(v, start, end) in chunk {
            for row in start..end {
                if live.as_ref().is_some_and(|live| !live[v].get(row)) {
                    continue;
                }
                if let Some(predicate) = filter {
                    if !predicate.matches(views[v].attrs(row)?.as_ref()) {
                        continue;
                    }
                }
                out.push((
                    search::distance(metric, query, views[v].vector(row)),
                    v,
                    row,
                ));
            }
        }
        Ok::<_, ItzmaError>(out)
    });
    let mut out = Vec::with_capacity(
        chunks
            .iter()
            .map(|chunk| chunk.as_ref().map_or(0, Vec::len))
            .sum(),
    );
    for chunk in chunks {
        out.extend(chunk?);
    }
    Ok(out)
}

/// [`search_flat`]'s loop over a columnar run: distances on the pool once,
/// then growing batches off the front, each checked against newer sources.
async fn search_columnar(
    q: &Query<'_>,
    sources: &[Source<'_>],
    at: usize,
    run: &Run,
    bound: &crate::bound::Bound,
) -> Result<Vec<Scored>> {
    // The views are `Arc` clones: dropping them on the pool frees nothing,
    // the run keeps them (the rule).
    let (views, query, filter, metric) = (
        run.views.clone(),
        q.vector.to_vec(),
        q.filter().cloned(),
        q.metric,
    );
    let cost = Cost::Distances {
        rows: run.views.iter().map(|view| view.rows()).sum(),
        dim: q.vector.len(),
    };
    let mut scored = cpu::run(Lane::Query, cost, move || {
        columnar_distances(&views, metric, &query, filter.as_ref())
    })
    .await?;
    let id = |&(_, v, row): &(f32, usize, usize)| run.views[v].id(row);
    let by = |a: &(f32, usize, usize), b: &(f32, usize, usize)| {
        a.0.total_cmp(&b.0).then_with(|| id(a).cmp(id(b)))
    };

    let mut live: Vec<Scored> = Vec::new();
    let mut returned = 0usize;
    let mut n = q.top_k;
    loop {
        let take = n.min(scored.len());
        if take > returned {
            scored[returned..].select_nth_unstable_by(take - returned - 1, by);
            scored[returned..take].sort_by(by);
        }
        let batch = &scored[returned..take];
        let ids: Vec<String> = batch.iter().map(|s| id(s).to_owned()).collect();
        let shadow = unshadowed(sources, at, &ids).await?;
        let mut scored_batch = Vec::new();
        for (&(distance, v, row), (id, live)) in batch.iter().zip(ids.into_iter().zip(shadow)) {
            if live {
                scored_batch.push(Scored {
                    distance,
                    id,
                    source: at,
                    attrs: Attrs::InView { view: v, row },
                });
            }
        }
        bound.offer(&scored_batch.iter().map(|s| s.distance).collect::<Vec<_>>());
        returned = take;
        live.extend(scored_batch);
        if live.len() >= q.top_k || returned >= scored.len() {
            return Ok(live);
        }
        // The rows past `returned` are in no order after the partial select;
        // the batch's largest distance is a true lower bound for all of them.
        let rest_lower_bound = scored[returned - 1].0;
        if bound.beats(rest_lower_bound) {
            return Ok(live);
        }
        n = n.saturating_mul(WIDEN);
    }
}

/// The distance from `query` to every document of `layer`, in `layer.docs()`
/// order, `None` where `filter` drops the document. Chunked so that on a pool
/// thread the chunks run on several cores; a chunk is a run of
/// documents in that order, so the result is the same either way.
pub(crate) fn flat_distances(
    layer: &Layer,
    metric: Metric,
    query: &[f32],
    filter: Option<&Predicate>,
) -> Vec<Option<f32>> {
    let docs: Vec<&crate::document::Document> = layer.docs().collect();
    distances_of(&docs, metric, query, filter)
}

/// [`flat_distances`] over a list of documents, in its order.
fn distances_of(
    docs: &[&crate::document::Document],
    metric: Metric,
    query: &[f32],
    filter: Option<&Predicate>,
) -> Vec<Option<f32>> {
    let chunks = cpu::chunked(docs, cpu::FLAT_CHUNK, |chunk| {
        chunk
            .iter()
            .map(|doc| {
                filter
                    .is_none_or(|p| p.matches(doc.attrs.as_ref()))
                    .then(|| search::distance(metric, query, &doc.vector))
            })
            .collect::<Vec<_>>()
    });
    let mut out = Vec::with_capacity(docs.len());
    for chunk in chunks {
        out.extend(chunk);
    }
    out
}

/// For each of `ids`, proposed by source `at`: whether it is still the live
/// version, that is, no newer source holds the id as a document or a tombstone.
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
    // Runs and layers in memory first: they cost no request.
    if newer.iter().any(|source| match source {
        Source::Flat(layer) => layer.holds(id),
        Source::Columnar(run) => run.holds(id),
        Source::Indexed(_) => false,
    }) {
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
    live.into_iter()
        .map(|hit| {
            let attrs = match hit.attrs {
                Attrs::Have(value) => value,
                Attrs::Row(row) => attrs.remove(&(hit.source, row)).flatten(),
                Attrs::InView { view, row } => match &sources[hit.source] {
                    Source::Columnar(run) => run.views[view].attrs(row)?,
                    _ => None,
                },
            };
            Ok(Hit {
                attrs,
                id: hit.id,
                distance: hit.distance,
            })
        })
        .collect()
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

/// Unit tests for [`Indexed::propose`] with a stub kind, so the engine's own
/// checks on what an algorithm returns are tested without a real index object:
/// nothing here names SQ8. The engine tests that need a real, published index
/// are integration tests in `tests/query.rs`.
#[cfg(test)]
mod tests {
    use futures::FutureExt;
    use futures::future::BoxFuture;
    use object_store::ObjectStoreExt;
    use object_store::memory::InMemory;

    use super::*;
    use crate::document::{Document, WalOp};
    use crate::segment::{self, SegmentView};

    const DIM: usize = 2;

    fn doc(id: &str, x: f32, y: f32) -> Document {
        Document {
            id: id.into(),
            vector: vec![x, y],
            attrs: None,
            sparse: None,
        }
    }

    /// `n` documents with the same vector of length `dim`, every other one
    /// `active`, so every distance ties and the filter drops half.
    fn layer_of_identical(n: usize, dim: usize) -> Layer {
        let mut layer = Layer::new();
        for i in 0..n {
            layer.apply(WalOp::Upsert(Document {
                id: format!("d{i:06}"),
                vector: vec![1.0; dim],
                attrs: Some(serde_json::json!({ "active": i % 2 == 0 })),
                sparse: None,
            }));
        }
        layer
    }

    fn view_of(layer: &Layer, dim: usize) -> Arc<SegmentView> {
        let bytes = segment::encode_as(layer, dim, 2).unwrap();
        Arc::new(SegmentView::decode("p", bytes, dim).unwrap())
    }

    /// The distance from `query` to every document of `layer` that `filter`
    /// keeps, by id: the reference the columnar scan is held to.
    fn reference(layer: &Layer, query: &[f32], filter: Option<&Predicate>) -> Vec<(String, f32)> {
        let mut out: Vec<(String, f32)> = layer
            .docs()
            .filter(|d| filter.is_none_or(|p| p.matches(d.attrs.as_ref())))
            .map(|d| (d.id.clone(), search::distance(Metric::L2, query, &d.vector)))
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// Random runs: the columnar distances are the flat ones over the
    /// overlaid layer, as (id, distance) sets, with and without a filter.
    #[test]
    fn columnar_distances_equal_flat_distances_over_the_overlay() {
        let dim = 4;
        let mut state = 0x0043_0005_u64;
        let mut next = move |n: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % n
        };
        let filter = Predicate::is_true("active");
        for case in 0..50 {
            let layers: Vec<Layer> = (0..1 + next(4))
                .map(|s| {
                    let mut layer = Layer::new();
                    for i in 0..next(700) {
                        let id = format!("i{:04}", next(500));
                        if next(5) == 0 {
                            layer.apply(WalOp::Delete(id));
                        } else {
                            let x = (s * 1000 + i) as f32;
                            layer.apply(WalOp::Upsert(Document {
                                id,
                                vector: vec![x, -x, 0.5, 1.0],
                                attrs: Some(serde_json::json!({ "active": next(2) == 0 })),
                                sparse: None,
                            }));
                        }
                    }
                    layer
                })
                .collect();
            let views: Vec<Arc<SegmentView>> = layers.iter().map(|l| view_of(l, dim)).collect();
            let mut overlaid = Layer::new();
            for l in layers.iter().cloned() {
                overlaid.overlay(l);
            }
            let query = [3.0, 1.0, 0.0, -2.0];
            for filter in [None, Some(&filter)] {
                let mut got: Vec<(String, f32)> =
                    columnar_distances(&views, Metric::L2, &query, filter)
                        .unwrap()
                        .into_iter()
                        .map(|(d, v, row)| (views[v].id(row).to_owned(), d))
                        .collect();
                got.sort_by(|a, b| a.0.cmp(&b.0));
                assert_eq!(got, reference(&overlaid, &query, filter), "case {case}");
            }
        }
    }

    #[tokio::test]
    async fn columnar_distances_on_the_pool_equal_the_inline_ones_across_chunks() {
        let dim = 64;
        let layer = layer_of_identical(crate::cpu::FLAT_CHUNK * 2 + 7, dim);
        let views = vec![view_of(&layer, dim)];
        let query = vec![0.5f32; dim];
        for filter in [None, Some(Predicate::is_true("active"))] {
            let inline = columnar_distances(&views, Metric::L2, &query, filter.as_ref()).unwrap();
            let (v, q, f) = (views.clone(), query.clone(), filter.clone());
            let cost = crate::cpu::Cost::Distances {
                rows: layer.doc_count(),
                dim,
            };
            let pooled = crate::cpu::run(crate::cpu::Lane::Query, cost, move || {
                columnar_distances(&v, Metric::L2, &q, f.as_ref())
            })
            .await
            .unwrap();
            assert_eq!(pooled, inline);
            let expected = if filter.is_some() {
                layer.doc_count().div_ceil(2)
            } else {
                layer.doc_count()
            };
            assert_eq!(inline.len(), expected);
        }
    }

    #[tokio::test]
    async fn flat_distances_on_the_pool_equal_the_inline_ones_across_chunks() {
        let dim = 64;
        let layer = Arc::new(layer_of_identical(crate::cpu::FLAT_CHUNK * 2 + 7, dim));
        let query = vec![0.5f32; dim];
        let filter = Predicate::is_true("active");
        for filter in [None, Some(filter)] {
            let inline = flat_distances(&layer, Metric::L2, &query, filter.as_ref());
            assert_eq!(inline.len(), layer.doc_count());
            let (l, q, f) = (layer.clone(), query.clone(), filter.clone());
            let cost = crate::cpu::Cost::Distances {
                rows: layer.doc_count(),
                dim,
            };
            let pooled = crate::cpu::run(crate::cpu::Lane::Query, cost, move || {
                flat_distances(&l, Metric::L2, &q, f.as_ref())
            })
            .await;
            assert_eq!(pooled, inline);
            let kept = inline.iter().filter(|d| d.is_some()).count();
            let expected = if filter.is_some() {
                layer.doc_count().div_ceil(2)
            } else {
                layer.doc_count()
            };
            assert_eq!(kept, expected);
        }
    }

    /// A version 2 segment of 3 documents at rows 0, 1, 2, written under `docs`.
    async fn small_segment(store: &dyn ObjectStore) -> String {
        let mut layer = Layer::new();
        for d in [doc("a", 0.0, 0.0), doc("b", 1.0, 0.0), doc("c", 2.0, 0.0)] {
            layer.apply(WalOp::Upsert(d));
        }
        layer.drop_tombstones();
        let id = segment::segment_id(1, 1);
        let bytes = segment::encode_as(&layer, DIM, 2).expect("encode");
        store
            .put(&segment::segment_path("docs", &id), bytes.into())
            .await
            .expect("put segment");
        id
    }

    /// An algorithm that returns exactly the rows it is built with, ignoring
    /// `want`: good enough to stand in for a real kind, and easy to make lie,
    /// since `propose`'s own dedup and cut have something to do.
    struct Stub(Vec<Candidate>);

    impl Algorithm for Stub {
        fn search<'a>(&'a self, probe: &'a Probe<'_>) -> BoxFuture<'a, Result<Vec<Candidate>>> {
            let _ = probe;
            let rows = self.0.clone();
            async move { Ok(rows) }.boxed()
        }
    }

    /// Build the pieces `Indexed::propose` needs against `id`'s segment. The
    /// query is a placeholder: `propose` never looks past what the stub gives it.
    fn probe_query<'a>(
        store: &'a dyn ObjectStore,
        vector: &'a [f32],
        budget: &'a ReadBudget,
        policy: &'a FetchPolicy,
        options: &'a QueryOptions,
    ) -> Query<'a> {
        Query {
            store,
            namespace: "docs",
            dim: DIM,
            metric: Metric::L2,
            vector,
            top_k: 1,
            options,
            budget,
            policy,
            registry: &Builtin,
            decoded: None,
        }
    }

    #[tokio::test]
    async fn a_row_past_the_segment_is_corrupt_index() {
        let store = InMemory::new();
        let id = small_segment(&store).await;
        let budget = ReadBudget::unlimited();
        let policy = FetchPolicy::local();
        let segment = RangedSegment::open(&store, "docs", &id, DIM, &budget, &policy, None)
            .await
            .expect("open");
        let stub = Stub(vec![Candidate {
            row: 3, // the segment only has rows 0..3
            distance: 0.0,
        }]);
        let path = index::index_path("docs", &id, 1);
        let probe = Probe {
            store: &store,
            budget: &budget,
            policy: &policy,
            path: &path,
            dim: DIM,
            rows: segment.doc_count(),
            data_size: segment.data_size(),
            metric: Metric::L2,
            query: &[0.0, 0.0],
            want: 1,
            nprobe: None,
            decoded: None,
        };

        // `DefaultCursor` is what now owns the checks `Indexed::propose` used
        // to run on whatever an algorithm hands back.
        let mut cursor = stub.cursor(probe).await.expect("cursor");
        let err = cursor.next(1).await.unwrap_err();
        assert!(
            matches!(err, ItzmaError::CorruptIndex { .. }),
            "a dishonest kind must not be trusted: {err:?}"
        );
    }

    #[tokio::test]
    async fn a_filter_that_matches_no_candidate_leaves_no_live_rows() {
        let store = InMemory::new();
        let id = small_segment(&store).await;
        let budget = ReadBudget::unlimited();
        let policy = FetchPolicy::local();
        let segment = RangedSegment::open(&store, "docs", &id, DIM, &budget, &policy, None)
            .await
            .expect("open");
        let stub = Stub((0..3).map(|row| Candidate { row, distance: 0.0 }).collect());
        let sources = vec![Source::Indexed(Box::new(Indexed {
            segment,
            algorithm: &stub,
            index_path: index::index_path("docs", &id, 1),
        }))];

        // The documents have no attributes, so no one of them is `x = true`.
        let none = QueryOptions::default().filter(Some(Predicate::is_true("x")));
        let q = probe_query(&store, &[0.0, 0.0], &budget, &policy, &none);
        let bound = crate::bound::Bound::new(q.top_k);
        let live = search_source(&q, &sources, 0, &bound)
            .await
            .expect("search");
        assert!(live.is_empty());

        // All of them are `x is null`, and the attributes fetched for the test
        // stay on the hit so `finish` does not fetch them again.
        let all = QueryOptions::default().filter(Some(Predicate::IsNull("x".into())));
        let q = probe_query(&store, &[0.0, 0.0], &budget, &policy, &all);
        let bound = crate::bound::Bound::new(q.top_k);
        let live = search_source(&q, &sources, 0, &bound)
            .await
            .expect("search");
        assert_eq!(live.len(), 3);
        assert!(
            live.iter()
                .all(|hit| matches!(hit.attrs, Attrs::Have(None)))
        );

        let plain = QueryOptions::default();
        let q = probe_query(&store, &[0.0, 0.0], &budget, &policy, &plain);
        let bound = crate::bound::Bound::new(q.top_k);
        let live = search_source(&q, &sources, 0, &bound)
            .await
            .expect("search");
        assert_eq!(live.len(), 3, "without a filter all three are live");
        assert!(
            live.iter().all(|hit| matches!(hit.attrs, Attrs::Row(_))),
            "and their attributes are left for `finish`"
        );
    }

    #[tokio::test]
    async fn asking_for_every_row_gives_every_row_even_if_the_algorithm_cannot_reach_one() {
        // A graph index can leave a row with no way in. The widest ask is the
        // last resort of the widening loop, so it has to mean every row, or a
        // filter that only that row matches would find nothing.
        let store = InMemory::new();
        let id = small_segment(&store).await;
        let budget = ReadBudget::unlimited();
        let policy = FetchPolicy::local();
        let segment = RangedSegment::open(&store, "docs", &id, DIM, &budget, &policy, None)
            .await
            .expect("open");
        let stub = Stub(vec![Candidate {
            row: 0,
            distance: 1.0,
        }]);
        let path = index::index_path("docs", &id, 1);
        let probe = Probe {
            store: &store,
            budget: &budget,
            policy: &policy,
            path: &path,
            dim: DIM,
            rows: segment.doc_count(),
            data_size: segment.data_size(),
            metric: Metric::L2,
            query: &[0.0, 0.0],
            want: usize::MAX,
            nprobe: None,
            decoded: None,
        };

        let mut cursor = stub.cursor(probe).await.expect("cursor");
        let rows: Vec<u32> = cursor
            .next(usize::MAX)
            .await
            .expect("next")
            .candidates
            .iter()
            .map(|c| c.row)
            .collect();
        assert_eq!(rows, [0, 1, 2], "the reachable row first, then the rest");
    }

    #[tokio::test]
    async fn an_index_that_gave_less_than_asked_is_next_asked_for_every_row() {
        // An index that proposed fewer rows than `want` returns the same rows for
        // any bigger `want` (it only probes some partitions), so the next ask has
        // to be for every row, not for eight times as many, twice more, in vain.
        struct Counting(std::sync::atomic::AtomicUsize);
        impl Algorithm for Counting {
            fn search<'a>(&'a self, probe: &'a Probe<'_>) -> BoxFuture<'a, Result<Vec<Candidate>>> {
                let _ = probe;
                self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async move {
                    Ok(vec![Candidate {
                        row: 0,
                        distance: 0.0,
                    }])
                }
                .boxed()
            }
        }

        let store = InMemory::new();
        let id = small_segment(&store).await;
        let budget = ReadBudget::unlimited();
        let policy = FetchPolicy::local();
        let options = QueryOptions::default();
        let segment = RangedSegment::open(&store, "docs", &id, DIM, &budget, &policy, None)
            .await
            .expect("open");
        let counting = Counting(std::sync::atomic::AtomicUsize::new(0));
        let sources = vec![Source::Indexed(Box::new(Indexed {
            segment,
            algorithm: &counting,
            index_path: index::index_path("docs", &id, 1),
        }))];
        let mut q = probe_query(&store, &[0.0, 0.0], &budget, &policy, &options);
        q.top_k = 5;

        let bound = crate::bound::Bound::new(q.top_k);
        let live = search_source(&q, &sources, 0, &bound)
            .await
            .expect("search");
        assert_eq!(live.len(), 3, "the fill-in reaches every row");
        assert_eq!(
            counting.0.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "one ask for a shortlist, one for every row"
        );
    }

    #[tokio::test]
    async fn duplicate_rows_count_once_and_extra_rows_are_cut() {
        let store = InMemory::new();
        let id = small_segment(&store).await;
        let budget = ReadBudget::unlimited();
        let policy = FetchPolicy::local();
        let segment = RangedSegment::open(&store, "docs", &id, DIM, &budget, &policy, None)
            .await
            .expect("open");
        let stub = Stub(vec![
            Candidate {
                row: 1,
                distance: 5.0,
            },
            Candidate {
                row: 1,
                distance: 5.0,
            }, // repeated: counts once
            Candidate {
                row: 0,
                distance: 1.0,
            },
            Candidate {
                row: 2,
                distance: 9.0,
            }, // more than `want`: cut
        ]);
        let path = index::index_path("docs", &id, 1);
        let probe = Probe {
            store: &store,
            budget: &budget,
            policy: &policy,
            path: &path,
            dim: DIM,
            rows: segment.doc_count(),
            data_size: segment.data_size(),
            metric: Metric::L2,
            query: &[0.0, 0.0],
            want: 3,
            nprobe: None,
            decoded: None,
        };

        // `want` matches the number of distinct rows, so a duplicate that
        // survived would show up in the result rather than being cut by
        // truncation alone.
        let found = stub
            .cursor(probe)
            .await
            .expect("cursor")
            .next(3)
            .await
            .expect("next")
            .candidates;
        assert_eq!(
            found,
            vec![
                Candidate {
                    row: 0,
                    distance: 1.0
                },
                Candidate {
                    row: 1,
                    distance: 5.0
                },
                Candidate {
                    row: 2,
                    distance: 9.0
                },
            ]
        );
    }

    /// Regression: a cursor that repeats a row across batches is
    /// `CorruptIndex`. A cursor's own internal dedup (`DefaultCursor`'s
    /// `returned` set, `Sq8Cursor`'s drain) guards against this for the
    /// two cursors this crate ships, but `search_indexed`'s own `seen`
    /// check is what the engine actually relies on — this test drives
    /// that check directly with a dishonest `Cursor` that repeats a row
    /// across two `next` calls, bypassing both real cursors entirely.
    #[tokio::test]
    async fn a_cursor_that_repeats_a_row_across_batches_is_corrupt_index() {
        struct RepeatsRowZero;
        impl Algorithm for RepeatsRowZero {
            fn search<'a>(&'a self, probe: &'a Probe<'_>) -> BoxFuture<'a, Result<Vec<Candidate>>> {
                let _ = probe;
                async move { Ok(Vec::new()) }.boxed()
            }

            fn cursor<'a>(
                &'a self,
                probe: Probe<'a>,
            ) -> BoxFuture<'a, Result<Box<dyn index::Cursor + 'a>>> {
                struct Dishonest(u32);
                impl index::Cursor for Dishonest {
                    fn next<'a>(&'a mut self, n: usize) -> BoxFuture<'a, Result<index::Batch>> {
                        let _ = n;
                        let row = self.0;
                        self.0 += 1; // never actually advances past row 0
                        async move {
                            Ok(index::Batch {
                                candidates: vec![Candidate {
                                    row: 0, // the same row, every batch
                                    distance: row as f32,
                                }],
                                rest_lower_bound: None,
                                exhausted: false,
                            })
                        }
                        .boxed()
                    }
                }
                let _ = probe;
                async move { Ok(Box::new(Dishonest(0)) as Box<dyn index::Cursor>) }.boxed()
            }
        }

        let store = InMemory::new();
        let id = small_segment(&store).await;
        let budget = ReadBudget::unlimited();
        let policy = FetchPolicy::local();
        // rerank 1 makes want = 1 per batch, forcing a second next() call.
        let options = QueryOptions::default().rerank(Some(1));
        let segment = RangedSegment::open(&store, "docs", &id, DIM, &budget, &policy, None)
            .await
            .expect("open");
        let dishonest = RepeatsRowZero;
        let sources = vec![Source::Indexed(Box::new(Indexed {
            segment,
            algorithm: &dishonest,
            index_path: index::index_path("docs", &id, 1),
        }))];
        let mut q = probe_query(&store, &[0.0, 0.0], &budget, &policy, &options);
        q.top_k = 3; // more than the dishonest cursor will ever truly deliver

        let bound = crate::bound::Bound::new(q.top_k);
        let err = match search_source(&q, &sources, 0, &bound).await {
            Ok(_) => panic!("a cursor that repeats a row across batches must not be trusted"),
            Err(e) => e,
        };
        assert!(
            matches!(err, ItzmaError::CorruptIndex { .. }),
            "a cursor that repeats a row across batches must not be trusted: {err:?}"
        );
    }
}
