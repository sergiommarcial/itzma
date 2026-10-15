//! Flat (brute-force) nearest-neighbor search.
//!
//! Every candidate is compared against the query, and a bounded max-heap keeps
//! the best `k`. That is O(n log k) time and O(k) extra memory for `n`
//! documents. The indexed path (`query.rs`) reranks its shortlist with the same
//! [`distance`].

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::document::{Document, Metric};
use crate::kernels;

/// One search result.
///
/// Results come back sorted by ascending `distance`, so the first hit is the
/// closest.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    /// Id of the matching [`Document`].
    pub id: String,
    /// Distance from the query under the namespace's metric. Lower is closer
    /// for every metric, and the value can be negative (see
    /// [`Metric::Dot`]). Compare distances within one query only; the scale
    /// differs between metrics.
    pub distance: f32,
    /// The document's attributes, exactly as stored.
    pub attrs: Option<serde_json::Value>,
}

/// Distance between `a` and `b` under `metric`. Lower is closer.
///
/// | Metric | Value |
/// |---|---|
/// | [`Metric::L2`] | `sum((a_i - b_i)^2)`, no square root |
/// | [`Metric::Dot`] | `-sum(a_i * b_i)` |
/// | [`Metric::Cosine`] | `1 - dot(a, b) / (norm(a) * norm(b))`, so 0 for the same direction, 1 for orthogonal, 2 for opposite |
///
/// The sums run in `f32` with eight independent accumulators, which the
/// compiler vectorizes. The distance is recomputed with `f64` sums
/// when that result is not finite, and under cosine also when a squared norm
/// is not finite or is too small for `f32` to hold its bits (below `1e-30`),
/// or when the distance is below `1e-4`, where `f32` resolves `1 - cos` only
/// to about `1e-7`. So a large component does not overflow part way through,
/// a tiny one does not underflow, and near-duplicates keep their order. The
/// result is the same bits for the same inputs on every call.
///
/// Under cosine, if either vector has zero norm the result is `1.0` (no
/// relationship) rather than NaN. Writes reject zero vectors, so this only
/// matters for data that reached storage some other way.
///
/// The slices are expected to have equal length. Extra components in the
/// longer one are ignored, because callers validate against the schema first.
///
/// # Examples
///
/// ```
/// use itzma_core::Metric;
/// use itzma_core::search::distance;
///
/// assert_eq!(distance(Metric::L2, &[0.0, 0.0], &[3.0, 4.0]), 25.0);
/// assert_eq!(distance(Metric::Dot, &[1.0, 2.0], &[3.0, 4.0]), -11.0);
/// assert!(distance(Metric::Cosine, &[1.0, 0.0], &[2.0, 0.0]).abs() < 1e-6);
/// ```
pub fn distance(metric: Metric, a: &[f32], b: &[f32]) -> f32 {
    let fast = match metric {
        Metric::L2 => kernels::l2(a, b),
        Metric::Dot => -kernels::dot(a, b),
        Metric::Cosine => {
            let (na, nb) = (kernels::norm2(a), kernels::norm2(b));
            let trusted = |n: f32| n.is_finite() && n >= kernels::NORM_FLOOR;
            if !(trusted(na) && trusted(nb)) {
                return reference(metric, a, b);
            }
            let cosine = 1.0 - kernels::dot(a, b) / (na.sqrt() * nb.sqrt());
            if cosine < kernels::COSINE_RECHECK {
                return reference(metric, a, b);
            }
            cosine
        }
    };
    if fast.is_finite() {
        fast
    } else {
        reference(metric, a, b)
    }
}

/// [`distance`] with `f64` sums, for the values the `f32` kernels cannot hold.
fn reference(metric: Metric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        Metric::L2 => a
            .iter()
            .zip(b)
            .map(|(x, y)| {
                let d = f64::from(*x) - f64::from(*y);
                d * d
            })
            .sum::<f64>() as f32,
        Metric::Dot => -dot_f64(a, b) as f32,
        Metric::Cosine => {
            let na = dot_f64(a, a);
            let nb = dot_f64(b, b);
            if na == 0.0 || nb == 0.0 {
                return 1.0;
            }
            (1.0 - dot_f64(a, b) / (na.sqrt() * nb.sqrt())) as f32
        }
    }
}

/// Inner product in `f64`, for [`reference`].
fn dot_f64(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum()
}

/// Heap entry ordered by (distance, id) so the worst candidate sits on top
/// and ties resolve deterministically by id.
struct Candidate<D> {
    distance: f32,
    doc: D,
}

impl<D: Borrow<Document>> PartialEq for Candidate<D> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl<D: Borrow<Document>> Eq for Candidate<D> {}
impl<D: Borrow<Document>> PartialOrd for Candidate<D> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<D: Borrow<Document>> Ord for Candidate<D> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.doc.borrow().id.cmp(&other.doc.borrow().id))
    }
}

/// The `k` documents closest to `query`, closest first.
///
/// Scans all of `docs` once. A max-heap of at most `k` entries holds the best
/// candidates so far, with the current worst on top so it can be evicted
/// cheaply. A document enters the heap only when it beats the current worst,
/// so most are compared and dropped without being moved.
///
/// Ties on distance are broken by id (ascending), so results are
/// deterministic no matter what order `docs` arrives in. That matters because
/// the caller feeds this from a `HashMap`, whose iteration order is random.
///
/// Returns fewer than `k` hits if there are fewer documents, and an empty
/// vector if `k == 0`. The vectors in `docs` must already match the query's
/// length.
///
/// # Examples
///
/// ```
/// use itzma_core::search::flat_top_k;
/// use itzma_core::{Document, Metric};
///
/// let doc = |id: &str, x: f32| Document {
///     id: id.into(),
///     vector: vec![x],
///     attrs: None,
///     sparse: None,
/// };
/// let docs = vec![doc("far", 9.0), doc("near", 1.0), doc("mid", 4.0)];
///
/// let hits = flat_top_k(Metric::L2, &[0.0], docs, 2);
/// let ids: Vec<_> = hits.iter().map(|h| h.id.as_str()).collect();
/// assert_eq!(ids, ["near", "mid"]);
/// ```
pub fn flat_top_k(
    metric: Metric,
    query: &[f32],
    docs: impl IntoIterator<Item = Document>,
    k: usize,
) -> Vec<Hit> {
    let candidates = docs.into_iter().map(|doc| Candidate {
        distance: distance(metric, query, &doc.vector),
        doc,
    });
    select(candidates, k)
        .into_iter()
        .map(|c| Hit {
            id: c.doc.id,
            distance: c.distance,
            attrs: c.doc.attrs,
        })
        .collect()
}

/// The `k` closest candidates, closest first, ties by id: a max-heap of at
/// most `k` with the current worst on top, entered only by a candidate that
/// beats it.
fn select<D: Borrow<Document>>(
    candidates: impl Iterator<Item = Candidate<D>>,
    k: usize,
) -> Vec<Candidate<D>> {
    if k == 0 {
        return Vec::new();
    }
    let mut heap: BinaryHeap<Candidate<D>> = BinaryHeap::with_capacity(k.min(1024) + 1);
    for candidate in candidates {
        if heap.len() < k {
            heap.push(candidate);
        } else if heap.peek().is_some_and(|worst| candidate < *worst) {
            heap.pop();
            heap.push(candidate);
        }
    }
    heap.into_sorted_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(id: &str, v: &[f32]) -> Document {
        Document {
            id: id.into(),
            vector: v.to_vec(),
            attrs: None,
            sparse: None,
        }
    }

    #[test]
    fn l2_is_squared_euclidean() {
        assert_eq!(distance(Metric::L2, &[0.0, 0.0], &[3.0, 4.0]), 25.0);
    }

    #[test]
    fn dot_is_negated() {
        assert_eq!(distance(Metric::Dot, &[1.0, 2.0], &[3.0, 4.0]), -11.0);
    }

    #[test]
    fn cosine_identical_is_zero_orthogonal_is_one_opposite_is_two() {
        assert!(distance(Metric::Cosine, &[1.0, 0.0], &[2.0, 0.0]).abs() < 1e-6);
        assert!((distance(Metric::Cosine, &[1.0, 0.0], &[0.0, 5.0]) - 1.0).abs() < 1e-6);
        assert!((distance(Metric::Cosine, &[1.0, 0.0], &[-1.0, 0.0]) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_with_zero_norm_stored_vector_is_one() {
        assert_eq!(distance(Metric::Cosine, &[1.0, 0.0], &[0.0, 0.0]), 1.0);
    }

    #[test]
    fn top_k_returns_closest_ascending() {
        let docs = vec![
            doc("far", &[10.0, 0.0]),
            doc("near", &[1.0, 0.0]),
            doc("mid", &[4.0, 0.0]),
        ];
        let hits = flat_top_k(Metric::L2, &[0.0, 0.0], docs, 2);
        let ids: Vec<_> = hits.iter().map(|h| h.id.as_str()).collect();
        assert_eq!(ids, ["near", "mid"]);
        assert!(hits[0].distance < hits[1].distance);
    }

    #[test]
    fn ties_break_by_id() {
        let docs = vec![doc("b", &[1.0]), doc("a", &[1.0]), doc("c", &[1.0])];
        let hits = flat_top_k(Metric::L2, &[0.0], docs, 2);
        let ids: Vec<_> = hits.iter().map(|h| h.id.as_str()).collect();
        assert_eq!(ids, ["a", "b"]);
    }

    #[test]
    fn k_larger_than_corpus_returns_all_and_zero_returns_none() {
        let docs = vec![doc("a", &[1.0]), doc("b", &[2.0])];
        assert_eq!(flat_top_k(Metric::L2, &[0.0], docs.clone(), 10).len(), 2);
        assert!(flat_top_k(Metric::L2, &[0.0], docs, 0).is_empty());
    }

    /// The `f64` reference for these tests, spelled out rather than calling
    /// the private fallback, so a bug there cannot hide.
    fn reference_f64(metric: Metric, a: &[f32], b: &[f32]) -> f64 {
        let dot = |x: &[f32], y: &[f32]| -> f64 {
            x.iter()
                .zip(y)
                .map(|(p, q)| f64::from(*p) * f64::from(*q))
                .sum()
        };
        match metric {
            Metric::L2 => a
                .iter()
                .zip(b)
                .map(|(x, y)| (f64::from(*x) - f64::from(*y)).powi(2))
                .sum(),
            Metric::Dot => -dot(a, b),
            Metric::Cosine => {
                let (na, nb) = (dot(a, a), dot(b, b));
                if na == 0.0 || nb == 0.0 {
                    return 1.0;
                }
                1.0 - dot(a, b) / (na.sqrt() * nb.sqrt())
            }
        }
    }

    #[test]
    fn tiny_vectors_keep_their_cosine() {
        // The squared norm, 4e-50, underflows f32 to zero; f64 keeps it.
        let a = [1e-25f32; 4];
        assert!(distance(Metric::Cosine, &a, &a).abs() < 1e-6);
        let b = [1e-25f32, -1e-25, 1e-25, -1e-25];
        assert!((distance(Metric::Cosine, &a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn subnormal_scale_vectors_keep_their_cosine() {
        // Squared norms of 1e-44 to 1e-46 are subnormal in f32: nonzero, but
        // with too few bits left for the products to survive.
        for s in [1e-21f32, 1e-22, 3e-23, 1e-23] {
            let (a, b) = ([3.0 * s, s], [s, 3.0 * s]);
            let got = distance(Metric::Cosine, &a, &b);
            assert!((got - 0.4).abs() < 1e-6, "s={s}: {got}");
        }
    }

    #[test]
    fn cosine_near_zero_keeps_its_resolution() {
        let a: Vec<f32> = (0..128)
            .map(|i| ((i * 7) % 13) as f32 / 13.0 + 0.1)
            .collect();
        let self_distance = distance(Metric::Cosine, &a, &a);
        assert!(self_distance.abs() < 1e-12, "self distance {self_distance}");
        let mut b = a.clone();
        b[0] += 1e-4;
        let near = distance(Metric::Cosine, &a, &b);
        let want = reference_f64(Metric::Cosine, &a, &b);
        assert!(near > 0.0, "a moved vector must be farther than itself");
        assert!(
            (f64::from(near) - want).abs() <= 1e-3 * want,
            "{near} vs {want}"
        );
    }

    #[test]
    fn huge_components_give_the_reference_value() {
        // Squared norms of 1e40 overflow f32, the cosine itself does not.
        let (a, b) = ([1e20f32, 0.0], [1e20f32, 1e20]);
        let want = reference_f64(Metric::Cosine, &a, &b) as f32;
        assert!((distance(Metric::Cosine, &a, &b) - want).abs() < 1e-6);
        assert!((want - 0.292_893_2).abs() < 1e-6);

        // Products of +-1e40 overflow to +inf and -inf, which cancel to NaN in
        // f32 and to 0 in f64.
        let (a, b) = ([1e20f32, 1e20], [1e20f32, -1e20]);
        assert_eq!(distance(Metric::Dot, &a, &b), 0.0);
    }

    #[test]
    fn distance_matches_the_f64_reference_at_every_dimension() {
        let mut seed = 9u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u32 << 23) as f32 - 1.0
        };
        for dim in [1, 3, 7, 8, 9, 25, 128, 960] {
            let a: Vec<f32> = (0..dim).map(|_| next()).collect();
            let b: Vec<f32> = (0..dim).map(|_| next()).collect();
            for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
                let want = reference_f64(metric, &a, &b);
                let got = distance(metric, &a, &b);
                assert!(
                    (f64::from(got) - want).abs() <= 1e-4 * (1.0 + want.abs()),
                    "{metric:?} at {dim}: {got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn flat_top_k_agrees_with_a_full_sort() {
        let docs: Vec<Document> = (0..200)
            .map(|i| {
                doc(
                    &format!("d{i:03}"),
                    &[((i * 37) % 101) as f32, (i % 7) as f32],
                )
            })
            .collect();
        let query = [50.0f32, 3.0];
        let mut all: Vec<(f32, String)> = docs
            .iter()
            .map(|d| (distance(Metric::L2, &query, &d.vector), d.id.clone()))
            .collect();
        all.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        for k in [1, 5, 50, 200, 300] {
            let want: Vec<&str> = all.iter().take(k).map(|(_, id)| id.as_str()).collect();
            let flat = flat_top_k(Metric::L2, &query, docs.clone(), k);
            let flat: Vec<&str> = flat.iter().map(|h| h.id.as_str()).collect();
            assert_eq!(flat, want, "flat_top_k k={k}");
        }
    }

    #[test]
    fn attrs_are_carried_onto_hits() {
        let mut d = doc("a", &[1.0]);
        d.attrs = Some(serde_json::json!({"tag": 1}));
        let hits = flat_top_k(Metric::L2, &[0.0], vec![d], 1);
        assert_eq!(hits[0].attrs, Some(serde_json::json!({"tag": 1})));
    }
}
