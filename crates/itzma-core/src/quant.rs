//! Scalar quantization to one byte per dimension (the SQ8 codec), public so
//! code outside this crate can reuse the exact quantization the SQ8 index
//! uses instead of a second implementation that could drift from it.
//!
//! Each dimension gets its own range, the smallest to the largest value seen
//! at [`Sq8Codec::fit`], split into 255 steps. A row is stored as one byte
//! per dimension and reconstructed as `min + code * scale`, off by at most
//! half a step per dimension. [`Sq8Codec::distance`] is asymmetric: the query
//! stays `f32`, only the stored side is dequantized, matching
//! [`search::distance`](crate::search::distance) for the reconstructed
//! vector. [`Sq8Codec::prepare`] does the per-query work once, for
//! a caller comparing one query with many codes.
//!
//! `encode`, `decode`, `distance` and [`Sq8Codec::prepare`] follow the same
//! length convention as
//! [`search::distance`](crate::search::distance): they never panic on a
//! length mismatch, they just work over the shortest of the lengths involved
//! (`dim`, and whichever slice arguments are given), on the assumption a
//! caller validates shapes against its own schema first.

use crate::document::Metric;
use crate::error::{ItzmaError, Result};
use crate::kernels;

/// Codes are 0 to 255.
const STEPS: f32 = 255.0;

/// Per-dimension scalar quantization parameters, fit from a set of vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct Sq8Codec {
    min: Vec<f32>,
    scale: Vec<f32>,
}

fn quantize_one(x: f32, min: f32, scale: f32) -> u8 {
    if scale == 0.0 {
        return 0;
    }
    ((x - min) / scale).round().clamp(0.0, STEPS) as u8
}

impl Sq8Codec {
    /// Fit one `(min, scale)` pair per dimension from `rows` row-major
    /// vectors of length `dim` packed into `matrix` (`matrix.len() == rows *
    /// dim`). `rows` may be 0, in which case every parameter is 0 and every
    /// row encodes to all-zero codes.
    ///
    /// # Errors
    ///
    /// [`ItzmaError::CorruptIndex`] (labelled `"(encoding)"`, matching the
    /// on-disk SQ8 index encoder, which calls this) if `dim` is 0 or does not
    /// fit a `u32`, `matrix.len()` is not a multiple of `dim`, or `matrix`
    /// holds a value that is not finite.
    pub fn fit(matrix: &[f32], dim: usize) -> Result<Self> {
        let bad = |reason: &str| ItzmaError::CorruptIndex {
            path: "(encoding)".to_string(),
            reason: reason.to_string(),
        };
        if dim == 0 || u32::try_from(dim).is_err() {
            return Err(bad("dimension must be between 1 and u32::MAX"));
        }
        if !matrix.len().is_multiple_of(dim) {
            return Err(bad("the matrix is not rows * dim values"));
        }
        if matrix.iter().any(|x| !x.is_finite()) {
            return Err(bad("a vector holds a value that is not finite"));
        }

        let mut min = vec![f32::INFINITY; dim];
        let mut max = vec![f32::NEG_INFINITY; dim];
        let rows = matrix.len() / dim;
        for row in matrix.chunks_exact(dim) {
            for (d, x) in row.iter().enumerate() {
                min[d] = min[d].min(*x);
                max[d] = max[d].max(*x);
            }
        }
        if rows == 0 {
            min.fill(0.0);
            max.fill(0.0);
        }
        let scale = min
            .iter()
            .zip(&max)
            .map(|(lo, hi)| (hi - lo) / STEPS)
            .collect();
        Ok(Sq8Codec { min, scale })
    }

    /// Build directly from already-fit parameters (one per dimension), for
    /// the on-disk index reader, which parses them back out of an object
    /// instead of re-fitting them.
    pub(crate) fn from_parts(min: Vec<f32>, scale: Vec<f32>) -> Self {
        Sq8Codec { min, scale }
    }

    /// The dimension this codec was fit for.
    pub fn dim(&self) -> usize {
        self.min.len()
    }

    /// This codec's per-dimension minimums, for the on-disk encoder.
    pub(crate) fn min(&self) -> &[f32] {
        &self.min
    }

    /// This codec's per-dimension scales, for the on-disk encoder.
    pub(crate) fn scale(&self) -> &[f32] {
        &self.scale
    }

    /// Half a quantization step's worth of L2 error, accumulated over every
    /// dimension: `sqrt(sum((scale_d / 2)^2))` in `f64`, loosened on the
    /// safe side by [`loosen_up`]. A constant of the index alone — every
    /// query against it shares one value.
    pub(crate) fn l2_error_bound(&self) -> f32 {
        let sum: f64 = self
            .scale
            .iter()
            .map(|s| {
                let half = f64::from(*s) / 2.0;
                half * half
            })
            .sum();
        loosen_up(sum.sqrt())
    }

    /// Quantize one row to one code per dimension. The output has
    /// `row.len().min(self.dim())` entries — see the module docs' length
    /// convention.
    pub fn encode(&self, row: &[f32]) -> Vec<u8> {
        let n = self.dim().min(row.len());
        (0..n)
            .map(|d| quantize_one(row[d], self.min[d], self.scale[d]))
            .collect()
    }

    /// Reconstruct a row from its codes, off by at most half a step per
    /// dimension. The output has `code.len().min(self.dim())` entries.
    pub fn decode(&self, code: &[u8]) -> Vec<f32> {
        let n = self.dim().min(code.len());
        (0..n)
            .map(|d| self.min[d] + f32::from(code[d]) * self.scale[d])
            .collect()
    }

    /// Approximate distance from a raw `query` to a stored `code`, under
    /// `metric`. Lower is closer, matching
    /// [`search::distance`](crate::search::distance), whose value this
    /// approximates for the reconstructed vector. Stops at the shortest of
    /// `self.dim()`, `query.len()` and `code.len()`.
    ///
    /// Prepares the query on every call. A caller comparing one query with
    /// many codes should call [`Sq8Codec::prepare`] once instead.
    pub fn distance(&self, metric: Metric, query: &[f32], code: &[u8]) -> f32 {
        self.prepare(metric, query).distance(code)
    }

    /// Prepare `query` for comparing with many codes under `metric`: the
    /// per-dimension terms each metric needs are computed here, once, so
    /// [`Sq8Query::distance`] is one multiply-add per dimension. The query is
    /// cut to `self.dim()` if it is longer.
    pub fn prepare(&self, metric: Metric, query: &[f32]) -> Sq8Query {
        let n = self.dim().min(query.len());
        let query = query[..n].to_vec();
        let min = self.min[..n].to_vec();
        let scale = self.scale[..n].to_vec();
        let (terms, constant) = match metric {
            Metric::L2 => (query.iter().zip(&min).map(|(q, m)| q - m).collect(), 0.0),
            Metric::Dot => (
                query.iter().zip(&scale).map(|(q, s)| q * s).collect(),
                kernels::dot(&query, &min),
            ),
            Metric::Cosine => (Vec::new(), kernels::norm2(&query)),
        };
        let dot_error = loosen_up(
            query
                .iter()
                .zip(&scale)
                .map(|(q, s)| f64::from(q.abs()) * f64::from(*s) / 2.0)
                .sum(),
        );
        Sq8Query {
            metric,
            query,
            min,
            scale,
            terms,
            constant,
            dot_error,
        }
    }
}

/// Loosen an `f64` error bound before it is used to prune: a bound that is
/// too tight could skip a row that belongs in the answer, one that is too
/// loose only costs work. Multiplied up from the exact value and nudged by a
/// small additive margin, mirroring `search::distance`'s own "err on the
/// side of never changing the answer" stance.
fn loosen_up(exact: f64) -> f32 {
    ((exact * (1.0 + 1e-4)) + 1e-6) as f32
}

/// A query prepared against one [`Sq8Codec`] by [`Sq8Codec::prepare`], for
/// comparing with many codes.
#[derive(Debug, Clone)]
pub struct Sq8Query {
    metric: Metric,
    query: Vec<f32>,
    min: Vec<f32>,
    scale: Vec<f32>,
    /// L2: `query - min`. Dot: `query * scale`. Cosine: empty.
    terms: Vec<f32>,
    /// Dot: `sum(query * min)`. Cosine: the query's squared norm. L2: 0.
    constant: f32,
    /// `sum(|query_d| * scale_d / 2)` in `f64`, loosened — the dot product's
    /// worst-case quantization error for this query. Unused for L2
    /// and cosine.
    dot_error: f32,
}

impl Sq8Query {
    /// The dot product's worst-case quantization error for this query,
    /// loosened. Meaningless for a metric other than
    /// [`Metric::Dot`] (`0.0` for L2 and cosine, since `scale` and `query`
    /// are still finite there, just not what the formula below means).
    pub(crate) fn dot_error_bound(&self) -> f32 {
        self.dot_error
    }

    /// Approximate distance to a stored `code`, the same value as
    /// [`Sq8Codec::distance`] with the query this was prepared from. A code
    /// shorter than the query is compared over its own length. A result the
    /// `f32` sums cannot hold (not finite; under cosine, a norm too small or
    /// too large, or a distance below `1e-4`) is recomputed with `f64` sums, as
    /// [`search::distance`](crate::search::distance) does.
    pub fn distance(&self, code: &[u8]) -> f32 {
        let Some(code) = code.get(..self.query.len()) else {
            return self.reference(code);
        };
        let fast = match self.metric {
            Metric::L2 => kernels::sq8_l2(&self.terms, &self.scale, code),
            Metric::Dot => -(self.constant + kernels::sq8_dot(&self.terms, code)),
            Metric::Cosine => {
                let (dot, norm) = kernels::sq8_cosine(&self.query, &self.min, &self.scale, code);
                let q = self.constant;
                let trusted = |n: f32| n.is_finite() && n >= kernels::NORM_FLOOR;
                if !(trusted(q) && trusted(norm)) {
                    return self.reference(code);
                }
                let cosine = 1.0 - dot / (q.sqrt() * norm.sqrt());
                if cosine < kernels::COSINE_RECHECK {
                    return self.reference(code);
                }
                cosine
            }
        };
        if fast.is_finite() {
            fast
        } else {
            self.reference(code)
        }
    }

    /// [`Sq8Query::distance`] with `f64` sums, over the shortest of the query
    /// and `code`.
    fn reference(&self, code: &[u8]) -> f32 {
        let n = self.query.len().min(code.len());
        let value =
            |d: usize| f64::from(self.min[d]) + f64::from(code[d]) * f64::from(self.scale[d]);
        match self.metric {
            Metric::L2 => (0..n)
                .map(|d| {
                    let diff = f64::from(self.query[d]) - value(d);
                    diff * diff
                })
                .sum::<f64>() as f32,
            Metric::Dot => -(0..n)
                .map(|d| f64::from(self.query[d]) * value(d))
                .sum::<f64>() as f32,
            Metric::Cosine => {
                let (mut dot, mut norm, mut qnorm) = (0f64, 0f64, 0f64);
                for d in 0..n {
                    let x = value(d);
                    let q = f64::from(self.query[d]);
                    dot += q * x;
                    norm += x * x;
                    qnorm += q * q;
                }
                if qnorm == 0.0 || norm == 0.0 {
                    return 1.0;
                }
                (1.0 - dot / (qnorm.sqrt() * norm.sqrt())) as f32
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_matrix() -> (usize, Vec<f32>) {
        // dim 2, 3 rows: column 0 spans 0..=10, column 1 is constant at 5.
        let dim = 2;
        let matrix = vec![0.0, 5.0, 5.0, 5.0, 10.0, 5.0];
        (dim, matrix)
    }

    #[test]
    fn fit_refuses_a_matrix_that_is_not_rows_times_dim() {
        let err = Sq8Codec::fit(&[1.0, 2.0, 3.0], 2).unwrap_err();
        assert!(matches!(err, ItzmaError::CorruptIndex { .. }));
    }

    #[test]
    fn fit_refuses_dimension_zero() {
        let err = Sq8Codec::fit(&[1.0, 2.0], 0).unwrap_err();
        assert!(matches!(err, ItzmaError::CorruptIndex { .. }));
    }

    #[test]
    fn fit_refuses_a_non_finite_value() {
        let err = Sq8Codec::fit(&[1.0, f32::NAN], 1).unwrap_err();
        assert!(matches!(err, ItzmaError::CorruptIndex { .. }));
    }

    #[test]
    fn a_round_trip_is_within_half_a_step_and_uses_the_full_range() {
        let (dim, matrix) = tiny_matrix();
        let codec = Sq8Codec::fit(&matrix, dim).unwrap();
        assert_eq!(codec.dim(), dim);
        for row in matrix.chunks_exact(dim) {
            let code = codec.encode(row);
            let back = codec.decode(&code);
            for (x, y) in row.iter().zip(&back) {
                assert!((x - y).abs() <= 10.0 / 255.0 + 1e-6, "{x} vs {y}");
            }
        }
        // The two ends of column 0's range hit code 0 and code 255.
        assert_eq!(codec.encode(&[0.0, 5.0])[0], 0);
        assert_eq!(codec.encode(&[10.0, 5.0])[0], 255);
    }

    #[test]
    fn a_constant_dimension_has_scale_zero_and_decodes_to_its_value() {
        let (dim, matrix) = tiny_matrix();
        let codec = Sq8Codec::fit(&matrix, dim).unwrap();
        // Column 1 is constant at 5.0, so every code for it is 0 and every
        // decode reconstructs exactly 5.0 (scale 0, no rounding error).
        let code = codec.encode(&[3.0, 5.0]);
        assert_eq!(code[1], 0);
        assert_eq!(codec.decode(&code)[1], 5.0);
    }

    #[test]
    fn no_rows_gives_a_zeroed_codec_that_never_panics() {
        let codec = Sq8Codec::fit(&[], 3).unwrap();
        assert_eq!(codec.dim(), 3);
        let code = codec.encode(&[1.0, 2.0, 3.0]);
        assert_eq!(code, vec![0, 0, 0]);
        assert_eq!(codec.decode(&code), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn distance_matches_search_distance_on_the_reconstructed_vector() {
        let (dim, matrix) = tiny_matrix();
        let codec = Sq8Codec::fit(&matrix, dim).unwrap();
        for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
            let query = [7.0, 5.0];
            for row in matrix.chunks_exact(dim) {
                let code = codec.encode(row);
                let reconstructed = codec.decode(&code);
                let want = crate::search::distance(metric, &query, &reconstructed);
                let got = codec.distance(metric, &query, &code);
                assert!((want - got).abs() < 1e-3, "{metric:?}: {want} vs {got}");
            }
        }
    }

    #[test]
    fn a_zero_vector_is_at_cosine_distance_one() {
        let codec = Sq8Codec::fit(&[0.0, 0.0], 2).unwrap();
        let code = codec.encode(&[0.0, 0.0]);
        assert_eq!(codec.distance(Metric::Cosine, &[1.0, 0.0], &code), 1.0);
    }

    #[test]
    fn mismatched_lengths_never_panic() {
        let (dim, matrix) = tiny_matrix();
        let codec = Sq8Codec::fit(&matrix, dim).unwrap();
        // A too-short row: fewer codes come back, no index panic.
        assert_eq!(codec.encode(&[1.0]).len(), 1);
        // A too-long code: decode/distance stop at `dim`, no index panic.
        let long_code = vec![0u8; dim + 5];
        assert_eq!(codec.decode(&long_code).len(), dim);
        let _ = codec.distance(Metric::L2, &[1.0, 2.0, 3.0, 4.0], &long_code);
    }

    /// The pre-kernel `Sq8Codec::distance`, as the reference.
    fn reference(codec: &Sq8Codec, metric: Metric, query: &[f32], code: &[u8]) -> f32 {
        let n = codec.dim().min(query.len()).min(code.len());
        let x: Vec<f32> = codec.decode(&code[..n]);
        crate::search::distance(metric, &query[..n], &x)
    }

    fn lcg_matrix(seed: u64, rows: usize, dim: usize) -> Vec<f32> {
        let mut s = seed;
        (0..rows * dim)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 40) as f32 / (1u32 << 23) as f32 - 1.0
            })
            .collect()
    }

    #[test]
    fn a_prepared_query_matches_the_reference_at_every_dimension() {
        for dim in [1, 3, 7, 8, 9, 25, 128, 960] {
            let matrix = lcg_matrix(dim as u64, 50, dim);
            let codec = Sq8Codec::fit(&matrix, dim).unwrap();
            let query = lcg_matrix(1000 + dim as u64, 1, dim);
            for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
                let prepared = codec.prepare(metric, &query);
                for row in matrix.chunks_exact(dim) {
                    let code = codec.encode(row);
                    let want = reference(&codec, metric, &query, &code);
                    let got = prepared.distance(&code);
                    assert!(
                        (got - want).abs() <= 1e-4 * (1.0 + want.abs()),
                        "{metric:?} at {dim}: {got} vs {want}"
                    );
                    assert_eq!(
                        codec.distance(metric, &query, &code).to_bits(),
                        got.to_bits()
                    );
                }
            }
        }
    }

    #[test]
    fn a_code_shorter_than_the_query_uses_the_shortest_length() {
        let (dim, matrix) = tiny_matrix();
        let codec = Sq8Codec::fit(&matrix, dim).unwrap();
        let code = codec.encode(&matrix[..dim]);
        for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
            let prepared = codec.prepare(metric, &[7.0, 5.0]);
            let want = reference(&codec, metric, &[7.0, 5.0], &code[..1]);
            assert!(
                (prepared.distance(&code[..1]) - want).abs() < 1e-5,
                "{metric:?}"
            );
            assert!(
                prepared.distance(&[]).is_finite(),
                "{metric:?} on an empty code"
            );
        }
    }

    #[test]
    fn extreme_values_give_the_reference_value() {
        // Huge: the reconstruction's squared norm overflows f32.
        let matrix = [1e20f32, 0.0, 1e20, 1e20];
        let codec = Sq8Codec::fit(&matrix, 2).unwrap();
        let code = codec.encode(&[1e20, 0.0]);
        for metric in [Metric::Dot, Metric::Cosine] {
            let want = reference(&codec, metric, &[1e20, 1e20], &code);
            let got = codec.prepare(metric, &[1e20, 1e20]).distance(&code);
            assert!(
                got == want || (got - want).abs() <= 1e-4 * (1.0 + want.abs()),
                "{metric:?}: {got} vs {want}"
            );
        }

        // Tiny: the query's squared norm underflows f32 to zero.
        let matrix = [1.0f32, 1.0, 2.0, 2.0];
        let codec = Sq8Codec::fit(&matrix, 2).unwrap();
        let code = codec.encode(&[2.0, 2.0]);
        let got = codec
            .prepare(Metric::Cosine, &[1e-25, 1e-25])
            .distance(&code);
        assert!(got.abs() < 1e-6, "same direction must be 0, got {got}");
    }

    #[test]
    fn subnormal_scale_and_near_zero_cosine_give_the_reference_value() {
        // Subnormal-scale query: norms nonzero in f32 but with few bits.
        let matrix = [1.0f32, 3.0, 3.0, 1.0];
        let codec = Sq8Codec::fit(&matrix, 2).unwrap();
        let code = codec.encode(&[3.0, 1.0]);
        for s in [1e-22f32, 3e-23, 1e-23] {
            let query = [s, 3.0 * s];
            let want = reference(&codec, Metric::Cosine, &query, &code);
            let got = codec.prepare(Metric::Cosine, &query).distance(&code);
            assert!((got - want).abs() < 1e-5, "s={s}: {got} vs {want}");
        }

        // Near zero: a query equal to a stored row is at distance 0, not a
        // negative rounding residue.
        let matrix = lcg_matrix(9, 20, 128);
        let codec = Sq8Codec::fit(&matrix, 128).unwrap();
        let code = codec.encode(&matrix[..128]);
        let query = codec.decode(&code);
        let got = codec.prepare(Metric::Cosine, &query).distance(&code);
        assert!(got.abs() < 1e-9, "self distance {got}");
    }

    #[test]
    fn l2_error_bound_is_never_above_the_exact_distance_to_the_reconstruction() {
        for dim in [1, 3, 8, 25, 128] {
            let matrix = lcg_matrix(dim as u64 + 7, 40, dim);
            let codec = Sq8Codec::fit(&matrix, dim).unwrap();
            let query = lcg_matrix(dim as u64 + 777, 1, dim);
            let eps = codec.l2_error_bound();
            for row in matrix.chunks_exact(dim) {
                let code = codec.encode(row);
                let reconstructed = codec.decode(&code);
                let exact = crate::search::distance(Metric::L2, &query, &reconstructed);
                let approx = codec.prepare(Metric::L2, &query).distance(&code);
                let lower = (approx.sqrt() - eps).max(0.0).powi(2);
                assert!(
                    lower <= exact + 1e-3,
                    "dim {dim}: lower {lower} above exact {exact}"
                );
            }
        }
    }

    #[test]
    fn dot_error_bound_is_never_above_the_exact_distance_to_the_reconstruction() {
        for dim in [1, 3, 8, 25, 128] {
            let matrix = lcg_matrix(dim as u64 + 11, 40, dim);
            let codec = Sq8Codec::fit(&matrix, dim).unwrap();
            let query = lcg_matrix(dim as u64 + 888, 1, dim);
            let prepared = codec.prepare(Metric::Dot, &query);
            let eps_q = prepared.dot_error_bound();
            for row in matrix.chunks_exact(dim) {
                let code = codec.encode(row);
                let reconstructed = codec.decode(&code);
                let exact = crate::search::distance(Metric::Dot, &query, &reconstructed);
                let approx = prepared.distance(&code);
                let lower = approx - eps_q;
                assert!(
                    lower <= exact + 1e-3,
                    "dim {dim}: lower {lower} above exact {exact}"
                );
            }
        }
    }

    #[test]
    fn a_prepared_query_gives_the_same_bits_every_call() {
        let matrix = lcg_matrix(5, 20, 128);
        let codec = Sq8Codec::fit(&matrix, 128).unwrap();
        let code = codec.encode(&matrix[..128]);
        for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
            let prepared = codec.prepare(metric, &matrix[128..256]);
            let first = prepared.distance(&code).to_bits();
            for _ in 0..10 {
                assert_eq!(prepared.distance(&code).to_bits(), first);
            }
        }
    }
}
