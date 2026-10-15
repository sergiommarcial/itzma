//! Distance arithmetic, written so the compiler turns it into vector
//! instructions with no `unsafe` and no nightly features.
//!
//! Every loop keeps a fixed number of independent `f32` accumulators over
//! `chunks_exact`, adds the scalar tail, and reduces the lanes in one fixed
//! order, so a call is deterministic: the same inputs give the same bits.
//! Slices of different lengths are cut to the shortest, never a panic.
//! Callers that promise `f64`-grade behavior on extreme values (overflow,
//! underflow) check the result and fall back to their own reference.
//!
//! The lane counts were measured, not derived: whether LLVM vectorizes a loop
//! well depends on its shape. The two SQ8 loops that reconstruct a row
//! (`sq8_l2`, `sq8_cosine`) run about four times faster with [`WIDE_LANES`]
//! than with [`LANES`], because the wider chunk lets the `u8` to `f32`
//! conversion vectorize; the others are no faster wider, and `sq8_dot` is
//! slower. Recheck with `make bench`'s `kernels`
//! group after changing a loop or the toolchain.

/// Independent accumulators for most loops. Eight `f32`s fill one 256-bit
/// register or two 128-bit ones, and break the chain of dependent adds a single
/// sum has.
const LANES: usize = 8;

/// Independent accumulators for the SQ8 loops that reconstruct a row.
const WIDE_LANES: usize = 32;

/// The smallest squared norm a cosine trusts from `f32` sums. Below it the
/// terms are subnormal (`f32::MIN_POSITIVE` is about `1.2e-38`) and lose their
/// bits, so callers recompute with `f64`. With 24 bits of headroom above the
/// subnormal range, what is lost is under one part in ten million.
pub(crate) const NORM_FLOOR: f32 = 1e-30;

/// A cosine distance from `f32` sums below this is recomputed with `f64` by
/// callers: `1 - cos` near zero is a difference of two numbers close to one,
/// which `f32` resolves only to about `1e-7`. Only near-duplicates get here,
/// so it costs almost nothing.
pub(crate) const COSINE_RECHECK: f32 = 1e-4;

/// The lanes' sum, in a fixed order: fold the upper half onto the lower half
/// until eight remain, then add those pairwise. `N` is a power of two, at
/// least eight.
fn reduce<const N: usize>(mut lanes: [f32; N]) -> f32 {
    let mut width = N;
    while width > 8 {
        width /= 2;
        for i in 0..width {
            lanes[i] += lanes[i + width];
        }
    }
    ((lanes[0] + lanes[4]) + (lanes[1] + lanes[5]))
        + ((lanes[2] + lanes[6]) + (lanes[3] + lanes[7]))
}

/// `sum(a_i * b_i)`.
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let (a_chunks, b_chunks) = (a.chunks_exact(LANES), b.chunks_exact(LANES));
    let tail: f32 = a_chunks
        .remainder()
        .iter()
        .zip(b_chunks.remainder())
        .map(|(x, y)| x * y)
        .sum();
    let mut lanes = [0f32; LANES];
    for (x, y) in a_chunks.zip(b_chunks) {
        for i in 0..LANES {
            lanes[i] += x[i] * y[i];
        }
    }
    reduce(lanes) + tail
}

/// `sum((a_i - b_i)^2)`.
pub(crate) fn l2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let (a_chunks, b_chunks) = (a.chunks_exact(LANES), b.chunks_exact(LANES));
    let tail: f32 = a_chunks
        .remainder()
        .iter()
        .zip(b_chunks.remainder())
        .map(|(x, y)| (x - y) * (x - y))
        .sum();
    let mut lanes = [0f32; LANES];
    for (x, y) in a_chunks.zip(b_chunks) {
        for i in 0..LANES {
            let d = x[i] - y[i];
            lanes[i] += d * d;
        }
    }
    reduce(lanes) + tail
}

/// `sum(a_i^2)`.
pub(crate) fn norm2(a: &[f32]) -> f32 {
    dot(a, a)
}

/// `sum((shifted_d - code_d * scale_d)^2)`: L2 to an SQ8 row, from a query
/// already shifted by the per-dimension minimums.
pub(crate) fn sq8_l2(shifted: &[f32], scale: &[f32], code: &[u8]) -> f32 {
    let n = shifted.len().min(scale.len()).min(code.len());
    let (shifted, scale, code) = (&shifted[..n], &scale[..n], &code[..n]);
    let (q_chunks, s_chunks, c_chunks) = (
        shifted.chunks_exact(WIDE_LANES),
        scale.chunks_exact(WIDE_LANES),
        code.chunks_exact(WIDE_LANES),
    );
    let tail: f32 = q_chunks
        .remainder()
        .iter()
        .zip(s_chunks.remainder())
        .zip(c_chunks.remainder())
        .map(|((q, s), c)| {
            let d = q - f32::from(*c) * s;
            d * d
        })
        .sum();
    let mut lanes = [0f32; WIDE_LANES];
    for ((q, s), c) in q_chunks.zip(s_chunks).zip(c_chunks) {
        for i in 0..WIDE_LANES {
            let d = q[i] - f32::from(c[i]) * s[i];
            lanes[i] += d * d;
        }
    }
    reduce(lanes) + tail
}

/// `sum(weights_d * code_d)`.
pub(crate) fn sq8_dot(weights: &[f32], code: &[u8]) -> f32 {
    let n = weights.len().min(code.len());
    let (weights, code) = (&weights[..n], &code[..n]);
    let (w_chunks, c_chunks) = (weights.chunks_exact(LANES), code.chunks_exact(LANES));
    let tail: f32 = w_chunks
        .remainder()
        .iter()
        .zip(c_chunks.remainder())
        .map(|(w, c)| w * f32::from(*c))
        .sum();
    let mut lanes = [0f32; LANES];
    for (w, c) in w_chunks.zip(c_chunks) {
        for i in 0..LANES {
            lanes[i] += w[i] * f32::from(c[i]);
        }
    }
    reduce(lanes) + tail
}

/// `(sum(query_d * x_d), sum(x_d^2))` with `x_d = min_d + code_d * scale_d`,
/// the row reconstructed in the loop so nothing cancels.
pub(crate) fn sq8_cosine(query: &[f32], min: &[f32], scale: &[f32], code: &[u8]) -> (f32, f32) {
    let n = query.len().min(min.len()).min(scale.len()).min(code.len());
    let (query, min, scale, code) = (&query[..n], &min[..n], &scale[..n], &code[..n]);
    let (q_chunks, m_chunks, s_chunks, c_chunks) = (
        query.chunks_exact(WIDE_LANES),
        min.chunks_exact(WIDE_LANES),
        scale.chunks_exact(WIDE_LANES),
        code.chunks_exact(WIDE_LANES),
    );
    let (mut tail_dot, mut tail_norm) = (0f32, 0f32);
    for (((q, m), s), c) in q_chunks
        .remainder()
        .iter()
        .zip(m_chunks.remainder())
        .zip(s_chunks.remainder())
        .zip(c_chunks.remainder())
    {
        let x = m + f32::from(*c) * s;
        tail_dot += q * x;
        tail_norm += x * x;
    }
    let (mut dots, mut norms) = ([0f32; WIDE_LANES], [0f32; WIDE_LANES]);
    for (((q, m), s), c) in q_chunks.zip(m_chunks).zip(s_chunks).zip(c_chunks) {
        for i in 0..WIDE_LANES {
            let x = m[i] + f32::from(c[i]) * s[i];
            dots[i] += q[i] * x;
            norms[i] += x * x;
        }
    }
    (reduce(dots) + tail_dot, reduce(norms) + tail_norm)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// xorshift64, values in [-1, 1).
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / (1u32 << 23) as f32 - 1.0
        }

        fn vector(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.next()).collect()
        }

        fn code(&mut self, n: usize) -> Vec<u8> {
            (0..n)
                .map(|_| ((self.next() + 1.0) * 127.5) as u8)
                .collect()
        }
    }

    const DIMS: [usize; 18] = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 25, 31, 32, 33, 63, 65, 128, 960,
    ];

    /// `got` is `want` up to `f32` rounding of a sum whose terms add up to
    /// `scale` in absolute value.
    fn close(got: f32, want: f64, scale: f64) -> bool {
        (f64::from(got) - want).abs() <= 1e-5 * (1.0 + scale)
    }

    #[test]
    fn dot_l2_and_norm2_match_f64_at_every_dimension() {
        let mut rng = Rng(1);
        for dim in DIMS {
            let (a, b) = (rng.vector(dim), rng.vector(dim));
            let pairs = || {
                a.iter()
                    .zip(&b)
                    .map(|(x, y)| (f64::from(*x), f64::from(*y)))
            };
            let want_dot: f64 = pairs().map(|(x, y)| x * y).sum();
            let dot_scale: f64 = pairs().map(|(x, y)| (x * y).abs()).sum();
            let want_l2: f64 = pairs().map(|(x, y)| (x - y) * (x - y)).sum();
            let want_norm: f64 = a.iter().map(|x| f64::from(*x) * f64::from(*x)).sum();
            assert!(close(dot(&a, &b), want_dot, dot_scale), "dot at {dim}");
            assert!(close(l2(&a, &b), want_l2, want_l2), "l2 at {dim}");
            assert!(close(norm2(&a), want_norm, want_norm), "norm2 at {dim}");
        }
    }

    #[test]
    fn sq8_loops_match_f64_at_every_dimension() {
        let mut rng = Rng(2);
        for dim in DIMS {
            let (q, min) = (rng.vector(dim), rng.vector(dim));
            let scale: Vec<f32> = rng.vector(dim).iter().map(|s| s.abs() / 100.0).collect();
            let code = rng.code(dim);
            let x: Vec<f64> = (0..dim)
                .map(|d| f64::from(min[d]) + f64::from(code[d]) * f64::from(scale[d]))
                .collect();
            let shifted: Vec<f32> = q.iter().zip(&min).map(|(q, m)| q - m).collect();
            let want_l2: f64 = (0..dim).map(|d| (f64::from(q[d]) - x[d]).powi(2)).sum();
            assert!(
                close(sq8_l2(&shifted, &scale, &code), want_l2, want_l2),
                "sq8_l2 at {dim}"
            );

            let weights: Vec<f32> = q.iter().zip(&scale).map(|(q, s)| q * s).collect();
            let want_dot: f64 = (0..dim)
                .map(|d| f64::from(weights[d]) * f64::from(code[d]))
                .sum();
            let dot_scale: f64 = (0..dim)
                .map(|d| (f64::from(weights[d]) * f64::from(code[d])).abs())
                .sum();
            assert!(
                close(sq8_dot(&weights, &code), want_dot, dot_scale),
                "sq8_dot at {dim}"
            );

            let (got_dot, got_norm) = sq8_cosine(&q, &min, &scale, &code);
            let want_qx: f64 = (0..dim).map(|d| f64::from(q[d]) * x[d]).sum();
            let qx_scale: f64 = (0..dim).map(|d| (f64::from(q[d]) * x[d]).abs()).sum();
            let want_xx: f64 = x.iter().map(|v| v * v).sum();
            assert!(close(got_dot, want_qx, qx_scale), "sq8_cosine dot at {dim}");
            assert!(
                close(got_norm, want_xx, want_xx),
                "sq8_cosine norm at {dim}"
            );
        }
    }

    #[test]
    fn the_tail_is_counted_once() {
        // Nine ones: eight in the lanes, one in the tail.
        let ones = [1.0f32; 9];
        assert_eq!(dot(&ones, &ones), 9.0);
        assert_eq!(norm2(&ones), 9.0);
        assert_eq!(l2(&ones, &[0.0; 9]), 9.0);
        assert_eq!(sq8_dot(&ones, &[1u8; 9]), 9.0);
        assert_eq!(sq8_l2(&ones, &[0.0; 9], &[0u8; 9]), 9.0);
        assert_eq!(sq8_cosine(&ones, &ones, &[0.0; 9], &[0u8; 9]), (9.0, 9.0));

        // Thirty-three: one full chunk of the widest lanes, one in the tail.
        let ones = [1.0f32; 33];
        assert_eq!(sq8_l2(&ones, &[0.0; 33], &[0u8; 33]), 33.0);
        assert_eq!(
            sq8_cosine(&ones, &ones, &[0.0; 33], &[0u8; 33]),
            (33.0, 33.0)
        );
        assert_eq!(sq8_dot(&ones, &[1u8; 33]), 33.0);
    }

    #[test]
    fn mismatched_lengths_use_the_shortest_and_never_panic() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [1.0f32, 1.0];
        assert_eq!(dot(&a, &b), 3.0);
        assert_eq!(l2(&a, &b), 1.0);
        assert_eq!(dot(&[], &a), 0.0);
        assert_eq!(norm2(&[]), 0.0);
        assert_eq!(sq8_l2(&a, &[1.0], &[1, 1, 1]), 0.0);
        assert_eq!(sq8_dot(&a, &[]), 0.0);
        assert_eq!(sq8_cosine(&a, &a, &a, &[0]), (1.0, 1.0));
    }

    #[test]
    fn the_same_inputs_give_the_same_bits() {
        let mut rng = Rng(3);
        let (a, b) = (rng.vector(960), rng.vector(960));
        let code = rng.code(960);
        let run = || {
            [
                dot(&a, &b),
                l2(&a, &b),
                sq8_l2(&a, &b, &code),
                sq8_dot(&a, &code),
                sq8_cosine(&a, &b, &a, &code).0,
            ]
            .map(f32::to_bits)
        };
        let first = run();
        for _ in 0..10 {
            assert_eq!(run(), first);
        }
    }
}
