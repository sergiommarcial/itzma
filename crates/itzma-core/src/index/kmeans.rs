//! Seeded k-means for the partitions of an SQ8 index.
//!
//! An index object is never compared byte for byte, so clustering
//! does not have to match across machines. It is deterministic for one input
//! anyway, which keeps the tests exact: a small fixed generator, no `rand`.

use crate::document::Metric;
use crate::kernels::l2;

/// Training uses at most this many sampled rows per centroid.
const SAMPLE_PER_CENTROID: usize = 32;
/// Lloyd iterations over the sample. Fixed and small: the partitions only have
/// to be roughly balanced and near, and the rerank makes the answer exact.
const ITERATIONS: usize = 5;
/// Seeding looks at at most this many pooled rows per centroid.
const INIT_POOL_PER_CENTROID: usize = 16;
const SEED: u64 = 0x9e37_79b9_7f4a_7c15;
/// Rows per chunk of an assignment pass: a constant, never the
/// thread count, so the sums and the clustering are the same on every machine.
pub(crate) const ASSIGN_CHUNK: usize = 4096;

/// One chunk's share of a Lloyd iteration.
struct Partial {
    sums: Vec<f64>,
    counts: Vec<usize>,
    distances: Vec<f32>,
}

/// The centroids and the cluster of every row.
pub(crate) struct Clustering {
    /// `k * dim` values, cluster `c` at `c * dim..(c + 1) * dim`.
    pub(crate) centroids: Vec<f32>,
    /// For each row, the cluster it belongs to, below `k`.
    pub(crate) assignment: Vec<u32>,
}

/// xorshift64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// A value below `n`, `n > 0`.
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// The index in `distances` that a roulette-wheel draw of `target` (in
/// `[0, total)`, `total` the sum of `distances`) lands on: walk in order,
/// subtracting each distance, and stop where the remainder goes to zero or
/// below. Floating rounding can leave a small positive remainder after summing
/// every distance; the fallback for that is the last entry with a nonzero
/// distance, never one already picked (whose distance to the nearest chosen
/// centroid is 0), so a rounding error never hands back a duplicate. If every
/// distance is 0 (every row is already a centroid), it falls back to index 0.
fn roulette_pick(distances: &[f32], target: f64) -> usize {
    let mut remaining = target;
    let mut last_nonzero = None;
    for (j, d) in distances.iter().enumerate() {
        if *d > 0.0 {
            last_nonzero = Some(j);
        }
        remaining -= f64::from(*d);
        if remaining <= 0.0 {
            return j;
        }
    }
    last_nonzero.unwrap_or(0)
}

/// Give each empty cluster in `counts` a distinct centroid: the sample rows
/// farthest from what they were assigned to, farthest first, so two empty
/// clusters never end up with the same reseeded centroid. A cluster with no
/// entry in `counts` at zero is left alone. `distances` is the distance from
/// sample row `sample[i]` to the centroid `nearest` gave it, in `sample`'s
/// order.
fn reseed_empty_clusters(
    centroids: &mut [f32],
    data: &[f32],
    dim: usize,
    counts: &[usize],
    sample: &[usize],
    distances: &[f32],
) {
    let empties: Vec<usize> = (0..counts.len()).filter(|&c| counts[c] == 0).collect();
    if empties.is_empty() {
        return;
    }
    let mut order: Vec<usize> = (0..sample.len()).collect();
    order.sort_by(|&a, &b| distances[b].total_cmp(&distances[a]));
    for (slot, &c) in empties.iter().enumerate() {
        // `sample.len() >= k` always (see `cluster`), so `slot` never runs past
        // `order`.
        let i = sample[order[slot]];
        centroids[c * dim..(c + 1) * dim].copy_from_slice(&data[i * dim..(i + 1) * dim]);
    }
}

/// The nearest centroid of `row` and its squared distance, first on ties.
fn nearest(centroids: &[f32], dim: usize, row: &[f32]) -> (usize, f32) {
    let mut best = (0, f32::INFINITY);
    for (c, centroid) in centroids.chunks_exact(dim).enumerate() {
        let d = l2(row, centroid);
        if d < best.1 {
            best = (c, d);
        }
    }
    best
}

/// `matrix`, with each row scaled to unit length under cosine (a zero row stays
/// zero), so L2 clustering groups by direction.
fn prepared(matrix: &[f32], dim: usize, metric: Metric) -> Vec<f32> {
    let mut out = matrix.to_vec();
    if metric == Metric::Cosine {
        for row in out.chunks_exact_mut(dim) {
            let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                row.iter_mut().for_each(|x| *x /= norm);
            }
        }
    }
    out
}

/// Cluster the `rows` rows of `matrix` (row-major, `dim` columns) into `k`
/// groups. `k` is clamped to `1..=rows` (to 1 for no rows). L2 and dot cluster
/// on the vectors as they are, cosine on their directions. No cluster is left
/// empty when there are at least `k` rows.
pub(crate) fn cluster(
    matrix: &[f32],
    dim: usize,
    rows: usize,
    k: usize,
    metric: Metric,
) -> Clustering {
    let k = k.clamp(1, rows.max(1));
    if rows == 0 {
        return Clustering {
            centroids: vec![0.0; dim],
            assignment: Vec::new(),
        };
    }
    let data = prepared(matrix, dim, metric);
    let row = |i: usize| &data[i * dim..(i + 1) * dim];

    // A stride over the rows, at most SAMPLE_PER_CENTROID per centroid.
    let sample_len = (k * SAMPLE_PER_CENTROID).min(rows);
    let sample: Vec<usize> = (0..sample_len).map(|i| i * rows / sample_len).collect();

    // Initial centroids by k-means++ over a bounded pool of the sample: each new
    // centroid is drawn with probability proportional to its squared distance
    // from the nearest one chosen so far, so the starts are spread out. The pool
    // keeps the seeding at `k * pool * dim` work however big the segment is.
    let pool_len = (k * INIT_POOL_PER_CENTROID).min(sample_len);
    let pool: Vec<usize> = (0..pool_len)
        .map(|i| sample[i * sample_len / pool_len])
        .collect();
    let mut rng = Rng(SEED);
    let mut centroids = Vec::with_capacity(k * dim);
    let first = pool[rng.below(pool_len)];
    centroids.extend_from_slice(row(first));
    let mut nearest_sq: Vec<f32> = pool.iter().map(|&i| l2(row(i), row(first))).collect();
    for _ in 1..k {
        let total: f64 = nearest_sq.iter().map(|d| f64::from(*d)).sum();
        let pick = if total > 0.0 {
            let target = (rng.next() as f64 / u64::MAX as f64) * total;
            roulette_pick(&nearest_sq, target)
        } else {
            // Every pooled row is already a centroid: any row will do.
            rng.below(pool_len)
        };
        let picked = row(pool[pick]);
        centroids.extend_from_slice(picked);
        for (d, &i) in nearest_sq.iter_mut().zip(&pool) {
            *d = d.min(l2(row(i), picked));
        }
    }

    for _ in 0..ITERATIONS {
        let partials = crate::cpu::chunked(&sample, ASSIGN_CHUNK, |chunk| {
            let mut p = Partial {
                sums: vec![0f64; k * dim],
                counts: vec![0usize; k],
                distances: Vec::with_capacity(chunk.len()),
            };
            for &i in chunk {
                let (c, d) = nearest(&centroids, dim, row(i));
                p.counts[c] += 1;
                for (s, x) in p.sums[c * dim..(c + 1) * dim].iter_mut().zip(row(i)) {
                    *s += f64::from(*x);
                }
                p.distances.push(d);
            }
            p
        });
        // Added in chunk order, never completion order, so the centroids are
        // the same however many threads ran the chunks.
        let mut sums = vec![0f64; k * dim];
        let mut counts = vec![0usize; k];
        let mut distances = Vec::with_capacity(sample.len());
        for p in partials {
            sums.iter_mut().zip(&p.sums).for_each(|(s, x)| *s += x);
            counts.iter_mut().zip(&p.counts).for_each(|(c, x)| *c += x);
            distances.extend(p.distances);
        }
        reseed_empty_clusters(&mut centroids, &data, dim, &counts, &sample, &distances);
        for c in 0..k {
            if counts[c] == 0 {
                continue;
            }
            let target = &mut centroids[c * dim..(c + 1) * dim];
            for (t, s) in target.iter_mut().zip(&sums[c * dim..(c + 1) * dim]) {
                *t = (*s / counts[c] as f64) as f32;
            }
        }
    }

    // Every row goes to its nearest centroid once. A cluster that ended empty
    // takes the row farthest from its centroid among the clusters with more than
    // one row, so with at least `k` rows none stays empty.
    let every: Vec<usize> = (0..rows).collect();
    let nearest_of: Vec<Vec<(usize, f32)>> = crate::cpu::chunked(&every, ASSIGN_CHUNK, |chunk| {
        chunk
            .iter()
            .map(|&i| nearest(&centroids, dim, row(i)))
            .collect()
    });
    let mut assignment = Vec::with_capacity(rows);
    let mut distance = Vec::with_capacity(rows);
    let mut counts = vec![0usize; k];
    for (c, d) in nearest_of.into_iter().flatten() {
        // Rows fit a u32: the index format numbers them that way.
        assignment.push(c as u32);
        distance.push(d);
        counts[c] += 1;
    }
    for c in 0..k {
        if counts[c] > 0 {
            continue;
        }
        let donor = (0..rows)
            .filter(|i| counts[assignment[*i] as usize] > 1)
            .max_by(|a, b| distance[*a].total_cmp(&distance[*b]).then(b.cmp(a)));
        let Some(i) = donor else { break };
        counts[assignment[i] as usize] -= 1;
        assignment[i] = c as u32;
        counts[c] += 1;
        distance[i] = 0.0;
        centroids[c * dim..(c + 1) * dim].copy_from_slice(row(i));
    }
    Clustering {
        centroids,
        assignment,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 200 clusters of a 30,000-row matrix: the sample (6,400 rows) and the
    /// final assignment (8 chunks) both span several chunks.
    #[test]
    fn clustering_is_the_same_on_one_two_and_eight_threads() {
        let (rows, dim, k) = (30_000usize, 16usize, 200usize);
        let mut x = 0x2545_f491_4f6c_dd1du64;
        let matrix: Vec<f32> = (0..rows * dim)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                // Exponents spread over 2^-20..2^20, so the f64 sums are inexact.
                ((x >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2f32.powi((x % 41) as i32 - 20)
            })
            .collect();
        let on = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|i| format!("itzma-maint-{i}"))
                .build()
                .unwrap()
                .install(|| cluster(&matrix, dim, rows, k, Metric::L2))
        };
        let one = on(1);
        for threads in [2, 8] {
            let other = on(threads);
            assert_eq!(other.centroids, one.centroids, "{threads} threads");
            assert_eq!(other.assignment, one.assignment, "{threads} threads");
        }
        let outside = cluster(&matrix, dim, rows, k, Metric::L2);
        assert_eq!(outside.centroids, one.centroids);
        assert_eq!(outside.assignment, one.assignment);
    }

    #[test]
    fn roulette_pick_walks_the_wheel_normally() {
        let distances = [5.0f32, 3.0, 2.0];
        assert_eq!(roulette_pick(&distances, 0.5), 0);
        assert_eq!(roulette_pick(&distances, 5.5), 1);
        assert_eq!(roulette_pick(&distances, 9.5), 2);
    }

    #[test]
    fn roulette_pick_never_lands_on_an_already_picked_row_when_rounding_overshoots() {
        // Rows 1 and 3 are already centroids (distance 0 from the nearest one
        // chosen so far). A target past the true total (5 + 0 + 3 + 0 = 8) is
        // what floating rounding can leave behind after the real walk.
        let distances = [5.0f32, 0.0, 3.0, 0.0];
        let picked = roulette_pick(&distances, 100.0);
        assert_eq!(
            picked, 2,
            "the last row with a nonzero distance, not index 3 (already a centroid)"
        );
    }

    #[test]
    fn roulette_pick_falls_back_to_zero_when_every_row_is_already_a_centroid() {
        assert_eq!(roulette_pick(&[0.0, 0.0, 0.0], 5.0), 0);
    }

    #[test]
    fn reseeding_gives_each_empty_cluster_a_different_row() {
        let (dim, k) = (1usize, 3usize);
        let mut centroids = vec![0.0f32; k * dim];
        let data = [10.0f32, 20.0, 30.0, 40.0];
        let sample = [0usize, 1, 2, 3];
        // Row 2 (value 30) is farthest, row 1 (value 20) is next, row 3 nearest.
        let distances = [1.0f32, 5.0, 9.0, 3.0];
        let counts = [0usize, 1, 0];
        reseed_empty_clusters(&mut centroids, &data, dim, &counts, &sample, &distances);
        assert_eq!(
            centroids[0], 30.0,
            "first empty cluster gets the farthest row"
        );
        assert_eq!(
            centroids[2], 20.0,
            "second empty cluster gets the next farthest, not a repeat of the first"
        );
    }

    #[test]
    fn reseeding_does_nothing_when_no_cluster_is_empty() {
        let mut centroids = vec![7.0f32, 8.0];
        reseed_empty_clusters(
            &mut centroids,
            &[1.0, 2.0],
            1,
            &[3, 4],
            &[0, 1],
            &[0.1, 0.2],
        );
        assert_eq!(centroids, [7.0, 8.0]);
    }

    /// Three tight groups far apart in two dimensions, `per` rows each.
    fn three_groups(per: usize) -> Vec<f32> {
        let mut m = Vec::new();
        for (cx, cy) in [(0.0f32, 0.0f32), (100.0, 0.0), (0.0, 100.0)] {
            for i in 0..per {
                m.push(cx + (i % 3) as f32 * 0.01);
                m.push(cy + (i % 5) as f32 * 0.01);
            }
        }
        m
    }

    #[test]
    fn it_separates_well_spread_groups() {
        let m = three_groups(30);
        let c = cluster(&m, 2, 90, 3, Metric::L2);
        assert_eq!(c.centroids.len(), 3 * 2);
        assert_eq!(c.assignment.len(), 90);
        let of = |g: usize| c.assignment[g * 30];
        for g in 0..3 {
            assert!(
                c.assignment[g * 30..(g + 1) * 30]
                    .iter()
                    .all(|a| *a == of(g))
            );
        }
        assert_ne!(of(0), of(1));
        assert_ne!(of(0), of(2));
        assert_ne!(of(1), of(2));
    }

    #[test]
    fn the_same_input_gives_the_same_clustering() {
        let m = three_groups(20);
        let a = cluster(&m, 2, 60, 3, Metric::L2);
        let b = cluster(&m, 2, 60, 3, Metric::L2);
        assert_eq!(a.centroids, b.centroids);
        assert_eq!(a.assignment, b.assignment);
    }

    #[test]
    fn every_cluster_is_used_and_every_assignment_is_in_range() {
        let m = three_groups(10);
        let c = cluster(&m, 2, 30, 3, Metric::L2);
        for k in 0..3u32 {
            assert!(c.assignment.contains(&k), "cluster {k} is empty");
        }
        assert!(c.assignment.iter().all(|a| *a < 3));
    }

    #[test]
    fn k_is_clamped_to_the_rows_and_to_at_least_one() {
        let m = three_groups(2); // 6 rows
        assert_eq!(cluster(&m, 2, 6, 100, Metric::L2).centroids.len(), 6 * 2);
        assert_eq!(cluster(&m, 2, 6, 0, Metric::L2).centroids.len(), 2);
    }

    #[test]
    fn no_rows_and_one_row_do_not_panic() {
        let none = cluster(&[], 3, 0, 4, Metric::L2);
        assert_eq!(none.assignment.len(), 0);
        assert_eq!(none.centroids.len(), 3);
        let one = cluster(&[1.0, 2.0, 3.0], 3, 1, 4, Metric::L2);
        assert_eq!(one.assignment, [0]);
        assert_eq!(one.centroids, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn identical_rows_do_not_loop_or_panic() {
        let m = vec![5.0f32; 2 * 20];
        let c = cluster(&m, 2, 20, 4, Metric::L2);
        assert_eq!(c.assignment.len(), 20);
        assert!(
            c.assignment
                .iter()
                .all(|a| (*a as usize) < c.centroids.len() / 2)
        );
    }

    #[test]
    fn cosine_groups_by_direction_and_ignores_length() {
        // Two directions, each at very different lengths.
        let mut m = Vec::new();
        for len in [1.0f32, 10.0, 100.0] {
            m.extend([len, 0.0]);
            m.extend([0.0, len]);
        }
        let c = cluster(&m, 2, 6, 2, Metric::Cosine);
        assert_eq!(c.assignment[0], c.assignment[2]);
        assert_eq!(c.assignment[0], c.assignment[4]);
        assert_eq!(c.assignment[1], c.assignment[3]);
        assert_ne!(c.assignment[0], c.assignment[1]);
    }
}
