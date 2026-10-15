//! A shared bound on the k-th best live, matching, exact distance any source
//! of one query attempt has found so far. Every source checks it
//! before doing the expensive parts of evaluating a candidate (the shadow
//! check, fetching attributes, fetching the exact vector): a candidate whose
//! lower bound on its exact distance is strictly worse than the bound cannot
//! end up in the final top-k, whichever source it came from.
//!
//! The bound only ever tightens (goes down), and only from exact distances of
//! candidates that already passed the shadow check and the predicate — a
//! candidate later discovered to be shadowed or non-matching never feeds it,
//! so it never prunes a row that belongs in the answer.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};

/// Max-heap entry so the worst of the best `top_k` sits on top, evictable in
/// O(log top_k).
struct Ranked(f32);

impl PartialEq for Ranked {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0) == Ordering::Equal
    }
}
impl Eq for Ranked {}
impl PartialOrd for Ranked {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Ranked {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// The k-th best live distance across every source of one query attempt.
/// `top_k` distinct documents are never offered by more than one source
/// (each live id belongs to exactly one source, the newest that holds it),
/// so this is a true upper bound on the final answer's k-th distance.
pub(crate) struct Bound {
    /// `f32::to_bits` of the current bound. `f32::INFINITY`'s bits until
    /// `top_k` offers have landed. Read with a relaxed load on the hot path;
    /// only ever lowered, under the lock, before this is updated.
    bits: AtomicU32,
    best: Mutex<BinaryHeap<Ranked>>,
    top_k: usize,
}

impl Bound {
    /// A fresh bound for a query that wants `top_k` hits. Unbounded
    /// (`f32::INFINITY`) until `top_k` live distances are known.
    pub(crate) fn new(top_k: usize) -> Self {
        Bound {
            bits: AtomicU32::new(f32::INFINITY.to_bits()),
            best: Mutex::new(BinaryHeap::with_capacity(top_k.min(1024) + 1)),
            top_k,
        }
    }

    /// The current bound. Non-negative for L2 and cosine, possibly negative
    /// for dot (distances are negated there).
    pub(crate) fn get(&self) -> f32 {
        f32::from_bits(self.bits.load(AtomicOrdering::Relaxed))
    }

    /// Record exact distances of candidates that already passed the shadow
    /// check and the predicate. Safe to call from many sources at once; the
    /// heap is behind a lock since this runs once per batch, not once per
    /// row — the hot per-row read is `get`, which takes no lock.
    pub(crate) fn offer(&self, live: &[f32]) {
        if self.top_k == 0 || live.is_empty() {
            return;
        }
        let mut best = self.best.lock().unwrap_or_else(|e| e.into_inner());
        for &d in live {
            if best.len() < self.top_k {
                best.push(Ranked(d));
            } else if best.peek().is_some_and(|worst| d < worst.0) {
                best.pop();
                best.push(Ranked(d));
            }
        }
        if best.len() >= self.top_k {
            if let Some(worst) = best.peek() {
                self.bits.store(worst.0.to_bits(), AtomicOrdering::Relaxed);
            }
        }
    }

    /// Whether `lower`, a lower bound on some candidate's exact distance, is
    /// strictly worse than the current bound — safe to drop that candidate
    /// without changing the answer. Strict, since a tie could still win on
    /// id (the merge's tie-break).
    pub(crate) fn beats(&self, lower: f32) -> bool {
        lower > self.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn starts_at_infinity_and_beats_nothing() {
        let bound = Bound::new(3);
        assert_eq!(bound.get(), f32::INFINITY);
        assert!(!bound.beats(f32::MAX));
        assert!(!bound.beats(f32::INFINITY)); // strict: infinity never beats infinity
    }

    #[test]
    fn stays_at_infinity_below_top_k_offers() {
        let bound = Bound::new(3);
        bound.offer(&[1.0, 2.0]);
        assert_eq!(bound.get(), f32::INFINITY, "only 2 of 3 needed offered");
    }

    #[test]
    fn tightens_to_the_kth_best_once_top_k_are_known() {
        let bound = Bound::new(2);
        bound.offer(&[5.0, 1.0, 9.0]);
        assert_eq!(bound.get(), 5.0, "2nd best of {{1,5,9}} is 5");
    }

    #[test]
    fn never_rises_across_offers() {
        let bound = Bound::new(2);
        bound.offer(&[5.0, 1.0]);
        let after_first = bound.get();
        bound.offer(&[100.0]); // worse than both kept, must not loosen anything
        assert_eq!(bound.get(), after_first);
        bound.offer(&[0.5]); // better than the worst kept, must tighten
        assert!(bound.get() < after_first);
    }

    #[test]
    fn beats_is_strict_so_ties_are_kept() {
        let bound = Bound::new(1);
        bound.offer(&[3.0]);
        assert_eq!(bound.get(), 3.0);
        assert!(
            !bound.beats(3.0),
            "equal must not be dropped: ties break by id"
        );
        assert!(bound.beats(3.0001));
    }

    #[test]
    fn top_k_zero_never_tightens() {
        let bound = Bound::new(0);
        bound.offer(&[1.0, 2.0]);
        assert_eq!(bound.get(), f32::INFINITY);
    }

    #[test]
    fn concurrent_offers_from_many_tasks_end_at_the_sequential_answer() {
        let bound = Arc::new(Bound::new(5));
        let mut handles = Vec::new();
        for chunk in (0..100).collect::<Vec<i32>>().chunks(10) {
            let bound = Arc::clone(&bound);
            let values: Vec<f32> = chunk.iter().map(|x| *x as f32).collect();
            handles.push(std::thread::spawn(move || bound.offer(&values)));
        }
        for h in handles {
            h.join().unwrap();
        }
        // The 5th best of 0..100 is 4.0.
        assert_eq!(bound.get(), 4.0);
    }
}
