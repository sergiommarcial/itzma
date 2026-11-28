//! The order a top-level `And` runs its conjuncts in.
//!
//! A scan keeps, per conjunct, the rows it was evaluated on and the rows it
//! kept, and runs the conjuncts from most to least rejecting, each on the
//! rows the earlier ones kept. The order is planned again after every block.
//! `And` commutes, so the order changes the work and never the answer.

use std::sync::Mutex;

use super::Predicate;
use crate::columnar::Bitmap;
use crate::segment::ColumnBlock;

/// What one conjunct has done so far.
#[derive(Debug, Clone, Copy, Default)]
struct Tally {
    seen: u64,
    kept: u64,
}

impl Tally {
    /// The range its selectivity (the share of rows it keeps) can lie in,
    /// given how few rows it was measured on. A conjunct never measured can
    /// lie anywhere.
    fn range(self) -> (f64, f64) {
        if self.seen == 0 {
            return (0.0, 1.0);
        }
        let seen = self.seen as f64;
        let share = self.kept as f64 / seen;
        let slack = seen.sqrt().recip();
        ((share - slack).max(0.0), (share + slack).min(1.0))
    }
}

/// The measured selectivity of each conjunct of one `And`, shared by every
/// block of a scan.
#[derive(Debug)]
pub(crate) struct Passes {
    tallies: Vec<Tally>,
}

impl Passes {
    /// A planner for `conjuncts` conjuncts, none measured yet.
    pub(crate) fn new(conjuncts: usize) -> Self {
        Self {
            tallies: vec![Tally::default(); conjuncts],
        }
    }

    /// Conjunct positions, most rejecting first. One moves before another
    /// only when its whole range of possible selectivity is below the
    /// other's, so a conjunct measured on a few rows keeps its place and the
    /// order is stable until the numbers say otherwise.
    pub(crate) fn order(&self) -> Vec<usize> {
        let mut order: Vec<usize> = Vec::with_capacity(self.tallies.len());
        for at in 0..self.tallies.len() {
            let (_, high) = self.tallies[at].range();
            let mut place = order.len();
            while place > 0 && high < self.tallies[order[place - 1]].range().0 {
                place -= 1;
            }
            order.insert(place, at);
        }
        order
    }

    /// Conjunct `at` kept `kept` of the `seen` rows it was given.
    pub(crate) fn record(&mut self, at: usize, seen: usize, kept: usize) {
        if let Some(tally) = self.tallies.get_mut(at) {
            tally.seen += seen as u64;
            tally.kept += kept as u64;
        }
    }

    #[cfg(test)]
    fn seen(&self, at: usize) -> u64 {
        self.tallies[at].seen
    }
}

impl Predicate {
    /// A planner sized for this predicate: one conjunct per item of a
    /// top-level `And`, none for any other predicate.
    pub(crate) fn passes(&self) -> Mutex<Passes> {
        Mutex::new(Passes::new(match self {
            Self::And(items) => items.len(),
            _ => 0,
        }))
    }

    /// [`on_columns`](Self::on_columns) with a top-level `And` run in the
    /// order `passes` plans, and what each conjunct did recorded back. The
    /// rows are the same for every order.
    pub(crate) fn on_columns_planned(
        &self,
        block: &ColumnBlock,
        within: &Bitmap,
        passes: &Mutex<Passes>,
    ) -> Bitmap {
        let Self::And(items) = self else {
            return self.on_columns(block, within);
        };
        let order = lock(passes).order();
        let mut rows = within.clone();
        let mut seen = Vec::with_capacity(order.len());
        for at in order {
            if rows.is_empty() {
                break;
            }
            let before = rows.count();
            rows = items[at].on_columns(block, &rows);
            seen.push((at, before, rows.count()));
        }
        let mut passes = lock(passes);
        for (at, before, kept) in seen {
            passes.record(at, before, kept);
        }
        rows
    }
}

/// The planner's state holds only counts, so a panic elsewhere cannot leave
/// it wrong, only stale.
fn lock(passes: &Mutex<Passes>) -> std::sync::MutexGuard<'_, Passes> {
    passes
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::super::block::tests::{Rng, predicate, row, schema, words};
    use super::*;
    use crate::predicate::Compare;
    use crate::segment::fields_of;

    fn permutations(n: usize) -> Vec<Vec<usize>> {
        fn go(rest: &mut Vec<usize>, chosen: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
            if rest.is_empty() {
                out.push(chosen.clone());
                return;
            }
            for i in 0..rest.len() {
                let item = rest.remove(i);
                chosen.push(item);
                go(rest, chosen, out);
                chosen.pop();
                rest.insert(i, item);
            }
        }
        let mut out = Vec::new();
        go(&mut (0..n).collect(), &mut Vec::new(), &mut out);
        out
    }

    /// Fed synthetic selectivities, the order settles on most rejecting first.
    #[test]
    fn the_order_converges_to_most_rejecting_first() {
        let mut passes = Passes::new(3);
        assert_eq!(
            passes.order(),
            vec![0, 1, 2],
            "nothing measured keeps order"
        );
        for _ in 0..20 {
            passes.record(0, 100, 90);
            passes.record(1, 100, 2);
            passes.record(2, 100, 50);
        }
        assert_eq!(passes.order(), vec![1, 2, 0]);
    }

    /// A conjunct measured on a few rows keeps its place, and one never
    /// measured is never moved past or before.
    #[test]
    fn a_conjunct_measured_on_few_rows_keeps_its_place() {
        let mut passes = Passes::new(3);
        passes.record(0, 4, 3);
        passes.record(1, 4, 1);
        assert_eq!(passes.order(), vec![0, 1, 2], "ranges overlap");
        passes.record(1, 10_000, 100);
        passes.record(0, 10_000, 9_000);
        assert_eq!(passes.order(), vec![1, 0, 2], "2 was never measured");
    }

    /// An early conjunct that empties the set stops the rest: they are never
    /// evaluated, so nothing is recorded for them.
    #[test]
    fn an_empty_set_stops_the_rest() {
        let fields = fields_of(&schema());
        let rows: Vec<Option<Value>> = (0..64)
            .map(|i| Some(serde_json::json!({ "n": i, "t": "x" })))
            .collect();
        let block = ColumnBlock::from_rows(&fields, &rows).unwrap();
        let none = Predicate::number("n", Compare::Gt, 1000.0);
        let some = Predicate::number("n", Compare::Ge, 0.0);
        let p = Predicate::And(vec![none, some]);
        let passes = p.passes();
        let hits = p.on_columns_planned(&block, &Bitmap::full(64), &passes);
        assert!(hits.is_empty());
        let passes = lock(&passes);
        assert_eq!(passes.seen(0), 64);
        assert_eq!(passes.seen(1), 0, "the second conjunct never ran");
    }

    /// Every order of the conjuncts, and the planned order, give the rows
    /// `matches` gives, whatever the planner has measured.
    #[test]
    fn the_result_is_independent_of_the_order() {
        let words = words();
        let fields = fields_of(&schema());
        let mut rng = Rng(23);
        for case in 0..400 {
            let n = rng.below(40);
            let rows: Vec<Option<Value>> = (0..n).map(|_| row(&mut rng, &words)).collect();
            let block = ColumnBlock::from_rows(&fields, &rows).unwrap();
            let items: Vec<Predicate> = (0..4).map(|_| predicate(&mut rng, &words, 1)).collect();
            let all = Bitmap::full(n);
            let want = |p: &Predicate| -> Vec<bool> {
                rows.iter().map(|r| p.matches(r.as_ref())).collect()
            };
            let truth = want(&Predicate::And(items.clone()));
            let planned = Predicate::And(items.clone());
            let passes = planned.passes();
            for round in 0..3 {
                let got = planned.on_columns_planned(&block, &all, &passes);
                let got: Vec<bool> = (0..n).map(|i| got.get(i)).collect();
                assert_eq!(got, truth, "case {case} planned round {round}");
            }
            for order in permutations(items.len()) {
                let permuted = Predicate::And(order.iter().map(|&i| items[i].clone()).collect());
                let got = permuted.on_columns(&block, &all);
                let got: Vec<bool> = (0..n).map(|i| got.get(i)).collect();
                assert_eq!(got, truth, "case {case} order {order:?}");
            }
        }
    }
}
