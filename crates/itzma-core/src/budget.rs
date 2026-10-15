//! The read budget: a cap on how many bytes one query or one compaction run
//! may read.
//!
//! Every fetch of the run charges the same [`ReadBudget`] before it reads a
//! body. A run that would go over stops with
//! [`ItzmaError::ReadBudgetExceeded`] instead of growing until the process is
//! killed. The count is stored bytes, a stand-in for memory: it does not see
//! the extra weight of attributes once they are decoded.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{ItzmaError, Result};

/// The default read budget of a [`Namespace`](crate::Namespace): 1 GiB.
pub const DEFAULT_READ_BUDGET: u64 = 1 << 30;

/// A running total of bytes read, with a ceiling.
pub(crate) struct ReadBudget {
    max: u64,
    used: AtomicU64,
}

impl ReadBudget {
    /// A budget of `max` bytes with nothing used.
    pub(crate) fn new(max: u64) -> Self {
        Self {
            max,
            used: AtomicU64::new(0),
        }
    }

    /// A budget that never refuses, for callers that want the whole log.
    pub(crate) fn unlimited() -> Self {
        Self::new(u64::MAX)
    }

    /// Bytes charged so far.
    #[cfg(test)]
    pub(crate) fn used(&self) -> u64 {
        self.used.load(Ordering::Relaxed)
    }

    /// Atomically reserve `bytes` if the total still fits, without the
    /// "stays counted" stickiness [`charge`](Self::charge) has: on
    /// failure, the budget is left exactly as it was, so a caller can try
    /// a smaller reservation afterward instead of being permanently
    /// blocked by a bigger first attempt. This is what lets the fetch
    /// planner's budget fallback decide between a merged and an
    /// unmerged plan atomically: deciding from a snapshot of the room left
    /// and charging later would leave a window where a concurrent caller
    /// sharing the same budget charges in between, so the snapshot the
    /// decision was made on is already stale by the time it is acted on.
    /// There is no matching `release`: every caller either keeps a
    /// reservation it won or never won one to begin with, so nothing here
    /// ever needs giving back (YAGNI — add the pair back if a caller
    /// actually needs it).
    pub(crate) fn try_reserve(&self, bytes: u64) -> bool {
        let mut before = self.used.load(Ordering::Relaxed);
        loop {
            let after = before.saturating_add(bytes);
            if after > self.max {
                return false;
            }
            match self.used.compare_exchange_weak(
                before,
                after,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(actual) => before = actual,
            }
        }
    }

    /// Add `bytes` for the object at `path`.
    ///
    /// # Errors
    ///
    /// [`ItzmaError::ReadBudgetExceeded`] if the total would pass the ceiling.
    /// The bytes stay counted, so every later charge fails too.
    pub(crate) fn charge(&self, path: &str, bytes: u64) -> Result<()> {
        // A manual compare-exchange loop, not `fetch_update` (deprecated in
        // favor of `try_update`, which this crate's MSRV of 1.87 does not
        // have yet): `compare_exchange_weak` has been stable since 1.34, so
        // this works on every supported toolchain without picking a side.
        let mut before = self.used.load(Ordering::Relaxed);
        loop {
            let after = before.saturating_add(bytes);
            match self.used.compare_exchange_weak(
                before,
                after,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => before = actual,
            }
        }
        let needed = before.saturating_add(bytes);
        if needed > self.max {
            return Err(ItzmaError::ReadBudgetExceeded {
                path: path.to_string(),
                needed,
                max: self.max,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    fn over(err: ItzmaError) -> (String, u64, u64) {
        match err {
            ItzmaError::ReadBudgetExceeded { path, needed, max } => (path, needed, max),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn the_default_is_one_gib() {
        assert_eq!(DEFAULT_READ_BUDGET, 1024 * 1024 * 1024);
    }

    #[test]
    fn charges_add_up_and_the_total_may_reach_the_ceiling() {
        let budget = ReadBudget::new(10);
        budget.charge("a", 4).unwrap();
        budget.charge("b", 6).unwrap();
    }

    #[test]
    fn one_byte_over_fails_and_names_the_object_and_the_total() {
        let budget = ReadBudget::new(10);
        budget.charge("a", 4).unwrap();
        let (path, needed, max) = over(budget.charge("b", 7).unwrap_err());
        assert_eq!((path.as_str(), needed, max), ("b", 11, 10));
    }

    #[test]
    fn a_refused_charge_stays_counted() {
        let budget = ReadBudget::new(10);
        budget.charge("a", 11).unwrap_err();
        let (_, needed, _) = over(budget.charge("b", 0).unwrap_err());
        assert_eq!(needed, 11);
    }

    #[test]
    fn a_budget_of_zero_allows_nothing_but_nothing() {
        let budget = ReadBudget::new(0);
        budget.charge("a", 0).unwrap();
        budget.charge("a", 1).unwrap_err();
    }

    #[test]
    fn unlimited_never_refuses_and_does_not_wrap() {
        let budget = ReadBudget::unlimited();
        budget.charge("a", u64::MAX).unwrap();
        budget.charge("b", u64::MAX).unwrap();
    }

    #[test]
    fn try_reserve_does_not_stick_on_failure_unlike_charge() {
        let budget = ReadBudget::new(10);
        assert!(!budget.try_reserve(11), "too big to reserve");
        // Unlike `charge`, a failed `try_reserve` leaves the budget exactly
        // as it was — a caller that tried a merged plan and lost can still
        // try a smaller unmerged one afterward.
        assert_eq!(budget.used(), 0);
        assert!(budget.try_reserve(10), "now it fits exactly");
        assert_eq!(budget.used(), 10);
    }

    #[test]
    fn concurrent_try_reserve_never_lets_the_total_exceed_max() {
        // The scenario Finding 3 was about: two sources sharing one budget
        // each see "room" and both try to reserve a merged (expensive)
        // plan at once. At most one of them may win when only one plan's
        // worth of room exists.
        let budget = Arc::new(ReadBudget::new(100));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let budget = Arc::clone(&budget);
                std::thread::spawn(move || budget.try_reserve(60))
            })
            .collect();
        let wins: usize = threads
            .into_iter()
            .map(|t| t.join().unwrap())
            .filter(|&won| won)
            .count();
        assert_eq!(
            wins, 1,
            "only one 60-byte reservation fits in a 100-byte budget"
        );
        assert_eq!(budget.used(), 60);
    }

    #[test]
    fn charges_from_many_threads_add_up() {
        let budget = Arc::new(ReadBudget::new(1000));
        let threads: Vec<_> = (0..10)
            .map(|_| {
                let budget = Arc::clone(&budget);
                std::thread::spawn(move || {
                    for _ in 0..10 {
                        budget.charge("a", 10).unwrap();
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        budget.charge("a", 1).unwrap_err();
    }
}
