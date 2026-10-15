//! Maintenance: decide whether a compaction is due, run it, then collect.
//!
//! [`Namespace::maintain`](crate::Namespace::maintain) is one call that a
//! caller schedules. It looks at the WAL tail, compacts if a threshold is
//! crossed, runs [`gc`](crate::Namespace::gc) if asked, and reports what it did.
//! The library starts no task of its own. This module holds the options, the
//! report and the decision, which needs no storage and is tested on its own.

use std::time::Duration;

use crate::compaction::{Compaction, CompactionOptions};
use crate::gc::{GcOptions, GcReport, WalObject};

/// The default for [`CompactTrigger::tail_objects`].
pub const DEFAULT_TAIL_OBJECTS: usize = 64;

/// The default for [`CompactTrigger::tail_bytes`]: 64 MiB.
pub const DEFAULT_TAIL_BYTES: u64 = 64 * 1024 * 1024;

/// When the WAL tail is long enough that it should be folded into a segment.
///
/// A query reads every tail object, one GET each, so the count is latency and
/// the bytes are the read budget. Any trigger that is crossed makes a
/// compaction due, as long as the tail is not empty. Turn one off with
/// `usize::MAX`, `u64::MAX` or `None`. Zero means "any tail at all".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactTrigger {
    /// Due once the tail has this many objects. Default
    /// [`DEFAULT_TAIL_OBJECTS`].
    pub tail_objects: usize,
    /// Due once the tail is this many stored bytes. Default
    /// [`DEFAULT_TAIL_BYTES`].
    pub tail_bytes: u64,
    /// Due once the oldest tail object is this old. Off by default, since a
    /// namespace with a slow trickle of writes would otherwise be merged over
    /// and over, and a merge rewrites every segment. It is what makes
    /// a small, slow namespace fold at all, and only folded objects can be
    /// fenced by [`gc`](crate::Namespace::gc). The age is the store's write
    /// time against this machine's clock.
    pub tail_age: Option<Duration>,
}

impl Default for CompactTrigger {
    fn default() -> Self {
        Self {
            tail_objects: DEFAULT_TAIL_OBJECTS,
            tail_bytes: DEFAULT_TAIL_BYTES,
            tail_age: None,
        }
    }
}

/// Settings for [`Namespace::maintain`](crate::Namespace::maintain).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaintenanceOptions {
    /// When a compaction is due.
    pub compact_when: CompactTrigger,
    /// What the compaction does when it runs. A namespace with more live
    /// segments than [`CompactionOptions::max_segments`] also makes one due, so
    /// the merge is not left waiting for a tail.
    pub compaction: CompactionOptions,
    /// The collection to run after the compaction, or `None` to run none. It
    /// runs even when no compaction was due. Fencing is a per-call flag in
    /// here ([`GcOptions::fence_wal`]), not a property of the namespace.
    pub gc: Option<GcOptions>,
}

impl Default for MaintenanceOptions {
    /// The default trigger and compaction, and a default collection, which
    /// deletes dead segments a day old and leaves the WAL alone.
    fn default() -> Self {
        Self {
            compact_when: CompactTrigger::default(),
            compaction: CompactionOptions::default(),
            gc: Some(GcOptions::default()),
        }
    }
}

/// What a call to [`Namespace::maintain`](crate::Namespace::maintain) saw and
/// did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaintenanceReport {
    /// WAL objects past the fold point when the call looked.
    pub tail_objects: usize,
    /// Their stored size in bytes.
    pub tail_bytes: u64,
    /// What the compaction did, or `None` if none was due. `Some(Superseded)`
    /// means another process compacted first, which is fine.
    pub compaction: Option<Compaction>,
    /// What the collection did, or `None` if none was asked for.
    pub gc: Option<GcReport>,
}

/// What the listing of the tail shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Tail {
    pub(crate) objects: usize,
    pub(crate) bytes: u64,
    /// Write time of the oldest object in unix milliseconds, if there is one.
    pub(crate) oldest_ms: Option<i64>,
}

impl Tail {
    pub(crate) fn of(objects: &[WalObject]) -> Self {
        Self {
            objects: objects.len(),
            bytes: objects.iter().map(|o| o.size).sum(),
            oldest_ms: objects.iter().map(|o| o.modified_ms).min(),
        }
    }
}

/// Whether a compaction is due. Pure: no storage and no clock, `now_ms` is
/// passed in.
///
/// An empty tail is never due by itself, whatever the thresholds say. Too many
/// live segments is due even then, since a merge is what brings them down.
pub(crate) fn due(
    trigger: &CompactTrigger,
    tail: Tail,
    live_segments: usize,
    max_segments: usize,
    now_ms: i64,
) -> bool {
    if live_segments > max_segments.max(1) {
        return true;
    }
    if tail.objects == 0 {
        return false;
    }
    let old_enough = match (trigger.tail_age, tail.oldest_ms) {
        (Some(age), Some(oldest)) => {
            let age_ms = i64::try_from(age.as_millis()).unwrap_or(i64::MAX);
            now_ms.saturating_sub(oldest).max(0) >= age_ms
        }
        _ => false,
    };
    tail.objects >= trigger.tail_objects || tail.bytes >= trigger.tail_bytes || old_enough
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: i64 = 1_000_000;

    fn tail(objects: usize, bytes: u64, oldest_ms: i64) -> Tail {
        Tail {
            objects,
            bytes,
            oldest_ms: (objects > 0).then_some(oldest_ms),
        }
    }

    fn only(f: impl FnOnce(&mut CompactTrigger)) -> CompactTrigger {
        let mut t = CompactTrigger {
            tail_objects: usize::MAX,
            tail_bytes: u64::MAX,
            tail_age: None,
        };
        f(&mut t);
        t
    }

    #[test]
    fn the_defaults_are_the_documented_ones() {
        let t = CompactTrigger::default();
        assert_eq!(t.tail_objects, 64);
        assert_eq!(t.tail_bytes, 64 * 1024 * 1024);
        assert_eq!(t.tail_age, None);
        let o = MaintenanceOptions::default();
        assert_eq!(o.compaction, CompactionOptions::default());
        assert_eq!(o.gc, Some(GcOptions::default()));
    }

    #[test]
    fn an_empty_tail_is_never_due_even_at_zero() {
        let zero = CompactTrigger {
            tail_objects: 0,
            tail_bytes: 0,
            tail_age: Some(Duration::ZERO),
        };
        assert!(!due(&zero, tail(0, 0, 0), 1, 8, NOW));
    }

    #[test]
    fn each_count_fires_at_its_edge_and_not_below() {
        let objects = only(|t| t.tail_objects = 3);
        assert!(!due(&objects, tail(2, 0, NOW), 1, 8, NOW));
        assert!(due(&objects, tail(3, 0, NOW), 1, 8, NOW));
        let bytes = only(|t| t.tail_bytes = 500);
        assert!(!due(&bytes, tail(1, 499, NOW), 1, 8, NOW));
        assert!(due(&bytes, tail(1, 500, NOW), 1, 8, NOW));
    }

    #[test]
    fn the_age_is_measured_from_the_oldest_object() {
        let age = only(|t| t.tail_age = Some(Duration::from_secs(60)));
        assert!(!due(&age, tail(2, 0, NOW - 59_999), 1, 8, NOW));
        assert!(due(&age, tail(2, 0, NOW - 60_000), 1, 8, NOW));
        // A write time in the future counts as age zero.
        assert!(!due(&age, tail(2, 0, NOW + 10_000), 1, 8, NOW));
    }

    #[test]
    fn a_trigger_that_is_off_never_fires() {
        let off = only(|_| {});
        assert!(!due(&off, tail(1_000_000, u64::MAX - 1, 0), 1, 8, NOW));
    }

    #[test]
    fn too_many_segments_is_due_with_no_tail_at_all() {
        let off = only(|_| {});
        assert!(due(&off, tail(0, 0, 0), 9, 8, NOW));
        assert!(!due(&off, tail(0, 0, 0), 8, 8, NOW));
        // Zero is treated as one, as in compaction.
        assert!(due(&off, tail(0, 0, 0), 2, 0, NOW));
        assert!(!due(&off, tail(0, 0, 0), 1, 0, NOW));
    }
}
