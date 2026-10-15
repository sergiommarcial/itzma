//! Core reports as JSON, for the Python dataclasses. `IndexReport` is serde
//! already; these three are not.

use itzma_core::{Compaction, GcReport, MaintenanceReport};
use serde_json::{Value, json};

pub(crate) fn compaction(c: &Compaction) -> Value {
    match c {
        Compaction::Nothing => json!({"kind": "nothing"}),
        Compaction::Superseded => json!({"kind": "superseded"}),
        Compaction::Published {
            segments,
            folded_through,
            merged_segments,
            docs,
            tombstones,
        } => json!({
            "kind": "published",
            "segments": segments,
            "folded_through": folded_through,
            "merged_segments": merged_segments,
            "docs": docs,
            "tombstones": tombstones,
        }),
    }
}

pub(crate) fn gc(r: &GcReport) -> Value {
    json!({
        "segments_deleted": r.segments_deleted,
        "bytes_freed": r.bytes_freed,
        "segments_too_young": r.segments_too_young,
        "wal_objects_fenced": r.wal_objects_fenced,
        "wal_bytes_freed": r.wal_bytes_freed,
        "wal_too_young": r.wal_too_young,
        "indexes_deleted": r.indexes_deleted,
        "indexes_too_young": r.indexes_too_young,
    })
}

pub(crate) fn maintenance(r: &MaintenanceReport) -> Value {
    json!({
        "tail_objects": r.tail_objects,
        "tail_bytes": r.tail_bytes,
        "compaction": r.compaction.as_ref().map(compaction),
        "gc": r.gc.as_ref().map(gc),
    })
}

#[cfg(test)]
mod tests {
    use itzma_core::{Compaction, GcReport, MaintenanceReport};
    use serde_json::json;

    use super::*;

    #[test]
    fn each_compaction_shape() {
        assert_eq!(compaction(&Compaction::Nothing), json!({"kind": "nothing"}));
        assert_eq!(
            compaction(&Compaction::Superseded),
            json!({"kind": "superseded"})
        );
        let published = Compaction::Published {
            segments: vec!["s".into()],
            folded_through: 3,
            merged_segments: 1,
            docs: 10,
            tombstones: 0,
        };
        assert_eq!(
            compaction(&published),
            json!({"kind": "published", "segments": ["s"], "folded_through": 3,
                   "merged_segments": 1, "docs": 10, "tombstones": 0})
        );
    }

    #[test]
    fn maintenance_nests() {
        let report = MaintenanceReport {
            tail_objects: 2,
            tail_bytes: 100,
            compaction: Some(Compaction::Nothing),
            gc: Some(GcReport::default()),
        };
        let v = maintenance(&report);
        assert_eq!(v["compaction"]["kind"], "nothing");
        assert_eq!(v["gc"]["segments_deleted"], 0);
    }
}
