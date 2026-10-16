//! Options across the boundary. Python sends JSON; each struct starts from
//! the core's `default()` and takes only the keys given.

use std::time::Duration;

use itzma_core::{
    CompactionOptions, DEFAULT_RAM_MAX_BYTES, DEFAULT_READ_BUDGET, FetchPolicy, FilterOptions,
    GcOptions, IndexOptions, IndexSparseOptions, MaintenanceOptions, QueryOptions, StreamOptions,
};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value, json};

/// `T::default()` with the keys of `given` (a JSON object) applied.
pub(crate) fn merge<T: Serialize + DeserializeOwned + Default>(given: &str) -> Result<T, String> {
    let given: Map<String, Value> =
        serde_json::from_str(given).map_err(|e| format!("options: {e}"))?;
    let Value::Object(mut base) = serde_json::to_value(T::default()).map_err(|e| e.to_string())?
    else {
        return Err("options: the defaults are not an object".into());
    };
    for (key, value) in given {
        if !base.contains_key(&key) {
            return Err(format!("unknown option {key:?}"));
        }
        base.insert(key, value);
    }
    serde_json::from_value(Value::Object(base)).map_err(|e| format!("options: {e}"))
}

fn object(given: &str) -> Result<Map<String, Value>, String> {
    serde_json::from_str(given).map_err(|e| format!("options: {e}"))
}

fn known(map: &Map<String, Value>, keys: &[&str]) -> Result<(), String> {
    match map.keys().find(|k| !keys.contains(&k.as_str())) {
        Some(k) => Err(format!("unknown option {k:?}")),
        None => Ok(()),
    }
}

fn u64_of(map: &Map<String, Value>, key: &str) -> Result<Option<u64>, String> {
    map.get(key)
        .map(|v| {
            v.as_u64()
                .ok_or_else(|| format!("option {key:?} must be a whole number"))
        })
        .transpose()
}

fn usize_of(map: &Map<String, Value>, key: &str) -> Result<Option<usize>, String> {
    u64_of(map, key)?
        .map(|n| usize::try_from(n).map_err(|e| format!("option {key:?}: {e}")))
        .transpose()
}

fn bool_of(map: &Map<String, Value>, key: &str) -> Result<Option<bool>, String> {
    map.get(key)
        .map(|v| {
            v.as_bool()
                .ok_or_else(|| format!("option {key:?} must be true or false"))
        })
        .transpose()
}

fn seconds(v: &Value, key: &str) -> Result<Duration, String> {
    let secs = v
        .as_f64()
        .ok_or_else(|| format!("option {key:?} must be a number of seconds"))?;
    Duration::try_from_secs_f64(secs)
        .map_err(|_| format!("option {key:?} must be 0 or more seconds"))
}

fn compaction_from(map: &Map<String, Value>) -> Result<CompactionOptions, String> {
    let mut o = CompactionOptions::default();
    if let Some(n) = usize_of(map, "max_segments")? {
        o.max_segments = n;
    }
    if let Some(n) = u64_of(map, "max_segment_bytes")? {
        o.max_segment_bytes = n;
    }
    Ok(o)
}

pub(crate) fn compaction(given: &str) -> Result<CompactionOptions, String> {
    let map = object(given)?;
    known(&map, &["max_segments", "max_segment_bytes"])?;
    compaction_from(&map)
}

fn gc_from(map: &Map<String, Value>) -> Result<GcOptions, String> {
    known(map, &["min_age_secs", "dry_run", "fence_wal"])?;
    let mut o = GcOptions::default();
    if let Some(v) = map.get("min_age_secs") {
        o.min_age = seconds(v, "min_age_secs")?;
    }
    if let Some(b) = bool_of(map, "dry_run")? {
        o.dry_run = b;
    }
    if let Some(b) = bool_of(map, "fence_wal")? {
        o.fence_wal = b;
    }
    Ok(o)
}

pub(crate) fn gc(given: &str) -> Result<GcOptions, String> {
    gc_from(&object(given)?)
}

pub(crate) fn maintenance(given: &str) -> Result<MaintenanceOptions, String> {
    let map = object(given)?;
    known(
        &map,
        &[
            "tail_objects",
            "tail_bytes",
            "tail_age_secs",
            "max_segments",
            "max_segment_bytes",
            "gc",
        ],
    )?;
    let mut o = MaintenanceOptions::default();
    if let Some(n) = usize_of(&map, "tail_objects")? {
        o.compact_when.tail_objects = n;
    }
    if let Some(n) = u64_of(&map, "tail_bytes")? {
        o.compact_when.tail_bytes = n;
    }
    if let Some(v) = map.get("tail_age_secs") {
        o.compact_when.tail_age = if v.is_null() {
            None
        } else {
            Some(seconds(v, "tail_age_secs")?)
        };
    }
    o.compaction = compaction_from(&map)?;
    if let Some(v) = map.get("gc") {
        o.gc = match v {
            Value::Null => None,
            Value::Object(gc_map) => Some(gc_from(gc_map)?),
            _ => return Err("option \"gc\" must be null or an object".into()),
        };
    }
    Ok(o)
}

fn compaction_json(o: &CompactionOptions) -> Value {
    json!({"max_segments": o.max_segments, "max_segment_bytes": o.max_segment_bytes})
}

fn gc_json(o: &GcOptions) -> Value {
    json!({"min_age_secs": o.min_age.as_secs_f64(), "dry_run": o.dry_run, "fence_wal": o.fence_wal})
}

fn fetch_json(p: &FetchPolicy) -> Value {
    json!({
        "min_gap": p.min_gap,
        "max_gap": p.max_gap,
        "gap_divisor": p.gap_divisor,
        "split_above": p.split_above,
        "part_size": p.part_size,
        "align": p.align,
    })
}

/// Every default the Python signatures must repeat.
pub(crate) fn defaults() -> Value {
    let m = MaintenanceOptions::default();
    json!({
        "query": serde_json::to_value(QueryOptions::default()).unwrap_or_default(),
        "filter": serde_json::to_value(FilterOptions::default()).unwrap_or_default(),
        "index": serde_json::to_value(IndexOptions::default()).unwrap_or_default(),
        "index_sparse": serde_json::to_value(IndexSparseOptions::default()).unwrap_or_default(),
        "compaction": compaction_json(&CompactionOptions::default()),
        "gc": gc_json(&GcOptions::default()),
        "maintain": {
            "tail_objects": m.compact_when.tail_objects,
            "tail_bytes": m.compact_when.tail_bytes,
            "tail_age_secs": m.compact_when.tail_age.map(|d| d.as_secs_f64()),
            "max_segments": m.compaction.max_segments,
            "max_segment_bytes": m.compaction.max_segment_bytes,
            "gc": m.gc.as_ref().map(gc_json),
        },
        "cache": {"ram_bytes": DEFAULT_RAM_MAX_BYTES},
        "read_budget": DEFAULT_READ_BUDGET,
        "stream": {
            "target_bytes": StreamOptions::default().target_bytes,
            "depth": StreamOptions::default().depth,
        },
        "fetch": fetch_json(&FetchPolicy::object_store()),
        "fetch_local": fetch_json(&FetchPolicy::local()),
    })
}

/// The core's defaults as JSON; the Python tests compare signatures to them.
#[pyo3::pyfunction]
pub(crate) fn default_options() -> String {
    defaults().to_string()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use itzma_core::{CompactionOptions, GcOptions, MaintenanceOptions, QueryOptions};

    use super::*;

    #[test]
    fn empty_is_the_default() {
        assert_eq!(
            merge::<QueryOptions>("{}").expect("merge"),
            QueryOptions::default()
        );
    }

    #[test]
    fn given_keys_override() {
        let o: QueryOptions = merge(r#"{"rerank": null, "exact": true}"#).expect("merge");
        assert_eq!(o, QueryOptions::default().rerank(None).exact(true));
    }

    #[test]
    fn unknown_keys_are_refused() {
        let err = merge::<QueryOptions>(r#"{"rerankk": 2}"#).expect_err("unknown");
        assert!(err.contains("rerankk"), "{err}");
    }

    #[test]
    fn wrong_types_are_refused() {
        assert!(merge::<QueryOptions>(r#"{"exact": "yes"}"#).is_err());
    }

    #[test]
    fn non_serde_defaults() {
        assert_eq!(compaction("{}").expect("c"), CompactionOptions::default());
        assert_eq!(gc("{}").expect("g"), GcOptions::default());
        assert_eq!(maintenance("{}").expect("m"), MaintenanceOptions::default());
    }

    #[test]
    fn non_serde_overrides() {
        let c = compaction(r#"{"max_segments": 1, "max_segment_bytes": 10}"#).expect("c");
        assert_eq!((c.max_segments, c.max_segment_bytes), (1, 10));
        let g = gc(r#"{"min_age_secs": 1.5, "dry_run": true, "fence_wal": true}"#).expect("g");
        assert_eq!(g.min_age, Duration::from_millis(1500));
        assert!(g.dry_run && g.fence_wal);
        let m = maintenance(r#"{"tail_objects": 1, "tail_age_secs": 2.0, "gc": null}"#).expect("m");
        assert_eq!(m.compact_when.tail_objects, 1);
        assert_eq!(m.compact_when.tail_age, Some(Duration::from_secs(2)));
        assert_eq!(m.gc, None);
        let m = maintenance(r#"{"gc": {"min_age_secs": 0}}"#).expect("m");
        assert_eq!(m.gc.expect("gc").min_age, Duration::ZERO);
    }

    #[test]
    fn non_serde_refusals() {
        assert!(
            compaction(r#"{"max_segmentz": 1}"#)
                .expect_err("unknown")
                .contains("max_segmentz")
        );
        assert!(gc(r#"{"min_age_secs": -1}"#).is_err());
        assert!(gc(r#"{"dry_run": 1}"#).is_err());
        assert!(maintenance(r#"{"gc": 3}"#).is_err());
    }

    #[test]
    fn defaults_round_trip() {
        let d = defaults();
        assert_eq!(
            merge::<QueryOptions>(&d["query"].to_string()).expect("q"),
            QueryOptions::default()
        );
        assert_eq!(
            compaction(&d["compaction"].to_string()).expect("c"),
            CompactionOptions::default()
        );
        assert_eq!(gc(&d["gc"].to_string()).expect("g"), GcOptions::default());
        assert_eq!(
            maintenance(&d["maintain"].to_string()).expect("m"),
            MaintenanceOptions::default()
        );
        assert_eq!(
            d["cache"]["ram_bytes"],
            serde_json::json!(itzma_core::DEFAULT_RAM_MAX_BYTES)
        );
    }
}
