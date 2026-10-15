//! WAL fences (phase B), from the side of everyone who is not the
//! collector: readers, compactors and writers that run into an object that
//! has been replaced by a fence. Fences are made by hand here. The collector
//! that makes them has its own tests below.

mod common;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use common::faulty::{Fault, FaultyStore};
use common::{
    Backend, FENCE, Snapshot, assert_consistent, assert_untouched, backends, cas_backends, doc,
    fence_by_hand, put_segment, schema, snapshot,
};
use itzma_core::gc::fenced_through_path;
use itzma_core::wal::{self, FENCE_FORMAT_VERSION, FORMAT_VERSION};
use itzma_core::{
    Compaction, CompactionOptions, GcOptions, GcReport, Hit, ItzmaError, Manifest, Metric,
    Namespace, WalOp, WriteKey, manifest,
};
use object_store::{ObjectStore, ObjectStoreExt};

const ROOMY: CompactionOptions = CompactionOptions {
    max_segments: 8,
    max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
};
const TIGHT: CompactionOptions = CompactionOptions {
    max_segments: 1,
    max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
};

/// Collect anything, however young, without touching the WAL.
const NOW: GcOptions = GcOptions {
    min_age: Duration::ZERO,
    dry_run: false,
    fence_wal: false,
};

/// Collect anything, however young, and fence the folded WAL objects.
const FENCING: GcOptions = GcOptions {
    fence_wal: true,
    ..NOW
};

fn ids(hits: &[Hit]) -> Vec<&str> {
    hits.iter().map(|h| h.id.as_str()).collect()
}

/// The key of the WAL object at `seq`, as a needle for `before_get`.
fn wal_key(seq: u64) -> String {
    format!("{seq:020}.json")
}

async fn three_writes(store: Arc<dyn ObjectStore>) -> Namespace {
    let ns = Namespace::create(store, "docs", schema(1, Metric::L2))
        .await
        .unwrap();
    for (id, x) in [("a", 1.0), ("b", 2.0), ("c", 3.0)] {
        ns.upsert(vec![doc(id, &[x])]).await.unwrap();
    }
    ns
}

/// What another process does between a reader's manifest read and its first
/// WAL read: fold everything into a segment, then fence the folded objects,
/// as a compaction followed by a fencing collection would.
async fn fold_and_fence(store: Arc<dyn ObjectStore>, through: u64) {
    let ns = Namespace::open(store.clone(), "docs").await.unwrap();
    let folded = ns.compact(&ROOMY).await.unwrap();
    assert!(matches!(folded, Compaction::Published { .. }), "{folded:?}");
    for seq in 1..=through {
        fence_by_hand(&*store, "docs", seq).await;
    }
}

#[tokio::test]
async fn the_build_is_at_the_fence_level_and_a_new_namespace_starts_there() {
    assert_eq!(FENCE_FORMAT_VERSION, 4);
    assert_eq!(FORMAT_VERSION, FENCE_FORMAT_VERSION);
    for b in backends() {
        eprintln!("backend: {}", b.name);
        Namespace::create(b.store.clone(), "docs", schema(1, Metric::L2))
            .await
            .unwrap();
        let (m, _) = manifest::load(&*b.store, "docs").await.unwrap();
        // A new namespace starts at the newest level this build reads.
        assert_eq!(m.wal_format, FENCE_FORMAT_VERSION);
    }
}

#[tokio::test]
async fn a_reader_behind_a_fold_and_a_fence_reads_again() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let faulty = FaultyStore::new(b.store.clone());
        let store: Arc<dyn ObjectStore> = faulty.clone();
        let ns = three_writes(store).await;

        faulty.before_get(&wal_key(1), fold_and_fence(b.store.clone(), 3));
        let hits = ns.query(&[0.0], 10).await.unwrap();
        assert_eq!(ids(&hits), ["a", "b", "c"]);
    }
}

#[tokio::test]
async fn a_fence_past_the_fold_point_of_a_current_manifest_is_corrupt() {
    for b in backends() {
        eprintln!("backend: {}", b.name);
        let ns = three_writes(b.store.clone()).await;
        fence_by_hand(&*b.store, "docs", 2).await;
        let before = snapshot(&*b.store).await;

        match ns.query(&[0.0], 10).await.unwrap_err() {
            ItzmaError::CorruptWal { path, .. } => {
                assert!(path.ends_with(&wal_key(2)), "{path}");
            }
            other => panic!("unexpected error: {other:?}"),
        }
        assert_untouched(&*b.store, &before).await;
        assert_eq!(snapshot(&*b.store).await, before);
    }
}

#[tokio::test]
async fn a_compaction_behind_a_fold_and_a_fence_is_superseded() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let faulty = FaultyStore::new(b.store.clone());
        let store: Arc<dyn ObjectStore> = faulty.clone();
        let ns = three_writes(store).await;

        faulty.before_get(&wal_key(1), fold_and_fence(b.store.clone(), 3));
        assert_eq!(ns.compact(&ROOMY).await.unwrap(), Compaction::Superseded);

        let hits = ns.query(&[0.0], 10).await.unwrap();
        assert_eq!(ids(&hits), ["a", "b", "c"]);
    }
}

#[tokio::test]
async fn a_compaction_over_a_fence_in_a_current_manifest_is_corrupt_and_writes_nothing() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = three_writes(b.store.clone()).await;
        fence_by_hand(&*b.store, "docs", 2).await;
        let before = snapshot(&*b.store).await;

        let err = ns.compact(&ROOMY).await.unwrap_err();
        assert!(matches!(err, ItzmaError::CorruptWal { .. }), "{err:?}");
        assert_eq!(snapshot(&*b.store).await, before);
    }
}

#[tokio::test]
async fn a_writer_with_a_stale_hint_lands_after_the_fences() {
    for b in backends() {
        eprintln!("backend: {}", b.name);
        let stale = Namespace::create(b.store.clone(), "docs", schema(1, Metric::L2))
            .await
            .unwrap();
        stale.upsert(vec![doc("a", &[1.0])]).await.unwrap();
        let other = Namespace::open(b.store.clone(), "docs").await.unwrap();
        for id in ["b", "c", "d", "e"] {
            other.upsert(vec![doc(id, &[2.0])]).await.unwrap();
        }
        for seq in 1..=5 {
            fence_by_hand(&*b.store, "docs", seq).await;
        }

        // The stale handle's hint is 2. Its put runs into a fence, it lists
        // past it and appends after the highest seq.
        let seq = stale.upsert(vec![doc("f", &[3.0])]).await.unwrap();
        assert_eq!(seq, 6);

        let tail = wal::read_log(&*b.store, "docs", 5).await.unwrap();
        assert_eq!(tail.iter().map(|(s, _)| *s).collect::<Vec<_>>(), [6]);
    }
}

#[tokio::test]
async fn a_deleted_key_would_have_let_the_stale_writer_in() {
    // Why folded WAL objects are fenced, not deleted, shown with a delete: the
    // put succeeds at a seq that is already folded, where no reader looks.
    for b in backends() {
        eprintln!("backend: {}", b.name);
        let stale = Namespace::create(b.store.clone(), "docs", schema(1, Metric::L2))
            .await
            .unwrap();
        stale.upsert(vec![doc("a", &[1.0])]).await.unwrap();
        let other = Namespace::open(b.store.clone(), "docs").await.unwrap();
        other.upsert(vec![doc("b", &[2.0])]).await.unwrap();
        for seq in 1..=2 {
            b.store.delete(&wal::wal_path("docs", seq)).await.unwrap();
        }

        let seq = stale.upsert(vec![doc("c", &[3.0])]).await.unwrap();
        assert_eq!(seq, 2, "acknowledged inside the range that was folded");
    }
}

#[tokio::test]
async fn a_keyed_retry_finds_its_object_until_it_is_fenced() {
    for b in backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", schema(1, Metric::L2))
            .await
            .unwrap();
        let first = ns
            .upsert_keyed(&WriteKey::new("load/1"), vec![doc("a", &[1.0])])
            .await
            .unwrap();
        assert_eq!(first, 1);

        let retry = WriteKey::retry("load/1", 1);
        let again = ns.upsert_keyed(&retry, vec![doc("a", &[1.0])]).await;
        assert_eq!(again.unwrap(), 1, "found while the object is there");

        // A fence has no key, so once the object is fenced the retry cannot
        // see it and writes again. That is the documented window.
        fence_by_hand(&*b.store, "docs", 1).await;
        let after = ns.upsert_keyed(&retry, vec![doc("a", &[1.0])]).await;
        assert_eq!(after.unwrap(), 2);
    }
}

// Fencing by the collector (phase B).

const DIM: usize = 1;
const L2: Metric = Metric::L2;

/// Three writes folded into one segment, then two more that are not. The WAL
/// holds seqs 1 to 5 and the fold point is 3.
async fn prepared(store: Arc<dyn ObjectStore>) -> Namespace {
    let ns = three_writes(store).await;
    ns.compact(&ROOMY).await.unwrap();
    for (id, x) in [("d", 4.0), ("e", 5.0)] {
        ns.upsert(vec![doc(id, &[x])]).await.unwrap();
    }
    ns
}

const ALL: [&str; 5] = ["a", "b", "c", "d", "e"];

/// The WAL objects in a snapshot, by seq.
fn wal_of(snap: &Snapshot) -> BTreeMap<u64, Bytes> {
    snap.iter()
        .filter_map(|(key, bytes)| {
            let name = key.strip_prefix("docs/wal/")?.strip_suffix(".json")?;
            Some((name.parse().ok()?, bytes.clone()))
        })
        .collect()
}

/// Everything in `after` that is not a WAL object is what `before` had, apart
/// from the manifest.
fn assert_only_wal_changed(before: &Snapshot, after: &Snapshot) {
    for (key, bytes) in before {
        if key.contains("/wal/") || key.ends_with("manifest/current.json") || key == HINT {
            continue;
        }
        assert_eq!(after.get(key), Some(bytes), "{key} changed or vanished");
    }
    for key in after.keys() {
        assert!(before.contains_key(key) || key == HINT, "gc created {key}");
    }
}

/// The fence hint's key. A fencing run may write it, and nothing else
/// outside the WAL.
const HINT: &str = "docs/gc/fenced-through";

async fn hint_bytes(store: &dyn ObjectStore) -> Option<Bytes> {
    match store.get(&fenced_through_path("docs")).await {
        Ok(got) => Some(got.bytes().await.unwrap()),
        Err(object_store::Error::NotFound { .. }) => None,
        Err(e) => panic!("{e}"),
    }
}

async fn plant_hint(store: &dyn ObjectStore, bytes: &'static [u8]) {
    store
        .put(
            &fenced_through_path("docs"),
            Bytes::from_static(bytes).into(),
        )
        .await
        .unwrap();
}

fn fenced_bytes() -> Bytes {
    Bytes::from_static(&FENCE)
}

#[tokio::test]
async fn without_the_flag_the_wal_is_left_alone() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = prepared(b.store.clone()).await;
        let before = snapshot(&*b.store).await;

        assert_eq!(ns.gc(&NOW).await.unwrap(), GcReport::default());
        assert_eq!(snapshot(&*b.store).await, before);
    }
}

#[tokio::test]
async fn a_fencing_run_fences_the_folded_objects_and_only_those() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = prepared(b.store.clone()).await;
        let before = snapshot(&*b.store).await;
        let old = wal_of(&before);

        let report = ns.gc(&FENCING).await.unwrap();
        let freed: u64 = (1..=3).map(|seq| old[&seq].len() as u64 - 8).sum();
        assert_eq!(
            report,
            GcReport {
                wal_objects_fenced: 3,
                wal_bytes_freed: freed,
                ..GcReport::default()
            }
        );

        let after = snapshot(&*b.store).await;
        let now = wal_of(&after);
        for seq in 1..=3 {
            assert_eq!(now[&seq], fenced_bytes(), "seq {seq}");
        }
        for seq in 4..=5 {
            assert_eq!(now[&seq], old[&seq], "seq {seq}");
        }
        assert_only_wal_changed(&before, &after);

        let docs = assert_consistent(b.store.clone(), DIM, L2).await;
        assert_eq!(docs.len(), 5);
    }
}

#[tokio::test]
async fn a_fenced_namespace_still_answers_and_takes_writes_and_compactions() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = prepared(b.store.clone()).await;
        ns.gc(&FENCING).await.unwrap();
        assert_eq!(ids(&ns.query(&[0.0], 10).await.unwrap()), ALL);

        ns.upsert(vec![doc("f", &[6.0])]).await.unwrap();
        assert!(matches!(
            ns.compact(&ROOMY).await.unwrap(),
            Compaction::Published { .. }
        ));
        let merged = ns.compact(&TIGHT).await.unwrap();
        assert!(matches!(merged, Compaction::Published { .. }), "{merged:?}");
        assert_eq!(
            ids(&ns.query(&[0.0], 10).await.unwrap()),
            ["a", "b", "c", "d", "e", "f"]
        );

        // The merged base was built from segments alone, and a second
        // collector fences what the new folds covered.
        ns.gc(&FENCING).await.unwrap();
        assert_eq!(assert_consistent(b.store.clone(), DIM, L2).await.len(), 6);
    }
}

#[tokio::test]
async fn a_fencing_run_records_how_far_the_fences_reach() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = prepared(b.store.clone()).await;
        assert_eq!(hint_bytes(&*b.store).await, None);

        ns.gc(&FENCING).await.unwrap();
        // Seqs 4 and 5 are past the fold point, so the run stops at 3.
        assert_eq!(hint_bytes(&*b.store).await.as_deref(), Some(&b"3"[..]));
        let before = snapshot(&*b.store).await;

        // Nothing new to fence, so nothing to write.
        assert_eq!(ns.gc(&FENCING).await.unwrap(), GcReport::default());
        assert_eq!(snapshot(&*b.store).await, before);

        // Folding the tail lets the next run take it, and the hint follows.
        ns.compact(&ROOMY).await.unwrap();
        let report = ns.gc(&FENCING).await.unwrap();
        assert_eq!(report.wal_objects_fenced, 2);
        assert_eq!(hint_bytes(&*b.store).await.as_deref(), Some(&b"5"[..]));
        assert_eq!(assert_consistent(b.store.clone(), DIM, L2).await.len(), 5);
    }
}

/// The hint is trusted: what lies below it is not looked at again. That is the
/// point, and it is why a hand-written one can leave objects unfenced.
#[tokio::test]
async fn a_run_lists_only_past_the_hint() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = prepared(b.store.clone()).await;
        let old = wal_of(&snapshot(&*b.store).await);
        plant_hint(&*b.store, b"2").await;

        let report = ns.gc(&FENCING).await.unwrap();
        assert_eq!(report.wal_objects_fenced, 1, "only seq 3 is past the hint");
        let now = wal_of(&snapshot(&*b.store).await);
        assert_eq!(now[&1], old[&1]);
        assert_eq!(now[&2], old[&2]);
        assert_eq!(now[&3], fenced_bytes());
        assert_eq!(hint_bytes(&*b.store).await.as_deref(), Some(&b"3"[..]));
    }
}

/// A hint that is missing, empty, not a number or not text never fails a run:
/// the run starts from the top, and writes a good one.
#[tokio::test]
async fn a_bad_hint_means_start_from_the_top() {
    let junk: [&'static [u8]; 7] = [
        b"",
        b"nonsense",
        b"-5",
        b"\xff\xfe",
        b"99999999999999999999999",
        b"3 3",
        // Numeric once trimmed, but too long to be a hint, so never read.
        b"                                        2",
    ];
    for bytes in junk {
        for b in cas_backends() {
            eprintln!("backend: {}, hint {bytes:?}", b.name);
            let ns = prepared(b.store.clone()).await;
            plant_hint(&*b.store, bytes).await;
            let report = ns.gc(&FENCING).await.unwrap();
            assert_eq!(report.wal_objects_fenced, 3, "{bytes:?}");
            assert_eq!(hint_bytes(&*b.store).await.as_deref(), Some(&b"3"[..]));
        }
    }
}

/// A hint beyond the end of the log skips everything and is not lowered. Only a
/// hand-written one can be that, and it costs space, never data.
#[tokio::test]
async fn a_hint_past_the_log_fences_nothing_and_changes_nothing() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = prepared(b.store.clone()).await;
        plant_hint(&*b.store, b"99").await;
        let before = snapshot(&*b.store).await;

        assert_eq!(ns.gc(&FENCING).await.unwrap(), GcReport::default());
        assert_eq!(snapshot(&*b.store).await, before);
    }
}

#[tokio::test]
async fn a_second_fencing_run_finds_nothing() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = prepared(b.store.clone()).await;
        ns.gc(&FENCING).await.unwrap();
        let before = snapshot(&*b.store).await;

        assert_eq!(ns.gc(&FENCING).await.unwrap(), GcReport::default());
        assert_eq!(snapshot(&*b.store).await, before);
    }
}

#[tokio::test]
async fn a_dry_run_fences_nothing_and_reports_what_a_real_run_would() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = prepared(b.store.clone()).await;
        let before = snapshot(&*b.store).await;

        let dry = ns
            .gc(&GcOptions {
                dry_run: true,
                ..FENCING
            })
            .await
            .unwrap();
        assert_eq!(snapshot(&*b.store).await, before);
        assert_eq!(dry.wal_objects_fenced, 3);

        assert_eq!(ns.gc(&FENCING).await.unwrap(), dry);
    }
}

#[tokio::test]
async fn the_default_min_age_leaves_fresh_wal_objects_alone() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = prepared(b.store.clone()).await;
        let before = snapshot(&*b.store).await;

        let options = GcOptions {
            fence_wal: true,
            ..GcOptions::default()
        };
        let report = ns.gc(&options).await.unwrap();
        assert_eq!(
            report,
            GcReport {
                wal_too_young: 3,
                ..GcReport::default()
            }
        );
        assert_eq!(snapshot(&*b.store).await, before);
    }
}

/// `min_age` against a real clock (see `tests/gc.rs`). A WAL object is fenced
/// once it and the segment that holds its seq are old enough. Seqs 4 and 5 are
/// written before the wait but folded after it, so they are old and their
/// segment is not: they must stay until it has aged.
const AGE: Duration = Duration::from_secs(3);
const WAIT: Duration = Duration::from_secs(4);

async fn fences_follow_the_age_of_the_segment_holding_each_seq(b: Backend) {
    eprintln!("backend: {}", b.name);
    let ns = three_writes(b.store.clone()).await;
    ns.compact(&ROOMY).await.unwrap();
    for (id, x) in [("d", 4.0), ("e", 5.0)] {
        ns.upsert(vec![doc(id, &[x])]).await.unwrap();
    }
    tokio::time::sleep(WAIT).await;
    ns.compact(&ROOMY).await.unwrap();

    let options = GcOptions {
        min_age: AGE,
        ..FENCING
    };
    let before = snapshot(&*b.store).await;
    let old = wal_of(&before);
    let report = ns.gc(&options).await.unwrap();
    let freed: u64 = (1..=3).map(|seq| old[&seq].len() as u64 - 8).sum();
    assert_eq!(
        report,
        GcReport {
            wal_objects_fenced: 3,
            wal_bytes_freed: freed,
            wal_too_young: 2,
            ..GcReport::default()
        }
    );
    let now = wal_of(&snapshot(&*b.store).await);
    for seq in 1..=3 {
        assert_eq!(now[&seq], fenced_bytes(), "seq {seq}");
    }
    for seq in 4..=5 {
        assert_eq!(now[&seq], old[&seq], "seq {seq}: its segment is too young");
    }
    assert_eq!(hint_bytes(&*b.store).await.as_deref(), Some(&b"3"[..]));

    tokio::time::sleep(WAIT).await;
    let report = ns.gc(&options).await.unwrap();
    assert_eq!(report.wal_objects_fenced, 2);
    assert_eq!(report.wal_too_young, 0);
    assert_eq!(hint_bytes(&*b.store).await.as_deref(), Some(&b"5"[..]));
    assert!(
        wal_of(&snapshot(&*b.store).await)
            .values()
            .all(|bytes| *bytes == fenced_bytes())
    );
    assert_eq!(ids(&ns.query(&[0.0], 10).await.unwrap()), ALL);
    assert_eq!(assert_consistent(b.store.clone(), DIM, L2).await.len(), 5);
}

#[tokio::test]
async fn wal_objects_are_fenced_only_once_the_segment_holding_them_is_old_enough() {
    futures::future::join_all(
        cas_backends()
            .into_iter()
            .map(fences_follow_the_age_of_the_segment_holding_each_seq),
    )
    .await;
}

#[tokio::test]
async fn a_namespace_below_level_4_is_refused_before_anything_is_touched() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = prepared(b.store.clone()).await;
        manifest::update(&*b.store, "docs", |m| m.wal_format = 3)
            .await
            .unwrap();
        let before = snapshot(&*b.store).await;

        for dry_run in [false, true] {
            let options = GcOptions { dry_run, ..FENCING };
            let err = ns.gc(&options).await.unwrap_err();
            assert!(
                matches!(err, ItzmaError::WalFormatTooOld { have: 3, need: 4 }),
                "{err:?}"
            );
            assert_eq!(snapshot(&*b.store).await, before);
        }

        // Without the flag the level does not matter.
        assert_eq!(ns.gc(&NOW).await.unwrap(), GcReport::default());
    }
}

/// Reach a state with folded WAL objects on a store that has no conditional
/// update, by writing the manifest and the segment by hand.
async fn folded_by_hand(store: Arc<dyn ObjectStore>) -> Namespace {
    let ns = Namespace::create(store.clone(), "docs", schema(DIM, L2))
        .await
        .unwrap();
    for (id, x) in [("a", 1.0), ("b", 2.0)] {
        ns.upsert(vec![doc(id, &[x])]).await.unwrap();
    }
    let up = |id: &str, x: f32| WalOp::Upsert(doc(id, &[x]));
    let id = put_segment(&*store, "docs", 1, 2, DIM, vec![up("a", 1.0), up("b", 2.0)]).await;
    let mut m = Manifest::new(schema(DIM, L2));
    m.live_segments = vec![id];
    m.wal_folded_through = 2;
    store
        .put(
            &manifest::manifest_path("docs"),
            serde_json::to_vec(&m).unwrap().into(),
        )
        .await
        .unwrap();
    ns
}

#[tokio::test]
async fn fencing_runs_on_a_store_without_conditional_update() {
    for b in backends() {
        eprintln!("backend: {}", b.name);
        let ns = folded_by_hand(b.store.clone()).await;

        let report = ns.gc(&FENCING).await.unwrap();
        assert_eq!(report.wal_objects_fenced, 2);
        let now = wal_of(&snapshot(&*b.store).await);
        assert_eq!(now.values().collect::<Vec<_>>(), [&fenced_bytes(); 2]);
        assert_eq!(ids(&ns.query(&[0.0], 10).await.unwrap()), ["a", "b"]);
    }
}

#[tokio::test]
async fn a_stale_writer_after_a_real_fencing_run_is_not_lost() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let stale = Namespace::create(b.store.clone(), "docs", schema(DIM, L2))
            .await
            .unwrap();
        stale.upsert(vec![doc("a", &[1.0])]).await.unwrap();
        let other = Namespace::open(b.store.clone(), "docs").await.unwrap();
        for (id, x) in [("b", 2.0), ("c", 3.0)] {
            other.upsert(vec![doc(id, &[x])]).await.unwrap();
        }
        other.compact(&ROOMY).await.unwrap();
        other.gc(&FENCING).await.unwrap();

        // The stale handle's hint is 2, inside the folded and fenced range.
        let seq = stale.upsert(vec![doc("d", &[4.0])]).await.unwrap();
        assert_eq!(seq, 4);
        assert_eq!(
            ids(&other.query(&[0.0], 10).await.unwrap()),
            ["a", "b", "c", "d"]
        );
        assert_eq!(assert_consistent(b.store.clone(), DIM, L2).await.len(), 4);
    }
}

fn faulty(ix: usize) -> Arc<FaultyStore> {
    FaultyStore::new(cas_backends().swap_remove(ix).store)
}

fn dynamic(store: &Arc<FaultyStore>) -> Arc<dyn ObjectStore> {
    store.clone()
}

/// A fault in the middle of a fencing run leaves any mix of fenced and whole
/// objects at or below the fold point, and nothing else changes.
fn assert_fenced_or_whole(before: &Snapshot, after: &Snapshot) {
    assert_only_wal_changed(before, after);
    let (old, now) = (wal_of(before), wal_of(after));
    for (seq, bytes) in &old {
        let got = &now[seq];
        if *seq <= 3 {
            assert!(got == bytes || *got == fenced_bytes(), "seq {seq}");
        } else {
            assert_eq!(got, bytes, "seq {seq} is past the fold point");
        }
    }
}

/// Run a fencing collection with a fault at every store operation it makes,
/// once as a plain error and once as a lost ack. After each: a restart on a
/// fresh handle sees a consistent namespace with all five documents, a rerun
/// finishes the fencing and then finds nothing more, and a writer with a stale
/// hint still lands after the end of the log.
#[tokio::test]
async fn fencing_survives_a_fault_at_every_step() {
    for ix in 0..cas_backends().len() {
        let store = faulty(ix);
        let ns = prepared(dynamic(&store)).await;
        store.reset_ops();
        ns.gc(&FENCING).await.unwrap();
        let total = store.ops();
        assert!(
            total >= 6,
            "the manifest, two listings and three puts: {total}"
        );

        for fault in [Fault::Generic, Fault::LostAck] {
            for n in 0..total {
                eprintln!("backend {ix}: {fault:?} at operation {n} of {total}");
                let store = faulty(ix);
                let ns = prepared(dynamic(&store)).await;
                let before = snapshot(&*store).await;

                store.fail_nth_op(n, fault);
                let _ = ns.gc(&FENCING).await;
                store.clear();
                drop(ns);

                assert_fenced_or_whole(&before, &snapshot(&*store).await);
                assert_eq!(assert_consistent(dynamic(&store), DIM, L2).await.len(), 5);

                let ns = Namespace::open(dynamic(&store), "docs").await.unwrap();
                ns.gc(&FENCING).await.unwrap();
                let now = wal_of(&snapshot(&*store).await);
                for seq in 1..=3 {
                    assert_eq!(now[&seq], fenced_bytes(), "seq {seq}");
                }
                assert_fenced_or_whole(&before, &snapshot(&*store).await);
                assert_eq!(ns.gc(&FENCING).await.unwrap(), GcReport::default());
                assert_eq!(hint_bytes(&*store).await.as_deref(), Some(&b"3"[..]));

                let seq = ns.upsert(vec![doc("f", &[6.0])]).await.unwrap();
                assert_eq!(seq, 6);
                assert_eq!(assert_consistent(dynamic(&store), DIM, L2).await.len(), 6);
            }
        }
    }
}
