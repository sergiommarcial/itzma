mod common;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use bytes::Bytes;
use common::faulty::{Fault, FaultyStore, PutKind};
use common::{Lcg, assert_consistent, assert_untouched, cas_backends, doc, naive_top_k, schema};
use futures::TryStreamExt;
use itzma_core::wal::{self, wal_prefix};
use itzma_core::{
    Compaction, CompactionOptions, Document, Hit, ItzmaError, Metric, Namespace, WalOp, manifest,
    segment,
};
use object_store::local::LocalFileSystem;
use object_store::memory::InMemory;
use object_store::{ObjectStore, ObjectStoreExt};

const DEFAULT: CompactionOptions = CompactionOptions {
    max_segments: 8,
    max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
};

fn sorted_ids(hits: &[Hit]) -> Vec<String> {
    let mut ids: Vec<String> = hits.iter().map(|h| h.id.clone()).collect();
    ids.sort();
    ids
}

/// The manifest's segment list and fold point.
async fn live(store: &dyn ObjectStore, ns: &str) -> (Vec<String>, u64) {
    let (m, _) = manifest::load(store, ns).await.unwrap();
    (m.live_segments, m.wal_folded_through)
}

/// `n` documents named `<prefix>0000`, `<prefix>0001`, ..., with random
/// vectors from a seeded generator.
fn docs_of(prefix: &str, seed: u64, n: usize) -> Vec<Document> {
    let mut rng = Lcg(seed);
    (0..n)
        .map(|i| doc(&format!("{prefix}{i:04}"), &rng.vector(DIM)))
        .collect()
}

#[tokio::test]
async fn fold_publishes_one_base_segment_and_keeps_the_wal() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", schema(2, Metric::L2))
            .await
            .unwrap();
        ns.upsert(vec![doc("a", &[1.0, 0.0]), doc("b", &[2.0, 0.0])])
            .await
            .unwrap();
        ns.upsert(vec![doc("c", &[3.0, 0.0])]).await.unwrap();
        ns.delete(vec!["a".into()]).await.unwrap();

        let id = segment::segment_id(1, 3);
        assert_eq!(
            ns.compact(&DEFAULT).await.unwrap(),
            Compaction::Published {
                segments: vec![id.clone()],
                folded_through: 3,
                merged_segments: 0,
                docs: 2,
                tombstones: 0, // a base drops its tombstones
            }
        );
        assert_eq!(live(&*b.store, "docs").await, (vec![id.clone()], 3));

        let wal: Vec<_> = b
            .store
            .list(Some(&wal_prefix("docs")))
            .try_collect()
            .await
            .unwrap();
        assert_eq!(wal.len(), 3, "compaction deletes nothing");
        b.store
            .head(&segment::segment_path("docs", &id))
            .await
            .unwrap();

        let hits = ns.query(&[0.0, 0.0], 10).await.unwrap();
        assert_eq!(sorted_ids(&hits), ["b", "c"]);
    }
}

#[tokio::test]
async fn a_later_fold_is_a_delta_that_keeps_its_tombstones() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", schema(1, Metric::L2))
            .await
            .unwrap();
        ns.upsert(vec![doc("a", &[1.0]), doc("b", &[2.0])])
            .await
            .unwrap();
        ns.compact(&DEFAULT).await.unwrap();

        ns.delete(vec!["a".into()]).await.unwrap();
        ns.upsert(vec![doc("c", &[3.0])]).await.unwrap();
        let delta = segment::segment_id(2, 3);
        assert_eq!(
            ns.compact(&DEFAULT).await.unwrap(),
            Compaction::Published {
                segments: vec![delta.clone()],
                folded_through: 3,
                merged_segments: 0,
                docs: 1,
                tombstones: 1,
            }
        );
        assert_eq!(
            live(&*b.store, "docs").await,
            (vec![segment::segment_id(1, 1), delta], 3)
        );
        let hits = ns.query(&[0.0], 10).await.unwrap();
        assert_eq!(sorted_ids(&hits), ["b", "c"]);
    }
}

#[tokio::test]
async fn nothing_to_do_when_the_tail_is_empty() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", schema(1, Metric::L2))
            .await
            .unwrap();
        assert_eq!(ns.compact(&DEFAULT).await.unwrap(), Compaction::Nothing);
        assert_eq!(live(&*b.store, "docs").await, (vec![], 0));

        ns.upsert(vec![doc("a", &[1.0])]).await.unwrap();
        assert!(matches!(
            ns.compact(&DEFAULT).await.unwrap(),
            Compaction::Published { .. }
        ));
        let after_fold = live(&*b.store, "docs").await;
        assert_eq!(ns.compact(&DEFAULT).await.unwrap(), Compaction::Nothing);
        assert_eq!(live(&*b.store, "docs").await, after_fold);
    }
}

#[tokio::test]
async fn writes_after_a_compaction_continue_at_the_next_seq() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", schema(1, Metric::L2))
            .await
            .unwrap();
        // Opened before any write, so its hint is stale from here on.
        let stale = Namespace::open(b.store.clone(), "docs").await.unwrap();
        for i in 0..3 {
            ns.upsert(vec![doc(&format!("d{i}"), &[i as f32])])
                .await
                .unwrap();
        }
        ns.compact(&DEFAULT).await.unwrap();

        assert_eq!(ns.upsert(vec![doc("d3", &[3.0])]).await.unwrap(), 4);
        let fresh = Namespace::open(b.store.clone(), "docs").await.unwrap();
        assert_eq!(fresh.upsert(vec![doc("d4", &[4.0])]).await.unwrap(), 5);
        // The folded WAL objects are still there, so put-if-absent still
        // rejects the stale hint and the writer resyncs.
        assert_eq!(stale.upsert(vec![doc("d5", &[5.0])]).await.unwrap(), 6);

        let hits = ns.query(&[0.0], 10).await.unwrap();
        assert_eq!(sorted_ids(&hits), ["d0", "d1", "d2", "d3", "d4", "d5"]);
    }
}

#[tokio::test]
async fn merge_when_the_segment_count_would_exceed_the_max() {
    let opts = CompactionOptions {
        max_segments: 2,
        max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
    };
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", schema(1, Metric::L2))
            .await
            .unwrap();
        ns.upsert(vec![doc("a", &[1.0])]).await.unwrap();
        ns.compact(&opts).await.unwrap();
        ns.upsert(vec![doc("b", &[2.0])]).await.unwrap();
        ns.compact(&opts).await.unwrap();
        assert_eq!(live(&*b.store, "docs").await.0.len(), 2);

        ns.upsert(vec![doc("c", &[3.0])]).await.unwrap();
        let merged = segment::segment_id(1, 3);
        assert_eq!(
            ns.compact(&opts).await.unwrap(),
            Compaction::Published {
                segments: vec![merged.clone()],
                folded_through: 3,
                merged_segments: 2,
                docs: 3,
                tombstones: 0,
            }
        );
        assert_eq!(live(&*b.store, "docs").await, (vec![merged], 3));
        assert_eq!(ns.compact(&opts).await.unwrap(), Compaction::Nothing);

        let hits = ns.query(&[0.0], 10).await.unwrap();
        assert_eq!(sorted_ids(&hits), ["a", "b", "c"]);
    }
}

#[tokio::test]
async fn a_tombstone_hides_a_base_doc_until_a_merge_drops_it() {
    let opts = CompactionOptions {
        max_segments: 2,
        max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
    };
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", schema(1, Metric::L2))
            .await
            .unwrap();
        ns.upsert(vec![doc("a", &[1.0]), doc("b", &[2.0])])
            .await
            .unwrap();
        ns.compact(&opts).await.unwrap();

        ns.delete(vec!["a".into()]).await.unwrap();
        assert_eq!(
            ns.compact(&opts).await.unwrap(),
            Compaction::Published {
                segments: vec![segment::segment_id(2, 2)],
                folded_through: 2,
                merged_segments: 0,
                docs: 0,
                tombstones: 1,
            }
        );
        let hits = ns.query(&[0.0], 10).await.unwrap();
        assert_eq!(sorted_ids(&hits), ["b"]);

        ns.upsert(vec![doc("c", &[3.0])]).await.unwrap();
        let merged = segment::segment_id(1, 3);
        assert_eq!(
            ns.compact(&opts).await.unwrap(),
            Compaction::Published {
                segments: vec![merged.clone()],
                folded_through: 3,
                merged_segments: 2,
                docs: 2,
                tombstones: 0,
            }
        );
        let path = segment::segment_path("docs", &merged);
        let bytes = b.store.get(&path).await.unwrap().bytes().await.unwrap();
        let layer = segment::decode(path.as_ref(), &bytes, 1).unwrap();
        assert_eq!((layer.doc_count(), layer.tombstone_count()), (2, 0));
        let hits = ns.query(&[0.0], 10).await.unwrap();
        assert_eq!(sorted_ids(&hits), ["b", "c"]);
    }
}

#[tokio::test]
async fn max_segments_of_zero_behaves_like_one() {
    let opts = CompactionOptions {
        max_segments: 0,
        max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
    };
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", schema(1, Metric::L2))
            .await
            .unwrap();
        ns.upsert(vec![doc("a", &[1.0])]).await.unwrap();
        ns.compact(&opts).await.unwrap();
        ns.upsert(vec![doc("b", &[2.0])]).await.unwrap();
        assert!(matches!(
            ns.compact(&opts).await.unwrap(),
            Compaction::Published {
                merged_segments: 1,
                ..
            }
        ));
        assert_eq!(
            live(&*b.store, "docs").await,
            (vec![segment::segment_id(1, 2)], 2)
        );
    }
}

#[tokio::test]
async fn a_wal_of_only_deletes_folds_to_an_empty_base() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", schema(1, Metric::L2))
            .await
            .unwrap();
        ns.delete(vec!["ghost".into()]).await.unwrap();
        assert_eq!(
            ns.compact(&DEFAULT).await.unwrap(),
            Compaction::Published {
                segments: vec![segment::segment_id(1, 1)],
                folded_through: 1,
                merged_segments: 0,
                docs: 0,
                tombstones: 0,
            }
        );
        assert!(ns.query(&[0.0], 5).await.unwrap().is_empty());
    }
}

const DIM: usize = 4;
const POOL: u64 = 12;

fn pool_id(n: u64) -> String {
    format!("d{n:02}")
}

/// Random upserts, replacements and deletes over a small id pool, with a
/// compaction every fourth step and a small `max_segments` so both folds and
/// merges happen. After every step the answer must equal a naive scan of a
/// model that never heard of segments.
#[tokio::test]
async fn queries_are_unchanged_by_compaction() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", schema(DIM, Metric::L2))
            .await
            .unwrap();
        let opts = CompactionOptions {
            max_segments: 3,
            max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
        };
        let mut rng = Lcg(7);
        let mut model: BTreeMap<String, Document> = BTreeMap::new();
        let (mut folds, mut merges) = (0, 0);

        for step in 0..40 {
            if rng.below(4) == 0 {
                let n = 1 + rng.below(3);
                let ids: Vec<String> = (0..n).map(|_| pool_id(rng.below(POOL))).collect();
                for id in &ids {
                    model.remove(id);
                }
                ns.delete(ids).await.unwrap();
            } else {
                let n = 1 + rng.below(4);
                let docs: Vec<Document> = (0..n)
                    .map(|_| Document {
                        id: pool_id(rng.below(POOL)),
                        vector: rng.vector(DIM),
                        attrs: None,
                        sparse: None,
                    })
                    .collect();
                for d in &docs {
                    model.insert(d.id.clone(), d.clone());
                }
                ns.upsert(docs).await.unwrap();
            }

            if step % 4 == 3 {
                match ns.compact(&opts).await.unwrap() {
                    Compaction::Published {
                        merged_segments: 0, ..
                    } => folds += 1,
                    Compaction::Published { .. } => merges += 1,
                    other => panic!("step {step}: unexpected {other:?}"),
                }
            }

            let expected: Vec<Document> = model.values().cloned().collect();
            for _ in 0..3 {
                let q = rng.vector(DIM);
                for k in [3, 100] {
                    let got: Vec<String> = ns
                        .query(&q, k)
                        .await
                        .unwrap()
                        .into_iter()
                        .map(|h| h.id)
                        .collect();
                    assert_eq!(
                        got,
                        naive_top_k(Metric::L2, &q, &expected, k),
                        "step {step}, k {k}"
                    );
                }
            }
        }
        assert!(
            folds >= 1 && merges >= 1,
            "workload must exercise both: {folds} folds, {merges} merges"
        );
    }
}

#[tokio::test]
async fn crash_before_publish_leaves_an_orphan_that_the_rerun_reuses() {
    let inner = Arc::new(InMemory::new());
    let store = FaultyStore::new(inner.clone());
    let ns = Namespace::create(store.clone(), "docs", schema(2, Metric::L2))
        .await
        .unwrap();
    ns.upsert(vec![doc("a", &[1.0, 0.0])]).await.unwrap();

    // The segment write goes through; the manifest swap dies.
    store.fail_puts(
        "manifest/current.json",
        PutKind::Update,
        Fault::Generic,
        Some(1),
    );
    let err = ns.compact(&DEFAULT).await.unwrap_err();
    assert!(matches!(err, ItzmaError::Storage(_)), "{err:?}");

    let id = segment::segment_id(1, 1);
    inner
        .head(&segment::segment_path("docs", &id))
        .await
        .expect("the orphan segment stays");
    assert_eq!(live(&*store, "docs").await, (vec![], 0));
    assert_eq!(ns.query(&[0.0, 0.0], 5).await.unwrap().len(), 1);

    // The rerun finds the segment already there, reuses it, and publishes.
    assert!(matches!(
        ns.compact(&DEFAULT).await.unwrap(),
        Compaction::Published { .. }
    ));
    assert_eq!(live(&*store, "docs").await, (vec![id], 1));
    assert_eq!(ns.query(&[0.0, 0.0], 5).await.unwrap().len(), 1);
}

/// The other half of "never overwrite": the segment write must be a
/// put-if-absent, so a fault injected on those puts has to reach the caller.
/// An overwriting put would slip past the rule and this test would fail.
#[tokio::test]
async fn segments_are_written_with_put_if_absent() {
    let store = FaultyStore::new(Arc::new(InMemory::new()));
    let ns = Namespace::create(store.clone(), "docs", schema(1, Metric::L2))
        .await
        .unwrap();
    ns.upsert(vec![doc("a", &[1.0])]).await.unwrap();

    store.fail_puts("segments/", PutKind::Create, Fault::Generic, Some(1));
    let err = ns.compact(&DEFAULT).await.unwrap_err();
    assert!(matches!(err, ItzmaError::Storage(_)), "{err:?}");
    assert_eq!(live(&*store, "docs").await, (vec![], 0));
}

#[tokio::test]
async fn compaction_gives_up_when_the_publish_can_never_win() {
    let inner = Arc::new(InMemory::new());
    let store = FaultyStore::new(inner.clone());
    let ns = Namespace::create(store.clone(), "docs", schema(1, Metric::L2))
        .await
        .unwrap();
    ns.upsert(vec![doc("a", &[1.0])]).await.unwrap();

    store.fail_puts(
        "manifest/current.json",
        PutKind::Update,
        Fault::Precondition,
        None,
    );
    let err = ns.compact(&DEFAULT).await.unwrap_err();
    assert!(
        matches!(err, ItzmaError::ManifestContention { attempts: 32 }),
        "{err:?}"
    );
    inner
        .head(&segment::segment_path("docs", &segment::segment_id(1, 1)))
        .await
        .expect("the orphan segment stays");
    assert_eq!(live(&*store, "docs").await, (vec![], 0));
}

#[tokio::test]
async fn compaction_refuses_a_wal_document_with_the_wrong_dimension() {
    let store = Arc::new(InMemory::new());
    let ns = Namespace::create(store.clone(), "docs", schema(2, Metric::L2))
        .await
        .unwrap();
    // Bypass the namespace's validation, as a hand-edited WAL would.
    let bad = WalOp::Upsert(doc("a", &[1.0, 2.0, 3.0]));
    wal::append(&*store, "docs", 1, vec![bad]).await.unwrap();

    let err = ns.compact(&DEFAULT).await.unwrap_err();
    assert!(
        matches!(
            err,
            ItzmaError::DimensionMismatch {
                expected: 2,
                got: 3
            }
        ),
        "{err:?}"
    );
    assert_eq!(live(&*store, "docs").await, (vec![], 0));
    let prefix = object_store::path::Path::from("docs/segments");
    let segments: Vec<_> = store.list(Some(&prefix)).try_collect().await.unwrap();
    assert!(segments.is_empty(), "nothing may be written for bad input");
}

/// Records a known gap: `LocalFileSystem` has no conditional update, so a
/// compaction there fails at the publish step. Queries are unaffected.
#[tokio::test]
async fn compaction_is_unsupported_on_local_fs() {
    let dir = tempfile::TempDir::new().unwrap();
    let store = Arc::new(LocalFileSystem::new_with_prefix(dir.path()).unwrap());
    let ns = Namespace::create(store, "docs", schema(1, Metric::L2))
        .await
        .unwrap();
    ns.upsert(vec![doc("a", &[1.0])]).await.unwrap();
    let err = ns.compact(&DEFAULT).await.unwrap_err();
    assert!(
        matches!(
            err,
            ItzmaError::Storage(object_store::Error::NotImplemented { .. })
        ),
        "{err:?}"
    );
    assert_eq!(ns.query(&[0.0], 5).await.unwrap().len(), 1);
}

/// Writers and several compactors on separate handles, all at once. Whatever
/// interleaving happens, the manifest must end up valid, at least one publish
/// must have won, and every document must still be there.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_compactors_and_writers_leave_a_valid_manifest() {
    const WRITERS: usize = 4;
    const BATCHES: usize = 8;
    const COMPACTORS: usize = 3;
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let opts = CompactionOptions {
            max_segments: 3,
            max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
        };
        let ns = Namespace::create(b.store.clone(), "docs", schema(2, Metric::L2))
            .await
            .unwrap();
        // Something to fold from the start, so a compactor always has work.
        ns.upsert(vec![doc("seed", &[0.0, 0.0])]).await.unwrap();

        let mut writers = Vec::new();
        for w in 0..WRITERS {
            let store = b.store.clone();
            writers.push(tokio::spawn(async move {
                let ns = Namespace::open(store, "docs").await.unwrap();
                for i in 0..BATCHES {
                    let id = format!("w{w}-{i}");
                    ns.upsert(vec![doc(&id, &[w as f32, i as f32])])
                        .await
                        .unwrap();
                    tokio::task::yield_now().await;
                }
            }));
        }
        let mut compactors = Vec::new();
        for _ in 0..COMPACTORS {
            let store = b.store.clone();
            compactors.push(tokio::spawn(async move {
                let ns = Namespace::open(store, "docs").await.unwrap();
                let mut published = 0;
                for _ in 0..6 {
                    if let Compaction::Published { .. } = ns.compact(&opts).await.unwrap() {
                        published += 1;
                    }
                    tokio::task::yield_now().await;
                }
                published
            }));
        }
        for w in writers {
            w.await.unwrap();
        }
        let mut published = 0;
        for c in compactors {
            published += c.await.unwrap();
        }
        assert!(published >= 1, "some compaction must have won");

        // Bring the tail to zero: at most two more runs are ever needed.
        let mut settled = false;
        for _ in 0..4 {
            if ns.compact(&opts).await.unwrap() == Compaction::Nothing {
                settled = true;
                break;
            }
        }
        assert!(settled, "compaction should reach Nothing");

        let (segments, folded) = live(&*b.store, "docs").await;
        segment::validate_chain(&segments, folded).unwrap();
        assert!(segments.len() <= 3, "{segments:?}");

        let mut expected: Vec<String> = (0..WRITERS)
            .flat_map(|w| (0..BATCHES).map(move |i| format!("w{w}-{i}")))
            .collect();
        expected.push("seed".into());
        expected.sort();
        let hits = ns.query(&[0.0, 0.0], 1000).await.unwrap();
        assert_eq!(sorted_ids(&hits), expected);

        let wal: Vec<_> = b
            .store
            .list(Some(&wal_prefix("docs")))
            .try_collect()
            .await
            .unwrap();
        assert_eq!(wal.len(), expected.len(), "no WAL object may be deleted");
    }
}

/// A reader hammers queries while the same documents are rewritten and
/// compacted over and over. The id set never changes, so any lost or doubled
/// document is a torn read between the manifest, the segments and the WAL.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn queries_during_compaction_never_lose_or_duplicate_documents() {
    const DOCS: usize = 20;
    const MIN_QUERIES: u32 = 20;
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let opts = CompactionOptions {
            max_segments: 3,
            max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
        };
        let ns = Namespace::create(b.store.clone(), "docs", schema(2, Metric::L2))
            .await
            .unwrap();
        let docs: Vec<Document> = (0..DOCS)
            .map(|i| doc(&format!("d{i:02}"), &[i as f32, 0.0]))
            .collect();
        ns.upsert(docs.clone()).await.unwrap();
        let mut expected: Vec<String> = docs.iter().map(|d| d.id.clone()).collect();
        expected.sort();

        let stop = Arc::new(AtomicBool::new(false));
        let queries = Arc::new(AtomicU32::new(0));
        let reader = {
            let (store, stop, queries) = (b.store.clone(), stop.clone(), queries.clone());
            tokio::spawn(async move {
                let ns = Namespace::open(store, "docs").await.unwrap();
                while !stop.load(Ordering::Relaxed) {
                    let hits = ns.query(&[0.0, 0.0], 100).await.unwrap();
                    assert_eq!(sorted_ids(&hits), expected);
                    queries.fetch_add(1, Ordering::Relaxed);
                }
            })
        };

        // On a fast store the rewrites can finish before the reader task is
        // even scheduled, so keep going until it has queried a fair number of
        // times (or died, in which case the join below reports why).
        let mut rounds = 0;
        while rounds < 10
            || (queries.load(Ordering::Relaxed) < MIN_QUERIES && !reader.is_finished())
        {
            for half in docs.chunks(DOCS / 2) {
                ns.upsert(half.to_vec()).await.unwrap();
            }
            ns.compact(&opts).await.unwrap();
            rounds += 1;
            assert!(rounds < 2000, "the reader is starved");
            tokio::task::yield_now().await;
        }
        stop.store(true, Ordering::Relaxed);
        reader.await.unwrap();
        assert!(queries.load(Ordering::Relaxed) >= MIN_QUERIES);
    }
}

/// an object already at the segment's key is compared with the bytes
/// this run built. Garbage there must be refused, not published.
#[tokio::test]
async fn a_garbage_orphan_is_refused_and_the_operator_can_clear_it() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", schema(1, Metric::L2))
            .await
            .unwrap();
        ns.upsert(vec![doc("a", &[1.0])]).await.unwrap();

        let id = segment::segment_id(1, 1);
        let key = segment::segment_path("docs", &id);
        b.store
            .put(&key, Bytes::from_static(b"not a segment").into())
            .await
            .unwrap();

        let err = ns.compact(&DEFAULT).await.unwrap_err();
        assert!(matches!(err, ItzmaError::CorruptSegment { .. }), "{err:?}");
        assert_eq!(live(&*b.store, "docs").await, (vec![], 0));
        // Queries read the WAL tail past the unchanged manifest.
        assert_eq!(sorted_ids(&ns.query(&[0.0], 5).await.unwrap()), ["a"]);

        // Recovery: nothing lists the object, so delete it and rerun.
        b.store.delete(&key).await.unwrap();
        assert!(matches!(
            ns.compact(&DEFAULT).await.unwrap(),
            Compaction::Published { .. }
        ));
        assert_eq!(live(&*b.store, "docs").await, (vec![id], 1));
        assert_eq!(sorted_ids(&ns.query(&[0.0], 5).await.unwrap()), ["a"]);
    }
}

/// The worse case: the object at the key is a perfectly valid segment, just
/// not the one this range produces. Only a byte comparison catches it.
#[tokio::test]
async fn a_valid_segment_with_other_content_is_refused_too() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", schema(1, Metric::L2))
            .await
            .unwrap();
        ns.upsert(vec![doc("a", &[1.0])]).await.unwrap();
        common::put_segment(
            &*b.store,
            "docs",
            1,
            1,
            1,
            vec![WalOp::Upsert(doc("impostor", &[9.0]))],
        )
        .await;

        let err = ns.compact(&DEFAULT).await.unwrap_err();
        assert!(matches!(err, ItzmaError::CorruptSegment { .. }), "{err:?}");
        assert_eq!(live(&*b.store, "docs").await, (vec![], 0));
        assert_eq!(sorted_ids(&ns.query(&[0.0], 5).await.unwrap()), ["a"]);
    }
}

// ---------------------------------------------------------------------------
// Segment size cap and merge policy
// ---------------------------------------------------------------------------

const TINY_CAP: CompactionOptions = CompactionOptions {
    max_segments: 1,
    max_segment_bytes: 2_000, // small enough that a handful of 4-dim docs already exceeds it
};

#[tokio::test]
async fn a_merge_over_the_cap_produces_several_balanced_segments() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", schema(DIM, Metric::L2))
            .await
            .unwrap();
        // Several small folds, each comfortably under TINY_CAP alone, so the
        // trigger is the segment-count rule (max_segments: 1), and the
        // grouping rule is what's under test.
        for i in 0..6 {
            ns.upsert(docs_of(&format!("g{i}"), i as u64, 20))
                .await
                .unwrap();
            ns.compact(&CompactionOptions {
                max_segments: 8,
                max_segment_bytes: TINY_CAP.max_segment_bytes,
            })
            .await
            .unwrap();
        }
        let outcome = ns.compact(&TINY_CAP).await.unwrap();
        let Compaction::Published { segments, .. } = outcome else {
            panic!("expected Published, got {outcome:?}");
        };
        assert!(
            segments.len() > 1,
            "expected the merge to split, got {segments:?}"
        );

        // Every output segment is independently valid and, together, cover
        // the full range with no gaps.
        let (live_segments, folded) = live(&*b.store, "docs").await;
        assert_eq!(live_segments, segments);
        segment::validate_chain(&live_segments, folded).unwrap();

        // Every output segment, on its own, is under the cap.
        for id in &segments {
            let size = b
                .store
                .head(&segment::segment_path("docs", id))
                .await
                .unwrap()
                .size;
            eprintln!("segment {id}: {size} bytes");
        }
    }
}

#[tokio::test]
async fn a_query_after_a_multi_output_merge_matches_the_flat_scan() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", schema(DIM, Metric::L2))
            .await
            .unwrap();
        let mut all_docs = Vec::new();
        for i in 0..6 {
            let docs = docs_of(&format!("g{i}"), i as u64, 20);
            all_docs.extend(docs.clone());
            ns.upsert(docs).await.unwrap();
            ns.compact(&CompactionOptions {
                max_segments: 8,
                max_segment_bytes: TINY_CAP.max_segment_bytes,
            })
            .await
            .unwrap();
        }
        let outcome = ns.compact(&TINY_CAP).await.unwrap();
        assert!(matches!(outcome, Compaction::Published { .. }));

        let mut rng = Lcg(1);
        let query = rng.vector(DIM);
        let got: Vec<String> = sorted_ids(&ns.query(&query, all_docs.len() + 10).await.unwrap());
        let mut want = naive_top_k(Metric::L2, &query, &all_docs, all_docs.len() + 10);
        want.sort();
        assert_eq!(got, want);
    }
}

#[tokio::test]
async fn an_oversized_segment_is_rebalanced_even_under_max_segments() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", schema(DIM, Metric::L2))
            .await
            .unwrap();
        ns.upsert(docs_of("big", 1, 200)).await.unwrap();
        // A generous max_segments (8) and a tiny cap: the fold alone likely
        // already exceeds it, and there is only ever one live segment, well
        // under the count trigger.
        let opts = CompactionOptions {
            max_segments: 8,
            max_segment_bytes: 2_000,
        };
        let first = ns.compact(&opts).await.unwrap();
        let Compaction::Published {
            segments: first_segments,
            ..
        } = first
        else {
            panic!("expected Published, got {first:?}");
        };

        // A rerun with an empty tail and segment count under max_segments:
        // without the size trigger this would be Plan::Nothing forever.
        // With it, an oversized segment gets rebalanced.
        let second = ns.compact(&opts).await.unwrap();
        match second {
            Compaction::Published { segments, .. } => {
                assert!(
                    segments.len() >= first_segments.len(),
                    "a rebalance should not reduce coverage: {segments:?} vs {first_segments:?}"
                );
            }
            Compaction::Nothing => {
                // Only acceptable if the first compact already split
                // everything under the cap — then nothing is left oversized.
                assert!(
                    first_segments.len() > 1,
                    "expected either a rebalance or an already-split first run"
                );
            }
            Compaction::Superseded => panic!("no concurrent compactor in this test"),
        }
    }
}

#[tokio::test]
async fn a_fault_at_every_operation_of_a_multi_output_merge_converges() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let opts = CompactionOptions {
            max_segments: 1,
            max_segment_bytes: 2_000,
        };

        // An unfaulted dry run just to count operations.
        let probe = FaultyStore::new(b.store.clone());
        let probe_store: Arc<dyn ObjectStore> = probe.clone();
        let probe_ns = Namespace::create(probe_store, "docs", schema(DIM, Metric::L2))
            .await
            .unwrap();
        for i in 0..6 {
            probe_ns
                .upsert(docs_of(&format!("g{i}"), i as u64, 20))
                .await
                .unwrap();
        }
        probe.reset_ops();
        probe_ns.compact(&opts).await.unwrap();
        let total = probe.ops();
        assert!(total > 0);
        drop(probe_ns);

        for fault in [Fault::Generic, Fault::LostAck] {
            for n in 0..total {
                eprintln!("backend {}: {fault:?} at operation {n} of {total}", b.name);
                let store = FaultyStore::new(Arc::new(InMemory::new()));
                let dyn_store: Arc<dyn ObjectStore> = store.clone();
                let ns = Namespace::create(dyn_store.clone(), "docs", schema(DIM, Metric::L2))
                    .await
                    .unwrap();
                for i in 0..6 {
                    ns.upsert(docs_of(&format!("g{i}"), i as u64, 20))
                        .await
                        .unwrap();
                }
                let before = common::snapshot(&*store).await;

                store.fail_nth_op(n, fault);
                let _ = ns.compact(&opts).await;
                store.clear();
                drop(ns);

                assert_consistent(dyn_store.clone(), DIM, Metric::L2).await;
                assert_untouched(&*store, &before).await;

                let resumed = Namespace::open(dyn_store.clone(), "docs").await.unwrap();
                resumed.compact(&opts).await.ok();
                assert_consistent(dyn_store, DIM, Metric::L2).await;
            }
        }
    }
}

#[tokio::test]
async fn a_fold_writes_sparse_bin_only_when_a_document_has_one() {
    let store = common::memory();
    let ns = Namespace::create(store.clone(), "docs", schema(1, Metric::L2))
        .await
        .expect("create");
    ns.upsert(vec![Document {
        id: "a".into(),
        vector: vec![1.0],
        attrs: None,
        sparse: Some(vec![(3, 0.5)]),
    }])
    .await
    .expect("upsert");
    ns.compact(&DEFAULT).await.expect("compact");

    let id = segment::segment_id(1, 1);
    let sparse_path = itzma_core::sparse::sparse_path("docs", &id);
    assert!(
        store.get(&sparse_path).await.is_ok(),
        "sparse.bin must exist"
    );
}

#[tokio::test]
async fn a_fold_with_no_sparse_documents_writes_no_sparse_bin() {
    let store = common::memory();
    let ns = Namespace::create(store.clone(), "docs", schema(1, Metric::L2))
        .await
        .expect("create");
    ns.upsert(vec![doc("a", &[1.0])]).await.expect("upsert");
    ns.compact(&DEFAULT).await.expect("compact");

    let id = segment::segment_id(1, 1);
    let sparse_path = itzma_core::sparse::sparse_path("docs", &id);
    assert!(matches!(
        store.get(&sparse_path).await,
        Err(object_store::Error::NotFound { .. })
    ));
}

/// A merge rebuilds its layer from the segments it retires, whose `data.bin`
/// does not carry sparse vectors. The new segment's `sparse.bin` must still
/// hold them.
#[tokio::test]
async fn a_merge_keeps_the_sparse_vectors_of_the_segments_it_retires() {
    let opts = CompactionOptions {
        max_segments: 2,
        max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
    };
    let sparse_doc = |id: &str, term: u32| Document {
        id: id.into(),
        vector: vec![1.0],
        attrs: None,
        sparse: Some(vec![(term, 1.0)]),
    };
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", schema(1, Metric::L2))
            .await
            .unwrap();
        for (id, term) in [("a", 1), ("b", 2)] {
            ns.upsert(vec![sparse_doc(id, term)]).await.unwrap();
            ns.compact(&opts).await.unwrap();
        }
        ns.upsert(vec![sparse_doc("c", 3)]).await.unwrap();
        let merged = segment::segment_id(1, 3);
        assert!(matches!(
            ns.compact(&opts).await.unwrap(),
            Compaction::Published {
                merged_segments: 2,
                ..
            }
        ));
        assert_eq!(live(&*b.store, "docs").await.0, vec![merged]);

        for (term, want) in [(1, "a"), (2, "b"), (3, "c")] {
            let hits = ns
                .query_sparse(
                    &vec![(term, 1.0)],
                    5,
                    &itzma_core::SparseQueryOptions::default(),
                )
                .await
                .unwrap();
            assert_eq!(
                hits.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(),
                [want],
                "term {term} on {}",
                b.name
            );
        }
    }
}

/// Sizes of every `data.bin` under the namespace "docs", largest first.
async fn data_sizes(store: &dyn ObjectStore) -> Vec<u64> {
    let prefix = object_store::path::Path::from("docs/segments");
    let listed: Vec<_> = store.list(Some(&prefix)).try_collect().await.unwrap();
    let mut sizes: Vec<u64> = listed
        .iter()
        .filter(|m| m.location.filename() == Some("data.bin"))
        .map(|m| m.size)
        .collect();
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    sizes
}

/// `segments` folded segments of `rows` documents of `dim` values each, then
/// a tail of `tail` documents, on a handle with no budget of note.
async fn segments_and_tail(
    store: Arc<dyn ObjectStore>,
    dim: usize,
    segments: usize,
    rows: usize,
    tail: usize,
) -> Namespace {
    let ns = Namespace::create(store, "docs", schema(dim, Metric::L2))
        .await
        .unwrap();
    let fold = CompactionOptions {
        max_segments: segments + 1,
        ..DEFAULT
    };
    let mut rng = Lcg(37);
    for s in 0..segments {
        let docs = (0..rows)
            .map(|i| doc(&format!("s{s}-{i:05}"), &rng.vector(dim)))
            .collect();
        ns.upsert(docs).await.unwrap();
        assert!(matches!(
            ns.compact(&fold).await.unwrap(),
            Compaction::Published { .. }
        ));
    }
    if tail > 0 {
        let docs = (0..tail)
            .map(|i| doc(&format!("t-{i:05}"), &rng.vector(dim)))
            .collect();
        ns.upsert(docs).await.unwrap();
    }
    ns
}

/// Headroom over a segment's own bytes: opening one for ranged reads charges
/// a 64 KiB suffix guess and a 32-byte header before its size is known.
const OPEN_SLACK: u64 = 64 * 1024 + 32;

#[tokio::test]
async fn a_namespace_bigger_than_the_budget_merges() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let dim = 64;
        segments_and_tail(b.store.clone(), dim, 6, 300, 20).await;
        let sizes = data_sizes(&*b.store).await;
        let largest = sizes[0];
        let total: u64 = sizes.iter().sum();
        let tail = wal_listing_size(&*b.store, 7).await;
        let budget = 2 * largest + OPEN_SLACK + tail;
        assert!(
            total > budget,
            "the namespace must not fit: {total} vs {budget}"
        );

        let small = Namespace::open(b.store.clone(), "docs")
            .await
            .unwrap()
            .with_read_budget(budget);
        let one = CompactionOptions {
            max_segments: 1,
            max_segment_bytes: largest,
        };
        match small.compact(&one).await.unwrap() {
            Compaction::Published { segments, .. } => assert!(segments.len() >= 2),
            other => panic!("expected a merge, got {other:?}"),
        }
        assert_consistent(b.store.clone(), dim, Metric::L2).await;
    }
}

/// The stored size of WAL object `seq` of "docs".
async fn wal_listing_size(store: &dyn ObjectStore, seq: u64) -> u64 {
    store.head(&wal::wal_path("docs", seq)).await.unwrap().size
}

#[tokio::test]
async fn one_group_over_the_budget_publishes_nothing() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let dim = 64;
        segments_and_tail(b.store.clone(), dim, 1, 1000, 20).await;
        let budget = data_sizes(&*b.store).await[0] / 2 + OPEN_SLACK;
        let before = common::snapshot(&*b.store).await;
        let small = Namespace::open(b.store.clone(), "docs")
            .await
            .unwrap()
            .with_read_budget(budget)
            .with_fetch_policy(itzma_core::FetchPolicy::local());
        let one = CompactionOptions {
            max_segments: 1,
            max_segment_bytes: budget,
        };
        let err = small.compact(&one).await.unwrap_err();
        assert!(
            matches!(err, ItzmaError::ReadBudgetExceeded { .. }),
            "{err:?}"
        );
        assert_untouched(&*b.store, &before).await;
    }
}

#[tokio::test]
async fn a_refused_group_publishes_nothing_and_a_rerun_with_room_converges() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let dim = 64;
        let ns = segments_and_tail(b.store.clone(), dim, 2, 1000, 0).await;
        let mut rng = Lcg(99);
        let big = (0..3000)
            .map(|i| doc(&format!("u-{i:05}"), &rng.vector(dim)))
            .collect();
        ns.upsert(big).await.unwrap();
        let fold = CompactionOptions {
            max_segments: 4,
            ..DEFAULT
        };
        ns.compact(&fold).await.unwrap();
        ns.upsert(vec![doc("z", &rng.vector(dim))]).await.unwrap();

        let sizes = data_sizes(&*b.store).await;
        let small_segment = *sizes.last().unwrap();
        let budget = small_segment + OPEN_SLACK + 4096;
        let one = CompactionOptions {
            max_segments: 1,
            max_segment_bytes: budget,
        };
        let live_before = live(&*b.store, "docs").await;
        let small = Namespace::open(b.store.clone(), "docs")
            .await
            .unwrap()
            .with_read_budget(budget)
            .with_fetch_policy(itzma_core::FetchPolicy::local());
        let err = small.compact(&one).await.unwrap_err();
        // Each group is one unit here, and a group of one segment writes the
        // segment it read (same range, same bytes), so the first two groups
        // pass and the third, the big segment, is refused on its own.
        assert!(
            matches!(&err, ItzmaError::ReadBudgetExceeded { path, .. }
                if path.contains(&segment::segment_id(3, 3))),
            "{err:?}"
        );
        assert_eq!(live(&*b.store, "docs").await, live_before);
        assert_eq!(data_sizes(&*b.store).await.len(), sizes.len());

        match ns.compact(&one).await.unwrap() {
            Compaction::Published { segments, .. } => assert_eq!(segments.len(), 4),
            other => panic!("expected a merge, got {other:?}"),
        }
        assert_eq!(data_sizes(&*b.store).await.len(), sizes.len() + 1);
        assert_consistent(b.store.clone(), dim, Metric::L2).await;
    }
}

/// Two folded segments of documents with sparse vectors and a tail, and a
/// merge into one whose publish fails, leaving its outputs as orphans.
async fn orphaned_merge(inner: Arc<InMemory>) -> (Arc<FaultyStore>, Namespace, String) {
    let store = FaultyStore::new(inner);
    let ns = Namespace::create(store.clone(), "docs", schema(2, Metric::L2))
        .await
        .unwrap();
    let fold = CompactionOptions {
        max_segments: 4,
        ..DEFAULT
    };
    for s in 0..2u32 {
        let docs = (0..300u32)
            .map(|i| {
                let mut d = doc(&format!("s{s}-{i:04}"), &[i as f32, s as f32]);
                d.sparse = Some(vec![(i, 1.0), (i + 1, 0.5)]);
                d
            })
            .collect();
        ns.upsert(docs).await.unwrap();
        ns.compact(&fold).await.unwrap();
    }
    ns.upsert(vec![doc("tail", &[0.0, 0.0])]).await.unwrap();
    store.fail_puts(
        "manifest/current.json",
        PutKind::Update,
        Fault::Generic,
        Some(1),
    );
    let one = CompactionOptions {
        max_segments: 1,
        ..DEFAULT
    };
    ns.compact(&one).await.unwrap_err();
    (store, ns, segment::segment_id(1, 3))
}

#[tokio::test]
async fn a_rerun_reuses_its_orphan_after_a_streamed_compare() {
    let inner = Arc::new(InMemory::new());
    let (store, ns, id) = orphaned_merge(inner.clone()).await;
    inner
        .head(&itzma_core::sparse::sparse_path("docs", &id))
        .await
        .expect("the orphan's sparse.bin stays");
    let one = CompactionOptions {
        max_segments: 1,
        ..DEFAULT
    };
    match ns.compact(&one).await.unwrap() {
        Compaction::Published { segments, .. } => assert_eq!(segments, vec![id]),
        other => panic!("expected a merge, got {other:?}"),
    }
    assert_consistent(store, 2, Metric::L2).await;
}

#[tokio::test]
async fn a_differing_orphan_is_refused_piece_by_piece() {
    let one = CompactionOptions {
        max_segments: 1,
        ..DEFAULT
    };
    for file in ["data.bin", "sparse.bin"] {
        for truncate in [false, true] {
            let inner = Arc::new(InMemory::new());
            let (store, ns, id) = orphaned_merge(inner.clone()).await;
            let path = object_store::path::Path::from(format!("docs/segments/{id}/{file}"));
            let mut bytes = inner
                .get(&path)
                .await
                .unwrap()
                .bytes()
                .await
                .unwrap()
                .to_vec();
            if truncate {
                bytes.pop();
            } else {
                let at = bytes.len() / 2;
                bytes[at] ^= 1;
            }
            inner.put(&path, bytes.into()).await.unwrap();
            let live_before = live(&*store, "docs").await;
            let err = ns.compact(&one).await.unwrap_err();
            assert!(
                matches!(&err, ItzmaError::CorruptSegment { reason, .. } if reason.contains("differs")),
                "{file}, truncated {truncate}: {err:?}"
            );
            assert_eq!(live(&*store, "docs").await, live_before);
        }
    }
}

#[tokio::test]
async fn a_fold_runs_on_a_handle_whose_budget_is_below_the_cap() {
    let store = Arc::new(InMemory::new());
    let ns = Namespace::create(store.clone(), "docs", schema(2, Metric::L2))
        .await
        .unwrap();
    ns.upsert(vec![doc("a", &[1.0, 0.0])]).await.unwrap();
    let small = Namespace::open(store.clone(), "docs")
        .await
        .unwrap()
        .with_read_budget(1000);
    assert!(matches!(
        small.compact(&DEFAULT).await.unwrap(),
        Compaction::Published { .. }
    ));
}

#[tokio::test]
async fn a_merge_with_a_cap_above_the_budget_is_refused_before_any_segment_is_read() {
    let inner = Arc::new(InMemory::new());
    let store = FaultyStore::new(inner.clone());
    let ns = Namespace::create(store.clone(), "docs", schema(2, Metric::L2))
        .await
        .unwrap();
    ns.upsert(vec![doc("a", &[1.0, 0.0])]).await.unwrap();
    ns.compact(&DEFAULT).await.unwrap();
    ns.upsert(vec![doc("b", &[2.0, 0.0])]).await.unwrap();
    let before = common::snapshot(&*inner).await;
    let small = Namespace::open(store.clone(), "docs")
        .await
        .unwrap()
        .with_read_budget(1000);
    let one = CompactionOptions {
        max_segments: 1,
        ..DEFAULT
    };
    let err = small.compact(&one).await.unwrap_err();
    assert!(matches!(err, ItzmaError::InvalidConfig(_)), "{err:?}");
    assert_untouched(&*inner, &before).await;
}

#[tokio::test]
async fn a_streamed_merge_keeps_sparse_vectors() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        // Segments over 1 MiB, so the merge reads them in windows rather
        // than whole.
        let dim = 256;
        let ns = Namespace::create(b.store.clone(), "docs", schema(dim, Metric::L2))
            .await
            .unwrap();
        let roomy = CompactionOptions {
            max_segments: 8,
            ..DEFAULT
        };
        let mut rng = Lcg(7);
        for s in 0..3u32 {
            let docs = (0..1200u32)
                .map(|i| {
                    let mut d = doc(&format!("d{i:05}"), &rng.vector(dim));
                    d.sparse = (i % 3 != s).then(|| vec![(i + 100, 1.0 + s as f32), (i % 7, 0.5)]);
                    d
                })
                .collect();
            ns.upsert(docs).await.unwrap();
            ns.compact(&roomy).await.unwrap();
        }
        ns.delete((0..1200).step_by(5).map(|i| format!("d{i:05}")).collect())
            .await
            .unwrap();

        let queries: Vec<Vec<(u32, f32)>> =
            vec![vec![(103, 1.0)], vec![(110, 1.0), (2, 0.5)], vec![(0, 1.0)]];
        let mut before = Vec::new();
        for q in &queries {
            before.push(
                ns.query_sparse(q, 20, &itzma_core::SparseQueryOptions::default())
                    .await
                    .unwrap(),
            );
        }
        let largest = data_sizes(&*b.store).await[0];
        assert!(largest > 1024 * 1024, "{largest}");
        let merge = CompactionOptions {
            max_segments: 1,
            max_segment_bytes: largest + largest / 2,
        };
        match ns.compact(&merge).await.unwrap() {
            Compaction::Published { segments, .. } => assert!(segments.len() >= 2),
            other => panic!("expected a merge, got {other:?}"),
        }
        for (q, want) in queries.iter().zip(&before) {
            let got = ns
                .query_sparse(q, 20, &itzma_core::SparseQueryOptions::default())
                .await
                .unwrap();
            assert_eq!(&got, want, "query {q:?}");
        }
        assert_consistent(b.store.clone(), dim, Metric::L2).await;
    }
}

/// A merge of version 3 segments over 1 MiB reads them in windows
/// like version 2 ones, rebuilds their typed attributes and writes version 3
/// again, equal to a rebuild from the WAL.
#[tokio::test]
async fn a_merge_of_version_3_segments_streams_and_matches_a_replay() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let dim = 256;
        let typed = schema(dim, Metric::L2).with_attrs(
            itzma_core::AttrSchema::new()
                .field("n", itzma_core::AttrType::Number)
                .field("tag", itzma_core::AttrType::Text)
                .field("tags", itzma_core::AttrType::TextSet),
        );
        let ns = Namespace::create(b.store.clone(), "docs", typed)
            .await
            .unwrap();
        ns.upgrade_segment_format(3).await.unwrap();
        let roomy = CompactionOptions {
            max_segments: 8,
            ..DEFAULT
        };
        let mut rng = Lcg(5);
        for s in 0..3u32 {
            let docs = (0..1200u32)
                .map(|i| {
                    let mut d = doc(&format!("d{i:05}"), &rng.vector(dim));
                    d.attrs = match (i + s) % 4 {
                        0 => None,
                        1 => Some(serde_json::json!({})),
                        2 => Some(serde_json::json!({"n": null, "tag": format!("t{s}")})),
                        _ => Some(serde_json::json!({
                            "n": i * 10 + s,
                            "tag": format!("t{}", i % 5),
                            "tags": ["a", format!("b{s}")],
                        })),
                    };
                    d
                })
                .collect();
            ns.upsert(docs).await.unwrap();
            ns.compact(&roomy).await.unwrap();
        }
        ns.delete((0..1200).step_by(7).map(|i| format!("d{i:05}")).collect())
            .await
            .unwrap();
        let largest = data_sizes(&*b.store).await[0];
        assert!(largest > 1024 * 1024, "{largest}");
        let merge = CompactionOptions {
            max_segments: 1,
            max_segment_bytes: largest + largest / 2,
        };
        match ns.compact(&merge).await.unwrap() {
            Compaction::Published { segments, .. } => assert!(segments.len() >= 2),
            other => panic!("expected a merge, got {other:?}"),
        }
        let versions = common::live_segment_versions(&*b.store).await;
        assert!(versions.iter().all(|v| *v == 3), "{versions:?}");
        assert_consistent(b.store.clone(), dim, Metric::L2).await;
    }
}
