//! Sizes that the small tests never reach.

mod common;

use std::time::Duration;

use common::{Lcg, assert_consistent, cas_backends, doc, naive_top_k, schema, wal_listing};
use itzma_core::{
    Compaction, CompactionOptions, GcOptions, GcReport, Metric, Namespace, segment, wal,
};

/// S3 lists 1000 keys a page. A WAL longer than that must come back whole and
/// in order, and compaction must fold all of it.
#[tokio::test]
async fn a_wal_longer_than_one_listing_page_reads_back_whole() {
    const WRITES: usize = 1_205;
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", schema(1, Metric::L2))
            .await
            .unwrap();
        for i in 0..WRITES {
            ns.upsert(vec![doc(&format!("d{i:05}"), &[i as f32])])
                .await
                .unwrap();
        }
        assert_eq!(
            wal::read_log(&*b.store, "docs", 0).await.unwrap().len(),
            WRITES
        );
        assert_eq!(ns.query(&[0.0], 5_000).await.unwrap().len(), WRITES);

        match ns.compact(&CompactionOptions::default()).await.unwrap() {
            Compaction::Published {
                folded_through,
                docs,
                ..
            } => {
                assert_eq!(folded_through, WRITES as u64);
                assert_eq!(docs, WRITES);
            }
            other => panic!("unexpected {other:?}"),
        }

        // A tail past the fold, listed from an offset inside the first page.
        for i in 0..3 {
            ns.upsert(vec![doc(&format!("t{i}"), &[0.0])])
                .await
                .unwrap();
        }
        assert_eq!(ns.query(&[0.0], 5_000).await.unwrap().len(), WRITES + 3);
    }
}

/// Fencing lists the whole WAL prefix (there is no `wal_start` yet), and S3
/// lists 1000 keys a page. A run must fence every folded object, including the
/// ones past the first page, leave the tail alone, and find nothing the second
/// time. A writer that starts from the front must still land after the end.
#[tokio::test]
async fn fencing_a_wal_longer_than_one_listing_page() {
    const WRITES: usize = 1_205;
    const TAIL: usize = 3;
    let fencing = GcOptions {
        min_age: Duration::ZERO,
        dry_run: false,
        fence_wal: true,
    };
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", schema(1, Metric::L2))
            .await
            .unwrap();
        for i in 0..WRITES {
            ns.upsert(vec![doc(&format!("d{i:05}"), &[i as f32])])
                .await
                .unwrap();
        }
        ns.compact(&CompactionOptions::default()).await.unwrap();
        for i in 0..TAIL {
            ns.upsert(vec![doc(&format!("t{i}"), &[0.0])])
                .await
                .unwrap();
        }

        let report = ns.gc(&fencing).await.unwrap();
        assert_eq!(report.wal_objects_fenced, WRITES);
        assert_eq!(report.wal_too_young, 0);
        let listing = wal_listing(&*b.store).await;
        assert_eq!(listing.len(), WRITES + TAIL);
        for (seq, size) in &listing {
            assert_eq!(*size == 8, *seq <= WRITES as u64, "seq {seq}, {size} bytes");
        }

        // The hint reaches past the first page, so the next run starts there.
        assert_eq!(common::fence_hint(&*b.store).await, Some(WRITES as u64));
        assert_eq!(ns.gc(&fencing).await.unwrap(), GcReport::default());
        assert_eq!(ns.query(&[0.0], 5_000).await.unwrap().len(), WRITES + TAIL);

        let fresh = Namespace::open(b.store.clone(), "docs").await.unwrap();
        let seq = fresh.upsert(vec![doc("late", &[0.0])]).await.unwrap();
        assert_eq!(seq, (WRITES + TAIL + 1) as u64);
        let docs = assert_consistent(b.store.clone(), 1, Metric::L2).await;
        assert_eq!(docs.len(), WRITES + TAIL + 1);
    }
}

/// 41 live segments, more than the 16 fetched at a time. The newest write to
/// an id must win, and a tombstone in a late delta must hide an early
/// document, however the fetches interleave.
#[tokio::test]
async fn more_segments_than_the_fetch_window_are_read_in_order() {
    let roomy = CompactionOptions {
        max_segments: 64,
        max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
    };
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", schema(1, Metric::L2))
            .await
            .unwrap();
        for i in 0..40 {
            ns.upsert(vec![
                doc("shared", &[i as f32]),
                doc(&format!("d{i:02}"), &[0.0]),
            ])
            .await
            .unwrap();
            ns.compact(&roomy).await.unwrap();
        }
        ns.delete(vec!["d00".into()]).await.unwrap();
        ns.compact(&roomy).await.unwrap();

        let check = |hits: Vec<itzma_core::Hit>| {
            assert_eq!(hits.len(), 40, "shared plus d01..d39");
            assert!(hits.iter().all(|h| h.id != "d00"));
            let shared = hits.iter().find(|h| h.id == "shared").unwrap();
            assert_eq!(shared.distance, 39.0 * 39.0, "the newest write wins");
        };
        check(ns.query(&[0.0], 100).await.unwrap());
        let (m, _) = itzma_core::manifest::load(&*b.store, "docs").await.unwrap();
        assert_eq!(m.live_segments.len(), 41);

        // Merge them all into one, and the answer does not change.
        ns.upsert(vec![doc("last", &[100.0])]).await.unwrap();
        let one = CompactionOptions {
            max_segments: 1,
            max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
        };
        assert!(matches!(
            ns.compact(&one).await.unwrap(),
            Compaction::Published {
                merged_segments: 41,
                ..
            }
        ));
        let (m, _) = itzma_core::manifest::load(&*b.store, "docs").await.unwrap();
        assert_eq!(m.live_segments, vec![segment::segment_id(1, 42)]);
        let mut hits = ns.query(&[0.0], 100).await.unwrap();
        hits.retain(|h| h.id != "last");
        check(hits);
    }
}

#[tokio::test]
async fn a_larger_corpus_compacts_and_answers_the_same() {
    const DIM: usize = 128;
    const DOCS: usize = 3_000;
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let mut rng = Lcg(5);
        let docs: Vec<_> = (0..DOCS)
            .map(|i| doc(&format!("d{i:04}"), &rng.vector(DIM)))
            .collect();
        let ns = Namespace::create(b.store.clone(), "docs", schema(DIM, Metric::L2))
            .await
            .unwrap();
        for chunk in docs.chunks(500) {
            ns.upsert(chunk.to_vec()).await.unwrap();
        }
        let queries: Vec<Vec<f32>> = (0..5).map(|_| rng.vector(DIM)).collect();
        let ids = |hits: Vec<itzma_core::Hit>| -> Vec<String> {
            hits.into_iter().map(|h| h.id).collect()
        };

        let mut before = Vec::new();
        for q in &queries {
            before.push(ids(ns.query(q, 10).await.unwrap()));
        }
        assert!(matches!(
            ns.compact(&CompactionOptions::default()).await.unwrap(),
            Compaction::Published { .. }
        ));
        for (q, expected) in queries.iter().zip(&before) {
            let after = ids(ns.query(q, 10).await.unwrap());
            assert_eq!(&after, expected, "compaction changed an answer");
            assert_eq!(after, naive_top_k(Metric::L2, q, &docs, 10));
        }
    }
}
