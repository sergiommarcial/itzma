mod common;

use common::{cas_backends, doc, publish, put_segment, schema};
use itzma_core::layer::Layer;
use itzma_core::{Hit, ItzmaError, Metric, Namespace, WalOp, manifest, segment};
use object_store::ObjectStoreExt;

fn up(id: &str, v: &[f32]) -> WalOp {
    WalOp::Upsert(doc(id, v))
}

fn ids(hits: &[Hit]) -> Vec<&str> {
    hits.iter().map(|h| h.id.as_str()).collect()
}

#[tokio::test]
async fn query_reads_segments_then_the_wal_tail() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        Namespace::create(b.store.clone(), "docs", schema(1, Metric::L2))
            .await
            .unwrap();
        let base = put_segment(
            &*b.store,
            "docs",
            1,
            2,
            1,
            vec![up("a", &[1.0]), up("b", &[2.0])],
        )
        .await;
        publish(&*b.store, "docs", &[base.as_str()], 2).await;

        // A handle opened after publishing resumes the WAL at seq 3.
        let ns = Namespace::open(b.store.clone(), "docs").await.unwrap();
        assert_eq!(ns.upsert(vec![doc("c", &[3.0])]).await.unwrap(), 3);
        let hits = ns.query(&[0.0], 10).await.unwrap();
        assert_eq!(ids(&hits), ["a", "b", "c"]);
    }
}

#[tokio::test]
async fn a_delta_tombstone_hides_a_document_in_an_older_segment() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        Namespace::create(b.store.clone(), "docs", schema(1, Metric::L2))
            .await
            .unwrap();
        let base = put_segment(
            &*b.store,
            "docs",
            1,
            2,
            1,
            vec![up("a", &[1.0]), up("b", &[2.0])],
        )
        .await;
        let delta = put_segment(
            &*b.store,
            "docs",
            3,
            3,
            1,
            vec![WalOp::Delete("a".into()), up("c", &[3.0])],
        )
        .await;
        publish(&*b.store, "docs", &[base.as_str(), delta.as_str()], 3).await;

        let ns = Namespace::open(b.store.clone(), "docs").await.unwrap();
        let hits = ns.query(&[0.0], 10).await.unwrap();
        assert_eq!(ids(&hits), ["b", "c"]);
    }
}

#[tokio::test]
async fn a_newer_segment_replaces_an_older_document() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        Namespace::create(b.store.clone(), "docs", schema(1, Metric::L2))
            .await
            .unwrap();
        let base = put_segment(&*b.store, "docs", 1, 2, 1, vec![up("a", &[100.0])]).await;
        let delta = put_segment(&*b.store, "docs", 3, 3, 1, vec![up("a", &[1.0])]).await;
        publish(&*b.store, "docs", &[base.as_str(), delta.as_str()], 3).await;

        let ns = Namespace::open(b.store.clone(), "docs").await.unwrap();
        let hits = ns.query(&[0.0], 10).await.unwrap();
        assert_eq!(ids(&hits), ["a"]);
        assert_eq!(hits[0].distance, 1.0);
    }
}

#[tokio::test]
async fn the_wal_tail_overrides_segments() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        Namespace::create(b.store.clone(), "docs", schema(1, Metric::L2))
            .await
            .unwrap();
        let base = put_segment(
            &*b.store,
            "docs",
            1,
            2,
            1,
            vec![up("a", &[100.0]), up("b", &[1.0])],
        )
        .await;
        publish(&*b.store, "docs", &[base.as_str()], 2).await;

        let ns = Namespace::open(b.store.clone(), "docs").await.unwrap();
        ns.upsert(vec![doc("a", &[0.5])]).await.unwrap();
        ns.delete(vec!["b".into()]).await.unwrap();
        let hits = ns.query(&[0.0], 10).await.unwrap();
        assert_eq!(ids(&hits), ["a"]);
        assert_eq!(hits[0].distance, 0.25);
    }
}

#[tokio::test]
async fn a_broken_segment_chain_is_a_corrupt_manifest() {
    let cases: [(&[(u64, u64)], u64); 5] = [
        (&[(1, 2), (4, 5)], 5), // gap
        (&[(1, 3), (3, 4)], 4), // overlap
        (&[(2, 3)], 3),         // does not start at 1
        (&[(1, 3)], 4),         // last range ends before wal_folded_through
        (&[], 3),               // folded with no segments at all
    ];
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        for (i, (ranges, folded)) in cases.iter().enumerate() {
            let name = format!("case{i}");
            let ns = Namespace::create(b.store.clone(), &name, schema(1, Metric::L2))
                .await
                .unwrap();
            let list: Vec<String> = ranges
                .iter()
                .map(|(lo, hi)| segment::segment_id(*lo, *hi))
                .collect();
            manifest::update(&*b.store, &name, |m| {
                m.live_segments = list.clone();
                m.wal_folded_through = *folded;
            })
            .await
            .unwrap();
            let err = ns.query(&[0.0], 1).await.unwrap_err();
            assert!(
                matches!(err, ItzmaError::CorruptManifest(_)),
                "case {i}: {err:?}"
            );
        }
    }
}

#[tokio::test]
async fn a_listed_segment_that_is_missing_is_corrupt() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", schema(1, Metric::L2))
            .await
            .unwrap();
        let id = segment::segment_id(1, 2);
        publish(&*b.store, "docs", &[id.as_str()], 2).await;
        match ns.query(&[0.0], 1).await.unwrap_err() {
            ItzmaError::CorruptSegment { path, reason } => {
                assert!(path.contains(&id), "{path}");
                assert!(reason.contains("missing"), "{reason}");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}

#[tokio::test]
async fn a_damaged_segment_is_corrupt() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", schema(1, Metric::L2))
            .await
            .unwrap();
        let id = segment::segment_id(1, 2);
        b.store
            .put(&segment::segment_path("docs", &id), "garbage".into())
            .await
            .unwrap();
        publish(&*b.store, "docs", &[id.as_str()], 2).await;
        let err = ns.query(&[0.0], 1).await.unwrap_err();
        assert!(matches!(err, ItzmaError::CorruptSegment { .. }), "{err:?}");
    }
}

#[tokio::test]
async fn a_segment_with_the_wrong_dimension_is_corrupt() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", schema(2, Metric::L2))
            .await
            .unwrap();
        let id = put_segment(&*b.store, "docs", 1, 1, 3, vec![up("a", &[1.0, 2.0, 3.0])]).await;
        publish(&*b.store, "docs", &[id.as_str()], 1).await;
        match ns.query(&[0.0, 0.0], 1).await.unwrap_err() {
            ItzmaError::CorruptSegment { reason, .. } => assert!(reason.contains("dimension")),
            other => panic!("unexpected error: {other:?}"),
        }
    }
}

#[tokio::test]
async fn a_segment_with_an_unknown_version_is_unsupported() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", schema(1, Metric::L2))
            .await
            .unwrap();
        let id = segment::segment_id(1, 1);
        let mut bytes = segment::encode(&Layer::new(), 1).unwrap().to_vec();
        // 4: the first version this build does not read.
        bytes[4..8].copy_from_slice(&4u32.to_le_bytes());
        b.store
            .put(&segment::segment_path("docs", &id), bytes.into())
            .await
            .unwrap();
        publish(&*b.store, "docs", &[id.as_str()], 1).await;
        let err = ns.query(&[0.0], 1).await.unwrap_err();
        assert!(
            matches!(err, ItzmaError::UnsupportedFormat { version: 4 }),
            "{err:?}"
        );
    }
}
