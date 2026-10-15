mod common;

use itzma_core::wal::{self, WriteKey};
use itzma_core::{ItzmaError, WalOp};
use object_store::{ObjectStore, ObjectStoreExt};

fn upsert(id: &str) -> Vec<WalOp> {
    vec![WalOp::Upsert(common::doc(id, &[1.0, 2.0]))]
}

#[tokio::test]
async fn append_assigns_sequential_seqs() {
    for b in common::backends() {
        eprintln!("backend: {}", b.name);
        assert_eq!(
            wal::append(&*b.store, "ns", 1, upsert("a")).await.unwrap(),
            1
        );
        assert_eq!(
            wal::append(&*b.store, "ns", 2, upsert("b")).await.unwrap(),
            2
        );
        assert_eq!(
            wal::append(&*b.store, "ns", 3, upsert("c")).await.unwrap(),
            3
        );
    }
}

#[tokio::test]
async fn stale_hint_resyncs_to_next_free_seq() {
    for b in common::backends() {
        eprintln!("backend: {}", b.name);
        for id in ["a", "b", "c"] {
            wal::append(&*b.store, "ns", 1, upsert(id)).await.unwrap();
        }
        // Hint 1 is long taken; append must land on 4.
        assert_eq!(
            wal::append(&*b.store, "ns", 1, upsert("d")).await.unwrap(),
            4
        );
    }
}

#[tokio::test]
async fn hint_zero_is_clamped_to_one() {
    for b in common::backends() {
        eprintln!("backend: {}", b.name);
        assert_eq!(
            wal::append(&*b.store, "ns", 0, upsert("a")).await.unwrap(),
            1
        );
    }
}

#[tokio::test]
async fn empty_batch_is_rejected() {
    let store = common::memory();
    let err = wal::append(&*store, "ns", 1, vec![]).await.unwrap_err();
    assert!(matches!(err, ItzmaError::EmptyBatch));
}

#[tokio::test]
async fn read_log_returns_entries_in_seq_order() {
    for b in common::backends() {
        eprintln!("backend: {}", b.name);
        for id in ["a", "b", "c"] {
            wal::append(&*b.store, "ns", 1, upsert(id)).await.unwrap();
        }
        let log = wal::read_log(&*b.store, "ns", 0).await.unwrap();
        let seqs: Vec<u64> = log.iter().map(|(s, _)| *s).collect();
        assert_eq!(seqs, [1, 2, 3]);
        assert_eq!(log[1].1.ops, upsert("b"));
    }
}

#[tokio::test]
async fn read_log_of_empty_namespace_is_empty() {
    for b in common::backends() {
        eprintln!("backend: {}", b.name);
        assert!(wal::read_log(&*b.store, "ns", 0).await.unwrap().is_empty());
    }
}

#[tokio::test]
async fn read_log_skips_folded_entries() {
    for b in common::backends() {
        eprintln!("backend: {}", b.name);
        for id in ["a", "b", "c"] {
            wal::append(&*b.store, "ns", 1, upsert(id)).await.unwrap();
        }
        let log = wal::read_log(&*b.store, "ns", 1).await.unwrap();
        let seqs: Vec<u64> = log.iter().map(|(s, _)| *s).collect();
        assert_eq!(seqs, [2, 3]);
    }
}

#[tokio::test]
async fn read_log_detects_a_gap_in_the_middle() {
    for b in common::backends() {
        eprintln!("backend: {}", b.name);
        for id in ["a", "b", "c"] {
            wal::append(&*b.store, "ns", 1, upsert(id)).await.unwrap();
        }
        b.store.delete(&wal::wal_path("ns", 2)).await.unwrap();
        let err = wal::read_log(&*b.store, "ns", 0).await.unwrap_err();
        assert!(
            matches!(
                err,
                ItzmaError::WalGap {
                    expected: 2,
                    found: 3
                }
            ),
            "unexpected error: {err:?}"
        );
    }
}

#[tokio::test]
async fn read_log_detects_a_missing_first_entry() {
    for b in common::backends() {
        eprintln!("backend: {}", b.name);
        for id in ["a", "b"] {
            wal::append(&*b.store, "ns", 1, upsert(id)).await.unwrap();
        }
        b.store.delete(&wal::wal_path("ns", 1)).await.unwrap();
        let err = wal::read_log(&*b.store, "ns", 0).await.unwrap_err();
        assert!(matches!(
            err,
            ItzmaError::WalGap {
                expected: 1,
                found: 2
            }
        ));
    }
}

#[tokio::test]
async fn read_log_rejects_unknown_format_version() {
    let store = common::memory();
    wal::append(&*store, "ns", 1, upsert("a")).await.unwrap();
    let path = wal::wal_path("ns", 1);
    let mut bad = store
        .get(&path)
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap()
        .to_vec();
    bad[4..8].copy_from_slice(&999u32.to_le_bytes());
    store.put(&path, bad.into()).await.unwrap();
    let err = wal::read_log(&*store, "ns", 0).await.unwrap_err();
    assert!(matches!(
        err,
        ItzmaError::UnsupportedFormat { version: 999 }
    ));
}

async fn raw(store: &dyn ObjectStore, seq: u64) -> Vec<u8> {
    store
        .get(&wal::wal_path("ns", seq))
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap()
        .to_vec()
}

#[tokio::test]
async fn appended_objects_are_binary_and_keep_the_json_key() {
    for b in common::backends() {
        eprintln!("backend: {}", b.name);
        wal::append(&*b.store, "ns", 1, upsert("a")).await.unwrap();
        let bytes = raw(&*b.store, 1).await;
        assert_eq!(&bytes[..4], b"ITZW");
        assert!(
            wal::wal_path("ns", 1).as_ref().ends_with(".json"),
            "the key must stay .json so put-if-absent and old listings still work"
        );
    }
}

#[tokio::test]
async fn read_log_rejects_an_unknown_binary_version() {
    let store = common::memory();
    wal::append(&*store, "ns", 1, upsert("a")).await.unwrap();
    let mut bytes = raw(&*store, 1).await;
    bytes[4..8].copy_from_slice(&(wal::FORMAT_VERSION + 1).to_le_bytes());
    store
        .put(&wal::wal_path("ns", 1), bytes.into())
        .await
        .unwrap();
    let err = wal::read_log(&*store, "ns", 0).await.unwrap_err();
    assert!(
        matches!(err, ItzmaError::UnsupportedFormat { version } if version == wal::FORMAT_VERSION + 1),
        "{err:?}"
    );
}

#[tokio::test]
async fn a_damaged_binary_object_is_corrupt_wal_naming_the_key() {
    let store = common::memory();
    wal::append(&*store, "ns", 1, upsert("a")).await.unwrap();
    let bytes = raw(&*store, 1).await;
    store
        .put(
            &wal::wal_path("ns", 1),
            bytes[..bytes.len() - 3].to_vec().into(),
        )
        .await
        .unwrap();
    match wal::read_log(&*store, "ns", 0).await.unwrap_err() {
        ItzmaError::CorruptWal { path, .. } => {
            assert!(path.contains("wal/00000000000000000001"), "{path}");
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[tokio::test]
async fn an_object_that_is_not_a_wal_object_is_corrupt_naming_the_key() {
    let store = common::memory();
    wal::append(&*store, "ns", 1, upsert("a")).await.unwrap();
    store
        .put(
            &wal::wal_path("ns", 1),
            "{\"format_version\": 1, \"ops\": [".into(),
        )
        .await
        .unwrap();
    match wal::read_log(&*store, "ns", 0).await.unwrap_err() {
        ItzmaError::CorruptWal { path, .. } => {
            assert!(path.contains("wal/00000000000000000001"), "{path}");
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[tokio::test]
async fn vectors_of_different_lengths_are_refused_and_nothing_is_written() {
    let store = common::memory();
    let before = common::snapshot(&*store).await;
    let ops = vec![
        WalOp::Upsert(common::doc("a", &[1.0, 2.0])),
        WalOp::Upsert(common::doc("b", &[1.0])),
    ];
    let err = wal::append(&*store, "ns", 1, ops).await.unwrap_err();
    assert!(matches!(
        err,
        ItzmaError::DimensionMismatch {
            expected: 2,
            got: 1
        }
    ));
    assert_eq!(common::snapshot(&*store).await, before);
}

fn two_upserts() -> Vec<WalOp> {
    vec![
        WalOp::Upsert(common::doc("a", &[1.0, 2.0])),
        WalOp::Upsert(common::doc("b", &[3.0, 4.0])),
    ]
}

/// Each entry's version and key, in seq order.
async fn versions_and_keys(store: &dyn ObjectStore) -> Vec<(u32, Option<String>)> {
    wal::read_log(store, "ns", 0)
        .await
        .unwrap()
        .into_iter()
        .map(|(_, entry)| (entry.format_version, entry.key))
        .collect()
}

#[tokio::test]
async fn a_first_attempt_writes_one_version_3_object() {
    for b in common::backends() {
        eprintln!("backend: {}", b.name);
        let write = WriteKey::new("load-1");
        let seq = wal::append_keyed(&*b.store, "ns", 1, &write, upsert("a"))
            .await
            .unwrap();
        assert_eq!(seq, 1);
        let log = wal::read_log(&*b.store, "ns", 0).await.unwrap();
        assert_eq!(log.len(), 1);
        assert_eq!(log[0].1.format_version, 3);
        assert_eq!(log[0].1.key.as_deref(), Some("load-1"));
        assert_eq!(log[0].1.ops, upsert("a"));
    }
}

#[tokio::test]
async fn keyed_and_unkeyed_writes_share_one_log() {
    for b in common::backends() {
        eprintln!("backend: {}", b.name);
        wal::append(&*b.store, "ns", 1, upsert("a")).await.unwrap();
        let write = WriteKey::new("load-1");
        wal::append_keyed(&*b.store, "ns", 2, &write, upsert("b"))
            .await
            .unwrap();
        wal::append(&*b.store, "ns", 3, upsert("c")).await.unwrap();
        assert_eq!(
            versions_and_keys(&*b.store).await,
            [(2, None), (3, Some("load-1".into())), (2, None)]
        );
    }
}

#[tokio::test]
async fn a_retry_finds_the_original_and_writes_nothing() {
    for b in common::backends() {
        eprintln!("backend: {}", b.name);
        let seq = wal::append_keyed(&*b.store, "ns", 1, &WriteKey::new("k"), upsert("a"))
            .await
            .unwrap();
        let before = common::snapshot(&*b.store).await;

        let retry = WriteKey::retry("k", 1);
        let again = wal::append_keyed(&*b.store, "ns", 1, &retry, upsert("a"))
            .await
            .unwrap();
        assert_eq!(again, seq);
        assert_eq!(common::snapshot(&*b.store).await, before);
    }
}

#[tokio::test]
async fn a_retry_finds_the_original_behind_later_writes_whatever_its_hint() {
    for b in common::backends() {
        eprintln!("backend: {}", b.name);
        wal::append_keyed(&*b.store, "ns", 1, &WriteKey::new("k"), upsert("a"))
            .await
            .unwrap();
        for (hint, id) in [(2, "b"), (3, "c"), (4, "d")] {
            wal::append(&*b.store, "ns", hint, upsert(id))
                .await
                .unwrap();
        }
        let before = common::snapshot(&*b.store).await;

        // A hint past the end says nothing about where the original is.
        let retry = WriteKey::retry("k", 1);
        let again = wal::append_keyed(&*b.store, "ns", 5, &retry, upsert("a"))
            .await
            .unwrap();
        assert_eq!(again, 1);
        assert_eq!(common::snapshot(&*b.store).await, before);
    }
}

/// The documented window: a search that starts past the original does not
/// see it, and the batch is appended a second time.
#[tokio::test]
async fn a_retry_that_starts_past_the_original_appends_a_copy() {
    for b in common::backends() {
        eprintln!("backend: {}", b.name);
        wal::append_keyed(&*b.store, "ns", 1, &WriteKey::new("k"), upsert("a"))
            .await
            .unwrap();
        wal::append(&*b.store, "ns", 2, upsert("b")).await.unwrap();

        let retry = WriteKey::retry("k", 2);
        let seq = wal::append_keyed(&*b.store, "ns", 3, &retry, upsert("a"))
            .await
            .unwrap();
        assert_eq!(seq, 3);
        assert_eq!(
            versions_and_keys(&*b.store).await,
            [(3, Some("k".into())), (2, None), (3, Some("k".into()))]
        );
    }
}

/// A first attempt does not go looking. Only a collision on its own put makes
/// it check the object it collided with.
#[tokio::test]
async fn a_first_attempt_that_lands_elsewhere_appends_normally() {
    for b in common::backends() {
        eprintln!("backend: {}", b.name);
        wal::append_keyed(&*b.store, "ns", 1, &WriteKey::new("k"), upsert("a"))
            .await
            .unwrap();
        // Hint 2 is free, so the put succeeds and nothing is compared.
        let seq = wal::append_keyed(&*b.store, "ns", 2, &WriteKey::new("k"), upsert("a"))
            .await
            .unwrap();
        assert_eq!(seq, 2);
    }
}

#[tokio::test]
async fn a_put_that_collides_with_its_own_object_returns_it() {
    for b in common::backends() {
        eprintln!("backend: {}", b.name);
        let write = WriteKey::new("k");
        wal::append_keyed(&*b.store, "ns", 1, &write, upsert("a"))
            .await
            .unwrap();
        let before = common::snapshot(&*b.store).await;

        // The stale hint runs into the object a lost acknowledgement left.
        let seq = wal::append_keyed(&*b.store, "ns", 1, &write, upsert("a"))
            .await
            .unwrap();
        assert_eq!(seq, 1);
        assert_eq!(common::snapshot(&*b.store).await, before);
    }
}

#[tokio::test]
async fn a_put_that_collides_with_another_object_moves_on() {
    for b in common::backends() {
        eprintln!("backend: {}", b.name);
        wal::append_keyed(&*b.store, "ns", 1, &WriteKey::new("one"), upsert("a"))
            .await
            .unwrap();
        wal::append(&*b.store, "ns", 2, upsert("b")).await.unwrap();
        let seq = wal::append_keyed(&*b.store, "ns", 1, &WriteKey::new("two"), upsert("c"))
            .await
            .unwrap();
        assert_eq!(seq, 3);
    }
}

#[tokio::test]
async fn the_same_key_with_a_different_batch_is_a_conflict_and_writes_nothing() {
    for b in common::backends() {
        eprintln!("backend: {}", b.name);
        wal::append_keyed(&*b.store, "ns", 1, &WriteKey::new("k"), upsert("a"))
            .await
            .unwrap();
        let before = common::snapshot(&*b.store).await;

        for write in [WriteKey::retry("k", 1), WriteKey::new("k")] {
            let err = wal::append_keyed(&*b.store, "ns", 1, &write, two_upserts())
                .await
                .unwrap_err();
            assert!(
                matches!(&err, ItzmaError::KeyConflict { key } if key == "k"),
                "{err:?}"
            );
            assert_eq!(common::snapshot(&*b.store).await, before);
        }
    }
}

#[tokio::test]
async fn a_bad_key_or_an_empty_batch_is_refused_before_any_storage_call() {
    let store = common::memory();
    for key in ["".to_string(), "k".repeat(129)] {
        let err = wal::append_keyed(&*store, "ns", 1, &WriteKey::new(key), upsert("a"))
            .await
            .unwrap_err();
        assert!(matches!(err, ItzmaError::InvalidKey(_)), "{err:?}");
    }
    let err = wal::append_keyed(&*store, "ns", 1, &WriteKey::new("k"), vec![])
        .await
        .unwrap_err();
    assert!(matches!(err, ItzmaError::EmptyBatch));
    assert!(common::snapshot(&*store).await.is_empty());
}

#[test]
fn the_writer_limit_is_64_mib() {
    assert_eq!(wal::MAX_BODY_BYTES, 64 * 1024 * 1024);
}

#[tokio::test]
async fn a_batch_over_the_writer_limit_is_refused_and_the_store_is_untouched() {
    let store = common::memory();
    wal::append(&*store, "ns", 1, upsert("a")).await.unwrap();
    let before = common::snapshot(&*store).await;

    // 17 million floats is 68 MB of vector, over 64 MiB (67.1 MB).
    let big = vec![WalOp::Upsert(common::doc("big", &vec![0.5; 17_000_000]))];
    let err = wal::append(&*store, "ns", 2, big).await.unwrap_err();
    match err {
        ItzmaError::BatchTooLarge { bytes, max } => {
            assert_eq!(max, 64 * 1024 * 1024);
            assert!(bytes > max, "{bytes}");
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(common::snapshot(&*store).await, before);
}
