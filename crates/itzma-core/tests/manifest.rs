mod common;

use std::sync::Arc;

use common::faulty::{Fault, FaultyStore, PutKind};
use itzma_core::manifest::{self, Manifest};
use itzma_core::{ItzmaError, Metric};
use object_store::ObjectStoreExt;
use object_store::local::LocalFileSystem;
use object_store::memory::InMemory;

fn new_manifest() -> Manifest {
    Manifest::new(common::schema(3, Metric::Cosine))
}

#[tokio::test]
async fn create_then_load_roundtrips() {
    for b in common::backends() {
        eprintln!("backend: {}", b.name);
        manifest::create(&*b.store, "ns1", &new_manifest())
            .await
            .unwrap();
        let (loaded, _version) = manifest::load(&*b.store, "ns1").await.unwrap();
        assert_eq!(loaded, new_manifest());
    }
}

#[tokio::test]
async fn create_twice_is_namespace_exists() {
    for b in common::backends() {
        eprintln!("backend: {}", b.name);
        manifest::create(&*b.store, "ns1", &new_manifest())
            .await
            .unwrap();
        let err = manifest::create(&*b.store, "ns1", &new_manifest())
            .await
            .unwrap_err();
        assert!(matches!(err, ItzmaError::NamespaceExists(ref n) if n == "ns1"));
    }
}

#[tokio::test]
async fn load_missing_is_namespace_not_found() {
    for b in common::backends() {
        eprintln!("backend: {}", b.name);
        let err = manifest::load(&*b.store, "nope").await.unwrap_err();
        assert!(matches!(err, ItzmaError::NamespaceNotFound(ref n) if n == "nope"));
    }
}

#[tokio::test]
async fn update_applies_the_change() {
    for b in common::cas_backends() {
        eprintln!("backend: {}", b.name);
        manifest::create(&*b.store, "ns1", &new_manifest())
            .await
            .unwrap();
        let updated = manifest::update(&*b.store, "ns1", |m| m.wal_folded_through = 7)
            .await
            .unwrap();
        assert_eq!(updated.wal_folded_through, 7);
        let (loaded, _) = manifest::load(&*b.store, "ns1").await.unwrap();
        assert_eq!(loaded.wal_folded_through, 7);
    }
}

/// K updaters each add 1. If CAS is correct, no increment is lost or doubled.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_updates_are_not_lost() {
    const K: u64 = 8;
    for b in common::cas_backends() {
        eprintln!("backend: {}", b.name);
        manifest::create(&*b.store, "scratch", &new_manifest())
            .await
            .unwrap();

        let mut tasks = Vec::new();
        for _ in 0..K {
            let store = Arc::clone(&b.store);
            tasks.push(tokio::spawn(async move {
                manifest::update(&*store, "scratch", |m| m.wal_folded_through += 1)
                    .await
                    .unwrap();
            }));
        }
        for t in tasks {
            t.await.unwrap();
        }
        let (loaded, _) = manifest::load(&*b.store, "scratch").await.unwrap();
        assert_eq!(loaded.wal_folded_through, K);
    }
}

/// Records a known gap: LocalFileSystem has no conditional update, so
/// manifest CAS is exercised on InMemory and S3 (`cas_backends`), never on local disk.
#[tokio::test]
async fn update_is_unsupported_on_local_fs() {
    let dir = tempfile::TempDir::new().unwrap();
    let store = LocalFileSystem::new_with_prefix(dir.path()).unwrap();
    manifest::create(&store, "ns1", &new_manifest())
        .await
        .unwrap();
    let err = manifest::update(&store, "ns1", |m| m.wal_folded_through = 1)
        .await
        .unwrap_err();
    assert!(
        matches!(
            err,
            ItzmaError::Storage(object_store::Error::NotImplemented { .. })
        ),
        "unexpected error: {err:?}"
    );
}

#[tokio::test]
async fn update_if_publishes_when_the_closure_accepts() {
    for b in common::cas_backends() {
        eprintln!("backend: {}", b.name);
        manifest::create(&*b.store, "ns1", &new_manifest())
            .await
            .unwrap();
        let updated = manifest::update_if(&*b.store, "ns1", |m| {
            m.wal_folded_through = 5;
            true
        })
        .await
        .unwrap();
        assert_eq!(updated.map(|m| m.wal_folded_through), Some(5));
        let (loaded, _) = manifest::load(&*b.store, "ns1").await.unwrap();
        assert_eq!(loaded.wal_folded_through, 5);
    }
}

#[tokio::test]
async fn update_if_declines_without_writing() {
    for b in common::cas_backends() {
        eprintln!("backend: {}", b.name);
        manifest::create(&*b.store, "ns1", &new_manifest())
            .await
            .unwrap();
        let declined = manifest::update_if(&*b.store, "ns1", |m| {
            m.wal_folded_through = 5; // must not reach the store
            false
        })
        .await
        .unwrap();
        assert!(declined.is_none());
        let (loaded, _) = manifest::load(&*b.store, "ns1").await.unwrap();
        assert_eq!(loaded, new_manifest());
    }
}

#[tokio::test]
async fn update_if_reruns_the_closure_on_fresh_state_after_a_lost_race() {
    let inner = Arc::new(InMemory::new());
    let store = FaultyStore::new(inner);
    manifest::create(&*store, "ns1", &new_manifest())
        .await
        .unwrap();
    store.fail_puts(
        "manifest/current.json",
        PutKind::Update,
        Fault::Precondition,
        Some(1),
    );

    let mut runs = 0;
    let updated = manifest::update_if(&*store, "ns1", |m| {
        runs += 1;
        m.wal_folded_through += 1;
        true
    })
    .await
    .unwrap();
    assert_eq!(runs, 2);
    assert_eq!(updated.map(|m| m.wal_folded_through), Some(1));
}

#[tokio::test]
async fn a_new_manifest_carries_the_builds_wal_format_into_the_object() {
    assert_eq!(new_manifest().wal_format, itzma_core::wal::FORMAT_VERSION);
    for b in common::backends() {
        eprintln!("backend: {}", b.name);
        manifest::create(&*b.store, "ns1", &new_manifest())
            .await
            .unwrap();
        let raw = b
            .store
            .get(&manifest::manifest_path("ns1"))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        assert_eq!(json["wal_format"], itzma_core::wal::FORMAT_VERSION);
    }
}

#[tokio::test]
async fn load_refuses_a_wal_format_this_build_cannot_read() {
    let supported = itzma_core::wal::FORMAT_VERSION;
    for b in common::backends() {
        eprintln!("backend: {}", b.name);
        let mut m = new_manifest();
        m.wal_format = supported + 1;
        manifest::create(&*b.store, "ns1", &m).await.unwrap();

        let err = manifest::load(&*b.store, "ns1").await.unwrap_err();
        assert!(
            matches!(
                err,
                ItzmaError::WalFormatTooNew { found, supported: s }
                    if found == supported + 1 && s == supported
            ),
            "unexpected error: {err:?}"
        );
        assert!(err.to_string().contains("upgrade"), "{err}");
    }
}

#[tokio::test]
async fn a_wal_format_at_the_builds_own_level_loads() {
    for b in common::backends() {
        eprintln!("backend: {}", b.name);
        manifest::create(&*b.store, "ns1", &new_manifest())
            .await
            .unwrap();
        let (loaded, _) = manifest::load(&*b.store, "ns1").await.unwrap();
        assert_eq!(loaded.wal_format, itzma_core::wal::FORMAT_VERSION);
    }
}

#[tokio::test]
async fn a_new_manifest_carries_the_default_segment_format_into_the_object() {
    assert_eq!(
        new_manifest().segment_format,
        itzma_core::segment::DEFAULT_FORMAT
    );
    for b in common::backends() {
        eprintln!("backend: {}", b.name);
        manifest::create(&*b.store, "ns1", &new_manifest())
            .await
            .unwrap();
        let raw = b
            .store
            .get(&manifest::manifest_path("ns1"))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        assert_eq!(json["segment_format"], itzma_core::segment::DEFAULT_FORMAT);
    }
}

#[tokio::test]
async fn load_refuses_a_segment_format_this_build_cannot_read() {
    let supported = itzma_core::segment::FORMAT_VERSION;
    for b in common::backends() {
        eprintln!("backend: {}", b.name);
        let mut m = new_manifest();
        m.segment_format = supported + 1;
        manifest::create(&*b.store, "ns1", &m).await.unwrap();

        let err = manifest::load(&*b.store, "ns1").await.unwrap_err();
        assert!(
            matches!(
                err,
                ItzmaError::SegmentFormatTooNew { found, supported: s }
                    if found == supported + 1 && s == supported
            ),
            "unexpected error: {err:?}"
        );
        assert!(err.to_string().contains("upgrade"), "{err}");
    }
}

#[tokio::test]
async fn a_segment_format_at_the_builds_own_level_loads() {
    for b in common::backends() {
        eprintln!("backend: {}", b.name);
        let mut m = new_manifest();
        m.segment_format = itzma_core::segment::FORMAT_VERSION;
        manifest::create(&*b.store, "ns1", &m).await.unwrap();
        let (loaded, _) = manifest::load(&*b.store, "ns1").await.unwrap();
        assert_eq!(loaded.segment_format, itzma_core::segment::FORMAT_VERSION);
    }
}
