//! The maintenance lease: who claims it, how long a silent leader
//! keeps it, and what happens when the object goes missing, turns to garbage
//! or the store fails part way. Time is passed in, so nothing here sleeps.

mod common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use itzma_core::{
    ItzmaError, GcOptions, LeaseOptions, MaintenanceLease, Metric, Namespace, Role, inspect,
};
use common::faulty::{Fault, FaultyStore, PutKind};
use common::{cas_backends, doc, memory, schema};
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt};

const DIM: usize = 2;
const L2: Metric = Metric::L2;
const SECOND: Duration = Duration::from_secs(1);
const OPTIONS: LeaseOptions = LeaseOptions {
    renew: SECOND,
    ttl: Duration::from_secs(4),
};

fn key() -> Path {
    Path::from("docs/maintenance/lease.json")
}

async fn namespace(store: &Arc<dyn ObjectStore>) -> Namespace {
    Namespace::create(store.clone(), "docs", schema(DIM, L2))
        .await
        .unwrap()
}

async fn lease(store: &Arc<dyn ObjectStore>) -> MaintenanceLease {
    let ns = Namespace::open(store.clone(), "docs").await.unwrap();
    ns.maintenance_lease(OPTIONS).unwrap()
}

async fn body(store: &dyn ObjectStore) -> serde_json::Value {
    let bytes = store.get(&key()).await.unwrap().bytes().await.unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

#[tokio::test]
async fn the_first_tick_claims() {
    for b in cas_backends() {
        namespace(&b.store).await;
        let mut a = lease(&b.store).await;
        assert_eq!(
            a.tick(Instant::now()).await.unwrap(),
            Role::Leader,
            "{}",
            b.name
        );
        assert_eq!(a.role(), Role::Leader, "{}", b.name);
        let held = body(&*b.store).await;
        assert_eq!(held["holder"], a.holder(), "{}", b.name);
        assert_eq!(held["released"], false, "{}", b.name);
    }
}

#[tokio::test]
async fn a_leader_stays_leader_across_renews() {
    for b in cas_backends() {
        namespace(&b.store).await;
        let mut a = lease(&b.store).await;
        let t0 = Instant::now();
        for beat in 0..3u64 {
            let role = a.tick(t0 + SECOND * beat as u32).await.unwrap();
            assert_eq!(role, Role::Leader, "{}", b.name);
            assert_eq!(body(&*b.store).await["beat"], beat, "{}", b.name);
        }
    }
}

#[tokio::test]
async fn a_second_lease_follows() {
    for b in cas_backends() {
        namespace(&b.store).await;
        let (mut a, mut other) = (lease(&b.store).await, lease(&b.store).await);
        let t0 = Instant::now();
        for s in 0..4 {
            let now = t0 + SECOND * s;
            assert_eq!(a.tick(now).await.unwrap(), Role::Leader, "{}", b.name);
            assert_eq!(other.tick(now).await.unwrap(), Role::Follower, "{}", b.name);
        }
        assert_ne!(a.holder(), other.holder(), "{}", b.name);
    }
}

#[tokio::test]
async fn a_silent_leader_is_taken_over_after_ttl_and_not_before() {
    for b in cas_backends() {
        namespace(&b.store).await;
        let (mut a, mut other) = (lease(&b.store).await, lease(&b.store).await);
        let t0 = Instant::now();
        assert_eq!(a.tick(t0).await.unwrap(), Role::Leader);
        assert_eq!(other.tick(t0).await.unwrap(), Role::Follower);
        assert_eq!(
            other.tick(t0 + SECOND * 3).await.unwrap(),
            Role::Follower,
            "{}",
            b.name
        );
        assert_eq!(
            other.tick(t0 + SECOND * 4).await.unwrap(),
            Role::Leader,
            "{}",
            b.name
        );
        assert_eq!(
            body(&*b.store).await["holder"],
            other.holder(),
            "{}",
            b.name
        );
        assert_eq!(
            a.tick(t0 + SECOND * 5).await.unwrap(),
            Role::Follower,
            "{}",
            b.name
        );
    }
}

#[tokio::test]
async fn one_of_three_followers_wins_a_takeover() {
    for b in cas_backends() {
        namespace(&b.store).await;
        let mut a = lease(&b.store).await;
        let mut f = [
            lease(&b.store).await,
            lease(&b.store).await,
            lease(&b.store).await,
        ];
        let t0 = Instant::now();
        assert_eq!(a.tick(t0).await.unwrap(), Role::Leader);
        for l in &mut f {
            assert_eq!(l.tick(t0).await.unwrap(), Role::Follower);
        }
        let later = t0 + OPTIONS.ttl;
        let [x, y, z] = &mut f;
        let (rx, ry, rz) = futures::join!(x.tick(later), y.tick(later), z.tick(later));
        let leaders = [rx.unwrap(), ry.unwrap(), rz.unwrap()]
            .into_iter()
            .filter(|r| *r == Role::Leader)
            .count();
        assert_eq!(leaders, 1, "{}", b.name);
    }
}

#[tokio::test]
async fn a_released_lease_is_claimed_on_the_next_tick() {
    for b in cas_backends() {
        namespace(&b.store).await;
        let (mut a, mut other) = (lease(&b.store).await, lease(&b.store).await);
        let t0 = Instant::now();
        assert_eq!(a.tick(t0).await.unwrap(), Role::Leader);
        a.release().await.unwrap();
        assert_eq!(a.role(), Role::Follower, "{}", b.name);
        assert_eq!(body(&*b.store).await["released"], true, "{}", b.name);
        assert_eq!(other.tick(t0).await.unwrap(), Role::Leader, "{}", b.name);
    }
}

#[tokio::test]
async fn a_deleted_lease_is_claimed_again() {
    for b in cas_backends() {
        namespace(&b.store).await;
        let (mut a, mut other) = (lease(&b.store).await, lease(&b.store).await);
        let t0 = Instant::now();
        assert_eq!(a.tick(t0).await.unwrap(), Role::Leader);
        b.store.delete(&key()).await.unwrap();
        assert_eq!(
            a.tick(t0 + SECOND).await.unwrap(),
            Role::Follower,
            "{}",
            b.name
        );
        assert_eq!(
            other.tick(t0 + SECOND).await.unwrap(),
            Role::Leader,
            "{}",
            b.name
        );
    }
}

#[tokio::test]
async fn a_garbage_lease_is_taken_over_after_ttl() {
    for b in cas_backends() {
        namespace(&b.store).await;
        b.store
            .put(&key(), b"not json".to_vec().into())
            .await
            .unwrap();
        let mut a = lease(&b.store).await;
        let t0 = Instant::now();
        assert_eq!(a.tick(t0).await.unwrap(), Role::Follower, "{}", b.name);
        assert_eq!(
            a.tick(t0 + SECOND * 3).await.unwrap(),
            Role::Follower,
            "{}",
            b.name
        );
        assert_eq!(
            a.tick(t0 + SECOND * 4).await.unwrap(),
            Role::Leader,
            "{}",
            b.name
        );
    }
}

#[tokio::test]
async fn a_renew_error_demotes_the_leader() {
    let faulty = FaultyStore::new(memory());
    let store: Arc<dyn ObjectStore> = faulty.clone();
    namespace(&store).await;
    let mut a = lease(&store).await;
    let t0 = Instant::now();
    assert_eq!(a.tick(t0).await.unwrap(), Role::Leader);

    faulty.fail_puts(
        "maintenance/lease",
        PutKind::Update,
        Fault::Generic,
        Some(1),
    );
    assert!(matches!(
        a.tick(t0 + SECOND).await,
        Err(ItzmaError::Storage(_))
    ));
    assert_eq!(a.role(), Role::Follower);

    assert_eq!(a.tick(t0 + SECOND * 2).await.unwrap(), Role::Follower);
    assert_eq!(a.tick(t0 + OPTIONS.ttl).await.unwrap(), Role::Leader);
}

#[tokio::test]
async fn gc_inspect_and_recover_ignore_the_lease() {
    for b in cas_backends() {
        let ns = namespace(&b.store).await;
        ns.upsert(vec![doc("a", &[1.0, 0.0])]).await.unwrap();
        let mut a = ns.maintenance_lease(OPTIONS).unwrap();
        a.tick(Instant::now()).await.unwrap();
        let before = b.store.get(&key()).await.unwrap().bytes().await.unwrap();

        let collect_now = GcOptions {
            min_age: Duration::ZERO,
            ..GcOptions::default()
        };
        ns.gc(&collect_now).await.unwrap();
        let report = inspect::report(&*b.store, "docs").await.unwrap();
        assert!(
            report.problems.iter().all(|p| !p.contains("maintenance")),
            "{}: {:?}",
            b.name,
            report.problems
        );
        b.store
            .delete(&Path::from("docs/manifest/current.json"))
            .await
            .unwrap();
        Namespace::recover(b.store.clone(), "docs").await.unwrap();

        let after = b.store.get(&key()).await.unwrap().bytes().await.unwrap();
        assert_eq!(before, after, "{}", b.name);
    }
}

#[tokio::test]
async fn bad_options_are_refused() {
    let store = memory();
    let ns = namespace(&store).await;
    let zero = LeaseOptions {
        renew: Duration::ZERO,
        ttl: SECOND,
    };
    let short = LeaseOptions {
        renew: SECOND,
        ttl: SECOND,
    };
    assert!(matches!(
        ns.maintenance_lease(zero),
        Err(ItzmaError::InvalidConfig(_))
    ));
    assert!(matches!(
        ns.maintenance_lease(short),
        Err(ItzmaError::InvalidConfig(_))
    ));
    assert!(ns.maintenance_lease(LeaseOptions::default()).is_ok());
}
