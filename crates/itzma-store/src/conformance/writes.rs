//! Conditional writes: put-if-absent, which picks WAL sequence numbers, and
//! the versioned update behind the manifest's compare-and-swap.

use std::sync::Arc;

use futures::future::join_all;
use object_store::path::Path;
use object_store::{
    Error, ObjectStore, ObjectStoreExt, PutMode, PutOptions, PutPayload, PutResult, UpdateVersion,
};

async fn create(
    store: &dyn ObjectStore,
    key: &Path,
    body: &[u8],
) -> object_store::Result<PutResult> {
    store
        .put_opts(
            key,
            PutPayload::from(body.to_vec()),
            PutOptions::from(PutMode::Create),
        )
        .await
}

async fn update(
    store: &dyn ObjectStore,
    key: &Path,
    body: &[u8],
    version: UpdateVersion,
) -> object_store::Result<PutResult> {
    store
        .put_opts(
            key,
            PutPayload::from(body.to_vec()),
            PutOptions::from(PutMode::Update(version)),
        )
        .await
}

async fn read(store: &dyn ObjectStore, key: &Path) -> Vec<u8> {
    store
        .get(key)
        .await
        .expect("get")
        .bytes()
        .await
        .expect("body")
        .to_vec()
}

fn version_of(put: &PutResult) -> UpdateVersion {
    UpdateVersion {
        e_tag: put.e_tag.clone(),
        version: put.version.clone(),
    }
}

/// Put-if-absent writes a new key and refuses an existing one with
/// `AlreadyExists`, leaving the first body in place.
pub async fn put_if_absent_creates(store: Arc<dyn ObjectStore>) {
    let key = Path::from("creates");
    create(store.as_ref(), &key, b"one")
        .await
        .expect("put-if-absent on a new key");
    let second = create(store.as_ref(), &key, b"two").await;
    assert!(
        matches!(second, Err(Error::AlreadyExists { .. })),
        "put-if-absent on an existing key must be AlreadyExists, got {second:?}"
    );
    assert_eq!(
        read(store.as_ref(), &key).await,
        b"one",
        "put-if-absent overwrote"
    );
}

/// Of 16 writers racing put-if-absent on one key, exactly one wins and every
/// other gets `AlreadyExists`. This is how a WAL sequence number is claimed.
pub async fn put_if_absent_one_winner(store: Arc<dyn ObjectStore>) {
    let key = Path::from("race");
    let results = join_all((0..16u8).map(|i| {
        let store = Arc::clone(&store);
        let key = key.clone();
        async move { (i, create(store.as_ref(), &key, &[i]).await) }
    }))
    .await;
    let winners: Vec<u8> = results
        .iter()
        .filter(|(_, r)| r.is_ok())
        .map(|(i, _)| *i)
        .collect();
    assert_eq!(
        winners.len(),
        1,
        "put-if-absent race must have one winner, got {winners:?}"
    );
    for (i, result) in &results {
        assert!(
            result.is_ok() || matches!(result, Err(Error::AlreadyExists { .. })),
            "losing writer {i} must get AlreadyExists, got {result:?}"
        );
    }
    assert_eq!(
        read(store.as_ref(), &key).await,
        [winners[0]],
        "body is not the winner's"
    );
}

/// An update carrying the version a put returned replaces the object.
pub async fn update_matches_version(store: Arc<dyn ObjectStore>) {
    let key = Path::from("update");
    let put = store
        .put(&key, PutPayload::from_static(b"one"))
        .await
        .expect("put");
    update(store.as_ref(), &key, b"two", version_of(&put))
        .await
        .expect("update with the current version");
    assert_eq!(
        read(store.as_ref(), &key).await,
        b"two",
        "update did not replace the body"
    );
}

/// A second update from the same version is a `Precondition` error and
/// leaves the first update's body. This is the manifest CAS losing a race.
pub async fn update_refuses_stale(store: Arc<dyn ObjectStore>) {
    let key = Path::from("stale");
    let put = store
        .put(&key, PutPayload::from_static(b"zero"))
        .await
        .expect("put");
    update(store.as_ref(), &key, b"one", version_of(&put))
        .await
        .expect("first update");
    let second = update(store.as_ref(), &key, b"two", version_of(&put)).await;
    assert!(
        matches!(second, Err(Error::Precondition { .. })),
        "update from a stale version must be Precondition, got {second:?}"
    );
    assert_eq!(
        read(store.as_ref(), &key).await,
        b"one",
        "stale update overwrote"
    );
}

/// The version `head` reports is one `Update` accepts, since the manifest is
/// reloaded before it is updated again.
pub async fn update_with_head_version(store: Arc<dyn ObjectStore>) {
    let key = Path::from("head-version");
    store
        .put(&key, PutPayload::from_static(b"one"))
        .await
        .expect("put");
    let meta = store.head(&key).await.expect("head");
    let version = UpdateVersion {
        e_tag: meta.e_tag,
        version: meta.version,
    };
    update(store.as_ref(), &key, b"two", version)
        .await
        .expect("update with the version head reported");
    assert_eq!(
        read(store.as_ref(), &key).await,
        b"two",
        "update did not replace the body"
    );
}

/// An update to a key that does not exist is a `Precondition` error.
pub async fn update_missing_is_precondition(store: Arc<dyn ObjectStore>) {
    let version = UpdateVersion {
        e_tag: Some("\"x\"".into()),
        version: None,
    };
    let result = update(store.as_ref(), &Path::from("missing"), b"one", version).await;
    assert!(
        matches!(result, Err(Error::Precondition { .. })),
        "update of a missing key must be Precondition, got {result:?}"
    );
}

/// The version `get` reports is one `Update` accepts. `manifest::load` reads
/// the manifest with `get` and updates it with that version, so an adapter
/// that fills the version only on `head` breaks every manifest CAS.
pub async fn update_with_get_version(store: Arc<dyn ObjectStore>) {
    let key = Path::from("get-version");
    store
        .put(&key, PutPayload::from_static(b"one"))
        .await
        .expect("put");
    let meta = store.get(&key).await.expect("get").meta;
    let version = UpdateVersion {
        e_tag: meta.e_tag,
        version: meta.version,
    };
    update(store.as_ref(), &key, b"two", version)
        .await
        .expect("update with the version get reported");
    assert_eq!(
        read(store.as_ref(), &key).await,
        b"two",
        "update did not replace the body"
    );
}

/// An update carrying no version at all fails and leaves the object alone.
/// Treating it as a plain write would turn a CAS into a blind overwrite.
pub async fn update_without_version_refused(store: Arc<dyn ObjectStore>) {
    let key = Path::from("versionless");
    store
        .put(&key, PutPayload::from_static(b"one"))
        .await
        .expect("put");
    let version = UpdateVersion {
        e_tag: None,
        version: None,
    };
    let result = update(store.as_ref(), &key, b"two", version).await;
    assert!(
        result.is_err(),
        "update with no version must fail, got {result:?}"
    );
    assert_eq!(
        read(store.as_ref(), &key).await,
        b"one",
        "versionless update overwrote"
    );
}

/// Of 16 writers racing an update from the same version, exactly one wins and
/// every other gets `Precondition`. This is two manifest CAS attempts racing.
pub async fn update_one_winner(store: Arc<dyn ObjectStore>) {
    let key = Path::from("update-race");
    let put = store
        .put(&key, PutPayload::from_static(b"start"))
        .await
        .expect("put");
    let version = version_of(&put);
    let results = join_all((0..16u8).map(|i| {
        let store = Arc::clone(&store);
        let key = key.clone();
        let version = version.clone();
        async move { (i, update(store.as_ref(), &key, &[i], version).await) }
    }))
    .await;
    let winners: Vec<u8> = results
        .iter()
        .filter(|(_, r)| r.is_ok())
        .map(|(i, _)| *i)
        .collect();
    assert_eq!(
        winners.len(),
        1,
        "update race must have one winner, got {winners:?}"
    );
    for (i, result) in &results {
        assert!(
            result.is_ok() || matches!(result, Err(Error::Precondition { .. })),
            "losing updater {i} must get Precondition, got {result:?}"
        );
    }
    assert_eq!(
        read(store.as_ref(), &key).await,
        [winners[0]],
        "body is not the winner's"
    );
}
