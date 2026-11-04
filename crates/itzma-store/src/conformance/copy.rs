//! Server-side copy-if-not-exists and multipart uploads, which streaming a
//! large segment through a staging key would be built on.

use std::sync::Arc;

use object_store::path::Path;
use object_store::{Error, ObjectStore, ObjectStoreExt, PutPayload};

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

/// Copy-if-not-exists writes a new key and refuses an existing one with
/// `AlreadyExists`, leaving it as it was.
pub async fn copy_if_absent(store: Arc<dyn ObjectStore>) {
    let (src, dst, taken) = (Path::from("src"), Path::from("dst"), Path::from("taken"));
    store
        .put(&src, PutPayload::from_static(b"source"))
        .await
        .expect("put source");
    store
        .put(&taken, PutPayload::from_static(b"old"))
        .await
        .expect("put taken");
    store
        .copy_if_not_exists(&src, &dst)
        .await
        .expect("copy-if-not-exists to a new key");
    assert_eq!(read(store.as_ref(), &dst).await, b"source", "copied body");
    let refused = store.copy_if_not_exists(&src, &taken).await;
    assert!(
        matches!(refused, Err(Error::AlreadyExists { .. })),
        "copy-if-not-exists onto an existing key must be AlreadyExists, got {refused:?}"
    );
    assert_eq!(
        read(store.as_ref(), &taken).await,
        b"old",
        "copy-if-not-exists overwrote"
    );
}

/// A two-part upload (5 MiB, S3's minimum for every part but the last, then
/// 1 KiB) reads back byte-equal with the right size.
pub async fn multipart_round_trip(store: Arc<dyn ObjectStore>) {
    let key = Path::from("multipart");
    let first = vec![1u8; 5 * 1024 * 1024];
    let second = vec![2u8; 1024];
    let mut upload = store.put_multipart(&key).await.expect("start multipart");
    upload
        .put_part(PutPayload::from(first.clone()))
        .await
        .expect("first part");
    upload
        .put_part(PutPayload::from(second.clone()))
        .await
        .expect("second part");
    upload.complete().await.expect("complete multipart");
    let size = (first.len() + second.len()) as u64;
    assert_eq!(
        store.head(&key).await.expect("head").size,
        size,
        "multipart size"
    );
    let body = read(store.as_ref(), &key).await;
    assert!(
        body.len() == first.len() + second.len()
            && body[..first.len()] == first[..]
            && body[first.len()..] == second[..],
        "multipart body is not the parts in order"
    );
}

/// A plain copy writes the source's bytes over an existing key.
pub async fn copy_overwrites(store: Arc<dyn ObjectStore>) {
    let (src, dst) = (Path::from("src"), Path::from("dst"));
    store
        .put(&src, PutPayload::from_static(b"new"))
        .await
        .expect("put source");
    store
        .put(&dst, PutPayload::from_static(b"old"))
        .await
        .expect("put target");
    store.copy(&src, &dst).await.expect("copy");
    assert_eq!(
        read(store.as_ref(), &dst).await,
        b"new",
        "copy onto an existing key must replace it"
    );
}
