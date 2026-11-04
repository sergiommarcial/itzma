//! Whole and ranged reads, and how a missing object reads (a reader
//! recognises a segment garbage collection removed by `NotFound`).

use std::ops::Range;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use futures::{FutureExt, TryStreamExt};
use object_store::path::Path;
use object_store::{
    Error, GetOptions, GetRange, GetResult, ObjectStore, ObjectStoreExt, PutPayload,
};

const BODY: &[u8] = b"0123456789";

async fn seeded(store: &dyn ObjectStore, name: &str) -> Path {
    let key = Path::from(name);
    store
        .put(&key, PutPayload::from_static(BODY))
        .await
        .expect("put");
    key
}

async fn read_range(
    store: &dyn ObjectStore,
    key: &Path,
    range: GetRange,
) -> object_store::Result<Vec<u8>> {
    Ok(get_range_opts(store, key, range)
        .await?
        .bytes()
        .await?
        .to_vec())
}

async fn get_range_opts(
    store: &dyn ObjectStore,
    key: &Path,
    range: GetRange,
) -> object_store::Result<GetResult> {
    let options = GetOptions {
        range: Some(range),
        ..GetOptions::default()
    };
    store.get_opts(key, options).await
}

/// The bytes of a ranged read, checking the range and size it reports, which
/// is how a segment's footer is found from a suffix read.
async fn ranged(store: &dyn ObjectStore, key: &Path, range: GetRange, at: Range<u64>) -> Vec<u8> {
    let got = get_range_opts(store, key, range.clone())
        .await
        .unwrap_or_else(|e| panic!("ranged read {range}: {e}"));
    assert_eq!(
        got.range, at,
        "ranged read {range} reports where it starts and ends"
    );
    assert_eq!(
        got.meta.size, 10,
        "ranged read {range} reports the object size"
    );
    got.bytes().await.expect("body").to_vec()
}

/// `get` returns the bytes written, and `head` and the get's metadata report
/// their size.
pub async fn get_returns_body_and_size(store: Arc<dyn ObjectStore>) {
    let key = seeded(store.as_ref(), "body").await;
    let got = store.get(&key).await.expect("get");
    assert_eq!(got.meta.size, 10, "get metadata size");
    assert_eq!(got.bytes().await.expect("body").as_ref(), BODY, "get body");
    assert_eq!(store.head(&key).await.expect("head").size, 10, "head size");
}

/// An empty object reads back as empty. Core writes some (a blank fence hint),
/// and `bytes()` polls the body again after its last chunk, so a stream that
/// isn't fused panics there.
pub async fn empty_object_reads_empty(store: Arc<dyn ObjectStore>) {
    let key = Path::from("empty");
    store
        .put(&key, PutPayload::default())
        .await
        .expect("put an empty object");
    let got = store.get(&key).await.expect("get an empty object");
    assert_eq!(got.meta.size, 0, "empty object metadata size");
    let body = AssertUnwindSafe(got.bytes())
        .catch_unwind()
        .await
        .expect("an empty object's body must read as empty, not panic")
        .expect("empty body");
    assert!(body.is_empty(), "an empty object's body must be empty");
    assert_eq!(
        store.head(&key).await.expect("head").size,
        0,
        "empty head size"
    );
}

/// Bounded, offset, suffix and multi-range reads return the bytes asked for.
pub async fn ranged_reads(store: Arc<dyn ObjectStore>) {
    let key = seeded(store.as_ref(), "ranges").await;
    let bounded = store.get_range(&key, 2..5).await.expect("bounded range");
    assert_eq!(bounded.as_ref(), b"234", "bounded range 2..5");
    let offset = ranged(store.as_ref(), &key, GetRange::Offset(7), 7..10).await;
    assert_eq!(offset, b"789", "offset range from 7");
    let suffix = ranged(store.as_ref(), &key, GetRange::Suffix(3), 7..10).await;
    assert_eq!(suffix, b"789", "suffix range of 3");
    let ranges: [Range<u64>; 2] = [0..2, 8..10];
    let many = store.get_ranges(&key, &ranges).await.expect("get_ranges");
    let many: Vec<&[u8]> = many.iter().map(|b| b.as_ref()).collect();
    assert_eq!(many, [b"01".as_slice(), b"89"], "get_ranges 0..2 and 8..10");
}

/// A range that ends past the object is clamped to it; one that starts past
/// it is an error, never an empty read that would hide a bad segment offset.
pub async fn range_past_end(store: Arc<dyn ObjectStore>) {
    let key = seeded(store.as_ref(), "past-end").await;
    let clamped = read_range(store.as_ref(), &key, GetRange::Bounded(5..100))
        .await
        .expect("range ending past the object");
    assert_eq!(clamped, b"56789", "range 5..100 on 10 bytes");
    for range in [GetRange::Bounded(20..30), GetRange::Offset(20)] {
        let result = read_range(store.as_ref(), &key, range.clone()).await;
        assert!(
            result.is_err() && !matches!(result, Err(Error::NotFound { .. })),
            "range {range} on 10 bytes must fail, not read as NotFound, got {result:?}"
        );
    }
}

/// A suffix longer than the object returns all of it. A segment's footer is
/// read with a suffix that may be longer than a small segment.
pub async fn suffix_longer_than_object(store: Arc<dyn ObjectStore>) {
    let key = seeded(store.as_ref(), "long-suffix").await;
    let got = get_range_opts(store.as_ref(), &key, GetRange::Suffix(100))
        .await
        .unwrap_or_else(|e| panic!("suffix longer than the object: {e}"));
    assert_eq!(
        got.range,
        0..10,
        "suffix longer than the object reports 0..10"
    );
    assert_eq!(
        got.meta.size, 10,
        "suffix longer than the object reports its size"
    );
    assert_eq!(
        got.bytes().await.expect("body").as_ref(),
        BODY,
        "suffix longer than the object"
    );
}

/// `get`, `head` and ranged reads of a missing key are `NotFound` (segment
/// reads are ranged, and that is how a stale read is told apart); `delete` of
/// one is `Ok` or `NotFound` (S3 answers `Ok`, and `gc` accepts both).
pub async fn missing_object(store: Arc<dyn ObjectStore>) {
    let key = Path::from("missing");
    let get = store.get(&key).await;
    assert!(
        matches!(get, Err(Error::NotFound { .. })),
        "get of a missing key must be NotFound, got {:?}",
        get.map(|r| r.meta)
    );
    let head = store.head(&key).await;
    assert!(
        matches!(head, Err(Error::NotFound { .. })),
        "head of a missing key must be NotFound, got {head:?}"
    );
    let range = store.get_range(&key, 0..4).await;
    assert!(
        matches!(range, Err(Error::NotFound { .. })),
        "ranged read of a missing key must be NotFound, got {range:?}"
    );
    let suffix = read_range(store.as_ref(), &key, GetRange::Suffix(4)).await;
    assert!(
        matches!(suffix, Err(Error::NotFound { .. })),
        "ranged read of a missing key must be NotFound (suffix), got {suffix:?}"
    );
    let delete = store.delete(&key).await;
    assert!(
        matches!(delete, Ok(()) | Err(Error::NotFound { .. })),
        "delete of a missing key must be Ok or NotFound, got {delete:?}"
    );
}

/// `delete` removes the object: `get` and `head` then find nothing, and a
/// listing no longer shows it. GC and folding depend on deletes being real.
pub async fn delete_removes(store: Arc<dyn ObjectStore>) {
    let key = seeded(store.as_ref(), "deleted").await;
    store.delete(&key).await.expect("delete");
    let get = store.get(&key).await;
    assert!(
        matches!(get, Err(Error::NotFound { .. })),
        "deleted key must be gone from get, got {:?}",
        get.map(|r| r.meta)
    );
    let head = store.head(&key).await;
    assert!(
        matches!(head, Err(Error::NotFound { .. })),
        "deleted key must be gone from head, got {head:?}"
    );
    let listed: Vec<_> = store.list(None).try_collect().await.expect("list");
    assert!(
        listed.iter().all(|m| m.location != key),
        "deleted key must be gone from list"
    );
}
