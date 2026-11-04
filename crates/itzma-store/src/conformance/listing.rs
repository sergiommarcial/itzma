//! Listings. GC, recovery and the WAL tail all find objects by listing, so a
//! listing that drops keys loses data silently. Order is not part of the
//! contract (core sorts what it needs), so these compare sets.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use futures::{StreamExt, TryStreamExt, stream};
use object_store::path::Path;
use object_store::{ObjectMeta, ObjectStore, ObjectStoreExt, PutPayload};

async fn put_all(store: &Arc<dyn ObjectStore>, keys: impl IntoIterator<Item = (String, usize)>) {
    stream::iter(keys)
        .map(|(key, len)| {
            let store = Arc::clone(store);
            async move {
                store
                    .put(&Path::from(key), PutPayload::from(vec![b'x'; len]))
                    .await
                    .expect("put")
            }
        })
        .buffer_unordered(32)
        .collect::<Vec<_>>()
        .await;
}

fn paths(listing: &[ObjectMeta]) -> BTreeSet<String> {
    listing.iter().map(|m| m.location.to_string()).collect()
}

/// `list` under a prefix returns exactly the objects below it, with sizes. A
/// prefix is whole path segments: `a` does not match `ab/1`, or a namespace
/// listing would pick up a sibling namespace's objects.
pub async fn list_under_prefix(store: Arc<dyn ObjectStore>) {
    put_all(
        &store,
        [
            ("a/1".into(), 1),
            ("a/2".into(), 2),
            ("ab/1".into(), 1),
            ("b/1".into(), 1),
        ],
    )
    .await;
    let listing: Vec<ObjectMeta> = store
        .list(Some(&Path::from("a")))
        .try_collect()
        .await
        .expect("list");
    let sizes: BTreeMap<String, u64> = listing
        .iter()
        .map(|m| (m.location.to_string(), m.size))
        .collect();
    let expected = BTreeMap::from([("a/1".to_string(), 1), ("a/2".to_string(), 2)]);
    assert_eq!(sizes, expected, "list under a/ with sizes");
}

/// A listing longer than one page (S3 pages at 1,000 keys) returns every key.
pub async fn list_across_pages(store: Arc<dyn ObjectStore>) {
    put_all(&store, (0..1050).map(|i| (format!("p/{i:04}"), 1))).await;
    let listing: Vec<ObjectMeta> = store
        .list(Some(&Path::from("p")))
        .try_collect()
        .await
        .expect("list");
    assert_eq!(listing.len(), 1050, "list across pages: entries");
    let expected: BTreeSet<String> = (0..1050).map(|i| format!("p/{i:04}")).collect();
    assert_eq!(paths(&listing), expected, "list across pages: keys");
}

/// `list_with_offset` returns only keys under the prefix strictly after the
/// offset.
pub async fn list_with_offset_after(store: Arc<dyn ObjectStore>) {
    put_all(
        &store,
        (1..=5)
            .map(|i| (format!("k/{i:02}"), 1))
            .chain([("kk/09".to_string(), 1)]),
    )
    .await;
    let listing: Vec<ObjectMeta> = store
        .list_with_offset(Some(&Path::from("k")), &Path::from("k/03"))
        .try_collect()
        .await
        .expect("list_with_offset");
    let expected = BTreeSet::from(["k/04".to_string(), "k/05".to_string()]);
    assert_eq!(paths(&listing), expected, "list_with_offset after k/03");
}

/// `list_with_delimiter` returns direct objects and the prefixes below.
pub async fn list_with_delimiter_splits(store: Arc<dyn ObjectStore>) {
    put_all(
        &store,
        [("d/x".into(), 1), ("d/y/z".into(), 1), ("dd/x".into(), 1)],
    )
    .await;
    let result = store
        .list_with_delimiter(Some(&Path::from("d")))
        .await
        .expect("list_with_delimiter");
    assert_eq!(
        paths(&result.objects),
        BTreeSet::from(["d/x".to_string()]),
        "list_with_delimiter objects"
    );
    let prefixes: BTreeSet<String> = result
        .common_prefixes
        .iter()
        .map(|p| p.to_string())
        .collect();
    assert_eq!(
        prefixes,
        BTreeSet::from(["d/y".to_string()]),
        "list_with_delimiter common prefixes"
    );
}
