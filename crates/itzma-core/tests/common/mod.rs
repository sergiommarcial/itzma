#![allow(dead_code)] // each test binary uses a subset of these helpers

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use futures::TryStreamExt;
use itzma_core::layer::Layer;
use itzma_core::search::distance;
use itzma_core::{Document, Metric, Namespace, Schema, WalOp, manifest, segment, wal};
use object_store::aws::AmazonS3Builder;
use object_store::local::LocalFileSystem;
use object_store::memory::InMemory;
use object_store::prefix::PrefixStore;
use object_store::{ObjectStore, ObjectStoreExt};
use tempfile::TempDir;

pub mod counting;
pub mod faulty;

/// Point this at an S3-compatible endpoint (for example `http://127.0.0.1:9000`)
/// to run the storage tests against it too. `make test-s3` does that. Unset,
/// the S3 backend is left out and no server is needed. Once set, an
/// unreachable server fails the tests instead of skipping them.
const S3_ENDPOINT_VAR: &str = "ITZMA_TEST_S3_ENDPOINT";

/// A named store; the tempdir guard keeps a local store's directory alive.
pub struct Backend {
    pub name: &'static str,
    pub store: Arc<dyn ObjectStore>,
    _dir: Option<TempDir>,
}

pub fn memory() -> Arc<dyn ObjectStore> {
    Arc::new(InMemory::new())
}

/// A key prefix no other test, and no other process, has used. It carries the
/// process id and the clock, so two test binaries sharing a bucket never meet.
pub fn s3_prefix() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    format!(
        "t-{}-{nanos}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

/// An S3 store under `prefix`, or `None` when no endpoint is configured.
/// Credentials default to the ones in `compose.yaml`. Dedicated
/// `ITZMA_TEST_S3_*` variables are used rather than `AWS_*`, so credentials for
/// a real account can never be picked up. Two processes given the same prefix
/// see the same objects, which is what the kill test relies on.
pub fn s3_at(prefix: &str) -> Option<Arc<dyn ObjectStore>> {
    let endpoint = std::env::var(S3_ENDPOINT_VAR)
        .ok()
        .filter(|v| !v.is_empty())?;
    let var = |name: &str, default: &str| std::env::var(name).unwrap_or_else(|_| default.into());
    let store = AmazonS3Builder::new()
        .with_endpoint(endpoint)
        .with_bucket_name(var("ITZMA_TEST_S3_BUCKET", "itzma-test"))
        .with_access_key_id(var("ITZMA_TEST_S3_ACCESS_KEY", "itzma"))
        .with_secret_access_key(var("ITZMA_TEST_S3_SECRET_KEY", "itzma-test-secret"))
        .with_region("us-east-1")
        .with_allow_http(true)
        .build()
        .expect("s3 store");
    Some(Arc::new(PrefixStore::new(store, prefix)))
}

/// An S3 store isolated under a fresh key prefix, so tests that share the
/// bucket never see each other's namespaces.
fn s3() -> Option<Backend> {
    Some(Backend {
        name: "s3",
        store: s3_at(&s3_prefix())?,
        _dir: None,
    })
}

/// Every backend that supports put-if-absent (`PutMode::Create`).
pub fn backends() -> Vec<Backend> {
    let dir = TempDir::new().expect("tempdir");
    let local = LocalFileSystem::new_with_prefix(dir.path()).expect("local fs");
    let mut all = vec![
        Backend {
            name: "memory",
            store: memory(),
            _dir: None,
        },
        Backend {
            name: "local",
            store: Arc::new(local),
            _dir: Some(dir),
        },
    ];
    all.extend(s3());
    all
}

/// Every backend that also supports conditional update (`PutMode::Update`).
/// `LocalFileSystem` does not, so it is left out.
pub fn cas_backends() -> Vec<Backend> {
    let mut all = vec![Backend {
        name: "memory",
        store: memory(),
        _dir: None,
    }];
    all.extend(s3());
    all
}

pub fn schema(dim: usize, metric: Metric) -> Schema {
    Schema::new(dim, metric)
}

pub fn doc(id: &str, vector: &[f32]) -> Document {
    Document {
        id: id.into(),
        vector: vector.to_vec(),
        attrs: None,
    }
}

/// Write a segment object by hand and return its id. Like compaction, a base
/// segment (`lo == 1`) drops its tombstones.
pub async fn put_segment(
    store: &dyn ObjectStore,
    ns: &str,
    lo: u64,
    hi: u64,
    dim: usize,
    ops: Vec<WalOp>,
) -> String {
    let mut layer = Layer::new();
    for op in ops {
        layer.apply(op);
    }
    if lo == 1 {
        layer.drop_tombstones();
    }
    let id = segment::segment_id(lo, hi);
    let bytes = segment::encode(&layer, dim).expect("encode");
    store
        .put(&segment::segment_path(ns, &id), bytes.into())
        .await
        .expect("put segment");
    id
}

/// Point the manifest at `ids`, folded through `folded`.
pub async fn publish(store: &dyn ObjectStore, ns: &str, ids: &[&str], folded: u64) {
    manifest::update(store, ns, |m| {
        m.live_segments = ids.iter().map(|id| id.to_string()).collect();
        m.wal_folded_through = folded;
    })
    .await
    .expect("publish");
}

/// What a fence looks like on disk: `ITZF` and a u32 version, 1.
pub const FENCE: [u8; 8] = *b"ITZF\x01\x00\x00\x00";

/// Replace the WAL object at `seq` with a fence, by hand, the way garbage
/// collection does. A plain put, so it works on every backend.
pub async fn fence_by_hand(store: &dyn ObjectStore, ns: &str, seq: u64) {
    store
        .put(&wal::wal_path(ns, seq), Bytes::from_static(&FENCE).into())
        .await
        .expect("fence");
}

/// Deterministic pseudo-random numbers; no rand dependency.
pub struct Lcg(pub u64);

impl Lcg {
    fn step(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    /// A float in [-1, 1).
    pub fn next_f32(&mut self) -> f32 {
        ((self.step() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    pub fn vector(&mut self, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| self.next_f32()).collect()
    }

    /// An integer in `0..n`.
    pub fn below(&mut self, n: u64) -> u64 {
        (self.step() >> 33) % n
    }
}

/// Full sort by (distance, id): the obviously-correct reference.
pub fn naive_top_k(metric: Metric, query: &[f32], docs: &[Document], k: usize) -> Vec<String> {
    let mut scored: Vec<(f32, &str)> = docs
        .iter()
        .map(|d| (distance(metric, query, &d.vector), d.id.as_str()))
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(b.1)));
    scored
        .into_iter()
        .take(k)
        .map(|(_, id)| id.to_string())
        .collect()
}

/// Every object in a store, by key.
pub type Snapshot = BTreeMap<String, Bytes>;

pub async fn snapshot(store: &dyn ObjectStore) -> Snapshot {
    let metas: Vec<_> = store.list(None).try_collect().await.expect("list");
    let mut all = Snapshot::new();
    for meta in metas {
        let bytes = store
            .get(&meta.location)
            .await
            .expect("get")
            .bytes()
            .await
            .expect("bytes");
        all.insert(meta.location.to_string(), bytes);
    }
    all
}

/// Everything in `before` except the manifest must still be there,
/// byte-identical. An interrupted or restarted load or compaction may add
/// objects and move the manifest, and nothing else.
pub async fn assert_untouched(store: &dyn ObjectStore, before: &Snapshot) {
    let after = snapshot(store).await;
    for (key, bytes) in before {
        if key.ends_with("manifest/current.json") {
            continue;
        }
        assert_eq!(after.get(key), Some(bytes), "{key} changed or vanished");
    }
}

/// The seq in the fence hint object, if there is one that parses.
pub async fn fence_hint(store: &dyn ObjectStore) -> Option<u64> {
    match store
        .get(&itzma_core::gc::fenced_through_path("docs"))
        .await
    {
        Ok(got) => std::str::from_utf8(&got.bytes().await.expect("bytes"))
            .ok()?
            .trim()
            .parse()
            .ok(),
        Err(object_store::Error::NotFound { .. }) => None,
        Err(e) => panic!("{e}"),
    }
}

/// The WAL objects in a store: seq and size, in seq order.
pub async fn wal_listing(store: &dyn ObjectStore) -> Vec<(u64, u64)> {
    let prefix = object_store::path::Path::from("docs/wal");
    let metas: Vec<_> = store.list(Some(&prefix)).try_collect().await.expect("list");
    let mut all: Vec<(u64, u64)> = metas
        .into_iter()
        .map(|m| {
            let name = m.location.filename().expect("filename").to_string();
            let seq = name
                .strip_suffix(".json")
                .expect("suffix")
                .parse()
                .expect("seq");
            (seq, m.size)
        })
        .collect();
    all.sort();
    all
}

/// What must hold once an interrupted load or compaction has stopped and
/// faults are off:
///
/// 1. the manifest's segment chain is valid;
/// 2. the WAL is a contiguous run from 1 and the manifest does not fold past it;
/// 3. every live segment is byte-identical to a rebuild from its WAL range;
/// 4. every index the manifest names belongs to a live segment, its object is
///    there, and its prefix says it was built for that segment's rows and data
///    object;
/// 5. a query returns what a naive scan of the state replayed from the WAL
///    alone returns.
///
/// Returns that replayed state, sorted by id, so a caller that also has a
/// model can compare the two.
///
/// Once garbage collection has fenced WAL objects, the segments are
/// the only copy of what those objects held, so 3 cannot be checked and 4
/// replays the live segments and the tail instead. What is checked then is that
/// the keys are still contiguous, that only folded objects are fenced, and that
/// a query agrees with a naive scan of that state. A caller with a model of what
/// was written should compare it with the returned documents.
pub async fn assert_consistent(
    store: Arc<dyn ObjectStore>,
    dim: usize,
    metric: Metric,
) -> Vec<Document> {
    let (manifest, _) = manifest::load(&*store, "docs").await.expect("manifest");
    segment::validate_chain(&manifest.live_segments, manifest.wal_folded_through)
        .expect("segment chain");

    let listing = wal_listing(&*store).await;
    let seqs: Vec<u64> = listing.iter().map(|(seq, _)| *seq).collect();
    let expected: Vec<u64> = (1..=listing.len() as u64).collect();
    assert_eq!(seqs, expected, "the WAL must be contiguous from 1");
    assert!(
        manifest.wal_folded_through <= listing.len() as u64,
        "the manifest folds past the end of the WAL"
    );
    let fenced: Vec<u64> = listing
        .iter()
        .filter(|(_, size)| *size == FENCE.len() as u64)
        .map(|(seq, _)| *seq)
        .collect();
    assert!(
        fenced.iter().all(|seq| *seq <= manifest.wal_folded_through),
        "a fence stands past the fold point: {fenced:?}"
    );

    // The fence hint is trusted by `gc`, so it must never run ahead of
    // the fences: every WAL object at or below it has to be one.
    if let Some(reach) = fence_hint(&*store).await {
        let unfenced: Vec<u64> = listing
            .iter()
            .filter(|(seq, size)| *seq <= reach && *size != FENCE.len() as u64)
            .map(|(seq, _)| *seq)
            .collect();
        assert!(
            unfenced.is_empty(),
            "the fence hint says {reach} but {unfenced:?} are not fences"
        );
    }

    check_indexes(&store, dim, &manifest).await;

    let state = if fenced.is_empty() {
        replayed_from_the_wal(&store, &manifest).await
    } else {
        replayed_from_segments(&store, dim, &manifest).await
    };
    let docs: Vec<Document> = state.sorted_docs().into_iter().cloned().collect();

    // Room for stale documents a bad segment might add: with k equal to the
    // document count, an extra one could sit just outside the top k unseen.
    let k = docs.len() + 1_000;
    let ns = Namespace::open(store, "docs").await.expect("open");
    let query = Lcg(99).vector(dim);
    let got: Vec<String> = ns
        .query(&query, k)
        .await
        .expect("query")
        .into_iter()
        .map(|hit| hit.id)
        .collect();
    assert_eq!(
        got,
        naive_top_k(metric, &query, &docs, k),
        "query vs the replayed state"
    );
    docs
}

/// Check every index the manifest names against its segment. The prefix is read
/// by hand here (see `src/index.rs` for the layout), so this is a second reading
/// of the format and not the decoder checking itself.
async fn check_indexes(store: &Arc<dyn ObjectStore>, dim: usize, manifest: &itzma_core::Manifest) {
    for (id, entry) in &manifest.indexes {
        assert!(
            manifest.live_segments.contains(id),
            "the manifest lists an index for {id}, which is not live"
        );
        let data = store
            .get(&segment::segment_path("docs", id))
            .await
            .expect("indexed segment is missing")
            .bytes()
            .await
            .expect("bytes");
        let path = object_store::path::Path::from(format!(
            "docs/segments/{id}/index-{}.bin",
            entry.generation
        ));
        let index = store
            .get(&path)
            .await
            .unwrap_or_else(|e| panic!("the index {path} is listed but missing: {e}"))
            .bytes()
            .await
            .expect("bytes");

        let u32_at = |b: &[u8], at: usize| u32::from_le_bytes(b[at..at + 4].try_into().unwrap());
        let u64_at = |b: &[u8], at: usize| u64::from_le_bytes(b[at..at + 8].try_into().unwrap());
        assert!(index.len() >= 36, "{path} is shorter than a prefix");
        assert_eq!(&index[..4], b"ITZI", "{path} magic");
        assert_eq!(u32_at(&index, 4), 1, "{path} container version");
        let kind = match entry.kind.as_str() {
            "sq8" => 1,
            "hnsw" => 2,
            other => panic!("{path} is listed with the unknown kind {other}"),
        };
        assert_eq!(u32_at(&index, 8), kind, "{path} kind");
        assert_eq!(u32_at(&index, 12) as usize, dim, "{path} dimension");
        assert_eq!(
            u64::from(u32_at(&index, 16)),
            u64_at(&data, 12),
            "{path} rows against the segment's document count"
        );
        assert_eq!(u64_at(&index, 20), data.len() as u64, "{path} data size");
        let head_len = u64_at(&index, 28);
        assert!(
            (36..=index.len() as u64).contains(&head_len),
            "{path} head length {head_len}"
        );
    }
}

/// The state of a WAL nothing has fenced: every live segment is checked
/// against a rebuild from its range, and the whole log is replayed.
async fn replayed_from_the_wal(
    store: &Arc<dyn ObjectStore>,
    manifest: &itzma_core::Manifest,
) -> Layer {
    let log = wal::read_log(&**store, "docs", 0).await.expect("wal");
    for id in &manifest.live_segments {
        let (lo, hi) = segment::parse_segment_id(id).expect("segment id");
        let mut layer = Layer::new();
        for (seq, entry) in &log {
            if (lo..=hi).contains(seq) {
                for op in entry.ops.clone() {
                    layer.apply(op);
                }
            }
        }
        if lo == 1 {
            layer.drop_tombstones();
        }
        let stored = store
            .get(&segment::segment_path("docs", id))
            .await
            .expect("live segment is missing")
            .bytes()
            .await
            .expect("bytes");
        // A namespace can hold segments of both versions, so rebuild in the
        // version the object was written in.
        let version = u32::from_le_bytes(stored[4..8].try_into().expect("version bytes"));
        let rebuilt =
            segment::encode_with_schema(&layer, &manifest.schema, version).expect("encode");
        assert_eq!(stored, rebuilt, "segment {id} differs from its WAL range");
    }

    let mut state = Layer::new();
    for (_, entry) in &log {
        for op in entry.ops.clone() {
            state.apply(op);
        }
    }
    state
}

/// The state of a WAL some of whose objects are fenced: the live segments,
/// oldest first, then the tail past the fold point.
async fn replayed_from_segments(
    store: &Arc<dyn ObjectStore>,
    dim: usize,
    manifest: &itzma_core::Manifest,
) -> Layer {
    let mut state = Layer::new();
    for id in &manifest.live_segments {
        let bytes = store
            .get(&segment::segment_path("docs", id))
            .await
            .expect("live segment is missing")
            .bytes()
            .await
            .expect("bytes");
        let layer = segment::decode(id, &bytes, dim).expect("live segment decodes");
        state.overlay(layer);
    }
    let tail = wal::read_log(&**store, "docs", manifest.wal_folded_through)
        .await
        .expect("wal tail");
    for (_, entry) in tail {
        for op in entry.ops {
            state.apply(op);
        }
    }
    state
}

/// Rewrite namespace `docs`'s manifest `segment_format` as raw JSON with a
/// plain put, so it works on every backend (`LocalFileSystem` has no
/// conditional update) and can set a level `upgrade_segment_format` refuses.
pub async fn set_segment_format(store: &dyn ObjectStore, level: u32) {
    let path = manifest::manifest_path("docs");
    let raw = store.get(&path).await.unwrap().bytes().await.unwrap();
    let mut json: serde_json::Value = serde_json::from_slice(&raw).unwrap();
    json["segment_format"] = level.into();
    store
        .put(&path, serde_json::to_vec(&json).unwrap().into())
        .await
        .unwrap();
}

/// The format version stamped in each live segment of namespace `docs`,
/// oldest first.
pub async fn live_segment_versions(store: &dyn ObjectStore) -> Vec<u32> {
    let (m, _) = manifest::load(store, "docs").await.expect("manifest");
    let mut versions = Vec::new();
    for id in &m.live_segments {
        let bytes = store
            .get_range(&segment::segment_path("docs", id), 4..8)
            .await
            .expect("segment");
        versions.push(u32::from_le_bytes(bytes[..].try_into().expect("4 bytes")));
    }
    versions
}

/// A namespace with no attribute schema: a manifest with no `attrs` and a format 1
/// schema object, so it is free-form.
pub async fn legacy_namespace(store: &dyn ObjectStore) {
    let manifest: manifest::Manifest = serde_json::from_value(serde_json::json!({
        "format_version": 1,
        "schema": {"dim": 2, "metric": "l2"},
        "live_segments": [],
        "wal_folded_through": 0,
        "wal_format": 4,
        "segment_format": 2
    }))
    .unwrap();
    manifest::create(store, "docs", &manifest).await.unwrap();
    store
        .put(
            &itzma_core::schema_object::schema_path("docs"),
            Bytes::from_static(br#"{"format_version":1,"dim":2,"metric":"l2"}"#).into(),
        )
        .await
        .unwrap();
}
