//! Baseline timings for the write, query and compaction paths.
//!
//! Everything runs on `InMemory`, so the numbers are CPU and code path, not
//! network. A flat query reads and decodes every live object each time
//! because there is no cache yet, and the query benchmarks include that. The
//! indexed query benchmark shows CPU cost only: `InMemory` has no request
//! latency, so it says nothing about the request and byte savings a real
//! object store would show (see `tests/query.rs`'s counting-store test for
//! that). Vectors are random, which ZSTD cannot shrink, so nothing here says
//! how well real embeddings compress. The `kernels` group times the distance
//! functions alone, per row.
//!
//! `make bench` runs it. `cargo test --benches` (part of `make test`) runs each
//! benchmark once as a smoke test, so a change that breaks the API breaks the
//! build.
//!
//! The `decoded` group repeats a few query benchmarks on a handle with a
//! decoded-structure cache. Criterion's warm-up fills it, so those
//! numbers are warm queries: the cold path is the `query` and `filtered`
//! groups.

use std::cell::OnceCell;
use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use itzma_core::quant::Sq8Codec;
use itzma_core::search::distance;
use itzma_core::{
    AttrColumn, AttrSchema, AttrType, CompactionOptions, Compare, DEFAULT_STREAM_TARGET_BYTES,
    DecodedCache, DecodedCacheOptions, DocColumns, Document, FilterOptions, IndexOptions, Metric,
    Namespace, Predicate, QueryOptions, Schema, StreamOptions,
};
use object_store::ObjectStore;
use object_store::memory::InMemory;
use tokio::runtime::Runtime;

const DIM: usize = 128;
const BATCH: usize = 1_000;
const SCHEMA: Schema = Schema::new(DIM, Metric::L2);

/// xorshift64, so the data is the same on every run without a `rand` dependency.
struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        // The top 24 bits fit an f32 mantissa exactly.
        (self.0 >> 40) as f32 / (1u32 << 23) as f32 - 1.0
    }

    fn vector(&mut self) -> Vec<f32> {
        (0..DIM).map(|_| self.next_f32()).collect()
    }
}

/// `count` documents with ids `doc-<first>` onward.
fn documents(rng: &mut Rng, first: usize, count: usize) -> Vec<Document> {
    (first..first + count)
        .map(|n| Document {
            id: format!("doc-{n}"),
            vector: rng.vector(),
            attrs: None,
        })
        .collect()
}

async fn new_namespace() -> Namespace {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    Namespace::create(store, "bench", SCHEMA)
        .await
        .expect("create")
}

/// A namespace whose whole content is a WAL tail of `objects` objects of 100
/// documents each, and its store, so a second handle can open it.
async fn small_tail(objects: usize) -> (Arc<dyn ObjectStore>, Namespace) {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let ns = Namespace::create(store.clone(), "bench", SCHEMA)
        .await
        .expect("create");
    let mut rng = Rng(32);
    for b in 0..objects {
        ns.upsert(documents(&mut rng, b * 100, 100))
            .await
            .expect("upsert");
    }
    (store, ns)
}

/// A namespace holding `batches` WAL objects of [`BATCH`] documents each.
async fn with_tail(batches: usize) -> Namespace {
    let ns = new_namespace().await;
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for b in 0..batches {
        ns.upsert(documents(&mut rng, b * BATCH, BATCH))
            .await
            .expect("upsert");
    }
    ns
}

fn upsert(c: &mut Criterion, rt: &Runtime) {
    let mut group = c.benchmark_group("upsert");
    group.throughput(Throughput::Elements(BATCH as u64));
    group.bench_function("1k_dim128", |b| {
        // Timed by hand: setup is async and criterion's `iter_batched` runs it
        // inside the runtime. A fresh namespace each time also keeps the store
        // from growing across iterations, which would swamp the timing.
        b.iter_custom(|iters| {
            rt.block_on(async {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let ns = new_namespace().await;
                    let docs = documents(&mut Rng(1), 0, BATCH);
                    let start = Instant::now();
                    black_box(ns.upsert(docs).await.expect("upsert"));
                    total += start.elapsed();
                }
                total
            })
        });
    });

    // a 100k-row load at dimension 128 with a number and a text
    // attribute, three ways. `documents` is the bridge's old path (rows split
    // into 32 MiB batches, each `upsert`ed); `columnar` writes the same
    // batches as `DocColumns`; `stream` hands 10k-row batches to
    // `upsert_stream` with the default options, which encodes the next chunk
    // while the current one is put.
    group.sample_size(10);
    group.throughput(Throughput::Elements(LOAD_ROWS as u64));
    let per_batch = DEFAULT_STREAM_TARGET_BYTES / (DIM * 4 + 64);
    let docs = OnceCell::new();
    group.bench_function("documents_100k_dim128", |b| {
        let docs = docs.get_or_init(load_documents);
        b.iter_custom(|iters| {
            rt.block_on(async {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let ns = load_namespace().await;
                    let batches: Vec<Vec<Document>> =
                        docs.chunks(per_batch).map(<[Document]>::to_vec).collect();
                    let start = Instant::now();
                    for batch in batches {
                        black_box(ns.upsert(batch).await.expect("upsert"));
                    }
                    total += start.elapsed();
                }
                total
            })
        });
    });
    let big = OnceCell::new();
    group.bench_function("columnar_100k_dim128", |b| {
        let big = big.get_or_init(|| load_columns(per_batch));
        b.iter_custom(|iters| {
            rt.block_on(async {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let ns = load_namespace().await;
                    let start = Instant::now();
                    for batch in big {
                        black_box(ns.upsert_columns(batch).await.expect("upsert_columns"));
                    }
                    total += start.elapsed();
                }
                total
            })
        });
    });
    let small = OnceCell::new();
    group.bench_function("stream_100k_dim128", |b| {
        let small = small.get_or_init(|| load_columns(10_000));
        b.iter_custom(|iters| {
            rt.block_on(async {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let ns = load_namespace().await;
                    let input = futures::stream::iter(small.clone().into_iter().map(Ok));
                    let start = Instant::now();
                    black_box(
                        ns.upsert_stream(input, &StreamOptions::default())
                            .await
                            .expect("upsert_stream"),
                    );
                    total += start.elapsed();
                }
                total
            })
        });
    });
    group.finish();
}

const LOAD_ROWS: usize = 100_000;

async fn load_namespace() -> Namespace {
    let schema = Schema::new(DIM, Metric::L2).with_attrs(
        AttrSchema::new()
            .field("price", AttrType::Number)
            .field("tag", AttrType::Text),
    );
    Namespace::create(Arc::new(InMemory::new()), "bench", schema)
        .await
        .expect("create")
}

fn load_documents() -> Vec<Document> {
    let mut rng = Rng(39);
    (0..LOAD_ROWS)
        .map(|n| Document {
            id: format!("doc-{n}"),
            vector: rng.vector(),
            attrs: Some(
                serde_json::json!({ "price": n as f64 / 4.0, "tag": format!("tag-{}", n % 50) }),
            ),
        })
        .collect()
}

/// The same rows as `load_documents`, as columnar batches of `per_batch`.
fn load_columns(per_batch: usize) -> Vec<DocColumns> {
    let mut rng = Rng(39);
    (0..LOAD_ROWS)
        .step_by(per_batch)
        .map(|from| {
            let to = (from + per_batch).min(LOAD_ROWS);
            let (mut ids, mut id_offsets) = (Vec::new(), vec![0u32]);
            let (mut tags, mut tag_offsets) = (Vec::new(), vec![0u32]);
            let mut vectors = Vec::with_capacity((to - from) * DIM);
            for n in from..to {
                ids.extend_from_slice(format!("doc-{n}").as_bytes());
                id_offsets.push(ids.len() as u32);
                tags.extend_from_slice(format!("tag-{}", n % 50).as_bytes());
                tag_offsets.push(tags.len() as u32);
                vectors.extend(rng.vector());
            }
            DocColumns::new(ids, id_offsets, vectors, DIM)
                .expect("columns")
                .with_attr(
                    "price",
                    AttrColumn::F64 {
                        valid: vec![true; to - from],
                        values: (from..to).map(|n| n as f64 / 4.0).collect(),
                    },
                )
                .expect("price")
                .with_attr(
                    "tag",
                    AttrColumn::Text {
                        valid: vec![true; to - from],
                        bytes: tags,
                        offsets: tag_offsets,
                    },
                )
                .expect("tag")
        })
        .collect()
}

fn query(c: &mut Criterion, rt: &Runtime) {
    let probe = Rng(7).vector();
    let mut group = c.benchmark_group("query");
    group.sample_size(20);

    for batches in [10, 100] {
        let docs = batches * BATCH;
        let ns = OnceCell::new();
        group.throughput(Throughput::Elements(docs as u64));
        group.bench_function(format!("segment/{}k_dim128", docs / 1000), |b| {
            let ns = ns.get_or_init(|| {
                rt.block_on(async {
                    let ns = with_tail(batches).await;
                    ns.compact(&CompactionOptions::default())
                        .await
                        .expect("compact");
                    ns
                })
            });
            b.to_async(rt)
                .iter(|| async { black_box(ns.query(&probe, 10).await.expect("query")) });
        });
    }

    group.throughput(Throughput::Elements(10 * BATCH as u64));
    let tail = OnceCell::new();
    group.bench_function("wal_tail/10k_dim128", |b| {
        let ns = tail.get_or_init(|| rt.block_on(with_tail(10)));
        b.to_async(rt)
            .iter(|| async { black_box(ns.query(&probe, 10).await.expect("query")) });
    });

    // a warm query against an unchanged tail of 32 objects. Before
    // the incremental tail every query lists, fetches, decodes and folds all
    // 32; after, it lists once and reuses the folded layer.
    group.throughput(Throughput::Elements(32 * 100));
    let tail32 = OnceCell::new();
    group.bench_function("wal_tail/32_objects", |b| {
        let ns = tail32.get_or_init(|| rt.block_on(small_tail(32)).1);
        b.to_async(rt)
            .iter(|| async { black_box(ns.query(&probe, 10).await.expect("query")) });
    });

    // The same 100k-vector segment as above, indexed with one SQ8 partition
    // (no clustering) and with 64, beside the flat one:
    // what an index saves, and what clustering adds. The default would be about
    // 316 here, but `make test` runs each benchmark once in a debug build, where
    // that build takes over a minute. Random vectors have no clusters, so this times the code path, not
    // recall (`examples/recall.rs` measures that).
    for (name, options) in [
        (
            "segment/100k_dim128_indexed",
            IndexOptions::default().nlist(Some(1)),
        ),
        (
            "segment/100k_dim128_indexed_clustered_64",
            IndexOptions::default().nlist(Some(64)),
        ),
    ] {
        let indexed = OnceCell::new();
        group.throughput(Throughput::Elements(100 * BATCH as u64));
        group.bench_function(name, |b| {
            let indexed = indexed.get_or_init(|| {
                rt.block_on(async {
                    let ns = with_tail(100).await;
                    ns.compact(&CompactionOptions::default())
                        .await
                        .expect("compact");
                    ns.index(&options).await.expect("index");
                    ns
                })
            });
            b.to_async(rt)
                .iter(|| async { black_box(indexed.query(&probe, 10).await.expect("query")) });
        });
    }

    // 16 indexed segments of 10,000 rows each (the own suggested
    // bench), rather than one segment with many partitions: this times the
    // engine's per-source fan-out and shadow checks across sources, which
    // is what the shared bound and the SQ8 cursor change, not what one
    // segment's own partition scan costs.
    // `max_segments: usize::MAX`, not `CompactionOptions::default()` (8):
    // the default triggers a merge the moment a 9th segment would exist,
    // collapsing most of these 16 folds back into one segment — exactly
    // the "no shadowing, no filter, nlist = 1, every source fills top_k on
    // its first batch" case that never exercises the bound's stopping
    // rules or a second wave.
    let unbounded_segments = CompactionOptions {
        max_segments: usize::MAX,
        max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
    };
    group.throughput(Throughput::Elements(16 * 10_000));
    let many = OnceCell::new();
    group.bench_function("many_segments/16x10k_dim128", |b| {
        let many = many.get_or_init(|| {
            rt.block_on(async {
                let ns = new_namespace().await;
                let mut rng = Rng(0x1234_5678_9abc_def0);
                let mut next_id = 0usize;
                for _segment in 0..16 {
                    for _ in 0..(10_000 / BATCH) {
                        ns.upsert(documents(&mut rng, next_id, BATCH))
                            .await
                            .expect("upsert");
                        next_id += BATCH;
                    }
                    ns.compact(&unbounded_segments).await.expect("compact");
                }
                ns.index(&IndexOptions::default()).await.expect("index");
                ns
            })
        });
        b.to_async(rt)
            .iter(|| async { black_box(many.query(&probe, 10).await.expect("query")) });
    });

    group.finish();
}

fn compact(c: &mut Criterion, rt: &Runtime) {
    let mut group = c.benchmark_group("compact");
    group.sample_size(10);
    group.throughput(Throughput::Elements(100 * BATCH as u64));
    group.bench_function("fold_100_batches_of_1k", |b| {
        b.iter_custom(|iters| {
            rt.block_on(async {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let ns = with_tail(100).await;
                    let start = Instant::now();
                    black_box(
                        ns.compact(&CompactionOptions::default())
                            .await
                            .expect("compact"),
                    );
                    total += start.elapsed();
                }
                total
            })
        });
    });
    // Eight folded segments of 12,500 documents and a tail of one batch,
    // merged into one segment.
    group.throughput(Throughput::Elements(
        MERGE_SEGMENTS * MERGE_ROWS + BATCH as u64,
    ));
    group.bench_function("merge_8_segments_100k_dim128", |b| {
        b.iter_custom(|iters| {
            rt.block_on(async {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let ns = merge_input().await;
                    let merge = CompactionOptions {
                        max_segments: 1,
                        ..CompactionOptions::default()
                    };
                    let start = Instant::now();
                    black_box(ns.compact(&merge).await.expect("merge"));
                    total += start.elapsed();
                }
                total
            })
        });
    });
    group.finish();
}

const MERGE_SEGMENTS: u64 = 8;
const MERGE_ROWS: u64 = 12_500;

/// [`MERGE_SEGMENTS`] folded segments of [`MERGE_ROWS`] documents each and a
/// tail of one batch, ready for a merge.
async fn merge_input() -> Namespace {
    let ns = new_namespace().await;
    let mut rng = Rng(0x5eed_0037);
    let fold = CompactionOptions {
        max_segments: MERGE_SEGMENTS as usize + 1,
        ..CompactionOptions::default()
    };
    let rows = MERGE_ROWS as usize;
    for s in 0..MERGE_SEGMENTS as usize {
        ns.upsert(documents(&mut rng, s * rows, rows))
            .await
            .expect("upsert");
        ns.compact(&fold).await.expect("fold");
    }
    ns.upsert(documents(&mut rng, MERGE_SEGMENTS as usize * rows, BATCH))
        .await
        .expect("upsert");
    ns
}

const FILTER_DIM: usize = 4;

/// The schema the `filter` and `filtered` groups use.
fn filter_schema() -> Schema {
    Schema::new(FILTER_DIM, Metric::L2).with_attrs(
        AttrSchema::new()
            .field("bucket", AttrType::Number)
            .field("active", AttrType::Bool),
    )
}

/// A compacted namespace of `batches` times [`BATCH`] documents whose `bucket`
/// is `n % 100` and whose `active` is true for even `n`.
async fn filter_segment(batches: usize) -> Namespace {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let ns = Namespace::create(store, "bench", filter_schema())
        .await
        .expect("create");
    for b in 0..batches {
        let docs = (b * BATCH..(b + 1) * BATCH)
            .map(|n| Document {
                id: format!("doc-{n:08}"),
                vector: vec![0.0; FILTER_DIM],
                attrs: Some(serde_json::json!({ "bucket": n % 100, "active": n % 2 == 0 })),
            })
            .collect();
        ns.upsert(docs).await.expect("upsert");
    }
    ns.compact(&CompactionOptions::default())
        .await
        .expect("compact");
    ns
}

/// `filter` with a loose predicate (half the documents match, so a page of 100
/// stops after the first block) and a selective one (one in a hundred, so it
/// reads most of the ids and attributes). `InMemory` has no request latency,
/// so this is CPU only; the requests a block read costs are counted in the
/// unit tests of `segment/ranged.rs`.
fn filter(c: &mut Criterion, rt: &Runtime) {
    let mut group = c.benchmark_group("filter");
    group.sample_size(10);
    let loose = Predicate::is_true("active");
    let selective = Predicate::number("bucket", Compare::Eq, 7.0);
    let options = FilterOptions::default();
    for batches in [1, 20] {
        let docs = batches * BATCH;
        let ns = OnceCell::new();
        for (kind, predicate) in [("loose", &loose), ("selective", &selective)] {
            group.throughput(Throughput::Elements(docs as u64));
            group.bench_function(format!("{kind}/{}k", docs / 1000), |b| {
                let ns = ns.get_or_init(|| rt.block_on(filter_segment(batches)));
                b.to_async(rt).iter(|| async {
                    black_box(ns.filter(predicate, &options).await.expect("filter"))
                });
            });
        }
    }
    group.finish();
}

/// A compacted namespace like [`filter_segment`] with vectors that vary by
/// document, indexed.
async fn filtered_query_segment(batches: usize) -> Namespace {
    filtered_query_segment_at(batches, 2).await
}

/// [`filtered_query_segment`] written at segment format `level`.
async fn filtered_query_segment_at(batches: usize, level: u32) -> Namespace {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let ns = Namespace::create(store, "bench", filter_schema())
        .await
        .expect("create");
    ns.upgrade_segment_format(level).await.expect("level");
    for b in 0..batches {
        let docs = (b * BATCH..(b + 1) * BATCH)
            .map(|n| {
                let x = (n % 1013) as f32;
                Document {
                    id: format!("doc-{n:08}"),
                    vector: vec![x, x * 0.5, (n % 7) as f32, 1.0],
                    attrs: Some(serde_json::json!({ "bucket": n % 100, "active": n % 2 == 0 })),
                }
            })
            .collect();
        ns.upsert(docs).await.expect("upsert");
    }
    ns.compact(&CompactionOptions::default())
        .await
        .expect("compact");
    ns.index(&IndexOptions::default().min_rows(0))
        .await
        .expect("index");
    ns
}

/// `query_with` with a loose and a selective filter beside
/// the unfiltered query. `InMemory` has no request latency, so this
/// is CPU only: what a filter adds in attribute reads and predicate tests.
fn filtered(c: &mut Criterion, rt: &Runtime) {
    let mut group = c.benchmark_group("filtered");
    group.sample_size(10);
    let probe = [3.0, 1.5, 2.0, 1.0];
    let filters = [
        ("unfiltered", None),
        ("loose", Some(Predicate::is_true("active"))),
        (
            "selective",
            Some(Predicate::number("bucket", Compare::Eq, 7.0)),
        ),
    ];
    for batches in [1, 20] {
        let docs = batches * BATCH;
        let ns = OnceCell::new();
        let ns = || ns.get_or_init(|| rt.block_on(filtered_query_segment(batches)));
        group.throughput(Throughput::Elements(docs as u64));
        for (kind, predicate) in &filters {
            let dense = QueryOptions::default().filter(predicate.clone());
            group.bench_function(format!("dense/{kind}/{}k", docs / 1000), |b| {
                let ns = ns();
                b.to_async(rt).iter(|| async {
                    black_box(ns.query_with(&probe, 10, &dense).await.expect("query"))
                });
            });
        }
    }
    group.finish();
}

/// Per-row cost of each distance, over 1,000 rows so the call
/// overhead of criterion does not dominate. Throughput is rows.
fn kernels(c: &mut Criterion) {
    const ROWS: usize = 1_000;
    let mut group = c.benchmark_group("kernels");
    for dim in [128, 960] {
        let mut rng = Rng(11);
        let mut vector = |n: usize| (0..n).map(|_| rng.next_f32()).collect::<Vec<f32>>();
        let query = vector(dim);
        let matrix = vector(ROWS * dim);
        let codec = Sq8Codec::fit(&matrix, dim).expect("fit");
        let codes: Vec<Vec<u8>> = matrix.chunks_exact(dim).map(|r| codec.encode(r)).collect();
        group.throughput(Throughput::Elements(ROWS as u64));

        for (name, metric) in [("l2", Metric::L2), ("dot", Metric::Dot)] {
            group.bench_function(format!("{name}/{dim}"), |b| {
                b.iter(|| {
                    let mut sum = 0f32;
                    for row in matrix.chunks_exact(dim) {
                        sum += distance(metric, black_box(&query), row);
                    }
                    black_box(sum)
                });
            });
        }
        for (name, metric) in [
            ("sq8_l2", Metric::L2),
            ("sq8_dot", Metric::Dot),
            ("sq8_cosine", Metric::Cosine),
        ] {
            let prepared = codec.prepare(metric, &query);
            group.bench_function(format!("{name}/{dim}"), |b| {
                b.iter(|| {
                    let mut sum = 0f32;
                    for code in &codes {
                        sum += prepared.distance(black_box(code));
                    }
                    black_box(sum)
                });
            });
        }
    }
    group.finish();
}

/// Warm queries on a handle with a decoded cache, beside the same
/// namespaces' cold numbers in `query` and `filtered`.
fn decoded(c: &mut Criterion, rt: &Runtime) {
    let mut group = c.benchmark_group("decoded");
    group.sample_size(20);
    let cached =
        |ns: Namespace| ns.with_decoded_cache(DecodedCache::new(DecodedCacheOptions::default()));
    let probe = Rng(7).vector();

    group.throughput(Throughput::Elements(100 * BATCH as u64));
    let flat = OnceCell::new();
    group.bench_function("query/segment/100k_dim128", |b| {
        let flat = flat.get_or_init(|| {
            rt.block_on(async {
                let ns = with_tail(100).await;
                ns.compact(&CompactionOptions::default())
                    .await
                    .expect("compact");
                cached(ns)
            })
        });
        b.to_async(rt)
            .iter(|| async { black_box(flat.query(&probe, 10).await.expect("query")) });
    });

    for (name, options) in [
        (
            "query/segment/100k_dim128_indexed",
            IndexOptions::default().nlist(Some(1)),
        ),
        (
            "query/segment/100k_dim128_indexed_clustered_64",
            IndexOptions::default().nlist(Some(64)),
        ),
    ] {
        let indexed = OnceCell::new();
        group.bench_function(name, |b| {
            let indexed = indexed.get_or_init(|| {
                rt.block_on(async {
                    let ns = with_tail(100).await;
                    ns.compact(&CompactionOptions::default())
                        .await
                        .expect("compact");
                    ns.index(&options).await.expect("index");
                    cached(ns)
                })
            });
            b.to_async(rt)
                .iter(|| async { black_box(indexed.query(&probe, 10).await.expect("query")) });
        });
    }

    let loose = QueryOptions::default().filter(Some(Predicate::is_true("active")));
    let small_probe = [3.0, 1.5, 2.0, 1.0];
    group.throughput(Throughput::Elements(20 * BATCH as u64));
    let filtered = OnceCell::new();
    group.bench_function("filtered/dense/loose/20k", |b| {
        let filtered = filtered
            .get_or_init(|| rt.block_on(async { cached(filtered_query_segment(20).await) }));
        b.to_async(rt).iter(|| async {
            black_box(
                filtered
                    .query_with(&small_probe, 10, &loose)
                    .await
                    .expect("query"),
            )
        });
    });
    group.finish();
}

/// A 100k-row namespace compacted into one segment, indexed when `options` is
/// given.
async fn segment_100k(options: Option<&IndexOptions>) -> Namespace {
    let ns = with_tail(100).await;
    ns.compact(&CompactionOptions::default())
        .await
        .expect("compact");
    if let Some(options) = options {
        ns.index(options).await.expect("index");
    }
    ns
}

/// Where CPU work runs. `concurrent` runs eight queries at once on a
/// 4-worker runtime, seven through a clustered index and one exact flat scan
/// of the same 100k rows: without a compute pool the flat scan holds a worker
/// for its whole scan. `build` times a clustered index rebuild of 100k rows
/// (64 partitions, not the default 316, for the same debug-build reason as
/// the `query` group).
fn cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu");
    group.sample_size(10);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("runtime");
    let clustered = IndexOptions::default().nlist(Some(64));
    let ns: OnceCell<Arc<Namespace>> = OnceCell::new();
    let ns = || ns.get_or_init(|| Arc::new(rt.block_on(segment_100k(Some(&clustered)))));
    let mut rng = Rng(11);
    let probes: Vec<Vec<f32>> = (0..8).map(|_| rng.vector()).collect();
    group.bench_function("concurrent/8_queries_4_workers", |b| {
        let ns = ns();
        b.iter_custom(|iters| {
            rt.block_on(async {
                let start = Instant::now();
                for _ in 0..iters {
                    let runs = probes.iter().enumerate().map(|(i, probe)| {
                        let (ns, probe) = (ns.clone(), probe.clone());
                        tokio::spawn(async move {
                            let options = QueryOptions::default().exact(i == 0);
                            ns.query_with(&probe, 10, &options).await.expect("query")
                        })
                    });
                    for run in futures::future::join_all(runs).await {
                        black_box(run.expect("join"));
                    }
                }
                start.elapsed()
            })
        });
    });

    let rebuild = clustered.clone().rebuild(true);
    group.bench_function("build/clustered_100k_dim128", |b| {
        let ns = ns();
        b.iter_custom(|iters| {
            rt.block_on(async {
                let start = Instant::now();
                for _ in 0..iters {
                    black_box(ns.index(&rebuild).await.expect("index"));
                }
                start.elapsed()
            })
        });
    });
    group.finish();
}

/// [`filter_segment`]'s documents, unindexed, folded at segment format
/// `level` (3 stores the attributes as columns), into `segments`
/// segments.
async fn filter_segment_at(batches: usize, level: u32, segments: usize) -> Namespace {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let ns = Namespace::create(store, "bench", filter_schema())
        .await
        .expect("create");
    ns.upgrade_segment_format(level).await.expect("level");
    let roomy = CompactionOptions {
        max_segments: segments + 1,
        ..CompactionOptions::default()
    };
    let per = batches.div_ceil(segments);
    for b in 0..batches {
        let docs = (b * BATCH..(b + 1) * BATCH)
            .map(|n| Document {
                id: format!("doc-{n:08}"),
                vector: vec![0.0; FILTER_DIM],
                attrs: Some(serde_json::json!({ "bucket": n % 100, "active": n % 2 == 0 })),
            })
            .collect();
        ns.upsert(docs).await.expect("upsert");
        if (b + 1) % per == 0 || b + 1 == batches {
            ns.compact(&roomy).await.expect("compact");
        }
    }
    ns
}

/// The same reads and merge over version 2 and version 3 segments:
/// `filter` with the `filter/` group's two predicates, a whole `scan`, and a
/// merge of eight segments. Version 3 rebuilds attributes from columns where
/// version 2 parses JSON; zone maps are not used yet.
fn columnar(c: &mut Criterion, rt: &Runtime) {
    let mut group = c.benchmark_group("columnar");
    group.sample_size(10);
    let loose = Predicate::is_true("active");
    let selective = Predicate::number("bucket", Compare::Eq, 7.0);
    let batches = 20;
    group.throughput(Throughput::Elements((batches * BATCH) as u64));
    for (name, level) in [("v2", 2), ("v3", 3)] {
        let ns = OnceCell::new();
        for (kind, predicate) in [("loose", &loose), ("selective", &selective)] {
            group.bench_function(format!("filter/{kind}/20k/{name}"), |b| {
                let ns = ns.get_or_init(|| rt.block_on(filter_segment_at(batches, level, 1)));
                b.to_async(rt).iter(|| async {
                    black_box(
                        ns.filter(predicate, &FilterOptions::default())
                            .await
                            .expect("filter"),
                    )
                });
            });
        }
        let dense = OnceCell::new();
        for (kind, predicate) in [("loose", &loose), ("selective", &selective)] {
            let options = QueryOptions::default().filter(Some(predicate.clone()));
            group.bench_function(format!("filtered/dense/{kind}/20k/{name}"), |b| {
                let ns =
                    dense.get_or_init(|| rt.block_on(filtered_query_segment_at(batches, level)));
                b.to_async(rt).iter(|| async {
                    black_box(
                        ns.query_with(&[3.0, 1.5, 2.0, 1.0], 10, &options)
                            .await
                            .expect("query"),
                    )
                });
            });
        }
        group.bench_function(format!("fold/20k_dim128/{name}"), |b| {
            b.iter_custom(|iters| {
                rt.block_on(async {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let ns = columnar_input(level, 1, false).await;
                        let start = Instant::now();
                        black_box(
                            ns.compact(&CompactionOptions::default())
                                .await
                                .expect("fold"),
                        );
                        total += start.elapsed();
                    }
                    total
                })
            });
        });
        // Eight segments of about 1.3 MB, over the merge's 1 MiB threshold,
        // so they are read in windows.
        group.bench_function(format!("merge_8_segments/20k_dim128/{name}"), |b| {
            b.iter_custom(|iters| {
                rt.block_on(async {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let ns = columnar_input(level, 8, true).await;
                        let merge = CompactionOptions {
                            max_segments: 1,
                            ..CompactionOptions::default()
                        };
                        let start = Instant::now();
                        black_box(ns.compact(&merge).await.expect("merge"));
                        total += start.elapsed();
                    }
                    total
                })
            });
        });
    }
    group.finish();
}

/// 20,000 documents of dimension 128 with [`filter_schema`]'s attributes, at
/// segment format `level`, in `parts` batches; each batch folded when
/// `fold`, else left in the tail.
async fn columnar_input(level: u32, parts: usize, fold: bool) -> Namespace {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let schema = Schema::new(DIM, Metric::L2).with_attrs(filter_schema().attrs);
    let ns = Namespace::create(store, "bench", schema)
        .await
        .expect("create");
    ns.upgrade_segment_format(level).await.expect("level");
    let roomy = CompactionOptions {
        max_segments: parts + 1,
        ..CompactionOptions::default()
    };
    let mut rng = Rng(0x5eed_0033);
    let per = 20_000 / parts;
    for p in 0..parts {
        let docs = (p * per..(p + 1) * per)
            .map(|n| Document {
                id: format!("doc-{n:08}"),
                vector: rng.vector(),
                attrs: Some(serde_json::json!({ "bucket": n % 100, "active": n % 2 == 0 })),
            })
            .collect();
        ns.upsert(docs).await.expect("upsert");
        if fold {
            ns.compact(&roomy).await.expect("fold");
        }
    }
    ns
}

fn benches(c: &mut Criterion) {
    kernels(c);
    cpu(c);
    let rt = Runtime::new().expect("runtime");
    upsert(c, &rt);
    query(c, &rt);
    compact(c, &rt);
    filter(c, &rt);
    filtered(c, &rt);
    columnar(c, &rt);
    decoded(c, &rt);
}

criterion_group!(core, benches);
criterion_main!(core);
