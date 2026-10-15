//! Itzma core: object-storage-backed vector retrieval.
//!
//! Vectors live as plain objects in a bucket. There is no server in the
//! middle: any process with access to the bucket can write, and any process
//! can query. Correctness comes from two primitives the object store already
//! provides, put-if-absent and conditional update.
//!
//! # Where to start
//!
//! [`Namespace`] is the public API. Create one, `upsert` documents, `query`
//! for nearest neighbors:
//!
//! ```
//! # #[tokio::main]
//! # async fn main() -> itzma_core::Result<()> {
//! use std::sync::Arc;
//!
//! use itzma_core::{Document, Metric, Namespace, Schema};
//! use object_store::memory::InMemory;
//!
//! let store = Arc::new(InMemory::new());
//! let schema = Schema::new(2, Metric::L2);
//! let ns = Namespace::create(store, "docs", schema).await?;
//!
//! ns.upsert(vec![Document { id: "a".into(), vector: vec![1.0, 0.0], attrs: None, sparse: None }]).await?;
//! let hits = ns.query(&[0.0, 0.0], 5).await?;
//! assert_eq!(hits[0].id, "a");
//! # Ok(())
//! # }
//! ```
//!
//! Any `Arc<dyn object_store::ObjectStore>` works as the backend: in-memory
//! for tests, local disk, or S3, GCS and Azure. Note that `LocalFileSystem`
//! does not implement conditional update, which [`manifest::update`],
//! [`manifest::update_if`] and so [`Namespace::compact`] and
//! [`Namespace::upgrade_wal_format`] and [`Namespace::upgrade_segment_format`] need.
//!
//! # Modules
//!
//! | Module | Job |
//! |---|---|
//! | [`namespace`] | The [`Namespace`] handle: create, open, upsert, delete (each also keyed), query, query_sparse, filter, index, compact, gc, maintain, recover |
//! | [`document`] | [`Document`], [`WalOp`], [`Schema`], [`Metric`] |
//! | [`attrs`] | [`AttrSchema`], [`AttrType`]: the typed attribute fields a namespace declares |
//! | [`predicate`] | [`Predicate`]: a typed condition over attributes |
//! | [`filter`] | [`FilterOptions`], [`FilterPage`], [`Row`] and the run behind [`Namespace::filter`] |
//! | [`wal`] | Sequenced write-ahead log: [`wal::append`], [`wal::append_keyed`], [`wal::read_log`] |
//! | [`manifest`] | The namespace manifest and its compare-and-swap update |
//! | [`schema_object`] | `schema.json`, the schema written once at create |
//! | [`segment`] | Segment binary codecs (versions 2 and 3), ids, and the manifest invariant check |
//! | [`layer`] | [`layer::Layer`]: the net effect of applying operations in order |
//! | [`compaction`] | [`Compaction`], [`CompactionOptions`] and the run behind [`Namespace::compact`] |
//! | [`gc`] | [`GcOptions`], [`GcReport`] and the run behind [`Namespace::gc`] |
//! | [`indexer`] | [`IndexOptions`], [`IndexReport`] and the run behind [`Namespace::index`] |
//! | [`sparse`] | [`SparseVector`] storage beside a segment: `sparse.bin` |
//! | [`maintain`] | [`MaintenanceOptions`] and the run behind [`Namespace::maintain`] |
//! | [`columns`] | [`DocColumns`]: a batch of upserts held as columns, for Arrow input |
//! | [`inspect`] | [`inspect::report`]: a read-only look at what a namespace holds and what looks wrong |
//! | [`search`] | Distance metrics and the flat top-k scan |
//! | [`error`] | [`ItzmaError`] and [`Result`] |
//!
//! A query or a compaction run reads at most a byte budget, [`DEFAULT_READ_BUDGET`] unless set
//! with [`Namespace::with_read_budget`], and stops with [`ItzmaError::ReadBudgetExceeded`] past
//! it.
//!
//! `wal`, `manifest`, `layer`, `segment` and `compaction` are public so
//! integration tests can drive them directly. Application code should stay on
//! [`Namespace`].
//!
//! # On-disk layout
//!
//! ```text
//! <namespace>/
//!   schema.json                    dimension, metric and attribute types, written once, never changed
//!   gc/fenced-through              how far the WAL fences reach, a hint for `gc` (only with fence_wal)
//!   manifest/current.json          schema, live segments, WAL fold point
//!   wal/00000000000000000001.json  one object per write batch (binary, see below)
//!   wal/00000000000000000002.json
//!   segments/<lo>-<hi>/data.bin    compacted data, named by the WAL range it covers
//!   segments/<lo>-<hi>/index-<n>.bin an index of that segment, generation n, once `index` built one
//!   segments/<lo>-<hi>/sparse.bin  the segment's sparse vectors, only if a document has one
//! ```
//!
//! The manifest is JSON. WAL objects and segments are binary, and a WAL object
//! is ZSTD-compressed when that makes it smaller. The WAL key keeps
//! its `.json` suffix on purpose: every writer has to try the same key for the
//! same sequence number, or put-if-absent stops deciding the race. A write that carries an
//! idempotency key ([`WriteKey`]) is version 3, the same object with the key in
//! the header. A batch with a sparse vector is version 4, or 5 with a key.
//!
//! Nothing in the layout is ever overwritten but the manifest, and the WAL
//! objects that [`Namespace::gc`] is asked to fence. GC deletes segments that a
//! merge or a later fold superseded, and with
//! [`GcOptions::fence_wal`] replaces folded WAL objects with 8-byte fences that
//! keep put-if-absent working at their keys. Nothing else is removed. The
//! schema is kept in the manifest and again in `schema.json`, which is
//! written once, so losing the manifest does not lose it.
//! [`Namespace::recover`] then writes a new manifest from the schema object, the
//! segment names and the WAL listing.
//! [`Namespace::maintain`] is the call to schedule: it compacts when the WAL tail
//! is long enough and then collects.
//!
//! Every object carries a `format_version` so the encoding can change later. A
//! reader that meets a version it does not know fails with
//! [`ItzmaError::UnsupportedFormat`] instead of guessing. The manifest also
//! records the newest WAL format the namespace may contain, and a build that
//! reads an older one refuses the namespace with
//! [`ItzmaError::WalFormatTooNew`] before it touches the WAL.
//! [`Namespace::upgrade_wal_format`] raises that level, one way. A second level,
//! `segment_format`, does the same for segments: a build that reads an older
//! segment format is refused with [`ItzmaError::SegmentFormatTooNew`], and
//! [`Namespace::upgrade_segment_format`] raises it. Compaction writes
//! the segment version that level allows.
#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![warn(missing_docs)]

pub mod attrs;
mod bloom;
mod bound;
mod budget;
mod cache;
mod columnar;
pub mod columns;
pub mod compaction;
mod cpu;
mod decoded;
pub mod document;
pub mod error;
mod fetch;
pub mod filter;
pub mod gc;
mod index;
pub mod indexer;
pub mod inspect;
mod kernels;
pub mod layer;
pub mod maintain;
pub mod manifest;
pub mod namespace;
pub mod predicate;
pub mod quant;
mod query;
pub mod recover;
mod scan;
pub mod schema_object;
pub mod search;
pub mod segment;
pub mod sparse;
mod sparse_query;
mod stream;
mod view;
pub mod wal;
mod wal_codec;

pub use attrs::{AttrSchema, AttrType};
pub use budget::DEFAULT_READ_BUDGET;
pub use cache::{CacheOptions, CacheStore, DEFAULT_RAM_MAX_BYTES, NvmeOptions};
pub use columns::{AttrColumn, DocColumns, SparseColumn};
pub use compaction::{Compaction, CompactionOptions};
pub use cpu::{CpuThreads, configure_cpu};
pub use decoded::{DEFAULT_DECODED_MAX_BYTES, DecodedCache, DecodedCacheOptions};
pub use document::{Document, Metric, Schema, SparseVector, WalOp};
pub use error::{ItzmaError, Result};
pub use fetch::FetchPolicy;
pub use filter::{FilterOptions, FilterPage, Row};
pub use gc::{GcOptions, GcReport};
pub use indexer::{IndexKind, IndexOptions, IndexReport};
pub use maintain::{CompactTrigger, MaintenanceOptions, MaintenanceReport};
pub use manifest::{IndexEntry, Manifest};
pub use namespace::Namespace;
pub use predicate::{AttrValue, Compare, Predicate, TextCompare};
pub use query::{DEFAULT_RERANK, QueryOptions};
pub use recover::RecoverReport;
pub use search::Hit;
pub use sparse_query::SparseQueryOptions;
pub use stream::{
    DEFAULT_STREAM_DEPTH, DEFAULT_STREAM_TARGET_BYTES, StreamError, StreamOptions, StreamReport,
};
pub use wal::WriteKey;
