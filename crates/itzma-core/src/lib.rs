//! Itzma core: object-storage-backed vector retrieval.
//!
//! Vectors live as plain objects in a bucket. There is no server in the
//! middle: any process with access to the bucket can write, and any process
//! can query. Correctness comes from two primitives the object store already
//! provides, put-if-absent and conditional update.
#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![warn(missing_docs)]

pub mod attrs;
mod bloom;
mod bound;
mod budget;
mod cache;
mod columnar;
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
mod view;
pub mod wal;
mod wal_codec;

pub use attrs::{AttrSchema, AttrType};
pub use budget::DEFAULT_READ_BUDGET;
pub use cache::{CacheOptions, CacheStore, DEFAULT_RAM_MAX_BYTES, NvmeOptions};
pub use compaction::{Compaction, CompactionOptions};
pub use cpu::{CpuThreads, configure_cpu};
pub use decoded::{DEFAULT_DECODED_MAX_BYTES, DecodedCache, DecodedCacheOptions};
pub use document::{Document, Metric, Schema, WalOp};
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
pub use wal::WriteKey;
