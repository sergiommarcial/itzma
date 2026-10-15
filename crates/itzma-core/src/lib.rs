//! Itzma core: object-storage-backed vector retrieval.
//!
//! Vectors live as plain objects in a bucket. There is no server in the
//! middle: any process with access to the bucket can write, and any process
//! can query. Correctness comes from two primitives the object store already
//! provides, put-if-absent and conditional update.
#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![warn(missing_docs)]
