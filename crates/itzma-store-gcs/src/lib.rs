//! An [`object_store::ObjectStore`] over Google's official Cloud Storage SDK.
//!
//! [`GcsStore`] wraps two clients the caller builds: `Storage` for reads and
//! writes (JSON over HTTP) and `StorageControl` for metadata, listings,
//! deletes and copies (gRPC `google.storage.v2`). Object versions are GCS
//! generations, so put-if-absent and the manifest's conditional update map to
//! `if_generation_match`. [`GcsStore::connect`] builds both clients from a few
//! settings, for callers that only have a URL (the Python bridge, the tests).
//! It passes the `itzma-store` conformance suite.
#![warn(missing_docs)]
#![cfg_attr(not(test), deny(clippy::unwrap_used))]

mod errors;
mod store;
mod upload;

pub use store::{GcsOptions, GcsStore};
