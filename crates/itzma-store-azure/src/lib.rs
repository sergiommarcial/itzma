//! An [`object_store::ObjectStore`] over Microsoft's official Azure Blob
//! Storage SDK.
//!
//! [`AzureStore`] wraps one `BlobContainerClient` the caller builds. Object
//! versions are ETags, so put-if-absent is `If-None-Match: *` and the
//! manifest's conditional update is `If-Match`. The SDK authenticates with
//! Entra ID tokens over HTTPS or with a SAS token in the URL;
//! [`AzureStore::connect`] picks one from [`AzureCredential`]. It passes the
//! `itzma-store` conformance suite.
#![warn(missing_docs)]
#![cfg_attr(not(test), deny(clippy::unwrap_used))]

mod errors;
mod store;
mod upload;

pub use store::{AzureCredential, AzureOptions, AzureStore};
