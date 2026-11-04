//! An [`object_store::ObjectStore`] over Amazon's official S3 SDK.
//!
//! [`S3Store`] wraps one `aws_sdk_s3::Client` the caller builds. Object
//! versions are ETags, so put-if-absent is `If-None-Match: *` and the
//! manifest's conditional update is `If-Match`. [`S3Store::connect`] builds a
//! client from [`S3Options`] for callers with only settings, as the Python
//! bridge is. It passes the `itzma-store` conformance suite.
#![warn(missing_docs)]
#![cfg_attr(not(test), deny(clippy::unwrap_used))]

mod errors;
mod retry;
mod store;
mod upload;

pub use store::{S3Options, S3Store};
