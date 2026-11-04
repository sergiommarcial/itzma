//! What every object store Itzma runs on has to get right.
//!
//! `itzma-core` takes any `Arc<dyn ObjectStore>`, but it leans on a few
//! behaviors the trait does not spell out: put-if-absent picks WAL sequence
//! numbers, a versioned update is the manifest's compare-and-swap, and a
//! missing object must read as `NotFound` so a reader that lost a race with
//! garbage collection reloads and retries.
//! The `conformance` feature holds one check per behavior and the
//! [`conformance_tests!`] macro that runs them against a store.
#![warn(missing_docs)]
#![cfg_attr(not(test), deny(clippy::unwrap_used))]

#[cfg(feature = "conformance")]
pub mod conformance;
