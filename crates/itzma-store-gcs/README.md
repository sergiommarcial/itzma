# itzma-store-gcs

An [`object_store::ObjectStore`](https://docs.rs/object_store) over Google's official Cloud Storage SDK, for [itzma](https://github.com/sergiommarcial/itzma) or anything else that takes an `ObjectStore`.

```rust,ignore
use itzma_store_gcs::{GcsOptions, GcsStore};

// Google's default credentials (Workload Identity on GKE and Cloud Run).
let store = GcsStore::connect("my-bucket", GcsOptions::default()).await?;
```

Or build the SDK's `Storage` and `StorageControl` clients yourself, with whatever credentials, retries and endpoints you need, and pass them to `GcsStore::new(storage, control, bucket)`. Object versions are GCS generations, so put-if-absent and conditional updates map to `if_generation_match`. For an emulator, set `endpoint` and `control_endpoint` together (one without the other is refused).

Needs Rust 1.91. Passes the [`itzma-store`](https://crates.io/crates/itzma-store) conformance suite.

Licensed under either of Apache-2.0 or MIT, at your option.
