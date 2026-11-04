# itzma-store

Checks that an [`object_store::ObjectStore`](https://docs.rs/object_store) does what [itzma](https://github.com/sergiommarcial/itzma) relies on: put-if-absent picks WAL sequence numbers, a versioned update is the manifest's compare-and-swap, and a missing object reads as `NotFound`. It also checks ranged and suffix reads, listings with offsets and delimiters, copies and multipart uploads.

Turn on the `conformance` feature in a dev-dependency and expand the macro once per store:

```toml
[dev-dependencies]
itzma-store = { version = "0.1", features = ["conformance"] }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust,ignore
use std::sync::Arc;
use object_store::{ObjectStore, memory::InMemory};

async fn memory() -> Option<Arc<dyn ObjectStore>> {
    Some(Arc::new(InMemory::new()))
}

itzma_store::conformance_tests!(memory, memory);
```

Each check gets a fresh store from the factory; return `None` to skip (for a backend whose endpoint isn't configured).

Licensed under either of Apache-2.0 or MIT, at your option.
