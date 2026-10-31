# itzma-core

Vector search on object storage, as an embedded Rust library. Vectors are
stored as plain objects in a bucket; there is no server between your process
and the data. Any process that can reach the bucket can write, and any can
query. Correctness rests on two things the object store already does:
put-if-absent and conditional update.

```toml
[dependencies]
itzma-core = "0.1"
object_store = "0.14"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust
use std::sync::Arc;

use itzma_core::{Document, Metric, Namespace, Schema};
use object_store::memory::InMemory;

#[tokio::main]
async fn main() -> itzma_core::Result<()> {
    let store = Arc::new(InMemory::new());
    let ns = Namespace::create(store, "articles", Schema::new(3, Metric::Cosine)).await?;
    ns.upsert(vec![Document {
        id: "intro".into(),
        vector: vec![0.9, 0.1, 0.0],
        attrs: None,
        sparse: None,
    }])
    .await?;
    Ok(())
}
```

Features: HNSW and SQ8 indexes, sparse vectors, typed attribute filters,
idempotent batched writes, compaction and garbage collection, read caches and
byte budgets. The [full guide](https://github.com/sergiommarcial/itzma#readme)
covers all of it; the Python package is `pip install itzma`.

Licensed under either of MIT or Apache-2.0, at your option.
