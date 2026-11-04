# itzma-store-azure

An [`object_store::ObjectStore`](https://docs.rs/object_store) over Microsoft's official Azure Blob Storage SDK, for [itzma](https://github.com/sergiommarcial/itzma) or anything else that takes an `ObjectStore`.

```rust,ignore
use itzma_store_azure::{AzureOptions, AzureStore};

// A managed identity by default; see AzureCredential for the others.
let store = AzureStore::connect("myaccount", "my-container", AzureOptions::default()).await?;
```

Or build a `BlobContainerClient` yourself and pass it to `AzureStore::new`. Object versions are ETags, so put-if-absent is `If-None-Match: *` and a conditional update is `If-Match`. Token credentials need HTTPS; reach an emulator such as Azurite with a SAS token.

Needs Rust 1.88. Passes the [`itzma-store`](https://crates.io/crates/itzma-store) conformance suite.

Licensed under either of Apache-2.0 or MIT, at your option.
