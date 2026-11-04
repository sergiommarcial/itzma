# itzma-store-aws

An [`object_store::ObjectStore`](https://docs.rs/object_store) over Amazon's official S3 SDK, for [itzma](https://github.com/sergiommarcial/itzma) or anything else that takes an `ObjectStore`.

```rust,ignore
use itzma_store_aws::{S3Options, S3Store};

// The AWS SDK's default chain: environment, ~/.aws profiles, web identity, roles.
let store = S3Store::connect("my-bucket", S3Options::default()).await?;
```

Or build an `aws_sdk_s3::Client` yourself and pass it to `S3Store::new(client, bucket)`. Object versions are ETags, so put-if-absent is `If-None-Match: *` and a conditional update is `If-Match`. Setting an `endpoint` (MinIO and other S3-compatible stores) switches to path-style addressing; an `http` endpoint needs `allow_http`.

Needs Rust 1.94.1. Passes the [`itzma-store`](https://crates.io/crates/itzma-store) conformance suite.

Licensed under either of Apache-2.0 or MIT, at your option.
