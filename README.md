# Itzma

Itzma is vector retrieval on top of object storage, written in Rust, with Python bindings. Vectors are stored as plain objects in a bucket. There is no server between the caller and the data: any process that can reach the bucket can write, and any process can query. Correctness rests on two things the object store already does, put-if-absent and conditional update.

## Develop

`make help` lists the targets.

```bash
make install-tools    # once: cargo-deny and cargo-machete
make test             # unit, integration and doctests
make test-fast TESTS="query filter"   # library unit tests plus the named files
make lint             # fmt, clippy, rustdoc, cargo-deny, cargo-machete
make ci               # what CI runs on stable
```

Requires Rust 1.87 or newer (edition 2024).
