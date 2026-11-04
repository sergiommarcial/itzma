# Changelog

All notable changes to itzma are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). The Rust crate (`itzma-core`) and
the Python package (`itzma`) share one version and are released together.

Add entries under `[Unreleased]` as changes land; the Release workflow turns
that section into the version's notes.

## [Unreleased]

### Added

- `gs://` and `az://` stores in Python, and three Rust crates that wrap the
  vendors' official SDKs as an `ObjectStore`: `itzma-store-gcs`,
  `itzma-store-azure` and `itzma-store-aws`, published to crates.io with
  `itzma-core` at the same version.
- `itzma-store`, a conformance suite that checks an `ObjectStore` does what
  itzma relies on (put-if-absent, conditional update, `NotFound` for a gone
  object, ranges, listings, copies, multipart uploads), for anyone writing a
  store of their own. Every adapter runs it in CI against an emulator.

### Changed

- Python `s3://` stores now go through the AWS SDK. With `env=True` (the
  default) they also read `~/.aws` profiles and web identity, not only `AWS_*`
  variables, and the region falls back to `us-east-1` without asking the
  instance metadata service.
- Building the Python extension from source needs Rust 1.94.1. `itzma-core`
  stays on 1.87.

## [0.1.0] - 2026-11-01

### Added

- First release. Namespaces with a fixed dimension and metric (L2, cosine,
  dot product) on any `object_store` backend: S3 and compatibles, local disk,
  memory.
- Dense vector search, exact or through an index: HNSW (the default) or SQ8
  with clustered partitions.
- Sparse vectors beside the dense one, with an exact sparse query and an
  optional inverted index.
- Typed attributes declared at create, and filtering on them alone or
  narrowing a vector query.
- Batched upserts and deletes, safe with many concurrent writers, with
  optional idempotency keys.
- Compaction, garbage collection, manifest recovery and a read-only health
  report.
- A RAM and NVMe read cache, a cache of decoded structures, and per-read byte
  budgets.
- Python bindings (`pip install itzma`) with NumPy and Arrow input, sync and
  async APIs.

