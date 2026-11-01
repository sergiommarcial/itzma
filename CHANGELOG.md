# Changelog

All notable changes to itzma are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). The Rust crate (`itzma-core`) and
the Python package (`itzma`) share one version and are released together.

Add entries under `[Unreleased]` as changes land; the Release workflow turns
that section into the version's notes.

## [Unreleased]

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

