# Changelog

All notable changes to itzma are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). The Rust crate (`itzma-core`) and
the Python package (`itzma`) share one version and are released together.

Add entries under `[Unreleased]` as changes land; the Release workflow turns
that section into the version's notes.

## [Unreleased]

### Added

- `Namespace::maintenance_lease` and `MaintenanceLease`, so the processes
  sharing a namespace can agree on one maintainer. The lease is one object,
  `<namespace>/maintenance/lease.json`, renewed every 15 seconds by default
  and taken over by another process after 60 seconds of silence. It saves
  work and decides nothing about correctness: two maintainers at once were
  always safe. `examples/maintenance_loop.rs` shows a Rust loop on it.
- `itzma.Maintenance`, the settings for the Python loop, and
  `ns.maintainer`, with a `role` of `"leader"` or `"follower"`.
- `GatedStore` and `GateOptions`, an opt-in store wrapper that admits reads
  only while the requests and bytes in flight stay under its limits, with a
  smaller `GateOptions::maintenance()` for compaction and index builds.
  Writes, deletes and listings are not gated. In Python,
  `Store(url, gate=itzma.Gate())` builds a query gate and a maintenance gate
  shared by every namespace opened from that store. The default limits are
  starting points, not measurements. `make test-gated` runs the read-path
  tests under a one-request, one-byte gate, and CI runs it on stable.

### Changed

- `Namespace.open` and `create` in Python (sync and async) now start a leased
  maintenance loop by default, which compacts, indexes and collects every
  minute while it holds the lease. The first run comes a minute after it
  takes the lease, so a short script never pays for one. Pass
  `maintenance=None` to turn it off, for instance when `maintain` already
  runs from cron. On a `file://` store the loop logs one warning and stops.
  `start_maintenance` still takes no lease and replaces the automatic loop.

### Security

- The lockfile now pins `serde_with` 3.21.0, which the Google Cloud SDK pulls
  in for `itzma-store-gcs` and the Python wheels. Older versions panic when
  serializing an empty entry through `KeyValueMap` (GHSA-7gcf-g7xr-8hxj).

## [0.2.0] - 2026-11-04

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

