# Changelog

All notable changes to itzma are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). The Rust crate (`itzma-core`) and
the Python package (`itzma`) share one version and are released together.

Add entries under `[Unreleased]` as changes land; the Release workflow turns
that section into the version's notes.

## [Unreleased]

### Fixed

- An indexed sparse query summed a row's terms in a `HashMap`'s order, which
  changes from run to run, so float rounding could move a score by a bit
  between two identical queries and differ from the flat path. Terms are now
  summed in ascending term id.
- A write key reused for a different batch of the same size was taken for a
  retry, and the second batch was silently dropped. The bodies are now
  compared when the lengths agree, and a different batch is `KeyConflict`.
- A merge sized its groups by `data.bin` alone, so segments with heavy sparse
  vectors, or many read in windows, could make a group whose read charge went
  past the read budget that `CompactionOptions::validate` had accepted. Groups
  are now sized by what each segment charges (both objects and the ranged
  open's guesses) and split to fit.

## [0.3.0] - 2026-11-27

### Security

- The lockfile now pins `url` 2.5.9, which fixes a stack overflow in
  `Url::origin()` on deeply nested `blob:` URLs (RUSTSEC-2026-0592). `url`
  comes in through `object_store`.

### Added

- `Namespace::maintenance_lease` and `MaintenanceLease`, so the processes
  sharing a namespace can agree on one maintainer. The lease is one object,
  `<namespace>/maintenance/lease.json`, renewed every 15 seconds by default
  and taken over by another process after 60 seconds of silence. It saves
  work and decides nothing about correctness: two maintainers at once were
  always safe. `examples/maintenance_loop.rs` shows a Rust loop on it.
- `itzma.Maintenance`, the settings for the Python loop, and
  `ns.maintainer`, with a `role` of `"leader"` or `"follower"`.
- `Namespace::with_staleness(max_age)`, and `staleness=` on `open` and
  `create` in Python: queries on the handle reuse its last view of the
  manifest and the WAL tail for up to `max_age`, so a warm query sends no
  request for either, and queries waiting on a refresh share one. The
  handle's own writes are always visible. Off by default; queries stay
  strict unless a handle opts in.
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

### Fixed

- The Python maintenance loop now counts a panicked run as a failed one and
  ignores a run that was aborted by stopping the loop. A leased loop also
  clears its leader role when it ends, so `ns.maintainer.role` no longer
  reports a stopped loop as the leader.

## [0.2.1] - 2026-11-15

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

