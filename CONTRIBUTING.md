# Contributing to itzma

Thanks for taking the time. Bug reports, fixes, docs and tests are all
welcome.

## Before you start

- For a bug, open an issue with a reproduction, or send the fix with a test
  that fails without it.
- For a new feature or a change in behavior, open an issue first so we can
  agree on the shape before you write it.
- Changes to anything on disk (WAL, segment, index or manifest formats,
  magic numbers, format versions) always need an issue first. Every object
  written by one build has to stay readable by the next, so these are
  decided carefully.
- Security problems go through [SECURITY.md](SECURITY.md), not an issue.

## Setting up

You need Rust 1.87 or newer and [uv](https://docs.astral.sh/uv/) for the
Python side. Docker is optional; it runs MinIO for the S3 tests.

```bash
make install-tools   # cargo-deny and cargo-machete
make ci              # everything CI runs on stable, except the docker jobs
```

`make help` lists every target. While working, `make test-fast
TESTS="query filter"` runs the library's unit tests plus the integration
files you name, which is much faster than the full suite. `make test-s3`
runs the tests against MinIO, and `make py` runs the Python bridge's checks.

## Making a change

- New behavior comes with tests. Storage tests loop over every backend in
  `crates/itzma-core/tests/common/mod.rs` rather than using one store, and
  decoders are tested against truncated and corrupted input.
- Every public item in `itzma-core` has a rustdoc comment saying what it
  does and what errors it returns.
- A change to the Python API updates the type stubs (`_itzma.pyi`) and the
  tests in `py-itzma/tests`, in both the sync and async form.
- Add a line under `[Unreleased]` in [CHANGELOG.md](CHANGELOG.md) for
  anything a user would notice.
- Run `make ci` before opening the pull request.

Pull requests are squash-merged, so the PR title becomes the commit message.
Keep it short and describe the change ("fixing filter paging past the tail").

## License

itzma is dual-licensed under MIT and Apache-2.0. Unless you say otherwise,
any contribution you submit for inclusion is licensed the same way, without
any additional terms or conditions.
