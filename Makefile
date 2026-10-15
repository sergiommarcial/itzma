.DEFAULT_GOAL := help

# Version floor this repo assumes, read from the workspace's rust-version so
# Cargo.toml stays the single source of truth.
RUST_MIN := $(shell sed -n 's/^rust-version = "\(.*\)"/\1/p' Cargo.toml)

# Where compose.yaml publishes its MinIO container.
S3_ENDPOINT := http://127.0.0.1:9000


.PHONY: help check-tools check-cargo check-cargo-deny check-cargo-machete check-docker \
	install install-tools \
	build build-release \
	test test-fast test-msrv test-s3 s3-up s3-down bench \
	lint fmt-check clippy doc deny machete \
	format doc-open \
	ci clean

##@ General

help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nitzma — available targets\n\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*##/ { printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)
	@echo ""

##@ Environment checks

check-tools: check-cargo check-cargo-deny check-cargo-machete ## Verify cargo (with rustfmt and clippy), cargo-deny and cargo-machete are installed

check-cargo: ## Verify cargo is installed and rustc meets the rust-version in Cargo.toml
	@command -v cargo >/dev/null 2>&1 || { echo "cargo not found — install: https://rustup.rs"; exit 1; }
	@[ -n "$(RUST_MIN)" ] || { echo "could not read rust-version from Cargo.toml"; exit 1; }
	@printf '%s\n%s\n' "$(RUST_MIN)" "$$(rustc --version | cut -d' ' -f2)" | sort -V -C \
		|| { echo "$$(rustc --version) found, need >=$(RUST_MIN) (rust-version in Cargo.toml)"; exit 1; }
	@cargo fmt --version >/dev/null 2>&1 || { echo "rustfmt not found — run: rustup component add rustfmt"; exit 1; }
	@cargo clippy --version >/dev/null 2>&1 || { echo "clippy not found — run: rustup component add clippy"; exit 1; }
	@echo "cargo: OK ($$(rustc --version))"

check-cargo-deny: ## Verify cargo-deny is installed
	@cargo deny --version >/dev/null 2>&1 || { echo "cargo-deny not found — run: make install-tools"; exit 1; }
	@echo "cargo-deny: OK ($$(cargo deny --version))"

check-cargo-machete: ## Verify cargo-machete is installed
	@cargo machete --version >/dev/null 2>&1 || { echo "cargo-machete not found — run: make install-tools"; exit 1; }
	@echo "cargo-machete: OK ($$(cargo machete --version))"

check-docker: ## Verify docker and docker compose are installed and the daemon is running
	@command -v docker >/dev/null 2>&1 || { echo "docker not found — install: https://docs.docker.com/get-docker/"; exit 1; }
	@docker compose version >/dev/null 2>&1 || { echo "docker compose not found"; exit 1; }
	@docker info >/dev/null 2>&1 || { echo "docker daemon is not running"; exit 1; }
	@echo "docker: OK ($$(docker --version))"

##@ Setup

install: check-cargo ## Fetch the locked dependencies (after verifying the toolchain)
	cargo fetch --locked

install-tools: check-cargo ## Install the dev tools that lint needs: cargo-deny and cargo-machete
	cargo install cargo-deny cargo-machete --locked

##@ Build

build: check-cargo ## Build the workspace (debug)
	cargo build --locked

build-release: check-cargo ## Build the workspace (release)
	cargo build --release --locked

##@ Testing

test: check-cargo ## Run unit, integration, and doctests, and each benchmark once
	cargo test --locked
	cargo test --locked --bench core

test-msrv: ## Run the tests on the rust-version floor (needs rustup; the second leg of CI's test job)
	@command -v rustup >/dev/null 2>&1 || { echo "rustup not found — install: https://rustup.rs"; exit 1; }
	rustup toolchain install $(RUST_MIN) --profile minimal
	rustup run $(RUST_MIN) cargo test --locked

test-fast: check-cargo ## Per-task check: the library's unit tests plus the named integration test files only, no doctests or benches. Example: make test-fast TESTS="query filter"
	cargo test --locked -p itzma-core --lib $(foreach t,$(TESTS),--test $(t))

test-s3: check-cargo s3-up ## Run the tests against a real S3-compatible store (MinIO via docker compose)
	ITZMA_TEST_S3_ENDPOINT=$(S3_ENDPOINT) cargo test --locked

bench: check-cargo ## Run the benchmarks (crates/itzma-core/benches, in-memory store, about 90 s)
	cargo bench --locked --bench core

##@ S3 test server

s3-up: check-docker ## Start MinIO and create the test bucket (compose.yaml)
	docker compose up -d --wait

s3-down: check-docker ## Stop MinIO and delete its data
	docker compose down -v

##@ Linting & formatting

lint: fmt-check clippy doc deny machete ## Run all static analysis (no files written)

fmt-check: check-cargo ## Check formatting with rustfmt
	cargo fmt --check

clippy: check-cargo ## Lint with clippy, warnings are errors
	cargo clippy --locked --all-targets -- -D warnings

doc: check-cargo ## Build rustdoc; broken links and missing docs are errors
	RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps -p itzma-core

deny: check-cargo-deny ## Check advisories, licenses, bans, and sources (policy in deny.toml)
	cargo deny --locked check --hide-inclusion-graph

machete: check-cargo-machete ## Find unused dependencies
	cargo machete

format: check-cargo ## Reformat the code with rustfmt, writing changes
	cargo fmt

doc-open: check-cargo ## Build rustdoc and open it in the browser
	cargo doc --locked --no-deps -p itzma-core --open

##@ CI

ci: lint test ## Run what .github/workflows/ci.yml runs on stable, without docker (test-s3 and test-msrv cover the other two test jobs)

##@ Cleanup

clean: check-cargo ## Remove build artifacts (target/)
	cargo clean
