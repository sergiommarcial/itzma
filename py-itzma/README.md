# itzma for Python

Python bindings for [Itzma](https://github.com/sergiommarcial/itzma#readme): vector retrieval with an object store as the only durable state. The package embeds Itzma in your process. You open a namespace once, keep it warm in memory, and S3 (or a local directory) holds the data.

## Install

```sh
pip install itzma               # or: uv add itzma
pip install "itzma[pyarrow]"    # Arrow input and output through pyarrow
```

Wheels cover macOS (Apple silicon and Intel) and Linux (x86_64 and aarch64, glibc and musl), Python 3.10 and later. Elsewhere pip builds from the source distribution, which needs Rust.

## Install from source

You need Rust (see `rust-version` in the root `Cargo.toml`) and [uv](https://docs.astral.sh/uv/).

```sh
make py-dev      # creates py-itzma/.venv and builds the extension into it
make py-test     # runs the Python tests
```

To use it from another project, build a wheel with `uv run --project py-itzma maturin build --release` and install that.

## Quick start

```python
import numpy as np
import itzma
from itzma import attr

store = itzma.Store("memory://")  # or "file:///var/lib/itzma", "s3://bucket/prefix", "gs://bucket/prefix", "az://container/prefix"
schema = itzma.Schema(
    dim=4,
    metric=itzma.Metric.COSINE,
    attrs={"tag": itzma.AttrType.STRING, "year": itzma.AttrType.NUMBER},
)
ns = itzma.Namespace.create(store, "docs", schema)

ns.upsert(
    [
        {"id": "a", "vector": [1, 0, 0, 0], "attrs": {"tag": "news", "year": 2024}},
        {"id": "b", "vector": [0, 1, 0, 0], "attrs": {"tag": "blog", "year": 2021}},
    ]
)

hits = ns.query([0.9, 0.1, 0, 0], top_k=5, filter=attr("year") >= 2022)
print(hits.ids, hits.distances)
print(hits.to_polars())  # id, distance, then one typed column per declared attribute
```

More runnable scripts are in [examples/](https://github.com/sergiommarcial/itzma/tree/main/py-itzma/examples).

Writes take three shapes:

```python
ns.upsert(rows)  # list of dicts
ns.upsert(ids=ids, vectors=matrix, attrs=attr_dicts)  # numpy (any float dtype)
ns.upsert(df)  # pyarrow, Polars or DuckDB table
```

A big unkeyed write is split into batches. If one fails, the exception's `last_seq` tells you how far it got. Pass `key="..."` to make a write idempotent; a retry passes the same key and `retry_from=` the sequence number it started from.

Filters are built from `attr(name)`, with comparison operators or the matching methods:

```python
(attr("tag") == "news") & (attr("year") >= 2022)
attr("tag").isin(["news", "blog"]) | attr("tag").starts_with("dev")
~(attr("year") < 2000)
attr("year").between(2020, 2024)  # a chained 2020 <= attr("year") <= 2024 raises
attr("tag") == None  # is_null()
```

Wrap each comparison in parentheses: `&` and `|` bind tighter than `==` and `>=`. Leaving them out raises a `TypeError` that says so. A comparison with a missing or null field is false, and `~` flips it, so `~(attr("tag") == "x")` also matches documents with no tag.

`ns.filter(where, limit=100, after=None)` pages through matching documents in id order with no vector. `ns.query_sparse(terms, top_k)` ranks by the dot product of sparse `(term id, weight)` vectors.

## Keeping it compacted and indexed

Writes land in a log that queries also read. Something has to fold it into segments and build indexes. You can call `ns.maintain()`, `ns.compact()`, `ns.index()` and `ns.gc()` yourself, or start a background loop:

```python
m = ns.start_maintenance(timedelta(seconds=60))  # maintain + index every minute
...
m.stop()  # or ns.close()
```

`ns.index()` builds HNSW by default; pass `kind="sq8"` for SQ8, which reads only the partitions a query probes and suits uncached reads. `ns.index_sparse()` builds the sparse inverted index.

Errors are logged to the `itzma.maintenance` logger and kept in `m.last_error`; the loop keeps going. A `file://` store has no conditional update, so compaction and indexing fail there with `StoreError`; use S3 or `memory://` for those.

## In a web server

Open each namespace once per process and share the handle across requests. Every call releases the GIL, and the async methods never block the event loop.

FastAPI:

```python
from contextlib import asynccontextmanager
from datetime import timedelta
from fastapi import FastAPI
import itzma


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = itzma.Store("s3://my-bucket/itzma", region="us-east-1")
    app.state.ns = await itzma.AsyncNamespace.open(
        store, "docs", cache=itzma.Cache(ram_bytes=2 << 30)
    )
    app.state.ns.start_maintenance(timedelta(seconds=60))
    yield
    app.state.ns.close()


app = FastAPI(lifespan=lifespan)


@app.post("/search")
async def search(vector: list[float]):
    hits = await app.state.ns.query(vector, top_k=10)
    return [{"id": h.id, "distance": h.distance} for h in hits]
```

Django or Flask under gunicorn: open the namespace in each worker, not in the master. With `--preload`, open it in a `post_fork` hook or lazily on first use:

```python
# gunicorn.conf.py
def post_fork(server, worker):
    import myapp.search

    myapp.search.ns = itzma.Namespace.open(itzma.Store("s3://my-bucket/itzma"), "docs")
```

A handle opened before a fork refuses to work in the child and says so. Itzma keeps one tokio runtime per process and builds a fresh one in each child.

## Settings

- **S3.** Region and credentials come from the AWS SDK's usual chain: `AWS_*` environment variables, `~/.aws` profiles, web identity, then the container or instance role. The region falls back to `us-east-1`. Keywords override them: `region`, `endpoint`, `access_key_id`, `secret_access_key`, `session_token`, `allow_http` (for a local MinIO). `env=False` ignores the environment and profiles; with no keys it then uses the instance role. Reading a missing object needs `s3:ListBucket`, or S3 answers 403 instead of 404.
- **GCS.** `gs://bucket/prefix` uses Google's default credentials (Workload Identity on GKE and Cloud Run, or `gcloud auth application-default login`). For an emulator, pass `endpoint` and `control_endpoint` (the gRPC one) together, and `anonymous=True` to send no credentials.
- **Azure.** `az://container/prefix` with `account`. `credential` picks how to sign in: `"managed_identity"` (the default), `"workload_identity"`, `"developer_tools"` (`az login`) or `"anonymous"`; or pass `sas_token` instead. `endpoint` overrides `https://<account>.blob.core.windows.net`; token credentials need `https`, so an emulator such as Azurite takes a SAS token.
- Each keyword is refused on a URL whose scheme does not use it.
- **Cache.** `itzma.Cache(ram_bytes=..., nvme_dir=..., nvme_bytes=...)` keeps segment and index bytes in RAM and optionally on a local disk. The manifest is never cached, so a write is visible to the next query either way.
- **Large loads.** An Arrow table or stream (pyarrow, Polars, DuckDB, anything with `__arrow_c_stream__`) is read one record batch at a time and written in chunks of about `target_bytes` (32 MiB by default), with `depth` (2) encoded chunks waiting at most, so a load of any size runs in bounded memory. If it fails part way, the error carries `first_seq`, `last_seq` and `rows_written`; resume an unkeyed load from `rows_written`. With `key=`, chunk `n` is written under `"<key>#<n>"`, so the key can be at most 107 bytes, and a retry with `retry_from=e.first_seq` writes only the missing chunks. Don't give a plain keyed write a key of the form `<load key>#<number>`: chunk keys share that space. A NaN in an Arrow number column raises `InvalidAttrs`. `target_bytes` and `depth` are checked for every input (0 is refused) but only Arrow input uses them.
- **Read budget.** `read_budget=` caps the bytes one query or compaction may read (1 GiB by default). Going over raises `ReadBudgetExceeded`.
- **Threads.** `itzma.configure(worker_threads=n)` sets the runtime's thread count; call it before anything else. `itzma.configure(cpu_threads=n, maintenance_cpu_threads=m)` sizes the two compute pools that queries, decoding and index builds run on (default every core for queries, a quarter of them for maintenance); call it before the first query or build.

## Errors

Everything raises a subclass of `itzma.ItzmaError`: `InvalidInput` (and its subclasses `InvalidVector`, `InvalidAttrs`, `InvalidQuery`, `BatchTooLarge`), `NamespaceExists`, `NamespaceNotFound`, `ReadBudgetExceeded`, `KeyConflict`, `Contention`, `FormatTooNew`, `FormatTooOld`, `Corrupt` and `StoreError`. The useful details are attributes, for example `e.expected` and `e.got` on `InvalidVector`.
