"""Load many documents from numpy, pyarrow or Polars, and make a write safe to retry."""

import numpy as np
import polars as pl
import pyarrow as pa

import itzma

DIM = 16
rng = np.random.default_rng(7)

store = itzma.Store("memory://")
schema = itzma.Schema(dim=DIM, metric=itzma.Metric.COSINE, attrs={"shard": itzma.AttrType.NUMBER})
ns = itzma.Namespace.create(store, "bulk", schema)

# numpy: ids, a 2-D matrix of any float dtype, and one attrs dict per row.
n = 10_000
seq = ns.upsert(
    ids=[f"np-{i}" for i in range(n)],
    vectors=rng.standard_normal((n, DIM)),
    attrs=[{"shard": i % 4} for i in range(n)],
)
print("numpy load ended at seq", seq)

# Arrow: an `id` column, a fixed-size list `vector` column, and one column per attribute.
vectors = rng.standard_normal((1_000, DIM)).astype(np.float32)
table = pa.table(
    {
        "id": [f"pa-{i}" for i in range(1_000)],
        "vector": pa.FixedSizeListArray.from_arrays(pa.array(vectors.ravel()), DIM),
        "shard": [i % 4 for i in range(1_000)],
    }
)
ns.upsert(table)

# Polars frames (and DuckDB results) go in the same way.
ns.upsert(pl.from_arrow(table).with_columns(pl.col("id").str.replace("pa-", "pl-")))

# A keyed write lands once however often it is retried. Keep the key and the
# seq you started from; a retry passes both.
start = seq
try:
    ns.upsert([{"id": "important", "vector": [1.0] * DIM}], key="load-2026-09-29")
except itzma.ItzmaError:
    ns.upsert(
        [{"id": "important", "vector": [1.0] * DIM}],
        key="load-2026-09-29",
        retry_from=start,
    )

print(ns.query([1.0] * DIM, top_k=3).ids)
