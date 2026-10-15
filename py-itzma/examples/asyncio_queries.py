"""AsyncNamespace: the same calls as awaitables, many queries in flight at once."""

import asyncio

import numpy as np

import itzma
from itzma import attr

DIM = 8


async def main() -> None:
    rng = np.random.default_rng(3)
    store = itzma.Store("memory://")
    schema = itzma.Schema(
        dim=DIM, metric=itzma.Metric.COSINE, attrs={"shard": itzma.AttrType.NUMBER}
    )

    async with await itzma.AsyncNamespace.create(store, "async", schema) as ns:
        n = 2_000
        await ns.upsert(
            ids=[str(i) for i in range(n)],
            vectors=rng.standard_normal((n, DIM)),
            attrs=[{"shard": i % 4} for i in range(n)],
        )

        # Each query runs on Itzma's own threads, so the event loop stays free.
        queries = rng.standard_normal((20, DIM))
        results = await asyncio.gather(
            *(ns.query(q, top_k=3, filter=attr("shard") == i % 4) for i, q in enumerate(queries))
        )
        for i, hits in enumerate(results[:5]):
            print(f"query {i} (shard {i % 4}):", hits.ids)


asyncio.run(main())
