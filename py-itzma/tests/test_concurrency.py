from __future__ import annotations

import asyncio
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from conftest import Model, assert_hits, random_rows, schema

import itzma

Q = [0.1, -0.2, 0.3, 0.05]


def big_namespace(ns_id: str, n: int = 50_000, dim: int = 64) -> itzma.Namespace:
    ns = itzma.Namespace.create(
        itzma.Store("memory://"), ns_id, itzma.Schema(dim=dim, metric=itzma.Metric.L2)
    )
    rng = np.random.default_rng(9)
    ns.upsert(ids=[f"v{i}" for i in range(n)], vectors=rng.standard_normal((n, dim)))
    return ns


def test_threads_share_one_handle(ns_id: str) -> None:
    ns = itzma.Namespace.create(itzma.Store("memory://"), ns_id, schema())
    model = Model()
    first = random_rows(100)
    ns.upsert(first)
    model.upsert(first)
    errors: list[BaseException] = []

    def reader() -> None:
        try:
            for _ in range(50):
                assert len(ns.query(Q, 5)) == 5
        except BaseException as e:
            errors.append(e)

    def writer() -> None:
        for i in range(50):
            rows = random_rows(5, seed=100 + i, prefix=f"w{i}_")
            ns.upsert(rows)
            model.upsert(rows)

    with ThreadPoolExecutor(max_workers=17) as pool:
        futures = [pool.submit(reader) for _ in range(16)] + [pool.submit(writer)]
        for f in futures:
            f.result()
    assert errors == []
    assert_hits(ns.query(Q, 20), model.top_k(Q, 20))


def test_a_sync_call_releases_the_gil(ns_id: str) -> None:
    ns = big_namespace(ns_id)
    counter = 0
    done = threading.Event()

    def count() -> None:
        nonlocal counter
        while not done.is_set():
            counter += 1

    t = threading.Thread(target=count)
    t.start()
    before = counter
    for _ in range(5):
        ns.query(np.zeros(64, dtype=np.float32), 10, exact=True)
    during = counter - before
    done.set()
    t.join()
    assert during > 1000


async def test_gathered_queries_agree(ns_id: str) -> None:
    store = itzma.Store("memory://")
    sync = itzma.Namespace.create(store, ns_id, schema())
    sync.upsert(random_rows(200))
    expected = sync.query(Q, 10).ids
    ans = await itzma.AsyncNamespace.open(store, ns_id)
    results = await asyncio.gather(*(ans.query(Q, 10) for _ in range(100)))
    assert all(r.ids == expected for r in results)


async def test_the_loop_is_not_blocked(ns_id: str) -> None:
    ns = big_namespace(ns_id)
    ans = itzma.AsyncNamespace(ns._ns)  # the same Rust handle, async face
    gaps: list[float] = []
    stop = asyncio.Event()

    async def heartbeat() -> None:
        last = time.perf_counter()
        while not stop.is_set():
            await asyncio.sleep(0.001)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    beat = asyncio.create_task(heartbeat())
    start = time.perf_counter()
    await asyncio.gather(
        *(ans.query(np.zeros(64, dtype=np.float32), 10, exact=True) for _ in range(20))
    )
    total = time.perf_counter() - start
    stop.set()
    await beat
    assert total > 0.05, "the calls were too fast to show anything; grow the namespace"
    assert max(gaps) < 0.05


async def test_cancelling_a_task_leaves_the_namespace_usable(ns_id: str) -> None:
    ans = await itzma.AsyncNamespace.create(itzma.Store("memory://"), ns_id, schema())
    await ans.upsert(random_rows(100))
    expected = (await ans.query(Q, 5)).ids
    task = asyncio.create_task(ans.query(Q, 5))
    await asyncio.sleep(0)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert (await ans.query(Q, 5)).ids == expected


async def test_cancelling_a_query_during_a_flat_scan_leaves_the_pools_working(ns_id: str) -> None:
    rng = random.Random(2)
    rows = [{"id": f"d{i:05d}", "vector": [rng.random() for _ in range(64)]} for i in range(20_000)]
    ans = await itzma.AsyncNamespace.create(
        itzma.Store("memory://"), ns_id, itzma.Schema(dim=64, metric=itzma.Metric.L2)
    )
    await ans.upsert(rows)
    # Time one whole query, then cancel each run halfway through one, so the
    # cancel lands while the scan is on the pool, not before it starts.
    start = time.perf_counter()
    await ans.query([0.5] * 64, 5, exact=True)
    whole = time.perf_counter() - start
    for _ in range(5):
        task = asyncio.create_task(ans.query([0.5] * 64, 5, exact=True))
        await asyncio.sleep(whole / 2)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    assert len((await ans.query([0.5] * 64, 5, exact=True)).ids) == 5
