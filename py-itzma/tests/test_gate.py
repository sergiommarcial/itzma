"""Store(gate=...): reads through a shared IO gate."""

from __future__ import annotations

import asyncio
import json
import statistics
import threading
import time
from collections.abc import Awaitable

import pytest
from conftest import random_rows, schema

import itzma
from itzma import _itzma

D = json.loads(_itzma.default_options())
Q = [0.1, -0.2, 0.3, 0.05]


def smallest() -> itzma.Gate:
    return itzma.Gate(
        max_bytes=1, max_requests=1, maintenance_max_bytes=1, maintenance_max_requests=1
    )


def test_gate_defaults_match_the_core() -> None:
    gate = itzma.Gate()
    assert gate.max_bytes == D["gate"]["max_bytes"]
    assert gate.max_requests == D["gate"]["max_requests"]
    assert gate.maintenance_max_bytes == D["gate_maintenance"]["max_bytes"]
    assert gate.maintenance_max_requests == D["gate_maintenance"]["max_requests"]


def test_a_namespace_works_through_the_smallest_gate(ns_id: str) -> None:
    plain = itzma.Namespace.create(itzma.Store("memory://"), ns_id, schema())
    plain.upsert(random_rows(200))
    expected = plain.query(Q, 5).ids

    store = itzma.Store("memory://", gate=smallest())
    ns = itzma.Namespace.create(store, ns_id, schema())
    ns.upsert(random_rows(200))
    assert ns.query(Q, 5).ids == expected
    ns.compact()
    assert ns.query(Q, 5).ids == expected
    cached = itzma.Namespace.open(store, ns_id, cache=itzma.Cache())
    assert cached.query(Q, 5).ids == expected
    assert cached.query(Q, 5).ids == expected


@pytest.mark.parametrize(
    "limits",
    [
        {"max_requests": 0},
        {"maintenance_max_bytes": 0},
        {"max_bytes": -1},
        {"maintenance_max_requests": -5},
        {"max_bytes": 2**32},
        {"maintenance_max_bytes": 2**70},
        {"max_requests": 2**64},
    ],
)
def test_a_refused_gate_is_invalid_input(limits: dict[str, int]) -> None:
    with pytest.raises(itzma.InvalidInput):
        itzma.Store("memory://", gate=itzma.Gate(**limits))


def test_threads_share_one_gate(ns_id: str) -> None:
    store = itzma.Store("memory://", gate=smallest())
    ns = itzma.Namespace.create(store, ns_id, schema())
    ns.upsert(random_rows(300))
    ns.compact()
    errors: list[BaseException] = []

    def query() -> None:
        try:
            mine = itzma.Namespace.open(store, ns_id)
            for _ in range(10):
                assert len(mine.query(Q, 3).ids) == 3
        except BaseException as e:  # surfaced below
            errors.append(e)

    threads = [threading.Thread(target=query) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in threads), "a thread is stuck on the gate"
    assert errors == []


def test_cancelled_query_releases_gate_permits(ns_id: str) -> None:
    # Through a one-request gate, 300 WAL objects make a query read one GET at
    # a time, so 16 queries queued on the gate make a 17th about 16 times
    # slower. Cancelled, they must leave the gate, and the 17th runs at about
    # the speed of one. The uncancelled round is the control: without it, a
    # cancel that came too early to touch the gate would pass for one that
    # released its permits. Each query opens its own handle: a handle keeps
    # the tail it folded, so a second query on it reads no WAL
    # object and the queue would drain before anything is measured.
    async def fresh_query(store: itzma.Store) -> itzma.Hits:
        ns = await itzma.AsyncNamespace.open(store, ns_id)
        return await ns.query(Q, 5)

    async def timed(query: Awaitable[itzma.Hits]) -> float:
        start = time.perf_counter()
        await query
        return time.perf_counter() - start

    async def run() -> None:
        store = itzma.Store("memory://", gate=smallest())
        ns = await itzma.AsyncNamespace.create(store, ns_id, schema())
        for i in range(300):
            await ns.upsert(random_rows(5, seed=i, prefix=f"b{i}-"))
        one = statistics.median([await timed(fresh_query(store)) for _ in range(3)])
        ratios = {}
        for cancel in (False, True):
            tasks = [asyncio.create_task(fresh_query(store)) for _ in range(16)]
            await asyncio.sleep(one)  # the queries are on the gate by now
            if cancel:
                for t in tasks:
                    t.cancel()
            ratios[cancel] = await asyncio.wait_for(timed(fresh_query(store)), timeout=60) / one
            await asyncio.gather(*tasks, return_exceptions=True)
        assert ratios[False] > 8, f"the control did not queue: {ratios}"
        assert ratios[True] < 4, f"cancelled queries kept their place: {ratios}"

    asyncio.run(run())
