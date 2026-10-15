from __future__ import annotations

import asyncio
import gc
import logging
import time
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import pytest
from conftest import random_rows, schema

import itzma

FAST = timedelta(milliseconds=50)


def wait_for(condition: Callable[[], bool], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("timed out")
        time.sleep(0.02)


def test_it_compacts_and_indexes(ns_id: str) -> None:
    ns = itzma.Namespace.create(itzma.Store("memory://"), ns_id, schema())
    ns.upsert(random_rows(50))
    m = ns.start_maintenance(FAST, tail_objects=1)

    def compacted() -> bool:
        report = m.last_report
        return (
            report is not None
            and report.index is not None
            and report.compaction is not None
            and report.compaction.kind == "published"
        )

    wait_for(compacted)
    assert m.errors == 0
    m.stop()


def test_it_survives_errors_and_logs_them(
    tmp_path: Path, ns_id: str, caplog: pytest.LogCaptureFixture
) -> None:
    ns = itzma.Namespace.create(itzma.Store(f"file://{tmp_path}"), ns_id, schema())
    ns.upsert(random_rows(5))
    with caplog.at_level(logging.WARNING, logger="itzma.maintenance"):
        m = ns.start_maintenance(FAST, tail_objects=1, index=False)
        wait_for(lambda: m.errors >= 2)
    assert m.running
    assert m.last_error is not None
    assert any(r.name == "itzma.maintenance" for r in caplog.records)
    m.stop()


def test_stop(ns_id: str) -> None:
    ns = itzma.Namespace.create(itzma.Store("memory://"), ns_id, schema())
    m = ns.start_maintenance(FAST)
    wait_for(lambda: m.runs >= 1)
    m.stop()
    wait_for(lambda: not m.running)
    runs = m.runs
    time.sleep(0.2)
    assert m.runs == runs


def test_close_and_with_stop_it(ns_id: str) -> None:
    store = itzma.Store("memory://")
    ns = itzma.Namespace.create(store, ns_id, schema())
    m = ns.start_maintenance(FAST)
    ns.close()
    wait_for(lambda: not m.running)
    with itzma.Namespace.open(store, ns_id) as again:
        m2 = again.start_maintenance(FAST)
    wait_for(lambda: not m2.running)


def test_dropping_the_namespace_ends_it(ns_id: str) -> None:
    ns = itzma.Namespace.create(itzma.Store("memory://"), ns_id, schema())
    m = ns.start_maintenance(FAST)
    ns._maintainers.clear()  # keep only our reference to the maintainer
    del ns
    gc.collect()
    wait_for(lambda: not m.running)


def test_every_must_be_positive(ns_id: str) -> None:
    ns = itzma.Namespace.create(itzma.Store("memory://"), ns_id, schema())
    with pytest.raises(itzma.InvalidInput):
        ns.start_maintenance(timedelta(0))


async def test_async_namespace_can_maintain(ns_id: str) -> None:
    ns = await itzma.AsyncNamespace.create(itzma.Store("memory://"), ns_id, schema())
    await ns.upsert(random_rows(5))
    m = ns.start_maintenance(FAST, tail_objects=1)
    deadline = time.monotonic() + 10
    while m.last_report is None or m.last_report.compaction is None:
        assert time.monotonic() < deadline
        await asyncio.sleep(0.02)
    ns.close()
    assert m.last_report.compaction.kind == "published"
