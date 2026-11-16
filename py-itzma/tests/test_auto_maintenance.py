"""Maintenance that starts on its own when a handle opens."""

from __future__ import annotations

import logging
import time
from datetime import timedelta
from pathlib import Path

import pytest
from conftest import random_rows, schema, wait_for

import itzma

FAST = itzma.Maintenance(
    every=timedelta(milliseconds=200),
    renew=timedelta(milliseconds=50),
    ttl=timedelta(milliseconds=200),
    tail_objects=1,
)


def compacted(m: itzma.Maintainer) -> bool:
    report = m.last_report
    return (
        report is not None
        and report.compaction is not None
        and report.compaction.kind == "published"
    )


def test_open_starts_a_loop(ns_id: str) -> None:
    store = itzma.Store("memory://")
    itzma.Namespace.create(store, ns_id, schema(), maintenance=None).upsert(random_rows(20))
    ns = itzma.Namespace.open(store, ns_id, maintenance=FAST)
    m = ns.maintainer
    assert m is not None
    assert m.running
    wait_for(lambda: m.role == "leader")
    wait_for(lambda: compacted(m))
    ns.close()


def test_none_turns_it_off(ns_id: str) -> None:
    store = itzma.Store("memory://")
    ns = itzma.Namespace.create(store, ns_id, schema(), maintenance=None)
    assert ns.maintainer is None
    assert itzma.Namespace.open(store, ns_id, maintenance=None).maintainer is None


def test_create_starts_a_loop_too(ns_id: str) -> None:
    ns = itzma.Namespace.create(itzma.Store("memory://"), ns_id, schema(), maintenance=FAST)
    assert ns.maintainer is not None
    assert ns.maintainer.running
    ns.close()


async def test_async_open_starts_a_loop(ns_id: str) -> None:
    store = itzma.Store("memory://")
    created = await itzma.AsyncNamespace.create(store, ns_id, schema(), maintenance=None)
    await created.upsert(random_rows(20))
    ns = await itzma.AsyncNamespace.open(store, ns_id, maintenance=FAST)
    m = ns.maintainer
    assert m is not None
    wait_for(lambda: compacted(m))
    ns.close()


async def test_async_create_starts_a_loop(ns_id: str) -> None:
    ns = await itzma.AsyncNamespace.create(
        itzma.Store("memory://"), ns_id, schema(), maintenance=FAST
    )
    assert ns.maintainer is not None
    assert ns.maintainer.running
    ns.close()


def two_handles(ns_id: str) -> tuple[itzma.Namespace, itzma.Namespace]:
    store = itzma.Store("memory://")
    a = itzma.Namespace.create(store, ns_id, schema(), maintenance=FAST)
    b = itzma.Namespace.open(store, ns_id, maintenance=FAST)
    return a, b


def roles(*handles: itzma.Namespace) -> list[str]:
    return sorted(h.maintainer.role for h in handles if h.maintainer is not None)


def test_two_handles_in_one_process_have_one_leader(ns_id: str) -> None:
    a, b = two_handles(ns_id)
    wait_for(lambda: roles(a, b) == ["follower", "leader"])
    time.sleep(0.3)
    assert roles(a, b) == ["follower", "leader"]
    a.close()
    b.close()


def test_closing_the_leader_hands_over(ns_id: str) -> None:
    a, b = two_handles(ns_id)
    wait_for(lambda: roles(a, b) == ["follower", "leader"])
    assert a.maintainer is not None and b.maintainer is not None
    leader, other = (a, b) if a.maintainer.role == "leader" else (b, a)
    still = other.maintainer
    assert still is not None
    leader.close()
    wait_for(lambda: still.role == "leader", timeout=1.0)
    other.close()


def test_start_maintenance_replaces_the_automatic_loop(ns_id: str) -> None:
    ns = itzma.Namespace.create(itzma.Store("memory://"), ns_id, schema(), maintenance=FAST)
    auto = ns.maintainer
    assert auto is not None
    m = ns.start_maintenance(timedelta(seconds=60))
    wait_for(lambda: not auto.running)
    assert ns.maintainer is None
    assert m.running
    ns.close()


def test_a_short_script_never_compacts(ns_id: str) -> None:
    store = itzma.Store("memory://")
    with itzma.Namespace.create(store, ns_id, schema()) as ns:
        assert ns.maintainer is not None
        for i in range(100):
            ns.upsert(random_rows(1, prefix=f"r{i}-"))
    later = itzma.Namespace.open(store, ns_id, maintenance=None)
    report = later.maintain(tail_objects=1 << 40, tail_bytes=1 << 62, gc=False)
    assert report.tail_objects == 100
    assert report.compaction is None


def test_file_store_ends_the_loop_quietly(
    tmp_path: Path, ns_id: str, caplog: pytest.LogCaptureFixture
) -> None:
    store = itzma.Store(f"file://{tmp_path}")
    with caplog.at_level(logging.WARNING, logger="itzma.maintenance"):
        ns = itzma.Namespace.create(store, ns_id, schema(), maintenance=FAST)
        ns.upsert(random_rows(5))
        m = ns.maintainer
        assert m is not None
        wait_for(lambda: not m.running)
        time.sleep(0.2)
    assert m.errors == 1
    assert m.last_error is not None
    assert len([r for r in caplog.records if r.name == "itzma.maintenance"]) == 1
    assert len(ns.query([0.1, 0.2, 0.3, 0.4], 3).ids) == 3
