from __future__ import annotations

import os
import random
import signal
import sys
import threading
import time
from pathlib import Path

import pytest
from conftest import random_rows, schema

import itzma

Q = [0.1, -0.2, 0.3, 0.05]


@pytest.mark.skipif(sys.platform == "win32", reason="no fork on Windows")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_a_forked_child_must_open_again(tmp_path: Path, ns_id: str) -> None:
    store = itzma.Store(f"file://{tmp_path}")
    ns = itzma.Namespace.create(store, ns_id, schema())
    ns.upsert(random_rows(20))
    expected = ns.query(Q, 5).ids  # the parent's runtime now exists

    pid = os.fork()
    if pid == 0:  # child
        code = 1
        try:
            try:
                ns.query(Q, 5)
            except itzma.ItzmaError as e:
                if "fork" in str(e):
                    fresh = itzma.Namespace.open(itzma.Store(f"file://{tmp_path}"), ns_id)
                    if fresh.query(Q, 5).ids == expected:
                        code = 0
        finally:
            os._exit(code)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert ns.query(Q, 5).ids == expected  # the parent is unaffected


@pytest.mark.skipif(sys.platform == "win32", reason="no fork on Windows")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_a_store_from_before_the_fork_is_refused(tmp_path: Path, ns_id: str) -> None:
    store = itzma.Store(f"file://{tmp_path}")
    itzma.Namespace.create(store, ns_id, schema()).upsert(random_rows(5))

    pid = os.fork()
    if pid == 0:  # child: the parent's Store holds the parent's connections
        code = 1
        try:
            try:
                itzma.Namespace.open(store, ns_id)
            except itzma.ItzmaError as e:
                if "fork" in str(e):
                    code = 0
        finally:
            os._exit(code)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0


def _big_namespace(ns_id: str) -> itzma.Namespace:
    """20,000 rows at dimension 64: big enough to reach the compute pools."""
    rng = random.Random(1)
    rows = [{"id": f"d{i:05d}", "vector": [rng.random() for _ in range(64)]} for i in range(20_000)]
    ns = itzma.Namespace.create(
        itzma.Store("memory://"), ns_id, itzma.Schema(dim=64, metric=itzma.Metric.L2)
    )
    ns.upsert(rows)
    return ns


@pytest.mark.skipif(sys.platform == "win32", reason="no fork on Windows")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_a_forked_child_queries_and_builds_after_the_parent_used_the_pools(ns_id: str) -> None:
    parent = _big_namespace(ns_id)
    expected = parent.query([0.5] * 64, 5, exact=True).ids  # the parent's pools exist now

    pid = os.fork()
    if pid == 0:  # child
        code = 1
        try:
            signal.alarm(120)  # a pool left from the parent would hang: die instead
            ns = _big_namespace(ns_id)
            if ns.query([0.5] * 64, 5, exact=True).ids == expected:
                ns.compact()
                ns.index(min_rows=0)  # the maintenance lane
                if len(ns.query([0.5] * 64, 5).ids) == 5:
                    code = 0
        finally:
            os._exit(code)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert parent.query([0.5] * 64, 5, exact=True).ids == expected


@pytest.mark.skipif(sys.platform == "win32", reason="no fork on Windows")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_a_fork_while_a_scan_runs_on_the_pools_leaves_the_child_working(ns_id: str) -> None:
    parent = _big_namespace(ns_id)
    stop = threading.Event()
    ran = []

    def scan() -> None:
        while not stop.is_set():
            ran.append(len(parent.query([0.5] * 64, 5, exact=True).ids))

    worker = threading.Thread(target=scan)
    worker.start()
    try:
        while not ran:  # at least one scan has been on the pools
            time.sleep(0.01)
        time.sleep(0.05)  # and another is likely on them now
        pid = os.fork()
        if pid == 0:  # child
            code = 1
            try:
                signal.alarm(120)  # a hang dies instead of stalling the suite
                ns = _big_namespace(ns_id)
                if len(ns.query([0.5] * 64, 5, exact=True).ids) == 5:
                    code = 0
            finally:
                os._exit(code)
        _, status = os.waitpid(pid, 0)
    finally:
        stop.set()
        worker.join()
    assert os.waitstatus_to_exitcode(status) == 0
    assert set(ran) == {5}
