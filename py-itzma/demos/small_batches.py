"""Demo for the maintenance lease, bounded staleness and the IO gate (0.3.0).

Reads the `small-batches` synthetic dataset, many small writes, one WAL object
each. Build it first, then run this file:

    make synthetic ONLY=small-batches SIZE=full
    py-itzma/.venv/bin/python py-itzma/demos/small_batches.py [path/to/rows.jsonl]

The tests import the three functions below and run them on the `small` size.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

import itzma

DEFAULT_ROWS = Path(__file__).resolve().parents[2] / "synthetic" / "small-batches" / "rows.jsonl"
PROBE = [0.0, 0.0, 0.0, 0.0]


def load(path: Path = DEFAULT_ROWS) -> list[list[dict[str, Any]]]:
    """The rows grouped into the writes they came from, in order."""
    batches: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        batches[row.pop("batch")].append(row)
    return [batches[number] for number in sorted(batches)]


def schema() -> itzma.Schema:
    return itzma.Schema(dim=4, metric=itzma.Metric.L2, attrs={"tag": itzma.AttrType.STRING})


def wait_until(condition: Any, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError("condition never became true")
        time.sleep(0.02)


def role(ns: itzma.Namespace) -> str:
    assert ns.maintainer is not None, "the handle was opened with maintenance=None"
    return ns.maintainer.role


def lease(batches: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Two handles on one namespace: one maintainer leads, and the other takes over."""
    settings = itzma.Maintenance(
        every=timedelta(seconds=1), renew=timedelta(milliseconds=100), ttl=timedelta(seconds=1)
    )
    store = itzma.Store("memory://")
    first = itzma.Namespace.create(store, "lease-demo", schema(), maintenance=settings)
    second = itzma.Namespace.open(store, "lease-demo", maintenance=settings)
    for batch in batches:
        first.upsert(batch)
    wait_until(lambda: {role(first), role(second)} == {"leader", "follower"})
    leader, follower = (first, second) if role(first) == "leader" else (second, first)
    roles_before = sorted([role(leader), role(follower)])
    leader.close()
    wait_until(lambda: role(follower) == "leader")
    follower.close()
    return {"batches": len(batches), "roles_before": roles_before, "taken_over": True}


def staleness(batches: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """A stale reader misses another writer's batches until it reopens; a writer sees its own."""
    total = sum(len(batch) for batch in batches)
    everything = total + 10  # a top_k past every row, so a query counts them all
    store = itzma.Store("memory://")
    writer = itzma.Namespace.create(store, "stale-demo", schema(), maintenance=None)
    reader = itzma.Namespace.open(
        store, "stale-demo", staleness=timedelta(hours=1), maintenance=None
    )
    assert len(reader.query(PROBE, everything)) == 0  # the reader's view is taken here
    for batch in batches:
        writer.upsert(batch)
    stale_saw = len(reader.query(PROBE, everything))
    fresh = itzma.Namespace.open(store, "stale-demo", maintenance=None)
    fresh_saw = len(fresh.query(PROBE, everything))
    own = itzma.Namespace.open(store, "stale-demo", staleness=timedelta(hours=1), maintenance=None)
    own.upsert([{"id": "own-write", "vector": PROBE}])
    return {
        "written": total,
        "stale_reader_saw": stale_saw,
        "fresh_reader_saw": fresh_saw,
        "writer_saw_own_write": len(own.query(PROBE, everything)) == total + 1,
    }


def gate(batches: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """A one-request, one-byte gate changes no answer, even across threads."""
    tight = itzma.Gate(
        max_bytes=1, max_requests=1, maintenance_max_bytes=1, maintenance_max_requests=1
    )
    answers = []
    for store in (itzma.Store("memory://"), itzma.Store("memory://", gate=tight)):
        ns = itzma.Namespace.create(store, "gate-demo", schema(), maintenance=None)
        for batch in batches:
            ns.upsert(batch)
        ns.compact()
        answers.append((store, ns.query(PROBE, 5).ids))
    gated_store, expected = answers[1]
    mismatches: list[list[str]] = []

    def worker() -> None:
        mine = itzma.Namespace.open(gated_store, "gate-demo", maintenance=None)
        for _ in range(5):
            ids = list(mine.query(PROBE, 5).ids)
            if ids != list(expected):
                mismatches.append(ids)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return {
        "same_as_ungated": list(answers[0][1]) == list(expected),
        "thread_mismatches": len(mismatches),
    }


def main(argv: list[str]) -> int:
    path = Path(argv[0]) if argv else DEFAULT_ROWS
    if not path.exists():
        print(
            f"{path} not found; run: make synthetic ONLY=small-batches SIZE=full", file=sys.stderr
        )
        return 1
    batches = load(path)
    print(f"{len(batches)} batches, {sum(len(b) for b in batches)} rows")
    for name, run in (("lease", lease), ("staleness", staleness), ("gate", gate)):
        print(f"{name}: {run(batches)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
