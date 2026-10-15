"""Compaction, indexing and garbage collection, by hand and in the background."""

import time
from datetime import timedelta

import numpy as np

import itzma

DIM = 8
rng = np.random.default_rng(1)

store = itzma.Store("memory://")  # file:// cannot compact or index; use S3 or memory
ns = itzma.Namespace.create(store, "maint", itzma.Schema(dim=DIM, metric=itzma.Metric.L2, attrs={}))


def load(prefix: str, n: int) -> None:
    ns.upsert(ids=[f"{prefix}-{i}" for i in range(n)], vectors=rng.standard_normal((n, DIM)))


# By hand. Writes land in a log; compact folds it into a segment.
load("a", 5_000)
print(ns.compact())

# An index is built per segment, for segments of at least min_rows rows.
print(ns.index())

# A merge leaves the old segments behind; gc deletes them once they are
# min_age old. Zero here only because nothing else reads this namespace.
load("b", 5_000)
print(ns.compact(max_segments=1))
print(ns.gc(min_age=timedelta(0)))

# maintain() is the one call to schedule: it compacts when a trigger is
# crossed, then collects.
load("c", 100)
print(ns.maintain(tail_objects=1))

# Or let a background loop do it. The first run starts right away.
load("d", 100)
m = ns.start_maintenance(timedelta(seconds=1), tail_objects=1)
while m.runs == 0:
    time.sleep(0.05)
m.stop()
print("runs:", m.runs, "errors:", m.errors, "last:", m.last_report)

ns.close()  # also stops any maintainer still running
