from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from conftest import Model, assert_hits, random_rows, schema

import itzma

Q = [0.1, -0.2, 0.3, 0.05]


def test_stages_match_the_model(cas_store: itzma.Store, ns_id: str) -> None:
    ns = itzma.Namespace.create(cas_store, ns_id, schema())
    model = Model()

    def check() -> None:
        assert_hits(ns.query(Q, 20, rerank=None), model.top_k(Q, 20))

    first = random_rows(300, seed=7)
    ns.upsert(first)
    model.upsert(first)
    check()

    folded = ns.compact()
    assert folded.kind == "published" and folded.docs == 300
    check()

    second = random_rows(100, seed=8, prefix="e")
    ns.upsert(second)
    ns.delete(["d00000", "d00001"])
    model.upsert(second)
    model.delete(["d00000", "d00001"])
    ns.compact()
    merged = ns.compact(max_segments=1)
    assert merged.kind in ("published", "nothing")
    check()

    report = ns.index(min_rows=0, nlist=1)
    assert report.indexed >= 1
    check()

    rebuilt = ns.index(kind="hnsw", min_rows=0, rebuild=True)
    assert rebuilt.indexed >= 1
    check()

    assert ns.compact().kind == "nothing"


def test_sparse_vectors_survive_compaction(cas_store: itzma.Store, ns_id: str) -> None:
    ns = itzma.Namespace.create(cas_store, ns_id, schema())
    rows = random_rows(50)
    for i, row in enumerate(rows):
        row["sparse"] = [(i % 7, 1.0), (100 + i, 0.5)]
    ns.upsert(rows)
    before = ns.query_sparse([(3, 1.0)], 10).ids
    ns.compact()
    assert ns.query_sparse([(3, 1.0)], 10).ids == before


def test_gc_after_a_merge(cas_store: itzma.Store, ns_id: str) -> None:
    ns = itzma.Namespace.create(cas_store, ns_id, schema())
    for seed in range(3):
        ns.upsert(random_rows(20, seed=seed, prefix=f"s{seed}"))
        ns.compact()
    ns.compact(max_segments=1)
    dry = ns.gc(min_age=timedelta(0), dry_run=True)
    assert dry.segments_deleted >= 1
    real = ns.gc(min_age=timedelta(0))
    assert real.segments_deleted == dry.segments_deleted
    assert ns.gc(min_age=timedelta(0)).segments_deleted == 0


async def test_async_maintain(cas_store: itzma.Store, ns_id: str) -> None:
    ns = await itzma.AsyncNamespace.create(cas_store, ns_id, schema())
    await ns.upsert(random_rows(10))
    report = await ns.maintain(tail_objects=1)
    assert report.compaction is not None and report.compaction.kind == "published"
    assert report.gc is not None
    assert (await ns.maintain(gc=False)).gc is None


def test_file_store_cannot_compact(tmp_path: Path, ns_id: str) -> None:
    ns = itzma.Namespace.create(itzma.Store(f"file://{tmp_path}"), ns_id, schema())
    ns.upsert(random_rows(5))
    with pytest.raises(itzma.StoreError):
        ns.compact()


def test_bad_index_options(cas_store: itzma.Store, ns_id: str) -> None:
    ns = itzma.Namespace.create(cas_store, ns_id, schema())
    with pytest.raises(itzma.InvalidQuery):
        ns.index(nlist=0)
    with pytest.raises(itzma.InvalidInput):
        ns.index(kind="ivf")  # type: ignore[arg-type]
