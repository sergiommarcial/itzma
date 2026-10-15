from __future__ import annotations

import numpy as np
import pytest
from conftest import Model, assert_hits, random_rows, schema

import itzma

Q = [0.1, -0.2, 0.3, 0.05]


def test_rows_then_query(store: itzma.Store, ns_id: str) -> None:
    ns = itzma.Namespace.create(store, ns_id, schema())
    rows = random_rows(50)
    assert ns.upsert(rows) == 1
    model = Model()
    model.upsert(rows)
    assert_hits(ns.query(Q, 10), model.top_k(Q, 10))
    best = ns.query(Q, 1)[0]
    assert best.attrs == model.docs[best.id]["attrs"]


async def test_async_rows_then_query(store: itzma.Store, ns_id: str) -> None:
    ns = await itzma.AsyncNamespace.create(store, ns_id, schema())
    rows = random_rows(20)
    await ns.upsert(rows)
    model = Model()
    model.upsert(rows)
    assert_hits(await ns.query(Q, 5), model.top_k(Q, 5))


def test_columns_equal_rows(store: itzma.Store, ns_id: str) -> None:
    rows = random_rows(40)
    a = itzma.Namespace.create(store, ns_id, schema())
    a.upsert(rows)
    b = itzma.Namespace.create(store, ns_id + "b", schema())
    b.upsert(
        ids=[r["id"] for r in rows],
        vectors=np.array([r["vector"] for r in rows], dtype=np.float32),
        attrs=[r["attrs"] for r in rows],
    )
    assert [(h.id, h.attrs) for h in a.query(Q, 40)] == [(h.id, h.attrs) for h in b.query(Q, 40)]


def test_float64_and_strided_numpy_are_accepted(store: itzma.Store, ns_id: str) -> None:
    rng = np.random.default_rng(1)
    wide = rng.standard_normal((30, 8))  # float64
    strided = wide[:, ::2]  # (30, 4), not contiguous
    ns = itzma.Namespace.create(store, ns_id, schema())
    ns.upsert(ids=[f"x{i}" for i in range(30)], vectors=strided)
    model = Model()
    model.upsert([{"id": f"x{i}", "vector": strided[i]} for i in range(30)])
    assert_hits(ns.query(np.asarray(Q), 5), model.top_k(Q, 5))


def test_numpy_scalars_in_attrs(store: itzma.Store, ns_id: str) -> None:
    ns = itzma.Namespace.create(store, ns_id, schema())
    ns.upsert(
        [
            {
                "id": "a",
                "vector": [1.0, 0.0, 0.0, 0.0],
                "attrs": {
                    "year": np.int64(2020),
                    "flag": np.bool_(True),
                    "scores": [np.float32(1.5)],
                },
            }
        ]
    )
    attrs = ns.query([1.0, 0.0, 0.0, 0.0], 1)[0].attrs
    assert attrs == {"year": 2020, "flag": True, "scores": [1.5]}
    assert attrs is not None
    assert type(attrs["year"]) is int and type(attrs["flag"]) is bool


def test_empty_writes_are_refused(store: itzma.Store, ns_id: str) -> None:
    ns = itzma.Namespace.create(store, ns_id, schema())
    with pytest.raises(itzma.InvalidInput):
        ns.upsert([])
    with pytest.raises(itzma.InvalidInput):
        ns.delete([])
    assert len(ns.query(Q, 10)) == 0


def test_wrong_dim_rejects_the_whole_batch(store: itzma.Store, ns_id: str) -> None:
    ns = itzma.Namespace.create(store, ns_id, schema())
    rows = random_rows(3)
    rows[1]["vector"] = [1.0, 2.0, 3.0]
    with pytest.raises(itzma.InvalidVector) as info:
        ns.upsert(rows)
    assert (info.value.expected, info.value.got) == (4, 3)
    assert info.value.last_seq is None
    assert len(ns.query(Q, 10)) == 0


def test_bad_attrs(store: itzma.Store, ns_id: str) -> None:
    ns = itzma.Namespace.create(store, ns_id, schema())
    with pytest.raises(itzma.InvalidAttrs) as info:
        ns.upsert([{"id": "a", "vector": Q, "attrs": {"nope": 1}}])
    assert info.value.field == "nope"
    with pytest.raises(itzma.InvalidAttrs) as info2:
        ns.upsert([{"id": "a", "vector": Q, "attrs": {"year": "old"}}])
    assert (info2.value.field, info2.value.expected, info2.value.found) == (
        "year",
        "number",
        "string",
    )


def test_unknown_row_key(store: itzma.Store, ns_id: str) -> None:
    ns = itzma.Namespace.create(store, ns_id, schema())
    with pytest.raises(itzma.InvalidInput, match="extra"):
        ns.upsert([{"id": "a", "vector": Q, "extra": 1}])


def test_delete(store: itzma.Store, ns_id: str) -> None:
    ns = itzma.Namespace.create(store, ns_id, schema())
    rows = random_rows(10)
    ns.upsert(rows)
    ns.delete(["d00000", "d00001"])
    model = Model()
    model.upsert(rows)
    model.delete(["d00000", "d00001"])
    assert_hits(ns.query(Q, 10), model.top_k(Q, 10))


def test_keyed_retry_lands_once(store: itzma.Store, ns_id: str) -> None:
    ns = itzma.Namespace.create(store, ns_id, schema())
    rows = random_rows(5)
    seq = ns.upsert(rows, key="req-1")
    assert ns.upsert(rows, key="req-1", retry_from=seq) == seq
    with pytest.raises(itzma.KeyConflict) as info:
        ns.upsert(random_rows(9), key="req-1", retry_from=seq)
    assert info.value.key == "req-1"
    with pytest.raises(itzma.InvalidInput, match="retry_from"):
        ns.upsert(rows, retry_from=1)


def big_load(dim: int = 1024, n: int = 10_000) -> tuple[list[str], np.ndarray]:
    """About 41 MB of vectors: two batches under the 32 MiB target."""
    rng = np.random.default_rng(2)
    return [f"v{i:05d}" for i in range(n)], rng.standard_normal((n, dim)).astype(np.float32)


def test_big_unkeyed_load_is_split(ns_id: str) -> None:
    ns = itzma.Namespace.create(
        itzma.Store("memory://"), ns_id, itzma.Schema(dim=1024, metric=itzma.Metric.L2)
    )
    ids, vectors = big_load()
    assert ns.upsert(ids=ids, vectors=vectors) == 2


def test_a_bad_vector_anywhere_in_a_big_load_writes_nothing(ns_id: str) -> None:
    ns = itzma.Namespace.create(
        itzma.Store("memory://"), ns_id, itzma.Schema(dim=1024, metric=itzma.Metric.L2)
    )
    ids, vectors = big_load()
    bad = np.vstack([vectors, np.full((1, 1024), np.nan, dtype=np.float32)])
    with pytest.raises(itzma.InvalidVector) as info:
        ns.upsert(ids=ids + ["nan"], vectors=bad)
    assert info.value.last_seq is None
    assert len(ns.query(np.zeros(1024, dtype=np.float32), 5)) == 0


def test_a_late_failure_reports_last_seq_and_resuming_converges(ns_id: str) -> None:
    ns = itzma.Namespace.create(
        itzma.Store("memory://"),
        ns_id,
        itzma.Schema(dim=1024, metric=itzma.Metric.L2, attrs={"n": itzma.AttrType.NUMBER}),
    )
    ids, vectors = big_load()
    attrs: list[dict[str, object]] = [{"n": i} for i in range(len(ids))]
    attrs[-1] = {"n": "not a number"}  # attributes are checked per batch, by the core
    with pytest.raises(itzma.InvalidAttrs) as info:
        ns.upsert(ids=ids, vectors=vectors, attrs=attrs)
    assert info.value.last_seq == 1

    # Resume: the first batch landed, so re-send from where it stopped.
    landed = len(ns.filter(itzma.attr("n").ge(0), limit=10_000))
    attrs[-1] = {"n": len(ids) - 1}
    ns.upsert(ids=ids[landed:], vectors=vectors[landed:], attrs=attrs[landed:])
    assert len(ns.filter(itzma.attr("n").ge(0), limit=10_000)) == len(ids)
