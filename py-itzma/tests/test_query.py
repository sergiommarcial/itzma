from __future__ import annotations

import numpy as np
import polars as pl
import pyarrow as pa
import pytest
from conftest import Model, assert_hits, random_rows, schema

import itzma
from itzma import attr

Q = [0.1, -0.2, 0.3, 0.05]


@pytest.mark.parametrize("metric", list(itzma.Metric))
def test_flat_query_matches_the_model(store: itzma.Store, ns_id: str, metric: itzma.Metric) -> None:
    ns = itzma.Namespace.create(store, ns_id, schema(metric=metric))
    rows = random_rows(200, seed=3)
    ns.upsert(rows)
    model = Model(metric)
    model.upsert(rows)
    assert_hits(ns.query(Q, 15), model.top_k(Q, 15))
    assert_hits(ns.query(Q, 15, exact=True), model.top_k(Q, 15))


def test_filtered_query_matches_the_model(store: itzma.Store, ns_id: str) -> None:
    ns = itzma.Namespace.create(store, ns_id, schema())
    rows = random_rows(200, seed=4)
    ns.upsert(rows)
    model = Model()
    model.upsert(rows)
    where = attr("tag").eq("a") & attr("labels").contains("red")
    hits = ns.query(Q, 10, filter=where)
    assert_hits(hits, model.top_k(Q, 10, lambda a: a["tag"] == "a" and "red" in a["labels"]))


async def test_async_query_options(store: itzma.Store, ns_id: str) -> None:
    ns = await itzma.AsyncNamespace.create(store, ns_id, schema())
    rows = random_rows(50)
    await ns.upsert(rows)
    model = Model()
    model.upsert(rows)
    assert_hits(await ns.query(Q, 5, rerank=None, exact=True), model.top_k(Q, 5))


def test_query_vector_forms_agree(store: itzma.Store, ns_id: str) -> None:
    ns = itzma.Namespace.create(store, ns_id, schema())
    ns.upsert(random_rows(30))
    expected = ns.query(Q, 5).ids
    wide = np.array([Q[0], 9.0, Q[1], 9.0, Q[2], 9.0, Q[3], 9.0])
    for vector in (np.array(Q), np.array(Q, dtype=np.float32), wide[::2], pa.array(Q), tuple(Q)):
        assert ns.query(vector, 5).ids == expected


def test_wrong_dim_query_vector(store: itzma.Store, ns_id: str) -> None:
    ns = itzma.Namespace.create(store, ns_id, schema())
    with pytest.raises(itzma.InvalidVector) as info:
        ns.query([1.0, 2.0], 5)
    assert (info.value.expected, info.value.got) == (4, 2)


def test_invalid_options(store: itzma.Store, ns_id: str) -> None:
    ns = itzma.Namespace.create(store, ns_id, schema())
    with pytest.raises(itzma.InvalidQuery):
        ns.query(Q, 5, rerank=0)
    with pytest.raises(itzma.InvalidQuery):
        ns.query(Q, 5, nprobe=0)
    with pytest.raises(itzma.InvalidAttrs) as info:
        ns.query(Q, 5, filter=attr("nope").is_null())
    assert info.value.field == "nope"


def test_sparse_query_matches_the_model(store: itzma.Store, ns_id: str) -> None:
    ns = itzma.Namespace.create(store, ns_id, schema())
    rng = np.random.default_rng(5)
    rows = random_rows(100)
    for row in rows:
        terms = rng.choice(50, size=5, replace=False)
        row["sparse"] = [(int(t), float(rng.random())) for t in terms]
    ns.upsert(rows)
    query = [(3, 1.0), (7, 0.5), (11, 2.0)]
    weights = dict(query)
    scored = sorted(
        (-sum(weights.get(t, 0.0) * w for t, w in row["sparse"]), row["id"]) for row in rows
    )
    expected = [i for score, i in scored if score < 0][:10]
    assert ns.query_sparse(query, 10).ids == expected


def test_arrow_output(store: itzma.Store, ns_id: str) -> None:
    ns = itzma.Namespace.create(store, ns_id, schema())
    ns.upsert(random_rows(10))
    hits = ns.query(Q, 3)
    table = hits.to_arrow()
    assert table.column_names == ["id", "distance", "flag", "labels", "scores", "tag", "year"]
    assert table.column("year").type == pa.float64()
    assert table.column("id").to_pylist() == hits.ids
    assert pa.table(hits).num_rows == 3
    df = hits.to_polars()
    assert isinstance(df, pl.DataFrame) and df["id"].to_list() == hits.ids
    assert df.schema == pl.Schema(HIT_SCHEMA)


# The typed layout of a hit table: distance as f32, numbers as f64 even when
# they were written as integers.
HIT_SCHEMA = {
    "id": pl.String,
    "distance": pl.Float32,
    "flag": pl.Boolean,
    "labels": pl.List(pl.String),
    "scores": pl.List(pl.Float64),
    "tag": pl.String,
    "year": pl.Float64,
}


def test_sparse_hits_to_polars_are_typed(store: itzma.Store, ns_id: str) -> None:
    ns = itzma.Namespace.create(store, ns_id, schema())
    rows = random_rows(10)
    for i, row in enumerate(rows):
        row["sparse"] = [(i % 3, 1.0)]
    ns.upsert(rows)
    hits = ns.query_sparse([(1, 1.0)], 3)
    df = hits.to_polars()
    assert df.schema == pl.Schema(HIT_SCHEMA)
    assert df["id"].to_list() == hits.ids
    assert pa.table(hits).schema == hits.to_arrow().schema


def test_read_budget(store: itzma.Store, ns_id: str) -> None:
    itzma.Namespace.create(store, ns_id, schema()).upsert(random_rows(50))
    ns = itzma.Namespace.open(store, ns_id, read_budget=10)
    with pytest.raises(itzma.ReadBudgetExceeded) as info:
        ns.query(Q, 5)
    assert info.value.max == 10
    assert info.value.needed is not None and info.value.needed > 10
    assert isinstance(info.value.path, str)
