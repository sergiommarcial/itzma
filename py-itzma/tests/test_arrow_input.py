from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
import pytest
from conftest import random_rows, schema

import itzma

Q = [0.1, -0.2, 0.3, 0.05]


def results(ns: itzma.Namespace) -> list[tuple[str, dict[str, Any] | None]]:
    return [(h.id, h.attrs) for h in ns.query(Q, 100)]


def test_every_input_form_agrees(store: itzma.Store, ns_id: str) -> None:
    rows = random_rows(60)
    ids = [r["id"] for r in rows]
    vectors = np.array([r["vector"] for r in rows], dtype=np.float32)
    attrs = [r["attrs"] for r in rows]

    by_rows = itzma.Namespace.create(store, ns_id + "r", schema())
    by_rows.upsert(rows)

    by_columns = itzma.Namespace.create(store, ns_id + "c", schema())
    by_columns.upsert(ids=ids, vectors=vectors, attrs=attrs)

    table = pa.table(
        {
            "id": ids,
            "vector": pa.FixedSizeListArray.from_arrays(pa.array(vectors.ravel()), 4),
            **{k: [a[k] for a in attrs] for k in attrs[0]},
        }
    )
    by_pyarrow = itzma.Namespace.create(store, ns_id + "p", schema())
    by_pyarrow.upsert(table)

    by_polars = itzma.Namespace.create(store, ns_id + "l", schema())
    by_polars.upsert(pl.from_arrow(table))

    expected = results(by_rows)
    assert results(by_columns) == expected
    assert results(by_pyarrow) == expected
    assert results(by_polars) == expected


def test_polars_types_are_accepted(store: itzma.Store, ns_id: str) -> None:
    df = pl.DataFrame(
        {
            "id": ["a", "b"],
            "vector": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],  # List(Float64)
            "tag": ["x", "y"],  # String, exported as Utf8View
            "year": [2020, 2021],
            "labels": [["red"], []],
        }
    )
    array_df = df.with_columns(pl.col("vector").cast(pl.Array(pl.Float32, 4)))
    for i, frame in enumerate([df, array_df]):
        ns = itzma.Namespace.create(store, f"{ns_id}{i}", schema())
        ns.upsert(frame)
        hit = ns.query([1.0, 0.0, 0.0, 0.0], 1)[0]
        assert hit.id == "a"
        assert hit.attrs == {"tag": "x", "year": 2020, "labels": ["red"]}


def test_a_sparse_column_is_an_undeclared_attribute(store: itzma.Store, ns_id: str) -> None:
    sparse_type = pa.list_(pa.struct([("term", pa.uint32()), ("weight", pa.float32())]))
    table = pa.table(
        {
            "id": ["a", "b"],
            "vector": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            "sparse": pa.array([[{"term": 3, "weight": 0.5}], None], type=sparse_type),
        }
    )
    ns = itzma.Namespace.create(store, ns_id, schema())
    with pytest.raises(itzma.InvalidInput):
        ns.upsert(table)
    assert len(ns.query(Q, 10)) == 0


def test_missing_vector_column(store: itzma.Store, ns_id: str) -> None:
    ns = itzma.Namespace.create(store, ns_id, schema())
    with pytest.raises(itzma.InvalidInput, match="vector"):
        ns.upsert(pa.table({"id": ["a"]}))


def test_rows_and_columns_together_are_refused(store: itzma.Store, ns_id: str) -> None:
    ns = itzma.Namespace.create(store, ns_id, schema())
    with pytest.raises(itzma.InvalidInput, match="not both"):
        ns.upsert(random_rows(1), ids=["a"], vectors=[[0.0] * 4])
