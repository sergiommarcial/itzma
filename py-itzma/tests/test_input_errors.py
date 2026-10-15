from __future__ import annotations

import numpy as np
import pytest
from conftest import random_rows, schema

import itzma
from itzma import attr

Q = [0.1, -0.2, 0.3, 0.05]


def test_delete_refuses_a_single_string(ns_id: str) -> None:
    ns = itzma.Namespace.create(itzma.Store("memory://"), ns_id, schema())
    ns.upsert([{"id": i, "vector": Q} for i in ("a", "b", "c", "abc")])
    with pytest.raises(itzma.InvalidInput, match="list of ids"):
        ns.delete("abc")
    assert len(ns.query(Q, 10)) == 4


async def test_async_delete_refuses_a_single_string(ns_id: str) -> None:
    ns = await itzma.AsyncNamespace.create(itzma.Store("memory://"), ns_id, schema())
    with pytest.raises(itzma.InvalidInput, match="list of ids"):
        await ns.delete("abc")


def test_negative_top_k_is_an_invalid_query(ns_id: str) -> None:
    ns = itzma.Namespace.create(itzma.Store("memory://"), ns_id, schema())
    ns.upsert(random_rows(5))
    with pytest.raises(itzma.InvalidQuery, match="top_k"):
        ns.query(Q, -1)


def test_a_row_with_a_sparse_key_is_refused(ns_id: str) -> None:
    ns = itzma.Namespace.create(itzma.Store("memory://"), ns_id, schema())
    row = random_rows(1)[0]
    row["sparse"] = [(1, 1.0)]
    with pytest.raises(itzma.InvalidInput, match="unknown key"):
        ns.upsert([row])


def test_numpy_arrays_in_attrs(ns_id: str) -> None:
    ns = itzma.Namespace.create(itzma.Store("memory://"), ns_id, schema())
    ns.upsert(
        [
            {
                "id": "a",
                "vector": Q,
                "attrs": {"labels": np.array(["red", "blue"]), "scores": np.array([1.5, 2.0])},
            }
        ]
    )
    assert ns.query(Q, 1)[0].attrs == {"labels": ["red", "blue"], "scores": [1.5, 2.0]}


def test_predicates_take_numpy_scalars() -> None:
    assert attr("year").eq(np.int64(2020)).to_dict() == attr("year").eq(2020).to_dict()
    assert attr("flag").eq(np.bool_(True)).to_dict() == {"IsTrue": "flag"}
    assert attr("tag").eq(np.str_("x")).to_dict() == attr("tag").eq("x").to_dict()
    assert attr("year").between(np.float32(1), np.int64(3)).to_dict() == (
        attr("year").between(1, 3).to_dict()
    )
    assert attr("year").isin(np.array([1, 2])).to_dict() == attr("year").isin([1, 2]).to_dict()
    assert attr("scores").contains(np.float64(2)).to_dict() == (
        attr("scores").contains(2).to_dict()
    )
    with pytest.raises(TypeError):
        attr("flag").ne(np.bool_(True))
