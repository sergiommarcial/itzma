from __future__ import annotations

from collections.abc import Callable
from typing import Any

import polars as pl
import pytest
from conftest import random_rows, schema

import itzma
from itzma import attr
from itzma.predicate import Expr

Attrs = dict[str, Any]


# Two-valued logic: a comparison against a missing, null or wrongly typed
# field is false; ~ flips it.
def num(a: Attrs, k: str) -> float | None:
    v = a.get(k)
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def txt(a: Attrs, k: str) -> str | None:
    v = a.get(k)
    return v if isinstance(v, str) else None


def sset(a: Attrs, k: str) -> list[Any] | None:
    v = a.get(k)
    return v if isinstance(v, list) else None


def cmp(a: Attrs, k: str, test: Callable[[float], bool]) -> bool:
    v = num(a, k)
    return v is not None and test(v)


CASES: list[tuple[Expr, Callable[[Attrs], bool]]] = [
    (attr("year").eq(2005), lambda a: num(a, "year") == 2005),
    (attr("year").ne(2005), lambda a: cmp(a, "year", lambda v: v != 2005)),
    (attr("year").lt(2005), lambda a: cmp(a, "year", lambda v: v < 2005)),
    (attr("year").le(2005), lambda a: cmp(a, "year", lambda v: v <= 2005)),
    (attr("year").gt(2025), lambda a: cmp(a, "year", lambda v: v > 2025)),
    (attr("year").ge(2025), lambda a: cmp(a, "year", lambda v: v >= 2025)),
    (attr("year").between(2003, 2006), lambda a: cmp(a, "year", lambda v: 2003 <= v <= 2006)),
    (attr("year").isin([2001, 2002]), lambda a: num(a, "year") in (2001, 2002)),
    (attr("tag").eq("a"), lambda a: txt(a, "tag") == "a"),
    (attr("tag").ne("a"), lambda a: txt(a, "tag") is not None and txt(a, "tag") != "a"),
    (attr("tag").starts_with("b"), lambda a: (txt(a, "tag") or "").startswith("b")),
    (attr("tag").isin(["a", "c"]), lambda a: txt(a, "tag") in ("a", "c")),
    (attr("flag").is_true(), lambda a: a.get("flag") is True),
    (attr("flag").is_false(), lambda a: a.get("flag") is False),
    (attr("tag").is_null(), lambda a: a.get("tag") is None),
    (attr("tag").is_not_null(), lambda a: a.get("tag") is not None),
    (attr("labels").contains("red"), lambda a: "red" in (sset(a, "labels") or [])),
    (
        attr("labels").contains_any(["red", "green"]),
        lambda a: bool({"red", "green"} & set(sset(a, "labels") or [])),
    ),
    (
        attr("labels").contains_all(["red", "blue"]),
        lambda a: {"red", "blue"} <= set(sset(a, "labels") or []),
    ),
    (attr("labels").is_empty(), lambda a: sset(a, "labels") == []),
    (attr("scores").contains(2), lambda a: 2.0 in (sset(a, "scores") or [])),
    (~attr("tag").eq("a"), lambda a: txt(a, "tag") != "a"),
    (
        attr("flag").is_true() | attr("tag").is_null(),
        lambda a: a.get("flag") is True or a.get("tag") is None,
    ),
]


def rows_with_nulls() -> list[dict[str, Any]]:
    rows = random_rows(120, seed=6)
    for i, row in enumerate(rows):
        if i % 7 == 0:
            row["attrs"]["tag"] = None
        if i % 11 == 0:
            del row["attrs"]["labels"]
        if i % 13 == 0:
            row["attrs"]["year"] = None
    return rows


@pytest.mark.parametrize("case", range(len(CASES)))
def test_truth_table(case: int, ns_id: str) -> None:
    where, reference = CASES[case]
    ns = itzma.Namespace.create(itzma.Store("memory://"), ns_id, schema())
    rows = rows_with_nulls()
    ns.upsert(rows)
    expected = sorted(r["id"] for r in rows if reference(r["attrs"]))
    page = ns.filter(where, limit=1000)
    assert [r.id for r in page] == expected
    assert page.next is None or len(page) == 1000


def test_paging_across_block_edges(store: itzma.Store, ns_id: str) -> None:
    ns = itzma.Namespace.create(store, ns_id, schema())
    rows = random_rows(600)
    ns.upsert(rows)
    expected = sorted(r["id"] for r in rows if r["attrs"]["flag"])
    got: list[str] = []
    after = None
    while True:
        page = ns.filter(attr("flag").is_true(), limit=97, after=after)
        got.extend(r.id for r in page)
        if page.next is None:
            break
        after = page.next
    assert got == expected


async def test_async_filter(store: itzma.Store, ns_id: str) -> None:
    ns = await itzma.AsyncNamespace.create(store, ns_id, schema())
    await ns.upsert(random_rows(10))
    page = await ns.filter(attr("tag").eq("a"))
    assert [r.id for r in page] == ["d00000", "d00003", "d00006", "d00009"]
    assert page[0].attrs is not None and page[0].attrs["tag"] == "a"
    assert page.to_arrow().column_names == ["id", "flag", "labels", "scores", "tag", "year"]
    assert page.to_arrow().column("tag").to_pylist() == ["a", "a", "a", "a"]


def test_a_filter_page_to_polars_is_typed(store: itzma.Store, ns_id: str) -> None:
    ns = itzma.Namespace.create(store, ns_id, schema())
    ns.upsert(random_rows(10))
    df = ns.filter(attr("tag").eq("a")).to_polars()
    assert df.schema == pl.Schema(
        {
            "id": pl.String,
            "flag": pl.Boolean,
            "labels": pl.List(pl.String),
            "scores": pl.List(pl.Float64),
            "tag": pl.String,
            "year": pl.Float64,
        }
    )
    assert df["year"].to_list() == [2000.0, 2003.0, 2006.0, 2009.0]


def test_bad_filter_requests(store: itzma.Store, ns_id: str) -> None:
    ns = itzma.Namespace.create(store, ns_id, schema())
    with pytest.raises(itzma.InvalidQuery):
        ns.filter(attr("tag").eq("a"), limit=0)
    with pytest.raises(itzma.InvalidAttrs):
        ns.filter(attr("year").eq("text"))
