from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from itzma import attr
from itzma.predicate import Expr

CASES = json.loads((Path(__file__).parent / "predicate_cases.json").read_text())

BUILT: dict[str, Expr] = {
    "eq_number": attr("year").eq(2020),
    "eq_text": attr("tag").eq("x"),
    "eq_true": attr("flag").eq(True),
    "eq_false": attr("flag").eq(False),
    "ne_number": attr("year").ne(1.5),
    "ne_text": attr("tag").ne("x"),
    "lt": attr("year").lt(3),
    "le": attr("year").le(3),
    "gt": attr("year").gt(3),
    "ge": attr("year").ge(3),
    "between": attr("year").between(2000, 2010),
    "starts_with": attr("tag").starts_with("ab"),
    "isin_numbers": attr("year").isin([1, 2]),
    "isin_text": attr("tag").isin(["a", "b"]),
    "contains_text": attr("labels").contains("red"),
    "contains_number": attr("scores").contains(3),
    "contains_any": attr("labels").contains_any(["a", "b"]),
    "contains_all": attr("scores").contains_all([1, 2]),
    "is_null": attr("tag").is_null(),
    "is_not_null": attr("tag").is_not_null(),
    "is_empty": attr("labels").is_empty(),
    "and_flat": attr("flag").is_true() & attr("tag").is_null() & attr("labels").is_empty(),
    "or_flat": attr("flag").is_true() | attr("tag").is_null() | attr("labels").is_empty(),
    "not": ~attr("flag").is_true(),
    "nested": (attr("flag").is_true() | attr("tag").is_null()) & ~attr("labels").is_empty(),
}


def test_every_case_is_built() -> None:
    assert set(BUILT) == set(CASES)


@pytest.mark.parametrize("name", sorted(CASES))
def test_builder_matches_the_contract(name: str) -> None:
    assert BUILT[name].to_dict() == CASES[name]
    assert json.loads(BUILT[name].to_json()) == CASES[name]


def test_misuse() -> None:
    with pytest.raises(TypeError, match="is_true"):
        attr("flag").ne(True)
    with pytest.raises(TypeError):
        attr("year").lt("x")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="mix"):
        attr("tag").isin(["a", 1])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="&"):
        bool(attr("flag").is_true())
    with pytest.raises(TypeError):
        attr("labels").contains(True)


OPERATORS: list[tuple[Expr, Expr]] = [
    (attr("year") == 2020, attr("year").eq(2020)),
    (attr("tag") == "x", attr("tag").eq("x")),
    (attr("flag") == True, attr("flag").is_true()),  # noqa: E712
    (attr("flag") == False, attr("flag").is_false()),  # noqa: E712
    (attr("tag") == None, attr("tag").is_null()),  # noqa: E711
    (attr("year") != 1.5, attr("year").ne(1.5)),
    (attr("tag") != "x", attr("tag").ne("x")),
    (attr("tag") != None, attr("tag").is_not_null()),  # noqa: E711
    (attr("year") < 3, attr("year").lt(3)),
    (attr("year") <= 3, attr("year").le(3)),
    (attr("year") > 3, attr("year").gt(3)),
    (attr("year") >= 3, attr("year").ge(3)),
    (3 <= attr("year"), attr("year").ge(3)),
    (3 > attr("year"), attr("year").lt(3)),
    (np.int64(3) <= attr("year"), attr("year").ge(3)),
    (attr("year") >= np.float32(3), attr("year").ge(3)),
    (
        (attr("tag") == "news") & (attr("year") >= 2022) | ~(attr("year") < 0),
        attr("tag").eq("news") & attr("year").ge(2022) | ~attr("year").lt(0),
    ),
]


@pytest.mark.parametrize("pair", OPERATORS, ids=lambda p: repr(p[1]))
def test_operators_match_the_methods(pair: tuple[Expr, Expr]) -> None:
    built, method = pair
    assert isinstance(built, Expr)
    assert built.to_dict() == method.to_dict()


def test_operator_misuse() -> None:
    with pytest.raises(TypeError, match="is_true"):
        _ = attr("flag") != True  # noqa: E712
    with pytest.raises(TypeError):
        _ = attr("year") < "x"  # type: ignore[operator]
    with pytest.raises(TypeError, match="parentheses"):
        _ = attr("tag") == "news" & attr("year") >= 2022  # type: ignore[operator]
    with pytest.raises(TypeError, match="parentheses"):
        _ = (attr("tag") == "news") & attr("year")  # type: ignore[operator]
    with pytest.raises(TypeError, match="between"):
        _ = 2020 <= attr("year") < 2025  # type: ignore[operator]
    with pytest.raises(TypeError, match="condition"):
        bool(attr("flag"))
    with pytest.raises(TypeError, match="unhashable"):
        hash(attr("tag"))
