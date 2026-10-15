"""Conditions over attributes, in the style of ``pl.col``.

>>> from itzma import attr
>>> where = (attr("tag") == "news") & (attr("year") >= 2022)
>>> same = attr("tag").eq("news") & attr("year").ge(2022)

Each comparison needs its own parentheses, because ``&`` and ``|`` bind
tighter than ``==`` and ``>=``. A chained range such as
``2020 <= attr("year") < 2025`` cannot work (Python joins the two halves
with ``and``) and raises; use ``between``. ``== None`` is ``is_null()``.

Comparisons against a missing or null field are false, and ``~`` flips
that, so ``~attr("tag").eq("x")`` matches documents with no tag (unlike
SQL). A field holding a value of the wrong type counts as null.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

Node = dict[str, Any]


class Expr:
    """A condition. Combine with ``&``, ``|`` and ``~``."""

    __slots__ = ("_node",)

    def __init__(self, node: Node) -> None:
        self._node = node

    def __and__(self, other: Expr) -> Expr:
        return Expr({"And": _flatten("And", self, _condition(other))})

    def __or__(self, other: Expr) -> Expr:
        return Expr({"Or": _flatten("Or", self, _condition(other))})

    def __invert__(self) -> Expr:
        return Expr({"Not": self._node})

    def __bool__(self) -> bool:
        # Also what a chained `2020 <= attr("year") < 2025` hits, since Python
        # joins the two comparisons with `and`.
        raise TypeError(
            "combine conditions with & and |, not `and` and `or`; "
            "for a range use attr(...).between(low, high)"
        )

    def to_dict(self) -> Node:
        return self._node

    def to_json(self) -> str:
        return json.dumps(self._node)

    def __repr__(self) -> str:
        return f"Expr({self._node!r})"


_PARENTHESES = (
    "wrap each comparison in parentheses: "
    '(attr("a") == 1) & (attr("b") >= 2), since & and | bind tighter than == and >='
)


def _condition(value: object) -> Expr:
    if not isinstance(value, Expr):
        raise TypeError(f"& and | take conditions, got {type(value).__name__}; {_PARENTHESES}")
    return value


def _flatten(op: str, a: Expr, b: Expr) -> list[Node]:
    parts: list[Node] = []
    for e in (a, b):
        parts.extend(e._node[op] if op in e._node else [e._node])
    return parts


def _plain(value: Any) -> Any:
    """numpy scalars as plain Python values, so they compare like writes do."""
    if hasattr(value, "dtype") and hasattr(value, "tolist"):
        return value.tolist()
    return value


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _number(value: object) -> float:
    value = _plain(value)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"expected a number, got {type(value).__name__}")
    return float(value)


def _element(value: object) -> Node:
    value = _plain(value)
    if isinstance(value, str):
        return {"Text": value}
    return {"Number": _number(value)}


class Attr:
    """One attribute field. Build conditions with its methods or with
    ``==``, ``!=``, ``<``, ``<=``, ``>`` and ``>=``."""

    __slots__ = ("_name",)
    # `==` builds a condition, so an Attr cannot be a dict key or set member.
    __hash__ = None  # type: ignore[assignment]
    # Makes numpy scalars defer, so `np.int64(3) <= attr("x")` reaches __ge__.
    __array_ufunc__ = None

    def __init__(self, name: str) -> None:
        self._name = name

    def __eq__(self, value: object) -> Expr:  # type: ignore[override]
        """``eq``, or ``is_null()`` for ``None``."""
        return self.is_null() if value is None else self.eq(value)  # type: ignore[arg-type]

    def __ne__(self, value: object) -> Expr:  # type: ignore[override]
        """``ne``, or ``is_not_null()`` for ``None``."""
        return self.is_not_null() if value is None else self.ne(value)  # type: ignore[arg-type]

    def __lt__(self, value: float) -> Expr:
        return self.lt(value)

    def __le__(self, value: float) -> Expr:
        return self.le(value)

    def __gt__(self, value: float) -> Expr:
        return self.gt(value)

    def __ge__(self, value: float) -> Expr:
        return self.ge(value)

    def __and__(self, other: object) -> Expr:
        raise TypeError(_PARENTHESES)

    __rand__ = __or__ = __ror__ = __and__

    def __bool__(self) -> bool:
        raise TypeError('an attribute is not a condition; compare it, e.g. attr("x") == 1')

    def _compare(self, op: str, value: float) -> Expr:
        return Expr({"Number": {"field": self._name, "op": op, "value": _number(value)}})

    def _text(self, op: str, value: str) -> Expr:
        return Expr({"Text": {"field": self._name, "op": op, "value": value}})

    def eq(self, value: bool | float | str) -> Expr:
        """Equal. ``eq(True)`` is ``is_true()``."""
        value = _plain(value)
        if isinstance(value, bool):
            return self.is_true() if value else self.is_false()
        if isinstance(value, str):
            return self._text("Eq", value)
        return self._compare("Eq", value)

    def ne(self, value: float | str) -> Expr:
        """Not equal. For a bool field use ``is_true()`` or ``is_false()``."""
        value = _plain(value)
        if isinstance(value, bool):
            raise TypeError("for a bool field use is_true() or is_false(), not ne()")
        if isinstance(value, str):
            return self._text("Ne", value)
        return self._compare("Ne", value)

    def lt(self, value: float) -> Expr:
        return self._compare("Lt", value)

    def le(self, value: float) -> Expr:
        return self._compare("Le", value)

    def gt(self, value: float) -> Expr:
        return self._compare("Gt", value)

    def ge(self, value: float) -> Expr:
        return self._compare("Ge", value)

    def between(self, low: float, high: float) -> Expr:
        """``low <= field <= high``."""
        return Expr({"Between": {"field": self._name, "low": _number(low), "high": _number(high)}})

    def starts_with(self, prefix: str) -> Expr:
        return self._text("StartsWith", prefix)

    def isin(self, values: Iterable[float] | Iterable[str]) -> Expr:
        """Equal to one of ``values`` (all numbers or all strings)."""
        items: list[Any] = [_plain(v) for v in values]
        if all(isinstance(v, str) for v in items):
            return Expr({"TextIn": {"field": self._name, "values": items}})
        if all(_is_number(v) for v in items):
            return Expr({"NumberIn": {"field": self._name, "values": [float(v) for v in items]}})
        raise TypeError("isin() takes all numbers or all strings, not a mix")

    def contains(self, value: float | str) -> Expr:
        """A set field holds ``value``."""
        return Expr({"Contains": {"field": self._name, "value": _element(value)}})

    def contains_any(self, values: Iterable[float] | Iterable[str]) -> Expr:
        return Expr({"ContainsAny": {"field": self._name, "values": [_element(v) for v in values]}})

    def contains_all(self, values: Iterable[float] | Iterable[str]) -> Expr:
        return Expr({"ContainsAll": {"field": self._name, "values": [_element(v) for v in values]}})

    def is_null(self) -> Expr:
        return Expr({"IsNull": self._name})

    def is_not_null(self) -> Expr:
        return Expr({"IsNotNull": self._name})

    def is_true(self) -> Expr:
        return Expr({"IsTrue": self._name})

    def is_false(self) -> Expr:
        return Expr({"IsFalse": self._name})

    def is_empty(self) -> Expr:
        """A set field is present and has no elements."""
        return Expr({"IsEmpty": self._name})


def attr(name: str) -> Attr:
    """The attribute field ``name``, to build a condition on."""
    return Attr(name)
