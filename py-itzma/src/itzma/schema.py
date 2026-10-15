"""A namespace's shape: vector length, distance, and typed attributes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum


class Metric(str, Enum):
    """How distance is measured. Lower is closer for all three."""

    COSINE = "cosine"
    L2 = "l2"
    DOT = "dot"


class AttrType(str, Enum):
    """The type of one attribute field."""

    BOOL = "bool"
    NUMBER = "number"
    STRING = "string"
    SET_STRING = "set<string>"
    SET_NUMBER = "set<number>"


@dataclass(frozen=True)
class Schema:
    """Declared once at ``create`` and never changed.

    ``attrs`` maps each field name to its type. At ``create``, ``None`` means
    no attributes. ``Namespace.schema`` reports ``attrs=None`` only for a
    namespace made before typed attributes existed.
    """

    dim: int
    metric: Metric
    attrs: Mapping[str, AttrType] | None = None

    def _attr_names(self) -> dict[str, str]:
        return {name: AttrType(ty).value for name, ty in (self.attrs or {}).items()}

    @staticmethod
    def _from_json(text: str) -> Schema:
        raw = json.loads(text)
        attrs = raw.get("attrs")
        return Schema(
            dim=raw["dim"],
            metric=Metric(raw["metric"]),
            attrs=None if attrs is None else {k: AttrType(v) for k, v in attrs.items()},
        )
