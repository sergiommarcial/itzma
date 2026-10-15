"""What maintenance calls did."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class Compaction:
    """``nothing`` (no tail, few segments), ``published``, or ``superseded``
    (another process compacted first, which is fine)."""

    kind: Literal["nothing", "published", "superseded"]
    segments: tuple[str, ...] = ()
    folded_through: int | None = None
    merged_segments: int = 0
    docs: int = 0
    tombstones: int = 0

    @staticmethod
    def _from(raw: dict[str, Any]) -> Compaction:
        return Compaction(**{**raw, "segments": tuple(raw.get("segments", ()))})


@dataclass(frozen=True)
class GcReport:
    segments_deleted: int
    bytes_freed: int
    segments_too_young: int
    wal_objects_fenced: int
    wal_bytes_freed: int
    wal_too_young: int
    indexes_deleted: int
    indexes_too_young: int


@dataclass(frozen=True)
class IndexReport:
    indexed: int
    skipped_small: int
    already_indexed: int
    superseded: int


@dataclass(frozen=True)
class MaintenanceReport:
    """One ``maintain`` run. ``index`` is set by the background maintainer
    when it also builds indexes."""

    tail_objects: int
    tail_bytes: int
    compaction: Compaction | None
    gc: GcReport | None
    index: IndexReport | None = None

    @staticmethod
    def _from(raw: dict[str, Any]) -> MaintenanceReport:
        index = raw.get("index")
        return MaintenanceReport(
            tail_objects=raw["tail_objects"],
            tail_bytes=raw["tail_bytes"],
            compaction=None if raw["compaction"] is None else Compaction._from(raw["compaction"]),
            gc=None if raw["gc"] is None else GcReport(**raw["gc"]),
            index=None if index is None else IndexReport(**index),
        )


def parse_maintenance(text: str) -> MaintenanceReport:
    return MaintenanceReport._from(json.loads(text))
