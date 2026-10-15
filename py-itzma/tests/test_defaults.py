from __future__ import annotations

import inspect
import json
from datetime import timedelta
from typing import Any

import pytest

import itzma
from itzma import _itzma

D = json.loads(_itzma.default_options())


def secs(v: Any) -> Any:
    return v.total_seconds() if isinstance(v, timedelta) else v


MAINTAIN = {
    "tail_objects": D["maintain"]["tail_objects"],
    "tail_bytes": D["maintain"]["tail_bytes"],
    "tail_age": D["maintain"]["tail_age_secs"],
    "max_segments": D["maintain"]["max_segments"],
    "max_segment_bytes": D["maintain"]["max_segment_bytes"],
    "gc": D["maintain"]["gc"] is not None,
    "gc_min_age": D["maintain"]["gc"]["min_age_secs"],
    "gc_dry_run": D["maintain"]["gc"]["dry_run"],
    "fence_wal": D["maintain"]["gc"]["fence_wal"],
}

# (method, parameter) -> the core default it must equal
EXPECTED: dict[tuple[str, str], Any] = {
    ("query", "rerank"): D["query"]["rerank"],
    ("query", "nprobe"): D["query"]["nprobe"],
    ("query", "exact"): D["query"]["exact"],
    ("filter", "limit"): D["filter"]["limit"],
    ("filter", "after"): D["filter"]["after"],
    ("index", "kind"): D["index"]["kind"],
    ("index", "min_rows"): D["index"]["min_rows"],
    ("index", "nlist"): D["index"]["nlist"],
    ("index", "rebuild"): D["index"]["rebuild"],
    ("compact", "max_segments"): D["compaction"]["max_segments"],
    ("compact", "max_segment_bytes"): D["compaction"]["max_segment_bytes"],
    ("gc", "min_age"): D["gc"]["min_age_secs"],
    ("gc", "dry_run"): D["gc"]["dry_run"],
    ("gc", "fence_wal"): D["gc"]["fence_wal"],
    ("upsert", "target_bytes"): D["stream"]["target_bytes"],
    ("upsert", "depth"): D["stream"]["depth"],
    **{("maintain", k): v for k, v in MAINTAIN.items()},
    **{("start_maintenance", k): v for k, v in MAINTAIN.items()},
}


@pytest.mark.parametrize("cls", [itzma.Namespace, itzma.AsyncNamespace])
@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_signature_defaults_match_the_core(cls: type, key: tuple[str, str]) -> None:
    method, param = key
    default = inspect.signature(getattr(cls, method)).parameters[param].default
    assert secs(default) == EXPECTED[key]


def test_cache_default_matches_the_core() -> None:
    assert itzma.Cache().ram_bytes == D["cache"]["ram_bytes"]


def test_fetch_policy_default_matches_the_core() -> None:
    assert itzma.FetchPolicy.object_store() == itzma.FetchPolicy(
        min_gap=D["fetch"]["min_gap"],
        max_gap=D["fetch"]["max_gap"],
        gap_divisor=D["fetch"]["gap_divisor"],
        split_above=D["fetch"]["split_above"],
        part_size=D["fetch"]["part_size"],
        align=D["fetch"]["align"],
    )


def test_fetch_policy_local_matches_the_core() -> None:
    # Finding 11: only object_store() (the actual signature default) went
    # through this lockstep check; local()'s six numbers had no check
    # against the Rust side at all, so a future edit to one side could
    # silently drift from the other.
    assert itzma.FetchPolicy.local() == itzma.FetchPolicy(
        min_gap=D["fetch_local"]["min_gap"],
        max_gap=D["fetch_local"]["max_gap"],
        gap_divisor=D["fetch_local"]["gap_divisor"],
        split_above=D["fetch_local"]["split_above"],
        part_size=D["fetch_local"]["part_size"],
        align=D["fetch_local"]["align"],
    )
