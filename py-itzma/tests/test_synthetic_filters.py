"""The 0.1.0 filters demo, run on the `small` synthetic dataset built into a temp dir."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import filters
import pytest
import synthetic

import itzma


@pytest.fixture(scope="module")
def docs(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, Any]]:
    out = tmp_path_factory.mktemp("synthetic")
    synthetic.build(["filters"], "small", out)
    return filters.load(out / "filters" / "docs.jsonl")


@pytest.fixture(scope="module")
def ns(docs: list[dict[str, Any]]) -> itzma.Namespace:
    return filters.populate(docs)


def test_a_filter_counts_what_the_data_holds(
    docs: list[dict[str, Any]], ns: itzma.Namespace
) -> None:
    result = filters.tags(docs, ns)
    assert result["found"] == result["expected"]
    assert 0 < result["expected"]["rare"] < result["expected"]["common"]


def test_a_filtered_query_returns_only_matching_documents(
    docs: list[dict[str, Any]], ns: itzma.Namespace
) -> None:
    result = filters.tags(docs, ns)
    assert result["query_hits"] == min(10, result["expected"]["rare"])
    assert result["query_hits_all_rare"]


def test_ranges_and_conjunctions_match_a_plain_scan(
    docs: list[dict[str, Any]], ns: itzma.Namespace
) -> None:
    result = filters.prices(docs, ns)
    for found, expected in result.values():
        assert found == expected
    assert result["cheap"][1] > 0


def test_main_reports_a_missing_dataset(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert filters.main([str(tmp_path / "nope.jsonl")]) == 1
    assert "make synthetic" in capsys.readouterr().err
