"""The 0.3.0 demo, run on the `small` synthetic dataset built into a temp dir."""

from __future__ import annotations

from pathlib import Path

import pytest
import small_batches
import synthetic


@pytest.fixture(scope="module")
def batches(tmp_path_factory: pytest.TempPathFactory) -> list[list[dict]]:
    out = tmp_path_factory.mktemp("synthetic")
    synthetic.build(["small-batches"], "small", out)
    return small_batches.load(out / "small-batches" / "rows.jsonl")


def test_the_dataset_is_many_small_writes(batches: list[list[dict]]) -> None:
    assert len(batches) == synthetic.BATCHES["small"][0]
    assert all(len(batch) == synthetic.BATCHES["small"][1] for batch in batches)


def test_one_maintainer_leads_and_the_other_takes_over(batches: list[list[dict]]) -> None:
    result = small_batches.lease(batches)
    assert result["roles_before"] == ["follower", "leader"]
    assert result["taken_over"]


def test_a_stale_reader_misses_writes_but_a_writer_sees_its_own(
    batches: list[list[dict]],
) -> None:
    result = small_batches.staleness(batches)
    assert result["stale_reader_saw"] == 0
    assert result["fresh_reader_saw"] == result["written"]
    assert result["writer_saw_own_write"]


def test_the_tightest_gate_changes_no_answer(batches: list[list[dict]]) -> None:
    result = small_batches.gate(batches)
    assert result["same_as_ungated"]
    assert result["thread_mismatches"] == 0


def test_main_reports_a_missing_dataset(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert small_batches.main([str(tmp_path / "nope.jsonl")]) == 1
    assert "make synthetic" in capsys.readouterr().err
