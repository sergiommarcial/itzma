"""Tests for synthetic.py, run by `make release-test`."""

import json
from pathlib import Path

import pytest

import synthetic


def hashes(out: Path) -> dict[str, str]:
    return {
        str(path.relative_to(out)): synthetic.sha256(path)
        for path in sorted(out.rglob("*"))
        if path.is_file()
    }


def test_two_builds_are_byte_identical(tmp_path: Path) -> None:
    synthetic.build(None, "small", tmp_path / "a")
    synthetic.build(None, "small", tmp_path / "b")
    assert hashes(tmp_path / "a") == hashes(tmp_path / "b")


def test_full_has_more_rows_than_small(tmp_path: Path) -> None:
    small = synthetic.build(["filters"], "small", tmp_path / "s")[0]["rows"]
    assert small == synthetic.FILTER_ROWS["small"]
    assert synthetic.FILTER_ROWS["full"] > small


def test_every_dataset_has_a_unique_name_and_a_version() -> None:
    names = [d.name for d in synthetic.REGISTRY]
    assert len(names) == len(set(names))
    assert all(d.since.count(".") == 2 for d in synthetic.REGISTRY)


def test_datasets_draw_from_their_own_seed() -> None:
    assert synthetic.rng_for("a").random() != synthetic.rng_for("b").random()
    assert synthetic.rng_for("a").random() == synthetic.rng_for("a").random()


def test_filters_rows_are_skewed_and_well_formed(tmp_path: Path) -> None:
    synthetic.build(["filters"], "small", tmp_path)
    rows = [
        json.loads(line)
        for line in (tmp_path / "filters" / "docs.jsonl").read_text().splitlines()
    ]
    assert len({row["id"] for row in rows}) == len(rows)
    assert all(len(row["vector"]) == synthetic.FILTER_DIM for row in rows)
    tags = [row["attrs"]["tag"] for row in rows]
    assert 0 < tags.count("rare") < tags.count("common")


def test_index_and_readme_name_what_was_built(tmp_path: Path) -> None:
    synthetic.build(["filters"], "small", tmp_path)
    index = (tmp_path / "INDEX.md").read_text()
    assert "[filters](filters/README.md)" in index
    readme = (tmp_path / "filters" / "README.md").read_text()
    assert "scripts/synthetic.py build --only filters --size small" in readme
    meta = json.loads((tmp_path / "filters" / synthetic.META).read_text())
    assert meta["files"]["docs.jsonl"] in readme


def test_building_one_leaves_the_index_of_the_others(tmp_path: Path) -> None:
    synthetic.build(None, "small", tmp_path)
    before = (tmp_path / "INDEX.md").read_text()
    synthetic.build(["filters"], "small", tmp_path)
    assert (tmp_path / "INDEX.md").read_text() == before


def test_unknown_name_and_size_are_refused(tmp_path: Path) -> None:
    with pytest.raises(synthetic.SyntheticError, match="no dataset named"):
        synthetic.build(["nope"], "small", tmp_path)
    with pytest.raises(synthetic.SyntheticError, match="size must be"):
        synthetic.build(None, "huge", tmp_path)


def test_cli_builds_and_reports(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert synthetic.main(["build", "--only", "filters", "--out", str(tmp_path)]) == 0
    assert "filters: 500 rows (small)" in capsys.readouterr().out
    assert synthetic.main(["build", "--only", "nope", "--out", str(tmp_path)]) == 1
    assert "no dataset named" in capsys.readouterr().err
