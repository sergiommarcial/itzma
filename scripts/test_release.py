"""Tests for release.py, run by `make release-test`."""

from pathlib import Path

import pytest

import release

CARGO = """[workspace]
members = ["crates/itzma-core", "crates/itzma-python"]

[workspace.package]
version = "0.1.0"
edition = "2024"
"""

PYTHON_CARGO = """[package]
name = "itzma-python"
version.workspace = true

[dependencies]
itzma-core = { path = "../itzma-core", version = "=0.1.0" }
"""

CORE_CARGO = """[package]
name = "itzma-core"
version.workspace = true
"""

STORE_CARGO = """[package]
name = "itzma-store"
version.workspace = true

[dev-dependencies]
itzma-store = { path = ".", features = ["conformance"] }
"""

ADAPTER_CARGO = """[package]
name = "itzma-store-gcs"
version.workspace = true

[dependencies]
object_store = "0.14"

[dev-dependencies]
itzma-store = { path = "../itzma-store", version = "=0.1.0", features = ["conformance"] }
"""

LOCK = """[[package]]
name = "itzma-core"
version = "0.1.0"

[[package]]
name = "itzma-store"
version = "0.1.0"

[[package]]
name = "itzma-store-gcs"
version = "0.1.0"

[[package]]
name = "itzma-python"
version = "0.1.0"

[[package]]
name = "itoa"
version = "0.1.0"
"""

# The Python package's version is dynamic (from Cargo), so uv.lock has none.
UV_LOCK = """[[package]]
name = "itzma"
source = { editable = "." }

[[package]]
name = "numpy"
version = "0.1.0"
"""

CHANGELOG = """# Changelog

Intro.

## [Unreleased]

### Added

- Sparse vectors.

## [0.1.0] - 2026-10-01

### Added

- Everything.
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    for crate, manifest in [
        ("itzma-core", CORE_CARGO),
        ("itzma-store", STORE_CARGO),
        ("itzma-store-gcs", ADAPTER_CARGO),
        ("itzma-python", PYTHON_CARGO),
    ]:
        (tmp_path / "crates" / crate).mkdir(parents=True)
        (tmp_path / "crates" / crate / "Cargo.toml").write_text(manifest)
    (tmp_path / "py-itzma").mkdir()
    (tmp_path / "Cargo.toml").write_text(CARGO)
    (tmp_path / "Cargo.lock").write_text(LOCK)
    (tmp_path / "py-itzma/uv.lock").write_text(UV_LOCK)
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG)
    return tmp_path


def test_next_version_bumps_each_part() -> None:
    assert release.next_version("0.1.9", "patch") == "0.1.10"
    assert release.next_version("0.1.9", "minor") == "0.2.0"
    assert release.next_version("0.1.9", "major") == "1.0.0"
    assert release.next_version("0.1.9", "0.3.0") == "0.3.0"


@pytest.mark.parametrize("bad", ["0.1.9", "0.1.8", "1.0", "v0.2.0", "lots"])
def test_next_version_refuses_anything_not_newer_or_malformed(bad: str) -> None:
    with pytest.raises(release.ReleaseError):
        release.next_version("0.1.9", bad)


def test_set_version_updates_every_place(repo: Path) -> None:
    release.set_version(repo, "0.2.0")
    assert 'version = "0.2.0"\nedition' in (repo / "Cargo.toml").read_text()
    assert 'version = "=0.2.0"' in (repo / "crates/itzma-python/Cargo.toml").read_text()
    adapter = (repo / "crates/itzma-store-gcs/Cargo.toml").read_text()
    assert 'version = "=0.2.0", features = ["conformance"]' in adapter
    assert (repo / "crates/itzma-store/Cargo.toml").read_text() == STORE_CARGO
    lock = (repo / "Cargo.lock").read_text()
    for name in ["itzma-core", "itzma-store", "itzma-store-gcs", "itzma-python"]:
        assert f'name = "{name}"\nversion = "0.2.0"' in lock
    assert 'name = "itoa"\nversion = "0.1.0"' in lock
    assert (repo / "py-itzma/uv.lock").read_text() == UV_LOCK
    assert release.current_version(repo) == "0.2.0"


def test_check_accepts_matching_tag_and_refuses_others(repo: Path) -> None:
    release.check(repo, "v0.1.0")
    with pytest.raises(release.ReleaseError, match="0.1.1"):
        release.check(repo, "v0.1.1")
    (repo / "crates/itzma-python/Cargo.toml").write_text(PYTHON_CARGO.replace("=0.1.0", "=0.0.9"))
    with pytest.raises(release.ReleaseError, match="itzma-python"):
        release.check(repo, "v0.1.0")


def test_check_refuses_an_adapter_pin_left_behind(repo: Path) -> None:
    (repo / "crates/itzma-store-gcs/Cargo.toml").write_text(ADAPTER_CARGO.replace("=0.1.0", "=0.0.9"))
    with pytest.raises(release.ReleaseError, match="itzma-store-gcs's itzma-store pin"):
        release.check(repo, "v0.1.0")


def test_a_path_dependency_without_an_exact_pin_is_refused(repo: Path) -> None:
    (repo / "crates/itzma-store-gcs/Cargo.toml").write_text(ADAPTER_CARGO.replace(', version = "=0.1.0"', ""))
    with pytest.raises(release.ReleaseError, match="does not pin itzma-store"):
        release.set_version(repo, "0.2.0")
    assert release.current_version(repo) == "0.1.0"


def test_cut_changelog_dates_the_unreleased_section() -> None:
    out = release.cut_changelog(CHANGELOG, "0.2.0", "2026-11-01")
    assert "## [Unreleased]\n\n## [0.2.0] - 2026-11-01\n\n### Added\n\n- Sparse vectors.\n\n## [0.1.0]" in out


def test_cut_changelog_refuses_an_empty_unreleased_section() -> None:
    empty = CHANGELOG.replace("### Added\n\n- Sparse vectors.\n\n", "")
    with pytest.raises(release.ReleaseError, match="Unreleased"):
        release.cut_changelog(empty, "0.2.0", "2026-11-01")


def test_notes_are_one_version_section() -> None:
    assert release.notes(CHANGELOG, "0.1.0") == "### Added\n\n- Everything."
    with pytest.raises(release.ReleaseError, match="0.9.9"):
        release.notes(CHANGELOG, "0.9.9")


def test_prepare_does_everything_or_nothing(repo: Path) -> None:
    assert release.prepare(repo, "minor", "2026-11-01") == "0.2.0"
    assert "## [0.2.0] - 2026-11-01" in (repo / "CHANGELOG.md").read_text()
    assert release.current_version(repo) == "0.2.0"
    # Nothing new under Unreleased: refused before any file changes.
    with pytest.raises(release.ReleaseError):
        release.prepare(repo, "patch", "2026-11-02")
    assert release.current_version(repo) == "0.2.0"


def test_the_current_version_can_be_released_once(repo: Path) -> None:
    # 0.1.0 is the current version but has no section yet: the first release.
    (repo / "CHANGELOG.md").write_text(CHANGELOG.split("## [0.1.0]")[0])
    assert release.prepare(repo, "0.1.0", "2026-11-01") == "0.1.0"
    assert "## [0.1.0] - 2026-11-01" in (repo / "CHANGELOG.md").read_text()
    # Once it has a section, the same version is refused.
    with pytest.raises(release.ReleaseError):
        release.prepare(repo, "0.1.0", "2026-11-02")
