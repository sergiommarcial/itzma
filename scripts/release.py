"""Version bumps, changelog sections and tag checks for a release.

The version lives in the workspace `Cargo.toml` (every crate inherits it, and
maturin reads it for the Python package), in the exact pin each crate puts on
another (`itzma-python` on `itzma-core`, the adapters on `itzma-store`), and
in `Cargo.lock`. This script keeps them in step.

    release.py prepare <patch|minor|major|X.Y.Z>   bump, cut CHANGELOG, print version
    release.py check vX.Y.Z                         fail unless every version matches
    release.py notes X.Y.Z                          print that CHANGELOG section
"""

import re
import sys
from datetime import date
from pathlib import Path

VERSION = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
UNRELEASED = "## [Unreleased]"


class ReleaseError(Exception):
    """A release step that must stop the workflow."""


def parse(version: str) -> tuple[int, int, int]:
    match = VERSION.match(version)
    if not match:
        raise ReleaseError(f"{version!r} is not a version like 1.2.3")
    major, minor, patch = (int(part) for part in match.groups())
    return major, minor, patch


def next_version(current: str, bump: str) -> str:
    major, minor, patch = parse(current)
    if bump == "patch":
        return f"{major}.{minor}.{patch + 1}"
    if bump == "minor":
        return f"{major}.{minor + 1}.0"
    if bump == "major":
        return f"{major + 1}.0.0"
    if parse(bump) <= (major, minor, patch):
        raise ReleaseError(f"{bump} is not newer than the current {current}")
    return bump


def _workspace_version(text: str) -> re.Match[str]:
    section = text.index("[workspace.package]")
    match = re.compile(r'^version = "([^"]*)"', re.MULTILINE).search(text, section)
    if not match:
        raise ReleaseError("no version in [workspace.package]")
    return match


def current_version(root: Path) -> str:
    return _workspace_version((root / "Cargo.toml").read_text()).group(1)


# A dependency of one workspace crate on another: a path plus an exact pin.
PIN = re.compile(r'^(itzma-[a-z-]+) = \{([^}\n]*)\}', re.MULTILINE)
EXACT = re.compile(r'version = "=([^"]*)"')


def _manifests(root: Path) -> list[Path]:
    return sorted(root.glob("crates/*/Cargo.toml"))


def _crate_name(manifest: Path) -> str:
    match = re.search(r'^\[package\]\nname = "([^"]*)"', manifest.read_text(), re.MULTILINE)
    if not match:
        raise ReleaseError(f"{manifest}: no package name")
    return match.group(1)


def _is_pinned(dep: re.Match[str], own: str) -> bool:
    # A crate's dev-dependency on itself (to turn on a feature for its own
    # tests) carries no version; cargo drops it when publishing.
    return "path = " in dep.group(2) and dep.group(1) != own


def _pins(manifest: Path) -> list[tuple[str, re.Match[str]]]:
    """Each workspace dependency in `manifest`, with the match of its exact pin."""
    own = _crate_name(manifest)
    pins = []
    for dep in PIN.finditer(manifest.read_text()):
        if not _is_pinned(dep, own):
            continue
        exact = EXACT.search(dep.group(2))
        if not exact:
            raise ReleaseError(
                f"{manifest.parent.name} does not pin {dep.group(1)} with version = \"=X.Y.Z\""
            )
        pins.append((dep.group(1), exact))
    return pins


def _lock_pattern(name: str) -> re.Pattern[str]:
    return re.compile(rf'(name = "{re.escape(name)}"\nversion = ")([^"]*)(")')


def set_version(root: Path, version: str) -> None:
    for manifest in _manifests(root):
        _pins(manifest)  # refuse a path dependency without an exact pin, before any write
    cargo = root / "Cargo.toml"
    text = cargo.read_text()
    match = _workspace_version(text)
    cargo.write_text(text[: match.start(1)] + version + text[match.end(1) :])

    for manifest in _manifests(root):
        own = _crate_name(manifest)
        text = PIN.sub(
            lambda dep: EXACT.sub(f'version = "={version}"', dep.group(0))
            if _is_pinned(dep, own)
            else dep.group(0),
            manifest.read_text(),
        )
        manifest.write_text(text)

    # uv.lock records no version for itzma: pyproject takes it from Cargo.
    lock = root / "Cargo.lock"
    text = lock.read_text()
    for name in [_crate_name(m) for m in _manifests(root)]:
        text, count = _lock_pattern(name).subn(rf"\g<1>{version}\g<3>", text)
        if count != 1:
            raise ReleaseError(f"Cargo.lock: expected one {name} entry, found {count}")
    lock.write_text(text)


def check(root: Path, tag: str) -> None:
    want = tag.removeprefix("v")
    found = {"Cargo.toml": current_version(root)}
    for manifest in _manifests(root):
        for dep, exact in _pins(manifest):
            found[f"{manifest.parent.name}'s {dep} pin"] = exact.group(1)
    lock = (root / "Cargo.lock").read_text()
    for name in [_crate_name(m) for m in _manifests(root)]:
        match = _lock_pattern(name).search(lock)
        found[f"Cargo.lock {name}"] = match.group(2) if match else "missing"
    wrong = {where: v for where, v in found.items() if v != want}
    if wrong:
        listed = ", ".join(f"{where} has {v}" for where, v in wrong.items())
        raise ReleaseError(f"tag {tag} wants {want}, but {listed}")


def _section(text: str, heading: str) -> tuple[int, int]:
    start = text.find(heading)
    if start < 0:
        raise ReleaseError(f"CHANGELOG.md has no {heading!r} section")
    body = start + len(heading)
    end = text.find("\n## [", body)
    return body, len(text) if end < 0 else end + 1


def cut_changelog(text: str, version: str, day: str) -> str:
    body_start, body_end = _section(text, UNRELEASED)
    body = text[body_start:body_end].strip("\n")
    if not body:
        raise ReleaseError("CHANGELOG.md's [Unreleased] section is empty; add notes first")
    section = f"{UNRELEASED}\n\n## [{version}] - {day}\n\n{body}\n\n"
    return text[: body_start - len(UNRELEASED)] + section + text[body_end:]


def notes(text: str, version: str) -> str:
    heading = f"## [{version}]"
    body_start, body_end = _section(text, heading)
    # Skip the rest of the heading line (" - date").
    body_start = text.index("\n", body_start)
    return text[body_start:body_end].strip("\n")


def prepare(root: Path, bump: str, day: str) -> str:
    current = current_version(root)
    changelog = root / "CHANGELOG.md"
    # The version in the manifests may itself be unreleased (the very first
    # release): it can be released as it is while it has no section yet.
    if bump == current and f"## [{current}]" not in changelog.read_text():
        version = current
    else:
        version = next_version(current, bump)
    # Cut the changelog first: if it refuses, no file has changed yet.
    cut = cut_changelog(changelog.read_text(), version, day)
    set_version(root, version)
    changelog.write_text(cut)
    return version


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parent.parent
    try:
        match argv:
            case ["prepare", bump]:
                print(prepare(root, bump, date.today().isoformat()))
            case ["check", tag]:
                check(root, tag)
            case ["notes", version]:
                print(notes((root / "CHANGELOG.md").read_text(), version.removeprefix("v")))
            case _:
                print(__doc__, file=sys.stderr)
                return 2
    except ReleaseError as e:
        print(f"release: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
