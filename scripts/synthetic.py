"""Deterministic synthetic data for itzma's tests and demos.

Every functionality gets a dataset: a registry entry here, a demo or test that
reads it, and a line in `synthetic/INDEX.md`. The data is generated, never
committed (`synthetic/` is in .gitignore). Same seed, same bytes: no clock, no
environment, only the standard library's `random.Random` seeded from the
dataset's name.

    synthetic.py list                              what exists and since when
    synthetic.py build --all [--size small|full]   build every dataset
    synthetic.py build --only NAME [NAME ...]      build some, leave the rest
    synthetic.py build ... --out DIR               write somewhere else

`small` is for tests (a temp dir, a second or two); `full` is for demos and
benchmarks.
"""

import argparse
import hashlib
import json
import random
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

SIZES = ("small", "full")
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "synthetic"
META = "meta.json"


class SyntheticError(Exception):
    """A request the generator cannot serve."""


@dataclass(frozen=True)
class Built:
    """What a builder wrote: the files (relative to its directory) and a row count."""

    files: tuple[str, ...]
    rows: int


@dataclass(frozen=True)
class Dataset:
    """One registry entry."""

    name: str
    since: str
    summary: str
    build: Callable[[Path, str, random.Random], Built]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """One JSON object per line, keys sorted, so the bytes never depend on dict order."""
    with path.open("w", encoding="utf-8", newline="\n") as out:
        for row in rows:
            out.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            out.write("\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- builders ----------------------------------------------------------------

FILTER_DIM = 8
FILTER_ROWS = {"small": 500, "full": 100_000}


def build_filters(directory: Path, size: str, rng: random.Random) -> Built:
    """Skewed attribute values: a rare tag, a common one and a numeric range."""
    rows = []
    for number in range(FILTER_ROWS[size]):
        draw = rng.random()
        tag = "rare" if draw < 0.01 else "common" if draw < 0.61 else "other"
        rows.append(
            {
                "id": f"doc-{number:07d}",
                "vector": [round(rng.gauss(0.0, 1.0), 6) for _ in range(FILTER_DIM)],
                "attrs": {"tag": tag, "price": round(rng.uniform(0.0, 1000.0), 2)},
            }
        )
    write_jsonl(directory / "docs.jsonl", rows)
    return Built(files=("docs.jsonl",), rows=len(rows))


REGISTRY: tuple[Dataset, ...] = (
    Dataset(
        name="filters",
        since="0.1.0",
        summary="Typed attributes with skewed values for filters: a rare tag, a common tag, a price range",
        build=build_filters,
    ),
)


# --- driver ------------------------------------------------------------------


def find(name: str) -> Dataset:
    for dataset in REGISTRY:
        if dataset.name == name:
            return dataset
    known = ", ".join(d.name for d in REGISTRY)
    raise SyntheticError(f"no dataset named {name!r} (known: {known})")


def rng_for(name: str) -> random.Random:
    # A str seed is hashed with SHA-512 by `random`, the same on every Python.
    return random.Random(f"itzma-synthetic/{name}")


def command_line(dataset: Dataset, size: str) -> str:
    return f"python3 scripts/synthetic.py build --only {dataset.name} --size {size}"


def build_one(dataset: Dataset, size: str, out: Path) -> dict:
    directory = out / dataset.name
    directory.mkdir(parents=True, exist_ok=True)
    built = dataset.build(directory, size, rng_for(dataset.name))
    meta = {
        "name": dataset.name,
        "since": dataset.since,
        "size": size,
        "rows": built.rows,
        "summary": dataset.summary,
        "files": {name: sha256(directory / name) for name in sorted(built.files)},
    }
    (directory / META).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    (directory / "README.md").write_text(readme(dataset, meta, size))
    return meta


def readme(dataset: Dataset, meta: dict, size: str) -> str:
    lines = [
        f"# {dataset.name}",
        "",
        dataset.summary + ".",
        "",
        f"Since itzma {dataset.since}. Size `{size}`, {meta['rows']} rows.",
        "",
        "Rebuild:",
        "",
        "```sh",
        command_line(dataset, size),
        "```",
        "",
        "| file | sha256 |",
        "|---|---|",
    ]
    lines += [f"| `{name}` | `{digest}` |" for name, digest in meta["files"].items()]
    return "\n".join(lines) + "\n"


def write_index(out: Path) -> None:
    """INDEX.md lists every dataset built under `out`, in registry order."""
    metas = {}
    for dataset in REGISTRY:
        path = out / dataset.name / META
        if path.exists():
            metas[dataset.name] = json.loads(path.read_text())
    lines = [
        "# Synthetic data",
        "",
        "Generated by `scripts/synthetic.py`. Not committed.",
        "",
        "| dataset | since | size | rows | what |",
        "|---|---|---|---|---|",
    ]
    for dataset in REGISTRY:
        meta = metas.get(dataset.name)
        if meta:
            lines.append(
                f"| [{dataset.name}]({dataset.name}/README.md) | {meta['since']} | "
                f"{meta['size']} | {meta['rows']} | {meta['summary']} |"
            )
    out.mkdir(parents=True, exist_ok=True)
    (out / "INDEX.md").write_text("\n".join(lines) + "\n")


def build(names: list[str] | None, size: str, out: Path) -> list[dict]:
    """Build the named datasets, or all of them when `names` is None."""
    if size not in SIZES:
        raise SyntheticError(f"size must be one of {', '.join(SIZES)}")
    chosen = list(REGISTRY) if names is None else [find(name) for name in names]
    metas = [build_one(dataset, size, out) for dataset in chosen]
    write_index(out)
    return metas


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="show the registry")
    builder = commands.add_parser("build", help="generate datasets")
    which = builder.add_mutually_exclusive_group(required=True)
    which.add_argument("--all", action="store_true")
    which.add_argument("--only", nargs="+", metavar="NAME")
    builder.add_argument("--size", choices=SIZES, default="small")
    builder.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            for dataset in REGISTRY:
                print(f"{dataset.name:<24} {dataset.since:<7} {dataset.summary}")
            return 0
        for meta in build(None if args.all else args.only, args.size, args.out):
            print(f"{meta['name']}: {meta['rows']} rows ({meta['size']})")
    except SyntheticError as error:
        print(f"synthetic: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
