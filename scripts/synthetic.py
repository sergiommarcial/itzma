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

A fix or invisible speedup with nothing to show is a registry entry with a
`test_only` reason naming its test, instead of a builder. It builds no data and
shows as "test only" in the index. A new API or feature never takes it.

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
    build: Callable[[Path, str, random.Random], Built] | None = None
    # `demo: none` in the release plan's words: a fix or invisible speedup with
    # nothing to show. Names the test that covers it, builds no dataset, and
    # shows as "test only" in the index. A new API or feature never takes it.
    test_only: str = ""

    def __post_init__(self) -> None:
        if (self.build is None) == (not self.test_only):
            raise SyntheticError(
                f"{self.name}: give either a builder or a test_only reason, not both or neither"
            )


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


BATCH_DIM = 4
BATCHES = {"small": (20, 10), "full": (400, 50)}


def build_small_batches(directory: Path, size: str, rng: random.Random) -> Built:
    """Many small writes, one WAL object each: `batch` says which write a row belongs to."""
    batches, per_batch = BATCHES[size]
    rows = []
    for batch in range(batches):
        for item in range(per_batch):
            rows.append(
                {
                    "id": f"b{batch:04d}-{item:03d}",
                    "batch": batch,
                    "vector": [round(rng.gauss(0.0, 1.0), 6) for _ in range(BATCH_DIM)],
                    "attrs": {"tag": ["a", "b", "c"][(batch + item) % 3]},
                }
            )
    write_jsonl(directory / "rows.jsonl", rows)
    return Built(files=("rows.jsonl",), rows=len(rows))


REGISTRY: tuple[Dataset, ...] = (
    Dataset(
        name="filters",
        since="0.1.0",
        summary="Typed attributes with skewed values for filters: a rare tag, a common tag, a price range",
        build=build_filters,
    ),
    Dataset(
        name="small-batches",
        since="0.3.0",
        summary="Many small writes, one WAL object each, for the maintenance lease, bounded staleness and the IO gate",
        build=build_small_batches,
    ),
    Dataset(
        name="sparse-score-order",
        since="0.3.1",
        summary="An indexed sparse query sums a row's terms in ascending term id, so it scores the flat path's bits",
        test_only="same bits on every run; the test builds its own sparse data, `tests/sparse_query.rs`",
    ),
    Dataset(
        name="write-key-body-compare",
        since="0.3.1",
        summary="A write key reused for a different batch of the same size is a conflict, not a retry",
        test_only="a refusal with nothing to show; `tests/wal.rs`",
    ),
    Dataset(
        name="merge-budget-fit",
        since="0.3.1",
        summary="Merge groups are sized by what each segment charges, so no group goes past the read budget",
        test_only="a limit that is never reached; `tests/compaction.rs` and the planner's unit tests",
    ),
    Dataset(
        name="hnsw-resume",
        since="0.3.1",
        summary="A widening HNSW query resumes its walk instead of starting a fresh search",
        test_only="same rows as a fresh search, speedup not measured; unit tests in `src/index/hnsw.rs`",
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
        if dataset.test_only:
            lines.append(
                f"| {dataset.name} | {dataset.since} | - | - | {dataset.summary} "
                f"(test only: {dataset.test_only}) |"
            )
        elif meta:
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
    chosen = [d for d in REGISTRY if d.build] if names is None else [find(name) for name in names]
    for dataset in chosen:
        if dataset.test_only:
            raise SyntheticError(
                f"{dataset.name} is test only ({dataset.test_only}); it has no dataset"
            )
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
                kind = "  [test only]" if dataset.test_only else ""
                print(f"{dataset.name:<24} {dataset.since:<7} {dataset.summary}{kind}")
            return 0
        for meta in build(None if args.all else args.only, args.size, args.out):
            print(f"{meta['name']}: {meta['rows']} rows ({meta['size']})")
    except SyntheticError as error:
        print(f"synthetic: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
