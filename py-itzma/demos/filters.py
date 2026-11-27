"""Demo for typed attributes and filters (0.1.0).

Reads the `filters` synthetic dataset: documents with a rare tag, a common tag
and a price. Build it first, then run this file:

    make synthetic ONLY=filters SIZE=full
    py-itzma/.venv/bin/python py-itzma/demos/filters.py [path/to/docs.jsonl]

The tests import the functions below and run them on the `small` size.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import itzma
from itzma import attr

DEFAULT_DOCS = Path(__file__).resolve().parents[2] / "synthetic" / "filters" / "docs.jsonl"
DIM = 8
PROBE = [0.0] * DIM


def load(path: Path = DEFAULT_DOCS) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def schema() -> itzma.Schema:
    return itzma.Schema(
        dim=DIM,
        metric=itzma.Metric.L2,
        attrs={"tag": itzma.AttrType.STRING, "price": itzma.AttrType.NUMBER},
    )


def populate(docs: list[dict[str, Any]]) -> itzma.Namespace:
    store = itzma.Store("memory://")
    ns = itzma.Namespace.create(store, "filters-demo", schema(), maintenance=None)
    for start in range(0, len(docs), 1000):
        ns.upsert(docs[start : start + 1000])
    return ns


def count(ns: itzma.Namespace, where: Any) -> int:
    """Every document matching `where`, counted by paging through `filter`."""
    total, after = 0, None
    while True:
        page = ns.filter(where, limit=1000, after=after)
        total += len(page)
        after = page.next
        if after is None:
            return total


def tags(docs: list[dict[str, Any]], docs_ns: itzma.Namespace) -> dict[str, Any]:
    """A filter alone counts what the data holds; a filtered query returns only matches."""
    expected = {tag: sum(1 for d in docs if d["attrs"]["tag"] == tag) for tag in ("rare", "common")}
    found = {tag: count(docs_ns, attr("tag").eq(tag)) for tag in expected}
    nearest_rare = docs_ns.query(PROBE, 10, filter=attr("tag").eq("rare"))
    rare_ids = {d["id"] for d in docs if d["attrs"]["tag"] == "rare"}
    return {
        "expected": expected,
        "found": found,
        "query_hits": len(nearest_rare),
        "query_hits_all_rare": all(hit_id in rare_ids for hit_id in nearest_rare.ids),
    }


def prices(docs: list[dict[str, Any]], docs_ns: itzma.Namespace) -> dict[str, Any]:
    """A range and a conjunction, checked against a plain scan of the rows."""
    cheap = attr("price").lt(100)
    cheap_rare = cheap & attr("tag").eq("rare")
    return {
        "cheap": [count(docs_ns, cheap), sum(1 for d in docs if d["attrs"]["price"] < 100)],
        "cheap_rare": [
            count(docs_ns, cheap_rare),
            sum(1 for d in docs if d["attrs"]["price"] < 100 and d["attrs"]["tag"] == "rare"),
        ],
    }


def main(argv: list[str]) -> int:
    path = Path(argv[0]) if argv else DEFAULT_DOCS
    if not path.exists():
        print(f"{path} not found; run: make synthetic ONLY=filters SIZE=full", file=sys.stderr)
        return 1
    docs = load(path)
    ns = populate(docs)
    print(f"{len(docs)} documents")
    print(f"tags: {tags(docs, ns)}")
    print(f"prices (found, expected): {prices(docs, ns)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
