"""Shared fixtures: the backend matrix, namespace ids, a standard schema."""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import itzma

S3_ENDPOINT = os.environ.get("ITZMA_TEST_S3_ENDPOINT")

ATTRS = {
    "tag": itzma.AttrType.STRING,
    "year": itzma.AttrType.NUMBER,
    "flag": itzma.AttrType.BOOL,
    "labels": itzma.AttrType.SET_STRING,
    "scores": itzma.AttrType.SET_NUMBER,
}


def make_store(kind: str, tmp_path: Path) -> itzma.Store:
    if kind == "memory":
        return itzma.Store("memory://")
    if kind == "file":
        return itzma.Store(f"file://{tmp_path}")
    if S3_ENDPOINT is None:
        pytest.skip("ITZMA_TEST_S3_ENDPOINT is not set")
    bucket = os.environ.get("ITZMA_TEST_S3_BUCKET", "itzma-test")
    return itzma.Store(
        f"s3://{bucket}/py-{uuid.uuid4().hex[:12]}",
        endpoint=S3_ENDPOINT,
        region="us-east-1",
        access_key_id=os.environ.get("ITZMA_TEST_S3_ACCESS_KEY", "itzma"),
        secret_access_key=os.environ.get("ITZMA_TEST_S3_SECRET_KEY", "itzma-test-secret"),
        allow_http=True,
        env=False,
    )


@pytest.fixture(params=["memory", "file", "s3"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> itzma.Store:
    """Every backend. file:// has no conditional update."""
    return make_store(request.param, tmp_path)


@pytest.fixture(params=["memory", "s3"])
def cas_store(request: pytest.FixtureRequest, tmp_path: Path) -> itzma.Store:
    """Backends with conditional update, for compact, index and maintain."""
    return make_store(request.param, tmp_path)


@pytest.fixture
def ns_id() -> str:
    return f"t{uuid.uuid4().hex[:16]}"


def schema(dim: int = 4, metric: itzma.Metric = itzma.Metric.L2) -> itzma.Schema:
    return itzma.Schema(dim=dim, metric=metric, attrs=ATTRS)


def distance(metric: itzma.Metric, a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if metric is itzma.Metric.L2:
        return float(np.sum((a - b) ** 2))
    if metric is itzma.Metric.DOT:
        return float(-np.dot(a, b))
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class Model:
    """What a namespace should hold: the last write of each id wins."""

    def __init__(self, metric: itzma.Metric = itzma.Metric.L2) -> None:
        self.metric = metric
        self.docs: dict[str, dict[str, Any]] = {}

    def upsert(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            self.docs[row["id"]] = row

    def delete(self, ids: list[str]) -> None:
        for i in ids:
            self.docs.pop(i, None)

    def top_k(
        self,
        vector: Any,
        k: int,
        where: Callable[[dict[str, Any]], bool] = lambda attrs: True,
    ) -> list[tuple[str, float]]:
        q = np.asarray(vector, dtype=np.float32)
        scored = [
            (distance(self.metric, np.asarray(d["vector"], dtype=np.float32), q), i)
            for i, d in self.docs.items()
            if where(d.get("attrs") or {})
        ]
        scored.sort()
        return [(i, dist) for dist, i in scored[:k]]


def random_rows(n: int, dim: int = 4, seed: int = 0, prefix: str = "d") -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rows.append(
            {
                "id": f"{prefix}{i:05d}",
                "vector": rng.standard_normal(dim).astype(np.float32).tolist(),
                "attrs": {
                    "tag": ["a", "b", "c"][i % 3],
                    "year": 2000 + i % 30,
                    "flag": i % 2 == 0,
                    "labels": [["red"], ["blue"], [], ["red", "blue"]][i % 4],
                    "scores": [float(i % 5)],
                },
            }
        )
    return rows


def assert_hits(hits: itzma.Hits, expected: list[tuple[str, float]]) -> None:
    assert hits.ids == [i for i, _ in expected]
    np.testing.assert_allclose(hits.distances, [d for _, d in expected], rtol=1e-4, atol=1e-5)
