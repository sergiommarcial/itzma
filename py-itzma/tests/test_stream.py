"""Arrow streams load in chunks, in bounded memory."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import textwrap
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
import pytest
from conftest import Model, assert_hits, random_rows, schema

import itzma

Q = [0.1, -0.2, 0.3, 0.05]


def arrow_table(rows: list[dict[str, Any]]) -> pa.Table:
    vectors = np.array([r["vector"] for r in rows], dtype=np.float32)
    attrs = [r["attrs"] for r in rows]
    return pa.table(
        {
            "id": [r["id"] for r in rows],
            "vector": pa.FixedSizeListArray.from_arrays(pa.array(vectors.ravel()), 4),
            **{k: [a[k] for a in attrs] for k in attrs[0]},
        }
    )


def reader(table: pa.Table, rows_per_batch: int) -> pa.RecordBatchReader:
    return pa.RecordBatchReader.from_batches(
        table.schema, table.to_batches(max_chunksize=rows_per_batch)
    )


def test_an_arrow_stream_lands_in_chunks(store: itzma.Store, ns_id: str) -> None:
    rows = random_rows(10_000)
    ns = itzma.Namespace.create(store, ns_id, schema())
    seq = ns.upsert(reader(arrow_table(rows), 200), target_bytes=64 * 1024)
    assert seq > 1
    model = Model()
    model.upsert(rows)
    assert_hits(ns.query(Q, 10, exact=True), model.top_k(Q, 10))


async def test_async_stream(store: itzma.Store, ns_id: str) -> None:
    rows = random_rows(500)
    ns = await itzma.AsyncNamespace.create(store, ns_id, schema())
    await ns.upsert(reader(arrow_table(rows), 50), target_bytes=8 * 1024)
    model = Model()
    model.upsert(rows)
    assert_hits(await ns.query(Q, 10, exact=True), model.top_k(Q, 10))


def test_polars_collect_batches(store: itzma.Store, ns_id: str) -> None:
    rows = random_rows(3_000)
    frame = pl.from_arrow(arrow_table(rows))
    assert isinstance(frame, pl.DataFrame)
    whole = itzma.Namespace.create(store, ns_id + "w", schema())
    whole.upsert(frame)

    lazy = frame.lazy()
    arrow_schema = frame.to_arrow().schema
    batches = (batch for part in lazy.collect_batches() for batch in part.to_arrow().to_batches())
    streamed = itzma.Namespace.create(store, ns_id + "s", schema())
    streamed.upsert(
        pa.RecordBatchReader.from_batches(arrow_schema, batches), target_bytes=16 * 1024
    )
    expected = [(h.id, h.attrs) for h in whole.query(Q, 100)]
    assert [(h.id, h.attrs) for h in streamed.query(Q, 100)] == expected


def test_keyed_stream_retry_lands_once(store: itzma.Store, ns_id: str) -> None:
    rows = random_rows(2_000)
    ns = itzma.Namespace.create(store, ns_id, schema())
    first = ns.upsert(reader(arrow_table(rows), 100), key="load", target_bytes=8 * 1024)
    again = ns.upsert(
        reader(arrow_table(rows), 300), key="load", retry_from=1, target_bytes=8 * 1024
    )
    assert again == first
    # Nothing new was written: the next write lands right after the stream.
    assert ns.upsert(random_rows(1, prefix="extra")) == first + 1


def test_a_failure_mid_stream_reports_progress(ns_id: str) -> None:
    ns = itzma.Namespace.create(
        itzma.Store("memory://"),
        ns_id,
        itzma.Schema(dim=4, metric=itzma.Metric.L2, attrs={"n": itzma.AttrType.NUMBER}),
    )
    n = np.arange(3_000, dtype=np.float64)
    n[1_500] = np.nan
    table = pa.table(
        {
            "id": [f"r{i:05d}" for i in range(3_000)],
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(np.ones(3_000 * 4, dtype=np.float32)), 4
            ),
            "n": n,
        }
    )
    with pytest.raises(itzma.InvalidAttrs) as info:
        ns.upsert(reader(table, 100), target_bytes=8 * 1024)
    written = info.value.rows_written
    assert 0 < written <= 1_500
    assert info.value.last_seq is not None
    landed = ns.filter(itzma.attr("n").ge(0), limit=10_000)
    assert len(landed) == written


async def test_cancelled_stream_leaves_whole_chunks(ns_id: str) -> None:
    ns = await itzma.AsyncNamespace.create(
        itzma.Store("memory://"), ns_id, itzma.Schema(dim=4, metric=itzma.Metric.L2)
    )
    arrow_schema = pa.schema([("id", pa.string()), ("vector", pa.list_(pa.float32(), 4))])

    def slow() -> Iterator[pa.RecordBatch]:
        for b in range(1_000):
            time.sleep(0.01)
            ids = [f"d{b * 10 + i:05d}" for i in range(10)]
            vectors = pa.FixedSizeListArray.from_arrays(pa.array(np.ones(40, dtype=np.float32)), 4)
            yield pa.record_batch([pa.array(ids), vectors], schema=arrow_schema)

    target = 2_048
    # Every row encodes to the same size: a kind byte, a 4-byte length and a
    # 6-byte id, 16 bytes of vector, and a 4-byte empty attribute entry.
    # A chunk is 12 bytes of counts and as many rows as fit.
    per_chunk = (target - 12) // (1 + 4 + 6 + 16 + 4)
    task = asyncio.ensure_future(
        ns.upsert(pa.RecordBatchReader.from_batches(arrow_schema, slow()), target_bytes=target)
    )
    await asyncio.sleep(0.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    found = len(await ns.query([1.0, 1.0, 1.0, 1.0], 10_000))
    assert found % per_chunk == 0, (found, per_chunk)
    # The handle still writes after the cancelled load.
    await ns.upsert([{"id": "after", "vector": [2.0, 2.0, 2.0, 2.0]}])
    assert len(await ns.query([1.0, 1.0, 1.0, 1.0], 10_000)) == found + 1


@pytest.mark.skipif(sys.platform == "win32", reason="resource is POSIX only")
def test_memory_stays_bounded(tmp_path: Path) -> None:
    script = textwrap.dedent(
        """
        import resource, sys
        import numpy as np, pyarrow as pa, itzma

        dim, rows, batches = 128, 1_000, 200  # about 100 MB of vectors
        schema = pa.schema([("id", pa.string()), ("vector", pa.list_(pa.float32(), dim))])

        def gen():
            rng = np.random.default_rng(0)
            for b in range(batches):
                v = rng.standard_normal((rows, dim), dtype=np.float32)
                yield pa.record_batch(
                    [
                        pa.array([f"b{b}-{i}" for i in range(rows)]),
                        pa.FixedSizeListArray.from_arrays(pa.array(v.ravel()), dim),
                    ],
                    schema=schema,
                )

        ns = itzma.Namespace.create(
            itzma.Store(sys.argv[1]), "big", itzma.Schema(dim=dim, metric=itzma.Metric.L2)
        )
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ns.upsert(pa.RecordBatchReader.from_batches(schema, gen()), target_bytes=1 << 20)
        after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Bytes on macOS, KiB on Linux.
        print((after - before) * (1 if sys.platform == "darwin" else 1024))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", script, f"file://{tmp_path}"],
        capture_output=True,
        text=True,
        check=True,
    )
    grew = int(out.stdout.strip())
    assert grew < 64 * 1024 * 1024, f"the load grew the peak RSS by {grew} bytes"


def test_a_keyed_stream_resumes_from_first_seq(ns_id: str) -> None:
    ns = itzma.Namespace.create(
        itzma.Store("memory://"),
        ns_id,
        itzma.Schema(dim=4, metric=itzma.Metric.L2, attrs={"n": itzma.AttrType.NUMBER}),
    )
    # Another write first, so the stream does not start at seq 1.
    ns.upsert([{"id": "before", "vector": [1.0, 1.0, 1.0, 1.0]}])

    def table(bad: bool) -> pa.Table:
        n = np.arange(3_000, dtype=np.float64)
        if bad:
            n[1_500] = np.nan
        return pa.table(
            {
                "id": [f"r{i:05d}" for i in range(3_000)],
                "vector": pa.FixedSizeListArray.from_arrays(
                    pa.array(np.ones(3_000 * 4, dtype=np.float32)), 4
                ),
                "n": n,
            }
        )

    with pytest.raises(itzma.InvalidAttrs) as info:
        ns.upsert(reader(table(True), 100), key="load", target_bytes=8 * 1024)
    assert info.value.first_seq == 2
    last = ns.upsert(
        reader(table(False), 250),
        key="load",
        retry_from=info.value.first_seq,
        target_bytes=8 * 1024,
    )
    # Every chunk is in the log once: the next write lands right after.
    assert ns.upsert([{"id": "after", "vector": [1.0, 1.0, 1.0, 1.0]}]) == last + 1
    assert len(ns.filter(itzma.attr("n").ge(0), limit=10_000)) == 3_000


@pytest.mark.parametrize("options", [{"target_bytes": 0}, {"depth": 0}])
def test_bad_stream_options_are_refused_for_every_input(
    ns_id: str, options: dict[str, int]
) -> None:
    ns = itzma.Namespace.create(itzma.Store("memory://"), ns_id, schema())
    rows = random_rows(5)
    with pytest.raises(itzma.InvalidInput):
        ns.upsert(rows, **options)
    with pytest.raises(itzma.InvalidInput):
        ns.upsert(reader(arrow_table(rows), 5), **options)
    assert len(ns.query(Q, 10)) == 0
