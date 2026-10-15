from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from conftest import ATTRS, schema

import itzma


def test_create_then_open(store: itzma.Store, ns_id: str) -> None:
    created = itzma.Namespace.create(store, ns_id, schema())
    opened = itzma.Namespace.open(store, ns_id)
    assert created.id == opened.id == ns_id
    assert opened.schema == itzma.Schema(dim=4, metric=itzma.Metric.L2, attrs=ATTRS)


async def test_async_create_then_open(store: itzma.Store, ns_id: str) -> None:
    await itzma.AsyncNamespace.create(store, ns_id, schema(8, itzma.Metric.COSINE))
    ns = await itzma.AsyncNamespace.open(store, ns_id)
    assert ns.schema.dim == 8
    assert ns.schema.metric is itzma.Metric.COSINE


def test_create_twice_raises_namespace_exists(store: itzma.Store, ns_id: str) -> None:
    itzma.Namespace.create(store, ns_id, schema())
    with pytest.raises(itzma.NamespaceExists) as info:
        itzma.Namespace.create(store, ns_id, schema())
    assert info.value.namespace == ns_id


def test_open_missing_raises_not_found(store: itzma.Store, ns_id: str) -> None:
    with pytest.raises(itzma.NamespaceNotFound) as info:
        itzma.Namespace.open(store, ns_id)
    assert info.value.namespace == ns_id


def test_bad_namespace_id_writes_nothing(tmp_path: Path) -> None:
    store = itzma.Store(f"file://{tmp_path}")
    with pytest.raises(itzma.InvalidInput):
        itzma.Namespace.create(store, "Not Valid", schema())
    assert list(tmp_path.iterdir()) == []


def test_no_attrs_means_typed_and_empty(store: itzma.Store, ns_id: str) -> None:
    ns = itzma.Namespace.create(store, ns_id, itzma.Schema(dim=2, metric=itzma.Metric.DOT))
    assert ns.schema.attrs == {}


@pytest.mark.parametrize(
    "url",
    ["memory", "file://relative/path", "s3://", "gs://bucket", ""],
)
def test_bad_urls(url: str) -> None:
    with pytest.raises(itzma.InvalidInput):
        itzma.Store(url)


def test_s3_options_on_a_memory_store_are_refused() -> None:
    with pytest.raises(itzma.InvalidInput, match="s3://"):
        itzma.Store("memory://", region="us-east-1")


def test_store_repr_hides_credentials() -> None:
    store = itzma.Store(
        "s3://bucket/p",
        region="us-east-1",
        access_key_id="AKIA",
        secret_access_key="hunter2",
        env=False,
    )
    assert "hunter2" not in repr(store) and "AKIA" not in repr(store)


def test_cache_options(tmp_path: Path, ns_id: str) -> None:
    store = itzma.Store("memory://")
    itzma.Namespace.create(store, ns_id, schema())
    cache = itzma.Cache(ram_bytes=1 << 20, nvme_dir=tmp_path / "nvme", nvme_bytes=1 << 20)
    itzma.Namespace.open(store, ns_id, cache=cache, read_budget=1 << 20)
    with pytest.raises(itzma.InvalidInput, match="both"):
        itzma.Cache(nvme_dir=tmp_path)


def test_fetch_policy_is_accepted_and_defaults_to_none(ns_id: str) -> None:
    store = itzma.Store("memory://")
    ns = itzma.Namespace.create(store, ns_id, schema(), fetch=itzma.FetchPolicy.local())
    ns.upsert([{"id": "a", "vector": [0.0, 0.0, 0.0, 0.0]}])
    assert ns.query([0.0, 0.0, 0.0, 0.0], 1)[0].id == "a"

    # No fetch= at all must still work (the default on the Rust side).
    ns2 = itzma.Namespace.open(store, ns_id)
    assert ns2.query([0.0, 0.0, 0.0, 0.0], 1)[0].id == "a"


def test_closed_handle_refuses_calls(ns_id: str) -> None:
    store = itzma.Store("memory://")
    with itzma.Namespace.create(store, ns_id, schema()) as ns:
        assert ns.schema.dim == 4
    with pytest.raises(itzma.ItzmaError, match="closed"):
        _ = ns.schema


def test_configure_after_start_is_refused(ns_id: str) -> None:
    itzma.Namespace.create(itzma.Store("memory://"), ns_id, schema())
    with pytest.raises(itzma.ItzmaError, match="already running"):
        itzma.configure(worker_threads=2)


def test_configure_before_start_works() -> None:
    code = (
        "import itzma\n"
        "itzma.configure(worker_threads=2)\n"
        "s = itzma.Store('memory://')\n"
        "itzma.Namespace.create(s, 'a', itzma.Schema(dim=2, metric=itzma.Metric.L2))\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "ok"


# 20,000 rows at dimension 64: an exact query over them is big enough to start
# the compute pools.
_BIG_ROWS = (
    "import itzma, random\n"
    "random.seed(1)\n"
    "rows = [{'id': f'd{i:05d}', 'vector': [random.random() for _ in range(64)]}"
    " for i in range(20000)]\n"
    "ns = itzma.Namespace.create(itzma.Store('memory://'), 'a',"
    " itzma.Schema(dim=64, metric=itzma.Metric.L2))\n"
    "ns.upsert(rows)\n"
)


def test_configure_cpu_threads_before_the_pools_start_works() -> None:
    code = (
        "import itzma\n"
        "itzma.configure(cpu_threads=2, maintenance_cpu_threads=1)\n"
        + _BIG_ROWS
        + "assert len(ns.query([0.5] * 64, 5, exact=True).ids) == 5\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "ok"


def test_configure_cpu_threads_after_the_pools_start_is_refused() -> None:
    code = (
        _BIG_ROWS + "ns.query([0.5] * 64, 5, exact=True)\n"
        "try:\n"
        "    itzma.configure(cpu_threads=2)\n"
        "except itzma.ItzmaError as e:\n"
        "    print('refused' if 'already running' in str(e) else e)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "refused"
