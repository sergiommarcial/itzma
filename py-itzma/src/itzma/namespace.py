"""Namespace handles: ``Namespace`` for blocking code, ``AsyncNamespace`` for asyncio.

Both wrap the same Rust object. Open each namespace once per process (per
worker, if the server forks) and share the handle across requests.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import timedelta
from types import TracebackType
from typing import Any, Literal

from itzma import _input, _itzma
from itzma.exceptions import InvalidInput, InvalidQuery, ItzmaError
from itzma.maintenance import Maintainer
from itzma.predicate import Expr
from itzma.reports import (
    Compaction,
    GcReport,
    IndexReport,
    MaintenanceReport,
    parse_maintenance,
)
from itzma.results import FilterPage, Hits
from itzma.schema import Metric, Schema
from itzma.store import Cache, FetchPolicy, Store


def _open_kwargs(
    cache: Cache | None,
    read_budget: int | None,
    fetch: FetchPolicy | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"read_budget": read_budget}
    if cache is not None:
        kwargs["cache_ram_bytes"] = cache.ram_bytes
        kwargs["cache_nvme_dir"] = None if cache.nvme_dir is None else str(cache.nvme_dir)
        kwargs["cache_nvme_bytes"] = cache.nvme_bytes
    if fetch is not None:
        kwargs["fetch_min_gap"] = fetch.min_gap
        kwargs["fetch_max_gap"] = fetch.max_gap
        kwargs["fetch_gap_divisor"] = fetch.gap_divisor
        kwargs["fetch_split_above"] = fetch.split_above
        kwargs["fetch_part_size"] = fetch.part_size
        kwargs["fetch_align"] = fetch.align
    return kwargs


def _ids(ids: Iterable[str]) -> list[str]:
    if isinstance(ids, str):
        raise InvalidInput("delete takes a list of ids, not one string")
    return list(ids)


def _top_k(top_k: int) -> int:
    if top_k < 0:
        raise InvalidQuery(f"top_k must be 0 or more, got {top_k}")
    return top_k


def _query_options(filter: Expr | None, rerank: int | None, nprobe: int | None, exact: bool) -> str:
    return json.dumps(
        {
            "rerank": rerank,
            "nprobe": nprobe,
            "exact": exact,
            "filter": None if filter is None else filter.to_dict(),
        }
    )


def _sparse_options(filter: Expr | None) -> str:
    return json.dumps({"filter": None if filter is None else filter.to_dict()})


def _filter_options(limit: int, after: str | None) -> str:
    return json.dumps({"limit": limit, "after": after})


def _index_options(kind: str, min_rows: int, nlist: int | None, rebuild: bool) -> str:
    return json.dumps({"kind": kind, "min_rows": min_rows, "nlist": nlist, "rebuild": rebuild})


def _compact_options(max_segments: int, max_segment_bytes: int) -> str:
    return json.dumps({"max_segments": max_segments, "max_segment_bytes": max_segment_bytes})


def _gc_options(min_age: timedelta, dry_run: bool, fence_wal: bool) -> str:
    return json.dumps(
        {"min_age_secs": min_age.total_seconds(), "dry_run": dry_run, "fence_wal": fence_wal}
    )


def _maintain_options(
    tail_objects: int,
    tail_bytes: int,
    tail_age: timedelta | None,
    max_segments: int,
    max_segment_bytes: int,
    gc: bool,
    gc_min_age: timedelta,
    gc_dry_run: bool,
    fence_wal: bool,
) -> str:
    return json.dumps(
        {
            "tail_objects": tail_objects,
            "tail_bytes": tail_bytes,
            "tail_age_secs": None if tail_age is None else tail_age.total_seconds(),
            "max_segments": max_segments,
            "max_segment_bytes": max_segment_bytes,
            "gc": (
                {
                    "min_age_secs": gc_min_age.total_seconds(),
                    "dry_run": gc_dry_run,
                    "fence_wal": fence_wal,
                }
                if gc
                else None
            ),
        }
    )


class _Handle:
    """What the sync and async classes share: the Rust object and its lifecycle."""

    def __init__(self, ns: _itzma.Namespace) -> None:
        self._ns = ns
        self._closed = False
        self._maintainers: list[Maintainer] = []

    def _handle(self) -> _itzma.Namespace:
        if self._closed:
            raise ItzmaError(f"namespace {self._ns.id!r} is closed")
        return self._ns

    @property
    def id(self) -> str:
        return self._ns.id

    @property
    def schema(self) -> Schema:
        return Schema._from_json(self._handle().schema_json())

    def close(self) -> None:
        """Stop any background maintenance and refuse further calls."""
        for m in self._maintainers:
            m.stop()
        self._maintainers.clear()
        self._closed = True

    def start_maintenance(
        self,
        every: timedelta = timedelta(seconds=60),
        *,
        index: bool = True,
        tail_objects: int = 64,
        tail_bytes: int = 64 << 20,
        tail_age: timedelta | None = None,
        max_segments: int = 8,
        max_segment_bytes: int = 128 << 20,
        gc: bool = True,
        gc_min_age: timedelta = timedelta(hours=24),
        gc_dry_run: bool = False,
        fence_wal: bool = False,
    ) -> Maintainer:
        """Start a background loop running ``maintain`` every ``every``.

        With ``index=True`` each run also builds indexes for new segments.
        The first run starts right away. Nothing runs unless you call this.
        """
        options = _maintain_options(
            tail_objects,
            tail_bytes,
            tail_age,
            max_segments,
            max_segment_bytes,
            gc,
            gc_min_age,
            gc_dry_run,
            fence_wal,
        )
        inner = self._handle().start_maintenance(
            every.total_seconds(),
            options,
            "{}" if index else None,
        )
        maintainer = Maintainer(inner)
        self._maintainers.append(maintainer)
        return maintainer

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._ns.id!r})"


class Namespace(_Handle):
    """A namespace for blocking code. Every call releases the GIL while it waits."""

    @classmethod
    def create(
        cls,
        store: Store,
        id: str,
        schema: Schema,
        *,
        cache: Cache | None = None,
        read_budget: int | None = None,
        fetch: FetchPolicy | None = None,
    ) -> Namespace:
        """Create a namespace. Raises ``NamespaceExists`` if one is there."""
        return cls(
            _itzma.Namespace.create(
                store._store,
                id,
                schema.dim,
                Metric(schema.metric).value,
                schema._attr_names(),
                **_open_kwargs(cache, read_budget, fetch),
            )
        )

    @classmethod
    def open(
        cls,
        store: Store,
        id: str,
        *,
        cache: Cache | None = None,
        read_budget: int | None = None,
        fetch: FetchPolicy | None = None,
    ) -> Namespace:
        """Open an existing namespace. Raises ``NamespaceNotFound`` if there is none."""
        return cls(
            _itzma.Namespace.open(store._store, id, **_open_kwargs(cache, read_budget, fetch))
        )

    def upsert(
        self,
        data: Iterable[_input.Row] | Any | None = None,
        /,
        *,
        ids: Iterable[str] | None = None,
        vectors: Any | None = None,
        attrs: Iterable[Mapping[str, Any] | None] | None = None,
        sparse: Iterable[_input.SparseLike | None] | None = None,
        key: str | None = None,
        retry_from: int | None = None,
        target_bytes: int = 33554432,
        depth: int = 2,
    ) -> int:
        """Write documents; returns the WAL sequence number of the last batch.

        Pass rows (dicts with ``id``, ``vector``, optional ``attrs`` and
        ``sparse``), an Arrow table or stream (pyarrow, Polars, DuckDB), or
        columns with ``ids=`` and ``vectors=`` (a 2-D array). A big unkeyed
        write is split into batches; if one fails, the error's ``last_seq``
        says how far it got. ``key`` makes the write idempotent;
        ``retry_from`` looks for an earlier attempt from that sequence number.

        An Arrow table or stream is read one record batch at a time and
        written in chunks of about ``target_bytes``, with at most ``depth``
        encoded chunks waiting, so a load of any size runs in bounded memory.
        A failure then also says ``first_seq`` and ``rows_written``: resume
        an unkeyed load by writing its rows from ``rows_written``. With
        ``key``, chunk ``n`` is written under ``"<key>#<n>"`` (so the key is
        at most 107 bytes), and a retry with ``retry_from=e.first_seq`` (not
        ``last_seq``) writes only the chunks that are missing. ``target_bytes`` and
        ``depth`` apply to Arrow input only.
        """
        batch = _input.to_batch(data, ids, vectors, attrs, sparse)
        return self._handle().upsert(
            batch, key=key, retry_from=retry_from, target_bytes=target_bytes, depth=depth
        )

    def delete(
        self, ids: Iterable[str], *, key: str | None = None, retry_from: int | None = None
    ) -> int:
        """Delete documents by id; returns the WAL sequence number."""
        return self._handle().delete(_ids(ids), key=key, retry_from=retry_from)

    def query(
        self,
        vector: _input.VectorLike | Any,
        top_k: int = 10,
        *,
        filter: Expr | None = None,
        rerank: int | None = 4,
        nprobe: int | None = None,
        exact: bool = False,
    ) -> Hits:
        """The ``top_k`` nearest documents to ``vector``, closest first.

        ``filter`` keeps only documents matching a condition built with
        ``itzma.attr``. ``rerank`` is how many candidates an index proposes
        per hit wanted (``None`` proposes every row, so the answer equals a
        flat scan). ``nprobe`` is how many index partitions to read (``None``
        lets the index choose). ``exact=True`` ignores indexes.
        """
        options = _query_options(filter, rerank, nprobe, exact)
        return Hits(self._handle().query(_input.as_vector(vector), _top_k(top_k), options))

    def query_sparse(
        self, terms: _input.SparseLike, top_k: int = 10, *, filter: Expr | None = None
    ) -> Hits:
        """The ``top_k`` documents with the highest dot product against ``terms``.

        ``terms`` is a list of ``(term id, weight)``. ``Hit.distance`` is the
        negated score, so closest first still means best first.
        """
        return Hits(
            self._handle().query_sparse(list(terms), _top_k(top_k), _sparse_options(filter))
        )

    def filter(self, where: Expr, *, limit: int = 100, after: str | None = None) -> FilterPage:
        """Documents matching ``where``, in id order, ``limit`` at a time."""
        return FilterPage(self._handle().filter(where.to_json(), _filter_options(limit, after)))

    def index(
        self,
        *,
        kind: Literal["sq8", "hnsw"] = "hnsw",
        min_rows: int = 4096,
        nlist: int | None = None,
        rebuild: bool = False,
    ) -> IndexReport:
        """Build and publish an index for each live segment that has none."""
        options = _index_options(kind, min_rows, nlist, rebuild)
        return IndexReport(**json.loads(self._handle().index(options)))

    def compact(self, *, max_segments: int = 8, max_segment_bytes: int = 128 << 20) -> Compaction:
        """Fold the WAL tail into a segment, merging when there are too many.

        A merge reads each group under the handle's read budget, so it raises
        ``InvalidInput`` when ``max_segment_bytes`` is above that budget. A
        fold does not.
        """
        raw = self._handle().compact(_compact_options(max_segments, max_segment_bytes))
        return Compaction._from(json.loads(raw))

    def gc(
        self,
        *,
        min_age: timedelta = timedelta(hours=24),
        dry_run: bool = False,
        fence_wal: bool = False,
    ) -> GcReport:
        """Delete dead segments at least ``min_age`` old; optionally fence old WAL objects."""
        raw = self._handle().gc(_gc_options(min_age, dry_run, fence_wal))
        return GcReport(**json.loads(raw))

    def maintain(
        self,
        *,
        tail_objects: int = 64,
        tail_bytes: int = 64 << 20,
        tail_age: timedelta | None = None,
        max_segments: int = 8,
        max_segment_bytes: int = 128 << 20,
        gc: bool = True,
        gc_min_age: timedelta = timedelta(hours=24),
        gc_dry_run: bool = False,
        fence_wal: bool = False,
    ) -> MaintenanceReport:
        """Compact if a trigger is crossed, then collect. The one call to schedule."""
        options = _maintain_options(
            tail_objects,
            tail_bytes,
            tail_age,
            max_segments,
            max_segment_bytes,
            gc,
            gc_min_age,
            gc_dry_run,
            fence_wal,
        )
        return parse_maintenance(self._handle().maintain(options))

    def __enter__(self) -> Namespace:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class AsyncNamespace(_Handle):
    """A namespace for asyncio code. Every call is awaitable and never blocks the loop."""

    @classmethod
    async def create(
        cls,
        store: Store,
        id: str,
        schema: Schema,
        *,
        cache: Cache | None = None,
        read_budget: int | None = None,
        fetch: FetchPolicy | None = None,
    ) -> AsyncNamespace:
        """Create a namespace. Raises ``NamespaceExists`` if one is there."""
        ns = await _itzma.Namespace.create_async(
            store._store,
            id,
            schema.dim,
            Metric(schema.metric).value,
            schema._attr_names(),
            **_open_kwargs(cache, read_budget, fetch),
        )
        return cls(ns)

    @classmethod
    async def open(
        cls,
        store: Store,
        id: str,
        *,
        cache: Cache | None = None,
        read_budget: int | None = None,
        fetch: FetchPolicy | None = None,
    ) -> AsyncNamespace:
        """Open an existing namespace. Raises ``NamespaceNotFound`` if there is none."""
        ns = await _itzma.Namespace.open_async(
            store._store, id, **_open_kwargs(cache, read_budget, fetch)
        )
        return cls(ns)

    async def upsert(
        self,
        data: Iterable[_input.Row] | Any | None = None,
        /,
        *,
        ids: Iterable[str] | None = None,
        vectors: Any | None = None,
        attrs: Iterable[Mapping[str, Any] | None] | None = None,
        sparse: Iterable[_input.SparseLike | None] | None = None,
        key: str | None = None,
        retry_from: int | None = None,
        target_bytes: int = 33554432,
        depth: int = 2,
    ) -> int:
        """Write documents; see ``Namespace.upsert``."""
        batch = _input.to_batch(data, ids, vectors, attrs, sparse)
        return await self._handle().upsert_async(
            batch, key=key, retry_from=retry_from, target_bytes=target_bytes, depth=depth
        )

    async def delete(
        self, ids: Iterable[str], *, key: str | None = None, retry_from: int | None = None
    ) -> int:
        """Delete documents by id; returns the WAL sequence number."""
        return await self._handle().delete_async(_ids(ids), key=key, retry_from=retry_from)

    async def query(
        self,
        vector: _input.VectorLike | Any,
        top_k: int = 10,
        *,
        filter: Expr | None = None,
        rerank: int | None = 4,
        nprobe: int | None = None,
        exact: bool = False,
    ) -> Hits:
        """The ``top_k`` nearest documents to ``vector``; see ``Namespace.query``."""
        options = _query_options(filter, rerank, nprobe, exact)
        return Hits(
            await self._handle().query_async(_input.as_vector(vector), _top_k(top_k), options)
        )

    async def query_sparse(
        self, terms: _input.SparseLike, top_k: int = 10, *, filter: Expr | None = None
    ) -> Hits:
        """Sparse top ``top_k``; see ``Namespace.query_sparse``."""
        data = await self._handle().query_sparse_async(
            list(terms), _top_k(top_k), _sparse_options(filter)
        )
        return Hits(data)

    async def filter(
        self, where: Expr, *, limit: int = 100, after: str | None = None
    ) -> FilterPage:
        """Documents matching ``where``, in id order, ``limit`` at a time."""
        data = await self._handle().filter_async(where.to_json(), _filter_options(limit, after))
        return FilterPage(data)

    async def index(
        self,
        *,
        kind: Literal["sq8", "hnsw"] = "hnsw",
        min_rows: int = 4096,
        nlist: int | None = None,
        rebuild: bool = False,
    ) -> IndexReport:
        """Build and publish an index for each live segment that has none."""
        options = _index_options(kind, min_rows, nlist, rebuild)
        return IndexReport(**json.loads(await self._handle().index_async(options)))

    async def compact(
        self, *, max_segments: int = 8, max_segment_bytes: int = 128 << 20
    ) -> Compaction:
        """Fold the WAL tail into a segment, merging when there are too many.

        A merge reads each group under the handle's read budget, so it raises
        ``InvalidInput`` when ``max_segment_bytes`` is above that budget. A
        fold does not.
        """
        raw = await self._handle().compact_async(_compact_options(max_segments, max_segment_bytes))
        return Compaction._from(json.loads(raw))

    async def gc(
        self,
        *,
        min_age: timedelta = timedelta(hours=24),
        dry_run: bool = False,
        fence_wal: bool = False,
    ) -> GcReport:
        """Delete dead segments at least ``min_age`` old; optionally fence old WAL objects."""
        raw = await self._handle().gc_async(_gc_options(min_age, dry_run, fence_wal))
        return GcReport(**json.loads(raw))

    async def maintain(
        self,
        *,
        tail_objects: int = 64,
        tail_bytes: int = 64 << 20,
        tail_age: timedelta | None = None,
        max_segments: int = 8,
        max_segment_bytes: int = 128 << 20,
        gc: bool = True,
        gc_min_age: timedelta = timedelta(hours=24),
        gc_dry_run: bool = False,
        fence_wal: bool = False,
    ) -> MaintenanceReport:
        """Compact if a trigger is crossed, then collect. The one call to schedule."""
        options = _maintain_options(
            tail_objects,
            tail_bytes,
            tail_age,
            max_segments,
            max_segment_bytes,
            gc,
            gc_min_age,
            gc_dry_run,
            fence_wal,
        )
        return parse_maintenance(await self._handle().maintain_async(options))

    async def __aenter__(self) -> AsyncNamespace:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
