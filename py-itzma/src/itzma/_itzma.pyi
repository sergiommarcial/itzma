from collections.abc import Awaitable
from typing import Any, final

import numpy as np
import numpy.typing as npt

__version__: str

def configure(
    *,
    worker_threads: int | None = ...,
    cpu_threads: int | None = ...,
    maintenance_cpu_threads: int | None = ...,
) -> None: ...
def exception_names() -> list[str]: ...
def default_options() -> str: ...
def batch_from_rows(rows: list[Any]) -> Batch: ...
def stream_from_arrow(data: Any) -> ArrowStream: ...
def batch_from_columns(
    ids: list[str],
    vectors: npt.NDArray[np.float32],
    attrs: list[Any] | None = None,
    sparse: list[Any] | None = None,
) -> Batch: ...

@final
class Batch:
    def __len__(self) -> int: ...

@final
class ArrowStream: ...

@final
class Hits:
    def __len__(self) -> int: ...
    def ids(self) -> list[str]: ...
    def distances(self) -> npt.NDArray[np.float32]: ...
    def rows(self) -> list[tuple[str, float, Any]]: ...
    def to_arrow(self) -> Any: ...

@final
class Maintainer:
    def stop(self) -> None: ...
    @property
    def running(self) -> bool: ...
    @property
    def runs(self) -> int: ...
    @property
    def errors(self) -> int: ...
    @property
    def last_error(self) -> str | None: ...
    @property
    def last_report(self) -> str | None: ...

@final
class Rows:
    @property
    def next(self) -> str | None: ...
    def __len__(self) -> int: ...
    def rows(self) -> list[tuple[str, Any]]: ...
    def to_arrow(self) -> Any: ...

@final
class Store:
    def __new__(
        cls,
        url: str,
        *,
        region: str | None = None,
        endpoint: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        session_token: str | None = None,
        allow_http: bool | None = None,
        env: bool = True,
        control_endpoint: str | None = None,
        anonymous: bool | None = None,
        account: str | None = None,
        credential: str | None = None,
        sas_token: str | None = None,
    ) -> Store: ...

@final
class Namespace:
    @staticmethod
    def create(
        store: Store,
        id: str,
        dim: int,
        metric: str,
        attrs: dict[str, str],
        *,
        read_budget: int | None = None,
        cache_ram_bytes: int | None = None,
        cache_nvme_dir: str | None = None,
        cache_nvme_bytes: int | None = None,
        fetch_min_gap: int | None = None,
        fetch_max_gap: int | None = None,
        fetch_gap_divisor: int | None = None,
        fetch_split_above: int | None = None,
        fetch_part_size: int | None = None,
        fetch_align: int | None = None,
    ) -> Namespace: ...
    @staticmethod
    def create_async(
        store: Store,
        id: str,
        dim: int,
        metric: str,
        attrs: dict[str, str],
        *,
        read_budget: int | None = None,
        cache_ram_bytes: int | None = None,
        cache_nvme_dir: str | None = None,
        cache_nvme_bytes: int | None = None,
        fetch_min_gap: int | None = None,
        fetch_max_gap: int | None = None,
        fetch_gap_divisor: int | None = None,
        fetch_split_above: int | None = None,
        fetch_part_size: int | None = None,
        fetch_align: int | None = None,
    ) -> Awaitable[Namespace]: ...
    @staticmethod
    def open(
        store: Store,
        id: str,
        *,
        read_budget: int | None = None,
        cache_ram_bytes: int | None = None,
        cache_nvme_dir: str | None = None,
        cache_nvme_bytes: int | None = None,
        fetch_min_gap: int | None = None,
        fetch_max_gap: int | None = None,
        fetch_gap_divisor: int | None = None,
        fetch_split_above: int | None = None,
        fetch_part_size: int | None = None,
        fetch_align: int | None = None,
    ) -> Namespace: ...
    @staticmethod
    def open_async(
        store: Store,
        id: str,
        *,
        read_budget: int | None = None,
        cache_ram_bytes: int | None = None,
        cache_nvme_dir: str | None = None,
        cache_nvme_bytes: int | None = None,
        fetch_min_gap: int | None = None,
        fetch_max_gap: int | None = None,
        fetch_gap_divisor: int | None = None,
        fetch_split_above: int | None = None,
        fetch_part_size: int | None = None,
        fetch_align: int | None = None,
    ) -> Awaitable[Namespace]: ...
    @property
    def id(self) -> str: ...
    def schema_json(self) -> str: ...
    def upsert(
        self,
        batch: Batch | ArrowStream,
        *,
        key: str | None = None,
        retry_from: int | None = None,
        target_bytes: int | None = None,
        depth: int | None = None,
    ) -> int: ...
    def upsert_async(
        self,
        batch: Batch | ArrowStream,
        *,
        key: str | None = None,
        retry_from: int | None = None,
        target_bytes: int | None = None,
        depth: int | None = None,
    ) -> Awaitable[int]: ...
    def delete(
        self, ids: list[str], *, key: str | None = None, retry_from: int | None = None
    ) -> int: ...
    def delete_async(
        self, ids: list[str], *, key: str | None = None, retry_from: int | None = None
    ) -> Awaitable[int]: ...
    def query(self, vector: Any, top_k: int, options: str) -> Hits: ...
    def query_async(self, vector: Any, top_k: int, options: str) -> Awaitable[Hits]: ...
    def query_sparse(self, terms: Any, top_k: int, options: str) -> Hits: ...
    def query_sparse_async(self, terms: Any, top_k: int, options: str) -> Awaitable[Hits]: ...
    def filter(self, predicate: str, options: str) -> Rows: ...
    def filter_async(self, predicate: str, options: str) -> Awaitable[Rows]: ...
    def index(self, options: str) -> str: ...
    def index_async(self, options: str) -> Awaitable[str]: ...
    def index_sparse(self, options: str) -> str: ...
    def index_sparse_async(self, options: str) -> Awaitable[str]: ...
    def compact(self, options: str) -> str: ...
    def compact_async(self, options: str) -> Awaitable[str]: ...
    def gc(self, options: str) -> str: ...
    def gc_async(self, options: str) -> Awaitable[str]: ...
    def maintain(self, options: str) -> str: ...
    def maintain_async(self, options: str) -> Awaitable[str]: ...
    def start_maintenance(
        self,
        every_secs: float,
        maintain: str,
        index: str | None = None,
        index_sparse: str | None = None,
    ) -> Maintainer: ...
