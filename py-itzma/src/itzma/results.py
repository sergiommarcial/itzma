"""Query results."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, overload

import numpy as np
import numpy.typing as npt

from itzma import _itzma


@dataclass(frozen=True)
class Hit:
    """One result: the document's id, its distance (lower is closer) and attributes."""

    id: str
    distance: float
    attrs: dict[str, Any] | None


class Hits(Sequence[Hit]):
    """The results of a query, closest first."""

    def __init__(self, data: _itzma.Hits) -> None:
        self._data = data
        self._hits = [Hit(id, distance, attrs) for id, distance, attrs in data.rows()]

    @overload
    def __getitem__(self, index: int) -> Hit: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[Hit]: ...
    def __getitem__(self, index: int | slice) -> Hit | Sequence[Hit]:
        return self._hits[index]

    def __len__(self) -> int:
        return len(self._hits)

    def __iter__(self) -> Iterator[Hit]:
        return iter(self._hits)

    def __repr__(self) -> str:
        return f"Hits({self._hits!r})"

    @property
    def ids(self) -> list[str]:
        return [h.id for h in self._hits]

    @property
    def distances(self) -> npt.NDArray[np.float32]:
        return self._data.distances()

    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object:
        return self._data.to_arrow().__arrow_c_stream__(requested_schema)

    def to_arrow(self) -> Any:
        """A ``pyarrow.Table``: id, distance and one typed column per attribute field.

        Needs pyarrow.
        """
        import pyarrow as pa

        return pa.table(self._data.to_arrow())

    def to_polars(self) -> Any:
        """A ``polars.DataFrame``: id, distance and one typed column per attribute field.

        Needs polars.
        """
        import polars as pl

        # Not ``pl.DataFrame(self)``: Polars reads a Sequence as rows before it
        # looks for ``__arrow_c_stream__``, which loses the typed layout.
        return pl.DataFrame(self._data.to_arrow())


@dataclass(frozen=True)
class Row:
    """One document from ``filter``: its id and attributes."""

    id: str
    attrs: dict[str, Any] | None


class FilterPage(Sequence[Row]):
    """One page of ``filter`` results in id order.

    Pass ``next`` as ``after`` to get the next page; it is ``None`` when the
    scan ran out of documents.
    """

    def __init__(self, data: _itzma.Rows) -> None:
        self._data = data
        self._rows = [Row(id, attrs) for id, attrs in data.rows()]
        self.next: str | None = data.next

    @overload
    def __getitem__(self, index: int) -> Row: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[Row]: ...
    def __getitem__(self, index: int | slice) -> Row | Sequence[Row]:
        return self._rows[index]

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[Row]:
        return iter(self._rows)

    def __repr__(self) -> str:
        return f"FilterPage({self._rows!r}, next={self.next!r})"

    @property
    def rows(self) -> list[Row]:
        return list(self._rows)

    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object:
        return self._data.to_arrow().__arrow_c_stream__(requested_schema)

    def to_arrow(self) -> Any:
        """A ``pyarrow.Table``: id and one typed column per attribute field.

        Needs pyarrow.
        """
        import pyarrow as pa

        return pa.table(self._data.to_arrow())

    def to_polars(self) -> Any:
        """A ``polars.DataFrame``: id and one typed column per attribute field.

        Needs polars.
        """
        import polars as pl

        # Not ``pl.DataFrame(self)``: Polars reads a Sequence as rows before it
        # looks for ``__arrow_c_stream__``, which loses the typed layout.
        return pl.DataFrame(self._data.to_arrow())
