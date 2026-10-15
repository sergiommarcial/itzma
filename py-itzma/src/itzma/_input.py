"""Turning what callers pass into Rust batches and vectors."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from itzma import _itzma
from itzma.exceptions import InvalidInput, InvalidVector

VectorLike = Sequence[float] | npt.NDArray[Any]
Row = Mapping[str, Any]


def as_vector(vector: Any) -> Any:
    """A query vector as float32 numpy, a list, or an Arrow array (passed through)."""
    if isinstance(vector, np.ndarray):
        if vector.ndim != 1:
            raise InvalidVector(f"a query vector must be 1-D, got shape {vector.shape}")
        return np.asarray(vector, dtype=np.float32)
    if hasattr(vector, "__arrow_c_array__"):
        return vector
    return list(vector)


def _matrix(vectors: Any) -> npt.NDArray[np.float32]:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise InvalidVector(f"vectors must be 2-D (rows, dim), got shape {matrix.shape}")
    return matrix


def to_batch(
    data: Iterable[Row] | Any | None,
    ids: Iterable[str] | None,
    vectors: Any | None,
    attrs: Iterable[Mapping[str, Any] | None] | None,
) -> _itzma.Batch | _itzma.ArrowStream:
    """Dispatch rows, an Arrow stream, or columns to the matching converter.

    An Arrow table or stream is not read here: the write reads it one record
    batch at a time.
    """
    columnar = ids is not None or vectors is not None
    if data is not None and columnar:
        raise InvalidInput("pass rows or an Arrow table, or ids= and vectors=, not both")
    if columnar:
        if ids is None or vectors is None:
            raise InvalidInput("columnar input needs both ids= and vectors=")
        return _itzma.batch_from_columns(
            list(ids),
            _matrix(vectors),
            None if attrs is None else list(attrs),
        )
    if attrs is not None:
        raise InvalidInput("attrs= goes with ids= and vectors=")
    if data is None:
        raise InvalidInput("nothing to write: pass rows, an Arrow table, or ids= and vectors=")
    if hasattr(data, "__arrow_c_stream__"):
        return _itzma.stream_from_arrow(data)
    return _itzma.batch_from_rows(list(data))
