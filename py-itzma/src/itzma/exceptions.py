"""The errors Itzma raises.

The classes live here, not in the Rust module, so they carry type hints and
docstrings. The Rust side raises them by name and sets the attributes below.
Every attribute has a class-level default, so reading one is always safe.
"""

from __future__ import annotations

from collections.abc import Sequence


class ItzmaError(Exception):
    """Base class of every error Itzma raises.

    ``last_seq`` is set on errors from a write that was split into several
    batches: the sequence number of the last batch that landed, or ``None``
    when none did. Resume from there.

    An Arrow load that stopped part way also sets ``first_seq`` (the first
    chunk that landed, ``None`` when none was acknowledged) and
    ``rows_written``. Resume an unkeyed load by writing its rows from
    ``rows_written``; retry a keyed one with ``retry_from=first_seq`` (or, if
    that is ``None``, a sequence number from before the load), never
    ``last_seq``, which would miss the chunks before it.
    """

    last_seq: int | None = None
    first_seq: int | None = None
    rows_written: int | None = None


class InvalidInput(ItzmaError):
    """The arguments were refused before anything was read or written."""

    namespace: str | None = None


class InvalidVector(InvalidInput):
    """A vector has the wrong length or a component that is not finite."""

    expected: int | None = None
    got: int | None = None


class InvalidAttrs(InvalidInput):
    """Attributes that do not fit the namespace's attribute schema."""

    field: str | None = None
    expected: str | None = None
    found: str | None = None


class InvalidQuery(InvalidInput):
    """Query options or a predicate the namespace cannot run."""


class BatchTooLarge(InvalidInput):
    """One write batch is over the WAL's size limit."""

    bytes: int | None = None
    max: int | None = None


class NamespaceExists(ItzmaError):
    """``create`` found a namespace already there."""

    namespace: str | None = None


class NamespaceNotFound(ItzmaError):
    """``open`` found no namespace."""

    namespace: str | None = None


class ReadBudgetExceeded(ItzmaError):
    """A query or compaction needed more bytes than the handle's read budget."""

    path: str | None = None
    needed: int | None = None
    max: int | None = None


class KeyConflict(ItzmaError):
    """An idempotency key was already used for a different batch."""

    key: str | None = None


class Contention(ItzmaError):
    """Too many writers raced for the same WAL slot or manifest update."""

    attempts: int | None = None


class FormatTooNew(ItzmaError):
    """The namespace was written by a newer Itzma. Upgrade this build."""

    found: int | None = None
    supported: int | None = None


class FormatTooOld(ItzmaError):
    """The operation needs a namespace format level it does not have yet."""

    have: int | None = None
    need: int | None = None


class Corrupt(ItzmaError):
    """Stored data failed a check. ``kind`` says which object."""

    kind: str | None = None
    path: str | None = None
    reason: str | None = None
    keys: Sequence[str] = ()


class StoreError(ItzmaError):
    """The object store failed (network, permissions, an unsupported call)."""


__all__ = [
    "BatchTooLarge",
    "ItzmaError",
    "Contention",
    "Corrupt",
    "FormatTooNew",
    "FormatTooOld",
    "InvalidAttrs",
    "InvalidInput",
    "InvalidQuery",
    "InvalidVector",
    "KeyConflict",
    "NamespaceExists",
    "NamespaceNotFound",
    "ReadBudgetExceeded",
    "StoreError",
]
