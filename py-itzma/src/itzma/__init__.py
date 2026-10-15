"""Itzma: vector retrieval on object storage, embedded in your Python service."""

from itzma._itzma import __version__, configure
from itzma.exceptions import (
    BatchTooLarge,
    Contention,
    Corrupt,
    FormatTooNew,
    FormatTooOld,
    InvalidAttrs,
    InvalidInput,
    InvalidQuery,
    InvalidVector,
    ItzmaError,
    KeyConflict,
    NamespaceExists,
    NamespaceNotFound,
    ReadBudgetExceeded,
    StoreError,
)
from itzma.maintenance import Maintainer
from itzma.namespace import AsyncNamespace, Namespace
from itzma.predicate import Attr, Expr, attr
from itzma.reports import (
    Compaction,
    GcReport,
    IndexReport,
    MaintenanceReport,
)
from itzma.results import FilterPage, Hit, Hits, Row
from itzma.schema import AttrType, Metric, Schema
from itzma.store import Cache, FetchPolicy, Store

__all__ = [
    "AsyncNamespace",
    "Attr",
    "AttrType",
    "Cache",
    "Compaction",
    "Expr",
    "FetchPolicy",
    "FilterPage",
    "GcReport",
    "Hit",
    "Hits",
    "IndexReport",
    "Maintainer",
    "MaintenanceReport",
    "Metric",
    "Namespace",
    "Row",
    "Schema",
    "Store",
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
    "__version__",
    "attr",
    "configure",
]
