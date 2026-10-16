"""A background loop that keeps a namespace compacted and indexed."""

from __future__ import annotations

from itzma import _itzma
from itzma.reports import MaintenanceReport, parse_maintenance


class Maintainer:
    """Runs ``maintain`` (and, if asked, ``index`` and ``index_sparse``) every so often.

    Errors are logged to the ``itzma.maintenance`` logger and kept in
    ``last_error``; the loop keeps going. It stops on ``stop()``, on the
    namespace's ``close()`` or ``with`` exit, and when the namespace is
    garbage collected.
    """

    def __init__(self, inner: _itzma.Maintainer) -> None:
        self._inner = inner

    def stop(self) -> None:
        self._inner.stop()

    @property
    def running(self) -> bool:
        return self._inner.running

    @property
    def runs(self) -> int:
        return self._inner.runs

    @property
    def errors(self) -> int:
        return self._inner.errors

    @property
    def last_error(self) -> str | None:
        return self._inner.last_error

    @property
    def last_report(self) -> MaintenanceReport | None:
        raw = self._inner.last_report
        return None if raw is None else parse_maintenance(raw)

    def __repr__(self) -> str:
        return f"Maintainer(running={self.running}, runs={self.runs}, errors={self.errors})"
