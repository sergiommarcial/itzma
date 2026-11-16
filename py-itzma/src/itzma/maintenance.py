"""A background loop that keeps a namespace compacted and indexed."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from itzma import _itzma
from itzma.reports import MaintenanceReport, parse_maintenance


@dataclass(frozen=True)
class Maintenance:
    """Settings for the loop a handle starts when it opens.

    Every handle on a namespace starts one, and a lease object in the
    namespace lets only one of them work at a time. Each loop checks the
    lease every ``renew``. The one holding it runs ``maintain`` every
    ``every``, the first run a full ``every`` after it took the lease, so a
    short script never pays for one. A leader that stops renewing for
    ``ttl`` is replaced. The rest are the settings ``start_maintenance``
    takes. Pass ``maintenance=None`` to ``open`` or ``create`` to start no
    loop.
    """

    every: timedelta = timedelta(seconds=60)
    index: bool = True
    tail_objects: int = 64
    tail_bytes: int = 64 << 20
    tail_age: timedelta | None = None
    max_segments: int = 8
    max_segment_bytes: int = 128 << 20
    gc: bool = True
    gc_min_age: timedelta = timedelta(hours=24)
    gc_dry_run: bool = False
    fence_wal: bool = False
    renew: timedelta = timedelta(seconds=15)
    ttl: timedelta = timedelta(seconds=60)


class Maintainer:
    """Runs ``maintain`` (and, if asked, ``index`` and ``index_sparse``) every so often.

    Errors are logged to the ``itzma.maintenance`` logger and kept in
    ``last_error``; the loop keeps going. A loop started by ``open`` or
    ``create`` ends instead when the store cannot hold its lease (a
    ``file://`` store, read-only credentials), after logging once. It stops on ``stop()``, on the
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
    def role(self) -> str:
        """``"leader"`` while this loop holds the namespace's lease, else ``"follower"``.

        A loop from ``start_maintenance`` takes no lease and is always the leader.
        """
        return self._inner.role

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
