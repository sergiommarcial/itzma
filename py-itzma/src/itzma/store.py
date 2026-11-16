"""Where namespaces live, and how much of them to keep close."""

from __future__ import annotations

import os
from dataclasses import dataclass

from itzma import _itzma
from itzma.exceptions import InvalidInput


@dataclass(frozen=True)
class Gate:
    """Bound the reads in flight through a :class:`Store`.

    Queries share one gate and maintenance (compaction, index builds, GC and
    the manifest read every call makes) another, smaller one, so a merge
    cannot take the reads queries need. Every namespace opened from the same
    ``Store`` shares both. A read larger than a gate's bytes is let through
    alone. The defaults are starting points, not measurements; a ``Store``
    has no gate unless it is given one.
    """

    max_bytes: int = 64 << 20
    max_requests: int = 256
    maintenance_max_bytes: int = 16 << 20
    maintenance_max_requests: int = 256

    def __post_init__(self) -> None:
        # The core refuses what it cannot use (bytes above 4 GiB, say); this
        # only keeps a value the bridge cannot even pass from surfacing as
        # PyO3's OverflowError instead of InvalidInput.
        for name in (
            "max_bytes",
            "max_requests",
            "maintenance_max_bytes",
            "maintenance_max_requests",
        ):
            value = getattr(self, name)
            if not 1 <= value < 2**64:
                raise InvalidInput(f"Gate.{name} must be at least 1, got {value}")


class Store:
    """An object store: ``s3://bucket[/prefix]``, ``gs://bucket[/prefix]``,
    ``az://container[/prefix]``, ``file:///abs/path`` or ``memory://``.

    For S3, credentials and region come from the AWS SDK's usual chain
    (``AWS_*`` environment variables, ``~/.aws`` profiles, web identity, then
    the container or instance role) unless ``env=False``, and the keywords
    override them; an endpoint may also come from ``AWS_ENDPOINT_URL`` or
    ``AWS_ENDPOINT`` (with ``AWS_ALLOW_HTTP``). The region defaults to
    ``us-east-1`` when neither the keywords, the environment nor a profile
    give one. With ``env=False`` only the keywords count, and with no keys the
    instance role. An ``http://`` endpoint needs ``allow_http=True``. For GCS,
    credentials come from Google's default credentials (Workload Identity on
    GKE and Cloud Run, or ``gcloud auth application-default login``);
    ``endpoint`` and ``control_endpoint`` (the gRPC one) point at an emulator
    and go together, and ``anonymous=True`` sends no credentials. Building a
    ``gs://`` store looks the credentials up, which off Google Cloud can wait
    on the metadata server; a store that cannot be built raises
    ``StoreError``. For Azure, ``account`` names the storage account and
    ``credential`` picks how to sign in: ``"managed_identity"`` (the default),
    ``"workload_identity"``, ``"developer_tools"`` (``az login``) or
    ``"anonymous"``; ``sas_token`` is a credential of its own and goes
    without ``credential``. ``account`` builds the default endpoint
    (``https://<account>.blob.core.windows.net``); with ``endpoint`` set it
    only names the store. Token
    credentials need an ``https`` endpoint, so an emulator such as Azurite
    takes a SAS token. Each keyword is refused on a URL whose scheme doesn't
    use it. A
    ``file://`` store has no conditional update, so ``compact``, ``index`` and
    ``maintain`` fail on it with ``StoreError``. A ``memory://`` store lives in
    this object: namespaces opened through the same ``Store`` share it.
    ``gate`` bounds the reads in flight through this store (see :class:`Gate`);
    there is none by default.
    """

    def __init__(
        self,
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
        gate: Gate | None = None,
    ) -> None:
        self.url = url
        self._store = _itzma.Store(
            url,
            region=region,
            endpoint=endpoint,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            session_token=session_token,
            allow_http=allow_http,
            env=env,
            control_endpoint=control_endpoint,
            anonymous=anonymous,
            account=account,
            credential=credential,
            sas_token=sas_token,
            gate_max_bytes=gate.max_bytes if gate else None,
            gate_max_requests=gate.max_requests if gate else None,
            gate_maintenance_max_bytes=gate.maintenance_max_bytes if gate else None,
            gate_maintenance_max_requests=gate.maintenance_max_requests if gate else None,
        )

    def __repr__(self) -> str:
        return f"Store({self.url!r})"


@dataclass(frozen=True)
class Cache:
    """Keep segment and index bytes in RAM, and optionally on a local disk.

    Only query reads go through the cache; the manifest never does, so a
    write is visible to the next query either way.
    """

    ram_bytes: int = 256 << 20
    nvme_dir: str | os.PathLike[str] | None = None
    nvme_bytes: int | None = None

    def __post_init__(self) -> None:
        if (self.nvme_dir is None) != (self.nvme_bytes is None):
            raise InvalidInput("the NVMe cache needs both nvme_dir and nvme_bytes")


@dataclass(frozen=True)
class FetchPolicy:
    """How ranged reads turn wanted byte ranges into requests.

    The default, :meth:`object_store`, is tuned for S3 and similar stores.
    Pass :meth:`local` for a ``memory://`` or ``file://`` store, where a
    request costs about as much as the bytes it reads.

    Both constructors' six numbers are checked against the core's own
    values by ``test_defaults.py`` (``default_options()``'s ``"fetch"`` and
    ``"fetch_local"`` keys), the same way :class:`Cache`'s default is — a
    mismatch there fails that test, not a comment.
    """

    min_gap: int
    max_gap: int
    gap_divisor: int
    split_above: int
    part_size: int
    align: int

    @classmethod
    def local(cls) -> FetchPolicy:
        """Tuned for ``memory://`` and ``file://``: merge within 8 KiB, never split."""
        return cls(
            min_gap=8 << 10,
            max_gap=64 << 10,
            gap_divisor=8,
            split_above=2**64 - 1,
            part_size=1,
            align=1,
        )

    @classmethod
    def object_store(cls) -> FetchPolicy:
        """The default: tuned for S3 and other real object stores."""
        return cls(
            min_gap=64 << 10,
            max_gap=4 << 20,
            gap_divisor=8,
            split_above=16 << 20,
            part_size=8 << 20,
            align=64 << 10,
        )
