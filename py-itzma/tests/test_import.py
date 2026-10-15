import pytest

import itzma


def test_version_matches_the_crate() -> None:
    assert itzma.__version__ == "0.1.0"


def test_configure_refuses_zero_workers() -> None:
    with pytest.raises(itzma.ItzmaError, match="at least 1"):
        itzma.configure(worker_threads=0)


def test_configure_refuses_zero_cpu_threads() -> None:
    with pytest.raises(itzma.ItzmaError, match="at least one thread"):
        itzma.configure(cpu_threads=0)
    with pytest.raises(itzma.ItzmaError, match="at least one thread"):
        itzma.configure(maintenance_cpu_threads=0)
