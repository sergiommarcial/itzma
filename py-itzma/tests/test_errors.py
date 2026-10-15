import itzma
from itzma import _itzma, exceptions


def test_every_class_the_rust_side_raises_exists() -> None:
    for name in _itzma.exception_names():
        cls = getattr(exceptions, name)
        assert issubclass(cls, itzma.ItzmaError), name


def test_hierarchy() -> None:
    for cls in (itzma.InvalidVector, itzma.InvalidAttrs, itzma.InvalidQuery, itzma.BatchTooLarge):
        assert issubclass(cls, itzma.InvalidInput)
    assert issubclass(itzma.InvalidInput, itzma.ItzmaError)
    assert issubclass(itzma.StoreError, itzma.ItzmaError)


def test_fields_default_to_none() -> None:
    e = itzma.ReadBudgetExceeded("x")
    assert (e.path, e.needed, e.max, e.last_seq) == (None, None, None, None)
