"""gs:// stores: the URL, and options that belong to another scheme."""

from __future__ import annotations

import pytest
from conftest import make_gcs_store

import itzma


def test_gs_url_shows_in_repr() -> None:
    store = make_gcs_store()
    assert repr(store).startswith("Store('gs://")


def test_a_gcs_store_that_cannot_connect_is_a_store_error() -> None:
    with pytest.raises(itzma.StoreError):
        itzma.Store("gs://b", endpoint="not a url", control_endpoint="not a url", anonymous=True)


@pytest.mark.parametrize(
    ("url", "options", "message"),
    [
        ("gs://b", {"region": "us-east-1"}, "region only applies to s3:// URLs"),
        ("s3://b", {"anonymous": True}, "anonymous only applies to gs:// URLs"),
        (
            "memory://",
            {"control_endpoint": "http://x"},
            "control_endpoint only applies to gs:// URLs",
        ),
        (
            "memory://",
            {"endpoint": "http://x"},
            "endpoint only applies to s3://, gs:// and az:// URLs",
        ),
    ],
)
def test_options_belong_to_their_scheme(url: str, options: dict[str, object], message: str) -> None:
    with pytest.raises(itzma.InvalidInput, match=message):
        itzma.Store(url, **options)
