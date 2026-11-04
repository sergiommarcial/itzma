"""az:// stores: the URL, and options that belong to another scheme."""

from __future__ import annotations

import pytest
from conftest import make_azure_store

import itzma


def test_az_url_shows_in_repr() -> None:
    store = make_azure_store()
    assert repr(store).startswith("Store('az://")


@pytest.mark.parametrize(
    ("url", "options", "message"),
    [
        ("az://c", {}, "an az:// store needs account"),
        ("s3://b", {"account": "a"}, "account only applies to az:// URLs"),
        ("gs://b", {"sas_token": "sv=x"}, "sas_token only applies to az:// URLs"),
        ("az://c", {"account": "a", "credential": "password"}, "expected managed_identity"),
        (
            "az://c",
            {"account": "a", "credential": "anonymous", "sas_token": "sv=x"},
            "sas_token is a credential of its own",
        ),
    ],
)
def test_options_belong_to_their_scheme(url: str, options: dict[str, object], message: str) -> None:
    with pytest.raises(itzma.InvalidInput, match=message):
        itzma.Store(url, **options)
