"""An account SAS for Azurite's development account, as scripts/azurite_sas.py
signs it. The key is Azurite's published development key, not a secret."""

from __future__ import annotations

import base64
import hashlib
import hmac
import urllib.parse

ACCOUNT = "devstoreaccount1"
KEY = "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw=="
VERSION = "2025-01-05"


def account_sas(expiry: str = "2099-01-01T00:00:00Z") -> str:
    fields = {
        "sv": VERSION,
        "ss": "b",
        "srt": "sco",
        "sp": "racwdl",
        "se": expiry,
        "spr": "https,http",
    }
    parts = [ACCOUNT, "racwdl", "b", "sco", "", expiry, "", "https,http", VERSION, "", ""]
    to_sign = "\n".join(parts)
    digest = hmac.new(base64.b64decode(KEY), to_sign.encode("utf-8"), hashlib.sha256).digest()
    fields["sig"] = base64.b64encode(digest).decode("ascii")
    return urllib.parse.urlencode(fields)
