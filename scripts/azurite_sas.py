"""Print an account SAS for Azurite's development account.

The key is Azurite's published development key, not a secret. The Rust SDK
authenticates only with Entra ID tokens over HTTPS or with what is in the URL,
so the tests reach Azurite over HTTP with this SAS in the container URL.
"""

import base64
import hashlib
import hmac
import urllib.parse

ACCOUNT = "devstoreaccount1"
KEY = "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw=="
VERSION = "2025-01-05"


def account_sas(account: str = ACCOUNT, key: str = KEY, expiry: str = "2099-01-01T00:00:00Z") -> str:
    """Blob service, service/container/object resources, read/add/create/write/delete/list."""
    fields = {"sv": VERSION, "ss": "b", "srt": "sco", "sp": "racwdl", "se": expiry, "spr": "https,http"}
    # Account SAS string-to-sign for versions 2020-12-06 and later.
    to_sign = "\n".join(
        [account, fields["sp"], fields["ss"], fields["srt"], "", fields["se"], "", fields["spr"], VERSION, "", ""]
    )
    digest = hmac.new(base64.b64decode(key), to_sign.encode("utf-8"), hashlib.sha256).digest()
    fields["sig"] = base64.b64encode(digest).decode("ascii")
    return urllib.parse.urlencode(fields)


if __name__ == "__main__":
    print(account_sas())
