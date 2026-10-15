"""Create a namespace, write two documents, and run a filtered query."""

import itzma
from itzma import attr

store = itzma.Store("memory://")  # or "file:///var/lib/itzma", "s3://bucket/prefix"
schema = itzma.Schema(
    dim=4,
    metric=itzma.Metric.COSINE,
    attrs={"tag": itzma.AttrType.STRING, "year": itzma.AttrType.NUMBER},
)
ns = itzma.Namespace.create(store, "docs", schema)

ns.upsert(
    [
        {"id": "a", "vector": [1, 0, 0, 0], "attrs": {"tag": "news", "year": 2024}},
        {"id": "b", "vector": [0, 1, 0, 0], "attrs": {"tag": "blog", "year": 2021}},
    ]
)

hits = ns.query([0.9, 0.1, 0, 0], top_k=5, filter=attr("year") >= 2022)
for hit in hits:
    print(hit.id, round(hit.distance, 4), hit.attrs)

# A later process opens it by id; the schema comes back with it.
again = itzma.Namespace.open(store, "docs")
print(again.schema)
