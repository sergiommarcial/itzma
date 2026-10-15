"""Conditions on attributes: in a vector query, and on their own with filter()."""

import itzma
from itzma import attr

store = itzma.Store("memory://")
schema = itzma.Schema(
    dim=2,
    metric=itzma.Metric.L2,
    attrs={
        "tag": itzma.AttrType.STRING,
        "year": itzma.AttrType.NUMBER,
        "draft": itzma.AttrType.BOOL,
        "labels": itzma.AttrType.SET_STRING,
    },
)
ns = itzma.Namespace.create(store, "articles", schema)

ns.upsert(
    [
        {
            "id": "a1",
            "vector": [0, 0],
            "attrs": {"tag": "news", "year": 2024, "draft": False, "labels": ["eu"]},
        },
        {
            "id": "a2",
            "vector": [1, 0],
            "attrs": {"tag": "news", "year": 2019, "draft": True, "labels": []},
        },
        {
            "id": "a3",
            "vector": [0, 1],
            "attrs": {"tag": "blog", "year": 2023, "labels": ["eu", "tech"]},
        },
        {"id": "a4", "vector": [1, 1], "attrs": {"year": 2022}},
        {"id": "a5", "vector": [2, 2], "attrs": {"tag": "devlog", "year": 2025, "draft": False}},
    ]
)

conditions = {
    "recent news": (attr("tag") == "news") & (attr("year") >= 2022),
    "news or blog": attr("tag").isin(["news", "blog"]),
    "starts with dev": attr("tag").starts_with("dev"),
    "2020 to 2023": attr("year").between(2020, 2023),
    "not a draft": attr("draft") == False,  # noqa: E712
    "no tag": attr("tag") == None,  # noqa: E711
    # A comparison with a missing field is false and ~ flips it, so this
    # also matches a4, which has no tag at all.
    "anything but news": ~(attr("tag") == "news"),
    "labelled eu": attr("labels").contains("eu"),
    "labelled eu and tech": attr("labels").contains_all(["eu", "tech"]),
    "no labels": attr("labels").is_empty(),
}
for name, where in conditions.items():
    print(f"{name:22} {[row.id for row in ns.filter(where)]}")

# The same conditions narrow a vector query.
hits = ns.query([0, 0], top_k=2, filter=attr("year") >= 2023)
print("nearest from 2023 on:", hits.ids)

# filter() pages in id order; pass `next` back as `after`.
page = ns.filter(attr("year") > 0, limit=2)
while True:
    print("page:", [row.id for row in page])
    if page.next is None:
        break
    page = ns.filter(attr("year") > 0, limit=2, after=page.next)
