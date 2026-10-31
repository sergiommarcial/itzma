"""Run against an installed wheel: import it and do one write and one query."""

import itzma

store = itzma.Store("memory://")
schema = itzma.Schema(dim=2, metric=itzma.Metric.L2, attrs={"tag": itzma.AttrType.STRING})
ns = itzma.Namespace.create(store, "smoke", schema)
ns.upsert([{"id": "a", "vector": [0, 1], "attrs": {"tag": "x"}}, {"id": "b", "vector": [1, 0]}])
hits = ns.query([0, 0.9], top_k=1)
assert hits[0].id == "a", hits
print("itzma", itzma.__version__, "ok")
