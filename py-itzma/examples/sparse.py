"""Sparse (term id, weight) vectors, such as BM25 or SPLADE output, beside dense ones."""

import itzma

store = itzma.Store("memory://")
ns = itzma.Namespace.create(store, "search", itzma.Schema(dim=2, metric=itzma.Metric.DOT, attrs={}))

# Term ids come from your own vocabulary; here 1 = "rust", 2 = "python", 3 = "vector".
ns.upsert(
    [
        {"id": "d1", "vector": [1, 0], "sparse": [(1, 0.9), (3, 0.4)]},
        {"id": "d2", "vector": [0, 1], "sparse": [(2, 1.2)]},
        {"id": "d3", "vector": [1, 1], "sparse": [(2, 0.3), (3, 1.1)]},
        {"id": "d4", "vector": [0, 0]},  # sparse is optional per document
    ]
)

hits = ns.query_sparse([(3, 1.0), (2, 0.5)], top_k=3)
for hit in hits:
    # distance is the negated dot product, so best first still means lowest.
    print(hit.id, round(-hit.distance, 3))

# Fold the log into a segment; answers stay the same.
ns.compact()
print(ns.query_sparse([(3, 1.0), (2, 0.5)], top_k=3).ids)
