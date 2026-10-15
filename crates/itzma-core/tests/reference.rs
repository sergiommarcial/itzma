mod common;

use itzma_core::{Document, Metric, Namespace};

use common::{Lcg, doc, naive_top_k, schema};

#[tokio::test]
async fn query_matches_naive_reference_for_every_metric() {
    const DIM: usize = 8;
    for metric in [Metric::Cosine, Metric::L2, Metric::Dot] {
        let mut rng = Lcg(42);
        let docs: Vec<Document> = (0..200)
            .map(|i| doc(&format!("d{i:03}"), &rng.vector(DIM)))
            .collect();

        let store = common::memory();
        let ns = Namespace::create(store, "docs", schema(DIM, metric))
            .await
            .unwrap();
        // Spread the corpus over several WAL objects.
        for chunk in docs.chunks(50) {
            ns.upsert(chunk.to_vec()).await.unwrap();
        }

        for _ in 0..10 {
            let q = rng.vector(DIM);
            for k in [1, 5, 37] {
                let got: Vec<String> = ns
                    .query(&q, k)
                    .await
                    .unwrap()
                    .into_iter()
                    .map(|h| h.id)
                    .collect();
                assert_eq!(got, naive_top_k(metric, &q, &docs, k), "{metric:?} k={k}");
            }
        }
    }
}
