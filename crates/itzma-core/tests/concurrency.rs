mod common;

use futures::TryStreamExt;
use itzma_core::wal;
use itzma_core::{Metric, Namespace};
use object_store::ObjectStore;

use common::{backends, doc, schema};

/// N writers, each with its own handle (so each starts with a stale seq
/// hint), all appending to one namespace at once. Put-if-absent must hand
/// every batch a distinct seq, with no gaps and nothing lost.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_writers_get_unique_contiguous_seqs() {
    const WRITERS: usize = 8;
    const APPENDS_EACH: usize = 5;

    for b in backends() {
        eprintln!("backend: {}", b.name);
        Namespace::create(b.store.clone(), "docs", schema(2, Metric::L2))
            .await
            .unwrap();

        let mut tasks = Vec::new();
        for w in 0..WRITERS {
            let store = b.store.clone();
            tasks.push(tokio::spawn(async move {
                let ns = Namespace::open(store, "docs").await.unwrap();
                let mut seqs = Vec::new();
                for i in 0..APPENDS_EACH {
                    let id = format!("w{w}-{i}");
                    seqs.push(
                        ns.upsert(vec![doc(&id, &[w as f32, i as f32])])
                            .await
                            .unwrap(),
                    );
                }
                seqs
            }));
        }
        let mut all_seqs = Vec::new();
        for t in tasks {
            all_seqs.extend(t.await.unwrap());
        }

        let total = (WRITERS * APPENDS_EACH) as u64;
        all_seqs.sort_unstable();
        let expected: Vec<u64> = (1..=total).collect();
        assert_eq!(all_seqs, expected, "seqs must be unique and contiguous");

        // The objects on storage agree with what the writers were told.
        let prefix = wal::wal_prefix("docs");
        let listed: Vec<_> = b.store.list(Some(&prefix)).try_collect().await.unwrap();
        assert_eq!(listed.len() as u64, total);

        // And a reader sees every document.
        let reader = Namespace::open(b.store.clone(), "docs").await.unwrap();
        let hits = reader.query(&[0.0, 0.0], 1000).await.unwrap();
        assert_eq!(hits.len() as u64, total);
    }
}
