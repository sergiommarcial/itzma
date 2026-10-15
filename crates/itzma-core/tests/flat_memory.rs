//! A flat query over an unindexed version 2 segment allocates a handful of
//! buffers, not one per document. A file of its own: the counting
//! allocator sees every allocation in the binary, so no other test may run
//! beside this one.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use itzma_core::{
    CompactionOptions, DecodedCache, DecodedCacheOptions, Document, Metric, Namespace,
    QueryOptions, Schema,
};
use object_store::ObjectStore;
use object_store::memory::InMemory;

/// Counts allocations.
struct Counting;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

const DIM: usize = 32;
const ROWS: usize = 20_000;

#[tokio::test]
async fn a_flat_query_does_not_allocate_per_document() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let ns = Namespace::create(store, "docs", Schema::new(DIM, Metric::L2))
        .await
        .unwrap()
        .with_decoded_cache(DecodedCache::new(DecodedCacheOptions::default()));
    let mut x = 0x5eed_u32;
    let docs: Vec<Document> = (0..ROWS)
        .map(|i| Document {
            id: format!("d{i:06}"),
            vector: (0..DIM)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 17;
                    x ^= x << 5;
                    x as f32 / u32::MAX as f32
                })
                .collect(),
            attrs: None,
        })
        .collect();
    ns.upsert(docs).await.unwrap();
    ns.compact(&CompactionOptions::default()).await.unwrap();
    let query = vec![0.5; DIM];
    let options = QueryOptions::default();

    // Cold: the decode is in here.
    let before = ALLOCS.load(Ordering::Relaxed);
    ns.query_with(&query, 10, &options).await.unwrap();
    let cold = ALLOCS.load(Ordering::Relaxed) - before;

    // Warm: the view is cached; what is left is the query's own buffers.
    let before = ALLOCS.load(Ordering::Relaxed);
    ns.query_with(&query, 10, &options).await.unwrap();
    let warm = ALLOCS.load(Ordering::Relaxed) - before;

    eprintln!("allocations: cold {cold}, warm {warm}, rows {ROWS}");
    assert!(
        warm < 500,
        "a warm flat query made {warm} allocations for {ROWS} rows"
    );
    assert!(
        cold < ROWS / 4,
        "a cold flat query made {cold} allocations for {ROWS} rows"
    );
}
