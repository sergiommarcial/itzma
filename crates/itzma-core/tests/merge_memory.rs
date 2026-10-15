//! A merge holds about one output segment and a read-ahead window per input,
//! not several copies of its input. A file of its own: the counting
//! allocator sees every allocation in the binary, so no other test may run
//! beside this one.

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use itzma_core::{Compaction, CompactionOptions, Document, Metric, Namespace, Schema};
use object_store::memory::InMemory;
use object_store::{ObjectStore, ObjectStoreExt};

/// Counts the bytes allocated right now and the most ever at once.
struct Counting;

static NOW: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            let now = NOW.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        NOW.fetch_sub(layout.size(), Ordering::Relaxed);
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

const DIM: usize = 128;
const SEGMENTS: usize = 8;
const ROWS: usize = 12_500;
const MIB: usize = 1024 * 1024;
/// `compaction::merge::MERGE_WINDOW`, which is crate-private.
const WINDOW: usize = 8 * MIB;

/// xorshift64, so the data is the same on every run.
struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u32 << 23) as f32 - 1.0
    }
}

#[tokio::test]
async fn a_merge_holds_about_one_output_segment() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let ns = Namespace::create(store.clone(), "docs", Schema::new(DIM, Metric::L2))
        .await
        .unwrap();
    let fold = CompactionOptions {
        max_segments: SEGMENTS + 1,
        ..CompactionOptions::default()
    };
    let mut rng = Rng(0x5eed_0037);
    for s in 0..SEGMENTS {
        let docs = (0..ROWS)
            .map(|i| Document {
                id: format!("s{s}-{i:05}"),
                vector: (0..DIM).map(|_| rng.next_f32()).collect(),
                attrs: None,
                sparse: None,
            })
            .collect();
        ns.upsert(docs).await.unwrap();
        ns.compact(&fold).await.unwrap();
    }

    let before = NOW.load(Ordering::SeqCst);
    PEAK.store(before, Ordering::SeqCst);
    let merge = CompactionOptions {
        max_segments: 1,
        ..CompactionOptions::default()
    };
    let segments = match ns.compact(&merge).await.unwrap() {
        Compaction::Published { segments, .. } => segments,
        other => panic!("expected a merge, got {other:?}"),
    };
    let grew = PEAK.load(Ordering::SeqCst) - before;

    assert_eq!(segments.len(), 1);
    let path = itzma_core::segment::segment_path("docs", &segments[0]);
    let output = store.head(&path).await.unwrap().size as usize;
    // The store keeps its own copy of the output, so the merge itself is
    // what grew beyond that.
    let held = grew.saturating_sub(output);
    eprintln!(
        "output {} MiB, peak growth {} MiB, merge held {} MiB",
        output / MIB,
        grew / MIB,
        held / MIB
    );
    assert!(
        held < output + SEGMENTS * WINDOW + 16 * MIB,
        "the merge held {held} bytes for an output of {output}"
    );
}
