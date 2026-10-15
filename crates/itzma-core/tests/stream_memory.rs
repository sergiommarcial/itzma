//! `upsert_stream` holds a bounded amount of memory, however long its input.
//! A file of its own: the counting allocator sees every
//! allocation in the binary, so no other test may run beside this one.

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use futures::stream::{self, BoxStream, StreamExt};
use itzma_core::{DocColumns, Metric, Namespace, Schema, StreamOptions};
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult,
};

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

/// A store that keeps every object empty, so what it holds does not count
/// against the load: put-if-absent still works, which is all a single
/// writer's appends need.
#[derive(Debug)]
struct Discard(object_store::memory::InMemory);

impl std::fmt::Display for Discard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Discard")
    }
}

#[async_trait]
impl ObjectStore for Discard {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        let keep = location.as_ref().contains("/wal/");
        let payload = if keep { PutPayload::new() } else { payload };
        self.0.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.0.put_multipart_opts(location, opts).await
    }

    async fn get_opts(
        &self,
        location: &Path,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        self.0.get_opts(location, options).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        self.0.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.0.list(prefix)
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.0.list_with_offset(prefix, offset)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> object_store::Result<ListResult> {
        self.0.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &Path,
        to: &Path,
        options: CopyOptions,
    ) -> object_store::Result<()> {
        self.0.copy_opts(from, to, options).await
    }
}

const DIM: usize = 128;
const ROWS: usize = 500;
const TARGET: usize = 1 << 20;
const DEPTH: usize = 2;

/// Batch `b`: 500 rows of 128 floats (about 260 KB), built only when the
/// stream is polled.
fn batch(b: usize) -> DocColumns {
    let (mut ids, mut offsets) = (Vec::new(), vec![0u32]);
    for i in 0..ROWS {
        ids.extend_from_slice(format!("b{b}-{i}").as_bytes());
        offsets.push(ids.len() as u32);
    }
    let vectors = (0..ROWS * DIM)
        .map(|i| ((b * ROWS * DIM + i) % 1_000) as f32 / 999.0 - 0.5)
        .collect();
    DocColumns::new(ids, offsets, vectors, DIM).unwrap()
}

/// What the stream may hold at once: the chunk being encoded and its body
/// and ZSTD frame (each about `TARGET`), `DEPTH` encoded chunks queued, the
/// one being written and the input batch being cut, with room for
/// allocator slack. A load that read its whole input first would hold all
/// of it, more than five times this.
const BOUND: usize = (DEPTH + 4) * TARGET * 3;

async fn namespace() -> Namespace {
    let store: Arc<dyn ObjectStore> = Arc::new(Discard(object_store::memory::InMemory::new()));
    Namespace::create(store, "docs", Schema::new(DIM, Metric::L2))
        .await
        .unwrap()
}

/// Both cases in one test: the allocator counts the whole binary, so they
/// must not run side by side.
#[tokio::test]
async fn memory_stays_bounded() {
    many_small_batches().await;
    one_batch_many_times_the_target().await;
}

async fn many_small_batches() {
    let ns = namespace().await;
    // About 50 times DEPTH * TARGET of input.
    let batches = 50 * DEPTH * TARGET / (ROWS * DIM * 4) + 1;
    let input = stream::iter(0..batches).map(|b| Ok(batch(b))).boxed();
    let options = StreamOptions::default().target_bytes(TARGET).depth(DEPTH);

    let before = NOW.load(Ordering::SeqCst);
    PEAK.store(before, Ordering::SeqCst);
    let report = ns.upsert_stream(input, &options).await.unwrap();
    let grew = PEAK.load(Ordering::SeqCst) - before;

    assert_eq!(report.rows as usize, batches * ROWS);
    assert!(
        grew < BOUND,
        "the load held {grew} bytes at once, over the bound of {BOUND}"
    );
    let total = batches * ROWS * DIM * 4;
    assert!(
        total > 5 * BOUND,
        "input of {total} bytes too small to tell"
    );
}

/// One input batch 20 times `TARGET` (an in-memory DataFrame handed over
/// whole). Cutting it must copy each row about once and hold about the
/// batch twice at most, not a copy of the rest of the batch per chunk.
async fn one_batch_many_times_the_target() {
    let ns = namespace().await;
    let rows = 20 * TARGET / (DIM * 4);
    let (mut ids, mut offsets) = (Vec::new(), vec![0u32]);
    for i in 0..rows {
        ids.extend_from_slice(format!("big-{i}").as_bytes());
        offsets.push(ids.len() as u32);
    }
    let vectors = (0..rows * DIM)
        .map(|i| (i % 1_000) as f32 / 999.0)
        .collect();
    let big = DocColumns::new(ids, offsets, vectors, DIM).unwrap();
    let size = rows * DIM * 4;
    let options = StreamOptions::default().target_bytes(TARGET).depth(DEPTH);

    let before = NOW.load(Ordering::SeqCst);
    PEAK.store(before, Ordering::SeqCst);
    let report = ns
        .upsert_stream(stream::iter([Ok(big)]), &options)
        .await
        .unwrap();
    let grew = PEAK.load(Ordering::SeqCst) - before;

    assert_eq!(report.rows as usize, rows);
    assert!(
        grew < 2 * size + BOUND,
        "cutting one {size}-byte batch held {grew} bytes at once"
    );
}
