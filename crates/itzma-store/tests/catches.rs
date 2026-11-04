//! Each check against a store that breaks exactly the item it pins, so a
//! check that stops catching its fault fails here instead of passing every
//! adapter.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use itzma_store::conformance;
use object_store::memory::InMemory;
use object_store::path::Path;
use object_store::{
    CopyMode, CopyOptions, GetOptions, GetRange, GetResult, GetResultPayload, ListResult,
    MultipartUpload, ObjectMeta, ObjectStore, ObjectStoreExt, PutMode, PutMultipartOptions,
    PutOptions, PutPayload, PutResult, Result,
};

#[derive(Debug, Clone, Copy, PartialEq)]
enum Fault {
    /// `get` metadata carries no version, as an adapter that only fills it on `head`.
    GetHidesVersion,
    /// An update with no version overwrites instead of failing.
    VersionlessUpdateOverwrites,
    /// An update compares the version, then writes: not atomic.
    RacyUpdate,
    /// A ranged read of a missing object is a generic error, not `NotFound`.
    RangedMissingIsGeneric,
    /// A suffix longer than the object fails instead of returning all of it.
    LongSuffixFails,
    /// A range starting past the end reads as `NotFound`.
    PastEndIsNotFound,
    /// `delete` reports success and keeps the object.
    DeleteKeeps,
    /// `list` matches the prefix as a raw string, so `a` also finds `ab/1`.
    RawStringPrefix,
    /// A plain copy onto an existing key leaves it unchanged.
    CopyKeepsTarget,
    /// An empty body stream panics when polled after its end, as an unfused
    /// `unfold` over an SDK response does.
    UnfusedEmptyBody,
}

#[derive(Debug)]
struct Faulty {
    inner: InMemory,
    fault: Fault,
}

impl fmt::Display for Faulty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Faulty({:?})", self.fault)
    }
}

fn faulty(fault: Fault) -> Arc<dyn ObjectStore> {
    Arc::new(Faulty {
        inner: InMemory::new(),
        fault,
    })
}

fn generic(path: &Path) -> object_store::Error {
    object_store::Error::Generic {
        store: "Faulty",
        source: format!("injected fault at {path}").into(),
    }
}

#[async_trait]
impl ObjectStore for Faulty {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> Result<PutResult> {
        if let PutMode::Update(version) = &opts.mode {
            let versionless = version.e_tag.is_none() && version.version.is_none();
            if self.fault == Fault::VersionlessUpdateOverwrites && versionless {
                return self.inner.put(location, payload).await;
            }
            if self.fault == Fault::RacyUpdate {
                let current = self.inner.head(location).await?;
                if current.e_tag != version.e_tag {
                    return Err(object_store::Error::Precondition {
                        path: location.to_string(),
                        source: "stale".into(),
                    });
                }
                tokio::task::yield_now().await;
                return self.inner.put(location, payload).await;
            }
        }
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> Result<GetResult> {
        let ranged = options.range.is_some();
        match (self.fault, &options.range) {
            (Fault::LongSuffixFails, Some(GetRange::Suffix(n))) => {
                if *n > self.inner.head(location).await?.size {
                    return Err(generic(location));
                }
            }
            (Fault::PastEndIsNotFound, Some(range)) => {
                let size = self.inner.head(location).await?.size;
                let start = match range {
                    GetRange::Bounded(r) => r.start,
                    GetRange::Offset(o) => *o,
                    GetRange::Suffix(_) => 0,
                };
                if start >= size {
                    return Err(object_store::Error::NotFound {
                        path: location.to_string(),
                        source: "past end".into(),
                    });
                }
            }
            _ => {}
        }
        let head = options.head;
        let result = self.inner.get_opts(location, options).await;
        match (self.fault, result) {
            (Fault::RangedMissingIsGeneric, Err(object_store::Error::NotFound { .. }))
                if ranged =>
            {
                Err(generic(location))
            }
            (Fault::UnfusedEmptyBody, Ok(mut got)) if got.meta.size == 0 => {
                let body = futures::stream::unfold((), |()| async { None });
                got.payload = GetResultPayload::Stream(body.boxed());
                Ok(got)
            }
            (Fault::GetHidesVersion, Ok(mut got)) if !head => {
                got.meta.e_tag = None;
                got.meta.version = None;
                Ok(got)
            }
            (_, result) => result,
        }
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, Result<Path>>,
    ) -> BoxStream<'static, Result<Path>> {
        if self.fault == Fault::DeleteKeeps {
            return locations;
        }
        self.inner.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, Result<ObjectMeta>> {
        if self.fault == Fault::RawStringPrefix {
            let raw = prefix.map(|p| p.to_string()).unwrap_or_default();
            return self
                .inner
                .list(None)
                .try_filter(move |m| futures::future::ready(m.location.as_ref().starts_with(&raw)))
                .boxed();
        }
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> Result<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(&self, from: &Path, to: &Path, options: CopyOptions) -> Result<()> {
        if self.fault == Fault::CopyKeepsTarget
            && matches!(options.mode, CopyMode::Overwrite)
            && self.inner.head(to).await.is_ok()
        {
            return Ok(());
        }
        self.inner.copy_opts(from, to, options).await
    }
}

#[tokio::test(flavor = "multi_thread")]
#[should_panic(expected = "update with the version get reported")]
async fn catches_get_hiding_version() {
    conformance::update_with_get_version(faulty(Fault::GetHidesVersion)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[should_panic(expected = "update with no version must fail")]
async fn catches_versionless_update() {
    conformance::update_without_version_refused(faulty(Fault::VersionlessUpdateOverwrites)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[should_panic(expected = "update race must have one winner")]
async fn catches_racy_update() {
    conformance::update_one_winner(faulty(Fault::RacyUpdate)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[should_panic(expected = "ranged read of a missing key must be NotFound")]
async fn catches_ranged_missing_as_generic() {
    conformance::missing_object(faulty(Fault::RangedMissingIsGeneric)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[should_panic(expected = "suffix longer than the object")]
async fn catches_long_suffix_failing() {
    conformance::suffix_longer_than_object(faulty(Fault::LongSuffixFails)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[should_panic(expected = "must fail, not read as NotFound")]
async fn catches_past_end_as_not_found() {
    conformance::range_past_end(faulty(Fault::PastEndIsNotFound)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[should_panic(expected = "deleted key must be gone")]
async fn catches_delete_keeping_object() {
    conformance::delete_removes(faulty(Fault::DeleteKeeps)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[should_panic(expected = "list under a/ with sizes")]
async fn catches_raw_string_prefix() {
    conformance::list_under_prefix(faulty(Fault::RawStringPrefix)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[should_panic(expected = "copy onto an existing key must replace it")]
async fn catches_copy_keeping_target() {
    conformance::copy_overwrites(faulty(Fault::CopyKeepsTarget)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[should_panic(expected = "an empty object's body must read as empty, not panic")]
async fn catches_unfused_empty_body() {
    conformance::empty_object_reads_empty(faulty(Fault::UnfusedEmptyBody)).await;
}
