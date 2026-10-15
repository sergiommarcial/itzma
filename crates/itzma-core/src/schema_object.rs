//! The schema object: `<namespace>/schema.json`.
//!
//! A namespace's schema, dimension and metric, is also written to an object of
//! its own, once, with put-if-absent. Nothing overwrites or deletes it. The
//! manifest still carries the schema and stays what the namespace runs on,
//! but the manifest is the one object that changes, so it is the one a bad
//! restore or a stray write can lose. This copy cannot be lost that way.
//!
//! Only [`Namespace::create`](crate::Namespace::create) writes it and only
//! [`Namespace::open`](crate::Namespace::open) reads it. Queries, writes,
//! compaction and GC never look at it.

use bytes::Bytes;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt, PutMode, PutOptions};
use serde::{Deserialize, Serialize};

use crate::attrs::AttrSchema;
use crate::document::{Metric, Schema};
use crate::error::{ItzmaError, Result};

/// Version stamped into a typed schema object. A reader that finds a version
/// above it fails with [`ItzmaError::UnsupportedFormat`]. Format 1, from before
/// attribute types, stays readable and means a free-form namespace.
pub const FORMAT_VERSION: u32 = 2;

/// Object key of a namespace's schema: `<namespace>/schema.json`.
///
/// Does not validate `namespace`; that is [`Namespace`](crate::Namespace)'s job.
pub fn schema_path(namespace: &str) -> Path {
    Path::from(format!("{namespace}/schema.json"))
}

/// The JSON form:
/// `{"format_version": 2, "dim": 3, "metric": "cosine", "attrs": {"price": "number"}}`.
#[derive(Serialize, Deserialize)]
struct Stored {
    format_version: u32,
    dim: usize,
    metric: Metric,
    #[serde(
        default = "AttrSchema::free_form",
        skip_serializing_if = "AttrSchema::is_free_form"
    )]
    attrs: AttrSchema,
}

fn encode(schema: &Schema) -> Result<Bytes> {
    let stored = Stored {
        format_version: FORMAT_VERSION,
        dim: schema.dim,
        metric: schema.metric,
        attrs: schema.attrs.clone(),
    };
    serde_json::to_vec(&stored)
        .map(Bytes::from)
        .map_err(|source| ItzmaError::Serde {
            path: "schema".into(),
            source,
        })
}

/// Read the schema object, or `None` if there is not one yet.
///
/// # Errors
///
/// - [`ItzmaError::Serde`] if the object is not valid schema JSON. The error
///   names the object key.
/// - [`ItzmaError::UnsupportedFormat`] if `format_version` is not
///   [`FORMAT_VERSION`].
/// - [`ItzmaError::Storage`] for any backend failure.
pub(crate) async fn read(store: &dyn ObjectStore, namespace: &str) -> Result<Option<Schema>> {
    let path = schema_path(namespace);
    let result = match store.get(&path).await {
        Ok(r) => r,
        Err(object_store::Error::NotFound { .. }) => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    let bytes = result.bytes().await?;
    let stored: Stored = serde_json::from_slice(&bytes).map_err(|source| ItzmaError::Serde {
        path: path.to_string(),
        source,
    })?;
    match (stored.format_version, stored.attrs.is_free_form()) {
        (FORMAT_VERSION, false) => {}
        (FORMAT_VERSION, true) => {
            return Err(ItzmaError::CorruptManifest(format!(
                "{path}: format {FORMAT_VERSION} needs an attrs field"
            )));
        }
        (version, _) => return Err(ItzmaError::UnsupportedFormat { version }),
    }
    Ok(Some(Schema {
        dim: stored.dim,
        metric: stored.metric,
        attrs: stored.attrs,
    }))
}

/// Fail with [`ItzmaError::CorruptManifest`], showing both, unless the schema
/// object holds what the caller expects.
pub(crate) fn check(namespace: &str, found: &Schema, expected: &Schema) -> Result<()> {
    if found == expected {
        return Ok(());
    }
    Err(ItzmaError::CorruptManifest(format!(
        "namespace {namespace}: schema.json holds {found:?} but {expected:?} was expected"
    )))
}

/// Write the schema object with put-if-absent. If one is already there, it
/// has to hold the same schema, which is what a retried create or a racing
/// healer leaves behind.
///
/// # Errors
///
/// - [`ItzmaError::CorruptManifest`] if an object with a different schema is
///   there.
/// - Anything [`read`] can return, and [`ItzmaError::Storage`] for a failed
///   put.
pub(crate) async fn write(store: &dyn ObjectStore, namespace: &str, schema: &Schema) -> Result<()> {
    let opts = PutOptions::from(PutMode::Create);
    match store
        .put_opts(&schema_path(namespace), encode(schema)?.into(), opts)
        .await
    {
        Ok(_) => Ok(()),
        Err(object_store::Error::AlreadyExists { .. }) => match read(store, namespace).await? {
            Some(found) => check(namespace, &found, schema),
            None => Err(ItzmaError::CorruptManifest(format!(
                "namespace {namespace}: schema.json disappeared while it was being written"
            ))),
        },
        Err(e) => Err(e.into()),
    }
}

/// Make sure the schema object exists and holds `expected`, writing it from
/// `expected` if it is missing. `open` calls this with the manifest's schema,
/// which heals a create that stopped between its two puts and a namespace
/// made before this object existed.
///
/// # Errors
///
/// Anything [`read`], [`check`] and [`write`] return.
pub(crate) async fn ensure(
    store: &dyn ObjectStore,
    namespace: &str,
    expected: &Schema,
) -> Result<()> {
    match read(store, namespace).await? {
        Some(found) => check(namespace, &found, expected),
        None => write(store, namespace, expected).await,
    }
}

#[cfg(test)]
mod tests {
    use object_store::memory::InMemory;

    use super::*;
    use crate::attrs::AttrType;

    fn cosine3() -> Schema {
        Schema::new(3, Metric::Cosine)
    }

    fn l2_of(dim: usize) -> Schema {
        Schema::new(dim, Metric::L2)
    }

    fn typed() -> Schema {
        cosine3().with_attrs(
            AttrSchema::new()
                .field("price", AttrType::Number)
                .field("tags", AttrType::TextSet),
        )
    }

    async fn stored_bytes(store: &InMemory, namespace: &str) -> Bytes {
        store
            .get(&schema_path(namespace))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap()
    }

    async fn put_raw(store: &InMemory, body: &'static [u8]) {
        store
            .put(&schema_path("docs"), Bytes::from_static(body).into())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn the_object_is_the_documented_json_at_the_documented_key() {
        let store = InMemory::new();
        write(&store, "docs", &cosine3()).await.unwrap();
        assert_eq!(schema_path("docs").as_ref(), "docs/schema.json");
        assert_eq!(
            stored_bytes(&store, "docs").await,
            Bytes::from_static(br#"{"format_version":2,"dim":3,"metric":"cosine","attrs":{}}"#)
        );
    }

    #[tokio::test]
    async fn a_typed_schema_is_written_with_its_attrs_and_read_back() {
        let store = InMemory::new();
        write(&store, "docs", &typed()).await.unwrap();
        assert_eq!(
            stored_bytes(&store, "docs").await,
            Bytes::from_static(
                br#"{"format_version":2,"dim":3,"metric":"cosine","attrs":{"price":"number","tags":"set<string>"}}"#
            )
        );
        assert_eq!(read(&store, "docs").await.unwrap(), Some(typed()));
    }

    #[tokio::test]
    async fn write_then_read_agree() {
        let store = InMemory::new();
        assert_eq!(read(&store, "docs").await.unwrap(), None);
        write(&store, "docs", &cosine3()).await.unwrap();
        assert_eq!(read(&store, "docs").await.unwrap(), Some(cosine3()));
    }

    #[tokio::test]
    async fn writing_the_same_schema_twice_is_fine() {
        let store = InMemory::new();
        write(&store, "docs", &cosine3()).await.unwrap();
        let before = stored_bytes(&store, "docs").await;
        write(&store, "docs", &cosine3()).await.unwrap();
        assert_eq!(stored_bytes(&store, "docs").await, before);
    }

    #[tokio::test]
    async fn writing_a_different_schema_over_one_is_refused_and_leaves_it() {
        let store = InMemory::new();
        write(&store, "docs", &cosine3()).await.unwrap();
        let before = stored_bytes(&store, "docs").await;
        for other in [l2_of(4), typed()] {
            let err = write(&store, "docs", &other).await.unwrap_err();
            assert!(matches!(err, ItzmaError::CorruptManifest(_)), "{err:?}");
            assert_eq!(stored_bytes(&store, "docs").await, before);
        }
    }

    #[tokio::test]
    async fn ensure_writes_a_missing_object_and_then_reads_it() {
        let store = InMemory::new();
        ensure(&store, "docs", &cosine3()).await.unwrap();
        assert_eq!(read(&store, "docs").await.unwrap(), Some(cosine3()));
        let before = stored_bytes(&store, "docs").await;
        ensure(&store, "docs", &cosine3()).await.unwrap();
        assert_eq!(stored_bytes(&store, "docs").await, before);
    }

    #[tokio::test]
    async fn ensure_refuses_a_disagreeing_object_and_leaves_it() {
        let store = InMemory::new();
        write(&store, "docs", &cosine3()).await.unwrap();
        let before = stored_bytes(&store, "docs").await;
        let err = ensure(&store, "docs", &l2_of(3)).await.unwrap_err();
        assert!(matches!(err, ItzmaError::CorruptManifest(_)), "{err:?}");
        assert_eq!(stored_bytes(&store, "docs").await, before);
    }

    #[tokio::test]
    async fn a_body_that_is_not_json_is_a_serde_error_naming_the_key() {
        let store = InMemory::new();
        put_raw(&store, b"nope").await;
        match read(&store, "docs").await.unwrap_err() {
            ItzmaError::Serde { path, .. } => assert_eq!(path, "docs/schema.json"),
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn an_unknown_format_version_is_unsupported() {
        let store = InMemory::new();
        put_raw(
            &store,
            br#"{"format_version":3,"dim":3,"metric":"l2","attrs":{}}"#,
        )
        .await;
        let err = read(&store, "docs").await.unwrap_err();
        assert!(
            matches!(err, ItzmaError::UnsupportedFormat { version: 3 }),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn format_and_attrs_that_disagree_are_corrupt() {
        let store = InMemory::new();
        put_raw(&store, br#"{"format_version":2,"dim":1,"metric":"l2"}"#).await;
        let err = read(&store, "docs").await.unwrap_err();
        assert!(matches!(err, ItzmaError::CorruptManifest(_)), "{err:?}");

        // Format 1, a namespace with no attribute schema, is not read here.
        let store = InMemory::new();
        put_raw(&store, br#"{"format_version":1,"dim":1,"metric":"l2"}"#).await;
        let err = read(&store, "docs").await.unwrap_err();
        assert!(
            matches!(err, ItzmaError::UnsupportedFormat { version: 1 }),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn fields_a_later_build_adds_are_ignored() {
        let store = InMemory::new();
        put_raw(
            &store,
            br#"{"format_version":2,"dim":3,"metric":"l2","attrs":{},"extra":true}"#,
        )
        .await;
        assert_eq!(
            read(&store, "docs").await.unwrap(),
            Some(Schema::new(3, Metric::L2))
        );
    }
}
