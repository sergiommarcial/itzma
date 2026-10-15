//! Where namespaces live: a URL and a few S3 settings to an object store.

use std::path::PathBuf;
use std::sync::Arc;

use object_store::ObjectStore;
use object_store::aws::AmazonS3Builder;
use object_store::local::LocalFileSystem;
use object_store::memory::InMemory;
use object_store::prefix::PrefixStore;
use pyo3::prelude::*;

use crate::{error, runtime};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Location {
    Memory,
    File(PathBuf),
    S3 { bucket: String, prefix: String },
}

/// Keyword overrides for an S3 store. `env` reads `AWS_*` first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct S3Options {
    pub(crate) region: Option<String>,
    pub(crate) endpoint: Option<String>,
    pub(crate) access_key_id: Option<String>,
    pub(crate) secret_access_key: Option<String>,
    pub(crate) session_token: Option<String>,
    pub(crate) allow_http: Option<bool>,
    pub(crate) env: bool,
}

impl Default for S3Options {
    fn default() -> Self {
        Self {
            region: None,
            endpoint: None,
            access_key_id: None,
            secret_access_key: None,
            session_token: None,
            allow_http: None,
            env: true,
        }
    }
}

impl S3Options {
    fn any_set(&self) -> bool {
        self.region.is_some()
            || self.endpoint.is_some()
            || self.access_key_id.is_some()
            || self.secret_access_key.is_some()
            || self.session_token.is_some()
            || self.allow_http.is_some()
    }
}

pub(crate) fn parse(url: &str) -> Result<Location, String> {
    let expected = "expected memory://, file:///<absolute path> or s3://<bucket>[/<prefix>]";
    if url == "memory://" {
        return Ok(Location::Memory);
    }
    if let Some(path) = url.strip_prefix("file://") {
        if !path.starts_with('/') {
            return Err(format!(
                "{url:?}: a file:// URL needs an absolute path, like file:///data/itzma"
            ));
        }
        return Ok(Location::File(PathBuf::from(path)));
    }
    if let Some(rest) = url.strip_prefix("s3://") {
        let (bucket, prefix) = rest.split_once('/').unwrap_or((rest, ""));
        if bucket.is_empty() {
            return Err(format!("{url:?}: no bucket; {expected}"));
        }
        return Ok(Location::S3 {
            bucket: bucket.into(),
            prefix: prefix.trim_matches('/').into(),
        });
    }
    Err(format!("{url:?}: {expected}"))
}

pub(crate) fn build(location: Location, s3: &S3Options) -> Result<Arc<dyn ObjectStore>, String> {
    if !matches!(location, Location::S3 { .. }) && s3.any_set() {
        return Err("region, endpoint, credentials and allow_http only apply to s3:// URLs".into());
    }
    match location {
        Location::Memory => Ok(Arc::new(InMemory::new())),
        Location::File(path) => {
            std::fs::create_dir_all(&path)
                .map_err(|e| format!("cannot create {}: {e}", path.display()))?;
            let store = LocalFileSystem::new_with_prefix(&path).map_err(|e| e.to_string())?;
            Ok(Arc::new(store))
        }
        Location::S3 { bucket, prefix } => {
            let mut builder = if s3.env {
                AmazonS3Builder::from_env()
            } else {
                AmazonS3Builder::new()
            };
            builder = builder.with_bucket_name(bucket);
            if let Some(v) = &s3.region {
                builder = builder.with_region(v);
            }
            if let Some(v) = &s3.endpoint {
                builder = builder.with_endpoint(v);
            }
            if let Some(v) = &s3.access_key_id {
                builder = builder.with_access_key_id(v);
            }
            if let Some(v) = &s3.secret_access_key {
                builder = builder.with_secret_access_key(v);
            }
            if let Some(v) = &s3.session_token {
                builder = builder.with_token(v);
            }
            if let Some(v) = s3.allow_http {
                builder = builder.with_allow_http(v);
            }
            let store = builder.build().map_err(|e| e.to_string())?;
            if prefix.is_empty() {
                Ok(Arc::new(store))
            } else {
                Ok(Arc::new(PrefixStore::new(store, prefix.as_str())))
            }
        }
    }
}

#[pyclass(frozen, name = "Store", module = "itzma._itzma")]
pub(crate) struct PyStore {
    inner: Arc<dyn ObjectStore>,
    pid: u32,
}

impl PyStore {
    /// The store, refused in a process other than the one that built it: an
    /// S3 client's pooled connections belong to the parent's runtime.
    pub(crate) fn handle(&self) -> PyResult<Arc<dyn ObjectStore>> {
        match runtime::fork_error(self.pid, std::process::id()) {
            Some(message) => Err(error::itzma_error(message.replace("handle", "Store"))),
            None => Ok(self.inner.clone()),
        }
    }
}

#[pymethods]
impl PyStore {
    #[new]
    #[pyo3(signature = (url, *, region=None, endpoint=None, access_key_id=None,
                        secret_access_key=None, session_token=None, allow_http=None, env=true))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        url: &str,
        region: Option<String>,
        endpoint: Option<String>,
        access_key_id: Option<String>,
        secret_access_key: Option<String>,
        session_token: Option<String>,
        allow_http: Option<bool>,
        env: bool,
    ) -> PyResult<Self> {
        let location = parse(url).map_err(error::input)?;
        let options = S3Options {
            region,
            endpoint,
            access_key_id,
            secret_access_key,
            session_token,
            allow_http,
            env,
        };
        let inner = build(location, &options).map_err(error::input)?;
        Ok(Self {
            inner,
            pid: std::process::id(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_each_scheme() {
        assert_eq!(parse("memory://"), Ok(Location::Memory));
        assert_eq!(
            parse("file:///data/itzma"),
            Ok(Location::File("/data/itzma".into()))
        );
        assert_eq!(
            parse("s3://bucket"),
            Ok(Location::S3 {
                bucket: "bucket".into(),
                prefix: String::new()
            })
        );
        assert_eq!(
            parse("s3://bucket/a/b/"),
            Ok(Location::S3 {
                bucket: "bucket".into(),
                prefix: "a/b".into()
            })
        );
    }

    #[test]
    fn refuses_bad_urls() {
        for url in [
            "",
            "memory",
            "file://relative",
            "s3://",
            "s3:///x",
            "gs://b",
            "http://x",
        ] {
            assert!(parse(url).is_err(), "{url}");
        }
    }

    #[test]
    fn s3_options_only_apply_to_s3() {
        let options = S3Options {
            region: Some("us-east-1".into()),
            ..S3Options::default()
        };
        assert!(build(Location::Memory, &options).is_err());
        let dir = tempfile::tempdir().expect("tempdir");
        assert!(build(Location::File(dir.path().into()), &options).is_err());
    }

    #[test]
    fn file_store_creates_its_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("a/b");
        build(Location::File(path.clone()), &S3Options::default()).expect("store");
        assert!(path.is_dir());
    }

    #[test]
    fn s3_store_builds_without_env() {
        let options = S3Options {
            region: Some("us-east-1".into()),
            endpoint: Some("http://127.0.0.1:9000".into()),
            access_key_id: Some("k".into()),
            secret_access_key: Some("s".into()),
            allow_http: Some(true),
            env: false,
            ..S3Options::default()
        };
        let location = Location::S3 {
            bucket: "b".into(),
            prefix: "p".into(),
        };
        build(location, &options).expect("store");
    }
}
