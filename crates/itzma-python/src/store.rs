//! Where namespaces live: a URL and a few settings to an object store.

use std::path::PathBuf;
use std::sync::Arc;

use itzma_store_aws::{S3Options, S3Store};
use itzma_store_azure::{AzureCredential, AzureOptions, AzureStore};
use itzma_store_gcs::{GcsOptions, GcsStore};
use object_store::ObjectStore;
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
    Gcs { bucket: String, prefix: String },
    Azure { container: String, prefix: String },
}

/// Keyword overrides for a store. The S3 ones (`env` reads the AWS SDK's
/// default chain first), the GCS ones and the Azure ones each apply to their
/// own scheme only; `endpoint` to all three.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StoreOptions {
    pub(crate) region: Option<String>,
    pub(crate) endpoint: Option<String>,
    pub(crate) access_key_id: Option<String>,
    pub(crate) secret_access_key: Option<String>,
    pub(crate) session_token: Option<String>,
    pub(crate) allow_http: Option<bool>,
    pub(crate) env: bool,
    pub(crate) control_endpoint: Option<String>,
    pub(crate) anonymous: Option<bool>,
    pub(crate) account: Option<String>,
    pub(crate) credential: Option<String>,
    pub(crate) sas_token: Option<String>,
}

impl Default for StoreOptions {
    fn default() -> Self {
        Self {
            region: None,
            endpoint: None,
            access_key_id: None,
            secret_access_key: None,
            session_token: None,
            allow_http: None,
            env: true,
            control_endpoint: None,
            anonymous: None,
            account: None,
            credential: None,
            sas_token: None,
        }
    }
}

pub(crate) fn parse(url: &str) -> Result<Location, String> {
    let expected = "expected memory://, file:///<absolute path>, s3://<bucket>[/<prefix>] gs://<bucket>[/<prefix>] or az://<container>[/<prefix>]";
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
    let bucket_and_prefix = |rest: &str| -> Result<(String, String), String> {
        let (bucket, prefix) = rest.split_once('/').unwrap_or((rest, ""));
        if bucket.is_empty() {
            return Err(format!("{url:?}: no bucket; {expected}"));
        }
        Ok((bucket.into(), prefix.trim_matches('/').into()))
    };
    if let Some(rest) = url.strip_prefix("s3://") {
        let (bucket, prefix) = bucket_and_prefix(rest)?;
        return Ok(Location::S3 { bucket, prefix });
    }
    if let Some(rest) = url.strip_prefix("gs://") {
        let (bucket, prefix) = bucket_and_prefix(rest)?;
        return Ok(Location::Gcs { bucket, prefix });
    }
    if let Some(rest) = url.strip_prefix("az://") {
        let (container, prefix) = bucket_and_prefix(rest)?;
        return Ok(Location::Azure { container, prefix });
    }
    Err(format!("{url:?}: {expected}"))
}

/// The Azure credential an `az://` store's options ask for. A SAS token is
/// its own credential, so it can't be combined with another one.
fn azure_credential(options: &StoreOptions) -> Result<AzureCredential, String> {
    let named = match options.credential.as_deref() {
        None | Some("managed_identity") => AzureCredential::ManagedIdentity,
        Some("workload_identity") => AzureCredential::WorkloadIdentity,
        Some("developer_tools") => AzureCredential::DeveloperTools,
        Some("anonymous") => AzureCredential::Anonymous,
        Some(other) => {
            return Err(format!(
                "credential {other:?}: expected managed_identity, workload_identity, developer_tools or anonymous"
            ));
        }
    };
    match (&options.sas_token, &options.credential) {
        (None, _) => Ok(named),
        (Some(sas), None) => Ok(AzureCredential::Sas(sas.clone())),
        (Some(_), Some(_)) => {
            Err("sas_token is a credential of its own; leave credential unset".into())
        }
    }
}

/// Refuses an option the URL's scheme doesn't use, naming the option and the
/// scheme it belongs to.
pub(crate) fn check(location: &Location, options: &StoreOptions) -> Result<(), String> {
    let s3 = matches!(location, Location::S3 { .. });
    let gcs = matches!(location, Location::Gcs { .. });
    let az = matches!(location, Location::Azure { .. });
    let s3_only = [
        ("region", options.region.is_some()),
        ("access_key_id", options.access_key_id.is_some()),
        ("secret_access_key", options.secret_access_key.is_some()),
        ("session_token", options.session_token.is_some()),
        ("allow_http", options.allow_http.is_some()),
    ];
    let gcs_only = [
        ("control_endpoint", options.control_endpoint.is_some()),
        ("anonymous", options.anonymous.is_some()),
    ];
    if let Some((name, _)) = s3_only.iter().find(|(_, set)| *set && !s3) {
        return Err(format!("{name} only applies to s3:// URLs"));
    }
    if let Some((name, _)) = gcs_only.iter().find(|(_, set)| *set && !gcs) {
        return Err(format!("{name} only applies to gs:// URLs"));
    }
    let az_only = [
        ("account", options.account.is_some()),
        ("credential", options.credential.is_some()),
        ("sas_token", options.sas_token.is_some()),
    ];
    if let Some((name, _)) = az_only.iter().find(|(_, set)| *set && !az) {
        return Err(format!("{name} only applies to az:// URLs"));
    }
    if !options.env && !s3 {
        return Err("env only applies to s3:// URLs".into());
    }
    if options.endpoint.is_some() && !s3 && !gcs && !az {
        return Err("endpoint only applies to s3://, gs:// and az:// URLs".into());
    }
    if az {
        if options.account.is_none() {
            return Err("an az:// store needs account".into());
        }
        azure_credential(options)?;
    }
    // One without the other would send listings, deletes and copies to
    // production while reads and writes go to the endpoint.
    if gcs && options.endpoint.is_some() != options.control_endpoint.is_some() {
        return Err(
            "a gs:// store needs endpoint and control_endpoint together, or neither".into(),
        );
    }
    Ok(())
}

fn prefixed(store: Arc<dyn ObjectStore>, prefix: &str) -> Arc<dyn ObjectStore> {
    if prefix.is_empty() {
        store
    } else {
        Arc::new(PrefixStore::new(store, prefix))
    }
}

/// The store for `location`, its options already through [`check`]. Async
/// because a GCS store connects its clients.
pub(crate) async fn build(
    location: Location,
    options: &StoreOptions,
) -> Result<Arc<dyn ObjectStore>, String> {
    match location {
        Location::Memory => Ok(Arc::new(InMemory::new())),
        Location::File(path) => {
            std::fs::create_dir_all(&path)
                .map_err(|e| format!("cannot create {}: {e}", path.display()))?;
            let store = LocalFileSystem::new_with_prefix(&path).map_err(|e| e.to_string())?;
            Ok(Arc::new(store))
        }
        Location::S3 { bucket, prefix } => {
            let mut s3 = S3Options::default()
                .env(options.env)
                .allow_http(options.allow_http.unwrap_or(false));
            if let Some(v) = &options.region {
                s3 = s3.region(v);
            }
            if let Some(v) = &options.endpoint {
                s3 = s3.endpoint(v);
            }
            if let Some(v) = &options.access_key_id {
                s3 = s3.access_key_id(v);
            }
            if let Some(v) = &options.secret_access_key {
                s3 = s3.secret_access_key(v);
            }
            if let Some(v) = &options.session_token {
                s3 = s3.session_token(v);
            }
            let store = S3Store::connect(&bucket, s3)
                .await
                .map_err(|e| e.to_string())?;
            Ok(prefixed(Arc::new(store), &prefix))
        }
        Location::Gcs { bucket, prefix } => {
            let mut gcs = GcsOptions::default().anonymous(options.anonymous.unwrap_or(false));
            if let Some(v) = &options.endpoint {
                gcs = gcs.endpoint(v);
            }
            if let Some(v) = &options.control_endpoint {
                gcs = gcs.control_endpoint(v);
            }
            let store = GcsStore::connect(&bucket, gcs)
                .await
                .map_err(|e| e.to_string())?;
            Ok(prefixed(Arc::new(store), &prefix))
        }
        Location::Azure { container, prefix } => {
            let account = options.account.clone().unwrap_or_default();
            let mut az = AzureOptions::default().credential(azure_credential(options)?);
            if let Some(v) = &options.endpoint {
                az = az.endpoint(v);
            }
            let store = AzureStore::connect(&account, &container, az)
                .await
                .map_err(|e| e.to_string())?;
            Ok(prefixed(Arc::new(store), &prefix))
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
                        secret_access_key=None, session_token=None, allow_http=None, env=true,
                        control_endpoint=None, anonymous=None,
                        account=None, credential=None, sas_token=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        url: &str,
        region: Option<String>,
        endpoint: Option<String>,
        access_key_id: Option<String>,
        secret_access_key: Option<String>,
        session_token: Option<String>,
        allow_http: Option<bool>,
        env: bool,
        control_endpoint: Option<String>,
        anonymous: Option<bool>,
        account: Option<String>,
        credential: Option<String>,
        sas_token: Option<String>,
    ) -> PyResult<Self> {
        let location = parse(url).map_err(error::input)?;
        let options = StoreOptions {
            region,
            endpoint,
            access_key_id,
            secret_access_key,
            session_token,
            allow_http,
            env,
            control_endpoint,
            anonymous,
            account,
            credential,
            sas_token,
        };
        check(&location, &options).map_err(error::input)?;
        // Past `check`, a failure is the store's (no credentials, a bad
        // endpoint), not the caller's input.
        let inner = runtime::block_on(py, async move {
            build(location, &options)
                .await
                .map_err(|e| error::raise("StoreError", e))
        })?;
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
        assert_eq!(
            parse("gs://b"),
            Ok(Location::Gcs {
                bucket: "b".into(),
                prefix: String::new()
            })
        );
        assert_eq!(
            parse("gs://b/p/q/"),
            Ok(Location::Gcs {
                bucket: "b".into(),
                prefix: "p/q".into()
            })
        );
        assert_eq!(
            parse("az://c"),
            Ok(Location::Azure {
                container: "c".into(),
                prefix: String::new()
            })
        );
        assert_eq!(
            parse("az://c/p/q/"),
            Ok(Location::Azure {
                container: "c".into(),
                prefix: "p/q".into()
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
            "gs://",
            "gs:///p",
            "az://",
            "az:///p",
            "http://x",
        ] {
            assert!(parse(url).is_err(), "{url}");
        }
    }

    #[test]
    fn options_belong_to_their_scheme() {
        let gcs = Location::Gcs {
            bucket: "b".into(),
            prefix: String::new(),
        };
        let s3 = Location::S3 {
            bucket: "b".into(),
            prefix: String::new(),
        };
        let dir = tempfile::tempdir().expect("tempdir");
        let region = StoreOptions {
            region: Some("us-east-1".into()),
            ..StoreOptions::default()
        };
        let control = StoreOptions {
            control_endpoint: Some("http://127.0.0.1:9101".into()),
            ..StoreOptions::default()
        };
        let anonymous = StoreOptions {
            anonymous: Some(true),
            ..StoreOptions::default()
        };
        let endpoint = StoreOptions {
            endpoint: Some("http://127.0.0.1:9000".into()),
            ..StoreOptions::default()
        };
        let refused = |location: &Location, options: &StoreOptions, needle: &str| {
            let message = check(location, options).expect_err("refused");
            assert!(message.contains(needle), "{message:?} names {needle:?}");
        };
        refused(&gcs, &region, "s3://");
        for location in [&s3, &Location::Memory, &Location::File(dir.path().into())] {
            refused(location, &control, "gs://");
            refused(location, &anonymous, "gs://");
        }
        refused(&Location::Memory, &endpoint, "s3://, gs:// and az://");
        refused(&gcs, &endpoint, "control_endpoint");
        refused(&gcs, &control, "endpoint");
        let no_env = StoreOptions {
            env: false,
            ..StoreOptions::default()
        };
        refused(&gcs, &no_env, "env");
        let both = StoreOptions {
            endpoint: Some("http://127.0.0.1:9100".into()),
            control_endpoint: Some("http://127.0.0.1:9101".into()),
            ..StoreOptions::default()
        };
        assert!(check(&s3, &endpoint).is_ok());
        assert!(check(&s3, &no_env).is_ok());
        assert!(check(&gcs, &both).is_ok());

        let az = Location::Azure {
            container: "c".into(),
            prefix: String::new(),
        };
        let account = |extra: StoreOptions| StoreOptions {
            account: Some("acct".into()),
            ..extra
        };
        refused(&az, &StoreOptions::default(), "account");
        for location in [&s3, &gcs, &Location::Memory] {
            refused(location, &account(StoreOptions::default()), "az://");
            let sas = StoreOptions {
                sas_token: Some("sv=x".into()),
                ..StoreOptions::default()
            };
            refused(location, &sas, "az://");
        }
        let bad_credential = account(StoreOptions {
            credential: Some("password".into()),
            ..StoreOptions::default()
        });
        refused(&az, &bad_credential, "managed_identity");
        let sas_with_token = account(StoreOptions {
            credential: Some("managed_identity".into()),
            sas_token: Some("sv=x".into()),
            ..StoreOptions::default()
        });
        refused(&az, &sas_with_token, "sas_token");
        refused(&az, &account(region.clone()), "s3://");
        assert!(check(&az, &account(StoreOptions::default())).is_ok());
        let azurite = account(StoreOptions {
            endpoint: Some("http://127.0.0.1:10000/devstoreaccount1".into()),
            sas_token: Some("sv=x".into()),
            ..StoreOptions::default()
        });
        assert!(check(&az, &azurite).is_ok());
    }

    #[test]
    fn file_store_creates_its_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("a/b");
        futures::executor::block_on(build(
            Location::File(path.clone()),
            &StoreOptions::default(),
        ))
        .expect("store");
        assert!(path.is_dir());
    }

    #[test]
    fn s3_store_builds_without_env() {
        let options = StoreOptions {
            region: Some("us-east-1".into()),
            endpoint: Some("http://127.0.0.1:9000".into()),
            access_key_id: Some("k".into()),
            secret_access_key: Some("s".into()),
            allow_http: Some(true),
            env: false,
            ..StoreOptions::default()
        };
        let location = Location::S3 {
            bucket: "b".into(),
            prefix: "p".into(),
        };
        futures::executor::block_on(build(location, &options)).expect("store");
    }

    #[test]
    fn s3_http_endpoint_needs_allow_http() {
        let options = StoreOptions {
            endpoint: Some("http://127.0.0.1:9000".into()),
            access_key_id: Some("k".into()),
            secret_access_key: Some("s".into()),
            env: false,
            ..StoreOptions::default()
        };
        let location = Location::S3 {
            bucket: "b".into(),
            prefix: "p".into(),
        };
        let message = futures::executor::block_on(build(location, &options)).expect_err("refused");
        assert!(message.contains("allow_http"), "{message}");
    }
}
