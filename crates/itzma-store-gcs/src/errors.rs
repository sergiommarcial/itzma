//! SDK errors to `object_store` errors. Itzma tells a lost race, a gone object
//! and a bad range apart by these variants, so the mapping is in one place.

use google_cloud_gax::error::rpc::Code;
use google_cloud_storage::Error as SdkError;

const STORE: &str = "GCS";

/// What the failed call was trying to do, since a failed precondition means
/// "taken" for put-if-absent and "stale" for an update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Write {
    None,
    Create,
    Update,
}

pub(crate) fn map(err: SdkError, path: &str, write: Write) -> object_store::Error {
    let http = err.http_status_code();
    let code = err.status().map(|s| s.code);
    let is = |status: u16, rpc: Code| http == Some(status) || code == Some(rpc);
    let path = path.to_string();
    if is(404, Code::NotFound) {
        return object_store::Error::NotFound {
            path,
            source: Box::new(err),
        };
    }
    if is(412, Code::FailedPrecondition) {
        return match write {
            Write::Create => object_store::Error::AlreadyExists {
                path,
                source: Box::new(err),
            },
            Write::Update | Write::None => object_store::Error::Precondition {
                path,
                source: Box::new(err),
            },
        };
    }
    object_store::Error::Generic {
        store: STORE,
        source: Box::new(err),
    }
}

pub(crate) fn not_implemented(operation: &str) -> object_store::Error {
    object_store::Error::NotImplemented {
        operation: operation.into(),
        implementer: STORE.into(),
    }
}
