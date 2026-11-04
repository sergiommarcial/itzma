//! SDK errors to `object_store` errors. Itzma tells a lost race, a gone object
//! and a bad range apart by these variants, so the mapping is in one place.

use aws_sdk_s3::error::{ProvideErrorMetadata, SdkError};

const STORE: &str = "S3";

/// What the failed call was trying to do, since a failed precondition means
/// "taken" for put-if-absent and "stale" for an update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Write {
    None,
    Create,
    Update,
}

/// The `object_store` error a status and S3 error code stand for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    NotFound,
    AlreadyExists,
    Precondition,
    Generic,
}

pub(crate) fn kind(status: Option<u16>, code: Option<&str>, write: Write) -> Kind {
    // A lost race is 412 PreconditionFailed, or 409 ConditionalRequestConflict
    // when two conditional writes overlap. Other 409s and 412s are real
    // failures, and reporting them as a race would send core into retries.
    // Without a code (a HEAD has no body) the status alone decides.
    let race = matches!(
        (status, code),
        (
            Some(409 | 412),
            None | Some("PreconditionFailed" | "ConditionalRequestConflict")
        )
    );
    match (status, write) {
        // S3 answers an If-Match on a missing key with 404; for an update
        // that is a stale version, as on every other store. A missing bucket
        // is not.
        (Some(404), Write::Update) if matches!(code, None | Some("NoSuchKey")) => {
            Kind::Precondition
        }
        (Some(404), _) => Kind::NotFound,
        (Some(409 | 412), Write::Create) if race => Kind::AlreadyExists,
        (Some(409 | 412), Write::Update) if race => Kind::Precondition,
        _ => Kind::Generic,
    }
}

pub(crate) fn map<E, R>(err: SdkError<E, R>, path: &str, write: Write) -> object_store::Error
where
    E: ProvideErrorMetadata + std::error::Error + Send + Sync + 'static,
    R: std::fmt::Debug + HasStatus,
{
    let status = err.raw_response().map(HasStatus::status);
    let code = match &err {
        SdkError::ServiceError(service) => service.err().code().map(str::to_string),
        _ => None,
    };
    let source: Box<dyn std::error::Error + Send + Sync> = describe(&err, status).into();
    let path = path.to_string();
    match kind(status, code.as_deref(), write) {
        Kind::NotFound => object_store::Error::NotFound { path, source },
        Kind::AlreadyExists => object_store::Error::AlreadyExists { path, source },
        Kind::Precondition => object_store::Error::Precondition { path, source },
        Kind::Generic => object_store::Error::Generic {
            store: STORE,
            source,
        },
    }
}

/// The error as text: the status, S3's code and message for a service error,
/// else the error and its sources. Never the raw response, whose body can
/// echo the access key id, nor the SDK's `Debug` form, which prints it.
fn describe<E, R>(err: &SdkError<E, R>, status: Option<u16>) -> String
where
    E: ProvideErrorMetadata + std::error::Error + 'static,
    R: std::fmt::Debug,
{
    if let SdkError::ServiceError(service) = err {
        let meta = service.err();
        let status = status.map(|s| format!("HTTP {s} ")).unwrap_or_default();
        return format!(
            "{status}{}: {}",
            meta.code().unwrap_or("no error code"),
            meta.message().unwrap_or("no message")
        );
    }
    let mut text = err.to_string();
    let mut source = std::error::Error::source(err);
    while let Some(next) = source {
        text.push_str(": ");
        text.push_str(&next.to_string());
        source = next.source();
    }
    text
}

/// The HTTP status of a raw response, so `map` takes the SDK's response type
/// without naming it.
pub(crate) trait HasStatus {
    fn status(&self) -> u16;
}

impl HasStatus for aws_sdk_s3::config::http::HttpResponse {
    fn status(&self) -> u16 {
        self.status().as_u16()
    }
}

pub(crate) fn generic(message: impl Into<String>) -> object_store::Error {
    object_store::Error::Generic {
        store: STORE,
        source: message.into().into(),
    }
}

pub(crate) fn not_implemented(operation: &str) -> object_store::Error {
    object_store::Error::NotImplemented {
        operation: operation.into(),
        implementer: STORE.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aws_sdk_s3::config::http::HttpResponse;
    use aws_sdk_s3::error::ErrorMetadata;
    use aws_sdk_s3::operation::put_object::PutObjectError;
    use aws_sdk_s3::primitives::SdkBody;

    #[test]
    fn errors_keep_the_code_and_drop_the_response_body() {
        // Real S3 echoes the access key id in an InvalidAccessKeyId body.
        let body = "<Error><Code>InvalidAccessKeyId</Code>\
            <AWSAccessKeyId>AKIDLEAKCHECK</AWSAccessKeyId></Error>";
        let raw = HttpResponse::new(403.try_into().expect("status"), SdkBody::from(body));
        let service = PutObjectError::generic(
            ErrorMetadata::builder()
                .code("InvalidAccessKeyId")
                .message("The AWS Access Key Id you provided does not exist in our records.")
                .build(),
        );
        let err = map(SdkError::service_error(service, raw), "k", Write::None);
        let text = format!("{err} {err:?}");
        assert!(text.contains("InvalidAccessKeyId"), "{text}");
        assert!(text.contains("403"), "{text}");
        assert!(!text.contains("AKIDLEAKCHECK"), "{text}");
    }

    #[test]
    fn races_are_told_from_other_conflicts_by_error_code() {
        for code in ["PreconditionFailed", "ConditionalRequestConflict"] {
            for status in [409, 412] {
                assert_eq!(
                    kind(Some(status), Some(code), Write::Create),
                    Kind::AlreadyExists,
                    "{status} {code} on a create"
                );
                assert_eq!(
                    kind(Some(status), Some(code), Write::Update),
                    Kind::Precondition,
                    "{status} {code} on an update"
                );
            }
        }
        for write in [Write::None, Write::Create, Write::Update] {
            assert_eq!(
                kind(Some(409), Some("OperationAborted"), write),
                Kind::Generic,
                "another 409 is a plain failure"
            );
            assert_eq!(
                kind(Some(416), Some("InvalidRange"), write),
                Kind::Generic,
                "a bad range"
            );
        }
        assert_eq!(kind(Some(412), None, Write::Create), Kind::AlreadyExists);
        assert_eq!(kind(Some(404), None, Write::None), Kind::NotFound);
        assert_eq!(
            kind(Some(404), Some("NoSuchKey"), Write::Update),
            Kind::Precondition,
            "an update of a missing key"
        );
        assert_eq!(
            kind(Some(404), Some("NoSuchBucket"), Write::Update),
            Kind::NotFound,
            "a missing bucket is not a stale version"
        );
    }
}
