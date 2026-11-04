//! SDK errors to `object_store` errors. Itzma tells a lost race, a gone object
//! and a bad range apart by these variants, so the mapping is in one place.
//! Messages never carry a SAS signature: it is a bearer secret.

const STORE: &str = "Azure";

/// What the failed call was trying to do, since a failed precondition means
/// "taken" for put-if-absent and "stale" for an update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Write {
    None,
    Create,
    Update,
}

pub(crate) fn map(err: azure_core::Error, path: &str, write: Write) -> object_store::Error {
    let status = err.http_status().map(u16::from);
    let code = match err.kind() {
        azure_core::error::ErrorKind::HttpResponse { error_code, .. } => error_code.clone(),
        _ => None,
    };
    let source: Box<dyn std::error::Error + Send + Sync> = redact(&err.to_string()).into();
    let path = path.to_string();
    // A lost race is 409 BlobAlreadyExists or 412 ConditionNotMet. Other 409s
    // and 412s (an archived or immutable blob, a missing lease) are real
    // failures, and reporting them as a race would send core into retries.
    // Without a code the status alone decides, as for a HEAD.
    let race = matches!(
        (status, code.as_deref()),
        (
            Some(409 | 412),
            None | Some("BlobAlreadyExists" | "ConditionNotMet")
        )
    );
    match (status, write) {
        (Some(404), _) => object_store::Error::NotFound { path, source },
        (Some(409 | 412), Write::Create) if race => {
            object_store::Error::AlreadyExists { path, source }
        }
        (Some(409 | 412), _) if race => object_store::Error::Precondition { path, source },
        _ => object_store::Error::Generic {
            store: STORE,
            source,
        },
    }
}

/// `text` with every `sig=` value replaced, so a URL in an SDK message cannot
/// leak the SAS signature.
pub(crate) fn redact(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find("sig=") {
        out.push_str(&rest[..at + 4]);
        out.push_str("REDACTED");
        let tail = &rest[at + 4..];
        let end = tail
            .find(|c: char| c == '&' || c == '"' || c == '\'' || c == ')' || c.is_whitespace())
            .unwrap_or(tail.len());
        rest = &tail[end..];
    }
    out.push_str(rest);
    out
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
    use azure_core::error::ErrorKind;
    use azure_core::http::StatusCode;

    fn http(status: StatusCode, code: &str) -> azure_core::Error {
        azure_core::Error::with_message(
            ErrorKind::HttpResponse {
                status,
                error_code: Some(code.into()),
                raw_response: None,
            },
            code.to_string(),
        )
    }

    #[test]
    fn races_are_told_from_other_conflicts_by_error_code() {
        let taken = map(
            http(StatusCode::Conflict, "BlobAlreadyExists"),
            "k",
            Write::Create,
        );
        assert!(
            matches!(taken, object_store::Error::AlreadyExists { .. }),
            "{taken:?}"
        );
        let stale = map(
            http(StatusCode::PreconditionFailed, "ConditionNotMet"),
            "k",
            Write::Update,
        );
        assert!(
            matches!(stale, object_store::Error::Precondition { .. }),
            "{stale:?}"
        );
        for (status, code) in [
            (StatusCode::Conflict, "BlobArchived"),
            (StatusCode::Conflict, "BlobImmutableDueToPolicy"),
            (StatusCode::PreconditionFailed, "LeaseIdMissing"),
        ] {
            for write in [Write::Create, Write::Update] {
                let err = map(http(status, code), "k", write);
                assert!(
                    matches!(err, object_store::Error::Generic { .. }),
                    "{code}: {err:?}"
                );
            }
        }
    }

    #[test]
    fn redact_hides_every_signature() {
        let text = "GET http://h/c/b?sv=1&sig=abc%2Bdef&se=2 failed; again sig=xyz";
        let out = redact(text);
        assert!(!out.contains("abc") && !out.contains("xyz"), "{out}");
        assert!(out.contains("sv=1") && out.contains("se=2"), "{out}");
    }
}
