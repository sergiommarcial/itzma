//! Conditional writes and the SDK's retries. S3 asks callers to retry a 409
//! `ConditionalRequestConflict`, which the SDK doesn't, and a retried write
//! can meet the object its own lost attempt wrote.

use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use aws_sdk_s3::config::interceptors::{BeforeTransmitInterceptorContextRef, InterceptorContext};
use aws_sdk_s3::config::retry::{ClassifyRetry, RetryAction};
use aws_sdk_s3::config::{ConfigBag, Intercept, RuntimeComponents};
use aws_sdk_s3::error::{BoxError, ProvideErrorMetadata};
use md5::{Digest, Md5};

/// Retries `E` with the code `ConditionalRequestConflict`, which S3 sends
/// while another conditional write or a delete of the key is in flight. The
/// retry then wins, or gets the 412 that says the other write did.
pub(crate) struct Conflict<E>(PhantomData<fn() -> E>);

impl<E> std::fmt::Debug for Conflict<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Conflict")
    }
}

impl<E> ClassifyRetry for Conflict<E>
where
    E: ProvideErrorMetadata + std::error::Error + Send + Sync + 'static,
{
    fn classify_retry(&self, ctx: &InterceptorContext) -> RetryAction {
        let code = ctx
            .output_or_error()
            .and_then(Result::err)
            .and_then(|e| e.as_operation_error())
            .and_then(|e| e.downcast_ref::<E>())
            .and_then(ProvideErrorMetadata::code);
        match code {
            Some("ConditionalRequestConflict") => RetryAction::transient_error(),
            _ => RetryAction::NoActionIndicated,
        }
    }

    fn name(&self) -> &'static str {
        "ConditionalRequestConflict"
    }
}

/// The config a conditional write overrides its client's with: the client's
/// retries, plus [`Conflict`].
pub(crate) fn conditional<E>() -> aws_sdk_s3::config::Builder
where
    E: ProvideErrorMetadata + std::error::Error + Send + Sync + 'static,
{
    aws_sdk_s3::Config::builder().retry_classifier(Conflict::<E>(PhantomData))
}

/// Counts a request's attempts, so a failed conditional write knows whether
/// an earlier attempt may have landed.
#[derive(Debug, Clone, Default)]
pub(crate) struct Attempts(Arc<AtomicU32>);

impl Attempts {
    pub(crate) fn retried(&self) -> bool {
        self.0.load(Ordering::SeqCst) > 1
    }
}

impl Intercept for Attempts {
    fn name(&self) -> &'static str {
        "Attempts"
    }

    fn read_before_attempt(
        &self,
        _context: &BeforeTransmitInterceptorContextRef<'_>,
        _components: &RuntimeComponents,
        _config: &mut ConfigBag,
    ) -> Result<(), BoxError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

/// Whether `e_tag` is that of a single-part upload of `body`: its MD5 in hex.
/// Objects encrypted with KMS or customer keys have other ETags and never match.
pub(crate) fn is_e_tag_of(e_tag: &str, body: &[u8]) -> bool {
    let hex: String = Md5::digest(body)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    e_tag.trim_matches('"') == hex
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e_tag_is_the_md5_of_the_body() {
        assert!(is_e_tag_of(
            "\"5d41402abc4b2a76b9719d911017c592\"",
            b"hello"
        ));
        assert!(is_e_tag_of("d41d8cd98f00b204e9800998ecf8427e", b""));
        assert!(!is_e_tag_of(
            "\"5d41402abc4b2a76b9719d911017c592\"",
            b"world"
        ));
        assert!(!is_e_tag_of(
            "\"5d41402abc4b2a76b9719d911017c592-1\"",
            b"hello"
        ));
    }
}
