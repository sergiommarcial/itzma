//! One check per item of the storage contract.
//!
//! Each check takes a fresh store, writes what it needs, and panics with the
//! contract item in the message when the store gets it wrong. Run them with
//! [`conformance_tests!`](crate::conformance_tests).

mod copy;
mod listing;
mod reads;
mod writes;

pub use copy::{copy_if_absent, copy_overwrites, multipart_round_trip};
pub use listing::{
    list_across_pages, list_under_prefix, list_with_delimiter_splits, list_with_offset_after,
};
pub use reads::{
    delete_removes, empty_object_reads_empty, get_returns_body_and_size, missing_object,
    range_past_end, ranged_reads, suffix_longer_than_object,
};
pub use writes::{
    put_if_absent_creates, put_if_absent_one_winner, update_matches_version,
    update_missing_is_precondition, update_one_winner, update_refuses_stale,
    update_with_get_version, update_with_head_version, update_without_version_refused,
};

/// Expands to `mod $name` holding one `#[tokio::test]` per check, each named
/// after its check and run against a store from `$factory`.
///
/// `$factory` is an `async fn() -> Option<Arc<dyn ObjectStore>>` called once
/// per test, so every check gets a store of its own; `None` skips the test
/// (for a backend whose endpoint is not configured) and says so on stderr,
/// so run with `--nocapture` to see which backends ran. The factory is
/// resolved inside the generated module, so pass a plain name in scope or a
/// `crate::` path, not a `self::` or `super::` one. The calling crate needs
/// `tokio` with the `macros` and `rt-multi-thread` features.
#[macro_export]
macro_rules! conformance_tests {
    ($name:ident, $factory:path) => {
        $crate::__conformance_tests!(
            $name,
            $factory,
            [
                put_if_absent_creates,
                put_if_absent_one_winner,
                update_matches_version,
                update_refuses_stale,
                update_with_head_version,
                update_missing_is_precondition,
                update_with_get_version,
                update_without_version_refused,
                update_one_winner,
                get_returns_body_and_size,
                empty_object_reads_empty,
                ranged_reads,
                range_past_end,
                missing_object,
                suffix_longer_than_object,
                delete_removes,
                list_under_prefix,
                list_across_pages,
                list_with_offset_after,
                list_with_delimiter_splits,
                copy_if_absent,
                copy_overwrites,
                multipart_round_trip,
            ]
        );
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __conformance_tests {
    ($name:ident, $factory:path, [$($check:ident),* $(,)?]) => {
        mod $name {
            #[allow(unused_imports)]
            use super::*;
            $(
                #[tokio::test(flavor = "multi_thread")]
                async fn $check() {
                    let Some(store) = $factory().await else {
                        eprintln!(
                            "{}::{}: skipped, no store configured",
                            stringify!($name),
                            stringify!($check)
                        );
                        return;
                    };
                    $crate::conformance::$check(store).await;
                }
            )*
        }
    };
}
