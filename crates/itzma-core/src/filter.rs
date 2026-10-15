//! The filter-only query behind [`Namespace::filter`](crate::Namespace::filter)
//!: one page of the id-order merge, read a block at a time, with
//! the predicate tested on each winner.

use std::future::Future;
use std::sync::Arc;

use object_store::ObjectStore;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::budget::ReadBudget;
use crate::error::{ItzmaError, Result};
use crate::fetch::FetchPolicy;
use crate::layer::Layer;
use crate::manifest::Manifest;
use crate::predicate::Predicate;
use crate::scan::merge::{Merge, Open};

/// The page size when none is asked for.
pub const DEFAULT_FILTER_LIMIT: usize = 100;
/// The most rows one page may ask for.
pub const MAX_FILTER_LIMIT: usize = 10_000;

/// Settings for [`Namespace::filter`](crate::Namespace::filter).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct FilterOptions {
    /// The most rows to return, 1 to [`MAX_FILTER_LIMIT`]. Default
    /// [`DEFAULT_FILTER_LIMIT`].
    pub limit: usize,
    /// Return only rows whose id is greater than this, in byte order. Pass the
    /// previous page's [`FilterPage::next`] to get the next page.
    pub after: Option<String>,
}

impl Default for FilterOptions {
    fn default() -> Self {
        Self {
            limit: DEFAULT_FILTER_LIMIT,
            after: None,
        }
    }
}

impl FilterOptions {
    /// Set [`limit`](Self::limit).
    #[must_use]
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Set [`after`](Self::after).
    #[must_use]
    pub fn after(mut self, id: impl Into<String>) -> Self {
        self.after = Some(id.into());
        self
    }

    /// Refuse options that cannot be run, before any storage call.
    pub(crate) fn validate(&self) -> Result<()> {
        if !(1..=MAX_FILTER_LIMIT).contains(&self.limit) {
            return Err(ItzmaError::InvalidQuery(
                "filter limit must be between 1 and 10000",
            ));
        }
        Ok(())
    }
}

/// One matching document: its id and its attributes as stored. No vector.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    /// The document's id.
    pub id: String,
    /// The document's attributes, exactly as stored.
    pub attrs: Option<Value>,
}

/// One page of a filter query.
#[derive(Debug, Clone, PartialEq)]
pub struct FilterPage {
    /// The matching rows, in ascending id order.
    pub rows: Vec<Row>,
    /// The id to pass as [`FilterOptions::after`] for the next page. `Some` when
    /// the page is full, which means there may be more: a page that ends exactly
    /// at the last match still returns it, and the next page is then empty with
    /// `next` set to `None`. `None` when the scan ran out of documents.
    pub next: Option<String>,
}

/// One filter query against one manifest.
pub(crate) struct Scan<'a> {
    pub(crate) store: &'a dyn ObjectStore,
    pub(crate) namespace: &'a str,
    pub(crate) dim: usize,
    pub(crate) predicate: &'a Predicate,
    pub(crate) limit: usize,
    pub(crate) after: Option<&'a str>,
    pub(crate) budget: &'a ReadBudget,
    pub(crate) policy: &'a FetchPolicy,
    /// Parsed structures kept between queries, if the handle has a
    /// decoded cache.
    pub(crate) decoded: Option<&'a crate::decoded::DecodedRef>,
}

impl<'a> Scan<'a> {
    /// What the shared merge reads for this page: one block at a time,
    /// through the decoded cache when the handle has one.
    fn open(&self) -> Open<'a> {
        Open {
            store: self.store,
            namespace: self.namespace,
            dim: self.dim,
            budget: self.budget,
            policy: self.policy,
            decoded: self.decoded,
            after: self.after,
        }
    }
}

/// Run the scan over `manifest`.
///
/// # Errors
///
/// The errors of reading segments and the tail (a missing segment, or a fenced
/// WAL object, is what the caller reloads the manifest for), and
/// [`ItzmaError::ReadBudgetExceeded`].
pub(crate) async fn run<'a>(
    scan: &Scan<'a>,
    manifest: &'a Manifest,
    tail: impl Future<Output = Result<Arc<Layer>>>,
) -> Result<FilterPage> {
    let mut merge = Merge::open(&scan.open(), &manifest.live_segments, tail).await?;
    let mut rows: Vec<Row> = Vec::new();
    while let Some(winner) = merge.next().await? {
        let attrs = winner.attrs;
        if scan.predicate.matches(attrs.as_ref()) {
            rows.push(Row {
                id: winner.id,
                attrs,
            });
            if rows.len() == scan.limit {
                let next = rows.last().map(|row| row.id.clone());
                return Ok(FilterPage { rows, next });
            }
        }
    }
    Ok(FilterPage { rows, next: None })
}
