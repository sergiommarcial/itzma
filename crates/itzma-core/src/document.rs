//! The data model: documents, WAL operations, and the per-namespace schema.

use serde::{Deserialize, Serialize};

use crate::attrs::AttrSchema;
use crate::error::{ItzmaError, Result};

/// One stored item: an id, a vector, and optional free-form JSON.
///
/// The id is the identity. Upserting a document whose id already exists
/// replaces it wholesale, vector and attributes together.
///
/// # JSON form
///
/// This is what appears inside WAL objects. `attrs` is left out when `None`:
///
/// ```json
/// {"id": "a", "vector": [0.1, 0.2], "attrs": {"title": "hello"}, "sparse": [[3, 0.9]]}
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Document {
    /// Caller-chosen identifier, unique within a namespace. Any string.
    pub id: String,
    /// Embedding. Its length must equal the namespace's `dim`, and every
    /// component must be finite.
    pub vector: Vec<f32>,
    /// Arbitrary JSON stored next to the vector and returned on every
    /// [`Hit`](crate::Hit). It is checked against the namespace's attribute
    /// schema on write.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attrs: Option<serde_json::Value>,
    /// An optional sparse vector, searched by
    /// [`Namespace::query_sparse`](crate::Namespace::query_sparse). A
    /// namespace does not declare sparse support up front — some documents
    /// carry one, some don't, the same shape `attrs` has. `None` on a
    /// document written before this field existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sparse: Option<SparseVector>,
}

/// A sparse vector: term id and weight pairs, in the style SPLADE and
/// similar learned-sparse models produce. Itzma assigns no meaning to a term
/// id — it is the caller's own vocabulary (a tokenizer's ids, typically) —
/// and keeps no string dictionary. See
/// [`Namespace::query_sparse`](crate::Namespace::query_sparse).
pub type SparseVector = Vec<(u32, f32)>;

/// The most terms one document's sparse vector may hold. The term count is a
/// `u16` in the WAL object and in `sparse.bin`, and a longer vector is refused
/// by `upsert` with [`ItzmaError::InvalidVector`] before anything is written.
pub const MAX_SPARSE_TERMS: usize = u16::MAX as usize;

/// The most bytes an id may be: a document's `id` on `upsert`, or an id
/// passed to `delete`. An id is a free-form string with no other structure
/// (a UUID, a natural key, whatever the caller already has), but it is
/// stored in every WAL object and segment that touches it, so an unbounded
/// one lets a single write blow up an object's size. 512 bytes covers any
/// realistic key (a UUID is 36) with room to spare.
pub const MAX_ID_BYTES: usize = 512;

/// One mutation recorded in the write-ahead log.
///
/// Ops replay in log order, and within a WAL object in list order. The last
/// op touching an id wins, whether that is an upsert or a delete.
///
/// # JSON form
///
/// Externally tagged, snake_case:
///
/// ```json
/// [{"upsert": {"id": "a", "vector": [1.0, 2.0]}}, {"delete": "b"}]
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WalOp {
    /// Insert the document, or replace the one with the same id.
    Upsert(Document),
    /// Remove the document with this id. Deleting an id that does not exist
    /// is not an error.
    Delete(String),
}

/// How distance between two vectors is measured.
///
/// The choice is made once per namespace. Every metric is reported so that
/// **lower means closer**; see [`crate::search::distance`] for the exact
/// formulas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    /// Angle between vectors, reported as `1 - cosine similarity`. Ignores
    /// vector length. Zero vectors are rejected on write.
    Cosine,
    /// Squared Euclidean distance. The square root is skipped because it does
    /// not change the ordering.
    L2,
    /// Inner product, negated so that a larger dot product sorts first.
    Dot,
}

/// The shape every document in a namespace must have.
///
/// Fixed when the namespace is created and stored in its manifest, and again in
/// `schema.json`, which is written once and never changed. It cannot be
/// changed afterwards; make a new namespace instead.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Schema {
    /// Number of components in every vector. Must be at least 1.
    pub dim: usize,
    /// Distance metric used by queries.
    pub metric: Metric,
    /// The typed attribute fields. Empty means the namespace takes
    /// no attributes. A schemaless value has none declared and
    /// stays free-form; nothing public can build that state.
    #[serde(
        default = "AttrSchema::free_form",
        skip_serializing_if = "AttrSchema::is_free_form"
    )]
    pub attrs: AttrSchema,
}

impl Schema {
    /// A schema with no attribute fields: documents may carry no `attrs`.
    /// Declare fields with [`with_attrs`](Self::with_attrs).
    pub const fn new(dim: usize, metric: Metric) -> Self {
        Self {
            dim,
            metric,
            attrs: AttrSchema::new(),
        }
    }

    /// This schema with the attribute fields `attrs`.
    #[must_use]
    pub fn with_attrs(mut self, attrs: AttrSchema) -> Self {
        self.attrs = attrs;
        self
    }

    /// Check a vector against this schema without touching storage.
    ///
    /// Used on every document written and on every query vector.
    ///
    /// # Errors
    ///
    /// - [`ItzmaError::DimensionMismatch`] if `vector.len() != self.dim`.
    /// - [`ItzmaError::InvalidVector`] if any component is NaN or infinite, or
    ///   if the metric is [`Metric::Cosine`] and every component is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use itzma_core::{Metric, Schema};
    ///
    /// let schema = Schema::new(2, Metric::Cosine);
    /// assert!(schema.validate(&[1.0, 0.5]).is_ok());
    /// assert!(schema.validate(&[1.0]).is_err());
    /// assert!(schema.validate(&[0.0, 0.0]).is_err());
    /// ```
    pub fn validate(&self, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dim {
            return Err(ItzmaError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        if vector.iter().any(|x| !x.is_finite()) {
            return Err(ItzmaError::InvalidVector("non-finite component"));
        }
        if self.metric == Metric::Cosine && vector.iter().all(|x| *x == 0.0) {
            return Err(ItzmaError::InvalidVector("zero vector under cosine"));
        }
        Ok(())
    }
}

/// Check an id's shape without touching storage: not empty, not longer than
/// [`MAX_ID_BYTES`]. Shared by [`Document::validate_id`] (a document's `id`
/// on `upsert`) and by `Namespace::delete`, which has no `Document` to hang
/// the check on.
///
/// # Errors
///
/// [`ItzmaError::InvalidId`] if the id is empty or over the limit.
pub(crate) fn validate_id(id: &str) -> Result<()> {
    if id.is_empty() {
        return Err(ItzmaError::InvalidId("empty"));
    }
    if id.len() > MAX_ID_BYTES {
        return Err(ItzmaError::InvalidId("too long"));
    }
    Ok(())
}

impl Document {
    /// Check this document's id without touching storage. Called on every
    /// upsert, before [`validate_sparse`](Self::validate_sparse).
    ///
    /// # Errors
    ///
    /// [`ItzmaError::InvalidId`] if the id is empty or longer than
    /// [`MAX_ID_BYTES`].
    pub fn validate_id(&self) -> Result<()> {
        validate_id(&self.id)
    }

    /// Check this document's sparse vector, if it has one, without touching
    /// storage. Called on every upsert, after
    /// [`Schema::validate`](Schema::validate).
    ///
    /// # Errors
    ///
    /// [`ItzmaError::InvalidVector`] if the same term id appears twice, a
    /// weight is NaN or infinite, or there are more than [`MAX_SPARSE_TERMS`]
    /// terms (the stored term count is a `u16`).
    pub fn validate_sparse(&self) -> Result<()> {
        match &self.sparse {
            None => Ok(()),
            Some(sparse) => validate_sparse_terms(
                sparse.iter().map(|(term, _)| *term),
                sparse.iter().map(|(_, weight)| *weight),
            ),
        }
    }
}

/// The checks of [`Document::validate_sparse`] on terms and weights given
/// apart, so a columnar batch runs exactly the same ones.
pub(crate) fn validate_sparse_terms(
    terms: impl ExactSizeIterator<Item = u32>,
    mut weights: impl Iterator<Item = f32>,
) -> Result<()> {
    if terms.len() > MAX_SPARSE_TERMS {
        return Err(ItzmaError::InvalidVector("too many sparse terms"));
    }
    if weights.any(|weight| !weight.is_finite()) {
        return Err(ItzmaError::InvalidVector("non-finite sparse weight"));
    }
    let mut seen = std::collections::HashSet::with_capacity(terms.len());
    let mut terms = terms;
    if terms.any(|term| !seen.insert(term)) {
        return Err(ItzmaError::InvalidVector(
            "duplicate term id in sparse vector",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema(metric: Metric) -> Schema {
        Schema::new(3, metric)
    }

    #[test]
    fn schema_json_carries_attrs_and_reads_old_json_as_free_form() {
        let typed = Schema::new(2, Metric::L2)
            .with_attrs(crate::AttrSchema::new().field("price", crate::AttrType::Number));
        let json = serde_json::to_string(&typed).unwrap();
        assert_eq!(
            json,
            r#"{"dim":2,"metric":"l2","attrs":{"price":"number"}}"#
        );
        assert_eq!(serde_json::from_str::<Schema>(&json).unwrap(), typed);

        let old: Schema = serde_json::from_str(r#"{"dim":2,"metric":"l2"}"#).unwrap();
        assert!(old.attrs.is_free_form());
        assert_eq!(
            serde_json::to_string(&old).unwrap(),
            r#"{"dim":2,"metric":"l2"}"#,
            "a free-form schema writes no attrs"
        );
    }

    #[test]
    fn validate_accepts_good_vector() {
        assert!(schema(Metric::L2).validate(&[1.0, 2.0, 3.0]).is_ok());
    }

    #[test]
    fn validate_rejects_wrong_dim() {
        let err = schema(Metric::L2).validate(&[1.0, 2.0]).unwrap_err();
        assert!(matches!(
            err,
            ItzmaError::DimensionMismatch {
                expected: 3,
                got: 2
            }
        ));
    }

    #[test]
    fn validate_rejects_non_finite() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let err = schema(Metric::L2).validate(&[1.0, bad, 3.0]).unwrap_err();
            assert!(matches!(err, ItzmaError::InvalidVector(_)));
        }
    }

    #[test]
    fn validate_rejects_zero_vector_only_for_cosine() {
        let zero = [0.0, 0.0, 0.0];
        assert!(matches!(
            schema(Metric::Cosine).validate(&zero),
            Err(ItzmaError::InvalidVector(_))
        ));
        assert!(schema(Metric::L2).validate(&zero).is_ok());
        assert!(schema(Metric::Dot).validate(&zero).is_ok());
    }

    #[test]
    fn wal_op_json_roundtrip() {
        let ops = vec![
            WalOp::Upsert(Document {
                id: "a".into(),
                vector: vec![0.1, -2.5, 3.0],
                attrs: Some(serde_json::json!({"k": "v"})),
                sparse: None,
            }),
            WalOp::Delete("b".into()),
        ];
        let json = serde_json::to_vec(&ops).unwrap();
        let back: Vec<WalOp> = serde_json::from_slice(&json).unwrap();
        assert_eq!(ops, back);
    }

    fn sparse_doc(id: &str, terms: &[(u32, f32)]) -> Document {
        Document {
            id: id.into(),
            vector: vec![1.0, 2.0, 3.0],
            attrs: None,
            sparse: Some(terms.to_vec()),
        }
    }

    #[test]
    fn validate_sparse_accepts_none_and_distinct_terms() {
        let mut d = sparse_doc("a", &[(3, 0.5), (10, 0.25)]);
        assert!(d.validate_sparse().is_ok());
        d.sparse = None;
        assert!(d.validate_sparse().is_ok());
    }

    #[test]
    fn validate_sparse_rejects_a_duplicate_term_id() {
        let d = sparse_doc("a", &[(3, 0.5), (3, 0.25)]);
        let err = d.validate_sparse().unwrap_err();
        assert!(matches!(err, ItzmaError::InvalidVector(_)));
    }

    #[test]
    fn validate_sparse_allows_exactly_the_most_terms_and_refuses_one_more() {
        let terms = |n: u32| (0..n).map(|t| (t, 1.0)).collect::<Vec<_>>();
        let at_limit = sparse_doc("a", &terms(MAX_SPARSE_TERMS as u32));
        assert!(at_limit.validate_sparse().is_ok());
        let over = sparse_doc("a", &terms(MAX_SPARSE_TERMS as u32 + 1));
        assert!(matches!(
            over.validate_sparse(),
            Err(ItzmaError::InvalidVector("too many sparse terms"))
        ));
    }

    #[test]
    fn validate_sparse_rejects_a_non_finite_weight() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let d = sparse_doc("a", &[(1, 0.5), (2, bad)]);
            assert!(matches!(
                d.validate_sparse(),
                Err(ItzmaError::InvalidVector(_))
            ));
        }
    }

    fn doc_with_id(id: &str) -> Document {
        Document {
            id: id.into(),
            vector: vec![1.0, 2.0, 3.0],
            attrs: None,
            sparse: None,
        }
    }

    #[test]
    fn validate_id_rejects_empty() {
        let err = doc_with_id("").validate_id().unwrap_err();
        assert!(matches!(err, ItzmaError::InvalidId(_)));
    }

    #[test]
    fn validate_id_allows_exactly_the_limit_and_refuses_one_more() {
        let at_limit = "a".repeat(MAX_ID_BYTES);
        assert!(doc_with_id(&at_limit).validate_id().is_ok());

        let over = "a".repeat(MAX_ID_BYTES + 1);
        assert!(matches!(
            doc_with_id(&over).validate_id(),
            Err(ItzmaError::InvalidId("too long"))
        ));
    }

    #[test]
    fn validate_id_accepts_a_normal_id() {
        assert!(doc_with_id("doc-1").validate_id().is_ok());
    }

    #[test]
    fn sparse_field_round_trips_and_is_omitted_when_none() {
        let with = sparse_doc("a", &[(1, 0.5)]);
        let json = serde_json::to_value(&with).unwrap();
        assert_eq!(json["sparse"], serde_json::json!([[1, 0.5]]));
        let back: Document = serde_json::from_value(json).unwrap();
        assert_eq!(back, with);

        let without = Document {
            id: "b".into(),
            vector: vec![1.0],
            attrs: None,
            sparse: None,
        };
        let json = serde_json::to_value(&without).unwrap();
        assert!(
            json.get("sparse").is_none(),
            "sparse must be omitted, not null, when None"
        );

        // A pre-existing WAL object, written before this field existed, must
        // still decode: no "sparse" key at all.
        let old = serde_json::json!({"id": "c", "vector": [1.0]});
        let decoded: Document = serde_json::from_value(old).unwrap();
        assert_eq!(decoded.sparse, None);
    }
}
