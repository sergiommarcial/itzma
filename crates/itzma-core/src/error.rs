//! The error type shared by every module in the crate.

use thiserror::Error;

/// Everything that can go wrong in `itzma-core`.
///
/// The enum is `#[non_exhaustive]`, so downstream `match` expressions need a
/// wildcard arm. New variants can then be added without a breaking change.
///
/// Input is validated before any storage call, so the input-shaped variants
/// ([`InvalidNamespaceId`](Self::InvalidNamespaceId),
/// [`InvalidSchema`](Self::InvalidSchema),
/// [`DimensionMismatch`](Self::DimensionMismatch),
/// [`InvalidVector`](Self::InvalidVector), [`InvalidId`](Self::InvalidId),
/// [`InvalidQuery`](Self::InvalidQuery), [`EmptyBatch`](Self::EmptyBatch))
/// never leave a partial write behind.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ItzmaError {
    /// The object store failed. Passed through unchanged, and never retried
    /// except inside the two contention loops ([`WriteContention`] and
    /// [`ManifestContention`]).
    ///
    /// [`WriteContention`]: Self::WriteContention
    /// [`ManifestContention`]: Self::ManifestContention
    #[error("storage error: {0}")]
    Storage(#[from] object_store::Error),

    /// An object was fetched but its JSON could not be decoded (or, on the
    /// write side, an entry could not be encoded).
    #[error("failed to decode {path}: {source}")]
    Serde {
        /// Object key of the offending object, or a short label such as
        /// `"wal entry"` when encoding something that has no key yet.
        path: String,
        /// The underlying `serde_json` error.
        #[source]
        source: serde_json::Error,
    },

    /// The namespace id is empty, longer than 64 bytes, or contains anything
    /// outside `[a-z0-9_-]`. Keeping `/` out of ids keeps namespaces from
    /// nesting into each other's prefixes.
    #[error("invalid namespace id {0:?}: must match [a-z0-9_-]{{1,64}}")]
    InvalidNamespaceId(String),

    /// `Namespace::create` was called for an id whose manifest already exists.
    #[error("namespace {0:?} already exists")]
    NamespaceExists(String),

    /// `Namespace::open` was called for an id with no manifest.
    #[error("namespace {0:?} not found")]
    NamespaceNotFound(String),

    /// `Namespace::recover` was called for an id whose manifest exists, or that
    /// another recovery wrote first. Nothing was changed. The namespace is
    /// there to [`open`](crate::Namespace::open).
    #[error("namespace {0:?} already has a manifest")]
    ManifestExists(String),

    /// `Namespace::recover` cannot rebuild the manifest from what the store
    /// holds. Nothing was written. `reason` says what is missing, and `keys` are
    /// the objects to put back from a replica, a versioned bucket or a backup
    ///. Once they are back, running `recover` again can succeed
    ///.
    #[error("cannot rebuild the manifest: {reason}; restore {keys:?} and run recover again")]
    Unrecoverable {
        /// What has no readable copy.
        reason: String,
        /// Object keys that would fill the gap, at most a handful.
        keys: Vec<String>,
    },

    /// The schema passed to `Namespace::create` cannot be used. Today that
    /// means `dim == 0`.
    #[error("invalid schema: {0}")]
    InvalidSchema(&'static str),

    /// A vector's length differs from the namespace's `dim`. Raised for
    /// documents on write and for the query vector on read.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// The namespace's `dim`.
        expected: usize,
        /// The length of the vector that was passed in.
        got: usize,
    },

    /// A vector has a NaN or infinite component, or is all zeros under the
    /// cosine metric (which has no direction to compare).
    #[error("invalid vector: {0}")]
    InvalidVector(&'static str),

    /// A document's attributes, or a predicate, name a field the namespace's
    /// attribute schema does not declare.
    #[error("unknown attribute field {field:?}")]
    UnknownAttr {
        /// The field name as given.
        field: String,
    },

    /// A value or an operator does not fit the declared type of a field.
    /// `expected` is what the schema or the operator needs, `found` what was
    /// given.
    #[error("attribute {field:?}: expected {expected}, found {found}")]
    AttrTypeMismatch {
        /// The field name.
        field: String,
        /// The type the schema or the operator needs.
        expected: &'static str,
        /// The type that was found.
        found: &'static str,
    },

    /// A document's `attrs` are not acceptable under the namespace's attribute
    /// schema for a reason that is not one field's type: not a JSON object, or
    /// a set with too many elements.
    #[error("invalid attributes: {0}")]
    InvalidAttrs(&'static str),

    /// A [`DocColumns`](crate::DocColumns) batch is malformed: offsets that do
    /// not start at zero, go down or overrun their buffer, a column that does
    /// not mark every row, text that is not UTF-8, a repeated or empty column
    /// name, or batches of one stream that do not share their columns.
    #[error("invalid columns: {0}")]
    InvalidColumns(&'static str),

    /// [`QueryOptions`](crate::QueryOptions) that cannot be run: a shortlist
    /// factor or a probe count of zero.
    #[error("invalid query options: {0}")]
    InvalidQuery(&'static str),

    /// A process-wide setting was refused: for instance sizing the compute
    /// pools with [`configure_cpu`](crate::configure_cpu) after they started,
    /// or with zero threads.
    #[error("invalid configuration: {0}")]
    InvalidConfig(&'static str),

    /// `upsert` or `delete` was called with nothing to write.
    #[error("empty batch")]
    EmptyBatch,

    /// A WAL append lost the put-if-absent race 32 times in a row and gave up.
    /// Retrying the whole call is safe: nothing was written.
    #[error("write contention: gave up after {attempts} attempts")]
    WriteContention {
        /// How many sequence numbers were tried.
        attempts: u32,
    },

    /// A manifest compare-and-swap lost the race 32 times in a row and gave up.
    /// Compaction's publish step is bound by the same cap, and so is raising a
    /// segment's generation mark before an index build writes its object
    ///; nothing was written then either.
    #[error("manifest contention: gave up after {attempts} attempts")]
    ManifestContention {
        /// How many read-modify-write rounds were tried.
        attempts: u32,
    },

    /// The WAL is not a contiguous run of sequence numbers. The write path
    /// never leaves a hole, so this means an object was deleted by hand or
    /// there is a bug.
    #[error("wal gap: expected seq {expected}, found {found}")]
    WalGap {
        /// The sequence number that should have come next.
        expected: u64,
        /// The sequence number that was there instead.
        found: u64,
    },

    /// An object carries a `format_version` this build does not understand.
    #[error("unsupported format version {version}")]
    UnsupportedFormat {
        /// The version found in the object.
        version: u32,
    },

    /// The namespace's manifest says its WAL may hold objects in a format newer
    /// than this build reads. Nothing was read or written. Upgrade the build,
    /// then retry. Raised for every operation on the namespace, since each one
    /// loads the manifest first.
    #[error(
        "namespace needs WAL format {found}, this build reads up to {supported}: upgrade this build"
    )]
    WalFormatTooNew {
        /// The `wal_format` in the manifest.
        found: u32,
        /// The newest WAL format this build reads.
        supported: u32,
    },

    /// The namespace's manifest says its segments may be in a format newer than
    /// this build reads. Nothing was read or written. Upgrade the build, then
    /// retry. Raised for every operation on the namespace, like
    /// [`ItzmaError::WalFormatTooNew`].
    #[error(
        "namespace has segment format {found}, this build reads up to {supported}: upgrade this build"
    )]
    SegmentFormatTooNew {
        /// The `segment_format` in the manifest.
        found: u32,
        /// The newest segment format this build reads.
        supported: u32,
    },

    /// The namespace's WAL format is lower than the operation needs. A keyed
    /// write needs level 3, fencing WAL objects in `Namespace::gc` needs level
    /// 4, and a namespace created by an older build is at 2. Nothing was
    /// written or changed. Once every build that touches the namespace reads
    /// the newer format, raise it with `Namespace::upgrade_wal_format`
    ///.
    #[error(
        "namespace is at WAL format {have}, this operation needs {need}: upgrade the namespace with upgrade_wal_format({need})"
    )]
    WalFormatTooOld {
        /// The `wal_format` in the manifest.
        have: u32,
        /// The level the operation needs.
        need: u32,
    },

    /// The namespace's segment format is lower than the operation needs.
    /// `Namespace::index` needs level 2, because an index is built for the
    /// block-addressable layout, and a namespace created by an older build is
    /// at level one. Nothing was written or changed. Once every build that
    /// touches the namespace reads version 2, raise it with
    /// `Namespace::upgrade_segment_format`.
    #[error(
        "namespace is at segment format {have}, this operation needs {need}: upgrade the namespace with upgrade_segment_format({need})"
    )]
    SegmentFormatTooOld {
        /// The `segment_format` in the manifest.
        have: u32,
        /// The level the operation needs.
        need: u32,
    },

    /// A segment object cannot be used: it is truncated, has the wrong
    /// magic, disagrees with its own counts, has ids out of order, or is
    /// missing although the manifest lists it. Also raised when a segment's
    /// dimension differs from the namespace schema.
    #[error("corrupt segment {path}: {reason}")]
    CorruptSegment {
        /// Object key of the segment.
        path: String,
        /// What is wrong with it.
        reason: String,
    },

    /// An index object cannot be used: it is truncated, has the wrong magic, a
    /// kind this build does not know, counts, offsets or a head length that
    /// disagree with each other, values that are not finite, or a row count or
    /// data size that differs from the segment it was built for.
    /// A query never falls back to a flat scan over one (`exact` skips indexes
    /// altogether), and `index` with `rebuild` replaces it.
    #[error("corrupt index {path}: {reason}")]
    CorruptIndex {
        /// Object key of the index, or `"(encoding)"` on the write side.
        path: String,
        /// What is wrong with it.
        reason: String,
    },

    /// A binary WAL object (format version 2) cannot be used: it is
    /// truncated, names an unknown codec, disagrees with its own lengths or
    /// counts, holds a ZSTD frame that does not match its header, has invalid
    /// UTF-8 or JSON inside, or has trailing bytes. Also raised on the write
    /// side when a length cannot be encoded. An object that is not binary is
    /// decoded as JSON and fails with [`Serde`](Self::Serde) instead.
    #[error("corrupt wal object {path}: {reason}")]
    CorruptWal {
        /// Object key of the WAL object, or `"(encoding)"` on the write side.
        path: String,
        /// What is wrong with it.
        reason: String,
    },

    /// An idempotency key is empty or longer than 128 bytes. Raised before
    /// anything is written.
    #[error("invalid idempotency key: {0}")]
    InvalidKey(&'static str),

    /// A document id (on `upsert`) or a delete id is empty or longer than
    /// [`document::MAX_ID_BYTES`](crate::document::MAX_ID_BYTES) bytes.
    /// Raised before anything is written.
    #[error("invalid id: {0}")]
    InvalidId(&'static str),

    /// A WAL object already carries this idempotency key but holds a batch of
    /// a different size, so the key is being reused for a different write.
    /// Nothing was written. A key names one batch.
    #[error("idempotency key {key:?} was already used for a different batch")]
    KeyConflict {
        /// The key.
        key: String,
    },

    /// A write batch is bigger than the writer limit,
    /// [`MAX_BODY_BYTES`](crate::wal::MAX_BODY_BYTES). Nothing was built or
    /// written. Split the load into several batches; each is atomic, the split
    /// as a whole is not.
    #[error("batch of {bytes} bytes is over the {max} byte limit")]
    BatchTooLarge {
        /// The exact size the batch would have, before compression.
        bytes: u64,
        /// The writer limit.
        max: u64,
    },

    /// A query or a compaction run would read more than the handle's read
    /// budget. Nothing was written. `needed` is the total of the
    /// objects the read had asked for when it hit the limit, this one
    /// included. Compact to shorten the WAL tail, or raise the budget with
    /// [`Namespace::with_read_budget`](crate::Namespace::with_read_budget).
    #[error("reading {path} needs {needed} bytes, over the {max} byte read budget")]
    ReadBudgetExceeded {
        /// Object key of the object that went over.
        path: String,
        /// The running total, including that object.
        needed: u64,
        /// The read budget.
        max: u64,
    },

    /// The manifest's segment list breaks the invariant: ids that do not
    /// parse, ranges that do not start at 1 or do not follow one another, or a
    /// last range that does not end at `wal_folded_through`. Also raised when
    /// `schema.json` holds a different schema from the manifest.
    #[error("corrupt manifest: {0}")]
    CorruptManifest(String),
}

/// `std::result::Result` with [`ItzmaError`] as the error type.
pub type Result<T> = std::result::Result<T, ItzmaError>;
