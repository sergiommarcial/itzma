//! Raising the Python exception classes in `itzma.exceptions`.
//!
//! The classes are defined in Python, so they carry type hints and
//! docstrings, and this module imports them by name when it raises one.

use itzma_core::ItzmaError as Core;
use pyo3::prelude::*;

/// Raise `itzma.exceptions.<class>` with `message`.
pub(crate) fn raise(class: &str, message: String) -> PyErr {
    Python::attach(|py| {
        let made = py
            .import("itzma.exceptions")
            .and_then(|module| module.getattr(class))
            .and_then(|class| class.call1((message.clone(),)));
        match made {
            Ok(instance) => PyErr::from_value(instance),
            Err(import_failed) => import_failed,
        }
    })
}

/// The base `ItzmaError`, for failures that are not a core error.
pub(crate) fn itzma_error(message: String) -> PyErr {
    raise(Kind::Other.name(), message)
}

/// A Python exception class in `itzma.exceptions`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    InvalidInput,
    InvalidVector,
    InvalidAttrs,
    InvalidQuery,
    BatchTooLarge,
    NamespaceExists,
    NamespaceNotFound,
    ReadBudgetExceeded,
    KeyConflict,
    Contention,
    FormatTooNew,
    FormatTooOld,
    Corrupt,
    StoreError,
    /// A core variant this build does not know (`ItzmaError` is
    /// non-exhaustive). Raised as the base class.
    Other,
}

impl Kind {
    pub(crate) const ALL: [Kind; 15] = [
        Kind::InvalidInput,
        Kind::InvalidVector,
        Kind::InvalidAttrs,
        Kind::InvalidQuery,
        Kind::BatchTooLarge,
        Kind::NamespaceExists,
        Kind::NamespaceNotFound,
        Kind::ReadBudgetExceeded,
        Kind::KeyConflict,
        Kind::Contention,
        Kind::FormatTooNew,
        Kind::FormatTooOld,
        Kind::Corrupt,
        Kind::StoreError,
        Kind::Other,
    ];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Kind::InvalidInput => "InvalidInput",
            Kind::InvalidVector => "InvalidVector",
            Kind::InvalidAttrs => "InvalidAttrs",
            Kind::InvalidQuery => "InvalidQuery",
            Kind::BatchTooLarge => "BatchTooLarge",
            Kind::NamespaceExists => "NamespaceExists",
            Kind::NamespaceNotFound => "NamespaceNotFound",
            Kind::ReadBudgetExceeded => "ReadBudgetExceeded",
            Kind::KeyConflict => "KeyConflict",
            Kind::Contention => "Contention",
            Kind::FormatTooNew => "FormatTooNew",
            Kind::FormatTooOld => "FormatTooOld",
            Kind::Corrupt => "Corrupt",
            Kind::StoreError => "StoreError",
            Kind::Other => "ItzmaError",
        }
    }
}

/// An attribute set on the raised exception.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Field {
    Text(String),
    Number(u64),
    Texts(Vec<String>),
}

type Fields = Vec<(&'static str, Field)>;

fn corrupt(kind: &str, path: &str, reason: &str) -> (Kind, Fields) {
    (
        Kind::Corrupt,
        vec![
            ("kind", Field::Text(kind.into())),
            ("path", Field::Text(path.into())),
            ("reason", Field::Text(reason.into())),
        ],
    )
}

/// Which class a core error raises, and the fields it carries.
pub(crate) fn classify(e: &Core) -> (Kind, Fields) {
    use Field::{Number, Text, Texts};
    match e {
        Core::Storage(_) => (Kind::StoreError, vec![]),
        Core::Serde { path, source } => corrupt("serde", path, &source.to_string()),
        Core::InvalidNamespaceId(id) => (Kind::InvalidInput, vec![("namespace", Text(id.clone()))]),
        Core::NamespaceExists(id) | Core::ManifestExists(id) => {
            (Kind::NamespaceExists, vec![("namespace", Text(id.clone()))])
        }
        Core::NamespaceNotFound(id) => (
            Kind::NamespaceNotFound,
            vec![("namespace", Text(id.clone()))],
        ),
        Core::Unrecoverable { reason, keys } => {
            let (kind, mut fields) = corrupt("unrecoverable", "", reason);
            fields.push(("keys", Texts(keys.clone())));
            (kind, fields)
        }
        Core::InvalidSchema(_) | Core::EmptyBatch | Core::InvalidKey(_) => {
            (Kind::InvalidInput, vec![])
        }
        Core::DimensionMismatch { expected, got } => (
            Kind::InvalidVector,
            vec![
                ("expected", Number(*expected as u64)),
                ("got", Number(*got as u64)),
            ],
        ),
        Core::InvalidVector(_) => (Kind::InvalidVector, vec![]),
        Core::UnknownAttr { field } => (Kind::InvalidAttrs, vec![("field", Text(field.clone()))]),
        Core::AttrTypeMismatch {
            field,
            expected,
            found,
        } => (
            Kind::InvalidAttrs,
            vec![
                ("field", Text(field.clone())),
                ("expected", Text((*expected).into())),
                ("found", Text((*found).into())),
            ],
        ),
        Core::InvalidAttrs(_) => (Kind::InvalidAttrs, vec![]),
        Core::InvalidColumns(_) => (Kind::InvalidInput, vec![]),
        Core::InvalidQuery(_) => (Kind::InvalidQuery, vec![]),
        Core::InvalidConfig(_) => (Kind::InvalidInput, vec![]),
        Core::WriteContention { attempts } | Core::ManifestContention { attempts } => (
            Kind::Contention,
            vec![("attempts", Number(u64::from(*attempts)))],
        ),
        Core::WalGap { .. } => corrupt("wal", "", &e.to_string()),
        Core::UnsupportedFormat { version } => (
            Kind::FormatTooNew,
            vec![("found", Number(u64::from(*version)))],
        ),
        Core::WalFormatTooNew { found, supported }
        | Core::SegmentFormatTooNew { found, supported } => (
            Kind::FormatTooNew,
            vec![
                ("found", Number(u64::from(*found))),
                ("supported", Number(u64::from(*supported))),
            ],
        ),
        Core::WalFormatTooOld { have, need } | Core::SegmentFormatTooOld { have, need } => (
            Kind::FormatTooOld,
            vec![
                ("have", Number(u64::from(*have))),
                ("need", Number(u64::from(*need))),
            ],
        ),
        Core::CorruptSegment { path, reason } => corrupt("segment", path, reason),
        Core::CorruptIndex { path, reason } => corrupt("index", path, reason),
        Core::CorruptSparseIndex { path, reason } => corrupt("sparse_index", path, reason),
        Core::CorruptWal { path, reason } => corrupt("wal", path, reason),
        Core::CorruptManifest(reason) => corrupt("manifest", "", reason),
        Core::KeyConflict { key } => (Kind::KeyConflict, vec![("key", Text(key.clone()))]),
        Core::BatchTooLarge { bytes, max } => (
            Kind::BatchTooLarge,
            vec![("bytes", Number(*bytes)), ("max", Number(*max))],
        ),
        Core::ReadBudgetExceeded { path, needed, max } => (
            Kind::ReadBudgetExceeded,
            vec![
                ("path", Text(path.clone())),
                ("needed", Number(*needed)),
                ("max", Number(*max)),
            ],
        ),
        _ => (Kind::Other, vec![]),
    }
}

fn set_fields(err: PyErr, fields: Fields) -> PyErr {
    Python::attach(|py| {
        let value = err.value(py).clone();
        for (name, field) in fields {
            let set = match field {
                Field::Text(s) => value.setattr(name, s),
                Field::Number(n) => value.setattr(name, n),
                Field::Texts(v) => value.setattr(name, v),
            };
            if let Err(failed) = set {
                return failed;
            }
        }
        err
    })
}

/// The Python exception for a core error, with its fields as attributes.
pub(crate) fn to_py(e: Core) -> PyErr {
    let (kind, fields) = classify(&e);
    set_fields(raise(kind.name(), e.to_string()), fields)
}

/// `InvalidInput`, for arguments the bridge refuses before calling the core.
pub(crate) fn input(message: String) -> PyErr {
    raise(Kind::InvalidInput.name(), message)
}

/// Record how far a split write got before it failed.
pub(crate) fn with_last_seq(err: PyErr, last_seq: Option<u64>) -> PyErr {
    Python::attach(|py| match err.value(py).setattr("last_seq", last_seq) {
        Ok(()) => err,
        Err(failed) => failed,
    })
}

/// Record how far a streaming write got: the first and last seq
/// that landed and how many rows. An unkeyed load resumes from
/// `rows_written`; a keyed one retries with `retry_from=first_seq`.
pub(crate) fn with_progress(
    err: PyErr,
    first_seq: Option<u64>,
    last_seq: Option<u64>,
    rows_written: u64,
) -> PyErr {
    let err = with_last_seq(err, last_seq);
    let err = Python::attach(|py| match err.value(py).setattr("first_seq", first_seq) {
        Ok(()) => err,
        Err(failed) => failed,
    });
    Python::attach(
        |py| match err.value(py).setattr("rows_written", rows_written) {
            Ok(()) => err,
            Err(failed) => failed,
        },
    )
}

/// The class names `to_py` can raise; the Python tests check each exists.
#[pyfunction]
pub(crate) fn exception_names() -> Vec<&'static str> {
    Kind::ALL.iter().map(|k| k.name()).collect()
}

#[cfg(test)]
mod tests {
    use itzma_core::ItzmaError as Core;

    use super::*;

    fn store_error() -> object_store::Error {
        object_store::Error::Generic {
            store: "test",
            source: "boom".into(),
        }
    }

    fn serde_error() -> serde_json::Error {
        serde_json::from_str::<u8>("x").expect_err("not a number")
    }

    /// One value of every variant `itzma-core` has today. A new variant must
    /// be added here and to `classify`.
    fn every_variant() -> Vec<Core> {
        vec![
            Core::Storage(store_error()),
            Core::Serde {
                path: "p".into(),
                source: serde_error(),
            },
            Core::InvalidNamespaceId("X".into()),
            Core::NamespaceExists("a".into()),
            Core::NamespaceNotFound("a".into()),
            Core::ManifestExists("a".into()),
            Core::Unrecoverable {
                reason: "r".into(),
                keys: vec!["k".into()],
            },
            Core::InvalidSchema("s"),
            Core::DimensionMismatch {
                expected: 4,
                got: 3,
            },
            Core::InvalidVector("v"),
            Core::UnknownAttr { field: "f".into() },
            Core::AttrTypeMismatch {
                field: "f".into(),
                expected: "number",
                found: "string",
            },
            Core::InvalidAttrs("a"),
            Core::InvalidColumns("c"),
            Core::InvalidQuery("q"),
            Core::InvalidConfig("c"),
            Core::EmptyBatch,
            Core::WriteContention { attempts: 32 },
            Core::ManifestContention { attempts: 32 },
            Core::WalGap {
                expected: 2,
                found: 3,
            },
            Core::UnsupportedFormat { version: 9 },
            Core::WalFormatTooNew {
                found: 9,
                supported: 5,
            },
            Core::SegmentFormatTooNew {
                found: 9,
                supported: 2,
            },
            Core::WalFormatTooOld { have: 2, need: 3 },
            Core::SegmentFormatTooOld { have: 1, need: 2 },
            Core::CorruptSegment {
                path: "p".into(),
                reason: "r".into(),
            },
            Core::CorruptIndex {
                path: "p".into(),
                reason: "r".into(),
            },
            Core::CorruptSparseIndex {
                path: "p".into(),
                reason: "r".into(),
            },
            Core::CorruptWal {
                path: "p".into(),
                reason: "r".into(),
            },
            Core::InvalidKey("k"),
            Core::KeyConflict { key: "k".into() },
            Core::BatchTooLarge { bytes: 10, max: 5 },
            Core::ReadBudgetExceeded {
                path: "p".into(),
                needed: 10,
                max: 5,
            },
            Core::CorruptManifest("m".into()),
        ]
    }

    #[test]
    fn no_variant_falls_through_to_the_base_class() {
        for e in every_variant() {
            let (kind, _) = classify(&e);
            assert_ne!(kind, Kind::Other, "{e:?} is not mapped");
        }
    }

    #[test]
    fn fields_carry_over() {
        let (kind, fields) = classify(&Core::ReadBudgetExceeded {
            path: "ns/segments/x/data.bin".into(),
            needed: 10,
            max: 5,
        });
        assert_eq!(kind, Kind::ReadBudgetExceeded);
        assert_eq!(
            fields,
            vec![
                ("path", Field::Text("ns/segments/x/data.bin".into())),
                ("needed", Field::Number(10)),
                ("max", Field::Number(5)),
            ]
        );

        let (kind, fields) = classify(&Core::DimensionMismatch {
            expected: 4,
            got: 3,
        });
        assert_eq!(kind, Kind::InvalidVector);
        assert_eq!(
            fields,
            vec![("expected", Field::Number(4)), ("got", Field::Number(3))]
        );

        let (kind, fields) = classify(&Core::CorruptIndex {
            path: "p".into(),
            reason: "r".into(),
        });
        assert_eq!(kind, Kind::Corrupt);
        assert_eq!(
            fields,
            vec![
                ("kind", Field::Text("index".into())),
                ("path", Field::Text("p".into())),
                ("reason", Field::Text("r".into())),
            ]
        );
    }

    #[test]
    fn groups_follow_the_spec_table() {
        let expect = [
            (Core::EmptyBatch, Kind::InvalidInput),
            (Core::InvalidKey("k"), Kind::InvalidInput),
            (Core::InvalidColumns("c"), Kind::InvalidInput),
            (Core::UnknownAttr { field: "f".into() }, Kind::InvalidAttrs),
            (Core::ManifestExists("a".into()), Kind::NamespaceExists),
            (Core::ManifestContention { attempts: 1 }, Kind::Contention),
            (Core::UnsupportedFormat { version: 9 }, Kind::FormatTooNew),
            (
                Core::SegmentFormatTooOld { have: 1, need: 2 },
                Kind::FormatTooOld,
            ),
            (
                Core::WalGap {
                    expected: 2,
                    found: 3,
                },
                Kind::Corrupt,
            ),
            (Core::Storage(store_error()), Kind::StoreError),
        ];
        for (e, kind) in expect {
            assert_eq!(classify(&e).0, kind, "{e:?}");
        }
    }

    #[test]
    fn every_kind_has_a_distinct_name() {
        let mut names: Vec<_> = Kind::ALL.iter().map(|k| k.name()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), Kind::ALL.len());
    }
}
