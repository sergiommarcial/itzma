//! Predicates arrive as the core `Predicate`'s serde form, built by
//! `itzma/predicate.py`. `py-itzma/tests/predicate_cases.json` pins it.

use itzma_core::Predicate;

pub(crate) fn parse(json: &str) -> Result<Predicate, String> {
    serde_json::from_str(json).map_err(|e| format!("filter: {e}"))
}

#[cfg(test)]
mod tests {
    use serde_json::{Map, Value};

    use super::*;

    #[test]
    fn every_contract_case_round_trips() {
        let cases: Map<String, Value> =
            serde_json::from_str(include_str!("../../../py-itzma/tests/predicate_cases.json"))
                .expect("cases");
        assert!(cases.len() >= 20);
        for (name, case) in cases {
            let parsed = parse(&case.to_string()).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(
                serde_json::to_value(&parsed).expect("serialize"),
                case,
                "{name}"
            );
        }
    }
}
