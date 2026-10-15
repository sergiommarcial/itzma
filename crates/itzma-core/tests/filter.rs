//! `Namespace::filter`: the answer equals a naive replay of the WAL
//! with the condition applied by hand, at every stage a namespace goes through.

mod common;

use common::counting::CountingStore;

use std::sync::Arc;

use common::faulty::{Fault, FaultyStore};
use common::{Lcg, cas_backends, doc, snapshot};
use itzma_core::{
    AttrSchema, AttrType, AttrValue, CompactionOptions, Compare, Document, FilterOptions,
    ItzmaError, Metric, Namespace, Predicate, Schema,
};
use object_store::{ObjectStore, ObjectStoreExt};
use serde_json::{Value, json};

const DIM: usize = 2;
const ROOMY: CompactionOptions = CompactionOptions {
    max_segments: 8,
    max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
};
const TIGHT: CompactionOptions = CompactionOptions {
    max_segments: 2,
    max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
};

fn typed() -> Schema {
    Schema::new(DIM, Metric::L2).with_attrs(
        AttrSchema::new()
            .field("active", AttrType::Bool)
            .field("price", AttrType::Number)
            .field("name", AttrType::Text)
            .field("tags", AttrType::TextSet),
    )
}

fn item(id: &str, attrs: Value) -> Document {
    let mut d = doc(id, &[1.0, 1.0]);
    d.attrs = Some(attrs);
    d
}

/// A predicate and the same condition written by hand over a stored document.
type Oracle = (Predicate, fn(&Document) -> bool);

fn price_under_20() -> Oracle {
    (Predicate::number("price", Compare::Lt, 20.0), |d| {
        d.attrs
            .as_ref()
            .and_then(|a| a["price"].as_f64())
            .is_some_and(|p| p < 20.0)
    })
}

fn active_and_tagged_red() -> Oracle {
    (
        Predicate::and(vec![
            Predicate::is_true("active"),
            Predicate::Contains {
                field: "tags".into(),
                value: AttrValue::Text("red".into()),
            },
        ]),
        |d| {
            d.attrs.as_ref().is_some_and(|a| {
                a["active"] == json!(true)
                    && a["tags"]
                        .as_array()
                        .is_some_and(|t| t.iter().any(|v| v == "red"))
            })
        },
    )
}

/// Every page of `predicate`, `limit` rows at a time, until `next` is `None`.
async fn all_pages(ns: &Namespace, predicate: &Predicate, limit: usize) -> Vec<String> {
    let mut ids = Vec::new();
    let mut after: Option<String> = None;
    loop {
        let mut options = FilterOptions::default().limit(limit);
        if let Some(a) = &after {
            options = options.after(a.clone());
        }
        let page = ns.filter(predicate, &options).await.expect("filter");
        assert!(page.rows.len() <= limit);
        ids.extend(page.rows.iter().map(|r| r.id.clone()));
        match page.next {
            Some(next) => after = Some(next),
            None => return ids,
        }
    }
}

/// What the WAL replayed by hand says matches, in id order.
async fn expected(store: &Arc<dyn ObjectStore>, matches: fn(&Document) -> bool) -> Vec<String> {
    common::assert_consistent(store.clone(), DIM, Metric::L2)
        .await
        .into_iter()
        .filter(matches)
        .map(|d| d.id)
        .collect()
}

async fn check_stage(
    ns: &Namespace,
    store: &Arc<dyn ObjectStore>,
    oracles: &[Oracle],
    stage: &str,
) {
    for (predicate, matches) in oracles {
        let want = expected(store, *matches).await;
        for limit in [1, 7, 100, 10_000] {
            assert_eq!(
                all_pages(ns, predicate, limit).await,
                want,
                "{stage}, limit {limit}"
            );
        }
    }
}

fn batch(lcg: &mut Lcg, from: usize, count: usize) -> Vec<Document> {
    (from..from + count)
        .map(|n| {
            let mut attrs = json!({
                "price": lcg.below(40) as f64,
                "active": lcg.below(2) == 0,
                "name": format!("n{}", lcg.below(5)),
            });
            if lcg.below(3) > 0 {
                let tags: Vec<&str> = ["red", "blue"][..1 + lcg.below(2) as usize].to_vec();
                attrs["tags"] = json!(tags);
            }
            item(&format!("d{n:05}"), attrs)
        })
        .collect()
}

/// The stages every filter answer is checked at. With `raise_to_3`, the
/// namespace moves to segment format 3 after its first fold, so it
/// holds a version 2 segment beside version 3 ones and then merges them.
async fn naive_replay_stages(raise_to_3: bool) {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", typed())
            .await
            .unwrap();
        let mut lcg = Lcg(7);
        let oracles = [price_under_20(), active_and_tagged_red()];

        check_stage(&ns, &b.store, &oracles, "empty").await;
        ns.upsert(batch(&mut lcg, 0, 300)).await.unwrap();
        check_stage(&ns, &b.store, &oracles, "tail only").await;
        ns.compact(&ROOMY).await.unwrap();
        check_stage(&ns, &b.store, &oracles, "folded").await;
        if raise_to_3 {
            ns.upgrade_segment_format(3).await.unwrap();
        }
        ns.upsert(batch(&mut lcg, 300, 300)).await.unwrap();
        check_stage(&ns, &b.store, &oracles, "folded plus tail").await;
        ns.compact(&ROOMY).await.unwrap();
        check_stage(&ns, &b.store, &oracles, "two segments").await;
        if raise_to_3 {
            assert_eq!(common::live_segment_versions(&*b.store).await, [2, 3]);
        }
        // Overwrites and deletes that reach back into both segments.
        ns.upsert(batch(&mut lcg, 100, 200)).await.unwrap();
        ns.delete(
            (0..300)
                .step_by(3)
                .map(|n| format!("d{n:05}"))
                .collect::<Vec<_>>(),
        )
        .await
        .unwrap();
        check_stage(&ns, &b.store, &oracles, "shadowing tail").await;
        ns.compact(&TIGHT).await.unwrap();
        check_stage(&ns, &b.store, &oracles, "three segments").await;
        ns.compact(&TIGHT).await.unwrap();
        check_stage(&ns, &b.store, &oracles, "merged").await;
        if raise_to_3 {
            let versions = common::live_segment_versions(&*b.store).await;
            assert!(versions.iter().all(|v| *v == 3), "{versions:?}");
        }
    }
}

#[tokio::test]
async fn equals_a_naive_replay_at_every_stage() {
    naive_replay_stages(false).await;
}

#[tokio::test]
async fn equals_a_naive_replay_at_every_stage_across_versions_2_and_3() {
    naive_replay_stages(true).await;
}

#[tokio::test]
async fn filter_rows_keep_none_empty_and_null_apart_at_level_3() {
    let store: Arc<dyn ObjectStore> = common::memory();
    let ns = Namespace::create(store.clone(), "docs", typed())
        .await
        .unwrap();
    ns.upgrade_segment_format(3).await.unwrap();
    let mut none = doc("a", &[1.0, 1.0]);
    none.attrs = None;
    ns.upsert(vec![
        none,
        item("b", json!({})),
        item("c", json!({"price": null})),
        item("d", json!({"price": 5.0, "name": "x"})),
    ])
    .await
    .unwrap();
    ns.compact(&ROOMY).await.unwrap();
    let not_active = Predicate::Not(Box::new(Predicate::is_true("active")));
    let page = ns
        .filter(&not_active, &FilterOptions::default())
        .await
        .unwrap();
    let got: Vec<(String, Option<Value>)> =
        page.rows.into_iter().map(|r| (r.id, r.attrs)).collect();
    assert_eq!(
        got,
        [
            ("a".to_string(), None),
            ("b".to_string(), Some(json!({}))),
            ("c".to_string(), Some(json!({"price": null}))),
            ("d".to_string(), Some(json!({"price": 5.0, "name": "x"}))),
        ]
    );
}

/// the fetch policy changes how many requests a read makes and how
/// big each one is, never what `filter` returns: the same namespace, with
/// shadowing across a folded segment and a tail, must answer identically
/// under each policy.
#[tokio::test]
async fn results_are_identical_under_both_fetch_policies() {
    for policy in [
        itzma_core::FetchPolicy::local(),
        itzma_core::FetchPolicy::object_store(),
    ] {
        let store = common::memory();
        let ns = Namespace::create(store.clone(), "docs", typed())
            .await
            .unwrap()
            .with_fetch_policy(policy);
        let mut lcg = Lcg(7);
        let oracles = [price_under_20(), active_and_tagged_red()];

        ns.upsert(batch(&mut lcg, 0, 300)).await.unwrap();
        ns.compact(&ROOMY).await.unwrap();
        ns.upsert(batch(&mut lcg, 300, 300)).await.unwrap(); // a tail over a folded segment
        check_stage(&ns, &store, &oracles, "folded plus tail").await;
    }
}

#[tokio::test]
async fn shadowing_a_newer_version_that_no_longer_matches_hides_the_old_match() {
    let store: Arc<dyn ObjectStore> = common::memory();
    let ns = Namespace::create(store.clone(), "docs", typed())
        .await
        .unwrap();
    let cheap = Predicate::number("price", Compare::Lt, 20.0);

    ns.upsert(vec![
        item("a", json!({"price": 5})),
        item("b", json!({"price": 6})),
    ])
    .await
    .unwrap();
    ns.compact(&ROOMY).await.unwrap();
    assert_eq!(all_pages(&ns, &cheap, 10).await, ["a", "b"]);

    // Newer segment: "a" gets expensive, so the old match must not come back.
    ns.upsert(vec![item("a", json!({"price": 50}))])
        .await
        .unwrap();
    ns.compact(&ROOMY).await.unwrap();
    assert_eq!(all_pages(&ns, &cheap, 10).await, ["b"]);

    // The tail: a delete hides "b", an upsert of a new id adds "c".
    ns.delete(vec!["b".into()]).await.unwrap();
    ns.upsert(vec![item("c", json!({"price": 1}))])
        .await
        .unwrap();
    assert_eq!(all_pages(&ns, &cheap, 10).await, ["c"]);

    // An id that matched, was deleted and comes back matching is there once.
    ns.upsert(vec![item("b", json!({"price": 2}))])
        .await
        .unwrap();
    assert_eq!(all_pages(&ns, &cheap, 10).await, ["b", "c"]);
}

#[tokio::test]
async fn paging_crosses_block_edges_and_never_repeats_or_skips() {
    let store: Arc<dyn ObjectStore> = common::memory();
    let ns = Namespace::create(store.clone(), "docs", typed())
        .await
        .unwrap();
    // 700 rows is three 256-row blocks in one segment.
    let docs: Vec<Document> = (0..700)
        .map(|n| item(&format!("d{n:05}"), json!({"price": (n % 2) as f64})))
        .collect();
    ns.upsert(docs).await.unwrap();
    ns.compact(&ROOMY).await.unwrap();
    let even = Predicate::number("price", Compare::Eq, 0.0);
    let want: Vec<String> = (0..700).step_by(2).map(|n| format!("d{n:05}")).collect();
    for limit in [1, 7, 255, 256, 257, 350, 351, 10_000] {
        assert_eq!(all_pages(&ns, &even, limit).await, want, "limit {limit}");
    }
}

#[tokio::test]
async fn a_cursor_past_the_last_id_gives_an_empty_final_page() {
    let ns = Namespace::create(common::memory(), "docs", typed())
        .await
        .unwrap();
    ns.upsert(vec![
        item("a", json!({"price": 1})),
        item("b", json!({"price": 1})),
    ])
    .await
    .unwrap();
    ns.compact(&ROOMY).await.unwrap();
    let has_price = Predicate::negate(Predicate::IsNull("price".into()));
    for after in ["b", "zzz"] {
        let page = ns
            .filter(&has_price, &FilterOptions::default().after(after))
            .await
            .unwrap();
        assert!(page.rows.is_empty() && page.next.is_none(), "after {after}");
    }
    // A full page that ends exactly at the last match still hands back a
    // cursor, and the next call is empty: `next` says "there may be more".
    let first = ns
        .filter(&has_price, &FilterOptions::default().limit(2))
        .await
        .unwrap();
    assert_eq!(first.rows.len(), 2);
    assert_eq!(first.next.as_deref(), Some("b"));
    let last = ns
        .filter(&has_price, &FilterOptions::default().limit(2).after("b"))
        .await
        .unwrap();
    assert!(last.rows.is_empty() && last.next.is_none());
}

#[tokio::test]
async fn rows_carry_the_id_and_the_attributes_as_stored() {
    let ns = Namespace::create(common::memory(), "docs", typed())
        .await
        .unwrap();
    let attrs = json!({"price": 3, "tags": ["b", "a", "a"]});
    ns.upsert(vec![item("a", attrs.clone()), doc("plain", &[0.0, 0.0])])
        .await
        .unwrap();
    let everything = Predicate::or(vec![
        Predicate::IsNull("price".into()),
        Predicate::IsNotNull("price".into()),
    ]);
    for stage in ["tail", "folded"] {
        let page = ns
            .filter(&everything, &FilterOptions::default())
            .await
            .unwrap();
        let rows: Vec<_> = page
            .rows
            .iter()
            .map(|r| (r.id.as_str(), r.attrs.clone()))
            .collect();
        assert_eq!(
            rows,
            [("a", Some(attrs.clone())), ("plain", None)],
            "{stage}"
        );
        ns.compact(&ROOMY).await.unwrap();
    }
}

#[tokio::test]
async fn paging_across_a_fold_between_pages_gives_every_match_once() {
    let store: Arc<dyn ObjectStore> = common::memory();
    let ns = Namespace::create(store.clone(), "docs", typed())
        .await
        .unwrap();
    let docs: Vec<Document> = (0..300)
        .map(|n| item(&format!("d{n:05}"), json!({"price": 1})))
        .collect();
    ns.upsert(docs).await.unwrap();
    let all = Predicate::number("price", Compare::Lt, 2.0);
    let mut ids = Vec::new();
    let mut after: Option<String> = None;
    let mut page_no = 0;
    loop {
        let mut options = FilterOptions::default().limit(100);
        if let Some(a) = &after {
            options = options.after(a.clone());
        }
        let page = ns.filter(&all, &options).await.unwrap();
        ids.extend(page.rows.iter().map(|r| r.id.clone()));
        // The tail becomes a segment, then a second segment is merged in,
        // between the pages.
        match page_no {
            0 => {
                ns.compact(&ROOMY).await.unwrap();
            }
            1 => {
                ns.upsert(vec![item("zzz", json!({"price": 5}))])
                    .await
                    .unwrap();
                ns.compact(&TIGHT).await.unwrap();
            }
            _ => {}
        }
        page_no += 1;
        match page.next {
            Some(next) => after = Some(next),
            None => break,
        }
    }
    let want: Vec<String> = (0..300).map(|n| format!("d{n:05}")).collect();
    assert_eq!(ids, want);
}

#[tokio::test]
async fn a_wrong_predicate_or_bad_options_is_refused_before_any_storage_call() {
    let store: Arc<dyn ObjectStore> = common::memory();
    let ns = Namespace::create(store.clone(), "docs", typed())
        .await
        .unwrap();
    ns.upsert(vec![item("a", json!({"price": 1}))])
        .await
        .unwrap();
    let before = snapshot(&*store).await;
    let counting = FaultyStore::new(store.clone());
    let as_store: Arc<dyn ObjectStore> = counting.clone();
    let ns = Namespace::open(as_store, "docs").await.unwrap();
    counting.reset_ops();

    let unknown = ns
        .filter(&Predicate::is_true("nope"), &FilterOptions::default())
        .await;
    assert!(matches!(unknown, Err(ItzmaError::UnknownAttr { .. })));
    let wrong = ns
        .filter(&Predicate::is_true("price"), &FilterOptions::default())
        .await;
    assert!(matches!(wrong, Err(ItzmaError::AttrTypeMismatch { .. })));
    let empty = ns
        .filter(&Predicate::And(vec![]), &FilterOptions::default())
        .await;
    assert!(matches!(empty, Err(ItzmaError::InvalidQuery(_))));
    let ok = Predicate::is_true("active");
    for limit in [0, 10_001] {
        let err = ns.filter(&ok, &FilterOptions::default().limit(limit)).await;
        assert!(
            matches!(err, Err(ItzmaError::InvalidQuery(_))),
            "limit {limit}"
        );
    }
    assert_eq!(counting.ops(), 0, "no storage call for a refused query");
    common::assert_untouched(&*store, &before).await;
}

#[tokio::test]
async fn a_budget_refusal_leaves_the_store_as_it_was() {
    let store: Arc<dyn ObjectStore> = common::memory();
    let ns = Namespace::create(store.clone(), "docs", typed())
        .await
        .unwrap();
    ns.upsert(batch(&mut Lcg(3), 0, 600)).await.unwrap();
    ns.compact(&ROOMY).await.unwrap();
    let before = snapshot(&*store).await;
    let tiny = Namespace::open(store.clone(), "docs")
        .await
        .unwrap()
        .with_read_budget(64);
    let err = tiny
        .filter(&price_under_20().0, &FilterOptions::default())
        .await
        .unwrap_err();
    assert!(
        matches!(err, ItzmaError::ReadBudgetExceeded { .. }),
        "{err:?}"
    );
    common::assert_untouched(&*store, &before).await;
}

#[tokio::test]
async fn a_missing_or_damaged_segment_is_corrupt_segment_and_never_a_wrong_answer() {
    let cheap = price_under_20().0;
    for damage in ["missing", "garbage"] {
        let store: Arc<dyn ObjectStore> = common::memory();
        let ns = Namespace::create(store.clone(), "docs", typed())
            .await
            .unwrap();
        ns.upsert(batch(&mut Lcg(5), 0, 300)).await.unwrap();
        ns.compact(&ROOMY).await.unwrap();
        let (m, _) = itzma_core::manifest::load(&*store, "docs").await.unwrap();
        let path = itzma_core::segment::segment_path("docs", &m.live_segments[0]);
        match damage {
            "missing" => store.delete(&path).await.unwrap(),
            _ => {
                store.put(&path, vec![0xFF_u8; 400].into()).await.unwrap();
            }
        }
        // The manifest has not moved, so the object really is damaged: the
        // retries end in the error and nothing partial comes back.
        let err = ns
            .filter(&cheap, &FilterOptions::default())
            .await
            .unwrap_err();
        assert!(
            matches!(err, ItzmaError::CorruptSegment { .. }),
            "{damage}: {err:?}"
        );
    }
}

#[tokio::test]
async fn filter_futures_can_be_spawned() {
    let ns = Arc::new(
        Namespace::create(common::memory(), "docs", typed())
            .await
            .unwrap(),
    );
    ns.upsert(vec![item("a", json!({"price": 1}))])
        .await
        .unwrap();
    let handle = {
        let ns = ns.clone();
        tokio::spawn(async move {
            ns.filter(
                &Predicate::number("price", Compare::Lt, 2.0),
                &FilterOptions::default(),
            )
            .await
        })
    };
    assert_eq!(handle.await.unwrap().unwrap().rows.len(), 1);
}

/// A filter never writes, so one namespace is reused: fault operation `n`, for
/// every `n` a run makes and then some, as a plain error and as a lost ack.
/// Each faulted run either fails or gives the right answer, and the run after
/// it, on a healthy store, gives the right answer.
#[tokio::test]
async fn a_sweep_over_every_operation_of_a_filter_never_returns_a_wrong_answer() {
    let store: Arc<dyn ObjectStore> = common::memory();
    let ns = Namespace::create(store.clone(), "docs", typed())
        .await
        .unwrap();
    let mut lcg = Lcg(11);
    ns.upsert(batch(&mut lcg, 0, 300)).await.unwrap();
    ns.compact(&ROOMY).await.unwrap();
    ns.upsert(batch(&mut lcg, 200, 200)).await.unwrap();
    ns.compact(&ROOMY).await.unwrap();
    ns.upsert(batch(&mut lcg, 350, 50)).await.unwrap();
    ns.delete(vec!["d00010".into(), "d00390".into()])
        .await
        .unwrap();

    let (predicate, matches) = price_under_20();
    let want = expected(&store, matches).await;
    assert!(!want.is_empty());

    let faulty = FaultyStore::new(store.clone());
    let as_store: Arc<dyn ObjectStore> = faulty.clone();
    let handle = Namespace::open(as_store, "docs").await.unwrap();
    let options = FilterOptions::default().limit(10_000);
    for fault in [Fault::Generic, Fault::LostAck] {
        for n in 1..=80 {
            faulty.clear();
            faulty.reset_ops();
            faulty.fail_nth_op(n, fault);
            if let Ok(page) = handle.filter(&predicate, &options).await {
                let ids: Vec<String> = page.rows.into_iter().map(|r| r.id).collect();
                assert_eq!(ids, want, "{fault:?} at operation {n}");
            }
            faulty.clear();
            let healthy = handle.filter(&predicate, &options).await.unwrap();
            let ids: Vec<String> = healthy.rows.into_iter().map(|r| r.id).collect();
            assert_eq!(ids, want, "after {fault:?} at operation {n}");
        }
    }
}

/// A second filter over an unchanged tail lists the WAL once
/// and fetches nothing but the manifest.
#[tokio::test]
async fn an_unchanged_tail_is_listed_but_not_fetched_again() {
    let counting = Arc::new(CountingStore::default());
    let ns = Namespace::create(counting.clone(), "docs", typed())
        .await
        .unwrap();
    for i in 0..3 {
        ns.upsert(vec![item(&format!("d{i}"), json!({"price": i}))])
            .await
            .unwrap();
    }
    let (predicate, _) = price_under_20();
    let first = all_pages(&ns, &predicate, 100).await;
    counting.reset();
    assert_eq!(all_pages(&ns, &predicate, 100).await, first);
    assert_eq!(counting.lists(), 1);
    let paths = counting.paths();
    assert_eq!(paths.len(), 1, "{paths:?}");
    assert!(paths[0].ends_with("manifest/current.json"), "{paths:?}");
}

/// Versions 1, 2 and 3 in one namespace: a version 1 run decoded
/// whole, a version 2 segment read in blocks and version 3 ones read as
/// columns, with updates and deletes shadowing across all three. `filter`, a
/// query and every segment's bytes match a replay of the WAL.
#[tokio::test]
async fn versions_2_and_3_together_answer_like_a_replay() {
    for b in cas_backends() {
        eprintln!("backend: {}", b.name);
        let ns = Namespace::create(b.store.clone(), "docs", typed())
            .await
            .unwrap();
        let mut lcg = Lcg(21);
        ns.upsert(batch(&mut lcg, 0, 300)).await.unwrap();
        ns.compact(&ROOMY).await.unwrap();
        ns.upsert(batch(&mut lcg, 200, 300)).await.unwrap();
        ns.compact(&ROOMY).await.unwrap();
        ns.upgrade_segment_format(3).await.unwrap();
        ns.upsert(batch(&mut lcg, 400, 300)).await.unwrap();
        ns.delete((0..700).step_by(11).map(|n| format!("d{n:05}")).collect())
            .await
            .unwrap();
        ns.compact(&ROOMY).await.unwrap();
        ns.upsert(batch(&mut lcg, 650, 100)).await.unwrap();
        assert_eq!(common::live_segment_versions(&*b.store).await, [2, 2, 3]);
        let oracles = [price_under_20(), active_and_tagged_red()];
        check_stage(&ns, &b.store, &oracles, "versions 2 and 3 and a tail").await;
    }
}
