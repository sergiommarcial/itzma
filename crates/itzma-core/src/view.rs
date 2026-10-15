//! The handle's view of a namespace between queries: the manifest
//! it last loaded and the WAL tail it folded on top of it, so a query lists
//! and folds only what is new.

use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

use object_store::ObjectStore;

use crate::budget::ReadBudget;
use crate::cpu::Lane;
use crate::error::{ItzmaError, Result};
use crate::layer::{self, Layer};
use crate::manifest::{self, Manifest};
use crate::namespace::is_stale_read;
use crate::{segment, wal};

/// The WAL past one `folded_through`, folded into a layer.
#[derive(Debug, Default)]
pub(crate) struct TailView {
    pub(crate) folded_through: u64,
    /// The last seq folded in, or `folded_through` when the tail is empty.
    pub(crate) last_seq: u64,
    pub(crate) layer: Arc<Layer>,
}

/// The tail past `folded_through`, reusing `base` when it was built on the
/// same `folded_through`: only objects past `base.last_seq` are listed,
/// fetched, charged and folded, onto a copy of its layer. `base` itself is
/// never changed, since queries may still hold it. A base on another
/// `folded_through` is ignored: a compaction moved it, and a fold cannot be
/// undone.
pub(crate) async fn extend(
    store: &dyn ObjectStore,
    namespace: &str,
    base: Option<&Arc<TailView>>,
    folded_through: u64,
    budget: &ReadBudget,
) -> Result<Arc<TailView>> {
    let base = base.filter(|view| view.folded_through == folded_through);
    let from = base.map_or(folded_through, |view| view.last_seq);
    let log = wal::stream_log(store, namespace, from, budget, Lane::Query).await?;
    let (new, last) = layer::fold_log(log).await?;
    let view = match (base, last) {
        (Some(base), None) => return Ok(base.clone()),
        (Some(base), Some(last)) => {
            let mut grown = (*base.layer).clone();
            grown.overlay(new);
            TailView {
                folded_through,
                last_seq: last,
                layer: Arc::new(grown),
            }
        }
        (None, last) => TailView {
            folded_through,
            last_seq: last.unwrap_or(folded_through),
            layer: Arc::new(new),
        },
    };
    Ok(Arc::new(view))
}

/// Load the manifest and check its segment list against the invariant, so
/// every reader fails the same way on a damaged manifest.
pub(crate) async fn load_checked(store: &dyn ObjectStore, namespace: &str) -> Result<Manifest> {
    let (manifest, _version) = manifest::load(store, namespace).await?;
    segment::validate_chain(&manifest.live_segments, manifest.wal_folded_through)?;
    Ok(manifest)
}

/// A snapshot the handle keeps between queries: the tail it last folded, and
/// when the manifest under it was loaded.
#[derive(Debug)]
struct Snapshot {
    tail: Arc<TailView>,
    loaded_at: Instant,
}

/// What one query attempt starts from: the manifest it plans against. The
/// tail is read by [`Views::tail`], which the engines run alongside their
/// segment reads.
#[derive(Debug)]
pub(crate) struct Plan {
    pub(crate) manifest: Arc<Manifest>,
    loaded_at: Instant,
}

/// Where a view is read from, and for whom.
pub(crate) struct Reader<'a> {
    /// The manifest is always read here, never through a cache.
    pub(crate) store: &'a dyn ObjectStore,
    /// WAL objects go through the query store, like every query read.
    pub(crate) query_store: &'a dyn ObjectStore,
    pub(crate) namespace: &'a str,
    /// The query attempt's budget: the tail shares it with the segments.
    pub(crate) budget: &'a ReadBudget,
}

/// A handle's last snapshot.
#[derive(Debug, Default)]
pub(crate) struct Views {
    current: Mutex<Option<Arc<Snapshot>>>,
}

impl Views {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// The plan a query starts from: a fresh manifest, with the tail left to
    /// [`tail`](Self::tail), which lists the WAL now.
    pub(crate) async fn plan(&self, r: &Reader<'_>) -> Result<Plan> {
        fresh_plan(r).await
    }

    /// `err` came from an attempt on `stale`. If the manifest moved, the
    /// reader was only behind: plan again on the current one. Otherwise the
    /// object really is missing or the fence really is misplaced, and `err`
    /// comes back as it was.
    pub(crate) async fn replan(
        &self,
        r: &Reader<'_>,
        stale: &Plan,
        err: ItzmaError,
    ) -> Result<Plan> {
        if !is_stale_read(&err) {
            return Err(err);
        }
        let plan = fresh_plan(r).await?;
        if *plan.manifest == *stale.manifest {
            return Err(err);
        }
        Ok(plan)
    }

    /// The tail on `plan`'s manifest: the handle's last tail extended with
    /// only what is new, which becomes its snapshot.
    pub(crate) async fn tail(&self, r: &Reader<'_>, plan: &Plan) -> Result<Arc<Layer>> {
        let base = self.current().map(|snap| snap.tail.clone());
        let folded_through = plan.manifest.wal_folded_through;
        let tail = extend(
            r.query_store,
            r.namespace,
            base.as_ref(),
            folded_through,
            r.budget,
        )
        .await?;
        Ok(self.publish(plan, tail).tail.layer.clone())
    }

    fn current(&self) -> Option<Arc<Snapshot>> {
        self.current
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Record `tail` on `plan`'s manifest as the handle's snapshot if it is
    /// the newest: a later `folded_through`, then a manifest loaded later
    /// (one that may have changed only its indexes), then more of the same
    /// tail. An own-write fold on an older manifest therefore never puts that
    /// manifest back over a refresh; a later query folds what it lacks. Two
    /// queries racing both produce correct views, and the newer one is kept.
    fn publish(&self, plan: &Plan, tail: Arc<TailView>) -> Arc<Snapshot> {
        let snap = Arc::new(Snapshot {
            tail,
            loaded_at: plan.loaded_at,
        });
        let mut current = self.current.lock().unwrap_or_else(PoisonError::into_inner);
        let newer = current.as_ref().is_none_or(|old| snap.rank() >= old.rank());
        if newer {
            *current = Some(snap.clone());
        }
        snap
    }
}

impl Snapshot {
    fn rank(&self) -> (u64, Instant, u64) {
        (self.tail.folded_through, self.loaded_at, self.tail.last_seq)
    }
}

/// A plan on a manifest loaded now, with the tail still to read.
async fn fresh_plan(r: &Reader<'_>) -> Result<Plan> {
    let loaded_at = Instant::now();
    let manifest = load_checked(r.store, r.namespace).await?;
    Ok(Plan {
        manifest: Arc::new(manifest),
        loaded_at,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use object_store::memory::InMemory;
    use object_store::path::Path;

    use super::*;
    use crate::budget::ReadBudget;
    use crate::cpu::Lane;
    use crate::document::{Document, Metric, Schema, WalOp};
    use crate::layer::Layer;
    use crate::manifest::Manifest;

    fn key(view: &TailView) -> (u64, u64) {
        (view.folded_through, view.last_seq)
    }

    fn up(id: &str, v: f32) -> WalOp {
        WalOp::Upsert(Document {
            id: id.into(),
            vector: vec![v],
            attrs: None,
        })
    }

    async fn write(store: &InMemory, ops: Vec<WalOp>) -> u64 {
        crate::wal::append(store, "ns", 1, ops).await.unwrap()
    }

    async fn scratch(store: &InMemory, folded: u64) -> Layer {
        let budget = ReadBudget::unlimited();
        let log = crate::wal::stream_log(store, "ns", folded, &budget, Lane::Query)
            .await
            .unwrap();
        crate::layer::fold_log(log).await.unwrap().0
    }

    #[tokio::test]
    async fn an_empty_tail_is_a_view_at_folded_through() {
        let store = InMemory::new();
        let budget = ReadBudget::unlimited();
        let view = extend(&store, "ns", None, 0, &budget).await.unwrap();
        assert_eq!(key(&view), (0, 0));
        assert!(view.layer.is_empty());
    }

    #[tokio::test]
    async fn nothing_new_returns_the_same_view() {
        let store = InMemory::new();
        write(&store, vec![up("a", 1.0)]).await;
        let budget = ReadBudget::unlimited();
        let first = extend(&store, "ns", None, 0, &budget).await.unwrap();
        let again = extend(&store, "ns", Some(&first), 0, &budget)
            .await
            .unwrap();
        assert!(Arc::ptr_eq(&first, &again));
    }

    #[tokio::test]
    async fn extend_applies_deletes_and_reinserts_across_the_split() {
        let store = InMemory::new();
        write(&store, vec![up("a", 1.0), up("b", 1.0)]).await;
        write(&store, vec![WalOp::Delete("c".into())]).await;
        let budget = ReadBudget::unlimited();
        let base = extend(&store, "ns", None, 0, &budget).await.unwrap();
        write(&store, vec![WalOp::Delete("a".into()), up("c", 2.0)]).await;
        write(&store, vec![up("b", 3.0)]).await;
        let grown = extend(&store, "ns", Some(&base), 0, &budget).await.unwrap();
        assert_eq!(grown.last_seq, 4);
        assert_eq!(*grown.layer, scratch(&store, 0).await);
        // The base is untouched: queries still holding it see the old state.
        assert_eq!(base.last_seq, 2);
        assert!(base.layer.holds("a"));
    }

    #[tokio::test]
    async fn a_base_on_another_folded_through_is_ignored() {
        let store = InMemory::new();
        write(&store, vec![up("a", 1.0)]).await;
        write(&store, vec![up("b", 1.0)]).await;
        let budget = ReadBudget::unlimited();
        let old = extend(&store, "ns", None, 0, &budget).await.unwrap();
        let moved = extend(&store, "ns", Some(&old), 1, &budget).await.unwrap();
        assert_eq!(key(&moved), (1, 2));
        assert_eq!(*moved.layer, scratch(&store, 1).await);
    }

    #[tokio::test]
    async fn only_new_objects_are_charged() {
        let store = InMemory::new();
        write(&store, vec![up("a", 1.0)]).await;
        let first = ReadBudget::unlimited();
        let base = extend(&store, "ns", None, 0, &first).await.unwrap();
        write(&store, vec![up("b", 1.0)]).await;
        let second = ReadBudget::unlimited();
        extend(&store, "ns", Some(&base), 0, &second).await.unwrap();
        let both = ReadBudget::unlimited();
        extend(&store, "ns", None, 0, &both).await.unwrap();
        assert!(first.used() > 0 && second.used() > 0);
        assert_eq!(first.used() + second.used(), both.used());
    }

    fn reader<'a>(store: &'a InMemory, budget: &'a ReadBudget) -> Reader<'a> {
        Reader {
            store,
            query_store: store,
            namespace: "ns",
            budget,
        }
    }

    async fn create(store: &InMemory) {
        let schema = Schema::new(1, Metric::L2);
        crate::Namespace::create(Arc::new(store.clone()), "ns", schema)
            .await
            .unwrap();
    }

    /// Plan, then read the tail, the way a query attempt does.
    async fn view(views: &Views, r: &Reader<'_>) -> (Plan, Arc<Layer>) {
        let plan = views.plan(r).await.unwrap();
        let tail = views.tail(r, &plan).await.unwrap();
        (plan, tail)
    }

    #[tokio::test]
    async fn a_strict_plan_sees_a_write_made_after_the_last_one() {
        let store = InMemory::new();
        create(&store).await;
        let views = Views::new();
        let budget = ReadBudget::unlimited();
        let (_, first) = view(&views, &reader(&store, &budget)).await;
        assert!(first.is_empty());
        write(&store, vec![up("a", 1.0)]).await;
        let (_, second) = view(&views, &reader(&store, &budget)).await;
        assert!(second.holds("a"));
    }

    #[tokio::test]
    async fn replan_with_an_unchanged_manifest_returns_the_error() {
        let store = InMemory::new();
        create(&store).await;
        let views = Views::new();
        let budget = ReadBudget::unlimited();
        let plan = views.plan(&reader(&store, &budget)).await.unwrap();
        let err = crate::segment::missing(&Path::from("ns/segments/x/data.bin"));
        let back = views
            .replan(&reader(&store, &budget), &plan, err)
            .await
            .unwrap_err();
        assert!(crate::segment::is_missing(&back));
    }

    #[test]
    fn a_later_load_on_the_same_fold_wins_over_more_of_an_older_tail() {
        let views = Views::new();
        let older = Plan {
            manifest: Arc::new(Manifest::new(Schema::new(1, Metric::L2))),
            loaded_at: Instant::now(),
        };
        let mut newer_manifest = Manifest::new(Schema::new(1, Metric::L2));
        newer_manifest.wal_format = 1;
        let newer = Plan {
            manifest: Arc::new(newer_manifest),
            loaded_at: older.loaded_at + Duration::from_millis(1),
        };
        let tail = |last| {
            Arc::new(TailView {
                folded_through: 0,
                last_seq: last,
                layer: Arc::default(),
            })
        };
        // A query on the older manifest folded further than one that loaded
        // later: the later load is kept, so the snapshot never goes back to
        // an older manifest's tail.
        views.publish(&older, tail(5));
        views.publish(&newer, tail(3));
        assert_eq!(views.current().unwrap().tail.last_seq, 3);
        // The reverse order keeps it too.
        let views = Views::new();
        views.publish(&newer, tail(3));
        views.publish(&older, tail(5));
        assert_eq!(views.current().unwrap().tail.last_seq, 3);
    }

    #[test]
    fn a_view_never_moves_backwards() {
        let views = Views::new();
        let plan = Plan {
            manifest: Arc::new(Manifest::new(Schema::new(1, Metric::L2))),
            loaded_at: Instant::now(),
        };
        let tail = |folded, last| {
            Arc::new(TailView {
                folded_through: folded,
                last_seq: last,
                layer: Arc::default(),
            })
        };
        views.publish(&plan, tail(0, 5));
        views.publish(&plan, tail(0, 3));
        assert_eq!(key(&views.current().unwrap().tail), (0, 5));
        views.publish(&plan, tail(4, 4));
        assert_eq!(key(&views.current().unwrap().tail), (4, 4));
    }
}
