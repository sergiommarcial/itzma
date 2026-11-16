//! A maintenance loop that only one process runs at a time.
//!
//! Every process that opens the namespace can run this loop. Each one ticks the
//! lease every `renew`; the one that holds it calls `maintain` every `EVERY`.
//! Itzma starts no task of its own, so a Rust program that wants automatic
//! maintenance runs something like this on its runtime.
//!
//! Run with `cargo run -p itzma-core --example maintenance_loop`. It stops after
//! a few ticks, with timings shortened so it ends quickly.

use std::sync::Arc;
use std::time::{Duration, Instant};

use itzma_core::{Document, LeaseOptions, MaintenanceOptions, Metric, Namespace, Role, Schema};
use object_store::memory::InMemory;

/// How often the leader maintains. 60 seconds in the Python default.
const EVERY: Duration = Duration::from_millis(200);

#[tokio::main]
async fn main() -> itzma_core::Result<()> {
    let store = Arc::new(InMemory::new());
    let ns = Namespace::create(store, "docs", Schema::new(2, Metric::L2)).await?;
    for i in 0..3 {
        let id = format!("doc-{i}");
        ns.upsert(vec![Document {
            id,
            vector: vec![1.0, 0.0],
            attrs: None,
            sparse: None,
        }])
        .await?;
    }

    let options = LeaseOptions {
        renew: Duration::from_millis(50),
        ttl: Duration::from_millis(200),
    };
    let mut lease = ns.maintenance_lease(options)?;
    let mut last_run: Option<Instant> = None;
    for _ in 0..10 {
        let now = Instant::now();
        match lease.tick(now).await {
            Ok(Role::Leader) => {
                // The first run waits a full interval after the claim, so a
                // short-lived process never pays for one.
                let since = *last_run.get_or_insert(now);
                if now.duration_since(since) >= EVERY {
                    let report = ns.maintain(&MaintenanceOptions::default()).await?;
                    println!("maintained: tail of {} objects", report.tail_objects);
                    last_run = Some(now);
                }
            }
            Ok(Role::Follower) => last_run = None,
            Err(e) => eprintln!("lease: {e}"),
        }
        tokio::time::sleep(options.renew).await;
    }
    lease.release().await
}
