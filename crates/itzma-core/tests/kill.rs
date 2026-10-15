//! Real process kills. `tests/recovery.rs` stops tasks between two store
//! calls, which covers every call boundary but never a process dying in the
//! middle of an HTTP request. Here a writer, two compactors and a collector run
//! as separate processes against MinIO and are sent SIGKILL at a random moment,
//! round after round.
//!
//! It runs only with `ITZMA_TEST_S3_ENDPOINT` set (`make test-s3`), since
//! processes can share nothing else. The timing is real, so a failure cannot be
//! replayed from a seed. The test prints the seed, the store prefix (for
//! `examples/dump.rs --s3 <prefix>` after a failure) and each round's delay.
#![cfg(unix)]

mod common;

use std::collections::HashSet;
use std::io::{BufRead, BufReader};
use std::os::unix::process::ExitStatusExt;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use common::{Lcg, assert_consistent, doc, s3_at, s3_prefix, schema, wal_listing};
use itzma_core::{Compaction, CompactionOptions, GcOptions, Metric, Namespace};

const DIM: usize = 2;
const L2: Metric = Metric::L2;
const ROUNDS: u32 = 15;
const SIGKILL: i32 = 9;

/// What the parent tells each child: its role, the key prefix it shares with the
/// others, and the round, which keeps writer ids unique across rounds.
const ROLE: &str = "ITZMA_KILL_ROLE";
const PREFIX: &str = "ITZMA_KILL_PREFIX";
const ROUND: &str = "ITZMA_KILL_ROUND";
const CHILD_TEST: &str = "kill_child";

/// A request that MinIO received before the kill can still land a moment
/// later. Checking the store sooner could see one appear halfway through.
const SETTLE: Duration = Duration::from_millis(400);
const READY_TIMEOUT: Duration = Duration::from_secs(30);

const WRITER: &str = "writer";
const COMPACTOR: &str = "compactor";
const COLLECTOR: &str = "collector";

/// Not a test of its own. `killing_real_processes_never_hurts` runs this binary
/// again with `ROLE` set, once per process it means to kill. Without `ROLE` it
/// returns at once, so a normal run passes it.
///
/// The writer prints `ACK <id>` after each `upsert` returns, so an id the
/// parent read was acknowledged and must survive. One acknowledged but not yet
/// printed is missed, which only makes the check weaker, never wrong.
#[tokio::test]
async fn kill_child() {
    let Ok(role) = std::env::var(ROLE) else {
        return;
    };
    let prefix = std::env::var(PREFIX).expect("prefix");
    let round: u32 = std::env::var(ROUND).expect("round").parse().expect("round");
    let store = s3_at(&prefix).expect("the parent checked the endpoint");
    let ns = Namespace::open(store, "docs").await.expect("open");
    println!("READY");

    let compact = CompactionOptions {
        max_segments: 2,
        max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
    };
    let collect = GcOptions {
        min_age: Duration::ZERO,
        dry_run: false,
        fence_wal: true,
    };
    let mut i = 0u32;
    loop {
        match role.as_str() {
            WRITER => {
                let id = format!("r{round}-{i}");
                ns.upsert(vec![doc(&id, &[round as f32, i as f32])])
                    .await
                    .expect("upsert");
                println!("ACK {id}");
                i += 1;
            }
            COMPACTOR => {
                ns.compact(&compact).await.expect("compact");
            }
            COLLECTOR => {
                ns.gc(&collect).await.expect("gc");
            }
            other => panic!("unknown role {other}"),
        }
    }
}

fn spawn(role: &str, prefix: &str, round: u32) -> Child {
    let exe = std::env::current_exe().expect("test binary");
    Command::new(exe)
        .args(["--exact", CHILD_TEST, "--nocapture"])
        .env(ROLE, role)
        .env(PREFIX, prefix)
        .env(ROUND, round.to_string())
        .stdout(if role == WRITER {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn")
}

/// The writer's output, a line at a time, until it dies and the pipe closes.
fn lines(child: &mut Child) -> Receiver<String> {
    let out = child.stdout.take().expect("piped");
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        for line in BufReader::new(out).lines().map_while(Result::ok) {
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    rx
}

/// Wait for the writer to have opened the namespace, so the kill lands in
/// work and not in process start-up.
fn wait_ready(lines: &Receiver<String>, round: u32) {
    loop {
        let line = lines
            .recv_timeout(READY_TIMEOUT)
            .unwrap_or_else(|e| panic!("round {round}: the writer never got ready: {e}"));
        if line == "READY" {
            return;
        }
    }
}

/// SIGKILL, and check it was the kill that ended the process. A child that
/// stopped by itself panicked on an error, which is a bug.
fn kill(child: &mut Child, role: &str, round: u32) {
    let _ = child.kill();
    let status = child.wait().expect("wait");
    assert_eq!(
        status.signal(),
        Some(SIGKILL),
        "round {round}: the {role} stopped on its own: {status}"
    );
}

fn fenced(listing: &[(u64, u64)]) -> usize {
    listing.iter().filter(|(_, size)| *size == 8).count()
}

#[tokio::test]
async fn killing_real_processes_never_hurts() {
    let prefix = s3_prefix();
    let Some(store) = s3_at(&prefix) else {
        eprintln!("skipped: ITZMA_TEST_S3_ENDPOINT is not set");
        return;
    };
    Namespace::create(store.clone(), "docs", schema(DIM, L2))
        .await
        .unwrap();

    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;
    eprintln!("seed {seed}, store prefix {prefix}");
    let mut rng = Lcg(seed);
    let mut acknowledged: Vec<String> = Vec::new();

    for round in 0..ROUNDS {
        let mut writer = spawn(WRITER, &prefix, round);
        let acks = lines(&mut writer);
        let mut others =
            [COMPACTOR, COMPACTOR, COLLECTOR].map(|role| (role, spawn(role, &prefix, round)));
        wait_ready(&acks, round);

        let delay = Duration::from_millis(50 + rng.below(450));
        thread::sleep(delay);
        kill(&mut writer, WRITER, round);
        for (role, child) in &mut others {
            kill(child, role, round);
        }
        let before = acknowledged.len();
        acknowledged.extend(
            acks.iter()
                .filter_map(|l| l.strip_prefix("ACK ").map(str::to_owned)),
        );
        thread::sleep(SETTLE);

        let present: HashSet<String> = assert_consistent(store.clone(), DIM, L2)
            .await
            .into_iter()
            .map(|d| d.id)
            .collect();
        for id in &acknowledged {
            assert!(
                present.contains(id),
                "round {round}: acknowledged write {id} is gone"
            );
        }
        eprintln!(
            "round {round}: killed after {delay:?}, {} acknowledged this round",
            acknowledged.len() - before
        );
    }

    let ns = Namespace::open(store.clone(), "docs").await.unwrap();
    let opts = CompactionOptions {
        max_segments: 2,
        max_segment_bytes: itzma_core::compaction::DEFAULT_MAX_SEGMENT_BYTES,
    };
    let mut settled = false;
    for _ in 0..4 {
        if ns.compact(&opts).await.unwrap() == Compaction::Nothing {
            settled = true;
            break;
        }
    }
    assert!(settled, "compaction did not converge");
    assert_consistent(store.clone(), DIM, L2).await;

    assert!(
        !acknowledged.is_empty(),
        "the writer never got a write through"
    );
    let listing = wal_listing(&*store).await;
    assert!(
        fenced(&listing) > 0,
        "the collector never fenced anything, so the test did not exercise fencing"
    );
    eprintln!(
        "{} writes acknowledged, {} of {} WAL objects fenced",
        acknowledged.len(),
        fenced(&listing),
        listing.len()
    );
}
