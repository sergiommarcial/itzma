//! Print what a namespace holds and what looks wrong with it. Read-only.
//!
//! ```text
//! cargo run --example dump -- <namespace> --dir <path>
//! cargo run --example dump -- <namespace> --s3 <key prefix>
//! ```
//!
//! `--dir` opens a local directory, the layout `LocalFileSystem` writes.
//! `--s3` opens the MinIO from `compose.yaml` (`make s3-up`) under a key
//! prefix. It reads the same `ITZMA_TEST_S3_*` variables as the tests, never
//! `AWS_*`, so it cannot pick up a real account. The exit code is 1 when the
//! report lists a problem, so a script can use it.

use std::process::ExitCode;
use std::sync::Arc;

use object_store::ObjectStore;
use object_store::aws::AmazonS3Builder;
use object_store::local::LocalFileSystem;
use object_store::prefix::PrefixStore;

const USAGE: &str = "usage: dump <namespace> (--dir <path> | --s3 <key prefix>)";

fn store(kind: &str, place: &str) -> Result<Arc<dyn ObjectStore>, String> {
    match kind {
        "--dir" => {
            let local = LocalFileSystem::new_with_prefix(place).map_err(|e| e.to_string())?;
            Ok(Arc::new(local))
        }
        "--s3" => {
            let var =
                |name: &str, default: &str| std::env::var(name).unwrap_or_else(|_| default.into());
            let s3 = AmazonS3Builder::new()
                .with_endpoint(var("ITZMA_TEST_S3_ENDPOINT", "http://127.0.0.1:9000"))
                .with_bucket_name(var("ITZMA_TEST_S3_BUCKET", "itzma-test"))
                .with_access_key_id(var("ITZMA_TEST_S3_ACCESS_KEY", "itzma"))
                .with_secret_access_key(var("ITZMA_TEST_S3_SECRET_KEY", "itzma-test-secret"))
                .with_region("us-east-1")
                .with_allow_http(true)
                .build()
                .map_err(|e| e.to_string())?;
            Ok(Arc::new(PrefixStore::new(s3, place)))
        }
        _ => Err(USAGE.to_string()),
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [namespace, kind, place] = args.as_slice() else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    let store = match store(kind, place) {
        Ok(store) => store,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::from(2);
        }
    };
    match itzma_core::inspect::report(&*store, namespace).await {
        Ok(report) => {
            print!("{report}");
            if report.problems.is_empty() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(e) => {
            eprintln!("dump: {e}");
            ExitCode::from(2)
        }
    }
}
