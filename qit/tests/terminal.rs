use std::process::Stdio;
use std::time::Duration;

use axum::routing::{get, post};
use axum::{Json, Router};
use qit_runtime::analysis::{analyze, rank, Metric};
use qit_runtime::config::Config;
use qit_runtime::model::{ProviderKind, RunRequest, RunStatus};
use qit_runtime::packs::PackCatalog;
use qit_runtime::paths::Paths;
use qit_runtime::runner::{execute_with_control, prepare_benchmark, ExecutionControl};
use qit_runtime::store::Store;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::net::TcpListener;
use tokio::process::Command;

#[tokio::test]
async fn model_catalog_defaults_to_listing_without_a_subcommand() {
    let directory = tempfile::tempdir().unwrap();
    let output = Command::new(binary())
        .env("QIT_HOME", directory.path())
        .arg("models")
        .output()
        .await
        .unwrap();
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("gradium/phonon-31m-en"));
}

#[tokio::test]
async fn piped_tui_explains_the_terminal_requirement_without_touching_state() {
    let directory = tempfile::tempdir().unwrap();
    let state = directory.path().join("state");
    let output = Command::new(binary())
        .env("QIT_HOME", &state)
        .arg("tui")
        .output()
        .await
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("requires an interactive terminal"));
    assert!(!state.exists());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn plain_run_reports_each_saved_sample_before_its_summary() {
    let (base_url, server) = provider().await;
    let directory = tempfile::tempdir().unwrap();
    let mut child = Command::new(binary())
        .env("QIT_HOME", directory.path())
        .args([
            "run",
            "core/text-generation-smoke",
            "--model",
            "fixture",
            "--base-url",
            &base_url,
            "--iterations",
            "1",
            "--warmups",
            "0",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut lines = BufReader::new(child.stderr.take().unwrap()).lines();
    let first = tokio::time::timeout(Duration::from_secs(10), lines.next_line())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(first.starts_with("[1/3]"), "{first}");
    assert!(child.try_wait().unwrap().is_none());
    let output = child.wait_with_output().await.unwrap();
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("Status: succeeded"));
    let store = Store::open(&directory.path().join("results.db")).unwrap();
    assert_eq!(store.runs(10, true).unwrap()[0].summary.sample_count, 3);
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancellation_keeps_completed_samples_and_comparisons_match_the_http_view() {
    let (base_url, server) = provider().await;
    let directory = tempfile::tempdir().unwrap();
    let paths = Paths::new(directory.path().into());
    paths.ensure().unwrap();
    let store = Store::open(&paths.database).unwrap();
    let catalog = PackCatalog::new(paths.packs.clone());
    let request = RunRequest {
        benchmark_id: "core/text-generation-smoke".into(),
        provider: ProviderKind::Ollama,
        base_url,
        api_key: None,
        model: "fixture".into(),
        iterations: 1,
        warmups: 0,
        max_output_tokens: None,
        timeout_seconds: 5,
        target_pid: None,
    };
    let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
    let control = ExecutionControl::new(move |progress| {
        let _ = sender.send(progress);
    });
    let prepared = prepare_benchmark(&store, &catalog, request.clone()).unwrap();
    let execution = execute_with_control(&store, prepared, control.clone());
    tokio::pin!(execution);
    let run = loop {
        tokio::select! {
            result = &mut execution => break result.unwrap(),
            progress = receiver.recv() => {
                if progress.unwrap().completed == 1 { control.cancel(); }
            }
        }
    };
    assert_eq!(run.status, RunStatus::Failed);
    assert_eq!(run.summary.sample_count, 1);
    assert!(run.error.unwrap().contains("interrupted"));
    assert_eq!(store.samples(&run.id).unwrap().len(), 1);
    assert!(analyze(store.runs(100, false).unwrap()).groups.is_empty());

    let prepared = prepare_benchmark(&store, &catalog, request).unwrap();
    let succeeded = execute_with_control(&store, prepared, ExecutionControl::new(|_| {}))
        .await
        .unwrap();
    assert_eq!(succeeded.status, RunStatus::Succeeded);
    let mut earlier = succeeded.clone();
    earlier.id = "earlier".into();
    earlier.started_at_ms -= 1;
    earlier.summary.latency_ms_p50 = Some(0.0);
    store.start_run(&earlier).unwrap();
    let analysis = analyze(store.runs(100, false).unwrap());
    assert_eq!(analysis.groups[0].models.len(), 1);
    assert_eq!(analysis.groups[0].models[0].id, succeeded.id);
    assert!(rank(&analysis.groups[0].models, Metric::Memory).is_empty());
    let dashboard = qit_runtime::bind(Config {
        home: directory.path().into(),
        listen: "127.0.0.1:0".parse().unwrap(),
    })
    .await
    .unwrap();
    let response: serde_json::Value =
        reqwest::get(format!("{}/api/analysis", dashboard.base_url()))
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
    assert_eq!(response, serde_json::to_value(analysis).unwrap());
    dashboard.shutdown().await;
    server.abort();
}

async fn provider() -> (String, tokio::task::JoinHandle<()>) {
    async fn generate() -> &'static str {
        tokio::time::sleep(Duration::from_millis(150)).await;
        "{\"response\":\"raspberry 42 edge\",\"done\":false}\n{\"done\":true,\"eval_count\":4,\"eval_duration\":200000000}\n"
    }
    let app = Router::new()
        .route(
            "/api/version",
            get(|| async { Json(serde_json::json!({"version": "0.40.3"})) }),
        )
        .route(
            "/api/show",
            post(|| async { Json(serde_json::json!({"capabilities": ["completion"]})) }),
        )
        .route("/api/generate", post(generate));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    (
        url,
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        }),
    )
}

fn binary() -> std::path::PathBuf {
    std::env::var_os("QIT_TEST_BINARY")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| env!("CARGO_BIN_EXE_qit").into())
}
