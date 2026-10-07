use std::process::Stdio;
use std::time::Duration;

use axum::response::IntoResponse;
use axum::routing::post;
use axum::{Json, Router};
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::net::TcpListener;
use tokio::process::Command;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn staged_qit_runs_benchmarks_classifies_results_and_serves_dashboard() {
    let temporary = TempDir::new().unwrap();
    let binary = temporary.path().join("qit");
    std::fs::copy(env!("CARGO_BIN_EXE_qit"), &binary).unwrap();
    let state = temporary.path().join("state");
    let provider = ollama_server().await;

    let list = command(&binary, temporary.path(), &state)
        .args(["benchmark", "list"])
        .output()
        .await
        .unwrap();
    assert!(list.status.success(), "{}", stderr(&list));
    assert!(stdout(&list).contains("core/image-to-text-smoke"));
    assert!(stdout(&list).contains("core/speech-to-text-smoke"));

    let success = command(&binary, temporary.path(), &state)
        .args([
            "run",
            "core/text-generation-smoke",
            "--provider",
            "ollama",
            "--model",
            "fixture-model",
            "--base-url",
            &provider.base_url,
            "--warmups",
            "0",
            "--iterations",
            "1",
        ])
        .output()
        .await
        .unwrap();
    assert!(success.status.success(), "{}", stderr(&success));
    assert!(stdout(&success).contains("Status: succeeded"));
    assert!(stdout(&success).contains("Tokens/s p50: 20.0"));

    let failure = command(&binary, temporary.path(), &state)
        .args([
            "run",
            "core/text-generation-smoke",
            "--provider",
            "ollama",
            "--model",
            "unavailable-model",
            "--base-url",
            "http://127.0.0.1:9",
            "--warmups",
            "0",
            "--iterations",
            "1",
            "--timeout",
            "1",
        ])
        .output()
        .await
        .unwrap();
    assert_eq!(failure.status.code(), Some(2), "{}", stderr(&failure));
    assert!(stdout(&failure).contains("Status: failed"));

    let history = command(&binary, temporary.path(), &state)
        .arg("results")
        .output()
        .await
        .unwrap();
    assert!(history.status.success(), "{}", stderr(&history));
    assert!(stdout(&history).contains("succeeded"));
    assert!(stdout(&history).contains("failed"));

    let export = command(&binary, temporary.path(), &state)
        .args(["export", "--format", "jsonl"])
        .output()
        .await
        .unwrap();
    assert!(export.status.success(), "{}", stderr(&export));
    let exported = stdout(&export);
    assert_eq!(exported.lines().count(), 1, "{exported}");
    assert!(exported.contains("\"status\":\"succeeded\""));
    assert!(!exported.contains("unavailable-model"));

    let mut dashboard = command(&binary, temporary.path(), &state)
        .args(["dashboard", "--port", "0"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let stdout = dashboard.stdout.take().unwrap();
    let mut lines = BufReader::new(stdout).lines();
    let line = tokio::time::timeout(Duration::from_secs(10), lines.next_line())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let base_url = line.strip_prefix("q.it dashboard listening on ").unwrap();

    let health: serde_json::Value = reqwest::get(format!("{base_url}/api/health"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(health["ok"], true);
    let runs: serde_json::Value = reqwest::get(format!("{base_url}/api/runs"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(runs["succeeded"], 1);
    assert_eq!(runs["failed"], 1);
    let html = reqwest::get(base_url).await.unwrap().text().await.unwrap();
    assert!(html.contains("<div id=\"root\"></div>"));
    let asset_path = html
        .split("src=\"")
        .nth(1)
        .and_then(|tail| tail.split('"').next())
        .unwrap();
    let asset = reqwest::get(format!("{base_url}{asset_path}"))
        .await
        .unwrap();
    assert!(asset.status().is_success());
    assert!(asset.bytes().await.unwrap().len() > 100_000);

    dashboard.kill().await.unwrap();
    dashboard.wait().await.unwrap();
    provider.join.abort();
    assert!(state.join("results.db").is_file());
}

fn command(
    binary: &std::path::Path,
    directory: &std::path::Path,
    state: &std::path::Path,
) -> Command {
    let mut command = Command::new(binary);
    command.current_dir(directory).env("QIT_HOME", state);
    command
}

fn stdout(output: &std::process::Output) -> String {
    String::from_utf8_lossy(&output.stdout).to_string()
}

fn stderr(output: &std::process::Output) -> String {
    String::from_utf8_lossy(&output.stderr).to_string()
}

struct TestServer {
    base_url: String,
    join: tokio::task::JoinHandle<()>,
}

async fn ollama_server() -> TestServer {
    async fn show() -> Json<serde_json::Value> {
        Json(serde_json::json!({ "capabilities": ["completion"] }))
    }
    async fn generate() -> impl IntoResponse {
        (
            [("content-type", "application/x-ndjson")],
            concat!(
                "{\"response\":\"raspberry 42 edge\",\"done\":false}\n",
                "{\"response\":\"\",\"done\":true,\"prompt_eval_count\":5,",
                "\"eval_count\":4,\"eval_duration\":200000000}\n"
            ),
        )
    }
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let join = tokio::spawn(async move {
        axum::serve(
            listener,
            Router::new()
                .route("/api/show", post(show))
                .route("/api/generate", post(generate)),
        )
        .await
        .unwrap();
    });
    TestServer {
        base_url: format!("http://{address}"),
        join,
    }
}
