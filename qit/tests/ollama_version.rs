use std::future::IntoFuture;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::net::TcpListener;
use tokio::process::Command;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ollama_versions_gate_embeddings_and_survive_history_and_exports() {
    let temporary = TempDir::new().unwrap();
    let state = temporary.path().join("state");
    let calls = Arc::new(AtomicUsize::new(0));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(
        axum::serve(
            listener,
            Router::new()
                .route("/{version}/api/version", get(version))
                .route("/{version}/api/show", post(show))
                .route("/{version}/api/embed", post(embed))
                .route("/{version}/api/generate", post(generate))
                .with_state(calls.clone()),
        )
        .into_future(),
    );
    for version in ["0.40.2", "0.9.9", "0.40.3-rc.1", "invalid"] {
        let output = run(&state, &endpoint, version, "core/embedding-retrieval").await;
        assert_eq!(output.status.code(), Some(2), "{version}: {output:?}");
        let text = String::from_utf8_lossy(&output.stdout);
        assert!(text.contains("Ollama 0.40.3 or later"), "{text}");
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    for version in ["missing", "empty", "numeric", "http-error"] {
        let output = run(&state, &endpoint, version, "core/embedding-retrieval").await;
        assert_eq!(output.status.code(), Some(2), "{version}: {output:?}");
        assert!(String::from_utf8_lossy(&output.stdout).contains("Ollama version"));
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    for version in ["0.40.3", "0.40.10", "0.41.0", "1.0.0", "0.40.3+build.1"] {
        let output = run(&state, &endpoint, version, "core/embedding-retrieval").await;
        assert!(output.status.success(), "{version}: {output:?}");
    }
    let text = run(&state, &endpoint, "0.40.2", "core/text-generation-smoke").await;
    assert!(text.status.success(), "{text:?}");
    let exported = command(&state)
        .args(["export", "--format", "jsonl"])
        .output()
        .await
        .unwrap();
    assert!(exported.status.success());
    let records: Vec<Value> = String::from_utf8(exported.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(records.len(), 6);
    for record in records {
        assert_eq!(
            record["run"]["provider_metadata"]["version"],
            record["run"]["model"]
        );
    }
    let csv = command(&state)
        .args(["export", "--format", "csv"])
        .output()
        .await
        .unwrap();
    assert!(csv.status.success());
    assert!(String::from_utf8_lossy(&csv.stdout).contains("provider_version"));

    let mut dashboard = command(&state)
        .args(["dashboard", "--port", "0"])
        .stdout(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let mut lines = BufReader::new(dashboard.stdout.take().unwrap()).lines();
    let line = tokio::time::timeout(Duration::from_secs(10), lines.next_line())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let url = line.strip_prefix("q.it dashboard listening on ").unwrap();
    let history: Value = reqwest::get(format!("{url}/api/runs"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(history["failed"], 8);
    assert_eq!(history["succeeded"], 6);
    for record in history["runs"].as_array().unwrap() {
        if matches!(
            record["model"].as_str().unwrap(),
            "missing" | "empty" | "numeric" | "http-error"
        ) {
            assert!(record["provider_metadata"]["version"].is_null());
        } else {
            assert_eq!(record["provider_metadata"]["version"], record["model"]);
        }
    }
    dashboard.kill().await.unwrap();
    dashboard.wait().await.unwrap();
    server.abort();
}

fn command(state: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_qit"));
    command.env("QIT_HOME", state).env_remove("QIT_API_KEY");
    command
}

async fn run(state: &Path, endpoint: &str, version: &str, benchmark: &str) -> std::process::Output {
    command(state)
        .args([
            "run",
            benchmark,
            "--provider",
            "ollama",
            "--model",
            version,
            "--base-url",
            &format!("{endpoint}/{version}"),
            "--api-key",
            "fixture-token",
            "--warmups",
            "1",
            "--iterations",
            "1",
        ])
        .output()
        .await
        .unwrap()
}

async fn version(
    axum::extract::Path(version): axum::extract::Path<String>,
    headers: HeaderMap,
) -> Result<Json<Value>, StatusCode> {
    if headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        != Some("Bearer fixture-token")
    {
        return Err(StatusCode::UNAUTHORIZED);
    }
    match version.as_str() {
        "missing" => Ok(Json(json!({}))),
        "empty" => Ok(Json(json!({"version": ""}))),
        "numeric" => Ok(Json(json!({"version": 40}))),
        "http-error" => Err(StatusCode::SERVICE_UNAVAILABLE),
        _ => Ok(Json(json!({"version": version}))),
    }
}

async fn show() -> Json<Value> {
    Json(json!({"capabilities": ["embedding", "completion"]}))
}

async fn embed(State(calls): State<Arc<AtomicUsize>>, Json(body): Json<Value>) -> Json<Value> {
    calls.fetch_add(1, Ordering::SeqCst);
    let inputs = body["input"].as_array().unwrap();
    Json(json!({"embeddings": inputs.iter().map(|_| [1.0, 0.0]).collect::<Vec<_>>()}))
}

async fn generate() -> impl IntoResponse {
    ([("content-type", "application/x-ndjson")], "{\"response\":\"raspberry 42 edge\",\"done\":true,\"eval_count\":4,\"eval_duration\":200000000}\n")
}
