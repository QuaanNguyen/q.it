use std::future::IntoFuture;
use std::path::Path;
use std::process::Stdio;

use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::net::TcpListener;
use tokio::process::Command;

const MODEL: &str = "datalab-to/indra-ocr";
const REVISION: &str = "ed345c7f158eddbc7dee0c6a7269afb75a94f7e8";

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ocr_output_evidence_reaches_history_export_and_the_dashboard() {
    let temporary = TempDir::new().unwrap();
    let state = temporary.path().join("state");
    let (endpoint, server) = provider("success").await;
    let output = run(&state, &endpoint).await;
    assert!(output.status.success(), "{output:?}");
    let row: Value = serde_json::from_str(&export(&state).await).unwrap();
    assert_eq!(row["run"]["task"], "image_to_text");
    let sample = &row["samples"][0];
    assert_eq!(sample["output_excerpt"], "red and blue");
    assert_eq!(sample["quality_score"], 1.0);
    assert_eq!(sample["output_metadata"]["schema_version"], 1);
    assert_eq!(sample["output_metadata"]["route"], "indra-full-ocr");
    assert_eq!(sample["output_metadata"]["stage_calls"]["det"], 1);
    let mut dashboard = command(&state)
        .args(["dashboard", "--port", "0"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut lines = BufReader::new(dashboard.stdout.take().unwrap()).lines();
    let line = tokio::time::timeout(std::time::Duration::from_secs(5), lines.next_line())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let base = line.strip_prefix("q.it dashboard listening on ").unwrap();
    let detail: Value = reqwest::get(format!(
        "{base}/api/runs/{}",
        row["run"]["id"].as_str().unwrap()
    ))
    .await
    .unwrap()
    .json()
    .await
    .unwrap();
    assert_eq!(
        detail["samples"][0]["output_metadata"],
        sample["output_metadata"]
    );
    dashboard.kill().await.unwrap();
    dashboard.wait().await.unwrap();
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invalid_output_evidence_and_unstated_onnx_bounds_are_refused() {
    let temporary = TempDir::new().unwrap();
    let state = temporary.path().join("state");
    for (mode, expected) in [
        ("array-metadata", "output metadata must be an object"),
        ("huge-metadata", "output metadata must be an object"),
        ("missing-basis", "fewer than one billion loaded parameters"),
        (
            "oversized-bound",
            "fewer than one billion loaded parameters",
        ),
    ] {
        let (endpoint, server) = provider(mode).await;
        let output = run(&state, &endpoint).await;
        assert_eq!(
            output.status.code(),
            Some(2),
            "{mode}: {}",
            String::from_utf8_lossy(&output.stdout)
        );
        let reported = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(reported.contains(expected), "{mode}: {reported}");
        server.abort();
    }
    let history = command(&state)
        .args(["results", "--limit", "100"])
        .output()
        .await
        .unwrap();
    let history = String::from_utf8_lossy(&history.stdout);
    assert_eq!(history.matches("failed").count(), 4);
}

fn command(state: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_qit"));
    command.env("QIT_HOME", state).env_remove("QIT_API_KEY");
    command
}

async fn provider(mode: &'static str) -> (String, tokio::task::JoinHandle<std::io::Result<()>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(
        axum::serve(
            listener,
            Router::new()
                .route("/v1/qit/model", get(metadata))
                .route("/v1/qit/invoke", post(invoke))
                .with_state(mode),
        )
        .into_future(),
    );
    (endpoint, server)
}

async fn run(state: &Path, endpoint: &str) -> std::process::Output {
    command(state)
        .args([
            "run",
            "core/image-to-text-smoke",
            "--provider",
            "local-cpu",
            "--model",
            MODEL,
            "--base-url",
            endpoint,
            "--warmups",
            "0",
            "--iterations",
            "1",
        ])
        .output()
        .await
        .unwrap()
}

async fn export(state: &Path) -> String {
    let output = command(state)
        .args(["export", "--format", "jsonl"])
        .output()
        .await
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    String::from_utf8(output.stdout).unwrap()
}

async fn metadata(State(mode): State<&'static str>) -> Json<Value> {
    let (count, basis, bound) = match mode {
        "missing-basis" => (Value::Null, Value::Null, json!(300_000_000)),
        "oversized-bound" => (
            Value::Null,
            json!("onnx-full-static-constant-upper-bound-v1"),
            json!(2_000_000_000),
        ),
        _ => (
            Value::Null,
            json!("onnx-full-static-constant-upper-bound-v1"),
            json!(300_000_000),
        ),
    };
    Json(json!({
        "schema_version": 1,
        "version": "0.1.0",
        "model": MODEL,
        "configuration": {
            "device": "cpu",
            "task": "image_to_text",
            "precision": "float32",
            "dtype": "float32",
            "revision": REVISION,
            "parameter_count": count,
            "parameter_count_basis": basis,
            "parameter_count_upper_bound": bound,
        },
    }))
}

async fn invoke(State(mode): State<&'static str>, Json(body): Json<Value>) -> String {
    assert_eq!(body["task"], "image_to_text");
    let mut metadata = json!({
        "schema_version": 1,
        "route": "indra-full-ocr",
        "stage_calls": {"det": 1, "rec": 1},
    });
    if mode == "array-metadata" {
        metadata = json!([1, 2, 3]);
    }
    if mode == "huge-metadata" {
        metadata = json!({"route": "x".repeat(70_000)});
    }
    format!(
        "{}\n",
        json!({
            "type": "result",
            "output_text": "red and blue",
            "output_tokens": 3,
            "output_metadata": metadata,
        })
    )
}
