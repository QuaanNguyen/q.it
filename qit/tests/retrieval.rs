use std::future::IntoFuture;
use std::path::Path;
use std::process::Stdio;

use axum::extract::Path as RoutePath;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::net::TcpListener;
use tokio::process::Command;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_retrieval_reaches_history_export_and_the_dashboard() {
    let temporary = TempDir::new().unwrap();
    let state = temporary.path().join("state");
    let (endpoint, server) = provider().await;
    let output = run(&state, &endpoint, "maxsim").await;
    assert!(output.status.success(), "{output:?}");
    let row: Value = serde_json::from_str(&export(&state, "jsonl").await).unwrap();
    assert_eq!(row["run"]["task"], "embedding");
    assert_eq!(row["run"]["status"], "succeeded");
    let samples = row["samples"].as_array().unwrap();
    assert_eq!(samples.len(), 3);
    for sample in samples {
        let retrieval = &sample["retrieval"];
        assert_eq!(retrieval["schema_version"], 1);
        assert_eq!(retrieval["method"], "maxsim");
        assert_eq!(retrieval["tie_break"], "document_order");
        assert_eq!(retrieval["query"].as_array().unwrap().len(), 1);
        assert_eq!(retrieval["documents"].as_array().unwrap().len(), 3);
        assert_eq!(retrieval["scores"], json!([3.0, 2.0, 1.0]));
        assert_eq!(retrieval["ranking"], json!([0, 1, 2]));
        assert_eq!(
            retrieval["query"].as_array().unwrap()[0]
                .as_array()
                .unwrap()
                .len(),
            128
        );
        for document in retrieval["documents"].as_array().unwrap() {
            for vector in document.as_array().unwrap() {
                assert_eq!(vector.as_array().unwrap().len(), 128);
            }
            assert_eq!(document.as_array().unwrap().len(), 1);
        }
        assert!(sample["output_excerpt"]
            .as_str()
            .unwrap()
            .starts_with("MaxSim scores"));
    }
    let quality: Vec<(String, f64)> = samples
        .iter()
        .map(|sample| {
            (
                sample["case_id"].as_str().unwrap().to_string(),
                sample["quality_score"].as_f64().unwrap(),
            )
        })
        .collect();
    assert_eq!(
        quality,
        vec![
            ("memory".to_string(), 1.0),
            ("raspberry-pi".to_string(), 0.5),
            ("weather".to_string(), 1.0),
        ]
    );
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
        detail["samples"][0]["retrieval"],
        row["samples"][0]["retrieval"]
    );
    dashboard.kill().await.unwrap();
    dashboard.wait().await.unwrap();
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invalid_native_retrieval_is_refused_before_it_is_stored() {
    let temporary = TempDir::new().unwrap();
    let state = temporary.path().join("state");
    let (endpoint, server) = provider().await;
    for (model, expected) in [
        ("maxsim:dense-result", "differs from its declared method"),
        ("dense:retrieval-result", "differs from its declared method"),
        ("maxsim:both", "must not also supply dense embeddings"),
        (
            "maxsim:score",
            "differs from MaxSim over the retained vectors",
        ),
        (
            "maxsim:ranking",
            "must match native scores with document-order ties",
        ),
        ("maxsim:width", "finite native 128-dimensional vectors"),
        (
            "maxsim:empty-query",
            "must retain the query and every candidate document",
        ),
    ] {
        let output = run(&state, &endpoint, model).await;
        assert_eq!(
            output.status.code(),
            Some(2),
            "{model}: code={:?} stdout={} stderr={}",
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let reported = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(reported.contains(expected), "{model}: {reported}");
    }
    let history = command(&state)
        .args(["results", "--limit", "100"])
        .output()
        .await
        .unwrap();
    let history = String::from_utf8_lossy(&history.stdout);
    assert_eq!(history.matches("failed").count(), 7);
    server.abort();
}

fn command(state: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_qit"));
    command.env("QIT_HOME", state).env_remove("QIT_API_KEY");
    command
}

async fn provider() -> (String, tokio::task::JoinHandle<std::io::Result<()>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(
        axum::serve(
            listener,
            Router::new()
                .route("/{model}/v1/qit/model", get(metadata))
                .route("/{model}/v1/qit/invoke", post(invoke)),
        )
        .into_future(),
    );
    (endpoint, server)
}

async fn run(state: &Path, endpoint: &str, model: &str) -> std::process::Output {
    command(state)
        .args([
            "run",
            "core/embedding-retrieval",
            "--provider",
            "local-cpu",
            "--model",
            model,
            "--base-url",
            &format!("{endpoint}/{model}"),
            "--warmups",
            "0",
            "--iterations",
            "1",
        ])
        .output()
        .await
        .unwrap()
}

async fn export(state: &Path, format: &str) -> String {
    let output = command(state)
        .args(["export", "--format", format])
        .output()
        .await
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    String::from_utf8(output.stdout).unwrap()
}

async fn metadata(RoutePath(model): RoutePath<String>) -> Json<Value> {
    let mut configuration = json!({
        "device": "cpu",
        "task": "embedding",
        "parameter_count": 10,
        "revision": "fixture-revision",
    });
    if !model.starts_with("dense") {
        configuration["retrieval"] = json!({
            "schema_version": 1,
            "method": "maxsim",
            "dimensions": 128,
            "tie_break": "document_order",
        });
    }
    Json(json!({
        "schema_version": 1,
        "version": "0.1.0",
        "model": model,
        "configuration": configuration,
    }))
}

fn vector(value: f64) -> Vec<f64> {
    let mut row = vec![0.0; 128];
    row[0] = value;
    row
}

async fn invoke(RoutePath(model): RoutePath<String>, Json(body): Json<Value>) -> String {
    assert_eq!(body["task"], "embedding");
    assert_eq!(body["model"], model);
    let case = &body["case"];
    assert!(case["input"].is_string());
    assert_eq!(case["documents"].as_array().unwrap().len(), 3);
    let dense = json!({
        "type": "result",
        "embeddings": [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 1.0]],
    });
    if model == "maxsim:dense-result" {
        return format!("{dense}\n");
    }
    let mut query = vec![vector(1.0)];
    let documents = vec![vec![vector(3.0)], vec![vector(2.0)], vec![vector(1.0)]];
    let mut scores = vec![3.0, 2.0, 1.0];
    let mut ranking = vec![0, 1, 2];
    if model.ends_with(":width") {
        query = vec![vec![1.0; 64]];
    }
    if model.ends_with(":empty-query") {
        query = Vec::new();
    }
    if model.ends_with(":score") {
        scores = vec![3.0, 2.0, 9.0];
    }
    if model.ends_with(":ranking") {
        ranking = vec![1, 0, 2];
    }
    let mut result = json!({
        "type": "result",
        "retrieval": {
            "schema_version": 1,
            "method": "maxsim",
            "query": query,
            "documents": documents,
            "scores": scores,
            "ranking": ranking,
            "tie_break": "document_order",
        },
    });
    if model == "maxsim:both" {
        result["embeddings"] = dense["embeddings"].clone();
    }
    format!("{result}\n")
}
