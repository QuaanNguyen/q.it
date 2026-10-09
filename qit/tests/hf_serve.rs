use std::path::Path;

use axum::http::{HeaderMap, StatusCode};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::process::Command;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hf_serve_embeddings_preserve_input_order_usage_and_failure_history() {
    let temporary = TempDir::new().unwrap();
    let state = temporary.path().join("state");
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        axum::serve(
            listener,
            Router::new()
                .route("/v1/models", get(models))
                .route("/v1/embeddings", post(embeddings)),
        )
        .await
        .unwrap();
    });

    let providers = command(&state)
        .args(["provider", "list"])
        .output()
        .await
        .unwrap();
    assert!(providers.status.success());
    assert!(String::from_utf8_lossy(&providers.stdout).contains("hf-serve"));
    let benchmarks = command(&state)
        .args(["benchmark", "list", "--provider", "hf-serve"])
        .output()
        .await
        .unwrap();
    assert!(benchmarks.status.success());
    let listing = String::from_utf8_lossy(&benchmarks.stdout);
    assert!(listing
        .lines()
        .any(|line| line.starts_with("core/embedding-retrieval") && line.ends_with("supported")));
    assert!(listing.lines().any(
        |line| line.starts_with("core/text-generation-smoke") && line.ends_with("unsupported")
    ));

    for model in ["fixture-model", "no-usage"] {
        let output = run(&state, &base_url, model, "core/embedding-retrieval").await;
        assert!(output.status.success(), "{output:?}");
    }
    let pack = temporary.path().join("pack");
    std::fs::create_dir(&pack).unwrap();
    std::fs::write(pack.join("pack.json"), serde_json::to_vec(&json!({
        "schema_version": 1, "id": "batched", "name": "Batched retrieval", "version": "1.0.0",
        "description": "Exercises retrieval across request batches", "publisher": "fixture", "license": "MIT", "source": "local fixture",
        "benchmarks": [{"id": "retrieval", "name": "Batched retrieval", "description": "Relevant document in the final batch", "task": "embedding", "cases": "cases.jsonl"}]
    })).unwrap()).unwrap();
    std::fs::write(pack.join("cases.jsonl"), serde_json::to_vec(&json!({
        "id": "last-batch", "input": "weather forecast",
        "documents": ["Unrelated one", "Unrelated two", "Unrelated three", "Unrelated four", "Unrelated five", "Unrelated six", "Unrelated seven", "Tomorrow may bring rain"],
        "expected_index": 7
    })).unwrap()).unwrap();
    let install = command(&state)
        .args(["benchmark", "install"])
        .arg(&pack)
        .output()
        .await
        .unwrap();
    assert!(install.status.success(), "{install:?}");
    let batched = run(&state, &base_url, "fixture-model", "batched/retrieval").await;
    assert!(batched.status.success(), "{batched:?}");
    for model in [
        "duplicate-index",
        "missing-index",
        "out-of-range",
        "missing-vector",
        "empty-vector",
        "wrong-dimensions",
        "http-error",
        "not-loaded",
    ] {
        let output = run(&state, &base_url, model, "core/embedding-retrieval").await;
        assert_eq!(output.status.code(), Some(2), "{model}: {output:?}");
        assert!(String::from_utf8_lossy(&output.stdout).contains("Status: failed"));
    }
    let unsupported = run(
        &state,
        &base_url,
        "fixture-model",
        "core/text-generation-smoke",
    )
    .await;
    assert_eq!(unsupported.status.code(), Some(2), "{unsupported:?}");

    let exported = command(&state)
        .args(["export", "--format", "jsonl"])
        .output()
        .await
        .unwrap();
    assert!(exported.status.success(), "{exported:?}");
    let records = String::from_utf8(exported.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(records.len(), 3);
    for record in &records {
        assert_eq!(record["run"]["provider"], "hf_serve");
        assert_eq!(record["run"]["status"], "succeeded");
        assert_eq!(record["run"]["summary"]["quality_score_mean"], 1.0);
        let samples = record["samples"].as_array().unwrap();
        let batched = record["run"]["benchmark_id"] == "batched/retrieval";
        assert_eq!(samples.len(), if batched { 1 } else { 3 });
        for sample in samples {
            assert_eq!(sample["quality_score"], 1.0);
            assert!(sample["ttft_ms"].is_null());
            assert!(sample["tokens_per_second"].is_null());
            assert!(sample["output_tokens"].is_null());
            if record["run"]["model"] == "fixture-model" {
                assert_eq!(sample["input_tokens"], if batched { 126 } else { 42 });
            } else {
                assert!(sample["input_tokens"].is_null());
            }
        }
    }
    let history = command(&state)
        .args(["results", "--limit", "100"])
        .output()
        .await
        .unwrap();
    assert!(history.status.success());
    let history = String::from_utf8_lossy(&history.stdout);
    assert_eq!(history.matches("succeeded").count(), 3);
    assert_eq!(history.matches("failed").count(), 9);
    assert!(history.contains("not-loaded"));
    server.abort();
}

fn command(state: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_qit"));
    command.env("QIT_HOME", state).env_remove("QIT_API_KEY");
    command
}

async fn run(state: &Path, base_url: &str, model: &str, benchmark: &str) -> std::process::Output {
    command(state)
        .args([
            "run",
            benchmark,
            "--provider",
            "hf-serve",
            "--model",
            model,
            "--base-url",
            base_url,
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

async fn models(headers: HeaderMap) -> Result<Json<Value>, StatusCode> {
    authorize(&headers)?;
    Ok(Json(json!({
        "object": "list",
        "data": (["fixture-model", "no-usage", "duplicate-index", "missing-index", "out-of-range", "missing-vector", "empty-vector", "wrong-dimensions", "http-error"]
            .map(|model| json!({"id": model, "object": "model"})))
    })))
}

async fn embeddings(
    headers: HeaderMap,
    Json(payload): Json<Value>,
) -> Result<Json<Value>, StatusCode> {
    authorize(&headers)?;
    assert_eq!(payload["encoding_format"], "float");
    let model = payload["model"].as_str().unwrap();
    let inputs = payload["input"].as_array().unwrap();
    assert!(!inputs.is_empty() && inputs.len() <= 4);
    if model == "http-error" {
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }
    let mut data = inputs
        .iter()
        .enumerate()
        .map(|(index, input)| {
            let text = input.as_str().unwrap();
            let relevant = matches!(
                text,
                "weather forecast"
                    | "small single-board computer"
                    | "reduce application memory consumption"
            ) || text.contains("Tomorrow")
                || text.contains("Raspberry Pi")
                || text.contains("Profile allocations");
            let vector = if relevant { [1.0, 0.0] } else { [0.0, 1.0] };
            json!({"object": "embedding", "index": index, "embedding": vector})
        })
        .collect::<Vec<_>>();
    match model {
        "duplicate-index" => data[1]["index"] = json!(0),
        "missing-index" => {
            data[0].as_object_mut().unwrap().remove("index");
        }
        "out-of-range" => data[0]["index"] = json!(4),
        "missing-vector" => {
            data.pop();
        }
        "empty-vector" => data[0]["embedding"] = json!([]),
        "wrong-dimensions" => data[1]["embedding"] = json!([1.0]),
        _ => {}
    }
    data.reverse();
    let mut response = json!({"object": "list", "model": model, "data": data});
    if model != "no-usage" {
        response["usage"] = json!({"prompt_tokens": 42, "total_tokens": 42});
    }
    Ok(Json(response))
}

fn authorize(headers: &HeaderMap) -> Result<(), StatusCode> {
    if headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        == Some("Bearer fixture-token")
    {
        Ok(())
    } else {
        Err(StatusCode::UNAUTHORIZED)
    }
}
