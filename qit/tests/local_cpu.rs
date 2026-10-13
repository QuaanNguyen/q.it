use std::future::IntoFuture;
use std::path::Path;

use axum::extract::Path as RoutePath;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::process::Command;

const PNG: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAfElEQVR4nNXOQREAMAjAsK7+PTMRPLhGQd7QJnESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ES53Vg6wNShQF/fRSLfgAAAABJRU5ErkJggg==";

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn local_cpu_runs_every_task_preserves_images_and_rejects_invalid_provider_output() {
    let temporary = TempDir::new().unwrap();
    let state = temporary.path().join("state");
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
    let pack = temporary.path().join("pack");
    std::fs::create_dir(&pack).unwrap();
    std::fs::write(pack.join("pack.json"), serde_json::to_vec(&json!({
        "schema_version":1,"id":"cpu-images","name":"CPU images","version":"1.0.0",
        "description":"Deterministic image output transport","publisher":"fixture","license":"CC0-1.0","source":"local fixture",
        "benchmarks":[{"id":"image","name":"Image","description":"Generated PNG","task":"text_to_image","cases":"cases.jsonl"}]
    })).unwrap()).unwrap();
    std::fs::write(pack.join("cases.jsonl"), serde_json::to_vec(&json!({
        "id":"red","prompt":"A red square","image_settings":{"width":64,"height":64,"steps":2,"guidance_scale":7.5,"seed":42}
    })).unwrap()).unwrap();
    let install = command(&state)
        .args(["benchmark", "install"])
        .arg(pack)
        .output()
        .await
        .unwrap();
    assert!(install.status.success(), "{install:?}");
    let benchmarks = [
        ("text_generation", "core/text-generation-smoke"),
        ("embedding", "core/embedding-retrieval"),
        ("reranking", "core/reranking-relevance"),
        ("image_to_text", "core/image-to-text-smoke"),
        ("speech_to_text", "core/speech-to-text-smoke"),
        ("text_to_image", "cpu-images/image"),
    ];
    for (model, benchmark) in benchmarks {
        let output = run(&state, &endpoint, model, benchmark).await;
        assert!(output.status.success(), "{model}: {output:?}");
    }
    for (model, benchmark) in [
        ("text_to_image:bad-png", "cpu-images/image"),
        ("text_to_image:wrong-dimensions", "cpu-images/image"),
        ("reranking:duplicate", "core/reranking-relevance"),
        ("text_generation:truncated", "core/text-generation-smoke"),
        ("text_generation:error", "core/text-generation-smoke"),
        ("embedding:missing", "core/embedding-retrieval"),
        ("embedding:gpu", "core/embedding-retrieval"),
        ("embedding:oversized", "core/embedding-retrieval"),
        ("embedding:wrong-model", "core/embedding-retrieval"),
        (
            "speech_to_text:missing-transcript",
            "core/speech-to-text-smoke",
        ),
    ] {
        let output = run(&state, &endpoint, model, benchmark).await;
        assert_eq!(output.status.code(), Some(2), "{model}: {output:?}");
    }
    let exported = command(&state)
        .args(["export", "--format", "jsonl"])
        .output()
        .await
        .unwrap();
    assert!(exported.status.success());
    let rows = String::from_utf8(exported.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 6);
    for row in rows {
        assert_eq!(
            row["run"]["provider_metadata"]["configuration"]["device"],
            "cpu"
        );
        assert_eq!(
            row["run"]["provider_metadata"]["configuration"]["revision"],
            "fixture-revision"
        );
        for sample in row["samples"].as_array().unwrap() {
            if row["run"]["task"] == "text_to_image" {
                assert_eq!(sample["output_image"], PNG);
                assert!(sample["quality_score"].is_null());
            }
            if matches!(
                row["run"]["task"].as_str(),
                Some("embedding" | "reranking" | "speech_to_text" | "text_to_image")
            ) {
                assert!(sample["ttft_ms"].is_null());
                assert!(sample["tokens_per_second"].is_null());
            } else {
                assert_eq!(sample["ttft_ms"], 20.0);
                assert_eq!(sample["tokens_per_second"], 50.0);
            }
        }
    }
    let history = command(&state)
        .args(["results", "--limit", "100"])
        .output()
        .await
        .unwrap();
    let history = String::from_utf8_lossy(&history.stdout);
    assert_eq!(history.matches("failed").count(), 10);
    server.abort();
}

fn command(state: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_qit"));
    command.env("QIT_HOME", state).env_remove("QIT_API_KEY");
    command
}

async fn run(state: &Path, endpoint: &str, model: &str, benchmark: &str) -> std::process::Output {
    command(state)
        .args([
            "run",
            benchmark,
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

async fn metadata(RoutePath(model): RoutePath<String>) -> Json<Value> {
    let task = model.split(':').next().unwrap();
    let device = if model.ends_with(":gpu") {
        "cuda"
    } else {
        "cpu"
    };
    let count = if model.ends_with(":oversized") {
        1_000_000_000
    } else {
        10
    };
    let identity = if model.ends_with(":wrong-model") {
        "other"
    } else {
        &model
    };
    Json(
        json!({"schema_version":1,"version":"0.1.0","model":identity,
        "configuration":{"device":device,"task":task,"parameter_count":count,"revision":"fixture-revision"}}),
    )
}

async fn invoke(RoutePath(model): RoutePath<String>, Json(body): Json<Value>) -> String {
    let task = body["task"].as_str().unwrap();
    let case = &body["case"];
    assert_eq!(body["model"], model);
    if matches!(task, "image_to_text" | "speech_to_text") {
        assert!(case["media"].as_str().unwrap().starts_with("data:"));
    }
    if model.ends_with(":error") {
        return "{\"type\":\"error\",\"error\":\"fixture failure\"}\n".into();
    }
    if model.ends_with(":truncated") {
        return "{\"type\":\"delta\",\"text\":\"partial\"}\n".into();
    }
    let mut result = json!({"type":"result"});
    match task {
        "embedding" => {
            let count = case["documents"].as_array().unwrap().len() + 1;
            result["embeddings"] = json!(vec![
                [1.0, 0.0];
                if model.ends_with(":missing") {
                    count - 1
                } else {
                    count
                }
            ]);
        }
        "reranking" => {
            result["ranking"] = if model.ends_with(":duplicate") {
                json!([0, 0, 1])
            } else {
                json!((0..case["documents"].as_array().unwrap().len()).collect::<Vec<_>>())
            };
        }
        "text_to_image" => {
            result["output_image"] = json!(if model.ends_with(":bad-png") {
                "data:image/png;base64,aGVsbG8="
            } else if model.ends_with(":wrong-dimensions") {
                "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jYZkAAAAASUVORK5CYII="
            } else {
                PNG
            });
        }
        _ => {
            if !model.ends_with(":missing-transcript") {
                result["output_text"] = json!("raspberry 42 edge red blue");
            }
            result["input_tokens"] = json!(2);
            result["output_tokens"] = json!(6);
        }
    }
    let mut stream = String::new();
    if matches!(task, "text_generation" | "image_to_text") {
        result["decode_duration_ns"] = json!(100_000_000);
        result["first_token_latency_ns"] = json!(20_000_000);
        stream.push_str("{\"type\":\"delta\",\"text\":\"raspberry\"}\n");
    }
    stream.push_str(&result.to_string());
    stream.push('\n');
    stream
}
