use std::future::IntoFuture;
use std::path::Path;

use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::process::Command;

fn transcript() -> String {
    "THE BIRCH CANOE SLID ON THE SMOOTH PLANKS. ".repeat(30)
}

fn command(state: &Path) -> Command {
    let binary =
        std::env::var_os("QIT_TEST_BINARY").unwrap_or_else(|| env!("CARGO_BIN_EXE_qit").into());
    let mut command = Command::new(binary);
    command
        .current_dir(state.parent().unwrap())
        .env("QIT_HOME", state)
        .env_remove("QIT_API_KEY")
        .kill_on_drop(true);
    command
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn speech_transcripts_and_duration_survive_the_installed_command_and_export() {
    let temporary = TempDir::new().unwrap();
    let state = temporary.path().join("state");
    let pack = temporary.path().join("pack");
    std::fs::create_dir(&pack).unwrap();
    let tone: Value = serde_json::from_str(
        include_str!("../../qit-runtime/benchmarks/core/speech-to-text-smoke.jsonl")
            .lines()
            .next()
            .unwrap(),
    )
    .unwrap();
    std::fs::write(pack.join("pack.json"), serde_json::to_vec(&json!({
        "schema_version":1,"id":"speech","name":"Speech","version":"1.0.0",
        "description":"Transcript transport","publisher":"fixture","license":"CC0-1.0","source":"local fixture",
        "benchmarks":[{"id":"transcript","name":"Transcript","description":"Full transcript retention","task":"speech_to_text","cases":"cases.jsonl"}]
    })).unwrap()).unwrap();
    std::fs::write(
        pack.join("cases.jsonl"),
        serde_json::to_vec(&json!({
            "id":"long","media":tone["media"],"expected_text":transcript()
        }))
        .unwrap(),
    )
    .unwrap();
    let installed = command(&state)
        .args(["benchmark", "install"])
        .arg(&pack)
        .output()
        .await
        .unwrap();
    assert!(installed.status.success(), "{installed:?}");
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(
        axum::serve(
            listener,
            Router::new()
                .route("/v1/qit/model", get(metadata))
                .route("/v1/qit/invoke", post(invoke)),
        )
        .into_future(),
    );
    let output = command(&state)
        .args([
            "run",
            "speech/transcript",
            "--provider",
            "local-cpu",
            "--model",
            "speech",
            "--base-url",
            &endpoint,
            "--iterations",
            "1",
            "--warmups",
            "0",
        ])
        .output()
        .await
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    assert!(command(&state)
        .args(["benchmark", "remove", "speech"])
        .output()
        .await
        .unwrap()
        .status
        .success());
    let exported = command(&state)
        .args(["export", "--format", "jsonl"])
        .output()
        .await
        .unwrap();
    assert!(exported.status.success(), "{exported:?}");
    let row: Value = serde_json::from_slice(&exported.stdout).unwrap();
    let sample = &row["samples"][0];
    assert_eq!(sample["speech"]["transcript"], transcript());
    assert_eq!(sample["speech"]["audio_duration_seconds"], 4.5);
    assert_eq!(sample["speech"]["routing"]["selected_route"], "english");
    assert_eq!(row["run"]["summary"]["quality_score_mean"], 1.0);
    for key in [
        "input_tokens",
        "output_tokens",
        "ttft_ms",
        "tokens_per_second",
    ] {
        assert!(sample[key].is_null(), "{key}: {sample}");
    }
    let mut manifest: Value =
        serde_json::from_slice(&std::fs::read(pack.join("pack.json")).unwrap()).unwrap();
    manifest["version"] = json!("2.0.0");
    std::fs::write(
        pack.join("pack.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    std::fs::write(
        pack.join("cases.jsonl"),
        serde_json::to_vec(&json!({"id":"invalid-duration","media":tone["media"]})).unwrap(),
    )
    .unwrap();
    assert!(command(&state)
        .args(["benchmark", "install"])
        .arg(&pack)
        .output()
        .await
        .unwrap()
        .status
        .success());
    let invalid = command(&state)
        .args([
            "run",
            "speech/transcript",
            "--provider",
            "local-cpu",
            "--model",
            "speech",
            "--base-url",
            &endpoint,
            "--iterations",
            "1",
            "--warmups",
            "0",
        ])
        .output()
        .await
        .unwrap();
    assert_eq!(invalid.status.code(), Some(2), "{invalid:?}");
    let exported = command(&state)
        .args(["export", "--format", "jsonl"])
        .output()
        .await
        .unwrap();
    assert_eq!(
        String::from_utf8(exported.stdout).unwrap().lines().count(),
        1
    );
    let history = command(&state).args(["results"]).output().await.unwrap();
    assert!(String::from_utf8_lossy(&history.stdout).contains("failed"));
    server.abort();
}

async fn metadata() -> Json<Value> {
    Json(
        json!({"schema_version":1,"version":"fixture","model":"speech","configuration":{"task":"speech_to_text","device":"cpu","dtype":"float32","parameter_count":10,"revision":"fixture"}}),
    )
}

async fn invoke(Json(payload): Json<Value>) -> String {
    let duration = if payload["case"]["id"] == "invalid-duration" {
        -1.0
    } else {
        4.5
    };
    format!(
        "{}\n",
        json!({"type":"result","output_text":transcript(),"audio_duration_seconds":duration,"speech_routing":{"requested_mode":"automatic","selected_route":"english","p_non_english":0.2,"threshold":0.8582},"output_tokens":999,"first_token_latency_ns":1})
    )
}
