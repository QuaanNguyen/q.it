use std::future::IntoFuture;
use std::path::Path;

use axum::routing::{get, post};
use axum::{Json, Router};
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::process::Command;

const SAMPLE_RATE_HZ: u32 = 24_000;
const AUDIO_BYTES: usize = 24_000;

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
async fn synthesized_audio_retains_first_audio_duration_rate_bytes_and_routing() {
    let temporary = TempDir::new().unwrap();
    let state = temporary.path().join("state");
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
            "core/text-to-speech-smoke",
            "--provider",
            "local-cpu",
            "--model",
            "phonon-fixture",
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
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("First audio p50:"), "{stdout}");
    assert!(stdout.contains("Real-time factor p50:"), "{stdout}");

    let exported = command(&state)
        .args(["export", "--format", "jsonl"])
        .output()
        .await
        .unwrap();
    assert!(exported.status.success(), "{exported:?}");
    let row: Value = serde_json::from_slice(&exported.stdout).unwrap();
    let sample = &row["samples"][0];
    assert_eq!(sample["synthesis"]["sample_rate_hz"], SAMPLE_RATE_HZ);
    assert_eq!(sample["synthesis"]["audio_bytes"], AUDIO_BYTES);
    assert_eq!(sample["synthesis"]["audio_duration_seconds"], 0.5);
    assert_eq!(sample["synthesis"]["routing"]["engine"], "phonon");
    assert!(sample["synthesis"]["time_to_first_audio_ms"]
        .as_f64()
        .is_some_and(|value| value >= 0.0));
    for key in ["ttft_ms", "input_tokens", "output_tokens", "quality_score"] {
        assert!(sample[key].is_null(), "{key}: {sample}");
    }
    let summary = &row["run"]["summary"];
    assert_eq!(summary["audio_duration_seconds_p50"], 0.5);
    assert!(summary["time_to_first_audio_ms_p50"].is_f64());
    assert!(summary["real_time_factor_p50"]
        .as_f64()
        .is_some_and(|value| value > 0.0));

    let csv = command(&state)
        .args(["export", "--format", "csv"])
        .output()
        .await
        .unwrap();
    let csv = String::from_utf8(csv.stdout).unwrap();
    assert!(csv
        .lines()
        .next()
        .unwrap()
        .ends_with("time_to_first_audio_ms_p50,real_time_factor_p50,audio_duration_seconds_p50"));
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streams_without_audio_or_with_inconsistent_byte_counts_fail_the_run() {
    let temporary = TempDir::new().unwrap();
    let state = temporary.path().join("state");
    let pack = temporary.path().join("pack");
    std::fs::create_dir(&pack).unwrap();
    std::fs::write(
        pack.join("pack.json"),
        serde_json::to_vec(&json!({
            "schema_version":1,"id":"speech","name":"Speech","version":"1.0.0",
            "description":"Synthesis transport","publisher":"fixture","license":"CC0-1.0","source":"local fixture",
            "benchmarks":[
                {"id":"silent","name":"Silent","description":"No audio","task":"text_to_speech","cases":"silent.jsonl"},
                {"id":"mismatch","name":"Mismatch","description":"Wrong bytes","task":"text_to_speech","cases":"mismatch.jsonl"}
            ]
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        pack.join("silent.jsonl"),
        serde_json::to_vec(&json!({"id":"missing-audio","prompt":"silence"})).unwrap(),
    )
    .unwrap();
    std::fs::write(
        pack.join("mismatch.jsonl"),
        serde_json::to_vec(&json!({"id":"byte-mismatch","prompt":"mismatch"})).unwrap(),
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
    for benchmark in ["speech/silent", "speech/mismatch"] {
        let output = command(&state)
            .args([
                "run",
                benchmark,
                "--provider",
                "local-cpu",
                "--model",
                "phonon-fixture",
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
        assert_eq!(output.status.code(), Some(2), "{benchmark}: {output:?}");
    }
    let exported = command(&state)
        .args(["export", "--format", "jsonl"])
        .output()
        .await
        .unwrap();
    let rows = String::from_utf8(exported.stdout).unwrap();
    assert!(rows.is_empty(), "{rows}");
    let results = command(&state).args(["results"]).output().await.unwrap();
    let results = String::from_utf8(results.stdout).unwrap();
    assert_eq!(results.matches("failed").count(), 2, "{results}");
    server.abort();
}

async fn metadata() -> Json<Value> {
    Json(
        json!({"schema_version":1,"version":"fixture","model":"phonon-fixture","configuration":{"task":"text_to_speech","device":"cpu","dtype":"float32","parameter_count":41_096_035,"revision":"fixture"}}),
    )
}

async fn invoke(Json(payload): Json<Value>) -> String {
    let case_id = payload["case"]["id"].as_str().unwrap_or_default();
    let audio = format!(
        "{}\n",
        json!({"type":"audio","pcm_base64":STANDARD.encode(vec![0u8; AUDIO_BYTES]),"sample_rate_hz":SAMPLE_RATE_HZ})
    );
    if case_id == "missing-audio" {
        return format!(
            "{}\n",
            json!({"type":"result","audio_duration_seconds":0.5,"sample_rate_hz":SAMPLE_RATE_HZ,"audio_bytes":AUDIO_BYTES})
        );
    }
    let reported = if case_id == "byte-mismatch" {
        AUDIO_BYTES - 2
    } else {
        AUDIO_BYTES
    };
    format!(
        "{audio}{}\n",
        json!({"type":"result","audio_duration_seconds":0.5,"sample_rate_hz":SAMPLE_RATE_HZ,"audio_bytes":reported,"speech_routing":{"engine":"phonon"}})
    )
}
