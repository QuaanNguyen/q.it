use std::future::IntoFuture;
use std::path::Path;

use axum::extract::Path as RoutePath;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::process::Command;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_retrieval_and_speech_survive_installed_command_exports() {
    let temporary = TempDir::new().unwrap();
    let state = temporary.path().join("state");
    let (endpoint, server) = provider().await;
    for (variant, benchmark, succeeds) in [
        ("retrieval", "core/embedding-retrieval", true),
        ("retrieval-bad-score", "core/embedding-retrieval", false),
        ("retrieval-bad-ranking", "core/embedding-retrieval", false),
        ("retrieval-undeclared", "core/embedding-retrieval", false),
        ("speech", "core/speech-to-text-smoke", true),
        ("speech-bad-duration", "core/speech-to-text-smoke", false),
        ("indra", "core/image-to-text-smoke", true),
        ("indra-wrong-revision", "core/image-to-text-smoke", false),
        ("indra-oversized", "core/image-to-text-smoke", false),
        (
            "indra-bad-output-metadata",
            "core/image-to-text-smoke",
            false,
        ),
        (
            "indra-oversized-output-metadata",
            "core/image-to-text-smoke",
            false,
        ),
    ] {
        let run = run(&state, &endpoint, variant, benchmark).await;
        assert_eq!(run.status.success(), succeeds, "{variant}: {run:?}");
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
    assert_eq!(rows.len(), 3);
    for row in &rows {
        for sample in row["samples"].as_array().unwrap() {
            assert!(sample["ttft_ms"].is_null());
            assert!(sample["tokens_per_second"].is_null());
            match row["run"]["task"].as_str().unwrap() {
                "embedding" => {
                    assert_eq!(
                        sample["retrieval"]["query"][0].as_array().unwrap().len(),
                        128
                    );
                    assert_eq!(sample["retrieval"]["scores"][0], 1.0);
                    assert_eq!(sample["retrieval"]["ranking"][0], 0);
                    assert!(sample["quality_score"].is_number());
                }
                "speech_to_text" => {
                    assert_eq!(
                        sample["speech"]["transcript"],
                        "complete native transcript ".repeat(50)
                    );
                    assert_eq!(sample["speech"]["audio_duration_seconds"], 1.25);
                    assert_eq!(sample["speech"]["routing"]["language"], "en-US");
                }
                "image_to_text" => {
                    assert_eq!(sample["output_metadata"]["stage_calls"]["det"], 1);
                    assert_eq!(sample["output_metadata"]["stage_calls"]["rec"], 2);
                    assert!(
                        row["run"]["provider_metadata"]["configuration"]["parameter_count"]
                            .is_null()
                    );
                    assert_eq!(
                        row["run"]["provider_metadata"]["configuration"]
                            ["parameter_count_upper_bound"],
                        100
                    );
                }
                _ => panic!("unexpected task"),
            }
        }
    }
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bf16_admission_is_limited_to_the_two_original_d3_revisions() {
    let temporary = TempDir::new().unwrap();
    let state = temporary.path().join("state");
    let pack = temporary.path().join("pack");
    std::fs::create_dir(&pack).unwrap();
    std::fs::write(pack.join("pack.json"), serde_json::to_vec(&json!({
        "schema_version":1,"id":"decisions","name":"Decisions","version":"1.0.0",
        "description":"Single-choice fixture","publisher":"fixture","license":"MIT","source":"fixture",
        "benchmarks":[{"id":"choice","name":"Choice","description":"Choice","task":"decision_scoring","decision_input":"text","cases":"cases.jsonl"}]
    })).unwrap()).unwrap();
    std::fs::write(pack.join("cases.jsonl"), serde_json::to_vec(&json!({
        "id":"one","decision":{"state":"A clear sky","question":"Which fits?","candidates":[{"id":"a","text":"Clear"},{"id":"b","text":"Rain"}],"expected":"a"}
    })).unwrap()).unwrap();
    assert!(command(&state)
        .args(["benchmark", "install"])
        .arg(&pack)
        .output()
        .await
        .unwrap()
        .status
        .success());
    let (endpoint, server) = provider().await;
    for (variant, succeeds) in [
        ("d3-edge", true),
        ("d3-lite", true),
        ("d3-wrong-revision", false),
        ("d3-no-admission", false),
        ("d3-other-model", false),
    ] {
        let result = run(&state, &endpoint, variant, "decisions/choice").await;
        assert_eq!(result.status.success(), succeeds, "{variant}: {result:?}");
    }
    server.abort();
}

fn model(variant: &str) -> &str {
    if variant.starts_with("retrieval") {
        "fixture/retrieval"
    } else if variant.starts_with("indra") {
        "datalab-to/indra-ocr"
    } else if variant == "d3-lite" {
        "vllm-sr/d3-lite"
    } else if variant == "d3-other-model" {
        "fixture/unadmitted"
    } else if variant.starts_with("d3") {
        "vllm-sr/d3-edge"
    } else {
        "moondream/parakeet-ultra"
    }
}

fn command(state: &Path) -> Command {
    let binary =
        std::env::var_os("QIT_TEST_BINARY").unwrap_or_else(|| env!("CARGO_BIN_EXE_qit").into());
    let mut command = Command::new(binary);
    command
        .env("QIT_HOME", state)
        .env_remove("QIT_API_KEY")
        .kill_on_drop(true);
    command
}

async fn run(state: &Path, endpoint: &str, variant: &str, benchmark: &str) -> std::process::Output {
    command(state)
        .args([
            "run",
            benchmark,
            "--provider",
            "local-cpu",
            "--model",
            model(variant),
            "--base-url",
            &format!("{endpoint}/{variant}"),
            "--iterations",
            "1",
            "--warmups",
            "0",
        ])
        .output()
        .await
        .unwrap()
}

async fn provider() -> (String, tokio::task::JoinHandle<Result<(), std::io::Error>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(
        axum::serve(
            listener,
            Router::new()
                .route("/{variant}/v1/qit/model", get(metadata))
                .route("/{variant}/v1/qit/invoke", post(invoke)),
        )
        .into_future(),
    );
    (endpoint, server)
}

async fn metadata(RoutePath(variant): RoutePath<String>) -> Json<Value> {
    let task = if variant.starts_with("retrieval") {
        "embedding"
    } else if variant.starts_with("d3") {
        "decision_scoring"
    } else if variant.starts_with("indra") {
        "image_to_text"
    } else {
        "speech_to_text"
    };
    let mut value = json!({"schema_version":1,"version":"fixture","model":model(&variant),"configuration":{"task":task,"device":"cpu","dtype":"float32","parameter_count":100,"revision":"fixture"}});
    if variant.starts_with("retrieval") && variant != "retrieval-undeclared" {
        value["configuration"]["retrieval"] = json!({"method":"maxsim"});
    }
    if variant.starts_with("indra") {
        value["configuration"]["parameter_count"] = Value::Null;
        value["configuration"]["parameter_count_upper_bound"] =
            json!(if variant == "indra-oversized" {
                1_000_000_000
            } else {
                100
            });
        value["configuration"]["parameter_count_basis"] =
            json!("onnx-full-static-constant-upper-bound-v1");
        value["configuration"]["revision"] = json!(if variant == "indra-wrong-revision" {
            "other"
        } else {
            "ed345c7f158eddbc7dee0c6a7269afb75a94f7e8"
        });
    }
    if variant.starts_with("d3") {
        value["configuration"]["dtype"] = json!("bfloat16");
        value["configuration"]["revision"] = json!(if variant == "d3-lite" {
            "03560c26f5199c3c04129a2ccb38960c913635e0"
        } else if variant == "d3-wrong-revision" {
            "other"
        } else {
            "c454ee39b862248d70abd7f84195d2f54ffe6d43"
        });
        if variant != "d3-no-admission" {
            value["configuration"]["precision_admission"] =
                json!("d3-native-bf16-backbone-fp32-readout-v1");
        }
        value["configuration"]["decision"] = json!({"schema_version":1,"inputs":["text"],"calibration":"native","selection_rule":"candidate_order","preprocessing":"fixture"});
    }
    Json(value)
}

async fn invoke(RoutePath(variant): RoutePath<String>, Json(body): Json<Value>) -> String {
    let value = if variant.starts_with("retrieval") {
        let count = body["case"]["documents"].as_array().unwrap().len();
        let mut a = vec![0.0; 128];
        a[0] = 1.0;
        let mut b = vec![0.0; 128];
        b[1] = 1.0;
        let documents = (0..count)
            .map(|index| vec![if index == 0 { a.clone() } else { b.clone() }])
            .collect::<Vec<_>>();
        let mut scores = vec![0.0; count];
        scores[0] = if variant == "retrieval-bad-score" {
            0.5
        } else {
            1.0
        };
        let mut ranking = (0..count).collect::<Vec<_>>();
        if variant == "retrieval-bad-ranking" {
            ranking.swap(0, 1);
        }
        json!({"type":"result","retrieval":{"schema_version":1,"method":"maxsim","query":[a],"documents":documents,"scores":scores,"ranking":ranking,"tie_break":"document_order"}})
    } else if variant.starts_with("d3") {
        json!({"type":"result","decision":{"schema_version":1,"candidates":[{"id":"a","probability":0.75},{"id":"b","probability":0.25}],"selected_id":"a","provenance":{"calibration":"native","selection_rule":"candidate_order","preprocessing":"fixture","input_digest":"a".repeat(64),"input_truncated":false,"native":{}}}})
    } else if variant.starts_with("indra") {
        json!({"type":"result","output_text":"fixture OCR","output_metadata":if variant=="indra-bad-output-metadata" {json!([])} else if variant=="indra-oversized-output-metadata" {json!({"value":"x".repeat(65_536)})} else {json!({"schema_version":1,"stage_calls":{"det":1,"rec":2}})}})
    } else {
        json!({"type":"result","output_text":"complete native transcript ".repeat(50),"audio_duration_seconds":if variant=="speech-bad-duration" {-1.0} else {1.25},"speech_routing":{"language":"en-US"},"output_tokens":999,"first_token_latency_ns":1000})
    };
    format!("{value}\n")
}
