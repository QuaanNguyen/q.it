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
async fn text_decision_preserves_candidate_scores_and_accuracy_through_the_staged_command() {
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
    write_pack(&pack, "text", None);
    let installed = command(&state)
        .args(["benchmark", "install"])
        .arg(&pack)
        .output()
        .await
        .unwrap();
    assert!(installed.status.success(), "{installed:?}");
    let run = command(&state)
        .args([
            "run",
            "decisions/choice",
            "--provider",
            "local-cpu",
            "--model",
            "good",
            "--base-url",
            &format!("{endpoint}/good"),
            "--warmups",
            "1",
            "--iterations",
            "1",
        ])
        .output()
        .await
        .unwrap();
    assert!(run.status.success(), "{run:?}");
    assert!(!String::from_utf8_lossy(&run.stdout).contains("TTFT"));
    assert!(!String::from_utf8_lossy(&run.stdout).contains("Tokens/s"));
    let rows = export(&state, "jsonl").await;
    let row: Value = serde_json::from_str(&rows).unwrap();
    assert_eq!(row["run"]["task"], "decision_scoring");
    assert_eq!(row["run"]["summary"]["quality_score_mean"], 1.0);
    assert_eq!(
        row["run"]["summary"]["decision_metrics"]["brier_score_mean"],
        0.125
    );
    let sample = &row["samples"][0];
    assert_eq!(sample["decision"]["result"]["selected_id"], "alpha");
    assert_eq!(sample["decision"]["case"]["expected"], "alpha");
    assert_eq!(sample["decision"]["result"]["candidates"][0]["id"], "zebra");
    assert_eq!(
        sample["decision"]["result"]["candidates"][1]["probability"],
        0.75
    );
    assert_eq!(sample["decision"]["metrics"]["brier_score"], 0.125);
    let loss = sample["decision"]["metrics"]["negative_log_likelihood"]["value"]
        .as_f64()
        .unwrap();
    assert!((loss - 0.2876820724517809).abs() < 1e-12);
    for key in [
        "ttft_ms",
        "input_tokens",
        "output_tokens",
        "tokens_per_second",
    ] {
        assert!(sample[key].is_null(), "{key}: {sample}");
    }
    let csv = export(&state, "csv").await;
    assert!(csv.lines().next().unwrap().contains("brier_score_mean"));
    assert!(csv.contains("0.125"));
    server.abort();
}

#[tokio::test]
async fn decision_image_pack_rejects_mislabeled_or_invalid_image_payloads() {
    let temporary = TempDir::new().unwrap();
    let pack = temporary.path().join("pack");
    write_pack(&pack, "image", Some("data:image/png;base64,aGVsbG8="));
    let result = command(&temporary.path().join("state"))
        .args(["benchmark", "verify"])
        .arg(&pack)
        .output()
        .await
        .unwrap();
    assert!(!result.status.success(), "{result:?}");
    assert!(
        String::from_utf8_lossy(&result.stderr).contains("decision image"),
        "{result:?}"
    );
}

async fn provider() -> (String, tokio::task::JoinHandle<Result<(), std::io::Error>>) {
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
            "decisions/choice",
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn valid_wrong_decisions_ties_and_infinite_loss_remain_successful_and_malformed_results_fail()
{
    let temporary = TempDir::new().unwrap();
    let state = temporary.path().join("state");
    let pack = temporary.path().join("pack");
    write_pack(&pack, "text", None);
    assert!(command(&state)
        .args(["benchmark", "install"])
        .arg(&pack)
        .output()
        .await
        .unwrap()
        .status
        .success());
    let (endpoint, server) = provider().await;
    for model in ["wrong", "zero", "tie"] {
        let output = run(&state, &endpoint, model).await;
        assert!(output.status.success(), "{model}: {output:?}");
    }
    let invalid = [
        "duplicate",
        "reordered",
        "missing",
        "negative",
        "sum",
        "null-probability",
        "truncated",
        "changed-calibration",
        "bad-digest",
        "bad-selection",
        "nonmax-selection",
        "missing-result",
        "no-terminal",
        "double-terminal",
        "wrong-dtype",
        "wrong-version",
    ];
    for model in invalid {
        let output = run(&state, &endpoint, model).await;
        assert_eq!(output.status.code(), Some(2), "{model}: {output:?}");
    }
    let rows = export(&state, "jsonl")
        .await
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 3);
    let zero = rows
        .iter()
        .find(|row| row["run"]["model"] == "zero")
        .unwrap();
    assert_eq!(zero["run"]["status"], "succeeded");
    assert_eq!(zero["run"]["summary"]["quality_score_mean"], 0.0);
    assert_eq!(
        zero["run"]["summary"]["decision_metrics"]["brier_score_mean"],
        2.0
    );
    assert_eq!(
        zero["run"]["summary"]["decision_metrics"]["negative_log_likelihood"],
        json!({"status":"infinite"})
    );
    assert_eq!(
        zero["run"]["summary"]["decision_metrics"]["zero_gold_probability_count"],
        1
    );
    assert_eq!(
        zero["samples"][0]["decision"]["metrics"]["negative_log_likelihood"],
        json!({"status":"infinite"})
    );
    let tie = rows
        .iter()
        .find(|row| row["run"]["model"] == "tie")
        .unwrap();
    assert_eq!(
        tie["samples"][0]["decision"]["result"]["selected_id"],
        "alpha"
    );
    assert_eq!(
        tie["samples"][0]["decision"]["result"]["provenance"]["selection_rule"],
        "lexical"
    );
    assert!(export(&state, "csv").await.contains("infinite,1"));
    let history = command(&state)
        .args(["results", "--limit", "100"])
        .output()
        .await
        .unwrap();
    assert_eq!(
        String::from_utf8_lossy(&history.stdout)
            .matches("failed")
            .count(),
        invalid.len()
    );
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn image_decisions_survive_pack_removal_and_are_exposed_by_the_dashboard() {
    use std::process::Stdio;
    use tokio::io::{AsyncBufReadExt, BufReader};
    let temporary = TempDir::new().unwrap();
    let state = temporary.path().join("state");
    let pack = temporary.path().join("pack");
    write_pack(&pack, "image", Some("image.png"));
    std::fs::write(
        pack.join("image.png"),
        include_bytes!("fixtures/decision.png"),
    )
    .unwrap();
    let installed = command(&state)
        .args(["benchmark", "install"])
        .arg(&pack)
        .output()
        .await
        .unwrap();
    assert!(installed.status.success(), "{installed:?}");
    let (endpoint, server) = provider().await;
    let output = run(&state, &endpoint, "good").await;
    assert!(output.status.success(), "{output:?}");
    let unsupported = run(&state, &endpoint, "text-only").await;
    assert_eq!(unsupported.status.code(), Some(2));
    assert!(command(&state)
        .args(["benchmark", "remove", "decisions"])
        .output()
        .await
        .unwrap()
        .status
        .success());
    let row: Value = serde_json::from_str(&export(&state, "jsonl").await).unwrap();
    assert!(row["samples"][0]["decision"]["media"]
        .as_str()
        .unwrap()
        .starts_with("data:image/png;base64,"));
    assert_eq!(
        row["samples"][0]["decision"]["case"]["candidates"][1]["text"],
        "Clear"
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
        detail["samples"][0]["decision"],
        row["samples"][0]["decision"]
    );
    dashboard.kill().await.unwrap();
    dashboard.wait().await.unwrap();
    server.abort();
}

#[tokio::test]
async fn decision_packs_reject_invalid_choice_shapes_and_mixed_media_groups() {
    let temporary = TempDir::new().unwrap();
    let state = temporary.path().join("state");
    let pack = temporary.path().join("pack");
    for mode in [
        "duplicate",
        "unknown-gold",
        "one-candidate",
        "empty-question",
        "multi-field",
        "media-in-text",
        "missing-image",
        "remote-image",
        "escaping-image",
    ] {
        write_pack(
            &pack,
            if mode.contains("image") {
                "image"
            } else {
                "text"
            },
            None,
        );
        let mut case: Value =
            serde_json::from_slice(&std::fs::read(pack.join("cases.jsonl")).unwrap()).unwrap();
        match mode {
            "duplicate" => case["decision"]["candidates"][1]["id"] = json!("zebra"),
            "unknown-gold" => case["decision"]["expected"] = json!("unknown"),
            "one-candidate" => {
                case["decision"]["candidates"].as_array_mut().unwrap().pop();
            }
            "empty-question" => case["decision"]["question"] = json!(" "),
            "multi-field" => case["decision"]["questions"] = json!(["first", "second"]),
            "media-in-text" => case["media"] = json!("data:image/png;base64,aGVsbG8="),
            "remote-image" => case["media"] = json!("https://example.invalid/image.png"),
            "escaping-image" => case["media"] = json!("../image.png"),
            _ => {}
        }
        std::fs::write(pack.join("cases.jsonl"), serde_json::to_vec(&case).unwrap()).unwrap();
        let result = command(&state)
            .args(["benchmark", "verify"])
            .arg(&pack)
            .output()
            .await
            .unwrap();
        assert!(!result.status.success(), "{mode}: {result:?}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unlabelled_decisions_keep_outputs_without_inventing_quality_and_token_limits_are_rejected()
{
    let temporary = TempDir::new().unwrap();
    let state = temporary.path().join("state");
    let pack = temporary.path().join("pack");
    write_pack(&pack, "text", None);
    let mut case: Value =
        serde_json::from_slice(&std::fs::read(pack.join("cases.jsonl")).unwrap()).unwrap();
    case["decision"].as_object_mut().unwrap().remove("expected");
    std::fs::write(pack.join("cases.jsonl"), serde_json::to_vec(&case).unwrap()).unwrap();
    assert!(command(&state)
        .args(["benchmark", "install"])
        .arg(&pack)
        .output()
        .await
        .unwrap()
        .status
        .success());
    let (endpoint, server) = provider().await;
    assert!(run(&state, &endpoint, "good").await.status.success());
    let row: Value = serde_json::from_str(&export(&state, "jsonl").await).unwrap();
    assert!(row["run"]["summary"]["quality_score_mean"].is_null());
    assert!(row["run"]["summary"]["decision_metrics"].is_null());
    assert!(row["samples"][0]["decision"]["metrics"].is_null());
    assert_eq!(
        row["samples"][0]["decision"]["result"]["selected_id"],
        "alpha"
    );
    let limited = command(&state)
        .args([
            "run",
            "decisions/choice",
            "--provider",
            "local-cpu",
            "--model",
            "good",
            "--max-tokens",
            "16",
        ])
        .output()
        .await
        .unwrap();
    assert!(!limited.status.success());
    assert!(String::from_utf8_lossy(&limited.stderr)
        .contains("does not accept generation token limits"));
    server.abort();
}

fn command(state: &Path) -> Command {
    let directory = state.parent().unwrap();
    let binary = directory.join("qit");
    if !binary.is_file() {
        std::fs::copy(
            std::env::var_os("QIT_TEST_BINARY").unwrap_or_else(|| env!("CARGO_BIN_EXE_qit").into()),
            &binary,
        )
        .unwrap();
    }
    let mut command = Command::new(binary);
    command
        .current_dir(directory)
        .kill_on_drop(true)
        .env("QIT_HOME", state)
        .env_remove("QIT_API_KEY");
    command
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

fn write_pack(directory: &Path, input: &str, media: Option<&str>) {
    std::fs::create_dir_all(directory).unwrap();
    std::fs::write(directory.join("pack.json"), serde_json::to_vec(&json!({
        "schema_version":1,"id":"decisions","name":"Decisions","version":"1.0.0",
        "description":"Single-choice decisions","publisher":"fixture","license":"CC0-1.0","source":"local fixture",
        "benchmarks":[{"id":"choice","name":"Choice","description":"One decision","task":"decision_scoring","decision_input":input,"cases":"cases.jsonl"}]
    })).unwrap()).unwrap();
    std::fs::write(directory.join("cases.jsonl"), serde_json::to_vec(&json!({
        "id":"one", "media":media,
        "decision":{"state":"A clear blue sky", "question":"Which description fits?", "candidates":[{"id":"zebra","text":"Rain"},{"id":"alpha","text":"Clear"}],"expected":"alpha"}
    })).unwrap()).unwrap();
}

async fn metadata(RoutePath(model): RoutePath<String>) -> Json<Value> {
    let mut value = json!({"schema_version":1,"version":"fixture","model":model,
        "configuration":{"device":"cpu","dtype":"float32","task":"decision_scoring","parameter_count":10,"revision":"fixture",
        "decision":{"schema_version":1,"inputs":["text","image"],"calibration":"native","selection_rule":"candidate_order","preprocessing":"fixture-v1"}}});
    if model == "tie" {
        value["configuration"]["decision"]["selection_rule"] = json!("lexical");
    }
    if model == "wrong-dtype" {
        value["configuration"]["dtype"] = json!("float16");
    }
    if model == "text-only" {
        value["configuration"]["decision"]["inputs"] = json!(["text"]);
    }
    if model == "wrong-version" {
        value["configuration"]["decision"]["schema_version"] = json!(2);
    }
    Json(value)
}

async fn invoke(RoutePath(model): RoutePath<String>, Json(body): Json<Value>) -> String {
    assert_eq!(body["task"], "decision_scoring");
    assert!(body.get("max_output_tokens").is_none());
    assert!(body["case"]["decision"].get("expected").is_none());
    assert!(body["case"].get("expected_index").is_none());
    if let Some(media) = body["case"].get("media") {
        assert!(media
            .as_str()
            .unwrap()
            .starts_with("data:image/png;base64,"));
    }
    let mut value = json!({"type":"result","output_tokens":999,"first_token_latency_ns":1000,
        "decision":{"schema_version":1,"candidates":[{"id":"zebra","probability":0.25},{"id":"alpha","probability":0.75}],"selected_id":"alpha",
        "provenance":{"calibration":"native","selection_rule":"candidate_order","preprocessing":"fixture-v1","input_digest":"a".repeat(64),"input_truncated":false,"native":{"fixture":true}}}});
    match model.as_str() {
        "wrong" => {
            value["decision"]["candidates"][0]["probability"] = json!(0.75);
            value["decision"]["candidates"][1]["probability"] = json!(0.25);
            value["decision"]["selected_id"] = json!("zebra");
        }
        "zero" => {
            value["decision"]["candidates"][0]["probability"] = json!(1.0);
            value["decision"]["candidates"][1]["probability"] = json!(0.0);
            value["decision"]["selected_id"] = json!("zebra");
        }
        "tie" => {
            value["decision"]["candidates"][0]["probability"] = json!(0.5);
            value["decision"]["candidates"][1]["probability"] = json!(0.5);
            value["decision"]["provenance"]["selection_rule"] = json!("lexical");
        }
        "duplicate" => value["decision"]["candidates"][1]["id"] = json!("zebra"),
        "reordered" => value["decision"]["candidates"]
            .as_array_mut()
            .unwrap()
            .swap(0, 1),
        "missing" => {
            value["decision"]["candidates"]
                .as_array_mut()
                .unwrap()
                .pop();
        }
        "negative" => {
            value["decision"]["candidates"][0]["probability"] = json!(-0.1);
            value["decision"]["candidates"][1]["probability"] = json!(1.1);
        }
        "sum" => value["decision"]["candidates"][1]["probability"] = json!(0.3),
        "null-probability" => value["decision"]["candidates"][1]["probability"] = Value::Null,
        "truncated" => value["decision"]["provenance"]["input_truncated"] = json!(true),
        "changed-calibration" => value["decision"]["provenance"]["calibration"] = json!("other"),
        "bad-digest" => value["decision"]["provenance"]["input_digest"] = json!("not-a-digest"),
        "bad-selection" => value["decision"]["selected_id"] = json!("unknown"),
        "nonmax-selection" => value["decision"]["selected_id"] = json!("zebra"),
        "missing-result" => {
            value.as_object_mut().unwrap().remove("decision");
        }
        "no-terminal" => return String::new(),
        "double-terminal" => return format!("{value}\n{value}\n"),
        _ => {}
    }
    format!("{value}\n")
}
