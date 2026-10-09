use std::future::IntoFuture;

use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::post;
use axum::Json;
use axum::Router;
use qit_runtime::config::Config;
use qit_runtime::model::{ProviderKind, RunRequest, RunStatus};
use qit_runtime::packs::PackCatalog;
use qit_runtime::paths::Paths;
use qit_runtime::runner::run_benchmark;
use qit_runtime::store::Store;
use tempfile::TempDir;
use tokio::net::TcpListener;

#[tokio::test]
async fn dashboard_exposes_benchmarks_history_and_embedded_assets() {
    let temporary = TempDir::new().unwrap();
    let listening = qit_runtime::bind(Config {
        home: temporary.path().join("state"),
        listen: "127.0.0.1:0".parse().unwrap(),
    })
    .await
    .unwrap();

    let health: serde_json::Value = reqwest::get(format!("{}/api/health", listening.base_url()))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(health["ok"], true);

    let benchmarks: serde_json::Value =
        reqwest::get(format!("{}/api/benchmarks", listening.base_url()))
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
    assert_eq!(benchmarks.as_array().unwrap().len(), 6);
    assert!(benchmarks
        .as_array()
        .unwrap()
        .iter()
        .any(|benchmark| benchmark["task"] == "image_to_text"));

    let history: serde_json::Value = reqwest::get(format!("{}/api/runs", listening.base_url()))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(history["runs"].as_array().unwrap().len(), 0);

    let html = reqwest::get(listening.base_url())
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    let asset = html
        .split("src=\"")
        .nth(1)
        .and_then(|tail| tail.split('"').next())
        .unwrap();
    let javascript = reqwest::get(format!("{}{}", listening.base_url(), asset))
        .await
        .unwrap();
    assert!(javascript.status().is_success());
    assert!(javascript.bytes().await.unwrap().len() > 100_000);

    listening.shutdown().await;
}

#[tokio::test]
async fn ollama_benchmark_persists_successful_samples_and_metrics() {
    let server = ollama_server(false).await;
    let temporary = TempDir::new().unwrap();
    let (store, catalog) = workspace(&temporary);
    let run = run_benchmark(
        &store,
        &catalog,
        RunRequest {
            benchmark_id: "core/text-generation-smoke".into(),
            provider: ProviderKind::Ollama,
            base_url: server.base_url.clone(),
            api_key: None,
            model: "fixture-model".into(),
            iterations: 2,
            warmups: 1,
            max_output_tokens: None,
            timeout_seconds: 5,
            target_pid: None,
        },
    )
    .await
    .unwrap();

    assert_eq!(run.status, RunStatus::Succeeded);
    assert_eq!(run.summary.sample_count, 6);
    assert_eq!(run.summary.successful_sample_count, 6);
    assert_eq!(run.summary.tokens_per_second_p50, Some(20.0));
    assert!(run.summary.ttft_ms_p50.is_some());
    assert_eq!(store.runs(10, true).unwrap().len(), 1);
    assert_eq!(store.samples(&run.id).unwrap().len(), 6);

    server.stop().await;
}

#[tokio::test]
async fn provider_failure_is_retained_but_excluded_from_successful_runs() {
    let server = ollama_server(true).await;
    let temporary = TempDir::new().unwrap();
    let (store, catalog) = workspace(&temporary);
    let run = run_benchmark(
        &store,
        &catalog,
        RunRequest {
            benchmark_id: "core/text-generation-smoke".into(),
            provider: ProviderKind::Ollama,
            base_url: server.base_url.clone(),
            api_key: None,
            model: "broken-model".into(),
            iterations: 1,
            warmups: 0,
            max_output_tokens: None,
            timeout_seconds: 5,
            target_pid: None,
        },
    )
    .await
    .unwrap();

    assert_eq!(run.status, RunStatus::Failed);
    assert_eq!(run.summary.sample_count, 3);
    assert_eq!(run.summary.successful_sample_count, 0);
    assert_eq!(store.runs(10, false).unwrap().len(), 1);
    assert!(store.runs(10, true).unwrap().is_empty());

    server.stop().await;
}

#[test]
fn installs_and_removes_a_declarative_benchmark_pack() {
    let temporary = TempDir::new().unwrap();
    let source = temporary.path().join("source");
    std::fs::create_dir_all(&source).unwrap();
    std::fs::write(
        source.join("pack.json"),
        r#"{
          "schema_version": 1,
          "id": "custom",
          "name": "Custom pack",
          "version": "1.0.0",
          "description": "A local benchmark pack.",
          "publisher": "test",
          "license": "MIT",
          "source": "local fixture",
          "benchmarks": [{
            "id": "hello",
            "name": "Hello",
            "description": "A small text benchmark.",
            "task": "text_generation",
            "cases": "cases.jsonl"
          }]
        }"#,
    )
    .unwrap();
    std::fs::write(
        source.join("cases.jsonl"),
        r#"{"id":"hello","prompt":"Say hello."}"#,
    )
    .unwrap();
    let catalog = PackCatalog::new(temporary.path().join("installed"));

    assert_eq!(catalog.install(&source).unwrap(), "custom");
    assert!(catalog.benchmark("custom/hello").is_ok());
    assert!(catalog.install(&source).is_err());
    catalog.remove("custom").unwrap();
    assert!(catalog.benchmark("custom/hello").is_err());
}

fn workspace(temporary: &TempDir) -> (Store, PackCatalog) {
    let paths = Paths::new(temporary.path().join("state"));
    paths.ensure().unwrap();
    (
        Store::open(&paths.database).unwrap(),
        PackCatalog::new(paths.packs),
    )
}

struct TestServer {
    base_url: String,
    join: tokio::task::JoinHandle<()>,
}

impl TestServer {
    async fn stop(self) {
        self.join.abort();
        let _ = self.join.await;
    }
}

async fn ollama_server(fail: bool) -> TestServer {
    async fn show() -> Json<serde_json::Value> {
        Json(serde_json::json!({ "capabilities": ["completion"] }))
    }
    async fn success() -> impl IntoResponse {
        (
            [("content-type", "application/x-ndjson")],
            concat!(
                "{\"response\":\"raspberry 42 edge\",\"done\":false}\n",
                "{\"response\":\"\",\"done\":true,\"prompt_eval_count\":5,",
                "\"eval_count\":4,\"eval_duration\":200000000}\n"
            ),
        )
    }
    async fn failure() -> impl IntoResponse {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "fixture provider unavailable",
        )
    }
    let app = if fail {
        Router::new()
            .route("/api/show", post(show))
            .route("/api/generate", post(failure))
    } else {
        Router::new()
            .route("/api/show", post(show))
            .route("/api/generate", post(success))
    };
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let join = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    TestServer {
        base_url: format!("http://{address}"),
        join,
    }
}

#[tokio::test]
async fn dashboard_starts_runs_without_blocking_history_and_retains_failures() {
    use axum::extract::State;
    use axum::http::HeaderMap;
    use std::sync::Arc;
    use tokio::sync::Semaphore;

    async fn show(
        State(gate): State<Arc<Semaphore>>,
        headers: HeaderMap,
    ) -> Json<serde_json::Value> {
        assert_eq!(
            headers.get("authorization").unwrap(),
            "Bearer fixture-secret"
        );
        let permit = gate.acquire().await.unwrap();
        permit.forget();
        Json(serde_json::json!({ "capabilities": ["completion"] }))
    }
    async fn generate(Json(body): Json<serde_json::Value>) -> axum::response::Response {
        if body["model"] == "broken-model" {
            return (StatusCode::SERVICE_UNAVAILABLE, "fixture model unavailable").into_response();
        }
        (
            [("content-type", "application/x-ndjson")],
            concat!(
                "{\"response\":\"raspberry 42 edge\",\"done\":false}\n",
                "{\"response\":\"\",\"done\":true,\"prompt_eval_count\":5,",
                "\"eval_count\":4,\"eval_duration\":200000000}\n"
            ),
        )
            .into_response()
    }
    let gate = Arc::new(Semaphore::new(0));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let provider_url = format!("http://{}", listener.local_addr().unwrap());
    let provider = tokio::spawn(
        axum::serve(
            listener,
            Router::new()
                .route("/api/show", post(show))
                .route("/api/generate", post(generate))
                .with_state(gate.clone()),
        )
        .into_future(),
    );
    let temporary = TempDir::new().unwrap();
    let home = temporary.path().join("state");
    let dashboard = qit_runtime::bind(Config {
        home: home.clone(),
        listen: "127.0.0.1:0".parse().unwrap(),
    })
    .await
    .unwrap();
    let url = format!("{}/api/runs", dashboard.base_url());
    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "benchmark_id": "core/text-generation-smoke", "provider": "ollama",
        "model": "fixture-model", "base_url": provider_url,
        "api_key": "fixture-secret", "iterations": 1, "warmups": 0,
        "max_output_tokens": 12, "timeout_seconds": 5, "target_pid": null
    });
    let forbidden = client
        .post(&url)
        .header("origin", "https://other.example")
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(forbidden.status(), StatusCode::FORBIDDEN);
    for (field, value) in [
        ("iterations", serde_json::json!(0)),
        ("model", serde_json::json!(" ")),
        ("provider", serde_json::json!("hf_serve")),
        ("base_url", serde_json::json!("file:///tmp/model")),
        ("timeout_seconds", serde_json::json!(0)),
        ("benchmark_id", serde_json::json!("missing/benchmark")),
        ("target_pid", serde_json::json!(0)),
    ] {
        let mut invalid = body.clone();
        invalid[field] = value;
        let response = client.post(&url).json(&invalid).send().await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{field}");
    }
    let initial: serde_json::Value = client.get(&url).send().await.unwrap().json().await.unwrap();
    assert!(initial["runs"].as_array().unwrap().is_empty());
    let accepted = client
        .post(&url)
        .header("origin", dashboard.base_url())
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(accepted.status(), StatusCode::ACCEPTED);
    let accepted: serde_json::Value = accepted.json().await.unwrap();
    let id = accepted["id"].as_str().unwrap();
    assert_eq!(accepted["status"], "running");
    assert!(!accepted.to_string().contains("fixture-secret"));
    let history: serde_json::Value =
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            client.get(&url).send().await.unwrap().json().await.unwrap()
        })
        .await
        .unwrap();
    assert_eq!(history["running"], 1);
    let observer = Store::open(&home.join("results.db")).unwrap();
    assert_eq!(
        observer.runs(10, false).unwrap()[0].status,
        RunStatus::Running
    );
    let conflict = client.post(&url).json(&body).send().await.unwrap();
    assert_eq!(conflict.status(), StatusCode::CONFLICT);
    gate.add_permits(1);
    let completed = await_run(&client, &dashboard.base_url(), id).await;
    assert_eq!(completed["run"]["status"], "succeeded");
    assert_eq!(completed["samples"].as_array().unwrap().len(), 3);
    assert_eq!(completed["run"]["summary"]["tokens_per_second_p50"], 20.0);
    assert_eq!(completed["run"]["summary"]["quality_score_mean"], 1.0);
    assert!(!completed.to_string().contains("fixture-secret"));
    gate.add_permits(1);
    let mut broken = body.clone();
    broken["model"] = serde_json::json!("broken-model");
    let failed: serde_json::Value = client
        .post(&url)
        .json(&broken)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let failed = await_run(
        &client,
        &dashboard.base_url(),
        failed["id"].as_str().unwrap(),
    )
    .await;
    assert_eq!(failed["run"]["status"], "failed");
    assert_eq!(failed["samples"].as_array().unwrap().len(), 3);
    let analysis: serde_json::Value = client
        .get(format!("{}/api/analysis", dashboard.base_url()))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(analysis["points"].as_array().unwrap().len(), 1);
    assert_eq!(observer.runs(10, false).unwrap().len(), 2);
    dashboard.shutdown().await;
    provider.abort();
}

async fn await_run(client: &reqwest::Client, base_url: &str, id: &str) -> serde_json::Value {
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            let detail: serde_json::Value = client
                .get(format!("{base_url}/api/runs/{id}"))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            if detail["run"]["status"] != "running" {
                return detail;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap()
}

#[test]
fn store_recovers_abandoned_runs_without_failing_live_runs() {
    let temporary = TempDir::new().unwrap();
    let (store, catalog) = workspace(&temporary);
    let prepared = qit_runtime::runner::prepare_benchmark(
        &store,
        &catalog,
        RunRequest {
            benchmark_id: "core/text-generation-smoke".into(),
            provider: ProviderKind::Ollama,
            model: "fixture".into(),
            base_url: "http://127.0.0.1:11434".into(),
            api_key: None,
            iterations: 1,
            warmups: 0,
            max_output_tokens: None,
            timeout_seconds: 1,
            target_pid: None,
        },
    )
    .unwrap();
    let database = temporary.path().join("state/results.db");
    let live = Store::open(&database).unwrap();
    assert_eq!(live.runs(10, false).unwrap()[0].status, RunStatus::Running);
    rusqlite::Connection::open(&database)
        .unwrap()
        .execute(
            "UPDATE benchmark_runs SET owner_pid = 0 WHERE id = ?1",
            [&prepared.run.id],
        )
        .unwrap();
    let recovered = Store::open(&database).unwrap();
    let run = &recovered.runs(10, false).unwrap()[0];
    assert_eq!(run.status, RunStatus::Failed);
    assert!(run
        .error
        .as_ref()
        .unwrap()
        .contains("before the run completed"));
}
