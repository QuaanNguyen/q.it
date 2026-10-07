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
