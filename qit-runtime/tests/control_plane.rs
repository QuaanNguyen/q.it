use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use qit_runtime::bind;
use qit_runtime::config::Config;
use qit_runtime::gguf::{write_test_gguf, GgufMeta};
use qit_runtime::probe::{FixedProbe, HardwareSnapshot};
use qit_runtime::supervisor::{LlamaServerLauncher, StubBinLauncher};
use serde_json::Value;
use tempfile::TempDir;

struct Harness {
    _tmp: TempDir,
    home: PathBuf,
    models: PathBuf,
    listening: qit_runtime::Listening,
}

impl Harness {
    async fn start(probe: HardwareSnapshot, extra_args: Vec<String>) -> Self {
        let launcher = Arc::new(StubBinLauncher {
            binary: PathBuf::from(env!("CARGO_BIN_EXE_qit-stub-worker")),
            extra_args,
        });
        Self::start_with(probe, launcher, None).await
    }

    async fn start_with(
        probe: HardwareSnapshot,
        launcher: Arc<dyn qit_runtime::supervisor::WorkerLauncher>,
        worker_path: Option<PathBuf>,
    ) -> Self {
        let tmp = TempDir::new().unwrap();
        let home = tmp.path().join("home");
        let models = tmp.path().join("models");
        std::fs::create_dir_all(&models).unwrap();
        let cfg = Config::test(
            home.clone(),
            models.clone(),
            "127.0.0.1:0".parse().unwrap(),
            FixedProbe {
                snapshot: probe.clone(),
            },
            worker_path,
            launcher,
            Some(2_000_000),
        );
        let listening = bind(cfg).await.unwrap();
        Self {
            _tmp: tmp,
            home,
            models,
            listening,
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.listening.base_url())
    }

    async fn get(&self, path: &str) -> reqwest::Response {
        let client = reqwest::Client::new();
        let url = self.url(path);
        let mut last = None;
        for _ in 0..50 {
            match client.get(&url).send().await {
                Ok(resp) => return resp,
                Err(e) => last = Some(e),
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("get {url}: {last:?}");
    }

    async fn json(&self, path: &str) -> Value {
        self.get(path).await.json().await.unwrap()
    }

    async fn post_json(&self, path: &str, body: Value) -> reqwest::Response {
        reqwest::Client::new()
            .post(self.url(path))
            .json(&body)
            .send()
            .await
            .unwrap()
    }

    async fn delete(&self, path: &str) -> reqwest::Response {
        reqwest::Client::new()
            .delete(self.url(path))
            .send()
            .await
            .unwrap()
    }
}

fn probe_with_free(free: Option<u64>) -> HardwareSnapshot {
    HardwareSnapshot {
        device_class: "apple_silicon".into(),
        chip: "test-chip".into(),
        unified_memory_bytes: 10_000_000,
        metal_recommended_working_set_bytes: Some(5_000_000),
        memory_pressure: None,
        free_ram_bytes: free,
    }
}

fn write_artifact(dir: &PathBuf, org: &str, name: &str, bytes: usize, meta: GgufMeta) -> PathBuf {
    let org_dir = dir.join(org);
    std::fs::create_dir_all(&org_dir).unwrap();
    let path = org_dir.join(name);
    write_test_gguf(&path, &meta).unwrap();
    if bytes > std::fs::metadata(&path).unwrap().len() as usize {
        let mut data = std::fs::read(&path).unwrap();
        data.resize(bytes, 0);
        std::fs::write(&path, data).unwrap();
    }
    path
}

fn llm_meta() -> GgufMeta {
    GgufMeta {
        architecture: Some("llama".into()),
        context_length: Some(32768),
        block_count: Some(1),
        embedding_length: Some(256),
        head_count: Some(4),
        head_count_kv: Some(4),
    }
}

#[tokio::test]
async fn health_and_shell_pages() {
    let h = Harness::start(probe_with_free(Some(100)), vec![]).await;
    let health: Value = h.json("/api/health").await;
    assert_eq!(health["ok"], true);
    let html = h.get("/").await.text().await.unwrap();
    assert!(html.contains("Catalog"));
    assert!(html.contains("Capacity"));
    assert!(html.contains("Settings"));
    let catalog = h.get("/#/catalog").await;
    assert_eq!(catalog.status(), 200);
    assert!(h.home.join("runtime.db").exists());
    h.listening.shutdown().await;
}

#[tokio::test]
async fn occupied_port_fails_without_hopping() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let tmp = TempDir::new().unwrap();
    let launcher = Arc::new(StubBinLauncher {
        binary: PathBuf::from(env!("CARGO_BIN_EXE_qit-stub-worker")),
        extra_args: vec![],
    });
    let cfg = Config::test(
        tmp.path().join("home"),
        tmp.path().join("models"),
        addr,
        FixedProbe {
            snapshot: probe_with_free(None),
        },
        None,
        launcher,
        Some(1),
    );
    let err = bind(cfg).await.err().unwrap();
    let msg = err.to_string();
    assert!(msg.contains(&addr.port().to_string()), "{msg}");
    drop(listener);
}

#[tokio::test]
async fn stable_budget_ignores_free_ram() {
    let h = Harness::start(probe_with_free(Some(99)), vec![]).await;
    let hw = h.json("/api/hardware").await;
    assert_eq!(hw["chip"], "test-chip");
    assert_eq!(hw["unified_memory_bytes"], 10_000_000);
    assert_eq!(hw["os_reserve_bytes"], 2_000_000);
    assert_eq!(hw["metal_recommended_working_set_bytes"], 5_000_000);
    assert_eq!(hw["budget_bytes"], 5_000_000);
    assert_eq!(hw["free_ram_bytes"], 99);
    h.listening.shutdown().await;
}

#[tokio::test]
async fn scan_registers_gguf_and_ignores_mlx() {
    let h = Harness::start(probe_with_free(None), vec![]).await;
    write_artifact(
        &h.models,
        "nvidia",
        "NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf",
        100_000,
        llm_meta(),
    );
    std::fs::create_dir_all(h.models.join("..").join("mlx").join("microsoft")).unwrap();
    std::fs::write(
        h.models
            .join("..")
            .join("mlx")
            .join("microsoft")
            .join("model.safetensors"),
        [0u8; 32],
    )
    .unwrap();
    std::fs::write(h.models.join("microsoft").join("notes.txt"), "nope").ok();
    let catalog = h
        .post_json("/api/scan", serde_json::json!({}))
        .await
        .json::<Value>()
        .await
        .unwrap();
    let ids: Vec<&str> = catalog["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, vec!["nvidia/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf"]);
    assert_eq!(catalog["artifacts"][0]["confidence"], "headers");
    h.listening.shutdown().await;
}

#[tokio::test]
async fn fit_badges_use_budget_and_context() {
    let h = Harness::start(probe_with_free(Some(1)), vec![]).await;
    write_artifact(&h.models, "org", "small.gguf", 100_000, llm_meta());
    write_artifact(&h.models, "org", "huge.gguf", 6_000_000, llm_meta());
    h.post_json("/api/scan", serde_json::json!({})).await;
    let catalog = h.json("/api/catalog?n_ctx=4096").await;
    let small = catalog["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["id"] == "org/small.gguf")
        .unwrap();
    let huge = catalog["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["id"] == "org/huge.gguf")
        .unwrap();
    assert_eq!(small["fit"], "Tight", "{small}");
    assert_eq!(huge["fit"], "No");
    let high_ctx = h.json("/api/catalog?n_ctx=32768").await;
    let small_hi = high_ctx["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["id"] == "org/small.gguf")
        .unwrap();
    assert_eq!(small_hi["fit"], "No");
    let hw_low_free = h.json("/api/hardware").await;
    assert_eq!(hw_low_free["budget_bytes"], 5_000_000);
    h.listening.shutdown().await;
}

#[tokio::test]
async fn what_if_and_pins_survive_rules() {
    let h = Harness::start(probe_with_free(None), vec![]).await;
    write_artifact(&h.models, "org", "small.gguf", 100_000, llm_meta());
    h.post_json("/api/scan", serde_json::json!({})).await;
    let what = h
        .post_json(
            "/api/what-ifs",
            serde_json::json!({"artifact_id":"org/small.gguf","n_ctx":32768}),
        )
        .await
        .json::<Value>()
        .await
        .unwrap();
    assert!(what["id"].as_str().unwrap().len() > 4);
    let cap = h.json("/api/capacity").await;
    assert_eq!(cap["what_ifs"].as_array().unwrap().len(), 1);
    let catalog = h.json("/api/catalog?n_ctx=4096").await;
    let small = catalog["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["id"] == "org/small.gguf")
        .unwrap();
    assert_eq!(small["fit"], "No");
    let pin = h
        .post_json(
            "/api/pins",
            serde_json::json!({"artifact_id":"org/small.gguf","n_ctx":32768}),
        )
        .await
        .json::<Value>()
        .await
        .unwrap();
    let pin_id = pin["id"].as_str().unwrap().to_string();
    h.delete("/api/what-ifs").await;
    let home = h.home.clone();
    let models = h.models.clone();
    h.listening.shutdown().await;

    let launcher = Arc::new(StubBinLauncher {
        binary: PathBuf::from(env!("CARGO_BIN_EXE_qit-stub-worker")),
        extra_args: vec![],
    });
    let cfg = Config::test(
        home,
        models,
        "127.0.0.1:0".parse().unwrap(),
        FixedProbe {
            snapshot: probe_with_free(None),
        },
        None,
        launcher,
        Some(2_000_000),
    );
    let listening = bind(cfg).await.unwrap();
    let client = reqwest::Client::new();
    let cap: Value = client
        .get(format!("{}/api/capacity", listening.base_url()))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(cap["what_ifs"].as_array().unwrap().len(), 0);
    assert_eq!(cap["pins"].as_array().unwrap().len(), 1);
    assert_eq!(cap["pins"][0]["id"], pin_id);
    assert_eq!(cap["sessions"].as_array().unwrap().len(), 0);
    listening.shutdown().await;
}

#[tokio::test]
async fn start_stop_and_crash_failed() {
    let h = Harness::start(probe_with_free(None), vec![]).await;
    write_artifact(&h.models, "org", "small.gguf", 100_000, llm_meta());
    h.post_json("/api/scan", serde_json::json!({})).await;
    let started = h
        .post_json(
            "/api/sessions",
            serde_json::json!({"artifact_id":"org/small.gguf","n_ctx":4096}),
        )
        .await
        .json::<Value>()
        .await
        .unwrap();
    assert_eq!(started["status"], "loaded");
    let id = started["id"].as_str().unwrap();
    let stopped = h
        .post_json(&format!("/api/sessions/{id}/stop"), serde_json::json!({}))
        .await
        .json::<Value>()
        .await
        .unwrap();
    assert_eq!(stopped["status"], "not_loaded");
    h.listening.shutdown().await;

    let crash = Harness::start(probe_with_free(None), vec!["--crash".into()]).await;
    write_artifact(&crash.models, "org", "small.gguf", 100_000, llm_meta());
    crash.post_json("/api/scan", serde_json::json!({})).await;
    let failed = crash
        .post_json(
            "/api/sessions",
            serde_json::json!({"artifact_id":"org/small.gguf"}),
        )
        .await;
    assert!(failed.status().is_client_error() || failed.status().is_server_error());
    let sessions = crash.json("/api/sessions").await;
    let any_failed = sessions
        .as_array()
        .unwrap()
        .iter()
        .any(|s| s["status"] == "failed");
    assert!(any_failed, "{sessions}");
    crash.listening.shutdown().await;
}

#[tokio::test]
async fn generate_streams_and_records_metrics() {
    let h = Harness::start(probe_with_free(None), vec![]).await;
    write_artifact(&h.models, "org", "small.gguf", 100_000, llm_meta());
    h.post_json("/api/scan", serde_json::json!({})).await;
    let client = reqwest::Client::new();
    let body = client
        .post(h.url("/api/generate"))
        .json(&serde_json::json!({
            "artifact_id": "org/small.gguf",
            "prompt": "hi",
            "n_ctx": 4096
        }))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(body.contains("event: token"), "{body}");
    assert!(body.contains("hello"), "{body}");
    assert!(body.contains("event: done"), "{body}");
    tokio::time::sleep(Duration::from_millis(200)).await;
    let catalog = h.json("/api/catalog").await;
    let small = catalog["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["id"] == "org/small.gguf")
        .unwrap();
    assert!(small["throughput_tps"].as_f64().is_some(), "{small}");
    let sessions = h.json("/api/sessions").await;
    let loaded = sessions
        .as_array()
        .unwrap()
        .iter()
        .any(|s| s["status"] == "loaded");
    assert!(!loaded, "{sessions}");
    h.listening.shutdown().await;
}

#[tokio::test]
async fn generate_reuses_loaded_session() {
    let h = Harness::start(probe_with_free(None), vec![]).await;
    write_artifact(&h.models, "org", "small.gguf", 100_000, llm_meta());
    h.post_json("/api/scan", serde_json::json!({})).await;
    let started = h
        .post_json(
            "/api/sessions",
            serde_json::json!({"artifact_id":"org/small.gguf","n_ctx":4096}),
        )
        .await
        .json::<Value>()
        .await
        .unwrap();
    let id = started["id"].as_str().unwrap().to_string();
    let client = reqwest::Client::new();
    let body = client
        .post(h.url("/api/generate"))
        .json(&serde_json::json!({
            "artifact_id": "org/small.gguf",
            "prompt": "hi",
            "n_ctx": 4096
        }))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(body.contains("event: done"), "{body}");
    let sessions = h.json("/api/sessions").await;
    let still = sessions
        .as_array()
        .unwrap()
        .iter()
        .any(|s| s["id"] == id && s["status"] == "loaded");
    assert!(still, "{sessions}");
    h.listening.shutdown().await;
}

#[tokio::test]
async fn hardware_reports_worker_path() {
    let launcher = Arc::new(StubBinLauncher {
        binary: PathBuf::from(env!("CARGO_BIN_EXE_qit-stub-worker")),
        extra_args: vec![],
    });
    let worker = PathBuf::from("/opt/test/llama-server");
    let h = Harness::start_with(probe_with_free(None), launcher, Some(worker.clone())).await;
    let hw = h.json("/api/hardware").await;
    assert_eq!(hw["worker_path"].as_str().unwrap(), worker.to_string_lossy());
    h.listening.shutdown().await;
}

#[tokio::test]
async fn start_without_worker_binary_returns_install_message() {
    let h = Harness::start_with(
        probe_with_free(None),
        Arc::new(LlamaServerLauncher { binary: None }),
        None,
    )
    .await;
    write_artifact(&h.models, "org", "small.gguf", 100_000, llm_meta());
    h.post_json("/api/scan", serde_json::json!({})).await;
    let resp = h
        .post_json(
            "/api/sessions",
            serde_json::json!({"artifact_id":"org/small.gguf","n_ctx":4096}),
        )
        .await;
    assert_eq!(resp.status(), 400);
    let body: Value = resp.json().await.unwrap();
    let err = body["error"].as_str().unwrap();
    assert!(err.contains("QIT_WORKER_PATH") || err.contains("llama-server"), "{err}");
    h.listening.shutdown().await;
}

#[tokio::test]
async fn worker_waits_for_health_200_before_loaded() {
    let h = Harness::start(probe_with_free(None), vec!["--health-warmup-ms".into(), "400".into()])
        .await;
    write_artifact(&h.models, "org", "small.gguf", 100_000, llm_meta());
    h.post_json("/api/scan", serde_json::json!({})).await;
    let started = h
        .post_json(
            "/api/sessions",
            serde_json::json!({"artifact_id":"org/small.gguf","n_ctx":4096}),
        )
        .await
        .json::<Value>()
        .await
        .unwrap();
    assert_eq!(started["status"], "loaded");
    h.listening.shutdown().await;
}

#[tokio::test]
async fn duplicate_start_reuses_session_row() {
    let h = Harness::start(probe_with_free(None), vec![]).await;
    write_artifact(&h.models, "org", "small.gguf", 100_000, llm_meta());
    h.post_json("/api/scan", serde_json::json!({})).await;
    let first = h
        .post_json(
            "/api/sessions",
            serde_json::json!({"artifact_id":"org/small.gguf","n_ctx":4096}),
        )
        .await
        .json::<Value>()
        .await
        .unwrap();
    h.post_json(&format!("/api/sessions/{}/stop", first["id"]), serde_json::json!({}))
        .await;
    let second = h
        .post_json(
            "/api/sessions",
            serde_json::json!({"artifact_id":"org/small.gguf","n_ctx":4096}),
        )
        .await
        .json::<Value>()
        .await
        .unwrap();
    assert_eq!(first["id"], second["id"]);
    let sessions = h.json("/api/sessions").await;
    let matches: Vec<_> = sessions
        .as_array()
        .unwrap()
        .iter()
        .filter(|s| s["artifact_id"] == "org/small.gguf" && s["n_ctx"] == 4096)
        .collect();
    assert_eq!(matches.len(), 1, "{sessions}");
    h.listening.shutdown().await;
}

#[tokio::test]
async fn sessions_survive_restart_as_not_loaded() {
    let h = Harness::start(probe_with_free(None), vec![]).await;
    write_artifact(&h.models, "org", "small.gguf", 100_000, llm_meta());
    h.post_json("/api/scan", serde_json::json!({})).await;
    let started = h
        .post_json(
            "/api/sessions",
            serde_json::json!({"artifact_id":"org/small.gguf","n_ctx":4096}),
        )
        .await
        .json::<Value>()
        .await
        .unwrap();
    let id = started["id"].as_str().unwrap().to_string();
    let home = h.home.clone();
    let models = h.models.clone();
    h.listening.shutdown().await;

    let launcher = Arc::new(StubBinLauncher {
        binary: PathBuf::from(env!("CARGO_BIN_EXE_qit-stub-worker")),
        extra_args: vec![],
    });
    let cfg = Config::test(
        home,
        models,
        "127.0.0.1:0".parse().unwrap(),
        FixedProbe {
            snapshot: probe_with_free(None),
        },
        None,
        launcher,
        Some(2_000_000),
    );
    let listening = bind(cfg).await.unwrap();
    let client = reqwest::Client::new();
    let sessions: Value = client
        .get(format!("{}/api/sessions", listening.base_url()))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let row = sessions
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["id"] == id)
        .expect("session row survives restart");
    assert_eq!(row["status"], "not_loaded");
    listening.shutdown().await;
}

#[tokio::test]
async fn crash_populates_last_error() {
    let h = Harness::start(probe_with_free(None), vec!["--crash".into()]).await;
    write_artifact(&h.models, "org", "small.gguf", 100_000, llm_meta());
    h.post_json("/api/scan", serde_json::json!({})).await;
    let resp = h
        .post_json(
            "/api/sessions",
            serde_json::json!({"artifact_id":"org/small.gguf"}),
        )
        .await;
    assert!(resp.status().is_client_error() || resp.status().is_server_error());
    let sessions = h.json("/api/sessions").await;
    let failed = sessions
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["status"] == "failed")
        .expect("failed session row");
    assert!(failed["last_error"].as_str().is_some(), "{failed}");
    h.listening.shutdown().await;
}

#[tokio::test]
async fn delete_inactive_session_row() {
    let h = Harness::start(probe_with_free(None), vec!["--crash".into()]).await;
    write_artifact(&h.models, "org", "small.gguf", 100_000, llm_meta());
    h.post_json("/api/scan", serde_json::json!({})).await;
    h.post_json(
        "/api/sessions",
        serde_json::json!({"artifact_id":"org/small.gguf"}),
    )
    .await;
    let sessions = h.json("/api/sessions").await;
    let id = sessions.as_array().unwrap()[0]["id"]
        .as_str()
        .unwrap()
        .to_string();
    let del = h.delete(&format!("/api/sessions/{id}")).await;
    assert_eq!(del.status(), 204);
    let after = h.json("/api/sessions").await;
    assert_eq!(after.as_array().unwrap().len(), 0);
    h.listening.shutdown().await;
}

fn small_ctx_meta() -> GgufMeta {
    GgufMeta {
        architecture: Some("llama".into()),
        context_length: Some(8192),
        block_count: Some(1),
        embedding_length: Some(256),
        head_count: Some(4),
        head_count_kv: Some(4),
    }
}

#[tokio::test]
async fn session_rejects_n_ctx_above_model_max() {
    let h = Harness::start(probe_with_free(None), vec![]).await;
    write_artifact(
        &h.models,
        "org",
        "small-ctx.gguf",
        100_000,
        small_ctx_meta(),
    );
    h.post_json("/api/scan", serde_json::json!({})).await;
    let resp = h
        .post_json(
            "/api/sessions",
            serde_json::json!({"artifact_id":"org/small-ctx.gguf","n_ctx":16384}),
        )
        .await;
    assert_eq!(resp.status(), 400);
    let body: Value = resp.json().await.unwrap();
    assert!(body["error"].as_str().unwrap().contains("8192"), "{body}");
    h.listening.shutdown().await;
}
