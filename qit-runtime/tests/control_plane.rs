use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use qit_runtime::bind;
use qit_runtime::config::Config;
use qit_runtime::gguf::{write_test_gguf, GgufMeta};
use qit_runtime::probe::{FixedProbe, HardwareSnapshot};
use qit_runtime::supervisor::{LlamaServerLauncher, RecipeWorkerLauncher, StubBinLauncher};
use serde_json::Value;
use tempfile::TempDir;

#[tokio::test]
async fn package_target_migration_removes_fabricated_artifact_identity() {
    let tmp = TempDir::new().unwrap();
    let home = tmp.path().join("home");
    let models = tmp.path().join("models");
    std::fs::create_dir_all(&home).unwrap();
    let db_path = home.join("runtime.db");
    let conn = rusqlite::Connection::open(&db_path).unwrap();
    conn.execute_batch(
        r#"
        CREATE TABLE pins (
            id TEXT PRIMARY KEY,
            target_id TEXT NOT NULL,
            artifact_id TEXT NOT NULL,
            package_id TEXT,
            runtime_recipe TEXT NOT NULL,
            serve_profile_json TEXT NOT NULL,
            UNIQUE(target_id, serve_profile_json)
        );
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            target_id TEXT NOT NULL,
            artifact_id TEXT NOT NULL,
            package_id TEXT,
            runtime_recipe TEXT NOT NULL,
            serve_profile_json TEXT NOT NULL,
            status TEXT NOT NULL,
            last_error TEXT,
            log_path TEXT,
            UNIQUE(target_id, serve_profile_json)
        );
        INSERT INTO pins VALUES (
            'pin', 'Qwen/Qwen2.5-0.5B-Instruct', 'Qwen/Qwen2.5-0.5B-Instruct',
            'Qwen/Qwen2.5-0.5B-Instruct', 'transformers_external',
            '{"context_length":4096,"runtime_settings":{}}'
        );
        INSERT INTO sessions VALUES (
            'session', 'qit/qwen2.5-0.5b-instruct-q4_k_m',
            'Qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf',
            'qit/qwen2.5-0.5b-instruct-q4_k_m', 'llama_cpp',
            '{"context_length":4096,"runtime_settings":{"gpu_layers":7,"parallel":2}}',
            'not_loaded', NULL, NULL
        );
        "#,
    )
    .unwrap();
    drop(conn);

    let listening = bind(Config::test(
        home,
        models,
        "127.0.0.1:0".parse().unwrap(),
        FixedProbe {
            snapshot: probe_with_free(None),
        },
        None,
        stub_launcher(),
        Some(2_000_000),
    ))
    .await
    .unwrap();
    let capacity: Value = reqwest::get(format!("{}/api/capacity", listening.base_url()))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let pin = &capacity["pins"][0];
    assert_eq!(pin["target_id"], "Qwen/Qwen2.5-0.5B-Instruct");
    assert!(pin.get("artifact_id").is_none(), "{pin}");
    assert_eq!(pin["package_id"], "Qwen/Qwen2.5-0.5B-Instruct");
    let session = &capacity["sessions"][0];
    assert_eq!(session["target_id"], "qit/qwen2.5-0.5b-instruct-q4_k_m");
    assert_eq!(
        session["artifact_id"],
        "Qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    );
    assert_eq!(session["package_id"], "qit/qwen2.5-0.5b-instruct-q4_k_m");
    listening.shutdown().await;
}

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
        Self::start_full(probe, launcher, worker_path, Some(2_000_000)).await
    }

    async fn start_full(
        probe: HardwareSnapshot,
        launcher: Arc<dyn qit_runtime::supervisor::WorkerLauncher>,
        worker_path: Option<PathBuf>,
        os_reserve_env: Option<u64>,
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
            os_reserve_env,
        );
        let listening = bind(cfg).await.unwrap();
        Self {
            _tmp: tmp,
            home,
            models,
            listening,
        }
    }

    async fn start_with_transformers(probe: HardwareSnapshot, extra_args: Vec<String>) -> Self {
        let worker = PathBuf::from(env!("CARGO_BIN_EXE_qit-stub-worker"));
        let launcher = Arc::new(RecipeWorkerLauncher {
            llama_cpp_binary: None,
            transformers_external_binary: Some(worker.clone()),
            transformers_external_extra_args: extra_args,
        });
        Self::start_with_transformers_path(probe, worker, launcher).await
    }

    async fn start_with_transformers_path(
        probe: HardwareSnapshot,
        worker: PathBuf,
        launcher: Arc<dyn qit_runtime::supervisor::WorkerLauncher>,
    ) -> Self {
        let tmp = TempDir::new().unwrap();
        let home = tmp.path().join("home");
        let models = tmp.path().join("models");
        std::fs::create_dir_all(&models).unwrap();
        let cfg = Config::test(
            home.clone(),
            models.clone(),
            "127.0.0.1:0".parse().unwrap(),
            FixedProbe { snapshot: probe },
            None,
            launcher,
            Some(200_000_000),
        )
        .with_transformers_worker_path(worker);
        let listening = bind(cfg).await.unwrap();
        Self {
            _tmp: tmp,
            home,
            models,
            listening,
        }
    }

    async fn restart(mut self, os_reserve_env: Option<u64>) -> Self {
        let listening = std::mem::replace(
            &mut self.listening,
            bind_same_home(&self.home, &self.models, os_reserve_env).await,
        );
        listening.shutdown().await;
        self
    }

    async fn put_json(&self, path: &str, body: Value) -> reqwest::Response {
        reqwest::Client::new()
            .put(self.url(path))
            .json(&body)
            .send()
            .await
            .unwrap()
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

fn stub_launcher() -> Arc<StubBinLauncher> {
    Arc::new(StubBinLauncher {
        binary: PathBuf::from(env!("CARGO_BIN_EXE_qit-stub-worker")),
        extra_args: vec![],
    })
}

async fn bind_same_home(
    home: &Path,
    models: &Path,
    os_reserve_env: Option<u64>,
) -> qit_runtime::Listening {
    let cfg = Config::test(
        home.to_path_buf(),
        models.to_path_buf(),
        "127.0.0.1:0".parse().unwrap(),
        FixedProbe {
            snapshot: probe_with_free(None),
        },
        None,
        stub_launcher(),
        os_reserve_env,
    );
    bind(cfg).await.unwrap()
}

fn probe_with_free(free: Option<u64>) -> HardwareSnapshot {
    probe_live(free, None)
}

fn probe_live(free: Option<u64>, pressure: Option<&str>) -> HardwareSnapshot {
    HardwareSnapshot {
        device_class: "apple_silicon".into(),
        chip: "test-chip".into(),
        unified_memory_bytes: 10_000_000,
        metal_recommended_working_set_bytes: Some(5_000_000),
        memory_pressure: pressure.map(str::to_string),
        free_ram_bytes: free,
    }
}

fn write_artifact(dir: &Path, org: &str, name: &str, bytes: usize, meta: GgufMeta) -> PathBuf {
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
        has_chat_template: true,
        ..GgufMeta::default()
    }
}

fn write_transformers_package(models: &std::path::Path) -> PathBuf {
    let package = models
        .parent()
        .unwrap()
        .join("transformers")
        .join("Qwen")
        .join("Qwen2.5-0.5B-Instruct");
    std::fs::create_dir_all(&package).unwrap();
    for (name, contents) in [
        ("config.json", r#"{"model_type":"qwen2"}"#),
        ("tokenizer.json", r#"{"version":"1.0"}"#),
        ("tokenizer_config.json", r#"{"model_max_length":32768}"#),
        (
            "model.safetensors.index.json",
            r#"{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}"#,
        ),
        ("model-00001-of-00002.safetensors", "weights-1"),
        ("model-00002-of-00002.safetensors", "weights-2"),
        ("README.md", "# Qwen2.5 0.5B Instruct"),
    ] {
        std::fs::write(package.join(name), contents).unwrap();
    }
    package
}

fn write_local_transformers_package(models: &Path, org: &str, name: &str) -> PathBuf {
    let package = models
        .parent()
        .unwrap()
        .join("transformers")
        .join(org)
        .join(name);
    std::fs::create_dir_all(&package).unwrap();
    for (filename, contents) in [
        (
            "config.json",
            r#"{"model_type":"qwen2","architectures":["Qwen2ForCausalLM"]}"#,
        ),
        ("tokenizer.json", r#"{"version":"1.0"}"#),
        ("tokenizer_config.json", r#"{"model_max_length":32768}"#),
        ("model.safetensors", "weights"),
        ("README.md", "# Local chat package"),
    ] {
        std::fs::write(package.join(filename), contents).unwrap();
    }
    package
}

fn write_sentencepiece_transformers_package(models: &Path) -> PathBuf {
    let package = models
        .parent()
        .unwrap()
        .join("transformers")
        .join("acme")
        .join("sentencepiece-chat");
    std::fs::create_dir_all(&package).unwrap();
    for (filename, contents) in [
        (
            "config.json",
            r#"{"architectures":["LlamaForCausalLM"],"model_type":"llama"}"#,
        ),
        ("tokenizer.model", "sentencepiece"),
        ("model.safetensors", "weights"),
        ("modelcard.md", "# SentencePiece chat package"),
    ] {
        std::fs::write(package.join(filename), contents).unwrap();
    }
    package
}

fn write_bpe_transformers_package(models: &Path) -> PathBuf {
    let package = models
        .parent()
        .unwrap()
        .join("transformers")
        .join("acme")
        .join("bpe-chat");
    std::fs::create_dir_all(&package).unwrap();
    for (filename, contents) in [
        (
            "config.json",
            r#"{"architectures":["MistralForCausalLM"],"model_type":"mistral"}"#,
        ),
        ("vocab.json", r#"{"hello":0}"#),
        ("merges.txt", "#version: 0.2"),
        ("model.safetensors", "weights"),
        ("README.md", "# BPE chat package"),
    ] {
        std::fs::write(package.join(filename), contents).unwrap();
    }
    package
}

fn write_config_estimated_transformers_package(models: &Path) -> PathBuf {
    let package = models
        .parent()
        .unwrap()
        .join("transformers")
        .join("acme")
        .join("config-estimated-chat");
    std::fs::create_dir_all(&package).unwrap();
    for (filename, contents) in [
        (
            "config.json",
            r#"{"architectures":["LlamaForCausalLM"],"model_type":"llama","hidden_size":64,"num_hidden_layers":2,"intermediate_size":256,"vocab_size":32000,"num_attention_heads":8,"num_key_value_heads":2}"#,
        ),
        ("tokenizer.model", "sentencepiece"),
        ("modelcard.md", "# Config estimated chat package"),
    ] {
        std::fs::write(package.join(filename), contents).unwrap();
    }
    package
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
async fn owned_package_is_visible_offline_with_missing_required_files() {
    let h = Harness::start(
        HardwareSnapshot {
            device_class: "apple_silicon".into(),
            chip: "test-chip".into(),
            unified_memory_bytes: 2_000_000_000,
            metal_recommended_working_set_bytes: Some(1_000_000_000),
            memory_pressure: None,
            free_ram_bytes: Some(1),
        },
        vec![],
    )
    .await;
    let catalog = h.json("/api/catalog").await;
    let packages = catalog["packages"].as_array().unwrap();
    let package = packages
        .iter()
        .find(|package| package["id"] == "qit/qwen2.5-0.5b-instruct-q4_k_m")
        .unwrap();
    assert_eq!(package["id"], "qit/qwen2.5-0.5b-instruct-q4_k_m");
    assert_eq!(package["family"], "Qwen 2.5");
    assert_eq!(package["name"], "Qwen2.5 0.5B Instruct Q4_K_M");
    assert_eq!(package["format"], "gguf");
    assert_eq!(package["fits"], true);
    assert_eq!(package["ready"], false);
    assert_eq!(package["readiness_reason"], "missing_required_files");
    assert!(package["estimate_bytes"].as_u64().unwrap() > 1);
    h.listening.shutdown().await;
}

#[tokio::test]
async fn complete_local_transformers_package_reports_runtime_missing() {
    let h = Harness::start(
        HardwareSnapshot {
            device_class: "apple_silicon".into(),
            chip: "test-chip".into(),
            unified_memory_bytes: 2_000_000_000,
            metal_recommended_working_set_bytes: Some(1_500_000_000),
            memory_pressure: None,
            free_ram_bytes: Some(1),
        },
        vec![],
    )
    .await;
    let package_dir = write_transformers_package(&h.models);
    std::fs::remove_file(package_dir.join("model-00002-of-00002.safetensors")).unwrap();
    let incomplete = h
        .post_json("/api/scan", serde_json::json!({}))
        .await
        .json::<Value>()
        .await
        .unwrap();
    let incomplete_package = incomplete["packages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|package| package["id"] == "Qwen/Qwen2.5-0.5B-Instruct")
        .unwrap();
    assert_eq!(
        incomplete_package["readiness_reason"],
        "missing_required_files"
    );
    std::fs::write(
        package_dir.join("model-00002-of-00002.safetensors"),
        "weights-2",
    )
    .unwrap();
    let catalog = h
        .post_json("/api/scan", serde_json::json!({}))
        .await
        .json::<Value>()
        .await
        .unwrap();
    let package = catalog["packages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|package| package["id"] == "Qwen/Qwen2.5-0.5B-Instruct")
        .unwrap();
    assert_eq!(package["id"], "Qwen/Qwen2.5-0.5B-Instruct");
    assert_eq!(package["family"], "Qwen 2.5");
    assert_eq!(package["format"], "transformers");
    assert_eq!(package["estimate_bytes"], 1_200_000_000_u64);
    assert_eq!(package["estimate_source"], "qit_catalog");
    assert_eq!(package["estimate_confidence"], "high");
    assert_eq!(package["runtime_recipe"], "transformers_external");
    assert_eq!(package["fits"], true);
    assert_eq!(package["ready"], false);
    assert_eq!(package["readiness_reason"], "runtime_missing");
    let pin = h
        .post_json(
            "/api/pins",
            serde_json::json!({"package_id": "Qwen/Qwen2.5-0.5B-Instruct"}),
        )
        .await;
    assert_eq!(pin.status(), 200);
    assert_eq!(
        pin.json::<Value>().await.unwrap()["estimate_bytes"],
        1_200_000_000_u64
    );
    assert_eq!(h.json("/api/hardware").await["headroom_bytes"], 300_000_000);
    h.listening.shutdown().await;
}

#[tokio::test]
async fn local_transformers_package_is_discovered_from_its_metadata() {
    let h = Harness::start(
        HardwareSnapshot {
            device_class: "apple_silicon".into(),
            chip: "test-chip".into(),
            unified_memory_bytes: 2_000_000_000,
            metal_recommended_working_set_bytes: Some(1_500_000_000),
            memory_pressure: None,
            free_ram_bytes: Some(1),
        },
        vec![],
    )
    .await;
    write_local_transformers_package(&h.models, "acme", "edge-chat");

    let catalog = h
        .post_json("/api/scan", serde_json::json!({}))
        .await
        .json::<Value>()
        .await
        .unwrap();
    let package = catalog["packages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|package| package["id"] == "acme/edge-chat")
        .expect("discovered local package");

    assert_eq!(package["format"], "transformers");
    assert_eq!(package["family"], "Qwen");
    assert_eq!(package["estimate_source"], "local_files");
    assert_eq!(package["estimate_confidence"], "medium");
    assert_eq!(
        package["capabilities"],
        serde_json::json!({
            "inputs": ["text"],
            "outputs": ["text"],
            "tasks": ["chat"]
        })
    );
    let weights = package["files"]
        .as_array()
        .unwrap()
        .iter()
        .find(|file| file["path"] == "model.safetensors")
        .expect("weights metadata");
    assert_eq!(weights["role"], "weights");
    assert_eq!(weights["bytes"], 7);
    assert_eq!(
        weights["sha256"],
        "9a129038d9a00aed0cf6a7ea059ca50a813449061ab87848cf1a13eafdf33b2c"
    );
    assert_eq!(weights["source"], "local_scan");
    assert_eq!(package["fits"], true);
    assert_eq!(package["ready"], false);
    assert_eq!(package["readiness_reason"], "runtime_missing");
    h.listening.shutdown().await;
}

#[tokio::test]
async fn local_transformers_package_supports_sentencepiece_tokenizers_and_model_cards() {
    let h = Harness::start(
        HardwareSnapshot {
            device_class: "apple_silicon".into(),
            chip: "test-chip".into(),
            unified_memory_bytes: 2_000_000_000,
            metal_recommended_working_set_bytes: Some(1_500_000_000),
            memory_pressure: None,
            free_ram_bytes: Some(1),
        },
        vec![],
    )
    .await;
    write_sentencepiece_transformers_package(&h.models);

    let catalog = h
        .post_json("/api/scan", serde_json::json!({}))
        .await
        .json::<Value>()
        .await
        .unwrap();
    let package = catalog["packages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|package| package["id"] == "acme/sentencepiece-chat")
        .expect("discovered sentencepiece package");

    assert_eq!(package["readiness_reason"], "runtime_missing");
    assert!(package["files"]
        .as_array()
        .unwrap()
        .iter()
        .any(|file| file["path"] == "tokenizer.model"));
    assert!(package["files"]
        .as_array()
        .unwrap()
        .iter()
        .any(|file| file["path"] == "modelcard.md"));
    h.listening.shutdown().await;
}

#[tokio::test]
async fn local_transformers_package_supports_bpe_tokenizers() {
    let h = Harness::start(
        HardwareSnapshot {
            device_class: "apple_silicon".into(),
            chip: "test-chip".into(),
            unified_memory_bytes: 2_000_000_000,
            metal_recommended_working_set_bytes: Some(1_500_000_000),
            memory_pressure: None,
            free_ram_bytes: Some(1),
        },
        vec![],
    )
    .await;
    write_bpe_transformers_package(&h.models);

    let catalog = h
        .post_json("/api/scan", serde_json::json!({}))
        .await
        .json::<Value>()
        .await
        .unwrap();
    let package = catalog["packages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|package| package["id"] == "acme/bpe-chat")
        .expect("discovered BPE package");

    assert_eq!(package["readiness_reason"], "runtime_missing");
    assert!(package["files"]
        .as_array()
        .unwrap()
        .iter()
        .any(|file| file["path"] == "vocab.json"));
    assert!(package["files"]
        .as_array()
        .unwrap()
        .iter()
        .any(|file| file["path"] == "merges.txt"));
    h.listening.shutdown().await;
}

#[tokio::test]
async fn local_transformers_package_uses_config_estimate_when_weights_are_missing() {
    let h = Harness::start(
        HardwareSnapshot {
            device_class: "apple_silicon".into(),
            chip: "test-chip".into(),
            unified_memory_bytes: 2_000_000_000,
            metal_recommended_working_set_bytes: Some(1_500_000_000),
            memory_pressure: None,
            free_ram_bytes: Some(1),
        },
        vec![],
    )
    .await;
    write_config_estimated_transformers_package(&h.models);

    let catalog = h
        .post_json("/api/scan", serde_json::json!({}))
        .await
        .json::<Value>()
        .await
        .unwrap();
    let package = catalog["packages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|package| package["id"] == "acme/config-estimated-chat")
        .expect("discovered config-estimated package");

    assert_eq!(package["estimate_source"], "config_architecture");
    assert_eq!(package["estimate_confidence"], "low");
    assert!(package["estimate_bytes"].as_u64().unwrap() > 0);
    assert_eq!(package["readiness_reason"], "missing_required_files");
    h.listening.shutdown().await;
}

#[tokio::test]
async fn nonexistent_transformers_worker_keeps_package_not_ready() {
    let h = Harness::start_with_transformers_path(
        HardwareSnapshot {
            device_class: "apple_silicon".into(),
            chip: "test-chip".into(),
            unified_memory_bytes: 2_000_000_000,
            metal_recommended_working_set_bytes: Some(1_500_000_000),
            memory_pressure: None,
            free_ram_bytes: None,
        },
        PathBuf::from("/missing/qit-transformers-worker"),
        stub_launcher(),
    )
    .await;
    write_transformers_package(&h.models);
    let catalog = h
        .post_json("/api/scan", serde_json::json!({}))
        .await
        .json::<Value>()
        .await
        .unwrap();
    let package = catalog["packages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|package| package["id"] == "Qwen/Qwen2.5-0.5B-Instruct")
        .unwrap();
    assert_eq!(package["ready"], false, "{package}");
    assert_eq!(package["readiness_reason"], "runtime_missing", "{package}");
    h.listening.shutdown().await;
}

#[tokio::test]
async fn changed_transformers_package_is_not_ready_after_scan() {
    let h = Harness::start_with_transformers(
        HardwareSnapshot {
            device_class: "apple_silicon".into(),
            chip: "test-chip".into(),
            unified_memory_bytes: 2_000_000_000,
            metal_recommended_working_set_bytes: Some(1_500_000_000),
            memory_pressure: None,
            free_ram_bytes: None,
        },
        vec![],
    )
    .await;
    let package_dir = write_transformers_package(&h.models);
    h.post_json("/api/scan", serde_json::json!({})).await;
    std::fs::write(
        package_dir.join("model-00002-of-00002.safetensors"),
        "replaced!",
    )
    .unwrap();

    let catalog = h.json("/api/catalog").await;
    let package = catalog["packages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|package| package["id"] == "Qwen/Qwen2.5-0.5B-Instruct")
        .unwrap();
    assert_eq!(package["ready"], false, "{package}");
    assert_eq!(package["readiness_reason"], "missing_required_files");
    h.listening.shutdown().await;
}

#[tokio::test]
async fn nonexistent_gguf_worker_keeps_package_not_ready() {
    let h = Harness::start_with(
        HardwareSnapshot {
            device_class: "apple_silicon".into(),
            chip: "test-chip".into(),
            unified_memory_bytes: 2_000_000_000,
            metal_recommended_working_set_bytes: Some(1_000_000_000),
            memory_pressure: None,
            free_ram_bytes: None,
        },
        stub_launcher(),
        Some(PathBuf::from("/missing/llama-server")),
    )
    .await;
    write_artifact(
        &h.models,
        "Qwen",
        "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        100_000,
        llm_meta(),
    );
    let catalog = h
        .post_json("/api/scan", serde_json::json!({}))
        .await
        .json::<Value>()
        .await
        .unwrap();
    let package = catalog["packages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|package| package["id"] == "qit/qwen2.5-0.5b-instruct-q4_k_m")
        .unwrap();
    assert_eq!(package["ready"], false, "{package}");
    assert_eq!(package["readiness_reason"], "runtime_missing");
    h.listening.shutdown().await;
}

#[tokio::test]
async fn transformers_package_launches_after_health_proxies_chat_and_stops() {
    let h = Harness::start_with_transformers(
        HardwareSnapshot {
            device_class: "apple_silicon".into(),
            chip: "test-chip".into(),
            unified_memory_bytes: 2_000_000_000,
            metal_recommended_working_set_bytes: Some(1_500_000_000),
            memory_pressure: None,
            free_ram_bytes: None,
        },
        vec![
            "--health-warmup-ms".into(),
            "400".into(),
            "--echo-usage".into(),
        ],
    )
    .await;
    let package_dir = write_transformers_package(&h.models);
    let catalog = h
        .post_json("/api/scan", serde_json::json!({}))
        .await
        .json::<Value>()
        .await
        .unwrap();
    let package = catalog["packages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|package| package["id"] == "Qwen/Qwen2.5-0.5B-Instruct")
        .unwrap();
    assert_eq!(package["ready"], true, "{package}");

    let serve_profile = serde_json::json!({
        "context_length": 4096,
        "runtime_settings": {}
    });
    let client = reqwest::Client::new();
    let start_url = h.url("/api/sessions");
    let start_profile = serve_profile.clone();
    let start = tokio::spawn(async move {
        client
            .post(start_url)
            .json(&serde_json::json!({
                "package_id": "Qwen/Qwen2.5-0.5B-Instruct",
                "serve_profile": start_profile
            }))
            .send()
            .await
            .unwrap()
    });
    let starting = wait_until(&h, "/api/sessions", |sessions| {
        sessions
            .as_array()
            .unwrap()
            .iter()
            .any(|session| session["status"] == "starting")
    })
    .await;
    assert_eq!(starting[0]["runtime_recipe"], "transformers_external");

    let started = start.await.unwrap();
    assert_eq!(started.status(), 200);
    let started: Value = started.json().await.unwrap();
    assert_eq!(started["status"], "loaded");
    assert_eq!(started["target_id"], "Qwen/Qwen2.5-0.5B-Instruct");
    assert!(started.get("artifact_id").is_none(), "{started}");
    assert_eq!(started["package_id"], "Qwen/Qwen2.5-0.5B-Instruct");
    assert_eq!(started["runtime_recipe"], "transformers_external");
    assert_eq!(started["serve_profile"], serve_profile);
    assert_eq!(h.json("/api/hardware").await["headroom_bytes"], 300_000_000);
    let session_id = started["id"].as_str().unwrap();
    let log_path = started["log_path"].as_str().unwrap();
    let pid = worker_pid_from_log(log_path);
    assert!(process_alive(pid));
    let log = std::fs::read_to_string(log_path).unwrap();
    assert!(log.contains("--model"), "{log}");
    assert!(log.contains(&package_dir.display().to_string()), "{log}");
    assert!(log.contains("--context-length"), "{log}");
    assert!(log.contains("4096"), "{log}");

    let generated = reqwest::Client::new()
        .post(h.url("/api/generate"))
        .json(&serde_json::json!({
            "package_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "serve_profile": serve_profile,
            "session_id": session_id,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 7
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(generated.status(), 200);
    let body = generated.text().await.unwrap();
    let events = sse_events(&body);
    assert_eq!(events[0], ("token".into(), "hello".into()), "{body}");
    assert_eq!(events[1], ("token".into(), " world".into()), "{body}");
    assert_eq!(events.last().unwrap().0, "done", "{body}");

    let stopped = h
        .post_json(
            &format!("/api/sessions/{session_id}/stop"),
            serde_json::json!({}),
        )
        .await;
    assert_eq!(stopped.status(), 200);
    assert_eq!(
        stopped.json::<Value>().await.unwrap()["status"],
        "not_loaded"
    );
    assert!(!process_alive(pid), "stub worker {pid} still running");
    h.listening.shutdown().await;
}

#[tokio::test]
async fn insufficient_memory_package_is_refused_before_worker_launch() {
    let h = Harness::start_with_transformers(
        HardwareSnapshot {
            device_class: "apple_silicon".into(),
            chip: "test-chip".into(),
            unified_memory_bytes: 1_000_000_000,
            metal_recommended_working_set_bytes: Some(500_000_000),
            memory_pressure: None,
            free_ram_bytes: Some(900_000_000),
        },
        vec![],
    )
    .await;
    write_transformers_package(&h.models);
    let catalog = h
        .post_json("/api/scan", serde_json::json!({}))
        .await
        .json::<Value>()
        .await
        .unwrap();
    let package = catalog["packages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|package| package["id"] == "Qwen/Qwen2.5-0.5B-Instruct")
        .unwrap();
    assert_eq!(package["fits"], false);
    assert_eq!(package["readiness_reason"], "insufficient_memory");

    let start = h
        .post_json(
            "/api/sessions",
            serde_json::json!({"package_id": "Qwen/Qwen2.5-0.5B-Instruct"}),
        )
        .await;
    assert_eq!(start.status(), 400);
    assert_eq!(
        start.json::<Value>().await.unwrap()["error"],
        "model package does not fit the stable budget"
    );
    assert!(h.json("/api/sessions").await.as_array().unwrap().is_empty());
    h.listening.shutdown().await;
}

#[tokio::test]
async fn transformers_worker_failure_retains_failed_session_and_log() {
    let h = Harness::start_with_transformers(
        HardwareSnapshot {
            device_class: "apple_silicon".into(),
            chip: "test-chip".into(),
            unified_memory_bytes: 2_000_000_000,
            metal_recommended_working_set_bytes: Some(1_500_000_000),
            memory_pressure: None,
            free_ram_bytes: None,
        },
        vec!["--crash".into()],
    )
    .await;
    write_transformers_package(&h.models);
    h.post_json("/api/scan", serde_json::json!({})).await;
    let response = h
        .post_json(
            "/api/sessions",
            serde_json::json!({"package_id": "Qwen/Qwen2.5-0.5B-Instruct"}),
        )
        .await;
    assert_eq!(response.status(), 400);
    let error: Value = response.json().await.unwrap();
    assert!(error["error"].as_str().unwrap().contains("worker exited"));
    let sessions = h.json("/api/sessions").await;
    assert_eq!(sessions[0]["status"], "failed", "{sessions}");
    assert!(
        sessions[0]["last_error"]
            .as_str()
            .unwrap()
            .contains("worker exited"),
        "{sessions}"
    );
    let log_path = sessions[0]["log_path"].as_str().unwrap();
    assert!(std::path::Path::new(log_path).is_file(), "{sessions}");
    h.listening.shutdown().await;
}

#[tokio::test]
async fn stopping_transformers_package_during_health_wait_cannot_reload_it() {
    let h = Harness::start_with_transformers(
        HardwareSnapshot {
            device_class: "apple_silicon".into(),
            chip: "test-chip".into(),
            unified_memory_bytes: 2_000_000_000,
            metal_recommended_working_set_bytes: Some(1_500_000_000),
            memory_pressure: None,
            free_ram_bytes: None,
        },
        vec!["--health-warmup-ms".into(), "1000".into()],
    )
    .await;
    write_transformers_package(&h.models);
    h.post_json("/api/scan", serde_json::json!({})).await;
    let start_url = h.url("/api/sessions");
    let start = tokio::spawn(async move {
        reqwest::Client::new()
            .post(start_url)
            .json(&serde_json::json!({
                "package_id": "Qwen/Qwen2.5-0.5B-Instruct"
            }))
            .send()
            .await
            .unwrap()
    });
    let starting = wait_until(&h, "/api/sessions", |sessions| {
        sessions
            .as_array()
            .unwrap()
            .iter()
            .any(|session| session["status"] == "starting")
    })
    .await;
    let session_id = starting[0]["id"].as_str().unwrap().to_string();
    let log_path = starting[0]["log_path"].as_str().unwrap();
    let pid = wait_for_worker_pid(log_path).await;
    let stopped = h
        .post_json(
            &format!("/api/sessions/{session_id}/stop"),
            serde_json::json!({}),
        )
        .await;
    assert_eq!(stopped.status(), 200);
    assert_eq!(
        stopped.json::<Value>().await.unwrap()["status"],
        "not_loaded"
    );

    let start_response = start.await.unwrap();
    assert_eq!(start_response.status(), 200);
    let start_response: Value = start_response.json().await.unwrap();
    if start_response["status"] == "loaded" {
        h.post_json(
            &format!("/api/sessions/{session_id}/stop"),
            serde_json::json!({}),
        )
        .await;
    }
    assert_eq!(start_response["status"], "not_loaded", "{start_response}");
    let sessions = h.json("/api/sessions").await;
    assert_eq!(sessions[0]["status"], "not_loaded", "{sessions}");
    assert!(!process_alive(pid), "stub worker {pid} still running");
    h.listening.shutdown().await;
}

#[tokio::test]
async fn gguf_package_uses_serve_profile_through_session_lifecycle() {
    let h = Harness::start_with(
        HardwareSnapshot {
            device_class: "apple_silicon".into(),
            chip: "test-chip".into(),
            unified_memory_bytes: 2_000_000_000,
            metal_recommended_working_set_bytes: Some(1_000_000_000),
            memory_pressure: None,
            free_ram_bytes: None,
        },
        stub_launcher(),
        Some(PathBuf::from(env!("CARGO_BIN_EXE_qit-stub-worker"))),
    )
    .await;
    write_artifact(
        &h.models,
        "Qwen",
        "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        100_000,
        llm_meta(),
    );
    h.post_json("/api/scan", serde_json::json!({})).await;
    let package_id = "qit/qwen2.5-0.5b-instruct-q4_k_m";
    let serve_profile = serde_json::json!({
        "context_length": 4096,
        "runtime_settings": {
            "gpu_layers": 7,
            "parallel": 2
        }
    });
    let catalog = h.json("/api/catalog").await;
    assert_eq!(catalog["packages"][0]["ready"], true, "{catalog}");
    assert_eq!(catalog["packages"][0]["runtime_recipe"], "llama_cpp");

    let pin = h
        .post_json(
            "/api/pins",
            serde_json::json!({
                "package_id": package_id,
                "serve_profile": serve_profile
            }),
        )
        .await;
    assert_eq!(pin.status(), 200);
    let pin: Value = pin.json().await.unwrap();
    assert_eq!(pin["package_id"], package_id);
    assert_eq!(pin["serve_profile"], serve_profile);
    assert!(pin.get("n_gpu_layers").is_none(), "{pin}");
    assert!(pin.get("n_parallel").is_none(), "{pin}");
    let what_if = h
        .post_json(
            "/api/what-ifs",
            serde_json::json!({
                "package_id": package_id,
                "serve_profile": serve_profile
            }),
        )
        .await;
    assert_eq!(what_if.status(), 200);

    let before_start = h.json("/api/capacity").await;
    let started = h
        .post_json(
            "/api/sessions",
            serde_json::json!({
                "package_id": package_id,
                "serve_profile": serve_profile
            }),
        )
        .await;
    assert_eq!(started.status(), 200);
    let started: Value = started.json().await.unwrap();
    assert_eq!(started["status"], "loaded");
    assert_eq!(started["package_id"], package_id);
    assert_eq!(started["serve_profile"], serve_profile);
    assert!(started.get("n_gpu_layers").is_none(), "{started}");
    assert!(started.get("n_parallel").is_none(), "{started}");
    let while_loaded = h.json("/api/capacity").await;
    assert_eq!(
        while_loaded["hardware"]["headroom_bytes"],
        before_start["hardware"]["headroom_bytes"]
    );

    let generated = h
        .post_json(
            "/api/generate",
            serde_json::json!({
                "package_id": package_id,
                "serve_profile": serve_profile,
                "session_id": started["id"],
                "messages": [{"role": "user", "content": "hi"}]
            }),
        )
        .await;
    assert_eq!(generated.status(), 200);
    let body = generated.text().await.unwrap();
    assert!(body.contains("event: token"), "{body}");
    assert!(body.contains("hello"), "{body}");
    assert!(body.contains("event: done"), "{body}");

    let stopped = h
        .post_json(
            &format!("/api/sessions/{}/stop", started["id"].as_str().unwrap()),
            serde_json::json!({}),
        )
        .await;
    assert_eq!(stopped.status(), 200);
    let stopped: Value = stopped.json().await.unwrap();
    assert_eq!(stopped["status"], "not_loaded");
    let after_stop = h.json("/api/capacity").await;
    assert_eq!(after_stop["pins"][0]["package_id"], package_id);
    assert_eq!(after_stop["pins"][0]["serve_profile"], serve_profile);
    assert_eq!(after_stop["what_ifs"][0]["package_id"], package_id);
    assert_eq!(after_stop["what_ifs"][0]["serve_profile"], serve_profile);
    assert_eq!(after_stop["sessions"][0]["package_id"], package_id);
    assert_eq!(after_stop["sessions"][0]["serve_profile"], serve_profile);
    let h = h.restart(Some(2_000_000)).await;
    let after_restart = h.json("/api/capacity").await;
    assert_eq!(after_restart["pins"][0]["package_id"], package_id);
    assert_eq!(after_restart["pins"][0]["serve_profile"], serve_profile);
    assert_eq!(after_restart["what_ifs"].as_array().unwrap().len(), 0);
    assert_eq!(after_restart["sessions"][0]["package_id"], package_id);
    assert_eq!(after_restart["sessions"][0]["serve_profile"], serve_profile);
    assert_eq!(after_restart["sessions"][0]["status"], "not_loaded");
    let restarted = h
        .post_json(
            "/api/sessions",
            serde_json::json!({
                "package_id": package_id,
                "serve_profile": serve_profile
            }),
        )
        .await;
    assert_eq!(restarted.status(), 200);
    let restarted: Value = restarted.json().await.unwrap();
    assert_eq!(restarted["id"], started["id"]);
    assert_eq!(h.json("/api/sessions").await.as_array().unwrap().len(), 1);
    h.listening.shutdown().await;
}

#[tokio::test]
async fn scan_registers_gguf_only() {
    let h = Harness::start(probe_with_free(None), vec![]).await;
    write_artifact(
        &h.models,
        "nvidia",
        "NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf",
        100_000,
        llm_meta(),
    );
    let other = h.models.join("microsoft");
    std::fs::create_dir_all(&other).unwrap();
    std::fs::write(other.join("model.safetensors"), [0u8; 32]).unwrap();
    std::fs::write(other.join("notes.txt"), "nope").unwrap();
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
    assert!(small["peak_rss_bytes"].as_u64().unwrap_or(0) > 0, "{small}");
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

fn slow_stream_args() -> Vec<String> {
    vec![
        "--token-delay-ms".into(),
        "200".into(),
        "--n-tokens".into(),
        "50".into(),
    ]
}

async fn read_first_token(resp: &mut reqwest::Response) {
    let chunk = resp.chunk().await.unwrap().expect("first sse chunk");
    let text = String::from_utf8_lossy(&chunk);
    assert!(text.contains("event: token"), "{text}");
}

async fn wait_until<F: Fn(&Value) -> bool>(h: &Harness, path: &str, ok: F) -> Value {
    let mut last = Value::Null;
    for _ in 0..50 {
        last = h.json(path).await;
        if ok(&last) {
            return last;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("condition not met for {path}: {last}");
}

fn worker_pid_from_log(log_path: &str) -> u32 {
    let log = std::fs::read_to_string(log_path).unwrap();
    worker_pid(&log).expect("stub worker pid in log")
}

fn worker_pid(log: &str) -> Option<u32> {
    log.lines().find_map(|l| {
        l.strip_prefix("stub worker pid ")
            .and_then(|p| p.trim().parse().ok())
    })
}

async fn wait_for_worker_pid(log_path: &str) -> u32 {
    for _ in 0..50 {
        if let Ok(log) = std::fs::read_to_string(log_path) {
            if let Some(pid) = worker_pid(&log) {
                return pid;
            }
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("stub worker pid not found in {log_path}");
}

fn process_alive(pid: u32) -> bool {
    std::process::Command::new("kill")
        .arg("-0")
        .arg(pid.to_string())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[tokio::test]
async fn client_abort_stops_ephemeral_worker_without_measurement() {
    let h = Harness::start(probe_with_free(None), slow_stream_args()).await;
    write_artifact(&h.models, "org", "small.gguf", 100_000, llm_meta());
    h.post_json("/api/scan", serde_json::json!({})).await;
    let client = reqwest::Client::new();
    let mut resp = client
        .post(h.url("/api/generate"))
        .json(&serde_json::json!({
            "artifact_id": "org/small.gguf",
            "prompt": "hi",
            "n_ctx": 4096
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    read_first_token(&mut resp).await;
    let sessions = h.json("/api/sessions").await;
    let log_path = sessions[0]["log_path"].as_str().unwrap().to_string();
    let pid = worker_pid_from_log(&log_path);
    assert!(process_alive(pid));
    drop(resp);

    let sessions = wait_until(&h, "/api/sessions", |v| {
        v.as_array()
            .unwrap()
            .iter()
            .all(|s| s["status"] == "not_loaded")
    })
    .await;
    assert_eq!(sessions.as_array().unwrap().len(), 1, "{sessions}");
    assert!(!process_alive(pid), "stub worker {pid} still running");
    let catalog = h.json("/api/catalog").await;
    let small = &catalog["artifacts"][0];
    assert!(small["throughput_tps"].is_null(), "{small}");
    h.listening.shutdown().await;
}

#[tokio::test]
async fn concurrent_generate_is_refused_until_slot_frees() {
    let h = Harness::start(probe_with_free(None), slow_stream_args()).await;
    write_artifact(&h.models, "org", "small.gguf", 100_000, llm_meta());
    h.post_json("/api/scan", serde_json::json!({})).await;
    let body = serde_json::json!({
        "artifact_id": "org/small.gguf",
        "prompt": "hi",
        "n_ctx": 4096
    });
    let client = reqwest::Client::new();
    let mut first = client
        .post(h.url("/api/generate"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(first.status(), 200);
    read_first_token(&mut first).await;

    let second = h.post_json("/api/generate", body.clone()).await;
    assert_eq!(second.status(), 409);
    let err: Value = second.json().await.unwrap();
    assert_eq!(err["error"], "generate in flight");
    let sessions = h.json("/api/sessions").await;
    assert_eq!(sessions.as_array().unwrap().len(), 1, "{sessions}");

    drop(first);
    wait_until(&h, "/api/sessions", |v| {
        v.as_array()
            .unwrap()
            .iter()
            .all(|s| s["status"] == "not_loaded")
    })
    .await;
    let mut third = client
        .post(h.url("/api/generate"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(third.status(), 200);
    read_first_token(&mut third).await;
    drop(third);
    h.listening.shutdown().await;
}

#[tokio::test]
async fn os_reserve_setting_moves_budget_and_survives_restart() {
    let h = Harness::start_full(probe_with_free(None), stub_launcher(), None, None).await;
    let settings = h.json("/api/settings").await;
    assert_eq!(settings["os_reserve_bytes"], Value::Null);
    assert_eq!(settings["os_reserve_source"], "default");
    assert_eq!(settings["effective_os_reserve_bytes"], 2_500_000);
    assert_eq!(h.json("/api/hardware").await["budget_bytes"], 5_000_000);

    let put = h
        .put_json(
            "/api/settings",
            serde_json::json!({"os_reserve_bytes": 7_000_000}),
        )
        .await;
    assert_eq!(put.status(), 200);
    let settings: Value = put.json().await.unwrap();
    assert_eq!(settings["os_reserve_bytes"], 7_000_000);
    assert_eq!(settings["os_reserve_source"], "setting");
    let hw = h.json("/api/hardware").await;
    assert_eq!(hw["os_reserve_bytes"], 7_000_000);
    assert_eq!(hw["budget_bytes"], 3_000_000);

    let h = h.restart(None).await;
    let hw = h.json("/api/hardware").await;
    assert_eq!(hw["os_reserve_bytes"], 7_000_000);
    assert_eq!(hw["budget_bytes"], 3_000_000);

    let h = h.restart(Some(2_000_000)).await;
    let settings = h.json("/api/settings").await;
    assert_eq!(settings["os_reserve_bytes"], 7_000_000);
    assert_eq!(settings["os_reserve_source"], "env");
    assert_eq!(settings["effective_os_reserve_bytes"], 2_000_000);
    assert_eq!(h.json("/api/hardware").await["os_reserve_bytes"], 2_000_000);

    let h = h.restart(None).await;
    let reset = h
        .put_json(
            "/api/settings",
            serde_json::json!({"os_reserve_bytes": null}),
        )
        .await;
    assert_eq!(reset.status(), 200);
    assert_eq!(h.json("/api/hardware").await["os_reserve_bytes"], 2_500_000);
    let rejected = h
        .put_json(
            "/api/settings",
            serde_json::json!({"os_reserve_bytes": 20_000_000}),
        )
        .await;
    assert_eq!(rejected.status(), 400);
    h.listening.shutdown().await;
}

fn sse_events(body: &str) -> Vec<(String, String)> {
    body.split("\n\n")
        .filter(|frame| frame.contains("event:"))
        .map(|frame| {
            let mut event = String::new();
            let mut data = String::new();
            for line in frame.lines() {
                if let Some(v) = line.strip_prefix("event: ") {
                    event = v.to_string();
                } else if let Some(v) = line.strip_prefix("data: ") {
                    data.push_str(v);
                }
            }
            (event, data)
        })
        .collect()
}

#[tokio::test]
async fn generate_accepts_messages_and_reports_usage_on_done() {
    let h = Harness::start(probe_with_free(None), vec!["--echo-usage".into()]).await;
    write_artifact(&h.models, "org", "small.gguf", 100_000, llm_meta());
    h.post_json("/api/scan", serde_json::json!({})).await;
    let body = reqwest::Client::new()
        .post(h.url("/api/generate"))
        .json(&serde_json::json!({
            "artifact_id": "org/small.gguf",
            "n_ctx": 4096,
            "max_tokens": 7,
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello world"},
                {"role": "user", "content": "again"}
            ]
        }))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    let events = sse_events(&body);
    let (last_event, last_data) = events.last().expect("done event");
    assert_eq!(last_event, "done", "{body}");
    let done: Value = serde_json::from_str(last_data).unwrap();
    assert_eq!(done["n_ctx"], 4096);
    assert_eq!(done["completion_tokens"], 2);
    assert_eq!(done["prompt_tokens"], 3 * 100 + 7, "{done}");
    h.listening.shutdown().await;
}

#[tokio::test]
async fn generate_without_prompt_or_messages_is_rejected() {
    let h = Harness::start(probe_with_free(None), vec![]).await;
    write_artifact(&h.models, "org", "small.gguf", 100_000, llm_meta());
    h.post_json("/api/scan", serde_json::json!({})).await;
    let resp = h
        .post_json(
            "/api/generate",
            serde_json::json!({"artifact_id": "org/small.gguf", "n_ctx": 4096}),
        )
        .await;
    assert_eq!(resp.status(), 400);
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
    assert_eq!(
        hw["worker_path"].as_str().unwrap(),
        worker.to_string_lossy()
    );
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
    assert!(
        err.contains("QIT_WORKER_PATH") || err.contains("llama-server"),
        "{err}"
    );
    h.listening.shutdown().await;
}

#[tokio::test]
async fn worker_waits_for_health_200_before_loaded() {
    let h = Harness::start(
        probe_with_free(None),
        vec!["--health-warmup-ms".into(), "400".into()],
    )
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
    h.post_json(
        &format!("/api/sessions/{}/stop", first["id"]),
        serde_json::json!({}),
    )
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
        .filter(|s| {
            s["artifact_id"] == "org/small.gguf" && s["serve_profile"]["context_length"] == 4096
        })
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
        has_chat_template: true,
        ..GgufMeta::default()
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

#[tokio::test]
async fn warn_pressure_surfaces_without_changing_fit() {
    let warn = Harness::start(probe_live(Some(50), Some("warn")), vec![]).await;
    write_artifact(&warn.models, "org", "small.gguf", 100_000, llm_meta());
    warn.post_json("/api/scan", serde_json::json!({})).await;
    let hw = warn.json("/api/hardware").await;
    assert_eq!(hw["memory_pressure"], "warn");
    assert_eq!(hw["free_ram_bytes"], 50);
    assert_eq!(hw["budget_bytes"], 5_000_000);
    let cap = warn.json("/api/capacity").await;
    assert_eq!(cap["hardware"]["memory_pressure"], "warn");
    let fit = warn.json("/api/catalog?n_ctx=4096").await["artifacts"][0]["fit"].clone();
    warn.listening.shutdown().await;

    let quiet = Harness::start(probe_live(Some(9_000_000), Some("normal")), vec![]).await;
    write_artifact(&quiet.models, "org", "small.gguf", 100_000, llm_meta());
    quiet.post_json("/api/scan", serde_json::json!({})).await;
    let other = quiet.json("/api/catalog?n_ctx=4096").await["artifacts"][0]["fit"].clone();
    assert_eq!(fit, other);
    quiet.listening.shutdown().await;
}

fn hybrid_meta() -> GgufMeta {
    GgufMeta {
        architecture: Some("nemotron_h".into()),
        context_length: Some(1_048_576),
        block_count: Some(42),
        embedding_length: Some(3136),
        head_count: Some(40),
        head_count_kv_layers: Some(vec![
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0,
            0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]),
        feed_forward_layers: Some(vec![
            0, 12544, 0, 12544, 0, 12544, 0, 0, 12544, 0, 12544, 0, 0, 12544, 0, 12544, 0, 0,
            12544, 0, 12544, 0, 12544, 0, 0, 12544, 0, 12544, 0, 12544, 0, 0, 0, 12544, 0, 0, 0,
            12544, 0, 12544, 0, 12544,
        ]),
        key_length: Some(128),
        value_length: Some(128),
        ssm_conv_kernel: Some(4),
        ssm_inner_size: Some(7680),
        ssm_state_size: Some(128),
        ssm_group_count: Some(8),
        has_chat_template: true,
        ..GgufMeta::default()
    }
}

#[tokio::test]
async fn hybrid_estimate_is_below_uniform_and_llama_is_unchanged() {
    let h = Harness::start(probe_with_free(None), vec![]).await;
    write_artifact(&h.models, "org", "llama.gguf", 100_000, llm_meta());
    write_artifact(&h.models, "org", "hybrid.gguf", 100_000, hybrid_meta());
    h.post_json("/api/scan", serde_json::json!({})).await;
    let catalog = h.json("/api/catalog?n_ctx=4096").await;
    let llama = catalog["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["id"] == "org/llama.gguf")
        .unwrap();
    let hybrid = catalog["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["id"] == "org/hybrid.gguf")
        .unwrap();
    assert_eq!(llama["estimate_bytes"], 4_294_304);
    assert_eq!(hybrid["estimate_bytes"], 152_235_680);
    assert_eq!(hybrid["confidence"], "headers");
    h.listening.shutdown().await;
}

#[tokio::test]
async fn generate_refused_for_embedding_artifact() {
    let h = Harness::start(probe_with_free(None), vec![]).await;
    write_artifact(&h.models, "org", "chat.gguf", 100_000, llm_meta());
    write_artifact(
        &h.models,
        "org",
        "embed.gguf",
        100_000,
        GgufMeta {
            architecture: Some("qwen3".into()),
            context_length: Some(32768),
            block_count: Some(1),
            embedding_length: Some(256),
            head_count: Some(4),
            pooling_type: Some(1),
            has_chat_template: true,
            ..GgufMeta::default()
        },
    );
    h.post_json("/api/scan", serde_json::json!({})).await;
    let catalog = h.json("/api/catalog").await;
    let chat = catalog["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["id"] == "org/chat.gguf")
        .unwrap();
    let embed = catalog["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["id"] == "org/embed.gguf")
        .unwrap();
    assert_eq!(chat["kind"], "instruct");
    assert_eq!(chat["generate_supported"], true);
    assert_eq!(embed["kind"], "embedding");
    assert_eq!(embed["generate_supported"], false);
    let refused = h
        .post_json(
            "/api/generate",
            serde_json::json!({"artifact_id":"org/embed.gguf","prompt":"hi","n_ctx":4096}),
        )
        .await;
    assert_eq!(refused.status(), 400);
    let body: Value = refused.json().await.unwrap();
    assert!(
        body["error"].as_str().unwrap().contains("embedding"),
        "{body}"
    );
    h.listening.shutdown().await;
}
