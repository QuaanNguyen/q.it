use std::collections::HashMap;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::store::ArtifactRow;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionStatus {
    NotLoaded,
    Starting,
    Loaded,
    Stopping,
    Failed,
}

#[derive(Clone, Debug, Serialize)]
pub struct SessionView {
    pub id: String,
    pub artifact_id: String,
    pub n_ctx: u32,
    pub n_gpu_layers: i32,
    pub n_parallel: u32,
    pub status: SessionStatus,
}

pub struct LaunchRequest {
    pub artifact_path: PathBuf,
    pub n_ctx: u32,
    pub n_gpu_layers: i32,
    pub n_parallel: u32,
    pub log_path: PathBuf,
}

pub struct LaunchedWorker {
    pub child: Child,
    pub base_url: String,
}

pub trait WorkerLauncher: Send + Sync {
    fn launch(&self, request: LaunchRequest) -> Result<LaunchedWorker, String>;
}

pub struct LlamaServerLauncher {
    pub binary: Option<PathBuf>,
}

impl WorkerLauncher for LlamaServerLauncher {
    fn launch(&self, request: LaunchRequest) -> Result<LaunchedWorker, String> {
        let binary = self.binary.as_ref().ok_or_else(|| {
            "no worker binary (set QIT_WORKER_PATH or LLAMA_SERVER_PATH)".to_string()
        })?;
        let port = free_loopback_port()?;
        let log =
            std::fs::File::create(&request.log_path).map_err(|e| format!("worker log: {e}"))?;
        let err = log.try_clone().map_err(|e| format!("worker log: {e}"))?;
        let child = Command::new(binary)
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .arg("-m")
            .arg(&request.artifact_path)
            .arg("-c")
            .arg(request.n_ctx.to_string())
            .arg("-ngl")
            .arg(request.n_gpu_layers.to_string())
            .arg("--parallel")
            .arg(request.n_parallel.to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::from(log))
            .stderr(Stdio::from(err))
            .spawn()
            .map_err(|e| format!("spawn worker: {e}"))?;
        Ok(LaunchedWorker {
            child,
            base_url: format!("http://127.0.0.1:{port}"),
        })
    }
}

pub struct StubBinLauncher {
    pub binary: PathBuf,
    pub extra_args: Vec<String>,
}

impl WorkerLauncher for StubBinLauncher {
    fn launch(&self, request: LaunchRequest) -> Result<LaunchedWorker, String> {
        let port = free_loopback_port()?;
        let log =
            std::fs::File::create(&request.log_path).map_err(|e| format!("worker log: {e}"))?;
        let err = log.try_clone().map_err(|e| format!("worker log: {e}"))?;
        let mut cmd = Command::new(&self.binary);
        cmd.arg("--port")
            .arg(port.to_string())
            .args(&self.extra_args)
            .stdin(Stdio::null())
            .stdout(Stdio::from(log))
            .stderr(Stdio::from(err));
        let child = cmd.spawn().map_err(|e| format!("spawn stub worker: {e}"))?;
        Ok(LaunchedWorker {
            child,
            base_url: format!("http://127.0.0.1:{port}"),
        })
    }
}

fn free_loopback_port() -> Result<u16, String> {
    let listener =
        std::net::TcpListener::bind("127.0.0.1:0").map_err(|e| format!("pick port: {e}"))?;
    let port = listener
        .local_addr()
        .map_err(|e| format!("pick port: {e}"))?
        .port();
    drop(listener);
    Ok(port)
}

struct LiveSession {
    view: SessionView,
    child: Option<Child>,
    base_url: Option<String>,
}

pub struct Supervisor {
    launcher: Arc<dyn WorkerLauncher>,
    sessions: Mutex<HashMap<String, LiveSession>>,
}

impl Supervisor {
    pub fn new(launcher: Arc<dyn WorkerLauncher>) -> Self {
        Self {
            launcher,
            sessions: Mutex::new(HashMap::new()),
        }
    }

    pub async fn list(&self) -> Vec<SessionView> {
        self.sessions
            .lock()
            .await
            .values()
            .map(|s| s.view.clone())
            .collect()
    }

    pub async fn get(&self, id: &str) -> Option<SessionView> {
        self.sessions.lock().await.get(id).map(|s| s.view.clone())
    }

    pub async fn find_loaded(
        &self,
        artifact_id: &str,
        n_ctx: u32,
        n_gpu_layers: i32,
        n_parallel: u32,
    ) -> Option<SessionView> {
        self.reap().await;
        self.sessions.lock().await.values().find_map(|s| {
            if s.view.artifact_id == artifact_id
                && s.view.n_ctx == n_ctx
                && s.view.n_gpu_layers == n_gpu_layers
                && s.view.n_parallel == n_parallel
                && s.view.status == SessionStatus::Loaded
            {
                Some(s.view.clone())
            } else {
                None
            }
        })
    }

    pub async fn loaded_rss_bytes(&self) -> u64 {
        0
    }

    pub async fn start(
        &self,
        artifact: &ArtifactRow,
        n_ctx: u32,
        n_gpu_layers: i32,
        n_parallel: u32,
        log_path: PathBuf,
    ) -> Result<SessionView, String> {
        self.reap().await;
        if let Some(existing) = self
            .find_loaded(&artifact.id, n_ctx, n_gpu_layers, n_parallel)
            .await
        {
            return Ok(existing);
        }
        let id = Uuid::new_v4().to_string();
        {
            let mut guard = self.sessions.lock().await;
            guard.insert(
                id.clone(),
                LiveSession {
                    view: SessionView {
                        id: id.clone(),
                        artifact_id: artifact.id.clone(),
                        n_ctx,
                        n_gpu_layers,
                        n_parallel,
                        status: SessionStatus::Starting,
                    },
                    child: None,
                    base_url: None,
                },
            );
        }
        let launched = match self.launcher.launch(LaunchRequest {
            artifact_path: artifact.path.clone(),
            n_ctx,
            n_gpu_layers,
            n_parallel,
            log_path,
        }) {
            Ok(w) => w,
            Err(e) => {
                if let Some(s) = self.sessions.lock().await.get_mut(&id) {
                    s.view.status = SessionStatus::Failed;
                }
                return Err(e);
            }
        };
        let mut launched = launched;
        if let Err(e) = wait_ready(&launched.base_url, &mut launched.child).await {
            let mut guard = self.sessions.lock().await;
            if let Some(s) = guard.get_mut(&id) {
                s.view.status = SessionStatus::Failed;
                s.child = Some(launched.child);
            }
            return Err(e);
        }
        let mut guard = self.sessions.lock().await;
        if let Some(s) = guard.get_mut(&id) {
            if s.view.status == SessionStatus::Stopping {
                drop(kill_child(launched.child));
                s.view.status = SessionStatus::NotLoaded;
                s.child = None;
                s.base_url = None;
                return Ok(s.view.clone());
            }
            s.view.status = SessionStatus::Loaded;
            s.base_url = Some(launched.base_url);
            s.child = Some(launched.child);
            return Ok(s.view.clone());
        }
        drop(kill_child(launched.child));
        Err("session vanished".into())
    }

    pub async fn stop(&self, id: &str) -> Result<SessionView, String> {
        let mut guard = self.sessions.lock().await;
        let session = guard
            .get_mut(id)
            .ok_or_else(|| "session not found".to_string())?;
        session.view.status = SessionStatus::Stopping;
        if let Some(child) = session.child.take() {
            drop(kill_child(child));
        }
        session.base_url = None;
        session.view.status = SessionStatus::NotLoaded;
        Ok(session.view.clone())
    }

    pub async fn base_url(&self, id: &str) -> Option<String> {
        self.sessions
            .lock()
            .await
            .get(id)
            .and_then(|s| s.base_url.clone())
    }

    pub async fn reap(&self) {
        let mut guard = self.sessions.lock().await;
        for session in guard.values_mut() {
            if matches!(
                session.view.status,
                SessionStatus::Loaded | SessionStatus::Starting
            ) {
                if let Some(child) = session.child.as_mut() {
                    if let Ok(Some(status)) = child.try_wait() {
                        if !status.success() || session.view.status == SessionStatus::Loaded {
                            session.view.status = SessionStatus::Failed;
                            session.base_url = None;
                            session.child = None;
                        }
                    }
                }
            }
        }
    }
}

fn kill_child(mut child: Child) -> Result<(), std::io::Error> {
    let _ = child.kill();
    let _ = child.wait();
    Ok(())
}

async fn wait_ready(base_url: &str, child: &mut Child) -> Result<(), String> {
    let client = reqwest::Client::new();
    let url = format!("{base_url}/health");
    let started = std::time::Instant::now();
    while started.elapsed() < Duration::from_secs(15) {
        if let Ok(Some(status)) = child.try_wait() {
            return Err(format!("worker exited ({status})"));
        }
        if client.get(&url).send().await.is_ok() {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    Err("worker did not become ready".into())
}

pub async fn proxy_generate<F, Fut>(
    base_url: &str,
    prompt: &str,
    cancel: tokio::sync::watch::Receiver<bool>,
    mut on_token: F,
) -> Result<(u32, f64), String>
where
    F: FnMut(String) -> Fut,
    Fut: std::future::Future<Output = Result<(), String>>,
{
    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": prompt}],
        "stream": true,
        "max_tokens": 64
    });
    let mut resp = client
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("worker http {}", resp.status()));
    }
    let t0 = std::time::Instant::now();
    let mut n_tokens = 0u32;
    let mut buf = bytes::BytesMut::new();
    while let Some(chunk) = resp.chunk().await.map_err(|e| e.to_string())? {
        if *cancel.borrow() {
            break;
        }
        buf.extend_from_slice(&chunk);
        while let Some(pos) = buf.windows(2).position(|w| w == b"\n\n") {
            let frame = buf.split_to(pos + 2);
            if let Some(token) = parse_openai_sse(&frame) {
                n_tokens += 1;
                on_token(token).await?;
            }
        }
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    Ok((n_tokens, ms))
}

fn parse_openai_sse(frame: &[u8]) -> Option<String> {
    let text = std::str::from_utf8(frame).ok()?;
    for line in text.lines() {
        let line = line.trim();
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data == "[DONE]" {
            return None;
        }
        let v: serde_json::Value = serde_json::from_str(data).ok()?;
        let content = v
            .pointer("/choices/0/delta/content")
            .and_then(|c| c.as_str())?;
        if content.is_empty() {
            return None;
        }
        return Some(content.to_string());
    }
    None
}
