use std::collections::HashMap;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log_path: Option<String>,
}

pub fn session_status_str(status: SessionStatus) -> &'static str {
    match status {
        SessionStatus::NotLoaded => "not_loaded",
        SessionStatus::Starting => "starting",
        SessionStatus::Loaded => "loaded",
        SessionStatus::Stopping => "stopping",
        SessionStatus::Failed => "failed",
    }
}

pub fn session_status_from_str(s: &str) -> SessionStatus {
    match s {
        "starting" => SessionStatus::Starting,
        "loaded" => SessionStatus::Loaded,
        "stopping" => SessionStatus::Stopping,
        "failed" => SessionStatus::Failed,
        _ => SessionStatus::NotLoaded,
    }
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

    pub async fn hydrate(&self, rows: Vec<crate::store::SessionRow>) {
        let mut guard = self.sessions.lock().await;
        for row in rows {
            guard.insert(
                row.id.clone(),
                LiveSession {
                    view: SessionView {
                        id: row.id,
                        artifact_id: row.artifact_id,
                        n_ctx: row.n_ctx,
                        n_gpu_layers: row.n_gpu_layers,
                        n_parallel: row.n_parallel,
                        status: session_status_from_str(&row.status),
                        last_error: row.last_error,
                        log_path: row.log_path,
                    },
                    child: None,
                    base_url: None,
                },
            );
        }
    }

    pub async fn find_by_tuple(
        &self,
        artifact_id: &str,
        n_ctx: u32,
        n_gpu_layers: i32,
        n_parallel: u32,
    ) -> Option<SessionView> {
        self.sessions.lock().await.values().find_map(|s| {
            if s.view.artifact_id == artifact_id
                && s.view.n_ctx == n_ctx
                && s.view.n_gpu_layers == n_gpu_layers
                && s.view.n_parallel == n_parallel
            {
                Some(s.view.clone())
            } else {
                None
            }
        })
    }

    pub async fn remove(&self, id: &str) -> Result<SessionView, String> {
        let mut guard = self.sessions.lock().await;
        let session = guard
            .get(id)
            .ok_or_else(|| "session not found".to_string())?;
        if session.child.is_some()
            || matches!(
                session.view.status,
                SessionStatus::Loaded | SessionStatus::Starting | SessionStatus::Stopping
            )
        {
            return Err("session is active".into());
        }
        let view = session.view.clone();
        guard.remove(id);
        Ok(view)
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
        let log_path_str = log_path.display().to_string();
        let id = self
            .find_by_tuple(&artifact.id, n_ctx, n_gpu_layers, n_parallel)
            .await
            .map(|s| s.id)
            .unwrap_or_else(|| Uuid::new_v4().to_string());
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
                        last_error: None,
                        log_path: Some(log_path_str),
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
                    s.view.last_error = Some(e.clone());
                }
                return Err(e);
            }
        };
        let mut launched = launched;
        if let Err(e) = wait_ready(&launched.base_url, &mut launched.child).await {
            drop(kill_child(launched.child));
            let mut guard = self.sessions.lock().await;
            if let Some(s) = guard.get_mut(&id) {
                s.view.status = SessionStatus::Failed;
                s.view.last_error = Some(e.clone());
                s.child = None;
                s.base_url = None;
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
            s.view.last_error = None;
            s.base_url = Some(launched.base_url);
            s.child = Some(launched.child);
            return Ok(s.view.clone());
        }
        drop(kill_child(launched.child));
        Err("session vanished".into())
    }

    pub async fn sample_resident(&self, id: &str) -> Option<u64> {
        let guard = self.sessions.lock().await;
        let pid = guard.get(id)?.child.as_ref()?.id();
        resident_bytes(pid)
    }

    pub async fn stop(&self, id: &str) -> Result<Stopped, String> {
        let mut guard = self.sessions.lock().await;
        let session = guard
            .get_mut(id)
            .ok_or_else(|| "session not found".to_string())?;
        session.view.status = SessionStatus::Stopping;
        let peak_rss_bytes = session.child.take().and_then(reap_child);
        session.base_url = None;
        session.view.status = SessionStatus::NotLoaded;
        session.view.last_error = None;
        Ok(Stopped {
            view: session.view.clone(),
            peak_rss_bytes,
        })
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
                            session.view.last_error =
                                Some(format!("worker exited ({status})"));
                            session.base_url = None;
                            session.child = None;
                        }
                    }
                }
            }
        }
    }
}

pub struct Stopped {
    pub view: SessionView,
    pub peak_rss_bytes: Option<u64>,
}

fn kill_child(child: Child) -> Result<(), std::io::Error> {
    let _ = reap_child(child);
    Ok(())
}

fn reap_child(child: Child) -> Option<u64> {
    let pid = child.id();
    let _ = unsafe { libc::kill(pid as i32, libc::SIGKILL) };
    #[cfg(unix)]
    {
        let mut status = 0;
        let mut usage = unsafe { std::mem::zeroed::<libc::rusage>() };
        let rc = unsafe { libc::wait4(pid as libc::pid_t, &mut status, 0, &mut usage) };
        std::mem::forget(child);
        if rc == pid as libc::pid_t && usage.ru_maxrss > 0 {
            Some(usage.ru_maxrss as u64)
        } else {
            None
        }
    }
    #[cfg(not(unix))]
    {
        let _ = child.wait();
        None
    }
}

#[cfg(target_os = "macos")]
fn resident_bytes(pid: u32) -> Option<u64> {
    let mut info = unsafe { std::mem::zeroed::<libc::rusage_info_v4>() };
    let rc = unsafe {
        libc::proc_pid_rusage(
            pid as i32,
            libc::RUSAGE_INFO_V4,
            &mut info as *mut _ as *mut _,
        )
    };
    if rc == 0 {
        Some(info.ri_resident_size)
    } else {
        None
    }
}

#[cfg(not(target_os = "macos"))]
fn resident_bytes(_pid: u32) -> Option<u64> {
    None
}

async fn wait_ready(base_url: &str, child: &mut Child) -> Result<(), String> {
    let client = reqwest::Client::new();
    let url = format!("{base_url}/health");
    let started = std::time::Instant::now();
    while started.elapsed() < Duration::from_secs(60) {
        if let Ok(Some(status)) = child.try_wait() {
            return Err(format!("worker exited ({status})"));
        }
        if let Ok(resp) = client.get(&url).send().await {
            if resp.status().is_success() {
                return Ok(());
            }
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    Err("worker did not become ready".into())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

pub struct GenerateOutcome {
    pub n_tokens: u32,
    pub generation_ms: f64,
    pub usage: Option<Usage>,
}

enum WorkerFrame {
    Token(String),
    Usage(Usage),
    Other,
}

pub async fn proxy_generate<F, Fut>(
    base_url: &str,
    messages: &[ChatMessage],
    max_tokens: u32,
    cancel: tokio::sync::watch::Receiver<bool>,
    mut on_token: F,
) -> Result<GenerateOutcome, String>
where
    F: FnMut(String) -> Fut,
    Fut: std::future::Future<Output = Result<(), String>>,
{
    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "messages": messages,
        "stream": true,
        "stream_options": {"include_usage": true},
        "max_tokens": max_tokens
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
    let mut usage = None;
    let mut buf = bytes::BytesMut::new();
    while let Some(chunk) = resp.chunk().await.map_err(|e| e.to_string())? {
        if *cancel.borrow() {
            break;
        }
        buf.extend_from_slice(&chunk);
        while let Some(pos) = buf.windows(2).position(|w| w == b"\n\n") {
            let frame = buf.split_to(pos + 2);
            match parse_openai_sse(&frame) {
                WorkerFrame::Token(token) => {
                    n_tokens += 1;
                    on_token(token).await?;
                }
                WorkerFrame::Usage(u) => usage = Some(u),
                WorkerFrame::Other => {}
            }
        }
    }
    Ok(GenerateOutcome {
        n_tokens,
        generation_ms: t0.elapsed().as_secs_f64() * 1000.0,
        usage,
    })
}

fn parse_openai_sse(frame: &[u8]) -> WorkerFrame {
    let Ok(text) = std::str::from_utf8(frame) else {
        return WorkerFrame::Other;
    };
    for line in text.lines() {
        let line = line.trim();
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data == "[DONE]" {
            return WorkerFrame::Other;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(data) else {
            return WorkerFrame::Other;
        };
        if let Some(content) = v
            .pointer("/choices/0/delta/content")
            .and_then(|c| c.as_str())
            .filter(|c| !c.is_empty())
        {
            return WorkerFrame::Token(content.to_string());
        }
        if let Some(u) = v.get("usage") {
            let prompt_tokens = u["prompt_tokens"].as_u64().unwrap_or(0) as u32;
            let completion_tokens = u["completion_tokens"].as_u64().unwrap_or(0) as u32;
            return WorkerFrame::Usage(Usage {
                prompt_tokens,
                completion_tokens,
            });
        }
        return WorkerFrame::Other;
    }
    WorkerFrame::Other
}
