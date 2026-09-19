use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use serde::Deserialize;

use crate::probe::{FixedProbe, HardwareProbe, SystemProbe};
use crate::runtime::{RuntimeAdapter, RuntimeRegistry};
use crate::serve::{RuntimeRecipe, ServeProfile};
use crate::supervisor::WorkerLauncher;

pub const DEFAULT_HOST: &str = "127.0.0.1";
pub const DEFAULT_PORT: u16 = 2471;
pub const DEFAULT_N_CTX: u32 = 4096;
pub const DEFAULT_N_GPU_LAYERS: i32 = 999;
pub const DEFAULT_N_PARALLEL: u32 = 1;
pub const DEFAULT_OS_RESERVE_FRACTION: f64 = 0.25;

#[derive(Clone)]
pub struct Config {
    pub listen: SocketAddr,
    pub home: PathBuf,
    pub models_dir: PathBuf,
    pub os_reserve_bytes: Option<u64>,
    pub runtimes: RuntimeRegistry,
    pub probe: Arc<dyn HardwareProbe>,
}

impl Config {
    pub fn from_env() -> Result<Self, crate::error::Error> {
        let home = std::env::var("QIT_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| default_home());
        let models_dir = std::env::var("QIT_MODELS_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| home.join("models").join("gguf"));
        let port = std::env::var("QIT_PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(DEFAULT_PORT);
        let listen = SocketAddr::from(([127, 0, 0, 1], port));
        let os_reserve_bytes = std::env::var("QIT_OS_RESERVE_BYTES")
            .ok()
            .and_then(|v| v.parse().ok());
        let llama_cpp = resolve_worker_path()
            .map(|path| RuntimeAdapter::executable(RuntimeRecipe::LlamaCpp, path, Vec::new()))
            .unwrap_or_else(|| RuntimeAdapter::unavailable(RuntimeRecipe::LlamaCpp));
        let transformers_external = resolve_transformers_worker_path()
            .map(|path| {
                RuntimeAdapter::executable(RuntimeRecipe::TransformersExternal, path, Vec::new())
            })
            .unwrap_or_else(|| RuntimeAdapter::unavailable(RuntimeRecipe::TransformersExternal));
        Ok(Self {
            listen,
            home,
            models_dir,
            os_reserve_bytes,
            runtimes: RuntimeRegistry::new(llama_cpp, transformers_external),
            probe: Arc::new(SystemProbe),
        })
    }

    pub fn test(
        home: PathBuf,
        models_dir: PathBuf,
        listen: SocketAddr,
        probe: FixedProbe,
        worker_path: Option<PathBuf>,
        worker_launcher: Arc<dyn WorkerLauncher>,
        os_reserve_bytes: Option<u64>,
    ) -> Self {
        Self {
            listen,
            home,
            models_dir,
            os_reserve_bytes,
            runtimes: RuntimeRegistry::new(
                RuntimeAdapter::injected(RuntimeRecipe::LlamaCpp, worker_launcher, worker_path),
                RuntimeAdapter::unavailable(RuntimeRecipe::TransformersExternal),
            ),
            probe: Arc::new(probe),
        }
    }

    pub fn with_runtime_adapter(mut self, adapter: RuntimeAdapter) -> Self {
        self.runtimes = self.runtimes.with_adapter(adapter);
        self
    }
}

pub fn resolve_worker_path() -> Option<PathBuf> {
    std::env::var("QIT_WORKER_PATH")
        .ok()
        .or_else(|| std::env::var("LLAMA_SERVER_PATH").ok())
        .map(PathBuf::from)
        .or_else(bundled_worker_path)
        .or_else(homebrew_worker_path)
}

pub fn resolve_transformers_worker_path() -> Option<PathBuf> {
    std::env::var("QIT_TRANSFORMERS_WORKER_PATH")
        .ok()
        .map(PathBuf::from)
}

fn homebrew_worker_path() -> Option<PathBuf> {
    [
        PathBuf::from("/opt/homebrew/bin/llama-server"),
        PathBuf::from("/usr/local/bin/llama-server"),
    ]
    .into_iter()
    .find(|candidate| candidate.is_file())
}

fn default_home() -> PathBuf {
    dirs_home()
        .map(|h| h.join("Library").join("Application Support").join("q.it"))
        .unwrap_or_else(|| PathBuf::from("q.it-data"))
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

fn bundled_worker_path() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    let candidate = dir.join("llama-server");
    candidate.exists().then_some(candidate)
}

#[derive(Clone, Debug, Deserialize)]
pub struct SessionShape {
    pub artifact_id: Option<String>,
    pub package_id: Option<String>,
    pub serve_profile: Option<ServeProfile>,
    pub n_ctx: Option<u32>,
    pub n_gpu_layers: Option<i32>,
    pub n_parallel: Option<u32>,
}
