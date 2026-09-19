use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::serve::{RuntimeRecipe, ServeProfile};
use crate::supervisor::{
    launch_worker, proxy_openai_chat_completions, ChatMessage, GenerateOutcome, LaunchRequest,
    LaunchedWorker, WorkerLauncher,
};

#[derive(Clone)]
enum RuntimeLaunch {
    Executable {
        path: PathBuf,
        extra_args: Vec<String>,
    },
    Injected {
        launcher: Arc<dyn WorkerLauncher>,
        path: Option<PathBuf>,
    },
    Unavailable,
}

#[derive(Clone)]
pub struct RuntimeAdapter {
    recipe: RuntimeRecipe,
    launch: RuntimeLaunch,
    diagnostic: String,
}

impl RuntimeAdapter {
    pub fn executable(recipe: RuntimeRecipe, path: PathBuf, extra_args: Vec<String>) -> Self {
        Self {
            recipe,
            launch: RuntimeLaunch::Executable { path, extra_args },
            diagnostic: unavailable_diagnostic(recipe),
        }
    }

    pub fn injected(
        recipe: RuntimeRecipe,
        launcher: Arc<dyn WorkerLauncher>,
        diagnostic_path: Option<PathBuf>,
    ) -> Self {
        Self {
            recipe,
            launch: RuntimeLaunch::Injected {
                launcher,
                path: diagnostic_path.clone(),
            },
            diagnostic: diagnostic_path
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "injected test runtime".into()),
        }
    }

    pub fn unavailable(recipe: RuntimeRecipe) -> Self {
        Self {
            recipe,
            launch: RuntimeLaunch::Unavailable,
            diagnostic: unavailable_diagnostic(recipe),
        }
    }

    pub fn recipe(&self) -> RuntimeRecipe {
        self.recipe
    }

    pub fn is_available(&self) -> bool {
        match &self.launch {
            RuntimeLaunch::Executable { path, .. } => executable_file(path),
            RuntimeLaunch::Injected { .. } => true,
            RuntimeLaunch::Unavailable => false,
        }
    }

    pub fn executable_path(&self) -> Option<&Path> {
        match &self.launch {
            RuntimeLaunch::Executable { path, .. } => Some(path),
            RuntimeLaunch::Injected { path, .. } => path.as_deref(),
            RuntimeLaunch::Unavailable => None,
        }
    }

    pub fn unavailable_diagnostic(&self) -> &str {
        &self.diagnostic
    }

    pub fn profile(
        &self,
        profile: Option<ServeProfile>,
        context_length: u32,
        gpu_layers: i32,
        parallel: u32,
    ) -> Result<ServeProfile, String> {
        let mut profile = profile.unwrap_or_else(|| match self.recipe {
            RuntimeRecipe::LlamaCpp => {
                ServeProfile::llama_cpp(context_length, gpu_layers, parallel)
            }
            RuntimeRecipe::TransformersExternal => {
                ServeProfile::transformers_external(context_length)
            }
        });
        match self.recipe {
            RuntimeRecipe::LlamaCpp => {
                let settings = profile.llama_cpp_settings()?;
                if settings.parallel == 0 {
                    return Err("parallel must be at least 1".into());
                }
                profile.runtime_settings =
                    serde_json::to_value(settings).map_err(|error| error.to_string())?;
            }
            RuntimeRecipe::TransformersExternal => {
                profile.transformers_external_settings()?;
                profile.runtime_settings = serde_json::Value::Object(serde_json::Map::new());
            }
        }
        Ok(profile)
    }

    pub fn launch_parameters(&self, profile: &ServeProfile) -> Result<(i32, u32), String> {
        match self.recipe {
            RuntimeRecipe::LlamaCpp => {
                let settings = profile.llama_cpp_settings()?;
                Ok((settings.gpu_layers, settings.parallel))
            }
            RuntimeRecipe::TransformersExternal => {
                profile.transformers_external_settings()?;
                Ok((0, 1))
            }
        }
    }

    pub fn launch(&self, request: LaunchRequest) -> Result<LaunchedWorker, String> {
        if !self.is_available() {
            return Err(self.diagnostic.clone());
        }
        match &self.launch {
            RuntimeLaunch::Executable { path, extra_args } => {
                self.launch_executable(path, extra_args, request)
            }
            RuntimeLaunch::Injected { launcher, .. } => launcher.launch(request),
            RuntimeLaunch::Unavailable => Err(self.diagnostic.clone()),
        }
    }

    pub fn health_url(&self, base_url: &str) -> String {
        format!("{base_url}/health")
    }

    pub fn generation_url(&self, base_url: &str) -> String {
        format!("{base_url}/v1/chat/completions")
    }

    pub async fn generate<F, Fut>(
        &self,
        base_url: &str,
        messages: &[ChatMessage],
        max_tokens: u32,
        cancel: tokio::sync::watch::Receiver<bool>,
        on_token: F,
    ) -> Result<GenerateOutcome, String>
    where
        F: FnMut(String) -> Fut,
        Fut: std::future::Future<Output = Result<(), String>>,
    {
        proxy_openai_chat_completions(
            &self.generation_url(base_url),
            messages,
            max_tokens,
            cancel,
            on_token,
        )
        .await
    }

    fn launch_executable(
        &self,
        path: &Path,
        extra_args: &[String],
        request: LaunchRequest,
    ) -> Result<LaunchedWorker, String> {
        match self.recipe {
            RuntimeRecipe::LlamaCpp => launch_worker(
                path,
                request,
                "llama.cpp worker",
                |command, port, request| {
                    command
                        .arg("--host")
                        .arg("127.0.0.1")
                        .arg("--port")
                        .arg(port.to_string())
                        .arg("-m")
                        .arg(&request.target_path)
                        .arg("-c")
                        .arg(request.n_ctx.to_string())
                        .arg("-ngl")
                        .arg(request.n_gpu_layers.to_string())
                        .arg("--parallel")
                        .arg(request.n_parallel.to_string());
                },
            ),
            RuntimeRecipe::TransformersExternal => launch_worker(
                path,
                request,
                "Transformers worker",
                |command, port, request| {
                    command
                        .args(extra_args)
                        .arg("--host")
                        .arg("127.0.0.1")
                        .arg("--port")
                        .arg(port.to_string())
                        .arg("--model")
                        .arg(&request.target_path)
                        .arg("--context-length")
                        .arg(request.n_ctx.to_string());
                },
            ),
        }
    }
}

#[derive(Clone)]
pub struct RuntimeRegistry {
    llama_cpp: Arc<RuntimeAdapter>,
    transformers_external: Arc<RuntimeAdapter>,
}

impl RuntimeRegistry {
    pub fn new(llama_cpp: RuntimeAdapter, transformers_external: RuntimeAdapter) -> Self {
        Self {
            llama_cpp: Arc::new(llama_cpp),
            transformers_external: Arc::new(transformers_external),
        }
    }

    pub fn resolve(&self, recipe: RuntimeRecipe) -> Arc<RuntimeAdapter> {
        match recipe {
            RuntimeRecipe::LlamaCpp => self.llama_cpp.clone(),
            RuntimeRecipe::TransformersExternal => self.transformers_external.clone(),
        }
    }

    pub fn with_adapter(mut self, adapter: RuntimeAdapter) -> Self {
        match adapter.recipe() {
            RuntimeRecipe::LlamaCpp => self.llama_cpp = Arc::new(adapter),
            RuntimeRecipe::TransformersExternal => self.transformers_external = Arc::new(adapter),
        }
        self
    }
}

fn executable_file(path: &Path) -> bool {
    let Ok(metadata) = std::fs::metadata(path) else {
        return false;
    };
    if !metadata.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        metadata.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

fn unavailable_diagnostic(recipe: RuntimeRecipe) -> String {
    match recipe {
        RuntimeRecipe::LlamaCpp => {
            "no llama.cpp runtime (set QIT_WORKER_PATH or LLAMA_SERVER_PATH)".into()
        }
        RuntimeRecipe::TransformersExternal => {
            "no Transformers runtime (set QIT_TRANSFORMERS_WORKER_PATH)".into()
        }
    }
}
