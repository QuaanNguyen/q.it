use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::serve::{RuntimeRecipe, ServeProfile};
use crate::supervisor::{
    launch_transformers_external, LaunchRequest, LaunchedWorker, LlamaServerLauncher,
    WorkerLauncher,
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
        let diagnostic = match recipe {
            RuntimeRecipe::LlamaCpp => {
                "no llama.cpp runtime (set QIT_WORKER_PATH or LLAMA_SERVER_PATH)".into()
            }
            RuntimeRecipe::TransformersExternal => {
                "no Transformers runtime (set QIT_TRANSFORMERS_WORKER_PATH)".into()
            }
        };
        Self {
            recipe,
            launch: RuntimeLaunch::Executable { path, extra_args },
            diagnostic,
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
        let diagnostic = match recipe {
            RuntimeRecipe::LlamaCpp => {
                "no llama.cpp runtime (set QIT_WORKER_PATH or LLAMA_SERVER_PATH)".into()
            }
            RuntimeRecipe::TransformersExternal => {
                "no Transformers runtime (set QIT_TRANSFORMERS_WORKER_PATH)".into()
            }
        };
        Self {
            recipe,
            launch: RuntimeLaunch::Unavailable,
            diagnostic,
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
        self.recipe
            .profile(profile, context_length, gpu_layers, parallel)
    }

    pub fn launch_parameters(&self, profile: &ServeProfile) -> Result<(i32, u32), String> {
        self.recipe.launch_parameters(profile)
    }

    pub fn launch(&self, request: LaunchRequest) -> Result<LaunchedWorker, String> {
        if !self.is_available() {
            return Err(self.diagnostic.clone());
        }
        match &self.launch {
            RuntimeLaunch::Executable { path, extra_args } => match self.recipe {
                RuntimeRecipe::LlamaCpp => LlamaServerLauncher {
                    binary: Some(path.clone()),
                }
                .launch(request),
                RuntimeRecipe::TransformersExternal => {
                    launch_transformers_external(path, extra_args, request)
                }
            },
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
