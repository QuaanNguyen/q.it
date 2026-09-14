use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::config::{DEFAULT_N_CTX, DEFAULT_N_GPU_LAYERS, DEFAULT_N_PARALLEL};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeRecipe {
    LlamaCpp,
    TransformersExternal,
}

impl RuntimeRecipe {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LlamaCpp => "llama_cpp",
            Self::TransformersExternal => "transformers_external",
        }
    }
    pub fn parse(value: &str) -> Self {
        match value {
            "transformers_external" => Self::TransformersExternal,
            _ => Self::LlamaCpp,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ServeProfile {
    #[serde(default = "default_context_length")]
    pub context_length: u32,
    #[serde(default = "empty_runtime_settings")]
    pub runtime_settings: Value,
}

impl ServeProfile {
    pub fn llama_cpp(context_length: u32, gpu_layers: i32, parallel: u32) -> Self {
        Self {
            context_length,
            runtime_settings: serde_json::json!({ "gpu_layers": gpu_layers, "parallel": parallel }),
        }
    }
    pub fn llama_cpp_settings(&self) -> Result<LlamaCppSettings, String> {
        serde_json::from_value(self.runtime_settings.clone())
            .map_err(|error| format!("invalid llama_cpp runtime settings: {error}"))
    }
}

impl Default for ServeProfile {
    fn default() -> Self {
        Self::llama_cpp(DEFAULT_N_CTX, DEFAULT_N_GPU_LAYERS, DEFAULT_N_PARALLEL)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LlamaCppSettings {
    #[serde(default = "default_gpu_layers")]
    pub gpu_layers: i32,
    #[serde(default = "default_parallel")]
    pub parallel: u32,
}

fn default_context_length() -> u32 {
    DEFAULT_N_CTX
}
fn default_gpu_layers() -> i32 {
    DEFAULT_N_GPU_LAYERS
}
fn default_parallel() -> u32 {
    DEFAULT_N_PARALLEL
}
fn empty_runtime_settings() -> Value {
    Value::Object(Map::new())
}
