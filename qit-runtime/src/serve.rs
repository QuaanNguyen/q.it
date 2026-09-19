use serde::ser::SerializeMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::config::{DEFAULT_N_CTX, DEFAULT_N_GPU_LAYERS, DEFAULT_N_PARALLEL};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TargetIdentity {
    Artifact(String),
    Package {
        id: String,
        artifact_id: Option<String>,
    },
}

impl TargetIdentity {
    pub fn id(&self) -> &str {
        match self {
            Self::Artifact(id) | Self::Package { id, .. } => id,
        }
    }

    pub fn artifact_id(&self) -> Option<&str> {
        match self {
            Self::Artifact(id) => Some(id),
            Self::Package { artifact_id, .. } => artifact_id.as_deref(),
        }
    }

    pub fn package_id(&self) -> Option<&str> {
        match self {
            Self::Artifact(_) => None,
            Self::Package { id, .. } => Some(id),
        }
    }
}

impl Serialize for TargetIdentity {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("target_id", self.id())?;
        if let Some(artifact_id) = self.artifact_id() {
            map.serialize_entry("artifact_id", artifact_id)?;
        }
        if let Some(package_id) = self.package_id() {
            map.serialize_entry("package_id", package_id)?;
        }
        map.end()
    }
}

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
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "llama_cpp" => Ok(Self::LlamaCpp),
            "transformers_external" => Ok(Self::TransformersExternal),
            _ => Err(format!("unknown runtime recipe: {value}")),
        }
    }

    pub(crate) fn profile(
        self,
        profile: Option<ServeProfile>,
        context_length: u32,
        gpu_layers: i32,
        parallel: u32,
    ) -> Result<ServeProfile, String> {
        let mut profile = profile.unwrap_or_else(|| match self {
            Self::LlamaCpp => ServeProfile::llama_cpp(context_length, gpu_layers, parallel),
            Self::TransformersExternal => ServeProfile::transformers_external(context_length),
        });
        match self {
            Self::LlamaCpp => {
                let settings = profile.llama_cpp_settings()?;
                if settings.parallel == 0 {
                    return Err("parallel must be at least 1".into());
                }
                profile.runtime_settings =
                    serde_json::to_value(settings).map_err(|error| error.to_string())?;
            }
            Self::TransformersExternal => {
                profile.transformers_external_settings()?;
                profile.runtime_settings = empty_runtime_settings();
            }
        }
        Ok(profile)
    }

    pub(crate) fn launch_parameters(self, profile: &ServeProfile) -> Result<(i32, u32), String> {
        match self {
            Self::LlamaCpp => {
                let settings = profile.llama_cpp_settings()?;
                Ok((settings.gpu_layers, settings.parallel))
            }
            Self::TransformersExternal => {
                profile.transformers_external_settings()?;
                Ok((0, 1))
            }
        }
    }

    pub fn requires_artifact(self) -> bool {
        self == Self::LlamaCpp
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
    pub fn transformers_external(context_length: u32) -> Self {
        Self {
            context_length,
            runtime_settings: empty_runtime_settings(),
        }
    }
    pub fn transformers_external_settings(&self) -> Result<TransformersExternalSettings, String> {
        serde_json::from_value(self.runtime_settings.clone())
            .map_err(|error| format!("invalid transformers_external runtime settings: {error}"))
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

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TransformersExternalSettings {}

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
