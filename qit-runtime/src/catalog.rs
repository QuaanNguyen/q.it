use std::path::Path;

use crate::serve::RuntimeRecipe;

#[derive(Clone, Copy)]
pub enum PackageFormat {
    Gguf,
    Transformers,
}

impl PackageFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gguf => "gguf",
            Self::Transformers => "transformers",
        }
    }
    fn library_dir(self, models_dir: &Path) -> std::path::PathBuf {
        match self {
            Self::Gguf => models_dir.to_path_buf(),
            Self::Transformers => models_dir
                .parent()
                .unwrap_or(models_dir)
                .join("transformers"),
        }
    }
}

pub struct PlannerHint {
    pub estimate_bytes: u64,
    pub source: &'static str,
    pub confidence: &'static str,
}
pub struct PackageFile {
    pub path: &'static str,
    pub role: &'static str,
}
pub struct OwnedPackage {
    pub id: &'static str,
    pub family: &'static str,
    pub name: &'static str,
    pub format: PackageFormat,
    pub planner_hint: PlannerHint,
    pub runtime_recipe: RuntimeRecipe,
    local_dir: &'static str,
    pub required_files: &'static [PackageFile],
}

impl OwnedPackage {
    pub fn has_required_files(&self, models_dir: &Path) -> bool {
        let package_dir = self.format.library_dir(models_dir).join(self.local_dir);
        self.required_files
            .iter()
            .all(|file| package_dir.join(file.path).is_file())
    }
    pub fn primary_artifact_id(&self) -> Option<String> {
        self.required_files
            .first()
            .map(|file| format!("{}/{}", self.local_dir, file.path))
    }
}

pub fn owned_package(id: &str) -> Option<OwnedPackage> {
    owned_packages()
        .into_iter()
        .find(|package| package.id == id)
}

pub fn owned_packages() -> [OwnedPackage; 2] {
    [
        OwnedPackage {
            id: "qit/qwen2.5-0.5b-instruct-q4_k_m",
            family: "Qwen 2.5",
            name: "Qwen2.5 0.5B Instruct Q4_K_M",
            format: PackageFormat::Gguf,
            planner_hint: PlannerHint {
                estimate_bytes: 850_000_000,
                source: "qit_catalog",
                confidence: "high",
            },
            runtime_recipe: RuntimeRecipe::LlamaCpp,
            local_dir: "Qwen",
            required_files: &[PackageFile {
                path: "qwen2.5-0.5b-instruct-q4_k_m.gguf",
                role: "weights",
            }],
        },
        OwnedPackage {
            id: "Qwen/Qwen2.5-0.5B-Instruct",
            family: "Qwen 2.5",
            name: "Qwen2.5 0.5B Instruct",
            format: PackageFormat::Transformers,
            planner_hint: PlannerHint {
                estimate_bytes: 1_200_000_000,
                source: "qit_catalog",
                confidence: "high",
            },
            runtime_recipe: RuntimeRecipe::TransformersExternal,
            local_dir: "Qwen/Qwen2.5-0.5B-Instruct",
            required_files: &[
                PackageFile {
                    path: "config.json",
                    role: "config",
                },
                PackageFile {
                    path: "tokenizer.json",
                    role: "tokenizer",
                },
                PackageFile {
                    path: "tokenizer_config.json",
                    role: "tokenizer",
                },
                PackageFile {
                    path: "model.safetensors.index.json",
                    role: "weights_index",
                },
                PackageFile {
                    path: "model-00001-of-00002.safetensors",
                    role: "weights",
                },
                PackageFile {
                    path: "model-00002-of-00002.safetensors",
                    role: "weights",
                },
                PackageFile {
                    path: "README.md",
                    role: "model_card",
                },
            ],
        },
    ]
}
