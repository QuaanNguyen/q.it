use serde_json::Value;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::serve::RuntimeRecipe;

const TRANSFORMERS_PACK_ID: &str = "transformers";
const WORKER_PROTOCOL_VERSION: u32 = 1;

struct ModelPackageRecipe {
    family: &'static str,
    name: &'static str,
    tokenizer_variants: &'static [&'static [&'static str]],
    planner_hint: PlannerHint,
    capabilities: PackageCapabilities,
    runtime_compatibility: RuntimeCompatibility,
}

impl ModelPackageRecipe {
    fn for_model_type(model_type: &str) -> Option<Self> {
        match model_type {
            "qwen3_5" => Some(Self {
                family: "Qwen 3.5",
                name: "Qwen3.5 0.8B",
                tokenizer_variants: &[
                    &["tokenizer.json", "tokenizer_config.json"],
                    &["vocab.json", "merges.txt", "tokenizer_config.json"],
                ],
                planner_hint: PlannerHint {
                    estimate_bytes: 1_200_000_000,
                    source: "qit_catalog",
                    confidence: "high",
                },
                capabilities: text_chat_capabilities(),
                runtime_compatibility: RuntimeCompatibility {
                    pack_id: TRANSFORMERS_PACK_ID,
                    protocol_version: WORKER_PROTOCOL_VERSION,
                    runtime_recipe: RuntimeRecipe::TransformersExternal,
                },
            }),
            "gemma4" => Some(Self {
                family: "Gemma 4",
                name: "Gemma 4 E2B it QAT Mobile",
                tokenizer_variants: &[
                    &["tokenizer.json", "tokenizer_config.json"],
                    &["tokenizer.model", "tokenizer_config.json"],
                ],
                planner_hint: PlannerHint {
                    estimate_bytes: 3_200_000_000,
                    source: "qit_catalog",
                    confidence: "high",
                },
                capabilities: text_chat_capabilities(),
                runtime_compatibility: RuntimeCompatibility {
                    pack_id: TRANSFORMERS_PACK_ID,
                    protocol_version: WORKER_PROTOCOL_VERSION,
                    runtime_recipe: RuntimeRecipe::TransformersExternal,
                },
            }),
            _ => None,
        }
    }

    fn required_files(&self, package_dir: &Path) -> (Vec<PackageFile>, bool) {
        let mut files = vec![local_file(package_dir, "config.json", "config")];
        let tokenizer_files = self
            .tokenizer_variants
            .iter()
            .find(|variant| variant.iter().all(|path| package_dir.join(path).is_file()))
            .copied()
            .unwrap_or(self.tokenizer_variants[0]);
        files.extend(
            tokenizer_files
                .iter()
                .map(|path| local_file(package_dir, path, "tokenizer")),
        );
        let (weight_files, weights_resolved) = weight_files(package_dir);
        files.extend(weight_files);
        (files, weights_resolved)
    }
}

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
}

#[derive(Clone)]
pub struct PlannerHint {
    pub estimate_bytes: u64,
    pub source: &'static str,
    pub confidence: &'static str,
}

#[derive(Clone)]
pub struct PackageCapabilities {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub tasks: Vec<String>,
}

#[derive(Clone)]
pub struct RuntimeCompatibility {
    pub pack_id: &'static str,
    pub protocol_version: u32,
    pub runtime_recipe: RuntimeRecipe,
}

#[derive(Clone)]
pub struct PackageFile {
    pub path: String,
    pub role: &'static str,
    pub bytes: Option<u64>,
    pub sha256: Option<String>,
    pub source: &'static str,
    modified: Option<SystemTime>,
}

#[derive(Clone)]
pub struct CatalogPackage {
    pub id: String,
    pub family: String,
    pub name: String,
    pub format: PackageFormat,
    pub planner_hint: PlannerHint,
    pub capabilities: PackageCapabilities,
    pub runtime_compatibility: RuntimeCompatibility,
    pub package_dir: PathBuf,
    pub artifact_id: Option<String>,
    pub required_files: Vec<PackageFile>,
    required_files_resolved: bool,
}

impl CatalogPackage {
    pub fn required_files_match_scan(&self) -> bool {
        self.required_files_resolved
            && self.required_files.iter().all(|file| {
                let path = self.package_dir.join(&file.path);
                if !path.is_file() {
                    return false;
                }
                if let Some(expected_bytes) = file.bytes {
                    let Ok(metadata) = std::fs::metadata(&path) else {
                        return false;
                    };
                    if metadata.len() != expected_bytes {
                        return false;
                    }
                    let changed_since_scan = file
                        .modified
                        .is_none_or(|expected| metadata.modified().ok() != Some(expected));
                    if changed_since_scan {
                        return false;
                    }
                }
                true
            })
    }

    pub fn primary_artifact_id(&self) -> Option<String> {
        self.artifact_id.clone()
    }

    pub fn runtime_recipe(&self) -> RuntimeRecipe {
        self.runtime_compatibility.runtime_recipe
    }
}

pub fn catalog_package(packages: &[CatalogPackage], id: &str) -> Option<CatalogPackage> {
    packages.iter().find(|package| package.id == id).cloned()
}

pub fn catalog_packages(models_dir: &Path) -> Vec<CatalogPackage> {
    discover_transformers_packages(models_dir)
}

fn discover_transformers_packages(models_dir: &Path) -> Vec<CatalogPackage> {
    let root = models_dir
        .parent()
        .unwrap_or(models_dir)
        .join("transformers");
    let Ok(orgs) = std::fs::read_dir(&root) else {
        return Vec::new();
    };
    orgs.flatten()
        .filter(|e| e.path().is_dir())
        .flat_map(|org| {
            let Ok(models) = std::fs::read_dir(org.path()) else {
                return Vec::new();
            };
            models
                .flatten()
                .filter_map(|entry| {
                    transformers_package(&org.file_name().to_string_lossy(), &entry.path())
                })
                .collect()
        })
        .collect()
}

fn transformers_package(org: &str, package_dir: &Path) -> Option<CatalogPackage> {
    let config: Value =
        serde_json::from_slice(&std::fs::read(package_dir.join("config.json")).ok()?).ok()?;
    let model_type = config["model_type"].as_str()?;
    let recipe = ModelPackageRecipe::for_model_type(model_type)?;
    let (required_files, required_files_resolved) = recipe.required_files(package_dir);
    CatalogPackage {
        id: format!("{org}/{}", package_dir.file_name()?.to_string_lossy()),
        family: recipe.family.into(),
        name: recipe.name.into(),
        format: PackageFormat::Transformers,
        planner_hint: recipe.planner_hint,
        capabilities: recipe.capabilities,
        runtime_compatibility: recipe.runtime_compatibility,
        package_dir: package_dir.to_path_buf(),
        artifact_id: None,
        required_files,
        required_files_resolved,
    }
    .into()
}

fn weight_files(package_dir: &Path) -> (Vec<PackageFile>, bool) {
    let index_path = "model.safetensors.index.json";
    if package_dir.join(index_path).is_file() {
        let mut files = vec![local_file(package_dir, index_path, "weights_index")];
        let mut complete_index = false;
        if let Some(index) = std::fs::read(package_dir.join(index_path))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        {
            if let Some(weights) = index["weight_map"].as_object() {
                let mut paths: Vec<&str> = weights.values().filter_map(Value::as_str).collect();
                paths.sort_unstable();
                paths.dedup();
                complete_index =
                    !paths.is_empty() && paths.iter().all(|path| safe_weight_path(path));
                if complete_index {
                    files.extend(
                        paths
                            .into_iter()
                            .map(|path| local_file(package_dir, path, "weights")),
                    );
                }
            }
        }
        return (files, complete_index);
    }
    let mut paths: Vec<String> = std::fs::read_dir(package_dir)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let path = entry.file_name().to_string_lossy().to_string();
            path.ends_with(".safetensors").then_some(path)
        })
        .collect();
    paths.sort_unstable();
    if paths.is_empty() {
        paths.push("model.safetensors".into());
    }
    (
        paths
            .into_iter()
            .map(|path| local_file(package_dir, &path, "weights"))
            .collect(),
        true,
    )
}

fn safe_weight_path(path: &str) -> bool {
    let mut components = Path::new(path).components();
    matches!(components.next(), Some(std::path::Component::Normal(_)))
        && components.next().is_none()
        && path.ends_with(".safetensors")
}

fn local_file(directory: &Path, path: &str, role: &'static str) -> PackageFile {
    let file = directory.join(path);
    let metadata = std::fs::metadata(&file).ok();
    let bytes = metadata.as_ref().map(|metadata| metadata.len());
    let modified = metadata.and_then(|metadata| metadata.modified().ok());
    PackageFile {
        path: path.into(),
        role,
        bytes,
        sha256: None,
        source: "local_scan",
        modified,
    }
}

fn text_chat_capabilities() -> PackageCapabilities {
    PackageCapabilities {
        inputs: vec!["text".into()],
        outputs: vec!["text".into()],
        tasks: vec!["chat".into()],
    }
}

#[cfg(test)]
mod tests {
    use super::local_file;

    #[test]
    fn local_file_metadata_does_not_hash_contents_during_cataloging() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("weights.safetensors");
        std::fs::write(&path, "weights").unwrap();
        let file = local_file(temp.path(), "weights.safetensors", "weights");
        assert_eq!(file.bytes, Some(7));
        assert_eq!(file.sha256, None);
    }
}
