use serde_json::Value;
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

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
    pub runtime_recipe: RuntimeRecipe,
    pub package_dir: PathBuf,
    pub artifact_id: Option<String>,
    pub required_files: Vec<PackageFile>,
}

impl CatalogPackage {
    pub fn required_files_match_scan(&self) -> bool {
        self.required_files.iter().all(|file| {
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
}

pub fn catalog_package(packages: &[CatalogPackage], id: &str) -> Option<CatalogPackage> {
    packages.iter().find(|package| package.id == id).cloned()
}

pub fn catalog_packages(models_dir: &Path) -> Vec<CatalogPackage> {
    let mut packages = vec![known_gguf_package(models_dir)];
    packages.extend(discover_transformers_packages(models_dir));
    packages
}

fn known_gguf_package(models_dir: &Path) -> CatalogPackage {
    CatalogPackage {
        id: "qit/qwen2.5-0.5b-instruct-q4_k_m".into(),
        family: "Qwen 2.5".into(),
        name: "Qwen2.5 0.5B Instruct Q4_K_M".into(),
        format: PackageFormat::Gguf,
        planner_hint: PlannerHint {
            estimate_bytes: 850_000_000,
            source: "qit_catalog",
            confidence: "high",
        },
        capabilities: text_chat_capabilities(),
        runtime_recipe: RuntimeRecipe::LlamaCpp,
        package_dir: models_dir.join("Qwen"),
        artifact_id: Some("Qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf".into()),
        required_files: vec![local_file(
            &models_dir.join("Qwen"),
            "qwen2.5-0.5b-instruct-q4_k_m.gguf",
            "weights",
        )],
    }
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
    let (family, name) = match model_type {
        "qwen2" if package_dir.file_name()?.to_string_lossy() == "Qwen2.5-0.5B-Instruct" => {
            ("Qwen 2.5", "Qwen2.5 0.5B Instruct")
        }
        "qwen2" => return None,
        "qwen3_5" => ("Qwen 3.5", "Qwen3.5 0.8B"),
        "gemma4" => ("Gemma 4", "Gemma 4 E2B it QAT Mobile"),
        _ => return None,
    };
    if !package_dir.join("tokenizer.json").is_file()
        || !package_dir.join("tokenizer_config.json").is_file()
        || !package_dir.join("README.md").is_file()
    {
        return None;
    }
    let mut required_files = vec![
        local_file(package_dir, "config.json", "config"),
        local_file(package_dir, "tokenizer.json", "tokenizer"),
        local_file(package_dir, "tokenizer_config.json", "tokenizer"),
        local_file(package_dir, "README.md", "model_card"),
    ];
    for entry in std::fs::read_dir(package_dir).ok()?.flatten() {
        let file = entry.file_name().to_string_lossy().to_string();
        if file.ends_with(".safetensors") || file == "model.safetensors.index.json" {
            required_files.push(local_file(package_dir, &file, "weights"));
        }
    }
    if let Some(index) = std::fs::read(package_dir.join("model.safetensors.index.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
    {
        if let Some(weights) = index["weight_map"].as_object() {
            for file in weights.values().filter_map(Value::as_str) {
                if !required_files.iter().any(|item| item.path == file) {
                    required_files.push(local_file(package_dir, file, "weights"));
                }
            }
        }
    }
    CatalogPackage {
        id: format!("{org}/{}", package_dir.file_name()?.to_string_lossy()),
        family: family.into(),
        name: name.into(),
        format: PackageFormat::Transformers,
        planner_hint: PlannerHint {
            estimate_bytes: 1_200_000_000,
            source: "qit_catalog",
            confidence: "high",
        },
        capabilities: text_chat_capabilities(),
        runtime_recipe: RuntimeRecipe::TransformersExternal,
        package_dir: package_dir.to_path_buf(),
        artifact_id: None,
        required_files,
    }
    .into()
}

fn local_file(directory: &Path, path: &str, role: &'static str) -> PackageFile {
    let file = directory.join(path);
    let metadata = std::fs::metadata(&file).ok();
    let bytes = metadata.as_ref().map(|metadata| metadata.len());
    let sha256 = bytes.and_then(|_| sha256_file(&file));
    let modified = metadata.and_then(|metadata| metadata.modified().ok());
    PackageFile {
        path: path.into(),
        role,
        bytes,
        sha256,
        source: "local_scan",
        modified,
    }
}

fn sha256_file(path: &Path) -> Option<String> {
    let mut file = std::fs::File::open(path).ok()?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let bytes = file.read(&mut buffer).ok()?;
        if bytes == 0 {
            return Some(hex::encode(hasher.finalize()));
        }
        hasher.update(&buffer[..bytes]);
    }
}

fn text_chat_capabilities() -> PackageCapabilities {
    PackageCapabilities {
        inputs: vec!["text".into()],
        outputs: vec!["text".into()],
        tasks: vec!["chat".into()],
    }
}
