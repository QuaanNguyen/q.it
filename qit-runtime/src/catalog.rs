use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use sha2::{Digest, Sha256};

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
    vec![
        known_gguf_package(models_dir),
        known_transformers_package(models_dir),
    ]
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

fn known_transformers_package(models_dir: &Path) -> CatalogPackage {
    let package_dir = models_dir
        .parent()
        .unwrap_or(models_dir)
        .join("transformers")
        .join("Qwen")
        .join("Qwen2.5-0.5B-Instruct");
    CatalogPackage {
        id: "Qwen/Qwen2.5-0.5B-Instruct".into(),
        family: "Qwen 2.5".into(),
        name: "Qwen2.5 0.5B Instruct".into(),
        format: PackageFormat::Transformers,
        planner_hint: PlannerHint {
            estimate_bytes: 1_200_000_000,
            source: "qit_catalog",
            confidence: "high",
        },
        capabilities: text_chat_capabilities(),
        runtime_recipe: RuntimeRecipe::TransformersExternal,
        package_dir: package_dir.clone(),
        artifact_id: None,
        required_files: vec![
            local_file(&package_dir, "config.json", "config"),
            local_file(&package_dir, "tokenizer.json", "tokenizer"),
            local_file(&package_dir, "tokenizer_config.json", "tokenizer"),
            local_file(
                &package_dir,
                "model.safetensors.index.json",
                "weights_index",
            ),
            local_file(&package_dir, "model-00001-of-00002.safetensors", "weights"),
            local_file(&package_dir, "model-00002-of-00002.safetensors", "weights"),
            local_file(&package_dir, "README.md", "model_card"),
        ],
    }
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
