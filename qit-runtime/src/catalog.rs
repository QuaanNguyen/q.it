use std::collections::{BTreeSet, VecDeque};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use serde_json::Value;
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
    let mut directories = VecDeque::from([root.clone()]);
    let mut packages = Vec::new();
    while let Some(directory) = directories.pop_front() {
        let Ok(entries) = std::fs::read_dir(&directory) else {
            continue;
        };
        let mut has_config = false;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                directories.push_back(path);
            } else if entry.file_name() == "config.json" {
                has_config = true;
            }
        }
        if has_config {
            if let Some(package) = discover_transformers_package(&root, &directory) {
                packages.push(package);
            }
        }
    }
    packages.sort_by(|left, right| left.id.cmp(&right.id));
    packages
}

fn discover_transformers_package(root: &Path, directory: &Path) -> Option<CatalogPackage> {
    let config: Value =
        serde_json::from_slice(&std::fs::read(directory.join("config.json")).ok()?).ok()?;
    if !supports_text_chat(&config) {
        return None;
    }
    let id = directory
        .strip_prefix(root)
        .ok()?
        .to_string_lossy()
        .replace('\\', "/");
    if id.is_empty() {
        return None;
    }
    let mut required_files = package_files(directory);
    required_files.sort_by(|left, right| left.path.cmp(&right.path));
    let model_type = config
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let (family, name, planner_hint) =
        normalized_metadata(&id, model_type, &config, &required_files);
    Some(CatalogPackage {
        id,
        family,
        name,
        format: PackageFormat::Transformers,
        planner_hint,
        capabilities: text_chat_capabilities(),
        runtime_recipe: RuntimeRecipe::TransformersExternal,
        package_dir: directory.to_path_buf(),
        artifact_id: None,
        required_files,
    })
}

fn supports_text_chat(config: &Value) -> bool {
    if config
        .get("model_type")
        .and_then(Value::as_str)
        .is_some_and(|model_type| model_type.starts_with("qwen"))
    {
        return true;
    }
    config
        .get("architectures")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .any(|architecture| architecture.ends_with("ForCausalLM"))
}

fn package_files(directory: &Path) -> Vec<PackageFile> {
    let mut files = vec![local_file(directory, "config.json", "config")];
    files.extend(
        tokenizer_paths(directory)
            .into_iter()
            .map(|path| local_file(directory, path, "tokenizer")),
    );
    if directory.join("tokenizer_config.json").is_file() {
        files.push(local_file(directory, "tokenizer_config.json", "tokenizer"));
    }
    let model_card = if directory.join("README.md").is_file() {
        "README.md"
    } else if directory.join("modelcard.md").is_file() {
        "modelcard.md"
    } else {
        "README.md"
    };
    files.push(local_file(directory, model_card, "model_card"));
    let weight_paths = indexed_weight_paths(directory).unwrap_or_else(|| {
        let mut weights = Vec::new();
        if let Ok(entries) = std::fs::read_dir(directory) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|extension| extension.to_str()) == Some("safetensors")
                {
                    weights.push(entry.file_name().to_string_lossy().into_owned());
                }
            }
        }
        weights
    });
    if weight_paths.is_empty() {
        files.push(local_file(directory, "model.safetensors", "weights"));
    } else {
        files.extend(
            weight_paths
                .into_iter()
                .map(|path| local_file(directory, &path, "weights")),
        );
    }
    if directory.join("model.safetensors.index.json").is_file() {
        files.push(local_file(
            directory,
            "model.safetensors.index.json",
            "weights_index",
        ));
    }
    files
}

fn tokenizer_paths(directory: &Path) -> Vec<&'static str> {
    if directory.join("tokenizer.json").is_file() {
        vec!["tokenizer.json"]
    } else if directory.join("tokenizer.model").is_file() {
        vec!["tokenizer.model"]
    } else if directory.join("vocab.json").is_file() && directory.join("merges.txt").is_file() {
        vec!["vocab.json", "merges.txt"]
    } else {
        vec!["tokenizer.json"]
    }
}

fn indexed_weight_paths(directory: &Path) -> Option<Vec<String>> {
    let index: Value = serde_json::from_slice(
        &std::fs::read(directory.join("model.safetensors.index.json")).ok()?,
    )
    .ok()?;
    let paths = index
        .get("weight_map")?
        .as_object()?
        .values()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    Some(paths)
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

fn normalized_metadata(
    id: &str,
    model_type: &str,
    config: &Value,
    files: &[PackageFile],
) -> (String, String, PlannerHint) {
    if id == "Qwen/Qwen2.5-0.5B-Instruct" {
        return (
            "Qwen 2.5".into(),
            "Qwen2.5 0.5B Instruct".into(),
            PlannerHint {
                estimate_bytes: 1_200_000_000,
                source: "qit_catalog",
                confidence: "high",
            },
        );
    }
    let family = if model_type.starts_with("qwen") {
        "Qwen".into()
    } else {
        uppercase_first(id.split('/').next().unwrap_or("Local"))
    };
    let name = id.rsplit('/').next().unwrap_or(id).into();
    let weight_bytes = files
        .iter()
        .filter(|file| file.role == "weights")
        .filter_map(|file| file.bytes)
        .sum::<u64>();
    let planner_hint = if weight_bytes > 0 {
        PlannerHint {
            estimate_bytes: weight_bytes.saturating_mul(5) / 4,
            source: "local_files",
            confidence: "medium",
        }
    } else if let Some(estimate_bytes) = config_architecture_estimate(config) {
        PlannerHint {
            estimate_bytes,
            source: "config_architecture",
            confidence: "low",
        }
    } else {
        PlannerHint {
            estimate_bytes: 0,
            source: "local_files",
            confidence: "low",
        }
    };
    (family, name, planner_hint)
}

fn config_architecture_estimate(config: &Value) -> Option<u64> {
    let hidden = config_number(config, "hidden_size")?;
    let layers = config_number(config, "num_hidden_layers")?;
    let intermediate = config_number(config, "intermediate_size")?;
    let vocab = config_number(config, "vocab_size")?;
    let heads = config_number(config, "num_attention_heads")?;
    let head_dim = config_number(config, "head_dim").or_else(|| hidden.checked_div(heads))?;
    let kv_heads = config_number(config, "num_key_value_heads").unwrap_or(heads);
    let attention = hidden.checked_mul(hidden)?.checked_mul(2)?.checked_add(
        hidden
            .checked_mul(head_dim)?
            .checked_mul(kv_heads)?
            .checked_mul(2)?,
    )?;
    let feed_forward = hidden.checked_mul(intermediate)?.checked_mul(3)?;
    let layer = attention
        .checked_add(feed_forward)?
        .checked_add(hidden.checked_mul(2)?)?;
    let embedding_copies = if config
        .get("tie_word_embeddings")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        1
    } else {
        2
    };
    let parameters = layers
        .checked_mul(layer)?
        .checked_add(vocab.checked_mul(hidden)?.checked_mul(embedding_copies)?)?;
    parameters.checked_mul(2)?.checked_mul(5)?.checked_div(4)
}

fn config_number(config: &Value, name: &str) -> Option<u64> {
    config.get(name)?.as_u64().filter(|value| *value > 0)
}

fn text_chat_capabilities() -> PackageCapabilities {
    PackageCapabilities {
        inputs: vec!["text".into()],
        outputs: vec!["text".into()],
        tasks: vec!["chat".into()],
    }
}

fn uppercase_first(value: &str) -> String {
    let mut characters = value.chars();
    match characters.next() {
        Some(first) => first.to_uppercase().chain(characters).collect(),
        None => String::new(),
    }
}
