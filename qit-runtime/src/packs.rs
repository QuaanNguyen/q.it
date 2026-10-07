use std::collections::HashSet;
use std::fs;
use std::path::{Component, Path, PathBuf};

use uuid::Uuid;

use crate::model::{BenchmarkCase, BenchmarkDefinition, BenchmarkPackManifest, ProviderKind, Task};

const CORE_MANIFEST: &str = include_str!("../benchmarks/core/pack.json");
const CORE_TEXT_GENERATION: &str = include_str!("../benchmarks/core/text-generation-smoke.jsonl");
const CORE_EMBEDDING: &str = include_str!("../benchmarks/core/embedding-retrieval.jsonl");
const CORE_IMAGE_TO_TEXT: &str = include_str!("../benchmarks/core/image-to-text-smoke.jsonl");
const CORE_SPEECH_TO_TEXT: &str = include_str!("../benchmarks/core/speech-to-text-smoke.jsonl");
const CORE_RERANKING: &str = include_str!("../benchmarks/core/reranking-relevance.jsonl");
const CORE_TEXT_TO_IMAGE: &str = include_str!("../benchmarks/core/text-to-image-smoke.jsonl");

#[derive(Clone)]
pub struct PackCatalog {
    packs_dir: PathBuf,
}

pub struct PackVerification {
    pub id: String,
    pub version: String,
    pub benchmark_count: usize,
    pub case_count: usize,
}

impl PackCatalog {
    pub fn new(packs_dir: PathBuf) -> Self {
        Self { packs_dir }
    }

    pub fn ensure(&self) -> Result<(), String> {
        fs::create_dir_all(&self.packs_dir).map_err(|error| {
            format!(
                "create benchmark pack directory {}: {error}",
                self.packs_dir.display()
            )
        })
    }

    pub fn benchmarks(&self) -> Result<Vec<BenchmarkDefinition>, String> {
        self.ensure()?;
        let mut definitions = load_builtin()?;
        let mut entries = fs::read_dir(&self.packs_dir)
            .map_err(|error| format!("read {}: {error}", self.packs_dir.display()))?
            .filter_map(Result::ok)
            .filter(|entry| entry.path().is_dir())
            .collect::<Vec<_>>();
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            definitions.extend(load_pack_dir(&entry.path())?);
        }
        let mut ids = HashSet::new();
        for definition in &definitions {
            if !ids.insert(definition.id.clone()) {
                return Err(format!(
                    "duplicate benchmark id '{}' across installed packs",
                    definition.id
                ));
            }
        }
        definitions.sort_by(|left, right| left.id.cmp(&right.id));
        Ok(definitions)
    }

    pub fn benchmark(&self, id: &str) -> Result<BenchmarkDefinition, String> {
        let matches = self
            .benchmarks()?
            .into_iter()
            .filter(|definition| {
                definition.id == id || definition.id.rsplit('/').next() == Some(id)
            })
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [definition] => Ok(definition.clone()),
            [] => Err(format!("benchmark '{id}' is not installed")),
            _ => Err(format!(
                "benchmark name '{id}' is ambiguous, use its full pack/id name"
            )),
        }
    }

    pub fn install(&self, source: &Path) -> Result<String, String> {
        self.ensure()?;
        let definitions = load_pack_dir(source)?;
        let pack_id = definitions
            .first()
            .map(|definition| definition.pack_id.clone())
            .ok_or_else(|| "benchmark pack contains no benchmarks".to_string())?;
        let destination = self.packs_dir.join(&pack_id);
        if destination.exists() {
            return Err(format!(
                "benchmark pack '{pack_id}' is already installed; remove it before installing another version"
            ));
        }
        let staging = self.packs_dir.join(format!(".install-{}", Uuid::new_v4()));
        copy_tree(source, &staging)?;
        if let Err(error) = fs::rename(&staging, &destination) {
            let _ = fs::remove_dir_all(&staging);
            return Err(format!(
                "activate benchmark pack at {}: {error}",
                destination.display()
            ));
        }
        Ok(pack_id)
    }

    pub fn verify(&self, source: &Path) -> Result<PackVerification, String> {
        let definitions = load_pack_dir(source)?;
        let first = definitions
            .first()
            .ok_or_else(|| "benchmark pack contains no benchmarks".to_string())?;
        Ok(PackVerification {
            id: first.pack_id.clone(),
            version: first.pack_version.clone(),
            benchmark_count: definitions.len(),
            case_count: definitions
                .iter()
                .map(|definition| definition.case_count)
                .sum(),
        })
    }

    pub fn remove(&self, pack_id: &str) -> Result<(), String> {
        validate_id(pack_id, "pack id")?;
        if pack_id == "core" {
            return Err("the built-in core pack cannot be removed".into());
        }
        let path = self.packs_dir.join(pack_id);
        if !path.is_dir() {
            return Err(format!("benchmark pack '{pack_id}' is not installed"));
        }
        fs::remove_dir_all(&path)
            .map_err(|error| format!("remove benchmark pack {}: {error}", path.display()))
    }
}

fn load_builtin() -> Result<Vec<BenchmarkDefinition>, String> {
    let manifest: BenchmarkPackManifest = serde_json::from_str(CORE_MANIFEST)
        .map_err(|error| format!("invalid built-in benchmark pack: {error}"))?;
    build_definitions(manifest, true, |case_file| match case_file {
        "text-generation-smoke.jsonl" => Ok(CORE_TEXT_GENERATION.to_string()),
        "embedding-retrieval.jsonl" => Ok(CORE_EMBEDDING.to_string()),
        "image-to-text-smoke.jsonl" => Ok(CORE_IMAGE_TO_TEXT.to_string()),
        "speech-to-text-smoke.jsonl" => Ok(CORE_SPEECH_TO_TEXT.to_string()),
        "reranking-relevance.jsonl" => Ok(CORE_RERANKING.to_string()),
        "text-to-image-smoke.jsonl" => Ok(CORE_TEXT_TO_IMAGE.to_string()),
        _ => Err(format!("unknown built-in case file '{case_file}'")),
    })
}

fn load_pack_dir(directory: &Path) -> Result<Vec<BenchmarkDefinition>, String> {
    if !directory.is_dir() {
        return Err(format!(
            "benchmark pack source {} is not a directory",
            directory.display()
        ));
    }
    let manifest_path = directory.join("pack.json");
    let bytes = fs::read(&manifest_path)
        .map_err(|error| format!("read {}: {error}", manifest_path.display()))?;
    let manifest: BenchmarkPackManifest = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse {}: {error}", manifest_path.display()))?;
    let mut definitions = build_definitions(manifest, false, |case_file| {
        let relative = validated_relative_path(case_file)?;
        let path = directory.join(relative);
        fs::read_to_string(&path).map_err(|error| format!("read {}: {error}", path.display()))
    })?;
    let root = fs::canonicalize(directory)
        .map_err(|error| format!("resolve benchmark pack {}: {error}", directory.display()))?;
    for definition in &mut definitions {
        for case in &mut definition.cases {
            let Some(media) = case.media.as_deref() else {
                continue;
            };
            if media.starts_with("data:")
                || media.starts_with("http://")
                || media.starts_with("https://")
            {
                continue;
            }
            let relative = validated_relative_path(media)?;
            let path = root.join(relative);
            let metadata = fs::symlink_metadata(&path)
                .map_err(|error| format!("inspect benchmark media {}: {error}", path.display()))?;
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                return Err(format!(
                    "benchmark media {} must be a regular file inside its pack",
                    path.display()
                ));
            }
            case.media = Some(path.to_string_lossy().to_string());
        }
    }
    Ok(definitions)
}

fn build_definitions<F>(
    manifest: BenchmarkPackManifest,
    built_in: bool,
    mut read_cases: F,
) -> Result<Vec<BenchmarkDefinition>, String>
where
    F: FnMut(&str) -> Result<String, String>,
{
    if manifest.schema_version != 1 {
        return Err(format!(
            "benchmark pack '{}' uses unsupported schema version {}",
            manifest.id, manifest.schema_version
        ));
    }
    validate_id(&manifest.id, "pack id")?;
    if manifest.name.trim().is_empty()
        || manifest.version.trim().is_empty()
        || manifest.publisher.trim().is_empty()
        || manifest.license.trim().is_empty()
        || manifest.source.trim().is_empty()
    {
        return Err(format!(
            "benchmark pack '{}' requires a name, version, publisher, license, and source",
            manifest.id
        ));
    }
    let mut local_ids = HashSet::new();
    let mut definitions = Vec::new();
    for benchmark in manifest.benchmarks {
        validate_id(&benchmark.id, "benchmark id")?;
        if !local_ids.insert(benchmark.id.clone()) {
            return Err(format!(
                "benchmark pack '{}' repeats benchmark id '{}'",
                manifest.id, benchmark.id
            ));
        }
        if benchmark.name.trim().is_empty() || benchmark.description.trim().is_empty() {
            return Err(format!(
                "benchmark '{}/{}' requires a name and description",
                manifest.id, benchmark.id
            ));
        }
        if benchmark.default_max_output_tokens == 0 {
            return Err(format!(
                "benchmark '{}/{}' must request at least one output token",
                manifest.id, benchmark.id
            ));
        }
        let content = read_cases(&benchmark.cases)?;
        let cases = parse_cases(&manifest.id, &benchmark.id, benchmark.task, &content)?;
        let supported_providers = ProviderKind::ALL
            .into_iter()
            .filter(|provider| provider.supports(benchmark.task))
            .collect();
        definitions.push(BenchmarkDefinition {
            id: format!("{}/{}", manifest.id, benchmark.id),
            pack_id: manifest.id.clone(),
            pack_name: manifest.name.clone(),
            pack_version: manifest.version.clone(),
            name: benchmark.name,
            description: benchmark.description,
            task: benchmark.task,
            case_count: cases.len(),
            tags: benchmark.tags,
            default_max_output_tokens: benchmark.default_max_output_tokens,
            built_in,
            supported_providers,
            cases,
        });
    }
    if definitions.is_empty() {
        return Err(format!(
            "benchmark pack '{}' contains no benchmarks",
            manifest.id
        ));
    }
    Ok(definitions)
}

fn parse_cases(
    pack_id: &str,
    benchmark_id: &str,
    task: Task,
    content: &str,
) -> Result<Vec<BenchmarkCase>, String> {
    let mut cases = Vec::new();
    let mut ids = HashSet::new();
    for (index, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let case: BenchmarkCase = serde_json::from_str(line).map_err(|error| {
            format!(
                "parse case {} for benchmark '{pack_id}/{benchmark_id}': {error}",
                index + 1
            )
        })?;
        validate_id(&case.id, "case id")?;
        if !ids.insert(case.id.clone()) {
            return Err(format!(
                "benchmark '{pack_id}/{benchmark_id}' repeats case id '{}'",
                case.id
            ));
        }
        validate_case(task, &case).map_err(|error| {
            format!(
                "invalid case '{}' for benchmark '{pack_id}/{benchmark_id}': {error}",
                case.id
            )
        })?;
        cases.push(case);
    }
    if cases.is_empty() {
        return Err(format!(
            "benchmark '{pack_id}/{benchmark_id}' contains no cases"
        ));
    }
    Ok(cases)
}

fn validate_case(task: Task, case: &BenchmarkCase) -> Result<(), String> {
    match task {
        Task::TextGeneration | Task::TextToImage => require_text(&case.prompt, "prompt"),
        Task::ImageToText => {
            require_text(&case.prompt, "prompt")?;
            require_text(&case.media, "media")
        }
        Task::SpeechToText => require_text(&case.media, "media"),
        Task::Embedding | Task::Reranking => {
            require_text(&case.input, "input")?;
            if case.documents.is_empty() {
                return Err("documents must contain at least one candidate".into());
            }
            if let Some(expected) = case.expected_index {
                if expected >= case.documents.len() {
                    return Err(format!(
                        "expected_index {expected} is outside {} documents",
                        case.documents.len()
                    ));
                }
            }
            Ok(())
        }
    }
}

fn require_text(value: &Option<String>, name: &str) -> Result<(), String> {
    match value.as_deref() {
        Some(value) if !value.trim().is_empty() => Ok(()),
        _ => Err(format!("{name} is required")),
    }
}

fn validate_id(value: &str, label: &str) -> Result<(), String> {
    let valid = !value.is_empty()
        && value.len() <= 80
        && value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        && value.as_bytes()[0].is_ascii_lowercase();
    if valid {
        Ok(())
    } else {
        Err(format!(
            "{label} '{value}' must start with a lowercase letter and contain only lowercase letters, digits, and hyphens"
        ))
    }
}

fn validated_relative_path(value: &str) -> Result<PathBuf, String> {
    let path = Path::new(value);
    if value.is_empty()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!("case path '{value}' must stay inside its pack"));
    }
    Ok(path.to_path_buf())
}

fn copy_tree(source: &Path, destination: &Path) -> Result<(), String> {
    fs::create_dir_all(destination)
        .map_err(|error| format!("create {}: {error}", destination.display()))?;
    let entries =
        fs::read_dir(source).map_err(|error| format!("read {}: {error}", source.display()))?;
    for entry in entries {
        let entry = entry.map_err(|error| format!("read {}: {error}", source.display()))?;
        let file_type = entry
            .file_type()
            .map_err(|error| format!("inspect {}: {error}", entry.path().display()))?;
        let target = destination.join(entry.file_name());
        if file_type.is_symlink() {
            return Err(format!(
                "benchmark packs cannot contain symlinks: {}",
                entry.path().display()
            ));
        }
        if file_type.is_dir() {
            copy_tree(&entry.path(), &target)?;
        } else if file_type.is_file() {
            fs::copy(entry.path(), &target)
                .map_err(|error| format!("copy {}: {error}", entry.path().display()))?;
        }
    }
    Ok(())
}
