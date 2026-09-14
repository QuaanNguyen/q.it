use std::path::{Path, PathBuf};

pub struct OwnedPackage {
    pub id: &'static str,
    pub family: &'static str,
    pub name: &'static str,
    pub format: &'static str,
    pub estimate_bytes: u64,
    required_files: &'static [&'static str],
}

impl OwnedPackage {
    pub fn has_required_files(&self, models_dir: &Path) -> bool {
        self.required_files
            .iter()
            .map(PathBuf::from)
            .all(|path| models_dir.join(path).is_file())
    }
}

pub fn owned_packages() -> [OwnedPackage; 1] {
    [OwnedPackage {
        id: "qit/qwen2.5-0.5b-instruct-q4_k_m",
        family: "Qwen 2.5",
        name: "Qwen2.5 0.5B Instruct Q4_K_M",
        format: "gguf",
        estimate_bytes: 850_000_000,
        required_files: &["Qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf"],
    }]
}
