use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let development_assets = manifest.join("../qit-web/dist");
    let packaged_assets = manifest.join("web-dist");
    let assets = if development_assets.join("index.html").is_file() {
        development_assets
    } else {
        packaged_assets
    };
    println!("cargo:rerun-if-changed={}", assets.display());
    let mut files = Vec::new();
    collect_files(&assets, &assets, &mut files);
    files.sort_by(|left, right| left.0.cmp(&right.0));
    let mut generated = String::from(
        "pub fn embedded_asset(path: &str) -> Option<(&'static str, &'static [u8])> {\n    match path {\n",
    );
    for (relative, absolute) in files {
        generated.push_str(&format!(
            "        {:?} => Some(({:?}, include_bytes!({:?}))),\n",
            relative,
            mime_for(&relative),
            absolute
        ));
    }
    generated.push_str("        _ => None,\n    }\n}\n");
    let output = PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("web_assets.rs");
    fs::write(output, generated).unwrap();
}

fn collect_files(root: &Path, directory: &Path, files: &mut Vec<(String, String)>) {
    for entry in fs::read_dir(directory).unwrap().flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_files(root, &path, files);
        } else if path.is_file() {
            let relative = path
                .strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            files.push((relative, path.to_string_lossy().to_string()));
        }
    }
}

fn mime_for(path: &str) -> &'static str {
    match Path::new(path).extension().and_then(|value| value.to_str()) {
        Some("js") => "text/javascript",
        Some("css") => "text/css",
        Some("html") => "text/html; charset=utf-8",
        Some("svg") => "image/svg+xml",
        Some("json") => "application/json",
        _ => "application/octet-stream",
    }
}
