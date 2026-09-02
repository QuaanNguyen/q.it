use std::fs;
use std::path::Path;

use crate::gguf::{read_gguf_meta, ArtifactKind, PlannerHints};
use crate::store::ArtifactRow;

pub fn scan_library(models_dir: &Path) -> Vec<ArtifactRow> {
    let mut out = Vec::new();
    if !models_dir.exists() {
        return out;
    }
    let Ok(orgs) = fs::read_dir(models_dir) else {
        return out;
    };
    for org_ent in orgs.flatten() {
        let org_path = org_ent.path();
        if !org_path.is_dir() {
            continue;
        }
        let org = org_ent.file_name().to_string_lossy().to_string();
        if org.starts_with('.') {
            continue;
        }
        let Ok(files) = fs::read_dir(&org_path) else {
            continue;
        };
        for file_ent in files.flatten() {
            let path = file_ent.path();
            let filename = file_ent.file_name().to_string_lossy().to_string();
            if !filename.to_ascii_lowercase().ends_with(".gguf") {
                continue;
            }
            let Ok(meta_fs) = fs::metadata(&path) else {
                continue;
            };
            if !meta_fs.is_file() {
                continue;
            }
            let gguf = read_gguf_meta(&path);
            let (
                architecture,
                context_length,
                block_count,
                embedding_length,
                head_count,
                head_count_kv,
                kind,
                planner,
                confidence,
            ) = match gguf {
                Some(m) => {
                    let complete = m.architecture.is_some()
                        && m.block_count.is_some()
                        && m.embedding_length.is_some();
                    (
                        m.architecture.clone(),
                        m.context_length,
                        m.block_count,
                        m.embedding_length,
                        m.head_count,
                        m.scalar_head_count_kv(),
                        m.kind(),
                        m.planner(),
                        if complete {
                            "headers".to_string()
                        } else {
                            "incomplete".to_string()
                        },
                    )
                }
                None => (
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    ArtifactKind::Unknown,
                    PlannerHints::default(),
                    "incomplete".to_string(),
                ),
            };
            out.push(ArtifactRow {
                id: format!("{org}/{filename}"),
                org: org.clone(),
                filename,
                path,
                bytes: meta_fs.len(),
                architecture,
                context_length,
                block_count,
                embedding_length,
                head_count,
                head_count_kv,
                kind,
                planner,
                confidence,
            });
        }
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}
