use std::fs;
use std::io;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct Paths {
    pub home: PathBuf,
    pub models_dir: PathBuf,
    pub catalog_dir: PathBuf,
    pub logs_dir: PathBuf,
    pub tmp_dir: PathBuf,
    pub db_path: PathBuf,
}

impl Paths {
    pub fn new(home: PathBuf, models_dir: PathBuf) -> Self {
        Self {
            catalog_dir: home.join("catalog"),
            logs_dir: home.join("logs"),
            tmp_dir: home.join("tmp"),
            db_path: home.join("runtime.db"),
            home,
            models_dir,
        }
    }

    pub fn ensure(&self) -> io::Result<()> {
        for dir in [&self.home, &self.catalog_dir, &self.logs_dir, &self.tmp_dir] {
            fs::create_dir_all(dir)?;
        }
        if let Some(parent) = self.models_dir.parent() {
            fs::create_dir_all(parent)?;
        }
        Ok(())
    }

    pub fn worker_log(&self, session_id: &str) -> PathBuf {
        self.logs_dir.join(format!("{session_id}.log"))
    }

    pub fn is_under_models(&self, path: &Path) -> bool {
        path.starts_with(&self.models_dir)
    }
}
