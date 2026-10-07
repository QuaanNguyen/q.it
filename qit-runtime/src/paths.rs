use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct Paths {
    pub home: PathBuf,
    pub database: PathBuf,
    pub packs: PathBuf,
}

impl Paths {
    pub fn new(home: PathBuf) -> Self {
        Self {
            database: home.join("results.db"),
            packs: home.join("benchmark-packs"),
            home,
        }
    }

    pub fn ensure(&self) -> Result<(), String> {
        std::fs::create_dir_all(&self.home)
            .map_err(|error| format!("create q.it home {}: {error}", self.home.display()))?;
        std::fs::create_dir_all(&self.packs).map_err(|error| {
            format!(
                "create benchmark pack directory {}: {error}",
                self.packs.display()
            )
        })
    }
}
