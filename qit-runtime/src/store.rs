use std::path::{Path, PathBuf};

use rusqlite::{params, Connection, OptionalExtension};

use crate::gguf::{ArtifactKind, PlannerHints};
use crate::serve::{RuntimeRecipe, ServeProfile, TargetIdentity};

#[derive(Clone, Debug)]
pub struct ArtifactRow {
    pub id: String,
    pub org: String,
    pub filename: String,
    pub path: PathBuf,
    pub bytes: u64,
    pub architecture: Option<String>,
    pub context_length: Option<u32>,
    pub block_count: Option<u32>,
    pub embedding_length: Option<u32>,
    pub head_count: Option<u32>,
    pub head_count_kv: Option<u32>,
    pub kind: ArtifactKind,
    pub planner: PlannerHints,
    pub confidence: String,
}

#[derive(Clone, Debug)]
pub struct PinRow {
    pub id: String,
    pub target: TargetIdentity,
    pub runtime_recipe: RuntimeRecipe,
    pub serve_profile: ServeProfile,
}

#[derive(Clone, Debug)]
pub struct MeasurementRow {
    pub artifact_id: String,
    pub throughput_tps: Option<f64>,
    pub peak_rss_bytes: Option<u64>,
    pub n_tokens: Option<u32>,
    pub generation_ms: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct SessionRow {
    pub id: String,
    pub target: TargetIdentity,
    pub runtime_recipe: RuntimeRecipe,
    pub serve_profile: ServeProfile,
    pub status: String,
    pub last_error: Option<String>,
    pub log_path: Option<String>,
}

const OS_RESERVE_KEY: &str = "os_reserve_bytes";

pub struct Store {
    conn: Connection,
}

impl Store {
    pub fn open(path: &Path) -> rusqlite::Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch(
            "
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS artifacts (
                id TEXT PRIMARY KEY,
                org TEXT NOT NULL,
                filename TEXT NOT NULL,
                path TEXT NOT NULL UNIQUE,
                bytes INTEGER NOT NULL,
                architecture TEXT,
                context_length INTEGER,
                block_count INTEGER,
                embedding_length INTEGER,
                head_count INTEGER,
                head_count_kv INTEGER,
                kind TEXT NOT NULL DEFAULT 'unknown',
                planner_json TEXT,
                confidence TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS pins (
                id TEXT PRIMARY KEY,
                target_id TEXT NOT NULL,
                artifact_id TEXT,
                package_id TEXT,
                runtime_recipe TEXT NOT NULL,
                serve_profile_json TEXT NOT NULL,
                UNIQUE(target_id, serve_profile_json)
            );
            CREATE TABLE IF NOT EXISTS measurements (
                artifact_id TEXT PRIMARY KEY,
                throughput_tps REAL,
                peak_rss_bytes INTEGER,
                n_tokens INTEGER,
                generation_ms REAL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                target_id TEXT NOT NULL,
                artifact_id TEXT,
                package_id TEXT,
                runtime_recipe TEXT NOT NULL,
                serve_profile_json TEXT NOT NULL,
                status TEXT NOT NULL,
                last_error TEXT,
                log_path TEXT,
                UNIQUE(target_id, serve_profile_json)
            );
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            ",
        )?;
        ensure_artifact_columns(&conn)?;
        ensure_serve_tables(&conn)?;
        Ok(Self { conn })
    }

    pub fn setting(&self, key: &str) -> rusqlite::Result<Option<String>> {
        self.conn
            .query_row("SELECT value FROM settings WHERE key = ?1", [key], |r| {
                r.get(0)
            })
            .optional()
    }

    pub fn set_setting(&self, key: &str, value: Option<&str>) -> rusqlite::Result<()> {
        match value {
            Some(v) => {
                self.conn.execute(
                    "INSERT INTO settings (key, value) VALUES (?1, ?2)
                     ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    params![key, v],
                )?;
            }
            None => {
                self.conn
                    .execute("DELETE FROM settings WHERE key = ?1", [key])?;
            }
        }
        Ok(())
    }

    pub fn os_reserve_setting(&self) -> rusqlite::Result<Option<u64>> {
        Ok(self.setting(OS_RESERVE_KEY)?.and_then(|v| v.parse().ok()))
    }

    pub fn set_os_reserve_setting(&self, bytes: Option<u64>) -> rusqlite::Result<()> {
        self.set_setting(OS_RESERVE_KEY, bytes.map(|b| b.to_string()).as_deref())
    }

    pub fn replace_artifacts(&self, rows: &[ArtifactRow]) -> rusqlite::Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute("DELETE FROM artifacts", [])?;
        {
            let mut stmt = tx.prepare(
                "INSERT INTO artifacts (
                    id, org, filename, path, bytes, architecture, context_length,
                    block_count, embedding_length, head_count, head_count_kv, kind,
                    planner_json, confidence
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
            )?;
            for row in rows {
                let planner = serde_json::to_string(&row.planner).unwrap_or_else(|_| "{}".into());
                stmt.execute(params![
                    row.id,
                    row.org,
                    row.filename,
                    row.path.to_string_lossy(),
                    row.bytes as i64,
                    row.architecture,
                    row.context_length.map(|v| v as i64),
                    row.block_count.map(|v| v as i64),
                    row.embedding_length.map(|v| v as i64),
                    row.head_count.map(|v| v as i64),
                    row.head_count_kv.map(|v| v as i64),
                    row.kind.as_str(),
                    planner,
                    row.confidence,
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn artifacts(&self) -> rusqlite::Result<Vec<ArtifactRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, org, filename, path, bytes, architecture, context_length,
                    block_count, embedding_length, head_count, head_count_kv, kind,
                    planner_json, confidence
             FROM artifacts ORDER BY org, filename",
        )?;
        let rows = stmt.query_map([], map_artifact)?;
        rows.collect()
    }

    pub fn artifact(&self, id: &str) -> rusqlite::Result<Option<ArtifactRow>> {
        self.conn
            .query_row(
                "SELECT id, org, filename, path, bytes, architecture, context_length,
                        block_count, embedding_length, head_count, head_count_kv, kind,
                        planner_json, confidence
                 FROM artifacts WHERE id = ?1",
                [id],
                map_artifact,
            )
            .optional()
    }

    pub fn insert_pin(&self, pin: &PinRow) -> rusqlite::Result<()> {
        let serve_profile_json = serde_json::to_string(&pin.serve_profile).unwrap_or_default();
        self.conn.execute(
            "INSERT INTO pins (
                id, target_id, artifact_id, package_id, runtime_recipe, serve_profile_json
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                pin.id,
                pin.target.id(),
                pin.target.artifact_id(),
                pin.target.package_id(),
                pin.runtime_recipe.as_str(),
                serve_profile_json
            ],
        )?;
        Ok(())
    }

    pub fn delete_pin(&self, id: &str) -> rusqlite::Result<bool> {
        let n = self.conn.execute("DELETE FROM pins WHERE id = ?1", [id])?;
        Ok(n > 0)
    }

    pub fn pins(&self) -> rusqlite::Result<Vec<PinRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, target_id, artifact_id, package_id, runtime_recipe, serve_profile_json
             FROM pins ORDER BY target_id, serve_profile_json",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(PinRow {
                id: r.get(0)?,
                target: map_target(r, 1, 2, 3)?,
                runtime_recipe: map_runtime_recipe(r, 4)?,
                serve_profile: serde_json::from_str(&r.get::<_, String>(5)?).unwrap_or_default(),
            })
        })?;
        rows.collect()
    }

    pub fn upsert_measurement(&self, row: &MeasurementRow) -> rusqlite::Result<()> {
        self.conn.execute(
            "INSERT INTO measurements (artifact_id, throughput_tps, peak_rss_bytes, n_tokens, generation_ms)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(artifact_id) DO UPDATE SET
                throughput_tps = excluded.throughput_tps,
                peak_rss_bytes = excluded.peak_rss_bytes,
                n_tokens = excluded.n_tokens,
                generation_ms = excluded.generation_ms",
            params![
                row.artifact_id,
                row.throughput_tps,
                row.peak_rss_bytes.map(|v| v as i64),
                row.n_tokens.map(|v| v as i64),
                row.generation_ms
            ],
        )?;
        Ok(())
    }

    pub fn measurement(&self, artifact_id: &str) -> rusqlite::Result<Option<MeasurementRow>> {
        self.conn
            .query_row(
                "SELECT artifact_id, throughput_tps, peak_rss_bytes, n_tokens, generation_ms
                 FROM measurements WHERE artifact_id = ?1",
                [artifact_id],
                |r| {
                    Ok(MeasurementRow {
                        artifact_id: r.get(0)?,
                        throughput_tps: r.get(1)?,
                        peak_rss_bytes: r.get::<_, Option<i64>>(2)?.map(|v| v as u64),
                        n_tokens: r.get::<_, Option<i64>>(3)?.map(|v| v as u32),
                        generation_ms: r.get(4)?,
                    })
                },
            )
            .optional()
    }

    pub fn reset_sessions_on_restart(&self) -> rusqlite::Result<()> {
        self.conn.execute(
            "UPDATE sessions SET status = 'not_loaded', last_error = NULL",
            [],
        )?;
        Ok(())
    }

    pub fn upsert_session(&self, row: &SessionRow) -> rusqlite::Result<()> {
        let serve_profile_json = serde_json::to_string(&row.serve_profile).unwrap_or_default();
        self.conn.execute(
            "INSERT INTO sessions (
                id, target_id, artifact_id, package_id, runtime_recipe, serve_profile_json,
                status, last_error, log_path
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
            ON CONFLICT(target_id, serve_profile_json) DO UPDATE SET
                id = excluded.id,
                artifact_id = excluded.artifact_id,
                package_id = excluded.package_id,
                runtime_recipe = excluded.runtime_recipe,
                status = excluded.status,
                last_error = excluded.last_error,
                log_path = excluded.log_path",
            params![
                row.id,
                row.target.id(),
                row.target.artifact_id(),
                row.target.package_id(),
                row.runtime_recipe.as_str(),
                serve_profile_json,
                row.status,
                row.last_error,
                row.log_path,
            ],
        )?;
        Ok(())
    }

    pub fn delete_session(&self, id: &str) -> rusqlite::Result<bool> {
        let n = self
            .conn
            .execute("DELETE FROM sessions WHERE id = ?1", [id])?;
        Ok(n > 0)
    }

    pub fn sessions(&self) -> rusqlite::Result<Vec<SessionRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, target_id, artifact_id, package_id, runtime_recipe, serve_profile_json,
                    status, last_error, log_path
             FROM sessions ORDER BY target_id, serve_profile_json",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(SessionRow {
                id: r.get(0)?,
                target: map_target(r, 1, 2, 3)?,
                runtime_recipe: map_runtime_recipe(r, 4)?,
                serve_profile: serde_json::from_str(&r.get::<_, String>(5)?).unwrap_or_default(),
                status: r.get(6)?,
                last_error: r.get(7)?,
                log_path: r.get(8)?,
            })
        })?;
        rows.collect()
    }
}

fn map_runtime_recipe(r: &rusqlite::Row<'_>, index: usize) -> rusqlite::Result<RuntimeRecipe> {
    let value: String = r.get(index)?;
    RuntimeRecipe::parse(&value).map_err(|message| {
        rusqlite::Error::FromSqlConversionFailure(
            index,
            rusqlite::types::Type::Text,
            Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                message,
            )),
        )
    })
}

fn map_artifact(r: &rusqlite::Row<'_>) -> rusqlite::Result<ArtifactRow> {
    let kind: String = r.get(11)?;
    let planner_json: Option<String> = r.get(12)?;
    let planner = planner_json
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    Ok(ArtifactRow {
        id: r.get(0)?,
        org: r.get(1)?,
        filename: r.get(2)?,
        path: PathBuf::from(r.get::<_, String>(3)?),
        bytes: r.get::<_, i64>(4)? as u64,
        architecture: r.get(5)?,
        context_length: r.get::<_, Option<i64>>(6)?.map(|v| v as u32),
        block_count: r.get::<_, Option<i64>>(7)?.map(|v| v as u32),
        embedding_length: r.get::<_, Option<i64>>(8)?.map(|v| v as u32),
        head_count: r.get::<_, Option<i64>>(9)?.map(|v| v as u32),
        head_count_kv: r.get::<_, Option<i64>>(10)?.map(|v| v as u32),
        kind: ArtifactKind::parse(&kind),
        planner,
        confidence: r.get(13)?,
    })
}

fn map_target(
    row: &rusqlite::Row<'_>,
    target_index: usize,
    artifact_index: usize,
    package_index: usize,
) -> rusqlite::Result<TargetIdentity> {
    let target_id: String = row.get(target_index)?;
    let artifact_id: Option<String> = row.get(artifact_index)?;
    let package_id: Option<String> = row.get(package_index)?;
    Ok(if package_id.is_some() {
        TargetIdentity::Package {
            id: target_id,
            artifact_id,
        }
    } else {
        TargetIdentity::Artifact(artifact_id.unwrap_or(target_id))
    })
}

fn ensure_artifact_columns(conn: &Connection) -> rusqlite::Result<()> {
    let mut stmt = conn.prepare("PRAGMA table_info(artifacts)")?;
    let names: Vec<String> = stmt
        .query_map([], |r| r.get::<_, String>(1))?
        .collect::<Result<_, _>>()?;
    if !names.iter().any(|n| n == "kind") {
        conn.execute(
            "ALTER TABLE artifacts ADD COLUMN kind TEXT NOT NULL DEFAULT 'unknown'",
            [],
        )?;
    }
    if !names.iter().any(|n| n == "planner_json") {
        conn.execute("ALTER TABLE artifacts ADD COLUMN planner_json TEXT", [])?;
    }
    Ok(())
}

fn ensure_serve_tables(conn: &Connection) -> rusqlite::Result<()> {
    if !table_columns(conn, "pins")?
        .iter()
        .any(|name| name == "target_id")
    {
        conn.execute_batch(
            r#"
            BEGIN IMMEDIATE;
            ALTER TABLE pins RENAME TO pins_legacy;
            CREATE TABLE pins (
                id TEXT PRIMARY KEY,
                target_id TEXT NOT NULL,
                artifact_id TEXT,
                package_id TEXT,
                runtime_recipe TEXT NOT NULL,
                serve_profile_json TEXT NOT NULL,
                UNIQUE(target_id, serve_profile_json)
            );
            INSERT INTO pins (
                id, target_id, artifact_id, package_id, runtime_recipe, serve_profile_json
            )
            SELECT id, artifact_id, artifact_id, NULL, 'llama_cpp',
                '{"context_length":' || n_ctx ||
                ',"runtime_settings":{"gpu_layers":' || n_gpu_layers ||
                ',"parallel":' || n_parallel || '}}'
            FROM pins_legacy;
            DROP TABLE pins_legacy;
            COMMIT;
            "#,
        )?;
    }
    if !table_columns(conn, "sessions")?
        .iter()
        .any(|name| name == "target_id")
    {
        conn.execute_batch(
            r#"
            BEGIN IMMEDIATE;
            ALTER TABLE sessions RENAME TO sessions_legacy;
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                target_id TEXT NOT NULL,
                artifact_id TEXT,
                package_id TEXT,
                runtime_recipe TEXT NOT NULL,
                serve_profile_json TEXT NOT NULL,
                status TEXT NOT NULL,
                last_error TEXT,
                log_path TEXT,
                UNIQUE(target_id, serve_profile_json)
            );
            INSERT INTO sessions (
                id, target_id, artifact_id, package_id, runtime_recipe, serve_profile_json,
                status, last_error, log_path
            )
            SELECT id, artifact_id, artifact_id, NULL, 'llama_cpp',
                '{"context_length":' || n_ctx ||
                ',"runtime_settings":{"gpu_layers":' || n_gpu_layers ||
                ',"parallel":' || n_parallel || '}}',
                status, last_error, log_path
            FROM sessions_legacy;
            DROP TABLE sessions_legacy;
            COMMIT;
            "#,
        )?;
    }
    make_artifact_nullable(conn, "pins")?;
    make_artifact_nullable(conn, "sessions")?;
    Ok(())
}

fn make_artifact_nullable(conn: &Connection, table: &str) -> rusqlite::Result<()> {
    if !column_is_not_null(conn, table, "artifact_id")? {
        return Ok(());
    }
    let (extra_columns, copied_columns) = if table == "pins" {
        (String::new(), "runtime_recipe, serve_profile_json")
    } else {
        (
            "status TEXT NOT NULL, last_error TEXT, log_path TEXT,".to_string(),
            "runtime_recipe, serve_profile_json, status, last_error, log_path",
        )
    };
    conn.execute_batch(&format!(
        "BEGIN IMMEDIATE;
         ALTER TABLE {table} RENAME TO {table}_required_artifact;
         CREATE TABLE {table} (
             id TEXT PRIMARY KEY,
             target_id TEXT NOT NULL,
             artifact_id TEXT,
             package_id TEXT,
             runtime_recipe TEXT NOT NULL,
             serve_profile_json TEXT NOT NULL,
             {extra_columns}
             UNIQUE(target_id, serve_profile_json)
         );
         INSERT INTO {table} (id, target_id, artifact_id, package_id, {copied_columns})
         SELECT id, target_id,
             CASE WHEN package_id IS NOT NULL AND artifact_id = target_id
                 THEN NULL ELSE artifact_id END,
             package_id, {copied_columns}
         FROM {table}_required_artifact;
         DROP TABLE {table}_required_artifact;
         COMMIT;"
    ))
}

fn column_is_not_null(conn: &Connection, table: &str, column: &str) -> rusqlite::Result<bool> {
    let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        if row.get::<_, String>(1)? == column {
            return Ok(row.get::<_, i64>(3)? != 0);
        }
    }
    Ok(false)
}

fn table_columns(conn: &Connection, table: &str) -> rusqlite::Result<Vec<String>> {
    let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
    let columns = stmt.query_map([], |row| row.get(1))?.collect();
    columns
}
