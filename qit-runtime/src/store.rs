use std::path::{Path, PathBuf};

use rusqlite::{params, Connection, OptionalExtension};

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
    pub confidence: String,
}

#[derive(Clone, Debug)]
pub struct PinRow {
    pub id: String,
    pub artifact_id: String,
    pub n_ctx: u32,
    pub n_gpu_layers: i32,
    pub n_parallel: u32,
}

#[derive(Clone, Debug)]
pub struct MeasurementRow {
    pub artifact_id: String,
    pub throughput_tps: Option<f64>,
    pub peak_rss_bytes: Option<u64>,
    pub n_tokens: Option<u32>,
    pub generation_ms: Option<f64>,
}

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
                confidence TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS pins (
                id TEXT PRIMARY KEY,
                artifact_id TEXT NOT NULL,
                n_ctx INTEGER NOT NULL,
                n_gpu_layers INTEGER NOT NULL,
                n_parallel INTEGER NOT NULL,
                UNIQUE(artifact_id, n_ctx, n_gpu_layers, n_parallel)
            );
            CREATE TABLE IF NOT EXISTS measurements (
                artifact_id TEXT PRIMARY KEY,
                throughput_tps REAL,
                peak_rss_bytes INTEGER,
                n_tokens INTEGER,
                generation_ms REAL
            );
            ",
        )?;
        Ok(Self { conn })
    }

    pub fn replace_artifacts(&self, rows: &[ArtifactRow]) -> rusqlite::Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute("DELETE FROM artifacts", [])?;
        {
            let mut stmt = tx.prepare(
                "INSERT INTO artifacts (
                    id, org, filename, path, bytes, architecture, context_length,
                    block_count, embedding_length, head_count, head_count_kv, confidence
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            )?;
            for row in rows {
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
                    block_count, embedding_length, head_count, head_count_kv, confidence
             FROM artifacts ORDER BY org, filename",
        )?;
        let rows = stmt.query_map([], |r| {
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
                confidence: r.get(11)?,
            })
        })?;
        rows.collect()
    }

    pub fn artifact(&self, id: &str) -> rusqlite::Result<Option<ArtifactRow>> {
        self.conn
            .query_row(
                "SELECT id, org, filename, path, bytes, architecture, context_length,
                        block_count, embedding_length, head_count, head_count_kv, confidence
                 FROM artifacts WHERE id = ?1",
                [id],
                |r| {
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
                        confidence: r.get(11)?,
                    })
                },
            )
            .optional()
    }

    pub fn insert_pin(&self, pin: &PinRow) -> rusqlite::Result<()> {
        self.conn.execute(
            "INSERT INTO pins (id, artifact_id, n_ctx, n_gpu_layers, n_parallel)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                pin.id,
                pin.artifact_id,
                pin.n_ctx as i64,
                pin.n_gpu_layers,
                pin.n_parallel as i64
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
            "SELECT id, artifact_id, n_ctx, n_gpu_layers, n_parallel FROM pins ORDER BY artifact_id, n_ctx",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(PinRow {
                id: r.get(0)?,
                artifact_id: r.get(1)?,
                n_ctx: r.get::<_, i64>(2)? as u32,
                n_gpu_layers: r.get(3)?,
                n_parallel: r.get::<_, i64>(4)? as u32,
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
}
