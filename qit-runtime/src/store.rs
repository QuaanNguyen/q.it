use std::path::Path;
use std::str::FromStr;

use rusqlite::types::Type;
use rusqlite::{params, Connection};

use crate::model::{HostInfo, ProviderKind, RunRecord, RunStatus, RunSummary, SampleRecord, Task};

pub struct Store {
    connection: Connection,
}

impl Store {
    pub fn open(path: &Path) -> Result<Self, String> {
        let connection = Connection::open(path)
            .map_err(|error| format!("open result database {}: {error}", path.display()))?;
        connection
            .execute_batch(
                "
                PRAGMA foreign_keys = ON;
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS benchmark_runs (
                    id TEXT PRIMARY KEY,
                    benchmark_id TEXT NOT NULL,
                    benchmark_name TEXT NOT NULL,
                    pack_version TEXT NOT NULL,
                    task TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    base_url TEXT NOT NULL,
                    model TEXT NOT NULL,
                    host_name TEXT NOT NULL,
                    host_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at_ms INTEGER NOT NULL,
                    finished_at_ms INTEGER,
                    iterations INTEGER NOT NULL,
                    warmups INTEGER NOT NULL,
                    max_output_tokens INTEGER NOT NULL,
                    target_pid INTEGER,
                    error TEXT,
                    summary_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS benchmark_samples (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES benchmark_runs(id) ON DELETE CASCADE,
                    case_id TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    succeeded INTEGER NOT NULL,
                    error TEXT,
                    latency_ms REAL,
                    ttft_ms REAL,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    tokens_per_second REAL,
                    host_cpu_percent_mean REAL,
                    host_cpu_percent_peak REAL,
                    host_memory_used_bytes_peak INTEGER,
                    process_rss_bytes_peak INTEGER,
                    quality_score REAL,
                    output_excerpt TEXT
                );
                CREATE INDEX IF NOT EXISTS benchmark_runs_started_at
                    ON benchmark_runs(started_at_ms DESC);
                CREATE INDEX IF NOT EXISTS benchmark_runs_status
                    ON benchmark_runs(status, started_at_ms DESC);
                CREATE INDEX IF NOT EXISTS benchmark_samples_run
                    ON benchmark_samples(run_id, iteration, case_id);
                ",
            )
            .map_err(|error| format!("initialize result database: {error}"))?;
        ensure_column(
            &connection,
            "benchmark_runs",
            "host_json",
            "TEXT NOT NULL DEFAULT '{}'",
        )?;
        let now = unix_time_ms();
        connection
            .execute(
                "UPDATE benchmark_runs
                 SET status = 'failed', finished_at_ms = ?1,
                     error = COALESCE(error, 'benchmark process ended before the run completed')
                 WHERE status = 'running'",
                [now],
            )
            .map_err(|error| format!("classify interrupted benchmark runs: {error}"))?;
        Ok(Self { connection })
    }

    pub fn start_run(&self, run: &RunRecord) -> Result<(), String> {
        let summary = serde_json::to_string(&run.summary)
            .map_err(|error| format!("serialize run summary: {error}"))?;
        let host = serde_json::to_string(&run.host)
            .map_err(|error| format!("serialize host snapshot: {error}"))?;
        self.connection
            .execute(
                "INSERT INTO benchmark_runs (
                    id, benchmark_id, benchmark_name, pack_version, task, provider,
                    base_url, model, host_name, host_json, status, started_at_ms, finished_at_ms,
                    iterations, warmups, max_output_tokens, target_pid, error, summary_json
                 ) VALUES (
                    ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12,
                    ?13, ?14, ?15, ?16, ?17, ?18, ?19
                 )",
                params![
                    run.id,
                    run.benchmark_id,
                    run.benchmark_name,
                    run.pack_version,
                    run.task.as_str(),
                    run.provider.as_str(),
                    run.base_url,
                    run.model,
                    run.host_name,
                    host,
                    run.status.as_str(),
                    run.started_at_ms,
                    run.finished_at_ms,
                    run.iterations,
                    run.warmups,
                    run.max_output_tokens,
                    run.target_pid,
                    run.error,
                    summary
                ],
            )
            .map_err(|error| format!("store benchmark run: {error}"))?;
        Ok(())
    }

    pub fn insert_sample(&self, sample: &SampleRecord) -> Result<(), String> {
        self.connection
            .execute(
                "INSERT INTO benchmark_samples (
                    id, run_id, case_id, iteration, succeeded, error, latency_ms,
                    ttft_ms, input_tokens, output_tokens, tokens_per_second,
                    host_cpu_percent_mean, host_cpu_percent_peak,
                    host_memory_used_bytes_peak, process_rss_bytes_peak,
                    quality_score, output_excerpt
                 ) VALUES (
                    ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12,
                    ?13, ?14, ?15, ?16, ?17
                 )",
                params![
                    sample.id,
                    sample.run_id,
                    sample.case_id,
                    sample.iteration,
                    sample.succeeded,
                    sample.error,
                    sample.latency_ms,
                    sample.ttft_ms,
                    sample.input_tokens,
                    sample.output_tokens,
                    sample.tokens_per_second,
                    sample.host_cpu_percent_mean,
                    sample.host_cpu_percent_peak,
                    sample.host_memory_used_bytes_peak,
                    sample.process_rss_bytes_peak,
                    sample.quality_score,
                    sample.output_excerpt,
                ],
            )
            .map_err(|error| format!("store benchmark sample: {error}"))?;
        Ok(())
    }

    pub fn finish_run(
        &self,
        id: &str,
        status: RunStatus,
        finished_at_ms: i64,
        error: Option<&str>,
        summary: &RunSummary,
    ) -> Result<(), String> {
        let summary = serde_json::to_string(summary)
            .map_err(|error| format!("serialize run summary: {error}"))?;
        let changed = self
            .connection
            .execute(
                "UPDATE benchmark_runs
                 SET status = ?2, finished_at_ms = ?3, error = ?4, summary_json = ?5
                 WHERE id = ?1",
                params![id, status.as_str(), finished_at_ms, error, summary],
            )
            .map_err(|error| format!("finish benchmark run: {error}"))?;
        if changed == 0 {
            return Err(format!("benchmark run '{id}' was not found"));
        }
        Ok(())
    }

    pub fn runs(&self, limit: usize, succeeded_only: bool) -> Result<Vec<RunRecord>, String> {
        let where_clause = if succeeded_only {
            "WHERE status = 'succeeded'"
        } else {
            ""
        };
        let sql = format!(
            "SELECT id, benchmark_id, benchmark_name, pack_version, task, provider,
                    base_url, model, host_name, host_json, status, started_at_ms, finished_at_ms,
                    iterations, warmups, max_output_tokens, target_pid, error, summary_json
             FROM benchmark_runs
             {where_clause}
             ORDER BY started_at_ms DESC
             LIMIT ?1"
        );
        let mut statement = self
            .connection
            .prepare(&sql)
            .map_err(|error| format!("prepare benchmark history query: {error}"))?;
        let rows = statement
            .query_map([limit as i64], map_run)
            .map_err(|error| format!("query benchmark history: {error}"))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|error| format!("read benchmark history: {error}"))
    }

    pub fn samples(&self, run_id: &str) -> Result<Vec<SampleRecord>, String> {
        let mut statement = self
            .connection
            .prepare(
                "SELECT id, run_id, case_id, iteration, succeeded, error, latency_ms,
                        ttft_ms, input_tokens, output_tokens, tokens_per_second,
                        host_cpu_percent_mean, host_cpu_percent_peak,
                        host_memory_used_bytes_peak, process_rss_bytes_peak,
                        quality_score, output_excerpt
                 FROM benchmark_samples
                 WHERE run_id = ?1
                 ORDER BY iteration, case_id",
            )
            .map_err(|error| format!("prepare sample query: {error}"))?;
        let rows = statement
            .query_map([run_id], |row| {
                Ok(SampleRecord {
                    id: row.get(0)?,
                    run_id: row.get(1)?,
                    case_id: row.get(2)?,
                    iteration: row.get(3)?,
                    succeeded: row.get(4)?,
                    error: row.get(5)?,
                    latency_ms: row.get(6)?,
                    ttft_ms: row.get(7)?,
                    input_tokens: row.get(8)?,
                    output_tokens: row.get(9)?,
                    tokens_per_second: row.get(10)?,
                    host_cpu_percent_mean: row.get(11)?,
                    host_cpu_percent_peak: row.get(12)?,
                    host_memory_used_bytes_peak: row.get(13)?,
                    process_rss_bytes_peak: row.get(14)?,
                    quality_score: row.get(15)?,
                    output_excerpt: row.get(16)?,
                })
            })
            .map_err(|error| format!("query benchmark samples: {error}"))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|error| format!("read benchmark samples: {error}"))
    }
}

fn map_run(row: &rusqlite::Row<'_>) -> rusqlite::Result<RunRecord> {
    let task: String = row.get(4)?;
    let provider: String = row.get(5)?;
    let host_name: String = row.get(8)?;
    let host: String = row.get(9)?;
    let status: String = row.get(10)?;
    let summary: String = row.get(18)?;
    Ok(RunRecord {
        id: row.get(0)?,
        benchmark_id: row.get(1)?,
        benchmark_name: row.get(2)?,
        pack_version: row.get(3)?,
        task: parse_task(&task).map_err(|error| conversion_error(4, error))?,
        provider: ProviderKind::from_str(&provider).map_err(|error| conversion_error(5, error))?,
        base_url: row.get(6)?,
        model: row.get(7)?,
        host_name: host_name.clone(),
        host: parse_host(&host, host_name).map_err(|error| conversion_error(9, error))?,
        status: RunStatus::parse(&status).map_err(|error| conversion_error(10, error))?,
        started_at_ms: row.get(11)?,
        finished_at_ms: row.get(12)?,
        iterations: row.get(13)?,
        warmups: row.get(14)?,
        max_output_tokens: row.get(15)?,
        target_pid: row.get(16)?,
        error: row.get(17)?,
        summary: serde_json::from_str(&summary)
            .map_err(|error| conversion_error(18, error.to_string()))?,
    })
}

fn parse_task(value: &str) -> Result<Task, String> {
    Task::ALL
        .into_iter()
        .find(|task| task.as_str() == value)
        .ok_or_else(|| format!("invalid benchmark task '{value}'"))
}

fn parse_host(value: &str, host_name: String) -> Result<HostInfo, String> {
    serde_json::from_str(value).or_else(|_| {
        Ok(HostInfo {
            host_name,
            operating_system: "unknown".into(),
            kernel: None,
            architecture: "unknown".into(),
            device_model: None,
            cpu: None,
            logical_cpu_count: None,
            total_memory_bytes: None,
            raspberry_pi: false,
            qit_version: "unknown".into(),
        })
    })
}

fn ensure_column(
    connection: &Connection,
    table: &str,
    column: &str,
    declaration: &str,
) -> Result<(), String> {
    let mut statement = connection
        .prepare(&format!("PRAGMA table_info({table})"))
        .map_err(|error| format!("inspect result database schema: {error}"))?;
    let columns = statement
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(|error| format!("read result database schema: {error}"))?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|error| format!("read result database schema: {error}"))?;
    if !columns.iter().any(|existing| existing == column) {
        connection
            .execute(
                &format!("ALTER TABLE {table} ADD COLUMN {column} {declaration}"),
                [],
            )
            .map_err(|error| format!("upgrade result database schema: {error}"))?;
    }
    Ok(())
}

fn conversion_error(index: usize, error: String) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(
        index,
        Type::Text,
        Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, error)),
    )
}

pub fn unix_time_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}
