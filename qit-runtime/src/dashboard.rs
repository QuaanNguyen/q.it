use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::model::{BenchmarkDefinition, HostInfo, ProviderKind, RunRecord, SampleRecord, Task};
use crate::packs::PackCatalog;
use crate::spa::{asset as embedded_asset, index_html};
use crate::store::Store;

#[derive(Clone)]
pub struct DashboardState {
    pub store: Arc<Mutex<Store>>,
    pub catalog: PackCatalog,
    pub host: HostInfo,
}

#[derive(Serialize)]
struct HealthBody {
    ok: bool,
}

#[derive(Serialize)]
struct ProviderBody {
    id: ProviderKind,
    name: &'static str,
    default_base_url: &'static str,
    tasks: Vec<Task>,
}

#[derive(Serialize)]
struct RunsBody {
    runs: Vec<RunRecord>,
    succeeded: usize,
    failed: usize,
    running: usize,
}

#[derive(Serialize)]
struct RunDetailBody {
    run: RunRecord,
    samples: Vec<SampleRecord>,
}

#[derive(Clone, Serialize)]
struct AnalysisPoint {
    run_id: String,
    benchmark_id: String,
    pack_version: String,
    model: String,
    provider: ProviderKind,
    host_name: String,
    latency_ms: Option<f64>,
    ttft_ms: Option<f64>,
    tokens_per_second: Option<f64>,
    memory_bytes: Option<u64>,
    cpu_percent: Option<f64>,
    quality_score: Option<f64>,
    pareto: bool,
}

#[derive(Serialize)]
struct AnalysisBody {
    points: Vec<AnalysisPoint>,
}

#[derive(Deserialize)]
struct RunsQuery {
    limit: Option<usize>,
}

pub fn router(state: DashboardState) -> Router {
    Router::new()
        .route("/api/health", get(health))
        .route("/api/host", get(host))
        .route("/api/providers", get(providers))
        .route("/api/benchmarks", get(benchmarks))
        .route("/api/runs", get(runs))
        .route("/api/runs/{id}", get(run_detail))
        .route("/api/analysis", get(analysis))
        .route("/", get(index))
        .route("/{*path}", get(asset_or_index))
        .with_state(state)
}

async fn health() -> Json<HealthBody> {
    Json(HealthBody { ok: true })
}

async fn host(State(state): State<DashboardState>) -> Json<HostInfo> {
    Json(state.host)
}

async fn providers() -> Json<Vec<ProviderBody>> {
    Json(
        ProviderKind::ALL
            .into_iter()
            .map(|provider| ProviderBody {
                id: provider,
                name: provider.label(),
                default_base_url: provider.default_base_url(),
                tasks: Task::ALL
                    .into_iter()
                    .filter(|task| provider.supports(*task))
                    .collect(),
            })
            .collect(),
    )
}

async fn benchmarks(
    State(state): State<DashboardState>,
) -> Result<Json<Vec<BenchmarkDefinition>>, ApiError> {
    state
        .catalog
        .benchmarks()
        .map(Json)
        .map_err(ApiError::internal)
}

async fn runs(
    State(state): State<DashboardState>,
    Query(query): Query<RunsQuery>,
) -> Result<Json<RunsBody>, ApiError> {
    let limit = query.limit.unwrap_or(250).clamp(1, 10_000);
    let runs = state
        .store
        .lock()
        .await
        .runs(limit, false)
        .map_err(ApiError::internal)?;
    let succeeded = runs
        .iter()
        .filter(|run| run.status == crate::model::RunStatus::Succeeded)
        .count();
    let failed = runs
        .iter()
        .filter(|run| run.status == crate::model::RunStatus::Failed)
        .count();
    let running = runs.len().saturating_sub(succeeded + failed);
    Ok(Json(RunsBody {
        runs,
        succeeded,
        failed,
        running,
    }))
}

async fn run_detail(
    State(state): State<DashboardState>,
    Path(id): Path<String>,
) -> Result<Json<RunDetailBody>, ApiError> {
    let store = state.store.lock().await;
    let run = store
        .runs(10_000, false)
        .map_err(ApiError::internal)?
        .into_iter()
        .find(|run| run.id == id)
        .ok_or_else(|| ApiError::not_found("benchmark run not found"))?;
    let samples = store.samples(&id).map_err(ApiError::internal)?;
    Ok(Json(RunDetailBody { run, samples }))
}

async fn analysis(State(state): State<DashboardState>) -> Result<Json<AnalysisBody>, ApiError> {
    let runs = state
        .store
        .lock()
        .await
        .runs(10_000, true)
        .map_err(ApiError::internal)?;
    let mut points = runs
        .into_iter()
        .map(|run| AnalysisPoint {
            run_id: run.id,
            benchmark_id: run.benchmark_id,
            pack_version: run.pack_version,
            model: run.model,
            provider: run.provider,
            host_name: run.host_name,
            latency_ms: run.summary.latency_ms_p50,
            ttft_ms: run.summary.ttft_ms_p50,
            tokens_per_second: run.summary.tokens_per_second_p50,
            memory_bytes: run
                .summary
                .process_rss_bytes_peak
                .or(run.summary.host_memory_used_bytes_peak),
            cpu_percent: run.summary.host_cpu_percent_mean,
            quality_score: run.summary.quality_score_mean,
            pareto: false,
        })
        .collect::<Vec<_>>();
    for index in 0..points.len() {
        points[index].pareto = is_pareto(&points, index);
    }
    Ok(Json(AnalysisBody { points }))
}

fn is_pareto(points: &[AnalysisPoint], index: usize) -> bool {
    let Some(latency) = points[index].latency_ms else {
        return false;
    };
    let Some(memory) = points[index].memory_bytes else {
        return false;
    };
    !points.iter().enumerate().any(|(other_index, other)| {
        if index == other_index {
            return false;
        }
        if other.benchmark_id != points[index].benchmark_id
            || other.pack_version != points[index].pack_version
            || other.host_name != points[index].host_name
        {
            return false;
        }
        match (other.latency_ms, other.memory_bytes) {
            (Some(other_latency), Some(other_memory)) => {
                other_latency <= latency
                    && other_memory <= memory
                    && (other_latency < latency || other_memory < memory)
            }
            _ => false,
        }
    })
}

async fn index() -> Html<String> {
    Html(index_html())
}

async fn asset_or_index(Path(path): Path<String>) -> Response {
    if path.starts_with("api/") {
        return StatusCode::NOT_FOUND.into_response();
    }
    match embedded_asset(&path) {
        Some((mime, bytes)) => {
            let mut headers = HeaderMap::new();
            headers.insert(header::CONTENT_TYPE, mime.parse().unwrap());
            (StatusCode::OK, headers, bytes).into_response()
        }
        None if !path.contains('.') => Html(index_html()).into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(serde_json::json!({ "error": self.message })),
        )
            .into_response()
    }
}
