use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, Semaphore};

use crate::model::{
    BenchmarkDefinition, HostInfo, ProviderKind, RunRecord, RunRequest, RunStatus, SampleRecord,
    Task,
};
use crate::packs::PackCatalog;
use crate::runner::{execute_benchmark, prepare_benchmark};
use crate::spa::{asset as embedded_asset, index_html};
use crate::store::{unix_time_ms, Store};

#[derive(Clone)]
pub struct DashboardState {
    pub store: Arc<Mutex<Store>>,
    pub catalog: PackCatalog,
    pub host: HostInfo,
    pub database: PathBuf,
    pub execution: Arc<Semaphore>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StartRunBody {
    benchmark_id: String,
    provider: ProviderKind,
    model: String,
    base_url: Option<String>,
    api_key: Option<String>,
    iterations: u32,
    warmups: u32,
    max_output_tokens: Option<u32>,
    timeout_seconds: u64,
    target_pid: Option<u32>,
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
        .route("/api/runs", get(runs).post(start_run))
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
        .run(&id)
        .map_err(ApiError::internal)?
        .ok_or_else(|| ApiError::not_found("benchmark run not found"))?;
    let samples = store.samples(&id).map_err(ApiError::internal)?;
    Ok(Json(RunDetailBody { run, samples }))
}

async fn start_run(
    State(state): State<DashboardState>,
    headers: HeaderMap,
    Json(body): Json<StartRunBody>,
) -> Result<(StatusCode, Json<RunRecord>), ApiError> {
    if let Some(origin) = headers.get(header::ORIGIN) {
        let origin = origin
            .to_str()
            .ok()
            .and_then(|value| reqwest::Url::parse(value).ok())
            .ok_or_else(|| ApiError::forbidden("invalid request origin"))?;
        let authority = headers
            .get(header::HOST)
            .and_then(|value| value.to_str().ok());
        if authority
            != Some(
                origin
                    .as_str()
                    .trim_end_matches('/')
                    .trim_start_matches("http://")
                    .trim_start_matches("https://"),
            )
        {
            return Err(ApiError::forbidden(
                "benchmark requests must come from this dashboard",
            ));
        }
    }
    let permit =
        state.execution.clone().try_acquire_owned().map_err(|_| {
            ApiError::conflict("a benchmark is already running; wait for it to finish")
        })?;
    let benchmark = state
        .catalog
        .benchmark(&body.benchmark_id)
        .map_err(ApiError::bad_request)?;
    if !body.provider.supports(benchmark.task) {
        return Err(ApiError::bad_request(
            "this provider does not support the selected benchmark",
        ));
    }
    let base_url = body
        .base_url
        .unwrap_or_else(|| body.provider.default_base_url().to_string());
    let endpoint = reqwest::Url::parse(&base_url)
        .map_err(|_| ApiError::bad_request("the endpoint must be an HTTP or HTTPS URL"))?;
    if !matches!(endpoint.scheme(), "http" | "https")
        || endpoint.host_str().is_none()
        || !endpoint.username().is_empty()
        || endpoint.password().is_some()
        || endpoint.query().is_some()
        || endpoint.fragment().is_some()
    {
        return Err(ApiError::bad_request(
            "use an HTTP or HTTPS endpoint without embedded credentials, a query, or a fragment",
        ));
    }
    if body.timeout_seconds > 86_400 || body.target_pid == Some(0) {
        return Err(ApiError::bad_request(
            "use a timeout of at most 86400 seconds and a positive process ID",
        ));
    }
    let request = RunRequest {
        benchmark_id: body.benchmark_id,
        provider: body.provider,
        base_url: endpoint.as_str().trim_end_matches('/').to_string(),
        api_key: body.api_key.filter(|value| !value.is_empty()),
        model: body.model.trim().to_string(),
        iterations: body.iterations,
        warmups: body.warmups,
        max_output_tokens: body.max_output_tokens,
        timeout_seconds: body.timeout_seconds,
        target_pid: body.target_pid,
    };
    let prepared = {
        let store = state.store.lock().await;
        if store.has_running_run().map_err(ApiError::internal)? {
            return Err(ApiError::conflict(
                "a benchmark is already running; wait for it to finish",
            ));
        }
        prepare_benchmark(&store, &state.catalog, request).map_err(ApiError::bad_request)?
    };
    let run = prepared.run.clone();
    let run_id = run.id.clone();
    let database = state.database.clone();
    let runtime = tokio::runtime::Handle::current();
    let worker = tokio::task::spawn_blocking(move || {
        let store = Store::connect(&database)?;
        runtime.block_on(execute_benchmark(&store, prepared))
    });
    tokio::spawn(async move {
        let result = match worker.await {
            Ok(result) => result.map(|_| ()),
            Err(error) => Err(format!("benchmark worker stopped: {error}")),
        };
        if let Err(error) = result {
            let store = state.store.lock().await;
            let summary = store
                .run(&run_id)
                .ok()
                .flatten()
                .map(|run| run.summary)
                .unwrap_or_default();
            if let Err(error) = store.finish_run(
                &run_id,
                RunStatus::Failed,
                unix_time_ms(),
                Some(&error),
                &summary,
            ) {
                tracing::error!("{error}");
            }
        }
        drop(permit);
    });
    Ok((StatusCode::ACCEPTED, Json(run)))
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
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn conflict(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::CONFLICT,
            message: message.into(),
        }
    }

    fn forbidden(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            message: message.into(),
        }
    }
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
