use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use std::path::PathBuf;

use axum::extract::{Path, Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use futures_util::stream;
use serde::{Deserialize, Serialize};
use tokio::sync::{watch, Mutex};
use uuid::Uuid;

use crate::config::{SessionShape, DEFAULT_N_CTX, DEFAULT_N_GPU_LAYERS, DEFAULT_N_PARALLEL};
use crate::estimate::{classify, estimate_bytes, Fit};
use crate::paths::Paths;
use crate::probe::{budget_bytes, resolve_os_reserve, HardwareProbe, HardwareSnapshot};
use crate::scan::scan_library;
use crate::spa::index_html;
use crate::store::{ArtifactRow, MeasurementRow, PinRow, SessionRow, Store};
use crate::supervisor::{proxy_generate, session_status_str, SessionStatus, SessionView, Supervisor};

#[derive(Clone)]
pub struct AppState {
    pub paths: Paths,
    pub store: Arc<Mutex<Store>>,
    pub probe: Arc<dyn HardwareProbe>,
    pub os_reserve_override: Option<u64>,
    pub worker_path: Option<PathBuf>,
    pub supervisor: Arc<Supervisor>,
    pub what_ifs: Arc<Mutex<Vec<PinRow>>>,
}

#[derive(Deserialize)]
pub struct CtxQuery {
    n_ctx: Option<u32>,
}

#[derive(Serialize)]
pub struct HealthBody {
    pub ok: bool,
}

#[derive(Serialize)]
pub struct HardwareBody {
    pub device_class: String,
    pub chip: String,
    pub unified_memory_bytes: u64,
    pub metal_recommended_working_set_bytes: Option<u64>,
    pub os_reserve_bytes: u64,
    pub budget_bytes: u64,
    pub headroom_bytes: u64,
    pub memory_pressure: Option<String>,
    pub free_ram_bytes: Option<u64>,
    pub loaded_rss_bytes: u64,
    pub worker_path: Option<String>,
}

#[derive(Serialize)]
pub struct CatalogBody {
    pub artifacts: Vec<ArtifactBody>,
}

#[derive(Serialize)]
pub struct ArtifactBody {
    pub id: String,
    pub org: String,
    pub filename: String,
    pub bytes: u64,
    pub architecture: Option<String>,
    pub context_length: Option<u32>,
    pub block_count: Option<u32>,
    pub head_count: Option<u32>,
    pub confidence: String,
    pub estimate_bytes: u64,
    pub fit: Fit,
    pub throughput_tps: Option<f64>,
    pub peak_rss_bytes: Option<u64>,
}

#[derive(Serialize)]
pub struct CapacityBody {
    pub hardware: HardwareBody,
    pub pins: Vec<ReservationBody>,
    pub what_ifs: Vec<ReservationBody>,
    pub sessions: Vec<SessionView>,
}

#[derive(Serialize)]
pub struct ReservationBody {
    pub id: String,
    pub artifact_id: String,
    pub n_ctx: u32,
    pub n_gpu_layers: i32,
    pub n_parallel: u32,
    pub estimate_bytes: u64,
}

#[derive(Deserialize)]
pub struct GenerateBody {
    pub artifact_id: String,
    pub prompt: String,
    pub n_ctx: Option<u32>,
    pub n_gpu_layers: Option<i32>,
    pub n_parallel: Option<u32>,
    pub session_id: Option<String>,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/api/health", get(health))
        .route("/api/hardware", get(hardware))
        .route("/api/scan", post(scan))
        .route("/api/catalog", get(catalog))
        .route("/api/capacity", get(capacity))
        .route("/api/what-ifs", post(add_what_if).delete(clear_what_ifs))
        .route("/api/what-ifs/{id}", delete(delete_what_if))
        .route("/api/pins", post(add_pin))
        .route("/api/pins/{id}", delete(delete_pin))
        .route("/api/sessions", get(list_sessions).post(start_session))
        .route("/api/sessions/{id}", delete(delete_session))
        .route("/api/sessions/{id}/stop", post(stop_session))
        .route("/api/generate", post(generate))
        .fallback(get(spa_or_asset))
        .with_state(state)
}

async fn spa_or_asset(axum::extract::OriginalUri(uri): axum::extract::OriginalUri) -> Response {
    if let Some(dir) = web_dist_dir() {
        let rel = uri.path().trim_start_matches('/');
        let file = if rel.is_empty() {
            dir.join("index.html")
        } else {
            dir.join(rel)
        };
        if file.is_file() {
            if let Ok(bytes) = std::fs::read(&file) {
                let mime = mime_for(&file);
                return ([(header::CONTENT_TYPE, mime)], bytes).into_response();
            }
        }
        if let Ok(bytes) = std::fs::read(dir.join("index.html")) {
            return ([(header::CONTENT_TYPE, "text/html; charset=utf-8")], bytes).into_response();
        }
    }
    Html(index_html()).into_response()
}

fn web_dist_dir() -> Option<PathBuf> {
    if let Ok(from_env) = std::env::var("QIT_WEB_DIST") {
        let path = PathBuf::from(from_env);
        if path.join("index.html").is_file() {
            return Some(path);
        }
    }
    let nested = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../qit-web/dist");
    nested.join("index.html").is_file().then_some(nested)
}

fn mime_for(path: &std::path::Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("js") => "text/javascript",
        Some("css") => "text/css",
        Some("html") => "text/html; charset=utf-8",
        Some("svg") => "image/svg+xml",
        Some("json") => "application/json",
        _ => "application/octet-stream",
    }
}

async fn health() -> Json<HealthBody> {
    Json(HealthBody { ok: true })
}

async fn hardware(State(state): State<AppState>) -> Result<Json<HardwareBody>, ApiError> {
    Ok(Json(hardware_body(&state).await?))
}

async fn scan(State(state): State<AppState>) -> Result<Json<CatalogBody>, ApiError> {
    rescan(&state).await?;
    catalog_body(&state, DEFAULT_N_CTX).await.map(Json)
}

async fn catalog(
    State(state): State<AppState>,
    Query(q): Query<CtxQuery>,
) -> Result<Json<CatalogBody>, ApiError> {
    catalog_body(&state, q.n_ctx.unwrap_or(DEFAULT_N_CTX))
        .await
        .map(Json)
}

async fn capacity(State(state): State<AppState>) -> Result<Json<CapacityBody>, ApiError> {
    let hardware = hardware_body(&state).await?;
    let store = state.store.lock().await;
    let artifacts = store.artifacts().map_err(ApiError::from)?;
    let pins = store.pins().map_err(ApiError::from)?;
    drop(store);
    let what_ifs = state.what_ifs.lock().await.clone();
    let sessions = state.supervisor.list().await;
    Ok(Json(CapacityBody {
        hardware,
        pins: map_reservations(&artifacts, &pins),
        what_ifs: map_reservations(&artifacts, &what_ifs),
        sessions,
    }))
}

async fn add_what_if(
    State(state): State<AppState>,
    Json(shape): Json<SessionShape>,
) -> Result<Json<ReservationBody>, ApiError> {
    let store = state.store.lock().await;
    let artifact = require_artifact(&store, &shape.artifact_id)?;
    validate_n_ctx(&artifact, shape.n_ctx())?;
    let row = PinRow {
        id: Uuid::new_v4().to_string(),
        artifact_id: shape.artifact_id.clone(),
        n_ctx: shape.n_ctx(),
        n_gpu_layers: shape.n_gpu_layers(),
        n_parallel: shape.n_parallel(),
    };
    let estimate = estimate_bytes(&artifact, row.n_ctx, row.n_parallel);
    drop(store);
    state.what_ifs.lock().await.push(row.clone());
    Ok(Json(ReservationBody {
        id: row.id,
        artifact_id: row.artifact_id,
        n_ctx: row.n_ctx,
        n_gpu_layers: row.n_gpu_layers,
        n_parallel: row.n_parallel,
        estimate_bytes: estimate,
    }))
}

async fn clear_what_ifs(State(state): State<AppState>) -> StatusCode {
    state.what_ifs.lock().await.clear();
    StatusCode::NO_CONTENT
}

async fn delete_what_if(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    let mut what_ifs = state.what_ifs.lock().await;
    let before = what_ifs.len();
    what_ifs.retain(|w| w.id != id);
    if what_ifs.len() == before {
        return Err(ApiError::not_found("what-if not found"));
    }
    Ok(StatusCode::NO_CONTENT)
}

async fn add_pin(
    State(state): State<AppState>,
    Json(shape): Json<SessionShape>,
) -> Result<Json<ReservationBody>, ApiError> {
    let store = state.store.lock().await;
    let artifact = require_artifact(&store, &shape.artifact_id)?;
    validate_n_ctx(&artifact, shape.n_ctx())?;
    let row = PinRow {
        id: Uuid::new_v4().to_string(),
        artifact_id: shape.artifact_id.clone(),
        n_ctx: shape.n_ctx(),
        n_gpu_layers: shape.n_gpu_layers(),
        n_parallel: shape.n_parallel(),
    };
    store.insert_pin(&row).map_err(ApiError::from)?;
    let estimate = estimate_bytes(&artifact, row.n_ctx, row.n_parallel);
    Ok(Json(ReservationBody {
        id: row.id,
        artifact_id: row.artifact_id,
        n_ctx: row.n_ctx,
        n_gpu_layers: row.n_gpu_layers,
        n_parallel: row.n_parallel,
        estimate_bytes: estimate,
    }))
}

async fn delete_pin(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    let store = state.store.lock().await;
    if !store.delete_pin(&id).map_err(ApiError::from)? {
        return Err(ApiError::not_found("pin not found"));
    }
    Ok(StatusCode::NO_CONTENT)
}

async fn list_sessions(State(state): State<AppState>) -> Json<Vec<SessionView>> {
    state.supervisor.reap().await;
    Json(state.supervisor.list().await)
}

async fn start_session(
    State(state): State<AppState>,
    Json(shape): Json<SessionShape>,
) -> Result<Json<SessionView>, ApiError> {
    let store = state.store.lock().await;
    let artifact = require_artifact(&store, &shape.artifact_id)?;
    validate_n_ctx(&artifact, shape.n_ctx())?;
    drop(store);
    let log = state.paths.worker_log(&Uuid::new_v4().to_string());
    let n_ctx = shape.n_ctx();
    let n_gpu_layers = shape.n_gpu_layers();
    let n_parallel = shape.n_parallel();
    let result = state
        .supervisor
        .start(
            &artifact,
            n_ctx,
            n_gpu_layers,
            n_parallel,
            log,
        )
        .await
        .map_err(|e| {
            tracing::warn!(
                artifact_id = %shape.artifact_id,
                error = %e,
                "session start failed"
            );
            ApiError::bad(e)
        });
    persist_session_tuple(
        &state,
        &shape.artifact_id,
        n_ctx,
        n_gpu_layers,
        n_parallel,
    )
    .await;
    result.map(Json)
}

async fn delete_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    state.supervisor.remove(&id).await.map_err(ApiError::bad)?;
    let store = state.store.lock().await;
    if !store.delete_session(&id).map_err(ApiError::from)? {
        return Err(ApiError::not_found("session not found"));
    }
    Ok(StatusCode::NO_CONTENT)
}

async fn stop_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<SessionView>, ApiError> {
    let view = state.supervisor.stop(&id).await.map_err(ApiError::bad)?;
    persist_session(&state, &view).await;
    Ok(Json(view))
}

async fn generate(
    State(state): State<AppState>,
    Json(body): Json<GenerateBody>,
) -> Result<Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>> + Send>, ApiError> {
    let n_ctx = body.n_ctx.unwrap_or(DEFAULT_N_CTX);
    let n_gpu_layers = body.n_gpu_layers.unwrap_or(DEFAULT_N_GPU_LAYERS);
    let n_parallel = body.n_parallel.unwrap_or(DEFAULT_N_PARALLEL);
    let store = state.store.lock().await;
    let artifact = require_artifact(&store, &body.artifact_id)?;
    validate_n_ctx(&artifact, n_ctx)?;
    drop(store);

    let mut ephemeral = false;
    let session = if let Some(id) = &body.session_id {
        state
            .supervisor
            .get(id)
            .await
            .filter(|s| s.status == SessionStatus::Loaded)
            .ok_or_else(|| ApiError::bad("session is not loaded"))?
    } else if let Some(existing) = state
        .supervisor
        .find_loaded(&artifact.id, n_ctx, n_gpu_layers, n_parallel)
        .await
    {
        existing
    } else {
        ephemeral = true;
        let log = state.paths.worker_log(&Uuid::new_v4().to_string());
        state
            .supervisor
            .start(&artifact, n_ctx, n_gpu_layers, n_parallel, log)
            .await
            .map_err(|e| {
                tracing::warn!(
                    artifact_id = %body.artifact_id,
                    error = %e,
                    "generate worker start failed"
                );
                ApiError::bad(e)
            })?
    };

    if session.status != SessionStatus::Loaded {
        return Err(ApiError::bad("worker failed to load"));
    }

    let base_url = state
        .supervisor
        .base_url(&session.id)
        .await
        .ok_or_else(|| ApiError::bad("worker has no endpoint"))?;

    let (cancel_tx, cancel_rx) = watch::channel(false);
    let prompt = body.prompt.clone();
    let artifact_id = artifact.id.clone();
    let runtime = state.clone();
    let session_id = session.id.clone();

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(32);
    tokio::spawn(async move {
        let token_tx = tx.clone();
        let result = proxy_generate(&base_url, &prompt, cancel_rx, move |token| {
            let token_tx = token_tx.clone();
            async move {
                token_tx
                    .send(Ok(Event::default().event("token").data(token)))
                    .await
                    .map_err(|_| "client closed".to_string())
            }
        })
        .await;
        match result {
            Ok((n_tokens, generation_ms)) => {
                let tps = if generation_ms > 0.0 {
                    Some((n_tokens as f64) / (generation_ms / 1000.0))
                } else {
                    None
                };
                let store = runtime.store.lock().await;
                let _ = store.upsert_measurement(&MeasurementRow {
                    artifact_id,
                    throughput_tps: tps,
                    peak_rss_bytes: None,
                    n_tokens: Some(n_tokens),
                    generation_ms: Some(generation_ms),
                });
                drop(store);
                let _ = tx.send(Ok(Event::default().event("done").data(""))).await;
            }
            Err(e) => {
                if e != "client closed" {
                    tracing::warn!(
                        artifact_id = %artifact_id,
                        error = %e,
                        "generate failed"
                    );
                    let _ = tx.send(Ok(Event::default().event("error").data(e))).await;
                }
            }
        }
        if ephemeral {
            let _ = runtime.supervisor.stop(&session_id).await;
        }
        drop(cancel_tx);
    });

    let sse = Sse::new(stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|item| (item, rx))
    }))
    .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)));
    Ok(sse)
}

fn map_reservations(artifacts: &[ArtifactRow], rows: &[PinRow]) -> Vec<ReservationBody> {
    rows.iter()
        .map(|row| {
            let estimate = artifacts
                .iter()
                .find(|a| a.id == row.artifact_id)
                .map(|a| estimate_bytes(a, row.n_ctx, row.n_parallel))
                .unwrap_or(0);
            ReservationBody {
                id: row.id.clone(),
                artifact_id: row.artifact_id.clone(),
                n_ctx: row.n_ctx,
                n_gpu_layers: row.n_gpu_layers,
                n_parallel: row.n_parallel,
                estimate_bytes: estimate,
            }
        })
        .collect()
}

fn require_artifact(store: &Store, id: &str) -> Result<ArtifactRow, ApiError> {
    store
        .artifact(id)
        .map_err(ApiError::from)?
        .ok_or_else(|| ApiError::not_found("artifact not found"))
}

fn validate_n_ctx(artifact: &ArtifactRow, n_ctx: u32) -> Result<(), ApiError> {
    if let Some(max) = artifact.context_length {
        if n_ctx > max {
            return Err(ApiError::bad(format!(
                "n_ctx {n_ctx} exceeds model max {max}"
            )));
        }
    }
    Ok(())
}

pub async fn rescan(state: &AppState) -> Result<(), ApiError> {
    let rows = scan_library(&state.paths.models_dir);
    let store = state.store.lock().await;
    store.replace_artifacts(&rows).map_err(ApiError::from)?;
    Ok(())
}

async fn catalog_body(state: &AppState, n_ctx: u32) -> Result<CatalogBody, ApiError> {
    let hw = hardware_body(state).await?;
    let store = state.store.lock().await;
    let artifacts = store.artifacts().map_err(ApiError::from)?;
    let mut list = Vec::new();
    for artifact in artifacts {
        let estimate = estimate_bytes(&artifact, n_ctx, DEFAULT_N_PARALLEL);
        let fit = classify(estimate, hw.headroom_bytes);
        let measurement = store.measurement(&artifact.id).map_err(ApiError::from)?;
        list.push(ArtifactBody {
            id: artifact.id,
            org: artifact.org,
            filename: artifact.filename,
            bytes: artifact.bytes,
            architecture: artifact.architecture,
            context_length: artifact.context_length,
            block_count: artifact.block_count,
            head_count: artifact.head_count,
            confidence: artifact.confidence,
            estimate_bytes: estimate,
            fit,
            throughput_tps: measurement.as_ref().and_then(|m| m.throughput_tps),
            peak_rss_bytes: measurement.as_ref().and_then(|m| m.peak_rss_bytes),
        });
    }
    Ok(CatalogBody { artifacts: list })
}

async fn hardware_body(state: &AppState) -> Result<HardwareBody, ApiError> {
    state.supervisor.reap().await;
    let snap: HardwareSnapshot = state.probe.probe();
    let os_reserve_bytes = resolve_os_reserve(snap.unified_memory_bytes, state.os_reserve_override);
    let budget = budget_bytes(&snap, os_reserve_bytes);
    let store = state.store.lock().await;
    let artifacts = store.artifacts().map_err(ApiError::from)?;
    let pins = store.pins().map_err(ApiError::from)?;
    drop(store);
    let what_ifs = state.what_ifs.lock().await.clone();
    let sessions = state.supervisor.list().await;
    let mut used = 0u64;
    for row in pins.iter().chain(what_ifs.iter()) {
        if let Some(a) = artifacts.iter().find(|a| a.id == row.artifact_id) {
            used = used.saturating_add(estimate_bytes(a, row.n_ctx, row.n_parallel));
        }
    }
    for session in sessions
        .iter()
        .filter(|s| matches!(s.status, SessionStatus::Loaded | SessionStatus::Starting))
    {
        let already_pinned = pins.iter().any(|p| {
            p.artifact_id == session.artifact_id
                && p.n_ctx == session.n_ctx
                && p.n_gpu_layers == session.n_gpu_layers
                && p.n_parallel == session.n_parallel
        });
        if already_pinned {
            continue;
        }
        if let Some(a) = artifacts.iter().find(|a| a.id == session.artifact_id) {
            used = used.saturating_add(estimate_bytes(a, session.n_ctx, session.n_parallel));
        }
    }
    Ok(HardwareBody {
        device_class: snap.device_class,
        chip: snap.chip,
        unified_memory_bytes: snap.unified_memory_bytes,
        metal_recommended_working_set_bytes: snap.metal_recommended_working_set_bytes,
        os_reserve_bytes,
        budget_bytes: budget,
        headroom_bytes: budget.saturating_sub(used),
        memory_pressure: snap.memory_pressure,
        free_ram_bytes: snap.free_ram_bytes,
        loaded_rss_bytes: state.supervisor.loaded_rss_bytes().await,
        worker_path: state
            .worker_path
            .as_ref()
            .map(|p| p.display().to_string()),
    })
}

pub struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn bad(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: message.into(),
        }
    }
}

impl From<rusqlite::Error> for ApiError {
    fn from(value: rusqlite::Error) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: value.to_string(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let mut headers = HeaderMap::new();
        headers.insert(header::CONTENT_TYPE, "application/json".parse().unwrap());
        let body = serde_json::json!({ "error": self.message });
        (self.status, headers, body.to_string()).into_response()
    }
}

impl From<crate::error::Error> for ApiError {
    fn from(value: crate::error::Error) -> Self {
        Self::bad(value.to_string())
    }
}

impl std::fmt::Debug for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

async fn persist_session(state: &AppState, view: &SessionView) {
    let row = SessionRow {
        id: view.id.clone(),
        artifact_id: view.artifact_id.clone(),
        n_ctx: view.n_ctx,
        n_gpu_layers: view.n_gpu_layers,
        n_parallel: view.n_parallel,
        status: session_status_str(view.status).to_string(),
        last_error: view.last_error.clone(),
        log_path: view.log_path.clone(),
    };
    let store = state.store.lock().await;
    let _ = store.upsert_session(&row);
}

async fn persist_session_tuple(
    state: &AppState,
    artifact_id: &str,
    n_ctx: u32,
    n_gpu_layers: i32,
    n_parallel: u32,
) {
    if let Some(view) = state
        .supervisor
        .find_by_tuple(artifact_id, n_ctx, n_gpu_layers, n_parallel)
        .await
    {
        persist_session(state, &view).await;
    }
}
