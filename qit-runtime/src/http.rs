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
use tokio::sync::{watch, Mutex, Semaphore};
use uuid::Uuid;

use crate::catalog::{owned_package, owned_packages};
use crate::config::{SessionShape, DEFAULT_N_CTX, DEFAULT_N_PARALLEL};
use crate::estimate::{classify, estimate_bytes, Fit};
use crate::paths::Paths;
use crate::probe::{budget_bytes, resolve_os_reserve, HardwareProbe, HardwareSnapshot};
use crate::scan::scan_library;
use crate::serve::{RuntimeRecipe, ServeProfile};
use crate::spa::index_html;
use crate::store::{ArtifactRow, MeasurementRow, PinRow, SessionRow, Store};
use crate::supervisor::{
    proxy_generate, session_status_str, ChatMessage, SessionStatus, SessionView, Supervisor,
};

const DEFAULT_MAX_TOKENS: u32 = 512;

#[derive(Clone)]
pub struct AppState {
    pub paths: Paths,
    pub store: Arc<Mutex<Store>>,
    pub probe: Arc<dyn HardwareProbe>,
    pub os_reserve_override: Option<u64>,
    pub worker_path: Option<PathBuf>,
    pub supervisor: Arc<Supervisor>,
    pub what_ifs: Arc<Mutex<Vec<PinRow>>>,
    pub generate_slot: Arc<Semaphore>,
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
    pub packages: Vec<PackageBody>,
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
    pub kind: String,
    pub generate_supported: bool,
}

#[derive(Serialize)]
pub struct PackageBody {
    pub id: String,
    pub family: String,
    pub name: String,
    pub format: String,
    pub estimate_bytes: u64,
    pub estimate_source: String,
    pub estimate_confidence: String,
    pub runtime_recipe: String,
    pub fits: bool,
    pub ready: bool,
    pub readiness_reason: Option<ReadinessReason>,
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadinessReason {
    MissingRequiredFiles,
    InsufficientMemory,
    RuntimeMissing,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub package_id: Option<String>,
    pub runtime_recipe: RuntimeRecipe,
    pub serve_profile: ServeProfile,
    pub estimate_bytes: u64,
}

#[derive(Deserialize)]
pub struct GenerateBody {
    pub artifact_id: Option<String>,
    pub package_id: Option<String>,
    pub serve_profile: Option<ServeProfile>,
    pub prompt: Option<String>,
    pub messages: Option<Vec<ChatMessage>>,
    pub max_tokens: Option<u32>,
    pub n_ctx: Option<u32>,
    pub n_gpu_layers: Option<i32>,
    pub n_parallel: Option<u32>,
    pub session_id: Option<String>,
}

impl GenerateBody {
    fn messages(&self) -> Result<Vec<ChatMessage>, ApiError> {
        match (&self.messages, &self.prompt) {
            (Some(messages), _) if !messages.is_empty() => Ok(messages.clone()),
            (_, Some(prompt)) => Ok(vec![ChatMessage {
                role: "user".into(),
                content: prompt.clone(),
            }]),
            _ => Err(ApiError::bad("messages or prompt is required")),
        }
    }

    fn session_shape(&self) -> SessionShape {
        SessionShape {
            artifact_id: self.artifact_id.clone(),
            package_id: self.package_id.clone(),
            serve_profile: self.serve_profile.clone(),
            n_ctx: self.n_ctx,
            n_gpu_layers: self.n_gpu_layers,
            n_parallel: self.n_parallel,
        }
    }
}

#[derive(Serialize)]
pub struct GenerateDone {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub n_ctx: u32,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/api/health", get(health))
        .route("/api/hardware", get(hardware))
        .route("/api/settings", get(settings).put(update_settings))
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
    let resolved = resolve_serve(&state, &store, &shape)?;
    let row = PinRow {
        id: Uuid::new_v4().to_string(),
        artifact_id: resolved.artifact.id.clone(),
        package_id: resolved.package_id,
        runtime_recipe: resolved.runtime_recipe,
        serve_profile: resolved.serve_profile,
    };
    let estimate = reservation_estimate(&resolved.artifact, &row);
    drop(store);
    state.what_ifs.lock().await.push(row.clone());
    Ok(Json(reservation_body(row, estimate)))
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
    let resolved = resolve_serve(&state, &store, &shape)?;
    let row = PinRow {
        id: Uuid::new_v4().to_string(),
        artifact_id: resolved.artifact.id.clone(),
        package_id: resolved.package_id,
        runtime_recipe: resolved.runtime_recipe,
        serve_profile: resolved.serve_profile,
    };
    store.insert_pin(&row).map_err(ApiError::from)?;
    let estimate = reservation_estimate(&resolved.artifact, &row);
    Ok(Json(reservation_body(row, estimate)))
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
    let resolved = resolve_serve(&state, &store, &shape)?;
    drop(store);
    let log = state.paths.worker_log(&Uuid::new_v4().to_string());
    let artifact_id = resolved.artifact.id.clone();
    let package_id = resolved.package_id.clone();
    let runtime_recipe = resolved.runtime_recipe;
    let serve_profile = resolved.serve_profile.clone();
    let result = state
        .supervisor
        .start(
            &resolved.artifact,
            package_id.clone(),
            runtime_recipe,
            serve_profile.clone(),
            log,
        )
        .await
        .map_err(|e| {
            tracing::warn!(
                artifact_id = %artifact_id,
                error = %e,
                "session start failed"
            );
            ApiError::bad(e)
        });
    persist_session_profile(&state, &artifact_id, package_id.as_deref(), &serve_profile).await;
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
    let stopped = state.supervisor.stop(&id).await.map_err(ApiError::bad)?;
    persist_session(&state, &stopped.view).await;
    Ok(Json(stopped.view))
}

async fn generate(
    State(state): State<AppState>,
    Json(body): Json<GenerateBody>,
) -> Result<Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>> + Send>, ApiError> {
    let messages = body.messages()?;
    let max_tokens = body.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    let store = state.store.lock().await;
    let resolved = resolve_serve(&state, &store, &body.session_shape())?;
    if !resolved.artifact.kind.generate_supported() {
        return Err(ApiError::bad(format!(
            "Try is only for instruct artifacts, this one is {}",
            resolved.artifact.kind.as_str()
        )));
    }
    drop(store);
    let n_ctx = resolved.serve_profile.context_length;
    let slot = state
        .generate_slot
        .clone()
        .try_acquire_owned()
        .map_err(|_| ApiError::conflict("generate in flight"))?;

    let mut ephemeral = false;
    let session = if let Some(id) = &body.session_id {
        state
            .supervisor
            .get(id)
            .await
            .filter(|session| {
                session.status == SessionStatus::Loaded
                    && session.artifact_id == resolved.artifact.id
                    && session.package_id == resolved.package_id
                    && session.serve_profile == resolved.serve_profile
            })
            .ok_or_else(|| {
                ApiError::bad("session is not loaded for the requested target and serve profile")
            })?
    } else if let Some(existing) = state
        .supervisor
        .find_loaded(
            &resolved.artifact.id,
            resolved.package_id.as_deref(),
            &resolved.serve_profile,
        )
        .await
    {
        existing
    } else {
        ephemeral = true;
        let log = state.paths.worker_log(&Uuid::new_v4().to_string());
        state
            .supervisor
            .start(
                &resolved.artifact,
                resolved.package_id.clone(),
                resolved.runtime_recipe,
                resolved.serve_profile.clone(),
                log,
            )
            .await
            .map_err(|e| {
                tracing::warn!(
                    artifact_id = %resolved.artifact.id,
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
    let artifact_id = resolved.artifact.id.clone();
    let runtime = state.clone();
    let session_id = session.id.clone();

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(32);
    tokio::spawn(async move {
        let token_tx = tx.clone();
        let result = proxy_generate(
            &base_url,
            &messages,
            max_tokens,
            cancel_rx,
            move |token| {
                let token_tx = token_tx.clone();
                async move {
                    token_tx
                        .send(Ok(Event::default().event("token").data(token)))
                        .await
                        .map_err(|_| "client closed".to_string())
                }
            },
        )
        .await;
        match result {
            Ok(outcome) => {
                let tps = if outcome.generation_ms > 0.0 {
                    Some((outcome.n_tokens as f64) / (outcome.generation_ms / 1000.0))
                } else {
                    None
                };
                let peak = if ephemeral {
                    runtime
                        .supervisor
                        .stop(&session_id)
                        .await
                        .ok()
                        .and_then(|s| s.peak_rss_bytes)
                } else {
                    runtime.supervisor.sample_resident(&session_id).await
                };
                let store = runtime.store.lock().await;
                let _ = store.upsert_measurement(&MeasurementRow {
                    artifact_id,
                    throughput_tps: tps,
                    peak_rss_bytes: peak,
                    n_tokens: Some(outcome.n_tokens),
                    generation_ms: Some(outcome.generation_ms),
                });
                drop(store);
                let usage = outcome.usage.unwrap_or_default();
                let done = GenerateDone {
                    prompt_tokens: usage.prompt_tokens,
                    completion_tokens: usage.completion_tokens.max(outcome.n_tokens),
                    n_ctx,
                };
                let data = serde_json::to_string(&done).unwrap_or_default();
                let _ = tx.send(Ok(Event::default().event("done").data(data))).await;
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
        drop(slot);
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
                .map(|artifact| reservation_estimate(artifact, row))
                .unwrap_or(0);
            reservation_body(row.clone(), estimate)
        })
        .collect()
}

fn reservation_body(row: PinRow, estimate_bytes: u64) -> ReservationBody {
    ReservationBody {
        id: row.id,
        artifact_id: row.artifact_id,
        package_id: row.package_id,
        runtime_recipe: row.runtime_recipe,
        serve_profile: row.serve_profile,
        estimate_bytes,
    }
}

fn reservation_estimate(artifact: &ArtifactRow, row: &PinRow) -> u64 {
    if let Some(package) = row.package_id.as_deref().and_then(owned_package) {
        return package.estimate_bytes;
    }
    let parallel = row
        .serve_profile
        .llama_cpp_settings()
        .map(|settings| settings.parallel)
        .unwrap_or(DEFAULT_N_PARALLEL);
    estimate_bytes(artifact, row.serve_profile.context_length, parallel)
}

struct ResolvedServe {
    artifact: ArtifactRow,
    package_id: Option<String>,
    runtime_recipe: RuntimeRecipe,
    serve_profile: ServeProfile,
}

fn resolve_serve(
    state: &AppState,
    store: &Store,
    shape: &SessionShape,
) -> Result<ResolvedServe, ApiError> {
    let mut serve_profile = shape.profile();
    let (artifact, package_id, runtime_recipe) = match (&shape.package_id, &shape.artifact_id) {
        (Some(_), Some(_)) => {
            return Err(ApiError::bad(
                "specify exactly one of package_id or artifact_id",
            ));
        }
        (Some(package_id), None) => {
            let package = owned_package(package_id)
                .ok_or_else(|| ApiError::not_found("model package not found"))?;
            if !package.has_required_files(&state.paths.models_dir) {
                return Err(ApiError::bad("model package is missing required files"));
            }
            let artifact_id = package
                .primary_artifact_id()
                .ok_or_else(|| ApiError::bad("model package has no servable files"))?;
            (
                require_artifact(store, &artifact_id)?,
                Some(package_id.clone()),
                package.runtime_recipe,
            )
        }
        (None, Some(artifact_id)) => (
            require_artifact(store, artifact_id)?,
            None,
            RuntimeRecipe::LlamaCpp,
        ),
        (None, None) => {
            return Err(ApiError::bad(
                "specify exactly one of package_id or artifact_id",
            ));
        }
    };
    match runtime_recipe {
        RuntimeRecipe::LlamaCpp => {
            let settings = serve_profile.llama_cpp_settings().map_err(ApiError::bad)?;
            if settings.parallel == 0 {
                return Err(ApiError::bad("parallel must be at least 1"));
            }
            serve_profile.runtime_settings = serde_json::to_value(settings)
                .map_err(|error| ApiError::bad(error.to_string()))?;
        }
        RuntimeRecipe::TransformersExternal => {
            return Err(ApiError::bad("model package runtime is not available"));
        }
    }
    validate_n_ctx(&artifact, serve_profile.context_length)?;
    Ok(ResolvedServe {
        artifact,
        package_id,
        runtime_recipe,
        serve_profile,
    })
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
            kind: artifact.kind.as_str().into(),
            generate_supported: artifact.kind.generate_supported(),
        });
    }
    let packages = owned_packages()
        .into_iter()
        .map(|package| {
            let fits = package.planner_hint.estimate_bytes <= hw.headroom_bytes;
            let readiness_reason = if !package.has_required_files(&state.paths.models_dir) {
                Some(ReadinessReason::MissingRequiredFiles)
            } else if !fits {
                Some(ReadinessReason::InsufficientMemory)
            } else if !runtime_available(&state, package.runtime_recipe) {
                Some(ReadinessReason::RuntimeMissing)
            } else {
                None
            };
            PackageBody {
                id: package.id.into(),
                family: package.family.into(),
                name: package.name.into(),
                format: package.format.as_str().into(),
                estimate_bytes: package.planner_hint.estimate_bytes,
                estimate_source: package.planner_hint.source.into(),
                estimate_confidence: package.planner_hint.confidence.into(),
                runtime_recipe: package.runtime_recipe.as_str().into(),
                fits,
                ready: readiness_reason.is_none(),
                readiness_reason,
            }
        })
        .collect();
    Ok(CatalogBody {
        artifacts: list,
        packages,
    })
}

#[derive(Serialize)]
pub struct SettingsBody {
    pub os_reserve_bytes: Option<u64>,
    pub os_reserve_source: OsReserveSource,
    pub effective_os_reserve_bytes: u64,
}

#[derive(Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OsReserveSource {
    Env,
    Setting,
    Default,
}

#[derive(Deserialize)]
pub struct SettingsUpdate {
    pub os_reserve_bytes: Option<u64>,
}

struct OsReserve {
    setting: Option<u64>,
    source: OsReserveSource,
    effective: u64,
}

fn os_reserve(state: &AppState, store: &Store, unified_memory_bytes: u64) -> Result<OsReserve, ApiError> {
    let setting = store.os_reserve_setting().map_err(ApiError::from)?;
    let (source, chosen) = match (state.os_reserve_override, setting) {
        (Some(env), _) => (OsReserveSource::Env, Some(env)),
        (None, Some(s)) => (OsReserveSource::Setting, Some(s)),
        (None, None) => (OsReserveSource::Default, None),
    };
    Ok(OsReserve {
        setting,
        source,
        effective: resolve_os_reserve(unified_memory_bytes, chosen),
    })
}

async fn settings(State(state): State<AppState>) -> Result<Json<SettingsBody>, ApiError> {
    let snap = state.probe.probe();
    let store = state.store.lock().await;
    let reserve = os_reserve(&state, &store, snap.unified_memory_bytes)?;
    Ok(Json(settings_body(reserve)))
}

async fn update_settings(
    State(state): State<AppState>,
    Json(update): Json<SettingsUpdate>,
) -> Result<Json<SettingsBody>, ApiError> {
    let snap = state.probe.probe();
    if let Some(bytes) = update.os_reserve_bytes {
        if bytes >= snap.unified_memory_bytes {
            return Err(ApiError::bad(format!(
                "os_reserve_bytes {bytes} must be below unified memory {}",
                snap.unified_memory_bytes
            )));
        }
    }
    let store = state.store.lock().await;
    store
        .set_os_reserve_setting(update.os_reserve_bytes)
        .map_err(ApiError::from)?;
    let reserve = os_reserve(&state, &store, snap.unified_memory_bytes)?;
    Ok(Json(settings_body(reserve)))
}

fn settings_body(reserve: OsReserve) -> SettingsBody {
    SettingsBody {
        os_reserve_bytes: reserve.setting,
        os_reserve_source: reserve.source,
        effective_os_reserve_bytes: reserve.effective,
    }
}

async fn hardware_body(state: &AppState) -> Result<HardwareBody, ApiError> {
    state.supervisor.reap().await;
    let snap: HardwareSnapshot = state.probe.probe();
    let store = state.store.lock().await;
    let os_reserve_bytes = os_reserve(state, &store, snap.unified_memory_bytes)?.effective;
    let budget = budget_bytes(&snap, os_reserve_bytes);
    let artifacts = store.artifacts().map_err(ApiError::from)?;
    let pins = store.pins().map_err(ApiError::from)?;
    drop(store);
    let what_ifs = state.what_ifs.lock().await.clone();
    let sessions = state.supervisor.list().await;
    let mut used = 0u64;
    for row in pins.iter().chain(what_ifs.iter()) {
        if let Some(a) = artifacts.iter().find(|a| a.id == row.artifact_id) {
            used = used.saturating_add(reservation_estimate(a, row));
        }
    }
    for session in sessions
        .iter()
        .filter(|s| matches!(s.status, SessionStatus::Loaded | SessionStatus::Starting))
    {
        let already_pinned = pins.iter().any(|p| {
            p.artifact_id == session.artifact_id
                && p.package_id == session.package_id
                && p.serve_profile == session.serve_profile
        });
        if already_pinned {
            continue;
        }
        if let Some(a) = artifacts.iter().find(|a| a.id == session.artifact_id) {
            let row = PinRow {
                id: session.id.clone(),
                artifact_id: session.artifact_id.clone(),
                package_id: session.package_id.clone(),
                runtime_recipe: session.runtime_recipe,
                serve_profile: session.serve_profile.clone(),
            };
            used = used.saturating_add(reservation_estimate(a, &row));
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

fn runtime_available(state: &AppState, runtime_recipe: RuntimeRecipe) -> bool {
    match runtime_recipe {
        RuntimeRecipe::LlamaCpp => state.worker_path.is_some(),
        RuntimeRecipe::TransformersExternal => false,
    }
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

    fn conflict(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::CONFLICT,
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
        package_id: view.package_id.clone(),
        runtime_recipe: view.runtime_recipe,
        serve_profile: view.serve_profile.clone(),
        status: session_status_str(view.status).to_string(),
        last_error: view.last_error.clone(),
        log_path: view.log_path.clone(),
    };
    let store = state.store.lock().await;
    let _ = store.upsert_session(&row);
}

async fn persist_session_profile(
    state: &AppState,
    artifact_id: &str,
    package_id: Option<&str>,
    serve_profile: &ServeProfile,
) {
    if let Some(view) = state
        .supervisor
        .find_by_profile(artifact_id, package_id, serve_profile)
        .await
    {
        persist_session(state, &view).await;
    }
}
