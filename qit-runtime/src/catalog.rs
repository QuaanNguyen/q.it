use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};
use std::sync::OnceLock;

use serde::Deserialize;

use crate::model::{ProviderKind, Task};

const CATALOG_JSON: &str = include_str!("model-catalog.generated.json");
const ADAPTER_BASE_URL: &str = "http://127.0.0.1:8766";

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ModelId(String);

impl ModelId {
    pub fn new(value: &str) -> Result<Self, String> {
        let trimmed = value.trim();
        if trimmed.is_empty() || trimmed.len() > 128 || trimmed.chars().any(char::is_control) {
            return Err(format!("invalid model identifier '{value}'"));
        }
        if trimmed != value {
            return Err(format!(
                "model identifier '{value}' has surrounding whitespace"
            ));
        }
        Ok(Self(trimmed.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Display for ModelId {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Implementation {
    Direct,
    Experimental,
    Pending,
}

impl Implementation {
    pub fn label(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Experimental => "adapter",
            Self::Pending => "needs adapter",
        }
    }

    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "direct" => Ok(Self::Direct),
            "experimental" => Ok(Self::Experimental),
            "pending" => Ok(Self::Pending),
            other => Err(format!("unknown implementation '{other}'")),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TaskSet(u16);

impl TaskSet {
    pub const EMPTY: Self = Self(0);

    pub fn of(task: Task) -> Self {
        Self(1 << task as u16)
    }

    pub fn from_tasks(tasks: impl IntoIterator<Item = Task>) -> Self {
        tasks
            .into_iter()
            .fold(Self::EMPTY, |set, task| Self(set.0 | Self::of(task).0))
    }

    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    pub fn contains(self, task: Task) -> bool {
        self.0 & Self::of(task).0 != 0
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub fn tasks(self) -> Vec<Task> {
        Task::ALL
            .into_iter()
            .filter(|task| self.contains(*task))
            .collect()
    }

    pub fn labels(self) -> String {
        self.tasks()
            .iter()
            .map(|task| task.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RouteKey {
    provider: ProviderKind,
    adapter: Option<String>,
    serve_as: String,
}

impl RouteKey {
    pub fn new(provider: ProviderKind, adapter: Option<&str>, serve_as: &str) -> Self {
        Self {
            provider,
            adapter: adapter.map(str::to_string),
            serve_as: serve_as.to_string(),
        }
    }

    pub fn provider(&self) -> ProviderKind {
        self.provider
    }

    pub fn adapter(&self) -> Option<&str> {
        self.adapter.as_deref()
    }

    pub fn serve_as(&self) -> &str {
        &self.serve_as
    }
}

#[derive(Clone, Debug)]
pub struct Route {
    key: RouteKey,
    model: ModelId,
    tasks: TaskSet,
    implementation: Implementation,
    qualification: Option<String>,
    precision: Option<String>,
    base_url: String,
}

impl Route {
    pub fn key(&self) -> &RouteKey {
        &self.key
    }

    pub fn provider(&self) -> ProviderKind {
        self.key.provider
    }

    pub fn model(&self) -> &ModelId {
        &self.model
    }

    pub fn adapter(&self) -> Option<&str> {
        self.key.adapter()
    }

    pub fn serve_as(&self) -> &str {
        self.key.serve_as()
    }

    pub fn tasks(&self) -> TaskSet {
        self.tasks
    }

    pub fn serves(&self, task: Task) -> bool {
        self.tasks.contains(task)
    }

    pub fn implementation(&self) -> Implementation {
        self.implementation
    }

    pub fn qualification(&self) -> Option<&str> {
        self.qualification.as_deref()
    }

    pub fn precision(&self) -> Option<&str> {
        self.precision.as_deref()
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

#[derive(Clone, Debug)]
pub struct Model {
    id: ModelId,
    parameters: Option<u64>,
    license: Option<String>,
    revision: String,
    routes: Vec<Route>,
}

impl Model {
    pub fn id(&self) -> &ModelId {
        &self.id
    }

    pub fn parameters(&self) -> Option<u64> {
        self.parameters
    }

    pub fn license(&self) -> Option<&str> {
        self.license.as_deref()
    }

    pub fn revision(&self) -> &str {
        &self.revision
    }

    pub fn routes(&self) -> &[Route] {
        &self.routes
    }

    pub fn tasks(&self) -> TaskSet {
        self.routes
            .iter()
            .fold(TaskSet::EMPTY, |set, route| set.union(route.tasks))
    }
}

#[derive(Clone, Debug)]
pub struct Catalog {
    models: Vec<Model>,
    routes: Vec<Route>,
    by_model: BTreeMap<ModelId, usize>,
    by_route: BTreeMap<RouteKey, usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Check {
    Compatible,
    Incompatible { declared: Vec<Task> },
    Unknown,
}

impl Catalog {
    pub fn builtin() -> Result<&'static Self, String> {
        static BUILT: OnceLock<Result<Catalog, String>> = OnceLock::new();
        BUILT
            .get_or_init(|| Self::from_json(CATALOG_JSON))
            .as_ref()
            .map_err(Clone::clone)
    }

    pub fn from_json(text: &str) -> Result<Self, String> {
        let document: Document =
            serde_json::from_str(text).map_err(|error| format!("parse model catalog: {error}"))?;
        if document.schema_version != 1 {
            return Err(format!(
                "model catalog uses schema version {}; this build reads 1",
                document.schema_version
            ));
        }
        let mut models = Vec::new();
        let mut routes = Vec::new();
        let mut by_model = BTreeMap::new();
        let mut by_route = BTreeMap::new();
        let mut route_keys = BTreeSet::new();
        for wire in document.models {
            let id = ModelId::new(&wire.id)?;
            if wire.revision.len() != 40
                || !wire.revision.bytes().all(|byte| byte.is_ascii_hexdigit())
            {
                return Err(format!(
                    "model '{}' must pin a forty character revision",
                    wire.id
                ));
            }
            if wire.routes.is_empty() {
                return Err(format!("model '{}' declares no routes", wire.id));
            }
            let mut model_routes = Vec::new();
            for wire_route in wire.routes {
                let provider = parse_provider(&wire_route.provider)?;
                let tasks = TaskSet::from_tasks(wire_route.tasks.iter().copied());
                if tasks.is_empty() {
                    return Err(format!("route for '{}' declares no tasks", wire.id));
                }
                for task in tasks.tasks() {
                    if !provider.supports(task) {
                        return Err(format!(
                            "catalog declares {} for {}, which has no {} transport",
                            task.as_str(),
                            provider.as_str(),
                            task.as_str()
                        ));
                    }
                }
                let key = RouteKey::new(
                    provider,
                    wire_route.adapter.as_deref(),
                    &wire_route.serve_as,
                );
                if !route_keys.insert(key.clone()) {
                    return Err(format!(
                        "duplicate route {} / {} / {}",
                        key.provider.as_str(),
                        key.adapter().unwrap_or("direct"),
                        key.serve_as()
                    ));
                }
                let base_url = if key.adapter().is_some() {
                    ADAPTER_BASE_URL.to_string()
                } else {
                    key.provider.default_base_url().to_string()
                };
                let route = Route {
                    key,
                    model: id.clone(),
                    tasks,
                    implementation: Implementation::parse(&wire_route.implementation)?,
                    qualification: wire_route.qualification,
                    precision: wire_route.precision,
                    base_url,
                };
                by_route.insert(route.key.clone(), routes.len());
                model_routes.push(route.clone());
                routes.push(route);
            }
            model_routes.sort_by(|left, right| {
                left.provider()
                    .cmp(&right.provider())
                    .then_with(|| left.adapter().cmp(&right.adapter()))
                    .then_with(|| left.serve_as().cmp(right.serve_as()))
            });
            by_model.insert(id.clone(), models.len());
            models.push(Model {
                id,
                parameters: wire.parameters,
                license: wire.license,
                revision: wire.revision,
                routes: model_routes,
            });
        }
        models.sort_by(|left, right| {
            left.parameters
                .is_none()
                .cmp(&right.parameters.is_none())
                .then_with(|| left.parameters.cmp(&right.parameters))
                .then_with(|| left.id.cmp(&right.id))
        });
        by_model.clear();
        for (index, model) in models.iter().enumerate() {
            by_model.insert(model.id.clone(), index);
        }
        Ok(Self {
            models,
            routes,
            by_model,
            by_route,
        })
    }

    pub fn models(&self) -> &[Model] {
        &self.models
    }

    pub fn model(&self, id: &ModelId) -> Option<&Model> {
        self.by_model.get(id).map(|index| &self.models[*index])
    }

    pub fn routes(&self) -> &[Route] {
        &self.routes
    }

    pub fn route(&self, key: &RouteKey) -> Option<&Route> {
        self.by_route.get(key).map(|index| &self.routes[*index])
    }

    pub fn routes_matching(
        &self,
        provider: Option<ProviderKind>,
        task: Option<Task>,
    ) -> Vec<&Route> {
        self.routes
            .iter()
            .filter(|route| provider.is_none_or(|value| route.provider() == value))
            .filter(|route| task.is_none_or(|value| route.serves(value)))
            .collect()
    }

    pub fn providers_for(&self, task: Option<Task>) -> Vec<ProviderKind> {
        ProviderKind::ALL
            .into_iter()
            .filter(|provider| {
                self.routes_matching(Some(*provider), task)
                    .into_iter()
                    .next()
                    .is_some()
            })
            .collect()
    }

    pub fn check(&self, task: Task, provider: ProviderKind, serve_as: &str) -> Check {
        let matches = self
            .routes
            .iter()
            .filter(|route| route.provider() == provider && route.serve_as() == serve_as)
            .collect::<Vec<_>>();
        if matches.is_empty() {
            return Check::Unknown;
        }
        if matches.iter().any(|route| route.serves(task)) {
            return Check::Compatible;
        }
        let declared = matches
            .iter()
            .fold(TaskSet::EMPTY, |set, route| set.union(route.tasks))
            .tasks();
        Check::Incompatible { declared }
    }

    pub fn parameters_label(&self, parameters: Option<u64>) -> String {
        match parameters {
            Some(value) if value >= 1_000_000 => format!("{:.1}M", value as f64 / 1_000_000.0),
            Some(value) => value.to_string(),
            None => "?".to_string(),
        }
    }
}

#[derive(Deserialize)]
struct Document {
    schema_version: u32,
    #[serde(rename = "sources")]
    _sources: Vec<Source>,
    models: Vec<WireModel>,
}

#[derive(Deserialize)]
struct Source {
    #[serde(rename = "name")]
    _name: String,
    #[serde(rename = "path")]
    _path: String,
    #[serde(rename = "sha256")]
    _sha256: String,
}

#[derive(Deserialize)]
struct WireModel {
    id: String,
    #[serde(default)]
    parameters: Option<u64>,
    #[serde(default)]
    license: Option<String>,
    revision: String,
    routes: Vec<WireRoute>,
}

#[derive(Deserialize)]
struct WireRoute {
    provider: String,
    serve_as: String,
    #[serde(default)]
    adapter: Option<String>,
    tasks: Vec<Task>,
    implementation: String,
    #[serde(default)]
    precision: Option<String>,
    #[serde(default)]
    qualification: Option<String>,
}

fn parse_provider(value: &str) -> Result<ProviderKind, String> {
    ProviderKind::ALL
        .into_iter()
        .find(|provider| provider.as_str() == value)
        .ok_or_else(|| format!("unknown provider '{value}' in model catalog"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn document(routes: &str) -> String {
        format!(
            r#"{{"schema_version":1,"sources":[],"models":[{{"id":"fixture/model","revision":"0123456789012345678901234567890123456789","parameters":1000000,"routes":[{routes}]}}]}}"#
        )
    }

    #[test]
    fn catalog_joins_routes_per_model_and_rejects_protocol_lies() {
        let catalog = Catalog::from_json(&document(
            r#"{"provider":"local-cpu","serve_as":"fixture/model","tasks":["text_to_speech"],"implementation":"direct"},
               {"provider":"ollama","serve_as":"fixture:1b","tasks":["text_generation"],"implementation":"direct","adapter":null}"#,
        ))
        .unwrap();
        let model = catalog.models().first().unwrap();
        assert_eq!(model.routes().len(), 2);
        assert!(model.tasks().contains(Task::TextToSpeech));
        assert_eq!(
            model
                .routes()
                .iter()
                .find(|route| route.provider() == ProviderKind::Ollama)
                .unwrap()
                .base_url(),
            ProviderKind::Ollama.default_base_url()
        );
        assert_eq!(
            catalog.check(Task::TextGeneration, ProviderKind::Ollama, "fixture:1b"),
            Check::Compatible
        );
        assert_eq!(
            catalog.check(Task::TextToSpeech, ProviderKind::Ollama, "fixture:1b"),
            Check::Incompatible {
                declared: vec![Task::TextGeneration]
            }
        );
        assert_eq!(
            catalog.check(Task::Embedding, ProviderKind::Tei, "fixture:1b"),
            Check::Unknown
        );
        assert!(Catalog::from_json(
            &document(r#"{"provider":"hf-serve","serve_as":"fixture/model","tasks":["text_to_speech"],"implementation":"direct"}"#)
        )
        .is_err());
    }

    #[test]
    fn adapter_routes_use_the_adapter_endpoint_and_pending_state() {
        let catalog = Catalog::from_json(&document(
            r#"{"provider":"local-cpu","serve_as":"fixture/model","adapter":"fixture","tasks":["text_to_speech"],"implementation":"pending","precision":"float32","qualification":"unverified"}"#,
        ))
        .unwrap();
        let route = &catalog.models()[0].routes()[0];
        assert_eq!(route.base_url(), ADAPTER_BASE_URL);
        assert_eq!(route.implementation(), Implementation::Pending);
        assert_eq!(route.implementation().label(), "needs adapter");
    }

    #[test]
    fn builtin_catalog_has_no_protocol_violations_and_lists_sub_billion_ollama_routes() {
        let catalog = Catalog::builtin().unwrap();
        assert!(catalog.models().len() > 40);
        let ollama =
            catalog.routes_matching(Some(ProviderKind::Ollama), Some(Task::TextGeneration));
        assert_eq!(ollama.len(), 4);
        for route in ollama {
            let model = catalog.model(route.model()).unwrap();
            assert!(model
                .parameters()
                .is_some_and(|value| value < 1_000_000_000));
        }
        assert!(catalog
            .routes_matching(Some(ProviderKind::HfServe), Some(Task::Embedding))
            .iter()
            .any(|route| route.model().as_str() == "google/embeddinggemma-300m"));
        assert!(catalog
            .routes_matching(Some(ProviderKind::LocalCpu), Some(Task::TextToSpeech))
            .iter()
            .any(|route| route.model().as_str() == "gradium/phonon-31m-en"));
    }
}
