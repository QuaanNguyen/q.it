use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::time::Duration;

use crate::catalog::{self, Catalog, Implementation, RouteKey};
use crate::model::{BenchmarkDefinition, ProviderKind, RunRequest, RunSettings, Task};
use crate::provider::ProviderClient;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Axis {
    Benchmark,
    Provider,
    Model,
}

impl Axis {
    pub fn label(self) -> &'static str {
        match self {
            Self::Benchmark => "Benchmark",
            Self::Provider => "Provider",
            Self::Model => "Model",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Choice {
    Benchmark(String),
    Provider(ProviderKind),
    Model(RouteKey),
}

impl Choice {
    pub fn axis(&self) -> Axis {
        match self {
            Self::Benchmark(_) => Axis::Benchmark,
            Self::Provider(_) => Axis::Provider,
            Self::Model(_) => Axis::Model,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BenchmarkFact {
    pub id: String,
    pub name: String,
    pub task: Task,
    pub pack_version: String,
    pub case_count: usize,
}

impl BenchmarkFact {
    pub fn from_definitions(definitions: &[BenchmarkDefinition]) -> Vec<Self> {
        definitions
            .iter()
            .map(|definition| Self {
                id: definition.id.clone(),
                name: definition.name.clone(),
                task: definition.task,
                pack_version: definition.pack_version.clone(),
                case_count: definition.case_count,
            })
            .collect()
    }
}

#[derive(Clone, Debug, Default)]
pub struct Facets {
    benchmark: Option<String>,
    provider: Option<ProviderKind>,
    route: Option<RouteKey>,
}

impl Facets {
    pub fn benchmark(&self) -> Option<&str> {
        self.benchmark.as_deref()
    }

    pub fn provider(&self) -> Option<ProviderKind> {
        self.provider
    }

    pub fn route(&self) -> Option<&RouteKey> {
        self.route.as_ref()
    }
}

pub struct Query<'a> {
    catalog: &'a Catalog,
    benchmarks: &'a [BenchmarkFact],
}

impl<'a> Query<'a> {
    pub fn new(catalog: &'a Catalog, benchmarks: &'a [BenchmarkFact]) -> Self {
        Self {
            catalog,
            benchmarks,
        }
    }

    pub fn catalog(&self) -> &Catalog {
        self.catalog
    }

    pub fn benchmark(&self, id: &str) -> Option<&BenchmarkFact> {
        self.benchmarks.iter().find(|fact| fact.id == id)
    }

    pub fn task(&self, facets: &Facets) -> Option<Task> {
        facets
            .benchmark
            .as_deref()
            .and_then(|id| self.benchmark(id))
            .map(|fact| fact.task)
    }

    pub fn benchmarks(&self, facets: &Facets) -> Vec<&BenchmarkFact> {
        let route = facets
            .route
            .as_ref()
            .and_then(|key| self.catalog.route(key));
        self.benchmarks
            .iter()
            .filter(|fact| match route {
                Some(route) => route.serves(fact.task),
                None => !self
                    .catalog
                    .routes_matching(facets.provider, Some(fact.task))
                    .is_empty(),
            })
            .collect()
    }

    pub fn providers(&self, facets: &Facets) -> Vec<ProviderKind> {
        self.catalog.providers_for(self.task(facets))
    }

    pub fn routes(&self, facets: &Facets) -> Vec<&catalog::Route> {
        self.catalog
            .routes_matching(facets.provider, self.task(facets))
    }

    pub fn admits(&self, facets: &Facets, choice: &Choice) -> bool {
        match choice {
            Choice::Benchmark(id) => self.benchmarks(facets).iter().any(|fact| &fact.id == id),
            Choice::Provider(provider) => self.providers(facets).contains(provider),
            Choice::Model(key) => self.routes(facets).iter().any(|route| route.key() == key),
        }
    }

    pub fn resolve(&self, facets: &Facets) -> Result<Resolved, Refusal> {
        let benchmark_id = facets
            .benchmark
            .as_deref()
            .ok_or(Refusal::Missing(Axis::Benchmark))?;
        let fact = self
            .benchmark(benchmark_id)
            .ok_or(Refusal::Missing(Axis::Benchmark))?;
        let provider = facets.provider.ok_or(Refusal::Missing(Axis::Provider))?;
        if !self.providers(facets).contains(&provider) {
            return Err(Refusal::ProviderCannotServeBenchmark {
                provider,
                task: fact.task,
            });
        }
        let key = facets.route.as_ref().ok_or(Refusal::Missing(Axis::Model))?;
        let route = self
            .catalog
            .route(key)
            .ok_or(Refusal::Missing(Axis::Model))?;
        if !route.serves(fact.task) {
            return Err(Refusal::ModelCannotServeBenchmark {
                model: route.model().to_string(),
                task: fact.task,
            });
        }
        if let Implementation::Pending = route.implementation() {
            return Err(Refusal::AdapterPending {
                model: route.model().to_string(),
                adapter: route.adapter().unwrap_or("unknown").to_string(),
            });
        }
        Ok(Resolved {
            benchmark: fact.id.clone(),
            task: fact.task,
            route: key.clone(),
            serve_as: route.serve_as().to_string(),
            base_url: route.base_url().to_string(),
            model: route.model().to_string(),
        })
    }
}

#[derive(Clone, Debug)]
pub struct Resolved {
    benchmark: String,
    task: Task,
    route: RouteKey,
    serve_as: String,
    base_url: String,
    model: String,
}

impl Resolved {
    pub fn benchmark(&self) -> &str {
        &self.benchmark
    }

    pub fn task(&self) -> Task {
        self.task
    }

    pub fn route(&self) -> &RouteKey {
        &self.route
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn serve_as(&self) -> &str {
        &self.serve_as
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn request(&self, settings: &RunSettings) -> RunRequest {
        RunRequest {
            benchmark_id: self.benchmark.clone(),
            provider: self.route.provider(),
            base_url: settings
                .base_url
                .clone()
                .unwrap_or_else(|| self.base_url.clone()),
            api_key: settings.api_key.clone(),
            model: self.serve_as.clone(),
            iterations: settings.iterations,
            warmups: settings.warmups,
            max_output_tokens: settings.max_output_tokens,
            timeout_seconds: settings.timeout_seconds,
            target_pid: settings.target_pid,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Refusal {
    Missing(Axis),
    ModelCannotServeBenchmark { model: String, task: Task },
    ProviderCannotServeBenchmark { provider: ProviderKind, task: Task },
    AdapterPending { model: String, adapter: String },
    MissingModel { model: String },
    Unreachable { endpoint: String, detail: String },
}

impl Display for Refusal {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing(axis) => write!(formatter, "choose a {}", axis.label().to_lowercase()),
            Self::ModelCannotServeBenchmark { model, task } => write!(
                formatter,
                "{model} cannot run {}",
                task.label().to_lowercase()
            ),
            Self::ProviderCannotServeBenchmark { provider, task } => write!(
                formatter,
                "{} cannot run {}",
                provider.label(),
                task.label().to_lowercase()
            ),
            Self::AdapterPending { model, adapter } => {
                write!(formatter, "{model} needs the {adapter} adapter")
            }
            Self::MissingModel { model } => write!(formatter, "{model} is not installed"),
            Self::Unreachable { endpoint, detail } => {
                write!(formatter, "{endpoint} is not reachable: {detail}")
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Cause {
    ModelCannotServeBenchmark {
        model: String,
        task: Task,
    },
    ProviderCannotServeBenchmark {
        provider: ProviderKind,
        task: Task,
    },
    NoRouteForProvider {
        provider: ProviderKind,
        model: String,
    },
}

impl Display for Cause {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ModelCannotServeBenchmark { model, task } => write!(
                formatter,
                "{model} cannot run {}",
                task.label().to_lowercase()
            ),
            Self::ProviderCannotServeBenchmark { provider, task } => write!(
                formatter,
                "{} cannot run {}",
                provider.label(),
                task.label().to_lowercase()
            ),
            Self::NoRouteForProvider { provider, model } => {
                write!(formatter, "{model} has no {} route", provider.label())
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Release {
    pub axis: Axis,
    pub value: String,
    pub cause: Cause,
}

#[derive(Clone, Debug, Default)]
pub struct Outcome {
    pub released: Option<Release>,
    pub refusal: Option<Refusal>,
}

impl Outcome {
    pub fn line(&self) -> Option<String> {
        if let Some(release) = &self.released {
            return Some(format!(
                "{} {} cleared: {}",
                release.axis.label(),
                release.value,
                release.cause
            ));
        }
        self.refusal.as_ref().map(|refusal| refusal.to_string())
    }
}

#[derive(Clone, Debug, Default)]
pub struct Selection {
    facets: Facets,
}

impl Selection {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn facets(&self) -> &Facets {
        &self.facets
    }

    pub fn pin(&mut self, query: &Query<'_>, choice: Choice) -> Outcome {
        let mut outcome = Outcome::default();
        match choice {
            Choice::Benchmark(id) => {
                if query.benchmark(&id).is_none() {
                    outcome.refusal = Some(Refusal::Missing(Axis::Benchmark));
                    return outcome;
                }
                self.facets.benchmark = Some(id.clone());
                self.repair(query, Axis::Benchmark, &mut outcome);
            }
            Choice::Provider(provider) => {
                if !query.admits(&self.facets, &Choice::Provider(provider)) {
                    if let Some(task) = query.task(&self.facets) {
                        outcome.refusal =
                            Some(Refusal::ProviderCannotServeBenchmark { provider, task });
                    } else {
                        outcome.refusal = Some(Refusal::Missing(Axis::Provider));
                    }
                    return outcome;
                }
                self.facets.provider = Some(provider);
                if self
                    .facets
                    .route
                    .as_ref()
                    .is_some_and(|route| route.provider() != provider)
                {
                    let value = self
                        .facets
                        .route
                        .take()
                        .map(|route| route.serve_as().to_string())
                        .unwrap_or_default();
                    outcome.released = Some(Release {
                        axis: Axis::Model,
                        value: value.clone(),
                        cause: Cause::NoRouteForProvider {
                            provider,
                            model: value,
                        },
                    });
                }
                self.repair(query, Axis::Provider, &mut outcome);
            }
            Choice::Model(key) => {
                if query.catalog().route(&key).is_none() {
                    outcome.refusal = Some(Refusal::ModelCannotServeBenchmark {
                        model: key.serve_as().to_string(),
                        task: query.task(&self.facets).unwrap_or(Task::TextGeneration),
                    });
                    return outcome;
                }
                self.facets.provider = Some(key.provider());
                self.facets.route = Some(key);
                self.repair(query, Axis::Model, &mut outcome);
            }
        }
        outcome
    }

    fn repair(&mut self, query: &Query<'_>, keep: Axis, outcome: &mut Outcome) {
        let task = query.task(&self.facets);
        if keep == Axis::Benchmark {
            if let Some(provider) = self.facets.provider {
                if !query.providers(&self.facets).contains(&provider) {
                    if let Some(task) = task {
                        outcome.released = Some(Release {
                            axis: Axis::Provider,
                            value: provider.label().to_string(),
                            cause: Cause::ProviderCannotServeBenchmark { provider, task },
                        });
                    }
                    self.facets.provider = None;
                    self.facets.route = None;
                    return;
                }
            }
        }
        if keep != Axis::Model {
            if let Some(key) = self.facets.route.clone() {
                let route = query.catalog.route(&key);
                let legal = route.is_some_and(|route| {
                    self.facets
                        .provider
                        .is_none_or(|provider| route.provider() == provider)
                        && task.is_none_or(|task| route.serves(task))
                });
                if !legal {
                    let value = key.serve_as().to_string();
                    let cause = match (route, task) {
                        (Some(route), Some(task)) => {
                            if route.serves(task) {
                                Cause::NoRouteForProvider {
                                    provider: self.facets.provider.unwrap_or(route.provider()),
                                    model: value.clone(),
                                }
                            } else {
                                Cause::ModelCannotServeBenchmark {
                                    model: route.model().to_string(),
                                    task,
                                }
                            }
                        }
                        _ => Cause::NoRouteForProvider {
                            provider: self.facets.provider.unwrap_or(key.provider()),
                            model: value.clone(),
                        },
                    };
                    self.facets.route = None;
                    if outcome.released.is_none() {
                        outcome.released = Some(Release {
                            axis: Axis::Model,
                            value,
                            cause,
                        });
                    }
                }
            }
        }
        if keep == Axis::Model {
            if let Some(benchmark) = self.facets.benchmark.clone() {
                if !query
                    .benchmarks(&self.facets)
                    .iter()
                    .any(|fact| fact.id == benchmark)
                {
                    let task = query.task(&self.facets);
                    let route = self.facets.route.clone();
                    self.facets.benchmark = None;
                    if outcome.released.is_none() {
                        if let Some(route) = route {
                            outcome.released = Some(Release {
                                axis: Axis::Benchmark,
                                value: benchmark,
                                cause: Cause::ModelCannotServeBenchmark {
                                    model: route.serve_as().to_string(),
                                    task: task.unwrap_or(Task::TextGeneration),
                                },
                            });
                        }
                    }
                }
            }
        }
    }

    pub fn release(&mut self, axis: Axis) -> Outcome {
        match axis {
            Axis::Benchmark => self.facets.benchmark = None,
            Axis::Provider => {
                self.facets.provider = None;
                self.facets.route = None;
            }
            Axis::Model => self.facets.route = None,
        }
        Outcome::default()
    }

    pub fn status(
        &self,
        query: &Query<'_>,
        evidence: &Evidence,
        endpoint: Option<&str>,
    ) -> Result<Resolved, Refusal> {
        let resolved = query.resolve(&self.facets)?;
        let endpoint = endpoint.unwrap_or(resolved.base_url());
        let availability = evidence
            .get(resolved.route(), endpoint)
            .map(|entry| entry.availability)
            .unwrap_or(Availability::Unprobed);
        match availability {
            Availability::Unprobed | Availability::Present => Ok(resolved),
            Availability::AdapterPending => Err(Refusal::AdapterPending {
                model: resolved.model().to_string(),
                adapter: resolved.route().adapter().unwrap_or("optional").to_string(),
            }),
            Availability::MissingModel => Err(Refusal::MissingModel {
                model: resolved.model().to_string(),
            }),
            Availability::Unreachable => Err(Refusal::Unreachable {
                endpoint: endpoint.to_string(),
                detail: evidence
                    .get(resolved.route(), endpoint)
                    .map(|entry| entry.detail.clone())
                    .unwrap_or_default(),
            }),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Availability {
    Unprobed,
    Present,
    AdapterPending,
    MissingModel,
    Unreachable,
}

impl Availability {
    pub fn label(self) -> &'static str {
        match self {
            Self::Unprobed => "allowed",
            Self::Present => "ok",
            Self::AdapterPending => "needs adapter",
            Self::MissingModel => "needs model",
            Self::Unreachable => "not running",
        }
    }

    pub fn permits_start(self) -> bool {
        matches!(self, Self::Unprobed | Self::Present)
    }
}

#[derive(Clone, Debug)]
pub struct Stamped {
    pub base_url: String,
    pub availability: Availability,
    pub detail: String,
}

#[derive(Clone, Debug, Default)]
pub struct Evidence {
    entries: BTreeMap<RouteKey, Stamped>,
}

impl Evidence {
    pub fn record(
        &mut self,
        route: RouteKey,
        base_url: String,
        availability: Availability,
        detail: String,
    ) {
        self.entries.insert(
            route,
            Stamped {
                base_url,
                availability,
                detail,
            },
        );
    }

    pub fn get(&self, route: &RouteKey, base_url: &str) -> Option<&Stamped> {
        self.entries
            .get(route)
            .filter(|entry| entry.base_url == base_url)
    }

    pub fn forget_provider(&mut self, provider: ProviderKind) {
        self.entries.retain(|route, _| route.provider() != provider);
    }
}

#[derive(Clone, Debug)]
pub struct ProbeRequest {
    pub route: RouteKey,
    pub provider: ProviderKind,
    pub base_url: String,
    pub serve_as: String,
}

impl ProbeRequest {
    pub fn run(&self, timeout: Duration) -> ProbeReport {
        let client = match ProviderClient::new(
            self.provider,
            self.base_url.clone(),
            None,
            self.serve_as.clone(),
            timeout.as_secs().max(1),
        ) {
            Ok(client) => client,
            Err(error) => {
                return ProbeReport::Failed {
                    route: self.route.clone(),
                    base_url: self.base_url.clone(),
                    detail: error,
                }
            }
        };
        let runtime = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(runtime) => runtime,
            Err(error) => {
                return ProbeReport::Failed {
                    route: self.route.clone(),
                    base_url: self.base_url.clone(),
                    detail: error.to_string(),
                }
            }
        };
        match runtime.block_on(client.probe()) {
            Ok(true) => ProbeReport::Settled {
                route: self.route.clone(),
                base_url: self.base_url.clone(),
                availability: Availability::Present,
                detail: String::new(),
            },
            Ok(false) => ProbeReport::Settled {
                route: self.route.clone(),
                base_url: self.base_url.clone(),
                availability: Availability::MissingModel,
                detail: String::new(),
            },
            Err(detail) => ProbeReport::Failed {
                route: self.route.clone(),
                base_url: self.base_url.clone(),
                detail,
            },
        }
    }
}

#[derive(Clone, Debug)]
pub enum ProbeReport {
    Settled {
        route: RouteKey,
        base_url: String,
        availability: Availability,
        detail: String,
    },
    Failed {
        route: RouteKey,
        base_url: String,
        detail: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::BenchmarkDefinition;

    fn catalog() -> Catalog {
        Catalog::from_json(
            r#"{"schema_version":1,"sources":[],"models":[
              {"id":"fixture/text","revision":"0123456789012345678901234567890123456789","parameters":1,"routes":[
                {"provider":"local-cpu","serve_as":"fixture/text","tasks":["text_generation"],"implementation":"direct"},
                {"provider":"ollama","serve_as":"fixture:text","tasks":["text_generation"],"implementation":"direct"}]},
              {"id":"fixture/speech","revision":"0123456789012345678901234567890123456789","parameters":2,"routes":[
                {"provider":"local-cpu","serve_as":"fixture/speech","adapter":"phonon","tasks":["text_to_speech"],"implementation":"pending"}]},
              {"id":"fixture/vision","revision":"0123456789012345678901234567890123456789","parameters":3,"routes":[
                {"provider":"transformers","serve_as":"fixture/vision","tasks":["image_to_text"],"implementation":"direct"}]}
            ]}"#,
        )
        .unwrap()
    }

    fn benchmark(id: &str, task: Task) -> BenchmarkDefinition {
        BenchmarkDefinition {
            id: id.into(),
            pack_id: "fixture".into(),
            pack_name: "Fixture".into(),
            pack_version: "1".into(),
            name: id.into(),
            description: "fixture".into(),
            task,
            decision_input: None,
            case_count: 1,
            tags: Vec::new(),
            default_max_output_tokens: 1,
            built_in: true,
            cases: Vec::new(),
        }
    }

    fn facts() -> Vec<BenchmarkFact> {
        BenchmarkFact::from_definitions(&[
            benchmark("fixture/text", Task::TextGeneration),
            benchmark("fixture/speech", Task::TextToSpeech),
            benchmark("fixture/vision", Task::ImageToText),
        ])
    }

    #[test]
    fn benchmark_first_narrows_providers_and_models() {
        let catalog = catalog();
        let facts = facts();
        let query = Query::new(&catalog, &facts);
        let mut selection = Selection::new();
        selection.pin(&query, Choice::Benchmark("fixture/text".into()));
        assert_eq!(
            query.providers(selection.facets()),
            vec![ProviderKind::Ollama, ProviderKind::LocalCpu]
        );
        let routes = query.routes(selection.facets());
        assert_eq!(routes.len(), 2);
        assert!(selection
            .pin(
                &query,
                Choice::Model(catalog::RouteKey::new(
                    ProviderKind::Ollama,
                    None,
                    "fixture:text"
                ))
            )
            .released
            .is_none());
        assert_eq!(selection.facets().provider(), Some(ProviderKind::Ollama));
        assert!(selection.status(&query, &Evidence::default(), None).is_ok());
    }

    #[test]
    fn model_first_clears_an_incompatible_benchmark_and_reports_the_cause() {
        let catalog = catalog();
        let facts = facts();
        let query = Query::new(&catalog, &facts);
        let mut selection = Selection::new();
        selection.pin(&query, Choice::Benchmark("fixture/text".into()));
        let outcome = selection.pin(
            &query,
            Choice::Model(catalog::RouteKey::new(
                ProviderKind::LocalCpu,
                Some("phonon"),
                "fixture/speech",
            )),
        );
        assert!(selection.facets().benchmark().is_none());
        assert_eq!(outcome.released.unwrap().axis, Axis::Benchmark);
    }

    #[test]
    fn model_first_keeps_a_model_whose_provider_cannot_run_the_old_benchmark() {
        let catalog = catalog();
        let facts = facts();
        let query = Query::new(&catalog, &facts);
        let mut selection = Selection::new();
        selection.pin(&query, Choice::Benchmark("fixture/text".into()));
        let key = catalog::RouteKey::new(ProviderKind::Transformers, None, "fixture/vision");
        let outcome = selection.pin(&query, Choice::Model(key.clone()));
        assert_eq!(selection.facets().route(), Some(&key));
        assert_eq!(
            selection.facets().provider(),
            Some(ProviderKind::Transformers)
        );
        assert!(selection.facets().benchmark().is_none());
        assert_eq!(outcome.released.unwrap().axis, Axis::Benchmark);
    }

    #[test]
    fn changing_the_benchmark_clears_a_model_that_cannot_serve_it() {
        let catalog = catalog();
        let facts = facts();
        let query = Query::new(&catalog, &facts);
        let mut selection = Selection::new();
        selection.pin(
            &query,
            Choice::Model(catalog::RouteKey::new(
                ProviderKind::LocalCpu,
                None,
                "fixture/text",
            )),
        );
        let outcome = selection.pin(&query, Choice::Benchmark("fixture/speech".into()));
        assert!(selection.facets().route().is_none());
        let release = outcome.released.unwrap();
        assert_eq!(release.axis, Axis::Model);
        assert!(release
            .cause
            .to_string()
            .contains("cannot run text to speech"));
    }

    #[test]
    fn pending_adapters_block_the_start_gate_until_the_adapter_ships() {
        let catalog = catalog();
        let facts = facts();
        let query = Query::new(&catalog, &facts);
        let mut selection = Selection::new();
        selection.pin(&query, Choice::Benchmark("fixture/speech".into()));
        selection.pin(
            &query,
            Choice::Model(catalog::RouteKey::new(
                ProviderKind::LocalCpu,
                Some("phonon"),
                "fixture/speech",
            )),
        );
        let refusal = selection
            .status(&query, &Evidence::default(), None)
            .unwrap_err();
        assert!(matches!(refusal, Refusal::AdapterPending { .. }));
    }

    #[test]
    fn observed_availability_blocks_start_and_is_scoped_to_the_endpoint() {
        let catalog = catalog();
        let facts = facts();
        let query = Query::new(&catalog, &facts);
        let mut selection = Selection::new();
        selection.pin(&query, Choice::Benchmark("fixture/text".into()));
        let key = catalog::RouteKey::new(ProviderKind::LocalCpu, None, "fixture/text");
        selection.pin(&query, Choice::Model(key.clone()));
        let mut evidence = Evidence::default();
        evidence.record(
            key.clone(),
            "http://127.0.0.1:8081".into(),
            Availability::Unreachable,
            "connection refused".into(),
        );
        assert!(matches!(
            selection.status(&query, &evidence, None).unwrap_err(),
            Refusal::Unreachable { .. }
        ));
        evidence.forget_provider(ProviderKind::LocalCpu);
        assert!(selection.status(&query, &evidence, None).is_ok());
    }
}
