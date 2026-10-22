use std::io::{self, IsTerminal};
use std::sync::mpsc::{self, Receiver};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Line;
use ratatui::widgets::{
    Block, Borders, Cell, Clear, Gauge, Paragraph, Row, Table, TableState, Tabs, Wrap,
};
use ratatui::Frame;

use crate::analysis::{analyze, rank, ComparisonGroup, Metric};
use crate::catalog::{Catalog, Implementation, RouteKey};
use crate::command::workspace;
use crate::model::{BenchmarkDefinition, RunRecord, RunSettings, RunStatus, SampleRecord};
use crate::packs::PackCatalog;
use crate::paths::Paths;
use crate::runner::{execute_with_control, prepare_benchmark, ExecutionControl, RunProgress};
use crate::select::{
    Availability, Axis, BenchmarkFact, Choice, Evidence, ProbeReport, ProbeRequest, Query,
    Selection,
};
use crate::store::Store;

const BLUE: Color = Color::Rgb(74, 174, 237);
const SCREENS: [&str; 5] = [
    "1 Benchmarks",
    "2 Start",
    "3 Progress",
    "4 Results",
    "5 Compare",
];
const FIELDS: [&str; 10] = [
    "Benchmark",
    "Provider",
    "Model",
    "Base URL",
    "Passes",
    "Warmups",
    "Max tokens",
    "Timeout s",
    "Provider PID",
    "Start run",
];

pub fn run() -> Result<(), String> {
    if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        return Err("qit tui requires an interactive terminal on stdin and stdout; use qit benchmark list, run, results, or export in scripts".into());
    }
    let (paths, catalog, store) = workspace()?;
    let mut app = App::new(paths, catalog, store)?;
    let restoration = TerminalRestoration;
    let mut terminal = ratatui::try_init().map_err(|error| format!("open terminal: {error}"))?;
    let result = app.run(&mut terminal);
    drop(restoration);
    result
}

struct TerminalRestoration;

impl Drop for TerminalRestoration {
    fn drop(&mut self) {
        ratatui::restore();
    }
}

struct Worker {
    control: ExecutionControl,
    receiver: Receiver<WorkerEvent>,
    join: Option<JoinHandle<()>>,
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.control.cancel();
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

enum WorkerEvent {
    Progress(Box<RunProgress>),
    Finished(Result<Box<RunRecord>, String>),
}

struct ProbeWorker {
    receiver: Receiver<ProbeReport>,
    join: Option<JoinHandle<()>>,
}

impl Drop for ProbeWorker {
    fn drop(&mut self) {
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

struct Picker {
    axis: Axis,
    query: String,
    selected: usize,
    all_tasks: bool,
}

struct PickerEntry {
    choice: Choice,
    label: String,
    detail: String,
}

struct App {
    paths: Paths,
    catalog: PackCatalog,
    model_catalog: &'static Catalog,
    store: Store,
    benchmarks: Vec<BenchmarkDefinition>,
    facts: Vec<BenchmarkFact>,
    runs: Vec<RunRecord>,
    groups: Vec<ComparisonGroup>,
    screen: usize,
    selection: usize,
    choice: Selection,
    settings: Vec<String>,
    field: usize,
    editing: bool,
    filter_editing: bool,
    filter: String,
    succeeded_only: bool,
    detail: Option<RunRecord>,
    samples: Vec<SampleRecord>,
    detail_scroll: u16,
    group: usize,
    metric: usize,
    progress: RunProgress,
    worker: Option<Worker>,
    probe: Option<ProbeWorker>,
    picker: Option<Picker>,
    evidence: Evidence,
    notice: String,
}

impl App {
    fn new(paths: Paths, catalog: PackCatalog, store: Store) -> Result<Self, String> {
        let benchmarks = catalog.benchmarks()?;
        let model_catalog = Catalog::builtin()?;
        let facts = BenchmarkFact::from_definitions(&benchmarks);
        let mut choice = Selection::new();
        {
            let query = Query::new(model_catalog, &facts);
            let available = query.benchmarks(choice.facets());
            let first = available
                .iter()
                .find(|fact| fact.task == crate::model::Task::TextGeneration)
                .or_else(|| available.first())
                .map(|fact| fact.id.clone());
            if let Some(id) = first {
                choice.pin(&query, Choice::Benchmark(id));
            }
            if let Some(provider) = query.providers(choice.facets()).first().copied() {
                choice.pin(&query, Choice::Provider(provider));
            }
            if let Some(route) = query.routes(choice.facets()).first() {
                let key = route.key().clone();
                choice.pin(&query, Choice::Model(key));
            }
        }
        let base_url = choice
            .facets()
            .route()
            .and_then(|key| model_catalog.route(key))
            .map(|route| route.base_url().to_string())
            .unwrap_or_default();
        let mut app = Self {
            paths,
            catalog,
            model_catalog,
            store,
            benchmarks,
            facts,
            runs: Vec::new(),
            groups: Vec::new(),
            screen: 0,
            selection: 0,
            choice,
            settings: vec![
                base_url,
                "3".into(),
                "1".into(),
                String::new(),
                "120".into(),
                String::new(),
            ],
            field: 0,
            editing: false,
            filter_editing: false,
            filter: String::new(),
            succeeded_only: false,
            detail: None,
            samples: Vec::new(),
            detail_scroll: 0,
            group: 0,
            metric: 0,
            progress: RunProgress::default(),
            worker: None,
            probe: None,
            picker: None,
            evidence: Evidence::default(),
            notice: String::new(),
        };
        app.refresh()?;
        Ok(app)
    }

    fn refresh(&mut self) -> Result<(), String> {
        self.runs = self.store.runs(10_000, false)?;
        self.groups = analyze(self.runs.clone()).groups;
        self.group = self.group.min(self.groups.len().saturating_sub(1));
        Ok(())
    }

    fn pin(&mut self, choice: Choice) {
        let previous = (
            self.choice.facets().route().cloned(),
            self.choice.facets().provider(),
        );
        let outcome = {
            let query = Query::new(self.model_catalog, &self.facts);
            self.choice.pin(&query, choice)
        };
        if let Some(line) = outcome.line() {
            self.notice = line;
        } else {
            self.notice.clear();
        }
        if let Some(release) = &outcome.released {
            self.field = match release.axis {
                Axis::Benchmark => 0,
                Axis::Provider => 1,
                Axis::Model => 2,
            };
        }
        if previous
            != (
                self.choice.facets().route().cloned(),
                self.choice.facets().provider(),
            )
        {
            self.sync_endpoint();
        }
    }

    fn sync_endpoint(&mut self) {
        let endpoint = self
            .choice
            .facets()
            .route()
            .and_then(|key| self.model_catalog.route(key))
            .map(|route| route.base_url().to_string())
            .or_else(|| {
                self.choice
                    .facets()
                    .provider()
                    .map(|provider| provider.default_base_url().to_string())
            });
        if let Some(endpoint) = endpoint {
            self.settings[0] = endpoint;
        }
    }

    fn visible_benchmarks(&self) -> Vec<&BenchmarkDefinition> {
        let query = Query::new(self.model_catalog, &self.facts);
        let allowed = query
            .benchmarks(self.choice.facets())
            .into_iter()
            .map(|fact| fact.id.clone())
            .collect::<Vec<_>>();
        self.benchmarks
            .iter()
            .filter(|benchmark| allowed.contains(&benchmark.id))
            .collect()
    }

    fn availability(&self, key: &RouteKey, base_url: &str) -> Availability {
        self.evidence
            .get(key, base_url)
            .map(|entry| entry.availability)
            .unwrap_or(Availability::Unprobed)
    }

    fn picker_entries(&self, axis: Axis, all_tasks: bool) -> Vec<PickerEntry> {
        let query = Query::new(self.model_catalog, &self.facts);
        match axis {
            Axis::Benchmark => self
                .facts
                .iter()
                .map(|fact| PickerEntry {
                    choice: Choice::Benchmark(fact.id.clone()),
                    label: fact.id.clone(),
                    detail: format!(
                        "{} / v{} / {} cases",
                        fact.task.label(),
                        fact.pack_version,
                        fact.case_count
                    ),
                })
                .collect(),
            Axis::Provider => query
                .providers(self.choice.facets())
                .into_iter()
                .map(|provider| {
                    let count = query
                        .routes(self.choice.facets())
                        .into_iter()
                        .filter(|route| route.provider() == provider)
                        .count();
                    PickerEntry {
                        choice: Choice::Provider(provider),
                        label: provider.as_str().to_string(),
                        detail: format!("{} / {count} models", provider.default_base_url()),
                    }
                })
                .collect(),
            Axis::Model => {
                let routes = if all_tasks {
                    self.model_catalog.routes_matching(None, None)
                } else {
                    query.routes(self.choice.facets())
                };
                routes
                    .into_iter()
                    .map(|route| {
                        let model = self.model_catalog.model(route.model());
                        let availability = if let Implementation::Pending = route.implementation() {
                            Availability::AdapterPending
                        } else {
                            self.availability(route.key(), route.base_url())
                        };
                        PickerEntry {
                            choice: Choice::Model(route.key().clone()),
                            label: route.serve_as().to_string(),
                            detail: format!(
                                "{} / {} / {} / {} / {}",
                                route.model(),
                                model
                                    .and_then(|model| model.parameters())
                                    .map(|value| self.model_catalog.parameters_label(Some(value)))
                                    .unwrap_or_else(|| "?".into()),
                                route.provider().as_str(),
                                route.implementation().label(),
                                availability.label()
                            ),
                        }
                    })
                    .collect()
            }
        }
    }

    fn picker_matches(&self, axis: Axis, query: &str, all_tasks: bool) -> Vec<PickerEntry> {
        let needle = query.to_lowercase();
        self.picker_entries(axis, all_tasks)
            .into_iter()
            .filter(|entry| {
                needle.is_empty()
                    || entry.label.to_lowercase().contains(&needle)
                    || entry.detail.to_lowercase().contains(&needle)
            })
            .collect()
    }

    fn open_picker(&mut self, axis: Axis, all_tasks: bool) {
        let entries = self.picker_entries(axis, all_tasks);
        let current = match axis {
            Axis::Benchmark => self
                .choice
                .facets()
                .benchmark()
                .map(|id| Choice::Benchmark(id.to_string())),
            Axis::Provider => self.choice.facets().provider().map(Choice::Provider),
            Axis::Model => self
                .choice
                .facets()
                .route()
                .map(|key| Choice::Model(key.clone())),
        };
        let selected = current
            .and_then(|choice| entries.iter().position(|entry| entry.choice == choice))
            .unwrap_or(0);
        self.picker = Some(Picker {
            axis,
            query: String::new(),
            selected,
            all_tasks,
        });
    }

    fn commit_picker(&mut self) {
        let Some(picker) = self.picker.take() else {
            return;
        };
        let entries = self.picker_matches(picker.axis, &picker.query, picker.all_tasks);
        let Some(entry) = entries
            .get(picker.selected.min(entries.len().saturating_sub(1)))
            .map(|entry| Choice::clone(&entry.choice))
        else {
            self.notice = format!("No {} matches '{}'", picker.axis.label(), picker.query);
            return;
        };
        self.pin(entry);
    }

    fn probe(&mut self) {
        let Some(key) = self.choice.facets().route().cloned() else {
            self.notice = "Choose a model before probing".into();
            return;
        };
        let Some(route) = self.model_catalog.route(&key) else {
            self.notice = "The selected route is no longer in the catalog".into();
            return;
        };
        if let Implementation::Pending = route.implementation() {
            self.evidence.record(
                key.clone(),
                route.base_url().to_string(),
                Availability::AdapterPending,
                format!(
                    "install the {} adapter",
                    route.adapter().unwrap_or("optional")
                ),
            );
            self.notice = format!("{} needs its optional adapter", route.model());
            return;
        }
        let base_url = if self.settings[0].trim().is_empty() {
            route.base_url().to_string()
        } else {
            self.settings[0].trim().to_string()
        };
        let request = ProbeRequest {
            route: key,
            provider: route.provider(),
            base_url,
            serve_as: route.serve_as().to_string(),
        };
        let (sender, receiver) = mpsc::channel();
        let join = std::thread::spawn(move || {
            let report = request.run(Duration::from_secs(3));
            let _ = sender.send(report);
        });
        self.probe = Some(ProbeWorker {
            receiver,
            join: Some(join),
        });
        self.notice = "Probing the selected endpoint".into();
    }

    fn record_probe(&mut self, report: ProbeReport) {
        match report {
            ProbeReport::Settled {
                route,
                base_url,
                availability,
                detail,
            } => {
                self.notice = format!("Probe: {}", availability.label());
                self.evidence.record(route, base_url, availability, detail);
            }
            ProbeReport::Failed {
                route,
                base_url,
                detail,
            } => {
                self.notice = format!("Probe failed: {detail}");
                self.evidence
                    .record(route, base_url, Availability::Unreachable, detail);
            }
        }
    }

    fn run_settings(&self) -> Result<RunSettings, String> {
        let number = |value: &str, name: &str, low: u32, high: u32| -> Result<u32, String> {
            let value = value.trim();
            let parsed = value
                .parse::<u32>()
                .map_err(|_| format!("{name} must be a whole number"))?;
            if !(low..=high).contains(&parsed) {
                return Err(format!("{name} must be between {low} and {high}"));
            }
            Ok(parsed)
        };
        let max_output_tokens = if self.settings[3].trim().is_empty() {
            None
        } else {
            Some(number(&self.settings[3], "max tokens", 1, u32::MAX)?)
        };
        let target_pid = if self.settings[5].trim().is_empty() {
            None
        } else {
            Some(number(
                &self.settings[5],
                "provider PID",
                1,
                i32::MAX as u32,
            )?)
        };
        Ok(RunSettings {
            api_key: std::env::var("QIT_API_KEY").ok(),
            base_url: if self.settings[0].trim().is_empty() {
                None
            } else {
                Some(self.settings[0].trim().to_string())
            },
            iterations: number(&self.settings[1], "passes", 1, 10_000)?,
            warmups: number(&self.settings[2], "warmups", 0, 10_000)?,
            max_output_tokens,
            timeout_seconds: u64::from(number(&self.settings[4], "timeout", 1, 86_400)?),
            target_pid,
        })
    }

    fn run(&mut self, terminal: &mut ratatui::DefaultTerminal) -> Result<(), String> {
        let mut refresh = Instant::now();
        let mut dirty = true;
        loop {
            if self.receive()? {
                dirty = true;
            }
            if refresh.elapsed() >= Duration::from_secs(2) {
                if self.screen == 3 || self.screen == 4 {
                    self.refresh()?;
                    dirty = true;
                }
                refresh = Instant::now();
            }
            if dirty {
                terminal
                    .draw(|frame| self.draw(frame))
                    .map_err(|error| format!("draw terminal: {error}"))?;
                dirty = false;
            }
            if event::poll(Duration::from_millis(200))
                .map_err(|error| format!("read terminal: {error}"))?
            {
                match event::read().map_err(|error| format!("read terminal: {error}"))? {
                    Event::Key(key) if key.kind == KeyEventKind::Press => {
                        if self.key(key)? {
                            return Ok(());
                        }
                        dirty = true;
                    }
                    Event::Resize(_, _) => dirty = true,
                    _ => {}
                }
            }
        }
    }

    fn receive(&mut self) -> Result<bool, String> {
        let mut changed = false;
        let reports = self
            .probe
            .as_ref()
            .map(|probe| probe.receiver.try_iter().collect::<Vec<_>>())
            .unwrap_or_default();
        if !reports.is_empty() {
            self.probe = None;
            for report in reports {
                self.record_probe(report);
            }
            changed = true;
        }
        let events = self
            .worker
            .as_ref()
            .map(|worker| worker.receiver.try_iter().collect::<Vec<_>>())
            .unwrap_or_default();
        changed |= !events.is_empty();
        for event in events {
            match event {
                WorkerEvent::Progress(progress) => {
                    if progress.expected == 0 {
                        self.progress.phase = progress.phase;
                        self.progress.summary = progress.summary;
                        self.progress.completed = progress.completed;
                    } else {
                        self.progress = *progress;
                    }
                }
                WorkerEvent::Finished(result) => {
                    self.notice = match result {
                        Ok(run) => format!(
                            "{}: {}",
                            run.status.as_str(),
                            run.error.as_deref().unwrap_or(&run.id)
                        ),
                        Err(error) => {
                            if let Some(run) = self.store.run(&self.progress.run_id)? {
                                if run.status == RunStatus::Running {
                                    self.store.finish_run(
                                        &run.id,
                                        RunStatus::Failed,
                                        crate::store::unix_time_ms(),
                                        Some(&error),
                                        &run.summary,
                                    )?;
                                }
                            }
                            error
                        }
                    };
                    self.worker.take();
                    self.refresh()?;
                }
            }
        }
        Ok(changed)
    }

    fn start(&mut self) -> Result<(), String> {
        if self.worker.is_some() || self.store.has_running_run()? {
            self.notice = "A benchmark is already running".into();
            return Ok(());
        }
        let resolved = {
            let query = Query::new(self.model_catalog, &self.facts);
            match self.choice.status(&query, &self.evidence, self.endpoint()) {
                Ok(resolved) => resolved,
                Err(refusal) => {
                    self.notice = refusal.to_string();
                    return Ok(());
                }
            }
        };
        let settings = match self.run_settings() {
            Ok(settings) => settings,
            Err(error) => {
                self.notice = error;
                return Ok(());
            }
        };
        let prepared =
            match prepare_benchmark(&self.store, &self.catalog, resolved.request(&settings)) {
                Ok(prepared) => prepared,
                Err(error) => {
                    self.notice = error;
                    return Ok(());
                }
            };
        self.progress = RunProgress {
            run_id: prepared.run.id.clone(),
            phase: "starting".into(),
            ..Default::default()
        };
        let (sender, receiver) = mpsc::channel();
        let observer = sender.clone();
        let control = ExecutionControl::new(move |progress| {
            let _ = observer.send(WorkerEvent::Progress(Box::new(progress)));
        });
        let execution = control.clone();
        let database = self.paths.database.clone();
        let join = std::thread::spawn(move || {
            let result = (|| {
                let store = Store::connect(&database)?;
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|error| format!("start runner: {error}"))?;
                runtime
                    .block_on(execute_with_control(&store, prepared, execution))
                    .map(Box::new)
            })();
            let _ = sender.send(WorkerEvent::Finished(result));
        });
        self.worker = Some(Worker {
            control,
            receiver,
            join: Some(join),
        });
        self.screen = 2;
        self.notice.clear();
        Ok(())
    }

    fn filtered_runs(&self) -> Vec<&RunRecord> {
        let needle = self.filter.to_lowercase();
        self.runs
            .iter()
            .filter(|run| {
                (!self.succeeded_only || run.status == RunStatus::Succeeded)
                    && format!(
                        "{} {} {} {} {}",
                        run.id,
                        run.model,
                        run.provider,
                        run.benchmark_id,
                        run.status.as_str()
                    )
                    .to_lowercase()
                    .contains(&needle)
            })
            .collect()
    }

    fn key(&mut self, key: KeyEvent) -> Result<bool, String> {
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            self.worker.take();
            return Ok(true);
        }
        if self.picker.is_some() {
            match key.code {
                KeyCode::Esc => {
                    self.picker = None;
                    return Ok(false);
                }
                KeyCode::Enter => {
                    self.commit_picker();
                    return Ok(false);
                }
                _ => {}
            }
            let count = {
                let picker = self.picker.as_ref().unwrap();
                self.picker_matches(picker.axis, &picker.query, picker.all_tasks)
                    .len()
            };
            if let Some(picker) = self.picker.as_mut() {
                match key.code {
                    KeyCode::Down => {
                        picker.selected = (picker.selected + 1).min(count.saturating_sub(1))
                    }
                    KeyCode::Up => picker.selected = picker.selected.saturating_sub(1),
                    KeyCode::Backspace => {
                        picker.query.pop();
                        picker.selected = 0;
                    }
                    KeyCode::Char(character) if !character.is_control() => {
                        picker.query.push(character);
                        picker.selected = 0;
                    }
                    _ => {}
                }
            }
            return Ok(false);
        }
        if self.filter_editing {
            match key.code {
                KeyCode::Esc | KeyCode::Enter => self.filter_editing = false,
                KeyCode::Backspace => {
                    self.filter.pop();
                }
                KeyCode::Char(character) if !character.is_control() => self.filter.push(character),
                _ => {}
            }
            self.selection = 0;
            return Ok(false);
        }
        if self.editing {
            match key.code {
                KeyCode::Esc | KeyCode::Enter => self.editing = false,
                KeyCode::Tab => {
                    self.editing = false;
                    self.field = (self.field + 1) % FIELDS.len();
                }
                KeyCode::Backspace => {
                    if let Some(value) = self.settings.get_mut(self.field.saturating_sub(3)) {
                        value.pop();
                    }
                }
                KeyCode::Char(character) if !character.is_control() => {
                    if let Some(value) = self.settings.get_mut(self.field.saturating_sub(3)) {
                        value.push(character);
                    }
                }
                _ => {}
            }
            return Ok(false);
        }
        if key.code == KeyCode::Char('q') {
            self.worker.take();
            return Ok(true);
        }
        if let KeyCode::Char(character @ '1'..='5') = key.code {
            self.screen = character as usize - '1' as usize;
            self.selection = 0;
            self.detail = None;
            return Ok(false);
        }
        if key.code == KeyCode::Tab {
            self.screen = (self.screen + 1) % SCREENS.len();
            self.selection = 0;
            self.detail = None;
            return Ok(false);
        }
        if key.code == KeyCode::Char('r') {
            self.refresh()?;
        }
        match self.screen {
            0 => match key.code {
                KeyCode::Down | KeyCode::Char('j') => {
                    self.selection =
                        (self.selection + 1).min(self.visible_benchmarks().len().saturating_sub(1))
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    self.selection = self.selection.saturating_sub(1)
                }
                KeyCode::Enter => {
                    let id = self
                        .visible_benchmarks()
                        .get(self.selection)
                        .map(|benchmark| benchmark.id.clone());
                    if let Some(id) = id {
                        self.pin(Choice::Benchmark(id));
                        self.screen = 1;
                        self.field = 2;
                    }
                }
                KeyCode::Char('b') => self.open_picker(Axis::Benchmark, false),
                KeyCode::Char('p') => self.open_picker(Axis::Provider, false),
                KeyCode::Char('m') => self.open_picker(Axis::Model, true),
                _ => {}
            },
            1 => match key.code {
                KeyCode::Down | KeyCode::Char('j') => self.field = (self.field + 1) % FIELDS.len(),
                KeyCode::Up | KeyCode::Char('k') => {
                    self.field = (self.field + FIELDS.len() - 1) % FIELDS.len()
                }
                KeyCode::Left | KeyCode::Right if self.field <= 2 => {
                    self.cycle_field(key.code == KeyCode::Right)
                }
                KeyCode::Enter if self.field == 9 => self.start()?,
                KeyCode::Enter if self.field <= 2 => {
                    let axis = match self.field {
                        0 => Axis::Benchmark,
                        1 => Axis::Provider,
                        _ => Axis::Model,
                    };
                    self.open_picker(axis, false);
                }
                KeyCode::Enter => self.editing = true,
                KeyCode::Char('p') => self.probe(),
                _ => {}
            },
            2 => {
                if key.code == KeyCode::Char('x') {
                    if let Some(worker) = &self.worker {
                        worker.control.cancel();
                        self.notice = "Interrupting benchmark".into();
                    }
                }
            }
            3 => match key.code {
                KeyCode::Esc => self.detail = None,
                KeyCode::Char('/') if self.detail.is_none() => self.filter_editing = true,
                KeyCode::Char('s') if self.detail.is_none() => {
                    self.succeeded_only = !self.succeeded_only;
                    self.selection = 0;
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    if self.detail.is_some() {
                        self.detail_scroll = self.detail_scroll.saturating_add(1)
                    } else {
                        self.selection =
                            (self.selection + 1).min(self.filtered_runs().len().saturating_sub(1))
                    }
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    if self.detail.is_some() {
                        self.detail_scroll = self.detail_scroll.saturating_sub(1)
                    } else {
                        self.selection = self.selection.saturating_sub(1)
                    }
                }
                KeyCode::PageDown if self.detail.is_some() => {
                    self.detail_scroll = self.detail_scroll.saturating_add(10)
                }
                KeyCode::PageUp if self.detail.is_some() => {
                    self.detail_scroll = self.detail_scroll.saturating_sub(10)
                }
                KeyCode::Enter if self.detail.is_none() => {
                    let run = self.filtered_runs().get(self.selection).copied().cloned();
                    if let Some(run) = run {
                        self.samples = self.store.samples(&run.id)?;
                        self.detail = Some(run);
                        self.detail_scroll = 0;
                    }
                }
                _ => {}
            },
            4 => match key.code {
                KeyCode::Down | KeyCode::Char('j') => {
                    self.group = (self.group + 1).min(self.groups.len().saturating_sub(1))
                }
                KeyCode::Up | KeyCode::Char('k') => self.group = self.group.saturating_sub(1),
                KeyCode::Right | KeyCode::Char('m') => {
                    self.metric = (self.metric + 1) % Metric::ALL.len()
                }
                KeyCode::Left => {
                    self.metric = (self.metric + Metric::ALL.len() - 1) % Metric::ALL.len()
                }
                _ => {}
            },
            _ => {}
        }
        Ok(false)
    }

    fn cycle_field(&mut self, forward: bool) {
        let axis = match self.field {
            0 => Axis::Benchmark,
            1 => Axis::Provider,
            2 => Axis::Model,
            _ => return,
        };
        let mut choices = match axis {
            Axis::Benchmark => self
                .visible_benchmarks()
                .iter()
                .map(|benchmark| Choice::Benchmark(benchmark.id.clone()))
                .collect::<Vec<_>>(),
            Axis::Provider => {
                let query = Query::new(self.model_catalog, &self.facts);
                query
                    .providers(self.choice.facets())
                    .into_iter()
                    .map(Choice::Provider)
                    .collect()
            }
            Axis::Model => {
                let query = Query::new(self.model_catalog, &self.facts);
                query
                    .routes(self.choice.facets())
                    .into_iter()
                    .map(|route| Choice::Model(route.key().clone()))
                    .collect()
            }
        };
        if choices.is_empty() {
            return;
        }
        let current = choices
            .iter()
            .position(|choice| match (choice, axis) {
                (Choice::Benchmark(id), Axis::Benchmark) => {
                    self.choice.facets().benchmark() == Some(id.as_str())
                }
                (Choice::Provider(provider), Axis::Provider) => {
                    self.choice.facets().provider() == Some(*provider)
                }
                (Choice::Model(key), Axis::Model) => self.choice.facets().route() == Some(key),
                _ => false,
            })
            .unwrap_or(0);
        let next = if forward {
            (current + 1) % choices.len()
        } else {
            (current + choices.len() - 1) % choices.len()
        };
        self.pin(choices.swap_remove(next));
    }

    fn endpoint(&self) -> Option<&str> {
        let value = self.settings[0].trim();
        if value.is_empty() {
            None
        } else {
            Some(value)
        }
    }

    fn status_line(&self) -> String {
        if !self.notice.is_empty() {
            return self.notice.clone();
        }
        let query = Query::new(self.model_catalog, &self.facts);
        match self.choice.status(&query, &self.evidence, self.endpoint()) {
            Ok(resolved) => format!(
                "Ready  {} / {} / {} at {}",
                resolved.benchmark(),
                resolved.route().provider().as_str(),
                resolved.serve_as(),
                self.endpoint().unwrap_or(resolved.base_url())
            ),
            Err(refusal) => format!("Cannot start  {refusal}"),
        }
    }

    fn draw(&self, frame: &mut Frame) {
        let area = frame.area();
        frame.render_widget(
            Block::default().style(Style::default().fg(Color::White).bg(Color::Black)),
            area,
        );
        if area.width < 60 || area.height < 20 {
            frame.render_widget(
                Paragraph::new("q.it\nTerminal too small; resize to 80 x 24.\nq quits"),
                area,
            );
            return;
        }
        let layout = Layout::vertical([
            Constraint::Length(3),
            Constraint::Min(1),
            Constraint::Length(2),
        ])
        .split(area);
        let tabs = Tabs::new(SCREENS.to_vec())
            .select(self.screen)
            .style(Style::default().fg(Color::Gray))
            .highlight_style(Style::default().fg(BLUE).add_modifier(Modifier::BOLD))
            .block(Block::default().borders(Borders::BOTTOM).title(" q.it "));
        frame.render_widget(tabs, layout[0]);
        match self.screen {
            0 => self.draw_benchmarks(frame, layout[1]),
            1 => self.draw_form(frame, layout[1]),
            2 => self.draw_progress(frame, layout[1]),
            3 => self.draw_results(frame, layout[1]),
            4 => self.draw_comparison(frame, layout[1]),
            _ => {}
        }
        let keys = if self.picker.is_some() {
            "Type to search  Up-Down choose  Enter apply  Esc cancel"
        } else if self.editing {
            "Type value  Enter save  Esc done"
        } else if self.filter_editing {
            "Type filter  Enter apply  Esc done"
        } else {
            match self.screen {
                0 => "Up-Down move  Enter setup  b benchmark  p provider  m model  q quit",
                1 => "Up-Down field  Enter choose or edit  Left-Right cycle  p probe  q quit",
                2 => "x interrupts run  q exits and interrupts run",
                _ => "Tab screen  Arrows select  Enter open  / filter  r refresh  q quit",
            }
        };
        frame.render_widget(
            Paragraph::new(vec![
                Line::from(self.status_line()),
                Line::from(keys).style(Style::default().fg(Color::Gray)),
            ])
            .style(Style::default().fg(BLUE))
            .wrap(Wrap { trim: false }),
            layout[2],
        );
        if self.picker.is_some() {
            self.draw_picker(frame);
        }
    }

    fn draw_benchmarks(&self, frame: &mut Frame, area: Rect) {
        let layout = Layout::vertical([Constraint::Min(4), Constraint::Length(7)]).split(area);
        let visible = self.visible_benchmarks();
        let rows = visible.iter().map(|benchmark| {
            let models = self
                .model_catalog
                .routes_matching(self.choice.facets().provider(), Some(benchmark.task))
                .len();
            Row::new([
                benchmark.id.clone(),
                benchmark.task.label().into(),
                benchmark.pack_version.clone(),
                benchmark.case_count.to_string(),
                models.to_string(),
            ])
        });
        render_table(
            frame,
            layout[0],
            "Benchmarks",
            &["Benchmark", "Task", "Version", "Cases", "Models"],
            rows,
            &[
                Constraint::Percentage(42),
                Constraint::Percentage(19),
                Constraint::Length(7),
                Constraint::Length(5),
                Constraint::Length(6),
            ],
            self.selection.min(visible.len().saturating_sub(1)),
        );
        let facets = self.choice.facets();
        let route = facets.route().and_then(|key| self.model_catalog.route(key));
        let model = route.map(|route| {
            let state = if let Implementation::Pending = route.implementation() {
                Availability::AdapterPending
            } else {
                self.availability(route.key(), self.endpoint().unwrap_or(route.base_url()))
            };
            format!(
                "{} / {} / {}",
                route.serve_as(),
                route.implementation().label(),
                state.label()
            )
        });
        let lines = vec![
            Line::from(format!("Benchmark  {}", facets.benchmark().unwrap_or("-"))),
            Line::from(format!(
                "Provider   {}",
                facets
                    .provider()
                    .map(|provider| provider.as_str())
                    .unwrap_or("-")
            )),
            Line::from(format!("Model      {}", model.as_deref().unwrap_or("-"))),
            Line::from(format!(
                "Endpoint   {}",
                self.endpoint()
                    .map(str::to_string)
                    .or_else(|| route.map(|route| route.base_url().to_string()))
                    .unwrap_or_else(|| "-".into())
            )),
            Line::from("Enter opens setup / Left-Right cycles on setup"),
        ];
        frame.render_widget(Paragraph::new(lines).block(panel("Selection")), layout[1]);
    }

    fn draw_form(&self, frame: &mut Frame, area: Rect) {
        let facets = self.choice.facets();
        let query = Query::new(self.model_catalog, &self.facts);
        let task = query.task(facets);
        let route = facets.route().and_then(|key| self.model_catalog.route(key));
        let availability = route.map(|route| {
            if let Implementation::Pending = route.implementation() {
                Availability::AdapterPending
            } else {
                self.availability(route.key(), self.endpoint().unwrap_or(route.base_url()))
            }
        });
        let rows = FIELDS.iter().enumerate().map(|(index, name)| {
            let value = match index {
                0 => facets
                    .benchmark()
                    .map(|id| format!("{id} / {}", task.map(|task| task.label()).unwrap_or("-")))
                    .unwrap_or_else(|| "(none)".into()),
                1 => facets
                    .provider()
                    .map(|provider| provider.label().to_string())
                    .unwrap_or_else(|| "(none)".into()),
                2 => route
                    .map(|route| {
                        let parameters = self
                            .model_catalog
                            .model(route.model())
                            .and_then(|model| model.parameters())
                            .map(|value| self.model_catalog.parameters_label(Some(value)))
                            .unwrap_or_else(|| "?".into());
                        format!(
                            "{} / {} / {} / {}",
                            route.serve_as(),
                            parameters,
                            route.implementation().label(),
                            availability.map(Availability::label).unwrap_or("allowed")
                        )
                    })
                    .unwrap_or_else(|| "(none)".into()),
                3..=8 => {
                    let value = self.settings[index - 3].as_str();
                    if value.is_empty() {
                        "(default)".into()
                    } else {
                        value.to_string()
                    }
                }
                _ => "[ Start ]".into(),
            };
            Row::new([Cell::from(*name), Cell::from(value)])
        });
        let title = if self.editing {
            "Start run / editing"
        } else {
            "Start run / Enter chooses or edits / Left-Right cycles"
        };
        render_table(
            frame,
            area,
            title,
            &["Setting", "Value"],
            rows,
            &[Constraint::Length(16), Constraint::Min(20)],
            self.field,
        );
    }

    fn draw_picker(&self, frame: &mut Frame) {
        let Some(picker) = &self.picker else {
            return;
        };
        let entries = self.picker_matches(picker.axis, &picker.query, picker.all_tasks);
        let area = frame.area();
        let width = area.width.saturating_sub(4).min(88);
        let height = area.height.saturating_sub(2).min(20);
        let area = Rect::new(
            (area.width - width) / 2,
            (area.height - height) / 2,
            width,
            height,
        );
        frame.render_widget(Clear, area);
        let layout = Layout::vertical([
            Constraint::Length(3),
            Constraint::Min(3),
            Constraint::Length(1),
        ])
        .split(area);
        let title = format!("Choose {} / {} matches", picker.axis.label(), entries.len());
        frame.render_widget(
            Paragraph::new(format!("Search: {}", picker.query)).block(panel(&title)),
            layout[0],
        );
        let rows = entries.iter().map(|entry| {
            Row::new([
                Cell::from(entry.label.clone()),
                Cell::from(entry.detail.clone()),
            ])
        });
        render_table(
            frame,
            layout[1],
            "Type to filter",
            &["Value", "Detail"],
            rows,
            &[Constraint::Length(30), Constraint::Min(10)],
            picker.selected.min(entries.len().saturating_sub(1)),
        );
        frame.render_widget(
            Paragraph::new("Up-Down choose  Enter apply  Esc cancel")
                .style(Style::default().fg(BLUE)),
            layout[2],
        );
    }

    fn draw_progress(&self, frame: &mut Frame, area: Rect) {
        let progress = &self.progress;
        if progress.run_id.is_empty() {
            frame.render_widget(
                Paragraph::new("No run started").block(panel("Progress")),
                area,
            );
            return;
        }
        let layout = Layout::vertical([Constraint::Length(3), Constraint::Min(5)]).split(area);
        let ratio = if progress.expected == 0 {
            0.0
        } else {
            (progress.completed as f64 / progress.expected as f64).min(1.0)
        };
        frame.render_widget(
            Gauge::default()
                .block(panel("Measured samples"))
                .gauge_style(Style::default().fg(BLUE))
                .ratio(ratio)
                .label(format!("{} / {}", progress.completed, progress.expected)),
            layout[0],
        );
        let summary = &progress.summary;
        let lines = vec![
            format!("Run       {}", progress.run_id),
            format!("Status    {}", progress.phase),
            format!("Case      {}", progress.case_id.as_deref().unwrap_or("-")),
            format!(
                "Warmups   {} / {}",
                progress.warmups_completed, progress.warmups_expected
            ),
            format!("Latency   {} ms", metric(summary.latency_ms_p50)),
            format!("TTFT      {} ms", metric(summary.ttft_ms_p50)),
            format!("Tokens/s  {}", metric(summary.tokens_per_second_p50)),
            format!(
                "First audio {} ms",
                metric(summary.time_to_first_audio_ms_p50)
            ),
            format!("RTF       {}", ratio_metric(summary.real_time_factor_p50)),
            format!("Audio     {} s", metric(summary.audio_duration_seconds_p50)),
            format!("Host CPU  {} %", metric(summary.host_cpu_percent_mean)),
            format!("Host RAM  {}", memory(summary.host_memory_used_bytes_peak)),
            format!("Server RSS {}", memory(summary.process_rss_bytes_peak)),
            "".into(),
            "x interrupts run   q exits and interrupts run".into(),
        ];
        frame.render_widget(
            Paragraph::new(lines.join("\n"))
                .block(panel("Live progress"))
                .wrap(Wrap { trim: false }),
            layout[1],
        );
    }

    fn draw_results(&self, frame: &mut Frame, area: Rect) {
        if let Some(run) = &self.detail {
            let mut lines = vec![
                format!("{} / {}", run.id, run.status.as_str()),
                format!("{} / {} / {}", run.benchmark_id, run.provider, run.model),
                format!("Endpoint {}", run.base_url),
                format!("Error {}", run.error.as_deref().unwrap_or("-")),
            ];
            for sample in &self.samples {
                lines.push(String::new());
                lines.push(format!(
                    "{} / pass {} / {}",
                    sample.case_id,
                    sample.iteration,
                    if sample.succeeded {
                        "succeeded"
                    } else {
                        "failed"
                    }
                ));
                lines.push(format!(
                    "Latency {} ms / TTFT {} ms / TPS {} / quality {}",
                    metric(sample.latency_ms),
                    metric(sample.ttft_ms),
                    metric(sample.tokens_per_second),
                    metric(sample.quality_score)
                ));
                lines.push(format!(
                    "CPU {} % / RAM {} / RSS {}",
                    metric(sample.host_cpu_percent_mean),
                    memory(sample.host_memory_used_bytes_peak),
                    memory(sample.process_rss_bytes_peak)
                ));
                if let Some(error) = &sample.error {
                    lines.push(format!("Error: {error}"));
                }
                if let Some(speech) = &sample.speech {
                    lines.push(format!(
                        "Audio duration {} s",
                        metric(speech.audio_duration_seconds)
                    ));
                    lines.push(speech.transcript.clone());
                    if let Some(routing) = &speech.routing {
                        lines.push(serde_json::to_string_pretty(routing).unwrap_or_default());
                    }
                } else if let Some(synthesis) = &sample.synthesis {
                    lines.push(format!(
                        "First audio {} ms / duration {} s / {} Hz / {} bytes",
                        metric(Some(synthesis.time_to_first_audio_ms)),
                        metric(Some(synthesis.audio_duration_seconds)),
                        synthesis.sample_rate_hz,
                        synthesis.audio_bytes
                    ));
                    if let Some(routing) = &synthesis.routing {
                        lines.push(serde_json::to_string_pretty(routing).unwrap_or_default());
                    }
                } else if let Some(output) = &sample.output_excerpt {
                    lines.push(output.clone());
                }
                if let Some(metadata) = &sample.output_metadata {
                    lines.push(format!("Output evidence: {metadata}"));
                }
                if let Some(image) = &sample.output_image {
                    lines.push(format!(
                        "Saved image: {} bytes (export to view)",
                        image.len()
                    ));
                }
                if let Some(decision) = &sample.decision {
                    lines.push(serde_json::to_string_pretty(decision).unwrap_or_default());
                }
                if let Some(retrieval) = &sample.retrieval {
                    lines.push(format!("MaxSim / {} query vectors", retrieval.query.len()));
                    for index in &retrieval.ranking {
                        lines.push(format!(
                            "Document {} / score {:.6} / {} vectors",
                            index + 1,
                            retrieval.scores[*index],
                            retrieval.documents[*index].len()
                        ));
                    }
                    lines.push("Complete token vectors are retained in JSONL exports".into());
                }
            }
            frame.render_widget(
                Paragraph::new(lines.join("\n"))
                    .block(panel("Run detail / Up-Down scroll / Esc back"))
                    .wrap(Wrap { trim: false })
                    .scroll((self.detail_scroll, 0)),
                area,
            );
            return;
        }
        let rows = self.filtered_runs().into_iter().map(|run| {
            Row::new([
                run.status.as_str().into(),
                run.benchmark_id.clone(),
                run.provider.as_str().into(),
                run.model.clone(),
                metric(run.summary.ttft_ms_p50),
                metric(run.summary.tokens_per_second_p50),
                metric(run.summary.latency_ms_p50),
                metric(run.summary.quality_score_mean),
            ])
        });
        let title = format!(
            "Results / {} / filter: {} / s toggles success / / filters",
            if self.succeeded_only {
                "succeeded"
            } else {
                "all"
            },
            self.filter
        );
        render_table(
            frame,
            area,
            &title,
            &[
                "Status",
                "Benchmark",
                "Provider",
                "Model",
                "TTFT",
                "TPS",
                "ms",
                "Quality",
            ],
            rows,
            &[
                Constraint::Length(9),
                Constraint::Percentage(22),
                Constraint::Length(10),
                Constraint::Percentage(20),
                Constraint::Length(7),
                Constraint::Length(6),
                Constraint::Length(8),
                Constraint::Min(6),
            ],
            self.selection,
        );
    }

    fn draw_comparison(&self, frame: &mut Frame, area: Rect) {
        let Some(group) = self.groups.get(self.group) else {
            frame.render_widget(
                Paragraph::new("No successful runs to compare").block(panel("Comparison")),
                area,
            );
            return;
        };
        let reference = &group.runs[0];
        let metric = Metric::ALL[self.metric];
        let layout = Layout::vertical([Constraint::Length(4), Constraint::Min(1)]).split(area);
        let info = format!("{} / v{} / group {} of {}\n{} / {} / {} passes / {} warmups / {} tokens\nUp-Down group   Left-Right metric: {}", reference.benchmark_id, reference.pack_version, self.group + 1, self.groups.len(), reference.host_name, reference.host.architecture, reference.iterations, reference.warmups, reference.max_output_tokens, metric.name());
        frame.render_widget(
            Paragraph::new(info).style(Style::default().fg(BLUE)),
            layout[0],
        );
        let ranking = rank(&group.models, metric);
        let rows = ranking.iter().map(|ranked| {
            let run = group
                .models
                .iter()
                .find(|run| run.id == ranked.run_id)
                .unwrap();
            Row::new([
                ranked.rank.to_string(),
                run.model.clone(),
                run.provider.as_str().into(),
                format!("{:.3}", ranked.value),
            ])
        });
        let missing = group.models.len() - ranking.len();
        render_table(
            frame,
            layout[1],
            &format!("{} / {} missing measurements", metric.name(), missing),
            &["Rank", "Model", "Provider", "Value"],
            rows,
            &[
                Constraint::Length(5),
                Constraint::Percentage(50),
                Constraint::Length(14),
                Constraint::Min(10),
            ],
            0,
        );
    }
}

fn panel(title: &str) -> Block<'_> {
    Block::default()
        .borders(Borders::ALL)
        .border_type(ratatui::widgets::BorderType::Plain)
        .border_style(Style::default().fg(BLUE))
        .title(title)
}

fn render_table<'a>(
    frame: &mut Frame,
    area: Rect,
    title: &'a str,
    headers: &[&'a str],
    rows: impl IntoIterator<Item = Row<'a>>,
    widths: &[Constraint],
    selected: usize,
) {
    let table = Table::new(rows, widths)
        .header(
            Row::new(headers.iter().copied())
                .style(Style::default().fg(BLUE).add_modifier(Modifier::BOLD)),
        )
        .block(panel(title))
        .row_highlight_style(Style::default().fg(Color::Black).bg(BLUE))
        .highlight_symbol("> ");
    let mut state = TableState::default().with_selected(Some(selected));
    frame.render_stateful_widget(table, area, &mut state);
}

fn metric(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.1}"))
        .unwrap_or_else(|| "-".into())
}

fn ratio_metric(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.2}"))
        .unwrap_or_else(|| "-".into())
}

fn memory(value: Option<u64>) -> String {
    value
        .map(|value| format!("{:.1} MiB", value as f64 / 1_048_576.0))
        .unwrap_or_else(|| "-".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ProviderKind;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn app() -> App {
        let directory = tempfile::tempdir().unwrap();
        let paths = Paths::new(directory.path().into());
        paths.ensure().unwrap();
        let store = Store::open(&paths.database).unwrap();
        let catalog = PackCatalog::new(paths.packs.clone());
        App::new(paths, catalog, store).unwrap()
    }

    fn press(app: &mut App, code: KeyCode) {
        app.key(KeyEvent::new(code, KeyModifiers::NONE)).unwrap();
    }

    fn type_text(app: &mut App, text: &str) {
        for character in text.chars() {
            press(app, KeyCode::Char(character));
        }
    }

    #[test]
    fn five_screens_fit_an_ssh_terminal() {
        let mut app = app();
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        for (screen, title) in SCREENS.iter().enumerate() {
            app.screen = screen;
            terminal.draw(|frame| app.draw(frame)).unwrap();
            let buffer = terminal.backend().buffer();
            let text = buffer
                .content
                .iter()
                .map(|cell| cell.symbol())
                .collect::<String>();
            assert!(text.contains(title));
            assert!(text.contains("q.it"));
        }
    }

    #[test]
    fn model_picker_searches_and_commits_without_downloading() {
        let mut app = app();
        app.pin(Choice::Benchmark("core/text-to-speech-smoke".into()));
        app.screen = 1;
        app.field = 2;
        press(&mut app, KeyCode::Enter);
        assert!(app.picker.is_some());
        type_text(&mut app, "phonon");
        let entries = app.picker_matches(Axis::Model, "phonon", false);
        assert_eq!(entries.len(), 2);
        press(&mut app, KeyCode::Enter);
        assert!(app.picker.is_none());
        let key = app.choice.facets().route().unwrap();
        let route = app.model_catalog.route(key).unwrap();
        assert!(route.model().as_str().starts_with("gradium/phonon"));
        assert_eq!(app.settings[0], route.base_url());
    }

    #[test]
    fn picker_renders_inside_eighty_by_twenty_four() {
        let mut app = app();
        app.open_picker(Axis::Model, false);
        type_text(&mut app, "qwen");
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal.draw(|frame| app.draw(frame)).unwrap();
        let text = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(text.contains("Choose Model"));
        assert!(text.contains("Search: qwen"));
    }

    #[test]
    fn benchmark_first_narrows_models_and_clears_the_old_one() {
        let mut app = app();
        let text_model = app.choice.facets().route().cloned().unwrap();
        app.pin(Choice::Benchmark("core/text-to-speech-smoke".into()));
        assert_eq!(
            app.choice.facets().benchmark(),
            Some("core/text-to-speech-smoke")
        );
        assert_ne!(app.choice.facets().route(), Some(&text_model));
        let providers = {
            let query = Query::new(app.model_catalog, &app.facts);
            query.providers(app.choice.facets())
        };
        assert!(providers.contains(&ProviderKind::LocalCpu));
        assert!(!providers.contains(&ProviderKind::Transformers));
    }

    #[test]
    fn switching_benchmark_clears_an_incompatible_model_and_focuses_it() {
        let mut app = app();
        app.pin(Choice::Benchmark("core/text-to-speech-smoke".into()));
        let speech = {
            let query = Query::new(app.model_catalog, &app.facts);
            query
                .routes(app.choice.facets())
                .first()
                .map(|route| Choice::Model(route.key().clone()))
                .unwrap()
        };
        app.pin(speech);
        assert!(app.choice.facets().route().is_some());
        app.pin(Choice::Benchmark("core/text-generation-smoke".into()));
        assert!(app.choice.facets().route().is_none());
        assert_eq!(app.field, 2);
        assert!(app.notice.contains("cleared"));
    }

    #[test]
    fn choosing_a_model_first_clears_an_incompatible_benchmark() {
        let mut app = app();
        app.screen = 0;
        press(&mut app, KeyCode::Char('m'));
        type_text(&mut app, "phonon");
        press(&mut app, KeyCode::Enter);
        assert!(app.choice.facets().benchmark().is_none());
        assert!(app.choice.facets().route().is_some());
        assert_eq!(app.field, 0);
        assert!(app.notice.contains("cleared"));
    }

    #[test]
    fn typed_settings_are_preserved_while_editing() {
        let mut app = app();
        app.screen = 1;
        app.field = 3;
        press(&mut app, KeyCode::Enter);
        press(&mut app, KeyCode::Backspace);
        type_text(&mut app, "q");
        assert!(app.settings[0].ends_with('q'));
        assert!(app.editing);
    }
}
