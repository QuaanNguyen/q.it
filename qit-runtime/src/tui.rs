use std::io::{self, IsTerminal};
use std::sync::mpsc::{self, Receiver};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Line;
use ratatui::widgets::{
    Block, Borders, Cell, Gauge, Paragraph, Row, Table, TableState, Tabs, Wrap,
};
use ratatui::Frame;

use crate::analysis::{analyze, rank, ComparisonGroup, Metric};
use crate::command::{parse_run_request, workspace};
use crate::model::{BenchmarkDefinition, ProviderKind, RunRecord, RunStatus, SampleRecord};
use crate::packs::PackCatalog;
use crate::paths::Paths;
use crate::runner::{execute_with_control, prepare_benchmark, ExecutionControl, RunProgress};
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

struct App {
    paths: Paths,
    catalog: PackCatalog,
    store: Store,
    benchmarks: Vec<BenchmarkDefinition>,
    runs: Vec<RunRecord>,
    groups: Vec<ComparisonGroup>,
    screen: usize,
    selection: usize,
    form: Vec<String>,
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
    message: String,
}

impl App {
    fn new(paths: Paths, catalog: PackCatalog, store: Store) -> Result<Self, String> {
        let benchmarks = catalog.benchmarks()?;
        let mut app = Self {
            paths,
            catalog,
            store,
            benchmarks,
            runs: Vec::new(),
            groups: Vec::new(),
            screen: 0,
            selection: 0,
            form: vec![
                String::new(),
                "ollama".into(),
                String::new(),
                "http://127.0.0.1:11434".into(),
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
            message: String::new(),
        };
        app.choose_benchmark(0);
        app.refresh()?;
        Ok(app)
    }

    fn refresh(&mut self) -> Result<(), String> {
        self.runs = self.store.runs(10_000, false)?;
        self.groups = analyze(self.runs.clone()).groups;
        self.group = self.group.min(self.groups.len().saturating_sub(1));
        Ok(())
    }

    fn choose_benchmark(&mut self, index: usize) {
        if let Some(benchmark) = self.benchmarks.get(index) {
            self.form[0] = benchmark.id.clone();
            self.form[6].clear();
            if !benchmark
                .supported_providers
                .iter()
                .any(|provider| provider.as_str() == self.form[1])
            {
                if let Some(provider) = benchmark.supported_providers.first() {
                    self.form[1] = provider.as_str().into();
                    self.form[3] = provider.default_base_url().into();
                }
            }
        }
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
        let Some(worker) = &self.worker else {
            return Ok(false);
        };
        let events = worker.receiver.try_iter().collect::<Vec<_>>();
        let changed = !events.is_empty();
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
                    self.message = match result {
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
            self.message = "A benchmark is already running".into();
            return Ok(());
        }
        let mut arguments = vec![self.form[0].clone()];
        for (index, name) in [
            "provider",
            "model",
            "base-url",
            "iterations",
            "warmups",
            "max-tokens",
            "timeout",
            "pid",
        ]
        .into_iter()
        .enumerate()
        {
            let value = &self.form[index + 1];
            if !value.is_empty() {
                arguments.push(format!("--{name}"));
                arguments.push(value.clone());
            }
        }
        let request = match parse_run_request(&arguments) {
            Ok(request) => request,
            Err(error) => {
                self.message = error;
                return Ok(());
            }
        };
        let prepared = match prepare_benchmark(&self.store, &self.catalog, request) {
            Ok(prepared) => prepared,
            Err(error) => {
                self.message = error;
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
        self.message.clear();
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
                    self.form[self.field].pop();
                }
                KeyCode::Char(character) if !character.is_control() => {
                    self.form[self.field].push(character)
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
                        (self.selection + 1).min(self.benchmarks.len().saturating_sub(1))
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    self.selection = self.selection.saturating_sub(1)
                }
                KeyCode::Enter => {
                    self.choose_benchmark(self.selection);
                    self.screen = 1;
                    self.field = 2;
                }
                _ => {}
            },
            1 => match key.code {
                KeyCode::Down | KeyCode::Char('j') => self.field = (self.field + 1) % FIELDS.len(),
                KeyCode::Up | KeyCode::Char('k') => {
                    self.field = (self.field + FIELDS.len() - 1) % FIELDS.len()
                }
                KeyCode::Left | KeyCode::Right if self.field <= 1 => {
                    self.cycle_field(key.code == KeyCode::Right)
                }
                KeyCode::Enter if self.field == 9 => self.start()?,
                KeyCode::Enter => self.editing = true,
                _ => {}
            },
            2 => {
                if key.code == KeyCode::Char('x') {
                    if let Some(worker) = &self.worker {
                        worker.control.cancel();
                        self.message = "Interrupting benchmark".into();
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
        if self.field == 0 {
            let current = self
                .benchmarks
                .iter()
                .position(|benchmark| benchmark.id == self.form[0])
                .unwrap_or(0);
            if !self.benchmarks.is_empty() {
                let next = if forward {
                    (current + 1) % self.benchmarks.len()
                } else {
                    (current + self.benchmarks.len() - 1) % self.benchmarks.len()
                };
                self.choose_benchmark(next);
            }
        } else {
            let providers = self
                .benchmarks
                .iter()
                .find(|benchmark| benchmark.id == self.form[0])
                .map(|benchmark| benchmark.supported_providers.clone())
                .unwrap_or_else(|| ProviderKind::ALL.to_vec());
            if !providers.is_empty() {
                let current = providers
                    .iter()
                    .position(|provider| provider.as_str() == self.form[1])
                    .unwrap_or(0);
                let next = if forward {
                    (current + 1) % providers.len()
                } else {
                    (current + providers.len() - 1) % providers.len()
                };
                self.form[1] = providers[next].as_str().into();
                self.form[3] = providers[next].default_base_url().into();
            }
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
        let keys = if self.editing {
            "Type value  Enter save  Esc done"
        } else if self.filter_editing {
            "Type filter  Enter apply  Esc done"
        } else {
            "Tab screen  Arrows select  Enter open  r refresh  q quit"
        };
        let message = if self.message.is_empty() {
            keys
        } else {
            &self.message
        };
        frame.render_widget(
            Paragraph::new(message)
                .style(Style::default().fg(BLUE))
                .wrap(Wrap { trim: false }),
            layout[2],
        );
    }

    fn draw_benchmarks(&self, frame: &mut Frame, area: Rect) {
        let layout = Layout::vertical([Constraint::Min(4), Constraint::Length(7)]).split(area);
        let rows = self.benchmarks.iter().map(|benchmark| {
            Row::new([
                benchmark.id.clone(),
                benchmark.task.label().into(),
                benchmark.pack_version.clone(),
                benchmark.case_count.to_string(),
                benchmark
                    .supported_providers
                    .iter()
                    .map(|provider| provider.as_str())
                    .collect::<Vec<_>>()
                    .join(", "),
            ])
        });
        render_table(
            frame,
            layout[0],
            "Benchmarks",
            &["Benchmark", "Task", "Version", "Cases", "Providers"],
            rows,
            &[
                Constraint::Percentage(34),
                Constraint::Percentage(20),
                Constraint::Length(7),
                Constraint::Length(5),
                Constraint::Min(10),
            ],
            self.selection,
        );
        let providers = ProviderKind::ALL
            .into_iter()
            .map(|provider| {
                Line::from(format!(
                    "{:<16} {}",
                    provider.as_str(),
                    provider.default_base_url()
                ))
            })
            .collect::<Vec<_>>();
        frame.render_widget(
            Paragraph::new(providers).block(panel("Providers")),
            layout[1],
        );
    }

    fn draw_form(&self, frame: &mut Frame, area: Rect) {
        let rows = FIELDS.iter().enumerate().map(|(index, name)| {
            let value = self
                .form
                .get(index)
                .map(String::as_str)
                .unwrap_or("[ Start ]");
            let value = if value.is_empty() { "(default)" } else { value };
            Row::new([Cell::from(*name), Cell::from(value)])
        });
        let title = if self.editing {
            "Start run / editing"
        } else {
            "Start run / Enter edits / Left-Right cycles benchmark or provider"
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
                if let Some(output) = &sample.output_excerpt {
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

fn memory(value: Option<u64>) -> String {
    value
        .map(|value| format!("{:.1} MiB", value as f64 / 1_048_576.0))
        .unwrap_or_else(|| "-".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    #[test]
    fn five_screens_fit_an_ssh_terminal_and_form_preserves_typed_model() {
        let directory = tempfile::tempdir().unwrap();
        let paths = Paths::new(directory.path().into());
        paths.ensure().unwrap();
        let store = Store::open(&paths.database).unwrap();
        let catalog = PackCatalog::new(paths.packs.clone());
        let mut app = App::new(paths, catalog, store).unwrap();
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
        app.screen = 1;
        app.field = 2;
        app.editing = true;
        app.key(KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE))
            .unwrap();
        assert_eq!(app.form[2], "q");
        assert!(app.editing);
    }
}
