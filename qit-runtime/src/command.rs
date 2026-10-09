use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::Path;
use std::str::FromStr;

use crate::config::{home_from_env, Config};
use crate::model::{ProviderKind, RunRecord, RunRequest, RunStatus, Task};
use crate::packs::PackCatalog;
use crate::paths::Paths;
use crate::runner::run_benchmark;
use crate::store::Store;

pub fn run() -> i32 {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("qit_runtime=warn".parse().unwrap()),
        )
        .try_init();
    match execute(std::env::args().skip(1).collect()) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("error: {error}");
            1
        }
    }
}

fn execute(arguments: Vec<String>) -> Result<i32, String> {
    let Some(command) = arguments.first().map(String::as_str) else {
        print_help();
        return Ok(0);
    };
    let rest = &arguments[1..];
    match command {
        "help" | "--help" | "-h" => {
            print_help();
            Ok(0)
        }
        "version" | "--version" | "-V" => {
            println!("qit {}", env!("CARGO_PKG_VERSION"));
            Ok(0)
        }
        "benchmark" | "benchmarks" => benchmark_command(rest),
        "provider" | "providers" => provider_command(rest),
        "run" => run_command(rest),
        "results" => results_command(rest),
        "export" => export_command(rest),
        "dashboard" => dashboard_command(rest),
        unknown => Err(format!(
            "unknown command '{unknown}'\nRun 'qit help' to see available commands."
        )),
    }
}

fn benchmark_command(arguments: &[String]) -> Result<i32, String> {
    let action = arguments.first().map(String::as_str).unwrap_or("list");
    match action {
        "list" => {
            let options = Options::parse(&arguments[1..], &["provider"], &[])?;
            options.require_no_positionals()?;
            let provider = options
                .value("provider")
                .map(ProviderKind::from_str)
                .transpose()?;
            let (_, catalog, _) = workspace()?;
            let benchmarks = catalog.benchmarks()?;
            println!(
                "{:<34} {:<19} {:<7} PROVIDERS",
                "BENCHMARK", "TASK", "CASES"
            );
            for benchmark in benchmarks {
                if let Some(provider) = provider {
                    println!(
                        "{:<34} {:<19} {:<7} {}",
                        benchmark.id,
                        benchmark.task.as_str(),
                        benchmark.case_count,
                        if provider.supports(benchmark.task) {
                            "supported"
                        } else {
                            "unsupported"
                        }
                    );
                } else {
                    let providers = benchmark
                        .supported_providers
                        .iter()
                        .map(|provider| provider.as_str())
                        .collect::<Vec<_>>()
                        .join(",");
                    let providers = if providers.is_empty() {
                        "reserved".to_string()
                    } else {
                        providers
                    };
                    println!(
                        "{:<34} {:<19} {:<7} {}",
                        benchmark.id,
                        benchmark.task.as_str(),
                        benchmark.case_count,
                        providers
                    );
                }
            }
            Ok(0)
        }
        "install" => {
            let options = Options::parse(&arguments[1..], &[], &[])?;
            let source = options.one_positional("benchmark pack directory")?;
            let (_, catalog, _) = workspace()?;
            let pack_id = catalog.install(Path::new(source))?;
            println!("Installed benchmark pack {pack_id}");
            Ok(0)
        }
        "verify" => {
            let options = Options::parse(&arguments[1..], &[], &[])?;
            let source = options.one_positional("benchmark pack directory")?;
            let (_, catalog, _) = workspace()?;
            let verification = catalog.verify(Path::new(source))?;
            println!("Pack: {} {}", verification.id, verification.version);
            println!("Benchmarks: {}", verification.benchmark_count);
            println!("Cases: {}", verification.case_count);
            println!("Status: valid");
            Ok(0)
        }
        "remove" => {
            let options = Options::parse(&arguments[1..], &[], &[])?;
            let pack_id = options.one_positional("benchmark pack id")?;
            let (_, catalog, _) = workspace()?;
            catalog.remove(pack_id)?;
            println!("Removed benchmark pack {pack_id}");
            Ok(0)
        }
        "help" | "--help" | "-h" => {
            print_benchmark_help();
            Ok(0)
        }
        unknown => Err(format!("unknown benchmark action '{unknown}'")),
    }
}

fn provider_command(arguments: &[String]) -> Result<i32, String> {
    let action = arguments.first().map(String::as_str).unwrap_or("list");
    match action {
        "list" => {
            if arguments.len() > 1 {
                return Err("provider list takes no arguments".into());
            }
            println!("{:<13} {:<24} TASKS", "PROVIDER", "DEFAULT URL");
            for provider in ProviderKind::ALL {
                let tasks = Task::ALL
                    .into_iter()
                    .filter(|task| provider.supports(*task))
                    .map(Task::as_str)
                    .collect::<Vec<_>>()
                    .join(",");
                println!(
                    "{:<13} {:<24} {}",
                    provider.as_str(),
                    provider.default_base_url(),
                    tasks
                );
            }
            Ok(0)
        }
        "help" | "--help" | "-h" => {
            println!("Usage: qit provider list");
            Ok(0)
        }
        unknown => Err(format!("unknown provider action '{unknown}'")),
    }
}

fn run_command(arguments: &[String]) -> Result<i32, String> {
    if arguments
        .iter()
        .any(|value| value == "--help" || value == "-h")
    {
        print_run_help();
        return Ok(0);
    }
    let options = Options::parse(
        arguments,
        &[
            "provider",
            "model",
            "base-url",
            "api-key",
            "iterations",
            "warmups",
            "max-tokens",
            "timeout",
            "pid",
        ],
        &[],
    )?;
    let benchmark_id = options.one_positional("benchmark id")?.to_string();
    let provider = options
        .value("provider")
        .map(ProviderKind::from_str)
        .transpose()?
        .unwrap_or(ProviderKind::Ollama);
    let model = options
        .value("model")
        .ok_or_else(|| "--model is required".to_string())?
        .to_string();
    let base_url = options
        .value("base-url")
        .map(str::to_string)
        .unwrap_or_else(|| provider.default_base_url().to_string());
    let api_key = options
        .value("api-key")
        .map(str::to_string)
        .or_else(|| std::env::var("QIT_API_KEY").ok())
        .or_else(|| match provider {
            ProviderKind::Transformers => std::env::var("HF_TOKEN")
                .ok()
                .or_else(|| std::env::var("OPENAI_API_KEY").ok()),
            ProviderKind::Tei | ProviderKind::HfServe => std::env::var("HF_TOKEN").ok(),
            ProviderKind::Ollama => None,
        });
    let request = RunRequest {
        benchmark_id,
        provider,
        base_url,
        api_key,
        model,
        iterations: options.number("iterations")?.unwrap_or(3),
        warmups: options.number("warmups")?.unwrap_or(1),
        max_output_tokens: options.number("max-tokens")?,
        timeout_seconds: options.number("timeout")?.unwrap_or(120),
        target_pid: options.number("pid")?,
    };
    let (_, catalog, store) = workspace()?;
    println!(
        "Running {} with {} against {}",
        request.benchmark_id, request.provider, request.model
    );
    let runtime = runtime()?;
    let run = runtime.block_on(run_benchmark(&store, &catalog, request))?;
    print_run_summary(&run);
    Ok(if run.status == RunStatus::Succeeded {
        0
    } else {
        2
    })
}

fn results_command(arguments: &[String]) -> Result<i32, String> {
    if arguments
        .iter()
        .any(|value| value == "--help" || value == "-h")
    {
        println!("Usage: qit results [--limit N] [--succeeded]");
        return Ok(0);
    }
    let options = Options::parse(arguments, &["limit"], &["succeeded"])?;
    options.require_no_positionals()?;
    let limit = options
        .number::<usize>("limit")?
        .unwrap_or(50)
        .clamp(1, 10_000);
    let (_, _, store) = workspace()?;
    let runs = store.runs(limit, options.flag("succeeded"))?;
    if runs.is_empty() {
        println!("No benchmark runs yet.");
        return Ok(0);
    }
    println!(
        "{:<9} {:<10} {:<31} {:<12} {:<20} {:>9} {:>9}",
        "RUN", "STATUS", "BENCHMARK", "PROVIDER", "MODEL", "TTFT", "TPS"
    );
    for run in runs {
        println!(
            "{:<9} {:<10} {:<31} {:<12} {:<20} {:>9} {:>9}",
            short_id(&run.id),
            run.status.as_str(),
            truncate(&run.benchmark_id, 31),
            run.provider.as_str(),
            truncate(&run.model, 20),
            format_metric(run.summary.ttft_ms_p50, "ms"),
            format_metric(run.summary.tokens_per_second_p50, "")
        );
    }
    Ok(0)
}

fn export_command(arguments: &[String]) -> Result<i32, String> {
    if arguments
        .iter()
        .any(|value| value == "--help" || value == "-h")
    {
        println!("Usage: qit export [--format jsonl|csv] [--output PATH]");
        return Ok(0);
    }
    let options = Options::parse(arguments, &["format", "output"], &[])?;
    options.require_no_positionals()?;
    let format = options.value("format").unwrap_or("jsonl");
    let (_, _, store) = workspace()?;
    let runs = store.runs(100_000, true)?;
    let data = match format {
        "jsonl" => export_jsonl(&store, &runs)?,
        "csv" => export_csv(&runs),
        _ => return Err("--format must be jsonl or csv".into()),
    };
    if let Some(path) = options.value("output") {
        std::fs::write(path, data.as_bytes())
            .map_err(|error| format!("write export {}: {error}", path))?;
        println!("Exported {} successful runs to {path}", runs.len());
    } else {
        print!("{data}");
        std::io::stdout()
            .flush()
            .map_err(|error| format!("write export: {error}"))?;
    }
    Ok(0)
}

fn dashboard_command(arguments: &[String]) -> Result<i32, String> {
    if arguments
        .iter()
        .any(|value| value == "--help" || value == "-h")
    {
        println!("Usage: qit dashboard [--host ADDRESS] [--port PORT]");
        return Ok(0);
    }
    let options = Options::parse(arguments, &["host", "port"], &[])?;
    options.require_no_positionals()?;
    let port = options.number::<u16>("port")?;
    let config = Config::from_env()?.with_dashboard_overrides(options.value("host"), port)?;
    let runtime = runtime()?;
    runtime.block_on(async move {
        let listening = crate::bind(config).await?;
        println!("q.it dashboard listening on {}", listening.base_url());
        std::io::stdout()
            .flush()
            .map_err(|error| format!("write dashboard address: {error}"))?;
        tokio::signal::ctrl_c()
            .await
            .map_err(|error| format!("wait for shutdown signal: {error}"))?;
        listening.shutdown().await;
        Ok::<_, String>(())
    })?;
    Ok(0)
}

fn workspace() -> Result<(Paths, PackCatalog, Store), String> {
    let paths = Paths::new(home_from_env());
    paths.ensure()?;
    let catalog = PackCatalog::new(paths.packs.clone());
    let store = Store::open(&paths.database)?;
    Ok((paths, catalog, store))
}

fn runtime() -> Result<tokio::runtime::Runtime, String> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|error| format!("start async runtime: {error}"))
}

fn print_run_summary(run: &RunRecord) {
    println!("Run: {}", run.id);
    println!("Status: {}", run.status.as_str());
    println!(
        "Samples: {}/{} succeeded",
        run.summary.successful_sample_count, run.summary.sample_count
    );
    println!(
        "Latency p50: {}",
        format_metric(run.summary.latency_ms_p50, "ms")
    );
    println!("TTFT p50: {}", format_metric(run.summary.ttft_ms_p50, "ms"));
    println!(
        "Tokens/s p50: {}",
        format_metric(run.summary.tokens_per_second_p50, "")
    );
    println!(
        "CPU mean: {}",
        format_metric(run.summary.host_cpu_percent_mean, "%")
    );
    println!(
        "RAM peak: {}",
        run.summary
            .host_memory_used_bytes_peak
            .map(format_bytes)
            .unwrap_or_else(|| "n/a".into())
    );
    if let Some(error) = &run.error {
        println!("Failure: {error}");
    }
}

fn export_jsonl(store: &Store, runs: &[RunRecord]) -> Result<String, String> {
    let mut output = String::new();
    for run in runs {
        let samples = store.samples(&run.id)?;
        let value = serde_json::json!({ "run": run, "samples": samples });
        output.push_str(
            &serde_json::to_string(&value).map_err(|error| format!("serialize export: {error}"))?,
        );
        output.push('\n');
    }
    Ok(output)
}

fn export_csv(runs: &[RunRecord]) -> String {
    let mut output = String::from(
        "run_id,benchmark_id,task,provider,model,host_name,started_at_ms,latency_ms_p50,ttft_ms_p50,tokens_per_second_p50,cpu_percent_mean,memory_used_bytes_peak,process_rss_bytes_peak,quality_score_mean\n",
    );
    for run in runs {
        let fields = [
            run.id.clone(),
            run.benchmark_id.clone(),
            run.task.as_str().into(),
            run.provider.as_str().into(),
            run.model.clone(),
            run.host_name.clone(),
            run.started_at_ms.to_string(),
            optional_string(run.summary.latency_ms_p50),
            optional_string(run.summary.ttft_ms_p50),
            optional_string(run.summary.tokens_per_second_p50),
            optional_string(run.summary.host_cpu_percent_mean),
            run.summary
                .host_memory_used_bytes_peak
                .map(|value| value.to_string())
                .unwrap_or_default(),
            run.summary
                .process_rss_bytes_peak
                .map(|value| value.to_string())
                .unwrap_or_default(),
            optional_string(run.summary.quality_score_mean),
        ];
        output.push_str(
            &fields
                .iter()
                .map(|field| csv(field))
                .collect::<Vec<_>>()
                .join(","),
        );
        output.push('\n');
    }
    output
}

fn csv(value: &str) -> String {
    if value.contains([',', '"', '\n']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn optional_string(value: Option<f64>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

fn short_id(value: &str) -> &str {
    value.get(..8).unwrap_or(value)
}

fn truncate(value: &str, length: usize) -> String {
    if value.chars().count() <= length {
        value.to_string()
    } else {
        let mut shortened = value
            .chars()
            .take(length.saturating_sub(1))
            .collect::<String>();
        shortened.push('…');
        shortened
    }
}

fn format_metric(value: Option<f64>, suffix: &str) -> String {
    value
        .map(|value| format!("{value:.1}{suffix}"))
        .unwrap_or_else(|| "n/a".into())
}

fn format_bytes(bytes: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    if bytes as f64 >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB)
    } else {
        format!("{:.1} MiB", bytes as f64 / MIB)
    }
}

fn print_help() {
    println!(
        "q.it - Raspberry Pi edge-model benchmarks\n\n\
Usage:\n  qit <command> [options]\n\n\
Commands:\n  benchmark list              List installed benchmarks\n  benchmark verify PATH       Validate a benchmark pack\n  benchmark install PATH      Install a declarative benchmark pack\n  benchmark remove ID         Remove an installed benchmark pack\n  provider list               List serving protocols and supported tasks\n  run BENCHMARK               Run a benchmark and store its result\n  results                     Show run history\n  export                      Export the successful run dataset\n  dashboard                   Start the local analysis dashboard\n  help                        Show this help\n\n\
Run 'qit run --help' for execution options."
    );
}

fn print_benchmark_help() {
    println!(
        "Usage:\n  qit benchmark list [--provider PROVIDER]\n  qit benchmark verify PATH\n  qit benchmark install PATH\n  qit benchmark remove ID"
    );
}

fn print_run_help() {
    println!(
        "Usage: qit run BENCHMARK --model MODEL [options]\n\n\
Options:\n  --provider NAME      ollama, transformers, hf-serve, or tei (default: ollama)\n  --base-url URL       Provider root URL\n  --api-key TOKEN      Bearer token, or use QIT_API_KEY\n  --iterations N       Measured iterations (default: 3)\n  --warmups N          Warmup requests (default: 1)\n  --max-tokens N       Generation limit from the benchmark by default\n  --timeout SECONDS    Per-request timeout (default: 120)\n  --pid PID            Provider process to sample for peak RSS"
    );
}

struct Options {
    values: HashMap<String, String>,
    flags: HashSet<String>,
    positionals: Vec<String>,
}

impl Options {
    fn parse(
        arguments: &[String],
        value_names: &[&str],
        flag_names: &[&str],
    ) -> Result<Self, String> {
        let value_names = value_names.iter().copied().collect::<HashSet<_>>();
        let flag_names = flag_names.iter().copied().collect::<HashSet<_>>();
        let mut values = HashMap::new();
        let mut flags = HashSet::new();
        let mut positionals = Vec::new();
        let mut index = 0;
        while index < arguments.len() {
            let argument = &arguments[index];
            if let Some(option) = argument.strip_prefix("--") {
                if let Some((name, value)) = option.split_once('=') {
                    if !value_names.contains(name) {
                        return Err(format!("unknown option '--{name}'"));
                    }
                    if values.insert(name.to_string(), value.to_string()).is_some() {
                        return Err(format!("option '--{name}' was provided more than once"));
                    }
                } else if value_names.contains(option) {
                    index += 1;
                    let value = arguments
                        .get(index)
                        .filter(|value| !value.starts_with("--"))
                        .ok_or_else(|| format!("option '--{option}' requires a value"))?;
                    if values.insert(option.to_string(), value.clone()).is_some() {
                        return Err(format!("option '--{option}' was provided more than once"));
                    }
                } else if flag_names.contains(option) {
                    if !flags.insert(option.to_string()) {
                        return Err(format!("flag '--{option}' was provided more than once"));
                    }
                } else {
                    return Err(format!("unknown option '--{option}'"));
                }
            } else {
                positionals.push(argument.clone());
            }
            index += 1;
        }
        Ok(Self {
            values,
            flags,
            positionals,
        })
    }

    fn value(&self, name: &str) -> Option<&str> {
        self.values.get(name).map(String::as_str)
    }

    fn flag(&self, name: &str) -> bool {
        self.flags.contains(name)
    }

    fn number<T>(&self, name: &str) -> Result<Option<T>, String>
    where
        T: FromStr,
        T::Err: std::fmt::Display,
    {
        self.value(name)
            .map(|value| {
                value
                    .parse::<T>()
                    .map_err(|error| format!("invalid --{name} value '{value}': {error}"))
            })
            .transpose()
    }

    fn one_positional(&self, label: &str) -> Result<&str, String> {
        match self.positionals.as_slice() {
            [value] => Ok(value),
            [] => Err(format!("{label} is required")),
            _ => Err(format!("expected one {label}")),
        }
    }

    fn require_no_positionals(&self) -> Result<(), String> {
        if self.positionals.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "unexpected argument '{}'",
                self.positionals.join(" ")
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Options;

    #[test]
    fn parses_mixed_options_and_positionals() {
        let arguments = [
            "core/text".to_string(),
            "--model".to_string(),
            "gemma".to_string(),
            "--iterations=2".to_string(),
        ];
        let options = Options::parse(&arguments, &["model", "iterations"], &[]).unwrap();
        assert_eq!(options.one_positional("benchmark").unwrap(), "core/text");
        assert_eq!(options.value("model"), Some("gemma"));
        assert_eq!(options.number::<u32>("iterations").unwrap(), Some(2));
    }
}
