use std::cmp::Ordering;
use std::sync::Arc;

use tokio::sync::watch;

use uuid::Uuid;

use crate::model::{
    BenchmarkCase, BenchmarkDefinition, HostInfo, InvocationResult, RunRecord, RunRequest,
    RunStatus, RunSummary, SampleRecord, Task,
};
use crate::packs::PackCatalog;
use crate::provider::ProviderClient;
use crate::store::{unix_time_ms, Store};
use crate::telemetry::TelemetrySampler;

pub async fn run_benchmark(
    store: &Store,
    catalog: &PackCatalog,
    request: RunRequest,
) -> Result<RunRecord, String> {
    execute_benchmark(store, prepare_benchmark(store, catalog, request)?).await
}

#[derive(Clone, Debug, Default)]
pub struct RunProgress {
    pub run_id: String,
    pub phase: String,
    pub case_id: Option<String>,
    pub completed: usize,
    pub expected: usize,
    pub warmups_completed: u32,
    pub warmups_expected: u32,
    pub summary: RunSummary,
}

#[derive(Clone)]
pub struct ExecutionControl {
    cancellation: watch::Sender<bool>,
    observer: Arc<dyn Fn(RunProgress) + Send + Sync>,
}

impl ExecutionControl {
    pub fn new(observer: impl Fn(RunProgress) + Send + Sync + 'static) -> Self {
        Self {
            cancellation: watch::channel(false).0,
            observer: Arc::new(observer),
        }
    }

    pub fn cancel(&self) {
        self.cancellation.send_replace(true);
    }

    fn report(&self, progress: &RunProgress) {
        (self.observer)(progress.clone());
    }
}

pub struct PreparedBenchmark {
    pub run: RunRecord,
    benchmark: BenchmarkDefinition,
    provider: ProviderClient,
    request: RunRequest,
}

pub fn prepare_benchmark(
    store: &Store,
    catalog: &PackCatalog,
    request: RunRequest,
) -> Result<PreparedBenchmark, String> {
    if request.iterations == 0 {
        return Err("iterations must be at least 1".into());
    }
    if request.iterations > 10_000 || request.warmups > 10_000 {
        return Err("iterations and warmups must not exceed 10000".into());
    }
    if request.timeout_seconds == 0 {
        return Err("timeout must be at least one second".into());
    }
    if request.model.trim().is_empty() {
        return Err("model is required".into());
    }
    let benchmark = catalog.benchmark(&request.benchmark_id)?;
    let mut max_output_tokens = request
        .max_output_tokens
        .unwrap_or(benchmark.default_max_output_tokens);
    if max_output_tokens == 0 {
        return Err("max output tokens must be at least 1".into());
    }
    if benchmark.task == Task::DecisionScoring {
        if request.max_output_tokens.is_some() {
            return Err("decision scoring does not accept generation token limits".into());
        }
        max_output_tokens = 0;
    }
    let host = HostInfo::detect();
    let run = RunRecord {
        id: Uuid::new_v4().to_string(),
        benchmark_id: benchmark.id.clone(),
        benchmark_name: benchmark.name.clone(),
        pack_version: benchmark.pack_version.clone(),
        task: benchmark.task,
        provider: request.provider,
        provider_metadata: Default::default(),
        base_url: request.base_url.trim_end_matches('/').to_string(),
        model: request.model.clone(),
        host_name: host.host_name.clone(),
        host,
        status: RunStatus::Running,
        started_at_ms: unix_time_ms(),
        finished_at_ms: None,
        iterations: request.iterations,
        warmups: request.warmups,
        max_output_tokens,
        target_pid: request.target_pid,
        error: None,
        summary: RunSummary::default(),
    };
    let provider = ProviderClient::new(
        request.provider,
        run.base_url.clone(),
        request.api_key.clone(),
        request.model.clone(),
        request.timeout_seconds,
    )?;
    store.start_run(&run)?;
    Ok(PreparedBenchmark {
        run,
        benchmark,
        provider,
        request,
    })
}

pub async fn execute_benchmark(
    store: &Store,
    prepared: PreparedBenchmark,
) -> Result<RunRecord, String> {
    execute_with_control(store, prepared, ExecutionControl::new(|_| {})).await
}

pub async fn execute_with_control(
    store: &Store,
    prepared: PreparedBenchmark,
    control: ExecutionControl,
) -> Result<RunRecord, String> {
    let run_id = prepared.run.id.clone();
    let mut cancelled = control.cancellation.subscribe();
    let outcome = tokio::select! {
        biased;
        _ = cancelled.wait_for(|value| *value) => Err("benchmark interrupted".to_string()),
        result = execute_observed(store, prepared, &control) => result,
    };
    let run = match outcome {
        Ok(run) => run,
        Err(error) => {
            let mut run = store
                .run(&run_id)?
                .ok_or_else(|| "run disappeared".to_string())?;
            run.summary = summarize(&store.samples(&run_id)?);
            finish_failed(store, &mut run, error)?;
            run
        }
    };
    control.report(&RunProgress {
        run_id: run.id.clone(),
        phase: run.status.as_str().into(),
        completed: run.summary.sample_count as usize,
        summary: run.summary.clone(),
        ..Default::default()
    });
    Ok(run)
}

async fn execute_observed(
    store: &Store,
    prepared: PreparedBenchmark,
    control: &ExecutionControl,
) -> Result<RunRecord, String> {
    let PreparedBenchmark {
        mut run,
        benchmark,
        provider,
        request,
    } = prepared;
    let mut progress = RunProgress {
        run_id: run.id.clone(),
        phase: "preflight".into(),
        expected: benchmark.cases.len() * request.iterations as usize,
        warmups_expected: request.warmups,
        ..Default::default()
    };
    control.report(&progress);
    let max_output_tokens = run.max_output_tokens;
    match provider.metadata().await {
        Ok(metadata) => {
            store.set_provider_metadata(&run.id, &metadata)?;
            run.provider_metadata = metadata;
        }
        Err(error) => {
            finish_failed(store, &mut run, format!("preflight failed: {error}"))?;
            return Ok(run);
        }
    }
    if let Err(error) = provider
        .preflight(benchmark.task, &run.provider_metadata)
        .await
    {
        finish_failed(store, &mut run, format!("preflight failed: {error}"))?;
        return Ok(run);
    }
    for warmup in 0..request.warmups {
        progress.phase = "warmup".into();
        progress.case_id = benchmark.cases.first().map(|case| case.id.clone());
        progress.warmups_completed = warmup;
        control.report(&progress);
        if let Err(error) = warm_up(&provider, &benchmark, 1, max_output_tokens).await {
            finish_failed(store, &mut run, format!("warmup failed: {error}"))?;
            return Ok(run);
        }
        progress.warmups_completed = warmup + 1;
        control.report(&progress);
    }
    let mut samples = Vec::new();
    for iteration in 1..=request.iterations {
        for case in &benchmark.cases {
            progress.phase = "measuring".into();
            progress.case_id = Some(case.id.clone());
            control.report(&progress);
            let sampler = TelemetrySampler::start(request.target_pid);
            let result = provider
                .invoke(benchmark.task, case, max_output_tokens)
                .await;
            let telemetry = sampler.finish().await;
            let mut sample = SampleRecord {
                id: Uuid::new_v4().to_string(),
                run_id: run.id.clone(),
                case_id: case.id.clone(),
                iteration,
                succeeded: false,
                error: None,
                latency_ms: None,
                ttft_ms: None,
                input_tokens: None,
                output_tokens: None,
                tokens_per_second: None,
                host_cpu_percent_mean: telemetry.host_cpu_percent_mean,
                host_cpu_percent_peak: telemetry.host_cpu_percent_peak,
                host_memory_used_bytes_peak: telemetry.host_memory_used_bytes_peak,
                process_rss_bytes_peak: telemetry.process_rss_bytes_peak,
                quality_score: None,
                output_excerpt: None,
                output_image: None,
                decision: None,
                speech: None,
            };
            match result {
                Ok(invocation) => {
                    match measured_sample(benchmark.task, case, invocation, &mut sample) {
                        Ok(()) => sample.succeeded = true,
                        Err(error) => sample.error = Some(error),
                    }
                }
                Err(error) => sample.error = Some(error),
            }
            store.insert_sample(&sample)?;
            samples.push(sample);
            progress.phase = "saved".into();
            progress.completed = samples.len();
            progress.summary = summarize(&samples);
            control.report(&progress);
        }
    }
    run.summary = summarize(&samples);
    run.finished_at_ms = Some(unix_time_ms());
    let failures = samples
        .iter()
        .filter(|sample| !sample.succeeded)
        .collect::<Vec<_>>();
    if failures.is_empty() {
        run.status = RunStatus::Succeeded;
    } else {
        run.status = RunStatus::Failed;
        run.error = Some(failure_summary(&failures));
    }
    store.finish_run(
        &run.id,
        run.status,
        run.finished_at_ms.unwrap_or_else(unix_time_ms),
        run.error.as_deref(),
        &run.summary,
    )?;
    Ok(run)
}

async fn warm_up(
    provider: &ProviderClient,
    benchmark: &BenchmarkDefinition,
    warmups: u32,
    max_output_tokens: u32,
) -> Result<(), String> {
    let case = benchmark
        .cases
        .first()
        .ok_or_else(|| "benchmark has no cases".to_string())?;
    for _ in 0..warmups {
        provider
            .invoke(benchmark.task, case, max_output_tokens)
            .await?;
    }
    Ok(())
}

fn measured_sample(
    task: Task,
    case: &BenchmarkCase,
    invocation: InvocationResult,
    sample: &mut SampleRecord,
) -> Result<(), String> {
    sample.latency_ms = Some(invocation.latency_ms);
    sample.ttft_ms = invocation.ttft_ms;
    sample.input_tokens = invocation.input_tokens;
    sample.output_tokens = invocation.output_tokens;
    sample.tokens_per_second = invocation.tokens_per_second;
    sample.quality_score = quality_score(task, case, &invocation)?;
    sample.output_excerpt = output_excerpt(task, &invocation)?;
    sample.decision = if let Some(result) = invocation.decision {
        let case = case
            .decision
            .as_ref()
            .ok_or_else(|| "decision input is required".to_string())?;
        Some(crate::decision::DecisionObservation {
            case: case.clone(),
            media: invocation.decision_media,
            metrics: result.score(case),
            result,
        })
    } else {
        None
    };
    if task == Task::SpeechToText {
        sample.speech = Some(crate::model::SpeechObservation {
            transcript: invocation
                .output_text
                .ok_or_else(|| "provider returned no transcript".to_string())?,
            audio_duration_seconds: invocation.audio_duration_seconds,
            routing: invocation.speech_routing,
        });
    }
    sample.output_image = invocation.output_image;
    Ok(())
}

fn quality_score(
    task: Task,
    case: &BenchmarkCase,
    invocation: &InvocationResult,
) -> Result<Option<f64>, String> {
    match task {
        Task::DecisionScoring => {
            let result = invocation
                .decision
                .as_ref()
                .ok_or_else(|| "provider returned no decision".to_string())?;
            Ok(case
                .decision
                .as_ref()
                .and_then(|case| case.expected.as_ref())
                .map(|expected| {
                    if result.selected_id == *expected {
                        1.0
                    } else {
                        0.0
                    }
                }))
        }
        Task::TextGeneration | Task::ImageToText => {
            if case.expected_contains.is_empty() {
                return Ok(None);
            }
            let output = invocation
                .output_text
                .as_deref()
                .ok_or_else(|| "provider returned no text".to_string())?
                .to_ascii_lowercase();
            let matches = case
                .expected_contains
                .iter()
                .filter(|expected| output.contains(&expected.to_ascii_lowercase()))
                .count();
            Ok(Some(matches as f64 / case.expected_contains.len() as f64))
        }
        Task::SpeechToText => match &case.expected_text {
            Some(expected) => {
                let output = invocation
                    .output_text
                    .as_deref()
                    .ok_or_else(|| "provider returned no transcript".to_string())?;
                Ok(Some(word_accuracy(expected, output)))
            }
            None => Ok(None),
        },
        Task::Embedding => {
            let expected_count = case.documents.len() + 1;
            if invocation.embeddings.len() != expected_count {
                return Err(format!(
                    "provider returned {} embeddings for {expected_count} inputs",
                    invocation.embeddings.len()
                ));
            }
            let query = &invocation.embeddings[0];
            if query.is_empty() {
                return Err("provider returned an empty embedding".into());
            }
            let mut ranked = invocation.embeddings[1..]
                .iter()
                .enumerate()
                .map(|(index, vector)| cosine_similarity(query, vector).map(|score| (index, score)))
                .collect::<Result<Vec<_>, _>>()?;
            ranked.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal));
            Ok(case.expected_index.map(|expected| {
                ranked
                    .iter()
                    .position(|(index, _)| *index == expected)
                    .map(|position| 1.0 / (position as f64 + 1.0))
                    .unwrap_or(0.0)
            }))
        }
        Task::Reranking => Ok(case.expected_index.map(|expected| {
            invocation
                .ranking
                .iter()
                .position(|index| *index == expected)
                .map(|position| 1.0 / (position as f64 + 1.0))
                .unwrap_or(0.0)
        })),
        Task::TextToImage => {
            let image = invocation
                .output_image
                .as_deref()
                .ok_or_else(|| "provider returned no generated image".to_string())?;
            crate::provider::validate_generated_image(image, case.image_settings.as_ref())?;
            Ok(None)
        }
    }
}

fn output_excerpt(task: Task, invocation: &InvocationResult) -> Result<Option<String>, String> {
    match task {
        Task::Embedding => {
            let query = invocation
                .embeddings
                .first()
                .ok_or_else(|| "provider returned no embeddings".to_string())?;
            Ok(Some(format!(
                "{} vectors, {} dimensions",
                invocation.embeddings.len(),
                query.len()
            )))
        }
        Task::Reranking => Ok(Some(format!("ranking: {:?}", invocation.ranking))),
        Task::DecisionScoring => Ok(invocation
            .decision
            .as_ref()
            .map(|result| format!("selected: {}", result.selected_id))),
        _ => Ok(invocation.output_text.as_deref().map(excerpt)),
    }
}

fn summarize(samples: &[SampleRecord]) -> RunSummary {
    let successful = samples
        .iter()
        .filter(|sample| sample.succeeded)
        .collect::<Vec<_>>();
    RunSummary {
        sample_count: samples.len() as u32,
        successful_sample_count: successful.len() as u32,
        success_rate: if samples.is_empty() {
            0.0
        } else {
            successful.len() as f64 / samples.len() as f64
        },
        latency_ms_p50: percentile(
            successful
                .iter()
                .filter_map(|sample| sample.latency_ms)
                .collect(),
            50.0,
        ),
        latency_ms_p95: percentile(
            successful
                .iter()
                .filter_map(|sample| sample.latency_ms)
                .collect(),
            95.0,
        ),
        ttft_ms_p50: percentile(
            successful
                .iter()
                .filter_map(|sample| sample.ttft_ms)
                .collect(),
            50.0,
        ),
        ttft_ms_p95: percentile(
            successful
                .iter()
                .filter_map(|sample| sample.ttft_ms)
                .collect(),
            95.0,
        ),
        tokens_per_second_p50: percentile(
            successful
                .iter()
                .filter_map(|sample| sample.tokens_per_second)
                .collect(),
            50.0,
        ),
        host_cpu_percent_mean: mean(
            successful
                .iter()
                .filter_map(|sample| sample.host_cpu_percent_mean),
        ),
        host_cpu_percent_peak: max_f64(
            successful
                .iter()
                .filter_map(|sample| sample.host_cpu_percent_peak),
        ),
        host_memory_used_bytes_peak: successful
            .iter()
            .filter_map(|sample| sample.host_memory_used_bytes_peak)
            .max(),
        process_rss_bytes_peak: successful
            .iter()
            .filter_map(|sample| sample.process_rss_bytes_peak)
            .max(),
        quality_score_mean: mean(successful.iter().filter_map(|sample| sample.quality_score)),
        decision_metrics: crate::decision::summarize(
            successful
                .iter()
                .filter_map(|sample| sample.decision.as_ref()?.metrics.as_ref()),
        ),
    }
}

fn finish_failed(store: &Store, run: &mut RunRecord, error: String) -> Result<(), String> {
    run.status = RunStatus::Failed;
    run.error = Some(error);
    run.finished_at_ms = Some(unix_time_ms());
    store.finish_run(
        &run.id,
        run.status,
        run.finished_at_ms.unwrap_or_else(unix_time_ms),
        run.error.as_deref(),
        &run.summary,
    )
}

fn failure_summary(samples: &[&SampleRecord]) -> String {
    let mut details = samples
        .iter()
        .take(3)
        .map(|sample| {
            format!(
                "{} iteration {}: {}",
                sample.case_id,
                sample.iteration,
                sample.error.as_deref().unwrap_or("unknown failure")
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    if samples.len() > 3 {
        details.push_str(&format!("; and {} more", samples.len() - 3));
    }
    details
}

fn percentile(mut values: Vec<f64>, percentile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    let rank = ((percentile / 100.0) * values.len() as f64).ceil() as usize;
    values.get(rank.saturating_sub(1)).copied()
}

fn mean(values: impl Iterator<Item = f64>) -> Option<f64> {
    let values = values.collect::<Vec<_>>();
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn max_f64(values: impl Iterator<Item = f64>) -> Option<f64> {
    values.reduce(f64::max)
}

fn cosine_similarity(left: &[f64], right: &[f64]) -> Result<f64, String> {
    if left.len() != right.len() || left.is_empty() {
        return Err("embedding dimensions do not match".into());
    }
    let dot = left
        .iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum::<f64>();
    let left_norm = left.iter().map(|value| value * value).sum::<f64>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f64>().sqrt();
    if left_norm == 0.0 || right_norm == 0.0 {
        return Err("embedding vector has zero magnitude".into());
    }
    Ok(dot / (left_norm * right_norm))
}

fn word_accuracy(expected: &str, actual: &str) -> f64 {
    let expected = words(expected);
    let actual = words(actual);
    if expected.is_empty() {
        return if actual.is_empty() { 1.0 } else { 0.0 };
    }
    let mut previous = (0..=actual.len()).collect::<Vec<_>>();
    for (left_index, left) in expected.iter().enumerate() {
        let mut current = vec![left_index + 1];
        for (right_index, right) in actual.iter().enumerate() {
            let substitution = previous[right_index] + usize::from(left != right);
            let insertion = current[right_index] + 1;
            let deletion = previous[right_index + 1] + 1;
            current.push(substitution.min(insertion).min(deletion));
        }
        previous = current;
    }
    let distance = *previous.last().unwrap_or(&expected.len());
    (1.0 - distance as f64 / expected.len() as f64).max(0.0)
}

fn words(value: &str) -> Vec<String> {
    value
        .split_whitespace()
        .map(|word| {
            word.chars()
                .filter(|character| character.is_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect::<String>()
        })
        .filter(|word| !word.is_empty())
        .collect()
}

fn excerpt(value: &str) -> String {
    value.chars().take(500).collect()
}

#[cfg(test)]
mod tests {
    use super::{cosine_similarity, percentile, word_accuracy};

    #[test]
    fn metrics_are_deterministic() {
        assert_eq!(percentile(vec![4.0, 1.0, 3.0, 2.0], 50.0), Some(2.0));
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]), Ok(1.0));
        assert_eq!(word_accuracy("hello pi", "hello pie"), 0.5);
    }
}
