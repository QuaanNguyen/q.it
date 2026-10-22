use std::collections::{BTreeMap, HashSet};

use serde::Serialize;
use serde_json::json;

use crate::model::{ProviderKind, ProviderMetadata, RunRecord, RunStatus};

#[derive(Clone, Serialize)]
pub struct AnalysisPoint {
    pub run_id: String,
    pub benchmark_id: String,
    pub pack_version: String,
    pub model: String,
    pub provider: ProviderKind,
    pub host_name: String,
    pub provider_metadata: ProviderMetadata,
    pub latency_ms: Option<f64>,
    pub ttft_ms: Option<f64>,
    pub tokens_per_second: Option<f64>,
    pub memory_bytes: Option<u64>,
    pub cpu_percent: Option<f64>,
    pub quality_score: Option<f64>,
    pub time_to_first_audio_ms: Option<f64>,
    pub real_time_factor: Option<f64>,
    pub pareto: bool,
}

#[derive(Clone, Serialize)]
pub struct ComparisonGroup {
    pub key: String,
    pub benchmark_key: String,
    pub runs: Vec<RunRecord>,
    pub models: Vec<RunRecord>,
    pub rankings: BTreeMap<String, Vec<Ranking>>,
}

#[derive(Clone, Serialize)]
pub struct Ranking {
    pub run_id: String,
    pub rank: usize,
    pub value: f64,
}

#[derive(Serialize)]
pub struct Analysis {
    pub points: Vec<AnalysisPoint>,
    pub groups: Vec<ComparisonGroup>,
}

#[derive(Clone, Copy)]
pub enum Metric {
    Latency,
    Ttft,
    Throughput,
    Quality,
    Cpu,
    Memory,
    FirstAudio,
    RealTimeFactor,
}

impl Metric {
    pub const ALL: [Self; 8] = [
        Self::Latency,
        Self::Ttft,
        Self::Throughput,
        Self::Quality,
        Self::Cpu,
        Self::Memory,
        Self::FirstAudio,
        Self::RealTimeFactor,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Latency => "Latency ms",
            Self::Ttft => "TTFT ms",
            Self::Throughput => "Tokens/s",
            Self::Quality => "Quality",
            Self::Cpu => "CPU %",
            Self::Memory => "Provider RSS MiB",
            Self::FirstAudio => "First audio ms",
            Self::RealTimeFactor => "RTF",
        }
    }

    pub fn value(self, run: &RunRecord) -> Option<f64> {
        match self {
            Self::Latency => run.summary.latency_ms_p50,
            Self::Ttft => run.summary.ttft_ms_p50,
            Self::Throughput => run.summary.tokens_per_second_p50,
            Self::Quality => run.summary.quality_score_mean,
            Self::Cpu => run.summary.host_cpu_percent_mean,
            Self::Memory => run
                .summary
                .process_rss_bytes_peak
                .map(|value| value as f64 / 1_048_576.0),
            Self::FirstAudio => run.summary.time_to_first_audio_ms_p50,
            Self::RealTimeFactor => run.summary.real_time_factor_p50,
        }
        .filter(|value| value.is_finite())
    }

    pub fn higher_is_better(self) -> bool {
        matches!(self, Self::Throughput | Self::Quality)
    }
}

pub fn setup_key(run: &RunRecord) -> String {
    let configuration = &run.provider_metadata.configuration;
    json!([
        run.host_name,
        run.host.operating_system,
        run.host.kernel,
        run.host.architecture,
        run.host.cpu,
        run.host.logical_cpu_count,
        run.host.total_memory_bytes,
        run.host.qit_version,
        run.iterations,
        run.warmups,
        run.max_output_tokens,
        run.provider_metadata.version,
        [
            configuration.get("device"),
            configuration.get("dtype"),
            configuration.get("threads"),
            configuration.get("packages")
        ]
    ])
    .to_string()
}

pub fn model_key(run: &RunRecord) -> String {
    let mut configuration = run.provider_metadata.configuration.clone();
    if let Some(object) = configuration.as_object_mut() {
        object.remove("pid");
    }
    json!([run.model, run.provider, run.base_url, configuration]).to_string()
}

pub fn rank(models: &[RunRecord], metric: Metric) -> Vec<Ranking> {
    let mut values = models
        .iter()
        .filter(|run| run.status == RunStatus::Succeeded)
        .filter_map(|run| Some((run, metric.value(run)?)))
        .collect::<Vec<_>>();
    values.sort_by(|(left, a), (right, b)| {
        let order = a.total_cmp(b);
        (if metric.higher_is_better() {
            order.reverse()
        } else {
            order
        })
        .then_with(|| left.model.cmp(&right.model))
        .then_with(|| left.id.cmp(&right.id))
    });
    let mut rank = 1;
    values
        .iter()
        .enumerate()
        .map(|(index, (run, value))| {
            if index > 0 && *value != values[index - 1].1 {
                rank = index + 1;
            }
            Ranking {
                run_id: run.id.clone(),
                rank,
                value: *value,
            }
        })
        .collect()
}

pub fn analyze(runs: Vec<RunRecord>) -> Analysis {
    let mut grouped: BTreeMap<String, ComparisonGroup> = BTreeMap::new();
    for run in runs
        .into_iter()
        .filter(|run| run.status == RunStatus::Succeeded)
    {
        let benchmark_key = json!([run.benchmark_id, run.pack_version]).to_string();
        let key = json!([benchmark_key, setup_key(&run)]).to_string();
        grouped
            .entry(key.clone())
            .or_insert_with(|| ComparisonGroup {
                key,
                benchmark_key,
                runs: Vec::new(),
                models: Vec::new(),
                rankings: BTreeMap::new(),
            })
            .runs
            .push(run);
    }
    let mut groups = grouped.into_values().collect::<Vec<_>>();
    for group in &mut groups {
        group.runs.sort_by(|left, right| {
            right
                .started_at_ms
                .cmp(&left.started_at_ms)
                .then_with(|| left.id.cmp(&right.id))
        });
        let mut identities = HashSet::new();
        group.models = group
            .runs
            .iter()
            .filter(|run| identities.insert(model_key(run)))
            .cloned()
            .collect();
        group.models.sort_by(|left, right| {
            left.model
                .cmp(&right.model)
                .then_with(|| left.id.cmp(&right.id))
        });
        group.rankings = Metric::ALL
            .into_iter()
            .map(|metric| (metric.name().into(), rank(&group.models, metric)))
            .collect();
    }
    groups.sort_by(|left, right| {
        right
            .models
            .len()
            .cmp(&left.models.len())
            .then_with(|| right.runs[0].started_at_ms.cmp(&left.runs[0].started_at_ms))
            .then_with(|| left.key.cmp(&right.key))
    });
    let points = groups
        .iter()
        .flat_map(|group| {
            group.models.iter().map(|run| {
                let latency = run.summary.latency_ms_p50;
                let memory = run.summary.process_rss_bytes_peak;
                let pareto = latency.zip(memory).is_some_and(|(latency, memory)| {
                    !group.models.iter().any(|other| {
                        other
                            .summary
                            .latency_ms_p50
                            .zip(other.summary.process_rss_bytes_peak)
                            .is_some_and(|(a, b)| {
                                a <= latency && b <= memory && (a < latency || b < memory)
                            })
                    })
                });
                AnalysisPoint {
                    run_id: run.id.clone(),
                    benchmark_id: run.benchmark_id.clone(),
                    pack_version: run.pack_version.clone(),
                    model: run.model.clone(),
                    provider: run.provider,
                    host_name: run.host_name.clone(),
                    provider_metadata: run.provider_metadata.clone(),
                    latency_ms: latency,
                    ttft_ms: run.summary.ttft_ms_p50,
                    tokens_per_second: run.summary.tokens_per_second_p50,
                    memory_bytes: memory,
                    cpu_percent: run.summary.host_cpu_percent_mean,
                    quality_score: run.summary.quality_score_mean,
                    time_to_first_audio_ms: run.summary.time_to_first_audio_ms_p50,
                    real_time_factor: run.summary.real_time_factor_p50,
                    pareto,
                }
            })
        })
        .collect();
    Analysis { points, groups }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{HostInfo, RunSummary, Task};

    fn run(id: &str, model: &str, latency: Option<f64>) -> RunRecord {
        RunRecord {
            id: id.into(),
            benchmark_id: "fixture/text".into(),
            benchmark_name: "Fixture".into(),
            pack_version: "1.0.0".into(),
            task: Task::TextGeneration,
            provider: ProviderKind::LocalCpu,
            provider_metadata: ProviderMetadata {
                version: Some("1".into()),
                configuration: json!({"device":"cpu", "dtype":"float32", "threads":4, "revision":model, "pid":123}),
            },
            base_url: "http://127.0.0.1:8099".into(),
            model: model.into(),
            host_name: "fixture".into(),
            host: HostInfo {
                host_name: "fixture".into(),
                operating_system: "linux".into(),
                kernel: None,
                architecture: "aarch64".into(),
                device_model: None,
                cpu: None,
                logical_cpu_count: Some(4),
                total_memory_bytes: Some(8_000_000_000),
                raspberry_pi: true,
                qit_version: "0.1.0".into(),
            },
            status: RunStatus::Succeeded,
            started_at_ms: 100,
            finished_at_ms: Some(200),
            iterations: 3,
            warmups: 1,
            max_output_tokens: 64,
            target_pid: None,
            error: None,
            summary: RunSummary {
                latency_ms_p50: latency,
                ..Default::default()
            },
        }
    }

    #[test]
    fn comparison_separates_setups_preserves_routes_and_ranks_ties_without_missing_values() {
        let first = run("first", "first-model", Some(20.0));
        let mut older = first.clone();
        older.id = "old".into();
        older.started_at_ms = 1;
        older.summary.latency_ms_p50 = Some(1.0);
        let second = run("second", "second-model", Some(20.0));
        let missing = run("missing", "third-model", None);
        let mut failed = run("failed", "failed-model", Some(0.0));
        failed.status = RunStatus::Failed;
        let analysis = analyze(vec![first.clone(), older, second, missing, failed]);
        assert_eq!(analysis.groups.len(), 1);
        let group = &analysis.groups[0];
        assert_eq!(group.models.len(), 3);
        let ranking = rank(&group.models, Metric::Latency);
        assert_eq!(ranking.len(), 2);
        assert!(ranking
            .iter()
            .all(|entry| entry.rank == 1 && entry.value == 20.0));
        let mut variants = Vec::new();
        for index in 0..8 {
            let mut variant = first.clone();
            variant.id = index.to_string();
            match index {
                0 => variant.pack_version = "2".into(),
                1 => variant.host.cpu = Some("other".into()),
                2 => variant.host.kernel = Some("other".into()),
                3 => variant.iterations += 1,
                4 => variant.warmups += 1,
                5 => variant.max_output_tokens += 1,
                6 => variant.provider_metadata.configuration["dtype"] = json!("ternary"),
                _ => variant.provider_metadata.configuration["threads"] = json!(8),
            }
            variants.push(variant);
        }
        variants.push(first.clone());
        assert_eq!(analyze(variants).groups.len(), 9);
        let mut restarted = first.clone();
        restarted.provider_metadata.configuration["pid"] = json!(999);
        assert_eq!(model_key(&first), model_key(&restarted));
        restarted.provider_metadata.configuration["route"] = json!("automatic");
        assert_ne!(model_key(&first), model_key(&restarted));
    }
}
