use std::fmt::{Display, Formatter};
use std::str::FromStr;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct HostInfo {
    pub host_name: String,
    pub operating_system: String,
    pub kernel: Option<String>,
    pub architecture: String,
    pub device_model: Option<String>,
    pub cpu: Option<String>,
    pub logical_cpu_count: Option<u32>,
    pub total_memory_bytes: Option<u64>,
    pub raspberry_pi: bool,
    pub qit_version: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Task {
    TextGeneration,
    Embedding,
    ImageToText,
    SpeechToText,
    Reranking,
    TextToImage,
}

impl Task {
    pub const ALL: [Self; 6] = [
        Self::TextGeneration,
        Self::Embedding,
        Self::ImageToText,
        Self::SpeechToText,
        Self::Reranking,
        Self::TextToImage,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::TextGeneration => "text_generation",
            Self::Embedding => "embedding",
            Self::ImageToText => "image_to_text",
            Self::SpeechToText => "speech_to_text",
            Self::Reranking => "reranking",
            Self::TextToImage => "text_to_image",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::TextGeneration => "Text to text",
            Self::Embedding => "Text embedding",
            Self::ImageToText => "Image to text",
            Self::SpeechToText => "Speech to text",
            Self::Reranking => "Reranking",
            Self::TextToImage => "Text to image",
        }
    }
}

impl Display for Task {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    Ollama,
    Transformers,
    Tei,
}

impl ProviderKind {
    pub const ALL: [Self; 3] = [Self::Ollama, Self::Transformers, Self::Tei];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ollama => "ollama",
            Self::Transformers => "transformers",
            Self::Tei => "tei",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Ollama => "Ollama",
            Self::Transformers => "Transformers Serve",
            Self::Tei => "Hugging Face TEI",
        }
    }

    pub fn default_base_url(self) -> &'static str {
        match self {
            Self::Ollama => "http://127.0.0.1:11434",
            Self::Transformers => "http://127.0.0.1:8000",
            Self::Tei => "http://127.0.0.1:8080",
        }
    }

    pub fn supports(self, task: Task) -> bool {
        match self {
            Self::Ollama => matches!(
                task,
                Task::TextGeneration | Task::Embedding | Task::ImageToText
            ),
            Self::Transformers => matches!(
                task,
                Task::TextGeneration | Task::ImageToText | Task::SpeechToText
            ),
            Self::Tei => matches!(task, Task::Embedding | Task::Reranking),
        }
    }
}

impl Display for ProviderKind {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for ProviderKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "ollama" => Ok(Self::Ollama),
            "transformers" | "transformers-serve" | "openai" | "openai-compatible" => {
                Ok(Self::Transformers)
            }
            "tei" | "huggingface-tei" => Ok(Self::Tei),
            _ => Err(format!(
                "unknown provider '{value}', expected ollama, transformers, or tei"
            )),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BenchmarkPackManifest {
    pub schema_version: u32,
    pub id: String,
    pub name: String,
    pub version: String,
    pub description: String,
    pub publisher: String,
    pub license: String,
    pub source: String,
    pub benchmarks: Vec<BenchmarkManifest>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BenchmarkManifest {
    pub id: String,
    pub name: String,
    pub description: String,
    pub task: Task,
    pub cases: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default = "default_max_output_tokens")]
    pub default_max_output_tokens: u32,
}

fn default_max_output_tokens() -> u32 {
    128
}

#[derive(Clone, Debug, Serialize)]
pub struct BenchmarkDefinition {
    pub id: String,
    pub pack_id: String,
    pub pack_name: String,
    pub pack_version: String,
    pub name: String,
    pub description: String,
    pub task: Task,
    pub case_count: usize,
    pub tags: Vec<String>,
    pub default_max_output_tokens: u32,
    pub built_in: bool,
    pub supported_providers: Vec<ProviderKind>,
    #[serde(skip)]
    pub cases: Vec<BenchmarkCase>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BenchmarkCase {
    pub id: String,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub input: Option<String>,
    #[serde(default)]
    pub media: Option<String>,
    #[serde(default)]
    pub media_type: Option<String>,
    #[serde(default)]
    pub documents: Vec<String>,
    #[serde(default)]
    pub expected_index: Option<usize>,
    #[serde(default)]
    pub expected_contains: Vec<String>,
    #[serde(default)]
    pub expected_text: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Running,
    Succeeded,
    Failed,
}

impl RunStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
        }
    }

    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "running" => Ok(Self::Running),
            "succeeded" => Ok(Self::Succeeded),
            "failed" => Ok(Self::Failed),
            _ => Err(format!("invalid run status '{value}'")),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct RunSummary {
    pub sample_count: u32,
    pub successful_sample_count: u32,
    pub success_rate: f64,
    pub latency_ms_p50: Option<f64>,
    pub latency_ms_p95: Option<f64>,
    pub ttft_ms_p50: Option<f64>,
    pub ttft_ms_p95: Option<f64>,
    pub tokens_per_second_p50: Option<f64>,
    pub host_cpu_percent_mean: Option<f64>,
    pub host_cpu_percent_peak: Option<f64>,
    pub host_memory_used_bytes_peak: Option<u64>,
    pub process_rss_bytes_peak: Option<u64>,
    pub quality_score_mean: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RunRecord {
    pub id: String,
    pub benchmark_id: String,
    pub benchmark_name: String,
    pub pack_version: String,
    pub task: Task,
    pub provider: ProviderKind,
    pub base_url: String,
    pub model: String,
    pub host_name: String,
    pub host: HostInfo,
    pub status: RunStatus,
    pub started_at_ms: i64,
    pub finished_at_ms: Option<i64>,
    pub iterations: u32,
    pub warmups: u32,
    pub max_output_tokens: u32,
    pub target_pid: Option<u32>,
    pub error: Option<String>,
    pub summary: RunSummary,
}

#[derive(Clone, Debug, Serialize)]
pub struct SampleRecord {
    pub id: String,
    pub run_id: String,
    pub case_id: String,
    pub iteration: u32,
    pub succeeded: bool,
    pub error: Option<String>,
    pub latency_ms: Option<f64>,
    pub ttft_ms: Option<f64>,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub tokens_per_second: Option<f64>,
    pub host_cpu_percent_mean: Option<f64>,
    pub host_cpu_percent_peak: Option<f64>,
    pub host_memory_used_bytes_peak: Option<u64>,
    pub process_rss_bytes_peak: Option<u64>,
    pub quality_score: Option<f64>,
    pub output_excerpt: Option<String>,
}

#[derive(Clone, Debug)]
pub struct RunRequest {
    pub benchmark_id: String,
    pub provider: ProviderKind,
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    pub iterations: u32,
    pub warmups: u32,
    pub max_output_tokens: Option<u32>,
    pub timeout_seconds: u64,
    pub target_pid: Option<u32>,
}

#[derive(Clone, Debug, Default)]
pub struct InvocationResult {
    pub output_text: Option<String>,
    pub embeddings: Vec<Vec<f64>>,
    pub ranking: Vec<usize>,
    pub latency_ms: f64,
    pub ttft_ms: Option<f64>,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub tokens_per_second: Option<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct TelemetrySummary {
    pub host_cpu_percent_mean: Option<f64>,
    pub host_cpu_percent_peak: Option<f64>,
    pub host_memory_used_bytes_peak: Option<u64>,
    pub process_rss_bytes_peak: Option<u64>,
}
