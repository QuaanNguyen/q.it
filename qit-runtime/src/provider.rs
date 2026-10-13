use std::path::Path;
use std::time::{Duration, Instant};

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use futures_util::StreamExt;
use reqwest::{RequestBuilder, Response};
use serde_json::{json, Value};

use crate::model::{BenchmarkCase, InvocationResult, ProviderKind, ProviderMetadata, Task};

const HF_SERVE_MAX_BATCH_INPUTS: usize = 4;

#[derive(Clone)]
pub struct ProviderClient {
    kind: ProviderKind,
    base_url: String,
    api_key: Option<String>,
    model: String,
    client: reqwest::Client,
}

impl ProviderClient {
    pub fn new(
        kind: ProviderKind,
        base_url: String,
        api_key: Option<String>,
        model: String,
        timeout_seconds: u64,
    ) -> Result<Self, String> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(timeout_seconds))
            .build()
            .map_err(|error| format!("build HTTP client: {error}"))?;
        Ok(Self {
            kind,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            model,
            client,
        })
    }

    pub async fn invoke(
        &self,
        task: Task,
        case: &BenchmarkCase,
        max_output_tokens: u32,
    ) -> Result<InvocationResult, String> {
        if !self.kind.supports(task) {
            return Err(format!(
                "{} does not support {} benchmarks",
                self.kind.label(),
                task.label()
            ));
        }
        match (self.kind, task) {
            (ProviderKind::Ollama, Task::TextGeneration | Task::ImageToText) => {
                self.ollama_generate(case, max_output_tokens).await
            }
            (ProviderKind::Ollama, Task::Embedding) => self.ollama_embed(case).await,
            (ProviderKind::Transformers, Task::TextGeneration | Task::ImageToText) => {
                self.transformers_generate(case, max_output_tokens).await
            }
            (ProviderKind::Transformers, Task::SpeechToText) => {
                self.transformers_transcribe(case).await
            }
            (ProviderKind::Tei, Task::Embedding) => self.tei_embed(case).await,
            (ProviderKind::HfServe, Task::Embedding) => self.hf_serve_embed(case).await,
            (ProviderKind::Tei, Task::Reranking) => self.tei_rerank(case).await,
            (ProviderKind::LocalCpu, task) => {
                self.local_cpu_invoke(task, case, max_output_tokens).await
            }
            _ => Err(format!(
                "{} does not support {} benchmarks",
                self.kind.label(),
                task.label()
            )),
        }
    }

    pub async fn metadata(&self) -> Result<ProviderMetadata, String> {
        if self.kind == ProviderKind::LocalCpu {
            return self.local_cpu_metadata().await;
        }
        if self.kind != ProviderKind::Ollama {
            return Ok(ProviderMetadata::default());
        }
        let response = self
            .authorized(self.client.get(self.url("/api/version")))
            .send()
            .await
            .map_err(|error| format!("Ollama version preflight failed: {error}"))?;
        let value = response_json(response, "Ollama version preflight").await?;
        let version = value
            .get("version")
            .and_then(Value::as_str)
            .filter(|version| !version.trim().is_empty())
            .ok_or_else(|| "Ollama version response has no non-empty version string".to_string())?;
        Ok(ProviderMetadata {
            version: Some(version.to_string()),
            configuration: Value::Null,
        })
    }

    pub async fn preflight(&self, task: Task, metadata: &ProviderMetadata) -> Result<(), String> {
        if !self.kind.supports(task) {
            return Err(format!(
                "{} does not support {} benchmarks",
                self.kind.label(),
                task.label()
            ));
        }
        match self.kind {
            ProviderKind::Ollama => {
                if task == Task::Embedding {
                    let version = metadata.version.as_deref().unwrap_or("unknown");
                    let parsed = semver::Version::parse(version).map_err(|_| {
                        format!("embedding benchmarks require Ollama 0.40.3 or later; cannot verify server version '{version}'")
                    })?;
                    if parsed < semver::Version::new(0, 40, 3) {
                        return Err(format!("embedding benchmarks require Ollama 0.40.3 or later; server reports {version}. Upgrade the Ollama server before retrying"));
                    }
                }
                self.ollama_preflight(task).await
            }
            ProviderKind::Transformers => self.transformers_preflight().await,
            ProviderKind::HfServe => self.hf_serve_preflight().await,
            ProviderKind::Tei => self.tei_preflight().await,
            ProviderKind::LocalCpu => {
                if metadata.configuration.get("task").and_then(Value::as_str) != Some(task.as_str())
                {
                    return Err(format!(
                        "the loaded Local CPU model does not support {}",
                        task.label()
                    ));
                }
                Ok(())
            }
        }
    }

    async fn local_cpu_metadata(&self) -> Result<ProviderMetadata, String> {
        let response = self
            .authorized(self.client.get(self.url("/v1/qit/model")))
            .send()
            .await
            .map_err(|error| format!("Local CPU preflight failed: {error}"))?;
        let value = response_json(response, "Local CPU preflight").await?;
        if value.get("schema_version").and_then(Value::as_u64) != Some(1)
            || value.get("model").and_then(Value::as_str) != Some(self.model.as_str())
        {
            return Err(
                "Local CPU endpoint does not report the requested model and protocol version"
                    .into(),
            );
        }
        let configuration = value
            .get("configuration")
            .filter(|configuration| configuration.is_object())
            .ok_or_else(|| "Local CPU endpoint has no model configuration".to_string())?;
        if configuration.get("device").and_then(Value::as_str) != Some("cpu") {
            return Err("Local CPU endpoint must report CPU-only execution".into());
        }
        if !configuration
            .get("parameter_count")
            .and_then(Value::as_u64)
            .is_some_and(|count| count > 0 && count < 1_000_000_000)
        {
            return Err(
                "Local CPU model must report fewer than one billion loaded parameters".into(),
            );
        }
        let version = value
            .get("version")
            .and_then(Value::as_str)
            .filter(|version| !version.is_empty())
            .ok_or_else(|| "Local CPU endpoint has no server version".to_string())?;
        Ok(ProviderMetadata {
            version: Some(version.to_string()),
            configuration: configuration.clone(),
        })
    }

    async fn local_cpu_invoke(
        &self,
        task: Task,
        case: &BenchmarkCase,
        max_output_tokens: u32,
    ) -> Result<InvocationResult, String> {
        let mut body = json!({"model": self.model, "task": task.as_str(), "case": case, "max_output_tokens": max_output_tokens});
        if let Some(media) = &case.media {
            let (mime, bytes) = media_bytes(media, case.media_type.as_deref())?;
            body["case"]["media"] = json!(format!("data:{mime};base64,{}", STANDARD.encode(bytes)));
        }
        let started = Instant::now();
        let response = self
            .authorized(self.client.post(self.url("/v1/qit/invoke")))
            .json(&body)
            .send()
            .await
            .map_err(|error| format!("Local CPU request failed: {error}"))?;
        let response = successful(response, "Local CPU invocation").await?;
        let mut stream = response.bytes_stream();
        let mut buffer = Vec::new();
        let mut output = InvocationResult::default();
        let mut terminal = false;
        while let Some(chunk) = stream.next().await {
            buffer.extend_from_slice(
                &chunk.map_err(|error| format!("read Local CPU stream: {error}"))?,
            );
            if buffer.len() > 16 * 1024 * 1024 {
                return Err("Local CPU response exceeded the maximum event size".into());
            }
            while let Some(newline) = buffer.iter().position(|byte| *byte == b'\n') {
                let line = buffer.drain(..=newline).collect::<Vec<_>>();
                parse_local_cpu_event(&line, task, started, &mut output, &mut terminal)?;
            }
        }
        if !buffer.is_empty() {
            parse_local_cpu_event(&buffer, task, started, &mut output, &mut terminal)?;
        }
        if !terminal {
            return Err("Local CPU stream ended without a terminal result".into());
        }
        output.latency_ms = elapsed_ms(started);
        if matches!(task, Task::TextGeneration | Task::ImageToText) && output.output_text.is_none()
        {
            return Err("Local CPU provider returned no text".into());
        }
        if task == Task::SpeechToText && output.output_text.is_none() {
            return Err("Local CPU provider returned no transcript".into());
        }
        if task == Task::TextToImage {
            validate_generated_image(
                output
                    .output_image
                    .as_deref()
                    .ok_or_else(|| "Local CPU provider returned no generated image".to_string())?,
                case.image_settings.as_ref(),
            )?;
        }
        if task == Task::Reranking {
            let mut ranking = output.ranking.clone();
            ranking.sort_unstable();
            if ranking != (0..case.documents.len()).collect::<Vec<_>>() {
                return Err(
                    "Local CPU ranking must contain every candidate index exactly once".into(),
                );
            }
        }
        Ok(output)
    }

    async fn ollama_preflight(&self, task: Task) -> Result<(), String> {
        let response = self
            .authorized(self.client.post(self.url("/api/show")))
            .json(&json!({ "model": self.model }))
            .send()
            .await
            .map_err(|error| format!("Ollama model preflight failed: {error}"))?;
        let value = response_json(response, "Ollama model preflight").await?;
        let capabilities = value
            .get("capabilities")
            .and_then(Value::as_array)
            .ok_or_else(|| "Ollama model details do not report capabilities".to_string())?
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>();
        let required = match task {
            Task::TextGeneration => "completion",
            Task::ImageToText => "vision",
            Task::Embedding => "embedding",
            _ => return Err(format!("Ollama does not support {}", task.label())),
        };
        if capabilities.contains(&required) {
            Ok(())
        } else {
            Err(format!(
                "Ollama model '{}' does not report the required '{required}' capability",
                self.model
            ))
        }
    }

    async fn transformers_preflight(&self) -> Result<(), String> {
        let response = self
            .authorized(self.client.get(self.url("/v1/models")))
            .send()
            .await
            .map_err(|error| format!("Transformers Serve preflight failed: {error}"))?;
        successful(response, "Transformers Serve preflight")
            .await
            .map(|_| ())
    }

    async fn tei_preflight(&self) -> Result<(), String> {
        let response = self
            .authorized(self.client.get(self.url("/health")))
            .send()
            .await
            .map_err(|error| format!("TEI preflight failed: {error}"))?;
        successful(response, "TEI preflight").await.map(|_| ())
    }

    async fn hf_serve_preflight(&self) -> Result<(), String> {
        let response = self
            .authorized(self.client.get(self.url("/v1/models")))
            .send()
            .await
            .map_err(|error| format!("Hugging Face Serve preflight failed: {error}"))?;
        let value = response_json(response, "Hugging Face Serve preflight").await?;
        let models = value
            .get("data")
            .and_then(Value::as_array)
            .ok_or_else(|| "Hugging Face Serve model list has no data array".to_string())?;
        if models
            .iter()
            .any(|model| model.get("id").and_then(Value::as_str) == Some(self.model.as_str()))
        {
            Ok(())
        } else {
            Err(format!(
                "Hugging Face Serve does not list the requested model '{}'",
                self.model
            ))
        }
    }

    async fn hf_serve_embed(&self, case: &BenchmarkCase) -> Result<InvocationResult, String> {
        let inputs = embedding_inputs(case)?;
        let started = Instant::now();
        let mut embeddings = Vec::with_capacity(inputs.len());
        let mut input_tokens = Some(0_u64);
        for batch in inputs.chunks(HF_SERVE_MAX_BATCH_INPUTS) {
            let response = self
                .authorized(self.client.post(self.url("/v1/embeddings")))
                .json(&json!({ "model": self.model, "input": batch, "encoding_format": "float" }))
                .send()
                .await
                .map_err(|error| format!("Hugging Face Serve embedding request failed: {error}"))?;
            let value = response_json(response, "Hugging Face Serve embedding").await?;
            embeddings.extend(parse_openai_embeddings(&value, batch.len())?);
            input_tokens = input_tokens.and_then(|total| {
                value
                    .get("usage")
                    .and_then(|usage| usage.get("prompt_tokens"))
                    .and_then(Value::as_u64)
                    .and_then(|tokens| total.checked_add(tokens))
            });
        }
        Ok(InvocationResult {
            embeddings,
            latency_ms: elapsed_ms(started),
            input_tokens,
            ..InvocationResult::default()
        })
    }

    async fn ollama_generate(
        &self,
        case: &BenchmarkCase,
        max_output_tokens: u32,
    ) -> Result<InvocationResult, String> {
        let mut body = json!({
            "model": self.model,
            "prompt": required(&case.prompt, "prompt")?,
            "stream": true,
            "options": { "num_predict": max_output_tokens }
        });
        if let Some(media) = &case.media {
            body["images"] = json!([media_base64(media)?]);
        }
        let started = Instant::now();
        let response = self
            .authorized(self.client.post(self.url("/api/generate")))
            .json(&body)
            .send()
            .await
            .map_err(|error| format!("Ollama generation request failed: {error}"))?;
        let response = successful(response, "Ollama generation").await?;
        let mut stream = response.bytes_stream();
        let mut buffer = Vec::new();
        let mut state = OllamaGenerationState::default();
        while let Some(chunk) = stream.next().await {
            buffer.extend_from_slice(
                &chunk.map_err(|error| format!("read Ollama generation stream: {error}"))?,
            );
            while let Some(newline) = buffer.iter().position(|byte| *byte == b'\n') {
                let line = buffer.drain(..=newline).collect::<Vec<_>>();
                parse_ollama_event(&line, started, &mut state)?;
            }
        }
        if !buffer.is_empty() {
            parse_ollama_event(&buffer, started, &mut state)?;
        }
        if !state.terminal {
            return Err("Ollama generation stream ended without a terminal marker".into());
        }
        let latency_ms = elapsed_ms(started);
        let tokens_per_second = match (state.output_tokens, state.eval_duration_ns) {
            (Some(tokens), Some(duration)) if duration > 0 => {
                Some(tokens as f64 / (duration as f64 / 1_000_000_000.0))
            }
            _ => timed_throughput(state.output_tokens, state.ttft_ms, latency_ms),
        };
        Ok(InvocationResult {
            output_text: Some(state.output),
            latency_ms,
            ttft_ms: state.ttft_ms,
            input_tokens: state.input_tokens,
            output_tokens: state.output_tokens,
            tokens_per_second,
            ..InvocationResult::default()
        })
    }

    async fn ollama_embed(&self, case: &BenchmarkCase) -> Result<InvocationResult, String> {
        let inputs = embedding_inputs(case)?;
        let started = Instant::now();
        let response = self
            .authorized(self.client.post(self.url("/api/embed")))
            .json(&json!({ "model": self.model, "input": inputs }))
            .send()
            .await
            .map_err(|error| format!("Ollama embedding request failed: {error}"))?;
        let value = response_json(response, "Ollama embedding").await?;
        let embeddings = parse_embeddings(
            value
                .get("embeddings")
                .ok_or_else(|| "Ollama embedding response has no embeddings".to_string())?,
        )?;
        Ok(InvocationResult {
            embeddings,
            latency_ms: elapsed_ms(started),
            input_tokens: value.get("prompt_eval_count").and_then(Value::as_u64),
            ..InvocationResult::default()
        })
    }

    async fn transformers_generate(
        &self,
        case: &BenchmarkCase,
        max_output_tokens: u32,
    ) -> Result<InvocationResult, String> {
        let prompt = required(&case.prompt, "prompt")?;
        let content = match &case.media {
            Some(media) => json!([
                { "type": "text", "text": prompt },
                { "type": "image_url", "image_url": { "url": media_data_url(media, case.media_type.as_deref())? } }
            ]),
            None => Value::String(prompt.to_string()),
        };
        let body = json!({
            "model": self.model,
            "messages": [{ "role": "user", "content": content }],
            "max_tokens": max_output_tokens,
            "temperature": 0,
            "stream": true,
            "stream_options": { "include_usage": true }
        });
        let started = Instant::now();
        let response = self
            .authorized(self.client.post(self.url("/v1/chat/completions")))
            .json(&body)
            .send()
            .await
            .map_err(|error| format!("Transformers Serve generation request failed: {error}"))?;
        let response = successful(response, "Transformers Serve generation").await?;
        let mut stream = response.bytes_stream();
        let mut buffer = Vec::new();
        let mut raw = Vec::new();
        let mut output = String::new();
        let mut ttft_ms = None;
        let mut input_tokens = None;
        let mut output_tokens = None;
        let mut saw_event = false;
        let mut terminal = false;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk
                .map_err(|error| format!("read Transformers Serve generation stream: {error}"))?;
            raw.extend_from_slice(&chunk);
            buffer.extend_from_slice(&chunk);
            while let Some(newline) = buffer.iter().position(|byte| *byte == b'\n') {
                let line = buffer.drain(..=newline).collect::<Vec<_>>();
                if parse_transformers_event(
                    &line,
                    started,
                    &mut output,
                    &mut ttft_ms,
                    &mut input_tokens,
                    &mut output_tokens,
                    &mut terminal,
                )? {
                    saw_event = true;
                }
            }
        }
        if !buffer.is_empty()
            && parse_transformers_event(
                &buffer,
                started,
                &mut output,
                &mut ttft_ms,
                &mut input_tokens,
                &mut output_tokens,
                &mut terminal,
            )?
        {
            saw_event = true;
        }
        if !saw_event {
            let value: Value = serde_json::from_slice(&raw).map_err(|error| {
                format!("parse Transformers Serve generation response: {error}")
            })?;
            output = value
                .pointer("/choices/0/message/content")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            input_tokens = value
                .pointer("/usage/prompt_tokens")
                .and_then(Value::as_u64);
            output_tokens = value
                .pointer("/usage/completion_tokens")
                .and_then(Value::as_u64);
        } else if !terminal {
            return Err("Transformers Serve stream ended without a terminal marker".into());
        }
        let latency_ms = elapsed_ms(started);
        Ok(InvocationResult {
            output_text: Some(output),
            latency_ms,
            ttft_ms,
            input_tokens,
            output_tokens,
            tokens_per_second: timed_throughput(output_tokens, ttft_ms, latency_ms),
            ..InvocationResult::default()
        })
    }

    async fn transformers_transcribe(
        &self,
        case: &BenchmarkCase,
    ) -> Result<InvocationResult, String> {
        let media = required(&case.media, "media")?;
        let (mime, bytes) = media_bytes(media, case.media_type.as_deref())?;
        let part = reqwest::multipart::Part::bytes(bytes)
            .file_name(audio_filename(&mime))
            .mime_str(&mime)
            .map_err(|error| format!("invalid audio media type '{mime}': {error}"))?;
        let form = reqwest::multipart::Form::new()
            .text("model", self.model.clone())
            .part("file", part);
        let started = Instant::now();
        let response = self
            .authorized(self.client.post(self.url("/v1/audio/transcriptions")))
            .multipart(form)
            .send()
            .await
            .map_err(|error| format!("Transformers Serve transcription request failed: {error}"))?;
        let value = response_json(response, "Transformers Serve transcription").await?;
        let text = value
            .get("text")
            .and_then(Value::as_str)
            .ok_or_else(|| "transcription response has no text".to_string())?;
        Ok(InvocationResult {
            output_text: Some(text.to_string()),
            latency_ms: elapsed_ms(started),
            ..InvocationResult::default()
        })
    }

    async fn tei_embed(&self, case: &BenchmarkCase) -> Result<InvocationResult, String> {
        let started = Instant::now();
        let response = self
            .authorized(self.client.post(self.url("/embed")))
            .json(&json!({ "inputs": embedding_inputs(case)? }))
            .send()
            .await
            .map_err(|error| format!("TEI embedding request failed: {error}"))?;
        let value = response_json(response, "TEI embedding").await?;
        let embeddings = parse_embeddings(&value)?;
        Ok(InvocationResult {
            embeddings,
            latency_ms: elapsed_ms(started),
            ..InvocationResult::default()
        })
    }

    async fn tei_rerank(&self, case: &BenchmarkCase) -> Result<InvocationResult, String> {
        let started = Instant::now();
        let response = self
            .authorized(self.client.post(self.url("/rerank")))
            .json(&json!({
                "query": required(&case.input, "input")?,
                "texts": case.documents,
                "return_text": false,
                "raw_scores": false
            }))
            .send()
            .await
            .map_err(|error| format!("TEI reranking request failed: {error}"))?;
        let value = response_json(response, "TEI reranking").await?;
        let items = value
            .as_array()
            .or_else(|| value.get("results").and_then(Value::as_array))
            .ok_or_else(|| "TEI reranking response is not a result array".to_string())?;
        let ranking = items
            .iter()
            .map(|item| {
                item.get("index")
                    .and_then(Value::as_u64)
                    .map(|index| index as usize)
                    .ok_or_else(|| "TEI reranking result has no index".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(InvocationResult {
            ranking,
            latency_ms: elapsed_ms(started),
            ..InvocationResult::default()
        })
    }

    fn authorized(&self, request: RequestBuilder) -> RequestBuilder {
        match &self.api_key {
            Some(api_key) => request.bearer_auth(api_key),
            None => request,
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base_url)
    }
}

#[derive(Default)]
struct OllamaGenerationState {
    output: String,
    ttft_ms: Option<f64>,
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    eval_duration_ns: Option<u64>,
    terminal: bool,
}

fn parse_ollama_event(
    bytes: &[u8],
    started: Instant,
    state: &mut OllamaGenerationState,
) -> Result<(), String> {
    let line = String::from_utf8_lossy(bytes);
    let line = line.trim();
    if line.is_empty() {
        return Ok(());
    }
    let value: Value = serde_json::from_str(line)
        .map_err(|error| format!("parse Ollama generation event: {error}"))?;
    if let Some(error) = value.get("error").and_then(Value::as_str) {
        return Err(format!("Ollama generation failed: {error}"));
    }
    let response = value
        .get("response")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let thinking = value
        .get("thinking")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if (!response.is_empty() || !thinking.is_empty()) && state.ttft_ms.is_none() {
        state.ttft_ms = Some(elapsed_ms(started));
    }
    if !response.is_empty() {
        state.output.push_str(response);
    }
    if value.get("done").and_then(Value::as_bool) == Some(true) {
        state.terminal = true;
    }
    if let Some(value) = value.get("prompt_eval_count").and_then(Value::as_u64) {
        state.input_tokens = Some(value);
    }
    if let Some(value) = value.get("eval_count").and_then(Value::as_u64) {
        state.output_tokens = Some(value);
    }
    if let Some(value) = value.get("eval_duration").and_then(Value::as_u64) {
        state.eval_duration_ns = Some(value);
    }
    Ok(())
}

fn parse_transformers_event(
    bytes: &[u8],
    started: Instant,
    output: &mut String,
    ttft_ms: &mut Option<f64>,
    input_tokens: &mut Option<u64>,
    output_tokens: &mut Option<u64>,
    terminal: &mut bool,
) -> Result<bool, String> {
    let line = String::from_utf8_lossy(bytes);
    let line = line.trim();
    let Some(data) = line.strip_prefix("data:") else {
        return Ok(false);
    };
    let data = data.trim();
    if data.is_empty() {
        return Ok(true);
    }
    if data == "[DONE]" {
        *terminal = true;
        return Ok(true);
    }
    let value: Value = serde_json::from_str(data)
        .map_err(|error| format!("parse Transformers Serve generation event: {error}"))?;
    if let Some(error) = value.pointer("/error/message").and_then(Value::as_str) {
        return Err(format!("Transformers Serve generation failed: {error}"));
    }
    if let Some(fragment) = value
        .pointer("/choices/0/delta/content")
        .and_then(Value::as_str)
    {
        if !fragment.is_empty() {
            if ttft_ms.is_none() {
                *ttft_ms = Some(elapsed_ms(started));
            }
            output.push_str(fragment);
        }
    }
    if ttft_ms.is_none()
        && value
            .pointer("/choices/0/delta/reasoning_content")
            .and_then(Value::as_str)
            .is_some_and(|fragment| !fragment.is_empty())
    {
        *ttft_ms = Some(elapsed_ms(started));
    }
    if let Some(value) = value
        .pointer("/usage/prompt_tokens")
        .and_then(Value::as_u64)
    {
        *input_tokens = Some(value);
    }
    if let Some(value) = value
        .pointer("/usage/completion_tokens")
        .and_then(Value::as_u64)
    {
        *output_tokens = Some(value);
    }
    Ok(true)
}

fn embedding_inputs(case: &BenchmarkCase) -> Result<Vec<String>, String> {
    let query = required(&case.input, "input")?;
    let mut inputs = Vec::with_capacity(case.documents.len() + 1);
    let prepared_query = match &case.query_prefix {
        Some(prefix) => format!("{prefix}{query}"),
        None => query.to_string(),
    };
    inputs.push(prepared_query);
    for document in &case.documents {
        let prepared = match &case.document_prefix {
            Some(prefix) => format!("{prefix}{document}"),
            None => document.clone(),
        };
        inputs.push(prepared);
    }
    Ok(inputs)
}

fn parse_embeddings(value: &Value) -> Result<Vec<Vec<f64>>, String> {
    let values = value
        .as_array()
        .ok_or_else(|| "embedding response is not an array".to_string())?;
    if values.is_empty() {
        return Err("embedding response is empty".into());
    }
    if values.first().and_then(Value::as_f64).is_some() {
        return Ok(vec![parse_vector(values)?]);
    }
    values
        .iter()
        .map(|value| {
            value
                .as_array()
                .ok_or_else(|| "embedding item is not an array".to_string())
                .and_then(|values| parse_vector(values))
        })
        .collect()
}

fn parse_openai_embeddings(value: &Value, expected_count: usize) -> Result<Vec<Vec<f64>>, String> {
    let data = value
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| "embedding response has no data array".to_string())?;
    if data.len() != expected_count {
        return Err(format!(
            "provider returned {} embeddings for {expected_count} inputs",
            data.len()
        ));
    }
    let mut embeddings = vec![None; expected_count];
    for item in data {
        let index = item
            .get("index")
            .and_then(Value::as_u64)
            .and_then(|index| usize::try_from(index).ok())
            .filter(|index| *index < expected_count)
            .ok_or_else(|| "embedding item has an invalid input index".to_string())?;
        if embeddings[index].is_some() {
            return Err(format!("embedding response repeats input index {index}"));
        }
        let values = item
            .get("embedding")
            .and_then(Value::as_array)
            .ok_or_else(|| "embedding item has no numeric vector".to_string())?;
        if values.is_empty() {
            return Err("provider returned an empty embedding".into());
        }
        embeddings[index] = Some(parse_vector(values)?);
    }
    embeddings
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| "embedding response is missing an input index".to_string())
}

fn parse_vector(values: &[Value]) -> Result<Vec<f64>, String> {
    values
        .iter()
        .map(|value| {
            value
                .as_f64()
                .ok_or_else(|| "embedding contains a non-numeric value".to_string())
        })
        .collect()
}

fn required<'a>(value: &'a Option<String>, label: &str) -> Result<&'a str, String> {
    value
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| format!("benchmark case requires {label}"))
}

fn media_base64(media: &str) -> Result<String, String> {
    if let Some((_, encoded)) = split_data_url(media)? {
        return Ok(encoded.to_string());
    }
    let bytes = std::fs::read(media).map_err(|error| format!("read media {media}: {error}"))?;
    Ok(STANDARD.encode(bytes))
}

fn media_data_url(media: &str, declared_type: Option<&str>) -> Result<String, String> {
    if media.starts_with("data:") || media.starts_with("http://") || media.starts_with("https://") {
        return Ok(media.to_string());
    }
    let mime = declared_type
        .map(str::to_string)
        .unwrap_or_else(|| media_type_for_path(media));
    let bytes = std::fs::read(media).map_err(|error| format!("read media {media}: {error}"))?;
    Ok(format!("data:{mime};base64,{}", STANDARD.encode(bytes)))
}

fn media_bytes(media: &str, declared_type: Option<&str>) -> Result<(String, Vec<u8>), String> {
    if let Some((mime, encoded)) = split_data_url(media)? {
        let bytes = STANDARD
            .decode(encoded)
            .map_err(|error| format!("decode media data URL: {error}"))?;
        return Ok((mime.to_string(), bytes));
    }
    if media.starts_with("http://") || media.starts_with("https://") {
        return Err("speech benchmark media must be a local path or data URL".into());
    }
    let mime = declared_type
        .map(str::to_string)
        .unwrap_or_else(|| media_type_for_path(media));
    let bytes = std::fs::read(media).map_err(|error| format!("read media {media}: {error}"))?;
    Ok((mime, bytes))
}

fn split_data_url(value: &str) -> Result<Option<(&str, &str)>, String> {
    let Some(rest) = value.strip_prefix("data:") else {
        return Ok(None);
    };
    let (metadata, data) = rest
        .split_once(',')
        .ok_or_else(|| "media data URL has no comma".to_string())?;
    let mime = metadata
        .strip_suffix(";base64")
        .ok_or_else(|| "media data URL must use base64 encoding".to_string())?;
    if mime.is_empty() || data.is_empty() {
        return Err("media data URL is incomplete".into());
    }
    Ok(Some((mime, data)))
}

fn media_type_for_path(value: &str) -> String {
    match Path::new(value)
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "wav" => "audio/wav",
        "mp3" => "audio/mpeg",
        "m4a" => "audio/mp4",
        "flac" => "audio/flac",
        _ => "application/octet-stream",
    }
    .to_string()
}

fn audio_filename(mime: &str) -> String {
    let extension = match mime {
        "audio/mpeg" => "mp3",
        "audio/mp4" => "m4a",
        "audio/flac" => "flac",
        _ => "wav",
    };
    format!("benchmark.{extension}")
}

async fn successful(response: Response, action: &str) -> Result<Response, String> {
    if response.status().is_success() {
        return Ok(response);
    }
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    Err(format!("{action} returned {status}: {}", excerpt(&body)))
}

async fn response_json(response: Response, action: &str) -> Result<Value, String> {
    let response = successful(response, action).await?;
    response
        .json()
        .await
        .map_err(|error| format!("parse {action} response: {error}"))
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

fn timed_throughput(
    output_tokens: Option<u64>,
    ttft_ms: Option<f64>,
    latency_ms: f64,
) -> Option<f64> {
    match (output_tokens, ttft_ms) {
        (Some(tokens), Some(ttft)) if tokens >= 2 && latency_ms > ttft => {
            Some((tokens - 1) as f64 / ((latency_ms - ttft) / 1000.0))
        }
        _ => None,
    }
}

fn excerpt(value: &str) -> String {
    value.chars().take(500).collect()
}

fn parse_local_cpu_event(
    line: &[u8],
    task: Task,
    started: Instant,
    output: &mut InvocationResult,
    terminal: &mut bool,
) -> Result<(), String> {
    if line.iter().all(u8::is_ascii_whitespace) {
        return Ok(());
    }
    if *terminal {
        return Err("Local CPU stream contains data after its terminal result".into());
    }
    let value: Value =
        serde_json::from_slice(line).map_err(|error| format!("parse Local CPU stream: {error}"))?;
    match value.get("type").and_then(Value::as_str) {
        Some("error") => Err(format!(
            "Local CPU inference failed: {}",
            value
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("unknown error")
        )),
        Some("delta") if matches!(task, Task::TextGeneration | Task::ImageToText) => {
            let text = value
                .get("text")
                .and_then(Value::as_str)
                .ok_or_else(|| "Local CPU text event has no text".to_string())?;
            if !text.is_empty() {
                output.ttft_ms.get_or_insert_with(|| elapsed_ms(started));
            }
            output
                .output_text
                .get_or_insert_with(String::new)
                .push_str(text);
            Ok(())
        }
        Some("result") => {
            match task {
                Task::TextGeneration | Task::ImageToText | Task::SpeechToText => {
                    if let Some(text) = value.get("output_text").and_then(Value::as_str) {
                        output.output_text = Some(text.to_string());
                    }
                    if task != Task::SpeechToText {
                        output.input_tokens = value.get("input_tokens").and_then(Value::as_u64);
                        output.output_tokens = value.get("output_tokens").and_then(Value::as_u64);
                        output.ttft_ms = value
                            .get("first_token_latency_ns")
                            .and_then(Value::as_u64)
                            .map(|duration| duration as f64 / 1_000_000.0);
                        if let (Some(tokens), Some(duration)) = (
                            output.output_tokens,
                            value.get("decode_duration_ns").and_then(Value::as_u64),
                        ) {
                            if tokens >= 2 && duration > 0 {
                                output.tokens_per_second =
                                    Some((tokens - 1) as f64 / (duration as f64 / 1_000_000_000.0));
                            }
                        }
                    }
                }
                Task::Embedding => {
                    output.embeddings = parse_embeddings(
                        value
                            .get("embeddings")
                            .ok_or_else(|| "Local CPU result has no embeddings".to_string())?,
                    )?;
                }
                Task::Reranking => {
                    output.ranking = value
                        .get("ranking")
                        .and_then(Value::as_array)
                        .ok_or_else(|| "Local CPU result has no ranking".to_string())?
                        .iter()
                        .map(|index| {
                            index
                                .as_u64()
                                .and_then(|index| usize::try_from(index).ok())
                                .ok_or_else(|| "Local CPU ranking has an invalid index".to_string())
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                }
                Task::TextToImage => {
                    output.output_image = Some(
                        value
                            .get("output_image")
                            .and_then(Value::as_str)
                            .ok_or_else(|| "Local CPU result has no generated image".to_string())?
                            .to_string(),
                    );
                }
            }
            *terminal = true;
            Ok(())
        }
        _ => Err("Local CPU stream contains an unsupported event".into()),
    }
}

pub fn validate_generated_image(
    value: &str,
    settings: Option<&crate::model::ImageSettings>,
) -> Result<(), String> {
    let encoded = value
        .strip_prefix("data:image/png;base64,")
        .ok_or_else(|| "generated image must be a base64 PNG".to_string())?;
    if encoded.len() > 16 * 1024 * 1024 {
        return Err("generated image exceeds the maximum size".into());
    }
    let bytes = STANDARD
        .decode(encoded)
        .map_err(|error| format!("decode generated PNG: {error}"))?;
    let mut reader =
        image::ImageReader::with_format(std::io::Cursor::new(bytes), image::ImageFormat::Png);
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(1024);
    limits.max_image_height = Some(1024);
    limits.max_alloc = Some(32 * 1024 * 1024);
    reader.limits(limits);
    let image = reader
        .decode()
        .map_err(|error| format!("invalid generated PNG: {error}"))?;
    let expected = settings.cloned().unwrap_or_default();
    if image.width() != expected.width || image.height() != expected.height {
        return Err(format!(
            "generated image is {}x{}, expected {}x{}",
            image.width(),
            image.height(),
            expected.width,
            expected.height
        ));
    }
    Ok(())
}
