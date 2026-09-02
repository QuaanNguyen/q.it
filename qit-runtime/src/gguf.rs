use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

const EMBED_ARCHS: &[&str] = &[
    "bert",
    "nomic-bert",
    "nomic-bert-moe",
    "jina-bert-v2",
    "jina-bert-v3",
    "modern-bert",
    "neo-bert",
    "eurobert",
    "gemma-embedding",
    "t5encoder",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    Instruct,
    Base,
    Embedding,
    Rerank,
    VisionProjector,
    Unknown,
}

impl ArtifactKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ArtifactKind::Instruct => "instruct",
            ArtifactKind::Base => "base",
            ArtifactKind::Embedding => "embedding",
            ArtifactKind::Rerank => "rerank",
            ArtifactKind::VisionProjector => "vision_projector",
            ArtifactKind::Unknown => "unknown",
        }
    }

    pub fn generate_supported(self) -> bool {
        self == ArtifactKind::Instruct
    }

    pub fn parse(s: &str) -> Self {
        match s {
            "instruct" => ArtifactKind::Instruct,
            "base" => ArtifactKind::Base,
            "embedding" => ArtifactKind::Embedding,
            "rerank" => ArtifactKind::Rerank,
            "vision_projector" => ArtifactKind::VisionProjector,
            _ => ArtifactKind::Unknown,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PlannerHints {
    pub head_count_kv_layers: Option<Vec<u32>>,
    pub feed_forward_layers: Option<Vec<u32>>,
    pub key_length: Option<u32>,
    pub value_length: Option<u32>,
    pub ssm_conv_kernel: Option<u32>,
    pub ssm_inner_size: Option<u32>,
    pub ssm_state_size: Option<u32>,
    pub ssm_group_count: Option<u32>,
}

#[derive(Clone, Debug, Default)]
pub struct GgufMeta {
    pub architecture: Option<String>,
    pub general_type: Option<String>,
    pub context_length: Option<u32>,
    pub block_count: Option<u32>,
    pub embedding_length: Option<u32>,
    pub head_count: Option<u32>,
    pub head_count_kv: Option<u32>,
    pub head_count_kv_layers: Option<Vec<u32>>,
    pub feed_forward_layers: Option<Vec<u32>>,
    pub key_length: Option<u32>,
    pub value_length: Option<u32>,
    pub ssm_conv_kernel: Option<u32>,
    pub ssm_inner_size: Option<u32>,
    pub ssm_state_size: Option<u32>,
    pub ssm_group_count: Option<u32>,
    pub has_chat_template: bool,
    pub has_rerank_template: bool,
    pub pooling_type: Option<u32>,
    pub causal_attn: Option<bool>,
    pub has_classifier_labels: bool,
    pub has_clip_keys: bool,
}

impl GgufMeta {
    pub fn kind(&self) -> ArtifactKind {
        classify_kind(self)
    }

    pub fn planner(&self) -> PlannerHints {
        PlannerHints {
            head_count_kv_layers: self.head_count_kv_layers.clone(),
            feed_forward_layers: self.feed_forward_layers.clone(),
            key_length: self.key_length,
            value_length: self.value_length,
            ssm_conv_kernel: self.ssm_conv_kernel,
            ssm_inner_size: self.ssm_inner_size,
            ssm_state_size: self.ssm_state_size,
            ssm_group_count: self.ssm_group_count,
        }
    }

    pub fn scalar_head_count_kv(&self) -> Option<u32> {
        if let Some(layers) = &self.head_count_kv_layers {
            return layers.iter().copied().find(|n| *n > 0).or(Some(0));
        }
        self.head_count_kv
    }
}

pub fn classify_kind(meta: &GgufMeta) -> ArtifactKind {
    let arch = meta.architecture.as_deref().unwrap_or("");
    if meta.general_type.as_deref() == Some("mmproj")
        || arch == "clip"
        || meta.has_clip_keys
    {
        return ArtifactKind::VisionProjector;
    }
    if let Some(ty) = meta.general_type.as_deref() {
        if ty != "model" {
            return ArtifactKind::Unknown;
        }
    }
    if meta.pooling_type == Some(4) || meta.has_classifier_labels || meta.has_rerank_template {
        return ArtifactKind::Rerank;
    }
    if meta
        .pooling_type
        .is_some_and(|p| (1..=3).contains(&p))
        || meta.causal_attn == Some(false)
        || EMBED_ARCHS.contains(&arch)
    {
        return ArtifactKind::Embedding;
    }
    if arch.is_empty() {
        return ArtifactKind::Unknown;
    }
    if meta.has_chat_template {
        ArtifactKind::Instruct
    } else {
        ArtifactKind::Base
    }
}

pub fn read_gguf_meta(path: &Path) -> Option<GgufMeta> {
    let mut file = File::open(path).ok()?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic).ok()?;
    if &magic != b"GGUF" {
        return None;
    }
    let version = read_u32(&mut file)?;
    if version < 2 {
        return None;
    }
    let _n_tensors = read_u64(&mut file)?;
    let n_kv = read_u64(&mut file)?;
    let mut map = BTreeMap::new();
    let mut has_clip_keys = false;
    let mut has_rerank_template = false;
    let mut has_classifier_labels = false;
    for _ in 0..n_kv {
        let key = read_string(&mut file)?;
        let ty = read_u32(&mut file)?;
        let value = read_value(&mut file, ty)?;
        if key.starts_with("clip.") {
            has_clip_keys = true;
        }
        if key == "tokenizer.chat_template.rerank" {
            has_rerank_template = true;
        }
        if key.ends_with(".classifier.output_labels") {
            has_classifier_labels = true;
        }
        map.insert(key, value);
    }
    let architecture = map.get("general.architecture").and_then(Value::as_string);
    let prefix = architecture.clone().unwrap_or_default();
    let chat = map
        .get("tokenizer.chat_template")
        .and_then(Value::as_string)
        .map(|s| !s.is_empty())
        .unwrap_or(false);
    let (head_count_kv, head_count_kv_layers) =
        u32_or_arr(&map, &prefix, "attention.head_count_kv");
    let (_, feed_forward_layers) = u32_or_arr(&map, &prefix, "feed_forward_length");
    Some(GgufMeta {
        architecture: architecture.clone(),
        general_type: map.get("general.type").and_then(Value::as_string),
        context_length: u32_at(&map, &prefix, "context_length"),
        block_count: u32_at(&map, &prefix, "block_count"),
        embedding_length: u32_at(&map, &prefix, "embedding_length"),
        head_count: u32_at(&map, &prefix, "attention.head_count"),
        head_count_kv,
        head_count_kv_layers,
        feed_forward_layers,
        key_length: u32_at(&map, &prefix, "attention.key_length"),
        value_length: u32_at(&map, &prefix, "attention.value_length"),
        ssm_conv_kernel: u32_at(&map, &prefix, "ssm.conv_kernel"),
        ssm_inner_size: u32_at(&map, &prefix, "ssm.inner_size"),
        ssm_state_size: u32_at(&map, &prefix, "ssm.state_size"),
        ssm_group_count: u32_at(&map, &prefix, "ssm.group_count"),
        has_chat_template: chat,
        has_rerank_template,
        pooling_type: u32_at(&map, &prefix, "pooling_type"),
        causal_attn: bool_at(&map, &prefix, "attention.causal"),
        has_classifier_labels,
        has_clip_keys,
    })
}

fn u32_at(map: &BTreeMap<String, Value>, arch: &str, field: &str) -> Option<u32> {
    let key = format!("{arch}.{field}");
    map.get(&key).and_then(Value::as_u64).map(|v| v as u32)
}

fn bool_at(map: &BTreeMap<String, Value>, arch: &str, field: &str) -> Option<bool> {
    let key = format!("{arch}.{field}");
    map.get(&key).and_then(Value::as_bool)
}

fn u32_or_arr(
    map: &BTreeMap<String, Value>,
    arch: &str,
    field: &str,
) -> (Option<u32>, Option<Vec<u32>>) {
    let key = format!("{arch}.{field}");
    match map.get(&key) {
        Some(Value::Array(items)) => {
            let first = items.first().copied();
            (first, Some(items.clone()))
        }
        Some(Value::Uint(v)) => (Some(*v as u32), None),
        _ => (None, None),
    }
}

#[derive(Clone, Debug)]
enum Value {
    Uint(u64),
    Str(String),
    Bool(bool),
    Array(Vec<u32>),
    Other,
}

impl Value {
    fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Uint(v) => Some(*v),
            _ => None,
        }
    }

    fn as_string(&self) -> Option<String> {
        match self {
            Value::Str(v) => Some(v.clone()),
            _ => None,
        }
    }

    fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

fn read_u32(file: &mut File) -> Option<u32> {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf).ok()?;
    Some(u32::from_le_bytes(buf))
}

fn read_u64(file: &mut File) -> Option<u64> {
    let mut buf = [0u8; 8];
    file.read_exact(&mut buf).ok()?;
    Some(u64::from_le_bytes(buf))
}

fn read_string(file: &mut File) -> Option<String> {
    let len = read_u64(file)? as usize;
    if len > 1024 * 1024 {
        return None;
    }
    let mut buf = vec![0u8; len];
    file.read_exact(&mut buf).ok()?;
    String::from_utf8(buf).ok()
}

fn read_value(file: &mut File, ty: u32) -> Option<Value> {
    match ty {
        0 => {
            let mut b = [0u8; 1];
            file.read_exact(&mut b).ok()?;
            Some(Value::Uint(b[0] as u64))
        }
        4 => Some(Value::Uint(read_u32(file)? as u64)),
        5 => {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf).ok()?;
            Some(Value::Other)
        }
        6 => {
            file.seek(SeekFrom::Current(4)).ok()?;
            Some(Value::Other)
        }
        7 => {
            let mut b = [0u8; 1];
            file.read_exact(&mut b).ok()?;
            Some(Value::Bool(b[0] != 0))
        }
        8 => Some(Value::Str(read_string(file)?)),
        10 => Some(Value::Uint(read_u64(file)?)),
        9 => read_array(file),
        11 | 12 => {
            file.seek(SeekFrom::Current(8)).ok()?;
            Some(Value::Other)
        }
        1 | 2 | 3 => {
            let skip = match ty {
                1 => 1,
                2 | 3 => 2,
                _ => 0,
            };
            file.seek(SeekFrom::Current(skip)).ok()?;
            Some(Value::Other)
        }
        _ => None,
    }
}

fn read_array(file: &mut File) -> Option<Value> {
    let elem = read_u32(file)?;
    let n = read_u64(file)?;
    if n > 4096 {
        for _ in 0..n {
            read_value(file, elem)?;
        }
        return Some(Value::Other);
    }
    if matches!(elem, 0 | 4 | 10) {
        let mut items = Vec::with_capacity(n as usize);
        for _ in 0..n {
            match read_value(file, elem)? {
                Value::Uint(v) => items.push(v as u32),
                _ => return Some(Value::Other),
            }
        }
        return Some(Value::Array(items));
    }
    for _ in 0..n {
        read_value(file, elem)?;
    }
    Some(Value::Other)
}

pub fn write_test_gguf(path: &Path, meta: &GgufMeta) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    file.write_all(b"GGUF")?;
    file.write_all(&3u32.to_le_bytes())?;
    file.write_all(&0u64.to_le_bytes())?;
    let arch = meta.architecture.clone().unwrap_or_else(|| "llama".into());
    let mut kvs: Vec<(String, TestVal)> =
        vec![("general.architecture".into(), TestVal::Str(arch.clone()))];
    if let Some(v) = &meta.general_type {
        kvs.push(("general.type".into(), TestVal::Str(v.clone())));
    }
    if let Some(v) = meta.context_length {
        kvs.push((format!("{arch}.context_length"), TestVal::U64(v as u64)));
    }
    if let Some(v) = meta.block_count {
        kvs.push((format!("{arch}.block_count"), TestVal::U64(v as u64)));
    }
    if let Some(v) = meta.embedding_length {
        kvs.push((format!("{arch}.embedding_length"), TestVal::U64(v as u64)));
    }
    if let Some(v) = meta.head_count {
        kvs.push((
            format!("{arch}.attention.head_count"),
            TestVal::U64(v as u64),
        ));
    }
    if let Some(layers) = &meta.head_count_kv_layers {
        kvs.push((
            format!("{arch}.attention.head_count_kv"),
            TestVal::Arr(layers.clone()),
        ));
    } else if let Some(v) = meta.head_count_kv {
        kvs.push((
            format!("{arch}.attention.head_count_kv"),
            TestVal::U64(v as u64),
        ));
    }
    if let Some(layers) = &meta.feed_forward_layers {
        kvs.push((format!("{arch}.feed_forward_length"), TestVal::Arr(layers.clone())));
    }
    if let Some(v) = meta.key_length {
        kvs.push((format!("{arch}.attention.key_length"), TestVal::U64(v as u64)));
    }
    if let Some(v) = meta.value_length {
        kvs.push((
            format!("{arch}.attention.value_length"),
            TestVal::U64(v as u64),
        ));
    }
    if let Some(v) = meta.ssm_conv_kernel {
        kvs.push((format!("{arch}.ssm.conv_kernel"), TestVal::U64(v as u64)));
    }
    if let Some(v) = meta.ssm_inner_size {
        kvs.push((format!("{arch}.ssm.inner_size"), TestVal::U64(v as u64)));
    }
    if let Some(v) = meta.ssm_state_size {
        kvs.push((format!("{arch}.ssm.state_size"), TestVal::U64(v as u64)));
    }
    if let Some(v) = meta.ssm_group_count {
        kvs.push((format!("{arch}.ssm.group_count"), TestVal::U64(v as u64)));
    }
    if let Some(v) = meta.pooling_type {
        kvs.push((format!("{arch}.pooling_type"), TestVal::U64(v as u64)));
    }
    if let Some(v) = meta.causal_attn {
        kvs.push((format!("{arch}.attention.causal"), TestVal::Bool(v)));
    }
    if meta.has_chat_template {
        kvs.push(("tokenizer.chat_template".into(), TestVal::Str("{% endif %}".into())));
    }
    if meta.has_rerank_template {
        kvs.push((
            "tokenizer.chat_template.rerank".into(),
            TestVal::Str("rerank".into()),
        ));
    }
    if meta.has_clip_keys {
        kvs.push(("clip.has_vision_encoder".into(), TestVal::Bool(true)));
    }
    if meta.has_classifier_labels {
        kvs.push((
            format!("{arch}.classifier.output_labels"),
            TestVal::Str("yes".into()),
        ));
    }
    file.write_all(&(kvs.len() as u64).to_le_bytes())?;
    for (k, v) in kvs {
        write_string(&mut file, &k)?;
        match v {
            TestVal::Str(s) => {
                file.write_all(&8u32.to_le_bytes())?;
                write_string(&mut file, &s)?;
            }
            TestVal::U64(n) => {
                file.write_all(&10u32.to_le_bytes())?;
                file.write_all(&n.to_le_bytes())?;
            }
            TestVal::Bool(b) => {
                file.write_all(&7u32.to_le_bytes())?;
                file.write_all(&[u8::from(b)])?;
            }
            TestVal::Arr(items) => {
                file.write_all(&9u32.to_le_bytes())?;
                file.write_all(&4u32.to_le_bytes())?;
                file.write_all(&(items.len() as u64).to_le_bytes())?;
                for n in items {
                    file.write_all(&n.to_le_bytes())?;
                }
            }
        }
    }
    Ok(())
}

enum TestVal {
    Str(String),
    U64(u64),
    Bool(bool),
    Arr(Vec<u32>),
}

fn write_string(file: &mut File, s: &str) -> std::io::Result<()> {
    file.write_all(&(s.len() as u64).to_le_bytes())?;
    file.write_all(s.as_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_attention_heads() {
        let dir = std::env::temp_dir().join(format!("qit-gguf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.gguf");
        let meta = GgufMeta {
            architecture: Some("llama".into()),
            context_length: Some(32768),
            block_count: Some(1),
            embedding_length: Some(256),
            head_count: Some(4),
            head_count_kv: Some(4),
            has_chat_template: true,
            ..GgufMeta::default()
        };
        write_test_gguf(&path, &meta).unwrap();
        let got = read_gguf_meta(&path).unwrap();
        assert_eq!(got.block_count, Some(1));
        assert_eq!(got.head_count, Some(4));
        assert_eq!(got.head_count_kv, Some(4));
        assert!(got.has_chat_template);
        assert_eq!(got.kind(), ArtifactKind::Instruct);
        let _ = std::fs::remove_dir_all(dir);
    }
}
