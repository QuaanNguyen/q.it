use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

#[derive(Clone, Debug, Default)]
pub struct GgufMeta {
    pub architecture: Option<String>,
    pub context_length: Option<u32>,
    pub block_count: Option<u32>,
    pub embedding_length: Option<u32>,
    pub head_count: Option<u32>,
    pub head_count_kv: Option<u32>,
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
    for _ in 0..n_kv {
        let key = read_string(&mut file)?;
        let ty = read_u32(&mut file)?;
        let value = read_value(&mut file, ty)?;
        map.insert(key, value);
    }
    let architecture = map.get("general.architecture").and_then(Value::as_string);
    let prefix = architecture.clone().unwrap_or_default();
    Some(GgufMeta {
        architecture: architecture.clone(),
        context_length: u32_at(&map, &prefix, "context_length"),
        block_count: u32_at(&map, &prefix, "block_count"),
        embedding_length: u32_at(&map, &prefix, "embedding_length"),
        head_count: u32_at(&map, &prefix, "attention.head_count"),
        head_count_kv: u32_at(&map, &prefix, "attention.head_count_kv"),
    })
}

fn u32_at(map: &BTreeMap<String, Value>, arch: &str, field: &str) -> Option<u32> {
    let key = format!("{arch}.{field}");
    map.get(&key).and_then(Value::as_u64).map(|v| v as u32)
}

#[derive(Clone, Debug)]
enum Value {
    Uint(u64),
    Str(String),
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
            file.seek(SeekFrom::Current(1)).ok()?;
            Some(Value::Other)
        }
        8 => Some(Value::Str(read_string(file)?)),
        10 => Some(Value::Uint(read_u64(file)?)),
        9 => skip_array(file),
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

fn skip_array(file: &mut File) -> Option<Value> {
    let elem = read_u32(file)?;
    let n = read_u64(file)?;
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
    if let Some(v) = meta.head_count_kv {
        kvs.push((
            format!("{arch}.attention.head_count_kv"),
            TestVal::U64(v as u64),
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
        }
    }
    Ok(())
}

enum TestVal {
    Str(String),
    U64(u64),
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
        };
        write_test_gguf(&path, &meta).unwrap();
        let got = read_gguf_meta(&path).unwrap();
        assert_eq!(got.block_count, Some(1));
        assert_eq!(got.head_count, Some(4));
        assert_eq!(got.head_count_kv, Some(4));
        let _ = std::fs::remove_dir_all(dir);
    }
}
