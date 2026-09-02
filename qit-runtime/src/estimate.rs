use crate::store::ArtifactRow;

use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Fit {
    Fits,
    Tight,
    No,
}

impl Fit {
    pub fn as_str(self) -> &'static str {
        match self {
            Fit::Fits => "Fits",
            Fit::Tight => "Tight",
            Fit::No => "No",
        }
    }
}

pub fn estimate_bytes(artifact: &ArtifactRow, n_ctx: u32, n_parallel: u32) -> u64 {
    artifact
        .bytes
        .saturating_add(kv_cache_bytes(artifact, n_ctx, n_parallel))
}

fn kv_cache_bytes(artifact: &ArtifactRow, n_ctx: u32, n_parallel: u32) -> u64 {
    let layers = artifact.block_count.unwrap_or(0) as u64;
    if layers == 0 {
        return 0;
    }
    let heads = artifact.head_count.unwrap_or(1).max(1) as u64;
    let kv_heads = artifact
        .head_count_kv
        .unwrap_or(artifact.head_count.unwrap_or(1)) as u64;
    let embed = artifact.embedding_length.unwrap_or(0) as u64;
    let head_dim = if embed > 0 { embed / heads } else { 0 };
    let n_ctx = n_ctx as u64;
    let n_parallel = n_parallel.max(1) as u64;
    2 * layers * kv_heads * head_dim * n_ctx * n_parallel * 2
}

pub fn classify(estimate: u64, headroom: u64) -> Fit {
    if estimate > headroom {
        Fit::No
    } else if headroom > 0 && estimate.saturating_mul(5) > headroom.saturating_mul(4) {
        Fit::Tight
    } else {
        Fit::Fits
    }
}
