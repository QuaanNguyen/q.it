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
        .saturating_add(kv_and_state_bytes(artifact, n_ctx, n_parallel))
}

fn kv_and_state_bytes(artifact: &ArtifactRow, n_ctx: u32, n_parallel: u32) -> u64 {
    let layers = artifact.block_count.unwrap_or(0);
    if layers == 0 {
        return 0;
    }
    let n_ctx_eff = ((n_ctx as u64) + 255) / 256 * 256;
    let heads = artifact.head_count.unwrap_or(1).max(1) as u64;
    let embed = artifact.embedding_length.unwrap_or(0) as u64;
    let fallback_dim = if embed > 0 { embed / heads } else { 0 };
    let head_dim_k = artifact.planner.key_length.unwrap_or(fallback_dim as u32) as u64;
    let head_dim_v = artifact
        .planner
        .value_length
        .unwrap_or(fallback_dim as u32) as u64;
    let kv_layers = artifact
        .planner
        .head_count_kv_layers
        .clone()
        .unwrap_or_else(|| {
            let kv = artifact
                .head_count_kv
                .unwrap_or(artifact.head_count.unwrap_or(1));
            vec![kv; layers as usize]
        });
    let ff_layers = artifact
        .planner
        .feed_forward_layers
        .clone()
        .unwrap_or_else(|| vec![0; layers as usize]);
    let ssm = artifact.planner.ssm_inner_size.unwrap_or(0) > 0;
    let mut kv = 0u64;
    let mut rs = 0u64;
    for il in 0..layers as usize {
        let kv_heads = *kv_layers.get(il).unwrap_or(&0) as u64;
        let n_ff = *ff_layers.get(il).unwrap_or(&0);
        if kv_heads > 0 {
            kv = kv.saturating_add(n_ctx_eff * kv_heads * (head_dim_k + head_dim_v) * 2);
        } else if n_ff == 0 && ssm {
            let d_conv = artifact.planner.ssm_conv_kernel.unwrap_or(0) as u64;
            let d_inner = artifact.planner.ssm_inner_size.unwrap_or(0) as u64;
            let d_state = artifact.planner.ssm_state_size.unwrap_or(0) as u64;
            let n_group = artifact.planner.ssm_group_count.unwrap_or(0) as u64;
            let n_embd_r = d_conv.saturating_sub(1) * (d_inner + 2 * n_group * d_state);
            let n_embd_s = d_state * d_inner;
            rs = rs.saturating_add(n_parallel.max(1) as u64 * (n_embd_r + n_embd_s) * 4);
        }
    }
    kv.saturating_add(rs)
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
