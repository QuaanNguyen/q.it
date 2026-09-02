use serde::Serialize;

pub trait HardwareProbe: Send + Sync {
    fn probe(&self) -> HardwareSnapshot;
}

#[derive(Clone, Debug, Serialize)]
pub struct HardwareSnapshot {
    pub device_class: String,
    pub chip: String,
    pub unified_memory_bytes: u64,
    pub metal_recommended_working_set_bytes: Option<u64>,
    pub memory_pressure: Option<String>,
    pub free_ram_bytes: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct FixedProbe {
    pub snapshot: HardwareSnapshot,
}

impl HardwareProbe for FixedProbe {
    fn probe(&self) -> HardwareSnapshot {
        self.snapshot.clone()
    }
}

pub struct SystemProbe;

impl HardwareProbe for SystemProbe {
    fn probe(&self) -> HardwareSnapshot {
        system_snapshot()
    }
}

fn system_snapshot() -> HardwareSnapshot {
    #[cfg(target_os = "macos")]
    {
        macos_snapshot()
    }
    #[cfg(not(target_os = "macos"))]
    {
        HardwareSnapshot {
            device_class: "unknown".into(),
            chip: "unknown".into(),
            unified_memory_bytes: 0,
            metal_recommended_working_set_bytes: None,
            memory_pressure: None,
            free_ram_bytes: None,
        }
    }
}

#[cfg(target_os = "macos")]
fn macos_snapshot() -> HardwareSnapshot {
    HardwareSnapshot {
        device_class: "apple_silicon".into(),
        chip: sysctl_n("machdep.cpu.brand_string").unwrap_or_else(|| "Apple Silicon".into()),
        unified_memory_bytes: sysctl_n("hw.memsize")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0),
        metal_recommended_working_set_bytes: metal_working_set(),
        memory_pressure: None,
        free_ram_bytes: None,
    }
}

#[cfg(target_os = "macos")]
fn metal_working_set() -> Option<u64> {
    let device = metal::Device::system_default()?;
    Some(device.recommended_max_working_set_size())
}

#[cfg(target_os = "macos")]
fn sysctl_n(name: &str) -> Option<String> {
    let out = std::process::Command::new("sysctl")
        .args(["-n", name])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?;
    let s = s.trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

pub fn budget_bytes(snapshot: &HardwareSnapshot, os_reserve_bytes: u64) -> u64 {
    let after_reserve = snapshot
        .unified_memory_bytes
        .saturating_sub(os_reserve_bytes);
    match snapshot.metal_recommended_working_set_bytes {
        Some(cap) => after_reserve.min(cap),
        None => after_reserve,
    }
}

pub fn resolve_os_reserve(unified_memory_bytes: u64, override_bytes: Option<u64>) -> u64 {
    override_bytes.unwrap_or_else(|| {
        ((unified_memory_bytes as f64) * crate::config::DEFAULT_OS_RESERVE_FRACTION) as u64
    })
}
