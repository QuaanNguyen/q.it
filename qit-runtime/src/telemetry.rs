use std::path::PathBuf;
use std::time::Duration;

use tokio::sync::oneshot;

use crate::model::{HostInfo, TelemetrySummary};

pub struct TelemetrySampler {
    stop: oneshot::Sender<()>,
    join: tokio::task::JoinHandle<TelemetrySummary>,
}

impl TelemetrySampler {
    pub fn start(target_pid: Option<u32>) -> Self {
        let (stop, mut stopped) = oneshot::channel();
        let join = tokio::spawn(async move {
            let mut accumulator = Accumulator::default();
            let mut previous = SystemSnapshot::read(target_pid);
            accumulator.observe(previous.as_ref(), None);
            let mut interval = tokio::time::interval(Duration::from_millis(100));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        let current = SystemSnapshot::read(target_pid);
                        accumulator.observe(current.as_ref(), previous.as_ref());
                        previous = current;
                    }
                    _ = &mut stopped => {
                        let current = SystemSnapshot::read(target_pid);
                        accumulator.observe(current.as_ref(), previous.as_ref());
                        break;
                    }
                }
            }
            accumulator.finish()
        });
        Self { stop, join }
    }

    pub async fn finish(self) -> TelemetrySummary {
        let _ = self.stop.send(());
        self.join.await.unwrap_or_default()
    }
}

impl HostInfo {
    pub fn detect() -> Self {
        let cpu = cpu_name();
        let device_model = device_model();
        let raspberry_pi = device_model
            .as_deref()
            .map(|model| model.to_ascii_lowercase().contains("raspberry pi"))
            .unwrap_or(false);
        Self {
            host_name: host_name(),
            operating_system: std::env::consts::OS.to_string(),
            kernel: std::fs::read_to_string("/proc/sys/kernel/osrelease")
                .ok()
                .map(|value| value.trim().to_string()),
            architecture: std::env::consts::ARCH.to_string(),
            device_model,
            cpu,
            logical_cpu_count: std::thread::available_parallelism()
                .ok()
                .map(|count| count.get() as u32),
            total_memory_bytes: memory_values().map(|(total, _)| total),
            raspberry_pi,
            qit_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

#[derive(Clone, Copy)]
struct SystemSnapshot {
    total_cpu_ticks: u64,
    idle_cpu_ticks: u64,
    memory_used_bytes: Option<u64>,
    process_rss_bytes: Option<u64>,
}

impl SystemSnapshot {
    fn read(target_pid: Option<u32>) -> Option<Self> {
        let (total_cpu_ticks, idle_cpu_ticks) = cpu_ticks()?;
        Some(Self {
            total_cpu_ticks,
            idle_cpu_ticks,
            memory_used_bytes: memory_values()
                .map(|(total, available)| total.saturating_sub(available)),
            process_rss_bytes: target_pid.and_then(process_rss),
        })
    }
}

#[derive(Default)]
struct Accumulator {
    cpu: Vec<f64>,
    memory_peak: Option<u64>,
    process_peak: Option<u64>,
}

impl Accumulator {
    fn observe(&mut self, current: Option<&SystemSnapshot>, previous: Option<&SystemSnapshot>) {
        let Some(current) = current else {
            return;
        };
        if let Some(previous) = previous {
            let total = current
                .total_cpu_ticks
                .saturating_sub(previous.total_cpu_ticks);
            let idle = current
                .idle_cpu_ticks
                .saturating_sub(previous.idle_cpu_ticks);
            if total > 0 {
                self.cpu
                    .push((total.saturating_sub(idle) as f64 / total as f64) * 100.0);
            }
        }
        if let Some(value) = current.memory_used_bytes {
            self.memory_peak = Some(self.memory_peak.unwrap_or(0).max(value));
        }
        if let Some(value) = current.process_rss_bytes {
            self.process_peak = Some(self.process_peak.unwrap_or(0).max(value));
        }
    }

    fn finish(self) -> TelemetrySummary {
        let host_cpu_percent_mean =
            (!self.cpu.is_empty()).then(|| self.cpu.iter().sum::<f64>() / self.cpu.len() as f64);
        let host_cpu_percent_peak = self.cpu.into_iter().reduce(f64::max);
        TelemetrySummary {
            host_cpu_percent_mean,
            host_cpu_percent_peak,
            host_memory_used_bytes_peak: self.memory_peak,
            process_rss_bytes_peak: self.process_peak,
        }
    }
}

#[cfg(target_os = "linux")]
fn cpu_ticks() -> Option<(u64, u64)> {
    let stat = std::fs::read_to_string("/proc/stat").ok()?;
    let values = stat.lines().next()?.split_whitespace().collect::<Vec<_>>();
    if values.first().copied() != Some("cpu") || values.len() < 5 {
        return None;
    }
    let ticks = values[1..]
        .iter()
        .map(|value| value.parse::<u64>().ok())
        .collect::<Option<Vec<_>>>()?;
    let total = ticks.iter().sum();
    let idle = ticks.get(3).copied().unwrap_or(0) + ticks.get(4).copied().unwrap_or(0);
    Some((total, idle))
}

#[cfg(not(target_os = "linux"))]
fn cpu_ticks() -> Option<(u64, u64)> {
    None
}

#[cfg(target_os = "linux")]
fn memory_values() -> Option<(u64, u64)> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut total = None;
    let mut available = None;
    for line in meminfo.lines() {
        let (name, rest) = line.split_once(':')?;
        let value = rest.split_whitespace().next()?.parse::<u64>().ok()? * 1024;
        match name {
            "MemTotal" => total = Some(value),
            "MemAvailable" => available = Some(value),
            _ => {}
        }
        if total.is_some() && available.is_some() {
            break;
        }
    }
    Some((total?, available?))
}

#[cfg(not(target_os = "linux"))]
fn memory_values() -> Option<(u64, u64)> {
    None
}

#[cfg(target_os = "linux")]
fn process_rss(pid: u32) -> Option<u64> {
    let status = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    status.lines().find_map(|line| {
        let value = line.strip_prefix("VmRSS:")?;
        value
            .split_whitespace()
            .next()?
            .parse::<u64>()
            .ok()
            .map(|kilobytes| kilobytes * 1024)
    })
}

#[cfg(not(target_os = "linux"))]
fn process_rss(_pid: u32) -> Option<u64> {
    None
}

#[cfg(target_os = "linux")]
fn cpu_name() -> Option<String> {
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    for key in ["Model", "model name", "Hardware"] {
        if let Some(value) = cpuinfo.lines().find_map(|line| {
            let (name, value) = line.split_once(':')?;
            (name.trim() == key).then(|| value.trim().to_string())
        }) {
            return Some(value);
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
fn cpu_name() -> Option<String> {
    None
}

fn host_name() -> String {
    std::env::var("HOSTNAME")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            std::fs::read_to_string("/etc/hostname")
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
        })
        .unwrap_or_else(|| "unknown-host".into())
}

fn device_model() -> Option<String> {
    [
        PathBuf::from("/sys/firmware/devicetree/base/model"),
        PathBuf::from("/proc/device-tree/model"),
    ]
    .into_iter()
    .find_map(|path| std::fs::read_to_string(path).ok())
    .map(|value| value.trim_matches(char::from(0)).trim().to_string())
}
