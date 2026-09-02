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
        chip: sysctl_string("machdep.cpu.brand_string").unwrap_or_else(|| "Apple Silicon".into()),
        unified_memory_bytes: sysctl_u64("hw.memsize").unwrap_or(0),
        metal_recommended_working_set_bytes: metal_working_set(),
        memory_pressure: vm_pressure_level(),
        free_ram_bytes: reclaimable_ram_bytes(),
    }
}

#[cfg(target_os = "macos")]
fn metal_working_set() -> Option<u64> {
    let device = metal::Device::system_default()?;
    Some(device.recommended_max_working_set_size())
}

#[cfg(target_os = "macos")]
fn sysctl_string(name: &str) -> Option<String> {
    let mut len = 0usize;
    let cname = std::ffi::CString::new(name).ok()?;
    unsafe {
        if libc::sysctlbyname(cname.as_ptr(), std::ptr::null_mut(), &mut len, std::ptr::null_mut(), 0)
            != 0
        {
            return None;
        }
        let mut buf = vec![0u8; len];
        if libc::sysctlbyname(
            cname.as_ptr(),
            buf.as_mut_ptr() as *mut _,
            &mut len,
            std::ptr::null_mut(),
            0,
        ) != 0
        {
            return None;
        }
        if len > 0 && buf[len - 1] == 0 {
            buf.truncate(len - 1);
        }
        let s = String::from_utf8(buf).ok()?.trim().to_string();
        if s.is_empty() {
            None
        } else {
            Some(s)
        }
    }
}

#[cfg(target_os = "macos")]
fn sysctl_u64(name: &str) -> Option<u64> {
    let mut val: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    let cname = std::ffi::CString::new(name).ok()?;
    let rc = unsafe {
        libc::sysctlbyname(
            cname.as_ptr(),
            &mut val as *mut u64 as *mut _,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc == 0 {
        Some(val)
    } else {
        None
    }
}

#[cfg(target_os = "macos")]
fn vm_pressure_level() -> Option<String> {
    let mut val: i32 = 0;
    let mut len = std::mem::size_of::<i32>();
    let cname = std::ffi::CString::new("kern.memorystatus_vm_pressure_level").ok()?;
    let rc = unsafe {
        libc::sysctlbyname(
            cname.as_ptr(),
            &mut val as *mut i32 as *mut _,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 {
        return None;
    }
    match val {
        1 => Some("normal".into()),
        2 => Some("warn".into()),
        4 => Some("critical".into()),
        _ => None,
    }
}

#[cfg(target_os = "macos")]
#[allow(deprecated)]
fn reclaimable_ram_bytes() -> Option<u64> {
    let mut stats = unsafe { std::mem::zeroed::<libc::vm_statistics64>() };
    let mut count = libc::HOST_VM_INFO64_COUNT;
    let rc = unsafe {
        libc::host_statistics64(
            libc::mach_host_self(),
            libc::HOST_VM_INFO64,
            &mut stats as *mut _ as *mut _,
            &mut count,
        )
    };
    if rc != libc::KERN_SUCCESS {
        return None;
    }
    let page = sysctl_u64("hw.pagesize")?;
    let pages = u64::from(stats.free_count)
        + u64::from(stats.inactive_count)
        + u64::from(stats.purgeable_count);
    Some(pages * page)
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
