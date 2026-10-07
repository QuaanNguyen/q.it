use std::net::SocketAddr;
use std::path::PathBuf;

pub const DEFAULT_HOST: &str = "127.0.0.1";
pub const DEFAULT_PORT: u16 = 2471;

#[derive(Clone, Debug)]
pub struct Config {
    pub home: PathBuf,
    pub listen: SocketAddr,
}

impl Config {
    pub fn from_env() -> Result<Self, String> {
        let host = std::env::var("QIT_HOST").unwrap_or_else(|_| DEFAULT_HOST.into());
        let port = std::env::var("QIT_PORT")
            .ok()
            .map(|value| {
                value
                    .parse::<u16>()
                    .map_err(|error| format!("invalid QIT_PORT '{value}': {error}"))
            })
            .transpose()?
            .unwrap_or(DEFAULT_PORT);
        Ok(Self {
            home: home_from_env(),
            listen: socket_addr(&host, port)?,
        })
    }

    pub fn with_dashboard_overrides(
        mut self,
        host: Option<&str>,
        port: Option<u16>,
    ) -> Result<Self, String> {
        let host = host
            .map(str::to_string)
            .unwrap_or_else(|| self.listen.ip().to_string());
        self.listen = socket_addr(&host, port.unwrap_or(self.listen.port()))?;
        Ok(self)
    }
}

pub fn home_from_env() -> PathBuf {
    if let Some(home) = std::env::var_os("QIT_HOME") {
        return PathBuf::from(home);
    }
    if let Some(data) = std::env::var_os("XDG_DATA_HOME") {
        return PathBuf::from(data).join("qit");
    }
    if let Some(home) = std::env::var_os("HOME") {
        let home = PathBuf::from(home);
        if cfg!(target_os = "macos") {
            return home
                .join("Library")
                .join("Application Support")
                .join("q.it");
        }
        return home.join(".local").join("share").join("qit");
    }
    PathBuf::from("qit-data")
}

fn socket_addr(host: &str, port: u16) -> Result<SocketAddr, String> {
    format!("{host}:{port}")
        .parse()
        .map_err(|error| format!("invalid dashboard address {host}:{port}: {error}"))
}
