pub mod command;
pub mod config;
pub mod dashboard;
pub mod model;
pub mod packs;
pub mod paths;
pub mod provider;
pub mod runner;
pub mod spa;
pub mod store;
pub mod telemetry;

use std::net::SocketAddr;
use std::sync::Arc;

use tokio::net::TcpListener;
use tokio::sync::{oneshot, Mutex};

use crate::config::Config;
use crate::dashboard::{router, DashboardState};
use crate::model::HostInfo;
use crate::packs::PackCatalog;
use crate::paths::Paths;
use crate::store::Store;

pub struct Listening {
    pub addr: SocketAddr,
    shutdown: Option<oneshot::Sender<()>>,
    join: Option<tokio::task::JoinHandle<Result<(), std::io::Error>>>,
}

impl Listening {
    pub fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }

    pub async fn shutdown(mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        if let Some(join) = self.join.take() {
            let _ = join.await;
        }
    }
}

pub async fn bind(config: Config) -> Result<Listening, String> {
    let paths = Paths::new(config.home);
    paths.ensure()?;
    let store = Store::open(&paths.database)?;
    let catalog = PackCatalog::new(paths.packs);
    catalog.benchmarks()?;
    let state = DashboardState {
        store: Arc::new(Mutex::new(store)),
        catalog,
        host: HostInfo::detect(),
    };
    let listener = TcpListener::bind(config.listen)
        .await
        .map_err(|error| format!("bind dashboard to {}: {error}", config.listen))?;
    let addr = listener
        .local_addr()
        .map_err(|error| format!("read dashboard address: {error}"))?;
    let (shutdown, stopped) = oneshot::channel();
    let join = tokio::spawn(async move {
        axum::serve(listener, router(state))
            .with_graceful_shutdown(async {
                let _ = stopped.await;
            })
            .await
    });
    Ok(Listening {
        addr,
        shutdown: Some(shutdown),
        join: Some(join),
    })
}
