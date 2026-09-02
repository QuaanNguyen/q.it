pub mod config;
pub mod error;
pub mod estimate;
pub mod gguf;
pub mod http;
pub mod paths;
pub mod probe;
pub mod scan;
pub mod spa;
pub mod store;
pub mod supervisor;

use std::net::SocketAddr;
use std::sync::Arc;

use tokio::net::TcpListener;
use tokio::sync::{oneshot, Mutex};

use crate::config::Config;
use crate::error::Error;
use crate::http::{rescan, router, AppState};
use crate::paths::Paths;
use crate::store::Store;
use crate::supervisor::Supervisor;

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
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        if let Some(join) = self.join.take() {
            let _ = join.await;
        }
    }
}

pub async fn bind(config: Config) -> Result<Listening, Error> {
    let paths = Paths::new(config.home.clone(), config.models_dir.clone());
    paths.ensure().map_err(|source| Error::Home {
        path: paths.home.clone(),
        source,
    })?;
    let store = Store::open(&paths.db_path)?;
    store.reset_sessions_on_restart()?;
    let session_rows = store.sessions()?;
    let supervisor = Arc::new(Supervisor::new(config.worker_launcher.clone()));
    supervisor.hydrate(session_rows).await;
    let state = AppState {
        paths,
        store: Arc::new(Mutex::new(store)),
        probe: config.probe.clone(),
        os_reserve_override: config.os_reserve_bytes,
        worker_path: config.worker_path.clone(),
        supervisor,
        what_ifs: Arc::new(Mutex::new(Vec::new())),
    };
    rescan(&state)
        .await
        .map_err(|e| Error::Message(format!("{e:?}")))?;
    let listener = TcpListener::bind(config.listen)
        .await
        .map_err(|err| Error::from_bind(config.listen, err))?;
    let addr = listener
        .local_addr()
        .map_err(|err| Error::from_bind(config.listen, err))?;
    let app = router(state);
    let (tx, rx) = oneshot::channel::<()>();
    let join = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = rx.await;
            })
            .await
    });
    Ok(Listening {
        addr,
        shutdown: Some(tx),
        join: Some(join),
    })
}
