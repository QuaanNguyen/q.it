use std::io;
use std::net::SocketAddr;
use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("port {port} is already in use ({addr})")]
    PortInUse { port: u16, addr: SocketAddr },
    #[error("listen {addr}: {source}")]
    Listen {
        addr: SocketAddr,
        #[source]
        source: io::Error,
    },
    #[error("qit home {path}: {source}")]
    Home {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error(transparent)]
    Sqlite(#[from] rusqlite::Error),
    #[error("{0}")]
    Message(String),
}

impl Error {
    pub fn from_bind(addr: SocketAddr, err: io::Error) -> Self {
        if err.kind() == io::ErrorKind::AddrInUse {
            Self::PortInUse {
                port: addr.port(),
                addr,
            }
        } else {
            Self::Listen { addr, source: err }
        }
    }
}
