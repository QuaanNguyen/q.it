use crate::bind;
use crate::config::Config;

pub fn run() -> i32 {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("qit_runtime=info".parse().unwrap()),
        )
        .init();
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            eprintln!("{error}");
            return 1;
        }
    };
    runtime.block_on(async {
        let config = match Config::from_env() {
            Ok(config) => config,
            Err(error) => {
                eprintln!("{error}");
                return 1;
            }
        };
        match bind(config).await {
            Ok(listening) => {
                println!("q.it listening on {}", listening.base_url());
                tokio::signal::ctrl_c().await.ok();
                listening.shutdown().await;
                0
            }
            Err(error) => {
                eprintln!("{error}");
                1
            }
        }
    })
}
