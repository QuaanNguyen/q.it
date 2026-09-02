use qit_runtime::bind;
use qit_runtime::config::Config;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("qit_runtime=info".parse().unwrap()),
        )
        .init();
    let config = match Config::from_env() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    match bind(config).await {
        Ok(listening) => {
            println!("q.it listening on {}", listening.base_url());
            tokio::signal::ctrl_c().await.ok();
            listening.shutdown().await;
        }
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}
