use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};

#[test]
fn staged_qit_serves_health_and_embedded_production_assets_outside_the_checkout() {
    let staging = tempfile::tempdir().unwrap();
    let binary = staging.path().join("qit");
    std::fs::copy(env!("CARGO_BIN_EXE_qit"), &binary).unwrap();
    let home = staging.path().join("state");
    let models = staging.path().join("models");
    let mut child = Command::new(&binary)
        .current_dir(staging.path())
        .env("QIT_HOME", &home)
        .env("QIT_MODELS_DIR", &models)
        .env("QIT_PORT", "0")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let stdout = child.stdout.take().unwrap();
    let mut lines = BufReader::new(stdout).lines();
    let line = lines.next().unwrap().unwrap();
    let base_url = line.strip_prefix("q.it listening on ").unwrap();

    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        let health: serde_json::Value = reqwest::get(format!("{base_url}/api/health"))
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(health["ok"], true);
        let html = reqwest::get(base_url).await.unwrap().text().await.unwrap();
        assert!(html.contains("<div id=\"root\"></div>"), "{html}");
        let asset_path = html
            .split("src=\"")
            .nth(1)
            .and_then(|tail| tail.split('\"').next())
            .unwrap();
        let asset = reqwest::get(format!("{base_url}{asset_path}"))
            .await
            .unwrap();
        assert!(asset.status().is_success());
        assert!(asset.bytes().await.unwrap().len() > 100_000);
    });

    child.kill().unwrap();
    child.wait().unwrap();
    assert!(home.join("runtime.db").is_file());
}
