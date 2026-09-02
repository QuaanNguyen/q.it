use axum::response::sse::{Event, KeepAlive, Sse};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::stream;
use serde_json::{json, Value};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::time::Duration;
use tokio::net::TcpListener;

#[tokio::main]
async fn main() {
    let mut port = 0u16;
    let mut crash = false;
    let mut delay_ms = 0u64;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--port" => port = args.next().and_then(|v| v.parse().ok()).unwrap_or(0),
            "--crash" => crash = true,
            "--delay-ms" => delay_ms = args.next().and_then(|v| v.parse().ok()).unwrap_or(0),
            "-m" | "-c" | "-ngl" | "--parallel" | "--host" => {
                let _ = args.next();
            }
            _ => {}
        }
    }
    if crash {
        std::process::exit(1);
    }
    if delay_ms > 0 {
        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
    }
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let listener = TcpListener::bind(addr).await.expect("bind stub worker");
    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/chat/completions", post(chat));
    axum::serve(listener, app).await.expect("stub worker");
}

async fn chat(
    Json(_req): Json<Value>,
) -> Sse<impl futures_util::Stream<Item = Result<Event, Infallible>>> {
    let frames = ["hello", " world"];
    let events = frames.into_iter().map(|tok| {
        let payload = json!({
            "choices": [{"delta": {"content": tok}}]
        });
        Ok(Event::default().data(payload.to_string()))
    });
    let done = std::iter::once(Ok(Event::default().data("[DONE]")));
    Sse::new(stream::iter(events.chain(done))).keep_alive(KeepAlive::default())
}
