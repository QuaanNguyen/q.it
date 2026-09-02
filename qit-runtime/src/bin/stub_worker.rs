use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::stream::{self, StreamExt};
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
    let mut health_warmup_ms = 0u64;
    let mut stream = StreamShape::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--port" => port = args.next().and_then(|v| v.parse().ok()).unwrap_or(0),
            "--crash" => crash = true,
            "--delay-ms" => delay_ms = args.next().and_then(|v| v.parse().ok()).unwrap_or(0),
            "--health-warmup-ms" => {
                health_warmup_ms = args.next().and_then(|v| v.parse().ok()).unwrap_or(0)
            }
            "--token-delay-ms" => {
                stream.token_delay_ms = args.next().and_then(|v| v.parse().ok()).unwrap_or(0)
            }
            "--n-tokens" => {
                stream.n_tokens = args.next().and_then(|v| v.parse().ok()).unwrap_or(2)
            }
            "--echo-usage" => stream.echo_usage = true,
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
    println!("stub worker pid {}", std::process::id());
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let listener = TcpListener::bind(addr).await.expect("bind stub worker");
    let ready_at = std::time::Instant::now() + Duration::from_millis(health_warmup_ms);
    let app = Router::new()
        .route(
            "/health",
            get(move || {
                let ready_at = ready_at;
                async move {
                    if std::time::Instant::now() < ready_at {
                        (StatusCode::SERVICE_UNAVAILABLE, "loading")
                    } else {
                        (StatusCode::OK, "ok")
                    }
                }
            }),
        )
        .route(
            "/v1/chat/completions",
            post(move |req| chat(stream, req)),
        );
    axum::serve(listener, app).await.expect("stub worker");
}

#[derive(Clone, Copy)]
struct StreamShape {
    token_delay_ms: u64,
    n_tokens: usize,
    echo_usage: bool,
}

impl Default for StreamShape {
    fn default() -> Self {
        Self {
            token_delay_ms: 0,
            n_tokens: 2,
            echo_usage: false,
        }
    }
}

const FAKE_TOKENS_PER_MESSAGE: u64 = 100;

fn usage_chunk(req: &Value, n_tokens: usize) -> Value {
    let n_messages = req["messages"].as_array().map(|m| m.len()).unwrap_or(0) as u64;
    let max_tokens = req["max_tokens"].as_u64().unwrap_or(0);
    json!({
        "choices": [],
        "usage": {
            "prompt_tokens": n_messages * FAKE_TOKENS_PER_MESSAGE + max_tokens,
            "completion_tokens": n_tokens,
            "total_tokens": n_messages * FAKE_TOKENS_PER_MESSAGE + max_tokens + n_tokens as u64
        }
    })
}

fn token_text(index: usize) -> String {
    match index {
        0 => "hello".into(),
        1 => " world".into(),
        n => format!(" t{n}"),
    }
}

async fn chat(
    shape: StreamShape,
    Json(req): Json<Value>,
) -> Sse<impl futures_util::Stream<Item = Result<Event, Infallible>>> {
    let delay = Duration::from_millis(shape.token_delay_ms);
    let tokens = stream::iter(0..shape.n_tokens).then(move |i| async move {
        if i > 0 && !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
        let payload = json!({
            "choices": [{"delta": {"content": token_text(i)}}]
        });
        Ok(Event::default().data(payload.to_string()))
    });
    let usage = if shape.echo_usage {
        vec![Ok(Event::default().data(usage_chunk(&req, shape.n_tokens).to_string()))]
    } else {
        vec![]
    };
    let done = stream::once(async { Ok(Event::default().data("[DONE]")) });
    Sse::new(tokens.chain(stream::iter(usage)).chain(done)).keep_alive(KeepAlive::default())
}
