pub fn index_html() -> String {
    asset("index.html")
        .and_then(|(_, bytes)| std::str::from_utf8(bytes).ok())
        .unwrap_or(include_str!("fallback.html"))
        .to_string()
}

include!(concat!(env!("OUT_DIR"), "/web_assets.rs"));

pub fn asset(path: &str) -> Option<(&'static str, &'static [u8])> {
    embedded_asset(path).map(|bytes| (mime_for(path), bytes))
}

fn mime_for(path: &str) -> &'static str {
    match std::path::Path::new(path)
        .extension()
        .and_then(|value| value.to_str())
    {
        Some("js") => "text/javascript",
        Some("css") => "text/css",
        Some("html") => "text/html; charset=utf-8",
        Some("svg") => "image/svg+xml",
        Some("json") => "application/json",
        _ => "application/octet-stream",
    }
}
