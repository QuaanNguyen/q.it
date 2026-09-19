pub fn index_html() -> String {
    asset("index.html")
        .and_then(|(_, bytes)| std::str::from_utf8(bytes).ok())
        .unwrap_or_else(|| include_str!("fallback.html"))
        .to_string()
}

include!(concat!(env!("OUT_DIR"), "/web_assets.rs"));

pub fn asset(path: &str) -> Option<(&'static str, &'static [u8])> {
    embedded_asset(path)
}
