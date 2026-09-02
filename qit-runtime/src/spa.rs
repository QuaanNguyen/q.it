pub fn index_html() -> String {
    include_str!("fallback.html").to_string()
}
