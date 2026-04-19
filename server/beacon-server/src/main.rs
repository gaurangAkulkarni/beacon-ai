//! Beacon HTTP server entry point.
//!
//! Listens on `0.0.0.0:11434` (configurable via `BEACON_PORT` env var),
//! serving the Ollama-compatible REST API.

#[tokio::main]
async fn main() {
    let port = std::env::var("BEACON_PORT").unwrap_or_else(|_| String::from("11434"));
    let addr = format!("0.0.0.0:{port}");
    println!("Beacon server listening on {addr}");

    let app = beacon_server::create_router();
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("failed to bind TCP listener");

    axum::serve(listener, app)
        .await
        .expect("server exited with error");
}
