//! Beacon HTTP server — Ollama-compatible REST API.
//!
//! Exposes `/api/generate`, `/api/chat`, `/api/tags`, and `/api/pull`
//! endpoints that match the Ollama wire format, allowing clients like
//! Open `WebUI` to connect to Beacon without modification.

pub mod routes;
pub mod types;

use axum::routing::{get, post};
use tower_http::cors::{Any, CorsLayer};

/// Build the Axum router with all Ollama-compatible endpoints and CORS
/// middleware.
///
/// CORS is permissive (`Any` origin) to match Ollama's behaviour — local
/// inference servers are typically accessed from browser-based UIs on
/// `localhost`.
pub fn create_router() -> axum::Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    axum::Router::new()
        .route("/api/generate", post(routes::generate))
        .route("/api/chat", post(routes::chat))
        .route("/api/tags", get(routes::tags))
        .route("/api/pull", post(routes::pull))
        .layer(cors)
}
