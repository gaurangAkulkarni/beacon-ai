//! Beacon HTTP server — Ollama-compatible and `OpenAI`-compatible REST APIs.
//!
//! Exposes Ollama endpoints (`/api/generate`, `/api/chat`, `/api/tags`,
//! `/api/pull`) and `OpenAI` endpoints (`/v1/chat/completions`, `/v1/models`)
//! that allow clients like Open `WebUI` and `OpenAI` SDKs to connect to Beacon
//! without modification.

pub mod routes;
pub mod types;

use std::sync::{Arc, Mutex};

use axum::routing::{get, post};
use tower_http::cors::{Any, CorsLayer};

/// Shared application state, available to all route handlers via
/// `axum::extract::State`.
///
/// The engine is behind a `Mutex` because `Engine::forward` mutates the KV
/// cache. The tokenizer is read-only and safe to share.
#[derive(Debug)]
pub struct AppState {
    /// The inference engine (MLX backend). Mutex-protected because
    /// `forward()` mutates the KV cache.
    pub engine: Mutex<beacon_core::Engine<beacon_core::MlxBackend>>,
    /// Tokenizer loaded from `tokenizer.json`.
    pub tokenizer: beacon_tokenizer::BeaconTokenizer,
    /// Human-readable model name for API responses.
    pub model_name: String,
}

/// Build the Axum router with all endpoints and CORS middleware.
///
/// CORS is permissive (`Any` origin) to match Ollama's behaviour — local
/// inference servers are typically accessed from browser-based UIs on
/// `localhost`.
pub fn create_router(state: Arc<AppState>) -> axum::Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    axum::Router::new()
        // Ollama API
        .route("/api/generate", post(routes::generate))
        .route("/api/chat", post(routes::chat))
        .route("/api/tags", get(routes::tags))
        .route("/api/pull", post(routes::pull))
        // `OpenAI` API
        .route("/v1/chat/completions", post(routes::openai_chat))
        .route("/v1/models", get(routes::openai_models))
        .layer(cors)
        .with_state(state)
}
