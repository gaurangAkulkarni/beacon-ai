//! Ollama-compatible JSON request/response types.
//!
//! These types match the Ollama REST API wire format so that clients like
//! Open `WebUI` can connect to Beacon without modification.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// POST /api/generate
// ---------------------------------------------------------------------------

/// Request body for `POST /api/generate`.
#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    /// Model name (e.g. `"qwen2.5-3b"`).
    pub model: String,

    /// The prompt to generate a completion for.
    pub prompt: String,

    /// Whether to stream the response as NDJSON. Defaults to `true` in
    /// Ollama; we accept it but currently always return a single response.
    #[serde(default)]
    pub stream: Option<bool>,

    /// Sampling / generation options.
    #[serde(default)]
    pub options: Option<GenerateOptions>,
}

/// Optional generation parameters matching Ollama's `options` object.
#[derive(Debug, Default, Deserialize)]
pub struct GenerateOptions {
    /// Sampling temperature (0.0 = greedy).
    pub temperature: Option<f32>,
    /// Top-k filtering.
    pub top_k: Option<usize>,
    /// Nucleus (top-p) filtering.
    pub top_p: Option<f32>,
    /// Maximum number of tokens to predict.
    pub num_predict: Option<usize>,
    /// Repetition penalty.
    pub repeat_penalty: Option<f32>,
    /// Stop sequences.
    pub stop: Option<Vec<String>>,
}

/// A single chunk in the `/api/generate` response stream.
#[derive(Debug, Serialize)]
pub struct GenerateResponse {
    /// Model name echoed back.
    pub model: String,
    /// Generated text (full response when `done` is true, partial when
    /// streaming).
    pub response: String,
    /// Whether generation is complete.
    pub done: bool,
    /// Total wall-clock duration in nanoseconds (present on final chunk).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    /// Number of tokens evaluated (present on final chunk).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u64>,
}

// ---------------------------------------------------------------------------
// POST /api/chat
// ---------------------------------------------------------------------------

/// Request body for `POST /api/chat`.
#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    /// Model name.
    pub model: String,
    /// Conversation history.
    pub messages: Vec<ChatMessage>,
    /// Whether to stream the response as NDJSON.
    #[serde(default)]
    pub stream: Option<bool>,
    /// Sampling / generation options.
    #[serde(default)]
    pub options: Option<GenerateOptions>,
}

/// A single message in a chat conversation.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    /// Role: `"system"`, `"user"`, or `"assistant"`.
    pub role: String,
    /// Message content.
    pub content: String,
}

/// A single chunk in the `/api/chat` response stream.
#[derive(Debug, Serialize)]
pub struct ChatResponse {
    /// Model name echoed back.
    pub model: String,
    /// The assistant's reply message.
    pub message: ChatMessage,
    /// Whether generation is complete.
    pub done: bool,
    /// Total wall-clock duration in nanoseconds (present on final chunk).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
}

// ---------------------------------------------------------------------------
// GET /api/tags
// ---------------------------------------------------------------------------

/// Response body for `GET /api/tags`.
#[derive(Debug, Serialize)]
pub struct TagsResponse {
    /// List of locally available models.
    pub models: Vec<ModelInfo>,
}

/// Summary information about a locally cached model.
#[derive(Debug, Serialize)]
pub struct ModelInfo {
    /// Model name (directory name under `~/.beacon/models/`).
    pub name: String,
    /// Total size on disk in bytes.
    pub size: u64,
    /// ISO-8601 timestamp of last modification.
    pub modified_at: String,
}

// ---------------------------------------------------------------------------
// POST /api/pull
// ---------------------------------------------------------------------------

/// Request body for `POST /api/pull`.
#[derive(Debug, Deserialize)]
pub struct PullRequest {
    /// Model name to pull (e.g. `"qwen2.5-3b"`).
    pub name: String,
}

/// Response body for `POST /api/pull`.
#[derive(Debug, Serialize)]
pub struct PullResponse {
    /// Human-readable status message.
    pub status: String,
}
