//! Axum route handlers implementing the Ollama-compatible REST API.
//!
//! For v0.1 these are structural handlers that return proper Ollama-format
//! JSON but do not yet perform real inference. Model loading and the
//! inference loop will be wired in once the full engine integration is
//! available.

use std::fs;
use std::path::PathBuf;
use std::time::SystemTime;

use axum::http::StatusCode;
use axum::Json;

use crate::types::{
    ChatMessage, ChatRequest, ChatResponse, GenerateRequest, GenerateResponse, ModelInfo,
    PullRequest, PullResponse, TagsResponse,
};

/// Default Beacon model cache directory (`~/.beacon/models/`).
fn models_dir() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".beacon").join("models"))
}

// ---------------------------------------------------------------------------
// POST /api/generate
// ---------------------------------------------------------------------------

/// Handle `POST /api/generate`.
///
/// In v0.1 this returns a single non-streaming response acknowledging the
/// request. Real inference will be wired once the engine integration is
/// complete.
// TODO: stream NDJSON chunks via `axum::body::Body::from_stream` once the
// scheduler is integrated.
pub async fn generate(
    Json(req): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, StatusCode> {
    // v0.1 stub: echo back the model name with a placeholder response.
    Ok(Json(GenerateResponse {
        model: req.model,
        response: String::from(
            "Beacon inference engine is not yet wired. \
             This is a structural v0.1 stub response.",
        ),
        done: true,
        total_duration: Some(0),
        eval_count: Some(0),
    }))
}

// ---------------------------------------------------------------------------
// POST /api/chat
// ---------------------------------------------------------------------------

/// Handle `POST /api/chat`.
///
/// Same as `/api/generate` but with the chat message format.
// TODO: stream NDJSON chunks once the scheduler is integrated.
pub async fn chat(Json(req): Json<ChatRequest>) -> Result<Json<ChatResponse>, StatusCode> {
    // v0.1 stub: return a placeholder assistant message.
    Ok(Json(ChatResponse {
        model: req.model,
        message: ChatMessage {
            role: String::from("assistant"),
            content: String::from(
                "Beacon inference engine is not yet wired. \
                 This is a structural v0.1 stub response.",
            ),
        },
        done: true,
        total_duration: Some(0),
    }))
}

// ---------------------------------------------------------------------------
// GET /api/tags
// ---------------------------------------------------------------------------

/// Handle `GET /api/tags`.
///
/// Lists models found under `~/.beacon/models/`. Each subdirectory is
/// treated as a model name.
pub async fn tags() -> Result<Json<TagsResponse>, StatusCode> {
    let Some(models_path) = models_dir() else {
        // No home directory — return empty list rather than error.
        return Ok(Json(TagsResponse { models: Vec::new() }));
    };

    let mut models = Vec::new();

    if let Ok(entries) = fs::read_dir(&models_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }

            let name = entry.file_name().to_string_lossy().into_owned();
            let size = dir_size(&path);
            let modified_at = entry
                .metadata()
                .ok()
                .and_then(|m| m.modified().ok())
                .and_then(|t| {
                    t.duration_since(SystemTime::UNIX_EPOCH)
                        .ok()
                        .map(|d| format_unix_timestamp(d.as_secs()))
                })
                .unwrap_or_default();

            models.push(ModelInfo {
                name,
                size,
                modified_at,
            });
        }
    }

    Ok(Json(TagsResponse { models }))
}

// ---------------------------------------------------------------------------
// POST /api/pull
// ---------------------------------------------------------------------------

/// Handle `POST /api/pull`.
///
/// Model pulling is not yet implemented (requires `beacon-registry`).
pub async fn pull(Json(_req): Json<PullRequest>) -> Result<Json<PullResponse>, StatusCode> {
    Ok(Json(PullResponse {
        status: String::from("pull not yet implemented"),
    }))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Recursively compute total size of a directory in bytes.
fn dir_size(path: &std::path::Path) -> u64 {
    let mut total: u64 = 0;
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                total += dir_size(&p);
            } else if let Ok(meta) = entry.metadata() {
                total += meta.len();
            }
        }
    }
    total
}

/// Format a Unix timestamp (seconds since epoch) as an ISO-8601-ish string.
/// We avoid pulling in `chrono` for this single use case.
fn format_unix_timestamp(secs: u64) -> String {
    // Very simple: seconds since epoch → UTC date-time string.
    // This is intentionally approximate (no leap-second handling) and is
    // sufficient for the Ollama `modified_at` field.
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Days since Unix epoch → year/month/day (proleptic Gregorian, UTC).
    let (year, month, day) = days_to_ymd(days);
    format!("{year:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
}

/// Convert days since Unix epoch (1970-01-01) to (year, month, day).
fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    // Algorithm from Howard Hinnant's `chrono`-compatible date math.
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}
