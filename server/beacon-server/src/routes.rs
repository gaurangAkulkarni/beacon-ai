//! Axum route handlers implementing the Ollama-compatible and
//! OpenAI-compatible REST APIs.
//!
//! All inference routes delegate to [`run_inference`], which locks the
//! engine, resets the KV cache, encodes the prompt, runs the forward pass
//! decode loop, and returns the generated text plus token count.

use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Instant, SystemTime};

use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use beacon_core::ComputeBackend as _;

use crate::types::{
    ChatMessage, ChatRequest, ChatResponse, GenerateRequest, GenerateResponse, ModelInfo,
    OpenAiChatRequest, OpenAiChatResponse, OpenAiChoice, OpenAiMessage, OpenAiModelInfo,
    OpenAiModelsResponse, OpenAiUsage, PullRequest, PullResponse, TagsResponse,
};
use crate::AppState;

/// Default Beacon model cache directory (`~/.beacon/models/`).
fn models_dir() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".beacon").join("models"))
}

// ===========================================================================
// Inference helper
// ===========================================================================

/// Result of a single inference run.
struct InferenceResult {
    /// The generated text.
    text: String,
    /// Number of generated tokens.
    generated_tokens: usize,
    /// Number of prompt tokens.
    prompt_tokens: usize,
    /// Whether generation stopped because of an EOS token (as opposed to
    /// hitting `max_tokens`).
    stopped_by_eos: bool,
}

/// Run inference: lock engine, reset cache, encode prompt, decode loop.
///
/// Returns the generated text and token counts, or an error string.
#[allow(clippy::cast_possible_truncation)]
fn run_inference(
    state: &AppState,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
) -> Result<InferenceResult, String> {
    let mut engine = state
        .engine
        .lock()
        .map_err(|e| format!("engine lock poisoned: {e}"))?;

    // Reset KV cache so each request starts fresh.
    engine.reset_cache();

    // Encode the prompt.
    let token_ids = state
        .tokenizer
        .encode(prompt, false)
        .map_err(|e| format!("tokenizer encode error: {e}"))?;

    if token_ids.is_empty() {
        return Err("prompt encoded to zero tokens".to_owned());
    }

    let vocab_size = engine.config.vocab_size;
    let eos_tokens = engine.config.eos_token_ids.clone();

    // Build sampling parameters.
    let params = beacon_scheduler::GenerationParams {
        max_tokens,
        temperature,
        top_k,
        top_p,
        stop_tokens: eos_tokens.clone(),
        ..beacon_scheduler::GenerationParams::default()
    };

    let mut rng = rand::rng();
    let prompt_token_count = token_ids.len();

    // Prefill: forward the entire prompt at once.
    let logits = engine
        .forward(&token_ids, 0)
        .map_err(|e| format!("prefill forward failed: {e}"))?;

    let all_logits = engine
        .backend_ref()
        .read_f32(&logits, token_ids.len() * vocab_size)
        .map_err(|e| format!("read prefill logits failed: {e}"))?;

    let last_row_start = (token_ids.len() - 1) * vocab_size;
    let mut logit_values = all_logits[last_row_start..last_row_start + vocab_size].to_vec();

    let mut prev_tokens = token_ids.clone();
    let mut position = token_ids.len();

    // Sample first generated token.
    let mut next_token =
        beacon_scheduler::sampling::sample(&mut logit_values, &prev_tokens, &params, &mut rng);

    let mut output = String::new();
    let mut generated_count: usize = 0;
    let mut stopped_by_eos = false;

    if eos_tokens.contains(&next_token) {
        stopped_by_eos = true;
        return Ok(InferenceResult {
            text: output,
            generated_tokens: generated_count,
            prompt_tokens: prompt_token_count,
            stopped_by_eos,
        });
    }

    if let Ok(text) = state.tokenizer.decode(&[next_token], true) {
        let _ = write!(output, "{text}");
    }
    generated_count += 1;

    // Auto-regressive decode loop.
    for _ in 1..max_tokens {
        let logits = engine
            .forward(&[next_token], position)
            .map_err(|e| format!("decode forward failed: {e}"))?;
        position += 1;

        let mut logit_vals = engine
            .backend_ref()
            .read_f32(&logits, vocab_size)
            .map_err(|e| format!("read decode logits failed: {e}"))?;

        prev_tokens.push(next_token);
        next_token =
            beacon_scheduler::sampling::sample(&mut logit_vals, &prev_tokens, &params, &mut rng);

        if eos_tokens.contains(&next_token) {
            stopped_by_eos = true;
            break;
        }

        if let Ok(text) = state.tokenizer.decode(&[next_token], true) {
            let _ = write!(output, "{text}");
        }
        generated_count += 1;
    }

    Ok(InferenceResult {
        text: output,
        generated_tokens: generated_count,
        prompt_tokens: prompt_token_count,
        stopped_by_eos,
    })
}

/// Format chat messages into a flat prompt string.
///
/// Simple concatenation: `"role: content\n"` per message, which works
/// reasonably for instruction-tuned models even without a chat template.
fn format_messages_as_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        let _ = writeln!(prompt, "{}: {}", msg.role, msg.content);
    }
    let _ = writeln!(prompt, "assistant:");
    prompt
}

/// Format `OpenAI` messages into a flat prompt string.
fn format_openai_messages_as_prompt(messages: &[OpenAiMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        let _ = writeln!(prompt, "{}: {}", msg.role, msg.content);
    }
    let _ = writeln!(prompt, "assistant:");
    prompt
}

// ===========================================================================
// Ollama endpoints
// ===========================================================================

// ---------------------------------------------------------------------------
// POST /api/generate
// ---------------------------------------------------------------------------

/// Handle `POST /api/generate`.
///
/// Runs real inference against the loaded model. Non-streaming for now.
#[allow(clippy::cast_possible_truncation)]
pub async fn generate(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, (StatusCode, String)> {
    let start = Instant::now();

    let opts = req.options.as_ref();
    let max_tokens = opts.and_then(|o| o.num_predict).unwrap_or(512);
    let temperature = opts.and_then(|o| o.temperature).unwrap_or(0.0);
    let top_k = opts.and_then(|o| o.top_k);
    let top_p = opts.and_then(|o| o.top_p);

    let result = run_inference(&state, &req.prompt, max_tokens, temperature, top_k, top_p)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    let elapsed_ns = start.elapsed().as_nanos() as u64;

    Ok(Json(GenerateResponse {
        model: state.model_name.clone(),
        response: result.text,
        done: true,
        total_duration: Some(elapsed_ns),
        eval_count: Some(result.generated_tokens as u64),
    }))
}

// ---------------------------------------------------------------------------
// POST /api/chat
// ---------------------------------------------------------------------------

/// Handle `POST /api/chat`.
///
/// Formats messages as a flat prompt, runs inference, and returns the
/// assistant's reply.
#[allow(clippy::cast_possible_truncation)]
pub async fn chat(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, (StatusCode, String)> {
    let start = Instant::now();

    let prompt = format_messages_as_prompt(&req.messages);

    let opts = req.options.as_ref();
    let max_tokens = opts.and_then(|o| o.num_predict).unwrap_or(512);
    let temperature = opts.and_then(|o| o.temperature).unwrap_or(0.0);
    let top_k = opts.and_then(|o| o.top_k);
    let top_p = opts.and_then(|o| o.top_p);

    let result = run_inference(&state, &prompt, max_tokens, temperature, top_k, top_p)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    let elapsed_ns = start.elapsed().as_nanos() as u64;

    Ok(Json(ChatResponse {
        model: state.model_name.clone(),
        message: ChatMessage {
            role: String::from("assistant"),
            content: result.text,
        },
        done: true,
        total_duration: Some(elapsed_ns),
    }))
}

// ---------------------------------------------------------------------------
// GET /api/tags
// ---------------------------------------------------------------------------

/// Handle `GET /api/tags`.
///
/// Lists models found under `~/.beacon/models/`. Each subdirectory is
/// treated as a model name.
pub async fn tags(
    State(_state): State<Arc<AppState>>,
) -> Result<Json<TagsResponse>, (StatusCode, String)> {
    let Some(models_path) = models_dir() else {
        // No home directory -- return empty list rather than error.
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
pub async fn pull(
    State(_state): State<Arc<AppState>>,
    Json(_req): Json<PullRequest>,
) -> Result<Json<PullResponse>, (StatusCode, String)> {
    Ok(Json(PullResponse {
        status: String::from("pull not yet implemented"),
    }))
}

// ===========================================================================
// OpenAI endpoints
// ===========================================================================

// ---------------------------------------------------------------------------
// POST /v1/chat/completions
// ---------------------------------------------------------------------------

/// Handle `POST /v1/chat/completions`.
///
/// Runs real inference and returns an OpenAI-compatible chat completion
/// response.
#[allow(clippy::cast_possible_truncation)]
pub async fn openai_chat(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OpenAiChatRequest>,
) -> Result<Json<OpenAiChatResponse>, (StatusCode, String)> {
    let prompt = format_openai_messages_as_prompt(&req.messages);

    let max_tokens = req.max_tokens.unwrap_or(512);
    let temperature = req.temperature.unwrap_or(0.0);
    let top_p = req.top_p;

    let result = run_inference(&state, &prompt, max_tokens, temperature, None, top_p)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    let created = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());

    let finish_reason = if result.stopped_by_eos {
        "stop"
    } else {
        "length"
    };

    Ok(Json(OpenAiChatResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: String::from("chat.completion"),
        created,
        model: state.model_name.clone(),
        choices: vec![OpenAiChoice {
            index: 0,
            message: OpenAiMessage {
                role: String::from("assistant"),
                content: result.text,
            },
            finish_reason: String::from(finish_reason),
        }],
        usage: OpenAiUsage {
            prompt_tokens: result.prompt_tokens as u32,
            completion_tokens: result.generated_tokens as u32,
            total_tokens: (result.prompt_tokens + result.generated_tokens) as u32,
        },
    }))
}

// ---------------------------------------------------------------------------
// GET /v1/models
// ---------------------------------------------------------------------------

/// Handle `GET /v1/models`.
///
/// Returns the currently loaded model in OpenAI-compatible format.
pub async fn openai_models(
    State(state): State<Arc<AppState>>,
) -> Result<Json<OpenAiModelsResponse>, (StatusCode, String)> {
    let created = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());

    Ok(Json(OpenAiModelsResponse {
        object: String::from("list"),
        data: vec![OpenAiModelInfo {
            id: state.model_name.clone(),
            object: String::from("model"),
            created,
            owned_by: String::from("beacon"),
        }],
    }))
}

// ===========================================================================
// Helpers
// ===========================================================================

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
    // Very simple: seconds since epoch -> UTC date-time string.
    // This is intentionally approximate (no leap-second handling) and is
    // sufficient for the Ollama `modified_at` field.
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Days since Unix epoch -> year/month/day (proleptic Gregorian, UTC).
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
