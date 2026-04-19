//! Error types for the beacon-tokenizer crate.

/// Errors produced by tokenization and chat template rendering.
#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("failed to load tokenizer: {0}")]
    Load(String),

    #[error("encoding error: {0}")]
    Encode(String),

    #[error("decoding error: {0}")]
    Decode(String),

    #[error("chat template error: {0}")]
    Template(String),

    #[error("no chat template configured")]
    NoChatTemplate,

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}
