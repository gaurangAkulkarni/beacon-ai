//! Error types for the beacon-format crate.

/// Errors produced by GGUF parsing, `.beacon` reading/writing, and conversion.
#[derive(Debug, thiserror::Error)]
pub enum FormatError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid magic: expected {expected:?}, got {got:?}")]
    InvalidMagic {
        expected: &'static [u8],
        got: [u8; 4],
    },

    #[error("unsupported GGUF version: {0} (expected 3)")]
    UnsupportedVersion(u32),

    #[error("unsupported GGUF tensor type: {0}")]
    UnsupportedGgufType(u32),

    #[error("missing metadata key: {0}")]
    MissingMetadata(String),

    #[error("invalid metadata: {0}")]
    InvalidMetadata(String),

    #[error("unsupported architecture: {0}")]
    UnsupportedArchitecture(String),

    #[error("truncated file: need {needed} bytes at offset {offset}, file is {file_len} bytes")]
    Truncated {
        needed: u64,
        offset: u64,
        file_len: u64,
    },

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}
