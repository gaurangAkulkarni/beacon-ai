//! Error types for the beacon-registry crate.

/// Errors produced by model resolution, download, and caching.
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    /// The model name could not be resolved to a known alias or valid repo:file
    /// specification.
    #[error("unknown model: {0}")]
    UnknownModel(String),

    /// An HTTP download failed.
    #[error("download failed: {0}")]
    Download(String),

    /// A filesystem I/O error occurred.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A beacon-format error occurred during GGUF conversion.
    #[error("format error: {0}")]
    Format(#[from] beacon_format::FormatError),
}
