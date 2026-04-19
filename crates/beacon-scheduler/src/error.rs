//! Error types for the scheduler.

/// Errors produced during generation.
#[derive(Debug, thiserror::Error)]
pub enum SchedulerError {
    /// The engine returned an error during the forward pass.
    #[error("engine error: {0}")]
    Engine(String),

    /// Generation was cancelled by the caller.
    #[error("generation cancelled")]
    Cancelled,

    /// Maximum token limit reached.
    #[error("max tokens ({0}) reached")]
    MaxTokens(usize),

    /// Timeout expired.
    #[error("generation timeout")]
    Timeout,
}
