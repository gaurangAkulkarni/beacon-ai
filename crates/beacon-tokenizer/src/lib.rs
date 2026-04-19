//! BPE tokenizer with chat template support.
//!
//! Wraps the `HuggingFace` [`tokenizers`] crate for BPE encoding/decoding and
//! [`minijinja`] for Jinja2-subset chat template rendering. Designed to
//! produce token-for-token identical output to the `HuggingFace` `tokenizers`
//! Python library.

mod chat;
mod error;
mod tokenizer;

pub use chat::ChatMessage;
pub use error::TokenizerError;
pub use tokenizer::BeaconTokenizer;

#[cfg(test)]
mod tests;
