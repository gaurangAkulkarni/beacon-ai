//! Batching, streaming, KV cache lifecycle, and sampling.
//!
//! The scheduler drives the decode loop: it calls the engine's forward pass,
//! applies the sampling pipeline, emits tokens via `tokio::sync::mpsc`
//! channels, and enforces stop conditions (stop tokens, max tokens, timeout).

mod error;
pub mod generate;
pub mod params;
pub mod sampling;

pub use error::SchedulerError;
pub use generate::{generate_stream, generate_stream_sync, GeneratedToken};
pub use params::GenerationParams;

#[cfg(test)]
mod tests;
