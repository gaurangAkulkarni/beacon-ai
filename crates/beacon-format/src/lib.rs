//! The `.beacon` model format and GGUF importer.
//!
//! This crate provides:
//!
//! - A GGUF v3 parser ([`GgufFile`]) that reads model metadata and tensor info
//!   from GGUF files via mmap.
//! - A `.beacon` format reader ([`BeaconFile`]) and writer that stores model
//!   config, tokenizer data, and 64-byte-aligned tensor data in a single
//!   self-contained file.
//! - A converter ([`convert_gguf_to_beacon`], [`load_or_convert`]) that
//!   transforms GGUF files into `.beacon` format with caching.

pub mod beacon;
mod config;
pub mod convert;
mod dtype;
mod error;
pub mod gguf;
mod tensor_meta;

pub use beacon::BeaconFile;
pub use config::{Architecture, ModelConfig, RopeScaling};
pub use convert::{convert_gguf_to_beacon, load_or_convert};
pub use dtype::BeaconDtype;
pub use error::FormatError;
pub use gguf::GgufFile;
pub use tensor_meta::TensorMeta;

#[cfg(test)]
mod tests;
