//! Model registry: resolve model names, download from `HuggingFace` Hub,
//! convert GGUF to `.beacon`, and manage the local model cache.
//!
//! # Usage
//!
//! ```rust,no_run
//! use beacon_registry::{resolve_model, pull_model, default_models_dir};
//!
//! let spec = resolve_model("qwen2.5-0.5b").unwrap();
//! let result = pull_model(&spec, &default_models_dir()).unwrap();
//! println!("Model ready at: {}", result.beacon_path.display());
//! ```

pub mod download;
pub mod error;
pub mod models;

pub use download::{default_models_dir, pull_model, PullResult};
pub use error::RegistryError;
pub use models::{resolve_model, ModelSpec};
