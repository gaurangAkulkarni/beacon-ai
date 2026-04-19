//! Node.js bindings for the Beacon LLM inference engine.
//!
//! Exposes `Engine` — the primary entry point for loading models and running
//! inference from Node.js / TypeScript. Built with napi-rs.
//!
//! # Example
//!
//! ```js
//! const { Engine } = require("beacon-ai");
//!
//! const engine = Engine.load("path/to/model.gguf");
//! const response = engine.complete("Explain unified memory.");
//! console.log(response);
//! ```

#[macro_use]
extern crate napi_derive;

/// A Beacon inference engine instance.
///
/// Use `Engine.load(modelPath)` to create an engine from a model file,
/// then call `complete()` for one-shot generation.
#[napi]
pub struct Engine {
    model_path: String,
}

#[napi]
impl Engine {
    /// Load a model from the given path.
    ///
    /// Accepts `.gguf` files (auto-converted to `.beacon` on first load)
    /// or `.beacon` files directly.
    #[napi(factory)]
    pub fn load(model_path: String) -> Self {
        // Structural placeholder — real implementation will wire up
        // beacon-core::Engine with the appropriate backend.
        Self { model_path }
    }

    /// Generate a complete response for the given prompt.
    ///
    /// Returns the full generated text as a single string.
    #[napi]
    pub fn complete(&self, prompt: String) -> String {
        // Structural placeholder — will call beacon-scheduler's generate
        // pipeline with the configured engine.
        format!(
            "[beacon-ai placeholder] model={}, prompt={}",
            self.model_path, prompt
        )
    }
}
