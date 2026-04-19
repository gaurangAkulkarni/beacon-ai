//! Python bindings for the Beacon LLM inference engine.
//!
//! Exposes `beacon_ai.Engine` — the primary entry point for loading models
//! and running inference from Python. Built with PyO3 + maturin.
//!
//! # Example
//!
//! ```python
//! import beacon_ai
//!
//! engine = beacon_ai.Engine.load("path/to/model.gguf")
//! print(engine.complete("Hello, world!"))
//!
//! for token in engine.stream("Explain unified memory."):
//!     print(token, end="", flush=True)
//! ```

use pyo3::prelude::*;

/// A Beacon inference engine instance.
///
/// Use `Engine.load(model_path)` to create an engine from a model file,
/// then call `complete()` for one-shot generation or `stream()` for
/// token-by-token iteration.
#[pyclass]
#[allow(dead_code)]
struct Engine {
    model_path: String,
}

#[pymethods]
impl Engine {
    /// Load a model from the given path.
    ///
    /// Accepts `.gguf` files (auto-converted to `.beacon` on first load)
    /// or `.beacon` files directly.
    #[staticmethod]
    fn load(model_path: String) -> PyResult<Self> {
        // Structural placeholder — real implementation will wire up
        // beacon-core::Engine with the appropriate backend.
        Ok(Self { model_path })
    }

    /// Generate a complete response for the given prompt.
    ///
    /// Returns the full generated text as a single string.
    #[pyo3(signature = (prompt, *, max_tokens = 512, temperature = 0.0))]
    fn complete(&self, prompt: String, max_tokens: u32, temperature: f64) -> PyResult<String> {
        // Structural placeholder — will call beacon-scheduler's generate
        // pipeline with the configured engine.
        let _ = max_tokens;
        let _ = temperature;
        Ok(format!(
            "[beacon-ai placeholder] model={}, prompt={}",
            self.model_path, prompt
        ))
    }

    /// Stream generated tokens for the given prompt.
    ///
    /// Returns an iterator that yields one token string at a time.
    #[pyo3(signature = (prompt, *, max_tokens = 512, temperature = 0.0))]
    fn stream(&self, prompt: String, max_tokens: u32, temperature: f64) -> PyResult<TokenIterator> {
        // Structural placeholder — will wrap a tokio mpsc receiver from
        // beacon-scheduler's streaming pipeline.
        let _ = max_tokens;
        let _ = temperature;
        let tokens = vec![
            format!("[beacon-ai placeholder] model={}, ", self.model_path),
            format!("prompt={}", prompt),
        ];
        Ok(TokenIterator {
            tokens,
            position: 0,
        })
    }
}

/// Iterator yielding generated tokens one at a time.
#[pyclass]
struct TokenIterator {
    tokens: Vec<String>,
    position: usize,
}

#[pymethods]
impl TokenIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<String> {
        if self.position < self.tokens.len() {
            let token = self.tokens[self.position].clone();
            self.position += 1;
            Some(token)
        } else {
            None
        }
    }
}

/// The `beacon_ai` Python module.
#[pymodule]
fn beacon_ai(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engine>()?;
    m.add_class::<TokenIterator>()?;
    Ok(())
}
