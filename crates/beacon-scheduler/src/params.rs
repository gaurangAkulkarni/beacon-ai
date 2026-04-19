//! Generation parameters controlling sampling and stop conditions.

/// Parameters for text generation.
///
/// Defaults produce greedy decoding (temperature = 0, no sampling).
#[derive(Debug, Clone)]
pub struct GenerationParams {
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,

    /// Sampling temperature. 0.0 = greedy, 1.0 = standard, >1.0 = more random.
    pub temperature: f32,

    /// Top-k sampling: only consider the `k` most likely tokens.
    pub top_k: Option<usize>,

    /// Top-p (nucleus) sampling: only consider tokens with cumulative
    /// probability up to `p`.
    pub top_p: Option<f32>,

    /// Min-p sampling: only consider tokens with probability ≥ `min_p *
    /// max_prob`.
    pub min_p: Option<f32>,

    /// Repeat penalty applied to previously generated tokens. 1.0 = no
    /// penalty.
    pub repeat_penalty: f32,

    /// Token IDs that signal end of generation.
    pub stop_tokens: Vec<u32>,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.0,
            top_k: None,
            top_p: None,
            min_p: None,
            repeat_penalty: 1.0,
            stop_tokens: Vec::new(),
        }
    }
}
