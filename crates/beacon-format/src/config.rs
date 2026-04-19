//! Model configuration types.
//!
//! These are serialised as JSON in the `.beacon` header and extracted from GGUF
//! metadata during conversion. The struct mirrors the architecture doc §7.1.

/// Supported model architecture families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Architecture {
    Llama,
    Qwen,
    Phi,
    Gemma,
}

impl Architecture {
    /// Map a GGUF `general.architecture` string to an `Architecture`.
    pub fn from_gguf_name(name: &str) -> Option<Self> {
        match name {
            "llama" => Some(Self::Llama),
            "qwen2" | "qwen" => Some(Self::Qwen),
            "phi3" | "phi" => Some(Self::Phi),
            "gemma" | "gemma2" => Some(Self::Gemma),
            _ => None,
        }
    }
}

/// `RoPE` scaling configuration (optional; absent for models using the default).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RopeScaling {
    /// Scaling type (e.g. `"linear"`, `"dynamic"`).
    #[serde(rename = "type")]
    pub type_: String,
    /// Scaling factor.
    pub factor: f32,
}

/// Full model configuration.
///
/// Serialised as length-prefixed JSON in the `.beacon` header. Fields map 1:1
/// to the architecture doc §7.1.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ModelConfig {
    pub architecture: Architecture,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    /// GQA: `num_kv_heads <= num_heads`.
    pub num_kv_heads: usize,
    /// FFN hidden dimension.
    pub intermediate_size: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rope_scaling: Option<RopeScaling>,
    pub rms_norm_eps: f32,
    pub tie_word_embeddings: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bos_token_id: Option<u32>,
    pub eos_token_ids: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template: Option<String>,
}
