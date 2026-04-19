//! Extract [`ModelConfig`] from GGUF metadata.
//!
//! GGUF stores model parameters as typed key-value metadata. Keys are prefixed
//! with the architecture name (e.g. `qwen2.embedding_length`). This module
//! reads those keys and populates a [`ModelConfig`].

use std::collections::BTreeMap;

use crate::config::{Architecture, ModelConfig, RopeScaling};
use crate::error::FormatError;
use crate::gguf::types::GgufValue;

/// Extract a [`ModelConfig`] from parsed GGUF metadata.
pub fn model_config_from_metadata(
    meta: &BTreeMap<String, GgufValue>,
) -> Result<ModelConfig, FormatError> {
    // Determine architecture.
    let arch_name = get_str(meta, "general.architecture")?;
    let architecture = Architecture::from_gguf_name(arch_name)
        .ok_or_else(|| FormatError::UnsupportedArchitecture(arch_name.to_owned()))?;
    let prefix = arch_name;

    let hidden_size = get_usize(meta, &format!("{prefix}.embedding_length"))?;
    let num_layers = get_usize(meta, &format!("{prefix}.block_count"))?;
    let num_heads = get_usize(meta, &format!("{prefix}.attention.head_count"))?;
    let num_kv_heads = get_usize_or(
        meta,
        &format!("{prefix}.attention.head_count_kv"),
        num_heads,
    );
    let intermediate_size = get_usize(meta, &format!("{prefix}.feed_forward_length"))?;
    let head_dim = get_usize_or(
        meta,
        &format!("{prefix}.attention.key_length"),
        hidden_size / num_heads,
    );
    let vocab_size = get_usize_optional(meta, &format!("{prefix}.vocab_size"));
    let max_position_embeddings = get_usize_or(meta, &format!("{prefix}.context_length"), 4096);
    let rope_theta = get_f32_or(meta, &format!("{prefix}.rope.freq_base"), 10000.0);
    let rms_norm_eps = get_f32_or(
        meta,
        &format!("{prefix}.attention.layer_norm_rms_epsilon"),
        1e-5,
    );

    // RoPE scaling (optional).
    let rope_scaling = meta
        .get(&format!("{prefix}.rope.scaling.type"))
        .and_then(GgufValue::as_str)
        .map(|type_| {
            let factor = get_f32_or(meta, &format!("{prefix}.rope.scaling.factor"), 1.0);
            RopeScaling {
                type_: type_.to_owned(),
                factor,
            }
        });

    // Tokenizer special tokens.
    let bos_token_id = meta
        .get("tokenizer.ggml.bos_token_id")
        .and_then(GgufValue::as_u32);
    let eos_token_ids = extract_eos_token_ids(meta);

    // `tie_word_embeddings` is not always present in GGUF; default to false.
    let tie_word_embeddings = meta
        .get(&format!("{prefix}.tie_word_embeddings"))
        .or_else(|| meta.get("general.tie_word_embeddings"))
        .and_then(GgufValue::as_bool)
        .unwrap_or(false);

    // vocab_size: try metadata, fall back to tokenizer.ggml.tokens array length.
    let vocab_size = vocab_size.unwrap_or_else(|| {
        meta.get("tokenizer.ggml.tokens")
            .and_then(GgufValue::as_array)
            .map_or(0, <[GgufValue]>::len)
    });

    // Chat template (optional string).
    let chat_template = meta
        .get("tokenizer.chat_template")
        .and_then(GgufValue::as_str)
        .map(String::from);

    Ok(ModelConfig {
        architecture,
        hidden_size,
        num_layers,
        num_heads,
        num_kv_heads,
        intermediate_size,
        head_dim,
        vocab_size,
        max_position_embeddings,
        rope_theta,
        rope_scaling,
        rms_norm_eps,
        tie_word_embeddings,
        bos_token_id,
        eos_token_ids,
        chat_template,
    })
}

// --- Helpers -----------------------------------------------------------------

fn get_str<'a>(meta: &'a BTreeMap<String, GgufValue>, key: &str) -> Result<&'a str, FormatError> {
    meta.get(key)
        .and_then(GgufValue::as_str)
        .ok_or_else(|| FormatError::MissingMetadata(key.to_owned()))
}

fn get_usize(meta: &BTreeMap<String, GgufValue>, key: &str) -> Result<usize, FormatError> {
    meta.get(key)
        .and_then(GgufValue::as_usize)
        .ok_or_else(|| FormatError::MissingMetadata(key.to_owned()))
}

fn get_usize_or(meta: &BTreeMap<String, GgufValue>, key: &str, default: usize) -> usize {
    meta.get(key)
        .and_then(GgufValue::as_usize)
        .unwrap_or(default)
}

fn get_usize_optional(meta: &BTreeMap<String, GgufValue>, key: &str) -> Option<usize> {
    meta.get(key).and_then(GgufValue::as_usize)
}

fn get_f32_or(meta: &BTreeMap<String, GgufValue>, key: &str, default: f32) -> f32 {
    meta.get(key).and_then(GgufValue::as_f32).unwrap_or(default)
}

/// Extract EOS token IDs.
///
/// GGUF may store a single `tokenizer.ggml.eos_token_id` or an array.
fn extract_eos_token_ids(meta: &BTreeMap<String, GgufValue>) -> Vec<u32> {
    // Try array first.
    if let Some(arr) = meta
        .get("tokenizer.ggml.eos_token_id")
        .and_then(GgufValue::as_array)
    {
        return arr.iter().filter_map(GgufValue::as_u32).collect();
    }
    // Fall back to single value.
    if let Some(id) = meta
        .get("tokenizer.ggml.eos_token_id")
        .and_then(GgufValue::as_u32)
    {
        return vec![id];
    }
    Vec::new()
}
