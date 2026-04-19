//! KV cache for transformer attention.
//!
//! Per-layer, contiguous, preallocated at engine load. Layout follows
//! architecture doc section 8:
//! ```text
//! cache_k: [max_context, num_kv_heads, head_dim]
//! cache_v: [max_context, num_kv_heads, head_dim]
//! ```

/// KV cache for a single transformer layer.
///
/// `T` is the backend's tensor type (`MlxTensor` for MLX, etc.).
#[derive(Debug)]
pub struct KvCache<T> {
    /// Key cache: `[max_context, num_kv_heads, head_dim]`.
    pub cache_k: T,
    /// Value cache: `[max_context, num_kv_heads, head_dim]`.
    pub cache_v: T,
    /// Number of tokens currently stored in the cache.
    pub current_length: usize,
}
