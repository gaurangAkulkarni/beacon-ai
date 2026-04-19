//! The inference engine — owns model weights, KV cache, and backend.
//!
//! Implements the transformer forward pass following architecture doc sections
//! 7.3 (attention), 7.4 (FFN), and 7.5 (full forward).

use std::sync::Arc;

use beacon_format::{BeaconFile, ModelConfig};
use beacon_mlx::{Dtype, MlxTensor};

use crate::backend::ComputeBackend;
use crate::cpu_backend::{CpuBackend, CpuTensor};
use crate::error::EngineError;
use crate::kv_cache::KvCache;
use crate::mlx_backend::MlxBackend;
use crate::weights::{load_weights, LayerWeights};

/// The inference engine.
///
/// Owns the model weights, KV cache per layer, model config, and the compute
/// backend. Parameterised over `B: ComputeBackend` so the same forward pass
/// code works with MLX, CPU, or CUDA.
#[derive(Debug)]
pub struct Engine<B: ComputeBackend> {
    /// Model configuration (architecture, dims, etc.).
    pub config: ModelConfig,
    backend: B,
    embed_tokens: B::Tensor,
    layers: Vec<LayerWeights<B::Tensor>>,
    final_norm: B::Tensor,
    lm_head: B::Tensor,
    cache: Vec<KvCache<B::Tensor>>,
}

impl Engine<MlxBackend> {
    /// Load a model from a `.beacon` file using the MLX backend.
    ///
    /// This loads all weight tensors (zero-copy via mmap) and allocates the
    /// KV cache in unified memory.
    #[allow(clippy::cast_possible_wrap)]
    pub fn load(beacon: &BeaconFile, backend: MlxBackend) -> Result<Self, EngineError> {
        let config = beacon.config.clone();
        let ctx = Arc::clone(backend.context());

        let weights = load_weights(beacon, &ctx, config.num_layers, config.tie_word_embeddings)?;

        // Allocate KV caches — one per layer, preallocated to max context.
        let mut cache = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            let shape = [
                config.max_position_embeddings as i64,
                config.num_kv_heads as i64,
                config.head_dim as i64,
            ];
            let cache_k = MlxTensor::zeros(Arc::clone(&ctx), &shape, Dtype::F16)?;
            let cache_v = MlxTensor::zeros(Arc::clone(&ctx), &shape, Dtype::F16)?;
            cache.push(KvCache {
                cache_k,
                cache_v,
                current_length: 0,
            });
        }

        Ok(Self {
            config,
            backend,
            embed_tokens: weights.embed_tokens,
            layers: weights.layers,
            final_norm: weights.final_norm,
            lm_head: weights.lm_head,
            cache,
        })
    }
}

impl Engine<CpuBackend> {
    /// Create a CPU-backend engine from synthetic/pre-built weights.
    ///
    /// In v0.1, the CPU backend is the fallback path. This constructor takes
    /// pre-built `CpuTensor` weights directly. Loading from `.beacon` files
    /// with F32 weight conversion is deferred until real model testing.
    #[allow(clippy::cast_possible_wrap)]
    pub fn load_cpu(
        config: ModelConfig,
        embed_tokens: CpuTensor,
        layers: Vec<LayerWeights<CpuTensor>>,
        final_norm: CpuTensor,
        lm_head: CpuTensor,
    ) -> Result<Self, EngineError> {
        let backend = CpuBackend;

        // Allocate KV caches — zero-filled CpuTensors.
        let mut cache = Vec::with_capacity(config.num_layers);
        let kv_dim = config.num_kv_heads * config.head_dim;
        for _ in 0..config.num_layers {
            let cache_k = CpuTensor {
                data: vec![0.0; config.max_position_embeddings * kv_dim],
                shape: vec![
                    config.max_position_embeddings as i64,
                    config.num_kv_heads as i64,
                    config.head_dim as i64,
                ],
            };
            let cache_v = CpuTensor {
                data: vec![0.0; config.max_position_embeddings * kv_dim],
                shape: vec![
                    config.max_position_embeddings as i64,
                    config.num_kv_heads as i64,
                    config.head_dim as i64,
                ],
            };
            cache.push(KvCache {
                cache_k,
                cache_v,
                current_length: 0,
            });
        }

        Ok(Self {
            config,
            backend,
            embed_tokens,
            layers,
            final_norm,
            lm_head,
            cache,
        })
    }
}

impl<B: ComputeBackend> Engine<B> {
    /// Run the full transformer forward pass on a sequence of tokens.
    ///
    /// Returns the logits tensor. `position` is the starting position in the
    /// KV cache (0 for the first call, incremented by `tokens.len()` for
    /// subsequent calls).
    ///
    /// Follows architecture doc section 7.5.
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    pub fn forward(&mut self, tokens: &[u32], position: usize) -> Result<B::Tensor, EngineError> {
        let stream = self.backend.new_stream()?;
        let seq_len = tokens.len();

        // 1. Embed tokens.
        // GGUF stores embedding as [hidden, vocab] so we transpose to [vocab, hidden]
        // before the embedding lookup.
        let embed_t = self.backend.transpose(&stream, &self.embed_tokens)?;
        let indices = self.backend.create_token_tensor(tokens)?;
        let mut h = self.backend.embedding(&stream, &embed_t, &indices)?;

        // 2. For each layer: attn_norm -> attention -> residual -> ffn_norm -> ffn -> residual.
        for i in 0..self.layers.len() {
            // Attention sub-block.
            let normed = self.backend.rms_norm(
                &stream,
                &h,
                &self.layers[i].attn_norm,
                self.config.rms_norm_eps,
            )?;

            let attn = self.attention_block(&stream, &normed, i, position, seq_len)?;
            h = self.backend.add(&stream, &h, &attn)?;

            // FFN sub-block.
            let normed = self.backend.rms_norm(
                &stream,
                &h,
                &self.layers[i].ffn_norm,
                self.config.rms_norm_eps,
            )?;

            let ffn = self.ffn_block(&stream, &normed, i)?;
            h = self.backend.add(&stream, &h, &ffn)?;

            // Eval at layer boundary to bound MLX graph size (arch doc section 4.4).
            self.backend.eval(&h, &stream)?;
        }

        // 3. Final norm + lm_head matmul.
        let h = self
            .backend
            .rms_norm(&stream, &h, &self.final_norm, self.config.rms_norm_eps)?;
        let logits = self.backend.matmul(&stream, &h, &self.lm_head)?;
        self.backend.eval(&logits, &stream)?;

        // Update KV cache positions.
        for kv in &mut self.cache {
            kv.current_length = position + seq_len;
        }

        Ok(logits)
    }

    /// Generate the next token given a single input token (greedy, T=0).
    ///
    /// Runs a single-token forward pass, reads the logits, and returns the
    /// argmax token ID.
    #[allow(clippy::cast_possible_truncation)]
    pub fn generate_next_token(&mut self, token: u32, position: usize) -> Result<u32, EngineError> {
        let logits = self.forward(&[token], position)?;

        // Read the logits for the last (only) token.
        let vocab_size = self.config.vocab_size;
        let logit_values = self.backend.read_f32(&logits, vocab_size)?;

        // Greedy argmax.
        let next_token = logit_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(idx, _)| idx as u32);

        Ok(next_token)
    }

    /// Attention block for a single layer (architecture doc section 7.3).
    ///
    /// In v0.1, all projections use regular `matmul` instead of
    /// `quantized_matmul` because GGUF quantized weights are not in MLX's
    /// internal quantization format. TODO: add `quantized_matmul` path once
    /// the format bridge is in place.
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss
    )]
    fn attention_block(
        &mut self,
        stream: &B::Stream,
        hidden: &B::Tensor,
        layer_idx: usize,
        position: usize,
        seq_len: usize,
    ) -> Result<B::Tensor, EngineError> {
        let cfg = &self.config;
        let attn = &self.layers[layer_idx].attn;

        // Q/K/V projections — use matmul for v0.1 (see doc comment above).
        // GGUF stores weights as [in_features, out_features], so matmul(x, W)
        // gives the correct result without transposing.
        let q = self.backend.matmul(stream, hidden, &attn.q_proj)?;
        let k = self.backend.matmul(stream, hidden, &attn.k_proj)?;
        let v = self.backend.matmul(stream, hidden, &attn.v_proj)?;

        // Reshape for multi-head attention:
        // q: [seq, hidden] -> [1, n_heads, seq, head_dim]
        // k,v: [seq, hidden] -> [1, n_kv_heads, seq, head_dim]
        let q = self.backend.reshape(
            stream,
            &q,
            &[1, seq_len as i64, cfg.num_heads as i64, cfg.head_dim as i64],
        )?;
        // Note: a proper transpose [1,seq,heads,dim] -> [1,heads,seq,dim] is
        // needed here. For v0.1, we use reshape which assumes contiguous layout.
        // TODO: add transpose op to the backend trait for correct multi-head layout.
        let q = self.backend.reshape(
            stream,
            &q,
            &[1, cfg.num_heads as i64, seq_len as i64, cfg.head_dim as i64],
        )?;

        let k = self.backend.reshape(
            stream,
            &k,
            &[
                1,
                cfg.num_kv_heads as i64,
                seq_len as i64,
                cfg.head_dim as i64,
            ],
        )?;

        let v = self.backend.reshape(
            stream,
            &v,
            &[
                1,
                cfg.num_kv_heads as i64,
                seq_len as i64,
                cfg.head_dim as i64,
            ],
        )?;

        // RoPE on Q and K.
        let q = self.backend.rope(
            stream,
            &q,
            position as i32,
            cfg.rope_theta,
            cfg.head_dim as i32,
        )?;
        let k = self.backend.rope(
            stream,
            &k,
            position as i32,
            cfg.rope_theta,
            cfg.head_dim as i32,
        )?;

        // Context overflow check.
        if position + seq_len > cfg.max_position_embeddings {
            return Err(EngineError::ContextOverflow);
        }

        // Squeeze batch dim from K/V for cache update: [1,heads,seq,dim] → [seq,heads,dim]
        // The KV cache is allocated as [max_ctx, num_kv_heads, head_dim].
        let k_squeezed = self.backend.reshape(
            stream,
            &k,
            &[seq_len as i64, cfg.num_kv_heads as i64, cfg.head_dim as i64],
        )?;
        let v_squeezed = self.backend.reshape(
            stream,
            &v,
            &[seq_len as i64, cfg.num_kv_heads as i64, cfg.head_dim as i64],
        )?;

        // KV cache update — write new K/V at current position and get views
        // over the full cached sequence for attention.
        let (k_cached, v_cached) = self.backend.kv_cache_update(
            stream,
            &self.cache[layer_idx].cache_k,
            &self.cache[layer_idx].cache_v,
            &k_squeezed,
            &v_squeezed,
            position as i64,
        )?;

        // Re-add batch dim for attention: [cached_len, heads, dim] → [1, heads, cached_len, dim]
        let cached_len = (position + seq_len) as i64;
        let k_full = self.backend.reshape(
            stream,
            &k_cached,
            &[1, cfg.num_kv_heads as i64, cached_len, cfg.head_dim as i64],
        )?;
        let v_full = self.backend.reshape(
            stream,
            &v_cached,
            &[1, cfg.num_kv_heads as i64, cached_len, cfg.head_dim as i64],
        )?;

        // Scaled dot-product attention (fused in MLX backend).
        let scale = 1.0 / (cfg.head_dim as f32).sqrt();
        let attn_out = self
            .backend
            .attention(stream, &q, &k_full, &v_full, None, scale)?;

        // Reshape attention output back to [seq, hidden].
        let attn_out =
            self.backend
                .reshape(stream, &attn_out, &[seq_len as i64, cfg.hidden_size as i64])?;

        // Output projection.
        self.backend.matmul(stream, &attn_out, &attn.o_proj)
    }

    /// FFN block for a single layer (architecture doc section 7.4).
    ///
    /// `SwiGLU`: gate projection + `SiLU`, element-wise multiply with up
    /// projection, then down projection.
    fn ffn_block(
        &self,
        stream: &B::Stream,
        hidden: &B::Tensor,
        layer_idx: usize,
    ) -> Result<B::Tensor, EngineError> {
        let ffn = &self.layers[layer_idx].ffn;

        // Gate projection + SiLU.
        // GGUF stores weights as [in, out], so matmul(x, W) is correct.
        let gate = self.backend.matmul(stream, hidden, &ffn.gate_proj)?;
        let gate = self.backend.silu(stream, &gate)?;

        // Up projection.
        let up = self.backend.matmul(stream, hidden, &ffn.up_proj)?;

        // Gate * Up.
        let inter = self.backend.mul(stream, &gate, &up)?;

        // Down projection.
        self.backend.matmul(stream, &inter, &ffn.down_proj)
    }
}
