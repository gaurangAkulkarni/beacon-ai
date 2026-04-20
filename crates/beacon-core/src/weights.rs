//! Model weight structures and loading from `.beacon` files.
//!
//! Weight tensors follow the `HuggingFace` naming convention for Llama-family
//! models. Loading creates zero-copy `MlxTensor` handles backed by the file's
//! mmap. Quantized GGUF weights are re-quantized into MLX's native format at
//! load time so the forward pass can use `quantized_matmul` directly.

use std::sync::Arc;

use beacon_format::{BeaconFile, TensorMeta};
use beacon_mlx::{Dtype, MlxContext, MlxTensor};

use crate::error::EngineError;

/// A weight that may be plain (unquantized) or quantized in MLX's native format.
///
/// Plain weights use `transpose + matmul`; quantized weights use `quantized_matmul`
/// which handles the transpose internally.
#[derive(Debug)]
pub enum ProjectionWeight<T> {
    /// Unquantized — use `transpose + matmul` in the forward pass.
    Plain(T),
    /// Quantized via MLX's `quantize()` — use `quantized_matmul` in the forward pass.
    Quantized {
        /// Packed weight data (uint32).
        packed: T,
        /// Per-group scales.
        scales: T,
        /// Per-group biases.
        biases: T,
        /// Number of elements per quantization group.
        group_size: i32,
        /// Bits per weight element.
        bits: i32,
    },
}

/// Attention projection weights for a single transformer layer.
///
/// The optional `q_bias`, `k_bias`, `v_bias` fields support models that have
/// biases on the QKV projections (e.g. Qwen2). Models without biases (e.g.
/// `LLaMA`) leave these as `None`.
#[derive(Debug)]
pub struct AttentionWeights<T> {
    pub q_proj: ProjectionWeight<T>,
    pub k_proj: ProjectionWeight<T>,
    pub v_proj: ProjectionWeight<T>,
    pub o_proj: ProjectionWeight<T>,
    pub q_bias: Option<T>,
    pub k_bias: Option<T>,
    pub v_bias: Option<T>,
}

/// FFN (feed-forward network) weights for a single transformer layer.
#[derive(Debug)]
pub struct FfnWeights<T> {
    pub gate_proj: ProjectionWeight<T>,
    pub up_proj: ProjectionWeight<T>,
    pub down_proj: ProjectionWeight<T>,
}

/// All weights for a single transformer layer.
#[derive(Debug)]
pub struct LayerWeights<T> {
    /// Input layernorm weight (pre-attention).
    pub attn_norm: T,
    /// Attention projection weights.
    pub attn: AttentionWeights<T>,
    /// Post-attention layernorm weight (pre-FFN).
    pub ffn_norm: T,
    /// FFN weights.
    pub ffn: FfnWeights<T>,
}

/// All loaded model weights.
#[derive(Debug)]
pub struct ModelWeights {
    /// Token embedding matrix.
    pub embed_tokens: MlxTensor,
    /// Per-layer transformer weights.
    pub layers: Vec<LayerWeights<MlxTensor>>,
    /// Final RMS norm weight.
    pub final_norm: MlxTensor,
    /// Language model head (output projection).
    pub lm_head: ProjectionWeight<MlxTensor>,
}

/// Convert a `BeaconDtype` to the MLX `Dtype`.
///
/// The two enums have matching discriminant values by design, so this is a
/// direct mapping.
pub fn beacon_dtype_to_mlx(bd: beacon_format::BeaconDtype) -> Dtype {
    match bd {
        beacon_format::BeaconDtype::F32 => Dtype::F32,
        beacon_format::BeaconDtype::F16 => Dtype::F16,
        beacon_format::BeaconDtype::BF16 => Dtype::BF16,
        beacon_format::BeaconDtype::I32 => Dtype::I32,
        beacon_format::BeaconDtype::I8 => Dtype::I8,
        beacon_format::BeaconDtype::Q4_0 | beacon_format::BeaconDtype::Q4_1 => Dtype::Q4_0,
        beacon_format::BeaconDtype::Q4K
        | beacon_format::BeaconDtype::Q2K
        | beacon_format::BeaconDtype::Q3K => Dtype::Q4K,
        beacon_format::BeaconDtype::Q5_0
        | beacon_format::BeaconDtype::Q5_1
        | beacon_format::BeaconDtype::Q5K => Dtype::Q5K,
        beacon_format::BeaconDtype::Q6K => Dtype::Q6K,
        beacon_format::BeaconDtype::Q8_0 | beacon_format::BeaconDtype::Q8K => Dtype::Q8_0,
    }
}

/// Find a tensor by name in the beacon file's tensor list.
///
/// Tries `HuggingFace` names first, then GGUF names as fallback.
fn find_tensor<'a>(beacon: &'a BeaconFile, name: &str) -> Result<&'a TensorMeta, EngineError> {
    // Try exact match first.
    if let Some(t) = beacon.tensors.iter().find(|t| t.name == name) {
        return Ok(t);
    }
    // Try GGUF equivalent name.
    let gguf_name = hf_to_gguf_name(name);
    if let Some(t) = beacon.tensors.iter().find(|t| t.name == gguf_name) {
        return Ok(t);
    }
    Err(EngineError::WeightNotFound(name.to_owned()))
}

/// Map a `HuggingFace` tensor name to the GGUF equivalent.
fn hf_to_gguf_name(hf_name: &str) -> String {
    // Global tensors.
    match hf_name {
        "model.embed_tokens.weight" => return "token_embd.weight".to_owned(),
        "model.norm.weight" => return "output_norm.weight".to_owned(),
        "lm_head.weight" => return "output.weight".to_owned(),
        _ => {}
    }

    // Layer tensors: model.layers.{i}.xxx → blk.{i}.yyy
    if let Some(rest) = hf_name.strip_prefix("model.layers.") {
        if let Some(dot_pos) = rest.find('.') {
            let layer_num = &rest[..dot_pos];
            let suffix = &rest[dot_pos + 1..];
            let gguf_suffix = match suffix {
                "self_attn.q_proj.weight" => "attn_q.weight",
                "self_attn.k_proj.weight" => "attn_k.weight",
                "self_attn.v_proj.weight" => "attn_v.weight",
                "self_attn.o_proj.weight" => "attn_output.weight",
                "self_attn.q_proj.bias" => "attn_q.bias",
                "self_attn.k_proj.bias" => "attn_k.bias",
                "self_attn.v_proj.bias" => "attn_v.bias",
                "mlp.gate_proj.weight" => "ffn_gate.weight",
                "mlp.up_proj.weight" => "ffn_up.weight",
                "mlp.down_proj.weight" => "ffn_down.weight",
                "input_layernorm.weight" => "attn_norm.weight",
                "post_attention_layernorm.weight" => "ffn_norm.weight",
                other => other, // pass through unknown suffixes
            };
            return format!("blk.{layer_num}.{gguf_suffix}");
        }
    }

    hf_name.to_owned()
}

/// Map `BeaconDtype` to the GGUF tensor type ID for the shim's dequantizer.
fn beacon_dtype_to_gguf_type(dt: beacon_format::BeaconDtype) -> u32 {
    match dt {
        beacon_format::BeaconDtype::Q4_0 => 2,
        beacon_format::BeaconDtype::Q4_1 => 3,
        beacon_format::BeaconDtype::Q5_0 => 6,
        beacon_format::BeaconDtype::Q5_1 => 7,
        beacon_format::BeaconDtype::Q8_0 => 8,
        beacon_format::BeaconDtype::Q2K => 10,
        beacon_format::BeaconDtype::Q3K => 11,
        beacon_format::BeaconDtype::Q4K => 12,
        beacon_format::BeaconDtype::Q5K => 13,
        beacon_format::BeaconDtype::Q6K => 14,
        beacon_format::BeaconDtype::Q8K => 15,
        _ => 0, // F32 fallback (shouldn't reach here)
    }
}

/// Re-quantize an F16 tensor into MLX's native quantized format.
///
/// If quantization succeeds, returns `Quantized`; if it fails (e.g. the tensor
/// is too small for the given group size), falls back to `Plain`.
fn quantize_to_mlx(
    ctx: &Arc<MlxContext>,
    tensor: MlxTensor,
    group_size: i32,
    bits: i32,
) -> ProjectionWeight<MlxTensor> {
    let Ok(stream) = ctx.new_stream() else {
        return ProjectionWeight::Plain(tensor);
    };
    match beacon_mlx::ops::quantize(&stream, &tensor, group_size, bits) {
        Ok(qt) => ProjectionWeight::Quantized {
            packed: qt.packed,
            scales: qt.scales,
            biases: qt.biases,
            group_size,
            bits,
        },
        Err(_) => ProjectionWeight::Plain(tensor),
    }
}

/// Create an `MlxTensor` from a tensor metadata entry in a beacon file.
///
/// For non-quantized tensors (F16, F32, etc.), the tensor is backed by the
/// file's mmap (zero-copy). For quantized tensors (`Q4_0`, `Q8_0`, `Q4_K`, etc.),
/// the raw bytes are dequantized to F16 at load time and placed in an
/// anonymous mmap, because MLX's `matmul` cannot operate on packed quantized
/// byte data directly.
///
/// This costs memory (~2x for Q4 → F16) but is correct and enables all GGUF
/// quantization types to work through the standard `matmul` path.
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn load_tensor(
    beacon: &BeaconFile,
    meta: &TensorMeta,
    ctx: &Arc<MlxContext>,
) -> Result<MlxTensor, EngineError> {
    let shape: Vec<i64> = meta.shape.iter().map(|&d| d as i64).collect();

    if beacon_format::dequant::is_quantized(meta.dtype) {
        // Dequantize via the C shim's gguflib-based dequant (battle-tested).
        // This produces an F32 MlxTensor directly, bypassing the buggy Rust
        // K-quant dequant code.
        let mmap = beacon.mmap();
        let offset = meta.data_offset as usize;
        let data_len = meta.data_length as usize;
        let raw_data = &mmap[offset..offset + data_len];
        let num_elements = meta.num_elements();
        let gguf_type = beacon_dtype_to_gguf_type(meta.dtype);

        let stream = ctx.new_stream().map_err(EngineError::from)?;
        beacon_mlx::ops::dequantize_gguf(&stream, ctx, raw_data, gguf_type, num_elements, &shape)
            .map_err(EngineError::from)
    } else {
        // Non-quantized: zero-copy from the file's mmap.
        let dtype = beacon_dtype_to_mlx(meta.dtype);
        let mmap = Arc::clone(beacon.mmap());
        let offset = meta.data_offset as usize;
        MlxTensor::from_mmap(Arc::clone(ctx), mmap, offset, &shape, dtype)
            .map_err(EngineError::from)
    }
}

/// Load a named tensor from the beacon file.
fn load_named(
    beacon: &BeaconFile,
    name: &str,
    ctx: &Arc<MlxContext>,
) -> Result<MlxTensor, EngineError> {
    let meta = find_tensor(beacon, name)?;
    load_tensor(beacon, meta, ctx)
}

/// Try to load a named tensor, returning `None` if the tensor does not exist.
///
/// This is used for optional weights like QKV biases that are present in some
/// model families (Qwen2) but not others (`LLaMA`).
fn try_load_named(
    beacon: &BeaconFile,
    name: &str,
    ctx: &Arc<MlxContext>,
) -> Result<Option<MlxTensor>, EngineError> {
    match find_tensor(beacon, name) {
        Ok(meta) => Ok(Some(load_tensor(beacon, meta, ctx)?)),
        Err(EngineError::WeightNotFound(_)) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Load a named tensor and re-quantize it into MLX's native format.
///
/// This is used for all projection weights (Q/K/V/O projections, FFN gate/up/down,
/// `lm_head`). The tensor is first loaded (dequantized to F16 if GGUF-quantized),
/// then re-quantized via MLX's `quantize()` into packed uint32 + scales + biases.
fn load_and_quantize(
    beacon: &BeaconFile,
    name: &str,
    ctx: &Arc<MlxContext>,
    group_size: i32,
    bits: i32,
) -> Result<ProjectionWeight<MlxTensor>, EngineError> {
    let meta = find_tensor(beacon, name)?;
    let tensor = load_tensor(beacon, meta, ctx)?;
    // Only quantize if the original tensor was from a quantized GGUF type,
    // meaning it was dequantized to F16 at load time. Non-quantized (native
    // F16/F32) tensors are left plain because they are already in optimal form.
    if beacon_format::dequant::is_quantized(meta.dtype) {
        Ok(quantize_to_mlx(ctx, tensor, group_size, bits))
    } else {
        Ok(ProjectionWeight::Plain(tensor))
    }
}

/// Load all model weights from a `.beacon` file.
///
/// Tensor names follow `HuggingFace` conventions:
/// - `model.embed_tokens.weight`
/// - `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`
/// - `model.layers.{i}.mlp.{gate,up,down}_proj.weight`
/// - `model.layers.{i}.input_layernorm.weight`
/// - `model.layers.{i}.post_attention_layernorm.weight`
/// - `model.norm.weight`
/// - `lm_head.weight`
///
/// Quantized GGUF weights (`Q4_K`, `Q4_0`, etc.) are first dequantized to F16,
/// then re-quantized into MLX's native format (packed uint32 + scales + biases)
/// so that the forward pass can use `quantized_matmul`.
pub fn load_weights(
    beacon: &BeaconFile,
    ctx: &Arc<MlxContext>,
    num_layers: usize,
    tie_word_embeddings: bool,
) -> Result<ModelWeights, EngineError> {
    let embed_tokens = load_named(beacon, "model.embed_tokens.weight", ctx)?;

    // Default quantization parameters matching MLX defaults.
    let group_size = 64;
    let bits = 4;

    let mut layers = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let prefix = format!("model.layers.{i}");
        let attn = AttentionWeights {
            q_proj: load_and_quantize(
                beacon,
                &format!("{prefix}.self_attn.q_proj.weight"),
                ctx,
                group_size,
                bits,
            )?,
            k_proj: load_and_quantize(
                beacon,
                &format!("{prefix}.self_attn.k_proj.weight"),
                ctx,
                group_size,
                bits,
            )?,
            v_proj: load_and_quantize(
                beacon,
                &format!("{prefix}.self_attn.v_proj.weight"),
                ctx,
                group_size,
                bits,
            )?,
            o_proj: load_and_quantize(
                beacon,
                &format!("{prefix}.self_attn.o_proj.weight"),
                ctx,
                group_size,
                bits,
            )?,
            q_bias: try_load_named(beacon, &format!("{prefix}.self_attn.q_proj.bias"), ctx)?,
            k_bias: try_load_named(beacon, &format!("{prefix}.self_attn.k_proj.bias"), ctx)?,
            v_bias: try_load_named(beacon, &format!("{prefix}.self_attn.v_proj.bias"), ctx)?,
        };
        let ffn = FfnWeights {
            gate_proj: load_and_quantize(
                beacon,
                &format!("{prefix}.mlp.gate_proj.weight"),
                ctx,
                group_size,
                bits,
            )?,
            up_proj: load_and_quantize(
                beacon,
                &format!("{prefix}.mlp.up_proj.weight"),
                ctx,
                group_size,
                bits,
            )?,
            down_proj: load_and_quantize(
                beacon,
                &format!("{prefix}.mlp.down_proj.weight"),
                ctx,
                group_size,
                bits,
            )?,
        };
        let attn_norm = load_named(beacon, &format!("{prefix}.input_layernorm.weight"), ctx)?;
        let ffn_norm = load_named(
            beacon,
            &format!("{prefix}.post_attention_layernorm.weight"),
            ctx,
        )?;

        layers.push(LayerWeights {
            attn_norm,
            attn,
            ffn_norm,
            ffn,
        });
    }

    let final_norm = load_named(beacon, "model.norm.weight", ctx)?;

    // When `tie_word_embeddings` is true, `lm_head` reuses `embed_tokens`.
    // Since we cannot cheaply clone an MlxTensor (it owns a raw pointer),
    // we load it again from the same mmap offset — this is zero-copy, just
    // a new handle. Tied embeddings are kept plain (not quantized) because
    // the embedding table must also work with the `embedding` op.
    // When `tie_word_embeddings` is true or the output weight is missing,
    // reuse the embedding table for lm_head.
    let lm_head = if tie_word_embeddings || find_tensor(beacon, "lm_head.weight").is_err() {
        let meta = find_tensor(beacon, "model.embed_tokens.weight")?;
        let tensor = load_tensor(beacon, meta, ctx)?;
        ProjectionWeight::Plain(tensor)
    } else {
        load_and_quantize(beacon, "lm_head.weight", ctx, group_size, bits)?
    };

    Ok(ModelWeights {
        embed_tokens,
        layers,
        final_norm,
        lm_head,
    })
}
