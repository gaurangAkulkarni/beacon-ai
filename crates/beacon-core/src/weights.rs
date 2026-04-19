//! Model weight structures and loading from `.beacon` files.
//!
//! Weight tensors follow the `HuggingFace` naming convention for Llama-family
//! models. Loading creates zero-copy `MlxTensor` handles backed by the file's
//! mmap.

use std::sync::Arc;

use beacon_format::{BeaconFile, TensorMeta};
use beacon_mlx::{Dtype, MlxContext, MlxTensor};

use crate::error::EngineError;

/// Attention projection weights for a single transformer layer.
#[derive(Debug)]
pub struct AttentionWeights<T> {
    pub q_proj: T,
    pub k_proj: T,
    pub v_proj: T,
    pub o_proj: T,
}

/// FFN (feed-forward network) weights for a single transformer layer.
#[derive(Debug)]
pub struct FfnWeights<T> {
    pub gate_proj: T,
    pub up_proj: T,
    pub down_proj: T,
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
    pub lm_head: MlxTensor,
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
        beacon_format::BeaconDtype::Q4_0 => Dtype::Q4_0,
        beacon_format::BeaconDtype::Q4K => Dtype::Q4K,
        beacon_format::BeaconDtype::Q5K => Dtype::Q5K,
        beacon_format::BeaconDtype::Q6K => Dtype::Q6K,
        beacon_format::BeaconDtype::Q8_0 => Dtype::Q8_0,
    }
}

/// Find a tensor by name in the beacon file's tensor list.
fn find_tensor<'a>(beacon: &'a BeaconFile, name: &str) -> Result<&'a TensorMeta, EngineError> {
    beacon
        .tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| EngineError::WeightNotFound(name.to_owned()))
}

/// Create an `MlxTensor` from a tensor metadata entry in a beacon file.
///
/// The tensor is backed by the file's mmap (zero-copy).
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn load_tensor(
    beacon: &BeaconFile,
    meta: &TensorMeta,
    ctx: &Arc<MlxContext>,
) -> Result<MlxTensor, EngineError> {
    let shape: Vec<i64> = meta.shape.iter().map(|&d| d as i64).collect();
    let dtype = beacon_dtype_to_mlx(meta.dtype);
    let mmap = Arc::clone(beacon.mmap());
    let offset = meta.data_offset as usize;
    MlxTensor::from_mmap(Arc::clone(ctx), mmap, offset, &shape, dtype).map_err(EngineError::from)
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
pub fn load_weights(
    beacon: &BeaconFile,
    ctx: &Arc<MlxContext>,
    num_layers: usize,
    tie_word_embeddings: bool,
) -> Result<ModelWeights, EngineError> {
    let embed_tokens = load_named(beacon, "model.embed_tokens.weight", ctx)?;

    let mut layers = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let prefix = format!("model.layers.{i}");
        let attn = AttentionWeights {
            q_proj: load_named(beacon, &format!("{prefix}.self_attn.q_proj.weight"), ctx)?,
            k_proj: load_named(beacon, &format!("{prefix}.self_attn.k_proj.weight"), ctx)?,
            v_proj: load_named(beacon, &format!("{prefix}.self_attn.v_proj.weight"), ctx)?,
            o_proj: load_named(beacon, &format!("{prefix}.self_attn.o_proj.weight"), ctx)?,
        };
        let ffn = FfnWeights {
            gate_proj: load_named(beacon, &format!("{prefix}.mlp.gate_proj.weight"), ctx)?,
            up_proj: load_named(beacon, &format!("{prefix}.mlp.up_proj.weight"), ctx)?,
            down_proj: load_named(beacon, &format!("{prefix}.mlp.down_proj.weight"), ctx)?,
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
    // a new handle.
    let lm_head = if tie_word_embeddings {
        let meta = find_tensor(beacon, "model.embed_tokens.weight")?;
        load_tensor(beacon, meta, ctx)?
    } else {
        load_named(beacon, "lm_head.weight", ctx)?
    };

    Ok(ModelWeights {
        embed_tokens,
        layers,
        final_norm,
        lm_head,
    })
}
