// beacon_shim.h
// Beacon MLX shim — C ABI. Authoritative surface for the shim; mirror of
// docs/architecture.md §3.3. All functions return a BeaconStatus code (int32_t);
// output parameters are the final argument. C++ exceptions never cross this
// boundary — they are caught at the ABI and translated to status codes plus a
// thread-local error message (beacon_last_error_message).

#ifndef BEACON_SHIM_H
#define BEACON_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// === Handles (opaque) ===
typedef struct BeaconContext BeaconContext;
typedef struct BeaconTensor BeaconTensor;
typedef struct BeaconStream BeaconStream;

// === Status codes ===
typedef enum {
    BEACON_OK = 0,
    BEACON_ERR_INVALID_ARGUMENT = 1,
    BEACON_ERR_OUT_OF_MEMORY = 2,
    BEACON_ERR_SHAPE_MISMATCH = 3,
    BEACON_ERR_UNSUPPORTED_DTYPE = 4,
    BEACON_ERR_METAL_COMPILE = 5,
    BEACON_ERR_MLX_INTERNAL = 6,
    BEACON_ERR_UNKNOWN = 99
} BeaconStatus;

// === Dtypes ===
typedef enum {
    BEACON_DTYPE_F32 = 0,
    BEACON_DTYPE_F16 = 1,
    BEACON_DTYPE_BF16 = 2,
    BEACON_DTYPE_I32 = 3,
    BEACON_DTYPE_I8 = 4,
    BEACON_DTYPE_Q4_0 = 100,
    BEACON_DTYPE_Q4_K = 101,
    BEACON_DTYPE_Q5_K = 102,
    BEACON_DTYPE_Q6_K = 103,
    BEACON_DTYPE_Q8_0 = 104
} BeaconDtype;

// === Lifecycle ===
int32_t beacon_context_create(BeaconContext** out_ctx);
void    beacon_context_destroy(BeaconContext* ctx);

int32_t beacon_stream_create(BeaconContext* ctx, BeaconStream** out_stream);
void    beacon_stream_destroy(BeaconStream* stream);

// === Tensor creation and lifecycle ===
// Creates a tensor from an existing memory-mapped buffer WITHOUT copying.
// The caller must keep the buffer alive for the tensor's lifetime.
int32_t beacon_tensor_from_mmap(
    BeaconContext* ctx,
    const void* data,
    const int64_t* shape, size_t ndim,
    BeaconDtype dtype,
    BeaconTensor** out_tensor);

// Creates a zero-initialized tensor in unified memory.
int32_t beacon_tensor_zeros(
    BeaconContext* ctx,
    const int64_t* shape, size_t ndim,
    BeaconDtype dtype,
    BeaconTensor** out_tensor);

void    beacon_tensor_destroy(BeaconTensor* tensor);

// === Tensor introspection ===
// On entry, *out_ndim is the capacity of out_shape (element count).
// On return, *out_ndim is set to the actual ndim; if the provided capacity was
// smaller than the tensor's ndim, returns BEACON_ERR_INVALID_ARGUMENT.
int32_t beacon_tensor_shape(const BeaconTensor* t, int64_t* out_shape, size_t* out_ndim);
int32_t beacon_tensor_dtype(const BeaconTensor* t, BeaconDtype* out_dtype);

// Force evaluation of lazy MLX graph for this tensor.
// Called at decode boundaries where eager semantics are required.
int32_t beacon_tensor_eval(BeaconTensor* t, BeaconStream* stream);

// Copy tensor contents to a host-provided buffer. Triggers eval first.
// Used for sampling (reading logits) and debugging.
int32_t beacon_tensor_read_f32(const BeaconTensor* t, float* out, size_t n_elements);

// === Ops (the subset we need) ===
int32_t beacon_op_matmul(
    BeaconContext* ctx, BeaconStream* stream,
    const BeaconTensor* a, const BeaconTensor* b,
    BeaconTensor** out);

int32_t beacon_op_quantized_matmul(
    BeaconContext* ctx, BeaconStream* stream,
    const BeaconTensor* x, const BeaconTensor* w_quantized,
    const BeaconTensor* scales, const BeaconTensor* biases,  // biases nullable
    int32_t group_size, int32_t bits,
    BeaconTensor** out);

int32_t beacon_op_rms_norm(
    BeaconContext* ctx, BeaconStream* stream,
    const BeaconTensor* x, const BeaconTensor* weight,
    float eps,
    BeaconTensor** out);

int32_t beacon_op_rope(
    BeaconContext* ctx, BeaconStream* stream,
    const BeaconTensor* x,
    int32_t position_offset, float theta, int32_t dim,
    BeaconTensor** out);

int32_t beacon_op_silu(
    BeaconContext* ctx, BeaconStream* stream,
    const BeaconTensor* x,
    BeaconTensor** out);

int32_t beacon_op_softmax(
    BeaconContext* ctx, BeaconStream* stream,
    const BeaconTensor* x, int32_t axis,
    BeaconTensor** out);

// Fused scaled dot-product attention with GQA support.
// q: [batch, n_heads, seq_len, head_dim]
// k: [batch, n_kv_heads, kv_seq_len, head_dim]
// v: [batch, n_kv_heads, kv_seq_len, head_dim]
int32_t beacon_op_attention(
    BeaconContext* ctx, BeaconStream* stream,
    const BeaconTensor* q, const BeaconTensor* k, const BeaconTensor* v,
    const BeaconTensor* mask,  // nullable
    float scale,
    BeaconTensor** out);

int32_t beacon_op_add(
    BeaconContext* ctx, BeaconStream* stream,
    const BeaconTensor* a, const BeaconTensor* b,
    BeaconTensor** out);

int32_t beacon_op_elementwise_mul(
    BeaconContext* ctx, BeaconStream* stream,
    const BeaconTensor* a, const BeaconTensor* b,
    BeaconTensor** out);

// === KV cache primitives ===
// The Rust scheduler owns cache lifecycle. The shim provides update primitives.
// cache_k shape: [max_context, num_kv_heads, head_dim]
// cache_v shape: [max_context, num_kv_heads, head_dim]
// new_k shape:   [1,           num_kv_heads, head_dim]
// new_v shape:   [1,           num_kv_heads, head_dim]
// Updates cache[position] with new values. Returns views over [0..=position]
// for attention to consume.
int32_t beacon_op_kv_cache_update(
    BeaconContext* ctx, BeaconStream* stream,
    BeaconTensor* cache_k, BeaconTensor* cache_v,
    const BeaconTensor* new_k, const BeaconTensor* new_v,
    int64_t position,
    BeaconTensor** out_k_view, BeaconTensor** out_v_view);

// === Tensor manipulation ===
// Reshape a tensor to a new shape (same total element count).
int32_t beacon_op_reshape(
    BeaconContext* ctx, BeaconStream* stream,
    const BeaconTensor* x,
    const int64_t* new_shape, size_t new_ndim,
    BeaconTensor** out);

// Transpose the last two dimensions of a tensor.
int32_t beacon_op_transpose(
    BeaconContext* ctx, BeaconStream* stream,
    const BeaconTensor* x,
    BeaconTensor** out);

// Swap two axes of a tensor.
int32_t beacon_op_swapaxes(
    BeaconContext* ctx, BeaconStream* stream,
    const BeaconTensor* x,
    int32_t axis1, int32_t axis2,
    BeaconTensor** out);

// Embedding lookup: select rows from a weight matrix by index.
// weight: [vocab_size, hidden_dim], indices: [seq_len] (int32)
// out: [seq_len, hidden_dim]
int32_t beacon_op_embedding(
    BeaconContext* ctx, BeaconStream* stream,
    const BeaconTensor* weight, const BeaconTensor* indices,
    BeaconTensor** out);

// === Quantization ===
// Quantize a float matrix into MLX's internal quantized format.
// Returns three tensors: packed weights (uint32), scales (float16), biases (float16).
// The input tensor w must be float16 or float32 with shape [rows, cols].
int32_t beacon_op_quantize(
    BeaconContext* ctx, BeaconStream* stream,
    const BeaconTensor* w,
    int32_t group_size, int32_t bits,
    BeaconTensor** out_packed, BeaconTensor** out_scales, BeaconTensor** out_biases);

// === GGUF dequantization ===
// Dequantize raw GGUF quantized bytes to an F32 tensor using gguflib.
// gguf_type: GGUF tensor type ID (2=Q4_0, 8=Q8_0, 12=Q4_K, etc.)
// data: raw quantized bytes
// data_len: byte length of the data
// shape: desired output shape (row-major)
// ndim: number of dimensions
int32_t beacon_op_dequantize_gguf(
    BeaconContext* ctx, BeaconStream* stream,
    const void* data, size_t data_len,
    int32_t gguf_type, uint64_t num_elements,
    const int64_t* shape, size_t ndim,
    BeaconTensor** out);

// === Custom kernels (Metal) ===
// Used where MLX's built-in ops are insufficient.
// NOTE: Step 2 ships a stub; the real Q4 dequant+matmul kernel lands in Step 6 / v0.2.
int32_t beacon_kernel_q4_dequant_mul(
    BeaconContext* ctx, BeaconStream* stream,
    const BeaconTensor* x, const BeaconTensor* w_q4, const BeaconTensor* scales,
    BeaconTensor** out);

// === Error introspection ===
// Returns human-readable error message for the most recent error on this thread.
// Caller does not free the returned pointer; it's valid until the next shim call
// on this thread.
const char* beacon_last_error_message(void);

#ifdef __cplusplus
}
#endif

#endif  // BEACON_SHIM_H
