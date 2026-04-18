# Beacon Architecture

This document specifies how Beacon is built. It is the authoritative reference for all implementation decisions. When the README says *what*, this document says *how*.

**Product name:** Beacon (distributed as `beacon-ai` on package registries).
**C++ layer:** the `shim`. Exactly three responsibilities, tightly bounded.

---

## 1. Design Principles

These principles resolve ambiguity. When in doubt, apply them in order.

1. **Rust for everything except MLX.** The shim exists only because MLX is C++. Every line of code that doesn't directly touch MLX or Metal Shading Language belongs in Rust.
2. **Zero-copy by default.** Apple Silicon unified memory means weights in RAM are already accessible to the GPU. Beacon never copies tensors between "host" and "device" — that distinction does not exist on the primary platform.
3. **Explicit over implicit.** MLX uses lazy evaluation; Rust is eager. The boundary between them is marked by explicit `.eval()` calls. No hidden materialization.
4. **One backend selection per engine instance.** The backend (MLX, CPU, CUDA) is chosen at `Engine::load()` time and does not change. No per-op dispatch overhead.
5. **Fail loud, fail early.** Model loading, backend selection, and kernel dispatch validate aggressively at startup. The hot inference loop has no runtime checks beyond what the hardware already enforces.
6. **Shim stays small.** Target: shim under 2,000 lines of C++ across the entire v1 product. Every function in the shim justifies its existence.

---

## 2. Repository Layout (authoritative)

```
beacon/
├── Cargo.toml                       # Workspace root
├── rust-toolchain.toml              # Pin to stable Rust
├── crates/
│   ├── beacon-core/                 # Engine orchestration, model loading
│   ├── beacon-mlx/                  # Rust FFI bindings to the shim
│   ├── beacon-kernels/              # CPU SIMD kernels (secondary backend)
│   ├── beacon-cuda/                 # CUDA backend (tertiary, feature-gated)
│   ├── beacon-format/               # .beacon format + GGUF import
│   ├── beacon-tokenizer/            # BPE tokenizer
│   ├── beacon-scheduler/            # Batching, streaming, KV cache
│   ├── beacon-arrow/                # Arrow-native I/O
│   ├── beacon-registry/             # Model pull/cache/verify
│   └── beacon-cli/                  # CLI binary
├── bindings/
│   ├── python/                      # PyO3 bindings → beacon-ai on PyPI
│   └── node/                        # napi-rs bindings → beacon-ai on npm
├── server/
│   └── beacon-server/               # Ollama-compatible HTTP server
├── shim/                            # C++ MLX shim (SMALL)
│   ├── CMakeLists.txt
│   ├── include/
│   │   └── beacon_shim.h            # C ABI header (generated bindgen input)
│   ├── src/
│   │   ├── beacon_shim.cpp          # C ABI implementation
│   │   ├── tensor.cpp               # Tensor lifecycle
│   │   ├── ops.cpp                  # MLX op invocations
│   │   ├── kernels.cpp              # Custom Metal kernel hosting
│   │   └── errors.cpp               # Error translation
│   ├── metal/
│   │   ├── q4_dequant_mul.metal     # Custom Metal kernel: Q4 dequant + matmul
│   │   ├── rope.metal               # Rotary embedding (if fused version needed)
│   │   └── sparse_ffn.metal         # Sparse FFN (v0.2)
│   └── third_party/
│       └── mlx/                     # Git submodule, pinned version
├── benches/
├── tests/
│   ├── integration/
│   └── eval/
├── scripts/
├── docs/
│   ├── architecture.md              # (this file)
│   ├── mlx-integration.md           # Shim API reference
│   ├── performance.md               # Benchmark methodology
│   └── contributing.md
└── README.md
```

---

## 3. The Shim (C++ Layer)

The shim is the single concession to C++. It exists because MLX is C++ and because Metal Shading Language integrates naturally only with C++. Everything else is Rust.

### 3.1 Shim Responsibilities (exhaustive list)

The shim does exactly these three things. If work doesn't fit one of these, it belongs in Rust.

1. **Expose MLX primitives** Beacon needs, through a stable C ABI.
2. **Host custom Metal kernels** Beacon writes for hot paths MLX doesn't cover.
3. **Translate errors and lifetimes** from C++ exceptions and RAII to C-compatible return codes and explicit handles.

### 3.2 Shim Non-Responsibilities

The shim does NOT:

- Parse model files. (Rust: `beacon-format`)
- Tokenize text. (Rust: `beacon-tokenizer`)
- Manage the KV cache lifecycle. (Rust: `beacon-scheduler`, allocating MLX tensors via shim)
- Implement sampling. (Rust: `beacon-scheduler`)
- Make backend selection decisions. (Rust: `beacon-core`)
- Handle HTTP, CLI, or bindings. (Rust: server, CLI, bindings crates)
- Track model metadata, configs, or registry state. (Rust: `beacon-registry`, `beacon-core`)

If a PR grows the shim to handle any of the above, reject it.

### 3.3 C ABI Surface (beacon_shim.h)

The shim exposes a C ABI with opaque handles. Rust consumes this via `bindgen`.

```c
// beacon_shim.h
// All functions return int32_t status code. 0 = success, non-zero = error.
// Output parameters are the final argument.

#ifndef BEACON_SHIM_H
#define BEACON_SHIM_H

#include <stdint.h>
#include <stddef.h>

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
    BEACON_ERR_UNKNOWN = 99,
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
    BEACON_DTYPE_Q8_0 = 104,
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
    BeaconTensor** out_tensor
);

// Creates a zero-initialized tensor in unified memory.
int32_t beacon_tensor_zeros(
    BeaconContext* ctx,
    const int64_t* shape, size_t ndim,
    BeaconDtype dtype,
    BeaconTensor** out_tensor
);

void    beacon_tensor_destroy(BeaconTensor* tensor);

// === Tensor introspection ===
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
    BeaconTensor** out
);

int32_t beacon_op_quantized_matmul(
    BeaconContext* ctx, BeaconStream* stream,
    const BeaconTensor* x, const BeaconTensor* w_quantized, const BeaconTensor* scales,
    int32_t group_size, int32_t bits,
    BeaconTensor** out
);

int32_t beacon_op_rms_norm(
    BeaconContext* ctx, BeaconStream* stream,
    const BeaconTensor* x, const BeaconTensor* weight,
    float eps,
    BeaconTensor** out
);

int32_t beacon_op_rope(
    BeaconContext* ctx, BeaconStream* stream,
    const BeaconTensor* x,
    int32_t position_offset, float theta, int32_t dim,
    BeaconTensor** out
);

int32_t beacon_op_silu(
    BeaconContext* ctx, BeaconStream* stream,
    const BeaconTensor* x,
    BeaconTensor** out
);

int32_t beacon_op_softmax(
    BeaconContext* ctx, BeaconStream* stream,
    const BeaconTensor* x, int32_t axis,
    BeaconTensor** out
);

// Fused scaled dot-product attention with GQA support.
// q: [batch, n_heads, seq_len, head_dim]
// k: [batch, n_kv_heads, kv_seq_len, head_dim]
// v: [batch, n_kv_heads, kv_seq_len, head_dim]
int32_t beacon_op_attention(
    BeaconContext* ctx, BeaconStream* stream,
    const BeaconTensor* q, const BeaconTensor* k, const BeaconTensor* v,
    const BeaconTensor* mask,  // nullable
    float scale,
    BeaconTensor** out
);

int32_t beacon_op_add(
    BeaconContext* ctx, BeaconStream* stream,
    const BeaconTensor* a, const BeaconTensor* b,
    BeaconTensor** out
);

int32_t beacon_op_elementwise_mul(
    BeaconContext* ctx, BeaconStream* stream,
    const BeaconTensor* a, const BeaconTensor* b,
    BeaconTensor** out
);

// === KV cache primitives ===
// The Rust scheduler owns cache lifecycle. The shim provides update primitives.
int32_t beacon_op_kv_cache_update(
    BeaconContext* ctx, BeaconStream* stream,
    BeaconTensor* cache_k, BeaconTensor* cache_v,
    const BeaconTensor* new_k, const BeaconTensor* new_v,
    int64_t position,
    BeaconTensor** out_k_view, BeaconTensor** out_v_view
);

// === Custom kernels (Metal) ===
// Used where MLX's built-in ops are insufficient.
int32_t beacon_kernel_q4_dequant_mul(
    BeaconContext* ctx, BeaconStream* stream,
    const BeaconTensor* x, const BeaconTensor* w_q4, const BeaconTensor* scales,
    BeaconTensor** out
);

// === Error introspection ===
// Returns human-readable error message for the most recent error on this thread.
// Caller does not free the returned pointer; it's valid until the next shim call.
const char* beacon_last_error_message(void);

#ifdef __cplusplus
}
#endif

#endif // BEACON_SHIM_H
```

**Surface size:** approximately 20 functions. This is the complete list for v0.1. Every addition requires justification.

### 3.4 Shim Implementation Rules

1. **No C++ exceptions cross the ABI.** All C++ exceptions caught at the C ABI boundary and translated to `BeaconStatus` codes + thread-local error message.
2. **No Rust-facing headers include C++ types.** The shim header is pure C. C++ internals stay in `.cpp` files.
3. **Handles are opaque.** Rust never dereferences a `BeaconTensor*`. It passes it back to the shim for every operation.
4. **Lifetimes are explicit.** Every `_create` or `_from_*` function has a corresponding `_destroy`. No reference counting across the ABI.
5. **Streams are explicit.** All ops take a `BeaconStream*` parameter. The scheduler decides whether ops serialize or overlap.
6. **Metal kernel sources are embedded as strings** in the shim binary (via CMake `configure_file` or `xxd -i`). Runtime kernel compilation happens once at `beacon_context_create`, cached for the process lifetime.

### 3.5 Shim Build System

```cmake
# shim/CMakeLists.txt (skeleton)
cmake_minimum_required(VERSION 3.24)
project(beacon_shim CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# MLX as subdirectory (submodule at third_party/mlx)
add_subdirectory(third_party/mlx EXCLUDE_FROM_ALL)

add_library(beacon_shim STATIC
    src/beacon_shim.cpp
    src/tensor.cpp
    src/ops.cpp
    src/kernels.cpp
    src/errors.cpp
)

target_include_directories(beacon_shim
    PUBLIC include
    PRIVATE src
)

target_link_libraries(beacon_shim
    PRIVATE mlx
)

# Embed Metal kernel source as C string
file(READ metal/q4_dequant_mul.metal Q4_KERNEL_SRC)
configure_file(src/kernels.cpp.in src/kernels.cpp @ONLY)

# On non-Apple platforms, build without MLX (CPU + CUDA backends only)
if(NOT APPLE)
    target_compile_definitions(beacon_shim PRIVATE BEACON_NO_MLX)
endif()
```

The `beacon-mlx` Rust crate builds the shim via `cc` or `cmake` crate in `build.rs`.

---

## 4. Rust FFI Layer (beacon-mlx)

The `beacon-mlx` crate is the Rust-side bridge. It wraps the C ABI in safe Rust types.

### 4.1 Type Hierarchy

```rust
// crates/beacon-mlx/src/lib.rs

use std::sync::Arc;
use std::marker::PhantomData;

/// A shim context. Thread-safe. Create once per process.
pub struct MlxContext {
    inner: *mut ffi::BeaconContext,
}

unsafe impl Send for MlxContext {}
unsafe impl Sync for MlxContext {}

impl MlxContext {
    pub fn new() -> Result<Arc<Self>, MlxError> { /* ... */ }

    pub fn new_stream(self: &Arc<Self>) -> Result<MlxStream, MlxError> { /* ... */ }
}

impl Drop for MlxContext {
    fn drop(&mut self) {
        unsafe { ffi::beacon_context_destroy(self.inner); }
    }
}

/// A stream. Ops scheduled on the same stream serialize; different streams may overlap.
pub struct MlxStream {
    inner: *mut ffi::BeaconStream,
    ctx: Arc<MlxContext>,
}

/// A tensor. Owns its handle; dropping releases shim resources.
pub struct MlxTensor {
    inner: *mut ffi::BeaconTensor,
    ctx: Arc<MlxContext>,
    // Keep backing mmap alive if tensor was created from mmap
    _backing: Option<Arc<memmap2::Mmap>>,
}

impl MlxTensor {
    pub fn from_mmap(
        ctx: Arc<MlxContext>,
        mmap: Arc<memmap2::Mmap>,
        offset: usize,
        shape: &[i64],
        dtype: Dtype,
    ) -> Result<Self, MlxError> { /* ... */ }

    pub fn zeros(ctx: Arc<MlxContext>, shape: &[i64], dtype: Dtype) -> Result<Self, MlxError> { /* ... */ }

    pub fn shape(&self) -> Vec<i64> { /* ... */ }
    pub fn dtype(&self) -> Dtype { /* ... */ }

    /// Force evaluation of the lazy graph for this tensor.
    pub fn eval(&self, stream: &MlxStream) -> Result<(), MlxError> { /* ... */ }

    /// Read values to host buffer (triggers eval). For sampling and debugging only.
    pub fn read_f32(&self) -> Result<Vec<f32>, MlxError> { /* ... */ }
}

impl Drop for MlxTensor {
    fn drop(&mut self) {
        unsafe { ffi::beacon_tensor_destroy(self.inner); }
    }
}
```

### 4.2 Ops as Free Functions

Ops are exposed as free functions in an `ops` module, taking `&MlxStream` and `&MlxTensor` arguments. This matches MLX's Python API ergonomically.

```rust
// crates/beacon-mlx/src/ops.rs

pub fn matmul(
    stream: &MlxStream,
    a: &MlxTensor,
    b: &MlxTensor,
) -> Result<MlxTensor, MlxError> { /* ... */ }

pub fn quantized_matmul(
    stream: &MlxStream,
    x: &MlxTensor,
    w: &MlxTensor,
    scales: &MlxTensor,
    group_size: i32,
    bits: i32,
) -> Result<MlxTensor, MlxError> { /* ... */ }

pub fn rms_norm(
    stream: &MlxStream,
    x: &MlxTensor,
    weight: &MlxTensor,
    eps: f32,
) -> Result<MlxTensor, MlxError> { /* ... */ }

pub fn rope(
    stream: &MlxStream,
    x: &MlxTensor,
    position_offset: i32,
    theta: f32,
    dim: i32,
) -> Result<MlxTensor, MlxError> { /* ... */ }

pub fn attention(
    stream: &MlxStream,
    q: &MlxTensor,
    k: &MlxTensor,
    v: &MlxTensor,
    mask: Option<&MlxTensor>,
    scale: f32,
) -> Result<MlxTensor, MlxError> { /* ... */ }

// ... and so on, matching the shim surface
```

### 4.3 Error Handling

```rust
#[derive(Debug, thiserror::Error)]
pub enum MlxError {
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("out of memory")]
    OutOfMemory,
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),
    #[error("unsupported dtype")]
    UnsupportedDtype,
    #[error("Metal kernel compilation failed: {0}")]
    MetalCompile(String),
    #[error("MLX internal error: {0}")]
    MlxInternal(String),
    #[error("unknown error: {0}")]
    Unknown(String),
}

fn status_to_result(status: i32) -> Result<(), MlxError> {
    if status == 0 { return Ok(()); }
    let msg = unsafe {
        let ptr = ffi::beacon_last_error_message();
        if ptr.is_null() { String::from("<no message>") }
        else { std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned() }
    };
    Err(match status {
        1 => MlxError::InvalidArgument(msg),
        2 => MlxError::OutOfMemory,
        3 => MlxError::ShapeMismatch(msg),
        4 => MlxError::UnsupportedDtype,
        5 => MlxError::MetalCompile(msg),
        6 => MlxError::MlxInternal(msg),
        _ => MlxError::Unknown(msg),
    })
}
```

### 4.4 Lazy vs Eager

MLX is lazy; Rust callers expect eager-ish semantics. The rule:

- Tensor operations return immediately with unevaluated handles.
- `eval()` is called explicitly at specific points in the forward pass.
- In v0.1, `eval()` is called **once per transformer layer** to bound graph size. This can be tuned in v0.2 (bigger fusion windows = better perf, higher latency to first token).
- `read_f32()` implicitly evals before copying.

---

## 5. Backend Selection

```rust
// crates/beacon-core/src/backend.rs

pub enum Backend {
    Mlx(Arc<MlxContext>),
    Cpu(CpuContext),
    Cuda(CudaContext),  // feature-gated
}

impl Backend {
    pub fn auto_select() -> Result<Self, BackendError> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            if let Ok(ctx) = MlxContext::new() {
                return Ok(Backend::Mlx(ctx));
            }
        }

        #[cfg(feature = "cuda")]
        {
            if CudaContext::is_available() {
                return Ok(Backend::Cuda(CudaContext::new()?));
            }
        }

        Ok(Backend::Cpu(CpuContext::new()?))
    }

    pub fn force(name: &str) -> Result<Self, BackendError> {
        // Honors BEACON_BACKEND env var and --backend CLI flag
    }
}
```

Selection happens **once** at `Engine::load()`. The `Engine` carries its backend through its lifetime. No per-op dispatch.

The forward pass is written generically against a `ComputeBackend` trait:

```rust
pub trait ComputeBackend {
    type Tensor;
    type Stream;

    fn new_stream(&self) -> Self::Stream;
    fn matmul(&self, stream: &Self::Stream, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, BackendError>;
    fn quantized_matmul(&self, /* ... */) -> Result<Self::Tensor, BackendError>;
    fn rms_norm(&self, /* ... */) -> Result<Self::Tensor, BackendError>;
    fn rope(&self, /* ... */) -> Result<Self::Tensor, BackendError>;
    fn attention(&self, /* ... */) -> Result<Self::Tensor, BackendError>;
    fn silu(&self, /* ... */) -> Result<Self::Tensor, BackendError>;
    fn softmax(&self, /* ... */) -> Result<Self::Tensor, BackendError>;
    fn add(&self, /* ... */) -> Result<Self::Tensor, BackendError>;
    fn mul(&self, /* ... */) -> Result<Self::Tensor, BackendError>;
    fn eval(&self, t: &Self::Tensor, stream: &Self::Stream) -> Result<(), BackendError>;
}
```

`beacon-mlx`, `beacon-kernels`, and `beacon-cuda` each implement this trait. The forward pass is monomorphized per backend at compile time, so there is zero runtime dispatch cost.

---

## 6. Memory Model

### 6.1 Unified Memory (MLX Backend)

On Apple Silicon, all tensors live in unified memory. There is no host/device distinction.

- **Weight tensors:** memory-mapped directly from `.beacon` files via `memmap2`. Shim receives the pointer and wraps it as an MLX tensor without copying.
- **KV cache:** allocated in unified memory at engine load time, sized to `max_context_length`. Persists for the lifetime of the engine.
- **Activation tensors:** allocated lazily by MLX during graph execution. Freed automatically when no longer referenced.
- **Logits:** the single point where data is copied to a CPU-owned buffer, for sampling in Rust.

### 6.2 CPU Backend

On CPU-only systems, the memory model is:

- **Weight tensors:** `mmap`'d, read directly by SIMD kernels.
- **KV cache:** preallocated `Vec<u8>` buffer, indexed by layer/head/position.
- **Activations:** stack-allocated where possible, heap-allocated via a simple arena for transformer layer scope.

### 6.3 Cross-Backend Rule

Tensors never cross backends. If a user forces the CPU backend on an Apple Silicon machine, all tensors are CPU tensors. No mixed execution.

---

## 7. Forward Pass Architecture

All four supported model families (Qwen 2.5, Llama 3.x, Phi-3.x, Gemma 2) share a Llama-style transformer architecture with minor config variations. One forward pass implementation, parameterized by config.

### 7.1 Model Config

```rust
// crates/beacon-core/src/config.rs

#[derive(Clone, Debug, serde::Deserialize)]
pub struct ModelConfig {
    pub architecture: Architecture,       // Llama | Qwen | Phi | Gemma
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,              // GQA: ≤ num_heads
    pub intermediate_size: usize,         // FFN hidden dim
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub rope_scaling: Option<RopeScaling>,
    pub rms_norm_eps: f32,
    pub tie_word_embeddings: bool,
    pub bos_token_id: Option<u32>,
    pub eos_token_ids: Vec<u32>,
    pub chat_template: Option<String>,
}
```

### 7.2 Transformer Block Structure

```
Input (hidden_state) ─┬──────────────────┐
                      │                  │
                      ▼                  │
                RMSNorm (attn_norm)      │
                      │                  │
                      ▼                  │
                 Attention Block         │
                (q_proj, k_proj,         │
                 v_proj, RoPE,           │
                 KV cache update,        │
                 SDPA, o_proj)           │
                      │                  │
                      ▼                  │
                  Residual Add  ◄────────┘
                      │
                      ├──────────────────┐
                      │                  │
                      ▼                  │
                RMSNorm (ffn_norm)       │
                      │                  │
                      ▼                  │
                   FFN Block             │
                (gate_proj, up_proj,     │
                 SiLU, elementwise mul,  │
                 down_proj)              │
                      │                  │
                      ▼                  │
                  Residual Add  ◄────────┘
                      │
                      ▼
                 (to next layer)
```

### 7.3 Attention Block (pseudocode)

```rust
fn attention_block<B: ComputeBackend>(
    backend: &B,
    stream: &B::Stream,
    hidden: &B::Tensor,           // [batch, seq, hidden]
    weights: &AttentionWeights<B>,
    cache: &mut KvCache<B>,
    position: usize,
    cfg: &ModelConfig,
) -> Result<B::Tensor, Error> {
    // Project Q, K, V
    let q = backend.quantized_matmul(stream, hidden, &weights.q_proj, ...)?;
    let k = backend.quantized_matmul(stream, hidden, &weights.k_proj, ...)?;
    let v = backend.quantized_matmul(stream, hidden, &weights.v_proj, ...)?;

    // Reshape for multi-head: q → [batch, n_heads, seq, head_dim]
    //                        k,v → [batch, n_kv_heads, seq, head_dim]

    // RoPE on Q and K
    let q = backend.rope(stream, &q, position as i32, cfg.rope_theta, cfg.head_dim as i32)?;
    let k = backend.rope(stream, &k, position as i32, cfg.rope_theta, cfg.head_dim as i32)?;

    // Update KV cache, get full-length views for attention
    let (k_full, v_full) = cache.update(backend, stream, k, v, position)?;

    // Scaled dot-product attention (fused in MLX backend, manual in CPU backend)
    let scale = 1.0 / (cfg.head_dim as f32).sqrt();
    let attn_out = backend.attention(stream, &q, &k_full, &v_full, None, scale)?;

    // Output projection
    let out = backend.quantized_matmul(stream, &attn_out, &weights.o_proj, ...)?;
    Ok(out)
}
```

**GQA correctness note:** when `num_kv_heads < num_heads`, the attention kernel must broadcast K and V across query head groups. In the MLX backend, `beacon_op_attention` handles this natively. In the CPU backend, this is implemented by repeating K/V reads per head group (no materialization of expanded tensors).

### 7.4 FFN Block (pseudocode)

All four model families use SwiGLU (gate × up, SiLU activation on gate):

```rust
fn ffn_block<B: ComputeBackend>(
    backend: &B, stream: &B::Stream,
    hidden: &B::Tensor,
    weights: &FfnWeights<B>,
) -> Result<B::Tensor, Error> {
    let gate = backend.quantized_matmul(stream, hidden, &weights.gate_proj, ...)?;
    let up = backend.quantized_matmul(stream, hidden, &weights.up_proj, ...)?;
    let gate = backend.silu(stream, &gate)?;
    let inter = backend.mul(stream, &gate, &up)?;
    backend.quantized_matmul(stream, &inter, &weights.down_proj, ...)
}
```

### 7.5 Full Forward Pass

```rust
fn forward<B: ComputeBackend>(
    engine: &Engine<B>,
    tokens: &[u32],
    position: usize,
) -> Result<B::Tensor, Error> {
    let stream = engine.backend.new_stream();
    let mut h = embed_tokens(&engine, &stream, tokens)?;

    for (i, layer) in engine.layers.iter().enumerate() {
        let normed = engine.backend.rms_norm(&stream, &h, &layer.attn_norm, engine.cfg.rms_norm_eps)?;
        let attn = attention_block(&engine.backend, &stream, &normed, &layer.attn, &mut engine.cache[i], position, &engine.cfg)?;
        h = engine.backend.add(&stream, &h, &attn)?;

        let normed = engine.backend.rms_norm(&stream, &h, &layer.ffn_norm, engine.cfg.rms_norm_eps)?;
        let ffn = ffn_block(&engine.backend, &stream, &normed, &layer.ffn)?;
        h = engine.backend.add(&stream, &h, &ffn)?;

        // Eval at layer boundary: bounds MLX graph size, makes debugging easier
        engine.backend.eval(&h, &stream)?;
    }

    let h = engine.backend.rms_norm(&stream, &h, &engine.final_norm, engine.cfg.rms_norm_eps)?;
    let logits = engine.backend.matmul(&stream, &h, &engine.lm_head)?;
    engine.backend.eval(&logits, &stream)?;
    Ok(logits)
}
```

---

## 8. KV Cache Design

### 8.1 Layout

KV cache is per-layer, contiguous, preallocated at engine load:

```
cache[layer] = {
    k: Tensor [max_context, num_kv_heads, head_dim],
    v: Tensor [max_context, num_kv_heads, head_dim],
    current_length: usize,
}
```

### 8.2 Update

On each decode step, new K and V for the current token are written at `cache[layer].k[current_length]` and `cache[layer].v[current_length]`, then `current_length` is incremented. Attention reads the slice `[0..current_length]`.

**No reallocation during generation.** If `current_length` would exceed `max_context`, the engine returns an error. Context extension (sliding window, KV eviction) is a v0.2 feature.

### 8.3 Quantization (v0.2)

KV cache quantization to 2-bit is additive: the cache stores quantized K/V, and the attention op dequantizes on read inside the fused kernel. This changes the cache memory footprint by 4x but does not change the attention op API as seen by the forward pass.

---

## 9. Model Format (.beacon)

### 9.1 Header Layout

```
┌─────────────────────────────────────┐
│ Magic: "BCN1"             (4 bytes) │
│ Version: u32              (4 bytes) │
│ Header size: u64          (8 bytes) │
│ Tensor count: u64         (8 bytes) │
├─────────────────────────────────────┤
│ Config JSON (length-prefixed)       │
│   - ModelConfig fields              │
├─────────────────────────────────────┤
│ Tokenizer JSON (length-prefixed)    │
│   - BPE vocab, merges, special      │
├─────────────────────────────────────┤
│ Tensor metadata table               │
│   For each tensor:                  │
│     - name (length-prefixed UTF-8)  │
│     - dtype (u32)                   │
│     - shape (u32 ndim + u64[] dims) │
│     - data offset (u64)             │
│     - data length (u64)             │
├─────────────────────────────────────┤
│ Tensor data (64-byte aligned)       │
│   Stored consecutively.             │
└─────────────────────────────────────┘
```

Memory-map the file, parse the header eagerly, keep tensor data zero-copy.

### 9.2 GGUF Import

First load of a GGUF file converts to `.beacon` and caches at `~/.beacon/models/<name>/model.beacon`. Subsequent loads skip the conversion. Conversion is straightforward: remap GGUF quantization variants to shim dtypes, copy tensors with new alignment.

---

## 10. Tokenizer (beacon-tokenizer)

Rust-native BPE, matching Hugging Face `tokenizers` exactly. Reference: `tokenizers` crate (Hugging Face's Rust implementation) — Beacon wraps it with chat template rendering via `minijinja`.

Success criterion: token-for-token match with `tokenizers` Python library on 10k samples per model family. Do not ship without this.

---

## 11. Scheduler (beacon-scheduler)

### 11.1 Responsibilities

- Own KV cache lifecycle.
- Drive the forward pass loop.
- Apply sampling.
- Emit tokens to consumers via `tokio::sync::mpsc` channels.
- Enforce stop conditions (stop tokens, max tokens, timeout, cancellation).

### 11.2 Sampling

Implemented in pure Rust on CPU over f32 logits. Order:

1. Apply repeat penalty
2. Apply temperature
3. Apply top-k (if set)
4. Apply top-p (if set)
5. Apply min-p (if set)
6. Sample via multinomial from softmax

### 11.3 Streaming

`engine.stream(prompt)` returns `Stream<Item = Result<Token, Error>>`. Internally: spawn a tokio task that runs decode loop, emits tokens, closes channel on stop condition or error.

---

## 12. Bindings

### 12.1 Python (PyO3, maturin)

Published as `beacon-ai` on PyPI. Surface mirrors the Rust API:

```python
import beacon_ai as beacon

engine = beacon.Engine.load("qwen2.5-7b-q4")
for token in engine.stream("Explain unified memory."):
    print(token, end="", flush=True)
```

Prebuilt wheels for macOS ARM64 (priority), macOS x86_64, Linux x86_64, Linux aarch64. No Windows wheels in v0.1; source distribution only.

### 12.2 Node (napi-rs)

Published as `beacon-ai` on npm. TypeScript definitions included.

### 12.3 CLI (beacon-cli)

Binary named `beacon`. Commands: `pull`, `run`, `list`, `remove`, `info`, `serve`.

### 12.4 HTTP Server (beacon-server)

Axum-based. Ollama-compatible endpoints: `/api/generate`, `/api/chat`, `/api/tags`, `/api/pull`. NDJSON streaming.

---

## 13. Testing Strategy

### 13.1 Unit Tests

Per-crate, in-crate (`#[cfg(test)]` modules). Target: 70% coverage on non-FFI code.

### 13.2 Integration Tests

`tests/integration/` runs end-to-end scenarios: load model, generate text, verify outputs match reference.

### 13.3 Correctness Tests

`tests/eval/` runs:
- Logit-level exactness: for a fixed prompt at T=0, first token logits must match Hugging Face reference to 3 decimal places.
- MMLU subset (1k questions).
- HumanEval subset (50 problems).
- Perplexity on held-out Wikipedia slice.

Ember output on these benchmarks must be within 0.5% of the Hugging Face reference for each supported model.

### 13.4 Performance Tests

`benches/` runs Criterion benchmarks per op and end-to-end. Regression detection on every PR (GitHub Actions).

---

## 14. Build and Distribution

### 14.1 Local Development Build

```bash
# First time
git clone --recurse-submodules <repo>
cd beacon
cargo build --release --workspace

# Incremental (Rust only)
cargo build --release -p beacon-core

# Shim rebuild (after editing shim/ or updating MLX submodule)
cargo build --release -p beacon-mlx --features rebuild-shim
```

### 14.2 CI

- macOS ARM64: primary, runs all tests including MLX.
- macOS x86_64: build only, no MLX tests.
- Linux x86_64: build + CPU tests + CUDA tests (feature-gated).
- Linux aarch64: build + CPU tests.
- Windows x86_64: build only.

### 14.3 Distribution

- Rust crate: `cargo publish` for each public crate (beacon-core, beacon-mlx, etc.).
- CLI binary: GitHub Releases with macOS ARM64, macOS x86_64, Linux x86_64 tarballs.
- Python: wheels via maturin-action, uploaded to PyPI.
- Node: prebuilt binaries via napi-rs, uploaded to npm.
- Install script: `curl -fsSL https://beacon.ai/install.sh | sh` (future).

---

## 15. Non-Negotiables

These rules are load-bearing. Violating them means the architecture no longer holds.

1. **Shim stays under 2,000 lines.** Measured across all `.cpp` and `.h` files in `shim/src/` and `shim/include/`.
2. **No Rust-to-C++ exceptions.** All errors translate at the ABI boundary.
3. **No tensor copies on MLX backend.** Weights are mmap'd, KV cache is allocated once, activations are MLX-managed. If you find yourself calling `memcpy` on a weight tensor, stop.
4. **No cross-backend tensors.** One backend per engine, chosen at load time.
5. **No blocking ops in the decode hot path.** Model I/O, file parsing, allocation must all happen at load time. Decode calls into MLX/CPU kernels only.
6. **Logit-level tests pass before any optimization.** Never optimize a broken implementation.

---

## 16. Extension Points (v0.2+)

These are deliberately left open for v0.2 without architectural refactoring:

- **Contextual sparsity:** adds FFN neuron predictor MLP, modifies `ffn_block` to dispatch sparse or dense path. Shim adds `beacon_op_sparse_ffn`.
- **Speculative decoding:** adds draft-model forward pass alongside main, adds verification step in scheduler. No shim changes.
- **KV quantization:** shim adds `beacon_op_attention_quantized_kv`, KV cache stores quantized tensors.
- **Low-rank FFN:** format stores low-rank factors, `ffn_block` does two smaller matmuls instead of one large.
- **LSH routing:** CPU backend only. Adds LSH index tables to format, adds dispatch logic before each FFN.
- **ANE prefill:** shim adds `beacon_op_ane_prefill`, backend routes prefill to ANE when available and model is compatible.

---

## 17. Questions Deliberately Left Open

Resolved in practice, not in this document:

- Exact quantization schemes to support in v0.1 beyond GGUF passthrough. (Decide empirically — what do real users pull?)
- Whether to support speculative decoding in v0.1 or defer to v0.2. (Defer; get correctness first.)
- Windows as a first-class target. (v0.1: build only; v0.2: decide based on user demand.)
- Licensing model specifics. (Apache 2.0 is the leading candidate; finalize before public launch.)

---

## 18. References

- **MLX**: https://github.com/ml-explore/mlx
- **llama.cpp**: https://github.com/ggerganov/llama.cpp (reference for GGUF format, quantization)
- **Candle**: https://github.com/huggingface/candle (reference for Rust ML architecture)
- **GPTQ**: Frantar et al., 2023
- **AWQ**: Lin et al., 2023
- **Deja Vu**: Liu et al., 2023 (contextual sparsity)
- **SVD-LLM**: Wang et al., 2024 (low-rank decomposition)
- **EAGLE-2**: Li et al., 2024 (speculative decoding)
- **SLIDE**: Chen et al., 2019 (LSH routing)
- **KVQuant**: Hooper et al., 2024 (KV cache quantization)
- **DeepSeek MLA**: DeepSeek-V2 paper, 2024 (attention-level KV compression)
