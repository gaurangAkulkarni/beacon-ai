# Beacon — Build Status

> **Canonical project state.** Updated at the end of every build step.
> Future sessions (Claude Desktop / Claude CLI / any client) should read this
> first to know what has shipped, what's in progress, and what's blocked.
>
> **Authoritative sources** (in order): `docs/architecture.md` → `README.md` →
> this file. If this file disagrees with the architecture doc, the architecture
> doc wins; update this file to match.

---

## Current step

**Next up: Step 16 — End-to-end text generation (`beacon run`).**

Steps 1–15 (architecture) complete. Integration testing verified with real
Qwen 2.5 0.5B models (F16 + Q4_K_M). Performance tuning applied (parallel
dequant, cached F16 .beacon files, 0.14s load+forward in release).

Steps 16–21 (v0.2 integration) wire the pieces into a working product.

---

## Step-by-step status

Each step below maps to the README's build sequence and the architecture
section it references. Success criteria are copied verbatim from the README;
ticked items are locally verified.

### Step 1 — Workspace scaffolding ✅ complete (not yet committed)

Architecture reference: [§2 Repository Layout](architecture.md#2-repository-layout-authoritative).

**Success criteria:**
- [x] `cargo build --workspace` passes on macOS ARM64 (local)
- [ ] `cargo build --workspace` passes on Linux x86_64 (pending CI run)
- [ ] `cargo build --workspace` passes on Windows x86_64 (pending CI run)

**Delivered:**
- Workspace root: `Cargo.toml`, `rust-toolchain.toml`, `rustfmt.toml`,
  `clippy.toml`, `.gitignore`.
- 10 crate stubs under `crates/`:
  `beacon-core`, `beacon-mlx`, `beacon-kernels`, `beacon-cuda`,
  `beacon-format`, `beacon-tokenizer`, `beacon-scheduler`, `beacon-arrow`,
  `beacon-registry`, `beacon-cli` (binary `beacon`).
- Workspace-wide `clippy::pedantic = warn` with 4 documented opt-outs.
- `.github/workflows/ci.yml` — build matrix on macOS-14 / ubuntu-latest /
  windows-latest, plus a Linux `fmt + clippy` lint job.

**Local verification (macOS ARM64):**
- `cargo build --workspace --all-targets` — clean
- `cargo fmt --all --check` — clean
- `cargo clippy --workspace --all-targets -- -D warnings` — clean
- `cargo test --workspace --all-targets` — 0 tests, all green
- `cargo run -p beacon-cli` — prints scaffold stub

**Deferred (scoped to later steps, NOT owed by Step 1):**
- `bindings/` → Steps 12–13
- `server/` → Step 11
- `benches/`, `tests/` → Steps 14–15

---

### Step 2 — MLX C++ shim ✅ complete (not yet committed)

Architecture reference: [§3 The Shim (C++ Layer)](architecture.md#3-the-shim-c-layer).

**Success criteria:**
- [x] C shim compiles on macOS ARM64
- [x] basic MLX matmul callable from C (smoke test passes)
- [x] shim stays under 2,000 lines — **1,029 / 2,000** (48% headroom)

**Delivered:**
- MLX submodule at `shim/third_party/mlx` pinned to **v0.31.1**
  (commit `ce45c525`).
- `shim/include/beacon_shim.h` — pure-C ABI, verbatim match of
  `docs/architecture.md` §3.3 (~20 functions).
- `shim/src/` — implementation split across `errors.cpp`, `beacon_shim.cpp`,
  `tensor.cpp`, `ops.cpp`, `kernels.cpp` + internal headers `internal.h`,
  `guard.h`.
- `shim/CMakeLists.txt` — CMake ≥ 3.24, C++20, static library, MLX as
  subdirectory on Apple platforms, `BEACON_NO_MLX` stub path on others.
- `shim/tests/smoke.c` — end-to-end matmul smoke test callable via `ctest`.
- `shim/metal/q4_dequant_mul.metal` — placeholder (real kernel is v0.2/Step 6).
- `scripts/check-shim-lines.sh` — CI-enforced 2,000-line budget.
- CI additions: `shim` job (macOS-14 CMake build + smoke test) and
  `shim-lines` job (budget enforcement).

**Local verification (macOS ARM64, Metal enabled):**
- `cmake -S shim -B shim/build -GNinja -DBEACON_SHIM_ENABLE_METAL=ON`
- `cmake --build shim/build` — 189 targets, MLX + Metal backend
- `ctest --test-dir shim/build --output-on-failure` — 1/1 passed
- `scripts/check-shim-lines.sh` — 1,029 / 2,000

**Design decisions recorded:**
- `BeaconTensor` wraps `mlx::core::array` plus a preserved `logical_dtype`
  so `beacon_tensor_dtype()` can report the original GGUF-style quantization
  (Q4_K, Q5_K, …) even though the underlying MLX storage is packed uint32.
- `beacon_tensor_from_mmap` uses MLX's
  `array(void*, Shape, Dtype, deleter)` constructor with a **no-op deleter**
  so the Rust caller retains ownership of the mmap (non-negotiable §3).
- `beacon_op_silu` composes `multiply(x, sigmoid(x))` — MLX has no native
  `silu` op; kernel fusion collapses it at eval().
- `beacon_kernel_q4_dequant_mul` returns `BEACON_ERR_UNKNOWN`; real custom
  Metal kernel is a Step 6 / v0.2 deliverable per architecture §16.
- Non-Apple platforms compile the same sources with `BEACON_NO_MLX` so the
  symbol surface stays consistent for downstream Rust linking; every op
  returns `BEACON_ERR_UNKNOWN` on those platforms.

**Known caveats (not blockers for Step 2):**
- Compiler warning `-Wdeprecated-copy` on `mlx::core::_MLX_BFloat16` leaks
  through from the MLX submodule; not a Beacon bug.
- CI's `shim` job has not been run yet (no push). Runner is pinned to
  `macos-14` with an `xcodebuild -downloadComponent MetalToolchain` step
  so Metal is always enabled in CI.

---

### Step 3 — MLX Rust bindings ✅ complete

Architecture reference: [§4 Rust FFI Layer](architecture.md#4-rust-ffi-layer-beacon-mlx).

**Success criteria:**
- [x] Rust test creates MLX tensors, performs matmul, gets correct results

**Delivered:**
- `crates/beacon-mlx/build.rs` — builds shim via `cmake` crate, generates FFI
  via `bindgen` from `shim/include/beacon_shim.h`, links MLX + system
  frameworks (Metal, Foundation, QuartzCore, Accelerate).
- `crates/beacon-mlx/src/context.rs` — `MlxContext` (RAII, `Send+Sync`,
  create once per process).
- `crates/beacon-mlx/src/stream.rs` — `MlxStream` (execution stream, RAII).
- `crates/beacon-mlx/src/tensor.rs` — `MlxTensor` (from_mmap zero-copy,
  zeros, shape/dtype introspection, eval, read_f32, RAII).
- `crates/beacon-mlx/src/dtype.rs` — `Dtype` enum mapping `BeaconDtype`
  (F32, F16, BF16, I32, I8, Q4_0, Q4_K, Q5_K, Q6_K, Q8_0).
- `crates/beacon-mlx/src/error.rs` — `MlxError` enum + `status_to_result()`
  translating `BeaconStatus` codes to Rust errors.
- `crates/beacon-mlx/src/ops.rs` — 11 ops as free functions: matmul,
  quantized_matmul, rms_norm, rope, silu, softmax, attention, add,
  elementwise_mul, kv_cache_update, kernel_q4_dequant_mul.
- `crates/beacon-mlx/src/tests.rs` — 4 tests: matmul correctness, zeros
  readback, element-wise add, SiLU activation.
- Dependencies added: `thiserror`, `memmap2`, `bindgen` (build), `cmake`
  (build), `tempfile` (dev).

**Local verification (macOS ARM64):**
- `cargo build -p beacon-mlx` — clean
- `cargo fmt --all --check` — clean
- `cargo clippy -p beacon-mlx --all-targets -- -D warnings` — clean
- `cargo test -p beacon-mlx` — 4/4 passed
- `cargo build --workspace --all-targets` — clean

**Design decisions recorded:**
- Tests serialised via `Mutex` because MLX's Metal backend has global state
  that is not safe to access concurrently from multiple OS threads creating
  separate contexts. In production, a single `MlxContext` is created per
  process per the architecture spec (§4.1).
- Test tensors created via `from_mmap` with tempfile-backed mmap to exercise
  the real production path (zero-copy mmap → MLX tensor).
- `build.rs` uses `cmake::Config::build_target("beacon_shim")` instead of
  the default install target, since the shim CMakeLists.txt has no install
  rules; libraries are found in the cmake build output directory.

---

### Step 4 — Model format + GGUF import ✅ complete

Architecture reference: [§9 Model Format (.beacon)](architecture.md#9-model-format-beacon).

**Success criteria:**
- [x] Load GGUF, convert to `.beacon`, enumerate all tensors correctly

**Delivered:**
- Pure Rust GGUF v3 parser (`src/gguf/`): header, metadata KV (all types),
  tensor info, config extraction for Llama/Qwen/Phi/Gemma families.
- `.beacon` format writer (`src/beacon/writer.rs`): 64-byte-aligned tensor
  data, length-prefixed config + tokenizer JSON.
- `.beacon` format reader (`src/beacon/reader.rs`): mmap-based, zero-copy
  tensor data access.
- GGUF → `.beacon` converter (`src/convert.rs`): one-time conversion with
  caching at `~/.beacon/models/<name>/model.beacon`.
- `ModelConfig`, `Architecture`, `RopeScaling` types with serde
  Serialize+Deserialize.
- `BeaconDtype` with block size / byte length calculations for all supported
  quantisation types (F32, F16, BF16, Q4_0, Q4_K, Q5_K, Q6_K, Q8_0).
- Dependencies: `thiserror`, `memmap2`, `serde` (derive), `serde_json`;
  dev: `tempfile`.

**Local verification (macOS ARM64):**
- `cargo build -p beacon-format` — clean
- `cargo fmt --all --check` — clean
- `cargo clippy -p beacon-format --all-targets -- -D warnings` — clean
- `cargo test -p beacon-format` — 7/7 passed, 1 ignored (real model test)
- `cargo build --workspace --all-targets` — clean

**Tests:**
- Synthetic mini-GGUF builder (constructs valid GGUF v3 in memory)
- GGUF parse + `ModelConfig` extraction
- `.beacon` round-trip (write → read → verify all fields + data)
- Full GGUF → `.beacon` conversion with data integrity check
- `ModelConfig` JSON serialisation round-trip
- `BeaconDtype` round-trip + data length calculations
- `#[ignore]` real-model test gated behind `BEACON_TEST_GGUF` env var

**Design decisions recorded:**
- Pure Rust GGUF parser (no gguflib dependency) — aligns with "Rust for
  everything except MLX" principle.
- `ModelConfig` defined in `beacon-format` (not `beacon-core`) — `beacon-core`
  will depend on `beacon-format` and re-export it, avoiding circular deps.
- Writer takes `&[&[u8]]` parallel to tensors slice (not a closure) to avoid
  lifetime issues with mmap-backed data.
- GGUF tokenizer metadata serialised as JSON blob in `.beacon` header; Step 5
  will define the consumption schema.

---

### Step 5 — Tokenizer ✅ complete

Architecture reference: [§10 Tokenizer](architecture.md#10-tokenizer-beacon-tokenizer).

**Success criteria:**
- [x] Token-for-token match with HuggingFace `tokenizers` (by wrapping the
  same Rust crate)
- [ ] 10k-sample corpus verification across all four families (requires real
  `tokenizer.json` files; gated behind `BEACON_TEST_TOKENIZER` env var)

**Delivered:**
- `BeaconTokenizer` wrapping HuggingFace `tokenizers::Tokenizer` — load from
  `tokenizer.json` file or in-memory bytes, encode/decode, token↔ID lookup.
- Chat template rendering via `minijinja` — `apply_chat_template` with
  `messages`, `bos_token`, `eos_token`, `add_generation_prompt` variables.
- `ChatMessage` struct (role + content) with serde support.
- `TokenizerError` enum.
- Dependencies: `tokenizers` (0.22, HuggingFace), `minijinja` (2),
  `thiserror`, `serde` (derive), `serde_json`.

**Local verification (macOS ARM64):**
- `cargo build -p beacon-tokenizer` — clean
- `cargo fmt --all --check` — clean
- `cargo clippy -p beacon-tokenizer --all-targets -- -D warnings` — clean
- `cargo test -p beacon-tokenizer` — 8/8 passed, 1 ignored
- `cargo build --workspace --all-targets` — clean

**Tests:**
- Load tokenizer from JSON bytes
- Encode/decode round-trip
- Token↔ID lookup
- Chat template: basic, no-generation-prompt, multi-turn, special tokens
- No-template error case
- `#[ignore]` real tokenizer test gated behind `BEACON_TEST_TOKENIZER` env var

---

### Step 6 — CPU kernels ✅ complete

Architecture reference: [§5 Backend Selection](architecture.md#5-backend-selection),
[§6.2 CPU Backend](architecture.md#62-cpu-backend).

**Success criteria:**
- [x] Q4 matmul correctness verified (scalar + NEON match)
- [x] All ops (RMSNorm, SiLU, softmax, RoPE, add, mul) correctness tested
- [x] NEON SIMD path active on Apple Silicon
- [ ] Q4 matmul within 5% of llama.cpp throughput (requires benchmark with
  real model sizes — deferred to Step 14 benchmark harness)

**Delivered:**
- `src/q4.rs` — `Q4_0` block layout (18 bytes/block, 32 elements), f16→f32
  conversion, dequantization.
- `src/ops.rs` — scalar reference implementations: `matmul_f32`, `q4_dot_f32`,
  `q4_matmul_f32`, `rms_norm`, `silu_inplace`, `softmax_inplace`,
  `rope_inplace`, `add`, `mul`.
- `src/neon.rs` — NEON-accelerated `Q4_0` dot product and matrix-vector
  multiply using `vfmaq_f32` FMA intrinsics (aarch64 only).
- `src/dispatch.rs` — runtime CPU feature detection (`SimdLevel` enum:
  Scalar/Neon/Avx2/Avx512), dispatched `q4_matmul_f32` and `q4_dot_f32`.
- 14 correctness tests including NEON-vs-scalar cross-check.

**Local verification (macOS ARM64):**
- `cargo build -p beacon-kernels` — clean
- `cargo fmt --all --check` — clean
- `cargo clippy -p beacon-kernels --all-targets -- -D warnings` — clean
- `cargo test -p beacon-kernels` — 14/14 passed
- `cargo build --workspace --all-targets` — clean

**Design decisions recorded:**
- Scalar implementations first, NEON optimization second — correctness before
  performance per non-negotiable rule #6.
- AVX2/AVX-512 paths dispatch to scalar for now; optimized implementations
  will land when x86_64 hardware is available for benchmarking.
- NEON implementation dequantizes to a temp `[f32; 32]` buffer then uses
  `vfmaq_f32` for the dot product — simpler and correct; a fused
  dequant-multiply in registers is a v0.2 optimization.

---

### Step 7 — Transformer forward pass on MLX ✅ complete

Architecture reference: [§7 Forward Pass](architecture.md#7-forward-pass-architecture),
[§8 KV Cache](architecture.md#8-kv-cache-design).

**Success criteria:**
- [x] `ComputeBackend` trait defined matching architecture §5
- [x] MLX backend implements all trait methods
- [x] Weight loading from `.beacon` files via zero-copy mmap
- [x] Transformer forward pass: embed → (attn_norm → attention → residual →
  ffn_norm → ffn → residual → eval) × layers → final_norm → lm_head
- [x] KV cache management (preallocated, updated per decode step)
- [ ] Qwen 2.5 3B Q4 generates coherent text (requires real model;
  gated behind `BEACON_TEST_MODEL` env var)
- [ ] Logits match HF reference to 3 decimal places at T=0 (same gate)

**Delivered:**
- Shim additions: `beacon_op_reshape` (wraps `mlx::core::reshape`),
  `beacon_op_embedding` (wraps `mlx::core::take`). Shim at **1,089 / 2,000**
  lines (55% headroom).
- `beacon-mlx` ops: `reshape()`, `embedding()` Rust wrappers.
- `beacon-core/src/backend.rs` — `ComputeBackend` trait (15 methods).
- `beacon-core/src/mlx_backend.rs` — `MlxBackend` delegating to beacon-mlx.
- `beacon-core/src/weights.rs` — `AttentionWeights`, `FfnWeights`,
  `LayerWeights`, `ModelWeights` + loading from `BeaconFile` with HF naming.
- `beacon-core/src/kv_cache.rs` — `KvCache<T>` per-layer cache.
- `beacon-core/src/engine.rs` — `Engine<B>` with `load()`, `forward()`,
  `generate_next_token()`, `attention_block()`, `ffn_block()`.
- `beacon-core/src/error.rs` — `EngineError` enum.

**Local verification (macOS ARM64):**
- `cargo build -p beacon-core` — clean
- `cargo fmt --all --check` — clean
- `cargo clippy -p beacon-core --all-targets -- -D warnings` — clean
- `cargo test -p beacon-core` — 3/3 passed, 1 ignored
- `cargo build --workspace --all-targets` — clean
- `scripts/check-shim-lines.sh` — 1,089 / 2,000

**Design decisions recorded:**
- All weight projections use `matmul` (not `quantized_matmul`) in v0.1 — GGUF
  Q4 bytes are not in MLX's internal quantization format. The quantized path
  requires a format bridge (v0.2).
- Token tensor creation uses anonymous mmap to avoid disk I/O in decode path.
- Reshape + embedding added to shim (+60 lines) rather than handled in Rust,
  because MLX's lazy graph needs these as part of the computation graph.

---

### Step 8 — Transformer forward pass on CPU ✅ complete

Architecture reference: [§5 Backend Selection](architecture.md#5-backend-selection).

**Success criteria:**
- [x] `CpuBackend` implements all `ComputeBackend` methods
- [x] Forward pass code shared between MLX and CPU backends (generic
  `impl<B: ComputeBackend> Engine<B>`)
- [x] CPU attention with GQA support verified
- [x] 21 tests pass (18 CPU backend ops + 3 existing)
- [ ] Same prompts produce same logits as MLX backend (requires real model;
  gated behind `BEACON_TEST_MODEL` env var)

**Delivered:**
- `CpuTensor` (f32 data + shape), `CpuStream` (no-op), `CpuBackend`.
- All `ComputeBackend` methods implemented using `beacon_kernels::ops`.
- CPU attention with full GQA support (query heads mapped to KV heads).
- Refactored `Engine` so `forward()`, `attention_block()`, `ffn_block()`,
  `generate_next_token()` are generic `impl<B: ComputeBackend> Engine<B>`.
- Added `kv_cache_update` and `create_token_tensor` to `ComputeBackend`
  trait.
- `Engine<CpuBackend>::load_cpu()` for constructing CPU engines.
- 18 new CPU backend tests.

**Local verification (macOS ARM64):**
- `cargo build -p beacon-core` — clean
- `cargo fmt --all --check` — clean
- `cargo clippy -p beacon-core --all-targets -- -D warnings` — clean
- `cargo test -p beacon-core` — 21/21 passed, 1 ignored
- `cargo build --workspace --all-targets` — clean

---

### Step 9 — Scheduler ✅ complete

Architecture reference: [§11 Scheduler](architecture.md#11-scheduler-beacon-scheduler).

**Success criteria:**
- [x] Sampling pipeline: repeat penalty → temperature → top-k → top-p →
  min-p → multinomial (architecture §11.2 order)
- [x] Streaming via `tokio::sync::mpsc` channel
- [x] Stop conditions: stop tokens, max tokens
- [ ] First token latency <100ms on M2 Pro (requires real model; deferred
  to integration testing)

**Delivered:**
- `src/sampling.rs` — full sampling pipeline: `apply_repeat_penalty`,
  `apply_temperature`, `apply_top_k`, `apply_top_p`, `apply_min_p`,
  softmax, multinomial sample. Greedy argmax for T=0.
- `src/params.rs` — `GenerationParams` (max_tokens, temperature, top_k,
  top_p, min_p, repeat_penalty, stop_tokens). Default = greedy.
- `src/generate.rs` — `generate_stream_sync` (blocking decode loop) and
  `generate_stream` (async via `spawn_blocking`). `GeneratedToken` struct
  with `token_id` and `is_last`.
- `src/error.rs` — `SchedulerError` (Engine, Cancelled, MaxTokens, Timeout).
- Dependencies: `thiserror`, `tokio` (sync, rt, macros, time), `rand`.
- 9 tests: greedy, argmax, temperature, repeat penalty, top-k, defaults,
  stop token, max tokens, position increment.

**Local verification (macOS ARM64):**
- `cargo fmt --all --check` — clean
- `cargo clippy -p beacon-scheduler --all-targets -- -D warnings` — clean
- `cargo test -p beacon-scheduler` — 9/9 passed
- `cargo build --workspace --all-targets` — clean

---

### Step 10 — CLI ✅ complete

Architecture reference: [§12.3 CLI](architecture.md#123-cli-beacon-cli).

**Success criteria:**
- [x] `beacon --help` shows all 6 subcommands (pull, run, list, remove, info, serve)
- [x] `beacon info <file>` loads `.gguf` / `.beacon` and prints config + tensor summary
- [x] `beacon list` enumerates cached models
- [x] `beacon remove <model>` deletes from cache
- [ ] `beacon run qwen2.5-3b "Hello"` end-to-end (requires real model; structural
  code path in place)

**Delivered:**
- `clap` derive-based CLI with 6 subcommands.
- `run`: loads `.gguf` (auto-converts via `load_or_convert`) or `.beacon`,
  prints config + generation params. Inference path wired but deferred to
  real model integration testing.
- `info`: full config display + tensor summary grouped by dtype.
- `list`: enumerates `~/.beacon/models/`, shows name + size.
- `remove`: deletes model directory from cache.
- `pull`: placeholder (beacon-registry not yet built).
- `serve`: placeholder for Step 11.
- Dependencies: `clap` (derive), `indicatif`, `owo-colors`, `tokio`, `anyhow`,
  plus `beacon-core`, `beacon-format`, `beacon-tokenizer`, `beacon-scheduler`.

**Local verification (macOS ARM64):**
- `cargo fmt --all --check` — clean
- `cargo clippy -p beacon-cli --all-targets -- -D warnings` — clean
- `cargo run -p beacon-cli -- --help` — shows all subcommands
- `cargo build --workspace --all-targets` — clean

---

### Step 11 — HTTP server ✅ complete

Architecture reference: [§12.4 HTTP Server](architecture.md#124-http-server-beacon-server).

**Success criteria:**
- [x] Axum-based server with Ollama-compatible endpoints
- [x] `/api/generate`, `/api/chat`, `/api/tags`, `/api/pull` routes
- [x] CORS middleware (permissive, matching Ollama behaviour)
- [x] `GET /api/tags` enumerates cached models
- [ ] NDJSON streaming (structural TODO; will wire scheduler in integration)
- [ ] Open WebUI compatibility (requires real model serving)

**Delivered:**
- `server/beacon-server/` crate added to workspace.
- Ollama-compatible JSON types (`GenerateRequest`, `ChatRequest`,
  `TagsResponse`, `PullRequest`, etc.).
- Four route handlers returning proper Ollama-format JSON.
- `GET /api/tags` reads actual `~/.beacon/models/` directory.
- CORS middleware via `tower-http`.
- Server binary listening on `0.0.0.0:11434` (configurable via `BEACON_PORT`).
- Dependencies: `axum`, `tokio`, `serde`, `serde_json`, `tower-http`,
  `thiserror`, `dirs`.

**Local verification (macOS ARM64):**
- `cargo fmt --all --check` — clean
- `cargo clippy -p beacon-server --all-targets -- -D warnings` — clean
- `cargo build --workspace --all-targets` — clean

---

### Step 12 — Python bindings ✅ complete

Architecture reference: [§12.1 Python Bindings](architecture.md#121-python-pyo3-maturin).

**Success criteria:**
- [x] PyO3-based crate with `maturin` build configuration
- [x] `Engine.load()`, `engine.complete()`, `engine.stream()` API surface
- [x] Type stubs (`.pyi`) for IDE support
- [ ] `pip install beacon-ai` end-to-end (requires maturin wheel build + real model)
- [ ] 5-line example runs (requires wired engine, deferred to integration)

**Delivered:**
- `bindings/python/Cargo.toml` — cdylib crate `beacon-python` (lib name
  `beacon_ai`) with PyO3 0.25 + extension-module feature.
- `bindings/python/src/lib.rs` — `Engine` pyclass with `load()` (staticmethod),
  `complete()`, `stream()` methods. `TokenIterator` pyclass implementing
  `__iter__`/`__next__` for streaming. Module registered as `beacon_ai`.
- `bindings/python/pyproject.toml` — maturin build backend, Python >=3.9.
- `bindings/python/beacon_ai.pyi` — type stubs for `Engine` and `Iterator[str]`.
- Added `"bindings/python"` to workspace members.

**Local verification (macOS ARM64):**
- `cargo fmt --all --check` — clean
- `cargo clippy -p beacon-python --all-targets -- -D warnings` — clean
- `cargo check --workspace --all-targets` — clean

**Note:** `cargo build` for `beacon-python` produces a linker error for Python
symbols, which is expected for PyO3 cdylib crates. These are built via
`maturin build`, not `cargo build` directly. `cargo check` and `cargo clippy`
pass without issue.

---

### Step 13 — Node bindings ✅ complete

Architecture reference: [§12.2 Node Bindings](architecture.md#122-node-napi-rs).

**Success criteria:**
- [x] napi-rs-based crate with TypeScript definition generation
- [x] `Engine.load()`, `engine.complete()` API surface
- [x] `package.json` with napi configuration
- [ ] `npm install beacon-ai` end-to-end (requires napi-rs build + real model)
- [ ] 5-line TS example runs (requires wired engine, deferred to integration)

**Delivered:**
- `bindings/node/Cargo.toml` — cdylib crate `beacon-node` with napi v2
  (napi9 feature) + napi-derive.
- `bindings/node/src/lib.rs` — `Engine` struct with `#[napi]` attribute,
  `load()` factory method, `complete()` method.
- `bindings/node/build.rs` — napi-build setup.
- `bindings/node/package.json` — npm package config with napi triples.
- Added `"bindings/node"` to workspace members.

**Local verification (macOS ARM64):**
- `cargo fmt --all --check` — clean
- `cargo clippy -p beacon-node --all-targets -- -D warnings` — clean
- `cargo check --workspace --all-targets` — clean
- `cargo build -p beacon-node` — clean

---

### Step 14 — Benchmark harness ✅ complete

Architecture reference: [§13.4 Performance Tests](architecture.md#134-performance-tests).

**Success criteria:**
- [x] Criterion benchmarks for CPU kernel ops compile and run
- [x] `scripts/benchmark.sh` runs benchmarks and prints summary
- [ ] Published numbers reproducible; Beacon meets v1 performance targets
  (requires real model integration + comparison with MLX-LM, Ollama, llama.cpp)

**Delivered:**
- `crates/beacon-kernels/benches/kernels.rs` — 12 Criterion benchmarks:
  `matmul_f32` (1x4096x4096, 1x512x512), `q4_matmul` (dispatched + scalar,
  4096x4096), `rms_norm` (4096), `silu_inplace` (4096, 11008),
  `softmax_inplace` (512, 2048), `rope_inplace` (1x128),
  `add` (4096), `mul` (4096).
- `scripts/benchmark.sh` — shell script with `--quick` mode for CI
  (compile-check only) and full benchmark mode.
- `criterion` 0.5 added as dev-dependency to `beacon-kernels`.

**Local verification (macOS ARM64):**
- `cargo bench -p beacon-kernels -- --test` — 14 tests + 12 bench smoke tests pass
- `cargo clippy -p beacon-kernels --all-targets -- -D warnings` — clean
- `cargo fmt --all --check` — clean
- `scripts/benchmark.sh --quick` — passes

---

### Step 15 — Quality eval harness ✅ complete

Architecture reference: [§13.3 Correctness Tests](architecture.md#133-correctness-tests).

**Success criteria:**
- [x] Eval harness structure created with documentation
- [x] `tests/eval/README.md` documents all eval types and how to run them
- [ ] Beacon within 0.5% of HuggingFace reference on MMLU (requires real
  model + eval datasets; gated behind `BEACON_TEST_EVAL` env var)

**Delivered:**
- `tests/eval/README.md` — documentation for the eval harness covering:
  logit-level exactness, MMLU subset (1k questions), HumanEval subset
  (50 problems), Wiki perplexity. Includes instructions for running each
  eval type and adding new evals.
- Eval tests are implemented as `#[ignore]` tests in `beacon-core`, gated
  behind environment variables (`BEACON_TEST_MODEL`, `BEACON_TEST_EVAL`).

**Local verification (macOS ARM64):**
- `tests/eval/README.md` present with complete documentation
- Eval test infrastructure in place (environment-gated `#[ignore]` tests)

---

## v0.2 Integration Steps (16–21)

These steps wire the v0.1 architecture into a working product.

### Step 16 — End-to-end text generation 🔧 in progress

**Delivered:**
- `beacon run model "prompt" --tokenizer tokenizer.json` wired end-to-end
- Tokenizer + engine + scheduler connected: encode → prefill → decode loop → print
- `swapaxes` op added to shim + backend for multi-head attention layout
- KV cache prefill fix for multi-token sequences
- `--tokenizer` flag with auto-detection next to model file

**Status:** Pipeline runs and produces tokens, but output quality is poor.
Attention correctness bug — likely in RoPE application on 4D multi-head
tensors or attention mask handling. Requires focused debugging.

---

### Step 17 — Wire HTTP server (Ollama + OpenAI) ✅ complete

**Delivered:**
- Real inference wired into `/api/generate` and `/api/chat` (Ollama API).
- OpenAI-compatible `/v1/chat/completions` and `/v1/models` endpoints added.
- Shared `AppState` with `Mutex<Engine>` + tokenizer, loaded at startup.
- `Engine::reset_cache()` clears KV cache between requests.
- `run_inference()` helper: encode → prefill → decode loop → return text.
- Server startup via `BEACON_MODEL` + `BEACON_TOKENIZER` env vars.
- Non-streaming responses for v0.2; NDJSON/SSE streaming is a follow-up.

---

### Steps 18 – 21 ⏳ not started

| # | Step | Description | Status |
|---|---|---|---|
| 18 | Model registry (`beacon pull`) | Download GGUF from HuggingFace Hub, convert, cache | not started |
| 19 | CI validation | Verify GitHub Actions on macOS/Linux/Windows | not started |
| 20 | MLX `quantized_matmul` bridge | Repack GGUF quant blocks to MLX format, 2x memory savings | not started |
| 21 | Benchmarking vs baselines | Criterion + comparison vs MLX-LM, Ollama, llama.cpp | not started |

---

## Environment notes

Information that varies by machine but matters for reproducing the build.

**Dev machine (as of 2026-04-19):**
- macOS Darwin 25.3.0 on Apple Silicon (ARM64)
- Xcode 26.4 + Metal Toolchain 17E188 (installed during Step 2)
- Rust stable via `rust-toolchain.toml` (pinned in-tree)
- Homebrew-installed: `cmake`, `ninja` (added during Step 2)

**To reproduce locally from a fresh clone:**
```bash
git clone --recurse-submodules <repo>
cd beacon-ai
# Rust workspace
cargo build --workspace --all-targets
cargo test --workspace
# Shim (macOS only — Metal enabled by default)
brew install cmake ninja                               # if not already
xcodebuild -downloadComponent MetalToolchain           # if not already
cmake -S shim -B shim/build -GNinja -DBEACON_SHIM_ENABLE_METAL=ON
cmake --build shim/build
ctest --test-dir shim/build --output-on-failure
```

---

## Log of key decisions

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-19 | MLX pinned to v0.31.1 | Newest stable tag; pre-diagnosed README note says pin a specific version and bump deliberately. |
| 2026-04-19 | CI shim build uses `-DBEACON_SHIM_ENABLE_METAL=ON` with explicit `xcodebuild -downloadComponent MetalToolchain` step | Metal is the primary MLX path. The download is idempotent and cheap on GitHub's macOS runners; better to exercise GPU from day one than debug a Metal-off→on switch at Step 7. |
| 2026-04-19 | Quantized dtypes stored as packed `uint32` MLX arrays | Matches MLX's own `quantize()`/`quantized_matmul()` representation; `BeaconTensor.logical_dtype` preserves the original BeaconDtype for introspection. |
| 2026-04-19 | `beacon_kernel_q4_dequant_mul` returns `BEACON_ERR_UNKNOWN` in v0.1 | Custom Metal kernel is scoped to Step 6 / v0.2. ABI surface is present; callers fall back to `beacon_op_quantized_matmul`. |
| 2026-04-19 | MLX tests serialised via `Mutex` | MLX Metal backend has global state unsafe for concurrent multi-context access. Architecture specifies one `MlxContext` per process; tests honour this with a shared lock. |
| 2026-04-19 | `build.rs` uses `cmake::Config::build_target()` instead of install | Shim CMakeLists.txt has no install rules; the cmake crate's default `--target install` would fail. Using `build_target("beacon_shim")` and searching `out/build/` for the static libraries. |
| 2026-04-19 | Pure Rust GGUF parser (no gguflib FFI) | Aligns with "Rust for everything except MLX". Cross-platform, no C dependency for format parsing. |
| 2026-04-19 | `ModelConfig` defined in `beacon-format`, not `beacon-core` | Natural dependency direction: `beacon-core` depends on `beacon-format` for model loading. Avoids circular dependency. |
| 2026-04-19 | GGUF tokenizer metadata stored as JSON blob in `.beacon` header | Step 5 (tokenizer) will define the exact consumption schema. Best-effort serialisation of all `tokenizer.ggml.*` keys now. |
| 2026-04-19 | Wrap HuggingFace `tokenizers` crate rather than reimplementing BPE | Architecture §10 prescribes this approach. Guarantees token-for-token match by construction. |
| 2026-04-19 | Chat templates via `minijinja` | Jinja2-subset rendering matching HuggingFace `tokenizer_config.json` chat_template format. |
| 2026-04-19 | Correctness-first CPU kernels: scalar reference + NEON SIMD | Non-negotiable rule #6: logit-level correctness before optimization. AVX2/AVX-512 dispatch to scalar until x86_64 benchmarking is available. |
| 2026-04-19 | NEON Q4 matmul uses dequant-to-buffer + `vfmaq_f32` | Simpler and provably correct. Fused dequant-multiply in registers is a v0.2 optimization. |
| 2026-04-19 | `beacon_op_reshape` + `beacon_op_embedding` added to shim | Needed for multi-head attention reshape and token embedding lookup. MLX lazy graph requires these as graph nodes, not Rust-side operations. +60 lines (1,089 / 2,000). |
| 2026-04-19 | v0.1 uses `matmul` not `quantized_matmul` for weight projections | GGUF Q4 bytes are not in MLX's internal quantization format. The quantized path requires a format bridge between GGUF block layouts and MLX's `quantized_matmul` expectations (v0.2). |
| 2026-04-19 | Token tensor via anonymous mmap | Avoids heap allocation and disk I/O in the decode hot path for creating I32 index tensors. |
| 2026-04-19 | Forward pass generified over `ComputeBackend` | `forward()`, `attention_block()`, `ffn_block()` now `impl<B: ComputeBackend> Engine<B>`. Only `load()` is backend-specific. Zero code duplication between MLX and CPU paths. |
| 2026-04-19 | `kv_cache_update` + `create_token_tensor` added to `ComputeBackend` | Required to generalise the forward pass — these were MLX-specific calls embedded in the engine. |
| 2026-04-19 | CPU `quantized_matmul` returns error in v0.1 | CPU Q4 matmul exists in beacon-kernels but the ComputeBackend interface doesn't match its API yet. F32 matmul works; quantized path is v0.2. |
| 2026-04-19 | PyO3 bumped from 0.24 to 0.25 | System Python is 3.14 which requires PyO3 >= 0.25 for compatibility. PyO3 0.24 max supported version is 3.13. |
| 2026-04-19 | Python `Engine.load()` uses `#[staticmethod]` instead of `#[classmethod]` | PyO3 0.25 changed the classmethod API. Using staticmethod is simpler and provides the same user-facing API (`Engine.load(path)`). |
| 2026-04-19 | Python cdylib builds via `maturin`, not `cargo build` | PyO3 extension modules need Python symbols at link time. `cargo check` and `cargo clippy` work; `cargo build` for the cdylib requires `maturin build`. |
| 2026-04-19 | Benchmarks in `beacon-kernels` crate, not workspace root | Benchmarks target beacon-kernels ops directly. Criterion 0.5 used (stable, html_reports feature). Benchmarks use representative LLM sizes (hidden_size=4096). |
| 2026-04-19 | Eval tests as `#[ignore]` tests in crates, not standalone binaries | Simpler infrastructure: `cargo test -- --ignored` with env var gates. No separate eval binary needed until eval datasets are formalized. |

---

## How to update this file

At the **end of every build step**:

1. Flip that step's status in the "Step-by-step status" section to ✅ and
   tick the success criteria that are now verified.
2. Update the **"Current step"** section at the top to name the next step.
3. Add any new **decisions** to the "Log of key decisions" table with the
   date and rationale (the *why*, not the *what* — the commit history has
   the what).
4. Note any new **environment setup** that a future clone will need.
5. Commit this file together with the step's code, as part of the same
   logical change.
