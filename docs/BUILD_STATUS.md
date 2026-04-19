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

**Next up: Step 5 — Tokenizer (`beacon-tokenizer` crate).**

Steps 1–4 complete and committed. All locally verified on macOS ARM64.

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

### Steps 5 – 15 ⏳ not started

See [README build sequence](../README.md#build-sequence-for-claude-code-handoff)
for the authoritative list. Summary:

| # | Step | Architecture § | Status |
|---|---|---|---|
| 5 | Tokenizer | §10 | not started |
| 6 | CPU kernels | §5, §6.2 | not started |
| 7 | Transformer forward pass on MLX | §7, §8 | not started |
| 8 | Transformer forward pass on CPU | §5 | not started |
| 9 | Scheduler | §11 | not started |
| 10 | CLI | §12.3 | not started |
| 11 | HTTP server | §12.4 | not started |
| 12 | Python bindings | §12.1 | not started |
| 13 | Node bindings | §12.2 | not started |
| 14 | Benchmark harness | §13.4 | not started |
| 15 | Quality eval harness | §13.3 | not started |

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
