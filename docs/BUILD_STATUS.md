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

**Next up: Step 3 — MLX Rust bindings (`beacon-mlx` crate).**

Steps 1 and 2 complete and locally verified on macOS ARM64. Nothing is
committed to git yet — the working tree contains the Step 1 + Step 2 changes
as untracked/modified files.

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

### Steps 3 – 15 ⏳ not started

See [README build sequence](../README.md#build-sequence-for-claude-code-handoff)
for the authoritative list. Summary:

| # | Step | Architecture § | Status |
|---|---|---|---|
| 3 | MLX Rust bindings (`beacon-mlx`) | §4 | not started |
| 4 | Model format + GGUF import | §9 | not started |
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
