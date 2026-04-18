# Beacon

**MLX-first inference for Apple Silicon. The fastest way to run LLMs on your Mac.**

Beacon is a production-grade, embeddable LLM inference engine designed around Apple Silicon's unified memory architecture. One binary, zero config, built to exploit every watt and every byte of bandwidth M-series chips provide. CPU fallback runs everywhere else.

Think of it as a lighthouse for local AI: a single point of reliable, high-performance inference that every app on your machine can connect to — or embed directly.

---

> **📐 For implementers:** This README is the product vision and build plan.
> The full technical specification — shim API, FFI design, memory model, forward pass, KV cache layout, non-negotiable rules — lives in [`docs/architecture.md`](docs/architecture.md).
> **Read both documents in full before writing any code.** The architecture doc is the authoritative reference; when the README says *what*, the architecture doc says *how*.

---

## Why Beacon exists

LLM inference today has two shapes:

1. **Cloud APIs** (OpenAI, Anthropic) — fast and capable, but every token goes through someone else's servers. Privacy, latency, cost, and vendor lock-in compound.
2. **GPU-bound local stacks** (vLLM, TGI, MLC) — require high-end NVIDIA GPUs that are expensive and supply-constrained. Inaccessible for most developers.

Ollama and llama.cpp made local inference approachable, but they're generic runtimes that treat every platform as a second-class citizen. Apple Silicon — the hardware most developers actually own — is an afterthought in their architectures. Metal support is bolted on. Unified memory is not exploited. The Neural Engine is ignored.

**The insight:** An M3 Max with 128GB unified memory has ~400 GB/s of memory bandwidth shared across CPU, GPU, and Neural Engine — no PCIe transfers, no host-device copies. For LLM inference, which is fundamentally bandwidth-bound, this is world-class hardware. It's just been waiting for a runtime that treats it as the primary target rather than a port target.

Beacon is that runtime.

**DuckDB did for analytics what Beacon does for inference on Apple Silicon:** ruthlessly exploit the fact that most workloads fit on one machine when the single-node experience is engineered properly. Beacon applies this philosophy with Apple Silicon as the design center:

- **MLX-first**, with CPU and CUDA as secondary paths.
- **Unified memory-aware** from day one. Zero-copy everything. No host-device distinction.
- **Embedded**, not a server. Link it into your app as a Rust crate or Python module.
- **Single binary**, zero config. `curl | sh` and you have inference.
- **Arrow-native** data path for tokens and tensors.
- **Aggressive compression and sparsity stack** — quantization, low-rank decomposition, contextual sparsity, speculative decoding — packaged so "it just works faster."

Target user: developers on MacBooks building agents, local AI features, privacy-sensitive pipelines, and batch inference. Plus anyone on a workstation CPU who wants local inference without the GPU tax.

---

## Positioning

| | Ollama | llama.cpp | MLX-LM | LM Studio | **Beacon** |
|---|---|---|---|---|---|
| MLX-first design | ❌ | ❌ | ✅ | ⚠️ | ✅ |
| Unified memory exploitation | ⚠️ | ⚠️ | ✅ | ⚠️ | ✅ |
| Embedded (in-process) | ❌ | ⚠️ | ⚠️ | ❌ | ✅ |
| Single binary | ✅ | ⚠️ | ❌ | ✅ | ✅ |
| Rust core | ❌ | ❌ | ❌ | ❌ | ✅ |
| Arrow-native | ❌ | ❌ | ❌ | ❌ | ✅ |
| Sparsity-aware | ❌ | ❌ | ❌ | ❌ | ✅ |
| Production-grade API | ✅ | ⚠️ | ❌ | ✅ | ✅ |
| CPU fallback | ✅ | ✅ | ❌ | ✅ | ✅ |

**The gap:** MLX-LM is research-grade, rough around the edges, Python-only, and not embeddable. Ollama is great UX but generic and leaves Apple Silicon performance on the table. There's no production-ready, embeddable, MLX-first runtime. Beacon fills that gap.

---

## Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                     Integration Surface                       │
│   Rust crate  │  Python (PyO3)  │  Node (napi-rs)  │  CLI     │
├───────────────────────────────────────────────────────────────┤
│                        Beacon Core (Rust)                      │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Inference Scheduler (batching, streaming, KV cache)    │  │
│  ├─────────────────────────────────────────────────────────┤  │
│  │  Performance Stack                                      │  │
│  │   • Quantization (2-8 bit, MLX-native + GGUF import)    │  │
│  │   • Contextual sparsity (FFN neuron routing)            │  │
│  │   • Low-rank FFN decomposition                          │  │
│  │   • KV cache quantization + MLA-style compression       │  │
│  │   • Speculative decoding (self-draft / EAGLE-style)     │  │
│  ├─────────────────────────────────────────────────────────┤  │
│  │  Compute Backends (auto-selected, MLX preferred)        │  │
│  │   ┌────────────────────────────────────────────┐        │  │
│  │   │  MLX Backend (PRIMARY, Apple Silicon)      │        │  │
│  │   │   • Zero-copy unified memory               │        │  │
│  │   │   • Lazy graph compilation + kernel fusion │        │  │
│  │   │   • Metal kernels for hot paths            │        │  │
│  │   │   • ANE for prefill (experimental)         │        │  │
│  │   └────────────────────────────────────────────┘        │  │
│  │   ┌────────────────────────────────────────────┐        │  │
│  │   │  CPU Backend (secondary)                   │        │  │
│  │   │   • AVX-512 VNNI / AVX2 / NEON+SVE         │        │  │
│  │   │   • Accelerate/MKL/OpenBLAS for prefill    │        │  │
│  │   └────────────────────────────────────────────┘        │  │
│  │   ┌────────────────────────────────────────────┐        │  │
│  │   │  CUDA Backend (tertiary, optional)         │        │  │
│  │   └────────────────────────────────────────────┘        │  │
│  ├─────────────────────────────────────────────────────────┤  │
│  │  Arrow-Native Data Path (zero-copy I/O)                 │  │
│  ├─────────────────────────────────────────────────────────┤  │
│  │  Model Format: .beacon (MLX-native) + GGUF import        │  │
│  └─────────────────────────────────────────────────────────┘  │
├───────────────────────────────────────────────────────────────┤
│            Model Registry (pull, cache, verify)               │
└───────────────────────────────────────────────────────────────┘
```

**Why Rust core:** memory safety without GC, predictable latency, zero-cost FFI for Python/Node bindings, first-class SIMD intrinsics, single-binary distribution.

**Why MLX bridge rather than rewriting:** MLX is thousands of person-years of Apple + community work on Metal kernels, lazy graph compilation, and ANE integration. Rewriting would be insane. Beacon wraps MLX's C++ API via a thin Rust FFI layer (`beacon-mlx` crate), getting all the performance with Rust's safety and ergonomics on top.

**Pragmatic performance choices:**
- Inline assembly for 3-4 innermost CPU kernels (Q4 dequant-multiply hot loops) where Rust intrinsics leave 5-10% on the table.
- Vendor BLAS (Accelerate on macOS, MKL on x86, OpenBLAS fallback) for prefill-time large GEMMs. Not reinventing what thousands of person-years have already solved.
- Custom decode-time kernels in MLX/Metal where vendor libraries don't help (single-token, quantized, sparse matmuls).

---

## Supported Models (v1)

First-class support — tuned, benchmarked, shipped with reference configs:

- **Qwen 2.5** (0.5B, 1.5B, 3B, 7B, 14B, 32B on M-series with enough memory)
- **Llama 3.2 / 3.3** (1B, 3B, 8B, 70B on M3/M4 Max 64GB+)
- **Phi-3 / Phi-3.5** (mini, small, medium)
- **Gemma 2** (2B, 9B, 27B)

GGUF import for everything else. MLX-native format (`.beacon`) for maximum performance.

---

## Integration Surfaces (v1 ships both embedded and server)

### Rust crate

```rust
use beacon::{Engine, Model};

let engine = Engine::load(Model::Qwen2_5_7B_Q4)?;
let response = engine.complete("Explain unified memory.", &Default::default())?;
println!("{}", response.text);
```

### Python (PyO3)

```python
import beacon

engine = beacon.Engine.load("qwen2.5-7b-q4")
for token in engine.stream("Explain unified memory."):
    print(token, end="", flush=True)
```

### Node (napi-rs)

```js
import { Engine } from "beacon";
const engine = await Engine.load("qwen2.5-7b-q4");
const response = await engine.complete("Explain unified memory.");
```

### CLI + HTTP server (Ollama-compatible)

```bash
beacon pull qwen2.5-7b
beacon run qwen2.5-7b "Explain unified memory."
beacon serve --port 11434   # Ollama API compatible
```

---

## Performance Targets (v1)

Measured on M2 Pro (10-core, 32GB), M3 Max (16-core, 64GB), and Ryzen 7 7840HS (8-core, AVX-512, 32GB DDR5):

### Apple Silicon (primary target)

| Model | MLX-LM baseline | Ollama baseline | **Beacon v1 target** |
|---|---|---|---|
| Qwen 2.5 3B Q4 | ~75 tok/s | ~50 tok/s | **95+ tok/s** |
| Llama 3.2 3B Q4 | ~70 tok/s | ~48 tok/s | **90+ tok/s** |
| Qwen 2.5 7B Q4 | ~42 tok/s | ~28 tok/s | **55+ tok/s** |
| Llama 3.3 70B Q4 (M3 Max 64GB) | ~8 tok/s | ~5 tok/s | **12+ tok/s** |
| Phi-3.5 mini Q4 | ~85 tok/s | ~60 tok/s | **105+ tok/s** |

### CPU (secondary target, Ryzen 7840HS)

| Model | llama.cpp baseline | **Beacon v1 target** |
|---|---|---|
| Qwen 2.5 3B Q4 | ~35 tok/s | **55+ tok/s** |
| Llama 3.2 3B Q4 | ~32 tok/s | **50+ tok/s** |
| Qwen 2.5 7B Q4 | ~15 tok/s | **25+ tok/s** |

Experimental targets (v0.2+ with sparsity + speculative decoding): **2-3x** over these numbers.

---

## Roadmap

**v0.1 — MLX core + baseline (month 1-2)**
- Rust FFI bindings to MLX C++ API (`beacon-mlx` crate).
- MLX-native model format (`.beacon`) and loader.
- GGUF importer that converts to `.beacon` on first load.
- Transformer forward pass on MLX for supported model families.
- KV cache management in unified memory (zero-copy).
- Tokenizer (Rust-native, matches Hugging Face `tokenizers`).
- Streaming, sampling (top-k, top-p, temperature, min-p, repeat penalty).
- Python and Node bindings.
- CLI with `pull`, `run`, `list`, `remove`, `info`.
- HTTP server with Ollama-compatible API.
- CPU fallback backend (AVX-512/AVX2/NEON via `beacon-kernels` crate).

**v0.2 — Performance stack (month 3-4)**
- Lazy graph compilation and kernel fusion via MLX.
- Custom Metal kernels for quantized decode paths.
- 2-bit KV cache quantization.
- Contextual sparsity (Deja Vu-style FFN neuron prediction).
- Speculative decoding with self-draft (EAGLE-2 inspired).
- Vendor BLAS integration for prefill (Accelerate on macOS).
- Benchmarks published against MLX-LM, Ollama, llama.cpp, LM Studio.

**v0.3 — Research bets (month 5-6)**
- Low-rank FFN decomposition pipeline (SVD-LLM + activation-aware).
- MLA-style KV cache compression.
- Neural Engine (ANE) experiments for prefill.
- LSH-based FFN routing (SLIDE-inspired) for CPU backend.
- Publish results regardless of outcome.

**v0.4+ — Ecosystem**
- BitNet 1.58-bit support when weights are stable.
- OpenAI-compatible API endpoint.
- Tool use / function calling.
- Structured output (JSON schema enforcement).
- Multi-model serving on a single engine instance.

---

## Algorithmic Stack (what makes Beacon fast)

Each technique below is a published result. Beacon's contribution is integrating them correctly and tuning for Apple Silicon + CPU.

**Quantization.** 2-8 bit weight quantization. GGUF schemes for compatibility (Q4_K_M, Q5_K_M, Q6_K, Q8_0). MLX-native quantization for `.beacon` format. AQLM-style additive quantization (research bet).
> References: GPTQ (Frantar et al., 2023), AWQ (Lin et al., 2023), AQLM (Egiazarian et al., 2024).

**Contextual sparsity.** Small MLPs predict which FFN neurons will fire for each token. Skip the rest. ~70-85% FFN compute reduction with <1% quality loss.
> Reference: Deja Vu (Liu et al., 2023).

**Low-rank FFN decomposition.** Activation-aware truncated SVD factors large FFN matrices into smaller low-rank products. 3-4x parameter reduction on FFN layers.
> Reference: SVD-LLM (Wang et al., 2024).

**KV cache quantization and compression.** 2-bit KV cache for memory savings. MLA-style projection for further compression.
> References: KVQuant (Hooper et al., 2024), DeepSeek-V2 MLA (2024).

**Speculative decoding.** Use the model's own early layers (self-draft) to propose tokens, verify with full forward pass. 2-3x decode speedup.
> References: EAGLE / EAGLE-2 (Li et al., 2024), Medusa (Cai et al., 2024).

**Kernel fusion and lazy evaluation.** MLX's lazy graph execution lets Beacon fuse entire transformer layers into single Metal kernels, eliminating launch overhead.
> Reference: MLX framework design (Apple ML Research, 2023).

**LSH-based FFN routing (research, v0.3).** Locality-sensitive hashing to look up relevant FFN neurons in sub-linear time. CPU-native because LSH is cache-friendly.
> Reference: SLIDE (Chen et al., 2019).

---

## Repository Layout

```
beacon/
├── Cargo.toml                    # Workspace root
├── crates/
│   ├── beacon-core/              # Engine orchestration, model loading
│   ├── beacon-mlx/               # MLX FFI bridge (PRIMARY backend)
│   ├── beacon-kernels/           # CPU SIMD kernels (secondary backend)
│   ├── beacon-cuda/              # CUDA backend (tertiary, feature-gated)
│   ├── beacon-format/            # .beacon format + GGUF import
│   ├── beacon-tokenizer/         # BPE / SentencePiece tokenizer
│   ├── beacon-scheduler/         # Batching, streaming, KV cache
│   ├── beacon-arrow/             # Arrow-native I/O
│   ├── beacon-registry/          # Model pull/cache/verify
│   └── beacon-cli/               # CLI binary
├── bindings/
│   ├── python/                  # PyO3 bindings
│   └── node/                    # napi-rs bindings
├── server/
│   └── beacon-server/            # Ollama-compatible HTTP server
├── mlx-cpp/                     # Thin C++ shim over MLX C++ API
├── benches/                     # Criterion benchmarks
├── tests/
│   ├── integration/             # End-to-end correctness tests
│   └── eval/                    # MMLU, HumanEval quality harness
├── scripts/
│   ├── download-models.sh
│   └── benchmark.sh
├── docs/
│   ├── architecture.md
│   ├── mlx-integration.md
│   ├── performance.md
│   └── contributing.md
└── README.md
```

---

## Build Sequence (for Claude Code handoff)

Execute in order. Each step has concrete success criteria. Do not advance until the previous step's criteria pass.

**Before starting any step:** read [`docs/architecture.md`](docs/architecture.md) in full. Each step below references the relevant architecture sections that spec out the *how*. The README gives the step's goal and success criteria; the architecture doc gives the authoritative technical contract.

**Step 1: Workspace scaffolding**
- Cargo workspace with all crates, empty lib.rs stubs.
- GitHub Actions CI: build on macOS (ARM64, primary), Linux (x86_64), Windows (x86_64).
- `rustfmt.toml` and `clippy.toml` with strict settings.
- 📐 Architecture reference: [Section 2 — Repository Layout](docs/architecture.md#2-repository-layout-authoritative).
- ✅ Success: `cargo build --workspace` passes on all three platforms.

**Step 2: MLX C++ shim (shim/)**
- Vendor MLX as a git submodule, pinned to a specific version.
- Thin C ABI wrapper exposing the subset of MLX we need: tensor creation, matmul, attention, RMSNorm, SiLU, softmax, quantized ops, KV cache primitives.
- CMake build producing a static library.
- 📐 Architecture reference: [Section 3 — The Shim (C++ Layer)](docs/architecture.md#3-the-shim-c-layer). The C ABI in Section 3.3 is authoritative — implement exactly that surface, no more.
- ✅ Success: C shim compiles on macOS ARM64; basic MLX matmul callable from C; shim stays under 2,000 lines.

**Step 3: MLX Rust bindings (beacon-mlx crate)**
- `bindgen` generates Rust bindings from the C shim header.
- Safe Rust wrapper: `MlxTensor`, `MlxContext`, `MlxStream`.
- RAII for MLX resources.
- 📐 Architecture reference: [Section 4 — Rust FFI Layer](docs/architecture.md#4-rust-ffi-layer-beacon-mlx).
- ✅ Success: Rust test creates MLX tensors, performs matmul, gets correct results.

**Step 4: Model format + GGUF import (beacon-format)**
- Parse GGUF v3 headers and quantization schemes (Q4_0, Q4_K_M, Q5_K_M, Q6_K, Q8_0).
- Memory-map weight tensors via `memmap2` (zero-copy).
- `.beacon` format: MLX-native quantized weights + metadata + sparsity indexes (v0.2+).
- GGUF→.beacon converter on first model load.
- 📐 Architecture reference: [Section 9 — Model Format (.beacon)](docs/architecture.md#9-model-format-beacon).
- ✅ Success: Load `qwen2.5-3b-instruct-q4_k_m.gguf`, convert to `.beacon`, enumerate all tensors correctly.

**Step 5: Tokenizer (beacon-tokenizer)**
- BPE tokenizer compatible with Qwen, Llama, Phi, Gemma `tokenizer.json`.
- Chat template support (Jinja2 subset).
- 📐 Architecture reference: [Section 10 — Tokenizer](docs/architecture.md#10-tokenizer-beacon-tokenizer).
- ✅ Success: Token-for-token match with Hugging Face `tokenizers` on 10k-sample corpus across all four families.

**Step 6: CPU kernels (beacon-kernels)**
- Q4 × FP32 matmul: AVX-512 VNNI, AVX2, NEON paths.
- Runtime CPU feature detection and dispatch.
- RMSNorm, SiLU, softmax, rotary embedding.
- Inline assembly for innermost Q4 dequant-multiply loop (AVX-512 and NEON).
- Criterion benchmarks.
- 📐 Architecture reference: [Section 5 — Backend Selection](docs/architecture.md#5-backend-selection) and [Section 6.2 — CPU Backend Memory Model](docs/architecture.md#62-cpu-backend).
- ✅ Success: Q4 matmul within 5% of llama.cpp throughput on M2 Pro and Ryzen 7840HS for representative sizes.

**Step 7: Transformer forward pass on MLX (beacon-core)**
- Llama-family architecture on MLX backend (covers Qwen, Llama, Phi, Gemma — all Llama-style with config variations).
- Configurable via model config struct.
- KV cache in unified memory, indexed correctly for GQA.
- 📐 Architecture reference: [Section 7 — Forward Pass Architecture](docs/architecture.md#7-forward-pass-architecture) and [Section 8 — KV Cache Design](docs/architecture.md#8-kv-cache-design).
- ✅ Success: Qwen 2.5 3B Q4 generates coherent text for 20 diverse prompts; logits match Hugging Face reference to 3 decimal places at temperature 0.

**Step 8: Transformer forward pass on CPU (beacon-core)**
- Same forward pass wired to CPU kernels for the fallback path.
- 📐 Architecture reference: [Section 5 — Backend Selection](docs/architecture.md#5-backend-selection). The `ComputeBackend` trait means the forward pass code from Step 7 is reused; only the backend implementation changes.
- ✅ Success: Same prompts produce same logits (within FP precision differences) as MLX backend.

**Step 9: Scheduler (beacon-scheduler)**
- Streaming via `tokio::sync::mpsc`.
- Sampling: greedy, top-k, top-p, temperature, min-p, repeat penalty.
- Stop tokens, max tokens, timeout.
- 📐 Architecture reference: [Section 11 — Scheduler](docs/architecture.md#11-scheduler-beacon-scheduler).
- ✅ Success: Streaming works from Rust API; first token latency <100ms on M2 Pro for Qwen 2.5 3B.

**Step 10: CLI (beacon-cli)**
- `beacon pull <model>` — download from Hugging Face Hub, convert to `.beacon`.
- `beacon run <model> "<prompt>"` — one-shot generation.
- `beacon list`, `beacon remove`, `beacon info <model>`.
- Progress bars (`indicatif`), color output (`owo-colors`).
- 📐 Architecture reference: [Section 12.3 — CLI](docs/architecture.md#123-cli-beacon-cli).
- ✅ Success: `beacon run qwen2.5-3b "Hello"` works end-to-end from fresh install on macOS.

**Step 11: HTTP server (beacon-server)**
- Ollama-compatible `/api/generate`, `/api/chat`, `/api/tags`, `/api/pull`.
- NDJSON streaming (Ollama wire format).
- Built on `axum`.
- 📐 Architecture reference: [Section 12.4 — HTTP Server](docs/architecture.md#124-http-server-beacon-server).
- ✅ Success: Open WebUI connects to Beacon and works identically to Ollama for all four model families.

**Step 12: Python bindings (bindings/python)**
- PyO3-based package, `maturin` build.
- `Engine.load()`, `engine.complete()`, `engine.stream()`, `engine.chat()`.
- GitHub Actions for wheel publishing (macOS ARM64 priority).
- Type stubs (`.pyi`).
- 📐 Architecture reference: [Section 12.1 — Python Bindings](docs/architecture.md#121-python-pyo3-maturin).
- ✅ Success: `pip install beacon-ai`, 5-line example runs.

**Step 13: Node bindings (bindings/node)**
- napi-rs-based with TypeScript definitions.
- Prebuilt binaries for macOS (ARM64, x86_64), Linux (x86_64), Windows (x86_64).
- 📐 Architecture reference: [Section 12.2 — Node Bindings](docs/architecture.md#122-node-napi-rs).
- ✅ Success: `npm install beacon-ai`, 5-line TS example runs on all platforms.

**Step 14: Benchmark harness (benches/)**
- Criterion benchmarks for all four families at Q4_K_M.
- Comparison script vs MLX-LM, Ollama, llama.cpp, LM Studio on macOS; vs llama.cpp on Linux.
- Output: markdown table with tok/s, time-to-first-token, memory usage.
- 📐 Architecture reference: [Section 13.4 — Performance Tests](docs/architecture.md#134-performance-tests).
- ✅ Success: Published numbers reproducible via `scripts/benchmark.sh`; Beacon meets or exceeds v1 performance targets on M2 Pro and M3 Max.

**Step 15: Quality eval harness (tests/eval/)**
- MMLU subset (1000 questions across 10 categories).
- HumanEval subset (50 problems).
- Wiki perplexity on held-out slice.
- 📐 Architecture reference: [Section 13.3 — Correctness Tests](docs/architecture.md#133-correctness-tests).
- ✅ Success: Beacon within 0.5% of Hugging Face reference on MMLU for each supported model.

---

## Non-Negotiable Rules

These rules are authoritative and apply to every step above. See [Section 15 of the architecture doc](docs/architecture.md#15-non-negotiables) for the full list and rationale. Summary:

1. Shim stays under 2,000 lines of C++.
2. No Rust-to-C++ exceptions. All errors translate at the ABI boundary.
3. No tensor copies on the MLX backend. Weights are mmap'd; KV cache allocated once.
4. No cross-backend tensors. One backend per engine, chosen at load time.
5. No blocking operations in the decode hot path.
6. Logit-level correctness tests pass before any optimization work.

If any implementation violates these, flag it and revert.

---

## Non-Goals (v1)

- Training. Beacon is inference-only.
- Distributed inference across machines.
- Tensor parallelism across GPUs.
- Supporting every model architecture — we ship what we can tune well.
- Web-first deployment (WASM, WebGPU). Future work.
- Windows as a first-class target. Builds, but not optimized.

---

## Pre-Diagnosed Issues (read before starting)

- **MLX API is evolving.** Pin to a specific MLX version in the git submodule. Apple ships breaking changes between minor versions. Update deliberately, not automatically.
- **MLX Python vs C++ surface differs.** The Python API is more complete. Some ops (especially newer fused kernels) are Python-only. Check MLX C++ source for availability before designing around a feature.
- **Unified memory is not magic.** You still need to keep the working set small enough to fit in the system memory budget. A 70B Q4 model is ~40GB; on a 64GB M3 Max, you have ~20GB of headroom for KV cache and OS. Plan memory accordingly.
- **GGUF quantization variants are underspecified.** Q4_K_M block layout is defined in llama.cpp source, not in the GGUF spec. Cross-reference `ggml-quants.c`.
- **Tokenizer edge cases.** Llama-family tokenizers handle leading spaces inconsistently. Match Hugging Face `tokenizers` exactly.
- **RoPE variants.** Qwen uses different `rope_theta` than Llama; Phi uses partial-rotation. Read each config.
- **GQA KV cache layout.** Index by kv_head, not query head. Common correctness bug.
- **Apple Silicon dispatch.** M-series chips report as `aarch64` but need distinct tuning from Linux ARM. Test on real M-series.
- **AVX-512 on Zen 4.** Implemented via double-pumped 256-bit units. Benchmark before assuming it beats AVX2.
- **Metal GPU family differences.** M1, M2, M3, M4 have different GPU architectures. MLX abstracts most of this, but performance characteristics differ. Benchmark across families before publishing numbers.
- **ANE is fickle.** The Neural Engine exposes a narrow op surface and quantization format. Most LLM ops don't map cleanly. Keep ANE experiments isolated and optional.

---

## License

TBD — leaning toward Apache 2.0 for the core runtime, with a commercial license for hosted service and enterprise features.

---

## Origin

Beacon is built by the team behind [Blazer](https://gaurangakulkarni.github.io/blazer) (Arrow-native DataFrame engine) and Molten (Rust inference server for Apple Silicon). The same philosophy — Rust foundations, local-first, obsessive attention to the single-node experience — applied to language models, with Apple Silicon as the design center.

The name reflects the thesis: a single, reliable signal of local AI inference that every developer, every app, and every workflow can navigate by.
