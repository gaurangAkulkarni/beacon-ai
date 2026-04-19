# Beacon Performance

Benchmark methodology and measured results for Beacon inference.

## System

- Apple Silicon M-series (macOS ARM64)
- Measured on Qwen 2.5 0.5B Instruct

Results will vary by chip generation, memory bandwidth, thermal state, and
background load. Always reproduce on your own hardware before drawing
conclusions.

## End-to-End Inference (Qwen 2.5 0.5B)

| Metric              | F16     | Q4_K_M  |
|---------------------|---------|---------|
| Load time           | TBD     | TBD     |
| Time to first token | TBD     | TBD     |
| Tokens/sec (decode) | ~8 tok/s | ~27 tok/s |
| Memory (model)      | 1.2 GB  | 464 MB  |

## How to Reproduce

```bash
# Download model
beacon pull qwen2.5-0.5b

# Run with timing (timing summary prints automatically)
beacon run qwen2.5-0.5b "Explain machine learning" --max-tokens 100

# Or run directly from a GGUF file
beacon run /path/to/model.gguf "Hello" --tokenizer /path/to/tokenizer.json --max-tokens 20

# Run kernel benchmarks
cargo bench -p beacon-kernels
```

## Kernel Benchmarks

Per-op CPU kernel timings are measured with Criterion. Run:

```bash
cargo bench -p beacon-kernels
```

Results are saved to `target/criterion/` with HTML reports. Open
`target/criterion/report/index.html` in a browser to view detailed plots.

Benchmarked operations (at representative LLM sizes, hidden_size=4096):

| Benchmark                         | Description                         |
|-----------------------------------|-------------------------------------|
| `matmul_f32_1x4096x4096`         | FP32 matrix-vector multiply         |
| `matmul_f32_1x512x512`           | FP32 matmul (smaller)               |
| `q4_matmul_4096x4096_dispatched` | Q4_0 matmul with SIMD dispatch      |
| `q4_matmul_4096x4096_scalar`     | Q4_0 matmul (scalar reference)      |
| `rms_norm_4096`                   | RMSNorm over hidden dimension       |
| `silu_inplace_4096`              | SiLU activation (hidden_size)       |
| `silu_inplace_11008`             | SiLU activation (FFN intermediate)  |
| `softmax_inplace_512`            | Softmax over 512 elements           |
| `softmax_inplace_2048`           | Softmax over 2048 elements          |
| `rope_inplace_1x128`            | Rotary position embedding           |
| `add_4096`                       | Element-wise add                    |
| `mul_4096`                       | Element-wise multiply               |

## Full Benchmark Script

The `scripts/benchmark.sh` script runs both kernel benchmarks and (when a
model is available) end-to-end inference timing:

```bash
# Full benchmarks
./scripts/benchmark.sh

# Quick mode (compile-check only, for CI)
./scripts/benchmark.sh --quick

# End-to-end inference benchmark (requires model)
./scripts/benchmark.sh --e2e /path/to/model.gguf --tokenizer /path/to/tokenizer.json
```

## Performance Targets (v1)

See the [README](../README.md#performance-targets-v1) for target throughput
numbers across model sizes and platforms.
