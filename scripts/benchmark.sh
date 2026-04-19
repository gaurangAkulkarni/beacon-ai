#!/usr/bin/env bash
# scripts/benchmark.sh — Run Criterion benchmarks and optional end-to-end
# inference timing.
#
# Usage:
#   ./scripts/benchmark.sh                     # Run all kernel benchmarks
#   ./scripts/benchmark.sh --quick             # Compile-check only (no full run)
#   ./scripts/benchmark.sh --e2e MODEL         # Also run end-to-end inference
#       [--tokenizer PATH] [--max-tokens N]
#
# Results are saved to target/criterion/ with HTML reports.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Beacon Benchmarks ==="
echo "Platform: $(uname -sm)"
echo "Rust:     $(rustc --version)"
echo ""

# Parse arguments
QUICK=false
E2E_MODEL=""
TOKENIZER_ARG=""
MAX_TOKENS=20

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)
            QUICK=true
            shift
            ;;
        --e2e)
            E2E_MODEL="$2"
            shift 2
            ;;
        --tokenizer)
            TOKENIZER_ARG="--tokenizer $2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if $QUICK; then
    echo "--- Quick mode: compile-check only ---"
    cargo bench -p beacon-kernels -- --test 2>&1
    echo ""
    echo "Benchmarks compile successfully."
    exit 0
fi

echo "--- Kernel Benchmarks ---"
echo "Running full benchmarks (this may take a few minutes)..."
echo ""
cargo bench -p beacon-kernels 2>&1
echo ""
echo "HTML reports saved to: target/criterion/"
echo "Open target/criterion/report/index.html in a browser to view."

# End-to-end inference benchmark (optional)
if [[ -n "$E2E_MODEL" ]]; then
    echo ""
    echo "--- End-to-End Inference Benchmark ---"
    echo "Model: $E2E_MODEL"
    echo "Max tokens: $MAX_TOKENS"
    echo ""

    # Build release binary if needed
    cargo build --release -p beacon-cli 2>&1

    # Run inference — the timing summary is printed automatically by the CLI
    # shellcheck disable=SC2086
    cargo run --release -p beacon-cli -- run "$E2E_MODEL" "Explain the concept of machine learning in simple terms." \
        $TOKENIZER_ARG --max-tokens "$MAX_TOKENS" 2>&1

    echo ""
fi

echo "=== Done ==="
