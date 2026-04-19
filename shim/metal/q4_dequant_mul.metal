// q4_dequant_mul.metal — placeholder for the fused Q4 dequantize + matmul
// kernel. The real implementation lands in Step 6 / v0.2; until then the
// shim falls back to MLX's built-in quantized_matmul via
// beacon_op_quantized_matmul.
//
// Keeping an empty Metal source file reserves the path referenced by the
// CMake layout (shim/metal/) and by docs/architecture.md §2.

#include <metal_stdlib>
using namespace metal;

// Intentionally empty. Do not add the real kernel here without coordinating
// with the forward pass work in Step 7.
