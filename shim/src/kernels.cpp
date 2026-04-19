// Custom Metal kernels. Hot paths MLX's built-in ops don't cover.
//
// Step 2: all custom kernels return BEACON_ERR_UNKNOWN. The Q4 dequant+matmul
// Metal kernel is a v0.2 / Step 6 deliverable, not Step 1/2. The source file
// shim/metal/q4_dequant_mul.metal exists as a placeholder.

#include "guard.h"
#include "internal.h"

extern "C" {

int32_t beacon_kernel_q4_dequant_mul(
    BeaconContext* ctx, BeaconStream* stream,
    const BeaconTensor* x, const BeaconTensor* w_q4, const BeaconTensor* scales,
    BeaconTensor** out) {
    return beacon::guard([&]() -> int32_t {
        (void)ctx;
        (void)stream;
        (void)x;
        (void)w_q4;
        (void)scales;
        (void)out;
        beacon::set_error_message(
            "beacon_kernel_q4_dequant_mul: not implemented in v0.1; "
            "use beacon_op_quantized_matmul instead (custom Metal kernel "
            "lands in Step 6 / v0.2)");
        return BEACON_ERR_UNKNOWN;
    });
}

}  // extern "C"
