// Context and stream lifecycle. Ties the C ABI to MLX's Device/Stream.

#include "guard.h"
#include "internal.h"

#ifndef BEACON_NO_MLX
#include "mlx/stream.h"
#endif

extern "C" {

int32_t beacon_context_create(BeaconContext** out_ctx) {
    return beacon::guard([&]() -> int32_t {
        if (out_ctx == nullptr) {
            beacon::set_error_message("beacon_context_create: out_ctx is null");
            return BEACON_ERR_INVALID_ARGUMENT;
        }
#ifdef BEACON_NO_MLX
        beacon::set_error_message(
            "MLX is not available on this platform (BEACON_NO_MLX)");
        return BEACON_ERR_UNKNOWN;
#else
        *out_ctx = new BeaconContext();
        return BEACON_OK;
#endif
    });
}

void beacon_context_destroy(BeaconContext* ctx) {
    delete ctx;
}

int32_t beacon_stream_create(BeaconContext* ctx, BeaconStream** out_stream) {
    return beacon::guard([&]() -> int32_t {
        if (ctx == nullptr || out_stream == nullptr) {
            beacon::set_error_message(
                "beacon_stream_create: ctx or out_stream is null");
            return BEACON_ERR_INVALID_ARGUMENT;
        }
#ifdef BEACON_NO_MLX
        beacon::set_error_message("MLX not available");
        return BEACON_ERR_UNKNOWN;
#else
        auto s = mlx::core::new_stream(ctx->device);
        *out_stream = new BeaconStream(s);
        return BEACON_OK;
#endif
    });
}

void beacon_stream_destroy(BeaconStream* stream) {
    delete stream;
}

}  // extern "C"
