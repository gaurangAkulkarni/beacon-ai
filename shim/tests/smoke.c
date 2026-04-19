// smoke.c — minimal C program that exercises the Beacon shim end-to-end.
// Implements the Step 2 success criterion: "basic MLX matmul callable from C".
//
// Creates two 2x2 f32 tensors from stack buffers, multiplies them through the
// shim, reads the result back, and compares against a hand-computed reference.
// Exits non-zero on any failure.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "beacon_shim.h"

#define CHECK(expr)                                                            \
    do {                                                                       \
        int32_t _status = (expr);                                              \
        if (_status != BEACON_OK) {                                            \
            fprintf(stderr, "%s:%d: " #expr " failed (status=%d): %s\n",       \
                    __FILE__, __LINE__, _status,                               \
                    beacon_last_error_message());                              \
            return 1;                                                          \
        }                                                                      \
    } while (0)

static int float_close(float a, float b) {
    float d = a - b;
    if (d < 0) d = -d;
    return d < 1e-4f;
}

int main(void) {
    BeaconContext* ctx = NULL;
    BeaconStream* stream = NULL;
    CHECK(beacon_context_create(&ctx));
    CHECK(beacon_stream_create(ctx, &stream));

    // A = [[1, 2],   B = [[5, 6],   A @ B = [[1*5+2*7, 1*6+2*8],
    //      [3, 4]]        [7, 8]]             [3*5+4*7, 3*6+4*8]]
    //                                       = [[19, 22],
    //                                          [43, 50]]
    float a_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    int64_t shape[2] = {2, 2};

    BeaconTensor* a = NULL;
    BeaconTensor* b = NULL;
    BeaconTensor* c = NULL;
    CHECK(beacon_tensor_from_mmap(ctx, a_data, shape, 2, BEACON_DTYPE_F32, &a));
    CHECK(beacon_tensor_from_mmap(ctx, b_data, shape, 2, BEACON_DTYPE_F32, &b));
    CHECK(beacon_op_matmul(ctx, stream, a, b, &c));

    // Verify shape.
    int64_t out_shape[4] = {0};
    size_t ndim = 4;
    CHECK(beacon_tensor_shape(c, out_shape, &ndim));
    if (ndim != 2 || out_shape[0] != 2 || out_shape[1] != 2) {
        fprintf(stderr, "smoke: unexpected result shape: ndim=%zu [%lld, %lld]\n",
                ndim, (long long)out_shape[0], (long long)out_shape[1]);
        return 1;
    }

    // Read back and compare.
    float result[4] = {0};
    CHECK(beacon_tensor_read_f32(c, result, 4));
    const float expected[4] = {19.0f, 22.0f, 43.0f, 50.0f};
    for (int i = 0; i < 4; ++i) {
        if (!float_close(result[i], expected[i])) {
            fprintf(stderr, "smoke: result[%d] = %f, expected %f\n",
                    i, result[i], expected[i]);
            return 1;
        }
    }

    beacon_tensor_destroy(c);
    beacon_tensor_destroy(b);
    beacon_tensor_destroy(a);
    beacon_stream_destroy(stream);
    beacon_context_destroy(ctx);

    printf("smoke: matmul OK (result = [[19, 22], [43, 50]])\n");
    return 0;
}
