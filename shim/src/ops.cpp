// MLX op invocations. Every function wraps a single MLX call, plus input
// validation and result boxing. No business logic lives here.

#include <optional>
#include <stdexcept>

#include "guard.h"
#include "internal.h"

#ifndef BEACON_NO_MLX
#include "mlx/fast.h"
#include "mlx/ops.h"
#endif

namespace {

#ifndef BEACON_NO_MLX

// Wrap an MLX array result into a freshly allocated BeaconTensor, tagged with
// the best-effort logical dtype derived from the underlying MLX array.
BeaconTensor* box_result(mlx::core::array arr) {
    auto logical = beacon::from_mlx_dtype(arr.dtype());
    return new BeaconTensor(std::move(arr), logical);
}

// Ensure a list of tensor pointers are all non-null.
bool any_null(std::initializer_list<const void*> ptrs) {
    for (const void* p : ptrs) {
        if (p == nullptr) return true;
    }
    return false;
}

#endif  // BEACON_NO_MLX

}  // namespace

extern "C" {

int32_t beacon_op_matmul(
    BeaconContext* ctx, BeaconStream* stream,
    const BeaconTensor* a, const BeaconTensor* b,
    BeaconTensor** out) {
    return beacon::guard([&]() -> int32_t {
        if (any_null({ctx, stream, a, b, out})) {
            beacon::set_error_message("beacon_op_matmul: null argument");
            return BEACON_ERR_INVALID_ARGUMENT;
        }
#ifdef BEACON_NO_MLX
        return BEACON_ERR_UNKNOWN;
#else
        auto result = mlx::core::matmul(a->arr, b->arr, stream->stream);
        *out = box_result(std::move(result));
        return BEACON_OK;
#endif
    });
}

int32_t beacon_op_quantized_matmul(
    BeaconContext* ctx, BeaconStream* stream,
    const BeaconTensor* x, const BeaconTensor* w_quantized,
    const BeaconTensor* scales,
    int32_t group_size, int32_t bits,
    BeaconTensor** out) {
    return beacon::guard([&]() -> int32_t {
        if (any_null({ctx, stream, x, w_quantized, scales, out})) {
            beacon::set_error_message(
                "beacon_op_quantized_matmul: null argument");
            return BEACON_ERR_INVALID_ARGUMENT;
        }
#ifdef BEACON_NO_MLX
        return BEACON_ERR_UNKNOWN;
#else
        auto result = mlx::core::quantized_matmul(
            x->arr,
            w_quantized->arr,
            scales->arr,
            /*biases=*/std::nullopt,
            /*transpose=*/true,
            std::optional<int>{group_size},
            std::optional<int>{bits},
            /*mode=*/"affine",
            stream->stream);
        *out = box_result(std::move(result));
        return BEACON_OK;
#endif
    });
}

int32_t beacon_op_rms_norm(
    BeaconContext* ctx, BeaconStream* stream,
    const BeaconTensor* x, const BeaconTensor* weight,
    float eps,
    BeaconTensor** out) {
    return beacon::guard([&]() -> int32_t {
        if (any_null({ctx, stream, x, weight, out})) {
            beacon::set_error_message("beacon_op_rms_norm: null argument");
            return BEACON_ERR_INVALID_ARGUMENT;
        }
#ifdef BEACON_NO_MLX
        return BEACON_ERR_UNKNOWN;
#else
        auto result = mlx::core::fast::rms_norm(
            x->arr,
            std::optional<mlx::core::array>{weight->arr},
            eps,
            stream->stream);
        *out = box_result(std::move(result));
        return BEACON_OK;
#endif
    });
}

int32_t beacon_op_rope(
    BeaconContext* ctx, BeaconStream* stream,
    const BeaconTensor* x,
    int32_t position_offset, float theta, int32_t dim,
    BeaconTensor** out) {
    return beacon::guard([&]() -> int32_t {
        if (any_null({ctx, stream, x, out})) {
            beacon::set_error_message("beacon_op_rope: null argument");
            return BEACON_ERR_INVALID_ARGUMENT;
        }
#ifdef BEACON_NO_MLX
        return BEACON_ERR_UNKNOWN;
#else
        auto result = mlx::core::fast::rope(
            x->arr,
            /*dims=*/dim,
            /*traditional=*/false,
            /*base=*/std::optional<float>{theta},
            /*scale=*/1.0f,
            /*offset=*/position_offset,
            /*freqs=*/std::nullopt,
            stream->stream);
        *out = box_result(std::move(result));
        return BEACON_OK;
#endif
    });
}

int32_t beacon_op_silu(
    BeaconContext* ctx, BeaconStream* stream,
    const BeaconTensor* x,
    BeaconTensor** out) {
    return beacon::guard([&]() -> int32_t {
        if (any_null({ctx, stream, x, out})) {
            beacon::set_error_message("beacon_op_silu: null argument");
            return BEACON_ERR_INVALID_ARGUMENT;
        }
#ifdef BEACON_NO_MLX
        return BEACON_ERR_UNKNOWN;
#else
        // SiLU(x) = x * sigmoid(x). MLX has no fused silu in ops.h; composed
        // via sigmoid + multiply so kernel fusion at eval() collapses it.
        auto sig = mlx::core::sigmoid(x->arr, stream->stream);
        auto result = mlx::core::multiply(x->arr, sig, stream->stream);
        *out = box_result(std::move(result));
        return BEACON_OK;
#endif
    });
}

int32_t beacon_op_softmax(
    BeaconContext* ctx, BeaconStream* stream,
    const BeaconTensor* x, int32_t axis,
    BeaconTensor** out) {
    return beacon::guard([&]() -> int32_t {
        if (any_null({ctx, stream, x, out})) {
            beacon::set_error_message("beacon_op_softmax: null argument");
            return BEACON_ERR_INVALID_ARGUMENT;
        }
#ifdef BEACON_NO_MLX
        return BEACON_ERR_UNKNOWN;
#else
        auto result = mlx::core::softmax(
            x->arr,
            std::vector<int>{static_cast<int>(axis)},
            /*precise=*/false,
            stream->stream);
        *out = box_result(std::move(result));
        return BEACON_OK;
#endif
    });
}

int32_t beacon_op_attention(
    BeaconContext* ctx, BeaconStream* stream,
    const BeaconTensor* q, const BeaconTensor* k, const BeaconTensor* v,
    const BeaconTensor* mask,
    float scale,
    BeaconTensor** out) {
    return beacon::guard([&]() -> int32_t {
        if (any_null({ctx, stream, q, k, v, out})) {
            beacon::set_error_message("beacon_op_attention: null argument");
            return BEACON_ERR_INVALID_ARGUMENT;
        }
#ifdef BEACON_NO_MLX
        return BEACON_ERR_UNKNOWN;
#else
        std::optional<mlx::core::array> mask_arr;
        std::string mask_mode;  // "" → no mask (or use provided mask_arr)
        if (mask != nullptr) {
            mask_arr = mask->arr;
            mask_mode = "array";
        }
        auto result = mlx::core::fast::scaled_dot_product_attention(
            q->arr,
            k->arr,
            v->arr,
            scale,
            mask_mode,
            mask_arr,
            /*sinks=*/std::nullopt,
            stream->stream);
        *out = box_result(std::move(result));
        return BEACON_OK;
#endif
    });
}

int32_t beacon_op_add(
    BeaconContext* ctx, BeaconStream* stream,
    const BeaconTensor* a, const BeaconTensor* b,
    BeaconTensor** out) {
    return beacon::guard([&]() -> int32_t {
        if (any_null({ctx, stream, a, b, out})) {
            beacon::set_error_message("beacon_op_add: null argument");
            return BEACON_ERR_INVALID_ARGUMENT;
        }
#ifdef BEACON_NO_MLX
        return BEACON_ERR_UNKNOWN;
#else
        auto result = mlx::core::add(a->arr, b->arr, stream->stream);
        *out = box_result(std::move(result));
        return BEACON_OK;
#endif
    });
}

int32_t beacon_op_elementwise_mul(
    BeaconContext* ctx, BeaconStream* stream,
    const BeaconTensor* a, const BeaconTensor* b,
    BeaconTensor** out) {
    return beacon::guard([&]() -> int32_t {
        if (any_null({ctx, stream, a, b, out})) {
            beacon::set_error_message(
                "beacon_op_elementwise_mul: null argument");
            return BEACON_ERR_INVALID_ARGUMENT;
        }
#ifdef BEACON_NO_MLX
        return BEACON_ERR_UNKNOWN;
#else
        auto result = mlx::core::multiply(a->arr, b->arr, stream->stream);
        *out = box_result(std::move(result));
        return BEACON_OK;
#endif
    });
}

int32_t beacon_op_kv_cache_update(
    BeaconContext* ctx, BeaconStream* stream,
    BeaconTensor* cache_k, BeaconTensor* cache_v,
    const BeaconTensor* new_k, const BeaconTensor* new_v,
    int64_t position,
    BeaconTensor** out_k_view, BeaconTensor** out_v_view) {
    return beacon::guard([&]() -> int32_t {
        if (any_null(
                {ctx, stream, cache_k, cache_v, new_k, new_v,
                 out_k_view, out_v_view})) {
            beacon::set_error_message(
                "beacon_op_kv_cache_update: null argument");
            return BEACON_ERR_INVALID_ARGUMENT;
        }
#ifdef BEACON_NO_MLX
        return BEACON_ERR_UNKNOWN;
#else
        // Expect rank-3 caches: [max_context, num_kv_heads, head_dim].
        const auto& cache_shape = cache_k->arr.shape();
        if (cache_shape.size() != 3) {
            beacon::set_error_message(
                "beacon_op_kv_cache_update: cache must be rank-3 "
                "[max_context, num_kv_heads, head_dim]");
            return BEACON_ERR_SHAPE_MISMATCH;
        }
        const auto n_heads = cache_shape[1];
        const auto head_dim = cache_shape[2];
        const auto max_context = static_cast<int64_t>(cache_shape[0]);
        if (position < 0 || position >= max_context) {
            beacon::set_error_message(
                "beacon_op_kv_cache_update: position out of range");
            return BEACON_ERR_INVALID_ARGUMENT;
        }
        const int32_t pos = static_cast<int32_t>(position);
        mlx::core::Shape start_k = {pos, 0, 0};
        mlx::core::Shape stop_k = {pos + 1, n_heads, head_dim};
        // slice_update is functional — returns a new array with the slot
        // written. Replacing the cache_k/v array handle does not copy memory;
        // MLX reuses the underlying storage when possible.
        cache_k->arr = mlx::core::slice_update(
            cache_k->arr, new_k->arr, start_k, stop_k, stream->stream);
        cache_v->arr = mlx::core::slice_update(
            cache_v->arr, new_v->arr, start_k, stop_k, stream->stream);
        // Return views over [0..=position] for attention.
        mlx::core::Shape view_start = {0, 0, 0};
        mlx::core::Shape view_stop = {pos + 1, n_heads, head_dim};
        auto k_view = mlx::core::slice(
            cache_k->arr, view_start, view_stop, stream->stream);
        auto v_view = mlx::core::slice(
            cache_v->arr, view_start, view_stop, stream->stream);
        *out_k_view = box_result(std::move(k_view));
        *out_v_view = box_result(std::move(v_view));
        return BEACON_OK;
#endif
    });
}

int32_t beacon_op_reshape(
    BeaconContext* ctx, BeaconStream* stream,
    const BeaconTensor* x,
    const int64_t* new_shape, size_t new_ndim,
    BeaconTensor** out) {
    return beacon::guard([&]() -> int32_t {
        if (any_null({ctx, stream, x, out}) || new_shape == nullptr) {
            beacon::set_error_message("beacon_op_reshape: null argument");
            return BEACON_ERR_INVALID_ARGUMENT;
        }
#ifdef BEACON_NO_MLX
        return BEACON_ERR_UNKNOWN;
#else
        mlx::core::Shape shape(new_ndim);
        for (size_t i = 0; i < new_ndim; ++i) {
            shape[i] = static_cast<int>(new_shape[i]);
        }
        auto result = mlx::core::reshape(x->arr, shape, stream->stream);
        *out = box_result(std::move(result));
        return BEACON_OK;
#endif
    });
}

int32_t beacon_op_transpose(
    BeaconContext* ctx, BeaconStream* stream,
    const BeaconTensor* x,
    BeaconTensor** out) {
    return beacon::guard([&]() -> int32_t {
        if (any_null({ctx, stream, x, out})) {
            beacon::set_error_message("beacon_op_transpose: null argument");
            return BEACON_ERR_INVALID_ARGUMENT;
        }
#ifdef BEACON_NO_MLX
        return BEACON_ERR_UNKNOWN;
#else
        auto ndim = x->arr.ndim();
        if (ndim < 2) {
            beacon::set_error_message("beacon_op_transpose: need at least 2 dims");
            return BEACON_ERR_INVALID_ARGUMENT;
        }
        // Swap the last two axes.
        auto result = mlx::core::swapaxes(
            x->arr,
            static_cast<int>(ndim - 2),
            static_cast<int>(ndim - 1),
            stream->stream);
        *out = box_result(std::move(result));
        return BEACON_OK;
#endif
    });
}

int32_t beacon_op_embedding(
    BeaconContext* ctx, BeaconStream* stream,
    const BeaconTensor* weight, const BeaconTensor* indices,
    BeaconTensor** out) {
    return beacon::guard([&]() -> int32_t {
        if (any_null({ctx, stream, weight, indices, out})) {
            beacon::set_error_message("beacon_op_embedding: null argument");
            return BEACON_ERR_INVALID_ARGUMENT;
        }
#ifdef BEACON_NO_MLX
        return BEACON_ERR_UNKNOWN;
#else
        // take(weight, indices, axis=0) selects rows by index.
        auto result = mlx::core::take(weight->arr, indices->arr, 0, stream->stream);
        *out = box_result(std::move(result));
        return BEACON_OK;
#endif
    });
}

}  // extern "C"
