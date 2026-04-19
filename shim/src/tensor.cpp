// Tensor lifecycle, dtype mapping, introspection, and host read.

#include <cstring>
#include <stdexcept>
#include <vector>

#include "guard.h"
#include "internal.h"

#ifndef BEACON_NO_MLX
#include "mlx/array.h"
#include "mlx/ops.h"
#endif

namespace beacon {

#ifndef BEACON_NO_MLX

mlx::core::Dtype to_mlx_dtype(BeaconDtype dtype) {
    using mlx::core::Dtype;
    switch (dtype) {
        case BEACON_DTYPE_F32:
            return mlx::core::float32;
        case BEACON_DTYPE_F16:
            return mlx::core::float16;
        case BEACON_DTYPE_BF16:
            return mlx::core::bfloat16;
        case BEACON_DTYPE_I32:
            return mlx::core::int32;
        case BEACON_DTYPE_I8:
            return mlx::core::int8;
        // Quantized dtypes are stored as packed uint32 to match MLX's
        // quantize()/quantized_matmul() representation.
        case BEACON_DTYPE_Q4_0:
        case BEACON_DTYPE_Q4_K:
        case BEACON_DTYPE_Q5_K:
        case BEACON_DTYPE_Q6_K:
        case BEACON_DTYPE_Q8_0:
            return mlx::core::uint32;
    }
    throw std::invalid_argument("unknown BeaconDtype");
}

size_t dtype_size_bytes(BeaconDtype dtype) {
    switch (dtype) {
        case BEACON_DTYPE_F32:
        case BEACON_DTYPE_I32:
            return 4;
        case BEACON_DTYPE_F16:
        case BEACON_DTYPE_BF16:
            return 2;
        case BEACON_DTYPE_I8:
            return 1;
        case BEACON_DTYPE_Q4_0:
        case BEACON_DTYPE_Q4_K:
        case BEACON_DTYPE_Q5_K:
        case BEACON_DTYPE_Q6_K:
        case BEACON_DTYPE_Q8_0:
            return 4;  // packed uint32 storage unit
    }
    throw std::invalid_argument("unknown BeaconDtype");
}

bool dtype_is_quantized(BeaconDtype dtype) {
    switch (dtype) {
        case BEACON_DTYPE_Q4_0:
        case BEACON_DTYPE_Q4_K:
        case BEACON_DTYPE_Q5_K:
        case BEACON_DTYPE_Q6_K:
        case BEACON_DTYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

BeaconDtype from_mlx_dtype(mlx::core::Dtype dtype) noexcept {
    using mlx::core::Dtype;
    switch (dtype.val()) {
        case Dtype::Val::float32:
            return BEACON_DTYPE_F32;
        case Dtype::Val::float16:
            return BEACON_DTYPE_F16;
        case Dtype::Val::bfloat16:
            return BEACON_DTYPE_BF16;
        case Dtype::Val::int32:
            return BEACON_DTYPE_I32;
        case Dtype::Val::int8:
            return BEACON_DTYPE_I8;
        default:
            return BEACON_DTYPE_F32;
    }
}

namespace {

mlx::core::Shape to_mlx_shape(const int64_t* shape, size_t ndim) {
    if (shape == nullptr && ndim != 0) {
        throw std::invalid_argument("shape pointer is null but ndim != 0");
    }
    mlx::core::Shape s;
    s.reserve(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        int64_t dim = shape[i];
        if (dim < 0 || dim > static_cast<int64_t>(INT32_MAX)) {
            throw std::invalid_argument("shape dimension out of int32 range");
        }
        s.push_back(static_cast<mlx::core::ShapeElem>(dim));
    }
    return s;
}

}  // namespace

#endif  // BEACON_NO_MLX

}  // namespace beacon

extern "C" {

int32_t beacon_tensor_from_mmap(
    BeaconContext* ctx,
    const void* data,
    const int64_t* shape, size_t ndim,
    BeaconDtype dtype,
    BeaconTensor** out_tensor) {
    return beacon::guard([&]() -> int32_t {
        if (ctx == nullptr || data == nullptr || out_tensor == nullptr) {
            beacon::set_error_message(
                "beacon_tensor_from_mmap: null ctx/data/out_tensor");
            return BEACON_ERR_INVALID_ARGUMENT;
        }
#ifdef BEACON_NO_MLX
        beacon::set_error_message("MLX not available");
        return BEACON_ERR_UNKNOWN;
#else
        auto mlx_shape = beacon::to_mlx_shape(shape, ndim);
        auto mlx_dtype = beacon::to_mlx_dtype(dtype);
        // Zero-copy: pass the mmap pointer with a no-op deleter. The caller
        // (Rust: beacon-mlx) keeps the Mmap alive for the tensor's lifetime,
        // per ABI contract. This satisfies non-negotiable rule #3 (no tensor
        // copies on MLX backend for weights).
        auto arr = mlx::core::array(
            const_cast<void*>(data),
            std::move(mlx_shape),
            mlx_dtype,
            [](void*) {});
        *out_tensor = new BeaconTensor(std::move(arr), dtype);
        return BEACON_OK;
#endif
    });
}

int32_t beacon_tensor_zeros(
    BeaconContext* ctx,
    const int64_t* shape, size_t ndim,
    BeaconDtype dtype,
    BeaconTensor** out_tensor) {
    return beacon::guard([&]() -> int32_t {
        if (ctx == nullptr || out_tensor == nullptr) {
            beacon::set_error_message(
                "beacon_tensor_zeros: null ctx/out_tensor");
            return BEACON_ERR_INVALID_ARGUMENT;
        }
#ifdef BEACON_NO_MLX
        beacon::set_error_message("MLX not available");
        return BEACON_ERR_UNKNOWN;
#else
        auto mlx_shape = beacon::to_mlx_shape(shape, ndim);
        auto mlx_dtype = beacon::to_mlx_dtype(dtype);
        auto arr = mlx::core::zeros(mlx_shape, mlx_dtype);
        *out_tensor = new BeaconTensor(std::move(arr), dtype);
        return BEACON_OK;
#endif
    });
}

void beacon_tensor_destroy(BeaconTensor* tensor) {
    delete tensor;
}

int32_t beacon_tensor_shape(
    const BeaconTensor* t, int64_t* out_shape, size_t* out_ndim) {
    return beacon::guard([&]() -> int32_t {
        if (t == nullptr || out_ndim == nullptr) {
            beacon::set_error_message(
                "beacon_tensor_shape: null tensor/out_ndim");
            return BEACON_ERR_INVALID_ARGUMENT;
        }
#ifdef BEACON_NO_MLX
        return BEACON_ERR_UNKNOWN;
#else
        const auto& s = t->arr.shape();
        size_t capacity = *out_ndim;
        *out_ndim = s.size();
        if (out_shape == nullptr) {
            return capacity == 0 ? BEACON_OK : BEACON_ERR_INVALID_ARGUMENT;
        }
        if (capacity < s.size()) {
            beacon::set_error_message(
                "beacon_tensor_shape: out_shape capacity < tensor ndim");
            return BEACON_ERR_INVALID_ARGUMENT;
        }
        for (size_t i = 0; i < s.size(); ++i) {
            out_shape[i] = static_cast<int64_t>(s[i]);
        }
        return BEACON_OK;
#endif
    });
}

int32_t beacon_tensor_dtype(const BeaconTensor* t, BeaconDtype* out_dtype) {
    return beacon::guard([&]() -> int32_t {
        if (t == nullptr || out_dtype == nullptr) {
            beacon::set_error_message(
                "beacon_tensor_dtype: null tensor/out_dtype");
            return BEACON_ERR_INVALID_ARGUMENT;
        }
        *out_dtype = t->logical_dtype;
        return BEACON_OK;
    });
}

int32_t beacon_tensor_eval(BeaconTensor* t, BeaconStream* /*stream*/) {
    return beacon::guard([&]() -> int32_t {
        if (t == nullptr) {
            beacon::set_error_message("beacon_tensor_eval: null tensor");
            return BEACON_ERR_INVALID_ARGUMENT;
        }
#ifdef BEACON_NO_MLX
        return BEACON_ERR_UNKNOWN;
#else
        // MLX's eval() does not take a stream argument — ops carry their stream
        // through the graph. We accept a stream parameter for future-proofing
        // and to keep the ABI stable.
        t->arr.eval();
        return BEACON_OK;
#endif
    });
}

int32_t beacon_tensor_read_f32(
    const BeaconTensor* t, float* out, size_t n_elements) {
    return beacon::guard([&]() -> int32_t {
        if (t == nullptr || out == nullptr) {
            beacon::set_error_message(
                "beacon_tensor_read_f32: null tensor/out");
            return BEACON_ERR_INVALID_ARGUMENT;
        }
#ifdef BEACON_NO_MLX
        return BEACON_ERR_UNKNOWN;
#else
        // Trigger eval; cast through mutable since MLX::array::eval is
        // non-const. The observable tensor value is unchanged.
        auto& mut = const_cast<mlx::core::array&>(t->arr);
        mut.eval();
        if (mut.dtype() != mlx::core::float32) {
            beacon::set_error_message(
                "beacon_tensor_read_f32: tensor dtype is not float32");
            return BEACON_ERR_UNSUPPORTED_DTYPE;
        }
        if (mut.size() != n_elements) {
            beacon::set_error_message(
                "beacon_tensor_read_f32: n_elements mismatch");
            return BEACON_ERR_SHAPE_MISMATCH;
        }
        std::memcpy(out, mut.data<float>(), mut.nbytes());
        return BEACON_OK;
#endif
    });
}

}  // extern "C"
