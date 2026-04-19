// Internal C++ declarations shared across shim translation units.
// Not exposed across the C ABI (see beacon_shim.h for that).

#pragma once

#include <cstdint>
#include <string>

#include "beacon_shim.h"

#ifndef BEACON_NO_MLX
#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/stream.h"
#endif

namespace beacon {

// Set the thread-local error message. Safe to call with any string.
void set_error_message(const std::string& msg) noexcept;
void set_error_message(const char* msg) noexcept;

// Clear the thread-local error message.
void clear_error_message() noexcept;

// Get the thread-local error message pointer (valid until next shim call).
const char* get_error_message() noexcept;

#ifndef BEACON_NO_MLX

// Map a BeaconDtype to the underlying MLX Dtype used to store the buffer.
// Quantized dtypes are stored as packed uint32 arrays, matching MLX's internal
// representation for quantize()/quantized_matmul().
mlx::core::Dtype to_mlx_dtype(BeaconDtype dtype);

// Map a BeaconDtype to the corresponding element size in bytes for host I/O.
// For quantized dtypes, returns the size of the packed uint32 storage unit.
size_t dtype_size_bytes(BeaconDtype dtype);

// True if the dtype is a GGUF-style packed quantization.
bool dtype_is_quantized(BeaconDtype dtype);

// Best-effort reverse map from an MLX array's dtype back to a BeaconDtype,
// used when constructing output tensors where the caller did not specify a
// logical dtype. Unrecognised MLX dtypes map to BEACON_DTYPE_F32.
BeaconDtype from_mlx_dtype(mlx::core::Dtype dtype) noexcept;

#endif  // BEACON_NO_MLX

}  // namespace beacon

// === Opaque handle definitions ===
// These are only visible to shim .cpp files; Rust and external C code see
// forward-declared structs from beacon_shim.h.

#ifndef BEACON_NO_MLX

struct BeaconContext {
    mlx::core::Device device;
    BeaconContext() : device(mlx::core::default_device()) {}
};

struct BeaconStream {
    mlx::core::Stream stream;
    explicit BeaconStream(mlx::core::Stream s) : stream(s) {}
};

struct BeaconTensor {
    mlx::core::array arr;
    // Preserve the caller-facing dtype so Q4/Q5/Q6/Q8 tensors report their
    // original dtype via beacon_tensor_dtype even though they're stored as
    // packed uint32 internally.
    BeaconDtype logical_dtype;
    BeaconTensor(mlx::core::array a, BeaconDtype d)
        : arr(std::move(a)), logical_dtype(d) {}
};

#else  // BEACON_NO_MLX

struct BeaconContext { int _unused; };
struct BeaconStream { int _unused; };
struct BeaconTensor { int _unused; };

#endif  // BEACON_NO_MLX
