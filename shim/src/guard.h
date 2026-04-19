// ABI exception guard.
// Every C ABI function in the shim wraps its body in guard(), which catches
// all exceptions and translates them to BeaconStatus codes + a thread-local
// error message. This enforces non-negotiable rule #2: no Rust-to-C++
// exceptions.

#pragma once

#include <exception>
#include <stdexcept>
#include <utility>

#include "internal.h"

namespace beacon {

template <typename F>
int32_t guard(F&& fn) noexcept {
    try {
        clear_error_message();
        return std::forward<F>(fn)();
    } catch (const std::bad_alloc& e) {
        set_error_message(e.what());
        return BEACON_ERR_OUT_OF_MEMORY;
    } catch (const std::invalid_argument& e) {
        set_error_message(e.what());
        return BEACON_ERR_INVALID_ARGUMENT;
    } catch (const std::out_of_range& e) {
        set_error_message(e.what());
        return BEACON_ERR_SHAPE_MISMATCH;
    } catch (const std::exception& e) {
        set_error_message(e.what());
        return BEACON_ERR_MLX_INTERNAL;
    } catch (...) {
        set_error_message("unknown C++ exception caught at ABI boundary");
        return BEACON_ERR_UNKNOWN;
    }
}

}  // namespace beacon
