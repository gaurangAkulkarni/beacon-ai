// Thread-local error message storage and translation helpers.
// No exceptions cross the ABI — see guard() in beacon_shim.cpp.

#include "internal.h"

namespace beacon {

namespace {

// thread_local so concurrent callers never see each other's messages.
// Lives for the process lifetime; the returned C string is valid until the
// next shim call on the same thread overwrites it.
thread_local std::string g_last_error;

}  // namespace

void set_error_message(const std::string& msg) noexcept {
    try {
        g_last_error = msg;
    } catch (...) {
        // If even string assignment throws, fall back to empty.
        g_last_error.clear();
    }
}

void set_error_message(const char* msg) noexcept {
    if (msg == nullptr) {
        clear_error_message();
        return;
    }
    try {
        g_last_error.assign(msg);
    } catch (...) {
        g_last_error.clear();
    }
}

void clear_error_message() noexcept {
    g_last_error.clear();
}

const char* get_error_message() noexcept {
    return g_last_error.c_str();
}

}  // namespace beacon

extern "C" const char* beacon_last_error_message(void) {
    return beacon::get_error_message();
}
