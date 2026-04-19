cmake_minimum_required(VERSION 3.24)
project(beacon_shim CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Export compile_commands.json for tooling (clangd, rust-analyzer bindgen hooks).
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- MLX submodule --------------------------------------------------------
# Only add MLX on Apple platforms. On Linux/Windows we still compile the shim
# (with every op stubbed to BEACON_ERR_UNKNOWN) so that downstream Rust crates
# can link a consistent symbol surface across platforms.
if(APPLE)
    # Turn off MLX features we don't need so the submodule compile stays fast.
    set(MLX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(MLX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(MLX_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
    set(MLX_BUILD_PYTHON_STUBS OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    # Metal is the primary path; v0.1 ships with it on. Users without the
    # Xcode Metal Toolchain can configure with -DBEACON_SHIM_ENABLE_METAL=OFF
    # to fall back to CPU-only MLX (Accelerate).
    option(BEACON_SHIM_ENABLE_METAL "Build MLX with Metal backend" ON)
    if(NOT BEACON_SHIM_ENABLE_METAL)
        set(MLX_BUILD_METAL OFF CACHE BOOL "" FORCE)
    endif()
    add_subdirectory(third_party/mlx EXCLUDE_FROM_ALL)
endif()

# --- Shim static library --------------------------------------------------
add_library(beacon_shim STATIC
    src/errors.cpp
    src/beacon_shim.cpp
    src/tensor.cpp
    src/ops.cpp
    src/kernels.cpp
)

target_include_directories(beacon_shim
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(beacon_shim PUBLIC cxx_std_20)

if(APPLE)
    target_link_libraries(beacon_shim PUBLIC mlx)
else()
    # Stub build: compile the same sources with BEACON_NO_MLX; all ops return
    # BEACON_ERR_UNKNOWN until a non-MLX backend is wired in.
    target_compile_definitions(beacon_shim PUBLIC BEACON_NO_MLX)
endif()

# Treat warnings seriously in our own sources without poisoning the MLX build.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(beacon_shim PRIVATE
        -Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()

# --- Smoke test ----------------------------------------------------------
# A tiny C program that links the shim and exercises matmul end-to-end.
# Enabled by default; disable with -DBEACON_SHIM_BUILD_TESTS=OFF.
option(BEACON_SHIM_BUILD_TESTS "Build shim smoke tests" ON)
if(BEACON_SHIM_BUILD_TESTS AND APPLE)
    enable_testing()
    add_subdirectory(tests)
endif()
