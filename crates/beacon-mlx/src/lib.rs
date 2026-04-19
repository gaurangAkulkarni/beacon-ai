//! Rust FFI bindings to the Beacon MLX shim.
//!
//! This crate wraps the C ABI exposed by `shim/include/beacon_shim.h` in safe
//! Rust types with RAII semantics. The primary types are:
//!
//! - [`MlxContext`] — a shim context; create once per process.
//! - [`MlxStream`] — an execution stream; ops on the same stream serialize.
//! - [`MlxTensor`] — an opaque tensor handle with automatic cleanup.
//!
//! Compute operations live in the [`ops`] module as free functions.

#[allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code,
    clippy::unreadable_literal
)]
mod ffi {
    include!(concat!(env!("OUT_DIR"), "/ffi.rs"));
}

mod context;
mod dtype;
mod error;
pub mod ops;
mod stream;
mod tensor;

pub use context::MlxContext;
pub use dtype::Dtype;
pub use error::MlxError;
pub use stream::MlxStream;
pub use tensor::MlxTensor;

#[cfg(test)]
mod tests;
