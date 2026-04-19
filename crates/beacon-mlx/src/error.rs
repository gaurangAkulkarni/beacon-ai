//! Error types mapping `BeaconStatus` codes to Rust.

use crate::ffi;

/// Errors returned by the MLX shim.
#[derive(Debug, thiserror::Error)]
pub enum MlxError {
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error("out of memory")]
    OutOfMemory,

    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),

    #[error("unsupported dtype")]
    UnsupportedDtype,

    #[error("Metal kernel compilation failed: {0}")]
    MetalCompile(String),

    #[error("MLX internal error: {0}")]
    MlxInternal(String),

    #[error("unknown error: {0}")]
    Unknown(String),
}

/// Convert a `BeaconStatus` return code into a `Result`.
///
/// On non-zero status, reads the thread-local error message from the shim.
pub(crate) fn status_to_result(status: i32) -> Result<(), MlxError> {
    if status == 0 {
        return Ok(());
    }
    let msg = unsafe {
        let ptr = ffi::beacon_last_error_message();
        if ptr.is_null() {
            String::from("<no message>")
        } else {
            std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };
    Err(match status {
        1 => MlxError::InvalidArgument(msg),
        2 => MlxError::OutOfMemory,
        3 => MlxError::ShapeMismatch(msg),
        4 => MlxError::UnsupportedDtype,
        5 => MlxError::MetalCompile(msg),
        6 => MlxError::MlxInternal(msg),
        _ => MlxError::Unknown(msg),
    })
}
