//! MLX context — create once per process.

use std::sync::Arc;

use crate::error::{status_to_result, MlxError};
use crate::ffi;
use crate::stream::MlxStream;

/// A shim context wrapping `BeaconContext`.
///
/// Thread-safe (the underlying MLX context is thread-safe). Create once per
/// process via [`MlxContext::new`] and share via `Arc`.
pub struct MlxContext {
    pub(crate) inner: *mut ffi::BeaconContext,
}

// SAFETY: The MLX context is internally thread-safe. All mutations go through
// the C ABI which serialises via streams.
unsafe impl Send for MlxContext {}
unsafe impl Sync for MlxContext {}

impl std::fmt::Debug for MlxContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlxContext")
            .field("inner", &self.inner)
            .finish()
    }
}

impl MlxContext {
    /// Create a new MLX context.
    pub fn new() -> Result<Arc<Self>, MlxError> {
        let mut ptr: *mut ffi::BeaconContext = std::ptr::null_mut();
        let status = unsafe { ffi::beacon_context_create(&raw mut ptr) };
        status_to_result(status)?;
        Ok(Arc::new(Self { inner: ptr }))
    }

    /// Create a new execution stream bound to this context.
    pub fn new_stream(self: &Arc<Self>) -> Result<MlxStream, MlxError> {
        let mut ptr: *mut ffi::BeaconStream = std::ptr::null_mut();
        let status = unsafe { ffi::beacon_stream_create(self.inner, &raw mut ptr) };
        status_to_result(status)?;
        Ok(MlxStream {
            inner: ptr,
            _ctx: Arc::clone(self),
        })
    }
}

impl Drop for MlxContext {
    fn drop(&mut self) {
        unsafe {
            ffi::beacon_context_destroy(self.inner);
        }
    }
}
