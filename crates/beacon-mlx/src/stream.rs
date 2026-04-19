//! MLX execution stream.

use std::sync::Arc;

use crate::context::MlxContext;
use crate::ffi;

/// An execution stream. Ops scheduled on the same stream serialize; different
/// streams may overlap.
///
/// Created via [`MlxContext::new_stream`].
pub struct MlxStream {
    pub(crate) inner: *mut ffi::BeaconStream,
    /// Keep the parent context alive.
    pub(crate) _ctx: Arc<MlxContext>,
}

impl std::fmt::Debug for MlxStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlxStream")
            .field("inner", &self.inner)
            .finish_non_exhaustive()
    }
}

impl Drop for MlxStream {
    fn drop(&mut self) {
        unsafe {
            ffi::beacon_stream_destroy(self.inner);
        }
    }
}
