//! MLX tensor — opaque handle with RAII cleanup.

use std::sync::Arc;

use crate::context::MlxContext;
use crate::dtype::Dtype;
use crate::error::{status_to_result, MlxError};
use crate::ffi;
use crate::stream::MlxStream;

/// An opaque tensor handle. Dropping releases the shim-side resources.
///
/// Tensors are lazy — operations build a computation graph that is only
/// materialised when [`MlxTensor::eval`] or [`MlxTensor::read_f32`] is called.
pub struct MlxTensor {
    pub(crate) inner: *mut ffi::BeaconTensor,
    pub(crate) ctx: Arc<MlxContext>,
    /// Keeps the backing mmap alive when the tensor was created from mmap.
    _backing: Option<Arc<memmap2::Mmap>>,
}

impl std::fmt::Debug for MlxTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlxTensor")
            .field("shape", &self.shape())
            .field("dtype", &self.dtype())
            .finish()
    }
}

// SAFETY: Tensor handles are thread-safe when used through the C ABI (which
// serialises via streams). We never dereference the pointer in Rust.
unsafe impl Send for MlxTensor {}
unsafe impl Sync for MlxTensor {}

impl MlxTensor {
    /// Create a tensor from a memory-mapped buffer **without copying**.
    ///
    /// The `mmap` is kept alive (via `Arc`) for the tensor's lifetime. The
    /// `offset` is a byte offset into the mmap where the tensor data begins.
    pub fn from_mmap(
        ctx: Arc<MlxContext>,
        mmap: Arc<memmap2::Mmap>,
        offset: usize,
        shape: &[i64],
        dtype: Dtype,
    ) -> Result<Self, MlxError> {
        let data_ptr = unsafe { mmap.as_ptr().add(offset).cast::<std::ffi::c_void>() };
        let mut tensor_ptr: *mut ffi::BeaconTensor = std::ptr::null_mut();
        let status = unsafe {
            ffi::beacon_tensor_from_mmap(
                ctx.inner,
                data_ptr,
                shape.as_ptr(),
                shape.len(),
                dtype.to_ffi(),
                &raw mut tensor_ptr,
            )
        };
        status_to_result(status)?;
        Ok(Self {
            inner: tensor_ptr,
            ctx,
            _backing: Some(mmap),
        })
    }

    /// Create a zero-initialised tensor in unified memory.
    pub fn zeros(ctx: Arc<MlxContext>, shape: &[i64], dtype: Dtype) -> Result<Self, MlxError> {
        let mut tensor_ptr: *mut ffi::BeaconTensor = std::ptr::null_mut();
        let status = unsafe {
            ffi::beacon_tensor_zeros(
                ctx.inner,
                shape.as_ptr(),
                shape.len(),
                dtype.to_ffi(),
                &raw mut tensor_ptr,
            )
        };
        status_to_result(status)?;
        Ok(Self {
            inner: tensor_ptr,
            ctx,
            _backing: None,
        })
    }

    /// Wrap a raw shim pointer. Used internally by ops that produce new tensors.
    pub(crate) fn from_raw(ctx: Arc<MlxContext>, ptr: *mut ffi::BeaconTensor) -> Self {
        Self {
            inner: ptr,
            ctx,
            _backing: None,
        }
    }

    /// Return the tensor's shape.
    pub fn shape(&self) -> Vec<i64> {
        let mut ndim: usize = 8;
        let mut buf = vec![0i64; ndim];
        let status =
            unsafe { ffi::beacon_tensor_shape(self.inner, buf.as_mut_ptr(), &raw mut ndim) };
        // If ndim > 8 we need a bigger buffer (unlikely for LLM tensors).
        if status != 0 && ndim > buf.len() {
            buf.resize(ndim, 0);
            unsafe {
                ffi::beacon_tensor_shape(self.inner, buf.as_mut_ptr(), &raw mut ndim);
            }
        }
        buf.truncate(ndim);
        buf
    }

    /// Return the tensor's dtype.
    pub fn dtype(&self) -> Dtype {
        let mut d = ffi::BeaconDtype::BEACON_DTYPE_F32;
        unsafe {
            ffi::beacon_tensor_dtype(self.inner, &raw mut d);
        }
        Dtype::from_ffi(d)
    }

    /// Force evaluation of the lazy MLX graph for this tensor.
    pub fn eval(&self, stream: &MlxStream) -> Result<(), MlxError> {
        let status = unsafe { ffi::beacon_tensor_eval(self.inner, stream.inner) };
        status_to_result(status)
    }

    /// Read tensor values into a host `Vec<f32>`. Triggers eval first.
    ///
    /// The caller must know how many elements to read (`n_elements`).
    pub fn read_f32(&self, n_elements: usize) -> Result<Vec<f32>, MlxError> {
        let mut buf = vec![0f32; n_elements];
        let status =
            unsafe { ffi::beacon_tensor_read_f32(self.inner, buf.as_mut_ptr(), n_elements) };
        status_to_result(status)?;
        Ok(buf)
    }
}

impl Drop for MlxTensor {
    fn drop(&mut self) {
        unsafe {
            ffi::beacon_tensor_destroy(self.inner);
        }
    }
}
