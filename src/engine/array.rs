use std::{ffi::c_void, fmt, rc::Rc};

use crate::engine::{
	error::{Error, Result, check, install_error_handler, take_last_error},
	stream::{ensure_runtime, stream},
};

/// Element dtype of an [`Array`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
	Bool,
	UInt8,
	UInt16,
	UInt32,
	UInt64,
	Int8,
	Int16,
	Int32,
	Int64,
	Float16,
	Float32,
	Float64,
	BFloat16,
	Complex64,
	/// MLX dtype added after this binding snapshot.
	Unknown(crate::engine::sys::mlx_dtype),
}

impl Dtype {
	pub(crate) fn to_raw(self) -> crate::engine::sys::mlx_dtype {
		use crate::engine::sys::*;
		match self {
			Dtype::Bool => mlx_dtype__MLX_BOOL,
			Dtype::UInt8 => mlx_dtype__MLX_UINT8,
			Dtype::UInt16 => mlx_dtype__MLX_UINT16,
			Dtype::UInt32 => mlx_dtype__MLX_UINT32,
			Dtype::UInt64 => mlx_dtype__MLX_UINT64,
			Dtype::Int8 => mlx_dtype__MLX_INT8,
			Dtype::Int16 => mlx_dtype__MLX_INT16,
			Dtype::Int32 => mlx_dtype__MLX_INT32,
			Dtype::Int64 => mlx_dtype__MLX_INT64,
			Dtype::Float16 => mlx_dtype__MLX_FLOAT16,
			Dtype::Float32 => mlx_dtype__MLX_FLOAT32,
			Dtype::Float64 => mlx_dtype__MLX_FLOAT64,
			Dtype::BFloat16 => mlx_dtype__MLX_BFLOAT16,
			Dtype::Complex64 => mlx_dtype__MLX_COMPLEX64,
			Dtype::Unknown(raw) => raw,
		}
	}

	pub(crate) fn from_raw(raw: crate::engine::sys::mlx_dtype) -> Self {
		match raw {
			crate::engine::sys::mlx_dtype__MLX_BOOL => Dtype::Bool,
			crate::engine::sys::mlx_dtype__MLX_UINT8 => Dtype::UInt8,
			crate::engine::sys::mlx_dtype__MLX_UINT16 => Dtype::UInt16,
			crate::engine::sys::mlx_dtype__MLX_UINT32 => Dtype::UInt32,
			crate::engine::sys::mlx_dtype__MLX_UINT64 => Dtype::UInt64,
			crate::engine::sys::mlx_dtype__MLX_INT8 => Dtype::Int8,
			crate::engine::sys::mlx_dtype__MLX_INT16 => Dtype::Int16,
			crate::engine::sys::mlx_dtype__MLX_INT32 => Dtype::Int32,
			crate::engine::sys::mlx_dtype__MLX_INT64 => Dtype::Int64,
			crate::engine::sys::mlx_dtype__MLX_FLOAT16 => Dtype::Float16,
			crate::engine::sys::mlx_dtype__MLX_FLOAT32 => Dtype::Float32,
			crate::engine::sys::mlx_dtype__MLX_FLOAT64 => Dtype::Float64,
			crate::engine::sys::mlx_dtype__MLX_BFLOAT16 => Dtype::BFloat16,
			crate::engine::sys::mlx_dtype__MLX_COMPLEX64 => Dtype::Complex64,
			other => Dtype::Unknown(other),
		}
	}
}

/// An owned, lazily-evaluated MLX array.
///
/// Cloning is an infallible Rust reference-count bump over one validated
/// native handle. Arrays are deliberately confined to their creating thread;
/// a loaded session and all arrays it owns stay on one inference worker.
// emelex patch: checked construction plus thread-confined shared ownership.
#[derive(Clone)]
pub struct Array {
	pub(crate) raw: crate::engine::sys::mlx_array,
	_owner: Rc<ArrayOwner>,
}

struct ArrayOwner(crate::engine::sys::mlx_array);

impl Drop for ArrayOwner {
	fn drop(&mut self) {
		unsafe {
			let _ = crate::engine::sys::mlx_array_free(self.0);
		}
	}
}

impl fmt::Debug for Array {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_struct("Array")
			.field("shape", &self.shape())
			.field("dtype", &self.dtype())
			.finish()
	}
}

impl Array {
	pub(crate) fn from_raw(raw: crate::engine::sys::mlx_array) -> Result<Self> {
		if raw.ctx.is_null() {
			return Err(take_last_error("MLX returned an empty array handle"));
		}
		Ok(Array {
			raw,
			_owner: Rc::new(ArrayOwner(raw)),
		})
	}

	/// Allocate the output handle used by mlx-c out-parameters.
	pub(crate) fn new_handle() -> crate::engine::sys::mlx_array {
		install_error_handler();
		unsafe { crate::engine::sys::mlx_array_new() }
	}

	/// Create an array by copying `data` with the given shape.
	///
	/// Dimensions must be non-negative, their checked product must fit
	/// `usize`, and that product must equal `data.len()`.
	pub fn from_slice<T: ArrayElement>(data: &[T], shape: &[i32]) -> Result<Self> {
		install_error_handler();
		let rank = i32::try_from(shape.len())
			.map_err(|_| Error::Config("array rank exceeds native i32 range".to_string()))?;
		let expected =
			shape
				.iter()
				.enumerate()
				.try_fold(1_usize, |elements, (axis, &dimension)| {
					let dimension = usize::try_from(dimension).map_err(|_| {
						Error::Config(format!(
							"array shape dimension {axis} is negative: {dimension}"
						))
					})?;
					elements.checked_mul(dimension).ok_or_else(|| {
						Error::Config(format!(
							"array shape element count overflows usize: {shape:?}"
						))
					})
				})?;
		if data.len() != expected {
			return Err(Error::Config(format!(
				"array data length {} does not match shape {shape:?} ({expected} elements)",
				data.len()
			)));
		}
		ensure_runtime()?;
		unsafe {
			let raw = crate::engine::sys::mlx_array_new_data(
				data.as_ptr() as *const c_void,
				shape.as_ptr(),
				rank,
				T::DTYPE.to_raw(),
			);
			Self::from_raw(raw)
		}
	}

	/// Create a rank-0 (scalar) float32 array.
	pub fn scalar_f32(value: f32) -> Result<Self> {
		ensure_runtime()?;
		install_error_handler();
		// emelex patch: the dedicated scalar constructor avoids initializing
		// the default Metal device merely to create host scalar metadata.
		unsafe { Self::from_raw(crate::engine::sys::mlx_array_new_float32(value)) }
	}

	/// Create a rank-0 (scalar) int32 array.
	pub fn scalar_i32(value: i32) -> Result<Self> {
		ensure_runtime()?;
		install_error_handler();
		unsafe { Self::from_raw(crate::engine::sys::mlx_array_new_int(value)) }
	}

	pub fn ndim(&self) -> usize {
		unsafe { crate::engine::sys::mlx_array_ndim(self.raw) }
	}

	pub fn shape(&self) -> Vec<i32> {
		unsafe {
			let ndim = crate::engine::sys::mlx_array_ndim(self.raw);
			// emelex patch: mlx-c may expose a null empty-vector data pointer
			// for rank-zero arrays; never pass it to from_raw_parts.
			if ndim == 0 {
				return Vec::new();
			}
			let ptr = crate::engine::sys::mlx_array_shape(self.raw);
			if ptr.is_null() {
				return Vec::new();
			}
			std::slice::from_raw_parts(ptr, ndim).to_vec()
		}
	}

	pub fn dim(&self, axis: i32) -> i32 {
		let shape = self.shape();
		let ndim = shape.len() as i32;
		let axis = if axis < 0 { axis + ndim } else { axis };
		shape[axis as usize]
	}

	pub fn size(&self) -> usize {
		self.shape().iter().map(|&d| d as usize).product()
	}

	pub fn dtype(&self) -> Dtype {
		unsafe { Dtype::from_raw(crate::engine::sys::mlx_array_dtype(self.raw)) }
	}

	/// Force evaluation of this array (MLX is lazy).
	pub fn eval(&self) -> Result<()> {
		unsafe { check(crate::engine::sys::mlx_array_eval(self.raw)) }
	}

	/// Extract a scalar float (evaluates if needed). Works on any float dtype.
	pub fn item_f32(&self) -> Result<f32> {
		let mut out: f32 = 0.0;
		unsafe {
			let arr = crate::engine::ops::astype(self, Dtype::Float32)?;
			arr.eval()?;
			check(crate::engine::sys::mlx_array_item_float32(
				&mut out, arr.raw,
			))?;
		}
		Ok(out)
	}

	/// Extract a scalar u32 (evaluates if needed).
	pub fn item_u32(&self) -> Result<u32> {
		let mut out: u32 = 0;
		unsafe {
			let arr = crate::engine::ops::astype(self, Dtype::UInt32)?;
			arr.eval()?;
			check(crate::engine::sys::mlx_array_item_uint32(&mut out, arr.raw))?;
		}
		Ok(out)
	}

	/// Copy the contents out as `f32` values (converting dtype if needed).
	pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
		// emelex patch: astype is a no-op for an already-f32 array, so a
		// sliced/strided lazy view would be read through its raw parent
		// buffer (wrong data). Force a contiguous materialization first.
		let as_f32 =
			crate::engine::ops::contiguous(&crate::engine::ops::astype(self, Dtype::Float32)?)?;
		as_f32.eval()?;
		unsafe {
			let ptr = crate::engine::sys::mlx_array_data_float32(as_f32.raw);
			if ptr.is_null() {
				return Err(crate::engine::error::Error::Mlx(
					"null data pointer reading array".into(),
				));
			}
			Ok(std::slice::from_raw_parts(ptr, as_f32.size()).to_vec())
		}
	}

	/// Copy the contents out as `u32` values (must already be uint32).
	pub fn to_vec_u32(&self) -> Result<Vec<u32>> {
		// emelex patch: see to_vec_f32 - materialize before raw reads.
		let arr =
			crate::engine::ops::contiguous(&crate::engine::ops::astype(self, Dtype::UInt32)?)?;
		arr.eval()?;
		unsafe {
			let ptr = crate::engine::sys::mlx_array_data_uint32(arr.raw);
			if ptr.is_null() {
				return Err(crate::engine::error::Error::Mlx(
					"null data pointer reading array".into(),
				));
			}
			Ok(std::slice::from_raw_parts(ptr, arr.size()).to_vec())
		}
	}
}

#[cfg(test)]
mod tests {
	use super::Array;
	use crate::engine::error::{Error, check, install_error_handler};

	const SCHEDULER_CHILD_MODE: &str = "EMELEX_CPU_SCHEDULER_EXCEPTION_CHILD";
	const IO_CHILD_MODE: &str = "EMELEX_NATIVE_IO_REGRESSION_CHILD";
	const ARRAY_CONSTRUCTION_CHILD_MODE: &str = "EMELEX_ARRAY_CONSTRUCTION_CHILD";

	#[test]
	fn scalar_shape_is_empty_without_dereferencing_native_null() {
		if std::env::var_os(ARRAY_CONSTRUCTION_CHILD_MODE).is_none() {
			let output = std::process::Command::new(std::env::current_exe().unwrap())
				.args([
					"--exact",
					"engine::array::tests::scalar_shape_is_empty_without_dereferencing_native_null",
					"--nocapture",
				])
				.env(ARRAY_CONSTRUCTION_CHILD_MODE, "1")
				.output()
				.unwrap();
			assert!(
				output.status.success(),
				"array construction child failed: status={:?}, stdout={}, stderr={}",
				output.status.code(),
				String::from_utf8_lossy(&output.stdout),
				String::from_utf8_lossy(&output.stderr)
			);
			return;
		}

		let float = Array::scalar_f32(1.0);
		if let Err(Error::Mlx(message)) = &float
			&& (message.contains("No Metal device") || message.contains("no Metal device"))
		{
			// Array metadata creation initializes MLX's default device.
			// Headless macOS cannot reach the nullable rank-zero shape path.
			return;
		}
		assert!(float.unwrap().shape().is_empty());
		assert!(Array::scalar_i32(1).unwrap().shape().is_empty());
		assert_eq!(Array::from_slice(&[1.0_f32], &[1]).unwrap().shape(), [1]);
	}

	#[test]
	fn from_slice_rejects_negative_dimensions() {
		let error = Array::from_slice(&[] as &[f32], &[-1]).unwrap_err();
		assert!(matches!(error, Error::Config(message) if message.contains("negative")));
	}

	#[test]
	fn from_slice_rejects_shape_product_overflow() {
		let error = Array::from_slice(&[] as &[f32], &[i32::MAX, i32::MAX, i32::MAX]).unwrap_err();
		assert!(matches!(error, Error::Config(message) if message.contains("overflows")));
	}

	#[test]
	fn from_slice_rejects_mismatched_data_length() {
		let error = Array::from_slice(&[1.0_f32], &[2]).unwrap_err();
		assert!(matches!(error, Error::Config(message) if message.contains("does not match")));
	}

	#[test]
	fn null_native_handle_consumes_recorded_mlx_error() {
		install_error_handler();
		let error = unsafe {
			let mut output = crate::engine::sys::mlx_array_new();
			let empty = crate::engine::sys::mlx_array_new();
			let status = crate::engine::sys::mlx_array_set(&mut output, empty);
			assert_ne!(status, 0);
			Array::from_raw(output).unwrap_err()
		};
		assert!(matches!(error, Error::Mlx(message) if message.contains("non-empty mlx_array")));
	}

	#[test]
	fn cpu_scheduler_exception_returns_through_c_abi_without_terminating() {
		if std::env::var_os(SCHEDULER_CHILD_MODE).is_some() {
			crate::runtime::initialize_default_if_needed().unwrap();
			unsafe extern "C" {
				fn mlx_emelex_test_cpu_scheduler_exception() -> i32;
				fn mlx_emelex_test_cpu_scheduler_nonstandard_exception() -> i32;
				fn mlx_emelex_test_cpu_enqueue_failure_recovery() -> i32;
				fn mlx_emelex_test_cpu_skipped_group_completion() -> i32;
			}
			let error = unsafe { check(mlx_emelex_test_cpu_scheduler_exception()) }.unwrap_err();
			assert!(
				matches!(error, Error::Mlx(message) if message.contains("scheduler exception probe"))
			);
			let error = unsafe { check(mlx_emelex_test_cpu_scheduler_nonstandard_exception()) }
				.unwrap_err();
			assert!(
				matches!(error, Error::Mlx(message) if message.contains("non-standard C++ exception"))
			);
			unsafe { check(mlx_emelex_test_cpu_enqueue_failure_recovery()) }.unwrap();
			unsafe { check(mlx_emelex_test_cpu_skipped_group_completion()) }.unwrap();
			return;
		}

		let output = std::process::Command::new(std::env::current_exe().unwrap())
			.args([
				"--exact",
				"engine::array::tests::cpu_scheduler_exception_returns_through_c_abi_without_terminating",
				"--nocapture",
			])
			.env(SCHEDULER_CHILD_MODE, "1")
			.output()
			.unwrap();
		assert!(
			output.status.success(),
			"scheduler exception child failed: status={:?}, stdout={}, stderr={}",
			output.status.code(),
			String::from_utf8_lossy(&output.stdout),
			String::from_utf8_lossy(&output.stderr)
		);
	}

	#[test]
	fn native_io_handles_partial_pread_and_descriptor_zero() {
		if std::env::var_os(IO_CHILD_MODE).is_some() {
			unsafe extern "C" {
				fn mlx_emelex_test_partial_pread_offsets() -> i32;
				fn mlx_emelex_test_fd_zero_io() -> i32;
			}
			unsafe {
				check(mlx_emelex_test_partial_pread_offsets()).unwrap();
				check(mlx_emelex_test_fd_zero_io()).unwrap();
			}
			return;
		}

		let output = std::process::Command::new(std::env::current_exe().unwrap())
			.args([
				"--exact",
				"engine::array::tests::native_io_handles_partial_pread_and_descriptor_zero",
				"--nocapture",
			])
			.env(IO_CHILD_MODE, "1")
			.output()
			.unwrap();
		assert!(
			output.status.success(),
			"native I/O child failed: status={:?}, stdout={}, stderr={}",
			output.status.code(),
			String::from_utf8_lossy(&output.stdout),
			String::from_utf8_lossy(&output.stderr)
		);
	}
}

/// Rust element types that map onto MLX dtypes for [`Array::from_slice`].
pub trait ArrayElement {
	const DTYPE: Dtype;
}

impl ArrayElement for f32 {
	const DTYPE: Dtype = Dtype::Float32;
}
impl ArrayElement for u32 {
	const DTYPE: Dtype = Dtype::UInt32;
}
impl ArrayElement for i32 {
	const DTYPE: Dtype = Dtype::Int32;
}
impl ArrayElement for u8 {
	const DTYPE: Dtype = Dtype::UInt8;
}
impl ArrayElement for bool {
	const DTYPE: Dtype = Dtype::Bool;
}

/// Synchronize the default stream, blocking until queued work completes.
pub fn synchronize() -> Result<()> {
	unsafe { check(crate::engine::sys::mlx_synchronize(stream()?)) }
}
