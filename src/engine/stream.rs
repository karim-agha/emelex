use std::{cell::RefCell, sync::Once};

use crate::engine::error::{Error, Result, check, install_error_handler};

struct StreamHolder(crate::engine::sys::mlx_stream);

impl Drop for StreamHolder {
	fn drop(&mut self) {
		// SAFETY: this holder owns the mlx-c stream handle.
		let _ = unsafe { crate::engine::sys::mlx_stream_free(self.0) };
	}
}

// MLX's GPU backend binds a stream's command-buffer machinery to whatever
// OS thread created it ("no Stream(gpu, 0) in current thread" if used from
// another thread) - so each thread gets its own lazily-created stream
// rather than one process-wide stream shared (and potentially used) across
// threads. This matters once test binaries run multiple `#[test]`s (each
// on its own thread) that touch array ops, not just the single-threaded
// CLI examples this originally targeted.
thread_local! {
	static DEFAULT_STREAM: RefCell<Option<StreamHolder>> = const { RefCell::new(None) };
	static CPU_STREAM: RefCell<Option<StreamHolder>> = const { RefCell::new(None) };
}

static INSTALL_ERROR_HANDLER_ONCE: Once = Once::new();

fn ensure_error_handler() {
	INSTALL_ERROR_HANDLER_ONCE.call_once(install_error_handler);
}

pub(crate) fn ensure_runtime() -> Result<()> {
	crate::runtime::initialize_default_if_needed()
		.map(|_| ())
		.map_err(|error| Error::Mlx(format!("runtime initialization: {error}")))
}

/// The current thread's default stream (GPU on Apple Silicon, CPU otherwise).
pub(crate) fn stream() -> Result<crate::engine::sys::mlx_stream> {
	ensure_runtime()?;
	ensure_error_handler();
	DEFAULT_STREAM.with(|cell| {
		let mut slot = cell.borrow_mut();
		if slot.is_none() {
			let created = unsafe {
				let mut device = crate::engine::sys::mlx_device_new();
				if let Err(error) = check(crate::engine::sys::mlx_get_default_device(&mut device)) {
					crate::engine::sys::mlx_device_free(device);
					return Err(error);
				}
				let mut stream = crate::engine::sys::mlx_stream_new();
				let stream_result = check(crate::engine::sys::mlx_get_default_stream(
					&mut stream,
					device,
				));
				crate::engine::sys::mlx_device_free(device);
				if let Err(error) = stream_result {
					let _ = crate::engine::sys::mlx_stream_free(stream);
					return Err(error);
				}
				StreamHolder(stream)
			};
			*slot = Some(created);
		}
		slot.as_ref()
			.map(|stream| stream.0)
			.ok_or_else(|| Error::Mlx("default stream was not initialized".to_string()))
	})
}

/// A stream pinned to the CPU device. `Load` (safetensors I/O) only has a
/// CPU eval kernel, so checkpoint loading must run here rather than on the
/// default (GPU) stream.
pub(crate) fn cpu_stream() -> Result<crate::engine::sys::mlx_stream> {
	ensure_runtime()?;
	ensure_error_handler();
	CPU_STREAM.with(|cell| {
		let mut slot = cell.borrow_mut();
		if slot.is_none() {
			let created = unsafe {
				let mut stream = crate::engine::sys::mlx_stream_new();
				if let Err(error) =
					check(crate::engine::sys::mlx_get_default_cpu_stream(&mut stream))
				{
					let _ = crate::engine::sys::mlx_stream_free(stream);
					return Err(error);
				}
				StreamHolder(stream)
			};
			*slot = Some(created);
		}
		slot.as_ref()
			.map(|stream| stream.0)
			.ok_or_else(|| Error::Mlx("CPU stream was not initialized".to_string()))
	})
}
