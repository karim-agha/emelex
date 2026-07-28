use std::{
	cell::RefCell,
	ffi::{CStr, c_char, c_void},
};

/// Errors surfaced by the MLX runtime or by model loading.
#[derive(Debug, thiserror::Error)]
pub enum Error {
	/// An error reported by MLX itself (shape mismatch, bad dtype, ...).
	#[error("mlx: {0}")]
	Mlx(String),
	#[error("io: {0}")]
	Io(#[from] std::io::Error),
	#[error("config: {0}")]
	Config(String),
	#[error("model: {0}")]
	Model(String),
	#[error("tokenizer: {0}")]
	Tokenizer(String),
	#[error("template: {0}")]
	Template(String),
	#[error(
		"context window exceeded: prompt {prompt_tokens} + output {max_output_tokens} > {limit}"
	)]
	ContextExceeded {
		prompt_tokens: usize,
		max_output_tokens: usize,
		limit: usize,
	},
	#[error("capability {capability} is unavailable: {reason}")]
	CapabilityUnavailable {
		capability: &'static str,
		reason: String,
	},
	#[error("generation cancelled")]
	Cancelled,
}

pub type Result<T> = std::result::Result<T, Error>;

thread_local! {
		static LAST_MLX_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
}

unsafe extern "C" fn record_error(msg: *const c_char, _data: *mut c_void) {
	let text = if msg.is_null() {
		String::from("unknown MLX error")
	} else {
		// SAFETY: mlx-c invokes this handler with a valid NUL-terminated
		// message pointer when the pointer is non-null.
		unsafe { CStr::from_ptr(msg) }
			.to_string_lossy()
			.into_owned()
	};
	LAST_MLX_ERROR.with(|slot| *slot.borrow_mut() = Some(text));
}

/// Install the process-wide MLX error handler that records errors instead of
/// aborting the process (the mlx-c default handler calls `exit(-1)`).
///
/// Called automatically the first time any MLX object is created; safe to
/// call repeatedly.
pub fn install_error_handler() {
	use std::sync::Once;
	static ONCE: Once = Once::new();
	ONCE.call_once(|| unsafe {
		crate::engine::sys::mlx_set_error_handler(Some(record_error), std::ptr::null_mut(), None);
	});
}

/// Convert an mlx-c status code into a `Result`, attaching the recorded
/// error message when the call failed.
pub(crate) fn check(status: i32) -> Result<()> {
	if status == 0 {
		return Ok(());
	}
	// emelex patch: mlx-c errors are thread-local and single-consumption.
	Err(take_last_error("MLX call failed without a message"))
}

/// Consume the current thread's native error after an MLX function reports
/// failure without a status code, such as a null returned handle.
pub(crate) fn take_last_error(fallback: &str) -> Error {
	let message = LAST_MLX_ERROR
		.with(|slot| slot.borrow_mut().take())
		.unwrap_or_else(|| fallback.to_string());
	Error::Mlx(message)
}
