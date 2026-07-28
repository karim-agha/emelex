//! Self-contained video capability boundary.
//!
//! Emelex never resolves or executes ambient codec binaries. Encoded video
//! remains unavailable until a decoder backed by bundled code or macOS system
//! frameworks is part of the runtime.

use crate::engine::error::{Error, Result};

/// Maximum future frame sample count retained as a stable engine policy.
pub const MAX_FRAMES: usize = 8;

/// Reject encoded video before any ambient executable or external service can
/// be consulted.
pub fn extract_video_frames(_data: &[u8]) -> Result<Vec<Vec<u8>>> {
	Err(Error::Model(
		"self-contained video decoding is not available".to_string(),
	))
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn video_fails_closed_without_an_ambient_codec() {
		let error = extract_video_frames(b"encoded video").expect_err("video is unavailable");
		assert!(error.to_string().contains("self-contained video"));
	}
}
