//! Benchmark-only diagnostics (feature `bench`).
//!
//! MLX memory instrumentation re-exposed from the private engine so the
//! MTP benchmark harness (`examples/mtp_bench.rs`) can attribute peak
//! and cache memory per benchmark cell. Not part of the supported
//! public API - the `bench` feature exists for the MTP benchmark harness
//! and may change without notice.

/// Peak bytes MLX has held in buffers since the last reset.
///
/// # Errors
///
/// Returns an MLX diagnostic failure.
pub fn peak_memory() -> Result<u64, crate::Error> {
	crate::engine::ops::peak_memory().map_err(crate::error::from_engine)
}

/// Reset the peak-memory watermark (call at the start of a cell run).
///
/// # Errors
///
/// Returns an MLX diagnostic failure.
pub fn reset_peak_memory() -> Result<(), crate::Error> {
	crate::engine::ops::reset_peak_memory().map_err(crate::error::from_engine)
}

/// Bytes MLX currently holds in live buffers.
///
/// # Errors
///
/// Returns an MLX diagnostic failure.
pub fn active_memory() -> Result<u64, crate::Error> {
	crate::engine::ops::active_memory().map_err(crate::error::from_engine)
}

/// Bytes MLX holds in freed-but-cached buffers awaiting reuse. The
/// 2 GiB freed-buffer cache carries kernel/buffer warmth across cells;
/// record it per cell so interleaved runs stay comparable.
///
/// # Errors
///
/// Returns an MLX diagnostic failure.
pub fn cache_memory() -> Result<u64, crate::Error> {
	crate::engine::ops::cache_memory().map_err(crate::error::from_engine)
}
