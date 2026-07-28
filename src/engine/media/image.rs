//! Image preprocessing matching Gemma4's vision processor (HF
//! `preprocessor_config.json`: `do_rescale=true` (÷255), `do_normalize=false`
//! (mean=0, std=1) - i.e. resize + rescale to `[0, 1]` only; the model's own
//! patch embedder applies the `2*(x-0.5)` normalization).
//!
//! Resize policy: a "smart_resize" style used by several vision processors:
//! round each side to the nearest multiple of `patch_size *
//! pooling_kernel_size`, then only rescale (preserving aspect ratio) if the
//! rounded pixel count falls outside `[min_tokens, max_tokens] *
//! (patch_size * pooling_kernel_size)^2` - i.e. most naturally-sized photos
//! pass through close to their native resolution instead of always being
//! stretched to fill the token budget (an "always fill the budget" policy
//! instead causes small photos to be upscaled ~1.5x, blurring detail and
//! degrading vision quality).

use std::io::Cursor;

use image::{GenericImageView, ImageReader, Limits, RgbImage, imageops::FilterType};

use crate::engine::{
	Cancellation,
	array::Array,
	error::{Error, Result},
	media::MAX_ENCODED_MEDIA_BYTES,
};

/// A resized, patch-grid-aligned image ready for a vision tower.
#[derive(Debug, Clone)]
pub struct ProcessedImage {
	/// `[1, 3, H, W]` float32, values in `[0, 1]` (channel-first, no
	/// mean/std normalization - matches Gemma4's processor config).
	pub pixel_values: Array,
	/// Patch grid height (`H / patch_size`).
	pub patch_h: i32,
	/// Patch grid width (`W / patch_size`).
	pub patch_w: i32,
	/// Soft tokens this image expands to after pooling:
	/// `patch_h * patch_w / pooling_kernel_size^2`.
	pub num_soft_tokens: i32,
}

impl ProcessedImage {
	pub(crate) fn retained_tensor_bytes(&self) -> Result<usize> {
		self.pixel_values
			.size()
			.checked_mul(std::mem::size_of::<f32>())
			.ok_or_else(|| Error::Model("processed image tensor byte count overflow".to_string()))
	}
}

/// Gemma4 vision's hardcoded soft-token budget - not read from
/// `config.json`, so we hardcode it here too.
pub const MIN_SOFT_TOKENS: i32 = 40;
const MAX_IMAGE_DIMENSION: u32 = 8192;
const MAX_DECODED_IMAGE_PIXELS: u64 = 32 << 20;
const MAX_IMAGE_DECODE_ALLOC_BYTES: u64 = 256 << 20;
const MAX_PROCESSED_IMAGE_PIXELS: u64 = 16 << 20;
const MAX_PATCH_SIZE: i32 = 256;
const MAX_POOLING_KERNEL_SIZE: i32 = 16;
const MAX_SOFT_TOKENS: i32 = 16_384;
const MAX_ASPECT_RATIO: u32 = 200;

/// Decode `data` (JPEG/PNG/...) and resize it per Gemma4's smart-resize
/// policy: [`MIN_SOFT_TOKENS`]..=`max_soft_tokens` worth of `patch_size`x
/// `patch_size` patches (grouped into `pooling_kernel_size`x
/// `pooling_kernel_size` pooling blocks).
pub fn preprocess_image_bytes(
	data: &[u8],
	patch_size: i32,
	max_soft_tokens: i32,
	pooling_kernel_size: i32,
) -> Result<ProcessedImage> {
	preprocess_image_bytes_cancellable(
		data,
		patch_size,
		max_soft_tokens,
		pooling_kernel_size,
		Cancellation::disabled(),
	)
}

pub(crate) fn preprocess_image_bytes_cancellable(
	data: &[u8],
	patch_size: i32,
	max_soft_tokens: i32,
	pooling_kernel_size: i32,
	cancellation: Cancellation<'_>,
) -> Result<ProcessedImage> {
	cancellation.checkpoint()?;
	// emelex patch: reject oversized encodings before decoder sniffing and
	// enforce decoder-side dimensions/allocation limits against image bombs.
	validate_encoded_image_len(data.len())?;
	validate_model_geometry(
		patch_size,
		MIN_SOFT_TOKENS,
		max_soft_tokens,
		pooling_kernel_size,
	)?;
	let mut reader = ImageReader::new(Cursor::new(data))
		.with_guessed_format()
		.map_err(|e| Error::Model(format!("failed to inspect image: {e}")))?;
	let mut limits = Limits::default();
	limits.max_image_width = Some(MAX_IMAGE_DIMENSION);
	limits.max_image_height = Some(MAX_IMAGE_DIMENSION);
	limits.max_alloc = Some(MAX_IMAGE_DECODE_ALLOC_BYTES);
	reader.limits(limits);
	let img = reader
		.decode()
		.map_err(|e| Error::Model(format!("failed to decode image: {e}")))?;
	cancellation.checkpoint()?;
	let (orig_w, orig_h) = img.dimensions();
	validate_source_dimensions(orig_w, orig_h)?;
	let (target_w, target_h) = compute_target_size(
		orig_h,
		orig_w,
		patch_size,
		MIN_SOFT_TOKENS,
		max_soft_tokens,
		pooling_kernel_size,
	)?;

	let resized = if target_h == orig_h && target_w == orig_w {
		img.to_rgb8()
	} else {
		// Bilinear resize (not a higher-order filter), matching Gemma4's
		// expected image resize algorithm.
		img.resize_exact(target_w, target_h, FilterType::Triangle)
			.to_rgb8()
	};
	cancellation.checkpoint()?;

	let pixel_values = rgb_to_chw_array(&resized, cancellation)?;
	let patch_h = i32::try_from(target_h)
		.map_err(|_| Error::Model("processed image height exceeds i32".to_string()))?
		/ patch_size;
	let patch_w = i32::try_from(target_w)
		.map_err(|_| Error::Model("processed image width exceeds i32".to_string()))?
		/ patch_size;
	let patch_count = patch_h
		.checked_mul(patch_w)
		.ok_or_else(|| Error::Model("processed image patch count overflow".to_string()))?;
	let pool_area = pooling_kernel_size
		.checked_mul(pooling_kernel_size)
		.ok_or_else(|| Error::Model("image pooling area overflow".to_string()))?;
	let num_soft_tokens = patch_count / pool_area;
	if num_soft_tokens < MIN_SOFT_TOKENS || num_soft_tokens > max_soft_tokens {
		return Err(Error::Model(format!(
			"processed image needs {num_soft_tokens} soft tokens, outside \
			 {MIN_SOFT_TOKENS}..={max_soft_tokens}"
		)));
	}

	Ok(ProcessedImage {
		pixel_values,
		patch_h,
		patch_w,
		num_soft_tokens,
	})
}

/// "smart_resize" style size calculation: round both
/// sides up to the nearest multiple of `align_size` (`patch_size *
/// pooling_kernel_size`) first: if that lands within `[min_pixels,
/// max_pixels]`, use it as-is (near-native resolution, no big up/downscale);
/// otherwise scale by `sqrt(area_ratio)` and re-align (floor when shrinking
/// to fit under `max_pixels`, ceil when growing to clear `min_pixels`).
fn compute_target_size(
	height: u32,
	width: u32,
	patch_size: i32,
	min_soft_tokens: i32,
	max_soft_tokens: i32,
	pooling_kernel_size: i32,
) -> Result<(u32, u32)> {
	// emelex patch: model-provided geometry is untrusted. Validate before
	// signed multiplication or float conversion, then re-check aligned output.
	validate_geometry(
		height,
		width,
		patch_size,
		min_soft_tokens,
		max_soft_tokens,
		pooling_kernel_size,
	)?;
	let align_i32 = patch_size
		.checked_mul(pooling_kernel_size)
		.ok_or_else(|| Error::Model("image alignment overflow".to_string()))?;
	let align_u64 = u64::try_from(align_i32)
		.map_err(|_| Error::Model("image alignment must be positive".to_string()))?;
	let patch_area_u64 = align_u64
		.checked_mul(align_u64)
		.ok_or_else(|| Error::Model("image aligned patch area overflow".to_string()))?;
	let configured_max_pixels = u64::try_from(max_soft_tokens)
		.map_err(|_| Error::Model("image soft-token budget must be positive".to_string()))?
		.checked_mul(patch_area_u64)
		.ok_or_else(|| Error::Model("image pixel budget overflow".to_string()))?;
	let min_pixels_u64 = u64::try_from(min_soft_tokens)
		.map_err(|_| Error::Model("image minimum soft-token budget must be positive".to_string()))?
		.checked_mul(patch_area_u64)
		.ok_or_else(|| Error::Model("image minimum pixel budget overflow".to_string()))?;
	if min_pixels_u64 > MAX_PROCESSED_IMAGE_PIXELS {
		return Err(Error::Model(format!(
			"image minimum geometry needs {min_pixels_u64} processed pixels, exceeding {MAX_PROCESSED_IMAGE_PIXELS}"
		)));
	}
	let max_pixels_u64 = configured_max_pixels.min(MAX_PROCESSED_IMAGE_PIXELS);

	let align = f64::from(align_i32);
	let patch_area = align * align;
	let min_pixels = min_soft_tokens as f64 * patch_area;
	let max_pixels = max_pixels_u64 as f64;

	let (h, w) = (height as f64, width as f64);
	let round_by = |x: f64| -> f64 { (x / align).round() * align };
	let floor_by = |x: f64| -> f64 { (x / align).floor() * align };
	let ceil_by = |x: f64| -> f64 { (x / align).ceil() * align };

	let mut h_bar = round_by(h).max(align);
	let mut w_bar = round_by(w).max(align);

	if h_bar * w_bar > max_pixels {
		let beta = (h * w / max_pixels).sqrt();
		h_bar = floor_by(h / beta).max(align);
		w_bar = floor_by(w / beta).max(align);
	} else if h_bar * w_bar < min_pixels {
		let beta = (min_pixels / (h * w)).sqrt();
		h_bar = ceil_by(h * beta).max(align);
		w_bar = ceil_by(w * beta).max(align);
	}

	if !h_bar.is_finite()
		|| !w_bar.is_finite()
		|| h_bar < align
		|| w_bar < align
		|| h_bar > f64::from(u32::MAX)
		|| w_bar > f64::from(u32::MAX)
	{
		return Err(Error::Model(
			"image resize produced invalid dimensions".to_string(),
		));
	}
	let target_h = h_bar as u32;
	let target_w = w_bar as u32;
	let target_pixels = u64::from(target_h)
		.checked_mul(u64::from(target_w))
		.ok_or_else(|| Error::Model("processed image area overflow".to_string()))?;
	if target_pixels == 0
		|| target_pixels < min_pixels_u64
		|| target_pixels > max_pixels_u64
		|| target_pixels > MAX_PROCESSED_IMAGE_PIXELS
		|| u64::from(target_h) % align_u64 != 0
		|| u64::from(target_w) % align_u64 != 0
	{
		return Err(Error::Model(format!(
			"image resize produced unsafe {target_w}x{target_h} geometry"
		)));
	}

	Ok((target_w, target_h))
}

/// Rescale an RGB image to `[0, 1]` and lay it out channel-first as
/// `[1, 3, H, W]` float32 (no mean/std normalization).
fn rgb_to_chw_array(rgb: &RgbImage, cancellation: Cancellation<'_>) -> Result<Array> {
	let width = usize::try_from(rgb.width())
		.map_err(|_| Error::Model("processed image width exceeds address space".to_string()))?;
	let height = usize::try_from(rgb.height())
		.map_err(|_| Error::Model("processed image height exceeds address space".to_string()))?;
	let pixels = width
		.checked_mul(height)
		.ok_or_else(|| Error::Model("processed image area overflow".to_string()))?;
	if u64::try_from(pixels).unwrap_or(u64::MAX) > MAX_PROCESSED_IMAGE_PIXELS {
		return Err(Error::Model(format!(
			"processed image exceeds {MAX_PROCESSED_IMAGE_PIXELS} pixel limit"
		)));
	}
	let values = pixels
		.checked_mul(3)
		.ok_or_else(|| Error::Model("processed image tensor size overflow".to_string()))?;
	let mut chw = vec![0f32; values];
	for y in 0..height {
		if y % 64 == 0 {
			cancellation.checkpoint()?;
		}
		for x in 0..width {
			let pixel = rgb.get_pixel(x as u32, y as u32);
			for c in 0..3usize {
				chw[c * pixels + y * width + x] = pixel[c] as f32 / 255.0;
			}
		}
	}
	let height = i32::try_from(height)
		.map_err(|_| Error::Model("processed image height exceeds i32".to_string()))?;
	let width = i32::try_from(width)
		.map_err(|_| Error::Model("processed image width exceeds i32".to_string()))?;
	cancellation.checkpoint()?;
	Array::from_slice(&chw, &[1, 3, height, width])
}

fn validate_encoded_image_len(length: usize) -> Result<()> {
	if length > MAX_ENCODED_MEDIA_BYTES {
		return Err(Error::Model(format!(
			"encoded image exceeds {MAX_ENCODED_MEDIA_BYTES} byte limit"
		)));
	}
	Ok(())
}

fn validate_source_dimensions(width: u32, height: u32) -> Result<()> {
	if width == 0 || height == 0 {
		return Err(Error::Model(
			"decoded image dimensions must be positive".to_string(),
		));
	}
	if width > MAX_IMAGE_DIMENSION || height > MAX_IMAGE_DIMENSION {
		return Err(Error::Model(format!(
			"decoded image exceeds {MAX_IMAGE_DIMENSION} pixel dimension limit"
		)));
	}
	let pixels = u64::from(width)
		.checked_mul(u64::from(height))
		.ok_or_else(|| Error::Model("decoded image area overflow".to_string()))?;
	if pixels > MAX_DECODED_IMAGE_PIXELS {
		return Err(Error::Model(format!(
			"decoded image exceeds {MAX_DECODED_IMAGE_PIXELS} pixel limit"
		)));
	}
	let short = width.min(height);
	let long = width.max(height);
	if long > short.saturating_mul(MAX_ASPECT_RATIO) {
		return Err(Error::Model(format!(
			"decoded image aspect ratio exceeds {MAX_ASPECT_RATIO}:1"
		)));
	}
	Ok(())
}

fn validate_geometry(
	height: u32,
	width: u32,
	patch_size: i32,
	min_soft_tokens: i32,
	max_soft_tokens: i32,
	pooling_kernel_size: i32,
) -> Result<()> {
	validate_source_dimensions(width, height)?;
	validate_model_geometry(
		patch_size,
		min_soft_tokens,
		max_soft_tokens,
		pooling_kernel_size,
	)
}

fn validate_model_geometry(
	patch_size: i32,
	min_soft_tokens: i32,
	max_soft_tokens: i32,
	pooling_kernel_size: i32,
) -> Result<()> {
	if !(1..=MAX_PATCH_SIZE).contains(&patch_size) {
		return Err(Error::Model(format!(
			"image patch size must be within 1..={MAX_PATCH_SIZE}"
		)));
	}
	if !(1..=MAX_POOLING_KERNEL_SIZE).contains(&pooling_kernel_size) {
		return Err(Error::Model(format!(
			"image pooling kernel must be within 1..={MAX_POOLING_KERNEL_SIZE}"
		)));
	}
	if min_soft_tokens <= 0
		|| max_soft_tokens < min_soft_tokens
		|| max_soft_tokens > MAX_SOFT_TOKENS
	{
		return Err(Error::Model(format!(
			"image soft-token budget must satisfy 1 <= min <= max <= {MAX_SOFT_TOKENS}"
		)));
	}
	Ok(())
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn compute_target_size_is_a_multiple_of_side_mult() {
		// patch_size=16, pooling=3 -> align=48.
		let (w, h) = compute_target_size(1080, 1920, 16, 40, 280, 3).unwrap();
		assert_eq!(w % 48, 0);
		assert_eq!(h % 48, 0);
		assert!(w > 0 && h > 0);
	}

	#[test]
	fn num_soft_tokens_never_exceeds_budget() {
		let (w, h) = compute_target_size(4000, 3000, 16, 40, 280, 3).unwrap();
		let patch_h = h as i32 / 16;
		let patch_w = w as i32 / 16;
		let soft = (patch_h * patch_w) / 9;
		assert!(soft <= 280, "soft={soft}");
	}

	#[test]
	fn a_naturally_sized_photo_stays_near_native_resolution() {
		// 640x426 (samples/image1.jpg) rounds to within the [40, 280]
		// soft-token budget already, so it should NOT be upscaled to fill
		// the budget (a prior, incorrect port of a different reference's
		// "always fill the budget" policy stretched this to 960x624).
		let (w, h) = compute_target_size(426, 640, 16, 40, 280, 3).unwrap();
		assert_eq!((w, h), (624, 432));
	}

	#[test]
	fn invalid_model_geometry_is_rejected_before_math() {
		for parameters in [(0, 280, 3), (-1, 280, 3), (16, 39, 3), (16, 280, 0)] {
			assert!(
				compute_target_size(
					100,
					100,
					parameters.0,
					MIN_SOFT_TOKENS,
					parameters.1,
					parameters.2,
				)
				.is_err()
			);
		}
	}

	#[test]
	fn excessive_encoded_size_is_rejected_without_allocating_input() {
		assert!(validate_encoded_image_len(MAX_ENCODED_MEDIA_BYTES + 1).is_err());
	}

	#[test]
	fn oversized_or_extreme_source_geometry_is_rejected() {
		assert!(validate_source_dimensions(MAX_IMAGE_DIMENSION + 1, 1).is_err());
		assert!(validate_source_dimensions(1000, 1).is_err());
		assert!(validate_source_dimensions(8000, 8000).is_err());
	}

	#[test]
	fn processed_pixel_budget_rejects_hostile_model_geometry() {
		assert!(compute_target_size(100, 100, 256, 40, 40, 16).is_err());
	}

	#[test]
	fn invalid_model_geometry_is_rejected_before_image_decode() {
		let error = preprocess_image_bytes(b"not an image", 0, 280, 3).unwrap_err();
		assert!(error.to_string().contains("patch size"));
	}

	#[test]
	fn resize_fails_when_alignment_cannot_reach_minimum_token_budget() {
		assert!(compute_target_size(2, 24, 1, 40, 40, 1).is_err());
	}
}
