//! Shared helpers used by every model architecture.

use crate::engine::{
	Cancellation,
	array::{Array, Dtype},
	error::{Error, Result},
	ops::{self, AttentionMask},
};

/// Decide the attention mask mode for one forward pass, mirroring mlx-lm's
/// `create_attention_mask`: no mask is needed when decoding a single new
/// token against an existing cache (the new token can attend to everything
/// already cached), otherwise use a causal mask.
pub fn attention_mask_for(seq_len: i32) -> AttentionMask {
	if seq_len == 1 {
		AttentionMask::None
	} else {
		AttentionMask::Causal
	}
}

/// Run an already-fused embedding stream through a decoder in bounded
/// prefill chunks. Intermediate last-logit rows are evaluated before each
/// cancellation checkpoint so the next chunk's graph is never constructed
/// after cancellation is observed. Disabled cancellation deliberately keeps
/// the historical one-pass call shape.
pub(super) fn forward_embeds_chunked(
	input_ids: &Array,
	embeds: &Array,
	chunk_tokens: usize,
	cancellation: Cancellation<'_>,
	mut forward: impl FnMut(&Array, Array) -> Result<Array>,
) -> Result<Array> {
	let input_shape = input_ids.shape();
	let embed_shape = embeds.shape();
	if input_shape.len() != 2
		|| embed_shape.len() != 3
		|| input_shape[0] != embed_shape[0]
		|| input_shape[1] != embed_shape[1]
		|| input_shape[0] <= 0
		|| input_shape[1] <= 0
		|| embed_shape[2] <= 0
	{
		return Err(Error::Model(format!(
			"media prefill expected input [B, L] and embeddings [B, L, H], got \
			 {input_shape:?} and {embed_shape:?}"
		)));
	}
	if chunk_tokens == 0 {
		return Err(Error::Model(
			"media prefill chunk size must be positive".to_string(),
		));
	}

	let sequence_tokens = usize::try_from(input_shape[1])
		.map_err(|_| Error::Model("media prefill sequence length exceeds usize".to_string()))?;
	let chunk_tokens = if cancellation.is_cooperative() {
		chunk_tokens
	} else {
		sequence_tokens
	};
	let mut output = None;
	for start in (0..sequence_tokens).step_by(chunk_tokens) {
		cancellation.checkpoint()?;
		let end = start.saturating_add(chunk_tokens).min(sequence_tokens);
		let start = i32::try_from(start)
			.map_err(|_| Error::Model("media prefill offset exceeds i32".to_string()))?;
		let end = i32::try_from(end)
			.map_err(|_| Error::Model("media prefill offset exceeds i32".to_string()))?;
		let ids = ops::slice(input_ids, &[0, start], &[input_shape[0], end])?;
		let chunk_embeds = ops::slice(
			embeds,
			&[0, start, 0],
			&[embed_shape[0], end, embed_shape[2]],
		)?;
		let logits = forward(&ids, chunk_embeds)?;
		if usize::try_from(end)
			.map_err(|_| Error::Model("media prefill offset exceeds usize".to_string()))?
			< sequence_tokens
		{
			eval_last_logits(&logits)?;
		}
		output = Some(logits);
		cancellation.checkpoint()?;
	}
	output.ok_or_else(|| Error::Model("cannot prefill an empty media prompt".to_string()))
}

fn eval_last_logits(logits: &Array) -> Result<()> {
	let shape = logits.shape();
	if shape.len() != 3 || shape[0] <= 0 || shape[1] <= 0 || shape[2] <= 0 {
		return Err(Error::Model(format!(
			"media prefill produced invalid logits shape {shape:?}"
		)));
	}
	let last = ops::slice(
		logits,
		&[0, shape[1] - 1, 0],
		&[shape[0], shape[1], shape[2]],
	)?;
	last.eval()
}

/// RoPE configuration for one attention layer.
#[derive(Debug, Clone, Copy)]
pub struct RopeConfig {
	pub dims: i32,
	pub base: f32,
	pub traditional: bool,
	/// Linear scaling factor (`rope_scaling: {"type": "linear", "factor": f}`).
	/// MLX's fused RoPE kernel applies `1/scale` to position indices, i.e.
	/// passing `scale = 1/factor` stretches the effective context.
	pub scale: f32,
}

impl RopeConfig {
	pub fn new(dims: i32, base: f32) -> Self {
		RopeConfig {
			dims,
			base,
			traditional: false,
			scale: 1.0,
		}
	}

	pub fn apply(&self, x: &Array, offset: i32) -> Result<Array> {
		ops::rope(
			x,
			self.dims,
			self.traditional,
			Some(self.base),
			self.scale,
			offset,
			None,
		)
	}
}

/// Per-dimension YaRN rotary frequencies (the *denominators* the fused
/// rope kernel expects via its `freqs` argument), mirroring mlx-lm's
/// `rope_utils.YarnRoPE`: high-frequency dims keep the original base
/// frequencies (extrapolation), low-frequency dims are stretched by
/// `factor` (interpolation), with a linear ramp between the correction
/// dims derived from `beta_fast`/`beta_slow`.
///
/// `dims` is the number of *rotated* dims (`head_dim *
/// partial_rotary_factor`); the returned vector has `dims / 2` entries.
pub fn yarn_freqs(
	dims: i32,
	base: f32,
	factor: f32,
	original_max_position_embeddings: i32,
	beta_fast: f32,
	beta_slow: f32,
) -> Vec<f32> {
	let dims_f = f64::from(dims);
	let base_f = f64::from(base);
	let orig = f64::from(original_max_position_embeddings);
	let correction_dim = |num_rotations: f64| -> f64 {
		dims_f * (orig / (num_rotations * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base_f.ln())
	};
	let low = correction_dim(f64::from(beta_fast)).floor().max(0.0);
	let mut high = correction_dim(f64::from(beta_slow))
		.ceil()
		.min(dims_f - 1.0);
	if low == high {
		high += 0.001; // prevent singularity
	}

	(0..dims / 2)
		.map(|i| {
			let freq_extra = base_f.powf(f64::from(2 * i) / dims_f);
			let freq_inter = f64::from(factor) * freq_extra;
			let ramp = ((f64::from(i) - low) / (high - low)).clamp(0.0, 1.0);
			let mask = 1.0 - ramp;
			let freq = (freq_inter * freq_extra) / (freq_inter * mask + freq_extra * (1.0 - mask));
			freq as f32
		})
		.collect()
}

/// YaRN-scaled RoPE for one attention layer: a precomputed per-dim
/// frequency array handed to the fused rope kernel, plus the attention
/// scaling factor ("mscale") applied to the *rotated* dims of the input.
///
/// The canonical implementation (Poolside's `modeling_laguna.py`, HF
/// yarn convention) multiplies cos/sin by `attention_factor`, which
/// scales only the rotated `dims` of q and k — the partial-rotary
/// pass-through dims stay unscaled, so this cannot be folded into the
/// attention softmax scale. Instead, when `mscale != 1`, the input is
/// multiplied by a `[head_dim]` vector (`mscale` on the rotated dims,
/// `1` elsewhere) before the rotation — equivalent, since the rotation
/// is linear.
#[derive(Clone)]
pub struct YarnRope {
	dims: i32,
	freqs: Array,
	/// `[head_dim]` broadcast multiplier; `None` when mscale is 1.
	mscale_vec: Option<Array>,
}

impl YarnRope {
	#[allow(
		clippy::too_many_arguments,
		reason = "constructor exposes the complete published YaRN parameterization"
	)]
	pub fn new(
		dims: i32,
		head_dim: i32,
		base: f32,
		factor: f32,
		original_max_position_embeddings: i32,
		beta_fast: f32,
		beta_slow: f32,
		attention_factor: f32,
	) -> Result<Self> {
		let freqs = yarn_freqs(
			dims,
			base,
			factor,
			original_max_position_embeddings,
			beta_fast,
			beta_slow,
		);
		let freqs = Array::from_slice(&freqs, &[dims / 2])?;
		let mscale_vec = if (attention_factor - 1.0).abs() > f32::EPSILON {
			let mut v = vec![1.0f32; head_dim as usize];
			v[..dims as usize].fill(attention_factor);
			Some(Array::from_slice(&v, &[head_dim])?)
		} else {
			None
		};
		Ok(YarnRope {
			dims,
			freqs,
			mscale_vec,
		})
	}

	pub fn apply(&self, x: &Array, offset: i32) -> Result<Array> {
		let x = match &self.mscale_vec {
			Some(v) => {
				let v = ops::astype(v, x.dtype())?;
				ops::multiply(x, &v)?
			}
			None => x.clone(),
		};
		ops::rope(&x, self.dims, false, None, 1.0, offset, Some(&self.freqs))
	}
}

/// Reshape `[B, L, H*D]` into `[B, H, L, D]` for attention.
pub fn split_heads(x: &Array, batch: i32, seq: i32, heads: i32) -> Result<Array> {
	let reshaped = ops::reshape(x, &[batch, seq, heads, -1])?;
	ops::transpose_axes(&reshaped, &[0, 2, 1, 3])
}

/// Reshape `[B, H, L, D]` back into `[B, L, H*D]`.
pub fn merge_heads(x: &Array, batch: i32, seq: i32) -> Result<Array> {
	let t = ops::transpose_axes(x, &[0, 2, 1, 3])?;
	ops::reshape(&t, &[batch, seq, -1])
}

/// Repeat KV heads along the head axis to match the query head count (GQA).
pub fn repeat_kv_heads(x: &Array, n_repeats: i32) -> Result<Array> {
	if n_repeats == 1 {
		return Ok(x.clone());
	}
	let shape = x.shape();
	let (b, h, l, d) = (shape[0], shape[1], shape[2], shape[3]);
	let expanded = ops::expand_dims(x, 2)?;
	let broadcasted = ops::broadcast_to(&expanded, &[b, h, n_repeats, l, d])?;
	ops::reshape(&broadcasted, &[b, h * n_repeats, l, d])
}

/// Splice `features` (one `[1, N_i, hidden]` tensor per media item, in
/// prompt order) into `h` at the positions where `input_ids` equals
/// `placeholder_token_id`, erroring on any placeholder/feature count
/// mismatch. Shared by every architecture's image/audio fusion path
/// (Gemma4's classic + unified towers, Qwen3.5-VL's vision tower, ...).
pub fn splice_media_features(
	h: &Array,
	input_ids: &Array,
	mut features: Vec<Array>,
	placeholder_token_id: i32,
	modality: &str,
) -> Result<Array> {
	let features = if features.len() == 1 {
		features.remove(0)
	} else {
		let refs: Vec<&Array> = features.iter().collect();
		ops::concatenate(&refs, 1)?
	};
	let features = ops::astype(&features, h.dtype())?;

	let placeholder = ops::astype(&Array::scalar_i32(placeholder_token_id)?, input_ids.dtype())?;
	let mask = ops::equal(input_ids, &placeholder)?;
	let mask_count_arr = ops::sum_axes(
		&ops::reshape(&ops::astype(&mask, Dtype::Int32)?, &[-1])?,
		&[0],
		false,
	)?;
	let mask_count = mask_count_arr.item_f32()? as i32;
	let feature_count = features.dim(1);
	if mask_count != feature_count {
		return Err(Error::Model(format!(
			"{modality} token count ({mask_count}) does not match {modality} \
			 feature count ({feature_count}); check that {modality} placeholder \
			 expansion produced the right number of tokens"
		)));
	}

	let mask_expanded = ops::broadcast_to(&ops::expand_dims(&mask, -1)?, &h.shape())?;
	masked_scatter(h, &mask_expanded, &features)
}

/// Replace positions where `mask` is true with values from `source` (in
/// mask-order), keeping `input`'s values everywhere else. Out-of-range
/// indices (positions before the first `true`, which would cumsum to -1)
/// are clamped into `[0, source_size)` instead of wrapping via modulo -
/// equivalent here since those lookups are always discarded by the
/// `where_cond` below (only `true` positions ever select `aligned`).
pub fn masked_scatter(input: &Array, mask: &Array, source: &Array) -> Result<Array> {
	let input_shape = input.shape();
	let mask_flat = ops::reshape(&ops::astype(mask, Dtype::Int32)?, &[-1])?;
	let input_flat = ops::reshape(input, &[-1])?;
	let source_flat = ops::reshape(source, &[-1])?;
	let source_size = source_flat.dim(0);

	let idx = ops::subtract(&ops::cumsum(&mask_flat, 0)?, &Array::scalar_i32(1)?)?;
	let idx = ops::maximum(&idx, &Array::scalar_i32(0)?)?;
	let idx = ops::minimum(&idx, &Array::scalar_i32((source_size - 1).max(0))?)?;
	let idx = ops::astype(&idx, Dtype::UInt32)?;

	let aligned = ops::take(&source_flat, &idx)?;
	let mask_bool = ops::astype(&mask_flat, Dtype::Bool)?;
	let result = ops::where_cond(&mask_bool, &aligned, &input_flat)?;
	ops::reshape(&result, &input_shape)
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Laguna M.1's full-attention rope: head_dim 128 fully rotated,
	/// base 500000, YaRN factor 32, original context 4096, β 64/1.
	#[test]
	fn yarn_freqs_laguna_m1_values() {
		let dims = 128;
		let base = 500000.0f32;
		let factor = 32.0f32;
		let freqs = yarn_freqs(dims, base, factor, 4096, 64.0, 1.0);
		assert_eq!(freqs.len(), 64);

		// Correction range for these parameters is low=11, high=32: dims
		// below the ramp keep the unscaled base frequencies
		// (extrapolation), dims past it are stretched by `factor`
		// (interpolation).
		assert!((freqs[0] - 1.0).abs() < 1e-6);
		let extra = |i: i32| (base as f64).powf(f64::from(2 * i) / 128.0);
		let interp_63 = (32.0 * extra(63)) as f32;
		assert!((freqs[63] - interp_63).abs() / interp_63 < 1e-4);

		// A mid-ramp dim mixes the two: at i=20, ramp = (20-11)/(32-11).
		let mask = 1.0 - (20.0 - 11.0) / (32.0 - 11.0);
		let e = extra(20);
		let expected_20 = ((32.0 * e * e) / (32.0 * e * mask + e * (1.0 - mask))) as f32;
		assert!((freqs[20] - expected_20).abs() / expected_20 < 1e-4);

		// Frequencies (denominators) must grow monotonically.
		for w in freqs.windows(2) {
			assert!(w[1] > w[0]);
		}
	}

	/// Laguna S-2.1's full-attention rope: partial rotary (64 of 128
	/// dims), YaRN factor 128, original context 8192, β 32/1.
	#[test]
	fn yarn_freqs_laguna_s21_values() {
		let base = 500000.0f32;
		let freqs = yarn_freqs(64, base, 128.0, 8192, 32.0, 1.0);
		assert_eq!(freqs.len(), 32);
		assert!((freqs[0] - 1.0).abs() < 1e-6);
		let interp_tail = (128.0 * (base as f64).powf(62.0 / 64.0)) as f32;
		assert!((freqs[31] - interp_tail).abs() / interp_tail < 1e-4);
		for w in freqs.windows(2) {
			assert!(w[1] > w[0]);
		}
	}
}
