//! Per-layer caches used during incremental generation.
//!
//! Two kinds exist: the standard quadratic-attention `KvCache` (growing
//! keys/values) and `GatedDeltaCache` for linear-attention layers
//! (Qwen3.5/3.6 `GatedDeltaNet`), which instead carries a fixed-size causal
//! conv window plus a recurrent `[B, H, Dv, Dk]` state matrix. `LayerCache`
//! lets `Model::forward` treat both uniformly across architectures.

use crate::engine::{
	array::Array,
	error::{Error, Result},
	ops,
};

/// Growing key/value cache for one attention layer.
///
/// emelex patch (not upstream): amortized growth in
/// [`KV_GROWTH_STEP`]-sized chunks with per-step `slice_update` writes,
/// returning a sliced view of the filled prefix - the classic KV cache
/// (mlx-lm's `KVCache`). Upstream concatenated the *entire* K/V tensors
/// on every decode step (O(n^2) memory traffic over a generation). MLX
/// arrays are copy-on-write values: `slice_update` donates the buffer
/// when nothing else references it (the common decode path) and
/// degrades to one full copy when a prompt-cache snapshot still holds
/// it - correct either way, amortized-fast in the common path.
#[derive(Default, Clone)]
pub struct KvCache {
	keys: Option<Array>,
	values: Option<Array>,
	/// Absolute token count seen (the rope/position offset).
	offset: i32,
	/// Sliding-window retention: keep roughly this many trailing
	/// positions (plus growth slack); `None` retains everything.
	///
	/// emelex patch (not upstream): without this, sliding-window layers
	/// (Laguna S-2.1 has 36 of 48) retain the *entire* history their
	/// mask can never attend to, growing KV memory ~4x and walking a
	/// 61.6 GiB checkpoint into the Metal wired limit over a long chat.
	/// The buffer may retain up to `window + KV_GROWTH_STEP` tokens so
	/// trims amortize; the sliding-window mask handles exactness.
	window: Option<i32>,
	/// Absolute position of buffer index 0 (advances once trimmed).
	start: i32,
}

/// Buffer growth quantum, matching mlx-lm's default step.
const KV_GROWTH_STEP: i32 = 256;

impl KvCache {
	pub fn new() -> Self {
		Self::default()
	}

	/// A cache that retains only a trailing window of positions.
	pub fn windowed(window: i32) -> Self {
		Self {
			window: Some(window.max(1)),
			..Self::default()
		}
	}

	pub fn offset(&self) -> i32 {
		self.offset
	}

	/// Absolute position of the first retained key (0 until a windowed
	/// cache trims).
	pub fn start(&self) -> i32 {
		self.start
	}

	/// Current buffer capacity in positions (diagnostics).
	pub fn buffer_capacity(&self) -> i32 {
		self.keys.as_ref().map_or(0, |k| k.dim(-2))
	}

	/// Append `keys`/`values` (shape `[B, H, L, D]`) and return the
	/// retained cache contents (everything seen, unless windowed).
	pub fn update_and_fetch(&mut self, keys: Array, values: Array) -> Result<(Array, Array)> {
		let shape = keys.shape();
		let (b, h, added, d) = (shape[0], shape[1], shape[2], shape[3]);
		self.trim_to_window(b, h, d, added)?;
		let prev = self.offset - self.start;
		let needed = prev + added;

		let capacity = self.keys.as_ref().map_or(0, |k| k.dim(-2));
		if needed > capacity {
			let new_capacity = (needed + KV_GROWTH_STEP - 1) / KV_GROWTH_STEP * KV_GROWTH_STEP;
			let grown_k = ops::zeros(&[b, h, new_capacity, d], keys.dtype())?;
			let grown_v = ops::zeros(&[b, h, new_capacity, d], values.dtype())?;
			let (grown_k, grown_v) = match (self.keys.take(), self.values.take()) {
				(Some(old_k), Some(old_v)) if prev > 0 => (
					ops::slice_update(
						&grown_k,
						&ops::slice(&old_k, &[0, 0, 0, 0], &[b, h, prev, d])?,
						&[0, 0, 0, 0],
						&[b, h, prev, d],
					)?,
					ops::slice_update(
						&grown_v,
						&ops::slice(&old_v, &[0, 0, 0, 0], &[b, h, prev, d])?,
						&[0, 0, 0, 0],
						&[b, h, prev, d],
					)?,
				),
				_ => (grown_k, grown_v),
			};
			self.keys = Some(grown_k);
			self.values = Some(grown_v);
		}

		let buf_k = self
			.keys
			.take()
			.ok_or_else(|| Error::Model("KV cache key allocation disappeared".to_string()))?;
		let buf_v = self
			.values
			.take()
			.ok_or_else(|| Error::Model("KV cache value allocation disappeared".to_string()))?;
		let buf_k = ops::slice_update(&buf_k, &keys, &[0, 0, prev, 0], &[b, h, needed, d])?;
		let buf_v = ops::slice_update(&buf_v, &values, &[0, 0, prev, 0], &[b, h, needed, d])?;
		// `needed` is buffer-relative; the absolute offset re-adds the
		// trimmed prefix.
		self.offset = self.start + needed;
		let full_k = ops::slice(&buf_k, &[0, 0, 0, 0], &[b, h, needed, d])?;
		let full_v = ops::slice(&buf_v, &[0, 0, 0, 0], &[b, h, needed, d])?;
		self.keys = Some(buf_k);
		self.values = Some(buf_v);
		// Trim again after the append: a single large prefill chunk
		// (whole prompts arrive in one call) must not linger untrimmed —
		// prompt-cache snapshots taken at the turn boundary would pin it.
		// The returned views above are unaffected; this only bounds the
		// *stored* state.
		self.trim_to_window(b, h, d, 0)?;
		Ok((full_k, full_v))
	}

	/// Windowed caches: once the retained prefix would exceed the window
	/// by more than the growth slack after `added` new tokens, drop the
	/// leading positions the sliding mask can never attend to again,
	/// keeping the last `window` (the mask masks any slack precisely).
	fn trim_to_window(&mut self, b: i32, h: i32, d: i32, added: i32) -> Result<()> {
		let Some(window) = self.window else {
			return Ok(());
		};
		let retained = self.offset - self.start;
		if retained + added <= window + KV_GROWTH_STEP {
			return Ok(());
		}
		let keep = (window - added).max(0).min(retained);
		let drop = retained - keep;
		if drop <= 0 {
			return Ok(());
		}
		if keep > 0
			&& let (Some(old_k), Some(old_v)) = (self.keys.take(), self.values.take())
		{
			// Copy the tail into fresh buffers rather than storing slice
			// *views*: a view pins the entire old allocation (a 31k-token
			// prefill buffer would survive the trim), and prompt-cache
			// snapshots would pin it across turns.
			let capacity = (keep + KV_GROWTH_STEP - 1) / KV_GROWTH_STEP * KV_GROWTH_STEP;
			let fresh_k = ops::zeros(&[b, h, capacity, d], old_k.dtype())?;
			let fresh_v = ops::zeros(&[b, h, capacity, d], old_v.dtype())?;
			let tail_k = ops::slice(&old_k, &[0, 0, drop, 0], &[b, h, retained, d])?;
			let tail_v = ops::slice(&old_v, &[0, 0, drop, 0], &[b, h, retained, d])?;
			self.keys = Some(ops::slice_update(
				&fresh_k,
				&tail_k,
				&[0, 0, 0, 0],
				&[b, h, keep, d],
			)?);
			self.values = Some(ops::slice_update(
				&fresh_v,
				&tail_v,
				&[0, 0, 0, 0],
				&[b, h, keep, d],
			)?);
		} else {
			self.keys = None;
			self.values = None;
		}
		self.start += drop;
		Ok(())
	}

	/// Roll the cache back to absolute position `offset` (dropping the
	/// suffix fed after that point, e.g. a rejected speculative step).
	///
	/// emelex patch (not upstream): rollback primitive. Only the offset
	/// moves — buffer contents are untouched. Stale rows past `offset`
	/// are overwritten by the next append (`slice_update` at buffer
	/// index `offset - start`), and fetched views slice only
	/// `0..needed`, so they can never leak. Positions below `start`
	/// were physically discarded by a window trim and cannot be
	/// restored, so rolling back past them is an error, as is
	/// "rolling back" forward past the current end.
	pub fn truncate_to(&mut self, offset: i32) -> Result<()> {
		if offset < self.start {
			return Err(Error::Model(format!(
				"cannot truncate KV cache to position {offset}: positions before {} \
				 were discarded by a window trim",
				self.start
			)));
		}
		if offset > self.offset {
			return Err(Error::Model(format!(
				"cannot truncate KV cache to position {offset}: only {} positions \
				 have been fed",
				self.offset
			)));
		}
		self.offset = offset;
		Ok(())
	}

	/// Whether appending `added` positions would fire
	/// [`trim_to_window`](Self::trim_to_window), irreversibly discarding
	/// leading positions (observable as `start` advancing).
	///
	/// emelex patch (not upstream): exact mirror of `trim_to_window`'s
	/// fire condition, so callers planning a rollback can detect — before
	/// feeding — that the append would destroy history `truncate_to`
	/// might later need.
	pub fn would_trim(&self, added: i32) -> bool {
		self.window
			.is_some_and(|window| (self.offset - self.start) + added > window + KV_GROWTH_STEP)
	}

	/// True iff the cache has never been fed (no appends, no trims, no
	/// poisoned forward).
	///
	/// emelex patch (not upstream): rollback support (see
	/// [`LayerCache::is_pristine`]).
	fn is_pristine(&self) -> bool {
		self.offset == self.start && self.keys.is_none()
	}
}

/// Recurrent state for one `GatedDeltaNet` (linear-attention) layer.
#[derive(Default, Clone)]
pub struct GatedDeltaCache {
	/// Trailing `kernel_size - 1` conv input frames, `[B, K-1, conv_dim]`.
	pub conv_state: Option<Array>,
	/// Recurrent delta-rule state, `[B, Hv, Dv, Dk]` (kept in f32).
	pub recur_state: Option<Array>,
}

impl GatedDeltaCache {
	pub fn new() -> Self {
		Self::default()
	}
}

/// Per-layer state for a DharaAR decoder block: one attention `KvCache`
/// plus up to four causal-conv "Canon layer" states (positions A/B(q,k,v)/C/D
/// — see `models::dhara`), each a trailing `[B, kernel-1, dim]` window of
/// prior inputs, `None` until the first token has been fed through.
#[derive(Default, Clone)]
pub struct DharaCache {
	pub attn: KvCache,
	pub canon_a: Option<Array>,
	pub canon_b_q: Option<Array>,
	pub canon_b_k: Option<Array>,
	pub canon_b_v: Option<Array>,
	pub canon_c: Option<Array>,
	pub canon_d: Option<Array>,
}

impl DharaCache {
	pub fn new() -> Self {
		Self::default()
	}
}

/// Lightweight per-layer rollback state, captured *before* feeding
/// tokens that may later be rejected and restored to undo them.
///
/// emelex patch (not upstream): `Attention` deliberately holds NO
/// arrays — a `KvCache` rolls back by decrementing its offset
/// ([`KvCache::truncate_to`]), so snapshotting it must not pin the
/// (large) K/V buffers. Gated-delta / dhara recurrent state is folded
/// destructively on every step and cannot be rewound, so their (small,
/// fixed-size) state arrays are cloned — an O(1) refcount bump per
/// `Array` — and swapped back wholesale on rollback.
pub enum LayerRollback {
	Attention {
		offset: i32,
	},
	GatedDelta {
		conv: Option<Array>,
		recur: Option<Array>,
	},
	Dhara {
		offset: i32,
		canon_a: Option<Array>,
		canon_b_q: Option<Array>,
		canon_b_k: Option<Array>,
		canon_b_v: Option<Array>,
		canon_c: Option<Array>,
		canon_d: Option<Array>,
	},
}

/// Either flavor of per-layer cache a model architecture may need.
///
/// Cloning is O(1) per `Array` field (a refcount bump on MLX's
/// `shared_ptr`-backed buffer, not a deep copy - see `Array::clone` in
/// `array.rs`), so cloning a whole `Vec<LayerCache>` to fork a cached
/// prefix (see `prompt_cache.rs`) costs O(num_layers), not O(cache size).
#[derive(Clone)]
pub enum LayerCache {
	Attention(KvCache),
	GatedDelta(GatedDeltaCache),
	Dhara(DharaCache),
}

impl LayerRollback {
	/// emelex patch: variant name for error messages.
	fn kind_name(&self) -> &'static str {
		match self {
			LayerRollback::Attention { .. } => "attention",
			LayerRollback::GatedDelta { .. } => "gated-delta",
			LayerRollback::Dhara { .. } => "dhara",
		}
	}
}

impl LayerCache {
	/// emelex patch: variant name for error messages.
	fn kind_name(&self) -> &'static str {
		match self {
			LayerCache::Attention(_) => "attention",
			LayerCache::GatedDelta(_) => "gated-delta",
			LayerCache::Dhara(_) => "dhara",
		}
	}

	pub fn new_attention() -> Self {
		LayerCache::Attention(KvCache::new())
	}

	/// An attention cache retaining only a trailing window of positions
	/// (sliding-window layers).
	pub fn new_attention_windowed(window: i32) -> Self {
		LayerCache::Attention(KvCache::windowed(window))
	}

	pub fn new_gated_delta() -> Self {
		LayerCache::GatedDelta(GatedDeltaCache::new())
	}

	pub fn new_dhara() -> Self {
		LayerCache::Dhara(DharaCache::new())
	}

	pub fn as_attention(&mut self) -> Result<&mut KvCache> {
		match self {
			LayerCache::Attention(c) => Ok(c),
			_ => Err(Error::Model(
				"expected an attention cache, found a different cache kind".into(),
			)),
		}
	}

	pub fn as_gated_delta(&mut self) -> Result<&mut GatedDeltaCache> {
		match self {
			LayerCache::GatedDelta(c) => Ok(c),
			_ => Err(Error::Model(
				"expected a gated-delta cache, found a different cache kind".into(),
			)),
		}
	}

	pub fn as_dhara(&mut self) -> Result<&mut DharaCache> {
		match self {
			LayerCache::Dhara(c) => Ok(c),
			_ => Err(Error::Model(
				"expected a dhara cache, found a different cache kind".into(),
			)),
		}
	}

	/// Capture this layer's rollback state (cheap: no attention buffers,
	/// O(1) refcount bumps for recurrent state).
	///
	/// emelex patch (not upstream): rollback primitive — see
	/// [`LayerRollback`].
	pub fn rollback_state(&self) -> LayerRollback {
		match self {
			LayerCache::Attention(c) => LayerRollback::Attention { offset: c.offset },
			LayerCache::GatedDelta(c) => LayerRollback::GatedDelta {
				conv: c.conv_state.clone(),
				recur: c.recur_state.clone(),
			},
			LayerCache::Dhara(c) => LayerRollback::Dhara {
				offset: c.attn.offset,
				canon_a: c.canon_a.clone(),
				canon_b_q: c.canon_b_q.clone(),
				canon_b_k: c.canon_b_k.clone(),
				canon_b_v: c.canon_b_v.clone(),
				canon_c: c.canon_c.clone(),
				canon_d: c.canon_d.clone(),
			},
		}
	}

	/// Restore a snapshot taken by [`rollback_state`](Self::rollback_state):
	/// attention rewinds via [`KvCache::truncate_to`]; gated-delta / dhara
	/// swap the cloned state arrays back in.
	///
	/// emelex patch (not upstream): rollback primitive. Errs on a
	/// kind mismatch, or when a window trim since the snapshot already
	/// discarded the target attention positions.
	pub fn rollback(&mut self, state: &LayerRollback) -> Result<()> {
		// By-reference on purpose: a failed rollback must not consume the
		// snapshot (the caller may need it for diagnostics or a retry).
		// The clones below are refcount bumps on small state arrays.
		match (self, state) {
			(LayerCache::Attention(c), LayerRollback::Attention { offset }) => {
				c.truncate_to(*offset)
			}
			(LayerCache::GatedDelta(c), LayerRollback::GatedDelta { conv, recur }) => {
				c.conv_state = conv.clone();
				c.recur_state = recur.clone();
				Ok(())
			}
			(
				LayerCache::Dhara(c),
				LayerRollback::Dhara {
					offset,
					canon_a,
					canon_b_q,
					canon_b_k,
					canon_b_v,
					canon_c,
					canon_d,
				},
			) => {
				// Restore the owned canon states BEFORE the fallible
				// truncate, so a truncate error cannot leave the cache
				// half-rolled-back (states restored either way; the offset
				// error still propagates).
				c.canon_a = canon_a.clone();
				c.canon_b_q = canon_b_q.clone();
				c.canon_b_k = canon_b_k.clone();
				c.canon_b_v = canon_b_v.clone();
				c.canon_c = canon_c.clone();
				c.canon_d = canon_d.clone();
				c.attn.truncate_to(*offset)
			}
			(cache, state) => Err(Error::Model(format!(
				"rollback state kind ({}) does not match the cache kind ({})",
				state.kind_name(),
				cache.kind_name(),
			))),
		}
	}

	/// True iff this layer's cache has never been fed.
	///
	/// emelex patch (not upstream): rollback support — a pristine cache
	/// needs no snapshot/rollback bookkeeping, and a rollback-to-zero can
	/// be satisfied by resetting to a fresh cache. Caveat: a cache
	/// poisoned by a *first* failed forward (state taken, nothing
	/// written back) is indistinguishable from pristine — irrelevant
	/// under the invalid-cache exit contract (poisoned caches must not be
	/// reused), noted for honesty.
	pub fn is_pristine(&self) -> bool {
		match self {
			LayerCache::Attention(c) => c.is_pristine(),
			LayerCache::GatedDelta(c) => c.conv_state.is_none() && c.recur_state.is_none(),
			LayerCache::Dhara(c) => {
				c.attn.is_pristine()
					&& c.canon_a.is_none()
					&& c.canon_b_q.is_none()
					&& c.canon_b_k.is_none()
					&& c.canon_b_v.is_none()
					&& c.canon_c.is_none()
					&& c.canon_d.is_none()
			}
		}
	}
}

/// Whether rolling `caches` back to an *arbitrary* position (rather than
/// a saved [`LayerRollback`] snapshot) requires re-feeding the tokens
/// from that position.
///
/// emelex patch (not upstream): attention caches rewind in place to any
/// retained position ([`KvCache::truncate_to`]), but gated-delta / dhara
/// recurrent state only exists at snapshot boundaries — with any
/// non-attention layer present, positions between snapshots can only be
/// reached by restoring an earlier state and re-feeding.
pub fn needs_refeed(caches: &[LayerCache]) -> bool {
	caches
		.iter()
		.any(|c| !matches!(c, LayerCache::Attention(_)))
}

#[cfg(test)]
mod tests {
	use super::*;

	fn seq_array(start: f32, b: i32, h: i32, l: i32, d: i32) -> Array {
		let len = (b * h * l * d) as usize;
		let data: Vec<f32> = (0..len).map(|i| start + i as f32).collect();
		Array::from_slice(&data, &[b, h, l, d]).unwrap()
	}

	/// emelex patch: a windowed cache must agree with the unwindowed
	/// cache on the retained tail, stay memory-bounded, and keep the
	/// absolute offset counting for rope positions.
	#[test]
	fn windowed_cache_bounds_retention_and_matches_tail() {
		let (b, h, d) = (1, 2, 4);
		let window = 16;
		let mut windowed = KvCache::windowed(window);
		let mut full = KvCache::new();
		let mut cursor = 0.0f32;
		// Mixed prefill chunks and single-token decode steps, crossing
		// several growth-slack trims (window 16, slack 256 -> trims
		// begin past 272 retained).
		for &l in &[40i32, 1, 1, 250, 1, 1, 30, 1] {
			let k = seq_array(cursor, b, h, l, d);
			let v = seq_array(cursor + 0.5, b, h, l, d);
			cursor += (h * l * d) as f32;

			let (win_k, _win_v) = windowed.update_and_fetch(k.clone(), v.clone()).unwrap();
			let (full_k, _full_v) = full.update_and_fetch(k, v).unwrap();

			// Absolute offsets agree; the windowed retention is bounded.
			assert_eq!(windowed.offset(), full.offset());
			let retained = win_k.dim(-2);
			assert!(
				retained <= window + 256 + l,
				"retained {retained} exceeds window + slack"
			);
			assert_eq!(retained, windowed.offset() - windowed.start());

			// The retained tail is element-for-element the full cache's
			// tail.
			let total = full_k.dim(-2);
			let tail =
				ops::slice(&full_k, &[0, 0, total - retained, 0], &[b, h, total, d]).unwrap();
			assert_eq!(win_k.to_vec_f32().unwrap(), tail.to_vec_f32().unwrap());
		}
		// Enough history flowed to force at least one trim.
		assert!(windowed.start() > 0, "no trim happened");
	}

	/// emelex patch regression: the amortized cache must be
	/// element-for-element identical to the naive concat it replaced,
	/// across single-token steps, multi-token prefill chunks, and the
	/// growth boundary.
	#[test]
	fn amortized_cache_matches_naive_concat() {
		let (b, h, d) = (1, 2, 4);
		let mut cache = KvCache::new();
		let mut naive_k: Option<Array> = None;
		let mut naive_v: Option<Array> = None;
		let mut cursor = 0.0f32;
		for &l in &[3i32, 1, 1, 300, 1, 255, 1] {
			let k = seq_array(cursor, b, h, l, d);
			let v = seq_array(cursor + 0.5, b, h, l, d);
			cursor += (h * l * d) as f32;

			naive_k = Some(match naive_k.take() {
				Some(old) => ops::concatenate(&[&old, &k], -2).unwrap(),
				None => k.clone(),
			});
			naive_v = Some(match naive_v.take() {
				Some(old) => ops::concatenate(&[&old, &v], -2).unwrap(),
				None => v.clone(),
			});

			let (full_k, full_v) = cache.update_and_fetch(k, v).unwrap();
			let want_k = naive_k.as_ref().unwrap();
			let want_v = naive_v.as_ref().unwrap();
			assert_eq!(full_k.shape(), want_k.shape());
			assert_eq!(full_k.to_vec_f32().unwrap(), want_k.to_vec_f32().unwrap());
			assert_eq!(full_v.to_vec_f32().unwrap(), want_v.to_vec_f32().unwrap());
			assert_eq!(cache.offset(), want_k.dim(-2));
		}
	}

	fn filled(value: f32, shape: &[i32]) -> Array {
		let len: usize = shape.iter().map(|&d| d as usize).product();
		Array::from_slice(&vec![value; len], shape).unwrap()
	}

	/// emelex patch regression: truncating back and re-appending the SAME
	/// K/V must be element-identical to never having truncated — the
	/// stale rows are simply overwritten in place. The rollback here
	/// crosses a `KV_GROWTH_STEP` boundary (offset 258 -> 255, buffer
	/// grown past 256) to prove the grown buffer needs no shrinking.
	#[test]
	fn truncate_then_reappend_matches_never_truncated() {
		let (b, h, d) = (1, 2, 4);
		let mut reference = KvCache::new();
		let mut rolled = KvCache::new();
		let mut cursor = 0.0f32;
		let mut chunks = Vec::new();
		for &l in &[10i32, 248] {
			let k = seq_array(cursor, b, h, l, d);
			let v = seq_array(cursor + 0.5, b, h, l, d);
			cursor += (h * l * d) as f32;
			reference.update_and_fetch(k.clone(), v.clone()).unwrap();
			rolled.update_and_fetch(k.clone(), v.clone()).unwrap();
			chunks.push((k, v));
		}
		assert_eq!(rolled.offset(), 258);
		assert!(
			rolled.buffer_capacity() > KV_GROWTH_STEP,
			"growth not crossed"
		);

		// Roll back 3 positions (258 -> 255, below the 256 boundary) and
		// re-append the same last 3 positions of the second chunk.
		rolled.truncate_to(rolled.offset() - 3).unwrap();
		assert_eq!(rolled.offset(), 255);
		let (k2, v2) = &chunks[1];
		let tail_k = ops::slice(k2, &[0, 0, 245, 0], &[b, h, 248, d]).unwrap();
		let tail_v = ops::slice(v2, &[0, 0, 245, 0], &[b, h, 248, d]).unwrap();
		let (rb_k, rb_v) = rolled.update_and_fetch(tail_k, tail_v).unwrap();
		assert_eq!(rolled.offset(), reference.offset());

		// The re-appended cache's view equals the naive concat of
		// everything ever fed.
		let want_k = ops::concatenate(&[&chunks[0].0, &chunks[1].0], -2).unwrap();
		let want_v = ops::concatenate(&[&chunks[0].1, &chunks[1].1], -2).unwrap();
		assert_eq!(rb_k.to_vec_f32().unwrap(), want_k.to_vec_f32().unwrap());
		assert_eq!(rb_v.to_vec_f32().unwrap(), want_v.to_vec_f32().unwrap());

		// A common probe append returns identical views from both caches.
		let pk = seq_array(cursor, b, h, 1, d);
		let pv = seq_array(cursor + 0.5, b, h, 1, d);
		let (ref_k, ref_v) = reference.update_and_fetch(pk.clone(), pv.clone()).unwrap();
		let (rol_k, rol_v) = rolled.update_and_fetch(pk, pv).unwrap();
		assert_eq!(reference.offset(), rolled.offset());
		assert_eq!(ref_k.to_vec_f32().unwrap(), rol_k.to_vec_f32().unwrap());
		assert_eq!(ref_v.to_vec_f32().unwrap(), rol_v.to_vec_f32().unwrap());
	}

	/// emelex patch regression: after a windowed cache's front trim,
	/// `truncate_to` still works for positions at/above `start` and
	/// re-appending the same K/V matches a never-truncated twin; positions
	/// below `start` are gone for good and must Err.
	#[test]
	fn truncate_then_reappend_windowed_after_front_trim() {
		let (b, h, d) = (1, 1, 2);
		let window = 16;
		let mut reference = KvCache::windowed(window);
		let mut rolled = KvCache::windowed(window);
		// One 280-token prefill: the post-append trim fires
		// (280 > 16 + 256), keeping the last 16 (start = 264).
		let k = seq_array(0.0, b, h, 280, d);
		let v = seq_array(0.5, b, h, 280, d);
		reference.update_and_fetch(k.clone(), v.clone()).unwrap();
		rolled.update_and_fetch(k.clone(), v.clone()).unwrap();
		assert_eq!(rolled.start(), 264, "front trim did not happen");
		assert_eq!(rolled.offset(), 280);

		// Below start: those positions were physically discarded.
		assert!(rolled.truncate_to(263).is_err());
		// At start and above: fine.
		rolled.clone().truncate_to(264).unwrap();
		rolled.truncate_to(277).unwrap();

		// Re-append the same positions 277..280 and compare against the
		// never-truncated twin via a common probe.
		let tail_k = ops::slice(&k, &[0, 0, 277, 0], &[b, h, 280, d]).unwrap();
		let tail_v = ops::slice(&v, &[0, 0, 277, 0], &[b, h, 280, d]).unwrap();
		rolled.update_and_fetch(tail_k, tail_v).unwrap();
		assert_eq!(rolled.offset(), reference.offset());
		assert_eq!(rolled.start(), reference.start());

		let pk = seq_array(1000.0, b, h, 1, d);
		let pv = seq_array(1000.5, b, h, 1, d);
		let (ref_k, ref_v) = reference.update_and_fetch(pk.clone(), pv.clone()).unwrap();
		let (rol_k, rol_v) = rolled.update_and_fetch(pk, pv).unwrap();
		assert_eq!(ref_k.to_vec_f32().unwrap(), rol_k.to_vec_f32().unwrap());
		assert_eq!(ref_v.to_vec_f32().unwrap(), rol_v.to_vec_f32().unwrap());
		assert_eq!(reference.offset(), rolled.offset());
	}

	/// emelex patch: truncate_to bounds — below `start` and above the
	/// current `offset` are both hard errors, never silent clamps.
	#[test]
	fn truncate_to_rejects_out_of_range() {
		let mut cache = KvCache::new();
		assert!(cache.truncate_to(-1).is_err());
		assert!(cache.truncate_to(1).is_err());
		cache.truncate_to(0).unwrap(); // no-op on a fresh cache

		let (b, h, d) = (1, 1, 2);
		let k = seq_array(0.0, b, h, 5, d);
		cache.update_and_fetch(k.clone(), k).unwrap();
		assert!(cache.truncate_to(6).is_err());
		cache.truncate_to(3).unwrap();
		assert_eq!(cache.offset(), 3);
		// After rolling back, the dropped suffix is no longer "fed".
		assert!(cache.truncate_to(4).is_err());
		cache.truncate_to(3).unwrap(); // idempotent at the boundary
	}

	/// emelex patch: `would_trim(added)` must agree exactly with whether
	/// a subsequent `update_and_fetch` of `added` rows fires
	/// `trim_to_window` (observable as `start` advancing), swept across
	/// the boundary.
	#[test]
	fn would_trim_mirrors_trim_fire_condition() {
		let (b, h, d) = (1, 1, 2);
		let mut cache = KvCache::windowed(16);
		let k = seq_array(0.0, b, h, 100, d);
		cache.update_and_fetch(k.clone(), k).unwrap();
		assert_eq!(cache.start(), 0);
		// Fire condition: retained(100) + added > window(16) + 256, so
		// the boundary sits at added = 173.
		for added in [1, 100, 171, 172, 173, 174, 200, 400] {
			let mut probe = cache.clone();
			let predicted = probe.would_trim(added);
			let start_before = probe.start();
			let ka = seq_array(0.0, b, h, added, d);
			probe.update_and_fetch(ka.clone(), ka).unwrap();
			let fired = probe.start() > start_before;
			assert_eq!(
				predicted, fired,
				"would_trim({added}) = {predicted} but trim fired = {fired}"
			);
		}
		// Unwindowed caches never trim.
		assert!(!KvCache::new().would_trim(1_000_000));
	}

	/// emelex patch: gated-delta rollback round-trip — the snapshot's
	/// cloned state arrays survive arbitrary mutation of the live cache.
	#[test]
	fn gated_delta_rollback_round_trip() {
		let conv0 = seq_array(0.0, 1, 1, 2, 3);
		let recur0 = seq_array(100.0, 1, 2, 2, 2);
		let mut layer = LayerCache::new_gated_delta();
		{
			let c = layer.as_gated_delta().unwrap();
			c.conv_state = Some(conv0.clone());
			c.recur_state = Some(recur0.clone());
		}
		let snapshot = layer.rollback_state();

		// Mutate: feed different state arrays.
		{
			let c = layer.as_gated_delta().unwrap();
			c.conv_state = Some(filled(-7.0, &[1, 1, 2, 3]));
			c.recur_state = Some(filled(-9.0, &[1, 2, 2, 2]));
		}
		layer.rollback(&snapshot).unwrap();
		let c = layer.as_gated_delta().unwrap();
		assert_eq!(
			c.conv_state.as_ref().unwrap().to_vec_f32().unwrap(),
			conv0.to_vec_f32().unwrap()
		);
		assert_eq!(
			c.recur_state.as_ref().unwrap().to_vec_f32().unwrap(),
			recur0.to_vec_f32().unwrap()
		);

		// Kind mismatch is a hard error.
		assert!(
			layer
				.rollback(&LayerRollback::Attention { offset: 0 })
				.is_err()
		);
	}

	/// emelex patch: attention and dhara rollback through the
	/// `LayerCache` API — attention restores the offset only (no arrays
	/// pinned); dhara restores the attention offset plus the cloned
	/// canon-conv states.
	#[test]
	fn layer_rollback_attention_and_dhara() {
		let (b, h, d) = (1, 1, 2);
		let k = seq_array(0.0, b, h, 4, d);

		let mut attn = LayerCache::new_attention();
		attn.as_attention()
			.unwrap()
			.update_and_fetch(k.clone(), k.clone())
			.unwrap();
		let snap = attn.rollback_state();
		let k2 = seq_array(100.0, b, h, 2, d);
		attn.as_attention()
			.unwrap()
			.update_and_fetch(k2.clone(), k2)
			.unwrap();
		assert_eq!(attn.as_attention().unwrap().offset(), 6);
		attn.rollback(&snap).unwrap();
		assert_eq!(attn.as_attention().unwrap().offset(), 4);

		let mut dhara = LayerCache::new_dhara();
		let canon0 = filled(3.0, &[1, 1, 3]);
		{
			let c = dhara.as_dhara().unwrap();
			c.attn.update_and_fetch(k.clone(), k.clone()).unwrap();
			c.canon_a = Some(canon0.clone());
		}
		let snap = dhara.rollback_state();
		{
			let c = dhara.as_dhara().unwrap();
			let k3 = seq_array(200.0, b, h, 3, d);
			c.attn.update_and_fetch(k3.clone(), k3).unwrap();
			c.canon_a = Some(filled(-1.0, &[1, 1, 3]));
			c.canon_c = Some(filled(-2.0, &[1, 1, 3]));
		}
		dhara.rollback(&snap).unwrap();
		let c = dhara.as_dhara().unwrap();
		assert_eq!(c.attn.offset(), 4);
		assert_eq!(
			c.canon_a.as_ref().unwrap().to_vec_f32().unwrap(),
			canon0.to_vec_f32().unwrap()
		);
		assert!(c.canon_c.is_none());
	}

	/// emelex patch: WHY forward failures are terminal for the cache.
	/// The plain-append path `take()`s the buffers before the fallible
	/// `slice_update`, so a failed append leaves the cache poisoned
	/// (keys/values None with offset > start) rather than rolled back.
	/// The wrong-head-dim chunk here is sized so `needed <= capacity`,
	/// pinning the fault to the plain-append `slice_update` and NOT the
	/// growth branch (which has its own take).
	#[test]
	fn failed_append_poisons_cache_at_plain_append() {
		let (b, h, d) = (1, 2, 4);
		let mut cache = KvCache::new();
		// Valid small append creates one growth step of capacity.
		let k = seq_array(0.0, b, h, 4, d);
		cache.update_and_fetch(k.clone(), k).unwrap();
		assert_eq!(cache.buffer_capacity(), KV_GROWTH_STEP);
		assert_eq!(cache.offset(), 4);

		// Head dim 8 > 4: the buffer slice clamps to 4 columns, the
		// update has 8 — slice_update must fail. needed = 4 + 2 = 6,
		// far below capacity 256, so the growth branch is skipped by
		// construction.
		let bad = seq_array(0.0, b, h, 2, 8);
		assert!(cache.update_and_fetch(bad.clone(), bad).is_err());

		// Poisoned exactly as the terminal contract says: the take()
		// preceding the failed slice_update is never undone.
		assert!(cache.keys.is_none());
		assert!(cache.values.is_none());
		// The fault fired in the plain-append phase: neither the offset
		// bump (post-append) nor a trim (windowless) happened.
		assert_eq!(cache.offset(), 4);
		assert_eq!(cache.start(), 0);
		assert!(!LayerCache::Attention(cache).is_pristine());
	}

	/// emelex patch: gated-delta post-take fault. `GatedDeltaNet::forward`
	/// `take()`s `conv_state` before the fallible concatenate; a planted
	/// wrong-width conv state makes that concatenate fail, and the cache
	/// is left with `conv_state` None — same terminal-poison contract as
	/// the attention append.
	#[test]
	fn failed_gated_delta_forward_poisons_conv_state() {
		use std::collections::HashMap;

		use crate::engine::{
			models::gated_delta::{GatedDeltaConfig, GatedDeltaNet},
			nn::WeightMap,
			quant::Quantization,
		};

		// Tiny net: hidden 4, 2 v-heads / 2 k-heads of dim 2 =>
		// key_dim 4, value_dim 4, conv_dim 12, kernel 2.
		let mut tensors = HashMap::new();
		for (key, shape) in [
			("gdn.conv1d.weight", &[12, 2, 1][..]),
			("gdn.in_proj_qkv.weight", &[12, 4]),
			("gdn.in_proj_z.weight", &[4, 4]),
			("gdn.in_proj_b.weight", &[2, 4]),
			("gdn.in_proj_a.weight", &[2, 4]),
			("gdn.dt_bias", &[2]),
			("gdn.A_log", &[2]),
			("gdn.norm.weight", &[2]),
			("gdn.out_proj.weight", &[4, 4]),
		] {
			tensors.insert(key.to_string(), filled(0.1, shape));
		}
		let mut w = WeightMap::new(tensors, Quantization::default());
		let cfg = GatedDeltaConfig {
			num_v_heads: 2,
			num_k_heads: 2,
			head_k_dim: 2,
			head_v_dim: 2,
			conv_kernel_size: 2,
			rms_norm_eps: 1e-6,
		};
		let net = GatedDeltaNet::load(&mut w, "gdn", &cfg).unwrap();

		let mut cache = GatedDeltaCache::new();
		// conv_dim is 12; plant a 13-wide conv state so the concatenate
		// right after the take() fails.
		cache.conv_state = Some(filled(0.0, &[1, 1, 13]));
		let inputs = filled(0.5, &[1, 1, 4]);
		assert!(net.forward(&inputs, &mut cache).is_err());
		assert!(cache.conv_state.is_none(), "take() was undone?");
		// The failure precedes the recurrence: recur_state untouched.
		assert!(cache.recur_state.is_none());
	}

	/// emelex patch: is_pristine per cache kind, and needs_refeed over a
	/// layer stack.
	#[test]
	fn pristine_and_needs_refeed() {
		let (b, h, d) = (1, 1, 2);
		let k = seq_array(0.0, b, h, 1, d);

		let mut attn = LayerCache::new_attention();
		assert!(attn.is_pristine());
		attn.as_attention()
			.unwrap()
			.update_and_fetch(k.clone(), k.clone())
			.unwrap();
		assert!(!attn.is_pristine());

		assert!(LayerCache::new_attention_windowed(8).is_pristine());

		let mut gated = LayerCache::new_gated_delta();
		assert!(gated.is_pristine());
		gated.as_gated_delta().unwrap().recur_state = Some(filled(1.0, &[1, 2, 2, 2]));
		assert!(!gated.is_pristine());

		let mut dhara = LayerCache::new_dhara();
		assert!(dhara.is_pristine());
		dhara.as_dhara().unwrap().canon_a = Some(filled(1.0, &[1, 1, 3]));
		assert!(!dhara.is_pristine());
		let mut dhara_attn_only = LayerCache::new_dhara();
		dhara_attn_only
			.as_dhara()
			.unwrap()
			.attn
			.update_and_fetch(k.clone(), k)
			.unwrap();
		assert!(!dhara_attn_only.is_pristine());

		assert!(!needs_refeed(&[]));
		assert!(!needs_refeed(&[
			LayerCache::new_attention(),
			LayerCache::new_attention_windowed(8),
		]));
		assert!(needs_refeed(&[
			LayerCache::new_attention(),
			LayerCache::new_gated_delta(),
		]));
		assert!(needs_refeed(&[LayerCache::new_dhara()]));
	}
}
