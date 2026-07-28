// emelex patch (not upstream): this entire module is an emelex addition —
// the generic multi-token-prediction (MTP) surface for self-speculative
// decoding. Upstream deletes MTP weights at load time.

//! Architecture-neutral types for multi-token-prediction (MTP)
//! speculative-decoding support.
//!
//! The concrete v1 implementation lives in the Qwen3.5 module
//! (`qwen3_5::Qwen35Mtp`); the types here are the shared vocabulary the
//! [`super::Model`] fan-outs (`forward_hidden`, `forward_mtp`,
//! `new_mtp_caches`, `has_mtp`) and the decode loop speak.

use super::cache::LayerCache;
use crate::engine::{array::Array, error::Result};

/// One backbone forward pass, split at the layer-loop → final-norm → head
/// boundary.
///
/// `hidden_pre_norm` is the decoder-stack output BEFORE the final norm;
/// `logits = head(norm(hidden_pre_norm))`. The MTP module consumes the
/// pre-norm hidden rows (`prev_hidden` in `forward_mtp`), so both must
/// come out of a single pass with identical op order to the plain
/// `forward` path.
pub struct BackboneOutput {
	pub hidden_pre_norm: Array,
	pub logits: Array,
}

/// One MTP draft/priming step.
///
/// `recycle_hidden = mtp.norm(mtp_stack)` — the POST-norm MTP-stack
/// output. It is what the next *recursive* draft call consumes as
/// `prev_hidden`; committed pairs always use verified target backbone
/// hiddens instead. `logits` is the shared head projected over
/// `recycle_hidden`.
pub struct MtpStepOutput {
	pub recycle_hidden: Array,
	pub logits: Array,
}

/// The live working cache of the MTP module (v1: exactly one
/// full-attention, non-windowed [`super::cache::KvCache`]).
#[derive(Clone)]
pub struct MtpCaches(pub Vec<LayerCache>);

/// Poolable MTP snapshot for the prompt cache.
///
/// `frontier` must be a DETACHED array (contiguous + eval'd) — an
/// evaluated slice still pins its `[1, L, H]` parent, so the caller
/// enforces detachment before constructing an `MtpState`.
#[derive(Clone)]
pub struct MtpState {
	pub caches: MtpCaches,
	pub pairs_fed: usize,
	pub frontier: Array,
}

impl MtpCaches {
	/// Roll every attention layer back to `pairs_fed` fed pairs
	/// ([`super::cache::KvCache::truncate_to`]); the first `Err`
	/// propagates.
	pub fn truncate_to(&mut self, pairs_fed: usize) -> Result<()> {
		for layer in &mut self.0 {
			if let LayerCache::Attention(kv) = layer {
				kv.truncate_to(pairs_fed as i32)?;
			}
		}
		Ok(())
	}

	/// Number of (token, hidden) pairs fed so far: the offset of the
	/// (single) attention layer, `0` when empty.
	pub fn pairs_fed(&self) -> usize {
		self.0
			.iter()
			.find_map(|layer| match layer {
				LayerCache::Attention(kv) => Some(kv.offset() as usize),
				_ => None,
			})
			.unwrap_or(0)
	}
}

/// Outcome of pre-sanitize MTP detection on the raw checkpoint key set,
/// handed to `qwen3_5::sanitize` so it can preserve and canonicalize MTP
/// keys instead of deleting them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MtpDetection {
	/// No MTP weights detected (or detection rejected the layout — raw-HF
	/// orientation, or any population of a forbidden on-disk namespace):
	/// sanitize behaves byte-identically to the historical drop-all-MTP
	/// behavior.
	None,
	/// MTP weights are present under this source prefix; sanitize
	/// canonicalizes `{prefix}.*` keys (and the matching
	/// quantization-override keys) to the internal bare `mtp.*` prefix.
	///
	/// The sole supported on-disk prefix is `language_model.mtp`; detection
	/// never produces any other value here.
	/// Bare-root `mtp.*` and `language_model.model.mtp.*` are forbidden
	/// on-disk namespaces: any population of either (alone or mixed with
	/// the supported namespace) resolves to [`MtpDetection::None`].
	Prefix(String),
}
