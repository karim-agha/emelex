//! Media preprocessing shared by multi-modal architectures: turning raw
//! image/audio/video bytes into the tensors a vision/audio tower expects.
//!
//! Kept independent of any one architecture (`crate::engine::models::gemma4`,
//! and future VLMs) since the resize/patchify/frame-extraction math here is
//! largely model-family-agnostic; per-architecture code only supplies the
//! numeric parameters (patch size, pooling, token budget, ...).

pub mod audio;
pub mod image;
pub mod video;

pub(super) const MAX_ENCODED_MEDIA_BYTES: usize = 128 << 20;
pub(super) const MAX_TOTAL_ENCODED_MEDIA_BYTES: usize = 256 << 20;
pub(super) const MAX_PROCESSED_MEDIA_ITEMS: usize = 64;
/// Hard ceiling for image/audio tensors retained until the multimodal
/// prefill. This bounds aggregate amplification independently of encoded
/// request size; one item is still governed by its decoder-specific limit.
pub(super) const MAX_RETAINED_MEDIA_TENSOR_BYTES: usize = 512 << 20;
/// Defense-in-depth ceiling for aggregate image/audio placeholder expansion.
/// A generation call additionally applies its smaller effective context
/// window while each processed item is admitted.
pub(super) const MAX_MEDIA_SOFT_TOKENS: usize = 16_384;

use crate::engine::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ProcessedMediaKind {
	Image,
	Audio,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct PromptBudget {
	pub max_output_tokens: usize,
	pub context_limit: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ProcessedMediaUsage {
	encoded_items: usize,
	encoded_bytes: usize,
	images: usize,
	audio_clips: usize,
	soft_tokens: usize,
	retained_tensor_bytes: usize,
	prompt_tokens: usize,
}

/// Checked aggregate accounting performed before each processed tensor is
/// retained in the request's media queues.
pub(super) struct ProcessedMediaBudget {
	usage: ProcessedMediaUsage,
	prompt: Option<PromptBudget>,
}

impl ProcessedMediaBudget {
	pub(super) fn new(prompt_tokens: usize, prompt: Option<PromptBudget>) -> Result<Self> {
		if let Some(budget) = prompt {
			let requested = prompt_tokens
				.checked_add(budget.max_output_tokens)
				.ok_or_else(|| context_error(prompt_tokens, budget))?;
			if requested > budget.context_limit {
				return Err(context_error(prompt_tokens, budget));
			}
		}
		Ok(Self {
			usage: ProcessedMediaUsage {
				prompt_tokens,
				..ProcessedMediaUsage::default()
			},
			prompt,
		})
	}

	/// Account encoded inputs before decoding any attachment.
	pub(super) fn reserve_encoded(&mut self, bytes: usize) -> Result<()> {
		if bytes > MAX_ENCODED_MEDIA_BYTES {
			return Err(Error::Model(format!(
				"encoded media item exceeds {MAX_ENCODED_MEDIA_BYTES} byte limit"
			)));
		}
		let items = self
			.usage
			.encoded_items
			.checked_add(1)
			.ok_or_else(|| Error::Model("media attachment count overflow".to_string()))?;
		let encoded_bytes = self
			.usage
			.encoded_bytes
			.checked_add(bytes)
			.ok_or_else(|| Error::Model("encoded media byte count overflow".to_string()))?;
		if items > MAX_PROCESSED_MEDIA_ITEMS {
			return Err(Error::Model(format!(
				"request exceeds {MAX_PROCESSED_MEDIA_ITEMS} media attachment limit"
			)));
		}
		if encoded_bytes > MAX_TOTAL_ENCODED_MEDIA_BYTES {
			return Err(Error::Model(format!(
				"encoded media exceeds {MAX_TOTAL_ENCODED_MEDIA_BYTES} aggregate byte limit"
			)));
		}
		self.usage.encoded_items = items;
		self.usage.encoded_bytes = encoded_bytes;
		Ok(())
	}

	/// Admit one fully processed item. All prospective totals are validated
	/// before the usage ledger changes, so callers can run this immediately
	/// before pushing the tensor into an aggregate queue.
	pub(super) fn retain(
		&mut self,
		kind: ProcessedMediaKind,
		tensor_bytes: usize,
		soft_tokens: usize,
		prompt_extra_tokens: usize,
	) -> Result<()> {
		let processed_items = self
			.usage
			.images
			.checked_add(self.usage.audio_clips)
			.and_then(|items| items.checked_add(1))
			.ok_or_else(|| Error::Model("processed media item count overflow".to_string()))?;
		if processed_items > MAX_PROCESSED_MEDIA_ITEMS {
			return Err(Error::Model(format!(
				"processed media exceeds {MAX_PROCESSED_MEDIA_ITEMS} item limit"
			)));
		}
		let retained_tensor_bytes = self
			.usage
			.retained_tensor_bytes
			.checked_add(tensor_bytes)
			.ok_or_else(|| {
				Error::Model("processed media tensor byte count overflow".to_string())
			})?;
		if retained_tensor_bytes > MAX_RETAINED_MEDIA_TENSOR_BYTES {
			return Err(Error::Model(format!(
				"processed media tensors exceed {MAX_RETAINED_MEDIA_TENSOR_BYTES} aggregate byte limit"
			)));
		}
		let aggregate_soft_tokens = self
			.usage
			.soft_tokens
			.checked_add(soft_tokens)
			.ok_or_else(|| Error::Model("media soft-token count overflow".to_string()))?;
		if aggregate_soft_tokens > MAX_MEDIA_SOFT_TOKENS {
			return Err(Error::Model(format!(
				"processed media exceeds {MAX_MEDIA_SOFT_TOKENS} aggregate soft-token limit"
			)));
		}
		let prompt_tokens = self
			.usage
			.prompt_tokens
			.checked_add(prompt_extra_tokens)
			.ok_or_else(|| Error::Model("expanded media prompt length overflow".to_string()))?;
		if let Some(budget) = self.prompt {
			let requested = prompt_tokens
				.checked_add(budget.max_output_tokens)
				.ok_or_else(|| context_error(prompt_tokens, budget))?;
			if requested > budget.context_limit {
				return Err(context_error(prompt_tokens, budget));
			}
		}

		match kind {
			ProcessedMediaKind::Image => self.usage.images += 1,
			ProcessedMediaKind::Audio => self.usage.audio_clips += 1,
		}
		self.usage.soft_tokens = aggregate_soft_tokens;
		self.usage.retained_tensor_bytes = retained_tensor_bytes;
		self.usage.prompt_tokens = prompt_tokens;
		Ok(())
	}
}

const fn context_error(prompt_tokens: usize, budget: PromptBudget) -> Error {
	Error::ContextExceeded {
		prompt_tokens,
		max_output_tokens: budget.max_output_tokens,
		limit: budget.context_limit,
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn processed_tensor_limit_rejects_before_mutating_usage() {
		let mut budget = ProcessedMediaBudget::new(0, None).unwrap();
		budget
			.retain(
				ProcessedMediaKind::Image,
				MAX_RETAINED_MEDIA_TENSOR_BYTES,
				1,
				2,
			)
			.unwrap();
		let before = budget.usage;

		let error = budget
			.retain(ProcessedMediaKind::Audio, 1, 1, 2)
			.unwrap_err();

		assert!(matches!(error, Error::Model(_)));
		assert_eq!(budget.usage, before);
	}

	#[test]
	fn aggregate_prompt_budget_rejects_next_media_span_atomically() {
		let mut budget = ProcessedMediaBudget::new(
			10,
			Some(PromptBudget {
				max_output_tokens: 4,
				context_limit: 20,
			}),
		)
		.unwrap();
		budget.retain(ProcessedMediaKind::Image, 16, 4, 5).unwrap();
		let before = budget.usage;

		let error = budget
			.retain(ProcessedMediaKind::Audio, 16, 1, 2)
			.unwrap_err();

		assert!(matches!(
			error,
			Error::ContextExceeded {
				prompt_tokens: 17,
				max_output_tokens: 4,
				limit: 20
			}
		));
		assert_eq!(budget.usage, before);
	}

	#[test]
	fn aggregate_encoded_limit_is_checked_before_preprocessing() {
		let mut budget = ProcessedMediaBudget::new(0, None).unwrap();
		budget.reserve_encoded(MAX_ENCODED_MEDIA_BYTES).unwrap();
		budget.reserve_encoded(MAX_ENCODED_MEDIA_BYTES).unwrap();
		let before = budget.usage;

		let error = budget.reserve_encoded(1).unwrap_err();

		assert!(matches!(error, Error::Model(_)));
		assert_eq!(budget.usage, before);
	}

	#[test]
	fn aggregate_soft_token_limit_rejects_next_item_atomically() {
		let mut budget = ProcessedMediaBudget::new(0, None).unwrap();
		budget
			.retain(ProcessedMediaKind::Image, 0, MAX_MEDIA_SOFT_TOKENS, 0)
			.unwrap();
		let before = budget.usage;

		let error = budget
			.retain(ProcessedMediaKind::Audio, 0, 1, 0)
			.unwrap_err();

		assert!(matches!(error, Error::Model(_)));
		assert_eq!(budget.usage, before);
	}

	#[test]
	fn processed_image_and_audio_counts_share_one_item_limit() {
		let mut budget = ProcessedMediaBudget::new(0, None).unwrap();
		for index in 0..MAX_PROCESSED_MEDIA_ITEMS {
			let kind = if index % 2 == 0 {
				ProcessedMediaKind::Image
			} else {
				ProcessedMediaKind::Audio
			};
			budget.retain(kind, 0, 0, 0).unwrap();
		}
		assert_eq!(budget.usage.images, MAX_PROCESSED_MEDIA_ITEMS / 2);
		assert_eq!(budget.usage.audio_clips, MAX_PROCESSED_MEDIA_ITEMS / 2);
		let before = budget.usage;

		let error = budget
			.retain(ProcessedMediaKind::Image, 0, 0, 0)
			.unwrap_err();

		assert!(matches!(error, Error::Model(_)));
		assert_eq!(budget.usage, before);
	}
}
