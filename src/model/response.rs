//! Serializable response DTOs, independent of engine types.

#[cfg(feature = "rig")]
use rig_core::completion::{GetTokenUsage, Usage};
use serde::{Deserialize, Serialize};

/// Raw response carried by rig's `CompletionResponse` for non-streaming
/// calls: a serializable mirror of the engine's reply.
#[cfg(feature = "rig")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
	/// Final answer text (reasoning stripped).
	pub text: String,
	/// Extracted reasoning/"thinking" content, when the model emitted a
	/// recognized reasoning span.
	pub reasoning: Option<String>,
	/// Parsed tool invocations.
	pub tool_calls: Vec<ToolCallData>,
	/// Token accounting for this call.
	pub usage: UsageData,
	/// Why generation stopped (`stop`, `length`, `tool_calls`, `aborted`).
	pub finish_reason: String,
	/// Per-call MTP accounting when speculative decoding ran.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub speculation: Option<SpeculationStatsData>,
}

/// Per-call MTP self-speculative-decoding accounting, mirroring the
/// engine's `SpeculationStats`.
///
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeculationStatsData {
	/// Total draft tokens proposed across the call.
	pub drafted: u64,
	/// Speculative rounds run (a round = one draft + verify cycle).
	/// `0` means the completed generation never speculated.
	pub rounds: u64,
	/// Index `i` counts accepted draft tokens at one-based depth `i + 1`
	/// (length = max observed depth; zero-filled). Depth-1 acceptances
	/// land at index 0; full-rejection rounds increment no bucket, so
	/// `rounds - sum(accepted_by_depth)` counts full rejections.
	pub accepted_by_depth: Vec<u64>,
}

/// One parsed tool invocation.
#[cfg(feature = "rig")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallData {
	/// Engine-assigned call ID.
	pub id: String,
	/// Tool name.
	pub name: String,
	/// Parsed JSON arguments.
	pub arguments: serde_json::Value,
}

/// Per-call token accounting, mirroring the engine's `usage` block.
#[cfg(feature = "rig")]
#[allow(
	clippy::struct_field_names,
	reason = "public token accounting follows established provider vocabulary"
)]
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct UsageData {
	/// Total input tokens in the fully rendered prompt.
	pub prompt_tokens: u64,
	/// How many prompt tokens were served from the KV prompt cache.
	pub cached_tokens: u64,
	/// Tokens generated in the reply.
	pub completion_tokens: u64,
}

#[cfg(feature = "rig")]
impl UsageData {
	pub(crate) fn to_rig(self) -> Usage {
		Usage {
			input_tokens: self.prompt_tokens,
			output_tokens: self.completion_tokens,
			total_tokens: self.prompt_tokens.saturating_add(self.completion_tokens),
			cached_input_tokens: self.cached_tokens,
			..Usage::new()
		}
	}
}

/// Raw response carried by rig's `StreamingCompletionResponse`, yielded
/// once at end of stream with the final token accounting.
#[cfg(feature = "rig")]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StreamingResponse {
	/// Token accounting for this call.
	pub usage: UsageData,
	/// Why generation stopped (`stop`, `length`, `tool_calls`, `aborted`).
	pub finish_reason: String,
	/// Per-call MTP accounting when speculative decoding ran.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub speculation: Option<SpeculationStatsData>,
}

/// Stable wire labels for the engine's finish reasons (documented as
/// `stop` / `length` / `tool_calls` / `aborted`).
#[cfg(feature = "rig")]
pub const fn finish_reason_label(reason: crate::engine::generate::FinishReason) -> &'static str {
	match reason {
		crate::engine::generate::FinishReason::Stop => "stop",
		crate::engine::generate::FinishReason::Length => "length",
		crate::engine::generate::FinishReason::ToolCalls => "tool_calls",
		crate::engine::generate::FinishReason::Aborted => "aborted",
	}
}

#[cfg(feature = "rig")]
impl GetTokenUsage for StreamingResponse {
	fn token_usage(&self) -> Usage {
		self.usage.to_rig()
	}
}
