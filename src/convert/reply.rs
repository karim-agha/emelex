//! Engine `GenerateReply` → rig assistant content and usage.

use rig_core::{
	OneOrMany,
	completion::AssistantContent,
	message::{Reasoning, ToolCall, ToolFunction},
};

use crate::{
	engine::generate::{GenerateReply, Usage},
	model::UsageData,
};

/// Order: reasoning first, then text (omitted when empty), then tool
/// calls — matching how the model produced them.
pub fn choice(reply: &GenerateReply) -> OneOrMany<AssistantContent> {
	let mut items: Vec<AssistantContent> = Vec::new();
	if let Some(reasoning) = &reply.reasoning
		&& !reasoning.is_empty()
	{
		items.push(AssistantContent::Reasoning(Reasoning::new(reasoning)));
	}
	if !reply.text.is_empty() {
		items.push(AssistantContent::text(&reply.text));
	}
	for call in &reply.tool_calls {
		items.push(AssistantContent::ToolCall(ToolCall::new(
			call.id.clone(),
			ToolFunction::new(call.name.clone(), call.arguments.clone()),
		)));
	}
	// `OneOrMany` cannot be empty; an entirely empty reply degrades to
	// empty text.
	OneOrMany::many(items).unwrap_or_else(|_| OneOrMany::one(AssistantContent::text("")))
}

/// Mirror the engine's per-call token accounting into the serializable
/// DTO carried by both response types.
pub const fn usage_data(usage: Usage) -> UsageData {
	UsageData {
		prompt_tokens: usage.prompt_tokens as u64,
		cached_tokens: usage.cached_tokens as u64,
		completion_tokens: usage.completion_tokens as u64,
	}
}

/// Mirror this reply's speculative-decoding accounting into its public
/// response DTO.
pub fn speculation_data(
	stats: Option<&crate::engine::generate::SpeculationStats>,
) -> Option<crate::model::SpeculationStatsData> {
	let stats = stats?;
	Some(crate::model::SpeculationStatsData {
		drafted: stats.drafted,
		rounds: stats.rounds,
		accepted_by_depth: stats.accepted_by_depth.clone(),
	})
}
