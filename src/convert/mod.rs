//! Pure translation between rig's completion types and the engine's chat
//! types.
//!
//! No I/O and no engine `Session` — everything here is testable without a
//! loaded model. Capability gating (vision/audio) is parameterized via
//! [`Capabilities`] for the same reason.

use std::{
	collections::{BTreeMap, BTreeSet},
	io::Write as _,
};

mod message;
mod options;
mod reply;

pub use reply::{choice, speculation_data, usage_data};
use rig_core::{completion::CompletionRequest, message::ToolChoice};
use serde_json::Value;

use crate::{
	client::Defaults,
	engine::{
		generate::GenerateOptions,
		tokenizer::{ChatMessage, ContentPart},
		tools::Tool,
	},
	error::Error,
};

const MAX_MESSAGES: usize = 4_096;
const MAX_TOOLS: usize = 256;
const MAX_TOOL_CALLS: usize = 4_096;
const MAX_MEDIA_PARTS: usize = 64;
const MAX_SINGLE_MESSAGE_BYTES: usize = 128 << 20;
const MAX_TOTAL_CONTENT_BYTES: usize = 256 << 20;
const MAX_SINGLE_TOOL_SCHEMA_BYTES: usize = 1 << 20;
const MAX_TOTAL_TOOL_SCHEMA_BYTES: usize = 8 << 20;
const MAX_SINGLE_TOOL_ARGUMENT_BYTES: usize = 1 << 20;
const MAX_TOOL_CALL_ID_BYTES: usize = 4 << 10;
const MAX_SINGLE_TOOL_DESCRIPTION_BYTES: usize = 1 << 20;
const MAX_TOTAL_TOOL_DESCRIPTION_BYTES: usize = 8 << 20;

struct BoundedJsonWriter {
	bytes: Vec<u8>,
	limit: usize,
	exceeded: bool,
}

impl std::io::Write for BoundedJsonWriter {
	fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
		let Some(next_len) = self.bytes.len().checked_add(buffer.len()) else {
			self.exceeded = true;
			return Err(std::io::Error::other("JSON size overflow"));
		};
		if next_len > self.limit {
			self.exceeded = true;
			return Err(std::io::Error::other("JSON size limit exceeded"));
		}
		self.bytes.extend_from_slice(buffer);
		Ok(buffer.len())
	}

	fn flush(&mut self) -> std::io::Result<()> {
		Ok(())
	}
}

fn bounded_json_bytes(value: &Value, limit: usize, what: &str) -> Result<Vec<u8>, Error> {
	if !crate::json::structurally_bounded(value) {
		return Err(Error::InvalidRequest(format!(
			"{what} exceeds JSON structural limits"
		)));
	}
	let mut writer = BoundedJsonWriter {
		bytes: Vec::new(),
		limit,
		exceeded: false,
	};
	if let Err(error) = serde_json::to_writer(&mut writer, value) {
		if writer.exceeded {
			return Err(Error::InvalidRequest(format!(
				"{what} cannot exceed {limit} bytes"
			)));
		}
		return Err(Error::InvalidRequest(format!(
			"cannot serialize {what}: {error}"
		)));
	}
	writer.flush().map_err(|error| {
		Error::InvalidRequest(format!("cannot finish serializing {what}: {error}"))
	})?;
	Ok(writer.bytes)
}

fn bounded_json_len(value: &Value, limit: usize, what: &str) -> Result<usize, Error> {
	Ok(bounded_json_bytes(value, limit, what)?.len())
}

/// What the loaded checkpoint can consume, used to reject unsupported
/// media early with a clear error instead of a deep engine failure.
#[derive(Debug, Clone, Copy)]
pub struct Capabilities {
	pub images: bool,
	pub audio: bool,
}

/// A fully translated request, ready for the engine.
#[derive(Debug)]
pub struct EngineRequest {
	pub messages: Vec<ChatMessage>,
	pub tools: Option<Vec<Tool>>,
	pub options: GenerateOptions,
}

/// Translate a rig [`CompletionRequest`] into engine inputs.
pub fn request(
	request: &CompletionRequest,
	capabilities: Capabilities,
	defaults: &Defaults,
) -> Result<EngineRequest, Error> {
	preflight_request_inputs(request)?;
	let options = options::generate_options(request, defaults)?;
	let (tools, tool_instruction) = options::tools_and_instruction(request)?;
	let schema_instruction = options::schema_instruction(request)?;

	let mut system_parts: Vec<String> = Vec::new();
	if let Some(preamble) = &request.preamble
		&& !preamble.trim().is_empty()
	{
		system_parts.push(preamble.clone());
	}

	// Walk history once: System messages merge into the system block (in
	// order), everything else converts in place.
	let mut history: Vec<ChatMessage> = Vec::new();
	for message in request.chat_history.iter() {
		message::push_message(&mut history, &mut system_parts, message, capabilities)?;
	}
	system_parts.extend(tool_instruction);
	system_parts.extend(schema_instruction);

	let mut messages: Vec<ChatMessage> = Vec::new();
	if !system_parts.is_empty() {
		messages.push(ChatMessage::system(join_bounded(
			&system_parts,
			"\n\n",
			MAX_SINGLE_MESSAGE_BYTES,
			"merged system block",
		)?));
	}
	if let Some(documents) = request.normalized_documents() {
		message::push_message(&mut messages, &mut Vec::new(), &documents, capabilities)?;
	}
	messages.append(&mut history);
	validate_tool_choice_history(request.tool_choice.as_ref(), &messages)?;
	validate_engine_request(&messages, tools.as_deref().unwrap_or_default())?;

	Ok(EngineRequest {
		messages,
		tools,
		options,
	})
}

fn validate_tool_choice_history(
	tool_choice: Option<&ToolChoice>,
	messages: &[ChatMessage],
) -> Result<(), Error> {
	let historical_tools = messages
		.iter()
		.flat_map(|message| message.tool_calls.iter())
		.map(|call| call.name.as_str())
		.collect::<BTreeSet<_>>();
	if historical_tools.is_empty() {
		return Ok(());
	}
	match tool_choice {
		Some(ToolChoice::None) => Err(Error::InvalidRequest(
			"tool_choice=None cannot be used with tool-call history because replay requires its \
			 declarations"
				.to_string(),
		)),
		Some(ToolChoice::Specific { function_names }) => {
			let excluded = historical_tools
				.into_iter()
				.filter(|name| !function_names.iter().any(|allowed| allowed == *name))
				.collect::<Vec<_>>();
			if excluded.is_empty() {
				Ok(())
			} else {
				Err(Error::InvalidRequest(format!(
					"tool_choice=Specific excludes historical tools required for replay: {}",
					excluded.join(", ")
				)))
			}
		}
		None | Some(ToolChoice::Auto | ToolChoice::Required) => Ok(()),
	}
}

fn join_bounded(
	parts: &[String],
	separator: &str,
	limit: usize,
	what: &str,
) -> Result<String, Error> {
	let content_bytes = parts.iter().try_fold(0_usize, |total, part| {
		total
			.checked_add(part.len())
			.ok_or_else(|| Error::InvalidRequest(format!("{what} size overflow")))
	})?;
	let separator_bytes = parts
		.len()
		.saturating_sub(1)
		.checked_mul(separator.len())
		.ok_or_else(|| Error::InvalidRequest(format!("{what} size overflow")))?;
	let total_bytes = content_bytes
		.checked_add(separator_bytes)
		.ok_or_else(|| Error::InvalidRequest(format!("{what} size overflow")))?;
	if total_bytes > limit {
		return Err(Error::InvalidRequest(format!(
			"{what} cannot exceed {limit} bytes"
		)));
	}

	let mut joined = String::with_capacity(total_bytes);
	for (index, part) in parts.iter().enumerate() {
		if index > 0 {
			joined.push_str(separator);
		}
		joined.push_str(part);
	}
	Ok(joined)
}

fn preflight_request_inputs(request: &CompletionRequest) -> Result<(), Error> {
	if request.chat_history.len() > MAX_MESSAGES {
		return Err(Error::InvalidRequest(format!(
			"generation accepts at most {MAX_MESSAGES} messages"
		)));
	}
	if request
		.preamble
		.as_ref()
		.is_some_and(|preamble| preamble.len() > MAX_SINGLE_MESSAGE_BYTES)
	{
		return Err(Error::InvalidRequest(
			"preamble cannot exceed 128 MiB".to_string(),
		));
	}
	if request.tools.len() > MAX_TOOLS {
		return Err(Error::InvalidRequest(format!(
			"generation accepts at most {MAX_TOOLS} tools"
		)));
	}
	let mut schema_bytes = 0_usize;
	let mut description_bytes = 0_usize;
	for tool in &request.tools {
		validate_tool_name(&tool.name)?;
		if tool.description.len() > MAX_SINGLE_TOOL_DESCRIPTION_BYTES {
			return Err(Error::InvalidRequest(format!(
				"tool {:?} description exceeds 1 MiB",
				tool.name
			)));
		}
		description_bytes = description_bytes
			.checked_add(tool.description.len())
			.ok_or_else(|| Error::InvalidRequest("tool description size overflow".to_string()))?;
		let bytes = bounded_json_len(
			&tool.parameters,
			MAX_SINGLE_TOOL_SCHEMA_BYTES,
			&format!("tool {:?} schema", tool.name),
		)?;
		schema_bytes = schema_bytes
			.checked_add(bytes)
			.ok_or_else(|| Error::InvalidRequest("tool schema size overflow".to_string()))?;
	}
	if description_bytes > MAX_TOTAL_TOOL_DESCRIPTION_BYTES {
		return Err(Error::InvalidRequest(
			"aggregate tool descriptions cannot exceed 8 MiB".to_string(),
		));
	}
	if schema_bytes > MAX_TOTAL_TOOL_SCHEMA_BYTES {
		return Err(Error::InvalidRequest(
			"aggregate tool schemas cannot exceed 8 MiB".to_string(),
		));
	}
	if let Some(rig_core::message::ToolChoice::Specific { function_names }) = &request.tool_choice {
		if function_names.len() > MAX_TOOLS {
			return Err(Error::InvalidRequest(format!(
				"specific tool choice accepts at most {MAX_TOOLS} function names"
			)));
		}
		for name in function_names {
			validate_tool_name(name)?;
		}
	}
	if request.documents.len() > MAX_MESSAGES {
		return Err(Error::InvalidRequest(format!(
			"generation accepts at most {MAX_MESSAGES} documents"
		)));
	}
	let document_bytes = request
		.documents
		.iter()
		.try_fold(0_usize, |total, document| {
			let metadata_bytes = document.additional_props.iter().try_fold(
				0_usize,
				|metadata_total, (key, value)| {
					metadata_total
						.checked_add(key.len())
						// `Document::fmt` writes values with Debug escaping.
						// Ten output bytes per input byte is a conservative
						// upper bound, plus field punctuation.
						.and_then(|sum| value.len().checked_mul(10)?.checked_add(sum))
						.and_then(|sum| sum.checked_add(8))
						.ok_or_else(|| Error::InvalidRequest("document size overflow".to_string()))
				},
			)?;
			total
				.checked_add(document.id.len())
				.and_then(|sum| sum.checked_add(document.text.len()))
				.and_then(|sum| sum.checked_add(metadata_bytes))
				.and_then(|sum| sum.checked_add(32))
				.ok_or_else(|| Error::InvalidRequest("document size overflow".to_string()))
		})?;
	if document_bytes > MAX_SINGLE_MESSAGE_BYTES {
		return Err(Error::InvalidRequest(
			"rendered documents cannot exceed 128 MiB".to_string(),
		));
	}
	Ok(())
}

// One ordered pass keeps tool-call/result protocol state and aggregate
// allocation counters synchronized; splitting it would duplicate that state.
#[expect(
	clippy::too_many_lines,
	reason = "one ordered validation pass keeps protocol state and byte accounting synchronized"
)]
fn validate_engine_request(messages: &[ChatMessage], tools: &[Tool]) -> Result<(), Error> {
	if messages.is_empty() {
		return Err(Error::InvalidRequest(
			"generation requires at least one message".to_string(),
		));
	}
	if messages.len() > MAX_MESSAGES {
		return Err(Error::InvalidRequest(format!(
			"generation accepts at most {MAX_MESSAGES} messages"
		)));
	}
	if tools.len() > MAX_TOOLS {
		return Err(Error::InvalidRequest(format!(
			"generation accepts at most {MAX_TOOLS} tools"
		)));
	}

	let mut content_bytes = 0_usize;
	let mut media_parts = 0_usize;
	let mut tool_calls = 0_usize;
	let mut seen_call_ids = BTreeSet::new();
	let mut called_tool_names = BTreeSet::new();
	let mut pending = BTreeMap::<&str, &str>::new();
	let mut saw_non_system = false;
	for message in messages {
		match message.role.as_str() {
			"system" if saw_non_system => {
				return Err(Error::InvalidRequest(
					"system messages must precede conversation turns".to_string(),
				));
			}
			"system" => {}
			"user" | "assistant" | "tool" => saw_non_system = true,
			role => {
				return Err(Error::InvalidRequest(format!(
					"unsupported message role {role:?}"
				)));
			}
		}
		if message.role != "tool" && !pending.is_empty() {
			return Err(Error::InvalidRequest(format!(
				"conversation continues before tool result(s) for {}",
				pending.keys().copied().collect::<Vec<_>>().join(", ")
			)));
		}
		if message.content.is_empty() && message.tool_calls.is_empty() {
			return Err(Error::InvalidRequest(
				"messages require content or tool calls".to_string(),
			));
		}
		if message.role == "tool"
			&& message
				.tool_call_id
				.as_deref()
				.is_none_or(|call_id| call_id.trim().is_empty())
		{
			return Err(Error::InvalidRequest(
				"tool messages require a non-empty tool_call_id".to_string(),
			));
		}
		if message.role != "tool" && message.tool_call_id.is_some() {
			return Err(Error::InvalidRequest(
				"tool_call_id is valid only on tool messages".to_string(),
			));
		}
		if message.role != "assistant" && !message.tool_calls.is_empty() {
			return Err(Error::InvalidRequest(
				"tool_calls are valid only on assistant messages".to_string(),
			));
		}
		if message.role != "assistant" && message.reasoning_content.is_some() {
			return Err(Error::InvalidRequest(
				"reasoning is valid only on assistant messages".to_string(),
			));
		}

		let mut message_bytes = 0_usize;
		for part in &message.content {
			let part_bytes = match part {
				ContentPart::Text(text) => text.len(),
				ContentPart::Image(image) => {
					if message.role != "user" {
						return Err(Error::InvalidRequest(
							"media content is valid only on user messages".to_string(),
						));
					}
					media_parts = checked_increment(media_parts, "media part count")?;
					image.bytes.len()
				}
				ContentPart::Audio(audio) => {
					if message.role != "user" {
						return Err(Error::InvalidRequest(
							"media content is valid only on user messages".to_string(),
						));
					}
					media_parts = checked_increment(media_parts, "media part count")?;
					audio.bytes.len()
				}
				ContentPart::Video(video) => {
					if message.role != "user" {
						return Err(Error::InvalidRequest(
							"media content is valid only on user messages".to_string(),
						));
					}
					media_parts = checked_increment(media_parts, "media part count")?;
					video.bytes.len()
				}
				ContentPart::Translation(_) => {
					return Err(Error::InvalidRequest(
						"translation content is not supported through the rig bridge; use \
						 the native generation API or `emelex translate`"
							.to_string(),
					));
				}
			};
			message_bytes = message_bytes.checked_add(part_bytes).ok_or_else(|| {
				Error::InvalidRequest("message content size overflow".to_string())
			})?;
		}
		if let Some(reasoning) = &message.reasoning_content {
			message_bytes = message_bytes.checked_add(reasoning.len()).ok_or_else(|| {
				Error::InvalidRequest("message content size overflow".to_string())
			})?;
		}

		for call in &message.tool_calls {
			called_tool_names.insert(call.name.as_str());
			tool_calls = checked_increment(tool_calls, "tool call count")?;
			if tool_calls > MAX_TOOL_CALLS {
				return Err(Error::InvalidRequest(format!(
					"conversation accepts at most {MAX_TOOL_CALLS} tool calls"
				)));
			}
			validate_tool_name(&call.name)?;
			if call.id.trim().is_empty() {
				return Err(Error::InvalidRequest(
					"tool call IDs cannot be empty".to_string(),
				));
			}
			if call.id.len() > MAX_TOOL_CALL_ID_BYTES {
				return Err(Error::InvalidRequest(format!(
					"tool call ID {:?} exceeds {MAX_TOOL_CALL_ID_BYTES} bytes",
					call.id
				)));
			}
			if !seen_call_ids.insert(call.id.as_str()) {
				return Err(Error::InvalidRequest(format!(
					"duplicate tool call ID {:?} across conversation",
					call.id
				)));
			}
			if !call.arguments.is_object() {
				return Err(Error::InvalidRequest(format!(
					"tool call {:?} arguments must be a JSON object",
					call.id
				)));
			}
			let argument_bytes = bounded_json_len(
				&call.arguments,
				MAX_SINGLE_TOOL_ARGUMENT_BYTES,
				&format!("tool call {:?} arguments", call.id),
			)?;
			message_bytes = message_bytes
				.checked_add(call.id.len())
				.and_then(|sum| sum.checked_add(call.name.len()))
				.and_then(|sum| sum.checked_add(argument_bytes))
				.ok_or_else(|| {
					Error::InvalidRequest("message content size overflow".to_string())
				})?;
			pending.insert(call.id.as_str(), call.name.as_str());
		}
		if let Some(call_id) = &message.tool_call_id {
			if call_id.len() > MAX_TOOL_CALL_ID_BYTES {
				return Err(Error::InvalidRequest(format!(
					"tool call ID {call_id:?} exceeds {MAX_TOOL_CALL_ID_BYTES} bytes"
				)));
			}
			message_bytes = message_bytes.checked_add(call_id.len()).ok_or_else(|| {
				Error::InvalidRequest("message content size overflow".to_string())
			})?;
		}
		if message_bytes > MAX_SINGLE_MESSAGE_BYTES {
			return Err(Error::InvalidRequest(
				"one message cannot exceed 128 MiB".to_string(),
			));
		}
		content_bytes = content_bytes.checked_add(message_bytes).ok_or_else(|| {
			Error::InvalidRequest("conversation content size overflow".to_string())
		})?;
		if media_parts > MAX_MEDIA_PARTS {
			return Err(Error::InvalidRequest(format!(
				"generation accepts at most {MAX_MEDIA_PARTS} media parts"
			)));
		}
		if content_bytes > MAX_TOTAL_CONTENT_BYTES {
			return Err(Error::InvalidRequest(
				"conversation content cannot exceed 256 MiB".to_string(),
			));
		}
		if message.role == "tool" {
			let call_id = message.tool_call_id.as_deref().unwrap_or_default();
			if pending.remove(call_id).is_none() {
				return Err(Error::InvalidRequest(format!(
					"tool result references unknown or already answered call ID {call_id:?}"
				)));
			}
		}
	}
	if !pending.is_empty() {
		return Err(Error::InvalidRequest(format!(
			"conversation has unresolved tool call(s): {}",
			pending.keys().copied().collect::<Vec<_>>().join(", ")
		)));
	}

	let mut schema_bytes = 0_usize;
	let mut tool_names = BTreeSet::new();
	for tool in tools {
		if tool.kind != "function" {
			return Err(Error::InvalidRequest(format!(
				"unsupported tool type {:?}",
				tool.kind
			)));
		}
		validate_tool_name(&tool.function.name)?;
		if !tool_names.insert(tool.function.name.as_str()) {
			return Err(Error::InvalidRequest(format!(
				"duplicate tool name {:?}",
				tool.function.name
			)));
		}
		if !tool.function.parameters.is_object() {
			return Err(Error::InvalidRequest(format!(
				"tool {:?} parameters must be a JSON Schema object",
				tool.function.name
			)));
		}
		let bytes = bounded_json_len(
			&tool.function.parameters,
			MAX_SINGLE_TOOL_SCHEMA_BYTES,
			&format!("tool {:?} schema", tool.function.name),
		)?;
		crate::engine::tools::validate_tool_schema(&tool.function.parameters).map_err(
			|reason| {
				Error::InvalidRequest(format!(
					"tool {:?} schema is invalid: {reason}",
					tool.function.name
				))
			},
		)?;
		schema_bytes = schema_bytes
			.checked_add(bytes)
			.ok_or_else(|| Error::InvalidRequest("tool schema size overflow".to_string()))?;
	}
	if let Some(undeclared) = called_tool_names
		.iter()
		.find(|name| !tool_names.contains(**name))
	{
		return Err(Error::InvalidRequest(format!(
			"tool protocol history references undeclared tool {undeclared:?}"
		)));
	}
	if schema_bytes > MAX_TOTAL_TOOL_SCHEMA_BYTES {
		return Err(Error::InvalidRequest(
			"aggregate tool schemas cannot exceed 8 MiB".to_string(),
		));
	}
	Ok(())
}

fn checked_increment(value: usize, what: &str) -> Result<usize, Error> {
	value
		.checked_add(1)
		.ok_or_else(|| Error::InvalidRequest(format!("{what} overflow")))
}

fn validate_tool_name(name: &str) -> Result<(), Error> {
	if name.is_empty()
		|| name.len() > 64
		|| !name
			.bytes()
			.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
	{
		return Err(Error::InvalidRequest(format!(
			"invalid tool name {name:?}; use 1-64 ASCII letters, digits, '_' or '-'"
		)));
	}
	Ok(())
}

#[cfg(test)]
mod tests {
	#![allow(clippy::expect_used, clippy::unwrap_used)]

	use rig_core::{
		OneOrMany,
		completion::{CompletionRequest, Document, Message, ToolDefinition},
		message::{
			AssistantContent, DocumentSourceKind, Image, Reasoning, ToolCall, ToolChoice,
			ToolFunction, ToolResult, ToolResultContent, UserContent,
		},
	};
	use serde_json::json;

	use super::*;
	use crate::{
		client::Defaults,
		engine::{
			generate::{FinishReason, GenerateReply, Usage},
			sampling::SamplingConfig,
			tools::ToolCall as EngineToolCall,
		},
	};

	fn defaults() -> Defaults {
		Defaults {
			max_tokens: 4096,
			context_tokens: 16_384,
			sampling: SamplingConfig::default(),
			enable_thinking: None,
			reasoning_budget_tokens: None,
			prompt_cache: None,
			speculative_tokens: None,
		}
	}

	fn caps() -> Capabilities {
		Capabilities {
			images: true,
			audio: true,
		}
	}

	fn base_request(history: Vec<Message>) -> CompletionRequest {
		CompletionRequest {
			model: None,
			preamble: None,
			chat_history: OneOrMany::many(history)
				.unwrap_or_else(|_| OneOrMany::one(Message::user("hi"))),
			documents: Vec::new(),
			tools: Vec::new(),
			temperature: None,
			max_tokens: None,
			tool_choice: None,
			additional_params: None,
			output_schema: None,
		}
	}

	fn completed_tool_request(tool_name: &str) -> CompletionRequest {
		let mut request = base_request(vec![
			Message::user("use a tool"),
			Message::Assistant {
				id: None,
				content: OneOrMany::one(AssistantContent::ToolCall(ToolCall::new(
					"call-1".to_string(),
					ToolFunction::new(tool_name.to_string(), json!({})),
				))),
			},
			Message::User {
				content: OneOrMany::one(UserContent::ToolResult(ToolResult {
					id: "call-1".to_string(),
					call_id: None,
					content: OneOrMany::one(ToolResultContent::text("done")),
				})),
			},
		]);
		request.tools = vec![ToolDefinition {
			name: tool_name.to_string(),
			description: "test tool".to_string(),
			parameters: json!({"type": "object"}),
		}];
		request
	}

	#[test]
	fn preamble_and_system_messages_merge_into_one_leading_system_turn() {
		let mut req = base_request(vec![
			Message::System {
				content: "Always answer in French.".to_string(),
			},
			Message::user("hello"),
		]);
		req.preamble = Some("You are helpful.".to_string());
		let er = request(&req, caps(), &defaults()).unwrap();
		assert_eq!(er.messages.len(), 2);
		assert_eq!(er.messages[0].role, "system");
		assert_eq!(
			er.messages[0].text(),
			"You are helpful.\n\nAlways answer in French."
		);
		assert_eq!(er.messages[1].role, "user");
	}

	#[test]
	fn documents_insert_after_system_before_history() {
		let mut req = base_request(vec![Message::user("what does the doc say?")]);
		req.preamble = Some("preamble".to_string());
		req.documents = vec![Document {
			id: "doc-1".to_string(),
			text: "the answer is 42".to_string(),
			additional_props: std::collections::HashMap::new(),
		}];
		let er = request(&req, caps(), &defaults()).unwrap();
		assert_eq!(er.messages.len(), 3);
		assert_eq!(er.messages[0].role, "system");
		assert_eq!(er.messages[1].role, "user");
		assert!(er.messages[1].text().contains("the answer is 42"));
		assert_eq!(er.messages[2].text(), "what does the doc say?");
	}

	#[test]
	fn tool_results_become_their_own_tool_turn() {
		let mut req = base_request(vec![
			Message::user("add 1 and 2"),
			Message::Assistant {
				id: None,
				content: OneOrMany::one(AssistantContent::ToolCall(ToolCall::new(
					"call-1".to_string(),
					ToolFunction::new("add".to_string(), json!({"a": 1, "b": 2})),
				))),
			},
			Message::User {
				content: OneOrMany::one(UserContent::ToolResult(ToolResult {
					id: "call-1".to_string(),
					call_id: None,
					content: OneOrMany::one(ToolResultContent::text("3")),
				})),
			},
		]);
		req.tools = vec![ToolDefinition {
			name: "add".to_string(),
			description: "adds".to_string(),
			parameters: json!({"type": "object"}),
		}];
		let er = request(&req, caps(), &defaults()).unwrap();
		assert_eq!(er.messages.len(), 3);
		let assistant = &er.messages[1];
		assert_eq!(assistant.role, "assistant");
		assert_eq!(assistant.tool_calls.len(), 1);
		assert_eq!(assistant.tool_calls[0].id, "call-1");
		assert_eq!(assistant.tool_calls[0].name, "add");
		let tool = &er.messages[2];
		assert_eq!(tool.role, "tool");
		assert_eq!(tool.tool_call_id.as_deref(), Some("call-1"));
		assert_eq!(tool.text(), "3");
	}

	#[test]
	fn tool_result_prefers_call_id_over_id() {
		let mut out = Vec::new();
		message::push_message(
			&mut out,
			&mut Vec::new(),
			&Message::User {
				content: OneOrMany::one(UserContent::ToolResult(ToolResult {
					id: "id-1".to_string(),
					call_id: Some("provider-id".to_string()),
					content: OneOrMany::one(ToolResultContent::text("ok")),
				})),
			},
			caps(),
		)
		.unwrap();
		assert_eq!(out[0].tool_call_id.as_deref(), Some("provider-id"));
	}

	#[test]
	fn base64_images_decode_and_urls_are_rejected() {
		let req = base_request(vec![Message::User {
			content: OneOrMany::one(UserContent::Image(Image {
				data: DocumentSourceKind::Base64("aGVsbG8=".to_string()),
				media_type: None,
				detail: None,
				additional_params: None,
			})),
		}]);
		let er = request(&req, caps(), &defaults()).unwrap();
		assert!(
			er.messages[0]
				.images()
				.next()
				.is_some_and(|image| { image.bytes == b"hello" })
		);

		let req = base_request(vec![Message::User {
			content: OneOrMany::one(UserContent::Image(Image {
				data: DocumentSourceKind::url("https://example.com/x.png"),
				media_type: None,
				detail: None,
				additional_params: None,
			})),
		}]);
		let error = request(&req, caps(), &defaults()).unwrap_err();
		assert!(matches!(error, Error::UnsupportedContent(_)));
	}

	#[test]
	fn images_rejected_without_vision_capability() {
		let req = base_request(vec![Message::User {
			content: OneOrMany::one(UserContent::Image(Image {
				data: DocumentSourceKind::raw(vec![1, 2, 3]),
				media_type: None,
				detail: None,
				additional_params: None,
			})),
		}]);
		let error = request(
			&req,
			Capabilities {
				images: false,
				audio: false,
			},
			&defaults(),
		)
		.unwrap_err();
		assert!(matches!(error, Error::UnsupportedContent(_)));
	}

	#[test]
	fn tool_result_images_fail_instead_of_disappearing() {
		let req = base_request(vec![Message::User {
			content: OneOrMany::one(UserContent::ToolResult(ToolResult {
				id: "call-1".to_string(),
				call_id: None,
				content: OneOrMany::one(ToolResultContent::Image(Image {
					data: DocumentSourceKind::raw(vec![1, 2, 3]),
					media_type: None,
					detail: None,
					additional_params: None,
				})),
			})),
		}]);
		let error = request(&req, caps(), &defaults()).unwrap_err();
		assert!(matches!(error, Error::UnsupportedContent(_)));
	}

	#[test]
	fn assistant_images_fail_instead_of_disappearing() {
		let req = base_request(vec![Message::Assistant {
			id: None,
			content: OneOrMany::one(AssistantContent::Image(Image {
				data: DocumentSourceKind::raw(vec![1, 2, 3]),
				media_type: None,
				detail: None,
				additional_params: None,
			})),
		}]);
		let error = request(&req, caps(), &defaults()).unwrap_err();
		assert!(matches!(error, Error::UnsupportedContent(_)));
	}

	#[test]
	fn non_text_reasoning_fails_instead_of_disappearing() {
		let req = base_request(vec![Message::Assistant {
			id: None,
			content: OneOrMany::one(AssistantContent::Reasoning(Reasoning::encrypted("opaque"))),
		}]);
		let error = request(&req, caps(), &defaults()).unwrap_err();
		assert!(matches!(error, Error::UnsupportedContent(_)));
	}

	#[test]
	fn assistant_reasoning_round_trips_into_reasoning_content() {
		let req = base_request(vec![
			Message::user("hi"),
			Message::Assistant {
				id: None,
				content: OneOrMany::many(vec![
					AssistantContent::Reasoning(Reasoning::new("thinking...")),
					AssistantContent::text("answer"),
				])
				.unwrap(),
			},
		]);
		let er = request(&req, caps(), &defaults()).unwrap();
		let assistant = &er.messages[1];
		assert_eq!(assistant.reasoning_content.as_deref(), Some("thinking..."));
		assert_eq!(assistant.text(), "answer");
	}

	#[test]
	fn output_schema_injects_into_system_block() {
		let mut req = base_request(vec![Message::user("extract")]);
		req.output_schema = Some(serde_json::from_value(json!({"type": "object"})).unwrap());
		let er = request(&req, caps(), &defaults()).unwrap();
		assert_eq!(er.messages[0].role, "system");
		assert!(er.messages[0].text().contains("JSON"));
		assert!(er.messages[0].text().contains("\"type\":\"object\""));
	}

	#[test]
	fn tool_choice_none_drops_tools_and_required_adds_instruction() {
		let tool = ToolDefinition {
			name: "add".to_string(),
			description: "adds".to_string(),
			parameters: json!({"type": "object"}),
		};
		let mut req = base_request(vec![Message::user("hi")]);
		req.tools = vec![tool.clone()];
		req.tool_choice = Some(ToolChoice::None);
		let er = request(&req, caps(), &defaults()).unwrap();
		assert!(er.tools.is_none());

		let mut req = base_request(vec![Message::user("hi")]);
		req.tools = vec![tool];
		req.tool_choice = Some(ToolChoice::Required);
		let er = request(&req, caps(), &defaults()).unwrap();
		assert!(er.tools.is_some());
		assert_eq!(er.messages[0].role, "system");
		assert!(er.messages[0].text().contains("MUST"));
	}

	#[test]
	fn required_and_specific_tool_choice_reject_empty_tool_set() {
		let mut required = base_request(vec![Message::user("hi")]);
		required.tool_choice = Some(ToolChoice::Required);
		assert!(matches!(
			request(&required, caps(), &defaults()),
			Err(Error::InvalidRequest(message))
				if message.contains("requires at least one available tool")
		));

		let mut specific = base_request(vec![Message::user("hi")]);
		specific.tool_choice = Some(ToolChoice::Specific {
			function_names: vec!["missing".to_string()],
		});
		assert!(matches!(
			request(&specific, caps(), &defaults()),
			Err(Error::InvalidRequest(message))
				if message.contains("unavailable functions")
		));
	}

	#[test]
	fn additional_params_overlay_and_request_fields_apply() {
		let mut req = base_request(vec![Message::user("hi")]);
		req.temperature = Some(0.7);
		req.max_tokens = Some(128);
		req.additional_params = Some(json!({
			"top_p": 0.9,
			"top_k": 40,
			"seed": 7,
			"enable_thinking": true,
			"reasoning_budget_tokens": 64,
			"prompt_cache": false,
			"ignored_unknown_key": "x"
		}));
		let er = request(&req, caps(), &defaults()).unwrap();
		let options = er.options;
		assert!((options.sampling.temperature - 0.7).abs() < f32::EPSILON);
		assert!((options.sampling.top_p - 0.9).abs() < f32::EPSILON);
		assert_eq!(options.sampling.top_k, Some(40));
		assert_eq!(options.sampling.seed, Some(7));
		assert_eq!(options.max_tokens, 128);
		assert_eq!(options.enable_thinking, Some(true));
		assert_eq!(options.reasoning_budget_tokens, Some(64));
		assert_eq!(options.prompt_cache, Some(false));
	}

	#[test]
	fn reasoning_ext_params_drive_generate_options() {
		// The typed `ReasoningExt` helpers and this conversion must agree
		// on the additional_params key names.
		let mut req = base_request(vec![Message::user("hi")]);
		req.additional_params = Some(crate::client::reasoning_params(true, Some(320)));
		let er = request(&req, caps(), &defaults()).unwrap();
		assert_eq!(er.options.enable_thinking, Some(true));
		assert_eq!(er.options.reasoning_budget_tokens, Some(320));

		let mut req = base_request(vec![Message::user("hi")]);
		req.additional_params = Some(crate::client::reasoning_params(false, None));
		let er = request(&req, caps(), &defaults()).unwrap();
		assert_eq!(er.options.enable_thinking, Some(false));
	}

	#[test]
	fn reasoning_ext_off_clears_client_default_budget() {
		let defaults = Defaults {
			enable_thinking: Some(true),
			reasoning_budget_tokens: Some(320),
			..defaults()
		};
		let mut req = base_request(vec![Message::user("hi")]);
		req.additional_params = Some(crate::client::reasoning_params(false, None));
		let er = request(&req, caps(), &defaults).unwrap();
		assert_eq!(
			(
				er.options.enable_thinking,
				er.options.reasoning_budget_tokens
			),
			(Some(false), None)
		);
	}

	#[test]
	fn wrongly_typed_additional_params_fail_loudly() {
		let mut req = base_request(vec![Message::user("hi")]);
		req.additional_params = Some(json!({"top_p": "not a number"}));
		let error = request(&req, caps(), &defaults()).unwrap_err();
		assert!(matches!(error, Error::InvalidParams(_)));
	}

	#[test]
	fn reply_choice_orders_reasoning_text_then_tool_calls() {
		let reply = GenerateReply {
			text: "answer".to_string(),
			tool_calls: vec![EngineToolCall {
				id: "call-1".to_string(),
				name: "add".to_string(),
				arguments: json!({"a": 1}),
			}],
			usage: Usage {
				prompt_tokens: 10,
				cached_tokens: 4,
				completion_tokens: 5,
			},
			reasoning: Some("hmm".to_string()),
			finish_reason: FinishReason::ToolCalls,
			speculation: None,
		};
		let choice = choice(&reply);
		let items: Vec<_> = choice.iter().collect();
		assert_eq!(items.len(), 3);
		assert!(matches!(items[0], AssistantContent::Reasoning(_)));
		assert!(matches!(items[1], AssistantContent::Text(_)));
		assert!(matches!(items[2], AssistantContent::ToolCall(call) if call.id == "call-1"));
	}

	#[test]
	fn empty_reply_degrades_to_empty_text() {
		let reply = GenerateReply::default();
		let choice = choice(&reply);
		assert_eq!(choice.len(), 1);
		assert!(matches!(choice.first(), AssistantContent::Text(_)));
	}

	#[test]
	fn usage_maps_to_rig_fields() {
		let usage = usage_data(Usage {
			prompt_tokens: 100,
			cached_tokens: 60,
			completion_tokens: 20,
		})
		.to_rig();
		assert_eq!(usage.input_tokens, 100);
		assert_eq!(usage.output_tokens, 20);
		assert_eq!(usage.total_tokens, 120);
		assert_eq!(usage.cached_input_tokens, 60);
	}

	#[test]
	fn tool_choice_specific_advertises_only_named_tools() {
		let add = ToolDefinition {
			name: "add".to_string(),
			description: "adds".to_string(),
			parameters: json!({"type": "object"}),
		};
		let sub = ToolDefinition {
			name: "subtract".to_string(),
			description: "subtracts".to_string(),
			parameters: json!({"type": "object"}),
		};
		let mut req = base_request(vec![Message::user("hi")]);
		req.tools = vec![add.clone(), sub];
		req.tool_choice = Some(ToolChoice::Specific {
			function_names: vec!["subtract".to_string()],
		});
		let er = request(&req, caps(), &defaults()).unwrap();
		let tools = er.tools.unwrap();
		assert_eq!(tools.len(), 1);
		assert_eq!(tools[0].function.name, "subtract");

		// Unknown names fail instead of silently broadening the tool set.
		let mut req = base_request(vec![Message::user("hi")]);
		req.tools = vec![add];
		req.tool_choice = Some(ToolChoice::Specific {
			function_names: vec!["no_such_tool".to_string()],
		});
		assert!(matches!(
			request(&req, caps(), &defaults()),
			Err(Error::InvalidRequest(_))
		));
	}

	#[test]
	fn tool_choice_none_rejects_tool_history_with_policy_diagnostic() {
		let mut req = completed_tool_request("add");
		req.tool_choice = Some(ToolChoice::None);
		let error = request(&req, caps(), &defaults()).unwrap_err();
		assert!(matches!(
			error,
			Error::InvalidRequest(message) if message.contains("tool_choice=None")
		));
	}

	#[test]
	fn tool_choice_specific_rejects_excluded_historical_tool() {
		let mut req = completed_tool_request("add");
		req.tools.push(ToolDefinition {
			name: "subtract".to_string(),
			description: "subtracts".to_string(),
			parameters: json!({"type": "object"}),
		});
		req.tool_choice = Some(ToolChoice::Specific {
			function_names: vec!["subtract".to_string()],
		});
		let error = request(&req, caps(), &defaults()).unwrap_err();
		assert!(matches!(
			error,
			Error::InvalidRequest(message)
				if message.contains("excludes historical tools")
		));
	}

	#[test]
	fn unresolved_tool_calls_are_rejected_before_inference() {
		let req = base_request(vec![
			Message::user("add"),
			Message::Assistant {
				id: None,
				content: OneOrMany::one(AssistantContent::ToolCall(ToolCall::new(
					"call-1".to_string(),
					ToolFunction::new("add".to_string(), json!({"a": 1})),
				))),
			},
		]);
		let error = request(&req, caps(), &defaults()).unwrap_err();
		assert!(matches!(error, Error::InvalidRequest(_)));
	}

	#[test]
	fn duplicate_tool_call_ids_are_rejected_across_one_turn() {
		let call = || {
			AssistantContent::ToolCall(ToolCall::new(
				"call-1".to_string(),
				ToolFunction::new("add".to_string(), json!({"a": 1})),
			))
		};
		let req = base_request(vec![Message::Assistant {
			id: None,
			content: OneOrMany::many(vec![call(), call()]).unwrap(),
		}]);
		let error = request(&req, caps(), &defaults()).unwrap_err();
		assert!(matches!(error, Error::InvalidRequest(_)));
	}

	#[test]
	fn invalid_tool_schemas_are_rejected_before_inference() {
		let mut req = base_request(vec![Message::user("hi")]);
		req.tools = vec![ToolDefinition {
			name: "broken".to_string(),
			description: "broken schema".to_string(),
			parameters: json!({"type": "not-a-json-schema-type"}),
		}];
		let error = request(&req, caps(), &defaults()).unwrap_err();
		assert!(matches!(error, Error::InvalidRequest(_)));
	}

	#[test]
	fn excessive_history_is_rejected_before_conversion() {
		let history = (0..=MAX_MESSAGES).map(|_| Message::user("x")).collect();
		let req = base_request(history);
		let error = request(&req, caps(), &defaults()).unwrap_err();
		assert!(matches!(error, Error::InvalidRequest(_)));
	}

	#[test]
	fn bounded_join_rejects_before_allocating_the_joined_result() {
		let parts = vec!["abcd".to_string(), "efgh".to_string()];
		let error = join_bounded(&parts, "::", 9, "test block")
			.expect_err("ten output bytes exceed the artificial limit");
		assert!(matches!(error, Error::InvalidRequest(_)));
	}

	#[test]
	fn bounded_json_accepts_exact_limit_and_rejects_next_byte() {
		let exact = Value::String("x".repeat(14));
		assert_eq!(
			bounded_json_len(&exact, 16, "test JSON").expect("exact limit"),
			16
		);
		let oversized = Value::String("x".repeat(15));
		assert!(matches!(
			bounded_json_len(&oversized, 16, "test JSON"),
			Err(Error::InvalidRequest(message)) if message.contains("16 bytes")
		));
	}

	#[test]
	fn additional_params_override_request_fields_and_reject_oversized_budgets() {
		let mut req = base_request(vec![Message::user("hi")]);
		req.temperature = Some(0.2);
		req.max_tokens = Some(u64::MAX);
		req.additional_params = Some(json!({
			"temperature": 0.9,
			"max_tokens": 64,
			"top_k": -3
		}));
		let er = request(&req, caps(), &defaults()).unwrap();
		assert!((er.options.sampling.temperature - 0.9).abs() < f32::EPSILON);
		assert_eq!(er.options.max_tokens, 64);
		assert_eq!(er.options.sampling.top_k, None);

		// Without the overlay, a nonsense request budget is rejected.
		let mut req = base_request(vec![Message::user("hi")]);
		req.max_tokens = Some(u64::MAX);
		assert!(matches!(
			request(&req, caps(), &defaults()),
			Err(Error::InvalidRequest(_))
		));
	}

	#[test]
	fn adjacent_text_parts_coalesce_into_one() {
		let req = base_request(vec![Message::User {
			content: OneOrMany::many(vec![
				UserContent::text("look at this:"),
				UserContent::Document(rig_core::message::Document {
					data: DocumentSourceKind::string("doc body"),
					media_type: None,
					additional_params: None,
				}),
				UserContent::text("what does it mean?"),
			])
			.unwrap(),
		}]);
		let er = request(&req, caps(), &defaults()).unwrap();
		assert_eq!(er.messages.len(), 1);
		assert_eq!(er.messages[0].content.len(), 1);
		assert_eq!(
			er.messages[0].text(),
			"look at this:\n\ndoc body\n\nwhat does it mean?"
		);
	}

	#[test]
	fn finish_reason_labels_are_stable() {
		use crate::{engine::generate::FinishReason, model::finish_reason_label};
		assert_eq!(finish_reason_label(FinishReason::Stop), "stop");
		assert_eq!(finish_reason_label(FinishReason::Length), "length");
		assert_eq!(finish_reason_label(FinishReason::ToolCalls), "tool_calls");
		assert_eq!(finish_reason_label(FinishReason::Aborted), "aborted");
	}
}
