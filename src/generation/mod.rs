//! Native generation request, response, and streaming types.

use std::{
	collections::{BTreeMap, BTreeSet},
	io::{self, Write},
	pin::Pin,
	sync::{
		Arc,
		atomic::{AtomicBool, Ordering},
	},
	task::{Context, Poll},
};

use futures::Stream;
use serde::{Deserialize, Serialize};

use crate::{
	config::ThinkingMode,
	engine::{
		generate::{
			FinishReason as EngineFinishReason, GenerateOptions as EngineOptions, GenerateReply,
			GenerationProgress as EngineGenerationProgress,
			GenerationProgressPhase as EngineGenerationProgressPhase,
			SpeculationStats as EngineSpeculationStats,
		},
		sampling::SamplingConfig,
		tokenizer::{AudioContent, ChatMessage, ContentPart, ImageContent, VideoContent},
		tools::{Tool, ToolCall as EngineToolCall, ToolFunction},
	},
	error::Error,
};

pub(crate) const MAX_MESSAGES: usize = 4_096;
const MAX_TOOLS: usize = 256;
const MAX_TOOL_CALLS: usize = 4_096;
pub(crate) const MAX_MESSAGE_CONTENT_PARTS: usize = 1_024;
const MAX_TOTAL_CONTENT_PARTS: usize = 16_384;
const MAX_MEDIA_PARTS: usize = 64;
const MAX_TOTAL_CONTENT_BYTES: usize = 256 << 20;
pub(crate) const MAX_MESSAGE_CONTENT_BYTES: usize = 128 << 20;
const MAX_REASONING_BYTES: usize = 16 << 20;
const MAX_TOTAL_REASONING_BYTES: usize = 64 << 20;
const MAX_TOOL_CALL_ID_BYTES: usize = 256;
const MAX_TOOL_ARGUMENT_BYTES: usize = 1 << 20;
pub(crate) const MAX_TOTAL_TOOL_ARGUMENT_BYTES: usize = 8 << 20;
const MAX_TOTAL_PROTOCOL_METADATA_BYTES: usize = 2 << 20;
const MAX_TOOL_DESCRIPTION_BYTES: usize = 16 << 10;
const MAX_TOTAL_TOOL_DESCRIPTION_BYTES: usize = 1 << 20;
const MAX_TOOL_SCHEMA_BYTES: usize = 1 << 20;
const MAX_TOTAL_TOOL_SCHEMA_BYTES: usize = 8 << 20;

/// Validate one encoded audio attachment without loading a model or
/// initializing MLX.
///
/// This accepts the same bounded RIFF/WAVE PCM16 and float32 container surface
/// as generation. It parses metadata and sample framing, and scans float32
/// samples for non-finite values, but does not allocate decoded or resampled
/// samples.
///
/// # Errors
///
/// Returns [`Error::UnsupportedContent`] for malformed, oversized, empty, or
/// unsupported audio.
pub fn validate_audio_bytes(bytes: &[u8]) -> Result<(), Error> {
	crate::engine::media::audio::validate_audio_bytes(bytes)
		.map_err(|error| Error::UnsupportedContent(error.to_string()))
}

/// One generation request.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct GenerationRequest {
	/// Complete conversation history.
	pub messages: Vec<Message>,
	/// Callable tools exposed to the model.
	pub tools: Vec<ToolDefinition>,
	/// Per-request option overrides.
	pub options: GenerationOptions,
}

impl GenerationRequest {
	/// Construct one text-only user request.
	pub fn text(text: impl Into<String>) -> Self {
		Self {
			messages: vec![Message::user(text)],
			..Self::default()
		}
	}

	/// Append one conversation message.
	#[must_use]
	pub fn message(mut self, message: Message) -> Self {
		self.messages.push(message);
		self
	}

	/// Replace the complete ordered conversation.
	#[must_use]
	pub fn messages(mut self, messages: impl IntoIterator<Item = Message>) -> Self {
		self.messages = messages.into_iter().collect();
		self
	}

	/// Append one callable tool definition.
	#[must_use]
	pub fn tool(mut self, tool: ToolDefinition) -> Self {
		self.tools.push(tool);
		self
	}

	/// Replace all callable tool definitions.
	#[must_use]
	pub fn tools(mut self, tools: impl IntoIterator<Item = ToolDefinition>) -> Self {
		self.tools = tools.into_iter().collect();
		self
	}

	/// Replace per-request generation options.
	#[must_use]
	pub const fn options(mut self, options: GenerationOptions) -> Self {
		self.options = options;
		self
	}

	pub(crate) fn into_engine(
		self,
		defaults: &crate::client::Defaults,
		supports_images: bool,
		supports_audio: bool,
	) -> Result<EngineRequest, Error> {
		if self.messages.is_empty() {
			return Err(Error::InvalidRequest(
				"generation requires at least one message".to_string(),
			));
		}
		self.options.validate(defaults)?;
		let mut names = BTreeSet::new();
		for tool in &self.tools {
			tool.validate()?;
			if !names.insert(tool.name.as_str()) {
				return Err(Error::InvalidRequest(format!(
					"duplicate tool name {:?}",
					tool.name
				)));
			}
		}
		validate_request_shape(&self.messages, &self.tools)?;
		let messages = self
			.messages
			.into_iter()
			.map(|message| message.into_engine(supports_images, supports_audio))
			.collect::<Result<Vec<_>, _>>()?;
		let tools = self
			.tools
			.into_iter()
			.map(ToolDefinition::into_engine)
			.collect::<Vec<_>>();
		Ok(EngineRequest {
			messages,
			tools,
			options: self.options.resolve(defaults),
		})
	}
}

#[expect(
	clippy::too_many_lines,
	reason = "single validation pass shares aggregate counters and protocol state without cloning"
)]
pub(crate) fn validate_request_shape(
	messages: &[Message],
	tools: &[ToolDefinition],
) -> Result<(), Error> {
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
	let declared_tool_names = tools
		.iter()
		.map(|tool| tool.name.as_str())
		.collect::<BTreeSet<_>>();

	let mut content_bytes = 0_usize;
	let mut content_parts = 0_usize;
	let mut reasoning_bytes = 0_usize;
	let mut media_parts = 0_usize;
	let mut tool_calls = 0_usize;
	let mut tool_argument_bytes = 0_usize;
	let mut protocol_metadata_bytes = 0_usize;
	let mut seen_call_ids = BTreeSet::new();
	let mut pending = BTreeMap::<&str, &str>::new();
	let mut saw_non_system = false;
	for message in messages {
		let mut message_content_bytes = 0_usize;
		if message.content.len() > MAX_MESSAGE_CONTENT_PARTS {
			return Err(Error::InvalidRequest(format!(
				"one message accepts at most {MAX_MESSAGE_CONTENT_PARTS} content parts"
			)));
		}
		content_parts = content_parts
			.checked_add(message.content.len())
			.ok_or_else(|| Error::InvalidRequest("content part count overflow".to_string()))?;
		if content_parts > MAX_TOTAL_CONTENT_PARTS {
			return Err(Error::InvalidRequest(format!(
				"conversation accepts at most {MAX_TOTAL_CONTENT_PARTS} content parts"
			)));
		}
		if message.role == Role::System {
			if saw_non_system {
				return Err(Error::InvalidRequest(
					"system messages must precede conversation turns".to_string(),
				));
			}
		} else {
			saw_non_system = true;
		}
		if message.role != Role::Tool && !pending.is_empty() {
			return Err(Error::InvalidRequest(format!(
				"conversation continues before tool result(s) for {}",
				pending.keys().copied().collect::<Vec<_>>().join(", ")
			)));
		}
		for part in &message.content {
			let bytes = match part {
				Content::Text(text) => text.len(),
				Content::Image(data) | Content::Audio(data) | Content::Video(data) => {
					media_parts = media_parts.checked_add(1).ok_or_else(|| {
						Error::InvalidRequest("media part count overflow".to_string())
					})?;
					data.len()
				}
				Content::Translation {
					source_lang,
					target_lang,
					text,
				} => source_lang
					.len()
					.saturating_add(target_lang.len())
					.saturating_add(text.len()),
			};
			content_bytes = content_bytes.checked_add(bytes).ok_or_else(|| {
				Error::InvalidRequest("conversation content size overflow".to_string())
			})?;
			message_content_bytes = message_content_bytes.checked_add(bytes).ok_or_else(|| {
				Error::InvalidRequest("message content size overflow".to_string())
			})?;
			if message_content_bytes > MAX_MESSAGE_CONTENT_BYTES {
				return Err(Error::InvalidRequest(
					"one message cannot exceed 128 MiB".to_string(),
				));
			}
		}
		if let Some(reasoning) = &message.reasoning {
			validate_bounded_generated_text("assistant reasoning", reasoning, MAX_REASONING_BYTES)?;
			reasoning_bytes = reasoning_bytes
				.checked_add(reasoning.len())
				.ok_or_else(|| {
					Error::InvalidRequest("conversation reasoning size overflow".to_string())
				})?;
			if reasoning_bytes > MAX_TOTAL_REASONING_BYTES {
				return Err(Error::InvalidRequest(
					"conversation reasoning cannot exceed 64 MiB".to_string(),
				));
			}
		}
		if let Some(call_id) = &message.tool_call_id {
			validate_bounded_protocol_text("tool result ID", call_id, MAX_TOOL_CALL_ID_BYTES)?;
			protocol_metadata_bytes = protocol_metadata_bytes
				.checked_add(call_id.len())
				.ok_or_else(|| {
					Error::InvalidRequest("protocol metadata size overflow".to_string())
				})?;
		}
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
		for call in &message.tool_calls {
			if !declared_tool_names.contains(call.name.as_str()) {
				return Err(Error::InvalidRequest(format!(
					"tool call {:?} references undeclared tool {:?}",
					call.id, call.name
				)));
			}
			tool_calls = tool_calls
				.checked_add(1)
				.ok_or_else(|| Error::InvalidRequest("tool call count overflow".to_string()))?;
			if tool_calls > MAX_TOOL_CALLS {
				return Err(Error::InvalidRequest(format!(
					"conversation accepts at most {MAX_TOOL_CALLS} tool calls"
				)));
			}
			if !seen_call_ids.insert(call.id.as_str()) {
				return Err(Error::InvalidRequest(format!(
					"duplicate tool call ID {:?} across conversation",
					call.id
				)));
			}
			validate_bounded_protocol_text("tool call ID", &call.id, MAX_TOOL_CALL_ID_BYTES)?;
			validate_tool_name(&call.name)?;
			protocol_metadata_bytes = protocol_metadata_bytes
				.checked_add(call.id.len())
				.and_then(|bytes| bytes.checked_add(call.name.len()))
				.ok_or_else(|| {
					Error::InvalidRequest("protocol metadata size overflow".to_string())
				})?;
			let argument_bytes = bounded_json_len(&call.arguments, MAX_TOOL_ARGUMENT_BYTES)
				.ok_or_else(|| {
					Error::InvalidRequest(format!("tool call {:?} arguments exceed 1 MiB", call.id))
				})?;
			tool_argument_bytes = tool_argument_bytes
				.checked_add(argument_bytes)
				.ok_or_else(|| Error::InvalidRequest("tool argument size overflow".to_string()))?;
			if tool_argument_bytes > MAX_TOTAL_TOOL_ARGUMENT_BYTES {
				return Err(Error::InvalidRequest(
					"aggregate tool-call arguments cannot exceed 8 MiB".to_string(),
				));
			}
			pending.insert(call.id.as_str(), call.name.as_str());
		}
		if message.role == Role::Tool {
			let call_id = message.tool_call_id.as_deref().unwrap_or_default();
			if pending.remove(call_id).is_none() {
				return Err(Error::InvalidRequest(format!(
					"tool result references unknown or already answered call ID {call_id:?}"
				)));
			}
		}
	}
	if protocol_metadata_bytes > MAX_TOTAL_PROTOCOL_METADATA_BYTES {
		return Err(Error::InvalidRequest(
			"aggregate protocol metadata cannot exceed 2 MiB".to_string(),
		));
	}
	if !pending.is_empty() {
		return Err(Error::InvalidRequest(format!(
			"conversation has unresolved tool call(s): {}",
			pending.keys().copied().collect::<Vec<_>>().join(", ")
		)));
	}

	let mut description_bytes = 0_usize;
	let mut schema_bytes = 0_usize;
	for tool in tools {
		protocol_metadata_bytes = protocol_metadata_bytes
			.checked_add(tool.name.len())
			.ok_or_else(|| Error::InvalidRequest("protocol metadata size overflow".to_string()))?;
		if tool.description.len() > MAX_TOOL_DESCRIPTION_BYTES {
			return Err(Error::InvalidRequest(format!(
				"tool {:?} description exceeds 16 KiB",
				tool.name
			)));
		}
		description_bytes = description_bytes
			.checked_add(tool.description.len())
			.ok_or_else(|| Error::InvalidRequest("tool description size overflow".to_string()))?;
		let bytes = bounded_json_len(&tool.parameters, MAX_TOOL_SCHEMA_BYTES).ok_or_else(|| {
			Error::InvalidRequest(format!("tool {:?} schema exceeds 1 MiB", tool.name))
		})?;
		schema_bytes = schema_bytes
			.checked_add(bytes)
			.ok_or_else(|| Error::InvalidRequest("tool schema size overflow".to_string()))?;
	}
	if description_bytes > MAX_TOTAL_TOOL_DESCRIPTION_BYTES {
		return Err(Error::InvalidRequest(
			"aggregate tool descriptions cannot exceed 1 MiB".to_string(),
		));
	}
	if schema_bytes > MAX_TOTAL_TOOL_SCHEMA_BYTES {
		return Err(Error::InvalidRequest(
			"aggregate tool schemas cannot exceed 8 MiB".to_string(),
		));
	}
	if protocol_metadata_bytes > MAX_TOTAL_PROTOCOL_METADATA_BYTES {
		return Err(Error::InvalidRequest(
			"aggregate protocol metadata cannot exceed 2 MiB".to_string(),
		));
	}
	Ok(())
}

struct BoundedJsonWriter {
	length: usize,
	limit: usize,
}

impl Write for BoundedJsonWriter {
	fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
		let Some(length) = self.length.checked_add(bytes.len()) else {
			return Err(io::Error::other("serialized JSON size overflow"));
		};
		if length > self.limit {
			return Err(io::Error::other("serialized JSON exceeds limit"));
		}
		self.length = length;
		Ok(bytes.len())
	}

	fn flush(&mut self) -> io::Result<()> {
		Ok(())
	}
}

pub(crate) fn bounded_json_len(value: &serde_json::Value, limit: usize) -> Option<usize> {
	if !crate::json::structurally_bounded(value) {
		return None;
	}
	let mut writer = BoundedJsonWriter { length: 0, limit };
	serde_json::to_writer(&mut writer, value)
		.ok()
		.map(|()| writer.length)
}

/// One conversation message.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Message {
	/// Message role.
	pub role: Role,
	/// Ordered text/media parts.
	pub content: Vec<Content>,
	/// Tool calls emitted by an assistant turn.
	#[serde(default, skip_serializing_if = "Vec::is_empty")]
	pub tool_calls: Vec<ToolCall>,
	/// Tool call answered by a tool-result turn.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub tool_call_id: Option<String>,
	/// Reasoning retained for template round-tripping.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub reasoning: Option<String>,
}

impl Message {
	/// Construct a message from an ordered set of text/media parts.
	pub fn with_content(role: Role, content: impl IntoIterator<Item = Content>) -> Self {
		Self {
			role,
			content: content.into_iter().collect(),
			..Self::default()
		}
	}

	/// Text-only system message.
	pub fn system(text: impl Into<String>) -> Self {
		Self::text(Role::System, text)
	}

	/// Text-only user message.
	pub fn user(text: impl Into<String>) -> Self {
		Self::text(Role::User, text)
	}

	/// Text-only assistant message.
	pub fn assistant(text: impl Into<String>) -> Self {
		Self::text(Role::Assistant, text)
	}

	/// Structured translation request (user role): translate `text` from
	/// `source_lang` to `target_lang` (BCP-47-style codes). Only accepted
	/// by models whose chat template supports translation content — check
	/// [`crate::Client::supports_translation`].
	pub fn translation(
		source_lang: impl Into<String>,
		target_lang: impl Into<String>,
		text: impl Into<String>,
	) -> Self {
		Self {
			role: Role::User,
			content: vec![Content::Translation {
				source_lang: source_lang.into(),
				target_lang: target_lang.into(),
				text: text.into(),
			}],
			..Self::default()
		}
	}

	/// Tool-result message.
	pub fn tool(call_id: impl Into<String>, text: impl Into<String>) -> Self {
		Self {
			role: Role::Tool,
			content: vec![Content::Text(text.into())],
			tool_call_id: Some(call_id.into()),
			..Self::default()
		}
	}

	fn text(role: Role, text: impl Into<String>) -> Self {
		Self {
			role,
			content: vec![Content::Text(text.into())],
			..Self::default()
		}
	}

	#[expect(
		clippy::too_many_lines,
		reason = "single conversion boundary validates every message field before moving it"
	)]
	fn into_engine(
		self,
		supports_images: bool,
		supports_audio: bool,
	) -> Result<ChatMessage, Error> {
		if self.content.len() > MAX_MESSAGE_CONTENT_PARTS {
			return Err(Error::InvalidRequest(format!(
				"one message accepts at most {MAX_MESSAGE_CONTENT_PARTS} content parts"
			)));
		}
		if self.content.is_empty() && self.tool_calls.is_empty() {
			return Err(Error::InvalidRequest(
				"messages require content or tool calls".to_string(),
			));
		}
		if self.role == Role::Tool
			&& self
				.tool_call_id
				.as_deref()
				.is_none_or(|call_id| call_id.trim().is_empty())
		{
			return Err(Error::InvalidRequest(
				"tool messages require a non-empty tool_call_id".to_string(),
			));
		}
		if self.role != Role::Tool && self.tool_call_id.is_some() {
			return Err(Error::InvalidRequest(
				"tool_call_id is valid only on tool messages".to_string(),
			));
		}
		if self.role != Role::Assistant && !self.tool_calls.is_empty() {
			return Err(Error::InvalidRequest(
				"tool_calls are valid only on assistant messages".to_string(),
			));
		}
		if self.role != Role::Assistant && self.reasoning.is_some() {
			return Err(Error::InvalidRequest(
				"reasoning is valid only on assistant messages".to_string(),
			));
		}
		if let Some(reasoning) = &self.reasoning {
			validate_bounded_generated_text("assistant reasoning", reasoning, MAX_REASONING_BYTES)?;
		}
		let mut call_ids = BTreeSet::new();
		for call in &self.tool_calls {
			validate_tool_name(&call.name)?;
			validate_bounded_protocol_text("tool call ID", &call.id, MAX_TOOL_CALL_ID_BYTES)?;
			let call_id = call.id.as_str();
			if !call_ids.insert(call_id) {
				return Err(Error::InvalidRequest(format!(
					"duplicate tool call ID {:?}",
					call.id
				)));
			}
			if !call.arguments.is_object() {
				return Err(Error::InvalidRequest(format!(
					"tool call {:?} arguments must be a JSON object",
					call.id
				)));
			}
			if bounded_json_len(&call.arguments, MAX_TOOL_ARGUMENT_BYTES).is_none() {
				return Err(Error::InvalidRequest(format!(
					"tool call {:?} arguments exceed 1 MiB",
					call.id
				)));
			}
		}
		let has_translation = self
			.content
			.iter()
			.any(|part| matches!(part, Content::Translation { .. }));
		if has_translation {
			if self.role != Role::User {
				return Err(Error::InvalidRequest(
					"translation content is valid only on user messages".to_string(),
				));
			}
			if self.content.len() != 1 {
				return Err(Error::InvalidRequest(
					"translation content must be the only part of its message".to_string(),
				));
			}
		}
		let mut bytes = 0_usize;
		for part in &self.content {
			let length = match part {
				Content::Text(text) => text.len(),
				Content::Image(data) | Content::Audio(data) | Content::Video(data) => data.len(),
				Content::Translation {
					source_lang,
					target_lang,
					text,
				} => source_lang
					.len()
					.saturating_add(target_lang.len())
					.saturating_add(text.len()),
			};
			bytes = bytes.checked_add(length).ok_or_else(|| {
				Error::InvalidRequest("message content size overflow".to_string())
			})?;
			match part {
				Content::Video(_) => {
					return Err(Error::UnsupportedContent(
						"self-contained video decoding is not available".to_string(),
					));
				}
				Content::Image(_) | Content::Audio(_) if self.role != Role::User => {
					return Err(Error::InvalidRequest(
						"media content is valid only on user messages".to_string(),
					));
				}
				Content::Image(_) if !supports_images => {
					return Err(Error::UnsupportedContent(
						"loaded model does not support image input".to_string(),
					));
				}
				Content::Audio(_) if !supports_audio => {
					return Err(Error::UnsupportedContent(
						"loaded model does not support audio input".to_string(),
					));
				}
				Content::Translation {
					source_lang,
					target_lang,
					text,
				} => {
					if source_lang.trim().is_empty() || target_lang.trim().is_empty() {
						return Err(Error::InvalidRequest(
							"translation content requires non-empty source and target \
							 language codes"
								.to_string(),
						));
					}
					if text.trim().is_empty() {
						return Err(Error::InvalidRequest(
							"translation content requires non-empty text".to_string(),
						));
					}
				}
				_ => {}
			}
		}
		if bytes > MAX_MESSAGE_CONTENT_BYTES {
			return Err(Error::InvalidRequest(
				"one message cannot exceed 128 MiB".to_string(),
			));
		}
		if bytes == 0 && self.tool_calls.is_empty() {
			return Err(Error::InvalidRequest(
				"message content cannot be empty".to_string(),
			));
		}
		let content = self.content.into_iter().map(Content::into_engine).collect();
		Ok(ChatMessage {
			role: self.role.as_str().to_string(),
			content,
			tool_calls: self
				.tool_calls
				.into_iter()
				.map(ToolCall::into_engine)
				.collect(),
			tool_call_id: self.tool_call_id,
			reasoning_content: self.reasoning,
		})
	}
}

/// Message role.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Role {
	/// System instruction.
	System,
	/// Human input.
	#[default]
	User,
	/// Model output.
	Assistant,
	/// Tool result.
	Tool,
}

impl Role {
	const fn as_str(self) -> &'static str {
		match self {
			Self::System => "system",
			Self::User => "user",
			Self::Assistant => "assistant",
			Self::Tool => "tool",
		}
	}
}

/// One text or media part.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Content {
	/// UTF-8 text.
	Text(String),
	/// Encoded image bytes.
	Image(Vec<u8>),
	/// Encoded audio bytes.
	Audio(Vec<u8>),
	/// Encoded video bytes.
	Video(Vec<u8>),
	/// Structured translation request for translation models
	/// (TranslateGemma-style templates): translate `text` from
	/// `source_lang` to `target_lang` (BCP-47-style codes, e.g. "en",
	/// "pt-BR"). Must be the only content part of a user message.
	Translation {
		/// Source language code.
		source_lang: String,
		/// Target language code.
		target_lang: String,
		/// Text to translate.
		text: String,
	},
}

impl Content {
	fn into_engine(self) -> ContentPart {
		match self {
			Self::Text(text) => ContentPart::Text(text),
			Self::Image(bytes) => ContentPart::Image(ImageContent { bytes }),
			Self::Audio(bytes) => ContentPart::Audio(AudioContent { bytes }),
			Self::Video(bytes) => ContentPart::Video(VideoContent { bytes }),
			Self::Translation {
				source_lang,
				target_lang,
				text,
			} => ContentPart::Translation(crate::engine::tokenizer::TranslationContent {
				source_lang,
				target_lang,
				text,
			}),
		}
	}
}

/// Callable function declaration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolDefinition {
	/// Function name.
	pub name: String,
	/// Human-readable behavior.
	pub description: String,
	/// JSON Schema object for arguments.
	pub parameters: serde_json::Value,
}

impl ToolDefinition {
	/// Construct a tool declaration.
	pub fn new(
		name: impl Into<String>,
		description: impl Into<String>,
		parameters: serde_json::Value,
	) -> Self {
		Self {
			name: name.into(),
			description: description.into(),
			parameters,
		}
	}

	fn into_engine(self) -> Tool {
		Tool {
			kind: "function".to_string(),
			function: ToolFunction {
				name: self.name,
				description: Some(self.description),
				parameters: self.parameters,
			},
		}
	}

	fn validate(&self) -> Result<(), Error> {
		validate_tool_name(&self.name)?;
		if self.description.len() > MAX_TOOL_DESCRIPTION_BYTES {
			return Err(Error::InvalidRequest(format!(
				"tool {:?} description exceeds 16 KiB",
				self.name
			)));
		}
		if self.description.trim().is_empty() {
			return Err(Error::InvalidRequest(format!(
				"tool {:?} description cannot be empty",
				self.name
			)));
		}
		if !self.parameters.is_object() {
			return Err(Error::InvalidRequest(format!(
				"tool {:?} parameters must be a JSON Schema object",
				self.name
			)));
		}
		crate::engine::tools::validate_tool_schema(&self.parameters).map_err(|reason| {
			Error::InvalidRequest(format!("tool {:?} schema is invalid: {reason}", self.name))
		})?;
		if bounded_json_len(&self.parameters, MAX_TOOL_SCHEMA_BYTES).is_none() {
			return Err(Error::InvalidRequest(format!(
				"tool {:?} schema exceeds structural or 1 MiB limits",
				self.name
			)));
		}
		Ok(())
	}
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

fn validate_bounded_protocol_text(label: &str, value: &str, maximum: usize) -> Result<(), Error> {
	if value.len() > maximum {
		return Err(Error::InvalidRequest(format!(
			"{label} exceeds {maximum} bytes"
		)));
	}
	if value.trim().is_empty() {
		return Err(Error::InvalidRequest(format!("{label} cannot be empty")));
	}
	if value.chars().any(char::is_control) {
		return Err(Error::InvalidRequest(format!(
			"{label} cannot contain control characters"
		)));
	}
	Ok(())
}

fn validate_bounded_generated_text(label: &str, value: &str, maximum: usize) -> Result<(), Error> {
	if value.is_empty() {
		return Err(Error::InvalidRequest(format!("{label} cannot be empty")));
	}
	if value.len() > maximum {
		return Err(Error::InvalidRequest(format!(
			"{label} exceeds {maximum} bytes"
		)));
	}
	if value
		.chars()
		.any(|character| character.is_control() && !matches!(character, '\n' | '\r' | '\t'))
	{
		return Err(Error::InvalidRequest(format!(
			"{label} contains an unsupported control character"
		)));
	}
	Ok(())
}

/// One parsed model tool invocation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct ToolCall {
	/// Process-unique call ID.
	pub id: String,
	/// Function name.
	pub name: String,
	/// Parsed JSON arguments.
	pub arguments: serde_json::Value,
}

impl ToolCall {
	/// Construct one tool invocation.
	pub fn new(
		id: impl Into<String>,
		name: impl Into<String>,
		arguments: serde_json::Value,
	) -> Self {
		Self {
			id: id.into(),
			name: name.into(),
			arguments,
		}
	}

	fn into_engine(self) -> EngineToolCall {
		EngineToolCall {
			id: self.id,
			name: self.name,
			arguments: self.arguments,
		}
	}
}

impl From<EngineToolCall> for ToolCall {
	fn from(call: EngineToolCall) -> Self {
		Self {
			id: call.id,
			name: call.name,
			arguments: call.arguments,
		}
	}
}

/// Optional per-request generation overrides.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct GenerationOptions {
	/// Maximum generated tokens.
	pub max_tokens: Option<usize>,
	/// Sampling temperature.
	pub temperature: Option<f32>,
	/// Nucleus threshold.
	pub top_p: Option<f32>,
	/// Top-k cutoff; zero disables.
	pub top_k: Option<u32>,
	/// Deterministic seed.
	pub seed: Option<u64>,
	/// Thinking policy.
	pub thinking: Option<ThinkingMode>,
	/// MTP draft depth; zero disables.
	pub speculative_tokens: Option<usize>,
	/// Maximum tokens retained for a reasoning span.
	pub reasoning_budget_tokens: Option<usize>,
	/// Prompt-cache override.
	pub prompt_cache: Option<bool>,
}

impl GenerationOptions {
	pub(crate) fn validate_shape(self) -> Result<(), Error> {
		if self.max_tokens == Some(0) {
			return Err(Error::InvalidRequest(
				"max_tokens must be positive".to_string(),
			));
		}
		if self.max_tokens.is_some_and(|value| value > 1 << 20) {
			return Err(Error::InvalidRequest(
				"max_tokens must be at most 1048576".to_string(),
			));
		}
		if self
			.temperature
			.is_some_and(|value| !value.is_finite() || !(0.0..=2.0).contains(&value))
		{
			return Err(Error::InvalidRequest(
				"temperature must be finite and in 0..=2".to_string(),
			));
		}
		if self
			.top_p
			.is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
		{
			return Err(Error::InvalidRequest(
				"top_p must be finite and in 0..=1".to_string(),
			));
		}
		if self.speculative_tokens.is_some_and(|value| value > 8) {
			return Err(Error::InvalidRequest(
				"speculative_tokens must be at most 8".to_string(),
			));
		}
		if self.reasoning_budget_tokens == Some(0) {
			return Err(Error::InvalidRequest(
				"reasoning_budget_tokens must be positive".to_string(),
			));
		}
		if self
			.reasoning_budget_tokens
			.is_some_and(|value| value > 1 << 20)
		{
			return Err(Error::InvalidRequest(
				"reasoning_budget_tokens must be at most 1048576".to_string(),
			));
		}
		if self.top_k.is_some_and(|value| value > i32::MAX as u32) {
			return Err(Error::InvalidRequest(format!(
				"top_k must be at most {}",
				i32::MAX
			)));
		}
		if self
			.reasoning_budget_tokens
			.zip(self.max_tokens)
			.is_some_and(|(budget, maximum)| budget > maximum)
		{
			return Err(Error::InvalidRequest(
				"reasoning_budget_tokens cannot exceed max_tokens".to_string(),
			));
		}
		Ok(())
	}

	/// Override the generated-token ceiling.
	#[must_use]
	pub const fn max_tokens(mut self, value: usize) -> Self {
		self.max_tokens = Some(value);
		self
	}

	/// Override sampling temperature.
	#[must_use]
	pub const fn temperature(mut self, value: f32) -> Self {
		self.temperature = Some(value);
		self
	}

	/// Override nucleus-sampling threshold.
	#[must_use]
	pub const fn top_p(mut self, value: f32) -> Self {
		self.top_p = Some(value);
		self
	}

	/// Override top-k sampling. Zero disables top-k.
	#[must_use]
	pub const fn top_k(mut self, value: u32) -> Self {
		self.top_k = Some(value);
		self
	}

	/// Select a deterministic sampling seed.
	#[must_use]
	pub const fn seed(mut self, value: u64) -> Self {
		self.seed = Some(value);
		self
	}

	/// Override the client thinking default.
	///
	/// [`ThinkingMode::Auto`] clears the per-request override. If the client
	/// has no explicit default, Emelex asks the chat template to disable
	/// reasoning; checkpoints or templates that ignore the variable may still
	/// reason.
	#[must_use]
	pub const fn thinking(mut self, value: ThinkingMode) -> Self {
		self.thinking = Some(value);
		self
	}

	/// Override MTP draft depth. Zero disables speculative decoding.
	#[must_use]
	pub const fn speculative_tokens(mut self, value: usize) -> Self {
		self.speculative_tokens = Some(value);
		self
	}

	/// Bound retained reasoning tokens.
	#[must_use]
	pub const fn reasoning_budget_tokens(mut self, value: usize) -> Self {
		self.reasoning_budget_tokens = Some(value);
		self
	}

	/// Enable or disable prompt-cache reuse.
	#[must_use]
	pub const fn prompt_cache(mut self, enabled: bool) -> Self {
		self.prompt_cache = Some(enabled);
		self
	}

	fn validate(self, defaults: &crate::client::Defaults) -> Result<(), Error> {
		self.validate_shape()?;
		let effective_max_tokens = self.max_tokens.unwrap_or(defaults.max_tokens);
		let effective_thinking = match self.thinking {
			Some(ThinkingMode::On) => Some(true),
			Some(ThinkingMode::Off) => Some(false),
			Some(ThinkingMode::Auto) | None => defaults.enable_thinking,
		};
		let reasoning_budget_tokens = match (self.reasoning_budget_tokens, self.thinking) {
			(Some(budget), _) => Some(budget),
			(None, Some(ThinkingMode::Off)) => None,
			(None, _) => defaults.reasoning_budget_tokens,
		};
		if reasoning_budget_tokens.is_some_and(|budget| budget > effective_max_tokens) {
			return Err(Error::InvalidRequest(
				"reasoning_budget_tokens cannot exceed max_tokens".to_string(),
			));
		}
		if reasoning_budget_tokens.is_some() && effective_thinking != Some(true) {
			return Err(Error::InvalidRequest(
				"reasoning_budget_tokens requires thinking to be enabled".to_string(),
			));
		}
		Ok(())
	}

	fn resolve(self, defaults: &crate::client::Defaults) -> EngineOptions {
		#[expect(
			clippy::cast_possible_wrap,
			reason = "validate rejects values above i32::MAX before option resolution"
		)]
		let top_k = self.top_k.map(|value| value as i32);
		let enable_thinking = match self.thinking {
			Some(ThinkingMode::On) => Some(true),
			Some(ThinkingMode::Off) => Some(false),
			Some(ThinkingMode::Auto) | None => defaults.enable_thinking,
		};
		let reasoning_budget_tokens = match (self.reasoning_budget_tokens, self.thinking) {
			(Some(budget), _) => Some(budget),
			(None, Some(ThinkingMode::Off)) => None,
			(None, _) => defaults.reasoning_budget_tokens,
		};
		EngineOptions {
			max_tokens: self.max_tokens.unwrap_or(defaults.max_tokens),
			context_tokens: defaults.context_tokens,
			sampling: SamplingConfig {
				temperature: self.temperature.unwrap_or(defaults.sampling.temperature),
				top_p: self.top_p.unwrap_or(defaults.sampling.top_p),
				top_k: top_k.or(defaults.sampling.top_k),
				seed: self.seed.or(defaults.sampling.seed),
			},
			enable_thinking,
			reasoning_budget_tokens,
			prompt_cache: self.prompt_cache.or(defaults.prompt_cache),
			speculative_tokens: self.speculative_tokens.or(defaults.speculative_tokens),
		}
	}
}

/// Successful generation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct GenerationResponse {
	/// Final answer text.
	pub text: String,
	/// Extracted reasoning text.
	pub reasoning: Option<String>,
	/// Parsed tool invocations.
	pub tool_calls: Vec<ToolCall>,
	/// Token counts.
	pub usage: Usage,
	/// Stop classification.
	pub finish_reason: FinishReason,
	/// MTP accounting when speculation ran.
	pub speculation: Option<SpeculationStats>,
}

impl GenerationResponse {
	pub(crate) fn from_engine(reply: GenerateReply) -> Self {
		Self {
			text: reply.text,
			reasoning: reply.reasoning,
			tool_calls: reply.tool_calls.into_iter().map(ToolCall::from).collect(),
			usage: Usage {
				prompt_tokens: reply.usage.prompt_tokens as u64,
				cached_tokens: reply.usage.cached_tokens as u64,
				completion_tokens: reply.usage.completion_tokens as u64,
			},
			finish_reason: FinishReason::from_engine(reply.finish_reason),
			speculation: reply.speculation.map(SpeculationStats::from_engine),
		}
	}
}

/// Token accounting.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
#[expect(
	clippy::struct_field_names,
	reason = "public token-counter names are explicit and stable"
)]
pub struct Usage {
	/// Rendered prompt tokens.
	pub prompt_tokens: u64,
	/// Prompt tokens reused from KV cache.
	pub cached_tokens: u64,
	/// Generated tokens.
	pub completion_tokens: u64,
}

/// Exact cumulative progress for one native model round.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct GenerationProgress {
	/// Current native generation phase.
	pub phase: GenerationProgressPhase,
	/// Fully rendered prompt tokens.
	pub prompt_tokens: u64,
	/// Prompt tokens served from the KV cache, once cache lookup completes.
	pub cached_tokens: Option<u64>,
	/// Token IDs admitted to the exact generated-token ledger so far.
	pub completion_tokens: u64,
	/// Output tokens reserved for this request.
	pub max_output_tokens: u64,
	/// Effective model/config context limit for this request.
	pub context_limit: u64,
}

impl From<EngineGenerationProgress> for GenerationProgress {
	fn from(progress: EngineGenerationProgress) -> Self {
		Self {
			phase: progress.phase.into(),
			prompt_tokens: progress.prompt_tokens as u64,
			cached_tokens: progress.cached_tokens.map(|tokens| tokens as u64),
			completion_tokens: progress.completion_tokens as u64,
			max_output_tokens: progress.max_output_tokens as u64,
			context_limit: progress.context_limit as u64,
		}
	}
}

/// Native phase represented by [`GenerationProgress`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum GenerationProgressPhase {
	/// Prompt rendering and tokenization completed; cache usage is not known yet.
	Prompt,
	/// Prompt-cache lookup completed and uncached prompt evaluation is beginning.
	Prefill,
	/// At least one generated token entered the exact completion ledger.
	Decode,
}

impl From<EngineGenerationProgressPhase> for GenerationProgressPhase {
	fn from(phase: EngineGenerationProgressPhase) -> Self {
		match phase {
			EngineGenerationProgressPhase::Prompt => Self::Prompt,
			EngineGenerationProgressPhase::Prefill => Self::Prefill,
			EngineGenerationProgressPhase::Decode => Self::Decode,
		}
	}
}

/// Why generation stopped.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum FinishReason {
	/// Natural stop token.
	#[default]
	Stop,
	/// Token limit.
	Length,
	/// Tool invocation.
	ToolCalls,
	/// Caller cancellation.
	Aborted,
}

impl FinishReason {
	const fn from_engine(reason: EngineFinishReason) -> Self {
		match reason {
			EngineFinishReason::Stop => Self::Stop,
			EngineFinishReason::Length => Self::Length,
			EngineFinishReason::ToolCalls => Self::ToolCalls,
			EngineFinishReason::Aborted => Self::Aborted,
		}
	}
}

/// MTP speculative-decoding accounting.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SpeculationStats {
	/// Proposed draft tokens.
	pub drafted: u64,
	/// Completed speculate/verify rounds.
	pub rounds: u64,
	/// Accepted prefix counts indexed by one-based depth minus one.
	pub accepted_by_depth: Vec<u64>,
}

impl SpeculationStats {
	fn from_engine(stats: EngineSpeculationStats) -> Self {
		Self {
			drafted: stats.drafted,
			rounds: stats.rounds,
			accepted_by_depth: stats.accepted_by_depth,
		}
	}
}

/// One item from bounded streaming generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
#[non_exhaustive]
pub enum GenerationEvent {
	/// Exact cumulative native generation progress.
	Progress(GenerationProgress),
	/// Answer text delta.
	Text(String),
	/// Reasoning delta.
	Reasoning(String),
	/// Parsed tool invocation.
	ToolCall(ToolCall),
	/// Terminal complete response.
	Completed(GenerationResponse),
}

/// Cancel-on-drop bounded native generation stream.
pub struct GenerationStream {
	receiver: tokio::sync::mpsc::Receiver<Result<GenerationEvent, Error>>,
	cancelled: Arc<AtomicBool>,
	completion: Option<tokio::sync::oneshot::Receiver<()>>,
}

impl GenerationStream {
	pub(crate) const fn new(
		receiver: tokio::sync::mpsc::Receiver<Result<GenerationEvent, Error>>,
		cancelled: Arc<AtomicBool>,
		completion: tokio::sync::oneshot::Receiver<()>,
	) -> Self {
		Self {
			receiver,
			cancelled,
			completion: Some(completion),
		}
	}

	/// Wait for the next delta, terminal response, or error.
	pub async fn recv(&mut self) -> Option<Result<GenerationEvent, Error>> {
		self.receiver.recv().await
	}

	/// Request cooperative cancellation and unblock a backpressured producer.
	pub fn cancel(&mut self) {
		self.cancelled.store(true, Ordering::Relaxed);
		self.receiver.close();
	}

	/// Request cooperative cancellation and wait for the inference job to
	/// leave the loaded model's dedicated thread.
	///
	/// # Errors
	///
	/// Returns an inference-channel error if the worker exits without
	/// completing the submitted job.
	pub async fn cancel_and_wait(&mut self) -> Result<(), Error> {
		self.cancel();
		let Some(completion) = self.completion.as_mut() else {
			return Ok(());
		};
		let outcome = completion.await;
		self.completion = None;
		outcome.map_err(|_| Error::InferenceChannel {
			operation: "receive",
		})
	}
}

impl Drop for GenerationStream {
	fn drop(&mut self) {
		self.cancel();
	}
}

impl Stream for GenerationStream {
	type Item = Result<GenerationEvent, Error>;

	fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
		Pin::new(&mut self.receiver).poll_recv(context)
	}
}

pub(crate) struct EngineRequest {
	pub messages: Vec<ChatMessage>,
	pub tools: Vec<Tool>,
	pub options: EngineOptions,
}

#[cfg(test)]
mod tests {
	#![allow(clippy::expect_used)]

	use std::time::Duration;

	use super::*;
	use crate::{client::Defaults, engine::sampling::SamplingConfig};

	fn defaults() -> Defaults {
		Defaults {
			max_tokens: 32,
			context_tokens: 128,
			sampling: SamplingConfig::default(),
			enable_thinking: None,
			reasoning_budget_tokens: None,
			prompt_cache: None,
			speculative_tokens: None,
		}
	}

	#[test]
	fn progress_event_round_trips_exact_usage_state() {
		let event = GenerationEvent::Progress(GenerationProgress {
			phase: GenerationProgressPhase::Prefill,
			prompt_tokens: 44_167,
			cached_tokens: Some(16_000),
			completion_tokens: 0,
			max_output_tokens: 4_096,
			context_limit: 65_536,
		});
		let encoded = serde_json::to_value(&event).expect("encode progress event");
		let decoded: GenerationEvent =
			serde_json::from_value(encoded.clone()).expect("decode progress event");

		assert_eq!(
			encoded,
			serde_json::json!({
				"type": "progress",
				"data": {
					"phase": "prefill",
					"prompt_tokens": 44_167,
					"cached_tokens": 16_000,
					"completion_tokens": 0,
					"max_output_tokens": 4_096,
					"context_limit": 65_536
				}
			})
		);
		assert!(matches!(
			decoded,
			GenerationEvent::Progress(GenerationProgress {
				phase: GenerationProgressPhase::Prefill,
				prompt_tokens: 44_167,
				cached_tokens: Some(16_000),
				completion_tokens: 0,
				max_output_tokens: 4_096,
				context_limit: 65_536,
			})
		));
	}

	#[test]
	fn rejects_invalid_message_and_tool_states_before_inference() {
		let request = GenerationRequest {
			messages: vec![Message {
				role: Role::Tool,
				content: vec![Content::Text("result".to_string())],
				..Message::default()
			}],
			tools: vec![ToolDefinition::new(
				"not valid",
				"invalid name",
				serde_json::json!([]),
			)],
			options: GenerationOptions::default(),
		};
		assert!(matches!(
			request.into_engine(&defaults(), false, false),
			Err(Error::InvalidRequest(_))
		));
	}

	#[test]
	fn message_translation_constructor_builds_sole_user_part() {
		let message = Message::translation("en", "de", "hello");
		assert_eq!(message.role, Role::User);
		assert!(matches!(
			message.content.as_slice(),
			[Content::Translation { source_lang, target_lang, text }]
				if source_lang == "en" && target_lang == "de" && text == "hello"
		));
		let request = GenerationRequest::default().message(message);
		assert!(request.into_engine(&defaults(), false, false).is_ok());
	}

	#[test]
	fn translation_part_must_be_sole_user_content() {
		let mixed = Message {
			role: Role::User,
			content: vec![
				Content::Text("extra".to_string()),
				Content::Translation {
					source_lang: "en".to_string(),
					target_lang: "de".to_string(),
					text: "hello".to_string(),
				},
			],
			..Message::default()
		};
		assert!(matches!(
			GenerationRequest::default()
				.message(mixed)
				.into_engine(&defaults(), false, false),
			Err(Error::InvalidRequest(message)) if message.contains("only part")
		));

		let mut assistant = Message::translation("en", "de", "hello");
		assistant.role = Role::Assistant;
		assert!(matches!(
			GenerationRequest::default()
				.message(assistant)
				.into_engine(&defaults(), false, false),
			Err(Error::InvalidRequest(message)) if message.contains("user messages")
		));

		let empty_codes = Message::translation(" ", "de", "hello");
		assert!(matches!(
			GenerationRequest::default()
				.message(empty_codes)
				.into_engine(&defaults(), false, false),
			Err(Error::InvalidRequest(message)) if message.contains("language codes")
		));

		let empty_text = Message::translation("en", "de", "  ");
		assert!(matches!(
			GenerationRequest::default()
				.message(empty_text)
				.into_engine(&defaults(), false, false),
			Err(Error::InvalidRequest(message)) if message.contains("non-empty text")
		));
	}

	#[test]
	fn content_translation_serde_round_trip_is_stable() {
		let content = Content::Translation {
			source_lang: "en".to_string(),
			target_lang: "de".to_string(),
			text: "hello".to_string(),
		};
		let encoded = serde_json::to_value(&content).expect("serialize");
		assert_eq!(
			encoded,
			serde_json::json!({
				"type": "translation",
				"data": {"source_lang": "en", "target_lang": "de", "text": "hello"}
			})
		);
		let decoded: Content = serde_json::from_value(encoded).expect("deserialize");
		assert!(matches!(
			decoded,
			Content::Translation { source_lang, target_lang, text }
				if source_lang == "en" && target_lang == "de" && text == "hello"
		));
	}

	#[test]
	fn rejects_unsupported_executable_schema_before_inference() {
		let request = GenerationRequest {
			messages: vec![Message::user("hello")],
			tools: vec![ToolDefinition::new(
				"lookup",
				"Lookup a value",
				serde_json::json!({
					"type": "object",
					"properties": {
						"key": {"type": "string", "pattern": "^safe$"}
					}
				}),
			)],
			..GenerationRequest::default()
		};
		assert!(matches!(
			request.into_engine(&defaults(), false, false),
			Err(Error::InvalidRequest(_))
		));
	}

	#[test]
	fn rejects_unsupported_media_before_inference() {
		let request = GenerationRequest {
			messages: vec![Message {
				role: Role::User,
				content: vec![Content::Image(vec![1, 2, 3])],
				..Message::default()
			}],
			..GenerationRequest::default()
		};
		assert!(matches!(
			request.into_engine(&defaults(), false, false),
			Err(Error::UnsupportedContent(_))
		));

		let video = GenerationRequest {
			messages: vec![Message {
				role: Role::User,
				content: vec![Content::Video(vec![1, 2, 3])],
				..Message::default()
			}],
			..GenerationRequest::default()
		};
		assert!(matches!(
			video.into_engine(&defaults(), true, true),
			Err(Error::UnsupportedContent(message)) if message.contains("video decoding")
		));
	}

	#[test]
	fn rejects_reasoning_budget_above_effective_token_limit() {
		let request = GenerationRequest {
			messages: vec![Message::user("hello")],
			options: GenerationOptions {
				max_tokens: Some(8),
				reasoning_budget_tokens: Some(9),
				..GenerationOptions::default()
			},
			..GenerationRequest::default()
		};
		assert!(matches!(
			request.into_engine(&defaults(), false, false),
			Err(Error::InvalidRequest(_))
		));
	}

	#[test]
	fn explicit_thinking_off_clears_default_reasoning_budget() {
		let defaults = Defaults {
			enable_thinking: Some(true),
			reasoning_budget_tokens: Some(16),
			..defaults()
		};
		let request = GenerationRequest {
			messages: vec![Message::user("hello")],
			options: GenerationOptions {
				thinking: Some(ThinkingMode::Off),
				..GenerationOptions::default()
			},
			..GenerationRequest::default()
		};
		let engine = request
			.into_engine(&defaults, false, false)
			.expect("explicit off");
		assert_eq!(
			(
				engine.options.enable_thinking,
				engine.options.reasoning_budget_tokens
			),
			(Some(false), None)
		);
	}

	#[test]
	fn explicit_thinking_off_rejects_explicit_reasoning_budget() {
		let defaults = Defaults {
			enable_thinking: Some(true),
			reasoning_budget_tokens: Some(16),
			..defaults()
		};
		let request = GenerationRequest {
			messages: vec![Message::user("hello")],
			options: GenerationOptions {
				thinking: Some(ThinkingMode::Off),
				reasoning_budget_tokens: Some(8),
				..GenerationOptions::default()
			},
			..GenerationRequest::default()
		};
		assert!(matches!(
			request.into_engine(&defaults, false, false),
			Err(Error::InvalidRequest(message))
				if message.contains("requires thinking")
		));
	}

	#[test]
	fn validates_tool_call_protocol_across_entire_conversation() {
		let definition =
			ToolDefinition::new("lookup", "Lookup", serde_json::json!({"type": "object"}));
		let call = ToolCall {
			id: "call-1".to_string(),
			name: "lookup".to_string(),
			arguments: serde_json::json!({}),
		};
		let valid = GenerationRequest {
			messages: vec![
				Message::user("look it up"),
				Message {
					role: Role::Assistant,
					content: Vec::new(),
					tool_calls: vec![call.clone()],
					..Message::default()
				},
				Message::tool("call-1", "done"),
				Message::user("summarize"),
			],
			tools: vec![definition.clone()],
			..GenerationRequest::default()
		};
		assert!(valid.into_engine(&defaults(), false, false).is_ok());

		let unresolved = GenerationRequest {
			messages: vec![
				Message::user("look it up"),
				Message {
					role: Role::Assistant,
					content: Vec::new(),
					tool_calls: vec![call.clone()],
					..Message::default()
				},
			],
			tools: vec![definition.clone()],
			..GenerationRequest::default()
		};
		assert!(matches!(
			unresolved.into_engine(&defaults(), false, false),
			Err(Error::InvalidRequest(message)) if message.contains("unresolved")
		));

		let duplicate = GenerationRequest {
			messages: vec![
				Message::user("first"),
				Message {
					role: Role::Assistant,
					content: Vec::new(),
					tool_calls: vec![call.clone()],
					..Message::default()
				},
				Message::tool("call-1", "one"),
				Message::user("second"),
				Message {
					role: Role::Assistant,
					content: Vec::new(),
					tool_calls: vec![call],
					..Message::default()
				},
				Message::tool("call-1", "two"),
			],
			tools: vec![definition],
			..GenerationRequest::default()
		};
		assert!(matches!(
			duplicate.into_engine(&defaults(), false, false),
			Err(Error::InvalidRequest(message)) if message.contains("duplicate tool call ID")
		));
	}

	#[test]
	fn rejects_aggregate_request_amplification() {
		let too_many_messages = GenerationRequest {
			messages: (0..=MAX_MESSAGES).map(|_| Message::user("x")).collect(),
			..GenerationRequest::default()
		};
		assert!(matches!(
			too_many_messages.into_engine(&defaults(), false, false),
			Err(Error::InvalidRequest(message)) if message.contains("messages")
		));

		let too_many_media = GenerationRequest {
			messages: vec![Message {
				role: Role::User,
				content: (0..=MAX_MEDIA_PARTS)
					.map(|_| Content::Image(Vec::new()))
					.collect(),
				..Message::default()
			}],
			..GenerationRequest::default()
		};
		assert!(matches!(
			too_many_media.into_engine(&defaults(), true, false),
			Err(Error::InvalidRequest(message)) if message.contains("media parts")
		));

		let too_many_text_parts = GenerationRequest {
			messages: vec![Message {
				role: Role::User,
				content: (0..=MAX_MESSAGE_CONTENT_PARTS)
					.map(|_| Content::Text(String::new()))
					.collect(),
				..Message::default()
			}],
			..GenerationRequest::default()
		};
		assert!(matches!(
			too_many_text_parts.into_engine(&defaults(), false, false),
			Err(Error::InvalidRequest(message)) if message.contains("content parts")
		));

		let late_system = GenerationRequest {
			messages: vec![Message::user("hello"), Message::system("too late")],
			..GenerationRequest::default()
		};
		assert!(matches!(
			late_system.into_engine(&defaults(), false, false),
			Err(Error::InvalidRequest(message)) if message.contains("must precede")
		));

		let oversized_description = GenerationRequest {
			messages: vec![Message::user("hello")],
			tools: vec![ToolDefinition::new(
				"lookup",
				"x".repeat(MAX_TOOL_DESCRIPTION_BYTES + 1),
				serde_json::json!({"type": "object"}),
			)],
			..GenerationRequest::default()
		};
		assert!(matches!(
			oversized_description.into_engine(&defaults(), false, false),
			Err(Error::InvalidRequest(message)) if message.contains("description exceeds")
		));

		let oversized_schema = GenerationRequest {
			messages: vec![Message::user("hello")],
			tools: vec![ToolDefinition::new(
				"lookup",
				"Lookup",
				serde_json::json!({
					"type": "object",
					"description": "x".repeat(MAX_TOOL_SCHEMA_BYTES)
				}),
			)],
			..GenerationRequest::default()
		};
		assert!(matches!(
			oversized_schema.into_engine(&defaults(), false, false),
			Err(Error::InvalidRequest(message)) if message.contains("schema exceeds")
		));
	}

	#[test]
	fn rejects_oversized_tool_protocol_fields_before_engine_conversion() {
		let declaration =
			ToolDefinition::new("lookup", "Lookup", serde_json::json!({"type": "object"}));
		let oversized_arguments = ToolCall {
			id: "call-1".to_string(),
			name: "lookup".to_string(),
			arguments: serde_json::json!({"value": "x".repeat(MAX_TOOL_ARGUMENT_BYTES)}),
		};
		let request = GenerationRequest {
			messages: vec![
				Message::user("lookup"),
				Message {
					role: Role::Assistant,
					content: Vec::new(),
					tool_calls: vec![oversized_arguments],
					..Message::default()
				},
				Message::tool("call-1", "done"),
			],
			tools: vec![declaration.clone()],
			..GenerationRequest::default()
		};
		assert!(matches!(
			request.into_engine(&defaults(), false, false),
			Err(Error::InvalidRequest(message)) if message.contains("arguments exceed")
		));

		let oversized_id = "x".repeat(MAX_TOOL_CALL_ID_BYTES + 1);
		let request = GenerationRequest {
			messages: vec![
				Message::user("lookup"),
				Message {
					role: Role::Assistant,
					content: Vec::new(),
					tool_calls: vec![ToolCall {
						id: oversized_id.clone(),
						name: "lookup".to_string(),
						arguments: serde_json::json!({}),
					}],
					..Message::default()
				},
				Message::tool(oversized_id, "done"),
			],
			tools: vec![declaration],
			..GenerationRequest::default()
		};
		assert!(matches!(
			request.into_engine(&defaults(), false, false),
			Err(Error::InvalidRequest(message)) if message.contains("tool call ID")
		));
	}

	#[test]
	fn rejects_reasoning_and_json_beyond_structural_bounds() {
		let request = GenerationRequest {
			messages: vec![
				Message::user("hello"),
				Message {
					role: Role::Assistant,
					content: vec![Content::Text("answer".to_string())],
					reasoning: Some("x".repeat(MAX_REASONING_BYTES + 1)),
					..Message::default()
				},
			],
			..GenerationRequest::default()
		};
		assert!(matches!(
			request.into_engine(&defaults(), false, false),
			Err(Error::InvalidRequest(message)) if message.contains("reasoning")
		));

		let mut nested = serde_json::Value::Null;
		for _ in 0..=crate::json::MAX_DEPTH {
			nested = serde_json::Value::Array(vec![nested]);
		}
		assert_eq!(bounded_json_len(&nested, usize::MAX), None);
	}

	#[test]
	fn message_content_limit_is_inclusive_and_shared() {
		let exact = vec![Message::user("x".repeat(MAX_MESSAGE_CONTENT_BYTES))];
		assert!(validate_request_shape(&exact, &[]).is_ok());
		drop(exact);

		let oversized = vec![Message::user("x".repeat(MAX_MESSAGE_CONTENT_BYTES + 1))];
		assert!(matches!(
			validate_request_shape(&oversized, &[]),
			Err(Error::InvalidRequest(message)) if message.contains("one message")
		));
	}

	#[test]
	fn bounded_json_length_accepts_exact_limit_and_rejects_next_byte() {
		let exact = serde_json::Value::String("x".repeat(14));
		assert_eq!(bounded_json_len(&exact, 16), Some(16));
		let oversized = serde_json::Value::String("x".repeat(15));
		assert_eq!(bounded_json_len(&oversized, 16), None);
	}

	#[tokio::test]
	async fn explicit_cancel_waits_after_unblocking_full_stream_channel() {
		let (sender, receiver) = tokio::sync::mpsc::channel(1);
		assert!(
			sender
				.try_send(Ok(GenerationEvent::Text("one".to_string())))
				.is_ok()
		);
		let cancelled = Arc::new(AtomicBool::new(false));
		let (completion_sender, completion_receiver) = tokio::sync::oneshot::channel();
		let mut stream =
			GenerationStream::new(receiver, Arc::clone(&cancelled), completion_receiver);
		let (done_sender, done_receiver) = std::sync::mpsc::channel();
		let worker = std::thread::spawn(move || {
			let result = sender.blocking_send(Ok(GenerationEvent::Text("two".to_string())));
			let _ = done_sender.send(result.is_err());
			let _ = completion_sender.send(());
		});

		stream
			.cancel_and_wait()
			.await
			.expect("cancelled worker completion");

		assert!(cancelled.load(Ordering::Relaxed));
		assert!(
			done_receiver
				.recv_timeout(Duration::from_secs(1))
				.expect("cancel must wake blocked sender")
		);
		assert!(worker.join().is_ok());
	}

	#[tokio::test]
	async fn dropped_cancel_wait_can_be_awaited_again() {
		let (_sender, receiver) = tokio::sync::mpsc::channel(1);
		let cancelled = Arc::new(AtomicBool::new(false));
		let (completion_sender, completion_receiver) = tokio::sync::oneshot::channel();
		let mut stream =
			GenerationStream::new(receiver, Arc::clone(&cancelled), completion_receiver);

		{
			let first_wait = stream.cancel_and_wait();
			tokio::pin!(first_wait);
			assert!(
				tokio::time::timeout(Duration::from_millis(10), &mut first_wait)
					.await
					.is_err()
			);
		}
		assert!(cancelled.load(Ordering::Relaxed));

		let second_wait = stream.cancel_and_wait();
		tokio::pin!(second_wait);
		assert!(
			tokio::time::timeout(Duration::from_millis(10), &mut second_wait)
				.await
				.is_err()
		);
		completion_sender.send(()).expect("complete worker");
		second_wait.await.expect("second wait observes completion");
	}
}
