//! rig message → engine `ChatMessage` conversion.

use base64::Engine as _;
use rig_core::{
	OneOrMany,
	completion::Message,
	message::{
		AssistantContent, DocumentSourceKind, ReasoningContent, ToolResult, ToolResultContent,
		UserContent,
	},
};

use super::{
	Capabilities, MAX_SINGLE_TOOL_ARGUMENT_BYTES, MAX_TOOL_CALL_ID_BYTES, bounded_json_len,
	validate_tool_name,
};
use crate::{
	engine::{
		tokenizer::{AudioContent, ChatMessage, ContentPart, ImageContent, VideoContent},
		tools::ToolCall,
	},
	error::Error,
};

const MAX_SINGLE_CONTENT_BYTES: usize = 128 << 20;

/// Convert one rig message, appending the result(s) to `out`. System
/// content is diverted into `system_parts` instead (merged into a single
/// leading system turn by the caller). A user message mixing tool results
/// with other content is split, because tool results are their own
/// `role: "tool"` turn in chat templates.
pub(super) fn push_message(
	out: &mut Vec<ChatMessage>,
	system_parts: &mut Vec<String>,
	message: &Message,
	capabilities: Capabilities,
) -> Result<(), Error> {
	match message {
		Message::System { content } => {
			if !content.trim().is_empty() {
				ensure_bounded(content.len(), "system message")?;
				system_parts.push(content.clone());
			}
		}
		Message::User { content } => push_user_content(out, content, capabilities)?,
		Message::Assistant { content, .. } => out.push(assistant_message(content)?),
	}
	Ok(())
}

fn push_user_content(
	out: &mut Vec<ChatMessage>,
	content: &OneOrMany<UserContent>,
	capabilities: Capabilities,
) -> Result<(), Error> {
	let mut parts: Vec<ContentPart> = Vec::new();
	for item in content.iter() {
		match item {
			UserContent::Text(text) => {
				ensure_bounded(text.text.len(), "text content")?;
				push_text(&mut parts, &text.text);
			}
			UserContent::ToolResult(result) => {
				// Tool results become their own role:"tool" turn; flush any
				// accumulated user parts first to preserve ordering.
				flush_user_parts(out, &mut parts);
				out.push(tool_result_message(result)?);
			}
			UserContent::Image(image) => {
				if !capabilities.images {
					return Err(Error::UnsupportedContent(
						"this model has no vision tower; images are not supported".to_string(),
					));
				}
				parts.push(ContentPart::Image(ImageContent {
					bytes: media_bytes(&image.data, "image")?,
				}));
			}
			UserContent::Audio(audio) => {
				if !capabilities.audio {
					return Err(Error::UnsupportedContent(
						"this model has no audio tower; audio is not supported".to_string(),
					));
				}
				parts.push(ContentPart::Audio(AudioContent {
					bytes: media_bytes(&audio.data, "audio")?,
				}));
			}
			UserContent::Video(video) => {
				// Video frames feed through the vision path.
				if !capabilities.images {
					return Err(Error::UnsupportedContent(
						"this model has no vision tower; video is not supported".to_string(),
					));
				}
				parts.push(ContentPart::Video(VideoContent {
					bytes: media_bytes(&video.data, "video")?,
				}));
			}
			UserContent::Document(document) => {
				push_text(&mut parts, &document_text(&document.data)?);
			}
		}
	}
	flush_user_parts(out, &mut parts);
	Ok(())
}

/// Coalesce adjacent text parts: several chat templates render a
/// multi-part `content` array by keeping only the text parts they
/// expect, and two adjacent `Text` parts (e.g. prompt text plus an
/// attached document) can be silently dropped or misformatted. One
/// merged part is always safe.
fn push_text(parts: &mut Vec<ContentPart>, text: &str) {
	if let Some(ContentPart::Text(existing)) = parts.last_mut() {
		existing.push_str("\n\n");
		existing.push_str(text);
	} else {
		parts.push(ContentPart::Text(text.to_string()));
	}
}

fn flush_user_parts(out: &mut Vec<ChatMessage>, parts: &mut Vec<ContentPart>) {
	if !parts.is_empty() {
		out.push(ChatMessage {
			role: "user".to_string(),
			content: std::mem::take(parts),
			..ChatMessage::default()
		});
	}
}

fn tool_result_message(result: &ToolResult) -> Result<ChatMessage, Error> {
	ensure_identifier(&result.id, "tool result ID")?;
	if let Some(call_id) = &result.call_id {
		ensure_identifier(call_id, "tool result call ID")?;
	}
	let mut text_parts = Vec::new();
	for content in result.content.iter() {
		match content {
			ToolResultContent::Text(text) => text_parts.push(text.text.as_str()),
			ToolResultContent::Image(_) => {
				return Err(Error::UnsupportedContent(format!(
					"tool result {:?} contains image content, but local chat templates \
					 support text tool results only",
					result.id
				)));
			}
		}
	}
	ensure_bounded(joined_len(&text_parts, 1)?, "tool result")?;
	let text = text_parts.join("\n");
	let call_id = result.call_id.clone().unwrap_or_else(|| result.id.clone());
	Ok(ChatMessage::tool_result(call_id, text))
}

fn assistant_message(content: &OneOrMany<AssistantContent>) -> Result<ChatMessage, Error> {
	let mut text_parts: Vec<&str> = Vec::new();
	let mut tool_calls: Vec<ToolCall> = Vec::new();
	let mut reasoning_parts: Vec<&str> = Vec::new();
	for item in content.iter() {
		match item {
			AssistantContent::Text(text) => text_parts.push(&text.text),
			AssistantContent::ToolCall(call) => {
				ensure_identifier(&call.id, "tool call ID")?;
				validate_tool_name(&call.function.name)?;
				if !call.function.arguments.is_object() {
					return Err(Error::InvalidRequest(format!(
						"tool call {:?} arguments must be a JSON object",
						call.id
					)));
				}
				bounded_json_len(
					&call.function.arguments,
					MAX_SINGLE_TOOL_ARGUMENT_BYTES,
					&format!("tool call {:?} arguments", call.id),
				)?;
				tool_calls.push(ToolCall {
					id: call.id.clone(),
					name: call.function.name.clone(),
					arguments: call.function.arguments.clone(),
				});
			}
			AssistantContent::Reasoning(reasoning) => {
				for block in &reasoning.content {
					match block {
						ReasoningContent::Text { text, .. } => {
							reasoning_parts.push(text);
						}
						_ => {
							return Err(Error::UnsupportedContent(
								"non-text reasoning content cannot be represented by local chat \
								 templates"
									.to_string(),
							));
						}
					}
				}
			}
			AssistantContent::Image(_) => {
				return Err(Error::UnsupportedContent(
					"assistant image history cannot be represented by the local engine".to_string(),
				));
			}
		}
	}
	ensure_bounded(joined_len(&text_parts, 0)?, "assistant text")?;
	ensure_bounded(joined_len(&reasoning_parts, 1)?, "assistant reasoning")?;
	let reasoning_content = if reasoning_parts.is_empty() {
		None
	} else {
		Some(reasoning_parts.join("\n"))
	};
	Ok(ChatMessage {
		role: "assistant".to_string(),
		content: vec![ContentPart::Text(text_parts.join(""))],
		tool_calls,
		tool_call_id: None,
		reasoning_content,
	})
}

/// Extract raw media bytes from a rig source. Local inference has no
/// fetcher: only raw bytes and base64 payloads are accepted.
fn media_bytes(data: &DocumentSourceKind, what: &str) -> Result<Vec<u8>, Error> {
	match data {
		DocumentSourceKind::Raw(bytes) => {
			ensure_bounded(bytes.len(), what)?;
			Ok(bytes.clone())
		}
		DocumentSourceKind::Base64(encoded) => {
			decode_base64_bounded(encoded, what, MAX_SINGLE_CONTENT_BYTES)
		}
		other => Err(Error::UnsupportedContent(format!(
			"{what} source must be raw or base64 bytes for local inference, got \
			 {other:?}"
		))),
	}
}

fn document_text(data: &DocumentSourceKind) -> Result<String, Error> {
	match data {
		DocumentSourceKind::String(text) => {
			ensure_bounded(text.len(), "document")?;
			Ok(text.clone())
		}
		DocumentSourceKind::Raw(bytes) => {
			ensure_bounded(bytes.len(), "document")?;
			String::from_utf8(bytes.clone()).map_err(|_| {
				Error::UnsupportedContent("binary document content is not supported".to_string())
			})
		}
		DocumentSourceKind::Base64(encoded) => {
			let bytes = decode_base64_bounded(encoded, "document", MAX_SINGLE_CONTENT_BYTES)?;
			String::from_utf8(bytes).map_err(|_| {
				Error::UnsupportedContent("binary document content is not supported".to_string())
			})
		}
		other => Err(Error::UnsupportedContent(format!(
			"document source must be text for local inference, got {other:?}"
		))),
	}
}

fn decode_base64_bounded(encoded: &str, what: &str, limit: usize) -> Result<Vec<u8>, Error> {
	if base64::decoded_len_estimate(encoded.len()) > limit {
		return Err(Error::InvalidRequest(format!(
			"{what} base64 payload exceeds {limit} decoded bytes"
		)));
	}
	let decoded = base64::engine::general_purpose::STANDARD
		.decode(encoded)
		.map_err(|error| Error::UnsupportedContent(format!("invalid base64 {what}: {error}")))?;
	if decoded.len() > limit {
		return Err(Error::InvalidRequest(format!(
			"{what} payload exceeds {limit} bytes"
		)));
	}
	Ok(decoded)
}

fn joined_len(parts: &[&str], separator_len: usize) -> Result<usize, Error> {
	let content = parts.iter().try_fold(0_usize, |total, part| {
		total
			.checked_add(part.len())
			.ok_or_else(|| Error::InvalidRequest("content size overflow".to_string()))
	})?;
	let separators = parts
		.len()
		.saturating_sub(1)
		.checked_mul(separator_len)
		.ok_or_else(|| Error::InvalidRequest("content size overflow".to_string()))?;
	content
		.checked_add(separators)
		.ok_or_else(|| Error::InvalidRequest("content size overflow".to_string()))
}

fn ensure_bounded(length: usize, what: &str) -> Result<(), Error> {
	if length > MAX_SINGLE_CONTENT_BYTES {
		return Err(Error::InvalidRequest(format!(
			"{what} cannot exceed 128 MiB"
		)));
	}
	Ok(())
}

fn ensure_identifier(value: &str, what: &str) -> Result<(), Error> {
	if value.trim().is_empty() {
		return Err(Error::InvalidRequest(format!("{what} cannot be empty")));
	}
	if value.len() > MAX_TOOL_CALL_ID_BYTES {
		return Err(Error::InvalidRequest(format!(
			"{what} cannot exceed {MAX_TOOL_CALL_ID_BYTES} bytes"
		)));
	}
	Ok(())
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn base64_preflight_rejects_before_large_decode_allocation() {
		let error = decode_base64_bounded("aGVsbG8=", "image", 4)
			.expect_err("five decoded bytes exceed the test limit");
		assert!(matches!(error, Error::InvalidRequest(_)));
	}
}
