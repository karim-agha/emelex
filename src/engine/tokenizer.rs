//! Tokenizer + chat-template handling, matching the Hugging Face
//! `tokenizer.json`, tokenizer/processor configuration, and optional
//! `chat_template.jinja` shipped alongside MLX checkpoints.

use std::{io::Write as _, path::Path};

use chrono::{DateTime, FixedOffset, Local};
use minijinja::{Environment, value::Value as JinjaValue};
use serde::Serialize;
use serde_json::Value;
use tokenizers::Tokenizer as HfTokenizer;

use crate::engine::{
	error::{Error, Result},
	tools::{Tool, ToolCall},
};

pub(crate) const MAX_CHAT_TEMPLATE_BYTES: usize = 1 << 20;
pub(crate) const LEGACY_CHAT_TEMPLATE_FILE: &str = "chat_template.json";
pub(crate) const CURRENT_CHAT_TEMPLATE_DIR: &str = "additional_chat_templates";
pub(crate) const LEGACY_CHAT_TEMPLATE_DIR: &str = "chat_templates";
const MAX_RENDERED_PROMPT_BYTES: usize = 16 << 20;
const CHAT_TEMPLATE_FUEL: u64 = 2_000_000;
const MAX_NAMED_CHAT_TEMPLATES: usize = 32;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ChatTemplateCapabilities {
	/// The template renders ordinary plain-string chat turns.
	pub chat: bool,
	/// The template renders structured translation messages
	/// (TranslateGemma-style single-mapping content with language codes).
	pub translation: bool,
	pub system_prompt: bool,
	pub tools: bool,
	pub reasoning_history: bool,
	pub thinking_toggle: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ChatTemplates {
	default: String,
	tool_use: Option<String>,
}

impl ChatTemplates {
	pub(crate) fn single(template: String) -> Self {
		Self {
			default: template,
			tool_use: None,
		}
	}

	pub(crate) fn with_tool_use(default: String, tool_use: String) -> Self {
		Self {
			default,
			tool_use: Some(tool_use),
		}
	}

	pub(crate) fn replace_tool_use(mut self, tool_use: Option<String>) -> Self {
		if tool_use.is_some() {
			self.tool_use = tool_use;
		}
		self
	}

	pub(crate) fn into_parts(self) -> (String, Option<String>) {
		(self.default, self.tool_use)
	}

	fn selected(&self, has_tools: bool) -> &str {
		if has_tools {
			self.tool_use.as_deref().unwrap_or(&self.default)
		} else {
			&self.default
		}
	}
}

pub(crate) fn legacy_chat_templates_from_value(value: &Value) -> Result<ChatTemplates> {
	let object = value.as_object().ok_or_else(|| {
		Error::Template("chat_template.json must contain a JSON object".to_string())
	})?;
	if object.len() != 1 {
		return Err(Error::Template(
			"chat_template.json must contain only `chat_template`".to_string(),
		));
	}
	let embedded = object
		.get("chat_template")
		.ok_or_else(|| Error::Template("chat_template.json lacks `chat_template`".to_string()))?;
	chat_templates_from_value(embedded)?
		.ok_or_else(|| Error::Template("chat_template.json chat_template is null".to_string()))
}

pub(crate) fn resolve_chat_template_artifacts(
	processor_embedded: &Value,
	legacy_file: Option<&Value>,
	standalone_default: Option<String>,
	standalone_tool: Option<String>,
	tokenizer_embedded: &Value,
) -> Result<Option<ChatTemplates>> {
	if let Some(processor) = chat_templates_from_value(processor_embedded)? {
		return Ok(Some(processor));
	}
	if let Some(legacy) = legacy_file {
		return legacy_chat_templates_from_value(legacy).map(Some);
	}
	let base = match standalone_default {
		Some(default) => Some(ChatTemplates::single(default)),
		None => chat_templates_from_value(tokenizer_embedded)?,
	};
	Ok(base.map(|templates| templates.replace_tool_use(standalone_tool)))
}

pub(crate) fn chat_templates_from_value(value: &Value) -> Result<Option<ChatTemplates>> {
	match value {
		Value::Null => Ok(None),
		Value::String(template) => Ok(Some(ChatTemplates::single(template.clone()))),
		Value::Object(templates) => {
			let mut parsed = std::collections::BTreeMap::new();
			for (name, value) in templates {
				let template = value.as_str().ok_or_else(|| {
					Error::Template(format!("chat template {name:?} must be a string"))
				})?;
				validate_chat_template_name(name)?;
				parsed.insert(name.clone(), template.to_string());
			}
			chat_templates_from_map(parsed)
		}
		Value::Array(entries) => {
			if entries.len() > MAX_NAMED_CHAT_TEMPLATES {
				return Err(Error::Template(format!(
					"chat template list accepts at most {MAX_NAMED_CHAT_TEMPLATES} entries"
				)));
			}
			let mut parsed = std::collections::BTreeMap::new();
			for entry in entries {
				let object = entry.as_object().ok_or_else(|| {
					Error::Template(
						"chat template list entries must be {name, template} objects".to_string(),
					)
				})?;
				if object.len() != 2 {
					return Err(Error::Template(
						"chat template list entries must contain only `name` and `template`"
							.to_string(),
					));
				}
				let name = object.get("name").and_then(Value::as_str).ok_or_else(|| {
					Error::Template("chat template list entry needs string `name`".to_string())
				})?;
				let template = object
					.get("template")
					.and_then(Value::as_str)
					.ok_or_else(|| {
						Error::Template(
							"chat template list entry needs string `template`".to_string(),
						)
					})?;
				validate_chat_template_name(name)?;
				if parsed
					.insert(name.to_string(), template.to_string())
					.is_some()
				{
					return Err(Error::Template(format!(
						"duplicate chat template name {name:?}"
					)));
				}
			}
			chat_templates_from_map(parsed)
		}
		_ => Err(Error::Template(
			"chat_template must be a string, dictionary of strings, or named template list"
				.to_string(),
		)),
	}
}

fn validate_chat_template_name(name: &str) -> Result<()> {
	if name.is_empty()
		|| name.len() > 128
		|| name
			.bytes()
			.any(|byte| byte.is_ascii_control() || byte == b'/')
	{
		return Err(Error::Template(format!(
			"invalid chat template name {name:?}"
		)));
	}
	Ok(())
}

fn chat_templates_from_map(
	mut templates: std::collections::BTreeMap<String, String>,
) -> Result<Option<ChatTemplates>> {
	if templates.len() > MAX_NAMED_CHAT_TEMPLATES {
		return Err(Error::Template(format!(
			"chat template dictionary accepts at most {MAX_NAMED_CHAT_TEMPLATES} entries"
		)));
	}
	let tool_use = templates.get("tool_use").cloned();
	let default = templates
		.remove("default")
		.or_else(|| {
			(templates.len() == 1)
				.then(|| templates.values().next().cloned())
				.flatten()
		})
		.ok_or_else(|| {
			Error::Template("multiple named chat templates require a `default` entry".to_string())
		})?;
	Ok(Some(ChatTemplates { default, tool_use }))
}

/// One part of a (possibly multi-modal) chat message's content.
#[derive(Debug, Clone)]
pub enum ContentPart {
	Text(String),
	Image(ImageContent),
	Audio(AudioContent),
	Video(VideoContent),
	Translation(TranslationContent),
}

/// A structured translation request (TranslateGemma-style templates):
/// translate `text` from `source_lang` to `target_lang` (BCP-47-style
/// codes, e.g. "en", "pt-BR"). Must be the sole content part of a user
/// message — the template contract is exactly one mapping per turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranslationContent {
	pub source_lang: String,
	pub target_lang: String,
	pub text: String,
}

/// A single image attachment (raw encoded bytes - JPEG/PNG/...; decoded and
/// preprocessed later by `crate::engine::media::image`).
#[derive(Debug, Clone)]
pub struct ImageContent {
	pub bytes: Vec<u8>,
}

/// A single audio attachment (bounded RIFF/WAVE PCM16 or float32 bytes;
/// decoded and preprocessed later by `crate::engine::media::audio`).
#[derive(Debug, Clone)]
pub struct AudioContent {
	pub bytes: Vec<u8>,
}

/// A single video attachment (raw encoded bytes - MP4/WebM/...; frames
/// extracted later by `crate::engine::media::video` and fed through the image
/// path).
#[derive(Debug, Clone)]
pub struct VideoContent {
	pub bytes: Vec<u8>,
}

/// One turn of a chat conversation.
///
/// `content` is a list of parts (text and/or media) rather than a plain
/// string, so a single turn can interleave free text with images (and,
/// eventually, audio/video). The common text-only case stays ergonomic via
/// [`ChatMessage::user`]/[`ChatMessage::system`]/[`ChatMessage::assistant`]
/// (each produces a single `ContentPart::Text`) and the [`ChatMessage::text`]
/// accessor (concatenates every `Text` part, ignoring media).
#[derive(Debug, Clone, Default)]
pub struct ChatMessage {
	pub role: String,
	pub content: Vec<ContentPart>,
	/// Set on assistant turns that contain tool calls (rendered into the
	/// template's `message.tool_calls`).
	pub tool_calls: Vec<ToolCall>,
	/// Set on `role: "tool"` turns: which call this result answers.
	pub tool_call_id: Option<String>,
	/// Set on assistant turns whose reasoning/"thinking" span (see
	/// `crate::engine::reasoning`) should round-trip back into multi-turn
	/// history (rendered into the template's `message.reasoning_content`,
	/// matched by Gemma4/Qwen-style templates that special-case it).
	pub reasoning_content: Option<String>,
}

impl ChatMessage {
	pub fn user(content: impl Into<String>) -> Self {
		ChatMessage {
			role: "user".into(),
			content: vec![ContentPart::Text(content.into())],
			..Default::default()
		}
	}

	/// A user turn carrying one structured translation request — the sole
	/// content shape TranslateGemma-style templates accept.
	pub fn user_translation(
		source_lang: impl Into<String>,
		target_lang: impl Into<String>,
		text: impl Into<String>,
	) -> Self {
		ChatMessage {
			role: "user".into(),
			content: vec![ContentPart::Translation(TranslationContent {
				source_lang: source_lang.into(),
				target_lang: target_lang.into(),
				text: text.into(),
			})],
			..Default::default()
		}
	}

	pub fn system(content: impl Into<String>) -> Self {
		ChatMessage {
			role: "system".into(),
			content: vec![ContentPart::Text(content.into())],
			..Default::default()
		}
	}

	pub fn assistant(content: impl Into<String>) -> Self {
		ChatMessage {
			role: "assistant".into(),
			content: vec![ContentPart::Text(content.into())],
			..Default::default()
		}
	}

	pub fn assistant_with_tool_calls(
		content: impl Into<String>,
		tool_calls: Vec<ToolCall>,
	) -> Self {
		ChatMessage {
			role: "assistant".into(),
			content: vec![ContentPart::Text(content.into())],
			tool_calls,
			..Default::default()
		}
	}

	/// An assistant turn carrying its reasoning/"thinking" content
	/// alongside the final answer, so it round-trips into the next turn's
	/// history exactly as [`Session::generate_cached`] split it out (see
	/// `crate::engine::reasoning::split_reasoning`).
	pub fn assistant_with_reasoning(
		content: impl Into<String>,
		reasoning_content: impl Into<String>,
	) -> Self {
		ChatMessage {
			role: "assistant".into(),
			content: vec![ContentPart::Text(content.into())],
			reasoning_content: Some(reasoning_content.into()),
			..Default::default()
		}
	}

	pub fn tool_result(call_id: impl Into<String>, content: impl Into<String>) -> Self {
		ChatMessage {
			role: "tool".into(),
			content: vec![ContentPart::Text(content.into())],
			tool_call_id: Some(call_id.into()),
			..Default::default()
		}
	}

	/// A user turn pairing free text with one image (raw encoded bytes,
	/// e.g. a `.jpg`/`.png` file's contents). The chat template renders
	/// this as a single `<|image|>`-style placeholder that
	/// `Session::encode_chat_with_media` later expands into the model's
	/// `boi + image_token × num_soft_tokens + eoi` span.
	pub fn user_with_image(text: impl Into<String>, image_bytes: impl Into<Vec<u8>>) -> Self {
		ChatMessage {
			role: "user".into(),
			content: vec![
				ContentPart::Text(text.into()),
				ContentPart::Image(ImageContent {
					bytes: image_bytes.into(),
				}),
			],
			..Default::default()
		}
	}

	/// A user turn pairing free text with one audio clip (raw encoded
	/// bytes, e.g. a `.wav`/`.mp3` file's contents). The chat template
	/// renders this as a single `<|audio|>`-style placeholder that
	/// `Session::encode_chat_with_media` later expands into the model's
	/// `boa + audio_token × num_soft_tokens + eoa` span.
	pub fn user_with_audio(text: impl Into<String>, audio_bytes: impl Into<Vec<u8>>) -> Self {
		ChatMessage {
			role: "user".into(),
			content: vec![
				ContentPart::Text(text.into()),
				ContentPart::Audio(AudioContent {
					bytes: audio_bytes.into(),
				}),
			],
			..Default::default()
		}
	}

	/// A user turn pairing free text with one video clip (raw encoded
	/// bytes, e.g. an `.mp4` file's contents). The chat template renders
	/// this as a single `<|video|>`-style placeholder that
	/// `Session::encode_chat_with_media` later expands into one
	/// `boi + image_token × N + eoi` span per sampled frame.
	pub fn user_with_video(text: impl Into<String>, video_bytes: impl Into<Vec<u8>>) -> Self {
		ChatMessage {
			role: "user".into(),
			content: vec![
				ContentPart::Text(text.into()),
				ContentPart::Video(VideoContent {
					bytes: video_bytes.into(),
				}),
			],
			..Default::default()
		}
	}

	/// Concatenation of every `Text` part (media parts contribute nothing);
	/// the common accessor for the plain-text case.
	pub fn text(&self) -> String {
		self.content
			.iter()
			.filter_map(|p| match p {
				ContentPart::Text(t) => Some(t.as_str()),
				_ => None,
			})
			.collect::<Vec<_>>()
			.concat()
	}

	/// Every image attached to this message, in order.
	pub fn images(&self) -> impl Iterator<Item = &ImageContent> {
		self.content.iter().filter_map(|p| match p {
			ContentPart::Image(i) => Some(i),
			_ => None,
		})
	}

	/// Every audio clip attached to this message, in order.
	pub fn audios(&self) -> impl Iterator<Item = &AudioContent> {
		self.content.iter().filter_map(|p| match p {
			ContentPart::Audio(a) => Some(a),
			_ => None,
		})
	}

	/// Every video clip attached to this message, in order.
	pub fn videos(&self) -> impl Iterator<Item = &VideoContent> {
		self.content.iter().filter_map(|p| match p {
			ContentPart::Video(v) => Some(v),
			_ => None,
		})
	}

	/// True if this message carries any media content part. Translation
	/// parts are structured text, not media — they must not trigger the
	/// media preprocessing or byte-budget machinery.
	pub fn has_media(&self) -> bool {
		self.content.iter().any(|p| {
			matches!(
				p,
				ContentPart::Image(_) | ContentPart::Audio(_) | ContentPart::Video(_)
			)
		})
	}
}

/// Render `content` the way the Jinja context expects: a plain string when
/// it's a single text part (matches every existing text-only template's
/// `message['content'] is string` branch byte-for-byte), otherwise a
/// sequence of `{"type": ..., ...}` part objects (matches the
/// `message['content'] is sequence` branch multimodal templates use).
fn content_to_json(content: &[ContentPart]) -> Value {
	if let [ContentPart::Text(t)] = content {
		return Value::String(t.clone());
	}
	Value::Array(
		content
			.iter()
			.map(|p| match p {
				ContentPart::Text(t) => serde_json::json!({"type": "text", "text": t}),
				ContentPart::Image(_) => serde_json::json!({"type": "image"}),
				ContentPart::Audio(_) => serde_json::json!({"type": "audio"}),
				ContentPart::Video(_) => serde_json::json!({"type": "video"}),
				// TranslateGemma-style contract: one mapping whose `type`
				// is the payload kind ("text"), carrying the language pair.
				ContentPart::Translation(t) => serde_json::json!({
					"type": "text",
					"source_lang_code": t.source_lang,
					"target_lang_code": t.target_lang,
					"text": t.text,
				}),
			})
			.collect(),
	)
}

pub struct Tokenizer {
	inner: HfTokenizer,
	chat_template: Option<String>,
	tool_chat_template: Option<String>,
	bos_token: Option<String>,
	eos_token: Option<String>,
	eos_token_ids: Vec<u32>,
	/// emelex patch: compiled chat-template environment, built once on
	/// first render - upstream re-parsed the template and rebuilt the
	/// minijinja Environment on every call (twice per generate_cached).
	template_env: std::sync::OnceLock<Environment<'static>>,
	tool_template_env: std::sync::OnceLock<Environment<'static>>,
}

impl Tokenizer {
	pub fn load(model_dir: &Path) -> Result<Self> {
		let tokenizer_path = model_dir.join("tokenizer.json");
		let tokenizer_bytes =
			crate::artifact::read_bytes(&tokenizer_path, crate::artifact::MAX_TOKENIZER_BYTES)
				.map_err(|error| {
					Error::Tokenizer(format!(
						"cannot safely read {}: {error}",
						tokenizer_path.display()
					))
				})?;
		let tokenizer_config_path = model_dir.join("tokenizer_config.json");
		let processor_config_path = model_dir.join("processor_config.json");
		let chat_template_path = model_dir.join("chat_template.jinja");
		let tool_chat_template_path = model_dir.join("chat_template_tool_use.jinja");
		let legacy_chat_template_path = model_dir.join(LEGACY_CHAT_TEMPLATE_FILE);
		let tokenizer_config_bytes = crate::artifact::read_optional_bytes(
			&tokenizer_config_path,
			crate::artifact::MAX_TOKENIZER_CONFIG_BYTES,
		)
		.map_err(|error| {
			Error::Tokenizer(format!(
				"cannot safely read {}: {error}",
				tokenizer_config_path.display()
			))
		})?;
		let processor_config_bytes = crate::artifact::read_optional_bytes(
			&processor_config_path,
			crate::artifact::MAX_TOKENIZER_CONFIG_BYTES,
		)
		.map_err(|error| {
			Error::Tokenizer(format!(
				"cannot safely read {}: {error}",
				processor_config_path.display()
			))
		})?;
		let mut chat_template_bytes = crate::artifact::read_optional_bytes(
			&chat_template_path,
			crate::artifact::MAX_CHAT_TEMPLATE_BYTES,
		)
		.map_err(|error| {
			Error::Tokenizer(format!(
				"cannot safely read {}: {error}",
				chat_template_path.display()
			))
		})?;
		let mut tool_chat_template_bytes = crate::artifact::read_optional_bytes(
			&tool_chat_template_path,
			crate::artifact::MAX_CHAT_TEMPLATE_BYTES,
		)
		.map_err(|error| {
			Error::Tokenizer(format!(
				"cannot safely read {}: {error}",
				tool_chat_template_path.display()
			))
		})?;
		let legacy_chat_template_bytes = crate::artifact::read_optional_bytes(
			&legacy_chat_template_path,
			crate::artifact::MAX_TOKENIZER_CONFIG_BYTES,
		)
		.map_err(|error| {
			Error::Tokenizer(format!(
				"cannot safely read {}: {error}",
				legacy_chat_template_path.display()
			))
		})?;
		let mut found_named = false;
		for directory in [CURRENT_CHAT_TEMPLATE_DIR, LEGACY_CHAT_TEMPLATE_DIR] {
			let default_path = model_dir.join(directory).join("default.jinja");
			let named_default = crate::artifact::read_optional_bytes(
				&default_path,
				crate::artifact::MAX_CHAT_TEMPLATE_BYTES,
			)
			.map_err(|error| {
				Error::Tokenizer(format!(
					"cannot safely read {}: {error}",
					default_path.display()
				))
			})?;
			if let Some(named_default) = named_default {
				found_named = true;
				if chat_template_bytes.replace(named_default).is_some() {
					return Err(Error::Template(
						"multiple default chat template files are present".to_string(),
					));
				}
			}
			let tool_path = model_dir.join(directory).join("tool_use.jinja");
			let named_tool = crate::artifact::read_optional_bytes(
				&tool_path,
				crate::artifact::MAX_CHAT_TEMPLATE_BYTES,
			)
			.map_err(|error| {
				Error::Tokenizer(format!(
					"cannot safely read {}: {error}",
					tool_path.display()
				))
			})?;
			if let Some(named_tool) = named_tool {
				found_named = true;
				if tool_chat_template_bytes.replace(named_tool).is_some() {
					return Err(Error::Template(
						"multiple tool-use chat template files are present".to_string(),
					));
				}
			}
		}
		if legacy_chat_template_bytes.is_some() && found_named {
			return Err(Error::Template(
				"chat_template.json conflicts with named chat template files".to_string(),
			));
		}
		Self::from_artifacts(
			&tokenizer_bytes,
			tokenizer_config_bytes.as_deref(),
			processor_config_bytes.as_deref(),
			legacy_chat_template_bytes.as_deref(),
			chat_template_bytes.as_deref(),
			tool_chat_template_bytes.as_deref(),
		)
	}

	pub(crate) fn load_snapshot(
		snapshot: &crate::model::layout::CheckpointSnapshot,
	) -> Result<Self> {
		let tokenizer = snapshot.runtime_metadata("tokenizer.json").ok_or_else(|| {
			Error::Tokenizer("descriptor-backed snapshot has no tokenizer.json".to_string())
		})?;
		Self::from_artifacts(
			tokenizer,
			snapshot.runtime_metadata("tokenizer_config.json"),
			snapshot.runtime_metadata("processor_config.json"),
			snapshot.runtime_metadata(LEGACY_CHAT_TEMPLATE_FILE),
			snapshot.runtime_metadata("chat_template.jinja"),
			snapshot.runtime_metadata("chat_template_tool_use.jinja"),
		)
	}

	fn from_artifacts(
		tokenizer_bytes: &[u8],
		tokenizer_config_bytes: Option<&[u8]>,
		processor_config_bytes: Option<&[u8]>,
		legacy_chat_template_bytes: Option<&[u8]>,
		chat_template_bytes: Option<&[u8]>,
		tool_chat_template_bytes: Option<&[u8]>,
	) -> Result<Self> {
		let inner = HfTokenizer::from_bytes(tokenizer_bytes)
			.map_err(|error| Error::Tokenizer(format!("failed to load tokenizer.json: {error}")))?;
		let tokenizer_config: Value = tokenizer_config_bytes
			.map(serde_json::from_slice)
			.transpose()
			.map_err(|error| Error::Tokenizer(format!("bad tokenizer_config.json: {error}")))?
			.unwrap_or(Value::Null);
		let processor_config: Value = processor_config_bytes
			.map(serde_json::from_slice)
			.transpose()
			.map_err(|error| Error::Tokenizer(format!("bad processor_config.json: {error}")))?
			.unwrap_or(Value::Null);
		let legacy_chat_template: Option<Value> = legacy_chat_template_bytes
			.map(serde_json::from_slice)
			.transpose()
			.map_err(|error| Error::Tokenizer(format!("bad chat_template.json: {error}")))?;
		let standalone_default = chat_template_bytes
			.map(|bytes| {
				std::str::from_utf8(bytes)
					.map(str::to_string)
					.map_err(|error| {
						Error::Tokenizer(format!("chat_template.jinja is not UTF-8: {error}"))
					})
			})
			.transpose()?;
		let standalone_tool = tool_chat_template_bytes
			.map(|bytes| {
				std::str::from_utf8(bytes)
					.map(str::to_string)
					.map_err(|error| {
						Error::Tokenizer(format!(
							"chat_template_tool_use.jinja is not UTF-8: {error}"
						))
					})
			})
			.transpose()?;
		let chat_templates = resolve_chat_template_artifacts(
			processor_config
				.get("chat_template")
				.unwrap_or(&Value::Null),
			legacy_chat_template.as_ref(),
			standalone_default,
			standalone_tool,
			tokenizer_config
				.get("chat_template")
				.unwrap_or(&Value::Null),
		)?;
		if chat_templates.as_ref().is_some_and(|templates| {
			templates.default.len() > MAX_CHAT_TEMPLATE_BYTES
				|| templates
					.tool_use
					.as_ref()
					.is_some_and(|template| template.len() > MAX_CHAT_TEMPLATE_BYTES)
		}) {
			return Err(Error::Template(format!(
				"chat template exceeds {MAX_CHAT_TEMPLATE_BYTES} byte limit"
			)));
		}

		let bos_token = extract_special_token(&tokenizer_config, "bos_token");
		let eos_token = extract_special_token(&tokenizer_config, "eos_token");

		let mut eos_token_ids = Vec::new();
		if let Some(t) = &eos_token {
			if let Some(id) = inner.token_to_id(t) {
				eos_token_ids.push(id);
			}
		}
		// generation_config.json / config.json commonly list eos_token_id
		// as a scalar or list; callers can extend this via `add_eos_id`.
		//
		// Gemma-family templates terminate every turn with the dedicated
		// `<end_of_turn>` token; instruction-tuned checkpoints emit it, not
		// `<eos>`. Conversions with sparse metadata (no tokenizer_config
		// eos_token, no numeric eos_token_id anywhere — some TranslateGemma
		// exports) would otherwise never stop generating. The template
		// using the marker is the evidence the checkpoint stops with it.
		if chat_templates
			.as_ref()
			.is_some_and(|templates| templates.default.contains("<end_of_turn>"))
			&& let Some(id) = inner.token_to_id("<end_of_turn>")
			&& !eos_token_ids.contains(&id)
		{
			eos_token_ids.push(id);
		}

		Ok(Tokenizer {
			inner,
			chat_template: chat_templates
				.as_ref()
				.map(|templates| templates.default.clone()),
			tool_chat_template: chat_templates.and_then(|templates| templates.tool_use),
			bos_token,
			eos_token,
			eos_token_ids,
			template_env: std::sync::OnceLock::new(),
			tool_template_env: std::sync::OnceLock::new(),
		})
	}

	pub fn add_eos_id(&mut self, id: u32) {
		if !self.eos_token_ids.contains(&id) {
			self.eos_token_ids.push(id);
		}
	}

	pub fn eos_token_ids(&self) -> &[u32] {
		&self.eos_token_ids
	}

	pub(crate) fn resolved_chat_template_capabilities(
		&self,
	) -> Result<(
		ChatTemplateCapabilities,
		crate::engine::tools::ToolCallFormat,
	)> {
		let Some(default) = self.chat_template.clone() else {
			return Ok((
				ChatTemplateCapabilities::default(),
				crate::engine::tools::ToolCallFormat::None,
			));
		};
		let templates = ChatTemplates {
			default,
			tool_use: self.tool_chat_template.clone(),
		};
		resolve_chat_templates_capabilities(
			&templates,
			(
				self.bos_token.as_deref().unwrap_or_default(),
				self.eos_token.as_deref().unwrap_or_default(),
			),
		)
	}

	/// The language table embedded in the default chat template, when the
	/// template ships one (TranslateGemma-style `set languages = {...}`).
	/// `None` when there is no template or no recognizable table.
	pub(crate) fn translation_language_table(
		&self,
	) -> Option<std::collections::BTreeMap<String, String>> {
		self.chat_template
			.as_deref()
			.and_then(translation_language_table)
	}

	pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
		let enc = self
			.inner
			.encode(text, false)
			.map_err(|e| Error::Tokenizer(e.to_string()))?;
		Ok(enc.get_ids().to_vec())
	}

	pub fn decode(&self, ids: &[u32]) -> Result<String> {
		self.inner
			.decode(ids, true)
			.map_err(|e| Error::Tokenizer(e.to_string()))
	}

	/// Decode a single new token, useful for incremental streaming
	/// (does not attempt to merge partial multi-byte UTF-8 sequences).
	pub fn decode_piece(&self, id: u32) -> Result<String> {
		self.decode(&[id])
	}

	/// Same as [`Tokenizer::decode`], but keeps special tokens as literal
	/// text instead of stripping them.
	///
	/// Some chat templates render reasoning/tool-call delimiters (e.g.
	/// Gemma4's `<|channel>`/`<channel|>`, `<|tool_call>`/`<tool_call|>`)
	/// as *special* vocabulary entries (`added_tokens_decoder[...].special
	/// == true`), which [`Tokenizer::decode`]'s `skip_special_tokens`
	/// unconditionally strips - silently deleting the very markers
	/// `crate::engine::reasoning`/`crate::engine::streaming`/
	/// `crate::engine::tools` need to see to detect a reasoning or tool-call
	/// span at all (other families, e.g. Qwen's `<think>`/`<tool_call>`, ship
	/// the same markers as *ordinary*, non-special vocabulary entries instead,
	/// so this distinction is purely a per-checkpoint tokenizer detail, not an
	/// architecture one). This method is what those modules decode
	/// through internally so marker detection works uniformly across
	/// both conventions.
	pub fn decode_raw(&self, ids: &[u32]) -> Result<String> {
		self.inner
			.decode(ids, false)
			.map_err(|e| Error::Tokenizer(e.to_string()))
	}

	/// Same as [`Tokenizer::decode_raw`], for a single token.
	pub fn decode_piece_raw(&self, id: u32) -> Result<String> {
		self.decode_raw(&[id])
	}

	/// Render the chat template for `messages`, returning the prompt text
	/// to feed into [`Tokenizer::encode`].
	pub fn apply_chat_template(
		&self,
		messages: &[ChatMessage],
		add_generation_prompt: bool,
	) -> Result<String> {
		self.apply_chat_template_with_tools(messages, add_generation_prompt, None)
	}

	/// Same as [`Tokenizer::apply_chat_template`], additionally threading a
	/// `tools` list into the jinja context (rendered as the OpenAI-style
	/// `[{"type": "function", "function": {...}}, ...]` shape every
	/// downloaded checkpoint's template expects).
	pub fn apply_chat_template_with_tools(
		&self,
		messages: &[ChatMessage],
		add_generation_prompt: bool,
		tools: Option<&[Tool]>,
	) -> Result<String> {
		self.apply_chat_template_full(messages, add_generation_prompt, tools, None)
	}

	/// Same as [`Tokenizer::apply_chat_template_with_tools`], additionally
	/// threading `enable_thinking` into the jinja context - the variable
	/// Qwen3/3.5/3.6, Gemma4, MiniCPM5, and NemotronH templates check to
	/// decide whether to open a reasoning span. `None` omits the key
	/// entirely, leaving it `Undefined` (the template's own default).
	/// Note that several of these templates only special-case
	/// `enable_thinking` when it's explicitly `false` (forcing a
	/// pre-closed `<think></think>`) and otherwise open a reasoning span
	/// unprompted - i.e. `Undefined` does *not* reliably mean "off" here.
	/// Callers that want reasoning disabled by default should pass an
	/// explicit `Some(false)` rather than relying on `None`; see
	/// [`crate::engine::generate::Session::encode_chat_with_media_full`], which
	/// does exactly that.
	pub fn apply_chat_template_full(
		&self,
		messages: &[ChatMessage],
		add_generation_prompt: bool,
		tools: Option<&[Tool]>,
		enable_thinking: Option<bool>,
	) -> Result<String> {
		self.apply_chat_template_full_for_format(
			messages,
			add_generation_prompt,
			tools,
			enable_thinking,
			crate::engine::tools::ToolCallFormat::None,
		)
	}

	pub(crate) fn apply_chat_template_full_for_format(
		&self,
		messages: &[ChatMessage],
		add_generation_prompt: bool,
		tools: Option<&[Tool]>,
		enable_thinking: Option<bool>,
		tool_format: crate::engine::tools::ToolCallFormat,
	) -> Result<String> {
		let use_tool_template =
			tools.is_some_and(|tools| !tools.is_empty()) && self.tool_chat_template.is_some();
		let template_env = if use_tool_template {
			&self.tool_template_env
		} else {
			&self.template_env
		};
		// emelex patch: build the environment once and cache it; only the
		// per-call context varies.
		if template_env.get().is_none() {
			let template_src = if use_tool_template {
				self.tool_chat_template.as_deref()
			} else {
				self.chat_template.as_deref()
			}
			.ok_or_else(|| {
				Error::Template("model has no chat_template.jinja / chat_template config".into())
			})?;
			let built = build_template_env(strip_generation_tags(template_src))?;
			let _ = template_env.set(built);
		}
		let env = template_env
			.get()
			.ok_or_else(|| Error::Template("chat template initialization failed".to_string()))?;
		let tools_value = tools.map(|items| {
			items
				.iter()
				.map(JinjaValue::from_serialize)
				.collect::<Vec<_>>()
		});
		render_template(
			env,
			chat_messages_to_jinja_for_format(messages, tool_format)?,
			add_generation_prompt,
			tools_value,
			enable_thinking,
			(
				self.bos_token.as_deref().unwrap_or_default(),
				self.eos_token.as_deref().unwrap_or_default(),
			),
		)
	}
}

fn chat_messages_to_jinja_for_format(
	messages: &[ChatMessage],
	format: crate::engine::tools::ToolCallFormat,
) -> Result<Vec<JinjaValue>> {
	if format != crate::engine::tools::ToolCallFormat::Gemma {
		return Ok(chat_messages_to_jinja(messages));
	}
	gemma_chat_messages_to_jinja(messages)
}

fn chat_messages_to_jinja(messages: &[ChatMessage]) -> Vec<JinjaValue> {
	let tool_names = messages
		.iter()
		.flat_map(|message| {
			message
				.tool_calls
				.iter()
				.map(|call| (call.id.as_str(), call.name.as_str()))
		})
		.collect::<std::collections::BTreeMap<_, _>>();
	messages
		.iter()
		.map(|message| JinjaValue::from_serialize(chat_message_to_json(message, &tool_names)))
		.collect()
}

fn chat_message_to_json(
	message: &ChatMessage,
	tool_names: &std::collections::BTreeMap<&str, &str>,
) -> Value {
	let mut fields = serde_json::Map::new();
	fields.insert("role".to_string(), Value::String(message.role.clone()));
	fields.insert("content".to_string(), content_to_json(&message.content));
	if !message.tool_calls.is_empty() {
		fields.insert(
			"tool_calls".to_string(),
			Value::Array(
				message
					.tool_calls
					.iter()
					.map(|call| {
						serde_json::json!({
								"id": call.id,
								"type": "function",
								"function": {
									"name": call.name,
									"arguments": call.arguments
								},
						})
					})
					.collect(),
			),
		);
	}
	if let Some(tool_call_id) = &message.tool_call_id {
		fields.insert(
			"tool_call_id".to_string(),
			Value::String(tool_call_id.clone()),
		);
		if let Some(name) = tool_names.get(tool_call_id.as_str()) {
			fields.insert("name".to_string(), Value::String((*name).to_string()));
		}
	}
	if let Some(reasoning) = &message.reasoning_content {
		fields.insert(
			"reasoning_content".to_string(),
			Value::String(reasoning.clone()),
		);
	}
	Value::Object(fields)
}

fn gemma_chat_messages_to_jinja(messages: &[ChatMessage]) -> Result<Vec<JinjaValue>> {
	let mut rendered = Vec::with_capacity(messages.len());
	let mut index = 0_usize;
	while index < messages.len() {
		let message = &messages[index];
		if message.role == "tool" {
			return Err(Error::Template(
				"Gemma tool result has no immediately preceding assistant tool call".to_string(),
			));
		}
		let mut value = chat_message_to_json(message, &std::collections::BTreeMap::new());
		if message.role == "assistant" && !message.tool_calls.is_empty() {
			let call_names = message
				.tool_calls
				.iter()
				.map(|call| (call.id.as_str(), call.name.as_str()))
				.collect::<std::collections::BTreeMap<_, _>>();
			let mut answered = std::collections::BTreeSet::new();
			let mut responses = Vec::new();
			while messages
				.get(index + 1)
				.is_some_and(|candidate| candidate.role == "tool")
			{
				index += 1;
				let result = &messages[index];
				let call_id = result.tool_call_id.as_deref().ok_or_else(|| {
					Error::Template("Gemma tool result lacks tool_call_id".to_string())
				})?;
				let expected_call = message.tool_calls.get(responses.len()).ok_or_else(|| {
					Error::Template(
						"Gemma history contains more results than assistant calls".to_string(),
					)
				})?;
				if expected_call.id != call_id {
					return Err(Error::Template(
						"Gemma tool results must follow assistant call order".to_string(),
					));
				}
				let name = call_names.get(call_id).ok_or_else(|| {
					Error::Template(format!(
						"Gemma tool result references unknown call id {call_id:?}"
					))
				})?;
				if !answered.insert(call_id) {
					return Err(Error::Template(format!(
						"Gemma tool call {call_id:?} has duplicate results"
					)));
				}
				let response = match result.content.as_slice() {
					[ContentPart::Text(text)] => text.clone(),
					_ => {
						return Err(Error::Template(
							"Gemma tool result must contain exactly one text part".to_string(),
						));
					}
				};
				responses.push(serde_json::json!({
					"name": **name,
					"response": response,
				}));
			}
			if !responses.is_empty() {
				value
					.as_object_mut()
					.ok_or_else(|| Error::Template("cannot serialize Gemma history".to_string()))?
					.insert("tool_responses".to_string(), Value::Array(responses));
			}
		}
		rendered.push(JinjaValue::from_serialize(value));
		index += 1;
	}
	Ok(rendered)
}

fn render_template(
	env: &Environment<'_>,
	messages: Vec<JinjaValue>,
	add_generation_prompt: bool,
	tools: Option<Vec<JinjaValue>>,
	enable_thinking: Option<bool>,
	special_tokens: (&str, &str),
) -> Result<String> {
	let (bos_token, eos_token) = special_tokens;
	let template = env
		.get_template("chat")
		.map_err(|error| Error::Template(error.to_string()))?;
	// Undefined differs from None in HF templates: omit optional keys when
	// callers did not provide them.
	let base_context = minijinja::context! {
			messages => messages,
			add_generation_prompt => add_generation_prompt,
			bos_token => bos_token,
			eos_token => eos_token,
	};
	let context_with_tools = match tools {
		Some(tools) => minijinja::context! { tools => tools, ..base_context },
		None => base_context,
	};
	let full_context = match enable_thinking {
		Some(value) => {
			minijinja::context! { enable_thinking => value, ..context_with_tools }
		}
		None => context_with_tools,
	};
	let mut rendered = BoundedPrompt::new(MAX_RENDERED_PROMPT_BYTES);
	template
		.render_captured_to(full_context, &mut rendered)
		.map_err(|error| Error::Template(error.to_string()))?;
	rendered.finish()
}

/// emelex patch: one-time construction of the chat-template environment
/// (see `Tokenizer::template_env`). `template_src` is the
/// generation-tag-stripped template source, owned so the environment can
/// be `'static`.
fn build_template_env(template_src: String) -> Result<Environment<'static>> {
	build_template_env_with_clock(template_src, TemplateClock::Live)
}

#[derive(Clone)]
enum TemplateClock {
	Live,
	Fixed(DateTime<FixedOffset>),
}

fn build_template_env_with_clock(
	template_src: String,
	clock: TemplateClock,
) -> Result<Environment<'static>> {
	if template_src.len() > MAX_CHAT_TEMPLATE_BYTES {
		return Err(Error::Template(format!(
			"chat template exceeds {MAX_CHAT_TEMPLATE_BYTES} byte limit"
		)));
	}
	let mut env = Environment::new();
	env.set_lstrip_blocks(true);
	env.set_trim_blocks(true);
	env.set_fuel(Some(CHAT_TEMPLATE_FUEL));
	// HF chat templates lean on Python string/list methods (startswith,
	// strip, join, ...); minijinja-contrib's pycompat shim maps these
	// onto minijinja's native `unknown_method_callback`.
	env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
	env.add_function(
		"raise_exception",
		|msg: String| -> std::result::Result<(), minijinja::Error> {
			Err(minijinja::Error::new(
				minijinja::ErrorKind::InvalidOperation,
				msg,
			))
		},
	);
	env.add_function(
		"strftime_now",
		move |format: String| -> std::result::Result<String, minijinja::Error> {
			let now = match &clock {
				TemplateClock::Live => Local::now().fixed_offset(),
				TemplateClock::Fixed(value) => *value,
			};
			format_python_datetime(now, &format).map_err(template_operation_error)
		},
	);
	env.add_filter(
		"tojson",
		|value: JinjaValue,
		 kwargs: minijinja::value::Kwargs|
		 -> std::result::Result<String, minijinja::Error> {
			let indent = kwargs.get::<Option<usize>>("indent")?;
			let ensure_ascii = kwargs.get::<Option<bool>>("ensure_ascii")?.unwrap_or(false);
			let sort_keys = kwargs.get::<Option<bool>>("sort_keys")?.unwrap_or(false);
			let separators = kwargs.get::<Option<Vec<String>>>("separators")?;
			kwargs.assert_all_used()?;
			if indent.is_some_and(|value| value > 16) {
				return Err(template_operation_error(
					"tojson indent must be at most 16".to_string(),
				));
			}
			let separator_pair = json_separator_pair(separators, indent.is_some())
				.map_err(template_operation_error)?;
			let json = if sort_keys {
				let compact =
					serialize_bounded_json(&value, None).map_err(template_operation_error)?;
				let sorted = serde_json::from_str::<Value>(&compact)
					.map_err(|error| template_operation_error(error.to_string()))?;
				serialize_bounded_json(&sorted, indent).map_err(template_operation_error)?
			} else {
				serialize_bounded_json(&value, indent).map_err(template_operation_error)?
			};
			let json = replace_json_separators(&json, &separator_pair)
				.map_err(template_operation_error)?;
			if ensure_ascii {
				ascii_escape_json(&json).map_err(template_operation_error)
			} else {
				Ok(json)
			}
		},
	);
	env.add_template_owned("chat", template_src)
		.map_err(|e| Error::Template(e.to_string()))?;
	Ok(env)
}

fn template_operation_error(message: String) -> minijinja::Error {
	minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, message)
}

fn serialize_bounded_json(
	value: &(impl Serialize + ?Sized),
	indent: Option<usize>,
) -> std::result::Result<String, String> {
	let mut output = BoundedPrompt::new(MAX_RENDERED_PROMPT_BYTES);
	match indent {
		Some(indent) => {
			let spaces = vec![b' '; indent];
			let formatter = serde_json::ser::PrettyFormatter::with_indent(&spaces);
			let mut serializer = serde_json::Serializer::with_formatter(&mut output, formatter);
			value
				.serialize(&mut serializer)
				.map_err(|error| error.to_string())?;
		}
		None => {
			let mut serializer = serde_json::Serializer::new(&mut output);
			value
				.serialize(&mut serializer)
				.map_err(|error| error.to_string())?;
		}
	}
	output.finish().map_err(|error| error.to_string())
}

#[derive(Debug)]
struct JsonSeparatorPair {
	item: String,
	key: String,
}

fn json_separator_pair(
	separators: Option<Vec<String>>,
	pretty: bool,
) -> std::result::Result<JsonSeparatorPair, String> {
	let (item, key) = match separators {
		Some(values) => {
			let [item, key]: [String; 2] = values.try_into().map_err(|values: Vec<String>| {
				format!(
					"tojson separators needs exactly two strings, got {}",
					values.len()
				)
			})?;
			(item, key)
		}
		None if pretty => (",".to_string(), ": ".to_string()),
		None => (", ".to_string(), ": ".to_string()),
	};
	if [&item, &key].iter().any(|separator| {
		separator.is_empty()
			|| separator.len() > 16
			|| !separator.is_ascii()
			|| separator.chars().any(char::is_control)
	}) {
		return Err(
			"tojson separators must be non-empty control-free ASCII strings of at most 16 bytes"
				.to_string(),
		);
	}
	Ok(JsonSeparatorPair { item, key })
}

fn replace_json_separators(
	json: &str,
	separators: &JsonSeparatorPair,
) -> std::result::Result<String, String> {
	let mut output = BoundedPrompt::new(MAX_RENDERED_PROMPT_BYTES);
	let mut in_string = false;
	let mut escaped = false;
	let mut bytes = json.bytes().peekable();
	while let Some(byte) = bytes.next() {
		if in_string {
			output
				.write_all(&[byte])
				.map_err(|error| error.to_string())?;
			if escaped {
				escaped = false;
			} else if byte == b'\\' {
				escaped = true;
			} else if byte == b'"' {
				in_string = false;
			}
			continue;
		}
		match byte {
			b'"' => {
				in_string = true;
				output
					.write_all(&[byte])
					.map_err(|error| error.to_string())?;
			}
			b',' => output
				.write_all(separators.item.as_bytes())
				.map_err(|error| error.to_string())?,
			b':' => {
				output
					.write_all(separators.key.as_bytes())
					.map_err(|error| error.to_string())?;
				if bytes.peek() == Some(&b' ') {
					let _ = bytes.next();
				}
			}
			_ => output
				.write_all(&[byte])
				.map_err(|error| error.to_string())?,
		}
	}
	output.finish().map_err(|error| error.to_string())
}

fn format_python_datetime(
	timestamp: DateTime<FixedOffset>,
	format: &str,
) -> std::result::Result<String, String> {
	const MAX_FORMAT_BYTES: usize = 256;
	const MAX_OUTPUT_BYTES: usize = 1_024;
	if format.len() > MAX_FORMAT_BYTES || format.chars().any(char::is_control) {
		return Err(format!(
			"strftime_now format must be control-free and at most {MAX_FORMAT_BYTES} bytes"
		));
	}
	let mut output = String::new();
	let mut characters = format.chars();
	while let Some(character) = characters.next() {
		if character != '%' {
			output.push(character);
		} else {
			let Some(directive) = characters.next() else {
				output.push('%');
				continue;
			};
			match directive {
				'f' => {
					use std::fmt::Write as _;
					write!(output, "{:06}", timestamp.timestamp_subsec_micros())
						.map_err(|error| error.to_string())?;
				}
				'%' => output.push('%'),
				's' => output.push_str(&timestamp.timestamp().to_string()),
				'a' | 'A' | 'w' | 'd' | 'b' | 'B' | 'm' | 'y' | 'Y' | 'H' | 'I' | 'p' | 'M'
				| 'S' | 'z' | 'j' | 'U' | 'W' | 'G' | 'g' | 'u' | 'V' | 'c' | 'x' | 'X' => {
					output.push_str(&timestamp.format(&format!("%{directive}")).to_string());
				}
				unknown if unknown.is_ascii_alphabetic() => {
					// Darwin's strftime, which backs Python on Emelex's
					// supported platform, drops `%` for unknown alphabetic
					// directives (for example `%i` renders `i`).
					output.push(unknown);
				}
				_ => {
					return Err(format!(
						"strftime_now directive `%{directive}` is unsupported"
					));
				}
			}
		}
		if output.len() > MAX_OUTPUT_BYTES {
			return Err(format!(
				"strftime_now output exceeds {MAX_OUTPUT_BYTES} bytes"
			));
		}
	}
	Ok(output)
}

fn ascii_escape_json(json: &str) -> std::result::Result<String, String> {
	let mut output = BoundedPrompt::new(MAX_RENDERED_PROMPT_BYTES);
	for character in json.chars() {
		if character.is_ascii() {
			let byte = u8::try_from(u32::from(character)).map_err(|error| error.to_string())?;
			output
				.write_all(&[byte])
				.map_err(|error| error.to_string())?;
		} else {
			let mut encoded = [0_u16; 2];
			for &unit in character.encode_utf16(&mut encoded).iter() {
				write!(output, "\\u{unit:04x}").map_err(|error| error.to_string())?;
			}
		}
	}
	output.finish().map_err(|error| error.to_string())
}

pub(crate) fn probe_chat_template_capabilities(
	template_src: &str,
) -> Result<ChatTemplateCapabilities> {
	probe_chat_templates_capabilities(
		&ChatTemplates::single(template_src.to_string()),
		("", ""),
		crate::engine::tools::ToolCallFormat::Hermes,
	)
}

const TOOL_PROBE_NAMES: [&str; 2] = ["emelex_probe_tool_4e7a", "emelex_probe_tool_b891"];
const TOOL_PROBE_SCHEMA_SENTINELS: [&str; 2] =
	["emelex_probe_schema_7ad3", "emelex_probe_schema_53c1"];
const TOOL_PROBE_CALL_ARGUMENTS: [&str; 2] =
	["emelex_probe_argument_91c2", "emelex_probe_argument_321e"];
const TOOL_PROBE_RESULTS: [&str; 2] = ["emelex_probe_result_2bd8", "emelex_probe_result_772f"];
const TOOL_PROBE_CALL_IDS: [&str; 2] = ["emelex_probe_call_6f03", "emelex_probe_call_197c"];
const TOOL_CONTROL_NAMES: [&str; 2] = ["emelex_control_tool_27a9", "emelex_control_tool_f137"];
const TOOL_CONTROL_SCHEMA_SENTINELS: [&str; 2] =
	["emelex_control_schema_96bc", "emelex_control_schema_e402"];

fn semantic_tools(names: [&str; 2], schema_sentinels: [&str; 2]) -> Vec<Tool> {
	names
		.into_iter()
		.zip(schema_sentinels)
		.map(|(name, schema_sentinel)| {
			let mut properties = serde_json::Map::new();
			properties.insert(
				schema_sentinel.to_string(),
				serde_json::json!({"type": "string"}),
			);
			Tool::new(
				name,
				"semantic capability probe",
				serde_json::json!({
					"type": "object",
					"properties": properties
				}),
			)
		})
		.collect()
}

pub(crate) fn semantic_probe_tools() -> Vec<Tool> {
	semantic_tools(TOOL_PROBE_NAMES, TOOL_PROBE_SCHEMA_SENTINELS)
}

fn tool_probe_values(tools: &[Tool]) -> Option<Vec<JinjaValue>> {
	Some(
		tools
			.iter()
			.map(JinjaValue::from_serialize)
			.collect::<Vec<_>>(),
	)
}

const fn tool_probe_call_count(format: crate::engine::tools::ToolCallFormat) -> usize {
	if matches!(format, crate::engine::tools::ToolCallFormat::LlamaJson) {
		1
	} else {
		2
	}
}

fn tool_probe_calls(format: crate::engine::tools::ToolCallFormat) -> Vec<ToolCall> {
	(0..tool_probe_call_count(format))
		.map(|index| ToolCall {
			id: TOOL_PROBE_CALL_IDS[index].to_string(),
			name: TOOL_PROBE_NAMES[index].to_string(),
			arguments: serde_json::json!({"probe": TOOL_PROBE_CALL_ARGUMENTS[index]}),
		})
		.collect()
}

pub(crate) fn semantic_probe_tool_turns(
	format: crate::engine::tools::ToolCallFormat,
) -> Vec<ChatMessage> {
	let call_count = tool_probe_call_count(format);
	std::iter::once(ChatMessage::assistant_with_tool_calls(
		"",
		tool_probe_calls(format),
	))
	.chain((0..call_count).map(|index| {
		ChatMessage::tool_result(TOOL_PROBE_CALL_IDS[index], TOOL_PROBE_RESULTS[index])
	}))
	.collect()
}

pub(crate) fn resolve_chat_templates_capabilities(
	templates: &ChatTemplates,
	special_tokens: (&str, &str),
) -> Result<(
	ChatTemplateCapabilities,
	crate::engine::tools::ToolCallFormat,
)> {
	use crate::engine::tools::ToolCallFormat;
	let now = Local::now().fixed_offset();
	let baseline =
		probe_chat_templates_capabilities_at(templates, special_tokens, ToolCallFormat::None, now)?;
	let mut supported = Vec::new();
	for format in [
		ToolCallFormat::Hermes,
		ToolCallFormat::Gemma,
		ToolCallFormat::Laguna,
		ToolCallFormat::LlamaJson,
	] {
		if let Ok(capabilities) =
			probe_chat_templates_capabilities_at(templates, special_tokens, format, now)
			&& capabilities.tools
		{
			supported.push((format, capabilities));
		}
	}
	match supported.as_slice() {
		[] => Ok((baseline, ToolCallFormat::None)),
		[(format, capabilities)] => Ok((*capabilities, *format)),
		_ => Ok((baseline, ToolCallFormat::None)),
	}
}

pub(crate) fn probe_chat_templates_capabilities(
	templates: &ChatTemplates,
	special_tokens: (&str, &str),
	tool_format: crate::engine::tools::ToolCallFormat,
) -> Result<ChatTemplateCapabilities> {
	probe_chat_templates_capabilities_at(
		templates,
		special_tokens,
		tool_format,
		Local::now().fixed_offset(),
	)
}

fn probe_chat_templates_capabilities_at(
	templates: &ChatTemplates,
	special_tokens: (&str, &str),
	tool_format: crate::engine::tools::ToolCallFormat,
	now: DateTime<FixedOffset>,
) -> Result<ChatTemplateCapabilities> {
	const USER_SENTINEL: &str = "emelex_probe_user_a281";
	let default_env = build_template_env_with_clock(
		strip_generation_tags(&templates.default),
		TemplateClock::Fixed(now),
	)?;
	// Fail-soft structured-translation probe. Runs unconditionally so
	// dual-mode templates report both shapes; a template without the
	// translation contract simply fails this render.
	let translation = probe_translation_capability(&default_env, special_tokens);
	let translation_only = ChatTemplateCapabilities {
		translation: true,
		..ChatTemplateCapabilities::default()
	};
	let baseline_messages = vec![ChatMessage::user(USER_SENTINEL)];
	let baseline = match render_template(
		&default_env,
		chat_messages_to_jinja(&baseline_messages),
		true,
		None,
		None,
		special_tokens,
	) {
		Ok(rendered) => rendered,
		// TranslateGemma-style templates reject plain-string chat turns
		// by design; the successful translation render is the authority.
		// Secondary probes are skipped — they all send plain strings and
		// would raise the same way.
		Err(_) if translation => return Ok(translation_only),
		Err(error) => return Err(error),
	};
	if baseline.trim().is_empty() || !baseline.contains(USER_SENTINEL) {
		if translation {
			return Ok(translation_only);
		}
		return Err(Error::Template(
			"chat template does not preserve a required user message".to_string(),
		));
	}
	let system_prompt = probe_system_capability(&default_env, special_tokens, None, tool_format);
	let tool_env = if tool_format != crate::engine::tools::ToolCallFormat::None
		&& templates.tool_use.is_some()
	{
		Some(build_template_env_with_clock(
			strip_generation_tags(templates.selected(true)),
			TemplateClock::Fixed(now),
		)?)
	} else {
		None
	};
	let tool_env = tool_env.as_ref().unwrap_or(&default_env);
	let tools = semantic_probe_tools();
	let tool_capability = tool_format != crate::engine::tools::ToolCallFormat::None
		&& probe_tool_capability(tool_env, &tools, special_tokens, tool_format).unwrap_or(false)
		&& probe_tool_parser(tool_format);
	let default_reasoning_history =
		probe_reasoning_history(&default_env, special_tokens, None, tool_format).unwrap_or(false);
	let default_thinking_toggle =
		probe_reasoning_toggle(&default_env, special_tokens, None, tool_format).unwrap_or(false);
	Ok(ChatTemplateCapabilities {
		chat: true,
		translation,
		system_prompt: system_prompt
			&& (!tool_capability
				|| probe_system_capability(tool_env, special_tokens, Some(&tools), tool_format)),
		tools: tool_capability,
		reasoning_history: default_reasoning_history
			&& (!tool_capability
				|| probe_reasoning_history(tool_env, special_tokens, Some(&tools), tool_format)
					.unwrap_or(false)),
		thinking_toggle: default_thinking_toggle
			&& (!tool_capability
				|| probe_reasoning_toggle(tool_env, special_tokens, Some(&tools), tool_format)
					.unwrap_or(false)),
	})
}

/// Fail-soft probe for the structured-translation contract: render one
/// user message whose content is a single translation mapping and require
/// the sentinel text to survive WITHOUT the mapping's internal key names
/// leaking into the prompt. A genuine translation template consumes
/// `source_lang_code`/`target_lang_code`; a plain chat template that
/// stringifies unknown list content dumps them verbatim — rejecting that
/// leak keeps ordinary chat models from claiming the translation task.
/// "en"/"de" are used because every known translation template's language
/// table contains them; templates without the contract fail the render
/// (or drop the sentinel) and report `false`.
fn probe_translation_capability(env: &Environment<'_>, special_tokens: (&str, &str)) -> bool {
	const TRANSLATION_SENTINEL: &str = "emelex_probe_translation_c417";
	let messages = vec![ChatMessage::user_translation(
		"en",
		"de",
		TRANSLATION_SENTINEL,
	)];
	render_template(
		env,
		chat_messages_to_jinja(&messages),
		true,
		None,
		None,
		special_tokens,
	)
	.ok()
	.is_some_and(|rendered| {
		!rendered.trim().is_empty()
			&& rendered.contains(TRANSLATION_SENTINEL)
			&& !rendered.contains("source_lang_code")
	})
}

/// Extract the `set languages = { "code": "Name", ... }` table a
/// TranslateGemma-style template embeds. Returns `None` unless a
/// well-formed table with at least 50 entries is found (guards against
/// matching unrelated small dicts) — fail-open: callers treat `None` as
/// "no validation possible", never as an error.
pub(crate) fn translation_language_table(
	template: &str,
) -> Option<std::collections::BTreeMap<String, String>> {
	const MIN_LANGUAGE_TABLE_ENTRIES: usize = 50;
	let start = template.find("set languages")?;
	let brace = start + template[start..].find('{')?;
	let literal = balanced_brace_span(&template[brace..])?;
	let mut entries = std::collections::BTreeMap::new();
	let mut rest = literal.strip_prefix('{')?;
	loop {
		rest = rest.trim_start_matches(|c: char| c.is_whitespace() || c == ',');
		if rest.starts_with('}') {
			break;
		}
		let (key, after_key) = leading_quoted_string(rest)?;
		let after_colon = after_key.trim_start().strip_prefix(':')?;
		let (value, after_value) = leading_quoted_string(after_colon.trim_start())?;
		entries.insert(key, value);
		rest = after_value;
	}
	(entries.len() >= MIN_LANGUAGE_TABLE_ENTRIES).then_some(entries)
}

/// The balanced `{...}` span starting at the first byte of `text`,
/// honoring double-quoted strings with backslash escapes.
fn balanced_brace_span(text: &str) -> Option<&str> {
	let mut depth = 0_usize;
	let mut in_string = false;
	let mut escaped = false;
	for (index, ch) in text.char_indices() {
		if in_string {
			if escaped {
				escaped = false;
			} else if ch == '\\' {
				escaped = true;
			} else if ch == '"' {
				in_string = false;
			}
			continue;
		}
		match ch {
			'"' => in_string = true,
			'{' => depth += 1,
			'}' => {
				depth = depth.checked_sub(1)?;
				if depth == 0 {
					return Some(&text[..=index]);
				}
			}
			_ => {}
		}
	}
	None
}

/// Split a leading double-quoted string into its contents and the
/// remainder after the closing quote.
fn leading_quoted_string(text: &str) -> Option<(String, &str)> {
	let rest = text.strip_prefix('"')?;
	let end = rest.find('"')?;
	Some((rest[..end].to_string(), &rest[end + 1..]))
}

fn probe_system_capability(
	env: &Environment<'_>,
	special_tokens: (&str, &str),
	tools: Option<&[Tool]>,
	tool_format: crate::engine::tools::ToolCallFormat,
) -> bool {
	const SYSTEM_SENTINEL: &str = "emelex_probe_system_f4c9";
	const USER_SENTINEL: &str = "emelex_probe_user_a281";
	const FOLLOWUP_SENTINEL: &str = "emelex_probe_followup_0a64";
	let mut system_messages = vec![
		ChatMessage::system(SYSTEM_SENTINEL),
		ChatMessage::user(USER_SENTINEL),
	];
	if tools.is_some() {
		system_messages.extend(semantic_probe_tool_turns(tool_format));
		system_messages.push(ChatMessage::user(FOLLOWUP_SENTINEL));
	}
	let rendered = render_template(
		env,
		match chat_messages_to_jinja_for_format(&system_messages, tool_format) {
			Ok(messages) => messages,
			Err(_) => return false,
		},
		true,
		tools.and_then(tool_probe_values),
		None,
		special_tokens,
	)
	.ok()
	.filter(|rendered| {
		rendered.contains(SYSTEM_SENTINEL)
			&& rendered.contains(USER_SENTINEL)
			&& (tools.is_none() || rendered.contains(FOLLOWUP_SENTINEL))
	});
	let Some(rendered) = rendered else {
		return false;
	};
	let mut role_blind_control = vec![
		ChatMessage::user(SYSTEM_SENTINEL),
		ChatMessage::user(USER_SENTINEL),
	];
	if tools.is_some() {
		role_blind_control.extend(semantic_probe_tool_turns(tool_format));
		role_blind_control.push(ChatMessage::user(FOLLOWUP_SENTINEL));
	}
	match chat_messages_to_jinja_for_format(&role_blind_control, tool_format).and_then(|messages| {
		render_template(
			env,
			messages,
			true,
			tools.and_then(tool_probe_values),
			None,
			special_tokens,
		)
	}) {
		Ok(control) => control != rendered,
		Err(_) => true,
	}
}

fn probe_tool_capability(
	env: &Environment<'_>,
	tools: &[Tool],
	special_tokens: (&str, &str),
	tool_format: crate::engine::tools::ToolCallFormat,
) -> Result<bool> {
	const USER_SENTINEL: &str = "emelex_probe_tool_user_85da";
	let baseline_messages = vec![ChatMessage::user(USER_SENTINEL)];
	let tools_value = || tool_probe_values(tools);
	let declaration = render_template(
		env,
		chat_messages_to_jinja_for_format(&baseline_messages, tool_format)?,
		true,
		tools_value(),
		None,
		special_tokens,
	)?;
	let control_tools = semantic_tools(TOOL_CONTROL_NAMES, TOOL_CONTROL_SCHEMA_SENTINELS);
	let control = render_template(
		env,
		chat_messages_to_jinja_for_format(&baseline_messages, tool_format)?,
		true,
		tool_probe_values(&control_tools),
		None,
		special_tokens,
	)?;
	if declaration == control
		|| TOOL_PROBE_NAMES
			.into_iter()
			.chain(TOOL_PROBE_SCHEMA_SENTINELS)
			.any(|sentinel| !declaration.contains(sentinel))
		|| TOOL_CONTROL_NAMES
			.into_iter()
			.chain(TOOL_CONTROL_SCHEMA_SENTINELS)
			.any(|sentinel| !control.contains(sentinel))
		|| TOOL_PROBE_NAMES
			.into_iter()
			.chain(TOOL_PROBE_SCHEMA_SENTINELS)
			.any(|sentinel| control.contains(sentinel))
		|| TOOL_CONTROL_NAMES
			.into_iter()
			.chain(TOOL_CONTROL_SCHEMA_SENTINELS)
			.any(|sentinel| declaration.contains(sentinel))
		|| !declaration.contains(USER_SENTINEL)
		|| !control.contains(USER_SENTINEL)
	{
		return Ok(false);
	}
	let call_count = tool_probe_call_count(tool_format);
	let calls = tool_probe_calls(tool_format);
	let history = vec![
		ChatMessage::user(USER_SENTINEL),
		ChatMessage::assistant_with_tool_calls("", calls.clone()),
	]
	.into_iter()
	.chain((0..call_count).map(|index| {
		ChatMessage::tool_result(TOOL_PROBE_CALL_IDS[index], TOOL_PROBE_RESULTS[index])
	}))
	.collect::<Vec<_>>();
	let rendered_history = render_template(
		env,
		chat_messages_to_jinja_for_format(&history, tool_format)?,
		true,
		tools_value(),
		None,
		special_tokens,
	)?;
	let role_blind_result = [
		ChatMessage::user(USER_SENTINEL),
		ChatMessage::assistant_with_tool_calls("", calls),
	]
	.into_iter()
	.chain((0..call_count).map(|index| ChatMessage::user(TOOL_PROBE_RESULTS[index])))
	.collect::<Vec<_>>();
	let distinguishes_tool_result = match render_template(
		env,
		chat_messages_to_jinja_for_format(&role_blind_result, tool_format)?,
		true,
		tools_value(),
		None,
		special_tokens,
	) {
		Ok(control) => control != rendered_history,
		Err(_) => true,
	};
	let names_round_trip = (0..call_count).all(|index| {
		rendered_history.matches(TOOL_PROBE_NAMES[index]).count()
			> declaration.matches(TOOL_PROBE_NAMES[index]).count()
	});
	let parsed_round_trip = if tool_format == crate::engine::tools::ToolCallFormat::LlamaJson {
		crate::engine::tools::rendered_llama_history_contains_call(
			&rendered_history,
			TOOL_PROBE_NAMES[0],
			&serde_json::json!({"probe": TOOL_PROBE_CALL_ARGUMENTS[0]}),
		)
	} else {
		parsed_tool_history_round_trips(&declaration, &rendered_history, tool_format, call_count)
	};
	let result_envelope = match tool_format {
		crate::engine::tools::ToolCallFormat::Gemma => gemma_results_round_trip(
			&rendered_history,
			&TOOL_PROBE_NAMES[..call_count],
			&TOOL_PROBE_RESULTS[..call_count],
		),
		_ => true,
	};
	Ok(rendered_history != declaration
		&& names_round_trip
		&& TOOL_PROBE_CALL_ARGUMENTS[..call_count]
			.iter()
			.chain(&TOOL_PROBE_RESULTS[..call_count])
			.all(|sentinel| rendered_history.contains(sentinel))
		&& distinguishes_tool_result
		&& result_envelope
		&& parsed_round_trip)
}

// emelex patch (not upstream): templates may include a literal, parseable
// tool-call example in their static instructions. Preserve fail-closed dynamic
// history certification by accepting only an identical static-call prefix
// followed by the exact ordered probe calls.
fn parsed_tool_history_round_trips(
	declaration: &str,
	rendered_history: &str,
	format: crate::engine::tools::ToolCallFormat,
	call_count: usize,
) -> bool {
	let declared = crate::engine::tools::parse_tool_calls(declaration, format);
	if declared
		.iter()
		.any(|call| TOOL_PROBE_NAMES.contains(&call.name.as_str()))
	{
		return false;
	}
	let rendered = crate::engine::tools::parse_tool_calls(rendered_history, format);
	let Some((prefix, probed)) = rendered.split_at_checked(declared.len()) else {
		return false;
	};
	let Some(expected_names) = TOOL_PROBE_NAMES.get(..call_count) else {
		return false;
	};
	let Some(expected_arguments) = TOOL_PROBE_CALL_ARGUMENTS.get(..call_count) else {
		return false;
	};
	prefix.iter().zip(&declared).all(|(actual, expected)| {
		actual.name == expected.name && actual.arguments == expected.arguments
	}) && probed.len() == call_count
		&& probed
			.iter()
			.zip(expected_names)
			.zip(expected_arguments)
			.all(|((call, name), arguments)| {
				call.name == *name && call.arguments == serde_json::json!({"probe": arguments})
			})
}

fn gemma_result_round_trips(rendered: &str, name: &str, result: &str) -> bool {
	gemma_results_round_trip(rendered, &[name], &[result])
}

fn gemma_results_round_trip(rendered: &str, names: &[&str], results: &[&str]) -> bool {
	const OPEN: &str = "<|tool_response>";
	const CLOSE: &str = "<tool_response|>";
	if names.len() != results.len()
		|| rendered.matches(OPEN).count() != names.len()
		|| rendered.matches(CLOSE).count() != names.len()
		|| results
			.iter()
			.any(|result| rendered.matches(result).count() != 1)
	{
		return false;
	}
	let mut cursor = rendered;
	names.iter().zip(results).all(|(name, result)| {
		let Some((_, after_open)) = cursor.split_once(OPEN) else {
			return false;
		};
		let Some((payload, after_close)) = after_open.split_once(CLOSE) else {
			return false;
		};
		cursor = after_close;
		let Some(arguments) = payload
			.trim()
			.strip_prefix(&format!("response:{name}"))
			.and_then(|payload| payload.trim_start().strip_prefix('{'))
			.and_then(|payload| payload.trim().strip_suffix('}'))
			.and_then(|payload| payload.trim().strip_prefix("value:"))
		else {
			return false;
		};
		let value = arguments.trim();
		value == *result
			|| value == format!("\"{result}\"")
			|| value == format!("<|\"|>{result}<|\"|>")
	})
}

fn probe_tool_parser(format: crate::engine::tools::ToolCallFormat) -> bool {
	const NAME: &str = "emelex_probe_tool_4e7a";
	let text = match format {
		crate::engine::tools::ToolCallFormat::Hermes => {
			r#"<tool_call>{"name":"emelex_probe_tool_4e7a","arguments":{"probe":"roundtrip"}}</tool_call>"#
		}
		crate::engine::tools::ToolCallFormat::Gemma => {
			r#"<|tool_call>call:emelex_probe_tool_4e7a{probe:"roundtrip"}<tool_call|>"#
		}
		crate::engine::tools::ToolCallFormat::Laguna => {
			"<tool_call>emelex_probe_tool_4e7a<arg_key>probe</arg_key><arg_value>roundtrip</arg_value></tool_call>"
		}
		crate::engine::tools::ToolCallFormat::LlamaJson => {
			r#"{"name":"emelex_probe_tool_4e7a","parameters":{"probe":"roundtrip"}}"#
		}
		crate::engine::tools::ToolCallFormat::None => return false,
	};
	let calls = crate::engine::tools::parse_tool_calls(text, format);
	matches!(
		calls.as_slice(),
		[call] if call.name == NAME && call.arguments == serde_json::json!({"probe": "roundtrip"})
	)
}

fn probe_reasoning_history(
	env: &Environment<'_>,
	special_tokens: (&str, &str),
	tools: Option<&[Tool]>,
	tool_format: crate::engine::tools::ToolCallFormat,
) -> Result<bool> {
	const REASONING: &str = "emelex_probe_reasoning_8ac1";
	let mut prefix = vec![ChatMessage::user("emelex probe user")];
	if tools.is_some() {
		prefix.extend(semantic_probe_tool_turns(tool_format));
	}
	let mut control = prefix.clone();
	control.push(ChatMessage::assistant("emelex probe answer"));
	control.push(ChatMessage::user("emelex probe followup"));
	let mut with_reasoning = prefix;
	with_reasoning.push(ChatMessage::assistant_with_reasoning(
		"emelex probe answer",
		REASONING,
	));
	with_reasoning.push(ChatMessage::user("emelex probe followup"));
	let control = render_template(
		env,
		chat_messages_to_jinja_for_format(&control, tool_format)?,
		true,
		tools.and_then(tool_probe_values),
		None,
		special_tokens,
	)?;
	let with_reasoning = render_template(
		env,
		chat_messages_to_jinja_for_format(&with_reasoning, tool_format)?,
		true,
		tools.and_then(tool_probe_values),
		None,
		special_tokens,
	)?;
	Ok(with_reasoning != control
		&& crate::engine::reasoning::MARKER_PAIRS
			.iter()
			.any(|(open, close)| {
				with_reasoning.find(open).is_some_and(|start| {
					with_reasoning[start + open.len()..]
						.find(close)
						.is_some_and(|end| {
							with_reasoning[start + open.len()..start + open.len() + end]
								.contains(REASONING)
						})
				})
			}))
}

fn probe_reasoning_toggle(
	env: &Environment<'_>,
	special_tokens: (&str, &str),
	tools: Option<&[Tool]>,
	tool_format: crate::engine::tools::ToolCallFormat,
) -> Result<bool> {
	let mut messages = vec![ChatMessage::user("emelex probe user")];
	if tools.is_some() {
		messages.extend(semantic_probe_tool_turns(tool_format));
		messages.push(ChatMessage::user("emelex probe followup"));
	}
	let disabled = render_template(
		env,
		chat_messages_to_jinja_for_format(&messages, tool_format)?,
		true,
		tools.and_then(tool_probe_values),
		Some(false),
		special_tokens,
	)?;
	let enabled = render_template(
		env,
		chat_messages_to_jinja_for_format(&messages, tool_format)?,
		true,
		tools.and_then(tool_probe_values),
		Some(true),
		special_tokens,
	)?;
	let enabled_pending = crate::engine::reasoning::pending_marker(&enabled);
	let disabled_pending = crate::engine::reasoning::pending_marker(&disabled);
	let enabled_empty = crate::engine::reasoning::trailing_empty_marker(&enabled);
	let disabled_empty = crate::engine::reasoning::trailing_empty_marker(&disabled);
	Ok(enabled != disabled
		&& ((disabled_pending.is_none() && enabled_pending.is_some())
			|| (disabled_empty.is_some() && enabled_empty.is_none())))
}

struct BoundedPrompt {
	bytes: Vec<u8>,
	limit: usize,
}

impl BoundedPrompt {
	fn new(limit: usize) -> Self {
		Self {
			bytes: Vec::new(),
			limit,
		}
	}

	fn finish(self) -> Result<String> {
		String::from_utf8(self.bytes)
			.map_err(|error| Error::Template(format!("template rendered invalid UTF-8: {error}")))
	}
}

impl std::io::Write for BoundedPrompt {
	fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
		let next = self
			.bytes
			.len()
			.checked_add(bytes.len())
			.ok_or_else(|| std::io::Error::other("rendered prompt size overflow"))?;
		if next > self.limit {
			return Err(std::io::Error::other(format!(
				"rendered prompt exceeds {} byte limit",
				self.limit
			)));
		}
		self.bytes.extend_from_slice(bytes);
		Ok(bytes.len())
	}

	fn flush(&mut self) -> std::io::Result<()> {
		Ok(())
	}
}

fn strip_generation_tags(src: &str) -> String {
	let mut out = String::with_capacity(src.len());
	let mut rest = src;
	loop {
		let Some(start) = rest.find("{%") else {
			out.push_str(rest);
			break;
		};
		let Some(rel_end) = rest[start..].find("%}") else {
			out.push_str(rest);
			break;
		};
		let end = start + rel_end + 2;
		let tag_body = rest[start + 2..end - 2].trim().trim_matches('-').trim();
		if tag_body == "generation" || tag_body == "endgeneration" {
			out.push_str(&rest[..start]);
		} else {
			out.push_str(&rest[..end]);
		}
		rest = &rest[end..];
	}
	out
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn strips_generation_tags() {
		let src = "a{% generation %}b{% endgeneration %}c{% if x %}d{% endif %}";
		let stripped = strip_generation_tags(src);
		assert_eq!(stripped, "abc{% if x %}d{% endif %}");
	}

	#[test]
	fn strips_generation_tags_with_whitespace_control() {
		let src = "a{%- generation -%}b{%- endgeneration -%}c";
		let stripped = strip_generation_tags(src);
		assert_eq!(stripped, "abc");
	}

	fn dummy_tokenizer_with_template(template: &str) -> Tokenizer {
		// Minimal `HfTokenizer` covering just the bytes needed for tests
		// that only exercise `apply_chat_template` (never `encode`).
		let inner = HfTokenizer::from_bytes(
			br#"{"version":"1.0","model":{"type":"BPE","vocab":{"a":0},"merges":[]}}"#,
		)
		.unwrap();
		Tokenizer {
			inner,
			chat_template: Some(template.to_string()),
			tool_chat_template: None,
			bos_token: Some("<bos>".into()),
			eos_token: Some("<eos>".into()),
			eos_token_ids: Vec::new(),
			template_env: std::sync::OnceLock::new(),
			tool_template_env: std::sync::OnceLock::new(),
		}
	}

	#[test]
	fn renders_simple_template() {
		let tok = dummy_tokenizer_with_template(
			"{%- for m in messages %}{{ m.role }}: {{ m.content }}\n{%- endfor %}",
		);
		let messages = vec![ChatMessage::user("hi"), ChatMessage::assistant("hello")];
		let rendered = tok.apply_chat_template(&messages, false).unwrap();
		assert_eq!(rendered, "user: hiassistant: hello");
	}

	#[test]
	fn renders_tools_into_context() {
		let tok = dummy_tokenizer_with_template(
			"{%- if tools %}TOOLS:{% for t in tools %}{{ t.function.name }},{% \
			 endfor %}{%- endif %}",
		);
		let tools = vec![Tool::new("get_weather", "desc", serde_json::json!({}))];
		let rendered = tok
			.apply_chat_template_with_tools(&[ChatMessage::user("hi")], false, Some(&tools))
			.unwrap();
		assert_eq!(rendered, "TOOLS:get_weather,");
	}

	#[test]
	fn processor_template_precedes_tokenizer_template() {
		let tokenizer = br#"{"version":"1.0","model":{"type":"BPE","vocab":{"a":0},"merges":[]}}"#;
		let tokenizer_config = br#"{"chat_template":"tokenizer {{ messages[0].content }}"}"#;
		let processor_config = br#"{"chat_template":"processor {{ messages[0].content }}"}"#;
		let tok = Tokenizer::from_artifacts(
			tokenizer,
			Some(tokenizer_config),
			Some(processor_config),
			None,
			None,
			None,
		)
		.unwrap();
		assert_eq!(
			tok.apply_chat_template(&[ChatMessage::user("a")], false)
				.unwrap(),
			"processor a"
		);
	}

	#[test]
	fn named_template_list_selects_default_and_tool_use() {
		let value = serde_json::json!([
			{"name": "default", "template": "plain"},
			{"name": "tool_use", "template": "tools"}
		]);
		assert_eq!(
			chat_templates_from_value(&value).unwrap(),
			Some(ChatTemplates::with_tool_use(
				"plain".to_string(),
				"tools".to_string()
			))
		);
		let single = serde_json::json!([{"name": "chat", "template": "only"}]);
		assert_eq!(
			chat_templates_from_value(&single).unwrap(),
			Some(ChatTemplates::single("only".to_string()))
		);
	}

	#[test]
	fn named_template_list_rejects_ambiguous_or_malformed_entries() {
		for value in [
			serde_json::json!([
				{"name": "one", "template": "a"},
				{"name": "two", "template": "b"}
			]),
			serde_json::json!([
				{"name": "default", "template": "a"},
				{"name": "default", "template": "b"}
			]),
			serde_json::json!([{"name": "default", "template": "a", "extra": true}]),
			serde_json::json!([{"name": "default"}]),
		] {
			assert!(chat_templates_from_value(&value).is_err(), "{value}");
		}
	}

	#[test]
	fn artifact_resolver_applies_processor_legacy_and_file_precedence() {
		let processor = serde_json::json!("processor");
		let tokenizer = serde_json::json!("tokenizer");
		let legacy = serde_json::json!({"chat_template": "legacy"});
		assert_eq!(
			resolve_chat_template_artifacts(
				&processor,
				Some(&legacy),
				Some("file".to_string()),
				None,
				&tokenizer,
			)
			.unwrap(),
			Some(ChatTemplates::single("processor".to_string()))
		);
		assert_eq!(
			resolve_chat_template_artifacts(
				&Value::Null,
				Some(&legacy),
				Some("file".to_string()),
				None,
				&tokenizer,
			)
			.unwrap(),
			Some(ChatTemplates::single("legacy".to_string()))
		);
		assert_eq!(
			resolve_chat_template_artifacts(
				&Value::Null,
				None,
				Some("file".to_string()),
				Some("tool".to_string()),
				&tokenizer,
			)
			.unwrap(),
			Some(ChatTemplates::with_tool_use(
				"file".to_string(),
				"tool".to_string()
			))
		);
		assert_eq!(
			resolve_chat_template_artifacts(
				&Value::Null,
				Some(&legacy),
				Some("file".to_string()),
				Some("tool".to_string()),
				&tokenizer,
			)
			.unwrap(),
			Some(ChatTemplates::single("legacy".to_string()))
		);
	}

	#[test]
	fn legacy_chat_template_file_rejects_wrong_shape() {
		for value in [
			serde_json::json!("template"),
			serde_json::json!({}),
			serde_json::json!({"chat_template": null}),
			serde_json::json!({"chat_template": "ok", "extra": true}),
		] {
			assert!(legacy_chat_templates_from_value(&value).is_err());
		}
	}

	#[test]
	fn normalized_tool_template_is_selected_only_when_tools_are_present() {
		let tokenizer = br#"{"version":"1.0","model":{"type":"BPE","vocab":{"a":0},"merges":[]}}"#;
		let tok = Tokenizer::from_artifacts(
			tokenizer,
			None,
			None,
			None,
			Some(b"default {{ messages[0].content }}"),
			Some(b"tool {{ tools[0].function.name }} {{ messages[0].content }}"),
		)
		.unwrap();
		let messages = [ChatMessage::user("a")];
		assert_eq!(
			tok.apply_chat_template(&messages, false).unwrap(),
			"default a"
		);
		let tools = [Tool::new("read", "desc", serde_json::json!({}))];
		assert_eq!(
			tok.apply_chat_template_with_tools(&messages, false, Some(&tools))
				.unwrap(),
			"tool read a"
		);
	}

	#[test]
	fn missing_template_errors() {
		let mut tok = dummy_tokenizer_with_template("x");
		tok.chat_template = None;
		assert!(
			tok.apply_chat_template(&[ChatMessage::user("hi")], false)
				.is_err()
		);
	}

	#[test]
	fn rejects_invalid_or_oversized_templates_before_use() {
		assert!(probe_chat_template_capabilities("{% if %}").is_err());
		let oversized = "x".repeat(MAX_CHAT_TEMPLATE_BYTES + 1);
		assert!(probe_chat_template_capabilities(&oversized).is_err());
	}

	#[test]
	fn template_execution_has_fuel_limit() {
		let tok =
			dummy_tokenizer_with_template("{% for item in range(3000000) %}{{ item }}{% endfor %}");
		let error = tok
			.apply_chat_template(&[ChatMessage::user("hi")], false)
			.unwrap_err();
		assert!(matches!(error, Error::Template(_)));
	}

	#[test]
	fn strftime_now_uses_bounded_python_style_directives() {
		use chrono::{TimeZone as _, Timelike as _};
		let offset = FixedOffset::east_opt(2 * 60 * 60).unwrap();
		let now = offset
			.with_ymd_and_hms(2026, 7, 27, 14, 5, 6)
			.single()
			.unwrap()
			.with_nanosecond(123_456_000)
			.unwrap();
		assert_eq!(
			format_python_datetime(now, "%Y-%m-%d %H:%M:%S.%f %z").unwrap(),
			"2026-07-27 14:05:06.123456 +0200"
		);
		assert_eq!(format_python_datetime(now, "%Q").unwrap(), "Q");
		assert_eq!(format_python_datetime(now, "%").unwrap(), "%");
		assert_eq!(
			format_python_datetime(now, "%G:%i:%s").unwrap(),
			format!("2026:i:{}", now.timestamp())
		);
	}

	#[test]
	fn tojson_honors_indent_ascii_and_rejects_unknown_kwargs() {
		let env = build_template_env(
			r#"{{ {"greeting": messages[0].content}|tojson(indent=4, ensure_ascii=True) }}"#
				.to_string(),
		)
		.unwrap();
		let rendered = render_template(
			&env,
			chat_messages_to_jinja(&[ChatMessage::user("café")]),
			false,
			None,
			None,
			("", ""),
		)
		.unwrap();
		assert_eq!(rendered, "{\n    \"greeting\": \"caf\\u00e9\"\n}");

		let env = build_template_env(
			r#"{{ {"b": 1, "a": 2}|tojson(sort_keys=True, separators=[",", ":"]) }}"#.to_string(),
		)
		.unwrap();
		let rendered = render_template(
			&env,
			chat_messages_to_jinja(&[ChatMessage::user("hello")]),
			false,
			None,
			None,
			("", ""),
		)
		.unwrap();
		assert_eq!(rendered, r#"{"a":2,"b":1}"#);

		let env =
			build_template_env(r#"{{ messages|tojson(allow_nan=False) }}"#.to_string()).unwrap();
		assert!(
			render_template(
				&env,
				chat_messages_to_jinja(&[ChatMessage::user("hello")]),
				false,
				None,
				None,
				("", ""),
			)
			.is_err()
		);
	}

	#[test]
	fn semantic_role_control_uses_one_fixed_datetime() {
		let capabilities = probe_chat_template_capabilities(
			r#"
{{ strftime_now("%Y-%m-%d %H:%M:%S.%f") }}
{% for message in messages %}{{ message.content }}{% endfor %}
"#,
		)
		.unwrap();
		assert!(!capabilities.system_prompt);
	}

	#[test]
	fn semantic_probe_rejects_keywords_comments_and_dead_branches() {
		let capabilities = probe_chat_template_capabilities(
			r"
{# tools tool_calls function enable_thinking reasoning content #}
{% if false %}
	{{ tools|tojson }}
	{{ messages[0].tool_calls|tojson }}
	{{ messages[0].reasoning_content }}
{% endif %}
{% for message in messages %}{{ message.content }}{% endfor %}
",
		)
		.unwrap();
		assert_eq!(
			capabilities,
			ChatTemplateCapabilities {
				chat: true,
				..ChatTemplateCapabilities::default()
			}
		);
	}

	#[test]
	fn semantic_probe_requires_a_successful_baseline_chat_render() {
		let error = probe_chat_template_capabilities(
			r"
{% if messages %}{{ raise_exception('runtime failure') }}{% endif %}
",
		)
		.expect_err("runtime-raising template cannot claim chat");
		assert!(matches!(error, Error::Template(_)));
	}

	#[test]
	fn semantic_probe_requires_tool_declarations_calls_and_results() {
		let declarations_only = probe_chat_template_capabilities(
			r#"
{% if tools %}{{ tools|tojson }}{% endif %}
{% for message in messages %}{{ message.content }}{% endfor %}
"#,
		)
		.unwrap();
		assert!(!declarations_only.tools);

		let partial_fields = probe_chat_template_capabilities(
			r#"
{% if tools %}{% for tool in tools %}{{ tool.function.name }}{% endfor %}{% endif %}
{% for message in messages %}
	{% if message.tool_calls %}
		{% for call in message.tool_calls %}
			<call>{{ call.function.arguments|tojson }}</call>
		{% endfor %}
	{% elif message.role == "tool" %}
		<result>{{ message.content }}</result>
	{% else %}
		{{ message.content }}
	{% endif %}
{% endfor %}
"#,
		)
		.unwrap();
		assert!(!partial_fields.tools);

		let first_tool_only = probe_chat_template_capabilities(
			r#"
{% if tools %}{{ tools[0]|tojson }}{% endif %}
{% for message in messages %}
	{% if message.tool_calls %}
		{% if message.tool_calls|length > 1 %}{{ raise_exception("one call only") }}{% endif %}
		<tool_call>{"name":{{ message.tool_calls[0].function.name|tojson }},"arguments":{{ message.tool_calls[0].function.arguments|tojson }}}</tool_call>
	{% elif message.role == "tool" %}
		<result>{{ message.content }}</result>
	{% else %}
		{{ message.content }}
	{% endif %}
{% endfor %}
"#,
		)
		.unwrap();
		assert!(!first_tool_only.tools);

		let complete = probe_chat_template_capabilities(
			r#"
{% if tools %}<tools>{{ tools|tojson }}</tools>{% endif %}
{% for message in messages %}
	{% if message.tool_calls %}
		{% for call in message.tool_calls %}
			<tool_call>{"name":{{ call.function.name|tojson }},"arguments":{{ call.function.arguments|tojson }}}</tool_call>
		{% endfor %}
	{% elif message.role == "tool" %}
		<result>{{ message.content }}</result>
	{% else %}
		{{ message.content }}
	{% endif %}
{% endfor %}
"#,
		)
		.unwrap();
		assert!(complete.tools);

		let generation_gated = probe_chat_template_capabilities(
			r#"
{% if add_generation_prompt %}
	{% if tools %}<tools>{{ tools|tojson }}</tools>{% endif %}
	{% for message in messages %}
			{% if message.tool_calls %}
			{% for call in message.tool_calls %}
				<tool_call>{"name":{{ call.function.name|tojson }},"arguments":{{ call.function.arguments|tojson }}}</tool_call>
			{% endfor %}
		{% elif message.role == "tool" %}
			<result>{{ message.content }}</result>
		{% else %}
			{{ message.content }}
		{% endif %}
	{% endfor %}
{% else %}
	{% for message in messages %}{{ message.content }}{% endfor %}
{% endif %}
"#,
		)
		.unwrap();
		assert!(generation_gated.tools);
	}

	#[test]
	fn semantic_probe_separates_chat_from_system_prompt_support() {
		let capabilities = probe_chat_template_capabilities(
			r#"
{% for message in messages %}
	{% if message.role == "system" %}{% else %}{{ message.content }}{% endif %}
{% endfor %}
"#,
		)
		.unwrap();
		assert!(!capabilities.system_prompt);
	}

	#[test]
	fn semantic_probe_rejects_role_blind_system_passthrough() {
		let capabilities = probe_chat_template_capabilities(
			"{% for message in messages %}{{ message.content }}{% endfor %}",
		)
		.unwrap();
		assert!(!capabilities.system_prompt);
	}

	#[test]
	fn jinja_messages_omit_optional_protocol_keys_until_present() {
		let env = build_template_env(
			r#"
{% for message in messages -%}
{{ message.role }}:
{%- if "tool_calls" in message %}calls{% endif -%}
{%- if "tool_call_id" in message %}result{% endif -%}
{%- if "name" in message %}({{ message.name }}){% endif -%}
{%- if "reasoning_content" in message %}reasoning{% endif %};
{%- endfor %}
"#
			.to_string(),
		)
		.unwrap();
		let messages = vec![
			ChatMessage::user("plain"),
			ChatMessage::assistant_with_tool_calls(
				"",
				vec![ToolCall {
					id: "call_1".to_string(),
					name: "lookup".to_string(),
					arguments: serde_json::json!({}),
				}],
			),
			ChatMessage::tool_result("call_1", "done"),
			ChatMessage::assistant_with_reasoning("answer", "thought"),
		];
		let rendered = render_template(
			&env,
			chat_messages_to_jinja(&messages),
			false,
			None,
			None,
			("", ""),
		)
		.unwrap();
		assert_eq!(
			rendered.split_whitespace().collect::<String>(),
			"user:;assistant:calls;tool:result(lookup);assistant:reasoning;"
		);
	}

	#[test]
	fn exact_template_resolves_llama_json_tool_protocol() {
		let templates = ChatTemplates::single(
			r#"
{% if tools %}<tools>{{ tools|tojson }}</tools>{% endif %}
{% for message in messages %}
	{% if message.tool_calls %}
		{% set call = message.tool_calls[0].function %}
		{{ {"name": call.name, "parameters": call.arguments}|tojson }}
	{% elif message.role == "tool" %}
		<tool_result name="{{ message.name }}">{{ message.content }}</tool_result>
	{% else %}
		<{{ message.role }}>{{ message.content }}</{{ message.role }}>
	{% endif %}
{% endfor %}
"#
			.to_string(),
		);
		let (capabilities, format) =
			resolve_chat_templates_capabilities(&templates, ("", "")).unwrap();
		assert!(capabilities.system_prompt);
		assert!(capabilities.tools);
		assert_eq!(format, crate::engine::tools::ToolCallFormat::LlamaJson);
	}

	#[test]
	fn qwen_xml_tool_example_does_not_hide_tool_support() {
		let templates = ChatTemplates::single(
			r#"
{% if tools %}
<tools>{% for tool in tools %}{{ tool|tojson }}{% endfor %}</tools>
<tool_call>
<function=example_function_name>
<parameter=example_parameter>example_value</parameter>
</function>
</tool_call>
{% endif %}
{% for message in messages %}
	{% if message.tool_calls %}
		{% for call in message.tool_calls %}
<tool_call>
<function={{ call.function.name }}>
			{% for name, value in call.function.arguments|items %}
<parameter={{ name }}>{{ value }}</parameter>
			{% endfor %}
</function>
</tool_call>
		{% endfor %}
	{% elif message.role == "tool" %}
<tool_response>{{ message.content }}</tool_response>
	{% else %}
<{{ message.role }}>{{ message.content }}</{{ message.role }}>
	{% endif %}
{% endfor %}
"#
			.to_string(),
		);
		let (capabilities, format) =
			resolve_chat_templates_capabilities(&templates, ("", "")).expect("capability probe");
		assert!(capabilities.tools);
		assert_eq!(format, crate::engine::tools::ToolCallFormat::Hermes);
	}

	#[test]
	fn gemma_native_tool_results_are_grouped_and_fail_closed() {
		let template = r#"
{% if tools %}<tools>{{ tools|tojson }}</tools>{% endif %}
{% for message in messages %}
	{% if message.tool_calls %}
		{% for call in message.tool_calls %}
			<|tool_call>call:{{ call.function.name }}{probe:"{{ call.function.arguments.probe }}"}<tool_call|>
		{% endfor %}
	{% endif %}
	{% if message.tool_responses %}
		{% for response in message.tool_responses %}
			<|tool_response>response:{{ response.name }}{value:"{{ response.response }}"}<tool_response|>
		{% endfor %}
	{% elif not message.tool_calls %}
		{{ message.role }}:{{ message.content }}
	{% endif %}
{% endfor %}
"#;
		let templates = ChatTemplates::single(template.to_string());
		let (capabilities, format) =
			resolve_chat_templates_capabilities(&templates, ("", "")).unwrap();
		assert!(capabilities.tools);
		assert_eq!(format, crate::engine::tools::ToolCallFormat::Gemma);

		let messages = vec![
			ChatMessage::user("question"),
			ChatMessage::assistant_with_tool_calls(
				"",
				vec![
					ToolCall {
						id: "call_1".to_string(),
						name: "first".to_string(),
						arguments: serde_json::json!({"probe": "one"}),
					},
					ToolCall {
						id: "call_2".to_string(),
						name: "second".to_string(),
						arguments: serde_json::json!({"probe": "two"}),
					},
				],
			),
			ChatMessage::tool_result("call_1", "result-one"),
			ChatMessage::tool_result("call_2", "result-two"),
		];
		let env = build_template_env(template.to_string()).unwrap();
		let rendered = render_template(
			&env,
			chat_messages_to_jinja_for_format(
				&messages,
				crate::engine::tools::ToolCallFormat::Gemma,
			)
			.unwrap(),
			false,
			None,
			None,
			("", ""),
		)
		.unwrap();
		assert_eq!(rendered.matches("<|tool_response>").count(), 2);
		assert_eq!(rendered.matches("result-one").count(), 1);
		assert_eq!(rendered.matches("result-two").count(), 1);
		assert!(
			rendered.find("response:first").unwrap() < rendered.find("response:second").unwrap()
		);
		assert!(gemma_result_round_trips(
			"<|tool_response>response:first{value:\"result-one\"}<tool_response|>",
			"first",
			"result-one"
		));
		assert!(!gemma_result_round_trips(
			"result-one<|tool_response><tool_response|>",
			"first",
			"result-one"
		));
		assert!(!gemma_result_round_trips(
			"<|tool_response>response:first{value:\"result-one\",junk:true}<tool_response|>",
			"first",
			"result-one"
		));

		let unknown = vec![
			messages[0].clone(),
			messages[1].clone(),
			ChatMessage::tool_result("unknown", "result"),
		];
		assert!(
			chat_messages_to_jinja_for_format(
				&unknown,
				crate::engine::tools::ToolCallFormat::Gemma
			)
			.is_err()
		);
		let duplicate = vec![
			messages[0].clone(),
			messages[1].clone(),
			ChatMessage::tool_result("call_1", "one"),
			ChatMessage::tool_result("call_1", "again"),
		];
		assert!(
			chat_messages_to_jinja_for_format(
				&duplicate,
				crate::engine::tools::ToolCallFormat::Gemma
			)
			.is_err()
		);
		let reversed = vec![
			messages[0].clone(),
			messages[1].clone(),
			ChatMessage::tool_result("call_2", "two"),
			ChatMessage::tool_result("call_1", "one"),
		];
		assert!(
			chat_messages_to_jinja_for_format(
				&reversed,
				crate::engine::tools::ToolCallFormat::Gemma
			)
			.is_err()
		);
	}

	#[test]
	fn llama_style_membership_check_keeps_plain_user_on_chat_branch() {
		let env = build_template_env(
			r#"
{% for message in messages %}
{% if "tool_calls" in message %}
	{% if message.tool_calls|length != 1 %}{{ raise_exception("single call required") }}{% endif %}
	TOOL={{ message.tool_calls[0].function.name }}
{% else %}
	{{ message.role }}={{ message.content }}
{% endif %}
{% endfor %}
"#
			.to_string(),
		)
		.unwrap();
		let rendered = render_template(
			&env,
			chat_messages_to_jinja(&[ChatMessage::user("hello")]),
			true,
			None,
			None,
			("", ""),
		)
		.unwrap();
		assert!(rendered.contains("user=hello"));
	}

	#[test]
	fn semantic_probe_keeps_tool_use_orthogonal_to_system_prompt_support() {
		let capabilities = probe_chat_template_capabilities(
			r#"
{% if tools %}<tools>{{ tools|tojson }}</tools>{% endif %}
{% for message in messages %}
	{% if message.role == "system" %}
	{% elif message.tool_calls %}
		{% for call in message.tool_calls %}
			<tool_call>{"name":{{ call.function.name|tojson }},"arguments":{{ call.function.arguments|tojson }}}</tool_call>
		{% endfor %}
	{% elif message.role == "tool" %}
		<result>{{ message.content }}</result>
	{% else %}
		{{ message.content }}
	{% endif %}
{% endfor %}
"#,
		)
		.unwrap();
		assert!(capabilities.tools);
		assert!(!capabilities.system_prompt);
	}

	#[test]
	fn semantic_probe_requires_capabilities_across_dedicated_tool_history() {
		let default = r#"
{% for message in messages %}
	{% if message.role == "system" %}<system>{{ message.content }}</system>{% endif %}
	{% if message.reasoning_content %}<think>{{ message.reasoning_content }}</think>{% endif %}
	{% if message.role != "system" %}{{ message.content }}{% endif %}
{% endfor %}
{% if add_generation_prompt and enable_thinking %}<think>{% endif %}
"#;
		let tool_use = r#"
{% if not tools %}{{ raise_exception("tools required") }}{% endif %}
<tools>{{ tools|tojson }}</tools>
{% set state = namespace(has_tool_history=false) %}
{% for message in messages %}
	{% if message.tool_calls or message.role == "tool" %}
		{% set state.has_tool_history = true %}
	{% endif %}
{% endfor %}
{% for message in messages %}
	{% if message.tool_calls %}
		{% for call in message.tool_calls %}
			<tool_call>{"name":{{ call.function.name|tojson }},"arguments":{{ call.function.arguments|tojson }}}</tool_call>
		{% endfor %}
	{% elif message.role == "tool" %}
		<tool_result>{{ message.content }}</tool_result>
	{% elif message.role == "system" and not state.has_tool_history %}
		<system>{{ message.content }}</system>
	{% elif message.role != "system" %}
		{{ message.content }}
	{% endif %}
{% endfor %}
"#;
		let templates = ChatTemplates::with_tool_use(default.to_string(), tool_use.to_string());
		let (capabilities, format) =
			resolve_chat_templates_capabilities(&templates, ("", "")).unwrap();
		assert_eq!(format, crate::engine::tools::ToolCallFormat::Hermes);
		assert!(capabilities.tools);
		assert!(!capabilities.system_prompt);
		assert!(!capabilities.reasoning_history);
		assert!(!capabilities.thinking_toggle);
	}

	#[test]
	fn semantic_probe_accepts_rendered_reasoning_history() {
		let capabilities = probe_chat_template_capabilities(
			r#"
{% for message in messages %}
	{% if message.reasoning_content %}
		<think>{{ message.reasoning_content }}</think>
	{% endif %}
	{{ message.content }}
{% endfor %}
"#,
		)
		.unwrap();
		assert!(capabilities.reasoning_history);
		assert!(!capabilities.thinking_toggle);
	}

	#[test]
	fn semantic_probe_accepts_recognized_reasoning_toggle() {
		let capabilities = probe_chat_template_capabilities(
			r#"
{% for message in messages %}{{ message.content }}{% endfor %}
{% if add_generation_prompt and enable_thinking %}<think>{% endif %}
"#,
		)
		.unwrap();
		assert!(!capabilities.reasoning_history);
		assert!(capabilities.thinking_toggle);
	}

	#[test]
	fn semantic_probe_accepts_gemma_style_disabled_empty_thought_suffix() {
		let capabilities = probe_chat_template_capabilities(
			r#"
{% if enable_thinking %}<|think|>{% endif %}
{% for message in messages %}<|turn>{{ message.role }}
{{ message.content }}<turn|>
{% endfor %}
{% if add_generation_prompt %}<|turn>model
{% if not enable_thinking | default(false) %}<|channel>thought
<channel|>{% endif %}{% endif %}
"#,
		)
		.unwrap();
		assert!(!capabilities.reasoning_history);
		assert!(capabilities.thinking_toggle);
	}

	#[test]
	fn semantic_probe_rejects_non_terminal_empty_thought_difference() {
		let capabilities = probe_chat_template_capabilities(
			r#"
{% if enable_thinking %}<|think|>{% else %}<think></think>{% endif %}
{% for message in messages %}{{ message.content }}{% endfor %}
{% if add_generation_prompt %}model{% endif %}
"#,
		)
		.unwrap();
		assert!(!capabilities.thinking_toggle);
	}

	// emelex patch (not upstream): tiny-model fixture gate. The committed
	// fixture trio under tests/fixtures/tiny-model backs the non-live MTP
	// test suites; this test is the "fixtures load" gate for it.
	#[test]
	fn tiny_fixture_tokenizer_loads_and_keeps_boundary_prefix() {
		let dir =
			std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-model");
		let tok = Tokenizer::load(&dir).unwrap();
		assert_eq!(tok.eos_token_ids(), &[2]);

		// Close marker and words round-trip to stable ids.
		let ids = tok.encode("<think> hello </think>").unwrap();
		assert_eq!(ids, vec![3, 8, 4]);

		// The generation prompt appends non-history tokens strictly after
		// the conversation boundary - the property the boundary-snapshot
		// prompt cache (and its MtpState alignment) depends on.
		let messages = vec![
			ChatMessage::user("hello world"),
			ChatMessage::assistant("hi"),
			ChatMessage::user("ok"),
		];
		let boundary = tok.apply_chat_template(&messages, false).unwrap();
		let full = tok.apply_chat_template(&messages, true).unwrap();
		assert!(full.starts_with(&boundary));
		assert!(full.len() > boundary.len());
		let boundary_ids = tok.encode(&boundary).unwrap();
		let full_ids = tok.encode(&full).unwrap();
		assert!(!boundary_ids.is_empty());
		assert!(boundary_ids.len() < full_ids.len());
		assert_eq!(&full_ids[..boundary_ids.len()], &boundary_ids[..]);
	}

	// ------------------------------------------------------------------
	// Structured translation (TranslateGemma-style templates)
	// ------------------------------------------------------------------

	/// A TranslateGemma-shaped fixture: raises on plain-string content,
	/// consumes a single translation mapping, and embeds a ≥50-entry
	/// language table.
	fn translation_fixture_template() -> String {
		let mut table = String::from("{%- set languages = {\n");
		table.push_str("    \"en\": \"English\",\n    \"de\": \"German\",\n");
		for index in 0..60 {
			table.push_str(&format!("    \"x{index}\": \"Language {index}\",\n"));
		}
		table.push_str("}\n-%}\n");
		table.push_str(
			r#"{{ bos_token }}
{%- for message in messages -%}
{%- if message['role'] == 'user' -%}
{%- if message['content'] is none or message['content'] is string or message['content'] | length != 1 -%}
{{ raise_exception("User content must be a single translation mapping") }}
{%- endif -%}
{%- set content = message['content'][0] -%}
<start_of_turn>user
You are a professional {{ languages[content['source_lang_code']] }} to {{ languages[content['target_lang_code']] }} translator.
{{ content['text'] | trim }}<end_of_turn>
{%- elif message['role'] == 'assistant' -%}
<start_of_turn>model
{{ message['content'] | trim }}<end_of_turn>
{%- else -%}
{{ raise_exception("only user and assistant turns") }}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}<start_of_turn>model
{% endif -%}"#,
		);
		table
	}

	#[test]
	fn translation_content_renders_as_single_map_list() {
		let json = chat_message_to_json(
			&ChatMessage::user_translation("en", "de", "hello"),
			&std::collections::BTreeMap::new(),
		);
		assert_eq!(
			json["content"],
			serde_json::json!([{
				"type": "text",
				"source_lang_code": "en",
				"target_lang_code": "de",
				"text": "hello",
			}])
		);
	}

	#[test]
	fn probe_marks_translation_capability_on_translategemma_style_template() {
		let capabilities = probe_chat_template_capabilities(&translation_fixture_template())
			.expect("translation-only template resolves");
		assert!(capabilities.translation);
		assert!(!capabilities.chat);
		assert!(!capabilities.system_prompt);
		assert!(!capabilities.tools);
	}

	#[test]
	fn probe_translation_is_fail_soft_on_plain_chat_template() {
		// A plain template stringifies the translation mapping, leaking its
		// key names — the probe must not claim the translation task.
		let capabilities = probe_chat_template_capabilities(
			"{% for message in messages %}{{ message.content }}{% endfor %}",
		)
		.unwrap();
		assert!(capabilities.chat);
		assert!(!capabilities.translation);
	}

	#[test]
	fn dual_mode_template_reports_chat_and_translation() {
		let capabilities = probe_chat_template_capabilities(
			r"
{%- for message in messages -%}
{%- if message['content'] is string -%}
{{ message['content'] }}
{%- else -%}
{{ message['content'][0]['text'] }}
{%- endif -%}
{%- endfor -%}
",
		)
		.unwrap();
		assert!(capabilities.chat);
		assert!(capabilities.translation);
	}

	#[test]
	fn has_media_ignores_translation_content() {
		let message = ChatMessage::user_translation("en", "de", "hello");
		assert!(!message.has_media());
		assert!(
			ChatMessage::user_with_image("look", Vec::new()).has_media(),
			"image content is still media"
		);
	}

	#[test]
	fn translation_language_table_extracts_codes() {
		let table =
			translation_language_table(&translation_fixture_template()).expect("table present");
		assert!(table.len() >= 50);
		assert_eq!(table.get("en").map(String::as_str), Some("English"));
		assert_eq!(table.get("de").map(String::as_str), Some("German"));
	}

	#[test]
	fn translation_language_table_returns_none_without_table() {
		assert!(
			translation_language_table(
				"{% for message in messages %}{{ message.content }}{% endfor %}"
			)
			.is_none()
		);
	}

	#[test]
	fn translation_language_table_rejects_tiny_false_matches() {
		assert!(
			translation_language_table(
				r#"{%- set languages = {"en": "English", "de": "German"} -%}"#
			)
			.is_none()
		);
	}
}

fn extract_special_token(cfg: &Value, key: &str) -> Option<String> {
	match cfg.get(key) {
		Some(Value::String(s)) => Some(s.clone()),
		Some(Value::Object(o)) => o.get("content").and_then(|c| c.as_str()).map(String::from),
		_ => None,
	}
}
