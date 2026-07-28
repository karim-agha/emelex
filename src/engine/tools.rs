//! Tool ("function") calling: types passed into chat-template rendering,
//! plus parsers that recover structured `ToolCall`s from raw generated
//! text for the output conventions seen across the supported model
//! families:
//!
//! - **Hermes-style** `<tool_call>...</tool_call>` blocks (Qwen2/2.5/3/3.5/3.6,
//!   NemotronH), whose payload is either
//!   - JSON: `{"name": "...", "arguments": {...}}`, or
//!   - XML-function style (used by newer mlx-community/NVIDIA templates):
//!     `<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n...</function>`.
//! - **Gemma-native** key/value macros (Gemma4):
//!   `<|tool_call>call:NAME{key:value,...}<tool_call|>`.
//! - **Laguna-style** XML key/value blocks (Poolside Laguna M.1/S-2.1):
//!   `<tool_call>NAME\n<arg_key>KEY</arg_key>\n<arg_value>VALUE</arg_value>\n
//!   ...</tool_call>` — string values raw, non-string values JSON-encoded by
//!   the template (so JSON-coercible values parse as JSON).
//! - **Llama JSON** objects: `{"name":"NAME","parameters":{...}}`.

use std::{cmp::Ordering, fmt, ops::Range};

use serde::{
	Deserialize, Deserializer, Serialize,
	de::{MapAccess, SeqAccess, Visitor},
};
use serde_json::Value;

const MAX_TOOL_CALLS: usize = 64;
const MAX_TOOL_MARKERS: usize = 256;
const MAX_TOOL_ARGUMENTS: usize = 256;
const MAX_SCHEMA_ENUM_VALUES: usize = 256;
const MAX_SCHEMA_DEPTH: usize = 64;
const MAX_SCHEMA_NODES: usize = 4_096;
const MAX_INSTANCE_NODES: usize = 8_192;

/// One callable function's JSON-schema description (OpenAI `function` shape).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
	pub name: String,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub description: Option<String>,
	#[serde(default = "default_params")]
	pub parameters: Value,
}

fn default_params() -> Value {
	serde_json::json!({"type": "object", "properties": {}})
}

/// A tool declaration, OpenAI-style `{"type": "function", "function": {...}}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
	#[serde(rename = "type", default = "default_tool_type")]
	pub kind: String,
	pub function: ToolFunction,
}

fn default_tool_type() -> String {
	"function".to_string()
}

impl Tool {
	pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
		Tool {
			kind: "function".to_string(),
			function: ToolFunction {
				name: name.into(),
				description: Some(description.into()),
				parameters,
			},
		}
	}
}

/// A single parsed tool invocation from model output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
	pub id: String,
	pub name: String,
	pub arguments: Value,
}

/// Tool-call output convention resolved from one exact chat template.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallFormat {
	Hermes,
	Gemma,
	Laguna,
	/// Meta Llama-style raw JSON object with `name` and `parameters`.
	LlamaJson,
	/// Family has no documented tool-calling convention; parsing is a no-op.
	None,
}

/// Extract every tool call found in `text`, in order of appearance.
/// emelex patch: tool-call ids are process-unique, not per-generation -
/// per-generation `call_0` counters produced duplicate ids across the
/// turns of one conversation, which breaks consumers keying on the id.
fn next_call_id() -> String {
	use std::sync::atomic::{AtomicU64, Ordering};
	static COUNTER: AtomicU64 = AtomicU64::new(0);
	format!("call_{}", COUNTER.fetch_add(1, Ordering::Relaxed))
}

pub fn parse_tool_calls(text: &str, format: ToolCallFormat) -> Vec<ToolCall> {
	parse_tool_proposals(text, format)
		.into_iter()
		.map(ToolProposal::into_call)
		.collect()
}

/// Remove fully parsed tool-call spans from `text`, returning the
/// surrounding prose.
///
/// Malformed and truncated markup remains visible. Treating an opening
/// marker as sufficient proof would hide model output that was never a
/// valid invocation.
pub fn strip_tool_calls(text: &str, format: ToolCallFormat) -> String {
	let spans = parse_tool_proposals(text, format)
		.into_iter()
		.map(|proposal| proposal.span)
		.collect::<Vec<_>>();
	strip_spans(text, &spans)
}

/// Parse calls as untrusted proposals, keep only calls advertised for this
/// request whose arguments satisfy their JSON Schema, and strip only those
/// accepted spans.
///
/// emelex patch: model-produced names and arguments are never executable
/// merely because their surface syntax parsed.
pub(crate) fn parse_and_strip_tool_calls(
	text: &str,
	format: ToolCallFormat,
	tools: &[Tool],
) -> (String, Vec<ToolCall>) {
	let proposals = parse_tool_proposals(text, format);
	let mut spans = Vec::with_capacity(proposals.len());
	let mut calls = Vec::with_capacity(proposals.len());
	for proposal in proposals {
		if proposal_is_advertised_and_valid(&proposal, tools) {
			spans.push(proposal.span.clone());
			calls.push(proposal.into_call());
		}
	}
	(strip_spans(text, &spans), calls)
}

#[derive(Debug)]
struct ToolProposal {
	name: String,
	arguments: Value,
	span: Range<usize>,
}

impl ToolProposal {
	fn into_call(self) -> ToolCall {
		ToolCall {
			id: next_call_id(),
			name: self.name,
			arguments: self.arguments,
		}
	}
}

fn parse_tool_proposals(text: &str, format: ToolCallFormat) -> Vec<ToolProposal> {
	match format {
		ToolCallFormat::Hermes => parse_hermes(text),
		ToolCallFormat::Gemma => parse_gemma(text),
		ToolCallFormat::Laguna => parse_laguna(text),
		ToolCallFormat::LlamaJson => parse_llama_json(text),
		ToolCallFormat::None => Vec::new(),
	}
}

fn parse_llama_json(text: &str) -> Vec<ToolProposal> {
	let trimmed = text.trim();
	let start = text.len().saturating_sub(text.trim_start().len());
	let Some((name, arguments)) = parse_llama_json_fields(trimmed) else {
		return Vec::new();
	};
	vec![ToolProposal {
		name,
		arguments,
		span: start..start + trimmed.len(),
	}]
}

fn parse_llama_json_fields(raw: &str) -> Option<(String, Value)> {
	let Value::Object(mut object) = parse_strict_json(raw)? else {
		return None;
	};
	let Some(name) = object
		.remove("name")
		.and_then(|value| value.as_str().map(str::to_string))
	else {
		return None;
	};
	let Some(arguments) = object.remove("parameters") else {
		return None;
	};
	if !object.is_empty()
		|| !valid_tool_name(&name)
		|| !arguments.is_object()
		|| arguments
			.as_object()
			.is_some_and(|values| values.len() > MAX_TOOL_ARGUMENTS)
	{
		return None;
	}
	Some((name, arguments))
}

pub(crate) fn rendered_llama_history_contains_call(
	text: &str,
	expected_name: &str,
	expected_arguments: &Value,
) -> bool {
	const MAX_PROBE_CANDIDATE_BYTES: usize = 64 << 10;
	const MAX_PROBE_PARSED_BYTES: usize = 256 << 10;
	let mut starts = Vec::new();
	let mut ranges = Vec::new();
	let mut in_string = false;
	let mut escaped = false;
	for (index, byte) in text.bytes().enumerate() {
		if in_string {
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
			b'"' => in_string = true,
			b'{' => {
				if ranges.len().saturating_add(starts.len()) >= MAX_TOOL_MARKERS {
					return false;
				}
				starts.push(index);
			}
			b'}' => {
				if let Some(start) = starts.pop() {
					ranges.push(start..index + 1);
				}
			}
			_ => {}
		}
	}
	let mut parsed_bytes = 0_usize;
	for range in ranges {
		let candidate_bytes = range.len();
		if candidate_bytes > MAX_PROBE_CANDIDATE_BYTES {
			continue;
		}
		let Some(next) = parsed_bytes.checked_add(candidate_bytes) else {
			return false;
		};
		if next > MAX_PROBE_PARSED_BYTES {
			return false;
		}
		parsed_bytes = next;
		if text
			.get(range)
			.and_then(parse_llama_json_fields)
			.is_some_and(|(name, arguments)| {
				name == expected_name && arguments == *expected_arguments
			}) {
			return true;
		}
	}
	false
}

fn strip_spans(text: &str, spans: &[Range<usize>]) -> String {
	if spans.is_empty() {
		return text.to_string();
	}
	let mut out = String::with_capacity(text.len());
	let mut copied_through = 0;
	for span in spans {
		if span.start < copied_through || span.end > text.len() {
			continue;
		}
		out.push_str(&text[copied_through..span.start]);
		copied_through = span.end;
	}
	out.push_str(&text[copied_through..]);
	out
}

fn parse_hermes(text: &str) -> Vec<ToolProposal> {
	const OPEN: &str = "<tool_call>";
	const CLOSE: &str = "</tool_call>";
	let mut calls = Vec::new();
	let mut cursor = 0;
	let mut markers = 0;
	while calls.len() < MAX_TOOL_CALLS && markers < MAX_TOOL_MARKERS {
		let Some(relative_start) = text[cursor..].find(OPEN) else {
			break;
		};
		markers += 1;
		let start = cursor + relative_start;
		let payload_start = start + OPEN.len();
		let after_open = &text[payload_start..];
		let Some(end) = after_open.find(CLOSE) else {
			break;
		};
		let payload = after_open[..end].trim();
		if let Some((name, arguments)) = parse_hermes_payload(payload) {
			let span_end = payload_start + end + CLOSE.len();
			calls.push(ToolProposal {
				name,
				arguments,
				span: start..span_end,
			});
		}
		cursor = payload_start + end + CLOSE.len();
	}
	calls
}

/// A `<tool_call>` block's payload comes in one of two shapes, depending
/// on the checkpoint's chat template:
/// - JSON: `{"name": "...", "arguments": {...}}` (classic Hermes/Qwen).
/// - XML-function style (newer Qwen3.5/NemotronH templates from
///   mlx-community/NVIDIA):
///   `<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>...\n</function>`.
fn parse_hermes_payload(payload: &str) -> Option<(String, Value)> {
	if payload.starts_with('{') {
		let Value::Object(mut object) = parse_strict_json(payload)? else {
			return None;
		};
		let name = object.remove("name")?.as_str()?.to_string();
		let arguments = object.remove("arguments")?;
		if !object.is_empty()
			|| !valid_tool_name(&name)
			|| !arguments.is_object()
			|| arguments
				.as_object()
				.is_some_and(|arguments| arguments.len() > MAX_TOOL_ARGUMENTS)
		{
			return None;
		}
		return Some((name, arguments));
	}
	parse_xml_function(payload)
}

/// Parse the XML-function tool-call payload convention:
/// `<function=NAME><parameter=KEY>VALUE</parameter>...</function>`.
///
/// The outer function and every parameter must be closed and the payload
/// must be fully consumed. Parameter values are raw text; each one is
/// coerced through a duplicate-safe JSON parse when it forms a complete
/// JSON value and kept as a plain string otherwise.
fn parse_xml_function(payload: &str) -> Option<(String, Value)> {
	const FUNC_OPEN: &str = "<function=";
	const FUNC_CLOSE: &str = "</function>";
	const PARAM_OPEN: &str = "<parameter=";
	const PARAM_CLOSE: &str = "</parameter>";

	let payload = payload.trim();
	let without_close = payload.strip_suffix(FUNC_CLOSE)?;
	let after_open = without_close.strip_prefix(FUNC_OPEN)?;
	let name_end = after_open.find('>')?;
	let name = after_open[..name_end].trim().to_string();
	if !valid_tool_name(&name) {
		return None;
	}

	let mut obj = serde_json::Map::new();
	let mut cursor = after_open[name_end + 1..].trim_start();
	while !cursor.is_empty() {
		if obj.len() >= MAX_TOOL_ARGUMENTS {
			return None;
		}
		let after_param = cursor.strip_prefix(PARAM_OPEN)?;
		let key_end = after_param.find('>')?;
		let key = after_param[..key_end].trim().to_string();
		if !valid_argument_name(&key) || obj.contains_key(&key) {
			return None;
		}
		let value_body = &after_param[key_end + 1..];
		let value_end = value_body.find(PARAM_CLOSE)?;
		let raw = value_body[..value_end].trim();
		obj.insert(key, coerce_xml_value(raw));
		cursor = value_body[value_end + PARAM_CLOSE.len()..].trim_start();
	}
	Some((name, Value::Object(obj)))
}

/// Coerce a raw XML parameter value: valid standalone JSON (numbers,
/// booleans, null, arrays, objects, quoted strings) parses as such;
/// anything else stays a plain string.
fn coerce_xml_value(raw: &str) -> Value {
	parse_strict_json(raw).unwrap_or_else(|| Value::from(raw.to_string()))
}

fn parse_strict_json(raw: &str) -> Option<Value> {
	serde_json::from_str::<StrictValue>(raw)
		.ok()
		.map(|value| value.0)
}

struct StrictValue(Value);

impl<'de> Deserialize<'de> for StrictValue {
	fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
	where
		D: Deserializer<'de>,
	{
		deserializer.deserialize_any(StrictValueVisitor)
	}
}

struct StrictValueVisitor;

impl<'de> Visitor<'de> for StrictValueVisitor {
	type Value = StrictValue;

	fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter.write_str("a JSON value without duplicate object keys")
	}

	fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
		Ok(StrictValue(Value::Bool(value)))
	}

	fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
		Ok(StrictValue(Value::from(value)))
	}

	fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
		Ok(StrictValue(Value::from(value)))
	}

	fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
	where
		E: serde::de::Error,
	{
		serde_json::Number::from_f64(value)
			.map(Value::Number)
			.map(StrictValue)
			.ok_or_else(|| E::custom("non-finite JSON number"))
	}

	fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
		Ok(StrictValue(Value::String(value.to_string())))
	}

	fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
		Ok(StrictValue(Value::String(value)))
	}

	fn visit_none<E>(self) -> Result<Self::Value, E> {
		Ok(StrictValue(Value::Null))
	}

	fn visit_unit<E>(self) -> Result<Self::Value, E> {
		Ok(StrictValue(Value::Null))
	}

	fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
	where
		D: Deserializer<'de>,
	{
		StrictValue::deserialize(deserializer)
	}

	fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
	where
		A: SeqAccess<'de>,
	{
		let mut values = Vec::new();
		while let Some(value) = sequence.next_element::<StrictValue>()? {
			values.push(value.0);
		}
		Ok(StrictValue(Value::Array(values)))
	}

	fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
	where
		A: MapAccess<'de>,
	{
		let mut values = serde_json::Map::new();
		while let Some(key) = map.next_key::<String>()? {
			if values.contains_key(&key) {
				return Err(serde::de::Error::custom(format!(
					"duplicate object key {key:?}"
				)));
			}
			let value = map.next_value::<StrictValue>()?;
			values.insert(key, value.0);
		}
		Ok(StrictValue(Value::Object(values)))
	}
}

fn valid_tool_name(name: &str) -> bool {
	!name.is_empty()
		&& name.len() <= 64
		&& name
			.bytes()
			.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
}

fn valid_argument_name(name: &str) -> bool {
	!name.is_empty()
		&& name.len() <= 128
		&& !name.chars().any(|character| {
			character.is_control() || matches!(character, '<' | '>' | '{' | '}' | '[' | ']')
		})
}

/// Parse the Laguna tool-call convention (GLM-style XML key/value pairs):
/// `<tool_call>NAME\n<arg_key>KEY</arg_key>\n<arg_value>VALUE</arg_value>\n
/// ...</tool_call>`. The name is everything before the first `<arg_key>`
/// (canonically the first line). Values are raw text for strings and
/// JSON-encoded otherwise, so each is coerced through a JSON parse like the
/// XML-function convention. The outer call and every key/value pair must be
/// closed and fully consumed.
fn parse_laguna(text: &str) -> Vec<ToolProposal> {
	const OPEN: &str = "<tool_call>";
	const CLOSE: &str = "</tool_call>";
	let mut calls = Vec::new();
	let mut cursor = 0;
	let mut markers = 0;
	while calls.len() < MAX_TOOL_CALLS && markers < MAX_TOOL_MARKERS {
		let Some(relative_start) = text[cursor..].find(OPEN) else {
			break;
		};
		markers += 1;
		let start = cursor + relative_start;
		let payload_start = start + OPEN.len();
		let after_open = &text[payload_start..];
		let Some(end) = after_open.find(CLOSE) else {
			break;
		};
		let payload = &after_open[..end];
		if let Some((name, arguments)) = parse_laguna_payload(payload) {
			let span_end = payload_start + end + CLOSE.len();
			calls.push(ToolProposal {
				name,
				arguments,
				span: start..span_end,
			});
		}
		cursor = payload_start + end + CLOSE.len();
	}
	calls
}

fn parse_laguna_payload(payload: &str) -> Option<(String, Value)> {
	const KEY_OPEN: &str = "<arg_key>";
	const KEY_CLOSE: &str = "</arg_key>";
	const VALUE_OPEN: &str = "<arg_value>";
	const VALUE_CLOSE: &str = "</arg_value>";

	let payload = payload.trim();
	let name_end = payload.find(KEY_OPEN).unwrap_or(payload.len());
	let name = payload[..name_end].trim().to_string();
	if !valid_tool_name(&name) {
		return None;
	}

	let mut obj = serde_json::Map::new();
	let mut cursor = payload[name_end..].trim_start();
	while !cursor.is_empty() {
		if obj.len() >= MAX_TOOL_ARGUMENTS {
			return None;
		}
		let after_key = cursor.strip_prefix(KEY_OPEN)?;
		let key_end = after_key.find(KEY_CLOSE)?;
		let key = after_key[..key_end].trim().to_string();
		if !valid_argument_name(&key) || obj.contains_key(&key) {
			return None;
		}
		let after_key_close = after_key[key_end + KEY_CLOSE.len()..].trim_start();
		let value_body = after_key_close.strip_prefix(VALUE_OPEN)?;
		let value_end = value_body.find(VALUE_CLOSE)?;
		obj.insert(key, coerce_xml_value(value_body[..value_end].trim()));
		cursor = value_body[value_end + VALUE_CLOSE.len()..].trim_start();
	}
	Some((name, Value::Object(obj)))
}

/// Parse Gemma-native calls with a quote-, escape-, and nesting-aware grammar.
///
/// Quoted values always remain strings. Bare JSON primitives keep their
/// primitive type; bare non-JSON values are strings.
fn parse_gemma(text: &str) -> Vec<ToolProposal> {
	const OPEN: &str = "<|tool_call>call:";
	const CLOSE: &str = "<tool_call|>";
	let mut calls = Vec::new();
	let mut cursor = 0;
	let mut markers = 0;
	while calls.len() < MAX_TOOL_CALLS && markers < MAX_TOOL_MARKERS {
		let Some(relative_start) = text[cursor..].find(OPEN) else {
			break;
		};
		markers += 1;
		let start = cursor + relative_start;
		let body_start = start + OPEN.len();
		let body = &text[body_start..];
		let Some((name, arguments, consumed)) = parse_gemma_call(body) else {
			cursor = body_start;
			continue;
		};
		let after_call = &body[consumed..];
		let whitespace = after_call.len() - after_call.trim_start().len();
		let Some(after_close) = after_call[whitespace..].strip_prefix(CLOSE) else {
			cursor = body_start;
			continue;
		};
		let span_end = text.len() - after_close.len();
		calls.push(ToolProposal {
			name,
			arguments,
			span: start..span_end,
		});
		cursor = span_end;
	}
	calls
}

fn parse_gemma_call(body: &str) -> Option<(String, Value, usize)> {
	let brace_start = body.find('{')?;
	let name = body[..brace_start].trim().to_string();
	if !valid_tool_name(&name) {
		return None;
	}
	let mut parser = GemmaParser::new(body, brace_start);
	let arguments = parser.parse_object()?;
	Some((name, arguments, parser.position()))
}

struct GemmaParser<'a> {
	input: &'a str,
	position: usize,
	depth: usize,
	nodes: usize,
}

impl<'a> GemmaParser<'a> {
	fn new(input: &'a str, position: usize) -> Self {
		Self {
			input,
			position,
			depth: 0,
			nodes: 0,
		}
	}

	fn position(&self) -> usize {
		self.position
	}

	fn parse_object(&mut self) -> Option<Value> {
		self.enter_container()?;
		self.consume_char('{')?;
		self.skip_whitespace();
		let mut object = serde_json::Map::new();
		if self.consume_if('}') {
			self.leave_container();
			return Some(Value::Object(object));
		}
		loop {
			if object.len() >= MAX_TOOL_ARGUMENTS {
				return None;
			}
			let key = self.parse_key()?;
			if object.contains_key(&key) {
				return None;
			}
			self.skip_whitespace();
			self.consume_char(':')?;
			self.skip_whitespace();
			let value = self.parse_value()?;
			object.insert(key, value);
			self.skip_whitespace();
			if self.consume_if('}') {
				self.leave_container();
				return Some(Value::Object(object));
			}
			self.consume_char(',')?;
			self.skip_whitespace();
			if self.peek_char() == Some('}') {
				return None;
			}
		}
	}

	fn parse_array(&mut self) -> Option<Value> {
		self.enter_container()?;
		self.consume_char('[')?;
		self.skip_whitespace();
		let mut values = Vec::new();
		if self.consume_if(']') {
			self.leave_container();
			return Some(Value::Array(values));
		}
		loop {
			if values.len() >= MAX_TOOL_ARGUMENTS {
				return None;
			}
			values.push(self.parse_value()?);
			self.skip_whitespace();
			if self.consume_if(']') {
				self.leave_container();
				return Some(Value::Array(values));
			}
			self.consume_char(',')?;
			self.skip_whitespace();
			if self.peek_char() == Some(']') {
				return None;
			}
		}
	}

	fn parse_key(&mut self) -> Option<String> {
		self.skip_whitespace();
		let key = if self.remaining().starts_with("<|\"|>") {
			self.parse_special_quoted_string()?
		} else if matches!(self.peek_char(), Some('"' | '\'')) {
			self.parse_quoted_string()?
		} else {
			let start = self.position;
			while let Some(character) = self.peek_char() {
				if character == ':' {
					break;
				}
				if matches!(character, ',' | '{' | '}' | '[' | ']') {
					return None;
				}
				self.advance_char();
			}
			self.input[start..self.position].trim().to_string()
		};
		valid_argument_name(&key).then_some(key)
	}

	fn parse_value(&mut self) -> Option<Value> {
		self.nodes = self.nodes.checked_add(1)?;
		if self.nodes > MAX_INSTANCE_NODES {
			return None;
		}
		if self.remaining().starts_with("<|\"|>") {
			return self.parse_special_quoted_string().map(Value::String);
		}
		match self.peek_char()? {
			'"' | '\'' => self.parse_quoted_string().map(Value::String),
			'[' => self.parse_array(),
			'{' => self.parse_object(),
			_ => self.parse_bare_value(),
		}
	}

	fn parse_bare_value(&mut self) -> Option<Value> {
		let start = self.position;
		while let Some(character) = self.peek_char() {
			if matches!(character, ',' | ']' | '}') {
				break;
			}
			self.advance_char();
		}
		let raw = self.input[start..self.position].trim();
		if raw.is_empty() {
			return None;
		}
		match parse_strict_json(raw) {
			Some(value @ (Value::Null | Value::Bool(_) | Value::Number(_))) => Some(value),
			_ => Some(Value::String(raw.to_string())),
		}
	}

	fn parse_special_quoted_string(&mut self) -> Option<String> {
		const QUOTE: &str = "<|\"|>";
		self.consume_str(QUOTE)?;
		let end = self.remaining().find(QUOTE)?;
		let value = self.remaining()[..end].to_string();
		self.position += end + QUOTE.len();
		Some(value)
	}

	fn parse_quoted_string(&mut self) -> Option<String> {
		let quote = self.advance_char()?;
		let mut value = String::new();
		loop {
			let character = self.advance_char()?;
			if character == quote {
				return Some(value);
			}
			if character == '\\' {
				self.parse_escape(&mut value)?;
			} else if character.is_control() {
				return None;
			} else {
				value.push(character);
			}
		}
	}

	fn parse_escape(&mut self, output: &mut String) -> Option<()> {
		match self.advance_char()? {
			'"' => output.push('"'),
			'\'' => output.push('\''),
			'\\' => output.push('\\'),
			'/' => output.push('/'),
			'b' => output.push('\u{0008}'),
			'f' => output.push('\u{000c}'),
			'n' => output.push('\n'),
			'r' => output.push('\r'),
			't' => output.push('\t'),
			'u' => {
				let first = self.parse_hex_quad()?;
				let scalar = if (0xd800..=0xdbff).contains(&first) {
					self.consume_char('\\')?;
					self.consume_char('u')?;
					let second = self.parse_hex_quad()?;
					if !(0xdc00..=0xdfff).contains(&second) {
						return None;
					}
					0x1_0000 + ((u32::from(first) - 0xd800) << 10) + (u32::from(second) - 0xdc00)
				} else if (0xdc00..=0xdfff).contains(&first) {
					return None;
				} else {
					u32::from(first)
				};
				output.push(char::from_u32(scalar)?);
			}
			_ => return None,
		}
		Some(())
	}

	fn parse_hex_quad(&mut self) -> Option<u16> {
		let mut value = 0u16;
		for _ in 0..4 {
			let digit = self.advance_char()?.to_digit(16)?;
			value = value
				.checked_mul(16)?
				.checked_add(u16::try_from(digit).ok()?)?;
		}
		Some(value)
	}

	fn enter_container(&mut self) -> Option<()> {
		self.depth = self.depth.checked_add(1)?;
		(self.depth <= MAX_SCHEMA_DEPTH).then_some(())
	}

	fn leave_container(&mut self) {
		self.depth = self.depth.saturating_sub(1);
	}

	fn remaining(&self) -> &'a str {
		&self.input[self.position..]
	}

	fn peek_char(&self) -> Option<char> {
		self.remaining().chars().next()
	}

	fn advance_char(&mut self) -> Option<char> {
		let character = self.peek_char()?;
		self.position += character.len_utf8();
		Some(character)
	}

	fn skip_whitespace(&mut self) {
		while self.peek_char().is_some_and(char::is_whitespace) {
			self.advance_char();
		}
	}

	fn consume_if(&mut self, expected: char) -> bool {
		if self.peek_char() == Some(expected) {
			self.advance_char();
			true
		} else {
			false
		}
	}

	fn consume_char(&mut self, expected: char) -> Option<()> {
		self.consume_if(expected).then_some(())
	}

	fn consume_str(&mut self, expected: &str) -> Option<()> {
		self.remaining().starts_with(expected).then(|| {
			self.position += expected.len();
		})
	}
}

fn proposal_is_advertised_and_valid(proposal: &ToolProposal, tools: &[Tool]) -> bool {
	let mut matching = tools.iter().filter(|tool| {
		tool.kind == "function" && tool.function.name.as_str() == proposal.name.as_str()
	});
	let Some(tool) = matching.next() else {
		return false;
	};
	if matching.next().is_some()
		|| !proposal.arguments.is_object()
		|| validate_tool_schema(&tool.function.parameters).is_err()
	{
		return false;
	}
	arguments_satisfy_schema(&tool.function.parameters, &proposal.arguments)
}

/// Validate one bounded JSON instance against the executable schema contract.
pub(crate) fn arguments_satisfy_schema(schema: &Value, arguments: &Value) -> bool {
	if validate_tool_schema(schema).is_err() || !arguments.is_object() {
		return false;
	}
	let mut instance_nodes = MAX_INSTANCE_NODES;
	if !bounded_instance(arguments, 0, &mut instance_nodes) {
		return false;
	}
	let mut budget = ValidationBudget::new(MAX_INSTANCE_NODES);
	schema_accepts(schema, arguments, 0, &mut budget)
}

/// Validate the bounded JSON Schema vocabulary enforced for tool arguments.
///
/// References, regular-expression patterns, and unevaluated vocabularies are
/// rejected rather than silently ignored. Emelex currently accepts the core
/// structural, conditional, object, array, string-length, numeric-range, enum,
/// and constant constraints implemented by [`schema_accepts`].
pub(crate) fn validate_tool_schema(schema: &Value) -> Result<(), String> {
	let mut budget = ValidationBudget::new(MAX_SCHEMA_NODES);
	validate_schema_node(schema, 0, &mut budget)
}

struct ValidationBudget {
	remaining: usize,
}

impl ValidationBudget {
	fn new(limit: usize) -> Self {
		Self { remaining: limit }
	}

	fn consume(&mut self) -> bool {
		let Some(remaining) = self.remaining.checked_sub(1) else {
			return false;
		};
		self.remaining = remaining;
		true
	}
}

fn validate_schema_node(
	schema: &Value,
	depth: usize,
	budget: &mut ValidationBudget,
) -> Result<(), String> {
	if depth > MAX_SCHEMA_DEPTH || !budget.consume() {
		return Err("schema exceeds structural limits".to_string());
	}
	if schema.is_boolean() {
		return Ok(());
	}
	let object = schema
		.as_object()
		.ok_or_else(|| "schema nodes must be objects or booleans".to_string())?;
	for keyword in object.keys() {
		if !supported_schema_keyword(keyword) {
			return Err(format!(
				"schema keyword {keyword:?} is not supported for executable tools"
			));
		}
	}
	if let Some(types) = object.get("type") {
		validate_schema_types(types)?;
	}
	if let Some(values) = object.get("enum") {
		let values = values
			.as_array()
			.filter(|values| !values.is_empty() && values.len() <= MAX_SCHEMA_ENUM_VALUES)
			.ok_or_else(|| "enum must be a non-empty array".to_string())?;
		for (index, value) in values.iter().enumerate() {
			let mut nodes = MAX_INSTANCE_NODES;
			if !bounded_instance(value, 0, &mut nodes) {
				return Err("enum value exceeds structural limits".to_string());
			}
			if values[..index]
				.iter()
				.any(|previous| json_equal(previous, value))
			{
				return Err("enum values must be unique".to_string());
			}
		}
	}
	if let Some(value) = object.get("const") {
		let mut nodes = MAX_INSTANCE_NODES;
		if !bounded_instance(value, 0, &mut nodes) {
			return Err("const value exceeds structural limits".to_string());
		}
	}
	for keyword in ["allOf", "anyOf", "oneOf"] {
		if let Some(schemas) = object.get(keyword) {
			let schemas = schemas
				.as_array()
				.filter(|schemas| !schemas.is_empty())
				.ok_or_else(|| format!("{keyword} must be a non-empty array"))?;
			for child in schemas {
				validate_schema_node(child, depth + 1, budget)?;
			}
		}
	}
	for keyword in [
		"not",
		"if",
		"then",
		"else",
		"items",
		"contains",
		"propertyNames",
		"additionalProperties",
	] {
		if let Some(child) = object.get(keyword) {
			validate_schema_node(child, depth + 1, budget)
				.map_err(|error| format!("{keyword}: {error}"))?;
		}
	}
	for keyword in ["properties", "dependentSchemas"] {
		if let Some(children) = object.get(keyword) {
			let children = children
				.as_object()
				.ok_or_else(|| format!("{keyword} must be an object"))?;
			for child in children.values() {
				validate_schema_node(child, depth + 1, budget)?;
			}
		}
	}
	if let Some(children) = object.get("prefixItems") {
		let children = children
			.as_array()
			.ok_or_else(|| "prefixItems must be an array".to_string())?;
		for child in children {
			validate_schema_node(child, depth + 1, budget)?;
		}
	}
	validate_string_array_keyword(object, "required", true)?;
	validate_dependent_required(object)?;
	validate_nonnegative_integer_keywords(
		object,
		&[
			"minLength",
			"maxLength",
			"minItems",
			"maxItems",
			"minContains",
			"maxContains",
			"minProperties",
			"maxProperties",
		],
	)?;
	if let Some(unique) = object.get("uniqueItems")
		&& !unique.is_boolean()
	{
		return Err("uniqueItems must be a boolean".to_string());
	}
	for keyword in ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"] {
		if object.get(keyword).is_some_and(|value| !value.is_number()) {
			return Err(format!("{keyword} must be a number"));
		}
	}
	validate_min_max_pair(object, "minLength", "maxLength")?;
	validate_min_max_pair(object, "minItems", "maxItems")?;
	validate_min_max_pair(object, "minContains", "maxContains")?;
	validate_min_max_pair(object, "minProperties", "maxProperties")?;
	validate_numeric_bounds(object)?;
	validate_schema_annotations(object)?;
	Ok(())
}

fn supported_schema_keyword(keyword: &str) -> bool {
	matches!(
		keyword,
		"type"
			| "enum" | "const"
			| "allOf" | "anyOf"
			| "oneOf" | "not"
			| "if" | "then"
			| "else" | "properties"
			| "required"
			| "additionalProperties"
			| "propertyNames"
			| "minProperties"
			| "maxProperties"
			| "dependentRequired"
			| "dependentSchemas"
			| "prefixItems"
			| "items" | "contains"
			| "minContains"
			| "maxContains"
			| "minItems"
			| "maxItems"
			| "uniqueItems"
			| "minLength"
			| "maxLength"
			| "minimum"
			| "maximum"
			| "exclusiveMinimum"
			| "exclusiveMaximum"
			| "title" | "description"
			| "default"
			| "examples"
			| "$comment"
			| "deprecated"
			| "readOnly"
			| "writeOnly"
	)
}

fn validate_schema_annotations(object: &serde_json::Map<String, Value>) -> Result<(), String> {
	for keyword in ["title", "description", "$comment"] {
		if object.get(keyword).is_some_and(|value| !value.is_string()) {
			return Err(format!("{keyword} must be a string"));
		}
	}
	if object
		.get("examples")
		.is_some_and(|value| !value.is_array())
	{
		return Err("examples must be an array".to_string());
	}
	for value in object
		.get("examples")
		.and_then(Value::as_array)
		.into_iter()
		.flatten()
		.chain(object.get("default"))
	{
		let mut nodes = MAX_INSTANCE_NODES;
		if !bounded_instance(value, 0, &mut nodes) {
			return Err("schema annotation value exceeds structural limits".to_string());
		}
	}
	for keyword in ["deprecated", "readOnly", "writeOnly"] {
		if object.get(keyword).is_some_and(|value| !value.is_boolean()) {
			return Err(format!("{keyword} must be a boolean"));
		}
	}
	Ok(())
}

fn validate_schema_types(types: &Value) -> Result<(), String> {
	if let Some(kind) = types.as_str() {
		return valid_schema_type(kind)
			.then_some(())
			.ok_or_else(|| format!("unknown schema type {kind:?}"));
	}
	let kinds = types
		.as_array()
		.filter(|kinds| !kinds.is_empty() && kinds.len() <= 7)
		.ok_or_else(|| "type must be a string or non-empty array".to_string())?;
	let mut seen = Vec::with_capacity(kinds.len());
	for kind in kinds {
		let kind = kind
			.as_str()
			.ok_or_else(|| "type array entries must be strings".to_string())?;
		if !valid_schema_type(kind) {
			return Err(format!("unknown schema type {kind:?}"));
		}
		if seen.contains(&kind) {
			return Err("type array entries must be unique".to_string());
		}
		seen.push(kind);
	}
	Ok(())
}

fn valid_schema_type(kind: &str) -> bool {
	matches!(
		kind,
		"null" | "boolean" | "object" | "array" | "number" | "string" | "integer"
	)
}

fn validate_string_array_keyword(
	object: &serde_json::Map<String, Value>,
	keyword: &str,
	unique: bool,
) -> Result<(), String> {
	let Some(values) = object.get(keyword) else {
		return Ok(());
	};
	let values = values
		.as_array()
		.filter(|values| values.len() <= MAX_TOOL_ARGUMENTS)
		.ok_or_else(|| format!("{keyword} must be an array"))?;
	let mut seen = Vec::with_capacity(values.len());
	for value in values {
		let value = value
			.as_str()
			.ok_or_else(|| format!("{keyword} entries must be strings"))?;
		if unique && seen.contains(&value) {
			return Err(format!("{keyword} entries must be unique"));
		}
		seen.push(value);
	}
	Ok(())
}

fn validate_dependent_required(object: &serde_json::Map<String, Value>) -> Result<(), String> {
	let Some(dependencies) = object.get("dependentRequired") else {
		return Ok(());
	};
	let dependencies = dependencies
		.as_object()
		.ok_or_else(|| "dependentRequired must be an object".to_string())?;
	for required in dependencies.values() {
		let wrapper = serde_json::Map::from_iter([("required".to_string(), required.clone())]);
		validate_string_array_keyword(&wrapper, "required", true)?;
	}
	Ok(())
}

fn validate_nonnegative_integer_keywords(
	object: &serde_json::Map<String, Value>,
	keywords: &[&str],
) -> Result<(), String> {
	for keyword in keywords {
		if let Some(value) = object.get(*keyword)
			&& value.as_u64().is_none()
		{
			return Err(format!("{keyword} must be a non-negative integer"));
		}
	}
	Ok(())
}

fn validate_min_max_pair(
	object: &serde_json::Map<String, Value>,
	minimum: &str,
	maximum: &str,
) -> Result<(), String> {
	if let (Some(minimum), Some(maximum)) = (
		object.get(minimum).and_then(Value::as_u64),
		object.get(maximum).and_then(Value::as_u64),
	) && minimum > maximum
	{
		return Err("minimum constraint exceeds maximum constraint".to_string());
	}
	Ok(())
}

fn validate_numeric_bounds(object: &serde_json::Map<String, Value>) -> Result<(), String> {
	for (minimum, maximum) in [
		("minimum", "maximum"),
		("exclusiveMinimum", "exclusiveMaximum"),
	] {
		if let (Some(Value::Number(minimum)), Some(Value::Number(maximum))) =
			(object.get(minimum), object.get(maximum))
			&& number_cmp(minimum, maximum) == Some(Ordering::Greater)
		{
			return Err(format!("{minimum} exceeds {maximum}"));
		}
	}
	Ok(())
}

fn bounded_instance(value: &Value, depth: usize, nodes: &mut usize) -> bool {
	if depth > MAX_SCHEMA_DEPTH {
		return false;
	}
	let Some(remaining) = nodes.checked_sub(1) else {
		return false;
	};
	*nodes = remaining;
	match value {
		Value::Array(values) => {
			values.len() <= MAX_TOOL_ARGUMENTS
				&& values
					.iter()
					.all(|value| bounded_instance(value, depth + 1, nodes))
		}
		Value::Object(values) => {
			values.len() <= MAX_TOOL_ARGUMENTS
				&& values
					.values()
					.all(|value| bounded_instance(value, depth + 1, nodes))
		}
		Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => true,
	}
}

fn schema_accepts(
	schema: &Value,
	instance: &Value,
	depth: usize,
	budget: &mut ValidationBudget,
) -> bool {
	if depth > MAX_SCHEMA_DEPTH || !budget.consume() {
		return false;
	}
	if let Some(accepts) = schema.as_bool() {
		return accepts;
	}
	let Some(object) = schema.as_object() else {
		return false;
	};
	if object
		.get("const")
		.is_some_and(|expected| !json_equal(expected, instance))
		|| object.get("enum").is_some_and(|values| {
			values
				.as_array()
				.is_none_or(|values| !values.iter().any(|value| json_equal(value, instance)))
		}) || object
		.get("type")
		.is_some_and(|types| !schema_type_accepts(types, instance))
	{
		return false;
	}
	if !schema_array_matches(object.get("allOf"), |child| {
		schema_accepts(child, instance, depth + 1, budget)
	}) || !schema_array_any_matches(object.get("anyOf"), |child| {
		schema_accepts(child, instance, depth + 1, budget)
	}) || !schema_array_one_matches(object.get("oneOf"), |child| {
		schema_accepts(child, instance, depth + 1, budget)
	}) {
		return false;
	}
	if object
		.get("not")
		.is_some_and(|child| schema_accepts(child, instance, depth + 1, budget))
	{
		return false;
	}
	if let Some(condition) = object.get("if") {
		let condition_matches = schema_accepts(condition, instance, depth + 1, budget);
		let branch = if condition_matches { "then" } else { "else" };
		if object
			.get(branch)
			.is_some_and(|child| !schema_accepts(child, instance, depth + 1, budget))
		{
			return false;
		}
	}
	match instance {
		Value::Object(values) => schema_accepts_object(object, instance, values, depth, budget),
		Value::Array(instance) => schema_accepts_array(object, instance, depth, budget),
		Value::String(instance) => schema_accepts_string(object, instance),
		Value::Number(instance) => schema_accepts_number(object, instance),
		Value::Null | Value::Bool(_) => true,
	}
}

fn schema_array_matches(
	schemas: Option<&Value>,
	mut predicate: impl FnMut(&Value) -> bool,
) -> bool {
	schemas
		.and_then(Value::as_array)
		.is_none_or(|schemas| schemas.iter().all(&mut predicate))
}

fn schema_array_any_matches(
	schemas: Option<&Value>,
	mut predicate: impl FnMut(&Value) -> bool,
) -> bool {
	schemas
		.and_then(Value::as_array)
		.is_none_or(|schemas| schemas.iter().any(&mut predicate))
}

fn schema_array_one_matches(
	schemas: Option<&Value>,
	mut predicate: impl FnMut(&Value) -> bool,
) -> bool {
	schemas
		.and_then(Value::as_array)
		.is_none_or(|schemas| schemas.iter().filter(|schema| predicate(schema)).count() == 1)
}

fn schema_type_accepts(types: &Value, instance: &Value) -> bool {
	if let Some(kind) = types.as_str() {
		return schema_type_matches(kind, instance);
	}
	types.as_array().is_some_and(|types| {
		types.iter().any(|kind| {
			kind.as_str()
				.is_some_and(|kind| schema_type_matches(kind, instance))
		})
	})
}

fn schema_type_matches(kind: &str, instance: &Value) -> bool {
	match kind {
		"null" => instance.is_null(),
		"boolean" => instance.is_boolean(),
		"object" => instance.is_object(),
		"array" => instance.is_array(),
		"number" => instance.is_number(),
		"string" => instance.is_string(),
		"integer" => {
			instance.as_i64().is_some()
				|| instance.as_u64().is_some()
				|| instance
					.as_f64()
					.is_some_and(|number| number.fract() == 0.0)
		}
		_ => false,
	}
}

fn schema_accepts_object(
	schema: &serde_json::Map<String, Value>,
	instance_value: &Value,
	instance: &serde_json::Map<String, Value>,
	depth: usize,
	budget: &mut ValidationBudget,
) -> bool {
	if !within_usize_bounds(
		instance.len(),
		schema.get("minProperties"),
		schema.get("maxProperties"),
	) {
		return false;
	}
	if schema.get("required").is_some_and(|required| {
		required.as_array().is_none_or(|required| {
			required.iter().any(|name| {
				name.as_str()
					.is_none_or(|name| !instance.contains_key(name))
			})
		})
	}) {
		return false;
	}
	if let Some(property_names) = schema.get("propertyNames")
		&& instance.keys().any(|name| {
			!schema_accepts(
				property_names,
				&Value::String(name.clone()),
				depth + 1,
				budget,
			)
		}) {
		return false;
	}
	let properties = schema.get("properties").and_then(Value::as_object);
	for (name, value) in instance {
		if let Some(property) = properties.and_then(|properties| properties.get(name)) {
			if !schema_accepts(property, value, depth + 1, budget) {
				return false;
			}
		} else if let Some(additional) = schema.get("additionalProperties")
			&& !schema_accepts(additional, value, depth + 1, budget)
		{
			return false;
		}
	}
	if let Some(dependencies) = schema.get("dependentRequired").and_then(Value::as_object) {
		for (name, required) in dependencies {
			if instance.contains_key(name)
				&& required.as_array().is_none_or(|required| {
					required.iter().any(|required| {
						required
							.as_str()
							.is_none_or(|required| !instance.contains_key(required))
					})
				}) {
				return false;
			}
		}
	}
	if let Some(dependencies) = schema.get("dependentSchemas").and_then(Value::as_object) {
		for (name, dependent) in dependencies {
			if instance.contains_key(name)
				&& !schema_accepts(dependent, instance_value, depth + 1, budget)
			{
				return false;
			}
		}
	}
	true
}

fn schema_accepts_array(
	schema: &serde_json::Map<String, Value>,
	instance: &[Value],
	depth: usize,
	budget: &mut ValidationBudget,
) -> bool {
	if !within_usize_bounds(
		instance.len(),
		schema.get("minItems"),
		schema.get("maxItems"),
	) || schema
		.get("uniqueItems")
		.and_then(Value::as_bool)
		.is_some_and(|unique| {
			unique
				&& instance.iter().enumerate().any(|(index, value)| {
					instance[..index]
						.iter()
						.any(|previous| json_equal(previous, value))
				})
		}) {
		return false;
	}
	let prefix = schema
		.get("prefixItems")
		.and_then(Value::as_array)
		.map_or(&[][..], Vec::as_slice);
	for (value, child) in instance.iter().zip(prefix) {
		if !schema_accepts(child, value, depth + 1, budget) {
			return false;
		}
	}
	if let Some(items) = schema.get("items") {
		for value in instance.iter().skip(prefix.len()) {
			if !schema_accepts(items, value, depth + 1, budget) {
				return false;
			}
		}
	}
	if let Some(contains) = schema.get("contains") {
		let matches = instance
			.iter()
			.filter(|value| schema_accepts(contains, value, depth + 1, budget))
			.count();
		let minimum = schema
			.get("minContains")
			.and_then(Value::as_u64)
			.unwrap_or(1);
		let maximum = schema
			.get("maxContains")
			.and_then(Value::as_u64)
			.unwrap_or(u64::MAX);
		let Ok(matches) = u64::try_from(matches) else {
			return false;
		};
		if matches < minimum || matches > maximum {
			return false;
		}
	}
	true
}

fn schema_accepts_string(schema: &serde_json::Map<String, Value>, instance: &str) -> bool {
	within_usize_bounds(
		instance.chars().count(),
		schema.get("minLength"),
		schema.get("maxLength"),
	)
}

fn schema_accepts_number(
	schema: &serde_json::Map<String, Value>,
	instance: &serde_json::Number,
) -> bool {
	if schema
		.get("minimum")
		.and_then(Value::as_number)
		.is_some_and(|minimum| number_cmp(instance, minimum) == Some(Ordering::Less))
		|| schema
			.get("maximum")
			.and_then(Value::as_number)
			.is_some_and(|maximum| number_cmp(instance, maximum) == Some(Ordering::Greater))
		|| schema
			.get("exclusiveMinimum")
			.and_then(Value::as_number)
			.is_some_and(|minimum| {
				matches!(
					number_cmp(instance, minimum),
					Some(Ordering::Less | Ordering::Equal)
				)
			}) || schema
		.get("exclusiveMaximum")
		.and_then(Value::as_number)
		.is_some_and(|maximum| {
			matches!(
				number_cmp(instance, maximum),
				Some(Ordering::Greater | Ordering::Equal)
			)
		}) {
		return false;
	}
	true
}

fn within_usize_bounds(value: usize, minimum: Option<&Value>, maximum: Option<&Value>) -> bool {
	let value = u64::try_from(value).unwrap_or(u64::MAX);
	!minimum
		.and_then(Value::as_u64)
		.is_some_and(|minimum| value < minimum)
		&& !maximum
			.and_then(Value::as_u64)
			.is_some_and(|maximum| value > maximum)
}

fn json_equal(left: &Value, right: &Value) -> bool {
	match (left, right) {
		(Value::Number(left), Value::Number(right)) => {
			number_cmp(left, right) == Some(Ordering::Equal)
		}
		(Value::Array(left), Value::Array(right)) => {
			left.len() == right.len()
				&& left
					.iter()
					.zip(right)
					.all(|(left, right)| json_equal(left, right))
		}
		(Value::Object(left), Value::Object(right)) => {
			left.len() == right.len()
				&& left
					.iter()
					.all(|(key, left)| right.get(key).is_some_and(|right| json_equal(left, right)))
		}
		_ => left == right,
	}
}

#[derive(Clone, Copy)]
enum ExactNumber {
	Signed(i64),
	Unsigned(u64),
	Float(f64),
}

fn exact_number(number: &serde_json::Number) -> Option<ExactNumber> {
	if let Some(value) = number.as_i64() {
		Some(ExactNumber::Signed(value))
	} else if let Some(value) = number.as_u64() {
		Some(ExactNumber::Unsigned(value))
	} else {
		number.as_f64().map(ExactNumber::Float)
	}
}

fn number_cmp(left: &serde_json::Number, right: &serde_json::Number) -> Option<Ordering> {
	match (exact_number(left)?, exact_number(right)?) {
		(ExactNumber::Signed(left), ExactNumber::Signed(right)) => Some(left.cmp(&right)),
		(ExactNumber::Unsigned(left), ExactNumber::Unsigned(right)) => Some(left.cmp(&right)),
		(ExactNumber::Signed(left), ExactNumber::Unsigned(right)) => {
			Some(u64::try_from(left).map_or(Ordering::Less, |left| left.cmp(&right)))
		}
		(ExactNumber::Unsigned(left), ExactNumber::Signed(right)) => {
			Some(u64::try_from(right).map_or(Ordering::Greater, |right| left.cmp(&right)))
		}
		(ExactNumber::Signed(left), ExactNumber::Float(right)) => signed_float_cmp(left, right),
		(ExactNumber::Float(left), ExactNumber::Signed(right)) => {
			signed_float_cmp(right, left).map(Ordering::reverse)
		}
		(ExactNumber::Unsigned(left), ExactNumber::Float(right)) => unsigned_float_cmp(left, right),
		(ExactNumber::Float(left), ExactNumber::Unsigned(right)) => {
			unsigned_float_cmp(right, left).map(Ordering::reverse)
		}
		(ExactNumber::Float(left), ExactNumber::Float(right)) => left.partial_cmp(&right),
	}
}

fn signed_float_cmp(integer: i64, float: f64) -> Option<Ordering> {
	if !float.is_finite() {
		return None;
	}
	const I64_UPPER_EXCLUSIVE: f64 = 9_223_372_036_854_775_808.0;
	const I64_LOWER_INCLUSIVE: f64 = -9_223_372_036_854_775_808.0;
	if float < I64_LOWER_INCLUSIVE {
		return Some(Ordering::Greater);
	}
	if float >= I64_UPPER_EXCLUSIVE {
		return Some(Ordering::Less);
	}
	let truncated = float.trunc() as i64;
	match integer.cmp(&truncated) {
		Ordering::Equal if float.fract() > 0.0 => Some(Ordering::Less),
		Ordering::Equal if float.fract() < 0.0 => Some(Ordering::Greater),
		ordering => Some(ordering),
	}
}

fn unsigned_float_cmp(integer: u64, float: f64) -> Option<Ordering> {
	if !float.is_finite() {
		return None;
	}
	const U64_UPPER_EXCLUSIVE: f64 = 18_446_744_073_709_551_616.0;
	if float < 0.0 {
		return Some(Ordering::Greater);
	}
	if float >= U64_UPPER_EXCLUSIVE {
		return Some(Ordering::Less);
	}
	let truncated = float.trunc() as u64;
	match integer.cmp(&truncated) {
		Ordering::Equal if float.fract() > 0.0 => Some(Ordering::Less),
		ordering => Some(ordering),
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn advertised(name: &str, parameters: Value) -> Tool {
		Tool::new(name, "test tool", parameters)
	}

	#[test]
	fn hermes_json_call_is_fully_parsed() {
		let text = r#"<tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call>"#;
		let calls = parse_tool_calls(text, ToolCallFormat::Hermes);
		assert_eq!(
			calls.first().map(|call| (&call.name, &call.arguments)),
			Some((
				&"weather".to_string(),
				&serde_json::json!({"city": "Paris"})
			))
		);
	}

	#[test]
	fn llama_json_call_requires_the_complete_trimmed_generation() {
		let text = " \n{\"name\":\"weather\",\"parameters\":{\"city\":\"Paris\"}}\n ";
		let calls = parse_tool_calls(text, ToolCallFormat::LlamaJson);
		assert_eq!(
			calls.first().map(|call| (&call.name, &call.arguments)),
			Some((
				&"weather".to_string(),
				&serde_json::json!({"city": "Paris"})
			))
		);
		assert_eq!(strip_tool_calls(text, ToolCallFormat::LlamaJson), " \n\n ");
		assert!(
			parse_tool_calls(
				r#"Example: {"name":"weather","parameters":{"city":"Paris"}}"#,
				ToolCallFormat::LlamaJson
			)
			.is_empty()
		);
		assert!(
			parse_tool_calls(
				r#"{"outer":{"name":"weather","parameters":{"city":"Paris"}}}"#,
				ToolCallFormat::LlamaJson
			)
			.is_empty()
		);
	}

	#[test]
	fn llama_json_rejects_arguments_alias_extras_and_duplicates() {
		for text in [
			r#"{"name":"weather","arguments":{}}"#,
			r#"{"name":"weather","parameters":{},"extra":true}"#,
			r#"{"name":"weather","name":"other","parameters":{}}"#,
			r#"{"name":"weather","parameters":[]}"#,
		] {
			assert!(
				parse_tool_calls(text, ToolCallFormat::LlamaJson).is_empty(),
				"{text}"
			);
		}
	}

	#[test]
	fn hermes_multiple_calls_preserve_order() {
		let text = concat!(
			r#"<tool_call>{"name":"a","arguments":{}}</tool_call>"#,
			"text",
			r#"<tool_call>{"name":"b","arguments":{"x":1}}</tool_call>"#,
		);
		let calls = parse_tool_calls(text, ToolCallFormat::Hermes);
		assert_eq!(
			calls
				.iter()
				.map(|call| call.name.as_str())
				.collect::<Vec<_>>(),
			["a", "b"]
		);
	}

	#[test]
	fn hermes_malformed_json_is_rejected() {
		assert!(
			parse_tool_calls("<tool_call>{bad}</tool_call>", ToolCallFormat::Hermes).is_empty()
		);
	}

	#[test]
	fn hermes_duplicate_outer_field_is_rejected() {
		let text = r#"<tool_call>{"name":"a","name":"b","arguments":{}}</tool_call>"#;
		assert!(parse_tool_calls(text, ToolCallFormat::Hermes).is_empty());
	}

	#[test]
	fn hermes_duplicate_json_argument_is_rejected() {
		let text = r#"<tool_call>{"name":"a","arguments":{"x":1,"x":2}}</tool_call>"#;
		assert!(parse_tool_calls(text, ToolCallFormat::Hermes).is_empty());
	}

	#[test]
	fn hermes_non_object_arguments_are_rejected() {
		let text = r#"<tool_call>{"name":"a","arguments":[]}</tool_call>"#;
		assert!(parse_tool_calls(text, ToolCallFormat::Hermes).is_empty());
	}

	#[test]
	fn hermes_extra_outer_field_is_rejected() {
		let text = r#"<tool_call>{"name":"a","arguments":{},"extra":true}</tool_call>"#;
		assert!(parse_tool_calls(text, ToolCallFormat::Hermes).is_empty());
	}

	#[test]
	fn hermes_truncated_outer_call_is_rejected() {
		let text = r#"<tool_call>{"name":"a","arguments":{}}"#;
		assert!(parse_tool_calls(text, ToolCallFormat::Hermes).is_empty());
	}

	#[test]
	fn hermes_xml_typed_and_multiline_parameters_parse() {
		let text = concat!(
			"<tool_call><function=create_task>",
			"<parameter=priority>2</parameter>",
			"<parameter=urgent>true</parameter>",
			"<parameter=body>line one\nline two</parameter>",
			"</function></tool_call>",
		);
		let calls = parse_tool_calls(text, ToolCallFormat::Gemma);
		assert!(calls.is_empty());
		let calls = parse_tool_calls(text, ToolCallFormat::Hermes);
		assert_eq!(
			calls.first().map(|call| &call.arguments),
			Some(&serde_json::json!({
				"priority": 2,
				"urgent": true,
				"body": "line one\nline two"
			}))
		);
	}

	#[test]
	fn hermes_xml_missing_function_close_is_rejected() {
		let text = "<tool_call><function=a><parameter=x>1</parameter></tool_call>";
		assert!(parse_tool_calls(text, ToolCallFormat::Hermes).is_empty());
	}

	#[test]
	fn hermes_xml_missing_parameter_name_close_is_rejected() {
		let text = "<tool_call><function=a><parameter=x</function></tool_call>";
		assert!(parse_tool_calls(text, ToolCallFormat::Hermes).is_empty());
	}

	#[test]
	fn hermes_xml_missing_parameter_close_is_rejected() {
		let text = "<tool_call><function=a><parameter=x>1</function></tool_call>";
		assert!(parse_tool_calls(text, ToolCallFormat::Hermes).is_empty());
	}

	#[test]
	fn hermes_xml_duplicate_parameter_is_rejected() {
		let text = concat!(
			"<tool_call><function=a>",
			"<parameter=x>1</parameter><parameter=x>2</parameter>",
			"</function></tool_call>",
		);
		assert!(parse_tool_calls(text, ToolCallFormat::Hermes).is_empty());
	}

	#[test]
	fn hermes_xml_trailing_junk_is_rejected() {
		let text = "<tool_call><function=a></function>junk</tool_call>";
		assert!(parse_tool_calls(text, ToolCallFormat::Hermes).is_empty());
	}

	#[test]
	fn hermes_xml_quoted_primitive_remains_string() {
		let text = r#"<tool_call><function=a><parameter=x>"42"</parameter></function></tool_call>"#;
		let calls = parse_tool_calls(text, ToolCallFormat::Hermes);
		assert_eq!(
			calls.first().map(|call| &call.arguments["x"]),
			Some(&Value::from("42"))
		);
	}

	#[test]
	fn laguna_typed_and_multiline_arguments_parse() {
		let text = concat!(
			"<tool_call>create_task",
			"<arg_key>priority</arg_key><arg_value>2</arg_value>",
			"<arg_key>meta</arg_key><arg_value>{\"x\":1}</arg_value>",
			"<arg_key>body</arg_key><arg_value>one\ntwo</arg_value>",
			"</tool_call>",
		);
		let calls = parse_tool_calls(text, ToolCallFormat::Laguna);
		assert_eq!(
			calls.first().map(|call| &call.arguments),
			Some(&serde_json::json!({
				"priority": 2,
				"meta": {"x": 1},
				"body": "one\ntwo"
			}))
		);
	}

	#[test]
	fn laguna_truncated_outer_call_is_rejected() {
		let text = "<tool_call>a<arg_key>x</arg_key><arg_value>1</arg_value>";
		assert!(parse_tool_calls(text, ToolCallFormat::Laguna).is_empty());
	}

	#[test]
	fn laguna_missing_key_close_is_rejected() {
		let text = "<tool_call>a<arg_key>x<arg_value>1</arg_value></tool_call>";
		assert!(parse_tool_calls(text, ToolCallFormat::Laguna).is_empty());
	}

	#[test]
	fn laguna_missing_value_open_is_rejected() {
		let text = "<tool_call>a<arg_key>x</arg_key>1</arg_value></tool_call>";
		assert!(parse_tool_calls(text, ToolCallFormat::Laguna).is_empty());
	}

	#[test]
	fn laguna_missing_value_close_is_rejected() {
		let text = "<tool_call>a<arg_key>x</arg_key><arg_value>1</tool_call>";
		assert!(parse_tool_calls(text, ToolCallFormat::Laguna).is_empty());
	}

	#[test]
	fn laguna_duplicate_argument_is_rejected() {
		let text = concat!(
			"<tool_call>a",
			"<arg_key>x</arg_key><arg_value>1</arg_value>",
			"<arg_key>x</arg_key><arg_value>2</arg_value>",
			"</tool_call>",
		);
		assert!(parse_tool_calls(text, ToolCallFormat::Laguna).is_empty());
	}

	#[test]
	fn laguna_trailing_junk_is_rejected() {
		let text = "<tool_call>a<arg_key>x</arg_key><arg_value>1</arg_value>junk</tool_call>";
		assert!(parse_tool_calls(text, ToolCallFormat::Laguna).is_empty());
	}

	#[test]
	fn laguna_quoted_primitive_remains_string() {
		let text = r#"<tool_call>a<arg_key>x</arg_key><arg_value>"true"</arg_value></tool_call>"#;
		let calls = parse_tool_calls(text, ToolCallFormat::Laguna);
		assert_eq!(
			calls.first().map(|call| &call.arguments["x"]),
			Some(&Value::from("true"))
		);
	}

	#[test]
	fn gemma_nested_values_parse() {
		let text = "<|tool_call>call:sum{values:[1,2,{x:true}]}<tool_call|>";
		let calls = parse_tool_calls(text, ToolCallFormat::Gemma);
		assert_eq!(
			calls.first().map(|call| &call.arguments["values"]),
			Some(&serde_json::json!([1, 2, {"x": true}]))
		);
	}

	#[test]
	fn gemma_quoted_punctuation_is_not_structural() {
		let text = r#"<|tool_call>call:write{text:"a,b:c{d}[e]"}<tool_call|>"#;
		let calls = parse_tool_calls(text, ToolCallFormat::Gemma);
		assert_eq!(
			calls.first().map(|call| &call.arguments["text"]),
			Some(&Value::from("a,b:c{d}[e]"))
		);
	}

	#[test]
	fn gemma_escapes_and_surrogate_pair_parse() {
		let text = r#"<|tool_call>call:write{text:"line\nquote:\" \uD83D\uDE00"}<tool_call|>"#;
		let calls = parse_tool_calls(text, ToolCallFormat::Gemma);
		assert_eq!(
			calls.first().map(|call| &call.arguments["text"]),
			Some(&Value::from("line\nquote:\" 😀"))
		);
	}

	#[test]
	fn gemma_special_quote_contains_delimiters() {
		let text = "<|tool_call>call:write{text:<|\"|>a,b:c{d}<tool_call|><|\"|>}<tool_call|>";
		let calls = parse_tool_calls(text, ToolCallFormat::Gemma);
		assert_eq!(
			calls.first().map(|call| &call.arguments["text"]),
			Some(&Value::from("a,b:c{d}<tool_call|>"))
		);
	}

	#[test]
	fn gemma_quoted_primitives_remain_strings() {
		let text = r#"<|tool_call>call:a{x:"42",y:'true',z:"null"}<tool_call|>"#;
		let calls = parse_tool_calls(text, ToolCallFormat::Gemma);
		assert_eq!(
			calls.first().map(|call| &call.arguments),
			Some(&serde_json::json!({"x": "42", "y": "true", "z": "null"}))
		);
	}

	#[test]
	fn gemma_missing_colon_is_rejected() {
		let text = "<|tool_call>call:a{x 1}<tool_call|>";
		assert!(parse_tool_calls(text, ToolCallFormat::Gemma).is_empty());
	}

	#[test]
	fn gemma_duplicate_argument_is_rejected() {
		let text = "<|tool_call>call:a{x:1,x:2}<tool_call|>";
		assert!(parse_tool_calls(text, ToolCallFormat::Gemma).is_empty());
	}

	#[test]
	fn gemma_nested_duplicate_argument_is_rejected() {
		let text = "<|tool_call>call:a{x:{y:1,y:2}}<tool_call|>";
		assert!(parse_tool_calls(text, ToolCallFormat::Gemma).is_empty());
	}

	#[test]
	fn gemma_missing_outer_close_is_rejected() {
		let text = "<|tool_call>call:a{x:1}";
		assert!(parse_tool_calls(text, ToolCallFormat::Gemma).is_empty());
	}

	#[test]
	fn gemma_missing_object_close_is_rejected() {
		let text = "<|tool_call>call:a{x:1<tool_call|>";
		assert!(parse_tool_calls(text, ToolCallFormat::Gemma).is_empty());
	}

	#[test]
	fn gemma_unterminated_quote_is_rejected() {
		let text = r#"<|tool_call>call:a{x:"one}<tool_call|>"#;
		assert!(parse_tool_calls(text, ToolCallFormat::Gemma).is_empty());
	}

	#[test]
	fn gemma_unterminated_special_quote_is_rejected() {
		let text = "<|tool_call>call:a{x:<|\"|>one}<tool_call|>";
		assert!(parse_tool_calls(text, ToolCallFormat::Gemma).is_empty());
	}

	#[test]
	fn gemma_truncated_escape_is_rejected() {
		let text = r#"<|tool_call>call:a{x:"one\"#;
		assert!(parse_tool_calls(text, ToolCallFormat::Gemma).is_empty());
	}

	#[test]
	fn gemma_unterminated_list_is_rejected() {
		let text = "<|tool_call>call:a{x:[1,2}<tool_call|>";
		assert!(parse_tool_calls(text, ToolCallFormat::Gemma).is_empty());
	}

	#[test]
	fn gemma_unterminated_nested_object_is_rejected() {
		let text = "<|tool_call>call:a{x:{y:1}<tool_call|>";
		assert!(parse_tool_calls(text, ToolCallFormat::Gemma).is_empty());
	}

	#[test]
	fn gemma_trailing_comma_is_rejected() {
		let text = "<|tool_call>call:a{x:1,}<tool_call|>";
		assert!(parse_tool_calls(text, ToolCallFormat::Gemma).is_empty());
	}

	#[test]
	fn gemma_trailing_junk_is_rejected() {
		let text = "<|tool_call>call:a{x:1}junk<tool_call|>";
		assert!(parse_tool_calls(text, ToolCallFormat::Gemma).is_empty());
	}

	#[test]
	fn strip_preserves_surrounding_whitespace_exactly() {
		let text = " before <tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call> after ";
		assert_eq!(
			strip_tool_calls(text, ToolCallFormat::Hermes),
			" before  after "
		);
	}

	#[test]
	fn strip_keeps_truncated_hermes_markup_visible() {
		let text = "prefix<tool_call><function=a>";
		assert_eq!(strip_tool_calls(text, ToolCallFormat::Hermes), text);
	}

	#[test]
	fn strip_keeps_malformed_laguna_markup_visible() {
		let text = "prefix<tool_call>a<arg_key>x</arg_key></tool_call>";
		assert_eq!(strip_tool_calls(text, ToolCallFormat::Laguna), text);
	}

	#[test]
	fn strip_keeps_malformed_gemma_markup_visible() {
		let text = "prefix<|tool_call>call:a{x:1,}<tool_call|>";
		assert_eq!(strip_tool_calls(text, ToolCallFormat::Gemma), text);
	}

	#[test]
	fn malformed_span_does_not_hide_later_valid_call() {
		let text = concat!(
			"<tool_call>{bad}</tool_call>",
			r#"<tool_call>{"name":"a","arguments":{}}</tool_call>"#,
		);
		let calls = parse_tool_calls(text, ToolCallFormat::Hermes);
		assert_eq!(
			calls
				.iter()
				.map(|call| call.name.as_str())
				.collect::<Vec<_>>(),
			["a"]
		);
	}

	#[test]
	fn advertised_schema_valid_call_is_returned_and_stripped() {
		let text = r#"say <tool_call>{"name":"lookup","arguments":{"key":"x"}}</tool_call> done"#;
		let tools = [advertised(
			"lookup",
			serde_json::json!({
				"type": "object",
				"properties": {"key": {"type": "string"}},
				"required": ["key"],
				"additionalProperties": false
			}),
		)];
		let (visible, calls) = parse_and_strip_tool_calls(text, ToolCallFormat::Hermes, &tools);
		assert_eq!((visible, calls.len()), ("say  done".to_string(), 1));
	}

	#[test]
	fn unknown_tool_is_not_returned_or_stripped() {
		let text = r#"<tool_call>{"name":"unknown","arguments":{}}</tool_call>"#;
		let tools = [advertised("known", serde_json::json!({"type": "object"}))];
		let (visible, calls) = parse_and_strip_tool_calls(text, ToolCallFormat::Hermes, &tools);
		assert_eq!((visible, calls), (text.to_string(), Vec::new()));
	}

	#[test]
	fn required_argument_failure_is_not_returned_or_stripped() {
		let text = r#"<tool_call>{"name":"lookup","arguments":{}}</tool_call>"#;
		let tools = [advertised(
			"lookup",
			serde_json::json!({
				"type": "object",
				"required": ["key"]
			}),
		)];
		let (visible, calls) = parse_and_strip_tool_calls(text, ToolCallFormat::Hermes, &tools);
		assert_eq!((visible, calls), (text.to_string(), Vec::new()));
	}

	#[test]
	fn wrong_argument_type_is_not_returned_or_stripped() {
		let text = r#"<tool_call>{"name":"lookup","arguments":{"key":7}}</tool_call>"#;
		let tools = [advertised(
			"lookup",
			serde_json::json!({
				"type": "object",
				"properties": {"key": {"type": "string"}}
			}),
		)];
		let (visible, calls) = parse_and_strip_tool_calls(text, ToolCallFormat::Hermes, &tools);
		assert_eq!((visible, calls), (text.to_string(), Vec::new()));
	}

	#[test]
	fn extra_argument_is_not_returned_or_stripped() {
		let text = r#"<tool_call>{"name":"lookup","arguments":{"key":"x","extra":1}}</tool_call>"#;
		let tools = [advertised(
			"lookup",
			serde_json::json!({
				"type": "object",
				"properties": {"key": {"type": "string"}},
				"additionalProperties": false
			}),
		)];
		let (visible, calls) = parse_and_strip_tool_calls(text, ToolCallFormat::Hermes, &tools);
		assert_eq!((visible, calls), (text.to_string(), Vec::new()));
	}

	#[test]
	fn nested_schema_failure_is_not_returned_or_stripped() {
		let text =
			r#"<tool_call>{"name":"lookup","arguments":{"meta":{"count":"one"}}}</tool_call>"#;
		let tools = [advertised(
			"lookup",
			serde_json::json!({
				"type": "object",
				"properties": {
					"meta": {
						"type": "object",
						"properties": {"count": {"type": "integer"}}
					}
				}
			}),
		)];
		let (visible, calls) = parse_and_strip_tool_calls(text, ToolCallFormat::Hermes, &tools);
		assert_eq!((visible, calls), (text.to_string(), Vec::new()));
	}

	#[test]
	fn array_item_schema_failure_is_not_returned_or_stripped() {
		let text = r#"<tool_call>{"name":"sum","arguments":{"values":[1,"two"]}}</tool_call>"#;
		let tools = [advertised(
			"sum",
			serde_json::json!({
				"type": "object",
				"properties": {
					"values": {"type": "array", "items": {"type": "integer"}}
				}
			}),
		)];
		let (visible, calls) = parse_and_strip_tool_calls(text, ToolCallFormat::Hermes, &tools);
		assert_eq!((visible, calls), (text.to_string(), Vec::new()));
	}

	#[test]
	fn unconstrained_nested_collection_over_limit_is_rejected() {
		let proposal = ToolProposal {
			name: "lookup".to_string(),
			arguments: serde_json::json!({"values": vec![0; MAX_TOOL_ARGUMENTS + 1]}),
			span: 0..0,
		};
		let tools = [advertised("lookup", serde_json::json!({"type": "object"}))];
		assert!(!proposal_is_advertised_and_valid(&proposal, &tools));
	}

	#[test]
	fn unsupported_schema_is_rejected() {
		let error = validate_tool_schema(&serde_json::json!({
			"type": "object",
			"properties": {"key": {"type": "string", "pattern": "^safe$"}}
		}));
		assert!(error.is_err());
	}

	#[test]
	fn unknown_schema_keywords_are_rejected_fail_closed() {
		let error = validate_tool_schema(&serde_json::json!({
			"type": "string",
			"format": "uri"
		}));
		assert!(
			error
				.expect_err("unsupported format")
				.contains("\"format\"")
		);
	}

	#[test]
	fn inexact_multiple_of_keyword_is_rejected_fail_closed() {
		let error = validate_tool_schema(&serde_json::json!({
			"type": "number",
			"multipleOf": 0.1
		}));
		assert!(
			error
				.expect_err("unsupported multipleOf")
				.contains("\"multipleOf\"")
		);
	}

	#[test]
	fn adjacent_large_integers_remain_distinct_in_const_and_bounds() {
		let exact = 9_007_199_254_740_993_u64;
		let lower = exact - 1;
		let const_schema = serde_json::json!({
			"type": "object",
			"properties": {"value": {"type": "integer", "const": exact}},
			"required": ["value"],
			"additionalProperties": false
		});
		assert!(arguments_satisfy_schema(
			&const_schema,
			&serde_json::json!({"value": exact})
		));
		assert!(!arguments_satisfy_schema(
			&const_schema,
			&serde_json::json!({"value": lower})
		));

		let minimum_schema = serde_json::json!({
			"type": "object",
			"properties": {"value": {"type": "integer", "minimum": exact}},
			"required": ["value"],
			"additionalProperties": false
		});
		assert!(arguments_satisfy_schema(
			&minimum_schema,
			&serde_json::json!({"value": exact})
		));
		assert!(!arguments_satisfy_schema(
			&minimum_schema,
			&serde_json::json!({"value": lower})
		));
	}

	#[test]
	fn annotation_keywords_do_not_weaken_argument_validation() {
		let schema = serde_json::json!({
			"type": "object",
			"title": "Lookup",
			"description": "One bounded lookup",
			"default": {},
			"examples": [{"key": "example"}],
			"readOnly": false,
			"properties": {"key": {"type": "string", "minLength": 2}},
			"required": ["key"],
			"additionalProperties": false
		});
		assert!(validate_tool_schema(&schema).is_ok());
		assert!(arguments_satisfy_schema(
			&schema,
			&serde_json::json!({"key": "ok"})
		));
		assert!(!arguments_satisfy_schema(
			&schema,
			&serde_json::json!({"key": "x"})
		));
	}

	#[test]
	fn oversized_enum_schema_is_rejected() {
		let error = validate_tool_schema(&serde_json::json!({
			"enum": vec![Value::Null; MAX_SCHEMA_ENUM_VALUES + 1]
		}));
		assert!(error.is_err());
	}
}
