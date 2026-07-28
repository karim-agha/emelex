//! Incremental classification of generated text into reasoning, answer,
//! and tool-call spans.
//!
//! Marker syntax may cross tokenizer pieces, or one piece may contain both
//! sides of a boundary. [`StreamClassifier`] therefore acts as a small
//! transducer: it withholds only a possible marker suffix, removes complete
//! wire markers, and returns zero or more lossless typed text segments.

use crate::engine::{
	reasoning::{MARKER_PAIRS, prefix_candidate},
	tools::ToolCallFormat,
};

/// Which part of a reply a streamed text segment belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
	/// Ordinary reply text.
	Text,
	/// Text inside a reasoning span.
	Reasoning,
	/// Raw tool-call payload text.
	ToolCall,
}

/// One marker-free display segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ClassifiedText {
	pub kind: TokenKind,
	pub text: String,
}

#[derive(Debug, Clone, Copy)]
enum Boundary {
	ReasoningOpen { close: &'static str },
	ReasoningClose,
	ToolOpen,
	ToolClose,
}

/// Stateful marker transducer.
pub struct StreamClassifier {
	tool_open: &'static str,
	tool_close: &'static str,
	state: TokenKind,
	reasoning_close: &'static str,
	reasoning_prefix_possible: bool,
	raw_pending: String,
	display_pending: String,
	defer_until_terminal: bool,
}

impl StreamClassifier {
	pub fn new(tool_format: ToolCallFormat) -> Self {
		let (tool_open, tool_close) = match tool_format {
			ToolCallFormat::Hermes | ToolCallFormat::Laguna => ("<tool_call>", "</tool_call>"),
			ToolCallFormat::Gemma => ("<|tool_call>", "<tool_call|>"),
			ToolCallFormat::LlamaJson | ToolCallFormat::None => ("", ""),
		};
		let defer_until_terminal = tool_format == ToolCallFormat::LlamaJson;
		Self {
			tool_open,
			tool_close,
			state: TokenKind::Text,
			reasoning_close: "",
			reasoning_prefix_possible: true,
			raw_pending: String::new(),
			display_pending: String::new(),
			defer_until_terminal,
		}
	}

	/// Seed a reasoning span opened by the rendered prompt.
	pub(crate) fn seed_reasoning(&mut self, close: &'static str) {
		self.state = TokenKind::Reasoning;
		self.reasoning_close = close;
		self.reasoning_prefix_possible = false;
		self.raw_pending.clear();
		self.display_pending.clear();
	}

	pub(crate) const fn current_kind(&self) -> TokenKind {
		self.state
	}

	/// Consume corresponding raw and display-decoded text.
	///
	/// `raw` retains special tokens so marker detection is reliable;
	/// `display` is the tokenizer's user-facing decode for the same token
	/// IDs. Returned segments never contain recognized marker syntax.
	pub(crate) fn push(&mut self, raw: &str, display: &str) -> Vec<ClassifiedText> {
		if self.defer_until_terminal {
			return vec![ClassifiedText {
				kind: TokenKind::ToolCall,
				text: display.to_string(),
			}];
		}
		self.raw_pending.push_str(raw);
		self.display_pending.push_str(display);
		let mut output = Vec::new();

		loop {
			if let Some((at, marker, boundary)) = self.find_boundary() {
				let (before, after) =
					split_display_at_marker(&self.raw_pending, &self.display_pending, at, marker);
				if !matches!(boundary, Boundary::ReasoningOpen { .. }) {
					push_nonempty(&mut output, self.state, before);
				}
				let raw_after = self.raw_pending[at + marker.len()..].to_string();
				self.raw_pending = raw_after;
				self.display_pending = after;
				self.apply_boundary(boundary);
				if matches!(boundary, Boundary::ToolOpen) {
					// Preserve a structural signal even when an opening and
					// closing marker enclose no payload in one decoded piece.
					// Stream bridges use it to withhold later text until the
					// terminal parser has validated the proposed call.
					output.push(ClassifiedText {
						kind: TokenKind::ToolCall,
						text: String::new(),
					});
				}
				continue;
			}

			let retained = self.partial_marker_suffix_len();
			let safe = self.raw_pending.len().saturating_sub(retained);
			if safe == 0 {
				break;
			}
			let (visible, retained_display) =
				split_display_for_safe_prefix(&self.raw_pending, &self.display_pending, safe);
			let safe_raw = self.raw_pending[..safe].to_string();
			self.raw_pending = self.raw_pending[safe..].to_string();
			self.display_pending = retained_display;
			push_nonempty(&mut output, self.state, visible);
			if self.state == TokenKind::Text && !safe_raw.trim().is_empty() {
				self.reasoning_prefix_possible = false;
			}
			if retained == 0 {
				break;
			}
		}
		output
	}

	/// Flush a terminal partial marker as ordinary span text.
	pub(crate) fn finish(&mut self) -> Vec<ClassifiedText> {
		if self.defer_until_terminal {
			return Vec::new();
		}
		let display = std::mem::take(&mut self.display_pending);
		let raw = std::mem::take(&mut self.raw_pending);
		if self.state == TokenKind::Text && !raw.trim().is_empty() {
			self.reasoning_prefix_possible = false;
		}
		let mut output = Vec::new();
		push_nonempty(&mut output, self.state, display);
		output
	}

	fn find_boundary(&self) -> Option<(usize, &'static str, Boundary)> {
		let mut found = None;
		match self.state {
			TokenKind::Text => {
				if self.reasoning_prefix_possible {
					if let Some(candidate) = prefix_candidate(&self.raw_pending) {
						let leading = self.raw_pending.len() - candidate.len();
						for (open, close) in MARKER_PAIRS {
							if candidate.starts_with(open) {
								choose_earlier(
									&mut found,
									leading,
									open,
									Boundary::ReasoningOpen { close },
								);
							}
						}
					}
				}
				if !self.tool_open.is_empty()
					&& let Some(at) = self.raw_pending.find(self.tool_open)
				{
					choose_earlier(&mut found, at, self.tool_open, Boundary::ToolOpen);
				}
			}
			TokenKind::Reasoning => {
				if let Some(at) = self.raw_pending.find(self.reasoning_close) {
					found = Some((at, self.reasoning_close, Boundary::ReasoningClose));
				}
			}
			TokenKind::ToolCall => {
				if let Some(at) = self.raw_pending.find(self.tool_close) {
					found = Some((at, self.tool_close, Boundary::ToolClose));
				}
			}
		}
		found
	}

	fn partial_marker_suffix_len(&self) -> usize {
		let mut retained = 0;
		match self.state {
			TokenKind::Text => {
				if self.reasoning_prefix_possible {
					if prefix_candidate(&self.raw_pending).is_some_and(|candidate| {
						candidate.is_empty()
							|| MARKER_PAIRS
								.iter()
								.any(|(open, _)| open.starts_with(candidate))
					}) {
						retained = self.raw_pending.len();
					}
				}
				retained = retained.max(marker_suffix_len(&self.raw_pending, self.tool_open));
			}
			TokenKind::Reasoning => {
				retained = marker_suffix_len(&self.raw_pending, self.reasoning_close);
			}
			TokenKind::ToolCall => {
				retained = marker_suffix_len(&self.raw_pending, self.tool_close);
			}
		}
		retained
	}

	fn apply_boundary(&mut self, boundary: Boundary) {
		match boundary {
			Boundary::ReasoningOpen { close } => {
				self.state = TokenKind::Reasoning;
				self.reasoning_close = close;
				self.reasoning_prefix_possible = false;
			}
			Boundary::ReasoningClose => {
				self.state = TokenKind::Text;
				self.reasoning_close = "";
				self.reasoning_prefix_possible = false;
			}
			Boundary::ToolOpen => {
				self.state = TokenKind::ToolCall;
				self.reasoning_prefix_possible = false;
			}
			Boundary::ToolClose => {
				self.state = TokenKind::Text;
				self.reasoning_prefix_possible = false;
			}
		}
	}
}

fn choose_earlier(
	found: &mut Option<(usize, &'static str, Boundary)>,
	at: usize,
	marker: &'static str,
	boundary: Boundary,
) {
	if found.as_ref().is_none_or(|(current, _, _)| at < *current) {
		*found = Some((at, marker, boundary));
	}
}

fn marker_suffix_len(text: &str, marker: &str) -> usize {
	if marker.is_empty() {
		return 0;
	}
	let max = marker.len().saturating_sub(1).min(text.len());
	(1..=max)
		.rev()
		.find(|&length| text.ends_with(&marker[..length]))
		.unwrap_or(0)
}

fn split_display_at_marker(raw: &str, display: &str, at: usize, marker: &str) -> (String, String) {
	if raw == display {
		return (
			display[..at].to_string(),
			display[at + marker.len()..].to_string(),
		);
	}
	if let Some(display_at) = display.find(marker) {
		return (
			display[..display_at].to_string(),
			display[display_at + marker.len()..].to_string(),
		);
	}

	let raw_before = &raw[..at];
	let (before, remaining) = display.strip_prefix(raw_before).map_or_else(
		|| (String::new(), display),
		|remaining| (raw_before.to_string(), remaining),
	);
	let marker_display_len = (1..=marker.len().min(remaining.len()))
		.rev()
		.find(|&length| {
			remaining.is_char_boundary(length) && marker.ends_with(&remaining[..length])
		})
		.unwrap_or(0);
	(before, remaining[marker_display_len..].to_string())
}

fn split_display_for_safe_prefix(raw: &str, display: &str, safe: usize) -> (String, String) {
	if raw == display {
		return (display[..safe].to_string(), display[safe..].to_string());
	}
	if safe == raw.len() {
		return (display.to_string(), String::new());
	}
	if safe == 0 {
		return (String::new(), display.to_string());
	}
	let raw_safe = &raw[..safe];
	if let Some(remaining) = display.strip_prefix(raw_safe) {
		return (raw_safe.to_string(), remaining.to_string());
	}
	let raw_tail = &raw[safe..];
	if let Some(visible) = display.strip_suffix(raw_tail) {
		return (visible.to_string(), raw_tail.to_string());
	}
	// A retained special-token marker prefix can be absent from display
	// decoding. In that case every visible byte belongs to the safe prefix.
	(display.to_string(), String::new())
}

fn push_nonempty(output: &mut Vec<ClassifiedText>, kind: TokenKind, text: String) {
	if text.is_empty() {
		return;
	}
	if let Some(last) = output.last_mut()
		&& last.kind == kind
	{
		last.text.push_str(&text);
	} else {
		output.push(ClassifiedText { kind, text });
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn classify(format: ToolCallFormat, pieces: &[(&str, &str)]) -> Vec<(TokenKind, String)> {
		let mut classifier = StreamClassifier::new(format);
		let mut output = Vec::new();
		for (raw, display) in pieces {
			output.extend(
				classifier
					.push(raw, display)
					.into_iter()
					.map(|segment| (segment.kind, segment.text)),
			);
		}
		output.extend(
			classifier
				.finish()
				.into_iter()
				.map(|segment| (segment.kind, segment.text)),
		);
		output
	}

	#[test]
	fn plain_text_is_preserved() {
		let output = classify(
			ToolCallFormat::None,
			&[("hello", "hello"), (" ", " "), ("world", "world")],
		);
		assert_eq!(
			output.into_iter().map(|(_, text)| text).collect::<String>(),
			"hello world"
		);
	}

	#[test]
	fn llama_json_defers_every_delta_until_terminal_validation() {
		let output = classify(
			ToolCallFormat::LlamaJson,
			&[
				(r#"{"name":"lookup","#, r#"{"name":"lookup","#),
				(r#""parameters":{}}"#, r#""parameters":{}}"#),
			],
		);
		assert_eq!(
			output,
			vec![
				(TokenKind::ToolCall, r#"{"name":"lookup","#.to_string()),
				(TokenKind::ToolCall, r#""parameters":{}}"#.to_string())
			]
		);
	}

	#[test]
	fn split_reasoning_markers_do_not_leak() {
		let output = classify(
			ToolCallFormat::None,
			&[
				("<thi", "<thi"),
				("nk>reason", "nk>reason"),
				("</thi", "</thi"),
				("nk>answer", "nk>answer"),
			],
		);
		assert_eq!(
			output,
			vec![
				(TokenKind::Reasoning, "reason".to_string()),
				(TokenKind::Text, "answer".to_string()),
			]
		);
	}

	#[test]
	fn one_piece_can_cross_both_reasoning_boundaries() {
		let output = classify(
			ToolCallFormat::None,
			&[(
				"<think>private</think>public",
				"<think>private</think>public",
			)],
		);
		assert_eq!(
			output,
			vec![
				(TokenKind::Reasoning, "private".to_string()),
				(TokenKind::Text, "public".to_string()),
			]
		);
	}

	#[test]
	fn every_two_piece_reasoning_split_is_lossless() {
		let input = "<think>private</think>public";
		for split in 0..=input.len() {
			let output = classify(
				ToolCallFormat::None,
				&[
					(&input[..split], &input[..split]),
					(&input[split..], &input[split..]),
				],
			);
			let reasoning = output
				.iter()
				.filter(|(kind, _)| *kind == TokenKind::Reasoning)
				.map(|(_, text)| text.as_str())
				.collect::<String>();
			let answer = output
				.iter()
				.filter(|(kind, _)| *kind == TokenKind::Text)
				.map(|(_, text)| text.as_str())
				.collect::<String>();
			assert_eq!(
				(reasoning.as_str(), answer.as_str()),
				("private", "public"),
				"split at byte {split}"
			);
			assert!(
				output.iter().all(|(kind, _)| *kind != TokenKind::ToolCall),
				"split at byte {split}"
			);
		}
	}

	#[test]
	fn streamed_reasoning_fields_equal_terminal_extraction() {
		for input in [
			"<think>\nthought\n</think>\n\nanswer",
			"<think>\n</think>answer",
		] {
			for split in 0..=input.len() {
				let output = classify(
					ToolCallFormat::None,
					&[
						(&input[..split], &input[..split]),
						(&input[split..], &input[split..]),
					],
				);
				let streamed_reasoning = output
					.iter()
					.filter(|(kind, _)| *kind == TokenKind::Reasoning)
					.map(|(_, text)| text.as_str())
					.collect::<String>();
				let streamed_answer = output
					.iter()
					.filter(|(kind, _)| *kind == TokenKind::Text)
					.map(|(_, text)| text.as_str())
					.collect::<String>();
				let (terminal_reasoning, terminal_answer) =
					crate::engine::reasoning::split_reasoning(input);
				assert_eq!(
					streamed_reasoning,
					terminal_reasoning.unwrap_or_default(),
					"reasoning split at byte {split}"
				);
				assert_eq!(
					streamed_answer, terminal_answer,
					"answer split at byte {split}"
				);
			}
		}
	}

	#[test]
	fn disproved_partial_marker_flushes_losslessly() {
		let output = classify(ToolCallFormat::None, &[("<thi", "<thi"), ("mble", "mble")]);
		assert_eq!(output, vec![(TokenKind::Text, "<thimble".to_string())]);
	}

	#[test]
	fn terminal_partial_marker_flushes_losslessly() {
		let output = classify(ToolCallFormat::None, &[("<thi", "<thi")]);
		assert_eq!(output, vec![(TokenKind::Text, "<thi".to_string())]);
	}

	#[test]
	fn reasoning_marker_quoted_after_text_stays_text() {
		let output = classify(
			ToolCallFormat::None,
			&[("quote <think> literally", "quote <think> literally")],
		);
		assert_eq!(
			output,
			vec![(TokenKind::Text, "quote <think> literally".to_string())]
		);
	}

	#[test]
	fn excessive_prefix_whitespace_flushes_as_plain_text() {
		let input = format!(
			"{}<think>private</think>",
			" ".repeat(crate::engine::reasoning::MAX_PREFIX_WHITESPACE_BYTES + 1)
		);
		let output = classify(ToolCallFormat::None, &[(&input, &input)]);
		assert_eq!(output, vec![(TokenKind::Text, input)]);
	}

	#[test]
	fn prompt_seed_classifies_from_first_piece() {
		let mut classifier = StreamClassifier::new(ToolCallFormat::None);
		classifier.seed_reasoning("</think>");
		let mut output = classifier.push("secret</think>answer", "secret</think>answer");
		output.extend(classifier.finish());
		assert_eq!(
			output,
			vec![
				ClassifiedText {
					kind: TokenKind::Reasoning,
					text: "secret".to_string(),
				},
				ClassifiedText {
					kind: TokenKind::Text,
					text: "answer".to_string(),
				},
			]
		);
	}

	#[test]
	fn special_token_reasoning_marker_strips_only_visible_suffix() {
		let output = classify(
			ToolCallFormat::None,
			&[
				("<|channel>", ""),
				("thoughtsecret", "thoughtsecret"),
				("<channel|>", ""),
				("answer", "answer"),
			],
		);
		assert_eq!(
			output,
			vec![
				(TokenKind::Reasoning, "secret".to_string()),
				(TokenKind::Text, "answer".to_string()),
			]
		);
	}

	#[test]
	fn hermes_tool_call_payload_is_typed_and_markers_are_removed() {
		let output = classify(
			ToolCallFormat::Hermes,
			&[(
				"before<tool_call>{\"name\":\"x\"}</tool_call>after",
				"before<tool_call>{\"name\":\"x\"}</tool_call>after",
			)],
		);
		assert_eq!(
			output,
			vec![
				(TokenKind::Text, "before".to_string()),
				(TokenKind::ToolCall, "{\"name\":\"x\"}".to_string()),
				(TokenKind::Text, "after".to_string()),
			]
		);
	}

	#[test]
	fn empty_tool_span_still_emits_structural_boundary_signal() {
		let output = classify(
			ToolCallFormat::Hermes,
			&[(
				"before<tool_call></tool_call>after",
				"before<tool_call></tool_call>after",
			)],
		);
		assert_eq!(
			output,
			vec![
				(TokenKind::Text, "before".to_string()),
				(TokenKind::ToolCall, String::new()),
				(TokenKind::Text, "after".to_string()),
			]
		);
	}

	#[test]
	fn gemma_tool_call_special_marker_is_typed() {
		let output = classify(
			ToolCallFormat::Gemma,
			&[
				("<|tool_call>", ""),
				("call:get_weather{}", "call:get_weather{}"),
				("<tool_call|>", ""),
				("done", "done"),
			],
		);
		assert_eq!(
			output,
			vec![
				(TokenKind::ToolCall, String::new()),
				(TokenKind::ToolCall, "call:get_weather{}".to_string()),
				(TokenKind::Text, "done".to_string()),
			]
		);
	}
}
