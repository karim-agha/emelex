//! Reasoning ("thinking") span detection, extraction, and budget
//! enforcement.
//!
//! Several model families shipped as `mlx-community` checkpoints (Qwen3,
//! Qwen3.5, Qwen3.6, Gemma4, MiniCPM5, NemotronH, ...) support an opt-in
//! "thinking" mode - toggled via the chat template's `enable_thinking`
//! variable (see [`crate::engine::generate::GenerateOptions::enable_thinking`])
//! - where the model prefixes its reply with a delimited reasoning span
//! before the actual answer. Two delimiter conventions show up across
//! these templates:
//! - Qwen-style: plain-text `<think>...</think>` tags.
//! - Gemma4-style: `<|channel>thought` ... `<channel|>` tags.
//!
//! This module is deliberately architecture-agnostic. A reasoning marker
//! is recognized only at the generated reply's start (after optional
//! whitespace), matching the prefix contract of supported chat templates
//! and avoiding false positives when an ordinary answer quotes `<think>`.

/// `(open, close)` marker pairs, checked in order; the first `open`
/// marker found in the generated text fixes which `close` marker
/// terminates its reasoning span. Also consumed by
/// `crate::engine::streaming::StreamClassifier` to tag live-streamed tokens.
pub(crate) const MARKER_PAIRS: &[(&str, &str)] =
	&[("<think>", "</think>"), ("<|channel>thought", "<channel|>")];

/// Maximum leading whitespace retained while deciding whether a reply starts
/// with a reasoning marker. Supported templates emit at most a newline here;
/// this limit prevents an all-whitespace generation from growing streaming
/// state without bound.
pub(crate) const MAX_PREFIX_WHITESPACE_BYTES: usize = 64;
/// Maximum leading whitespace accepted before an immediately duplicated
/// model close after Emelex teacher-forces the reasoning boundary.
pub(crate) const MAX_FORCED_CLOSE_WHITESPACE_BYTES: usize = 8;

/// Return the marker candidate after a bounded amount of leading whitespace.
pub(crate) fn prefix_candidate(text: &str) -> Option<&str> {
	let candidate = text.trim_start();
	(text.len() - candidate.len() <= MAX_PREFIX_WHITESPACE_BYTES).then_some(candidate)
}

/// Split a raw generated reply into `(reasoning, answer)`.
///
/// If no known opening marker starts the reply, `reasoning` is `None`
/// and `answer` is `text` unchanged. If a prefix marker is never closed
/// (for example, `max_tokens` cuts generation off mid-thought), the
/// remainder is reasoning and `answer` is empty.
pub fn split_reasoning(text: &str) -> (Option<String>, String) {
	split_reasoning_inner(text, false)
}

/// Split a reply after Emelex teacher-forced a reasoning close and generation
/// continued. One immediately duplicated model close is suppressed; once
/// answer text begins, the forced boundary is authoritative and later close
/// markers remain literal answer text.
pub(crate) fn split_reasoning_after_forced_close(text: &str) -> (Option<String>, String) {
	split_reasoning_inner(text, true)
}

fn split_reasoning_inner(text: &str, drop_immediate_duplicate: bool) -> (Option<String>, String) {
	let Some(candidate) = prefix_candidate(text) else {
		return (None, text.to_string());
	};
	for (open, close) in MARKER_PAIRS {
		if let Some(after_open) = candidate.strip_prefix(open) {
			return match after_open.find(close) {
				Some(close_at) => {
					let reasoning = after_open[..close_at].to_string();
					let remainder = &after_open[close_at + close.len()..];
					let answer = if drop_immediate_duplicate {
						strip_immediate_forced_close(remainder, close)
							.unwrap_or(remainder)
							.to_string()
					} else {
						remainder.to_string()
					};
					((!reasoning.is_empty()).then_some(reasoning), answer)
				}
				None => {
					let reasoning = after_open.to_string();
					((!reasoning.is_empty()).then_some(reasoning), String::new())
				}
			};
		}
	}
	(None, text.to_string())
}

/// Remove one immediately duplicated post-force close, accepting only bounded
/// leading whitespace so a long answer prefix cannot be mistaken for syntax.
pub(crate) fn strip_immediate_forced_close<'a>(text: &'a str, close: &str) -> Option<&'a str> {
	let candidate = text.trim_start();
	let leading_bytes = text.len() - candidate.len();
	if leading_bytes <= MAX_FORCED_CLOSE_WHITESPACE_BYTES {
		candidate.strip_prefix(close)
	} else {
		None
	}
}

/// Detect whether a *rendered chat-template prompt* (i.e. the text fed to
/// the model, not its output) already opens an unclosed reasoning span at
/// its very end.
///
/// Several `enable_thinking`-style templates (Qwen3/3.5/3.6, NemotronH)
/// bake the opening marker into the generation prompt itself - e.g.
/// Qwen3.5's template ends `add_generation_prompt` with
/// `'<|im_start|>assistant\n<think>\n'` rather than letting the model
/// generate `<think>` itself. On those checkpoints the model's *generated*
/// text starts already inside the reasoning span and never contains the
/// literal open marker, so [`split_reasoning`]/`StreamClassifier` would
/// otherwise never detect it (Gemma4's template, by contrast, leaves the
/// open marker for the model to generate itself when thinking is
/// enabled - see its template's `add_generation_prompt` block - so it
/// needs no such treatment).
///
/// Returns the `(open, close)` pair whose `open` marker the prompt ends
/// with (after trimming trailing whitespace), so callers can seed
/// [`ReasoningBudget`] / `crate::engine::streaming::StreamClassifier` as if
/// that marker had just been generated, and prepend it back before calling
/// [`split_reasoning`] on the model's actual output.
pub(crate) fn pending_marker(prompt: &str) -> Option<(&'static str, &'static str)> {
	let trimmed = prompt.trim_end();
	MARKER_PAIRS
		.iter()
		.find(|(open, _)| trimmed.ends_with(open))
		.copied()
}

/// Detect a complete empty reasoning span at the rendered generation
/// prompt's end. Some templates disable thinking by pre-closing an empty
/// thought channel instead of leaving an open marker when thinking is enabled.
pub(crate) fn trailing_empty_marker(prompt: &str) -> Option<(&'static str, &'static str)> {
	let trimmed = prompt.trim_end();
	MARKER_PAIRS.iter().find_map(|&(open, close)| {
		let before_close = trimmed.strip_suffix(close)?.trim_end();
		before_close.ends_with(open).then_some((open, close))
	})
}

/// Tracks generated text against a token budget for the *reasoning* span
/// only, so [`crate::engine::generate::Session`]'s decode loop can force it
/// closed once the budget is exhausted - mirroring Anthropic's
/// extended-thinking `budget_tokens`: once the budget runs out mid-thought,
/// generation is cut over to the final answer rather than left to ramble
/// indefinitely.
pub struct ReasoningBudget {
	budget: usize,
	buffer: String,
	open_close: Option<(&'static str, &'static str)>,
	closed: bool,
	tokens_since_open: usize,
	searching_prefix: bool,
}

impl ReasoningBudget {
	pub fn new(budget: usize) -> Self {
		ReasoningBudget {
			budget,
			buffer: String::new(),
			open_close: None,
			closed: false,
			tokens_since_open: 0,
			searching_prefix: true,
		}
	}

	/// Seed the budget as if `pair.0` (the open marker) had already been
	/// observed - used when the reasoning span was opened by the *prompt*
	/// itself rather than by generated text (see
	/// [`pending_marker`]), so tokens generated from the very first one
	/// still count against the budget.
	pub(crate) fn seed_open(&mut self, pair: (&'static str, &'static str)) {
		if !self.closed && self.open_close.is_none() {
			self.open_close = Some(pair);
			self.searching_prefix = false;
			self.buffer.clear();
		}
	}

	/// Feed one newly generated token's decoded text. Returns the close
	/// marker to force-inject the first time the budget is exhausted while
	/// still inside an (unclosed) reasoning span; after that, this always
	/// returns `None` (a budget only ever fires once per generation).
	pub fn observe(&mut self, piece: &str) -> Option<&'static str> {
		if self.closed {
			return None;
		}
		self.buffer.push_str(piece);
		const TAIL_CAP: usize = 64;
		if self.searching_prefix {
			let Some(candidate) = prefix_candidate(&self.buffer) else {
				self.closed = true;
				self.buffer.clear();
				return None;
			};
			if let Some((pair, remainder)) = MARKER_PAIRS.iter().find_map(|pair| {
				candidate
					.strip_prefix(pair.0)
					.map(|remainder| (*pair, remainder))
			}) {
				self.open_close = Some(pair);
				self.searching_prefix = false;
				self.buffer = bounded_tail(remainder, TAIL_CAP);
				if self.buffer.contains(pair.1) {
					self.closed = true;
				}
				return None;
			}
			if candidate.is_empty()
				|| MARKER_PAIRS
					.iter()
					.any(|(open, _)| open.starts_with(candidate))
			{
				return None;
			}
			// Reasoning markers are reply prefixes. Once ordinary content
			// disproves every opener prefix, a later quoted marker must not
			// start budget accounting.
			self.closed = true;
			self.buffer.clear();
			return None;
		}

		// Keep only a bounded tail: markers are short, and rescanning an
		// unbounded reply on every token would be O(n²) time and O(n) memory.
		if self.buffer.len() > TAIL_CAP {
			let mut cut = self.buffer.len() - TAIL_CAP;
			while !self.buffer.is_char_boundary(cut) {
				cut += 1;
			}
			self.buffer.drain(..cut);
		}
		let (_, close) = self.open_close?;
		if self.buffer.contains(close) {
			// Model closed the span itself before hitting the budget.
			self.closed = true;
			return None;
		}
		self.tokens_since_open += 1;
		if self.tokens_since_open >= self.budget {
			self.closed = true;
			return Some(close);
		}
		None
	}
}

fn bounded_tail(text: &str, cap: usize) -> String {
	if text.len() <= cap {
		return text.to_string();
	}
	let mut start = text.len() - cap;
	while !text.is_char_boundary(start) {
		start += 1;
	}
	text[start..].to_string()
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn delayed_close_after_forced_boundary_remains_answer_text() {
		// Once non-close text follows the teacher-forced boundary, live
		// streaming has exposed it as answer text. Terminal reconciliation
		// therefore preserves that text and every later literal close.
		let (reasoning, answer) = split_reasoning_after_forced_close(
			"<think>\ncut off mid-thought\n</think>\nnearly the \
			 answer\n</think>\n\nthe answer is 4",
		);
		assert_eq!(reasoning.as_deref(), Some("\ncut off mid-thought\n"));
		assert_eq!(answer, "\nnearly the answer\n</think>\n\nthe answer is 4");
	}

	#[test]
	fn drops_orphan_close_after_forced_budget_close() {
		// emelex patch: forced close immediately followed by the model's
		// own close marker must not leak into the answer.
		let (reasoning, answer) = split_reasoning_after_forced_close(
			"<think>\nworking it out\n</think>\n\n</think>\n\nthe answer is 4",
		);
		assert_eq!(reasoning.as_deref(), Some("\nworking it out\n"));
		assert_eq!(answer, "\n\nthe answer is 4");
	}

	#[test]
	fn forced_close_consumes_one_duplicate_and_preserves_quoted_close() {
		let (reasoning, answer) = split_reasoning_after_forced_close(
			"<think>x</think></think>To write it, use </think>.",
		);
		assert_eq!(reasoning.as_deref(), Some("x"));
		assert_eq!(answer, "To write it, use </think>.");
	}

	#[test]
	fn forced_duplicate_requires_bounded_leading_whitespace() {
		let padding = " ".repeat(MAX_FORCED_CLOSE_WHITESPACE_BYTES + 1);
		let raw = format!("<think>x</think>{padding}</think>answer");
		let (reasoning, answer) = split_reasoning_after_forced_close(&raw);
		assert_eq!(reasoning.as_deref(), Some("x"));
		assert_eq!(answer, format!("{padding}</think>answer"));
	}

	#[test]
	fn budget_still_detects_close_after_long_reasoning() {
		// emelex patch regression: the budget buffer is a bounded tail;
		// a close marker arriving after a long span must still be seen.
		let mut b = ReasoningBudget::new(10_000);
		assert!(b.observe("<think>").is_none());
		for _ in 0..500 {
			assert!(b.observe("reasoning words flowing along ").is_none());
		}
		assert!(b.observe("</think>").is_none());
		// Span closed by the model itself: budget never fires afterwards.
		for _ in 0..50 {
			assert!(b.observe("answer text ").is_none());
		}
	}

	#[test]
	fn no_markers_passes_through_unchanged() {
		let (reasoning, answer) = split_reasoning("just a plain answer");
		assert_eq!(reasoning, None);
		assert_eq!(answer, "just a plain answer");
	}

	#[test]
	fn quoted_reasoning_marker_inside_answer_is_plain_text() {
		let text = "Explain the literal tag <think> without treating it as metadata.";
		let (reasoning, answer) = split_reasoning(text);
		assert_eq!(reasoning, None);
		assert_eq!(answer, text);
	}

	#[test]
	fn quoted_close_marker_after_reasoning_is_plain_answer_text() {
		let text = "<think>x</think>To close it, write </think>.";
		let (reasoning, answer) = split_reasoning(text);
		assert_eq!(reasoning.as_deref(), Some("x"));
		assert_eq!(answer, "To close it, write </think>.");
	}

	#[test]
	fn empty_reasoning_span_is_absent() {
		assert_eq!(
			split_reasoning("<think></think>answer"),
			(None, "answer".to_string())
		);
		assert_eq!(
			split_reasoning("<|channel>thought<channel|>answer"),
			(None, "answer".to_string())
		);
	}

	#[test]
	fn trailing_empty_marker_requires_empty_terminal_span() {
		assert_eq!(
			trailing_empty_marker("model\n<|channel>thought\n<channel|>\n"),
			Some(("<|channel>thought", "<channel|>"))
		);
		assert_eq!(trailing_empty_marker("<think>not empty</think>"), None);
		assert_eq!(
			trailing_empty_marker("<think></think>\nanswer"),
			None,
			"history markers are not generation-suffix evidence"
		);
	}

	#[test]
	fn excessive_prefix_whitespace_does_not_grow_or_start_reasoning() {
		let text = format!(
			"{}<think>private</think>public",
			" ".repeat(MAX_PREFIX_WHITESPACE_BYTES + 1)
		);
		assert_eq!(split_reasoning(&text), (None, text.clone()));

		let mut budget = ReasoningBudget::new(1);
		assert!(budget.observe(&text).is_none());
		assert!(budget.observe("later").is_none());
	}

	#[test]
	fn answer_boundary_and_trailing_whitespace_are_preserved() {
		let (_, answer) = split_reasoning("<think>x</think>\nanswer  \n");
		assert_eq!(answer, "\nanswer  \n");
	}

	#[test]
	fn extracts_qwen_style_think_tags() {
		let (reasoning, answer) =
			split_reasoning("<think>\nlet me work this out\n</think>\n\nthe answer is 4");
		assert_eq!(reasoning.as_deref(), Some("\nlet me work this out\n"));
		assert_eq!(answer, "\n\nthe answer is 4");
	}

	#[test]
	fn extracts_gemma4_channel_style() {
		let (reasoning, answer) =
			split_reasoning("<|channel>thought\nhmm\n<channel|>final answer here");
		assert_eq!(reasoning.as_deref(), Some("\nhmm\n"));
		assert_eq!(answer, "final answer here");
	}

	#[test]
	fn unclosed_span_is_all_reasoning() {
		let (reasoning, answer) = split_reasoning("<think>\nstill thinking with no end in sight");
		assert_eq!(
			reasoning.as_deref(),
			Some("\nstill thinking with no end in sight")
		);
		assert_eq!(answer, "");
	}

	#[test]
	fn budget_fires_once_after_threshold_tokens_inside_span() {
		let mut budget = ReasoningBudget::new(3);
		assert_eq!(budget.observe("<think>"), None);
		assert_eq!(budget.observe("a"), None);
		assert_eq!(budget.observe("b"), None);
		assert_eq!(budget.observe("c"), Some("</think>"));
		// Fires only once.
		assert_eq!(budget.observe("d"), None);
	}

	#[test]
	fn budget_does_not_fire_if_model_closes_span_itself() {
		let mut budget = ReasoningBudget::new(100);
		assert_eq!(budget.observe("<think>"), None);
		assert_eq!(budget.observe("quick"), None);
		assert_eq!(budget.observe("</think>"), None);
		for _ in 0..200 {
			assert_eq!(budget.observe("more text"), None);
		}
	}

	#[test]
	fn pending_marker_detects_qwen_style_generation_prompt() {
		// Qwen3.5/NemotronH-style templates bake the open marker into the
		// generation prompt itself instead of letting the model generate
		// it.
		let prompt = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n";
		assert_eq!(pending_marker(prompt), Some(("<think>", "</think>")));
	}

	#[test]
	fn pending_marker_is_none_when_thinking_disabled_and_already_closed() {
		let prompt = "<|im_start|>assistant\n<think>\n\n</think>\n\n";
		assert_eq!(pending_marker(prompt), None);
	}

	#[test]
	fn pending_marker_is_none_for_gemma_style_prompt_that_leaves_marker_to_model() {
		let prompt = "<|turn>model\n";
		assert_eq!(pending_marker(prompt), None);
	}

	#[test]
	fn pending_marker_is_none_for_plain_prompt() {
		assert_eq!(pending_marker("<|im_start|>assistant\n"), None);
	}

	#[test]
	fn budget_seed_open_lets_a_pending_span_count_from_the_first_token() {
		let mut budget = ReasoningBudget::new(2);
		budget.seed_open(("<think>", "</think>"));
		assert_eq!(budget.observe("a"), None);
		assert_eq!(budget.observe("b"), Some("</think>"));
	}

	#[test]
	fn budget_seed_open_is_a_noop_once_a_span_is_already_tracked() {
		let mut budget = ReasoningBudget::new(100);
		assert_eq!(budget.observe("<think>"), None);
		// A second seed (e.g. a defensive call site) must not reset the
		// in-progress span's marker pair.
		budget.seed_open(("<|channel>thought", "<channel|>"));
		assert_eq!(budget.observe("hmm"), None);
		assert_eq!(budget.observe("</think>"), None);
	}

	#[test]
	fn budget_ignores_generation_with_no_reasoning_span() {
		let mut budget = ReasoningBudget::new(1);
		for _ in 0..50 {
			assert_eq!(budget.observe("no markers here "), None);
		}
	}

	#[test]
	fn budget_ignores_reasoning_marker_quoted_after_plain_text() {
		let mut budget = ReasoningBudget::new(1);
		assert_eq!(budget.observe("plain answer "), None);
		assert_eq!(budget.observe("<think>quoted"), None);
	}

	#[test]
	fn budget_handles_open_and_close_in_one_piece() {
		let mut budget = ReasoningBudget::new(1);
		assert_eq!(budget.observe("<think>brief</think>answer"), None);
		assert_eq!(budget.observe(" more"), None);
	}
}
