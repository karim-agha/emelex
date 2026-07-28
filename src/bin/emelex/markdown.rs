//! Incremental terminal rendering for streamed Markdown.

use std::{fmt::Write as _, sync::OnceLock};

use syntect::{
	easy::HighlightLines,
	highlighting::{Theme, ThemeSet},
	parsing::SyntaxSet,
	util::as_24_bit_terminal_escaped,
};

const PREFIX_CAP: usize = 8;
const CODE_THEME: &str = "base16-ocean.dark";

static SYNTAXES: OnceLock<SyntaxSet> = OnceLock::new();
static THEME: OnceLock<Theme> = OnceLock::new();

fn syntaxes() -> &'static SyntaxSet {
	SYNTAXES.get_or_init(SyntaxSet::load_defaults_newlines)
}

fn theme() -> &'static Theme {
	THEME.get_or_init(|| {
		ThemeSet::load_defaults()
			.themes
			.remove(CODE_THEME)
			.unwrap_or_default()
	})
}

enum LineState {
	Start(String),
	Inline,
	FenceHeader(String),
	Code(String),
}

/// Streaming Markdown renderer used by interactive model output.
#[expect(
	clippy::struct_excessive_bools,
	reason = "independent inline parser flags model orthogonal Markdown delimiters"
)]
pub(crate) struct MarkdownStream {
	enabled: bool,
	base: &'static str,
	started: bool,
	state: LineState,
	asterisks: usize,
	bold: bool,
	italic: bool,
	code_span: bool,
	line_style: &'static str,
	highlighter: Option<HighlightLines<'static>>,
}

impl MarkdownStream {
	pub(crate) const fn new(enabled: bool) -> Self {
		Self::with_base(enabled, "")
	}

	pub(crate) const fn with_base(enabled: bool, base: &'static str) -> Self {
		Self {
			enabled,
			base,
			started: false,
			state: LineState::Start(String::new()),
			asterisks: 0,
			bold: false,
			italic: false,
			code_span: false,
			line_style: "",
			highlighter: None,
		}
	}

	/// Render one exact stream chunk.
	pub(crate) fn push(&mut self, chunk: &str) -> String {
		if !self.enabled {
			return chunk.to_string();
		}
		let mut output = String::new();
		if !self.started {
			output.push_str(self.base);
			self.started = true;
		}
		for character in chunk.chars() {
			self.step(character, &mut output);
		}
		output
	}

	/// Flush buffered prefixes or code and reset terminal styling.
	pub(crate) fn finish(&mut self) -> String {
		if !self.enabled {
			return String::new();
		}
		let mut output = String::new();
		match std::mem::replace(&mut self.state, LineState::Start(String::new())) {
			LineState::Start(prefix) => {
				for character in prefix.chars() {
					self.inline(character, &mut output);
				}
			}
			LineState::FenceHeader(header) => {
				let _ = write!(output, "\u{1b}[2m```{header}\u{1b}[0m{}", self.base);
			}
			LineState::Inline => {}
			LineState::Code(line) if line.trim() == "```" => {
				let _ = write!(output, "\u{1b}[2m{line}\u{1b}[0m");
			}
			LineState::Code(line) => output.push_str(&self.highlight(&line)),
		}
		self.flush_asterisks(&mut output);
		output.push_str("\u{1b}[0m");
		self.reset_line();
		output
	}

	fn step(&mut self, character: char, output: &mut String) {
		match std::mem::replace(&mut self.state, LineState::Inline) {
			LineState::Start(prefix) => self.start(prefix, character, output),
			LineState::Inline => self.inline(character, output),
			LineState::FenceHeader(mut header) => {
				if character == '\n' {
					self.open_fence(&header, output);
				} else {
					header.push(character);
					self.state = LineState::FenceHeader(header);
				}
			}
			LineState::Code(mut line) => {
				if character == '\n' {
					if line.trim() == "```" {
						let _ = write!(output, "\u{1b}[2m{line}");
						let _ = writeln!(output, "{}", self.reset());
						self.highlighter = None;
						self.state = LineState::Start(String::new());
					} else {
						output.push_str(&self.highlight(&line));
						output.push('\n');
						self.state = LineState::Code(String::new());
					}
				} else {
					line.push(character);
					self.state = LineState::Code(line);
				}
			}
		}
	}

	fn start(&mut self, mut prefix: String, character: char, output: &mut String) {
		let classified = match character {
			'\n' => {
				self.replay(&prefix, output);
				self.inline('\n', output);
				true
			}
			' ' if prefix.chars().all(|value| value == '#') && !prefix.is_empty() => {
				self.line_style = "\u{1b}[1;36m";
				output.push_str(&self.sgr());
				output.push_str(&prefix);
				output.push(' ');
				self.state = LineState::Inline;
				true
			}
			' ' if matches!(prefix.trim_start_matches(' '), "-" | "*" | "+")
				&& !prefix.trim_start_matches(' ').is_empty() =>
			{
				let indent = prefix.len() - prefix.trim_start_matches(' ').len();
				output.push_str(&" ".repeat(indent));
				output.push_str("\u{1b}[33m\u{2022}");
				output.push_str(&self.reset());
				output.push(' ');
				output.push_str(&self.sgr());
				self.state = LineState::Inline;
				true
			}
			_ => false,
		};
		if classified {
			return;
		}

		prefix.push(character);
		let markers = prefix.trim_start_matches(' ');
		if markers == "```" {
			self.state = LineState::FenceHeader(String::new());
			return;
		}
		if markers == ">" {
			self.line_style = "\u{1b}[2m";
			output.push_str(&self.sgr());
			output.push_str(&prefix);
			self.state = LineState::Inline;
			return;
		}
		let ambiguous = prefix.len() < PREFIX_CAP
			&& (markers.is_empty()
				|| markers.chars().all(|value| value == '#')
				|| matches!(markers, "-" | "+" | "*" | "`" | "``"));
		if ambiguous {
			self.state = LineState::Start(prefix);
		} else {
			self.replay(&prefix, output);
		}
	}

	fn replay(&mut self, prefix: &str, output: &mut String) {
		self.state = LineState::Inline;
		for character in prefix.chars() {
			self.inline(character, output);
		}
	}

	fn inline(&mut self, character: char, output: &mut String) {
		match character {
			'*' if !self.code_span => {
				self.asterisks += 1;
				self.state = LineState::Inline;
			}
			'`' => {
				self.flush_asterisks(output);
				self.code_span = !self.code_span;
				output.push_str(&self.sgr());
				self.state = LineState::Inline;
			}
			'\n' => {
				self.flush_asterisks(output);
				self.reset_line();
				output.push_str("\u{1b}[0m\n");
				output.push_str(self.base);
				self.state = LineState::Start(String::new());
			}
			other => {
				self.flush_asterisks(output);
				output.push(other);
				self.state = LineState::Inline;
			}
		}
	}

	fn flush_asterisks(&mut self, output: &mut String) {
		match self.asterisks {
			0 => return,
			1 => self.italic = !self.italic,
			2 => self.bold = !self.bold,
			_ => {
				self.bold = !self.bold;
				self.italic = !self.italic;
			}
		}
		self.asterisks = 0;
		output.push_str(&self.sgr());
	}

	fn reset(&self) -> String {
		format!("\u{1b}[0m{}", self.base)
	}

	fn sgr(&self) -> String {
		let mut sequence = self.reset();
		sequence.push_str(self.line_style);
		if self.bold {
			sequence.push_str("\u{1b}[1m");
		}
		if self.italic {
			sequence.push_str("\u{1b}[3m");
		}
		if self.code_span {
			sequence.push_str("\u{1b}[36m");
		}
		sequence
	}

	const fn reset_line(&mut self) {
		self.bold = false;
		self.italic = false;
		self.code_span = false;
		self.asterisks = 0;
		self.line_style = "";
	}

	fn open_fence(&mut self, header: &str, output: &mut String) {
		let language = header.trim();
		let _ = write!(output, "\u{1b}[2m```{language}");
		let _ = writeln!(output, "{}", self.reset());
		let syntax = if language.is_empty() {
			None
		} else {
			syntaxes()
				.find_syntax_by_token(language)
				.or_else(|| syntaxes().find_syntax_by_extension(language))
		};
		self.highlighter = syntax.map(|syntax| HighlightLines::new(syntax, theme()));
		self.reset_line();
		self.state = LineState::Code(String::new());
	}

	fn highlight(&mut self, line: &str) -> String {
		let base = self.base;
		self.highlighter.as_mut().map_or_else(
			|| line.to_string(),
			|highlighter| {
				let with_newline = format!("{line}\n");
				highlighter
					.highlight_line(&with_newline, syntaxes())
					.map_or_else(
						|_| line.to_string(),
						|ranges| {
							let mut rendered = as_24_bit_terminal_escaped(&ranges, false);
							if rendered.ends_with('\n') {
								rendered.pop();
							}
							rendered.push_str("\u{1b}[0m");
							rendered.push_str(base);
							rendered
						},
					)
			},
		)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn render(chunks: &[&str]) -> String {
		let mut stream = MarkdownStream::new(true);
		let mut output = String::new();
		for chunk in chunks {
			output.push_str(&stream.push(chunk));
		}
		output.push_str(&stream.finish());
		output
	}

	#[test]
	fn disabled_stream_is_byte_identical() {
		let text = "# title\n**bold**\n";
		let mut stream = MarkdownStream::new(false);
		assert_eq!(stream.push(text), text);
		assert!(stream.finish().is_empty());
	}

	#[test]
	fn split_markers_preserve_styling() {
		let output = render(&["some *", "*bold*", "* text"]);
		assert!(output.contains("\u{1b}[1mbold"));
		assert!(!output.contains("**"));
	}

	#[test]
	fn blocks_and_fences_render_across_chunks() {
		let output = render(&["## Plan\n- item\n``", "`rust\nfn main() {}\n`", "``\n"]);
		assert!(output.contains("\u{1b}[1;36m## Plan"));
		assert!(output.contains('\u{2022}'));
		assert!(output.contains("\u{1b}[38;2;"));
		assert!(output.contains("\u{1b}[2m```rust"));
	}

	#[test]
	fn finish_resets_base_overlay() {
		let mut stream = MarkdownStream::with_base(true, "\u{1b}[2;3m");
		let mut output = stream.push("unfinished **thought");
		output.push_str(&stream.finish());
		assert!(output.ends_with("\u{1b}[0m"));
	}

	#[test]
	fn finish_preserves_unclosed_fence_header() {
		let output = render(&["```rust"]);
		assert!(output.contains("```rust"));
	}
}
