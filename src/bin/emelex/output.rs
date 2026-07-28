//! Fallible terminal output, machine-readable output, and no-clobber exports.

use std::{
	borrow::Cow,
	fs::File,
	io::{self, Write as _},
	path::{Path, PathBuf},
};

use anyhow::Context as _;
use serde::Serialize;

/// Write exact bytes to stdout without panicking on a closed pipe.
pub(crate) fn stdout(text: &str) -> anyhow::Result<()> {
	let stdout = io::stdout();
	stdout
		.lock()
		.write_all(text.as_bytes())
		.context("write stdout")
}

/// Write one line to stdout without panicking on a closed pipe.
pub(crate) fn stdout_line(text: &str) -> anyhow::Result<()> {
	let stdout = io::stdout();
	let mut output = stdout.lock();
	output.write_all(text.as_bytes()).context("write stdout")?;
	output.write_all(b"\n").context("write stdout")
}

/// Write exact bytes to stderr without panicking.
pub(crate) fn stderr(text: &str) -> anyhow::Result<()> {
	let stderr = io::stderr();
	stderr
		.lock()
		.write_all(text.as_bytes())
		.context("write stderr")
}

/// Write one line to stderr without panicking.
pub(crate) fn stderr_line(text: &str) -> anyhow::Result<()> {
	let stderr = io::stderr();
	let mut output = stderr.lock();
	output.write_all(text.as_bytes()).context("write stderr")?;
	output.write_all(b"\n").context("write stderr")
}

/// Print one compact JSON value followed by a newline.
pub(crate) fn json_line(value: &impl Serialize) -> anyhow::Result<()> {
	let stdout = io::stdout();
	let mut output = stdout.lock();
	serde_json::to_writer(&mut output, value).context("encode JSON output")?;
	output.write_all(b"\n").context("write JSON output")
}

/// Write a structured command error to stderr.
pub(crate) fn json_error(error: &anyhow::Error) -> anyhow::Result<()> {
	#[derive(Serialize)]
	struct ErrorEnvelope<'a> {
		kind: &'static str,
		message: &'a str,
		causes: Vec<&'a str>,
	}

	let message = error.to_string();
	let causes = error
		.chain()
		.skip(1)
		.map(ToString::to_string)
		.collect::<Vec<_>>();
	let envelope = ErrorEnvelope {
		kind: "error",
		message: &message,
		causes: causes.iter().map(String::as_str).collect(),
	};
	let stderr = io::stderr();
	let mut output = stderr.lock();
	serde_json::to_writer(&mut output, &envelope).context("encode JSON error")?;
	output.write_all(b"\n").context("write JSON error")
}

/// Whether an error chain represents a consumer closing stdout or stderr.
pub(crate) fn is_broken_pipe(error: &anyhow::Error) -> bool {
	error.chain().any(|cause| {
		cause
			.downcast_ref::<io::Error>()
			.is_some_and(|error| error.kind() == io::ErrorKind::BrokenPipe)
	})
}

/// Neutralize terminal control and direction-override characters in untrusted text.
///
/// Newlines and horizontal tabs remain useful in model output. Every other
/// terminal control is rendered visibly, while Unicode direction controls are
/// replaced, preventing model, tool, path, and Hub metadata from injecting ANSI
/// sequences or visually reordering trusted UI text.
pub(crate) fn terminal_safe(text: &str) -> Cow<'_, str> {
	if !text.chars().any(needs_terminal_escaping) {
		return Cow::Borrowed(text);
	}
	let mut safe = String::with_capacity(text.len());
	for character in text.chars() {
		match character {
			'\n' | '\t' => safe.push(character),
			'\r' => safe.push('\u{240d}'),
			'\u{1b}' => safe.push('\u{241b}'),
			'\u{7f}' => safe.push('\u{2421}'),
			value if value.is_control() || is_direction_control(value) => safe.push('\u{fffd}'),
			value => safe.push(value),
		}
	}
	Cow::Owned(safe)
}

/// Neutralize every terminal control in one trusted-layout field.
///
/// Unlike [`terminal_safe`], this replaces line and tab separators so an
/// untrusted path, title, diagnostic, or status value cannot forge another UI
/// row or prompt.
pub(crate) fn terminal_safe_inline(text: &str) -> Cow<'_, str> {
	if !text.chars().any(needs_inline_escaping) {
		return Cow::Borrowed(text);
	}
	let mut safe = String::with_capacity(text.len());
	for character in text.chars() {
		match character {
			'\n' | '\u{2028}' | '\u{2029}' => safe.push('\u{240a}'),
			'\t' => safe.push('\u{2409}'),
			'\r' => safe.push('\u{240d}'),
			'\u{1b}' => safe.push('\u{241b}'),
			'\u{7f}' => safe.push('\u{2421}'),
			value if value.is_control() || is_direction_control(value) => safe.push('\u{fffd}'),
			value => safe.push(value),
		}
	}
	Cow::Owned(safe)
}

fn needs_terminal_escaping(character: char) -> bool {
	!matches!(character, '\n' | '\t') && (character.is_control() || is_direction_control(character))
}

fn needs_inline_escaping(character: char) -> bool {
	character.is_control()
		|| is_direction_control(character)
		|| matches!(character, '\u{2028}' | '\u{2029}')
}

const fn is_direction_control(character: char) -> bool {
	matches!(
		character,
		'\u{061c}'
			| '\u{200e}'
			| '\u{200f}'
			| '\u{202a}'..='\u{202e}'
			| '\u{2066}'..='\u{2069}'
	)
}

/// Write pretty JSON to stdout or to a newly created file.
pub(crate) fn export_json(
	value: &impl Serialize,
	destination: Option<&Path>,
) -> anyhow::Result<()> {
	export_stream(destination, |output| {
		serde_json::to_writer_pretty(&mut *output, value).context("encode JSON export")
	})
}

/// Stream one JSON export to stdout or a newly created file.
pub(crate) fn export_stream(
	destination: Option<&Path>,
	write: impl FnOnce(&mut dyn io::Write) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
	match destination {
		None => {
			let stdout = io::stdout();
			let mut output = stdout.lock();
			write(&mut output)?;
			output.write_all(b"\n").context("write JSON export")
		}
		Some(path) => export_stream_file(path, write),
	}
}

fn export_stream_file(
	path: &Path,
	write: impl FnOnce(&mut dyn io::Write) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
	let absolute = std::path::absolute(path)
		.with_context(|| format!("resolve export path {}", path.display()))?;
	let parent = absolute
		.parent()
		.map_or_else(|| PathBuf::from("."), Path::to_path_buf);
	let mut temporary = tempfile::NamedTempFile::new_in(&parent)
		.with_context(|| format!("create temporary export in {}", parent.display()))?;
	write(temporary.as_file_mut())?;
	temporary
		.as_file_mut()
		.write_all(b"\n")
		.context("finish JSON export")?;
	temporary.as_file().sync_all().context("sync JSON export")?;
	temporary
		.persist_noclobber(&absolute)
		.map_err(|error| error.error)
		.with_context(|| format!("publish export without replacing {}", absolute.display()))?;
	File::open(&parent)
		.and_then(|directory| directory.sync_all())
		.with_context(|| format!("sync export directory {}", parent.display()))
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn export_refuses_to_replace_existing_file() {
		let directory = tempfile::tempdir().expect("directory");
		let path = directory.path().join("memory.json");
		std::fs::write(&path, "original").expect("existing file");

		let error = export_json(&serde_json::json!({"new": true}), Some(&path))
			.expect_err("must not replace");
		assert!(error.to_string().contains("without replacing"));
		assert_eq!(
			std::fs::read_to_string(path).expect("read original"),
			"original"
		);
	}

	#[test]
	fn terminal_text_cannot_inject_ansi_or_bidi_overrides() {
		assert_eq!(
			terminal_safe("ok\u{1b}[31m\r\u{202e}x"),
			"ok\u{241b}[31m\u{240d}\u{fffd}x"
		);
		assert!(matches!(terminal_safe("safe\n\ttext"), Cow::Borrowed(_)));
		assert_eq!(
			terminal_safe_inline("safe\n\tfield\u{2028}next"),
			"safe\u{240a}\u{2409}field\u{240a}next"
		);
	}

	#[test]
	fn every_untrusted_human_surface_neutralizes_osc_and_bidi() {
		for (surface, value) in [
			("reasoning", "think\u{1b}]0;owned\u{7}\u{202e}"),
			("tool output", "result\u{1b}]8;;file:///tmp/x\u{7}\u{2067}"),
			("error", "failed\u{1b}[2J\u{200f}"),
			("workspace path", "/tmp/work\u{1b}]2;x\u{7}\u{202d}"),
			("attachment path", "/tmp/image\u{1b}]0;x\u{7}\u{2066}.png"),
		] {
			let rendered = terminal_safe(value);
			assert!(!rendered.contains('\u{1b}'), "{surface}");
			assert!(!rendered.contains('\u{7}'), "{surface}");
			assert!(!rendered.chars().any(is_direction_control), "{surface}");
		}
	}
}
