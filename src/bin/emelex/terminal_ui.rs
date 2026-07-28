//! Small, scrollback-friendly terminal primitives.

use anyhow::Context as _;
use dialoguer::console::{Term, measure_text_width, truncate_str};

/// One redrawable terminal region.
///
/// Frames always end at the start of a fresh line. The region owns cursor
/// visibility until it is cleared or settled, and restores the cursor on drop.
pub(crate) struct LiveRegion {
	term: Term,
	rendered_frame: Option<String>,
	cursor_hidden: bool,
}

impl LiveRegion {
	/// A buffered stdout region.
	pub(crate) fn stdout() -> Self {
		Self::new(Term::buffered_stdout())
	}

	/// A buffered stderr region.
	pub(crate) fn stderr() -> Self {
		Self::new(Term::buffered_stderr())
	}

	const fn new(term: Term) -> Self {
		Self {
			term,
			rendered_frame: None,
			cursor_hidden: false,
		}
	}

	/// Current `(rows, columns)`.
	pub(crate) fn size(&self) -> (u16, u16) {
		self.term.size()
	}

	/// Read one attended-terminal key.
	pub(crate) fn read_key(&self) -> anyhow::Result<dialoguer::console::Key> {
		self.term.read_key_raw().context("read terminal key")
	}

	/// Atomically replace this region with `frame`.
	pub(crate) fn draw(&mut self, frame: &str) -> anyhow::Result<()> {
		self.hide_cursor()?;
		self.erase()?;
		let frame = frame.trim_end_matches('\n');
		for line in frame.split('\n') {
			self.term.write_line(line).context("draw terminal frame")?;
		}
		self.term.flush().context("flush terminal frame")?;
		self.rendered_frame = Some(frame.to_string());
		Ok(())
	}

	/// Remove the live frame and restore the cursor.
	pub(crate) fn clear(&mut self) -> anyhow::Result<()> {
		self.erase()?;
		self.show_cursor()
	}

	fn erase(&mut self) -> anyhow::Result<()> {
		let columns = usize::from(self.size().1).max(1);
		if let Some(rows) = self
			.rendered_frame
			.as_deref()
			.map(|frame| rendered_rows(frame, columns))
		{
			self.term
				.clear_last_lines(rows)
				.context("clear terminal frame")?;
			self.rendered_frame = None;
		}
		Ok(())
	}

	fn hide_cursor(&mut self) -> anyhow::Result<()> {
		if !self.cursor_hidden {
			self.term.hide_cursor().context("hide terminal cursor")?;
			self.cursor_hidden = true;
		}
		Ok(())
	}

	fn show_cursor(&mut self) -> anyhow::Result<()> {
		if self.cursor_hidden {
			self.term.show_cursor().context("show terminal cursor")?;
			self.term.flush().context("flush terminal cursor")?;
			self.cursor_hidden = false;
		}
		Ok(())
	}
}

impl Drop for LiveRegion {
	fn drop(&mut self) {
		let _ = self.term.show_cursor();
		let _ = self.term.flush();
	}
}

/// Truncate one styled line without splitting ANSI sequences.
pub(crate) fn fit_line(line: &str, columns: usize) -> String {
	let width = columns.saturating_sub(1).max(1);
	let tail = if width > 1 { "…" } else { "" };
	truncate_str(line, width, tail).into_owned()
}

fn rendered_rows(frame: &str, columns: usize) -> usize {
	frame
		.split('\n')
		.map(|line| measure_text_width(line).max(1).div_ceil(columns))
		.sum()
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn fitted_lines_leave_the_terminal_wrap_column_unused() {
		assert_eq!(fit_line("abcdefgh", 6), "abcd…");
		assert!(measure_text_width(&fit_line("\u{1b}[36mabcdefgh\u{1b}[0m", 6)) <= 5);
	}

	#[test]
	fn rendered_rows_accounts_for_wrapping_and_blank_lines() {
		assert_eq!(rendered_rows("one\n\n123456", 4), 4);
	}

	#[test]
	fn rendered_rows_are_recomputed_after_a_resize() {
		let frame = "1234567\nabc";
		assert_eq!(rendered_rows(frame, 8), 2);
		assert_eq!(rendered_rows(frame, 4), 3);
	}
}
