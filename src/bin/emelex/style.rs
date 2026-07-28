//! Terminal color and compact human-readable units.

use std::io::IsTerminal as _;

use clap::ValueEnum;

use super::output;

/// Command-line color policy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, ValueEnum)]
pub(crate) enum ColorMode {
	/// Detect terminal capability and honor `NO_COLOR`.
	#[default]
	Auto,
	/// Always emit ANSI styling.
	Always,
	/// Never emit ANSI styling.
	Never,
}

/// Color capability for one output stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Palette {
	enabled: bool,
}

impl Palette {
	/// Palette for stdout.
	pub(crate) fn stdout(mode: ColorMode) -> Self {
		Self::new(
			mode,
			std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none(),
		)
	}

	/// Palette for stderr.
	pub(crate) fn stderr(mode: ColorMode) -> Self {
		Self::new(
			mode,
			std::io::stderr().is_terminal() && std::env::var_os("NO_COLOR").is_none(),
		)
	}

	const fn new(mode: ColorMode, detected: bool) -> Self {
		Self {
			enabled: match mode {
				ColorMode::Auto => detected,
				ColorMode::Always => true,
				ColorMode::Never => false,
			},
		}
	}

	/// Whether ANSI styling is active.
	pub(crate) const fn is_enabled(self) -> bool {
		self.enabled
	}

	pub(crate) fn bold(self, text: &str) -> String {
		self.wrap("1", text)
	}

	pub(crate) fn dim(self, text: &str) -> String {
		self.wrap("2", text)
	}

	pub(crate) fn red(self, text: &str) -> String {
		self.wrap("31", text)
	}

	pub(crate) fn green(self, text: &str) -> String {
		self.wrap("32", text)
	}

	pub(crate) fn yellow(self, text: &str) -> String {
		self.wrap("33", text)
	}

	pub(crate) fn cyan(self, text: &str) -> String {
		self.wrap("36", text)
	}

	fn wrap(self, code: &str, text: &str) -> String {
		let text = output::terminal_safe(text);
		if self.enabled {
			format!("\u{1b}[{code}m{text}\u{1b}[0m")
		} else {
			text.into_owned()
		}
	}
}

/// Compact decimal token count.
pub(crate) fn tokens(count: u64) -> String {
	human_number(count, 1_000)
}

/// Compact binary byte count.
pub(crate) fn bytes(count: u64) -> String {
	const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB"];
	let mut value = count as f64;
	let mut unit = 0_usize;
	while value >= 1024.0 && unit + 1 < UNITS.len() {
		value /= 1024.0;
		unit += 1;
	}
	if unit == 0 {
		format!("{count} {}", UNITS[unit])
	} else if value >= 10.0 {
		format!("{value:.0} {}", UNITS[unit])
	} else {
		format!("{value:.1} {}", UNITS[unit])
	}
}

fn human_number(count: u64, base: u64) -> String {
	let (divisor, suffix) = match count {
		0..=999 => return count.to_string(),
		1_000..=999_999 => (base as f64, "k"),
		1_000_000..=999_999_999 => ((base * base) as f64, "m"),
		_ => ((base * base * base) as f64, "b"),
	};
	let value = count as f64 / divisor;
	if value >= 100.0 || value.fract().abs() < f64::EPSILON {
		format!("{value:.0}{suffix}")
	} else {
		format!("{value:.1}{suffix}")
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn palette_can_be_forced() {
		assert_eq!(
			Palette::new(ColorMode::Always, false).green("ok"),
			"\u{1b}[32mok\u{1b}[0m"
		);
		assert_eq!(Palette::new(ColorMode::Never, true).green("ok"), "ok");
	}

	#[test]
	fn units_stay_compact() {
		assert_eq!(tokens(842), "842");
		assert_eq!(tokens(1_500), "1.5k");
		assert_eq!(tokens(2_000_000), "2m");
		assert_eq!(bytes(42), "42 B");
		assert_eq!(bytes(1536), "1.5 KiB");
	}
}
