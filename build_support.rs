//! Pure helpers shared by the native build script and its regression tests.

use std::fmt;

/// Strict one- or two-component Apple platform version.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct AppleVersion {
	major: u32,
	minor: u32,
}

impl AppleVersion {
	/// Parses exactly `MAJOR` or `MAJOR.MINOR`, using ASCII digits only.
	pub fn parse(value: &str, kind: &str) -> Result<Self, String> {
		let mut parts = value.split('.');
		let major = parse_component(parts.next(), value, kind)?;
		let minor = match parts.next() {
			Some(part) => parse_component(Some(part), value, kind)?,
			None => 0,
		};
		if parts.next().is_some() {
			return Err(invalid_version(value, kind));
		}
		Ok(Self { major, minor })
	}
}

impl fmt::Display for AppleVersion {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(formatter, "{}.{}", self.major, self.minor)
	}
}

fn parse_component(part: Option<&str>, value: &str, kind: &str) -> Result<u32, String> {
	let part = part.ok_or_else(|| invalid_version(value, kind))?;
	if part.is_empty() || !part.bytes().all(|byte| byte.is_ascii_digit()) {
		return Err(invalid_version(value, kind));
	}
	part.parse::<u32>()
		.map_err(|_| invalid_version(value, kind))
}

fn invalid_version(value: &str, kind: &str) -> String {
	format!("invalid {kind} {value:?}; expected MAJOR or MAJOR.MINOR")
}
