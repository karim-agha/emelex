//! Stored Hugging Face authentication presentation.

use std::io::{IsTerminal as _, Read};

use anyhow::{Context as _, bail};
use emelex::{config::Config, home::EmelexHome, hub::HubCredentials};

use super::{args::HubAuthCommand, output, style::Palette};

const MAX_TOKEN_BYTES: usize = 4_096;
const MAX_TOKEN_READ_BYTES: u64 = 4_097;

pub(crate) fn run(
	home: &EmelexHome,
	command: HubAuthCommand,
	json: bool,
	stdout_palette: Palette,
	stderr_palette: Palette,
) -> anyhow::Result<()> {
	match command {
		HubAuthCommand::Login { token_stdin } => {
			login(home, token_stdin, json, stdout_palette, stderr_palette)
		}
		HubAuthCommand::Status => status(home, json, stdout_palette),
		HubAuthCommand::Logout => logout(home, json, stdout_palette, stderr_palette),
	}
}

fn login(
	home: &EmelexHome,
	token_stdin: bool,
	json: bool,
	stdout_palette: Palette,
	stderr_palette: Palette,
) -> anyhow::Result<()> {
	let environment = environment_auth();
	if json && !token_stdin {
		bail!("`hub auth login --json` requires `--token-stdin`");
	}
	let token = if token_stdin {
		read_token(std::io::stdin().lock()).context("read Hugging Face token from stdin")?
	} else {
		if !std::io::stdin().is_terminal() || !std::io::stderr().is_terminal() {
			bail!("non-interactive Hub login requires `--token-stdin`");
		}
		normalize_token(
			dialoguer::Password::new()
				.with_prompt("Hugging Face token")
				.interact()
				.context("read Hugging Face token")?,
		)?
	};
	HubCredentials::bearer_token(&token).context("validate Hugging Face token")?;
	Config::write_global_hub_token(home, Some(&token)).context("store Hugging Face token")?;

	let effective = effective_auth_with_environment(home, environment)?;
	if json {
		return output::json_line(&serde_json::json!({
			"stored": true,
			"authenticated": effective.authenticated,
			"source": effective.source,
		}));
	}
	output::stdout_line(&format!(
		"{} {}",
		stdout_palette.green("✓ Stored"),
		stdout_palette.bold("Hugging Face token")
	))?;
	if let Some(warning) = effective.stored_token_warning() {
		output::stderr_line(&stderr_palette.yellow(&format!("! {warning}")))?;
	}
	Ok(())
}

fn status(home: &EmelexHome, json: bool, palette: Palette) -> anyhow::Result<()> {
	let effective = effective_auth_with_environment(home, environment_auth())?;
	if json {
		output::json_line(&serde_json::json!({
			"authenticated": effective.authenticated,
			"source": effective.source,
			"stored": effective.stored,
		}))
	} else {
		output::stdout_line(&effective.render_human_status(palette))
	}
}

fn logout(
	home: &EmelexHome,
	json: bool,
	stdout_palette: Palette,
	stderr_palette: Palette,
) -> anyhow::Result<()> {
	let environment = environment_auth();
	Config::write_global_hub_token(home, None).context("remove stored Hugging Face token")?;
	let effective = effective_auth_with_environment(home, environment)?;
	if json {
		return output::json_line(&serde_json::json!({
			"removed": true,
			"authenticated": effective.authenticated,
			"source": effective.source,
		}));
	}
	output::stdout_line(&format!(
		"{} {}",
		stdout_palette.green("✓ Removed"),
		stdout_palette.bold("stored Hugging Face token")
	))?;
	if let Some(warning) = effective.environment_warning() {
		output::stderr_line(&stderr_palette.yellow(&format!("! {warning}")))?;
	}
	Ok(())
}

fn read_token(mut reader: impl Read) -> anyhow::Result<String> {
	let mut bytes = Vec::new();
	reader
		.by_ref()
		.take(MAX_TOKEN_READ_BYTES)
		.read_to_end(&mut bytes)
		.context("read token bytes")?;
	if bytes.len() > MAX_TOKEN_BYTES {
		bail!("Hugging Face token exceeds {MAX_TOKEN_BYTES} bytes");
	}
	let token = String::from_utf8(bytes).context("Hugging Face token must be valid UTF-8")?;
	normalize_token(token)
}

fn normalize_token(mut token: String) -> anyhow::Result<String> {
	if token.ends_with('\n') {
		token.pop();
		if token.ends_with('\r') {
			token.pop();
		}
	}
	if token.is_empty() || token.trim() != token || token.chars().any(char::is_control) {
		bail!("Hugging Face token must be one non-empty line without surrounding whitespace");
	}
	Ok(token)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EnvironmentAuth {
	Absent,
	Empty,
	Set,
	Invalid,
}

fn environment_auth() -> EnvironmentAuth {
	match std::env::var("HF_TOKEN") {
		Ok(token) if token.is_empty() => EnvironmentAuth::Empty,
		Ok(token) if HubCredentials::bearer_token(&token).is_ok() => EnvironmentAuth::Set,
		Ok(_) | Err(std::env::VarError::NotUnicode(_)) => EnvironmentAuth::Invalid,
		Err(std::env::VarError::NotPresent) => EnvironmentAuth::Absent,
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EffectiveAuth {
	authenticated: bool,
	source: &'static str,
	stored: bool,
	environment: EnvironmentAuth,
}

impl EffectiveAuth {
	fn render_human_status(self, palette: Palette) -> String {
		match (self.authenticated, self.source) {
			(true, "environment") => format!(
				"{}  {}",
				palette.green("✓ Authenticated"),
				palette.dim("Hugging Face Hub · HF_TOKEN")
			),
			(true, "stored") => format!(
				"{}  {}",
				palette.green("✓ Authenticated"),
				palette.dim("Hugging Face Hub · stored token")
			),
			(false, "anonymous") => format!(
				"{}  {}",
				palette.yellow("! Anonymous"),
				"empty HF_TOKEN suppresses the stored token"
			),
			(false, "invalid_environment") => {
				format!(
					"{}  {}",
					palette.red("× Authentication blocked"),
					"HF_TOKEN is invalid; unset or replace it"
				)
			}
			_ => format!(
				"{}  {}",
				palette.yellow("! Anonymous"),
				"no Hugging Face token configured"
			),
		}
	}

	const fn stored_token_warning(self) -> Option<&'static str> {
		match self.environment {
			EnvironmentAuth::Set => {
				Some("HF_TOKEN remains active and takes precedence over the stored token")
			}
			EnvironmentAuth::Empty => Some("empty HF_TOKEN keeps Hub access anonymous"),
			EnvironmentAuth::Invalid => Some("invalid HF_TOKEN prevents use of the stored token"),
			EnvironmentAuth::Absent => None,
		}
	}

	const fn environment_warning(self) -> Option<&'static str> {
		match self.environment {
			EnvironmentAuth::Set => Some("HF_TOKEN remains active"),
			EnvironmentAuth::Invalid => Some("invalid HF_TOKEN still prevents Hub access"),
			EnvironmentAuth::Empty | EnvironmentAuth::Absent => None,
		}
	}
}

fn effective_auth_with_environment(
	home: &EmelexHome,
	environment: EnvironmentAuth,
) -> anyhow::Result<EffectiveAuth> {
	let stored =
		Config::global_hub_token_configured(home).context("inspect stored Hugging Face token")?;
	let (authenticated, source) = match environment {
		EnvironmentAuth::Set => (true, "environment"),
		EnvironmentAuth::Empty => (false, "anonymous"),
		EnvironmentAuth::Invalid => (false, "invalid_environment"),
		EnvironmentAuth::Absent if stored => (true, "stored"),
		EnvironmentAuth::Absent => (false, "none"),
	};
	Ok(EffectiveAuth {
		authenticated,
		source,
		stored,
		environment,
	})
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::style::ColorMode;

	#[test]
	fn stdin_token_accepts_one_line_ending_only() {
		assert_eq!(
			read_token(b"hf_example\n".as_slice()).expect("newline token"),
			"hf_example"
		);
		assert_eq!(
			read_token(b"hf_example\r\n".as_slice()).expect("CRLF token"),
			"hf_example"
		);
		assert!(read_token(b"hf_example\nsecond".as_slice()).is_err());
		assert!(read_token(b" hf_example\n".as_slice()).is_err());
		assert!(read_token(b"\n".as_slice()).is_err());
	}

	#[test]
	fn stdin_token_is_bounded() {
		let oversized = vec![b'x'; MAX_TOKEN_BYTES + 1];
		assert!(read_token(oversized.as_slice()).is_err());
	}

	#[test]
	fn effective_status_copy_is_secret_free() {
		let palette = Palette::stdout(ColorMode::Never);
		let status = EffectiveAuth {
			authenticated: true,
			source: "environment",
			stored: true,
			environment: EnvironmentAuth::Set,
		};
		assert_eq!(
			status.render_human_status(palette),
			"✓ Authenticated  Hugging Face Hub · HF_TOKEN"
		);
		assert_eq!(
			status.stored_token_warning(),
			Some("HF_TOKEN remains active and takes precedence over the stored token")
		);

		let invalid = EffectiveAuth {
			authenticated: false,
			source: "invalid_environment",
			stored: true,
			environment: EnvironmentAuth::Invalid,
		};
		assert_eq!(
			invalid.render_human_status(palette),
			"× Authentication blocked  HF_TOKEN is invalid; unset or replace it"
		);
	}

	#[test]
	fn anonymous_status_distinguishes_missing_and_suppressing_environment() {
		let palette = Palette::stdout(ColorMode::Never);
		let missing = EffectiveAuth {
			authenticated: false,
			source: "none",
			stored: false,
			environment: EnvironmentAuth::Absent,
		};
		assert_eq!(
			missing.render_human_status(palette),
			"! Anonymous  no Hugging Face token configured"
		);

		let suppressed = EffectiveAuth {
			authenticated: false,
			source: "anonymous",
			stored: true,
			environment: EnvironmentAuth::Empty,
		};
		assert_eq!(
			suppressed.render_human_status(palette),
			"! Anonymous  empty HF_TOKEN suppresses the stored token"
		);
	}
}
