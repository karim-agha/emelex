//! Emelex command-line entry point.

#![allow(
	clippy::redundant_pub_crate,
	reason = "sibling command modules intentionally share crate-scoped items; a binary has no external Rust API"
)]
#![cfg_attr(
	test,
	allow(
		clippy::expect_used,
		clippy::panic,
		clippy::unwrap_used,
		reason = "CLI unit tests use fail-fast fixture setup and assertions"
	)
)]

use std::process::ExitCode;

pub(crate) mod args;
pub(crate) mod chat_cmd;
pub(crate) mod doctor_cmd;
pub(crate) mod generate_cmd;
pub(crate) mod hub_auth_cmd;
pub(crate) mod hub_cmd;
pub(crate) mod markdown;
pub(crate) mod media;
pub(crate) mod memory_cmd;
pub(crate) mod memory_worker;
pub(crate) mod model_select;
pub(crate) mod models_cmd;
pub(crate) mod output;
pub(crate) mod style;
pub(crate) mod terminal_ui;
pub(crate) mod web_search;

use anyhow::Context as _;
use args::{ChatArgs, Cli, Command, HubCommand};
use clap::Parser as _;
use emelex::{Emelex, home::EmelexHome, hub::HubCredentials};

#[tokio::main]
async fn main() -> ExitCode {
	let cli = Cli::parse();
	let json = cli.json;
	let stderr_palette = style::Palette::stderr(cli.color);
	match run(cli).await {
		Ok(()) => ExitCode::SUCCESS,
		Err(error) if output::is_broken_pipe(&error) => ExitCode::SUCCESS,
		Err(error) => {
			let reported = if json {
				output::json_error(&error)
			} else {
				output::stderr_line(&format_human_error(&error, stderr_palette))
			};
			if let Err(report_error) = reported
				&& !output::is_broken_pipe(&report_error)
			{
				let _ = output::stderr_line("emelex: unable to report command error");
			}
			ExitCode::FAILURE
		}
	}
}

fn format_human_error(error: &anyhow::Error, palette: style::Palette) -> String {
	let formatted = format!("{error:#}");
	let error = output::terminal_safe_inline(&formatted);
	format!("{} {error}", palette.red("error:"))
}

async fn run(cli: Cli) -> anyhow::Result<()> {
	let stdout_palette = style::Palette::stdout(cli.color);
	let stderr_palette = style::Palette::stderr(cli.color);
	if let Command::Hub {
		command: HubCommand::Auth { command },
	} = &cli.command
	{
		let home = EmelexHome::resolve(cli.home.as_deref()).context("initialize Emelex Home")?;
		return hub_auth_cmd::run(&home, *command, cli.json, stdout_palette, stderr_palette);
	}
	let mut builder = Emelex::builder().project_config(!cli.no_project_config);
	if let Some(home) = cli.home {
		builder = builder.home(home);
	}
	if let Some(directory) = cli.directory {
		builder = builder.invocation_root(directory);
	}
	match hf_credentials_from_env()? {
		HubCredentialOverride::Inherit => {}
		HubCredentialOverride::Anonymous => builder = builder.anonymous_hub(),
		HubCredentialOverride::Explicit(credentials) => {
			builder = builder.hub_credentials(credentials);
		}
	}
	let emelex = builder.build().context("initialize Emelex")?;
	match cli.command {
		Command::Chat(args) => {
			chat_cmd::run(&emelex, args, cli.json, stdout_palette, stderr_palette).await
		}
		Command::Resume(args) => {
			let chat = chat_cmd::resume_args(args.session, args.approve_all, args.prompt);
			chat_cmd::run(&emelex, chat, cli.json, stdout_palette, stderr_palette).await
		}
		Command::Generate(args) => {
			generate_cmd::run(&emelex, args, cli.json, stdout_palette, stderr_palette).await
		}
		Command::Hub { command } => {
			match hub_cmd::run(&emelex, command, cli.json, stdout_palette, stderr_palette).await? {
				hub_cmd::HubRunOutcome::Done => Ok(()),
				hub_cmd::HubRunOutcome::StartChat(model) => {
					chat_cmd::run(
						&emelex,
						ChatArgs::for_model(model),
						cli.json,
						stdout_palette,
						stderr_palette,
					)
					.await
				}
			}
		}
		Command::Model { command } => {
			models_cmd::run_model(&emelex, command, cli.json, stdout_palette, stderr_palette)
		}
		Command::Models { command } => {
			models_cmd::run(&emelex, command, cli.json, stdout_palette, stderr_palette).await
		}
		Command::Memory { command } => {
			memory_cmd::run(&emelex, command, cli.json, stdout_palette, stderr_palette).await
		}
		Command::Doctor(args) => doctor_cmd::run(&emelex, args, cli.json, stdout_palette),
	}
}

#[derive(Debug)]
enum HubCredentialOverride {
	Inherit,
	Anonymous,
	Explicit(HubCredentials),
}

fn hf_credentials_from_env() -> anyhow::Result<HubCredentialOverride> {
	hf_credentials_from_value(std::env::var("HF_TOKEN"))
}

fn hf_credentials_from_value(
	value: Result<String, std::env::VarError>,
) -> anyhow::Result<HubCredentialOverride> {
	match value {
		Ok(token) if token.is_empty() => Ok(HubCredentialOverride::Anonymous),
		Ok(token) => HubCredentials::bearer_token(&token)
			.map(HubCredentialOverride::Explicit)
			.context("invalid HF_TOKEN"),
		Err(std::env::VarError::NotPresent) => Ok(HubCredentialOverride::Inherit),
		Err(std::env::VarError::NotUnicode(_)) => {
			anyhow::bail!("HF_TOKEN must contain valid UTF-8")
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn human_command_errors_neutralize_terminal_controls() {
		let error = anyhow::anyhow!("bad\u{1b}]0;owned\u{7}\u{202e}txt");
		let rendered = format_human_error(&error, style::Palette::stderr(style::ColorMode::Never));
		assert!(!rendered.contains('\u{1b}'));
		assert!(!rendered.contains('\u{202e}'));
		assert!(rendered.contains('\u{241b}'));
	}

	#[test]
	fn environment_credentials_distinguish_inherit_and_explicit_anonymous() {
		assert!(matches!(
			hf_credentials_from_value(Err(std::env::VarError::NotPresent))
				.expect("absent environment"),
			HubCredentialOverride::Inherit
		));
		assert!(matches!(
			hf_credentials_from_value(Ok(String::new())).expect("empty environment"),
			HubCredentialOverride::Anonymous
		));
		assert!(matches!(
			hf_credentials_from_value(Ok("hf_example".to_string())).expect("explicit environment"),
			HubCredentialOverride::Explicit(_)
		));
	}
}
