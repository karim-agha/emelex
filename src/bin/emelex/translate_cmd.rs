//! Structured translation against translation-capable models
//! (TranslateGemma-style templates): one-shot and interactive REPL.

use std::io::IsTerminal as _;

use anyhow::{Context as _, bail};
use emelex::{
	Emelex,
	config::ThinkingMode,
	generation::{FinishReason, GenerationEvent, GenerationOptions, GenerationRequest, Message},
	models::ModelLoadOptions,
};
use rustyline::error::ReadlineError;

use super::{
	args::TranslateArgs,
	chat_cmd::{
		ChatInput, build_editor, classify_chat_input, load_prompt_history, report_history_warning,
		save_prompt_history, slash_parts,
	},
	generate_cmd::{prompt as resolve_prompt, usage_footer},
	model_select, output,
	style::Palette,
};

const TRANSLATE_HELP: &str = "/from CODE   set the source language\n\
	/to CODE     set the target language\n\
	/swap        swap the language pair\n\
	/langs [Q]   list supported language codes (optional filter)\n\
	/model       show the loaded model\n\
	/help        this summary\n\
	/quit        leave";

/// The live language pair; both sides must be set before translating.
#[derive(Debug, Clone, Default)]
struct LanguagePair {
	source: Option<String>,
	target: Option<String>,
}

impl LanguagePair {
	fn resolved(&self) -> Option<(&str, &str)> {
		match (self.source.as_deref(), self.target.as_deref()) {
			(Some(source), Some(target)) => Some((source, target)),
			_ => None,
		}
	}

	fn prompt(&self) -> String {
		format!(
			"\n{}\u{2192}{}\u{276f} ",
			self.source.as_deref().unwrap_or("??"),
			self.target.as_deref().unwrap_or("??")
		)
	}
}

/// Run one-shot or interactive translation.
pub(crate) async fn run(
	emelex: &Emelex,
	args: TranslateArgs,
	json: bool,
	stdout_palette: Palette,
	stderr_palette: Palette,
) -> anyhow::Result<()> {
	let configured = &emelex.config().translate;
	let pair = LanguagePair {
		source: args.from.or_else(|| configured.source.clone()),
		target: args.to.or_else(|| configured.target.clone()),
	};
	let interactive = args.text.is_none()
		&& !json
		&& std::io::stdin().is_terminal()
		&& std::io::stderr().is_terminal();

	let inference = &emelex.config().inference;
	let required = model_select::filters(model_select::InvocationRequirements {
		chat: false,
		translation: true,
		system_prompt: false,
		agent: false,
		image: false,
		audio: false,
		reasoning_history: false,
		thinking_toggle: false,
		mtp: inference.mtp && inference.speculative_tokens > 0,
	})?;
	let installed = model_select::resolve(
		emelex,
		args.model.as_ref(),
		&required,
		interactive,
		stdout_palette,
		stderr_palette,
	)
	.await?;
	let mut load_options = ModelLoadOptions::default().thinking(ThinkingMode::Off);
	if let Some(max_tokens) = args.max_tokens {
		load_options = load_options.max_tokens(max_tokens);
	}
	if inference.mtp {
		load_options = load_options.speculative_tokens(inference.speculative_tokens);
	}
	let client = emelex
		.models()
		.context("initialize model manager")?
		.load(&installed, &load_options)
		.with_context(|| format!("load {}", installed.reference()))?;
	if !client.supports_translation() {
		bail!(
			"model {} does not accept structured translation requests; find one with \
			 `emelex hub search --require task:translation`",
			installed.reference()
		);
	}
	let languages = client.translation_languages();

	let mut options = GenerationOptions::default();
	if let Some(max_tokens) = args.max_tokens {
		options = options.max_tokens(max_tokens.min(client.effective_max_tokens()));
	}
	if let Some(temperature) = args.temperature {
		options = options.temperature(temperature);
	}

	if !interactive {
		let (source, target) = pair.resolved().context(
			"set a language pair: pass --from CODE and --to CODE, or configure \
			 [translate] source/target in emelex configuration",
		)?;
		validate_code(languages.as_deref(), source)?;
		validate_code(languages.as_deref(), target)?;
		let text = resolve_prompt(args.text)?;
		let message = Message::translation(source, target, text);
		return stream_translation(&client, message, options, json, stderr_palette).await;
	}
	run_repl(
		emelex,
		&client,
		&installed,
		pair,
		languages,
		options,
		stderr_palette,
	)
	.await
}

async fn run_repl(
	emelex: &Emelex,
	client: &emelex::Client,
	installed: &emelex::model::InstalledModel,
	mut pair: LanguagePair,
	languages: Option<std::sync::Arc<std::collections::BTreeMap<String, String>>>,
	options: GenerationOptions,
	stderr_palette: Palette,
) -> anyhow::Result<()> {
	output::stderr_line(&stderr_palette.dim(&format!(
		"model: {}  (task:translation)",
		installed.reference()
	)))?;
	if pair.resolved().is_none() {
		output::stderr_line(
			&stderr_palette
				.dim("set a language pair first: /from en, /to de  (/langs lists supported codes)"),
		)?;
	}

	let mut editor = build_editor(stderr_palette.is_enabled())?;
	let history_path = emelex.home().cache_dir().join("translate_history");
	let mut history_warning_reported = false;
	if let Err(error) = load_prompt_history(&mut editor, &history_path, &emelex.home().temp_dir()) {
		report_history_warning(&mut history_warning_reported, &error, stderr_palette)?;
	}
	loop {
		let line = match editor.readline(&pair.prompt()) {
			Ok(line) => line,
			Err(ReadlineError::Interrupted | ReadlineError::Eof) => break,
			Err(error) => return Err(error).context("read translation input"),
		};
		let Some(input) = classify_chat_input(line) else {
			continue;
		};
		match editor.add_history_entry(input.as_str()) {
			Ok(true) => {
				if let Err(error) = save_prompt_history(&mut editor, &history_path) {
					report_history_warning(&mut history_warning_reported, &error, stderr_palette)?;
				}
			}
			Ok(false) => {}
			Err(error) => report_history_warning(
				&mut history_warning_reported,
				&anyhow::Error::new(error).context("add prompt history"),
				stderr_palette,
			)?,
		}
		if let ChatInput::SlashCommand(command) = &input {
			match slash(
				command,
				&mut pair,
				languages.as_deref(),
				installed,
				stderr_palette,
			) {
				Ok(true) => break,
				Ok(false) => {}
				Err(error) => output::stderr_line(&stderr_palette.yellow(&format!(
					"{}",
					output::terminal_safe_inline(&format!("{error:#}"))
				)))?,
			}
			continue;
		}
		let ChatInput::Message(text) = input else {
			continue;
		};
		let Some((source, target)) = pair.resolved() else {
			output::stderr_line(
				&stderr_palette.yellow("set a language pair first: /from en, /to de"),
			)?;
			continue;
		};
		let message = Message::translation(source, target, text);
		if let Err(error) =
			stream_translation(client, message, options, false, stderr_palette).await
		{
			output::stderr_line(
				&stderr_palette.yellow(&output::terminal_safe_inline(&format!("{error:#}"))),
			)?;
		}
	}
	Ok(())
}

fn validate_code(
	languages: Option<&std::collections::BTreeMap<String, String>>,
	code: &str,
) -> anyhow::Result<()> {
	if code.trim().is_empty() {
		bail!("language codes cannot be empty");
	}
	let Some(languages) = languages else {
		// No embedded table — pass through; the template render is the authority.
		return Ok(());
	};
	if !languages.contains_key(code) {
		bail!(
			"unknown language code {code:?} for this model; run `emelex translate` and \
			 use /langs to list supported codes"
		);
	}
	Ok(())
}

fn slash(
	line: &str,
	pair: &mut LanguagePair,
	languages: Option<&std::collections::BTreeMap<String, String>>,
	installed: &emelex::model::InstalledModel,
	palette: Palette,
) -> anyhow::Result<bool> {
	let (command, argument) = slash_parts(line);
	match command.as_str() {
		"/bye" | "/exit" | "/quit" => Ok(true),
		"/help" => {
			output::stdout_line(TRANSLATE_HELP)?;
			Ok(false)
		}
		"/from" => {
			set_language(argument, languages, &mut pair.source, "/from")?;
			Ok(false)
		}
		"/to" => {
			set_language(argument, languages, &mut pair.target, "/to")?;
			Ok(false)
		}
		"/swap" => {
			std::mem::swap(&mut pair.source, &mut pair.target);
			Ok(false)
		}
		"/langs" => {
			list_languages(argument, languages, palette)?;
			Ok(false)
		}
		"/model" => {
			output::stderr_line(&palette.dim(&format!("{}", installed.reference())))?;
			Ok(false)
		}
		other => bail!("unknown command {other}; try /help"),
	}
}

fn set_language(
	argument: &str,
	languages: Option<&std::collections::BTreeMap<String, String>>,
	slot: &mut Option<String>,
	flag: &str,
) -> anyhow::Result<()> {
	let code = argument.trim();
	if code.is_empty() {
		bail!("{flag} requires a language code (e.g. {flag} en)");
	}
	validate_code(languages, code)?;
	*slot = Some(code.to_string());
	Ok(())
}

fn list_languages(
	filter: &str,
	languages: Option<&std::collections::BTreeMap<String, String>>,
	palette: Palette,
) -> anyhow::Result<()> {
	let Some(languages) = languages else {
		output::stderr_line(
			&palette.dim("this model's template does not publish a language table"),
		)?;
		return Ok(());
	};
	let filter = filter.trim().to_lowercase();
	let mut shown = 0_usize;
	for (code, name) in languages {
		if !filter.is_empty()
			&& !code.to_lowercase().contains(&filter)
			&& !name.to_lowercase().contains(&filter)
		{
			continue;
		}
		output::stdout_line(&format!(
			"{:<10} {}",
			output::terminal_safe_inline(code),
			output::terminal_safe_inline(name)
		))?;
		shown += 1;
	}
	if shown == 0 {
		output::stderr_line(&palette.dim("no language matches that filter"))?;
	}
	Ok(())
}

/// Stream one translation to stdout as plain text (translations are not
/// markdown; render them verbatim).
async fn stream_translation(
	client: &emelex::Client,
	message: Message,
	options: GenerationOptions,
	json: bool,
	stderr_palette: Palette,
) -> anyhow::Result<()> {
	let mut stream = client
		.stream(
			GenerationRequest::default()
				.message(message)
				.options(options),
		)
		.context("start translation")?;
	let mut terminal = None;
	let signal = tokio::signal::ctrl_c();
	tokio::pin!(signal);
	let drive_result: anyhow::Result<()> = async {
		loop {
			let event = tokio::select! {
				event = stream.recv() => event,
				signal_result = &mut signal => {
					signal_result.context("listen for Ctrl-C")?;
					bail!("translation cancelled");
				}
			};
			let Some(event) = event else {
				break;
			};
			let event = event.context("translate")?;
			if json {
				if let GenerationEvent::Completed(response) = &event {
					terminal = Some(response.clone());
				}
				output::json_line(&event)?;
				continue;
			}
			match event {
				GenerationEvent::Text(text) => output::stdout(&output::terminal_safe(&text))?,
				GenerationEvent::Completed(response) => terminal = Some(response),
				_ => {}
			}
		}
		Ok(())
	}
	.await;
	if let Err(primary) = drive_result {
		let primary = match stream.cancel_and_wait().await {
			Ok(()) => primary,
			Err(cleanup) => {
				anyhow::anyhow!("{primary:#}; translation cleanup also failed: {cleanup}")
			}
		};
		return Err(primary);
	}
	let response = terminal.context("translation stream ended without a terminal response")?;
	if json {
		return Ok(());
	}
	if response.finish_reason == FinishReason::Length {
		output::stderr_line(&stderr_palette.yellow("\ntranslation was cut off by max tokens"))?;
	}
	output::stdout("\n")?;
	output::stderr_line(&stderr_palette.dim(&usage_footer(
		response.usage.prompt_tokens,
		response.usage.cached_tokens,
		response.usage.completion_tokens,
		None,
	)))?;
	Ok(())
}
