//! One-shot raw and agent generation.

use std::{
	fmt::Write as _,
	future::Future,
	io::{IsTerminal as _, Read},
	sync::Arc,
};

use anyhow::{Context as _, bail};
use emelex::{
	Emelex,
	agent::{AgentCancellation, AgentEvent, AgentSession, AllowAllApprovals, DenyAllApprovals},
	config::ThinkingMode,
	generation::{Content, GenerationEvent, GenerationOptions, GenerationRequest, Message, Role},
	models::{LoadOverride, ModelLoadOptions},
};
use sha2::{Digest as _, Sha256};

use super::{
	args::{GenerateArgs, ThinkingArg},
	chat_cmd::{ToolAvailability, agent_system_prompt},
	markdown::MarkdownStream,
	media, model_select, output,
	style::{Palette, tokens},
};

const MAX_STDIN_BYTES: u64 = 16 << 20;
const MAX_TOOL_EVENT_PREVIEW_BYTES: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ThinkingPlan {
	effective: ThinkingMode,
	load: ThinkingMode,
	request: Option<ThinkingMode>,
}

fn thinking_plan(configured: ThinkingMode, argument: Option<ThinkingArg>) -> ThinkingPlan {
	let requested = argument.map(thinking_mode);
	let effective = requested.map_or(configured, |mode| {
		if mode == ThinkingMode::Auto {
			configured
		} else {
			mode
		}
	});
	ThinkingPlan {
		effective,
		load: effective,
		request: requested,
	}
}

/// Run one bounded inference request.
pub(crate) async fn run(
	emelex: &Emelex,
	args: GenerateArgs,
	json: bool,
	stdout_palette: Palette,
	stderr_palette: Palette,
) -> anyhow::Result<()> {
	let prompt = prompt(args.prompt)?;
	let attachments = media::load_all(&args.attachments)?;
	let has_image = attachments
		.iter()
		.any(|attachment| matches!(attachment.content, Content::Image(_)));
	let has_audio = attachments
		.iter()
		.any(|attachment| matches!(attachment.content, Content::Audio(_)));
	let thinking = thinking_plan(emelex.config().inference.thinking, args.thinking);
	let inference = &emelex.config().inference;
	let required = model_select::filters(model_select::InvocationRequirements {
		chat: true,
		translation: false,
		system_prompt: args.agent,
		agent: args.agent,
		image: has_image,
		audio: has_audio,
		thinking_toggle: thinking.effective == ThinkingMode::On,
		mtp: inference.mtp && inference.speculative_tokens > 0,
	})?;
	let interactive = !json && std::io::stdin().is_terminal() && std::io::stderr().is_terminal();
	let installed = model_select::resolve(
		emelex,
		args.model.as_ref(),
		&required,
		interactive,
		stdout_palette,
		stderr_palette,
	)
	.await?;
	let mut load_options = ModelLoadOptions::default()
		.temperature(
			args.temperature
				.map_or(LoadOverride::Inherit, LoadOverride::Set),
		)
		.thinking(thinking.load);
	if let Some(max_tokens) = args.max_tokens {
		load_options = load_options.max_tokens(max_tokens);
	}
	if emelex.config().inference.mtp {
		load_options =
			load_options.speculative_tokens(emelex.config().inference.speculative_tokens);
	}
	let client = emelex
		.models()
		.context("initialize model manager")?
		.load(&installed, &load_options)
		.with_context(|| format!("load {}", installed.reference()))?;
	let message = user_message(prompt, attachments);
	let mut generation_options = GenerationOptions::default();
	if let Some(max_tokens) = request_max_tokens(args.max_tokens, client.effective_max_tokens()) {
		generation_options = generation_options.max_tokens(max_tokens);
	}
	if let Some(temperature) = args.temperature {
		generation_options = generation_options.temperature(temperature);
	}
	if let Some(requested) = thinking.request {
		generation_options = generation_options.thinking(requested);
	}

	if args.agent {
		run_agent(
			emelex,
			client,
			message,
			generation_options,
			args.approve_all,
			json,
			stdout_palette,
			stderr_palette,
		)
		.await
	} else {
		run_raw(
			client,
			message,
			generation_options,
			json,
			stdout_palette,
			stderr_palette,
		)
		.await
	}
}

pub(crate) fn prompt(argument: Option<String>) -> anyhow::Result<String> {
	if let Some(argument) = argument {
		if argument.trim().is_empty() {
			bail!("prompt cannot be empty");
		}
		return Ok(argument);
	}
	if std::io::stdin().is_terminal() {
		bail!("provide PROMPT or pipe UTF-8 text on stdin");
	}
	read_prompt_from(std::io::stdin().lock(), MAX_STDIN_BYTES)
}

fn read_prompt_from(reader: impl Read, max_bytes: u64) -> anyhow::Result<String> {
	let mut bytes = Vec::new();
	reader
		.take(max_bytes.saturating_add(1))
		.read_to_end(&mut bytes)
		.context("read prompt from stdin")?;
	if bytes.len() as u64 > max_bytes {
		bail!("stdin prompt exceeds {max_bytes} bytes");
	}
	let prompt = String::from_utf8(bytes).context("stdin prompt is not UTF-8")?;
	if prompt.trim().is_empty() {
		bail!("stdin prompt cannot be empty");
	}
	Ok(prompt)
}

fn user_message(prompt: String, attachments: Vec<media::Attachment>) -> Message {
	let mut content = vec![Content::Text(prompt)];
	content.extend(attachments.into_iter().map(|attachment| attachment.content));
	Message::with_content(Role::User, content)
}

#[allow(
	clippy::too_many_lines,
	reason = "generation, cancellation, output, and cleanup form one ordered stream lifecycle"
)]
async fn run_raw(
	client: emelex::Client,
	message: Message,
	options: GenerationOptions,
	json: bool,
	stdout_palette: Palette,
	stderr_palette: Palette,
) -> anyhow::Result<()> {
	let mut stream = client
		.stream(
			GenerationRequest::default()
				.message(message)
				.options(options),
		)
		.context("start generation")?;
	let mut markdown = MarkdownStream::new(stdout_palette.is_enabled());
	let mut terminal = None;
	let mut output_failed = false;
	let signal = tokio::signal::ctrl_c();
	tokio::pin!(signal);
	let drive_result: anyhow::Result<_> = async {
		loop {
			let event = tokio::select! {
				event = stream.recv() => event,
				signal_result = &mut signal => {
					signal_result.context("listen for Ctrl-C")?;
					bail!("generation cancelled");
				}
			};
			let Some(event) = event else {
				break;
			};
			let event = event.context("generate")?;
			if json {
				if terminal.is_some() {
					bail!("generation emitted an event after its terminal response");
				}
				if let GenerationEvent::Completed(response) = &event {
					terminal = Some(response.clone());
				}
				if let Err(error) = output::json_line(&event) {
					output_failed = true;
					return Err(error);
				}
				continue;
			}
			if terminal.is_some() {
				bail!("generation emitted an event after its terminal response");
			}
			match event {
				GenerationEvent::Text(text) => {
					let text = output::terminal_safe(&text);
					if let Err(error) = output::stdout(&markdown.push(&text)) {
						output_failed = true;
						return Err(error);
					}
				}
				GenerationEvent::Reasoning(text) => {
					let text = output::terminal_safe(&text);
					if let Err(error) = output::stderr(&stderr_palette.dim(&text)) {
						output_failed = true;
						return Err(error);
					}
				}
				GenerationEvent::ToolCall(call) => {
					let name = output::terminal_safe_inline(&call.name);
					if let Err(error) = output::stderr_line(
						&stderr_palette.yellow(&format!("unexpected tool call {name}")),
					) {
						output_failed = true;
						return Err(error);
					}
				}
				GenerationEvent::Completed(response) => terminal = Some(response),
				_ => {}
			}
		}
		let response = terminal.context("generation stream ended without a terminal response")?;
		if !response.tool_calls.is_empty()
			|| response.finish_reason == emelex::generation::FinishReason::ToolCalls
		{
			bail!("raw generation returned tool calls; rerun with `--agent`");
		}
		Ok(response)
	}
	.await;
	let response = match drive_result {
		Ok(response) => response,
		Err(primary) => {
			let primary = match stream.cancel_and_wait().await {
				Ok(()) => primary,
				Err(cleanup) => {
					anyhow::anyhow!("{primary:#}; generation cleanup also failed: {cleanup}")
				}
			};
			if !json
				&& !output_failed
				&& let Err(flush) = output::stdout(&markdown.finish())
			{
				return Err(flush.context(format!(
					"generation failed before terminal flush: {primary:#}"
				)));
			}
			return Err(primary);
		}
	};
	if json {
		return Ok(());
	}
	output::stdout(&markdown.finish())?;
	output::stderr_line(&format!(
		"\n{}",
		stderr_palette.dim(&usage_footer(
			response.usage.prompt_tokens,
			response.usage.cached_tokens,
			response.usage.completion_tokens,
			None,
		))
	))?;
	Ok(())
}

async fn run_agent(
	emelex: &Emelex,
	client: emelex::Client,
	message: Message,
	generation_options: GenerationOptions,
	approve_all: bool,
	json: bool,
	stdout_palette: Palette,
	stderr_palette: Palette,
) -> anyhow::Result<()> {
	let config = &emelex.config().agent;
	let approval: Arc<dyn emelex::agent::ApprovalPolicy> = if approve_all {
		Arc::new(AllowAllApprovals)
	} else {
		Arc::new(DenyAllApprovals)
	};
	let mut builder = AgentSession::builder(client, emelex.invocation_root())
		.approval_policy(approval)
		.generation_options(generation_options)
		.include_file_tools(config.files)
		.include_shell_tool(config.shell)
		.shell_timeout_seconds(config.shell_timeout_seconds)
		.shell_output_bytes(config.shell_output_bytes)
		.include_web_fetch(config.web)
		.web_response_bytes(config.web_response_bytes)
		.include_datetime(true)
		.max_model_rounds(config.max_turns);
	let system = agent_system_prompt(
		emelex.invocation_root(),
		emelex.config(),
		None,
		ToolAvailability {
			files: config.files,
			shell: config.shell,
			web_fetch: config.web,
			web_search: false,
		},
	);
	builder = builder.system_prompt(system);
	let mut session = builder.build().context("build agent")?;
	let cancellation = AgentCancellation::new();
	let mut markdown = MarkdownStream::new(stdout_palette.is_enabled());
	let mut reasoning = MarkdownStream::with_base(stderr_palette.is_enabled(), "\u{1b}[2;3m");
	let mut reasoning_active = false;
	let mut output_error = None;
	let turn = session.try_run_message(message, &cancellation, |event| {
		if let Err(error) = render_agent_event(
			&event,
			json,
			stderr_palette,
			&mut markdown,
			&mut reasoning,
			&mut reasoning_active,
		) {
			output_error = Some(error);
			return Err("event output failed");
		}
		Ok(())
	});
	let result = await_with_cancellation(turn, &cancellation, tokio::signal::ctrl_c()).await;
	if let Some(error) = output_error {
		return Err(error);
	}
	if !json {
		finish_human_streams(&mut markdown, &mut reasoning, &mut reasoning_active)?;
	}
	let result = result?;
	let turn = result.context("run agent")?;
	if !json {
		output::stderr_line(&format!(
			"\n{}",
			stderr_palette.dim(&usage_footer(
				turn.usage.prompt_tokens,
				turn.usage.cached_tokens,
				turn.usage.completion_tokens,
				Some(turn.model_rounds),
			))
		))?;
	}
	Ok(())
}

pub(crate) fn usage_footer(
	prompt_tokens: u64,
	cached_tokens: u64,
	completion_tokens: u64,
	model_rounds: Option<usize>,
) -> String {
	let mut footer = format!(
		"↑ {} prompt · ↺ {} cached · ↓ {} generated",
		tokens(prompt_tokens),
		tokens(cached_tokens),
		tokens(completion_tokens)
	);
	if let Some(rounds) = model_rounds {
		let label = if rounds == 1 { "round" } else { "rounds" };
		let _ = write!(&mut footer, " · {rounds} {label}");
	}
	footer
}

pub(crate) async fn await_with_cancellation<F, S, T>(
	future: F,
	cancellation: &AgentCancellation,
	signal: S,
) -> anyhow::Result<T>
where
	F: Future<Output = T>,
	S: Future<Output = std::io::Result<()>>,
{
	tokio::pin!(future);
	tokio::pin!(signal);
	tokio::select! {
		result = &mut future => Ok(result),
		signal_result = &mut signal => {
			cancellation.cancel();
			let result = future.await;
			signal_result.context("listen for Ctrl-C")?;
			Ok(result)
		}
	}
}

pub(crate) fn finish_human_streams(
	markdown: &mut MarkdownStream,
	reasoning: &mut MarkdownStream,
	reasoning_active: &mut bool,
) -> anyhow::Result<()> {
	if *reasoning_active {
		output::stderr_line(&reasoning.finish())?;
		*reasoning_active = false;
	}
	output::stdout(&markdown.finish())
}

pub(crate) fn render_agent_event(
	event: &AgentEvent,
	json: bool,
	stderr_palette: Palette,
	markdown: &mut MarkdownStream,
	reasoning: &mut MarkdownStream,
	reasoning_active: &mut bool,
) -> anyhow::Result<()> {
	if json {
		return output::json_line(event);
	}
	match event {
		AgentEvent::TextDelta { text, .. } => {
			if *reasoning_active {
				output::stderr_line(&reasoning.finish())?;
				*reasoning_active = false;
			}
			let text = output::terminal_safe(text);
			output::stdout(&markdown.push(&text))?;
		}
		AgentEvent::ReasoningDelta { text, .. } => {
			*reasoning_active = true;
			let text = output::terminal_safe(text);
			output::stderr(&reasoning.push(&text))?;
		}
		AgentEvent::ToolCall { call, .. } => {
			let name = human_tool_name(&call.name);
			let arguments = serde_json::to_string(&call.arguments)
				.context("encode tool-call arguments for terminal preview")?;
			let arguments = bounded_tool_event_preview(&arguments);
			let arguments = output::terminal_safe_inline(&arguments);
			output::stderr_line(&stderr_palette.dim(&format!("→ {name}  {arguments}")))?;
		}
		AgentEvent::ToolCompleted {
			tool_name, output, ..
		} if output.is_error => {
			let tool_name = human_tool_name(tool_name);
			let content = bounded_tool_event_preview(&output.content);
			let content = super::output::terminal_safe_inline(&content);
			output::stderr_line(
				&stderr_palette.yellow(&format!("! {tool_name} failed · {content}")),
			)?;
		}
		AgentEvent::ApprovalResolved { decision, .. } => {
			let copy = approval_decision_copy(decision);
			if matches!(decision, emelex::agent::ApprovalDecision::AllowOnce) {
				output::stderr_line(&stderr_palette.green(&copy))?;
			} else {
				output::stderr_line(&stderr_palette.yellow(&copy))?;
			}
		}
		AgentEvent::TurnFailed { message, .. } => {
			let message = output::terminal_safe_inline(message);
			output::stderr_line(&stderr_palette.red(&format!("× Turn failed · {message}")))?;
		}
		_ => {}
	}
	Ok(())
}

fn human_tool_name(value: &str) -> String {
	let value = output::terminal_safe_inline(value);
	let words = value.replace(['_', '-'], " ");
	let mut chars = words.chars();
	let Some(first) = chars.next() else {
		return "Tool".to_string();
	};
	first.to_uppercase().chain(chars).collect::<String>()
}

fn approval_decision_copy(decision: &emelex::agent::ApprovalDecision) -> String {
	match decision {
		emelex::agent::ApprovalDecision::AllowOnce => "✓ Approved once".to_string(),
		emelex::agent::ApprovalDecision::Deny { reason } => {
			let reason = bounded_tool_event_preview(reason);
			let reason = output::terminal_safe_inline(&reason);
			format!("! Not approved · {reason}")
		}
		_ => "Approval resolved".to_string(),
	}
}

fn bounded_tool_event_preview(value: &str) -> String {
	if value.len() <= MAX_TOOL_EVENT_PREVIEW_BYTES {
		return value.to_string();
	}
	let mut end = MAX_TOOL_EVENT_PREVIEW_BYTES;
	while !value.is_char_boundary(end) {
		end -= 1;
	}
	let omitted = value.len() - end;
	let digest = hex::encode(Sha256::digest(value.as_bytes()));
	format!(
		"{}… [{omitted} bytes omitted; sha256:{digest}]",
		&value[..end]
	)
}

const fn thinking_mode(value: ThinkingArg) -> ThinkingMode {
	match value {
		ThinkingArg::Auto => ThinkingMode::Auto,
		ThinkingArg::On => ThinkingMode::On,
		ThinkingArg::Off => ThinkingMode::Off,
	}
}

fn request_max_tokens(requested: Option<usize>, effective: usize) -> Option<usize> {
	requested.map(|requested| requested.min(effective))
}

#[cfg(test)]
mod tests {
	use std::{
		io::Cursor,
		sync::atomic::{AtomicBool, Ordering},
	};

	use super::*;

	#[test]
	fn piped_prompt_is_bounded_and_validated() {
		assert_eq!(
			read_prompt_from(Cursor::new(b"hello".as_slice()), 5).expect("bounded prompt"),
			"hello"
		);
		assert!(
			read_prompt_from(Cursor::new(b"hello".as_slice()), 4)
				.expect_err("oversized prompt")
				.to_string()
				.contains("exceeds 4 bytes")
		);
		assert!(read_prompt_from(Cursor::new(b"  \n".as_slice()), 8).is_err());
		assert!(read_prompt_from(Cursor::new([0xff]), 1).is_err());
	}

	#[test]
	fn explicit_off_overrides_global_on_for_selection_load_and_request() {
		let plan = thinking_plan(ThinkingMode::On, Some(ThinkingArg::Off));

		assert_eq!(
			plan,
			ThinkingPlan {
				effective: ThinkingMode::Off,
				load: ThinkingMode::Off,
				request: Some(ThinkingMode::Off),
			}
		);
	}

	#[test]
	fn request_max_tokens_cannot_reexpand_loaded_checkpoint_ceiling() {
		assert_eq!(request_max_tokens(Some(4_096), 2_048), Some(2_048));
		assert_eq!(request_max_tokens(Some(512), 2_048), Some(512));
		assert_eq!(request_max_tokens(None, 2_048), None);
	}

	#[test]
	fn tool_event_preview_is_bounded_and_commits_to_full_payload() {
		let payload = format!("{}終", "x".repeat(MAX_TOOL_EVENT_PREVIEW_BYTES + 10));
		let preview = bounded_tool_event_preview(&payload);

		assert!(preview.len() < MAX_TOOL_EVENT_PREVIEW_BYTES + 160);
		assert!(preview.contains("bytes omitted"));
		assert!(preview.contains(&hex::encode(Sha256::digest(payload.as_bytes()))));
		assert!(!preview.contains('終'));
	}

	#[test]
	fn usage_footer_has_one_stable_shape_and_correct_round_grammar() {
		assert_eq!(
			usage_footer(1_500, 240, 42, None),
			"↑ 1.5k prompt · ↺ 240 cached · ↓ 42 generated"
		);
		assert_eq!(
			usage_footer(1_500, 240, 42, Some(1)),
			"↑ 1.5k prompt · ↺ 240 cached · ↓ 42 generated · 1 round"
		);
		assert_eq!(
			usage_footer(1_500, 240, 42, Some(2)),
			"↑ 1.5k prompt · ↺ 240 cached · ↓ 42 generated · 2 rounds"
		);
	}

	#[test]
	fn human_event_labels_are_calm_and_terminal_safe() {
		assert_eq!(human_tool_name("web_search"), "Web search");
		assert_eq!(human_tool_name(""), "Tool");

		let allowed = approval_decision_copy(&emelex::agent::ApprovalDecision::AllowOnce);
		assert_eq!(allowed, "✓ Approved once");

		let denied = approval_decision_copy(&emelex::agent::ApprovalDecision::Deny {
			reason: "unsafe\u{1b}]0;title\u{7}\nforged".to_string(),
		});
		assert!(denied.starts_with("! Not approved · "));
		assert!(!denied.contains("Deny"));
		assert!(!denied.contains('\u{1b}'));
		assert!(!denied.contains('\u{7}'));
		assert!(!denied.contains('\n'));
	}

	#[tokio::test]
	async fn cancellation_signal_is_observed_before_turn_returns() {
		let cancellation = AgentCancellation::new();
		let cleaned = Arc::new(AtomicBool::new(false));
		let cleaned_by_turn = Arc::clone(&cleaned);
		let turn = async {
			cancellation.cancelled().await;
			tokio::task::yield_now().await;
			cleaned_by_turn.store(true, Ordering::Release);
			7
		};

		let result = await_with_cancellation(turn, &cancellation, async { Ok(()) })
			.await
			.expect("signal handling");

		assert_eq!(result, 7);
		assert!(cancellation.is_cancelled());
		assert!(cleaned.load(Ordering::Acquire));
	}

	#[tokio::test]
	async fn signal_listener_error_is_reported_after_turn_cleanup() {
		let cancellation = AgentCancellation::new();
		let cleaned = Arc::new(AtomicBool::new(false));
		let cleaned_by_turn = Arc::clone(&cleaned);
		let turn = async {
			cancellation.cancelled().await;
			tokio::task::yield_now().await;
			cleaned_by_turn.store(true, Ordering::Release);
		};

		let error = await_with_cancellation(turn, &cancellation, async {
			Err(std::io::Error::other("signal backend failed"))
		})
		.await
		.expect_err("signal error");

		assert!(cancellation.is_cancelled());
		assert!(cleaned.load(Ordering::Acquire));
		assert!(error.to_string().contains("listen for Ctrl-C"));
		assert!(
			error
				.chain()
				.any(|source| source.to_string().contains("signal backend failed"))
		);
	}
}
