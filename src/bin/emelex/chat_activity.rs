//! Attended chat-turn activity presentation.

use std::{
	cell::{RefCell, RefMut},
	fmt::Write as _,
	future::Future,
	rc::Rc,
	time::Duration,
};

use anyhow::Context as _;
use emelex::{
	agent::{AgentCancellation, AgentEvent},
	generation::{FinishReason, GenerationProgress, GenerationProgressPhase, Usage},
};
use tokio::time::{Instant, MissedTickBehavior};

use super::{
	output,
	style::{self, Palette},
	terminal_ui::{LiveRegion, fit_line},
};

const FRAME_INTERVAL: Duration = Duration::from_millis(120);
const STREAM_PREVIEW_CHARS: usize = 160;
const SPINNER_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// Cloneable handle shared by one turn future and its single-task driver.
#[derive(Clone)]
pub(crate) struct ChatActivity {
	inner: Rc<RefCell<ActivityInner>>,
}

struct ActivityInner {
	region: Option<LiveRegion>,
	palette: Palette,
	state: ActivityState,
	frame: usize,
	completed_usage: Usage,
	progress: Option<GenerationProgress>,
	decode_timing: Option<DecodeTiming>,
	answer_terminal: bool,
	redraw_after_event: bool,
	preview_kind: Option<PreviewKind>,
	preview: String,
	preview_truncated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ActivityState {
	Hidden,
	Thinking,
	PreparingPrompt,
	CheckingContext,
	Prefilling,
	Answering,
	PreparingTools,
	Running(String),
	RecordingToolResult,
	Saving,
}

#[derive(Debug, Clone, Copy)]
struct DecodeTiming {
	first_completion_tokens: u64,
	started_at: Instant,
}

#[derive(Debug, Clone, Copy)]
struct ActivityPreview<'a> {
	text: &'a str,
	truncated: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreviewKind {
	Reasoning,
	Answer,
}

impl ChatActivity {
	/// Create an attended stderr region, or a no-op presenter when disabled.
	pub(crate) fn new(enabled: bool, answer_terminal: bool, palette: Palette) -> Self {
		Self {
			inner: Rc::new(RefCell::new(ActivityInner {
				region: enabled.then(LiveRegion::stderr),
				palette,
				state: ActivityState::Hidden,
				frame: 0,
				completed_usage: Usage::default(),
				progress: None,
				decode_timing: None,
				answer_terminal,
				redraw_after_event: false,
				preview_kind: None,
				preview: String::new(),
				preview_truncated: false,
			})),
		}
	}

	/// Remove the live line before persistent output for `event` is rendered.
	pub(crate) fn before_event(&self, event: &AgentEvent) -> anyhow::Result<()> {
		let mut inner = self.inner()?;
		let erase = event_requires_activity_erase(event, &inner.state, inner.answer_terminal);
		inner.redraw_after_event = erase;
		if erase {
			inner.erase()?;
		}
		Ok(())
	}

	/// Advance and redraw the live state after persistent event output.
	pub(crate) fn after_event(&self, event: &AgentEvent) -> anyhow::Result<()> {
		self.inner()?.observe(event)
	}

	/// Drive one turn, its 120 ms animation, and its sole Ctrl-C listener on one task.
	pub(crate) async fn drive<F, T>(
		&self,
		future: F,
		cancellation: &AgentCancellation,
	) -> anyhow::Result<T>
	where
		F: Future<Output = T>,
	{
		if let Err(error) = self.inner()?.begin_turn() {
			return Err(self.cleanup_error(error));
		}

		let start = Instant::now() + FRAME_INTERVAL;
		let mut interval = tokio::time::interval_at(start, FRAME_INTERVAL);
		interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
		let enabled = match self.enabled() {
			Ok(enabled) => enabled,
			Err(error) => return Err(self.cleanup_error(error)),
		};
		let signal = tokio::signal::ctrl_c();
		tokio::pin!(future);
		tokio::pin!(signal);

		loop {
			tokio::select! {
				biased;
				signal_result = &mut signal => {
					cancellation.cancel();
					let before_wait = self.clear();
					let result = future.await;
					let after_wait = self.clear();
					if let Err(error) = signal_result.context("listen for Ctrl-C") {
						let error = with_clear_context(error, before_wait, "before cancellation wait");
						return Err(with_clear_context(error, after_wait, "after cancellation wait"));
					}
					if let Err(error) = before_wait {
						return Err(with_clear_context(error, after_wait, "after cancellation wait"));
					}
					after_wait?;
					return Ok(result);
				}
				result = &mut future => {
					self.clear()?;
					return Ok(result);
				}
				_ = interval.tick(), if enabled => {
					let tick_result = {
						let mut inner = self.inner()?;
						inner.tick()
					};
					if let Err(error) = tick_result {
						cancellation.cancel();
						let before_wait = self.clear();
						drop(future.await);
						let after_wait = self.clear();
						let error = with_clear_context(error, before_wait, "before cancellation wait");
						return Err(with_clear_context(error, after_wait, "after cancellation wait"));
					}
				}
			}
		}
	}

	fn enabled(&self) -> anyhow::Result<bool> {
		Ok(self.inner()?.region.is_some())
	}

	fn clear(&self) -> anyhow::Result<()> {
		self.inner()?.clear()
	}

	fn cleanup_error(&self, error: anyhow::Error) -> anyhow::Error {
		match self.clear() {
			Ok(()) => error,
			Err(cleanup) => error.context(format!(
				"clear chat activity after presentation failure: {cleanup:#}"
			)),
		}
	}

	fn inner(&self) -> anyhow::Result<RefMut<'_, ActivityInner>> {
		self.inner
			.try_borrow_mut()
			.map_err(|_| anyhow::anyhow!("chat activity was accessed reentrantly"))
	}
}

fn with_clear_context(
	error: anyhow::Error,
	cleanup: anyhow::Result<()>,
	phase: &str,
) -> anyhow::Error {
	match cleanup {
		Ok(()) => error,
		Err(cleanup) => error.context(format!("clear chat activity {phase}: {cleanup:#}")),
	}
}

impl ActivityInner {
	fn begin_turn(&mut self) -> anyhow::Result<()> {
		self.completed_usage = Usage::default();
		self.progress = None;
		self.decode_timing = None;
		self.clear_preview();
		self.show(ActivityState::Thinking, true)
	}

	fn observe(&mut self, event: &AgentEvent) -> anyhow::Result<()> {
		let redraw_after_event = std::mem::take(&mut self.redraw_after_event);
		match event {
			AgentEvent::TurnStarted { .. } => self.begin_turn(),
			AgentEvent::ModelStarted { .. } => {
				self.progress = None;
				self.decode_timing = None;
				self.clear_preview();
				self.show(ActivityState::PreparingPrompt, true)
			}
			AgentEvent::ModelProgress { progress, .. } => self.observe_progress(*progress),
			AgentEvent::TextDelta { text, .. } => {
				if self.answer_terminal {
					self.update_preview(PreviewKind::Answer, text);
				} else {
					self.clear_preview();
				}
				self.show(ActivityState::Answering, redraw_after_event)
			}
			AgentEvent::ReasoningDelta { text, .. } => {
				self.update_preview(PreviewKind::Reasoning, text);
				self.show(ActivityState::Thinking, redraw_after_event)
			}
			AgentEvent::ToolCall { .. } | AgentEvent::ApprovalResolved { .. } => {
				self.clear_preview();
				self.show(ActivityState::PreparingTools, true)
			}
			AgentEvent::ApprovalRequested { .. } => {
				self.clear_preview();
				self.hide()
			}
			AgentEvent::ToolStarted { tool_name, .. } => {
				self.clear_preview();
				self.show(ActivityState::Running(tool_name.clone()), true)
			}
			AgentEvent::ToolCompleted { .. } => {
				self.clear_preview();
				self.show(ActivityState::RecordingToolResult, true)
			}
			AgentEvent::ModelCompleted {
				finish_reason,
				usage,
				..
			} => {
				self.add_completed_usage(usage);
				self.progress = None;
				self.decode_timing = None;
				self.clear_preview();
				let state = if matches!(finish_reason, FinishReason::ToolCalls) {
					ActivityState::PreparingTools
				} else {
					ActivityState::Saving
				};
				self.show(state, true)
			}
			AgentEvent::TurnCompleted { .. }
			| AgentEvent::Cancelled { .. }
			| AgentEvent::TurnFailed { .. } => self.clear(),
			_ => Ok(()),
		}
	}

	fn observe_progress(&mut self, progress: GenerationProgress) -> anyhow::Result<()> {
		self.progress = Some(progress);
		match progress.phase {
			GenerationProgressPhase::Prompt => {
				self.decode_timing = None;
				self.show(ActivityState::CheckingContext, true)
			}
			GenerationProgressPhase::Prefill => {
				self.decode_timing = None;
				self.show(ActivityState::Prefilling, true)
			}
			GenerationProgressPhase::Decode => {
				if self.decode_timing.is_none() {
					self.decode_timing = Some(DecodeTiming {
						first_completion_tokens: progress.completion_tokens,
						started_at: Instant::now(),
					});
				}
				if !matches!(
					self.state,
					ActivityState::Thinking | ActivityState::Answering
				) {
					self.show(ActivityState::Thinking, false)?;
				}
				Ok(())
			}
			_ => Ok(()),
		}
	}

	const fn add_completed_usage(&mut self, usage: &Usage) {
		self.completed_usage.prompt_tokens = self
			.completed_usage
			.prompt_tokens
			.saturating_add(usage.prompt_tokens);
		self.completed_usage.cached_tokens = self
			.completed_usage
			.cached_tokens
			.saturating_add(usage.cached_tokens);
		self.completed_usage.completion_tokens = self
			.completed_usage
			.completion_tokens
			.saturating_add(usage.completion_tokens);
	}

	fn update_preview(&mut self, kind: PreviewKind, delta: &str) {
		if self.preview_kind != Some(kind) {
			self.clear_preview();
			self.preview_kind = Some(kind);
		}
		if let Some(newline) = delta.rfind('\n') {
			self.preview.clear();
			self.preview_truncated = false;
			self.preview.push_str(&delta[newline + 1..]);
		} else {
			self.preview.push_str(delta);
		}
		let excess = self
			.preview
			.chars()
			.count()
			.saturating_sub(STREAM_PREVIEW_CHARS);
		if excess > 0
			&& let Some((boundary, _)) = self.preview.char_indices().nth(excess)
		{
			self.preview.drain(..boundary);
			self.preview_truncated = true;
		}
	}

	fn clear_preview(&mut self) {
		self.preview_kind = None;
		self.preview.clear();
		self.preview_truncated = false;
	}

	fn show(&mut self, state: ActivityState, redraw: bool) -> anyhow::Result<()> {
		if self.state != state {
			self.state = state;
			self.frame = 0;
		}
		if redraw {
			self.draw()?;
		}
		Ok(())
	}

	fn hide(&mut self) -> anyhow::Result<()> {
		self.state = ActivityState::Hidden;
		self.frame = 0;
		self.erase()
	}

	fn tick(&mut self) -> anyhow::Result<()> {
		if matches!(self.state, ActivityState::Hidden) {
			return Ok(());
		}
		self.frame = (self.frame + 1) % SPINNER_FRAMES.len();
		self.draw()
	}

	fn draw(&mut self) -> anyhow::Result<()> {
		let Some(region) = self.region.as_mut() else {
			return Ok(());
		};
		let frame = activity_frame(
			&self.state,
			self.frame,
			self.completed_usage,
			self.progress,
			self.decode_timing,
			ActivityPreview {
				text: &self.preview,
				truncated: self.preview_truncated,
			},
			Instant::now(),
			self.palette,
		);
		let columns = usize::from(region.size().1).max(1);
		let frame = frame
			.split('\n')
			.map(|line| fit_line(line, columns))
			.collect::<Vec<_>>()
			.join("\n");
		region.draw(&frame).context("draw chat activity")
	}

	fn erase(&mut self) -> anyhow::Result<()> {
		if let Some(region) = self.region.as_mut() {
			region.clear().context("clear chat activity")?;
		}
		Ok(())
	}

	fn clear(&mut self) -> anyhow::Result<()> {
		self.state = ActivityState::Hidden;
		self.frame = 0;
		self.completed_usage = Usage::default();
		self.progress = None;
		self.decode_timing = None;
		self.redraw_after_event = false;
		self.clear_preview();
		self.erase()
	}
}

fn event_requires_activity_erase(
	event: &AgentEvent,
	state: &ActivityState,
	answer_terminal: bool,
) -> bool {
	match event {
		AgentEvent::TextDelta { text, .. } => {
			matches!(state, ActivityState::Thinking) || (answer_terminal && text.contains('\n'))
		}
		AgentEvent::ReasoningDelta { text, .. } => text.contains('\n'),
		AgentEvent::ToolCall { .. }
		| AgentEvent::ModelStarted { .. }
		| AgentEvent::ApprovalRequested { .. }
		| AgentEvent::ToolCompleted { .. }
		| AgentEvent::ApprovalResolved { .. }
		| AgentEvent::ModelCompleted { .. }
		| AgentEvent::Cancelled { .. }
		| AgentEvent::TurnFailed { .. } => true,
		_ => false,
	}
}

fn activity_frame(
	state: &ActivityState,
	frame: usize,
	completed_usage: Usage,
	progress: Option<GenerationProgress>,
	decode_timing: Option<DecodeTiming>,
	preview: ActivityPreview<'_>,
	now: Instant,
	palette: Palette,
) -> String {
	let spinner = palette.cyan(SPINNER_FRAMES[frame % SPINNER_FRAMES.len()]);
	let label = match state {
		ActivityState::Hidden => String::new(),
		ActivityState::Thinking => "Thinking…".to_string(),
		ActivityState::PreparingPrompt => "Preparing prompt…".to_string(),
		ActivityState::CheckingContext => "Checking context…".to_string(),
		ActivityState::Prefilling => "Prefilling…".to_string(),
		ActivityState::Answering => "Answering…".to_string(),
		ActivityState::PreparingTools => "Preparing tools…".to_string(),
		ActivityState::Running(tool_name) => {
			format!("Running {}…", human_tool_name(tool_name))
		}
		ActivityState::RecordingToolResult => "Recording tool result…".to_string(),
		ActivityState::Saving => "Saving response…".to_string(),
	};
	let usage = live_usage(completed_usage, progress, decode_timing, now);
	let label = usage.map_or_else(|| label.clone(), |usage| format!("{label} · {usage}"));
	let status = format!("{spinner} {}", palette.dim(&label));
	if preview.text.is_empty() {
		return status;
	}
	let text = output::terminal_safe_inline(preview.text);
	let prefix = if preview.truncated { "…" } else { "" };
	format!("{}\n{status}", palette.dim(&format!("{prefix}{text}")))
}

fn live_usage(
	completed: Usage,
	progress: Option<GenerationProgress>,
	decode_timing: Option<DecodeTiming>,
	now: Instant,
) -> Option<String> {
	let current_prompt = progress.map_or(0, |value| value.prompt_tokens);
	let current_cached = progress.and_then(|value| value.cached_tokens);
	let current_completion = progress.map_or(0, |value| value.completion_tokens);
	let prompt = completed.prompt_tokens.saturating_add(current_prompt);
	let cached = completed
		.cached_tokens
		.saturating_add(current_cached.unwrap_or(0));
	let completion = completed
		.completion_tokens
		.saturating_add(current_completion);
	if progress.is_none() && prompt == 0 && cached == 0 && completion == 0 {
		return None;
	}
	let mut usage = format!("↑{}", style::tokens(prompt));
	if current_cached.is_some() || cached > 0 {
		let _ = write!(usage, " ↺{}", style::tokens(cached));
	}
	let _ = write!(usage, " ↓{}", style::tokens(completion));
	if let Some(progress) = progress {
		if matches!(
			progress.phase,
			GenerationProgressPhase::Prompt | GenerationProgressPhase::Prefill
		) {
			let reserved = progress
				.prompt_tokens
				.saturating_add(progress.max_output_tokens);
			let _ = write!(
				usage,
				" · {}/{} ctx",
				style::tokens(reserved),
				style::tokens(progress.context_limit)
			);
		}
		if matches!(progress.phase, GenerationProgressPhase::Decode)
			&& let Some(timing) = decode_timing
			&& let Some(speed) = tokens_per_second(
				progress.completion_tokens,
				timing.first_completion_tokens,
				now.saturating_duration_since(timing.started_at),
			) {
			let _ = write!(usage, " · {speed:.1} tok/s");
		}
	}
	Some(usage)
}

fn tokens_per_second(completion: u64, first: u64, elapsed: Duration) -> Option<f64> {
	let generated = completion.saturating_sub(first);
	let seconds = elapsed.as_secs_f64();
	if generated == 0 || seconds <= f64::EPSILON {
		return None;
	}
	Some(generated as f64 / seconds)
}

fn human_tool_name(value: &str) -> String {
	let words = output::terminal_safe_inline(value).replace(['_', '-'], " ");
	let mut chars = words.chars();
	let Some(first) = chars.next() else {
		return "Tool".to_string();
	};
	first.to_uppercase().chain(chars).collect()
}

#[cfg(test)]
mod tests {
	use uuid::Uuid;

	use super::*;
	use crate::style::ColorMode;

	fn progress_event(phase: &str, completion_tokens: u64) -> AgentEvent {
		serde_json::from_value(serde_json::json!({
			"type": "model_progress",
			"turn_id": Uuid::nil(),
			"round": 1,
			"progress": {
				"phase": phase,
				"prompt_tokens": 44_167,
				"cached_tokens": if phase == "prompt" { None } else { Some(12_000_u64) },
				"completion_tokens": completion_tokens,
				"max_output_tokens": 4_096,
				"context_limit": 65_536
			}
		}))
		.expect("progress event fixture should decode")
	}

	#[test]
	fn lifecycle_keeps_relevant_status_between_model_and_tools() {
		let mut inner = ActivityInner {
			region: None,
			palette: Palette::stderr(ColorMode::Never),
			state: ActivityState::Hidden,
			frame: 0,
			completed_usage: Usage::default(),
			progress: None,
			decode_timing: None,
			answer_terminal: true,
			redraw_after_event: false,
			preview_kind: None,
			preview: String::new(),
			preview_truncated: false,
		};
		inner
			.observe(&AgentEvent::ModelStarted {
				turn_id: Uuid::nil(),
				round: 1,
			})
			.expect("model status");
		assert_eq!(inner.state, ActivityState::PreparingPrompt);
		inner
			.observe(&progress_event("prefill", 0))
			.expect("prefill status");
		assert_eq!(inner.state, ActivityState::Prefilling);
		inner
			.observe(&AgentEvent::ToolStarted {
				call_id: "call-1".to_string(),
				tool_name: "web_fetch".to_string(),
			})
			.expect("tool status");
		assert_eq!(inner.state, ActivityState::Running("web_fetch".to_string()));
	}

	#[test]
	fn progress_frame_shows_context_usage_and_decode_speed() {
		let palette = Palette::stderr(ColorMode::Never);
		let AgentEvent::ModelProgress {
			progress: prompt, ..
		} = progress_event("prompt", 0)
		else {
			panic!("expected progress fixture");
		};
		let prompt_frame = activity_frame(
			&ActivityState::CheckingContext,
			0,
			Usage::default(),
			Some(prompt),
			None,
			ActivityPreview {
				text: "",
				truncated: false,
			},
			Instant::now(),
			palette,
		);
		assert!(prompt_frame.contains("48.3k/65.5k ctx"));
		assert!(prompt_frame.contains("↑44.2k"));

		let AgentEvent::ModelProgress {
			progress: decode, ..
		} = progress_event("decode", 21)
		else {
			panic!("expected progress fixture");
		};
		let now = Instant::now();
		let decode_frame = activity_frame(
			&ActivityState::Thinking,
			0,
			Usage::default(),
			Some(decode),
			Some(DecodeTiming {
				first_completion_tokens: 1,
				started_at: now - Duration::from_secs(2),
			}),
			ActivityPreview {
				text: "",
				truncated: false,
			},
			now,
			palette,
		);
		assert!(decode_frame.contains("↺12k"));
		assert!(decode_frame.contains("↓21"));
		assert!(decode_frame.contains("10.0 tok/s"));
	}

	#[test]
	fn only_stream_flushes_and_reasoning_transitions_erase_live_activity() {
		let turn_id = Uuid::nil();
		let partial_text = AgentEvent::TextDelta {
			turn_id,
			round: 1,
			text: "hello".to_string(),
		};
		assert!(!event_requires_activity_erase(
			&partial_text,
			&ActivityState::Answering,
			true
		));
		assert!(event_requires_activity_erase(
			&partial_text,
			&ActivityState::Thinking,
			true
		));
		assert!(!event_requires_activity_erase(
			&AgentEvent::TextDelta {
				turn_id,
				round: 1,
				text: "redirected line\n".to_string(),
			},
			&ActivityState::Answering,
			false
		));
		assert!(!event_requires_activity_erase(
			&AgentEvent::ReasoningDelta {
				turn_id,
				round: 1,
				text: "partial thought".to_string(),
			},
			&ActivityState::Thinking,
			true
		));
		assert!(event_requires_activity_erase(
			&AgentEvent::ReasoningDelta {
				turn_id,
				round: 1,
				text: "complete thought\n".to_string(),
			},
			&ActivityState::Thinking,
			true
		));
		assert!(event_requires_activity_erase(
			&AgentEvent::TextDelta {
				turn_id,
				round: 1,
				text: "complete line\n".to_string(),
			},
			&ActivityState::Answering,
			true
		));
		assert!(!event_requires_activity_erase(
			&progress_event("decode", 1),
			&ActivityState::Thinking,
			true
		));
		let approval_requested: AgentEvent = serde_json::from_value(serde_json::json!({
			"type": "approval_requested",
			"context": {
				"call_id": "call-1",
				"tool_name": "web_fetch",
				"arguments": {"url": "https://example.com"},
				"workspace_root": "/tmp/workspace",
				"workspace_device": 1,
				"workspace_inode": 2,
				"reason": "network access"
			}
		}))
		.expect("approval event fixture should decode");
		assert!(event_requires_activity_erase(
			&approval_requested,
			&ActivityState::Thinking,
			true
		));
		assert!(event_requires_activity_erase(
			&AgentEvent::Cancelled { turn_id },
			&ActivityState::Answering,
			true
		));
	}

	#[test]
	fn streamed_deltas_restore_and_animate_live_activity() {
		let mut inner = ActivityInner {
			region: None,
			palette: Palette::stderr(ColorMode::Never),
			state: ActivityState::Thinking,
			frame: 3,
			completed_usage: Usage::default(),
			progress: None,
			decode_timing: None,
			answer_terminal: true,
			redraw_after_event: false,
			preview_kind: None,
			preview: String::new(),
			preview_truncated: false,
		};
		inner
			.observe(&AgentEvent::TextDelta {
				turn_id: Uuid::nil(),
				round: 1,
				text: "partial answer".to_string(),
			})
			.expect("partial answer state");
		assert_eq!(inner.state, ActivityState::Answering);
		let frame = inner.frame;
		inner.tick().expect("animate answer status");
		assert_ne!(inner.frame, frame, "decode status must remain animated");
	}

	#[test]
	fn stream_preview_is_bounded_and_terminal_safe() {
		let mut inner = ActivityInner {
			region: None,
			palette: Palette::stderr(ColorMode::Never),
			state: ActivityState::Answering,
			frame: 0,
			completed_usage: Usage::default(),
			progress: None,
			decode_timing: None,
			answer_terminal: true,
			redraw_after_event: false,
			preview_kind: None,
			preview: String::new(),
			preview_truncated: false,
		};
		inner.update_preview(
			PreviewKind::Answer,
			&format!("{}tail\u{1b}[31m", "x".repeat(STREAM_PREVIEW_CHARS + 20)),
		);
		assert_eq!(inner.preview.chars().count(), STREAM_PREVIEW_CHARS);
		assert!(inner.preview_truncated);
		let frame = activity_frame(
			&inner.state,
			inner.frame,
			inner.completed_usage,
			inner.progress,
			inner.decode_timing,
			ActivityPreview {
				text: &inner.preview,
				truncated: inner.preview_truncated,
			},
			Instant::now(),
			inner.palette,
		);
		assert!(frame.starts_with('…'));
		assert!(frame.contains("tail"));
		assert!(!frame.contains('\u{1b}'));
		assert_eq!(frame.matches('\n').count(), 1);
	}

	#[test]
	fn running_frames_neutralize_untrusted_tool_names() {
		let palette = Palette::stderr(ColorMode::Never);
		let frame = activity_frame(
			&ActivityState::Running("shell\n\t\u{1b}[31m\u{202e}".to_string()),
			0,
			Usage::default(),
			None,
			None,
			ActivityPreview {
				text: "",
				truncated: false,
			},
			Instant::now(),
			palette,
		);
		assert!(frame.starts_with("⠋ Running Shell"));
		assert!(!frame.contains('\n'));
		assert!(!frame.contains('\t'));
		assert!(!frame.contains('\u{1b}'));
		assert!(!frame.contains('\u{202e}'));
		assert!(frame.contains('\u{240a}'));
		assert!(frame.contains('\u{2409}'));
		assert!(frame.contains('\u{241b}'));
	}
}
