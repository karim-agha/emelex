//! Attended chat-turn activity presentation.

use std::{
	cell::{RefCell, RefMut},
	future::Future,
	rc::Rc,
	time::Duration,
};

use anyhow::Context as _;
use emelex::agent::{AgentCancellation, AgentEvent};
use tokio::time::{Instant, MissedTickBehavior};

use super::{
	output,
	style::Palette,
	terminal_ui::{LiveRegion, fit_line},
};

const FRAME_INTERVAL: Duration = Duration::from_millis(120);
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ActivityState {
	Hidden,
	Thinking,
	Running(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ActivityAction {
	Clear,
	Show(ActivityState),
}

impl ChatActivity {
	/// Create an attended stderr region, or a no-op presenter when disabled.
	pub(crate) fn new(enabled: bool, palette: Palette) -> Self {
		Self {
			inner: Rc::new(RefCell::new(ActivityInner {
				region: enabled.then(LiveRegion::stderr),
				palette,
				state: ActivityState::Hidden,
				frame: 0,
			})),
		}
	}

	/// Clear or replace activity before another event renderer sees `event`.
	pub(crate) fn before_event(&self, event: &AgentEvent) -> anyhow::Result<()> {
		self.inner()?.apply(event_action(event))
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
		if let Err(error) = self
			.inner()?
			.apply(ActivityAction::Show(ActivityState::Thinking))
		{
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
	fn apply(&mut self, action: ActivityAction) -> anyhow::Result<()> {
		match action {
			ActivityAction::Clear => self.clear(),
			ActivityAction::Show(state) if self.state == state => Ok(()),
			ActivityAction::Show(state) => {
				self.state = state;
				self.frame = 0;
				self.draw()
			}
		}
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
		let frame = activity_frame(&self.state, self.frame, self.palette);
		let columns = usize::from(region.size().1).max(1);
		region
			.draw(&fit_line(&frame, columns))
			.context("draw chat activity")
	}

	fn clear(&mut self) -> anyhow::Result<()> {
		self.state = ActivityState::Hidden;
		self.frame = 0;
		if let Some(region) = self.region.as_mut() {
			region.clear().context("clear chat activity")?;
		}
		Ok(())
	}
}

fn event_action(event: &AgentEvent) -> ActivityAction {
	match event {
		AgentEvent::TurnStarted { .. } | AgentEvent::ModelStarted { .. } => {
			ActivityAction::Show(ActivityState::Thinking)
		}
		AgentEvent::ToolStarted { tool_name, .. } => {
			ActivityAction::Show(ActivityState::Running(tool_name.clone()))
		}
		_ => ActivityAction::Clear,
	}
}

fn activity_frame(state: &ActivityState, frame: usize, palette: Palette) -> String {
	let spinner = palette.cyan(SPINNER_FRAMES[frame % SPINNER_FRAMES.len()]);
	let label = match state {
		ActivityState::Hidden => String::new(),
		ActivityState::Thinking => "Thinking…".to_string(),
		ActivityState::Running(tool_name) => {
			format!("Running {}…", human_tool_name(tool_name))
		}
	};
	format!("{spinner} {}", palette.dim(&label))
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

	#[test]
	fn events_map_to_presentation_actions_before_rendering() {
		let turn_id = Uuid::nil();
		assert_eq!(
			event_action(&AgentEvent::ModelStarted { turn_id, round: 1 }),
			ActivityAction::Show(ActivityState::Thinking)
		);
		assert_eq!(
			event_action(&AgentEvent::ToolStarted {
				call_id: "call-1".to_string(),
				tool_name: "web_fetch".to_string(),
			}),
			ActivityAction::Show(ActivityState::Running("web_fetch".to_string()))
		);
		assert_eq!(
			event_action(&AgentEvent::TextDelta {
				turn_id,
				round: 1,
				text: "hello".to_string(),
			}),
			ActivityAction::Clear
		);
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
		assert_eq!(event_action(&approval_requested), ActivityAction::Clear);
		assert_eq!(
			event_action(&AgentEvent::Cancelled { turn_id }),
			ActivityAction::Clear
		);
	}

	#[test]
	fn running_frames_neutralize_untrusted_tool_names() {
		let palette = Palette::stderr(ColorMode::Never);
		let frame = activity_frame(
			&ActivityState::Running("shell\n\t\u{1b}[31m\u{202e}".to_string()),
			0,
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
