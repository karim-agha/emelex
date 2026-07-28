//! A long multi-turn chat session on a local MLX model, with the KV
//! prompt cache doing the heavy lifting.
//!
//! Every turn re-sends the full conversation (rig's stateless model),
//! and the engine's prompt cache reuses the KV state of the shared
//! prefix - watch the `cached/input` column climb as the conversation
//! grows: by the late turns almost the entire prompt is served from
//! cache. Facts stated in early turns are referenced in late turns to
//! verify real history retention.
//!
//! ```sh
//! cargo run -p emelex --release --example chat -- \
//!   "$EMELEX_TEST_MODEL"
//! # or interactively:
//! cargo run -p emelex --release --example chat -- <model-dir> --interactive
//! ```

// Example code: unwraps and panics are acceptable here.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::io::{BufRead, Write as _};

use emelex::ReasoningExt as _;
use rig_core::completion::{Message, Prompt};

const SCRIPT: &[&str] = &[
	"Hi! My name is Priya and I run the payments platform team.",
	"We deploy to production every Tuesday and Friday morning.",
	"Our checkout error budget is 0.5% per month.",
	"The primary database is Postgres 16, with a replica in eu-west-1.",
	"Our on-call rotation is weekly and hands over on Mondays at 10:00.",
	"This month we have already burned 0.3% of the error budget.",
	"We use feature flags for every risky change.",
	"The two most fragile services are payments-api and ledger-sync.",
	"Card payments time out after 5 seconds at the gateway.",
	"Remind me: what is my name and which team do I run?",
	"On which days do we deploy to production?",
	"How much error budget do we have left this month?",
	"Which database do we run, and where is the replica?",
	"Summarize everything you know about my team in three bullets.",
];

#[tokio::main]
#[allow(clippy::significant_drop_tightening)] // stdin lock spans the REPL loop
async fn main() -> Result<(), Box<dyn std::error::Error>> {
	let model_dir = std::env::args()
		.nth(1)
		.expect("usage: chat <mlx-model-dir> [--interactive]");
	let interactive = std::env::args().any(|a| a == "--interactive");

	let agent = emelex::Client::from_path(model_dir)?
		.agent()
		.preamble(
			"You are a concise assistant. Answer from the conversation so far; keep \
			 replies to one or two sentences.",
		)
		.enable_thinking(false)
		.build();

	// The `Chat` trait (`agent.chat(line, &mut history)`) is the compact
	// version of this loop; the explicit form below is equivalent but
	// also surfaces per-turn usage so the cache behavior is visible.
	let mut history: Vec<Message> = Vec::new();
	println!(
		"{:>4}  {:>7}  {:>13}  answer",
		"turn", "history", "cached/input"
	);

	let stdin = std::io::stdin();
	let mut interactive_lines = interactive.then(|| stdin.lock().lines());
	let mut turn = 0usize;
	loop {
		let line: String = if let Some(lines) = interactive_lines.as_mut() {
			print!("> ");
			std::io::stdout().flush()?;
			match lines.next() {
				Some(Ok(line)) if !line.trim().is_empty() => line,
				Some(Ok(_)) => continue,
				_ => break,
			}
		} else {
			match SCRIPT.get(turn) {
				Some(line) => (*line).to_string(),
				None => break,
			}
		};
		turn += 1;

		let response = agent
			.prompt(line.as_str())
			.history(history.clone())
			.extended_details()
			.await?;
		if let Some(messages) = response.messages {
			history.extend(messages);
		}
		let one_line = response.output.replace('\n', " ");
		let shown = one_line.chars().take(72).collect::<String>();
		println!(
			"{turn:>4}  {:>7}  {:>6}/{:<6}  {shown}",
			history.len(),
			response.usage.cached_input_tokens,
			response.usage.input_tokens
		);
	}

	Ok(())
}
