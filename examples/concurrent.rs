//! Concurrency, contention, and cancellation across TWO locally loaded
//! models.
//!
//! Each `emelex::Client` owns a dedicated inference thread, so two
//! clients generate truly in parallel while requests on one client
//! queue FIFO. Three phases:
//!
//! - **A: parallel load.** Four prompts and one stream spread over two clients
//!   via `tokio::join!` - per-request wall times show same-client serialization
//!   vs cross-client parallelism.
//! - **B: cancellation.** A prompt cut off by `tokio::time::timeout` (dropping
//!   the future aborts generation at the next token) and a stream dropped after
//!   a few chunks (channel closure aborts the decode loop); the client must
//!   answer a fresh prompt right after.
//! - **C: same-client contention.** Three simultaneous prompts on one client,
//!   the middle one cancelled while queued - neighbors must be undisturbed.
//!
//! Requires enough RAM for two copies of the checkpoint.
//!
//! ```sh
//! cargo run -p emelex --release --example concurrent -- \
//!   "$EMELEX_TEST_MODEL"
//! ```

// Example code: unwraps and panics are acceptable here.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::time::{Duration, Instant};

use emelex::ReasoningExt as _;
use futures::StreamExt;
use rig_core::{agent::Agent, completion::Prompt, prelude::StreamingPrompt};

fn quick_agent(client: &emelex::Client) -> Agent<emelex::CompletionModel> {
	client
		.agent()
		.preamble("Answer in one short sentence.")
		.enable_thinking(false)
		.build()
}

async fn timed_prompt(agent: &Agent<emelex::CompletionModel>, label: &str, prompt: &str) -> String {
	let start = Instant::now();
	let answer = agent.prompt(prompt).await.expect("prompt should succeed");
	println!(
		"  [{label}] {:>5.1}s  {}",
		start.elapsed().as_secs_f32(),
		answer
			.replace('\n', " ")
			.chars()
			.take(60)
			.collect::<String>()
	);
	answer
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
	let model_dir = std::env::args()
		.nth(1)
		.expect("usage: concurrent <mlx-model-dir>");

	println!("== loading the checkpoint twice ==");
	let start = Instant::now();
	let client_a = emelex::Client::from_path(&model_dir)?;
	println!("  client A loaded in {:.1}s", start.elapsed().as_secs_f32());
	let start = Instant::now();
	let client_b = emelex::Client::from_path(&model_dir)?;
	println!("  client B loaded in {:.1}s", start.elapsed().as_secs_f32());

	let agent_a = quick_agent(&client_a);
	let agent_b = quick_agent(&client_b);

	// --- Phase A: parallel across clients, FIFO within one. -----------
	println!("\n== phase A: 4 prompts + 1 stream across two clients ==");
	let stream_task = async {
		let mut stream = agent_b.stream_prompt("Name three colors.").await;
		let mut chunks = 0usize;
		while let Some(item) = stream.next().await {
			item.expect("stream item should be ok");
			chunks += 1;
		}
		println!("  [B/stream] completed with {chunks} chunks");
	};
	tokio::join!(
		timed_prompt(&agent_a, "A/1", "What is 2 + 2?"),
		timed_prompt(&agent_b, "B/1", "What is 10 * 10?"),
		timed_prompt(&agent_a, "A/2", "Name a primary color."),
		timed_prompt(&agent_b, "B/2", "Name a big city."),
		stream_task,
	);

	// --- Phase B: cancellation, then prove the client still works. ----
	println!("\n== phase B: timeout + dropped stream, then recovery ==");
	// A dedicated verbose agent: the terse one-sentence agents above
	// would finish before any reasonable timeout fires.
	let verbose_agent = client_a
		.agent()
		.preamble("You are a thorough, long-form writer.")
		.enable_thinking(false)
		.build();
	let cut = tokio::time::timeout(
		Duration::from_secs(3),
		verbose_agent
			.prompt("Write a detailed 1000 word essay about the history of ocean navigation."),
	)
	.await;
	println!(
		"  timeout fired: {}",
		if cut.is_err() {
			"yes (future dropped mid-generation)"
		} else {
			"no (model finished early)"
		}
	);
	let mut stream = agent_a
		.stream_prompt("Count from 1 to 200, one number per line.")
		.await;
	let mut taken = 0usize;
	while let Some(item) = stream.next().await {
		item.expect("stream item should be ok");
		taken += 1;
		if taken >= 5 {
			break;
		}
	}
	drop(stream);
	println!("  stream dropped after {taken} chunks");
	timed_prompt(&agent_a, "A/recovery", "Say the word: recovered").await;

	// --- Phase C: contention + cancel-while-queued on ONE client. -----
	println!("\n== phase C: 3 simultaneous prompts, middle one cancelled ==");
	let (first, second, third) = tokio::join!(
		timed_prompt(&agent_a, "C/1", "What is 1 + 1?"),
		async {
			// Cancelled while (most likely) still queued behind C/1.
			let cut = tokio::time::timeout(
				Duration::from_millis(300),
				agent_a.prompt("Write a long story about dragons."),
			)
			.await;
			println!(
				"  [C/2] cancelled while queued: {}",
				if cut.is_err() { "yes" } else { "no (finished)" }
			);
			String::new()
		},
		timed_prompt(&agent_a, "C/3", "What is 3 + 3?"),
	);
	assert!(!first.is_empty() && !third.is_empty());
	drop(second);

	println!("\nall phases completed; no client wedged.");
	Ok(())
}
