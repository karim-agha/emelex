//! Live integration tests against a real MLX checkpoint.
//!
//! Every test is skipped unless `EMELEX_TEST_MODEL` points at an MLX model
//! directory, e.g.:
//!
//! ```sh
//! EMELEX_TEST_MODEL=/path/to/installed/model-snapshot \
//!   cargo test -p emelex --release --test live -- --test-threads=1
//! ```

// Test code: unwraps and panics are the assertion mechanism.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

use std::{
	path::PathBuf,
	sync::{
		OnceLock,
		atomic::{AtomicUsize, Ordering},
	},
};

use emelex::Client;
use futures::StreamExt;
use rig_core::{
	completion::{CompletionModel as _, Prompt},
	prelude::{StreamingPrompt, Tool},
};
use serde::Deserialize;

fn model_path() -> Option<PathBuf> {
	let Some(path) = std::env::var_os("EMELEX_TEST_MODEL") else {
		eprintln!("skipped: set EMELEX_TEST_MODEL to an MLX model directory");
		return None;
	};
	Some(PathBuf::from(path))
}

/// One shared client per test process: the checkpoint is multi-GB and
/// loading it once per test would thrash memory.
fn client() -> Option<Client> {
	static CLIENT: OnceLock<Client> = OnceLock::new();
	let path = model_path()?;
	Some(
		CLIENT
			.get_or_init(|| Client::from_path(path).expect("model should load"))
			.clone(),
	)
}

#[derive(Deserialize)]
struct MathArgs {
	a: i64,
	b: i64,
}

#[derive(Debug, thiserror::Error)]
#[error("math tool failed")]
struct MathError;

static ADD_CALLS: AtomicUsize = AtomicUsize::new(0);
static SUBTRACT_CALLS: AtomicUsize = AtomicUsize::new(0);

struct Add;

impl Tool for Add {
	type Args = MathArgs;
	type Error = MathError;
	type Output = i64;

	const NAME: &'static str = "add";

	fn description(&self) -> String {
		"Add two numbers and return a + b".to_string()
	}

	fn parameters(&self) -> serde_json::Value {
		serde_json::json!({
			"type": "object",
			"properties": {
				"a": {"type": "number"},
				"b": {"type": "number"}
			},
			"required": ["a", "b"]
		})
	}

	async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
		ADD_CALLS.fetch_add(1, Ordering::SeqCst);
		Ok(args.a + args.b)
	}
}

struct Subtract;

impl Tool for Subtract {
	type Args = MathArgs;
	type Error = MathError;
	type Output = i64;

	const NAME: &'static str = "subtract";

	fn description(&self) -> String {
		"Subtract two numbers and return a - b".to_string()
	}

	fn parameters(&self) -> serde_json::Value {
		serde_json::json!({
			"type": "object",
			"properties": {
				"a": {"type": "number"},
				"b": {"type": "number"}
			},
			"required": ["a", "b"]
		})
	}

	async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
		SUBTRACT_CALLS.fetch_add(1, Ordering::SeqCst);
		Ok(args.a - args.b)
	}
}

#[tokio::test]
async fn plain_prompt_generates_text() {
	let Some(client) = client() else { return };
	let agent = client
		.agent()
		.preamble("You are a terse assistant.")
		.build();
	let answer = agent
		.prompt("Reply with the single word: pong")
		.await
		.expect("prompt should succeed");
	println!("plain answer: {answer}");
	assert!(!answer.trim().is_empty());
}

#[tokio::test]
async fn multi_turn_tool_calling_invokes_tools() {
	let Some(client) = client() else { return };
	let agent = client
		.agent()
		.preamble(
			"You are a calculator assistant. Use the provided tools for all \
			 arithmetic; never compute yourself.",
		)
		.tool(Add)
		.tool(Subtract)
		.build();
	let answer = agent
		.prompt("Calculate 5 - 2 = ?")
		.max_turns(20)
		.await
		.expect("tool-calling prompt should succeed");
	println!("tool answer: {answer}");
	assert!(
		SUBTRACT_CALLS.load(Ordering::SeqCst) + ADD_CALLS.load(Ordering::SeqCst) > 0,
		"the model should have called at least one tool"
	);
	assert!(answer.contains('3'), "expected 3 in: {answer}");
}

#[tokio::test]
async fn streaming_yields_incremental_chunks_and_final_usage() {
	let Some(client) = client() else { return };
	let model = client.model();
	let request = model
		.completion_request("Write a haiku about Rust.")
		.preamble("You are a poet.".to_string())
		.build();
	let mut stream = model.stream(request).await.expect("stream should start");
	let mut chunks = 0usize;
	while let Some(item) = stream.next().await {
		item.expect("stream item should be ok");
		chunks += 1;
	}
	println!("streamed {chunks} chunks");
	assert!(chunks > 1, "expected incremental chunks, got {chunks}");
	let usage = stream.usage();
	assert!(usage.output_tokens > 0, "final usage should be reported");
	let text = stream
		.choice
		.iter()
		.filter_map(|content| match content {
			rig_core::completion::AssistantContent::Text(text) => Some(text.text.clone()),
			_ => None,
		})
		.collect::<String>();
	assert!(
		!text.trim().is_empty(),
		"aggregated text should be non-empty"
	);
}

#[tokio::test]
async fn agent_stream_prompt_matches_target_api() {
	let Some(client) = client() else { return };
	let agent = client.agent().preamble("You are a poet.").build();
	let mut stream = agent.stream_prompt("Write a haiku about Rust").await;
	let mut items = 0usize;
	while let Some(item) = stream.next().await {
		item.expect("stream item should be ok");
		items += 1;
	}
	assert!(items > 1, "expected incremental stream items, got {items}");
}

#[tokio::test]
async fn thinking_mode_produces_reasoning() {
	let Some(path) = model_path() else { return };
	// Dedicated client: thinking is a client-level default here.
	let client = Client::builder(path)
		.enable_thinking(true)
		.reasoning_budget_tokens(512)
		.build()
		.expect("model should load");
	let model = client.model();
	let request = model
		.completion_request("Which is larger: 17 * 19 or 320? Answer briefly.")
		.build();
	let response = model
		.completion(request)
		.await
		.expect("completion should succeed");
	println!("thinking raw: {:?}", response.raw_response);
	// Reasoning content is model-dependent; assert the call succeeds and
	// produces a non-empty choice either way.
	assert!(!response.choice.is_empty());
}

#[tokio::test]
async fn prompt_cache_reuses_prefix_on_multi_turn_growth() {
	let Some(client) = client() else { return };
	let model = client.model();
	// The engine's cache requires an *exact* token-prefix match, which is
	// the multi-turn growth shape: turn 2's rendered prompt starts with
	// turn 1's rendered prompt plus its reply. Thinking is pinned off on
	// both calls so both turns render identically.
	let no_thinking = serde_json::json!({"enable_thinking": false});
	let preamble = "You are a terse assistant.".to_string();
	let turn1_prompt = "My favorite color is blue.";
	let first = model
		.completion(
			model
				.completion_request(turn1_prompt)
				.preamble(preamble.clone())
				.additional_params(no_thinking.clone())
				.build(),
		)
		.await
		.expect("first completion should succeed");
	let turn1_reply = first
		.choice
		.iter()
		.filter_map(|content| match content {
			rig_core::completion::AssistantContent::Text(text) => Some(text.text.clone()),
			_ => None,
		})
		.collect::<String>();
	let second = model
		.completion(
			model
				.completion_request("What is my favorite color?")
				.preamble(preamble)
				.messages(vec![
					rig_core::completion::Message::user(turn1_prompt),
					rig_core::completion::Message::assistant(&turn1_reply),
				])
				.additional_params(no_thinking)
				.build(),
		)
		.await
		.expect("second completion should succeed");
	println!(
		"first cached: {}, second cached: {} (of {} input)",
		first.usage.cached_input_tokens,
		second.usage.cached_input_tokens,
		second.usage.input_tokens
	);
	assert!(
		second.usage.cached_input_tokens > 0,
		"turn 2 should reuse turn 1's cached prefix (got 0 of {} input tokens)",
		second.usage.input_tokens
	);
}

// ---------------------------------------------------------------------------
// Characterization tests for MTP decode-loop and cache behavior.
//
// These pin the decode-loop behaviors the TokenEmitter refactor
// emitted/committed ledger split, forced-close reorder) must preserve. They
// are dev-machine live tests: CI skips them without EMELEX_TEST_MODEL.
//
// Honest-scoping note: the historical forced-close
// cache desync (feed-then-callbacks at generate.rs:787 vs the pool insert)
// is NOT expressible through this public surface on text-only prompts —
// think-family templates take the boundary-only insertion branch, which
// snapshots mid-prefill and never pools generation-dependent ids. The
// deterministic regression demonstration therefore lives at the engine level
// with a dedicated engine seam; what follows characterizes the publicly observable
// contract around the same machinery.
// ---------------------------------------------------------------------------

/// Forced close via a tiny reasoning budget completes the reply and leaves
/// the prompt cache serving the next turn.
#[tokio::test]
async fn reasoning_budget_forced_close_completes_and_next_turn_caches() {
	let Some(path) = model_path() else { return };
	let client = Client::builder(path)
		.enable_thinking(true)
		.reasoning_budget_tokens(24)
		.build()
		.expect("model should load");
	let model = client.model();
	let preamble = "You are a terse assistant.".to_string();
	let first = model
		.completion(
			model
				.completion_request("Which is larger: 17 * 19 or 320?")
				.preamble(preamble.clone())
				.build(),
		)
		.await
		.expect("forced-close completion should succeed");
	let text = first
		.choice
		.iter()
		.filter_map(|content| match content {
			rig_core::completion::AssistantContent::Text(text) => Some(text.text.clone()),
			_ => None,
		})
		.collect::<String>();
	assert!(
		!text.trim().is_empty(),
		"forced-close reply should still contain answer text"
	);
	// A follow-up turn on the same conversation must be servable and sane.
	let second = model
		.completion(
			model
				.completion_request("Now add 5 to the larger one.")
				.preamble(preamble)
				.build(),
		)
		.await
		.expect("follow-up after forced close should succeed");
	assert!(second.usage.input_tokens >= second.usage.cached_input_tokens);
}

/// Dropping a stream mid-generation (with a reasoning budget armed so the
/// drop can land anywhere around a forced close) must leave the session
/// able to serve a later cached turn that matches an uncached run exactly
/// under greedy sampling.
#[tokio::test]
async fn cancelled_stream_keeps_next_turn_cache_consistent() {
	let Some(path) = model_path() else { return };
	let client = Client::builder(path)
		.enable_thinking(true)
		.reasoning_budget_tokens(16)
		.build()
		.expect("model should load");
	let model = client.model();
	let preamble = "You are a terse assistant.".to_string();
	for drop_after in [2usize, 5, 9, 14] {
		let request = model
			.completion_request("Explain why the sky is blue, step by step.")
			.preamble(preamble.clone())
			.build();
		let mut stream = model.stream(request).await.expect("stream starts");
		let mut seen = 0usize;
		while let Some(item) = stream.next().await {
			item.expect("stream item ok");
			seen += 1;
			if seen >= drop_after {
				break;
			}
		}
		drop(stream);

		// The follow-up turn must be identical with and without the pool.
		let followup = |cache: bool| {
			let params = serde_json::json!({
				"enable_thinking": false,
				"prompt_cache": cache,
			});
			let request = model
				.completion_request("Reply with the single word: sky")
				.preamble(preamble.clone())
				.additional_params(params)
				.build();
			async { model.completion(request).await }
		};
		let cached = followup(true).await.expect("cached follow-up succeeds");
		let uncached = followup(false).await.expect("uncached follow-up succeeds");
		let text = |response: &rig_core::completion::CompletionResponse<emelex::Response>| {
			response
				.choice
				.iter()
				.filter_map(|content| match content {
					rig_core::completion::AssistantContent::Text(text) => Some(text.text.clone()),
					_ => None,
				})
				.collect::<String>()
		};
		assert_eq!(
			text(&cached),
			text(&uncached),
			"cached and uncached follow-ups diverged after dropping the stream at \
			 {drop_after} chunks - pool ids no longer match KV"
		);
	}
}

// ---------------------------------------------------------------------------
// Speculation-enabled live suite. Direct completion tests assert
// `drafted > 0` on the response they just received, so concurrent calls
// cannot overwrite or misattribute MTP accounting. All tests skip without
// EMELEX_TEST_MODEL and only exercise speculation when the fixture carries
// an MTP module (`Client::supports_mtp`).
// ---------------------------------------------------------------------------

fn spec_client(k: usize) -> Option<Client> {
	let path = model_path()?;
	let client = Client::builder(path)
		.speculative_tokens(k)
		.build()
		.expect("model should load");
	if !client.supports_mtp() {
		eprintln!("skipped: fixture has no MTP module (supports_mtp = false)");
		return None;
	}
	Some(client)
}

fn response_text(response: &rig_core::completion::CompletionResponse<emelex::Response>) -> String {
	response
		.choice
		.iter()
		.filter_map(|content| match content {
			rig_core::completion::AssistantContent::Text(text) => Some(text.text.clone()),
			_ => None,
		})
		.collect()
}

const fn response_speculation(
	response: &rig_core::completion::CompletionResponse<emelex::Response>,
) -> Option<&emelex::SpeculationStatsData> {
	response.raw_response.speculation.as_ref()
}

/// Greedy diagnostic: spec-on output must equal spec-off output token
/// for token (batched-vs-scalar near-tie flips are the documented
/// escape hatch - a reproducible divergence is a bug), and speculation
/// must actually draft.
#[tokio::test]
async fn mtp_greedy_spec_on_matches_spec_off_and_drafts() {
	let Some(client) = spec_client(4) else { return };
	let model = client.model();
	let run = |k: usize| {
		let params = serde_json::json!({
			"enable_thinking": false,
			"prompt_cache": false,
			"speculative_tokens": k,
			"max_tokens": 96,
		});
		let request = model
			.completion_request("List the first eight prime numbers.")
			.preamble("You are a terse assistant.".to_string())
			.additional_params(params)
			.build();
		async { model.completion(request).await.expect("completion") }
	};
	let spec_on = run(4).await;
	let stats = response_speculation(&spec_on).expect("spec-on call is a completed generation");
	assert!(stats.drafted > 0, "speculation never drafted");
	assert!(stats.rounds > 0);
	let spec_off = run(0).await;
	assert!(
		response_speculation(&spec_off).is_none(),
		"spec-off call must carry no speculative accounting"
	);
	assert_eq!(
		response_text(&spec_on),
		response_text(&spec_off),
		"greedy spec-on diverged from spec-off (see fp-tie escape hatch)"
	);
}

/// Cold vs prompt-cache-hit parity with speculation on: the second turn
/// must reuse the pool (boundary lineage with aligned `MtpState`), still
/// draft, and produce the same continuation as a cache-disabled run.
#[tokio::test]
async fn mtp_cache_hit_parity_keeps_drafting() {
	let Some(client) = spec_client(4) else { return };
	let model = client.model();
	let preamble = "You are a terse assistant.".to_string();
	let params = serde_json::json!({
		"enable_thinking": false,
		"speculative_tokens": 4,
		"max_tokens": 64,
	});
	// Turn 2 must be a genuine multi-turn extension of turn 1 (the
	// exact-prefix pool only serves prefixes), so thread turn 1's reply
	// back as message history.
	let turn1_prompt = "My favorite number is 42. Say ok.";
	let first = model
		.completion(
			model
				.completion_request(turn1_prompt)
				.preamble(preamble.clone())
				.additional_params(params.clone())
				.build(),
		)
		.await
		.expect("turn 1");
	let turn1_reply = response_text(&first);
	let turn2 = |cache: bool| {
		let mut p = params.clone();
		p["prompt_cache"] = cache.into();
		let request = model
			.completion_request("What is my favorite number?")
			.preamble(preamble.clone())
			.messages(vec![
				rig_core::completion::Message::user(turn1_prompt),
				rig_core::completion::Message::assistant(&turn1_reply),
			])
			.additional_params(p)
			.build();
		async { model.completion(request).await.expect("turn 2") }
	};
	let cached = turn2(true).await;
	// The cached turn must actually hit the pool — without this the test
	// passes vacuously on a cold miss.
	assert!(
		cached.usage.cached_input_tokens > 0,
		"turn 2 must reuse turn 1's cached prefix (got 0 of {} input tokens)",
		cached.usage.input_tokens
	);
	let stats = response_speculation(&cached).expect("cached turn is a completed generation");
	assert!(stats.drafted > 0, "cache-hit turn stopped drafting");
	let uncached = turn2(false).await;
	assert_eq!(
		response_text(&cached),
		response_text(&uncached),
		"cache-hit continuation diverged from cold (fp-tie escape hatch applies \
		 to single-token wobbles only)"
	);
}

/// Mode switches over one lineage: spec-off serves any entry; a
/// spec-on call over a spec-off lineage is a cold rebuild (invariant-
/// shaped assertions only - exact `cached_tokens` is interleaving-
/// dependent).
#[tokio::test]
async fn mtp_mode_switches_keep_pool_invariants() {
	let Some(client) = spec_client(4) else { return };
	let model = client.model();
	let preamble = "You are a terse assistant.".to_string();
	let ask = |k: usize, text: &'static str| {
		let params = serde_json::json!({
			"enable_thinking": false,
			"speculative_tokens": k,
			"max_tokens": 48,
		});
		let request = model
			.completion_request(text)
			.preamble(preamble.clone())
			.additional_params(params)
			.build();
		async { model.completion(request).await.expect("completion") }
	};
	// off -> on -> off over the same growing conversation shape.
	let off1 = ask(0, "Remember the word banana. Say ok.").await;
	assert!(off1.usage.input_tokens >= off1.usage.cached_input_tokens);
	let on = ask(4, "What word did I ask you to remember?").await;
	assert!(on.usage.input_tokens >= on.usage.cached_input_tokens);
	let on_stats = response_speculation(&on).expect("spec-on call is a completed generation");
	assert!(on_stats.drafted > 0, "spec-on call never drafted");
	let off2 = ask(0, "Say the word once more.").await;
	assert!(off2.usage.input_tokens >= off2.usage.cached_input_tokens);
	assert!(
		response_speculation(&off2).is_none(),
		"spec-off call must carry no speculative accounting"
	);
}

/// Streaming spec-on: drain the stream, then the FIFO-ordered stats
/// query observes the final streaming completion's snapshot with real
/// drafting.
#[tokio::test]
async fn mtp_streaming_spec_on_drafts() {
	let Some(client) = spec_client(4) else { return };
	let model = client.model();
	let request = model
		.completion_request("Write two short sentences about oceans.")
		.preamble("You are a terse assistant.".to_string())
		.additional_params(serde_json::json!({
			"enable_thinking": false,
			"prompt_cache": false,
			"speculative_tokens": 4,
			"max_tokens": 64,
		}))
		.build();
	let mut stream = model.stream(request).await.expect("stream starts");
	let mut chunks = 0usize;
	while let Some(item) = stream.next().await {
		item.expect("stream item ok");
		chunks += 1;
	}
	assert!(chunks > 1, "expected incremental chunks, got {chunks}");
	let stats = stream
		.response
		.as_ref()
		.and_then(|response| response.speculation.as_ref())
		.expect("a drained stream is a completed generation");
	assert!(stats.drafted > 0, "streaming spec-on run never drafted");
}

/// Tool calls with speculation on: an agent with a tool on a
/// `speculative_tokens` client completes the tool-calling loop.
#[tokio::test]
async fn mtp_tool_calls_spec_on_completes() {
	let Some(client) = spec_client(4) else { return };
	let agent = client
		.agent()
		.preamble(
			"You are a calculator assistant. Use the provided tools for all \
			 arithmetic; never compute yourself.",
		)
		.tool(Add)
		.build();
	let answer = agent
		.prompt("Calculate 19 + 23 = ?")
		.max_turns(20)
		.await
		.expect("spec-on tool-calling prompt should succeed");
	assert!(
		!answer.trim().is_empty(),
		"tool loop must produce an answer"
	);
}

/// Thinking + reasoning budget with speculation on: the forced close
/// lands mid-thinking while speculating, the reply still carries answer
/// text, and the call drafts.
#[tokio::test]
async fn mtp_thinking_budget_spec_on_forced_close_replies() {
	let Some(path) = model_path() else { return };
	let client = Client::builder(path)
		.enable_thinking(true)
		.reasoning_budget_tokens(24)
		.speculative_tokens(4)
		.build()
		.expect("model should load");
	if !client.supports_mtp() {
		eprintln!("skipped: fixture has no MTP module (supports_mtp = false)");
		return;
	}
	let model = client.model();
	let response = model
		.completion(
			model
				.completion_request("Which is larger: 17 * 19 or 320?")
				.preamble("You are a terse assistant.".to_string())
				.build(),
		)
		.await
		.expect("spec-on forced-close completion should succeed");
	let text = response_text(&response);
	assert!(
		!text.trim().is_empty(),
		"forced-close spec-on reply should still contain answer text"
	);
	let stats =
		response_speculation(&response).expect("forced-close call is a completed generation");
	assert!(stats.drafted > 0, "spec-on forced-close call never drafted");
}

// Scope note: neither non-pristine caller caches nor media-driven
// speculation disabling is expressible through this public
// surface. Caller-supplied caches never cross `Client` (the engine is
// `pub(crate)`; the rig provider only exposes the pooled
// `generate_cached` path), so the non-pristine-caches → `Disabled` row
// is pinned by the engine-level test
// `non_pristine_caller_caches_disable_speculation`
// (src/engine/generate.rs). Media → spec-off is not expressible against
// the pinned dense fixture at all: the Qwen3.5-4B text-only class has no
// vision/audio tower, so a media request fails at encode rather than
// exercising the media-Disabled branch; that branch is covered by the
// engine's spec-state resolution (`media.is_empty()` gating) and its
// unit tests.

/// Sampled seeded comparison: two spec-on runs with the same seed emit
/// identical text; a different-seed spec-on run and a spec-off run are
/// both non-empty. NO distributional assertion live (kept-set flips at
/// filter thresholds are the documented tolerance class).
#[tokio::test]
async fn mtp_sampled_seeded_runs_reproduce() {
	let Some(client) = spec_client(4) else { return };
	let model = client.model();
	let run = |k: usize, seed: u64| {
		let params = serde_json::json!({
			"enable_thinking": false,
			"prompt_cache": false,
			"speculative_tokens": k,
			"max_tokens": 48,
			"temperature": 0.7,
			"top_p": 0.9,
			"seed": seed,
		});
		let request = model
			.completion_request("Describe a mountain sunrise in one sentence.")
			.preamble("You are a terse assistant.".to_string())
			.additional_params(params)
			.build();
		async { model.completion(request).await.expect("sampled completion") }
	};
	let a = run(4, 7).await;
	let stats = response_speculation(&a).expect("sampled spec-on call is a completed generation");
	assert!(stats.drafted > 0, "sampled spec-on run never drafted");
	let b = run(4, 7).await;
	assert_eq!(
		response_text(&a),
		response_text(&b),
		"same seed, same spec-on sampled text"
	);
	let c = run(4, 8).await;
	let d = run(0, 9).await;
	assert!(!response_text(&c).trim().is_empty());
	assert!(!response_text(&d).trim().is_empty());
}

/// Mode re-enable on one lineage: spec-on builds the `MtpState` lineage, a
/// spec-off extension overwrites its `mtp` with `None`, the next spec-on
/// call is a COLD rebuild (`cached_tokens == 0`), and a further spec-on
/// call hits the rebuilt lineage warm.
#[tokio::test]
async fn mtp_reenable_after_off_extension_cold_rebuilds_then_warms() {
	let Some(client) = spec_client(4) else { return };
	let model = client.model();
	let preamble = "You are a terse assistant.".to_string();
	let mut history: Vec<rig_core::completion::Message> = Vec::new();
	macro_rules! turn {
		($k:expr, $prompt:expr) => {{
			let prompt: &'static str = $prompt;
			let params = serde_json::json!({
				"enable_thinking": false,
				"speculative_tokens": $k,
				"max_tokens": 32,
			});
			let request = model
				.completion_request(prompt)
				.preamble(preamble.clone())
				.messages(history.clone())
				.additional_params(params)
				.build();
			let response = model.completion(request).await.expect("turn");
			let reply = response_text(&response);
			history.push(rig_core::completion::Message::user(prompt));
			history.push(rig_core::completion::Message::assistant(&reply));
			response
		}};
	}
	// Turn 1 (spec-on) builds the MtpState-bearing lineage.
	let t1 = turn!(4, "Remember the word banana. Say ok.");
	assert!(t1.usage.input_tokens >= t1.usage.cached_input_tokens);
	// Turn 2 (spec-off) extends the lineage (any entry is compatible for
	// a non-speculating call) and overwrites its mtp with None.
	let t2 = turn!(0, "What word did I ask you to remember?");
	assert!(
		t2.usage.cached_input_tokens > 0,
		"spec-off extension must hit the lineage"
	);
	// Turn 3 (spec-on) over the now mtp-less lineage: cold rebuild.
	let t3 = turn!(4, "Say the word again.");
	assert_eq!(
		t3.usage.cached_input_tokens, 0,
		"spec-on over an mtp-less lineage must cold rebuild"
	);
	let t3_stats = response_speculation(&t3).expect("cold rebuild is a completed generation");
	assert!(t3_stats.drafted > 0, "the cold rebuild still speculates");
	// Turn 4 (spec-on) hits the rebuilt MtpState-bearing lineage warm.
	let t4 = turn!(4, "And once more, please.");
	assert!(
		t4.usage.cached_input_tokens > 0,
		"the further spec-on call must hit the rebuilt lineage warm"
	);
	let t4_stats = response_speculation(&t4).expect("warm turn is a completed generation");
	assert!(t4_stats.drafted > 0, "the warm hit still speculates");
}

/// Forced-close cancellation at several stream drop points with
/// speculation ON - mirrors the spec-off characterization test
/// `cancelled_stream_keeps_next_turn_cache_consistent` with a
/// speculating client, asserting the same cached-vs-uncached follow-up
/// equality.
#[tokio::test]
async fn mtp_cancelled_stream_spec_on_keeps_next_turn_cache_consistent() {
	let Some(path) = model_path() else { return };
	let client = Client::builder(path)
		.enable_thinking(true)
		.reasoning_budget_tokens(16)
		.speculative_tokens(4)
		.build()
		.expect("model should load");
	if !client.supports_mtp() {
		eprintln!("skipped: fixture has no MTP module (supports_mtp = false)");
		return;
	}
	let model = client.model();
	let preamble = "You are a terse assistant.".to_string();
	for drop_after in [2usize, 5, 9, 14] {
		let request = model
			.completion_request("Explain why the sky is blue, step by step.")
			.preamble(preamble.clone())
			.build();
		let mut stream = model.stream(request).await.expect("stream starts");
		let mut seen = 0usize;
		while let Some(item) = stream.next().await {
			item.expect("stream item ok");
			seen += 1;
			if seen >= drop_after {
				break;
			}
		}
		drop(stream);

		// The follow-up turn must be identical with and without the pool
		// (greedy; spec-on on both sides).
		let followup = |cache: bool| {
			let params = serde_json::json!({
				"enable_thinking": false,
				"prompt_cache": cache,
			});
			let request = model
				.completion_request("Reply with the single word: sky")
				.preamble(preamble.clone())
				.additional_params(params)
				.build();
			async { model.completion(request).await }
		};
		let cached = followup(true).await.expect("cached follow-up succeeds");
		let uncached = followup(false).await.expect("uncached follow-up succeeds");
		assert_eq!(
			response_text(&cached),
			response_text(&uncached),
			"spec-on cached and uncached follow-ups diverged after dropping the \
			 stream at {drop_after} chunks - pool ids no longer match KV"
		);
	}
}

/// Interleaved sequential spec-on/spec-off calls over one growing
/// conversation on one Session: invariant-shaped pool sanity only
/// (`cached_tokens <= prompt_tokens`, spec-off records no rounds, no
/// panic) - exact cached counts are interleaving-dependent by design
/// (the documented mixed-traffic ping-pong).
#[tokio::test]
async fn mtp_interleaved_spec_modes_keep_pool_invariant_shaped() {
	let Some(client) = spec_client(4) else { return };
	let model = client.model();
	let preamble = "You are a terse assistant.".to_string();
	let mut history: Vec<rig_core::completion::Message> = Vec::new();
	let prompts = [
		"Say the letter A.",
		"Now say B.",
		"Now say C.",
		"Now say D.",
		"Now say E.",
	];
	for (i, prompt) in prompts.iter().enumerate() {
		let k = if i % 2 == 0 { 4 } else { 0 };
		let params = serde_json::json!({
			"enable_thinking": false,
			"speculative_tokens": k,
			"max_tokens": 24,
		});
		let request = model
			.completion_request(*prompt)
			.preamble(preamble.clone())
			.messages(history.clone())
			.additional_params(params)
			.build();
		let response = model
			.completion(request)
			.await
			.expect("interleaved completion succeeds");
		assert!(
			response.usage.cached_input_tokens <= response.usage.input_tokens,
			"cached_tokens must never exceed prompt_tokens (turn {i})"
		);
		if k == 0 {
			assert!(
				response_speculation(&response).is_none(),
				"spec-off turn {i} must carry no speculative accounting"
			);
		} else {
			let stats = response_speculation(&response)
				.expect("spec-on interleaved call must carry accounting");
			assert!(stats.drafted > 0, "spec-on turn {i} must draft");
		}
		let reply = response_text(&response);
		history.push(rig_core::completion::Message::user(*prompt));
		history.push(rig_core::completion::Message::assistant(&reply));
	}
}
