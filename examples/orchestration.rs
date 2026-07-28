//! Multi-agent orchestration on ONE locally loaded MLX model.
//!
//! A realistic on-call incident-response pipeline with three specialized
//! agents, all built from the same [`emelex::Client`] — the multi-GB
//! checkpoint is loaded once, the agents differ only in configuration,
//! and the engine's KV prompt cache is shared between them (watch
//! `cached` climb in the per-stage usage lines):
//!
//! | agent          | thinking            | temp | `max_tokens` | tools        | static context |
//! |----------------|---------------------|------|------------|--------------|----------------|
//! | triage-analyst | ON (320-token budget) | 0.0  | 768        | none         | runbook        |
//! | ops-executor   | OFF                 | 0.0  | 320/turn   | status/logs/restart | none    |
//! | comms-writer   | OFF                 | 0.7  | 180        | none         | style guide    |
//!
//! The analyst reasons carefully about the incident and produces a
//! diagnosis plan (its thinking stays internal, capped by the reasoning
//! budget). The executor mechanically works the plan through mocked ops
//! tools in a multi-turn loop — thinking is off because a tool loop
//! wants low latency, not deliberation. The writer turns the technical
//! outcome into a short customer-facing status update, streamed, with a
//! higher temperature for natural prose.
//!
//! Thinking is configured at two layers, both strongly typed:
//! `emelex::ClientBuilder` sets the *client-wide default* for every
//! agent sharing the loaded model, and the `emelex::ReasoningExt`
//! extension trait overrides it per agent
//! (`.enable_thinking(bool)` / `.reasoning_budget_tokens(n)`). Here the
//! client default is thinking OFF and only the analyst opts in - its
//! reasoning streams to the console as it happens. The remaining
//! per-agent knobs come from rig's `AgentBuilder` (`temperature`,
//! `max_tokens`, `context`, `tool`).
//!
//! ```sh
//! cargo run -p emelex --release --example orchestration -- \
//!   "$EMELEX_TEST_MODEL"
//! ```

// Example code: unwraps and panics are acceptable here.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::{
	io::{self, Write as _},
	sync::atomic::{AtomicBool, Ordering},
};

use emelex::ReasoningExt as _;
use futures::StreamExt;
use rig_core::{
	agent::{PromptResponse, Text},
	completion::Prompt,
	prelude::{MultiTurnStreamItem, StreamingPrompt, Tool},
	streaming::StreamedAssistantContent,
};
use serde::Deserialize;

// ============================================================
// Mocked ops world: one degraded service that a restart fixes.
// ============================================================

static PAYMENTS_RESTARTED: AtomicBool = AtomicBool::new(false);

#[derive(Debug, thiserror::Error)]
#[error("ops tool failed: {0}")]
struct OpsError(String);

#[derive(Deserialize)]
struct ServiceArgs {
	service: String,
}

#[derive(Deserialize)]
struct LogsArgs {
	service: String,
	#[serde(default = "default_lines")]
	lines: usize,
}

const fn default_lines() -> usize {
	20
}

fn known_service(service: &str) -> Result<(), OpsError> {
	match service {
		"payments-api" | "checkout-web" | "ledger-db" => Ok(()),
		other => Err(OpsError(format!(
			"unknown service {other:?}; known services: payments-api, checkout-web, \
			 ledger-db"
		))),
	}
}

struct ServiceStatus;

impl Tool for ServiceStatus {
	type Args = ServiceArgs;
	type Error = OpsError;
	type Output = String;

	const NAME: &'static str = "service_status";

	fn description(&self) -> String {
		"Get the current health of a service (status, error rate, latency)".to_string()
	}

	fn parameters(&self) -> serde_json::Value {
		serde_json::json!({
			"type": "object",
			"properties": {
				"service": {
					"type": "string",
					"description": "Service name, e.g. payments-api"
				}
			},
			"required": ["service"]
		})
	}

	async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
		println!("  [tool] service_status({})", args.service);
		known_service(&args.service)?;
		Ok(match args.service.as_str() {
			"payments-api" if !PAYMENTS_RESTARTED.load(Ordering::SeqCst) => {
				"payments-api: DEGRADED - 5xx rate 34%, p99 latency 8.2s, last deploy \
				 2026-07-19-r3 (14 minutes ago)"
					.to_string()
			}
			"payments-api" => "payments-api: HEALTHY - 5xx rate 0.1%, p99 latency \
			                   180ms, uptime 40s since restart"
				.to_string(),
			other => format!("{other}: HEALTHY - error rate nominal"),
		})
	}
}

struct RecentLogs;

impl Tool for RecentLogs {
	type Args = LogsArgs;
	type Error = OpsError;
	type Output = String;

	const NAME: &'static str = "recent_logs";

	fn description(&self) -> String {
		"Fetch the most recent error-level log lines for a service".to_string()
	}

	fn parameters(&self) -> serde_json::Value {
		serde_json::json!({
			"type": "object",
			"properties": {
				"service": {
					"type": "string",
					"description": "Service name, e.g. payments-api"
				},
				"lines": {
					"type": "integer",
					"description": "How many recent lines to fetch (default 20)"
				}
			},
			"required": ["service"]
		})
	}

	async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
		println!("  [tool] recent_logs({}, {})", args.service, args.lines);
		known_service(&args.service)?;
		Ok(match args.service.as_str() {
			"payments-api" if !PAYMENTS_RESTARTED.load(Ordering::SeqCst) => {
				"ERROR db-pool: connection pool exhausted (100/100 in use, 4832 \
				 waiters)\nERROR handler: timeout acquiring connection after \
				 5000ms\nWARN  db-pool: pool size changed 100 -> 10 by config \
				 2026-07-19-r3\nERROR handler: timeout acquiring connection after \
				 5000ms"
					.to_string()
			}
			_ => "(no recent error-level log lines)".to_string(),
		})
	}
}

struct RestartService;

impl Tool for RestartService {
	type Args = ServiceArgs;
	type Error = OpsError;
	type Output = String;

	const NAME: &'static str = "restart_service";

	fn description(&self) -> String {
		"Rolling-restart a service, reloading its last known-good configuration".to_string()
	}

	fn parameters(&self) -> serde_json::Value {
		serde_json::json!({
			"type": "object",
			"properties": {
				"service": {
					"type": "string",
					"description": "Service name, e.g. payments-api"
				}
			},
			"required": ["service"]
		})
	}

	async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
		println!("  [tool] restart_service({})", args.service);
		known_service(&args.service)?;
		if args.service == "payments-api" {
			PAYMENTS_RESTARTED.store(true, Ordering::SeqCst);
		}
		Ok(format!(
			"{}: rolling restart completed, last known-good config reloaded",
			args.service
		))
	}
}

// ============================================================
// Orchestration
// ============================================================

const INCIDENT: &str = "PagerDuty alert 04:12 UTC: checkout success rate dropped from 99.2% to \
	 61%. Customers report card payments hanging and then failing. The \
	 checkout-web frontend shows intermittent 502s from its payment backend. A \
	 deploy went out to payments-api about 15 minutes before the alert.";

#[tokio::main]
#[allow(clippy::too_many_lines)] // three agent definitions read best inline
async fn main() -> Result<(), Box<dyn std::error::Error>> {
	let model_dir = std::env::args()
		.nth(1)
		.expect("usage: orchestration <mlx-model-dir>");

	// One loaded checkpoint; every agent below shares it (and its KV
	// prompt cache). ClientBuilder's typed knobs set the client-wide
	// defaults - thinking OFF for everyone - and individual agents
	// override them with the `ReasoningExt` methods below.
	let client = emelex::Client::builder(model_dir)
		.enable_thinking(false)
		.build()?;

	// --- Agent 1: triage analyst — thinks before it speaks. -----------
	// Deep-reasoning configuration: `reasoning_budget_tokens` overrides
	// the client's thinking-off default AND caps the reasoning span, so
	// diagnosis quality doesn't come at unbounded latency. Deterministic
	// sampling, a runbook as static context, no tools: its job is
	// analysis. The budget must sit comfortably below max_tokens - both
	// draw from the same generation window, and a plan that is all
	// thinking and no answer helps nobody.
	let analyst = client
		.agent()
		.name("triage-analyst")
		.preamble(
			"You are the on-call triage analyst. Diagnose the incident from the \
			 report and the runbook. Think through the evidence carefully, then \
			 output: (1) most likely root cause in one sentence, (2) a numbered \
			 action plan of at most 3 concrete steps for the operator, naming exact \
			 service names and tools to use.",
		)
		.context(
			"RUNBOOK excerpt - payment stack:\n- checkout-web (frontend) calls \
			 payments-api; payments-api uses ledger-db.\n- 502s from checkout-web \
			 usually mean payments-api is unhealthy, not the frontend itself.\n- \
			 After a bad deploy, `restart_service` reloads the last known-good \
			 configuration.\n- Always confirm service health via `service_status` \
			 before and after any intervention.",
		)
		.temperature(0.0)
		.max_tokens(768)
		.reasoning_budget_tokens(320)
		.build();

	println!("== stage 1: triage-analyst (thinking ON, reasoning streamed) ==");
	let mut stream = analyst.stream_prompt(INCIDENT).await;
	let mut triage: Option<PromptResponse> = None;
	let mut in_thinking = false;
	while let Some(chunk) = stream.next().await {
		match chunk {
			Ok(MultiTurnStreamItem::StreamAssistantItem(
				StreamedAssistantContent::ReasoningDelta { reasoning, .. },
			)) => {
				if !in_thinking {
					println!("--- thinking ---");
					in_thinking = true;
				}
				print!("{reasoning}");
				io::stdout().flush()?;
			}
			Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Reasoning(
				reasoning,
			))) => {
				// Some providers emit one complete reasoning block instead
				// of deltas; handle both.
				println!("--- thinking ---\n{reasoning:?}");
				in_thinking = true;
			}
			Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(
				Text { text, .. },
			))) => {
				if in_thinking {
					println!("\n--- answer ---");
					in_thinking = false;
				}
				print!("{text}");
				io::stdout().flush()?;
			}
			Ok(MultiTurnStreamItem::FinalResponse(response)) => {
				triage = Some(response);
			}
			Err(error) => eprintln!("stream error: {error:?}"),
			_ => {}
		}
	}
	let triage = triage.expect("analyst run should yield a final response");
	println!("\n");
	print_usage("triage-analyst", &triage);

	// --- Agent 2: ops executor — mechanical tool loop. ----------------
	// Thinking is OFF: the plan is already made, the loop should be
	// fast and predictable. Smaller per-turn budget, deterministic
	// sampling, and the ops tools. `max_turns` bounds the loop.
	let executor = client
		.agent()
		.name("ops-executor")
		.preamble(
			"You are the operations executor. Follow the triage plan exactly using \
			 your tools; verify health after any intervention. When done, report \
			 each step you took and its outcome as a short bullet list, ending with \
			 the final service status.",
		)
		// No additional_params: inherits the client-wide thinking-off
		// default.
		.temperature(0.0)
		.max_tokens(320)
		.tool(ServiceStatus)
		.tool(RecentLogs)
		.tool(RestartService)
		.build();

	println!("\n== stage 2: ops-executor (thinking OFF, tools) ==");
	let execution = executor
		.prompt(format!(
			"Incident report:\n{INCIDENT}\n\nTriage plan from the analyst:\n{}",
			triage.output
		))
		.max_turns(12)
		.extended_details()
		.await?;
	println!("{}\n", execution.output);
	print_usage("ops-executor", &execution);

	// --- Agent 3: comms writer — customer-facing prose, streamed. -----
	// No tools, no thinking, a tight token budget, and a higher
	// temperature: status updates should read naturally, not
	// mechanically. Static context carries the comms style guide.
	let writer = client
		.agent()
		.name("comms-writer")
		.preamble(
			"You write public status-page updates. From the incident report and the \
			 operator's resolution notes, write ONE short update (max 3 sentences): \
			 what customers experienced, what was done, current status. Follow the \
			 style guide.",
		)
		.context(
			"STYLE GUIDE: plain language, no internal service names, no blame, no \
			 promises about future incidents. Refer to the affected capability as \
			 'card payments at checkout'.",
		)
		.temperature(0.7)
		.max_tokens(180)
		.build();

	println!("\n== stage 3: comms-writer (thinking OFF, streamed) ==");
	let mut stream = writer
		.stream_prompt(format!(
			"Incident report:\n{INCIDENT}\n\nOperator resolution notes:\n{}",
			execution.output
		))
		.await;
	while let Some(chunk) = stream.next().await {
		match chunk {
			Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(
				Text { text, .. },
			))) => print!("{text}"),
			Ok(MultiTurnStreamItem::FinalResponse(PromptResponse { usage, .. })) => {
				println!("\n");
				println!(
					"[usage] comms-writer: {} in ({} cached), {} out",
					usage.input_tokens, usage.cached_input_tokens, usage.output_tokens
				);
			}
			Err(error) => eprintln!("stream error: {error:?}"),
			_ => {}
		}
	}

	Ok(())
}

fn print_usage(agent: &str, response: &PromptResponse) {
	println!(
		"[usage] {agent}: {} model call(s), {} in ({} cached), {} out",
		response.completion_calls.len(),
		response.usage.input_tokens,
		response.usage.cached_input_tokens,
		response.usage.output_tokens
	);
}
