//! Several tool calls in one assistant turn on a local MLX model.
//!
//! rig executes every tool call the model emits in a single turn -
//! sequentially by default, concurrently with `.tool_concurrency(n)` -
//! persists all results in call order, and then re-prompts the model.
//! The prompt below invites the model to batch three independent
//! calculations into one turn; the per-run summary shows how many model
//! calls (turns) the whole exchange actually took.
//!
//! ```sh
//! cargo run -p emelex --release --example multi_tool -- \
//!   "$EMELEX_TEST_MODEL"
//! ```

// Example code: unwraps and panics are acceptable here. `tool_macro`
// derives a tool's Output and Error from a `Result` return type, so the
// infallible tools below must still be written as fallible.
#![allow(
	clippy::unwrap_used,
	clippy::expect_used,
	clippy::panic,
	clippy::unnecessary_wraps
)]

use std::sync::atomic::{AtomicUsize, Ordering};

use rig_core::{completion::Prompt, tool::ToolError, tool_macro};

static TOOL_CALLS: AtomicUsize = AtomicUsize::new(0);

fn log_call(name: &str, a: i64, b: i64) {
	let n = TOOL_CALLS.fetch_add(1, Ordering::SeqCst) + 1;
	println!("  [tool #{n}] {name}({a}, {b})");
}

// rig's `tool_macro` turns a plain function into a `Tool`: the struct is
// the PascalCase function name (`add` -> `Add`), the tool the model sees
// is named after the function, and its JSON Schema is derived from the
// parameter types.
#[tool_macro(description = "Add two numbers and return a + b")]
fn add(a: i64, b: i64) -> Result<i64, ToolError> {
	log_call("add", a, b);
	Ok(a + b)
}

#[tool_macro(description = "Subtract two numbers and return a - b")]
fn subtract(a: i64, b: i64) -> Result<i64, ToolError> {
	log_call("subtract", a, b);
	Ok(a - b)
}

#[tool_macro(description = "Multiply two numbers and return a * b")]
fn multiply(a: i64, b: i64) -> Result<i64, ToolError> {
	log_call("multiply", a, b);
	Ok(a * b)
}

const PROMPT: &str = "Compute all three of these independently: 3 + 4, 10 - \
                      2, and 6 * 7. Use one tool call per calculation - you \
                      may issue all three calls at once - then report the \
                      three results in one line.";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
	let model_dir = std::env::args()
		.nth(1)
		.expect("usage: multi_tool <mlx-model-dir>");

	let client = emelex::Client::from_path(model_dir)?;

	for (label, concurrency) in [("sequential tools", 1), ("parallel tools", 3)] {
		TOOL_CALLS.store(0, Ordering::SeqCst);
		let agent = client
			.agent()
			.preamble(
				"You are a calculator. Use the provided tools for every arithmetic \
				 operation; never compute yourself.",
			)
			.tool(Add)
			.tool(Subtract)
			.tool(Multiply)
			.build();

		println!("== {label} (tool_concurrency = {concurrency}) ==");
		let response = agent
			.prompt(PROMPT)
			.max_turns(8)
			.tool_concurrency(concurrency)
			.extended_details()
			.await?;
		println!("answer: {}", response.output);
		println!(
			"[summary] {} tool call(s) across {} model turn(s), {} in ({} cached), \
			 {} out\n",
			TOOL_CALLS.load(Ordering::SeqCst),
			response.completion_calls.len(),
			response.usage.input_tokens,
			response.usage.cached_input_tokens,
			response.usage.output_tokens
		);
	}

	Ok(())
}
