//! The target public-API sample: a rig agent with tools on a local MLX
//! model, then a streamed prompt.
//!
//! ```sh
//! cargo run -p emelex --release --example agent -- \
//!   "$EMELEX_TEST_MODEL"
//! ```

// Example code: unwraps and panics are acceptable here.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use futures::StreamExt;
use rig_core::{
	agent::{PromptResponse, Text},
	completion::Prompt,
	prelude::{MultiTurnStreamItem, StreamingPrompt, Tool},
	streaming::StreamedAssistantContent,
	tool::ToolError,
	tool_macro,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct MathArgs {
	a: i64,
	b: i64,
}

#[derive(Debug, thiserror::Error)]
#[error("math tool failed")]
struct MathError;

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
		println!("[tool] add({}, {})", args.a, args.b);
		Ok(args.a + args.b)
	}
}

#[allow(clippy::unnecessary_wraps)]
#[tool_macro(description = "Subtract two numbers and return a - b")]
fn subtract_tool(x: i64, y: i64) -> Result<i64, ToolError> {
	println!("[tool] subtract({x}, {y})");
	Ok(x - y)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
	let model_dir = std::env::args()
		.nth(1)
		.expect("usage: agent <mlx-model-dir>");

	let agent = emelex::Client::from_path(model_dir)?
		.agent()
		.preamble("You're a helpful assistant")
		.tool(Add)
		.tool(SubtractTool)
		.build();

	let result = agent.prompt("Calculate 5 + 2 = ?").max_turns(20).await?;
	println!("answer: {result}");

	let result = agent.prompt("Calculate 5 - 2 = ?").max_turns(20).await?;
	println!("answer: {result}");

	let mut stream = agent.stream_prompt("Write a haiku about Rust").await;

	while let Some(chunk) = stream.next().await {
		match chunk {
			Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(
				Text { text: chunk, .. },
			))) => {
				print!("{chunk}");
			}

			Ok(MultiTurnStreamItem::FinalResponse(PromptResponse { usage, .. })) => {
				println!();
				println!();
				println!("Token {usage:#?}");
			}
			Err(e) => eprintln!("error: {e:?}"),
			_ => {}
		}
	}

	Ok(())
}
