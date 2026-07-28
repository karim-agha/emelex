//! Structured output on a local MLX model: rig's `Extractor` turning
//! unstructured text into a typed Rust struct.
//!
//! Under the hood rig registers a single `submit` tool whose JSON Schema
//! is derived from `T`, forces it via `tool_choice = Required`, and
//! deserializes the submitted arguments — so this exercises emelex's
//! tool-calling pipeline end to end. A local model is not
//! grammar-constrained, so conformance is probabilistic; `.retries(n)`
//! covers the occasional miss. Two practical lessons baked in below:
//! the extractor runs with a non-zero temperature (under greedy
//! sampling every retry reproduces the same malformed output, making
//! retries useless), and the fragile boolean field uses a lenient serde
//! deserializer - small local models love Python-style `"True"`, and
//! accepting it beats failing the whole extraction over a quote pair.
//!
//! ```sh
//! cargo run -p emelex --release --example extract -- \
//!   "$EMELEX_TEST_MODEL"
//! ```

// Example code: unwraps and panics are acceptable here.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use rig_core::schemars;

/// The shape we want pulled out of the incident report.
#[derive(Debug, serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
struct Incident {
	/// The service that failed, e.g. "payments-api".
	service: String,
	/// One of: low, medium, high, critical.
	severity: String,
	/// Customer-visible symptoms, one short phrase each.
	symptoms: Vec<String>,
	/// Whether a recent deploy is implicated.
	#[serde(deserialize_with = "lenient_bool")]
	#[schemars(with = "bool")]
	deploy_related: bool,
}

/// Accept `true`/`false` as well as the `"True"`/`"false"` strings that
/// non-grammar-constrained models frequently emit.
fn lenient_bool<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
	D: serde::Deserializer<'de>,
{
	#[derive(serde::Deserialize)]
	#[serde(untagged)]
	enum BoolIsh {
		Bool(bool),
		Text(String),
	}
	use serde::Deserialize as _;
	match BoolIsh::deserialize(deserializer)? {
		BoolIsh::Bool(value) => Ok(value),
		BoolIsh::Text(text) => match text.to_lowercase().as_str() {
			"true" | "yes" => Ok(true),
			"false" | "no" => Ok(false),
			other => Err(serde::de::Error::custom(format!(
				"not a boolean: {other:?}"
			))),
		},
	}
}

const REPORT: &str = "PagerDuty alert 04:12 UTC: checkout success rate dropped from 99.2% to \
	 61%. Customers report card payments hanging and then failing. The \
	 checkout-web frontend shows intermittent 502s from its payment backend \
	 payments-api. A deploy went out to payments-api about 15 minutes before \
	 the alert. Impact is severe and ongoing.";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
	let model_dir = std::env::args()
		.nth(1)
		.expect("usage: extract <mlx-model-dir>");

	let extractor = emelex::Client::from_path(model_dir)?
		.extractor::<Incident>()
		.preamble(
			"Extract the incident facts from the report. Severity must be one of: \
			 low, medium, high, critical.",
		)
		.additional_params(serde_json::json!({ "temperature": 0.3 }))
		.retries(2)
		.build();

	let response = extractor.extract_with_usage(REPORT).await?;
	println!("extracted: {:#?}", response.data);
	println!(
		"[usage] {} in ({} cached), {} out",
		response.usage.input_tokens,
		response.usage.cached_input_tokens,
		response.usage.output_tokens
	);

	Ok(())
}
