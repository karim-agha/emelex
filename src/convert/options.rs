//! Generation-knob resolution: request params, `additional_params`
//! overlay, tool-choice policy, and best-effort structured output.

use rig_core::{completion::CompletionRequest, message::ToolChoice};
use serde::Deserialize;

use crate::{
	client::Defaults,
	engine::{generate::GenerateOptions, tools::Tool},
	error::Error,
};

/// Provider-specific knobs accepted via `additional_params`. Unknown keys
/// are ignored; known keys with the wrong type fail loudly.
#[derive(Debug, Default, Deserialize)]
struct ExtraParams {
	temperature: Option<f32>,
	max_tokens: Option<usize>,
	top_p: Option<f32>,
	top_k: Option<i32>,
	seed: Option<u64>,
	enable_thinking: Option<bool>,
	reasoning_budget_tokens: Option<usize>,
	prompt_cache: Option<bool>,
	speculative_tokens: Option<usize>,
}

/// Ceiling on a single call's generation budget: large enough for any
/// real reply, small enough that a nonsense request value (`u64::MAX`)
/// cannot blow up allocation or run unbounded.
const MAX_TOKENS_CEILING: usize = 1 << 20;

/// Resolve engine options. Precedence: request `additional_params` >
/// request fields > client-builder defaults.
pub(super) fn generate_options(
	request: &CompletionRequest,
	defaults: &Defaults,
) -> Result<GenerateOptions, Error> {
	let extra = match &request.additional_params {
		Some(value) => {
			if !crate::json::structurally_bounded(value) {
				return Err(Error::InvalidRequest(
					"additional_params exceeds JSON structural limits".to_string(),
				));
			}
			ExtraParams::deserialize(value)?
		}
		None => ExtraParams::default(),
	};
	let mut sampling = defaults.sampling;
	if let Some(temperature) = request.temperature {
		sampling.temperature = temperature as f32;
	}
	if let Some(temperature) = extra.temperature {
		sampling.temperature = temperature;
	}
	if let Some(top_p) = extra.top_p {
		sampling.top_p = top_p;
	}
	if let Some(top_k) = extra.top_k {
		// Non-positive top_k means "no cutoff", not an integer wrap.
		sampling.top_k = (top_k > 0).then_some(top_k);
	}
	if let Some(seed) = extra.seed {
		sampling.seed = Some(seed);
	}
	let requested_max_tokens = extra
		.max_tokens
		.or_else(|| request.max_tokens.and_then(|max| usize::try_from(max).ok()))
		.unwrap_or(defaults.max_tokens);
	let requested_speculative_tokens = extra.speculative_tokens.or(defaults.speculative_tokens);
	let enable_thinking = extra.enable_thinking.or(defaults.enable_thinking);
	let reasoning_budget_tokens = match (extra.reasoning_budget_tokens, extra.enable_thinking) {
		(Some(budget), _) => Some(budget),
		(None, Some(false)) => None,
		(None, _) => defaults.reasoning_budget_tokens,
	};
	let resolved = GenerateOptions {
		max_tokens: requested_max_tokens,
		context_tokens: defaults.context_tokens,
		sampling,
		enable_thinking,
		reasoning_budget_tokens,
		prompt_cache: extra.prompt_cache.or(defaults.prompt_cache),
		// emelex patch (not upstream): `Some(0)` normalizes to off.
		speculative_tokens: requested_speculative_tokens.filter(|&tokens| tokens > 0),
	};
	if resolved.max_tokens == 0 {
		return Err(Error::InvalidRequest(
			"max_tokens must be positive".to_string(),
		));
	}
	if resolved.max_tokens > MAX_TOKENS_CEILING {
		return Err(Error::InvalidRequest(format!(
			"max_tokens must be at most {MAX_TOKENS_CEILING}"
		)));
	}
	if !resolved.sampling.temperature.is_finite()
		|| !(0.0..=2.0).contains(&resolved.sampling.temperature)
	{
		return Err(Error::InvalidRequest(
			"temperature must be finite and in 0..=2".to_string(),
		));
	}
	if !resolved.sampling.top_p.is_finite() || !(0.0..=1.0).contains(&resolved.sampling.top_p) {
		return Err(Error::InvalidRequest(
			"top_p must be finite and in 0..=1".to_string(),
		));
	}
	if requested_speculative_tokens
		.is_some_and(|tokens| tokens > crate::engine::generate::SPECULATIVE_TOKENS_CEILING)
	{
		return Err(Error::InvalidRequest(format!(
			"speculative_tokens must be at most {}",
			crate::engine::generate::SPECULATIVE_TOKENS_CEILING
		)));
	}
	if resolved
		.reasoning_budget_tokens
		.is_some_and(|budget| budget == 0 || budget > resolved.max_tokens)
	{
		return Err(Error::InvalidRequest(
			"reasoning_budget_tokens must be positive and not exceed max_tokens".to_string(),
		));
	}
	if resolved.reasoning_budget_tokens.is_some() && resolved.enable_thinking != Some(true) {
		return Err(Error::InvalidRequest(
			"reasoning_budget_tokens requires thinking to be enabled".to_string(),
		));
	}
	Ok(resolved)
}

/// Apply the tool-choice policy. The engine has no forced-tool decoding
/// mode, so `Required`/`Specific` become a system instruction
/// (best-effort, mirroring the `output_schema` policy); `None` drops the
/// tools entirely.
pub(super) fn tools_and_instruction(
	request: &CompletionRequest,
) -> Result<(Option<Vec<Tool>>, Option<String>), Error> {
	let tools: Vec<Tool> = request
		.tools
		.iter()
		.map(|tool| {
			Tool::new(
				tool.name.clone(),
				tool.description.clone(),
				tool.parameters.clone(),
			)
		})
		.collect();
	if tools.is_empty() {
		return match &request.tool_choice {
			Some(ToolChoice::Required) => Err(Error::InvalidRequest(
				"tool_choice=Required requires at least one available tool".to_string(),
			)),
			Some(ToolChoice::Specific { function_names }) if function_names.is_empty() => {
				Err(Error::InvalidRequest(
					"specific tool choice requires at least one function name".to_string(),
				))
			}
			Some(ToolChoice::Specific { function_names }) => Err(Error::InvalidRequest(format!(
				"specific tool choice names unavailable functions: {}",
				function_names.join(", ")
			))),
			None | Some(ToolChoice::Auto | ToolChoice::None) => Ok((None, None)),
		};
	}
	let resolved = match &request.tool_choice {
		None | Some(ToolChoice::Auto) => (Some(tools), None),
		Some(ToolChoice::None) => (None, None),
		Some(ToolChoice::Required) => {
			tracing::warn!(
				"tool_choice=Required is best-effort: the local engine cannot force \
				 tool-call decoding"
			);
			(
				Some(tools),
				Some(
					"You MUST respond by calling one of the available tools; do not \
					 answer in plain text."
						.to_string(),
				),
			)
		}
		Some(ToolChoice::Specific { function_names }) => {
			if function_names.is_empty() {
				return Err(Error::InvalidRequest(
					"specific tool choice requires at least one function name".to_string(),
				));
			}
			tracing::warn!(
				"tool_choice=Specific is best-effort: the local engine cannot force \
				 tool-call decoding"
			);
			// Advertise only the named tools - showing the model the full
			// list invites calls the caller explicitly excluded.
			let named: Vec<Tool> = tools
				.iter()
				.filter(|tool| function_names.contains(&tool.function.name))
				.cloned()
				.collect();
			let unknown = function_names
				.iter()
				.filter(|name| !tools.iter().any(|tool| tool.function.name == name.as_str()))
				.cloned()
				.collect::<Vec<_>>();
			if !unknown.is_empty() {
				return Err(Error::InvalidRequest(format!(
					"specific tool choice names unavailable functions: {}",
					unknown.join(", ")
				)));
			}
			(
				Some(named),
				Some(format!(
					"You MUST respond by calling one of these tools: {}; do not answer \
					 in plain text.",
					function_names.join(", ")
				)),
			)
		}
	};
	Ok(resolved)
}

/// Best-effort structured output: inject the JSON schema into the system
/// block. Decoding is not grammar-constrained, so conformance is
/// probabilistic; rig's extractor retry loop covers occasional misses.
pub(super) fn schema_instruction(request: &CompletionRequest) -> Result<Option<String>, Error> {
	let Some(schema) = &request.output_schema else {
		return Ok(None);
	};
	let encoded = super::bounded_json_bytes(schema.as_value(), 1 << 20, "output schema")?;
	let schema_json = String::from_utf8(encoded)
		.map_err(|_| Error::InvalidRequest("output schema is not valid UTF-8 JSON".to_string()))?;
	Ok(Some(format!(
		"Respond ONLY with a JSON object that conforms to this JSON Schema, with \
		 no surrounding prose or code fences:\n{schema_json}"
	)))
}
