//! Downstream-style construction pins for extensible public input types.

// Test code: panics are the assertion mechanism.
#![allow(clippy::panic, missing_docs)]

use emelex::{
	generation::{ToolCall, ToolDefinition},
	model::{EvidenceSource, ModelGenerationDefaults, ModelSizing, TraitEvidence},
};

#[test]
fn public_inputs_have_forward_compatible_construction_paths() {
	let definition = ToolDefinition::new(
		"lookup",
		"Look up one value",
		serde_json::json!({"type": "object"}),
	);
	let call = ToolCall::new("call-1", "lookup", serde_json::json!({"key": "value"}));
	let evidence = TraitEvidence::new(
		"interaction:tool_use",
		EvidenceSource::Runtime,
		"probe passed",
	);
	let mut sizing = ModelSizing::default();
	sizing.weights_bytes = Some(42);
	let mut generation = ModelGenerationDefaults::default();
	generation.max_new_tokens = Some(128);

	assert_eq!(definition.name, call.name);
	assert_eq!(evidence.trait_key, "interaction:tool_use");
	assert_eq!(sizing.weights_bytes, Some(42));
	assert_eq!(generation.max_new_tokens, Some(128));
}
