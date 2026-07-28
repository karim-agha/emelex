//! Feature-independent public API contract.

use emelex::{
	Client, ClientBuilder,
	generation::{
		GenerationEvent, GenerationRequest, GenerationResponse, GenerationStream, ToolCall,
		ToolDefinition,
	},
};

#[test]
fn native_generation_surface_compiles_without_rig() {
	fn accepts_client(_: Option<Client>) {}
	fn accepts_builder(_: Option<ClientBuilder>) {}
	fn accepts_stream(_: Option<GenerationStream>) {}
	fn accepts_event(_: Option<GenerationEvent>) {}
	fn accepts_response(_: Option<GenerationResponse>) {}
	fn accepts_call(_: Option<ToolCall>) {}

	let mut request = GenerationRequest::text("hello");
	request.tools.push(ToolDefinition::new(
		"lookup",
		"Lookup a value",
		serde_json::json!({
			"type": "object",
			"properties": {"key": {"type": "string"}}
		}),
	));
	request.options.max_tokens = Some(16);

	assert_eq!(request.messages.len(), 1);
	accepts_client(None);
	accepts_builder(None);
	accepts_stream(None);
	accepts_event(None);
	accepts_response(None);
	accepts_call(None);
}
