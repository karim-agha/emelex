#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::{
	collections::{BTreeSet, VecDeque},
	os::unix::fs::MetadataExt as _,
	pin::Pin,
	sync::{
		Arc, Mutex,
		atomic::{AtomicBool, AtomicUsize, Ordering},
	},
	task::{Context, Poll},
	time::Duration,
};

use async_trait::async_trait;
use futures::Stream;
use tokio::sync::Notify;

use super::*;

struct FakeModel {
	rounds: Mutex<VecDeque<Vec<GenerationEvent>>>,
}

impl FakeModel {
	fn new(rounds: Vec<Vec<GenerationEvent>>) -> Self {
		Self {
			rounds: Mutex::new(rounds.into()),
		}
	}
}

impl AgentModel for FakeModel {
	fn stream(&self, _request: GenerationRequest) -> Result<AgentGeneration, Error> {
		let events = self
			.rounds
			.lock()
			.expect("fake model lock")
			.pop_front()
			.expect("fake model round");
		Ok(AgentGeneration::new(futures::stream::iter(
			events.into_iter().map(Ok),
		)))
	}
}

struct CountingCheckpoint {
	calls: Arc<AtomicUsize>,
}

#[async_trait]
impl CheckpointEmitter for CountingCheckpoint {
	async fn checkpoint(&mut self, _checkpoint: AgentCheckpoint<'_>) -> Result<(), AgentError> {
		self.calls.fetch_add(1, Ordering::Relaxed);
		Ok(())
	}
}

fn response(text: &str, calls: Vec<ToolCall>, finish_reason: FinishReason) -> GenerationResponse {
	GenerationResponse {
		text: text.to_string(),
		reasoning: None,
		tool_calls: calls,
		usage: Usage {
			prompt_tokens: 10,
			cached_tokens: 2,
			completion_tokens: 3,
		},
		finish_reason,
		speculation: None,
	}
}

fn completed(text: &str) -> GenerationEvent {
	GenerationEvent::Completed(response(text, Vec::new(), FinishReason::Stop))
}

fn builder(model: Arc<dyn AgentModel>, root: &Path) -> AgentSessionBuilder {
	AgentSessionBuilder::from_model(model, root).include_workspace_tools(false)
}

fn native_builder(root: &Path, system_prompt: bool, tools: bool) -> AgentSessionBuilder {
	let model = Arc::new(FakeModel::new(Vec::new()));
	let mut builder = AgentSessionBuilder::from_model(model, root);
	builder.native_capabilities = Some(NativeModelCapabilities {
		system_prompt: Some(system_prompt),
		tools: Some(tools),
		reasoning_history: Some(true),
		thinking_toggle: Some(true),
		default_thinking: Some(false),
	});
	builder
}

#[test]
fn loaded_client_capabilities_fail_during_authority_resolution() {
	let directory = tempfile::tempdir().expect("tempdir");
	let system_error = native_builder(directory.path(), false, true)
		.include_workspace_tools(false)
		.system_prompt("trusted")
		.authority_snapshot()
		.expect_err("unsupported system prompt");
	assert!(system_error.to_string().contains("system prompts"));

	let tools_error = native_builder(directory.path(), true, false)
		.authority_snapshot()
		.expect_err("unsupported built-in tools");
	assert!(tools_error.to_string().contains("tool declarations"));

	let history_system_error = native_builder(directory.path(), false, true)
		.include_workspace_tools(false)
		.history(vec![Message::system("stored")])
		.authority_snapshot()
		.expect_err("unsupported resumed system message");
	assert!(history_system_error.to_string().contains("system prompts"));

	let custom_tool_error = native_builder(directory.path(), true, false)
		.include_workspace_tools(false)
		.tool(Arc::new(EchoTool {
			invocations: Arc::new(AtomicUsize::new(0)),
			approval: false,
		}))
		.authority_snapshot()
		.expect_err("unsupported custom tool");
	assert!(custom_tool_error.to_string().contains("tool declarations"));

	let resumed_tool_history = vec![
		Message {
			role: Role::Assistant,
			tool_calls: vec![echo_call("stored-call", &serde_json::json!("value"))],
			..Message::default()
		},
		Message::tool("stored-call", "stored-result"),
	];
	let history_tool_error = native_builder(directory.path(), true, false)
		.include_workspace_tools(false)
		.history(resumed_tool_history)
		.authority_snapshot()
		.expect_err("unsupported resumed tool history");
	assert!(history_tool_error.to_string().contains("tool declarations"));

	native_builder(directory.path(), true, true)
		.system_prompt("trusted")
		.authority_snapshot()
		.expect("supported native authority");
}

#[test]
fn resumed_tool_history_requires_its_current_declaration() {
	let directory = tempfile::tempdir().expect("tempdir");
	let model: Arc<dyn AgentModel> = Arc::new(FakeModel::new(Vec::new()));
	let call_id = Uuid::now_v7().to_string();
	let history = vec![
		Message {
			role: Role::Assistant,
			tool_calls: vec![echo_call(&call_id, &serde_json::json!("value"))],
			..Message::default()
		},
		Message::tool(&call_id, "stored-result"),
	];
	let missing = builder(Arc::clone(&model), directory.path())
		.history(history.clone())
		.build()
		.err()
		.expect("missing current declaration");
	assert!(matches!(
		missing,
		AgentError::Configuration(message) if message.contains("references undeclared tool")
	));

	builder(model, directory.path())
		.history(history)
		.tool(Arc::new(EchoTool {
			invocations: Arc::new(AtomicUsize::new(0)),
			approval: false,
		}))
		.build()
		.expect("matching current declaration");
}

#[test]
fn loaded_client_identity_cannot_be_overridden() {
	let directory = tempfile::tempdir().expect("tempdir");
	let bound =
		ModelSnapshotId::parse(format!("owner/model@{}", "a".repeat(40))).expect("bound identity");
	let forged =
		ModelSnapshotId::parse(format!("owner/model@{}", "b".repeat(40))).expect("forged identity");
	let model = Arc::new(FakeModel::new(Vec::new()));
	let mut builder = builder(model, directory.path()).model_identity(forged);
	builder.model_identity_authority = ModelIdentityAuthority::LoadedClient(Some(bound));

	let error = builder
		.authority_snapshot()
		.expect_err("loaded client identity override must fail");
	assert!(matches!(
		error,
		AgentError::Configuration(message) if message.contains("cannot override")
	));
}

#[tokio::test]
async fn turn_forwards_text_and_reasoning_deltas_losslessly() {
	let directory = tempfile::tempdir().expect("tempdir");
	let mut terminal = response("hello", Vec::new(), FinishReason::Stop);
	terminal.reasoning = Some("think again".to_string());
	let progress = crate::generation::GenerationProgress {
		phase: crate::generation::GenerationProgressPhase::Decode,
		prompt_tokens: 10,
		cached_tokens: Some(2),
		completion_tokens: 1,
		max_output_tokens: 32,
		context_limit: 64,
	};
	let model = Arc::new(FakeModel::new(vec![vec![
		GenerationEvent::Progress(progress),
		GenerationEvent::Text("hel".to_string()),
		GenerationEvent::Reasoning("think ".to_string()),
		GenerationEvent::Text("lo".to_string()),
		GenerationEvent::Reasoning("again".to_string()),
		GenerationEvent::Completed(terminal),
	]]));
	let mut session = builder(model, directory.path()).build().expect("session");
	let mut events = Vec::new();

	let turn = session
		.run_turn("hi", &AgentCancellation::new(), |event| events.push(event))
		.await
		.expect("turn");

	let text = events
		.iter()
		.filter_map(|event| match event {
			AgentEvent::TextDelta { text, .. } => Some(text.as_str()),
			_ => None,
		})
		.collect::<String>();
	let reasoning = events
		.iter()
		.filter_map(|event| match event {
			AgentEvent::ReasoningDelta { text, .. } => Some(text.as_str()),
			_ => None,
		})
		.collect::<String>();
	assert_eq!(text, "hello");
	assert_eq!(reasoning, "think again");
	assert!(events.iter().any(|event| matches!(
		event,
		AgentEvent::ModelProgress {
			round: 1,
			progress: observed,
			..
		} if *observed == progress
	)));
	assert_eq!(turn.response.text, "hello");
	assert_eq!(session.history().len(), 2);
	assert!(matches!(
		events.first(),
		Some(AgentEvent::TurnStarted { .. })
	));
	assert!(matches!(
		events.get(1),
		Some(AgentEvent::ModelStarted { round: 1, .. })
	));
	assert!(matches!(
		events.get(events.len().saturating_sub(2)),
		Some(AgentEvent::ModelCompleted { round: 1, .. })
	));
	assert!(matches!(
		events.last(),
		Some(AgentEvent::TurnCompleted {
			model_rounds: 1,
			..
		})
	));
}

#[tokio::test]
async fn multimodal_user_message_is_preserved_in_committed_turn() {
	let directory = tempfile::tempdir().expect("tempdir");
	let model = Arc::new(FakeModel::new(vec![vec![completed("seen")]]));
	let mut session = builder(model, directory.path()).build().expect("session");
	let input = Message {
		role: Role::User,
		content: vec![
			Content::Text("describe".to_string()),
			Content::Image(vec![1, 2, 3]),
		],
		..Message::default()
	};

	let turn = session
		.run_message(input, &AgentCancellation::new(), |_| {})
		.await
		.expect("turn");

	assert_eq!(turn.messages.len(), 2);
	assert!(matches!(
		turn.messages[0].content.get(1),
		Some(Content::Image(bytes)) if bytes == &[1, 2, 3]
	));
}

#[tokio::test]
async fn non_user_input_message_is_rejected_before_model_start() {
	let directory = tempfile::tempdir().expect("tempdir");
	let model: Arc<dyn AgentModel> = Arc::new(FakeModel::new(Vec::new()));
	let mut session = builder(model, directory.path()).build().expect("session");

	let error = session
		.run_message(
			Message::assistant("not input"),
			&AgentCancellation::new(),
			|_| {},
		)
		.await
		.expect_err("role");

	assert!(matches!(error, AgentError::Configuration(_)));
	assert!(session.history().is_empty());
}

struct EchoTool {
	invocations: Arc<AtomicUsize>,
	approval: bool,
}

#[async_trait]
impl AgentTool for EchoTool {
	fn definition(&self) -> ToolDefinition {
		ToolDefinition::new(
			"echo",
			"Echo one string.",
			serde_json::json!({
				"type": "object",
				"properties": {"value": {"type": "string"}},
				"required": ["value"],
				"additionalProperties": false
			}),
		)
	}

	fn approval_requirement(
		&self,
		_context: &ToolContext,
		_arguments: &serde_json::Value,
	) -> ApprovalRequirement {
		if self.approval {
			ApprovalRequirement::Required {
				reason: "test approval".to_string(),
			}
		} else {
			ApprovalRequirement::None
		}
	}

	async fn invoke(
		&self,
		_context: &ToolContext,
		arguments: serde_json::Value,
	) -> Result<ToolOutput, ToolError> {
		self.invocations.fetch_add(1, Ordering::Relaxed);
		let value = arguments
			.get("value")
			.and_then(serde_json::Value::as_str)
			.ok_or_else(|| ToolError::RespondToModel("value is missing".to_string()))?;
		Ok(ToolOutput::success(value))
	}
}

fn echo_call(id: &str, value: &serde_json::Value) -> ToolCall {
	ToolCall {
		id: id.to_string(),
		name: "echo".to_string(),
		arguments: serde_json::json!({"value": value}),
	}
}

#[tokio::test]
async fn tool_loop_assigns_uuid_v7_and_preserves_it_in_result_history() {
	let directory = tempfile::tempdir().expect("tempdir");
	let raw_call = echo_call("call_0", &serde_json::json!("hello"));
	let model = Arc::new(FakeModel::new(vec![
		vec![
			GenerationEvent::ToolCall(raw_call.clone()),
			GenerationEvent::Completed(response("", vec![raw_call], FinishReason::ToolCalls)),
		],
		vec![completed("done")],
	]));
	let invocations = Arc::new(AtomicUsize::new(0));
	let mut session = builder(model, directory.path())
		.tool(Arc::new(EchoTool {
			invocations: Arc::clone(&invocations),
			approval: false,
		}))
		.build()
		.expect("session");

	let turn = session
		.run_turn("go", &AgentCancellation::new(), |_| {})
		.await
		.expect("turn");

	assert_eq!(turn.model_rounds, 2);
	assert_eq!(turn.messages.len(), 4);
	assert_eq!(invocations.load(Ordering::Relaxed), 1);
	let call_id = &session.history()[1].tool_calls[0].id;
	assert!(is_uuid_v7(call_id));
	assert_eq!(
		session.history()[2].tool_call_id.as_deref(),
		Some(call_id.as_str())
	);
	assert_eq!(validate_history(session.history()).expect("valid").len(), 1);
}

#[tokio::test]
async fn cancellation_after_tool_start_checkpoints_result_and_stops_batch() {
	let directory = tempfile::tempdir().expect("tempdir");
	let first = echo_call("provider-first", &serde_json::json!("first"));
	let second = echo_call("provider-second", &serde_json::json!("second"));
	let model = Arc::new(FakeModel::new(vec![vec![
		GenerationEvent::ToolCall(first.clone()),
		GenerationEvent::ToolCall(second.clone()),
		GenerationEvent::Completed(response("", vec![first, second], FinishReason::ToolCalls)),
	]]));
	let invocations = Arc::new(AtomicUsize::new(0));
	let mut session = builder(model, directory.path())
		.tool(Arc::new(EchoTool {
			invocations: Arc::clone(&invocations),
			approval: false,
		}))
		.build()
		.expect("session");
	let cancellation = AgentCancellation::new();
	let trigger = cancellation.clone();

	let error = session
		.run_turn("go", &cancellation, move |event| {
			if matches!(event, AgentEvent::ToolStarted { .. }) {
				trigger.cancel();
			}
		})
		.await
		.expect_err("cancelled batch");

	assert!(matches!(error, AgentError::Cancelled));
	assert_eq!(invocations.load(Ordering::Relaxed), 1);
	assert_eq!(session.history().len(), 4);
	assert!(matches!(
		session.history()[2].content.first(),
		Some(Content::Text(text)) if text == "first"
	));
	assert!(matches!(
		session.history()[3].content.first(),
		Some(Content::Text(text)) if text.contains("not executed")
	));
	validate_history(session.history()).expect("complete cancelled batch");
}

#[tokio::test]
async fn cancellation_from_model_events_stops_before_tool_start() {
	for cancel_on_completed in [false, true] {
		let directory = tempfile::tempdir().expect("tempdir");
		let call = echo_call("provider-id", &serde_json::json!("value"));
		let model = Arc::new(FakeModel::new(vec![vec![
			GenerationEvent::ToolCall(call.clone()),
			GenerationEvent::Completed(response("", vec![call], FinishReason::ToolCalls)),
		]]));
		let invocations = Arc::new(AtomicUsize::new(0));
		let mut session = builder(model, directory.path())
			.tool(Arc::new(EchoTool {
				invocations: Arc::clone(&invocations),
				approval: false,
			}))
			.build()
			.expect("session");
		let cancellation = AgentCancellation::new();
		let trigger = cancellation.clone();
		let mut tool_started = false;
		let mut tool_completed = false;

		let error = session
			.run_turn("go", &cancellation, |event| {
				let should_cancel = if cancel_on_completed {
					matches!(event, AgentEvent::ModelCompleted { .. })
				} else {
					matches!(event, AgentEvent::ToolCall { .. })
				};
				if should_cancel {
					trigger.cancel();
				}
				tool_started |= matches!(event, AgentEvent::ToolStarted { .. });
				tool_completed |= matches!(event, AgentEvent::ToolCompleted { .. });
			})
			.await
			.expect_err("cancelled before tool");

		assert!(matches!(error, AgentError::Cancelled));
		assert_eq!(invocations.load(Ordering::Relaxed), 0);
		assert!(!tool_started);
		assert!(!tool_completed);
		assert!(session.history().is_empty());
	}
}

#[tokio::test]
async fn duplicate_source_tool_ids_fail_before_execution() {
	let directory = tempfile::tempdir().expect("tempdir");
	let first = echo_call("duplicate", &serde_json::json!("one"));
	let second = echo_call("duplicate", &serde_json::json!("two"));
	let model = Arc::new(FakeModel::new(vec![
		vec![
			GenerationEvent::ToolCall(first.clone()),
			GenerationEvent::ToolCall(second.clone()),
			GenerationEvent::Completed(response("", vec![first, second], FinishReason::ToolCalls)),
		],
		vec![completed("clean")],
	]));
	let mut session = builder(model, directory.path()).build().expect("session");

	let error = session
		.run_turn("go", &AgentCancellation::new(), |_| {})
		.await
		.expect_err("duplicate must fail");

	assert!(matches!(error, AgentError::ModelProtocol(_)));
	assert!(session.history().is_empty());
	assert!(session.issued_tool_ids.is_empty());
	let recovered = session
		.run_turn("again", &AgentCancellation::new(), |_| {})
		.await
		.expect("clean retry");
	assert_eq!(recovered.response.text, "clean");
	assert_eq!(session.history().len(), 2);
}

#[tokio::test]
async fn event_after_terminal_response_rolls_back_turn() {
	let directory = tempfile::tempdir().expect("tempdir");
	let model = Arc::new(FakeModel::new(vec![vec![
		completed("done"),
		GenerationEvent::Text("late".to_string()),
	]]));
	let mut session = builder(model, directory.path()).build().expect("session");

	let error = session
		.run_turn("go", &AgentCancellation::new(), |_| {})
		.await
		.expect_err("late event");

	assert!(matches!(error, AgentError::ModelProtocol(_)));
	assert!(session.history().is_empty());
}

#[tokio::test]
async fn streamed_text_and_reasoning_must_match_terminal_response() {
	let directory = tempfile::tempdir().expect("tempdir");
	let text_model = Arc::new(FakeModel::new(vec![vec![
		GenerationEvent::Text("streamed".to_string()),
		completed("terminal"),
	]]));
	let mut text_session = builder(text_model, directory.path())
		.build()
		.expect("text session");
	let text_error = text_session
		.run_turn("go", &AgentCancellation::new(), |_| {})
		.await
		.expect_err("text mismatch");
	assert!(matches!(
		text_error,
		AgentError::ModelProtocol(message) if message.contains("answer text")
	));

	let mut terminal = response("answer", Vec::new(), FinishReason::Stop);
	terminal.reasoning = Some("terminal thought".to_string());
	let reasoning_model = Arc::new(FakeModel::new(vec![vec![
		GenerationEvent::Reasoning("streamed thought".to_string()),
		GenerationEvent::Completed(terminal),
	]]));
	let mut reasoning_session = builder(reasoning_model, directory.path())
		.build()
		.expect("reasoning session");
	let reasoning_error = reasoning_session
		.run_turn("go", &AgentCancellation::new(), |_| {})
		.await
		.expect_err("reasoning mismatch");
	assert!(matches!(
		reasoning_error,
		AgentError::ModelProtocol(message) if message.contains("reasoning")
	));
	assert!(text_session.history().is_empty());
	assert!(reasoning_session.history().is_empty());
}

#[tokio::test]
async fn terminal_answer_suffix_is_emitted_before_turn_completion() {
	let directory = tempfile::tempdir().expect("tempdir");
	let model = Arc::new(FakeModel::new(vec![vec![
		GenerationEvent::Text("pre".to_string()),
		completed("prefix"),
	]]));
	let mut session = builder(model, directory.path()).build().expect("session");
	let mut events = Vec::new();

	let turn = session
		.run_turn("go", &AgentCancellation::new(), |event| events.push(event))
		.await
		.expect("terminal suffix");

	let visible = events
		.iter()
		.filter_map(|event| match event {
			AgentEvent::TextDelta { text, .. } => Some(text.as_str()),
			_ => None,
		})
		.collect::<String>();
	assert_eq!(visible, "prefix");
	assert_eq!(visible, turn.response.text);
	assert!(matches!(
		events.last(),
		Some(AgentEvent::TurnCompleted { .. })
	));
}

#[tokio::test]
async fn terminal_reasoning_without_deltas_is_emitted_losslessly() {
	let directory = tempfile::tempdir().expect("tempdir");
	let mut terminal = response("answer", Vec::new(), FinishReason::Stop);
	terminal.reasoning = Some("complete thought".to_string());
	let model = Arc::new(FakeModel::new(vec![vec![GenerationEvent::Completed(
		terminal,
	)]]));
	let mut session = builder(model, directory.path()).build().expect("session");
	let mut events = Vec::new();

	let turn = session
		.run_turn("go", &AgentCancellation::new(), |event| events.push(event))
		.await
		.expect("terminal reasoning");
	let visible = events
		.iter()
		.filter_map(|event| match event {
			AgentEvent::ReasoningDelta { text, .. } => Some(text.as_str()),
			_ => None,
		})
		.collect::<String>();
	assert_eq!(visible, "complete thought");
	assert_eq!(turn.response.reasoning.as_deref(), Some(visible.as_str()));
}

#[tokio::test]
async fn terminal_reasoning_suffix_is_emitted_losslessly() {
	let directory = tempfile::tempdir().expect("tempdir");
	let mut terminal = response("answer", Vec::new(), FinishReason::Stop);
	terminal.reasoning = Some("think again".to_string());
	let model = Arc::new(FakeModel::new(vec![vec![
		GenerationEvent::Reasoning("think ".to_string()),
		GenerationEvent::Completed(terminal),
	]]));
	let mut session = builder(model, directory.path()).build().expect("session");
	let mut events = Vec::new();

	let turn = session
		.run_turn("go", &AgentCancellation::new(), |event| events.push(event))
		.await
		.expect("terminal reasoning suffix");
	let visible = events
		.iter()
		.filter_map(|event| match event {
			AgentEvent::ReasoningDelta { text, .. } => Some(text.as_str()),
			_ => None,
		})
		.collect::<String>();
	assert_eq!(visible, "think again");
	assert_eq!(turn.response.reasoning.as_deref(), Some(visible.as_str()));
}

#[tokio::test]
async fn streamed_tool_arguments_are_aggregate_bounded_before_cloning() {
	let directory = tempfile::tempdir().expect("tempdir");
	let argument = "x".repeat(MAX_TOOL_SCHEMA_BYTES - 128);
	let events = (0..=8)
		.map(|index| {
			GenerationEvent::ToolCall(ToolCall {
				id: format!("provider-{index}"),
				name: "echo".to_string(),
				arguments: serde_json::json!({"value": argument.clone()}),
			})
		})
		.collect::<Vec<_>>();
	let model = Arc::new(FakeModel::new(vec![events]));
	let mut session = builder(model, directory.path()).build().expect("session");

	let error = session
		.run_turn("go", &AgentCancellation::new(), |_| {})
		.await
		.expect_err("aggregate arguments");

	assert!(matches!(
		error,
		AgentError::ModelProtocol(message) if message.contains("streamed tool arguments exceed")
	));
	assert!(session.history().is_empty());
}

#[tokio::test]
async fn reasoning_only_terminal_response_is_rejected() {
	let directory = tempfile::tempdir().expect("tempdir");
	let mut terminal = response("", Vec::new(), FinishReason::Stop);
	terminal.reasoning = Some("private thought".to_string());
	let model = Arc::new(FakeModel::new(vec![vec![GenerationEvent::Completed(
		terminal,
	)]]));
	let mut session = builder(model, directory.path()).build().expect("session");

	let error = session
		.run_turn("go", &AgentCancellation::new(), |_| {})
		.await
		.expect_err("reasoning-only response");

	assert!(matches!(
		error,
		AgentError::ModelProtocol(message) if message.contains("no answer text")
	));
	assert!(session.history().is_empty());
}

#[tokio::test]
async fn terminal_reasoning_with_protocol_controls_is_rejected() {
	let directory = tempfile::tempdir().expect("tempdir");
	let mut terminal = response("answer", Vec::new(), FinishReason::Stop);
	terminal.reasoning = Some("unsafe\0reasoning".to_string());
	let model = Arc::new(FakeModel::new(vec![vec![GenerationEvent::Completed(
		terminal,
	)]]));
	let mut session = builder(model, directory.path()).build().expect("session");

	let error = session
		.run_turn("go", &AgentCancellation::new(), |_| {})
		.await
		.expect_err("reasoning control");

	assert!(matches!(
		error,
		AgentError::ModelProtocol(message) if message.contains("valid generated text")
	));
	assert!(session.history().is_empty());
}

struct TextThenPendingStream {
	emitted: bool,
	dropped: Arc<AtomicBool>,
}

impl Stream for TextThenPendingStream {
	type Item = Result<GenerationEvent, Error>;

	fn poll_next(mut self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
		if self.emitted {
			Poll::Pending
		} else {
			self.emitted = true;
			Poll::Ready(Some(Ok(GenerationEvent::Text("partial".to_string()))))
		}
	}
}

impl Drop for TextThenPendingStream {
	fn drop(&mut self) {
		self.dropped.store(true, Ordering::Relaxed);
	}
}

struct TextThenPendingModel {
	dropped: Arc<AtomicBool>,
}

impl AgentModel for TextThenPendingModel {
	fn stream(&self, _request: GenerationRequest) -> Result<AgentGeneration, Error> {
		Ok(AgentGeneration::new(TextThenPendingStream {
			emitted: false,
			dropped: Arc::clone(&self.dropped),
		}))
	}
}

struct NativeBarrierModel {
	release: Arc<Notify>,
	cancelled: Arc<AtomicBool>,
	completed: Arc<AtomicBool>,
	emit_text: bool,
}

impl AgentModel for NativeBarrierModel {
	fn stream(&self, _request: GenerationRequest) -> Result<AgentGeneration, Error> {
		let (sender, receiver) = tokio::sync::mpsc::channel(2);
		if self.emit_text {
			sender
				.try_send(Ok(GenerationEvent::Text("partial".to_string())))
				.expect("room for one native event");
		}
		let (completion_sender, completion_receiver) = tokio::sync::oneshot::channel();
		let release = Arc::clone(&self.release);
		let completed = Arc::clone(&self.completed);
		tokio::spawn(async move {
			release.notified().await;
			drop(sender);
			completed.store(true, Ordering::Relaxed);
			let _ = completion_sender.send(());
		});
		Ok(
			GenerationStream::new(receiver, Arc::clone(&self.cancelled), completion_receiver)
				.into(),
		)
	}
}

#[tokio::test]
async fn fallible_event_sink_drops_stream_and_rolls_back_turn() {
	let directory = tempfile::tempdir().expect("tempdir");
	let dropped = Arc::new(AtomicBool::new(false));
	let model = Arc::new(TextThenPendingModel {
		dropped: Arc::clone(&dropped),
	});
	let mut session = builder(model, directory.path()).build().expect("session");
	let mut observed = Vec::new();

	let error = session
		.try_run_turn("go", &AgentCancellation::new(), |event| {
			let fail = matches!(event, AgentEvent::TextDelta { .. });
			observed.push(event);
			if fail {
				Err("event receiver closed")
			} else {
				Ok(())
			}
		})
		.await
		.expect_err("sink failure");

	assert!(matches!(
		error,
		AgentError::EventSink(message) if message == "event receiver closed"
	));
	assert!(dropped.load(Ordering::Relaxed));
	assert!(session.history().is_empty());
	assert!(matches!(
		observed.as_slice(),
		[
			AgentEvent::TurnStarted { .. },
			AgentEvent::ModelStarted { round: 1, .. },
			AgentEvent::TextDelta { .. }
		]
	));
}

#[tokio::test]
async fn native_event_sink_failure_waits_for_inference_completion() {
	let directory = tempfile::tempdir().expect("tempdir");
	let release = Arc::new(Notify::new());
	let cancelled = Arc::new(AtomicBool::new(false));
	let completed = Arc::new(AtomicBool::new(false));
	let model = Arc::new(NativeBarrierModel {
		release: Arc::clone(&release),
		cancelled: Arc::clone(&cancelled),
		completed: Arc::clone(&completed),
		emit_text: true,
	});
	let mut session = builder(model, directory.path()).build().expect("session");
	let cancellation = AgentCancellation::new();
	let mut turn = Box::pin(session.try_run_turn("go", &cancellation, |event| {
		if matches!(event, AgentEvent::TextDelta { .. }) {
			Err("sink closed")
		} else {
			Ok(())
		}
	}));

	assert!(
		tokio::time::timeout(Duration::from_millis(20), &mut turn)
			.await
			.is_err(),
		"turn must remain pending at the native completion barrier"
	);
	assert!(cancelled.load(Ordering::Relaxed));
	assert!(!completed.load(Ordering::Relaxed));

	release.notify_one();
	let error = turn.await.expect_err("sink failure");
	assert!(matches!(error, AgentError::EventSink(message) if message == "sink closed"));
	assert!(completed.load(Ordering::Relaxed));
}

#[tokio::test]
async fn native_cancellation_waits_for_inference_completion() {
	let directory = tempfile::tempdir().expect("tempdir");
	let release = Arc::new(Notify::new());
	let cancelled = Arc::new(AtomicBool::new(false));
	let completed = Arc::new(AtomicBool::new(false));
	let model = Arc::new(NativeBarrierModel {
		release: Arc::clone(&release),
		cancelled: Arc::clone(&cancelled),
		completed: Arc::clone(&completed),
		emit_text: false,
	});
	let mut session = builder(model, directory.path()).build().expect("session");
	let cancellation = AgentCancellation::new();
	let mut turn = Box::pin(session.run_turn("go", &cancellation, |_| {}));

	assert!(
		tokio::time::timeout(Duration::from_millis(20), &mut turn)
			.await
			.is_err(),
		"uncancelled native stream must be pending"
	);
	assert!(!cancelled.load(Ordering::Relaxed));
	cancellation.cancel();

	assert!(
		tokio::time::timeout(Duration::from_millis(20), &mut turn)
			.await
			.is_err(),
		"turn must remain pending at the native completion barrier"
	);
	assert!(cancelled.load(Ordering::Relaxed));
	assert!(!completed.load(Ordering::Relaxed));

	release.notify_one();
	assert!(matches!(turn.await, Err(AgentError::Cancelled)));
	assert!(completed.load(Ordering::Relaxed));
}

#[tokio::test]
async fn schema_mismatch_becomes_complete_recoverable_tool_result() {
	let directory = tempfile::tempdir().expect("tempdir");
	let raw_call = echo_call("call_0", &serde_json::json!(7));
	let model = Arc::new(FakeModel::new(vec![
		vec![
			GenerationEvent::ToolCall(raw_call.clone()),
			GenerationEvent::Completed(response("", vec![raw_call], FinishReason::ToolCalls)),
		],
		vec![completed("recovered")],
	]));
	let invocations = Arc::new(AtomicUsize::new(0));
	let mut session = builder(model, directory.path())
		.tool(Arc::new(EchoTool {
			invocations: Arc::clone(&invocations),
			approval: false,
		}))
		.build()
		.expect("session");

	session
		.run_turn("go", &AgentCancellation::new(), |_| {})
		.await
		.expect("turn");

	assert_eq!(invocations.load(Ordering::Relaxed), 0);
	assert!(matches!(
		session.history()[2].content.first(),
		Some(Content::Text(text)) if text.contains("do not satisfy")
	));
	assert!(validate_history(session.history()).is_ok());
}

#[tokio::test]
async fn denied_approval_does_not_invoke_tool_and_keeps_history_complete() {
	let directory = tempfile::tempdir().expect("tempdir");
	let raw_call = echo_call("call_0", &serde_json::json!("hello"));
	let model = Arc::new(FakeModel::new(vec![
		vec![
			GenerationEvent::ToolCall(raw_call.clone()),
			GenerationEvent::Completed(response("", vec![raw_call], FinishReason::ToolCalls)),
		],
		vec![completed("understood")],
	]));
	let invocations = Arc::new(AtomicUsize::new(0));
	let mut session = builder(model, directory.path())
		.tool(Arc::new(EchoTool {
			invocations: Arc::clone(&invocations),
			approval: true,
		}))
		.build()
		.expect("session");

	session
		.run_turn("go", &AgentCancellation::new(), |_| {})
		.await
		.expect("turn");

	assert_eq!(invocations.load(Ordering::Relaxed), 0);
	assert!(matches!(
		session.history()[2].content.first(),
		Some(Content::Text(text)) if text.contains("denied")
	));
	assert!(validate_history(session.history()).is_ok());
}

struct PendingStream {
	dropped: Arc<AtomicBool>,
}

impl Stream for PendingStream {
	type Item = Result<GenerationEvent, Error>;

	fn poll_next(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
		Poll::Pending
	}
}

impl Drop for PendingStream {
	fn drop(&mut self) {
		self.dropped.store(true, Ordering::Relaxed);
	}
}

struct PendingModel {
	dropped: Arc<AtomicBool>,
}

impl AgentModel for PendingModel {
	fn stream(&self, _request: GenerationRequest) -> Result<AgentGeneration, Error> {
		Ok(AgentGeneration::new(PendingStream {
			dropped: Arc::clone(&self.dropped),
		}))
	}
}

#[tokio::test]
async fn cancellation_drops_underlying_generation_stream() {
	let directory = tempfile::tempdir().expect("tempdir");
	let dropped = Arc::new(AtomicBool::new(false));
	let model = Arc::new(PendingModel {
		dropped: Arc::clone(&dropped),
	});
	let mut session = builder(model, directory.path()).build().expect("session");
	let cancellation = AgentCancellation::new();
	let trigger = cancellation.clone();
	tokio::spawn(async move {
		tokio::time::sleep(Duration::from_millis(10)).await;
		trigger.cancel();
	});

	let attachment = Message {
		role: Role::User,
		content: vec![
			Content::Text("wait".to_string()),
			Content::Audio(vec![1, 2, 3]),
		],
		..Message::default()
	};
	let error = session
		.run_message(attachment, &cancellation, |_| {})
		.await
		.expect_err("cancel");

	assert!(matches!(error, AgentError::Cancelled));
	assert!(dropped.load(Ordering::Relaxed));
	assert!(session.history().is_empty());
	assert!(session.issued_tool_ids.is_empty());
}

#[tokio::test]
async fn cancellation_waiter_handles_existing_and_future_requests() {
	let already = AgentCancellation::new();
	already.cancel();
	tokio::time::timeout(Duration::from_millis(10), already.cancelled())
		.await
		.expect("already-cancelled waiter");

	let future = AgentCancellation::new();
	let waiter = future.clone();
	let waiting = tokio::spawn(async move {
		waiter.cancelled().await;
	});
	tokio::task::yield_now().await;
	future.cancel();
	tokio::time::timeout(Duration::from_millis(100), waiting)
		.await
		.expect("future waiter wake")
		.expect("waiter task");
}

#[test]
fn resume_rejects_unresolved_and_mismatched_tool_results() {
	let directory = tempfile::tempdir().expect("tempdir");
	let model: Arc<dyn AgentModel> = Arc::new(FakeModel::new(Vec::new()));
	let call = ToolCall {
		id: Uuid::now_v7().to_string(),
		name: "echo".to_string(),
		arguments: serde_json::json!({"value": "x"}),
	};
	let unresolved = Message {
		role: Role::Assistant,
		content: Vec::new(),
		tool_calls: vec![call],
		tool_call_id: None,
		reasoning: None,
	};
	let unresolved_error = builder(Arc::clone(&model), directory.path())
		.history(vec![unresolved])
		.build()
		.err()
		.expect("unresolved");
	let mismatched_error = builder(model, directory.path())
		.history(vec![Message::tool(Uuid::now_v7().to_string(), "orphan")])
		.build()
		.err()
		.expect("mismatched");

	assert!(matches!(
		unresolved_error,
		AgentError::History(HistoryError::Unresolved { .. })
	));
	assert!(matches!(
		mismatched_error,
		AgentError::History(HistoryError::MismatchedResult { .. })
	));
}

#[tokio::test]
async fn max_round_limit_rolls_back_complete_tool_batch() {
	let directory = tempfile::tempdir().expect("tempdir");
	let raw_call = echo_call("call_0", &serde_json::json!("hello"));
	let model = Arc::new(FakeModel::new(vec![vec![
		GenerationEvent::ToolCall(raw_call.clone()),
		GenerationEvent::Completed(response("", vec![raw_call], FinishReason::ToolCalls)),
	]]));
	let invocations = Arc::new(AtomicUsize::new(0));
	let mut session = builder(model, directory.path())
		.tool(Arc::new(EchoTool {
			invocations: Arc::clone(&invocations),
			approval: false,
		}))
		.max_model_rounds(1)
		.build()
		.expect("session");

	let error = session
		.run_turn("go", &AgentCancellation::new(), |_| {})
		.await
		.expect_err("max rounds");

	assert!(matches!(error, AgentError::MaxModelRounds { limit: 1 }));
	assert_eq!(invocations.load(Ordering::Relaxed), 0);
	assert!(session.history().is_empty());
	assert!(session.issued_tool_ids.is_empty());
	assert!(validate_history(session.history()).is_ok());
}

#[tokio::test]
async fn unlimited_model_rounds_run_past_the_bounded_ceiling() {
	let directory = tempfile::tempdir().expect("tempdir");
	let tool_rounds = MAX_AGENT_MODEL_ROUNDS + 5;
	let mut rounds = Vec::with_capacity(tool_rounds + 1);
	for index in 0..tool_rounds {
		let raw_call = echo_call(&format!("call_{index}"), &serde_json::json!("hello"));
		rounds.push(vec![
			GenerationEvent::ToolCall(raw_call.clone()),
			GenerationEvent::Completed(response("", vec![raw_call], FinishReason::ToolCalls)),
		]);
	}
	rounds.push(vec![GenerationEvent::Completed(response(
		"done",
		Vec::new(),
		FinishReason::Stop,
	))]);
	let model = Arc::new(FakeModel::new(rounds));
	let invocations = Arc::new(AtomicUsize::new(0));
	let mut session = builder(model, directory.path())
		.tool(Arc::new(EchoTool {
			invocations: Arc::clone(&invocations),
			approval: false,
		}))
		.unlimited_model_rounds()
		.build()
		.expect("session");

	let turn = session
		.run_turn("go", &AgentCancellation::new(), |_| {})
		.await
		.expect("unbounded turn");

	assert_eq!(turn.model_rounds, tool_rounds + 1);
	assert_eq!(invocations.load(Ordering::Relaxed), tool_rounds);
}

#[tokio::test]
async fn event_sink_failure_before_tool_invocation_rolls_back() {
	let directory = tempfile::tempdir().expect("tempdir");
	let raw_call = echo_call("call_0", &serde_json::json!("hello"));
	let model = Arc::new(FakeModel::new(vec![vec![
		GenerationEvent::ToolCall(raw_call.clone()),
		GenerationEvent::Completed(response("", vec![raw_call], FinishReason::ToolCalls)),
	]]));
	let invocations = Arc::new(AtomicUsize::new(0));
	let mut session = builder(model, directory.path())
		.tool(Arc::new(EchoTool {
			invocations: Arc::clone(&invocations),
			approval: false,
		}))
		.build()
		.expect("session");

	let error = session
		.try_run_turn("go", &AgentCancellation::new(), |event| {
			if matches!(event, AgentEvent::ToolStarted { .. }) {
				Err("tool-start sink failed")
			} else {
				Ok(())
			}
		})
		.await
		.expect_err("sink");

	assert!(matches!(error, AgentError::EventSink(_)));
	assert_eq!(invocations.load(Ordering::Relaxed), 0);
	assert!(session.history().is_empty());
	assert!(session.issued_tool_ids.is_empty());
}

#[tokio::test]
async fn event_sink_failure_after_tool_invocation_checkpoints_batch() {
	let directory = tempfile::tempdir().expect("tempdir");
	let raw_call = echo_call("call_0", &serde_json::json!("hello"));
	let model = Arc::new(FakeModel::new(vec![vec![
		GenerationEvent::ToolCall(raw_call.clone()),
		GenerationEvent::Completed(response("", vec![raw_call], FinishReason::ToolCalls)),
	]]));
	let invocations = Arc::new(AtomicUsize::new(0));
	let mut session = builder(model, directory.path())
		.tool(Arc::new(EchoTool {
			invocations: Arc::clone(&invocations),
			approval: false,
		}))
		.build()
		.expect("session");

	let error = session
		.try_run_turn("go", &AgentCancellation::new(), |event| {
			if matches!(event, AgentEvent::ToolCompleted { .. }) {
				Err("tool-complete sink failed")
			} else {
				Ok(())
			}
		})
		.await
		.expect_err("sink");

	assert!(matches!(error, AgentError::EventSink(_)));
	assert_eq!(invocations.load(Ordering::Relaxed), 1);
	assert_eq!(session.history().len(), 3);
	assert_eq!(session.issued_tool_ids.len(), 1);
	assert!(validate_history(session.history()).is_ok());
	assert!(matches!(
		session.history()[2].content.first(),
		Some(Content::Text(text)) if text == "hello"
	));
}

struct InvalidTool;

#[async_trait]
impl AgentTool for InvalidTool {
	fn definition(&self) -> ToolDefinition {
		ToolDefinition::new("bad name", "invalid", serde_json::json!({"type": "object"}))
	}

	fn approval_requirement(
		&self,
		_context: &ToolContext,
		_arguments: &serde_json::Value,
	) -> ApprovalRequirement {
		ApprovalRequirement::None
	}

	async fn invoke(
		&self,
		_context: &ToolContext,
		_arguments: serde_json::Value,
	) -> Result<ToolOutput, ToolError> {
		Ok(ToolOutput::success(""))
	}
}

#[test]
fn builder_rejects_invalid_tool_definition() {
	let directory = tempfile::tempdir().expect("tempdir");
	let model = Arc::new(FakeModel::new(Vec::new()));
	let error = builder(model, directory.path())
		.tool(Arc::new(InvalidTool))
		.build()
		.err()
		.expect("invalid tool");

	assert!(matches!(error, AgentError::Configuration(_)));
}

#[test]
fn file_tools_remain_when_shell_is_authoritatively_disabled() {
	let directory = tempfile::tempdir().expect("tempdir");
	let model = Arc::new(FakeModel::new(Vec::new()));
	let session = AgentSessionBuilder::from_model(model, directory.path())
		.include_file_tools(true)
		.include_shell_tool(false)
		.build()
		.expect("session");

	assert!(session.tools.contains_key("read_file"));
	assert!(session.tools.contains_key("edit_file"));
	assert!(!session.tools.contains_key("shell"));
}

#[test]
fn all_authorized_tools_begin_enabled_and_available() {
	let directory = tempfile::tempdir().expect("tempdir");
	let model = Arc::new(FakeModel::new(Vec::new()));
	let session = builder(model, directory.path())
		.tool(Arc::new(EchoTool {
			invocations: Arc::new(AtomicUsize::new(0)),
			approval: false,
		}))
		.tool(Arc::new(FatalTool))
		.build()
		.expect("session");
	let available = session
		.available_tools()
		.map(|definition| definition.name.as_str())
		.collect::<Vec<_>>();

	assert_eq!(available, vec!["echo", "fatal"]);
	assert_eq!(
		session.enabled_tools(),
		&BTreeSet::from(["echo".to_string(), "fatal".to_string()])
	);
}

#[test]
fn unknown_enabled_tool_rejection_is_atomic() {
	let directory = tempfile::tempdir().expect("tempdir");
	let model = Arc::new(FakeModel::new(Vec::new()));
	let mut session = builder(model, directory.path())
		.tool(Arc::new(EchoTool {
			invocations: Arc::new(AtomicUsize::new(0)),
			approval: false,
		}))
		.build()
		.expect("session");
	let original = session.enabled_tools().clone();

	let error = session
		.set_enabled_tools(BTreeSet::from(["echo".to_string(), "missing".to_string()]))
		.expect_err("unknown tool");

	assert!(matches!(
		error,
		AgentError::ToolUnavailable { tool_name } if tool_name == "missing"
	));
	assert_eq!(session.enabled_tools(), &original);
}

#[test]
fn enabled_tool_changes_do_not_change_authority_snapshot() {
	let directory = tempfile::tempdir().expect("tempdir");
	let model = Arc::new(FakeModel::new(Vec::new()));
	let mut session = builder(model, directory.path())
		.tool(Arc::new(EchoTool {
			invocations: Arc::new(AtomicUsize::new(0)),
			approval: false,
		}))
		.build()
		.expect("session");
	let authority = session.authority_snapshot().clone();

	session
		.set_enabled_tools(BTreeSet::new())
		.expect("disable tools");

	assert_eq!(session.authority_snapshot(), &authority);
}

#[test]
fn disabled_tools_still_validate_recorded_authority_ceilings() {
	let directory = tempfile::tempdir().expect("tempdir");
	for seconds in [0, MAX_SHELL_TIMEOUT_SECONDS + 1] {
		let model = Arc::new(FakeModel::new(Vec::new()));
		let error = AgentSessionBuilder::from_model(model, directory.path())
			.include_shell_tool(false)
			.shell_timeout_seconds(seconds)
			.build()
			.err()
			.expect("invalid disabled shell timeout");
		assert!(
			matches!(error, AgentError::Configuration(message) if message.contains("shell_timeout"))
		);
	}
	for bytes in [0, MAX_SHELL_OUTPUT_BYTES + 1] {
		let model = Arc::new(FakeModel::new(Vec::new()));
		let error = AgentSessionBuilder::from_model(model, directory.path())
			.include_shell_tool(false)
			.shell_output_bytes(bytes)
			.build()
			.err()
			.expect("invalid disabled shell output");
		assert!(
			matches!(error, AgentError::Configuration(message) if message.contains("shell_output"))
		);
	}
	for bytes in [0, MAX_WEB_RESPONSE_BYTES + 1] {
		let model = Arc::new(FakeModel::new(Vec::new()));
		let error = AgentSessionBuilder::from_model(model, directory.path())
			.include_web_fetch(false)
			.web_response_bytes(bytes)
			.build()
			.err()
			.expect("invalid disabled web response");
		assert!(
			matches!(error, AgentError::Configuration(message) if message.contains("web_response"))
		);
	}
}

#[test]
fn authority_snapshot_matches_built_session_and_preserves_builder() {
	let directory = tempfile::tempdir().expect("tempdir");
	let model = Arc::new(FakeModel::new(Vec::new()));
	let builder = AgentSessionBuilder::from_model(model, directory.path())
		.include_file_tools(true)
		.include_shell_tool(false)
		.include_datetime(true)
		.max_model_rounds(7)
		.max_tool_output_bytes(32 * 1024);

	let snapshot = builder.authority_snapshot().expect("authority snapshot");
	let serialized = serde_json::to_vec(&snapshot).expect("serialize snapshot");
	let decoded: AgentAuthoritySnapshot =
		serde_json::from_slice(&serialized).expect("deserialize snapshot");
	let session = builder.build().expect("session after snapshot");
	let session_definitions = session
		.tools
		.values()
		.map(|registered| registered.definition.clone())
		.collect::<Vec<_>>();

	assert_eq!(snapshot, decoded);
	assert_eq!(&snapshot, session.authority_snapshot());
	assert_eq!(snapshot.tools, session_definitions);
	assert_eq!(snapshot.tool_implementations.len(), snapshot.tools.len());
	assert_eq!(
		snapshot.tool_cancellation_policies.len(),
		snapshot.tools.len()
	);
	assert!(
		snapshot
			.tool_implementations
			.values()
			.all(|identity| !identity.is_empty())
	);
	assert!(
		snapshot
			.tool_implementations
			.values()
			.all(|identity| !identity.contains(env!("CARGO_PKG_VERSION"))),
		"durable tool authority must not change for a package-only version bump"
	);
	assert_eq!(
		snapshot.schema_version,
		AGENT_AUTHORITY_SNAPSHOT_SCHEMA_VERSION
	);
	assert_eq!(
		snapshot.workspace_root,
		directory.path().canonicalize().unwrap()
	);
	let workspace_metadata = directory.path().metadata().unwrap();
	assert_eq!(snapshot.workspace_device, workspace_metadata.dev());
	assert_eq!(snapshot.workspace_inode, workspace_metadata.ino());
	assert!(
		snapshot
			.enabled_capabilities
			.contains(&AgentBuiltinCapability::FileTools)
	);
	assert!(
		!snapshot
			.enabled_capabilities
			.contains(&AgentBuiltinCapability::ShellTool)
	);
	assert!(
		snapshot
			.enabled_capabilities
			.contains(&AgentBuiltinCapability::Datetime)
	);
	assert_eq!(snapshot.max_model_rounds, Some(7));
	assert_eq!(snapshot.max_tool_output_bytes, 32 * 1024);
}

#[test]
fn authority_snapshot_round_ceiling_is_optional_and_legacy_compatible() {
	let directory = tempfile::tempdir().expect("tempdir");
	let model = Arc::new(FakeModel::new(Vec::new()));
	let snapshot = AgentSessionBuilder::from_model(model, directory.path())
		.unlimited_model_rounds()
		.authority_snapshot()
		.expect("authority snapshot");
	assert_eq!(snapshot.max_model_rounds, None);
	let mut encoded = serde_json::to_value(&snapshot).expect("serialize snapshot");
	assert!(
		encoded.get("max_model_rounds").is_none(),
		"unbounded sessions must omit the ceiling field"
	);
	encoded["max_model_rounds"] = serde_json::json!(7);
	let decoded: AgentAuthoritySnapshot =
		serde_json::from_value(encoded).expect("legacy numeric ceiling");
	assert_eq!(decoded.max_model_rounds, Some(7));
}

#[test]
fn explicit_tool_implementation_identity_is_durable_authority() {
	let directory = tempfile::tempdir().expect("tempdir");
	let model: Arc<dyn AgentModel> = Arc::new(FakeModel::new(Vec::new()));
	let invocations = Arc::new(AtomicUsize::new(0));
	let tool = Arc::new(EchoTool {
		invocations,
		approval: false,
	});
	let first = builder(Arc::clone(&model), directory.path())
		.tool_with_identity(tool.clone(), "echo-implementation@v1")
		.authority_snapshot()
		.expect("first authority");
	let second = builder(model, directory.path())
		.tool_with_identity(tool, "echo-implementation@v2")
		.authority_snapshot()
		.expect("second authority");

	assert_eq!(
		first.tool_implementations.get("echo").map(String::as_str),
		Some("echo-implementation@v1")
	);
	assert_ne!(first, second);
}

struct InterruptibleEchoTool(EchoTool);

#[async_trait]
impl AgentTool for InterruptibleEchoTool {
	fn definition(&self) -> ToolDefinition {
		self.0.definition()
	}

	fn implementation_identity(&self) -> String {
		self.0.implementation_identity()
	}

	fn cancellation_policy(&self) -> ToolCancellationPolicy {
		ToolCancellationPolicy::Interruptible
	}

	fn approval_requirement(
		&self,
		context: &ToolContext,
		arguments: &serde_json::Value,
	) -> ApprovalRequirement {
		self.0.approval_requirement(context, arguments)
	}

	async fn invoke(
		&self,
		context: &ToolContext,
		arguments: serde_json::Value,
	) -> Result<ToolOutput, ToolError> {
		self.0.invoke(context, arguments).await
	}
}

#[test]
fn tool_cancellation_policy_is_durable_authority() {
	let directory = tempfile::tempdir().expect("tempdir");
	let model: Arc<dyn AgentModel> = Arc::new(FakeModel::new(Vec::new()));
	let regular = builder(Arc::clone(&model), directory.path())
		.tool_with_identity(
			Arc::new(EchoTool {
				invocations: Arc::new(AtomicUsize::new(0)),
				approval: false,
			}),
			"same-implementation",
		)
		.authority_snapshot()
		.expect("finish policy");
	let interruptible = builder(model, directory.path())
		.tool_with_identity(
			Arc::new(InterruptibleEchoTool(EchoTool {
				invocations: Arc::new(AtomicUsize::new(0)),
				approval: false,
			})),
			"same-implementation",
		)
		.authority_snapshot()
		.expect("interruptible policy");

	assert_ne!(regular, interruptible);
	assert_eq!(
		regular.tool_cancellation_policies.get("echo"),
		Some(&ToolCancellationPolicy::FinishOnceStarted)
	);
	assert_eq!(
		interruptible.tool_cancellation_policies.get("echo"),
		Some(&ToolCancellationPolicy::Interruptible)
	);
}

#[test]
fn builder_rejects_invalid_tool_implementation_identity() {
	let directory = tempfile::tempdir().expect("tempdir");
	let model = Arc::new(FakeModel::new(Vec::new()));
	let invocations = Arc::new(AtomicUsize::new(0));
	let result = builder(model, directory.path())
		.tool_with_identity(
			Arc::new(EchoTool {
				invocations,
				approval: false,
			}),
			"bad\nidentity",
		)
		.build();

	assert!(matches!(result, Err(AgentError::Configuration(_))));
}

struct FatalTool;

#[async_trait]
impl AgentTool for FatalTool {
	fn definition(&self) -> ToolDefinition {
		ToolDefinition::new(
			"fatal",
			"Always fail fatally.",
			serde_json::json!({
				"type": "object",
				"properties": {},
				"additionalProperties": false
			}),
		)
	}

	fn approval_requirement(
		&self,
		_context: &ToolContext,
		_arguments: &serde_json::Value,
	) -> ApprovalRequirement {
		ApprovalRequirement::None
	}

	async fn invoke(
		&self,
		_context: &ToolContext,
		_arguments: serde_json::Value,
	) -> Result<ToolOutput, ToolError> {
		Err(ToolError::Fatal("boom".to_string()))
	}
}

#[tokio::test]
async fn fatal_tool_checkpoints_complete_uncertain_batch() {
	let directory = tempfile::tempdir().expect("tempdir");
	let call = ToolCall {
		id: "provider-id".to_string(),
		name: "fatal".to_string(),
		arguments: serde_json::json!({}),
	};
	let model = Arc::new(FakeModel::new(vec![vec![
		GenerationEvent::ToolCall(call.clone()),
		GenerationEvent::Completed(response("", vec![call], FinishReason::ToolCalls)),
	]]));
	let mut session = builder(model, directory.path())
		.tool(Arc::new(FatalTool))
		.build()
		.expect("session");

	let error = session
		.run_turn("go", &AgentCancellation::new(), |_| {})
		.await
		.expect_err("fatal");

	assert!(matches!(error, AgentError::ToolFatal { .. }));
	assert_eq!(session.history().len(), 3);
	assert_eq!(session.issued_tool_ids.len(), 1);
	assert!(validate_history(session.history()).is_ok());
}

struct RecordingModel {
	requests: Arc<Mutex<Vec<GenerationRequest>>>,
}

impl AgentModel for RecordingModel {
	fn stream(&self, request: GenerationRequest) -> Result<AgentGeneration, Error> {
		self.requests
			.lock()
			.expect("request recorder")
			.push(request);
		Ok(AgentGeneration::new(futures::stream::iter([Ok(
			completed("done"),
		)])))
	}
}

#[tokio::test]
async fn coalesced_history_cannot_exceed_shared_message_limit() {
	let directory = tempfile::tempdir().expect("tempdir");
	let requests = Arc::new(Mutex::new(Vec::new()));
	let model = Arc::new(RecordingModel {
		requests: Arc::clone(&requests),
	});
	let history = vec![Message::user(
		"x".repeat(crate::generation::MAX_MESSAGE_CONTENT_BYTES),
	)];
	let mut session = builder(model, directory.path())
		.history(history)
		.build()
		.expect("session");

	let error = session
		.run_turn("y", &AgentCancellation::new(), |_| {})
		.await
		.expect_err("coalesced message must remain bounded");

	assert!(matches!(
		error,
		AgentError::Generation(Error::InvalidRequest(message))
			if message.contains("one message")
	));
	assert!(requests.lock().expect("requests").is_empty());
	assert_eq!(session.history().len(), 1);
}

struct RecordingRoundsModel {
	requests: Arc<Mutex<Vec<GenerationRequest>>>,
	rounds: Mutex<VecDeque<Vec<GenerationEvent>>>,
}

impl AgentModel for RecordingRoundsModel {
	fn stream(&self, request: GenerationRequest) -> Result<AgentGeneration, Error> {
		self.requests
			.lock()
			.expect("request recorder")
			.push(request);
		let events = self
			.rounds
			.lock()
			.expect("round queue")
			.pop_front()
			.expect("recording model round");
		Ok(AgentGeneration::new(futures::stream::iter(
			events.into_iter().map(Ok),
		)))
	}
}

#[tokio::test]
async fn disabled_unused_tool_is_omitted_from_generation_request() {
	let directory = tempfile::tempdir().expect("tempdir");
	let requests = Arc::new(Mutex::new(Vec::new()));
	let model = Arc::new(RecordingModel {
		requests: Arc::clone(&requests),
	});
	let mut session = builder(model, directory.path())
		.tool(Arc::new(EchoTool {
			invocations: Arc::new(AtomicUsize::new(0)),
			approval: false,
		}))
		.tool(Arc::new(FatalTool))
		.build()
		.expect("session");
	session
		.set_enabled_tools(BTreeSet::from(["echo".to_string()]))
		.expect("disable fatal");

	session
		.run_turn("answer directly", &AgentCancellation::new(), |_| {})
		.await
		.expect("turn");

	let requests = requests.lock().expect("requests");
	assert_eq!(requests.len(), 1);
	assert_eq!(requests[0].tools.len(), 1);
	assert_eq!(requests[0].tools[0].name, "echo");
	drop(requests);
}

#[tokio::test]
async fn disabled_tool_call_never_reaches_approval_or_invocation() {
	let directory = tempfile::tempdir().expect("tempdir");
	let requests = Arc::new(Mutex::new(Vec::new()));
	let raw_call = echo_call("disabled-call", &serde_json::json!("value"));
	let model = Arc::new(RecordingRoundsModel {
		requests: Arc::clone(&requests),
		rounds: Mutex::new(
			vec![
				vec![
					GenerationEvent::ToolCall(raw_call.clone()),
					GenerationEvent::Completed(response(
						"",
						vec![raw_call],
						FinishReason::ToolCalls,
					)),
				],
				vec![completed("answered without tools")],
			]
			.into(),
		),
	});
	let invocations = Arc::new(AtomicUsize::new(0));
	let mut session = builder(model, directory.path())
		.tool(Arc::new(EchoTool {
			invocations: Arc::clone(&invocations),
			approval: true,
		}))
		.build()
		.expect("session");
	session
		.set_enabled_tools(BTreeSet::new())
		.expect("disable tools");
	let mut events = Vec::new();

	session
		.run_turn("do not use tools", &AgentCancellation::new(), |event| {
			events.push(event);
		})
		.await
		.expect("turn");

	assert_eq!(invocations.load(Ordering::Relaxed), 0);
	assert!(
		events
			.iter()
			.all(|event| !matches!(event, AgentEvent::ApprovalRequested { .. }))
	);
	assert!(
		events
			.iter()
			.all(|event| !matches!(event, AgentEvent::ToolStarted { .. }))
	);
	assert!(matches!(
		session.history()[2].content.first(),
		Some(Content::Text(text)) if text.contains("unavailable")
	));
	let requests = requests.lock().expect("requests");
	assert_eq!(requests.len(), 2);
	assert!(requests[0].tools.is_empty());
	assert_eq!(requests[1].tools.len(), 1);
	assert_eq!(requests[1].tools[0].name, "echo");
	drop(requests);
}

#[tokio::test]
async fn reenabled_tool_is_advertised_and_invoked() {
	let directory = tempfile::tempdir().expect("tempdir");
	let requests = Arc::new(Mutex::new(Vec::new()));
	let raw_call = echo_call("reenabled-call", &serde_json::json!("value"));
	let model = Arc::new(RecordingRoundsModel {
		requests: Arc::clone(&requests),
		rounds: Mutex::new(
			vec![
				vec![
					GenerationEvent::ToolCall(raw_call.clone()),
					GenerationEvent::Completed(response(
						"",
						vec![raw_call],
						FinishReason::ToolCalls,
					)),
				],
				vec![completed("done")],
			]
			.into(),
		),
	});
	let invocations = Arc::new(AtomicUsize::new(0));
	let mut session = builder(model, directory.path())
		.tool(Arc::new(EchoTool {
			invocations: Arc::clone(&invocations),
			approval: false,
		}))
		.build()
		.expect("session");
	session
		.set_enabled_tools(BTreeSet::new())
		.expect("disable tools");
	session
		.set_enabled_tools(BTreeSet::from(["echo".to_string()]))
		.expect("reenable echo");

	session
		.run_turn("use echo", &AgentCancellation::new(), |_| {})
		.await
		.expect("turn");

	assert_eq!(invocations.load(Ordering::Relaxed), 1);
	let requests = requests.lock().expect("requests");
	assert_eq!(requests[0].tools.len(), 1);
	assert_eq!(requests[0].tools[0].name, "echo");
	drop(requests);
}

#[tokio::test]
async fn historical_disabled_tool_remains_declared_for_replay() {
	let directory = tempfile::tempdir().expect("tempdir");
	let requests = Arc::new(Mutex::new(Vec::new()));
	let model = Arc::new(RecordingModel {
		requests: Arc::clone(&requests),
	});
	let call_id = Uuid::now_v7().to_string();
	let history = vec![
		Message {
			role: Role::Assistant,
			tool_calls: vec![echo_call(&call_id, &serde_json::json!("value"))],
			..Message::default()
		},
		Message::tool(&call_id, "stored-result"),
	];
	let mut session = builder(model, directory.path())
		.history(history)
		.tool(Arc::new(EchoTool {
			invocations: Arc::new(AtomicUsize::new(0)),
			approval: false,
		}))
		.build()
		.expect("session");
	session
		.set_enabled_tools(BTreeSet::new())
		.expect("disable tools");

	session
		.run_turn("continue", &AgentCancellation::new(), |_| {})
		.await
		.expect("turn");

	let requests = requests.lock().expect("requests");
	assert_eq!(requests.len(), 1);
	assert_eq!(requests[0].tools.len(), 1);
	assert_eq!(requests[0].tools[0].name, "echo");
	drop(requests);
}

#[tokio::test]
async fn per_turn_thinking_proceeds_and_strips_reasoning_when_history_is_unsupported() {
	let directory = tempfile::tempdir().expect("tempdir");
	let requests = Arc::new(Mutex::new(Vec::new()));
	let checkpoints = Arc::new(AtomicUsize::new(0));
	let mut terminal = response("answer", Vec::new(), FinishReason::Stop);
	terminal.reasoning = Some("fresh thought".to_string());
	let model = Arc::new(RecordingRoundsModel {
		requests: Arc::clone(&requests),
		rounds: Mutex::new(vec![vec![GenerationEvent::Completed(terminal)]].into()),
	});
	let mut session_builder = builder(model, directory.path());
	session_builder.native_capabilities = Some(NativeModelCapabilities {
		system_prompt: Some(true),
		tools: Some(true),
		reasoning_history: Some(false),
		thinking_toggle: Some(true),
		default_thinking: Some(false),
	});
	let mut session = session_builder.build().expect("thinking-capable session");

	let turn = session
		.run_message_core(
			Message::user("think"),
			GenerationOptions {
				thinking: Some(crate::config::ThinkingMode::On),
				..GenerationOptions::default()
			},
			&AgentCancellation::new(),
			InfallibleEmitter(|_| {}),
			CountingCheckpoint {
				calls: Arc::clone(&checkpoints),
			},
		)
		.await
		.expect("thinking-on turn without reasoning-history support");

	let recorded = requests.lock().expect("requests");
	assert_eq!(recorded.len(), 1);
	assert_eq!(
		recorded[0].options.thinking,
		Some(crate::config::ThinkingMode::On)
	);
	assert!(
		recorded[0]
			.messages
			.iter()
			.all(|message| message.reasoning.is_none())
	);
	drop(recorded);
	assert!(checkpoints.load(Ordering::Relaxed) > 0);
	assert_eq!(turn.response.reasoning.as_deref(), Some("fresh thought"));
	assert_eq!(session.history().len(), 2);
	assert!(
		session
			.history()
			.iter()
			.all(|message| message.reasoning.is_none())
	);
}

#[tokio::test]
async fn resumed_reasoning_history_degrades_to_stripped_replay_for_unpreserving_model() {
	let directory = tempfile::tempdir().expect("tempdir");
	let requests = Arc::new(Mutex::new(Vec::new()));
	let model = Arc::new(RecordingModel {
		requests: Arc::clone(&requests),
	});
	let mut recorded = Message::assistant("earlier answer");
	recorded.reasoning = Some("recorded by a reasoning-capable model".to_string());
	let mut session_builder = builder(model, directory.path())
		.history(vec![Message::user("earlier"), recorded])
		.generation_options(GenerationOptions {
			thinking: Some(crate::config::ThinkingMode::On),
			..GenerationOptions::default()
		});
	session_builder.native_capabilities = Some(NativeModelCapabilities {
		system_prompt: Some(true),
		tools: Some(true),
		reasoning_history: Some(false),
		thinking_toggle: Some(true),
		default_thinking: Some(false),
	});
	let mut session = session_builder.build().expect("resumed session");

	session
		.run_turn("continue", &AgentCancellation::new(), |_| {})
		.await
		.expect("stripped replay turn");

	let requests = requests.lock().expect("requests");
	assert_eq!(requests.len(), 1);
	assert!(
		requests[0]
			.messages
			.iter()
			.all(|message| message.reasoning.is_none())
	);
	drop(requests);
	// The stored transcript keeps its recorded reasoning; only the request
	// boundary degrades, so resuming later on a capable model loses nothing.
	assert!(
		session
			.history()
			.iter()
			.any(|message| message.reasoning.is_some())
	);
}

#[tokio::test]
async fn per_turn_generation_options_override_session_options_field_by_field() {
	let directory = tempfile::tempdir().expect("tempdir");
	let requests = Arc::new(Mutex::new(Vec::new()));
	let model = Arc::new(RecordingModel {
		requests: Arc::clone(&requests),
	});
	let mut session = builder(model, directory.path())
		.generation_options(GenerationOptions {
			max_tokens: Some(64),
			temperature: Some(0.25),
			thinking: Some(crate::config::ThinkingMode::Auto),
			..GenerationOptions::default()
		})
		.build()
		.expect("session");

	session
		.run_turn_with_options(
			"go",
			GenerationOptions {
				max_tokens: Some(32),
				thinking: Some(crate::config::ThinkingMode::On),
				..GenerationOptions::default()
			},
			&AgentCancellation::new(),
			|_| {},
		)
		.await
		.expect("turn");

	let options = {
		let requests = requests.lock().expect("requests");
		requests.first().expect("request").options
	};
	assert_eq!(options.max_tokens, Some(32));
	assert_eq!(options.temperature, Some(0.25));
	assert_eq!(options.thinking, Some(crate::config::ThinkingMode::On));
}

#[tokio::test]
async fn per_turn_thinking_off_clears_session_reasoning_budget() {
	let directory = tempfile::tempdir().expect("tempdir");
	let requests = Arc::new(Mutex::new(Vec::new()));
	let model = Arc::new(RecordingModel {
		requests: Arc::clone(&requests),
	});
	let mut session = builder(model, directory.path())
		.generation_options(GenerationOptions {
			thinking: Some(crate::config::ThinkingMode::On),
			reasoning_budget_tokens: Some(32),
			..GenerationOptions::default()
		})
		.build()
		.expect("thinking session");

	session
		.run_turn_with_options(
			"go",
			GenerationOptions {
				thinking: Some(crate::config::ThinkingMode::Off),
				..GenerationOptions::default()
			},
			&AgentCancellation::new(),
			|_| {},
		)
		.await
		.expect("thinking off turn");

	let options = {
		let requests = requests.lock().expect("requests");
		requests.first().expect("request").options
	};
	assert_eq!(
		(options.thinking, options.reasoning_budget_tokens),
		(Some(crate::config::ThinkingMode::Off), None)
	);
}

#[tokio::test]
async fn per_turn_thinking_auto_clears_session_reasoning_budget() {
	let directory = tempfile::tempdir().expect("tempdir");
	let requests = Arc::new(Mutex::new(Vec::new()));
	let model = Arc::new(RecordingModel {
		requests: Arc::clone(&requests),
	});
	let mut session_builder =
		builder(model, directory.path()).generation_options(GenerationOptions {
			thinking: Some(crate::config::ThinkingMode::On),
			reasoning_budget_tokens: Some(32),
			..GenerationOptions::default()
		});
	session_builder.native_capabilities = Some(NativeModelCapabilities {
		system_prompt: Some(true),
		tools: Some(true),
		reasoning_history: Some(true),
		thinking_toggle: Some(true),
		default_thinking: Some(false),
	});
	let mut session = session_builder.build().expect("thinking session");

	session
		.run_turn_with_options(
			"go",
			GenerationOptions {
				thinking: Some(crate::config::ThinkingMode::Auto),
				..GenerationOptions::default()
			},
			&AgentCancellation::new(),
			|_| {},
		)
		.await
		.expect("automatic thinking turn");

	let options = {
		let requests = requests.lock().expect("requests");
		requests.first().expect("request").options
	};
	assert_eq!(
		(options.thinking, options.reasoning_budget_tokens),
		(Some(crate::config::ThinkingMode::Auto), None)
	);
}

#[tokio::test]
async fn unpreservable_terminal_reasoning_is_visible_but_not_committed() {
	let directory = tempfile::tempdir().expect("tempdir");
	let requests = Arc::new(Mutex::new(Vec::new()));
	let mut terminal = response("answer", Vec::new(), FinishReason::Stop);
	terminal.reasoning = Some("template ignored thinking off".to_string());
	let model = Arc::new(RecordingRoundsModel {
		requests,
		rounds: Mutex::new(vec![vec![GenerationEvent::Completed(terminal)]].into()),
	});
	let mut session_builder = builder(model, directory.path());
	session_builder.native_capabilities = Some(NativeModelCapabilities {
		system_prompt: Some(true),
		tools: Some(true),
		reasoning_history: Some(false),
		thinking_toggle: Some(true),
		default_thinking: Some(false),
	});
	let mut session = session_builder.build().expect("non-thinking session");

	let turn = session
		.run_turn("go", &AgentCancellation::new(), |_| {})
		.await
		.expect("terminal answer");

	assert_eq!(
		turn.response.reasoning.as_deref(),
		Some("template ignored thinking off")
	);
	assert!(turn.messages[1].reasoning.is_none());
	assert!(session.history()[1].reasoning.is_none());
}

#[tokio::test]
async fn unpreservable_tool_round_reasoning_is_removed_before_follow_up() {
	let directory = tempfile::tempdir().expect("tempdir");
	let requests = Arc::new(Mutex::new(Vec::new()));
	let call = echo_call("reasoning-call", &serde_json::json!("value"));
	let mut first = response("", vec![call.clone()], FinishReason::ToolCalls);
	first.reasoning = Some("unreplayable thought".to_string());
	let model = Arc::new(RecordingRoundsModel {
		requests: Arc::clone(&requests),
		rounds: Mutex::new(
			vec![
				vec![
					GenerationEvent::ToolCall(call),
					GenerationEvent::Completed(first),
				],
				vec![completed("done")],
			]
			.into(),
		),
	});
	let invocations = Arc::new(AtomicUsize::new(0));
	let mut session_builder = builder(model, directory.path()).tool(Arc::new(EchoTool {
		invocations: Arc::clone(&invocations),
		approval: false,
	}));
	session_builder.native_capabilities = Some(NativeModelCapabilities {
		system_prompt: Some(true),
		tools: Some(true),
		reasoning_history: Some(false),
		thinking_toggle: Some(true),
		default_thinking: Some(false),
	});
	let mut session = session_builder.build().expect("non-thinking session");

	session
		.run_turn("go", &AgentCancellation::new(), |_| {})
		.await
		.expect("tool round");

	let requests = requests.lock().expect("requests");
	assert_eq!(requests.len(), 2);
	let replayed_assistant = requests[1]
		.messages
		.iter()
		.find(|message| message.role == Role::Assistant && !message.tool_calls.is_empty())
		.expect("assistant tool call");
	assert!(replayed_assistant.reasoning.is_none());
	drop(requests);
	assert_eq!(invocations.load(Ordering::Relaxed), 1);
	assert!(
		session
			.history()
			.iter()
			.all(|message| message.reasoning.is_none())
	);
}

#[test]
fn explicit_system_prompt_rejects_ambiguous_resumed_system_message() {
	let directory = tempfile::tempdir().expect("tempdir");
	let model = Arc::new(FakeModel::new(Vec::new()));
	let error = builder(model, directory.path())
		.history(vec![Message::system("persisted system")])
		.system_prompt("new system")
		.build()
		.err()
		.expect("system conflict");

	assert!(matches!(
		error,
		AgentError::Configuration(message) if message.contains("conflicts")
	));
}

struct UnsafeApprovalReasonTool;

#[async_trait]
impl AgentTool for UnsafeApprovalReasonTool {
	fn definition(&self) -> ToolDefinition {
		ToolDefinition::new(
			"unsafe_reason",
			"Request an intentionally malformed approval reason.",
			serde_json::json!({
				"type": "object",
				"properties": {},
				"additionalProperties": false
			}),
		)
	}

	fn approval_requirement(
		&self,
		_context: &ToolContext,
		_arguments: &serde_json::Value,
	) -> ApprovalRequirement {
		ApprovalRequirement::Required {
			reason: "trusted prefix\nforged prompt".to_string(),
		}
	}

	async fn invoke(
		&self,
		_context: &ToolContext,
		_arguments: serde_json::Value,
	) -> Result<ToolOutput, ToolError> {
		Err(ToolError::Fatal(
			"unsafe approval reason must never reach invocation".to_string(),
		))
	}
}

#[tokio::test]
async fn approval_reason_control_characters_fail_before_policy_or_invocation() {
	let directory = tempfile::tempdir().expect("tempdir");
	let call = ToolCall {
		id: "provider-id".to_string(),
		name: "unsafe_reason".to_string(),
		arguments: serde_json::json!({}),
	};
	let model = Arc::new(FakeModel::new(vec![vec![
		GenerationEvent::ToolCall(call.clone()),
		GenerationEvent::Completed(response("", vec![call], FinishReason::ToolCalls)),
	]]));
	let mut session = builder(model, directory.path())
		.tool(Arc::new(UnsafeApprovalReasonTool))
		.build()
		.expect("session");

	let error = session
		.run_turn("go", &AgentCancellation::new(), |_| {})
		.await
		.expect_err("unsafe approval reason");

	assert!(matches!(
		error,
		AgentError::ToolFatal { message, .. } if message.contains("control characters")
	));
	assert!(session.history().is_empty());
}

struct OversizedDenial;

#[async_trait]
impl ApprovalPolicy for OversizedDenial {
	async fn decide(&self, _context: &ApprovalContext) -> ApprovalDecision {
		ApprovalDecision::Deny {
			reason: "x".repeat(MAX_APPROVAL_REASON_BYTES + 1),
		}
	}
}

#[tokio::test]
async fn oversized_denial_is_rejected_before_resolved_event() {
	let directory = tempfile::tempdir().expect("tempdir");
	let call = echo_call("provider-id", &serde_json::json!("hello"));
	let model = Arc::new(FakeModel::new(vec![vec![
		GenerationEvent::ToolCall(call.clone()),
		GenerationEvent::Completed(response("", vec![call], FinishReason::ToolCalls)),
	]]));
	let mut session = builder(model, directory.path())
		.tool(Arc::new(EchoTool {
			invocations: Arc::new(AtomicUsize::new(0)),
			approval: true,
		}))
		.approval_policy(Arc::new(OversizedDenial))
		.build()
		.expect("session");
	let mut resolved = false;

	let error = session
		.run_turn("go", &AgentCancellation::new(), |event| {
			resolved |= matches!(event, AgentEvent::ApprovalResolved { .. });
		})
		.await
		.expect_err("oversized denial");

	assert!(matches!(
		error,
		AgentError::ToolFatal { message, .. } if message.contains("approval denial exceeded")
	));
	assert!(!resolved);
	assert!(session.history().is_empty());
}

struct PendingApproval {
	started: Arc<Notify>,
	dropped: Arc<AtomicBool>,
}

struct PendingApprovalGuard(Arc<AtomicBool>);

impl Drop for PendingApprovalGuard {
	fn drop(&mut self) {
		self.0.store(true, Ordering::Release);
	}
}

#[async_trait]
impl ApprovalPolicy for PendingApproval {
	async fn decide(&self, _context: &ApprovalContext) -> ApprovalDecision {
		let _guard = PendingApprovalGuard(Arc::clone(&self.dropped));
		self.started.notify_one();
		std::future::pending::<ApprovalDecision>().await
	}
}

#[tokio::test]
async fn cancellation_abandons_pending_approval_without_waiting() {
	let directory = tempfile::tempdir().expect("tempdir");
	let call = echo_call("provider-id", &serde_json::json!("hello"));
	let model = Arc::new(FakeModel::new(vec![vec![
		GenerationEvent::ToolCall(call.clone()),
		GenerationEvent::Completed(response("", vec![call], FinishReason::ToolCalls)),
	]]));
	let started = Arc::new(Notify::new());
	let dropped = Arc::new(AtomicBool::new(false));
	let invocations = Arc::new(AtomicUsize::new(0));
	let mut session = builder(model, directory.path())
		.tool(Arc::new(EchoTool {
			invocations: Arc::clone(&invocations),
			approval: true,
		}))
		.approval_policy(Arc::new(PendingApproval {
			started: Arc::clone(&started),
			dropped: Arc::clone(&dropped),
		}))
		.build()
		.expect("session");
	let cancellation = AgentCancellation::new();
	let cancellation_task = cancellation.clone();
	let cancel_when_started = tokio::spawn(async move {
		started.notified().await;
		cancellation_task.cancel();
	});

	let result = tokio::time::timeout(
		Duration::from_secs(1),
		session.run_turn("go", &cancellation, |_| {}),
	)
	.await
	.expect("turn must stop promptly after cancellation");
	cancel_when_started.await.expect("cancellation task");

	assert!(matches!(result, Err(AgentError::Cancelled)));
	assert!(dropped.load(Ordering::Acquire));
	assert_eq!(invocations.load(Ordering::Relaxed), 0);
	assert!(session.history().is_empty());
}

#[test]
fn agent_rejects_translation_content_with_actionable_error() {
	let message = Message::translation("en", "de", "hello");
	let error = validate_user_message(&message).expect_err("translation content rejected");
	assert!(error.to_string().contains("emelex translate"));
}
