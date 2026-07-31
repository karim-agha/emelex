//! Native, provider-independent agent loop and tool approval boundary.

mod web;
mod workspace;

use std::{
	collections::{BTreeMap, BTreeSet},
	fmt::{self, Write as _},
	path::{Path, PathBuf},
	pin::Pin,
	sync::Arc,
	task::{Context, Poll},
};

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
pub use web::{
	MAX_WEB_RESPONSE_BYTES, WebError, WebSearchError, WebSearchProvider, WebSearchResult,
	datetime_tool, web_fetch_tool, web_fetch_tool_with_limit, web_search_tool,
};
pub use workspace::{
	MAX_SHELL_OUTPUT_BYTES, MAX_SHELL_TIMEOUT_SECONDS, WorkspaceError, file_tools, shell_tool,
	workspace_tools,
};

use crate::{
	Client, Error,
	generation::{
		Content, FinishReason, GenerationEvent, GenerationOptions, GenerationRequest,
		GenerationResponse, GenerationStream, Message, Role, ToolCall, ToolDefinition, Usage,
	},
	model::ModelSnapshotId,
};

const DEFAULT_MAX_MODEL_ROUNDS: usize = 16;
/// Maximum model/tool cycles in one agent turn.
pub const MAX_AGENT_MODEL_ROUNDS: usize = 20;
const MAX_TOOLS: usize = 256;
pub(crate) const MAX_TOOL_CALLS_PER_ROUND: usize = 16;
pub(crate) const MAX_TOOL_OUTPUT_BYTES: usize = 256 * 1024;
pub(crate) const MAX_TOTAL_TOOL_OUTPUT_BYTES: usize = 768 * 1024;
const MAX_TOOL_SCHEMA_BYTES: usize = 1024 * 1024;
const MAX_TOOL_CALL_ID_BYTES: usize = 256;
pub(crate) const MAX_TOTAL_TOOL_ARGUMENT_BYTES: usize = 512 * 1024;
const MAX_TOOL_DESCRIPTION_BYTES: usize = 16 * 1024;
const MAX_APPROVAL_REASON_BYTES: usize = 4 * 1024;
const MAX_USER_CONTENT_BYTES: usize = 128 * 1024 * 1024;
const MAX_SYSTEM_PROMPT_BYTES: usize = 4 * 1024 * 1024;
const MAX_TOOL_IMPLEMENTATION_IDENTITY_BYTES: usize = 1024;
const MAX_MODEL_OUTPUT_BYTES: usize = 192 * 1024;
const MAX_TOTAL_MODEL_OUTPUT_BYTES: usize = 768 * 1024;
const MAX_EVENT_SINK_ERROR_BYTES: usize = 4 * 1024;
/// Current serialized shape of [`AgentAuthoritySnapshot`].
pub const AGENT_AUTHORITY_SNAPSHOT_SCHEMA_VERSION: u32 = 4;

/// A model capable of starting one native Emelex generation stream.
///
/// [`Client`] implements this trait. The abstraction also permits deterministic
/// tests and alternate local schedulers without changing the agent loop.
pub trait AgentModel: Send + Sync {
	/// Whether this model is known to preserve a distinct system role.
	///
	/// `None` keeps alternate implementations backward-compatible but defers
	/// this capability check to their request boundary.
	fn supports_system_prompt(&self) -> Option<bool> {
		None
	}

	/// Whether this model is known to preserve tool declarations and rounds.
	///
	/// `None` keeps alternate implementations backward-compatible but defers
	/// this capability check to their request boundary.
	fn supports_tools(&self) -> Option<bool> {
		None
	}

	/// Whether explicit reasoning spans survive a follow-up turn.
	fn supports_reasoning_history(&self) -> Option<bool> {
		None
	}

	/// Whether the model's template distinguishes thinking on from off.
	fn supports_thinking_toggle(&self) -> Option<bool> {
		None
	}

	/// Loaded-client thinking default, when this implementation exposes one.
	fn default_thinking_enabled(&self) -> Option<bool> {
		None
	}

	/// Validate and start one bounded generation stream.
	///
	/// # Errors
	///
	/// Returns native request-validation, queue-admission, or model errors.
	fn stream(&self, request: GenerationRequest) -> Result<AgentGeneration, Error>;
}

impl AgentModel for Client {
	fn supports_system_prompt(&self) -> Option<bool> {
		Some(Self::supports_system_prompt(self))
	}

	fn supports_tools(&self) -> Option<bool> {
		Some(Self::supports_tools(self))
	}

	fn supports_reasoning_history(&self) -> Option<bool> {
		Some(Self::supports_reasoning_history(self))
	}

	fn supports_thinking_toggle(&self) -> Option<bool> {
		Some(Self::supports_thinking_toggle(self))
	}

	fn default_thinking_enabled(&self) -> Option<bool> {
		self.inner.defaults.enable_thinking
	}

	fn stream(&self, request: GenerationRequest) -> Result<AgentGeneration, Error> {
		Self::stream(self, request).map(AgentGeneration::from)
	}
}

/// Type-erased, cancel-on-drop generation stream consumed by an agent.
pub struct AgentGeneration {
	inner: Option<AgentGenerationInner>,
}

enum AgentGenerationInner {
	Native(GenerationStream),
	Custom(Pin<Box<dyn Stream<Item = Result<GenerationEvent, Error>> + Send>>),
}

impl AgentGeneration {
	/// Wrap a custom generation stream.
	///
	/// Custom streams are cancelled by dropping them. Use
	/// [`AgentGeneration::from`] for a native [`GenerationStream`] so early
	/// agent exits can also wait for the inference job to leave its worker.
	pub fn new<S>(stream: S) -> Self
	where
		S: Stream<Item = Result<GenerationEvent, Error>> + Send + 'static,
	{
		Self {
			inner: Some(AgentGenerationInner::Custom(Box::pin(stream))),
		}
	}

	/// Cooperatively request generation cancellation.
	///
	/// Native streams receive a cancellation signal and retain their
	/// completion receiver so a later [`Self::cancel_and_wait`] can establish
	/// the worker barrier. Custom streams are dropped.
	pub fn cancel(&mut self) {
		match self.inner.as_mut() {
			Some(AgentGenerationInner::Native(stream)) => stream.cancel(),
			Some(AgentGenerationInner::Custom(_)) => self.inner = None,
			None => {}
		}
	}

	/// Cancel generation and, for a native stream, wait until its inference
	/// job leaves the loaded model's dedicated thread.
	///
	/// Custom streams created with [`Self::new`] have no completion hook and
	/// are cancelled by dropping them.
	///
	/// # Errors
	///
	/// Returns a native inference-channel error if the worker exits without
	/// completing the submitted job.
	pub async fn cancel_and_wait(&mut self) -> Result<(), Error> {
		match self.inner.as_mut() {
			Some(AgentGenerationInner::Native(stream)) => stream.cancel_and_wait().await,
			Some(AgentGenerationInner::Custom(_)) => {
				self.inner = None;
				Ok(())
			}
			None => Ok(()),
		}
	}
}

impl Stream for AgentGeneration {
	type Item = Result<GenerationEvent, Error>;

	fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
		let Some(inner) = self.inner.as_mut() else {
			return Poll::Ready(None);
		};
		match inner {
			AgentGenerationInner::Native(stream) => Pin::new(stream).poll_next(context),
			AgentGenerationInner::Custom(stream) => stream.as_mut().poll_next(context),
		}
	}
}

impl From<GenerationStream> for AgentGeneration {
	fn from(stream: GenerationStream) -> Self {
		Self {
			inner: Some(AgentGenerationInner::Native(stream)),
		}
	}
}

/// Cloneable cooperative cancellation handle for one or more agent turns.
#[derive(Clone)]
pub struct AgentCancellation {
	sender: tokio::sync::watch::Sender<bool>,
}

impl Default for AgentCancellation {
	fn default() -> Self {
		let (sender, _receiver) = tokio::sync::watch::channel(false);
		Self { sender }
	}
}

impl AgentCancellation {
	/// Construct a live cancellation handle.
	pub fn new() -> Self {
		Self::default()
	}

	/// Request cancellation. Repeated calls are harmless.
	pub fn cancel(&self) {
		self.sender.send_replace(true);
	}

	/// Whether cancellation has already been requested.
	pub fn is_cancelled(&self) -> bool {
		*self.sender.borrow()
	}

	/// Wait until cancellation is requested.
	///
	/// Tool and provider implementations can select this future against their
	/// own I/O without polling or sleeping.
	pub async fn cancelled(&self) {
		let mut receiver = self.sender.subscribe();
		if *receiver.borrow() {
			return;
		}
		let _ = receiver.wait_for(|cancelled| *cancelled).await;
	}
}

/// Why one tool invocation needs an approval decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
#[non_exhaustive]
pub enum ApprovalRequirement {
	/// Invocation may run without approval.
	None,
	/// Invocation crosses a boundary that needs a one-shot decision.
	Required {
		/// Human-readable risk or boundary.
		reason: String,
	},
}

/// One-shot approval decision.
///
/// Durable adapters may audit the resolved decision, but approval grants are
/// never restored or reused after the invocation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
#[non_exhaustive]
pub enum ApprovalDecision {
	/// Permit only this exact invocation.
	AllowOnce,
	/// Refuse this invocation.
	Deny {
		/// Human-readable refusal reason.
		reason: String,
	},
}

/// Immutable context presented to an approval policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct ApprovalContext {
	/// Durable tool-call identifier.
	pub call_id: String,
	/// Registered tool name.
	pub tool_name: String,
	/// Exact model-proposed arguments.
	pub arguments: serde_json::Value,
	/// Canonical workspace root.
	pub workspace_root: PathBuf,
	/// Workspace device identity captured by the opened root descriptor.
	pub workspace_device: u64,
	/// Workspace inode identity captured by the opened root descriptor.
	pub workspace_inode: u64,
	/// Why approval is required.
	pub reason: String,
}

/// Asynchronous, per-call approval policy.
#[async_trait]
pub trait ApprovalPolicy: Send + Sync {
	/// Decide one exact invocation. Implementations must not infer durable
	/// grants from a prior call.
	async fn decide(&self, context: &ApprovalContext) -> ApprovalDecision;
}

/// Safe default: deny every invocation that asks for approval.
#[derive(Debug, Default, Clone, Copy)]
pub struct DenyAllApprovals;

#[async_trait]
impl ApprovalPolicy for DenyAllApprovals {
	async fn decide(&self, _context: &ApprovalContext) -> ApprovalDecision {
		ApprovalDecision::Deny {
			reason: "approval policy denied this invocation".to_string(),
		}
	}
}

/// Explicit non-interactive policy that permits every requested invocation.
///
/// This policy does not sandbox tools. Callers should use it only in an
/// already-isolated environment.
#[derive(Debug, Default, Clone, Copy)]
pub struct AllowAllApprovals;

#[async_trait]
impl ApprovalPolicy for AllowAllApprovals {
	async fn decide(&self, _context: &ApprovalContext) -> ApprovalDecision {
		ApprovalDecision::AllowOnce
	}
}

/// Per-call execution context passed to tools.
#[derive(Clone)]
pub struct ToolContext {
	call_id: String,
	workspace: Arc<workspace::WorkspaceRoot>,
	cancellation: AgentCancellation,
	approved: bool,
}

impl ToolContext {
	/// Durable identifier preserved in assistant and tool-result messages.
	pub fn call_id(&self) -> &str {
		&self.call_id
	}

	/// Canonical workspace root.
	pub fn workspace_root(&self) -> &Path {
		self.workspace.path()
	}

	/// Cooperative turn cancellation handle.
	pub const fn cancellation(&self) -> &AgentCancellation {
		&self.cancellation
	}

	/// Whether the approval policy allowed this exact invocation.
	pub const fn approved(&self) -> bool {
		self.approved
	}
}

/// Bounded text returned by a tool.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct ToolOutput {
	/// Text sent back to the model.
	pub content: String,
	/// Whether the text represents a recoverable tool failure.
	pub is_error: bool,
}

impl ToolOutput {
	/// Construct successful output.
	pub fn success(content: impl Into<String>) -> Self {
		Self {
			content: content.into(),
			is_error: false,
		}
	}

	/// Construct recoverable failure output.
	pub fn error(content: impl Into<String>) -> Self {
		Self {
			content: content.into(),
			is_error: true,
		}
	}
}

/// Tool execution failure routing.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ToolError {
	/// Return this error text to the model so it can recover.
	#[error("{0}")]
	RespondToModel(String),
	/// Stop the current tool batch because cooperative cancellation completed.
	#[error("tool invocation cancelled")]
	Cancelled,
	/// Abort the turn because continuing is unsafe or impossible.
	#[error("fatal tool failure: {0}")]
	Fatal(String),
}

/// How the harness responds to cancellation after a tool starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ToolCancellationPolicy {
	/// Drop the invocation future as soon as cancellation wins.
	///
	/// Use only when dropping [`AgentTool::invoke`] cannot leave background
	/// work or host effects running.
	Interruptible,
	/// Await one terminal invocation result, then stop the remaining batch.
	FinishOnceStarted,
}

/// Asynchronous executable tool.
#[async_trait]
pub trait AgentTool: Send + Sync {
	/// Stable declaration advertised to the model.
	fn definition(&self) -> ToolDefinition;

	/// Stable executable implementation identity used by durable sessions.
	///
	/// Override this when behavior or configuration can change independently
	/// of the concrete Rust type or Emelex version. Alternatively, callers can
	/// supply an exact digest/version through
	/// [`AgentSessionBuilder::tool_with_identity`].
	fn implementation_identity(&self) -> String {
		format!("rust:{}@protocol-1", std::any::type_name::<Self>())
	}

	/// Cancellation behavior after the durable `ToolStarted` boundary.
	///
	/// The conservative default finishes one invocation. Implementations that
	/// start host work must also ensure dropping their invocation future cannot
	/// leave that work detached.
	fn cancellation_policy(&self) -> ToolCancellationPolicy {
		ToolCancellationPolicy::FinishOnceStarted
	}

	/// Classify the exact invocation before it runs.
	fn approval_requirement(
		&self,
		context: &ToolContext,
		arguments: &serde_json::Value,
	) -> ApprovalRequirement;

	/// Execute one invocation.
	///
	/// # Errors
	///
	/// Returns either recoverable model-facing text or a fatal turn error.
	async fn invoke(
		&self,
		context: &ToolContext,
		arguments: serde_json::Value,
	) -> Result<ToolOutput, ToolError>;
}

/// Typed lifecycle item emitted synchronously as an agent turn advances.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
#[non_exhaustive]
pub enum AgentEvent {
	/// User input was accepted.
	TurnStarted {
		/// Durable turn identifier.
		turn_id: Uuid,
	},
	/// One model round started.
	ModelStarted {
		/// Durable turn identifier.
		turn_id: Uuid,
		/// One-based round within this turn.
		round: usize,
	},
	/// Lossless answer-text delta.
	TextDelta {
		/// Durable turn identifier.
		turn_id: Uuid,
		/// One-based model round.
		round: usize,
		/// Exact native delta.
		text: String,
	},
	/// Lossless reasoning-text delta.
	ReasoningDelta {
		/// Durable turn identifier.
		turn_id: Uuid,
		/// One-based model round.
		round: usize,
		/// Exact native delta.
		text: String,
	},
	/// A validated model tool proposal received its durable ID.
	ToolCall {
		/// Durable turn identifier.
		turn_id: Uuid,
		/// One-based model round.
		round: usize,
		/// Re-identified tool call.
		call: ToolCall,
	},
	/// Approval was requested for one exact call.
	ApprovalRequested {
		/// Approval context.
		context: ApprovalContext,
	},
	/// Approval policy resolved one exact call.
	ApprovalResolved {
		/// Durable tool-call identifier.
		call_id: String,
		/// One-shot decision.
		decision: ApprovalDecision,
	},
	/// Tool execution started.
	ToolStarted {
		/// Durable tool-call identifier.
		call_id: String,
		/// Registered tool name.
		tool_name: String,
	},
	/// Tool execution produced bounded output.
	ToolCompleted {
		/// Durable tool-call identifier.
		call_id: String,
		/// Registered tool name.
		tool_name: String,
		/// Output sent to the model.
		output: ToolOutput,
	},
	/// One model round reached its terminal native response.
	ModelCompleted {
		/// Durable turn identifier.
		turn_id: Uuid,
		/// One-based model round.
		round: usize,
		/// Native stop classification.
		finish_reason: FinishReason,
		/// Native token accounting.
		usage: Usage,
	},
	/// Turn completed without unresolved calls.
	TurnCompleted {
		/// Durable turn identifier.
		turn_id: Uuid,
		/// Number of model rounds consumed.
		model_rounds: usize,
		/// Aggregate token accounting across every model round.
		usage: Usage,
	},
	/// Cancellation won the current race.
	Cancelled {
		/// Durable turn identifier.
		turn_id: Uuid,
	},
	/// Turn failed after emitting a terminal diagnostic.
	TurnFailed {
		/// Durable turn identifier.
		turn_id: Uuid,
		/// Stable human-readable diagnostic.
		message: String,
	},
}

/// Completed agent turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AgentTurn {
	/// Durable turn identifier.
	pub id: Uuid,
	/// Final assistant response after all tool rounds.
	pub response: GenerationResponse,
	/// Exact ordered message batch committed by this turn.
	///
	/// The batch starts with user input and ends with the final assistant
	/// message. Intermediate assistant calls and matching tool results are
	/// included without presentation-only stream deltas.
	pub messages: Vec<Message>,
	/// Number of model rounds consumed.
	pub model_rounds: usize,
	/// Aggregate token accounting across every model round.
	pub usage: Usage,
}

/// Invalid ordered conversation history.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum HistoryError {
	/// A message broke a role-specific invariant.
	#[error("history message {index} is invalid: {reason}")]
	InvalidMessage {
		/// Zero-based message index.
		index: usize,
		/// Violated invariant.
		reason: String,
	},
	/// A tool-call ID was reused.
	#[error("duplicate tool-call ID {call_id:?} at message {index}")]
	DuplicateCall {
		/// Reused ID.
		call_id: String,
		/// Zero-based message index.
		index: usize,
	},
	/// A result did not match one pending call.
	#[error("tool result {call_id:?} at message {index} has no matching pending call")]
	MismatchedResult {
		/// Result ID.
		call_id: String,
		/// Zero-based message index.
		index: usize,
	},
	/// Conversation advanced while calls remained unresolved.
	#[error("history has unresolved tool calls before message {index}: {call_ids:?}")]
	Unresolved {
		/// Sorted pending IDs.
		call_ids: Vec<String>,
		/// Index where the invariant was observed.
		index: usize,
	},
}

/// Agent construction or turn failure.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AgentError {
	/// Session configuration is invalid.
	#[error("invalid agent configuration: {0}")]
	Configuration(String),
	/// Conversation protocol is invalid.
	#[error(transparent)]
	History(#[from] HistoryError),
	/// Workspace root or descriptor operation failed.
	#[error(transparent)]
	Workspace(#[from] WorkspaceError),
	/// Opt-in web tool construction failed.
	#[error(transparent)]
	Web(#[from] WebError),
	/// Native generation failed.
	#[error(transparent)]
	Generation(#[from] Error),
	/// A turn failed and its native inference cleanup also failed.
	#[error("{primary}; native generation cleanup also failed: {cleanup}")]
	GenerationCleanup {
		/// Original turn failure.
		#[source]
		primary: Box<Self>,
		/// Failure while waiting for the native inference worker.
		cleanup: Error,
	},
	/// Model stream violated its terminal protocol.
	#[error("agent model protocol error: {0}")]
	ModelProtocol(String),
	/// A fatal tool failure aborted the turn.
	#[error("tool {tool_name:?} failed fatally: {message}")]
	ToolFatal {
		/// Registered tool name.
		tool_name: String,
		/// Tool-provided diagnostic.
		message: String,
	},
	/// A requested tool is unavailable to this Session's active subset.
	#[error("tool {tool_name:?} is unavailable in this agent session")]
	ToolUnavailable {
		/// Unavailable tool name.
		tool_name: String,
	},
	/// Turn hit its configured model-round ceiling.
	#[error("agent turn exceeded its {limit}-round limit")]
	MaxModelRounds {
		/// Configured ceiling.
		limit: usize,
	},
	/// Cooperative cancellation won.
	#[error("agent turn cancelled")]
	Cancelled,
	/// A fallible event consumer stopped the turn.
	#[error("agent event sink failed: {0}")]
	EventSink(String),
	/// Final success committed before its terminal event was rejected.
	#[error("agent turn {turn_id} committed, but its terminal event sink failed: {message}")]
	EventSinkAfterCommit {
		/// Committed turn identifier.
		turn_id: Uuid,
		/// Bounded sink diagnostic.
		message: String,
	},
	/// A durable boundary rejected or failed to persist turn state.
	#[error("agent checkpoint sink failed: {0}")]
	CheckpointSink(String),
}

trait EventEmitter {
	fn emit(&mut self, event: AgentEvent) -> Result<(), AgentError>;
}

struct InfallibleEmitter<F>(F);

impl<F> EventEmitter for InfallibleEmitter<F>
where
	F: FnMut(AgentEvent),
{
	fn emit(&mut self, event: AgentEvent) -> Result<(), AgentError> {
		(self.0)(event);
		Ok(())
	}
}

struct FallibleEmitter<F>(F);

impl<F, E> EventEmitter for FallibleEmitter<F>
where
	F: FnMut(AgentEvent) -> Result<(), E>,
	E: fmt::Display,
{
	fn emit(&mut self, event: AgentEvent) -> Result<(), AgentError> {
		(self.0)(event).map_err(|error| AgentError::EventSink(bounded_display(&error)))
	}
}

#[derive(Clone, Copy)]
pub(crate) enum AgentCheckpoint<'a> {
	PreflightInput {
		turn_id: Uuid,
		message: &'a Message,
	},
	ToolBatchStarted {
		turn_id: Uuid,
		messages: &'a [Message],
	},
	ToolAudit {
		turn_id: Uuid,
	},
	ToolInvocationStarted {
		turn_id: Uuid,
		call: &'a ToolCall,
	},
	ToolInvocationCompleted {
		turn_id: Uuid,
		message: &'a Message,
		event: &'a AgentEvent,
	},
	Messages {
		turn_id: Uuid,
		messages: &'a [Message],
		terminal_event: Option<&'a AgentEvent>,
	},
}

#[async_trait]
pub(crate) trait CheckpointEmitter: Send {
	async fn checkpoint(&mut self, checkpoint: AgentCheckpoint<'_>) -> Result<(), AgentError>;
}

struct NoopCheckpointEmitter;

#[async_trait]
impl CheckpointEmitter for NoopCheckpointEmitter {
	async fn checkpoint(&mut self, _checkpoint: AgentCheckpoint<'_>) -> Result<(), AgentError> {
		Ok(())
	}
}

struct RegisteredTool {
	definition: ToolDefinition,
	tool: Arc<dyn AgentTool>,
	cancellation_policy: ToolCancellationPolicy,
}

#[derive(Clone)]
struct ConfiguredTool {
	tool: Arc<dyn AgentTool>,
	implementation_identity: Option<String>,
}

impl ConfiguredTool {
	fn inferred(tool: Arc<dyn AgentTool>) -> Self {
		Self {
			tool,
			implementation_identity: None,
		}
	}
}

/// Durable description of one resolved agent authority boundary.
///
/// This captures the canonical workspace, exact declarations advertised to the
/// model, built-in capability enablement, and effective configurable ceilings.
/// Approval policy is intentionally excluded because approvals are one-shot
/// runtime decisions, not durable authority.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct AgentAuthoritySnapshot {
	/// Serialized shape version.
	pub schema_version: u32,
	/// Canonical workspace root anchoring built-in file and shell tools.
	pub workspace_root: PathBuf,
	/// Device number of the opened workspace descriptor.
	pub workspace_device: u64,
	/// Inode number of the opened workspace descriptor.
	pub workspace_inode: u64,
	/// Exact caller-supplied model/checkpoint identity for durable replay.
	pub model_identity: Option<ModelSnapshotId>,
	/// Trusted system prompt resolved when the Session was built.
	pub system_prompt: Option<String>,
	/// Session-wide generation policy below per-turn overrides.
	pub generation_options: GenerationOptions,
	/// Exact resolved tool declarations, sorted by unique tool name.
	pub tools: Vec<ToolDefinition>,
	/// Executable implementation identity keyed by exact tool name.
	pub tool_implementations: BTreeMap<String, String>,
	/// Post-start cancellation policy keyed by exact tool name.
	pub tool_cancellation_policies: BTreeMap<String, ToolCancellationPolicy>,
	/// Built-in capabilities installed in this authority boundary.
	pub enabled_capabilities: BTreeSet<AgentBuiltinCapability>,
	/// Authoritative shell duration ceiling per invocation.
	pub shell_timeout_seconds: u64,
	/// Authoritative combined shell stdout/stderr capture ceiling.
	pub shell_output_bytes: usize,
	/// Authoritative decoded HTTP response ceiling.
	pub web_response_bytes: usize,
	/// Authoritative model-round ceiling per turn.
	pub max_model_rounds: usize,
	/// Authoritative tool-result ceiling before model history.
	pub max_tool_output_bytes: usize,
}

/// One built-in capability installed in an agent authority boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum AgentBuiltinCapability {
	/// Descriptor-anchored read, traversal, write, and edit tools.
	FileTools,
	/// Host shell execution.
	ShellTool,
	/// Bounded HTTP(S) fetch.
	WebFetch,
	/// Local fixed-offset date/time.
	Datetime,
	/// Search through an explicitly supplied provider.
	WebSearch,
}

struct ResolvedAuthority {
	snapshot: AgentAuthoritySnapshot,
	workspace: Arc<workspace::WorkspaceRoot>,
	tools: BTreeMap<String, RegisteredTool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Availability {
	Enabled,
	Disabled,
}

impl Availability {
	const fn from_bool(enabled: bool) -> Self {
		if enabled {
			Self::Enabled
		} else {
			Self::Disabled
		}
	}

	const fn is_enabled(self) -> bool {
		matches!(self, Self::Enabled)
	}
}

enum ModelIdentityAuthority {
	LoadedClient(Option<ModelSnapshotId>),
	AlternateModel,
}

#[derive(Debug, Clone, Copy)]
struct NativeModelCapabilities {
	system_prompt: Option<bool>,
	tools: Option<bool>,
	reasoning_history: Option<bool>,
	thinking_toggle: Option<bool>,
	default_thinking: Option<bool>,
}

/// Builder for one in-memory agent session.
pub struct AgentSessionBuilder {
	model: Arc<dyn AgentModel>,
	native_capabilities: Option<NativeModelCapabilities>,
	workspace_root: PathBuf,
	approval_policy: Arc<dyn ApprovalPolicy>,
	tools: Vec<ConfiguredTool>,
	file_tools: Availability,
	shell_tool: Availability,
	shell_timeout_seconds: u64,
	shell_output_bytes: usize,
	web_fetch: Availability,
	web_response_bytes: usize,
	datetime: Availability,
	web_search_provider: Option<Arc<dyn WebSearchProvider>>,
	web_search_provider_identity: Option<String>,
	history: Vec<Message>,
	system_prompt: Option<String>,
	model_identity: Option<ModelSnapshotId>,
	model_identity_authority: ModelIdentityAuthority,
	generation_options: GenerationOptions,
	max_model_rounds: usize,
	max_tool_output_bytes: usize,
}

impl AgentSessionBuilder {
	/// Start a session builder on one loaded native model.
	pub fn new(client: Client, workspace_root: impl Into<PathBuf>) -> Self {
		let model_identity = client.model_snapshot_id().cloned();
		let mut builder = Self::from_model(Arc::new(client), workspace_root);
		builder.model_identity.clone_from(&model_identity);
		builder.model_identity_authority = ModelIdentityAuthority::LoadedClient(model_identity);
		builder
	}

	/// Start a session builder on an alternate model implementation.
	pub fn from_model(model: Arc<dyn AgentModel>, workspace_root: impl Into<PathBuf>) -> Self {
		let native_capabilities = Some(NativeModelCapabilities {
			system_prompt: model.supports_system_prompt(),
			tools: model.supports_tools(),
			reasoning_history: model.supports_reasoning_history(),
			thinking_toggle: model.supports_thinking_toggle(),
			default_thinking: model.default_thinking_enabled(),
		});
		Self {
			model,
			native_capabilities,
			workspace_root: workspace_root.into(),
			approval_policy: Arc::new(DenyAllApprovals),
			tools: Vec::new(),
			file_tools: Availability::Enabled,
			shell_tool: Availability::Enabled,
			shell_timeout_seconds: 30,
			shell_output_bytes: 128 * 1024,
			web_fetch: Availability::Disabled,
			web_response_bytes: 512 * 1024,
			datetime: Availability::Disabled,
			web_search_provider: None,
			web_search_provider_identity: None,
			history: Vec::new(),
			system_prompt: None,
			model_identity: None,
			model_identity_authority: ModelIdentityAuthority::AlternateModel,
			generation_options: GenerationOptions::default(),
			max_model_rounds: DEFAULT_MAX_MODEL_ROUNDS,
			max_tool_output_bytes: MAX_TOOL_OUTPUT_BYTES,
		}
	}

	/// Replace the default deny policy.
	#[must_use]
	pub fn approval_policy(mut self, policy: Arc<dyn ApprovalPolicy>) -> Self {
		self.approval_policy = policy;
		self
	}

	/// Bind an exact model/checkpoint identity for durable replay.
	///
	/// In-memory sessions may omit this. [`crate::memory::DurableAgentSession`]
	/// requires it so a resumed Session cannot silently switch models. Builders
	/// created with [`Self::new`] already carry the loaded client's immutable
	/// identity and reject a different override; this setter is intended for
	/// alternate [`AgentModel`] implementations supplied through
	/// [`Self::from_model`].
	#[must_use]
	pub fn model_identity(mut self, identity: ModelSnapshotId) -> Self {
		self.model_identity = Some(identity);
		self
	}

	/// Register one additional tool.
	#[must_use]
	pub fn tool(mut self, tool: Arc<dyn AgentTool>) -> Self {
		self.tools.push(ConfiguredTool::inferred(tool));
		self
	}

	/// Register one tool with an explicit executable identity.
	///
	/// Use a code/config digest or independently versioned implementation ID
	/// when the tool can change without the concrete Rust type or Emelex
	/// package version changing.
	#[must_use]
	pub fn tool_with_identity(
		mut self,
		tool: Arc<dyn AgentTool>,
		identity: impl Into<String>,
	) -> Self {
		self.tools.push(ConfiguredTool {
			tool,
			implementation_identity: Some(identity.into()),
		});
		self
	}

	/// Enable or disable the seven built-in workspace tools.
	#[must_use]
	pub const fn include_workspace_tools(mut self, include: bool) -> Self {
		self.file_tools = Availability::from_bool(include);
		self.shell_tool = Availability::from_bool(include);
		self
	}

	/// Enable or disable descriptor-anchored file tools independently.
	#[must_use]
	pub const fn include_file_tools(mut self, include: bool) -> Self {
		self.file_tools = Availability::from_bool(include);
		self
	}

	/// Enable or disable host shell independently.
	#[must_use]
	pub const fn include_shell_tool(mut self, include: bool) -> Self {
		self.shell_tool = Availability::from_bool(include);
		self
	}

	/// Set authoritative maximum shell duration per invocation.
	#[must_use]
	pub const fn shell_timeout_seconds(mut self, seconds: u64) -> Self {
		self.shell_timeout_seconds = seconds;
		self
	}

	/// Set authoritative combined shell stdout/stderr capture ceiling.
	#[must_use]
	pub const fn shell_output_bytes(mut self, bytes: usize) -> Self {
		self.shell_output_bytes = bytes;
		self
	}

	/// Enable or disable bounded `web_fetch`.
	///
	/// Disabled by default so building a session never silently adds network
	/// capability.
	#[must_use]
	pub const fn include_web_fetch(mut self, include: bool) -> Self {
		self.web_fetch = Availability::from_bool(include);
		self
	}

	/// Set authoritative maximum `web_fetch` response output.
	#[must_use]
	pub const fn web_response_bytes(mut self, bytes: usize) -> Self {
		self.web_response_bytes = bytes;
		self
	}

	/// Enable or disable the local `datetime` tool.
	#[must_use]
	pub const fn include_datetime(mut self, include: bool) -> Self {
		self.datetime = Availability::from_bool(include);
		self
	}

	/// Install generic `web_search` backed by an explicit provider.
	///
	/// Emelex does not select or contact a search vendor implicitly.
	#[must_use]
	pub fn web_search_provider(mut self, provider: Arc<dyn WebSearchProvider>) -> Self {
		self.web_search_provider = Some(provider);
		self.web_search_provider_identity = None;
		self
	}

	/// Install generic `web_search` with an exact backend/config identity.
	#[must_use]
	pub fn web_search_provider_with_identity(
		mut self,
		provider: Arc<dyn WebSearchProvider>,
		identity: impl Into<String>,
	) -> Self {
		self.web_search_provider = Some(provider);
		self.web_search_provider_identity = Some(identity.into());
		self
	}

	/// Resume from complete, ordered history.
	#[must_use]
	pub fn history(mut self, history: Vec<Message>) -> Self {
		self.history = history;
		self
	}

	/// Insert one system instruction before existing history.
	#[must_use]
	pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
		self.system_prompt = Some(prompt.into());
		self
	}

	/// Set option overrides applied to every model round in the session.
	///
	/// Per-turn options supplied to `run_*_with_options` take precedence for
	/// fields they set.
	#[must_use]
	pub const fn generation_options(mut self, options: GenerationOptions) -> Self {
		self.generation_options = options;
		self
	}

	/// Bound model rounds consumed by one user turn.
	#[must_use]
	pub const fn max_model_rounds(mut self, rounds: usize) -> Self {
		self.max_model_rounds = rounds;
		self
	}

	/// Bound one tool result before it enters model history.
	#[must_use]
	pub const fn max_tool_output_bytes(mut self, bytes: usize) -> Self {
		self.max_tool_output_bytes = bytes;
		self
	}

	/// Resolve and validate the durable authority boundary without consuming
	/// this builder.
	///
	/// The returned snapshot includes model identity, trusted prompt,
	/// generation policy, declarations, executable tool identities, workspace,
	/// and configurable ceilings. It excludes model implementation, history,
	/// and one-shot approval decisions. Calling this method does not prevent
	/// further builder configuration or a later [`Self::build`] call.
	///
	/// # Errors
	///
	/// Returns configuration, tool-schema, or workspace errors.
	pub fn authority_snapshot(&self) -> Result<AgentAuthoritySnapshot, AgentError> {
		self.resolve_authority().map(|resolved| resolved.snapshot)
	}

	/// Validate history, tools, workspace, and limits.
	///
	/// # Errors
	///
	/// Returns configuration, history, tool-schema, or workspace errors.
	pub fn build(mut self) -> Result<AgentSession, AgentError> {
		let resolved = self.resolve_authority()?;
		let ResolvedAuthority {
			snapshot: authority,
			workspace,
			tools,
		} = resolved;
		if let Some(prompt) = self.system_prompt.take() {
			if prompt.len() > MAX_SYSTEM_PROMPT_BYTES {
				return Err(AgentError::Configuration(format!(
					"system prompt exceeds {MAX_SYSTEM_PROMPT_BYTES} bytes"
				)));
			}
			if prompt.trim().is_empty() {
				return Err(AgentError::Configuration(
					"system prompt cannot be empty".to_string(),
				));
			}
			if self
				.history
				.first()
				.is_some_and(|message| message.role == Role::System)
			{
				return Err(AgentError::Configuration(
					"system_prompt conflicts with the system message already present in history"
						.to_string(),
				));
			}
			self.history.insert(0, Message::system(prompt));
		}
		let issued_tool_ids = validate_history(&self.history)?;
		let definitions = tools
			.values()
			.map(|registered| registered.definition.clone())
			.collect::<Vec<_>>();
		crate::generation::validate_request_shape(&self.history, &definitions).map_err(
			|error| {
				AgentError::Configuration(format!(
					"resumed history is invalid or incompatible with current tool declarations: \
					 {error}"
				))
			},
		)?;
		let enabled_tools = tools.keys().cloned().collect();
		Ok(AgentSession {
			model: self.model,
			native_capabilities: self.native_capabilities,
			workspace,
			approval_policy: self.approval_policy,
			tools,
			enabled_tools,
			history: self.history,
			issued_tool_ids,
			authority,
			generation_options: self.generation_options,
		})
	}

	#[expect(
		clippy::too_many_lines,
		reason = "authority validation stays contiguous so every immutable field is audited together"
	)]
	fn resolve_authority(&self) -> Result<ResolvedAuthority, AgentError> {
		if let ModelIdentityAuthority::LoadedClient(identity) = &self.model_identity_authority
			&& self.model_identity.as_ref() != identity.as_ref()
		{
			return Err(AgentError::Configuration(
				"model_identity cannot override the loaded Client snapshot identity".to_string(),
			));
		}
		if !(1..=MAX_AGENT_MODEL_ROUNDS).contains(&self.max_model_rounds) {
			return Err(AgentError::Configuration(format!(
				"max_model_rounds must be in 1..={MAX_AGENT_MODEL_ROUNDS}"
			)));
		}
		if self.max_tool_output_bytes == 0 || self.max_tool_output_bytes > MAX_TOOL_OUTPUT_BYTES {
			return Err(AgentError::Configuration(format!(
				"max_tool_output_bytes must be in 1..={MAX_TOOL_OUTPUT_BYTES}"
			)));
		}
		if !(1..=MAX_SHELL_TIMEOUT_SECONDS).contains(&self.shell_timeout_seconds) {
			return Err(AgentError::Configuration(format!(
				"shell_timeout_seconds must be in 1..={MAX_SHELL_TIMEOUT_SECONDS}"
			)));
		}
		if !(1..=MAX_SHELL_OUTPUT_BYTES).contains(&self.shell_output_bytes) {
			return Err(AgentError::Configuration(format!(
				"shell_output_bytes must be in 1..={MAX_SHELL_OUTPUT_BYTES}"
			)));
		}
		if !(1..=MAX_WEB_RESPONSE_BYTES).contains(&self.web_response_bytes) {
			return Err(AgentError::Configuration(format!(
				"web_response_bytes must be in 1..={MAX_WEB_RESPONSE_BYTES}"
			)));
		}
		validate_native_generation_policy(
			self.native_capabilities,
			self.generation_options,
			self.history
				.iter()
				.any(|message| message.reasoning.is_some()),
		)?;
		if self
			.native_capabilities
			.is_some_and(|capabilities| capabilities.system_prompt == Some(false))
			&& (self.system_prompt.is_some()
				|| self
					.history
					.iter()
					.any(|message| message.role == Role::System))
		{
			return Err(AgentError::Configuration(
				"loaded model does not preserve system prompts".to_string(),
			));
		}
		let workspace = Arc::new(workspace::WorkspaceRoot::open(&self.workspace_root)?);
		let mut resolved_tools = self.tools.clone();
		if self.file_tools.is_enabled() {
			resolved_tools.extend(file_tools().into_iter().map(ConfiguredTool::inferred));
		}
		if self.shell_tool.is_enabled() {
			resolved_tools.push(ConfiguredTool::inferred(shell_tool(
				self.shell_timeout_seconds,
				self.shell_output_bytes,
			)?));
		}
		if self.web_fetch.is_enabled() {
			resolved_tools.push(ConfiguredTool::inferred(web::web_fetch_tool_with_limit(
				self.web_response_bytes,
			)?));
		}
		if self.datetime.is_enabled() {
			resolved_tools.push(ConfiguredTool::inferred(datetime_tool()));
		}
		if let Some(provider) = &self.web_search_provider {
			let provider_identity = self
				.web_search_provider_identity
				.clone()
				.unwrap_or_else(|| provider.implementation_identity());
			resolved_tools.push(ConfiguredTool {
				tool: web_search_tool(Arc::clone(provider)),
				implementation_identity: Some(format!(
					"emelex-web-search@protocol-1;provider:{provider_identity}"
				)),
			});
		}
		if self
			.native_capabilities
			.is_some_and(|capabilities| capabilities.tools == Some(false))
			&& (!resolved_tools.is_empty()
				|| self.history.iter().any(|message| {
					message.role == Role::Tool
						|| message.tool_call_id.is_some()
						|| !message.tool_calls.is_empty()
				})) {
			return Err(AgentError::Configuration(
				"loaded model does not preserve tool declarations and complete tool rounds"
					.to_string(),
			));
		}
		if resolved_tools.len() > MAX_TOOLS {
			return Err(AgentError::Configuration(format!(
				"agent cannot advertise more than {MAX_TOOLS} tools"
			)));
		}
		let definitions = resolved_tools
			.iter()
			.map(|configured| configured.tool.definition())
			.collect::<Vec<_>>();
		for definition in &definitions {
			validate_tool_definition(definition)?;
		}
		crate::generation::validate_request_shape(&[], &definitions).map_err(|error| {
			AgentError::Configuration(format!(
				"advertised tools exceed generation request limits: {error}"
			))
		})?;
		let mut tools = BTreeMap::new();
		let mut tool_implementations = BTreeMap::new();
		let mut tool_cancellation_policies = BTreeMap::new();
		for (definition, configured) in definitions.into_iter().zip(resolved_tools) {
			let name = definition.name.clone();
			let implementation_identity = configured
				.implementation_identity
				.unwrap_or_else(|| configured.tool.implementation_identity());
			if implementation_identity.trim().is_empty()
				|| implementation_identity.len() > MAX_TOOL_IMPLEMENTATION_IDENTITY_BYTES
				|| implementation_identity.chars().any(char::is_control)
			{
				return Err(AgentError::Configuration(format!(
					"tool {name:?} implementation identity must contain \
					 1..={MAX_TOOL_IMPLEMENTATION_IDENTITY_BYTES} bytes without control \
					 characters"
				)));
			}
			let cancellation_policy = configured.tool.cancellation_policy();
			if tools
				.insert(
					name.clone(),
					RegisteredTool {
						definition,
						tool: configured.tool,
						cancellation_policy,
					},
				)
				.is_some()
			{
				return Err(AgentError::Configuration(format!(
					"duplicate tool name {name:?}"
				)));
			}
			tool_implementations.insert(name.clone(), implementation_identity);
			tool_cancellation_policies.insert(name, cancellation_policy);
		}
		let (workspace_device, workspace_inode) = workspace.identity();
		let mut enabled_capabilities = BTreeSet::new();
		if self.file_tools.is_enabled() {
			enabled_capabilities.insert(AgentBuiltinCapability::FileTools);
		}
		if self.shell_tool.is_enabled() {
			enabled_capabilities.insert(AgentBuiltinCapability::ShellTool);
		}
		if self.web_fetch.is_enabled() {
			enabled_capabilities.insert(AgentBuiltinCapability::WebFetch);
		}
		if self.datetime.is_enabled() {
			enabled_capabilities.insert(AgentBuiltinCapability::Datetime);
		}
		if self.web_search_provider.is_some() {
			enabled_capabilities.insert(AgentBuiltinCapability::WebSearch);
		}
		let snapshot = AgentAuthoritySnapshot {
			schema_version: AGENT_AUTHORITY_SNAPSHOT_SCHEMA_VERSION,
			workspace_root: workspace.path().to_path_buf(),
			workspace_device,
			workspace_inode,
			model_identity: self.model_identity.clone(),
			system_prompt: self.system_prompt.clone(),
			generation_options: self.generation_options,
			tools: tools
				.values()
				.map(|registered| registered.definition.clone())
				.collect(),
			tool_implementations,
			tool_cancellation_policies,
			enabled_capabilities,
			shell_timeout_seconds: self.shell_timeout_seconds,
			shell_output_bytes: self.shell_output_bytes,
			web_response_bytes: self.web_response_bytes,
			max_model_rounds: self.max_model_rounds,
			max_tool_output_bytes: self.max_tool_output_bytes,
		};
		Ok(ResolvedAuthority {
			snapshot,
			workspace,
			tools,
		})
	}
}

/// In-memory ordered conversation and native agent loop.
pub struct AgentSession {
	model: Arc<dyn AgentModel>,
	native_capabilities: Option<NativeModelCapabilities>,
	workspace: Arc<workspace::WorkspaceRoot>,
	approval_policy: Arc<dyn ApprovalPolicy>,
	tools: BTreeMap<String, RegisteredTool>,
	enabled_tools: BTreeSet<String>,
	history: Vec<Message>,
	issued_tool_ids: BTreeSet<String>,
	authority: AgentAuthoritySnapshot,
	generation_options: GenerationOptions,
}

struct ToolBatchOutcome {
	results: Vec<Message>,
	terminal_error: Option<AgentError>,
	checkpoint_required: bool,
}

struct ToolCallFailure {
	error: AgentError,
	effect_possible: bool,
	checkpointed_result: Option<Message>,
}

struct ToolCallOutcome {
	message: Message,
	terminal_error: Option<AgentError>,
}

struct ToolCompletionFailure {
	error: AgentError,
	checkpointed_result: Option<Message>,
}

impl AgentSession {
	/// Start building a session on one loaded client and workspace.
	pub fn builder(client: Client, workspace_root: impl Into<PathBuf>) -> AgentSessionBuilder {
		AgentSessionBuilder::new(client, workspace_root)
	}

	/// Complete ordered history.
	pub fn history(&self) -> &[Message] {
		&self.history
	}

	/// Current history length, suitable as a stable in-memory delta cursor.
	pub const fn history_cursor(&self) -> usize {
		self.history.len()
	}

	/// Ordered history committed at or after `cursor`.
	///
	/// Returns `None` when the cursor came from another or older session
	/// snapshot and exceeds this history.
	pub fn history_since(&self, cursor: usize) -> Option<&[Message]> {
		self.history.get(cursor..)
	}

	/// Canonical workspace root used by built-in tools.
	pub fn workspace_root(&self) -> &Path {
		self.workspace.path()
	}

	/// Exact authority resolved by the successful build.
	///
	/// Durable adapters can compare this with preflight metadata to detect
	/// workspace or tool-definition drift during construction.
	pub const fn authority_snapshot(&self) -> &AgentAuthoritySnapshot {
		&self.authority
	}

	/// Tool definitions inside this Session's immutable authority boundary.
	pub fn available_tools(&self) -> impl ExactSizeIterator<Item = &ToolDefinition> {
		self.tools.values().map(|registered| &registered.definition)
	}

	/// Tool names currently enabled for new calls.
	pub const fn enabled_tools(&self) -> &BTreeSet<String> {
		&self.enabled_tools
	}

	/// Replace the active tool subset without changing durable authority.
	///
	/// Disabled tools referenced by existing complete history remain declared to
	/// the model for replay validity, but cannot execute. All requested names are
	/// validated before the current subset changes.
	///
	/// # Errors
	///
	/// Returns [`AgentError::ToolUnavailable`] when any requested name is outside
	/// this Session's immutable authority.
	pub fn set_enabled_tools(&mut self, enabled_tools: BTreeSet<String>) -> Result<(), AgentError> {
		if let Some(tool_name) = enabled_tools
			.iter()
			.find(|tool_name| !self.tools.contains_key(tool_name.as_str()))
		{
			return Err(AgentError::ToolUnavailable {
				tool_name: tool_name.clone(),
			});
		}
		self.enabled_tools = enabled_tools;
		Ok(())
	}

	/// Run one text-only user turn.
	///
	/// # Errors
	///
	/// Returns invalid-input or [`Self::run_message`] failures.
	pub async fn run_turn<F>(
		&mut self,
		input: impl Into<String>,
		cancellation: &AgentCancellation,
		emit: F,
	) -> Result<AgentTurn, AgentError>
	where
		F: FnMut(AgentEvent),
	{
		self.run_turn_with_options(input, GenerationOptions::default(), cancellation, emit)
			.await
	}

	/// Run one text-only user turn with per-turn generation overrides.
	///
	/// # Errors
	///
	/// Returns the same failures as [`Self::run_turn`].
	pub async fn run_turn_with_options<F>(
		&mut self,
		input: impl Into<String>,
		options: GenerationOptions,
		cancellation: &AgentCancellation,
		emit: F,
	) -> Result<AgentTurn, AgentError>
	where
		F: FnMut(AgentEvent),
	{
		let input = input.into();
		if input.len() > MAX_USER_CONTENT_BYTES {
			return Err(AgentError::Configuration(format!(
				"turn input exceeds {MAX_USER_CONTENT_BYTES} bytes"
			)));
		}
		if input.trim().is_empty() {
			return Err(AgentError::Configuration(
				"turn input cannot be empty".to_string(),
			));
		}
		self.run_message_core(
			Message::user(input),
			options,
			cancellation,
			InfallibleEmitter(emit),
			NoopCheckpointEmitter,
		)
		.await
	}

	/// Run one text-only turn whose event consumer can abort the turn.
	///
	/// # Errors
	///
	/// Returns [`AgentError::EventSink`] when `emit` fails. Failure before any
	/// tool invocation leaves turn state unchanged. Once a tool may have
	/// produced a host side effect, a structurally complete assistant/tool
	/// batch is checkpointed so retries cannot silently repeat it.
	pub async fn try_run_turn<F, E>(
		&mut self,
		input: impl Into<String>,
		cancellation: &AgentCancellation,
		emit: F,
	) -> Result<AgentTurn, AgentError>
	where
		F: FnMut(AgentEvent) -> Result<(), E>,
		E: fmt::Display,
	{
		self.try_run_turn_with_options(input, GenerationOptions::default(), cancellation, emit)
			.await
	}

	/// Run one text-only turn with generation overrides and a fallible event
	/// consumer.
	///
	/// # Errors
	///
	/// Returns the same failures as [`Self::try_run_turn`].
	pub async fn try_run_turn_with_options<F, E>(
		&mut self,
		input: impl Into<String>,
		options: GenerationOptions,
		cancellation: &AgentCancellation,
		emit: F,
	) -> Result<AgentTurn, AgentError>
	where
		F: FnMut(AgentEvent) -> Result<(), E>,
		E: fmt::Display,
	{
		let input = input.into();
		if input.len() > MAX_USER_CONTENT_BYTES {
			return Err(AgentError::Configuration(format!(
				"turn input exceeds {MAX_USER_CONTENT_BYTES} bytes"
			)));
		}
		if input.trim().is_empty() {
			return Err(AgentError::Configuration(
				"turn input cannot be empty".to_string(),
			));
		}
		self.run_message_core(
			Message::user(input),
			options,
			cancellation,
			FallibleEmitter(emit),
			NoopCheckpointEmitter,
		)
		.await
	}

	/// Run one text or multimodal user message, streaming lifecycle events.
	///
	/// The callback is invoked synchronously in event order. Tool-call batches
	/// are committed to history only after every matching result exists, so a
	/// cancellation or fatal tool error cannot leave unresolved durable state.
	///
	/// # Errors
	///
	/// Returns cancellation, generation, protocol, approval-independent tool,
	/// workspace, or model-round-limit failures.
	pub async fn run_message<F>(
		&mut self,
		input: Message,
		cancellation: &AgentCancellation,
		emit: F,
	) -> Result<AgentTurn, AgentError>
	where
		F: FnMut(AgentEvent),
	{
		self.run_message_with_options(input, GenerationOptions::default(), cancellation, emit)
			.await
	}

	/// Run one user message with per-turn generation overrides.
	///
	/// # Errors
	///
	/// Returns the same failures as [`Self::run_message`].
	pub async fn run_message_with_options<F>(
		&mut self,
		input: Message,
		options: GenerationOptions,
		cancellation: &AgentCancellation,
		emit: F,
	) -> Result<AgentTurn, AgentError>
	where
		F: FnMut(AgentEvent),
	{
		self.run_message_core(
			input,
			options,
			cancellation,
			InfallibleEmitter(emit),
			NoopCheckpointEmitter,
		)
		.await
	}

	/// Run one user message whose event consumer can abort the turn.
	///
	/// # Errors
	///
	/// Returns [`AgentError::EventSink`] when `emit` fails. Failure before any
	/// tool invocation leaves turn state unchanged. Once a tool may have
	/// produced a host side effect, a structurally complete assistant/tool
	/// batch is checkpointed so retries cannot silently repeat it.
	pub async fn try_run_message<F, E>(
		&mut self,
		input: Message,
		cancellation: &AgentCancellation,
		emit: F,
	) -> Result<AgentTurn, AgentError>
	where
		F: FnMut(AgentEvent) -> Result<(), E>,
		E: fmt::Display,
	{
		self.try_run_message_with_options(input, GenerationOptions::default(), cancellation, emit)
			.await
	}

	/// Run one user message with generation overrides and a fallible event
	/// consumer.
	///
	/// # Errors
	///
	/// Returns the same failures as [`Self::try_run_message`].
	pub async fn try_run_message_with_options<F, E>(
		&mut self,
		input: Message,
		options: GenerationOptions,
		cancellation: &AgentCancellation,
		emit: F,
	) -> Result<AgentTurn, AgentError>
	where
		F: FnMut(AgentEvent) -> Result<(), E>,
		E: fmt::Display,
	{
		self.run_message_core(
			input,
			options,
			cancellation,
			FallibleEmitter(emit),
			NoopCheckpointEmitter,
		)
		.await
	}

	pub(crate) async fn try_run_message_with_checkpoint<F, E, P>(
		&mut self,
		input: Message,
		cancellation: &AgentCancellation,
		emit: F,
		checkpoint: P,
	) -> Result<AgentTurn, AgentError>
	where
		F: FnMut(AgentEvent) -> Result<(), E>,
		E: fmt::Display,
		P: CheckpointEmitter,
	{
		self.run_message_core(
			input,
			GenerationOptions::default(),
			cancellation,
			FallibleEmitter(emit),
			checkpoint,
		)
		.await
	}

	#[expect(
		clippy::too_many_lines,
		reason = "one turn state machine keeps persistence, tool, and model ordering auditable"
	)]
	async fn run_message_core<S, P>(
		&mut self,
		input: Message,
		options: GenerationOptions,
		cancellation: &AgentCancellation,
		mut emit: S,
		mut persist: P,
	) -> Result<AgentTurn, AgentError>
	where
		S: EventEmitter,
		P: CheckpointEmitter,
	{
		validate_user_message(&input)?;
		let effective_options = merge_generation_options(self.generation_options, options);
		validate_native_generation_policy(
			self.native_capabilities,
			effective_options,
			self.history
				.iter()
				.any(|message| message.reasoning.is_some()),
		)?;
		let turn_id = Uuid::now_v7();
		persist
			.checkpoint(AgentCheckpoint::PreflightInput {
				turn_id,
				message: &input,
			})
			.await?;
		if cancellation.is_cancelled() {
			emit.emit(AgentEvent::Cancelled { turn_id })?;
			return Err(AgentError::Cancelled);
		}
		emit.emit(AgentEvent::TurnStarted { turn_id })?;
		let mut pending_messages = vec![input];
		let mut turn_messages = Vec::new();
		let mut issued_this_turn = BTreeSet::new();
		let mut aggregate_usage = Usage::default();
		let mut model_output_bytes = 0_usize;
		let mut tool_argument_bytes = 0_usize;
		let mut tool_output_bytes = 0_usize;
		macro_rules! fail_turn {
			($error:expr) => {{
				let error = $error;
				if !matches!(error, AgentError::EventSink(_)) {
					emit_terminal_error(turn_id, &error, &mut emit)?;
				}
				return Err(error);
			}};
		}

		for round in 1..=self.authority.max_model_rounds {
			emit.emit(AgentEvent::ModelStarted { turn_id, round })?;
			let request = match self.generation_request(&pending_messages, effective_options) {
				Ok(request) => request,
				Err(error) => fail_turn!(error),
			};
			let response = match self
				.stream_model_round(
					request,
					turn_id,
					round,
					cancellation,
					&mut issued_this_turn,
					&mut tool_argument_bytes,
					&mut model_output_bytes,
					&mut emit,
				)
				.await
			{
				Ok(response) => response,
				Err(error) => fail_turn!(error),
			};
			add_usage(&mut aggregate_usage, response.usage);
			emit.emit(AgentEvent::ModelCompleted {
				turn_id,
				round,
				finish_reason: response.finish_reason,
				usage: response.usage,
			})?;
			if response.finish_reason == FinishReason::Aborted {
				fail_turn!(AgentError::ModelProtocol(
					"model returned an aborted terminal response".to_string(),
				));
			}
			let preserve_reasoning = self
				.native_capabilities
				.is_none_or(|capabilities| capabilities.reasoning_history != Some(false));
			let assistant = assistant_message(&response, preserve_reasoning);
			if response.tool_calls.is_empty() {
				if response.finish_reason == FinishReason::ToolCalls {
					fail_turn!(AgentError::ModelProtocol(
						"tool_calls finish reason carried no tool calls".to_string(),
					));
				}
				pending_messages.push(assistant);
				turn_messages.extend(pending_messages.iter().cloned());
				let turn = AgentTurn {
					id: turn_id,
					response,
					messages: turn_messages.clone(),
					model_rounds: round,
					usage: aggregate_usage,
				};
				let completed = AgentEvent::TurnCompleted {
					turn_id,
					model_rounds: round,
					usage: aggregate_usage,
				};
				persist
					.checkpoint(AgentCheckpoint::Messages {
						turn_id,
						messages: &pending_messages,
						terminal_event: Some(&completed),
					})
					.await?;
				self.history.append(&mut pending_messages);
				self.issued_tool_ids.extend(issued_this_turn);
				if let Err(error) = emit.emit(completed) {
					return Err(match error {
						AgentError::EventSink(message) => {
							AgentError::EventSinkAfterCommit { turn_id, message }
						}
						other => other,
					});
				}
				return Ok(turn);
			}
			if response.finish_reason != FinishReason::ToolCalls {
				fail_turn!(AgentError::ModelProtocol(
					"response carried tool calls without tool_calls finish reason".to_string(),
				));
			}
			if round == self.authority.max_model_rounds {
				fail_turn!(AgentError::MaxModelRounds {
					limit: self.authority.max_model_rounds,
				});
			}
			let mut planned_messages = pending_messages.clone();
			planned_messages.push(assistant.clone());
			if let Err(error) = persist
				.checkpoint(AgentCheckpoint::ToolBatchStarted {
					turn_id,
					messages: &planned_messages,
				})
				.await
			{
				fail_turn!(error);
			}
			let tool_batch = self
				.execute_tool_batch(
					&response.tool_calls,
					cancellation,
					&mut tool_output_bytes,
					&mut emit,
					&mut persist,
					turn_id,
				)
				.await;
			if !tool_batch.checkpoint_required {
				fail_turn!(tool_batch.terminal_error.unwrap_or_else(|| {
					AgentError::ModelProtocol(
						"tool batch produced neither a checkpoint nor an error".to_string(),
					)
				}));
			}
			pending_messages.push(assistant);
			pending_messages.extend(tool_batch.results);
			if let Err(error) = persist
				.checkpoint(AgentCheckpoint::Messages {
					turn_id,
					messages: &pending_messages,
					terminal_event: None,
				})
				.await
			{
				fail_turn!(error);
			}
			turn_messages.extend(pending_messages.iter().cloned());
			self.history.append(&mut pending_messages);
			self.issued_tool_ids
				.extend(issued_this_turn.iter().cloned());
			debug_assert!(validate_history(&self.history).is_ok());
			if let Some(error) = tool_batch.terminal_error {
				fail_turn!(error);
			}
		}
		fail_turn!(AgentError::MaxModelRounds {
			limit: self.authority.max_model_rounds,
		})
	}

	fn generation_request(
		&self,
		turn_messages: &[Message],
		effective_options: GenerationOptions,
	) -> Result<GenerationRequest, AgentError> {
		let replay_tools = self
			.history
			.iter()
			.chain(turn_messages)
			.flat_map(|message| message.tool_calls.iter().map(|call| call.name.as_str()))
			.collect::<BTreeSet<_>>();
		let mut messages = Vec::with_capacity(self.history.len() + turn_messages.len());
		messages.extend(self.history.iter().cloned());
		messages.extend(turn_messages.iter().cloned());
		let messages = coalesce_adjacent_user_messages(messages);
		let request = GenerationRequest {
			messages,
			tools: self
				.tools
				.iter()
				.filter(|(name, _)| {
					self.enabled_tools.contains(name.as_str())
						|| replay_tools.contains(name.as_str())
				})
				.map(|(_, registered)| registered.definition.clone())
				.collect(),
			options: effective_options,
		};
		request
			.options
			.validate_shape()
			.map_err(|error| AgentError::Configuration(error.to_string()))?;
		crate::generation::validate_request_shape(&request.messages, &request.tools)?;
		Ok(request)
	}

	#[expect(
		clippy::too_many_arguments,
		clippy::too_many_lines,
		reason = "single stream state machine keeps lossless terminal reconciliation auditable"
	)]
	async fn stream_model_round<S>(
		&self,
		request: GenerationRequest,
		turn_id: Uuid,
		round: usize,
		cancellation: &AgentCancellation,
		issued_this_turn: &mut BTreeSet<String>,
		observed_argument_bytes: &mut usize,
		observed_model_output_bytes: &mut usize,
		emit: &mut S,
	) -> Result<GenerationResponse, AgentError>
	where
		S: EventEmitter,
	{
		let mut stream = self.model.stream(request)?;
		let result = async {
			let mut observed = Vec::new();
			let mut observed_source = Vec::new();
			let mut source_ids = BTreeSet::new();
			let mut streamed_text = String::new();
			let mut streamed_reasoning = String::new();
			let mut completed = None;
			loop {
				let item = tokio::select! {
					biased;
					() = cancellation.cancelled() => {
						return Err(AgentError::Cancelled);
					}
					item = stream.next() => item,
				};
				let Some(item) = item else {
					break;
				};
				if completed.is_some() {
					return Err(AgentError::ModelProtocol(
						"generation emitted data after its terminal response".to_string(),
					));
				}
				match item? {
					GenerationEvent::Text(text) => {
						append_model_delta(
							&mut streamed_text,
							&text,
							"answer text",
							observed_model_output_bytes,
						)?;
						emit.emit(AgentEvent::TextDelta {
							turn_id,
							round,
							text,
						})?;
					}
					GenerationEvent::Reasoning(text) => {
						append_model_delta(
							&mut streamed_reasoning,
							&text,
							"reasoning",
							observed_model_output_bytes,
						)?;
						emit.emit(AgentEvent::ReasoningDelta {
							turn_id,
							round,
							text,
						})?;
					}
					GenerationEvent::ToolCall(call) => {
						let (source_call, call) = self.reidentify_call(
							call,
							&mut source_ids,
							issued_this_turn,
							observed_argument_bytes,
						)?;
						observed_source.push(source_call);
						emit.emit(AgentEvent::ToolCall {
							turn_id,
							round,
							call: call.clone(),
						})?;
						observed.push(call);
					}
					GenerationEvent::Completed(response) => {
						if completed.is_some() {
							return Err(AgentError::ModelProtocol(
								"generation emitted more than one terminal response".to_string(),
							));
						}
						validate_terminal_response(&response)?;
						completed = Some(response);
					}
				}
			}
			let Some(mut response) = completed else {
				return Err(AgentError::ModelProtocol(
					"generation ended without a terminal response".to_string(),
				));
			};
			let suffix = response
				.text
				.strip_prefix(&streamed_text)
				.ok_or_else(|| {
					AgentError::ModelProtocol(
						"streamed answer text is not a prefix of terminal response".to_string(),
					)
				})?
				.to_string();
			if !suffix.is_empty() {
				append_model_delta(
					&mut streamed_text,
					&suffix,
					"answer text",
					observed_model_output_bytes,
				)?;
				emit.emit(AgentEvent::TextDelta {
					turn_id,
					round,
					text: suffix,
				})?;
			}
			debug_assert_eq!(streamed_text, response.text);
			let terminal_reasoning = response.reasoning.as_deref().unwrap_or_default();
			let reasoning_suffix = terminal_reasoning
				.strip_prefix(&streamed_reasoning)
				.ok_or_else(|| {
					AgentError::ModelProtocol(
						"streamed reasoning is not a prefix of terminal response".to_string(),
					)
				})?
				.to_string();
			if !reasoning_suffix.is_empty() {
				append_model_delta(
					&mut streamed_reasoning,
					&reasoning_suffix,
					"reasoning",
					observed_model_output_bytes,
				)?;
				emit.emit(AgentEvent::ReasoningDelta {
					turn_id,
					round,
					text: reasoning_suffix,
				})?;
			}
			debug_assert_eq!(streamed_reasoning, terminal_reasoning);
			if observed.is_empty() {
				for call in std::mem::take(&mut response.tool_calls) {
					let (_, call) = self.reidentify_call(
						call,
						&mut source_ids,
						issued_this_turn,
						observed_argument_bytes,
					)?;
					emit.emit(AgentEvent::ToolCall {
						turn_id,
						round,
						call: call.clone(),
					})?;
					observed.push(call);
				}
			} else if observed_source != response.tool_calls {
				return Err(AgentError::ModelProtocol(
					"streamed tool calls differ from terminal response".to_string(),
				));
			}
			response.tool_calls = observed;
			if response.tool_calls.is_empty() && response.text.trim().is_empty() {
				return Err(AgentError::ModelProtocol(
					"terminal response contained no answer text or tool calls".to_string(),
				));
			}
			Ok(response)
		}
		.await;

		match result {
			Ok(response) => Ok(response),
			Err(primary) => match stream.cancel_and_wait().await {
				Ok(()) => Err(primary),
				Err(cleanup) => Err(AgentError::GenerationCleanup {
					primary: Box::new(primary),
					cleanup,
				}),
			},
		}
	}

	fn reidentify_call(
		&self,
		call: ToolCall,
		source_ids: &mut BTreeSet<String>,
		issued_this_turn: &mut BTreeSet<String>,
		argument_bytes: &mut usize,
	) -> Result<(ToolCall, ToolCall), AgentError> {
		if call.id.len() > MAX_TOOL_CALL_ID_BYTES
			|| call.id.trim().is_empty()
			|| call.id.chars().any(char::is_control)
		{
			return Err(AgentError::ModelProtocol(format!(
				"model emitted an invalid tool-call ID (maximum {MAX_TOOL_CALL_ID_BYTES} bytes)"
			)));
		}
		if !valid_tool_name(&call.name) {
			return Err(AgentError::ModelProtocol(format!(
				"model emitted invalid tool name {:?}",
				call.name
			)));
		}
		let bytes = crate::generation::bounded_json_len(&call.arguments, MAX_TOOL_SCHEMA_BYTES)
			.ok_or_else(|| {
				AgentError::ModelProtocol(format!(
					"tool {:?} arguments exceed structural or {MAX_TOOL_SCHEMA_BYTES}-byte limits",
					call.name
				))
			})?;
		*argument_bytes = argument_bytes
			.checked_add(bytes)
			.ok_or_else(|| AgentError::ModelProtocol("tool argument size overflow".to_string()))?;
		if *argument_bytes > MAX_TOTAL_TOOL_ARGUMENT_BYTES {
			return Err(AgentError::ModelProtocol(format!(
				"streamed tool arguments exceed {MAX_TOTAL_TOOL_ARGUMENT_BYTES} aggregate bytes"
			)));
		}
		if !source_ids.insert(call.id.clone()) {
			return Err(AgentError::ModelProtocol(format!(
				"model reused tool-call ID {:?} within one round",
				call.id
			)));
		}
		if !call.arguments.is_object() {
			return Err(AgentError::ModelProtocol(format!(
				"tool {:?} arguments are not an object",
				call.name
			)));
		}
		if source_ids.len() > MAX_TOOL_CALLS_PER_ROUND {
			return Err(AgentError::ModelProtocol(format!(
				"model emitted more than {MAX_TOOL_CALLS_PER_ROUND} tool calls"
			)));
		}
		let source_call = call.clone();
		for _ in 0..8 {
			let id = Uuid::now_v7().to_string();
			if !self.issued_tool_ids.contains(&id) && issued_this_turn.insert(id.clone()) {
				return Ok((
					source_call,
					ToolCall {
						id,
						name: call.name,
						arguments: call.arguments,
					},
				));
			}
		}
		Err(AgentError::ModelProtocol(
			"could not allocate a unique UUIDv7 tool-call ID".to_string(),
		))
	}

	#[expect(
		clippy::too_many_lines,
		reason = "tool-batch checkpoint state machine stays contiguous for crash-audit review"
	)]
	async fn execute_tool_batch<S, P>(
		&self,
		calls: &[ToolCall],
		cancellation: &AgentCancellation,
		total_output_bytes: &mut usize,
		emit: &mut S,
		persist: &mut P,
		turn_id: Uuid,
	) -> ToolBatchOutcome
	where
		S: EventEmitter,
		P: CheckpointEmitter,
	{
		let mut results = Vec::with_capacity(calls.len());
		for (index, call) in calls.iter().enumerate() {
			match self
				.execute_tool_call(
					call,
					cancellation,
					total_output_bytes,
					emit,
					persist,
					turn_id,
				)
				.await
			{
				Ok(outcome) => {
					results.push(outcome.message);
					if let Some(error) = outcome.terminal_error {
						for remaining in &calls[index + 1..] {
							let (message, event) = synthetic_tool_completion(
								remaining,
								"tool not executed because an earlier call ended the batch",
							);
							if let Err(checkpoint) = persist
								.checkpoint(AgentCheckpoint::ToolInvocationCompleted {
									turn_id,
									message: &message,
									event: &event,
								})
								.await
							{
								return ToolBatchOutcome {
									results,
									terminal_error: Some(checkpoint),
									checkpoint_required: false,
								};
							}
							results.push(message);
						}
						return ToolBatchOutcome {
							results,
							terminal_error: Some(error),
							checkpoint_required: true,
						};
					}
				}
				Err(mut failure) => {
					if let Some(message) = failure.checkpointed_result.take() {
						results.push(message);
					} else if matches!(&failure.error, AgentError::CheckpointSink(_))
						|| (!failure.effect_possible && results.is_empty())
					{
						return ToolBatchOutcome {
							results,
							terminal_error: Some(failure.error),
							checkpoint_required: false,
						};
					} else {
						let (message, event) = synthetic_tool_completion(
							call,
							if failure.effect_possible {
								"tool execution failed or was interrupted; side effect may have \
								 occurred"
							} else {
								"tool was not executed because the harness failed before invocation"
							},
						);
						if let Err(checkpoint) = persist
							.checkpoint(AgentCheckpoint::ToolInvocationCompleted {
								turn_id,
								message: &message,
								event: &event,
							})
							.await
						{
							return ToolBatchOutcome {
								results,
								terminal_error: Some(checkpoint),
								checkpoint_required: false,
							};
						}
						results.push(message);
					}
					for remaining in &calls[index + 1..] {
						let (message, event) = synthetic_tool_completion(
							remaining,
							"tool not executed because an earlier call failed",
						);
						if let Err(checkpoint) = persist
							.checkpoint(AgentCheckpoint::ToolInvocationCompleted {
								turn_id,
								message: &message,
								event: &event,
							})
							.await
						{
							return ToolBatchOutcome {
								results,
								terminal_error: Some(checkpoint),
								checkpoint_required: false,
							};
						}
						results.push(message);
					}
					return ToolBatchOutcome {
						results,
						terminal_error: Some(failure.error),
						checkpoint_required: true,
					};
				}
			}
		}
		ToolBatchOutcome {
			results,
			terminal_error: None,
			checkpoint_required: !calls.is_empty(),
		}
	}

	#[expect(
		clippy::too_many_lines,
		reason = "one invocation keeps approval, execution, and checkpoint ordering explicit"
	)]
	async fn execute_tool_call<S, P>(
		&self,
		call: &ToolCall,
		cancellation: &AgentCancellation,
		total_output_bytes: &mut usize,
		emit: &mut S,
		persist: &mut P,
		turn_id: Uuid,
	) -> Result<ToolCallOutcome, ToolCallFailure>
	where
		S: EventEmitter,
		P: CheckpointEmitter,
	{
		let cancelled = || ToolCallFailure {
			error: AgentError::Cancelled,
			effect_possible: false,
			checkpointed_result: None,
		};
		if cancellation.is_cancelled() {
			return Err(cancelled());
		}
		let Some(registered) = self.tools.get(&call.name) else {
			return self
				.finalize_tool_output(
					call,
					ToolOutput::error(format!("unknown tool {:?}", call.name)),
					total_output_bytes,
					emit,
					persist,
					turn_id,
				)
				.await
				.map_err(|failure| ToolCallFailure {
					error: failure.error,
					effect_possible: false,
					checkpointed_result: failure.checkpointed_result,
				});
		};
		if !self.enabled_tools.contains(&call.name) {
			let unavailable = AgentError::ToolUnavailable {
				tool_name: call.name.clone(),
			};
			return self
				.finalize_tool_output(
					call,
					ToolOutput::error(unavailable.to_string()),
					total_output_bytes,
					emit,
					persist,
					turn_id,
				)
				.await
				.map_err(|failure| ToolCallFailure {
					error: failure.error,
					effect_possible: false,
					checkpointed_result: failure.checkpointed_result,
				});
		}
		if !tool_arguments_match(&registered.definition, &call.arguments) {
			return self
				.finalize_tool_output(
					call,
					ToolOutput::error(format!(
						"arguments for tool {:?} do not satisfy its JSON Schema",
						call.name
					)),
					total_output_bytes,
					emit,
					persist,
					turn_id,
				)
				.await
				.map_err(|failure| ToolCallFailure {
					error: failure.error,
					effect_possible: false,
					checkpointed_result: failure.checkpointed_result,
				});
		}
		let mut context = ToolContext {
			call_id: call.id.clone(),
			workspace: Arc::clone(&self.workspace),
			cancellation: cancellation.clone(),
			approved: false,
		};
		let approval_output = self
			.approve_tool(
				registered,
				call,
				&mut context,
				cancellation,
				emit,
				persist,
				turn_id,
			)
			.await
			.map_err(|error| ToolCallFailure {
				error,
				effect_possible: false,
				checkpointed_result: None,
			})?;
		if cancellation.is_cancelled() {
			return Err(cancelled());
		}
		if let Some(output) = approval_output {
			return self
				.finalize_tool_output(call, output, total_output_bytes, emit, persist, turn_id)
				.await
				.map_err(|failure| ToolCallFailure {
					error: failure.error,
					effect_possible: false,
					checkpointed_result: failure.checkpointed_result,
				});
		}
		emit.emit(AgentEvent::ToolStarted {
			call_id: call.id.clone(),
			tool_name: call.name.clone(),
		})
		.map_err(|error| ToolCallFailure {
			error,
			effect_possible: false,
			checkpointed_result: None,
		})?;
		persist
			.checkpoint(AgentCheckpoint::ToolInvocationStarted { turn_id, call })
			.await
			.map_err(|error| ToolCallFailure {
				error,
				effect_possible: false,
				checkpointed_result: None,
			})?;
		let outcome = match registered.cancellation_policy {
			ToolCancellationPolicy::Interruptible => tokio::select! {
				biased;
				() = cancellation.cancelled() => Err(ToolError::Cancelled),
				outcome = registered.tool.invoke(&context, call.arguments.clone()) => outcome,
			},
			ToolCancellationPolicy::FinishOnceStarted => {
				registered
					.tool
					.invoke(&context, call.arguments.clone())
					.await
			}
		};
		let (output, cancelled) = match outcome {
			Ok(output) => (output, cancellation.is_cancelled()),
			Err(ToolError::RespondToModel(message)) => {
				(ToolOutput::error(message), cancellation.is_cancelled())
			}
			Err(ToolError::Cancelled) => (
				ToolOutput::error("tool invocation cancelled before completion"),
				true,
			),
			Err(ToolError::Fatal(message)) => {
				return Err(ToolCallFailure {
					error: tool_fatal(call, message),
					effect_possible: true,
					checkpointed_result: None,
				});
			}
		};
		let mut outcome = self
			.finalize_tool_output(call, output, total_output_bytes, emit, persist, turn_id)
			.await
			.map_err(|failure| ToolCallFailure {
				error: failure.error,
				effect_possible: true,
				checkpointed_result: failure.checkpointed_result,
			})?;
		if cancelled {
			outcome.terminal_error = Some(AgentError::Cancelled);
		}
		Ok(outcome)
	}

	async fn approve_tool<S, P>(
		&self,
		registered: &RegisteredTool,
		call: &ToolCall,
		context: &mut ToolContext,
		cancellation: &AgentCancellation,
		emit: &mut S,
		persist: &mut P,
		turn_id: Uuid,
	) -> Result<Option<ToolOutput>, AgentError>
	where
		S: EventEmitter,
		P: CheckpointEmitter,
	{
		let ApprovalRequirement::Required { reason } = registered
			.tool
			.approval_requirement(context, &call.arguments)
		else {
			return Ok(None);
		};
		validate_approval_text("approval reason", &reason)
			.map_err(|message| tool_fatal(call, message))?;
		let approval_context = ApprovalContext {
			call_id: call.id.clone(),
			tool_name: call.name.clone(),
			arguments: call.arguments.clone(),
			workspace_root: self.workspace.path().to_path_buf(),
			workspace_device: self.authority.workspace_device,
			workspace_inode: self.authority.workspace_inode,
			reason,
		};
		emit.emit(AgentEvent::ApprovalRequested {
			context: approval_context.clone(),
		})?;
		persist
			.checkpoint(AgentCheckpoint::ToolAudit { turn_id })
			.await?;
		let decision = tokio::select! {
			biased;
			() = cancellation.cancelled() => return Err(AgentError::Cancelled),
			decision = self.approval_policy.decide(&approval_context) => decision,
		};
		if let ApprovalDecision::Deny { reason } = &decision {
			validate_approval_text("approval denial", reason)
				.map_err(|message| tool_fatal(call, message))?;
		}
		emit.emit(AgentEvent::ApprovalResolved {
			call_id: call.id.clone(),
			decision: decision.clone(),
		})?;
		persist
			.checkpoint(AgentCheckpoint::ToolAudit { turn_id })
			.await?;
		match decision {
			ApprovalDecision::AllowOnce => {
				context.approved = true;
				Ok(None)
			}
			ApprovalDecision::Deny { reason } => Ok(Some(ToolOutput::error(format!(
				"tool invocation denied: {reason}"
			)))),
		}
	}

	#[allow(
		clippy::result_large_err,
		reason = "private checkpoint failure preserves typed owned recovery evidence"
	)]
	async fn finalize_tool_output<S, P>(
		&self,
		call: &ToolCall,
		mut output: ToolOutput,
		total_output_bytes: &mut usize,
		emit: &mut S,
		persist: &mut P,
		turn_id: Uuid,
	) -> Result<ToolCallOutcome, ToolCompletionFailure>
	where
		S: EventEmitter,
		P: CheckpointEmitter,
	{
		if output.content.is_empty() {
			output.content = "(no output)".to_string();
		}
		let mut terminal_error = None;
		if output.content.len() > self.authority.max_tool_output_bytes {
			terminal_error = Some(tool_fatal(
				call,
				format!(
					"output exceeded {} bytes",
					self.authority.max_tool_output_bytes
				),
			));
			output = ToolOutput::error(
				"tool output omitted because its configured byte limit was exceeded; side effect \
				 may have occurred",
			);
		} else {
			let next = total_output_bytes
				.checked_add(output.content.len())
				.unwrap_or(usize::MAX);
			if next > MAX_TOTAL_TOOL_OUTPUT_BYTES {
				terminal_error = Some(tool_fatal(
					call,
					format!("aggregate tool output exceeds {MAX_TOTAL_TOOL_OUTPUT_BYTES} bytes"),
				));
				output = ToolOutput::error(
					"tool output omitted because the durable turn budget was exhausted; side \
					 effect may have occurred",
				);
			} else {
				*total_output_bytes = next;
			}
		}
		let message = Message::tool(&call.id, output.content.clone());
		let completed = AgentEvent::ToolCompleted {
			call_id: call.id.clone(),
			tool_name: call.name.clone(),
			output,
		};
		if let Err(error) = persist
			.checkpoint(AgentCheckpoint::ToolInvocationCompleted {
				turn_id,
				message: &message,
				event: &completed,
			})
			.await
		{
			return Err(ToolCompletionFailure {
				error,
				checkpointed_result: None,
			});
		}
		if let Err(error) = emit.emit(completed) {
			return Err(ToolCompletionFailure {
				error,
				checkpointed_result: Some(message),
			});
		}
		Ok(ToolCallOutcome {
			message,
			terminal_error,
		})
	}
}

fn synthetic_tool_completion(call: &ToolCall, content: &str) -> (Message, AgentEvent) {
	let output = ToolOutput::error(content);
	(
		Message::tool(&call.id, output.content.clone()),
		AgentEvent::ToolCompleted {
			call_id: call.id.clone(),
			tool_name: call.name.clone(),
			output,
		},
	)
}

fn coalesce_adjacent_user_messages(messages: Vec<Message>) -> Vec<Message> {
	let mut coalesced: Vec<Message> = Vec::with_capacity(messages.len());
	for mut message in messages {
		if message.role == Role::User
			&& let Some(previous) = coalesced.last_mut()
			&& previous.role == Role::User
		{
			if let (Some(Content::Text(left)), Some(Content::Text(right))) =
				(previous.content.last_mut(), message.content.first())
			{
				left.push_str("\n\n");
				left.push_str(right);
				message.content.remove(0);
			}
			previous.content.extend(message.content);
		} else {
			coalesced.push(message);
		}
	}
	coalesced
}

fn tool_fatal(call: &ToolCall, message: String) -> AgentError {
	AgentError::ToolFatal {
		tool_name: call.name.clone(),
		message: truncate_owned_utf8(message, MAX_APPROVAL_REASON_BYTES),
	}
}

fn truncate_owned_utf8(mut text: String, limit: usize) -> String {
	if text.len() <= limit {
		return text;
	}
	let suffix = "…";
	let mut boundary = limit.saturating_sub(suffix.len());
	while boundary > 0 && !text.is_char_boundary(boundary) {
		boundary -= 1;
	}
	text.truncate(boundary);
	text.push_str(suffix);
	text
}

fn assistant_message(response: &GenerationResponse, preserve_reasoning: bool) -> Message {
	let content = if response.text.is_empty() {
		Vec::new()
	} else {
		vec![Content::Text(response.text.clone())]
	};
	Message {
		role: Role::Assistant,
		content,
		tool_calls: response.tool_calls.clone(),
		tool_call_id: None,
		reasoning: if preserve_reasoning {
			response.reasoning.clone()
		} else {
			None
		},
	}
}

const fn merge_generation_options(
	session: GenerationOptions,
	turn: GenerationOptions,
) -> GenerationOptions {
	GenerationOptions {
		max_tokens: match turn.max_tokens {
			Some(value) => Some(value),
			None => session.max_tokens,
		},
		temperature: match turn.temperature {
			Some(value) => Some(value),
			None => session.temperature,
		},
		top_p: match turn.top_p {
			Some(value) => Some(value),
			None => session.top_p,
		},
		top_k: match turn.top_k {
			Some(value) => Some(value),
			None => session.top_k,
		},
		seed: match turn.seed {
			Some(value) => Some(value),
			None => session.seed,
		},
		thinking: match turn.thinking {
			Some(value) => Some(value),
			None => session.thinking,
		},
		speculative_tokens: match turn.speculative_tokens {
			Some(value) => Some(value),
			None => session.speculative_tokens,
		},
		reasoning_budget_tokens: match (turn.reasoning_budget_tokens, turn.thinking) {
			(Some(value), _) => Some(value),
			(None, Some(crate::config::ThinkingMode::Off | crate::config::ThinkingMode::Auto)) => {
				None
			}
			(None, _) => session.reasoning_budget_tokens,
		},
		prompt_cache: match turn.prompt_cache {
			Some(value) => Some(value),
			None => session.prompt_cache,
		},
	}
}

fn validate_native_generation_policy(
	capabilities: Option<NativeModelCapabilities>,
	options: GenerationOptions,
	history_has_reasoning: bool,
) -> Result<(), AgentError> {
	options
		.validate_shape()
		.map_err(|error| AgentError::Configuration(error.to_string()))?;
	let effective_thinking = match options.thinking {
		Some(crate::config::ThinkingMode::On) => Some(true),
		Some(crate::config::ThinkingMode::Off) => Some(false),
		Some(crate::config::ThinkingMode::Auto) | None => {
			capabilities.and_then(|capabilities| capabilities.default_thinking)
		}
	};
	if options.reasoning_budget_tokens.is_some() && effective_thinking != Some(true) {
		return Err(AgentError::Configuration(
			"reasoning_budget_tokens requires thinking to be enabled".to_string(),
		));
	}
	if effective_thinking == Some(true)
		&& capabilities.is_some_and(|capabilities| capabilities.thinking_toggle == Some(false))
	{
		return Err(AgentError::Configuration(
			"loaded model does not support an explicit thinking toggle".to_string(),
		));
	}
	if (effective_thinking == Some(true) || history_has_reasoning)
		&& capabilities.is_some_and(|capabilities| capabilities.reasoning_history == Some(false))
	{
		return Err(AgentError::Configuration(
			"loaded model does not preserve reasoning across agent rounds".to_string(),
		));
	}
	Ok(())
}

fn append_model_delta(
	output: &mut String,
	delta: &str,
	field: &str,
	aggregate_bytes: &mut usize,
) -> Result<(), AgentError> {
	let length = output
		.len()
		.checked_add(delta.len())
		.ok_or_else(|| AgentError::ModelProtocol(format!("{field} size overflow")))?;
	if length > MAX_MODEL_OUTPUT_BYTES {
		return Err(AgentError::ModelProtocol(format!(
			"streamed {field} exceeds {MAX_MODEL_OUTPUT_BYTES} bytes"
		)));
	}
	let aggregate = aggregate_bytes
		.checked_add(delta.len())
		.ok_or_else(|| AgentError::ModelProtocol("model output size overflow".to_string()))?;
	if aggregate > MAX_TOTAL_MODEL_OUTPUT_BYTES {
		return Err(AgentError::ModelProtocol(format!(
			"aggregate model output exceeds {MAX_TOTAL_MODEL_OUTPUT_BYTES} bytes"
		)));
	}
	*aggregate_bytes = aggregate;
	output.push_str(delta);
	Ok(())
}

fn validate_terminal_response(response: &GenerationResponse) -> Result<(), AgentError> {
	if response.text.len() > MAX_MODEL_OUTPUT_BYTES {
		return Err(AgentError::ModelProtocol(format!(
			"terminal answer text exceeds {MAX_MODEL_OUTPUT_BYTES} bytes"
		)));
	}
	if response.reasoning.as_ref().is_some_and(|reasoning| {
		reasoning.is_empty()
			|| reasoning.len() > MAX_MODEL_OUTPUT_BYTES
			|| reasoning
				.chars()
				.any(|character| character.is_control() && !matches!(character, '\n' | '\r' | '\t'))
	}) {
		return Err(AgentError::ModelProtocol(format!(
			"terminal reasoning must be valid generated text in 1..={MAX_MODEL_OUTPUT_BYTES} bytes"
		)));
	}
	if response.tool_calls.len() > MAX_TOOL_CALLS_PER_ROUND {
		return Err(AgentError::ModelProtocol(format!(
			"terminal response exceeds {MAX_TOOL_CALLS_PER_ROUND} tool calls"
		)));
	}
	let mut argument_bytes = 0_usize;
	for call in &response.tool_calls {
		if call.id.len() > MAX_TOOL_CALL_ID_BYTES
			|| call.id.trim().is_empty()
			|| call.id.chars().any(char::is_control)
		{
			return Err(AgentError::ModelProtocol(
				"terminal response contains an invalid tool-call ID".to_string(),
			));
		}
		if !valid_tool_name(&call.name) || !call.arguments.is_object() {
			return Err(AgentError::ModelProtocol(format!(
				"terminal response contains an invalid call to {:?}",
				call.name
			)));
		}
		let bytes = crate::generation::bounded_json_len(&call.arguments, MAX_TOOL_SCHEMA_BYTES)
			.ok_or_else(|| {
				AgentError::ModelProtocol(format!(
					"tool {:?} arguments exceed structural or {MAX_TOOL_SCHEMA_BYTES}-byte limits",
					call.name
				))
			})?;
		argument_bytes = argument_bytes
			.checked_add(bytes)
			.ok_or_else(|| AgentError::ModelProtocol("tool argument size overflow".to_string()))?;
		if argument_bytes > MAX_TOTAL_TOOL_ARGUMENT_BYTES {
			return Err(AgentError::ModelProtocol(format!(
				"terminal tool arguments exceed {MAX_TOTAL_TOOL_ARGUMENT_BYTES} aggregate bytes"
			)));
		}
	}
	if response
		.speculation
		.as_ref()
		.is_some_and(|stats| stats.accepted_by_depth.len() > 8)
	{
		return Err(AgentError::ModelProtocol(
			"terminal speculation metadata exceeds eight draft depths".to_string(),
		));
	}
	Ok(())
}

fn validate_approval_text(label: &str, text: &str) -> Result<(), String> {
	if text.len() > MAX_APPROVAL_REASON_BYTES {
		return Err(format!(
			"{label} exceeded {MAX_APPROVAL_REASON_BYTES} bytes"
		));
	}
	if text.trim().is_empty() {
		return Err(format!("{label} cannot be empty"));
	}
	if text.chars().any(char::is_control) {
		return Err(format!("{label} cannot contain control characters"));
	}
	Ok(())
}

fn bounded_display(value: &impl fmt::Display) -> String {
	struct Writer {
		text: String,
		truncated: bool,
	}

	impl fmt::Write for Writer {
		fn write_str(&mut self, text: &str) -> fmt::Result {
			let remaining = MAX_EVENT_SINK_ERROR_BYTES.saturating_sub(self.text.len());
			if text.len() <= remaining {
				self.text.push_str(text);
				return Ok(());
			}
			let mut boundary = remaining;
			while boundary > 0 && !text.is_char_boundary(boundary) {
				boundary -= 1;
			}
			self.text.push_str(&text[..boundary]);
			self.truncated = true;
			Ok(())
		}
	}

	let mut writer = Writer {
		text: String::new(),
		truncated: false,
	};
	let _ = write!(&mut writer, "{value}");
	if writer.truncated && writer.text.len() < MAX_EVENT_SINK_ERROR_BYTES {
		writer.text.push('…');
	}
	if writer.text.is_empty() {
		"event consumer returned an empty error".to_string()
	} else {
		writer.text
	}
}

#[derive(Debug, thiserror::Error)]
pub(super) enum BoundedJsonError {
	#[error("serialized JSON exceeds {limit} bytes")]
	Limit { limit: usize },
	#[error("cannot serialize JSON: {0}")]
	Serialize(#[source] serde_json::Error),
	#[error("JSON serializer produced invalid UTF-8")]
	Utf8,
}

pub(super) fn serialize_json_pretty_bounded<T: Serialize>(
	value: &T,
	limit: usize,
) -> Result<String, BoundedJsonError> {
	struct Writer {
		bytes: Vec<u8>,
		limit: usize,
		exceeded: bool,
	}

	impl std::io::Write for Writer {
		fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
			let Some(length) = self.bytes.len().checked_add(bytes.len()) else {
				self.exceeded = true;
				return Err(std::io::Error::other("serialized JSON size overflow"));
			};
			if length > self.limit {
				self.exceeded = true;
				return Err(std::io::Error::other("serialized JSON exceeds limit"));
			}
			self.bytes.extend_from_slice(bytes);
			Ok(bytes.len())
		}

		fn flush(&mut self) -> std::io::Result<()> {
			Ok(())
		}
	}

	let mut writer = Writer {
		bytes: Vec::with_capacity(limit.min(8 * 1024)),
		limit,
		exceeded: false,
	};
	if let Err(error) = serde_json::to_writer_pretty(&mut writer, value) {
		return if writer.exceeded {
			Err(BoundedJsonError::Limit { limit })
		} else {
			Err(BoundedJsonError::Serialize(error))
		};
	}
	String::from_utf8(writer.bytes).map_err(|_| BoundedJsonError::Utf8)
}

const fn add_usage(total: &mut Usage, next: Usage) {
	total.prompt_tokens = total.prompt_tokens.saturating_add(next.prompt_tokens);
	total.cached_tokens = total.cached_tokens.saturating_add(next.cached_tokens);
	total.completion_tokens = total
		.completion_tokens
		.saturating_add(next.completion_tokens);
}

fn emit_terminal_error<S>(turn_id: Uuid, error: &AgentError, emit: &mut S) -> Result<(), AgentError>
where
	S: EventEmitter,
{
	if matches!(error, AgentError::Cancelled) {
		emit.emit(AgentEvent::Cancelled { turn_id })
	} else {
		emit.emit(AgentEvent::TurnFailed {
			turn_id,
			message: bounded_display(error),
		})
	}
}

fn validate_tool_definition(definition: &ToolDefinition) -> Result<(), AgentError> {
	let name = definition.name.as_str();
	if !valid_tool_name(name) {
		return Err(AgentError::Configuration(format!(
			"invalid tool name {name:?}; use 1-64 ASCII letters, digits, '_' or '-'"
		)));
	}
	if definition.description.len() > MAX_TOOL_DESCRIPTION_BYTES
		|| definition.description.trim().is_empty()
	{
		return Err(AgentError::Configuration(format!(
			"tool {name:?} description must be in 1..={MAX_TOOL_DESCRIPTION_BYTES} bytes"
		)));
	}
	if !definition.parameters.is_object() {
		return Err(AgentError::Configuration(format!(
			"tool {name:?} parameters must be a JSON Schema object"
		)));
	}
	crate::engine::tools::validate_tool_schema(&definition.parameters).map_err(|reason| {
		AgentError::Configuration(format!("tool {name:?} schema is invalid: {reason}"))
	})?;
	if crate::generation::bounded_json_len(&definition.parameters, MAX_TOOL_SCHEMA_BYTES).is_none()
	{
		return Err(AgentError::Configuration(format!(
			"tool {name:?} schema exceeds structural or {MAX_TOOL_SCHEMA_BYTES}-byte limits"
		)));
	}
	Ok(())
}

fn tool_arguments_match(definition: &ToolDefinition, arguments: &serde_json::Value) -> bool {
	crate::engine::tools::arguments_satisfy_schema(&definition.parameters, arguments)
}

pub(crate) fn validate_user_message(message: &Message) -> Result<(), AgentError> {
	if message.role != Role::User {
		return Err(AgentError::Configuration(
			"agent input message must have user role".to_string(),
		));
	}
	if message.content.is_empty() {
		return Err(AgentError::Configuration(
			"agent input message requires content".to_string(),
		));
	}
	if message.content.len() > crate::generation::MAX_MESSAGE_CONTENT_PARTS {
		return Err(AgentError::Configuration(format!(
			"agent input exceeds {} content parts",
			crate::generation::MAX_MESSAGE_CONTENT_PARTS
		)));
	}
	if !message.tool_calls.is_empty()
		|| message.tool_call_id.is_some()
		|| message.reasoning.is_some()
	{
		return Err(AgentError::Configuration(
			"user input cannot carry tool or reasoning state".to_string(),
		));
	}
	let mut bytes = 0_usize;
	for content in &message.content {
		let length = match content {
			Content::Text(text) => text.len(),
			Content::Image(data) | Content::Audio(data) | Content::Video(data) => data.len(),
		};
		if matches!(content, Content::Video(_)) {
			return Err(AgentError::Configuration(
				"self-contained video decoding is not available".to_string(),
			));
		}
		bytes = bytes
			.checked_add(length)
			.ok_or_else(|| AgentError::Configuration("user content size overflow".to_string()))?;
	}
	if bytes == 0 {
		return Err(AgentError::Configuration(
			"agent input content cannot be empty".to_string(),
		));
	}
	if bytes > MAX_USER_CONTENT_BYTES {
		return Err(AgentError::Configuration(format!(
			"agent input exceeds {MAX_USER_CONTENT_BYTES} bytes"
		)));
	}
	Ok(())
}

pub(crate) fn validate_history(history: &[Message]) -> Result<BTreeSet<String>, HistoryError> {
	if history.len() > crate::generation::MAX_MESSAGES {
		return Err(invalid_history(
			crate::generation::MAX_MESSAGES,
			format!(
				"history exceeds {} messages",
				crate::generation::MAX_MESSAGES
			),
		));
	}
	let mut issued = BTreeSet::new();
	let mut pending = BTreeSet::new();
	for (index, message) in history.iter().enumerate() {
		if message.content.len() > crate::generation::MAX_MESSAGE_CONTENT_PARTS {
			return Err(invalid_history(
				index,
				format!(
					"message exceeds {} content parts",
					crate::generation::MAX_MESSAGE_CONTENT_PARTS
				),
			));
		}
		if !pending.is_empty() && message.role != Role::Tool {
			return Err(HistoryError::Unresolved {
				call_ids: pending.iter().cloned().collect(),
				index,
			});
		}
		validate_history_message(index, message)?;
		match message.role {
			Role::System | Role::User => {}
			Role::Assistant => {
				for call in &message.tool_calls {
					if !issued.insert(call.id.clone()) {
						return Err(HistoryError::DuplicateCall {
							call_id: call.id.clone(),
							index,
						});
					}
					pending.insert(call.id.clone());
				}
			}
			Role::Tool => {
				let Some(call_id) = message.tool_call_id.as_deref() else {
					return Err(invalid_history(index, "tool messages require tool_call_id"));
				};
				if !pending.remove(call_id) {
					return Err(HistoryError::MismatchedResult {
						call_id: call_id.to_string(),
						index,
					});
				}
			}
		}
	}
	if !pending.is_empty() {
		return Err(HistoryError::Unresolved {
			call_ids: pending.iter().cloned().collect(),
			index: history.len(),
		});
	}
	Ok(issued)
}

pub(crate) fn validate_history_message(
	index: usize,
	message: &Message,
) -> Result<(), HistoryError> {
	match message.role {
		Role::System => {
			if index != 0 {
				return Err(invalid_history(
					index,
					"system message is valid only at history start",
				));
			}
			if message.content.is_empty()
				|| message
					.content
					.iter()
					.any(|content| !matches!(content, Content::Text(_)))
				|| !message.tool_calls.is_empty()
				|| message.tool_call_id.is_some()
				|| message.reasoning.is_some()
			{
				return Err(invalid_history(
					index,
					"system message requires only text content",
				));
			}
			let bytes = message
				.content
				.iter()
				.filter_map(|content| match content {
					Content::Text(text) => Some(text.len()),
					Content::Image(_) | Content::Audio(_) | Content::Video(_) => None,
				})
				.try_fold(0_usize, usize::checked_add)
				.ok_or_else(|| invalid_history(index, "system content size overflow"))?;
			let nonempty = message
				.content
				.iter()
				.any(|content| matches!(content, Content::Text(text) if !text.trim().is_empty()));
			if !nonempty || bytes > MAX_SYSTEM_PROMPT_BYTES {
				return Err(invalid_history(
					index,
					format!(
						"system text must be non-empty and at most {MAX_SYSTEM_PROMPT_BYTES} bytes"
					),
				));
			}
		}
		Role::User => {
			validate_user_message(message)
				.map_err(|error| invalid_history(index, error.to_string()))?;
		}
		Role::Assistant => validate_assistant_history_message(index, message)?,
		Role::Tool => {
			if message.content.is_empty()
				|| !message.tool_calls.is_empty()
				|| message.reasoning.is_some()
				|| message.tool_call_id.is_none()
				|| message
					.content
					.iter()
					.any(|content| !matches!(content, Content::Text(_)))
			{
				return Err(invalid_history(
					index,
					"tool messages require text and one tool_call_id",
				));
			}
			let bytes = message
				.content
				.iter()
				.filter_map(|content| match content {
					Content::Text(text) => Some(text.len()),
					Content::Image(_) | Content::Audio(_) | Content::Video(_) => None,
				})
				.try_fold(0_usize, usize::checked_add)
				.ok_or_else(|| invalid_history(index, "tool content size overflow"))?;
			if bytes == 0 || bytes > MAX_TOOL_OUTPUT_BYTES {
				return Err(invalid_history(
					index,
					format!("tool text must be in 1..={MAX_TOOL_OUTPUT_BYTES} bytes"),
				));
			}
			if message
				.tool_call_id
				.as_deref()
				.is_none_or(|call_id| !is_uuid_v7(call_id))
			{
				return Err(invalid_history(
					index,
					"tool_call_id must be a UUIDv7 issued by the agent",
				));
			}
		}
	}
	Ok(())
}

fn validate_assistant_history_message(index: usize, message: &Message) -> Result<(), HistoryError> {
	if (message.content.is_empty() && message.tool_calls.is_empty())
		|| message.tool_call_id.is_some()
		|| message
			.content
			.iter()
			.any(|content| !matches!(content, Content::Text(_)))
	{
		return Err(invalid_history(
			index,
			"assistant messages require text or calls and cannot be tool results",
		));
	}
	if message.tool_calls.len() > MAX_TOOL_CALLS_PER_ROUND {
		return Err(invalid_history(
			index,
			format!("assistant message exceeds {MAX_TOOL_CALLS_PER_ROUND} tool calls"),
		));
	}
	let text_bytes = message
		.content
		.iter()
		.filter_map(|content| match content {
			Content::Text(text) => Some(text.len()),
			Content::Image(_) | Content::Audio(_) | Content::Video(_) => None,
		})
		.try_fold(0_usize, usize::checked_add)
		.ok_or_else(|| invalid_history(index, "assistant text size overflow"))?;
	if text_bytes > MAX_MODEL_OUTPUT_BYTES
		|| message.reasoning.as_ref().is_some_and(|reasoning| {
			reasoning.is_empty()
				|| reasoning.len() > MAX_MODEL_OUTPUT_BYTES
				|| reasoning.chars().any(|character| {
					character.is_control() && !matches!(character, '\n' | '\r' | '\t')
				})
		}) {
		return Err(invalid_history(
			index,
			format!(
				"assistant text and reasoning must each fit within {MAX_MODEL_OUTPUT_BYTES} bytes"
			),
		));
	}
	let mut argument_bytes = 0_usize;
	for call in &message.tool_calls {
		let reason = if !valid_tool_name(&call.name) {
			Some(format!("invalid tool name {:?}", call.name))
		} else if !is_uuid_v7(&call.id) {
			Some(format!("tool-call ID {:?} is not a UUIDv7", call.id))
		} else if !call.arguments.is_object() {
			Some(format!(
				"tool-call {:?} arguments are not an object",
				call.id
			))
		} else {
			None
		};
		if let Some(reason) = reason {
			return Err(invalid_history(index, reason));
		}
		let bytes = crate::generation::bounded_json_len(&call.arguments, MAX_TOOL_SCHEMA_BYTES)
			.ok_or_else(|| {
				invalid_history(
					index,
					format!(
						"tool-call {:?} arguments exceed structural or {MAX_TOOL_SCHEMA_BYTES}-byte limits",
						call.id
					),
				)
			})?;
		argument_bytes = argument_bytes
			.checked_add(bytes)
			.ok_or_else(|| invalid_history(index, "tool-call argument size overflow"))?;
		if argument_bytes > MAX_TOTAL_TOOL_ARGUMENT_BYTES {
			return Err(invalid_history(
				index,
				format!(
					"assistant tool-call arguments exceed {MAX_TOTAL_TOOL_ARGUMENT_BYTES} aggregate bytes"
				),
			));
		}
	}
	Ok(())
}

fn invalid_history(index: usize, reason: impl Into<String>) -> HistoryError {
	HistoryError::InvalidMessage {
		index,
		reason: reason.into(),
	}
}

fn is_uuid_v7(value: &str) -> bool {
	if value.len() != 36 {
		return false;
	}
	Uuid::parse_str(value).ok().is_some_and(|uuid| {
		uuid.get_version_num() == 7
			&& uuid.get_variant() == uuid::Variant::RFC4122
			&& uuid.hyphenated().to_string() == value
	})
}

fn valid_tool_name(name: &str) -> bool {
	!name.is_empty()
		&& name.len() <= 64
		&& name
			.bytes()
			.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
}

#[cfg(test)]
#[path = "agent/tests.rs"]
mod tests;
