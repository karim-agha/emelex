//! Client construction and the loaded-model handle.
//!
//! A [`Client`] owns exactly one loaded MLX checkpoint, hosted on a
//! dedicated inference thread that both loads the session and runs
//! every generation, and carries the generation defaults applied when a
//! request leaves a knob unset. Cloning a `Client` shares the same
//! loaded model and inference queue.

use std::{
	path::{Path, PathBuf},
	sync::{
		Arc,
		atomic::{AtomicBool, Ordering},
		mpsc,
	},
	time::Duration,
};

#[cfg(feature = "rig")]
use rig_core::{agent::AgentBuilder, client::CompletionClient};

#[cfg(feature = "rig")]
use crate::model::CompletionModel;
use crate::{
	engine::{
		generate::{GenerateOptions as EngineOptions, Session},
		prompt_cache::PromptCacheConfig,
		sampling::SamplingConfig,
		streaming::TokenKind,
		tokenizer::ChatMessage,
		tools::Tool,
	},
	error::Error,
	generation::{GenerationEvent, GenerationRequest, GenerationResponse, GenerationStream},
	home::EmelexHome,
	model::{ModelFile, ModelSnapshotId},
	runtime,
};

/// One unit of inference work, executed on the client's dedicated
/// thread with exclusive access to the loaded session.
pub type Job = Box<dyn FnOnce(&Session) + Send>;

/// Generation defaults applied when a request leaves a knob unset.
#[derive(Debug, Clone)]
pub struct Defaults {
	pub max_tokens: usize,
	pub context_tokens: usize,
	pub sampling: SamplingConfig,
	pub enable_thinking: Option<bool>,
	pub reasoning_budget_tokens: Option<usize>,
	pub prompt_cache: Option<bool>,
	/// emelex patch (not upstream): default draft depth for MTP
	/// self-speculative decoding. `None` = off; resolution normalizes
	/// `Some(0)` to off. Public request boundaries reject values above 8.
	pub speculative_tokens: Option<usize>,
}

pub struct Inner {
	/// Queue feeding the dedicated inference thread. MLX binds GPU
	/// streams (and their Metal command encoders) to the OS thread that
	/// created them, and evaluates lazily - so the session must live its
	/// entire life (load AND every generation) on ONE thread. Running
	/// generations on a shared blocking pool worked for text only by
	/// luck (the text tower holds no lazy load-time arrays; the vision
	/// tower does) and minted a fresh GPU stream per pool thread.
	pub jobs: mpsc::SyncSender<Job>,
	pub defaults: Defaults,
	pub path: PathBuf,
	pub model_snapshot_id: Option<ModelSnapshotId>,
	media_capabilities: MediaCapabilities,
	chat_capabilities: ChatCapabilities,
	/// emelex patch (not upstream): whether the loaded checkpoint carries
	/// a usable MTP module (speculative decoding can draft).
	pub supports_mtp: bool,
}

#[derive(Debug, Clone, Copy)]
struct MediaCapabilities {
	images: bool,
	audio: bool,
}

#[derive(Debug, Clone, Copy)]
#[allow(
	clippy::struct_excessive_bools,
	reason = "independent template capabilities must remain separately addressable"
)]
struct ChatCapabilities {
	system_prompt: bool,
	tools: bool,
	reasoning_history: bool,
	thinking_toggle: bool,
}

#[derive(Debug, Clone, Copy)]
struct LoadedCapabilities {
	media: MediaCapabilities,
	chat: ChatCapabilities,
	mtp: bool,
}

impl Inner {
	/// Submit a job without allowing callers to grow memory unboundedly.
	pub fn submit(&self, job: Job) -> Result<(), SubmitError> {
		self.jobs.try_send(job).map_err(|error| match error {
			mpsc::TrySendError::Full(_) => SubmitError::Full,
			mpsc::TrySendError::Disconnected(_) => SubmitError::Closed,
		})
	}
}

#[derive(Debug, Clone, Copy)]
pub enum SubmitError {
	Full,
	Closed,
}

impl std::fmt::Display for SubmitError {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		formatter.write_str(match self {
			Self::Full => "inference queue is full",
			Self::Closed => "inference thread is gone",
		})
	}
}

struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
	fn drop(&mut self) {
		self.0.store(true, Ordering::Release);
	}
}

struct NotifyOnDrop(Option<tokio::sync::oneshot::Sender<()>>);

impl NotifyOnDrop {
	const fn new(sender: tokio::sync::oneshot::Sender<()>) -> Self {
		Self(Some(sender))
	}
}

impl Drop for NotifyOnDrop {
	fn drop(&mut self) {
		if let Some(sender) = self.0.take() {
			let _ = sender.send(());
		}
	}
}

fn is_cancelled(cancelled: &AtomicBool) -> bool {
	cancelled.load(Ordering::Acquire)
}

/// Keeps answer deltas an exact prefix of the terminal answer.
///
/// Tool-call syntax cannot be accepted until the complete reply has been
/// parsed against the advertised schemas. Once raw tool markup begins, later
/// answer-looking text is withheld and reconciled from the validated terminal
/// response instead of risking a live/terminal mismatch.
#[derive(Debug, Default)]
pub struct StreamTextReconciler {
	emitted: String,
	withhold: bool,
}

impl StreamTextReconciler {
	/// Stop forwarding answer deltas until terminal validation.
	pub const fn observe_tool_span(&mut self) {
		self.withhold = true;
	}

	/// Record and return a forwardable answer delta, or withhold it.
	pub fn push_text(&mut self, text: String) -> Option<String> {
		if self.withhold {
			return None;
		}
		self.emitted.push_str(&text);
		Some(text)
	}

	/// Return terminal answer bytes not already emitted.
	///
	/// # Errors
	///
	/// Returns [`Error::StreamProtocol`] when prior deltas are not an exact
	/// prefix of `terminal`.
	pub fn terminal_suffix<'a>(&self, terminal: &'a str) -> Result<&'a str, Error> {
		terminal.strip_prefix(&self.emitted).ok_or_else(|| {
			Error::StreamProtocol(
				"incremental answer is not a prefix of the terminal answer".to_string(),
			)
		})
	}
}

/// A handle to one locally loaded MLX model.
#[derive(Clone)]
pub struct Client {
	pub(crate) inner: Arc<Inner>,
}

impl Client {
	/// Load the MLX checkpoint at `path` with default settings.
	pub fn from_path(path: impl Into<PathBuf>) -> Result<Self, Error> {
		ClientBuilder::new(path).build()
	}

	/// Start building a client with non-default generation or cache
	/// settings.
	pub fn builder(path: impl Into<PathBuf>) -> ClientBuilder {
		ClientBuilder::new(path)
	}

	/// The completion model backed by this client's loaded checkpoint.
	#[cfg(feature = "rig")]
	pub fn model(&self) -> CompletionModel {
		CompletionModel::from_client(self, self.inner.path.display().to_string())
	}

	/// Build an extractor for structured output on this client's loaded
	/// model (structured output is best-effort prompt injection; rig's
	/// extractor retry loop covers occasional schema misses).
	#[cfg(feature = "rig")]
	pub fn extractor<T>(&self) -> rig_core::extractor::ExtractorBuilder<CompletionModel, T>
	where
		T: rig_core::schemars::JsonSchema
			+ for<'a> serde::Deserialize<'a>
			+ serde::Serialize
			+ Send
			+ Sync,
	{
		rig_core::extractor::ExtractorBuilder::new(self.model())
	}

	/// Build a rig agent on this client's loaded model.
	///
	/// Takes no model name — a `Client` is bound to the one checkpoint it
	/// loaded. (The [`CompletionClient`] trait's `agent(name)` remains
	/// callable through the trait; the name is used for tracing only.)
	#[cfg(feature = "rig")]
	pub fn agent(&self) -> AgentBuilder<CompletionModel> {
		AgentBuilder::new(self.model())
	}

	/// emelex patch (not upstream): whether the loaded checkpoint's exact
	/// bytes carry a usable, parity-certified MTP module.
	///
	/// Requests with a non-zero speculative depth fail explicitly when this
	/// is `false`; they are never silently downgraded to target-only decoding.
	pub fn supports_mtp(&self) -> bool {
		self.inner.supports_mtp
	}

	/// Whether the loaded checkpoint accepts images.
	pub fn supports_images(&self) -> bool {
		self.inner.media_capabilities.images
	}

	/// Whether the loaded checkpoint accepts audio.
	pub fn supports_audio(&self) -> bool {
		self.inner.media_capabilities.audio
	}

	/// Whether the loaded chat template preserves a distinct system role.
	pub fn supports_system_prompt(&self) -> bool {
		self.inner.chat_capabilities.system_prompt
	}

	/// Whether the loaded template and parser preserve complete tool rounds.
	pub fn supports_tools(&self) -> bool {
		self.inner.chat_capabilities.tools
	}

	/// Whether explicit reasoning spans survive a follow-up turn.
	pub fn supports_reasoning_history(&self) -> bool {
		self.inner.chat_capabilities.reasoning_history
	}

	/// Whether the template distinguishes thinking enabled from disabled.
	pub fn supports_thinking_toggle(&self) -> bool {
		self.inner.chat_capabilities.thinking_toggle
	}

	/// Whether the template supports either explicit reasoning dimension.
	pub fn supports_reasoning(&self) -> bool {
		self.supports_reasoning_history() || self.supports_thinking_toggle()
	}

	/// Exact installed snapshot identity when loaded by
	/// [`crate::models::ModelManager`].
	pub fn model_snapshot_id(&self) -> Option<&ModelSnapshotId> {
		self.inner.model_snapshot_id.as_ref()
	}

	/// Effective context ceiling after model/config clamping.
	pub fn effective_context_tokens(&self) -> usize {
		self.inner.defaults.context_tokens
	}

	/// Effective default output ceiling after model/config clamping.
	pub fn effective_max_tokens(&self) -> usize {
		self.inner.defaults.max_tokens
	}

	/// Canonical loaded checkpoint path.
	pub fn path(&self) -> &Path {
		&self.inner.path
	}

	/// Run one deterministic token through tokenizer, prefill, model forward,
	/// and decode. Used before an installed snapshot is marked verified.
	pub(crate) fn runtime_probe(&self) -> Result<(), Error> {
		let mut request = GenerationRequest::text("Respond with one token.");
		request.options.max_tokens = Some(1);
		request.options.temperature = Some(0.0);
		request.options.thinking = Some(crate::config::ThinkingMode::Off);
		request.options.speculative_tokens = Some(0);
		request.options.prompt_cache = Some(false);
		let request = request.into_engine(
			&self.inner.defaults,
			self.inner.media_capabilities.images,
			self.inner.media_capabilities.audio,
		)?;
		let (sender, receiver) = mpsc::channel();
		self.inner
			.submit(Box::new(move |session| {
				let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
					session.generate_cached(&request.messages, None, request.options, |_| true)
				}));
				let result = match outcome {
					Ok(Ok(_)) => Ok(()),
					Ok(Err(error)) => Err(crate::error::from_engine(error)),
					Err(_) => Err(Error::InferencePanic),
				};
				let _ = sender.send(result);
			}))
			.map_err(submit_error)?;
		receiver.recv().map_err(|_| Error::InferenceChannel {
			operation: "receive",
		})?
	}

	/// Generate one complete response through the native API.
	///
	/// Dropping the returned future cancels queued or in-flight work
	/// cooperatively.
	///
	/// # Errors
	///
	/// Returns request validation, queue admission, engine, or worker errors.
	pub async fn generate(&self, request: GenerationRequest) -> Result<GenerationResponse, Error> {
		let request = request.into_engine(
			&self.inner.defaults,
			self.inner.media_capabilities.images,
			self.inner.media_capabilities.audio,
		)?;
		self.inner.validate_generation_capabilities(
			&request.messages,
			&request.tools,
			request.options,
		)?;
		let cancelled = Arc::new(AtomicBool::new(false));
		let guard = CancelOnDrop(Arc::clone(&cancelled));
		let (sender, receiver) = tokio::sync::oneshot::channel();
		self.inner
			.submit(Box::new(move |session| {
				if is_cancelled(&cancelled) {
					let _ = sender.send(Err(Error::Cancelled));
					return;
				}
				let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
					let request_cancelled = || is_cancelled(&cancelled);
					session.generate_cached_cancellable(
						&request.messages,
						(!request.tools.is_empty()).then_some(request.tools.as_slice()),
						request.options,
						&request_cancelled,
						|_| !is_cancelled(&cancelled),
					)
				}));
				let result = match outcome {
					Ok(Ok(reply)) => Ok(GenerationResponse::from_engine(reply)),
					Ok(Err(error)) => Err(crate::error::from_engine(error)),
					Err(_) => Err(Error::InferencePanic),
				};
				let _ = sender.send(result);
			}))
			.map_err(submit_error)?;
		let result = receiver.await.map_err(|_| Error::InferenceChannel {
			operation: "receive",
		})?;
		drop(guard);
		result
	}

	/// Start bounded streaming generation through the native API.
	///
	/// # Errors
	///
	/// Returns before queueing when the request is invalid or queue admission
	/// fails.
	pub fn stream(&self, request: GenerationRequest) -> Result<GenerationStream, Error> {
		let request = request.into_engine(
			&self.inner.defaults,
			self.inner.media_capabilities.images,
			self.inner.media_capabilities.audio,
		)?;
		self.inner.validate_generation_capabilities(
			&request.messages,
			&request.tools,
			request.options,
		)?;
		let cancelled = Arc::new(AtomicBool::new(false));
		let job_cancelled = Arc::clone(&cancelled);
		let (sender, receiver) = tokio::sync::mpsc::channel(64);
		let (completion_sender, completion_receiver) = tokio::sync::oneshot::channel();
		self.inner
			.submit(Box::new(move |session| {
				let _completion = NotifyOnDrop::new(completion_sender);
				if is_cancelled(&job_cancelled) {
					return;
				}
				let mut visible_text = StreamTextReconciler::default();
				let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
					let request_cancelled = || is_cancelled(&job_cancelled) || sender.is_closed();
					session.generate_cached_cancellable(
						&request.messages,
						(!request.tools.is_empty()).then_some(request.tools.as_slice()),
						request.options,
						&request_cancelled,
						|token| {
							if is_cancelled(&job_cancelled) {
								return false;
							}
							let event = match token.kind {
								TokenKind::Text => {
									let Some(text) = visible_text.push_text(token.text) else {
										return !sender.is_closed();
									};
									if text.is_empty() {
										return !sender.is_closed();
									}
									GenerationEvent::Text(text)
								}
								TokenKind::Reasoning => GenerationEvent::Reasoning(token.text),
								TokenKind::ToolCall => {
									visible_text.observe_tool_span();
									return !sender.is_closed();
								}
							};
							if matches!(&event, GenerationEvent::Reasoning(text) if text.is_empty())
							{
								return !sender.is_closed();
							}
							sender.blocking_send(Ok(event)).is_ok()
						},
					)
				}));
				match outcome {
					Ok(Ok(reply)) => {
						let response = GenerationResponse::from_engine(reply);
						let suffix = match visible_text.terminal_suffix(&response.text) {
							Ok(suffix) => suffix,
							Err(error) => {
								let _ = sender.blocking_send(Err(error));
								return;
							}
						};
						if !suffix.is_empty()
							&& sender
								.blocking_send(Ok(GenerationEvent::Text(suffix.to_string())))
								.is_err()
						{
							return;
						}
						for call in &response.tool_calls {
							if sender
								.blocking_send(Ok(GenerationEvent::ToolCall(call.clone())))
								.is_err()
							{
								return;
							}
						}
						let _ = sender.blocking_send(Ok(GenerationEvent::Completed(response)));
					}
					Ok(Err(error)) => {
						let _ = sender.blocking_send(Err(crate::error::from_engine(error)));
					}
					Err(_) => {
						let _ = sender.blocking_send(Err(Error::InferencePanic));
					}
				}
			}))
			.map_err(submit_error)?;
		Ok(GenerationStream::new(
			receiver,
			cancelled,
			completion_receiver,
		))
	}
}

impl Inner {
	pub(crate) fn validate_generation_capabilities(
		&self,
		messages: &[ChatMessage],
		tools: &[Tool],
		options: EngineOptions,
	) -> Result<(), Error> {
		if messages.iter().any(|message| message.role == "system")
			&& !self.chat_capabilities.system_prompt
		{
			return Err(Error::CapabilityUnavailable {
				capability: "interaction:system_prompt",
				reason: "loaded chat template does not preserve a distinct system role".to_string(),
			});
		}
		let has_tool_history = messages.iter().any(|message| {
			message.role == "tool"
				|| !message.tool_calls.is_empty()
				|| message.tool_call_id.is_some()
		});
		if has_tool_history && tools.is_empty() {
			return Err(Error::InvalidRequest(
				"tool protocol history requires the matching current tool definitions".to_string(),
			));
		}
		if (!tools.is_empty() || has_tool_history) && !self.chat_capabilities.tools {
			return Err(Error::CapabilityUnavailable {
				capability: "interaction:tools",
				reason: "loaded template/parser pair does not preserve complete tool rounds"
					.to_string(),
			});
		}
		if messages
			.iter()
			.any(|message| message.reasoning_content.is_some())
			&& !self.chat_capabilities.reasoning_history
		{
			return Err(Error::CapabilityUnavailable {
				capability: "interaction:reasoning_history",
				reason: "loaded chat template does not preserve explicit reasoning across turns"
					.to_string(),
			});
		}
		if options.reasoning_budget_tokens.is_some() && options.enable_thinking != Some(true) {
			return Err(Error::InvalidRequest(
				"reasoning_budget_tokens requires thinking to be enabled".to_string(),
			));
		}
		if options.enable_thinking == Some(true) && !self.chat_capabilities.thinking_toggle {
			return Err(Error::CapabilityUnavailable {
				capability: "interaction:thinking_toggle",
				reason: "loaded chat template does not distinguish thinking enabled from disabled"
					.to_string(),
			});
		}
		Ok(())
	}
}

const fn submit_error(error: SubmitError) -> Error {
	match error {
		SubmitError::Full => Error::InferenceBusy,
		SubmitError::Closed => Error::InferenceChannel {
			operation: "submit",
		},
	}
}

/// Per-agent reasoning ("thinking") overrides for agents built on an
/// emelex model.
///
/// [`ClientBuilder::enable_thinking`] sets the client-wide default;
/// these methods override it for a single agent:
///
/// ```no_run
/// use emelex::ReasoningExt as _;
/// # fn demo(client: &emelex::Client) {
/// let terse = client.agent().enable_thinking(false).build();
/// let deliberate = client.agent().reasoning_budget_tokens(320).build();
/// # let _ = (terse, deliberate);
/// # }
/// ```
///
/// rig has no typed channel for provider-specific agent options, so
/// under the hood these write emelex's keys into the builder's
/// `additional_params`. That setter *replaces* rather than merges, so
/// each method here writes a complete reasoning configuration and the
/// last call wins - if you also need unrelated provider parameters on
/// the same agent, set them all yourself with `additional_params`.
#[cfg(feature = "rig")]
pub trait ReasoningExt {
	/// Turn the model's thinking mode on or off for this agent.
	#[must_use]
	fn enable_thinking(self, enabled: bool) -> Self;

	/// Enable thinking with a cap, in tokens, on how long the reasoning
	/// span may run before it is force-closed. Implies
	/// `enable_thinking(true)`.
	///
	/// The budget draws from the same window as `max_tokens`: keep it
	/// comfortably below, or the reply can be all thinking and no answer.
	#[must_use]
	fn reasoning_budget_tokens(self, tokens: usize) -> Self;
}

#[cfg(feature = "rig")]
impl<S> ReasoningExt for AgentBuilder<CompletionModel, S> {
	fn enable_thinking(self, enabled: bool) -> Self {
		self.additional_params(reasoning_params(enabled, None))
	}

	fn reasoning_budget_tokens(self, tokens: usize) -> Self {
		self.additional_params(reasoning_params(true, Some(tokens)))
	}
}

/// The `additional_params` payload emelex's request conversion reads
/// back as reasoning configuration (see `convert::options`).
#[cfg(feature = "rig")]
pub fn reasoning_params(enabled: bool, budget_tokens: Option<usize>) -> serde_json::Value {
	let mut params = serde_json::Map::new();
	params.insert("enable_thinking".to_string(), enabled.into());
	if let Some(tokens) = budget_tokens {
		params.insert("reasoning_budget_tokens".to_string(), tokens.into());
	}
	serde_json::Value::Object(params)
}

#[cfg(feature = "rig")]
impl CompletionClient for Client {
	type CompletionModel = CompletionModel;
}

#[cfg(feature = "rig")]
impl rig_core::client::ProviderClient for Client {
	type Error = Error;
	type Input = PathBuf;

	/// Load the checkpoint named by the `EMELEX_MODEL_PATH` environment
	/// variable.
	fn from_env() -> Result<Self, Self::Error> {
		let path = std::env::var_os("EMELEX_MODEL_PATH").ok_or_else(|| Error::ModelPath {
			path: PathBuf::new(),
			reason: "EMELEX_MODEL_PATH is not set".to_string(),
		})?;
		Self::from_path(PathBuf::from(path))
	}

	fn from_val(input: Self::Input) -> Result<Self, Self::Error> {
		Self::from_path(input)
	}
}

impl std::fmt::Debug for Client {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("Client")
			.field("path", &self.inner.path)
			.finish_non_exhaustive()
	}
}

/// Builder for [`Client`] exposing generation defaults and prompt-cache
/// configuration.
#[derive(Debug, Clone)]
pub struct ClientBuilder {
	path: PathBuf,
	home: Option<PathBuf>,
	expected_files: Option<Vec<ModelFile>>,
	model_snapshot_id: Option<ModelSnapshotId>,
	queue_capacity: usize,
	invalid_top_k: bool,
	cache: PromptCacheConfig,
	defaults: Defaults,
}

impl ClientBuilder {
	/// Start a builder for the checkpoint at `path`.
	pub fn new(path: impl Into<PathBuf>) -> Self {
		Self {
			path: path.into(),
			home: None,
			expected_files: None,
			model_snapshot_id: None,
			queue_capacity: 8,
			invalid_top_k: false,
			cache: PromptCacheConfig::default(),
			defaults: Defaults {
				max_tokens: 4096,
				context_tokens: 16_384,
				sampling: SamplingConfig::default(),
				enable_thinking: Some(false),
				reasoning_budget_tokens: None,
				prompt_cache: None,
				speculative_tokens: None,
			},
		}
	}

	/// Maximum queued inference jobs (default 8).
	///
	/// The currently running job is not counted. Values must be in `1..=64`.
	#[must_use]
	pub const fn queue_capacity(mut self, capacity: usize) -> Self {
		self.queue_capacity = capacity;
		self
	}

	/// Select the Emelex storage root for this process.
	///
	/// The first successfully initialized home wins until process exit.
	#[must_use]
	pub fn home(mut self, home: impl Into<PathBuf>) -> Self {
		self.home = Some(home.into());
		self
	}

	pub(crate) fn expected_files(mut self, files: &[ModelFile]) -> Self {
		self.expected_files = Some(files.to_vec());
		self
	}

	pub(crate) fn model_snapshot_id(mut self, snapshot_id: Option<ModelSnapshotId>) -> Self {
		self.model_snapshot_id = snapshot_id;
		self
	}

	/// Default maximum tokens generated per call (default 4096).
	#[must_use]
	pub const fn max_tokens(mut self, max_tokens: usize) -> Self {
		self.defaults.max_tokens = max_tokens;
		self
	}

	/// Maximum prompt plus generated tokens per request (default 16,384).
	///
	/// A lower architecture-declared model limit always wins.
	#[must_use]
	pub const fn context_tokens(mut self, context_tokens: usize) -> Self {
		self.defaults.context_tokens = context_tokens;
		self.cache.max_total_tokens = context_tokens;
		self
	}

	/// Default sampling temperature (default 0.0, greedy).
	#[must_use]
	pub const fn temperature(mut self, temperature: f32) -> Self {
		self.defaults.sampling.temperature = temperature;
		self
	}

	/// Default nucleus-sampling `top_p` (default 1.0).
	#[must_use]
	pub const fn top_p(mut self, top_p: f32) -> Self {
		self.defaults.sampling.top_p = top_p;
		self
	}

	/// Default `top_k` cutoff (default: none). Zero disables the cutoff.
	#[must_use]
	pub const fn top_k(mut self, top_k: u32) -> Self {
		self.invalid_top_k = top_k > i32::MAX as u32;
		#[expect(
			clippy::cast_possible_wrap,
			reason = "invalid values are retained only as a builder validation error"
		)]
		let value = top_k as i32;
		self.defaults.sampling.top_k = if self.invalid_top_k || value == 0 {
			None
		} else {
			Some(value)
		};
		self
	}

	/// Fixed sampling seed for reproducible generation.
	#[must_use]
	pub const fn seed(mut self, seed: u64) -> Self {
		self.defaults.sampling.seed = Some(seed);
		self
	}

	/// Opt into the model's "thinking" mode by default.
	#[must_use]
	pub const fn enable_thinking(mut self, enable: bool) -> Self {
		self.defaults.enable_thinking = Some(enable);
		self
	}

	/// Cap, in tokens, on reasoning spans before they are force-closed.
	///
	/// The budget draws from the same generation window as `max_tokens`:
	/// keep it comfortably below `max_tokens`, or the reply can be all
	/// thinking with no room left for the answer.
	#[must_use]
	pub const fn reasoning_budget_tokens(mut self, budget: usize) -> Self {
		self.defaults.reasoning_budget_tokens = Some(budget);
		self
	}

	/// Enable or disable KV prompt caching (default: enabled).
	#[must_use]
	pub const fn prompt_cache(mut self, enabled: bool) -> Self {
		self.defaults.prompt_cache = Some(enabled);
		self
	}

	/// Tokens drafted per round by MTP self-speculative decoding
	/// (default: off). `0` disables; values above 8 are rejected. Works only
	/// on checkpoints with a parity-certified MTP module - see
	/// [`Client::supports_mtp`]. Building fails when a non-zero default is
	/// requested for an uncertified checkpoint.
	#[must_use]
	pub const fn speculative_tokens(mut self, tokens: usize) -> Self {
		self.defaults.speculative_tokens = Some(tokens);
		self
	}

	/// Maximum entries held by the prompt-cache pool (default 16).
	#[must_use]
	pub const fn cache_max_entries(mut self, max_entries: usize) -> Self {
		self.cache.max_entries = max_entries;
		self
	}

	/// Aggregate prompt tokens whose MLX state may remain cached.
	#[must_use]
	pub const fn cache_max_tokens(mut self, max_total_tokens: usize) -> Self {
		self.cache.max_total_tokens = max_total_tokens;
		self
	}

	/// Idle time-to-live for prompt-cache entries (default 5 minutes).
	#[must_use]
	pub const fn cache_ttl(mut self, ttl: Duration) -> Self {
		self.cache.ttl = ttl;
		self
	}

	/// Minimum prompt length, in tokens, worth caching (default 8).
	#[must_use]
	pub const fn cache_min_tokens(mut self, min_cacheable_tokens: usize) -> Self {
		self.cache.min_cacheable_tokens = min_cacheable_tokens;
		self
	}

	/// Load the model and produce a [`Client`].
	///
	/// Spawns the client's dedicated inference thread, loads the
	/// checkpoint on it, and blocks until the load finishes (mirroring
	/// the previous synchronous-load behavior).
	pub fn build(mut self) -> Result<Client, Error> {
		validate_model_dir(&self.path)?;
		self.path = std::fs::canonicalize(&self.path).map_err(|error| Error::ModelPath {
			path: self.path.clone(),
			reason: format!("cannot canonicalize checkpoint: {error}"),
		})?;
		validate_builder(&self)?;
		if let Some(selected_home) = self.home.as_deref() {
			let home = EmelexHome::resolve(Some(selected_home))?;
			runtime::initialize(home.root())?;
		} else {
			#[cfg(test)]
			runtime::initialize_default_if_needed()?;
			#[cfg(not(test))]
			{
				let home = EmelexHome::resolve(None)?;
				runtime::initialize(home.root())?;
			}
		}
		let (job_tx, capabilities) = spawn_inference_worker(
			&self.path,
			self.queue_capacity,
			self.cache,
			self.expected_files,
		)?;
		if self
			.defaults
			.speculative_tokens
			.is_some_and(|tokens| tokens > 0)
			&& !capabilities.mtp
		{
			return Err(Error::CapabilityUnavailable {
				capability: "acceleration:mtp",
				reason: format!(
					"loaded checkpoint is not covered by {}",
					crate::engine::mtp_certification::IMPLEMENTATION_ID
				),
			});
		}
		if self.defaults.enable_thinking == Some(true) && !capabilities.chat.thinking_toggle {
			return Err(Error::CapabilityUnavailable {
				capability: "interaction:thinking_toggle",
				reason: "loaded chat template does not distinguish thinking enabled from disabled"
					.to_string(),
			});
		}
		Ok(Client {
			inner: Arc::new(Inner {
				jobs: job_tx,
				defaults: self.defaults,
				path: self.path,
				model_snapshot_id: self.model_snapshot_id,
				media_capabilities: capabilities.media,
				chat_capabilities: capabilities.chat,
				supports_mtp: capabilities.mtp,
			}),
		})
	}
}

fn spawn_inference_worker(
	path: &Path,
	queue_capacity: usize,
	cache: PromptCacheConfig,
	expected_files: Option<Vec<ModelFile>>,
) -> Result<(mpsc::SyncSender<Job>, LoadedCapabilities), Error> {
	let (job_tx, job_rx) = mpsc::sync_channel::<Job>(queue_capacity);
	let (ready_tx, ready_rx) = mpsc::channel();
	let load_path = path.to_path_buf();
	std::thread::Builder::new()
		.name("emelex-inference".to_string())
		.spawn(move || {
			let loaded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
				Session::load_with_cache_config_and_manifest(
					&load_path,
					cache,
					expected_files.as_deref(),
				)
			}));
			let session = match loaded {
				Ok(Ok(session)) => session,
				Ok(Err(error)) => {
					let _ = ready_tx.send(Err(error.to_string()));
					return;
				}
				Err(_) => {
					let _ = ready_tx.send(Err(
						"inference thread panicked while loading the model".to_string()
					));
					return;
				}
			};
			let chat = session.chat_template_capabilities();
			let capabilities = LoadedCapabilities {
				media: MediaCapabilities {
					images: session.supports_images(),
					audio: session.supports_audio(),
				},
				chat: ChatCapabilities {
					system_prompt: chat.system_prompt,
					tools: chat.tools,
					reasoning_history: chat.reasoning_history,
					thinking_toggle: chat.thinking_toggle,
				},
				mtp: session.supports_mtp(),
			};
			if ready_tx.send(Ok(capabilities)).is_err() {
				return;
			}
			while let Ok(job) = job_rx.recv() {
				// A panicking request cannot kill the inference thread for
				// every later caller. The job's reply channel reports failure.
				let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
					job(&session);
				}));
			}
			// Last Client clone dropped; Session dies on its owning thread.
		})
		.map_err(|error| Error::ModelPath {
			path: path.to_path_buf(),
			reason: format!("failed to spawn inference thread: {error}"),
		})?;
	let capabilities = ready_rx
		.recv()
		.map_err(|_| Error::ModelPath {
			path: path.to_path_buf(),
			reason: "inference thread died during model load".to_string(),
		})?
		.map_err(|message| Error::ModelLoad {
			path: path.to_path_buf(),
			message,
		})?;
	Ok((job_tx, capabilities))
}

fn validate_builder(builder: &ClientBuilder) -> Result<(), Error> {
	if builder.invalid_top_k {
		return Err(Error::InvalidConfiguration(format!(
			"top_k must be at most {}",
			i32::MAX
		)));
	}
	if builder.defaults.max_tokens == 0 {
		return Err(Error::InvalidConfiguration(
			"max_tokens must be positive".to_string(),
		));
	}
	if builder.defaults.max_tokens > 1 << 20 {
		return Err(Error::InvalidConfiguration(
			"max_tokens must be at most 1048576".to_string(),
		));
	}
	if !(1..=1 << 24).contains(&builder.defaults.context_tokens) {
		return Err(Error::InvalidConfiguration(
			"context_tokens must be in 1..=16777216".to_string(),
		));
	}
	if builder.defaults.max_tokens > builder.defaults.context_tokens {
		return Err(Error::InvalidConfiguration(
			"max_tokens cannot exceed context_tokens".to_string(),
		));
	}
	let temperature = builder.defaults.sampling.temperature;
	let top_p = builder.defaults.sampling.top_p;
	if !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) {
		return Err(Error::InvalidConfiguration(
			"temperature must be finite and in 0..=2".to_string(),
		));
	}
	if !top_p.is_finite() || !(0.0..=1.0).contains(&top_p) {
		return Err(Error::InvalidConfiguration(
			"top_p must be finite and in 0..=1".to_string(),
		));
	}
	if !(1..=64).contains(&builder.queue_capacity) {
		return Err(Error::InvalidConfiguration(
			"queue_capacity must be in 1..=64".to_string(),
		));
	}
	if builder.cache.max_entries == 0 {
		return Err(Error::InvalidConfiguration(
			"cache_max_entries must be positive".to_string(),
		));
	}
	if builder.cache.max_total_tokens == 0 {
		return Err(Error::InvalidConfiguration(
			"cache_max_tokens must be positive".to_string(),
		));
	}
	if builder.cache.ttl.is_zero() {
		return Err(Error::InvalidConfiguration(
			"cache_ttl must be positive".to_string(),
		));
	}
	if builder
		.defaults
		.speculative_tokens
		.is_some_and(|tokens| tokens > 8)
	{
		return Err(Error::InvalidConfiguration(
			"speculative_tokens must be at most 8".to_string(),
		));
	}
	if builder
		.defaults
		.reasoning_budget_tokens
		.is_some_and(|budget| budget == 0 || budget > builder.defaults.max_tokens)
	{
		return Err(Error::InvalidConfiguration(
			"reasoning_budget_tokens must be positive and not exceed max_tokens".to_string(),
		));
	}
	if builder.defaults.reasoning_budget_tokens.is_some()
		&& builder.defaults.enable_thinking != Some(true)
	{
		return Err(Error::InvalidConfiguration(
			"reasoning_budget_tokens requires thinking to be enabled".to_string(),
		));
	}
	Ok(())
}

fn validate_model_dir(path: &Path) -> Result<(), Error> {
	if !path.is_dir() {
		return Err(Error::ModelPath {
			path: path.to_path_buf(),
			reason: "not a directory".to_string(),
		});
	}
	if !path.join("config.json").is_file() {
		return Err(Error::ModelPath {
			path: path.to_path_buf(),
			reason: "no config.json (not an MLX checkpoint directory)".to_string(),
		});
	}
	Ok(())
}

#[cfg(all(test, feature = "rig"))]
mod tests {
	#![allow(clippy::unwrap_used, clippy::expect_used)]

	use super::*;

	/// A tiny-MTP-model client with a short token budget and speculation
	/// on, mirroring the engine-level prompt-cache reuse configuration.
	fn tiny_mtp_client() -> (crate::engine::test_support::TempModelDir, Client) {
		let dir = crate::engine::test_support::write_tiny_model(true).unwrap();
		let client = Client::builder(dir.path())
			.max_tokens(6)
			.speculative_tokens(2)
			.cache_min_tokens(0)
			.build()
			.expect("tiny model should load");
		assert!(client.supports_mtp());
		(dir, client)
	}

	#[tokio::test]
	async fn completion_carries_its_own_speculation_stats() {
		use rig_core::completion::CompletionModel as _;
		let (_dir, client) = tiny_mtp_client();
		let model = client.model();
		let request = model.completion_request("hello world").build();
		let response = model
			.completion(request)
			.await
			.expect("completion succeeds");
		let stats = response
			.raw_response
			.speculation
			.expect("a speculative completion must carry stats");
		assert!(stats.rounds >= 1, "spec-on tiny-model call never drafted");
		assert!(stats.drafted >= stats.accepted_by_depth.iter().sum::<u64>());
	}

	#[tokio::test]
	async fn final_streaming_response_carries_its_own_speculation_stats() {
		use futures::StreamExt as _;
		use rig_core::completion::CompletionModel as _;
		let (_dir, client) = tiny_mtp_client();
		let model = client.model();
		let request = model.completion_request("hello world").build();
		let mut stream = model.stream(request).await.expect("stream starts");
		let mut stats = None;
		while let Some(item) = stream.next().await {
			if let rig_core::streaming::StreamedAssistantContent::Final(response) =
				item.expect("stream item ok")
			{
				stats = response.speculation;
			}
		}
		let stats = stats.expect("a completed streaming generation must yield Some");
		assert!(stats.rounds >= 1, "spec-on tiny-model stream never drafted");
	}

	#[test]
	fn reasoning_params_write_a_complete_config() {
		assert_eq!(
			reasoning_params(false, None),
			serde_json::json!({ "enable_thinking": false })
		);
		assert_eq!(
			reasoning_params(true, Some(320)),
			serde_json::json!({
				"enable_thinking": true,
				"reasoning_budget_tokens": 320
			})
		);
	}
}

#[cfg(test)]
mod native_tests {
	#![allow(clippy::expect_used, clippy::unwrap_used)]

	use super::*;

	fn doctored_client(jobs: mpsc::SyncSender<Job>) -> Client {
		Client {
			inner: Arc::new(Inner {
				jobs,
				defaults: Defaults {
					max_tokens: 8,
					context_tokens: 64,
					sampling: SamplingConfig::default(),
					enable_thinking: None,
					reasoning_budget_tokens: None,
					prompt_cache: None,
					speculative_tokens: None,
				},
				path: PathBuf::new(),
				model_snapshot_id: None,
				media_capabilities: MediaCapabilities {
					images: false,
					audio: false,
				},
				chat_capabilities: ChatCapabilities {
					system_prompt: true,
					tools: true,
					reasoning_history: true,
					thinking_toggle: true,
				},
				supports_mtp: false,
			}),
		}
	}

	#[test]
	fn full_queue_is_reported_as_busy() {
		let (jobs, _receiver) = mpsc::sync_channel::<Job>(1);
		assert!(jobs.try_send(Box::new(|_| {})).is_ok());
		let client = doctored_client(jobs);
		assert!(matches!(
			client.stream(GenerationRequest::text("hello")),
			Err(Error::InferenceBusy)
		));
	}

	#[test]
	fn requests_fail_closed_when_chat_template_drops_authority_tools_or_reasoning() {
		let (jobs, _receiver) = mpsc::sync_channel::<Job>(1);
		let mut client = doctored_client(jobs);
		Arc::get_mut(&mut client.inner)
			.expect("unique test client")
			.chat_capabilities
			.system_prompt = false;
		assert!(matches!(
			client.stream(GenerationRequest::default().messages([
				crate::generation::Message::system("trusted policy"),
				crate::generation::Message::user("hello"),
			])),
			Err(Error::CapabilityUnavailable {
				capability: "interaction:system_prompt",
				..
			})
		));

		let (jobs, _receiver) = mpsc::sync_channel::<Job>(1);
		let mut client = doctored_client(jobs);
		Arc::get_mut(&mut client.inner)
			.expect("unique test client")
			.chat_capabilities
			.tools = false;
		assert!(matches!(
			client.stream(GenerationRequest::text("hello").tool(
				crate::generation::ToolDefinition::new(
					"lookup",
					"lookup",
					serde_json::json!({"type": "object"})
				)
			)),
			Err(Error::CapabilityUnavailable {
				capability: "interaction:tools",
				..
			})
		));

		let (jobs, _receiver) = mpsc::sync_channel::<Job>(1);
		let mut client = doctored_client(jobs);
		Arc::get_mut(&mut client.inner)
			.expect("unique test client")
			.chat_capabilities
			.reasoning_history = false;
		let mut reasoning = crate::generation::Message::assistant("prior answer");
		reasoning.reasoning = Some("prior reasoning".to_string());
		assert!(matches!(
			client.stream(GenerationRequest::default().messages([
				crate::generation::Message::user("first"),
				reasoning,
				crate::generation::Message::user("continue"),
			])),
			Err(Error::CapabilityUnavailable {
				capability: "interaction:reasoning_history",
				..
			})
		));

		let (jobs, _receiver) = mpsc::sync_channel::<Job>(1);
		let mut client = doctored_client(jobs);
		Arc::get_mut(&mut client.inner)
			.expect("unique test client")
			.chat_capabilities
			.thinking_toggle = false;
		assert!(matches!(
			client.stream(
				GenerationRequest::text("think").options(
					crate::generation::GenerationOptions::default()
						.thinking(crate::config::ThinkingMode::On),
				)
			),
			Err(Error::CapabilityUnavailable {
				capability: "interaction:thinking_toggle",
				..
			})
		));
	}

	#[test]
	fn tool_history_requires_matching_current_declarations_before_queueing() {
		let definition = crate::generation::ToolDefinition::new(
			"lookup",
			"lookup",
			serde_json::json!({"type": "object"}),
		);
		let call_id = uuid::Uuid::now_v7().to_string();
		let call = crate::generation::ToolCall {
			id: call_id.clone(),
			name: "lookup".to_string(),
			arguments: serde_json::json!({}),
		};
		let history = [
			crate::generation::Message::user("first"),
			crate::generation::Message {
				role: crate::generation::Role::Assistant,
				tool_calls: vec![call],
				..crate::generation::Message::default()
			},
			crate::generation::Message::tool(&call_id, "result"),
			crate::generation::Message::user("continue"),
		];

		let (jobs, receiver) = mpsc::sync_channel::<Job>(1);
		let client = doctored_client(jobs);
		assert!(matches!(
			client.stream(GenerationRequest::default().messages(history.clone())),
			Err(Error::InvalidRequest(message)) if message.contains("undeclared tool")
		));
		assert!(matches!(
			receiver.try_recv(),
			Err(mpsc::TryRecvError::Empty)
		));

		let (jobs, receiver) = mpsc::sync_channel::<Job>(1);
		let client = doctored_client(jobs);
		let mismatched = crate::generation::ToolDefinition::new(
			"other",
			"other",
			serde_json::json!({"type": "object"}),
		);
		assert!(matches!(
			client.stream(
				GenerationRequest::default()
					.messages(history)
					.tool(mismatched)
			),
			Err(Error::InvalidRequest(message)) if message.contains("undeclared tool")
		));
		assert!(matches!(
			receiver.try_recv(),
			Err(mpsc::TryRecvError::Empty)
		));

		let (jobs, _receiver) = mpsc::sync_channel::<Job>(1);
		let client = doctored_client(jobs);
		client
			.stream(
				GenerationRequest::default()
					.messages([
						crate::generation::Message::user("first"),
						crate::generation::Message {
							role: crate::generation::Role::Assistant,
							tool_calls: vec![crate::generation::ToolCall {
								id: call_id.clone(),
								name: "lookup".to_string(),
								arguments: serde_json::json!({}),
							}],
							..crate::generation::Message::default()
						},
						crate::generation::Message::tool(&call_id, "result"),
						crate::generation::Message::user("continue"),
					])
					.tool(definition),
			)
			.expect("matching current declaration reaches queue");
	}

	#[test]
	fn builder_rejects_cross_field_budget_errors_before_loading() {
		let builder = ClientBuilder::new("missing-model")
			.max_tokens(8)
			.reasoning_budget_tokens(9);
		assert!(matches!(
			validate_builder(&builder),
			Err(Error::InvalidConfiguration(_))
		));
	}

	#[test]
	fn stream_text_reconciliation_reveals_rejected_tool_markup_at_terminal() {
		for markup in [
			"<tool_call>{bad}</tool_call>",
			r#"<tool_call>{"name":"unknown","arguments":{}}</tool_call>"#,
			r#"<tool_call>{"name":"known","arguments":{"count":"wrong"}}</tool_call>"#,
		] {
			let mut visible = StreamTextReconciler::default();
			assert_eq!(
				visible.push_text("before ".to_string()).as_deref(),
				Some("before ")
			);
			visible.observe_tool_span();
			assert!(visible.push_text(" after".to_string()).is_none());

			let terminal = format!("before {markup} after");
			assert_eq!(
				visible.terminal_suffix(&terminal).expect("prefix"),
				format!("{markup} after")
			);
		}
	}

	#[test]
	fn stream_text_reconciliation_rejects_non_prefix_terminal_text() {
		let mut visible = StreamTextReconciler::default();
		assert_eq!(
			visible.push_text("streamed".to_string()).as_deref(),
			Some("streamed")
		);
		assert!(matches!(
			visible.terminal_suffix("different"),
			Err(Error::StreamProtocol(_))
		));
	}

	#[test]
	fn two_clients_can_load_and_probe_concurrently() {
		let model = crate::engine::test_support::write_tiny_model(false)
			.expect("tiny model fixture should be written");
		let first_path = model.path().to_path_buf();
		let second_path = first_path.clone();
		let results = std::thread::scope(|scope| {
			let first = scope.spawn(move || {
				let client = Client::builder(first_path)
					.max_tokens(1)
					.prompt_cache(false)
					.build()?;
				client.runtime_probe()
			});
			let second = scope.spawn(move || {
				let client = Client::builder(second_path)
					.max_tokens(1)
					.prompt_cache(false)
					.build()?;
				client.runtime_probe()
			});
			[
				first.join().expect("first client thread should not panic"),
				second
					.join()
					.expect("second client thread should not panic"),
			]
		});
		let no_metal_device = |result: &Result<(), Error>| {
			matches!(
				result,
				Err(Error::ModelLoad { message, .. })
					if message.contains("No Metal device")
						|| message.contains("no Metal device")
			)
		};
		if results.iter().all(no_metal_device) {
			// Both worker-local sessions completed their concurrent CPU load
			// path before reaching the same environmental GPU boundary.
			return;
		}
		for result in results {
			result.expect("both clients should complete native inference");
		}
	}
}
