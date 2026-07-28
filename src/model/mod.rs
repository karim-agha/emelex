//! Model identity, compatibility, manifests, and optional Rig integration.
//!
//! Execution model: every generation is queued onto the client's dedicated
//! inference thread, so concurrent Rig calls serialize without moving MLX
//! state across OS threads. Request conversion happens before queue admission
//! so malformed requests fail before GPU work.

mod compatibility;
mod identity;
pub(crate) mod layout;
mod manifest;
mod response;
#[cfg(feature = "rig")]
mod stream;
mod traits;

pub use compatibility::{
	CompatibilityReport, FitReport, InspectionError, WorkloadError, WorkloadProfile,
	inspect_directory,
};
pub(crate) use compatibility::{estimate_fit_from_config, supported_model_type};
pub use identity::{
	HubModelId, LocalModelName, ModelRef, ModelRefError, ModelSnapshotId, ResolvedRevision,
	SnapshotDigest,
};
pub use manifest::{InstalledModel, ManifestError, ModelFile, ModelManifest, ModelSource};
pub use response::SpeculationStatsData;
#[cfg(feature = "rig")]
pub(crate) use response::finish_reason_label;
#[cfg(feature = "rig")]
pub use response::{Response, StreamingResponse, ToolCallData, UsageData};
pub use traits::{
	EvidenceSource, Modality, ModelGenerationDefaults, ModelSizing, ModelTraits, MtpSupport, Task,
	TraitConfidence, TraitEvidence, TraitFilter, TraitFilterError, TraitMetric, TraitPredicate,
	VerificationStatus,
};
#[cfg(feature = "rig")]
use {
	crate::{
		client::Client,
		convert::{self, Capabilities},
	},
	rig_core::{
		completion::{CompletionError, CompletionRequest, CompletionResponse},
		streaming::StreamingCompletionResponse,
	},
};

/// A rig completion model backed by one locally loaded MLX checkpoint.
#[cfg(feature = "rig")]
#[derive(Clone)]
pub struct CompletionModel {
	client: Client,
	/// Display name used for tracing only; the loaded checkpoint is fixed
	/// by the client.
	name: String,
}

#[cfg(feature = "rig")]
impl std::fmt::Debug for CompletionModel {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("CompletionModel")
			.field("name", &self.name)
			.field("client", &self.client)
			.finish()
	}
}

/// Aborts an in-flight blocking generation when the owning future is
/// dropped: the engine polls the flag during preprocessing, between
/// evaluated prefill chunks, and every generated token.
#[cfg(feature = "rig")]
struct CancelOnDrop(std::sync::Arc<std::sync::atomic::AtomicBool>);

#[cfg(feature = "rig")]
impl Drop for CancelOnDrop {
	fn drop(&mut self) {
		self.0.store(true, std::sync::atomic::Ordering::Release);
	}
}

#[cfg(feature = "rig")]
impl CompletionModel {
	pub(crate) fn from_client(client: &Client, name: String) -> Self {
		Self {
			client: client.clone(),
			name,
		}
	}

	fn capabilities(&self) -> Capabilities {
		Capabilities {
			images: self.client.supports_images(),
			audio: self.client.supports_audio(),
		}
	}
}

#[cfg(feature = "rig")]
impl rig_core::completion::CompletionModel for CompletionModel {
	type Client = Client;
	type Response = Response;
	type StreamingResponse = StreamingResponse;

	fn make(client: &Self::Client, model: impl Into<String>) -> Self {
		Self::from_client(client, model.into())
	}

	async fn completion(
		&self,
		request: CompletionRequest,
	) -> Result<CompletionResponse<Self::Response>, CompletionError> {
		let engine_request =
			convert::request(&request, self.capabilities(), &self.client.inner.defaults)
				.map_err(CompletionError::from)?;
		self.client
			.inner
			.validate_generation_capabilities(
				&engine_request.messages,
				engine_request.tools.as_deref().unwrap_or_default(),
				engine_request.options,
			)
			.map_err(CompletionError::from)?;
		tracing::debug!(model = %self.name, "starting local generation");
		// If this future is dropped mid-generation, the guard flips the
		// flag and the engine aborts at the next token, freeing the
		// inference thread for the next queued request.
		let cancelled = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
		let _cancel_guard = CancelOnDrop(std::sync::Arc::clone(&cancelled));
		let (done_tx, done_rx) = tokio::sync::oneshot::channel();
		let job_cancelled = std::sync::Arc::clone(&cancelled);
		self.client
			.inner
			.submit(Box::new(move |session| {
				if job_cancelled.load(std::sync::atomic::Ordering::Acquire) {
					return;
				}
				let request_cancelled = || job_cancelled.load(std::sync::atomic::Ordering::Acquire);
				let result = session.generate_cached_cancellable(
					&engine_request.messages,
					engine_request.tools.as_deref(),
					engine_request.options,
					&request_cancelled,
					|_| !job_cancelled.load(std::sync::atomic::Ordering::Acquire),
				);
				let _ = done_tx.send(result);
			}))
			.map_err(|reason| CompletionError::ProviderError(reason.to_string()))?;
		let reply = done_rx
			.await
			.map_err(|_| {
				CompletionError::ProviderError(
					"generation was dropped by the inference thread (it likely \
					 panicked; see stderr)"
						.to_string(),
				)
			})?
			.map_err(|error| CompletionError::from(crate::error::from_engine(error)))?;

		tracing::debug!(
			text_chars = reply.text.len(),
			tool_calls = reply.tool_calls.len(),
			finish = finish_reason_label(reply.finish_reason),
			"generation finished"
		);
		let usage = convert::usage_data(reply.usage);
		Ok(CompletionResponse {
			choice: convert::choice(&reply),
			usage: usage.to_rig(),
			raw_response: Response {
				text: reply.text,
				reasoning: reply.reasoning,
				tool_calls: reply
					.tool_calls
					.into_iter()
					.map(|call| ToolCallData {
						id: call.id,
						name: call.name,
						arguments: call.arguments,
					})
					.collect(),
				usage,
				finish_reason: finish_reason_label(reply.finish_reason).to_string(),
				speculation: convert::speculation_data(reply.speculation.as_ref()),
			},
			message_id: None,
		})
	}

	async fn stream(
		&self,
		request: CompletionRequest,
	) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
		let engine_request =
			convert::request(&request, self.capabilities(), &self.client.inner.defaults)
				.map_err(CompletionError::from)?;
		self.client
			.inner
			.validate_generation_capabilities(
				&engine_request.messages,
				engine_request.tools.as_deref().unwrap_or_default(),
				engine_request.options,
			)
			.map_err(CompletionError::from)?;
		let raw_stream = stream::spawn(&self.client.inner, engine_request);
		Ok(StreamingCompletionResponse::stream(raw_stream))
	}
}
