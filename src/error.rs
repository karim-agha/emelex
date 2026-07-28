//! Public error type for the emelex provider surface.
//!
//! `Error` is what the crate's own API returns (`Client::from_path`,
//! `ClientBuilder::build`). Everything reached through rig's traits speaks
//! rig's `CompletionError` instead, via the `From` impl below.

use std::path::PathBuf;

#[cfg(feature = "rig")]
use rig_core::completion::CompletionError;

/// Errors surfaced by the emelex public API.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
	/// Emelex home resolution or preparation failed.
	#[error(transparent)]
	Home(#[from] crate::home::HomeError),
	/// Embedded MLX runtime initialization failed.
	#[error(transparent)]
	Runtime(#[from] crate::runtime::RuntimeError),
	/// The model directory is missing or is not a loadable checkpoint
	/// layout (no `config.json`).
	#[error("model directory {path:?} is not loadable: {reason}")]
	#[non_exhaustive]
	ModelPath {
		/// The offending directory.
		path: PathBuf,
		/// Why it cannot be loaded.
		reason: String,
	},
	/// The engine rejected the checkpoint while loading (unsupported
	/// `model_type`, bad quantization, weight or tokenizer failure).
	#[error("failed to load model at {path:?}: {message}")]
	#[non_exhaustive]
	ModelLoad {
		/// The model directory that failed to load.
		path: PathBuf,
		/// Engine-provided diagnostic text.
		message: String,
	},
	/// The request contains content the local engine cannot represent
	/// (e.g. URL-sourced media, provider file IDs).
	#[error("unsupported content: {0}")]
	UnsupportedContent(String),
	/// A native generation request violates API invariants.
	#[error("invalid generation request: {0}")]
	InvalidRequest(String),
	/// Client or toolkit configuration is invalid.
	#[error("invalid Emelex configuration: {0}")]
	InvalidConfiguration(String),
	/// Generation failed inside the engine.
	#[error("generation failed: {0}")]
	Generation(String),
	/// Incremental text was not an exact prefix of the terminal response.
	#[error("generation stream protocol failed: {0}")]
	StreamProtocol(String),
	/// Rendered prompt plus requested output exceeds effective context.
	#[error(
		"context window exceeded: prompt {prompt_tokens} + output {max_output_tokens} > {limit}"
	)]
	ContextExceeded {
		/// Rendered prompt token count.
		prompt_tokens: usize,
		/// Requested output-token ceiling.
		max_output_tokens: usize,
		/// Lower of configured and architecture-declared limits.
		limit: usize,
	},
	/// A requested optional capability is not verified for this model.
	#[error("capability {capability} is unavailable: {reason}")]
	#[non_exhaustive]
	CapabilityUnavailable {
		/// Namespaced capability key.
		capability: &'static str,
		/// Verification or support explanation.
		reason: String,
	},
	/// Cooperative request cancellation completed.
	#[error("generation cancelled")]
	Cancelled,
	/// Model emitted malformed tool-call syntax.
	#[error("malformed tool call: {0}")]
	MalformedToolCall(String),
	/// `additional_params` held knobs that failed to deserialize.
	#[error("invalid additional_params: {0}")]
	InvalidParams(#[from] serde_json::Error),
	/// A command could not be exchanged with the client's dedicated
	/// inference thread: the command channel rejected the send, or the reply
	/// channel disconnected before an answer arrived.
	#[error("inference channel {operation} failed")]
	#[non_exhaustive]
	InferenceChannel {
		/// Which half of the exchange failed: `"submit"` (command-channel
		/// send) or `"receive"` (reply-channel disconnect). No other
		/// values are produced.
		operation: &'static str,
	},
	/// The bounded inference queue has no admission capacity.
	#[error("inference queue is full")]
	InferenceBusy,
	/// One inference job panicked; the worker remains available.
	#[error("inference job panicked")]
	InferencePanic,
}

#[cfg(feature = "rig")]
impl From<Error> for CompletionError {
	fn from(error: Error) -> Self {
		match error {
			Error::UnsupportedContent(_)
			| Error::InvalidRequest(_)
			| Error::InvalidConfiguration(_)
			| Error::CapabilityUnavailable { .. }
			| Error::InvalidParams(_) => Self::RequestError(Box::new(error)),
			Error::Home(_)
			| Error::Runtime(_)
			| Error::ModelPath { .. }
			| Error::ModelLoad { .. }
			| Error::Generation(_)
			| Error::StreamProtocol(_)
			| Error::ContextExceeded { .. }
			| Error::Cancelled
			| Error::MalformedToolCall(_) => Self::ProviderError(error.to_string()),
			// Transport failure to the inference thread is a provider-side
			// fault. Keep its own arm so the mapping stays explicit and
			// individually reviewable.
			Error::InferenceChannel { .. } | Error::InferenceBusy | Error::InferencePanic => {
				Self::ProviderError(error.to_string())
			}
		}
	}
}

pub fn from_engine(error: crate::engine::error::Error) -> Error {
	match error {
		crate::engine::error::Error::ContextExceeded {
			prompt_tokens,
			max_output_tokens,
			limit,
		} => Error::ContextExceeded {
			prompt_tokens,
			max_output_tokens,
			limit,
		},
		crate::engine::error::Error::CapabilityUnavailable { capability, reason } => {
			Error::CapabilityUnavailable { capability, reason }
		}
		crate::engine::error::Error::Cancelled => Error::Cancelled,
		other => Error::Generation(other.to_string()),
	}
}

#[cfg(test)]
mod tests {
	#[cfg(feature = "rig")]
	use rig_core::completion::CompletionError;

	use super::*;

	#[test]
	fn engine_cancellation_preserves_public_error_identity() {
		assert!(matches!(
			from_engine(crate::engine::error::Error::Cancelled),
			Error::Cancelled
		));
	}

	/// The Display text is exactly `inference channel {operation} failed`.
	#[test]
	fn inference_channel_display_text_is_exact() {
		assert_eq!(
			Error::InferenceChannel {
				operation: "submit"
			}
			.to_string(),
			"inference channel submit failed"
		);
		assert_eq!(
			Error::InferenceChannel {
				operation: "receive"
			}
			.to_string(),
			"inference channel receive failed"
		);
	}

	/// The exhaustive `From<Error>` match maps `InferenceChannel` to
	/// `CompletionError::ProviderError`.
	#[test]
	#[cfg(feature = "rig")]
	fn inference_channel_converts_to_provider_error() {
		let converted = CompletionError::from(Error::InferenceChannel {
			operation: "receive",
		});
		assert!(matches!(
			converted,
			CompletionError::ProviderError(message)
				if message == "inference channel receive failed"
		));
	}
}
