//! The streaming bridge: engine per-token callback → bounded channel →
//! rig's raw streaming choices.
//!
//! Cancellation is cooperative: when the consumer drops the stream, the
//! channel receiver goes away; the engine observes that during media
//! preprocessing, between evaluated prefill chunks, or at the next token.

use rig_core::{
	completion::CompletionError,
	streaming::{RawStreamingChoice, RawStreamingToolCall, StreamingResult},
};

use super::response::StreamingResponse;
use crate::{
	client::StreamTextReconciler,
	convert::{self, EngineRequest},
	engine::{
		generate::{GenerateReply, Session},
		streaming::TokenKind,
	},
};

type Item = Result<RawStreamingChoice<StreamingResponse>, CompletionError>;

/// Run one generation on the client's dedicated inference thread,
/// bridging tokens into a stream.
pub(super) fn spawn(
	inner: &crate::client::Inner,
	engine_request: EngineRequest,
) -> StreamingResult<StreamingResponse> {
	let (tx, rx) = tokio::sync::mpsc::channel::<Item>(64);
	let job_tx = tx.clone();
	let submitted = inner.submit(Box::new(move |session| {
		run_job(session, &job_tx, engine_request);
	}));
	if let Err(reason) = submitted {
		// Inference thread is gone: surface it as the stream's only item.
		let _ = tx.try_send(Err(CompletionError::ProviderError(reason.to_string())));
	}
	drop(tx);
	Box::pin(futures::stream::unfold(rx, |mut rx| async move {
		rx.recv().await.map(|item| (item, rx))
	}))
}

fn run_job(session: &Session, tx: &tokio::sync::mpsc::Sender<Item>, engine_request: EngineRequest) {
	if tx.is_closed() {
		return;
	}
	let EngineRequest {
		messages,
		tools,
		options,
	} = engine_request;
	let mut visible_text = StreamTextReconciler::default();
	// The worker loop also catches panics, but catching here lets the panic
	// surface as an explicit stream error instead of silent truncation.
	let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
		let request_cancelled = || tx.is_closed();
		session.generate_cached_cancellable(
			&messages,
			tools.as_deref(),
			options,
			&request_cancelled,
			|token| {
				let choice = match token.kind {
					TokenKind::Text => {
						let Some(text) = visible_text.push_text(token.text) else {
							return !tx.is_closed();
						};
						if text.is_empty() {
							return !tx.is_closed();
						}
						RawStreamingChoice::Message(text)
					}
					TokenKind::Reasoning => RawStreamingChoice::ReasoningDelta {
						id: None,
						reasoning: token.text,
					},
					// Raw markup is emitted structurally only after terminal
					// validation. Keep cancellation observable meanwhile.
					TokenKind::ToolCall => {
						visible_text.observe_tool_span();
						return !tx.is_closed();
					}
				};
				if matches!(
					&choice,
					RawStreamingChoice::ReasoningDelta { reasoning, .. } if reasoning.is_empty()
				) {
					return !tx.is_closed();
				}
				tx.blocking_send(Ok(choice)).is_ok()
			},
		)
	}));
	match outcome {
		Ok(Ok(reply)) => send_completed(tx, &visible_text, &reply),
		Ok(Err(error)) => {
			let _ = tx.blocking_send(Err(CompletionError::from(crate::error::from_engine(error))));
		}
		Err(_) => {
			let _ = tx.blocking_send(Err(CompletionError::ProviderError(
				"generation panicked; the panic message was printed to stderr".to_string(),
			)));
		}
	}
}

fn send_completed(
	tx: &tokio::sync::mpsc::Sender<Item>,
	visible_text: &StreamTextReconciler,
	reply: &GenerateReply,
) {
	tracing::debug!(
		text_chars = reply.text.len(),
		tool_calls = reply.tool_calls.len(),
		finish = super::response::finish_reason_label(reply.finish_reason),
		"generation finished"
	);
	let suffix = match visible_text.terminal_suffix(&reply.text) {
		Ok(suffix) => suffix,
		Err(error) => {
			let _ = tx.blocking_send(Err(CompletionError::from(error)));
			return;
		}
	};
	if !suffix.is_empty()
		&& tx
			.blocking_send(Ok(RawStreamingChoice::Message(suffix.to_string())))
			.is_err()
	{
		return;
	}
	for call in &reply.tool_calls {
		// Ignore send failures from here on: the receiver being gone just
		// means nobody is listening anymore.
		let _ = tx.blocking_send(Ok(RawStreamingChoice::ToolCall(RawStreamingToolCall::new(
			call.id.clone(),
			call.name.clone(),
			call.arguments.clone(),
		))));
	}
	let response = StreamingResponse {
		usage: convert::usage_data(reply.usage),
		finish_reason: super::response::finish_reason_label(reply.finish_reason).to_string(),
		speculation: convert::speculation_data(reply.speculation.as_ref()),
	};
	let _ = tx.blocking_send(Ok(RawStreamingChoice::FinalResponse(response)));
}
