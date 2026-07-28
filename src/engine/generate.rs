//! High-level generation loop: prompt in, tokens out.

use std::{path::Path, sync::Mutex};

use serde_json::Value;

use crate::engine::{
	Cancellation,
	array::Array,
	error::{Error, Result},
	media::{
		ProcessedMediaBudget, ProcessedMediaKind, PromptBudget,
		audio::{
			ProcessedAudio, preprocess_audio_bytes_cancellable,
			preprocess_audio_bytes_raw_cancellable,
		},
		image::{ProcessedImage, preprocess_image_bytes_cancellable},
		video::extract_video_frames,
	},
	models::{
		Model,
		cache::LayerCache,
		mtp::{MtpCaches, MtpState},
	},
	ops,
	prompt_cache::{PromptCacheConfig, PromptCachePool, is_prefix},
	reasoning::{self, ReasoningBudget},
	sampling::{Sampler, SamplingConfig},
	spec,
	streaming::{StreamClassifier, TokenKind},
	tokenizer::{ChatMessage, ContentPart, Tokenizer},
	tools::{Tool, ToolCall, ToolCallFormat},
};

pub(crate) const MLX_FREED_BUFFER_CACHE_BYTES: u64 = 2 << 30;
const PREFILL_CHUNK_TOKENS: usize = 512;

/// A loaded model + tokenizer pair, ready to generate.
///
/// Prompt caching is stateless from the caller's perspective (mirroring
/// the OpenAI/Anthropic chat APIs): [`Session::generate_cached`] takes the
/// *full* message list on every call rather than a session handle, and an
/// internal [`PromptCachePool`] transparently reuses KV state for whatever
/// prefix (if any) a previous call already computed - see
/// `crate::engine::prompt_cache` for the pool's eviction/matching semantics.
pub struct Session {
	model: Model,
	tokenizer: Tokenizer,
	prompt_cache: Mutex<PromptCachePool>,
	model_context_limit: Option<usize>,
	chat_template_capabilities: crate::engine::tokenizer::ChatTemplateCapabilities,
	tool_call_format: crate::engine::tools::ToolCallFormat,
	/// `true` only when the loaded MTP module belongs to the exact
	/// checkpoint bytes covered by Emelex's checked-in parity certificate.
	mtp_certified: bool,
	/// emelex patch (not upstream): one-shot failure hook for the MTP
	/// priming helpers ([`Session::prime_mtp`] /
	/// [`Session::prime_mtp_resume`]) - real MLX detach faults cannot be
	/// phase-targeted, so the explicit-request failure path is driven
	/// through this seam. Compiled out of production builds.
	#[cfg(test)]
	priming_fault: std::sync::atomic::AtomicBool,
	/// Number of MTP prefill chunks whose cache-producing graph has reached
	/// the native execution boundary. Deterministic cancellation seam only.
	#[cfg(test)]
	mtp_prefill_materialized_chunks: std::sync::atomic::AtomicUsize,
}

#[derive(Debug, Clone, Copy)]
enum MtpCertificatePolicy {
	Exact,
	#[cfg(test)]
	SyntheticFixture,
}

/// One step of streamed generation.
pub struct GeneratedToken {
	pub id: u32,
	pub text: String,
	pub finished: bool,
	/// Which span this token belongs to (plain text, reasoning, or a
	/// raw, not-yet-parsed tool-call span) - see [`crate::engine::streaming`].
	/// Best-effort: a marker straddling two tokens is still detected,
	/// but only once its second half arrives.
	pub kind: TokenKind,
}

/// Token accounting for one [`Session::generate_cached`] call, mirroring
/// the `usage` block OpenAI/Anthropic return alongside a chat completion.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
	/// Total input tokens for this call's fully-rendered prompt (the sum
	/// of `cached_tokens` and however many had to be freshly computed).
	pub prompt_tokens: usize,
	/// How many of `prompt_tokens` were served from the prompt-cache pool
	/// (an exact-prefix hit) rather than run through the model this call.
	pub cached_tokens: usize,
	/// Tokens generated in this call's reply.
	pub completion_tokens: usize,
}

/// Why one [`Session::generate_cached`] call stopped generating,
/// mirroring OpenAI's `finish_reason` / Anthropic's `stop_reason`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum FinishReason {
	/// The model emitted an end-of-sequence token - a natural end of
	/// turn.
	#[default]
	Stop,
	/// [`GenerateOptions::max_tokens`] was exhausted before the model
	/// finished its reply.
	Length,
	/// The reply issued one or more tool calls (see
	/// [`GenerateReply::tool_calls`]).
	ToolCalls,
	/// The caller's `on_token` callback stopped generation early by
	/// returning `false`.
	Aborted,
}

/// Classify why generation stopped, given what the decode loop produced.
/// Tool calls take precedence (the natural "end" of a tool-calling turn,
/// whether or not an eos token followed), then a trailing eos token,
/// then an explicit caller abort; anything else means the token budget
/// ran out.
fn classify_finish(
	generated: &[u32],
	eos_ids: &[u32],
	has_tool_calls: bool,
	aborted: bool,
) -> FinishReason {
	if has_tool_calls {
		FinishReason::ToolCalls
	} else if generated.last().is_some_and(|id| eos_ids.contains(id)) {
		FinishReason::Stop
	} else if aborted {
		FinishReason::Aborted
	} else {
		FinishReason::Length
	}
}

/// emelex patch (not upstream): what [`Session::decode_loop`] hands back.
/// Two ledgers replace the old `(ids, last_unfed)` pair:
///
/// - `emitted` is the consumer-facing sequence - every token delivered through
///   the `on_token` callback, in order. `classify_finish`, usage accounting,
///   reply-text assembly, and streaming all read this ledger.
/// - `committed_len` indexes into `emitted`: the prefix whose tokens the caches
///   actually contain KV/state for. The prompt-cache pool reads this ledger and
///   nothing else.
///
/// Invariant: the committed sequence is always a prefix of `emitted`
/// (`committed_len <= emitted.len()`). The two diverge on purpose: an EOS
/// or cancellation token is emitted but never fed, and a forced-close
/// cancellation leaves the cancelled close token emitted while the cache
/// holds only the accepted close prefix.
pub(crate) struct DecodeOutcome {
	pub emitted: Vec<u32>,
	pub committed_len: usize,
	/// Speculation accounting; `Some` iff the call drafted or decided at
	/// least one speculative round (`rounds > 0 || drafted > 0` -
	/// `drafted` counts at draft time, so a round that failed before its
	/// decision still surfaces its proposals).
	pub speculation: Option<SpeculationStats>,
	/// Final MTP state when the module survived the call: the working
	/// caches aligned to the committed prefix plus the detached frontier
	/// hidden — the prompt-cache pool handoff and the exact-offset
	/// assertion surface for engine-level tests.
	pub mtp: Option<MtpState>,
	/// Whether the reasoning budget injected a complete close marker and
	/// generation continued. That teacher-forced boundary is authoritative;
	/// terminal extraction may suppress only one bounded immediate duplicate
	/// model close.
	pub reasoning_forced_closed: bool,
}

/// emelex patch (not upstream): outcome of pushing one token through
/// [`TokenEmitter`].
pub(crate) enum Emit {
	/// Token emitted; generation continues. Carries the raw (special-token
	/// preserving) decode for [`ReasoningBudget::observe`].
	Continue { raw_text: String },
	/// Token emitted and it is a registered stop token.
	Eos,
	/// The `on_token` callback declined further tokens.
	Cancelled,
}

/// emelex patch (not upstream): the per-token display/classification
/// pipeline, extracted from the decode loop so every path that emits
/// tokens - the ordinary decode step, the forced-close path, and the
/// speculative accepted-block path - runs the identical state machine
/// (UTF-8 withholding, marker classification, stop-token strip,
/// bounded immediate forced-close duplicate filtering, callback cancellation).
///
/// Ordering contract: the fallible decodes run BEFORE the token is pushed
/// to `emitted` and before the callback, so a decode error leaves the
/// token unemitted with zero callbacks - the exact-prefix exit contract.
pub(crate) struct TokenEmitter<'a, F: FnMut(GeneratedToken) -> bool> {
	tokenizer: &'a Tokenizer,
	eos_ids: &'a [u32],
	classifier: StreamClassifier,
	display_decoder: StreamDecoder,
	/// Pending raw/display bytes while deciding whether the model immediately
	/// duplicated a teacher-forced close.
	orphan_close: Option<(&'static str, String, String)>,
	emitted: Vec<u32>,
	on_token: F,
	callback_open: bool,
	/// Fault-injection hook: `Some(n)` makes the emit that would push
	/// `emitted[n]` fail instead, with zero callbacks for that token.
	#[cfg(test)]
	fail_at: Option<usize>,
	/// Test hook: overrides [`TokenEmitter::encode_close`]'s encoding so
	/// the empty-close-encoding forced-close branches are drivable on
	/// tokenizers whose close marker never encodes empty.
	#[cfg(test)]
	close_override: Option<Vec<u32>>,
}

impl<'a, F: FnMut(GeneratedToken) -> bool> TokenEmitter<'a, F> {
	/// emelex patch: constructor shared by `decode_loop` and the spec.rs
	/// fake suite (which runs the round driver over the committed
	/// tiny-model fixture tokenizer).
	pub(crate) fn new(
		tokenizer: &'a Tokenizer,
		eos_ids: &'a [u32],
		classifier: StreamClassifier,
		max_tokens: usize,
		on_token: F,
	) -> Self {
		TokenEmitter {
			tokenizer,
			eos_ids,
			classifier,
			display_decoder: StreamDecoder::default(),
			orphan_close: None,
			// Clamp the pre-allocation - a caller-supplied huge max_tokens
			// must not abort the process in Vec::with_capacity.
			emitted: Vec::with_capacity(max_tokens.min(4096)),
			on_token,
			callback_open: true,
			#[cfg(test)]
			fail_at: None,
			#[cfg(test)]
			close_override: None,
		}
	}

	pub(crate) fn emitted(&self) -> &[u32] {
		&self.emitted
	}

	pub(crate) fn into_emitted(mut self) -> Vec<u32> {
		self.flush_terminal();
		self.emitted
	}

	/// Encode a forced-close marker. The driver owns forced close for
	/// every mode, so the tokenizer rides along inside the emitter.
	pub(crate) fn encode_close(&self, marker: &str) -> Result<Vec<u32>> {
		#[cfg(test)]
		if let Some(ids) = &self.close_override {
			return Ok(ids.clone());
		}
		self.tokenizer.encode(marker)
	}

	/// Arm the immediate-duplicate filter after a teacher-forced close.
	pub(crate) fn arm_close_filters(&mut self, close_marker: &'static str) {
		self.orphan_close = Some((close_marker, String::new(), String::new()));
	}

	#[cfg(test)]
	pub(crate) fn set_fail_at(&mut self, at: Option<usize>) {
		self.fail_at = at;
	}

	#[cfg(test)]
	pub(crate) fn set_close_override(&mut self, ids: Option<Vec<u32>>) {
		self.close_override = ids;
	}
}

impl<F: FnMut(GeneratedToken) -> bool> TokenEmitter<'_, F> {
	/// Emit one ordinarily-generated token.
	pub(crate) fn emit(&mut self, id: u32) -> Result<Emit> {
		#[cfg(test)]
		if self.fail_at == Some(self.emitted.len()) {
			return Err(Error::Model("token emitter test fault".into()));
		}
		let finished = self.eos_ids.contains(&id);
		let raw_text = self.tokenizer.decode_piece_raw(id)?;
		let mut decoded = self.display_decoder.next(self.tokenizer, id, &raw_text)?;
		if finished && decoded.is_none() {
			decoded = self.display_decoder.finish();
		}
		// emelex patch: a registered stop token never carries reply
		// text, but some checkpoints (Laguna's `</assistant>`, id 24)
		// register theirs as a *non-special* vocab entry that the
		// skip-special display decode would leak verbatim. Strip
		// exactly the stop token's own text, keeping any pending
		// UTF-8 remnant the decoder flushed alongside it.
		if finished
			&& let Some(piece) = decoded.as_mut()
			&& let Some(rest) = piece.display.strip_suffix(raw_text.as_str())
		{
			piece.display = rest.to_string();
		}
		self.emitted.push(id);
		let keep_going = match decoded {
			Some(piece) => self.emit_decoded(id, finished, piece, true),
			None => self.send_empty(id, finished),
		};
		if finished {
			return Ok(Emit::Eos);
		}
		if !keep_going {
			return Ok(Emit::Cancelled);
		}
		Ok(Emit::Continue { raw_text })
	}

	/// Emit one teacher-forced close-marker token. Forced tokens share
	/// UTF-8 decoding and marker classification with ordinary tokens, but
	/// never pass through the post-budget immediate-duplicate filter.
	pub(crate) fn emit_forced(&mut self, id: u32) -> Result<Emit> {
		#[cfg(test)]
		if self.fail_at == Some(self.emitted.len()) {
			return Err(Error::Model("token emitter test fault".into()));
		}
		let raw_piece = self.tokenizer.decode_piece_raw(id)?;
		let decoded = self.display_decoder.next(self.tokenizer, id, &raw_piece)?;
		self.emitted.push(id);
		let keep_going = match decoded {
			Some(piece) => self.emit_decoded(id, false, piece, false),
			None => self.send_empty(id, false),
		};
		if keep_going {
			Ok(Emit::Continue {
				raw_text: raw_piece,
			})
		} else {
			Ok(Emit::Cancelled)
		}
	}

	fn emit_decoded(
		&mut self,
		id: u32,
		finished: bool,
		piece: DecodedPiece,
		apply_close_filters: bool,
	) -> bool {
		let display = if apply_close_filters {
			self.filter_post_budget_closes(&piece.raw, piece.display)
		} else {
			piece.display
		};
		let segments = self.classifier.push(&piece.raw, &display);
		self.send_segments(id, finished, segments)
	}

	fn filter_post_budget_closes(&mut self, raw: &str, display: String) -> String {
		// After a budget-forced close, suppress at most one immediate duplicate
		// model close. While undecided, withhold only a whitespace-padded marker
		// prefix under the shared eight-byte bound. Any divergence flushes every
		// withheld display byte, making the teacher-forced boundary
		// authoritative and delayed closes literal answer text.
		let display = if let Some((close, mut raw_buffer, mut display_buffer)) =
			self.orphan_close.take()
		{
			raw_buffer.push_str(raw);
			display_buffer.push_str(&display);
			let candidate = raw_buffer.trim_start();
			let leading_bytes = raw_buffer.len() - candidate.len();
			if leading_bytes <= reasoning::MAX_FORCED_CLOSE_WHITESPACE_BYTES
				&& candidate.starts_with(close)
			{
				let marker_at = leading_bytes;
				display_after_forced_close(&raw_buffer, &display_buffer, marker_at, close)
			} else if (candidate.is_empty() || close.starts_with(candidate))
				&& leading_bytes <= reasoning::MAX_FORCED_CLOSE_WHITESPACE_BYTES
				&& raw_buffer.len() <= close.len() + reasoning::MAX_FORCED_CLOSE_WHITESPACE_BYTES
			{
				self.orphan_close = Some((close, raw_buffer, display_buffer));
				String::new()
			} else {
				// Diverged: not an immediate duplicate. Emit everything that
				// was withheld through the display decoder so split
				// multi-byte text and partial literal markers stay lossless.
				display_buffer
			}
		} else {
			display
		};
		display
	}

	fn send_segments(
		&mut self,
		id: u32,
		finished: bool,
		segments: Vec<crate::engine::streaming::ClassifiedText>,
	) -> bool {
		if segments.is_empty() {
			return self.send_empty(id, finished);
		}
		let last = segments.len() - 1;
		for (index, segment) in segments.into_iter().enumerate() {
			if !(self.on_token)(GeneratedToken {
				id,
				text: segment.text,
				finished: finished && index == last,
				kind: segment.kind,
			}) {
				self.callback_open = false;
				return false;
			}
		}
		true
	}

	fn send_empty(&mut self, id: u32, finished: bool) -> bool {
		let keep_going = (self.on_token)(GeneratedToken {
			id,
			text: String::new(),
			finished,
			kind: self.classifier.current_kind(),
		});
		if !keep_going {
			self.callback_open = false;
		}
		keep_going
	}

	fn flush_terminal(&mut self) {
		if !self.callback_open {
			return;
		}
		let id = self.emitted.last().copied().unwrap_or_default();
		if let Some(piece) = self.display_decoder.finish()
			&& !self.emit_decoded(id, false, piece, true)
		{
			return;
		}
		if let Some((_close, raw, display)) = self.orphan_close.take() {
			let segments = self.classifier.push(&raw, &display);
			if !self.send_segments(id, false, segments) {
				return;
			}
		}
		let segments = self.classifier.finish();
		let _ = self.send_segments(id, false, segments);
	}
}

fn display_after_forced_close(raw: &str, display: &str, marker_at: usize, close: &str) -> String {
	if raw == display {
		return display[marker_at + close.len()..].to_string();
	}
	let raw_prefix = &raw[..marker_at];
	let visible = display.strip_prefix(raw_prefix).unwrap_or(display);
	let raw_after = &raw[marker_at + close.len()..];
	if visible == raw_after {
		// The tokenizer omitted the special close marker from display.
		return visible.to_string();
	}
	visible.strip_prefix(close).unwrap_or(visible).to_string()
}

/// Result of one [`Session::generate_cached`] call.
#[derive(Debug, Clone, Default)]
pub struct GenerateReply {
	/// The final answer text, with any reasoning span (see `reasoning`)
	/// already stripped out.
	pub text: String,
	pub tool_calls: Vec<ToolCall>,
	pub usage: Usage,
	/// Extracted reasoning/"thinking" content, if the model emitted a
	/// recognized reasoning span (`<think>...</think>` or Gemma4's
	/// `<|channel>thought...<channel|>`) - present regardless of whether
	/// `enable_thinking` was explicitly requested, since some checkpoints
	/// reason unconditionally. See [`crate::engine::reasoning`].
	pub reasoning: Option<String>,
	/// Why generation stopped - a natural end of turn, the token budget,
	/// a tool call, or a caller-initiated abort.
	pub finish_reason: FinishReason,
	/// emelex patch (not upstream): speculative-decoding accounting for
	/// this call; `None` when speculation never ran.
	pub speculation: Option<SpeculationStats>,
}

/// Generation parameters for a single call.
#[derive(Debug, Clone, Copy)]
pub struct GenerateOptions {
	pub max_tokens: usize,
	/// Configured context ceiling. The architecture-declared ceiling,
	/// when lower, wins inside [`Session`].
	pub context_tokens: usize,
	pub sampling: SamplingConfig,
	/// Opt into a model's "thinking" mode via its chat template's
	/// `enable_thinking` variable (Qwen3/3.5/3.6, Gemma4, MiniCPM5,
	/// NemotronH, ...; see `crate::engine::reasoning`). `Some(true)` opts in.
	/// `None` and `Some(false)` both resolve to an explicit false template
	/// variable so hybrid checkpoints cannot silently enable reasoning.
	pub enable_thinking: Option<bool>,
	/// Cap, in tokens, on how long the model may spend inside a detected
	/// reasoning span (`<think>...</think>` or Gemma4's
	/// `<|channel>thought...<channel|>`) before it is force-closed and
	/// generation moves on to the final answer - mirroring Anthropic's
	/// extended-thinking `budget_tokens`. `None` means no cap. Has no
	/// effect if the model never opens a recognized reasoning span.
	pub reasoning_budget_tokens: Option<usize>,
	/// Whether [`Session::generate_cached`] may reuse (and store) KV state
	/// in the session's [`PromptCachePool`]. `None`/`Some(true)` keeps
	/// caching on (the default); `Some(false)` runs this call fully cold
	/// and leaves the pool untouched - useful when the caller wants
	/// deterministic from-scratch prefill or to bound memory held by
	/// cached KV state.
	pub prompt_cache: Option<bool>,
	/// emelex patch (not upstream): how many tokens MTP self-speculative
	/// decoding drafts per round. `None` (the default) disables
	/// speculation; option resolution normalizes `Some(0)` to `None` and
	/// clamps to 8. A non-zero request fails when the checkpoint is not
	/// covered by Emelex's checked-in MTP parity certificate.
	pub speculative_tokens: Option<usize>,
}

impl Default for GenerateOptions {
	fn default() -> Self {
		GenerateOptions {
			max_tokens: 256,
			context_tokens: 16_384,
			sampling: SamplingConfig::default(),
			enable_thinking: None,
			reasoning_budget_tokens: None,
			prompt_cache: None,
			speculative_tokens: None,
		}
	}
}

/// emelex patch (not upstream): resolve `GenerateOptions.speculative_tokens`
/// into the effective per-round draft depth: `Some(0)` normalizes to
/// `None` and requests clamp to [`SPECULATIVE_TOKENS_CEILING`].
pub(crate) fn resolve_speculative_tokens(options: &GenerateOptions) -> Option<usize> {
	options
		.speculative_tokens
		.filter(|&k| k > 0)
		.map(|k| k.min(SPECULATIVE_TOKENS_CEILING))
}

/// emelex patch (not upstream): ceiling on the MTP draft depth per
/// speculative round. Option resolution clamps request values here, and
/// the decode round re-derives `k = min(config_k, SPECULATIVE_TOKENS_CEILING,
/// remaining - 1)` as defense in depth - one constant so the two cannot
/// drift.
pub const SPECULATIVE_TOKENS_CEILING: usize = 8;

/// emelex patch (not upstream): per-call accounting for MTP
/// self-speculative decoding. Absent (`None` on [`GenerateReply`]) when
/// the checkpoint has no MTP module or speculation was disabled.
///
/// Depth indexing is one-based: `accepted_by_depth[i]`
/// counts rounds that accepted exactly `i + 1` draft tokens, so a
/// round accepting one draft lands at index 0. Full-rejection rounds
/// (`accepted == 0`) increment no bucket - `rounds -
/// sum(accepted_by_depth)` is the full-rejection count. All counters
/// use saturating addition.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpeculationStats {
	/// Total draft tokens proposed across the call, counted AT DRAFT TIME
	/// (the moment a draft token is drawn), not per decided round -
	/// proposals from rounds that later fail (verify-phase host failure,
	/// invariant-Err recovery, mid-draft MTP failure) still count, so
	/// `drafted` can exceed the sum a round-level ledger would imply and
	/// can be positive while `rounds == 0`.
	pub drafted: u64,
	/// Index `i` counts rounds whose accepted prefix length was exactly
	/// `i + 1` (one-based depth; length = max observed depth,
	/// zero-filled). Full rejections increment no bucket.
	pub accepted_by_depth: Vec<u64>,
	/// Speculative rounds run (a round = one draft + verify cycle).
	pub rounds: u64,
}

impl SpeculationStats {
	/// emelex patch (not upstream): count `n` draft tokens at DRAFT time
	/// (tokens proposed - failed rounds' proposals count). Saturating
	/// Proposals from failed rounds still count.
	pub(crate) fn record_drafted(&mut self, n: usize) {
		self.drafted = self.drafted.saturating_add(n as u64);
	}

	/// emelex patch (not upstream): the round/depth counter-increment
	/// site. Records one DECIDED draft + verify round with `accepted`
	/// drafts accepted (`drafted` is counted separately, at draft time -
	/// see [`SpeculationStats::record_drafted`]). Depth-1 acceptances
	/// land at index 0; a full rejection (`accepted == 0`) increments no
	/// `accepted_by_depth` bucket, so `sum(accepted_by_depth) <= rounds`.
	/// Uses saturating addition throughout.
	pub(crate) fn record_round(&mut self, accepted: usize) {
		self.rounds = self.rounds.saturating_add(1);
		if let Some(index) = accepted.checked_sub(1) {
			if self.accepted_by_depth.len() <= index {
				self.accepted_by_depth.resize(index + 1, 0);
			}
			self.accepted_by_depth[index] = self.accepted_by_depth[index].saturating_add(1);
		}
	}
}

/// Some checkpoints stop generation on more than one token id (e.g. a
/// dedicated end-of-turn token in addition to the tokenizer's primary
/// `eos_token`). `Tokenizer::load` only registers the latter, so this
/// folds in every `eos_token_id` (scalar or list form) declared in
/// `config.json` (top-level or nested `text_config`) and
/// `generation_config.json` - without this, checkpoints whose model
/// actually prefers an alternate stop id keep generating past the
/// intended end of the turn until `max_tokens` is hit.
fn register_extra_eos_ids(
	config: &Value,
	generation_config_bytes: Option<&[u8]>,
	tokenizer: &mut Tokenizer,
) -> Result<()> {
	let collect_ids = |v: &Value, out: &mut Vec<u32>| -> Result<()> {
		let mut push = |id: u64| -> Result<()> {
			out.push(u32::try_from(id).map_err(|_| {
				Error::Config(format!("eos_token_id {id} does not fit unsigned 32-bit"))
			})?);
			Ok(())
		};
		match v {
			Value::Number(n) => {
				if let Some(id) = n.as_u64() {
					push(id)?;
				}
			}
			Value::Array(items) => {
				for item in items {
					if let Some(id) = item.as_u64() {
						push(id)?;
					}
				}
			}
			_ => {}
		}
		Ok(())
	};

	let mut ids = Vec::new();
	if let Some(v) = config.get("eos_token_id") {
		collect_ids(v, &mut ids)?;
	}
	if let Some(v) = config
		.get("text_config")
		.and_then(|text| text.get("eos_token_id"))
	{
		collect_ids(v, &mut ids)?;
	}
	if let Some(gen_config) = generation_config_bytes
		.map(serde_json::from_slice::<Value>)
		.transpose()
		.map_err(|error| Error::Config(format!("bad generation_config.json: {error}")))?
	{
		if let Some(v) = gen_config.get("eos_token_id") {
			collect_ids(v, &mut ids)?;
		}
	}
	for id in ids {
		tokenizer.add_eos_id(id);
	}
	Ok(())
}

fn declared_context_limit(config: &Value) -> Result<Option<usize>> {
	let text = config.get("text_config").unwrap_or(config);
	let limit = [
		"max_position_embeddings",
		"max_sequence_length",
		"seq_length",
		"model_max_length",
	]
	.into_iter()
	.filter_map(|key| text.get(key).and_then(Value::as_u64))
	.min()
	.map(|value| {
		usize::try_from(value)
			.map_err(|_| Error::Config("model context limit exceeds usize".to_string()))
	})
	.transpose()?;
	Ok(limit.filter(|value| *value > 0))
}

fn media_soft_tokens(value: i32) -> Result<usize> {
	usize::try_from(value).map_err(|_| {
		Error::Model("processed media produced a negative soft-token count".to_string())
	})
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MediaBindingKind {
	Image,
	Audio,
	Video,
}

#[derive(Debug, Clone, Copy, Default)]
struct MediaPlaceholderIds {
	image: Option<u32>,
	audio: Option<u32>,
	video: Option<u32>,
}

fn content_part_media_kind(part: &ContentPart) -> Option<MediaBindingKind> {
	match part {
		ContentPart::Image(_) => Some(MediaBindingKind::Image),
		ContentPart::Audio(_) => Some(MediaBindingKind::Audio),
		ContentPart::Video(_) => Some(MediaBindingKind::Video),
		ContentPart::Text(_) => None,
	}
}

fn validate_media_bindings(
	parts: &[&ContentPart],
	prompt_ids: &[u32],
	ids: MediaPlaceholderIds,
) -> Result<()> {
	let attachments = parts
		.iter()
		.filter_map(|part| content_part_media_kind(part))
		.collect::<Vec<_>>();
	let known = [
		(MediaBindingKind::Image, ids.image),
		(MediaBindingKind::Audio, ids.audio),
		(MediaBindingKind::Video, ids.video),
	];
	for (index, (left_kind, left_id)) in known.iter().enumerate() {
		let Some(left_id) = left_id else {
			continue;
		};
		for (right_kind, right_id) in &known[index + 1..] {
			if Some(*left_id) == *right_id
				&& (prompt_ids.contains(left_id)
					|| attachments.contains(left_kind)
					|| attachments.contains(right_kind))
			{
				return Err(Error::Model(format!(
					"ambiguous multimodal placeholder token {left_id}: {left_kind:?} and \
					 {right_kind:?} use the same ID"
				)));
			}
		}
	}
	let placeholders = prompt_ids
		.iter()
		.filter_map(|token| {
			known.iter().find_map(|(kind, id)| {
				if id == &Some(*token) {
					Some(*kind)
				} else {
					None
				}
			})
		})
		.collect::<Vec<_>>();
	if attachments != placeholders {
		return Err(Error::Model(format!(
			"rendered multimodal placeholders do not exactly match attachments: attachments \
			 {attachments:?}, placeholders {placeholders:?}"
		)));
	}
	Ok(())
}

fn run_prefill_chunks<T>(
	prompt_ids: &[u32],
	cancellation: Cancellation<'_>,
	mut forward: impl FnMut(&[u32], bool) -> Result<T>,
) -> Result<T> {
	if prompt_ids.is_empty() {
		return Err(Error::Model(
			"cannot prefill an empty prompt suffix".to_string(),
		));
	}
	let chunk_tokens = if cancellation.is_cooperative() {
		PREFILL_CHUNK_TOKENS
	} else {
		prompt_ids.len()
	};
	let chunk_count = prompt_ids.len().div_ceil(chunk_tokens);
	let mut output = None;
	for (index, chunk) in prompt_ids.chunks(chunk_tokens).enumerate() {
		cancellation.checkpoint()?;
		output = Some(forward(chunk, index + 1 == chunk_count)?);
		// MLX work is evaluated by the closure before this boundary. A drop
		// observed here prevents construction of the next chunk's graph.
		cancellation.checkpoint()?;
	}
	output.ok_or_else(|| Error::Model("cannot prefill an empty prompt suffix".to_string()))
}

fn eval_last_logits(logits: &Array) -> Result<()> {
	let shape = logits.shape();
	if shape.len() != 3 || shape[0] <= 0 || shape[1] <= 0 || shape[2] <= 0 {
		return Err(Error::Model(format!(
			"prefill produced invalid logits shape {shape:?}"
		)));
	}
	let last = ops::slice(
		logits,
		&[0, shape[1] - 1, 0],
		&[shape[0], shape[1], shape[2]],
	)?;
	last.eval()
}

impl Session {
	pub fn load(model_dir: &Path) -> Result<Self> {
		Self::load_with_cache_config(model_dir, PromptCacheConfig::default())
	}

	/// Like [`Session::load`], but with the prompt-cache pool's sizing
	/// (max entries, idle TTL, minimum-cacheable-tokens gate) overridden
	/// instead of [`PromptCacheConfig::default`].
	pub fn load_with_cache_config(
		model_dir: &Path,
		cache_config: PromptCacheConfig,
	) -> Result<Self> {
		Self::load_with_cache_config_and_manifest(model_dir, cache_config, None)
	}

	pub(crate) fn load_with_cache_config_and_manifest(
		model_dir: &Path,
		cache_config: PromptCacheConfig,
		expected_files: Option<&[crate::model::ModelFile]>,
	) -> Result<Self> {
		let runtime = crate::runtime::initialize_default_if_needed()
			.map_err(|error| Error::Mlx(error.to_string()))?;
		// emelex patch: bound MLX's freed-buffer cache. Left at its
		// default the cache accumulates prefill transients without
		// bound (>16 GiB after one long-context prompt) - wired memory
		// that crowds a large checkpoint into the Metal wired limit and
		// kills the process.
		ops::set_cache_limit(MLX_FREED_BUFFER_CACHE_BYTES)?;
		// emelex patch: config bytes and every selected shard descriptor are
		// captured once, then shared by model loading and MTP certification.
		// No model-owned path is reopened after this point.
		let temp_dir = runtime.home().join("temp");
		let checkpoint = match expected_files {
			Some(files) => crate::model::layout::CheckpointSnapshot::open_verified_in(
				model_dir, &temp_dir, files,
			),
			None => crate::model::layout::CheckpointSnapshot::open_in(model_dir, &temp_dir),
		}
		.map_err(|error| Error::Config(error.to_string()))?;
		#[cfg(not(test))]
		let certificate_policy = MtpCertificatePolicy::Exact;
		#[cfg(test)]
		let certificate_policy = MtpCertificatePolicy::SyntheticFixture;
		Self::load_checkpoint(model_dir, cache_config, checkpoint, certificate_policy)
	}

	fn load_checkpoint(
		model_dir: &Path,
		cache_config: PromptCacheConfig,
		mut checkpoint: crate::model::layout::CheckpointSnapshot,
		certificate_policy: MtpCertificatePolicy,
	) -> Result<Self> {
		let config: Value = serde_json::from_slice(checkpoint.config_bytes())
			.map_err(|error| Error::Config(format!("bad config.json: {error}")))?;
		let allow_mtp = match certificate_policy {
			MtpCertificatePolicy::Exact => {
				crate::engine::mtp_certification::model_is_certified(&checkpoint)?
			}
			#[cfg(test)]
			MtpCertificatePolicy::SyntheticFixture => true,
		};
		let model = Model::load_snapshot(&mut checkpoint, model_dir, allow_mtp)?;
		let mtp_certified = allow_mtp && model.has_mtp();
		let mut tokenizer = Tokenizer::load_snapshot(&checkpoint)?;
		register_extra_eos_ids(
			&config,
			checkpoint.runtime_metadata("generation_config.json"),
			&mut tokenizer,
		)?;
		let (chat_template_capabilities, tool_call_format) =
			tokenizer.resolved_chat_template_capabilities()?;
		let model_context_limit = declared_context_limit(&config)?;
		Ok(Session {
			model,
			tokenizer,
			prompt_cache: Mutex::new(PromptCachePool::from_config(cache_config)),
			model_context_limit,
			chat_template_capabilities,
			tool_call_format,
			mtp_certified,
			#[cfg(test)]
			priming_fault: std::sync::atomic::AtomicBool::new(false),
			#[cfg(test)]
			mtp_prefill_materialized_chunks: std::sync::atomic::AtomicUsize::new(0),
		})
	}

	/// Load the exact descriptor-backed checkpoint verified by the external
	/// parity gate. Unlike ordinary unit fixtures, this always executes the
	/// production byte certificate.
	#[cfg(test)]
	pub(crate) fn load_certified_snapshot_for_parity(
		model_dir: &Path,
		checkpoint: crate::model::layout::CheckpointSnapshot,
	) -> Result<Self> {
		crate::runtime::initialize_default_if_needed()
			.map_err(|error| Error::Mlx(error.to_string()))?;
		ops::set_cache_limit(MLX_FREED_BUFFER_CACHE_BYTES)?;
		Self::load_checkpoint(
			model_dir,
			PromptCacheConfig::default(),
			checkpoint,
			MtpCertificatePolicy::Exact,
		)
	}

	/// emelex patch (not upstream): arm the one-shot priming fault - the
	/// next [`Session::prime_mtp`] / [`Session::prime_mtp_resume`] call
	/// fails, exercising the boundary/suffix-priming failure rows.
	#[cfg(test)]
	pub(crate) fn inject_priming_failure(&self) {
		self.priming_fault
			.store(true, std::sync::atomic::Ordering::SeqCst);
	}

	#[cfg(test)]
	fn take_priming_fault(&self) -> Result<()> {
		if self
			.priming_fault
			.swap(false, std::sync::atomic::Ordering::SeqCst)
		{
			return Err(Error::Model(String::from("injected priming fault")));
		}
		Ok(())
	}

	#[cfg(test)]
	fn mtp_prefill_materialized_chunks(&self) -> usize {
		self.mtp_prefill_materialized_chunks
			.load(std::sync::atomic::Ordering::SeqCst)
	}

	pub fn tokenizer(&self) -> &Tokenizer {
		&self.tokenizer
	}

	/// The tool-call output convention this model's chat template uses.
	pub fn tool_call_format(&self) -> crate::engine::tools::ToolCallFormat {
		self.tool_call_format
	}

	pub(crate) fn chat_template_capabilities(
		&self,
	) -> crate::engine::tokenizer::ChatTemplateCapabilities {
		self.chat_template_capabilities
	}

	/// Whether the loaded model can accept image attachments.
	pub fn supports_images(&self) -> bool {
		self.model.supports_images()
			&& self.probe_media_template_binding(ChatMessage::user_with_image(
				"emelex image capability probe",
				Vec::new(),
			))
	}

	/// emelex patch (not upstream): whether the loaded checkpoint carries
	/// an MTP module whose exact bytes passed the checked-in parity
	/// certificate (see `crate::engine::models::mtp`).
	pub fn supports_mtp(&self) -> bool {
		self.mtp_certified
	}

	/// emelex patch (not upstream): direct model access for the
	/// env-gated parity-gate test (`crate::engine::parity`).
	#[cfg(test)]
	pub fn model_for_tests(&self) -> &crate::engine::models::Model {
		&self.model
	}

	/// Whether the loaded model can accept audio attachments.
	pub fn supports_audio(&self) -> bool {
		self.model.supports_audio()
			&& self.probe_media_template_binding(ChatMessage::user_with_audio(
				"emelex audio capability probe",
				Vec::new(),
			))
	}

	fn probe_media_template_binding(&self, message: ChatMessage) -> bool {
		self.probe_media_template_binding_with_tools(&message, None)
			&& (!self.chat_template_capabilities.tools || {
				let tools = crate::engine::tokenizer::semantic_probe_tools();
				self.probe_media_template_binding_with_tools(&message, Some(&tools))
			})
	}

	fn probe_media_template_binding_with_tools(
		&self,
		message: &ChatMessage,
		tools: Option<&[crate::engine::tools::Tool]>,
	) -> bool {
		let mut messages = Vec::new();
		if tools.is_some() {
			messages.push(ChatMessage::user("emelex media capability preflight"));
			messages.extend(crate::engine::tokenizer::semantic_probe_tool_turns(
				self.tool_call_format,
			));
		}
		messages.push(message.clone());
		let Ok(prompt) = self.tokenizer.apply_chat_template_full_for_format(
			&messages,
			true,
			tools,
			Some(false),
			self.tool_call_format,
		) else {
			return false;
		};
		let Ok(prompt_ids) = self.tokenizer.encode(&prompt) else {
			return false;
		};
		let parts = messages
			.iter()
			.flat_map(|message| message.content.iter())
			.collect::<Vec<_>>();
		validate_media_bindings(
			&parts,
			&prompt_ids,
			MediaPlaceholderIds {
				image: self.model.image_token_ids().map(|(image, _, _)| image),
				audio: self.model.audio_token_ids().map(|(audio, _, _)| audio),
				video: self.model.video_token_id(),
			},
		)
		.is_ok()
	}

	/// Test/debug hook: fresh per-layer caches for this model.
	pub fn debug_new_caches(&self) -> Vec<crate::engine::models::cache::LayerCache> {
		self.model.new_caches()
	}

	/// Test/debug hook: per-layer hidden state stats (NemotronH only).
	pub fn debug_nemotron_layer_stats(&self, input_ids: &Array) -> Result<Vec<(f32, f32)>> {
		self.model.debug_nemotron_layer_stats(input_ids)
	}

	/// Test/debug hook: run one raw forward pass.
	pub fn debug_forward(
		&self,
		input_ids: &Array,
		caches: &mut [crate::engine::models::cache::LayerCache],
	) -> Result<Array> {
		self.model.forward(input_ids, caches)
	}

	/// Render `messages` through the model's chat template and tokenize.
	pub fn encode_chat(&self, messages: &[ChatMessage]) -> Result<Vec<u32>> {
		let prompt = self.tokenizer.apply_chat_template(messages, true)?;
		self.tokenizer.encode(&prompt)
	}

	/// Same as [`Session::encode_chat`], but also preprocesses any media
	/// attached to `messages` and expands each rendered placeholder into
	/// its model-specific span, matching the number of soft tokens the
	/// corresponding tower will actually produce:
	/// - `<|image|>` -> `boi + image_token × N + eoi`,
	/// - `<|audio|>` -> `boa + audio_token × N + eoa`,
	/// - `<|video|>` -> one `boi + image_token × N + eoi` span per
	///   uniformly-sampled frame (video reuses the vision tower).
	///
	/// Returns `(expanded_prompt_ids, media)`; `media` is empty (and no
	/// tower work happens) for prompts with no attachments, including on
	/// models with no multimodal support at all.
	pub fn encode_chat_with_media(
		&self,
		messages: &[ChatMessage],
	) -> Result<(Vec<u32>, MediaInputs)> {
		self.encode_chat_with_media_tools(messages, None)
	}

	/// Same as [`Session::encode_chat_with_media`], additionally threading
	/// a `tools` list into the chat template (mirroring
	/// [`crate::engine::tokenizer::Tokenizer::apply_chat_template_with_tools`]).
	pub fn encode_chat_with_media_tools(
		&self,
		messages: &[ChatMessage],
		tools: Option<&[crate::engine::tools::Tool]>,
	) -> Result<(Vec<u32>, MediaInputs)> {
		self.encode_chat_with_media_full(messages, tools, None)
	}

	/// Same as [`Session::encode_chat_with_media_tools`], additionally
	/// threading `enable_thinking` into the chat template (see
	/// [`GenerateOptions::enable_thinking`] /
	/// [`crate::engine::tokenizer::Tokenizer::apply_chat_template_full`]).
	pub fn encode_chat_with_media_full(
		&self,
		messages: &[ChatMessage],
		tools: Option<&[crate::engine::tools::Tool]>,
		enable_thinking: Option<bool>,
	) -> Result<(Vec<u32>, MediaInputs)> {
		let (ids, media, _pending_reasoning) = self.encode_chat_with_media_full_inner(
			messages,
			tools,
			enable_thinking,
			None,
			Cancellation::disabled(),
		)?;
		Ok((ids, media))
	}

	/// Same as [`Session::encode_chat_with_media_full`], but additionally
	/// returns whether the rendered prompt itself already opened an
	/// unclosed reasoning span (see [`reasoning::pending_marker`]) - used
	/// internally by [`Session::generate_cached`] to correctly classify/
	/// extract reasoning on checkpoints (Qwen3/3.5/3.6, NemotronH) whose
	/// template bakes the open marker into the generation prompt rather
	/// than letting the model generate it.
	fn encode_chat_with_media_full_inner(
		&self,
		messages: &[ChatMessage],
		tools: Option<&[crate::engine::tools::Tool]>,
		enable_thinking: Option<bool>,
		prompt_budget: Option<PromptBudget>,
		cancellation: Cancellation<'_>,
	) -> Result<(Vec<u32>, MediaInputs, Option<(&'static str, &'static str)>)> {
		cancellation.checkpoint()?;
		// Reasoning is opt-in: several hybrid-thinking checkpoints (Qwen3/
		// 3.5/3.6, MiniCPM5) only special-case `enable_thinking` in their
		// template when it's explicitly `false` and otherwise open a
		// `<think>` span unprompted, so leaving the key entirely undefined
		// (rather than explicitly forcing it off) would silently turn
		// reasoning "on by default" for those families. Default `None` to
		// `false` here so callers get a direct answer unless they opt in
		// with `Some(true)`.
		let enable_thinking = enable_thinking.or(Some(false));
		let prompt = self.tokenizer.apply_chat_template_full_for_format(
			messages,
			true,
			tools,
			enable_thinking,
			self.tool_call_format,
		)?;
		let pending_reasoning = reasoning::pending_marker(&prompt);
		let base_ids = self.tokenizer.encode(&prompt)?;
		cancellation.checkpoint()?;

		let parts: Vec<&ContentPart> = messages.iter().flat_map(|m| m.content.iter()).collect();
		let has_visual_media = parts
			.iter()
			.any(|p| matches!(p, ContentPart::Image(_) | ContentPart::Video(_)));
		let has_video = parts
			.iter()
			.any(|part| matches!(part, ContentPart::Video(_)));
		let has_audio = parts.iter().any(|p| matches!(p, ContentPart::Audio(_)));

		let model_image_ids = self.model.image_token_ids();
		let image_params = if has_visual_media {
			let params = self.model.image_processing_params().ok_or_else(|| {
				Error::Model(
					"images/videos were attached but this model has no vision support \
					 (no vision_config)"
						.into(),
				)
			})?;
			let ids = self.model.image_token_ids().ok_or_else(|| {
				Error::Model(
					"model vision configuration is incomplete: image token IDs are missing".into(),
				)
			})?;
			Some((params, ids))
		} else {
			None
		};
		let image_preprocess_params = image_params.map(|(params, _)| params);
		let model_audio_ids = self.model.audio_token_ids();
		let audio_ids = if has_audio {
			Some(model_audio_ids.ok_or_else(|| {
				Error::Model(
					"audio was attached but this model has no audio support (no \
					 audio_config)"
						.into(),
				)
			})?)
		} else {
			None
		};
		let video_token_id = self.model.video_token_id();
		if has_video && video_token_id.is_none() {
			return Err(Error::Model(
				"model vision configuration is incomplete: video token ID is missing".to_string(),
			));
		}
		validate_media_bindings(
			&parts,
			&base_ids,
			MediaPlaceholderIds {
				image: model_image_ids.map(|(image, _, _)| image),
				audio: model_audio_ids.map(|(audio, _, _)| audio),
				video: video_token_id,
			},
		)?;
		if !has_visual_media && !has_audio {
			return Ok((base_ids, MediaInputs::default(), pending_reasoning));
		}
		let mut resource_budget = ProcessedMediaBudget::new(base_ids.len(), prompt_budget)?;

		// Reject aggregate encoded amplification before starting any decoder.
		for part in &parts {
			match part {
				ContentPart::Image(image) => resource_budget.reserve_encoded(image.bytes.len())?,
				ContentPart::Audio(audio) => resource_budget.reserve_encoded(audio.bytes.len())?,
				ContentPart::Video(video) => resource_budget.reserve_encoded(video.bytes.len())?,
				ContentPart::Text(_) => {}
			}
		}

		// Preprocess in content-part order, now proven equal to placeholder
		// order in the rendered prompt: standalone images, per-video frame
		// groups, audio clips - each into its own per-type queue.
		let mut image_queue: Vec<ProcessedImage> = Vec::new();
		let mut video_queue: Vec<Vec<ProcessedImage>> = Vec::new();
		let mut audio_queue: Vec<ProcessedAudio> = Vec::new();
		for part in &parts {
			cancellation.checkpoint()?;
			match part {
				ContentPart::Image(img) => {
					let (patch, max_soft, pool) = image_preprocess_params.ok_or_else(|| {
						Error::Model(
							"cannot preprocess attached image: model has no vision support".into(),
						)
					})?;
					let processed = preprocess_image_bytes_cancellable(
						&img.bytes,
						patch,
						max_soft,
						pool,
						cancellation,
					)?;
					let soft_tokens = media_soft_tokens(processed.num_soft_tokens)?;
					resource_budget.retain(
						ProcessedMediaKind::Image,
						processed.retained_tensor_bytes()?,
						soft_tokens,
						soft_tokens.checked_add(1).ok_or_else(|| {
							Error::Model("image prompt expansion overflow".to_string())
						})?,
					)?;
					image_queue.push(processed);
				}
				ContentPart::Video(vid) => {
					let (patch, max_soft, pool) = image_preprocess_params.ok_or_else(|| {
						Error::Model(
							"cannot preprocess attached video: model has no vision support".into(),
						)
					})?;
					let frames = extract_video_frames(&vid.bytes)?;
					let mut processed = Vec::with_capacity(frames.len());
					for (index, frame) in frames.iter().enumerate() {
						let image = preprocess_image_bytes_cancellable(
							frame,
							patch,
							max_soft,
							pool,
							cancellation,
						)?;
						let soft_tokens = media_soft_tokens(image.num_soft_tokens)?;
						// One video placeholder is replaced by every frame
						// span: the first span contributes N+1 net tokens,
						// each additional span N+2.
						let boundary_tokens = if index == 0 { 1 } else { 2 };
						resource_budget.retain(
							ProcessedMediaKind::Image,
							image.retained_tensor_bytes()?,
							soft_tokens,
							soft_tokens.checked_add(boundary_tokens).ok_or_else(|| {
								Error::Model("video prompt expansion overflow".to_string())
							})?,
						)?;
						processed.push(image);
					}
					video_queue.push(processed);
				}
				ContentPart::Audio(aud) => {
					let processed = match self.model.audio_samples_per_token() {
						Some(spt) => {
							preprocess_audio_bytes_raw_cancellable(&aud.bytes, spt, cancellation)?
						}
						None => preprocess_audio_bytes_cancellable(&aud.bytes, cancellation)?,
					};
					let soft_tokens = media_soft_tokens(processed.num_soft_tokens())?;
					resource_budget.retain(
						ProcessedMediaKind::Audio,
						processed.retained_tensor_bytes()?,
						soft_tokens,
						soft_tokens.checked_add(1).ok_or_else(|| {
							Error::Model("audio prompt expansion overflow".to_string())
						})?,
					)?;
					audio_queue.push(processed);
				}
				ContentPart::Text(_) => {}
			}
		}

		// Walk the rendered token stream, expanding each placeholder and
		// recording the media in placeholder order (the fusion pass fills
		// placeholder positions sequentially with the concatenated
		// per-modality features, so ordering must match exactly).
		let mut media = MediaInputs::default();
		let mut image_iter = image_queue.into_iter();
		let mut video_iter = video_queue.into_iter();
		let mut audio_iter = audio_queue.into_iter();
		let mut expanded = Vec::with_capacity(base_ids.len() * 2);
		for &t in &base_ids {
			cancellation.checkpoint()?;
			if let Some(((_, _, _), (image_token_id, boi, eoi))) = image_params {
				if t == image_token_id {
					let img = image_iter.next().ok_or_else(|| {
						Error::Model(
							"validated image placeholder exhausted its attachment queue"
								.to_string(),
						)
					})?;
					push_image_span(&mut expanded, img.num_soft_tokens, image_token_id, boi, eoi);
					media.images.push(img);
					continue;
				} else if video_token_id == Some(t) {
					let frames = video_iter.next().ok_or_else(|| {
						Error::Model(
							"validated video placeholder exhausted its attachment queue"
								.to_string(),
						)
					})?;
					for frame in frames {
						push_image_span(
							&mut expanded,
							frame.num_soft_tokens,
							image_token_id,
							boi,
							eoi,
						);
						media.images.push(frame);
					}
					continue;
				}
			}
			if let Some((audio_token_id, boa, eoa)) = audio_ids {
				if t == audio_token_id {
					let clip = audio_iter.next().ok_or_else(|| {
						Error::Model(
							"validated audio placeholder exhausted its attachment queue"
								.to_string(),
						)
					})?;
					expanded.push(boa);
					for _ in 0..clip.num_soft_tokens() {
						expanded.push(audio_token_id);
					}
					expanded.push(eoa);
					media.audios.push(clip);
					continue;
				}
			}
			expanded.push(t);
		}
		if image_iter.next().is_some() || video_iter.next().is_some() || audio_iter.next().is_some()
		{
			return Err(Error::Model(
				"validated multimodal expansion left unbound attachments".to_string(),
			));
		}

		Ok((expanded, media, pending_reasoning))
	}

	/// Generate up to `options.max_tokens` tokens continuing `prompt_ids`,
	/// invoking `on_token` for each generated token (stop early by
	/// returning `false`). Returns the full generated id sequence.
	pub fn generate(
		&self,
		prompt_ids: &[u32],
		options: GenerateOptions,
		on_token: impl FnMut(GeneratedToken) -> bool,
	) -> Result<Vec<u32>> {
		let mut caches = self.model.new_caches();
		self.generate_with_caches(prompt_ids, &mut caches, options, on_token)
	}

	/// Same as [`Session::generate`] but reuses (and mutates) an existing
	/// set of per-layer caches, only running the forward pass over
	/// `new_prompt_ids` (the *new* suffix, not the whole conversation so
	/// far) before decoding. This is the primitive [`Session::generate_cached`]
	/// builds its prompt-cache pool on top of.
	pub fn generate_with_caches(
		&self,
		new_prompt_ids: &[u32],
		caches: &mut [crate::engine::models::cache::LayerCache],
		options: GenerateOptions,
		on_token: impl FnMut(GeneratedToken) -> bool,
	) -> Result<Vec<u32>> {
		let mut sampler = Sampler::new(options.sampling);
		// emelex patch: SpecState resolution before prefill. Non-pristine
		// caller-supplied caches disable speculation for the call (the
		// supplied prefix's hiddens are unavailable for MTP priming).
		let speculate = resolve_speculative_tokens(&options).is_some()
			&& self.mtp_certified
			&& caches.iter().all(LayerCache::is_pristine);
		let (next, mtp) = self.prefill_prompt(
			new_prompt_ids,
			caches,
			&mut sampler,
			speculate,
			Cancellation::disabled(),
		)?;
		Ok(self
			.decode_loop(next, caches, sampler, options, None, mtp, on_token)?
			.emitted)
	}

	/// Same as [`Session::generate`], but the prefill forward pass splices
	/// `media`'s image/audio features in at their placeholder positions
	/// (fresh caches, single-shot).
	pub fn generate_media(
		&self,
		prompt_ids: &[u32],
		media: &MediaInputs,
		options: GenerateOptions,
		on_token: impl FnMut(GeneratedToken) -> bool,
	) -> Result<Vec<u32>> {
		let mut caches = self.model.new_caches();
		self.generate_with_media(prompt_ids, media, &mut caches, options, on_token)
	}

	/// Same as [`Session::generate_with_caches`], but the prefill forward
	/// pass splices `media`'s image/audio features in at their placeholder
	/// positions (see [`Session::encode_chat_with_media`]). Pass an empty
	/// `media` for a text-only prompt - equivalent to
	/// [`Session::generate_with_caches`].
	pub fn generate_with_media(
		&self,
		new_prompt_ids: &[u32],
		media: &MediaInputs,
		caches: &mut [crate::engine::models::cache::LayerCache],
		options: GenerateOptions,
		on_token: impl FnMut(GeneratedToken) -> bool,
	) -> Result<Vec<u32>> {
		Ok(self
			.generate_with_media_inner(
				new_prompt_ids,
				media,
				caches,
				options,
				None,
				None,
				Cancellation::disabled(),
				on_token,
			)?
			.emitted)
	}

	/// Same as [`Session::generate_with_media`], but additionally accepts
	/// `pending_reasoning` - the `(open, close)` marker pair the caller
	/// already knows the prompt opened (unclosed) at its very end, per
	/// [`reasoning::pending_marker`]. Used internally by
	/// [`Session::generate_cached`] so the decode loop's
	/// [`StreamClassifier`]/[`ReasoningBudget`] treat generation as
	/// already inside that reasoning span from its very first token,
	/// matching checkpoints (Qwen3/3.5/3.6, NemotronH) whose chat template
	/// bakes the open marker into the generation prompt instead of
	/// leaving the model to generate it.
	fn generate_with_media_inner(
		&self,
		new_prompt_ids: &[u32],
		media: &MediaInputs,
		caches: &mut [crate::engine::models::cache::LayerCache],
		options: GenerateOptions,
		pending_reasoning: Option<(&'static str, &'static str)>,
		resume_mtp: Option<(MtpCaches, Array)>,
		cancellation: Cancellation<'_>,
		on_token: impl FnMut(GeneratedToken) -> bool,
	) -> Result<DecodeOutcome> {
		cancellation.checkpoint()?;
		let mut sampler = Sampler::new(options.sampling);
		// emelex patch: SpecState resolution before prefill. Media calls
		// are Disabled (MRoPE). `resume_mtp` is the pooled Reuse(MtpState)
		// path: `generate_cached` passes the entry's (or boundary
		// snapshot's) MTP caches + stored frontier, and the suffix prefill
		// continues priming from it - the caches are then legitimately
		// non-pristine. Without it, non-pristine caches (a prompt-cache
		// hit or a boundary prefill already fed) stay Disabled.
		let (next, mtp) = if media.is_empty() {
			if let Some(state) = resume_mtp {
				self.prefill_resume(new_prompt_ids, caches, &mut sampler, state, cancellation)?
			} else {
				let speculate = resolve_speculative_tokens(&options).is_some()
					&& self.mtp_certified
					&& caches.iter().all(LayerCache::is_pristine);
				self.prefill_prompt(
					new_prompt_ids,
					caches,
					&mut sampler,
					speculate,
					cancellation,
				)?
			}
		} else {
			cancellation.checkpoint()?;
			let prompt_arr = Array::from_slice(new_prompt_ids, &[1, new_prompt_ids.len() as i32])?;
			let logits = self.model.forward_with_media_cancellable(
				&prompt_arr,
				&media.images,
				&media.audios,
				caches,
				PREFILL_CHUNK_TOKENS,
				cancellation,
			)?;
			let next = self.sample_last(&logits, &mut sampler)?;
			cancellation.checkpoint()?;
			(next, None)
		};
		cancellation.checkpoint()?;
		self.decode_loop(
			next,
			caches,
			sampler,
			options,
			pending_reasoning,
			mtp,
			on_token,
		)
	}

	/// emelex patch (not upstream): shared prefill. With `speculate`
	/// false this is byte-identical to the historical prefill (plain
	/// `forward` + last-row sample). With `speculate` true the prefill
	/// runs through `forward_hidden` and additionally primes the MTP
	/// module (BuildFresh). A priming failure aborts an explicitly
	/// speculative request instead of silently changing its semantics.
	fn prefill_prompt(
		&self,
		prompt_ids: &[u32],
		caches: &mut [LayerCache],
		sampler: &mut Sampler,
		speculate: bool,
		cancellation: Cancellation<'_>,
	) -> Result<(u32, Option<(MtpCaches, Array)>)> {
		if !speculate {
			let logits = self.prefill_plain(prompt_ids, caches, cancellation)?;
			return Ok((self.sample_last(&logits, sampler)?, None));
		}
		let (logits, state) = self.prefill_mtp(prompt_ids, caches, None, cancellation)?;
		// The first token samples from the SAME BackboneOutput's logits -
		// the normal prefill logits, which ARE evaluated for sampling.
		// The MTP priming pass below never evaluates its own logits.
		let next = self.sample_last(&logits, sampler)?;
		cancellation.checkpoint()?;
		Ok((next, Some(state)))
	}

	fn prefill_plain(
		&self,
		prompt_ids: &[u32],
		caches: &mut [LayerCache],
		cancellation: Cancellation<'_>,
	) -> Result<Array> {
		run_prefill_chunks(prompt_ids, cancellation, |chunk, is_last| {
			let length = i32::try_from(chunk.len())
				.map_err(|_| Error::Model("prefill chunk length exceeds i32".to_string()))?;
			let prompt_arr = Array::from_slice(chunk, &[1, length])?;
			let logits = self.model.forward(&prompt_arr, caches)?;
			if !is_last {
				eval_last_logits(&logits)?;
			}
			Ok(logits)
		})
	}

	/// Cooperative target+MTP prefill. Every target chunk advances the
	/// target caches through `forward_hidden`; its detached hidden block
	/// then advances MTP through either BuildFresh (first cold chunk) or
	/// Reuse (every following/warm chunk). The bridge pair at each chunk
	/// boundary makes the chunked pair stream exactly
	/// `(prompt[1..], hidden[..L-1])`.
	fn prefill_mtp(
		&self,
		prompt_ids: &[u32],
		caches: &mut [LayerCache],
		initial_mtp: Option<(MtpCaches, Array)>,
		cancellation: Cancellation<'_>,
	) -> Result<(Array, (MtpCaches, Array))> {
		let mut mtp = initial_mtp;
		let logits = run_prefill_chunks(prompt_ids, cancellation, |chunk, is_last| {
			let length = i32::try_from(chunk.len())
				.map_err(|_| Error::Model("MTP prefill chunk length exceeds i32".to_string()))?;
			let prompt_arr = Array::from_slice(chunk, &[1, length])?;
			let out = self.model.forward_hidden(&prompt_arr, caches)?;
			if !is_last {
				eval_last_logits(&out.logits)?;
			}
			mtp = Some(match mtp.take() {
				Some(state) => {
					self.prime_mtp_resume(state, chunk, &out.hidden_pre_norm, cancellation)?
				}
				None => self.prime_mtp(chunk, &out.hidden_pre_norm, cancellation)?,
			});
			Ok(out.logits)
		})?;
		let mtp = mtp.ok_or_else(|| Error::Model("MTP prefill produced no state".to_string()))?;
		Ok((logits, mtp))
	}

	/// emelex patch (not upstream): BuildFresh MTP priming over the
	/// shifted prompt (`prompt[1..]` with `prev_hidden` = detached hidden
	/// rows `[..L-1]`) via ONE `forward_mtp` call. Its cache-producing
	/// `recycle_hidden` is evaluated before the next cancellation boundary,
	/// while its vocabulary-sized logits remain unevaluated. A 1-token
	/// prompt skips the priming call entirely
	/// (`pairs_fed = 0`, frontier = detach(h_0)); the hidden block is
	/// detached once (contiguous + eval) and sliced into views.
	fn prime_mtp(
		&self,
		prompt_ids: &[u32],
		hidden_pre_norm: &Array,
		cancellation: Cancellation<'_>,
	) -> Result<(MtpCaches, Array)> {
		#[cfg(test)]
		self.take_priming_fault()?;
		let mut mtp_caches = self.model.new_mtp_caches();
		let len = prompt_ids.len() as i32;
		let width = hidden_pre_norm.dim(2);
		let block = ops::contiguous(hidden_pre_norm)?;
		block.eval()?;
		cancellation.checkpoint()?;
		if len >= 2 {
			let prev = ops::slice(&block, &[0, 0, 0], &[1, len - 1, width])?;
			let shifted = Array::from_slice(&prompt_ids[1..], &[1, len - 1])?;
			// emelex patch: materialize the cache-producing branch before
			// cancellation can discard this chunk. Priming logits remain lazy,
			// avoiding a [1, L, V] allocation.
			let step = self.model.forward_mtp(&shifted, &prev, &mut mtp_caches)?;
			step.recycle_hidden.eval()?;
			#[cfg(test)]
			self.mtp_prefill_materialized_chunks
				.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
			cancellation.checkpoint()?;
		}
		// The frontier outlives the priming block: detach the single row
		// separately so pooled state pins [1, 1, H], not [1, L, H].
		let last = ops::slice(&block, &[0, len - 1, 0], &[1, len, width])?;
		let frontier = ops::contiguous(&last)?;
		frontier.eval()?;
		cancellation.checkpoint()?;
		Ok((mtp_caches, frontier))
	}

	/// emelex patch (not upstream): Reuse(MtpState) priming - extend an
	/// already-primed MTP cache over `suffix_ids` (the tokens a
	/// `forward_hidden` pre-feed just pushed through the target), starting
	/// from the STORED frontier. The first primed pair is the warm-hit
	/// bridge pair `(stored_frontier, suffix_ids[0])`; the remaining pairs
	/// shift along the suffix exactly like [`Session::prime_mtp`]. One
	/// `forward_mtp` call. Its cache-producing `recycle_hidden` is evaluated
	/// before the next cancellation boundary; priming logits are NEVER
	/// evaluated. Returns the advanced caches plus the new detached
	/// frontier (hidden of the last suffix token); on entry `frontier` is
	/// already detached (`MtpState` contract), so the concat below reads
	/// only detached/small arrays.
	fn prime_mtp_resume(
		&self,
		mtp: (MtpCaches, Array),
		suffix_ids: &[u32],
		hidden_pre_norm: &Array,
		cancellation: Cancellation<'_>,
	) -> Result<(MtpCaches, Array)> {
		#[cfg(test)]
		self.take_priming_fault()?;
		debug_assert!(!suffix_ids.is_empty(), "resume priming needs a suffix");
		let (mut mtp_caches, frontier) = mtp;
		let len = suffix_ids.len() as i32;
		let width = hidden_pre_norm.dim(2);
		// Batched detach (glossary rule): one contiguous block, one eval,
		// then views of the detached block.
		let block = ops::contiguous(hidden_pre_norm)?;
		block.eval()?;
		cancellation.checkpoint()?;
		let prev = if len >= 2 {
			let head = ops::slice(&block, &[0, 0, 0], &[1, len - 1, width])?;
			ops::concatenate(&[&frontier, &head], 1)?
		} else {
			// One-token suffix: the bridge pair alone.
			frontier.clone()
		};
		let ids = Array::from_slice(suffix_ids, &[1, len])?;
		// emelex patch: materialize the cache-producing branch before
		// cancellation can discard this chunk. Priming logits remain lazy,
		// avoiding a [1, L, V] allocation.
		let step = self.model.forward_mtp(&ids, &prev, &mut mtp_caches)?;
		step.recycle_hidden.eval()?;
		#[cfg(test)]
		self.mtp_prefill_materialized_chunks
			.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
		cancellation.checkpoint()?;
		// New frontier = detached hidden of the last suffix token.
		let last = ops::slice(&block, &[0, len - 1, 0], &[1, len, width])?;
		let new_frontier = ops::contiguous(&last)?;
		new_frontier.eval()?;
		cancellation.checkpoint()?;
		Ok((mtp_caches, new_frontier))
	}

	/// emelex patch (not upstream): suffix prefill for the pooled
	/// Reuse(MtpState) path. The target suffix runs through
	/// `forward_hidden` with only the sampling row's logits evaluated
	/// (mirroring [`Session::prefill_prompt`]'s speculate arm), then MTP
	/// priming continues from the stored frontier via
	/// [`Session::prime_mtp_resume`] - the warm-hit bridge pair. A priming
	/// failure aborts the explicitly speculative request.
	fn prefill_resume(
		&self,
		suffix_ids: &[u32],
		caches: &mut [LayerCache],
		sampler: &mut Sampler,
		mtp: (MtpCaches, Array),
		cancellation: Cancellation<'_>,
	) -> Result<(u32, Option<(MtpCaches, Array)>)> {
		let (logits, state) = self.prefill_mtp(suffix_ids, caches, Some(mtp), cancellation)?;
		let next = self.sample_last(&logits, sampler)?;
		cancellation.checkpoint()?;
		Ok((next, Some(state)))
	}

	/// Shared token-by-token decode loop used by both
	/// [`Session::generate_with_caches`] and [`Session::generate_with_media`]
	/// once the prefill forward pass has produced the first sampled token.
	///
	/// When `options.reasoning_budget_tokens` is set, tracks generated text
	/// against a [`ReasoningBudget`] and, the moment it's exceeded inside
	/// an open reasoning span, teacher-forces that span's close marker's
	/// tokens through the model (updating `caches` exactly as if the model
	/// had generated them) before resuming normal sampling - moving
	/// generation over to the final answer instead of letting reasoning
	/// run unbounded.
	/// emelex patch (restructured; not upstream): the loop body now lives
	/// in `crate::engine::spec::RoundDriver` behind the `RoundOps` seam -
	/// one `run_round` per iteration covers the target-only round, the
	/// speculative round, and forced close for every mode, so the shared
	/// transition is single-sourced. With `mtp` `None` (spec off, media
	/// call, non-pristine caller caches, no MTP module) the driver's
	/// target-only path reproduces the historical decode loop.
	#[allow(
		clippy::too_many_arguments,
		reason = "decode-loop state remains explicit across target and speculative modes"
	)]
	pub(crate) fn decode_loop(
		&self,
		mut next: u32,
		caches: &mut [crate::engine::models::cache::LayerCache],
		sampler: Sampler,
		options: GenerateOptions,
		pending_reasoning: Option<(&'static str, &'static str)>,
		mtp: Option<(MtpCaches, Array)>,
		on_token: impl FnMut(GeneratedToken) -> bool,
	) -> Result<DecodeOutcome> {
		let eos_ids = self.tokenizer.eos_token_ids();
		let mut budget = options.reasoning_budget_tokens.map(ReasoningBudget::new);
		let mut classifier = StreamClassifier::new(self.tool_call_format());
		if let Some(pair @ (_, close)) = pending_reasoning {
			classifier.seed_reasoning(close);
			if let Some(b) = budget.as_mut() {
				b.seed_open(pair);
			}
		}

		let emitter = TokenEmitter::new(
			&self.tokenizer,
			eos_ids,
			classifier,
			options.max_tokens,
			on_token,
		);
		let (mtp_caches, frontier) = match mtp {
			Some((caches, frontier)) => (Some(caches), Some(frontier)),
			None => (None, None),
		};
		let spec_k = resolve_speculative_tokens(&options).filter(|_| frontier.is_some());
		let ops_impl = spec::SessionOps::new(&self.model, caches, mtp_caches);
		let mut driver = spec::RoundDriver::new(
			ops_impl,
			emitter,
			budget,
			sampler,
			options.max_tokens,
			spec_k,
			frontier,
		);
		while driver.used() < options.max_tokens {
			// Loop invariant: `next` is sampled but unfed and unemitted;
			// the caches contain exactly `emitted[..committed_len]` (plus
			// the prompt).
			match driver.run_round(next)? {
				spec::RoundEnd::Continue { next: successor } => next = successor,
				spec::RoundEnd::Finished | spec::RoundEnd::Aborted | spec::RoundEnd::Budget => {
					break;
				}
			}
		}

		let reasoning_forced_closed = driver.reasoning_forced_closed();
		let (emitted, committed_len, stats, ops_impl, frontier) = driver.finish();
		debug_assert!(
			committed_len <= emitted.len(),
			"committed ledger must be a prefix of the emitted ledger"
		);
		let mtp = match (ops_impl.into_mtp(), frontier) {
			(Some(caches), Some(frontier)) => {
				// emelex patch (MLX review): the driver's frontier is a row
				// VIEW of the last feed's detached [1, r+1, H] block —
				// storing the view would pin that whole block for the pooled
				// entry's lifetime. Detach (ops::contiguous + eval) the
				// single row so the pooled MtpState pins only [1, 1, H],
				// honoring MtpState's detached-frontier contract. A detach
				// failure here never invalidates the target (detach-failure
				// rule): the completed call stands and only the MTP state is
				// dropped from the pool handoff.
				let detached = ops::contiguous(&frontier).and_then(|row| row.eval().map(|()| row));
				match detached {
					Ok(frontier) => Some(MtpState {
						pairs_fed: caches.pairs_fed(),
						caches,
						frontier,
					}),
					Err(e) => {
						tracing::warn!(
							"MTP frontier detach failed at pool handoff ({e}); dropping the \
							 call's MTP state"
						);
						None
					}
				}
			}
			_ => None,
		};
		// `drafted` is counted at draft time, so a call whose only round
		// failed before its decision still reports its proposals.
		let speculation = (stats.rounds > 0 || stats.drafted > 0).then_some(stats);
		if let Some(stats) = &speculation {
			tracing::debug!(
				drafted = stats.drafted,
				rounds = stats.rounds,
				accepted_by_depth = ?stats.accepted_by_depth,
				"mtp speculative decoding stats"
			);
		}
		Ok(DecodeOutcome {
			emitted,
			committed_len,
			speculation,
			mtp,
			reasoning_forced_closed,
		})
	}

	pub(crate) fn new_caches(&self) -> Vec<crate::engine::models::cache::LayerCache> {
		self.model.new_caches()
	}

	/// Stateless, cache-aware chat completion: render + encode the *full*
	/// `messages` transcript (mirroring how OpenAI/Anthropic's APIs take
	/// the whole conversation on every call, not a delta), look up the
	/// longest cached prefix of it in this session's [`PromptCachePool`],
	/// run only the uncached suffix (and any not-yet-fed media) through
	/// the model, then store the extended prefix back into the pool.
	///
	/// Two independent calls that happen to share a prefix (the common
	/// case: the next turn of the same conversation, but also just two
	/// unrelated calls sharing a system prompt) both benefit - there is no
	/// caller-held session handle, so nothing needs to be reset when
	/// switching to an unrelated conversation; it simply misses the pool
	/// and starts cold, exactly like a fresh prompt would.
	/// emelex patch (not upstream): length, in tokens, of the rendered
	/// transcript *without* the generation prompt, when that rendering is
	/// a strict prefix of `full_ids` (it is for chat templates that
	/// append a generation-prompt suffix). `None` when it isn't, or when
	/// rendering/encoding fails - boundary caching is then skipped.
	fn conversation_boundary_len(
		&self,
		messages: &[ChatMessage],
		tools: Option<&[Tool]>,
		enable_thinking: Option<bool>,
		full_ids: &[u32],
	) -> Option<usize> {
		let rendered = self
			.tokenizer
			.apply_chat_template_full_for_format(
				messages,
				false,
				tools,
				enable_thinking,
				self.tool_call_format,
			)
			.ok()?;
		let ids = self.tokenizer.encode(&rendered).ok()?;
		is_prefix(&ids, full_ids).then_some(ids.len())
	}

	pub fn generate_cached(
		&self,
		messages: &[ChatMessage],
		tools: Option<&[Tool]>,
		options: GenerateOptions,
		on_token: impl FnMut(GeneratedToken) -> bool,
	) -> Result<GenerateReply> {
		self.generate_cached_inner(messages, tools, options, Cancellation::disabled(), on_token)
	}

	pub(crate) fn generate_cached_cancellable(
		&self,
		messages: &[ChatMessage],
		tools: Option<&[Tool]>,
		options: GenerateOptions,
		is_cancelled: &dyn Fn() -> bool,
		on_token: impl FnMut(GeneratedToken) -> bool,
	) -> Result<GenerateReply> {
		self.generate_cached_inner(
			messages,
			tools,
			options,
			Cancellation::cooperative(is_cancelled),
			on_token,
		)
	}

	fn generate_cached_inner(
		&self,
		messages: &[ChatMessage],
		tools: Option<&[Tool]>,
		options: GenerateOptions,
		cancellation: Cancellation<'_>,
		mut on_token: impl FnMut(GeneratedToken) -> bool,
	) -> Result<GenerateReply> {
		cancellation.checkpoint()?;
		if resolve_speculative_tokens(&options).is_some() && !self.mtp_certified {
			return Err(Error::CapabilityUnavailable {
				capability: "acceleration:mtp",
				reason: format!(
					"loaded checkpoint is not covered by {}",
					crate::engine::mtp_certification::IMPLEMENTATION_ID
				),
			});
		}
		let context_limit = self
			.model_context_limit
			.map_or(options.context_tokens, |limit| {
				limit.min(options.context_tokens)
			});
		let (full_ids, media, pending_reasoning) = self.encode_chat_with_media_full_inner(
			messages,
			tools,
			options.enable_thinking,
			Some(PromptBudget {
				max_output_tokens: options.max_tokens,
				context_limit,
			}),
			cancellation,
		)?;
		let requested_context =
			full_ids
				.len()
				.checked_add(options.max_tokens)
				.ok_or_else(|| Error::ContextExceeded {
					prompt_tokens: full_ids.len(),
					max_output_tokens: options.max_tokens,
					limit: context_limit,
				})?;
		if requested_context > context_limit {
			return Err(Error::ContextExceeded {
				prompt_tokens: full_ids.len(),
				max_output_tokens: options.max_tokens,
				limit: context_limit,
			});
		}

		let cache_enabled = options.prompt_cache.unwrap_or(true);
		cancellation.checkpoint()?;
		// emelex patch: SpecState intent resolved BEFORE the pool lookup -
		// entry compatibility is scoped to calls that would actually
		// speculate. Media calls are Disabled (MRoPE), so they treat any
		// entry as compatible and keep full caching (their inserts below
		// overwrite `mtp` with `None` per the alignment rules).
		let spec_requested = resolve_speculative_tokens(&options).is_some()
			&& self.mtp_certified
			&& media.is_empty();
		// emelex patch: recover from a poisoned pool mutex instead of
		// permanently bricking the Session after one panicked generation
		// - the pool holds plain data whose invariants hold between
		// mutations.
		let (mut caches, mut fed_len, mut fed_images, mut fed_audios, mut mtp) = if cache_enabled {
			let mut pool = self
				.prompt_cache
				.lock()
				.unwrap_or_else(std::sync::PoisonError::into_inner);
			// A spec-enabled call hitting an aligned entry whose `mtp` is
			// `None` is a COLD MISS (cached usage = 0): the entry is not
			// used, not evicted, and its `last_used` refresh is skipped -
			// see `find_longest_compatible_prefix`.
			match pool.find_longest_compatible_prefix(&full_ids, spec_requested) {
				Some((entry, shared)) => (
					entry.caches,
					shared,
					entry.fed_images,
					entry.fed_audios,
					// A non-speculating call ignores a stored MtpState (it
					// would neither keep it aligned nor use it).
					entry
						.mtp
						.filter(|_| spec_requested)
						.map(|state| (state.caches, state.frontier)),
				),
				None => (self.new_caches(), 0, 0, 0, None),
			}
		} else {
			(self.new_caches(), 0, 0, 0, None)
		};
		// emelex patch: an entry covering the *entire* prompt leaves no
		// suffix to prefill - the forward pass needs at least one token
		// to produce logits, and re-feeding a token whose KV the cache
		// already contains would corrupt positions. Treat it as a miss:
		// the reset covers target caches, media counters, AND the working
		// MTP state together (alignment rule).
		if fed_len >= full_ids.len() {
			caches = self.new_caches();
			fed_len = 0;
			fed_images = 0;
			fed_audios = 0;
			mtp = None;
		}
		// How much of the prompt the *pool* actually served this call -
		// the boundary prefill below advances fed_len with freshly
		// computed tokens that must not be reported as cache hits.
		let pool_hit_tokens = fed_len;

		// emelex patch (not upstream): boundary snapshot. Full-prompt
		// entries can never serve the next turn on templates that insert
		// non-history tokens into the generation prompt (e.g. the empty
		// `<think>\n\n</think>` block Qwen3-family templates append),
		// because the next turn re-renders the assistant turn without
		// those tokens and the exact-prefix lookup then misses. So,
		// additionally snapshot the cache state at the *conversation
		// boundary* - the rendered transcript without the generation
		// prompt - which every later turn extends verbatim. Recurrent
		// (gated-delta) layer state cannot be truncated after the fact,
		// hence the snapshot is taken mid-prefill: feed up to the
		// boundary, clone (cheap - arrays are refcounted and never
		// mutated in place), then feed the rest. Text-only prompts only;
		// the entry is inserted after generation so the full-prompt
		// insert below cannot replace it (the boundary ids are a prefix
		// of the full ids).
		let mut boundary_snapshot: Option<(Vec<u32>, Vec<LayerCache>, Option<MtpState>)> = None;
		if cache_enabled && media.is_empty() {
			if let Some(boundary_len) =
				self.conversation_boundary_len(messages, tools, options.enable_thinking, &full_ids)
				&& boundary_len > fed_len
				&& boundary_len < full_ids.len()
			{
				let pre = &full_ids[fed_len..boundary_len];
				if spec_requested {
					// emelex patch: third forward site. When this call
					// speculates, the boundary pre-feed runs through
					// `forward_hidden` (its logits are dropped UNEVALUATED -
					// nothing samples at the boundary) and the boundary
					// pairs are primed BEFORE the clone below, so the
					// snapshot captures target caches plus an aligned
					// MtpState. On a warm hit the first primed pair is the
					// bridge pair `(stored_frontier, full_ids[fed_len])`; on
					// a cold start (`mtp` None implies `fed_len == 0` here -
					// the compatibility lookup never hands a speculating
					// call an mtp-less entry) this is BuildFresh over the
					// boundary prefix. A priming failure aborts the explicitly
					// speculative request rather than changing decode modes.
					debug_assert!(
						mtp.is_some() || fed_len == 0,
						"a speculating warm hit must carry an MtpState"
					);
					let (logits, state) =
						self.prefill_mtp(pre, &mut caches, mtp.take(), cancellation)?;
					// Nothing samples at the boundary. Intermediate chunks
					// were evaluated by `prefill_mtp`; the final logits stay
					// deliberately unevaluated and are dropped here.
					drop(logits);
					mtp = Some(state);
					cancellation.checkpoint()?;
				} else {
					let logits = self.prefill_plain(pre, &mut caches, cancellation)?;
					eval_last_logits(&logits)?;
					cancellation.checkpoint()?;
				}
				// The snapshot's MtpState is aligned to the boundary ids:
				// pairs_fed == boundary_len - 1 (asserted at pool insert).
				let snapshot_mtp = mtp.as_ref().map(|(mtp_caches, frontier)| MtpState {
					pairs_fed: mtp_caches.pairs_fed(),
					caches: mtp_caches.clone(),
					frontier: frontier.clone(),
				});
				boundary_snapshot = Some((
					full_ids[..boundary_len].to_vec(),
					caches.clone(),
					snapshot_mtp,
				));
				fed_len = boundary_len;
			}
		}

		let new_suffix = &full_ids[fed_len..];
		let new_media = MediaInputs {
			images: media.images[fed_images.min(media.images.len())..].to_vec(),
			audios: media.audios[fed_audios.min(media.audios.len())..].to_vec(),
		};

		let mut aborted = false;
		let outcome = self.generate_with_media_inner(
			new_suffix,
			&new_media,
			&mut caches,
			options,
			pending_reasoning,
			// emelex patch: Reuse(MtpState) - the suffix prefill continues
			// MTP priming from the stored (entry or boundary-snapshot)
			// frontier; `None` uses ordinary prompt prefill.
			mtp,
			cancellation,
			|tok| {
				let keep_going = on_token(tok);
				if !keep_going {
					aborted = true;
				}
				keep_going
			},
		)?;
		cancellation.checkpoint()?;
		// One token can produce several classified display callbacks, and
		// terminal decoder/classifier flushes reuse the last token ID. The
		// decode outcome is therefore the sole token ledger; callback count
		// cannot define generation IDs or usage.
		let DecodeOutcome {
			emitted: generated_ids,
			committed_len,
			speculation,
			mtp: outcome_mtp,
			reasoning_forced_closed,
		} = outcome;

		let usage = Usage {
			prompt_tokens: full_ids.len(),
			cached_tokens: pool_hit_tokens,
			completion_tokens: generated_ids.len(),
		};

		if cache_enabled {
			let mut pool = self
				.prompt_cache
				.lock()
				.unwrap_or_else(std::sync::PoisonError::into_inner);
			if let Some((boundary_ids, boundary_caches, boundary_mtp)) = boundary_snapshot {
				// emelex patch: when a boundary snapshot exists, store ONLY
				// the boundary lineage. The full-prompt entry (prompt +
				// reply KV) can rarely be extended by a later turn - on
				// think-family templates never - and storing both stranded
				// one dead full-KV entry in the pool per conversation turn.
				// The snapshot's aligned MtpState (or `None`) rides along;
				// `insert_or_update` overwrites the lineage's `mtp`
				// wholesale and asserts the alignment invariant.
				pool.insert_or_update(boundary_ids, boundary_caches, 0, 0, false, boundary_mtp);
			} else {
				// emelex patch: pooled ids are the prompt plus exactly the
				// committed ledger - the emitted prefix whose KV the caches
				// actually contain. Emitted-but-unfed tokens (a trailing
				// EOS, a cancellation token, a cancelled forced-close
				// suffix) never enter the pool, or a future prefix hit
				// would resume positions off. `outcome.mtp` is the decode
				// loop's surviving MtpState, aligned to exactly that
				// committed prefix (`None` whenever the call went
				// target-only - mid-call discard, spec-off, media).
				let mut cached_ids = full_ids;
				cached_ids.extend_from_slice(&generated_ids[..committed_len]);
				pool.insert_or_update(
					cached_ids,
					caches,
					media.images.len(),
					media.audios.len(),
					false,
					outcome_mtp,
				);
			}
		}

		// Decode without stripping special tokens (see
		// `Tokenizer::decode_raw`) so reasoning/tool-call markers survive
		// on checkpoints that implement them as special vocabulary
		// entries - except the eos token itself, which carries no
		// content and would otherwise leak its literal spelling (e.g.
		// `<end_of_turn>`) into the reply.
		let eos_ids = self.tokenizer.eos_token_ids();
		let content_ids: Vec<u32> = generated_ids
			.iter()
			.copied()
			.filter(|id| !eos_ids.contains(id))
			.collect();
		let raw_text = self.tokenizer.decode_raw(&content_ids)?;
		// If the prompt itself already opened a reasoning span (see
		// `pending_reasoning` above), the model's generated text never
		// contains the literal open marker - splice it back on so
		// `split_reasoning` still finds and extracts the span.
		let raw_reply = match pending_reasoning {
			Some((open, _)) => format!("{open}{raw_text}"),
			None => raw_text,
		};
		let (reasoning, text) = if reasoning_forced_closed {
			reasoning::split_reasoning_after_forced_close(&raw_reply)
		} else {
			reasoning::split_reasoning(&raw_reply)
		};
		let format = self.tool_call_format();
		let (text, calls) = if matches!(format, ToolCallFormat::None) {
			(text, Vec::new())
		} else {
			// Keep `text` and `tool_calls` separate (OpenAI/Anthropic
			// style). The parser returns untrusted proposals; only calls
			// advertised for this request with schema-valid arguments are
			// accepted and stripped from visible output.
			crate::engine::tools::parse_and_strip_tool_calls(
				&text,
				format,
				tools.unwrap_or_default(),
			)
		};
		let finish_reason = classify_finish(&generated_ids, eos_ids, !calls.is_empty(), aborted);

		Ok(GenerateReply {
			text,
			tool_calls: calls,
			usage,
			reasoning,
			finish_reason,
			// emelex patch: `Some` iff the call drafted or decided at least
			// one speculative round.
			speculation,
		})
	}

	fn sample_last(&self, logits: &Array, sampler: &mut Sampler) -> Result<u32> {
		let shape = logits.shape();
		let seq_len = shape[1];
		let last = ops::slice(logits, &[0, seq_len - 1, 0], &[shape[0], seq_len, shape[2]])?;
		let last = ops::reshape(&last, &[shape[2]])?;
		sampler.sample(&last)
	}
}

/// Preprocessed media accompanying one encoded prompt, in placeholder
/// order (video frames appear as ordinary `images` entries, one per
/// sampled frame). Produced by [`Session::encode_chat_with_media`] and
/// consumed by [`Session::generate_with_media`].
#[derive(Debug, Clone, Default)]
pub struct MediaInputs {
	pub images: Vec<ProcessedImage>,
	pub audios: Vec<ProcessedAudio>,
}

impl MediaInputs {
	pub fn is_empty(&self) -> bool {
		self.images.is_empty() && self.audios.is_empty()
	}
}

/// Raw/display text decoded from the same token IDs.
struct DecodedPiece {
	raw: String,
	display: String,
}

/// Incremental display detokenizer. Byte-level BPE tokenizers routinely
/// split one multi-byte character across token IDs; decoding IDs one at a
/// time would turn those pieces into U+FFFD. Up to four IDs are withheld
/// and decoded together. The last attempted decode is retained so a stream
/// ending mid-scalar still flushes a replacement character rather than
/// silently losing bytes.
#[derive(Default)]
struct StreamDecoder {
	pending_ids: Vec<u32>,
	pending_raw: String,
	pending_display: String,
}

impl StreamDecoder {
	fn next(&mut self, tokenizer: &Tokenizer, id: u32, raw: &str) -> Result<Option<DecodedPiece>> {
		self.pending_ids.push(id);
		self.pending_raw.push_str(raw);
		self.pending_display = tokenizer.decode(&self.pending_ids)?;
		if self.pending_display.ends_with('\u{FFFD}') && self.pending_ids.len() < 4 {
			Ok(None)
		} else {
			Ok(self.take())
		}
	}

	fn finish(&mut self) -> Option<DecodedPiece> {
		self.take()
	}

	fn take(&mut self) -> Option<DecodedPiece> {
		if self.pending_ids.is_empty() {
			return None;
		}
		self.pending_ids.clear();
		Some(DecodedPiece {
			raw: std::mem::take(&mut self.pending_raw),
			display: std::mem::take(&mut self.pending_display),
		})
	}
}

#[cfg(test)]
mod tests {
	use std::cell::Cell;

	use super::*;
	use crate::engine::tokenizer::{AudioContent, ImageContent, VideoContent};

	const EOS: &[u32] = &[2, 106];

	#[test]
	fn image_binding_requires_exact_placeholder_cardinality() {
		let image = ContentPart::Image(ImageContent { bytes: Vec::new() });
		assert!(
			validate_media_bindings(
				&[&image],
				&[1, 10, 2],
				MediaPlaceholderIds {
					image: Some(10),
					..MediaPlaceholderIds::default()
				},
			)
			.is_ok()
		);
	}

	#[test]
	fn text_only_binding_rejects_extra_image_placeholder() {
		let error = validate_media_bindings(
			&[],
			&[1, 10, 2],
			MediaPlaceholderIds {
				image: Some(10),
				..MediaPlaceholderIds::default()
			},
		)
		.unwrap_err();
		assert!(error.to_string().contains("placeholders [Image]"));
	}

	#[test]
	fn audio_binding_rejects_dropped_attachment() {
		let audio = ContentPart::Audio(AudioContent { bytes: Vec::new() });
		let error = validate_media_bindings(
			&[&audio],
			&[1, 2],
			MediaPlaceholderIds {
				audio: Some(20),
				..MediaPlaceholderIds::default()
			},
		)
		.unwrap_err();
		assert!(
			error
				.to_string()
				.contains("attachments [Audio], placeholders []")
		);
	}

	#[test]
	fn video_binding_accepts_one_distinct_placeholder() {
		let video = ContentPart::Video(VideoContent { bytes: Vec::new() });
		assert!(
			validate_media_bindings(
				&[&video],
				&[1, 30, 2],
				MediaPlaceholderIds {
					image: Some(10),
					video: Some(30),
					..MediaPlaceholderIds::default()
				},
			)
			.is_ok()
		);
	}

	#[test]
	fn video_binding_rejects_ambiguous_image_video_ids() {
		let video = ContentPart::Video(VideoContent { bytes: Vec::new() });
		let error = validate_media_bindings(
			&[&video],
			&[1, 10, 2],
			MediaPlaceholderIds {
				image: Some(10),
				video: Some(10),
				..MediaPlaceholderIds::default()
			},
		)
		.unwrap_err();
		assert!(
			error
				.to_string()
				.contains("Image and Video use the same ID")
		);
	}

	#[test]
	fn mixed_media_binding_rejects_placeholder_reordering() {
		let image = ContentPart::Image(ImageContent { bytes: Vec::new() });
		let audio = ContentPart::Audio(AudioContent { bytes: Vec::new() });
		let video = ContentPart::Video(VideoContent { bytes: Vec::new() });
		let error = validate_media_bindings(
			&[&image, &audio, &video],
			&[1, 10, 30, 20, 2],
			MediaPlaceholderIds {
				image: Some(10),
				audio: Some(20),
				video: Some(30),
			},
		)
		.unwrap_err();
		assert!(
			error
				.to_string()
				.contains("attachments [Image, Audio, Video], placeholders [Image, Video, Audio]")
		);
	}

	#[test]
	fn mixed_media_binding_accepts_attachment_order() {
		let image = ContentPart::Image(ImageContent { bytes: Vec::new() });
		let audio = ContentPart::Audio(AudioContent { bytes: Vec::new() });
		let video = ContentPart::Video(VideoContent { bytes: Vec::new() });
		assert!(
			validate_media_bindings(
				&[&image, &audio, &video],
				&[1, 10, 20, 30, 2],
				MediaPlaceholderIds {
					image: Some(10),
					audio: Some(20),
					video: Some(30),
				},
			)
			.is_ok()
		);
	}

	#[test]
	fn cooperative_prefill_stops_before_constructing_next_chunk() {
		let cancelled = Cell::new(false);
		let forwards = Cell::new(0_usize);
		let is_cancelled = || cancelled.get();
		let prompt = vec![7_u32; PREFILL_CHUNK_TOKENS + 1];

		let error = run_prefill_chunks(
			&prompt,
			Cancellation::cooperative(&is_cancelled),
			|chunk, _is_last| {
				forwards.set(forwards.get() + 1);
				cancelled.set(true);
				Ok(chunk.len())
			},
		)
		.unwrap_err();

		assert!(matches!(error, Error::Cancelled));
		assert_eq!(forwards.get(), 1);
	}

	#[test]
	fn disabled_prefill_preserves_single_forward_semantics() {
		let forwards = Cell::new(0_usize);
		let prompt = vec![7_u32; PREFILL_CHUNK_TOKENS + 1];

		let output = run_prefill_chunks(&prompt, Cancellation::disabled(), |chunk, is_last| {
			forwards.set(forwards.get() + 1);
			assert!(is_last);
			Ok(chunk.len())
		})
		.unwrap();

		assert_eq!(output, prompt.len());
		assert_eq!(forwards.get(), 1);
	}

	#[test]
	fn stream_decoder_terminal_flush_preserves_withheld_replacement() {
		let mut decoder = StreamDecoder {
			pending_ids: vec![7],
			pending_raw: "raw-byte-piece".to_string(),
			pending_display: "\u{FFFD}".to_string(),
		};
		let piece = decoder
			.finish()
			.expect("pending terminal decode must flush");
		assert_eq!(piece.display, "\u{FFFD}");
		assert!(decoder.finish().is_none());
	}

	#[test]
	fn forced_close_filter_matches_terminal_reasoning_boundary() {
		let dir =
			std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-model");
		let tokenizer = Tokenizer::load(&dir).expect("fixture tokenizer");
		let mut emitter = TokenEmitter::new(
			&tokenizer,
			tokenizer.eos_token_ids(),
			StreamClassifier::new(ToolCallFormat::Hermes),
			8,
			|_| true,
		);

		emitter.arm_close_filters("</think>");
		let immediate = emitter.filter_post_budget_closes("</think>", "</think>".to_string());
		let quoted = emitter.filter_post_budget_closes(
			"To write it, use </think>.",
			"To write it, use </think>.".to_string(),
		);
		let streamed = format!("{immediate}{quoted}");
		let (_, terminal) = reasoning::split_reasoning_after_forced_close(
			"<think>x</think></think>To write it, use </think>.",
		);
		assert_eq!(streamed, terminal);
		assert_eq!(terminal, "To write it, use </think>.");

		emitter.arm_close_filters("</think>");
		let prefix =
			emitter.filter_post_budget_closes("nearly answer", "nearly answer".to_string());
		let suffix = emitter
			.filter_post_budget_closes("</think>the answer", "</think>the answer".to_string());
		let streamed = format!("{prefix}{suffix}");
		let (_, terminal) = reasoning::split_reasoning_after_forced_close(
			"<think>x</think>nearly answer</think>the answer",
		);
		assert_eq!(streamed, terminal);
		assert_eq!(terminal, "nearly answer</think>the answer");

		emitter.arm_close_filters("</think>");
		let padding = " ".repeat(reasoning::MAX_FORCED_CLOSE_WHITESPACE_BYTES + 1);
		let prefix = emitter.filter_post_budget_closes(&padding, padding.clone());
		let suffix =
			emitter.filter_post_budget_closes("</think>answer", "</think>answer".to_string());
		let streamed = format!("{prefix}{suffix}");
		let raw = format!("<think>x</think>{padding}</think>answer");
		let (_, terminal) = reasoning::split_reasoning_after_forced_close(&raw);
		assert_eq!(streamed, terminal);
		assert_eq!(terminal, format!("{padding}</think>answer"));

		emitter.arm_close_filters("</think>");
		let held = emitter.filter_post_budget_closes("</thi", "</thi".to_string());
		let diverged = emitter.filter_post_budget_closes("\u{fffd}", "\u{1f4a1}".to_string());
		assert!(held.is_empty());
		assert_eq!(diverged, "</thi\u{1f4a1}");

		emitter.arm_close_filters("</think>");
		let duplicate = emitter.filter_post_budget_closes("</think>", "</think>".to_string());
		let boundary = emitter.filter_post_budget_closes("\n\n", "\n\n".to_string());
		let answer = emitter.filter_post_budget_closes("answer", "answer".to_string());
		let streamed = format!("{duplicate}{boundary}{answer}");
		let (_, terminal) =
			reasoning::split_reasoning_after_forced_close("<think>x</think></think>\n\nanswer");
		assert_eq!(streamed, terminal);
		assert_eq!(terminal, "\n\nanswer");

		emitter.arm_close_filters("</think>");
		let coalesced =
			emitter.filter_post_budget_closes("</think>\n\n", "</think>\n\n".to_string());
		assert_eq!(coalesced, "\n\n");
		assert_eq!(
			display_after_forced_close("</think>think>answer", "think>answer", 0, "</think>",),
			"think>answer",
			"special-marker omission must not consume answer prefix collision"
		);
	}

	#[test]
	fn forced_close_terminal_flush_preserves_whitespace_and_partial_literal() {
		for pending in [" \n", "</thi"] {
			let dir =
				std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-model");
			let tokenizer = Tokenizer::load(&dir).expect("fixture tokenizer");
			let captured = std::rc::Rc::new(std::cell::RefCell::new(String::new()));
			let callback_capture = std::rc::Rc::clone(&captured);
			let mut emitter = TokenEmitter::new(
				&tokenizer,
				tokenizer.eos_token_ids(),
				StreamClassifier::new(ToolCallFormat::Hermes),
				8,
				move |token| {
					callback_capture.borrow_mut().push_str(&token.text);
					true
				},
			);
			emitter.arm_close_filters("</think>");
			assert!(
				emitter
					.filter_post_budget_closes(pending, pending.to_string())
					.is_empty()
			);
			let _ = emitter.into_emitted();
			assert_eq!(&*captured.borrow(), pending);
		}
	}

	// emelex patch (not upstream): SpeculationStats counter-arithmetic
	// contract.

	/// One-based depth indexing: a round accepting exactly one draft
	/// lands at index 0, and a full rejection increments no bucket.
	#[test]
	fn speculation_stats_depth_indexing_is_one_based() {
		let mut stats = SpeculationStats::default();
		stats.record_drafted(3);
		stats.record_round(1);
		assert_eq!(stats.accepted_by_depth, vec![1]);
		stats.record_drafted(3);
		stats.record_round(3);
		assert_eq!(stats.accepted_by_depth, vec![1, 0, 1]);
		stats.record_drafted(3);
		stats.record_round(0); // full rejection: no bucket moves
		assert_eq!(stats.accepted_by_depth, vec![1, 0, 1]);
		assert_eq!(stats.rounds, 3);
		assert_eq!(stats.drafted, 9);
		// rounds - sum(accepted_by_depth) = full rejections.
		assert_eq!(
			stats.rounds - stats.accepted_by_depth.iter().sum::<u64>(),
			1
		);
	}

	/// Drafted counts at draft time: proposals from rounds that never
	/// reach a decision (no `record_round`) still land in `drafted`, so
	/// `drafted > 0` with `rounds == 0` is a representable, truthful
	/// state (a call whose only round failed mid-verify).
	#[test]
	fn speculation_stats_drafted_counts_undecided_rounds() {
		let mut stats = SpeculationStats::default();
		stats.record_drafted(2); // round drafted 2, then failed pre-decision
		assert_eq!(stats.drafted, 2);
		assert_eq!(stats.rounds, 0);
		assert!(stats.accepted_by_depth.is_empty());
	}

	/// All counters saturate instead of overflowing.
	#[test]
	fn speculation_stats_counters_saturate() {
		let mut stats = SpeculationStats {
			drafted: u64::MAX,
			accepted_by_depth: vec![u64::MAX, u64::MAX],
			rounds: u64::MAX,
		};
		stats.record_drafted(usize::MAX);
		stats.record_round(2);
		assert_eq!(stats.drafted, u64::MAX);
		assert_eq!(stats.rounds, u64::MAX);
		assert_eq!(stats.accepted_by_depth, vec![u64::MAX, u64::MAX]);
	}

	#[test]
	fn finish_stop_on_trailing_eos() {
		assert_eq!(
			classify_finish(&[5, 9, 2], EOS, false, false),
			FinishReason::Stop
		);
	}

	#[test]
	fn finish_tool_calls_takes_precedence_over_eos() {
		assert_eq!(
			classify_finish(&[5, 9, 2], EOS, true, false),
			FinishReason::ToolCalls
		);
	}

	#[test]
	fn finish_length_when_no_eos_and_not_aborted() {
		assert_eq!(
			classify_finish(&[5, 9, 7], EOS, false, false),
			FinishReason::Length
		);
	}

	#[test]
	fn finish_aborted_when_callback_stopped_early() {
		assert_eq!(
			classify_finish(&[5, 9, 7], EOS, false, true),
			FinishReason::Aborted
		);
	}

	#[test]
	fn finish_empty_generation_without_abort_is_length() {
		assert_eq!(
			classify_finish(&[], EOS, false, false),
			FinishReason::Length
		);
	}

	// -----------------------------------------------------------------
	// emelex patch (not upstream): engine-level tests over the written
	// tiny-model checkpoint (see `crate::engine::test_support`).
	// -----------------------------------------------------------------

	use crate::engine::test_support::write_tiny_model;

	/// Gate test for the test-support safetensors writer: `Session::load`
	/// accepts the written checkpoint (post-sanitize key names match what
	/// the qwen3_5 loader expects for a text-only checkpoint) and a short
	/// greedy `generate()` runs.
	#[test]
	fn tiny_written_model_loads_and_generates() {
		let dir = write_tiny_model(false).unwrap();
		let session = Session::load(dir.path()).unwrap();
		assert!(!session.supports_mtp());
		let prompt = session.tokenizer().encode("hello world").unwrap();
		assert_eq!(prompt.len(), 2);
		let out = session
			.generate(
				&prompt,
				GenerateOptions {
					max_tokens: 4,
					..GenerateOptions::default()
				},
				|_| true,
			)
			.unwrap();
		assert!(!out.is_empty() && out.len() <= 4);
	}

	#[test]
	fn exact_certificate_rejection_never_loads_synthetic_mtp_weights() {
		let dir = write_tiny_model(true).unwrap();
		let runtime = crate::runtime::initialize_default_if_needed().unwrap();
		let checkpoint = crate::model::layout::CheckpointSnapshot::open_in(
			dir.path(),
			&runtime.home().join("temp"),
		)
		.unwrap();
		assert!(
			!crate::engine::mtp_certification::model_is_certified(&checkpoint).unwrap(),
			"synthetic fixture must not match production certificate"
		);

		let session = Session::load_checkpoint(
			dir.path(),
			PromptCacheConfig::default(),
			checkpoint,
			MtpCertificatePolicy::Exact,
		)
		.unwrap();

		assert!(!session.supports_mtp());
		assert!(
			!session.model_for_tests().has_mtp(),
			"uncertified MTP tensors must be discarded before model construction"
		);
	}

	/// Desynchronization regression: the
	/// pre-refactor forced-close path fed the WHOLE close marker through
	/// the model before running per-token callbacks, so a cancellation
	/// mid-marker left the KV caches ahead of every ledger — pooled
	/// entries then claimed fewer tokens than their KV contained and a
	/// later prefix hit resumed positions off. The pool-relevant
	/// invariant is: caches contain exactly `prompt + committed prefix`,
	/// whatever the callback does — asserted here via attention offsets
	/// with a callback cancelling at each close-token position.
	#[test]
	fn forced_close_cancellation_keeps_caches_at_committed_prefix() {
		let dir = write_tiny_model(false).unwrap();
		let session = Session::load(dir.path()).unwrap();
		let prompt = session.tokenizer().encode("hello world").unwrap();
		// cancel_at 0 = the trigger token x itself; 1 = the close-marker
		// token (the historical desync position); 2.. = post-close.
		for cancel_at in [0usize, 1, 2, 3] {
			let mut caches = session.new_caches();
			let arr = Array::from_slice(&prompt, &[1, prompt.len() as i32]).unwrap();
			let _ = session.debug_forward(&arr, &mut caches).unwrap();
			let mut count = 0usize;
			let outcome = session
				.decode_loop(
					12,
					&mut caches,
					Sampler::new(SamplingConfig::default()),
					GenerateOptions {
						max_tokens: 8,
						reasoning_budget_tokens: Some(0),
						..GenerateOptions::default()
					},
					Some(("<think>", "</think>")),
					None,
					move |_| {
						let i = count;
						count += 1;
						i != cancel_at
					},
				)
				.unwrap();
			assert!(outcome.committed_len <= outcome.emitted.len());
			for cache in &caches {
				if let LayerCache::Attention(kv) = cache {
					assert_eq!(
						kv.offset() as usize,
						prompt.len() + outcome.committed_len,
						"cache/ledger desync at cancel_at {cancel_at}"
					);
				}
			}
		}
	}

	/// Speculative decode end-to-end on the real (tiny) model: priming,
	/// the decode_loop-level MTP entry, stats wiring, and the exact
	/// offset invariants required by prompt-cache MTP-state reuse.
	#[test]
	fn tiny_model_with_mtp_speculates_at_decode_loop_level() {
		let dir = write_tiny_model(true).unwrap();
		let session = Session::load(dir.path()).unwrap();
		assert!(session.supports_mtp());
		let prompt = session.tokenizer().encode("hello world").unwrap();
		let mut caches = session.new_caches();
		let mut sampler = Sampler::new(SamplingConfig::default());
		let (next, mtp) = session
			.prefill_prompt(
				&prompt,
				&mut caches,
				&mut sampler,
				true,
				Cancellation::disabled(),
			)
			.unwrap();
		let (mtp_caches, frontier) = mtp.expect("MTP primed at prefill");
		assert_eq!(mtp_caches.pairs_fed(), prompt.len() - 1);
		assert_eq!(frontier.shape(), vec![1, 1, 32]);
		let outcome = session
			.decode_loop(
				next,
				&mut caches,
				sampler,
				GenerateOptions {
					max_tokens: 6,
					speculative_tokens: Some(2),
					..GenerateOptions::default()
				},
				None,
				Some((mtp_caches, frontier)),
				|_| true,
			)
			.unwrap();
		assert!(outcome.committed_len <= outcome.emitted.len());
		// The pool-relevant invariant: committed prefix == cache offsets.
		for cache in &caches {
			if let LayerCache::Attention(kv) = cache {
				assert_eq!(kv.offset() as usize, prompt.len() + outcome.committed_len);
			}
		}
		// A non-trivial greedy run drafts at least once and the surviving
		// MTP state stays pair-aligned with the committed prefix.
		if outcome.emitted.len() > 1 {
			let stats = outcome.speculation.as_ref().expect("speculation ran");
			assert!(stats.rounds >= 1);
			assert!(stats.drafted >= 1);
			// One-based depth buckets: full rejections increment none, so
			// the bucket sum never exceeds the round count.
			assert!(stats.accepted_by_depth.iter().sum::<u64>() <= stats.rounds);
		}
		if let Some(state) = &outcome.mtp {
			assert_eq!(
				state.pairs_fed,
				prompt.len() + outcome.committed_len - 1,
				"pooled MTP state must be pair-aligned"
			);
			assert_eq!(state.caches.pairs_fed(), state.pairs_fed);
		}
	}

	#[test]
	fn cooperative_mtp_prefill_materializes_one_chunk_before_cancellation() {
		let dir = write_tiny_model(true).unwrap();
		let session = match Session::load(dir.path()) {
			Ok(session) => session,
			Err(Error::Mlx(message))
				if message.contains("No Metal device") || message.contains("no Metal device") =>
			{
				return;
			}
			Err(error) => panic!("unexpected tiny MTP load failure: {error}"),
		};
		let prompt = vec![8_u32; PREFILL_CHUNK_TOKENS * 2 + 1];
		let mut caches = session.new_caches();
		let mut sampler = Sampler::new(SamplingConfig::default());
		let cancelled = || session.mtp_prefill_materialized_chunks() >= 1;

		let error = session
			.prefill_prompt(
				&prompt,
				&mut caches,
				&mut sampler,
				true,
				Cancellation::cooperative(&cancelled),
			)
			.err()
			.expect("cancellation must stop before a second chunk graph");

		assert!(matches!(error, Error::Cancelled));
		assert_eq!(session.mtp_prefill_materialized_chunks(), 1);
		for cache in &caches {
			if let LayerCache::Attention(cache) = cache {
				assert_eq!(
					cache.offset() as usize,
					PREFILL_CHUNK_TOKENS,
					"target cache must stop at the same bounded chunk"
				);
			}
		}
	}

	#[test]
	fn cooperative_mtp_prefill_matches_single_pass_across_chunk_boundary() {
		fn assert_close(left: &[f32], right: &[f32]) {
			assert_eq!(left.len(), right.len());
			for (index, (&left, &right)) in left.iter().zip(right).enumerate() {
				let tolerance = 1e-3 * (1.0 + left.abs().max(right.abs()));
				assert!(
					(left - right).abs() <= tolerance,
					"value {index} differs: {left} versus {right}"
				);
			}
		}

		let dir = write_tiny_model(true).unwrap();
		let session = Session::load(dir.path()).unwrap();
		let prompt: Vec<u32> = (0..=PREFILL_CHUNK_TOKENS)
			.map(|index| if index % 2 == 0 { 8 } else { 9 })
			.collect();

		let mut whole_caches = session.new_caches();
		let mut whole_sampler = Sampler::new(SamplingConfig::default());
		let (whole_next, whole_mtp) = session
			.prefill_prompt(
				&prompt,
				&mut whole_caches,
				&mut whole_sampler,
				true,
				Cancellation::disabled(),
			)
			.unwrap();

		let never_cancel = || false;
		let mut chunked_caches = session.new_caches();
		let mut chunked_sampler = Sampler::new(SamplingConfig::default());
		let (chunked_next, chunked_mtp) = session
			.prefill_prompt(
				&prompt,
				&mut chunked_caches,
				&mut chunked_sampler,
				true,
				Cancellation::cooperative(&never_cancel),
			)
			.unwrap();

		assert_eq!(chunked_next, whole_next);
		let (mut whole_mtp_caches, whole_frontier) = whole_mtp.unwrap();
		let (mut chunked_mtp_caches, chunked_frontier) = chunked_mtp.unwrap();
		assert_eq!(whole_mtp_caches.pairs_fed(), prompt.len() - 1);
		assert_eq!(chunked_mtp_caches.pairs_fed(), prompt.len() - 1);
		assert_close(
			&whole_frontier.to_vec_f32().unwrap(),
			&chunked_frontier.to_vec_f32().unwrap(),
		);

		let next = Array::from_slice(&[whole_next], &[1, 1]).unwrap();
		let whole_target = session
			.model
			.forward(&next, &mut whole_caches)
			.unwrap()
			.to_vec_f32()
			.unwrap();
		let chunked_target = session
			.model
			.forward(&next, &mut chunked_caches)
			.unwrap()
			.to_vec_f32()
			.unwrap();
		assert_close(&whole_target, &chunked_target);

		let whole_draft = session
			.model
			.forward_mtp(&next, &whole_frontier, &mut whole_mtp_caches)
			.unwrap()
			.logits
			.to_vec_f32()
			.unwrap();
		let chunked_draft = session
			.model
			.forward_mtp(&next, &chunked_frontier, &mut chunked_mtp_caches)
			.unwrap()
			.logits
			.to_vec_f32()
			.unwrap();
		assert_close(&whole_draft, &chunked_draft);
	}

	// -----------------------------------------------------------------
	// emelex patch (not upstream): prompt-cache MtpState integration
	// over the tiny written MTP model.
	// -----------------------------------------------------------------

	/// A pool that accepts the tiny fixture's short boundary prefixes
	/// (the default 8-token minimum would silently drop them).
	fn load_mtp_session() -> (crate::engine::test_support::TempModelDir, Session) {
		let dir = write_tiny_model(true).unwrap();
		let session = Session::load_with_cache_config(dir.path(), {
			PromptCacheConfig {
				min_cacheable_tokens: 0,
				..PromptCacheConfig::default()
			}
		})
		.unwrap();
		(dir, session)
	}

	fn spec_on(max_tokens: usize) -> GenerateOptions {
		GenerateOptions {
			max_tokens,
			speculative_tokens: Some(2),
			..GenerateOptions::default()
		}
	}

	fn spec_off(max_tokens: usize) -> GenerateOptions {
		GenerateOptions {
			max_tokens,
			..GenerateOptions::default()
		}
	}

	/// Clone the pooled entry serving `query` (`find_longest_prefix` also
	/// refreshes it - fine for these tests) plus its `pairs_fed` if any.
	fn pool_entry(
		session: &Session,
		query: &[u32],
	) -> Option<crate::engine::prompt_cache::CacheEntry> {
		let mut pool = session.prompt_cache.lock().unwrap();
		pool.find_longest_prefix(query).map(|(entry, _)| entry)
	}

	fn boundary_len_of(session: &Session, messages: &[ChatMessage]) -> usize {
		let full_ids = session.encode_chat(messages).unwrap();
		session
			.conversation_boundary_len(messages, None, None, &full_ids)
			.expect("tiny template appends a generation prompt")
	}

	/// Boundary snapshot with spec on: the stored boundary entry carries
	/// an MtpState aligned to the boundary ids (pairs_fed == len - 1).
	#[test]
	fn boundary_entry_with_spec_on_carries_aligned_mtp_state() {
		let (_dir, session) = load_mtp_session();
		let turn1 = vec![ChatMessage::user("hello world")];
		let reply = session
			.generate_cached(&turn1, None, spec_on(6), |_| true)
			.unwrap();
		assert_eq!(reply.usage.cached_tokens, 0, "turn 1 is cold");

		let boundary_len = boundary_len_of(&session, &turn1);
		let full_ids = session.encode_chat(&turn1).unwrap();
		let entry = pool_entry(&session, &full_ids[..boundary_len]).expect("boundary entry stored");
		assert_eq!(entry.ids.len(), boundary_len, "boundary lineage only");
		let state = entry.mtp.expect("spec-on boundary entry carries MtpState");
		assert_eq!(
			state.pairs_fed,
			boundary_len - 1,
			"boundary MtpState must be pair-aligned"
		);
		assert_eq!(state.caches.pairs_fed(), state.pairs_fed);
	}

	/// An explicitly speculative request fails if MTP priming fails. It
	/// neither silently changes modes nor publishes partially advanced
	/// target state to the shared prompt-cache pool.
	#[test]
	fn boundary_priming_failure_aborts_without_publishing_cache() {
		let (_dir, session) = load_mtp_session();
		let turn1 = vec![ChatMessage::user("hello world")];
		session.inject_priming_failure();
		let error = match session.generate_cached(&turn1, None, spec_on(6), |_| true) {
			Err(error) => error,
			Ok(_) => panic!("explicit speculation must fail"),
		};
		assert!(
			matches!(&error, Error::Model(message) if message == "injected priming fault"),
			"unexpected failure: {error}"
		);
		let boundary_len = boundary_len_of(&session, &turn1);
		let full_ids = session.encode_chat(&turn1).unwrap();
		assert!(
			pool_entry(&session, &full_ids[..boundary_len]).is_none(),
			"failed request must not publish a boundary cache entry"
		);
	}

	/// Next-turn warm hit: the pooled MtpState is reused (cached_tokens >
	/// 0) and speculation still drafts; the extended boundary entry stays
	/// aligned.
	#[test]
	fn next_turn_hit_reuses_mtp_state_and_still_drafts() {
		let (_dir, session) = load_mtp_session();
		let turn1 = vec![ChatMessage::user("hello world")];
		session
			.generate_cached(&turn1, None, spec_on(6), |_| true)
			.unwrap();
		let boundary1 = boundary_len_of(&session, &turn1);

		let turn2 = vec![
			ChatMessage::user("hello world"),
			ChatMessage::assistant("ok"),
			ChatMessage::user("hi"),
		];
		let reply2 = session
			.generate_cached(&turn2, None, spec_on(6), |_| true)
			.unwrap();
		assert_eq!(
			reply2.usage.cached_tokens, boundary1,
			"turn 2 must hit the boundary entry"
		);
		let stats = reply2
			.speculation
			.expect("a warm spec-on turn must run speculative rounds");
		assert!(stats.drafted > 0, "speculation must still draft on a hit");

		let boundary2 = boundary_len_of(&session, &turn2);
		let full2 = session.encode_chat(&turn2).unwrap();
		let entry =
			pool_entry(&session, &full2[..boundary2]).expect("extended boundary entry stored");
		assert_eq!(entry.ids.len(), boundary2);
		let state = entry.mtp.expect("turn-2 boundary entry carries MtpState");
		assert_eq!(state.pairs_fed, boundary2 - 1);
	}

	/// Warm-hit bridge pair: suffix priming continues from the STORED
	/// frontier, so after the suffix prefill (before any decode) the MTP
	/// cache holds exactly `full_prompt_len - 1` pairs - the stored
	/// pairs, the bridge pair `(stored_frontier, id_fed_len)`, and the
	/// shifted suffix pairs.
	#[test]
	fn warm_hit_suffix_priming_bridges_from_stored_frontier() {
		let (_dir, session) = load_mtp_session();
		let turn1 = vec![ChatMessage::user("hello world")];
		session
			.generate_cached(&turn1, None, spec_on(6), |_| true)
			.unwrap();

		let turn2 = vec![
			ChatMessage::user("hello world"),
			ChatMessage::assistant("ok"),
			ChatMessage::user("hi"),
		];
		let full2 = session.encode_chat(&turn2).unwrap();
		let (entry, shared) = {
			let mut pool = session.prompt_cache.lock().unwrap();
			pool.find_longest_compatible_prefix(&full2, true)
				.expect("spec-compatible warm hit")
		};
		assert_eq!(shared, boundary_len_of(&session, &turn1));
		let state = entry.mtp.expect("stored MtpState");
		assert_eq!(state.pairs_fed, shared - 1);

		let mut caches = entry.caches;
		let mut sampler = Sampler::new(SamplingConfig::default());
		let suffix = &full2[shared..];
		let (_next, mtp) = session
			.prefill_resume(
				suffix,
				&mut caches,
				&mut sampler,
				(state.caches, state.frontier),
				Cancellation::disabled(),
			)
			.unwrap();
		let (mtp_caches, _frontier) = mtp.expect("suffix priming succeeded");
		assert_eq!(
			mtp_caches.pairs_fed(),
			full2.len() - 1,
			"bridge pair + shifted suffix pairs must land exactly at \
			 full_prompt_len - 1 before decode"
		);
		for cache in &caches {
			if let LayerCache::Attention(kv) = cache {
				assert_eq!(kv.offset() as usize, full2.len());
			}
		}
	}

	/// Compatibility: a spec-off call hits an mtp-less lineage normally; a
	/// spec-on call over the same lineage is a cold miss that neither
	/// evicts the entry nor (see the prompt_cache unit tests) refreshes
	/// it, and its rebuild stores an MtpState-bearing entry (mode switch
	/// off -> on).
	#[test]
	fn spec_on_over_mtp_less_lineage_cold_misses_without_eviction() {
		let (_dir, session) = load_mtp_session();
		let turn1 = vec![ChatMessage::user("hello world")];
		session
			.generate_cached(&turn1, None, spec_off(6), |_| true)
			.unwrap();
		let boundary1 = boundary_len_of(&session, &turn1);
		let full1 = session.encode_chat(&turn1).unwrap();
		let entry = pool_entry(&session, &full1[..boundary1]).unwrap();
		assert!(entry.mtp.is_none(), "spec-off turn stores mtp = None");

		// Same-lineage spec-off extension keeps hitting...
		let turn2 = vec![
			ChatMessage::user("hello world"),
			ChatMessage::assistant("ok"),
			ChatMessage::user("hi"),
		];
		let reply_off = session
			.generate_cached(&turn2, None, spec_off(6), |_| true)
			.unwrap();
		assert_eq!(reply_off.usage.cached_tokens, boundary1);

		// ...while the spec-on call over the same lineage is a COLD MISS.
		let turn3 = vec![
			ChatMessage::user("hello world"),
			ChatMessage::assistant("ok"),
			ChatMessage::user("hi"),
			ChatMessage::assistant("ok"),
			ChatMessage::user("a"),
		];
		let reply_on = session
			.generate_cached(&turn3, None, spec_on(6), |_| true)
			.unwrap();
		assert_eq!(
			reply_on.usage.cached_tokens, 0,
			"spec-on over an mtp-less lineage is a cold miss"
		);
		// The cold rebuild replaced the lineage in place with an
		// MtpState-bearing entry (off -> on switch), still one entry.
		{
			let pool = session.prompt_cache.lock().unwrap();
			assert_eq!(pool.len(), 1, "no eviction, no stray entries");
		}
		let boundary3 = boundary_len_of(&session, &turn3);
		let full3 = session.encode_chat(&turn3).unwrap();
		let entry = pool_entry(&session, &full3[..boundary3]).unwrap();
		let state = entry.mtp.expect("cold rebuild stores MtpState");
		assert_eq!(state.pairs_fed, boundary3 - 1);

		// A subsequent spec-off call still hits the (rebuilt) lineage.
		let turn4 = vec![
			ChatMessage::user("hello world"),
			ChatMessage::assistant("ok"),
			ChatMessage::user("hi"),
			ChatMessage::assistant("ok"),
			ChatMessage::user("a"),
			ChatMessage::assistant("ok"),
			ChatMessage::user("b"),
		];
		let reply4 = session
			.generate_cached(&turn4, None, spec_off(6), |_| true)
			.unwrap();
		assert_eq!(reply4.usage.cached_tokens, boundary3);
	}

	/// Mode switch on -> off -> on: the spec-off extension overwrites the
	/// lineage's mtp with None (wholesale), so the next spec-on turn cold
	/// rebuilds - the documented mixed-traffic ping-pong.
	#[test]
	fn mode_switch_on_off_on_ping_pongs_mtp_state() {
		let (_dir, session) = load_mtp_session();
		let turn1 = vec![ChatMessage::user("hello world")];
		session
			.generate_cached(&turn1, None, spec_on(6), |_| true)
			.unwrap();
		let boundary1 = boundary_len_of(&session, &turn1);
		let full1 = session.encode_chat(&turn1).unwrap();
		assert!(
			pool_entry(&session, &full1[..boundary1])
				.unwrap()
				.mtp
				.is_some()
		);

		// Spec-off turn: compatible with the mtp-bearing entry (hits), but
		// its insert overwrites mtp to None.
		let turn2 = vec![
			ChatMessage::user("hello world"),
			ChatMessage::assistant("ok"),
			ChatMessage::user("hi"),
		];
		let reply2 = session
			.generate_cached(&turn2, None, spec_off(6), |_| true)
			.unwrap();
		assert_eq!(
			reply2.usage.cached_tokens, boundary1,
			"spec-off treats any entry as compatible"
		);
		let boundary2 = boundary_len_of(&session, &turn2);
		let full2 = session.encode_chat(&turn2).unwrap();
		let entry = pool_entry(&session, &full2[..boundary2]).unwrap();
		assert_eq!(entry.ids.len(), boundary2);
		assert!(
			entry.mtp.is_none(),
			"a spec-off extension writes mtp = None wholesale"
		);

		// Next spec-on turn: cold rebuild.
		let turn3 = vec![
			ChatMessage::user("hello world"),
			ChatMessage::assistant("ok"),
			ChatMessage::user("hi"),
			ChatMessage::assistant("ok"),
			ChatMessage::user("a"),
		];
		let reply3 = session
			.generate_cached(&turn3, None, spec_on(6), |_| true)
			.unwrap();
		assert_eq!(reply3.usage.cached_tokens, 0, "on after off cold rebuilds");
		let boundary3 = boundary_len_of(&session, &turn3);
		let full3 = session.encode_chat(&turn3).unwrap();
		let state = pool_entry(&session, &full3[..boundary3])
			.unwrap()
			.mtp
			.expect("rebuilt lineage carries MtpState again");
		assert_eq!(state.pairs_fed, boundary3 - 1);
	}

	/// The exact-full-prompt reset resets target caches, media counters,
	/// and the working MTP state together: a full-prompt entry (even one
	/// carrying an MtpState) is treated as a miss and the call rebuilds
	/// fresh - a stale resume state here would desync the bridge pair and
	/// trip the pool's alignment assert.
	#[test]
	fn exact_full_prompt_reset_clears_mtp_with_target_state() {
		let (_dir, session) = load_mtp_session();
		let turn1 = vec![ChatMessage::user("hello world")];
		let full1 = session.encode_chat(&turn1).unwrap();
		{
			let mut pool = session.prompt_cache.lock().unwrap();
			pool.insert_or_update(
				full1.clone(),
				session.new_caches(),
				0,
				0,
				false,
				Some(MtpState {
					caches: MtpCaches(Vec::new()),
					pairs_fed: full1.len() - 1,
					frontier: Array::from_slice(&vec![0.0f32; 32], &[1, 1, 32]).unwrap(),
				}),
			);
		}
		let reply = session
			.generate_cached(&turn1, None, spec_on(6), |_| true)
			.unwrap();
		assert_eq!(
			reply.usage.cached_tokens, 0,
			"a full-prompt entry is a miss - and its MtpState must be dropped with \
			 it"
		);
		let boundary1 = boundary_len_of(&session, &turn1);
		let state = pool_entry(&session, &full1[..boundary1])
			.expect("fresh boundary entry stored")
			.mtp
			.expect("BuildFresh rebuild carries MtpState");
		assert_eq!(state.pairs_fed, boundary1 - 1);
	}

	/// Non-boundary fallback insertion (empty transcript renders only the
	/// generation prompt, so no strict-prefix boundary exists): the pooled
	/// ids are `full_ids ++ emitted[..committed_len]` and the entry
	/// carries `DecodeOutcome.mtp`, aligned to exactly that sequence.
	#[test]
	fn fallback_insertion_stores_decode_outcome_mtp_aligned() {
		let (_dir, session) = load_mtp_session();
		let messages: Vec<ChatMessage> = Vec::new();
		let full_ids = session.encode_chat(&messages).unwrap();
		assert!(
			session
				.conversation_boundary_len(&messages, None, None, &full_ids)
				.map(|len| len == 0)
				.unwrap_or(true),
			"empty transcript must not produce a usable boundary"
		);
		let mut emitted = Vec::new();
		let reply = session
			.generate_cached(&messages, None, spec_on(6), |tok| {
				emitted.push(tok.id);
				true
			})
			.unwrap();
		assert!(reply.usage.completion_tokens > 0);

		// The fallback entry's ids are `full_ids ++ committed` where
		// committed is a prefix of the emitted ledger, so querying with
		// `full_ids ++ emitted` finds it.
		let mut query = full_ids.clone();
		query.extend_from_slice(&emitted);
		let entry = pool_entry(&session, &query).expect("fallback entry");
		assert!(is_prefix(&full_ids, &entry.ids));
		assert!(is_prefix(&entry.ids, &query));
		let state = entry
			.mtp
			.expect("spec-on fallback insertion carries DecodeOutcome.mtp");
		assert_eq!(
			state.pairs_fed,
			entry.ids.len() - 1,
			"fallback-stored MtpState must be aligned to the pooled ids"
		);
	}

	/// Non-pristine caller caches disable speculation (Disabled state):
	/// the call still succeeds and behaves target-only.
	#[test]
	fn non_pristine_caller_caches_disable_speculation() {
		let dir = write_tiny_model(true).unwrap();
		let session = Session::load(dir.path()).unwrap();
		let prompt = session.tokenizer().encode("hello world").unwrap();
		let mut caches = session.new_caches();
		// Pre-feed part of the prompt so the caches are non-pristine.
		let head = Array::from_slice(&prompt[..1], &[1, 1]).unwrap();
		let _ = session.debug_forward(&head, &mut caches).unwrap();
		let spec_off = session
			.generate_with_caches(
				&prompt[1..],
				&mut caches,
				GenerateOptions {
					max_tokens: 4,
					speculative_tokens: Some(2),
					..GenerateOptions::default()
				},
				|_| true,
			)
			.unwrap();
		// Same greedy tokens as an explicitly spec-off run from scratch.
		let mut fresh = session.new_caches();
		let head = Array::from_slice(&prompt[..1], &[1, 1]).unwrap();
		let _ = session.debug_forward(&head, &mut fresh).unwrap();
		let plain = session
			.generate_with_caches(
				&prompt[1..],
				&mut fresh,
				GenerateOptions {
					max_tokens: 4,
					..GenerateOptions::default()
				},
				|_| true,
			)
			.unwrap();
		assert_eq!(spec_off, plain);
	}
}

/// Append one image's `boi + image_token × num_soft_tokens + eoi` span.
fn push_image_span(
	out: &mut Vec<u32>,
	num_soft_tokens: i32,
	image_token_id: u32,
	boi: u32,
	eoi: u32,
) {
	out.push(boi);
	for _ in 0..num_soft_tokens {
		out.push(image_token_id);
	}
	out.push(eoi);
}
