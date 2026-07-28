// emelex patch (not upstream): this entire module is an emelex addition —
// the speculative decode round behind an internal seam.

//! One decode round — target-only or speculative — behind the private
//! [`RoundOps`] seam.
//!
//! [`RoundDriver::run_round`] implements the decode-loop state
//! machine: the ordered transition at the current token `x` (emit →
//! EOS/cancel → budget/forced close → final slot → window preflight →
//! draft), the target-only round, the speculative round
//! (draft → verify → decide → sequential emission → single reconciliation
//! → MTP commit → stage-4 successor selection), forced close (empty-C
//! split by trigger), and the failure-transition table. Session's real
//! MLX implementation is [`SessionOps`]; the `#[cfg(test)]` fake drives
//! every disposition and failure row with batch-size-invariant,
//! deterministic f64 host arithmetic.
//!
//! Completion-path ordering (Operation-fallibility contract): within any
//! round completion, stages run strictly as (1) target reconciliation,
//! (2) hidden detaches, (3) MTP truncate + commit replay (or single
//! pair-commit), (4) successor selection — the ONLY point where a
//! successor token is drawn, and only on continuing dispositions.

use std::sync::Arc;

use crate::engine::{
	array::Array,
	error::{Error, Result},
	generate::{Emit, GeneratedToken, SPECULATIVE_TOKENS_CEILING, SpeculationStats, TokenEmitter},
	models::{
		Model,
		cache::{LayerCache, LayerRollback},
		mtp::MtpCaches,
	},
	ops,
	reasoning::ReasoningBudget,
	sampling::{self, Sampler},
};

/// Per-position sampling data for "the token AFTER this row's token".
///
/// `Greedy` carries the device argmax already `item`'d to host; `Sampled`
/// carries the filtered + renormalized distribution ([`Sampler::probs`]
/// contract). Successor distributions are host data and survive all cache
/// mutations.
#[derive(Debug, Clone)]
pub enum Dist {
	Greedy(u32),
	Sampled(Probabilities),
}

/// One normalized probability row. Batched target rows share one host buffer,
/// avoiding both per-row device synchronization and full-vocabulary copies.
#[derive(Debug, Clone)]
pub struct Probabilities {
	values: Arc<[f32]>,
	start: usize,
	end: usize,
}

impl Probabilities {
	fn shared(values: Arc<[f32]>, start: usize, end: usize) -> Self {
		Self { values, start, end }
	}

	fn as_slice(&self) -> &[f32] {
		&self.values[self.start..self.end]
	}
}

impl From<Vec<f32>> for Probabilities {
	fn from(values: Vec<f32>) -> Self {
		let end = values.len();
		Self {
			values: values.into(),
			start: 0,
			end,
		}
	}
}

impl PartialEq for Probabilities {
	fn eq(&self, other: &Self) -> bool {
		self.as_slice() == other.as_slice()
	}
}

/// Failure classification for [`RoundOps`] operations, mirroring the
/// decode loop's failure-transition table.
#[derive(Debug)]
pub enum OpError {
	/// A target forward failed mid-mutation (destructive `take()`s make
	/// restoration impossible) — invalid-cache exit: the round exits with an
	/// error and the caches must not be reused.
	Invalid(Error),
	/// The MTP forward failed — blanket rule: discard the entire MTP
	/// state and continue target-only.
	MtpForward(Error),
	/// A host-side operation failed AFTER every structural cache mutation
	/// of the call completed. `salvage` carries the per-row [`Dist`]s that
	/// had materialized before the failing operation (the real impl
	/// materializes dists before detaching hiddens, so a detach failure
	/// salvages every dist).
	Host { error: Error, salvage: Vec<Dist> },
}

impl OpError {
	fn into_error(self) -> Error {
		match self {
			OpError::Invalid(e) | OpError::MtpForward(e) => e,
			OpError::Host { error, .. } => error,
		}
	}
}

pub type OpResult<T> = std::result::Result<T, OpError>;

/// Ordered-trace events the driver reports back through
/// [`RoundOps::trace`] so the fake can assert stage ordering: the stage-4
/// successor hook fires exactly once, only after final
/// cache/MTP disposition, never on exiting rounds).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceEvent {
	/// Stage-4 successor selection is being performed.
	SuccessorSelect,
}

/// The private seam between the round driver and the model/cache stack.
///
/// The target side tracks its own absolute offset (committed prompt +
/// decode length). Hidden rows returned from feeds are DETACHED (real:
/// contiguous + eval'd views of a per-feed detached block; fake:
/// `Vec<f64>`), so they survive later cache mutations. Feeds return one
/// hidden per fed row **only when the MTP module is alive** (the hidden
/// rows exist to become MTP pair inputs); with MTP dead the hidden vector
/// is empty and must not be indexed.
pub trait RoundOps {
	type Hidden: Clone;
	type Snapshot;

	/// Whether any target layer carries recurrent (non-attention) state.
	fn is_hybrid(&self) -> bool;
	/// Absolute target offset: tokens the target caches contain.
	fn target_pos(&self) -> usize;
	/// Window preflight: would feeding `added` positions irreversibly
	/// trim any windowed target layer?
	fn window_would_trim(&self, added: usize) -> bool;

	/// Feed `ids` to the target, evaluating the logits tensor once and
	/// materializing one [`Dist`] per row (row `i` describes the successor
	/// of `ids[i]`), plus detached hidden rows when MTP is alive.
	fn verify_feed(
		&mut self,
		ids: &[u32],
		sampler: &Sampler,
	) -> OpResult<(Vec<Self::Hidden>, Vec<Dist>)>;
	/// Feed `ids` WITHOUT evaluating logits, except optionally the last
	/// row's dist (forced-close replay draws its post-close successor from
	/// it). Returns detached hidden rows when MTP is alive.
	fn replay_feed(
		&mut self,
		ids: &[u32],
		want_last_dist: bool,
		sampler: &Sampler,
	) -> OpResult<(Vec<Self::Hidden>, Option<Dist>)>;
	/// Capture per-layer rollback state (attention: offsets only — never
	/// clones KV buffers; recurrent: O(1) refcount bumps).
	fn capture(&self) -> Self::Snapshot;
	/// Restore a snapshot. Infallible by contract under correct use; an
	/// `Err` is a logic-error tripwire the driver maps to invalid-cache exit.
	fn restore(&mut self, snap: &Self::Snapshot) -> OpResult<()>;
	/// Truncate every attention layer to `abs_offset`. In-range truncation
	/// is infallible by contract; the out-of-range guard is invalid-cache exit.
	fn truncate_target(&mut self, abs_offset: usize) -> OpResult<()>;

	fn mtp_alive(&self) -> bool;
	fn mtp_pairs(&self) -> usize;
	/// One draft step (L = 1): consumes `(prev, id)`, returns the detached
	/// recycle hidden and the draft distribution.
	fn mtp_step(
		&mut self,
		id: u32,
		prev: &Self::Hidden,
		sampler: &Sampler,
	) -> OpResult<(Self::Hidden, Dist)>;
	/// Batched pair commit / priming: `ids[i]` fuses with `prevs[i]`.
	/// Logits are NEVER evaluated.
	fn mtp_commit(&mut self, ids: &[u32], prevs: &[Self::Hidden]) -> OpResult<()>;
	fn mtp_truncate(&mut self, pairs_fed: usize) -> OpResult<()>;
	fn mtp_discard(&mut self);

	/// Ordered-trace hook for the fake; the real impl ignores it.
	fn trace(&mut self, _event: TraceEvent) {}
}

/// How one round ended.
#[derive(Debug)]
pub enum RoundEnd {
	/// Next-round entry invariant restored; `next` is sampled, unfed and
	/// unemitted.
	Continue { next: u32 },
	/// EOS emitted (uncommitted).
	Finished,
	/// The `on_token` callback cancelled generation.
	Aborted,
	/// Slot exhaustion (final slot committed / output budget hit).
	Budget,
}

/// MTP handling mode for the shared target-only transition.
enum MtpMode {
	/// MTP dead or discarded: no MTP work.
	Dead,
	/// Healthy aligned MTP (ordinary target-only / window-skip): commit
	/// the shifted pair `(frontier, x)`.
	Commit,
	/// Spec-resume recovery: truncate the MTP back to `entry_pairs`
	/// first, then commit the `(frontier_entry, x)` pair.
	ResumeAt { entry_pairs: usize },
}

/// Outcome of the draft phase.
enum DraftOutcome {
	Drafted(Vec<u32>, Vec<Dist>),
	/// MTP forward failed: blanket discard, continue target-only.
	Discard,
	/// Host-side failure between MTP forwards (incl. a draft draw
	/// failure): truncate MTP to entry, re-commit the `x` pair,
	/// speculation resumes next round.
	Resume,
}

/// Outcome of the verify phase.
enum Verified<O: RoundOps> {
	Ready {
		hiddens: Vec<O::Hidden>,
		dists: Vec<Dist>,
		post_x: Option<O::Snapshot>,
	},
	/// A failure row already completed the round.
	Done(RoundEnd),
}

/// The acceptance decision; verification decides and never draws the successor.
struct Verdict {
	accepted: usize,
	/// Constructed, renormalized, undrawn successor data: residual at the
	/// rejection depth (`a < k`), bonus row (`a = k`), or the precomputed
	/// greedy successor token.
	successor: Dist,
}

/// Event observed while sequentially emitting accepted drafts.
enum SpecEvent {
	None,
	Eos,
	Cancel,
	Close(&'static str),
	Slot,
	EmitErr(Error),
}

/// Result of the close-prefix emission pass.
struct ClosePrefix {
	accepted: usize,
	cancelled: bool,
	error: Option<Error>,
}

/// Reconciliation result for the speculative round.
enum Reconciled<H> {
	Committed {
		frontier_h: H,
		prevs: Vec<H>,
	},
	/// Target reconciled to the retained prefix but the replay-pass
	/// hidden detach failed: MTP must be discarded; the stage-4 draw
	/// still proceeds (post-reconciliation detach-failure row).
	DetachFailed,
}

/// Everything the post-verify phases of a speculative round need.
struct SpecCtx<O: RoundOps> {
	entry_pos: usize,
	entry_pairs: usize,
	entry_snap: O::Snapshot,
	frontier_entry: O::Hidden,
	k: usize,
	drafts: Vec<u32>,
	hiddens: Vec<O::Hidden>,
	dists: Vec<Dist>,
	post_x: Option<O::Snapshot>,
}

fn internal(msg: &str) -> Error {
	Error::Model(format!("spec: {msg}"))
}

/// Per-call decode driver: owns the emitter/budget/sampler and the two
/// ledgers, and runs one [`RoundDriver::run_round`] per outer-loop
/// iteration. `decode_loop` (generate.rs) is a thin wrapper: prefill →
/// loop `run_round` → `DecodeOutcome`.
///
/// RNG accounting per speculative round: draft draws, then acceptance
/// uniforms, then at most one successor draw — in that order, successor
/// last; exiting rounds never draw a successor and never consume RNG for
/// one.
pub struct RoundDriver<'a, O: RoundOps, F: FnMut(GeneratedToken) -> bool> {
	ops: O,
	emitter: TokenEmitter<'a, F>,
	budget: Option<ReasoningBudget>,
	sampler: Sampler,
	max_tokens: usize,
	spec_k: Option<usize>,
	committed_len: usize,
	frontier: Option<O::Hidden>,
	stats: SpeculationStats,
	reasoning_forced_closed: bool,
	/// Tokens counted against `max_tokens` (close-marker tokens bypass
	/// this, matching the historical loop-counter behavior).
	used: usize,
}

impl<'a, O: RoundOps, F: FnMut(GeneratedToken) -> bool> RoundDriver<'a, O, F> {
	pub fn new(
		ops: O,
		emitter: TokenEmitter<'a, F>,
		budget: Option<ReasoningBudget>,
		sampler: Sampler,
		max_tokens: usize,
		spec_k: Option<usize>,
		frontier: Option<O::Hidden>,
	) -> Self {
		RoundDriver {
			ops,
			emitter,
			budget,
			sampler,
			max_tokens,
			spec_k,
			committed_len: 0,
			frontier,
			stats: SpeculationStats::default(),
			reasoning_forced_closed: false,
			used: 0,
		}
	}

	pub fn used(&self) -> usize {
		self.used
	}

	pub fn committed_len(&self) -> usize {
		self.committed_len
	}

	pub fn ops(&self) -> &O {
		&self.ops
	}

	#[cfg(test)]
	pub fn ops_mut(&mut self) -> &mut O {
		&mut self.ops
	}

	#[cfg(test)]
	pub fn frontier(&self) -> Option<&O::Hidden> {
		self.frontier.as_ref()
	}

	#[cfg(test)]
	pub fn stats(&self) -> &SpeculationStats {
		&self.stats
	}

	pub fn reasoning_forced_closed(&self) -> bool {
		self.reasoning_forced_closed
	}

	/// Tear down into `(emitted, committed_len, stats, ops, frontier)`.
	pub fn finish(self) -> (Vec<u32>, usize, SpeculationStats, O, Option<O::Hidden>) {
		(
			self.emitter.into_emitted(),
			self.committed_len,
			self.stats,
			self.ops,
			self.frontier,
		)
	}

	/// One decode round for the current sampled-but-unfed token `x`.
	///
	/// Round-entry invariant: target caches = exactly the committed
	/// prefix; `x` sampled/unfed/unemitted; `frontier` = detached
	/// pre-norm hidden of the last committed token (present iff MTP is
	/// live); MTP `pairs_fed` = committed length − 1 (prompt-relative).
	pub fn run_round(&mut self, x: u32) -> Result<RoundEnd> {
		debug_assert!(self.used < self.max_tokens);
		// Ordered transition at x: emission first — EOS and cancellation
		// win before any budget observation or forward pass. An emitter
		// failure here is exact-prefix exit at entry: target untouched, MTP
		// untouched, committed unchanged, zero callbacks for x.
		let raw_text = match self.emitter.emit(x)? {
			Emit::Eos => {
				self.used += 1;
				return Ok(RoundEnd::Finished);
			}
			Emit::Cancelled => {
				self.used += 1;
				return Ok(RoundEnd::Aborted);
			}
			Emit::Continue { raw_text } => {
				self.used += 1;
				raw_text
			}
		};
		// Only a continuing token reaches budget.observe.
		let forced_close = self.budget.as_mut().and_then(|b| b.observe(&raw_text));
		if let Some(close_marker) = forced_close {
			return self.forced_close_at_x(x, close_marker);
		}
		// Final slot: feed + commit + pair-commit, no draft, no
		// successor sample (deliberate change from the historical wasted
		// final-iteration sample).
		if self.used >= self.max_tokens {
			return self.final_slot(x);
		}
		// Speculative round when available; window preflight runs before
		// the first forward_mtp and demotes to a target-only round with
		// MTP preserved.
		if let Some(k) = self.spec_depth() {
			if !self.ops.window_would_trim(k + 1) {
				return self.speculative_round(x, k);
			}
		}
		let mode = if self.ops.mtp_alive() {
			MtpMode::Commit
		} else {
			MtpMode::Dead
		};
		self.feed_x_target_only(x, &mode, true)
	}

	/// `k = min(config_k, SPECULATIVE_TOKENS_CEILING, remaining − 1)`;
	/// `None` when speculation is unavailable this round.
	fn spec_depth(&self) -> Option<usize> {
		let k = self.spec_k?;
		if !self.ops.mtp_alive() || self.frontier.is_none() {
			return None;
		}
		// `used` already counts x, so `max_tokens - used` == remaining−1;
		// the final slot was handled above, so this is >= 1.
		Some(
			k.min(SPECULATIVE_TOKENS_CEILING)
				.min(self.max_tokens - self.used),
		)
	}

	fn discard_mtp(&mut self) {
		self.ops.mtp_discard();
		self.frontier = None;
	}

	fn restore_or_exit(&mut self, snap: &O::Snapshot) -> Result<()> {
		self.ops.restore(snap).map_err(OpError::into_error)
	}

	fn truncate_or_exit(&mut self, abs: usize) -> Result<()> {
		self.ops.truncate_target(abs).map_err(OpError::into_error)
	}

	/// Stage-4 successor selection: the single draw point. Greedy emits
	/// the precomputed token (zero RNG); sampled draws once via
	/// `sample_from_probs`.
	fn draw(&mut self, dist: &Dist) -> Result<u32> {
		self.ops.trace(TraceEvent::SuccessorSelect);
		match dist {
			Dist::Greedy(t) => Ok(*t),
			Dist::Sampled(p) => self.sampler.sample_from_probs(p.as_slice()),
		}
	}

	fn record_round(&mut self, accepted: usize) {
		// Counter arithmetic (one-based depth buckets, saturating adds)
		// lives on `SpeculationStats` itself - see generate.rs. `drafted`
		// is NOT recorded here: it counts at draft time inside
		// [`RoundDriver::draft`], so failed rounds' proposals count too.
		self.stats.record_round(accepted);
	}

	/// The ordinary non-speculative transition: feed `[x]`,
	/// commit it, run the mode's MTP disposition (pair-commit BEFORE
	/// successor selection), then optionally draw the successor.
	fn feed_x_target_only(
		&mut self,
		x: u32,
		mode: &MtpMode,
		draw_successor: bool,
	) -> Result<RoundEnd> {
		let alive = self.ops.mtp_alive() && !matches!(mode, MtpMode::Dead);
		let (hiddens, dists) = match self.ops.verify_feed(&[x], &self.sampler) {
			Ok(v) => v,
			// Target forward failure → invalid-cache exit.
			Err(OpError::Invalid(e) | OpError::MtpForward(e)) => return Err(e),
			Err(OpError::Host { error, salvage }) => {
				// Feed structurally complete: commit x. H_x is
				// unobtainable → frontier-detach row: discard MTP,
				// continue target-only from the salvaged dist, else
				// exact-prefix exit at `…, x`.
				self.committed_len += 1;
				self.discard_mtp();
				if !draw_successor {
					return Ok(RoundEnd::Budget);
				}
				return match salvage.into_iter().next() {
					Some(d) => Ok(RoundEnd::Continue {
						next: self.draw(&d)?,
					}),
					None => Err(error),
				};
			}
		};
		self.committed_len += 1;
		if alive {
			let Some(frontier) = self.frontier.clone() else {
				return Err(internal("live MTP without a frontier"));
			};
			let truncated = match mode {
				MtpMode::ResumeAt { entry_pairs } => self.ops.mtp_truncate(*entry_pairs).is_ok(),
				_ => true,
			};
			let committed = truncated
				&& self
					.ops
					.mtp_commit(&[x], std::slice::from_ref(&frontier))
					.is_ok();
			if committed {
				// pairs_fed advanced to t; frontier = detach(H_x).
				self.frontier = hiddens.first().cloned();
			} else {
				// Blanket MTP-forward-failure rule.
				self.discard_mtp();
			}
		}
		if !draw_successor {
			return Ok(RoundEnd::Budget);
		}
		let Some(dist) = dists.first() else {
			return Err(internal("verify feed returned no dist"));
		};
		// No clone before the draw: in sampled mode a `Dist::Sampled` row
		// is a full-vocab Vec (~1 MB at V=250k) and this is the ordinary
		// spec-off per-token path.
		let next = self.draw(dist)?;
		Ok(RoundEnd::Continue { next })
	}

	/// Final output slot: feed `[x]`, commit it and the MTP pair, exit —
	/// no draft, no successor sample, logits never evaluated.
	fn final_slot(&mut self, x: u32) -> Result<RoundEnd> {
		let alive = self.ops.mtp_alive();
		match self.ops.replay_feed(&[x], false, &self.sampler) {
			Ok((hiddens, _)) => {
				self.committed_len += 1;
				if alive {
					let Some(frontier) = self.frontier.clone() else {
						return Err(internal("live MTP without a frontier"));
					};
					if self
						.ops
						.mtp_commit(&[x], std::slice::from_ref(&frontier))
						.is_ok()
					{
						self.frontier = hiddens.first().cloned();
					} else {
						self.discard_mtp();
					}
				}
				Ok(RoundEnd::Budget)
			}
			Err(OpError::Invalid(e) | OpError::MtpForward(e)) => Err(e),
			Err(OpError::Host { .. }) => {
				// Frontier-detach failure in the final slot: the feed
				// committed; return with mtp = None, never drawing.
				self.committed_len += 1;
				self.discard_mtp();
				Ok(RoundEnd::Budget)
			}
		}
	}

	// ------------------------------------------------------------------
	// Forced close
	// ------------------------------------------------------------------

	fn forced_close_at_x(&mut self, x: u32, close_marker: &'static str) -> Result<RoundEnd> {
		// An encode failure surfaces with the target still at entry —
		// exact-prefix exit; x stays emitted but uncommitted, and the budget
		// fired exactly once.
		let close_ids = self.emitter.encode_close(close_marker)?;
		if close_ids.is_empty() {
			// Empty close encoding, trigger x: an ordinary feed-and-commit
			// of x (with the MTP pair when live); never arm the immediate
			// duplicate-close filter; the final slot skips the
			// successor draw.
			let mode = if self.ops.mtp_alive() {
				MtpMode::Commit
			} else {
				MtpMode::Dead
			};
			return self.feed_x_target_only(x, &mode, self.used < self.max_tokens);
		}
		// Callbacks FIRST, forward SECOND: emitting first pins the
		// accepted close prefix C, and the single replay feeds exactly
		// `[x] + C` — the caches and the committed ledger cannot diverge,
		// whatever the callback does.
		let close = self.emit_close_prefix(&close_ids);
		let entry_pairs = self.ops.mtp_pairs();
		let mut seq = Vec::with_capacity(1 + close.accepted);
		seq.push(x);
		seq.extend_from_slice(&close_ids[..close.accepted]);
		self.close_replay(&seq, close, close_marker, entry_pairs)
	}

	fn forced_close_at_draft(
		&mut self,
		x: u32,
		ctx: SpecCtx<O>,
		r: usize,
		close_marker: &'static str,
	) -> Result<RoundEnd> {
		let close_ids = match self.emitter.encode_close(close_marker) {
			Ok(ids) => ids,
			Err(e) => {
				// Reconcile the retained prefix so the caches match the
				// committed ledger (exact-prefix exit), then propagate.
				self.reconcile_and_discard(&ctx, r)?;
				return Err(e);
			}
		};
		if close_ids.is_empty() {
			// Empty close encoding at trigger d_i: NO extra feed —
			// reconcile exactly as the slot-exhaustion row, commit the
			// retained prefix's MTP pairs, and draw the successor from
			// the already-materialized verify row-i distribution. Never
			// arm the filters.
			self.reconcile_and_commit(x, &ctx, r)?;
			if self.used >= self.max_tokens {
				return Ok(RoundEnd::Budget);
			}
			let Some(dist) = ctx.dists.get(r) else {
				return Err(internal("missing verify row dist at close trigger"));
			};
			let next = self.draw(dist)?;
			return Ok(RoundEnd::Continue { next });
		}
		let close = self.emit_close_prefix(&close_ids);
		// Reconcile the target to ENTRY uniformly — never restore post-x
		// and then replay a sequence containing x.
		self.restore_or_exit(&ctx.entry_snap)?;
		let mut seq = Vec::with_capacity(1 + r + close.accepted);
		seq.push(x);
		seq.extend_from_slice(&ctx.drafts[..r]);
		seq.extend_from_slice(&close_ids[..close.accepted]);
		let entry_pairs = ctx.entry_pairs;
		// Snapshot lifetime: drop the round context — including the entry
		// snapshot's rollback state — BEFORE the close-replay forward, so
		// a held snapshot never blocks first-multiply donation on the
		// recurrent layers during that (potentially long) replay.
		drop(ctx);
		self.close_replay(&seq, close, close_marker, entry_pairs)
	}

	fn emit_close_prefix(&mut self, close_ids: &[u32]) -> ClosePrefix {
		let mut accepted = 0usize;
		let mut cancelled = false;
		let mut error = None;
		for &id in close_ids {
			match self.emitter.emit_forced(id) {
				Ok(Emit::Continue { .. }) => accepted += 1,
				Ok(Emit::Eos | Emit::Cancelled) => {
					cancelled = true;
					break;
				}
				Err(e) => {
					// The failed token itself was never emitted
					// (`emit_forced` decodes before pushing); the
					// already-emitted close prefix must still be fed
					// before the error propagates (exact-prefix exit).
					error = Some(e);
					break;
				}
			}
		}
		ClosePrefix {
			accepted,
			cancelled,
			error,
		}
	}

	/// Replay `seq` (trigger prefix + accepted close prefix) from the
	/// target entry state, commit both ledgers, replay the MTP pairs, arm
	/// the display filters, and draw the post-close successor when the
	/// round continues. Close ids bypass the loop counter (a final-slot
	/// close may exceed `max_tokens`) but never draw the unused
	/// post-close successor.
	fn close_replay(
		&mut self,
		seq: &[u32],
		close: ClosePrefix,
		close_marker: &'static str,
		entry_pairs: usize,
	) -> Result<RoundEnd> {
		let alive = self.ops.mtp_alive();
		let over_budget = self.used >= self.max_tokens;
		let want_dist = close.error.is_none() && !close.cancelled && !over_budget;
		let (hiddens, last_dist) = match self.ops.replay_feed(seq, want_dist, &self.sampler) {
			Ok(v) => v,
			// Forced-close target replay fails → invalid-cache exit.
			Err(OpError::Invalid(e) | OpError::MtpForward(e)) => {
				return Err(e);
			}
			Err(OpError::Host { error, salvage }) => {
				// Replay structurally complete; frontier-detach /
				// dist materialization failed: commit the replayed
				// seq, discard MTP, then per row.
				self.committed_len += seq.len();
				self.discard_mtp();
				if let Some(e) = close.error {
					return Err(e);
				}
				if close.cancelled {
					return Ok(RoundEnd::Aborted);
				}
				if over_budget {
					return Ok(RoundEnd::Budget);
				}
				self.reasoning_forced_closed = true;
				self.emitter.arm_close_filters(close_marker);
				return match salvage.into_iter().next_back() {
					Some(d) => Ok(RoundEnd::Continue {
						next: self.draw(&d)?,
					}),
					None => Err(error),
				};
			}
		};
		self.committed_len += seq.len();
		if let Some(e) = close.error {
			// Emission failure on a close token: the replayed prefix is
			// committed, MTP is discarded, then the error propagates
			// (exact-prefix exit). Never repeat callbacks.
			self.discard_mtp();
			return Err(e);
		}
		if alive {
			let Some(frontier_entry) = self.frontier.clone() else {
				return Err(internal("live MTP without a frontier"));
			};
			// MTP: reset to entry, replay every shifted pair for the
			// replayed sequence.
			let mut committed = self.ops.mtp_truncate(entry_pairs).is_ok();
			if committed {
				let mut prevs = Vec::with_capacity(seq.len());
				prevs.push(frontier_entry);
				prevs.extend(hiddens[..seq.len() - 1].iter().cloned());
				committed = self.ops.mtp_commit(seq, &prevs).is_ok();
			}
			if committed {
				self.frontier = hiddens.last().cloned();
			} else {
				// Forced-close MTP replay fails: target stays valid,
				// MTP discarded, stage-4 draw still proceeds.
				self.discard_mtp();
			}
		}
		if close.cancelled {
			// A cancelled close never samples a successor and never arms
			// the display filters.
			return Ok(RoundEnd::Aborted);
		}
		if over_budget {
			return Ok(RoundEnd::Budget);
		}
		self.reasoning_forced_closed = true;
		self.emitter.arm_close_filters(close_marker);
		let Some(dist) = last_dist else {
			return Err(internal("close replay returned no dist"));
		};
		let next = self.draw(&dist)?;
		Ok(RoundEnd::Continue { next })
	}

	// ------------------------------------------------------------------
	// Speculative round
	// ------------------------------------------------------------------

	fn speculative_round(&mut self, x: u32, k: usize) -> Result<RoundEnd> {
		let entry_pos = self.ops.target_pos();
		let entry_pairs = self.ops.mtp_pairs();
		let entry_snap = self.ops.capture();
		let Some(frontier_entry) = self.frontier.clone() else {
			return Err(internal("speculative round without a frontier"));
		};

		// --- Draft (provisional MTP mutations) ---
		let (drafts, draft_dists) = match self.draft(x, k, &frontier_entry) {
			DraftOutcome::Drafted(d, q) => (d, q),
			DraftOutcome::Discard => {
				self.discard_mtp();
				return self.feed_x_target_only(x, &MtpMode::Dead, true);
			}
			DraftOutcome::Resume => {
				return self.feed_x_target_only(x, &MtpMode::ResumeAt { entry_pairs }, true);
			}
		};

		// --- Verify ---
		let (hiddens, dists, post_x) = match self.verify(x, &drafts, entry_pos)? {
			Verified::Done(end) => return Ok(end),
			Verified::Ready {
				hiddens,
				dists,
				post_x,
			} => (hiddens, dists, post_x),
		};

		// --- Acceptance decision (DECIDES ONLY; successor undrawn) ---
		let Some(verdict) = self.decide(&drafts, &draft_dists, &dists) else {
			return self.invariant_err_recovery(
				x,
				entry_pos,
				entry_pairs,
				&frontier_entry,
				post_x.as_ref(),
				&hiddens,
				&dists,
			);
		};
		self.record_round(verdict.accepted);

		// --- Sequential emission decides the retained prefix ---
		let (r, event) = self.emit_drafts(&drafts, verdict.accepted);

		let ctx = SpecCtx {
			entry_pos,
			entry_pairs,
			entry_snap,
			frontier_entry,
			k,
			drafts,
			hiddens,
			dists,
			post_x,
		};
		match event {
			SpecEvent::Close(marker) => self.forced_close_at_draft(x, ctx, r, marker),
			SpecEvent::EmitErr(e) => {
				// Sequential-emission failure: reconcile to the emitted
				// prefix, discard MTP, then propagate (exact-prefix exit; the
				// reconcile replay's own failure → invalid-cache exit).
				self.reconcile_and_discard(&ctx, r)?;
				Err(e)
			}
			event => {
				// Stage 1-3: single reconciliation, then MTP truncate +
				// batched commit replay; rollback state drops with `ctx`
				// on every path including the r = k fast path.
				self.reconcile_and_commit(x, &ctx, r)?;
				match event {
					SpecEvent::None => {
						// Stage 4: continuing disposition only.
						let next = self.draw(&verdict.successor)?;
						Ok(RoundEnd::Continue { next })
					}
					SpecEvent::Eos => Ok(RoundEnd::Finished),
					SpecEvent::Cancel => Ok(RoundEnd::Aborted),
					SpecEvent::Slot => Ok(RoundEnd::Budget),
					SpecEvent::Close(_) | SpecEvent::EmitErr(_) => {
						Err(internal("unreachable spec event"))
					}
				}
			}
		}
	}

	/// Recursive drafting from `(frontier, x)` then `(recycle_i, d_i)`.
	/// Drafts are sampled from the filtered renormalized `q_i`
	/// exclusively via `sample_from_probs` (greedy: device argmax).
	fn draft(&mut self, x: u32, k: usize, frontier_entry: &O::Hidden) -> DraftOutcome {
		let mut drafts = Vec::with_capacity(k);
		let mut dists = Vec::with_capacity(k);
		let mut prev = frontier_entry.clone();
		let mut prev_id = x;
		for _ in 0..k {
			let (recycle, dist) = match self.ops.mtp_step(prev_id, &prev, &self.sampler) {
				Ok(v) => v,
				Err(OpError::Host { .. }) => return DraftOutcome::Resume,
				Err(OpError::MtpForward(_) | OpError::Invalid(_)) => {
					return DraftOutcome::Discard;
				}
			};
			let drafted = match &dist {
				Dist::Greedy(t) => *t,
				Dist::Sampled(q) => match self.sampler.sample_from_probs(q.as_slice()) {
					Ok(t) => t,
					// A draft draw failure is a host-side failure between
					// MTP forwards.
					Err(_) => return DraftOutcome::Resume,
				},
			};
			// `drafted` counts at DRAFT time (tokens proposed): a round
			// that later fails - mid-draft Resume/Discard, verify-phase
			// failure, invariant-Err - keeps the proposals it already made.
			self.stats.record_drafted(1);
			drafts.push(drafted);
			dists.push(dist);
			prev = recycle;
			prev_id = drafted;
		}
		DraftOutcome::Drafted(drafts, dists)
	}

	/// Verify feed: hybrid feeds `[x]` (post-x snapshot) then
	/// `[d1…dk]`; pure attention feeds `[x, d1…dk]` once. Row 0 = x.
	fn verify(&mut self, x: u32, drafts: &[u32], entry_pos: usize) -> Result<Verified<O>> {
		if !self.ops.is_hybrid() {
			let mut feed = Vec::with_capacity(1 + drafts.len());
			feed.push(x);
			feed.extend_from_slice(drafts);
			return match self.ops.verify_feed(&feed, &self.sampler) {
				Ok((hiddens, dists)) => Ok(Verified::Ready {
					hiddens,
					dists,
					post_x: None,
				}),
				// Target verify forward fails → invalid-cache exit.
				Err(OpError::Invalid(e) | OpError::MtpForward(e)) => Err(e),
				Err(OpError::Host { error, salvage }) => self
					.verify_host_recovery(error, salvage, None, entry_pos)
					.map(Verified::Done),
			};
		}
		let (h0, d0) = match self.ops.verify_feed(&[x], &self.sampler) {
			Ok(v) => v,
			Err(OpError::Invalid(e) | OpError::MtpForward(e)) => return Err(e),
			Err(OpError::Host { error, salvage }) => {
				// Chunk-1 host failure: the recurrent state is already
				// post-x (the forward completed structurally), so no
				// restore is needed.
				return self
					.verify_host_recovery(error, salvage, None, entry_pos)
					.map(Verified::Done);
			}
		};
		let post_x = self.ops.capture();
		match self.ops.verify_feed(drafts, &self.sampler) {
			Ok((hr, dr)) => {
				let mut hiddens = h0;
				hiddens.extend(hr);
				let mut dists = d0;
				dists.extend(dr);
				Ok(Verified::Ready {
					hiddens,
					dists,
					post_x: Some(post_x),
				})
			}
			Err(OpError::Invalid(e) | OpError::MtpForward(e)) => Err(e),
			Err(OpError::Host { error, .. }) => {
				// Chunk-2 host failure: row-0's dist came from chunk 1 —
				// it is the recovery's stage-4 source.
				self.verify_host_recovery(error, d0, Some(&post_x), entry_pos)
					.map(Verified::Done)
			}
		}
	}

	/// Verify-phase host-failure row: verify forward succeeded, nothing
	/// emitted beyond x. Reconcile to `…, x`, discard MTP, then attempt
	/// the stage-4 draw from the salvaged verify row-0 dist; without one,
	/// exact-prefix exit at `…, x`.
	fn verify_host_recovery(
		&mut self,
		error: Error,
		salvage: Vec<Dist>,
		post_x: Option<&O::Snapshot>,
		entry_pos: usize,
	) -> Result<RoundEnd> {
		if let Some(snap) = post_x {
			self.restore_or_exit(snap)?;
		}
		self.truncate_or_exit(entry_pos + 1)?;
		self.committed_len += 1;
		self.discard_mtp();
		match salvage.into_iter().next() {
			Some(d) => Ok(RoundEnd::Continue {
				next: self.draw(&d)?,
			}),
			None => Err(error),
		}
	}

	/// Acceptance decision; `None` routes to the invariant-Err recovery
	/// row (zero-`q`, all-zero residual, length mismatch).
	fn decide(&mut self, drafts: &[u32], draft_dists: &[Dist], dists: &[Dist]) -> Option<Verdict> {
		if self.sampler.is_greedy() {
			let mut target_argmax = Vec::with_capacity(dists.len());
			for d in dists {
				match d {
					Dist::Greedy(t) => target_argmax.push(*t),
					Dist::Sampled(_) => return None,
				}
			}
			let v = sampling::verify_greedy(drafts, &target_argmax).ok()?;
			Some(Verdict {
				accepted: v.accepted,
				successor: Dist::Greedy(v.next),
			})
		} else {
			let mut q = Vec::with_capacity(draft_dists.len());
			for d in draft_dists {
				match d {
					Dist::Sampled(p) => q.push(p.as_slice()),
					Dist::Greedy(_) => return None,
				}
			}
			let mut p = Vec::with_capacity(dists.len());
			for d in dists {
				match d {
					Dist::Sampled(row) => p.push(row.as_slice()),
					Dist::Greedy(_) => return None,
				}
			}
			let v = self.sampler.verify_accept_trusted(drafts, &q, &p).ok()?;
			Some(Verdict {
				accepted: v.accepted,
				successor: Dist::Sampled(v.successor_dist.into()),
			})
		}
	}

	/// Invariant-Err recovery: a float corner costs one round, never the
	/// call. Reconcile to `…, x`; MTP truncates to entry then re-commits
	/// the `(frontier_entry, x)` pair so speculation resumes next round;
	/// stage-4 draw from the verify row-0 distribution (undrawn so far).
	#[allow(
		clippy::too_many_arguments,
		reason = "speculative recovery keeps the complete frontier snapshot explicit"
	)]
	fn invariant_err_recovery(
		&mut self,
		x: u32,
		entry_pos: usize,
		entry_pairs: usize,
		frontier_entry: &O::Hidden,
		post_x: Option<&O::Snapshot>,
		hiddens: &[O::Hidden],
		dists: &[Dist],
	) -> Result<RoundEnd> {
		if let Some(snap) = post_x {
			self.restore_or_exit(snap)?;
		}
		self.truncate_or_exit(entry_pos + 1)?;
		self.committed_len += 1;
		let committed = self.ops.mtp_truncate(entry_pairs).is_ok()
			&& self
				.ops
				.mtp_commit(&[x], std::slice::from_ref(frontier_entry))
				.is_ok();
		if committed {
			self.frontier = hiddens.first().cloned();
		} else {
			self.discard_mtp();
		}
		let Some(dist) = dists.first() else {
			return Err(internal("missing verify row-0 dist"));
		};
		let next = self.draw(dist)?;
		Ok(RoundEnd::Continue { next })
	}

	/// Emit `d1…d_a`; event order per accepted draft matches current-x
	/// order: EOS/cancel → forced close → output-slot exhaustion.
	/// Returns the retained draft count `r` and the observed event.
	fn emit_drafts(&mut self, drafts: &[u32], a: usize) -> (usize, SpecEvent) {
		for i in 1..=a {
			match self.emitter.emit(drafts[i - 1]) {
				Err(e) => return (i - 1, SpecEvent::EmitErr(e)),
				Ok(Emit::Eos) => {
					self.used += 1;
					return (i - 1, SpecEvent::Eos);
				}
				Ok(Emit::Cancelled) => {
					self.used += 1;
					return (i - 1, SpecEvent::Cancel);
				}
				Ok(Emit::Continue { raw_text }) => {
					self.used += 1;
					let close = self.budget.as_mut().and_then(|b| b.observe(&raw_text));
					if let Some(marker) = close {
						return (i, SpecEvent::Close(marker));
					}
					if self.used >= self.max_tokens {
						return (i, SpecEvent::Slot);
					}
				}
			}
		}
		(a, SpecEvent::None)
	}

	/// Stage 1 for the speculative round: single reconciliation of the
	/// target to the retained prefix `P = [x, d1…d_r]`, returning the new
	/// frontier hidden and the MTP commit inputs.
	fn reconcile(&mut self, ctx: &SpecCtx<O>, r: usize) -> Result<Reconciled<O::Hidden>> {
		let row = |i: usize| -> Result<O::Hidden> {
			ctx.hiddens
				.get(i)
				.cloned()
				.ok_or_else(|| internal("missing verify hidden row"))
		};
		let prevs_from_verify = |upto: usize| -> Result<Vec<O::Hidden>> {
			let mut prevs = Vec::with_capacity(1 + upto);
			prevs.push(ctx.frontier_entry.clone());
			for i in 0..upto {
				prevs.push(row(i)?);
			}
			Ok(prevs)
		};
		if !self.ops.is_hybrid() {
			// Pure attention: truncate; frontier = detached verify row r.
			self.truncate_or_exit(ctx.entry_pos + 1 + r)?;
			return Ok(Reconciled::Committed {
				frontier_h: row(r)?,
				prevs: prevs_from_verify(r)?,
			});
		}
		if r == ctx.k {
			// Hybrid fast path: no restore, no replay — the attention
			// offset is already entry+1+k and the recurrent state is
			// post-d_k; commit inputs are the detached verify rows.
			return Ok(Reconciled::Committed {
				frontier_h: row(ctx.k)?,
				prevs: prevs_from_verify(ctx.k)?,
			});
		}
		let Some(post_x) = ctx.post_x.as_ref() else {
			return Err(internal("hybrid verify without a post-x snapshot"));
		};
		self.restore_or_exit(post_x)?;
		self.truncate_or_exit(ctx.entry_pos + 1)?;
		if r == 0 {
			// No empty replay; frontier = detached H_x.
			return Ok(Reconciled::Committed {
				frontier_h: row(0)?,
				prevs: prevs_from_verify(0)?,
			});
		}
		// Hybrid 1 <= r < k: replay [d1…d_r] once; frontier and the
		// d-pair commit inputs come from the replay pass.
		match self.ops.replay_feed(&ctx.drafts[..r], false, &self.sampler) {
			Ok((replay_hiddens, _)) => {
				let replay_row = |i: usize| -> Result<O::Hidden> {
					replay_hiddens
						.get(i)
						.cloned()
						.ok_or_else(|| internal("missing replay hidden row"))
				};
				let mut prevs = Vec::with_capacity(1 + r);
				prevs.push(ctx.frontier_entry.clone());
				prevs.push(row(0)?);
				for i in 0..r - 1 {
					prevs.push(replay_row(i)?);
				}
				Ok(Reconciled::Committed {
					frontier_h: replay_row(r - 1)?,
					prevs,
				})
			}
			// Hybrid reconciliation replay fails → invalid-cache exit.
			Err(OpError::Invalid(e) | OpError::MtpForward(e)) => Err(e),
			// Post-reconciliation detach failure: target valid at …,P;
			// MTP must be discarded; the stage-4 draw still proceeds.
			Err(OpError::Host { .. }) => Ok(Reconciled::DetachFailed),
		}
	}

	/// Stages 1-3 for continuing/exiting dispositions: reconcile, commit
	/// both ledgers, then MTP truncate + batched pair-commit replay.
	fn reconcile_and_commit(&mut self, x: u32, ctx: &SpecCtx<O>, r: usize) -> Result<()> {
		match self.reconcile(ctx, r)? {
			Reconciled::Committed { frontier_h, prevs } => {
				self.committed_len += 1 + r;
				let mut ids = Vec::with_capacity(1 + r);
				ids.push(x);
				ids.extend_from_slice(&ctx.drafts[..r]);
				let committed = self.ops.mtp_truncate(ctx.entry_pairs).is_ok()
					&& self.ops.mtp_commit(&ids, &prevs).is_ok();
				if committed {
					self.frontier = Some(frontier_h);
				} else {
					// MTP commit replay fails: discard, continue
					// target-only; the pending stage-4 draw proceeds.
					self.discard_mtp();
				}
			}
			Reconciled::DetachFailed => {
				self.committed_len += 1 + r;
				self.discard_mtp();
			}
		}
		Ok(())
	}

	/// Reconcile the target to the retained prefix and discard the MTP —
	/// the sequential-emission-failure disposition.
	fn reconcile_and_discard(&mut self, ctx: &SpecCtx<O>, r: usize) -> Result<()> {
		match self.reconcile(ctx, r)? {
			Reconciled::Committed { .. } | Reconciled::DetachFailed => {
				self.committed_len += 1 + r;
				self.discard_mtp();
			}
		}
		Ok(())
	}
}

// ---------------------------------------------------------------------
// Real implementation over MLX
// ---------------------------------------------------------------------

/// [`RoundOps`] over the real model and caches. Hidden rows are views of
/// a per-feed detached (contiguous + eval'd) block, so they pin only the
/// small `[1, r+1, H]` block rather than the feed's full activations.
pub struct SessionOps<'a> {
	model: &'a Model,
	caches: &'a mut [LayerCache],
	mtp: Option<MtpCaches>,
}

impl<'a> SessionOps<'a> {
	pub fn new(model: &'a Model, caches: &'a mut [LayerCache], mtp: Option<MtpCaches>) -> Self {
		SessionOps { model, caches, mtp }
	}

	pub fn into_mtp(self) -> Option<MtpCaches> {
		self.mtp
	}
}

/// One [`Dist`] per row of `logits` (`[1, L, V]`). Greedy and sampled modes
/// each evaluate one batched graph and perform one host read. Sampled rows then
/// normalize and validate in place over one shared host buffer; a failed row
/// returns the completed prefix for salvage.
///
/// Greedy mode runs ONE `argmax_axis` over all rows plus one host read
/// (instead of a per-row argmax + `item` pair — 2(k+1) device syncs per
/// verify feed). The reduction semantics match `Sampler::sample`'s greedy
/// path: the same `argmax_axis(_, -1, _)` reduction over the same
/// last-axis rows, so per-row results are identical.
fn materialize_dists(
	logits: &Array,
	sampler: &Sampler,
) -> std::result::Result<Vec<Dist>, (Error, Vec<Dist>)> {
	let shape = logits.shape();
	let (rows, vocab) = (shape[1], shape[2]);
	let mut dists = Vec::with_capacity(rows as usize);
	if let Err(e) = logits.eval() {
		return Err((e, dists));
	}
	if sampler.is_greedy() {
		// Batched: [1, L, V] -argmax-> [1, L] -> one host read of L u32s.
		// Salvage on failure is empty (the read is all-or-nothing), which
		// the host-failure rows already tolerate (exact-prefix exit without a
		// salvaged dist).
		let toks = ops::argmax_axis(logits, -1, false)
			.and_then(|am| am.to_vec_u32())
			.map_err(|e| (e, Vec::new()))?;
		if toks.len() != rows as usize {
			return Err((internal("greedy argmax row-count mismatch"), Vec::new()));
		}
		dists.extend(toks.into_iter().map(Dist::Greedy));
		return Ok(dists);
	}
	let rows = usize::try_from(rows).map_err(|_| {
		(
			internal("negative sampled distribution row count"),
			Vec::new(),
		)
	})?;
	let vocab = usize::try_from(vocab).map_err(|_| {
		(
			internal("negative sampled distribution vocabulary"),
			Vec::new(),
		)
	})?;
	let expected = rows
		.checked_mul(vocab)
		.ok_or_else(|| (internal("sampled distribution size overflow"), Vec::new()))?;
	let mut flat = sampler
		.batched_softmax(logits)
		.map_err(|error| (error, Vec::new()))?;
	if flat.len() != expected {
		return Err((
			internal("sampled softmax element-count mismatch"),
			Vec::new(),
		));
	}
	for row in 0..rows {
		let start = row * vocab;
		let end = start + vocab;
		if let Err(error) = sampler.finish_batched_probability_row(&mut flat[start..end]) {
			return Err((error, shared_sampled_dists(flat, row, vocab)));
		}
	}
	Ok(shared_sampled_dists(flat, rows, vocab))
}

fn shared_sampled_dists(flat: Vec<f32>, rows: usize, vocab: usize) -> Vec<Dist> {
	let values: Arc<[f32]> = flat.into();
	(0..rows)
		.map(|row| {
			let start = row * vocab;
			Dist::Sampled(Probabilities::shared(
				Arc::clone(&values),
				start,
				start + vocab,
			))
		})
		.collect()
}

/// The dist for one logits row, using the same argmax call shape as
/// `Sampler::sample`'s greedy path (minimizes tie-flips vs spec-off).
fn row_dist(logits: &Array, row: i32, vocab: i32, sampler: &Sampler) -> Result<Dist> {
	let sliced = ops::slice(logits, &[0, row, 0], &[1, row + 1, vocab])?;
	let sliced = ops::reshape(&sliced, &[vocab])?;
	if sampler.is_greedy() {
		Ok(Dist::Greedy(
			ops::argmax_axis(&sliced, -1, false)?.item_u32()?,
		))
	} else {
		Ok(Dist::Sampled(sampler.probs(&sliced)?.into()))
	}
}

/// Detach `hidden` (`[1, L, H]`) as ONE contiguous block with a single
/// eval, handing out per-row views of the detached block (glossary
/// rule 1: views then pin only the small block).
fn detach_rows(hidden: &Array) -> Result<Vec<Array>> {
	let block = ops::contiguous(hidden)?;
	block.eval()?;
	let shape = block.shape();
	let (rows, width) = (shape[1], shape[2]);
	(0..rows)
		.map(|i| ops::slice(&block, &[0, i, 0], &[1, i + 1, width]))
		.collect()
}

impl RoundOps for SessionOps<'_> {
	type Hidden = Array;
	type Snapshot = Vec<LayerRollback>;

	fn is_hybrid(&self) -> bool {
		self.caches
			.iter()
			.any(|c| !matches!(c, LayerCache::Attention(_)))
	}

	fn target_pos(&self) -> usize {
		self.caches
			.iter()
			.find_map(|c| match c {
				LayerCache::Attention(kv) => Some(kv.offset() as usize),
				LayerCache::Dhara(d) => Some(d.attn.offset() as usize),
				LayerCache::GatedDelta(_) => None,
			})
			.unwrap_or(0)
	}

	fn window_would_trim(&self, added: usize) -> bool {
		self.caches.iter().any(|c| match c {
			LayerCache::Attention(kv) => kv.would_trim(added as i32),
			LayerCache::Dhara(d) => d.attn.would_trim(added as i32),
			LayerCache::GatedDelta(_) => false,
		})
	}

	fn verify_feed(&mut self, ids: &[u32], sampler: &Sampler) -> OpResult<(Vec<Array>, Vec<Dist>)> {
		let arr =
			Array::from_slice(ids, &[1, ids.len() as i32]).map_err(|error| OpError::Host {
				error,
				salvage: Vec::new(),
			})?;
		let (hidden, logits) = if self.mtp.is_some() {
			let out = self
				.model
				.forward_hidden(&arr, self.caches)
				.map_err(OpError::Invalid)?;
			(Some(out.hidden_pre_norm), out.logits)
		} else {
			(
				None,
				self.model
					.forward(&arr, self.caches)
					.map_err(OpError::Invalid)?,
			)
		};
		// Host phase: dist materialization first, hidden detach second —
		// a detach failure salvages every dist.
		let dists = materialize_dists(&logits, sampler)
			.map_err(|(error, salvage)| OpError::Host { error, salvage })?;
		let hiddens = match hidden {
			Some(h) => detach_rows(&h).map_err(|error| OpError::Host {
				error,
				salvage: dists.clone(),
			})?,
			None => Vec::new(),
		};
		Ok((hiddens, dists))
	}

	fn replay_feed(
		&mut self,
		ids: &[u32],
		want_last_dist: bool,
		sampler: &Sampler,
	) -> OpResult<(Vec<Array>, Option<Dist>)> {
		let arr =
			Array::from_slice(ids, &[1, ids.len() as i32]).map_err(|error| OpError::Host {
				error,
				salvage: Vec::new(),
			})?;
		let (hidden, logits) = if self.mtp.is_some() {
			let out = self
				.model
				.forward_hidden(&arr, self.caches)
				.map_err(OpError::Invalid)?;
			(Some(out.hidden_pre_norm), out.logits)
		} else {
			(
				None,
				self.model
					.forward(&arr, self.caches)
					.map_err(OpError::Invalid)?,
			)
		};
		// Replay never evaluates the logits tensor as a whole; at most
		// the LAST row is materialized for the post-close successor.
		let last_dist = if want_last_dist {
			let vocab = logits.dim(2);
			let dist =
				row_dist(&logits, ids.len() as i32 - 1, vocab, sampler).map_err(|error| {
					OpError::Host {
						error,
						salvage: Vec::new(),
					}
				})?;
			Some(dist)
		} else {
			None
		};
		let hiddens = match hidden {
			Some(h) => detach_rows(&h).map_err(|error| OpError::Host {
				error,
				salvage: last_dist.clone().into_iter().collect(),
			})?,
			None => Vec::new(),
		};
		Ok((hiddens, last_dist))
	}

	fn capture(&self) -> Vec<LayerRollback> {
		self.caches.iter().map(LayerCache::rollback_state).collect()
	}

	fn restore(&mut self, snap: &Vec<LayerRollback>) -> OpResult<()> {
		for (cache, state) in self.caches.iter_mut().zip(snap.iter()) {
			cache.rollback(state).map_err(OpError::Invalid)?;
		}
		Ok(())
	}

	fn truncate_target(&mut self, abs_offset: usize) -> OpResult<()> {
		for cache in self.caches.iter_mut() {
			match cache {
				LayerCache::Attention(kv) => kv
					.truncate_to(abs_offset as i32)
					.map_err(OpError::Invalid)?,
				LayerCache::Dhara(d) => d
					.attn
					.truncate_to(abs_offset as i32)
					.map_err(OpError::Invalid)?,
				LayerCache::GatedDelta(_) => {}
			}
		}
		Ok(())
	}

	fn mtp_alive(&self) -> bool {
		self.mtp.is_some()
	}

	fn mtp_pairs(&self) -> usize {
		self.mtp.as_ref().map_or(0, MtpCaches::pairs_fed)
	}

	fn mtp_step(&mut self, id: u32, prev: &Array, sampler: &Sampler) -> OpResult<(Array, Dist)> {
		let Some(mtp) = self.mtp.as_mut() else {
			return Err(OpError::MtpForward(internal("mtp_step without live MTP")));
		};
		let ids = Array::from_slice(&[id], &[1, 1]).map_err(|error| OpError::Host {
			error,
			salvage: Vec::new(),
		})?;
		let out = self
			.model
			.forward_mtp(&ids, prev, mtp)
			.map_err(OpError::MtpForward)?;
		let host = |error: Error| OpError::Host {
			error,
			salvage: Vec::new(),
		};
		let vocab = out.logits.dim(2);
		let dist = row_dist(&out.logits, 0, vocab, sampler).map_err(host)?;
		let recycle = ops::contiguous(&out.recycle_hidden).map_err(host)?;
		recycle.eval().map_err(host)?;
		Ok((recycle, dist))
	}

	fn mtp_commit(&mut self, ids: &[u32], prevs: &[Array]) -> OpResult<()> {
		let Some(mtp) = self.mtp.as_mut() else {
			return Err(OpError::MtpForward(internal("mtp_commit without live MTP")));
		};
		let prev = if prevs.len() == 1 {
			prevs[0].clone()
		} else {
			let refs: Vec<&Array> = prevs.iter().collect();
			ops::concatenate(&refs, 1).map_err(|error| OpError::Host {
				error,
				salvage: Vec::new(),
			})?
		};
		let ids_arr =
			Array::from_slice(ids, &[1, ids.len() as i32]).map_err(|error| OpError::Host {
				error,
				salvage: Vec::new(),
			})?;
		// The MtpStepOutput is dropped whole: priming/commit logits are
		// NEVER evaluated.
		let _ = self
			.model
			.forward_mtp(&ids_arr, &prev, mtp)
			.map_err(OpError::MtpForward)?;
		Ok(())
	}

	fn mtp_truncate(&mut self, pairs_fed: usize) -> OpResult<()> {
		let Some(mtp) = self.mtp.as_mut() else {
			return Err(OpError::MtpForward(internal(
				"mtp_truncate without live MTP",
			)));
		};
		mtp.truncate_to(pairs_fed).map_err(|error| OpError::Host {
			error,
			salvage: Vec::new(),
		})
	}

	fn mtp_discard(&mut self) {
		self.mtp = None;
	}
}

// ---------------------------------------------------------------------
// Fake model (test seam)
// ---------------------------------------------------------------------

/// The fake `RoundOps` implementation: deterministic f64 host arithmetic,
/// batch-size-invariant BY CONSTRUCTION — every target output is a pure
/// function of the fed prefix CONTENT (never of feed chunking), the MTP
/// recycle hidden a pure function of the pair prefix. Scriptable
/// per-position outputs and per-method fail-on-Nth-call fault switches
/// drive every disposition and failure row; the fake tracks its own
/// offsets/pairs and an ordered event log for the exact-offset and
/// stage-ordering assertions required by the fault matrix.
#[cfg(test)]
pub mod fake {
	use super::{Dist, OpError, OpResult, RoundOps, Sampler, TraceEvent, internal};

	pub type Hidden = Vec<f64>;

	/// Deterministic "hidden state" of a target prefix: `[len, digest]`.
	pub fn hid(prefix: &[u32]) -> Hidden {
		let mut digest = 0.0f64;
		for &t in prefix {
			digest = (digest.mul_add(31.0, f64::from(t) + 1.0)) % 1_000_003.0;
		}
		vec![prefix.len() as f64, digest]
	}

	/// Deterministic MTP recycle hidden (3 entries, so it can never be
	/// mistaken for a target [`hid`] in pair assertions).
	pub fn mtp_hid(pairs: &[(u32, Hidden)]) -> Hidden {
		let mut digest = 7.0f64;
		for (id, h) in pairs {
			digest = (digest * 31.0 + f64::from(*id) + h.iter().sum::<f64>()) % 999_983.0;
		}
		vec![pairs.len() as f64, digest, 0.5]
	}

	/// Default deterministic token function (a pure hash of the prefix).
	pub fn default_tok(seq: &[u32], vocab: usize, salt: u64) -> u32 {
		let mut h = salt;
		for &t in seq {
			h = h
				.wrapping_mul(6_364_136_223_846_793_005)
				.wrapping_add(u64::from(t) + 1_442_695_040_888_963_407);
		}
		((h >> 33) as usize % vocab) as u32
	}

	pub fn one_hot(vocab: usize, tok: u32) -> Vec<f32> {
		let mut v = vec![0.0f32; vocab];
		v[tok as usize] = 1.0;
		v
	}

	/// Ordered event log entries.
	#[derive(Debug, Clone, PartialEq, Eq)]
	pub enum Ev {
		VerifyFeed(Vec<u32>),
		ReplayFeed(Vec<u32>),
		MtpStep(u32),
		MtpCommit(Vec<u32>),
		MtpTruncate(usize),
		Truncate(usize),
		Restore,
		Capture,
		MtpDiscard,
		Successor,
	}

	#[derive(Debug, Clone, Copy, PartialEq, Eq)]
	pub enum FaultKind {
		/// Structural failure mid-mutation (invalid-cache exit class).
		Invalid,
		/// MTP forward failure (blanket-discard class).
		MtpForward,
		/// Host-side failure after complete mutation, dists salvaged.
		HostSalvaged,
		/// Host-side failure after complete mutation, nothing salvaged.
		HostBare,
	}

	#[derive(Debug, Clone, Copy)]
	pub struct Fault {
		pub at_call: usize,
		pub kind: FaultKind,
	}

	#[derive(Default)]
	pub struct FaultSlot {
		plan: Option<Fault>,
		calls: usize,
	}

	impl FaultSlot {
		pub fn arm(&mut self, at_call: usize, kind: FaultKind) {
			self.plan = Some(Fault { at_call, kind });
		}

		fn fire(&mut self) -> Option<FaultKind> {
			let n = self.calls;
			self.calls += 1;
			self.plan.filter(|f| f.at_call == n).map(|f| f.kind)
		}
	}

	/// Per-method fail-on-Nth-call switches.
	#[derive(Default)]
	pub struct Faults {
		pub verify_feed: FaultSlot,
		pub replay_feed: FaultSlot,
		pub mtp_step: FaultSlot,
		pub mtp_commit: FaultSlot,
		pub restore: FaultSlot,
		pub truncate: FaultSlot,
		pub mtp_truncate: FaultSlot,
	}

	fn fault_err() -> crate::engine::error::Error {
		internal("fake fault")
	}

	pub type TokFn = Box<dyn Fn(&[u32]) -> u32>;
	pub type DistFn = Box<dyn Fn(&[u32]) -> Vec<f32>>;

	pub struct FakeOps {
		pub vocab: usize,
		pub hybrid: bool,
		pub prompt_len: usize,
		/// Absolute fed target sequence (prompt included).
		pub t_fed: Vec<u32>,
		/// MTP state: committed/provisional `(id, prev_hidden)` pairs;
		/// `None` = dead/discarded.
		pub pairs: Option<Vec<(u32, Hidden)>>,
		/// Window-preflight verdict.
		pub window_trim: bool,
		/// Target successor of the prefix ending at the given DECODE
		/// suffix (prompt-relative, includes the row's own token).
		pub target_tok: TokFn,
		pub target_dist: Option<DistFn>,
		/// MTP draft prediction, keyed by the decode-relative id sequence
		/// the MTP has consumed (includes the current step's id).
		pub draft_tok: TokFn,
		pub draft_dist: Option<DistFn>,
		pub faults: Faults,
		pub log: Vec<Ev>,
		pub restores: usize,
		/// Anti-vacuousness ledger: the exact `(id, prev)` argument pair of
		/// every `mtp_step` call, in order. `mtp_step`'s returned recycle
		/// hidden is a pure function of the pair prefix, so a driver that
		/// wired the draft recursion to a stale hidden (e.g. always
		/// `frontier_entry`) would leave a detectable wrong `prev` here even
		/// though the drafted tokens might not change.
		pub step_prevs: Vec<(u32, Hidden)>,
	}

	impl FakeOps {
		/// Entry state consistent with the round-entry invariant: the
		/// target holds `prompt`; the MTP (when alive) holds the shifted
		/// prompt pairs (`pairs_fed = prompt.len() - 1`); the returned
		/// frontier is `hid(prompt)`.
		pub fn new(
			prompt: &[u32],
			vocab: usize,
			hybrid: bool,
			with_mtp: bool,
		) -> (Self, Option<Hidden>) {
			let mut pairs = Vec::new();
			for i in 1..prompt.len() {
				pairs.push((prompt[i], hid(&prompt[..i])));
			}
			let frontier = with_mtp.then(|| hid(prompt));
			let vocab_t = vocab;
			let vocab_d = vocab;
			let ops = FakeOps {
				vocab,
				hybrid,
				prompt_len: prompt.len(),
				t_fed: prompt.to_vec(),
				pairs: with_mtp.then_some(pairs),
				window_trim: false,
				target_tok: Box::new(move |seq| default_tok(seq, vocab_t, 17)),
				target_dist: None,
				draft_tok: Box::new(move |seq| {
					let t = default_tok(seq, vocab_d, 17);
					// A controlled perturbation so default runs mix
					// acceptance depths.
					if seq.len() % 3 == 2 {
						(t + 1) % vocab_d as u32
					} else {
						t
					}
				}),
				draft_dist: None,
				faults: Faults::default(),
				log: Vec::new(),
				restores: 0,
				step_prevs: Vec::new(),
			};
			(ops, frontier)
		}

		pub fn decode_suffix(&self) -> &[u32] {
			&self.t_fed[self.prompt_len..]
		}

		pub fn pairs_fed(&self) -> usize {
			self.pairs.as_ref().map_or(0, Vec::len)
		}

		pub fn count(&self, want: &Ev) -> usize {
			self.log.iter().filter(|e| *e == want).count()
		}

		fn dist_for(&self, greedy: bool, suffix: &[u32], target: bool) -> Dist {
			let tok = if target {
				(self.target_tok)(suffix)
			} else {
				(self.draft_tok)(suffix)
			};
			if greedy {
				return Dist::Greedy(tok);
			}
			let dist_fn = if target {
				&self.target_dist
			} else {
				&self.draft_dist
			};
			match dist_fn {
				Some(f) => Dist::Sampled(f(suffix).into()),
				None => Dist::Sampled(one_hot(self.vocab, tok).into()),
			}
		}

		fn feed(&mut self, ids: &[u32], greedy: bool) -> (Vec<Hidden>, Vec<Dist>) {
			let mut hiddens = Vec::with_capacity(ids.len());
			let mut dists = Vec::with_capacity(ids.len());
			for &id in ids {
				self.t_fed.push(id);
				if self.pairs.is_some() {
					hiddens.push(hid(&self.t_fed));
				}
				let suffix = self.t_fed[self.prompt_len..].to_vec();
				dists.push(self.dist_for(greedy, &suffix, true));
			}
			(hiddens, dists)
		}
	}

	impl RoundOps for FakeOps {
		type Hidden = Hidden;
		type Snapshot = Vec<u32>;

		fn is_hybrid(&self) -> bool {
			self.hybrid
		}

		fn target_pos(&self) -> usize {
			self.t_fed.len()
		}

		fn window_would_trim(&self, _added: usize) -> bool {
			self.window_trim
		}

		fn verify_feed(
			&mut self,
			ids: &[u32],
			sampler: &Sampler,
		) -> OpResult<(Vec<Hidden>, Vec<Dist>)> {
			self.log.push(Ev::VerifyFeed(ids.to_vec()));
			let fault = self.faults.verify_feed.fire();
			if matches!(fault, Some(FaultKind::Invalid | FaultKind::MtpForward)) {
				// Structural failure: mid-mutation state is undefined and
				// unusable (invalid-cache exit) — leave it half-fed on purpose.
				self.t_fed.push(ids[0]);
				return Err(OpError::Invalid(fault_err()));
			}
			let (hiddens, dists) = self.feed(ids, sampler.is_greedy());
			match fault {
				Some(FaultKind::HostSalvaged) => Err(OpError::Host {
					error: fault_err(),
					salvage: dists,
				}),
				Some(FaultKind::HostBare) => Err(OpError::Host {
					error: fault_err(),
					salvage: Vec::new(),
				}),
				_ => Ok((hiddens, dists)),
			}
		}

		fn replay_feed(
			&mut self,
			ids: &[u32],
			want_last_dist: bool,
			sampler: &Sampler,
		) -> OpResult<(Vec<Hidden>, Option<Dist>)> {
			self.log.push(Ev::ReplayFeed(ids.to_vec()));
			let fault = self.faults.replay_feed.fire();
			if matches!(fault, Some(FaultKind::Invalid | FaultKind::MtpForward)) {
				self.t_fed.push(ids[0]);
				return Err(OpError::Invalid(fault_err()));
			}
			let (hiddens, mut dists) = self.feed(ids, sampler.is_greedy());
			let last = want_last_dist.then(|| dists.pop()).flatten();
			match fault {
				Some(FaultKind::HostSalvaged) => Err(OpError::Host {
					error: fault_err(),
					salvage: last.into_iter().collect(),
				}),
				Some(FaultKind::HostBare) => Err(OpError::Host {
					error: fault_err(),
					salvage: Vec::new(),
				}),
				_ => Ok((hiddens, last)),
			}
		}

		fn capture(&self) -> Vec<u32> {
			self.t_fed.clone()
		}

		fn restore(&mut self, snap: &Vec<u32>) -> OpResult<()> {
			self.log.push(Ev::Restore);
			self.restores += 1;
			if self.faults.restore.fire().is_some() {
				return Err(OpError::Invalid(fault_err()));
			}
			self.t_fed = snap.clone();
			Ok(())
		}

		fn truncate_target(&mut self, abs_offset: usize) -> OpResult<()> {
			self.log.push(Ev::Truncate(abs_offset));
			if self.faults.truncate.fire().is_some() || abs_offset > self.t_fed.len() {
				return Err(OpError::Invalid(fault_err()));
			}
			self.t_fed.truncate(abs_offset);
			Ok(())
		}

		fn mtp_alive(&self) -> bool {
			self.pairs.is_some()
		}

		fn mtp_pairs(&self) -> usize {
			self.pairs_fed()
		}

		fn mtp_step(
			&mut self,
			id: u32,
			prev: &Hidden,
			sampler: &Sampler,
		) -> OpResult<(Hidden, Dist)> {
			self.log.push(Ev::MtpStep(id));
			// Record the consumed prev BEFORE fault handling: even a failed
			// step pins what the driver actually passed.
			self.step_prevs.push((id, prev.clone()));
			let fault = self.faults.mtp_step.fire();
			if matches!(fault, Some(FaultKind::Invalid | FaultKind::MtpForward)) {
				return Err(OpError::MtpForward(fault_err()));
			}
			let primed = self.prompt_len.saturating_sub(1);
			let greedy = sampler.is_greedy();
			let Some(pairs) = self.pairs.as_mut() else {
				return Err(OpError::MtpForward(fault_err()));
			};
			pairs.push((id, prev.clone()));
			let recycle = mtp_hid(pairs);
			let seq: Vec<u32> = pairs[primed..].iter().map(|p| p.0).collect();
			let dist = self.dist_for(greedy, &seq, false);
			match fault {
				Some(FaultKind::HostSalvaged | FaultKind::HostBare) => Err(OpError::Host {
					error: fault_err(),
					salvage: Vec::new(),
				}),
				_ => Ok((recycle, dist)),
			}
		}

		fn mtp_commit(&mut self, ids: &[u32], prevs: &[Hidden]) -> OpResult<()> {
			self.log.push(Ev::MtpCommit(ids.to_vec()));
			let fault = self.faults.mtp_commit.fire();
			if let Some(kind) = fault {
				return match kind {
					FaultKind::HostSalvaged | FaultKind::HostBare => Err(OpError::Host {
						error: fault_err(),
						salvage: Vec::new(),
					}),
					_ => Err(OpError::MtpForward(fault_err())),
				};
			}
			assert_eq!(ids.len(), prevs.len(), "mtp_commit shape");
			let Some(pairs) = self.pairs.as_mut() else {
				return Err(OpError::MtpForward(fault_err()));
			};
			for (id, prev) in ids.iter().zip(prevs.iter()) {
				pairs.push((*id, prev.clone()));
			}
			Ok(())
		}

		fn mtp_truncate(&mut self, pairs_fed: usize) -> OpResult<()> {
			self.log.push(Ev::MtpTruncate(pairs_fed));
			if self.faults.mtp_truncate.fire().is_some() {
				return Err(OpError::Host {
					error: fault_err(),
					salvage: Vec::new(),
				});
			}
			let Some(pairs) = self.pairs.as_mut() else {
				return Err(OpError::MtpForward(fault_err()));
			};
			assert!(pairs_fed <= pairs.len(), "mtp_truncate overshoot");
			pairs.truncate(pairs_fed);
			Ok(())
		}

		fn mtp_discard(&mut self) {
			if self.pairs.take().is_some() {
				self.log.push(Ev::MtpDiscard);
			}
		}

		fn trace(&mut self, event: TraceEvent) {
			match event {
				TraceEvent::SuccessorSelect => self.log.push(Ev::Successor),
			}
		}
	}
}

// ---------------------------------------------------------------------
// Fake-model suite
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
	use std::{cell::RefCell, rc::Rc, sync::OnceLock};

	use super::{fake::*, *};
	use crate::engine::{
		sampling::SamplingConfig, streaming::StreamClassifier, tokenizer::Tokenizer,
		tools::ToolCallFormat,
	};

	const V: usize = 16;
	/// Fixture ids: 1 = `<|im_start|>`, 5 = `user`, 8 = `hello`.
	const PROMPT: &[u32] = &[1, 5, 8];
	/// Fixture EOS (`<|im_end|>`).
	const EOS: u32 = 2;
	/// Fixture `</think>` (the forced-close marker's single token).
	const CLOSE: u32 = 4;

	/// The fake suite runs the driver over the committed tiny-model
	/// fixture tokenizer — real decodes, real EOS registration, real
	/// close-marker encoding.
	fn tokenizer() -> &'static Tokenizer {
		static TOK: OnceLock<Tokenizer> = OnceLock::new();
		TOK.get_or_init(|| {
			let dir =
				std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-model");
			Tokenizer::load(&dir).unwrap()
		})
	}

	fn greedy() -> Sampler {
		Sampler::new(SamplingConfig::default())
	}

	fn sampled(seed: u64) -> Sampler {
		Sampler::new(SamplingConfig {
			temperature: 1.0,
			top_p: 1.0,
			top_k: None,
			seed: Some(seed),
		})
	}

	#[test]
	fn sampled_target_rows_share_one_batched_host_buffer() {
		let logits = Array::from_slice(&[1.0_f32, 2.0, 3.0, 3.0, 2.0, 1.0], &[1, 2, 3]).unwrap();
		let sampler = sampled(7);
		let dists = materialize_dists(&logits, &sampler).unwrap();
		let [Dist::Sampled(first), Dist::Sampled(second)] = dists.as_slice() else {
			panic!("expected two sampled rows");
		};
		assert!(Arc::ptr_eq(&first.values, &second.values));
		assert_eq!(first.as_slice().len(), 3);
		assert_eq!(second.as_slice().len(), 3);
		for row in [first.as_slice(), second.as_slice()] {
			let sum = row.iter().copied().map(f64::from).sum::<f64>();
			assert!((sum - 1.0).abs() < 1e-6);
		}
	}

	#[test]
	fn sampled_batched_row_failure_salvages_completed_prefix() {
		let logits = Array::from_slice(&[1.0_f32, 2.0, 3.0, 3.0, 2.0, 1.0], &[1, 2, 3]).unwrap();
		let sampler = sampled(7);
		sampler.inject_failure_at(1);
		let (_, salvage) = materialize_dists(&logits, &sampler).unwrap_err();
		assert_eq!(salvage.len(), 1);
		assert!(matches!(&salvage[0], Dist::Sampled(row) if row.as_slice().len() == 3));
	}

	type Cb = Box<dyn FnMut(GeneratedToken) -> bool>;

	fn recording_cb(record: Rc<RefCell<Vec<u32>>>, cancel_at: Option<usize>) -> Cb {
		let mut i = 0usize;
		Box::new(move |tok: GeneratedToken| {
			record.borrow_mut().push(tok.id);
			let keep_going = Some(i) != cancel_at;
			i += 1;
			keep_going
		})
	}

	struct Cfg {
		max_tokens: usize,
		spec_k: Option<usize>,
		budget: Option<usize>,
		cancel_at: Option<usize>,
		emitter_fail_at: Option<usize>,
		close_override: Option<Vec<u32>>,
	}

	impl Default for Cfg {
		fn default() -> Self {
			Cfg {
				max_tokens: 32,
				spec_k: Some(2),
				budget: None,
				cancel_at: None,
				emitter_fail_at: None,
				close_override: None,
			}
		}
	}

	fn driver(
		ops: FakeOps,
		frontier: Option<Hidden>,
		sampler: Sampler,
		cfg: &Cfg,
		callbacks: Rc<RefCell<Vec<u32>>>,
	) -> RoundDriver<'static, FakeOps, Cb> {
		let tok = tokenizer();
		let mut classifier = StreamClassifier::new(ToolCallFormat::Hermes);
		let mut budget = cfg.budget.map(ReasoningBudget::new);
		if budget.is_some() {
			// The reasoning span is open from the first token, as the
			// decode loop seeds for prompts that bake the open marker in.
			classifier.seed_reasoning("</think>");
			if let Some(b) = budget.as_mut() {
				b.seed_open(("<think>", "</think>"));
			}
		}
		let mut emitter = TokenEmitter::new(
			tok,
			tok.eos_token_ids(),
			classifier,
			cfg.max_tokens,
			recording_cb(callbacks, cfg.cancel_at),
		);
		emitter.set_fail_at(cfg.emitter_fail_at);
		emitter.set_close_override(cfg.close_override.clone());
		RoundDriver::new(
			ops,
			emitter,
			budget,
			sampler,
			cfg.max_tokens,
			cfg.spec_k,
			frontier,
		)
	}

	/// Drive rounds until an exit, an error, or the loop guard (the
	/// decode_loop equivalent). Returns the terminal RoundEnd (`None`
	/// when the guard stopped the loop) and the pending unfed successor.
	fn run_all(
		d: &mut RoundDriver<'static, FakeOps, Cb>,
		first: u32,
		max_tokens: usize,
	) -> Result<(Option<RoundEnd>, u32)> {
		let mut next = first;
		while d.used() < max_tokens {
			match d.run_round(next)? {
				RoundEnd::Continue { next: n } => next = n,
				end => return Ok((Some(end), next)),
			}
		}
		Ok((None, next))
	}

	/// Script the target as an explicit chain: `chain[i]` is the target's
	/// successor after the decode suffix `chain[..i]` — i.e. the target
	/// "wants" to generate exactly `chain` (falling back to the default
	/// hash beyond it).
	fn script_target(ops: &mut FakeOps, first: u32, chain: Vec<u32>) {
		let vocab = ops.vocab;
		ops.target_tok = Box::new(move |suffix| {
			let mut want = vec![first];
			want.extend_from_slice(&chain);
			if suffix.len() <= chain.len() && suffix[..] == want[..suffix.len()] {
				chain[suffix.len() - 1]
			} else {
				default_tok(suffix, vocab, 17)
			}
		});
	}

	/// Script the MTP to draft exactly `chain` after `x` (decode-relative
	/// MTP id sequence `[x, d1, ..]`), default hash beyond.
	fn script_draft(ops: &mut FakeOps, first: u32, chain: Vec<u32>) {
		let vocab = ops.vocab;
		ops.draft_tok = Box::new(move |seq| {
			let mut want = vec![first];
			want.extend_from_slice(&chain);
			if seq.len() <= chain.len() && seq[..] == want[..seq.len()] {
				chain[seq.len() - 1]
			} else {
				default_tok(seq, vocab, 23)
			}
		});
	}

	/// Expected MTP pair ledger for a committed decode prefix: the primed
	/// prompt pairs plus one shifted pair per committed decode token.
	fn expect_pairs(prompt: &[u32], decode: &[u32]) -> Vec<(u32, Hidden)> {
		let mut full: Vec<u32> = prompt.to_vec();
		let mut pairs = Vec::new();
		for i in 1..prompt.len() {
			pairs.push((prompt[i], hid(&prompt[..i])));
		}
		for &t in decode {
			pairs.push((t, hid(&full)));
			full.push(t);
		}
		pairs
	}

	// -----------------------------------------------------------------
	// Target-only round traces — greedy and sampled
	// -----------------------------------------------------------------

	fn target_only_disabled_trace(sampler: Sampler) {
		// SpecState::Disabled: mtp dead, spec_k None — N → N+1 offsets,
		// exactly-once callbacks, both ledgers, unfed successor,
		// next-round entry invariant.
		let (mut ops, _) = FakeOps::new(PROMPT, V, false, false);
		script_target(&mut ops, 6, vec![12, 13]);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg {
			spec_k: None,
			..Cfg::default()
		};
		let mut d = driver(ops, None, sampler, &cfg, cbs.clone());
		let end = d.run_round(6).unwrap();
		let RoundEnd::Continue { next } = end else {
			panic!("expected Continue")
		};
		// The successor comes from the feed's row-0 dist and stays unfed.
		assert_eq!(next, 12);
		assert_eq!(d.committed_len(), 1);
		assert_eq!(d.ops().t_fed, [PROMPT, &[6]].concat());
		assert_eq!(d.ops().log, vec![Ev::VerifyFeed(vec![6]), Ev::Successor]);
		assert_eq!(*cbs.borrow(), vec![6]);
		// Next round keeps the invariant: N+1 → N+2.
		let RoundEnd::Continue { next } = d.run_round(next).unwrap() else {
			panic!("expected Continue")
		};
		assert_eq!(next, 13);
		assert_eq!(d.committed_len(), 2);
		assert_eq!(d.ops().t_fed, [PROMPT, &[6, 12]].concat());
		assert_eq!(*cbs.borrow(), vec![6, 12]);
		assert_eq!(d.stats().rounds, 0);
	}

	#[test]
	fn target_only_disabled_trace_greedy() {
		target_only_disabled_trace(greedy());
	}

	#[test]
	fn target_only_disabled_trace_sampled() {
		target_only_disabled_trace(sampled(3));
	}

	fn window_skip_preserves_mtp(sampler: Sampler) {
		// Window preflight fails → target-only round with MTP preserved:
		// the pair commits, the frontier advances, and speculation is
		// available again the moment the predicate clears.
		let (mut ops, frontier) = FakeOps::new(PROMPT, V, false, true);
		script_target(&mut ops, 6, vec![12, 13]);
		ops.window_trim = true;
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg::default();
		let mut d = driver(ops, frontier, sampler, &cfg, cbs.clone());
		let RoundEnd::Continue { next } = d.run_round(6).unwrap() else {
			panic!("expected Continue")
		};
		assert_eq!(next, 12);
		assert_eq!(d.committed_len(), 1);
		assert_eq!(d.ops().t_fed, [PROMPT, &[6]].concat());
		// No draft ran; the MTP pair (frontier_entry, x) committed.
		assert_eq!(d.ops().count(&Ev::MtpStep(6)), 0);
		assert_eq!(d.ops().pairs.as_ref().unwrap(), &expect_pairs(PROMPT, &[6]));
		assert_eq!(d.frontier().unwrap(), &hid(&[PROMPT, &[6]].concat()));
		assert_eq!(d.stats().rounds, 0);
		// Clearing the window predicate re-enables speculation.
		d.ops_mut().window_trim = false;
		let _ = d.run_round(next).unwrap();
		assert!(d.ops().log.iter().any(|e| matches!(e, Ev::MtpStep(_))));
		assert!(d.stats().rounds == 1);
	}

	#[test]
	fn window_skip_preserves_mtp_greedy() {
		window_skip_preserves_mtp(greedy());
	}

	#[test]
	fn window_skip_preserves_mtp_sampled() {
		window_skip_preserves_mtp(sampled(4));
	}

	// -----------------------------------------------------------------
	// Speculative dispositions
	// -----------------------------------------------------------------

	fn full_acceptance_round(sampler: Sampler, hybrid: bool) {
		let (mut ops, frontier) = FakeOps::new(PROMPT, V, hybrid, true);
		script_target(&mut ops, 6, vec![12, 13, 14]);
		script_draft(&mut ops, 6, vec![12, 13]);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg::default();
		let mut d = driver(ops, frontier, sampler, &cfg, cbs.clone());
		let RoundEnd::Continue { next } = d.run_round(6).unwrap() else {
			panic!("expected Continue")
		};
		// Bonus successor, unfed.
		assert_eq!(next, 14);
		assert_eq!(d.committed_len(), 3);
		assert_eq!(d.ops().t_fed, [PROMPT, &[6, 12, 13]].concat());
		assert_eq!(*cbs.borrow(), vec![6, 12, 13]);
		// MTP pairs replayed from target hiddens (never recycle hiddens).
		assert_eq!(
			d.ops().pairs.as_ref().unwrap(),
			&expect_pairs(PROMPT, &[6, 12, 13])
		);
		assert_eq!(
			d.frontier().unwrap(),
			&hid(&[PROMPT, &[6, 12, 13]].concat())
		);
		// Stats: r = a = k = 2 - one-based depth, so depth 2 is index 1.
		assert_eq!(d.stats().rounds, 1);
		assert_eq!(d.stats().drafted, 2);
		assert_eq!(d.stats().accepted_by_depth, vec![0, 1]);
		// Draft recursion chaining (anti-vacuousness): step 0 consumed the
		// entry frontier; step 1 consumed step 0's recycle output - a
		// driver feeding a stale hidden (e.g. always frontier_entry) fails
		// here even when the drafted tokens happen to match.
		let recycle_0 = {
			let mut pairs = expect_pairs(PROMPT, &[]);
			pairs.push((6, hid(PROMPT)));
			mtp_hid(&pairs)
		};
		assert_eq!(
			d.ops().step_prevs,
			vec![(6, hid(PROMPT)), (12, recycle_0)],
			"draft step i must consume step i-1's recycle output"
		);
		// Ordered trace: the successor hook fires exactly once, after the
		// MTP disposition.
		let log = &d.ops().log;
		assert_eq!(d.ops().count(&Ev::Successor), 1);
		let commit_at = log
			.iter()
			.position(|e| matches!(e, Ev::MtpCommit(_)))
			.unwrap();
		let successor_at = log.iter().position(|e| *e == Ev::Successor).unwrap();
		assert!(successor_at > commit_at, "stage 4 must follow stage 3");
		if hybrid {
			// r == k fast path: no restore, no replay.
			assert_eq!(d.ops().restores, 0);
			assert_eq!(d.ops().count(&Ev::ReplayFeed(vec![12, 13])), 0);
			// Hybrid verify feeds two chunks.
			assert_eq!(d.ops().count(&Ev::VerifyFeed(vec![6])), 1);
			assert_eq!(d.ops().count(&Ev::VerifyFeed(vec![12, 13])), 1);
		} else {
			assert_eq!(d.ops().count(&Ev::VerifyFeed(vec![6, 12, 13])), 1);
		}

		// Repeated-round parity: the next round starts from a clean
		// entry invariant.
		let before = d.ops().t_fed.len();
		let _ = d.run_round(next).unwrap();
		assert_eq!(d.committed_len(), d.ops().t_fed.len() - PROMPT.len());
		assert!(d.ops().t_fed.len() > before);
		assert_eq!(d.ops().pairs_fed(), d.ops().t_fed.len() - 1);
	}

	#[test]
	fn full_acceptance_greedy_pure_attention() {
		full_acceptance_round(greedy(), false);
	}

	#[test]
	fn full_acceptance_sampled_pure_attention() {
		full_acceptance_round(sampled(11), false);
	}

	#[test]
	fn full_acceptance_greedy_hybrid_fast_path() {
		full_acceptance_round(greedy(), true);
	}

	fn rejection_at_depth(a: usize, hybrid: bool) {
		// k = 3, target diverges after `a` accepted drafts.
		let drafts = vec![12u32, 13, 14];
		let mut target_chain: Vec<u32> = drafts[..a].to_vec();
		target_chain.push(9); // diverges from drafts[a]
		let (mut ops, frontier) = FakeOps::new(PROMPT, V, hybrid, true);
		script_target(&mut ops, 6, target_chain);
		script_draft(&mut ops, 6, drafts.clone());
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg {
			spec_k: Some(3),
			..Cfg::default()
		};
		let mut d = driver(ops, frontier, greedy(), &cfg, cbs.clone());
		let RoundEnd::Continue { next } = d.run_round(6).unwrap() else {
			panic!("expected Continue")
		};
		// Successor = target argmax at the rejection depth, unfed.
		assert_eq!(next, 9);
		let committed: Vec<u32> = std::iter::once(6)
			.chain(drafts[..a].iter().copied())
			.collect();
		assert_eq!(d.committed_len(), 1 + a);
		assert_eq!(d.ops().t_fed, [PROMPT, &committed[..]].concat());
		assert_eq!(*cbs.borrow(), committed);
		assert_eq!(
			d.ops().pairs.as_ref().unwrap(),
			&expect_pairs(PROMPT, &committed)
		);
		assert_eq!(
			d.frontier().unwrap(),
			&hid(&[PROMPT, &committed[..]].concat())
		);
		if a == 0 {
			// Full rejection: no accepted_by_depth bucket is incremented.
			assert_eq!(d.stats().accepted_by_depth.iter().sum::<u64>(), 0);
		} else {
			// One-based depth: `a` accepted drafts land at index `a - 1`
			// (a round accepting 1 draft lands at index 0).
			assert_eq!(d.stats().accepted_by_depth[a - 1], 1);
		}
		assert_eq!(d.stats().drafted, 3);
		if hybrid {
			assert_eq!(d.ops().restores, 1, "post-x restore expected");
			if a == 0 {
				// r = 0: no replay at all.
				assert!(!d.ops().log.iter().any(|e| matches!(e, Ev::ReplayFeed(_))));
			} else {
				// 1 <= r < k: exactly one replay of [d1..dr].
				assert_eq!(d.ops().count(&Ev::ReplayFeed(drafts[..a].to_vec())), 1);
			}
		}
	}

	#[test]
	fn rejection_at_each_depth_pure_attention() {
		for a in 0..3 {
			rejection_at_depth(a, false);
		}
	}

	#[test]
	fn rejection_at_each_depth_hybrid() {
		for a in 0..3 {
			rejection_at_depth(a, true);
		}
	}

	#[test]
	fn eos_at_each_draft_position() {
		for i in 1..=2usize {
			// Drafts and target agree on a chain whose i-th draft is EOS.
			let mut chain = vec![12u32, 13];
			chain[i - 1] = EOS;
			chain.truncate(i);
			let mut full_chain = chain.clone();
			full_chain.push(14); // bonus beyond
			let (mut ops, frontier) = FakeOps::new(PROMPT, V, false, true);
			script_target(&mut ops, 6, full_chain);
			script_draft(&mut ops, 6, chain.clone());
			let cbs = Rc::new(RefCell::new(Vec::new()));
			let cfg = Cfg::default();
			let mut d = driver(ops, frontier, greedy(), &cfg, cbs.clone());
			let end = d.run_round(6).unwrap();
			assert!(matches!(end, RoundEnd::Finished), "EOS must finish");
			// EOS emitted but uncommitted: committed ends before d_i.
			let committed: Vec<u32> = std::iter::once(6)
				.chain(chain[..i - 1].iter().copied())
				.collect();
			let emitted: Vec<u32> = std::iter::once(6)
				.chain(chain[..i].iter().copied())
				.collect();
			assert_eq!(d.committed_len(), 1 + (i - 1));
			assert_eq!(d.ops().t_fed, [PROMPT, &committed[..]].concat());
			assert_eq!(*cbs.borrow(), emitted);
			// MTP pairs = t + i - 1; frontier = row i-1.
			assert_eq!(
				d.ops().pairs.as_ref().unwrap(),
				&expect_pairs(PROMPT, &committed)
			);
			// The successor is never drawn on an exiting row.
			assert_eq!(d.ops().count(&Ev::Successor), 0);
			assert_eq!(d.stats().rounds, 1);
		}
	}

	#[test]
	fn eos_and_cancel_at_x_leave_entry_state() {
		for cancel in [false, true] {
			let (mut ops, frontier) = FakeOps::new(PROMPT, V, false, true);
			let x = if cancel { 6 } else { EOS };
			script_target(&mut ops, x, vec![12]);
			let cbs = Rc::new(RefCell::new(Vec::new()));
			let cfg = Cfg {
				cancel_at: cancel.then_some(0),
				..Cfg::default()
			};
			let mut d = driver(ops, frontier, greedy(), &cfg, cbs.clone());
			let end = d.run_round(x).unwrap();
			if cancel {
				assert!(matches!(end, RoundEnd::Aborted));
			} else {
				assert!(matches!(end, RoundEnd::Finished));
			}
			// x emitted but uncommitted; caches and MTP exactly at entry;
			// entry frontier retained; successor never drawn.
			assert_eq!(d.committed_len(), 0);
			assert_eq!(d.ops().t_fed, PROMPT);
			assert_eq!(*cbs.borrow(), vec![x]);
			assert_eq!(d.ops().pairs.as_ref().unwrap(), &expect_pairs(PROMPT, &[]));
			assert_eq!(d.frontier().unwrap(), &hid(PROMPT));
			assert!(d.ops().log.is_empty(), "no forwards on an entry exit");
		}
	}

	#[test]
	fn cancel_at_draft_position() {
		// Cancel at d_1 (callback index 1): committed …,x; d_1 emitted.
		let (mut ops, frontier) = FakeOps::new(PROMPT, V, false, true);
		script_target(&mut ops, 6, vec![12, 13, 14]);
		script_draft(&mut ops, 6, vec![12, 13]);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg {
			cancel_at: Some(1),
			..Cfg::default()
		};
		let mut d = driver(ops, frontier, greedy(), &cfg, cbs.clone());
		let end = d.run_round(6).unwrap();
		assert!(matches!(end, RoundEnd::Aborted));
		assert_eq!(d.committed_len(), 1);
		assert_eq!(d.ops().t_fed, [PROMPT, &[6]].concat());
		assert_eq!(*cbs.borrow(), vec![6, 12]);
		assert_eq!(d.ops().pairs.as_ref().unwrap(), &expect_pairs(PROMPT, &[6]));
		assert_eq!(d.ops().count(&Ev::Successor), 0);
	}

	#[test]
	fn slot_exhaustion_at_draft_commits_retained_prefix() {
		// max_tokens 3, k clamps to remaining-1 = 2; full acceptance hits
		// the budget at d_2 — committed includes d_2, no successor drawn.
		let (mut ops, frontier) = FakeOps::new(PROMPT, V, false, true);
		script_target(&mut ops, 6, vec![12, 13, 14]);
		script_draft(&mut ops, 6, vec![12, 13]);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg {
			max_tokens: 3,
			spec_k: Some(5),
			..Cfg::default()
		};
		let mut d = driver(ops, frontier, greedy(), &cfg, cbs.clone());
		let end = d.run_round(6).unwrap();
		assert!(matches!(end, RoundEnd::Budget));
		assert_eq!(d.committed_len(), 3);
		assert_eq!(d.ops().t_fed, [PROMPT, &[6, 12, 13]].concat());
		assert_eq!(*cbs.borrow(), vec![6, 12, 13]);
		assert_eq!(
			d.ops().pairs.as_ref().unwrap(),
			&expect_pairs(PROMPT, &[6, 12, 13])
		);
		assert_eq!(d.ops().count(&Ev::Successor), 0);
	}

	#[test]
	fn eos_on_bonus_defers_to_next_round_with_zero_forwards() {
		// Bonus successor is EOS: this round continues; the next round
		// emits it and exits at entry, costing zero forwards.
		let (mut ops, frontier) = FakeOps::new(PROMPT, V, false, true);
		script_target(&mut ops, 6, vec![12, 13, EOS]);
		script_draft(&mut ops, 6, vec![12, 13]);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg::default();
		let mut d = driver(ops, frontier, greedy(), &cfg, cbs.clone());
		let RoundEnd::Continue { next } = d.run_round(6).unwrap() else {
			panic!("expected Continue")
		};
		assert_eq!(next, EOS);
		let log_len = d.ops().log.len();
		let end = d.run_round(next).unwrap();
		assert!(matches!(end, RoundEnd::Finished));
		assert_eq!(d.ops().log.len(), log_len, "deferral round is free");
		assert_eq!(d.committed_len(), 3);
		assert_eq!(*cbs.borrow(), vec![6, 12, 13, EOS]);
	}

	#[test]
	fn final_slot_commits_pair_without_successor() {
		// max_tokens 1: the very first token is the final slot — feed,
		// commit, pair-commit; no draft, no successor sample.
		let (ops, frontier) = FakeOps::new(PROMPT, V, false, true);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg {
			max_tokens: 1,
			..Cfg::default()
		};
		let mut d = driver(ops, frontier, greedy(), &cfg, cbs.clone());
		let end = d.run_round(6).unwrap();
		assert!(matches!(end, RoundEnd::Budget));
		assert_eq!(d.committed_len(), 1);
		assert_eq!(d.ops().t_fed, [PROMPT, &[6]].concat());
		assert_eq!(d.ops().count(&Ev::ReplayFeed(vec![6])), 1);
		assert!(!d.ops().log.iter().any(|e| matches!(e, Ev::MtpStep(_))));
		assert_eq!(d.ops().pairs.as_ref().unwrap(), &expect_pairs(PROMPT, &[6]));
		assert_eq!(d.frontier().unwrap(), &hid(&[PROMPT, &[6]].concat()));
		assert_eq!(d.ops().count(&Ev::Successor), 0);
	}

	#[test]
	fn max_tokens_zero_emits_nothing() {
		let (ops, frontier) = FakeOps::new(PROMPT, V, false, true);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg {
			max_tokens: 0,
			..Cfg::default()
		};
		let mut d = driver(ops, frontier, greedy(), &cfg, cbs.clone());
		let (end, _) = run_all(&mut d, 6, 0).unwrap();
		assert!(end.is_none());
		assert!(cbs.borrow().is_empty());
		assert_eq!(d.committed_len(), 0);
		assert_eq!(d.ops().t_fed, PROMPT);
	}

	#[test]
	fn max_tokens_boundary_clamps_draft_depth() {
		// max_tokens 4 with config k = 8: round 1's k clamps to 3.
		let (mut ops, frontier) = FakeOps::new(PROMPT, V, false, true);
		script_target(&mut ops, 6, vec![12, 13, 14, 15]);
		script_draft(&mut ops, 6, vec![12, 13, 14]);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg {
			max_tokens: 4,
			spec_k: Some(8),
			..Cfg::default()
		};
		let mut d = driver(ops, frontier, greedy(), &cfg, cbs.clone());
		let end = d.run_round(6).unwrap();
		assert!(matches!(end, RoundEnd::Budget));
		assert_eq!(d.stats().drafted, 3, "k = remaining - 1 = 3");
		assert_eq!(d.committed_len(), 4);
		assert_eq!(*cbs.borrow(), vec![6, 12, 13, 14]);
	}

	#[test]
	fn one_token_prompt_supports_speculation() {
		// 1-token prompt: pairs_fed = 0, frontier = hid(prompt).
		let prompt = &[1u32];
		let (mut ops, frontier) = FakeOps::new(prompt, V, false, true);
		assert_eq!(ops.pairs_fed(), 0);
		script_target(&mut ops, 6, vec![12, 13, 14]);
		script_draft(&mut ops, 6, vec![12, 13]);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg::default();
		let mut d = driver(ops, frontier, greedy(), &cfg, cbs.clone());
		let RoundEnd::Continue { next } = d.run_round(6).unwrap() else {
			panic!("expected Continue")
		};
		assert_eq!(next, 14);
		assert_eq!(d.committed_len(), 3);
		assert_eq!(
			d.ops().pairs.as_ref().unwrap(),
			&expect_pairs(prompt, &[6, 12, 13])
		);
	}

	// -----------------------------------------------------------------
	// Forced close
	// -----------------------------------------------------------------

	fn forced_close_at_x_case(sampler: Sampler) {
		// Budget 0 fires at x itself: emit close, replay [x] + C from
		// entry, commit both ledgers + all MTP pairs, arm, draw the
		// post-close successor from the replay's last-row dist.
		let (ops, frontier) = FakeOps::new(PROMPT, V, false, true);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg {
			budget: Some(0),
			..Cfg::default()
		};
		let mut d = driver(ops, frontier, sampler, &cfg, cbs.clone());
		let RoundEnd::Continue { next } = d.run_round(6).unwrap() else {
			panic!("expected Continue")
		};
		assert_eq!(next, default_tok(&[6, CLOSE], V, 17));
		assert_eq!(d.committed_len(), 2);
		assert_eq!(d.ops().t_fed, [PROMPT, &[6, CLOSE]].concat());
		assert_eq!(*cbs.borrow(), vec![6, CLOSE]);
		// Exactly one feed: the replay; never a verify double-feed.
		assert_eq!(d.ops().count(&Ev::ReplayFeed(vec![6, CLOSE])), 1);
		assert!(!d.ops().log.iter().any(|e| matches!(e, Ev::VerifyFeed(_))));
		assert_eq!(
			d.ops().pairs.as_ref().unwrap(),
			&expect_pairs(PROMPT, &[6, CLOSE])
		);
		assert_eq!(d.ops().count(&Ev::Successor), 1);
	}

	#[test]
	fn forced_close_at_x_greedy() {
		forced_close_at_x_case(greedy());
	}

	#[test]
	fn forced_close_at_x_sampled() {
		forced_close_at_x_case(sampled(21));
	}

	#[test]
	fn forced_close_trigger_cancellation_keeps_accepted_prefix() {
		// Cancellation at the close token: the cancelled token is emitted
		// but excluded from C and from committed_ids; no successor drawn.
		let (ops, frontier) = FakeOps::new(PROMPT, V, false, true);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg {
			budget: Some(0),
			cancel_at: Some(1),
			..Cfg::default()
		};
		let mut d = driver(ops, frontier, greedy(), &cfg, cbs.clone());
		let end = d.run_round(6).unwrap();
		assert!(matches!(end, RoundEnd::Aborted));
		assert_eq!(d.committed_len(), 1);
		assert_eq!(d.ops().t_fed, [PROMPT, &[6]].concat());
		assert_eq!(*cbs.borrow(), vec![6, CLOSE]);
		assert_eq!(d.ops().count(&Ev::ReplayFeed(vec![6])), 1);
		assert_eq!(d.ops().pairs.as_ref().unwrap(), &expect_pairs(PROMPT, &[6]));
		assert_eq!(d.ops().count(&Ev::Successor), 0);
	}

	#[test]
	fn forced_close_at_draft_replays_from_entry() {
		// Budget 2 fires at d_1 of a speculative round: x is reasoning token
		// one and d_1 is token two. Restore to ENTRY
		// (never post-x + replay-x), replay [x, d1] + C, commit pairs.
		let (mut ops, frontier) = FakeOps::new(PROMPT, V, false, true);
		script_target(&mut ops, 6, vec![12, 13, 14]);
		script_draft(&mut ops, 6, vec![12, 13]);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg {
			budget: Some(2),
			..Cfg::default()
		};
		let mut d = driver(ops, frontier, greedy(), &cfg, cbs.clone());
		let RoundEnd::Continue { next } = d.run_round(6).unwrap() else {
			panic!("expected Continue")
		};
		assert_eq!(next, default_tok(&[6, 12, CLOSE], V, 17));
		assert_eq!(d.committed_len(), 3);
		assert_eq!(d.ops().t_fed, [PROMPT, &[6, 12, CLOSE]].concat());
		assert_eq!(*cbs.borrow(), vec![6, 12, CLOSE]);
		assert_eq!(d.ops().restores, 1, "reconcile to entry via restore");
		assert_eq!(d.ops().count(&Ev::ReplayFeed(vec![6, 12, CLOSE])), 1);
		assert_eq!(
			d.ops().pairs.as_ref().unwrap(),
			&expect_pairs(PROMPT, &[6, 12, CLOSE])
		);
		assert_eq!(d.ops().count(&Ev::Successor), 1);
	}

	#[test]
	fn empty_close_at_x_feeds_once() {
		// Empty close encoding at trigger x: feed [x] only — no replay,
		// no double-feed; pair committed; successor from the feed's dist.
		let (mut ops, frontier) = FakeOps::new(PROMPT, V, false, true);
		script_target(&mut ops, 6, vec![12]);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg {
			budget: Some(0),
			close_override: Some(Vec::new()),
			..Cfg::default()
		};
		let mut d = driver(ops, frontier, greedy(), &cfg, cbs.clone());
		let RoundEnd::Continue { next } = d.run_round(6).unwrap() else {
			panic!("expected Continue")
		};
		assert_eq!(next, 12);
		assert_eq!(d.committed_len(), 1);
		assert_eq!(d.ops().count(&Ev::VerifyFeed(vec![6])), 1);
		assert!(!d.ops().log.iter().any(|e| matches!(e, Ev::ReplayFeed(_))));
		assert_eq!(d.ops().pairs.as_ref().unwrap(), &expect_pairs(PROMPT, &[6]));
		assert_eq!(*cbs.borrow(), vec![6]);
	}

	#[test]
	fn empty_close_at_draft_reconciles_without_extra_feed() {
		// Empty close encoding at trigger d_1 with budget 2 (x is token one):
		// NO extra feed — reconcile
		// as the slot-exhaustion row; successor from verify row-1's dist.
		let (mut ops, frontier) = FakeOps::new(PROMPT, V, false, true);
		script_target(&mut ops, 6, vec![12, 13, 14]);
		script_draft(&mut ops, 6, vec![12, 13]);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg {
			budget: Some(2),
			close_override: Some(Vec::new()),
			..Cfg::default()
		};
		let mut d = driver(ops, frontier, greedy(), &cfg, cbs.clone());
		let feeds_before = |log: &[Ev]| {
			log.iter()
				.filter(|e| matches!(e, Ev::VerifyFeed(_) | Ev::ReplayFeed(_)))
				.count()
		};
		let RoundEnd::Continue { next } = d.run_round(6).unwrap() else {
			panic!("expected Continue")
		};
		// Successor = row-1 target dist (scripted successor of [6, 12]).
		assert_eq!(next, 13);
		assert_eq!(d.committed_len(), 2);
		assert_eq!(d.ops().t_fed, [PROMPT, &[6, 12]].concat());
		// Exactly one feed happened: the verify feed itself.
		assert_eq!(feeds_before(&d.ops().log), 1);
		assert_eq!(
			d.ops().pairs.as_ref().unwrap(),
			&expect_pairs(PROMPT, &[6, 12])
		);
		assert_eq!(*cbs.borrow(), vec![6, 12]);
		assert_eq!(d.ops().count(&Ev::Successor), 1);
	}

	#[test]
	fn over_max_forced_close_skips_post_close_draw() {
		// Final-slot forced close may exceed max_tokens (close ids bypass
		// the counter) but never draws the unused post-close successor.
		let (ops, frontier) = FakeOps::new(PROMPT, V, false, true);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg {
			budget: Some(0),
			max_tokens: 1,
			..Cfg::default()
		};
		let mut d = driver(ops, frontier, greedy(), &cfg, cbs.clone());
		let end = d.run_round(6).unwrap();
		assert!(matches!(end, RoundEnd::Budget));
		assert_eq!(*cbs.borrow(), vec![6, CLOSE], "close exceeds max_tokens");
		assert_eq!(d.committed_len(), 2);
		assert_eq!(d.ops().t_fed, [PROMPT, &[6, CLOSE]].concat());
		assert_eq!(d.ops().count(&Ev::Successor), 0);
	}

	// -----------------------------------------------------------------
	// Failure table, row by row
	// -----------------------------------------------------------------

	#[test]
	fn emitter_failure_at_x_is_exit_exact_at_entry() {
		let (ops, frontier) = FakeOps::new(PROMPT, V, false, true);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg {
			emitter_fail_at: Some(0),
			..Cfg::default()
		};
		let mut d = driver(ops, frontier, greedy(), &cfg, cbs.clone());
		assert!(d.run_round(6).is_err());
		// Zero callbacks for x; target and MTP untouched at entry.
		assert!(cbs.borrow().is_empty());
		assert_eq!(d.committed_len(), 0);
		assert_eq!(d.ops().t_fed, PROMPT);
		assert_eq!(d.ops().pairs.as_ref().unwrap(), &expect_pairs(PROMPT, &[]));
		assert!(d.ops().log.is_empty());
	}

	#[test]
	fn draft_forward_failure_discards_and_continues_target_only() {
		let (mut ops, frontier) = FakeOps::new(PROMPT, V, false, true);
		script_target(&mut ops, 6, vec![12, 13]);
		ops.faults.mtp_step.arm(0, FaultKind::MtpForward);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg::default();
		let mut d = driver(ops, frontier, greedy(), &cfg, cbs.clone());
		let RoundEnd::Continue { next } = d.run_round(6).unwrap() else {
			panic!("expected Continue")
		};
		assert_eq!(next, 12);
		assert_eq!(d.committed_len(), 1);
		assert_eq!(d.ops().t_fed, [PROMPT, &[6]].concat());
		assert!(d.ops().pairs.is_none(), "MTP discarded");
		assert!(d.frontier().is_none());
		assert_eq!(d.ops().count(&Ev::MtpDiscard), 1);
		assert_eq!(*cbs.borrow(), vec![6]);
		// Subsequent rounds are target-only: no further MtpStep.
		let steps = d
			.ops()
			.log
			.iter()
			.filter(|e| matches!(e, Ev::MtpStep(_)))
			.count();
		let _ = d.run_round(next).unwrap();
		let steps_after = d
			.ops()
			.log
			.iter()
			.filter(|e| matches!(e, Ev::MtpStep(_)))
			.count();
		assert_eq!(steps, steps_after);
		assert_eq!(d.stats().rounds, 0);
		// The draft forward failed on the very first step: nothing was
		// proposed, so drafted stays 0 even under draft-time counting.
		assert_eq!(d.stats().drafted, 0);
	}

	#[test]
	fn draft_host_failure_truncates_recommits_and_resumes() {
		// Host-side failure between MTP forwards: the provisional draft
		// pair is dropped (truncate to entry), the (frontier_entry, x)
		// pair re-commits (pairs_fed == t), and speculation resumes.
		let (mut ops, frontier) = FakeOps::new(PROMPT, V, false, true);
		script_target(&mut ops, 6, vec![12, 13]);
		script_draft(&mut ops, 6, vec![12, 13]);
		ops.faults.mtp_step.arm(1, FaultKind::HostBare);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg::default();
		let mut d = driver(ops, frontier, greedy(), &cfg, cbs.clone());
		let RoundEnd::Continue { next } = d.run_round(6).unwrap() else {
			panic!("expected Continue")
		};
		assert_eq!(next, 12);
		assert_eq!(d.committed_len(), 1);
		assert_eq!(d.ops().pairs.as_ref().unwrap(), &expect_pairs(PROMPT, &[6]));
		assert_eq!(d.frontier().unwrap(), &hid(&[PROMPT, &[6]].concat()));
		assert_eq!(*cbs.borrow(), vec![6]);
		// Draft-time counting: d1 was proposed before the step-2 host
		// failure, so the failed round still reports drafted = 1 with
		// rounds = 0 (no decision was reached).
		assert_eq!(d.stats().drafted, 1);
		assert_eq!(d.stats().rounds, 0);
		// Speculation resumes next round.
		let steps = d
			.ops()
			.log
			.iter()
			.filter(|e| matches!(e, Ev::MtpStep(_)))
			.count();
		let _ = d.run_round(next).unwrap();
		let steps_after = d
			.ops()
			.log
			.iter()
			.filter(|e| matches!(e, Ev::MtpStep(_)))
			.count();
		assert!(steps_after > steps, "speculation must resume");
	}

	#[test]
	fn draft_draw_failure_resumes_speculation() {
		// A failed draft draw is a host-side failure between MTP forwards.
		let (mut ops, frontier) = FakeOps::new(PROMPT, V, false, true);
		script_target(&mut ops, 6, vec![12, 13]);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg::default();
		let sampler = sampled(5);
		sampler.inject_failure_at(0);
		let mut d = driver(ops, frontier, sampler, &cfg, cbs.clone());
		let end = d.run_round(6).unwrap();
		assert!(matches!(end, RoundEnd::Continue { .. }));
		assert_eq!(d.committed_len(), 1);
		assert_eq!(d.ops().pairs.as_ref().unwrap(), &expect_pairs(PROMPT, &[6]));
		assert_eq!(*cbs.borrow(), vec![6]);
	}

	#[test]
	fn stage4_draw_failure_target_only_is_exit_exact() {
		let (mut ops, _) = FakeOps::new(PROMPT, V, false, false);
		script_target(&mut ops, 6, vec![12]);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg {
			spec_k: None,
			..Cfg::default()
		};
		let sampler = sampled(5);
		sampler.inject_failure_at(0);
		let mut d = driver(ops, None, sampler, &cfg, cbs.clone());
		assert!(d.run_round(6).is_err());
		// exact-prefix exit at …, x: fed and committed, callback delivered once.
		assert_eq!(d.committed_len(), 1);
		assert_eq!(d.ops().t_fed, [PROMPT, &[6]].concat());
		assert_eq!(*cbs.borrow(), vec![6]);
	}

	#[test]
	fn stage4_residual_draw_failure_at_each_depth() {
		for a in 0..2usize {
			let drafts = vec![12u32, 13];
			let mut chain: Vec<u32> = drafts[..a].to_vec();
			chain.push(9);
			let (mut ops, frontier) = FakeOps::new(PROMPT, V, false, true);
			script_target(&mut ops, 6, chain);
			script_draft(&mut ops, 6, drafts.clone());
			let cbs = Rc::new(RefCell::new(Vec::new()));
			let cfg = Cfg::default();
			let sampler = sampled(5);
			// Calls: draft draws (0, 1), then the residual draw (2) -
			// verify_accept is NOT a fallible-call site (acceptance-uniform).
			sampler.inject_failure_at(2);
			let mut d = driver(ops, frontier, sampler, &cfg, cbs.clone());
			assert!(d.run_round(6).is_err());
			// Stage 3 preceded stage 4: the retained prefix is committed
			// and the MTP pairs already replayed.
			let committed: Vec<u32> = std::iter::once(6)
				.chain(drafts[..a].iter().copied())
				.collect();
			assert_eq!(d.committed_len(), 1 + a);
			assert_eq!(d.ops().t_fed, [PROMPT, &committed[..]].concat());
			assert_eq!(
				d.ops().pairs.as_ref().unwrap(),
				&expect_pairs(PROMPT, &committed)
			);
			assert_eq!(*cbs.borrow(), committed);
		}
	}

	#[test]
	fn stage4_bonus_draw_failure_is_exit_exact() {
		let (mut ops, frontier) = FakeOps::new(PROMPT, V, false, true);
		script_target(&mut ops, 6, vec![12, 13, 14]);
		script_draft(&mut ops, 6, vec![12, 13]);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg::default();
		let sampler = sampled(5);
		// Calls: draft draws (0, 1), then the bonus draw (2) -
		// verify_accept is NOT a fallible-call site (acceptance-uniform).
		sampler.inject_failure_at(2);
		let mut d = driver(ops, frontier, sampler, &cfg, cbs.clone());
		assert!(d.run_round(6).is_err());
		assert_eq!(d.committed_len(), 3);
		assert_eq!(d.ops().t_fed, [PROMPT, &[6, 12, 13]].concat());
		assert_eq!(
			d.ops().pairs.as_ref().unwrap(),
			&expect_pairs(PROMPT, &[6, 12, 13])
		);
	}

	#[test]
	fn verify_forward_failure_is_exit_invalid() {
		let (mut ops, frontier) = FakeOps::new(PROMPT, V, false, true);
		script_draft(&mut ops, 6, vec![12, 13]);
		ops.faults.verify_feed.arm(0, FaultKind::Invalid);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg::default();
		let mut d = driver(ops, frontier, greedy(), &cfg, cbs.clone());
		assert!(d.run_round(6).is_err());
	}

	#[test]
	fn verify_host_failure_with_salvage_continues_target_only() {
		let (mut ops, frontier) = FakeOps::new(PROMPT, V, false, true);
		script_target(&mut ops, 6, vec![12, 13, 14]);
		script_draft(&mut ops, 6, vec![12, 13]);
		ops.faults.verify_feed.arm(0, FaultKind::HostSalvaged);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg::default();
		let mut d = driver(ops, frontier, greedy(), &cfg, cbs.clone());
		let RoundEnd::Continue { next } = d.run_round(6).unwrap() else {
			panic!("expected Continue")
		};
		// Reconciled to …, x; MTP discarded; row-0 successor drawn.
		assert_eq!(next, 12);
		assert_eq!(d.committed_len(), 1);
		assert_eq!(d.ops().t_fed, [PROMPT, &[6]].concat());
		assert!(d.ops().pairs.is_none());
		assert_eq!(*cbs.borrow(), vec![6]);
	}

	#[test]
	fn verify_host_failure_without_salvage_is_exit_exact() {
		let (mut ops, frontier) = FakeOps::new(PROMPT, V, false, true);
		script_draft(&mut ops, 6, vec![12, 13]);
		ops.faults.verify_feed.arm(0, FaultKind::HostBare);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg::default();
		let mut d = driver(ops, frontier, greedy(), &cfg, cbs.clone());
		assert!(d.run_round(6).is_err());
		assert_eq!(d.committed_len(), 1);
		assert_eq!(d.ops().t_fed, [PROMPT, &[6]].concat());
		assert!(d.ops().pairs.is_none());
	}

	#[test]
	fn hybrid_verify_chunk2_host_failure_recovers_from_chunk1_dist() {
		let (mut ops, frontier) = FakeOps::new(PROMPT, V, true, true);
		script_target(&mut ops, 6, vec![12, 13, 14]);
		script_draft(&mut ops, 6, vec![12, 13]);
		ops.faults.verify_feed.arm(1, FaultKind::HostBare);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg::default();
		let mut d = driver(ops, frontier, greedy(), &cfg, cbs.clone());
		let RoundEnd::Continue { next } = d.run_round(6).unwrap() else {
			panic!("expected Continue")
		};
		// Row-0 dist came from chunk 1 despite the bare chunk-2 salvage.
		assert_eq!(next, 12);
		assert_eq!(d.committed_len(), 1);
		assert_eq!(d.ops().t_fed, [PROMPT, &[6]].concat());
		assert_eq!(d.ops().restores, 1, "post-x restore before truncate");
		assert!(d.ops().pairs.is_none());
	}

	/// Crafted q/p pair whose rejection residual is exactly all-zero
	/// (mirrors the sampling.rs unit test): q has tiny mass at 0, p has
	/// none, p <= q elementwise, both within the ~1 sum tolerance.
	fn zero_residual_dists(vocab: usize) -> (Vec<f32>, Vec<f32>) {
		let mut q = vec![0.0f32; vocab];
		q[0] = 0.0012;
		q[1] = 0.9993;
		let mut p = vec![0.0f32; vocab];
		p[1] = 0.9993;
		(q, p)
	}

	/// A seed whose first `sample_from_probs(q)` draw lands on token 0.
	fn seed_drawing_zero(q: &[f32]) -> u64 {
		(0..100_000u64)
			.find(|&s| {
				let mut smp = sampled(s);
				smp.sample_from_probs(q).unwrap() == 0
			})
			.expect("no seed draws the tiny-mass token")
	}

	fn invariant_err_setup() -> (FakeOps, Option<Hidden>, u64) {
		let (mut ops, frontier) = FakeOps::new(PROMPT, V, false, true);
		let (q, p) = zero_residual_dists(V);
		let seed = seed_drawing_zero(&q);
		ops.draft_dist = Some(Box::new(move |_| q.clone()));
		ops.target_dist = Some(Box::new(move |suffix| {
			if suffix == [6] {
				p.clone()
			} else {
				one_hot(V, default_tok(suffix, V, 17))
			}
		}));
		(ops, frontier, seed)
	}

	#[test]
	fn invariant_err_recommits_pair_and_resumes() {
		// verify_accept all-zero-residual invariant Err: reconcile to
		// …, x; MTP truncates to entry then re-commits the (frontier, x)
		// pair (pairs_fed == t); stage-4 draw from the UNDRAWN verify
		// row-0 distribution; speculation resumes next round.
		let (ops, frontier, seed) = invariant_err_setup();
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg {
			spec_k: Some(1),
			..Cfg::default()
		};
		let mut d = driver(ops, frontier, sampled(seed), &cfg, cbs.clone());
		let RoundEnd::Continue { next } = d.run_round(6).unwrap() else {
			panic!("expected Continue")
		};
		// Row-0 target dist has all its mass at token 1.
		assert_eq!(next, 1);
		assert_eq!(d.committed_len(), 1);
		assert_eq!(d.ops().t_fed, [PROMPT, &[6]].concat());
		assert_eq!(d.ops().pairs.as_ref().unwrap(), &expect_pairs(PROMPT, &[6]));
		assert_eq!(d.frontier().unwrap(), &hid(&[PROMPT, &[6]].concat()));
		assert_eq!(*cbs.borrow(), vec![6]);
		// An invariant-Err round records no ROUND stats (rounds and
		// accepted_by_depth stay per contract, sum <= rounds)...
		assert_eq!(d.stats().rounds, 0);
		assert!(d.stats().accepted_by_depth.is_empty());
		// ...but `drafted` counts at draft time, so the failed round's
		// proposal (spec_k = 1) is still visible.
		assert_eq!(d.stats().drafted, 1);
		// Speculation resumes next round.
		let steps = d.ops().count(&Ev::MtpStep(next));
		let _ = d.run_round(next).unwrap();
		assert!(d.ops().count(&Ev::MtpStep(next)) > steps);
	}

	#[test]
	fn invariant_err_row0_draw_failure_is_exit_exact() {
		let (ops, frontier, seed) = invariant_err_setup();
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg {
			spec_k: Some(1),
			..Cfg::default()
		};
		let sampler = sampled(seed);
		// Calls: draft draw (0), then the row-0 draw (1) - verify_accept
		// is NOT a fallible-call site (acceptance-uniform).
		sampler.inject_failure_at(1);
		let mut d = driver(ops, frontier, sampler, &cfg, cbs.clone());
		assert!(d.run_round(6).is_err());
		// The pair re-commit (stage 3) preceded the failed draw.
		assert_eq!(d.committed_len(), 1);
		assert_eq!(d.ops().t_fed, [PROMPT, &[6]].concat());
		assert_eq!(d.ops().pairs.as_ref().unwrap(), &expect_pairs(PROMPT, &[6]));
	}

	#[test]
	fn recovery_feed_draw_failure_is_exit_exact() {
		// Draft host failure resumes; the recovery feed's stage-4 draw
		// then fails: exact-prefix exit at …, x with the pair already committed.
		let (mut ops, frontier) = FakeOps::new(PROMPT, V, false, true);
		ops.faults.mtp_step.arm(1, FaultKind::HostBare);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg::default();
		let sampler = sampled(5);
		// Calls: draft d1 draw (0), recovery stage-4 draw (1).
		sampler.inject_failure_at(1);
		let mut d = driver(ops, frontier, sampler, &cfg, cbs.clone());
		assert!(d.run_round(6).is_err());
		assert_eq!(d.committed_len(), 1);
		assert_eq!(d.ops().t_fed, [PROMPT, &[6]].concat());
		assert_eq!(d.ops().pairs.as_ref().unwrap(), &expect_pairs(PROMPT, &[6]));
		assert_eq!(*cbs.borrow(), vec![6]);
	}

	#[test]
	fn post_reconciliation_detach_failure_discards_but_still_draws() {
		// Hybrid 1 <= r < k with a host-failing reconcile replay: target
		// valid at …, P; MTP discarded; the pending stage-4 draw proceeds.
		let (mut ops, frontier) = FakeOps::new(PROMPT, V, true, true);
		script_target(&mut ops, 6, vec![12, 9]);
		script_draft(&mut ops, 6, vec![12, 13]);
		ops.faults.replay_feed.arm(0, FaultKind::HostBare);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg::default();
		let mut d = driver(ops, frontier, greedy(), &cfg, cbs.clone());
		let RoundEnd::Continue { next } = d.run_round(6).unwrap() else {
			panic!("expected Continue")
		};
		assert_eq!(next, 9, "verdict successor still drawn");
		assert_eq!(d.committed_len(), 2);
		assert_eq!(d.ops().t_fed, [PROMPT, &[6, 12]].concat());
		assert!(d.ops().pairs.is_none());
		assert!(d.frontier().is_none());
		assert_eq!(d.ops().count(&Ev::Successor), 1);
	}

	#[test]
	fn frontier_detach_failure_in_target_only_discards_and_continues() {
		// Window-skip target-only with a host-failing feed: x committed,
		// MTP discarded, successor drawn from the salvaged dist.
		let (mut ops, frontier) = FakeOps::new(PROMPT, V, false, true);
		script_target(&mut ops, 6, vec![12]);
		ops.window_trim = true;
		ops.faults.verify_feed.arm(0, FaultKind::HostSalvaged);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg::default();
		let mut d = driver(ops, frontier, greedy(), &cfg, cbs.clone());
		let RoundEnd::Continue { next } = d.run_round(6).unwrap() else {
			panic!("expected Continue")
		};
		assert_eq!(next, 12);
		assert_eq!(d.committed_len(), 1);
		assert!(d.ops().pairs.is_none());
		assert!(d.frontier().is_none());
	}

	#[test]
	fn frontier_detach_failure_in_final_slot_returns_without_drawing() {
		let (mut ops, frontier) = FakeOps::new(PROMPT, V, false, true);
		ops.faults.replay_feed.arm(0, FaultKind::HostBare);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg {
			max_tokens: 1,
			..Cfg::default()
		};
		let mut d = driver(ops, frontier, greedy(), &cfg, cbs.clone());
		let end = d.run_round(6).unwrap();
		assert!(matches!(end, RoundEnd::Budget));
		assert_eq!(d.committed_len(), 1);
		assert!(d.ops().pairs.is_none());
		assert_eq!(d.ops().count(&Ev::Successor), 0);
	}

	#[test]
	fn out_of_range_truncate_guard_is_exit_invalid() {
		let (mut ops, frontier) = FakeOps::new(PROMPT, V, false, true);
		script_target(&mut ops, 6, vec![12, 9]);
		script_draft(&mut ops, 6, vec![12, 13]);
		ops.faults.truncate.arm(0, FaultKind::Invalid);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg::default();
		let mut d = driver(ops, frontier, greedy(), &cfg, cbs.clone());
		assert!(d.run_round(6).is_err());
	}

	#[test]
	fn restore_failure_is_exit_invalid() {
		// Hybrid r = 0: the post-x restore is the first restore call.
		let (mut ops, frontier) = FakeOps::new(PROMPT, V, true, true);
		script_target(&mut ops, 6, vec![9]);
		script_draft(&mut ops, 6, vec![12, 13]);
		ops.faults.restore.arm(0, FaultKind::Invalid);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg::default();
		let mut d = driver(ops, frontier, greedy(), &cfg, cbs.clone());
		assert!(d.run_round(6).is_err());
	}

	#[test]
	fn sequential_emission_failure_reconciles_and_discards() {
		// Emitter fails at d_2 (emitted[2]): reconcile to [x, d1],
		// discard MTP, propagate — exact-prefix exit.
		let (mut ops, frontier) = FakeOps::new(PROMPT, V, false, true);
		script_target(&mut ops, 6, vec![12, 13, 14, 15]);
		script_draft(&mut ops, 6, vec![12, 13, 14]);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg {
			spec_k: Some(3),
			emitter_fail_at: Some(2),
			..Cfg::default()
		};
		let mut d = driver(ops, frontier, greedy(), &cfg, cbs.clone());
		assert!(d.run_round(6).is_err());
		assert_eq!(d.committed_len(), 2);
		assert_eq!(d.ops().t_fed, [PROMPT, &[6, 12]].concat());
		assert!(d.ops().pairs.is_none());
		assert_eq!(*cbs.borrow(), vec![6, 12]);
		assert_eq!(d.ops().count(&Ev::Successor), 0);
	}

	#[test]
	fn close_token_emission_failure_feeds_prefix_then_propagates() {
		// Emission failure on the close token itself: the accepted close
		// prefix (empty here) is still replayed with the trigger, MTP is
		// discarded, and the error propagates (exact-prefix exit).
		let (ops, frontier) = FakeOps::new(PROMPT, V, false, true);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg {
			budget: Some(0),
			emitter_fail_at: Some(1),
			..Cfg::default()
		};
		let mut d = driver(ops, frontier, greedy(), &cfg, cbs.clone());
		assert!(d.run_round(6).is_err());
		assert_eq!(d.committed_len(), 1);
		assert_eq!(d.ops().t_fed, [PROMPT, &[6]].concat());
		assert_eq!(d.ops().count(&Ev::ReplayFeed(vec![6])), 1);
		assert!(d.ops().pairs.is_none());
		assert_eq!(*cbs.borrow(), vec![6]);
		assert_eq!(d.ops().count(&Ev::Successor), 0);
	}

	#[test]
	fn mtp_commit_replay_failure_discards_but_continues() {
		let (mut ops, frontier) = FakeOps::new(PROMPT, V, false, true);
		script_target(&mut ops, 6, vec![12, 13, 14]);
		script_draft(&mut ops, 6, vec![12, 13]);
		ops.faults.mtp_commit.arm(0, FaultKind::MtpForward);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg::default();
		let mut d = driver(ops, frontier, greedy(), &cfg, cbs.clone());
		let RoundEnd::Continue { next } = d.run_round(6).unwrap() else {
			panic!("expected Continue")
		};
		assert_eq!(next, 14, "pending stage-4 draw proceeds");
		assert_eq!(d.committed_len(), 3);
		assert_eq!(d.ops().t_fed, [PROMPT, &[6, 12, 13]].concat());
		assert!(d.ops().pairs.is_none());
		assert!(d.frontier().is_none());
	}

	#[test]
	fn forced_close_mtp_replay_failure_discards_but_continues() {
		let (ops, frontier) = FakeOps::new(PROMPT, V, false, true);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let mut cfg = Cfg {
			budget: Some(0),
			..Cfg::default()
		};
		cfg.spec_k = Some(2);
		let mut d = {
			let mut ops = ops;
			ops.faults.mtp_commit.arm(0, FaultKind::MtpForward);
			driver(ops, frontier, greedy(), &cfg, cbs.clone())
		};
		let RoundEnd::Continue { next } = d.run_round(6).unwrap() else {
			panic!("expected Continue")
		};
		assert_eq!(next, default_tok(&[6, CLOSE], V, 17));
		assert_eq!(d.committed_len(), 2);
		assert_eq!(d.ops().t_fed, [PROMPT, &[6, CLOSE]].concat());
		assert!(d.ops().pairs.is_none());
	}

	#[test]
	fn forced_close_replay_detach_failure_discards_and_draws_salvage() {
		// Frontier-detach failure inside the forced-close replay: the
		// replayed sequence is committed, the MTP is discarded, and the
		// post-close successor still comes from the salvaged last-row
		// dist (frontier-detach-in-forced-close row).
		let (mut ops, frontier) = FakeOps::new(PROMPT, V, false, true);
		ops.faults.replay_feed.arm(0, FaultKind::HostSalvaged);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg {
			budget: Some(0),
			..Cfg::default()
		};
		let mut d = driver(ops, frontier, greedy(), &cfg, cbs.clone());
		let RoundEnd::Continue { next } = d.run_round(6).unwrap() else {
			panic!("expected Continue")
		};
		assert_eq!(next, default_tok(&[6, CLOSE], V, 17));
		assert_eq!(d.committed_len(), 2);
		assert_eq!(d.ops().t_fed, [PROMPT, &[6, CLOSE]].concat());
		assert!(d.ops().pairs.is_none());
		assert!(d.frontier().is_none());
	}

	#[test]
	fn forced_close_target_replay_failure_is_exit_invalid() {
		let (mut ops, frontier) = FakeOps::new(PROMPT, V, false, true);
		ops.faults.replay_feed.arm(0, FaultKind::Invalid);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg {
			budget: Some(0),
			..Cfg::default()
		};
		let mut d = driver(ops, frontier, greedy(), &cfg, cbs.clone());
		assert!(d.run_round(6).is_err());
	}

	#[test]
	fn hybrid_reconciliation_replay_failure_is_exit_invalid() {
		let (mut ops, frontier) = FakeOps::new(PROMPT, V, true, true);
		script_target(&mut ops, 6, vec![12, 9]);
		script_draft(&mut ops, 6, vec![12, 13]);
		ops.faults.replay_feed.arm(0, FaultKind::Invalid);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg::default();
		let mut d = driver(ops, frontier, greedy(), &cfg, cbs.clone());
		assert!(d.run_round(6).is_err());
	}

	#[test]
	fn forced_close_post_replay_draw_failure_is_exit_exact() {
		// Spec-off sampled: the post-close successor draw is the round's
		// only sampler call; its failure exits with the replayed sequence
		// committed.
		let (ops, _) = FakeOps::new(PROMPT, V, false, false);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg {
			budget: Some(0),
			spec_k: None,
			..Cfg::default()
		};
		let sampler = sampled(5);
		sampler.inject_failure_at(0);
		let mut d = driver(ops, None, sampler, &cfg, cbs.clone());
		assert!(d.run_round(6).is_err());
		assert_eq!(d.committed_len(), 2);
		assert_eq!(d.ops().t_fed, [PROMPT, &[6, CLOSE]].concat());
		assert_eq!(*cbs.borrow(), vec![6, CLOSE]);
	}

	#[test]
	fn empty_close_row_i_draw_failure_is_exit_exact() {
		// Empty-C at d_1 with budget 2 (x is token one): the row-1 draw fails after the
		// reconciliation and MTP commit completed.
		let (mut ops, frontier) = FakeOps::new(PROMPT, V, false, true);
		script_target(&mut ops, 6, vec![12, 13, 14]);
		script_draft(&mut ops, 6, vec![12, 13]);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg {
			budget: Some(2),
			close_override: Some(Vec::new()),
			..Cfg::default()
		};
		let sampler = sampled(5);
		// Calls: draft draws (0, 1), then the row-1 draw (2) -
		// verify_accept is NOT a fallible-call site (acceptance-uniform).
		sampler.inject_failure_at(2);
		let mut d = driver(ops, frontier, sampler, &cfg, cbs.clone());
		assert!(d.run_round(6).is_err());
		assert_eq!(d.committed_len(), 2);
		assert_eq!(d.ops().t_fed, [PROMPT, &[6, 12]].concat());
		assert_eq!(
			d.ops().pairs.as_ref().unwrap(),
			&expect_pairs(PROMPT, &[6, 12])
		);
		assert_eq!(*cbs.borrow(), vec![6, 12]);
	}

	// -----------------------------------------------------------------
	// Properties, stats, determinism
	// -----------------------------------------------------------------

	#[test]
	fn greedy_spec_on_equals_spec_off_on_the_fake() {
		// Property: greedy spec-on ≡ spec-off token sequences AND exit
		// cache offsets, on the fake's deterministic default functions.
		for prompt in [&[1u32, 5, 8][..], &[1u32], &[7u32, 5, 10, 8, 9]] {
			for max_tokens in [1usize, 2, 7, 16, 31] {
				for k in [1usize, 2, 4, 8] {
					let run = |spec: bool| {
						let (ops, frontier) = FakeOps::new(prompt, V, false, spec);
						let cbs = Rc::new(RefCell::new(Vec::new()));
						let cfg = Cfg {
							max_tokens,
							spec_k: spec.then_some(k),
							..Cfg::default()
						};
						let mut d = driver(
							ops,
							if spec { frontier } else { None },
							greedy(),
							&cfg,
							cbs.clone(),
						);
						run_all(&mut d, 6, max_tokens).unwrap();
						let emitted = cbs.borrow().clone();
						(emitted, d.committed_len(), d.ops().t_fed.clone())
					};
					let (off_emitted, off_committed, off_fed) = run(false);
					let (on_emitted, on_committed, on_fed) = run(true);
					assert_eq!(
						off_emitted, on_emitted,
						"emitted diverged (prompt {prompt:?}, max {max_tokens}, k {k})"
					);
					assert_eq!(off_committed, on_committed);
					assert_eq!(off_fed, on_fed, "exit cache offsets diverged");
				}
			}
		}
	}

	#[test]
	fn stats_accounting_is_consistent() {
		let (ops, frontier) = FakeOps::new(PROMPT, V, false, true);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg {
			max_tokens: 24,
			spec_k: Some(3),
			..Cfg::default()
		};
		let mut d = driver(ops, frontier, greedy(), &cfg, cbs.clone());
		run_all(&mut d, 6, 24).unwrap();
		let stats = d.stats().clone();
		assert!(stats.rounds > 0);
		// One-based depth buckets: rounds accepting >= 1 draft record one
		// bucket each, full rejections record none - so the bucket sum
		// never exceeds the round count.
		assert!(stats.accepted_by_depth.iter().sum::<u64>() <= stats.rounds);
		// drafted == number of draft steps taken (no faults in this run).
		let steps = d
			.ops()
			.log
			.iter()
			.filter(|e| matches!(e, Ev::MtpStep(_)))
			.count();
		assert_eq!(stats.drafted, steps as u64);
		// completion_tokens source: every emitted token had exactly one
		// callback, and the committed ledger is a prefix of it.
		assert!(d.committed_len() <= cbs.borrow().len());
	}

	#[test]
	fn greedy_rounds_consume_zero_sampler_rng() {
		// Decision-only verification corollary: greedy traces never touch
		// the sampler's fallible surface at all (drafts, verification,
		// and the stage-4 successor are all argmax data) - so an armed
		// one-shot sampler fault can never fire across a whole greedy
		// spec-on decode.
		let (ops, frontier) = FakeOps::new(PROMPT, V, false, true);
		let cbs = Rc::new(RefCell::new(Vec::new()));
		let cfg = Cfg {
			max_tokens: 16,
			spec_k: Some(3),
			..Cfg::default()
		};
		let sampler = greedy();
		sampler.inject_failure_at(0);
		let mut d = driver(ops, frontier, sampler, &cfg, cbs.clone());
		run_all(&mut d, 6, 16).unwrap();
		assert!(d.stats().rounds > 0, "speculation ran without RNG");
	}

	fn spread_dist(seq: &[u32], vocab: usize, salt: u64) -> Vec<f32> {
		let values: Vec<f64> = (0..vocab)
			.map(|i| {
				let mut key = seq.to_vec();
				key.push(i as u32);
				f64::from(default_tok(&key, vocab, salt)) + 1.0
			})
			.collect();
		let total: f64 = values.iter().sum();
		values.iter().map(|v| (v / total) as f32).collect()
	}

	/// Sampled acceptance depths 0..=k and every rejection
	/// point driven EXCLUSIVELY by the non-failing acceptance-uniform
	/// control hook - genuinely spread (non-one-hot) target and draft
	/// distributions, so no scripted agreement decides the depth and
	/// every forced rejection has a non-degenerate residual to
	/// construct. Control-hook assertions: every steered round completes
	/// `Ok` (the hook introduces no error path; it is compiled out of
	/// production builds - the hook field and all consultation sites are
	/// `#[cfg(test)]`, so `cfg(not(test))` contains no hook code).
	#[test]
	fn acceptance_depths_driven_by_control_hook() {
		let k = 3usize;
		// Genuinely varied (non-one-hot, non-uniform) dists with the EOS
		// row zeroed (renormalized) so a drafted token can never end the
		// round early via Emit::Eos. `spread_dist` is unusable here: its
		// per-token index is added AFTER the hash's last multiply, so it
		// never reaches the extracted high bits and the row degenerates to
		// uniform (p == q, zero residual). This variant pushes a trailing
		// mixer element so the index lands in the high bits.
		fn varied_dist(seq: &[u32], vocab: usize, salt: u64) -> Vec<f32> {
			let mut values: Vec<f64> = (0..vocab)
				.map(|i| {
					let mut key = seq.to_vec();
					key.push(i as u32);
					key.push(7); // extra mix round so `i` reaches the high bits
					f64::from(default_tok(&key, vocab, salt)) + 1.0
				})
				.collect();
			values[EOS as usize] = 0.0;
			let total: f64 = values.iter().sum();
			values.iter_mut().for_each(|v| *v /= total);
			values.into_iter().map(|v| v as f32).collect()
		}
		for a in 0..=k {
			let (mut ops, frontier) = FakeOps::new(PROMPT, V, false, true);
			ops.target_dist = Some(Box::new(move |seq| varied_dist(seq, V, 41)));
			ops.draft_dist = Some(Box::new(move |seq| varied_dist(seq, V, 43)));
			let cbs = Rc::new(RefCell::new(Vec::new()));
			let cfg = Cfg {
				spec_k: Some(k),
				..Cfg::default()
			};
			let sampler = sampled(9000 + a as u64);
			// Force `a` accepts then (below full depth) one reject.
			let mut decisions = vec![true; a];
			if a < k {
				decisions.push(false);
			}
			sampler.force_acceptance_decisions(decisions);
			let mut d = driver(ops, frontier, sampler, &cfg, cbs.clone());
			let end = d.run_round(6).expect("control hook adds no error path");
			assert!(
				matches!(end, RoundEnd::Continue { .. }),
				"forced depth {a} must complete the round"
			);
			// Retained prefix = x plus exactly the forced-accepted drafts.
			assert_eq!(d.committed_len(), 1 + a, "forced depth {a}");
			assert_eq!(cbs.borrow().len(), 1 + a);
			assert_eq!(d.ops().decode_suffix().len(), 1 + a);
			// Stats: drafted at draft time (k proposals), one decided
			// round, one-based depth bucket.
			assert_eq!(d.stats().drafted, k as u64);
			assert_eq!(d.stats().rounds, 1);
			if a == 0 {
				assert_eq!(d.stats().accepted_by_depth.iter().sum::<u64>(), 0);
			} else {
				assert_eq!(d.stats().accepted_by_depth[a - 1], 1);
			}
			// MTP pairs follow the retained prefix (entry pairs + x + a).
			assert_eq!(d.ops().pairs_fed(), PROMPT.len() - 1 + 1 + a);
			// Stage-4 successor drawn exactly once, from the verdict.
			assert_eq!(d.ops().count(&Ev::Successor), 1);
		}
	}

	#[test]
	fn same_seed_same_output_within_spec_on() {
		// Sampled spec-on determinism: identical seeds produce identical
		// emitted sequences, committed ledgers, and cache offsets (RNG
		// order pinned: drafts → acceptance uniforms → successor last).
		let run = |seed: u64| {
			let (mut ops, frontier) = FakeOps::new(PROMPT, V, false, true);
			ops.target_dist = Some(Box::new(|seq| spread_dist(seq, V, 41)));
			ops.draft_dist = Some(Box::new(|seq| spread_dist(seq, V, 43)));
			let cbs = Rc::new(RefCell::new(Vec::new()));
			let cfg = Cfg {
				max_tokens: 20,
				spec_k: Some(2),
				..Cfg::default()
			};
			let mut d = driver(ops, frontier, sampled(seed), &cfg, cbs.clone());
			run_all(&mut d, 6, 20).unwrap();
			(
				cbs.borrow().clone(),
				d.committed_len(),
				d.ops().t_fed.clone(),
				d.ops().pairs_fed(),
				d.stats().clone(),
			)
		};
		assert_eq!(run(1234), run(1234));
		assert_eq!(run(77), run(77));
	}
}
