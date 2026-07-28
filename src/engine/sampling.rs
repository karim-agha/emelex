//! Token sampling strategies applied to the final-step logits.

use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::engine::{array::Array, error::Result, ops};

/// Sampling configuration for one generation call.
#[derive(Debug, Clone, Copy)]
pub struct SamplingConfig {
	pub temperature: f32,
	pub top_p: f32,
	pub top_k: Option<i32>,
	pub seed: Option<u64>,
}

impl Default for SamplingConfig {
	fn default() -> Self {
		SamplingConfig {
			temperature: 0.0,
			top_p: 1.0,
			top_k: None,
			seed: None,
		}
	}
}

/// Stateful sampler holding the RNG across a generation session.
pub struct Sampler {
	config: SamplingConfig,
	rng: StdRng,
	/// emelex patch (not upstream): fault-injection hook for the decode
	/// loop's successor-sampling failure rows - real MLX faults cannot be
	/// phase-targeted reliably, so tests arm this one-shot trigger instead.
	/// `Cell` because `probs` takes `&self`.
	#[cfg(test)]
	fail_next: std::cell::Cell<bool>,
	/// emelex patch (not upstream): counted fault injection - `Some(n)`
	/// fails the (n+1)-th subsequent fallible sampler call, so a test
	/// driving a full decode round can target one specific draw site
	/// (draft draw, residual/bonus successor draw, recovery/row-i draw)
	/// by its call index. The acceptance-uniform phase is not a fallible
	/// call site: `verify_accept` never consults this hook.
	#[cfg(test)]
	fail_at_call: std::cell::Cell<Option<usize>>,
	/// emelex patch (not upstream): acceptance-uniform control
	/// hook - a queue of forced per-depth accept/reject decisions consumed
	/// by [`Sampler::verify_accept`]'s uniform phase. A control hook, not
	/// a failure hook: its API yields a decision VALUE (`bool`), so it
	/// cannot introduce an error path by construction, matching the
	/// contract that the acceptance-uniform phase is infallible. The
	/// uniform is still drawn (RNG consumption is unchanged); only the
	/// comparison outcome is overridden. Compiled out of production
	/// builds: the field and every consultation site are `#[cfg(test)]`,
	/// so `cfg(not(test))` builds contain no trace of the hook.
	#[cfg(test)]
	accept_control: std::cell::RefCell<std::collections::VecDeque<bool>>,
}

impl Sampler {
	pub fn new(config: SamplingConfig) -> Self {
		let rng = match config.seed {
			Some(seed) => StdRng::seed_from_u64(seed),
			None => StdRng::from_entropy(),
		};
		Sampler {
			config,
			rng,
			#[cfg(test)]
			fail_next: std::cell::Cell::new(false),
			#[cfg(test)]
			fail_at_call: std::cell::Cell::new(None),
			#[cfg(test)]
			accept_control: std::cell::RefCell::new(std::collections::VecDeque::new()),
		}
	}

	/// emelex patch (not upstream): arm the one-shot sampling fault. The
	/// next call to `sample`, `probs`, or `sample_from_probs` fails.
	/// `verify_accept` is NOT a fault site: its uniform phase is
	/// contractually infallible and is steered by the
	/// non-failing control hook instead.
	#[cfg(test)]
	pub fn inject_failure(&self) {
		self.fail_next.set(true);
	}

	/// emelex patch (not upstream): arm the counted fault - the
	/// `(nth + 1)`-th subsequent fallible sampler call fails (`0` behaves
	/// like [`Sampler::inject_failure`]).
	#[cfg(test)]
	pub fn inject_failure_at(&self, nth: usize) {
		self.fail_at_call.set(Some(nth));
	}

	/// emelex patch (not upstream): queue forced acceptance decisions for
	/// [`Sampler::verify_accept`]'s uniform phase. Each queued `bool` overrides
	/// one depth's accept/reject
	/// outcome, front first (`true` = accept); an empty queue falls back
	/// to the real uniform comparison. Decision-valued by construction -
	/// this API cannot fail and adds no `Err` source to the phase.
	#[cfg(test)]
	pub fn force_acceptance_decisions(&self, decisions: impl IntoIterator<Item = bool>) {
		self.accept_control.borrow_mut().extend(decisions);
	}

	#[cfg(test)]
	fn forced_acceptance(&self) -> Option<bool> {
		self.accept_control.borrow_mut().pop_front()
	}

	#[cfg(test)]
	fn take_injected_failure(&self) -> Result<()> {
		if self.fail_next.replace(false) {
			return Err(spec_err(String::from("sampler test fault")));
		}
		if let Some(n) = self.fail_at_call.get() {
			if n == 0 {
				self.fail_at_call.set(None);
				return Err(spec_err(String::from("sampler counted test fault")));
			}
			self.fail_at_call.set(Some(n - 1));
		}
		Ok(())
	}

	/// Sample one token id from `logits` (shape `[vocab]`, last-step only).
	pub fn sample(&mut self, logits: &Array) -> Result<u32> {
		#[cfg(test)]
		self.take_injected_failure()?;
		if self.config.temperature <= 0.0 {
			let idx = ops::argmax_axis(logits, -1, false)?;
			return idx.item_u32();
		}

		let scaled = ops::scale_by(logits, 1.0 / self.config.temperature)?;
		let mut probs = softmax_to_vec(&scaled)?;
		validate_probability_mass(&probs, "sample()", "softmax")?;

		// emelex patch: a non-positive top_k must be a no-op, not an
		// `as usize` wrap into a huge cutoff.
		if let Some(k) = self.config.top_k {
			if k > 0 {
				top_k_filter(&mut probs, k as usize);
			}
		}
		if self.config.top_p < 1.0 {
			top_p_filter(&mut probs, self.config.top_p);
		}

		let total: f32 = probs.iter().sum();
		if !total.is_finite() || total <= 0.0 {
			return Err(spec_err(format!(
				"sample() invariant violated: post-filter mass is {total}, expected positive \
				 and finite"
			)));
		}
		let mut draw = self.rng.r#gen::<f32>() * total;
		for (i, p) in probs.iter().enumerate() {
			draw -= p;
			// emelex patch: skip zero-mass entries on an exact-zero draw
			// (probability 2^-24 per token) - without the guard, a token
			// that top-k/top-p explicitly filtered out could be emitted,
			// and the spec-off path would diverge from the staged
			// probs()+sample_from_probs() pipeline at that edge. The only
			// behavioral change is on that buggy edge; RNG consumption is
			// unchanged.
			if *p > 0.0 && draw <= 0.0 {
				return Ok(i as u32);
			}
		}
		// emelex patch: numeric fallthrough (rounding on the final
		// iteration) must return the most probable token, not whatever
		// happens to sit at the last vocab index (often zero-probability
		// after filtering).
		let best = probs
			.iter()
			.enumerate()
			.max_by(|a, b| a.1.total_cmp(b.1))
			.map_or(0, |(i, _)| i);
		Ok(best as u32)
	}
}

fn softmax_to_vec(logits: &Array) -> Result<Vec<f32>> {
	let probs = ops::softmax_axis(logits, -1, true)?;
	probs.to_vec_f32()
}

fn validate_probability_mass(probs: &[f32], caller: &str, stage: &str) -> Result<f64> {
	let mut total = 0.0_f64;
	for (index, &probability) in probs.iter().enumerate() {
		if !probability.is_finite() || probability < 0.0 {
			return Err(spec_err(format!(
				"{caller} produced invalid {stage} probability {probability} at index {index}"
			)));
		}
		total += f64::from(probability);
	}
	if !total.is_finite() || total <= 0.0 {
		return Err(spec_err(format!(
			"{caller} invariant violated: {stage} mass is {total}, expected positive and finite"
		)));
	}
	Ok(total)
}

// emelex patch: O(n) selection instead of a full-vocab sort + HashSet,
// and `total_cmp` instead of `partial_cmp().unwrap()` (a NaN logit must
// not panic the generation thread).
fn top_k_filter(probs: &mut [f32], k: usize) {
	if k == 0 || k >= probs.len() {
		return;
	}
	let mut values: Vec<f32> = probs.to_vec();
	let (_, kth, _) = values.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
	let threshold = *kth;
	for p in probs.iter_mut() {
		if *p < threshold {
			*p = 0.0;
		}
	}
}

fn top_p_filter(probs: &mut [f32], top_p: f32) {
	let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
	// emelex patch: total_cmp - NaN must not panic the generation thread.
	indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
	let total: f32 = probs.iter().sum();
	if total <= 0.0 {
		return;
	}
	let mut cumulative = 0.0;
	let mut cutoff = indexed.len();
	for (rank, (_, p)) in indexed.iter().enumerate() {
		cumulative += p / total;
		if cumulative >= top_p {
			cutoff = rank + 1;
			break;
		}
	}
	let keep: std::collections::HashSet<usize> =
		indexed.iter().take(cutoff).map(|(i, _)| *i).collect();
	for (i, p) in probs.iter_mut().enumerate() {
		if !keep.contains(&i) {
			*p = 0.0;
		}
	}
}

// emelex patch (not upstream): speculative-decoding support. Everything in
// this section is additive — `sample` above keeps its exact arithmetic and
// RNG consumption byte-for-byte. `probs` + `sample_from_probs` expose the
// existing filter→CDF pipeline as reusable halves, and `verify_speculative`
// / `verify_greedy` implement Leviathan/Chen rejection sampling over
// draft-model proposals.

/// Outcome of verifying a run of drafted tokens against the target model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecVerdict {
	/// Number of draft tokens accepted (a prefix of `drafts`).
	pub accepted: usize,
	/// Token emitted after the accepted prefix: resampled from the residual
	/// on rejection, or the bonus token when every draft was accepted.
	pub next: u32,
}

fn spec_err(msg: String) -> crate::engine::error::Error {
	// Runtime invariant violations, not config parsing - `Error::Model`
	// per the crate's variant convention (cache.rs precedent).
	crate::engine::error::Error::Model(format!("sampling: {msg}"))
}

impl Sampler {
	/// True when the configured temperature selects greedy (argmax) decoding.
	pub fn is_greedy(&self) -> bool {
		self.config.temperature <= 0.0
	}

	/// The filtered and renormalized sampling distribution for `logits`
	/// (shape `[vocab]`): scale by `1/T` → softmax → top-k zeroing → top-p
	/// zeroing → renormalize so the vector sums to 1.
	///
	/// Errors at `temperature <= 0` (greedy decoding has no sampling
	/// distribution — callers must branch on [`Sampler::is_greedy`]), on
	/// non-1-D input, and when the post-filter mass is zero or non-finite.
	pub fn probs(&self, logits: &Array) -> Result<Vec<f32>> {
		#[cfg(test)]
		self.take_injected_failure()?;
		if self.is_greedy() {
			return Err(spec_err(String::from(
				"probs() requires temperature > 0; greedy decoding has no sampling \
				 distribution",
			)));
		}
		if logits.ndim() != 1 {
			return Err(spec_err(format!(
				"probs() expects 1-D logits, got shape {:?}",
				logits.shape()
			)));
		}
		let mut probs = self.batched_softmax(logits)?;
		self.filter_and_normalize_probability_row(&mut probs, "probs()")?;
		Ok(probs)
	}

	/// Build unfiltered softmax probabilities for every last-axis row in one
	/// MLX graph and one host read. The returned vector is row-major.
	pub(crate) fn batched_softmax(&self, logits: &Array) -> Result<Vec<f32>> {
		let scaled = ops::scale_by(logits, 1.0 / self.config.temperature)?;
		let probs = ops::softmax_axis(&scaled, -1, true)?;
		probs.to_vec_f32()
	}

	/// Apply configured filters and normalization to one host probability row.
	/// Per-row fault injection preserves salvage coverage while production
	/// callers batch device work.
	pub(crate) fn finish_batched_probability_row(&self, probs: &mut [f32]) -> Result<()> {
		#[cfg(test)]
		self.take_injected_failure()?;
		self.filter_and_normalize_probability_row(probs, "batched probs")
	}

	fn filter_and_normalize_probability_row(&self, probs: &mut [f32], caller: &str) -> Result<()> {
		validate_probability_mass(probs, caller, "softmax")?;
		if let Some(k) = self.config.top_k {
			if k > 0 {
				top_k_filter(probs, k as usize);
			}
		}
		if self.config.top_p < 1.0 {
			top_p_filter(probs, self.config.top_p);
		}
		// emelex patch: accumulate and divide through f64. A sequential
		// f32 sum over a ~250k-entry near-uniform row drops enough mass
		// (measured ~3e-3) to trip verify_speculative's own 1e-3 sum
		// gate, silently defeating speculation on high-entropy stretches;
		// through f64 the aggregate error is per-element rounding
		// (~3e-8), keeping that gate a genuine logic-error tripwire.
		let total = validate_probability_mass(probs, caller, "post-filter")?;
		for p in probs.iter_mut() {
			*p = (f64::from(*p) / total) as f32;
		}
		Ok(())
	}

	/// Draw one token from an explicit, already-normalized distribution
	/// using this sampler's RNG (one uniform draw per call). The CDF walk
	/// is scale-invariant and never returns an index whose probability is
	/// exactly zero. Errors on empty, non-finite, negative, or zero-mass
	/// input.
	pub fn sample_from_probs(&mut self, probs: &[f32]) -> Result<u32> {
		#[cfg(test)]
		self.take_injected_failure()?;
		if probs.is_empty() {
			return Err(spec_err(String::from(
				"sample_from_probs() requires a non-empty distribution",
			)));
		}
		let mut total = 0.0f32;
		for (i, &p) in probs.iter().enumerate() {
			if !p.is_finite() || p < 0.0 {
				return Err(spec_err(format!(
					"sample_from_probs() got invalid probability {p} at index {i}"
				)));
			}
			total += p;
		}
		if !total.is_finite() || total <= 0.0 {
			return Err(spec_err(format!(
				"sample_from_probs() invariant violated: total mass is {total}, \
				 expected positive and finite"
			)));
		}
		let mut draw = self.rng.r#gen::<f32>() * total;
		for (i, &p) in probs.iter().enumerate() {
			draw -= p;
			// `p > 0.0` guards the exact-zero draw: a zero-mass entry must
			// never be emitted even when `draw` starts at 0.
			if p > 0.0 && draw <= 0.0 {
				return Ok(i as u32);
			}
		}
		let best = probs
			.iter()
			.enumerate()
			.max_by(|a, b| a.1.total_cmp(b.1))
			.map_or(0, |(i, _)| i);
		Ok(best as u32)
	}

	/// Leviathan/Chen speculative verification of `drafts` (sampled from
	/// the draft model's `draft_probs`) against the target model's
	/// distributions `target` (one row per draft position plus one bonus
	/// row). Accepts draft `i` with probability `min(1, p_i(d)/q_i(d))`
	/// (computed in f64); on rejection resamples from the normalized
	/// residual `max(0, p_i - q_i)`. The emitted token stream is
	/// distributed exactly as if sampled from the target alone.
	pub fn verify_speculative(
		&mut self,
		drafts: &[u32],
		draft_probs: &[Vec<f32>],
		target: &[Vec<f32>],
	) -> Result<SpecVerdict> {
		let vocab = validate_speculative_inputs(drafts, draft_probs, target, true)?;

		for i in 0..drafts.len() {
			let d = drafts[i] as usize;
			let q_d = f64::from(draft_probs[i][d]);
			let p_d = f64::from(target[i][d]);
			if q_d == 0.0 {
				return Err(spec_err(format!(
					"verify_speculative() invariant violated: draft token {d} at \
					 position {i} has zero draft probability, but drafts must be \
					 sampled from draft_probs"
				)));
			}
			// Exact f64 ratio — a tiny positive q must not trigger an
			// epsilon rejection.
			let accept = (p_d / q_d).min(1.0);
			let u = self.rng.r#gen::<f64>();
			if u < accept {
				continue;
			}
			// Rejected at position i: resample from the normalized
			// residual max(0, p - q), computed elementwise in f64.
			let mut residual64 = vec![0.0f64; vocab];
			let mut mass = 0.0f64;
			for t in 0..vocab {
				let r = (f64::from(target[i][t]) - f64::from(draft_probs[i][t])).max(0.0);
				residual64[t] = r;
				mass += r;
			}
			if mass <= 0.0 {
				return Err(spec_err(format!(
					"verify_speculative() invariant violated: residual mass at position \
					 {i} is zero; refusing to silently sample the target distribution"
				)));
			}
			let residual: Vec<f32> = residual64.iter().map(|r| (r / mass) as f32).collect();
			let next = self.sample_from_probs(&residual)?;
			return Ok(SpecVerdict { accepted: i, next });
		}

		// Every draft accepted: emit the bonus token from the final
		// target row.
		let next = self.sample_from_probs(&target[drafts.len()])?;
		Ok(SpecVerdict {
			accepted: drafts.len(),
			next,
		})
	}

	/// emelex patch (not upstream): decision-only speculative
	/// verification. Identical acceptance arithmetic and RNG consumption to
	/// [`Sampler::verify_speculative`]'s decision
	/// phase - the acceptance uniforms are the ONLY RNG consumed here -
	/// but the successor is never drawn: on rejection at depth `a` the
	/// renormalized residual `max(0, p_a - q_a)` is constructed and
	/// returned; on full acceptance the bonus row `p_k` is returned. The
	/// decode round's stage-4 selection draws from
	/// [`AcceptVerdict::successor_dist`] via
	/// [`Sampler::sample_from_probs`] - and only on continuing
	/// dispositions, so exiting rounds consume no successor RNG.
	///
	/// Uniform-phase infallibility contract: the acceptance
	/// uniforms are pure host `StdRng` f64 draws over already-materialized
	/// probability data - no fallible operation exists inside the phase,
	/// so this function's only `Err` sources are the enumerated invariant
	/// checks (input validation, zero draft probability, all-zero
	/// residual). It deliberately does NOT consult
	/// [`Sampler::take_injected_failure`]: the test hook for this phase is
	/// the decision-valued acceptance-uniform CONTROL hook
	/// ([`Sampler::force_acceptance_decisions`]), which cannot return
	/// `Err` by construction and is compiled out of `cfg(not(test))`
	/// builds.
	pub(crate) fn verify_accept<Q, P>(
		&mut self,
		drafts: &[u32],
		draft_probs: &[Q],
		target: &[P],
	) -> Result<AcceptVerdict>
	where
		Q: AsRef<[f32]>,
		P: AsRef<[f32]>,
	{
		self.verify_accept_inner(drafts, draft_probs, target, true)
	}

	/// Hot-path decision over distributions already validated and normalized by
	/// [`Self::probs`] or [`Self::finish_batched_probability_row`].
	///
	/// Shape and draft-token bounds remain unconditional. Debug builds also
	/// repeat the complete value/mass sweep; optimized builds trust the private
	/// construction boundary and avoid rescanning `(2K + 1) * V` probabilities.
	pub(crate) fn verify_accept_trusted<Q, P>(
		&mut self,
		drafts: &[u32],
		draft_probs: &[Q],
		target: &[P],
	) -> Result<AcceptVerdict>
	where
		Q: AsRef<[f32]>,
		P: AsRef<[f32]>,
	{
		self.verify_accept_inner(drafts, draft_probs, target, cfg!(debug_assertions))
	}

	fn verify_accept_inner<Q, P>(
		&mut self,
		drafts: &[u32],
		draft_probs: &[Q],
		target: &[P],
		validate_values: bool,
	) -> Result<AcceptVerdict>
	where
		Q: AsRef<[f32]>,
		P: AsRef<[f32]>,
	{
		let vocab = validate_speculative_inputs(drafts, draft_probs, target, validate_values)?;

		for i in 0..drafts.len() {
			let d = drafts[i] as usize;
			let draft_row = draft_probs[i].as_ref();
			let target_row = target[i].as_ref();
			let q_d = f64::from(draft_row[d]);
			let p_d = f64::from(target_row[d]);
			if q_d == 0.0 {
				return Err(spec_err(format!(
					"verify_accept() invariant violated: draft token {d} at position \
					 {i} has zero draft probability, but drafts must be sampled from \
					 draft_probs"
				)));
			}
			// Exact f64 ratio - a tiny positive q must not trigger an
			// epsilon rejection.
			let accept = (p_d / q_d).min(1.0);
			// The uniform is ALWAYS drawn (RNG accounting is identical with
			// and without the control hook); the hook only overrides the
			// comparison's outcome with a forced decision value.
			let u = self.rng.r#gen::<f64>();
			let accepted = u < accept;
			#[cfg(test)]
			let accepted = self.forced_acceptance().unwrap_or(accepted);
			if accepted {
				continue;
			}
			// Rejected at position i: CONSTRUCT (never draw) the
			// normalized residual max(0, p - q) in f64.
			let mut residual64 = vec![0.0f64; vocab];
			let mut mass = 0.0f64;
			for t in 0..vocab {
				let r = (f64::from(target_row[t]) - f64::from(draft_row[t])).max(0.0);
				residual64[t] = r;
				mass += r;
			}
			if mass <= 0.0 {
				return Err(spec_err(format!(
					"verify_accept() invariant violated: residual mass at position {i} \
					 is zero; refusing to silently sample the target distribution"
				)));
			}
			let successor_dist: Vec<f32> = residual64.iter().map(|r| (r / mass) as f32).collect();
			return Ok(AcceptVerdict {
				accepted: i,
				successor_dist,
			});
		}

		Ok(AcceptVerdict {
			accepted: drafts.len(),
			successor_dist: target[drafts.len()].as_ref().to_vec(),
		})
	}
}

/// emelex patch (not upstream): outcome of decision-only speculative
/// verification ([`Sampler::verify_accept`]) - the acceptance depth plus
/// a constructed, renormalized, UNDRAWN successor distribution (residual
/// at the rejection depth when `accepted < k`, bonus `p_k` when
/// `accepted == k`). Pure host data that survives all cache mutations.
#[derive(Debug, Clone, PartialEq)]
pub struct AcceptVerdict {
	pub accepted: usize,
	pub successor_dist: Vec<f32>,
}

/// emelex patch (not upstream): the shared input validation of
/// [`Sampler::verify_speculative`] and [`Sampler::verify_accept`],
/// returning the vocab size. The checked elementwise sweep is
/// O((2K+1)*V) over K draft plus K+1 target rows. Checked entry points retain
/// that complete sweep in release. The private trusted hot path may disable
/// value/mass rescanning only after `Sampler::probs` or batched row completion
/// validated every row; shape and token bounds are always checked.
fn validate_speculative_inputs<Q, P>(
	drafts: &[u32],
	draft_probs: &[Q],
	target: &[P],
	validate_values: bool,
) -> Result<usize>
where
	Q: AsRef<[f32]>,
	P: AsRef<[f32]>,
{
	if target.len() != drafts.len() + 1 {
		return Err(spec_err(format!(
			"speculative verification expects target.len() == drafts.len() + 1, got \
			 {} target rows for {} drafts",
			target.len(),
			drafts.len()
		)));
	}
	if draft_probs.len() != drafts.len() {
		return Err(spec_err(format!(
			"speculative verification expects one draft_probs row per draft, got {} \
			 rows for {} drafts",
			draft_probs.len(),
			drafts.len()
		)));
	}
	let vocab = target[0].as_ref().len();
	if vocab == 0 {
		return Err(spec_err(String::from(
			"speculative verification got an empty target distribution",
		)));
	}
	validate_speculative_rows("target", target, vocab, validate_values)?;
	validate_speculative_rows("draft_probs", draft_probs, vocab, validate_values)?;
	for (i, &d) in drafts.iter().enumerate() {
		if d as usize >= vocab {
			return Err(spec_err(format!(
				"speculative verification draft token {d} at position {i} is out of \
				 bounds for vocab {vocab}"
			)));
		}
	}
	Ok(vocab)
}

fn validate_speculative_rows<R>(
	name: &str,
	rows: &[R],
	vocab: usize,
	validate_values: bool,
) -> Result<()>
where
	R: AsRef<[f32]>,
{
	for (i, row) in rows.iter().enumerate() {
		let row = row.as_ref();
		if row.len() != vocab {
			return Err(spec_err(format!(
				"speculative verification {name}[{i}] has length {}, expected {vocab}",
				row.len()
			)));
		}
		if !validate_values {
			continue;
		}
		let mut sum = 0.0f64;
		for (j, &v) in row.iter().enumerate() {
			if !v.is_finite() || v < 0.0 {
				return Err(spec_err(format!(
					"speculative verification {name}[{i}][{j}] is {v}, expected finite \
					 and non-negative"
				)));
			}
			sum += f64::from(v);
		}
		if (sum - 1.0).abs() > 1e-3 {
			return Err(spec_err(format!(
				"speculative verification {name}[{i}] sums to {sum}, expected ~1 \
				 (tolerance 1e-3)"
			)));
		}
	}
	Ok(())
}

/// Greedy-mode speculative verification: no RNG, no softmax — walk the
/// drafted prefix against the target model's per-position argmax and stop
/// at the first mismatch. `target_argmax` carries one entry per draft
/// position plus the bonus token.
pub fn verify_greedy(drafts: &[u32], target_argmax: &[u32]) -> Result<SpecVerdict> {
	if target_argmax.len() != drafts.len() + 1 {
		return Err(spec_err(format!(
			"verify_greedy() expects target_argmax.len() == drafts.len() + 1, got \
			 {} argmax entries for {} drafts",
			target_argmax.len(),
			drafts.len()
		)));
	}
	for (i, &d) in drafts.iter().enumerate() {
		if d != target_argmax[i] {
			return Ok(SpecVerdict {
				accepted: i,
				next: target_argmax[i],
			});
		}
	}
	Ok(SpecVerdict {
		accepted: drafts.len(),
		next: target_argmax[drafts.len()],
	})
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn greedy_sampling_picks_argmax() {
		let logits = Array::from_slice(&[0.1f32, 0.9, 0.3, -0.2], &[4]).unwrap();
		let mut sampler = Sampler::new(SamplingConfig {
			temperature: 0.0,
			..Default::default()
		});
		assert_eq!(sampler.sample(&logits).unwrap(), 1);
	}

	#[test]
	fn sampled_and_speculative_paths_reject_non_finite_logits() {
		for invalid in [f32::NAN, f32::INFINITY] {
			let logits = Array::from_slice(&[0.0_f32, invalid], &[2]).unwrap();
			let config = SamplingConfig {
				temperature: 0.7,
				..SamplingConfig::default()
			};
			assert!(Sampler::new(config).sample(&logits).is_err());
			assert!(Sampler::new(config).probs(&logits).is_err());
		}
	}

	#[test]
	fn greedy_sampling_is_deterministic_across_calls() {
		let logits = Array::from_slice(&[1.0f32, 5.0, 2.0], &[3]).unwrap();
		let mut a = Sampler::new(SamplingConfig::default());
		let mut b = Sampler::new(SamplingConfig::default());
		assert_eq!(a.sample(&logits).unwrap(), b.sample(&logits).unwrap());
	}

	#[test]
	fn seeded_sampling_is_reproducible() {
		let logits = Array::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[4]).unwrap();
		let cfg = SamplingConfig {
			temperature: 1.0,
			top_p: 1.0,
			top_k: None,
			seed: Some(42),
		};
		let mut a = Sampler::new(cfg);
		let mut b = Sampler::new(cfg);
		let seq_a: Vec<u32> = (0..10).map(|_| a.sample(&logits).unwrap()).collect();
		let seq_b: Vec<u32> = (0..10).map(|_| b.sample(&logits).unwrap()).collect();
		assert_eq!(seq_a, seq_b);
	}

	#[test]
	fn top_k_filter_keeps_only_k_largest() {
		let mut probs = vec![0.1, 0.4, 0.2, 0.3];
		top_k_filter(&mut probs, 2);
		let nonzero: Vec<usize> = probs
			.iter()
			.enumerate()
			.filter(|&(_, &p)| p > 0.0)
			.map(|(i, _)| i)
			.collect();
		assert_eq!(nonzero, vec![1, 3]);
	}

	#[test]
	fn top_k_filter_noop_when_k_covers_all() {
		let mut probs = vec![0.1, 0.4, 0.2, 0.3];
		let original = probs.clone();
		top_k_filter(&mut probs, 10);
		assert_eq!(probs, original);
	}

	#[test]
	fn top_k_filter_noop_when_k_is_zero() {
		let mut probs = vec![0.1, 0.4, 0.2, 0.3];
		let original = probs.clone();
		top_k_filter(&mut probs, 0);
		assert_eq!(probs, original);
	}

	#[test]
	fn top_p_filter_keeps_smallest_prefix_reaching_mass() {
		let mut probs = vec![0.5, 0.3, 0.15, 0.05];
		top_p_filter(&mut probs, 0.8);
		let nonzero: Vec<usize> = probs
			.iter()
			.enumerate()
			.filter(|&(_, &p)| p > 0.0)
			.map(|(i, _)| i)
			.collect();
		// 0.5 + 0.3 = 0.8 >= 0.8 cutoff, so only the top 2 survive.
		assert_eq!(nonzero, vec![0, 1]);
	}

	#[test]
	fn top_p_filter_keeps_everything_at_top_p_one() {
		let mut probs = vec![0.5, 0.3, 0.15, 0.05];
		let original = probs.clone();
		top_p_filter(&mut probs, 1.0);
		assert_eq!(probs, original);
	}

	#[test]
	fn sampling_config_default_is_greedy() {
		let cfg = SamplingConfig::default();
		assert_eq!(cfg.temperature, 0.0);
		assert_eq!(cfg.top_p, 1.0);
		assert!(cfg.top_k.is_none());
	}

	// emelex patch (not upstream): tests for the speculative-decoding
	// additions. Everything above this line is byte-identical to the
	// pre-patch test suite.

	fn spec_cfg(seed: u64) -> SamplingConfig {
		SamplingConfig {
			temperature: 1.0,
			top_p: 1.0,
			top_k: None,
			seed: Some(seed),
		}
	}

	#[test]
	fn is_greedy_follows_temperature() {
		assert!(Sampler::new(SamplingConfig::default()).is_greedy());
		assert!(
			Sampler::new(SamplingConfig {
				temperature: -1.0,
				..Default::default()
			})
			.is_greedy()
		);
		assert!(!Sampler::new(spec_cfg(0)).is_greedy());
	}

	#[test]
	fn verify_greedy_matching_prefix_resamples_on_mismatch() {
		let v = verify_greedy(&[3, 5, 7], &[3, 5, 9, 1]).unwrap();
		assert_eq!(v.accepted, 2);
		assert_eq!(v.next, 9);
	}

	#[test]
	fn verify_greedy_empty_drafts_returns_first_target() {
		let v = verify_greedy(&[], &[4]).unwrap();
		assert_eq!(v.accepted, 0);
		assert_eq!(v.next, 4);
	}

	#[test]
	fn verify_greedy_full_accept_takes_bonus_token() {
		let v = verify_greedy(&[1, 2], &[1, 2, 3]).unwrap();
		assert_eq!(v.accepted, 2);
		assert_eq!(v.next, 3);
	}

	#[test]
	fn verify_greedy_length_mismatch_errors() {
		assert!(verify_greedy(&[1, 2], &[1, 2]).is_err());
		assert!(verify_greedy(&[1], &[1, 2, 3]).is_err());
	}

	/// Toy distributions shared by the statistical tests: a 6-token vocab
	/// with a deliberately different draft (q) and target (p) at position
	/// 0, plus a uniform bonus row.
	fn toy_p_q_bonus() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
		let q = vec![0.30, 0.20, 0.20, 0.10, 0.10, 0.10];
		let p = vec![0.10, 0.30, 0.20, 0.05, 0.25, 0.10];
		let bonus = vec![1.0 / 6.0; 6];
		(p, q, bonus)
	}

	#[test]
	fn verify_speculative_preserves_target_distribution() {
		const TRIALS: usize = 100_000;
		let (p, q, bonus) = toy_p_q_bonus();
		let draft_probs = vec![q.clone()];
		let target = vec![p.clone(), bonus];
		let mut spec = Sampler::new(spec_cfg(2024));
		let mut direct = Sampler::new(spec_cfg(4048));
		let mut spec_counts = [0usize; 6];
		let mut direct_counts = [0usize; 6];
		for _ in 0..TRIALS {
			let d = spec.sample_from_probs(&q).unwrap();
			let v = spec
				.verify_speculative(&[d], &draft_probs, &target)
				.unwrap();
			let token = if v.accepted == 1 { d } else { v.next };
			spec_counts[token as usize] += 1;
			let t = direct.sample_from_probs(&p).unwrap();
			direct_counts[t as usize] += 1;
		}
		for t in 0..6 {
			let sf = spec_counts[t] as f64 / TRIALS as f64;
			let df = direct_counts[t] as f64 / TRIALS as f64;
			assert!(
				(sf - df).abs() < 0.01,
				"token {t}: speculative freq {sf:.4} vs direct freq {df:.4}"
			);
		}
	}

	// Per-branch tolerance is 0.015 (vs the marginal test's 0.01): each
	// branch sees only its conditional share of the 100k trials, so the
	// worst-case standard error grows by ~sqrt(1/branch_share); 0.015
	// keeps the assertion at the same ~6-sigma strength.
	#[test]
	fn verify_speculative_branch_conditionals_match_analytic() {
		const TRIALS: usize = 100_000;
		let (p, q, bonus) = toy_p_q_bonus();
		let draft_probs = vec![q.clone()];
		let target = vec![p.clone(), bonus];
		let mut spec = Sampler::new(spec_cfg(7777));
		let mut accept_counts = [0usize; 6];
		let mut reject_counts = [0usize; 6];
		for _ in 0..TRIALS {
			let d = spec.sample_from_probs(&q).unwrap();
			let v = spec
				.verify_speculative(&[d], &draft_probs, &target)
				.unwrap();
			if v.accepted == 1 {
				accept_counts[d as usize] += 1;
			} else {
				reject_counts[v.next as usize] += 1;
			}
		}
		// Analytic conditionals: accept-path joint prob per token is
		// min(1, p/q) * q = min(p, q); reject-path outcomes follow the
		// normalized residual max(0, p - q).
		let accept_joint: Vec<f64> = (0..6)
			.map(|t| f64::from(p[t]).min(f64::from(q[t])))
			.collect();
		let accept_mass: f64 = accept_joint.iter().sum();
		let residual: Vec<f64> = (0..6)
			.map(|t| (f64::from(p[t]) - f64::from(q[t])).max(0.0))
			.collect();
		let residual_mass: f64 = residual.iter().sum();
		let n_accept: usize = accept_counts.iter().sum();
		let n_reject: usize = reject_counts.iter().sum();
		assert!(n_accept > 0 && n_reject > 0);
		for t in 0..6 {
			let emp_a = accept_counts[t] as f64 / n_accept as f64;
			let ana_a = accept_joint[t] / accept_mass;
			assert!(
				(emp_a - ana_a).abs() < 0.015,
				"accept path token {t}: empirical {emp_a:.4} vs analytic {ana_a:.4}"
			);
			let emp_r = reject_counts[t] as f64 / n_reject as f64;
			let ana_r = residual[t] / residual_mass;
			assert!(
				(emp_r - ana_r).abs() < 0.015,
				"reject path token {t}: empirical {emp_r:.4} vs analytic {ana_r:.4}"
			);
		}
	}

	#[test]
	fn probs_returns_renormalized_filtered_distribution() {
		let logits = Array::from_slice(&[2.0f32, 1.0, 0.5, 0.1, -1.0, -2.0], &[6]).unwrap();
		let sampler = Sampler::new(SamplingConfig {
			temperature: 0.8,
			top_p: 0.9,
			top_k: Some(4),
			seed: Some(7),
		});
		let probs = sampler.probs(&logits).unwrap();
		assert_eq!(probs.len(), 6);
		let total: f32 = probs.iter().sum();
		assert!((total - 1.0).abs() < 1e-5, "probs sum to {total}");
		// top_k = 4 must zero the two smallest logits outright.
		assert_eq!(probs[4], 0.0);
		assert_eq!(probs[5], 0.0);
		let nonzero = probs.iter().filter(|&&p| p > 0.0).count();
		assert!(nonzero <= 4);
	}

	#[test]
	fn probs_errors_for_greedy_temperature() {
		let logits = Array::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
		let sampler = Sampler::new(SamplingConfig::default());
		assert!(sampler.probs(&logits).is_err());
	}

	#[test]
	fn probs_errors_for_non_1d_logits() {
		let logits = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
		let sampler = Sampler::new(spec_cfg(0));
		assert!(sampler.probs(&logits).is_err());
	}

	#[test]
	fn probs_then_sample_from_probs_matches_sample_with_same_seed() {
		let logits = Array::from_slice(&[1.5f32, 0.3, -0.7, 2.2, 0.0, -1.1], &[6]).unwrap();
		let cfg = SamplingConfig {
			temperature: 0.9,
			top_p: 0.95,
			top_k: Some(5),
			seed: Some(1337),
		};
		let mut direct = Sampler::new(cfg);
		let mut staged = Sampler::new(cfg);
		for step in 0..32 {
			let expected = direct.sample(&logits).unwrap();
			let probs = staged.probs(&logits).unwrap();
			let actual = staged.sample_from_probs(&probs).unwrap();
			assert_eq!(expected, actual, "diverged at step {step}");
		}
	}

	#[test]
	fn verify_speculative_zero_draft_prob_is_invariant_error() {
		let mut s = Sampler::new(spec_cfg(1));
		let err = s
			.verify_speculative(&[0], &[vec![0.0, 1.0]], &[vec![0.5, 0.5], vec![0.5, 0.5]])
			.unwrap_err();
		assert!(err.to_string().contains("zero draft probability"));
	}

	#[test]
	fn verify_speculative_all_zero_residual_is_invariant_error() {
		// p strictly below q at the drafted token (p = 0 forces a certain
		// rejection) and p == q elsewhere, keeping both sums within the
		// 1e-3 validation tolerance: the residual max(0, p - q) is
		// all-zero, which must surface as an invariant error rather than
		// a silent fallback sample.
		let mut s = Sampler::new(spec_cfg(2));
		let q = vec![0.0012f32, 0.9993];
		let p = vec![0.0f32, 0.9993];
		let bonus = vec![0.5f32, 0.5];
		let err = s.verify_speculative(&[0], &[q], &[p, bonus]).unwrap_err();
		assert!(err.to_string().contains("residual"));
	}

	#[test]
	fn sample_from_probs_never_returns_zero_mass_index() {
		let mut s = Sampler::new(spec_cfg(9));
		let probs = vec![0.0f32, 1.0];
		for _ in 0..10_000 {
			assert_eq!(s.sample_from_probs(&probs).unwrap(), 1);
		}
	}

	#[test]
	fn sample_from_probs_rejects_invalid_input() {
		let mut s = Sampler::new(spec_cfg(3));
		assert!(s.sample_from_probs(&[]).is_err());
		assert!(s.sample_from_probs(&[0.5, -0.5]).is_err());
		assert!(s.sample_from_probs(&[f32::NAN, 1.0]).is_err());
		assert!(s.sample_from_probs(&[f32::INFINITY, 1.0]).is_err());
		assert!(s.sample_from_probs(&[0.0, 0.0]).is_err());
	}

	#[test]
	fn verify_speculative_validates_shapes_and_values() {
		let mut s = Sampler::new(spec_cfg(4));
		let ok = vec![0.5f32, 0.5];
		// target.len() != drafts.len() + 1
		assert!(
			s.verify_speculative(&[0], &[ok.clone()], &[ok.clone()])
				.is_err()
		);
		// draft_probs.len() != drafts.len()
		assert!(
			s.verify_speculative(&[0], &[], &[ok.clone(), ok.clone()])
				.is_err()
		);
		// row length mismatch
		assert!(
			s.verify_speculative(&[0], &[vec![1.0]], &[ok.clone(), ok.clone()])
				.is_err()
		);
		// row does not sum to ~1
		assert!(
			s.verify_speculative(&[0], &[vec![0.4, 0.4]], &[ok.clone(), ok.clone()])
				.is_err()
		);
		// negative probability
		assert!(
			s.verify_speculative(&[0], &[vec![1.5, -0.5]], &[ok.clone(), ok.clone()])
				.is_err()
		);
		assert!(
			s.verify_accept(&[0], &[vec![0.4, 0.4]], &[ok.clone(), ok.clone()])
				.is_err()
		);
		assert!(
			s.verify_accept(&[0], &[vec![1.5, -0.5]], &[ok.clone(), ok.clone()])
				.is_err()
		);
		// draft token out of bounds
		assert!(
			s.verify_speculative(&[2], &[ok.clone()], &[ok.clone(), ok.clone()])
				.is_err()
		);
	}

	#[test]
	fn sample_seeded_sequence_regression() {
		// Pinned pre-patch output of `sample` for a fixed seed and logits:
		// guards that the speculative additions left `sample`'s arithmetic
		// and RNG consumption untouched.
		let logits = Array::from_slice(&[1.0f32, 0.5, 0.25, 2.0], &[4]).unwrap();
		let cfg = SamplingConfig {
			temperature: 0.7,
			top_p: 1.0,
			top_k: None,
			seed: Some(42),
		};
		let mut s = Sampler::new(cfg);
		let seq: Vec<u32> = (0..12).map(|_| s.sample(&logits).unwrap()).collect();
		let expected: Vec<u32> = vec![0, 3, 2, 3, 3, 3, 3, 3, 3, 0, 3, 3];
		assert_eq!(seq, expected);
	}

	/// emelex patch: decision-only verification must be RNG-equivalent to
	/// the drawing form - verify_accept consumes exactly the acceptance
	/// uniforms, so a subsequent sample_from_probs on the returned
	/// successor distribution reproduces verify_speculative's successor
	/// draw for the same seed, at every acceptance outcome.
	#[test]
	fn verify_accept_then_draw_matches_verify_speculative() {
		let (p, q, bonus) = toy_p_q_bonus();
		for seed in 0..200u64 {
			let mut drawing = Sampler::new(spec_cfg(seed));
			let mut deciding = Sampler::new(spec_cfg(seed));
			let d1 = drawing.sample_from_probs(&q).unwrap();
			let d2 = deciding.sample_from_probs(&q).unwrap();
			assert_eq!(d1, d2);
			let target = vec![p.clone(), bonus.clone()];
			let full = drawing
				.verify_speculative(&[d1], &[q.clone()], &target)
				.unwrap();
			let verdict = deciding
				.verify_accept(&[d2], &[q.clone()], &target)
				.unwrap();
			assert_eq!(verdict.accepted, full.accepted, "seed {seed}");
			let successor = deciding.sample_from_probs(&verdict.successor_dist).unwrap();
			assert_eq!(successor, full.next, "seed {seed}");
		}
	}

	/// emelex patch: the constructed successor distribution is normalized
	/// on both branches (residual on rejection, bonus on full acceptance).
	#[test]
	fn verify_accept_successor_dist_sums_to_one() {
		let (p, q, bonus) = toy_p_q_bonus();
		let target = vec![p.clone(), bonus.clone()];
		let mut residual_seen = false;
		let mut bonus_seen = false;
		for seed in 0..200u64 {
			let mut s = Sampler::new(spec_cfg(seed));
			let d = s.sample_from_probs(&q).unwrap();
			let v = s.verify_accept(&[d], &[q.clone()], &target).unwrap();
			let sum: f64 = v.successor_dist.iter().map(|&x| f64::from(x)).sum();
			assert!((sum - 1.0).abs() < 1e-3, "sum {sum}");
			if v.accepted == 0 {
				residual_seen = true;
				// The residual zeroes every token where q >= p.
				assert_eq!(v.successor_dist[0], 0.0);
			} else {
				bonus_seen = true;
				assert_eq!(v.successor_dist, bonus);
			}
		}
		assert!(residual_seen && bonus_seen);
	}

	/// emelex patch: verify_accept surfaces the same invariant errors as
	/// verify_speculative - zero draft probability and all-zero residual.
	#[test]
	fn verify_accept_invariant_errors_mirror_verify_speculative() {
		let mut s = Sampler::new(spec_cfg(1));
		let err = s
			.verify_accept(&[0], &[vec![0.0, 1.0]], &[vec![0.5, 0.5], vec![0.5, 0.5]])
			.unwrap_err();
		assert!(err.to_string().contains("zero draft probability"));

		let mut s = Sampler::new(spec_cfg(2));
		let q = vec![0.0012f32, 0.9993];
		let p = vec![0.0f32, 0.9993];
		let bonus = vec![0.5f32, 0.5];
		let err = s.verify_accept(&[0], &[q], &[p, bonus]).unwrap_err();
		assert!(err.to_string().contains("residual"));

		// Length mismatch.
		let mut s = Sampler::new(spec_cfg(3));
		assert!(
			s.verify_accept(&[0], &[vec![0.5, 0.5]], &[vec![0.5, 0.5]])
				.is_err()
		);
	}

	/// The acceptance-uniform contract test: `verify_accept` cannot
	/// return `Err` from the uniform phase - its only `Err` sources are
	/// the enumerated invariant checks. An armed one-shot AND an armed
	/// counted fault must both pass through `verify_accept` untouched
	/// (it is not a fallible-call site) and fire on the NEXT genuinely
	/// fallible sampler call instead.
	#[test]
	fn verify_accept_uniform_phase_cannot_fail_by_injection() {
		let (p, q, bonus) = toy_p_q_bonus();
		let target = vec![p.clone(), bonus];
		let mut s = Sampler::new(spec_cfg(31));
		let d = s.sample_from_probs(&q).unwrap();
		s.inject_failure();
		s.inject_failure_at(0);
		// The armed faults must not surface here...
		let verdict = s.verify_accept(&[d], &[q.clone()], &target);
		assert!(
			verdict.is_ok(),
			"uniform phase returned Err under fault injection: {verdict:?}"
		);
		// ...and the one-shot fault is still armed for the next fallible
		// call, proving verify_accept never consulted the failure hooks.
		assert!(s.sample_from_probs(&q).is_err());
	}

	/// The acceptance-uniform control hook
	/// forces exact acceptance depths - accept×n then reject - without
	/// introducing any error path (every steered call returns `Ok`), and
	/// RNG consumption is unchanged (the uniform is still drawn per
	/// depth). Production builds compile the hook out entirely: the
	/// field and all consultation sites are `#[cfg(test)]`, so a
	/// `cfg(not(test))` build contains no control-hook code (compile
	/// check by construction - see the field's doc comment).
	#[test]
	fn acceptance_control_hook_forces_depths_without_error_path() {
		// q strictly different from p so forced rejections always have a
		// non-zero residual to construct.
		let (p, q, bonus) = toy_p_q_bonus();
		let k = 3usize;
		let draft_probs = vec![q.clone(); k];
		let mut target = vec![p.clone(); k];
		target.push(bonus.clone());
		for depth in 0..=k {
			let mut s = Sampler::new(spec_cfg(97));
			let drafts: Vec<u32> = (0..k).map(|_| s.sample_from_probs(&q).unwrap()).collect();
			// Force `depth` accepts then (unless fully accepted) a reject.
			let mut decisions = vec![true; depth];
			if depth < k {
				decisions.push(false);
			}
			s.force_acceptance_decisions(decisions);
			let v = s
				.verify_accept(&drafts, &draft_probs, &target)
				.expect("control hook introduces no error path");
			assert_eq!(v.accepted, depth, "forced depth must be exact");
			let sum: f64 = v.successor_dist.iter().map(|&x| f64::from(x)).sum();
			assert!((sum - 1.0).abs() < 1e-3, "successor dist normalized");
			if depth == k {
				assert_eq!(v.successor_dist, bonus, "full acceptance returns p_k");
			} else {
				// Residual at the forced rejection depth zeroes q >= p mass.
				assert_eq!(v.successor_dist[0], 0.0);
			}
		}
	}

	/// RNG accounting is identical with and
	/// without the control hook - forcing decisions changes outcomes,
	/// never the number of uniforms consumed, so a successor draw after
	/// a fully-steered verify_accept matches the unsteered run's RNG
	/// stream position.
	#[test]
	fn acceptance_control_hook_preserves_rng_consumption() {
		let (p, q, bonus) = toy_p_q_bonus();
		let target = vec![p.clone(), p.clone(), bonus.clone()];
		let draft_probs = vec![q.clone(), q.clone()];
		let run = |force: bool| -> u32 {
			let mut s = Sampler::new(spec_cfg(1213));
			let d1 = s.sample_from_probs(&q).unwrap();
			let d2 = s.sample_from_probs(&q).unwrap();
			if force {
				s.force_acceptance_decisions([true, true]);
			}
			let v = s.verify_accept(&[d1, d2], &draft_probs, &target).unwrap();
			// Full acceptance either way for the chosen seed when unforced
			// is NOT required - only the post-verify RNG position matters.
			let _ = v;
			s.sample_from_probs(&bonus).unwrap()
		};
		assert_eq!(run(false), run(true));
	}

	/// emelex patch: the counted fault fires on exactly the armed call.
	#[test]
	fn inject_failure_at_targets_nth_fallible_call() {
		let mut s = Sampler::new(spec_cfg(6));
		let probs = vec![0.5f32, 0.5];
		s.inject_failure_at(2);
		assert!(s.sample_from_probs(&probs).is_ok()); // call 0
		assert!(s.sample_from_probs(&probs).is_ok()); // call 1
		assert!(s.sample_from_probs(&probs).is_err()); // call 2 fires
		assert!(s.sample_from_probs(&probs).is_ok()); // disarmed again
	}

	/// emelex patch regression: a near-uniform ~250k-entry row (the real
	/// Qwen3.5 vocab scale) must renormalize cleanly through `probs()`'s
	/// f64 accumulation and pass `verify_speculative`'s 1e-3 sum gate -
	/// with f32 accumulation the dropped mass measured ~3e-3 and the
	/// sampler rejected its own output, silently defeating speculation
	/// on high-entropy stretches.
	#[test]
	fn probs_renormalization_survives_realistic_vocab() {
		let vocab = 250_000usize;
		// Near-uniform with mild variation so the row is not degenerate.
		let logits_host: Vec<f32> = (0..vocab)
			.map(|i| ((i % 7) as f32).mul_add(1e-3, 1.0))
			.collect();
		let logits = Array::from_slice(&logits_host, &[i32::try_from(vocab).unwrap()]).unwrap();
		let sampler = Sampler::new(SamplingConfig {
			temperature: 1.0,
			top_p: 1.0,
			top_k: None,
			seed: Some(11),
		});
		let p = sampler.probs(&logits).unwrap();
		let sum: f64 = p.iter().map(|&v| f64::from(v)).sum();
		assert!(
			(sum - 1.0).abs() <= 1e-3,
			"renormalized row drifted to {sum}"
		);
		// And the round-trip through verification must accept the row.
		let mut verifier = Sampler::new(SamplingConfig {
			temperature: 1.0,
			top_p: 1.0,
			top_k: None,
			seed: Some(12),
		});
		let verdict = verifier
			.verify_speculative(&[0], &[p.clone()], &[p.clone(), p])
			.unwrap();
		assert_eq!(verdict.accepted, 1);
	}
}
