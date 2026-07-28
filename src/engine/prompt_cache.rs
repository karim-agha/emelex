//! Stateless, server-style prompt-cache pool.
//!
//! Mirrors how the hosted chat APIs actually expose prompt caching: the
//! caller sends the *full* message list on every call (no session handle),
//! and the server transparently keeps a small pool of recently-used
//! prefixes' KV state around so that two calls sharing a prefix (most
//! commonly: the same conversation's next turn, but also just two
//! unrelated calls sharing a system prompt) skip re-running the model over
//! the shared part. OpenAI does this fully automatically; Anthropic adds
//! explicit `cache_control` breakpoint hints - this pool supports the
//! automatic (unpinned, LRU + TTL) case, which is enough to reproduce
//! `Conversation`'s old speedup without requiring a caller-held handle.
//!
//! [`LayerCache`] cloning is a cheap refcount bump per `Array` field (MLX
//! arrays are `shared_ptr`-backed - see `Array::clone` in `array.rs`), so
//! forking a pooled entry's caches into a live generation call, or storing
//! a fresh snapshot back into the pool, costs O(num_layers) rather than
//! O(cache size). That's what makes a multi-entry pool practical here.

use std::time::{Duration, Instant};

use crate::engine::models::{cache::LayerCache, mtp::MtpState};

pub(crate) const DEFAULT_MAX_ENTRIES: usize = 4;
pub(crate) const DEFAULT_MAX_TOTAL_TOKENS: usize = 16_384;

/// One cached prefix: the token ids it represents, the per-layer KV/state
/// caches after processing them, and how much of the multimodal media
/// queue (in placeholder order) has already been fed through the towers
/// and spliced into those caches.
#[derive(Clone)]
pub struct CacheEntry {
	pub ids: Vec<u32>,
	pub caches: Vec<LayerCache>,
	pub fed_images: usize,
	pub fed_audios: usize,
	last_used: Instant,
	/// Exempt from LRU/TTL eviction (mirrors Anthropic's explicit
	/// `cache_control: {type: "ephemeral"}` breakpoints). Not yet wired to
	/// a public cache-hint API - see `ChatMessage`/`generate_cached` TODO.
	pub pinned: bool,
	/// emelex patch (not upstream): pooled MTP snapshot aligned to `ids`.
	/// `Some` iff the producing call speculated and the module survived it;
	/// alignment invariant `mtp.pairs_fed == ids.len() - 1` is enforced at
	/// every insert (see [`aligned_mtp`]). The stored `frontier` is
	/// DETACHED by the producer (`MtpState`'s contract) - the pool handoff
	/// itself moves already-detached arrays and introduces no new fallible
	/// operation.
	pub mtp: Option<MtpState>,
}

/// Pool sizing knobs, split out from [`PromptCachePool`] so callers —
/// `Session::load_with_cache_config`, fed by `ClientBuilder`'s
/// `cache_max_entries` / `cache_ttl` / `cache_min_tokens` knobs — can
/// override [`PromptCacheConfig::default`] piecemeal without reaching
/// into `Duration`/pool internals.
#[derive(Clone, Copy, Debug)]
pub struct PromptCacheConfig {
	pub max_entries: usize,
	pub max_total_tokens: usize,
	pub ttl: Duration,
	pub min_cacheable_tokens: usize,
}

impl Default for PromptCacheConfig {
	/// Mirrors [`PromptCachePool::with_defaults`]: four entries sharing one
	/// 16,384-token residency budget, 5 minute idle TTL, 8-token minimum.
	fn default() -> Self {
		PromptCacheConfig {
			max_entries: DEFAULT_MAX_ENTRIES,
			max_total_tokens: DEFAULT_MAX_TOTAL_TOKENS,
			ttl: Duration::from_secs(5 * 60),
			min_cacheable_tokens: 8,
		}
	}
}

/// A small, in-process pool of cached prompt prefixes, keyed by
/// longest-common-prefix match on token ids rather than a fixed-block
/// content hash (appropriate at this scale - tens of entries, not a
/// fleet-wide server cache).
pub struct PromptCachePool {
	entries: Vec<CacheEntry>,
	max_entries: usize,
	max_total_tokens: usize,
	ttl: Duration,
	min_cacheable_tokens: usize,
}

impl PromptCachePool {
	pub fn new(max_entries: usize, ttl: Duration, min_cacheable_tokens: usize) -> Self {
		Self::with_token_budget(
			max_entries,
			DEFAULT_MAX_TOTAL_TOKENS,
			ttl,
			min_cacheable_tokens,
		)
	}

	pub fn with_token_budget(
		max_entries: usize,
		max_total_tokens: usize,
		ttl: Duration,
		min_cacheable_tokens: usize,
	) -> Self {
		PromptCachePool {
			entries: Vec::new(),
			max_entries,
			max_total_tokens,
			ttl,
			min_cacheable_tokens,
		}
	}

	/// Default pool sizing: 16 entries, 5 minute idle TTL (mirrors
	/// Anthropic's ephemeral cache-control default lifetime), and an
	/// 8-token minimum before a prefix is worth keeping around at all -
	/// trivially short prompts (a couple of words) cost nothing to
	/// recompute, so caching them just churns pool slots that a
	/// meaningfully-sized prefix could otherwise occupy. This is a much
	/// lower bar than OpenAI's real 1024-token cutoff, which exists for a
	/// different reason (their cache is a shared, metered, multi-tenant
	/// resource; this one is just process-local memory).
	pub fn with_defaults() -> Self {
		Self::from_config(PromptCacheConfig::default())
	}

	pub fn from_config(config: PromptCacheConfig) -> Self {
		Self::with_token_budget(
			config.max_entries,
			config.max_total_tokens,
			config.ttl,
			config.min_cacheable_tokens,
		)
	}

	/// Find the entry whose `ids` is the longest exact prefix of `ids`,
	/// clone it (cheap - see module docs), and return it alongside how
	/// many leading tokens of `ids` it covers (`entry.ids.len()`).
	///
	/// Note this deliberately requires an exact-prefix match, not merely
	/// the longest *common* prefix: a KV cache is append-only, so an entry
	/// whose `ids` diverges from the query partway through (rather than
	/// being a strict prefix of it) holds state for tokens that don't
	/// belong in the new sequence at all and can't be reused - only a
	/// clean reset (equivalent to a pool miss) is correct there.
	///
	/// Returns `None` (equivalent to a fresh/cold start) if no entry is a
	/// prefix of `ids`, or the pool is empty.
	pub fn find_longest_prefix(&mut self, ids: &[u32]) -> Option<(CacheEntry, usize)> {
		self.find_longest_compatible_prefix(ids, false)
	}

	/// emelex patch (not upstream): compatibility-aware sibling of
	/// [`Self::find_longest_prefix`], scoped to calls that would actually
	/// speculate. With `require_mtp` false the behavior is byte-identical
	/// to `find_longest_prefix` (spec-disabled calls - media, spec-off, no
	/// MTP module - treat any entry as compatible, keeping full caching).
	/// With `require_mtp` true, entries without an [`MtpState`] are
	/// classified INCOMPATIBLE: they are skipped (a cold miss when no
	/// compatible entry remains), NOT evicted (they still serve
	/// non-speculating callers), and - crucially - their `last_used` is
	/// NOT refreshed, so repeated incompatible traffic age-demotes rather
	/// than keeping a never-reusable entry warm. The invariant is
	/// no eviction and age demotion through skipped refresh.
	///
	/// Tradeoff note: selection prefers a shorter compatible prefix over
	/// a longer incompatible one — under mixed spec-on/spec-off traffic
	/// this accepts an unbounded prefill loss (the whole incompatible
	/// entry's extra prefix re-runs, however long the conversation) to
	/// preserve a bounded per-token decode win (speculation on the
	/// rebuilt lineage), causing mixed-traffic cache ping-pong.
	pub fn find_longest_compatible_prefix(
		&mut self,
		ids: &[u32],
		require_mtp: bool,
	) -> Option<(CacheEntry, usize)> {
		self.evict_expired();

		let mut best: Option<usize> = None; // entry index
		for (i, entry) in self.entries.iter().enumerate() {
			if !entry.ids.is_empty()
				&& is_prefix(&entry.ids, ids)
				&& (!require_mtp || entry.mtp.is_some())
				&& best
					.map(|b| entry.ids.len() > self.entries[b].ids.len())
					.unwrap_or(true)
			{
				best = Some(i);
			}
		}

		let idx = best?;
		self.entries[idx].last_used = Instant::now();
		let shared = self.entries[idx].ids.len();
		Some((self.entries[idx].clone(), shared))
	}

	/// Insert or refresh a pool entry for `ids`. If an existing entry's
	/// `ids` is a prefix of (or equal to) the new `ids` - the common
	/// "extend this lineage by one more turn" case - it's replaced in
	/// place rather than growing the pool unboundedly across a long
	/// conversation.
	///
	/// A no-op (besides refreshing an existing entry - see below) if
	/// `ids` is shorter than [`Self::min_cacheable_tokens`]: below that,
	/// there's nothing worth keeping a cache slot warm for. Note this
	/// only gates *new* lineages - an existing entry that already cleared
	/// the bar is still refreshed/extended even if, hypothetically,
	/// `ids` were to shrink (it never does in practice: `ids` is always
	/// the previous call's ids plus newly-generated tokens).
	pub fn insert_or_update(
		&mut self,
		ids: Vec<u32>,
		caches: Vec<LayerCache>,
		fed_images: usize,
		fed_audios: usize,
		pinned: bool,
		mtp: Option<MtpState>,
	) {
		// emelex patch: EVERY pool insert enforces the MTP alignment
		// invariant (`pairs_fed == ids.len() - 1` when `mtp` is `Some`) -
		// a misaligned snapshot can never be stored (debug_assert plus a
		// warn-and-drop release fallback inside `aligned_mtp`).
		debug_assert!(
			mtp.as_ref()
				.is_none_or(|state| state.pairs_fed + 1 == ids.len()),
			"pooled MtpState misaligned: pairs_fed {} for {} ids",
			mtp.as_ref().map(|state| state.pairs_fed).unwrap_or(0),
			ids.len(),
		);
		let mtp = aligned_mtp(ids.len(), mtp);
		// emelex patch: expire on insert too - previously entries past
		// their TTL stayed resident (pinning their KV memory) until the
		// next lookup happened to run.
		self.evict_expired();
		let now = Instant::now();
		if ids.len() < self.min_cacheable_tokens
			&& !self.entries.iter().any(|e| is_prefix(&e.ids, &ids))
		{
			return;
		}
		if let Some(existing) = self.entries.iter_mut().find(|e| is_prefix(&e.ids, &ids)) {
			existing.ids = ids;
			existing.caches = caches;
			existing.fed_images = fed_images;
			existing.fed_audios = fed_audios;
			existing.last_used = now;
			existing.pinned = existing.pinned || pinned;
			// emelex patch: `mtp` is overwritten WHOLESALE on every
			// update (alignment rule) - a spec-off extension of an
			// MTP-bearing lineage writes `None`, a spec-on rebuild of an
			// `mtp`-less lineage writes `Some`.
			existing.mtp = mtp;
			self.evict_lru_if_over_capacity();
			return;
		}

		self.entries.push(CacheEntry {
			ids,
			caches,
			fed_images,
			fed_audios,
			last_used: now,
			pinned,
			mtp,
		});
		self.evict_lru_if_over_capacity();
	}

	fn evict_expired(&mut self) {
		let ttl = self.ttl;
		let now = Instant::now();
		self.entries
			.retain(|e| e.pinned || now.duration_since(e.last_used) < ttl);
	}

	fn evict_lru_if_over_capacity(&mut self) {
		while self.entries.len() > self.max_entries || self.total_tokens() > self.max_total_tokens {
			let unpinned = self
				.entries
				.iter()
				.enumerate()
				.filter(|(_, e)| !e.pinned)
				.min_by_key(|(_, e)| e.last_used)
				.map(|(i, _)| i);
			// emelex patch: pinning protects against ordinary LRU pressure,
			// never against the hard aggregate token-residency ceiling.
			let victim = unpinned.or_else(|| {
				self.entries
					.iter()
					.enumerate()
					.min_by_key(|(_, entry)| entry.last_used)
					.map(|(index, _)| index)
			});
			match victim {
				Some(i) => {
					self.entries.remove(i);
				}
				// Every entry is pinned - nothing evictable, stop trying.
				None => break,
			}
		}
	}

	fn total_tokens(&self) -> usize {
		self.entries.iter().fold(0_usize, |total, entry| {
			total.saturating_add(entry.ids.len())
		})
	}

	pub fn len(&self) -> usize {
		self.entries.len()
	}

	pub fn is_empty(&self) -> bool {
		self.entries.is_empty()
	}
}

pub(crate) fn is_prefix(shorter: &[u32], longer: &[u32]) -> bool {
	shorter.len() <= longer.len() && shorter == &longer[..shorter.len()]
}

/// emelex patch (not upstream): release-mode guard behind the pool's MTP
/// alignment invariant. A `Some` snapshot whose `pairs_fed` is not exactly
/// `ids_len - 1` is dropped (warn) rather than stored - a misaligned
/// `MtpState` served on a later hit would prime the bridge pair against
/// the wrong position and desync the MTP cache, so it must never enter
/// the pool. [`PromptCachePool::insert_or_update`] pairs this with a
/// `debug_assert` so the misalignment is loud in development; this
/// function is the never-panicking release behavior, unit-tested
/// directly.
fn aligned_mtp(ids_len: usize, mtp: Option<MtpState>) -> Option<MtpState> {
	match mtp {
		Some(state) if state.pairs_fed + 1 != ids_len => {
			tracing::warn!(
				pairs_fed = state.pairs_fed,
				ids_len,
				"dropping misaligned pooled MtpState (pairs_fed must equal ids_len - \
				 1)"
			);
			None
		}
		other => other,
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn empty_caches() -> Vec<LayerCache> {
		Vec::new()
	}

	#[test]
	fn find_longest_prefix_picks_the_best_match() {
		let mut pool = PromptCachePool::new(16, Duration::from_secs(300), 0);
		pool.insert_or_update(vec![1, 2, 3], empty_caches(), 0, 0, false, None);
		pool.insert_or_update(vec![1, 2, 3, 4, 5], empty_caches(), 0, 0, false, None);

		// Both entries share a prefix with [1,2,3,4,5,6]; the longer one
		// must win.
		let (entry, shared) = pool.find_longest_prefix(&[1, 2, 3, 4, 5, 6]).unwrap();
		assert_eq!(entry.ids, vec![1, 2, 3, 4, 5]);
		assert_eq!(shared, 5);
	}

	#[test]
	fn find_longest_prefix_returns_none_when_no_overlap() {
		let mut pool = PromptCachePool::new(16, Duration::from_secs(300), 0);
		pool.insert_or_update(vec![1, 2, 3], empty_caches(), 0, 0, false, None);
		assert!(pool.find_longest_prefix(&[9, 9, 9]).is_none());
	}

	#[test]
	fn insert_or_update_extends_existing_lineage_in_place() {
		let mut pool = PromptCachePool::new(16, Duration::from_secs(300), 0);
		pool.insert_or_update(vec![1, 2, 3], empty_caches(), 0, 0, false, None);
		pool.insert_or_update(vec![1, 2, 3, 4, 5], empty_caches(), 1, 0, false, None);
		assert_eq!(
			pool.len(),
			1,
			"extending a lineage should not grow the pool"
		);
	}

	#[test]
	fn insert_or_update_evicts_lru_over_capacity() {
		let mut pool = PromptCachePool::new(2, Duration::from_secs(300), 0);
		pool.insert_or_update(vec![1], empty_caches(), 0, 0, false, None);
		pool.insert_or_update(vec![2], empty_caches(), 0, 0, false, None);
		// Touch entry [1] so [2] becomes the least-recently-used.
		pool.find_longest_prefix(&[1]);
		pool.insert_or_update(vec![3], empty_caches(), 0, 0, false, None);

		assert_eq!(pool.len(), 2);
		assert!(
			pool.find_longest_prefix(&[2]).is_none(),
			"LRU entry should have been evicted"
		);
		assert!(pool.find_longest_prefix(&[1]).is_some());
		assert!(pool.find_longest_prefix(&[3]).is_some());
	}

	#[test]
	fn weighted_lru_never_exceeds_aggregate_token_budget() {
		let mut pool = PromptCachePool::with_token_budget(8, 6, Duration::from_secs(300), 0);
		pool.insert_or_update(vec![1, 1, 1, 1], empty_caches(), 0, 0, false, None);
		pool.insert_or_update(vec![2, 2, 2, 2], empty_caches(), 0, 0, false, None);

		assert_eq!(pool.len(), 1);
		assert!(pool.total_tokens() <= 6);
		assert!(pool.find_longest_prefix(&[1, 1, 1, 1]).is_none());
		assert!(pool.find_longest_prefix(&[2, 2, 2, 2]).is_some());
	}

	#[test]
	fn one_entry_larger_than_hard_budget_is_not_retained() {
		let mut pool = PromptCachePool::with_token_budget(8, 3, Duration::from_secs(300), 0);
		pool.insert_or_update(vec![1, 2, 3, 4], empty_caches(), 0, 0, true, None);
		assert!(pool.is_empty());
	}

	#[test]
	fn pinned_entries_survive_lru_pressure() {
		// Capacity 2: [1] (pinned) + [2], then adding [3] must evict [2]
		// (the only unpinned entry) rather than the pinned [1], even
		// though [1] is now the least-recently-touched by wall-clock time.
		let mut pool = PromptCachePool::new(2, Duration::from_secs(300), 0);
		pool.insert_or_update(vec![1], empty_caches(), 0, 0, true, None);
		pool.insert_or_update(vec![2], empty_caches(), 0, 0, false, None);
		pool.insert_or_update(vec![3], empty_caches(), 0, 0, false, None);

		assert_eq!(pool.len(), 2);
		assert!(
			pool.find_longest_prefix(&[1]).is_some(),
			"pinned entry must survive eviction"
		);
		assert!(
			pool.find_longest_prefix(&[2]).is_none(),
			"unpinned entry should have been evicted"
		);
		assert!(pool.find_longest_prefix(&[3]).is_some());
	}

	#[test]
	fn entries_shorter_than_the_minimum_are_not_cached() {
		let mut pool = PromptCachePool::new(16, Duration::from_secs(300), 8);
		pool.insert_or_update(vec![1, 2, 3], empty_caches(), 0, 0, false, None);
		assert!(
			pool.is_empty(),
			"a 3-token entry should be rejected by an 8-token minimum"
		);
		assert!(pool.find_longest_prefix(&[1, 2, 3]).is_none());
	}

	#[test]
	fn a_lineage_becomes_cacheable_once_it_crosses_the_minimum() {
		let mut pool = PromptCachePool::new(16, Duration::from_secs(300), 8);
		// Turn one: 3 tokens, below the minimum - not stored.
		pool.insert_or_update(vec![1, 2, 3], empty_caches(), 0, 0, false, None);
		assert!(pool.is_empty());

		// Turn two extends the same lineage past the minimum - now stored,
		// and reachable by a later exact-prefix lookup.
		let turn_two: Vec<u32> = (1..=10).collect();
		pool.insert_or_update(turn_two.clone(), empty_caches(), 0, 0, false, None);
		assert_eq!(pool.len(), 1);
		assert!(pool.find_longest_prefix(&turn_two).is_some());
	}

	#[test]
	fn expired_unpinned_entries_are_evicted() {
		let mut pool = PromptCachePool::new(16, Duration::from_millis(1), 0);
		pool.insert_or_update(vec![1, 2, 3], empty_caches(), 0, 0, false, None);
		std::thread::sleep(Duration::from_millis(5));
		assert!(pool.find_longest_prefix(&[1, 2, 3]).is_none());
	}

	#[test]
	fn insert_evicts_expired_entries_eagerly() {
		// emelex patch regression: entries past their TTL are dropped on
		// the next insert, not only on the next lookup.
		let mut pool = PromptCachePool::new(8, Duration::ZERO, 1);
		pool.insert_or_update(vec![1, 2, 3], Vec::new(), 0, 0, false, None);
		assert_eq!(pool.len(), 1);
		pool.insert_or_update(vec![9, 9, 9, 9], Vec::new(), 0, 0, false, None);
		assert_eq!(pool.len(), 1, "expired first entry should be gone");
	}

	// -----------------------------------------------------------------
	// emelex patch (not upstream): pooled MtpState tests.
	// -----------------------------------------------------------------

	use crate::engine::{
		array::Array,
		models::mtp::{MtpCaches, MtpState},
	};

	/// An `MtpState` whose `pairs_fed` field is set directly (the pool
	/// validates the field, not the underlying cache offsets - those are
	/// the decode loop's invariant).
	fn mtp_state(pairs_fed: usize) -> MtpState {
		MtpState {
			caches: MtpCaches(Vec::new()),
			pairs_fed,
			frontier: Array::from_slice(&[0.0f32], &[1, 1, 1]).unwrap(),
		}
	}

	#[test]
	fn aligned_mtp_insert_is_stored_and_served() {
		let mut pool = PromptCachePool::new(16, Duration::from_secs(300), 0);
		pool.insert_or_update(
			vec![1, 2, 3],
			empty_caches(),
			0,
			0,
			false,
			Some(mtp_state(2)),
		);
		let (entry, shared) = pool
			.find_longest_compatible_prefix(&[1, 2, 3, 4], true)
			.unwrap();
		assert_eq!(shared, 3);
		let state = entry.mtp.expect("aligned MtpState must be stored");
		assert_eq!(state.pairs_fed, entry.ids.len() - 1);
	}

	#[test]
	fn aligned_mtp_drops_misaligned_state_in_release() {
		// Release behavior of the insert helper: a misaligned snapshot is
		// warn-dropped, never stored (the debug_assert in insert_or_update
		// is the development-loudness half - see the should_panic test).
		assert!(aligned_mtp(3, Some(mtp_state(3))).is_none());
		assert!(aligned_mtp(3, Some(mtp_state(1))).is_none());
		assert!(aligned_mtp(3, Some(mtp_state(2))).is_some());
		assert!(aligned_mtp(3, None).is_none());
	}

	#[cfg(debug_assertions)]
	#[test]
	#[should_panic(expected = "pooled MtpState misaligned")]
	fn misaligned_mtp_insert_debug_asserts() {
		let mut pool = PromptCachePool::new(16, Duration::from_secs(300), 0);
		pool.insert_or_update(
			vec![1, 2, 3],
			empty_caches(),
			0,
			0,
			false,
			Some(mtp_state(7)),
		);
	}

	#[test]
	fn spec_lookup_misses_mtp_less_entry_without_evicting() {
		let mut pool = PromptCachePool::new(16, Duration::from_secs(300), 0);
		pool.insert_or_update(vec![1, 2, 3], empty_caches(), 0, 0, false, None);

		// A speculating call is a COLD MISS on an mtp-less entry...
		assert!(
			pool.find_longest_compatible_prefix(&[1, 2, 3, 4], true)
				.is_none(),
			"spec-enabled lookup must not use an mtp-less entry"
		);
		// ...but the entry is NOT evicted: it still serves spec-off calls.
		assert_eq!(pool.len(), 1);
		assert!(
			pool.find_longest_compatible_prefix(&[1, 2, 3, 4], false)
				.is_some(),
			"the entry must keep serving non-speculating callers"
		);
	}

	#[test]
	fn spec_lookup_prefers_a_compatible_shorter_prefix() {
		// Two distinct lineages, both exact prefixes of the query: the
		// longer one has no MtpState (insert order matters - the longer
		// entry first, so the shorter insert is not an in-place lineage
		// extension of it). A speculating lookup must fall back to the
		// shorter, compatible entry rather than cold-miss entirely.
		let mut pool = PromptCachePool::new(16, Duration::from_secs(300), 0);
		pool.insert_or_update(vec![1, 2, 3], empty_caches(), 0, 0, false, None);
		pool.insert_or_update(vec![1, 2], empty_caches(), 0, 0, false, Some(mtp_state(1)));
		assert_eq!(pool.len(), 2);

		let (entry, shared) = pool
			.find_longest_compatible_prefix(&[1, 2, 3, 4], true)
			.unwrap();
		assert_eq!(shared, 2);
		assert!(entry.mtp.is_some());
		// The plain lookup still prefers the longer entry.
		let (_, shared) = pool
			.find_longest_compatible_prefix(&[1, 2, 3, 4], false)
			.unwrap();
		assert_eq!(shared, 3);
	}

	#[test]
	fn incompatible_lookup_skips_last_used_refresh() {
		// Age-demotion via skipped refresh: entry A is older than B; a
		// spec-enabled lookup that classifies A incompatible must NOT
		// refresh it, so LRU pressure still evicts A first.
		let mut pool = PromptCachePool::new(2, Duration::from_secs(300), 0);
		pool.insert_or_update(vec![1, 2], empty_caches(), 0, 0, false, None);
		pool.insert_or_update(vec![8, 9], empty_caches(), 0, 0, false, None);

		assert!(
			pool.find_longest_compatible_prefix(&[1, 2, 3], true)
				.is_none()
		);
		pool.insert_or_update(vec![5, 6], empty_caches(), 0, 0, false, None);
		assert_eq!(pool.len(), 2);
		assert!(
			pool.find_longest_prefix(&[1, 2]).is_none(),
			"the incompatible lookup must not have refreshed [1, 2] - it stays LRU \
			 and is evicted"
		);
		assert!(pool.find_longest_prefix(&[8, 9]).is_some());

		// Control: the SAME sequence with a compatible (require_mtp false)
		// lookup refreshes, and the other entry is evicted instead.
		let mut pool = PromptCachePool::new(2, Duration::from_secs(300), 0);
		pool.insert_or_update(vec![1, 2], empty_caches(), 0, 0, false, None);
		pool.insert_or_update(vec![8, 9], empty_caches(), 0, 0, false, None);
		assert!(
			pool.find_longest_compatible_prefix(&[1, 2, 3], false)
				.is_some()
		);
		pool.insert_or_update(vec![5, 6], empty_caches(), 0, 0, false, None);
		assert!(
			pool.find_longest_prefix(&[1, 2]).is_some(),
			"the compatible lookup refreshed [1, 2], demoting [8, 9] to LRU"
		);
		assert!(pool.find_longest_prefix(&[8, 9]).is_none());
	}

	#[test]
	fn insert_overwrites_mtp_wholesale_on_lineage_updates() {
		let mut pool = PromptCachePool::new(16, Duration::from_secs(300), 0);
		pool.insert_or_update(vec![1, 2], empty_caches(), 0, 0, false, Some(mtp_state(1)));
		// A spec-off extension of the lineage writes `mtp = None`...
		pool.insert_or_update(vec![1, 2, 3], empty_caches(), 0, 0, false, None);
		assert_eq!(pool.len(), 1);
		let (entry, _) = pool.find_longest_prefix(&[1, 2, 3]).unwrap();
		assert!(
			entry.mtp.is_none(),
			"a None extension must clear the stored MtpState wholesale"
		);
		// ...and a later spec-on rebuild writes `Some` back.
		pool.insert_or_update(
			vec![1, 2, 3, 4],
			empty_caches(),
			0,
			0,
			false,
			Some(mtp_state(3)),
		);
		assert_eq!(pool.len(), 1);
		let (entry, _) = pool.find_longest_prefix(&[1, 2, 3, 4]).unwrap();
		assert!(entry.mtp.is_some());
	}
}
