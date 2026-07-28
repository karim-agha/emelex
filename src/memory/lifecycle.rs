//! Deterministic compaction, Knowledge distillation, and retention workers.

use std::{
	collections::BTreeSet,
	fmt,
	fs::{self, OpenOptions},
	os::unix::fs::OpenOptionsExt as _,
	path::PathBuf,
	sync::atomic::{AtomicBool, Ordering},
	time::Duration,
};

use chrono::{DateTime, Utc};
use rusqlite::{Connection, OptionalExtension as _, Transaction, TransactionBehavior, params};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{
	AssetGcReport, CompactionJob, CompactionLease, Knowledge, MAX_KNOWLEDGE_BYTES,
	MAX_KNOWLEDGE_KEY_BYTES, MAX_REPLAY_BYTES, MAX_REPLAY_EVENTS, MemoryError,
	MemoryJobFailureDisposition, MemoryJobFailureOutcome, MemoryStore, SessionEvent,
	TranscriptProvenance, bounded_replay_event, deadline_after, event_select,
	job_failure_transition, load_session_replay, parse_session_claim, parse_time, parse_uuid,
	raw_event, recover_one_expired_compaction, transcript_provenance, validate_compaction_lease,
	validate_required,
};

const MAX_DISTILLATION_CANDIDATES: usize = 128;
const MAX_MAINTENANCE_ROWS: usize = 10_000;
const DEFAULT_MAINTENANCE_ROWS: usize = 1_000;
// Conservative byte-based estimate used without a live selected tokenizer.
const TOKEN_BYTES: u64 = 1;

/// Fixed-ratio deterministic transcript compaction policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct CompactionPolicy {
	context_window_tokens: u64,
	preserve_recent_turns: usize,
}

impl CompactionPolicy {
	/// Construct an 80%-trigger, 50%-target policy.
	///
	/// # Errors
	///
	/// Returns an error for a zero context window.
	pub fn new(context_window_tokens: u64) -> Result<Self, MemoryError> {
		if context_window_tokens == 0 {
			return Err(MemoryError::Invalid(
				"compaction context window must be positive".to_string(),
			));
		}
		Ok(Self {
			context_window_tokens,
			preserve_recent_turns: 4,
		})
	}

	/// Preserve this many newest atomic turns outside the summary.
	///
	/// # Errors
	///
	/// Returns an error unless `turns` is in `1..=64`.
	pub fn preserve_recent_turns(mut self, turns: usize) -> Result<Self, MemoryError> {
		if !(1..=64).contains(&turns) {
			return Err(MemoryError::Invalid(
				"preserved recent turns must be in 1..=64".to_string(),
			));
		}
		self.preserve_recent_turns = turns;
		Ok(self)
	}

	/// Model context-window ceiling.
	pub const fn context_window_tokens(self) -> u64 {
		self.context_window_tokens
	}

	/// Number of newest atomic turns never covered by a new summary.
	pub const fn recent_turns(self) -> usize {
		self.preserve_recent_turns
	}
}

/// Deterministic compaction boundary and accounting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CompactionPlan {
	/// Inclusive complete-turn sequence boundary.
	pub through_sequence: u64,
	/// Caller-supplied current effective context estimate.
	pub current_tokens: u64,
	/// 80% trigger in tokens.
	pub trigger_tokens: u64,
	/// 50% target in tokens.
	pub target_tokens: u64,
	/// Persisted payload-token estimate removed by this boundary.
	pub estimated_removed_tokens: u64,
	/// Number of newest turns protected from this compaction.
	pub preserved_turns: usize,
}

/// Durable Knowledge-distillation queue state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum DistillationState {
	/// Waiting for a worker after clean exit.
	Pending,
	/// Exclusively claimed by one worker.
	Running,
	/// Candidate mutations committed atomically.
	Completed,
	/// Bounded retries were exhausted or a permanent failure was recorded.
	Failed,
}

/// Idempotent transcript-to-Knowledge work item.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DistillationJob {
	/// `UUIDv7` job identity.
	pub id: Uuid,
	/// Source Session.
	pub session_id: Uuid,
	/// Inclusive source boundary.
	pub through_sequence: u64,
	/// Exact immutable source prefix.
	pub source: TranscriptProvenance,
	/// Queue state.
	pub state: DistillationState,
	/// Failed attempts recorded for this job.
	pub failures: u32,
	/// Earliest time at which a pending retry may be claimed.
	pub retry_after: Option<DateTime<Utc>>,
	/// Most recent bounded worker failure.
	pub last_error: Option<String>,
	/// Time at which this job entered terminal failed state.
	pub failed_at: Option<DateTime<Utc>>,
	/// Creation time.
	pub created_at: DateTime<Utc>,
	/// Last state transition.
	pub updated_at: DateTime<Utc>,
	/// Completion time.
	pub completed_at: Option<DateTime<Utc>>,
}

/// Exclusive authority to read and complete one distillation job.
#[non_exhaustive]
pub struct DistillationLease {
	store: MemoryStore,
	job: DistillationJob,
	token: Uuid,
	lease_until: DateTime<Utc>,
	released: AtomicBool,
}

impl fmt::Debug for DistillationLease {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter
			.debug_struct("DistillationLease")
			.field("job", &self.job)
			.field("token", &"<redacted>")
			.field("lease_until", &self.lease_until)
			.finish_non_exhaustive()
	}
}

impl DistillationLease {
	/// Claimed work item.
	pub const fn job(&self) -> &DistillationJob {
		&self.job
	}

	/// Deadline after which another worker may recover this job.
	pub const fn lease_until(&self) -> DateTime<Utc> {
		self.lease_until
	}
}

impl Drop for DistillationLease {
	fn drop(&mut self) {
		if self.released.swap(true, Ordering::AcqRel) {
			return;
		}
		let _ = self.store.record_distillation_failure_best_effort(
			self,
			"worker claim dropped before completion",
			MemoryJobFailureDisposition::Retry,
		);
	}
}

/// One bounded, typed model proposal for durable Knowledge mutation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
#[non_exhaustive]
pub enum DistillationCandidateInput {
	/// Create a new active version.
	Upsert {
		/// Stable workspace-local key.
		key: String,
		/// Concise factual content.
		content: String,
		/// Calibrated confidence in `0.0..=1.0`.
		confidence: f64,
		/// Whether this proposal explicitly requests retention priority.
		pinned: bool,
	},
	/// Hide a stale key while retaining an auditable tombstone.
	Tombstone {
		/// Existing workspace-local key.
		key: String,
		/// Calibrated confidence in `0.0..=1.0`.
		confidence: f64,
	},
}

/// Applied result of one distillation candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
#[non_exhaustive]
pub enum DistillationCandidate {
	/// New Knowledge version became active.
	Upserted {
		/// Applied active Knowledge.
		knowledge: Knowledge,
	},
	/// Existing Knowledge became hidden.
	Tombstoned {
		/// Stable Knowledge identity.
		knowledge_id: Uuid,
		/// Tombstoned key.
		key: String,
		/// Distiller confidence.
		confidence: f64,
	},
}

/// Automatic retention policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct RetentionPolicy {
	/// Unclaimed Sessions older than this may be removed.
	pub session_max_age: Duration,
	/// Inactive, unpinned Knowledge older than this becomes tombstoned.
	pub knowledge_max_age: Duration,
	/// Tombstones remain auditable for at least this duration.
	pub tombstone_grace: Duration,
	/// Historical versions retained per unpinned key, including active rank.
	pub versions_per_key: usize,
	/// Unreferenced assets remain recoverable for at least this duration.
	pub asset_grace: Duration,
}

impl Default for RetentionPolicy {
	fn default() -> Self {
		Self {
			session_max_age: Duration::from_hours(2_160),
			knowledge_max_age: Duration::from_hours(4_320),
			tombstone_grace: Duration::from_hours(720),
			versions_per_key: 8,
			asset_grace: Duration::from_hours(168),
		}
	}
}

/// One bounded maintenance invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct MaintenanceOptions {
	/// Retention windows.
	pub retention: RetentionPolicy,
	/// Maximum rows changed by each retention phase.
	pub max_rows: usize,
	/// Rebuild the database after bounded retention.
	pub vacuum: bool,
}

impl Default for MaintenanceOptions {
	fn default() -> Self {
		Self {
			retention: RetentionPolicy::default(),
			max_rows: DEFAULT_MAINTENANCE_ROWS,
			vacuum: false,
		}
	}
}

/// Counts and checkpoint state from one maintenance pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct MaintenanceReport {
	/// Expired Session claims cleared.
	pub session_claims_recovered: usize,
	/// Expired compaction claims recorded as failed attempts.
	pub compactions_recovered: usize,
	/// Expired distillation claims recorded as failed attempts.
	pub distillations_recovered: usize,
	/// Old unclaimed Sessions removed.
	pub sessions_removed: usize,
	/// Old Knowledge entries newly tombstoned.
	pub knowledge_tombstoned: usize,
	/// Expired tombstones physically removed.
	pub knowledge_removed: usize,
	/// Superseded Knowledge versions removed.
	pub versions_removed: usize,
	/// Content-addressed asset collection.
	pub assets: AssetGcReport,
	/// `true` when another connection prevented a full WAL truncate.
	pub wal_busy: bool,
	/// Whether explicit `VACUUM` ran.
	pub vacuumed: bool,
}

impl MemoryStore {
	fn validate_distillation_lease_origin(
		&self,
		lease: &DistillationLease,
	) -> Result<(), MemoryError> {
		if lease.store.database != self.database {
			return Err(MemoryError::Invalid(
				"distillation lease belongs to another MemoryStore".to_string(),
			));
		}
		Ok(())
	}

	/// Compute the complete-turn boundary for 80%-to-50% compaction.
	///
	/// Returns `None` below the trigger or when preserving recent turns leaves
	/// insufficient source history.
	///
	/// # Errors
	///
	/// Returns missing Session, corrupt-turn, overflow, or database errors.
	#[expect(
		clippy::too_many_lines,
		reason = "turn grouping and token-boundary selection remain one deterministic pass"
	)]
	pub fn plan_compaction(
		&self,
		session_id: Uuid,
		current_tokens: u64,
		policy: CompactionPolicy,
	) -> Result<Option<CompactionPlan>, MemoryError> {
		self.session(session_id)?;
		let trigger_tokens = percent_ceil(policy.context_window_tokens, 80)?;
		if current_tokens < trigger_tokens {
			return Ok(None);
		}
		let target_tokens = percent_floor(policy.context_window_tokens, 50)?;
		let desired = current_tokens.saturating_sub(target_tokens);
		let connection = self.connection()?;
		let effective_after: i64 = connection.query_row(
			"SELECT COALESCE(MAX(through_sequence), 0)
			 FROM compaction_jobs
			 WHERE session_id = ?1 AND state = 'completed'",
			[session_id.to_string()],
			|row| row.get(0),
		)?;
		let mut statement = connection.prepare(
			"SELECT sequence, turn_id, turn_index, turn_size, length(payload_json)
			 FROM session_events
			 WHERE session_id = ?1 AND sequence > ?2
			 ORDER BY sequence ASC
			 LIMIT ?3",
		)?;
		let rows = statement.query_map(
			params![
				session_id.to_string(),
				effective_after,
				i64::try_from(MAX_REPLAY_EVENTS).map_err(|_| {
					MemoryError::Invalid("compaction event limit is invalid".to_string())
				})?
			],
			|row| {
				Ok((
					row.get::<_, i64>(0)?,
					row.get::<_, String>(1)?,
					row.get::<_, i64>(2)?,
					row.get::<_, i64>(3)?,
					row.get::<_, i64>(4)?,
				))
			},
		)?;
		let mut groups = Vec::<TurnWeight>::new();
		let mut current: Option<TurnWeight> = None;
		for row in rows {
			let (sequence, turn_id, index, size, bytes) = row?;
			if sequence <= 0 || index < 0 || size <= 0 || index >= size || bytes < 0 {
				return Err(MemoryError::Corrupt(format!(
					"session {session_id} has invalid compaction accounting metadata"
				)));
			}
			let sequence = u64::try_from(sequence)
				.map_err(|_| MemoryError::Corrupt("negative event sequence".to_string()))?;
			let bytes = u64::try_from(bytes)
				.map_err(|_| MemoryError::Corrupt("negative event length".to_string()))?;
			let tokens = bytes.saturating_add(TOKEN_BYTES - 1) / TOKEN_BYTES;
			if index == 0 {
				if let Some(group) = current.take() {
					if group.seen != group.size {
						return Err(MemoryError::Corrupt(format!(
							"session {session_id} compaction accounting splits turn {}",
							group.turn_id
						)));
					}
					groups.push(group);
				}
				current = Some(TurnWeight {
					turn_id,
					size,
					seen: 1,
					end_sequence: sequence,
					tokens: tokens.max(1),
				});
			} else {
				let group = current.as_mut().ok_or_else(|| {
					MemoryError::Corrupt(format!(
						"session {session_id} starts inside an atomic turn"
					))
				})?;
				if group.turn_id != turn_id || index != group.seen || size != group.size {
					return Err(MemoryError::Corrupt(format!(
						"session {session_id} has interleaved atomic turns"
					)));
				}
				group.seen += 1;
				group.end_sequence = sequence;
				group.tokens = group.tokens.checked_add(tokens.max(1)).ok_or_else(|| {
					MemoryError::Corrupt("compaction token estimate overflow".to_string())
				})?;
			}
		}
		if let Some(group) = current {
			if group.seen != group.size {
				return Err(MemoryError::Corrupt(format!(
					"session {session_id} ends inside atomic turn {}",
					group.turn_id
				)));
			}
			groups.push(group);
		}
		let removable = groups.len().saturating_sub(policy.preserve_recent_turns);
		let mut removed = 0_u64;
		let mut boundary = None;
		for group in groups.iter().take(removable) {
			removed = removed.checked_add(group.tokens).ok_or_else(|| {
				MemoryError::Corrupt("compaction token estimate overflow".to_string())
			})?;
			boundary = Some(group.end_sequence);
			if removed >= desired {
				break;
			}
		}
		if removed < desired {
			return Ok(None);
		}
		Ok(boundary.map(|through_sequence| CompactionPlan {
			through_sequence,
			current_tokens,
			trigger_tokens,
			target_tokens,
			estimated_removed_tokens: removed,
			preserved_turns: policy.preserve_recent_turns,
		}))
	}

	/// Plan and idempotently queue deterministic compaction when needed.
	///
	/// # Errors
	///
	/// Returns [`MemoryStore::plan_compaction`] or queue failures.
	pub fn queue_compaction_if_needed(
		&self,
		session_id: Uuid,
		current_tokens: u64,
		policy: CompactionPolicy,
	) -> Result<Option<(CompactionPlan, CompactionJob)>, MemoryError> {
		let Some(plan) = self.plan_compaction(session_id, current_tokens, policy)? else {
			return Ok(None);
		};
		let job = self.queue_compaction(session_id, plan.through_sequence)?;
		Ok(Some((plan, job)))
	}

	/// Load the effective bounded transcript prefix covered by a claimed
	/// compaction job.
	///
	/// Prior raw prefixes already replaced by a verified summary are not
	/// returned to the summarizer.
	///
	/// # Errors
	///
	/// Returns stale/provenance, replay-bound, corruption, or database errors.
	pub fn compaction_source(
		&self,
		lease: &CompactionLease,
	) -> Result<Vec<SessionEvent>, MemoryError> {
		let mut connection = self.connection()?;
		let transaction = connection.transaction_with_behavior(TransactionBehavior::Deferred)?;
		validate_compaction_lease(&transaction, lease, Utc::now())?;
		let replay = load_session_replay(&transaction, lease.job.session_id)?;
		let events = replay
			.events
			.into_iter()
			.filter(|event| event.sequence <= lease.job.through_sequence)
			.collect::<Vec<_>>();
		if events.is_empty() {
			return Err(MemoryError::Corrupt(format!(
				"compaction {} has no effective source events",
				lease.job.id
			)));
		}
		transaction.commit()?;
		Ok(events)
	}

	/// Queue idempotent distillation of the Session's current full prefix.
	///
	/// This is designed for clean exit: queueing is short and synchronous;
	/// model work happens later under a bounded worker lease.
	///
	/// # Errors
	///
	/// Returns missing/empty Session, provenance, overflow, or database errors.
	pub fn queue_distillation(
		&self,
		session_id: Uuid,
	) -> Result<Option<DistillationJob>, MemoryError> {
		let mut connection = self.connection()?;
		let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
		transaction
			.query_row(
				"SELECT 1 FROM sessions WHERE id = ?1",
				[session_id.to_string()],
				|_| Ok(()),
			)
			.optional()?
			.ok_or(MemoryError::NotFound {
				entity: "session",
				id: session_id,
			})?;
		let through: i64 = transaction.query_row(
			"SELECT COALESCE(MAX(sequence), 0)
			 FROM session_events WHERE session_id = ?1",
			[session_id.to_string()],
			|row| row.get(0),
		)?;
		if through == 0 {
			transaction.commit()?;
			return Ok(None);
		}
		let source = transcript_provenance(&transaction, &session_id.to_string(), through)?;
		let existing = transaction
			.query_row(
				&distillation_select("WHERE j.session_id = ?1 AND j.source_sha256 = ?2"),
				params![session_id.to_string(), &source.sha256],
				raw_distillation,
			)
			.optional()?;
		if let Some(raw) = existing {
			transaction.commit()?;
			return DistillationJob::try_from(raw).map(Some);
		}
		let id = Uuid::now_v7();
		let now = Utc::now();
		transaction.execute(
			"INSERT INTO distillation_jobs
			 (id, session_id, through_sequence, source_event_count,
			  source_first_event_id, source_last_event_id, source_sha256,
			  state, claim_token, lease_until, created_at, updated_at, completed_at)
			 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 'pending',
			         NULL, NULL, ?8, ?8, NULL)",
			params![
				id.to_string(),
				session_id.to_string(),
				through,
				i64::try_from(source.event_count).map_err(|_| {
					MemoryError::Corrupt("distillation source count overflow".to_string())
				})?,
				source.first_event_id.to_string(),
				source.last_event_id.to_string(),
				&source.sha256,
				now.to_rfc3339(),
			],
		)?;
		transaction.commit()?;
		Ok(Some(DistillationJob {
			id,
			session_id,
			through_sequence: u64::try_from(through)
				.map_err(|_| MemoryError::Corrupt("negative distillation boundary".to_string()))?,
			source,
			state: DistillationState::Pending,
			failures: 0,
			retry_after: None,
			last_error: None,
			failed_at: None,
			created_at: now,
			updated_at: now,
			completed_at: None,
		}))
	}

	/// Claim the oldest pending distillation whose retry deadline has arrived.
	///
	/// Expired worker claims first become counted failures with bounded backoff
	/// or terminal failed state. Jobs whose source Session has a live execution
	/// lease are skipped.
	///
	/// # Errors
	///
	/// Returns database or corruption errors.
	pub fn claim_distillation(&self) -> Result<Option<DistillationLease>, MemoryError> {
		let mut connection = self.connection()?;
		let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
		let now = Utc::now();
		for _ in 0..super::MAX_STALE_JOB_RECOVERIES {
			if !recover_one_expired_distillation(&transaction, now)? {
				break;
			}
		}
		let sql = distillation_select(
			"JOIN sessions s ON s.id = j.session_id
			 WHERE j.state = 'pending'
			   AND (j.retry_after IS NULL OR j.retry_after <= ?1)
			   AND (s.execution_token IS NULL OR s.execution_lease_until <= ?1)
			 ORDER BY j.created_at ASC, j.id ASC LIMIT 1",
		);
		let raw = transaction
			.query_row(&sql, [now.to_rfc3339()], raw_distillation)
			.optional()?;
		let Some(raw) = raw else {
			transaction.commit()?;
			return Ok(None);
		};
		let token = Uuid::now_v7();
		let lease_until = deadline_after(now, super::DISTILLATION_LEASE, "distillation lease")?;
		let changed = transaction.execute(
			"UPDATE distillation_jobs
			 SET state = 'running', claim_token = ?2, lease_until = ?3,
			     retry_after = NULL, updated_at = ?4
			 WHERE id = ?1
			   AND state = 'pending'
			   AND (retry_after IS NULL OR retry_after <= ?4)",
			params![
				&raw.id,
				token.to_string(),
				lease_until.to_rfc3339(),
				now.to_rfc3339(),
			],
		)?;
		if changed != 1 {
			return Err(MemoryError::Corrupt(
				"distillation claim lost inside immediate transaction".to_string(),
			));
		}
		transaction.commit()?;
		let mut job = DistillationJob::try_from(raw)?;
		job.state = DistillationState::Running;
		job.retry_after = None;
		job.updated_at = now;
		Ok(Some(DistillationLease {
			store: self.clone(),
			job,
			token,
			lease_until,
			released: AtomicBool::new(false),
		}))
	}

	/// Renew a distillation claim while this worker still owns it.
	///
	/// # Errors
	///
	/// Returns stale-authority or database errors.
	pub fn renew_distillation(&self, lease: &mut DistillationLease) -> Result<(), MemoryError> {
		self.validate_distillation_lease_origin(lease)?;
		let now = Utc::now();
		let lease_until = deadline_after(now, super::DISTILLATION_LEASE, "distillation lease")?;
		let changed = self.connection()?.execute(
			"UPDATE distillation_jobs
			 SET lease_until = ?3, updated_at = ?4
			 WHERE id = ?1 AND state = 'running' AND claim_token = ?2",
			params![
				lease.job.id.to_string(),
				lease.token.to_string(),
				lease_until.to_rfc3339(),
				now.to_rfc3339(),
			],
		)?;
		if changed != 1 {
			return Err(MemoryError::StaleDistillationLease {
				job_id: lease.job.id,
			});
		}
		lease.lease_until = lease_until;
		lease.job.updated_at = now;
		Ok(())
	}

	/// Record a bounded worker failure under the current distillation claim.
	///
	/// Retryable failures use persisted exponential backoff. The third
	/// retryable failure, or any permanent failure, moves the job to terminal
	/// failed state until [`MemoryStore::retry_failed_job`] is called.
	///
	/// # Errors
	///
	/// Returns an error for an empty/oversized diagnostic, stale authority, or
	/// database failure.
	pub fn record_distillation_failure(
		&self,
		lease: &DistillationLease,
		error: &str,
		disposition: MemoryJobFailureDisposition,
	) -> Result<MemoryJobFailureOutcome, MemoryError> {
		let outcome = self.record_distillation_failure_inner(lease, error, disposition)?;
		lease.released.store(true, Ordering::Release);
		Ok(outcome)
	}

	fn record_distillation_failure_inner(
		&self,
		lease: &DistillationLease,
		error: &str,
		disposition: MemoryJobFailureDisposition,
	) -> Result<MemoryJobFailureOutcome, MemoryError> {
		self.validate_distillation_lease_origin(lease)?;
		validate_required(error, super::MAX_JOB_FAILURE_BYTES, "distillation failure")?;
		let mut connection = self.connection()?;
		Self::record_distillation_failure_with_connection(
			&mut connection,
			lease,
			error,
			disposition,
		)
	}

	fn record_distillation_failure_best_effort(
		&self,
		lease: &DistillationLease,
		error: &str,
		disposition: MemoryJobFailureDisposition,
	) -> Result<MemoryJobFailureOutcome, MemoryError> {
		self.validate_distillation_lease_origin(lease)?;
		validate_required(error, super::MAX_JOB_FAILURE_BYTES, "distillation failure")?;
		let mut connection = self.best_effort_connection()?;
		Self::record_distillation_failure_with_connection(
			&mut connection,
			lease,
			error,
			disposition,
		)
	}

	fn record_distillation_failure_with_connection(
		connection: &mut Connection,
		lease: &DistillationLease,
		error: &str,
		disposition: MemoryJobFailureDisposition,
	) -> Result<MemoryJobFailureOutcome, MemoryError> {
		let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
		let failures = transaction
			.query_row(
				"SELECT failure_count FROM distillation_jobs
				 WHERE id = ?1 AND state = 'running' AND claim_token = ?2",
				params![lease.job.id.to_string(), lease.token.to_string()],
				|row| row.get::<_, i64>(0),
			)
			.optional()?
			.ok_or(MemoryError::StaleDistillationLease {
				job_id: lease.job.id,
			})?;
		let outcome = transition_distillation_failure(
			&transaction,
			lease.job.id,
			lease.token,
			failures,
			error,
			disposition,
			Utc::now(),
		)?;
		transaction.commit()?;
		Ok(outcome)
	}

	/// Load the exact bounded source prefix under a valid worker lease.
	///
	/// # Errors
	///
	/// Returns stale/provenance, replay-bound, corruption, or database errors.
	pub fn distillation_source(
		&self,
		lease: &DistillationLease,
	) -> Result<Vec<SessionEvent>, MemoryError> {
		self.validate_distillation_lease_origin(lease)?;
		let connection = self.connection()?;
		validate_distillation_lease(&connection, lease, Utc::now())?;
		let sql = format!(
			"{} ORDER BY sequence ASC",
			event_select("WHERE session_id = ?1 AND sequence <= ?2")
		);
		let mut statement = connection.prepare(&sql)?;
		let rows = statement.query_map(
			params![
				lease.job.session_id.to_string(),
				i64::try_from(lease.job.through_sequence).map_err(|_| {
					MemoryError::Corrupt("distillation boundary exceeds SQLite range".to_string())
				})?
			],
			raw_event,
		)?;
		let mut events = Vec::new();
		let mut bytes = 0_usize;
		for row in rows {
			events.push(bounded_replay_event(row?, &mut bytes, events.len())?);
		}
		if events.len() > MAX_REPLAY_EVENTS || bytes > MAX_REPLAY_BYTES {
			return Err(MemoryError::Invalid(
				"distillation source exceeds replay bounds".to_string(),
			));
		}
		Ok(events)
	}

	/// Atomically apply bounded candidates and complete one distillation.
	///
	/// Job/source provenance plus candidate ordinal provide idempotency.
	///
	/// # Errors
	///
	/// Returns candidate validation, stale/provenance, workspace, overflow, or
	/// database errors.
	#[expect(
		clippy::too_many_lines,
		reason = "candidate application and job completion share one atomic transaction"
	)]
	pub fn complete_distillation(
		&self,
		lease: &DistillationLease,
		candidates: &[DistillationCandidateInput],
	) -> Result<Vec<DistillationCandidate>, MemoryError> {
		self.validate_distillation_lease_origin(lease)?;
		validate_candidates(candidates)?;
		let mut connection = self.connection()?;
		let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
		let now = Utc::now();
		validate_distillation_lease(&transaction, lease, now)?;
		let (workspace, device, inode, execution_token, execution_lease_until) = transaction
			.query_row(
				"SELECT workspace, workspace_device, workspace_inode,
			        execution_token, execution_lease_until
			 FROM sessions WHERE id = ?1",
				[lease.job.session_id.to_string()],
				|row| {
					Ok((
						row.get::<_, String>(0)?,
						row.get::<_, String>(1)?,
						row.get::<_, String>(2)?,
						row.get::<_, Option<String>>(3)?,
						row.get::<_, Option<String>>(4)?,
					))
				},
			)?;
		if let Some((_token, deadline)) =
			parse_session_claim(execution_token, execution_lease_until)?
			&& deadline > now
		{
			return Err(MemoryError::SessionBusy {
				session_id: lease.job.session_id,
				lease_until: deadline,
			});
		}
		let mut applied = Vec::new();
		for (index, candidate) in candidates.iter().enumerate() {
			let index_sql = i64::try_from(index).map_err(|_| {
				MemoryError::Invalid("distillation candidate index is too large".to_string())
			})?;
			match candidate {
				DistillationCandidateInput::Upsert {
					key,
					content,
					confidence,
					pinned,
				} => {
					let knowledge = apply_distilled_version(
						&transaction,
						lease,
						&workspace,
						&device,
						&inode,
						index_sql,
						key,
						content,
						*confidence,
						*pinned,
						now,
					)?;
					applied.push(DistillationCandidate::Upserted { knowledge });
				}
				DistillationCandidateInput::Tombstone { key, confidence } => {
					if let Some((knowledge_id, version)) = transaction
						.query_row(
							"SELECT id, active_version FROM knowledge
							 WHERE workspace_device = ?1 AND workspace_inode = ?2
							   AND key = ?3 AND pinned = 0",
							params![&device, &inode, key],
							|row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)),
						)
						.optional()?
					{
						let id = parse_uuid(&knowledge_id, "Knowledge ID")?;
						transaction.execute(
							"UPDATE knowledge
							 SET tombstoned = 1, tombstoned_at = ?2, updated_at = ?2
							 WHERE id = ?1",
							params![&knowledge_id, now.to_rfc3339()],
						)?;
						transaction.execute(
							"INSERT INTO knowledge_tombstones
							 (id, knowledge_id, confidence, provenance_session_id,
							  source_first_sequence, source_last_sequence, source_sha256,
							  distillation_job_id, candidate_index, origin, created_at)
							 VALUES (?1, ?2, ?3, ?4, 1, ?5, ?6, ?7, ?8,
							         'distillation', ?9)",
							params![
								Uuid::now_v7().to_string(),
								&knowledge_id,
								confidence,
								lease.job.session_id.to_string(),
								i64::try_from(lease.job.through_sequence).map_err(|_| {
									MemoryError::Corrupt(
										"distillation boundary exceeds SQLite range".to_string(),
									)
								})?,
								&lease.job.source.sha256,
								lease.job.id.to_string(),
								index_sql,
								now.to_rfc3339(),
							],
						)?;
						transaction.execute(
							"INSERT INTO distillation_results
							 (job_id, candidate_index, knowledge_id,
							  knowledge_version, created_at)
							 VALUES (?1, ?2, ?3, ?4, ?5)",
							params![
								lease.job.id.to_string(),
								index_sql,
								&knowledge_id,
								version,
								now.to_rfc3339(),
							],
						)?;
						applied.push(DistillationCandidate::Tombstoned {
							knowledge_id: id,
							key: key.clone(),
							confidence: *confidence,
						});
					}
				}
			}
		}
		let changed = transaction.execute(
			"UPDATE distillation_jobs
			 SET state = 'completed', claim_token = NULL, lease_until = NULL,
			     retry_after = NULL, last_error = NULL, failed_at = NULL,
			     completed_at = ?3, updated_at = ?3
			 WHERE id = ?1 AND state = 'running' AND claim_token = ?2",
			params![
				lease.job.id.to_string(),
				lease.token.to_string(),
				now.to_rfc3339(),
			],
		)?;
		if changed != 1 {
			return Err(MemoryError::StaleDistillationLease {
				job_id: lease.job.id,
			});
		}
		transaction.commit()?;
		lease.released.store(true, Ordering::Release);
		Ok(applied)
	}

	/// Explicitly return one worker claim to pending.
	///
	/// # Errors
	///
	/// Returns stale-authority or database errors.
	pub fn abandon_distillation(&self, lease: &DistillationLease) -> Result<(), MemoryError> {
		self.validate_distillation_lease_origin(lease)?;
		let changed = self.connection()?.execute(
			"UPDATE distillation_jobs
			 SET state = 'pending', claim_token = NULL, lease_until = NULL,
			     retry_after = NULL, updated_at = ?3
			 WHERE id = ?1 AND state = 'running' AND claim_token = ?2",
			params![
				lease.job.id.to_string(),
				lease.token.to_string(),
				Utc::now().to_rfc3339(),
			],
		)?;
		if changed != 1 {
			return Err(MemoryError::StaleDistillationLease {
				job_id: lease.job.id,
			});
		}
		lease.released.store(true, Ordering::Release);
		Ok(())
	}

	/// Run bounded retention, stale-claim recovery, asset GC, WAL checkpoint,
	/// and optional `VACUUM`.
	///
	/// Live Session leases and pinned Knowledge are never removed. Every
	/// mutation phase is capped by `max_rows`.
	///
	/// # Errors
	///
	/// Returns invalid policy, database, filesystem, or asset-GC errors.
	#[expect(
		clippy::too_many_lines,
		reason = "bounded maintenance phases share one recovery and retention transaction"
	)]
	pub fn maintain(&self, options: MaintenanceOptions) -> Result<MaintenanceReport, MemoryError> {
		validate_maintenance(options)?;
		let now = Utc::now();
		let session_cutoff = cutoff(now, options.retention.session_max_age, "Session retention")?;
		let knowledge_cutoff = cutoff(
			now,
			options.retention.knowledge_max_age,
			"Knowledge retention",
		)?;
		let tombstone_cutoff = cutoff(
			now,
			options.retention.tombstone_grace,
			"tombstone retention",
		)?;
		let limit = i64::try_from(options.max_rows)
			.map_err(|_| MemoryError::Invalid("maintenance row limit is too large".to_string()))?;
		let mut connection = self.connection()?;
		let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
		let now_text = now.to_rfc3339();
		let session_claims_recovered = transaction.execute(
			"UPDATE sessions
			 SET execution_token = NULL, execution_lease_until = NULL
			 WHERE id IN (
			   SELECT id FROM sessions
			   WHERE execution_token IS NOT NULL AND execution_lease_until <= ?1
			   ORDER BY execution_lease_until ASC, id ASC LIMIT ?2
			 )",
			params![&now_text, limit],
		)?;
		let mut compactions_recovered = 0_usize;
		for _ in 0..options.max_rows {
			if !recover_one_expired_compaction(&transaction, now)? {
				break;
			}
			compactions_recovered = compactions_recovered.saturating_add(1);
		}
		let mut distillations_recovered = 0_usize;
		for _ in 0..options.max_rows {
			if !recover_one_expired_distillation(&transaction, now)? {
				break;
			}
			distillations_recovered = distillations_recovered.saturating_add(1);
		}
		let knowledge_tombstoned = transaction.execute(
			"UPDATE knowledge
			 SET tombstoned = 1, tombstoned_at = ?1, updated_at = ?1
			 WHERE id IN (
			   SELECT id FROM knowledge
			   WHERE pinned = 0 AND tombstoned = 0 AND updated_at <= ?2
			   ORDER BY updated_at ASC, id ASC LIMIT ?3
			 )",
			params![&now_text, knowledge_cutoff.to_rfc3339(), limit],
		)?;
		let knowledge_removed = transaction.execute(
			"DELETE FROM knowledge
			 WHERE id IN (
			   SELECT id FROM knowledge
			   WHERE pinned = 0 AND tombstoned = 1 AND tombstoned_at <= ?1
			   ORDER BY tombstoned_at ASC, id ASC LIMIT ?2
			 )",
			params![tombstone_cutoff.to_rfc3339(), limit],
		)?;
		let versions_removed = transaction.execute(
			"DELETE FROM knowledge_versions
			 WHERE rowid IN (
			   SELECT rowid FROM (
			     SELECT v.rowid AS rowid, v.version, k.active_version,
			            ROW_NUMBER() OVER (
			              PARTITION BY v.knowledge_id ORDER BY v.version DESC
			            ) AS retained_rank
			     FROM knowledge_versions v
			     JOIN knowledge k ON k.id = v.knowledge_id
			     WHERE k.pinned = 0
			   )
			   WHERE version != active_version AND retained_rank > ?1
			   LIMIT ?2
			 )",
			params![
				i64::try_from(options.retention.versions_per_key).map_err(|_| {
					MemoryError::Invalid("Knowledge version retention is too large".to_string())
				})?,
				limit
			],
		)?;
		let sessions_removed = transaction.execute(
			"DELETE FROM sessions
			 WHERE id IN (
			   SELECT s.id FROM sessions s
			   WHERE s.updated_at <= ?1
			     AND (s.execution_token IS NULL OR s.execution_lease_until <= ?2)
			     AND NOT EXISTS (
			       SELECT 1 FROM compaction_jobs c
			       WHERE c.session_id = s.id AND c.state = 'running'
			         AND c.lease_until > ?2
			     )
			     AND NOT EXISTS (
			       SELECT 1 FROM distillation_jobs d
			       WHERE d.session_id = s.id AND d.state = 'running'
			         AND d.lease_until > ?2
			     )
			   ORDER BY s.updated_at ASC, s.id ASC LIMIT ?3
			 )",
			params![session_cutoff.to_rfc3339(), &now_text, limit],
		)?;
		transaction.commit()?;
		let assets = self.gc_assets(options.retention.asset_grace)?;
		connection.execute_batch("PRAGMA optimize;")?;
		let (busy, _log_frames, _checkpointed): (i64, i64, i64) =
			connection.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |row| {
				Ok((row.get(0)?, row.get(1)?, row.get(2)?))
			})?;
		if options.vacuum {
			connection.execute_batch("VACUUM;")?;
		}
		drop(connection);
		sync_database(&self.database)?;
		Ok(MaintenanceReport {
			session_claims_recovered,
			compactions_recovered,
			distillations_recovered,
			sessions_removed,
			knowledge_tombstoned,
			knowledge_removed,
			versions_removed,
			assets,
			wal_busy: busy != 0,
			vacuumed: options.vacuum,
		})
	}
}

struct TurnWeight {
	turn_id: String,
	size: i64,
	seen: i64,
	end_sequence: u64,
	tokens: u64,
}

struct RawDistillation {
	id: String,
	session_id: String,
	through_sequence: i64,
	source_event_count: i64,
	source_first_event_id: String,
	source_last_event_id: String,
	source_sha256: String,
	state: String,
	claim_token: Option<String>,
	lease_until: Option<String>,
	failure_count: i64,
	retry_after: Option<String>,
	last_error: Option<String>,
	failed_at: Option<String>,
	created_at: String,
	updated_at: String,
	completed_at: Option<String>,
}

fn raw_distillation(row: &rusqlite::Row<'_>) -> rusqlite::Result<RawDistillation> {
	Ok(RawDistillation {
		id: row.get(0)?,
		session_id: row.get(1)?,
		through_sequence: row.get(2)?,
		source_event_count: row.get(3)?,
		source_first_event_id: row.get(4)?,
		source_last_event_id: row.get(5)?,
		source_sha256: row.get(6)?,
		state: row.get(7)?,
		claim_token: row.get(8)?,
		lease_until: row.get(9)?,
		failure_count: row.get(10)?,
		retry_after: row.get(11)?,
		last_error: row.get(12)?,
		failed_at: row.get(13)?,
		created_at: row.get(14)?,
		updated_at: row.get(15)?,
		completed_at: row.get(16)?,
	})
}

impl TryFrom<RawDistillation> for DistillationJob {
	type Error = MemoryError;

	#[expect(
		clippy::too_many_lines,
		reason = "one conversion validates the durable row's cross-field state atomically"
	)]
	fn try_from(raw: RawDistillation) -> Result<Self, Self::Error> {
		let through_sequence = u64::try_from(raw.through_sequence)
			.map_err(|_| MemoryError::Corrupt("negative distillation boundary".to_string()))?;
		let event_count = u64::try_from(raw.source_event_count)
			.map_err(|_| MemoryError::Corrupt("negative distillation source count".to_string()))?;
		if event_count != through_sequence {
			return Err(MemoryError::Corrupt(format!(
				"distillation {} source count does not match its boundary",
				raw.id
			)));
		}
		let has_claim = raw.claim_token.is_some() && raw.lease_until.is_some();
		let has_failure = raw.last_error.is_some() && raw.failed_at.is_some();
		let failures = u32::try_from(raw.failure_count).map_err(|_| {
			MemoryError::Corrupt(format!(
				"distillation {} has an invalid failure count",
				raw.id
			))
		})?;
		let valid_history = (failures == 0 && raw.last_error.is_none())
			|| (failures > 0 && raw.last_error.is_some());
		let state = match raw.state.as_str() {
			"pending"
				if !has_claim
					&& raw.completed_at.is_none()
					&& raw.failed_at.is_none()
					&& valid_history
					&& (raw.retry_after.is_none() || failures > 0) =>
			{
				DistillationState::Pending
			}
			"running"
				if has_claim
					&& raw.retry_after.is_none()
					&& raw.completed_at.is_none()
					&& raw.failed_at.is_none()
					&& valid_history =>
			{
				DistillationState::Running
			}
			"completed"
				if !has_claim
					&& raw.retry_after.is_none()
					&& raw.last_error.is_none()
					&& raw.completed_at.is_some()
					&& raw.failed_at.is_none() =>
			{
				DistillationState::Completed
			}
			"failed"
				if !has_claim
					&& raw.retry_after.is_none()
					&& raw.completed_at.is_none()
					&& has_failure && failures > 0 =>
			{
				DistillationState::Failed
			}
			"pending" | "running" | "completed" | "failed" => {
				return Err(MemoryError::Corrupt(format!(
					"distillation state {:?} has inconsistent claim metadata",
					raw.state
				)));
			}
			_ => {
				return Err(MemoryError::Corrupt(format!(
					"unknown distillation state {:?}",
					raw.state
				)));
			}
		};
		Ok(Self {
			id: parse_uuid(&raw.id, "distillation ID")?,
			session_id: parse_uuid(&raw.session_id, "distillation session ID")?,
			through_sequence,
			source: TranscriptProvenance {
				event_count,
				first_event_id: parse_uuid(
					&raw.source_first_event_id,
					"distillation first event ID",
				)?,
				last_event_id: parse_uuid(&raw.source_last_event_id, "distillation last event ID")?,
				sha256: raw.source_sha256,
			},
			state,
			failures,
			retry_after: raw
				.retry_after
				.as_deref()
				.map(|value| parse_time(value, "distillation retry deadline"))
				.transpose()?,
			last_error: raw.last_error,
			failed_at: raw
				.failed_at
				.as_deref()
				.map(|value| parse_time(value, "distillation failure time"))
				.transpose()?,
			created_at: parse_time(&raw.created_at, "distillation creation time")?,
			updated_at: parse_time(&raw.updated_at, "distillation update time")?,
			completed_at: raw
				.completed_at
				.as_deref()
				.map(|value| parse_time(value, "distillation completion time"))
				.transpose()?,
		})
	}
}

fn distillation_select(clause: &str) -> String {
	format!(
		"SELECT j.id, j.session_id, j.through_sequence, j.source_event_count,
		        j.source_first_event_id, j.source_last_event_id, j.source_sha256,
		        j.state, j.claim_token, j.lease_until, j.failure_count,
		        j.retry_after, j.last_error, j.failed_at, j.created_at,
		        j.updated_at, j.completed_at
		 FROM distillation_jobs j {clause}"
	)
}

fn transition_distillation_failure(
	transaction: &Transaction<'_>,
	job_id: Uuid,
	token: Uuid,
	current_failures: i64,
	error: &str,
	disposition: MemoryJobFailureDisposition,
	now: DateTime<Utc>,
) -> Result<MemoryJobFailureOutcome, MemoryError> {
	let transition = job_failure_transition(current_failures, disposition, now)?;
	let changed = transaction.execute(
		"UPDATE distillation_jobs
		 SET state = ?3, claim_token = NULL, lease_until = NULL,
		     failure_count = ?4, retry_after = ?5, last_error = ?6,
		     failed_at = ?7, updated_at = ?8
		 WHERE id = ?1 AND state = 'running' AND claim_token = ?2",
		params![
			job_id.to_string(),
			token.to_string(),
			transition.state,
			transition.failures,
			transition.retry_after,
			error,
			transition.failed_at,
			now.to_rfc3339(),
		],
	)?;
	if changed != 1 {
		return Err(MemoryError::StaleDistillationLease { job_id });
	}
	Ok(transition.outcome)
}

fn recover_one_expired_distillation(
	transaction: &Transaction<'_>,
	now: DateTime<Utc>,
) -> Result<bool, MemoryError> {
	let row = transaction
		.query_row(
			"SELECT id, claim_token, failure_count
			 FROM distillation_jobs
			 WHERE state = 'running' AND lease_until <= ?1
			 ORDER BY lease_until ASC, id ASC LIMIT 1",
			[now.to_rfc3339()],
			|row| {
				Ok((
					row.get::<_, String>(0)?,
					row.get::<_, String>(1)?,
					row.get::<_, i64>(2)?,
				))
			},
		)
		.optional()?;
	let Some((job_id, token, current_failures)) = row else {
		return Ok(false);
	};
	let job_id = parse_uuid(&job_id, "expired distillation ID")?;
	let token = parse_uuid(&token, "expired distillation claim token")?;
	transition_distillation_failure(
		transaction,
		job_id,
		token,
		current_failures,
		"worker lease expired before completion",
		MemoryJobFailureDisposition::Retry,
		now,
	)?;
	Ok(true)
}

fn validate_distillation_lease(
	connection: &rusqlite::Connection,
	lease: &DistillationLease,
	now: DateTime<Utc>,
) -> Result<(), MemoryError> {
	let raw = connection
		.query_row(
			&distillation_select("WHERE j.id = ?1"),
			[lease.job.id.to_string()],
			raw_distillation,
		)
		.optional()?
		.ok_or(MemoryError::NotFound {
			entity: "distillation job",
			id: lease.job.id,
		})?;
	let deadline = raw
		.lease_until
		.as_deref()
		.map(|value| parse_time(value, "distillation lease deadline"))
		.transpose()?;
	let expected_token = lease.token.to_string();
	if raw.state != "running"
		|| raw.claim_token.as_deref() != Some(expected_token.as_str())
		|| deadline.is_none_or(|deadline| deadline <= now)
	{
		return Err(MemoryError::StaleDistillationLease {
			job_id: lease.job.id,
		});
	}
	let job = DistillationJob::try_from(raw)?;
	if job.source != lease.job.source || job.through_sequence != lease.job.through_sequence {
		return Err(MemoryError::StaleDistillationLease {
			job_id: lease.job.id,
		});
	}
	let current = transcript_provenance(
		connection,
		&job.session_id.to_string(),
		i64::try_from(job.through_sequence)
			.map_err(|_| MemoryError::Corrupt("distillation boundary overflow".to_string()))?,
	)?;
	if current != job.source {
		return Err(MemoryError::StaleDistillationLease {
			job_id: lease.job.id,
		});
	}
	Ok(())
}

#[expect(
	clippy::too_many_arguments,
	reason = "all durable provenance and workspace fields must enter one transaction boundary"
)]
#[expect(
	clippy::too_many_lines,
	reason = "Knowledge row and immutable version insertion are one transactional operation"
)]
fn apply_distilled_version(
	transaction: &Transaction<'_>,
	lease: &DistillationLease,
	workspace: &str,
	device: &str,
	inode: &str,
	candidate_index: i64,
	key: &str,
	content: &str,
	confidence: f64,
	pinned: bool,
	now: DateTime<Utc>,
) -> Result<Knowledge, MemoryError> {
	let existing = transaction
		.query_row(
			"SELECT id, pinned, created_at
			 FROM knowledge
			 WHERE workspace_device = ?1 AND workspace_inode = ?2 AND key = ?3",
			params![device, inode, key],
			|row| {
				Ok((
					row.get::<_, String>(0)?,
					row.get::<_, bool>(1)?,
					row.get::<_, String>(2)?,
				))
			},
		)
		.optional()?;
	let (id, effective_pinned, created_at) =
		if let Some((id, existing_pinned, created_at)) = existing {
			(
				parse_uuid(&id, "Knowledge ID")?,
				existing_pinned || pinned,
				parse_time(&created_at, "Knowledge creation time")?,
			)
		} else {
			(Uuid::now_v7(), pinned, now)
		};
	let version: i64 = transaction.query_row(
		"SELECT COALESCE(MAX(version), 0) + 1
		 FROM knowledge_versions WHERE knowledge_id = ?1",
		[id.to_string()],
		|row| row.get(0),
	)?;
	let public_version = u32::try_from(version)
		.map_err(|_| MemoryError::Corrupt("Knowledge version overflow".to_string()))?;
	transaction.execute(
		"INSERT INTO knowledge
		 (id, workspace, workspace_device, workspace_inode, legacy_identity,
		  key, active_version, pinned, tombstoned, tombstoned_at,
		  created_at, updated_at)
		 VALUES (?1, ?2, ?3, ?4, 0, ?5, ?6, ?7, 0, NULL, ?8, ?9)
		 ON CONFLICT(workspace_device, workspace_inode, key) DO UPDATE SET
		   workspace = excluded.workspace,
		   active_version = excluded.active_version,
		   pinned = MAX(knowledge.pinned, excluded.pinned),
		   tombstoned = 0,
		   tombstoned_at = NULL,
		   updated_at = excluded.updated_at",
		params![
			id.to_string(),
			workspace,
			device,
			inode,
			key,
			version,
			effective_pinned,
			created_at.to_rfc3339(),
			now.to_rfc3339(),
		],
	)?;
	transaction.execute(
		"INSERT INTO knowledge_versions
		 (knowledge_id, version, content, confidence, source_session_id,
		  provenance_session_id, source_first_sequence, source_last_sequence,
		  source_sha256, distillation_job_id, candidate_index, created_at)
		 VALUES (?1, ?2, ?3, ?4, ?5, ?5, 1, ?6, ?7, ?8, ?9, ?10)",
		params![
			id.to_string(),
			version,
			content,
			confidence,
			lease.job.session_id.to_string(),
			i64::try_from(lease.job.through_sequence).map_err(|_| {
				MemoryError::Corrupt("distillation boundary exceeds SQLite range".to_string())
			})?,
			&lease.job.source.sha256,
			lease.job.id.to_string(),
			candidate_index,
			now.to_rfc3339(),
		],
	)?;
	transaction.execute(
		"INSERT INTO distillation_results
		 (job_id, candidate_index, knowledge_id, knowledge_version, created_at)
		 VALUES (?1, ?2, ?3, ?4, ?5)",
		params![
			lease.job.id.to_string(),
			candidate_index,
			id.to_string(),
			version,
			now.to_rfc3339(),
		],
	)?;
	Ok(Knowledge {
		id,
		workspace: PathBuf::from(workspace),
		workspace_identity: super::parse_workspace_identity(
			Some(device.to_string()),
			Some(inode.to_string()),
		)?,
		key: key.to_string(),
		active_version: public_version,
		content: content.to_string(),
		confidence,
		source_session_id: Some(lease.job.session_id),
		pinned: effective_pinned,
		tombstoned: false,
		created_at,
		updated_at: now,
	})
}

fn validate_candidates(candidates: &[DistillationCandidateInput]) -> Result<(), MemoryError> {
	if candidates.len() > MAX_DISTILLATION_CANDIDATES {
		return Err(MemoryError::Invalid(format!(
			"distillation exceeds {MAX_DISTILLATION_CANDIDATES} candidate limit"
		)));
	}
	let mut keys = BTreeSet::new();
	for candidate in candidates {
		let (key, confidence) = match candidate {
			DistillationCandidateInput::Upsert {
				key,
				content,
				confidence,
				..
			} => {
				validate_required(content, MAX_KNOWLEDGE_BYTES, "Knowledge content")?;
				(key, confidence)
			}
			DistillationCandidateInput::Tombstone { key, confidence } => (key, confidence),
		};
		validate_required(key, MAX_KNOWLEDGE_KEY_BYTES, "Knowledge key")?;
		if !confidence.is_finite() || !(0.0..=1.0).contains(confidence) {
			return Err(MemoryError::Invalid(format!(
				"Knowledge candidate {key:?} confidence must be finite and in 0.0..=1.0"
			)));
		}
		if !keys.insert(key.as_str()) {
			return Err(MemoryError::Invalid(format!(
				"distillation contains duplicate Knowledge key {key:?}"
			)));
		}
	}
	Ok(())
}

fn validate_maintenance(options: MaintenanceOptions) -> Result<(), MemoryError> {
	if !(1..=MAX_MAINTENANCE_ROWS).contains(&options.max_rows) {
		return Err(MemoryError::Invalid(format!(
			"maintenance row limit must be in 1..={MAX_MAINTENANCE_ROWS}"
		)));
	}
	if options.retention.versions_per_key == 0 || options.retention.versions_per_key > 1_000 {
		return Err(MemoryError::Invalid(
			"Knowledge versions per key must be in 1..=1000".to_string(),
		));
	}
	for (name, duration) in [
		("Session retention", options.retention.session_max_age),
		("Knowledge retention", options.retention.knowledge_max_age),
		("tombstone grace", options.retention.tombstone_grace),
		("asset grace", options.retention.asset_grace),
	] {
		if duration.is_zero() {
			return Err(MemoryError::Invalid(format!("{name} must be positive")));
		}
	}
	Ok(())
}

fn cutoff(
	now: DateTime<Utc>,
	duration: Duration,
	name: &str,
) -> Result<DateTime<Utc>, MemoryError> {
	let duration = chrono::Duration::from_std(duration)
		.map_err(|_| MemoryError::Invalid(format!("{name} duration is invalid")))?;
	now.checked_sub_signed(duration)
		.ok_or_else(|| MemoryError::Invalid(format!("{name} cutoff overflow")))
}

fn percent_ceil(value: u64, percent: u64) -> Result<u64, MemoryError> {
	value
		.checked_mul(percent)
		.and_then(|scaled| scaled.checked_add(99))
		.map(|scaled| scaled / 100)
		.ok_or_else(|| MemoryError::Invalid("compaction token threshold overflow".to_string()))
}

fn percent_floor(value: u64, percent: u64) -> Result<u64, MemoryError> {
	value
		.checked_mul(percent)
		.map(|scaled| scaled / 100)
		.ok_or_else(|| MemoryError::Invalid("compaction token target overflow".to_string()))
}

fn sync_database(database: &std::path::Path) -> Result<(), MemoryError> {
	let file = OpenOptions::new()
		.read(true)
		.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
		.open(database)
		.map_err(|source| MemoryError::Io {
			operation: "open memory database for maintenance sync",
			path: database.to_path_buf(),
			source,
		})?;
	file.sync_all().map_err(|source| MemoryError::Io {
		operation: "sync memory database after maintenance",
		path: database.to_path_buf(),
		source,
	})?;
	let parent = database.parent().ok_or_else(|| {
		MemoryError::Invalid("memory database has no parent directory".to_string())
	})?;
	let directory = fs::File::open(parent).map_err(|source| MemoryError::Io {
		operation: "open memory directory for maintenance sync",
		path: parent.to_path_buf(),
		source,
	})?;
	directory.sync_all().map_err(|source| MemoryError::Io {
		operation: "sync memory directory after maintenance",
		path: parent.to_path_buf(),
		source,
	})
}

#[cfg(test)]
#[expect(
	clippy::unwrap_used,
	reason = "lifecycle tests use panic-on-fixture-failure assertions"
)]
mod tests {
	use super::*;
	use crate::{
		home::EmelexHome,
		memory::{SessionEventInput, SessionEventKind},
	};

	fn store() -> (tempfile::TempDir, MemoryStore) {
		let directory = tempfile::tempdir().unwrap();
		let home = EmelexHome::prepare(&directory.path().join("home")).unwrap();
		let store = MemoryStore::open(&home).unwrap();
		(directory, store)
	}

	fn queued_distillation(store: &MemoryStore) -> (tempfile::TempDir, DistillationJob) {
		let workspace = tempfile::tempdir().unwrap();
		let session = store.start_session(workspace.path(), None).unwrap();
		store
			.append_event(
				session.id,
				SessionEventKind::UserMessage,
				&serde_json::json!({"text": "remember this"}),
			)
			.unwrap();
		let job = store.queue_distillation(session.id).unwrap().unwrap();
		(workspace, job)
	}

	fn make_distillation_retry_eligible(store: &MemoryStore, id: Uuid) {
		store
			.connection()
			.unwrap()
			.execute(
				"UPDATE distillation_jobs SET retry_after = ?2 WHERE id = ?1",
				params![
					id.to_string(),
					(Utc::now() - chrono::Duration::seconds(1)).to_rfc3339()
				],
			)
			.unwrap();
	}

	fn assert_drop_does_not_wait_for_database<T>(store: &MemoryStore, lease: T) {
		let mut blocker = store.connection().unwrap();
		let transaction = blocker
			.transaction_with_behavior(TransactionBehavior::Immediate)
			.unwrap();
		let started = std::time::Instant::now();
		drop(lease);
		let elapsed = started.elapsed();
		assert!(
			elapsed < Duration::from_secs(1),
			"lease drop waited for database lock for {elapsed:?}"
		);
		drop(transaction);
	}

	#[test]
	fn compaction_plan_triggers_at_eighty_and_preserves_four_turns() {
		let (_directory, store) = store();
		let workspace = tempfile::tempdir().unwrap();
		let session = store.start_session(workspace.path(), None).unwrap();
		for index in 0..10 {
			store
				.append_event(
					session.id,
					SessionEventKind::UserMessage,
					&serde_json::json!({"text": "x".repeat(400), "index": index}),
				)
				.unwrap();
		}
		let policy = CompactionPolicy::new(1_000).unwrap();
		assert!(
			store
				.plan_compaction(session.id, 799, policy)
				.unwrap()
				.is_none()
		);
		let plan = store
			.plan_compaction(session.id, 900, policy)
			.unwrap()
			.unwrap();
		assert!(plan.through_sequence <= 6);
		assert_eq!(plan.trigger_tokens, 800);
		assert_eq!(plan.target_tokens, 500);
	}

	#[test]
	fn distillation_is_idempotent_and_provenance_survives_session_deletion() {
		let (_directory, store) = store();
		let workspace = tempfile::tempdir().unwrap();
		let session = store.start_session(workspace.path(), None).unwrap();
		store
			.append_event(
				session.id,
				SessionEventKind::UserMessage,
				&serde_json::json!({"text": "use cargo test"}),
			)
			.unwrap();
		let queued = store.queue_distillation(session.id).unwrap().unwrap();
		assert_eq!(
			store.queue_distillation(session.id).unwrap().unwrap().id,
			queued.id
		);
		let lease = store.claim_distillation().unwrap().unwrap();
		let applied = store
			.complete_distillation(
				&lease,
				&[DistillationCandidateInput::Upsert {
					key: "test-command".to_string(),
					content: "cargo test".to_string(),
					confidence: 0.95,
					pinned: false,
				}],
			)
			.unwrap();
		let knowledge_id = applied
			.iter()
			.find_map(|candidate| match candidate {
				DistillationCandidate::Upserted { knowledge } => Some(knowledge.id),
				DistillationCandidate::Tombstoned { .. } => None,
			})
			.unwrap();
		store.delete_session(session.id).unwrap();
		let history = store.knowledge_history(knowledge_id, None, 10).unwrap();
		assert_eq!(
			history[0].provenance.as_ref().unwrap().session_id,
			session.id
		);
		assert_eq!(history[0].source_session_id, None);
	}

	#[test]
	fn distilled_tombstone_keeps_confidence_and_provenance() {
		let (_directory, store) = store();
		let workspace = tempfile::tempdir().unwrap();
		let knowledge = store
			.remember(workspace.path(), "obsolete", "old", None)
			.unwrap();
		let session = store.start_session(workspace.path(), None).unwrap();
		store
			.append_event(
				session.id,
				SessionEventKind::UserMessage,
				&serde_json::json!({"text": "obsolete is no longer true"}),
			)
			.unwrap();
		store.queue_distillation(session.id).unwrap();
		let lease = store.claim_distillation().unwrap().unwrap();
		let applied = store
			.complete_distillation(
				&lease,
				&[DistillationCandidateInput::Tombstone {
					key: "obsolete".to_string(),
					confidence: 0.8,
				}],
			)
			.unwrap();
		assert!(matches!(
			applied.as_slice(),
			[DistillationCandidate::Tombstoned {
				knowledge_id,
				..
			}] if *knowledge_id == knowledge.id
		));
		assert!(store.knowledge(knowledge.id).unwrap().tombstoned);
		store.delete_session(session.id).unwrap();
		let (confidence, provenance): (f64, String) = store
			.connection()
			.unwrap()
			.query_row(
				"SELECT confidence, provenance_session_id
				 FROM knowledge_tombstones WHERE knowledge_id = ?1",
				[knowledge.id.to_string()],
				|row| Ok((row.get(0)?, row.get(1)?)),
			)
			.unwrap();
		assert!((confidence - 0.8).abs() < f64::EPSILON);
		assert_eq!(provenance, session.id.to_string());
	}

	#[test]
	fn retention_never_removes_live_session_or_pinned_knowledge() {
		let (_directory, store) = store();
		let workspace = tempfile::tempdir().unwrap();
		let live = store.start_session(workspace.path(), None).unwrap();
		let lease = store.claim_session(live.id, workspace.path()).unwrap();
		let knowledge = store
			.remember(workspace.path(), "pinned", "keep", None)
			.unwrap();
		store
			.set_knowledge_pinned(workspace.path(), knowledge.id, true)
			.unwrap();
		let old = (Utc::now() - chrono::Duration::days(400)).to_rfc3339();
		let connection = store.connection().unwrap();
		connection
			.execute("UPDATE sessions SET updated_at = ?1", [&old])
			.unwrap();
		connection
			.execute("UPDATE knowledge SET updated_at = ?1", [&old])
			.unwrap();
		drop(connection);
		let report = store
			.maintain(MaintenanceOptions {
				retention: RetentionPolicy {
					session_max_age: Duration::from_secs(1),
					knowledge_max_age: Duration::from_secs(1),
					tombstone_grace: Duration::from_secs(1),
					versions_per_key: 1,
					asset_grace: Duration::from_secs(1),
				},
				max_rows: 100,
				vacuum: false,
			})
			.unwrap();
		assert_eq!(report.sessions_removed, 0);
		assert_eq!(report.knowledge_tombstoned, 0);
		assert!(store.session(live.id).is_ok());
		assert!(!store.knowledge(knowledge.id).unwrap().tombstoned);
		store.release_session(&lease).unwrap();
	}

	#[test]
	fn reserved_summary_is_rejected_by_legacy_append() {
		let (_directory, store) = store();
		let workspace = tempfile::tempdir().unwrap();
		let session = store.start_session(workspace.path(), None).unwrap();
		assert!(
			store
				.append_event(
					session.id,
					SessionEventKind::Summary,
					&serde_json::json!({"text": "forged"}),
				)
				.is_err()
		);
	}

	#[test]
	fn drop_releases_session_lease() {
		let (_directory, store) = store();
		let workspace = tempfile::tempdir().unwrap();
		let session = store.start_session(workspace.path(), None).unwrap();
		{
			let _lease = store.claim_session(session.id, workspace.path()).unwrap();
		}
		assert!(store.claim_session(session.id, workspace.path()).is_ok());
	}

	#[test]
	fn session_lease_drop_does_not_wait_for_database_lock() {
		let (_directory, store) = store();
		let workspace = tempfile::tempdir().unwrap();
		let session = store.start_session(workspace.path(), None).unwrap();
		let lease = store.claim_session(session.id, workspace.path()).unwrap();
		assert_drop_does_not_wait_for_database(&store, lease);
	}

	#[test]
	fn compaction_lease_drop_does_not_wait_for_database_lock() {
		let (_directory, store) = store();
		let workspace = tempfile::tempdir().unwrap();
		let session = store.start_session(workspace.path(), None).unwrap();
		store
			.append_event(
				session.id,
				SessionEventKind::UserMessage,
				&serde_json::json!({"text": "hello"}),
			)
			.unwrap();
		store.queue_compaction(session.id, 1).unwrap();
		let lease = store.claim_compaction().unwrap().unwrap();
		assert_drop_does_not_wait_for_database(&store, lease);
	}

	#[test]
	fn distillation_lease_drop_does_not_wait_for_database_lock() {
		let (_directory, store) = store();
		let (_workspace, _job) = queued_distillation(&store);
		let lease = store.claim_distillation().unwrap().unwrap();
		assert_drop_does_not_wait_for_database(&store, lease);
	}

	#[test]
	fn complete_compaction_refuses_live_session_lease() {
		let (_directory, store) = store();
		let workspace = tempfile::tempdir().unwrap();
		let session = store.start_session(workspace.path(), None).unwrap();
		store
			.append_event(
				session.id,
				SessionEventKind::UserMessage,
				&serde_json::json!({"text": "hello"}),
			)
			.unwrap();
		store.queue_compaction(session.id, 1).unwrap();
		let compaction = store.claim_compaction().unwrap().unwrap();
		let lease = store.claim_session(session.id, workspace.path()).unwrap();
		assert!(
			store
				.complete_compaction(&compaction, &serde_json::json!({"text": "summary"}))
				.is_err()
		);
		store.release_session(&lease).unwrap();
	}

	#[test]
	fn compaction_claim_skips_live_session_lease() {
		let (_directory, store) = store();
		let workspace = tempfile::tempdir().unwrap();
		let session = store.start_session(workspace.path(), None).unwrap();
		store
			.append_event(
				session.id,
				SessionEventKind::UserMessage,
				&serde_json::json!({"text": "hello"}),
			)
			.unwrap();
		let queued = store.queue_compaction(session.id, 1).unwrap();
		let lease = store.claim_session(session.id, workspace.path()).unwrap();

		assert!(store.claim_compaction().unwrap().is_none());
		store.release_session(&lease).unwrap();
		assert_eq!(
			store.claim_compaction().unwrap().unwrap().job().id,
			queued.id
		);
	}

	#[test]
	fn third_distillation_failure_dead_letters_and_manual_retry_resets_it() {
		let (_directory, store) = store();
		let (_workspace, job) = queued_distillation(&store);
		for expected in 1..=2 {
			let lease = store.claim_distillation().unwrap().unwrap();
			let outcome = store
				.record_distillation_failure(
					&lease,
					"model returned invalid JSON",
					MemoryJobFailureDisposition::Retry,
				)
				.unwrap();
			assert!(matches!(
				outcome,
				MemoryJobFailureOutcome::RetryScheduled { failures, .. }
					if failures == expected
			));
			make_distillation_retry_eligible(&store, job.id);
		}
		let lease = store.claim_distillation().unwrap().unwrap();
		let outcome = store
			.record_distillation_failure(
				&lease,
				"model returned invalid JSON",
				MemoryJobFailureDisposition::Retry,
			)
			.unwrap();
		assert!(matches!(
			outcome,
			MemoryJobFailureOutcome::Failed { failures: 3, .. }
		));
		assert!(store.claim_distillation().unwrap().is_none());
		assert_eq!(store.status().unwrap().failed_distillations, 1);
		store.retry_failed_job(job.id).unwrap();
		let retried = store.claim_distillation().unwrap().unwrap();
		assert_eq!((retried.job().id, retried.job().failures), (job.id, 0));
		store.abandon_distillation(&retried).unwrap();
	}

	#[test]
	fn cancelled_distillation_release_does_not_count_failure() {
		let (_directory, store) = store();
		let (_workspace, job) = queued_distillation(&store);
		let lease = store.claim_distillation().unwrap().unwrap();
		store.abandon_distillation(&lease).unwrap();
		let reclaimed = store.claim_distillation().unwrap().unwrap();
		assert_eq!((reclaimed.job().id, reclaimed.job().failures), (job.id, 0));
		store.abandon_distillation(&reclaimed).unwrap();
	}

	#[test]
	fn recovered_distillation_rejects_stale_failure_writer() {
		let (_directory, store) = store();
		let (_workspace, _job) = queued_distillation(&store);
		let lease = store.claim_distillation().unwrap().unwrap();
		store
			.connection()
			.unwrap()
			.execute(
				"UPDATE distillation_jobs SET lease_until = ?2 WHERE id = ?1",
				params![
					lease.job().id.to_string(),
					(Utc::now() - chrono::Duration::seconds(1)).to_rfc3339()
				],
			)
			.unwrap();
		assert!(store.claim_distillation().unwrap().is_none());
		assert!(matches!(
			store.record_distillation_failure(
				&lease,
				"late worker output",
				MemoryJobFailureDisposition::Retry
			),
			Err(MemoryError::StaleDistillationLease { .. })
		));
	}

	#[test]
	fn atomic_turn_can_reference_content_addressed_asset() {
		let (_directory, store) = store();
		let workspace = tempfile::tempdir().unwrap();
		let session = store.start_session(workspace.path(), None).unwrap();
		let asset = store
			.store_asset_bytes(super::super::AssetKind::Image, b"image")
			.unwrap();
		let mut lease = store.claim_session(session.id, workspace.path()).unwrap();
		store
			.append_turn(
				&mut lease,
				&[SessionEventInput::new(
					SessionEventKind::UserMessage,
					serde_json::json!({"asset": asset.sha256()}),
				)
				.with_assets(vec![asset.clone()])],
			)
			.unwrap();
		assert_eq!(store.session_assets(session.id).unwrap(), vec![asset]);
	}
}
