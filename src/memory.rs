//! Durable sessions, compaction work, and versioned workspace Knowledge.

use std::{
	fmt,
	fs::{self, OpenOptions},
	io,
	os::{
		fd::AsRawFd as _,
		unix::{
			ffi::OsStrExt as _,
			fs::{MetadataExt as _, OpenOptionsExt as _, PermissionsExt as _},
		},
	},
	path::{Path, PathBuf},
	sync::{
		Mutex,
		atomic::{AtomicBool, Ordering},
	},
	time::Duration,
};

use chrono::{DateTime, Utc};
use rusqlite::{Connection, OpenFlags, OptionalExtension as _, TransactionBehavior, params};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest as _, Sha256};
use uuid::Uuid;

use crate::{
	home::{EmelexHome, HomeError},
	model::{InstalledModel, ModelRef, ModelSnapshotId},
};

mod adapter;
mod assets;
mod lifecycle;

pub use adapter::{
	AgentTurnRecoveryReport, DurableAgentSession, DurableSessionError, SessionSnapshot,
	UncertainToolCall,
};
pub use assets::{AssetGcReport, AssetKind, AssetRecord, AssetRef, MAX_ASSET_BYTES};
pub use lifecycle::{
	CompactionPlan, CompactionPolicy, DistillationCandidate, DistillationCandidateInput,
	DistillationJob, DistillationLease, DistillationState, MaintenanceOptions, MaintenanceReport,
	RetentionPolicy,
};

/// Lazy bridge from installed-model removal to durable Session references.
///
/// Construction only captures the database path. `SQLite` is opened read-only
/// when a removal or quarantine sweep actually asks about one snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MemorySnapshotReferenceGuard {
	database: PathBuf,
}

impl MemorySnapshotReferenceGuard {
	/// Capture the durable-memory database path without opening or creating it.
	pub fn new(home: &EmelexHome) -> Self {
		Self {
			database: home.database_file(),
		}
	}

	/// Whether any retained Session binds the exact immutable snapshot.
	///
	/// A missing database means no durable references. Any other open, schema,
	/// or query failure is returned so callers can fail closed.
	///
	/// # Errors
	///
	/// Returns filesystem or read-only `SQLite` failures.
	pub fn is_snapshot_referenced(&self, snapshot: &ModelSnapshotId) -> Result<bool, MemoryError> {
		let exists = self
			.database
			.try_exists()
			.map_err(|source| MemoryError::Io {
				operation: "inspect memory database for model references",
				path: self.database.clone(),
				source,
			})?;
		if !exists {
			return Ok(false);
		}
		let connection = open_snapshot_reference_connection(&self.database)?;
		connection
			.query_row(
				"SELECT EXISTS(
				   SELECT 1 FROM sessions WHERE model_snapshot = ?1 LIMIT 1
				 )",
				[snapshot.to_string()],
				|row| row.get(0),
			)
			.map_err(MemoryError::Database)
	}
}

impl crate::models::SnapshotReferenceGuard for MemorySnapshotReferenceGuard {
	fn is_referenced(
		&self,
		snapshot: &ModelSnapshotId,
	) -> Result<bool, crate::models::SnapshotReferenceError> {
		self.is_snapshot_referenced(snapshot)
			.map_err(|error| crate::models::SnapshotReferenceError::new(error.to_string()))
	}
}

fn open_snapshot_reference_connection(database: &Path) -> Result<Connection, MemoryError> {
	let flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NOFOLLOW;
	let connection = Connection::open_with_flags(database, flags)?;
	connection.busy_timeout(Duration::from_secs(5))?;
	configure_memory_temp_store(&connection)?;
	connection.pragma_update(None, "trusted_schema", false)?;
	Ok(connection)
}

fn configure_memory_temp_store(connection: &Connection) -> Result<(), MemoryError> {
	connection.pragma_update(None, "temp_store", "MEMORY")?;
	Ok(())
}

const SCHEMA_VERSION: i64 = 6;
const MAX_TITLE_BYTES: usize = 512;
const MAX_MODEL_REFERENCE_BYTES: usize = 512;
const MAX_MODEL_SNAPSHOT_BYTES: usize = 512;
const MAX_EVENT_BYTES: usize = 4 << 20;
const MAX_TURN_EVENTS: usize = 4_096;
const MAX_TURN_BYTES: usize = 16 << 20;
const MAX_REPLAY_EVENTS: usize = 100_000;
const MAX_REPLAY_BYTES: usize = 64 << 20;
const MAX_KNOWLEDGE_KEY_BYTES: usize = 256;
const MAX_KNOWLEDGE_BYTES: usize = 1 << 20;
const MAX_SESSION_PAGE_ITEMS: usize = 500;
const MAX_EVENT_PAGE_ITEMS: usize = 100;
const MAX_KNOWLEDGE_PAGE_ITEMS: usize = 100;
const MAX_PAGE_PAYLOAD_BYTES: usize = 16 << 20;
const MAX_SNAPSHOT_BYTES: usize = 2 << 20;
const MAX_JOB_FAILURE_BYTES: usize = 4 << 10;
const MAX_JOB_FAILURE_PAGE_ITEMS: usize = 500;
const MAX_JOB_FAILURES: u32 = 3;
const MAX_STALE_JOB_RECOVERIES: usize = 128;
const JOB_RETRY_BASE: Duration = Duration::from_secs(30);
const JOB_RETRY_MAX: Duration = Duration::from_mins(5);
const COMPACTION_LEASE: Duration = Duration::from_mins(5);
const SESSION_LEASE: Duration = Duration::from_mins(5);
const DISTILLATION_LEASE: Duration = Duration::from_mins(5);
static DATABASE_OPEN_LOCK: Mutex<()> = Mutex::new(());
const CREATE_SCHEMA_V6: &str = "CREATE TABLE sessions (
   id TEXT PRIMARY KEY NOT NULL,
   workspace TEXT NOT NULL,
   workspace_device TEXT NOT NULL
     CHECK(workspace_device <> ''
           AND workspace_device NOT GLOB '*[^0-9]*'),
   workspace_inode TEXT NOT NULL
     CHECK(workspace_inode <> ''
           AND workspace_inode NOT GLOB '*[^0-9]*'),
   model_reference TEXT,
   model_snapshot TEXT,
   title TEXT,
   created_at TEXT NOT NULL,
   updated_at TEXT NOT NULL,
   execution_token TEXT,
   execution_lease_until TEXT,
   CHECK(
     (execution_token IS NULL AND execution_lease_until IS NULL)
     OR
     (execution_token IS NOT NULL AND execution_lease_until IS NOT NULL)
   )
 ) STRICT;
 CREATE INDEX sessions_workspace_identity_updated
   ON sessions(workspace_device, workspace_inode, updated_at DESC);
 CREATE TABLE session_events (
   id TEXT PRIMARY KEY NOT NULL,
   session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
   sequence INTEGER NOT NULL CHECK(sequence > 0),
   turn_id TEXT NOT NULL,
   turn_index INTEGER NOT NULL CHECK(turn_index >= 0),
   turn_size INTEGER NOT NULL CHECK(turn_size > 0),
   kind TEXT NOT NULL,
   payload_json TEXT NOT NULL CHECK(json_valid(payload_json)),
   created_at TEXT NOT NULL,
   CHECK(turn_index < turn_size),
   UNIQUE(session_id, sequence),
   UNIQUE(session_id, turn_id, turn_index)
 ) STRICT;
 CREATE TABLE compaction_jobs (
   id TEXT PRIMARY KEY NOT NULL,
   session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
   through_sequence INTEGER NOT NULL CHECK(through_sequence > 0),
   state TEXT NOT NULL CHECK(state IN ('pending','running','completed','failed')),
   claim_token TEXT,
   lease_until TEXT,
   failure_count INTEGER NOT NULL DEFAULT 0 CHECK(failure_count >= 0),
   retry_after TEXT,
   last_error TEXT CHECK(last_error IS NULL OR length(last_error) <= 4096),
   failed_at TEXT,
   source_event_count INTEGER NOT NULL
     CHECK(source_event_count = through_sequence),
   source_first_event_id TEXT NOT NULL,
   source_last_event_id TEXT NOT NULL,
   source_sha256 TEXT NOT NULL CHECK(length(source_sha256) = 64),
   summary_event_id TEXT,
   created_at TEXT NOT NULL,
   updated_at TEXT NOT NULL,
   CHECK(
     (state = 'pending' AND claim_token IS NULL
      AND lease_until IS NULL AND summary_event_id IS NULL
      AND failed_at IS NULL
      AND (
        (failure_count = 0 AND retry_after IS NULL AND last_error IS NULL)
        OR (failure_count > 0 AND last_error IS NOT NULL)
      ))
     OR
     (state = 'running' AND claim_token IS NOT NULL
      AND lease_until IS NOT NULL AND summary_event_id IS NULL
      AND retry_after IS NULL AND failed_at IS NULL
      AND (
        (failure_count = 0 AND last_error IS NULL)
        OR (failure_count > 0 AND last_error IS NOT NULL)
      ))
     OR
     (state = 'completed' AND claim_token IS NULL
      AND lease_until IS NULL AND summary_event_id IS NOT NULL
      AND retry_after IS NULL AND last_error IS NULL AND failed_at IS NULL)
     OR
     (state = 'failed' AND claim_token IS NULL
      AND lease_until IS NULL AND summary_event_id IS NULL
      AND retry_after IS NULL AND last_error IS NOT NULL
      AND failed_at IS NOT NULL AND failure_count > 0)
   ),
   UNIQUE(session_id, through_sequence)
 ) STRICT;
 CREATE INDEX compaction_state_created
   ON compaction_jobs(state, retry_after, created_at);
 CREATE UNIQUE INDEX compaction_summary_event
   ON compaction_jobs(summary_event_id)
   WHERE summary_event_id IS NOT NULL;
 CREATE TABLE knowledge (
   id TEXT PRIMARY KEY NOT NULL,
   workspace TEXT NOT NULL,
   workspace_device TEXT NOT NULL
     CHECK(workspace_device <> ''
           AND workspace_device NOT GLOB '*[^0-9]*'),
   workspace_inode TEXT NOT NULL
     CHECK(workspace_inode <> ''
           AND workspace_inode NOT GLOB '*[^0-9]*'),
   legacy_identity INTEGER NOT NULL DEFAULT 0 CHECK(legacy_identity IN (0,1)),
   key TEXT NOT NULL,
   active_version INTEGER NOT NULL CHECK(active_version > 0),
   pinned INTEGER NOT NULL CHECK(pinned IN (0,1)),
   tombstoned INTEGER NOT NULL DEFAULT 0 CHECK(tombstoned IN (0,1)),
   tombstoned_at TEXT,
   created_at TEXT NOT NULL,
   updated_at TEXT NOT NULL,
   CHECK(
     (tombstoned = 0 AND tombstoned_at IS NULL)
     OR
     (tombstoned = 1 AND tombstoned_at IS NOT NULL)
   ),
   UNIQUE(workspace_device, workspace_inode, key)
 ) STRICT;
 CREATE INDEX knowledge_workspace_identity_updated
   ON knowledge(workspace_device, workspace_inode, tombstoned,
                pinned DESC, updated_at DESC);
 CREATE TABLE knowledge_versions (
   knowledge_id TEXT NOT NULL REFERENCES knowledge(id) ON DELETE CASCADE,
   version INTEGER NOT NULL CHECK(version > 0),
   content TEXT NOT NULL,
   confidence REAL NOT NULL DEFAULT 1.0
     CHECK(confidence >= 0.0 AND confidence <= 1.0),
   source_session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
   provenance_session_id TEXT,
   source_first_sequence INTEGER,
   source_last_sequence INTEGER,
   source_sha256 TEXT,
   distillation_job_id TEXT,
   candidate_index INTEGER,
   created_at TEXT NOT NULL,
   CHECK(
     (provenance_session_id IS NULL AND source_first_sequence IS NULL
      AND source_last_sequence IS NULL AND source_sha256 IS NULL)
     OR
     (provenance_session_id IS NOT NULL AND source_first_sequence > 0
      AND source_last_sequence >= source_first_sequence
      AND length(source_sha256) = 64)
   ),
   CHECK(
     (distillation_job_id IS NULL AND candidate_index IS NULL)
     OR
     (distillation_job_id IS NOT NULL AND candidate_index >= 0)
   ),
   PRIMARY KEY(knowledge_id, version)
 ) STRICT;
 CREATE UNIQUE INDEX knowledge_distillation_result
   ON knowledge_versions(distillation_job_id, candidate_index)
   WHERE distillation_job_id IS NOT NULL;
 CREATE TABLE knowledge_tombstones (
   id TEXT PRIMARY KEY NOT NULL,
   knowledge_id TEXT NOT NULL REFERENCES knowledge(id) ON DELETE CASCADE,
   confidence REAL NOT NULL CHECK(confidence >= 0.0 AND confidence <= 1.0),
   provenance_session_id TEXT,
   source_first_sequence INTEGER,
   source_last_sequence INTEGER,
   source_sha256 TEXT,
   distillation_job_id TEXT,
   candidate_index INTEGER,
   origin TEXT NOT NULL CHECK(origin IN ('manual','distillation')),
   created_at TEXT NOT NULL,
   CHECK(
     (provenance_session_id IS NULL AND source_first_sequence IS NULL
      AND source_last_sequence IS NULL AND source_sha256 IS NULL)
     OR
     (provenance_session_id IS NOT NULL AND source_first_sequence > 0
      AND source_last_sequence >= source_first_sequence
      AND length(source_sha256) = 64)
   )
 ) STRICT;
 CREATE UNIQUE INDEX knowledge_tombstone_distillation
   ON knowledge_tombstones(distillation_job_id, candidate_index)
   WHERE distillation_job_id IS NOT NULL;
 CREATE TABLE session_snapshots (
   session_id TEXT PRIMARY KEY NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
   schema_version INTEGER NOT NULL CHECK(schema_version > 0),
   config_json TEXT NOT NULL
     CHECK(json_valid(config_json) AND length(config_json) <= 2097152),
   authority_json TEXT NOT NULL
     CHECK(json_valid(authority_json) AND length(authority_json) <= 2097152),
   created_at TEXT NOT NULL,
   updated_at TEXT NOT NULL
 ) STRICT;
 CREATE TABLE assets (
   sha256 TEXT PRIMARY KEY NOT NULL
     CHECK(length(sha256) = 64 AND sha256 NOT GLOB '*[^0-9a-f]*'),
   bytes INTEGER NOT NULL CHECK(bytes >= 0 AND bytes <= 134217728),
   created_at TEXT NOT NULL,
   verified_at TEXT NOT NULL
 ) STRICT;
 CREATE TABLE session_assets (
   session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
   event_id TEXT NOT NULL REFERENCES session_events(id) ON DELETE CASCADE,
   asset_sha256 TEXT NOT NULL REFERENCES assets(sha256) ON DELETE RESTRICT,
   kind TEXT NOT NULL CHECK(kind IN ('image','audio','video','other')),
   ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
   created_at TEXT NOT NULL,
   PRIMARY KEY(event_id, ordinal)
 ) STRICT;
 CREATE INDEX session_assets_sha256 ON session_assets(asset_sha256);
 CREATE TABLE distillation_jobs (
   id TEXT PRIMARY KEY NOT NULL,
   session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
   through_sequence INTEGER NOT NULL CHECK(through_sequence > 0),
   source_event_count INTEGER NOT NULL CHECK(source_event_count > 0),
   source_first_event_id TEXT NOT NULL,
   source_last_event_id TEXT NOT NULL,
   source_sha256 TEXT NOT NULL CHECK(length(source_sha256) = 64),
   state TEXT NOT NULL CHECK(state IN ('pending','running','completed','failed')),
   claim_token TEXT,
   lease_until TEXT,
   failure_count INTEGER NOT NULL DEFAULT 0 CHECK(failure_count >= 0),
   retry_after TEXT,
   last_error TEXT CHECK(last_error IS NULL OR length(last_error) <= 4096),
   failed_at TEXT,
   created_at TEXT NOT NULL,
   updated_at TEXT NOT NULL,
   completed_at TEXT,
   CHECK(
     (state = 'pending' AND claim_token IS NULL AND lease_until IS NULL
      AND completed_at IS NULL AND failed_at IS NULL
      AND (
        (failure_count = 0 AND retry_after IS NULL AND last_error IS NULL)
        OR (failure_count > 0 AND last_error IS NOT NULL)
      ))
     OR
     (state = 'running' AND claim_token IS NOT NULL AND lease_until IS NOT NULL
      AND retry_after IS NULL AND completed_at IS NULL AND failed_at IS NULL
      AND (
        (failure_count = 0 AND last_error IS NULL)
        OR (failure_count > 0 AND last_error IS NOT NULL)
      ))
     OR
     (state = 'completed' AND claim_token IS NULL AND lease_until IS NULL
      AND retry_after IS NULL AND last_error IS NULL
      AND completed_at IS NOT NULL AND failed_at IS NULL)
     OR
     (state = 'failed' AND claim_token IS NULL AND lease_until IS NULL
      AND retry_after IS NULL AND last_error IS NOT NULL
      AND completed_at IS NULL AND failed_at IS NOT NULL
      AND failure_count > 0)
   ),
   UNIQUE(session_id, source_sha256)
 ) STRICT;
 CREATE INDEX distillation_state_created
   ON distillation_jobs(state, retry_after, created_at);
 CREATE TABLE distillation_results (
   job_id TEXT NOT NULL REFERENCES distillation_jobs(id) ON DELETE CASCADE,
   candidate_index INTEGER NOT NULL CHECK(candidate_index >= 0),
   knowledge_id TEXT NOT NULL REFERENCES knowledge(id) ON DELETE CASCADE,
   knowledge_version INTEGER NOT NULL CHECK(knowledge_version > 0),
   created_at TEXT NOT NULL,
   PRIMARY KEY(job_id, candidate_index)
 ) STRICT;
 CREATE TABLE active_agent_turns (
   session_id TEXT PRIMARY KEY NOT NULL
     REFERENCES sessions(id) ON DELETE CASCADE,
   turn_id TEXT NOT NULL UNIQUE,
   input_json TEXT NOT NULL
     CHECK(json_valid(input_json) AND length(input_json) <= 2097152),
   checkpoint_count INTEGER NOT NULL DEFAULT 0 CHECK(checkpoint_count >= 0),
   created_at TEXT NOT NULL,
   updated_at TEXT NOT NULL
 ) STRICT;
 CREATE TABLE active_agent_turn_assets (
   session_id TEXT NOT NULL
     REFERENCES active_agent_turns(session_id) ON DELETE CASCADE,
   ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
   asset_sha256 TEXT NOT NULL REFERENCES assets(sha256) ON DELETE RESTRICT,
   kind TEXT NOT NULL CHECK(kind IN ('image','audio','video','other')),
   PRIMARY KEY(session_id, ordinal)
 ) STRICT;
 CREATE INDEX active_agent_turn_assets_sha256
   ON active_agent_turn_assets(asset_sha256);
 CREATE TABLE pending_tool_batches (
   session_id TEXT PRIMARY KEY NOT NULL
     REFERENCES sessions(id) ON DELETE CASCADE,
   turn_id TEXT NOT NULL UNIQUE,
   messages_json TEXT NOT NULL
     CHECK(json_valid(messages_json) AND length(messages_json) <= 15728640),
   audit_json TEXT NOT NULL
     CHECK(json_valid(audit_json) AND length(audit_json) <= 15728640),
   created_at TEXT NOT NULL,
   updated_at TEXT NOT NULL
 ) STRICT;
 CREATE TABLE pending_tool_invocations (
   session_id TEXT NOT NULL
     REFERENCES pending_tool_batches(session_id) ON DELETE CASCADE,
   call_id TEXT NOT NULL,
   ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
   tool_name TEXT NOT NULL CHECK(tool_name <> '' AND length(tool_name) <= 256),
   arguments_json TEXT NOT NULL
     CHECK(json_valid(arguments_json) AND length(arguments_json) <= 4194304),
   state TEXT NOT NULL CHECK(state IN ('planned','started','completed')),
   result_json TEXT CHECK(
     result_json IS NULL
     OR (json_valid(result_json) AND length(result_json) <= 4194304)
   ),
   result_origin TEXT CHECK(
     result_origin IS NULL
     OR result_origin IN ('tool','uncertain','not_executed')
   ),
   result_is_error INTEGER CHECK(
     result_is_error IS NULL OR result_is_error IN (0, 1)
   ),
   updated_at TEXT NOT NULL,
   CHECK(
     (state IN ('planned','started')
      AND result_json IS NULL AND result_origin IS NULL
      AND result_is_error IS NULL)
     OR (state = 'completed'
         AND result_json IS NOT NULL AND result_origin IS NOT NULL
         AND result_is_error IS NOT NULL)
   ),
   PRIMARY KEY(session_id, call_id),
   UNIQUE(session_id, ordinal)
 ) STRICT;
 CREATE TABLE pending_tool_assets (
   session_id TEXT NOT NULL
     REFERENCES pending_tool_batches(session_id) ON DELETE CASCADE,
   asset_sha256 TEXT NOT NULL REFERENCES assets(sha256) ON DELETE RESTRICT,
   PRIMARY KEY(session_id, asset_sha256)
 ) STRICT;
 CREATE INDEX pending_tool_assets_sha256
   ON pending_tool_assets(asset_sha256);
 PRAGMA user_version = 6;";
const MIGRATE_V1_TO_V2: &str = "ALTER TABLE sessions ADD COLUMN model_snapshot TEXT;
 DROP INDEX compaction_state_created;
 ALTER TABLE compaction_jobs RENAME TO compaction_jobs_v1;
 CREATE TABLE compaction_jobs (
   id TEXT PRIMARY KEY NOT NULL,
   session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
   through_sequence INTEGER NOT NULL CHECK(through_sequence > 0),
   state TEXT NOT NULL CHECK(state IN ('pending','running','completed')),
   claim_token TEXT,
   lease_until TEXT,
   created_at TEXT NOT NULL,
   updated_at TEXT NOT NULL,
   CHECK(
     (state = 'running' AND claim_token IS NOT NULL AND lease_until IS NOT NULL)
     OR
     (state != 'running' AND claim_token IS NULL AND lease_until IS NULL)
   ),
   UNIQUE(session_id, through_sequence)
 ) STRICT;
 INSERT INTO compaction_jobs
   (id, session_id, through_sequence, state, claim_token, lease_until,
    created_at, updated_at)
 SELECT id, session_id, through_sequence,
        CASE WHEN state = 'running' THEN 'pending' ELSE state END,
        NULL, NULL, created_at, updated_at
 FROM compaction_jobs_v1;
 DROP TABLE compaction_jobs_v1;
 CREATE INDEX compaction_state_created
   ON compaction_jobs(state, created_at);
 PRAGMA user_version = 2;";
const PREPARE_V2_TO_V3: &str = "DROP INDEX IF EXISTS sessions_workspace_updated;
 ALTER TABLE sessions ADD COLUMN workspace_device TEXT;
 ALTER TABLE sessions ADD COLUMN workspace_inode TEXT;
 ALTER TABLE sessions ADD COLUMN execution_token TEXT;
 ALTER TABLE sessions ADD COLUMN execution_lease_until TEXT;
 ALTER TABLE session_events ADD COLUMN turn_id TEXT;
 ALTER TABLE session_events ADD COLUMN turn_index INTEGER;
 ALTER TABLE session_events ADD COLUMN turn_size INTEGER;
 ALTER TABLE compaction_jobs ADD COLUMN source_event_count INTEGER;
 ALTER TABLE compaction_jobs ADD COLUMN source_first_event_id TEXT;
 ALTER TABLE compaction_jobs ADD COLUMN source_last_event_id TEXT;
 ALTER TABLE compaction_jobs ADD COLUMN source_sha256 TEXT;
 ALTER TABLE compaction_jobs ADD COLUMN summary_event_id TEXT;
 UPDATE sessions
 SET execution_token = NULL, execution_lease_until = NULL;
 UPDATE session_events
 SET turn_id = id, turn_index = 0, turn_size = 1;
 UPDATE compaction_jobs
 SET state = 'pending', claim_token = NULL, lease_until = NULL
 WHERE state = 'running';";
const FINISH_V2_TO_V3: &str = "CREATE TABLE sessions_v3 (
   id TEXT PRIMARY KEY NOT NULL,
   workspace TEXT NOT NULL,
   workspace_device TEXT NOT NULL
     CHECK(workspace_device <> ''
           AND workspace_device NOT GLOB '*[^0-9]*'),
   workspace_inode TEXT NOT NULL
     CHECK(workspace_inode <> ''
           AND workspace_inode NOT GLOB '*[^0-9]*'),
   model_reference TEXT,
   model_snapshot TEXT,
   title TEXT,
   created_at TEXT NOT NULL,
   updated_at TEXT NOT NULL,
   execution_token TEXT,
   execution_lease_until TEXT,
   CHECK(
     (execution_token IS NULL AND execution_lease_until IS NULL)
     OR
     (execution_token IS NOT NULL AND execution_lease_until IS NOT NULL)
   )
 ) STRICT;
 INSERT INTO sessions_v3
 SELECT id, workspace, workspace_device, workspace_inode, model_reference,
        model_snapshot, title, created_at, updated_at,
        execution_token, execution_lease_until
 FROM sessions;
 CREATE TABLE session_events_v3 (
   id TEXT PRIMARY KEY NOT NULL,
   session_id TEXT NOT NULL REFERENCES sessions_v3(id) ON DELETE CASCADE,
   sequence INTEGER NOT NULL CHECK(sequence > 0),
   turn_id TEXT NOT NULL,
   turn_index INTEGER NOT NULL CHECK(turn_index >= 0),
   turn_size INTEGER NOT NULL CHECK(turn_size > 0),
   kind TEXT NOT NULL,
   payload_json TEXT NOT NULL CHECK(json_valid(payload_json)),
   created_at TEXT NOT NULL,
   CHECK(turn_index < turn_size),
   UNIQUE(session_id, sequence),
   UNIQUE(session_id, turn_id, turn_index)
 ) STRICT;
 INSERT INTO session_events_v3
 SELECT id, session_id, sequence, turn_id, turn_index, turn_size,
        kind, payload_json, created_at
 FROM session_events;
 CREATE TABLE compaction_jobs_v3 (
   id TEXT PRIMARY KEY NOT NULL,
   session_id TEXT NOT NULL REFERENCES sessions_v3(id) ON DELETE CASCADE,
   through_sequence INTEGER NOT NULL CHECK(through_sequence > 0),
   state TEXT NOT NULL CHECK(state IN ('pending','running','completed')),
   claim_token TEXT,
   lease_until TEXT,
   source_event_count INTEGER NOT NULL
     CHECK(source_event_count = through_sequence),
   source_first_event_id TEXT NOT NULL,
   source_last_event_id TEXT NOT NULL,
   source_sha256 TEXT NOT NULL CHECK(length(source_sha256) = 64),
   summary_event_id TEXT,
   created_at TEXT NOT NULL,
   updated_at TEXT NOT NULL,
   CHECK(
     (state = 'pending' AND claim_token IS NULL
      AND lease_until IS NULL AND summary_event_id IS NULL)
     OR
     (state = 'running' AND claim_token IS NOT NULL
      AND lease_until IS NOT NULL AND summary_event_id IS NULL)
     OR
     (state = 'completed' AND claim_token IS NULL
      AND lease_until IS NULL AND summary_event_id IS NOT NULL)
   ),
   UNIQUE(session_id, through_sequence)
 ) STRICT;
 INSERT INTO compaction_jobs_v3
 SELECT id, session_id, through_sequence, state, claim_token, lease_until,
        source_event_count, source_first_event_id, source_last_event_id,
        source_sha256, summary_event_id, created_at, updated_at
 FROM compaction_jobs;
 CREATE TABLE knowledge_v3 (
   id TEXT PRIMARY KEY NOT NULL,
   workspace TEXT NOT NULL,
   key TEXT NOT NULL,
   active_version INTEGER NOT NULL CHECK(active_version > 0),
   pinned INTEGER NOT NULL CHECK(pinned IN (0,1)),
   created_at TEXT NOT NULL,
   updated_at TEXT NOT NULL,
   UNIQUE(workspace, key)
 ) STRICT;
 INSERT INTO knowledge_v3 SELECT * FROM knowledge;
 CREATE TABLE knowledge_versions_v3 (
   knowledge_id TEXT NOT NULL REFERENCES knowledge_v3(id) ON DELETE CASCADE,
   version INTEGER NOT NULL CHECK(version > 0),
   content TEXT NOT NULL,
   source_session_id TEXT REFERENCES sessions_v3(id) ON DELETE SET NULL,
   created_at TEXT NOT NULL,
   PRIMARY KEY(knowledge_id, version)
 ) STRICT;
 INSERT INTO knowledge_versions_v3 SELECT * FROM knowledge_versions;
 DROP TABLE knowledge_versions;
 DROP TABLE knowledge;
 DROP TABLE compaction_jobs;
 DROP TABLE session_events;
 DROP TABLE sessions;
 ALTER TABLE sessions_v3 RENAME TO sessions;
 ALTER TABLE session_events_v3 RENAME TO session_events;
 ALTER TABLE compaction_jobs_v3 RENAME TO compaction_jobs;
 ALTER TABLE knowledge_v3 RENAME TO knowledge;
 ALTER TABLE knowledge_versions_v3 RENAME TO knowledge_versions;
 CREATE INDEX sessions_workspace_identity_updated
   ON sessions(workspace_device, workspace_inode, updated_at DESC);
 CREATE INDEX compaction_state_created
   ON compaction_jobs(state, created_at);
 CREATE UNIQUE INDEX compaction_summary_event
   ON compaction_jobs(summary_event_id)
   WHERE summary_event_id IS NOT NULL;
 CREATE INDEX knowledge_workspace_updated
   ON knowledge(workspace, pinned DESC, updated_at DESC);
 PRAGMA user_version = 3;";
const PREPARE_V3_TO_V4: &str = "ALTER TABLE knowledge
   ADD COLUMN workspace_device TEXT;
 ALTER TABLE knowledge ADD COLUMN workspace_inode TEXT;
 ALTER TABLE knowledge
   ADD COLUMN legacy_identity INTEGER NOT NULL DEFAULT 0;";
const FINISH_V3_TO_V4: &str = "CREATE TABLE knowledge_v4 (
   id TEXT PRIMARY KEY NOT NULL,
   workspace TEXT NOT NULL,
   workspace_device TEXT NOT NULL
     CHECK(workspace_device <> ''
           AND workspace_device NOT GLOB '*[^0-9]*'),
   workspace_inode TEXT NOT NULL
     CHECK(workspace_inode <> ''
           AND workspace_inode NOT GLOB '*[^0-9]*'),
   legacy_identity INTEGER NOT NULL DEFAULT 0 CHECK(legacy_identity IN (0,1)),
   key TEXT NOT NULL,
   active_version INTEGER NOT NULL CHECK(active_version > 0),
   pinned INTEGER NOT NULL CHECK(pinned IN (0,1)),
   tombstoned INTEGER NOT NULL DEFAULT 0 CHECK(tombstoned IN (0,1)),
   tombstoned_at TEXT,
   created_at TEXT NOT NULL,
   updated_at TEXT NOT NULL,
   CHECK(
     (tombstoned = 0 AND tombstoned_at IS NULL)
     OR
     (tombstoned = 1 AND tombstoned_at IS NOT NULL)
   ),
   UNIQUE(workspace_device, workspace_inode, key)
 ) STRICT;
 CREATE TABLE knowledge_versions_v4 (
   knowledge_id TEXT NOT NULL REFERENCES knowledge_v4(id) ON DELETE CASCADE,
   version INTEGER NOT NULL CHECK(version > 0),
   content TEXT NOT NULL,
   confidence REAL NOT NULL DEFAULT 1.0
     CHECK(confidence >= 0.0 AND confidence <= 1.0),
   source_session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
   provenance_session_id TEXT,
   source_first_sequence INTEGER,
   source_last_sequence INTEGER,
   source_sha256 TEXT,
   distillation_job_id TEXT,
   candidate_index INTEGER,
   created_at TEXT NOT NULL,
   CHECK(
     (provenance_session_id IS NULL AND source_first_sequence IS NULL
      AND source_last_sequence IS NULL AND source_sha256 IS NULL)
     OR
     (provenance_session_id IS NOT NULL AND source_first_sequence > 0
      AND source_last_sequence >= source_first_sequence
      AND length(source_sha256) = 64)
   ),
   CHECK(
     (distillation_job_id IS NULL AND candidate_index IS NULL)
     OR
     (distillation_job_id IS NOT NULL AND candidate_index >= 0)
   ),
   PRIMARY KEY(knowledge_id, version)
 ) STRICT;
 INSERT INTO knowledge_v4
   (id, workspace, workspace_device, workspace_inode, legacy_identity,
    key, active_version, pinned, tombstoned, tombstoned_at,
    created_at, updated_at)
 SELECT id, workspace, workspace_device, workspace_inode, legacy_identity,
        key, active_version, pinned, 0, NULL, created_at, updated_at
 FROM knowledge;
 INSERT INTO knowledge_versions_v4
   (knowledge_id, version, content, confidence, source_session_id,
    provenance_session_id, source_first_sequence, source_last_sequence, source_sha256,
    distillation_job_id, candidate_index, created_at)
 SELECT knowledge_id, version, content, 1.0, source_session_id,
        NULL, NULL, NULL, NULL, NULL, NULL, created_at
 FROM knowledge_versions;
 DROP TABLE knowledge_versions;
 DROP TABLE knowledge;
 ALTER TABLE knowledge_v4 RENAME TO knowledge;
 ALTER TABLE knowledge_versions_v4 RENAME TO knowledge_versions;
 CREATE INDEX knowledge_workspace_identity_updated
   ON knowledge(workspace_device, workspace_inode, tombstoned,
                pinned DESC, updated_at DESC);
 CREATE UNIQUE INDEX knowledge_distillation_result
   ON knowledge_versions(distillation_job_id, candidate_index)
   WHERE distillation_job_id IS NOT NULL;
 CREATE TABLE knowledge_tombstones (
   id TEXT PRIMARY KEY NOT NULL,
   knowledge_id TEXT NOT NULL REFERENCES knowledge(id) ON DELETE CASCADE,
   confidence REAL NOT NULL CHECK(confidence >= 0.0 AND confidence <= 1.0),
   provenance_session_id TEXT,
   source_first_sequence INTEGER,
   source_last_sequence INTEGER,
   source_sha256 TEXT,
   distillation_job_id TEXT,
   candidate_index INTEGER,
   origin TEXT NOT NULL CHECK(origin IN ('manual','distillation')),
   created_at TEXT NOT NULL,
   CHECK(
     (provenance_session_id IS NULL AND source_first_sequence IS NULL
      AND source_last_sequence IS NULL AND source_sha256 IS NULL)
     OR
     (provenance_session_id IS NOT NULL AND source_first_sequence > 0
      AND source_last_sequence >= source_first_sequence
      AND length(source_sha256) = 64)
   )
 ) STRICT;
 CREATE UNIQUE INDEX knowledge_tombstone_distillation
   ON knowledge_tombstones(distillation_job_id, candidate_index)
   WHERE distillation_job_id IS NOT NULL;
 CREATE TABLE session_snapshots (
   session_id TEXT PRIMARY KEY NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
   schema_version INTEGER NOT NULL CHECK(schema_version > 0),
   config_json TEXT NOT NULL
     CHECK(json_valid(config_json) AND length(config_json) <= 2097152),
   tools_json TEXT NOT NULL
     CHECK(json_valid(tools_json) AND length(tools_json) <= 2097152),
   created_at TEXT NOT NULL,
   updated_at TEXT NOT NULL
 ) STRICT;
 CREATE TABLE assets (
   sha256 TEXT PRIMARY KEY NOT NULL
     CHECK(length(sha256) = 64 AND sha256 NOT GLOB '*[^0-9a-f]*'),
   bytes INTEGER NOT NULL CHECK(bytes >= 0 AND bytes <= 134217728),
   created_at TEXT NOT NULL,
   verified_at TEXT NOT NULL
 ) STRICT;
 CREATE TABLE session_assets (
   session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
   event_id TEXT NOT NULL REFERENCES session_events(id) ON DELETE CASCADE,
   asset_sha256 TEXT NOT NULL REFERENCES assets(sha256) ON DELETE RESTRICT,
   kind TEXT NOT NULL CHECK(kind IN ('image','audio','video','other')),
   ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
   created_at TEXT NOT NULL,
   PRIMARY KEY(event_id, ordinal)
 ) STRICT;
 CREATE INDEX session_assets_sha256 ON session_assets(asset_sha256);
 CREATE TABLE distillation_jobs (
   id TEXT PRIMARY KEY NOT NULL,
   session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
   through_sequence INTEGER NOT NULL CHECK(through_sequence > 0),
   source_event_count INTEGER NOT NULL CHECK(source_event_count > 0),
   source_first_event_id TEXT NOT NULL,
   source_last_event_id TEXT NOT NULL,
   source_sha256 TEXT NOT NULL CHECK(length(source_sha256) = 64),
   state TEXT NOT NULL CHECK(state IN ('pending','running','completed')),
   claim_token TEXT,
   lease_until TEXT,
   created_at TEXT NOT NULL,
   updated_at TEXT NOT NULL,
   completed_at TEXT,
   CHECK(
     (state = 'pending' AND claim_token IS NULL AND lease_until IS NULL
      AND completed_at IS NULL)
     OR
     (state = 'running' AND claim_token IS NOT NULL AND lease_until IS NOT NULL
      AND completed_at IS NULL)
     OR
     (state = 'completed' AND claim_token IS NULL AND lease_until IS NULL
      AND completed_at IS NOT NULL)
   ),
   UNIQUE(session_id, source_sha256)
 ) STRICT;
 CREATE INDEX distillation_state_created
   ON distillation_jobs(state, created_at);
 CREATE TABLE distillation_results (
   job_id TEXT NOT NULL REFERENCES distillation_jobs(id) ON DELETE CASCADE,
   candidate_index INTEGER NOT NULL CHECK(candidate_index >= 0),
   knowledge_id TEXT NOT NULL REFERENCES knowledge(id) ON DELETE CASCADE,
   knowledge_version INTEGER NOT NULL CHECK(knowledge_version > 0),
   created_at TEXT NOT NULL,
   PRIMARY KEY(job_id, candidate_index)
 ) STRICT;
 PRAGMA user_version = 4;";
const MIGRATE_V4_TO_V5: &str = "DROP INDEX compaction_state_created;
 DROP INDEX compaction_summary_event;
 ALTER TABLE compaction_jobs RENAME TO compaction_jobs_v4;
 CREATE TABLE compaction_jobs (
   id TEXT PRIMARY KEY NOT NULL,
   session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
   through_sequence INTEGER NOT NULL CHECK(through_sequence > 0),
   state TEXT NOT NULL CHECK(state IN ('pending','running','completed','failed')),
   claim_token TEXT,
   lease_until TEXT,
   failure_count INTEGER NOT NULL DEFAULT 0 CHECK(failure_count >= 0),
   retry_after TEXT,
   last_error TEXT CHECK(last_error IS NULL OR length(last_error) <= 4096),
   failed_at TEXT,
   source_event_count INTEGER NOT NULL
     CHECK(source_event_count = through_sequence),
   source_first_event_id TEXT NOT NULL,
   source_last_event_id TEXT NOT NULL,
   source_sha256 TEXT NOT NULL CHECK(length(source_sha256) = 64),
   summary_event_id TEXT,
   created_at TEXT NOT NULL,
   updated_at TEXT NOT NULL,
   CHECK(
     (state = 'pending' AND claim_token IS NULL
      AND lease_until IS NULL AND summary_event_id IS NULL
      AND failed_at IS NULL
      AND (
        (failure_count = 0 AND retry_after IS NULL AND last_error IS NULL)
        OR (failure_count > 0 AND last_error IS NOT NULL)
      ))
     OR
     (state = 'running' AND claim_token IS NOT NULL
      AND lease_until IS NOT NULL AND summary_event_id IS NULL
      AND retry_after IS NULL AND failed_at IS NULL
      AND (
        (failure_count = 0 AND last_error IS NULL)
        OR (failure_count > 0 AND last_error IS NOT NULL)
      ))
     OR
     (state = 'completed' AND claim_token IS NULL
      AND lease_until IS NULL AND summary_event_id IS NOT NULL
      AND retry_after IS NULL AND last_error IS NULL AND failed_at IS NULL)
     OR
     (state = 'failed' AND claim_token IS NULL
      AND lease_until IS NULL AND summary_event_id IS NULL
      AND retry_after IS NULL AND last_error IS NOT NULL
      AND failed_at IS NOT NULL AND failure_count > 0)
   ),
   UNIQUE(session_id, through_sequence)
 ) STRICT;
 INSERT INTO compaction_jobs
   (id, session_id, through_sequence, state, claim_token, lease_until,
    failure_count, retry_after, last_error, failed_at,
    source_event_count, source_first_event_id, source_last_event_id,
    source_sha256, summary_event_id, created_at, updated_at)
 SELECT id, session_id, through_sequence, state, claim_token, lease_until,
        0, NULL, NULL, NULL,
        source_event_count, source_first_event_id, source_last_event_id,
        source_sha256, summary_event_id, created_at, updated_at
 FROM compaction_jobs_v4;
 DROP TABLE compaction_jobs_v4;
 CREATE INDEX compaction_state_created
   ON compaction_jobs(state, retry_after, created_at);
 CREATE UNIQUE INDEX compaction_summary_event
   ON compaction_jobs(summary_event_id)
   WHERE summary_event_id IS NOT NULL;
 ALTER TABLE distillation_results RENAME TO distillation_results_v4;
 DROP INDEX distillation_state_created;
 ALTER TABLE distillation_jobs RENAME TO distillation_jobs_v4;
 CREATE TABLE distillation_jobs (
   id TEXT PRIMARY KEY NOT NULL,
   session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
   through_sequence INTEGER NOT NULL CHECK(through_sequence > 0),
   source_event_count INTEGER NOT NULL CHECK(source_event_count > 0),
   source_first_event_id TEXT NOT NULL,
   source_last_event_id TEXT NOT NULL,
   source_sha256 TEXT NOT NULL CHECK(length(source_sha256) = 64),
   state TEXT NOT NULL CHECK(state IN ('pending','running','completed','failed')),
   claim_token TEXT,
   lease_until TEXT,
   failure_count INTEGER NOT NULL DEFAULT 0 CHECK(failure_count >= 0),
   retry_after TEXT,
   last_error TEXT CHECK(last_error IS NULL OR length(last_error) <= 4096),
   failed_at TEXT,
   created_at TEXT NOT NULL,
   updated_at TEXT NOT NULL,
   completed_at TEXT,
   CHECK(
     (state = 'pending' AND claim_token IS NULL AND lease_until IS NULL
      AND completed_at IS NULL AND failed_at IS NULL
      AND (
        (failure_count = 0 AND retry_after IS NULL AND last_error IS NULL)
        OR (failure_count > 0 AND last_error IS NOT NULL)
      ))
     OR
     (state = 'running' AND claim_token IS NOT NULL AND lease_until IS NOT NULL
      AND retry_after IS NULL AND completed_at IS NULL AND failed_at IS NULL
      AND (
        (failure_count = 0 AND last_error IS NULL)
        OR (failure_count > 0 AND last_error IS NOT NULL)
      ))
     OR
     (state = 'completed' AND claim_token IS NULL AND lease_until IS NULL
      AND retry_after IS NULL AND last_error IS NULL
      AND completed_at IS NOT NULL AND failed_at IS NULL)
     OR
     (state = 'failed' AND claim_token IS NULL AND lease_until IS NULL
      AND retry_after IS NULL AND last_error IS NOT NULL
      AND completed_at IS NULL AND failed_at IS NOT NULL
      AND failure_count > 0)
   ),
   UNIQUE(session_id, source_sha256)
 ) STRICT;
 INSERT INTO distillation_jobs
   (id, session_id, through_sequence, source_event_count,
    source_first_event_id, source_last_event_id, source_sha256,
    state, claim_token, lease_until, failure_count, retry_after,
    last_error, failed_at, created_at, updated_at, completed_at)
 SELECT id, session_id, through_sequence, source_event_count,
        source_first_event_id, source_last_event_id, source_sha256,
        state, claim_token, lease_until, 0, NULL, NULL, NULL,
        created_at, updated_at, completed_at
 FROM distillation_jobs_v4;
 CREATE INDEX distillation_state_created
   ON distillation_jobs(state, retry_after, created_at);
 CREATE TABLE distillation_results (
   job_id TEXT NOT NULL REFERENCES distillation_jobs(id) ON DELETE CASCADE,
   candidate_index INTEGER NOT NULL CHECK(candidate_index >= 0),
   knowledge_id TEXT NOT NULL REFERENCES knowledge(id) ON DELETE CASCADE,
   knowledge_version INTEGER NOT NULL CHECK(knowledge_version > 0),
   created_at TEXT NOT NULL,
   PRIMARY KEY(job_id, candidate_index)
 ) STRICT;
 INSERT INTO distillation_results SELECT * FROM distillation_results_v4;
 DROP TABLE distillation_results_v4;
 DROP TABLE distillation_jobs_v4;
 PRAGMA user_version = 5;";
const MIGRATE_V5_TO_V6: &str = "ALTER TABLE session_snapshots
   RENAME COLUMN tools_json TO authority_json;
 CREATE TABLE active_agent_turns (
   session_id TEXT PRIMARY KEY NOT NULL
     REFERENCES sessions(id) ON DELETE CASCADE,
   turn_id TEXT NOT NULL UNIQUE,
   input_json TEXT NOT NULL
     CHECK(json_valid(input_json) AND length(input_json) <= 2097152),
   checkpoint_count INTEGER NOT NULL DEFAULT 0 CHECK(checkpoint_count >= 0),
   created_at TEXT NOT NULL,
   updated_at TEXT NOT NULL
 ) STRICT;
 CREATE TABLE active_agent_turn_assets (
   session_id TEXT NOT NULL
     REFERENCES active_agent_turns(session_id) ON DELETE CASCADE,
   ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
   asset_sha256 TEXT NOT NULL REFERENCES assets(sha256) ON DELETE RESTRICT,
   kind TEXT NOT NULL CHECK(kind IN ('image','audio','video','other')),
   PRIMARY KEY(session_id, ordinal)
 ) STRICT;
 CREATE INDEX active_agent_turn_assets_sha256
   ON active_agent_turn_assets(asset_sha256);
 CREATE TABLE pending_tool_batches (
   session_id TEXT PRIMARY KEY NOT NULL
     REFERENCES sessions(id) ON DELETE CASCADE,
   turn_id TEXT NOT NULL UNIQUE,
   messages_json TEXT NOT NULL
     CHECK(json_valid(messages_json) AND length(messages_json) <= 15728640),
   audit_json TEXT NOT NULL
     CHECK(json_valid(audit_json) AND length(audit_json) <= 15728640),
   created_at TEXT NOT NULL,
   updated_at TEXT NOT NULL
 ) STRICT;
 CREATE TABLE pending_tool_invocations (
   session_id TEXT NOT NULL
     REFERENCES pending_tool_batches(session_id) ON DELETE CASCADE,
   call_id TEXT NOT NULL,
   ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
   tool_name TEXT NOT NULL CHECK(tool_name <> '' AND length(tool_name) <= 256),
   arguments_json TEXT NOT NULL
     CHECK(json_valid(arguments_json) AND length(arguments_json) <= 4194304),
   state TEXT NOT NULL CHECK(state IN ('planned','started','completed')),
   result_json TEXT CHECK(
     result_json IS NULL
     OR (json_valid(result_json) AND length(result_json) <= 4194304)
   ),
   result_origin TEXT CHECK(
     result_origin IS NULL
     OR result_origin IN ('tool','uncertain','not_executed')
   ),
   result_is_error INTEGER CHECK(
     result_is_error IS NULL OR result_is_error IN (0, 1)
   ),
   updated_at TEXT NOT NULL,
   CHECK(
     (state IN ('planned','started')
      AND result_json IS NULL AND result_origin IS NULL
      AND result_is_error IS NULL)
     OR (state = 'completed'
         AND result_json IS NOT NULL AND result_origin IS NOT NULL
         AND result_is_error IS NOT NULL)
   ),
   PRIMARY KEY(session_id, call_id),
   UNIQUE(session_id, ordinal)
 ) STRICT;
 CREATE TABLE pending_tool_assets (
   session_id TEXT NOT NULL
     REFERENCES pending_tool_batches(session_id) ON DELETE CASCADE,
   asset_sha256 TEXT NOT NULL REFERENCES assets(sha256) ON DELETE RESTRICT,
   PRIMARY KEY(session_id, asset_sha256)
 ) STRICT;
 CREATE INDEX pending_tool_assets_sha256
   ON pending_tool_assets(asset_sha256);
 PRAGMA user_version = 6;";

/// Reopenable, thread-safe handle to Emelex's `SQLite` memory database.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryStore {
	database: PathBuf,
	home: Option<EmelexHome>,
}

impl MemoryStore {
	/// Open the standard database in `home`, applying migrations.
	///
	/// # Errors
	///
	/// Returns an error when the path is unsafe, `SQLite` cannot open it, or a
	/// newer incompatible schema is present.
	pub fn open(home: &EmelexHome) -> Result<Self, MemoryError> {
		Self::open_database(home.database_file(), Some(home.clone()))
	}

	/// Open an explicitly selected database path.
	///
	/// This is useful for embedding and tests. The parent directory must
	/// already exist and be a real directory.
	///
	/// # Errors
	///
	/// Returns an error when the path is unsafe, `SQLite` cannot open it, or a
	/// newer incompatible schema is present.
	pub fn open_path(database: impl Into<PathBuf>) -> Result<Self, MemoryError> {
		Self::open_database(database.into(), None)
	}

	fn open_database(database: PathBuf, home: Option<EmelexHome>) -> Result<Self, MemoryError> {
		let process_lock = DATABASE_OPEN_LOCK
			.lock()
			.unwrap_or_else(std::sync::PoisonError::into_inner);
		let database = absolute_database_path(database)?;
		validate_database_parent(&database)?;
		prepare_database_file(&database)?;
		let store = Self { database, home };
		let migration_lock = lock_database(&store.database)?;
		assets::prepare_assets_dir(&store.database)?;
		let mut connection = store.connection()?;
		Self::migrate(&mut connection)?;
		drop(connection);
		drop(migration_lock);
		validate_database_file(&store.database)?;
		drop(process_lock);
		Ok(store)
	}

	/// Selected database path.
	pub fn database_path(&self) -> &Path {
		&self.database
	}

	fn validate_session_lease_origin(&self, lease: &SessionLease) -> Result<(), MemoryError> {
		if lease.store.database != self.database {
			return Err(MemoryError::Invalid(
				"session lease belongs to another MemoryStore".to_string(),
			));
		}
		Ok(())
	}

	fn validate_compaction_lease_origin(&self, lease: &CompactionLease) -> Result<(), MemoryError> {
		if lease.store.database != self.database {
			return Err(MemoryError::Invalid(
				"compaction lease belongs to another MemoryStore".to_string(),
			));
		}
		Ok(())
	}

	/// Create a durable session rooted at an existing workspace.
	///
	/// # Errors
	///
	/// Returns an error for an invalid workspace/field or database failure.
	pub fn start_session(
		&self,
		workspace: &Path,
		title: Option<&str>,
	) -> Result<Session, MemoryError> {
		validate_optional(title, MAX_TITLE_BYTES, "session title")?;
		let workspace = workspace_binding(workspace)?;
		let id = Uuid::now_v7();
		let now = Utc::now();
		let connection = self.connection()?;
		connection.execute(
			"INSERT INTO sessions
				 (id, workspace, workspace_device, workspace_inode,
				  title, created_at, updated_at)
				 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6)",
			params![
				id.to_string(),
				path_text(&workspace.path)?,
				workspace.identity.device.to_string(),
				workspace.identity.inode.to_string(),
				title,
				now.to_rfc3339(),
			],
		)?;
		Ok(Session {
			id,
			workspace: workspace.path,
			workspace_identity: workspace.identity,
			model_reference: None,
			model_snapshot: None,
			title: title.map(str::to_string),
			created_at: now,
			updated_at: now,
		})
	}

	/// Fetch one session.
	///
	/// # Errors
	///
	/// Returns [`MemoryError::NotFound`] or a database/corruption error.
	pub fn session(&self, id: Uuid) -> Result<Session, MemoryError> {
		let connection = self.connection()?;
		let raw = connection
			.query_row(
				&session_select("WHERE id = ?1"),
				[id.to_string()],
				raw_session,
			)
			.optional()?
			.ok_or(MemoryError::NotFound {
				entity: "session",
				id,
			})?;
		Session::try_from(raw)
	}

	/// Claim exclusive, expiring authority to resume and append to a session.
	///
	/// The supplied workspace may use a new canonical path after a directory
	/// rename, but its filesystem device/inode identity must match the
	/// identity captured when the session started.
	///
	/// # Errors
	///
	/// Returns an error when the session is missing, another live execution
	/// owns it, the workspace identity changed, or storage is corrupt.
	pub fn claim_session(&self, id: Uuid, workspace: &Path) -> Result<SessionLease, MemoryError> {
		let workspace = workspace_binding(workspace)?;
		let mut connection = self.connection()?;
		let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
		let raw = transaction
			.query_row(
				&session_select("WHERE id = ?1"),
				[id.to_string()],
				raw_session,
			)
			.optional()?
			.ok_or(MemoryError::NotFound {
				entity: "session",
				id,
			})?;
		let mut session = Session::try_from(raw)?;
		if session.workspace_identity != workspace.identity {
			return Err(MemoryError::WorkspaceMismatch {
				session_id: id,
				expected: session.workspace_identity,
				actual: workspace.identity,
			});
		}
		let (claim_token, lease_until) = transaction.query_row(
			"SELECT execution_token, execution_lease_until
			 FROM sessions WHERE id = ?1",
			[id.to_string()],
			|row| {
				Ok((
					row.get::<_, Option<String>>(0)?,
					row.get::<_, Option<String>>(1)?,
				))
			},
		)?;
		let current_lease = parse_session_claim(claim_token, lease_until)?;
		let now = Utc::now();
		if let Some((_token, deadline)) = current_lease
			&& deadline > now
		{
			return Err(MemoryError::SessionBusy {
				session_id: id,
				lease_until: deadline,
			});
		}
		let token = Uuid::now_v7();
		let lease_until = deadline_after(now, SESSION_LEASE, "session lease")?;
		let workspace_text = path_text(&workspace.path)?;
		let changed = transaction.execute(
			"UPDATE sessions
			 SET execution_token = ?2, execution_lease_until = ?3,
			     workspace = ?4, updated_at = MAX(updated_at, ?5)
			 WHERE id = ?1
			   AND (execution_token IS NULL OR execution_lease_until <= ?5)",
			params![
				id.to_string(),
				token.to_string(),
				lease_until.to_rfc3339(),
				workspace_text,
				now.to_rfc3339(),
			],
		)?;
		if changed != 1 {
			return Err(MemoryError::SessionBusy {
				session_id: id,
				lease_until,
			});
		}
		let last_sequence = last_event_sequence(&transaction, id)?;
		transaction.commit()?;
		session.workspace = workspace.path;
		session.updated_at = session.updated_at.max(now);
		Ok(SessionLease {
			store: self.clone(),
			session,
			token,
			lease_until,
			last_sequence,
			replayed: last_sequence == 0,
			released: AtomicBool::new(false),
		})
	}

	/// Renew a session claim while the same token still owns it.
	///
	/// Renewal may recover an expired claim only when no other process has
	/// reclaimed it.
	///
	/// # Errors
	///
	/// Returns an error when workspace identity changed, ownership was lost,
	/// or the database failed.
	pub fn renew_session(&self, lease: &mut SessionLease) -> Result<(), MemoryError> {
		self.validate_session_lease_origin(lease)?;
		validate_workspace_identity(&lease.session)?;
		let now = Utc::now();
		let lease_until = deadline_after(now, SESSION_LEASE, "session lease")?;
		let changed = self.connection()?.execute(
			"UPDATE sessions
			 SET execution_lease_until = ?3
			 WHERE id = ?1 AND execution_token = ?2",
			params![
				lease.session.id.to_string(),
				lease.token.to_string(),
				lease_until.to_rfc3339(),
			],
		)?;
		if changed != 1 {
			return Err(MemoryError::StaleSessionLease {
				session_id: lease.session.id,
			});
		}
		lease.lease_until = lease_until;
		Ok(())
	}

	/// Release a session claim before its deadline.
	///
	/// # Errors
	///
	/// Returns an error when ownership was already lost or the database failed.
	pub fn release_session(&self, lease: &SessionLease) -> Result<(), MemoryError> {
		self.validate_session_lease_origin(lease)?;
		if lease.released.load(Ordering::Acquire) {
			return Ok(());
		}
		let changed = self.connection()?.execute(
			"UPDATE sessions
			 SET execution_token = NULL, execution_lease_until = NULL
			 WHERE id = ?1 AND execution_token = ?2",
			params![lease.session.id.to_string(), lease.token.to_string()],
		)?;
		if changed != 1 {
			return Err(MemoryError::StaleSessionLease {
				session_id: lease.session.id,
			});
		}
		lease.released.store(true, Ordering::Release);
		Ok(())
	}

	/// Replay the effective transcript under an exclusive session claim.
	///
	/// When a completed compaction exists, its verified summary replaces the
	/// covered prefix. The summary is returned first even when it was written
	/// after uncovered tail events. Replay is bounded to protect embedding
	/// processes from corrupt or unexpectedly large databases.
	///
	/// # Errors
	///
	/// Returns an error when the lease expired, ownership/workspace changed,
	/// compaction provenance fails verification, replay exceeds its bounds, or
	/// the database failed.
	pub fn replay_session(&self, lease: &mut SessionLease) -> Result<SessionReplay, MemoryError> {
		self.validate_session_lease_origin(lease)?;
		validate_workspace_identity(&lease.session)?;
		let mut connection = self.connection()?;
		let transaction = connection.transaction_with_behavior(TransactionBehavior::Deferred)?;
		validate_session_lease(&transaction, lease, Utc::now())?;
		let mut replay = load_session_replay(&transaction, lease.session.id)?;
		replay.snapshot = adapter::load_session_snapshot(&transaction, lease.session.id)?;
		transaction.commit()?;
		lease.last_sequence = replay.last_sequence;
		lease.replayed = true;
		Ok(replay)
	}

	/// Append one complete turn/audit batch atomically under a session claim.
	///
	/// Every input is validated before the write transaction. Either the full
	/// batch receives consecutive sequences or no event is committed. Summary
	/// events are reserved for [`MemoryStore::complete_compaction`].
	///
	/// # Errors
	///
	/// Returns an error for an empty/oversized batch, reserved event kind,
	/// expired or lost lease, stale replay cursor, changed workspace identity,
	/// or database failure.
	pub fn append_turn(
		&self,
		lease: &mut SessionLease,
		events: &[SessionEventInput],
	) -> Result<Vec<SessionEvent>, MemoryError> {
		self.append_turn_internal(lease, events, None, None)
	}

	pub(super) fn append_turn_closing_tool_batch(
		&self,
		lease: &mut SessionLease,
		events: &[SessionEventInput],
		turn_id: Uuid,
	) -> Result<Vec<SessionEvent>, MemoryError> {
		self.append_turn_internal(lease, events, Some(turn_id), None)
	}

	pub(super) fn append_turn_closing_agent_turn(
		&self,
		lease: &mut SessionLease,
		events: &[SessionEventInput],
		agent_turn_id: Uuid,
		pending_tool_turn: Option<Uuid>,
	) -> Result<Vec<SessionEvent>, MemoryError> {
		self.append_turn_internal(lease, events, pending_tool_turn, Some(agent_turn_id))
	}

	#[expect(
		clippy::too_many_lines,
		reason = "single SQLite transaction keeps sequence, lease, and turn atomicity explicit"
	)]
	fn append_turn_internal(
		&self,
		lease: &mut SessionLease,
		events: &[SessionEventInput],
		pending_tool_turn: Option<Uuid>,
		active_agent_turn: Option<Uuid>,
	) -> Result<Vec<SessionEvent>, MemoryError> {
		self.validate_session_lease_origin(lease)?;
		let payloads = prepare_turn(events)?;
		if !lease.replayed {
			return Err(MemoryError::ReplayRequired {
				session_id: lease.session.id,
			});
		}
		validate_workspace_identity(&lease.session)?;
		let mut connection = self.connection()?;
		let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
		let now = Utc::now();
		validate_session_lease(&transaction, lease, now)?;
		let current_sequence = last_event_sequence(&transaction, lease.session.id)?;
		if current_sequence != lease.last_sequence {
			lease.replayed = false;
			return Err(MemoryError::StaleReplay {
				session_id: lease.session.id,
				expected_sequence: lease.last_sequence,
				actual_sequence: current_sequence,
			});
		}
		let lease_until = deadline_after(now, SESSION_LEASE, "session lease")?;
		let turn_id = Uuid::now_v7();
		let turn_size = u32::try_from(events.len())
			.map_err(|_| MemoryError::Invalid("turn event count is too large".to_string()))?;
		let mut appended = Vec::with_capacity(events.len());
		for (offset, (input, payload_json)) in events.iter().zip(payloads).enumerate() {
			let increment = u64::try_from(offset)
				.map_err(|_| MemoryError::Invalid("turn event count is too large".to_string()))?
				.checked_add(1)
				.ok_or_else(|| MemoryError::Invalid("turn event count overflow".to_string()))?;
			let sequence = current_sequence.checked_add(increment).ok_or_else(|| {
				MemoryError::Corrupt("session event sequence overflow".to_string())
			})?;
			let sequence_sql = i64::try_from(sequence).map_err(|_| {
				MemoryError::Corrupt("session event sequence exceeds SQLite range".to_string())
			})?;
			let turn_index = u32::try_from(offset)
				.map_err(|_| MemoryError::Invalid("turn event count is too large".to_string()))?;
			let event_id = Uuid::now_v7();
			transaction.execute(
				"INSERT INTO session_events
				 (id, session_id, sequence, turn_id, turn_index, turn_size,
				  kind, payload_json, created_at)
				 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
				params![
					event_id.to_string(),
					lease.session.id.to_string(),
					sequence_sql,
					turn_id.to_string(),
					i64::from(turn_index),
					i64::from(turn_size),
					input.kind.as_str(),
					payload_json,
					now.to_rfc3339(),
				],
			)?;
			assets::record_event_assets(&transaction, lease, event_id, &input.assets, now)?;
			appended.push(SessionEvent {
				id: event_id,
				session_id: lease.session.id,
				sequence,
				turn_id,
				turn_index,
				turn_size,
				kind: input.kind.clone(),
				payload: input.payload.clone(),
				created_at: now,
			});
		}
		if let Some(pending_tool_turn) = pending_tool_turn {
			let changed = transaction.execute(
				"DELETE FROM pending_tool_batches
				 WHERE session_id = ?1 AND turn_id = ?2",
				params![lease.session.id.to_string(), pending_tool_turn.to_string()],
			)?;
			if changed != 1 {
				return Err(MemoryError::Corrupt(format!(
					"session {} has no matching pending tool batch {pending_tool_turn}",
					lease.session.id
				)));
			}
			if active_agent_turn.is_none() {
				let changed = transaction.execute(
					"UPDATE active_agent_turns
					 SET checkpoint_count = checkpoint_count + 1, updated_at = ?3
					 WHERE session_id = ?1 AND turn_id = ?2",
					params![
						lease.session.id.to_string(),
						pending_tool_turn.to_string(),
						now.to_rfc3339(),
					],
				)?;
				if changed != 1 {
					return Err(MemoryError::Corrupt(format!(
						"pending tool batch {pending_tool_turn} has no active agent turn"
					)));
				}
			}
		}
		if let Some(active_agent_turn) = active_agent_turn {
			let changed = transaction.execute(
				"DELETE FROM active_agent_turns
				 WHERE session_id = ?1 AND turn_id = ?2",
				params![lease.session.id.to_string(), active_agent_turn.to_string()],
			)?;
			if changed != 1 {
				return Err(MemoryError::Corrupt(format!(
					"session {} has no matching active agent turn {active_agent_turn}",
					lease.session.id
				)));
			}
		}
		let changed = transaction.execute(
			"UPDATE sessions
			 SET updated_at = ?3, execution_lease_until = ?4
			 WHERE id = ?1 AND execution_token = ?2",
			params![
				lease.session.id.to_string(),
				lease.token.to_string(),
				now.to_rfc3339(),
				lease_until.to_rfc3339(),
			],
		)?;
		if changed != 1 {
			return Err(MemoryError::StaleSessionLease {
				session_id: lease.session.id,
			});
		}
		transaction.commit()?;
		lease.last_sequence = appended
			.last()
			.map_or(current_sequence, |event| event.sequence);
		lease.lease_until = lease_until;
		lease.session.updated_at = now;
		Ok(appended)
	}

	/// Return one newest-first Session page, optionally scoped to a workspace.
	///
	/// # Errors
	///
	/// Returns an error for an invalid workspace, database failure, or corrupt
	/// row.
	pub fn sessions(
		&self,
		workspace: Option<&Path>,
		before: Option<&SessionCursor>,
		limit: usize,
	) -> Result<SessionPage, MemoryError> {
		let limit = page_limit(limit, MAX_SESSION_PAGE_ITEMS, "Session")?;
		let query_limit = limit + 1;
		let cursor_time = before.map(|cursor| cursor.updated_at.to_rfc3339());
		let cursor_id = before.map(|cursor| cursor.id.to_string());
		let connection = self.connection()?;
		if let Some(workspace) = workspace {
			let workspace = workspace_binding(workspace)?;
			let sql = format!(
				"{} AND (?3 IS NULL OR updated_at < ?3
				              OR (updated_at = ?3 AND id < ?4))
				 ORDER BY updated_at DESC, id DESC LIMIT ?5",
				session_select("WHERE workspace_device = ?1 AND workspace_inode = ?2")
			);
			let mut statement = connection.prepare(&sql)?;
			let mapped = statement.query_map(
				params![
					workspace.identity.device.to_string(),
					workspace.identity.inode.to_string(),
					cursor_time,
					cursor_id,
					query_limit
				],
				raw_session,
			)?;
			collect_session_page(mapped, limit)
		} else {
			let sql = format!(
				"{} ORDER BY updated_at DESC, id DESC LIMIT ?3",
				session_select(
					"WHERE ?1 IS NULL OR updated_at < ?1
					    OR (updated_at = ?1 AND id < ?2)",
				)
			);
			let mut statement = connection.prepare(&sql)?;
			let mapped =
				statement.query_map(params![cursor_time, cursor_id, query_limit], raw_session)?;
			collect_session_page(mapped, limit)
		}
	}

	/// Bind a session to one immutable installed-model snapshot.
	///
	/// Repeating the same binding is idempotent. A different binding is
	/// rejected so resumed sessions cannot silently change models.
	///
	/// # Errors
	///
	/// Returns an error for invalid input, a missing session, conflicting
	/// binding, or database failure.
	pub fn bind_session_model(
		&self,
		id: Uuid,
		installed: &InstalledModel,
	) -> Result<(), MemoryError> {
		let home = self.home.as_ref().ok_or_else(|| {
			MemoryError::Invalid(
				"installed-model binding requires MemoryStore::open with an EmelexHome".to_string(),
			)
		})?;
		let _mutation_lock = home.lock_snapshot_mutations()?;
		crate::models::revalidate_installed_snapshot(home, installed)
			.map_err(|error| MemoryError::ModelSnapshot(error.to_string()))?;
		let model_reference = installed.reference().to_string();
		let model_snapshot = installed.snapshot_id().to_string();
		validate_required(
			&model_reference,
			MAX_MODEL_REFERENCE_BYTES,
			"model reference",
		)?;
		validate_required(&model_snapshot, MAX_MODEL_SNAPSHOT_BYTES, "model snapshot")?;
		let mut connection = self.connection()?;
		let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
		let current = transaction
			.query_row(
				"SELECT model_reference, model_snapshot,
					        execution_token, execution_lease_until
					 FROM sessions WHERE id = ?1",
				[id.to_string()],
				|row| {
					Ok((
						row.get::<_, Option<String>>(0)?,
						row.get::<_, Option<String>>(1)?,
						row.get::<_, Option<String>>(2)?,
						row.get::<_, Option<String>>(3)?,
					))
				},
			)
			.optional()?
			.ok_or(MemoryError::NotFound {
				entity: "session",
				id,
			})?;
		let now = Utc::now();
		if let Some((_token, deadline)) = parse_session_claim(current.2, current.3)?
			&& deadline > now
		{
			return Err(MemoryError::SessionBusy {
				session_id: id,
				lease_until: deadline,
			});
		}
		transaction.execute(
			"UPDATE sessions
				 SET execution_token = NULL, execution_lease_until = NULL
				 WHERE id = ?1 AND execution_lease_until <= ?2",
			params![id.to_string(), now.to_rfc3339()],
		)?;
		match (current.0, current.1) {
			(Some(reference), Some(snapshot))
				if reference == model_reference && snapshot == model_snapshot => {}
			(None, None) => {
				transaction.execute(
					"UPDATE sessions
					 SET model_reference = ?2, model_snapshot = ?3, updated_at = ?4
					 WHERE id = ?1 AND model_snapshot IS NULL",
					params![
						id.to_string(),
						&model_reference,
						&model_snapshot,
						now.to_rfc3339()
					],
				)?;
			}
			(Some(reference), None) if reference == model_reference => {
				transaction.execute(
					"UPDATE sessions SET model_snapshot = ?2, updated_at = ?3
					 WHERE id = ?1 AND model_snapshot IS NULL",
					params![id.to_string(), &model_snapshot, now.to_rfc3339()],
				)?;
			}
			_ => {
				return Err(MemoryError::Invalid(format!(
					"session {id} is already bound to another model snapshot"
				)));
			}
		}
		transaction.commit()?;
		Ok(())
	}

	/// Update a session's optional title.
	///
	/// # Errors
	///
	/// Returns an error for invalid input, a missing session, or database
	/// failure.
	pub fn set_session_title(&self, id: Uuid, title: Option<&str>) -> Result<(), MemoryError> {
		validate_optional(title, MAX_TITLE_BYTES, "session title")?;
		let mut connection = self.connection()?;
		let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
		let claim = transaction
			.query_row(
				"SELECT execution_token, execution_lease_until
				 FROM sessions WHERE id = ?1",
				[id.to_string()],
				|row| {
					Ok((
						row.get::<_, Option<String>>(0)?,
						row.get::<_, Option<String>>(1)?,
					))
				},
			)
			.optional()?
			.ok_or(MemoryError::NotFound {
				entity: "session",
				id,
			})?;
		let now = Utc::now();
		if let Some((_token, deadline)) = parse_session_claim(claim.0, claim.1)?
			&& deadline > now
		{
			return Err(MemoryError::SessionBusy {
				session_id: id,
				lease_until: deadline,
			});
		}
		transaction.execute(
			"UPDATE sessions
			 SET title = ?2, updated_at = ?3,
			     execution_token = NULL, execution_lease_until = NULL
			 WHERE id = ?1",
			params![id.to_string(), title, now.to_rfc3339()],
		)?;
		transaction.commit()?;
		Ok(())
	}

	/// Update a claimed Session title and its in-memory lease snapshot.
	///
	/// # Errors
	///
	/// Returns an error for invalid input, changed workspace identity, stale
	/// execution authority, or database failure.
	pub fn set_claimed_session_title(
		&self,
		lease: &mut SessionLease,
		title: Option<&str>,
	) -> Result<(), MemoryError> {
		self.validate_session_lease_origin(lease)?;
		validate_optional(title, MAX_TITLE_BYTES, "session title")?;
		validate_workspace_identity(&lease.session)?;
		let mut connection = self.connection()?;
		let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
		let now = Utc::now();
		validate_session_lease(&transaction, lease, now)?;
		let changed = transaction.execute(
			"UPDATE sessions SET title = ?3, updated_at = ?4
			 WHERE id = ?1 AND execution_token = ?2",
			params![
				lease.session.id.to_string(),
				lease.token.to_string(),
				title,
				now.to_rfc3339()
			],
		)?;
		if changed != 1 {
			return Err(MemoryError::StaleSessionLease {
				session_id: lease.session.id,
			});
		}
		transaction.commit()?;
		lease.session.title = title.map(str::to_string);
		lease.session.updated_at = now;
		Ok(())
	}

	/// Append one JSON event with a transactionally allocated sequence.
	///
	/// # Errors
	///
	/// Returns an error for oversized/non-serializable content, a missing
	/// session, or database failure.
	pub fn append_event(
		&self,
		session_id: Uuid,
		kind: SessionEventKind,
		payload: &Value,
	) -> Result<SessionEvent, MemoryError> {
		if kind == SessionEventKind::Summary {
			return Err(MemoryError::Invalid(
				"summary events may only be written by compaction".to_string(),
			));
		}
		let payload_json = bounded_json_string(payload, MAX_EVENT_BYTES, "session event")?;
		let mut connection = self.connection()?;
		let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
		let now = Utc::now();
		let claim = transaction
			.query_row(
				"SELECT execution_token, execution_lease_until
				 FROM sessions WHERE id = ?1",
				[session_id.to_string()],
				|row| {
					Ok((
						row.get::<_, Option<String>>(0)?,
						row.get::<_, Option<String>>(1)?,
					))
				},
			)
			.optional()?
			.ok_or(MemoryError::NotFound {
				entity: "session",
				id: session_id,
			})?;
		if let Some((_token, deadline)) = parse_session_claim(claim.0, claim.1)?
			&& deadline > now
		{
			return Err(MemoryError::SessionBusy {
				session_id,
				lease_until: deadline,
			});
		}
		transaction.execute(
			"UPDATE sessions
			 SET execution_token = NULL, execution_lease_until = NULL
			 WHERE id = ?1 AND execution_lease_until <= ?2",
			params![session_id.to_string(), now.to_rfc3339()],
		)?;
		let sequence = last_event_sequence(&transaction, session_id)?
			.checked_add(1)
			.ok_or_else(|| MemoryError::Corrupt("session event sequence overflow".to_string()))?;
		let sequence_sql = i64::try_from(sequence).map_err(|_| {
			MemoryError::Corrupt("session event sequence exceeds SQLite range".to_string())
		})?;
		let event_id = Uuid::now_v7();
		transaction.execute(
			"INSERT INTO session_events
			 (id, session_id, sequence, turn_id, turn_index, turn_size,
			  kind, payload_json, created_at)
			 VALUES (?1, ?2, ?3, ?1, 0, 1, ?4, ?5, ?6)",
			params![
				event_id.to_string(),
				session_id.to_string(),
				sequence_sql,
				kind.as_str(),
				payload_json,
				now.to_rfc3339(),
			],
		)?;
		transaction.execute(
			"UPDATE sessions SET updated_at = ?2 WHERE id = ?1",
			params![session_id.to_string(), now.to_rfc3339()],
		)?;
		transaction.commit()?;
		Ok(SessionEvent {
			id: event_id,
			session_id,
			sequence,
			turn_id: event_id,
			turn_index: 0,
			turn_size: 1,
			kind,
			payload: payload.clone(),
			created_at: now,
		})
	}

	/// Return up to `limit` events after the exclusive sequence cursor.
	///
	/// # Errors
	///
	/// Returns an error for a missing session, database failure, or corrupt row.
	pub fn events(
		&self,
		session_id: Uuid,
		after_sequence: u64,
		limit: usize,
	) -> Result<Vec<SessionEvent>, MemoryError> {
		self.session(session_id)?;
		let limit = page_limit(limit, MAX_EVENT_PAGE_ITEMS, "event")?;
		let after_sequence = i64::try_from(after_sequence)
			.map_err(|_| MemoryError::Invalid("event sequence is too large".to_string()))?;
		let connection = self.connection()?;
		let sql = format!(
			"{} ORDER BY sequence ASC LIMIT ?3",
			event_select("WHERE session_id = ?1 AND sequence > ?2")
		);
		let mut statement = connection.prepare(&sql)?;
		let mapped = statement.query_map(
			params![
				session_id.to_string(),
				after_sequence,
				i64::try_from(limit).map_err(|_| {
					MemoryError::Invalid("event page limit is too large".to_string())
				})?
			],
			raw_event,
		)?;
		let mut events = Vec::new();
		let mut payload_bytes = 0_usize;
		for row in mapped {
			let raw = row?;
			let next = payload_bytes
				.checked_add(raw.payload.len())
				.ok_or_else(|| {
					MemoryError::Corrupt("event page byte count overflow".to_string())
				})?;
			if !events.is_empty() && next > MAX_PAGE_PAYLOAD_BYTES {
				break;
			}
			payload_bytes = next;
			events.push(SessionEvent::try_from(raw)?);
		}
		Ok(events)
	}

	/// Delete one session and its events/compaction jobs.
	///
	/// # Errors
	///
	/// Returns an error for a missing session or database failure.
	pub fn delete_session(&self, id: Uuid) -> Result<(), MemoryError> {
		let mut connection = self.connection()?;
		let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
		let claim = transaction
			.query_row(
				"SELECT execution_token, execution_lease_until
				 FROM sessions WHERE id = ?1",
				[id.to_string()],
				|row| {
					Ok((
						row.get::<_, Option<String>>(0)?,
						row.get::<_, Option<String>>(1)?,
					))
				},
			)
			.optional()?
			.ok_or(MemoryError::NotFound {
				entity: "session",
				id,
			})?;
		let now = Utc::now();
		if let Some((_token, deadline)) = parse_session_claim(claim.0, claim.1)?
			&& deadline > now
		{
			return Err(MemoryError::SessionBusy {
				session_id: id,
				lease_until: deadline,
			});
		}
		let worker_deadline: Option<String> = transaction.query_row(
			"SELECT MAX(lease_until) FROM (
			   SELECT lease_until FROM compaction_jobs
			   WHERE session_id = ?1 AND state = 'running'
			   UNION ALL
			   SELECT lease_until FROM distillation_jobs
			   WHERE session_id = ?1 AND state = 'running'
			 )",
			[id.to_string()],
			|row| row.get(0),
		)?;
		if let Some(worker_deadline) = worker_deadline {
			let deadline = parse_time(&worker_deadline, "Session worker lease deadline")?;
			if deadline > now {
				return Err(MemoryError::SessionBusy {
					session_id: id,
					lease_until: deadline,
				});
			}
		}
		let changed =
			transaction.execute("DELETE FROM sessions WHERE id = ?1", [id.to_string()])?;
		ensure_changed(changed, "session", id)?;
		transaction.commit()?;
		Ok(())
	}

	/// Queue idempotent transcript compaction through a sequence.
	///
	/// # Errors
	///
	/// Returns an error for an invalid sequence, missing session, or database
	/// failure.
	pub fn queue_compaction(
		&self,
		session_id: Uuid,
		through_sequence: u64,
	) -> Result<CompactionJob, MemoryError> {
		if through_sequence == 0 {
			return Err(MemoryError::Invalid(
				"compaction sequence must be positive".to_string(),
			));
		}
		let through = i64::try_from(through_sequence)
			.map_err(|_| MemoryError::Invalid("compaction sequence is too large".to_string()))?;
		let mut connection = self.connection()?;
		let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
		let session_text = session_id.to_string();
		let active_boundary = transaction
			.query_row(
				"SELECT through_sequence FROM compaction_jobs
				 WHERE session_id = ?1 AND state IN ('pending', 'running')
				 ORDER BY created_at ASC, id ASC LIMIT 1",
				[&session_text],
				|row| row.get::<_, i64>(0),
			)
			.optional()?;
		if let Some(active_boundary) = active_boundary
			&& active_boundary != through
		{
			return Err(MemoryError::Invalid(format!(
				"session {session_id} already has active compaction through sequence {active_boundary}"
			)));
		}
		let turn_boundary = transaction
			.query_row(
				"SELECT turn_index, turn_size FROM session_events
				 WHERE session_id = ?1 AND sequence = ?2",
				params![session_text, through],
				|row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
			)
			.optional()?;
		let Some((turn_index, turn_size)) = turn_boundary else {
			return Err(MemoryError::Invalid(format!(
				"session {session_id} has no event at sequence {through_sequence}"
			)));
		};
		if turn_index < 0
			|| turn_size <= 0
			|| turn_index
				.checked_add(1)
				.is_none_or(|next| next != turn_size)
		{
			return Err(MemoryError::Invalid(format!(
				"compaction sequence {through_sequence} splits an atomic turn"
			)));
		}
		let prior_summary_sequence: i64 = transaction.query_row(
			"SELECT COALESCE(MAX(e.sequence), 0)
			 FROM compaction_jobs j
			 JOIN session_events e ON e.id = j.summary_event_id
			 WHERE j.session_id = ?1 AND j.state = 'completed'",
			[&session_text],
			|row| row.get(0),
		)?;
		if prior_summary_sequence > through {
			return Err(MemoryError::Invalid(format!(
				"compaction through sequence {through_sequence} must include prior summary event sequence {prior_summary_sequence}"
			)));
		}
		let source = transcript_provenance(&transaction, &session_text, through)?;
		let id = Uuid::now_v7();
		let now = Utc::now();
		transaction.execute(
			"INSERT INTO compaction_jobs
			 (id, session_id, through_sequence, state, claim_token, lease_until,
			  source_event_count, source_first_event_id, source_last_event_id,
			  source_sha256, summary_event_id, created_at, updated_at)
			 VALUES (?1, ?2, ?3, 'pending', NULL, NULL, ?4, ?5, ?6, ?7,
			         NULL, ?8, ?8)
			 ON CONFLICT(session_id, through_sequence) DO NOTHING",
			params![
				id.to_string(),
				session_text,
				through,
				i64::try_from(source.event_count).map_err(|_| {
					MemoryError::Corrupt("compaction source event count overflow".to_string())
				})?,
				source.first_event_id.to_string(),
				source.last_event_id.to_string(),
				source.sha256,
				now.to_rfc3339()
			],
		)?;
		let raw = transaction.query_row(
			&compaction_select("WHERE session_id = ?1 AND through_sequence = ?2"),
			params![session_text, through],
			raw_compaction,
		)?;
		transaction.commit()?;
		CompactionJob::try_from(raw)
	}

	/// Claim the oldest pending compaction whose retry deadline has arrived.
	///
	/// Claims carry a five-minute lease. Before selection, expired claims are
	/// recorded as failed attempts and receive bounded backoff or enter
	/// terminal failed state. They are never silently reclaimed in place.
	///
	/// # Errors
	///
	/// Returns a database/corruption error.
	pub fn claim_compaction(&self) -> Result<Option<CompactionLease>, MemoryError> {
		let mut connection = self.connection()?;
		let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
		let now = Utc::now();
		let now_text = now.to_rfc3339();
		for _ in 0..MAX_STALE_JOB_RECOVERIES {
			if !recover_one_expired_compaction(&transaction, now)? {
				break;
			}
		}
		let claim_sql = format!(
			"{} ORDER BY created_at ASC, id ASC LIMIT 1",
			compaction_select(
				"WHERE state = 'pending'
				   AND (retry_after IS NULL OR retry_after <= ?1)
				   AND NOT EXISTS (
				     SELECT 1 FROM sessions s
				      WHERE s.id = compaction_jobs.session_id
				        AND s.execution_token IS NOT NULL
				        AND s.execution_lease_until > ?1
				   )"
			)
		);
		let raw = transaction
			.query_row(&claim_sql, [&now_text], raw_compaction)
			.optional()?;
		let Some(raw) = raw else {
			transaction.commit()?;
			return Ok(None);
		};
		let token = Uuid::now_v7();
		let lease_until = now
			+ chrono::Duration::from_std(COMPACTION_LEASE).map_err(|_| {
				MemoryError::Invalid("compaction lease duration is invalid".to_string())
			})?;
		let changed = transaction.execute(
			"UPDATE compaction_jobs
			 SET state = 'running', claim_token = ?2, lease_until = ?3,
			     retry_after = NULL, updated_at = ?4
			 WHERE id = ?1
			   AND state = 'pending'
			   AND (retry_after IS NULL OR retry_after <= ?4)",
			params![
				raw.id,
				token.to_string(),
				lease_until.to_rfc3339(),
				now_text
			],
		)?;
		if changed != 1 {
			return Err(MemoryError::Corrupt(
				"compaction claim lost inside immediate transaction".to_string(),
			));
		}
		transaction.commit()?;
		let mut job = CompactionJob::try_from(raw)?;
		job.state = CompactionState::Running;
		job.retry_after = None;
		job.updated_at = now;
		Ok(Some(CompactionLease {
			store: self.clone(),
			job,
			token,
			lease_until,
			released: AtomicBool::new(false),
		}))
	}

	/// Record every expired compaction claim as a failed attempt.
	///
	/// Jobs receive bounded backoff or terminal failed state. Claiming performs
	/// the same recovery in bounded batches; this method drains all currently
	/// expired claims for explicit maintenance.
	///
	/// # Errors
	///
	/// Returns a database failure.
	pub fn recover_stale_compactions(&self) -> Result<usize, MemoryError> {
		let mut connection = self.connection()?;
		let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
		let now = Utc::now();
		let mut changed = 0_usize;
		while recover_one_expired_compaction(&transaction, now)? {
			changed = changed.checked_add(1).ok_or_else(|| {
				MemoryError::Corrupt("recovered compaction count overflow".to_string())
			})?;
		}
		transaction.commit()?;
		Ok(changed)
	}

	/// Extend a claim while the same worker still owns its token.
	///
	/// Renewal remains possible after the prior deadline when no other worker
	/// has reclaimed the job.
	///
	/// # Errors
	///
	/// Returns an error when ownership changed or the database failed.
	pub fn renew_compaction(&self, lease: &mut CompactionLease) -> Result<(), MemoryError> {
		self.validate_compaction_lease_origin(lease)?;
		let now = Utc::now();
		let lease_until = now
			+ chrono::Duration::from_std(COMPACTION_LEASE).map_err(|_| {
				MemoryError::Invalid("compaction lease duration is invalid".to_string())
			})?;
		let changed = self.connection()?.execute(
			"UPDATE compaction_jobs
			 SET lease_until = ?3, updated_at = ?4
			 WHERE id = ?1 AND state = 'running' AND claim_token = ?2",
			params![
				lease.job.id.to_string(),
				lease.token.to_string(),
				lease_until.to_rfc3339(),
				now.to_rfc3339()
			],
		)?;
		if changed != 1 {
			return Err(MemoryError::Invalid(format!(
				"compaction claim {} is stale or belongs to another worker",
				lease.job.id
			)));
		}
		lease.lease_until = lease_until;
		lease.job.updated_at = now;
		Ok(())
	}

	/// Record a bounded worker failure under the current compaction claim.
	///
	/// Retryable failures use persisted exponential backoff. The third
	/// retryable failure, or any permanent failure, moves the job to terminal
	/// failed state until [`MemoryStore::retry_failed_job`] is called.
	///
	/// # Errors
	///
	/// Returns an error for an empty/oversized diagnostic, stale authority, or
	/// database failure.
	pub fn record_compaction_failure(
		&self,
		lease: &CompactionLease,
		error: &str,
		disposition: MemoryJobFailureDisposition,
	) -> Result<MemoryJobFailureOutcome, MemoryError> {
		let outcome = self.record_compaction_failure_inner(lease, error, disposition)?;
		lease.released.store(true, Ordering::Release);
		Ok(outcome)
	}

	fn record_compaction_failure_inner(
		&self,
		lease: &CompactionLease,
		error: &str,
		disposition: MemoryJobFailureDisposition,
	) -> Result<MemoryJobFailureOutcome, MemoryError> {
		self.validate_compaction_lease_origin(lease)?;
		validate_required(error, MAX_JOB_FAILURE_BYTES, "compaction failure")?;
		let mut connection = self.connection()?;
		Self::record_compaction_failure_with_connection(&mut connection, lease, error, disposition)
	}

	fn record_compaction_failure_best_effort(
		&self,
		lease: &CompactionLease,
		error: &str,
		disposition: MemoryJobFailureDisposition,
	) -> Result<MemoryJobFailureOutcome, MemoryError> {
		self.validate_compaction_lease_origin(lease)?;
		validate_required(error, MAX_JOB_FAILURE_BYTES, "compaction failure")?;
		let mut connection = self.best_effort_connection()?;
		Self::record_compaction_failure_with_connection(&mut connection, lease, error, disposition)
	}

	fn record_compaction_failure_with_connection(
		connection: &mut Connection,
		lease: &CompactionLease,
		error: &str,
		disposition: MemoryJobFailureDisposition,
	) -> Result<MemoryJobFailureOutcome, MemoryError> {
		let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
		let failures = transaction
			.query_row(
				"SELECT failure_count FROM compaction_jobs
				 WHERE id = ?1 AND state = 'running' AND claim_token = ?2",
				params![lease.job.id.to_string(), lease.token.to_string()],
				|row| row.get::<_, i64>(0),
			)
			.optional()?
			.ok_or(MemoryError::StaleCompactionLease {
				job_id: lease.job.id,
			})?;
		let outcome = transition_compaction_failure(
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

	/// Return a compaction claim to immediate pending state without counting a
	/// worker failure.
	///
	/// This is intended for explicit operator cancellation. Process crashes
	/// remain observable as lease-expiry failures.
	///
	/// # Errors
	///
	/// Returns stale-authority or database errors.
	pub fn release_compaction(&self, lease: &CompactionLease) -> Result<(), MemoryError> {
		self.validate_compaction_lease_origin(lease)?;
		let changed = self.connection()?.execute(
			"UPDATE compaction_jobs
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
			return Err(MemoryError::StaleCompactionLease {
				job_id: lease.job.id,
			});
		}
		lease.released.store(true, Ordering::Release);
		Ok(())
	}

	/// Complete a claimed compaction and append its summary atomically.
	///
	/// # Errors
	///
	/// Returns an error for oversized content, an expired/foreign claim,
	/// missing records, or database failure.
	pub fn complete_compaction(
		&self,
		lease: &CompactionLease,
		summary: &Value,
	) -> Result<SessionEvent, MemoryError> {
		self.validate_compaction_lease_origin(lease)?;
		let (payload, payload_json) = prepare_compaction_summary(lease, summary)?;
		let mut connection = self.connection()?;
		let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
		let now = Utc::now();
		let raw = validate_compaction_lease(&transaction, lease, now)?;
		let execution = transaction.query_row(
			"SELECT execution_token, execution_lease_until
			 FROM sessions WHERE id = ?1",
			[&raw.session_id],
			|row| {
				Ok((
					row.get::<_, Option<String>>(0)?,
					row.get::<_, Option<String>>(1)?,
				))
			},
		)?;
		if let Some((_token, deadline)) = parse_session_claim(execution.0, execution.1)?
			&& deadline > now
		{
			return Err(MemoryError::SessionBusy {
				session_id: parse_uuid(&raw.session_id, "compaction session ID")?,
				lease_until: deadline,
			});
		}
		let token_text = lease.token.to_string();
		let session_id = parse_uuid(&raw.session_id, "compaction session ID")?;
		let sequence = last_event_sequence(&transaction, session_id)?
			.checked_add(1)
			.ok_or_else(|| MemoryError::Corrupt("session event sequence overflow".to_string()))?;
		let sequence_sql = i64::try_from(sequence).map_err(|_| {
			MemoryError::Corrupt("session event sequence exceeds SQLite range".to_string())
		})?;
		let event_id = Uuid::now_v7();
		transaction.execute(
			"INSERT INTO session_events
			 (id, session_id, sequence, turn_id, turn_index, turn_size,
			  kind, payload_json, created_at)
			 VALUES (?1, ?2, ?3, ?1, 0, 1, 'summary', ?4, ?5)",
			params![
				event_id.to_string(),
				raw.session_id,
				sequence_sql,
				payload_json,
				now.to_rfc3339(),
			],
		)?;
		let changed = transaction.execute(
			"UPDATE compaction_jobs
			 SET state = 'completed', claim_token = NULL, lease_until = NULL,
			     retry_after = NULL, last_error = NULL, failed_at = NULL,
			     summary_event_id = ?3, updated_at = ?4
			 WHERE id = ?1 AND state = 'running' AND claim_token = ?2",
			params![
				lease.job.id.to_string(),
				token_text,
				event_id.to_string(),
				now.to_rfc3339()
			],
		)?;
		if changed != 1 {
			return Err(MemoryError::Invalid(format!(
				"compaction claim {} was lost before completion",
				lease.job.id
			)));
		}
		transaction.execute(
			"UPDATE sessions SET updated_at = ?2 WHERE id = ?1",
			params![raw.session_id, now.to_rfc3339()],
		)?;
		transaction.commit()?;
		lease.released.store(true, Ordering::Release);
		Ok(SessionEvent {
			id: event_id,
			session_id,
			sequence,
			turn_id: event_id,
			turn_index: 0,
			turn_size: 1,
			kind: SessionEventKind::Summary,
			payload,
			created_at: now,
		})
	}

	/// Store a new version for one workspace-scoped Knowledge key.
	///
	/// New versions become active. Pin state is preserved.
	///
	/// # Errors
	///
	/// Returns an error for invalid fields/workspace or database failure.
	#[expect(
		clippy::too_many_lines,
		reason = "identity validation and version append share one atomic Knowledge operation"
	)]
	pub fn remember(
		&self,
		workspace: &Path,
		key: &str,
		content: &str,
		source_session_id: Option<Uuid>,
	) -> Result<Knowledge, MemoryError> {
		validate_required(key, MAX_KNOWLEDGE_KEY_BYTES, "Knowledge key")?;
		validate_required(content, MAX_KNOWLEDGE_BYTES, "Knowledge content")?;
		let workspace = workspace_binding(workspace)?;
		let mut connection = self.connection()?;
		let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
		if let Some(session_id) = source_session_id {
			let source_identity = transaction
				.query_row(
					"SELECT workspace_device, workspace_inode
					 FROM sessions WHERE id = ?1",
					[session_id.to_string()],
					|row| {
						Ok((
							row.get::<_, Option<String>>(0)?,
							row.get::<_, Option<String>>(1)?,
						))
					},
				)
				.optional()?
				.ok_or(MemoryError::NotFound {
					entity: "session",
					id: session_id,
				})?;
			if parse_workspace_identity(source_identity.0, source_identity.1)? != workspace.identity
			{
				return Err(MemoryError::Invalid(
					"Knowledge source session belongs to another workspace".to_string(),
				));
			}
		}
		let workspace_text = path_text(&workspace.path)?;
		let device = workspace.identity.device.to_string();
		let inode = workspace.identity.inode.to_string();
		let existing = transaction
			.query_row(
				"SELECT id, active_version, pinned, created_at
				 FROM knowledge
				 WHERE workspace_device = ?1 AND workspace_inode = ?2 AND key = ?3",
				params![&device, &inode, key],
				|row| {
					Ok((
						row.get::<_, String>(0)?,
						row.get::<_, i64>(1)?,
						row.get::<_, bool>(2)?,
						row.get::<_, String>(3)?,
					))
				},
			)
			.optional()?;
		let now = Utc::now();
		let (id, version, pinned, created_at) =
			if let Some((id, _active_version, pinned, created_at)) = existing {
				let version: i64 = transaction.query_row(
					"SELECT COALESCE(MAX(version), 0) + 1
					 FROM knowledge_versions WHERE knowledge_id = ?1",
					[&id],
					|row| row.get(0),
				)?;
				if version <= 0 {
					return Err(MemoryError::Corrupt(
						"Knowledge version overflow".to_string(),
					));
				}
				(
					parse_uuid(&id, "Knowledge ID")?,
					version,
					pinned,
					parse_time(&created_at, "Knowledge creation time")?,
				)
			} else {
				(Uuid::now_v7(), 1, false, now)
			};
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
			   tombstoned = 0,
			   tombstoned_at = NULL,
			   updated_at = excluded.updated_at",
			params![
				id.to_string(),
				workspace_text,
				&device,
				&inode,
				key,
				version,
				pinned,
				created_at.to_rfc3339(),
				now.to_rfc3339()
			],
		)?;
		transaction.execute(
			"INSERT INTO knowledge_versions
			 (knowledge_id, version, content, confidence, source_session_id,
			  provenance_session_id, source_first_sequence, source_last_sequence, source_sha256,
			  distillation_job_id, candidate_index, created_at)
			 VALUES (?1, ?2, ?3, 1.0, ?4, NULL, NULL, NULL, NULL, NULL, NULL, ?5)",
			params![
				id.to_string(),
				version,
				content,
				source_session_id.map(|value| value.to_string()),
				now.to_rfc3339(),
			],
		)?;
		transaction.commit()?;
		Ok(Knowledge {
			id,
			workspace: workspace.path,
			workspace_identity: workspace.identity,
			key: key.to_string(),
			active_version: public_version,
			content: content.to_string(),
			confidence: 1.0,
			source_session_id,
			pinned,
			tombstoned: false,
			created_at,
			updated_at: now,
		})
	}

	/// Fetch active Knowledge by ID.
	///
	/// # Errors
	///
	/// Returns [`MemoryError::NotFound`] or a database/corruption error.
	pub fn knowledge(&self, id: Uuid) -> Result<Knowledge, MemoryError> {
		let connection = self.connection()?;
		let raw = connection
			.query_row(
				&knowledge_select("WHERE k.id = ?1"),
				[id.to_string()],
				raw_knowledge,
			)
			.optional()?
			.ok_or(MemoryError::NotFound {
				entity: "Knowledge",
				id,
			})?;
		Knowledge::try_from(raw)
	}

	/// List active Knowledge for one workspace, pinned entries first.
	///
	/// # Errors
	///
	/// Returns an error for an invalid workspace, database failure, or corrupt
	/// row.
	pub fn knowledge_for_workspace(
		&self,
		workspace: &Path,
		after: Option<&KnowledgeCursor>,
		limit: usize,
	) -> Result<KnowledgePage, MemoryError> {
		let workspace = workspace_binding(workspace)?;
		let limit = page_limit(limit, MAX_KNOWLEDGE_PAGE_ITEMS, "Knowledge")?;
		let query_limit = limit.saturating_add(1);
		let cursor_pinned = after.map(|cursor| cursor.pinned);
		let cursor_time = after.map(|cursor| cursor.updated_at.to_rfc3339());
		let cursor_key = after.map(|cursor| cursor.key.as_str());
		let connection = self.connection()?;
		let sql = format!(
			"{} ORDER BY k.pinned DESC, k.updated_at DESC, k.key ASC LIMIT ?6",
			knowledge_select(
				"WHERE k.workspace_device = ?1 AND k.workspace_inode = ?2
				   AND k.tombstoned = 0
				   AND (?3 IS NULL OR k.pinned < ?3
				        OR (k.pinned = ?3 AND k.updated_at < ?4)
				        OR (k.pinned = ?3 AND k.updated_at = ?4 AND k.key > ?5))",
			)
		);
		let mut statement = connection.prepare(&sql)?;
		let mapped = statement.query_map(
			params![
				workspace.identity.device.to_string(),
				workspace.identity.inode.to_string(),
				cursor_pinned,
				cursor_time,
				cursor_key,
				i64::try_from(query_limit).map_err(|_| {
					MemoryError::Invalid("Knowledge page limit is too large".to_string())
				})?
			],
			raw_knowledge,
		)?;
		collect_knowledge_page(mapped, limit)
	}

	/// Recall active Knowledge above a minimum confidence for one workspace.
	///
	/// Filtering happens in `SQLite` before the result limit is applied, so
	/// low-confidence entries cannot crowd eligible entries out of automatic
	/// recall. Pinned entries retain ordering priority but do not bypass the
	/// confidence floor.
	///
	/// # Errors
	///
	/// Returns an error for an invalid confidence/limit/workspace, database
	/// failure, or corrupt row.
	pub fn recall_knowledge(
		&self,
		workspace: &Path,
		minimum_confidence: f64,
		limit: usize,
	) -> Result<Vec<Knowledge>, MemoryError> {
		if !minimum_confidence.is_finite() || !(0.0..=1.0).contains(&minimum_confidence) {
			return Err(MemoryError::Invalid(
				"Knowledge recall confidence must be finite and in 0..=1".to_string(),
			));
		}
		let workspace = workspace_binding(workspace)?;
		let limit = page_limit(limit, MAX_KNOWLEDGE_PAGE_ITEMS, "Knowledge recall")?;
		let connection = self.connection()?;
		let sql = format!(
			"{} ORDER BY k.pinned DESC, k.updated_at DESC, k.key ASC LIMIT ?4",
			knowledge_select(
				"WHERE k.workspace_device = ?1 AND k.workspace_inode = ?2
				   AND k.tombstoned = 0 AND v.confidence >= ?3",
			)
		);
		let mut statement = connection.prepare(&sql)?;
		let mapped = statement.query_map(
			params![
				workspace.identity.device.to_string(),
				workspace.identity.inode.to_string(),
				minimum_confidence,
				i64::try_from(limit).map_err(|_| {
					MemoryError::Invalid("Knowledge recall limit is too large".to_string())
				})?
			],
			raw_knowledge,
		)?;
		mapped
			.collect::<Result<Vec<_>, _>>()?
			.into_iter()
			.map(Knowledge::try_from)
			.collect()
	}

	/// Search active Knowledge content and keys in one workspace.
	///
	/// # Errors
	///
	/// Returns an error for invalid input/workspace, database failure, or
	/// corrupt row.
	pub fn search_knowledge(
		&self,
		workspace: &Path,
		query: &str,
		limit: usize,
	) -> Result<Vec<Knowledge>, MemoryError> {
		validate_required(query, MAX_KNOWLEDGE_KEY_BYTES, "Knowledge query")?;
		let limit = page_limit(limit, MAX_KNOWLEDGE_PAGE_ITEMS, "Knowledge search")?;
		let workspace = workspace_binding(workspace)?;
		let connection = self.connection()?;
		let sql = format!(
			"{} AND
			 (instr(lower(k.key), lower(?3)) > 0
			  OR instr(lower(v.content), lower(?3)) > 0)
			 ORDER BY k.pinned DESC,
			  CASE WHEN lower(k.key) = lower(?3) THEN 0 ELSE 1 END,
			  k.updated_at DESC
			 LIMIT ?4",
			knowledge_select(
				"WHERE k.workspace_device = ?1 AND k.workspace_inode = ?2
				   AND k.tombstoned = 0",
			)
		);
		let mut statement = connection.prepare(&sql)?;
		let mapped = statement.query_map(
			params![
				workspace.identity.device.to_string(),
				workspace.identity.inode.to_string(),
				query,
				i64::try_from(limit)
					.map_err(|_| MemoryError::Invalid("search limit is too large".to_string()))?
			],
			raw_knowledge,
		)?;
		collect_knowledge_bounded(mapped)
	}

	/// Return up to `limit` versions below the exclusive version cursor.
	///
	/// # Errors
	///
	/// Returns an error for missing Knowledge, database failure, or corrupt row.
	pub fn knowledge_history(
		&self,
		id: Uuid,
		before_version: Option<u32>,
		limit: usize,
	) -> Result<Vec<KnowledgeVersion>, MemoryError> {
		self.knowledge(id)?;
		let limit = page_limit(limit, MAX_KNOWLEDGE_PAGE_ITEMS, "Knowledge history")?;
		let before = before_version.map_or(i64::MAX, i64::from);
		let connection = self.connection()?;
		let mut statement = connection.prepare(
			"SELECT knowledge_id, version, content, confidence, source_session_id,
			        provenance_session_id, source_first_sequence, source_last_sequence, source_sha256,
			        distillation_job_id, candidate_index, created_at
			 FROM knowledge_versions
			 WHERE knowledge_id = ?1 AND version < ?2
			 ORDER BY version DESC LIMIT ?3",
		)?;
		let mapped = statement.query_map(
			params![
				id.to_string(),
				before,
				i64::try_from(limit).map_err(|_| {
					MemoryError::Invalid("Knowledge history limit is too large".to_string())
				})?
			],
			raw_knowledge_version,
		)?;
		let mut versions = Vec::new();
		let mut payload_bytes = 0_usize;
		for row in mapped {
			let raw = row?;
			let next = payload_bytes
				.checked_add(raw.content.len())
				.ok_or_else(|| {
					MemoryError::Corrupt("Knowledge history byte count overflow".to_string())
				})?;
			if !versions.is_empty() && next > MAX_PAGE_PAYLOAD_BYTES {
				break;
			}
			payload_bytes = next;
			versions.push(KnowledgeVersion::try_from(raw)?);
		}
		Ok(versions)
	}

	/// Activate one historical Knowledge version.
	///
	/// # Errors
	///
	/// Returns an error for missing Knowledge/version or database failure.
	pub fn activate_knowledge(
		&self,
		workspace: &Path,
		id: Uuid,
		version: u32,
	) -> Result<(), MemoryError> {
		if version == 0 {
			return Err(MemoryError::Invalid(
				"Knowledge version must be positive".to_string(),
			));
		}
		let workspace = workspace_binding(workspace)?;
		let mut connection = self.connection()?;
		let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
		let exists = transaction
			.query_row(
				"SELECT 1
				 FROM knowledge_versions v
				 JOIN knowledge k ON k.id = v.knowledge_id
				 WHERE v.knowledge_id = ?1 AND v.version = ?2
				   AND k.workspace_device = ?3 AND k.workspace_inode = ?4",
				params![
					id.to_string(),
					i64::from(version),
					workspace.identity.device.to_string(),
					workspace.identity.inode.to_string()
				],
				|row| row.get::<_, i64>(0),
			)
			.optional()?
			.is_some();
		if !exists {
			return Err(MemoryError::Invalid(format!(
				"Knowledge {id} has no version {version}"
			)));
		}
		let changed = transaction.execute(
			"UPDATE knowledge
			 SET active_version = ?2, tombstoned = 0, tombstoned_at = NULL,
			     updated_at = ?3
			 WHERE id = ?1 AND workspace_device = ?4 AND workspace_inode = ?5",
			params![
				id.to_string(),
				i64::from(version),
				Utc::now().to_rfc3339(),
				workspace.identity.device.to_string(),
				workspace.identity.inode.to_string()
			],
		)?;
		ensure_changed(changed, "Knowledge", id)?;
		transaction.commit()?;
		Ok(())
	}

	/// Pin or unpin one Knowledge entry.
	///
	/// # Errors
	///
	/// Returns an error for missing Knowledge or database failure.
	pub fn set_knowledge_pinned(
		&self,
		workspace: &Path,
		id: Uuid,
		pinned: bool,
	) -> Result<(), MemoryError> {
		let workspace = workspace_binding(workspace)?;
		let changed = self.connection()?.execute(
			"UPDATE knowledge SET pinned = ?2, updated_at = ?3
			 WHERE id = ?1 AND workspace_device = ?4 AND workspace_inode = ?5",
			params![
				id.to_string(),
				pinned,
				Utc::now().to_rfc3339(),
				workspace.identity.device.to_string(),
				workspace.identity.inode.to_string()
			],
		)?;
		ensure_changed(changed, "Knowledge", id)
	}

	/// Tombstone one Knowledge entry for later retention.
	///
	/// # Errors
	///
	/// Returns an error for missing Knowledge or database failure.
	pub fn delete_knowledge(&self, workspace: &Path, id: Uuid) -> Result<(), MemoryError> {
		let workspace = workspace_binding(workspace)?;
		let now = Utc::now();
		let mut connection = self.connection()?;
		let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
		let changed = transaction.execute(
			"UPDATE knowledge
			 SET tombstoned = 1, tombstoned_at = ?2, pinned = 0, updated_at = ?2
			 WHERE id = ?1 AND workspace_device = ?3 AND workspace_inode = ?4
			   AND tombstoned = 0",
			params![
				id.to_string(),
				now.to_rfc3339(),
				workspace.identity.device.to_string(),
				workspace.identity.inode.to_string()
			],
		)?;
		if changed == 0 {
			let tombstoned = transaction
				.query_row(
					"SELECT tombstoned FROM knowledge
					 WHERE id = ?1 AND workspace_device = ?2 AND workspace_inode = ?3",
					params![
						id.to_string(),
						workspace.identity.device.to_string(),
						workspace.identity.inode.to_string()
					],
					|row| row.get::<_, bool>(0),
				)
				.optional()?
				.ok_or(MemoryError::NotFound {
					entity: "Knowledge",
					id,
				})?;
			if tombstoned {
				transaction.commit()?;
				return Ok(());
			}
			return Err(MemoryError::Corrupt(format!(
				"Knowledge {id} could not be tombstoned"
			)));
		}
		transaction.execute(
			"INSERT INTO knowledge_tombstones
			 (id, knowledge_id, confidence, provenance_session_id,
			  source_first_sequence, source_last_sequence, source_sha256,
			  distillation_job_id, candidate_index, origin, created_at)
			 VALUES (?1, ?2, 1.0, NULL, NULL, NULL, NULL,
			         NULL, NULL, 'manual', ?3)",
			params![Uuid::now_v7().to_string(), id.to_string(), now.to_rfc3339()],
		)?;
		transaction.commit()?;
		Ok(())
	}

	/// List terminal worker failures newest-first.
	///
	/// # Errors
	///
	/// Returns an error when `limit` is outside `1..=500`, persisted failure
	/// data is corrupt, or `SQLite` fails.
	pub fn failed_jobs(&self, limit: usize) -> Result<Vec<MemoryJobFailure>, MemoryError> {
		if !(1..=MAX_JOB_FAILURE_PAGE_ITEMS).contains(&limit) {
			return Err(MemoryError::Invalid(format!(
				"failed-job limit must be in 1..={MAX_JOB_FAILURE_PAGE_ITEMS}"
			)));
		}
		let connection = self.connection()?;
		let mut statement = connection.prepare(
			"SELECT kind, id, session_id, failure_count, last_error, failed_at
			 FROM (
			   SELECT 'compaction' AS kind, id, session_id, failure_count,
			          last_error, failed_at
			   FROM compaction_jobs WHERE state = 'failed'
			   UNION ALL
			   SELECT 'distillation' AS kind, id, session_id, failure_count,
			          last_error, failed_at
			   FROM distillation_jobs WHERE state = 'failed'
			 )
			 ORDER BY failed_at DESC, id DESC LIMIT ?1",
		)?;
		let rows = statement.query_map(
			[i64::try_from(limit)
				.map_err(|_| MemoryError::Invalid("failed-job limit is too large".to_string()))?],
			|row| {
				Ok((
					row.get::<_, String>(0)?,
					row.get::<_, String>(1)?,
					row.get::<_, String>(2)?,
					row.get::<_, i64>(3)?,
					row.get::<_, Option<String>>(4)?,
					row.get::<_, Option<String>>(5)?,
				))
			},
		)?;
		let mut failures = Vec::new();
		for row in rows {
			let (kind, id, session_id, failure_count, error, failed_at) = row?;
			let kind = match kind.as_str() {
				"compaction" => MemoryJobKind::Compaction,
				"distillation" => MemoryJobKind::Distillation,
				other => {
					return Err(MemoryError::Corrupt(format!(
						"unknown failed worker job kind {other:?}"
					)));
				}
			};
			failures.push(MemoryJobFailure {
				kind,
				id: parse_uuid(&id, "failed worker job ID")?,
				session_id: parse_uuid(&session_id, "failed worker Session ID")?,
				failures: u32::try_from(failure_count).map_err(|_| {
					MemoryError::Corrupt("failed worker job has invalid failure count".to_string())
				})?,
				error: error.ok_or_else(|| {
					MemoryError::Corrupt("failed worker job has no diagnostic".to_string())
				})?,
				failed_at: parse_time(
					failed_at.as_deref().ok_or_else(|| {
						MemoryError::Corrupt("failed worker job has no failure time".to_string())
					})?,
					"worker job failure time",
				)?,
			});
		}
		Ok(failures)
	}

	/// Explicitly reset one terminal worker job for immediate retry.
	///
	/// The bounded failure counter and diagnostic are cleared. The immutable
	/// source provenance and job identity are retained.
	///
	/// # Errors
	///
	/// Returns [`MemoryError::NotFound`] unless `id` names exactly one failed
	/// job, or a database error.
	pub fn retry_failed_job(&self, id: Uuid) -> Result<(), MemoryError> {
		let mut connection = self.connection()?;
		let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
		let now = Utc::now().to_rfc3339();
		let compactions = transaction.execute(
			"UPDATE compaction_jobs
			 SET state = 'pending', failure_count = 0, retry_after = NULL,
			     last_error = NULL, failed_at = NULL, updated_at = ?2
			 WHERE id = ?1 AND state = 'failed'",
			params![id.to_string(), &now],
		)?;
		let distillations = transaction.execute(
			"UPDATE distillation_jobs
			 SET state = 'pending', failure_count = 0, retry_after = NULL,
			     last_error = NULL, failed_at = NULL, updated_at = ?2
			 WHERE id = ?1 AND state = 'failed'",
			params![id.to_string(), &now],
		)?;
		match compactions.saturating_add(distillations) {
			1 => {
				transaction.commit()?;
				Ok(())
			}
			0 => Err(MemoryError::NotFound {
				entity: "failed memory job",
				id,
			}),
			_ => Err(MemoryError::Corrupt(format!(
				"worker job ID {id} exists in multiple queues"
			))),
		}
	}

	/// Aggregate durable-store counts and file size.
	///
	/// # Errors
	///
	/// Returns a database or filesystem error.
	pub fn status(&self) -> Result<MemoryStatus, MemoryError> {
		let connection = self.connection()?;
		let sessions = count(&connection, "sessions")?;
		let events = count(&connection, "session_events")?;
		let knowledge = count(&connection, "knowledge")?;
		let pending_compactions: u64 = connection.query_row(
			"SELECT COUNT(*) FROM compaction_jobs WHERE state IN ('pending', 'running')",
			[],
			|row| row.get(0),
		)?;
		let failed_compactions: u64 = connection.query_row(
			"SELECT COUNT(*) FROM compaction_jobs WHERE state = 'failed'",
			[],
			|row| row.get(0),
		)?;
		let assets: u64 =
			connection.query_row("SELECT COUNT(*) FROM assets", [], |row| row.get(0))?;
		let asset_bytes: u64 =
			connection.query_row("SELECT COALESCE(SUM(bytes), 0) FROM assets", [], |row| {
				row.get(0)
			})?;
		let pending_distillations: u64 = connection.query_row(
			"SELECT COUNT(*) FROM distillation_jobs
			 WHERE state IN ('pending', 'running')",
			[],
			|row| row.get(0),
		)?;
		let failed_distillations: u64 = connection.query_row(
			"SELECT COUNT(*) FROM distillation_jobs WHERE state = 'failed'",
			[],
			|row| row.get(0),
		)?;
		let tombstoned_knowledge: u64 = connection.query_row(
			"SELECT COUNT(*) FROM knowledge WHERE tombstoned = 1",
			[],
			|row| row.get(0),
		)?;
		let mut database_bytes = file_size(&self.database)?;
		for suffix in ["-wal", "-shm"] {
			let sidecar = PathBuf::from(format!("{}{suffix}", self.database.display()));
			match fs::metadata(&sidecar) {
				Ok(metadata) => {
					database_bytes =
						database_bytes.checked_add(metadata.len()).ok_or_else(|| {
							MemoryError::Corrupt("memory database byte count overflow".to_string())
						})?;
				}
				Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
				Err(source) => {
					return Err(MemoryError::Io {
						operation: "inspect memory database sidecar",
						path: sidecar,
						source,
					});
				}
			}
		}
		Ok(MemoryStatus {
			database_bytes,
			sessions,
			events,
			knowledge,
			pending_compactions,
			failed_compactions,
			assets,
			asset_bytes,
			pending_distillations,
			failed_distillations,
			tombstoned_knowledge,
		})
	}

	fn connection(&self) -> Result<Connection, MemoryError> {
		let flags = OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NOFOLLOW;
		let connection = Connection::open_with_flags(&self.database, flags)?;
		connection.busy_timeout(Duration::from_secs(5))?;
		configure_memory_temp_store(&connection)?;
		connection.pragma_update(None, "foreign_keys", true)?;
		connection.pragma_update(None, "trusted_schema", false)?;
		connection.pragma_update(None, "journal_mode", "WAL")?;
		connection.pragma_update(None, "synchronous", "EXTRA")?;
		connection.pragma_update(None, "fullfsync", true)?;
		connection.pragma_update(None, "checkpoint_fullfsync", true)?;
		Ok(connection)
	}

	fn best_effort_connection(&self) -> Result<Connection, MemoryError> {
		let flags = OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NOFOLLOW;
		let connection = Connection::open_with_flags(&self.database, flags)?;
		connection.busy_timeout(Duration::ZERO)?;
		configure_memory_temp_store(&connection)?;
		connection.pragma_update(None, "foreign_keys", true)?;
		connection.pragma_update(None, "trusted_schema", false)?;
		Ok(connection)
	}

	fn migrate(connection: &mut Connection) -> Result<(), MemoryError> {
		let transaction = connection.transaction_with_behavior(TransactionBehavior::Exclusive)?;
		let mut current: i64 =
			transaction.pragma_query_value(None, "user_version", |row| row.get(0))?;
		if current > SCHEMA_VERSION {
			return Err(MemoryError::NewerSchema {
				found: current,
				supported: SCHEMA_VERSION,
			});
		}
		if current == 0 {
			transaction.execute_batch(CREATE_SCHEMA_V6)?;
			current = 6;
		}
		if current == 1 {
			transaction.execute_batch(MIGRATE_V1_TO_V2)?;
			current = 2;
		}
		if current == 2 {
			transaction.execute_batch(PREPARE_V2_TO_V3)?;
			migrate_v3_rows(&transaction)?;
			transaction.execute_batch(FINISH_V2_TO_V3)?;
			current = 3;
		}
		if current == 3 {
			migrate_v3_to_v4(&transaction)?;
			current = 4;
		}
		if current == 4 {
			transaction.execute_batch(MIGRATE_V4_TO_V5)?;
			current = 5;
		}
		if current == 5 {
			transaction.execute_batch(MIGRATE_V5_TO_V6)?;
			current = 6;
		}
		if current != SCHEMA_VERSION {
			return Err(MemoryError::Corrupt(format!(
				"memory migration stopped at unexpected schema {current}"
			)));
		}
		let foreign_key_violation = transaction
			.query_row("PRAGMA foreign_key_check", [], |_| Ok(()))
			.optional()?
			.is_some();
		if foreign_key_violation {
			return Err(MemoryError::Corrupt(
				"memory migration produced a foreign-key violation".to_string(),
			));
		}
		transaction.commit()?;
		Ok(())
	}
}

/// Stable local filesystem identity for one workspace directory.
///
/// Emelex records the Unix device and inode reported by `macOS`. Canonical paths
/// remain useful presentation metadata, while this identity detects a directory
/// replaced at the same path and survives ordinary renames on the same volume.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub struct WorkspaceIdentity {
	device: u64,
	inode: u64,
}

impl WorkspaceIdentity {
	/// Filesystem device number.
	pub const fn device(self) -> u64 {
		self.device
	}

	/// Filesystem inode number.
	pub const fn inode(self) -> u64 {
		self.inode
	}
}

/// Durable conversation identity and metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Session {
	/// `UUIDv7` session identity.
	pub id: Uuid,
	/// Canonical workspace root.
	pub workspace: PathBuf,
	/// Stable filesystem identity captured for the workspace.
	pub workspace_identity: WorkspaceIdentity,
	/// Selected model reference at session creation.
	pub model_reference: Option<ModelRef>,
	/// Immutable installed snapshot identity, once bound.
	pub model_snapshot: Option<ModelSnapshotId>,
	/// Optional user-facing title.
	pub title: Option<String>,
	/// Creation time.
	pub created_at: DateTime<Utc>,
	/// Last event/metadata update time.
	pub updated_at: DateTime<Utc>,
}

impl Session {
	/// Verify that `workspace` is the same filesystem object this Session
	/// captured, allowing ordinary directory renames on the same volume.
	///
	/// # Errors
	///
	/// Returns a filesystem error or [`MemoryError::WorkspaceMismatch`].
	pub fn validate_workspace(&self, workspace: &Path) -> Result<(), MemoryError> {
		let actual = workspace_binding(workspace)?.identity;
		if actual != self.workspace_identity {
			return Err(MemoryError::WorkspaceMismatch {
				session_id: self.id,
				expected: self.workspace_identity,
				actual,
			});
		}
		Ok(())
	}
}

/// Exclusive, expiring authority to replay and append one durable session.
///
/// This type is intentionally not `Clone` or `Deserialize`; callers cannot
/// duplicate or reconstruct write authority from serialized data.
#[non_exhaustive]
pub struct SessionLease {
	store: MemoryStore,
	session: Session,
	token: Uuid,
	lease_until: DateTime<Utc>,
	last_sequence: u64,
	replayed: bool,
	released: AtomicBool,
}

impl fmt::Debug for SessionLease {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter
			.debug_struct("SessionLease")
			.field("session", &self.session)
			.field("token", &"<redacted>")
			.field("lease_until", &self.lease_until)
			.field("last_sequence", &self.last_sequence)
			.field("replayed", &self.replayed)
			.finish_non_exhaustive()
	}
}

impl SessionLease {
	/// Session metadata validated at claim time.
	pub const fn session(&self) -> &Session {
		&self.session
	}

	/// Deadline after which another process may reclaim the session.
	pub const fn lease_until(&self) -> DateTime<Utc> {
		self.lease_until
	}

	/// Last durable sequence observed by this authority.
	pub const fn last_sequence(&self) -> u64 {
		self.last_sequence
	}
}

impl Drop for SessionLease {
	fn drop(&mut self) {
		if self.released.swap(true, Ordering::AcqRel) {
			return;
		}
		let Ok(connection) = self.store.best_effort_connection() else {
			return;
		};
		let _ = connection.execute(
			"UPDATE sessions
			 SET execution_token = NULL, execution_lease_until = NULL
			 WHERE id = ?1 AND execution_token = ?2",
			params![self.session.id.to_string(), self.token.to_string()],
		);
	}
}

/// One event proposed for an atomic turn append.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SessionEventInput {
	/// Event category.
	pub kind: SessionEventKind,
	/// Structured event body.
	pub payload: Value,
	/// Content-addressed assets referenced by this event.
	#[serde(default, skip_serializing_if = "Vec::is_empty")]
	pub assets: Vec<AssetRef>,
}

impl SessionEventInput {
	/// Construct one proposed event.
	pub const fn new(kind: SessionEventKind, payload: Value) -> Self {
		Self {
			kind,
			payload,
			assets: Vec::new(),
		}
	}

	/// Attach already-persisted assets in payload reference order.
	#[must_use]
	pub fn with_assets(mut self, assets: Vec<AssetRef>) -> Self {
		self.assets = assets;
		self
	}
}

/// Effective transcript returned for agent-session reconstruction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SessionReplay {
	/// Verified summary followed by uncovered tail events, or all raw events
	/// when no compaction completed.
	pub events: Vec<SessionEvent>,
	/// Highest raw event sequence observed in the same `SQLite` snapshot.
	pub last_sequence: u64,
	/// Inclusive raw prefix replaced by the leading summary, when present.
	pub compacted_through: Option<u64>,
	/// Immutable semantic configuration and tool authority, when stored.
	pub snapshot: Option<SessionSnapshot>,
}

/// Stable cursor for the next newest-first Session page.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionCursor {
	updated_at: DateTime<Utc>,
	id: Uuid,
}

/// One bounded Session page.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SessionPage {
	/// Sessions in newest-first order.
	pub items: Vec<Session>,
	/// Pass this cursor to fetch the next page.
	pub next: Option<SessionCursor>,
}

/// Durable event category.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum SessionEventKind {
	/// Harness/system instruction.
	System,
	/// User message.
	UserMessage,
	/// Assistant message or reasoning/output record.
	AssistantMessage,
	/// Model-requested tool invocation.
	ToolCall,
	/// Tool execution result.
	ToolResult,
	/// Approval request and decision.
	Approval,
	/// Ordered non-transcript harness lifecycle audit.
	Audit,
	/// Transcript compaction summary.
	Summary,
	/// Recoverable harness error.
	Error,
}

impl SessionEventKind {
	const fn as_str(&self) -> &'static str {
		match self {
			Self::System => "system",
			Self::UserMessage => "user_message",
			Self::AssistantMessage => "assistant_message",
			Self::ToolCall => "tool_call",
			Self::ToolResult => "tool_result",
			Self::Approval => "approval",
			Self::Audit => "audit",
			Self::Summary => "summary",
			Self::Error => "error",
		}
	}

	fn parse(value: &str) -> Result<Self, MemoryError> {
		match value {
			"system" => Ok(Self::System),
			"user_message" => Ok(Self::UserMessage),
			"assistant_message" => Ok(Self::AssistantMessage),
			"tool_call" => Ok(Self::ToolCall),
			"tool_result" => Ok(Self::ToolResult),
			"approval" => Ok(Self::Approval),
			"audit" => Ok(Self::Audit),
			"summary" => Ok(Self::Summary),
			"error" => Ok(Self::Error),
			_ => Err(MemoryError::Corrupt(format!(
				"unknown session event kind {value:?}"
			))),
		}
	}
}

/// One ordered durable transcript event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SessionEvent {
	/// `UUIDv7` event identity.
	pub id: Uuid,
	/// Owning session.
	pub session_id: Uuid,
	/// Monotonic, one-based per-session sequence.
	pub sequence: u64,
	/// Atomic turn/batch identity shared by consecutively committed events.
	pub turn_id: Uuid,
	/// Zero-based position within the atomic turn.
	pub turn_index: u32,
	/// Total events committed in the atomic turn.
	pub turn_size: u32,
	/// Event category.
	pub kind: SessionEventKind,
	/// Structured event body.
	pub payload: Value,
	/// Creation time.
	pub created_at: DateTime<Utc>,
}

/// Compaction queue state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum CompactionState {
	/// Waiting for a worker.
	Pending,
	/// Claimed by this or another process.
	Running,
	/// Summary durably appended.
	Completed,
	/// Bounded retries were exhausted or a permanent failure was recorded.
	Failed,
}

/// Durable transcript-compaction work item.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CompactionJob {
	/// `UUIDv7` job identity.
	pub id: Uuid,
	/// Session to compact.
	pub session_id: Uuid,
	/// Inclusive transcript sequence covered by the summary.
	pub through_sequence: u64,
	/// Immutable digest and boundary identities captured when work was queued.
	pub source: TranscriptProvenance,
	/// Queue state.
	pub state: CompactionState,
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
	/// Last state transition time.
	pub updated_at: DateTime<Utc>,
}

/// Cryptographic provenance for a contiguous transcript prefix.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TranscriptProvenance {
	/// Number of source events.
	pub event_count: u64,
	/// First source event identity.
	pub first_event_id: Uuid,
	/// Last source event identity at the inclusive boundary.
	pub last_event_id: Uuid,
	/// Lowercase SHA-256 of the length-delimited persisted event records.
	pub sha256: String,
}

/// Exclusive, expiring authority to complete one compaction job.
#[non_exhaustive]
pub struct CompactionLease {
	store: MemoryStore,
	job: CompactionJob,
	token: Uuid,
	lease_until: DateTime<Utc>,
	released: AtomicBool,
}

impl fmt::Debug for CompactionLease {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter
			.debug_struct("CompactionLease")
			.field("job", &self.job)
			.field("token", &"<redacted>")
			.field("lease_until", &self.lease_until)
			.finish_non_exhaustive()
	}
}

impl CompactionLease {
	/// Claimed durable work item.
	pub const fn job(&self) -> &CompactionJob {
		&self.job
	}

	/// Deadline after which another worker may recover the job.
	pub const fn lease_until(&self) -> DateTime<Utc> {
		self.lease_until
	}
}

impl Drop for CompactionLease {
	fn drop(&mut self) {
		if self.released.swap(true, Ordering::AcqRel) {
			return;
		}
		let _ = self.store.record_compaction_failure_best_effort(
			self,
			"worker claim dropped before completion",
			MemoryJobFailureDisposition::Retry,
		);
	}
}

/// Durable worker job category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum MemoryJobKind {
	/// Transcript compaction.
	Compaction,
	/// Session-to-Knowledge distillation.
	Distillation,
}

/// Retry policy attached to one durable worker failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MemoryJobFailureDisposition {
	/// Retry after bounded exponential backoff.
	Retry,
	/// Move directly to terminal failed state.
	Permanent,
}

/// Durable transition produced by recording a worker failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
#[non_exhaustive]
pub enum MemoryJobFailureOutcome {
	/// Another attempt may start at the persisted deadline.
	RetryScheduled {
		/// Total failed attempts after this transition.
		failures: u32,
		/// Earliest next claim time.
		retry_after: DateTime<Utc>,
	},
	/// The job requires explicit operator retry.
	Failed {
		/// Total failed attempts after this transition.
		failures: u32,
		/// Terminal transition time.
		failed_at: DateTime<Utc>,
	},
}

/// Inspectable terminal durable-memory worker failure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct MemoryJobFailure {
	/// Worker queue category.
	pub kind: MemoryJobKind,
	/// Durable job identity.
	pub id: Uuid,
	/// Source Session identity.
	pub session_id: Uuid,
	/// Total failed attempts.
	pub failures: u32,
	/// Bounded terminal diagnostic.
	pub error: String,
	/// Terminal transition time.
	pub failed_at: DateTime<Utc>,
}

/// Active workspace Knowledge plus its selected version.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Knowledge {
	/// Stable Knowledge identity.
	pub id: Uuid,
	/// Canonical workspace scope.
	pub workspace: PathBuf,
	/// Filesystem identity authorizing recall and mutation.
	pub workspace_identity: WorkspaceIdentity,
	/// Stable user/agent-selected key.
	pub key: String,
	/// Active version number.
	pub active_version: u32,
	/// Active version content.
	pub content: String,
	/// Confidence attached to the active version.
	pub confidence: f64,
	/// Session from which this version was distilled, when retained.
	pub source_session_id: Option<Uuid>,
	/// Whether recall should prioritize this entry.
	pub pinned: bool,
	/// Whether normal recall must hide this entry pending retention.
	pub tombstoned: bool,
	/// Entry creation time.
	pub created_at: DateTime<Utc>,
	/// Active-version/pin update time.
	pub updated_at: DateTime<Utc>,
}

/// Stable cursor matching workspace Knowledge priority order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KnowledgeCursor {
	pinned: bool,
	updated_at: DateTime<Utc>,
	key: String,
}

/// One bounded workspace Knowledge page.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct KnowledgePage {
	/// Active Knowledge entries in recall order.
	pub items: Vec<Knowledge>,
	/// Pass this cursor to fetch the next page.
	pub next: Option<KnowledgeCursor>,
}

/// One immutable Knowledge version.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct KnowledgeVersion {
	/// Owning Knowledge identity.
	pub knowledge_id: Uuid,
	/// Positive version number.
	pub version: u32,
	/// Immutable content.
	pub content: String,
	/// Confidence assigned when this version was created.
	pub confidence: f64,
	/// Source session, when still present.
	pub source_session_id: Option<Uuid>,
	/// Transcript range and digest supporting this version.
	pub provenance: Option<KnowledgeProvenance>,
	/// Distillation job that produced this version, when applicable.
	pub distillation_job_id: Option<Uuid>,
	/// Stable zero-based candidate position within that job.
	pub candidate_index: Option<u32>,
	/// Creation time.
	pub created_at: DateTime<Utc>,
}

/// Immutable transcript provenance supporting one Knowledge version.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct KnowledgeProvenance {
	/// Source session, retained independently from nullable foreign keys.
	pub session_id: Uuid,
	/// First included event sequence.
	pub first_sequence: u64,
	/// Last included event sequence.
	pub last_sequence: u64,
	/// SHA-256 over the exact source prefix.
	pub sha256: String,
}

/// Durable-store summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct MemoryStatus {
	/// Total `SQLite` bytes, including `WAL` and shared-memory sidecars.
	pub database_bytes: u64,
	/// Session count.
	pub sessions: u64,
	/// Event count.
	pub events: u64,
	/// Knowledge entry count.
	pub knowledge: u64,
	/// Pending or running compaction count.
	pub pending_compactions: u64,
	/// Terminal compactions awaiting explicit operator retry.
	pub failed_compactions: u64,
	/// Cataloged content-addressed asset count.
	pub assets: u64,
	/// Cataloged content-addressed asset bytes.
	pub asset_bytes: u64,
	/// Pending or running Knowledge distillation count.
	pub pending_distillations: u64,
	/// Terminal distillations awaiting explicit operator retry.
	pub failed_distillations: u64,
	/// Knowledge entries hidden by tombstones.
	pub tombstoned_knowledge: u64,
}

/// Durable memory failure.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum MemoryError {
	/// Emelex Home storage or snapshot-mutation lock failure.
	#[error(transparent)]
	Home(#[from] HomeError),
	/// `SQLite` failure.
	#[error(transparent)]
	Database(#[from] rusqlite::Error),
	/// JSON payload failure.
	#[error(transparent)]
	Json(#[from] serde_json::Error),
	/// Installed snapshot disappeared or changed before durable binding.
	#[error("installed model snapshot cannot be bound: {0}")]
	ModelSnapshot(String),
	/// Filesystem failure.
	#[error("{operation} failed for {path:?}: {source}")]
	Io {
		/// Operation being attempted.
		operation: &'static str,
		/// Affected path.
		path: PathBuf,
		/// Underlying error.
		#[source]
		source: std::io::Error,
	},
	/// Caller input violated a public invariant.
	#[error("invalid memory input: {0}")]
	Invalid(String),
	/// Requested durable object does not exist.
	#[error("{entity} {id} was not found")]
	NotFound {
		/// Entity category.
		entity: &'static str,
		/// Requested identity.
		id: Uuid,
	},
	/// A live process already owns session execution.
	#[error("session {session_id} is busy until {lease_until}")]
	SessionBusy {
		/// Contended session.
		session_id: Uuid,
		/// Current owner's lease deadline.
		lease_until: DateTime<Utc>,
	},
	/// Session execution authority expired or was reclaimed.
	#[error("session lease for {session_id} is stale or belongs to another process")]
	StaleSessionLease {
		/// Session whose authority was lost.
		session_id: Uuid,
	},
	/// Existing durable history must be replayed before the next append.
	#[error("session {session_id} must be replayed before appending")]
	ReplayRequired {
		/// Session requiring reconstruction.
		session_id: Uuid,
	},
	/// Compaction authority no longer matches its durable source.
	#[error("compaction lease for {job_id} is stale or has different provenance")]
	StaleCompactionLease {
		/// Compaction whose authority was lost.
		job_id: Uuid,
	},
	/// Distillation authority expired, was reclaimed, or its source changed.
	#[error("distillation lease for {job_id} is stale or has different provenance")]
	StaleDistillationLease {
		/// Distillation whose authority was lost.
		job_id: Uuid,
	},
	/// Claimed workspace path resolves to a different filesystem object.
	#[error("session {session_id} workspace identity changed from {expected:?} to {actual:?}")]
	WorkspaceMismatch {
		/// Session being resumed.
		session_id: Uuid,
		/// Identity bound at session creation.
		expected: WorkspaceIdentity,
		/// Identity observed at claim time.
		actual: WorkspaceIdentity,
	},
	/// Replay state changed after this authority last observed it.
	#[error(
		"session {session_id} replay is stale: expected sequence {expected_sequence}, found {actual_sequence}"
	)]
	StaleReplay {
		/// Session whose tail changed.
		session_id: Uuid,
		/// Sequence last observed by the caller.
		expected_sequence: u64,
		/// Current durable tail.
		actual_sequence: u64,
	},
	/// Stored data violated the schema/API contract.
	#[error("corrupt Emelex memory: {0}")]
	Corrupt(String),
	/// Database was written by a newer Emelex schema.
	#[error("memory schema {found} is newer than supported schema {supported}")]
	NewerSchema {
		/// On-disk version.
		found: i64,
		/// Newest supported version.
		supported: i64,
	},
}

struct RawSession {
	id: String,
	workspace: String,
	workspace_device: Option<String>,
	workspace_inode: Option<String>,
	model_reference: Option<String>,
	model_snapshot: Option<String>,
	title: Option<String>,
	created_at: String,
	updated_at: String,
}

fn raw_session(row: &rusqlite::Row<'_>) -> rusqlite::Result<RawSession> {
	Ok(RawSession {
		id: row.get(0)?,
		workspace: row.get(1)?,
		workspace_device: row.get(2)?,
		workspace_inode: row.get(3)?,
		model_reference: row.get(4)?,
		model_snapshot: row.get(5)?,
		title: row.get(6)?,
		created_at: row.get(7)?,
		updated_at: row.get(8)?,
	})
}

impl TryFrom<RawSession> for Session {
	type Error = MemoryError;

	fn try_from(raw: RawSession) -> Result<Self, Self::Error> {
		Ok(Self {
			id: parse_uuid(&raw.id, "session ID")?,
			workspace: PathBuf::from(raw.workspace),
			workspace_identity: parse_workspace_identity(
				raw.workspace_device,
				raw.workspace_inode,
			)?,
			model_reference: raw
				.model_reference
				.map(ModelRef::parse)
				.transpose()
				.map_err(|error| {
					MemoryError::Corrupt(format!("invalid Session model reference: {error}"))
				})?,
			model_snapshot: raw
				.model_snapshot
				.map(ModelSnapshotId::parse)
				.transpose()
				.map_err(|error| {
					MemoryError::Corrupt(format!("invalid Session model snapshot: {error}"))
				})?,
			title: raw.title,
			created_at: parse_time(&raw.created_at, "session creation time")?,
			updated_at: parse_time(&raw.updated_at, "session update time")?,
		})
	}
}

fn session_select(clause: &str) -> String {
	format!(
		"SELECT id, workspace, workspace_device, workspace_inode,
		        model_reference, model_snapshot, title, created_at, updated_at
		 FROM sessions {clause}"
	)
}

fn event_select(clause: &str) -> String {
	format!(
		"SELECT id, session_id, sequence, turn_id, turn_index, turn_size,
		        kind, payload_json, created_at
		 FROM session_events {clause}"
	)
}

struct RawEvent {
	id: String,
	session_id: String,
	sequence: i64,
	turn_id: Option<String>,
	turn_index: Option<i64>,
	turn_size: Option<i64>,
	kind: String,
	payload: String,
	created_at: String,
}

fn raw_event(row: &rusqlite::Row<'_>) -> rusqlite::Result<RawEvent> {
	Ok(RawEvent {
		id: row.get(0)?,
		session_id: row.get(1)?,
		sequence: row.get(2)?,
		turn_id: row.get(3)?,
		turn_index: row.get(4)?,
		turn_size: row.get(5)?,
		kind: row.get(6)?,
		payload: row.get(7)?,
		created_at: row.get(8)?,
	})
}

impl TryFrom<RawEvent> for SessionEvent {
	type Error = MemoryError;

	fn try_from(raw: RawEvent) -> Result<Self, Self::Error> {
		let turn_id = raw.turn_id.as_deref().ok_or_else(|| {
			MemoryError::Corrupt(format!("event {} has no turn identity", raw.id))
		})?;
		let turn_index = raw
			.turn_index
			.ok_or_else(|| MemoryError::Corrupt(format!("event {} has no turn index", raw.id)))?;
		let turn_size = raw
			.turn_size
			.ok_or_else(|| MemoryError::Corrupt(format!("event {} has no turn size", raw.id)))?;
		if turn_index < 0 || turn_size <= 0 || turn_index >= turn_size {
			return Err(MemoryError::Corrupt(format!(
				"event {} has invalid turn position {turn_index}/{turn_size}",
				raw.id
			)));
		}
		Ok(Self {
			id: parse_uuid(&raw.id, "event ID")?,
			session_id: parse_uuid(&raw.session_id, "event session ID")?,
			sequence: u64::try_from(raw.sequence)
				.map_err(|_| MemoryError::Corrupt("negative event sequence".to_string()))?,
			turn_id: parse_uuid(turn_id, "event turn ID")?,
			turn_index: u32::try_from(turn_index)
				.map_err(|_| MemoryError::Corrupt("invalid event turn index".to_string()))?,
			turn_size: u32::try_from(turn_size)
				.map_err(|_| MemoryError::Corrupt("invalid event turn size".to_string()))?,
			kind: SessionEventKind::parse(&raw.kind)?,
			payload: serde_json::from_str(&raw.payload)?,
			created_at: parse_time(&raw.created_at, "event creation time")?,
		})
	}
}

#[derive(Clone)]
struct RawCompaction {
	id: String,
	session_id: String,
	through_sequence: i64,
	state: String,
	claim_token: Option<String>,
	lease_until: Option<String>,
	failure_count: i64,
	retry_after: Option<String>,
	last_error: Option<String>,
	failed_at: Option<String>,
	source_event_count: Option<i64>,
	source_first_event_id: Option<String>,
	source_last_event_id: Option<String>,
	source_sha256: Option<String>,
	summary_event_id: Option<String>,
	created_at: String,
	updated_at: String,
}

fn raw_compaction(row: &rusqlite::Row<'_>) -> rusqlite::Result<RawCompaction> {
	Ok(RawCompaction {
		id: row.get(0)?,
		session_id: row.get(1)?,
		through_sequence: row.get(2)?,
		state: row.get(3)?,
		claim_token: row.get(4)?,
		lease_until: row.get(5)?,
		failure_count: row.get(6)?,
		retry_after: row.get(7)?,
		last_error: row.get(8)?,
		failed_at: row.get(9)?,
		source_event_count: row.get(10)?,
		source_first_event_id: row.get(11)?,
		source_last_event_id: row.get(12)?,
		source_sha256: row.get(13)?,
		summary_event_id: row.get(14)?,
		created_at: row.get(15)?,
		updated_at: row.get(16)?,
	})
}

impl RawCompaction {
	fn provenance(&self) -> Result<TranscriptProvenance, MemoryError> {
		let event_count = self.source_event_count.ok_or_else(|| {
			MemoryError::Corrupt(format!("compaction {} has no source event count", self.id))
		})?;
		let first_event_id = self.source_first_event_id.as_deref().ok_or_else(|| {
			MemoryError::Corrupt(format!("compaction {} has no first source event", self.id))
		})?;
		let last_event_id = self.source_last_event_id.as_deref().ok_or_else(|| {
			MemoryError::Corrupt(format!("compaction {} has no last source event", self.id))
		})?;
		let sha256 = self.source_sha256.as_deref().ok_or_else(|| {
			MemoryError::Corrupt(format!("compaction {} has no source digest", self.id))
		})?;
		if sha256.len() != 64
			|| !sha256
				.bytes()
				.all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
		{
			return Err(MemoryError::Corrupt(format!(
				"compaction {} has an invalid source digest",
				self.id
			)));
		}
		Ok(TranscriptProvenance {
			event_count: u64::try_from(event_count).map_err(|_| {
				MemoryError::Corrupt(format!(
					"compaction {} has a negative source event count",
					self.id
				))
			})?,
			first_event_id: parse_uuid(first_event_id, "compaction first source event ID")?,
			last_event_id: parse_uuid(last_event_id, "compaction last source event ID")?,
			sha256: sha256.to_string(),
		})
	}
}

impl TryFrom<RawCompaction> for CompactionJob {
	type Error = MemoryError;

	fn try_from(raw: RawCompaction) -> Result<Self, Self::Error> {
		let has_claim = raw.claim_token.is_some() && raw.lease_until.is_some();
		let has_summary = raw.summary_event_id.is_some();
		let has_failure = raw.last_error.is_some() && raw.failed_at.is_some();
		let failures = u32::try_from(raw.failure_count).map_err(|_| {
			MemoryError::Corrupt(format!(
				"compaction {} has an invalid failure count",
				raw.id
			))
		})?;
		let valid_history = (failures == 0 && raw.last_error.is_none())
			|| (failures > 0 && raw.last_error.is_some());
		let state = match raw.state.as_str() {
			"pending"
				if !has_claim
					&& !has_summary && raw.failed_at.is_none()
					&& valid_history
					&& (raw.retry_after.is_none() || failures > 0) =>
			{
				CompactionState::Pending
			}
			"running"
				if has_claim
					&& !has_summary && raw.retry_after.is_none()
					&& raw.failed_at.is_none()
					&& valid_history =>
			{
				CompactionState::Running
			}
			"completed"
				if !has_claim
					&& has_summary && raw.retry_after.is_none()
					&& raw.last_error.is_none()
					&& raw.failed_at.is_none() =>
			{
				CompactionState::Completed
			}
			"failed"
				if !has_claim
					&& !has_summary && raw.retry_after.is_none()
					&& has_failure && failures > 0 =>
			{
				CompactionState::Failed
			}
			"pending" | "running" | "completed" | "failed" => {
				return Err(MemoryError::Corrupt(format!(
					"compaction state {:?} has inconsistent claim metadata",
					raw.state
				)));
			}
			other => {
				return Err(MemoryError::Corrupt(format!(
					"unknown compaction state {other:?}"
				)));
			}
		};
		let source = raw.provenance()?;
		let through_sequence = u64::try_from(raw.through_sequence)
			.map_err(|_| MemoryError::Corrupt("negative compaction sequence".to_string()))?;
		if source.event_count != through_sequence {
			return Err(MemoryError::Corrupt(format!(
				"compaction {} source count does not match its boundary",
				raw.id
			)));
		}
		Ok(Self {
			id: parse_uuid(&raw.id, "compaction ID")?,
			session_id: parse_uuid(&raw.session_id, "compaction session ID")?,
			through_sequence,
			source,
			state,
			failures,
			retry_after: raw
				.retry_after
				.as_deref()
				.map(|value| parse_time(value, "compaction retry deadline"))
				.transpose()?,
			last_error: raw.last_error,
			failed_at: raw
				.failed_at
				.as_deref()
				.map(|value| parse_time(value, "compaction failure time"))
				.transpose()?,
			created_at: parse_time(&raw.created_at, "compaction creation time")?,
			updated_at: parse_time(&raw.updated_at, "compaction update time")?,
		})
	}
}

struct RawKnowledge {
	id: String,
	workspace: String,
	workspace_device: String,
	workspace_inode: String,
	key: String,
	active_version: i64,
	content: String,
	confidence: f64,
	source_session_id: Option<String>,
	pinned: bool,
	tombstoned: bool,
	created_at: String,
	updated_at: String,
}

fn raw_knowledge(row: &rusqlite::Row<'_>) -> rusqlite::Result<RawKnowledge> {
	Ok(RawKnowledge {
		id: row.get(0)?,
		workspace: row.get(1)?,
		workspace_device: row.get(2)?,
		workspace_inode: row.get(3)?,
		key: row.get(4)?,
		active_version: row.get(5)?,
		content: row.get(6)?,
		confidence: row.get(7)?,
		source_session_id: row.get(8)?,
		pinned: row.get(9)?,
		tombstoned: row.get(10)?,
		created_at: row.get(11)?,
		updated_at: row.get(12)?,
	})
}

impl TryFrom<RawKnowledge> for Knowledge {
	type Error = MemoryError;

	fn try_from(raw: RawKnowledge) -> Result<Self, Self::Error> {
		Ok(Self {
			id: parse_uuid(&raw.id, "Knowledge ID")?,
			workspace: PathBuf::from(raw.workspace),
			workspace_identity: parse_workspace_identity(
				Some(raw.workspace_device),
				Some(raw.workspace_inode),
			)?,
			key: raw.key,
			active_version: u32::try_from(raw.active_version)
				.map_err(|_| MemoryError::Corrupt("invalid Knowledge version".to_string()))?,
			content: raw.content,
			confidence: parse_confidence(raw.confidence, "Knowledge confidence")?,
			source_session_id: raw
				.source_session_id
				.as_deref()
				.map(|id| parse_uuid(id, "Knowledge source session ID"))
				.transpose()?,
			pinned: raw.pinned,
			tombstoned: raw.tombstoned,
			created_at: parse_time(&raw.created_at, "Knowledge creation time")?,
			updated_at: parse_time(&raw.updated_at, "Knowledge update time")?,
		})
	}
}

struct RawKnowledgeVersion {
	knowledge_id: String,
	version: i64,
	content: String,
	confidence: f64,
	source_session_id: Option<String>,
	provenance_session_id: Option<String>,
	source_first_sequence: Option<i64>,
	source_last_sequence: Option<i64>,
	source_sha256: Option<String>,
	distillation_job_id: Option<String>,
	candidate_index: Option<i64>,
	created_at: String,
}

fn raw_knowledge_version(row: &rusqlite::Row<'_>) -> rusqlite::Result<RawKnowledgeVersion> {
	Ok(RawKnowledgeVersion {
		knowledge_id: row.get(0)?,
		version: row.get(1)?,
		content: row.get(2)?,
		confidence: row.get(3)?,
		source_session_id: row.get(4)?,
		provenance_session_id: row.get(5)?,
		source_first_sequence: row.get(6)?,
		source_last_sequence: row.get(7)?,
		source_sha256: row.get(8)?,
		distillation_job_id: row.get(9)?,
		candidate_index: row.get(10)?,
		created_at: row.get(11)?,
	})
}

impl TryFrom<RawKnowledgeVersion> for KnowledgeVersion {
	type Error = MemoryError;

	fn try_from(raw: RawKnowledgeVersion) -> Result<Self, Self::Error> {
		let source_session_id = raw
			.source_session_id
			.as_deref()
			.map(|id| parse_uuid(id, "Knowledge source session ID"))
			.transpose()?;
		let provenance = match (
			raw.provenance_session_id,
			raw.source_first_sequence,
			raw.source_last_sequence,
			raw.source_sha256,
		) {
			(None, None, None, None) => None,
			(Some(session_id), Some(first), Some(last), Some(sha256)) => {
				Some(KnowledgeProvenance {
					session_id: parse_uuid(&session_id, "Knowledge provenance session ID")?,
					first_sequence: u64::try_from(first).map_err(|_| {
						MemoryError::Corrupt(
							"Knowledge provenance has a negative first sequence".to_string(),
						)
					})?,
					last_sequence: u64::try_from(last).map_err(|_| {
						MemoryError::Corrupt(
							"Knowledge provenance has a negative last sequence".to_string(),
						)
					})?,
					sha256,
				})
			}
			_ => {
				return Err(MemoryError::Corrupt(
					"Knowledge version has incomplete provenance".to_string(),
				));
			}
		};
		Ok(Self {
			knowledge_id: parse_uuid(&raw.knowledge_id, "Knowledge ID")?,
			version: u32::try_from(raw.version)
				.map_err(|_| MemoryError::Corrupt("invalid Knowledge version".to_string()))?,
			content: raw.content,
			confidence: parse_confidence(raw.confidence, "Knowledge version confidence")?,
			source_session_id,
			provenance,
			distillation_job_id: raw
				.distillation_job_id
				.as_deref()
				.map(|id| parse_uuid(id, "distillation job ID"))
				.transpose()?,
			candidate_index: raw
				.candidate_index
				.map(u32::try_from)
				.transpose()
				.map_err(|_| {
					MemoryError::Corrupt(
						"Knowledge distillation candidate index is invalid".to_string(),
					)
				})?,
			created_at: parse_time(&raw.created_at, "Knowledge version creation time")?,
		})
	}
}

fn knowledge_select(clause: &str) -> String {
	format!(
		"SELECT k.id, k.workspace, k.workspace_device, k.workspace_inode,
		        k.key, k.active_version, v.content, v.confidence,
		        v.source_session_id, k.pinned, k.tombstoned,
		        k.created_at, k.updated_at
		 FROM knowledge k
		 JOIN knowledge_versions v
		   ON v.knowledge_id = k.id AND v.version = k.active_version
		 {clause}"
	)
}

fn compaction_select(clause: &str) -> String {
	format!(
		"SELECT id, session_id, through_sequence, state, claim_token, lease_until,
		        failure_count, retry_after, last_error, failed_at,
		        source_event_count, source_first_event_id, source_last_event_id,
		        source_sha256, summary_event_id, created_at, updated_at
		 FROM compaction_jobs {clause}"
	)
}

pub(super) struct JobFailureTransition {
	pub(super) failures: i64,
	pub(super) state: &'static str,
	pub(super) retry_after: Option<String>,
	pub(super) failed_at: Option<String>,
	pub(super) outcome: MemoryJobFailureOutcome,
}

pub(super) fn job_failure_transition(
	current_failures: i64,
	disposition: MemoryJobFailureDisposition,
	now: DateTime<Utc>,
) -> Result<JobFailureTransition, MemoryError> {
	let current = u32::try_from(current_failures)
		.map_err(|_| MemoryError::Corrupt("worker job has a negative failure count".to_string()))?;
	let failures = current
		.checked_add(1)
		.ok_or_else(|| MemoryError::Corrupt("worker job failure count overflow".to_string()))?;
	let terminal = matches!(disposition, MemoryJobFailureDisposition::Permanent)
		|| failures >= MAX_JOB_FAILURES;
	if terminal {
		return Ok(JobFailureTransition {
			failures: i64::from(failures),
			state: "failed",
			retry_after: None,
			failed_at: Some(now.to_rfc3339()),
			outcome: MemoryJobFailureOutcome::Failed {
				failures,
				failed_at: now,
			},
		});
	}
	let exponent = failures.saturating_sub(1).min(31);
	let multiplier = 1_u32.checked_shl(exponent).unwrap_or(u32::MAX);
	let delay = JOB_RETRY_BASE
		.checked_mul(multiplier)
		.unwrap_or(JOB_RETRY_MAX)
		.min(JOB_RETRY_MAX);
	let retry_after = deadline_after(now, delay, "worker retry backoff")?;
	Ok(JobFailureTransition {
		failures: i64::from(failures),
		state: "pending",
		retry_after: Some(retry_after.to_rfc3339()),
		failed_at: None,
		outcome: MemoryJobFailureOutcome::RetryScheduled {
			failures,
			retry_after,
		},
	})
}

fn transition_compaction_failure(
	transaction: &rusqlite::Transaction<'_>,
	job_id: Uuid,
	token: Uuid,
	current_failures: i64,
	error: &str,
	disposition: MemoryJobFailureDisposition,
	now: DateTime<Utc>,
) -> Result<MemoryJobFailureOutcome, MemoryError> {
	let transition = job_failure_transition(current_failures, disposition, now)?;
	let changed = transaction.execute(
		"UPDATE compaction_jobs
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
		return Err(MemoryError::StaleCompactionLease { job_id });
	}
	Ok(transition.outcome)
}

fn recover_one_expired_compaction(
	transaction: &rusqlite::Transaction<'_>,
	now: DateTime<Utc>,
) -> Result<bool, MemoryError> {
	let row = transaction
		.query_row(
			"SELECT id, claim_token, failure_count
			 FROM compaction_jobs
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
	let job_id = parse_uuid(&job_id, "expired compaction ID")?;
	let token = parse_uuid(&token, "expired compaction claim token")?;
	transition_compaction_failure(
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

fn prepare_compaction_summary(
	lease: &CompactionLease,
	summary: &Value,
) -> Result<(Value, String), MemoryError> {
	let payload = serde_json::json!({
		"compaction": {
			"job_id": lease.job.id,
			"through_sequence": lease.job.through_sequence,
			"source": &lease.job.source,
		},
		"summary": summary,
	});
	let payload_json = bounded_json_string(&payload, MAX_EVENT_BYTES, "compaction summary")?;
	Ok((payload, payload_json))
}

fn validate_compaction_lease(
	transaction: &rusqlite::Transaction<'_>,
	lease: &CompactionLease,
	now: DateTime<Utc>,
) -> Result<RawCompaction, MemoryError> {
	let raw = transaction
		.query_row(
			&compaction_select("WHERE id = ?1"),
			[lease.job.id.to_string()],
			raw_compaction,
		)
		.optional()?
		.ok_or(MemoryError::NotFound {
			entity: "compaction job",
			id: lease.job.id,
		})?;
	let token = lease.token.to_string();
	let deadline = raw
		.lease_until
		.as_deref()
		.map(|value| parse_time(value, "compaction lease deadline"))
		.transpose()?;
	if raw.state != "running"
		|| raw.claim_token.as_deref() != Some(token.as_str())
		|| deadline.is_none_or(|deadline| deadline <= now)
	{
		return Err(MemoryError::StaleCompactionLease {
			job_id: lease.job.id,
		});
	}
	let stored_source = raw.provenance()?;
	if stored_source != lease.job.source {
		return Err(MemoryError::StaleCompactionLease {
			job_id: lease.job.id,
		});
	}
	let current_source = transcript_provenance(transaction, &raw.session_id, raw.through_sequence)?;
	if current_source != stored_source {
		return Err(MemoryError::Corrupt(format!(
			"compaction source changed for job {}",
			lease.job.id
		)));
	}
	Ok(raw)
}

fn migrate_v3_rows(transaction: &rusqlite::Transaction<'_>) -> Result<(), MemoryError> {
	migrate_session_identities(transaction)?;
	migrate_compaction_provenance(transaction)?;
	validate_v3_migration(transaction)
}

fn migrate_v3_to_v4(transaction: &rusqlite::Transaction<'_>) -> Result<(), MemoryError> {
	transaction.execute_batch(PREPARE_V3_TO_V4)?;
	let mut cursor = String::new();
	loop {
		let row = transaction
			.query_row(
				"SELECT id, workspace FROM knowledge
				 WHERE id > ?1 ORDER BY id ASC LIMIT 1",
				[&cursor],
				|row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
			)
			.optional()?;
		let Some((id, workspace)) = row else {
			break;
		};
		let migrated = migration_workspace_binding(Path::new(&workspace));
		transaction.execute(
			"UPDATE knowledge
			 SET workspace_device = ?2, workspace_inode = ?3,
			     legacy_identity = ?4
			 WHERE id = ?1",
			params![
				&id,
				migrated.identity.device.to_string(),
				migrated.identity.inode.to_string(),
				migrated.legacy,
			],
		)?;
		cursor = id;
	}
	disambiguate_migrated_knowledge(transaction)?;
	transaction.execute_batch(FINISH_V3_TO_V4)?;
	let incomplete: i64 = transaction.query_row(
		"SELECT COUNT(*) FROM knowledge
		 WHERE workspace_device IS NULL OR workspace_inode IS NULL",
		[],
		|row| row.get(0),
	)?;
	if incomplete != 0 {
		return Err(MemoryError::Corrupt(
			"Knowledge identity migration left incomplete rows".to_string(),
		));
	}
	Ok(())
}

fn disambiguate_migrated_knowledge(
	transaction: &rusqlite::Transaction<'_>,
) -> Result<(), MemoryError> {
	let duplicates = {
		let mut statement = transaction.prepare(
			"WITH ranked AS (
			   SELECT id, key,
			          ROW_NUMBER() OVER (
			            PARTITION BY workspace_device, workspace_inode, key
			            ORDER BY updated_at DESC, id DESC
			          ) AS identity_rank
			   FROM knowledge
			 )
			 SELECT id, key FROM ranked WHERE identity_rank > 1 ORDER BY id ASC",
		)?;
		statement
			.query_map([], |row| {
				Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
			})?
			.collect::<Result<Vec<_>, _>>()?
	};
	for (id, key) in duplicates {
		let digest = Sha256::digest(id.as_bytes());
		let mut inode_bytes = [0_u8; 8];
		inode_bytes.copy_from_slice(&digest[..8]);
		let mut inode = u64::from_be_bytes(inode_bytes).max(1);
		loop {
			let collision = transaction
				.query_row(
					"SELECT 1 FROM knowledge
					 WHERE id != ?1 AND workspace_device = '0'
					   AND workspace_inode = ?2 AND key = ?3",
					params![&id, inode.to_string(), &key],
					|_| Ok(()),
				)
				.optional()?
				.is_some();
			if !collision {
				break;
			}
			inode = inode.checked_add(1).unwrap_or(1);
		}
		transaction.execute(
			"UPDATE knowledge
			 SET workspace_device = '0', workspace_inode = ?2, legacy_identity = 1
			 WHERE id = ?1",
			params![&id, inode.to_string()],
		)?;
	}
	Ok(())
}

fn migrate_session_identities(transaction: &rusqlite::Transaction<'_>) -> Result<(), MemoryError> {
	let mut session_cursor = String::new();
	loop {
		let row = transaction
			.query_row(
				"SELECT id, workspace FROM sessions
				 WHERE id > ?1 ORDER BY id ASC LIMIT 1",
				[&session_cursor],
				|row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
			)
			.optional()?;
		let Some((id, workspace)) = row else {
			break;
		};
		let binding = migration_workspace_binding(Path::new(&workspace));
		transaction.execute(
			"UPDATE sessions
			 SET workspace = ?2, workspace_device = ?3, workspace_inode = ?4
			 WHERE id = ?1",
			params![
				&id,
				path_text(&binding.path)?,
				binding.identity.device.to_string(),
				binding.identity.inode.to_string(),
			],
		)?;
		session_cursor = id;
	}
	Ok(())
}

fn migrate_compaction_provenance(
	transaction: &rusqlite::Transaction<'_>,
) -> Result<(), MemoryError> {
	let mut through_cursor = 0_i64;
	let mut job_cursor = String::new();
	loop {
		let sql = format!(
			"{} ORDER BY through_sequence ASC, id ASC LIMIT 1",
			migration_compaction_select(
				"WHERE through_sequence > ?1
				   OR (through_sequence = ?1 AND id > ?2)",
			)
		);
		let raw = transaction
			.query_row(&sql, params![through_cursor, &job_cursor], raw_compaction)
			.optional()?;
		let Some(raw) = raw else {
			break;
		};
		let source = transcript_provenance(transaction, &raw.session_id, raw.through_sequence)?;
		let summary_event_id = if raw.state == "completed" {
			Some(migrate_compaction_summary(transaction, &raw, &source)?)
		} else {
			None
		};
		transaction.execute(
			"UPDATE compaction_jobs
			 SET source_event_count = ?2, source_first_event_id = ?3,
			     source_last_event_id = ?4, source_sha256 = ?5,
			     summary_event_id = ?6
			 WHERE id = ?1",
			params![
				&raw.id,
				i64::try_from(source.event_count).map_err(|_| {
					MemoryError::Corrupt("compaction source event count overflow".to_string())
				})?,
				source.first_event_id.to_string(),
				source.last_event_id.to_string(),
				&source.sha256,
				summary_event_id,
			],
		)?;
		through_cursor = raw.through_sequence;
		job_cursor = raw.id;
	}
	Ok(())
}

fn migration_compaction_select(clause: &str) -> String {
	format!(
		"SELECT id, session_id, through_sequence, state, claim_token, lease_until,
		        0, NULL, NULL, NULL,
		        source_event_count, source_first_event_id, source_last_event_id,
		        source_sha256, summary_event_id, created_at, updated_at
		 FROM compaction_jobs {clause}"
	)
}

fn validate_v3_migration(transaction: &rusqlite::Transaction<'_>) -> Result<(), MemoryError> {
	let incomplete: i64 = transaction.query_row(
		"SELECT COUNT(*) FROM sessions
		 WHERE workspace_device IS NULL OR workspace_inode IS NULL",
		[],
		|row| row.get(0),
	)?;
	if incomplete != 0 {
		return Err(MemoryError::Corrupt(
			"workspace identity migration left incomplete sessions".to_string(),
		));
	}
	let incomplete: i64 = transaction.query_row(
		"SELECT COUNT(*) FROM session_events
		 WHERE turn_id IS NULL OR turn_index IS NULL OR turn_size IS NULL
		    OR turn_index < 0 OR turn_size <= 0 OR turn_index >= turn_size",
		[],
		|row| row.get(0),
	)?;
	if incomplete != 0 {
		return Err(MemoryError::Corrupt(
			"turn migration left invalid session events".to_string(),
		));
	}
	let incomplete: i64 = transaction.query_row(
		"SELECT COUNT(*) FROM compaction_jobs
		 WHERE source_event_count IS NULL OR source_first_event_id IS NULL
		    OR source_last_event_id IS NULL OR source_sha256 IS NULL
		    OR (state = 'completed' AND summary_event_id IS NULL)
		    OR (state != 'completed' AND summary_event_id IS NOT NULL)",
		[],
		|row| row.get(0),
	)?;
	if incomplete != 0 {
		return Err(MemoryError::Corrupt(
			"provenance migration left incomplete compaction jobs".to_string(),
		));
	}
	Ok(())
}

fn migrate_compaction_summary(
	transaction: &rusqlite::Transaction<'_>,
	job: &RawCompaction,
	source: &TranscriptProvenance,
) -> Result<String, MemoryError> {
	let matches: i64 = transaction.query_row(
		"SELECT COUNT(*) FROM session_events
		 WHERE session_id = ?1 AND kind = 'summary'
		   AND json_extract(payload_json, '$.compaction.job_id') = ?2",
		params![&job.session_id, &job.id],
		|row| row.get(0),
	)?;
	if matches != 1 {
		return Err(MemoryError::Corrupt(format!(
			"completed compaction {} has {matches} summary events",
			job.id
		)));
	}
	let (event_id, sequence, payload_json) = transaction.query_row(
		"SELECT id, sequence, payload_json FROM session_events
		 WHERE session_id = ?1 AND kind = 'summary'
		   AND json_extract(payload_json, '$.compaction.job_id') = ?2",
		params![&job.session_id, &job.id],
		|row| {
			Ok((
				row.get::<_, String>(0)?,
				row.get::<_, i64>(1)?,
				row.get::<_, String>(2)?,
			))
		},
	)?;
	if sequence <= job.through_sequence {
		return Err(MemoryError::Corrupt(format!(
			"compaction {} summary does not follow its source boundary",
			job.id
		)));
	}
	let mut payload: Value = serde_json::from_str(&payload_json)?;
	let root = payload.as_object_mut().ok_or_else(|| {
		MemoryError::Corrupt(format!("compaction {} summary is not an object", job.id))
	})?;
	let metadata = root
		.get_mut("compaction")
		.and_then(Value::as_object_mut)
		.ok_or_else(|| {
			MemoryError::Corrupt(format!(
				"compaction {} summary has no metadata object",
				job.id
			))
		})?;
	let payload_job = metadata
		.get("job_id")
		.and_then(Value::as_str)
		.ok_or_else(|| {
			MemoryError::Corrupt(format!("compaction {} summary has no job ID", job.id))
		})?;
	if payload_job != job.id {
		return Err(MemoryError::Corrupt(format!(
			"compaction {} summary names another job",
			job.id
		)));
	}
	let payload_through = metadata
		.get("through_sequence")
		.and_then(Value::as_u64)
		.ok_or_else(|| {
			MemoryError::Corrupt(format!(
				"compaction {} summary has no source boundary",
				job.id
			))
		})?;
	let through = u64::try_from(job.through_sequence)
		.map_err(|_| MemoryError::Corrupt("negative compaction boundary".to_string()))?;
	if payload_through != through {
		return Err(MemoryError::Corrupt(format!(
			"compaction {} summary boundary does not match its job",
			job.id
		)));
	}
	metadata.insert("source".to_string(), serde_json::to_value(source)?);
	let migrated = bounded_json_string(&payload, MAX_EVENT_BYTES, "migrated compaction summary")
		.map_err(|error| {
			MemoryError::Corrupt(format!(
				"compaction {} migrated summary is invalid: {error}",
				job.id
			))
		})?;
	transaction.execute(
		"UPDATE session_events SET payload_json = ?2 WHERE id = ?1",
		params![&event_id, migrated],
	)?;
	Ok(event_id)
}

fn deadline_after(
	now: DateTime<Utc>,
	duration: Duration,
	name: &str,
) -> Result<DateTime<Utc>, MemoryError> {
	let duration = chrono::Duration::from_std(duration)
		.map_err(|_| MemoryError::Invalid(format!("{name} duration is invalid")))?;
	now.checked_add_signed(duration)
		.ok_or_else(|| MemoryError::Invalid(format!("{name} deadline overflow")))
}

fn parse_session_claim(
	token: Option<String>,
	lease_until: Option<String>,
) -> Result<Option<(Uuid, DateTime<Utc>)>, MemoryError> {
	match (token, lease_until) {
		(None, None) => Ok(None),
		(Some(token), Some(lease_until)) => Ok(Some((
			parse_uuid(&token, "session execution token")?,
			parse_time(&lease_until, "session execution lease deadline")?,
		))),
		_ => Err(MemoryError::Corrupt(
			"session execution claim has incomplete metadata".to_string(),
		)),
	}
}

fn validate_session_lease(
	connection: &Connection,
	lease: &SessionLease,
	now: DateTime<Utc>,
) -> Result<(), MemoryError> {
	let claim = connection
		.query_row(
			"SELECT execution_token, execution_lease_until
			 FROM sessions WHERE id = ?1",
			[lease.session.id.to_string()],
			|row| {
				Ok((
					row.get::<_, Option<String>>(0)?,
					row.get::<_, Option<String>>(1)?,
				))
			},
		)
		.optional()?
		.ok_or(MemoryError::NotFound {
			entity: "session",
			id: lease.session.id,
		})?;
	let Some((token, deadline)) = parse_session_claim(claim.0, claim.1)? else {
		return Err(MemoryError::StaleSessionLease {
			session_id: lease.session.id,
		});
	};
	if token != lease.token || deadline <= now {
		return Err(MemoryError::StaleSessionLease {
			session_id: lease.session.id,
		});
	}
	Ok(())
}

fn last_event_sequence(connection: &Connection, session_id: Uuid) -> Result<u64, MemoryError> {
	let sequence: i64 = connection.query_row(
		"SELECT COALESCE(MAX(sequence), 0)
		 FROM session_events WHERE session_id = ?1",
		[session_id.to_string()],
		|row| row.get(0),
	)?;
	u64::try_from(sequence)
		.map_err(|_| MemoryError::Corrupt("negative session event sequence".to_string()))
}

fn prepare_turn(events: &[SessionEventInput]) -> Result<Vec<String>, MemoryError> {
	if events.is_empty() {
		return Err(MemoryError::Invalid(
			"turn must contain at least one event".to_string(),
		));
	}
	if events.len() > MAX_TURN_EVENTS {
		return Err(MemoryError::Invalid(format!(
			"turn exceeds {MAX_TURN_EVENTS} event limit"
		)));
	}
	let mut payloads = Vec::with_capacity(events.len());
	let mut total_bytes = 0_usize;
	for event in events {
		if event.kind == SessionEventKind::Summary {
			return Err(MemoryError::Invalid(
				"summary events may only be written by compaction".to_string(),
			));
		}
		let payload = bounded_json_string(&event.payload, MAX_EVENT_BYTES, "session event")?;
		total_bytes = total_bytes
			.checked_add(payload.len())
			.ok_or_else(|| MemoryError::Invalid("turn payload byte count overflow".to_string()))?;
		if total_bytes > MAX_TURN_BYTES {
			return Err(MemoryError::Invalid(format!(
				"turn exceeds {MAX_TURN_BYTES} byte payload limit"
			)));
		}
		payloads.push(payload);
	}
	Ok(payloads)
}

struct BoundedJsonBuffer {
	bytes: Vec<u8>,
	limit: usize,
	exceeded: bool,
}

impl io::Write for BoundedJsonBuffer {
	fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
		let Some(next) = self.bytes.len().checked_add(bytes.len()) else {
			self.exceeded = true;
			return Err(io::Error::other("serialized JSON size overflow"));
		};
		if next > self.limit {
			self.exceeded = true;
			return Err(io::Error::other("serialized JSON exceeds limit"));
		}
		self.bytes.extend_from_slice(bytes);
		Ok(bytes.len())
	}

	fn flush(&mut self) -> io::Result<()> {
		Ok(())
	}
}

pub(super) fn bounded_json_string(
	value: &Value,
	limit: usize,
	name: &str,
) -> Result<String, MemoryError> {
	if !crate::json::structurally_bounded(value) {
		return Err(MemoryError::Invalid(format!(
			"{name} exceeds JSON structural limits"
		)));
	}
	let mut writer = BoundedJsonBuffer {
		bytes: Vec::with_capacity(limit.min(64 * 1024)),
		limit,
		exceeded: false,
	};
	if let Err(error) = serde_json::to_writer(&mut writer, value) {
		if writer.exceeded {
			return Err(MemoryError::Invalid(format!(
				"{name} exceeds {limit} byte limit"
			)));
		}
		return Err(MemoryError::Json(error));
	}
	String::from_utf8(writer.bytes)
		.map_err(|error| MemoryError::Corrupt(format!("{name} serialized invalid UTF-8: {error}")))
}

pub(super) fn bounded_serializable_value<T: Serialize>(
	value: &T,
	limit: usize,
	name: &str,
) -> Result<Value, MemoryError> {
	let mut writer = BoundedJsonBuffer {
		bytes: Vec::with_capacity(limit.min(64 * 1024)),
		limit,
		exceeded: false,
	};
	if let Err(error) = serde_json::to_writer(&mut writer, value) {
		if writer.exceeded {
			return Err(MemoryError::Invalid(format!(
				"{name} exceeds {limit} byte limit"
			)));
		}
		return Err(MemoryError::Json(error));
	}
	let parsed = serde_json::from_slice(&writer.bytes).map_err(MemoryError::Json)?;
	if !crate::json::structurally_bounded(&parsed) {
		return Err(MemoryError::Invalid(format!(
			"{name} exceeds JSON structural limits"
		)));
	}
	Ok(parsed)
}

struct RawProvenanceEvent {
	id: String,
	sequence: i64,
	turn_id: Option<String>,
	turn_index: Option<i64>,
	turn_size: Option<i64>,
	kind: String,
	payload: String,
	created_at: String,
}

fn raw_provenance_event(row: &rusqlite::Row<'_>) -> rusqlite::Result<RawProvenanceEvent> {
	Ok(RawProvenanceEvent {
		id: row.get(0)?,
		sequence: row.get(1)?,
		turn_id: row.get(2)?,
		turn_index: row.get(3)?,
		turn_size: row.get(4)?,
		kind: row.get(5)?,
		payload: row.get(6)?,
		created_at: row.get(7)?,
	})
}

#[derive(Default)]
struct ProvenanceTurn {
	id: Option<String>,
	size: i64,
	next_index: i64,
}

impl ProvenanceTurn {
	fn validate<'a>(
		&mut self,
		session_id: &str,
		event: &'a RawProvenanceEvent,
	) -> Result<(&'a str, i64, i64), MemoryError> {
		let turn_id = event.turn_id.as_deref().ok_or_else(|| {
			MemoryError::Corrupt(format!("event {} has no turn identity", event.id))
		})?;
		let turn_index = event
			.turn_index
			.ok_or_else(|| MemoryError::Corrupt(format!("event {} has no turn index", event.id)))?;
		let turn_size = event
			.turn_size
			.ok_or_else(|| MemoryError::Corrupt(format!("event {} has no turn size", event.id)))?;
		parse_uuid(turn_id, "compaction source turn ID")?;
		if turn_index == 0 {
			if self.id.is_some() {
				return Err(MemoryError::Corrupt(format!(
					"session {session_id} starts a turn before the prior turn ends"
				)));
			}
			self.id = Some(turn_id.to_string());
			self.size = turn_size;
			self.next_index = 0;
		}
		if turn_size <= 0
			|| turn_index != self.next_index
			|| self.id.as_deref() != Some(turn_id)
			|| turn_size != self.size
		{
			return Err(MemoryError::Corrupt(format!(
				"session {session_id} has invalid atomic turn metadata at sequence {}",
				event.sequence
			)));
		}
		self.next_index = self
			.next_index
			.checked_add(1)
			.ok_or_else(|| MemoryError::Corrupt("turn index overflow".to_string()))?;
		if self.next_index == self.size {
			self.id = None;
			self.size = 0;
			self.next_index = 0;
		}
		Ok((turn_id, turn_index, turn_size))
	}
}

fn transcript_provenance(
	connection: &Connection,
	session_id: &str,
	through_sequence: i64,
) -> Result<TranscriptProvenance, MemoryError> {
	if through_sequence <= 0 {
		return Err(MemoryError::Invalid(
			"compaction source boundary must be positive".to_string(),
		));
	}
	let mut statement = connection.prepare(
		"SELECT id, sequence, turn_id, turn_index, turn_size,
		        kind, payload_json, created_at
		 FROM session_events
		 WHERE session_id = ?1 AND sequence <= ?2
		 ORDER BY sequence ASC",
	)?;
	let rows = statement.query_map(params![session_id, through_sequence], raw_provenance_event)?;
	let mut hasher = Sha256::new();
	hash_provenance_field(&mut hasher, b"emelex/session-transcript/v1")?;
	hash_provenance_field(&mut hasher, session_id.as_bytes())?;
	hasher.update(through_sequence.to_be_bytes());
	let mut expected_sequence = 1_i64;
	let mut first_event_id = None;
	let mut last_event_id = None;
	let mut turn = ProvenanceTurn::default();
	for row in rows {
		let event = row?;
		if event.sequence != expected_sequence {
			return Err(MemoryError::Corrupt(format!(
				"session {session_id} transcript is not contiguous at sequence {expected_sequence}"
			)));
		}
		let (turn_id, turn_index, turn_size) = turn.validate(session_id, &event)?;
		let event_id = parse_uuid(&event.id, "compaction source event ID")?;
		first_event_id.get_or_insert(event_id);
		last_event_id = Some(event_id);
		hasher.update(event.sequence.to_be_bytes());
		hash_provenance_field(&mut hasher, event.id.as_bytes())?;
		hash_provenance_field(&mut hasher, turn_id.as_bytes())?;
		hasher.update(turn_index.to_be_bytes());
		hasher.update(turn_size.to_be_bytes());
		hash_provenance_field(&mut hasher, event.kind.as_bytes())?;
		hash_provenance_field(&mut hasher, event.payload.as_bytes())?;
		hash_provenance_field(&mut hasher, event.created_at.as_bytes())?;
		expected_sequence = expected_sequence.checked_add(1).ok_or_else(|| {
			MemoryError::Corrupt("compaction source sequence overflow".to_string())
		})?;
	}
	let expected_after = through_sequence
		.checked_add(1)
		.ok_or_else(|| MemoryError::Corrupt("compaction boundary overflow".to_string()))?;
	if expected_sequence != expected_after {
		return Err(MemoryError::Corrupt(format!(
			"session {session_id} transcript ends before compaction boundary {through_sequence}"
		)));
	}
	if turn.id.is_some() {
		return Err(MemoryError::Corrupt(format!(
			"session {session_id} compaction boundary splits an atomic turn"
		)));
	}
	let first_event_id = first_event_id.ok_or_else(|| {
		MemoryError::Corrupt(format!(
			"session {session_id} compaction source is unexpectedly empty"
		))
	})?;
	let last_event_id = last_event_id.ok_or_else(|| {
		MemoryError::Corrupt(format!(
			"session {session_id} compaction source has no final event"
		))
	})?;
	Ok(TranscriptProvenance {
		event_count: u64::try_from(through_sequence).map_err(|_| {
			MemoryError::Corrupt("negative compaction source event count".to_string())
		})?,
		first_event_id,
		last_event_id,
		sha256: hex::encode(hasher.finalize()),
	})
}

fn hash_provenance_field(hasher: &mut Sha256, value: &[u8]) -> Result<(), MemoryError> {
	let length = u64::try_from(value.len())
		.map_err(|_| MemoryError::Corrupt("provenance field length overflow".to_string()))?;
	hasher.update(length.to_be_bytes());
	hasher.update(value);
	Ok(())
}

#[derive(Deserialize)]
struct CompactionSummaryPayload {
	compaction: CompactionSummaryMetadata,
	#[serde(rename = "summary")]
	_summary: Value,
}

#[derive(Deserialize)]
struct CompactionSummaryMetadata {
	job_id: Uuid,
	through_sequence: u64,
	source: TranscriptProvenance,
}

fn validate_compaction_summary(
	event: &SessionEvent,
	job: &CompactionJob,
) -> Result<(), MemoryError> {
	if event.session_id != job.session_id
		|| event.kind != SessionEventKind::Summary
		|| event.sequence <= job.through_sequence
	{
		return Err(MemoryError::Corrupt(format!(
			"compaction {} summary event violates its boundary",
			job.id
		)));
	}
	let payload: CompactionSummaryPayload = serde_json::from_value(event.payload.clone())?;
	if payload.compaction.job_id != job.id
		|| payload.compaction.through_sequence != job.through_sequence
		|| payload.compaction.source != job.source
	{
		return Err(MemoryError::Corrupt(format!(
			"compaction {} summary provenance does not match its job",
			job.id
		)));
	}
	Ok(())
}

fn load_session_replay(
	connection: &Connection,
	session_id: Uuid,
) -> Result<SessionReplay, MemoryError> {
	let session_text = session_id.to_string();
	let sql = format!(
		"{} ORDER BY through_sequence DESC, updated_at DESC, id DESC LIMIT 1",
		compaction_select("WHERE session_id = ?1 AND state = 'completed'")
	);
	let raw_compaction = connection
		.query_row(&sql, [&session_text], raw_compaction)
		.optional()?;
	let last_sequence = last_event_sequence(connection, session_id)?;
	let mut events = Vec::new();
	let mut payload_bytes = 0_usize;
	let compacted_through = if let Some(raw) = raw_compaction {
		let summary_id = raw.summary_event_id.clone().ok_or_else(|| {
			MemoryError::Corrupt(format!("completed compaction {} has no summary", raw.id))
		})?;
		let job = CompactionJob::try_from(raw)?;
		let through_sql = i64::try_from(job.through_sequence).map_err(|_| {
			MemoryError::Corrupt("compaction boundary exceeds SQLite range".to_string())
		})?;
		let actual_source = transcript_provenance(connection, &session_text, through_sql)?;
		if actual_source != job.source {
			return Err(MemoryError::Corrupt(format!(
				"compaction {} source provenance no longer matches transcript",
				job.id
			)));
		}
		let summary_raw = connection
			.query_row(
				&event_select("WHERE id = ?1 AND session_id = ?2"),
				params![&summary_id, &session_text],
				raw_event,
			)
			.optional()?
			.ok_or_else(|| {
				MemoryError::Corrupt(format!("compaction {} summary event is missing", job.id))
			})?;
		let summary = bounded_replay_event(summary_raw, &mut payload_bytes, events.len())?;
		validate_compaction_summary(&summary, &job)?;
		events.push(summary);
		let tail_sql = format!(
			"{} ORDER BY sequence ASC",
			event_select("WHERE session_id = ?1 AND sequence > ?2 AND id != ?3")
		);
		let mut statement = connection.prepare(&tail_sql)?;
		let rows =
			statement.query_map(params![&session_text, through_sql, &summary_id], raw_event)?;
		for row in rows {
			let raw = row?;
			if raw.kind == SessionEventKind::Summary.as_str() {
				return Err(MemoryError::Corrupt(format!(
					"compaction {} leaves another summary in its replay tail",
					job.id
				)));
			}
			let event = bounded_replay_event(raw, &mut payload_bytes, events.len())?;
			events.push(event);
		}
		Some(job.through_sequence)
	} else {
		let all_sql = format!(
			"{} ORDER BY sequence ASC",
			event_select("WHERE session_id = ?1")
		);
		let mut statement = connection.prepare(&all_sql)?;
		let rows = statement.query_map([session_text], raw_event)?;
		for row in rows {
			let event = bounded_replay_event(row?, &mut payload_bytes, events.len())?;
			events.push(event);
		}
		None
	};
	validate_replay_turns(&events, session_id)?;
	Ok(SessionReplay {
		events,
		last_sequence,
		compacted_through,
		snapshot: None,
	})
}

fn validate_replay_turns(events: &[SessionEvent], session_id: Uuid) -> Result<(), MemoryError> {
	let mut start = 0_usize;
	while start < events.len() {
		let first = &events[start];
		if first.turn_index != 0 || first.turn_size == 0 {
			return Err(MemoryError::Corrupt(format!(
				"session {session_id} replay starts inside atomic turn {}",
				first.turn_id
			)));
		}
		let size = usize::try_from(first.turn_size)
			.map_err(|_| MemoryError::Corrupt("turn size exceeds platform range".to_string()))?;
		let end = start
			.checked_add(size)
			.ok_or_else(|| MemoryError::Corrupt("turn replay index overflow".to_string()))?;
		let turn = events.get(start..end).ok_or_else(|| {
			MemoryError::Corrupt(format!(
				"session {session_id} replay ends inside atomic turn {}",
				first.turn_id
			))
		})?;
		for (index, event) in turn.iter().enumerate() {
			let expected_index = u32::try_from(index)
				.map_err(|_| MemoryError::Corrupt("turn index exceeds u32".to_string()))?;
			if event.turn_id != first.turn_id
				|| event.turn_size != first.turn_size
				|| event.turn_index != expected_index
			{
				return Err(MemoryError::Corrupt(format!(
					"session {session_id} replay interleaves atomic turn {}",
					first.turn_id
				)));
			}
		}
		start = end;
	}
	Ok(())
}

fn bounded_replay_event(
	raw: RawEvent,
	payload_bytes: &mut usize,
	event_count: usize,
) -> Result<SessionEvent, MemoryError> {
	if event_count >= MAX_REPLAY_EVENTS {
		return Err(MemoryError::Invalid(format!(
			"session replay exceeds {MAX_REPLAY_EVENTS} event limit; compact more history"
		)));
	}
	*payload_bytes = payload_bytes
		.checked_add(raw.payload.len())
		.ok_or_else(|| MemoryError::Corrupt("session replay byte count overflow".to_string()))?;
	if *payload_bytes > MAX_REPLAY_BYTES {
		return Err(MemoryError::Invalid(format!(
			"session replay exceeds {MAX_REPLAY_BYTES} byte limit; compact more history"
		)));
	}
	SessionEvent::try_from(raw)
}

fn collect_session_page(
	rows: rusqlite::MappedRows<'_, impl FnMut(&rusqlite::Row<'_>) -> rusqlite::Result<RawSession>>,
	limit: usize,
) -> Result<SessionPage, MemoryError> {
	let mut items = Vec::new();
	let mut payload_bytes = 0_usize;
	let mut has_more = false;
	for row in rows {
		let raw = row?;
		if items.len() == limit {
			has_more = true;
			break;
		}
		let row_bytes = raw
			.id
			.len()
			.saturating_add(raw.workspace.len())
			.saturating_add(raw.model_reference.as_ref().map_or(0, String::len))
			.saturating_add(raw.model_snapshot.as_ref().map_or(0, String::len))
			.saturating_add(raw.title.as_ref().map_or(0, String::len));
		let next = payload_bytes
			.checked_add(row_bytes)
			.ok_or_else(|| MemoryError::Corrupt("Session page byte count overflow".to_string()))?;
		if !items.is_empty() && next > MAX_PAGE_PAYLOAD_BYTES {
			has_more = true;
			break;
		}
		payload_bytes = next;
		items.push(Session::try_from(raw)?);
	}
	let next = if has_more {
		items.last().map(|session| SessionCursor {
			updated_at: session.updated_at,
			id: session.id,
		})
	} else {
		None
	};
	Ok(SessionPage { items, next })
}

fn collect_knowledge_page(
	rows: rusqlite::MappedRows<
		'_,
		impl FnMut(&rusqlite::Row<'_>) -> rusqlite::Result<RawKnowledge>,
	>,
	limit: usize,
) -> Result<KnowledgePage, MemoryError> {
	let mut items = Vec::new();
	let mut payload_bytes = 0_usize;
	let mut has_more = false;
	for row in rows {
		let raw = row?;
		if items.len() == limit {
			has_more = true;
			break;
		}
		let next = payload_bytes
			.checked_add(raw.key.len())
			.and_then(|bytes| bytes.checked_add(raw.content.len()))
			.ok_or_else(|| {
				MemoryError::Corrupt("Knowledge page byte count overflow".to_string())
			})?;
		if !items.is_empty() && next > MAX_PAGE_PAYLOAD_BYTES {
			has_more = true;
			break;
		}
		payload_bytes = next;
		items.push(Knowledge::try_from(raw)?);
	}
	let next = if has_more {
		items.last().map(|knowledge| KnowledgeCursor {
			pinned: knowledge.pinned,
			updated_at: knowledge.updated_at,
			key: knowledge.key.clone(),
		})
	} else {
		None
	};
	Ok(KnowledgePage { items, next })
}

fn collect_knowledge_bounded(
	rows: rusqlite::MappedRows<
		'_,
		impl FnMut(&rusqlite::Row<'_>) -> rusqlite::Result<RawKnowledge>,
	>,
) -> Result<Vec<Knowledge>, MemoryError> {
	let mut items = Vec::new();
	let mut payload_bytes = 0_usize;
	for row in rows {
		let raw = row?;
		let next = payload_bytes
			.checked_add(raw.key.len())
			.and_then(|bytes| bytes.checked_add(raw.content.len()))
			.ok_or_else(|| {
				MemoryError::Corrupt("Knowledge result byte count overflow".to_string())
			})?;
		if !items.is_empty() && next > MAX_PAGE_PAYLOAD_BYTES {
			break;
		}
		payload_bytes = next;
		items.push(Knowledge::try_from(raw)?);
	}
	Ok(items)
}

fn count(connection: &Connection, table: &str) -> Result<u64, MemoryError> {
	let sql = format!("SELECT COUNT(*) FROM {table}");
	connection
		.query_row(&sql, [], |row| row.get(0))
		.map_err(MemoryError::from)
}

fn absolute_database_path(database: PathBuf) -> Result<PathBuf, MemoryError> {
	let absolute = if database.is_absolute() {
		database
	} else {
		std::env::current_dir()
			.map(|current| current.join(database))
			.map_err(|source| MemoryError::Io {
				operation: "resolve relative memory database path",
				path: PathBuf::from("."),
				source,
			})?
	};
	let name = absolute
		.file_name()
		.ok_or_else(|| MemoryError::Invalid("memory database path must name a file".to_string()))?;
	let parent = absolute.parent().ok_or_else(|| {
		MemoryError::Invalid("memory database path must have a parent directory".to_string())
	})?;
	fs::canonicalize(parent)
		.map(|canonical| canonical.join(name))
		.map_err(|source| MemoryError::Io {
			operation: "canonicalize memory database parent",
			path: parent.to_path_buf(),
			source,
		})
}

fn validate_database_parent(database: &Path) -> Result<(), MemoryError> {
	let parent = database.parent().ok_or_else(|| {
		MemoryError::Invalid("memory database path must have a parent directory".to_string())
	})?;
	let metadata = fs::symlink_metadata(parent).map_err(|source| MemoryError::Io {
		operation: "inspect memory database parent",
		path: parent.to_path_buf(),
		source,
	})?;
	if !metadata.file_type().is_dir() || metadata.file_type().is_symlink() {
		return Err(MemoryError::Invalid(format!(
			"memory database parent {} is not a real directory",
			parent.display()
		)));
	}
	if metadata.uid() != current_user_id() || metadata.permissions().mode() & 0o022 != 0 {
		return Err(MemoryError::Invalid(format!(
			"memory database parent {} must be owned by the current user and not writable by group or others",
			parent.display()
		)));
	}
	Ok(())
}

fn prepare_database_file(database: &Path) -> Result<(), MemoryError> {
	match OpenOptions::new()
		.read(true)
		.write(true)
		.create_new(true)
		.mode(0o600)
		.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
		.open(database)
	{
		Ok(file) => {
			file.sync_all().map_err(|source| MemoryError::Io {
				operation: "sync new memory database",
				path: database.to_path_buf(),
				source,
			})?;
			sync_parent_directory(database, "sync memory directory after database creation")?;
			Ok(())
		}
		Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
			validate_database_file(database)
		}
		Err(source) => Err(MemoryError::Io {
			operation: "create memory database",
			path: database.to_path_buf(),
			source,
		}),
	}
}

fn lock_database(database: &Path) -> Result<fs::File, MemoryError> {
	let mut lock_name = database.as_os_str().to_os_string();
	lock_name.push(".open.lock");
	let lock_path = PathBuf::from(lock_name);
	let file = OpenOptions::new()
		.read(true)
		.write(true)
		.create(true)
		.mode(0o600)
		.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
		.open(&lock_path)
		.map_err(|source| MemoryError::Io {
			operation: "open memory database migration lock",
			path: lock_path.clone(),
			source,
		})?;
	let metadata = file.metadata().map_err(|source| MemoryError::Io {
		operation: "inspect memory database migration lock",
		path: lock_path.clone(),
		source,
	})?;
	if !metadata.is_file()
		|| metadata.uid() != current_user_id()
		|| metadata.permissions().mode() & 0o077 != 0
	{
		return Err(MemoryError::Invalid(format!(
			"memory database migration lock {} must be a private current-user file",
			lock_path.display()
		)));
	}
	// SAFETY: `file` owns a live descriptor and `LOCK_EX` is a valid `flock` flag.
	if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } != 0 {
		return Err(MemoryError::Io {
			operation: "lock memory database migration",
			path: lock_path,
			source: io::Error::last_os_error(),
		});
	}
	Ok(file)
}

fn sync_parent_directory(path: &Path, operation: &'static str) -> Result<(), MemoryError> {
	let parent = path.parent().ok_or_else(|| {
		MemoryError::Invalid("memory storage path has no parent directory".to_string())
	})?;
	let directory = OpenOptions::new()
		.read(true)
		.custom_flags(libc::O_CLOEXEC | libc::O_DIRECTORY | libc::O_NOFOLLOW)
		.open(parent)
		.map_err(|source| MemoryError::Io {
			operation,
			path: parent.to_path_buf(),
			source,
		})?;
	directory.sync_all().map_err(|source| MemoryError::Io {
		operation,
		path: parent.to_path_buf(),
		source,
	})
}

fn validate_database_file(database: &Path) -> Result<(), MemoryError> {
	let file = OpenOptions::new()
		.read(true)
		.write(true)
		.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
		.open(database)
		.map_err(|source| MemoryError::Io {
			operation: "open memory database securely",
			path: database.to_path_buf(),
			source,
		})?;
	let metadata = file.metadata().map_err(|source| MemoryError::Io {
		operation: "inspect memory database",
		path: database.to_path_buf(),
		source,
	})?;
	if !metadata.is_file()
		|| metadata.uid() != current_user_id()
		|| metadata.permissions().mode() & 0o7777 != 0o600
	{
		return Err(MemoryError::Invalid(format!(
			"memory database {} must be a current-user-owned regular file with mode 0600",
			database.display()
		)));
	}
	Ok(())
}

fn current_user_id() -> u32 {
	crate::home::effective_user_id()
}

fn page_limit(limit: usize, maximum: usize, collection: &str) -> Result<usize, MemoryError> {
	if !(1..=maximum).contains(&limit) {
		return Err(MemoryError::Invalid(format!(
			"{collection} page limit must be between 1 and {maximum}"
		)));
	}
	Ok(limit)
}

fn file_size(path: &Path) -> Result<u64, MemoryError> {
	fs::metadata(path)
		.map(|metadata| metadata.len())
		.map_err(|source| MemoryError::Io {
			operation: "inspect memory database",
			path: path.to_path_buf(),
			source,
		})
}

struct WorkspaceBinding {
	path: PathBuf,
	identity: WorkspaceIdentity,
}

struct MigrationWorkspaceBinding {
	path: PathBuf,
	identity: WorkspaceIdentity,
	legacy: bool,
}

fn migration_workspace_binding(workspace: &Path) -> MigrationWorkspaceBinding {
	if let Ok(binding) = workspace_binding(workspace) {
		MigrationWorkspaceBinding {
			path: binding.path,
			identity: binding.identity,
			legacy: false,
		}
	} else {
		let digest = Sha256::digest(workspace.as_os_str().as_bytes());
		let mut inode_bytes = [0_u8; 8];
		inode_bytes.copy_from_slice(&digest[..8]);
		let inode = u64::from_be_bytes(inode_bytes).max(1);
		MigrationWorkspaceBinding {
			path: workspace.to_path_buf(),
			identity: WorkspaceIdentity { device: 0, inode },
			legacy: true,
		}
	}
}

fn workspace_binding(workspace: &Path) -> Result<WorkspaceBinding, MemoryError> {
	let canonical = fs::canonicalize(workspace).map_err(|source| MemoryError::Io {
		operation: "canonicalize workspace",
		path: workspace.to_path_buf(),
		source,
	})?;
	let metadata = fs::metadata(&canonical).map_err(|source| MemoryError::Io {
		operation: "inspect canonical workspace",
		path: canonical.clone(),
		source,
	})?;
	if !metadata.is_dir() {
		return Err(MemoryError::Invalid(format!(
			"workspace {} is not a directory",
			canonical.display()
		)));
	}
	Ok(WorkspaceBinding {
		path: canonical,
		identity: WorkspaceIdentity {
			device: metadata.dev(),
			inode: metadata.ino(),
		},
	})
}

fn parse_workspace_identity(
	device: Option<String>,
	inode: Option<String>,
) -> Result<WorkspaceIdentity, MemoryError> {
	let (Some(device), Some(inode)) = (device, inode) else {
		return Err(MemoryError::Corrupt(
			"session has no durable workspace identity".to_string(),
		));
	};
	let device = device.parse::<u64>().map_err(|error| {
		MemoryError::Corrupt(format!("workspace device {device:?} is invalid: {error}"))
	})?;
	let inode = inode.parse::<u64>().map_err(|error| {
		MemoryError::Corrupt(format!("workspace inode {inode:?} is invalid: {error}"))
	})?;
	Ok(WorkspaceIdentity { device, inode })
}

fn validate_workspace_identity(session: &Session) -> Result<(), MemoryError> {
	let actual = workspace_binding(&session.workspace)?.identity;
	if actual != session.workspace_identity {
		return Err(MemoryError::WorkspaceMismatch {
			session_id: session.id,
			expected: session.workspace_identity,
			actual,
		});
	}
	Ok(())
}

fn path_text(path: &Path) -> Result<&str, MemoryError> {
	path.to_str()
		.ok_or_else(|| MemoryError::Invalid(format!("path is not valid UTF-8: {}", path.display())))
}

fn validate_optional(value: Option<&str>, limit: usize, name: &str) -> Result<(), MemoryError> {
	if let Some(value) = value {
		validate_required(value, limit, name)?;
	}
	Ok(())
}

fn validate_required(value: &str, limit: usize, name: &str) -> Result<(), MemoryError> {
	if value.trim().is_empty() {
		return Err(MemoryError::Invalid(format!("{name} must not be empty")));
	}
	if value.len() > limit {
		return Err(MemoryError::Invalid(format!(
			"{name} exceeds {limit} byte limit"
		)));
	}
	Ok(())
}

fn parse_confidence(value: f64, name: &str) -> Result<f64, MemoryError> {
	if value.is_finite() && (0.0..=1.0).contains(&value) {
		Ok(value)
	} else {
		Err(MemoryError::Corrupt(format!(
			"{name} must be finite and in 0.0..=1.0"
		)))
	}
}

const fn ensure_changed(changed: usize, entity: &'static str, id: Uuid) -> Result<(), MemoryError> {
	if changed == 0 {
		Err(MemoryError::NotFound { entity, id })
	} else {
		Ok(())
	}
}

fn parse_uuid(value: &str, field: &str) -> Result<Uuid, MemoryError> {
	Uuid::parse_str(value)
		.map_err(|error| MemoryError::Corrupt(format!("{field} {value:?} is invalid: {error}")))
}

fn parse_time(value: &str, field: &str) -> Result<DateTime<Utc>, MemoryError> {
	DateTime::parse_from_rfc3339(value)
		.map(|time| time.with_timezone(&Utc))
		.map_err(|error| MemoryError::Corrupt(format!("{field} {value:?} is invalid: {error}")))
}

#[cfg(test)]
#[expect(
	clippy::unwrap_used,
	reason = "storage tests use panic-on-setup-failure assertions to keep fixtures concise"
)]
mod tests {
	use super::*;

	fn store() -> (tempfile::TempDir, EmelexHome, MemoryStore) {
		let directory = tempfile::tempdir().unwrap();
		let home = EmelexHome::prepare(&directory.path().join("home")).unwrap();
		let store = MemoryStore::open(&home).unwrap();
		(directory, home, store)
	}

	fn temp_store_mode(connection: &Connection) -> i64 {
		connection
			.pragma_query_value(None, "temp_store", |row| row.get(0))
			.unwrap()
	}

	#[test]
	fn every_memory_connection_keeps_sqlite_temporaries_in_memory() {
		let (_directory, _home, store) = store();

		let normal = store.connection().unwrap();
		assert_eq!(temp_store_mode(&normal), 2);
		let best_effort = store.best_effort_connection().unwrap();
		assert_eq!(temp_store_mode(&best_effort), 2);
		let read_only = open_snapshot_reference_connection(&store.database).unwrap();
		assert_eq!(temp_store_mode(&read_only), 2);
	}

	fn queued_compaction(store: &MemoryStore) -> (tempfile::TempDir, CompactionJob) {
		let workspace = tempfile::tempdir().unwrap();
		let session = store.start_session(workspace.path(), None).unwrap();
		store
			.append_event(
				session.id,
				SessionEventKind::UserMessage,
				&serde_json::json!({"text": "hello"}),
			)
			.unwrap();
		let job = store.queue_compaction(session.id, 1).unwrap();
		(workspace, job)
	}

	fn make_compaction_retry_eligible(store: &MemoryStore, id: Uuid) {
		store
			.connection()
			.unwrap()
			.execute(
				"UPDATE compaction_jobs SET retry_after = ?2 WHERE id = ?1",
				params![
					id.to_string(),
					(Utc::now() - chrono::Duration::seconds(1)).to_rfc3339()
				],
			)
			.unwrap();
	}

	fn create_v2_schema(connection: &Connection) {
		connection
			.execute_batch(
				"PRAGMA foreign_keys = ON;
				 CREATE TABLE sessions (
				   id TEXT PRIMARY KEY NOT NULL,
				   workspace TEXT NOT NULL,
				   model_reference TEXT,
				   title TEXT,
				   created_at TEXT NOT NULL,
				   updated_at TEXT NOT NULL,
				   model_snapshot TEXT
				 ) STRICT;
				 CREATE TABLE session_events (
				   id TEXT PRIMARY KEY NOT NULL,
				   session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
				   sequence INTEGER NOT NULL CHECK(sequence > 0),
				   kind TEXT NOT NULL,
				   payload_json TEXT NOT NULL CHECK(json_valid(payload_json)),
				   created_at TEXT NOT NULL,
				   UNIQUE(session_id, sequence)
				 ) STRICT;
				 CREATE TABLE compaction_jobs (
				   id TEXT PRIMARY KEY NOT NULL,
				   session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
				   through_sequence INTEGER NOT NULL CHECK(through_sequence > 0),
				   state TEXT NOT NULL CHECK(state IN ('pending','running','completed')),
				   claim_token TEXT,
				   lease_until TEXT,
				   created_at TEXT NOT NULL,
				   updated_at TEXT NOT NULL,
				   CHECK(
				     (state = 'running' AND claim_token IS NOT NULL
				      AND lease_until IS NOT NULL)
				     OR
				     (state != 'running' AND claim_token IS NULL
				      AND lease_until IS NULL)
				   ),
				   UNIQUE(session_id, through_sequence)
				 ) STRICT;
				 CREATE INDEX compaction_state_created
				   ON compaction_jobs(state, created_at);
				 CREATE TABLE knowledge (
				   id TEXT PRIMARY KEY NOT NULL,
				   workspace TEXT NOT NULL,
				   key TEXT NOT NULL,
				   active_version INTEGER NOT NULL CHECK(active_version > 0),
				   pinned INTEGER NOT NULL CHECK(pinned IN (0,1)),
				   created_at TEXT NOT NULL,
				   updated_at TEXT NOT NULL,
				   UNIQUE(workspace, key)
				 ) STRICT;
				 CREATE TABLE knowledge_versions (
				   knowledge_id TEXT NOT NULL REFERENCES knowledge(id) ON DELETE CASCADE,
				   version INTEGER NOT NULL CHECK(version > 0),
				   content TEXT NOT NULL,
				   source_session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
				   created_at TEXT NOT NULL,
				   PRIMARY KEY(knowledge_id, version)
				 ) STRICT;
				 PRAGMA user_version = 2;",
			)
			.unwrap();
	}

	#[test]
	fn session_events_are_ordered_and_durable() {
		let (_directory, _home, store) = store();
		let workspace = tempfile::tempdir().unwrap();
		let session = store.start_session(workspace.path(), Some("test")).unwrap();
		let first = store
			.append_event(
				session.id,
				SessionEventKind::UserMessage,
				&serde_json::json!({"text": "hello"}),
			)
			.unwrap();
		let second = store
			.append_event(
				session.id,
				SessionEventKind::AssistantMessage,
				&serde_json::json!({"text": "hi"}),
			)
			.unwrap();
		assert_eq!((first.sequence, second.sequence), (1, 2));
		let events = store.events(session.id, 0, 100).unwrap();
		assert_eq!(events.len(), 2);
		assert_eq!(events[1].payload["text"], "hi");
	}

	#[test]
	fn snapshot_reference_guard_is_lazy_and_exact() {
		let directory = tempfile::tempdir().unwrap();
		let home = EmelexHome::prepare(&directory.path().join("home")).unwrap();
		let installed = crate::models::install_test_snapshot(&home).unwrap();
		let guard = MemorySnapshotReferenceGuard::new(&home);
		let snapshot = installed.snapshot_id().clone();
		assert!(!home.database_file().exists());
		assert!(!guard.is_snapshot_referenced(&snapshot).unwrap());
		assert!(!home.database_file().exists());

		let store = MemoryStore::open(&home).unwrap();
		let workspace = tempfile::tempdir().unwrap();
		let session = store.start_session(workspace.path(), None).unwrap();
		store.bind_session_model(session.id, &installed).unwrap();
		assert!(guard.is_snapshot_referenced(&snapshot).unwrap());
	}

	#[test]
	fn append_turn_commits_complete_batch_with_consecutive_sequences() {
		let (_directory, _home, store) = store();
		let workspace = tempfile::tempdir().unwrap();
		let session = store.start_session(workspace.path(), None).unwrap();
		let mut lease = store.claim_session(session.id, workspace.path()).unwrap();
		let appended = store
			.append_turn(
				&mut lease,
				&[
					SessionEventInput::new(
						SessionEventKind::UserMessage,
						serde_json::json!({"text": "hello"}),
					),
					SessionEventInput::new(
						SessionEventKind::AssistantMessage,
						serde_json::json!({"text": "hi"}),
					),
				],
			)
			.unwrap();
		assert_eq!(
			appended
				.iter()
				.map(|event| event.sequence)
				.collect::<Vec<_>>(),
			vec![1, 2]
		);
		assert_eq!(
			appended
				.iter()
				.map(|event| (event.turn_id, event.turn_index, event.turn_size))
				.collect::<Vec<_>>(),
			vec![(appended[0].turn_id, 0, 2), (appended[0].turn_id, 1, 2)]
		);
	}

	#[test]
	fn append_turn_rolls_back_every_event_when_one_insert_fails() {
		let (_directory, _home, store) = store();
		let workspace = tempfile::tempdir().unwrap();
		let session = store.start_session(workspace.path(), None).unwrap();
		let mut lease = store.claim_session(session.id, workspace.path()).unwrap();
		store
			.connection()
			.unwrap()
			.execute_batch(
				"CREATE TRIGGER reject_assistant
				 BEFORE INSERT ON session_events
				 WHEN NEW.kind = 'assistant_message'
				 BEGIN
				   SELECT RAISE(ABORT, 'injected failure');
				 END;",
			)
			.unwrap();
		let result = store.append_turn(
			&mut lease,
			&[
				SessionEventInput::new(
					SessionEventKind::UserMessage,
					serde_json::json!({"text": "hello"}),
				),
				SessionEventInput::new(
					SessionEventKind::AssistantMessage,
					serde_json::json!({"text": "hi"}),
				),
			],
		);
		assert!(result.is_err());
		assert!(store.events(session.id, 0, 100).unwrap().is_empty());
	}

	#[test]
	fn live_session_claim_excludes_another_resumer() {
		let (_directory, _home, store) = store();
		let workspace = tempfile::tempdir().unwrap();
		let session = store.start_session(workspace.path(), None).unwrap();
		let lease = store.claim_session(session.id, workspace.path()).unwrap();
		let second = store.claim_session(session.id, workspace.path());
		assert!(matches!(second, Err(MemoryError::SessionBusy { .. })));
		store.release_session(&lease).unwrap();
		assert!(store.claim_session(session.id, workspace.path()).is_ok());
	}

	#[test]
	fn resumed_session_must_replay_before_appending() {
		let (_directory, _home, store) = store();
		let workspace = tempfile::tempdir().unwrap();
		let session = store.start_session(workspace.path(), None).unwrap();
		store
			.append_event(
				session.id,
				SessionEventKind::UserMessage,
				&serde_json::json!({"text": "existing"}),
			)
			.unwrap();
		let mut lease = store.claim_session(session.id, workspace.path()).unwrap();
		let before_replay = store.append_turn(
			&mut lease,
			&[SessionEventInput::new(
				SessionEventKind::AssistantMessage,
				serde_json::json!({"text": "unsafe"}),
			)],
		);
		assert!(matches!(
			before_replay,
			Err(MemoryError::ReplayRequired { .. })
		));
		store.replay_session(&mut lease).unwrap();
		assert!(
			store
				.append_turn(
					&mut lease,
					&[SessionEventInput::new(
						SessionEventKind::AssistantMessage,
						serde_json::json!({"text": "safe"}),
					)],
				)
				.is_ok()
		);
	}

	#[test]
	fn reclaimed_session_claim_invalidates_first_writer() {
		let (_directory, _home, store) = store();
		let workspace = tempfile::tempdir().unwrap();
		let session = store.start_session(workspace.path(), None).unwrap();
		let mut first = store.claim_session(session.id, workspace.path()).unwrap();
		store
			.connection()
			.unwrap()
			.execute(
				"UPDATE sessions SET execution_lease_until = ?2 WHERE id = ?1",
				params![
					session.id.to_string(),
					(Utc::now() - chrono::Duration::seconds(1)).to_rfc3339()
				],
			)
			.unwrap();
		let _second = store.claim_session(session.id, workspace.path()).unwrap();
		let result = store.append_turn(
			&mut first,
			&[SessionEventInput::new(
				SessionEventKind::UserMessage,
				serde_json::json!({"text": "stale"}),
			)],
		);
		assert!(matches!(result, Err(MemoryError::StaleSessionLease { .. })));
	}

	#[test]
	fn workspace_identity_survives_directory_rename() {
		let (_directory, _home, store) = store();
		let parent = tempfile::tempdir().unwrap();
		let original = parent.path().join("original");
		let renamed = parent.path().join("renamed");
		fs::create_dir(&original).unwrap();
		let session = store.start_session(&original, None).unwrap();
		fs::rename(&original, &renamed).unwrap();
		let lease = store.claim_session(session.id, &renamed).unwrap();
		assert_eq!(
			lease.session().workspace,
			fs::canonicalize(&renamed).unwrap()
		);
	}

	#[test]
	fn workspace_identity_rejects_directory_replaced_at_same_path() {
		let (_directory, _home, store) = store();
		let parent = tempfile::tempdir().unwrap();
		let workspace = parent.path().join("workspace");
		fs::create_dir(&workspace).unwrap();
		let session = store.start_session(&workspace, None).unwrap();
		fs::remove_dir(&workspace).unwrap();
		fs::create_dir(&workspace).unwrap();
		let result = store.claim_session(session.id, &workspace);
		assert!(matches!(result, Err(MemoryError::WorkspaceMismatch { .. })));
	}

	#[test]
	fn knowledge_is_workspace_scoped_and_versioned() {
		let (_directory, _home, store) = store();
		let workspace = tempfile::tempdir().unwrap();
		let first = store
			.remember(workspace.path(), "build", "cargo test", None)
			.unwrap();
		let second = store
			.remember(workspace.path(), "build", "cargo nextest run", None)
			.unwrap();
		assert_eq!(first.id, second.id);
		assert_eq!(second.active_version, 2);
		assert_eq!(
			store.knowledge_history(first.id, None, 100).unwrap().len(),
			2
		);
		store
			.activate_knowledge(workspace.path(), first.id, 1)
			.unwrap();
		assert_eq!(store.knowledge(first.id).unwrap().content, "cargo test");
		let third = store
			.remember(workspace.path(), "build", "cargo clippy", None)
			.unwrap();
		assert_eq!(third.active_version, 3);
		assert_eq!(
			store.knowledge_history(first.id, None, 100).unwrap().len(),
			3
		);
	}

	#[test]
	fn automatic_recall_filters_confidence_before_limit_and_never_exempts_pins() {
		let (_directory, _home, store) = store();
		let workspace = tempfile::tempdir().unwrap();
		let low = store
			.remember(workspace.path(), "low", "below threshold", None)
			.unwrap();
		let boundary = store
			.remember(workspace.path(), "boundary", "at threshold", None)
			.unwrap();
		let high = store
			.remember(workspace.path(), "high", "above threshold", None)
			.unwrap();
		let connection = store.connection().unwrap();
		for (id, confidence) in [(low.id, 0.69), (boundary.id, 0.70), (high.id, 0.90)] {
			connection
				.execute(
					"UPDATE knowledge_versions SET confidence = ?2
					 WHERE knowledge_id = ?1 AND version = 1",
					params![id.to_string(), confidence],
				)
				.unwrap();
		}
		drop(connection);
		store
			.set_knowledge_pinned(workspace.path(), low.id, true)
			.unwrap();
		store
			.set_knowledge_pinned(workspace.path(), boundary.id, true)
			.unwrap();

		let recalled = store.recall_knowledge(workspace.path(), 0.70, 2).unwrap();
		assert_eq!(recalled.len(), 2);
		assert_eq!(recalled[0].id, boundary.id);
		assert!(recalled.iter().any(|entry| entry.id == high.id));
		assert!(!recalled.iter().any(|entry| entry.id == low.id));
		assert!(
			store
				.recall_knowledge(workspace.path(), f64::NAN, 1)
				.is_err()
		);
	}

	#[test]
	fn compaction_claim_and_completion_are_durable() {
		let (_directory, _home, store) = store();
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
		let claimed = store.claim_compaction().unwrap().unwrap();
		assert_eq!(claimed.job().id, queued.id);
		let summary = store
			.complete_compaction(&claimed, &serde_json::json!({"text": "hello"}))
			.unwrap();
		assert_eq!(summary.kind, SessionEventKind::Summary);
		assert_eq!(
			summary.payload["compaction"]["job_id"],
			queued.id.to_string()
		);
		assert_eq!(summary.payload["compaction"]["through_sequence"], 1);
		assert_eq!(summary.payload["summary"]["text"], "hello");
		assert!(store.claim_compaction().unwrap().is_none());
	}

	#[test]
	fn replay_places_verified_summary_before_uncovered_tail() {
		let (_directory, _home, store) = store();
		let workspace = tempfile::tempdir().unwrap();
		let session = store.start_session(workspace.path(), None).unwrap();
		for text in ["one", "two"] {
			store
				.append_event(
					session.id,
					SessionEventKind::UserMessage,
					&serde_json::json!({"text": text}),
				)
				.unwrap();
		}
		store.queue_compaction(session.id, 2).unwrap();
		let compaction = store.claim_compaction().unwrap().unwrap();
		store
			.append_event(
				session.id,
				SessionEventKind::AssistantMessage,
				&serde_json::json!({"text": "tail"}),
			)
			.unwrap();
		let summary = store
			.complete_compaction(&compaction, &serde_json::json!({"text": "summary"}))
			.unwrap();
		let mut lease = store.claim_session(session.id, workspace.path()).unwrap();
		let replay = store.replay_session(&mut lease).unwrap();
		assert_eq!(replay.compacted_through, Some(2));
		assert_eq!(
			replay
				.events
				.iter()
				.map(|event| (&event.kind, event.sequence))
				.collect::<Vec<_>>(),
			vec![
				(&SessionEventKind::Summary, summary.sequence),
				(&SessionEventKind::AssistantMessage, 3)
			]
		);
	}

	#[test]
	fn replay_rejects_source_modified_after_compaction() {
		let (_directory, _home, store) = store();
		let workspace = tempfile::tempdir().unwrap();
		let session = store.start_session(workspace.path(), None).unwrap();
		store
			.append_event(
				session.id,
				SessionEventKind::UserMessage,
				&serde_json::json!({"text": "original"}),
			)
			.unwrap();
		store.queue_compaction(session.id, 1).unwrap();
		let compaction = store.claim_compaction().unwrap().unwrap();
		store
			.complete_compaction(&compaction, &serde_json::json!({"text": "summary"}))
			.unwrap();
		store
			.connection()
			.unwrap()
			.execute(
				"UPDATE session_events SET payload_json = '{\"text\":\"tampered\"}'
				 WHERE session_id = ?1 AND sequence = 1",
				[session.id.to_string()],
			)
			.unwrap();
		let mut lease = store.claim_session(session.id, workspace.path()).unwrap();
		let replay = store.replay_session(&mut lease);
		assert!(matches!(replay, Err(MemoryError::Corrupt(_))));
	}

	#[test]
	fn compaction_append_forces_claimed_session_to_replay_before_next_turn() {
		let (_directory, _home, store) = store();
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
		let mut lease = store.claim_session(session.id, workspace.path()).unwrap();
		store.replay_session(&mut lease).unwrap();
		let busy = store.complete_compaction(&compaction, &serde_json::json!({"text": "summary"}));
		assert!(matches!(busy, Err(MemoryError::SessionBusy { .. })));
		store.release_session(&lease).unwrap();
		store
			.complete_compaction(&compaction, &serde_json::json!({"text": "summary"}))
			.unwrap();
		let mut resumed = store.claim_session(session.id, workspace.path()).unwrap();
		let replay = store.replay_session(&mut resumed).unwrap();
		assert_eq!(replay.events[0].kind, SessionEventKind::Summary);
		assert!(
			store
				.append_turn(
					&mut resumed,
					&[SessionEventInput::new(
						SessionEventKind::AssistantMessage,
						serde_json::json!({"text": "fresh"}),
					)],
				)
				.is_ok()
		);
	}

	#[test]
	fn compaction_queue_is_idempotent_and_requires_existing_boundary() {
		let (_directory, _home, store) = store();
		let workspace = tempfile::tempdir().unwrap();
		let session = store.start_session(workspace.path(), None).unwrap();
		assert!(store.queue_compaction(session.id, 1).is_err());
		store
			.append_event(
				session.id,
				SessionEventKind::UserMessage,
				&serde_json::json!({"text": "hello"}),
			)
			.unwrap();
		let first = store.queue_compaction(session.id, 1).unwrap();
		let second = store.queue_compaction(session.id, 1).unwrap();
		assert_eq!(first.id, second.id);
		store
			.append_event(
				session.id,
				SessionEventKind::AssistantMessage,
				&serde_json::json!({"text": "tail"}),
			)
			.unwrap();
		assert!(store.queue_compaction(session.id, 2).is_err());
	}

	#[test]
	fn compaction_boundary_cannot_split_atomic_turn() {
		let (_directory, _home, store) = store();
		let workspace = tempfile::tempdir().unwrap();
		let session = store.start_session(workspace.path(), None).unwrap();
		let mut lease = store.claim_session(session.id, workspace.path()).unwrap();
		store
			.append_turn(
				&mut lease,
				&[
					SessionEventInput::new(
						SessionEventKind::UserMessage,
						serde_json::json!({"text": "hello"}),
					),
					SessionEventInput::new(
						SessionEventKind::AssistantMessage,
						serde_json::json!({"text": "hi"}),
					),
				],
			)
			.unwrap();
		assert!(store.queue_compaction(session.id, 1).is_err());
		assert!(store.queue_compaction(session.id, 2).is_ok());
	}

	#[test]
	fn session_model_binding_is_immutable() {
		let (_directory, home, store) = store();
		let installed = crate::models::install_test_snapshot(&home).unwrap();
		let workspace = tempfile::tempdir().unwrap();
		let session = store.start_session(workspace.path(), None).unwrap();
		store.bind_session_model(session.id, &installed).unwrap();
		store.bind_session_model(session.id, &installed).unwrap();
		let loaded = store.session(session.id).unwrap();
		assert_eq!(
			loaded.model_snapshot.as_ref(),
			Some(installed.snapshot_id())
		);
	}

	#[test]
	fn session_model_binding_rejects_live_execution_claim() {
		let (_directory, home, store) = store();
		let installed = crate::models::install_test_snapshot(&home).unwrap();
		let workspace = tempfile::tempdir().unwrap();
		let session = store.start_session(workspace.path(), None).unwrap();
		let lease = store.claim_session(session.id, workspace.path()).unwrap();
		assert!(matches!(
			store.bind_session_model(session.id, &installed),
			Err(MemoryError::SessionBusy { .. })
		));
		store.release_session(&lease).unwrap();
		store.bind_session_model(session.id, &installed).unwrap();
	}

	#[test]
	fn snapshot_mutation_lock_prevents_binding_a_removed_install() {
		let (_directory, home, store) = store();
		let installed = crate::models::install_test_snapshot(&home).unwrap();
		let workspace = tempfile::tempdir().unwrap();
		let session = store.start_session(workspace.path(), None).unwrap();
		let (ready_sender, ready_receiver) = std::sync::mpsc::sync_channel(0);
		let (remove_sender, remove_receiver) = std::sync::mpsc::sync_channel(0);
		let remover_home = home;
		let removed_path = installed.path().to_path_buf();
		let remover = std::thread::spawn(move || {
			let _mutation_lock = remover_home.lock_snapshot_mutations().unwrap();
			ready_sender.send(()).unwrap();
			remove_receiver.recv().unwrap();
			fs::set_permissions(&removed_path, fs::Permissions::from_mode(0o700)).unwrap();
			fs::remove_dir_all(removed_path).unwrap();
		});
		ready_receiver.recv().unwrap();
		let binder_store = store.clone();
		let binder =
			std::thread::spawn(move || binder_store.bind_session_model(session.id, &installed));
		remove_sender.send(()).unwrap();
		remover.join().unwrap();
		assert!(matches!(
			binder.join().unwrap(),
			Err(MemoryError::ModelSnapshot(_))
		));
		assert!(store.session(session.id).unwrap().model_snapshot.is_none());
	}

	#[test]
	fn lease_can_renew_after_expiry_until_recovery_schedules_retry() {
		let (_directory, _home, store) = store();
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
		let mut first = store.claim_compaction().unwrap().unwrap();
		store
			.connection()
			.unwrap()
			.execute(
				"UPDATE compaction_jobs SET lease_until = ?2 WHERE id = ?1",
				params![
					first.job().id.to_string(),
					(Utc::now() - chrono::Duration::seconds(1)).to_rfc3339()
				],
			)
			.unwrap();
		store.renew_compaction(&mut first).unwrap();
		store
			.connection()
			.unwrap()
			.execute(
				"UPDATE compaction_jobs SET lease_until = ?2 WHERE id = ?1",
				params![
					first.job().id.to_string(),
					(Utc::now() - chrono::Duration::seconds(1)).to_rfc3339()
				],
			)
			.unwrap();
		assert!(store.claim_compaction().unwrap().is_none());
		assert!(store.renew_compaction(&mut first).is_err());
		store
			.connection()
			.unwrap()
			.execute(
				"UPDATE compaction_jobs SET retry_after = ?2 WHERE id = ?1",
				params![
					first.job().id.to_string(),
					(Utc::now() - chrono::Duration::seconds(1)).to_rfc3339()
				],
			)
			.unwrap();
		let mut second = store.claim_compaction().unwrap().unwrap();
		store.renew_compaction(&mut second).unwrap();
	}

	#[test]
	fn removing_opened_database_never_recreates_it() {
		let (_directory, _home, store) = store();
		let path = store.database_path().to_path_buf();
		fs::remove_file(&path).unwrap();
		assert!(store.status().is_err());
		assert!(!path.exists());
	}

	#[test]
	fn knowledge_version_overflow_does_not_commit_unreadable_state() {
		let (_directory, _home, store) = store();
		let workspace = tempfile::tempdir().unwrap();
		let knowledge = store
			.remember(workspace.path(), "overflow", "first", None)
			.unwrap();
		let connection = store.connection().unwrap();
		connection
			.execute(
				"UPDATE knowledge_versions SET version = ?2
				 WHERE knowledge_id = ?1 AND version = 1",
				params![knowledge.id.to_string(), i64::from(u32::MAX)],
			)
			.unwrap();
		connection
			.execute(
				"UPDATE knowledge SET active_version = ?2 WHERE id = ?1",
				params![knowledge.id.to_string(), i64::from(u32::MAX)],
			)
			.unwrap();
		drop(connection);
		assert!(
			store
				.remember(workspace.path(), "overflow", "second", None)
				.is_err()
		);
		assert_eq!(
			store.knowledge(knowledge.id).unwrap().active_version,
			u32::MAX
		);
	}

	#[test]
	fn concurrent_first_open_serializes_migration() {
		let directory = tempfile::tempdir().unwrap();
		let parent = directory.path().join("memory");
		fs::create_dir(&parent).unwrap();
		fs::set_permissions(&parent, fs::Permissions::from_mode(0o700)).unwrap();
		let database = parent.join("concurrent.sqlite3");
		let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
		let handles = (0..2)
			.map(|_| {
				let database = database.clone();
				let barrier = std::sync::Arc::clone(&barrier);
				std::thread::spawn(move || {
					barrier.wait();
					MemoryStore::open_path(database)
				})
			})
			.collect::<Vec<_>>();
		for handle in handles {
			let opened = handle.join().unwrap();
			assert!(opened.is_ok(), "concurrent open failed: {opened:?}");
		}
	}

	#[test]
	#[expect(
		clippy::too_many_lines,
		reason = "the complete legacy schema stays inline so migration coverage mirrors version one"
	)]
	fn version_one_migration_rebuilds_compaction_constraints() {
		let directory = tempfile::tempdir().unwrap();
		let database = directory.path().join("v1.sqlite3");
		OpenOptions::new()
			.read(true)
			.write(true)
			.create_new(true)
			.mode(0o600)
			.open(&database)
			.unwrap();
		let connection = Connection::open(&database).unwrap();
		connection
			.execute_batch(
				"PRAGMA foreign_keys = ON;
				 CREATE TABLE sessions (
				   id TEXT PRIMARY KEY NOT NULL,
				   workspace TEXT NOT NULL,
				   model_reference TEXT,
				   title TEXT,
				   created_at TEXT NOT NULL,
				   updated_at TEXT NOT NULL
				 ) STRICT;
				 CREATE TABLE session_events (
				   id TEXT PRIMARY KEY NOT NULL,
				   session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
				   sequence INTEGER NOT NULL CHECK(sequence > 0),
				   kind TEXT NOT NULL,
				   payload_json TEXT NOT NULL CHECK(json_valid(payload_json)),
				   created_at TEXT NOT NULL,
				   UNIQUE(session_id, sequence)
				 ) STRICT;
				 CREATE TABLE compaction_jobs (
				   id TEXT PRIMARY KEY NOT NULL,
				   session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
				   through_sequence INTEGER NOT NULL CHECK(through_sequence > 0),
				   state TEXT NOT NULL CHECK(state IN ('pending','running','completed')),
				   created_at TEXT NOT NULL,
				   updated_at TEXT NOT NULL,
				   UNIQUE(session_id, through_sequence)
				 ) STRICT;
				 CREATE INDEX compaction_state_created
				   ON compaction_jobs(state, created_at);
				 CREATE TABLE knowledge (
				   id TEXT PRIMARY KEY NOT NULL,
				   workspace TEXT NOT NULL,
				   key TEXT NOT NULL,
				   active_version INTEGER NOT NULL CHECK(active_version > 0),
				   pinned INTEGER NOT NULL CHECK(pinned IN (0,1)),
				   created_at TEXT NOT NULL,
				   updated_at TEXT NOT NULL,
				   UNIQUE(workspace, key)
				 ) STRICT;
				 CREATE TABLE knowledge_versions (
				   knowledge_id TEXT NOT NULL REFERENCES knowledge(id) ON DELETE CASCADE,
				   version INTEGER NOT NULL CHECK(version > 0),
				   content TEXT NOT NULL,
				   source_session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
				   created_at TEXT NOT NULL,
				   PRIMARY KEY(knowledge_id, version)
				 ) STRICT;
				 PRAGMA user_version = 1;",
			)
			.unwrap();
		let session = Uuid::now_v7();
		let job = Uuid::now_v7();
		let now = Utc::now().to_rfc3339();
		connection
			.execute(
				"INSERT INTO sessions
				 (id, workspace, model_reference, title, created_at, updated_at)
				 VALUES (?1, ?2, NULL, NULL, ?3, ?3)",
				params![
					session.to_string(),
					directory.path().display().to_string(),
					now
				],
			)
			.unwrap();
		connection
			.execute(
				"INSERT INTO session_events
				 (id, session_id, sequence, kind, payload_json, created_at)
				 VALUES (?1, ?2, 1, 'user_message', '{\"text\":\"hello\"}', ?3)",
				params![Uuid::now_v7().to_string(), session.to_string(), now],
			)
			.unwrap();
		connection
			.execute(
				"INSERT INTO compaction_jobs
				 (id, session_id, through_sequence, state, created_at, updated_at)
				 VALUES (?1, ?2, 1, 'running', ?3, ?3)",
				params![job.to_string(), session.to_string(), now],
			)
			.unwrap();
		drop(connection);

		let store = MemoryStore::open_path(&database).unwrap();
		let connection = store.connection().unwrap();
		let state: String = connection
			.query_row(
				"SELECT state FROM compaction_jobs WHERE id = ?1",
				[job.to_string()],
				|row| row.get(0),
			)
			.unwrap();
		assert_eq!(state, "pending");
		assert!(
			connection
				.execute(
					"INSERT INTO compaction_jobs
					 (id, session_id, through_sequence, state, claim_token, lease_until,
					  created_at, updated_at)
					 VALUES (?1, ?2, 2, 'running', NULL, NULL, ?3, ?3)",
					params![Uuid::now_v7().to_string(), session.to_string(), now],
				)
				.is_err()
		);
	}

	#[test]
	fn version_two_migration_is_lossless_for_vanished_and_duplicate_workspaces() {
		let directory = tempfile::tempdir().unwrap();
		let database = directory.path().join("v2.sqlite3");
		OpenOptions::new()
			.read(true)
			.write(true)
			.create_new(true)
			.mode(0o600)
			.open(&database)
			.unwrap();
		let connection = Connection::open(&database).unwrap();
		create_v2_schema(&connection);
		let session = Uuid::now_v7();
		let now = Utc::now().to_rfc3339();
		let vanished = directory.path().join("vanished-workspace");
		connection
			.execute(
				"INSERT INTO sessions
				 (id, workspace, model_reference, title, created_at, updated_at,
				  model_snapshot)
				 VALUES (?1, ?2, 'owner/model', NULL, ?3, ?3, ?4)",
				params![
					session.to_string(),
					vanished.display().to_string(),
					&now,
					"owner/model@0123456789012345678901234567890123456789"
				],
			)
			.unwrap();
		connection
			.execute(
				"INSERT INTO session_events
				 (id, session_id, sequence, kind, payload_json, created_at)
				 VALUES (?1, ?2, 1, 'user_message', '{\"text\":\"kept\"}', ?3)",
				params![Uuid::now_v7().to_string(), session.to_string(), &now],
			)
			.unwrap();

		let workspace = directory.path().join("workspace");
		let alias = directory.path().join("workspace-alias");
		fs::create_dir(&workspace).unwrap();
		std::os::unix::fs::symlink(&workspace, &alias).unwrap();
		for (path, content) in [(&workspace, "newer"), (&alias, "older")] {
			let id = Uuid::now_v7();
			connection
				.execute(
					"INSERT INTO knowledge
					 (id, workspace, key, active_version, pinned, created_at, updated_at)
					 VALUES (?1, ?2, 'duplicate', 1, 0, ?3, ?3)",
					params![id.to_string(), path.display().to_string(), &now],
				)
				.unwrap();
			connection
				.execute(
					"INSERT INTO knowledge_versions
					 (knowledge_id, version, content, source_session_id, created_at)
					 VALUES (?1, 1, ?2, NULL, ?3)",
					params![id.to_string(), content, &now],
				)
				.unwrap();
		}
		drop(connection);

		let store = MemoryStore::open_path(&database).unwrap();
		let migrated = store.session(session).unwrap();
		assert_eq!(migrated.workspace_identity.device, 0);
		assert_eq!(store.events(session, 0, 10).unwrap().len(), 1);
		let connection = store.connection().unwrap();
		let (rows, legacy, identities, versions): (i64, i64, i64, i64) = connection
			.query_row(
				"SELECT
				   (SELECT COUNT(*) FROM knowledge WHERE key = 'duplicate'),
				   (SELECT SUM(legacy_identity) FROM knowledge WHERE key = 'duplicate'),
				   (SELECT COUNT(DISTINCT workspace_device || ':' || workspace_inode)
				      FROM knowledge WHERE key = 'duplicate'),
				   (SELECT COUNT(*) FROM knowledge_versions)",
				[],
				|row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
			)
			.unwrap();
		assert_eq!((rows, legacy, identities, versions), (2, 1, 2, 2));
		assert!(
			connection
				.execute(
					"INSERT INTO session_events
					 (id, session_id, sequence, turn_id, turn_index, turn_size,
					  kind, payload_json, created_at)
					 VALUES (?1, ?2, 2, ?3, 1, 1, 'user_message', '{}', ?4)",
					params![
						Uuid::now_v7().to_string(),
						session.to_string(),
						Uuid::now_v7().to_string(),
						now
					],
				)
				.is_err()
		);
	}

	#[test]
	fn session_and_knowledge_cursors_reach_older_rows() {
		let (_directory, _home, store) = store();
		let workspace = tempfile::tempdir().unwrap();
		for title in ["one", "two", "three"] {
			store.start_session(workspace.path(), Some(title)).unwrap();
		}
		let first = store.sessions(Some(workspace.path()), None, 2).unwrap();
		assert_eq!(first.items.len(), 2);
		let second = store
			.sessions(Some(workspace.path()), first.next.as_ref(), 2)
			.unwrap();
		assert_eq!(second.items.len(), 1);

		for key in ["alpha", "beta", "gamma"] {
			store.remember(workspace.path(), key, key, None).unwrap();
		}
		let first = store
			.knowledge_for_workspace(workspace.path(), None, 2)
			.unwrap();
		assert_eq!(first.items.len(), 2);
		let second = store
			.knowledge_for_workspace(workspace.path(), first.next.as_ref(), 2)
			.unwrap();
		assert_eq!(second.items.len(), 1);
	}

	#[test]
	fn event_page_enforces_aggregate_payload_budget() {
		let (_directory, _home, store) = store();
		let workspace = tempfile::tempdir().unwrap();
		let session = store.start_session(workspace.path(), None).unwrap();
		let text = "x".repeat(3 << 20);
		for _ in 0..6 {
			store
				.append_event(
					session.id,
					SessionEventKind::UserMessage,
					&serde_json::json!({"text": text}),
				)
				.unwrap();
		}
		let first = store.events(session.id, 0, 100).unwrap();
		assert!(!first.is_empty());
		assert!(first.len() < 6);
		let last = first.last().unwrap().sequence;
		assert!(!store.events(session.id, last, 100).unwrap().is_empty());
	}

	#[test]
	fn retry_backoff_does_not_block_newer_compaction() {
		let (_directory, _home, store) = store();
		let (_first_workspace, first) = queued_compaction(&store);
		let (_second_workspace, second) = queued_compaction(&store);
		let first_lease = store.claim_compaction().unwrap().unwrap();
		assert_eq!(first_lease.job().id, first.id);
		let outcome = store
			.record_compaction_failure(
				&first_lease,
				"model returned invalid JSON",
				MemoryJobFailureDisposition::Retry,
			)
			.unwrap();
		assert!(matches!(
			outcome,
			MemoryJobFailureOutcome::RetryScheduled { failures: 1, .. }
		));
		let second_lease = store.claim_compaction().unwrap().unwrap();
		assert_eq!(second_lease.job().id, second.id);
		store
			.complete_compaction(&second_lease, &serde_json::json!({"text": "summary"}))
			.unwrap();
	}

	#[test]
	fn third_compaction_failure_is_inspectable_and_explicitly_retryable() {
		let (_directory, _home, store) = store();
		let (_workspace, job) = queued_compaction(&store);
		for expected in 1..=2 {
			let lease = store.claim_compaction().unwrap().unwrap();
			let outcome = store
				.record_compaction_failure(
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
			make_compaction_retry_eligible(&store, job.id);
		}
		let lease = store.claim_compaction().unwrap().unwrap();
		let outcome = store
			.record_compaction_failure(
				&lease,
				"model returned invalid JSON",
				MemoryJobFailureDisposition::Retry,
			)
			.unwrap();
		assert!(matches!(
			outcome,
			MemoryJobFailureOutcome::Failed { failures: 3, .. }
		));
		assert!(store.claim_compaction().unwrap().is_none());
		let status = store.status().unwrap();
		assert_eq!(
			(status.failed_compactions, status.pending_compactions),
			(1, 0)
		);
		let failures = store.failed_jobs(10).unwrap();
		assert_eq!(
			(
				failures[0].id,
				failures[0].failures,
				failures[0].error.as_str()
			),
			(job.id, 3, "model returned invalid JSON")
		);
		store.retry_failed_job(job.id).unwrap();
		let retried = store.claim_compaction().unwrap().unwrap();
		assert_eq!((retried.job().id, retried.job().failures), (job.id, 0));
		store.release_compaction(&retried).unwrap();
	}

	#[test]
	fn permanent_compaction_failure_dead_letters_immediately() {
		let (_directory, _home, store) = store();
		let (_workspace, job) = queued_compaction(&store);
		let lease = store.claim_compaction().unwrap().unwrap();
		let outcome = store
			.record_compaction_failure(
				&lease,
				"source exceeds model context",
				MemoryJobFailureDisposition::Permanent,
			)
			.unwrap();
		assert!(matches!(
			outcome,
			MemoryJobFailureOutcome::Failed { failures: 1, .. }
		));
		assert_eq!(store.failed_jobs(1).unwrap()[0].id, job.id);
	}

	#[test]
	fn worker_failure_columns_reject_inconsistent_queue_state() {
		let (_directory, _home, store) = store();
		let (_workspace, job) = queued_compaction(&store);
		let connection = store.connection().unwrap();
		assert!(
			connection
				.execute(
					"UPDATE compaction_jobs SET retry_after = ?2 WHERE id = ?1",
					params![job.id.to_string(), Utc::now().to_rfc3339()],
				)
				.is_err()
		);
		assert!(
			connection
				.execute(
					"UPDATE compaction_jobs
					 SET state = 'failed', last_error = 'bad', failed_at = ?2
					 WHERE id = ?1",
					params![job.id.to_string(), Utc::now().to_rfc3339()],
				)
				.is_err()
		);
	}

	#[test]
	#[expect(
		clippy::too_many_lines,
		reason = "the complete v4 queue schema verifies the lossless v5 table rebuild"
	)]
	fn version_four_migration_preserves_jobs_and_adds_failure_state() {
		let directory = tempfile::tempdir().unwrap();
		let database = directory.path().join("v4.sqlite3");
		OpenOptions::new()
			.read(true)
			.write(true)
			.create_new(true)
			.mode(0o600)
			.open(&database)
			.unwrap();
		let connection = Connection::open(&database).unwrap();
		connection
			.execute_batch(
				"PRAGMA foreign_keys = ON;
				 CREATE TABLE sessions (id TEXT PRIMARY KEY NOT NULL) STRICT;
				 CREATE TABLE knowledge (id TEXT PRIMARY KEY NOT NULL) STRICT;
				 CREATE TABLE session_snapshots (
				   session_id TEXT PRIMARY KEY NOT NULL
				     REFERENCES sessions(id) ON DELETE CASCADE,
				   schema_version INTEGER NOT NULL CHECK(schema_version > 0),
				   config_json TEXT NOT NULL CHECK(json_valid(config_json)),
				   tools_json TEXT NOT NULL CHECK(json_valid(tools_json)),
				   created_at TEXT NOT NULL,
				   updated_at TEXT NOT NULL
				 ) STRICT;
				 CREATE TABLE assets (
				   sha256 TEXT PRIMARY KEY NOT NULL
				     CHECK(length(sha256) = 64),
				   bytes INTEGER NOT NULL CHECK(bytes >= 0),
				   created_at TEXT NOT NULL,
				   verified_at TEXT NOT NULL
				 ) STRICT;
				 CREATE TABLE compaction_jobs (
				   id TEXT PRIMARY KEY NOT NULL,
				   session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
				   through_sequence INTEGER NOT NULL CHECK(through_sequence > 0),
				   state TEXT NOT NULL CHECK(state IN ('pending','running','completed')),
				   claim_token TEXT,
				   lease_until TEXT,
				   source_event_count INTEGER NOT NULL CHECK(source_event_count = through_sequence),
				   source_first_event_id TEXT NOT NULL,
				   source_last_event_id TEXT NOT NULL,
				   source_sha256 TEXT NOT NULL CHECK(length(source_sha256) = 64),
				   summary_event_id TEXT,
				   created_at TEXT NOT NULL,
				   updated_at TEXT NOT NULL,
				   UNIQUE(session_id, through_sequence)
				 ) STRICT;
				 CREATE INDEX compaction_state_created
				   ON compaction_jobs(state, created_at);
				 CREATE UNIQUE INDEX compaction_summary_event
				   ON compaction_jobs(summary_event_id) WHERE summary_event_id IS NOT NULL;
				 CREATE TABLE distillation_jobs (
				   id TEXT PRIMARY KEY NOT NULL,
				   session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
				   through_sequence INTEGER NOT NULL CHECK(through_sequence > 0),
				   source_event_count INTEGER NOT NULL CHECK(source_event_count > 0),
				   source_first_event_id TEXT NOT NULL,
				   source_last_event_id TEXT NOT NULL,
				   source_sha256 TEXT NOT NULL CHECK(length(source_sha256) = 64),
				   state TEXT NOT NULL CHECK(state IN ('pending','running','completed')),
				   claim_token TEXT,
				   lease_until TEXT,
				   created_at TEXT NOT NULL,
				   updated_at TEXT NOT NULL,
				   completed_at TEXT,
				   UNIQUE(session_id, source_sha256)
				 ) STRICT;
				 CREATE INDEX distillation_state_created
				   ON distillation_jobs(state, created_at);
				 CREATE TABLE distillation_results (
				   job_id TEXT NOT NULL REFERENCES distillation_jobs(id) ON DELETE CASCADE,
				   candidate_index INTEGER NOT NULL CHECK(candidate_index >= 0),
				   knowledge_id TEXT NOT NULL REFERENCES knowledge(id) ON DELETE CASCADE,
				   knowledge_version INTEGER NOT NULL CHECK(knowledge_version > 0),
				   created_at TEXT NOT NULL,
				   PRIMARY KEY(job_id, candidate_index)
				 ) STRICT;
				 PRAGMA user_version = 4;",
			)
			.unwrap();
		let session = Uuid::now_v7();
		let knowledge = Uuid::now_v7();
		let compaction = Uuid::now_v7();
		let distillation = Uuid::now_v7();
		let first_event = Uuid::now_v7();
		let now = Utc::now().to_rfc3339();
		let digest = "a".repeat(64);
		connection
			.execute(
				"INSERT INTO sessions (id) VALUES (?1)",
				[session.to_string()],
			)
			.unwrap();
		connection
			.execute(
				"INSERT INTO session_snapshots
				 (session_id, schema_version, config_json, tools_json, created_at, updated_at)
				 VALUES (?1, 1, ?2, ?3, ?4, ?4)",
				params![
					session.to_string(),
					r#"{"mode":"stable"}"#,
					r#"{"tools":["echo"]}"#,
					&now
				],
			)
			.unwrap();
		connection
			.execute(
				"INSERT INTO knowledge (id) VALUES (?1)",
				[knowledge.to_string()],
			)
			.unwrap();
		connection
			.execute(
				"INSERT INTO compaction_jobs
				 (id, session_id, through_sequence, state, claim_token, lease_until,
				  source_event_count, source_first_event_id, source_last_event_id,
				  source_sha256, summary_event_id, created_at, updated_at)
				 VALUES (?1, ?2, 1, 'pending', NULL, NULL, 1, ?3, ?3, ?4,
				         NULL, ?5, ?5)",
				params![
					compaction.to_string(),
					session.to_string(),
					first_event.to_string(),
					&digest,
					&now
				],
			)
			.unwrap();
		connection
			.execute(
				"INSERT INTO distillation_jobs
				 (id, session_id, through_sequence, source_event_count,
				  source_first_event_id, source_last_event_id, source_sha256,
				  state, claim_token, lease_until, created_at, updated_at, completed_at)
				 VALUES (?1, ?2, 1, 1, ?3, ?3, ?4, 'pending',
				         NULL, NULL, ?5, ?5, NULL)",
				params![
					distillation.to_string(),
					session.to_string(),
					first_event.to_string(),
					&digest,
					&now
				],
			)
			.unwrap();
		connection
			.execute(
				"INSERT INTO distillation_results
				 (job_id, candidate_index, knowledge_id, knowledge_version, created_at)
				 VALUES (?1, 0, ?2, 1, ?3)",
				params![distillation.to_string(), knowledge.to_string(), &now],
			)
			.unwrap();
		drop(connection);
		let store = MemoryStore::open_path(&database).unwrap();
		let connection = store.connection().unwrap();
		let version: i64 = connection
			.pragma_query_value(None, "user_version", |row| row.get(0))
			.unwrap();
		let compaction_row: (String, i64, Option<String>) = connection
			.query_row(
				"SELECT state, failure_count, last_error
				 FROM compaction_jobs WHERE id = ?1",
				[compaction.to_string()],
				|row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
			)
			.unwrap();
		let distillation_row: (String, i64, Option<String>) = connection
			.query_row(
				"SELECT state, failure_count, last_error
				 FROM distillation_jobs WHERE id = ?1",
				[distillation.to_string()],
				|row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
			)
			.unwrap();
		let result_rows: i64 = connection
			.query_row(
				"SELECT COUNT(*) FROM distillation_results WHERE job_id = ?1",
				[distillation.to_string()],
				|row| row.get(0),
			)
			.unwrap();
		let snapshot: (String, String) = connection
			.query_row(
				"SELECT config_json, authority_json
				 FROM session_snapshots WHERE session_id = ?1",
				[session.to_string()],
				|row| Ok((row.get(0)?, row.get(1)?)),
			)
			.unwrap();
		let recovery_tables: i64 = connection
			.query_row(
				"SELECT COUNT(*) FROM sqlite_master
				 WHERE type = 'table'
				   AND name IN (
				     'active_agent_turns',
				     'active_agent_turn_assets',
				     'pending_tool_batches',
				     'pending_tool_invocations',
				     'pending_tool_assets'
				   )",
				[],
				|row| row.get(0),
			)
			.unwrap();
		assert_eq!(
			(
				version,
				compaction_row,
				distillation_row,
				result_rows,
				snapshot,
				recovery_tables
			),
			(
				6,
				("pending".to_string(), 0, None),
				("pending".to_string(), 0, None),
				1,
				(
					r#"{"mode":"stable"}"#.to_string(),
					r#"{"tools":["echo"]}"#.to_string()
				),
				5
			)
		);
	}

	#[test]
	fn sqlite_open_rejects_database_symlink() {
		let directory = tempfile::tempdir().unwrap();
		let real = directory.path().join("real.sqlite3");
		fs::write(&real, []).unwrap();
		let linked = directory.path().join("linked.sqlite3");
		std::os::unix::fs::symlink(&real, &linked).unwrap();
		assert!(MemoryStore::open_path(linked).is_err());
	}
}
