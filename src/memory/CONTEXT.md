# Memory context

- Current SQLite schema version: 6.
- Database default: `<Emelex Home>/memory/emelex.sqlite3`; assets:
  `<Emelex Home>/memory/assets/<sha256>`.
- Every normal, read-only reference, and zero-wait best-effort connection sets
  `temp_store=MEMORY`; SQLite must not spill Session or Knowledge data into OS
  temporary storage outside Emelex Home.
- `MemoryStore::open(&home)` supports installed-model binding.
  `MemoryStore::open_path` is a memory-only embedding path and deliberately
  lacks Emelex Home snapshot-mutation authority.
- A Session's workspace device/inode is its authority boundary. Canonical path
  spelling is descriptive and may change after a rename.
  `Session::validate_workspace` applies this identity check before explicit
  resume model resolution.
- `SessionLease`, `CompactionLease`, and `DistillationLease` are opaque,
  non-cloneable, non-serializable RAII capabilities. Tokens are redacted.
  Their destructors use zero-wait best-effort cleanup, so database contention
  defers recovery to lease expiry instead of blocking an async runtime thread.
  Explicit release/failure APIs remain the durable completion path.
- Compaction and distillation persist `failure_count`, `retry_after`, a bounded
  4 KiB `last_error`, and `failed_at`. Retryable failures use 30-second
  exponential backoff capped at five minutes; failure three is terminal.
  Permanent failures are terminal immediately. Explicit cancellation releases
  a claim without incrementing the counter; lease expiry is a counted failure.
- Terminal jobs are excluded from claims and source-digest idempotency prevents
  silent recreation. `failed_jobs` inspects them and `retry_failed_job`
  explicitly resets the counter while retaining job identity/provenance.
- `append_turn` is the ordinary multi-event commit boundary. It validates all
  JSON and asset links before commit and never permits a caller-created
  Summary. Schema-v6 active-turn and pending-tool tables add explicit atomic
  boundaries for crash-safe agent checkpoints and recovery.
- A Session must replay before append. Replay returns the latest raw sequence
  cursor, verifies completed compaction provenance, substitutes one Summary for
  the covered prefix, and validates complete non-interleaved turns.
- `SessionSnapshot` stores resolved configuration and tool schemas outside
  transcript history. Resume with different authority fails. Only a Session
  with no durable sequence or replay events may bootstrap a missing snapshot;
  existing history without one fails closed.
- `DurableAgentSession` owns the pairing of in-memory model history and durable
  lease. Input is journaled before inference. Pending tool calls transition
  through planned, started, and completed states; complete call/result batches
  checkpoint into model history before another model round. A later failure
  preserves those checkpoints and adds a diagnostic. Failure before any
  checkpoint adds a diagnostic without model history. Title changes use the
  same live lease. Checkpoints, lease renewal, asset/fsync work, and failure
  commits run on blocking workers. The adapter is poisoned before its first
  await and disarmed only after full reconciliation; dropping a run future
  forbids reuse and clean-close distillation until resume/recovery.
- Resume automatically closes a tool-free interrupted active turn. Pending
  batches retain exact completed results, mark planned calls not executed, and
  never reinvoke tools. Started calls without results require explicit
  `memory sessions recover --accept-unknown-effects` reconciliation. Recovery
  advances invocation journals restart-idempotently, then atomically publishes
  one structurally complete replay batch and removes its recoverable journal.
- Exact model binding takes a typed `InstalledModel`. The Emelex Home
  snapshot-mutation lock is held across path/manifest/file revalidation and DB
  commit. Model removal holds the same lock across reference check and mutation.
- Direct `ModelManager` construction installs the lazy memory reference guard;
  reference-query failures are typed and fail closed.
- Assets are byte-addressed, mode `0600`, bounded to 128 MiB each, and verified
  on reuse/read. The default database uses `memory/assets`; explicit database
  paths use `<database-name>.assets` siblings. Event links are transactional
  and replay checks exact count, ordinal, digest, and kind. Active temporary
  files are inode-locked and have a one-hour minimum GC age. Replay has a
  256 MiB aggregate media limit.
- Compaction uses a fixed 80% trigger and 50% target, preserving four recent
  turns by default with a conservative byte-based token estimate. A live
  Session lease excludes both claim and completion; post-claim contention
  returns the job to pending without consuming retry budget.
- Chat `/compact` only queues. Exit chat, run `emelex memory work`, then resume
  so the worker can claim the job and replay can install its verified Summary.
- Distillation is idempotent by source digest, skips live Sessions, caps output
  at 128 unique keys, and records source range/digest plus candidate ordinal.
- `DurableAgentSession::close` only queues distillation and releases authority.
  It never executes worker/model work; callers opt into that separately.
- Knowledge identity is `(workspace_device, workspace_inode, key)`. Versions
  are immutable and confidence-bearing. Automatic tombstones cannot override
  pins. Tombstone and version provenance survives Session deletion.
- Every Knowledge mutation that accepts an ID also accepts and verifies the
  current workspace. Unscoped ID mutation is forbidden.
- Maintenance is bounded per phase. It records stale worker claims as failed
  attempts, retains live
  Session/worker leases and pins, tombstones before deletion, prunes historical
  versions, runs race-safe asset GC, checkpoints WAL, and vacuums only when
  explicitly requested.
- Session deletion is transactional logical deletion, not secure erasure.
  SQLite free pages/WAL retention remains governed by maintenance; CLI prompt
  history is separate from the Session store.
- JSON persistence uses capped streaming serialization before allocation of the
  stored string. Reads enforce row, item, and aggregate-byte bounds.
- v2-to-v3 migration rebuilds tables to restore `NOT NULL`, `CHECK`, turn, lease,
  and provenance constraints. v3-to-v4 preserves vanished-workspace rows with a
  fail-closed legacy identity and disambiguates alias paths without data loss.
  v4-to-v5 rebuilds both worker queues with constrained retry and terminal
  failure state while preserving every existing job and distillation result.
  v5-to-v6 renames snapshot tool JSON to authority JSON and adds constrained
  active-turn, pending-invocation, result-origin, and asset-reference journals;
  existing Sessions and transcript events remain unchanged.
