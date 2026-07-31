# Memory

`MemoryStore` is Emelex's durable local state engine. It owns Sessions,
transcript events, immutable execution snapshots, content-addressed media,
compaction and distillation work queues, versioned workspace Knowledge, and
bounded retention.

## Durability and authority

The database defaults to `<Emelex Home>/memory/emelex.sqlite3`. It is a
current-user-owned mode-`0600` file beneath a non-group-writable directory.
Connections use foreign keys, bounded busy waits, WAL, `synchronous=EXTRA`,
macOS full-fsync pragmas, `SQLITE_OPEN_NOFOLLOW`, and in-memory SQLite temporary
storage. The latter can consume memory for large sorts, but prevents prompt or
Session data from spilling into an OS temporary directory outside Emelex Home.
Normal reopen paths never recreate a deleted database. First-open and
migrations are serialized by a process mutex plus a private cross-process flock
sidecar.

Sessions bind a canonical workspace directory and its device/inode identity.
Directory renames remain valid; path replacement fails closed. Resuming requires
an opaque, expiring `SessionLease`. A lease must replay durable history before
its first append, and every append rechecks the lease, workspace identity, and
raw sequence cursor. `Session::validate_workspace` exposes the same identity
preflight, letting explicit CLI resumes fail before model resolution/load.
Lease guards release best-effort on `Drop` through a zero-wait SQLite
connection. A contended destructor returns immediately and lets lease expiry
recover authority; callers needing a durable transition use explicit
release/failure APIs.

`append_turn` is the ordinary atomic history boundary. It validates bounded
JSON by streaming into a capped serializer, assigns one turn ID, and commits
consecutive event positions and Session sequences under `BEGIN IMMEDIATE`. A
failed event or asset link rolls back the complete append. Summary events are
reserved for the compaction worker. Durable agent turns additionally use
schema-v6 active-turn and pending-tool journals so a process failure cannot
erase the fact that a host tool may already have run.

Model binding accepts a typed `InstalledModel`, not caller-supplied strings.
Emelex takes the Home-wide snapshot-mutation lock, revalidates the installed
path and manifest, then validates either the owned snapshot stamp or the linked
target's identity, inventory, and full hashes. It derives the stable reference
and exact snapshot ID and holds the lock through the database commit. Model
removal uses the same lock and a lazy `MemorySnapshotReferenceGuard`, closing
the check/mutate race across processes. `ModelManager::new` installs this guard
by default, so direct library construction cannot silently bypass retained
Session references.

## Durable agent adapter

`DurableAgentSession` pairs the native `AgentSession` with one Session lease.
Resume verifies replay, reconstructs model messages, and requires an identical
`SessionSnapshot` containing resolved semantic configuration and exact tool
authority. Snapshot creation is allowed only for an empty Session; existing
history without one fails closed rather than adopting current semantics.
Snapshots live outside the compactable transcript.

Each turn records its input before model work. Before tool execution, Emelex
journals the complete proposed batch and transitions every invocation through
`planned`, `started`, and `completed`. A complete assistant-call/result batch
is checkpointed into model history as soon as its results are known, even if a
later model round, tool, or event consumer fails. Successful terminal answers
close the active turn with the remaining messages and ordered, non-delta audit
events. Failures before a tool checkpoint add only a bounded diagnostic;
failures after one retain the checkpointed history and add a diagnostic.

Resume closes a tool-free interrupted turn with a visible failure record. It
also reconciles interrupted tool batches without invoking any tool again:
durable completed results remain exact, never-started calls become
`not executed`, and started calls without results become conservative
unknown-effect results. Unknown effects require explicit acceptance through
`emelex memory sessions recover SESSION --accept-unknown-effects`; ordinary
resume fails closed until they are reconciled. Invocation journals advance
restart-idempotently in small transactions; the final replay-visible
assistant-call/result batch publishes atomically and removes the recoverable
journal.

The adapter also delegates an in-memory enabled-tool subset for attended chat.
That subset may only remove or restore tools already present in the immutable
authority snapshot; it cannot grant new authority or bypass approval. It is
process-local rather than durable configuration, so a fresh resume starts with
every snapshotted tool enabled. Historical declarations required to replay a
complete prior tool round remain available to the model protocol, while the
execution gate still rejects calls to a currently disabled tool.

Lease renewal runs during inference and tool execution. Checkpoint, renewal,
asset, and failed-turn persistence work executes on blocking workers; SQLite
busy waits and fsync never run in the agent async poll. Title updates use the
adapter's live claim rather than contending with it. The adapter arms its poison
state before the first await. Dropping a turn future therefore prevents reuse
or distillation until explicit resume/recovery reconciles the journal. Clean
close queues idempotent Knowledge distillation only for an unpoisoned adapter
and releases execution authority.

## Assets

Media bytes are stored once at `memory/assets/<lowercase-sha256>` for the
standard database. Explicit `open_path` databases use a sibling
`<database-name>.assets/` namespace, preventing independent catalogs from
collecting each other's files. Writes stream through a 128 MiB cap into a
mode-`0600` temporary file, fsync it, publish with no-clobber semantics, and
verify existing content before cataloging it. Events carry small typed
`AssetRef` values and link them transactionally.

Reads verify ownership, mode, byte count, SHA-256, and catalog metadata.
Replayed media has an aggregate 256 MiB cap. Asset collection is bounded and
coordinates catalog deletion, file removal, and concurrent publication under
SQLite write locks. Active temporary files hold cross-process inode locks;
collection skips them and applies a one-hour minimum temporary-file age.
Referenced assets are never collected; crash-created orphans remain
recoverable through the grace period. Replay verifies each event's exact asset
count, ordinal, digest, and media kind against its transactional links.

## Compaction and distillation

`CompactionPolicy` triggers at 80% of the model context window and targets 50%,
preserving at least four newest atomic turns by default. Jobs record the exact
source prefix using event count, first/last IDs, and a domain-separated SHA-256.
Workers skip compaction jobs while their Session execution lease is live, and
completion rechecks the same boundary for post-claim races. It revalidates
worker authority and source bytes, then atomically appends a provenance-bearing
Summary. Replay verifies that provenance, places the Summary before the
uncovered tail, and never exposes the replaced raw prefix.

Chat's `/compact` command only plans and queues work. The current chat lease
prevents that job from running in place. Exit chat, run
`emelex memory work`, then resume the Session to use the verified Summary.

Clean exit queues distillation by `(Session, source digest)`. Workers skip live
Sessions, hold expiring RAII leases, read bounded source events, and atomically
apply at most 128 unique Knowledge candidates. Upserts append immutable
versions. Tombstones preserve confidence and source provenance; pinned entries
cannot be tombstoned by automatic distillation. Queueing and execution are
separate contracts: chat close only queues and releases its lease; an explicit
`emelex memory work` invocation (or library worker) performs model inference.

Both worker queues persist failed-attempt counts, bounded diagnostics, and
retry deadlines. Retryable failures back off from 30 seconds to a five-minute
cap; the third failure becomes terminal. Permanent failures, including a
source larger than the selected model's safe context budget, become terminal
immediately. `failed_jobs` exposes bounded diagnostics, and
`retry_failed_job` explicitly resets one terminal job without changing its
identity or source provenance. An expired worker lease is itself a counted
failure; explicit operator cancellation releases authority without consuming
an attempt. If a Session becomes live after a worker claim, interactive
execution wins: the worker releases that job to pending without consuming an
attempt.

## Knowledge and retention

Knowledge is scoped by workspace device/inode identity, not path spelling.
Every version stores confidence and optional immutable transcript provenance.
Deleting a source Session clears its nullable source foreign key while retaining
the source Session UUID, sequence range, and digest for audit. Mutations require
the caller's current workspace identity. Normal recall excludes tombstones.

`maintain` performs capped stale-lease failure transitions, age-based Session
and Knowledge retention, version pruning, asset GC, `PRAGMA optimize`, WAL
truncation, and optional `VACUUM`. It never deletes live Session/worker leases
or pinned Knowledge. Destructive Knowledge changes pass through tombstones and
a grace period.

`delete_session` removes the Session transactionally and deletes assets that
become unreferenced. It does not claim forensic erasure: ordinary SQLite free
pages and WAL bytes can remain until checkpoint/vacuum maintenance. CLI prompt
history is a separate cache outside this module and is not Session-addressable.

Collection APIs impose item and aggregate-byte limits. Session and active
Knowledge pages use stable composite cursors; transcript and version history use
exclusive monotonic cursors. Models decide summary and distillation content;
the CLI only orchestrates and presents these storage primitives.
