# Installed models

The model manager owns immutable snapshots and external-model link records
under Emelex Home. Hub installs pin a commit. Local copy and move imports select
a runtime-only subset, stage, hash, statically inspect, fully load through MLX,
write a validated manifest, and publish with one rename.

`emelex model import PATH [--name NAME]` copies by default and leaves its
source untouched. The name defaults from the canonical directory.
`--move` completes the same safe publication before retiring source material.
It removes only selected files that are still identical to what Emelex copied;
changed files and anything outside the runtime plan remain. Cleanup can
therefore warn and leave the source directory without invalidating the
committed snapshot. The copy-first transaction works across filesystems and
temporarily requires space for both copies.

`--symlink` writes an owner-only managed record whose controlled link points to
the canonical caller-owned model directory. External targets can change,
disappear, or live on an unavailable volume, so they are not immutable or
self-contained installs. Every resolve revalidates link identity, runtime
inventory, and full hashes. Every load repeats those checks before compatibility
validation and runtime loading.

One local name and content digest cannot silently change ownership mode or link
target. Such a collision preserves the healthy existing record and returns a
typed conflict; remove that exact snapshot before re-importing.

Candidate-specific failures from planning and certification are surfaced as
`ModelsError::Certification`, allowing an interactive catalog explorer to
offer another candidate. These include a repository becoming private or
incompatible between search and plan, the selected revision changing before
download, plan fit rejection, static artifact rejection, model load rejection,
and bounded probe rejection. Interactive discovery uses
`download_revision_controlled` so it never silently installs a revision
different from the result shown. A healthy exact snapshot is revalidated under
the mutation lock and reused before any Hub plan or disk preflight, so selecting
an already-downloaded result remains offline-safe. Network, cancellation, storage,
effective-policy configuration, task-join, panic, and global runtime failures
keep their original typed variants and remain fatal.

Hub storage encodes repository arity explicitly:
`models/hub/unnamespaced/<repo>/<revision>` or
`models/hub/namespaced/<namespace>/<repo>/<revision>`. This covers the full
catalog without making an unnamespaced repository an ancestor of a
namespaced one.

Each exact Hub revision also has one durable transfer workspace under
`temp/models/hub/`, using the same arity-safe path followed by
`emelex-transfer.json`, a stable `.emelex-transfer.lock`, and a `payload`
directory. The per-revision advisory lock is acquired before the installed
snapshot is rechecked. This prevents duplicate transfers across processes:
another caller waits with cooperative cancellation, then reuses the completed
install or resumes the same validated payload. Complete planned files and
bounded `.part` prefixes survive cancellation, observer failure, transient
transport failure, and retriable I/O. Disk preflight counts only bytes not
already staged.

`hub_transfer_statuses` exposes valid exact-revision workspaces without
following links or hashing payloads. A held lock reports `Downloading`; a
valid unlocked record and payload report `Paused`. Invalid or lock-only
coordination directories are omitted. Terminal protocol, integrity, and
certification failures move the payload to recoverable quarantine and remove
its transfer record. Successful publication also removes the record and any
redundant payload while leaving the stable lock inode in place, closing the
unlink-and-recreate race for later callers.

Controlled Hub installs observe cancellation between every local phase.
Post-transfer hashing is async and chunk-cancellable, and a mandatory final
checkpoint runs immediately before publication. Cancellation observed before
that commit point leaves no visible installed destination and retains a
resumable Hub payload; once the final checkpoint wins, the atomic publication
may complete. A post-rename failure quarantines the moved destination
synchronously while the snapshot-mutation lock is still held, so an incomplete
destination cannot race a later reuse or durable binding.
Every install owns an internal cancellation handle, so dropping either
download future asks detached blocking phases to stop before publication. A
controlled download links that private handle to caller cancellation without
gaining authority to cancel sibling operations. Non-Hub failed staging cleanup
and pre-publication Hub cleanup are dispatched to one named background
quarantine worker instead of running filesystem syncs or spawning overflow
threads in async-future Drop. Inventory/stamp checks and contained file opens
also run on blocking workers; cancellable hashing remains async.

`inventory` returns healthy owned snapshots and valid link records plus
diagnostics for corrupt or unavailable candidates. `resolve` selects the newest
healthy snapshot or fully revalidates the selected external target;
`resolve_snapshot` requires an exact content address. Owned-snapshot removal
accepts only that exact `ModelSnapshotId`; a stable reference cannot remove
whichever revision happens to be newest. Linked-model removal deletes only the
managed record and never modifies the external target.

`inspect_installed` revalidates one resolved record and derives static
compatibility using the current Emelex rules without loading MLX or rewriting
the immutable manifest. Callers may use current static evidence to fill gaps
left by an older certification pass while retaining recorded runtime-only
evidence.

`installed_hub_snapshots` and `hub_transfer_statuses` are the
cancellation-safe status paths for interactive Hub discovery. They scan only
managed Hub state, omit corrupt candidates, and never hash caller-owned linked
imports.

Owned loads validate the exact runtime inventory and either hash every file or
use a descriptor-bound stamp whose metadata still matches. Linked loads never
use that persistent immutable-snapshot fast path; they fully validate the
external target. `load_policy` exposes the fully resolved configuration,
including the single canonical `LoadOverride<T>` tri-state for each optional
sampling value and model context limits. Public load options are non-exhaustive
and have fluent setters. `ModelLoadOptions::maximum_context` selects the largest
architecture-declared context that fits the manager's Metal budget, capped by
the Client-supported 16,777,216-token load ceiling. A later fixed-context
setter replaces that mode, and a later maximum-context setter clears the fixed
override. Adaptive selection reserves and enforces a 16,384-token aggregate
prompt-cache ceiling even when caching is off by default, because per-request
options may re-enable it. Fixed contexts preserve full-context cache capacity.
Missing sizing, exact weight bytes, or a declared model maximum falls back to
resolved configuration instead of guessing. `load` and `load_policy` run the
same selection path. Nonzero speculation requires runtime-verified MTP.
The final compatibility gate repeats sizing with
`ModelLoadPolicy::prompt_cache_tokens`, the exact capacity also applied during
client construction.
`ModelLoadPolicy::context_selection` distinguishes a successful Metal-budget
maximum from configured, explicitly fixed, or incomplete-metadata fallback
contexts, so callers can describe the result without overstating its source.
Install, import, and verification probes are invocation-policy-neutral: they
force MTP off, thinking off, and clear the reasoning budget. Ordinary loads
still inherit resolved configuration unless a per-load override replaces it.

Owned-snapshot removal holds the shared snapshot-mutation lock, rejects
snapshots still bound to durable sessions, and moves data into recoverable
quarantine. Link removal applies the same managed-record authority but never
deletes caller-owned target data. A directly constructed `ModelManager`
installs the lazy durable-memory guard by default; library callers may replace
it with another typed `SnapshotReferenceGuard`. Quarantine records carry their
own durable timestamp; explicit garbage collection is the only permanent
deletion path.
