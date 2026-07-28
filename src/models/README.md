# Installed models

The model manager owns immutable snapshots under Emelex Home. Hub installs pin
a commit; local imports copy a runtime-only subset. Both paths stage, hash,
statically inspect, fully load through MLX, write a validated manifest, and
publish with one rename.

Candidate-specific failures from planning and certification are surfaced as
`ModelsError::Certification`, allowing an interactive catalog explorer to
offer another candidate. These include a repository becoming private or
incompatible between search and plan, plan fit rejection, static artifact
rejection, model load rejection, and bounded probe rejection. Network,
cancellation, storage, effective-policy configuration, task-join, panic, and
global runtime failures keep their original typed variants and remain fatal.

Hub storage encodes repository arity explicitly:
`models/hub/unnamespaced/<repo>/<revision>` or
`models/hub/namespaced/<namespace>/<repo>/<revision>`. This covers the full
catalog without making an unnamespaced repository an ancestor of a
namespaced one.

Controlled Hub installs observe cancellation between every local phase.
Post-transfer hashing is async and chunk-cancellable, and a mandatory final
checkpoint runs immediately before publication. Cancellation observed before
that commit point leaves no visible installed destination; once the final
checkpoint wins, the atomic publication may complete. Every install owns an
internal cancellation handle, so dropping either download future asks detached
blocking phases to stop before publication. A controlled download links that
private handle to caller cancellation without gaining authority to cancel
sibling operations. Failed staging cleanup is dispatched to one named
background quarantine worker instead of running filesystem syncs or spawning
overflow threads in async-future Drop. Inventory/stamp checks and contained
file opens also run on blocking workers; cancellable hashing remains async.

`inventory` returns healthy snapshots plus diagnostics for corrupt candidates.
`resolve` selects the newest healthy snapshot for a stable reference;
`resolve_snapshot` requires an exact immutable address. Removal accepts only
that exact `ModelSnapshotId`; a stable reference cannot remove whichever
revision happens to be newest.

Loads validate the exact runtime inventory and either hash every file or use a
descriptor-bound stamp whose metadata still matches. `load_policy` exposes the
fully resolved configuration, including the single canonical
`LoadOverride<T>` tri-state for each optional sampling value and model context
limits. Public load options are non-exhaustive and have fluent setters.
Nonzero speculation requires runtime-verified MTP.
Install, import, and verification probes are invocation-policy-neutral: they
force MTP off, thinking off, and clear the reasoning budget. Ordinary loads
still inherit resolved configuration unless a per-load override replaces it.

Removal holds the shared snapshot-mutation lock, rejects snapshots still bound
to durable sessions, and moves data into recoverable quarantine. A directly
constructed `ModelManager` installs the lazy durable-memory guard by default;
library callers may replace it with another typed `SnapshotReferenceGuard`.
Quarantine records carry their own durable timestamp; explicit garbage
collection is the only permanent deletion path.
