# Installed models — context

## Invariants

- Visible owned snapshots are complete, manifest-valid, hash-valid, and
  runtime load-verified. Managed external links are a distinct weaker storage
  mode: resolve revalidates link identity, runtime inventory, and full content
  hashes; load repeats that work before compatibility checks and runtime load.
- Candidate-specific planning and certification failures have a dedicated
  `ModelsError::Certification` boundary. Only static JSON/config/layout
  rejection, candidate privacy/compatibility/fit changes during planning,
  manifest validation, and narrow model load/probe failures enter it. Network,
  cancellation, transfer/hash/storage, effective policy, task-join, panic,
  Emelex Home, and global runtime failures do not.
- Controlled Hub installs checkpoint cancellation around local verification,
  inspection, load probing, manifest creation, and publication. Staged hashes
  use cancellable async chunks; the final checkpoint precedes the atomic rename.
- Both download APIs own cancel-on-future-drop authority, including the plain
  reporter API. Dropped joins may leave blocking workers running briefly, but
  they observe the owned cancellation at the final pre-rename checkpoint.
  Controlled operations link to, but never mutate, caller cancellation.
  Cancellation is cooperative: once that checkpoint wins, rename may commit.
- `StagingGuard::Drop` only dispatches best-effort quarantine work to one named
  background worker. Its nonblocking queue contains only tiny tasks for
  already-existing staging directories; no overflow threads or quarantine I/O
  run from an async runtime worker.
- Controlled verification dispatches inventory walks, stamp reads, and secure
  contained opens to blocking workers, with cancellation checks around every
  join. Descriptor hashing uses cancellable async chunks.
- Inventory skips corrupt or unavailable candidates and retains bounded
  per-entry diagnostics.
- Owned snapshot paths and all link records stay under the selected Emelex
  Home. A link record may name one canonical caller-owned external target.
- Hub snapshot paths carry an explicit `unnamespaced` or `namespaced`
  discriminator before validated repository components, preventing
  one-component IDs from colliding with two-component IDs.
- Owned files are read-only after publication; updates create a new revision.
- Copy import leaves its source untouched. Move import commits the copied
  snapshot first, then retires only selected source files whose identity and
  content still match the copy. Changed and unselected files remain, cleanup
  failures warn, and the committed snapshot is never rolled back.
- Move import is copy-then-retire rather than rename. It works across
  filesystems and requires full temporary duplicate capacity.
- Symlink import creates a managed record whose controlled link points to one
  canonical external target. That target is mutable and may be unavailable;
  every resolve/load performs full link, inventory, and hash validation, while
  load additionally performs compatibility checks and runtime loading.
- A healthy local name/digest collision across ownership modes or canonical
  link targets is a typed conflict. Import preserves the existing record;
  authority changes require exact removal followed by re-import.
- Failed staging and removals move to Emelex quarantine.
- Owned snapshots remain self-contained for offline load. A linked model uses
  no network but depends on its external target remaining locally available.
- A descriptor-bound verification stamp permits a persistent hash fast path
  only while file identity, size, mtime, and ctime remain unchanged.
- Removal, durable session binding, and quarantine deletion share one
  cross-process snapshot-mutation lock. Reference checks and mutations occur
  while that lock remains held.
- `ModelManager::new` installs the lazy durable-memory reference guard. A
  replacement guard returns `SnapshotReferenceError`; guard failures fail
  closed and preserve the snapshot.
- Runtime-verified MTP includes layout validation during actual Client load;
  parity certification is a separate gate.
- Lifecycle compatibility probes force speculation off, thinking off, and no
  reasoning budget. Normal loads continue to inherit configuration or apply
  explicit per-load overrides.
- Effective load policy is concrete and inspectable. Per-load tri-state
  overrides distinguish inheritance, explicit values, and clearing. Each
  option has one canonical field; no parallel `Option<T>` representation
  exists.
- Destructive removal resolves an exact `ModelSnapshotId`, never a mutable
  stable reference.
- Removing an external link deletes only its managed record. It never deletes,
  quarantines, chmods, or otherwise modifies the external target.
