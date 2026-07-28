# ADR 0007: Local model import ownership modes

## Status

Accepted on 2026-07-28. This supersedes the local-import ownership assumptions
in ADR 0002.

## Context

Local checkpoints may be large enough that always retaining two complete
copies is undesirable. Callers also need a deliberate distinction between
giving Emelex an owned snapshot and registering an external checkpoint that
remains under caller control.

Those choices must not blur their guarantees. An Emelex-owned snapshot is
immutable, self-contained below Emelex Home, and available without its import
source. An external target can change or disappear independently. A move must
also avoid deleting files that Emelex did not select into its runnable plan.

## Decision

`emelex model import PATH` has three explicit ownership modes. Its name
defaults to the canonical directory name; `--name NAME` overrides it.

The default mode copies the selected runnable file set into owner-only staging,
hashes and validates it, runtime-probes it, and atomically publishes an
Emelex-owned immutable snapshot. The source remains untouched.

`--move` uses the same copy, validation, and publication transaction. It does
not rename the supplied directory into Emelex Home. Only after publication
succeeds does Emelex retire selected source files that are still the exact
files copied. Files that changed are retained and reported. Files outside the
selected runtime plan are never removed, so the source directory may remain.
A retirement warning does not undo or hide the already committed install.
Because the transfer is copy-then-retire, it works across filesystems and
temporarily requires enough space for both copies.

`--symlink` creates an owner-only managed link record below Emelex Home. Its
controlled link points to the canonical external model directory. Emelex does
not claim that target as an immutable, self-contained, or always-available
snapshot. Resolve and load therefore perform full link, inventory, and content
validation instead of using the persistent immutable-snapshot fast path. Load
then performs normal compatibility checks and runtime loading against that
freshly validated target. Removing the model removes only the managed link
record and never removes or modifies the external target.

The three modes are mutually exclusive. Copy and move produce owned snapshots;
symlink produces an external link registration. A link target's canonical path
is provenance and authority, not Emelex-owned storage. If the same local name
and content address already exists under another ownership mode or link target,
import returns a typed conflict and preserves the healthy existing record.
Changing that authority requires exact removal followed by re-import.

## Consequences

- Copy remains the safest default and preserves the existing immutable-store
  contract.
- Move never risks unrelated source files and never depends on same-filesystem
  rename behavior.
- A successful move may leave source files or directories behind and must say
  so visibly.
- Linked models trade storage duplication for weaker availability and
  immutability guarantees.
- Durable and human-facing surfaces must distinguish owned snapshots from
  managed external links.
- Removing a linked model cannot become authority to delete caller-owned data.
