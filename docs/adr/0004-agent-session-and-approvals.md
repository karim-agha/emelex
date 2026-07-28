# ADR 0004: Agent sessions, workspaces, and approvals

## Status

Accepted on 2026-07-26.

## Context

`emelex chat` must be a useful coding-agent harness in any directory, while the
same behavior remains reusable as a library. Tool execution touches real
files, processes, and networks, so a vague “sandboxed” promise would be
incorrect. Durable conversation state also creates ordering and workspace
identity requirements that a presentation-only CLI cannot enforce safely.

## Decision

The library owns `AgentSession`, its typed event stream, tool loop, limits,
history validation, cancellation, durable session integration, built-in
workspace tools, and approval interfaces. The CLI owns terminal input,
rendering, and interactive approval prompts.

An agent session is bound to one canonical workspace. Ordinary reads are
workspace-first. Outside-root reads, likely-secret reads, mutations, and host
shell commands require approval. Descriptor-relative operations guard against
symlink and time-of-check/time-of-use substitution. Approved host shell
commands run in their own process group, with bounded time and output.
Approval grants are one-shot and process-local; they are never restored from
durable memory.

Emelex is not a security sandbox. Approved shell commands execute on the host.
Web tools use no ambient credentials but may reach HTTP(S) targets, including
local or private addresses. These facts remain visible in public documentation
and interactive prompts.

Durable turns are appended transactionally. A resumable session has one active
execution lease, an immutable workspace binding, ordered event history, and
explicit compaction provenance. Each input opens an active-turn journal before
model work. A proposed tool batch records every invocation as planned, started,
or completed before Emelex advances its side-effect boundary.

A complete assistant-call/result batch is checkpointed into model history as
soon as every result is known, even when a later model round or event consumer
fails. A process failure never causes automatic tool re-invocation. Recovery
retains exact completed results, records planned calls as not executed, and
records started calls without durable results as unknown effects. Unknown
effects require explicit operator acceptance. A tool-free interrupted turn is
closed with a visible failure record. Recovery advances invocation-journal
rows restart-idempotently, then publishes one complete replay-visible batch
atomically and removes its recoverable journal.

Each Session also stores an immutable semantic configuration and exact tool
authority snapshot, including schemas and limits. Resume uses that snapshot;
it never substitutes current project or global configuration. Changing model,
system instructions, generation semantics, or tool authority starts a new
Session rather than rewriting the meaning of prior turns.

The standalone CLI preserves portable chat controls without carrying host
product integrations:

- the current directory is the default workspace; global
  `-C`/`--directory` (and its visible `--root` alias) selects another
  invocation and project-configuration root;
- `--max-tokens`, `--temperature`, and `--thinking` apply only to a new
  Session and enter its immutable snapshot;
- generic file, shell, fetch, and datetime tools are controlled by resolved
  Emelex policy, with `--no-tools` and `--no-web` as authority reductions;
- `--with-web-search` opts into the bounded, approval-gated DuckDuckGo HTML
  provider supplied by the binary; the library continues to require an
  explicit provider and selects no vendor. Each query requires one-shot
  approval. The CLI provider uses a fixed endpoint with no ambient proxy,
  cookies, credentials, or redirects, bounds time and response bytes, and
  records its versioned implementation identity in durable authority; and
- product-specific code indexes, LSP project enrichment, catalog tiers, and
  host prompts do not cross the standalone boundary.

Terminal prompt, multiline editing/navigation, Markdown/reasoning rendering,
tool activity, usage summaries, slash commands, quit aliases, and
non-interactive/JSON behavior remain CLI presentation features over the same
library protocol.

## Consequences

- Embedders can replace terminal UI or approval policy without reimplementing
  the agent protocol.
- `emelex chat` and library-driven agents apply the same tool and history
  invariants.
- Cancellation and output bounds cover queued inference, generation, tool
  execution, and rendering hand-off.
- Security-sensitive actions are explicit even though the harness is local.
- Session replay cannot silently move a conversation to another workspace.
- Configuration edits affect new Sessions only; resume remains reproducible.
- Portable chat UX remains available without importing product-specific
  project semantics.
- Restart recovery never silently repeats a host-side tool effect.
- Failure after a tool checkpoint preserves that complete batch in model
  history; failure before invocation preserves no proposed batch.
