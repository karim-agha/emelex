# ADR 0002: Model references and evidence-backed traits

## Status

Accepted on 2026-07-26.

## Context

A fixed list of model tiers such as “utility”, “workhorse”, or “reasoning”
cannot represent the Hugging Face catalog. Those names mix policy, workload,
hardware fit, and capability into one unstable label. They also prevent a
caller from asking precise questions such as whether a checkpoint supports
images, tool calls, MLX layouts, or a verified MTP implementation.

Remote metadata is useful for discovery but is not proof that a checkpoint can
load or run correctly.

## Decision

A model is addressed by a stable `ModelRef`:

- `repo_name` or `namespace/repo_name` for a Hugging Face repository
  accessible anonymously or through explicit credentials;
- `local:<name>` for an Emelex-owned local import.

Hub installs resolve and record an immutable repository commit. Installed
snapshots are Emelex-owned, manifest-verified, and immutable.

Durable bindings use `ModelSnapshotId`, not a stable `ModelRef`:

- `<repo-id>@<full-commit>` for an installed Hub revision;
- `local:<name>@<sha256>` for a local import, where the digest covers sorted
  runtime-file paths, sizes, and content hashes.

Capabilities are represented as typed `ModelTraits`, not tiers. Core traits
cover modalities, tasks, MLX compatibility, MTP state, and one optional
`ModelSizing` aggregate for weight bytes, residency estimates, evaluated
context, and architecture limits. Namespaced extension facts permit new
capabilities without redefining model identity.

Reasoning is not one invocation capability. Exact template probes record
`interaction:reasoning_history` and `interaction:thinking_toggle` separately.
The broad `interaction:reasoning` trait is their union for discovery only.
Raw thinking-on generation requires toggle support; an agent or chat turn also
requires history preservation so later rounds cannot silently lose reasoning.

Trait filters parse into typed predicates. They support boolean capabilities,
minimum evidence confidence, MTP state, and numeric sizing/context bounds.
Unknown sizing remains `None` and fails numeric predicates closed; it is never
represented as a real zero.

Every nontrivial claim records its evidence source and confidence. The
important progression is:

1. advertised by repository metadata;
2. inferred from static configuration, tokenizer, tree, or weight headers;
3. verified by loading or executing the exact local snapshot.

Compatibility and fit fail closed when required evidence is missing, an
architecture or layout is unsupported, or estimated residency exceeds the
selected Metal budget. Search preserves Hugging Face ranking while filtering
by explicit traits. A static-only Hub client reports no fit. A profiled client
reports fit for one exact workload and Metal budget.

Hub discovery uses only remote evidence. It can filter metadata-advertised MTP
but rejects installed-only runtime-verification claims. Layout validation is
an internal prerequisite of runtime verification, not a third public MTP
support state.
Library clients are anonymously deterministic unless given explicit, redacted
credentials. The original environment-only persistence decision in this
paragraph is superseded by
[ADR 0008](0008-hub-credential-precedence.md). Hugging Face decides which
private or gated repositories a token can access; Emelex marks authorization
headers sensitive and never logs the token.

The original assumption above that every local model is Emelex-owned is
superseded by [ADR 0007](0007-local-model-import-ownership.md), which adds an
explicit managed external-link mode.

## Consequences

- Callers can explore the full accessible catalog with composable filters.
- A model may gain new evidence without being moved between product-defined
  tiers.
- “MTP advertised” and “MTP runtime verified” remain distinct facts.
- Defaults choose a model reference; they do not create a hidden selection
  policy.
- Static discovery never masquerades as runtime certification.
- Stable references may advance; sessions remain bound to the exact snapshot
  that produced their history.
- Checkpoint-advertised generation defaults are evidence below resolved Emelex
  configuration and explicit per-load overrides.
