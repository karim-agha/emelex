# Model

Typed model vocabulary replaces preset tiers.

- `ModelRef` addresses Hugging Face repositories as `repo_name` or
  `namespace/repo_name`, and local copy, move, or external-link imports as
  `local:<name>`.
- `ModelSnapshotId` adds an immutable Hub commit or local runtime-inventory
  digest for durable bindings.
- `ModelTraits` records modality, task, MLX, MTP, namespaced extension facts,
  evidence, and one optional `ModelSizing` aggregate. It has no duplicate
  scalar sizing fields.
- `TraitFilter` supports typed capability, confidence, sizing, context, and MTP
  progression predicates.
- `CompatibilityReport` keeps compatibility separate from capabilities and
  distinguishes static estimates from successful runtime verification.
- `ModelManifest` pins an owned immutable runnable file set and Hub commit.
  External-link records instead bind a canonical caller-owned target and
  require full validation whenever used.
- With optional feature `rig`, `CompletionModel` adapts one loaded Emelex client
  to Rig completion and streaming traits.
- Rig raw completion responses and terminal streaming responses carry optional
  per-call MTP speculation counters. `None` means the call never speculated.

Capability evidence and sizing/default input records are non-exhaustive.
Callers use `TraitEvidence::new` or `Default`; compatibility and fit reports are
output-only. This keeps future report fields semver-compatible.

Static inspection fails closed on missing or unsupported architecture,
quantization, tokenizer, weight layout, attention geometry, or machine fit.
Sliding-window architectures (Laguna, Gemma 3) charge sliding layers only
`min(context, window)` KV tokens, so large checkpoints are not rejected on a
full-context overestimate.
The same architecture-specific fit estimator can select the largest positive
context under an explicit ceiling and Metal working-set budget. Selection uses
a monotonic binary search over total prompt-plus-generation context; it reads
bounded `config.json` once and does not repeatedly parse tokenizer or weight
artifacts. Prompt-cache sizing consumes an exact aggregate token ceiling.
Public static inspection and fixed loads reserve the full context because a
request can enable caching even when the client default is off. Adaptive
maximum-context selection caps the cache pool at 16,384 aggregate tokens and
uses that same capacity for selection, load inspection, and client construction.
An owned immutable checkpoint snapshot first pins the model directory
descriptor, then owns `config.json` bytes and every selected shard descriptor
opened relative to it through MLX materialization and MTP certification. It
binds each shard by descriptor identity, length, validated header, and
complete-file SHA-256, and never reopens a model-owned pathname.
Rename, A→B→A, and whole-directory swaps therefore cannot change the loaded or
certified identity. Each private, unlinked shard clone receives one
complete-file hash before load; after MLX materialization, Emelex cheaply
revalidates that still-open descriptor's identity, length, and header/layout.
An external-link import keeps only its managed record under Emelex Home. Its
canonical target remains caller-owned and mutable. Resolve and load therefore
repeat complete link, inventory, and content validation; load then performs
compatibility checks and runtime loading. A missing or changed target never
inherits the owned-snapshot fast path.
Manifest schema v2 distinguishes managed external links from owned snapshots;
schema v1 owned-snapshot manifests remain readable.
Checkpoint `generation_config.json` values are recorded as evidence; resolved
Emelex configuration and explicit load overrides retain precedence.
Installed tool-use and reasoning traits come from bounded semantic
chat-template renders after a successful baseline render, not source-text
keyword matches. Tool evidence includes declaration schema, call identity and
arguments, and result content. Inert comments, dead branches, and ordinary
prose containing `tools`, `function`, or `reasoning` do not create capability
claims.

Reasoning is intentionally not one invocation capability.
`interaction:reasoning_history` means an explicit span survives a follow-up
turn; `interaction:thinking_toggle` means thinking-on and thinking-off renders
differ. `interaction:reasoning` remains their broad discovery union.
