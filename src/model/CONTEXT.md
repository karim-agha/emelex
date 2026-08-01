# Model — context

## Invariants

- Hub identity follows the catalog's `repo_name` or
  `namespace/repo_name` grammar; local copy, move, and external-link imports
  always use `local:<name>`.
- Hub revisions are full 40-character commit SHAs.
- Durable bindings use exact `ModelSnapshotId` values; stable references are
  never sufficient for session identity.
- Capability facts never imply machine compatibility.
- Extensible evidence, sizing, generation-default, fit, and compatibility
  records are non-exhaustive. Caller-created records retain constructor or
  `Default` paths; reports remain output-only.
- Chat-template source words are not capability evidence. Chat requires a
  successful baseline render. Installed tool use requires bounded renders that
  preserve declaration name/schema, call name/arguments, and result content.
  Reasoning history and thinking control are independent traits:
  `interaction:reasoning_history` requires preserved follow-up history;
  `interaction:thinking_toggle` requires a runtime-recognized enabled span and
  no pending span when disabled. `interaction:reasoning` is their discovery
  union, not a sufficient invocation requirement.
- Tool protocol is resolved from the exact selected template, never model-type
  naming. Emelex enables tools only when exactly one supported parser completes
  declaration, call, result, and parser round trips. Missing, malformed, or
  ambiguous evidence fails closed to ordinary chat. Gemma-native history is
  normalized into ordered assistant `tool_responses`; orphaned, reordered,
  duplicate, non-text, or mismatched results are rejected.
- Static metadata yields `Estimated`; only successful runtime load/probe yields
  `Verified`.
- Whitelisting a `model_type` and giving it an engine preflight arm are one
  atomic change: local inspection treats preflight failure as a hard error for
  whitelisted types, so a missing arm makes checkpoints uninspectable rather
  than incompatible. A table test pins every whitelisted type to an arm.
- Fit defaults to batch 1 and 16,384 total context tokens.
- Required residency is exact selected weights plus live KV/recurrent state,
  the configured aggregate prompt-cache capacity, the MLX freed-buffer cache,
  persistent activations, and `max(512 MiB, 10% of weights)`. Prompt-cache KV
  follows the exact token capacity while recurrent snapshots retain the maximum
  entry count.
- General inspection and fixed loads reserve full-context prompt-cache
  capacity. Maximum-context loads reserve `min(context, 16,384)` because their
  client cache is explicitly capped there. The reservation is independent of
  the default cache toggle because a request may re-enable caching.
- A managed load repeats compatibility inspection with the same exact cache
  capacity used by selection and client construction.
- Maximum-context selection uses this same non-decreasing residency model and
  returns the largest positive context whose required bytes do not exceed the
  Metal budget. It never extrapolates one manifest residency sample.
- MTP is either metadata-advertised or runtime-verified; layout checks are part
  of runtime verification, not a separately addressable support level.
- Manifests describe Emelex-owned immutable snapshots. Managed external-link
  records describe canonical caller-owned targets separately.
- Manifest schema v2 represents both ownership modes. Schema v1 owned-snapshot
  manifests remain readable.
- Owned snapshot loading is self-contained and offline. Linked loading uses no
  network but depends on the external target, which is mutable and may be
  unavailable. Resolve revalidates link identity, runtime inventory, and full
  content hashes; load repeats that work before compatibility checks and
  runtime loading.
- The entire sizing aggregate is optional. `ModelTraits::sizing == None` means
  no sizing evidence; individual `ModelSizing` fields may also be unknown.
  Numeric filters fail closed for every missing value.
- Published manifests require internally consistent selected-weight,
  evaluated-context, and estimated-residency sizing evidence.
- Unindexed safetensors, alternate indexes, adapters, and ambiguous variants
  are rejected rather than guessed.
- A checkpoint snapshot pins one model-directory descriptor, then owns the
  exact `config.json` bytes and selected shard descriptors opened relative to
  it and consumed by model construction, MLX loading, and MTP certification.
  It never reopens those model-owned paths. Each selected shard receives one
  complete-file digest before load. After MLX materialization, Emelex
  revalidates the still-open descriptor identity, length, and checkpoint
  layout without a second full-file hash.
- Removing a managed external link removes only the Emelex record and never
  mutates the caller-owned target.

Rig response DTOs remain independent of engine types. Optional speculation
accounting is copied onto the response produced by that exact call; no
client-global snapshot is involved. Generation jobs serialize on the
client-owned inference thread and use bounded streaming backpressure. Dropped
completion futures and streams are checked before queued inference starts and
cooperatively during decoding.
- `Task::Translation` is granted from the same template probe as
  `Task::Chat` but independently: translation-only templates yield
  `{Translation}` without `Chat`, and static compatibility accepts either
  task as the conversational-surface requirement. `task:translation` is
  part of the closed filter vocabulary and the remote catalog.
