# Configuration context

`Config` is a fully resolved immutable invocation snapshot. The agent records
the selected model reference and immutable installed snapshot in its Session
before its first model/tool action, so resume behavior remains auditable.

Optional nested fields use validated patch values. Missing means inherit;
`{ clear = true }` means explicit clear for model, `top_k`, seed, system
prompt, and memory model.

Resolved generation tokens never exceed the context ceiling. Thinking `auto`
inherits an explicit client default; without one, Emelex safely supplies
`enable_thinking = false`.

`Config::validate` is the single public validator for file-resolved and
caller-mutated snapshots. Any new field or cross-field invariant belongs
there; CLI overlays must call it rather than duplicate a subset.

`memory.recall_bytes` is an exact serialized-JSON byte ceiling. Agent shell
timeout validation reuses `agent::MAX_SHELL_TIMEOUT_SECONDS` (1,200 seconds),
while its built-in default remains 120 seconds, so a validated configuration
cannot fail later at tool construction.

Tool enablement is not approval. Enabling shell only registers it; each use
still crosses the approval boundary. Project configuration is untrusted
repository input, so tool booleans combine with global policy using logical
AND and numeric authority/resource limits can only decrease. Project files
cannot select models/seeds/prompts; forbidden fields fail parsing. Config
descriptors use nonblocking no-follow opens before regular-file validation, so
repository FIFOs cannot stall invocation.

Global default-model mutation is deliberately field-specific. Do not write a
resolved `Config` snapshot back to disk: it may contain project reductions and
built-in defaults that were never selected globally.
