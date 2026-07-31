# Configuration context

`Config` is a fully resolved immutable invocation snapshot for requested
generation behavior, authority, and resource fallbacks. The agent records the
selected model reference and exact content-addressed installed model in its
Session before its first model/tool action, so resume behavior remains
auditable. Chat's maximum-fit context policy is process-local capacity
resolution: `inference.context_tokens` remains its safe fallback when model
sizing is incomplete, while the loaded Client and attended header expose the
effective model- and machine-bounded context.

Optional nested fields use validated patch values. Missing means inherit;
`{ clear = true }` means explicit clear for model, `top_k`, seed, system
prompt, and memory model.

Resolved generation tokens never exceed the configured fallback context, then
remain capped by the effective loaded context. Thinking `auto` inherits an
explicit client default; without one, Emelex safely supplies
`enable_thinking = false`. Attended chat deliberately supplies thinking-on as
that client default unless the Session explicitly selected off.

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

`[hub].token` is optional secret state accepted only from global
`<home>/config.toml`. Project presence is a hard error. Resolution extracts it
outside the public immutable `Config` snapshot, so Session/config
serialization cannot reveal it.

Facade credential precedence is explicit library credentials, then stored
global credentials, then anonymous. The library does not read `HF_TOKEN`.
CLI presence of `HF_TOKEN` overrides storage; nonempty selects that token and
empty deliberately disables authentication.

Hub auth mutation is global and field-specific. Login takes either a hidden
terminal prompt or bounded one-line UTF-8 stdin, never argv. Status exposes
only effective source. Logout clears storage without claiming to neutralize a
present environment override.

Global default-model mutation is deliberately field-specific. Do not write a
resolved `Config` snapshot back to disk: it may contain project reductions and
built-in defaults that were never selected globally. The same rule applies to
the extracted Hub secret.
