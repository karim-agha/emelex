# Configuration

Configuration is strict TOML. Unknown keys and invalid bounds fail before any
model load.

Layers:

1. built-in defaults;
2. `<home>/config.toml`;
3. `.emelex.toml` at the nearest Git worktree root.

CLI/request overrides apply after this resolved snapshot. `--no-project-config`
skips layer 3.

Project files are untrusted repository input. They may disable tools and lower
resource/network limits, never re-enable or raise values denied by global
configuration. Model selection, deterministic seeds, system prompts, memory
models, and enabling thinking remain global/CLI authority; putting those fields
in `.emelex.toml` is a hard error, not a silent no-op. Hub credentials are also
global-only: optional `[hub].token` is accepted in `<home>/config.toml` and
forbidden in `.emelex.toml`. No configuration field can redirect the Emelex
home/database or grant approvals.

The global Hub token is extracted as secret invocation state rather than
stored in resolved `Config`. Explicit `HubCredentials` supplied to
`EmelexBuilder` override it; without either, facade Hub access is anonymous.
The library never reads `HF_TOKEN`. At the CLI boundary, a present `HF_TOKEN`
overrides global storage: a nonempty value authenticates and an empty value
explicitly selects anonymous access. When the variable is absent, the stored
token may be used.

`emelex hub auth login` reads a token from a hidden terminal prompt.
`emelex hub auth login --token-stdin` instead accepts one bounded UTF-8 line
from stdin; tokens are never command-line arguments. `status` reports only the
effective credential source, never the value. `logout` clears the stored token;
an active `HF_TOKEN` environment override can still authenticate the current
invocation.

Resolved `inference.max_tokens` must not exceed
`inference.context_tokens`, including after project reductions merge.
`thinking = "auto"` inherits an explicit client default; without one, Emelex
sends `enable_thinking = false`.

Memory recall is byte-bounded, not estimated as tokens:
`memory.recall_bytes` caps the serialized JSON injected into the system
context. Host-shell timeout defaults to 120 seconds and is limited to the same
hard 1,200-second ceiling enforced by the agent tool.

Nullable values clear explicitly in global configuration with an inline marker:

```toml
default_model = { clear = true }

[inference]
seed = { clear = true }
```

Files are opened nonblocking and must be regular, non-symlink TOML capped at
1 MiB. Repository FIFOs and devices fail before any read. Model references are
validated while loading configuration.

Library callers that create or modify a `Config` snapshot directly call
`Config::validate` before using it. The CLI uses the same public validator
after applying per-Session generation overrides, so command flags cannot
bypass file-loaded bounds or cross-field invariants.

`Config::write_global_default_model` is the narrow mutation used by model
management surfaces. It validates the existing global file, preserves all
other global keys, validates the new resolved global snapshot, and replaces
`config.toml` atomically with an owner-only file. It never reads or serializes
project configuration. Hub authentication uses the same global-only,
field-specific mutation principle: login/logout preserve unrelated global
keys, never merge project configuration, and never route the token through
resolved `Config`.
