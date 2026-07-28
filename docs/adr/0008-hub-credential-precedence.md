# ADR 0008: Hub credential storage and precedence

## Status

Accepted on 2026-07-28. This supersedes the environment-only credential
decision in ADR 0002.

## Context

Private and gated Hugging Face repositories require a token. Requiring an
environment variable for every CLI invocation is inconvenient, while placing
credentials in project configuration would let an untrusted repository choose
authentication state. A secret also must not leak into the ordinary resolved
`Config` snapshot, durable Session semantics, diagnostics, or logs.

Library callers and the CLI need deterministic precedence so a deliberately
anonymous invocation cannot accidentally fall back to a stored credential.

## Decision

Global `<home>/config.toml` may contain an optional `[hub].token`. Project
`.emelex.toml` files may not contain the token; its presence is a hard
configuration error. `emelex hub auth` is the CLI surface for managing the
global credential.

`emelex hub auth login` reads a token through a hidden terminal prompt.
`emelex hub auth login --token-stdin` instead reads one bounded UTF-8 line from
standard input. No auth command accepts a token as an argument.
`emelex hub auth status` reports only the effective credential source, never
token material. `emelex hub auth logout` clears the stored token; it does not
claim to disable a present environment override.

The configuration loader extracts the stored token as secret invocation state.
It is not a field of the resolved public `Config` value and is not copied into
Session semantics or serialized through normal configuration output.

For the library facade, explicitly supplied `HubCredentials` take precedence
over the stored global token. Without explicit credentials, the stored token
is used; without either, Hub access is anonymous.

At the CLI boundary, presence of `HF_TOKEN` overrides stored configuration. A
nonempty value supplies that invocation's credentials. An explicitly empty
value disables authentication for that invocation, even when a stored token
exists. When `HF_TOKEN` is absent, the stored token may be used.

Credential values use redacted debug output, secret-sensitive authorization
headers, and URL/body error suppression. The raw stored token exists only in
the owner-only global configuration file; Emelex does not copy it into project
configuration, manifests, memory, diagnostics, or logs.

## Consequences

- CLI users can authenticate once without exporting a token in every shell.
- A repository cannot request or override Hub credentials through project
  configuration.
- Embedders retain per-facade credential authority and can override global
  storage explicitly.
- `HF_TOKEN=` is a reliable one-invocation anonymous override.
- Configuration and diagnostic APIs remain safe to inspect without exposing
  the token.
