# Hugging Face Hub — context

## Invariants

- Direct Hub clients are deterministically anonymous unless their constructor
  receives explicit `HubCredentials`; the library never reads authentication
  environment variables. The `Emelex` facade may instead consume the
  separately extracted global `[hub].token`.
- Facade precedence is explicit credentials, stored global credentials, then
  anonymous. The token is absent from resolved `Config`.
- At the CLI boundary, present `HF_TOKEN` overrides storage. Nonempty supplies
  credentials; empty deliberately disables authentication. Absent allows the
  stored token to apply.
- Hub auth accepts a hidden prompt or bounded one-line UTF-8 stdin, never argv.
  Status reports source only. Logout clears storage even when an environment
  override remains effective.
- Authorization headers are secret-sensitive. Except for intentional
  owner-only global storage, tokens are never persisted elsewhere, returned in
  errors or diagnostics, or logged. Project configuration cannot contain one.
- Redirect policy disables automatic `Referer` generation. A client configured
  with an HTTPS Hub origin permits HTTPS redirects only and refuses downgrade
  destinations.
- Every `reqwest::Error` loses its request/final URL before it enters
  `HubError`, including body-stream failures and error-body diagnostics. This
  prevents signed redirect query credentials from escaping through display,
  debug, or source chains.
- HTTP error bodies are untrusted and may echo request credentials. They are
  suppressed whenever the client is authenticated, the final origin differs
  from the configured Hub origin, or the final URL has a query. Bounded body
  diagnostics remain only for anonymous, same-origin, query-free errors.
- Anonymous discovery skips private and gated candidates. Authenticated
  discovery and inspection may include repositories accessible to that token;
  Hugging Face remains the access-control authority.
- Hub IDs accept exactly `repo_name` or `namespace/repo_name` within the
  catalog's 96-byte bound. Revisions, cursors, pagination origins, and file
  paths also validate.
- Composite search cursors are bounded, opaque, and scoped to normalized
  query, MLX catalog selection, canonical filters, a domain-separated
  credential fingerprint, and the exact optional workload/Metal-budget fit
  profile. Neither token nor fingerprint is encoded directly into the cursor.
  Schema v3 rejects older cursors whose scope omitted catalog or fit inputs.
  Dynamic storage availability stays outside cursor identity; a resumed page
  does not revisit earlier ranks when free space changes. Upstream cursor text
  is treated as bounded printable opaque data and re-encoded through URL
  query-pair APIs.
- Search ranking survives concurrent metadata preflight.
- `HubSearch::mlx_library` maps the website's MLX-library selection to the Hub
  API's `filter=mlx` parameter without modifying optional user search text.
- Every fully inspected `HubModel` carries exact-revision quantization
  configuration parsed from `quantization` or `quantization_config`. Absence
  means not configured, not an inferred floating-point dtype. Configured
  summaries preserve mode, bits, group size, and whether per-layer overrides
  exist; they never claim uniform tensor quantization.
- Profiled CLI searches reject candidates whose exact selected runtime
  transfer plus `max(64 MiB, 5%)` exceeds available Emelex Home filesystem
  space. Search probes availability before accepting ranked page members, so
  rejected candidates do not underfill the page. Download performs the same
  check again against live availability.
- Candidate-local incompatibility, malformed candidate metadata, and HTTP
  missing/gone failures become bounded page diagnostics instead of aborting
  unrelated candidates. Transport, authentication, rate-limit, and server
  failures abort the page.
- A plan is immutable at one full revision and excludes remote code.
- Plans select one unambiguous root-level checkpoint: either exactly
  `model.safetensors`, or every and only shard named by the canonical index.
- Static-only clients do not claim machine fit. Profiled clients use one exact
  workload and Metal budget.
- Remote metadata may advertise MTP. Runtime verification, including its
  internal layout checks, requires an installed snapshot and is unavailable as
  a remote filter.
- `REMOTE_FILTERS` is the complete shared library/CLI help contract for remote
  predicates. Every row carries a validator-checked example; exact capability
  acceptance derives from those rows rather than a parallel allowlist. Its
  help rows are non-exhaustive and output-only.
- Advertised media-input extensions require exact directional Hub metadata.
  Generic or output-only media tags do not qualify. Each accepted extension
  records `HubMetadata` evidence and advertised confidence.
- Exact-revision chat-template capability evidence is semantic and bounded.
  Chat requires a successful baseline render. Tool use requires materially
  rendered declaration name/schema, call name/arguments, and result
  structures plus one unambiguous parser round trip for the exact selected
  template. Failure or ambiguity disables tool use without disabling baseline
  chat. Reasoning history and thinking-toggle proofs remain separate; generic
  reasoning is their discoverability union. Source substrings alone never
  qualify.
- Partial files stay inside Emelex-owned staging and use no-follow opens.
- Download header, error-body, and successful-body read timeouts all normalize
  to `DownloadIdleTimeout`; a reqwest inner timeout must not race into the
  generic request-error surface. Already-persisted partial bytes remain staged.
- Every completed file has an Emelex-computed SHA-256.
- Fallible observers and cancellation are checked during transfer, hashing,
  and retry waits.
- File hashing is async and future-owned. Dropping the download drops its
  reader after at most the in-flight chunk; no detached blocking hash survives.
- Both public Hub download futures own a linked private cancellation child.
  Dropping either future cancels detached pre-commit work without mutating a
  caller token shared by sibling operations.
