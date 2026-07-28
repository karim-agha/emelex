# Hugging Face Hub

The Hub client explores model repositories accessible to the current caller.
Repository IDs follow Hugging Face's full one- or two-component catalog
grammar: `repo_name` or `namespace/repo_name`.
Library constructors are deterministically anonymous unless given explicit,
redacted `HubCredentials`; the library never reads environment variables.
The CLI alone maps the standard environment-only `HF_TOKEN` into credentials.
Each token is installed only as a secret-sensitive `Authorization` header and
is never persisted or logged. Transport errors discard their request and
redirect URL before crossing the public error boundary, so signed download
query credentials cannot appear through error display, debug, or source
chains. HTTP error bodies are suppressed for authenticated requests,
cross-origin redirects, and final URLs with queries; only anonymous,
same-origin, query-free API errors retain a bounded body. Automatic `Referer`
generation is disabled, and clients configured with an HTTPS Hub origin refuse
redirect downgrades to HTTP.

Search preserves Hub rank, preflights candidates with bounded concurrency,
returns only models matching validated remote-evidence trait filters, and
retains candidate-local diagnostics. Candidate incompatibility, malformed
candidate metadata, and missing or gone repositories do not hide unrelated
models; transport, authentication, rate-limit, and server failures abort the
page instead of masquerading as an empty result. Its opaque composite cursor
carries a bounded printable upstream cursor plus an intra-page offset and is
scoped to normalized query, filters, a domain-separated credential
fingerprint, and the client's exact workload/Metal-budget fit profile, so
locally filtered results beyond the first 20 remain reachable without crossing
credential or fit scopes. Neither token nor fingerprint appears in the cursor.
Cursor schema v2 rejects older opaque cursors rather than resuming them under
incomplete scope.

Remote search distinguishes metadata-advertised MTP from installed runtime
verification. Layout validation is an internal part of runtime verification,
not a public MTP state. Unsupported remote predicates, including structured
output, output media, video, unknown extensions, and verified MTP, fail with
guidance to `emelex hub capabilities`. The two Hugging Face advertised-input
extension filters shown by that command are supported.
The library exports the same complete presentation contract as
`REMOTE_FILTERS`, whose `RemoteFilterHelp` rows drive the CLI and carry
validator-checked examples. Help rows are non-exhaustive, output-only records
so the catalog can gain presentation metadata without a semver-major break.
Advertised input direction comes only from exact
directional Hub pipeline metadata; output-only or generic media tags do not
become input claims. Every advertised-media fact records Hub evidence and
confidence.

`HubClient::new` is static-only: it validates exact-revision architecture,
configuration, template, and checkpoint-plan evidence without initializing
Metal or claiming machine fit. `HubClient::with_fit_profile` additionally
evaluates one workload against one Metal working-set budget.
Remote tool-use and reasoning claims execute the exact-revision template in a
fuel-limited, output-bounded semantic probe after a successful baseline chat
render. Varied declarations/schemas and ordered call/argument/result rounds
must survive control-vs-synthetic renders. Reasoning-history preservation and
thinking-toggle behavior are recorded separately; their broad reasoning trait
is only a discovery union. Keywords in comments, dead branches, or plain text
are ignored.

Downloads pin a full commit SHA, reject alternate indexes, adapters, extra
weights, and ambiguous variants, then select an explicit root-level runtime
set. Transfers resume into owner-only staging files, enforce sizes and LFS
digests, compute SHA-256 for every file, and leave publication to the model
manager. The controlled API supports fallible observers and cooperative
cancellation through transfer, hashing, and retry waits. Hashing uses async
chunk reads owned by the calling future, so dropping a download cannot leave a
detached multi-gigabyte hash worker running. Header, error-body, and successful
body stalls share one deterministic per-file idle classification even when
reqwest's inner read timer wins the race with Emelex's cancellation-aware
timer; resumable partial bytes remain staged.
