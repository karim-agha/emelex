# ADR 0003: Private vendored MLX engine

## Status

Accepted on 2026-07-26.

## Context

Emelex needs direct local inference, chat-template rendering, token streaming,
tool-call parsing, prompt caching, media preprocessing, and Apple GPU kernels.
Delegating these to a Python subprocess would make the package dependent on a
separate runtime and service lifecycle. Depending on a young inference crate
would make a load-bearing API and release dependent on that crate's continued
maintenance.

## Decision

Emelex vendors the adopted Rust engine and pinned MLX/mlx-c source tree. The
Rust engine stays private under `src/engine/`; no engine type crosses the
public library boundary. Attribution and upstream licenses remain in the
distribution.

The native build is offline and reproducible from the repository. It embeds
the compiled Metal library in the Rust artifact. Before any MLX call, Emelex
extracts that library atomically beneath the selected Emelex Home and selects
it for the process.

Process-global MLX caches and allocator diagnostics must remain safe across
multiple loaded Emelex Clients. Downstream patches make Metal library/kernel
entries immutable for the Device lifetime, key custom libraries by canonical
`(name, source)` identity, and synchronize allocator state even though each
model's lazy array graph remains confined to its own inference thread. The
immutable custom-library cache can grow with distinct generated sources until
process exit; this is the deliberate cost of keeping unretained Metal command
buffer references valid.

The public boundary consists of Emelex-owned request, response, event, model,
and error types. Local engine changes are documented in the engine module and
must be reconsidered during a deliberate re-vendor.

## Consequences

- Emelex has no Python, git, or path dependency at runtime.
- The library and executable are self-contained apart from macOS system
  frameworks.
- Emelex owns native correctness, ABI safety, and the cost of maintaining its
  engine patches.
- Native code, archive contents, embedded assets, and binary strings are part
  of release review.
- Release Cargo commands run from a temporary neutral directory with an empty
  environment, a config-free Cargo Home linked only to offline caches, pinned
  Apple tools, and source-building explicitly selected for AWS-LC. The release
  gate removes the complete shared Cargo target tree before packaging, so no
  host, transitive, or native object can predate that sanitized environment.
  The release audit rejects every non-system Mach-O dynamic dependency; only
  `/usr/lib/` and `/System/Library/` load paths are allowed.
- MLX state remains owned by one dedicated inference thread per loaded model.
