# ADR 0006: Lazy invocation facets

## Status

Accepted on 2026-07-27.

## Context

Emelex is both a library and a command-line toolkit. Callers may need only
configuration resolution, Hub discovery, installed-model inspection, durable
memory, or inference. Constructing the common facade must not activate
unrelated native or persistent subsystems. In particular, a configuration or
Hub-only caller should not open SQLite, create a Metal device, extract runtime
assets, or initialize MLX.

Those subsystems can fail independently. Hiding their failures behind eager
facade construction makes `emelex doctor` fail at the first unavailable
component instead of reporting every independent check.

## Decision

`EmelexBuilder::build` performs only the shared invocation work:

- resolve and prepare Emelex Home;
- canonicalize the invocation root; and
- load one strict configuration snapshot.

Hub policy, Metal fit budget, installed-model management, and durable memory
are fallible lazy facets. Their accessors initialize each facet at most once
per `Emelex` value and return a `Result`. Model management may depend on the
same facade's Metal budget and creates a fit-profiled Hub client with the
resolved inference workload. The Hub-only facet remains static-only and does
not initialize Metal. Memory remains independent.

Network requests still occur only through explicit Hub operations. Runtime
asset extraction and MLX initialization still occur only through explicit
model load, model verification, or runtime diagnostics.

The CLI activates only the facets required by the selected command. `doctor`
requests checks independently, preserves every result, and reports an
aggregate failure after rendering the complete diagnostic report.

## Consequences

- Library construction has no hidden SQLite or Metal-device side effects.
- Hub-only and memory-only embeddings do not inherit each other's failure
  modes.
- Facet access is explicitly fallible in the public API.
- `Emelex::hub()` makes no machine-fit claim; fit-aware CLI paths use
  `Emelex::models()?.hub()`.
- A facet initialization failure can be corrected and retried without
  rebuilding unrelated state.
- Diagnostics can distinguish home, configuration, Hub, Metal, memory,
  runtime, and model failures.
