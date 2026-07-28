# ADR 0001: Standalone product and storage boundary

## Status

Accepted on 2026-07-26.

## Context

Emelex is useful independently of any application that embeds it. Keeping its
runtime, model catalog, agent harness, and persistence inside another product
would couple public names, storage paths, release cadence, and behavior to that
host.

The same guarantees must apply when Emelex is used through its Rust API and
through its command-line program. A library call must not silently choose a
different home or configuration hierarchy than an equivalent CLI invocation.

## Decision

Emelex is one standalone Cargo package that publishes:

- the `emelex` Rust library;
- the `emelex` executable; and
- no product-specific integration layer.

The sole default root for Emelex-owned state is `~/.emelex`. Home selection
uses this precedence:

1. an explicit library builder or CLI `--home` value;
2. `EMELEX_HOME`;
3. `~/.emelex`.

Configuration, immutable model snapshots, caches, extracted native runtime
assets, durable memory, sessions, and temporary files all remain below that
root. Project configuration may be read from the nearest Git worktree's
`.emelex.toml`, but it does not move owned storage into the project. The exact
invocation directory remains the default agent workspace.

The library contains policy and behavior. The binary is a presentation and
orchestration layer over public library APIs.

## Consequences

- Embedders and CLI users share one model store and one persistence contract.
- Tests can select an isolated home without changing process-wide `HOME`.
- Project repositories receive no Emelex-owned cache or memory files.
- Product-specific concepts, paths, prompts, catalog tiers, and configuration
  keys are defects in this repository.
- Moving or deleting an Emelex Home has a complete and understandable blast
  radius.
