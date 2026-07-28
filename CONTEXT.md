# Emelex

This repository contains one Cargo package, one public library, and one
`emelex` executable.

The library owns:

- home and strict configuration resolution;
- Hugging Face discovery and immutable model installation;
- model traits, compatibility, fit, and runtime verification;
- native MLX generation plus the optional Rig adapter;
- agent tools and approval policy;
- durable SQLite sessions, compaction, and workspace Knowledge.

Worker inference renews durable queue authority in-task. Queue failures use
bounded retry/backoff, terminal failed state, CLI inspection, and explicit
operator retry rather than silent poison-job loops.

The binary is a thin presentation and orchestration layer over those APIs.
`emelex chat` starts a generic coding-agent session rooted at the exact
invocation directory. Portable sampling controls and generic file, shell,
fetch, datetime, and opt-in web-search tools remain available. It contains no
product-specific project index, LSP enrichment, model tier, or prompt.

See `agents/CONTEXT.md` for domain language and `docs/adr/` for durable design
decisions.
