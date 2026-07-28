# Emelex context

Emelex is one standalone Rust package exposing both a library and the `emelex`
binary. It is a local-inference toolkit and agent harness for Apple Silicon,
powered by a private, patched MLX engine.

Core language:

- **Emelex Home**: sole root for Emelex-owned config, models, caches, runtime
  assets, memory, sessions, and temporary files.
- **Model Reference**: stable address. Hub models use `repo_name` or
  `namespace/repo_name`; imported models use `local:<name>`.
- **Model Snapshot ID**: immutable address. Hub snapshots append a full commit;
  local snapshots append a digest of the exact runtime-file inventory.
- **Model Traits**: typed capabilities plus namespaced extension facts and
  evidence.
- **Compatibility Report**: static engine/layout/fit decision. Runtime
  verification is recorded separately.
- **Installed Model**: immutable, hash- and runtime-verified model snapshot
  pinned to a `ModelSnapshotId`.
- **Loaded Model**: one checkpoint owned by one dedicated inference thread.
- **Agent**: reusable tool-loop harness built on a loaded model.
- **Session**: ordered, durable or in-memory conversation event stream.
- **Knowledge**: versioned workspace-scoped memory distilled from sessions.

Default home is `~/.emelex`. Precedence is explicit API or CLI `--home`, then
`EMELEX_HOME`, then that default. Project configuration lives at
`.emelex.toml` in the Git worktree root. Tool root remains the invocation
directory unless explicitly changed.

Hub discovery covers Hugging Face repositories visible to the client:
public and ungated when anonymous, plus repositories accessible to explicit
credentials when authenticated. A static-only Hub client checks exact-revision
architecture and file-plan compatibility without initializing Metal or
claiming machine fit. Fit-aware discovery uses an explicit workload and Metal
working-set budget. Unknown architectures, missing `model_type`, unsupported
or ambiguous layouts, and profiled models that exceed the budget fail closed.
CLI Hub discovery also applies Hugging Face's MLX catalog filter and removes
downloads that do not fit currently available Emelex Home filesystem space.

Security boundaries are approval-driven, not a sandbox guarantee. Shell runs
`/bin/sh -c` on the host after approval. Web access intentionally permits any
HTTP(S) target without ambient cookies or credentials. Project instructions
and recalled Knowledge may influence model tool choices. These risks must stay
visible in docs and prompts.
