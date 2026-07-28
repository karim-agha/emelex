# bin — context

## Invariants

- `--home` is global and has precedence over `EMELEX_HOME`. Clap does not read
  the environment for this flag; when the flag is absent, the shared library
  resolver handles `EMELEX_HOME`, including empty-as-unset semantics.
- Help and version output do not initialize MLX.
- Commands render library results; durable behavior lives outside this module.
- Stdout and stderr failures propagate without committing successful agent
  presentation state; broken pipes terminate quietly.
- One-shot raw Ctrl-C closes stream backpressure, requests cancellation, and
  awaits inference-job completion. One-shot agent Ctrl-C sets the shared
  cancellation handle and awaits turn cleanup, so shell process groups are
  killed and reaped before exit. Both human paths flush buffered Markdown and
  terminal style state before returning cancellation.
- Untrusted human-facing text cannot inject terminal controls. JSON preserves
  the original typed data through JSON escaping. JSON chat emits a `session`
  envelope first, before recovery and agent events, with `session_id`, immutable
  `model_snapshot`, and whether the Session was `resumed`.
- Non-interactive `chat` and `resume` resolve their prompt from the optional
  positional argument or bounded UTF-8 stdin before validating command mode,
  so human and `--json` invocations share one input contract.
- Top-level `resume` takes an explicit Session only through `--session`.
  `chat --resume=SESSION` requires `=`; bare `chat --resume` means newest, so a
  following positional value remains the prompt.
- Interactive approval owns one nonblocking `/dev/tty` descriptor inside its
  future. Cancellation drops and closes it; no background terminal reader may
  outlive the approval call.
- `shell`, `web_search`, and `web_fetch` approval arguments must fit as complete
  canonical JSON within the 2,048-character preview. Oversized actions are
  automatically denied and must be split. Other tool previews preserve bounded
  head and tail content, state the exact omitted-character count, neutralize
  terminal controls, and retain a complete SHA-256.
- Agent prompts do not embed the invocation pathname. The harness enforces the
  invocation directory through its opened root descriptor and recorded
  device/inode identity. Prompts describe the workspace-first boundary:
  likely-secret reads, outside-root paths, all file mutations, and shell
  invocations require one-shot approval; displayed path labels remain
  untrusted data.
- `-C`/`--directory` and its visible `--root` alias select the invocation root
  for workspace-scoped Sessions, Knowledge, tools, and project configuration,
  without claiming to change process cwd or unrelated relative-path
  resolution.
- New-chat sampling/thinking overrides are validated through public
  `Config::validate` and become immutable Session semantics. Explicit
  `--thinking auto` inherits resolved configuration. Resume rejects semantic
  overrides. Per-request max tokens use the loaded `Client` ceiling, so stored
  semantics cannot re-expand a checkpoint-clamped load policy.
- CLI web search is explicit, approval-gated, bounded, credential/proxy-free,
  and identified solely by its provider implementation. A policy-disabled
  explicit request fails rather than silently omitting the tool. Known
  interactive provider challenges fail visibly instead of masquerading as an
  empty result set.
- Human color capability is detected separately for stdout and stderr.
- Hub download Ctrl-C handling runs in an independent watcher task, so the
  atomic cancellation flag changes even while the download future executes a
  synchronous local phase on another runtime worker.
- Hub auth login reads either a hidden terminal prompt or, with
  `--token-stdin`, one bounded UTF-8 stdin line. Token argv is forbidden.
  Status reveals only effective credential source. Logout clears stored global
  state but does not override a present environment token.
- At the CLI boundary, present `HF_TOKEN` has three-state semantics: nonempty
  overrides with that token, empty explicitly disables authentication, and
  absence permits stored global credentials.
- `hub capabilities` renders the library's complete `REMOTE_FILTERS` catalog;
  its displayed syntax and accepted remote predicates cannot drift apart.
- Every CLI Hub search enables `HubSearch::mlx_library` independently of user
  text and uses the model-manager client's local Metal/storage fit. Human
  results render as labeled vertical cards: local status comes from one
  verified inventory pass and distinguishes the exact revision, a different
  installed revision, and no installed revision. Quantization comes only from
  validated exact-revision config. The memory row names its evaluated workload
  separately from the model's maximum context, capability groups wrap with
  hanging indentation, and the MLX-only search does not repeat a runtime row.
  A human stdin/stdout/stderr terminal turns those same results into one
  stdout-owned inline viewport; it must never append a second selector list.
  The selected card carries a visible rail, frames fit the current terminal
  height, reserve one cursor row, and stay below its wrap column. Enter
  revalidates and downloads that displayed revision; Escape or `q` exits
  successfully; raw Ctrl-C becomes the explicit interrupt action. Buffered
  redraws recompute prior-frame height at the current terminal width, preserve
  scrollback, and restore cursor visibility on every exit path.
  Redirected or JSON searches never read selection input. Default diagnostics
  show only a count, verbose diagnostics group by sanitized candidate ID, and
  JSON stays complete.
- Human TTY downloads start in a truthful preparing phase, consume exact
  transfer lifecycle totals, coalesce state into an independently animated live
  region, and remain in finalizing after transfer until certification and
  publication return. Resumed prefixes contribute to completion but not
  throughput. Redirected output is deterministic and excludes chunk-level
  progress; existing JSON event records remain byte-compatible.
- Preferred local import uses singular `model import PATH`, with an optional
  `--name`; other lifecycle commands remain plural `models`. Import defaults to
  an owned immutable copy. Move publishes before selectively retiring only
  unchanged selected source files; changed or unselected files remain and
  cleanup warnings do not hide the committed install.
- Symlink import stores a managed record pointing to one canonical external
  target. Every resolve/load revalidates its link, runtime inventory, and full
  hashes; load also checks compatibility and opens the runtime. Removal
  deletes only the record.
- Attachment UX advertises only formats decoded by the embedded runtime.
- Media onboarding maps runtime image/audio requirements to remote
  advertised-input evidence, labels that evidence as provisional, and requires
  downloaded models to pass local certification before selection.
- Zero-model onboarding presents one bounded candidate page at a time across
  user-driven Hub cursors. Empty-page messages must distinguish one page from
  catalog exhaustion, and local certification failure must leave next-page
  discovery available.
- Raw thinking-on selection requires `interaction:thinking_toggle`. Agent and
  chat thinking-on selection require both that trait and
  `interaction:reasoning_history`. A generate-command override reaches model
  selection, load policy, and request policy as one resolved value.
- `doctor` records every independent facet result before returning aggregate
  failure; corrupt model entries do not hide healthy snapshots.
- Memory-model generation renews its durable worker lease every 60 seconds from
  the same async task. No detached heartbeat can race completion or survive
  cancellation.
- Empty `memory work` returns before model selection/load. Nonempty work starts
  with compaction and alternates queues. Source/output budgets use the loaded
  Client's effective checkpoint-clamped ceilings and conservative byte
  accounting.
- Interactive Session execution preempts a claimed maintenance completion.
  The worker releases that job to pending without incrementing failure count.
- Retryable worker failures are persisted with bounded diagnostics and
  backoff. `memory status` reports terminal counts, `memory failures` inspects
  them, and `memory retry JOB` is the only operator reset.
- Chat close queues distillation and releases its Session lease only. It never
  runs post-chat model inference; `memory work` is the explicit worker action.
- Session deletion is logical durable-store cleanup, not secure erasure.
  SQLite free pages or WAL bytes can persist until `memory gc`, and the global
  interactive `cache/prompt_history` is intentionally outside Session scope.
- `/compact` only queues while the chat lease is live. Exit chat, run
  `emelex memory work`, then resume so compaction can complete and replay can
  install the verified Summary.
