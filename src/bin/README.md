# bin

Command-line entry points. `emelex` keeps argument parsing thin and delegates
behavior to the library.
Only an actual `--home` value becomes an explicit builder override. Otherwise
the library resolves `EMELEX_HOME` and then `~/.emelex`, including treating an
empty `EMELEX_HOME` as unset.

Human output uses stream-specific color detection and neutralizes terminal
control characters from model, tool, path, and Hub text. All writes are
fallible, including streamed output; a closed consumer never panics the
process. After successful argument parsing, `--json` emits newline-delimited
success/events on stdout and a structured runtime-command error record on
stderr. For `chat` and `resume`, the first stdout record is always
`{"type":"session","session_id":...,"model_snapshot":...,"resumed":...}`
before recovery or agent events. Clap retains its native human help, version,
and grammar diagnostics. Non-interactive `chat` and `resume` accept a
positional prompt or the same bounded UTF-8 stdin path as `generate`.
`resume [PROMPT] [--session SESSION]` selects the newest workspace Session when
the flag is absent. The chat form uses `--resume` for newest or
`--resume=SESSION` for an explicit target; requiring `=` keeps a following
positional value unambiguously available as the prompt.

Interactive approvals read a bounded canonical line from nonblocking
`/dev/tty` inside the approval future. Cancelling the turn drops that future
and closes its descriptor; no blocking task remains to steal input from the
next chat prompt. The complete canonical JSON arguments for `shell`,
`web_search`, and `web_fetch` must fit the 2,048-character approval preview;
oversized actions are automatically denied with guidance to split the request.
Other tool previews may use bounded head and tail content, an exact omitted
character count, and a complete SHA-256.

The agent prompt names the workspace boundary without embedding its pathname.
The harness opens the invocation directory and enforces its device/inode
identity through a live root descriptor. Filesystem access is workspace-first
rather than workspace-confined: likely-secret reads, outside-workspace paths,
every write or edit, and every shell invocation require explicit one-shot
approval. Any path label returned to the model remains untrusted data.
`-C`/`--directory` (visible `--root` alias) changes this invocation root for
workspace-scoped Sessions, Knowledge, tools, and project configuration, not
the process current directory used by unrelated relative CLI path arguments.

New chats accept bounded max-token, temperature, and thinking overrides and
persist their resolved values in immutable Session semantics. Resume rejects
those flags instead of changing historical meaning. File, shell, fetch, and
datetime capabilities are generic built-ins governed by resolved policy;
`--no-tools` and `--no-web` reduce them. `--with-web-search` explicitly adds a
bounded DuckDuckGo HTML provider when web policy permits it. Search requires
one-shot approval, uses no ambient proxy or credentials, and treats unexpected
third-party markup as zero results. Known anti-bot challenge pages report the
provider as unavailable rather than pretending the query had no matches.
At runtime, request output tokens are capped again by the loaded checkpoint's
effective output ceiling. Immutable requested Session semantics stay
auditable, but cannot re-expand a model-clamped load policy.
Ctrl-C cooperatively cancels one-shot raw generation and waits for its
inference job to leave the model thread. One-shot agent generation likewise
awaits native model-thread completion and tool cleanup, including process-group
kill and reap for an active shell command. Human cancellation flushes buffered
Markdown and resets terminal styling before returning the cancellation error.

`hub capabilities` is the source of truth for explicit remote filters. Every
CLI search also applies Hugging Face's MLX catalog scope and local Metal/storage
fit. Human results use compact multi-line model cards showing exact-revision
download state and validated quantization. Because every result is already
MLX-scoped, cards omit a redundant runtime row and show MTP separately when
advertised. On a human terminal, arrow keys navigate a compact selector for the
current result page and Enter downloads the selected revision; Escape or `q`
leaves the results without downloading. Redirected and `--json` searches never
prompt. Skipped-candidate diagnostics collapse to a count unless `--verbose`
is present, while `--json` retains the complete structured page. Downloads
report files, bounded percentage progress, retries, and verification; Ctrl-C
cooperatively cancels transfer, hashing, or retry waits before returning. An
independently scheduled signal watcher sets the same cancellation flag during
synchronous local inspection/load phases, and a final checkpoint precedes
publication.

`hub auth login` reads the token from a hidden prompt. `--token-stdin` switches
to one bounded UTF-8 line on stdin; `--json` login requires this non-interactive
form. No command accepts a token through argv.
`hub auth status` prints only the effective source, never token material.
`hub auth logout` clears global storage, but a present `HF_TOKEN` still wins.
At the CLI boundary, nonempty `HF_TOKEN` overrides the stored token and empty
`HF_TOKEN` explicitly disables authentication for one invocation.

Preferred import uses singular `model import PATH`; its name defaults from the
canonical directory and `--name NAME` overrides it. It copies an owned
immutable snapshot by default. `--move` publishes first, then retires only
unchanged selected source files; extra or changed files remain and produce a
truthful cleanup warning. `--symlink` creates a managed record for a canonical
caller-owned external target. Resolve/load revalidate its link, runtime
inventory, and full hashes; load also checks compatibility and opens the
runtime. Removal deletes only the record. Other lifecycle operations
remain under `models`; `models remove` requires the exact snapshot ID printed
by `models list`.

Attachments are descriptor-stable and bounded. The self-contained CLI accepts
JPEG, PNG, WebP, and PCM16/float32 WAV input. Encoded video and formats that
would require an ambient codec executable fail before model loading.
When no installed media-capable model exists, onboarding searches
Hugging Face using advertised image/audio metadata. The selection text calls
that evidence a discovery hint; the downloaded checkpoint must still pass
local runtime capability certification before use. Onboarding presents one
bounded result page at a time while the user follows opaque Hub cursors, offers
the next page explicitly, and can continue after a downloaded candidate fails
certification.

Model selection uses invocation-specific capabilities. Raw generation with
thinking enabled requires `interaction:thinking_toggle`; agent generation and
chat additionally require `interaction:reasoning_history`, because later
rounds must preserve prior reasoning. One-shot `--thinking` overrides are
resolved once and passed consistently to selection, model loading, and the
request. Chat `--thinking auto` keeps resolved configuration, matching
one-shot generation semantics.

Closing chat queues idempotent distillation and returns without loading or
running a memory model. `/compact` likewise only queues deferred work because
the live chat lease excludes the compaction worker. Exit chat, run
`emelex memory work`, then resume the Session to use its verified Summary.
`memory sessions delete` removes the durable Session and unreferenced assets,
but is not a secure-erasure promise: SQLite free pages/WAL bytes can remain
until maintenance, and the separate interactive `cache/prompt_history` file is
not scoped to one Session. `memory gc` checkpoints and vacuums SQLite; deleting
the prompt-history file is a separate explicit filesystem action.
`memory work` returns a zero report without selecting or loading a model when
no jobs exist. Otherwise it processes bounded work in compaction-first,
alternating order. Post-load generation and source budgets use the Client's
effective checkpoint-clamped context/output ceilings and a conservative
byte-based source estimate.
During arbitrarily long local generation, an in-task 60-second heartbeat
renews the five-minute durable lease; completion, cancellation, and errors drop
that timer before any state transition. Retryable failures use durable backoff
and terminal failures remain visible through `memory status` and
`memory failures`. A chat Session that becomes live after worker claim
preempts maintenance; the job returns to pending without consuming retry
budget. `memory retry JOB` is the explicit operator reset.
