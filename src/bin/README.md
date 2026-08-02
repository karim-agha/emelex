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
Chat requests the largest model-declared context that fits the active machine
budget. When exact weights and a declared maximum are available, this adaptive
load replaces the fixed configured context for new and resumed chat processes
without rewriting immutable Session semantics. Incomplete sizing metadata keeps
the resolved configured context instead of guessing. The attended header labels
the effective context as `machine-fit` or `configured fallback` accordingly.

Attended chat draws an immediate animated status on stderr after a message is
submitted. It remains active across prompt checking, prefill, reasoning, answer
streaming, tool execution, durable tool-result recording, and final response
persistence. The live line reports exact cumulative prompt, cache, and output
tokens from native progress events, the current context reservation while it is
checked, and average decode tokens per second once two output tokens exist. It
is cleared only when persistent answer/reasoning lines or event reports will be
written. Attended terminal Markdown holds the unfinished raw line suffix,
flushes each complete line before an immediate redraw, and flushes the final
suffix when the stream changes or ends. Between line flushes, a bounded,
terminal-neutral live preview shows the suffix above coalesced 120 ms telemetry
updates. The terminal cursor therefore stays at a safe line boundary while
token counts and speed continue animating during ordinary no-newline decode.
Redirected answer stdout and all non-interactive rendering remain immediate;
reasoning remains line-buffered only when its stderr live region is active. The
terminal usage footer remains authoritative. JSON and non-interactive chat
never emit the live region. Rustyline prefers the attended terminal, so
redirected stdout receives assistant output rather than prompt redraws. A
human-rendered turn failure is not printed a second time by either the
interactive input loop or the top-level one-shot error reporter;
checkpoint-side-effect warnings remain separate.
Shift-Return inserts a newline when the terminal preserves the modifier;
Rustyline-decodable LF and Alt-Return variants provide the same operation while
plain Return submits. Multiline message bytes, including authored outer
whitespace, reach the agent unchanged. Slash commands remain single-line.

Interactive `/tools` opens an arrow-and-checkbox execution selector for tools
already installed in the Session's immutable authority. Space toggles, Enter
applies, Escape cancels, and Ctrl-C exits chat without applying. Its
process-local selection applies to future turns until chat exits, can restore a
previously deselected authorized tool, and can never add a tool excluded when
the Session was built. Historical declarations may remain model-visible only
when replay protocol requires them; their deselected execution remains denied.
Selection does not bypass one-shot approval. Fresh resume begins with every
snapshotted tool enabled; the durable authority snapshot itself never changes.
Ctrl-C cooperatively cancels one-shot raw generation and waits for its
inference job to leave the model thread. One-shot agent generation likewise
awaits native model-thread completion and tool cleanup, including process-group
kill and reap for an active shell command. Human cancellation flushes buffered
Markdown and resets terminal styling before returning the cancellation error.

`hub capabilities` is the source of truth for explicit remote filters. Every
CLI search also requires remote tool-use evidence, applies Hugging Face's MLX
catalog scope, and enforces local Metal/storage fit. Zero-installed chat
onboarding keeps that tool-use requirement. These are CLI additions; library
`HubSearch` keeps caller-selected requirements generic. Human results use
compact multi-line cards containing only model name, quantization, weights,
memory, context, and tasks. A checkmark marks an exact installed revision;
active and resumable paused transfers are labeled from durable model-manager
state. On a human terminal, those cards become one inline, height-bounded
viewport on stdout. Up and down move the selection rail, while left and right
move through cached cursor pages rendered only as
`< Prev | Page N | Next>`. Enter on an installed revision returns through the
normal `chat --model` dispatch. Enter on any other revision downloads that
exact revision, then asks whether to start chat. Escape or `q` closes the
browser. No second selector or opaque cursor text is appended.
The viewport reserves one physical cursor row, remeasures its prior frame
after terminal resizes, and reads Ctrl-C as a cancellation key instead of
raising an interrupt inside raw input cleanup.
Redirected and `--json` searches never prompt. Skipped-candidate diagnostics
collapse to a count unless `--verbose` is present, while `--json` retains the
complete structured page.

TTY downloads render one independently animated live region with exact
aggregate bytes, completion percentage, verified-file count, recent
persisted-byte speed and ETA when meaningful, plus one animated progress row
for every active file. Up to four planned files transfer concurrently while
returned manifest records remain in plan order. Resumed bytes count toward
completion but never toward network speed. Transfer completion changes the
label to finalizing until local certification and publication return.
Redirected human output stays deterministic and omits chunk-level progress
spam; JSON event schemas remain unchanged. Ctrl-C cooperatively pauses a
resumable transfer before returning. An independently scheduled signal watcher
sets the same cancellation flag during synchronous local inspection/load
phases, and a final checkpoint precedes publication. Live regions use buffered
redraws, bounded widths, and drop-safe cursor restoration without an alternate
screen.

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

Model selection uses invocation-specific capabilities. Generation and chat
with thinking enabled require `interaction:thinking_toggle`.
`interaction:reasoning_history` is never an invocation requirement: a model
whose template does not preserve prior reasoning still runs in thinking mode;
the agent strips prior-turn reasoning from each model request instead.
One-shot `--thinking` overrides are
resolved once and passed consistently to selection, model loading, and the
request. A new chat materializes an unresolved `auto` default as `on` before
writing immutable Session semantics, so reasoning is shown by default;
explicit `off` remains off. A resumed Session keeps its exact stored mode:
historical `auto` is neither migrated nor reinterpreted and retains its prior
auto/off behavior. New default/on chat selection requires
`interaction:thinking_toggle`, while stored
historical auto does not. For a new chat without `--model` or a configured
default, selection groups healthy snapshots by stable model reference and keeps
the newest from each group. One installed reference is automatic; multiple
references open the existing arrow selector on a human terminal, while
redirected and JSON invocations require an explicit model. Capability
validation follows selection, combining current static inspection evidence
with runtime-only facts recorded in the immutable manifest. Zero installed
references alone enters Hub onboarding.

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

`translate` drives translation-capable models (TranslateGemma-style
templates that require per-message language pairs). One-shot mode reads a
positional TEXT or bounded UTF-8 stdin and needs a resolved language pair
from `--from`/`--to` or the `[translate]` configuration section; on a
terminal with no TEXT it opens a stateless interactive translator whose
prompt shows the live pair (`en→de❯`) with `/from`, `/to`, `/swap`,
`/langs`, `/model`, `/help`, and `/quit` commands. Each line is one
independent request; no conversation history is sent. Language codes are
validated against the code→name table extracted from the loaded template
when one is present; without a table, codes pass through and the template
render is the authority. Model selection requires `task:translation`, and
`hub search --require task:translation` skips the implicit
`interaction:tools` search requirement because translation models are
tool-less by design. `chat` against a translation-only model fails its
trait check with a pointer to `translate`. Translation output streams as
plain text, never through the markdown renderer.
