# engine — context

## Terms

- **Session**: one loaded checkpoint (model + tokenizer + prompt-cache
  pool), constructed, used, and dropped on one dedicated inference worker.
  It is deliberately `!Send + !Sync`; clonable `Client` handles queue work to
  that worker.
- **Prompt cache**: a pool of KV states keyed by token-ID prefixes. On
  each `generate_cached` call the longest already-computed prefix is
  reused; the remainder is prefilled and the extended state is stored
  back. Transparent to callers — they always pass the full conversation.
- **TokenKind**: classifies each streamed display segment as plain text,
  reasoning (inside a prefix `<think>`-style span), or raw tool-call
  markup. One generated token may emit zero or multiple segments when a
  marker or UTF-8 boundary crosses tokenizer pieces.
- **MTP** (emelex patch, see `README.md` and ADR 0002): a checkpoint's
  multi-token-prediction module — fusion norms + a projection + one
  decoder layer sharing the backbone's embeddings and head. Used as a
  self-draft model for speculative decoding: draft k tokens cheaply,
  verify them with one batched backbone forward, keep the accepted
  prefix. Qwen3.5-only in v1; off by default. Actionable support requires
  both the exact reviewed layout and byte-for-byte identity with the
  checked-in three-step parity certificate.
- **MTP certification**: the checked-in machine-readable binding between
  the Emelex MTP implementation ID, pinned source and converter revisions,
  MLX version, the dense-BF16 config and two model shards, and four golden
  artifacts. `tools/party.py` verifies all seven file hashes before model
  loading, runs exactly three parity steps, and requires a success sentinel
  under a hard 1,200-second process-group deadline. Production resolves this
  exact certificate before model construction. When it does not match,
  checkpoint loading still validates every tensor name and descriptor but
  frees MTP lazy handles before `Array` conversion/evaluation, so their payloads
  do not materialize and no MTP module is retained.
- **Speculative round / SpecState**: a round is one draft → verify →
  decide cycle in the decode loop (`spec.rs`); a call's SpecState — will
  this call speculate at all? — is resolved BEFORE prefill: media
  input, non-pristine caller-supplied caches, a missing MTP module, or
  `speculative_tokens` unset/0 all resolve to disabled, and the call
  runs the historical one-token-per-forward loop byte-identically.
- **Frontier**: the backbone hidden state of the last *committed* token
  — the `prev_hidden` the next draft step consumes. Committed MTP pairs
  always use verified target backbone hiddens, never draft-time recycle
  hiddens; the frontier stored in a pooled `MtpState` is detached.
- **Emitted vs committed ledgers**: `DecodeOutcome.emitted` is the
  consumer-facing token sequence (finish classification, usage, reply
  text, and streaming read it); `committed_len` marks the prefix whose
  KV/state the caches actually contain (the prompt-cache pool reads it
  and nothing else). Committed is always a prefix of emitted — they
  diverge on purpose (an EOS or cancelled token is emitted but never
  fed).
- **MtpState / MtpCaches**: `MtpCaches` is the MTP module's live working
  cache (v1: one full-attention KvCache); `MtpState` is the poolable
  snapshot — caches + `pairs_fed` + the detached frontier — aligned to
  a pool entry's ids by the invariant `pairs_fed == ids.len() - 1`,
  enforced at every insert.
- **Detach**: force an array contiguous and evaluated so it no longer
  references its parent buffer. An evaluated *slice* still pins its
  `[1, L, H]` parent; hidden rows must be detached before cache
  mutations (or pool storage) so they survive truncation and reuse.
- **Self-contained media boundary** (emelex patch): image decoding and
  RIFF/WAVE PCM16/float32 audio decoding/resampling run in-process with
  explicit allocation and geometry bounds. Encoded video fails closed.
  Ambient codec executables are never resolved or run.
- **Processed-media budget** (emelex patch): request-wide accounting applied
  before a processed tensor enters the retained media queues. It independently
  caps attachment count, aggregate encoded bytes, retained tensor bytes, and
  media soft tokens; generation additionally charges exact placeholder-span
  growth against the effective context window.
- **Exact media binding**: after chat-template rendering, the prompt's image,
  audio, and video placeholder sequence must equal the attachment sequence.
  Cardinality, modality order, and placeholder IDs are unambiguous before any
  decoder runs; expansion also verifies that every retained attachment was
  consumed.
- **Semantic template capability probe**: capability discovery executes
  bounded control and synthetic renders through the production Jinja
  environment. Chat requires a successful baseline render. Source keywords,
  comments, and dead branches are not evidence. Tool support requires two
  independently varied declarations and schema sentinels, then two ordered
  call/argument/result round trips for Hermes, Gemma, and Laguna (one call for
  Llama JSON's single-call protocol). System, reasoning-history,
  thinking-toggle, and media support must survive the same tools-enabled
  history path. Failure or ambiguity disables
  tools without disabling baseline chat. Gemma-native
  rendering folds ordered tool-role results into assistant
  `tool_responses` and rejects any structurally ambiguous history. Reasoning
  history and thinking toggle remain independent semantic capabilities.
- **Cooperative cancellation checkpoint** (emelex patch): a synchronous,
  private predicate installed by provider futures/streams. It is polled through
  media preprocessing, between evaluated projected-media items/chunks, and at
  evaluated 512-token decoder-prefill boundaries for text, fused-media, and MTP
  paths. Direct engine calls disable it and retain their historical
  single-forward prefill behavior.
- **Invalid-cache exit / exact-prefix exit**: the two error-exit classes of the
  decode round (`spec::OpError`). Invalid-cache exit: a target forward failed
  mid-mutation — destructive buffer `take()`s make restoration
  impossible, so the caches must not be reused and nothing is pooled.
  Exact-prefix exit: a host-side failure after all structural feeds completed
  — the caches match the stated committed prefix exactly, so pooling
  the prefix remains sound. (MTP-forward failures are neither: the MTP
  state is discarded and the call continues target-only.)
- Metal allocator resource admission and both ordinary/no-copy resource
  creation remain inside one mutex critical section. The count therefore
  includes every admitted resource before another caller can inspect the
  device limit; no-copy exhaustion returns null for the existing copy fallback.

## Invariants (upstream's, relied on by emelex)

- Library execution never writes directly to process stdout or stderr and does
  not activate diagnostics through ambient debug environment variables.
- The `on_token` callback returning `false` stops the decode loop at the
  next token; `GenerateReply.finish_reason` becomes `Aborted`. This is
  emelex's streaming-cancellation mechanism.
- Stream decoding is lossless at completion: clean text, disproved marker
  prefixes, terminal partial markers, and terminal replacement-character
  decodes all flush. Recognized reasoning/tool wire markers alone are
  suppressed.
- Reasoning open markers are reply-prefix syntax. A later literal marker in
  ordinary answer text never changes classification or budget accounting.
- A teacher-forced reasoning close is the authoritative reasoning boundary.
  Stream and terminal paths suppress only one immediate duplicate close after
  at most eight leading whitespace bytes. Divergent, partial, or delayed marker
  bytes remain literal answer text; extraction does not trim payload bytes.
- Tool calls are parsed from the completed reply text, not streamed
  incrementally; `TokenKind::ToolCall` marks raw markup and always signals an
  opening boundary, with empty text when the span itself has no payload.
- Tool-call syntax is an untrusted proposal, not execution authority.
  Only fully closed and consumed proposals matching exactly one advertised
  function and its validated, bounded JSON Schema become `ToolCall`s.
  Only those accepted spans are removed from assistant text; malformed,
  truncated, unknown, ambiguous, and schema-invalid markup remains visible.
  Executable schemas accept only an explicit keyword vocabulary; unknown
  keywords fail closed rather than weakening argument validation.
- Non-zero speculative depth is an explicit capability request. An
  uncertified checkpoint or MTP priming failure returns an error; neither
  condition silently changes the request into target-only decoding.
- Sampled speculative verification materializes all target rows through one
  batched softmax and one host read. Row views share that backing allocation;
  host normalization validates non-finite values and salvages only the
  completed prefix if a later row fails.
- `GenerateReply.usage.cached_tokens` counts prompt tokens served from
  the cache pool — the number emelex reports as `cached_input_tokens`.
- A poisoned cache-pool mutex is recovered with
  `PoisonError::into_inner`; one panicked generation does not permanently
  brick the session.

- **Boundary entry** (emelex patch, see `README.md`): an extra pool
  entry snapshotted at the conversation boundary (transcript rendered
  without the generation prompt). It exists because full-prompt entries
  are unextendable on think-block templates, and because recurrent
  (gated-delta) layer state cannot be truncated retroactively — the
  snapshot must be taken mid-prefill.

- MLX thread affinity: default streams are per-thread, GPU evals encode
  on the calling thread, and Metal command encoders are registered only
  on a stream's creating thread. Lazy arrays therefore must be
  evaluated on the thread that recorded their ops - the reason the
  provider pins each Session to one dedicated inference thread. `Array`
  clones share one native handle through `Rc`, structurally preventing arrays
  and loaded sessions from crossing threads; only client job messages cross
  the boundary.
- MLX object construction is also a runtime boundary. Pure shape validation
  fails before runtime work, but every valid slice/scalar constructor installs
  Emelex's relocated metallib path before calling mlx-c. Empty output handles
  remain inert until a checked native operation fills them.
- CPU command encoders are also registered per thread. Scheduler workers catch
  task exceptions and retain the first failure per stream while continuing to
  service the queue. Caller synchronization first crosses a future barrier,
  then takes and rethrows the retained failure inside mlx-c's exception guard.
  After a failure, dependent tasks are destroyed without execution until that
  barrier. One shared grouped-dispatch completion owns both registration and
  release, balancing exactly once after enqueue failure, execution, throw, or
  quarantine skip; scheduler accounting cannot remain active after the error.
- Checkpoint shards are opened without following symlinks and loaded through
  the validated descriptor rather than a re-opened pathname. Complete shard
  digests are checked before and after eager tensor materialization.
- Metal completion failures never throw on framework callback threads.
  Each `CommandEncoder` owns a shared, mutex-protected error state and adds
  exactly one `noexcept` catch-all handler per committed command buffer. The
  handler retains the first `NSError`, accounts for all auto-committed
  buffers, and wakes synchronization. Caller-thread synchronization waits
  until the stream has no pending command buffers, takes and clears the first
  observed failure, then throws inside the mlx-c exception guard so Rust
  receives `Error::Mlx`. Synchronous GPU `Array::eval` includes this boundary;
  command-encoder destruction drains and discards errors without propagating.
- Metal library/kernel caches always lock library before kernel. Shared lookup
  never inserts, and entries remain immutable for the Device lifetime. Custom
  libraries use a canonical length-prefixed `(name, source)` key so distinct
  source cannot alias. The process-lifetime ownership intentionally accepts
  potentially unbounded custom-kernel cache growth for safety with Metal
  command buffers' unretained resource references. Allocator counters, limits,
  and cache-size diagnostics use the allocator mutex, including bench-feature
  reads concurrent with inference.
- Every mlx-c allocator diagnostic and cache-limit status is checked after
  installing Emelex's non-terminating error handler. Session load must fail if
  its freed-buffer cache bound cannot be installed.
- Gemma4 optional per-layer/KV projections are construction invariants but
  remain fallible at use. Missing components return `Error::Model`, never
  panic.
- Media preprocessing treats encoded bytes and checkpoint-provided
  geometry as hostile. Encoded size is checked before parsing; decoder
  dimensions/allocation, source aspect ratio/pixels, processed tensor
  pixels, image soft-token geometry, audio sample windows/padding, clip
  duration, individual frames, and aggregate frames all have independent
  bounds. Across a request, no more than 64 processed items, 256 MiB encoded,
  512 MiB retained tensors, or 16,384 media soft tokens may accumulate.
  Context-window charging happens before each local processed tensor is pushed,
  so failure drops that tensor without retaining a larger aggregate.
- Client and Rig generation cancellation is observed before and during media
  preprocessing, after each evaluated projected image/audio chunk, and between
  evaluated 512-token decoder-prefill chunks. This includes ordinary text,
  fused media, fresh MTP, pooled-MTP resume, and speculative boundary prefeed.
  Each MTP boundary carries the detached last target hidden forward as the
  bridge pair, preserving one-pass target/MTP cache alignment and outputs.
  Its cache-producing recycle-hidden branch is materialized before the
  cancellation checkpoint; priming logits remain lazy.
  A cancelled prefill mutates only request-local cache clones; it returns before
  boundary/full-prefix cache publication.
- A codec leader's successful exit is not completion: descendants in its
  process group are killed before pipe drains. Every post-spawn error,
  output overflow, and timeout takes the same whole-group kill and leader
  reap path through an RAII guard. Pipe readers retain only their byte
  budget while continuing to drain, and result collection has a separate
  deadline.
- Audio input is accepted only as bounded RIFF/WAVE PCM16 or float32 and
  is downmixed/resampled in process. Video content is rejected before
  inference until a bundled or macOS-framework decoder exists.

## What we may touch

Preferably nothing beyond the mechanical re-vendoring steps and the
documented `emelex patch` sites in `README.md`. New behavioral changes
need the same treatment: a marked comment at the site, an entry in
`README.md`, and ideally an upstream offer.
