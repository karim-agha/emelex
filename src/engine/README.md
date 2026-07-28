# engine

The vendored [mlex](https://github.com/vaibhavpandeyvpz/mlex) v0.1.3
inference runtime: safe Rust over vendored MLX/mlx-c C++ (built by the
crate's `build.rs` via CMake, statically linked, Metal GPU backend).
Loads quantized MLX checkpoints, applies chat templates, parses tool
calls, streams tokens, and maintains a KV prompt-cache pool.

Internal only (`pub(crate)`): no engine type appears in emelex's public
API, so the engine can be re-vendored or replaced without breaking
consumers. Lint policy: upstream style is preserved; the whole subtree is
exempted from workspace lints at the `mod engine;` declaration in
`src/lib.rs`. Provenance and license: see `../../ATTRIBUTION.md`.

## Local behavioral patches (beyond mechanical vendoring)

Every site is marked `emelex patch` in the source. Grouped by file:

`generate.rs`
- **Boundary-snapshot prompt caching**: upstream's full-prompt cache
  entries can never serve the next turn on templates that insert
  non-history tokens into the generation prompt (Qwen3-family
  `<think>` blocks) - the cache was effectively inert. The prefill is
  split at the conversation boundary (transcript rendered without the
  generation prompt), the cache state there is snapshotted and stored
  as the (only) pool entry for the lineage; every later turn extends
  it verbatim. Text-only prompts.
- **Cache-accounting honesty**: `usage.cached_tokens` reports only
  tokens actually served by the pool, not the boundary prefill
  computed this call; pooled ids exclude a final generated token whose
  forward pass never ran (eos/abort), so entry ids always match the KV
  they carry; an entry covering the entire prompt is treated as a miss
  (an empty prefill suffix cannot produce logits).
- **Forced-close KV integrity**: when a reasoning budget fires, the
  budget-exceeding token is fed through the model together with the
  injected close marker (it was previously skipped, desyncing KV from
  ids and sampling from a state that never saw it).
- **Incremental UTF-8 stream decoder** (`StreamDecoder`): per-token
  decodes garbled multi-byte characters split across tokens (CJK,
  emoji) into U+FFFD in streamed text; ids are withheld (bounded)
  until they decode cleanly. Terminal withheld bytes are flushed as the
  tokenizer's replacement-character decode instead of disappearing.
- **Forced-boundary duplicate handling** (with `reasoning.rs`): after a
  budget-forced close the teacher-forced boundary is authoritative. The
  decoder suppresses at most one immediately duplicated model close after no
  more than eight leading whitespace bytes. Divergent prefixes, partial
  markers at termination, and delayed close markers remain literal answer
  bytes, keeping stream and terminal output lossless and identical.
- **Poison recovery**: a panicked generation no longer bricks the
  Session - the pool mutex recovers via `PoisonError::into_inner`.
- **Allocation clamp**: `Vec::with_capacity` for generated ids is
  capped so a huge `max_tokens` cannot abort the process.
- **Non-special stop-token display strip**: a registered eos id whose
  vocab entry is *not* marked special (Laguna's `</assistant>`, id 24)
  survives the skip-special display decode and would leak its literal
  text as the reply's tail; the token loop strips exactly the stop
  token's own text from the final display piece.
- **Bounded MLX buffer cache**: `Session::load` caps MLX's
  freed-buffer cache (2 GiB); left at its default the cache accumulates
  prefill transients without bound (>16 GiB after one long-context
  prompt), crowding a large checkpoint into the Metal wired limit.

`models/cache.rs` — windowed `KvCache` (see the struct docs): sliding-
window layers retain only their window plus growth slack instead of
full history, trims materialize into fresh buffers (a slice *view*
would pin the old allocation), and the absolute `offset`/`start` split
keeps rope positions and mask coordinates correct
(`ops::sliding_window_mask` takes the retained keys' start position).
Adopted by both sliding-window architectures: Laguna (S-2.1) and
Gemma4, whose KV-shared tail layers receive the owning layer's start
position through the `SharedKv` tuple.

`reasoning.rs` — reasoning syntax is prefix-scoped: an opening marker is
recognized only at the reply start after optional whitespace, so quoted
markers in ordinary answers remain text. `ReasoningBudget` keeps a bounded
tail instead of buffering (and rescanning) the whole generation; its marker
token does not consume budget, and the Nth content token exhausts a budget
of N. `split_reasoning` preserves answer-tail whitespace while folding
only recognized wire markers out of the reply. After a teacher-forced close,
one bounded immediate duplicate may be removed; later closes are literal
answer text.

`streaming.rs` — `StreamClassifier` is a stateful, lossless marker
transducer. It handles delimiters split across tokenizer pieces and multiple
boundaries inside one piece, removes only recognized wire markers, and may
produce zero or multiple typed display segments for one generated token.
Only a possible marker suffix is withheld; terminal partial markers flush
literally. Every tool opening emits a `ToolCall` structural signal, including
an empty span, so provider bridges can defer post-marker text until terminal
schema validation.

`sampling.rs` — `total_cmp` ordering (a NaN logit must not panic the
generation thread); non-positive `top_k` is a no-op instead of an
`as usize` wrap; top-k filtering is O(n) selection instead of a
full-vocab sort + HashSet; the sampling fallthrough returns the argmax
instead of the last vocab index.

`tools.rs` — tool-call ids are process-unique (`call_N` from an atomic
counter), not per-generation, so multi-turn conversations never carry
duplicate ids. Tool output is parsed as an untrusted proposal with its
exact byte span: Hermes XML and Laguna calls must close and consume every
field; JSON objects reject duplicate keys; Gemma parsing tracks quotes,
escapes, and nested arrays/objects; duplicate arguments and malformed
pairs fail closed. Generation accepts and strips only proposals whose
name was advertised for the request and whose arguments satisfy Emelex's
bounded JSON Schema vocabulary. Unknown, malformed, truncated, and
schema-invalid markup remains visible and is never returned as a call.
Executable schemas use a fail-closed vocabulary: structural, conditional,
object, array, string-length, numeric-range, enum, and constant constraints are
enforced; unsupported keywords such as references, regular-expression
patterns, formats, decimal `multipleOf`, and unevaluated vocabularies reject
the declaration.

`tokenizer.rs` — the chat-template minijinja Environment is compiled
once per Session (upstream re-parsed the template on every render,
twice per `generate_cached` call). Static model discovery uses the same
fuel-limited, output-bounded evaluator for semantic capability probes. Chat
requires a successful baseline render. Tool use is inferred only when two
independently varied declarations and schemas survive rendering, followed by
two ordered call/argument/result parser round trips for Hermes, Gemma, and
Laguna (one call for Llama JSON's single-call protocol). System,
reasoning-history, thinking-toggle, and media capability probes also traverse
the tools-enabled history. Protocol
selection comes from the exact selected template, not model-type naming;
missing, malformed, or ambiguous tool evidence fails closed to ordinary chat.
Gemma-native rendering
normalizes generic assistant calls followed by tool-role results into the
template's ordered assistant `tool_responses` field and rejects orphaned,
reordered, duplicate, mismatched, or non-text results. Reasoning history and
thinking toggle are inferred independently; their generic reasoning label is
only the union used for discovery.

`models/cache.rs` — `KvCache` uses amortized chunked growth
(`slice_update` into a preallocated buffer, sliced-view reads) instead
of upstream's full-tensor concatenate on every decode step (O(n^2)
memory traffic over a generation). Element-for-element equivalence with
the naive concat is regression-tested across growth boundaries.

`array.rs` — `to_vec_f32`/`to_vec_u32` force a `contiguous`
materialization before raw host reads: `astype` no-ops for
already-matching dtypes, so a sliced/strided lazy view would otherwise
be read through its parent buffer (wrong data). Upstream only avoided
this by accident (bf16 logits always convert). Scalar arrays also return
an empty shape without constructing a Rust slice from mlx-c's nullable
empty-vector data pointer. Scalar item reads explicitly evaluate first so
asynchronous Metal failures cross a fallible Rust boundary before host data
is read. Construction validates rank, non-negative dimensions, checked element
count, and exact data length before entering mlx-c; an empty returned handle
consumes the native error and becomes `Error::Mlx`. Clones share one validated
native handle through `Rc`, making arrays—and therefore loaded sessions—
structurally `!Send + !Sync`. After pure shape validation, every constructor
that creates a non-empty MLX object initializes Emelex's relocated runtime
first; a fresh-process regression prevents default-metallib initialization
from escaping beneath the stream boundary.

`weights.rs`, `models/mod.rs`, and `model/layout.rs` — one checkpoint snapshot
pins the model-directory descriptor, then owns the exact `config.json` bytes
and selected file descriptors opened relative to it through model construction,
MLX loading, and MTP certification. No phase reopens a model-owned pathname.
Every private, unlinked shard clone is fully hashed once before loading. Every
selected lazy MLX load is eagerly materialized, then the still-open
descriptor's identity and header/layout are revalidated before Emelex accepts
those tensors.

`vendor/mlx/mlx/backend/metal/{device,eval,event,fence}.cpp` and
`vendor/mlx-c/mlx/c/{array,stream}.cpp` — every committed command buffer has
one central completion handler. The handler is a `noexcept` catch-all that
retains the first `NSError` in stream-local shared state; it never unwinds
through a Metal framework thread. Caller-thread synchronization waits for all
committed buffers, consumes the first error, and lets mlx-c translate it into
the existing Rust `Result`. Synchronous array evaluation synchronizes its
captured stream for CPU and GPU arrays before returning success,
auto-committed buffers are covered, and
the command-encoder destructor drains without throwing. Callback bookkeeping
owns shared state rather than a raw encoder pointer. A one-shot native fault
seam is exercised by a subprocess survival regression that forces an
auto-commit, observes `Err`, then proves the stream remains usable.
Metal library/kernel caches use one library-before-kernel lock order; shared
lookups are read-only and entries are immutable for the Device lifetime.
Custom libraries use a canonical length-prefixed `(name, source)` cache key.
The process-lifetime cache may grow with distinct generated sources, but its
strong ownership is required because Metal command buffers use unretained
resource references. Allocator diagnostics and limits take the same allocator
mutex as mutations, so benchmark reads remain race-free while another loaded
Client allocates.

`vendor/mlx/mlx/backend/cpu/encoder.{h,cpp}`,
`vendor/mlx/mlx/scheduler.{h,cpp}`, and
`vendor/mlx-c/mlx/c/transforms.cpp` — CPU command encoders are thread-local,
matching MLX's per-thread stream affinity. Scheduler workers catch task
exceptions (including non-`std::exception` throws), normalize and retain the
first one per stream; caller synchronization crosses a `future::get` barrier,
consumes it, and lets mlx-c translate it into `Error::Mlx`.
After one failure, dependent work is discarded until the synchronization
barrier reports and clears that failure. Grouped dispatch accounting uses one
shared RAII registration whose last owner balances enqueue failure, normal
execution, throwing execution, and quarantine-without-execution exactly once.
Subprocess regressions cover all four paths and prove stream reuse.

`quant.rs` — config parsing admits only tuples backed by the vendored MLX
kernels: affine groups 32/64/128 with 2/3/4/5/6/8 bits, mxfp4 32x4,
mxfp8 32x8, and nvfp4 16x4.

`models/gemma4` — configuration-controlled optional projections are rechecked
at use and return typed model errors if construction invariants drift; model
input cannot turn a missing component into a Rust panic.

`prompt_cache.rs` — `is_prefix` is `pub(crate)`; expired entries are
evicted on insert as well as lookup, so an idle process does not pin
multi-GB KV state past its TTL.

`mtp_certification.rs` — actionable MTP support is bound to the exact
`config.json` and two checkpoint-shard SHA-256 digests covered by the
checked-in three-step parity certificate. Layout-compatible but uncertified
weights remain non-actionable. Certificate eligibility is resolved from the
descriptor-backed snapshot before model construction. The loader still
iterates every name for duplicate and descriptor validation, but frees an
ineligible checkpoint's MTP lazy handles before Rust `Array` conversion or
evaluation; their payloads never materialize and no MTP module is retained. An
explicit speculation request fails instead of silently switching to
target-only decoding.

`media/image.rs`, `media/audio.rs`, and `media/video.rs` — hostile media
is constrained before model execution: encoded and decoded allocations,
dimensions, geometry, and raw-audio padding are bounded.
Image decoding uses the `image` crate's decoder limits and revalidates
actual source and processed geometry. WAV parsing rejects truncated
frames; multi-channel PCM is decoded and downmixed frame by frame rather
than retaining a full interleaved f32 copy. Sample-rate conversion is
native and bounded. The self-contained audio surface is RIFF/WAVE PCM16
or float32. Encoded video fails closed until a decoder backed by bundled
code or macOS system frameworks is part of the runtime. Emelex never
resolves or executes ambient codec binaries. The image crate's unused
Rayon feature is disabled. Request-wide accounting rejects more than 64
processed items, 256 MiB of aggregate encoded media, 512 MiB of retained
processed tensors, or 16,384 aggregate media soft tokens. A generation
call also charges each placeholder expansion against its effective
prompt-plus-output context window before retaining the processed tensor,
preventing individually valid attachments from amplifying into an
unbounded aggregate.

`generate.rs` validates the rendered placeholder stream before decoding any
attachment. Image, audio, and video placeholders must match attachments
exactly in count and cross-modality order. Dropped attachments, extra
placeholders, reordered mixed media, missing video IDs, or colliding
image/video/audio placeholder IDs fail closed instead of degrading to a
text-only prompt.

`generate.rs`, `media/image.rs`, `media/audio.rs`, and the multimodal model
fan-outs — provider-owned requests carry a cooperative cancel-on-drop probe
through template/media preprocessing and prefill. Image row conversion, WAV
parsing/downmixing/resampling, audio spectrogram frames, and attachment
boundaries poll it. Projected vision features are evaluated one image/frame
at a time; projected audio features are evaluated one audio chunk at a time.
Text, fused-media, fresh-MTP, resumed-MTP, and speculative-boundary prefill
all evaluate the decoder in 512-token chunks, so a dropped future or stream
stops before constructing the next chunk. Each MTP chunk evaluates only its
cache-producing recycle-hidden branch before the cancellation checkpoint;
vocabulary-sized priming logits stay lazy. Boundaries preserve the pair stream
with the prior chunk's detached last hidden as the next chunk's bridge
frontier. Direct engine calls keep the historical single-pass prefill path.
Cancellation returns before prompt-cache publication, so partially
advanced private working caches cannot enter the shared pool.

All of the above are worth offering upstream.

### MTP self-speculative decoding (one subsystem, many files)

ADR: `../../docs/adr/0005-mtp-certification.md`. Qwen3.5/3.6
checkpoints ship a multi-token-prediction (MTP) module that upstream
deletes at load time; these patches load it and use it as a self-draft
model for speculative decoding — draft k tokens cheaply, verify them in
one batched backbone forward, keep the accepted prefix. Off by default
(`GenerateOptions.speculative_tokens`). Unlike the independent patches
above, these pieces only work together; see the re-vendoring warning
below. Grouped by file:

`sampling.rs` — an additive speculative section (`sample`'s arithmetic
and RNG consumption stay byte-for-byte): `probs`/`sample_from_probs`
expose the existing filter→CDF pipeline as reusable halves;
`verify_speculative`/`verify_greedy` implement Leviathan/Chen rejection
sampling over draft proposals; `verify_accept` is the decision-only
variant the round driver uses (returns an `AcceptVerdict` carrying the
successor distribution instead of drawing from it). Two numeric guards
ride along: `sample_from_probs` never emits a zero-mass entry on an
exact-zero draw, and `probs` renormalizes through f64 — a sequential
f32 sum over a ~250k-entry near-uniform row drops ~3e-3 of mass, enough
to trip the verifier's own 1e-3 sum gate and silently defeat
speculation on high-entropy stretches. `#[cfg(test)]` one-shot and
counted fault-injection hooks (`inject_failure`/`inject_failure_at`)
drive the failure-transition rows.

`models/cache.rs` — rollback primitives for rejected drafts:
`KvCache::truncate_to` (offset-only rewind, in-range infallible by
contract; stale rows are overwritten by the next append and fetched
views slice only `0..needed`), `would_trim` (window preflight: would
this append irreversibly discard history a rollback might need?),
`is_pristine`, `rollback_state`/`rollback` with `LayerRollback` (O(1)
offset capture for attention, small state-array clones for
gated-delta/dhara; kind mismatches err via `kind_name`), and
`needs_refeed` (recurrent state only exists at snapshot boundaries, so
arbitrary-position rewind needs a re-feed when any non-attention layer
is present).

`generate.rs` — the decode loop restructured around speculation:
`TokenEmitter` extracts the per-token display/classification pipeline
so the ordinary step, the forced-close path, and the speculative
accepted-block path run one identical state machine (UTF-8 withholding,
marker classification, stop-token strip, immediate forced-close duplicate filter,
cancellation); `DecodeOutcome`'s two ledgers — `emitted` (consumer
facing: finish classification, usage, reply text, streaming) and
`committed_len` (exactly what the caches contain: the prompt-cache pool
reads this and nothing else) — replace the old `(ids, last_unfed)`
pair, with committed always a prefix of emitted; the forced-close path
now runs callbacks BEFORE feeding each close token, fixing a real
cancellation desync where mid-marker cancellation left KV ahead of
pooled ids (regression-pinned by
`forced_close_cancellation_keeps_caches_at_committed_prefix`);
`SPECULATIVE_TOKENS_CEILING` (= 8) with `resolve_speculative_tokens`
normalizing `Some(0)` to off and clamping; `SpeculationStats` per-call
accounting; prefill priming — `prefill_prompt` routes through
`forward_hidden` and primes the MTP module over the prefill hiddens
when speculating, `prefill_resume` continues priming from a pooled
`MtpState`, and both share the same bridge-correct cooperative chunk
driver used by speculative conversation-boundary snapshots. Each chunk
materializes `recycle_hidden` (and therefore its cache updates) before its
cancellation boundary without materializing priming logits. The loop
body itself now lives behind the `spec.rs` seam.

`models/mtp.rs` (entire file) — the architecture-neutral MTP
vocabulary: `BackboneOutput` (pre-final-norm hidden + logits),
`MtpStepOutput` (post-`mtp.norm` recycle hidden + logits), `MtpCaches`,
`MtpState` (poolable snapshot: caches + `pairs_fed` + detached
frontier), `MtpDetection`.

`models/mod.rs` — the `Model` fan-outs `has_mtp` / `new_mtp_caches` /
`forward_hidden` / `forward_mtp` (only the `Qwen35` arm implements
them; every other architecture errors or returns empty and keeps its
`forward` dispatch untouched), plus a load-bearing ordering in the
qwen3_5 load arm: `detect_mtp` runs BEFORE `sanitize` on the raw key
set when the caller has already established certificate eligibility, because
sanitize deletes every unpreserved MTP key. Uncertified production snapshots
pass `MtpDetection::None`, so MTP tensors never reach `Qwen35Mtp::load`.

`models/qwen3_5/mod.rs` — the backbone forward split at the layer-loop
→ final-norm → head boundary (`forward_hidden`, arithmetically
identical op order to `forward`, which is now a thin wrapper over it);
the `Qwen35Mtp` module (fusion norms + a 2H→H fc + one gated
full-attention block sharing the backbone's embeddings and head);
`detect_mtp` (raw-HF orientation predicate and an exact main-map namespace
contract: only `language_model.mtp.*` is accepted; sidecars,
`language_model.model.mtp.*`, and root `mtp.*` are rejected) and
`validate_mtp` (15-key sentinel set,
single-layer/shared-embeddings/gate-shape checks, dense-BF16 guards — every
rejection warns and leaves the backbone byte-identical to a no-MTP load);
`sanitize` now takes the detection outcome and preserves/canonicalizes detected
MTP keys to the internal bare `mtp.*` prefix instead of deleting them.

`nn.rs` — `WeightMap::peek`: non-consuming tensor inspection, so
pre-sanitize MTP detection is contractually non-mutating (no
take/reinsert while probing).

`ops.rs` — MLX memory diagnostics and cache-limit controls install the
non-terminating error handler, check every mlx-c status, and return `Result`.
`reset_peak_memory` lets a benchmark cell measure its own peak rather than the
process high tide. Model loading propagates cache-limit failure instead of
silently continuing without its wired-memory bound. Metal resource-limit
admission, ordinary buffer creation, and no-copy buffer creation share the
allocator mutex, so concurrent clients cannot oversubscribe the device's
resource count. `tools/test_native_invariants.py` pins that critical-section
ordering without requiring the vendored network-fetched C++ test harness.

`prompt_cache.rs` — `CacheEntry.mtp: Option<MtpState>`, aligned to the
entry's ids (`pairs_fed == ids.len() - 1`, enforced on EVERY insert: a
`debug_assert` plus the warn-and-drop release guard `aligned_mtp`) and
overwritten wholesale on every update (a spec-off extension writes
`None`); `find_longest_compatible_prefix(ids, require_mtp)` — with
`require_mtp` false byte-identical to `find_longest_prefix`, with it
true `mtp`-less entries are skipped (a cold miss when none remains),
NOT evicted (they still serve non-speculating callers), and NOT
LRU-refreshed, so repeated incompatible traffic age-demotes them.

`spec.rs` (entire file) — one decode round behind the private
`RoundOps` seam: `RoundDriver::run_round` implements the target-only
and speculative rounds, forced close (empty-close split by trigger),
the failure-transition table, and the completion-path ordering
(reconcile → detach → MTP commit → stage-4 successor selection, the
single draw point); `OpError` classifies every seam failure as
`Invalid` (invalid-cache exit: a target forward failed mid-mutation, caches
unusable), `MtpForward` (discard the MTP state, continue target-only),
or `Host { salvage }` (exact-prefix exit: caches exactly match the stated
committed prefix). The `#[cfg(test)]` fake drives every disposition and
failure row with batch-size-invariant deterministic f64 host
arithmetic.

`parity.rs` (entire file) — the logit-parity enablement gate: replays
the pinned Python dump recipe (`tools/mtp_parity_dump.py`) through this
engine and compares exactly three `.npy` goldens (`EMELEX_PARITY_GOLDENS` +
`EMELEX_TEST_MODEL`). The checked-in `mtp_certification.json` binds the
implementation ID, source and converter revisions, MLX version, model
config, both model shards, metadata, and all three golden rows. All seven
file hashes are verified before MLX loads the model or computes logits.
`tools/party.py` runs only this ignored external test, requires its success
sentinel, and kills its complete process group at a hard 1,200-second
deadline; missing, skipped, timed-out, or mismatched inputs fail.

`test_support.rs` (entire file, `#[cfg(test)]`) — a hand-rolled
safetensors writer plus a tiny on-disk model builder (optionally
including the 15-tensor dense MTP module) so `Session::load` and the
decode loop run end-to-end without a real checkpoint.

`tokenizer.rs` — the tiny-model fixture gate test: the committed
`tests/fixtures/tiny-model` trio loads, and its chat template appends
the generation prompt strictly after the conversation boundary (the
property the boundary snapshot and MtpState alignment depend on).

`mod.rs` — declares `spec`, `parity`, and the test-only `test_support`.

## Re-vendoring procedure

To bump to a future mlex version, repeat these steps against the new
crates.io package (`https://static.crates.io/crates/mlex/mlex-<V>.crate`):

1. Extract the package. Copy `src/*` here (`lib.rs` becomes `mod.rs`),
   `build.rs` to the crate root, `vendor/` to `../../vendor/`, and
   `LICENSE` to `../../licenses/mlex.LICENSE`.
2. Rewrite self-references: `crate::` → `crate::engine::` in every
   vendored `.rs` file (plain sed; there are no other `crate::` forms).
3. Edition-2024 fixes (mlex is edition 2021):
   - `sampling.rs`: `self.rng.gen::<f32>()` → `self.rng.r#gen::<f32>()`
     (`gen` is a reserved keyword).
   - `sampling.rs` tests: two string literals containing `\n` escapes
     carry `#[rustfmt::skip]` — `format_strings` would split inside the
     escape and change the value (see `tools.rs` hermes tests).
   - `build.rs`: keep the `#![allow(...)]` header and the bindgen
     `.rust_edition(bindgen::RustEdition::Edition2024)` call (plain
     `extern` blocks are a hard error in edition 2024).
   - Match-ergonomics: patterns like `|(_, &p)|` over references need an
     explicit `&` (`|&(_, &p)|`); the compiler pinpoints any new sites.
4. `cargo +nightly fmt -p emelex` (one-time reformat to house style),
   then `cargo clippy -p emelex --all-targets` must be clean and
   `cargo test -p emelex --lib` must pass (vendored unit tests run too).
5. Re-apply the behavioral patches listed above (search the previous
   version's tree for `emelex patch`) unless upstream has adopted them.

   **Warning — the MTP subsystem must be carried over as a BLOCK.** A
   grep-and-re-apply pass cannot reconstruct it: `models/mtp.rs`,
   `spec.rs`, `parity.rs`, and `test_support.rs` are whole-file emelex
   additions (copy them across verbatim), and `generate.rs`'s decode
   loop is *restructured* around them — `TokenEmitter`, the
   emitted/committed ledger split, the priming prefills, and the
   `spec.rs` round seam are woven through the loop rather than inserted
   at greppable marked sites. Port the restructured `decode_loop` and
   its helpers as a unit on top of the new upstream `generate.rs`, then
   lean on the safety net: the `spec.rs` fake suite, the tiny-model
   engine tests, and (live) the parity gate must all pass before the
   bump lands.
6. Update the version numbers here and in `../../ATTRIBUTION.md`.
