# client — context

## Terms

- **Client**: a handle to exactly one loaded MLX checkpoint. Clones share
  the same loaded weights, prompt cache, and inference queue.
- **Defaults**: per-client generation settings used by native and Rig
  requests. Native per-request values override defaults; Rig precedence is
  `additional_params` > request fields > client defaults.

## Invariants

- One `Client` = one checkpoint. There is no model-by-name registry;
  loading a different model means constructing another `Client`.
- All session access happens on the client's single dedicated inference
  thread (load included). This is an MLX correctness requirement, not a
  style choice: GPU streams and Metal command encoders are thread-bound
  and arrays evaluate lazily, so a lazy array created at load must be
  evaluated on the same thread. The FIFO job queue also means at most
  one generation runs at a time per client.
- Queue admission is bounded and non-blocking. The capacity counts waiting
  jobs, not the currently running job.
- Native streaming has bounded token backpressure. Cancellation closes the
  receiver before waiting for the next cooperative engine boundary.
- Native progress uses that same bounded channel. Prompt and cache snapshots
  come from engine accounting; decode snapshots advance from exact emitted
  token ordinals, not display-callback or string counts.
- Native and Rig answer streams are exact prefixes of their terminal text.
  A raw tool boundary withholds later answer deltas until terminal validation;
  the validated suffix is emitted before structured calls and completion.
  Non-prefix terminal text is a provider protocol failure.
- Every queued completion checks cancellation before starting inference.
  During generation, cancellation is observed at token/callback boundaries.
- Speculative-decoding accounting is response-scoped. A non-streaming
  response or terminal streaming response carries only that call's counters;
  the client keeps no shared "last response" slot.
- Reasoning configuration has exactly two layers: the client-wide
  default (`ClientBuilder`) and the per-agent override (`ReasoningExt`).
  Because rig's `additional_params` setter replaces rather than merges,
  each `ReasoningExt` method writes a *complete* reasoning config and
  the last call wins - `reasoning_budget_tokens` therefore implies
  thinking-on rather than depending on a prior call. Explicit thinking-off
  clears any client-default reasoning budget.
- Reasoning capability has two independent facts. `reasoning_history`
  preserves prior explicit spans; `thinking_toggle` distinguishes an enabled
  generation prompt from a disabled one. Request history requires the former.
  Effective thinking-on and reasoning budgets require the latter.
- Native and Rig requests use the same capability validator after bounded
  engine conversion and before queue submission. Tool protocol history also
  requires current matching declarations so a dedicated tool template is
  selected and supplied its declaration contract.
- `from_path`/`build` validate and canonicalize the directory before handing
  it to the engine so path mistakes produce `Error::ModelPath`.
- `Session` and native `Array` are structurally `!Send + !Sync`. The dedicated
  inference worker constructs, uses, and drops its session on that same OS
  thread; only Emelex job closures and result values cross the queue.
