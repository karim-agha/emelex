# Generation — context

## Invariants

- The native API does not depend on Rig.
- Extensible public input records are non-exhaustive and retain explicit
  constructors; downstream callers must not depend on exhaustive literals.
- Message validation occurs before inference-queue submission.
- Tool schemas are validated against a fail-closed executable vocabulary.
  Unknown keywords are never advertised while silently ignored.
- Tool protocol history requires matching current declarations on every
  request; declaration-free stateless repetition is rejected.
- JSON schema and argument size checks are exact, bounded, and preceded by
  iterative structural limits; reasoning, IDs, and protocol metadata also have
  aggregate request ceilings.
- Encoded video fails before engine conversion. Image support does not imply
  video support.
- `validate_audio_bytes` is the pre-model, pre-MLX attachment gate. It enforces
  bounded RIFF/WAVE PCM16 or float32 structure, metadata consistency, sample
  framing, duration, and finite float payloads without allocating decoded or
  resampled samples.
- One loaded model owns one bounded FIFO inference queue and one GPU thread.
- Stream backpressure is bounded; cancellation closes the receiver before
  waiting for the engine's cooperative token-boundary stop.
- Progress is cumulative and exact. `Prompt` has no cache result, `Prefill`
  carries the resolved cache hit, and each `Decode` event advances with one
  emitted token ID even when that token produces zero or multiple text spans.
- `GenerationStream::cancel_and_wait` observes worker completion after
  unblocking the bounded receiver. Its wait is cancellation-safe: dropping a
  pending wait leaves the completion receiver available for another call.
  One-shot CLI cancellation uses it before process exit.
- Responses keep answer, reasoning, tool calls, usage, finish reason, and MTP
  accounting separate.
- Explicit thinking-off clears an inherited reasoning budget. An explicit
  request budget remains authoritative and invalid while thinking is off.
- `Content::Translation` is user-role-only and must be the sole content
  part of its message (the TranslateGemma template contract is exactly one
  mapping per turn); language codes and text must be non-empty after
  trimming. The serde shape is `{"type":"translation","data":{...}}` and
  is pinned by a round-trip test.
