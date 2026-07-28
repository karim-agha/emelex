# Generation

Native request and response types are the primary inference API. Requests carry
complete message history, optional media, tool schemas, and per-call overrides.
An explicit thinking-off override clears the client-default reasoning budget;
an explicit request budget with thinking off is rejected.
Public tool input records are non-exhaustive so Emelex can add metadata without
a semver-major break. Callers construct them through `ToolDefinition::new` and
`ToolCall::new`, then may update their public fields.

Validation is whole-request and happens before engine conversion. Counts,
content, reasoning, tool identifiers, tool arguments, descriptions, and schemas
have per-item and aggregate ceilings. JSON sizes are counted through a bounded
writer after iterative depth/node checks, avoiding proportional serialization
allocations. Executable tool schemas use an explicit supported-keyword
allowlist; unknown constraints fail closed.

Any assistant-call/tool-result history must repeat the matching current tool
definitions. This keeps dedicated tool templates selected and prevents
historical protocol state from being rendered without the declarations against
which it was produced.

Image and native WAV audio inputs are supported when the loaded model advertises
them. Encoded video is rejected before inference until a self-contained decoder
ships.

`validate_audio_bytes` checks an encoded attachment before model loading or MLX
initialization. It enforces the encoded-size and ten-minute duration bounds,
accepts only structurally valid RIFF/WAVE PCM16 or float32 audio, verifies sample
framing and metadata rates, and rejects non-finite float samples. It performs no
decoded or resampled sample allocation.

`GenerationStream` uses a bounded channel. Dropping or cancelling it closes the
receiver immediately, wakes any producer blocked by backpressure, and stops the
engine cooperatively. `cancel_and_wait` additionally waits until the submitted
job has left the model's dedicated inference thread. Dropping that wait does
not consume its completion observation; a caller may await it again.
