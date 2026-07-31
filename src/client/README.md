# Client

`Client` is a cheaply cloneable handle to one canonical MLX checkpoint loaded
on one dedicated inference thread. Native APIs are primary:

- `generate(GenerationRequest)` returns a complete typed response;
- `stream(GenerationRequest)` returns a bounded cancel-on-drop stream;
- capability accessors describe the actually loaded checkpoint.

MTP counters belong to the call that produced them:
`GenerationResponse::speculation` carries native accounting, and Rig raw
responses carry the equivalent field. There is no client-global "latest"
snapshot, so overlapping callers cannot observe one another's counters.

Native streams also emit exact `GenerationEvent::Progress` snapshots: prompt
tokens before context validation fails, resolved cache use before prefill, and
one cumulative completion-token advance per generated token. Progress shares
the bounded stream and its backpressure; it is never inferred from text bytes.

The inference queue is bounded and non-blocking at admission. Saturation
returns `Error::InferenceBusy`; a disconnected worker returns
`Error::InferenceChannel`. Dropped futures and streams cancel cooperatively.
Queued work checks cancellation before touching MLX; stream cancellation
closes its receiver to wake a backpressured producer.

Native and Rig streams reconcile answer deltas against the validated terminal
answer. After raw tool markup begins, later answer-looking bytes stay withheld
until complete parsing decides whether the span is an accepted structured call
or visible rejected markup. The validated suffix is emitted before tool and
terminal events; a non-prefix terminal answer fails with a stream protocol
error.

`ClientBuilder` configures generation defaults, queue capacity, prompt cache,
reasoning, and MTP draft depth. Draft depth `0` disables speculation; values
above 8 are rejected. A request or Rig agent that explicitly disables thinking
also clears an inherited reasoning budget; supplying a new budget with thinking
off remains invalid. Model paths canonicalize before the client stores them.

Loaded reasoning capability is two-dimensional:
`supports_reasoning_history` proves explicit spans survive a follow-up turn,
while `supports_thinking_toggle` proves enabled and disabled renders differ.
Native and Rig requests share one pre-queue capability check for system
instructions, tool rounds, reasoning history, and thinking opt-in. A generic
`supports_reasoning` accessor is only their discoverability union.

With the optional `rig` feature, `model`, `agent`, `extractor`, and
`ReasoningExt` adapt the same loaded model to Rig. Rig model names are tracing
labels only.
