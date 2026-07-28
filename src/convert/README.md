# convert

Pure translation between rig's completion types and the engine's chat
types. No I/O and no Session dependency — the whole module is
unit-tested without a loaded model (capabilities are passed in as flags).

Key mappings (`mod.rs` orchestrates; details in submodules):

- **System block** (`mod.rs`): preamble + every `Message::System` +
  tool-choice instruction + output-schema instruction merge into one
  leading `role: system` turn, joined by blank lines — chat templates
  mishandle multiple system turns.
- **Messages** (`message.rs`): user text/media parts convert in order
  (media as raw or base64-decoded bytes only); adjacent text parts
  (prompt text + attached documents) coalesce into one part, since
  chat templates mishandle multi-part text arrays; `ToolResult` content
  splits into its own `role: "tool"` turn keyed by `call_id` (falling
  back to `id`); assistant history round-trips text, tool calls, and
  text reasoning. Content without an engine representation (assistant
  images, tool-result images, encrypted/redacted/summary reasoning)
  fails with `UnsupportedContent`; it is never silently discarded.
- **Options** (`options.rs`): request temperature/max_tokens plus an
  `additional_params` overlay (`temperature`, `max_tokens`, `top_p`,
  `top_k`, `seed`, `enable_thinking`, `reasoning_budget_tokens`,
  `prompt_cache`, `speculative_tokens`) over client defaults;
  precedence is overlay > request fields > client defaults,
  explicit `enable_thinking: false` clears an inherited reasoning budget,
  `max_tokens` is clamped to a sane ceiling, non-positive `top_k`
  disables the cutoff, and `speculative_tokens` normalizes `0` to off
  and clamps to the engine's draft-depth ceiling (8) so an untrusted
  request value cannot exceed it. `tool_choice`
  policy: `None` drops tools; `Required` injects a best-effort system
  instruction; `Specific` additionally advertises only the named
  tools. `Required` and `Specific` reject an empty available-tool set.
  `None` is invalid with tool-call history, and `Specific` must
  include every historical tool because replay requires its declaration;
  these combinations fail with policy-specific diagnostics instead of an
  undeclared-history error. `output_schema` injects the JSON Schema into the
  system block
  (decoding is not grammar-constrained).
- **Reply** (`reply.rs`): engine reply → rig assistant content in
  emission order (reasoning, text, tool calls) plus usage mapping
  (`cached_tokens` → `cached_input_tokens`) and the speculative
  accounting mirror (engine `SpeculationStats` →
  `SpeculationStatsData`). A reply that never speculated carries `None`;
  otherwise the exact call's mirror travels on its raw response or
  terminal streaming response.

Conversion enforces a bounded request envelope aligned with the native API's
main limits:
4,096 messages/tool calls, 256 tools, 64 media parts, 128 MiB per message,
256 MiB aggregate content, 1 MiB per tool/output schema, and 8 MiB aggregate
tool schemas. Tool descriptions and call arguments have equivalent 1 MiB
single-item bounds; call IDs are capped at 4 KiB. Base64 and JSON inputs are
size-checked during streaming decode/serialization, before an unbounded
allocation. Tool-call/result ordering, IDs, names, and schemas validate before
inference.
