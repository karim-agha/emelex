# convert — context

## Terms

- **System block**: the single merged `role: system` turn assembled from
  preamble, System messages, tool-choice instruction, and output-schema
  instruction. Always first in the engine message list.
- **Capabilities**: what the loaded checkpoint can consume (vision,
  audio), passed as plain flags so conversion stays Session-free.
- **Best-effort instruction**: a constraint the engine cannot enforce at
  decode time (forced tool choice, structured output), expressed as
  system-prompt text instead of being rejected.

## Invariants

- Conversion is pure: same inputs, same outputs, no side effects.
- Everything representable converts. Any content without an engine
  representation fails with `UnsupportedContent`; conversion never
  silently changes conversation semantics by dropping content.
- Request size, media-count, tool-count, schema-size, and conversation
  protocol bounds are checked before inference. Encoded media length is
  bounded before base64 decoding allocates.
- Tool-call IDs round-trip unchanged: engine `ToolCall.id` → rig
  `ToolCall.id` → `ToolResult.call_id.unwrap_or(id)` → engine
  `tool_call_id`, so chat templates can pair calls with results.
- Tool calls are globally unique within a request, every call is resolved
  exactly once before ordinary conversation resumes, and advertised tool
  schemas pass the engine's bounded schema validator.
- Tool-choice filtering never silently removes declarations required by
  history. `None` rejects any tool-call history; `Specific` rejects historical
  tool names outside its allowlist with a policy-specific diagnostic.
- `Required` and `Specific` require at least one available tool; neither
  silently degrades to ordinary text generation.
- Explicit `enable_thinking: false` clears the inherited client reasoning
  budget. A simultaneously explicit budget remains invalid.
- Message order is preserved exactly; only System messages move (into
  the leading system block) and documents insert directly after it.
