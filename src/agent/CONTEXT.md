# Agent context

The agent module is an in-memory orchestration boundary over native Emelex
generation. It does not own durable storage, terminal presentation, project
instructions, or model selection.

Invariants:

1. `Client` remains the production model implementation. `AgentModel` exists
   so the loop can be tested without MLX and adapted to another local
   scheduler.
2. A turn is sequential. One `&mut AgentSession` runs at most one model or
   tool step at a time.
3. Model text deltas must form a prefix of terminal answer text. The harness
   emits any missing terminal suffix before completion, so presentation
   concatenation equals the terminal field; a non-prefix fails the turn.
   Reasoning uses the same prefix/suffix reconciliation, including when no
   reasoning delta arrived before the terminal response. Only a terminal
   response containing answer text or tool calls becomes history.
4. Provider tool-call IDs are untrusted. The session mints UUIDv7 IDs and
   prevents reuse within its complete history.
5. Tool definitions are immutable after build and validated against the same
   bounded JSON Schema vocabulary used by generation. Proposed arguments are
   rechecked before dispatch.
6. Assistant tool calls and their results commit as one complete ordered batch.
   Before invocation, failures roll the batch back. Once an invocation may have
   produced effects, failures checkpoint a matching result for every call so
   history stays structurally complete and retry cannot silently repeat work.
7. Successful `AgentTurn::messages` and failure-time tool checkpoints are the
   durable message boundaries. A memory adapter journals invocation state,
   persists complete batches, and replays them through
   `AgentSessionBuilder::history`.
8. Approval is exact, asynchronous, one-shot, and process-local. The default
   policy denies. No persistent grant exists in this module. Approval and
   denial reasons are non-empty, bounded, single-line text.
9. Workspace tools are descriptor-anchored, reject parent traversal, never
   follow symlinks, and bound traversal, input, output, and process time.
10. Failed turns roll back history and newly issued UUIDs to the latest safe
    checkpoint. With no possible tool effect that is the entry cursor. After a
    tool starts, the complete call/result checkpoint and its UUIDs remain in
    history across cancellation, fatal tools, sink failure, protocol errors,
    and model-round exhaustion.
11. `web_fetch`, `datetime`, and provider-backed `web_search` are explicit
    opt-ins. Network tools require approval and never select hidden vendors.
    Built-in HTTP clients identify the exact compiled package version.
    `WebSearchResult::new` is the public construction boundary; the generic
    tool still validates provider result count, bounds, and HTTP(S) URLs. A
    provider future may be dropped when its supplied cancellation fires; it
    cannot detach I/O/effects, and internally spawned work must observe the
    token.
12. File and shell registration are independent. Resolved project policy can
    remove shell without removing file tools. Shell timeout/output and web
    response ceilings enter through public builder setters and remain bounded
    by hard module maxima. Shell timeout defaults remain conservative while
    the explicit hard ceiling permits one bounded 20-minute build/test gate.
13. `run_message` accepts only user-role messages and preserves multimodal
    content in the returned commit batch. Failure before possible tool effects
    rolls back that complete attachment message; a later tool checkpoint
    retains it with the structurally complete call/result batch.
14. Shell approval is not a sandbox promise. The child process executes on the
    host with a cleared environment plus fixed locale and PATH/HOME captured at
    tool construction. Values keep only absolute non-traversing paths; PATH
    also receives fixed system fallbacks. Domain-separated digests enter
    durable tool identity, so path/order configuration drift fails resume
    authority comparison without exposing raw directories. Executables and
    configuration files inside those directories remain mutable host state.
15. Session generation options are the base; fields set by one turn override
    them, including thinking policy. Unset turn fields retain session values.
16. A configured `system_prompt` and a resumed system message are mutually
    exclusive. The builder rejects ambiguous instruction precedence.
17. Filesystem mutation writes and syncs a descriptor-relative sibling temp,
    atomically installs it, then syncs the parent. Existing-target swaps verify
    the previous inode before deletion.
18. A fallible event sink is a cancellation boundary. Sink failure cancels the
    model stream and commits neither history nor newly issued tool IDs. For a
    native `GenerationStream`, every early model, sink, or protocol exit closes
    receiver backpressure and awaits inference-job completion before returning.
    Custom `AgentGeneration::new` streams have no completion hook and are
    drop-only; their implementations must not detach work that survives Drop.
19. `authority_snapshot` and `build` share one borrowed resolution path. The
    versioned snapshot contains canonical workspace path plus descriptor
    device/inode identity, exact name-sorted tool definitions, built-in
    enablement, and effective configured ceilings. Approval policy is
    intentionally absent and the builder remains reusable.
20. A loaded `Client` reports exact-template system-role, tool,
    reasoning-history, and thinking-toggle capabilities. Authority resolution
    fails closed before execution when configured or resumed state requires a
    capability the checkpoint did not prove. Thinking-on agents require both
    reasoning dimensions because generated reasoning becomes later-round
    history. Per-turn overrides are revalidated before the input checkpoint.
    Explicit `Off`/`Auto` clears a session reasoning budget. With thinking not
    enabled, reasoning emitted by a template lacking history preservation stays
    in presentation output but is stripped from committed replay messages.
21. Every resolved tool has one immutable `ToolCancellationPolicy` recorded in
    `AgentAuthoritySnapshot`. `FinishOnceStarted` is the default and stops the
    remaining batch after checkpointing one terminal result. `Interruptible`
    is allowed only for invocation futures whose Drop cannot detach work.
22. Approved write/edit work runs outside the async executor and has a
    join-on-Drop completion guard. Forced future cancellation cannot return
    while a bounded atomic mutation remains active.

Cancellation wins ties through biased `tokio::select!`. Cancelling model work
cancels `AgentGeneration`; a native `GenerationStream` sets its inference
cancellation flag, closes the bounded receiver, and establishes a completion
barrier before the turn returns.
`AgentCancellation::cancelled()` is public for tool/provider I/O. The shell
tool adds a drop guard that kills its dedicated process group.

The binary should treat `AgentEvent` as presentation only. For durable
sessions, the memory adapter journals active turns and tool invocation state,
persists each complete tool batch before another model round, closes terminal
success or failure atomically, and reconstructs the harness from strict
replayed messages.
