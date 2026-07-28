# Agent

`agent` is Emelex's native, provider-independent agent harness. It consumes the
same `GenerationRequest` and `GenerationEvent` types as `Client`, executes
validated tools, and keeps one complete ordered in-memory conversation.

## Public shape

- `AgentSessionBuilder` validates the canonical workspace, resumed history,
  tool declarations, and aggregate resource ceilings.
- `AgentSessionBuilder::authority_snapshot` resolves the same authority path as
  `build` without consuming the builder. Its versioned, serializable
  `AgentAuthoritySnapshot` records the canonical workspace and descriptor
  device/inode identity, exact sorted tool declarations and schemas, built-in
  enablement, and effective configurable ceilings. It deliberately excludes
  the one-shot approval policy. The built session exposes the same resolved
  value through `authority_snapshot`.
- `AgentSession::run_turn` streams typed `AgentEvent` values through an
  in-order callback and returns an `AgentTurn`. `run_message` is the same core
  operation for text, image, or audio user content. Encoded video fails closed
  until Emelex contains a bundled decoder.
- Answer deltas always concatenate to the terminal answer. If a model bridge
  supplies only an exact prefix, the harness emits the missing terminal suffix
  before model/turn completion; a non-prefix response fails the turn.
- Reasoning deltas follow the same prefix contract. Terminal-only reasoning or
  a missing terminal suffix is emitted before model completion; a non-prefix
  fails the turn.
- `run_*_with_options` overlays per-turn `GenerationOptions` on session
  defaults. Explicit `Off` or `Auto` clears a session reasoning budget;
  an explicit per-turn budget still wins and remains invalid with `Off`.
  Capability checks run on the merged options before the preflight checkpoint.
  `try_run_*` accepts a fallible event consumer; consumer failure drops active
  custom generation or cancels and awaits native inference completion, then
  rolls back the turn.
- `AgentTurn::messages` is the exact message batch committed by that turn. A
  durable adapter should persist this batch transactionally; text and reasoning
  deltas are presentation state, not replay state.
- Loaded `Client` capabilities are checked during authority resolution.
  Builders fail before execution when the exact checkpoint template cannot
  preserve a configured system prompt, tool declarations, or resumed tool
  history. Resumed reasoning requires proven reasoning-history preservation.
  Thinking-on agent sessions require both a proven thinking toggle and
  reasoning-history preservation because later rounds replay model reasoning.
  If a non-thinking template nevertheless emits reasoning and the checkpoint
  cannot preserve it, events and `AgentTurn::response` retain the presentation
  span while committed/replayed assistant messages omit it.
- `AgentTool` and `ApprovalPolicy` are asynchronous library extension points.
  `ToolCancellationPolicy::FinishOnceStarted` is the conservative default:
  after `ToolStarted`, the harness waits for one terminal result and then stops
  the remaining batch. `Interruptible` is opt-in and valid only when dropping
  `invoke` cannot leave host work or effects detached. The resolved policy is
  immutable durable authority.
- `AgentCancellation` cancels model generation, pending approvals, and
  cooperative tools. Its public `cancelled()` future lets extensions select
  cancellation directly against I/O. Every early model, sink, or protocol exit
  cancels a native `GenerationStream`, closes its bounded receiver, and waits
  until its inference job leaves the dedicated worker. Custom streams wrapped
  with `AgentGeneration::new` expose no completion hook and remain drop-only;
  custom implementations must not detach work that can outlive Drop.

The harness validates every complete request itself before invoking an
`AgentModel`, including alternate implementations. An extension cannot bypass
message, media, schema, metadata, or tool-argument aggregate ceilings.

Every model-produced tool call is treated as an untrusted proposal. The
harness validates its registered name and JSON Schema arguments, replaces its
provider-local ID with a UUIDv7, and preserves that ID through approval,
execution, events, assistant history, and the matching tool result. A tool-call
batch enters history only when every call has one matching result. Once any
tool may have run, a failure checkpoints a structurally complete
assistant-call/result batch into history so retry cannot silently repeat host
effects. Failures before invocation roll back the proposed batch. Resumed
history rejects duplicate, mismatched, unresolved, or declaration-free calls.

## Built-in tools

Sessions include seven workspace tools by default:

- `read_file`
- `list_directory`
- `find_files`
- `grep`
- `write_file`
- `edit_file`
- `shell`

Network and clock capabilities are explicit builder opt-ins. `web_fetch`
enforces HTTP(S), timeout, and decoded-body limits without ambient
proxies, cookies, or URL credentials. Network request identity carries the
exact compiled Emelex package version. `datetime` returns RFC 3339 time at a
requested fixed UTC offset. Generic `web_search` appears only when the
application supplies a `WebSearchProvider`; Emelex never selects a search
vendor implicitly. Both network tools require one-shot approval. Automatic
redirects are disabled: a redirect destination is returned to the model and
requires a new `web_fetch` call and approval.

Providers construct bounded records with `WebSearchResult::new`. The harness
then validates result count, field sizes, and HTTP(S) URLs before returning
provider output to the model. Applications should discard malformed
provider-side entries rather than let one bad remote result fail the whole
search. A provider search future is cancellation-safe: Emelex may drop it when
the supplied cancellation fires, so it must not detach I/O or effects, and
internally spawned work must observe that cancellation.

File and shell authority are independently configurable:

```rust,no_run
# fn configure(client: emelex::Client) -> Result<(), emelex::agent::AgentError> {
let session = emelex::agent::AgentSession::builder(client, ".")
	.include_file_tools(true)
	.include_shell_tool(false)
	.include_web_fetch(true)
	.web_response_bytes(256 * 1024)
	.shell_timeout_seconds(20)
	.shell_output_bytes(64 * 1024)
	.build()?;
# let _ = session;
# Ok(())
# }
```

This separation lets resolved project configuration reduce authority. A
disabled shell is not registered or advertised while file tools remain
available.

Durable chat metadata should persist the authority snapshot beside its session
configuration. Re-resolving a builder and comparing snapshots proves whether
workspace or tool authority drifted before resume.

Filesystem traversal starts from an open canonical root descriptor and uses
`openat` with `O_NOFOLLOW` for every path component. Parent traversal is
rejected. Absolute paths outside the root are available only after a one-shot
approval and receive the same descriptor-by-descriptor checks from filesystem
root. Recursive tools never follow symlinks and skip likely-secret paths unless
the exact call was approved.

Writes, edits, and host shell execution always require approval. Reads require
approval for likely-secret or outside-root paths. The default policy denies
every requested approval, and decisions are never cached.

Writes and edits use a sibling temporary file, sync it, atomically install it
with descriptor-relative `renameatx_np`, remove the replaced inode, then sync
the parent directory. Edits verify the inode read is the inode swapped out. If
a raced replacement cannot be rolled back, cleanup leaves the prior inode
under its sibling recovery name instead of risking data loss.
Mutation work runs on a blocking worker. Once started it completes before a
cooperative cancellation returns; dropping the invocation future joins that
bounded worker, so an atomic publish cannot continue invisibly afterward.

Shell is deliberately a host tool, not a sandbox. It clears the inherited
environment, then restores only construction-time PATH and HOME values
sanitized to absolute non-traversing paths, fixed system PATH fallbacks, and a
fixed locale. Domain-separated PATH/HOME digests are part of durable tool
identity, so resume detects environment path/order drift without persisting
raw directories. Directory contents can still change between calls. It runs
in a dedicated process group, captures bounded head-and-tail output while
continuing to drain pipes, and kills the process group on timeout,
cancellation, dropped execution, or a normal shell-leader exit. Members
remaining in that process group cannot outlive the tool call; an approved host
command can still daemonize into a new session or process group. The
configurable hard timeout ceiling is 20 minutes; defaults remain shorter.
Shell execution is not a security sandbox.

`system_prompt` inserts a system message only when resumed history does not
already contain one. Supplying both is rejected instead of silently merging
instructions.

## Minimal use

```rust,no_run
use std::sync::Arc;

use emelex::{
	Client,
	agent::{AgentCancellation, AgentSession, ApprovalContext, ApprovalDecision, ApprovalPolicy},
};

struct InteractiveApprovals;

#[async_trait::async_trait]
impl ApprovalPolicy for InteractiveApprovals {
	async fn decide(&self, context: &ApprovalContext) -> ApprovalDecision {
		eprintln!("approval required: {}", context.reason);
		ApprovalDecision::Deny {
			reason: "example policy".to_string(),
		}
	}
}

# async fn run(client: Client) -> Result<(), emelex::agent::AgentError> {
let mut session = AgentSession::builder(client, ".")
	.approval_policy(Arc::new(InteractiveApprovals))
	.build()?;
let cancellation = AgentCancellation::new();
let turn = session
	.run_turn("Summarize this project.", &cancellation, |event| {
		let _ = event;
	})
	.await?;
println!("{}", turn.response.text);
# Ok(())
# }
```
