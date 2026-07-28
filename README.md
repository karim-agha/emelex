# Emelex

Emelex is a library and command-line toolkit for local AI inference on Apple
Silicon. It loads MLX checkpoints directly, explores compatible models across
the Hugging Face catalog, manages owned snapshots and explicit external-model
links, and provides a self-contained coding-agent chat harness.

Emelex 1.0 supports Apple Silicon on macOS 26.5 or newer.

## Quick start

```sh
cargo install --locked --path .
emelex hub search qwen --require interaction:tools
emelex hub capabilities
emelex hub download mlx-community/Qwen3.5-4B-4bit
emelex model import /path/to/checkpoint --name local-name
cd your-project
emelex chat
```

All owned data defaults to `~/.emelex`. Override it with `--home PATH`, the
library builder, or `EMELEX_HOME`.

`chat` accepts `--max-tokens`, `--temperature`, and `--thinking
auto|on|off` when creating a Session. `--with-web-search` adds a
one-shot-approval-gated DuckDuckGo HTML search provider. Resumed Sessions keep
their original model, generation settings, prompt, and tool authority.
`-C PATH` (also `--directory` or `--root`) selects the Emelex invocation root
for workspace-scoped Sessions, Knowledge, tools, and project-configuration
discovery; it does not change the process working directory used to resolve
unrelated relative CLI path arguments.

## Commands

```text
emelex chat
emelex resume [PROMPT] [--session SESSION]
emelex generate [PROMPT]
emelex hub capabilities
emelex hub search [QUERY]
emelex hub inspect [NAMESPACE/]REPO
emelex hub download [NAMESPACE/]REPO
emelex hub auth login [--token-stdin]
emelex hub auth status|logout
emelex model import PATH [--name NAME] [--move|--symlink]
emelex models list|import|default|update|remove|verify|gc|path
emelex memory status|export|gc|work|failures|retry
emelex memory sessions list|show|export|recover|delete
emelex memory knowledge list|search|show|history|activate|pin|unpin|forget
emelex doctor
```

`resume` without `--session` selects the most recent Session in the invocation
workspace. `chat --resume` does the same; an explicit chat target uses
`chat --resume=SESSION` so a following positional argument remains the prompt.

`generate` emits raw model output by default. `--agent` enables the tool loop.
`--json` emits newline-delimited event objects. JSON `chat` and `resume` begin
with a `session` record containing `session_id`, immutable `model_snapshot`,
and `resumed`, before any recovery or agent event. Non-interactive `chat` and
`resume` accept a positional prompt or bounded UTF-8 text on stdin.

`emelex hub auth` manages the optional token stored in owner-only global
`<home>/config.toml` as `[hub].token`. Project `.emelex.toml` files may not
contain Hub credentials. At the CLI boundary, presence of the standard
`HF_TOKEN` environment variable overrides stored authentication: a nonempty
value supplies credentials and an explicitly empty value disables
authentication for that invocation. When the variable is absent, the stored
token is used when present; otherwise the CLI is anonymous.

The Rust library never reads `HF_TOKEN`. Explicit redacted `HubCredentials`
passed to `EmelexBuilder` take precedence over the stored global token,
followed by anonymous access. The secret is extracted separately from
configuration and is never present in the resolved `Config` value.
Authorization headers are marked sensitive, and tokens are redacted from
diagnostics and logs. The raw stored token exists only in global configuration.
Transport errors strip request and redirect URLs before entering public
errors, so signed download query credentials cannot leak through diagnostics.
HTTP error bodies are suppressed for authenticated requests, cross-origin
redirects, and final URLs with queries. Hub clients disable automatic
`Referer` generation, and HTTPS-origin clients refuse redirects to HTTP.

`emelex model import PATH` derives its local name from the canonical directory;
`--name NAME` overrides it. The command copies an immutable runtime-only
snapshot by default. `--move` first publishes that same verified copy, then
retires only selected source files that have not changed; ignored files remain,
so the source directory may remain and cleanup warnings are possible.
`--symlink` stores a managed record pointing to the canonical external model.
Its target is caller-owned, mutable, and not guaranteed to remain available, so
every resolve revalidates its link, runtime inventory, and full content hashes;
every load repeats that work before compatibility checks and runtime loading.
Removing a linked model removes only its Emelex record, never the external
target.

`hub capabilities` lists the predicates backed by remote evidence; stronger
installed-only claims such as runtime-verified MTP are intentionally
unavailable during remote search. CLI Hub searches always use Hugging Face's
MLX catalog filter, preserve optional user search text, and return only
candidates fitting this Mac's Metal budget and available Emelex Home storage.
When interactive onboarding needs image or audio input, remote discovery uses
Hugging Face's advertised-input metadata only. That is a candidate hint, not a
runtime claim; every downloaded checkpoint must pass local capability
certification before Emelex selects it. Onboarding presents one bounded result
page at a time while the user explicitly follows opaque next-page cursors,
including after a candidate fails local certification.

## Library

```rust,no_run
use emelex::{Emelex, generation::GenerationRequest, model::ModelRef};

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let emelex = Emelex::builder().build()?;
let models = emelex.models()?;
let installed = models.resolve(&ModelRef::parse(
	"mlx-community/Qwen3.5-4B-4bit",
)?)?;
let model = models.load(&installed, &Default::default())?;
let reply = model.generate(GenerationRequest::text("Hello")).await?;
println!("{}", reply.text);
# Ok(())
# }
```

## Security

Emelex is an agent harness, not a sandbox. Reads of likely-secret files,
outside-root paths, writes, edits, and shell commands require approval in
interactive sessions. Approved shell commands execute on the host through
`/bin/sh -c` and can access files, processes, and the network. Web tools accept
HTTP(S) targets, including private and local addresses, but agent web requests
send no ambient cookies or credentials and enforce response, redirect, and
time limits.

Approval grants exist only for the current process and never resume.
For `shell`, `web_search`, and `web_fetch`, the complete canonical JSON action
must fit the 2,048-character approval preview or Emelex denies it and asks the
model to split the request. Other tool previews may show bounded head and tail
content plus omitted length and full SHA-256.

CLI web search is off unless `--with-web-search` is supplied and resolved
configuration permits web tools. It sends the approved query to DuckDuckGo's
fixed no-JavaScript HTML endpoint without ambient proxy settings, cookies, or
credentials. Search markup is an external, unstable contract; known
interactive challenges report provider unavailability, while unknown markup
returns no invented results. Search-result URLs are untrusted model input.

## Building

The build compiles vendored MLX and mlx-c sources, then embeds a compressed
Metal library in the Rust binary/library. The first Emelex home used in a
process extracts that library atomically under
`cache/runtime/mlx/<digest>/mlx.metallib` before any MLX initialization.

Source builds require Xcode 26.5 or newer plus its separately downloadable
Metal Toolchain. Install that compiler component when `xcrun metal --version`
reports it missing:

```sh
xcodebuild -downloadComponent MetalToolchain
```

The build honors an explicit `TOOLCHAINS` selection. When the downloaded
Metal compiler is visible to plain `xcrun` but not to `xcrun --sdk macosx`,
Emelex reads its installed toolchain identifier and selects it only for the
vendored native build.

Docs.rs builds use a checked-in bindgen snapshot and skip native compilation;
this documentation-only path does not make non-Apple targets runnable.

```sh
cargo +nightly fmt --check
cargo clippy --all-targets --all-features --locked -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps
cargo test --all-targets --all-features --locked -- --test-threads=1
cargo deny check licenses
python3 tools/update_rust_licenses.py --check
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_native_invariants.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_mtp_fixture_safetensors.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_release_audit.py
EMELEX_REQUIRE_PHYSICAL_GPU=1 cargo test --test runtime \
  metal_completion_callback_failure_returns_error_without_aborting -- --nocapture
```

`rust-toolchain.toml` pins rustc 1.97.0 and its exact release artifact.
The license-bundle check also verifies the full rustc commit, Apple-Silicon
`compiler_builtins` archive presence, and compiler-runtime license digest.

Release candidates must also build, package, and pass the deterministic residue
audit:

```sh
tools/release_gate.sh
```

It scans tracked filenames and contents, package members and contents, and
release executable strings. Former-product residue, private/source/home paths,
unsafe archive members, generated Python/cache state, and harness worktrees are
release failures. The offline wrapper rejects ambient Rust flags,
compiler/wrapper overrides, and target overrides; pins the repository
toolchain, Apple-Silicon target, and target directory; and remaps the entire
dependency graph's source, Cargo-registry, Rustup, and resolved toolchain paths
to stable release labels before building. Cargo runs from a neutral directory
under an empty, explicit environment and config-free Cargo Home, with vendored
AWS-LC selected. The release host must provide CMake at
`/opt/homebrew/bin/cmake`. Before packaging, it removes the complete Cargo
target tree so host tools, transitive dependencies, and native objects cannot
be reused from an ungated build. It then snapshots the package and builds the
executable before auditing package and binary against the unchanged worktree.
The Mach-O audit permits dynamic dependencies only under `/usr/lib/` and
`/System/Library/`.

The external dense-BF16 MTP certification gate is intentionally ignored by
ordinary tests. With the recorded fixture and goldens available, run:

```sh
EMELEX_TEST_MODEL=<fixture> EMELEX_PARITY_GOLDENS=<goldens> tools/party.py
```

The party (parity) runner verifies the checked-in certification hashes, runs
exactly three steps, and enforces a hard 20-minute process-group deadline.
