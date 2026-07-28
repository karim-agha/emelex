# AGENTS.md

Guidance for every coding harness contributing to Emelex.

**Response style: caveman mode, always on.** Use
`agents/skills/caveman/SKILL.md`: terse fragments, no filler, full technical
accuracy. Documents, specs, ADRs, code, and product copy use normal prose.
Disable only when the user asks for normal mode.

## Repository layout

Keep harness-owned state under `agents/`, never in vendor-specific root
dotfolders.

- Context: `agents/CONTEXT.md`
- Skills: `agents/skills/`
- Worktrees: `agents/worktrees/`
- Module context: `src/<module>/README.md` and `src/<module>/CONTEXT.md`

All repository changes must be made from an isolated worktree under
`agents/worktrees/`; never edit the primary checkout directly. Commits from an
isolated worktree are normal; never add the `agents/worktrees/` directory
itself or harness-generated build state to repository contents.

## Non-negotiables

1. Ask when intent or architecture is materially ambiguous. During unattended
   work, use the narrowest reasonable interpretation and record it.
2. Prefer the simplest complete solution. Avoid speculative abstractions.
3. Do not change unrelated code. Report unrelated defects separately.
4. State uncertainty. Verify risky assumptions with small experiments.
5. Keep public library errors typed. `anyhow` belongs only in the binary.
6. Production Rust must not panic or use `unwrap`/`expect`.
7. Leave no formatter, compiler, Clippy, test, or rustdoc warnings.
8. Every top-level module under `src/` needs a sibling `README.md` and
   `CONTEXT.md`; update both when its public API, invariants, storage, or
   workflows change.
9. Preserve Emelex's single inference-thread ownership of MLX state.
10. Never add product-specific concepts, model tiers, hidden network access,
    persistent approval grants, or Emelex-owned internal state outside the
    configured Emelex home. Caller-selected exports and explicit approved
    workspace, shell, or network effects are outside that storage invariant.

## Rust workflow

Load and follow:

- `agents/skills/rust-lsp`
- `agents/skills/rust-best-practices`
- `agents/skills/openai-codex-rust-patterns`

Required checks:

```sh
cargo +nightly fmt --check
cargo clippy --all-targets --all-features --locked -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps
cargo test --all-targets --all-features --locked -- --test-threads=1
cargo deny check licenses
python3 tools/update_rust_licenses.py --check
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_mtp_fixture_safetensors.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_native_invariants.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_release_audit.py
```

Use `cargo +nightly fmt` for formatting. Keep imports grouped as configured by
`rustfmt.toml`.

## Versioning

The first standalone release is `1.0.0`. Later releases follow Semantic
Versioning; ordinary commits do not change the version. For a release, keep
`Cargo.toml`, `Cargo.lock`, binary `--version`, package metadata, tags, and
release artifacts aligned.

## Native and release constraints

- Supported target: Apple Silicon, macOS 26.5 or newer.
- Preserve recorded MLX/mlx-c/vendor pins unless the task explicitly upgrades
  them.
- Never initialize MLX before Emelex installs its extracted metallib path.
- Do not run multiple live MLX test processes concurrently.
- Run the post-build MTP correctness gate through `tools/party.py`. It runs
  exactly three parity steps under a hard 20-minute process-group deadline.
  Missing fixtures, hash drift, missing MTP, timeout, or parity failure is a
  failing exit.
- Release/package gates must scan tracked files, archives, and binary strings
  for private paths or product-specific residue.
- Do not publish to a registry, create a remote repository, sign with a
  Developer ID, or notarize without explicit authorization.

Release candidates additionally require:

```sh
tools/release_gate.sh
```

The audit rejects former-product names, private/source/home paths, unsafe
archive members, Python bytecode, cache directories, and harness worktrees
across tracked source, package members and contents, and executable strings.
The offline wrapper rejects ambient Rust flags, compiler/wrapper overrides, and
target overrides; pins the repository toolchain, Apple-Silicon target, and
target directory; runs Cargo from a neutral directory with a config-free Cargo
Home and explicit environment; and remaps the complete dependency graph's
source, Cargo-registry, Rustup, and resolved toolchain paths before building.
It removes the complete Cargo target tree before packaging so every host,
transitive, and native artifact is rebuilt inside that environment, then audits
both the package and executable.
