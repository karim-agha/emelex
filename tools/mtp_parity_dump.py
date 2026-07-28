#!/usr/bin/env python3
"""MTP logit-parity golden dump.

Runs the pinned reference implementation over the dense BF16 fixture and
writes first-step plus recursive-step MTP logits as .npy goldens for the
Rust parity test (`src/engine/parity.rs::mtp_logit_parity_gate`).

Recursive-step fidelity: the recycle hidden fed to step N+1 is captured
from INSIDE the step-N `mtp_forward` call via a documented monkey-patch
on `lm.mtp.norm` (mtp_forward computes `recycle = mtp.norm(stack_out)`
and `logits = head(recycle)`). The hidden therefore comes from the same
cache-advancing layer pass that produced the golden logits — re-running
the layer stack manually without a cache would diverge from step 2
onward (the reconstruction would attend over no MTP KV history while
the goldens' pass did), poisoning every recursive golden.

PINNED ENVIRONMENT (record actual versions in the parity manifest):
  - mlx-lm fork:  AirRunner/mlx-lm @ 45f53582d64287aa875c1606e479f7f66c0afb58
      install from the clean local checkout prepared by
      tools/mtp_fixture_convert.sh; the dump verifies direct_url.json,
      origin, full HEAD, cleanliness, and installed/source Python bytes
  - mlx (Python): pin the version installed alongside the fork at dump
      time; this script refuses to run without MLX_PIN_ACK=<mlx version>
      so the pin lands in the manifest consciously.

Usage:
  MLX_PIN_ACK=$(python -c 'import mlx.core; print(mlx.core.__version__)') \
  python tools/mtp_parity_dump.py --model /path/to/composed-dense-fixture \
      --out /path/to/goldens --steps 3

Outputs (float32 .npy, one logits row [vocab] each):
  step0.npy   first MTP step: fused (backbone hidden of prompt[-1], next
              token = greedy backbone continuation)
  step1.npy.. recursive steps: fused (recycle hidden captured from inside
              the previous mtp_forward call, previous MTP greedy token)
  meta.json   prompt ids, greedy tokens per step, Python/MLX/mlx-lm
              versions, verified fork identity/tree digest, and config digest

`--steps` must be exactly 3 (first step + two recursive steps). The
certified gate deliberately fixes this workload so its runtime stays bounded.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

PROMPT = "The quick brown fox jumps over the lazy dog. Explain why."
FORK_SHA = "45f53582d64287aa875c1606e479f7f66c0afb58"
FORK_GIT = "https://github.com/AirRunner/mlx-lm.git"
FORK_PACKAGE_VERSION = "0.31.3"


class CertificationError(RuntimeError):
    """Fail-closed certification precondition or generation error."""


def git_output(checkout: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(checkout), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired as error:
        raise CertificationError(f"git {' '.join(args)} timed out") from error
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise CertificationError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def python_tree(root: Path) -> dict[str, bytes]:
    files = {}
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] == "examples":
            continue
        files[relative.as_posix()] = path.read_bytes()
    return files


def tree_digest(files: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for relative, contents in sorted(files.items()):
        encoded = relative.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def verify_mlx_lm_provenance() -> tuple[str, str]:
    distribution = importlib.metadata.distribution("mlx-lm")
    version = distribution.version
    if version != FORK_PACKAGE_VERSION:
        raise CertificationError(
            f"mlx-lm version {version!r} != pinned {FORK_PACKAGE_VERSION!r}"
        )
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise CertificationError(
            "mlx-lm has no direct_url.json; install the pinned local checkout"
        )
    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError as error:
        raise CertificationError(f"mlx-lm direct_url.json is invalid: {error}") from error
    parsed = urllib.parse.urlparse(direct_url.get("url", ""))
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        raise CertificationError(
            "mlx-lm direct URL is not a local file checkout of the pinned fork"
        )
    checkout = Path(
        urllib.request.url2pathname(urllib.parse.unquote(parsed.path))
    ).resolve()
    if git_output(checkout, "rev-parse", "--is-inside-work-tree") != "true":
        raise CertificationError(f"mlx-lm direct URL is not a git checkout: {checkout}")
    if git_output(checkout, "rev-parse", "HEAD") != FORK_SHA:
        raise CertificationError("mlx-lm checkout HEAD does not match the pinned full SHA")
    if git_output(checkout, "remote", "get-url", "origin") != FORK_GIT:
        raise CertificationError("mlx-lm checkout origin does not match the pinned repository")
    if git_output(checkout, "status", "--porcelain", "--untracked-files=all"):
        raise CertificationError("mlx-lm checkout is dirty; refusing certification")

    mlx_lm = importlib.import_module("mlx_lm")
    if mlx_lm.__file__ is None:
        raise CertificationError("imported mlx_lm module has no filesystem origin")
    installed_root = Path(mlx_lm.__file__).resolve().parent
    metadata_root = Path(distribution.locate_file("mlx_lm")).resolve()
    if metadata_root != installed_root:
        raise CertificationError(
            "imported mlx_lm module is not the package described by mlx-lm metadata"
        )
    if not installed_root.is_dir():
        raise CertificationError(
            f"installed mlx-lm package directory is missing: {installed_root}"
        )
    source_root = checkout / "mlx_lm"
    installed = python_tree(installed_root)
    source = python_tree(source_root)
    if installed.keys() != source.keys():
        missing = sorted(source.keys() - installed.keys())
        unexpected = sorted(installed.keys() - source.keys())
        raise CertificationError(
            "installed mlx-lm Python file set differs from pinned checkout: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    mismatched = [relative for relative in source if source[relative] != installed[relative]]
    if mismatched:
        raise CertificationError(
            "installed mlx-lm bytes differ from pinned checkout: "
            f"{mismatched[:8]}"
        )
    return version, tree_digest(source)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    if args.steps != 3:
        print(
            "refusing to dump: --steps must be exactly 3 (first step plus two "
            "recursive steps; this is the certified bounded workload)",
            file=sys.stderr,
        )
        return 2

    out = Path(args.out)
    if out.exists():
        raise CertificationError(
            f"output path already exists; choose a fresh path for atomic publication: {out}"
        )
    out.parent.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx

    ack = os.environ.get("MLX_PIN_ACK")
    if ack != mx.__version__:
        raise CertificationError(
            f"refusing to dump: set MLX_PIN_ACK={mx.__version__} to acknowledge "
            "the mlx pin for the parity manifest"
        )
    mlx_lm_version, mlx_lm_tree_sha256 = verify_mlx_lm_provenance()

    from mlx_lm import load
    from mlx_lm.models import cache as lm_cache

    model, tokenizer = load(args.model)
    lm = model.language_model if hasattr(model, "language_model") else model
    if not hasattr(lm, "mtp"):
        raise CertificationError("fixture has no MTP module under the pinned fork")

    ids = tokenizer.encode(PROMPT)
    inputs = mx.array([ids])

    # Backbone pass: the fork's TextModel.__call__ supports
    # return_hidden=True, returning (logits, pre-norm hidden) from one
    # pass - the exact hidden mtp_forward consumes (fork qwen3_5.py),
    # with the head resolution (tied vs lm_head) handled by the fork
    # itself.
    backbone_cache = lm_cache.make_prompt_cache(lm)
    logits, hidden = lm(inputs, cache=backbone_cache, return_hidden=True)
    next_token = int(mx.argmax(logits[0, -1]).item())

    import numpy as np

    # A missing MTP layer stack is a hard error: silently passing a None
    # cache to mtp_forward would produce CACHELESS goldens whose recursive
    # steps attend over no KV history - subtly wrong, and exactly the
    # divergence class the gate exists to catch.
    if not hasattr(lm.mtp, "layers"):
        raise CertificationError(
            "refusing to dump: lm.mtp has no `layers` attribute under the pinned "
            "fork - the certified cache-composition point moved"
        )
    mtp_cache = lm_cache.make_prompt_cache(lm.mtp)

    # Documented monkey-patch (recursive-step fidelity, see module doc):
    # wrap lm.mtp.norm so each mtp_forward call records its recycle hidden
    # (norm output) - the exact tensor the golden logits were computed
    # from, produced by the same cached layer pass. No manual layer-stack
    # re-run, no parallel-cache bookkeeping to drift.
    class NormTap:
        def __init__(self, norm):
            self._norm = norm
            self.last = None

        def __call__(self, x):
            out = self._norm(x)
            self.last = out
            return out

    tap = NormTap(lm.mtp.norm)
    lm.mtp.norm = tap

    stage = Path(tempfile.mkdtemp(prefix=f".{out.name}.tmp-", dir=out.parent))
    try:
        prev_hidden = hidden[:, -1:, :]
        token = next_token
        greedy_tokens = [next_token]
        for step in range(args.steps):
            tap.last = None
            step_logits = lm.mtp_forward(prev_hidden, mx.array([[token]]), mtp_cache)
            row = np.asarray(step_logits[0, -1].astype(mx.float32))
            np.save(stage / f"step{step}.npy", row)
            token = int(row.argmax())
            greedy_tokens.append(token)
            # Recycle hidden for the next recursive step, captured from inside
            # this step's mtp_forward via the norm tap.
            if tap.last is None:
                raise CertificationError(
                    "mtp.norm was never called inside mtp_forward - the monkey-patch "
                    "point moved under the pinned fork; recertification is required"
                )
            prev_hidden = tap.last[:, -1:, :]

        config_digest = hashlib.sha256(
            Path(args.model, "config.json").read_bytes()
        ).hexdigest()
        (stage / "meta.json").write_text(
            json.dumps(
                {
                    "prompt_ids": ids,
                    "greedy_tokens": greedy_tokens,
                    "python_version": platform.python_version(),
                    "mlx_version": mx.__version__,
                    "mlx_lm_version": mlx_lm_version,
                    "mlx_lm_source": {
                        "kind": "local_git_checkout",
                        "repository": FORK_GIT,
                        "revision": FORK_SHA,
                    },
                    "mlx_lm_tree_sha256": mlx_lm_tree_sha256,
                    "config_sha256": config_digest,
                    "steps": args.steps,
                },
                indent=2,
            )
            + "\n"
        )
        os.replace(stage, out)
    finally:
        lm.mtp.norm = tap._norm
        if stage.exists():
            shutil.rmtree(stage)
    print(f"wrote {args.steps} golden steps to {out}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CertificationError as error:
        print(f"certification failed: {error}", file=sys.stderr)
        sys.exit(2)
