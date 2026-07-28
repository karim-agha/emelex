#!/usr/bin/env python3
"""Run Emelex's exact three-step MTP parity gate within 20 minutes."""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

PARTY_TIMEOUT_SECONDS = 20 * 60
MODEL_ENV = "EMELEX_TEST_MODEL"
GOLDENS_ENV = "EMELEX_PARITY_GOLDENS"
SENTINEL_ENV = "EMELEX_PARTY_SENTINEL"
TEST_NAME = "engine::parity::tests::mtp_logit_parity_gate"
IMPLEMENTATION_ID = "emelex-qwen3.5-mtp-dense-bf16-v1"


def _required_directory(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} must name the certified fixture directory")
    path = Path(value).expanduser()
    if not path.is_dir():
        raise ValueError(f"{name} is not a directory: {path}")
    return path.resolve()


def run_with_timeout(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float = PARTY_TIMEOUT_SECONDS,
) -> int:
    """Run one command in a fresh process group and kill the group on timeout."""
    print(f"$ {shlex.join(command)}", flush=True)
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            start_new_session=True,
        )
    except OSError as error:
        print(f"party gate could not start: {error}", file=sys.stderr)
        return 127

    try:
        returncode = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        print(
            f"party gate exceeded hard {timeout_seconds:g}-second deadline",
            file=sys.stderr,
        )
        return 124
    except KeyboardInterrupt:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        return 130

    if returncode < 0:
        return 128 - returncode
    return returncode


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    certification = repo / "tests/fixtures/mtp_certification.json"
    try:
        model = _required_directory(MODEL_ENV)
        goldens = _required_directory(GOLDENS_ENV)
        if not certification.is_file():
            raise ValueError(f"certification manifest is missing: {certification}")
    except ValueError as error:
        print(f"party gate refused to start: {error}", file=sys.stderr)
        return 2

    cargo = os.environ.get("CARGO", "cargo")
    command = [
        cargo,
        "test",
        "--release",
        "--locked",
        "--offline",
        "--lib",
        TEST_NAME,
        "--",
        "--ignored",
        "--exact",
        "--nocapture",
        "--test-threads=1",
    ]
    environment = os.environ.copy()
    environment[MODEL_ENV] = str(model)
    environment[GOLDENS_ENV] = str(goldens)
    with tempfile.TemporaryDirectory(prefix="emelex-party-") as temporary:
        sentinel = Path(temporary) / "completed"
        environment[SENTINEL_ENV] = str(sentinel)
        status = run_with_timeout(command, cwd=repo, env=environment)
        if status == 0:
            try:
                completed = sentinel.read_text(encoding="utf-8")
            except OSError:
                completed = ""
            if completed != f"{IMPLEMENTATION_ID}\n":
                print(
                    "party gate exact test did not complete; filter drift or skip detected",
                    file=sys.stderr,
                )
                status = 3
    if status != 0:
        print(f"party gate failed with exit status {status}", file=sys.stderr)
    return status


if __name__ == "__main__":
    sys.exit(main())
