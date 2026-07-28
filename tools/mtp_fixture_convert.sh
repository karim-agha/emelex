#!/usr/bin/env bash
# emelex patch (not upstream): dense BF16 MTP fixture conversion.
# Runs on a certification machine with the artifacts and emits the evidence
# summarized by tests/fixtures/mtp_fixture_manifest.md.
#
# Pipeline:
#   1. download the pinned, single authoritative weight source
#      and the pinned standalone cross-check artifact (NEVER a weight
#      source; byte-count + sha256 verified before use)
#   2. install the pinned mlx-lm fork and convert (BF16, no quantization)
#   3. post-conversion namespace inspection: assert the MTP key/shape/
#      dtype set equals EXACTLY the 15-key language_model.mtp.* set, all
#      BF16 — no bare-root, no model.language_model.*, no
#      language_model.model.mtp.*, no partial/mixed sets
#   4. mandatory equivalence cross-check: standalone bare-root K vs
#      converted language_model.mtp.K, byte-equality or max-abs-diff;
#      unexplained mismatch ABORTS fixture acceptance
#   5. compatibility checks (hidden 2560, vocab 248,320, gated shapes,
#      shared embeddings, exactly one MTP layer)
#   6. record: command lines, tool versions, sha256 of every output file
#   7. build + verify the strip-mtp variant (tools/mtp_fixture_strip.py)
#
# Usage: tools/mtp_fixture_convert.sh <workdir>
# Prerequisites: python3, git, hf (huggingface_hub CLI), network access.

set -euo pipefail

# ---- pinned references ------------------------------------------------------
TARGET_REPO="Qwen/Qwen3.5-4B"
TARGET_REV="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
FORK_GIT="https://github.com/AirRunner/mlx-lm.git"
FORK_SHA="45f53582d64287aa875c1606e479f7f66c0afb58"
FORK_PACKAGE_VERSION="0.31.3"
STANDALONE_REPO="mlx-community/Qwen3.5-4B-MTP-bf16"
STANDALONE_REV="c05eea475606a952730182a0308d05c7cf7ccd77"
STANDALONE_FILE="model.safetensors"
STANDALONE_BYTES="241200628"
STANDALONE_SHA256="ea2e38acd5abffb27510bfe00064ffc2e0186c9bbbd536daa568a081612fb1ab"
# ----------------------------------------------------------------------------

WORKDIR="${1:?usage: mtp_fixture_convert.sh <workdir>}"
mkdir -p "$WORKDIR"
WORKDIR="$(cd "$WORKDIR" && pwd)"
if [[ "$WORKDIR" == "/" ]]; then
	echo "ABORT: workdir must not be filesystem root" >&2
	exit 1
fi
TOOLS_DIR="$(cd "$(dirname "$0")" && pwd)"

TARGET_DIR="$WORKDIR/target-$TARGET_REV"
STANDALONE_DIR="$WORKDIR/standalone-$STANDALONE_REV"
CONVERTED_DIR="$WORKDIR/converted"
STRIP_DIR="$WORKDIR/converted-strip-mtp"
VENV="$WORKDIR/venv"
RECORD="$WORKDIR/mtp_fixture_record.json"
FORK_DIR="$WORKDIR/mlx-lm-fork"

log() { printf '\n== %s\n' "$*"; }

# ---- 1. pinned downloads ---------------------------------------------------
log "downloading weight source $TARGET_REPO @ $TARGET_REV"
hf download "$TARGET_REPO" --revision "$TARGET_REV" --local-dir "$TARGET_DIR"

log "downloading cross-check artifact $STANDALONE_REPO @ $STANDALONE_REV (cross-check ONLY)"
hf download "$STANDALONE_REPO" "$STANDALONE_FILE" --revision "$STANDALONE_REV" \
	--local-dir "$STANDALONE_DIR"

log "verifying cross-check artifact byte count + sha256 against the pins"
ACTUAL_BYTES="$(wc -c <"$STANDALONE_DIR/$STANDALONE_FILE" | tr -d ' ')"
[ "$ACTUAL_BYTES" = "$STANDALONE_BYTES" ] || {
	echo "ABORT: standalone artifact is $ACTUAL_BYTES bytes, pinned $STANDALONE_BYTES" >&2
	exit 1
}
ACTUAL_SHA="$(shasum -a 256 "$STANDALONE_DIR/$STANDALONE_FILE" | cut -d' ' -f1)"
[ "$ACTUAL_SHA" = "$STANDALONE_SHA256" ] || {
	echo "ABORT: standalone artifact sha256 $ACTUAL_SHA != pinned $STANDALONE_SHA256" >&2
	exit 1
}

# ---- 2. pinned fork + conversion -------------------------------------------
log "installing pinned mlx-lm fork @ $FORK_SHA"
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --quiet --upgrade pip
if [[ ! -e "$FORK_DIR" ]]; then
	git clone "$FORK_GIT" "$FORK_DIR"
elif [[ ! -d "$FORK_DIR/.git" ]]; then
	echo "ABORT: existing fork path is not a git checkout: $FORK_DIR" >&2
	exit 1
fi
if [[ "$(git -C "$FORK_DIR" remote get-url origin)" != "$FORK_GIT" ]]; then
	echo "ABORT: existing fork checkout has an unexpected origin" >&2
	exit 1
fi
if [[ -n "$(git -C "$FORK_DIR" status --porcelain --untracked-files=all)" ]]; then
	echo "ABORT: fork checkout is dirty before pinning" >&2
	exit 1
fi
git -C "$FORK_DIR" fetch --all --quiet
git -C "$FORK_DIR" checkout --detach --quiet "$FORK_SHA"
FORK_HEAD="$(git -C "$FORK_DIR" rev-parse HEAD)"
if [[ "$FORK_HEAD" != "$FORK_SHA" ]]; then
	echo "ABORT: fork HEAD $FORK_HEAD != pinned $FORK_SHA" >&2
	exit 1
fi
if [[ -n "$(git -C "$FORK_DIR" status --porcelain --untracked-files=all)" ]]; then
	echo "ABORT: fork checkout is dirty after pinning" >&2
	exit 1
fi
pip install --quiet "$FORK_DIR"
if [[ -n "$(git -C "$FORK_DIR" status --porcelain --untracked-files=all)" ]]; then
	echo "ABORT: installing mlx-lm modified the pinned source checkout" >&2
	exit 1
fi

PY_VERSION="$(python3 --version)"
MLX_VERSION="$(python3 -c 'import mlx.core; print(mlx.core.__version__)')"
MLXLM_VERSION="$(python3 -c 'import mlx_lm; print(mlx_lm.__version__)')"
if [[ "$MLXLM_VERSION" != "$FORK_PACKAGE_VERSION" ]]; then
	echo "ABORT: mlx-lm package $MLXLM_VERSION != pinned $FORK_PACKAGE_VERSION" >&2
	exit 1
fi
log "versions: $PY_VERSION / mlx $MLX_VERSION / mlx-lm $MLXLM_VERSION / fork $FORK_HEAD"

CONVERT_CMD=(python3 -m mlx_lm convert
	--hf-path "$TARGET_DIR" --mlx-path "$CONVERTED_DIR" --dtype bfloat16)
log "converting (BF16, no quantization): ${CONVERT_CMD[*]}"
if [[ -e "$CONVERTED_DIR" ]]; then
	echo "ABORT: converted output already exists; use a fresh workdir: $CONVERTED_DIR" >&2
	exit 1
fi
"${CONVERT_CMD[@]}"

# ---- 3-6. inspection + cross-check + compatibility + record ----------------
log "post-conversion namespace inspection, equivalence cross-check, compatibility"
python3 - "$CONVERTED_DIR" "$STANDALONE_DIR/$STANDALONE_FILE" "$RECORD" "$TOOLS_DIR" <<'PYEOF'
import hashlib, json, sys
from pathlib import Path

converted_dir = Path(sys.argv[1])
standalone_file = Path(sys.argv[2])
record_path = Path(sys.argv[3])
sys.path.insert(0, sys.argv[4])

from mtp_fixture_safetensors import (
    FixtureValidationError,
    load_unique_json,
    merge_unique_tensors,
    read_safetensors,
    require_mapped_reference_keys,
)

MTP_PREFIX = "language_model.mtp."
MTP_KEYS = {
    "language_model.mtp.fc.weight",
    "language_model.mtp.pre_fc_norm_embedding.weight",
    "language_model.mtp.pre_fc_norm_hidden.weight",
    "language_model.mtp.norm.weight",
    "language_model.mtp.layers.0.input_layernorm.weight",
    "language_model.mtp.layers.0.post_attention_layernorm.weight",
    "language_model.mtp.layers.0.self_attn.q_proj.weight",
    "language_model.mtp.layers.0.self_attn.k_proj.weight",
    "language_model.mtp.layers.0.self_attn.v_proj.weight",
    "language_model.mtp.layers.0.self_attn.o_proj.weight",
    "language_model.mtp.layers.0.self_attn.q_norm.weight",
    "language_model.mtp.layers.0.self_attn.k_norm.weight",
    "language_model.mtp.layers.0.mlp.gate_proj.weight",
    "language_model.mtp.layers.0.mlp.up_proj.weight",
    "language_model.mtp.layers.0.mlp.down_proj.weight",
}
FORBIDDEN_PREFIXES = ("mtp.", "model.language_model.", "language_model.model.mtp.")

def bf16_max_abs_diff(a, b):
    import numpy as np
    fa = np.frombuffer(a, dtype=np.uint16).astype(np.uint32) << 16
    fb = np.frombuffer(b, dtype=np.uint16).astype(np.uint32) << 16
    return float(np.abs(fa.view(np.float32) - fb.view(np.float32)).max())

failures = []
try:
    converted_records = merge_unique_tensors(
        sorted(converted_dir.glob("*.safetensors"))
    )
    _, standalone_records = read_safetensors(standalone_file)
    require_mapped_reference_keys(standalone_records, MTP_PREFIX, MTP_KEYS)
except FixtureValidationError as error:
    raise SystemExit(f"FIXTURE ACCEPTANCE FAILURE: {error}") from error

all_tensors = {
    name: (info["dtype"], info["shape"], data)
    for name, (info, data) in converted_records.items()
}

# 3. Namespace inspection: exactly the 15-key language_model.mtp.* set, all
# BF16; zero forbidden-namespace keys anywhere in the output.
mtp_keys = {k for k in all_tensors if k.startswith(MTP_PREFIX)}
if mtp_keys != MTP_KEYS:
    failures.append(
        f"MTP key set mismatch: missing {sorted(MTP_KEYS - mtp_keys)}, "
        f"unexpected {sorted(mtp_keys - MTP_KEYS)}"
    )
forbidden = sorted(
    k for k in all_tensors if any(k.startswith(p) for p in FORBIDDEN_PREFIXES)
)
if forbidden:
    failures.append(f"forbidden-namespace keys present: {forbidden[:10]}")
key_table = {}
for k in sorted(mtp_keys):
    dtype, shape, _ = all_tensors[k]
    key_table[k] = {"dtype": dtype, "shape": shape}
    if dtype != "BF16":
        failures.append(f"MTP tensor {k} dtype {dtype} != BF16")

# 4. Mandatory equivalence cross-check (bare-root K -> language_model.mtp.K).
crosscheck = {}
for name, (s_info, s_data) in standalone_records.items():
    s_dtype, s_shape = s_info["dtype"], s_info["shape"]
    mapped = MTP_PREFIX + name
    if mapped not in all_tensors:
        failures.append(f"cross-check: converted output lacks {mapped}")
        continue
    c_dtype, c_shape, c_data = all_tensors[mapped]
    if (s_dtype, s_shape) != (c_dtype, c_shape):
        failures.append(
            f"cross-check: {mapped} dtype/shape {(c_dtype, c_shape)} != "
            f"standalone {(s_dtype, s_shape)}"
        )
        continue
    if s_data == c_data:
        crosscheck[mapped] = {"byte_equal": True}
    elif s_dtype == "BF16":
        diff = bf16_max_abs_diff(s_data, c_data)
        crosscheck[mapped] = {"byte_equal": False, "max_abs_diff": diff}
        failures.append(
            f"cross-check: {mapped} not byte-equal (max_abs_diff {diff}) — "
            "unexplained mismatch aborts fixture acceptance"
        )
    else:
        crosscheck[mapped] = {"byte_equal": False}
        failures.append(f"cross-check: {mapped} not byte-equal (non-BF16)")

# 5. Compatibility checks.
try:
    config = load_unique_json(
        (converted_dir / "config.json").read_bytes(),
        str(converted_dir / "config.json"),
    )
except FixtureValidationError as error:
    raise SystemExit(f"FIXTURE ACCEPTANCE FAILURE: {error}") from error
if not isinstance(config, dict):
    raise SystemExit("FIXTURE ACCEPTANCE FAILURE: config.json root must be an object")
text = config.get("text_config", config)
if not isinstance(text, dict):
    raise SystemExit("FIXTURE ACCEPTANCE FAILURE: text_config must be an object")
compat = {
    "hidden_size": text.get("hidden_size"),
    "vocab_size": text.get("vocab_size"),
    "mtp_num_hidden_layers": text.get(
        "mtp_num_hidden_layers", config.get("mtp_num_hidden_layers")
    ),
    "mtp_use_dedicated_embeddings": text.get("mtp_use_dedicated_embeddings", False),
    "q_proj_shape": all_tensors.get(
        "language_model.mtp.layers.0.self_attn.q_proj.weight", (None, None, b"")
    )[1],
    "o_proj_shape": all_tensors.get(
        "language_model.mtp.layers.0.self_attn.o_proj.weight", (None, None, b"")
    )[1],
}
if compat["hidden_size"] != 2560:
    failures.append(f"hidden_size {compat['hidden_size']} != 2560")
if compat["vocab_size"] != 248320:
    failures.append(f"vocab_size {compat['vocab_size']} != 248320")
if compat["mtp_num_hidden_layers"] != 1:
    failures.append(f"mtp_num_hidden_layers {compat['mtp_num_hidden_layers']} != 1")
if compat["mtp_use_dedicated_embeddings"]:
    failures.append("mtp_use_dedicated_embeddings must be false")
if compat["q_proj_shape"] != [8192, 2560]:
    failures.append(f"mtp q_proj shape {compat['q_proj_shape']} != [8192, 2560]")
if compat["o_proj_shape"] != [2560, 4096]:
    failures.append(f"mtp o_proj shape {compat['o_proj_shape']} != [2560, 4096]")

# 6. Output digests.
def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

digests = {
    p.name: sha256_file(p) for p in sorted(converted_dir.iterdir()) if p.is_file()
}

record = {
    "mtp_key_table": key_table,
    "forbidden_namespace_keys": forbidden,
    "equivalence_crosscheck": crosscheck,
    "compatibility": compat,
    "converted_output_digests": digests,
    "acceptance": "PASS" if not failures else "FAIL",
    "failures": failures,
}
if record_path.exists():
    try:
        existing = load_unique_json(record_path.read_bytes(), str(record_path))
    except FixtureValidationError as error:
        raise SystemExit(f"FIXTURE ACCEPTANCE FAILURE: {error}") from error
    if not isinstance(existing, dict):
        raise SystemExit("FIXTURE ACCEPTANCE FAILURE: record root must be an object")
else:
    existing = {}
existing.update(record)
record_path.write_text(json.dumps(existing, indent=2) + "\n")

if failures:
    print("FIXTURE ACCEPTANCE FAILURE:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("namespace inspection, equivalence cross-check, compatibility: PASS")
PYEOF

# Merge pins/versions/command into the record. Every shell value travels as an
# argument so repository names and worktree paths cannot become Python source.
python3 - "$RECORD" \
	"$TARGET_REPO" "$TARGET_REV" \
	"$FORK_GIT" "$FORK_HEAD" \
	"$STANDALONE_REPO" "$STANDALONE_REV" "$STANDALONE_FILE" \
		"$STANDALONE_BYTES" "$STANDALONE_SHA256" \
		"$PY_VERSION" "$MLX_VERSION" "$MLXLM_VERSION" \
		"$TOOLS_DIR" \
		"${CONVERT_CMD[@]}" <<'PYEOF'
import json
import shlex
import sys
from pathlib import Path

(
    record_arg,
    target_repo,
    target_rev,
    fork_git,
    fork_head,
    standalone_repo,
    standalone_rev,
    standalone_file,
    standalone_bytes,
    standalone_sha256,
    python_version,
    mlx_version,
    mlx_lm_version,
    tools_dir,
    *conversion_argv,
) = sys.argv[1:]

sys.path.insert(0, tools_dir)
from mtp_fixture_safetensors import FixtureValidationError, load_unique_json

record_path = Path(record_arg)
try:
    record = load_unique_json(record_path.read_bytes(), str(record_path))
except FixtureValidationError as error:
    raise SystemExit(f"FIXTURE ACCEPTANCE FAILURE: {error}") from error
if not isinstance(record, dict):
    raise SystemExit("FIXTURE ACCEPTANCE FAILURE: record root must be an object")
record["pins"] = {
    "weight_source": {"repo": target_repo, "revision": target_rev,
                      "role": "single authoritative MTP weight source"},
    "converter": {"git": fork_git, "sha": fork_head},
    "crosscheck_artifact": {"repo": standalone_repo, "revision": standalone_rev,
                            "file": standalone_file, "bytes": int(standalone_bytes),
                            "sha256": standalone_sha256,
                            "role": "mandatory equivalence cross-check, NOT a weight source"},
}
record["versions"] = {"python": python_version, "mlx": mlx_version,
                      "mlx_lm": mlx_lm_version}
portable_argv = list(conversion_argv)
for index, argument in enumerate(portable_argv[:-1]):
    if argument == "--hf-path":
        portable_argv[index + 1] = f"<workdir>/target-{target_rev}"
    elif argument == "--mlx-path":
        portable_argv[index + 1] = "<workdir>/converted"
record["conversion_argv"] = portable_argv
record["conversion_command"] = shlex.join(portable_argv)
record_path.write_text(json.dumps(record, indent=2) + "\n")
PYEOF

# ---- 7. strip-mtp variant --------------------------------------------------
log "building + verifying the strip-mtp variant"
if [[ -e "$STRIP_DIR" ]]; then
	echo "ABORT: strip output already exists; use a fresh workdir: $STRIP_DIR" >&2
	exit 1
fi
python3 "$TOOLS_DIR/mtp_fixture_strip.py" --src "$CONVERTED_DIR" --dst "$STRIP_DIR"

log "DONE"
echo "converted fixture:  $CONVERTED_DIR   (EMELEX_TEST_MODEL points here)"
echo "strip-mtp variant:  $STRIP_DIR"
echo "record:             $RECORD  — summarize accepted evidence in"
echo "                    tests/fixtures/mtp_fixture_manifest.md"
echo "certification:      copy the recorded hashes into tests/fixtures/mtp_certification.json"
