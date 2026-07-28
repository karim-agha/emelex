#!/usr/bin/env python3
# emelex patch (not upstream): strip-mtp fixture variant builder — plan
# Dense fixture contract.
"""Produce the strip-mtp variant of the converted dense BF16 fixture.

Removes EXACTLY the enumerated 15 ``language_model.mtp.*`` keys from a
copy of the converted model directory, rewriting the safetensors shard(s)
and ``model.safetensors.index.json``, and verifies that no non-MTP key was
removed (input key set == output key set + exactly those 15). The variant
must load with ``supports_mtp() == false`` and behave byte-identically to
a never-MTP model.

Stdlib-only (manual safetensors header parsing — no framework deps), so it
runs on any machine that holds the artifacts.

Usage:
    python3 tools/mtp_fixture_strip.py --src <converted_dir> --dst <strip_dir>

Writes ``<dst>/strip_record.json``: removed-key audit, per-file sha256
digests of every output file, and the byte-preservation verification for
every surviving tensor.
"""

import argparse
import hashlib
import json
import shutil
import struct
import sys
from pathlib import Path

from mtp_fixture_safetensors import (
    FixtureValidationError,
    claim_unique_tensor_ownership,
    load_unique_json,
    read_safetensors,
)

# The complete on-disk MTP key set: the sole keys this
# script may remove. Anything else disappearing is a verification failure.
MTP_KEYS = [
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
]
assert len(MTP_KEYS) == 15


def write_safetensors(path: Path, metadata, tensors):
    """Write tensors (insertion-ordered) with recomputed contiguous offsets."""
    header = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    offset = 0
    payload = []
    for name, (info, data) in tensors.items():
        header[name] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
        payload.append(data)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for data in payload:
            f.write(data)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=Path, help="converted model dir")
    ap.add_argument("--dst", required=True, type=Path, help="strip-variant output dir")
    args = ap.parse_args()

    src, dst = args.src, args.dst
    if not (src / "config.json").exists():
        sys.exit(f"error: {src} has no config.json")
    if dst.exists() and any(dst.iterdir()):
        sys.exit(f"error: destination {dst} exists and is not empty")
    dst.mkdir(parents=True, exist_ok=True)

    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        sys.exit(f"error: no *.safetensors in {src}")

    # Copy every non-shard, non-index file verbatim (config, tokenizer, ...).
    index_name = "model.safetensors.index.json"
    for entry in sorted(src.iterdir()):
        if entry.is_dir():
            sys.exit(f"error: unexpected subdirectory {entry} (converted dirs are flat)")
        if entry.suffix == ".safetensors" or entry.name == index_name:
            continue
        shutil.copy2(entry, dst / entry.name)

    expected = set(MTP_KEYS)
    removed = {}  # key -> bytes removed
    input_keys, output_keys = set(), set()
    owners = {}
    output_owners = {}
    total_tensor_bytes = 0
    preserved_ok = True

    try:
        for shard in shards:
            _, tensors = read_safetensors(shard)
            claim_unique_tensor_ownership(owners, shard, tensors)
            total_tensor_bytes += sum(len(data) for _, data in tensors.values())
    except FixtureValidationError as error:
        sys.exit(f"VERIFICATION FAILURE: {error}")

    index_path = src / index_name
    if len(shards) > 1 and not index_path.exists():
        sys.exit("VERIFICATION FAILURE: multi-shard input requires a safetensors index")
    index = None
    if index_path.exists():
        try:
            index = load_unique_json(index_path.read_bytes(), str(index_path))
        except FixtureValidationError as error:
            sys.exit(f"VERIFICATION FAILURE: {error}")
        if not isinstance(index, dict):
            sys.exit("VERIFICATION FAILURE: index root must be an object")
        weight_map = index.get("weight_map")
        expected_map = {name: owner.name for name, owner in owners.items()}
        if weight_map != expected_map:
            sys.exit(
                "VERIFICATION FAILURE: index weight_map does not exactly match "
                "tensor shard ownership"
            )
        metadata = index.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            sys.exit("VERIFICATION FAILURE: index metadata must be an object")
        if metadata is not None and "total_size" in metadata:
            total_size = metadata["total_size"]
            if type(total_size) is not int or total_size != total_tensor_bytes:
                sys.exit(
                    "VERIFICATION FAILURE: index metadata.total_size does not "
                    "equal tensor payload bytes"
                )

    for shard in shards:
        metadata, tensors = read_safetensors(shard)
        input_keys.update(tensors)
        kept = {}
        for name, (info, data) in tensors.items():
            if name in expected:
                removed[name] = len(data)
            else:
                kept[name] = (info, data)
        out_path = dst / shard.name
        write_safetensors(out_path, metadata, kept)
        # Verification: every surviving tensor byte-identical after rewrite.
        _, reread = read_safetensors(out_path)
        try:
            claim_unique_tensor_ownership(output_owners, out_path, reread)
        except FixtureValidationError as error:
            sys.exit(f"VERIFICATION FAILURE: {error}")
        output_keys.update(reread)
        for name, (info, data) in kept.items():
            r_info, r_data = reread[name]
            if r_info != info or r_data != data:
                preserved_ok = False
                print(f"VERIFICATION FAILURE: tensor '{name}' altered by rewrite")

    # Removed-key audit: exactly the 15, nothing else touched.
    if set(removed) != expected:
        missing = sorted(expected - set(removed))
        sys.exit(
            "VERIFICATION FAILURE: removed key set != the 15-key "
            f"language_model.mtp.* set (missing from input: {missing})"
        )
    if input_keys - set(removed) != output_keys:
        sys.exit("VERIFICATION FAILURE: a non-MTP key was removed or added")
    if not preserved_ok:
        sys.exit(1)

    # Rewrite the index (multi-shard layout), dropping the 15 entries.
    if index is not None:
        weight_map = index["weight_map"]
        for key in MTP_KEYS:
            del weight_map[key]
        expected_output_map = {
            name: owner.name for name, owner in output_owners.items()
        }
        if weight_map != expected_output_map or set(weight_map) != output_keys:
            sys.exit(
                "VERIFICATION FAILURE: rewritten index does not exactly match "
                "surviving tensor ownership"
            )
        metadata = index.get("metadata")
        if metadata is not None and "total_size" in metadata:
            metadata["total_size"] = total_tensor_bytes - sum(removed.values())
        (dst / index_name).write_text(json.dumps(index, indent=2) + "\n")

    record = {
        "source": "converted dense BF16 fixture",
        "removed_keys": sorted(removed),
        "removed_bytes": sum(removed.values()),
        "surviving_tensor_count": len(output_keys),
        "output_digests": {
            entry.name: sha256_file(entry)
            for entry in sorted(dst.iterdir())
            if entry.name != "strip_record.json"
        },
    }
    (dst / "strip_record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"strip-mtp variant written to {dst}")
    print(f"removed exactly {len(removed)} keys ({sum(removed.values())} bytes)")
    print(f"record: {dst / 'strip_record.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
