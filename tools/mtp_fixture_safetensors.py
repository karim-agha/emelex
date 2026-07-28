#!/usr/bin/env python3
"""Strict safetensors evidence helpers for Emelex MTP fixture tooling."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Iterable, Mapping


class FixtureValidationError(ValueError):
    """Fixture evidence is structurally ambiguous or incomplete."""

MAX_HEADER_BYTES = 100 << 20
DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
    "C64": 8,
    "C128": 16,
}


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise FixtureValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise FixtureValidationError(f"non-standard JSON constant {value!r}")


def load_unique_json(data: bytes | str, source: str) -> object:
    """Decode JSON while rejecting duplicate keys at every object depth."""
    try:
        return json.loads(
            data,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, FixtureValidationError) as error:
        raise FixtureValidationError(f"{source}: invalid unambiguous JSON: {error}") from error


def read_safetensors(
    path: Path,
) -> tuple[object | None, dict[str, tuple[dict[str, object], bytes]]]:
    """Read one safetensors file and reject duplicate or invalid header entries."""
    blob = path.read_bytes()
    if len(blob) < 8:
        raise FixtureValidationError(f"{path}: safetensors header length is missing")
    (header_len,) = struct.unpack("<Q", blob[:8])
    if header_len == 0 or header_len > MAX_HEADER_BYTES:
        raise FixtureValidationError(f"{path}: safetensors header length is invalid")
    data_start = 8 + header_len
    if data_start > len(blob):
        raise FixtureValidationError(f"{path}: safetensors header is truncated")
    decoded = load_unique_json(blob[8:data_start], str(path))
    if not isinstance(decoded, dict):
        raise FixtureValidationError(f"{path}: safetensors header must be an object")
    header = decoded
    metadata = header.pop("__metadata__", None)
    if metadata is not None and (
        not isinstance(metadata, dict)
        or any(not isinstance(key, str) for key in metadata)
        or any(not isinstance(value, str) for value in metadata.values())
    ):
        raise FixtureValidationError(f"{path}: safetensors metadata must map strings to strings")
    payload_bytes = len(blob) - data_start
    tensors: dict[str, tuple[dict[str, object], bytes]] = {}
    intervals: list[tuple[int, int, str]] = []
    for name, raw_info in header.items():
        if not isinstance(name, str) or not isinstance(raw_info, dict):
            raise FixtureValidationError(f"{path}: invalid tensor header entry {name!r}")
        dtype = raw_info.get("dtype")
        shape = raw_info.get("shape")
        offsets = raw_info.get("data_offsets")
        if (
            set(raw_info) != {"dtype", "shape", "data_offsets"}
            or not isinstance(dtype, str)
            or not isinstance(shape, list)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(value) is not int for value in offsets)
        ):
            raise FixtureValidationError(f"{path}: invalid descriptor for tensor {name!r}")
        if dtype not in DTYPE_BYTES:
            raise FixtureValidationError(f"{path}: unknown dtype {dtype!r} for tensor {name!r}")
        if any(type(dimension) is not int or dimension < 0 for dimension in shape):
            raise FixtureValidationError(f"{path}: invalid shape for tensor {name!r}")
        begin, end = offsets
        if begin < 0 or end < begin or end > payload_bytes:
            raise FixtureValidationError(f"{path}: invalid data offsets for tensor {name!r}")
        elements = 1
        for dimension in shape:
            elements *= dimension
        expected_bytes = elements * DTYPE_BYTES[dtype]
        if end - begin != expected_bytes:
            raise FixtureValidationError(
                f"{path}: tensor {name!r} byte length does not match dtype and shape"
            )
        intervals.append((begin, end, name))
        tensors[name] = (
            {"dtype": dtype, "shape": shape},
            blob[data_start + begin : data_start + end],
        )
    cursor = 0
    for begin, end, name in sorted(intervals):
        if begin != cursor:
            raise FixtureValidationError(
                f"{path}: tensor {name!r} leaves a gap or overlaps another tensor"
            )
        cursor = end
    if cursor != payload_bytes:
        raise FixtureValidationError(f"{path}: tensor ranges do not cover the payload exactly")
    return metadata, tensors


def merge_unique_tensors(
    shards: Iterable[Path],
) -> dict[str, tuple[dict[str, object], bytes]]:
    """Merge shards only when every tensor has exactly one owning shard."""
    merged: dict[str, tuple[dict[str, object], bytes]] = {}
    owners: dict[str, Path] = {}
    saw_shard = False
    for shard in shards:
        saw_shard = True
        _, tensors = read_safetensors(shard)
        claim_unique_tensor_ownership(owners, shard, tensors)
        merged.update(tensors)
    if not saw_shard:
        raise FixtureValidationError("no safetensors shards were provided")
    return merged


def claim_unique_tensor_ownership(
    owners: dict[str, Path],
    shard: Path,
    tensors: Mapping[str, object],
) -> None:
    """Record shard ownership, rejecting a tensor claimed by two shards."""
    duplicates = sorted(set(owners).intersection(tensors))
    if duplicates:
        first = duplicates[0]
        raise FixtureValidationError(
            f"tensor {first!r} is owned by both {owners[first]} and {shard}"
        )
    owners.update((name, shard) for name in tensors)


def require_mapped_reference_keys(
    names: Iterable[str],
    prefix: str,
    expected: Mapping[str, object] | set[str],
) -> None:
    """Require standalone names to map bijectively onto the certified key set."""
    mapped = {prefix + name for name in names}
    expected_keys = set(expected)
    if mapped != expected_keys:
        missing = sorted(expected_keys - mapped)
        unexpected = sorted(mapped - expected_keys)
        raise FixtureValidationError(
            "standalone reference key set mismatch: "
            f"missing {missing}, unexpected {unexpected}"
        )
