#!/usr/bin/env python3
"""Regression tests for strict MTP fixture safetensors evidence parsing."""

from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from mtp_fixture_safetensors import (
    FixtureValidationError,
    load_unique_json,
    merge_unique_tensors,
    read_safetensors,
    require_mapped_reference_keys,
)


def write_raw(path: Path, header: str, payload: bytes) -> None:
    encoded = header.encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def write_shard(path: Path, names: list[str]) -> None:
    header = {}
    payload = bytearray()
    for name in names:
        begin = len(payload)
        payload.extend(b"\0\0")
        header[name] = {
            "dtype": "BF16",
            "shape": [1],
            "data_offsets": [begin, len(payload)],
        }
    write_raw(path, json.dumps(header), bytes(payload))


class FixtureSafetensorsTests(unittest.TestCase):
    def test_non_standard_json_constants_are_rejected(self) -> None:
        for constant in ["NaN", "Infinity", "-Infinity"]:
            with self.subTest(constant=constant):
                with self.assertRaisesRegex(FixtureValidationError, "non-standard"):
                    load_unique_json(f'{{"value":{constant}}}', "test")

    def test_duplicate_header_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.safetensors"
            descriptor = '{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}'
            write_raw(path, f'{{"same":{descriptor},"same":{descriptor}}}', b"\0\0")
            with self.assertRaisesRegex(FixtureValidationError, "duplicate JSON key"):
                read_safetensors(path)

    def test_duplicate_tensor_ownership_across_shards_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "model-00001.safetensors"
            second = root / "model-00002.safetensors"
            write_shard(first, ["shared"])
            write_shard(second, ["shared"])
            with self.assertRaisesRegex(FixtureValidationError, "owned by both"):
                merge_unique_tensors([first, second])

    def test_incomplete_standalone_reference_key_set_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.safetensors"
            write_shard(reference, ["first"])
            _, tensors = read_safetensors(reference)
            expected = {"language_model.mtp.first", "language_model.mtp.second"}
            with self.assertRaisesRegex(FixtureValidationError, "missing"):
                require_mapped_reference_keys(tensors, "language_model.mtp.", expected)

    def test_ambiguous_ranges_shapes_and_dtypes_are_rejected(self) -> None:
        cases = {
            "unknown dtype": (
                {
                    "x": {
                        "dtype": "MYSTERY",
                        "shape": [1],
                        "data_offsets": [0, 1],
                    }
                },
                b"\0",
            ),
            "invalid shape": (
                {
                    "x": {
                        "dtype": "BF16",
                        "shape": [-1],
                        "data_offsets": [0, 0],
                    }
                },
                b"",
            ),
            "byte length": (
                {
                    "x": {
                        "dtype": "BF16",
                        "shape": [2],
                        "data_offsets": [0, 2],
                    }
                },
                b"\0\0",
            ),
            "gap": (
                {
                    "x": {
                        "dtype": "BF16",
                        "shape": [1],
                        "data_offsets": [1, 3],
                    }
                },
                b"\0\0\0",
            ),
            "overlap": (
                {
                    "x": {
                        "dtype": "BF16",
                        "shape": [1],
                        "data_offsets": [0, 2],
                    },
                    "y": {
                        "dtype": "BF16",
                        "shape": [1],
                        "data_offsets": [1, 3],
                    },
                },
                b"\0\0\0",
            ),
            "payload exactly": (
                {
                    "x": {
                        "dtype": "BF16",
                        "shape": [1],
                        "data_offsets": [0, 2],
                    }
                },
                b"\0\0trailing",
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (expected, (header, payload)) in enumerate(cases.items()):
                with self.subTest(expected=expected):
                    path = root / f"invalid-{index}.safetensors"
                    write_raw(path, json.dumps(header), payload)
                    with self.assertRaisesRegex(FixtureValidationError, expected):
                        read_safetensors(path)


if __name__ == "__main__":
    unittest.main()
