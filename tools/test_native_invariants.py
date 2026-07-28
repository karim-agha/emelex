#!/usr/bin/env python3
"""Deterministic source invariants for Emelex's reviewed native patches."""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALLOCATOR = ROOT / "vendor" / "mlx" / "mlx" / "backend" / "metal" / "allocator.cpp"


def function_body(source: str, signature: str, next_signature: str) -> str:
    start = source.index(signature)
    end = source.index(next_signature, start)
    return source[start:end]


class NativeInvariantTests(unittest.TestCase):
    def test_metal_resource_admission_and_creation_share_one_lock(self) -> None:
        source = ALLOCATOR.read_text()
        malloc = function_body(
            source,
            "Buffer MetalAllocator::malloc(size_t size)",
            "void MetalAllocator::clear_cache()",
        )
        lock = malloc.index("std::unique_lock lk(mutex_);")
        admission = malloc.index("if (num_resources_ >= resource_limit_)", lock)
        creation = malloc.index("heap_->newBuffer", admission)
        fallback_creation = malloc.index("device_->newBuffer", creation)
        count = malloc.index("num_resources_++;", fallback_creation)
        self.assertNotIn("lk.unlock()", malloc[lock:count])

        no_copy = function_body(
            source,
            "Buffer MetalAllocator::make_buffer(void* ptr, size_t size)",
            "void MetalAllocator::release(Buffer buffer)",
        )
        lock = no_copy.index("std::unique_lock lk(mutex_);")
        admission = no_copy.index("if (num_resources_ >= resource_limit_)", lock)
        fallback = no_copy.index("return Buffer{nullptr};", admission)
        creation = no_copy.index("device_->newBuffer", fallback)
        count = no_copy.index("num_resources_++;", creation)
        self.assertNotIn("lk.unlock()", no_copy[lock:count])


if __name__ == "__main__":
    unittest.main()
