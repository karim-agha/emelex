# Runtime context

Invariant: no engine load or generation may occur before
`runtime::initialize`.

Storage layout:

```text
<home>/cache/runtime/mlx/<sha256>/mlx.metallib
```

`recommended_max_working_set_size` calls a small mlx-c shim that creates a
temporary Metal device and reads `recommendedMaxWorkingSetSize`. It does not
construct MLX's singleton Metal device and therefore does not freeze the
metallib path.

The C++ layer also latches the first non-empty path, rejects later differences,
and loads an explicitly configured metallib before any colocated fallback.

The engine array-construction and default-stream guards are the lowest Rust
MLX-object boundaries. They initialize runtime before direct array or operation
tests can create MLX's singleton Metal device.

The completion-callback subprocess may skip only when no Metal device exists.
`EMELEX_REQUIRE_PHYSICAL_GPU=1` converts that skip into failure and is mandatory
on the release Apple-Silicon GPU lane.
