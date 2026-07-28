# Runtime

`runtime` extracts the build-generated, zstd-compressed `mlx.metallib` beneath
the selected Emelex home and installs its path through the patched mlx-c API
before any MLX object exists.

The first successfully initialized canonical home wins for the process.
Repeated initialization with that home is idempotent. A different home returns
`RuntimeError::HomeConflict`.

Runtime files are content-addressed by SHA-256. Directories are mode `0700`;
the metallib is mode `0600`. Extraction uses a same-directory temporary file,
digest/size verification through an `O_NOFOLLOW` descriptor, `fsync`, and
atomic rename. Paths crossing the mlx-c ABI must be UTF-8.

`verify_engine` performs a tiny evaluated GPU operation, proving the relocated
asset is loadable rather than merely configured.

The Metal completion-callback regression explicitly skips on a headless host.
Release CI must run its physical-GPU sentinel lane:

```sh
EMELEX_REQUIRE_PHYSICAL_GPU=1 cargo test --test runtime \
  metal_completion_callback_failure_returns_error_without_aborting -- --nocapture
```
