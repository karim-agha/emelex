# Vendored sources

These checkouts are vendored with their own `.git` metadata stripped.
`build.rs` builds them via CMake and directs every FetchContent dependency to a
vendored source directory. Native source builds require no network access.

To refresh either checkout, re-clone at the desired ref, delete its `.git`
directory, and update the pin below.

| Directory      | Upstream                                | Pinned commit                              |
| -------------- | --------------------------------------- | ------------------------------------------ |
| `vendor/mlx`   | https://github.com/ml-explore/mlx.git   | `68cf2fddd8de5edd8ab3d926391772b2e2cedad8` |
| `vendor/mlx-c` | https://github.com/ml-explore/mlx-c.git | `fba4470b89073180056c9ea46c443051375f7399` |
| `vendor/metal-cpp` | https://developer.apple.com/metal/cpp/ | metal-cpp 26 archive, SHA-256 `4df3c078b9aadcb516212e9cb03004cbc5ce9a3e9c068fa3144d021db585a3a4` |
| `vendor/nlohmann-json` | https://github.com/nlohmann/json | 3.11.3 archive, SHA-256 `d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d` |
| `vendor/fmt` | https://github.com/fmtlib/fmt.git | `407c905e45ad75fc29bf0f9bb7c5c2fd3475976f` |

## Emelex patches

- `vendor/mlx/CMakeLists.txt`: pin CMake downloads, expose
  `MLX_BUILD_JACCL`, disable its SDK-based automatic selection, and route
  Metal compiler modules into Cargo `OUT_DIR`.
- `vendor/mlx/mlx/distributed/jaccl/CMakeLists.txt`: select `no_jaccl.cpp`
  when `MLX_BUILD_JACCL=OFF`.
- `vendor/mlx/mlx/backend/metal/device.{h,cpp}`: remove the compiled absolute
  metallib fallback and add a process-latched default-library path installed
  before Metal initialization. Device enumeration copies the returned Metal
  device array before the autorelease pool drains and reports a typed
  no-device error instead of indexing an empty list on headless hosts. Each
  committed command buffer also receives one non-throwing completion handler;
  it retains the first Metal error in a stream-local, thread-safe state for
  caller-thread synchronization. Destruction drains and discards errors
  without throwing. Callback bookkeeping captures shared state instead of a
  raw `CommandEncoder` pointer, so teardown cannot create a use-after-free.
  Library and kernel caches share a fixed library-before-kernel lock order,
  read paths never mutate maps under shared locks, and cache entries remain
  immutable for the process lifetime.
- `vendor/mlx/mlx/backend/metal/custom_kernel.cpp`: key custom libraries by a
  canonical length-prefixed `(name, source)` tuple. Libraries and pipelines
  remain strongly owned by the process-lifetime Device cache because Metal
  command buffers use unretained resource references. Distinct generated
  sources can grow that cache until process exit; safe native object lifetime
  takes precedence over mutable eviction.
- `vendor/mlx/mlx/backend/metal/allocator.{h,cpp}`: synchronize allocator
  counter, limit, and buffer-cache observations while keeping already-locked
  allocation paths on direct field reads to avoid recursive locking.
- `vendor/mlx/mlx/backend/{cpu/simd/math.h,metal/kernels/erf.h}`: add the
  float32 `erf` approximation pinned to PyTorch commit
  `abf28982a8cb43342e7669d859de9543fd804cc9` and inverse-erf rational
  approximations pinned to Boost.Math commit
  `6a4487453d95c1fbf5ecf3da18f2c020a89fd612`. Exact source links,
  copyrights, BSD-3-Clause terms, and BSL-1.0 terms are retained in
  `vendor/mlx/ACKNOWLEDGMENTS.md`.
- `vendor/mlx/mlx/backend/metal/{eval,event,fence}.cpp`: completion handlers
  are `noexcept` catch-all boundaries. Per-evaluation handlers retain
  lifetimes and update scheduler accounting only; they never throw Metal
  failures on framework threads. Auto-commit accounting uses a one-shot RAII
  token, including handler-registration and commit rollback.
- `vendor/mlx/mlx/backend/cpu/encoder.{h,cpp}` and
  `vendor/mlx/mlx/scheduler.{h,cpp}`: keep CPU encoders thread-local,
  balance grouped-task completion through RAII, make active-task observation
  atomic, normalize non-standard C++ throws, retain the first exception per
  stream worker, and rethrow it only after a caller-thread synchronization
  barrier.
- `vendor/mlx/mlx/backend/common/load.cpp`: observe asynchronous load failures
  with `future::get`, not a wait that discards the exception.
- `vendor/mlx/mlx/io/load.{h,cpp}`: treat descriptor zero as valid/owned and
  advance positional offsets after short `pread` results. Private deterministic
  seams cover both contracts.
- `vendor/mlx-c/mlx/c/metal.{h,cpp}`: expose the path setter and
  `recommendedMaxWorkingSetSize` through the C ABI.
- `vendor/mlx-c/mlx/c/stream.{h,cpp}`: expose a status-returning default CPU
  stream getter so Rust never caches an empty handle after a native exception.
  Synchronization translates retained asynchronous Metal failures through the
  existing mlx-c status/error channel. A private one-shot completion-failure
  seam supports the subprocess regression.
- `vendor/mlx-c/mlx/c/array.cpp`: synchronize an array's captured evaluation
  stream (CPU or GPU) before returning success, so `Array::eval` observes
  asynchronous load or command-buffer failure as a Rust `Result` instead of
  deferred process termination.
- `vendor/mlx-c/mlx/c/transforms.cpp`: provide a private downstream regression
  seam that injects standard and non-standard grouped CPU scheduler exceptions
  and verifies both cross the mlx-c status/error boundary without terminating
  the process or leaking active-task accounting.
- `vendor/mlx-c/mlx/c/*.cpp` and `vendor/mlx-c/python/*` generator templates:
  pass exception text to `_mlx_error` through a literal `"%s"` format.
  Upstream's direct `mlx_error(e.what())` calls treat model paths and exception
  text as varargs format strings; patching generators prevents regeneration
  from restoring that vulnerability.
