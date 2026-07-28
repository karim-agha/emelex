# Attribution

Emelex includes and modifies code derived from `mlex` 0.1.3, distributed
under the MIT License. Its license is reproduced in
`licenses/mlex.LICENSE`.

The private inference engine has substantial Emelex changes, including
additional model families, multimodal preprocessing, reasoning/tool streaming,
prompt-cache behavior, MTP self-speculative decoding, failure recovery, and
runtime packaging. It is not an unmodified upstream copy.

Emelex also vendors pinned native build inputs:

- MLX: `vendor/mlx`, upstream license at `vendor/mlx/LICENSE`
- mlx-c: `vendor/mlx-c`, upstream license at `vendor/mlx-c/LICENSE`
- Apple metal-cpp 26: `vendor/metal-cpp`, upstream license at
  `vendor/metal-cpp/LICENSE.txt`
- nlohmann/json 3.11.3: `vendor/nlohmann-json`, upstream license at
  `vendor/nlohmann-json/LICENSE.MIT`
- fmt 12.1.0: `vendor/fmt`, upstream license at `vendor/fmt/LICENSE`
- PyTorch float32 `erf` approximation: pinned upstream commit
  `abf28982a8cb43342e7669d859de9543fd804cc9`, source at
  <https://github.com/pytorch/pytorch/blob/abf28982a8cb43342e7669d859de9543fd804cc9/aten/src/ATen/cpu/vec/vec256/vec256_float.h#L175>,
  BSD-3-Clause notices and terms reproduced in
  `vendor/mlx/ACKNOWLEDGMENTS.md`
- Boost.Math inverse-erf rational approximations: John Maddock (2006) and
  Matt Borland (2024), pinned upstream commit
  `6a4487453d95c1fbf5ecf3da18f2c020a89fd612`, source at
  <https://github.com/boostorg/math/blob/6a4487453d95c1fbf5ecf3da18f2c020a89fd612/include/boost/math/special_functions/detail/erf_inv.hpp>,
  Boost Software License 1.0 reproduced in
  `vendor/mlx/ACKNOWLEDGMENTS.md`

Exact revisions and refresh instructions live in `vendor/PINS.md`.

Locked Rust dependencies and their exact license documents are listed in
`licenses/RUST-DEPENDENCIES.md`. Regenerate that bundle with
`tools/update_rust_licenses.py` whenever `Cargo.lock` changes.
