# Third-party notices

Emelex's source distribution contains or builds the following third-party
components. Their original licenses remain authoritative.

| Component | Location or build input | License |
| --- | --- | --- |
| mlex 0.1.3 | engine-derived Rust source | MIT |
| MLX | `vendor/mlx` | MIT |
| mlx-c | `vendor/mlx-c` | MIT |
| metal-cpp 26 | `vendor/metal-cpp` | Apache-2.0 |
| nlohmann/json 3.11.3 | `vendor/nlohmann-json` | MIT |
| fmt 12.1.0 | `vendor/fmt` | MIT |
| PyTorch `erf` approximation at `abf28982a8cb43342e7669d859de9543fd804cc9` | MLX CPU/Metal math kernels; source and terms in `vendor/mlx/ACKNOWLEDGMENTS.md` | BSD-3-Clause |
| Boost.Math inverse-erf approximations at `6a4487453d95c1fbf5ecf3da18f2c020a89fd612` | MLX CPU/Metal math kernels; John Maddock (2006), Matt Borland (2024); source and terms in `vendor/mlx/ACKNOWLEDGMENTS.md` | BSL-1.0 |
| Locked Rust dependencies | `licenses/RUST-DEPENDENCIES.md` | Per dependency |

`licenses/RUST-DEPENDENCIES.md` contains every Rust package selected by
`Cargo.lock`, its declared license expression, and exact license documents
from its Cargo source archive. Release validation checks that generated bundle
against the lockfile.
