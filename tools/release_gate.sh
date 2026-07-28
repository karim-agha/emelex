#!/usr/bin/env bash
# Build, package, and audit one reproducible local Emelex release candidate.

set -euo pipefail
unset CDPATH

if [[ ${RUSTFLAGS+x} == x || ${CARGO_ENCODED_RUSTFLAGS+x} == x ]]; then
	echo "error: release gate refuses ambient Rust flags" >&2
	exit 2
fi
if [[ ${CARGO_TARGET_DIR+x} == x || ${CARGO_BUILD_TARGET+x} == x ]]; then
	echo "error: release gate refuses ambient Cargo target overrides" >&2
	exit 2
fi
if [[ ${RUSTUP_TOOLCHAIN+x} == x || ${RUSTC+x} == x || ${RUSTC_WRAPPER+x} == x || ${RUSTC_WORKSPACE_WRAPPER+x} == x ]]; then
	echo "error: release gate refuses ambient Rust toolchain overrides" >&2
	exit 2
fi
if [[ ${CARGO_BUILD_RUSTC+x} == x || ${CARGO_BUILD_RUSTC_WRAPPER+x} == x || ${CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER+x} == x ]]; then
	echo "error: release gate refuses Cargo-config Rust compiler overrides" >&2
	exit 2
fi
native_build_overrides=(
	MACOSX_DEPLOYMENT_TARGET
	CC
	CXX
	CPP
	AR
	RANLIB
	CFLAGS
	CXXFLAGS
	CPPFLAGS
	LDFLAGS
	HOST_CC
	HOST_CXX
	HOST_AR
	HOST_RANLIB
	HOST_CFLAGS
	HOST_CXXFLAGS
	HOST_CPPFLAGS
	HOST_LDFLAGS
	TARGET_CC
	TARGET_CXX
	TARGET_AR
	TARGET_RANLIB
	TARGET_CFLAGS
	TARGET_CXXFLAGS
	TARGET_CPPFLAGS
	TARGET_LDFLAGS
	CC_aarch64-apple-darwin
	CXX_aarch64-apple-darwin
	AR_aarch64-apple-darwin
	RANLIB_aarch64-apple-darwin
	CFLAGS_aarch64-apple-darwin
	CXXFLAGS_aarch64-apple-darwin
	CPPFLAGS_aarch64-apple-darwin
	LDFLAGS_aarch64-apple-darwin
	CC_aarch64_apple_darwin
	CXX_aarch64_apple_darwin
	AR_aarch64_apple_darwin
	RANLIB_aarch64_apple_darwin
	CFLAGS_aarch64_apple_darwin
	CXXFLAGS_aarch64_apple_darwin
	CPPFLAGS_aarch64_apple_darwin
	LDFLAGS_aarch64_apple_darwin
	LIBCLANG_PATH
	CLANG_PATH
	LLVM_CONFIG_PATH
	LD_LIBRARY_PATH
	BINDGEN_EXTRA_CLANG_ARGS
	BINDGEN_EXTRA_CLANG_ARGS_aarch64-apple-darwin
	BINDGEN_EXTRA_CLANG_ARGS_aarch64_apple_darwin
	DEVELOPER_DIR
	TOOLCHAINS
	SDKROOT
	CPATH
	CPLUS_INCLUDE_PATH
	LIBRARY_PATH
	CLANG_MODULE_CACHE_PATH
	CRATE_CC_NO_DEFAULTS
	CMAKE
	TARGET_CMAKE
	CMAKE_aarch64-apple-darwin
	CMAKE_aarch64_apple_darwin
	CMAKE_GENERATOR
	TARGET_CMAKE_GENERATOR
	CMAKE_GENERATOR_aarch64-apple-darwin
	CMAKE_GENERATOR_aarch64_apple_darwin
	CMAKE_GENERATOR_PLATFORM
	CMAKE_GENERATOR_TOOLSET
	CMAKE_TOOLCHAIN_FILE
	TARGET_CMAKE_TOOLCHAIN_FILE
	CMAKE_TOOLCHAIN_FILE_aarch64-apple-darwin
	CMAKE_TOOLCHAIN_FILE_aarch64_apple_darwin
	CMAKE_PREFIX_PATH
	TARGET_CMAKE_PREFIX_PATH
	CMAKE_PREFIX_PATH_aarch64-apple-darwin
	CMAKE_PREFIX_PATH_aarch64_apple_darwin
	CMAKE_OSX_ARCHITECTURES
	CMAKE_OSX_DEPLOYMENT_TARGET
	CMAKE_OSX_SYSROOT
	AWS_LC_SYS_USE_SYSTEM
	AWS_LC_SYS_USE_SYSTEM_aarch64_apple_darwin
	LIBSQLITE3_SYS_USE_PKG_CONFIG
	SQLITE_MAX_VARIABLE_NUMBER
	SQLITE_MAX_EXPR_DEPTH
	SQLITE_MAX_COLUMN
	LIBSQLITE3_FLAGS
	ZSTD_SYS_USE_PKG_CONFIG
	RUSTONIG_SYSTEM_LIBONIG
	RUSTONIG_DYNAMIC_LIBONIG
	RUSTONIG_STATIC_LIBONIG
	DOCS_RS
)
for override_name in "${native_build_overrides[@]}"; do
	if /usr/bin/printenv "$override_name" >/dev/null; then
		echo "error: release gate refuses ambient native build override $override_name" >&2
		exit 2
	fi
done
release_deployment_target="26.5"

allow_dirty_argument=""
if (($# > 0)); then
	if [[ $# == 1 && $1 == "--allow-dirty" ]]; then
		allow_dirty_argument="--allow-dirty"
	else
		echo "usage: tools/release_gate.sh [--allow-dirty]" >&2
		exit 2
	fi
fi

repository="$(cd -- "$(dirname -- "$0")/.." && pwd -P)"
cd "$repository"
builder_home="$(cd -- "${HOME:?HOME is required}" && pwd -P)"
cargo_home="$(cd -- "${CARGO_HOME:-$builder_home/.cargo}" && pwd -P)"
rustup_home="$(cd -- "${RUSTUP_HOME:-$builder_home/.rustup}" && pwd -P)"
rustup_path="$cargo_home/bin/rustup"
if [[ ! -x $rustup_path ]]; then
	echo "error: release gate cannot find rustup in Cargo home" >&2
	exit 2
fi
tool_resolution_environment=(
	/usr/bin/env -i
	"HOME=$builder_home"
	"PATH=/usr/bin:/bin:/usr/sbin:/sbin"
	"RUSTUP_HOME=$rustup_home"
	"LC_ALL=C"
	"LANG=C"
)
rustc_path="$("${tool_resolution_environment[@]}" "$rustup_path" which rustc)"
cargo_path="$("${tool_resolution_environment[@]}" "$rustup_path" which cargo)"
rustc_directory="$(cd -- "$(dirname -- "$rustc_path")" && pwd -P)"
rustc_path="$rustc_directory/$(basename -- "$rustc_path")"
cargo_path="$(cd -- "$(dirname -- "$cargo_path")" && pwd -P)/$(basename -- "$cargo_path")"
if [[ ! -x $cargo_path || $cargo_path != "$rustc_directory/cargo" ]]; then
	echo "error: pinned Rust toolchain has no matching executable cargo" >&2
	exit 2
fi
sysroot="$("$rustc_path" --print sysroot)"
sysroot="$(cd -- "$sysroot" && pwd -P)"
if [[ ! -x $rustc_path || $rustc_path != "$sysroot/bin/rustc" ]]; then
	echo "error: pinned Rust toolchain has no executable rustc" >&2
	exit 2
fi
resolved_sysroot="$("$rustc_path" --print sysroot)"
resolved_sysroot="$(cd -- "$resolved_sysroot" && pwd -P)"
if [[ $resolved_sysroot != "$sysroot" ]]; then
	echo "error: pinned rustc reports a different sysroot" >&2
	exit 2
fi
cmake_path="/opt/homebrew/bin/cmake"
if [[ ! -x $cmake_path ]]; then
	echo "error: release gate requires CMake at /opt/homebrew/bin/cmake" >&2
	exit 2
fi
release_driver="$(/usr/bin/mktemp -d /private/tmp/emelex-release-driver.XXXXXX)"
cleanup_release_driver() {
	/bin/rm -rf -- "$release_driver"
}
trap cleanup_release_driver EXIT
release_temp="$release_driver/tmp"
/bin/mkdir -m 700 "$release_temp"
tool_resolution_environment+=("TMPDIR=$release_temp")
sdkroot="$("${tool_resolution_environment[@]}" /usr/bin/xcrun --sdk macosx --show-sdk-path)"
sdkroot="$(cd -- "$sdkroot" && pwd -P)"
isolated_cargo_home="$release_driver/cargo-home"
/bin/mkdir -m 700 "$isolated_cargo_home"
if [[ ! -d $cargo_home/registry ]]; then
	echo "error: release gate cannot find Cargo's offline registry cache" >&2
	exit 2
fi
/bin/ln -s "$cargo_home/registry" "$isolated_cargo_home/registry"
if [[ -d $cargo_home/git ]]; then
	/bin/ln -s "$cargo_home/git" "$isolated_cargo_home/git"
fi
target_triple="aarch64-apple-darwin"
encoded_separator=$'\x1f'
CARGO_ENCODED_RUSTFLAGS="--remap-path-prefix=${repository}=/emelex-source"
CARGO_ENCODED_RUSTFLAGS+="${encoded_separator}"
CARGO_ENCODED_RUSTFLAGS+="--remap-path-prefix=${cargo_home}=/emelex-cargo"
CARGO_ENCODED_RUSTFLAGS+="${encoded_separator}"
CARGO_ENCODED_RUSTFLAGS+="--remap-path-prefix=${sysroot}=/emelex-toolchain"
CARGO_ENCODED_RUSTFLAGS+="${encoded_separator}"
CARGO_ENCODED_RUSTFLAGS+="--remap-path-prefix=${rustup_home}=/emelex-rustup"
CARGO_ENCODED_RUSTFLAGS+="${encoded_separator}"
CARGO_ENCODED_RUSTFLAGS+="--remap-path-prefix=${builder_home}=/emelex-builder"
CARGO_ENCODED_RUSTFLAGS+="${encoded_separator}"
CARGO_ENCODED_RUSTFLAGS+="--remap-path-prefix=${release_driver}=/emelex-release-driver"
release_cargo_environment=(
	/usr/bin/env -i
	"HOME=$builder_home"
	"PATH=$sysroot/bin:/usr/bin:/bin:/usr/sbin:/sbin"
	"LC_ALL=C"
	"LANG=C"
	"TMPDIR=$release_temp"
	"CARGO_HOME=$isolated_cargo_home"
	"RUSTUP_HOME=$rustup_home"
	"RUSTC=$rustc_path"
	"RUSTC_WRAPPER="
	"RUSTC_WORKSPACE_WRAPPER="
	"CARGO_ENCODED_RUSTFLAGS=$CARGO_ENCODED_RUSTFLAGS"
	"CARGO_INCREMENTAL=0"
	"MACOSX_DEPLOYMENT_TARGET=$release_deployment_target"
	"SDKROOT=$sdkroot"
	"CC=/usr/bin/clang"
	"CXX=/usr/bin/clang++"
	"AR=/usr/bin/ar"
	"RANLIB=/usr/bin/ranlib"
	"CMAKE=$cmake_path"
	"CMAKE_GENERATOR=Unix Makefiles"
	"AWS_LC_SYS_USE_SYSTEM=0"
	"ZERO_AR_DATE=1"
)
release_cargo() {
	"${release_cargo_environment[@]}" "$cargo_path" "$@"
}

cd "$release_driver"
release_cargo clean --manifest-path "$repository/Cargo.toml" --target-dir "$repository/target"
release_cargo package --manifest-path "$repository/Cargo.toml" --locked --offline --target "$target_triple" --target-dir "$repository/target" ${allow_dirty_argument:+"$allow_dirty_argument"}
release_cargo build --manifest-path "$repository/Cargo.toml" --release --locked --offline --target "$target_triple" --target-dir "$repository/target"
cd "$repository"
PYTHONDONTWRITEBYTECODE=1 python3 tools/release_audit.py --repository "$repository"
