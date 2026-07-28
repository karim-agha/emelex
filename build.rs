//! Native MLX build and embedded metallib preparation.

use std::{
	env,
	ffi::OsStr,
	fmt,
	fs::{self, File},
	io::{BufReader, BufWriter, Read as _, Write as _},
	path::{Path, PathBuf},
	process::Command,
};

use sha2::{Digest as _, Sha256};

const MINIMUM_MACOS: &str = "26.5";

fn main() {
	if let Err(error) = run() {
		eprintln!("Emelex native build failed: {error}");
		std::process::exit(1);
	}
}

fn run() -> Result<(), String> {
	let manifest_dir = required_path("CARGO_MANIFEST_DIR")?;
	let out_dir = required_path("OUT_DIR")?;
	println!("cargo:rerun-if-env-changed=DOCS_RS");
	if env::var_os("DOCS_RS").is_some() {
		return prepare_docs_rs(&manifest_dir, &out_dir);
	}
	let target_os = required_env("CARGO_CFG_TARGET_OS")?;
	let target_arch = required_env("CARGO_CFG_TARGET_ARCH")?;
	if target_os != "macos" || target_arch != "aarch64" {
		return Err(format!(
			"unsupported target {target_arch}-{target_os}; Emelex 1.0 supports \
			 only aarch64-apple-darwin"
		));
	}
	let deployment_target = validate_deployment_target()?;
	let sdk_path = validate_sdk()?;
	let metal_toolchain = validate_metal_toolchain()?;

	let mlx_c_dir = manifest_dir.join("vendor/mlx-c");
	let mlx_dir = manifest_dir.join("vendor/mlx");
	let metal_cpp_dir = manifest_dir.join("vendor/metal-cpp");
	let json_dir = manifest_dir.join("vendor/nlohmann-json");
	let fmt_dir = manifest_dir.join("vendor/fmt");
	emit_rerun_directives(&[&mlx_c_dir, &mlx_dir, &metal_cpp_dir, &json_dir, &fmt_dir]);

	let module_cache = out_dir.join("clang-module-cache");
	let fetch_cache = out_dir.join("fetchcontent");
	fs::create_dir_all(&module_cache)
		.map_err(|error| io_error("create Clang module cache", &module_cache, &error))?;
	fs::create_dir_all(&fetch_cache)
		.map_err(|error| io_error("create CMake download cache", &fetch_cache, &error))?;

	let dst = build_native(&NativeBuild {
		source_root: &manifest_dir,
		mlx_c_dir: &mlx_c_dir,
		mlx_dir: &mlx_dir,
		metal_cpp_dir: &metal_cpp_dir,
		json_dir: &json_dir,
		fmt_dir: &fmt_dir,
		module_cache: &module_cache,
		fetch_cache: &fetch_cache,
		sdk_path: &sdk_path,
		deployment_target: &deployment_target,
		metal_toolchain: metal_toolchain.as_deref(),
	})?;
	verify_native_artifacts(&dst, &manifest_dir)?;
	emit_link_directives(&dst);
	prepare_metallib(&dst, &out_dir)?;
	generate_bindings(
		&mlx_c_dir,
		&out_dir,
		&module_cache,
		&sdk_path,
		&deployment_target,
	)?;
	Ok(())
}

fn prepare_docs_rs(manifest_dir: &Path, out_dir: &Path) -> Result<(), String> {
	let source = manifest_dir.join("src/engine/docs_bindings.rs");
	let bindings = out_dir.join("bindings.rs");
	fs::copy(&source, &bindings)
		.map_err(|error| io_error("copy documentation-only mlx-c bindings", &source, &error))?;
	let compressed = out_dir.join("mlx.metallib.zst");
	let empty = zstd::stream::encode_all([].as_slice(), 0)
		.map_err(|error| format!("encode documentation-only metallib placeholder: {error}"))?;
	fs::write(&compressed, empty).map_err(|error| {
		io_error(
			"write documentation-only metallib placeholder",
			&compressed,
			&error,
		)
	})?;
	println!("cargo:rerun-if-changed={}", source.display());
	println!(
		"cargo:rustc-env=EMELEX_METALLIB_ZST_PATH={}",
		compressed.display()
	);
	println!(
		"cargo:rustc-env=EMELEX_METALLIB_SHA256={}",
		hex::encode(Sha256::digest([]))
	);
	println!("cargo:rustc-env=EMELEX_METALLIB_SIZE=0");
	println!("cargo:rustc-env=EMELEX_MINIMUM_MACOS={MINIMUM_MACOS}");
	println!("cargo:rustc-link-arg=-mmacosx-version-min={MINIMUM_MACOS}");
	Ok(())
}

fn required_env(name: &str) -> Result<String, String> {
	env::var(name).map_err(|error| format!("required environment {name} is absent: {error}"))
}

fn required_path(name: &str) -> Result<PathBuf, String> {
	required_env(name).map(PathBuf::from)
}

fn validate_deployment_target() -> Result<String, String> {
	let requested =
		env::var("MACOSX_DEPLOYMENT_TARGET").unwrap_or_else(|_| MINIMUM_MACOS.to_string());
	let requested = AppleVersion::parse(&requested, "macOS deployment target")?;
	let minimum = AppleVersion::parse(MINIMUM_MACOS, "minimum macOS version")?;
	if requested < minimum {
		return Err(format!(
			"MACOSX_DEPLOYMENT_TARGET={requested} is unsupported; minimum is \
			 {MINIMUM_MACOS}"
		));
	}
	println!("cargo:rustc-link-arg=-mmacosx-version-min={requested}");
	println!("cargo:rustc-env=EMELEX_MINIMUM_MACOS={MINIMUM_MACOS}");
	Ok(requested.to_string())
}

fn validate_sdk() -> Result<PathBuf, String> {
	let version_output = Command::new("/usr/bin/xcrun")
		.args(["--sdk", "macosx", "--show-sdk-version"])
		.output()
		.map_err(|error| format!("failed to query macOS SDK with xcrun: {error}"))?;
	if !version_output.status.success() {
		return Err(format!(
			"xcrun could not query macOS SDK: {}",
			String::from_utf8_lossy(&version_output.stderr).trim()
		));
	}
	let sdk_output = String::from_utf8(version_output.stdout)
		.map_err(|error| format!("xcrun returned a non-UTF-8 macOS SDK version: {error}"))?;
	let sdk = sdk_output.trim_end_matches(['\r', '\n']);
	let sdk = AppleVersion::parse(sdk, "macOS SDK version")?;
	let minimum = AppleVersion::parse(MINIMUM_MACOS, "minimum macOS version")?;
	if sdk < minimum {
		return Err(format!(
			"macOS SDK {sdk} is unsupported; Emelex requires SDK {MINIMUM_MACOS} \
			 or newer"
		));
	}
	let path_output = Command::new("/usr/bin/xcrun")
		.args(["--sdk", "macosx", "--show-sdk-path"])
		.output()
		.map_err(|error| format!("failed to query macOS SDK path with xcrun: {error}"))?;
	if !path_output.status.success() {
		return Err(format!(
			"xcrun could not query macOS SDK path: {}",
			String::from_utf8_lossy(&path_output.stderr).trim()
		));
	}
	let path_output = String::from_utf8(path_output.stdout)
		.map_err(|error| format!("xcrun returned a non-UTF-8 macOS SDK path: {error}"))?;
	let path = PathBuf::from(path_output.trim_end_matches(['\r', '\n']));
	if !path.is_dir() {
		return Err(format!("macOS SDK path does not exist: {}", path.display()));
	}
	Ok(path)
}

fn validate_metal_toolchain() -> Result<Option<String>, String> {
	let configured = env::var("TOOLCHAINS").ok();
	if metal_version(configured.as_deref())?.0 {
		return Ok(configured);
	}
	if let Some(configured) = configured.as_deref() {
		return Err(format!(
			"TOOLCHAINS={configured} does not provide the Metal compiler for the macOS SDK"
		));
	}

	let find = Command::new("/usr/bin/xcrun")
		.args(["--find", "metal"])
		.output()
		.map_err(|error| format!("failed to locate the Metal compiler with xcrun: {error}"))?;
	if !find.status.success() {
		return Err(format!(
			"Metal Toolchain is unavailable; install it with `xcodebuild \
			 -downloadComponent MetalToolchain`: {}",
			String::from_utf8_lossy(&find.stderr).trim()
		));
	}
	let executable = PathBuf::from(
		String::from_utf8(find.stdout)
			.map_err(|error| format!("xcrun returned a non-UTF-8 Metal path: {error}"))?
			.trim_end_matches(['\r', '\n']),
	);
	let toolchain = executable
		.ancestors()
		.find(|path| path.extension() == Some(OsStr::new("xctoolchain")))
		.ok_or_else(|| {
			format!(
				"xcrun located Metal outside an xctoolchain: {}",
				executable.display()
			)
		})?;
	let info = toolchain.join("ToolchainInfo.plist");
	let identifier = Command::new("/usr/bin/plutil")
		.args(["-extract", "Identifier", "raw", "-o", "-"])
		.arg(&info)
		.output()
		.map_err(|error| format!("failed to inspect {}: {error}", info.display()))?;
	if !identifier.status.success() {
		return Err(format!(
			"cannot read Metal toolchain identifier from {}: {}",
			info.display(),
			String::from_utf8_lossy(&identifier.stderr).trim()
		));
	}
	let identifier = String::from_utf8(identifier.stdout)
		.map_err(|error| format!("Metal toolchain identifier is not UTF-8: {error}"))?
		.trim_end_matches(['\r', '\n'])
		.to_string();
	if identifier.is_empty() || identifier.chars().any(char::is_whitespace) {
		return Err(format!(
			"Metal toolchain has an invalid identifier in {}",
			info.display()
		));
	}
	let (available, failure) = metal_version(Some(&identifier))?;
	if !available {
		return Err(format!(
			"downloaded Metal toolchain {identifier} cannot compile for the macOS SDK: {failure}"
		));
	}
	Ok(Some(identifier))
}

fn metal_version(toolchains: Option<&str>) -> Result<(bool, String), String> {
	let mut command = Command::new("/usr/bin/xcrun");
	command.args(["--sdk", "macosx", "metal", "--version"]);
	if let Some(toolchains) = toolchains {
		command.env("TOOLCHAINS", toolchains);
	}
	let output = command
		.output()
		.map_err(|error| format!("failed to run the Metal compiler with xcrun: {error}"))?;
	let failure = String::from_utf8_lossy(&output.stderr).trim().to_string();
	Ok((output.status.success(), failure))
}

fn emit_rerun_directives(paths: &[&Path]) {
	println!("cargo:rerun-if-changed=build.rs");
	for path in paths {
		println!("cargo:rerun-if-changed={}", path.display());
	}
	println!("cargo:rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");
	for name in [
		"CC",
		"CXX",
		"CFLAGS",
		"CXXFLAGS",
		"HOST_CC",
		"HOST_CXX",
		"HOST_CFLAGS",
		"HOST_CXXFLAGS",
		"TARGET_CC",
		"TARGET_CXX",
		"TARGET_CFLAGS",
		"TARGET_CXXFLAGS",
		"CC_aarch64-apple-darwin",
		"CXX_aarch64-apple-darwin",
		"CFLAGS_aarch64-apple-darwin",
		"CXXFLAGS_aarch64-apple-darwin",
		"CC_aarch64_apple_darwin",
		"CXX_aarch64_apple_darwin",
		"CFLAGS_aarch64_apple_darwin",
		"CXXFLAGS_aarch64_apple_darwin",
		"LIBCLANG_PATH",
		"DEVELOPER_DIR",
		"TOOLCHAINS",
		"SDKROOT",
	] {
		println!("cargo:rerun-if-env-changed={name}");
	}
}

struct NativeBuild<'a> {
	source_root: &'a Path,
	mlx_c_dir: &'a Path,
	mlx_dir: &'a Path,
	metal_cpp_dir: &'a Path,
	json_dir: &'a Path,
	fmt_dir: &'a Path,
	module_cache: &'a Path,
	fetch_cache: &'a Path,
	sdk_path: &'a Path,
	deployment_target: &'a str,
	metal_toolchain: Option<&'a str>,
}

fn build_native(build: &NativeBuild<'_>) -> Result<PathBuf, String> {
	let mlx_source = path_text(build.mlx_dir)?;
	let metal_cpp_source = path_text(build.metal_cpp_dir)?;
	let json_source = path_text(build.json_dir)?;
	let fmt_source = path_text(build.fmt_dir)?;
	let module_cache_text = path_text(build.module_cache)?;
	let mut cfg = cmake::Config::new(build.mlx_c_dir);
	cfg.define("MLX_C_BUILD_EXAMPLES", "OFF")
		.define("BUILD_SHARED_LIBS", "OFF")
		.define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
		.define("CMAKE_OSX_ARCHITECTURES", "arm64")
		.define("CMAKE_OSX_DEPLOYMENT_TARGET", build.deployment_target)
		.define("CMAKE_OSX_SYSROOT", path_text(build.sdk_path)?)
		.define("EMELEX_SOURCE_PREFIX", path_text(build.source_root)?)
		.define("FETCHCONTENT_BASE_DIR", path_text(build.fetch_cache)?)
		.define("FETCHCONTENT_FULLY_DISCONNECTED", "ON")
		.define("FETCHCONTENT_UPDATES_DISCONNECTED", "ON")
		.define("FETCHCONTENT_SOURCE_DIR_MLX", mlx_source)
		.define("FETCHCONTENT_SOURCE_DIR_METAL_CPP", metal_cpp_source)
		.define("FETCHCONTENT_SOURCE_DIR_JSON", json_source)
		.define("FETCHCONTENT_SOURCE_DIR_FMT", fmt_source)
		.define("MLX_CLANG_MODULE_CACHE_PATH", module_cache_text)
		.define("MLX_BUILD_BENCHMARKS", "OFF")
		.define("MLX_BUILD_CUDA", "OFF")
		.define("MLX_BUILD_EXAMPLES", "OFF")
		.define("MLX_BUILD_GGUF", "OFF")
		.define("MLX_BUILD_JACCL", "OFF")
		.define("MLX_BUILD_PYTHON_BINDINGS", "OFF")
		.define("MLX_BUILD_TESTS", "OFF")
		.define("MLX_ENABLE_X64_MAC", "OFF")
		.define("MLX_METAL_JIT", "OFF")
		.define("MLX_USE_CCACHE", "OFF")
		.env("CLANG_MODULE_CACHE_PATH", build.module_cache)
		.profile("Release");
	if let Some(toolchain) = build.metal_toolchain {
		cfg.env("TOOLCHAINS", toolchain);
	}
	Ok(cfg.build())
}

fn verify_native_artifacts(dst: &Path, source_root: &Path) -> Result<(), String> {
	let source_root = path_text(source_root)?.as_bytes();
	let private_home_prefix = [b"/".as_slice(), b"Users", b"/"].concat();
	let forbidden = [source_root, private_home_prefix.as_slice()];
	for name in ["libmlx.a", "libmlxc.a", "mlx.metallib"] {
		let artifact = dst.join("lib").join(name);
		let mut file = File::open(&artifact).map_err(|error| {
			io_error(
				"open native artifact for provenance audit",
				&artifact,
				&error,
			)
		})?;
		let mut bytes = Vec::new();
		file.read_to_end(&mut bytes).map_err(|error| {
			io_error(
				"read native artifact for provenance audit",
				&artifact,
				&error,
			)
		})?;
		for needle in forbidden {
			if bytes.windows(needle.len()).any(|window| window == needle) {
				return Err(format!(
					"native artifact {} embeds forbidden build provenance {:?}",
					artifact.display(),
					String::from_utf8_lossy(needle)
				));
			}
		}
	}
	Ok(())
}

fn emit_link_directives(dst: &Path) {
	println!(
		"cargo:rustc-link-search=native={}",
		dst.join("lib").display()
	);
	let build_dir = dst.join("build");
	for subdir in ["_deps/mlx-build", "_deps/mlx-build/mlx", "lib"] {
		let candidate = build_dir.join(subdir);
		if candidate.exists() {
			println!("cargo:rustc-link-search=native={}", candidate.display());
		}
	}
	println!("cargo:rustc-link-lib=static=mlxc");
	println!("cargo:rustc-link-lib=static=mlx");
	println!("cargo:rustc-link-lib=c++");
	for framework in [
		"Accelerate",
		"Foundation",
		"Metal",
		"MetalPerformanceShaders",
		"QuartzCore",
	] {
		println!("cargo:rustc-link-lib=framework={framework}");
	}
}

fn prepare_metallib(dst: &Path, out_dir: &Path) -> Result<(), String> {
	let build_dir = dst.join("build");
	let candidates = [
		dst.join("lib/mlx.metallib"),
		build_dir.join("_deps/mlx-build/mlx/backend/metal/kernels/mlx.metallib"),
		build_dir.join("_deps/mlx-build/mlx/backend/metal/mlx.metallib"),
	];
	let source = candidates
		.iter()
		.find(|candidate| candidate.is_file())
		.ok_or_else(|| {
			format!(
				"MLX build completed without mlx.metallib; checked {}",
				candidates
					.iter()
					.map(|path| path.display().to_string())
					.collect::<Vec<_>>()
					.join(", ")
			)
		})?;
	let bytes =
		fs::read(source).map_err(|error| io_error("read built metallib", source, &error))?;
	let digest = hex::encode(Sha256::digest(&bytes));
	let compressed = out_dir.join("mlx.metallib.zst");
	let reader = BufReader::new(bytes.as_slice());
	let file = File::create(&compressed)
		.map_err(|error| io_error("create compressed metallib", &compressed, &error))?;
	let mut writer = BufWriter::new(file);
	zstd::stream::copy_encode(reader, &mut writer, 19)
		.map_err(|error| format!("compress metallib: {error}"))?;
	writer
		.flush()
		.map_err(|error| io_error("flush compressed metallib", &compressed, &error))?;
	println!(
		"cargo:rustc-env=EMELEX_METALLIB_ZST_PATH={}",
		compressed.display()
	);
	println!("cargo:rustc-env=EMELEX_METALLIB_SHA256={digest}");
	println!("cargo:rustc-env=EMELEX_METALLIB_SIZE={}", bytes.len());
	Ok(())
}

fn generate_bindings(
	mlx_c_dir: &Path,
	out_dir: &Path,
	module_cache: &Path,
	sdk_path: &Path,
	deployment_target: &str,
) -> Result<(), String> {
	let header = mlx_c_dir.join("mlx/c/mlx.h");
	let header_text = path_text(&header)?;
	let include = format!("-I{}", path_text(mlx_c_dir)?);
	let module_cache = format!("-fmodules-cache-path={}", path_text(module_cache)?);
	let sdk_path = path_text(sdk_path)?;
	let minimum = format!("-mmacosx-version-min={deployment_target}");
	let bindings = bindgen::Builder::default()
		.rust_edition(bindgen::RustEdition::Edition2024)
		.header(header_text)
		.clang_arg(include)
		.clang_arg(module_cache)
		.clang_arg("-target")
		.clang_arg("arm64-apple-macos")
		.clang_arg("-isysroot")
		.clang_arg(sdk_path)
		.clang_arg(minimum)
		.allowlist_function("mlx_.*")
		.allowlist_function("_mlx_.*")
		.allowlist_type("mlx_.*")
		.allowlist_var("MLX_.*")
		.layout_tests(false)
		.generate_comments(false)
		.derive_default(true)
		.generate()
		.map_err(|error| format!("bindgen failed for {}: {error}", header.display()))?;
	let output = out_dir.join("bindings.rs");
	bindings
		.write_to_file(&output)
		.map_err(|error| io_error("write generated mlx-c bindings", &output, &error))
}

fn path_text(path: &Path) -> Result<&str, String> {
	path.to_str()
		.ok_or_else(|| format!("path is not valid UTF-8: {}", path.display()))
}

fn io_error(operation: &str, path: &Path, error: &std::io::Error) -> String {
	format!("{operation} {}: {error}", path.display())
}

/// Strict one- or two-component Apple platform version.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct AppleVersion {
	major: u32,
	minor: u32,
}

impl AppleVersion {
	/// Parses exactly `MAJOR` or `MAJOR.MINOR`, using ASCII digits only.
	pub fn parse(value: &str, kind: &str) -> Result<Self, String> {
		let mut parts = value.split('.');
		let major = parse_component(parts.next(), value, kind)?;
		let minor = match parts.next() {
			Some(part) => parse_component(Some(part), value, kind)?,
			None => 0,
		};
		if parts.next().is_some() {
			return Err(invalid_version(value, kind));
		}
		Ok(Self { major, minor })
	}
}

impl fmt::Display for AppleVersion {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(formatter, "{}.{}", self.major, self.minor)
	}
}

fn parse_component(part: Option<&str>, value: &str, kind: &str) -> Result<u32, String> {
	let part = part.ok_or_else(|| invalid_version(value, kind))?;
	if part.is_empty() || !part.bytes().all(|byte| byte.is_ascii_digit()) {
		return Err(invalid_version(value, kind));
	}
	part.parse::<u32>()
		.map_err(|_| invalid_version(value, kind))
}

fn invalid_version(value: &str, kind: &str) -> String {
	format!("invalid {kind} {value:?}; expected MAJOR or MAJOR.MINOR")
}
