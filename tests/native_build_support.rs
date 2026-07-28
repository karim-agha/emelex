//! Regression tests for native-build parsing and argv-safe preamble generators.

#![allow(clippy::expect_used)]

#[path = "../build_support.rs"]
mod build_support;

use std::{fs, os::unix::fs::PermissionsExt as _, path::Path, process::Command};

use build_support::AppleVersion;
use tempfile::TempDir;

#[test]
fn apple_versions_accept_one_or_two_ascii_numeric_components() {
	let one = AppleVersion::parse("26", "version").expect("valid one-component version");
	let two = AppleVersion::parse("026.05", "version").expect("valid two-component version");

	assert_eq!(one.to_string(), "26.0");
	assert_eq!(two.to_string(), "26.5");
}

#[test]
fn apple_versions_reject_ambiguous_or_extra_input() {
	for value in [
		"",
		".",
		"26.",
		".5",
		"26.5.1",
		" 26.5",
		"26.5 ",
		"26.a",
		"２６.５",
	] {
		assert!(
			AppleVersion::parse(value, "version").is_err(),
			"{value:?} should fail"
		);
	}
}

#[test]
fn preamble_generators_preserve_paths_with_spaces() {
	let fixture = NativeScriptFixture::new();
	fixture.run_cpu(true);
	fixture.run_metal(true);

	let cpu = fs::read_to_string(&fixture.cpu_output).expect("read CPU output");
	let metal = fs::read_to_string(&fixture.metal_output).expect("read Metal output");
	assert!(cpu.contains("cpu_preamble_marker"));
	assert!(metal.contains("metal_dependency_marker"));
	assert!(metal.contains("metal_input_marker"));
}

#[test]
fn preamble_generators_propagate_compiler_failure() {
	let fixture = NativeScriptFixture::new();
	fs::create_dir_all(fixture.cpu_output.parent().expect("CPU output parent"))
		.expect("create CPU output parent");
	fs::create_dir_all(fixture.metal_output.parent().expect("Metal output parent"))
		.expect("create Metal output parent");
	fs::write(&fixture.cpu_output, "existing CPU output").expect("write CPU sentinel");
	fs::write(&fixture.metal_output, "existing Metal output").expect("write Metal sentinel");
	fixture.run_cpu(false);
	fixture.run_metal(false);

	assert_eq!(
		fs::read_to_string(&fixture.cpu_output).expect("read CPU sentinel"),
		"existing CPU output"
	);
	assert_eq!(
		fs::read_to_string(&fixture.metal_output).expect("read Metal sentinel"),
		"existing Metal output"
	);
}

struct NativeScriptFixture {
	_temp: TempDir,
	source: std::path::PathBuf,
	compiler: std::path::PathBuf,
	metal_compiler: std::path::PathBuf,
	cpu_output: std::path::PathBuf,
	metal_output: std::path::PathBuf,
}

impl NativeScriptFixture {
	fn new() -> Self {
		let temp = tempfile::tempdir().expect("create temp directory");
		let root = temp.path().join("native fixture with spaces");
		let source = root.join("source tree");
		let cpu_dir = source.join("mlx/backend/cpu");
		let metal_dir = source.join("mlx/backend/metal/kernels");
		let jit_dir = metal_dir.join("jit");
		fs::create_dir_all(&cpu_dir).expect("create CPU source tree");
		fs::create_dir_all(&jit_dir).expect("create Metal source tree");
		fs::write(cpu_dir.join("compiled_preamble.h"), "cpu_preamble_marker\n")
			.expect("write CPU input");
		fs::write(
			metal_dir.join("fixture.h"),
			"#pragma once\nmetal_input_marker\n",
		)
		.expect("write Metal input");
		let dependency = jit_dir.join("dependency with spaces.h");
		fs::write(&dependency, "#pragma once\nmetal_dependency_marker\n")
			.expect("write Metal dependency");

		let compiler = root.join("fake cpu compiler");
		#[allow(
			clippy::literal_string_with_formatting_args,
			reason = "fixture is a literal shell script whose parameter expansion resembles Rust formatting"
		)]
		write_executable(
			&compiler,
			"#!/bin/bash\nset -eu\nif [ \"${FAIL_COMPILER:-0}\" = 1 ]; then exit 19; fi\nfound_sysroot=0\nprevious=\nfor arg in \"$@\"; do\n  if [ \"$previous\" = -isysroot ]; then\n    [ \"$arg\" = \"$EXPECT_SDK_ROOT\" ] || exit 29\n    found_sysroot=1\n  fi\n  if [ -f \"$arg\" ]; then /bin/cat \"$arg\"; fi\n  previous=$arg\ndone\n[ \"$found_sysroot\" = 1 ] || exit 31\n",
		);
		let metal_compiler = root.join("fake metal compiler");
		#[allow(
			clippy::literal_string_with_formatting_args,
			reason = "fixture is a literal shell script whose parameter expansion resembles Rust formatting"
		)]
		write_executable(
			&metal_compiler,
			"#!/bin/bash\nset -eu\nif [ \"${FAIL_COMPILER:-0}\" = 1 ]; then exit 23; fi\nfound_flag=0\nfor arg in \"$@\"; do\n  if [ \"$arg\" = \"$EXPECT_METAL_FLAG\" ]; then found_flag=1; fi\ndone\n[ \"$found_flag\" = 1 ] || exit 37\nprintf '. %s\\n' \"$FAKE_METAL_HEADER\" >&2\n",
		);

		let cpu_output = root.join("output tree/cpu preamble.cpp");
		let metal_output = root.join("output tree/fixture.cpp");
		Self {
			_temp: temp,
			source,
			compiler,
			metal_compiler,
			cpu_output,
			metal_output,
		}
	}

	fn run_cpu(&self, success: bool) {
		let script = Path::new(env!("CARGO_MANIFEST_DIR"))
			.join("vendor/mlx/mlx/backend/cpu/make_compiled_preamble.sh");
		let output = Command::new("/bin/bash")
			.arg(script)
			.args([
				self.cpu_output.as_os_str(),
				self.compiler.as_os_str(),
				self.source.as_os_str(),
				"TRUE".as_ref(),
				"arm64".as_ref(),
				self.source.as_os_str(),
			])
			.env("FAIL_COMPILER", if success { "0" } else { "1" })
			.env("EXPECT_SDK_ROOT", &self.source)
			.output()
			.expect("run CPU preamble script");
		assert_eq!(
			output.status.success(),
			success,
			"CPU script stderr: {}",
			String::from_utf8_lossy(&output.stderr)
		);
	}

	fn run_metal(&self, success: bool) {
		let script = Path::new(env!("CARGO_MANIFEST_DIR"))
			.join("vendor/mlx/mlx/backend/metal/make_compiled_preamble.sh");
		let output_dir = self.metal_output.parent().expect("Metal output parent");
		let header = self
			.source
			.join("mlx/backend/metal/kernels/jit/dependency with spaces.h");
		let output = Command::new("/bin/bash")
			.arg(script)
			.args([
				output_dir.as_os_str(),
				self.metal_compiler.as_os_str(),
				self.source.as_os_str(),
				"fixture".as_ref(),
				"-fmodules-cache-path=module cache with spaces".as_ref(),
			])
			.env("FAIL_COMPILER", if success { "0" } else { "1" })
			.env("FAKE_METAL_HEADER", header)
			.env(
				"EXPECT_METAL_FLAG",
				"-fmodules-cache-path=module cache with spaces",
			)
			.output()
			.expect("run Metal preamble script");
		assert_eq!(
			output.status.success(),
			success,
			"Metal script stderr: {}",
			String::from_utf8_lossy(&output.stderr)
		);
	}
}

fn write_executable(path: &Path, contents: &str) {
	fs::write(path, contents).expect("write fake compiler");
	let mut permissions = fs::metadata(path)
		.expect("stat fake compiler")
		.permissions();
	permissions.set_mode(0o700);
	fs::set_permissions(path, permissions).expect("make fake compiler executable");
}
