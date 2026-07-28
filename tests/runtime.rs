//! Embedded runtime relocation, extraction, and process-latch tests.

#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{
	ffi::CString,
	fs,
	os::unix::{
		ffi::OsStrExt as _,
		fs::{PermissionsExt as _, symlink},
	},
	path::{Path, PathBuf},
	process::{Command, Output},
};

const CHILD_MODE: &str = "EMELEX_RUNTIME_TEST_MODE";
const CHILD_HOME: &str = "EMELEX_RUNTIME_TEST_HOME";
const CHILD_SECOND_HOME: &str = "EMELEX_RUNTIME_TEST_SECOND_HOME";
const METAL_COMPLETION_SURVIVED: &str = "emelex-metal-completion-failure-survived";
const METAL_COMPLETION_HEADLESS_SKIP: &str = "emelex-metal-completion-headless-skip";
const REQUIRE_PHYSICAL_GPU: &str = "EMELEX_REQUIRE_PHYSICAL_GPU";
const MLX_MAX_OPS_PER_BUFFER: &str = "MLX_MAX_OPS_PER_BUFFER";

unsafe extern "C" {
	fn mlx_emelex_test_inject_metal_completion_failure();
}

#[test]
fn runtime_child() {
	let Some(mode) = std::env::var_os(CHILD_MODE) else {
		return;
	};
	let home =
		PathBuf::from(std::env::var_os(CHILD_HOME).expect("child Emelex home is configured"));
	match mode.to_string_lossy().as_ref() {
		"initialize" => {
			let asset = emelex::runtime::initialize(&home).expect("runtime initializes");
			assert!(asset.metallib().is_file());
		}
		"recommended_then_initialize" => {
			match emelex::runtime::recommended_max_working_set_size() {
				Ok(bytes) => {
					assert!(bytes > 0);
					emelex::runtime::initialize(&home)
						.expect("runtime initializes after budget query");
					emelex::runtime::verify_engine().expect("embedded MLX runtime evaluates");
				}
				Err(emelex::runtime::RuntimeError::MetalDeviceUnavailable) => {
					emelex::runtime::initialize(&home)
						.expect("runtime asset initializes without a Metal device");
					let error = emelex::runtime::verify_engine()
						.expect_err("headless engine verification fails cleanly");
					assert!(
						matches!(error, emelex::runtime::RuntimeError::MetalDeviceUnavailable),
						"{error}"
					);
				}
				Err(error) => panic!("unexpected Metal budget failure: {error}"),
			}
		}
		"conflict" => {
			emelex::runtime::initialize(&home).expect("first home initializes");
			let second = PathBuf::from(
				std::env::var_os(CHILD_SECOND_HOME).expect("second child home is configured"),
			);
			assert!(matches!(
				emelex::runtime::initialize(&second),
				Err(emelex::runtime::RuntimeError::HomeConflict { .. })
			));
		}
		"reject_symlink" => {
			let expected = fs::canonicalize(&home)
				.expect("canonical owned home")
				.join("cache");
			let error = emelex::runtime::initialize(&home)
				.expect_err("symlinked owned cache must be rejected");
			assert!(
				matches!(
					&error,
					emelex::runtime::RuntimeError::Home(
						emelex::home::HomeError::Io { path, .. }
					) if path == &expected
				),
				"{error:?}"
			);
		}
		"invalid_metallib_error" => {
			let asset = emelex::runtime::initialize(&home).expect("runtime initializes");
			fs::write(asset.metallib(), b"invalid metallib")
				.expect("replace metallib before MLX initialization");
			let error = emelex::runtime::verify_engine().expect_err("invalid metallib fails");
			assert!(matches!(error, emelex::runtime::RuntimeError::Mlx(_)));
			let message = error.to_string();
			let private_home_prefix = ["/", "Users", "/"].concat();
			assert!(!message.contains(&private_home_prefix), "{message}");
		}
		"metal_completion_failure" => {
			emelex::runtime::initialize(&home).expect("runtime initializes");
			match emelex::runtime::verify_engine() {
				Ok(()) => {}
				Err(emelex::runtime::RuntimeError::MetalDeviceUnavailable) => {
					println!("{METAL_COMPLETION_HEADLESS_SKIP}");
					return;
				}
				Err(error) => panic!("unexpected baseline Metal failure: {error}"),
			}
			// SAFETY: this private vendored test seam has no arguments and
			// atomically arms exactly one completion callback.
			unsafe {
				mlx_emelex_test_inject_metal_completion_failure();
			}
			let error = emelex::runtime::verify_engine()
				.expect_err("completion callback failure reaches the Rust Result");
			assert!(
				matches!(
					error,
					emelex::runtime::RuntimeError::Mlx(ref detail)
						if detail.contains("Command buffer execution failed without an NSError")
				),
				"{error}"
			);
			emelex::runtime::verify_engine()
				.expect("stream remains usable after observing the callback failure");
			println!("{METAL_COMPLETION_SURVIVED}");
		}
		other => panic!("unknown runtime child mode {other}"),
	}
}

#[test]
fn runtime_is_relocatable_and_budget_query_does_not_latch_mlx() {
	let temp = tempfile::tempdir().expect("temporary directory");
	let relocated = temp.path().join("relocated-runtime-test");
	fs::copy(
		std::env::current_exe().expect("current test executable"),
		&relocated,
	)
	.expect("copy test executable");
	let output = child_output(
		&relocated,
		&temp.path().join("home"),
		"recommended_then_initialize",
	);
	assert_success(&output);
	assert!(!temp.path().join("mlx.metallib").exists());
}

#[test]
fn runtime_extraction_is_race_safe_and_repairs_corruption() {
	let temp = tempfile::tempdir().expect("temporary directory");
	let home = temp.path().join("home");
	let executable = std::env::current_exe().expect("current test executable");
	let children = (0..4)
		.map(|_| child_command(&executable, &home, "initialize").spawn())
		.collect::<Result<Vec<_>, _>>()
		.expect("spawn extraction children");
	for child in children {
		assert_success(&child.wait_with_output().expect("wait for extraction child"));
	}
	assert_eq!(
		fs::read(home.join(".emelex-root")).expect("read ownership marker"),
		b"emelex-home-v1\n"
	);
	let marker = fs::metadata(home.join(".emelex-root")).expect("ownership marker metadata");
	assert!(marker.is_file());
	assert_eq!(marker.permissions().mode() & 0o777, 0o600);
	for directory in ["cache", "memory", "models", "temp"] {
		let metadata =
			fs::metadata(home.join(directory)).expect("standard home directory metadata");
		assert!(metadata.is_dir());
		assert_eq!(metadata.permissions().mode() & 0o777, 0o700);
	}
	let metallib = extracted_metallib(&home);
	fs::write(&metallib, b"corrupt").expect("corrupt cached metallib");
	let output = child_output(&executable, &home, "initialize");
	assert_success(&output);
	let metadata = fs::metadata(&metallib).expect("repaired metallib metadata");
	assert!(metadata.len() > 7);
	assert_eq!(metadata.permissions().mode() & 0o777, 0o600);

	fs::remove_file(&metallib).expect("remove repaired metallib");
	let fifo = CString::new(metallib.as_os_str().as_bytes()).expect("FIFO path");
	// SAFETY: `fifo` is a valid path and mode is a valid owner-only FIFO mode.
	assert_eq!(unsafe { libc::mkfifo(fifo.as_ptr(), 0o600) }, 0);
	let output = child_output(&executable, &home, "initialize");
	assert_success(&output);
	assert!(fs::metadata(&metallib).expect("replaced FIFO").is_file());
}

#[test]
fn runtime_rejects_owned_symlinks_and_conflicting_home() {
	let temp = tempfile::tempdir().expect("temporary directory");
	let home = temp.path().join("home");
	let outside = temp.path().join("outside");
	fs::create_dir(&outside).expect("outside directory");
	emelex::home::EmelexHome::prepare(&home).expect("owned home");
	fs::remove_dir(home.join("cache")).expect("remove owned cache");
	symlink(&outside, home.join("cache")).expect("cache symlink");
	let executable = std::env::current_exe().expect("current test executable");
	assert_success(&child_output(&executable, &home, "reject_symlink"));

	let first = temp.path().join("first");
	let second = temp.path().join("second");
	let output = child_command(&executable, &first, "conflict")
		.env(CHILD_SECOND_HOME, second)
		.output()
		.expect("run conflict child");
	assert_success(&output);
}

#[test]
fn runtime_reports_invalid_percent_path_without_c_format_ub() {
	let temp = tempfile::tempdir().expect("temporary directory");
	let home = temp.path().join("home-%n-%s");
	let executable = std::env::current_exe().expect("current test executable");
	assert_success(&child_output(&executable, &home, "invalid_metallib_error"));
}

#[test]
fn metal_completion_callback_failure_returns_error_without_aborting() {
	let temp = tempfile::tempdir().expect("temporary directory");
	let executable = std::env::current_exe().expect("current test executable");
	let output = child_output(
		&executable,
		&temp.path().join("home"),
		"metal_completion_failure",
	);
	assert_success(&output);
	let stdout = String::from_utf8_lossy(&output.stdout);
	if stdout.contains(METAL_COMPLETION_HEADLESS_SKIP) {
		assert!(
			std::env::var_os(REQUIRE_PHYSICAL_GPU).is_none(),
			"{REQUIRE_PHYSICAL_GPU} requires the physical-GPU sentinel, but this host is headless"
		);
		eprintln!("skipped Metal completion regression: no physical Metal device");
		return;
	}
	assert!(
		stdout.contains(METAL_COMPLETION_SURVIVED),
		"child omitted survival sentinel\nstdout:\n{}\nstderr:\n{}",
		stdout,
		String::from_utf8_lossy(&output.stderr)
	);
}

fn child_command(executable: &Path, home: &Path, mode: &str) -> Command {
	let mut command = Command::new(executable);
	command
		.arg("--exact")
		.arg("runtime_child")
		.arg("--nocapture")
		.env(CHILD_MODE, mode)
		.env(CHILD_HOME, home);
	if mode == "metal_completion_failure" {
		// Force the evaluated operation into an auto-committed buffer. The
		// following synchronization must still observe that earlier failure.
		command.env(MLX_MAX_OPS_PER_BUFFER, "0");
	}
	command
}

fn child_output(executable: &Path, home: &Path, mode: &str) -> Output {
	child_command(executable, home, mode)
		.output()
		.expect("run runtime child")
}

fn extracted_metallib(home: &Path) -> PathBuf {
	let digest_dir = fs::read_dir(home.join("cache/runtime/mlx"))
		.expect("runtime digest root")
		.next()
		.expect("one digest directory")
		.expect("read digest directory")
		.path();
	digest_dir.join("mlx.metallib")
}

fn assert_success(output: &Output) {
	assert!(
		output.status.success(),
		"child failed\nstdout:\n{}\nstderr:\n{}",
		String::from_utf8_lossy(&output.stdout),
		String::from_utf8_lossy(&output.stderr)
	);
}
