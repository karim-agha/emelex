use std::io::Write as _;

use super::*;

#[test]
fn unknown_global_key_fails() {
	let temp = tempfile::tempdir().expect("temporary directory");
	let home = EmelexHome::prepare(&temp.path().join("home")).expect("home");
	fs::write(home.config_file(), "mystery = true\n").expect("write config");
	let error = Config::load(&home, temp.path(), false).expect_err("unknown key");
	assert!(matches!(error, ConfigError::Parse { .. }));
}

#[test]
fn project_fifo_is_rejected_without_blocking() {
	use std::{
		ffi::CString,
		os::unix::{ffi::OsStrExt as _, fs::OpenOptionsExt as _},
		sync::mpsc,
		time::Duration,
	};

	let temp = tempfile::tempdir().expect("temporary directory");
	let root = temp.path().join("project");
	fs::create_dir_all(root.join(".git")).expect("git marker");
	let fifo = root.join(".emelex.toml");
	let fifo_name = CString::new(fifo.as_os_str().as_bytes()).expect("FIFO path");
	// SAFETY: `fifo_name` is a live NUL-terminated path and the mode is valid.
	assert_eq!(unsafe { libc::mkfifo(fifo_name.as_ptr(), 0o600) }, 0);
	let home = EmelexHome::prepare(&temp.path().join("home")).expect("home");
	let (sender, receiver) = mpsc::channel();
	let worker = std::thread::spawn(move || {
		let _ = sender.send(Config::load(&home, &root, true));
	});

	let result = match receiver.recv_timeout(Duration::from_secs(1)) {
		Ok(result) => result,
		Err(error) => {
			// Unblock an old blocking reader so this regression fails cleanly
			// instead of leaving a stuck test thread behind.
			let _writer = fs::OpenOptions::new()
				.write(true)
				.custom_flags(libc::O_NONBLOCK)
				.open(&fifo)
				.expect("unblock FIFO reader");
			worker.join().expect("config worker");
			panic!("project FIFO blocked configuration loading: {error}");
		}
	};
	worker.join().expect("config worker");
	let error = result.expect_err("FIFO is not a regular configuration file");
	assert!(error.to_string().contains("regular file"));
}

#[test]
fn project_config_can_only_reduce_global_resource_limits() {
	let temp = tempfile::tempdir().expect("temporary directory");
	let root = temp.path().join("project");
	fs::create_dir_all(root.join(".git")).expect("git marker");
	let home = EmelexHome::prepare(&temp.path().join("home")).expect("home");
	fs::write(
		home.config_file(),
		"[inference]\nmax_tokens = 100\n[agent]\nweb = false\n",
	)
	.expect("global config");
	let mut project = fs::File::create(root.join(".emelex.toml")).expect("project config");
	writeln!(
		project,
		"[inference]\nmax_tokens = 200\n[memory]\nrecall_entries = 4"
	)
	.expect("project config text");
	let (config, sources) = Config::load(&home, &root, true).expect("config loads");
	assert_eq!(config.inference.max_tokens, 100);
	assert_eq!(config.memory.recall_entries, 4);
	assert!(sources.project.is_some());
}

#[test]
fn global_generation_limit_cannot_exceed_context() {
	let temp = tempfile::tempdir().expect("temporary directory");
	let home = EmelexHome::prepare(&temp.path().join("home")).expect("home");
	fs::write(
		home.config_file(),
		"[inference]\nmax_tokens = 200\ncontext_tokens = 100\n",
	)
	.expect("global config");

	let error = Config::load(&home, temp.path(), false).expect_err("invalid token relationship");
	assert!(error.to_string().contains("max_tokens"));
	assert!(error.to_string().contains("context_tokens"));
}

#[test]
fn project_context_reduction_cannot_cross_global_generation_limit() {
	let temp = tempfile::tempdir().expect("temporary directory");
	let root = temp.path().join("project");
	fs::create_dir_all(root.join(".git")).expect("git marker");
	let home = EmelexHome::prepare(&temp.path().join("home")).expect("home");
	fs::write(
		home.config_file(),
		"[inference]\nmax_tokens = 400\ncontext_tokens = 1000\n",
	)
	.expect("global config");
	fs::write(
		root.join(".emelex.toml"),
		"[inference]\ncontext_tokens = 200\n",
	)
	.expect("project config");

	let error = Config::load(&home, &root, true).expect_err("invalid merged token relationship");
	assert!(error.to_string().contains("max_tokens"));
	assert!(error.to_string().contains("context_tokens"));
}

#[test]
fn project_config_cannot_disable_top_k_or_shorten_retention() {
	let temp = tempfile::tempdir().expect("temporary directory");
	let root = temp.path().join("project");
	fs::create_dir_all(root.join(".git")).expect("git marker");
	let home = EmelexHome::prepare(&temp.path().join("home")).expect("home");
	fs::write(
		root.join(".emelex.toml"),
		"[inference]\ntop_k = 0\n[memory]\nretention_days = 1\n",
	)
	.expect("project config");

	let error = Config::load(&home, &root, true).expect_err("destructive authority rejected");
	let rendered = error.to_string();
	assert!(rendered.contains("inference.top_k = 0"));
	assert!(rendered.contains("memory.retention_days"));
}

#[test]
fn default_agent_limits_match_builtin_tool_bounds() {
	let config = Config::default();
	config.validate().expect("default config");
	assert_eq!(
		config.agent.shell_output_bytes,
		crate::agent::MAX_SHELL_OUTPUT_BYTES
	);
	assert_eq!(
		config.agent.web_response_bytes,
		crate::agent::MAX_WEB_RESPONSE_BYTES
	);
}

#[test]
fn no_project_config_uses_global_only() {
	let temp = tempfile::tempdir().expect("temporary directory");
	let root = temp.path().join("project");
	fs::create_dir_all(root.join(".git")).expect("git marker");
	fs::write(root.join(".emelex.toml"), "[inference]\nmax_tokens = 200\n")
		.expect("project config");
	let home = EmelexHome::prepare(&temp.path().join("home")).expect("home");
	fs::write(home.config_file(), "[inference]\nmax_tokens = 100\n").expect("global config");
	let (config, sources) = Config::load(&home, &root, false).expect("config loads");
	assert_eq!(config.inference.max_tokens, 100);
	assert!(sources.project.is_none());
}

#[test]
fn project_config_rejects_model_and_seed_authority() {
	let temp = tempfile::tempdir().expect("temporary directory");
	let root = temp.path().join("project");
	fs::create_dir_all(root.join(".git")).expect("git marker");
	let home = EmelexHome::prepare(&temp.path().join("home")).expect("home");
	fs::write(
		home.config_file(),
		"default_model = \"mlx-community/example\"\n[inference]\nseed = 42\n",
	)
	.expect("global config");
	fs::write(
		root.join(".emelex.toml"),
		"default_model = { clear = true }\n[inference]\nseed = { clear = true }\n",
	)
	.expect("project config");

	let error = Config::load(&home, &root, true).expect_err("authority fields rejected");
	let rendered = error.to_string();
	assert!(rendered.contains("default_model"));
	assert!(rendered.contains("inference.seed"));
}

#[test]
fn project_config_rejects_model_selection_without_global_model() {
	let temp = tempfile::tempdir().expect("temporary directory");
	let root = temp.path().join("project");
	fs::create_dir_all(root.join(".git")).expect("git marker");
	let home = EmelexHome::prepare(&temp.path().join("home")).expect("home");
	fs::write(
		root.join(".emelex.toml"),
		"default_model = \"mlx-community/example\"\n",
	)
	.expect("project config");

	assert!(matches!(
		Config::load(&home, &root, true),
		Err(ConfigError::Parse { .. })
	));
}

#[test]
fn project_config_cannot_raise_global_tool_authority_or_resource_limits() {
	let temp = tempfile::tempdir().expect("temporary directory");
	let root = temp.path().join("project");
	fs::create_dir_all(root.join(".git")).expect("git marker");
	let home = EmelexHome::prepare(&temp.path().join("home")).expect("home");
	fs::write(
		home.config_file(),
		"[agent]\nshell = false\nweb = false\nmax_turns = 5\nshell_timeout_seconds = 10\n",
	)
	.expect("global config");
	fs::write(
		root.join(".emelex.toml"),
		"[agent]\nshell = true\nweb = true\nmax_turns = 50\nshell_timeout_seconds = 100\n",
	)
	.expect("project config");

	let (config, _) = Config::load(&home, &root, true).expect("config loads");

	assert!(!config.agent.shell);
	assert!(!config.agent.web);
	assert_eq!(config.agent.max_turns, 5);
	assert_eq!(config.agent.shell_timeout_seconds, 10);
}

#[test]
fn shell_timeout_validation_matches_shell_tool_boundary() {
	let mut config = Config::default();
	config.agent.shell_timeout_seconds = crate::agent::MAX_SHELL_TIMEOUT_SECONDS;
	config.validate().expect("hard ceiling remains valid");
	config.agent.shell_timeout_seconds = crate::agent::MAX_SHELL_TIMEOUT_SECONDS + 1;
	assert!(matches!(config.validate(), Err(ConfigError::Invalid(_))));
}

#[test]
fn invalid_model_reference_fails_strict_config_loading() {
	let temp = tempfile::tempdir().expect("temporary directory");
	let home = EmelexHome::prepare(&temp.path().join("home")).expect("home");
	fs::write(home.config_file(), "default_model = \"../escape\"\n").expect("global config");

	assert!(matches!(
		Config::load(&home, temp.path(), false),
		Err(ConfigError::Parse { .. })
	));
}

#[test]
fn symlinked_global_config_is_rejected() {
	use std::os::unix::fs::symlink;

	let temp = tempfile::tempdir().expect("temporary directory");
	let home = EmelexHome::prepare(&temp.path().join("home")).expect("home");
	let target = temp.path().join("outside.toml");
	fs::write(&target, "[agent]\nshell = true\n").expect("target config");
	symlink(&target, home.config_file()).expect("config symlink");

	assert!(matches!(
		Config::load(&home, temp.path(), false),
		Err(ConfigError::Read { .. })
	));
}

#[test]
fn global_default_model_update_preserves_other_settings() {
	let temp = tempfile::tempdir().expect("temporary directory");
	let home = EmelexHome::prepare(&temp.path().join("home")).expect("home");
	fs::write(
		home.config_file(),
		"# keep behavior\n[inference]\nmax_tokens = 100\n[agent]\nshell = false\n",
	)
	.expect("global config");
	let model = ModelRef::parse("mlx-community/example").expect("model");

	Config::write_global_default_model(&home, Some(&model)).expect("write default");
	let (config, _) = Config::load(&home, temp.path(), false).expect("load updated config");
	assert_eq!(config.default_model.as_ref(), Some(&model));
	assert_eq!(config.inference.max_tokens, 100);
	assert!(!config.agent.shell);

	Config::write_global_default_model(&home, None).expect("clear default");
	let (config, _) = Config::load(&home, temp.path(), false).expect("load cleared config");
	assert!(config.default_model.is_none());
	assert_eq!(config.inference.max_tokens, 100);
}

#[test]
fn global_default_model_update_rejects_unsafe_existing_file() {
	use std::os::unix::fs::symlink;

	let temp = tempfile::tempdir().expect("temporary directory");
	let home = EmelexHome::prepare(&temp.path().join("home")).expect("home");
	let target = temp.path().join("outside.toml");
	fs::write(&target, "[agent]\nshell = false\n").expect("target config");
	symlink(&target, home.config_file()).expect("config symlink");
	let model = ModelRef::parse("mlx-community/example").expect("model");

	assert!(matches!(
		Config::write_global_default_model(&home, Some(&model)),
		Err(ConfigError::Read { .. })
	));
	assert_eq!(
		fs::read_to_string(target).expect("read target"),
		"[agent]\nshell = false\n"
	);
}
