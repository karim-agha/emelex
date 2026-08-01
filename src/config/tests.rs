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
fn translate_language_pair_parses_and_is_project_scopeable() {
	let temp = tempfile::tempdir().expect("temporary directory");
	let root = temp.path().join("project");
	fs::create_dir_all(root.join(".git")).expect("git marker");
	let home = EmelexHome::prepare(&temp.path().join("home")).expect("home");
	fs::write(
		home.config_file(),
		"[translate]\nsource = \"en\"\ntarget = \"fr\"\n",
	)
	.expect("global config");
	// The language pair is not authority-bearing: a project may override it.
	fs::write(root.join(".emelex.toml"), "[translate]\ntarget = \"de\"\n").expect("project config");

	let (config, _) = Config::load(&home, &root, true).expect("config loads");
	assert_eq!(config.translate.source.as_deref(), Some("en"));
	assert_eq!(config.translate.target.as_deref(), Some("de"));
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

#[test]
fn global_hub_token_stays_outside_resolved_config_and_debug() {
	let temp = tempfile::tempdir().expect("temporary directory");
	let home = EmelexHome::prepare(&temp.path().join("home")).expect("home");
	let token = "hf_private_config_token";
	Config::write_global_hub_token(&home, Some(token)).expect("store Hub token");

	let loaded =
		Config::load_for_emelex(&home, temp.path(), false).expect("load config and credentials");
	assert!(loaded.hub_credentials.is_some());
	let json = serde_json::to_string(&loaded.config).expect("serialize resolved config");
	assert!(!json.contains(token));
	assert!(!format!("{:?}", loaded.config).contains(token));
	let patch = read_optional_patch(&home.config_file())
		.expect("read global patch")
		.expect("global patch");
	assert!(!format!("{patch:?}").contains(token));
}

#[test]
fn project_config_cannot_set_or_clear_hub_token() {
	let temp = tempfile::tempdir().expect("temporary directory");
	let root = temp.path().join("project");
	fs::create_dir_all(root.join(".git")).expect("Git marker");
	let home = EmelexHome::prepare(&temp.path().join("home")).expect("home");

	for text in [
		"[hub]\ntoken = \"hf_project_secret\"\n",
		"[hub]\ntoken = \"hf_project_secret with-space\"\n",
		"[hub]\ntoken = { clear = true }\n",
	] {
		fs::write(root.join(".emelex.toml"), text).expect("project config");
		let error = Config::load(&home, &root, true).expect_err("project Hub token rejected");
		assert!(error.to_string().contains("hub.token"));
		assert!(!error.to_string().contains("hf_project_secret"));
	}
}

#[test]
fn global_hub_token_writer_sets_replaces_and_clears_narrowly() {
	use std::os::unix::fs::PermissionsExt as _;

	let temp = tempfile::tempdir().expect("temporary directory");
	let home = EmelexHome::prepare(&temp.path().join("home")).expect("home");
	fs::write(
		home.config_file(),
		"[hub]\nresults = 7\n[agent]\nshell = false\n",
	)
	.expect("global config");

	Config::write_global_hub_token(&home, Some("hf_first_secret")).expect("store first token");
	assert!(
		Config::global_hub_token_configured(&home).expect("inspect stored credential presence")
	);
	let first = fs::read_to_string(home.config_file()).expect("read first update");
	assert!(first.contains("hf_first_secret"));
	assert!(first.contains("results = 7"));
	assert!(first.contains("shell = false"));
	assert_eq!(
		fs::metadata(home.config_file())
			.expect("global config metadata")
			.permissions()
			.mode() & 0o777,
		0o600
	);

	Config::write_global_hub_token(&home, Some("hf_second_secret")).expect("replace token");
	let second = fs::read_to_string(home.config_file()).expect("read replacement");
	assert!(!second.contains("hf_first_secret"));
	assert!(second.contains("hf_second_secret"));

	Config::write_global_hub_token(&home, None).expect("clear token");
	assert!(
		!Config::global_hub_token_configured(&home).expect("inspect cleared credential presence")
	);
	let cleared = fs::read_to_string(home.config_file()).expect("read cleared config");
	assert!(!cleared.contains("hf_second_secret"));
	assert!(cleared.contains("results = 7"));
	assert!(cleared.contains("shell = false"));
}

#[test]
fn global_hub_token_writer_rejects_invalid_secret_without_echoing_it() {
	let temp = tempfile::tempdir().expect("temporary directory");
	let home = EmelexHome::prepare(&temp.path().join("home")).expect("home");
	let token = "hf_secret\nforged";
	let error =
		Config::write_global_hub_token(&home, Some(token)).expect_err("invalid token rejected");

	assert!(!error.to_string().contains("hf_secret"));
	assert!(!home.config_file().exists());
}

#[test]
fn invalid_stored_hub_token_error_does_not_echo_secret() {
	let temp = tempfile::tempdir().expect("temporary directory");
	let home = EmelexHome::prepare(&temp.path().join("home")).expect("home");
	let token = "hf_stored_secret with-space";
	fs::write(home.config_file(), format!("[hub]\ntoken = {token:?}\n")).expect("global config");

	let error = Config::load(&home, temp.path(), false).expect_err("invalid stored token rejected");

	assert!(error.to_string().contains(HUB_TOKEN_REQUIREMENT));
	assert!(!error.to_string().contains("hf_stored_secret"));
}

#[test]
fn malformed_stored_hub_token_error_does_not_echo_source_line() {
	let temp = tempfile::tempdir().expect("temporary directory");
	let home = EmelexHome::prepare(&temp.path().join("home")).expect("home");
	fs::write(home.config_file(), "[hub]\ntoken = \"hf_malformed_secret\n").expect("global config");

	let error = Config::load(&home, temp.path(), false).expect_err("malformed token rejected");

	assert!(!error.to_string().contains("hf_malformed_secret"));
}

#[test]
fn default_model_writer_preserves_stored_hub_token() {
	let temp = tempfile::tempdir().expect("temporary directory");
	let home = EmelexHome::prepare(&temp.path().join("home")).expect("home");
	let token = "hf_preserved_secret";
	Config::write_global_hub_token(&home, Some(token)).expect("store token");
	let model = ModelRef::parse("mlx-community/example").expect("model");

	Config::write_global_default_model(&home, Some(&model)).expect("write default model");

	assert!(
		Config::global_hub_token_configured(&home).expect("inspect stored credential presence")
	);
	let text = fs::read_to_string(home.config_file()).expect("read global config");
	assert!(text.contains(token));
	assert!(text.contains("default_model"));
}
