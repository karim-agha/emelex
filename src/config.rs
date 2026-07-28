//! Strict global and project configuration loading.

use std::{
	fs::{self, File, OpenOptions},
	io::{Read as _, Write as _},
	os::{
		fd::AsRawFd as _,
		unix::fs::{MetadataExt as _, OpenOptionsExt as _, PermissionsExt as _},
	},
	path::{Path, PathBuf},
	time::Duration,
};

use serde::{Deserialize, Serialize};

use crate::{
	agent::{MAX_SHELL_OUTPUT_BYTES, MAX_WEB_RESPONSE_BYTES},
	home::EmelexHome,
	model::ModelRef,
};

macro_rules! apply_copy {
	($source:ident, $target:ident, $($field:ident),+ $(,)?) => {
		$(
			if let Some(value) = $source.$field {
				$target.$field = value;
			}
		)+
	};
}

/// Fully resolved configuration snapshot.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[non_exhaustive]
pub struct Config {
	/// Default model reference used when no command override exists.
	pub default_model: Option<ModelRef>,
	/// Generation behavior.
	pub inference: InferenceConfig,
	/// Agent and tool behavior.
	pub agent: AgentConfig,
	/// Hub discovery and transfer bounds.
	pub hub: HubConfig,
	/// Durable memory behavior.
	pub memory: MemoryConfig,
}

/// Generation defaults.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[non_exhaustive]
pub struct InferenceConfig {
	/// Maximum generated tokens.
	pub max_tokens: usize,
	/// Workload context assumption used for fit estimation.
	pub context_tokens: usize,
	/// Sampling temperature.
	pub temperature: f32,
	/// Nucleus-sampling threshold.
	pub top_p: f32,
	/// Optional top-k cutoff.
	pub top_k: Option<u32>,
	/// Optional deterministic seed.
	pub seed: Option<u64>,
	/// Thinking-mode policy.
	pub thinking: ThinkingMode,
	/// Enable verified MTP automatically.
	pub mtp: bool,
	/// Draft depth when MTP is enabled.
	pub speculative_tokens: usize,
	/// Reuse prompt KV state.
	pub prompt_cache: bool,
}

impl Default for InferenceConfig {
	fn default() -> Self {
		Self {
			max_tokens: 4096,
			context_tokens: 16_384,
			temperature: 0.0,
			top_p: 1.0,
			top_k: None,
			seed: None,
			thinking: ThinkingMode::Auto,
			mtp: false,
			speculative_tokens: 4,
			prompt_cache: true,
		}
	}
}

/// Thinking-mode selection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ThinkingMode {
	/// Use the client default; when unset, ask the template to disable reasoning.
	#[default]
	Auto,
	/// Request reasoning.
	On,
	/// Ask the template to disable reasoning.
	Off,
}

/// Agent and built-in tool defaults.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[non_exhaustive]
pub struct AgentConfig {
	/// Optional generic system instruction appended after immutable policy.
	pub system_prompt: Option<String>,
	/// Maximum assistant/tool loop turns.
	pub max_turns: usize,
	/// Enable HTTP(S) tools.
	pub web: bool,
	/// Enable workspace file tools.
	pub files: bool,
	/// Enable host shell tool behind approval.
	pub shell: bool,
	/// Per-command shell timeout in seconds.
	pub shell_timeout_seconds: u64,
	/// Maximum captured shell output bytes.
	pub shell_output_bytes: usize,
	/// Maximum fetched web response bytes.
	pub web_response_bytes: usize,
}

impl Default for AgentConfig {
	fn default() -> Self {
		Self {
			system_prompt: None,
			max_turns: 20,
			web: true,
			files: true,
			shell: true,
			shell_timeout_seconds: 120,
			shell_output_bytes: MAX_SHELL_OUTPUT_BYTES,
			web_response_bytes: MAX_WEB_RESPONSE_BYTES,
		}
	}
}

impl AgentConfig {
	/// Shell timeout.
	pub const fn shell_timeout(&self) -> Duration {
		Duration::from_secs(self.shell_timeout_seconds)
	}
}

/// Hugging Face discovery and transfer limits.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[non_exhaustive]
pub struct HubConfig {
	/// Compatible results requested per page.
	pub results: usize,
	/// Maximum Hub candidates inspected per search call.
	pub scan_limit: usize,
	/// Concurrent metadata preflights.
	pub metadata_concurrency: usize,
	/// HTTP request timeout in seconds.
	pub request_timeout_seconds: u64,
	/// Transfer retry count after the first attempt.
	pub retries: usize,
}

impl Default for HubConfig {
	fn default() -> Self {
		Self {
			results: 20,
			scan_limit: 200,
			metadata_concurrency: 8,
			request_timeout_seconds: 30,
			retries: 3,
		}
	}
}

impl HubConfig {
	/// HTTP timeout.
	pub const fn request_timeout(&self) -> Duration {
		Duration::from_secs(self.request_timeout_seconds)
	}

	pub(crate) fn validate(&self) -> Result<(), ConfigError> {
		if self.results == 0
			|| self.results > 100
			|| self.scan_limit < self.results
			|| self.scan_limit > 1000
			|| self.metadata_concurrency == 0
			|| self.metadata_concurrency > 32
			|| !(1..=300).contains(&self.request_timeout_seconds)
			|| self.retries > 10
		{
			return Err(ConfigError::Invalid(
				"Hub limits require results 1..=100, scan_limit between results \
				 and 1000, metadata_concurrency 1..=32, request timeout 1..=300 \
				 seconds, and retries at most 10"
					.to_string(),
			));
		}
		Ok(())
	}
}

/// Session and Knowledge defaults.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[non_exhaustive]
pub struct MemoryConfig {
	/// Inactive-data retention window.
	pub retention_days: u32,
	/// Optional model reference used for compaction/distillation.
	pub model: Option<ModelRef>,
	/// Maximum automatically recalled entries.
	pub recall_entries: usize,
	/// Maximum serialized bytes injected by automatic memory recall.
	pub recall_bytes: usize,
	/// Minimum automatic Knowledge confidence.
	pub confidence_threshold: f32,
}

impl Default for MemoryConfig {
	fn default() -> Self {
		Self {
			retention_days: 30,
			model: None,
			recall_entries: 8,
			recall_bytes: 6 * 1024,
			confidence_threshold: 0.70,
		}
	}
}

/// Configuration files selected for one invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ConfigSources {
	/// Global configuration, when present.
	pub global: Option<PathBuf>,
	/// Git-root project configuration, when present.
	pub project: Option<PathBuf>,
}

/// Strict configuration loading failures.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ConfigError {
	/// Configuration file could not be read.
	#[error("cannot read configuration {path:?}: {source}")]
	Read {
		/// Configuration path.
		path: PathBuf,
		/// Underlying I/O failure.
		#[source]
		source: std::io::Error,
	},
	/// Global configuration could not be replaced durably.
	#[error("cannot write configuration {path:?}: {source}")]
	Write {
		/// Affected configuration or temporary path.
		path: PathBuf,
		/// Underlying I/O failure.
		#[source]
		source: std::io::Error,
	},
	/// Global configuration changed during a read-modify-write update.
	#[error("configuration {path:?} changed concurrently; retry the operation")]
	ConcurrentModification {
		/// Configuration path.
		path: PathBuf,
	},
	/// Configuration syntax or fields are invalid.
	#[error("invalid configuration {path:?}: {message}")]
	Parse {
		/// Configuration path.
		path: PathBuf,
		/// Parser explanation.
		message: String,
	},
	/// A resolved numeric policy is unsafe or nonsensical.
	#[error("invalid resolved configuration: {0}")]
	Invalid(String),
}

impl Config {
	/// Load defaults, global configuration, then optional project
	/// configuration.
	///
	/// Project discovery starts at `invocation_root` and stops at the nearest
	/// ancestor containing `.git` as a file or directory.
	///
	/// # Errors
	///
	/// Unknown keys, malformed TOML, unreadable files, or invalid bounds fail
	/// before model loading.
	pub fn load(
		home: &EmelexHome,
		invocation_root: &Path,
		load_project: bool,
	) -> Result<(Self, ConfigSources), ConfigError> {
		let global_path = home.config_file();
		let global = read_optional_patch(&global_path)?;
		let project_path = load_project
			.then(|| project_root(invocation_root).map(|root| root.join(".emelex.toml")))
			.flatten();
		let project = match project_path.as_ref() {
			Some(path) => {
				let patch = read_optional_patch(path)?;
				if let Some(patch) = &patch {
					patch.validate_project(path)?;
				}
				patch
			}
			None => None,
		};
		let mut config = Self::default();
		if let Some(patch) = global {
			patch.apply(&mut config, PatchAuthority::Global);
		}
		if let Some(patch) = project {
			patch.apply(&mut config, PatchAuthority::Project);
		}
		config.validate()?;
		Ok((
			config,
			ConfigSources {
				global: global_path.is_file().then_some(global_path),
				project: project_path.filter(|path| path.is_file()),
			},
		))
	}

	/// Atomically set or clear the global default model while preserving every
	/// other global setting.
	///
	/// Project configuration is never read or written by this operation.
	///
	/// # Errors
	///
	/// Returns when the existing global file is unsafe or invalid, the updated
	/// configuration would be invalid, or durable replacement fails.
	pub fn write_global_default_model(
		home: &EmelexHome,
		model: Option<&ModelRef>,
	) -> Result<(), ConfigError> {
		let _lock = ConfigWriteLock::acquire(home)?;
		let path = home.config_file();
		let original = read_optional_text(&path)?;
		let text = original.clone().unwrap_or_default();
		let mut table =
			toml::from_str::<toml::Table>(&text).map_err(|error| ConfigError::Parse {
				path: path.clone(),
				message: error.to_string(),
			})?;
		match model {
			Some(model) => {
				table.insert(
					"default_model".to_string(),
					toml::Value::String(model.to_string()),
				);
			}
			None => {
				table.remove("default_model");
			}
		}
		let rendered = toml::to_string_pretty(&table).map_err(|error| ConfigError::Parse {
			path: path.clone(),
			message: error.to_string(),
		})?;
		let patch =
			toml::from_str::<ConfigPatch>(&rendered).map_err(|error| ConfigError::Parse {
				path: path.clone(),
				message: error.to_string(),
			})?;
		let mut resolved = Self::default();
		patch.apply(&mut resolved, PatchAuthority::Global);
		resolved.validate()?;
		write_global_config(home, &path, rendered.as_bytes(), original.as_deref())
	}

	/// Validate every resolved limit and cross-field invariant.
	///
	/// # Errors
	///
	/// Returns [`ConfigError::Invalid`] when any field is outside its
	/// supported range or conflicts with another field.
	pub fn validate(&self) -> Result<(), ConfigError> {
		if !(1..=1 << 20).contains(&self.inference.max_tokens)
			|| !(1..=1 << 20).contains(&self.inference.context_tokens)
			|| self.inference.speculative_tokens > 8
		{
			return Err(ConfigError::Invalid(
				"inference token limits must be in 1..=1048576 and \
				 speculative_tokens at most 8"
					.to_string(),
			));
		}
		if self.inference.max_tokens > self.inference.context_tokens {
			return Err(ConfigError::Invalid(
				"inference.max_tokens must not exceed inference.context_tokens".to_string(),
			));
		}
		if !self.inference.temperature.is_finite()
			|| !self.inference.top_p.is_finite()
			|| !(0.0..=2.0).contains(&self.inference.temperature)
			|| !(0.0..=1.0).contains(&self.inference.top_p)
		{
			return Err(ConfigError::Invalid(
				"temperature must be 0..=2 and top_p must be 0..=1".to_string(),
			));
		}
		if !(1..=crate::agent::MAX_AGENT_MODEL_ROUNDS).contains(&self.agent.max_turns)
			|| !(1..=crate::agent::MAX_SHELL_TIMEOUT_SECONDS)
				.contains(&self.agent.shell_timeout_seconds)
			|| !(1..=MAX_SHELL_OUTPUT_BYTES).contains(&self.agent.shell_output_bytes)
			|| !(1..=MAX_WEB_RESPONSE_BYTES).contains(&self.agent.web_response_bytes)
		{
			return Err(ConfigError::Invalid(
				"agent bounds require max_turns 1..=20, shell timeout 1..=1200 \
				 seconds, shell output 1..=524288 bytes, and web response \
				 1..=1048576 bytes"
					.to_string(),
			));
		}
		self.hub.validate()?;
		if !(1..=3650).contains(&self.memory.retention_days)
			|| !(1..=100).contains(&self.memory.recall_entries)
			|| !(1..=1_048_576).contains(&self.memory.recall_bytes)
			|| !self.memory.confidence_threshold.is_finite()
			|| !(0.0..=1.0).contains(&self.memory.confidence_threshold)
		{
			return Err(ConfigError::Invalid(
				"memory bounds require retention_days 1..=3650, recall_entries \
				 1..=100, recall_bytes 1..=1048576, and confidence_threshold \
				 0..=1"
					.to_string(),
			));
		}
		Ok(())
	}
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ConfigPatch {
	default_model: Option<PatchValue<ModelRef>>,
	inference: Option<InferencePatch>,
	agent: Option<AgentPatch>,
	hub: Option<HubPatch>,
	memory: Option<MemoryPatch>,
}

impl ConfigPatch {
	fn validate_project(&self, path: &Path) -> Result<(), ConfigError> {
		let mut forbidden = Vec::new();
		if self.default_model.is_some() {
			forbidden.push("default_model");
		}
		if let Some(inference) = &self.inference {
			if inference.seed.is_some() {
				forbidden.push("inference.seed");
			}
			if matches!(inference.top_k, Some(PatchValue::Clear(_))) {
				forbidden.push("inference.top_k clear");
			}
			if matches!(inference.top_k, Some(PatchValue::Set(0))) {
				forbidden.push("inference.top_k = 0");
			}
			if inference
				.thinking
				.is_some_and(|mode| mode != ThinkingMode::Off)
			{
				forbidden.push("inference.thinking on/auto");
			}
		}
		if self
			.agent
			.as_ref()
			.is_some_and(|agent| agent.system_prompt.is_some())
		{
			forbidden.push("agent.system_prompt");
		}
		if let Some(memory) = &self.memory {
			if memory.model.is_some() {
				forbidden.push("memory.model");
			}
			if memory.retention_days.is_some() {
				forbidden.push("memory.retention_days");
			}
		}
		if forbidden.is_empty() {
			Ok(())
		} else {
			Err(ConfigError::Parse {
				path: path.to_path_buf(),
				message: format!(
					"project configuration cannot set authority-bearing field(s): {}",
					forbidden.join(", ")
				),
			})
		}
	}

	fn apply(self, config: &mut Config, authority: PatchAuthority) {
		if authority == PatchAuthority::Global
			&& let Some(value) = self.default_model
		{
			config.default_model = value.into_option();
		}
		if let Some(patch) = self.inference {
			patch.apply(&mut config.inference, authority);
		}
		if let Some(patch) = self.agent {
			patch.apply(&mut config.agent, authority);
		}
		if let Some(patch) = self.hub {
			patch.apply(&mut config.hub, authority);
		}
		if let Some(patch) = self.memory {
			patch.apply(&mut config.memory, authority);
		}
	}
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct InferencePatch {
	max_tokens: Option<usize>,
	context_tokens: Option<usize>,
	temperature: Option<f32>,
	top_p: Option<f32>,
	top_k: Option<PatchValue<u32>>,
	seed: Option<PatchValue<u64>>,
	thinking: Option<ThinkingMode>,
	mtp: Option<bool>,
	speculative_tokens: Option<usize>,
	prompt_cache: Option<bool>,
}

impl InferencePatch {
	fn apply(self, target: &mut InferenceConfig, authority: PatchAuthority) {
		if authority == PatchAuthority::Project {
			if let Some(value) = self.max_tokens {
				target.max_tokens = target.max_tokens.min(value);
			}
			if let Some(value) = self.context_tokens {
				target.context_tokens = target.context_tokens.min(value);
			}
			if let Some(value) = self.temperature {
				target.temperature = target.temperature.min(value);
			}
			if let Some(value) = self.top_p {
				target.top_p = target.top_p.min(value);
			}
			if let Some(PatchValue::Set(value)) = self.top_k {
				target.top_k = Some(target.top_k.map_or(value, |current| current.min(value)));
			}
			if self.thinking == Some(ThinkingMode::Off) {
				target.thinking = ThinkingMode::Off;
			}
			if let Some(value) = self.mtp {
				target.mtp &= value;
			}
			if let Some(value) = self.speculative_tokens {
				target.speculative_tokens = target.speculative_tokens.min(value);
			}
			if let Some(value) = self.prompt_cache {
				target.prompt_cache &= value;
			}
			return;
		}
		apply_copy!(self, target, max_tokens, context_tokens, temperature, top_p);
		if let Some(value) = self.top_k {
			target.top_k = value.into_option();
		}
		if let Some(value) = self.seed {
			target.seed = value.into_option();
		}
		apply_copy!(self, target, thinking, mtp);
		apply_copy!(self, target, speculative_tokens, prompt_cache);
	}
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct AgentPatch {
	system_prompt: Option<PatchValue<String>>,
	max_turns: Option<usize>,
	web: Option<bool>,
	files: Option<bool>,
	shell: Option<bool>,
	shell_timeout_seconds: Option<u64>,
	shell_output_bytes: Option<usize>,
	web_response_bytes: Option<usize>,
}

impl AgentPatch {
	fn apply(self, target: &mut AgentConfig, authority: PatchAuthority) {
		if authority == PatchAuthority::Global
			&& let Some(value) = self.system_prompt
		{
			target.system_prompt = value.into_option();
		}
		if authority == PatchAuthority::Project {
			if let Some(value) = self.web {
				target.web &= value;
			}
			if let Some(value) = self.files {
				target.files &= value;
			}
			if let Some(value) = self.shell {
				target.shell &= value;
			}
			if let Some(value) = self.max_turns {
				target.max_turns = target.max_turns.min(value);
			}
			if let Some(value) = self.shell_timeout_seconds {
				target.shell_timeout_seconds = target.shell_timeout_seconds.min(value);
			}
			if let Some(value) = self.shell_output_bytes {
				target.shell_output_bytes = target.shell_output_bytes.min(value);
			}
			if let Some(value) = self.web_response_bytes {
				target.web_response_bytes = target.web_response_bytes.min(value);
			}
			return;
		}
		apply_copy!(self, target, max_turns, web, files, shell);
		apply_copy!(
			self,
			target,
			shell_timeout_seconds,
			shell_output_bytes,
			web_response_bytes
		);
	}
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct HubPatch {
	results: Option<usize>,
	scan_limit: Option<usize>,
	metadata_concurrency: Option<usize>,
	request_timeout_seconds: Option<u64>,
	retries: Option<usize>,
}

impl HubPatch {
	fn apply(self, target: &mut HubConfig, authority: PatchAuthority) {
		if authority == PatchAuthority::Project {
			if let Some(value) = self.results {
				target.results = target.results.min(value);
			}
			if let Some(value) = self.scan_limit {
				target.scan_limit = target.scan_limit.min(value);
			}
			if let Some(value) = self.metadata_concurrency {
				target.metadata_concurrency = target.metadata_concurrency.min(value);
			}
			if let Some(value) = self.request_timeout_seconds {
				target.request_timeout_seconds = target.request_timeout_seconds.min(value);
			}
			if let Some(value) = self.retries {
				target.retries = target.retries.min(value);
			}
			return;
		}
		apply_copy!(
			self,
			target,
			results,
			scan_limit,
			metadata_concurrency,
			request_timeout_seconds,
			retries
		);
	}
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct MemoryPatch {
	retention_days: Option<u32>,
	model: Option<PatchValue<ModelRef>>,
	recall_entries: Option<usize>,
	recall_bytes: Option<usize>,
	confidence_threshold: Option<f32>,
}

impl MemoryPatch {
	fn apply(self, target: &mut MemoryConfig, authority: PatchAuthority) {
		if authority == PatchAuthority::Project {
			if let Some(value) = self.recall_entries {
				target.recall_entries = target.recall_entries.min(value);
			}
			if let Some(value) = self.recall_bytes {
				target.recall_bytes = target.recall_bytes.min(value);
			}
			if let Some(value) = self.confidence_threshold {
				target.confidence_threshold = target.confidence_threshold.max(value);
			}
			return;
		}
		apply_copy!(self, target, retention_days, recall_entries, recall_bytes);
		apply_copy!(self, target, confidence_threshold);
		if let Some(value) = self.model {
			target.model = value.into_option();
		}
	}
}

fn read_optional_patch(path: &Path) -> Result<Option<ConfigPatch>, ConfigError> {
	let Some(text) = read_optional_text(path)? else {
		return Ok(None);
	};
	toml::from_str::<ConfigPatch>(&text)
		.map(Some)
		.map_err(|error| ConfigError::Parse {
			path: path.to_path_buf(),
			message: error.to_string(),
		})
}

fn read_optional_text(path: &Path) -> Result<Option<String>, ConfigError> {
	const MAX_CONFIG_BYTES: u64 = 1 << 20;
	let mut file = match OpenOptions::new()
		.read(true)
		.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK)
		.open(path)
	{
		Ok(file) => file,
		Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
		Err(source) => {
			return Err(ConfigError::Read {
				path: path.to_path_buf(),
				source,
			});
		}
	};
	let metadata = file.metadata().map_err(|source| ConfigError::Read {
		path: path.to_path_buf(),
		source,
	})?;
	if !metadata.is_file() || metadata.len() > MAX_CONFIG_BYTES {
		return Err(ConfigError::Parse {
			path: path.to_path_buf(),
			message: "configuration must be a regular file no larger than 1 MiB".to_string(),
		});
	}
	let capacity = usize::try_from(metadata.len()).map_err(|_| ConfigError::Parse {
		path: path.to_path_buf(),
		message: "configuration size does not fit memory".to_string(),
	})?;
	let mut bytes = Vec::with_capacity(capacity);
	std::io::Read::by_ref(&mut file)
		.take(MAX_CONFIG_BYTES + 1)
		.read_to_end(&mut bytes)
		.map_err(|source| ConfigError::Read {
			path: path.to_path_buf(),
			source,
		})?;
	if bytes.len() as u64 > MAX_CONFIG_BYTES {
		return Err(ConfigError::Parse {
			path: path.to_path_buf(),
			message: "configuration must be no larger than 1 MiB".to_string(),
		});
	}
	let text = String::from_utf8(bytes).map_err(|error| ConfigError::Parse {
		path: path.to_path_buf(),
		message: format!("configuration is not UTF-8: {error}"),
	})?;
	Ok(Some(text))
}

fn write_global_config(
	home: &EmelexHome,
	path: &Path,
	contents: &[u8],
	expected: Option<&str>,
) -> Result<(), ConfigError> {
	let temporary = home.root().join(format!(
		".config.toml.{}.{}.tmp",
		std::process::id(),
		uuid::Uuid::now_v7()
	));
	let result = write_global_config_inner(home, path, &temporary, contents, expected);
	if result.is_err() {
		let _ = fs::remove_file(&temporary);
	}
	result
}

fn write_global_config_inner(
	home: &EmelexHome,
	path: &Path,
	temporary: &Path,
	contents: &[u8],
	expected: Option<&str>,
) -> Result<(), ConfigError> {
	let mut file = OpenOptions::new()
		.create_new(true)
		.write(true)
		.mode(0o600)
		.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
		.open(temporary)
		.map_err(|source| ConfigError::Write {
			path: temporary.to_path_buf(),
			source,
		})?;
	file.write_all(contents)
		.and_then(|()| file.sync_all())
		.map_err(|source| ConfigError::Write {
			path: temporary.to_path_buf(),
			source,
		})?;
	if read_optional_text(path)?.as_deref() != expected {
		return Err(ConfigError::ConcurrentModification {
			path: path.to_path_buf(),
		});
	}
	fs::rename(temporary, path).map_err(|source| ConfigError::Write {
		path: path.to_path_buf(),
		source,
	})?;
	File::open(home.root())
		.and_then(|directory| directory.sync_all())
		.map_err(|source| ConfigError::Write {
			path: home.root().to_path_buf(),
			source,
		})
}

struct ConfigWriteLock(File);

impl ConfigWriteLock {
	fn acquire(home: &EmelexHome) -> Result<Self, ConfigError> {
		let path = home.root().join(".config.lock");
		let file = OpenOptions::new()
			.read(true)
			.write(true)
			.create(true)
			.mode(0o600)
			.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
			.open(&path)
			.map_err(|source| ConfigError::Write {
				path: path.clone(),
				source,
			})?;
		let metadata = file.metadata().map_err(|source| ConfigError::Write {
			path: path.clone(),
			source,
		})?;
		if !metadata.is_file()
			|| metadata.uid() != crate::home::effective_user_id()
			|| metadata.permissions().mode() & 0o777 != 0o600
		{
			return Err(ConfigError::Write {
				path,
				source: std::io::Error::new(
					std::io::ErrorKind::PermissionDenied,
					"configuration lock must be an owner-only regular file",
				),
			});
		}
		loop {
			// SAFETY: file owns a live descriptor and LOCK_EX is a valid operation.
			if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } == 0 {
				return Ok(Self(file));
			}
			let source = std::io::Error::last_os_error();
			if source.kind() != std::io::ErrorKind::Interrupted {
				return Err(ConfigError::Write { path, source });
			}
		}
	}
}

impl Drop for ConfigWriteLock {
	fn drop(&mut self) {
		// SAFETY: the descriptor remains live until this guard finishes dropping.
		unsafe {
			libc::flock(self.0.as_raw_fd(), libc::LOCK_UN);
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PatchAuthority {
	Global,
	Project,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum PatchValue<T> {
	Set(T),
	Clear(ClearValue),
}

impl<T> PatchValue<T> {
	fn into_option(self) -> Option<T> {
		match self {
			Self::Set(value) => Some(value),
			Self::Clear(_) => None,
		}
	}
}

#[derive(Debug, Clone)]
struct ClearValue;

impl<'de> Deserialize<'de> for ClearValue {
	fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
	where
		D: serde::Deserializer<'de>,
	{
		#[derive(Deserialize)]
		#[serde(deny_unknown_fields)]
		struct Wire {
			clear: bool,
		}

		let wire = Wire::deserialize(deserializer)?;
		if wire.clear {
			Ok(Self)
		} else {
			Err(serde::de::Error::custom("clear marker must be true"))
		}
	}
}

fn project_root(start: &Path) -> Option<PathBuf> {
	let start = fs::canonicalize(start).ok()?;
	start
		.ancestors()
		.find(|candidate| candidate.join(".git").exists())
		.map(Path::to_path_buf)
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
