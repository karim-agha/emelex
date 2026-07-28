//! High-level Emelex invocation facade.

use std::{
	path::{Path, PathBuf},
	sync::Arc,
};

use once_cell::sync::OnceCell;

use crate::{
	config::{Config, ConfigError, ConfigSources},
	home::{EmelexHome, HomeError},
	hub::{HubClient, HubCredentials, HubError},
	memory::{MemoryError, MemorySnapshotReferenceGuard, MemoryStore},
	model::{WorkloadError, WorkloadProfile},
	models::ModelManager,
	runtime::{self, RuntimeError},
};

/// One resolved Emelex invocation.
pub struct Emelex {
	home: EmelexHome,
	invocation_root: PathBuf,
	config: Config,
	config_sources: ConfigSources,
	hub: OnceCell<HubClient>,
	models: OnceCell<ModelManager>,
	memory: OnceCell<MemoryStore>,
	metal_budget_bytes: OnceCell<u64>,
	metal_budget_override: Option<u64>,
	hub_credentials: Option<HubCredentials>,
}

impl Emelex {
	/// Start configuring an invocation.
	pub fn builder() -> EmelexBuilder {
		EmelexBuilder::default()
	}

	/// Resolve defaults for the current directory.
	///
	/// # Errors
	///
	/// Returns home, working-directory, or configuration errors.
	pub fn current() -> Result<Self, ToolkitError> {
		Self::builder().build()
	}

	/// Selected Emelex Home.
	pub const fn home(&self) -> &EmelexHome {
		&self.home
	}

	/// Canonical directory from which this invocation started.
	pub fn invocation_root(&self) -> &Path {
		&self.invocation_root
	}

	/// Fully resolved immutable configuration.
	pub const fn config(&self) -> &Config {
		&self.config
	}

	/// Configuration files that contributed to this snapshot.
	pub const fn config_sources(&self) -> &ConfigSources {
		&self.config_sources
	}

	/// Static Hugging Face discovery client.
	///
	/// # Errors
	///
	/// Returns Hub-client initialization failures. This facet makes no
	/// machine-fit claim and does not query Metal.
	pub fn hub(&self) -> Result<&HubClient, ToolkitError> {
		self.hub.get_or_try_init(|| {
			Ok(match &self.hub_credentials {
				Some(credentials) => {
					HubClient::with_credentials(self.config.hub.clone(), credentials.clone())?
				}
				None => HubClient::new(self.config.hub.clone())?,
			})
		})
	}

	/// Immutable installed-model manager.
	///
	/// # Errors
	///
	/// Returns Hub, workload, Metal budget, or model-policy initialization
	/// failures.
	pub fn models(&self) -> Result<&ModelManager, ToolkitError> {
		self.models.get_or_try_init(|| {
			let workload = WorkloadProfile::new(1, self.config.inference.context_tokens)?;
			let metal_budget_bytes = self.metal_budget_bytes()?;
			let hub = match &self.hub_credentials {
				Some(credentials) => HubClient::with_fit_profile_and_credentials(
					self.config.hub.clone(),
					workload,
					metal_budget_bytes,
					credentials.clone(),
				)?,
				None => HubClient::with_fit_profile(
					self.config.hub.clone(),
					workload,
					metal_budget_bytes,
				)?,
			};
			Ok(ModelManager::new(
				self.home.clone(),
				self.config.clone(),
				hub,
				metal_budget_bytes,
			)?
			.with_reference_guard(Arc::new(MemorySnapshotReferenceGuard::new(&self.home))))
		})
	}

	/// Durable Sessions and workspace Knowledge.
	///
	/// # Errors
	///
	/// Returns durable-store initialization or migration failures.
	pub fn memory(&self) -> Result<&MemoryStore, ToolkitError> {
		self.memory
			.get_or_try_init(|| Ok(MemoryStore::open(&self.home)?))
	}

	/// Metal recommended working-set maximum used for fit reports.
	///
	/// # Errors
	///
	/// Returns an error when no supported Metal device is available.
	pub fn metal_budget_bytes(&self) -> Result<u64, ToolkitError> {
		self.metal_budget_bytes
			.get_or_try_init(|| {
				self.metal_budget_override.map_or_else(
					|| runtime::recommended_max_working_set_size().map_err(Into::into),
					Ok,
				)
			})
			.copied()
	}
}

/// Builder for one resolved Emelex invocation.
#[derive(Debug, Clone)]
pub struct EmelexBuilder {
	home: Option<PathBuf>,
	invocation_root: Option<PathBuf>,
	load_project_config: bool,
	metal_budget_bytes: Option<u64>,
	hub_credentials: Option<HubCredentials>,
}

impl Default for EmelexBuilder {
	fn default() -> Self {
		Self {
			home: None,
			invocation_root: None,
			load_project_config: true,
			metal_budget_bytes: None,
			hub_credentials: None,
		}
	}
}

impl EmelexBuilder {
	/// Select the sole storage root.
	#[must_use]
	pub fn home(mut self, path: impl Into<PathBuf>) -> Self {
		self.home = Some(path.into());
		self
	}

	/// Select the tool/configuration invocation directory.
	#[must_use]
	pub fn invocation_root(mut self, path: impl Into<PathBuf>) -> Self {
		self.invocation_root = Some(path.into());
		self
	}

	/// Enable or disable nearest-Git-root `.emelex.toml` loading.
	#[must_use]
	pub const fn project_config(mut self, enabled: bool) -> Self {
		self.load_project_config = enabled;
		self
	}

	/// Override Metal fit budget, primarily for deterministic embedding/tests.
	#[must_use]
	pub const fn metal_budget_bytes(mut self, bytes: u64) -> Self {
		self.metal_budget_bytes = Some(bytes);
		self
	}

	/// Use explicit Hugging Face credentials for this invocation's Hub facets.
	///
	/// No environment variable is read by the library. Separate builders may
	/// therefore carry distinct credentials in the same process.
	#[must_use]
	pub fn hub_credentials(mut self, credentials: HubCredentials) -> Self {
		self.hub_credentials = Some(credentials);
		self
	}

	/// Resolve invocation directory, storage root, and configuration.
	///
	/// Hub, Metal, model management, memory, and MLX remain uninitialized until
	/// their corresponding accessors or model operations are used.
	///
	/// # Errors
	///
	/// Returns home, directory, or configuration errors.
	pub fn build(self) -> Result<Emelex, ToolkitError> {
		let home = EmelexHome::resolve(self.home.as_deref())?;
		let invocation_root = if let Some(path) = self.invocation_root {
			std::fs::canonicalize(&path)
				.map_err(|source| ToolkitError::Directory { path, source })?
		} else {
			let current = std::env::current_dir().map_err(|source| ToolkitError::Directory {
				path: PathBuf::from("."),
				source,
			})?;
			std::fs::canonicalize(&current).map_err(|source| ToolkitError::Directory {
				path: current,
				source,
			})?
		};
		if !invocation_root.is_dir() {
			return Err(ToolkitError::Directory {
				path: invocation_root,
				source: std::io::Error::new(
					std::io::ErrorKind::InvalidInput,
					"invocation root is not a directory",
				),
			});
		}
		let (config, config_sources) =
			Config::load(&home, &invocation_root, self.load_project_config)?;
		if self.metal_budget_bytes == Some(0) {
			return Err(ToolkitError::Configuration(
				"Metal budget override must be positive".to_string(),
			));
		}
		let metal_budget_override = self.metal_budget_bytes;
		Ok(Emelex {
			home,
			invocation_root,
			config,
			config_sources,
			hub: OnceCell::new(),
			models: OnceCell::new(),
			memory: OnceCell::new(),
			metal_budget_bytes: OnceCell::new(),
			metal_budget_override,
			hub_credentials: self.hub_credentials,
		})
	}
}

/// High-level invocation construction failure.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ToolkitError {
	/// Home resolution/preparation failed.
	#[error(transparent)]
	Home(#[from] HomeError),
	/// Invocation directory failed validation.
	#[error("cannot use invocation directory {path:?}: {source}")]
	Directory {
		/// Requested path.
		path: PathBuf,
		/// Underlying error.
		#[source]
		source: std::io::Error,
	},
	/// Strict configuration failed.
	#[error(transparent)]
	Config(#[from] ConfigError),
	/// Hub client construction failed.
	#[error(transparent)]
	Hub(#[from] HubError),
	/// Durable memory initialization failed.
	#[error(transparent)]
	Memory(#[from] MemoryError),
	/// Model-manager policy was invalid.
	#[error(transparent)]
	Models(#[from] crate::models::ModelsError),
	/// Workload assumptions were invalid.
	#[error(transparent)]
	Workload(#[from] WorkloadError),
	/// Metal budget query failed.
	#[error(transparent)]
	Runtime(#[from] RuntimeError),
	/// Builder override was invalid.
	#[error("invalid Emelex builder configuration: {0}")]
	Configuration(String),
}

#[cfg(test)]
mod tests {
	#![allow(clippy::expect_used)]

	use super::*;

	#[test]
	fn facade_construction_and_static_hub_leave_memory_and_metal_lazy() {
		let directory = tempfile::tempdir().expect("temporary invocation root");
		let requested_home = directory.path().join("home");
		let emelex = Emelex::builder()
			.home(&requested_home)
			.invocation_root(directory.path())
			.metal_budget_bytes(123_456)
			.build()
			.expect("build invocation facade");
		let database = emelex.home().database_file();

		assert!(!database.exists());
		let _ = emelex.hub().expect("initialize static Hub client");
		assert!(!database.exists());
		assert_eq!(
			emelex
				.metal_budget_bytes()
				.expect("configured Metal budget"),
			123_456
		);
		assert!(!database.exists());

		let first = emelex.memory().expect("initialize memory");
		let second = emelex.memory().expect("reuse memory");
		assert!(std::ptr::eq(first, second));
		assert!(database.exists());
	}

	#[test]
	fn explicit_credentials_reach_both_lazy_hub_clients() {
		let directory = tempfile::tempdir().expect("temporary invocation root");
		let emelex = Emelex::builder()
			.home(directory.path().join("home"))
			.invocation_root(directory.path())
			.metal_budget_bytes(123_456)
			.hub_credentials(HubCredentials::bearer_token("hf_example").expect("valid credentials"))
			.build()
			.expect("build authenticated invocation facade");

		assert!(
			emelex
				.hub()
				.expect("initialize static Hub client")
				.is_authenticated()
		);
		assert!(
			emelex
				.models()
				.expect("initialize model manager")
				.hub()
				.is_authenticated()
		);
	}

	#[test]
	fn zero_budget_override_fails_before_any_facet_activation() {
		let directory = tempfile::tempdir().expect("temporary invocation root");
		let error = Emelex::builder()
			.home(directory.path().join("home"))
			.invocation_root(directory.path())
			.metal_budget_bytes(0)
			.build()
			.err()
			.expect("zero budget must fail");
		assert!(matches!(error, ToolkitError::Configuration(_)));
	}
}
