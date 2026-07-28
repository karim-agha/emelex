//! Immutable installed-model lifecycle.

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{
	collections::BTreeSet,
	fs::{self, OpenOptions},
	io::{Read as _, Write as _},
	os::unix::fs::{MetadataExt as _, OpenOptionsExt as _, PermissionsExt as _},
	path::{Path, PathBuf},
	sync::{Arc, OnceLock, mpsc},
	time::{Duration, SystemTime},
};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::Digest as _;
use tokio::io::AsyncReadExt as _;
use walkdir::WalkDir;

use crate::{
	Client, Error,
	config::Config,
	home::{EmelexHome, create_owner_subdir},
	hub::{
		DownloadCancellation, DownloadObserver, DownloadOperationGuard, DownloadReporter,
		HubClient, HubError, required_download_storage_bytes,
	},
	model::{
		CompatibilityReport, HubModelId, InspectionError, InstalledModel, LocalModelName,
		ManifestError, ModelFile, ModelManifest, ModelRef, ModelSnapshotId, ModelSource,
		VerificationStatus, WorkloadProfile, inspect_directory,
	},
};

const MANIFEST_NAME: &str = "emelex-model.json";
const VERIFIED_STAMP_NAME: &str = ".emelex-verified.json";
const QUARANTINE_RECORD_NAME: &str = "emelex-quarantine.json";
const MAX_MANIFEST_BYTES: u64 = 4 << 20;
const MAX_VERIFICATION_STAMP_BYTES: u64 = 8 << 20;

/// Tri-state per-load setting.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[non_exhaustive]
pub enum LoadOverride<T> {
	/// Keep resolved Emelex configuration.
	#[default]
	Inherit,
	/// Set an explicit value.
	Set(T),
	/// Clear the configured value and use the engine's unset/default state.
	Clear,
}

/// Per-load overrides over resolved Emelex inference configuration.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct ModelLoadOptions {
	/// Bounded waiting jobs.
	pub queue_capacity: Option<usize>,
	/// Generation-token default.
	pub max_tokens: Option<usize>,
	/// Total prompt plus generation context.
	pub context_tokens: Option<usize>,
	/// Sampling temperature.
	pub temperature: LoadOverride<f32>,
	/// Nucleus sampling.
	pub top_p: LoadOverride<f32>,
	/// Top-k cutoff.
	pub top_k: LoadOverride<u32>,
	/// Fixed seed.
	pub seed: LoadOverride<u64>,
	/// Prompt cache.
	pub prompt_cache: Option<bool>,
	/// MTP draft depth.
	pub speculative_tokens: Option<usize>,
	/// Per-load thinking mode.
	pub thinking: Option<crate::config::ThinkingMode>,
	/// Explicit set/clear reasoning budget.
	pub reasoning_budget_tokens: LoadOverride<usize>,
}

impl ModelLoadOptions {
	/// Set bounded waiting jobs.
	#[must_use]
	pub const fn queue_capacity(mut self, capacity: usize) -> Self {
		self.queue_capacity = Some(capacity);
		self
	}

	/// Set the generation-token ceiling.
	#[must_use]
	pub const fn max_tokens(mut self, tokens: usize) -> Self {
		self.max_tokens = Some(tokens);
		self
	}

	/// Set total prompt plus generation context.
	#[must_use]
	pub const fn context_tokens(mut self, tokens: usize) -> Self {
		self.context_tokens = Some(tokens);
		self
	}

	/// Set, clear, or inherit sampling temperature.
	#[must_use]
	pub const fn temperature(mut self, setting: LoadOverride<f32>) -> Self {
		self.temperature = setting;
		self
	}

	/// Set, clear, or inherit nucleus sampling.
	#[must_use]
	pub const fn top_p(mut self, setting: LoadOverride<f32>) -> Self {
		self.top_p = setting;
		self
	}

	/// Set, clear, or inherit the top-k cutoff.
	#[must_use]
	pub const fn top_k(mut self, setting: LoadOverride<u32>) -> Self {
		self.top_k = setting;
		self
	}

	/// Set, clear, or inherit the deterministic seed.
	#[must_use]
	pub const fn seed(mut self, setting: LoadOverride<u64>) -> Self {
		self.seed = setting;
		self
	}

	/// Set prompt-cache behavior.
	#[must_use]
	pub const fn prompt_cache(mut self, enabled: bool) -> Self {
		self.prompt_cache = Some(enabled);
		self
	}

	/// Set MTP draft depth. Zero explicitly disables speculation.
	#[must_use]
	pub const fn speculative_tokens(mut self, tokens: usize) -> Self {
		self.speculative_tokens = Some(tokens);
		self
	}

	/// Set per-load thinking behavior.
	#[must_use]
	pub const fn thinking(mut self, thinking: crate::config::ThinkingMode) -> Self {
		self.thinking = Some(thinking);
		self
	}

	/// Set, clear, or inherit the reasoning-span budget.
	#[must_use]
	pub const fn reasoning_budget_tokens(mut self, setting: LoadOverride<usize>) -> Self {
		self.reasoning_budget_tokens = setting;
		self
	}
}

/// Fully resolved policy applied to one managed model load.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ModelLoadPolicy {
	/// Bounded waiting jobs.
	pub queue_capacity: usize,
	/// Generation-token ceiling.
	pub max_tokens: usize,
	/// Prompt plus generated context.
	pub context_tokens: usize,
	/// Effective sampling temperature.
	pub temperature: f32,
	/// Effective nucleus threshold.
	pub top_p: f32,
	/// Effective top-k cutoff.
	pub top_k: Option<u32>,
	/// Effective deterministic seed.
	pub seed: Option<u64>,
	/// Thinking behavior.
	pub thinking: crate::config::ThinkingMode,
	/// Optional reasoning-span budget.
	pub reasoning_budget_tokens: Option<usize>,
	/// Prompt-cache behavior.
	pub prompt_cache: bool,
	/// MTP draft depth.
	pub speculative_tokens: usize,
}

/// One corrupt or unsafe installed-model entry skipped by inventory.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ModelDiagnostic {
	/// Entry path.
	pub path: PathBuf,
	/// Bounded failure text.
	pub message: String,
}

/// Healthy snapshots plus candidate-local diagnostics.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ModelInventory {
	/// Manifest- and hash-valid snapshots.
	pub models: Vec<InstalledModel>,
	/// Invalid entries skipped without hiding healthy snapshots.
	pub diagnostics: Vec<ModelDiagnostic>,
}

/// Failure returned by a snapshot-reference policy.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
#[non_exhaustive]
pub struct SnapshotReferenceError {
	message: String,
}

impl SnapshotReferenceError {
	/// Construct a bounded policy failure.
	pub fn new(message: impl Into<String>) -> Self {
		let message = bounded_diagnostic(&message.into());
		let message = if message.is_empty() {
			"snapshot reference policy failed without a diagnostic".to_string()
		} else {
			message
		};
		Self { message }
	}

	/// Bounded diagnostic text.
	pub fn message(&self) -> &str {
		&self.message
	}
}

/// Session/reference policy consulted before snapshot removal.
pub trait SnapshotReferenceGuard: Send + Sync {
	/// Whether durable state still references an exact snapshot.
	///
	/// # Errors
	///
	/// Returns a bounded implementation-specific failure.
	fn is_referenced(&self, snapshot: &ModelSnapshotId) -> Result<bool, SnapshotReferenceError>;
}

/// Outcome of a full model verification.
#[derive(Debug)]
#[non_exhaustive]
pub struct ModelVerification {
	/// Recomputed static report, promoted after load.
	pub compatibility: CompatibilityReport,
	/// Freshly loaded client proving the snapshot remains runnable.
	pub client: Client,
}

/// Emelex-owned immutable model store.
#[derive(Clone)]
pub struct ModelManager {
	home: EmelexHome,
	config: Config,
	hub: HubClient,
	metal_budget_bytes: u64,
	reference_guard: Arc<dyn SnapshotReferenceGuard>,
}

impl ModelManager {
	/// Construct a manager from one resolved Emelex invocation.
	///
	/// # Errors
	///
	/// Returns an error for invalid configuration or a zero Metal budget.
	pub fn new(
		home: EmelexHome,
		config: Config,
		hub: HubClient,
		metal_budget_bytes: u64,
	) -> Result<Self, ModelsError> {
		config
			.validate()
			.map_err(|error| ModelsError::Configuration(error.to_string()))?;
		if metal_budget_bytes == 0 {
			return Err(ModelsError::Configuration(
				"Metal budget must be positive".to_string(),
			));
		}
		let workload = WorkloadProfile::new(1, config.inference.context_tokens)
			.map_err(|error| ModelsError::Configuration(error.to_string()))?;
		if hub.fit_workload() != Some(workload)
			|| hub.metal_budget_bytes() != Some(metal_budget_bytes)
		{
			return Err(ModelsError::Configuration(
				"Hub fit profile must match ModelManager context and Metal budget".to_string(),
			));
		}
		let reference_guard = Arc::new(crate::memory::MemorySnapshotReferenceGuard::new(&home));
		Ok(Self {
			home,
			config,
			hub,
			metal_budget_bytes,
			reference_guard,
		})
	}

	/// Replace the default durable-session snapshot protection.
	#[must_use]
	pub fn with_reference_guard(mut self, guard: Arc<dyn SnapshotReferenceGuard>) -> Self {
		self.reference_guard = guard;
		self
	}

	/// Hub discovery client using the same bounds.
	pub const fn hub(&self) -> &HubClient {
		&self.hub
	}

	/// List every manifest-valid installed snapshot.
	///
	/// # Errors
	///
	/// Returns only when the model-store root itself cannot be traversed.
	/// Invalid individual snapshots are omitted; use [`Self::inventory`] when
	/// their diagnostics are needed.
	pub fn list(&self) -> Result<Vec<InstalledModel>, ModelsError> {
		Ok(self.inventory()?.models)
	}

	/// List healthy snapshots while retaining diagnostics for corrupt entries.
	///
	/// # Errors
	///
	/// Returns only when the model-store root itself cannot be traversed.
	pub fn inventory(&self) -> Result<ModelInventory, ModelsError> {
		let root = self.home.models_dir();
		let metadata = fs::symlink_metadata(&root).map_err(|source| ModelsError::Io {
			path: root.clone(),
			source,
		})?;
		if !metadata.is_dir() || metadata.file_type().is_symlink() {
			return Err(ModelsError::UnsafeInstall(root));
		}
		fs::read_dir(&root).map_err(|source| ModelsError::Io {
			path: root.clone(),
			source,
		})?;
		let mut models = Vec::new();
		let mut diagnostics = Vec::new();
		for entry in WalkDir::new(&root).follow_links(false).max_depth(6) {
			let entry = match entry {
				Ok(entry) => entry,
				Err(error) => {
					diagnostics.push(ModelDiagnostic {
						path: error.path().map_or_else(|| root.clone(), Path::to_path_buf),
						message: bounded_diagnostic(&error.to_string()),
					});
					continue;
				}
			};
			if entry.file_type().is_symlink() {
				diagnostics.push(ModelDiagnostic {
					path: entry.path().to_path_buf(),
					message: "symlinks are forbidden in the managed model store".to_string(),
				});
				continue;
			}
			if entry.file_type().is_file() && entry.file_name() == MANIFEST_NAME {
				let directory = entry
					.path()
					.parent()
					.ok_or_else(|| ModelsError::UnsafeInstall(entry.path().to_path_buf()))?;
				match self.load_installed_at(directory) {
					Ok(installed) => models.push(installed),
					Err(error) => diagnostics.push(ModelDiagnostic {
						path: directory.to_path_buf(),
						message: bounded_diagnostic(&error.to_string()),
					}),
				}
			}
		}
		models.sort_by(|left, right| {
			left.reference()
				.to_string()
				.cmp(&right.reference().to_string())
				.then_with(|| {
					left.manifest()
						.installed_at()
						.cmp(&right.manifest().installed_at())
				})
		});
		Ok(ModelInventory {
			models,
			diagnostics,
		})
	}

	/// Resolve the newest installed snapshot for a stable reference.
	///
	/// # Errors
	///
	/// Returns [`ModelsError::NotInstalled`] when no snapshot exists.
	pub fn resolve(&self, reference: &ModelRef) -> Result<InstalledModel, ModelsError> {
		self.list()?
			.into_iter()
			.filter(|installed| installed.reference() == reference)
			.max_by_key(|installed| installed.manifest().installed_at())
			.ok_or_else(|| ModelsError::NotInstalled(reference.clone()))
	}

	/// Resolve one exact immutable snapshot.
	///
	/// # Errors
	///
	/// Returns [`ModelsError::SnapshotNotInstalled`] when no healthy exact
	/// snapshot exists.
	pub fn resolve_snapshot(
		&self,
		snapshot: &ModelSnapshotId,
	) -> Result<InstalledModel, ModelsError> {
		self.list()?
			.into_iter()
			.find(|installed| installed.snapshot_id() == snapshot)
			.ok_or_else(|| ModelsError::SnapshotNotInstalled(snapshot.clone()))
	}

	/// Download, verify, runtime-load, and atomically publish one Hub model.
	///
	/// # Errors
	///
	/// Candidate-local planning, compatibility, manifest, model-load, and
	/// bounded-probe failures return [`ModelsError::Certification`]. Hub
	/// transport, storage, policy, task, panic, and global runtime failures
	/// retain their specific variants.
	pub async fn download(
		&self,
		id: &HubModelId,
		reporter: Option<&DownloadReporter>,
	) -> Result<InstalledModel, ModelsError> {
		let mut operation = DownloadOperationGuard::new(None);
		let result = self
			.download_inner(id, reporter, None, Some(operation.cancellation()))
			.await;
		operation.finish();
		result
	}

	/// Download with a fallible observer and cooperative cancellation.
	///
	/// # Errors
	///
	/// Candidate-local planning, compatibility, manifest, model-load, and
	/// bounded-probe failures return [`ModelsError::Certification`]. Hub
	/// transport, observer, cancellation, storage, policy, task, panic, and
	/// global runtime failures retain their specific variants.
	pub async fn download_controlled(
		&self,
		id: &HubModelId,
		observer: Option<&DownloadObserver>,
		cancellation: Option<&DownloadCancellation>,
	) -> Result<InstalledModel, ModelsError> {
		let mut operation = DownloadOperationGuard::new(cancellation);
		let result = self
			.download_inner(id, None, observer, Some(operation.cancellation()))
			.await;
		operation.finish();
		result
	}

	#[allow(
		clippy::too_many_lines,
		reason = "one install transaction keeps plan, staging, verification, and publication state explicit"
	)]
	async fn download_inner(
		&self,
		id: &HubModelId,
		reporter: Option<&DownloadReporter>,
		observer: Option<&DownloadObserver>,
		cancellation: Option<&DownloadCancellation>,
	) -> Result<InstalledModel, ModelsError> {
		check_download_cancellation(cancellation)?;
		let plan = self
			.hub
			.plan(id)
			.await
			.map_err(mark_hub_candidate_certification_error)?;
		check_download_cancellation(cancellation)?;
		let fit = plan.model().fit.as_ref().ok_or_else(|| {
			mark_candidate_certification_error(ModelsError::Incompatible(vec![
				"Hub plan lacks an exact residency estimate for this invocation".to_string(),
			]))
		})?;
		let workload = self.workload()?;
		if fit.budget_bytes != self.metal_budget_bytes || fit.workload != workload || !fit.fits {
			return Err(mark_candidate_certification_error(
				ModelsError::Incompatible(vec![format!(
					"Hub plan does not fit the active workload and Metal budget: required={}, budget={}",
					fit.required_bytes, self.metal_budget_bytes
				)]),
			));
		}
		let temp_dir = self.home.temp_dir();
		let total_bytes = plan.total_bytes();
		tokio::task::spawn_blocking(move || preflight_disk(&temp_dir, total_bytes))
			.await
			.map_err(blocking_model_task_error)??;
		check_download_cancellation(cancellation)?;
		let reference = ModelRef::Hub(id.clone());
		let destination = self.hub_destination(id, plan.model().revision.as_str());
		let expected_snapshot = ModelSnapshotId::Hub {
			id: id.clone(),
			revision: plan.model().revision.clone(),
		};
		let mutation_lock = self.snapshot_mutation_lock_controlled(cancellation).await?;
		let manager = self.clone();
		let existing_destination = destination.clone();
		let existing_reference = reference.clone();
		let existing_revision = plan.model().revision.clone();
		let existing_snapshot = expected_snapshot.clone();
		let existing = tokio::task::spawn_blocking(move || {
			let _mutation_lock = mutation_lock;
			manager.reuse_existing_locked(
				&existing_destination,
				&existing_reference,
				Some(&existing_revision),
				&existing_snapshot,
			)
		})
		.await
		.map_err(blocking_model_task_error)??;
		if let Some(existing) = existing {
			check_download_cancellation(cancellation)?;
			return Ok(existing);
		}
		let manager = self.clone();
		let staging = tokio::task::spawn_blocking(move || manager.create_staging("hub"))
			.await
			.map_err(blocking_model_task_error)??;
		check_download_cancellation(cancellation)?;
		let files = self
			.hub
			.download_with_controls(&plan, staging.path(), reporter, observer, cancellation)
			.await?;
		check_download_cancellation(cancellation)?;
		verify_files_controlled(staging.path(), &files, cancellation).await?;
		check_download_cancellation(cancellation)?;
		let manager = self.clone();
		let staging_path = staging.path().to_path_buf();
		let probe_files = files.clone();
		let revision = plan.model().revision.clone();
		let license = plan.model().license.clone();
		let certification = tokio::task::spawn_blocking(move || {
			let mut report = inspect_directory(
				reference.clone(),
				&staging_path,
				workload,
				manager.metal_budget_bytes,
			)?;
			if !report.compatible {
				return Err(ModelsError::Incompatible(report.reasons));
			}
			let policy = manager.resolve_load_policy(
				&report.traits,
				&ModelLoadOptions {
					speculative_tokens: Some(0),
					thinking: Some(crate::config::ThinkingMode::Off),
					reasoning_budget_tokens: LoadOverride::Clear,
					..ModelLoadOptions::default()
				},
			)?;
			let client = manager.build_client(&staging_path, &policy, &probe_files, None)?;
			client.runtime_probe()?;
			report.mark_runtime_loaded(
				client.supports_mtp(),
				client.supports_images(),
				client.supports_audio(),
			);
			drop(client);
			let manifest = ModelManifest::new(
				reference,
				ModelSource::Hub,
				Some(revision),
				files,
				report.traits,
				VerificationStatus::Verified,
				license,
			)
			.map_err(ModelsError::from)?;
			Ok::<_, ModelsError>((staging, manifest))
		})
		.await
		.map_err(blocking_model_task_error)?;
		let (staging, manifest) = certification.map_err(mark_candidate_certification_error)?;
		check_download_cancellation(cancellation)?;
		let mutation_lock = self.snapshot_mutation_lock_controlled(cancellation).await?;
		let manager = self.clone();
		let owned_cancellation = cancellation.cloned();
		tokio::task::spawn_blocking(move || {
			write_manifest(staging.path(), &manifest)?;
			manager.publish_with_lock(
				staging,
				&destination,
				&manifest,
				owned_cancellation.as_ref(),
				mutation_lock,
			)
		})
		.await
		.map_err(blocking_model_task_error)?
	}

	/// Copy, verify, runtime-load, and atomically publish a local checkpoint.
	///
	/// # Errors
	///
	/// Returns source, storage, compatibility, manifest, or load errors.
	pub fn import(
		&self,
		name: &LocalModelName,
		source: &Path,
	) -> Result<InstalledModel, ModelsError> {
		let source = fs::canonicalize(source).map_err(|error| ModelsError::Io {
			path: source.to_path_buf(),
			source: error,
		})?;
		let reference = ModelRef::Local(name.clone());
		let source_report = inspect_directory(
			reference.clone(),
			&source,
			self.workload()?,
			self.metal_budget_bytes,
		)?;
		if !source_report.compatible {
			return Err(ModelsError::Incompatible(source_report.reasons));
		}
		let copy_plan = local_runtime_plan(&source)?;
		let transfer_bytes = copy_plan.iter().try_fold(0_u64, |total, path| {
			let metadata = fs::symlink_metadata(source.join(path)).map_err(|source_error| {
				ModelsError::Io {
					path: source.join(path),
					source: source_error,
				}
			})?;
			total.checked_add(metadata.len()).ok_or_else(|| {
				ModelsError::Configuration("local import byte count overflow".to_string())
			})
		})?;
		preflight_disk(&self.home.temp_dir(), transfer_bytes)?;
		let staging = self.create_staging("local")?;
		let files = copy_runtime_files(&source, staging.path(), &copy_plan)?;
		verify_files(staging.path(), &files)?;
		let digest = snapshot_digest(&files);
		let destination = self.local_destination(name, &digest);
		let expected_snapshot = ModelSnapshotId::Local {
			name: name.clone(),
			digest: crate::model::SnapshotDigest::parse(digest)
				.map_err(|error| ModelsError::Configuration(error.to_string()))?,
		};
		{
			let _mutation_lock = self.snapshot_mutation_lock()?;
			if let Some(existing) =
				self.reuse_existing_locked(&destination, &reference, None, &expected_snapshot)?
			{
				return Ok(existing);
			}
		}
		let mut report = inspect_directory(
			reference.clone(),
			staging.path(),
			self.workload()?,
			self.metal_budget_bytes,
		)?;
		if !report.compatible {
			return Err(ModelsError::Incompatible(report.reasons));
		}
		let policy = self.resolve_load_policy(
			&report.traits,
			&ModelLoadOptions {
				speculative_tokens: Some(0),
				thinking: Some(crate::config::ThinkingMode::Off),
				reasoning_budget_tokens: LoadOverride::Clear,
				..ModelLoadOptions::default()
			},
		)?;
		let client = self.build_client(staging.path(), &policy, &files, None)?;
		client.runtime_probe()?;
		report.mark_runtime_loaded(
			client.supports_mtp(),
			client.supports_images(),
			client.supports_audio(),
		);
		drop(client);
		let manifest = ModelManifest::new(
			reference,
			ModelSource::LocalImport {
				original_path: source,
			},
			None,
			files,
			report.traits,
			VerificationStatus::Verified,
			None,
		)?;
		write_manifest(staging.path(), &manifest)?;
		self.publish(staging, &destination, &manifest, None)
	}

	/// Hash, inspect, and runtime-load one installed snapshot.
	///
	/// # Errors
	///
	/// Returns when any immutable file changed or the model no longer loads.
	pub fn verify(&self, installed: &InstalledModel) -> Result<ModelVerification, ModelsError> {
		self.validate_owned_install(installed)?;
		verify_files(installed.path(), installed.manifest().files())?;
		let mut compatibility = inspect_directory(
			installed.reference().clone(),
			installed.path(),
			self.workload()?,
			self.metal_budget_bytes,
		)?;
		if !compatibility.compatible {
			return Err(ModelsError::Incompatible(compatibility.reasons));
		}
		let policy = self.resolve_load_policy(
			installed.manifest().traits(),
			&ModelLoadOptions {
				speculative_tokens: Some(0),
				thinking: Some(crate::config::ThinkingMode::Off),
				reasoning_budget_tokens: LoadOverride::Clear,
				..ModelLoadOptions::default()
			},
		)?;
		let client = self.build_client(
			installed.path(),
			&policy,
			installed.manifest().files(),
			Some(installed.snapshot_id()),
		)?;
		client.runtime_probe()?;
		compatibility.mark_runtime_loaded(
			client.supports_mtp(),
			client.supports_images(),
			client.supports_audio(),
		);
		Ok(ModelVerification {
			compatibility,
			client,
		})
	}

	/// Verify immutable files and load a snapshot with inference overrides.
	///
	/// # Errors
	///
	/// Returns on ownership, hash, configuration, or engine-load failure.
	pub fn load(
		&self,
		installed: &InstalledModel,
		options: &ModelLoadOptions,
	) -> Result<Client, ModelsError> {
		self.validate_owned_install(installed)?;
		verify_files(installed.path(), installed.manifest().files())?;
		let policy = self.resolve_load_policy(installed.manifest().traits(), options)?;
		let compatibility = inspect_directory(
			installed.reference().clone(),
			installed.path(),
			WorkloadProfile::new(1, policy.context_tokens)
				.map_err(|error| ModelsError::Configuration(error.to_string()))?,
			self.metal_budget_bytes,
		)?;
		if !compatibility.compatible {
			return Err(ModelsError::Incompatible(compatibility.reasons));
		}
		let client = self.build_client(
			installed.path(),
			&policy,
			installed.manifest().files(),
			Some(installed.snapshot_id()),
		)?;
		if policy.speculative_tokens > 0 && !client.supports_mtp() {
			return Err(ModelsError::Configuration(
				"installed snapshot no longer exposes its runtime-verified MTP module".to_string(),
			));
		}
		Ok(client)
	}

	/// Resolve the complete policy that [`Self::load`] will apply.
	///
	/// # Errors
	///
	/// Returns when an override is invalid for the model or active machine.
	pub fn load_policy(
		&self,
		installed: &InstalledModel,
		options: &ModelLoadOptions,
	) -> Result<ModelLoadPolicy, ModelsError> {
		self.validate_owned_install(installed)?;
		self.resolve_load_policy(installed.manifest().traits(), options)
	}

	/// Move one snapshot into Emelex quarantine.
	///
	/// The operation is recoverable until model garbage collection.
	///
	/// # Errors
	///
	/// Returns when the install is outside this home or cannot be moved.
	pub fn remove(&self, installed: &InstalledModel) -> Result<PathBuf, ModelsError> {
		let _mutation_lock = self.snapshot_mutation_lock()?;
		self.validate_owned_install(installed)?;
		self.ensure_unreferenced(installed.snapshot_id())?;
		self.move_to_quarantine(
			installed.path(),
			"removed",
			Some(installed.snapshot_id().clone()),
		)
	}

	/// Permanently delete quarantined model data older than `age`.
	///
	/// # Errors
	///
	/// Returns on unsafe entries or deletion I/O failure.
	pub fn gc_quarantine(&self, age: Duration) -> Result<usize, ModelsError> {
		let root = self.home.cache_dir().join("quarantine/models");
		if !root.is_dir() {
			return Ok(0);
		}
		let now = SystemTime::now();
		let mut removed = 0_usize;
		for entry in fs::read_dir(&root).map_err(|source| ModelsError::Io {
			path: root.clone(),
			source,
		})? {
			let entry = entry.map_err(|source| ModelsError::Io {
				path: root.clone(),
				source,
			})?;
			if !entry
				.file_type()
				.map_err(|source| ModelsError::Io {
					path: entry.path(),
					source,
				})?
				.is_dir()
			{
				return Err(ModelsError::UnsafeInstall(entry.path()));
			}
			let record = read_quarantine_record(&entry.path())?;
			let quarantined_at = SystemTime::from(record.quarantined_at);
			if now.duration_since(quarantined_at).unwrap_or_default() >= age {
				let _mutation_lock = self.snapshot_mutation_lock()?;
				if let Some(snapshot) = &record.snapshot_id {
					self.ensure_unreferenced(snapshot)?;
				}
				make_writable(&entry.path())?;
				fs::remove_dir_all(entry.path()).map_err(|source| ModelsError::Io {
					path: entry.path(),
					source,
				})?;
				sync_directory(&root)?;
				removed += 1;
			}
		}
		Ok(removed)
	}

	fn snapshot_mutation_lock(&self) -> Result<crate::home::SnapshotMutationLock, ModelsError> {
		self.home.lock_snapshot_mutations().map_err(|error| {
			ModelsError::SnapshotMutationLock(bounded_diagnostic(&error.to_string()))
		})
	}

	async fn snapshot_mutation_lock_controlled(
		&self,
		cancellation: Option<&DownloadCancellation>,
	) -> Result<crate::home::SnapshotMutationLock, ModelsError> {
		loop {
			check_download_cancellation(cancellation)?;
			let home = self.home.clone();
			let lock = tokio::task::spawn_blocking(move || home.try_lock_snapshot_mutations())
				.await
				.map_err(|error| {
					ModelsError::SnapshotMutationLock(format!("snapshot lock task failed: {error}"))
				})?
				.map_err(|error| {
					ModelsError::SnapshotMutationLock(bounded_diagnostic(&error.to_string()))
				})?;
			if let Some(lock) = lock {
				return Ok(lock);
			}
			tokio::time::sleep(Duration::from_millis(25)).await;
		}
	}

	fn ensure_unreferenced(&self, snapshot: &ModelSnapshotId) -> Result<(), ModelsError> {
		match self.reference_guard.is_referenced(snapshot) {
			Ok(false) => Ok(()),
			Ok(true) => Err(ModelsError::SnapshotReferenced(snapshot.clone())),
			Err(error) => Err(ModelsError::ReferenceGuard(error)),
		}
	}

	fn move_to_quarantine(
		&self,
		source: &Path,
		reason: &str,
		snapshot_id: Option<ModelSnapshotId>,
	) -> Result<PathBuf, ModelsError> {
		let parent = create_owner_subdir(self.home.root(), &["cache", "quarantine", "models"])
			.map_err(|source_error| ModelsError::Io {
				path: self.home.cache_dir().join("quarantine/models"),
				source: source_error,
			})?;
		move_to_quarantine(&parent, source, reason, snapshot_id)
	}

	fn workload(&self) -> Result<WorkloadProfile, ModelsError> {
		WorkloadProfile::new(1, self.config.inference.context_tokens)
			.map_err(|error| ModelsError::Configuration(error.to_string()))
	}

	fn resolve_load_policy(
		&self,
		traits: &crate::model::ModelTraits,
		options: &ModelLoadOptions,
	) -> Result<ModelLoadPolicy, ModelsError> {
		let inference = &self.config.inference;
		let temperature = resolve_override(options.temperature, inference.temperature, 0.0);
		let top_p = resolve_override(options.top_p, inference.top_p, 1.0);
		let top_k = resolve_optional_override(options.top_k, inference.top_k);
		let seed = resolve_optional_override(options.seed, inference.seed);
		let requested_context = options.context_tokens.unwrap_or(inference.context_tokens);
		let context_tokens = traits
			.sizing
			.as_ref()
			.and_then(|sizing| sizing.max_context_tokens)
			.map_or(requested_context, |limit| requested_context.min(limit));
		let max_tokens = options
			.max_tokens
			.unwrap_or(inference.max_tokens)
			.min(context_tokens);
		let thinking = options.thinking.unwrap_or(inference.thinking);
		let reasoning_budget_tokens = match options.reasoning_budget_tokens {
			LoadOverride::Inherit | LoadOverride::Clear => None,
			LoadOverride::Set(value) => Some(value),
		};
		let speculative_tokens = options.speculative_tokens.unwrap_or({
			if inference.mtp {
				inference.speculative_tokens
			} else {
				0
			}
		});
		let policy = ModelLoadPolicy {
			queue_capacity: options.queue_capacity.unwrap_or(8),
			max_tokens,
			context_tokens,
			temperature,
			top_p,
			top_k,
			seed,
			thinking,
			reasoning_budget_tokens,
			prompt_cache: options.prompt_cache.unwrap_or(inference.prompt_cache),
			speculative_tokens,
		};
		validate_load_policy(&policy, traits)?;
		Ok(policy)
	}

	fn build_client(
		&self,
		path: &Path,
		policy: &ModelLoadPolicy,
		expected_files: &[ModelFile],
		model_snapshot_id: Option<&ModelSnapshotId>,
	) -> Result<Client, ModelsError> {
		let mut builder = Client::builder(path)
			.home(self.home.root())
			.expected_files(expected_files)
			.model_snapshot_id(model_snapshot_id.cloned())
			.queue_capacity(policy.queue_capacity)
			.max_tokens(policy.max_tokens)
			.context_tokens(policy.context_tokens)
			.temperature(policy.temperature)
			.top_p(policy.top_p)
			.top_k(policy.top_k.unwrap_or(0))
			.prompt_cache(policy.prompt_cache)
			.speculative_tokens(policy.speculative_tokens);
		if let Some(seed) = policy.seed {
			builder = builder.seed(seed);
		}
		builder =
			builder.enable_thinking(matches!(policy.thinking, crate::config::ThinkingMode::On));
		if let Some(budget) = policy.reasoning_budget_tokens {
			builder = builder.reasoning_budget_tokens(budget);
		}
		builder.build().map_err(ModelsError::Client)
	}

	fn validate_owned_install(&self, installed: &InstalledModel) -> Result<(), ModelsError> {
		revalidate_installed_snapshot(&self.home, installed)
	}

	fn load_installed_at(&self, path: &Path) -> Result<InstalledModel, ModelsError> {
		let canonical = contained_directory(&self.home.models_dir(), path)?;
		let manifest = read_manifest(&canonical)?;
		if manifest.verification() != VerificationStatus::Verified {
			return Err(ModelsError::UnverifiedSnapshot(
				manifest.snapshot_id().clone(),
			));
		}
		let expected = self.snapshot_destination(manifest.snapshot_id());
		if canonical != expected {
			return Err(ModelsError::UnsafeInstall(canonical));
		}
		verify_installed_files(&canonical, &manifest)?;
		Ok(InstalledModel::new(canonical, manifest))
	}

	fn reuse_existing_locked(
		&self,
		destination: &Path,
		reference: &ModelRef,
		revision: Option<&crate::model::ResolvedRevision>,
		expected_snapshot: &ModelSnapshotId,
	) -> Result<Option<InstalledModel>, ModelsError> {
		match fs::symlink_metadata(destination) {
			Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
			Err(source) => {
				return Err(ModelsError::Io {
					path: destination.to_path_buf(),
					source,
				});
			}
			Ok(_) => {}
		}
		let existing = self.load_installed_at(destination);
		if let Ok(installed) = &existing
			&& installed.reference() == reference
			&& installed.manifest().resolved_revision() == revision
		{
			return Ok(Some(installed.clone()));
		}
		let occupant_snapshot = existing
			.as_ref()
			.ok()
			.map(|installed| installed.snapshot_id().clone())
			.or_else(|| {
				read_manifest(destination)
					.ok()
					.map(|manifest| manifest.snapshot_id().clone())
			});
		if let Some(snapshot) = &occupant_snapshot {
			self.ensure_unreferenced(snapshot)?;
		}
		if occupant_snapshot.as_ref() != Some(expected_snapshot) {
			self.ensure_unreferenced(expected_snapshot)?;
		}
		self.move_to_quarantine(
			destination,
			"invalid-existing",
			occupant_snapshot.or_else(|| Some(expected_snapshot.clone())),
		)?;
		Ok(None)
	}

	fn create_staging(&self, kind: &str) -> Result<StagingGuard, ModelsError> {
		let id = uuid::Uuid::now_v7().to_string();
		let path = create_owner_subdir(self.home.root(), &["temp", "models", kind, &id]).map_err(
			|source| ModelsError::Io {
				path: self.home.temp_dir().join("models").join(kind).join(&id),
				source,
			},
		)?;
		let quarantine_parent =
			create_owner_subdir(self.home.root(), &["cache", "quarantine", "models"]).map_err(
				|source| ModelsError::Io {
					path: self.home.cache_dir().join("quarantine/models"),
					source,
				},
			)?;
		Ok(StagingGuard {
			path,
			quarantine_parent,
			active: true,
			#[cfg(test)]
			cleanup_delay: None,
		})
	}

	fn publish(
		&self,
		staging: StagingGuard,
		destination: &Path,
		expected: &ModelManifest,
		cancellation: Option<&DownloadCancellation>,
	) -> Result<InstalledModel, ModelsError> {
		self.publish_inner_with_lock(staging, destination, expected, cancellation, None, || {})
	}

	fn publish_with_lock(
		&self,
		staging: StagingGuard,
		destination: &Path,
		expected: &ModelManifest,
		cancellation: Option<&DownloadCancellation>,
		lock: crate::home::SnapshotMutationLock,
	) -> Result<InstalledModel, ModelsError> {
		self.publish_inner_with_lock(
			staging,
			destination,
			expected,
			cancellation,
			Some(lock),
			|| {},
		)
	}

	#[cfg(test)]
	fn publish_inner<F>(
		&self,
		staging: StagingGuard,
		destination: &Path,
		expected: &ModelManifest,
		cancellation: Option<&DownloadCancellation>,
		before_rename: F,
	) -> Result<InstalledModel, ModelsError>
	where
		F: FnOnce(),
	{
		self.publish_inner_with_lock(
			staging,
			destination,
			expected,
			cancellation,
			None,
			before_rename,
		)
	}

	fn publish_inner_with_lock<F>(
		&self,
		mut staging: StagingGuard,
		destination: &Path,
		expected: &ModelManifest,
		cancellation: Option<&DownloadCancellation>,
		mutation_lock: Option<crate::home::SnapshotMutationLock>,
		before_rename: F,
	) -> Result<InstalledModel, ModelsError>
	where
		F: FnOnce(),
	{
		check_download_cancellation(cancellation)?;
		let parent = destination
			.parent()
			.ok_or_else(|| ModelsError::UnsafeInstall(destination.to_path_buf()))?;
		prepare_owned_directory(self.home.root(), parent)?;
		check_download_cancellation(cancellation)?;
		sync_directory(staging.path())?;
		make_read_only_contents(staging.path())?;
		write_verification_stamp(staging.path(), expected)?;
		set_mode(&staging.path().join(VERIFIED_STAMP_NAME), 0o400)?;
		set_mode(staging.path(), 0o500)?;
		before_rename();
		check_download_cancellation(cancellation)?;
		let _mutation_lock = match mutation_lock {
			Some(lock) => lock,
			None => self.snapshot_mutation_lock()?,
		};
		if fs::symlink_metadata(destination).is_ok()
			&& let Some(existing) = self.reuse_existing_locked(
				destination,
				expected.reference(),
				expected.resolved_revision(),
				expected.snapshot_id(),
			)? {
			check_download_cancellation(cancellation)?;
			return Ok(existing);
		}
		check_download_cancellation(cancellation)?;
		fs::rename(staging.path(), destination).map_err(|source| ModelsError::Io {
			path: destination.to_path_buf(),
			source,
		})?;
		sync_directory(parent)?;
		staging.moved_to(destination);
		let installed = self.load_installed_at(destination)?;
		if installed.manifest() != expected {
			return Err(ModelsError::ManifestEncoding(
				"published manifest differs from verified staging manifest".to_string(),
			));
		}
		staging.commit();
		Ok(installed)
	}

	fn hub_destination(&self, id: &HubModelId, revision: &str) -> PathBuf {
		let root = self.home.models_dir().join("hub");
		id.namespace().map_or_else(
			|| {
				root.join("unnamespaced")
					.join(id.repo_name())
					.join(revision)
			},
			|namespace| {
				root.join("namespaced")
					.join(namespace)
					.join(id.repo_name())
					.join(revision)
			},
		)
	}

	fn local_destination(&self, name: &LocalModelName, digest: &str) -> PathBuf {
		self.home
			.models_dir()
			.join("local")
			.join(name.as_str())
			.join(digest)
	}

	fn snapshot_destination(&self, snapshot: &ModelSnapshotId) -> PathBuf {
		match snapshot {
			ModelSnapshotId::Hub { id, revision } => self.hub_destination(id, revision.as_str()),
			ModelSnapshotId::Local { name, digest } => {
				self.local_destination(name, digest.as_str())
			}
		}
	}
}

/// Revalidate an installed snapshot while the caller holds the Emelex Home
/// snapshot-mutation lock.
///
/// This is crate-visible for durable session binding. It performs no locking
/// itself, so the caller must retain [`crate::home::SnapshotMutationLock`]
/// through its own reference commit.
pub(crate) fn revalidate_installed_snapshot(
	home: &EmelexHome,
	installed: &InstalledModel,
) -> Result<(), ModelsError> {
	let root = fs::canonicalize(home.models_dir()).map_err(|source| ModelsError::Io {
		path: home.models_dir(),
		source,
	})?;
	let path = fs::canonicalize(installed.path()).map_err(|source| ModelsError::Io {
		path: installed.path().to_path_buf(),
		source,
	})?;
	if !path.starts_with(&root) || path == root {
		return Err(ModelsError::UnsafeInstall(path));
	}
	let stored = read_manifest(&path)?;
	if stored.verification() != VerificationStatus::Verified {
		return Err(ModelsError::UnverifiedSnapshot(
			stored.snapshot_id().clone(),
		));
	}
	if &stored != installed.manifest() {
		return Err(ModelsError::ManifestEncoding(
			"installed manifest changed since resolution".to_string(),
		));
	}
	verify_installed_files(&path, &stored)
}

#[cfg(test)]
pub(crate) fn install_test_snapshot(home: &EmelexHome) -> Result<InstalledModel, ModelsError> {
	use crate::model::{EvidenceSource, ModelSizing, TraitConfidence, TraitEvidence};

	let id = HubModelId::parse("emelex-test/model")
		.map_err(|error| ModelsError::Configuration(error.to_string()))?;
	let revision = crate::model::ResolvedRevision::parse("a".repeat(40))
		.map_err(|error| ModelsError::Configuration(error.to_string()))?;
	let destination = home
		.models_dir()
		.join("hub")
		.join("namespaced")
		.join("emelex-test")
		.join("model")
		.join(revision.as_str());
	let parent = destination
		.parent()
		.ok_or_else(|| ModelsError::UnsafeInstall(destination.clone()))?;
	prepare_owned_directory(home.root(), parent)?;
	fs::create_dir(&destination).map_err(|source| ModelsError::Io {
		path: destination.clone(),
		source,
	})?;
	set_mode(&destination, 0o700)?;
	let fixtures = [
		("config.json", br"{}".as_slice()),
		("model.safetensors", b"test-weights".as_slice()),
		("tokenizer.json", br"{}".as_slice()),
	];
	let mut files = Vec::with_capacity(fixtures.len());
	for (name, bytes) in fixtures {
		let path = destination.join(name);
		fs::write(&path, bytes).map_err(|source| ModelsError::Io {
			path: path.clone(),
			source,
		})?;
		set_mode(&path, 0o600)?;
		files.push(ModelFile::new(
			name,
			u64::try_from(bytes.len()).map_err(|_| {
				ModelsError::Configuration("test fixture byte count overflow".to_string())
			})?,
			hex::encode(sha2::Sha256::digest(bytes)),
		)?);
	}
	let weights_bytes = files
		.iter()
		.filter(|file| file.path().ends_with(".safetensors"))
		.map(ModelFile::size)
		.sum();
	let mut traits = crate::model::ModelTraits {
		input: std::collections::BTreeSet::from([crate::model::Modality::Text]),
		output: std::collections::BTreeSet::from([crate::model::Modality::Text]),
		tasks: std::collections::BTreeSet::from([crate::model::Task::TextGeneration]),
		mlx: true,
		sizing: Some(ModelSizing {
			weights_bytes: Some(weights_bytes),
			estimated_residency_bytes: Some(weights_bytes.saturating_add(1)),
			evaluated_context_tokens: Some(16),
			max_context_tokens: Some(32),
		}),
		..crate::model::ModelTraits::default()
	};
	traits.evidence.push(TraitEvidence {
		trait_key: "compatibility:runtime_load".to_string(),
		source: EvidenceSource::Runtime,
		detail: "test-only runtime evidence".to_string(),
	});
	for key in ["acceleration:mlx", "task:text_generation"] {
		traits
			.confidence
			.insert(key.to_string(), TraitConfidence::RuntimeVerified);
	}
	let manifest = ModelManifest::new(
		ModelRef::Hub(id),
		ModelSource::Hub,
		Some(revision),
		files,
		traits,
		VerificationStatus::Verified,
		None,
	)?;
	write_manifest(&destination, &manifest)?;
	make_read_only_contents(&destination)?;
	write_verification_stamp(&destination, &manifest)?;
	set_mode(&destination.join(VERIFIED_STAMP_NAME), 0o400)?;
	set_mode(&destination, 0o500)?;
	sync_directory(parent)?;
	Ok(InstalledModel::new(destination, manifest))
}

const fn resolve_override<T: Copy>(setting: LoadOverride<T>, inherited: T, cleared: T) -> T {
	match setting {
		LoadOverride::Inherit => inherited,
		LoadOverride::Set(value) => value,
		LoadOverride::Clear => cleared,
	}
}

const fn resolve_optional_override<T: Copy>(
	setting: LoadOverride<T>,
	inherited: Option<T>,
) -> Option<T> {
	match setting {
		LoadOverride::Inherit => inherited,
		LoadOverride::Set(value) => Some(value),
		LoadOverride::Clear => None,
	}
}

fn validate_load_policy(
	policy: &ModelLoadPolicy,
	traits: &crate::model::ModelTraits,
) -> Result<(), ModelsError> {
	if !(1..=64).contains(&policy.queue_capacity) {
		return Err(ModelsError::Configuration(
			"effective queue_capacity must be in 1..=64".to_string(),
		));
	}
	if policy.max_tokens == 0 || policy.max_tokens > 1 << 20 {
		return Err(ModelsError::Configuration(
			"effective max_tokens must be in 1..=1048576".to_string(),
		));
	}
	if policy.context_tokens == 0 || policy.context_tokens > 1 << 20 {
		return Err(ModelsError::Configuration(
			"effective context_tokens must be in 1..=1048576".to_string(),
		));
	}
	if policy.max_tokens > policy.context_tokens {
		return Err(ModelsError::Configuration(
			"effective max_tokens cannot exceed context_tokens".to_string(),
		));
	}
	if !policy.temperature.is_finite() || !(0.0..=2.0).contains(&policy.temperature) {
		return Err(ModelsError::Configuration(
			"effective temperature must be finite and in 0..=2".to_string(),
		));
	}
	if !policy.top_p.is_finite() || !(0.0..=1.0).contains(&policy.top_p) {
		return Err(ModelsError::Configuration(
			"effective top_p must be finite and in 0..=1".to_string(),
		));
	}
	if policy
		.top_k
		.is_some_and(|value| value == 0 || value > i32::MAX.unsigned_abs())
	{
		return Err(ModelsError::Configuration(format!(
			"effective top_k must be in 1..={}",
			i32::MAX
		)));
	}
	if policy.speculative_tokens > 8 {
		return Err(ModelsError::Configuration(
			"effective speculative_tokens must be at most 8".to_string(),
		));
	}
	if policy
		.reasoning_budget_tokens
		.is_some_and(|value| value == 0 || value > policy.max_tokens)
	{
		return Err(ModelsError::Configuration(
			"effective reasoning budget must be in 1..=max_tokens".to_string(),
		));
	}
	if policy.reasoning_budget_tokens.is_some()
		&& policy.thinking != crate::config::ThinkingMode::On
	{
		return Err(ModelsError::Configuration(
			"reasoning budget requires thinking to be on".to_string(),
		));
	}
	if policy.speculative_tokens > 0 && traits.mtp != crate::model::MtpSupport::RuntimeVerified {
		return Err(ModelsError::Configuration(
			"nonzero speculative_tokens requires runtime-verified MTP".to_string(),
		));
	}
	Ok(())
}

fn copy_runtime_files(
	source: &Path,
	destination: &Path,
	plan: &[String],
) -> Result<Vec<ModelFile>, ModelsError> {
	let mut files = Vec::with_capacity(plan.len());
	let mut destinations = BTreeSet::new();
	for name in plan {
		if !runtime_source_file_name(name) {
			return Err(ModelsError::UnsafeInstall(source.join(name)));
		}
		let local_name = crate::hub::local_runtime_path(name);
		if !destinations.insert(local_name) {
			return Err(ModelsError::Configuration(format!(
				"runtime paths collide after normalization: {name:?}"
			)));
		}
		let source_path = source.join(name);
		let target = destination.join(local_name);
		let mut input = open_regular(&source_path)?;
		let mut output = OpenOptions::new()
			.write(true)
			.create_new(true)
			.mode(0o600)
			.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
			.open(&target)
			.map_err(|source_error| ModelsError::Io {
				path: target.clone(),
				source: source_error,
			})?;
		let mut hash = sha2::Sha256::new();
		let mut size = 0_u64;
		let mut buffer = vec![0_u8; 1024 * 1024];
		loop {
			let read = input
				.read(&mut buffer)
				.map_err(|source_error| ModelsError::Io {
					path: source_path.clone(),
					source: source_error,
				})?;
			if read == 0 {
				break;
			}
			output
				.write_all(&buffer[..read])
				.map_err(|source_error| ModelsError::Io {
					path: target.clone(),
					source: source_error,
				})?;
			hash.update(&buffer[..read]);
			size = size
				.checked_add(u64::try_from(read).map_err(|_| {
					ModelsError::Configuration("copied byte count overflow".to_string())
				})?)
				.ok_or_else(|| {
					ModelsError::Configuration("copied byte count overflow".to_string())
				})?;
		}
		output.sync_all().map_err(|source_error| ModelsError::Io {
			path: target,
			source: source_error,
		})?;
		files.push(ModelFile::new(
			local_name.to_string(),
			size,
			hex::encode(hash.finalize()),
		)?);
	}
	files.sort_by(|left, right| left.path().cmp(right.path()));
	sync_directory(destination)?;
	Ok(files)
}

fn local_runtime_plan(source: &Path) -> Result<Vec<String>, ModelsError> {
	let checkpoint = crate::model::layout::checkpoint_plan(source).map_err(|error| {
		ModelsError::Inspection(InspectionError::Layout {
			path: error.path().to_path_buf(),
			message: error.message().to_string(),
		})
	})?;
	if checkpoint.files().is_empty() {
		return Err(ModelsError::Incompatible(vec![
			"local checkpoint has no unambiguous runnable weights".to_string(),
		]));
	}
	let mut selected = checkpoint
		.files()
		.iter()
		.map(|path| {
			path.file_name()
				.and_then(|name| name.to_str())
				.map(str::to_string)
				.ok_or_else(|| ModelsError::UnsafeInstall(path.clone()))
		})
		.collect::<Result<BTreeSet<_>, _>>()?;
	for entry in fs::read_dir(source).map_err(|source_error| ModelsError::Io {
		path: source.to_path_buf(),
		source: source_error,
	})? {
		let entry = entry.map_err(|source_error| ModelsError::Io {
			path: source.to_path_buf(),
			source: source_error,
		})?;
		let name = entry
			.file_name()
			.into_string()
			.map_err(|_| ModelsError::UnsafeInstall(entry.path()))?;
		if crate::hub::runtime_metadata_file_name(&name) || name == "model.safetensors.index.json" {
			let file_type = entry.file_type().map_err(|source_error| ModelsError::Io {
				path: entry.path(),
				source: source_error,
			})?;
			if !file_type.is_file() || file_type.is_symlink() {
				return Err(ModelsError::UnsafeInstall(entry.path()));
			}
			selected.insert(name);
		}
	}
	let (named_defaults, named_tools) = local_named_chat_templates(source)?;
	if selected.contains(crate::engine::tokenizer::LEGACY_CHAT_TEMPLATE_FILE) {
		if !named_defaults.is_empty() || !named_tools.is_empty() {
			return Err(ModelsError::Configuration(
				"chat_template.json conflicts with named chat template files".to_string(),
			));
		}
		selected.remove("chat_template.jinja");
		selected.remove("chat_template_tool_use.jinja");
	} else {
		if selected.contains("chat_template.jinja") && !named_defaults.is_empty() {
			return Err(ModelsError::Configuration(
				"root and named default chat templates map to the same runtime file".to_string(),
			));
		}
		if selected.contains("chat_template_tool_use.jinja") && !named_tools.is_empty() {
			return Err(ModelsError::Configuration(
				"root and named tool-use chat templates map to the same runtime file".to_string(),
			));
		}
		match named_defaults.as_slice() {
			[] => {}
			[default] => {
				selected.insert(default.clone());
			}
			_ => {
				return Err(ModelsError::Configuration(
					"multiple named default chat templates are present".to_string(),
				));
			}
		}
		match named_tools.as_slice() {
			[] => {}
			[tool_use] => {
				selected.insert(tool_use.clone());
			}
			_ => {
				return Err(ModelsError::Configuration(
					"multiple named tool-use chat templates are present".to_string(),
				));
			}
		}
	}
	if !selected.contains("config.json") || !selected.contains("tokenizer.json") {
		return Err(ModelsError::Incompatible(vec![
			"local checkpoint lacks config.json or tokenizer.json".to_string(),
		]));
	}
	Ok(selected.into_iter().collect())
}

fn local_named_chat_templates(source: &Path) -> Result<(Vec<String>, Vec<String>), ModelsError> {
	let mut defaults = Vec::new();
	let mut tools = Vec::new();
	for directory in [
		crate::engine::tokenizer::CURRENT_CHAT_TEMPLATE_DIR,
		crate::engine::tokenizer::LEGACY_CHAT_TEMPLATE_DIR,
	] {
		let named_templates = source.join(directory);
		match fs::symlink_metadata(&named_templates) {
			Ok(metadata) if metadata.file_type().is_dir() && !metadata.file_type().is_symlink() => {
				local_named_chat_template(
					&named_templates,
					directory,
					"default.jinja",
					&mut defaults,
				)?;
				local_named_chat_template(
					&named_templates,
					directory,
					"tool_use.jinja",
					&mut tools,
				)?;
			}
			Ok(_) => return Err(ModelsError::UnsafeInstall(named_templates)),
			Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
			Err(source_error) => {
				return Err(ModelsError::Io {
					path: named_templates,
					source: source_error,
				});
			}
		}
	}
	Ok((defaults, tools))
}

fn local_named_chat_template(
	directory_path: &Path,
	directory_name: &str,
	name: &str,
	selected: &mut Vec<String>,
) -> Result<(), ModelsError> {
	let path = directory_path.join(name);
	match fs::symlink_metadata(&path) {
		Ok(metadata) if metadata.file_type().is_file() && !metadata.file_type().is_symlink() => {
			selected.push(format!("{directory_name}/{name}"));
			Ok(())
		}
		Ok(_) => Err(ModelsError::UnsafeInstall(path)),
		Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
		Err(source) => Err(ModelsError::Io { path, source }),
	}
}

fn runtime_file_name(name: &str) -> bool {
	crate::model::layout::safe_relative_path(name)
		&& Path::new(name).components().count() == 1
		&& name.len() <= 255
		&& (name.ends_with(".safetensors")
			|| name == "model.safetensors.index.json"
			|| crate::hub::runtime_metadata_file_name(name))
}

fn runtime_source_file_name(name: &str) -> bool {
	runtime_file_name(name)
		|| matches!(
			name,
			"additional_chat_templates/default.jinja"
				| "additional_chat_templates/tool_use.jinja"
				| "chat_templates/default.jinja"
				| "chat_templates/tool_use.jinja"
		)
}

fn snapshot_digest(files: &[ModelFile]) -> String {
	let mut files = files.iter().collect::<Vec<_>>();
	files.sort_by(|left, right| left.path().cmp(right.path()));
	let mut hash = sha2::Sha256::new();
	for file in files {
		hash.update(file.path().as_bytes());
		hash.update([0]);
		hash.update(file.size().to_le_bytes());
		hash.update([0]);
		hash.update(file.sha256().as_bytes());
		hash.update([0]);
	}
	hex::encode(hash.finalize())
}

fn write_manifest(path: &Path, manifest: &ModelManifest) -> Result<(), ModelsError> {
	let bytes = serde_json::to_vec_pretty(manifest)
		.map_err(|error| ModelsError::ManifestEncoding(error.to_string()))?;
	if bytes.len() > MAX_MANIFEST_BYTES as usize {
		return Err(ModelsError::ManifestEncoding(
			"manifest exceeds 4 MiB".to_string(),
		));
	}
	let target = path.join(MANIFEST_NAME);
	let mut file = OpenOptions::new()
		.write(true)
		.create_new(true)
		.mode(0o600)
		.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
		.open(&target)
		.map_err(|source| ModelsError::Io {
			path: target.clone(),
			source,
		})?;
	file.write_all(&bytes).map_err(|source| ModelsError::Io {
		path: target.clone(),
		source,
	})?;
	file.sync_all().map_err(|source| ModelsError::Io {
		path: target.clone(),
		source,
	})?;
	sync_directory(path)
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct VerificationStamp {
	schema_version: u32,
	manifest_sha256: String,
	files: Vec<StampedFile>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StampedFile {
	path: String,
	size: u64,
	sha256: String,
	device: u64,
	inode: u64,
	mtime_seconds: i64,
	mtime_nanoseconds: i64,
	ctime_seconds: i64,
	ctime_nanoseconds: i64,
}

fn write_verification_stamp(root: &Path, manifest: &ModelManifest) -> Result<(), ModelsError> {
	let manifest_bytes = read_bounded_regular(&root.join(MANIFEST_NAME), MAX_MANIFEST_BYTES)?;
	let mut files = Vec::with_capacity(manifest.files().len());
	for expected in manifest.files() {
		let path = root.join(expected.path());
		let mut file = open_regular(&path)?;
		let before = file.metadata().map_err(|source| ModelsError::Io {
			path: path.clone(),
			source,
		})?;
		let (size, sha256) = hash_reader(&mut file, &path)?;
		let metadata = file.metadata().map_err(|source| ModelsError::Io {
			path: path.clone(),
			source,
		})?;
		if !same_file_snapshot(&before, &metadata) {
			return Err(ModelsError::Configuration(format!(
				"runtime file changed while recording verification stamp: {}",
				path.display()
			)));
		}
		if size != expected.size() || sha256 != expected.sha256() {
			return Err(ModelsError::CorruptFile {
				path,
				expected_size: expected.size(),
				actual_size: size,
				expected_sha256: expected.sha256().to_string(),
				actual_sha256: sha256,
			});
		}
		files.push(StampedFile {
			path: expected.path().to_string(),
			size,
			sha256,
			device: metadata.dev(),
			inode: metadata.ino(),
			mtime_seconds: metadata.mtime(),
			mtime_nanoseconds: metadata.mtime_nsec(),
			ctime_seconds: metadata.ctime(),
			ctime_nanoseconds: metadata.ctime_nsec(),
		});
	}
	let stamp = VerificationStamp {
		schema_version: 1,
		manifest_sha256: hex::encode(sha2::Sha256::digest(&manifest_bytes)),
		files,
	};
	let bytes = serde_json::to_vec(&stamp)
		.map_err(|error| ModelsError::ManifestEncoding(error.to_string()))?;
	if bytes.len() > MAX_VERIFICATION_STAMP_BYTES as usize {
		return Err(ModelsError::ManifestEncoding(
			"verification stamp exceeds 8 MiB".to_string(),
		));
	}
	let target = root.join(VERIFIED_STAMP_NAME);
	let mut file = OpenOptions::new()
		.write(true)
		.create_new(true)
		.mode(0o600)
		.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
		.open(&target)
		.map_err(|source| ModelsError::Io {
			path: target.clone(),
			source,
		})?;
	file.write_all(&bytes).map_err(|source| ModelsError::Io {
		path: target.clone(),
		source,
	})?;
	file.sync_all().map_err(|source| ModelsError::Io {
		path: target,
		source,
	})?;
	sync_directory(root)
}

fn read_manifest(path: &Path) -> Result<ModelManifest, ModelsError> {
	let target = path.join(MANIFEST_NAME);
	let bytes = read_bounded_regular(&target, MAX_MANIFEST_BYTES)?;
	serde_json::from_slice(&bytes).map_err(|error| ModelsError::ManifestEncoding(error.to_string()))
}

fn verify_file_inventory(root: &Path, files: &[ModelFile]) -> Result<(), ModelsError> {
	let mut expected_names = std::collections::BTreeSet::new();
	for expected in files {
		if !expected_names.insert(expected.path().to_string()) {
			return Err(ModelsError::ManifestEncoding(format!(
				"duplicate runtime path {:?}",
				expected.path()
			)));
		}
	}
	let mut actual_names = std::collections::BTreeSet::new();
	for entry in fs::read_dir(root).map_err(|source| ModelsError::Io {
		path: root.to_path_buf(),
		source,
	})? {
		let entry = entry.map_err(|source| ModelsError::Io {
			path: root.to_path_buf(),
			source,
		})?;
		let name = entry
			.file_name()
			.into_string()
			.map_err(|_| ModelsError::UnsafeInstall(entry.path()))?;
		let file_type = entry.file_type().map_err(|source| ModelsError::Io {
			path: entry.path(),
			source,
		})?;
		if matches!(name.as_str(), MANIFEST_NAME | VERIFIED_STAMP_NAME) {
			if !file_type.is_file() || file_type.is_symlink() {
				return Err(ModelsError::UnsafeInstall(entry.path()));
			}
			continue;
		}
		if !file_type.is_file() || file_type.is_symlink() || !runtime_file_name(&name) {
			return Err(ModelsError::UnexpectedRuntimeFile(entry.path()));
		}
		actual_names.insert(name);
	}
	if actual_names != expected_names {
		return Err(ModelsError::RuntimeInventory {
			expected: expected_names.into_iter().collect(),
			actual: actual_names.into_iter().collect(),
		});
	}
	Ok(())
}

fn verify_files(root: &Path, files: &[ModelFile]) -> Result<(), ModelsError> {
	verify_file_inventory(root, files)?;
	if verification_stamp_matches(root, files)? {
		return Ok(());
	}
	for expected in files {
		let path = root.join(expected.path());
		let mut file = open_contained(root, &path)?;
		let mut hash = sha2::Sha256::new();
		let mut size = 0_u64;
		let mut buffer = vec![0_u8; 1024 * 1024];
		loop {
			let read = file.read(&mut buffer).map_err(|source| ModelsError::Io {
				path: path.clone(),
				source,
			})?;
			if read == 0 {
				break;
			}
			hash.update(&buffer[..read]);
			size = size
				.checked_add(u64::try_from(read).map_err(|_| {
					ModelsError::Configuration("verified byte count overflow".to_string())
				})?)
				.ok_or_else(|| {
					ModelsError::Configuration("verified byte count overflow".to_string())
				})?;
		}
		let digest = hex::encode(hash.finalize());
		if size != expected.size() || digest != expected.sha256() {
			return Err(ModelsError::CorruptFile {
				path,
				expected_size: expected.size(),
				actual_size: size,
				expected_sha256: expected.sha256().to_string(),
				actual_sha256: digest,
			});
		}
	}
	Ok(())
}

fn verify_installed_files(root: &Path, manifest: &ModelManifest) -> Result<(), ModelsError> {
	verify_file_inventory(root, manifest.files())?;
	if !verification_stamp_matches(root, manifest.files())? {
		return Err(ModelsError::InvalidVerificationStamp(
			manifest.snapshot_id().clone(),
		));
	}
	Ok(())
}

async fn verify_files_controlled(
	root: &Path,
	files: &[ModelFile],
	cancellation: Option<&DownloadCancellation>,
) -> Result<(), ModelsError> {
	check_download_cancellation(cancellation)?;
	let inventory_root = root.to_path_buf();
	let inventory_files = files.to_vec();
	let stamp_matches = tokio::task::spawn_blocking(move || {
		verify_file_inventory(&inventory_root, &inventory_files)?;
		verification_stamp_matches(&inventory_root, &inventory_files)
	})
	.await
	.map_err(blocking_model_task_error)??;
	check_download_cancellation(cancellation)?;
	if stamp_matches {
		return Ok(());
	}
	for expected in files {
		check_download_cancellation(cancellation)?;
		let path = root.join(expected.path());
		let open_root = root.to_path_buf();
		let open_path = path.clone();
		let file = tokio::task::spawn_blocking(move || open_contained(&open_root, &open_path))
			.await
			.map_err(blocking_model_task_error)??;
		check_download_cancellation(cancellation)?;
		let mut file = tokio::fs::File::from_std(file);
		let mut hash = sha2::Sha256::new();
		let mut size = 0_u64;
		let mut buffer = vec![0_u8; 1024 * 1024];
		loop {
			check_download_cancellation(cancellation)?;
			let read = file
				.read(&mut buffer)
				.await
				.map_err(|source| ModelsError::Io {
					path: path.clone(),
					source,
				})?;
			if read == 0 {
				break;
			}
			hash.update(&buffer[..read]);
			size = size
				.checked_add(u64::try_from(read).map_err(|_| {
					ModelsError::Configuration("verified byte count overflow".to_string())
				})?)
				.ok_or_else(|| {
					ModelsError::Configuration("verified byte count overflow".to_string())
				})?;
		}
		let digest = hex::encode(hash.finalize());
		if size != expected.size() || digest != expected.sha256() {
			return Err(ModelsError::CorruptFile {
				path,
				expected_size: expected.size(),
				actual_size: size,
				expected_sha256: expected.sha256().to_string(),
				actual_sha256: digest,
			});
		}
	}
	check_download_cancellation(cancellation)
}

fn check_download_cancellation(
	cancellation: Option<&DownloadCancellation>,
) -> Result<(), ModelsError> {
	if cancellation.is_some_and(DownloadCancellation::is_cancelled) {
		Err(HubError::Cancelled.into())
	} else {
		Ok(())
	}
}

#[allow(
	clippy::needless_pass_by_value,
	reason = "Result::map_err supplies an owned JoinError directly"
)]
fn blocking_model_task_error(error: tokio::task::JoinError) -> ModelsError {
	ModelsError::Configuration(format!("blocking model task failed: {error}"))
}

fn verification_stamp_matches(root: &Path, files: &[ModelFile]) -> Result<bool, ModelsError> {
	let target = root.join(VERIFIED_STAMP_NAME);
	let bytes = match read_bounded_regular(&target, MAX_VERIFICATION_STAMP_BYTES) {
		Ok(bytes) => bytes,
		Err(ModelsError::Io { source, .. }) if source.kind() == std::io::ErrorKind::NotFound => {
			return Ok(false);
		}
		Err(_) => return Ok(false),
	};
	let Ok(stamp) = serde_json::from_slice::<VerificationStamp>(&bytes) else {
		return Ok(false);
	};
	if stamp.schema_version != 1 || stamp.files.len() != files.len() {
		return Ok(false);
	}
	let manifest_bytes = read_bounded_regular(&root.join(MANIFEST_NAME), MAX_MANIFEST_BYTES)?;
	if stamp.manifest_sha256 != hex::encode(sha2::Sha256::digest(&manifest_bytes)) {
		return Ok(false);
	}
	let expected = files
		.iter()
		.map(|file| (file.path(), file))
		.collect::<std::collections::BTreeMap<_, _>>();
	for stamped in stamp.files {
		let Some(file) = expected.get(stamped.path.as_str()) else {
			return Ok(false);
		};
		if stamped.size != file.size() || stamped.sha256 != file.sha256() {
			return Ok(false);
		}
		let path = root.join(&stamped.path);
		let Ok(opened) = open_regular(&path) else {
			return Ok(false);
		};
		let Ok(metadata) = opened.metadata() else {
			return Ok(false);
		};
		if metadata.len() != stamped.size
			|| metadata.dev() != stamped.device
			|| metadata.ino() != stamped.inode
			|| metadata.mtime() != stamped.mtime_seconds
			|| metadata.mtime_nsec() != stamped.mtime_nanoseconds
			|| metadata.ctime() != stamped.ctime_seconds
			|| metadata.ctime_nsec() != stamped.ctime_nanoseconds
		{
			return Ok(false);
		}
	}
	Ok(true)
}

fn read_bounded_regular(path: &Path, limit: u64) -> Result<Vec<u8>, ModelsError> {
	let mut file = open_regular(path)?;
	let metadata = file.metadata().map_err(|source| ModelsError::Io {
		path: path.to_path_buf(),
		source,
	})?;
	if metadata.len() > limit {
		return Err(ModelsError::ManifestEncoding(format!(
			"file {} exceeds {limit} bytes",
			path.display()
		)));
	}
	let capacity = usize::try_from(metadata.len()).map_err(|_| {
		ModelsError::ManifestEncoding("bounded file size does not fit memory".to_string())
	})?;
	let mut bytes = Vec::with_capacity(capacity);
	std::io::Read::by_ref(&mut file)
		.take(limit.saturating_add(1))
		.read_to_end(&mut bytes)
		.map_err(|source| ModelsError::Io {
			path: path.to_path_buf(),
			source,
		})?;
	if bytes.len() as u64 > limit {
		return Err(ModelsError::ManifestEncoding(format!(
			"file {} exceeds {limit} bytes",
			path.display()
		)));
	}
	Ok(bytes)
}

fn open_regular(path: &Path) -> Result<fs::File, ModelsError> {
	let file = OpenOptions::new()
		.read(true)
		.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK)
		.open(path)
		.map_err(|source| ModelsError::Io {
			path: path.to_path_buf(),
			source,
		})?;
	if !file
		.metadata()
		.map_err(|source| ModelsError::Io {
			path: path.to_path_buf(),
			source,
		})?
		.is_file()
	{
		return Err(ModelsError::UnsafeInstall(path.to_path_buf()));
	}
	Ok(file)
}

fn hash_reader(file: &mut fs::File, path: &Path) -> Result<(u64, String), ModelsError> {
	let mut hash = sha2::Sha256::new();
	let mut size = 0_u64;
	let mut buffer = vec![0_u8; 1024 * 1024];
	loop {
		let read = file.read(&mut buffer).map_err(|source| ModelsError::Io {
			path: path.to_path_buf(),
			source,
		})?;
		if read == 0 {
			break;
		}
		hash.update(&buffer[..read]);
		size = size
			.checked_add(u64::try_from(read).map_err(|_| {
				ModelsError::Configuration("verified byte count overflow".to_string())
			})?)
			.ok_or_else(|| {
				ModelsError::Configuration("verified byte count overflow".to_string())
			})?;
	}
	Ok((size, hex::encode(hash.finalize())))
}

fn same_file_snapshot(left: &fs::Metadata, right: &fs::Metadata) -> bool {
	left.len() == right.len()
		&& left.dev() == right.dev()
		&& left.ino() == right.ino()
		&& left.mtime() == right.mtime()
		&& left.mtime_nsec() == right.mtime_nsec()
		&& left.ctime() == right.ctime()
		&& left.ctime_nsec() == right.ctime_nsec()
}

fn open_contained(root: &Path, path: &Path) -> Result<fs::File, ModelsError> {
	let root = fs::canonicalize(root).map_err(|source| ModelsError::Io {
		path: root.to_path_buf(),
		source,
	})?;
	let parent = path
		.parent()
		.ok_or_else(|| ModelsError::UnsafeInstall(path.to_path_buf()))?;
	let parent = fs::canonicalize(parent).map_err(|source| ModelsError::Io {
		path: parent.to_path_buf(),
		source,
	})?;
	if !parent.starts_with(&root) {
		return Err(ModelsError::UnsafeInstall(path.to_path_buf()));
	}
	open_regular(path)
}

fn contained_directory(root: &Path, path: &Path) -> Result<PathBuf, ModelsError> {
	let root = fs::canonicalize(root).map_err(|source| ModelsError::Io {
		path: root.to_path_buf(),
		source,
	})?;
	let path = fs::canonicalize(path).map_err(|source| ModelsError::Io {
		path: path.to_path_buf(),
		source,
	})?;
	if path == root || !path.starts_with(&root) {
		return Err(ModelsError::UnsafeInstall(path));
	}
	Ok(path)
}

fn prepare_owned_directory(root: &Path, path: &Path) -> Result<PathBuf, ModelsError> {
	let relative = path
		.strip_prefix(root)
		.map_err(|_| ModelsError::UnsafeInstall(path.to_path_buf()))?;
	let components = relative
		.components()
		.map(|component| {
			component
				.as_os_str()
				.to_str()
				.map(str::to_string)
				.ok_or_else(|| ModelsError::UnsafeInstall(path.to_path_buf()))
		})
		.collect::<Result<Vec<_>, _>>()?;
	let borrowed = components.iter().map(String::as_str).collect::<Vec<_>>();
	create_owner_subdir(root, &borrowed).map_err(|source| ModelsError::Io {
		path: path.to_path_buf(),
		source,
	})
}

fn make_read_only_contents(path: &Path) -> Result<(), ModelsError> {
	for entry in WalkDir::new(path)
		.min_depth(1)
		.contents_first(true)
		.follow_links(false)
	{
		let entry = entry.map_err(|error| ModelsError::Walk(error.to_string()))?;
		if entry.file_type().is_symlink() {
			return Err(ModelsError::UnsafeInstall(entry.path().to_path_buf()));
		}
		let mode = if entry.file_type().is_dir() {
			0o500
		} else {
			0o400
		};
		set_mode(entry.path(), mode)?;
	}
	Ok(())
}

fn set_mode(path: &Path, mode: u32) -> Result<(), ModelsError> {
	fs::set_permissions(path, fs::Permissions::from_mode(mode)).map_err(|source| ModelsError::Io {
		path: path.to_path_buf(),
		source,
	})
}

fn make_writable(path: &Path) -> Result<(), ModelsError> {
	for entry in WalkDir::new(path).follow_links(false) {
		let entry = entry.map_err(|error| ModelsError::Walk(error.to_string()))?;
		if entry.file_type().is_symlink() {
			return Err(ModelsError::UnsafeInstall(entry.path().to_path_buf()));
		}
		let mode = if entry.file_type().is_dir() {
			0o700
		} else {
			0o600
		};
		fs::set_permissions(entry.path(), fs::Permissions::from_mode(mode)).map_err(|source| {
			ModelsError::Io {
				path: entry.path().to_path_buf(),
				source,
			}
		})?;
	}
	Ok(())
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct QuarantineRecord {
	schema_version: u32,
	quarantined_at: DateTime<Utc>,
	reason: String,
	snapshot_id: Option<ModelSnapshotId>,
}

fn move_to_quarantine(
	parent: &Path,
	source: &Path,
	reason: &str,
	snapshot_id: Option<ModelSnapshotId>,
) -> Result<PathBuf, ModelsError> {
	if !valid_quarantine_reason(reason) {
		return Err(ModelsError::Configuration(
			"invalid quarantine reason".to_string(),
		));
	}
	let destination = parent.join(format!("{reason}-{}", uuid::Uuid::now_v7()));
	fs::create_dir(&destination).map_err(|source_error| ModelsError::Io {
		path: destination.clone(),
		source: source_error,
	})?;
	fs::set_permissions(&destination, fs::Permissions::from_mode(0o700)).map_err(
		|source_error| ModelsError::Io {
			path: destination.clone(),
			source: source_error,
		},
	)?;
	let record = QuarantineRecord {
		schema_version: 1,
		quarantined_at: Utc::now(),
		reason: reason.to_string(),
		snapshot_id,
	};
	let encoded = serde_json::to_vec(&record)
		.map_err(|error| ModelsError::ManifestEncoding(error.to_string()))?;
	let record_path = destination.join(QUARANTINE_RECORD_NAME);
	let mut record_file = OpenOptions::new()
		.write(true)
		.create_new(true)
		.mode(0o600)
		.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
		.open(&record_path)
		.map_err(|source_error| ModelsError::Io {
			path: record_path.clone(),
			source: source_error,
		})?;
	record_file
		.write_all(&encoded)
		.map_err(|source_error| ModelsError::Io {
			path: record_path.clone(),
			source: source_error,
		})?;
	record_file
		.sync_all()
		.map_err(|source_error| ModelsError::Io {
			path: record_path,
			source: source_error,
		})?;
	drop(record_file);
	let payload = destination.join("payload");
	if let Err(source_error) = fs::rename(source, &payload) {
		let _ = fs::remove_file(destination.join(QUARANTINE_RECORD_NAME));
		let _ = fs::remove_dir(&destination);
		return Err(ModelsError::Io {
			path: source.to_path_buf(),
			source: source_error,
		});
	}
	if let Some(source_parent) = source.parent() {
		sync_directory(source_parent)?;
	}
	sync_directory(&destination)?;
	sync_directory(parent)?;
	Ok(destination)
}

fn read_quarantine_record(path: &Path) -> Result<QuarantineRecord, ModelsError> {
	let record_path = path.join(QUARANTINE_RECORD_NAME);
	let bytes = read_bounded_regular(&record_path, 64 << 10)?;
	let record: QuarantineRecord = serde_json::from_slice(&bytes)
		.map_err(|error| ModelsError::ManifestEncoding(error.to_string()))?;
	if record.schema_version != 1 || !valid_quarantine_reason(&record.reason) {
		return Err(ModelsError::ManifestEncoding(
			"invalid quarantine record".to_string(),
		));
	}
	Ok(record)
}

fn valid_quarantine_reason(reason: &str) -> bool {
	!reason.is_empty()
		&& reason.len() <= 64
		&& reason
			.bytes()
			.all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
}

fn preflight_disk(path: &Path, transfer_bytes: u64) -> Result<(), ModelsError> {
	let available = crate::home::available_disk_bytes(path).map_err(|source| {
		if matches!(
			source.kind(),
			std::io::ErrorKind::InvalidInput | std::io::ErrorKind::InvalidData
		) {
			ModelsError::Configuration(source.to_string())
		} else {
			ModelsError::Io {
				path: path.to_path_buf(),
				source,
			}
		}
	})?;
	let required = required_download_storage_bytes(transfer_bytes)
		.ok_or_else(|| ModelsError::Configuration("required disk size overflow".to_string()))?;
	if available < required {
		return Err(ModelsError::InsufficientDisk {
			path: path.to_path_buf(),
			required,
			available,
		});
	}
	Ok(())
}

fn sync_directory(path: &Path) -> Result<(), ModelsError> {
	let directory = OpenOptions::new()
		.read(true)
		.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
		.open(path)
		.map_err(|source| ModelsError::Io {
			path: path.to_path_buf(),
			source,
		})?;
	directory.sync_all().map_err(|source| ModelsError::Io {
		path: path.to_path_buf(),
		source,
	})
}

fn bounded_diagnostic(message: &str) -> String {
	const MAX_CHARS: usize = 4_096;
	let mut bounded = message.chars().take(MAX_CHARS).collect::<String>();
	if message.chars().count() > MAX_CHARS {
		bounded.push_str(" [truncated]");
	}
	bounded
}

struct StagingGuard {
	path: PathBuf,
	quarantine_parent: PathBuf,
	active: bool,
	#[cfg(test)]
	cleanup_delay: Option<Duration>,
}

struct StagingCleanupTask {
	path: PathBuf,
	quarantine_parent: PathBuf,
	#[cfg(test)]
	delay: Option<Duration>,
}

#[allow(
	clippy::needless_pass_by_value,
	reason = "cleanup worker consumes the queued task and owns its path state"
)]
fn run_staging_cleanup(task: StagingCleanupTask) {
	#[cfg(test)]
	STAGING_CLEANUP_TASKS_STARTED.fetch_add(1, Ordering::Relaxed);
	#[cfg(test)]
	if let Some(delay) = task.delay {
		std::thread::sleep(delay);
	}
	let _ = move_to_quarantine(&task.quarantine_parent, &task.path, "failed", None);
}

#[cfg(test)]
static STAGING_CLEANUP_WORKERS_STARTED: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
static STAGING_CLEANUP_TASKS_STARTED: AtomicUsize = AtomicUsize::new(0);

fn defer_staging_cleanup(task: StagingCleanupTask) {
	static CLEANUP_QUEUE: OnceLock<Option<mpsc::Sender<StagingCleanupTask>>> = OnceLock::new();
	let sender = CLEANUP_QUEUE.get_or_init(|| {
		let (sender, receiver) = mpsc::channel::<StagingCleanupTask>();
		std::thread::Builder::new()
			.name("emelex-staging-cleanup".to_string())
			.spawn(move || {
				#[cfg(test)]
				STAGING_CLEANUP_WORKERS_STARTED.fetch_add(1, Ordering::Relaxed);
				while let Ok(task) = receiver.recv() {
					run_staging_cleanup(task);
				}
			})
			.ok()
			.map(|_| sender)
	});
	if let Some(sender) = sender {
		let _ = sender.send(task);
	}
}

impl StagingGuard {
	fn path(&self) -> &Path {
		&self.path
	}

	const fn commit(&mut self) {
		self.active = false;
	}

	fn moved_to(&mut self, path: &Path) {
		self.path = path.to_path_buf();
	}
}

impl Drop for StagingGuard {
	fn drop(&mut self) {
		if !self.active {
			return;
		}
		self.active = false;
		let path = std::mem::take(&mut self.path);
		let quarantine_parent = std::mem::take(&mut self.quarantine_parent);
		defer_staging_cleanup(StagingCleanupTask {
			path,
			quarantine_parent,
			#[cfg(test)]
			delay: self.cleanup_delay,
		});
	}
}

/// Installed-model lifecycle failure.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ModelsError {
	/// Hub operation failed.
	#[error(transparent)]
	Hub(#[from] HubError),
	/// Static inspection failed.
	#[error(transparent)]
	Inspection(#[from] InspectionError),
	/// Manifest validation failed.
	#[error(transparent)]
	Manifest(#[from] ManifestError),
	/// Client load failed.
	#[error(transparent)]
	Client(#[from] Error),
	/// Candidate planning, inspection, load, or runtime probing failed.
	///
	/// Interactive discovery may offer another candidate for this variant.
	/// Infrastructure, cancellation, storage, and global runtime failures retain
	/// their original variants and must not be downgraded to this error.
	#[error("model candidate certification failed: {0}")]
	Certification(#[source] Box<Self>),
	/// Model is not installed.
	#[error("model is not installed: {0}")]
	NotInstalled(ModelRef),
	/// Exact snapshot is not installed.
	#[error("model snapshot is not installed: {0}")]
	SnapshotNotInstalled(ModelSnapshotId),
	/// A managed-store entry lacks successful runtime verification.
	#[error("model snapshot is not runtime verified: {0}")]
	UnverifiedSnapshot(ModelSnapshotId),
	/// Durable state still references the snapshot.
	#[error("model snapshot is still referenced by a durable session: {0}")]
	SnapshotReferenced(ModelSnapshotId),
	/// Snapshot-reference policy could not decide safely.
	#[error("model snapshot reference guard failed: {0}")]
	ReferenceGuard(#[source] SnapshotReferenceError),
	/// Cross-process snapshot/reference mutation lock could not be acquired.
	#[error("model snapshot mutation lock failed: {0}")]
	SnapshotMutationLock(String),
	/// Static compatibility failed.
	#[error("model is incompatible: {0:?}")]
	Incompatible(Vec<String>),
	/// Owned storage I/O failed.
	#[error("I/O failed for {path:?}: {source}")]
	Io {
		/// Affected path.
		path: PathBuf,
		/// Underlying failure.
		#[source]
		source: std::io::Error,
	},
	/// Directory walk failed.
	#[error("cannot inspect model store: {0}")]
	Walk(String),
	/// Path or entry escaped immutable store invariants.
	#[error("unsafe installed-model path {0:?}")]
	UnsafeInstall(PathBuf),
	/// An unlisted or unsupported runtime file was found.
	#[error("unexpected installed-model runtime file {0:?}")]
	UnexpectedRuntimeFile(PathBuf),
	/// Exact runtime file inventory disagrees with the manifest.
	#[error("installed-model runtime inventory differs: expected {expected:?}, actual {actual:?}")]
	RuntimeInventory {
		/// Manifest paths.
		expected: Vec<String>,
		/// On-disk paths.
		actual: Vec<String>,
	},
	/// Manifest encoding failed.
	#[error("invalid model manifest: {0}")]
	ManifestEncoding(String),
	/// Immutable file changed.
	#[error(
		"corrupt model file {path:?}: expected {expected_size}/{expected_sha256}, \
		 got {actual_size}/{actual_sha256}"
	)]
	CorruptFile {
		/// Affected file.
		path: PathBuf,
		/// Manifest size.
		expected_size: u64,
		/// Actual size.
		actual_size: u64,
		/// Manifest digest.
		expected_sha256: String,
		/// Actual digest.
		actual_sha256: String,
	},
	/// Managed snapshot's immutable verification stamp no longer matches.
	#[error("model snapshot verification stamp is missing or invalid: {0}")]
	InvalidVerificationStamp(ModelSnapshotId),
	/// Resolved configuration was unusable.
	#[error("invalid model configuration: {0}")]
	Configuration(String),
	/// Available storage cannot hold a complete staged transfer.
	#[error(
		"insufficient disk space at {path:?}: required {required} bytes, available {available}"
	)]
	InsufficientDisk {
		/// Target filesystem.
		path: PathBuf,
		/// Transfer plus safety margin.
		required: u64,
		/// Filesystem-reported available bytes.
		available: u64,
	},
}

fn mark_candidate_certification_error(error: ModelsError) -> ModelsError {
	let candidate_local = match &error {
		ModelsError::Inspection(
			InspectionError::Json { .. }
			| InspectionError::Config { .. }
			| InspectionError::Layout { .. },
		)
		| ModelsError::Incompatible(_)
		| ModelsError::Manifest(_) => true,
		ModelsError::Client(error) => matches!(
			error,
			Error::ModelLoad { .. }
				| Error::Generation(_)
				| Error::ContextExceeded { .. }
				| Error::CapabilityUnavailable { .. }
		),
		_ => false,
	};
	if candidate_local {
		ModelsError::Certification(Box::new(error))
	} else {
		error
	}
}

fn mark_hub_candidate_certification_error(error: HubError) -> ModelsError {
	match error {
		error @ (HubError::NotPublic(_) | HubError::Incompatible(_)) => {
			ModelsError::Certification(Box::new(ModelsError::Hub(error)))
		}
		error => ModelsError::Hub(error),
	}
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
