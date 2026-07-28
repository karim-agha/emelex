//! Installed model manifest and immutable file plan.

use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Deserializer, Serialize};
use sha2::Digest as _;

use super::{
	EvidenceSource, ModelRef, ModelSnapshotId, ModelTraits, MtpSupport, ResolvedRevision,
	SnapshotDigest, Task, TraitConfidence, VerificationStatus, layout::safe_relative_path,
};

/// Source of an Emelex-owned install.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
#[non_exhaustive]
pub enum ModelSource {
	/// Hugging Face repository snapshot.
	Hub,
	/// User-selected local directory copied or moved into Emelex home.
	LocalImport {
		/// Original canonical source path for provenance.
		original_path: PathBuf,
	},
	/// User-selected local directory referenced outside Emelex home.
	///
	/// The managed snapshot contains only a link record. Runtime files remain
	/// caller-owned and are revalidated before every resolution or load.
	LocalSymlink {
		/// Canonical external source path.
		original_path: PathBuf,
	},
}

/// One selected checkpoint file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[non_exhaustive]
pub struct ModelFile {
	path: String,
	size: u64,
	sha256: String,
}

impl ModelFile {
	/// Construct one validated immutable-file record.
	///
	/// # Errors
	///
	/// Returns [`ManifestError`] for traversal paths or malformed SHA-256.
	pub fn new(
		path: impl Into<String>,
		size: u64,
		sha256: impl Into<String>,
	) -> Result<Self, ManifestError> {
		let path = path.into();
		let sha256 = sha256.into().to_ascii_lowercase();
		if !safe_relative_path(&path) {
			return Err(ManifestError::UnsafePath(path));
		}
		if sha256.len() != 64 || !sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
			return Err(ManifestError::InvalidDigest {
				path,
				digest: sha256,
			});
		}
		Ok(Self { path, size, sha256 })
	}

	/// Repository-relative safe path.
	pub fn path(&self) -> &str {
		&self.path
	}

	/// Expected bytes.
	pub const fn size(&self) -> u64 {
		self.size
	}

	/// Lowercase SHA-256 digest.
	pub fn sha256(&self) -> &str {
		&self.sha256
	}
}

impl<'de> Deserialize<'de> for ModelFile {
	fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
	where
		D: Deserializer<'de>,
	{
		#[derive(Deserialize)]
		#[serde(deny_unknown_fields)]
		struct Wire {
			path: String,
			size: u64,
			sha256: String,
		}

		let wire = Wire::deserialize(deserializer)?;
		Self::new(wire.path, wire.size, wire.sha256).map_err(serde::de::Error::custom)
	}
}

/// Versioned install record.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[non_exhaustive]
pub struct ModelManifest {
	schema_version: u32,
	reference: ModelRef,
	source: ModelSource,
	resolved_revision: Option<ResolvedRevision>,
	snapshot_id: ModelSnapshotId,
	files: Vec<ModelFile>,
	traits: ModelTraits,
	verification: VerificationStatus,
	compatibility_engine: String,
	installed_at: DateTime<Utc>,
	license: Option<String>,
}

impl ModelManifest {
	/// Current manifest schema.
	pub const SCHEMA_VERSION: u32 = 2;
	/// Compatibility implementation identifier.
	pub const COMPATIBILITY_ENGINE: &'static str = "emelex-1";

	/// Construct a validated install manifest.
	///
	/// # Errors
	///
	/// Returns [`ManifestError`] when source/revision, file, trait, or
	/// verification invariants do not describe a runnable immutable snapshot.
	pub fn new(
		reference: ModelRef,
		source: ModelSource,
		resolved_revision: Option<ResolvedRevision>,
		files: Vec<ModelFile>,
		traits: ModelTraits,
		verification: VerificationStatus,
		license: Option<String>,
	) -> Result<Self, ManifestError> {
		let snapshot_id = snapshot_id(&reference, resolved_revision.as_ref(), &files)?;
		Self::from_parts(ManifestParts {
			schema_version: Self::SCHEMA_VERSION,
			reference,
			source,
			resolved_revision,
			snapshot_id,
			files,
			traits,
			verification,
			compatibility_engine: Self::COMPATIBILITY_ENGINE.to_string(),
			installed_at: Utc::now(),
			license,
		})
	}

	#[allow(
		clippy::too_many_lines,
		reason = "manifest deserialization and construction share one fail-closed invariant gate"
	)]
	fn from_parts(parts: ManifestParts) -> Result<Self, ManifestError> {
		if !matches!(parts.schema_version, 1 | Self::SCHEMA_VERSION) {
			return Err(ManifestError::UnsupportedSchema(parts.schema_version));
		}
		if parts.schema_version == 1 && matches!(parts.source, ModelSource::LocalSymlink { .. }) {
			return Err(ManifestError::SourceRequiresSchema {
				kind: "local_symlink",
				schema_version: Self::SCHEMA_VERSION,
			});
		}
		if parts.compatibility_engine != Self::COMPATIBILITY_ENGINE {
			return Err(ManifestError::UnsupportedCompatibilityEngine(
				parts.compatibility_engine,
			));
		}
		match (&parts.source, &parts.resolved_revision) {
			(ModelSource::Hub, None) => return Err(ManifestError::MissingHubRevision),
			(ModelSource::LocalImport { .. } | ModelSource::LocalSymlink { .. }, Some(_)) => {
				return Err(ManifestError::LocalImportHasRevision);
			}
			(
				ModelSource::LocalImport { original_path }
				| ModelSource::LocalSymlink { original_path },
				None,
			) if !original_path.is_absolute() => {
				return Err(ManifestError::LocalImportPathNotAbsolute(
					original_path.clone(),
				));
			}
			_ => {}
		}
		if !matches!(
			(&parts.source, &parts.reference),
			(ModelSource::Hub, ModelRef::Hub(_))
				| (
					ModelSource::LocalImport { .. } | ModelSource::LocalSymlink { .. },
					ModelRef::Local(_)
				)
		) {
			return Err(ManifestError::SourceReferenceMismatch);
		}
		let expected_snapshot = snapshot_id(
			&parts.reference,
			parts.resolved_revision.as_ref(),
			&parts.files,
		)?;
		if parts.snapshot_id != expected_snapshot {
			return Err(ManifestError::SnapshotIdentityMismatch {
				recorded: parts.snapshot_id,
				expected: expected_snapshot,
			});
		}
		if parts.files.is_empty() || parts.files.len() > 10_000 {
			return Err(ManifestError::EmptyFilePlan);
		}
		let mut paths = std::collections::BTreeSet::new();
		let mut weights_bytes = 0_u64;
		for file in &parts.files {
			if Path::new(file.path()).components().count() != 1 {
				return Err(ManifestError::UnsafePath(file.path().to_string()));
			}
			if !paths.insert(file.path()) {
				return Err(ManifestError::DuplicatePath(file.path().to_string()));
			}
			if file.path().ends_with(".safetensors") {
				if file.size() == 0 {
					return Err(ManifestError::EmptyWeight(file.path().to_string()));
				}
				weights_bytes = weights_bytes
					.checked_add(file.size())
					.ok_or(ManifestError::WeightBytesOverflow)?;
			}
		}
		if !paths.contains("config.json") || !paths.contains("tokenizer.json") || weights_bytes == 0
		{
			return Err(ManifestError::MissingRuntimeFile);
		}
		if !parts.traits.mlx {
			return Err(ManifestError::NotMlxCompatible);
		}
		if !parts.traits.tasks.contains(&Task::TextGeneration) {
			return Err(ManifestError::MissingTextGeneration);
		}
		let defaults = &parts.traits.generation_defaults;
		if defaults
			.temperature
			.is_some_and(|value| !value.is_finite() || !(0.0..=2.0).contains(&value))
		{
			return Err(ManifestError::InvalidGenerationDefaults(
				"temperature must be finite and in 0..=2",
			));
		}
		if defaults
			.top_p
			.is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
		{
			return Err(ManifestError::InvalidGenerationDefaults(
				"top_p must be finite and in 0..=1",
			));
		}
		if defaults.max_new_tokens == Some(0) {
			return Err(ManifestError::InvalidGenerationDefaults(
				"max_new_tokens must be positive",
			));
		}
		let sizing = parts
			.traits
			.sizing
			.as_ref()
			.ok_or(ManifestError::SizingMismatch)?;
		if sizing.weights_bytes != Some(weights_bytes) {
			return Err(ManifestError::WeightBytesMismatch {
				manifest: weights_bytes,
				traits: sizing.weights_bytes.unwrap_or_default(),
			});
		}
		if sizing.evaluated_context_tokens == Some(0) {
			return Err(ManifestError::ZeroContext);
		}
		if sizing.evaluated_context_tokens.is_none()
			|| sizing.max_context_tokens == Some(0)
			|| sizing
				.estimated_residency_bytes
				.is_none_or(|residency| residency < weights_bytes)
		{
			return Err(ManifestError::SizingMismatch);
		}
		if parts.verification == VerificationStatus::Verified {
			let runtime_evidence = parts.traits.evidence.iter().any(|evidence| {
				evidence.source == EvidenceSource::Runtime
					&& evidence.trait_key == "compatibility:runtime_load"
			});
			let runtime_confidence =
				["acceleration:mlx", "task:text_generation"]
					.iter()
					.all(|key| {
						parts.traits.confidence.get(*key) == Some(&TraitConfidence::RuntimeVerified)
					});
			if !runtime_evidence || !runtime_confidence {
				return Err(ManifestError::MissingRuntimeEvidence);
			}
		}
		if parts.traits.mtp == MtpSupport::RuntimeVerified {
			if parts.verification != VerificationStatus::Verified {
				return Err(ManifestError::MissingRuntimeEvidence);
			}
			if !parts.traits.evidence.iter().any(|evidence| {
				evidence.source == EvidenceSource::Runtime
					&& evidence.trait_key == "acceleration:mtp"
			}) {
				return Err(ManifestError::MissingMtpEvidence);
			}
		}
		if parts
			.license
			.as_ref()
			.is_some_and(|value| value.len() > 64 << 10)
		{
			return Err(ManifestError::LicenseTooLarge);
		}
		Ok(Self {
			schema_version: parts.schema_version,
			reference: parts.reference,
			source: parts.source,
			resolved_revision: parts.resolved_revision,
			snapshot_id: parts.snapshot_id,
			files: parts.files,
			traits: parts.traits,
			verification: parts.verification,
			compatibility_engine: parts.compatibility_engine,
			installed_at: parts.installed_at,
			license: parts.license,
		})
	}

	/// Manifest schema.
	pub const fn schema_version(&self) -> u32 {
		self.schema_version
	}

	/// Stable model reference.
	pub const fn reference(&self) -> &ModelRef {
		&self.reference
	}

	/// Snapshot source.
	pub const fn source(&self) -> &ModelSource {
		&self.source
	}

	/// Immutable Hub revision, when applicable.
	pub const fn resolved_revision(&self) -> Option<&ResolvedRevision> {
		self.resolved_revision.as_ref()
	}

	/// Exact immutable snapshot address.
	pub const fn snapshot_id(&self) -> &ModelSnapshotId {
		&self.snapshot_id
	}

	/// Runnable immutable file plan.
	pub fn files(&self) -> &[ModelFile] {
		&self.files
	}

	/// Recorded capability traits.
	pub const fn traits(&self) -> &ModelTraits {
		&self.traits
	}

	/// Static or runtime verification state.
	pub const fn verification(&self) -> VerificationStatus {
		self.verification
	}

	/// Compatibility implementation identifier.
	pub fn compatibility_engine(&self) -> &str {
		&self.compatibility_engine
	}

	/// Installation timestamp.
	pub const fn installed_at(&self) -> DateTime<Utc> {
		self.installed_at
	}

	/// Repository-provided license identifier or label.
	pub fn license(&self) -> Option<&str> {
		self.license.as_deref()
	}
}

impl<'de> Deserialize<'de> for ModelManifest {
	fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
	where
		D: Deserializer<'de>,
	{
		#[derive(Deserialize)]
		#[serde(deny_unknown_fields)]
		struct Wire {
			schema_version: u32,
			reference: ModelRef,
			source: ModelSource,
			resolved_revision: Option<ResolvedRevision>,
			snapshot_id: ModelSnapshotId,
			files: Vec<ModelFile>,
			traits: ModelTraits,
			verification: VerificationStatus,
			compatibility_engine: String,
			installed_at: DateTime<Utc>,
			license: Option<String>,
		}

		let wire = Wire::deserialize(deserializer)?;
		Self::from_parts(ManifestParts {
			schema_version: wire.schema_version,
			reference: wire.reference,
			source: wire.source,
			resolved_revision: wire.resolved_revision,
			snapshot_id: wire.snapshot_id,
			files: wire.files,
			traits: wire.traits,
			verification: wire.verification,
			compatibility_engine: wire.compatibility_engine,
			installed_at: wire.installed_at,
			license: wire.license,
		})
		.map_err(serde::de::Error::custom)
	}
}

struct ManifestParts {
	schema_version: u32,
	reference: ModelRef,
	source: ModelSource,
	resolved_revision: Option<ResolvedRevision>,
	snapshot_id: ModelSnapshotId,
	files: Vec<ModelFile>,
	traits: ModelTraits,
	verification: VerificationStatus,
	compatibility_engine: String,
	installed_at: DateTime<Utc>,
	license: Option<String>,
}

/// Invalid immutable install manifest.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ManifestError {
	/// Manifest schema is not supported.
	#[error("unsupported model manifest schema {0}")]
	UnsupportedSchema(u32),
	/// A source kind requires a newer manifest schema.
	#[error("model source {kind} requires manifest schema {schema_version}")]
	SourceRequiresSchema {
		/// Serialized source kind.
		kind: &'static str,
		/// First schema supporting it.
		schema_version: u32,
	},
	/// Compatibility engine differs from this build.
	#[error("unsupported compatibility engine {0:?}")]
	UnsupportedCompatibilityEngine(String),
	/// Hub snapshots must pin a full revision.
	#[error("Hub model manifest is missing a resolved revision")]
	MissingHubRevision,
	/// Local imports cannot carry a Hub revision.
	#[error("local import manifest cannot carry a Hub revision")]
	LocalImportHasRevision,
	/// Local import provenance must be canonical.
	#[error("local import provenance is not absolute: {0:?}")]
	LocalImportPathNotAbsolute(PathBuf),
	/// Source kind and stable reference kind disagree.
	#[error("model source does not match model reference kind")]
	SourceReferenceMismatch,
	/// Exact snapshot ID disagrees with the immutable manifest contents.
	#[error("model snapshot ID mismatch: recorded {recorded}, expected {expected}")]
	SnapshotIdentityMismatch {
		/// Serialized snapshot ID.
		recorded: ModelSnapshotId,
		/// ID derived from source, revision, and files.
		expected: ModelSnapshotId,
	},
	/// Exact snapshot ID could not be derived from validated manifest data.
	#[error("cannot derive model snapshot ID: {0}")]
	SnapshotIdentity(String),
	/// File plan is empty.
	#[error("model manifest file plan is empty")]
	EmptyFilePlan,
	/// File path is unsafe.
	#[error("unsafe model file path {0:?}")]
	UnsafePath(String),
	/// SHA-256 is malformed.
	#[error("invalid SHA-256 for {path:?}: {digest:?}")]
	InvalidDigest {
		/// Affected relative path.
		path: String,
		/// Malformed digest.
		digest: String,
	},
	/// File path appears more than once.
	#[error("duplicate model file path {0:?}")]
	DuplicatePath(String),
	/// Required runtime artifacts are missing.
	#[error("model manifest requires config.json, tokenizer.json, and weight files")]
	MissingRuntimeFile,
	/// A weight artifact was empty.
	#[error("model weight file is empty: {0:?}")]
	EmptyWeight(String),
	/// Aggregate weight bytes overflowed.
	#[error("model manifest weight-byte total overflow")]
	WeightBytesOverflow,
	/// File-plan weight bytes disagree with recorded sizing.
	#[error("model weight bytes disagree: files={manifest}, sizing={traits}")]
	WeightBytesMismatch {
		/// Sum from file plan.
		manifest: u64,
		/// Recorded sizing value, or zero when absent.
		traits: u64,
	},
	/// Trait snapshot did not establish MLX compatibility.
	#[error("manifest traits do not establish MLX compatibility")]
	NotMlxCompatible,
	/// Text generation was not established.
	#[error("manifest traits do not establish text generation")]
	MissingTextGeneration,
	/// Checkpoint-advertised generation defaults are outside supported bounds.
	#[error("invalid model generation defaults: {0}")]
	InvalidGenerationDefaults(&'static str),
	/// Sizing context was zero.
	#[error("manifest sizing context must be positive")]
	ZeroContext,
	/// Sizing facts are absent or describe impossible residency.
	#[error("manifest sizing facts are inconsistent")]
	SizingMismatch,
	/// Verified manifests must carry runtime evidence.
	#[error("verified manifest lacks runtime evidence")]
	MissingRuntimeEvidence,
	/// Runtime-verified MTP must carry exact parity evidence.
	#[error("runtime-verified MTP lacks parity evidence")]
	MissingMtpEvidence,
	/// License payload is unexpectedly large.
	#[error("manifest license field exceeds 64 KiB")]
	LicenseTooLarge,
}

/// Emelex-owned immutable model snapshot.
#[derive(Debug, Clone)]
pub struct InstalledModel {
	path: PathBuf,
	manifest: ModelManifest,
}

impl InstalledModel {
	/// Construct from a verified manifest and canonical install directory.
	pub(crate) const fn new(path: PathBuf, manifest: ModelManifest) -> Self {
		Self { path, manifest }
	}

	/// Snapshot directory.
	pub fn path(&self) -> &Path {
		&self.path
	}

	/// Verified manifest.
	pub const fn manifest(&self) -> &ModelManifest {
		&self.manifest
	}

	/// Stable reference.
	pub const fn reference(&self) -> &ModelRef {
		self.manifest.reference()
	}

	/// Exact immutable snapshot address.
	pub const fn snapshot_id(&self) -> &ModelSnapshotId {
		self.manifest.snapshot_id()
	}
}

fn snapshot_id(
	reference: &ModelRef,
	revision: Option<&ResolvedRevision>,
	files: &[ModelFile],
) -> Result<ModelSnapshotId, ManifestError> {
	match (reference, revision) {
		(ModelRef::Hub(id), Some(revision)) => Ok(ModelSnapshotId::Hub {
			id: id.clone(),
			revision: revision.clone(),
		}),
		(ModelRef::Local(name), None) => {
			let mut ordered = files.iter().collect::<Vec<_>>();
			ordered.sort_by(|left, right| left.path().cmp(right.path()));
			let mut hash = sha2::Sha256::new();
			for file in ordered {
				hash.update(file.path().as_bytes());
				hash.update([0]);
				hash.update(file.size().to_le_bytes());
				hash.update([0]);
				hash.update(file.sha256().as_bytes());
				hash.update([0]);
			}
			let digest = SnapshotDigest::parse(hex::encode(hash.finalize()))
				.map_err(|error| ManifestError::SnapshotIdentity(error.to_string()))?;
			Ok(ModelSnapshotId::Local {
				name: name.clone(),
				digest,
			})
		}
		_ => Err(ManifestError::SnapshotIdentity(
			"reference and revision kinds disagree".to_string(),
		)),
	}
}

#[cfg(test)]
#[path = "manifest_tests.rs"]
mod tests;
