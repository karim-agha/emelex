use std::collections::BTreeSet;

use super::*;
use crate::model::{HubModelId, Modality, ModelGenerationDefaults, ModelSizing};

fn files() -> Vec<ModelFile> {
	vec![
		ModelFile::new("config.json", 2, "a".repeat(64)).expect("config"),
		ModelFile::new("model.safetensors", 4, "b".repeat(64)).expect("weights"),
		ModelFile::new("tokenizer.json", 2, "c".repeat(64)).expect("tokenizer"),
	]
}

fn traits() -> ModelTraits {
	ModelTraits {
		input: BTreeSet::from([Modality::Text]),
		output: BTreeSet::from([Modality::Text]),
		tasks: BTreeSet::from([Task::TextGeneration]),
		mlx: true,
		sizing: Some(ModelSizing {
			weights_bytes: Some(4),
			estimated_residency_bytes: Some(12),
			evaluated_context_tokens: Some(16),
			max_context_tokens: Some(32),
		}),
		..ModelTraits::default()
	}
}

#[test]
fn hub_snapshot_id_pins_exact_revision() {
	let id = HubModelId::parse("owner/model").expect("valid Hub ID");
	let revision = ResolvedRevision::parse("a".repeat(40)).expect("valid revision");
	let manifest = ModelManifest::new(
		ModelRef::Hub(id.clone()),
		ModelSource::Hub,
		Some(revision.clone()),
		files(),
		traits(),
		VerificationStatus::Estimated,
		None,
	)
	.expect("valid manifest");
	assert_eq!(
		manifest.snapshot_id(),
		&ModelSnapshotId::Hub { id, revision }
	);
}

#[test]
fn local_snapshot_digest_is_independent_of_file_order() {
	let name = crate::model::LocalModelName::parse("experiment").expect("valid local name");
	let mut reversed = files();
	reversed.reverse();
	let first = ModelManifest::new(
		ModelRef::Local(name.clone()),
		ModelSource::LocalImport {
			original_path: PathBuf::from("/tmp/model"),
		},
		None,
		files(),
		traits(),
		VerificationStatus::Estimated,
		None,
	)
	.expect("valid manifest");
	let second = ModelManifest::new(
		ModelRef::Local(name),
		ModelSource::LocalImport {
			original_path: PathBuf::from("/tmp/model"),
		},
		None,
		reversed,
		traits(),
		VerificationStatus::Estimated,
		None,
	)
	.expect("valid manifest");
	assert_eq!(first.snapshot_id(), second.snapshot_id());
}

#[test]
fn deserialization_rejects_forged_snapshot_id() {
	let manifest = ModelManifest::new(
		ModelRef::Hub(HubModelId::parse("owner/model").expect("valid Hub ID")),
		ModelSource::Hub,
		Some(ResolvedRevision::parse("a".repeat(40)).expect("valid revision")),
		files(),
		traits(),
		VerificationStatus::Estimated,
		None,
	)
	.expect("valid manifest");
	let mut value = serde_json::to_value(manifest).expect("serialize manifest");
	value["snapshot_id"] = serde_json::Value::String(format!("owner/model@{}", "b".repeat(40)));
	assert!(serde_json::from_value::<ModelManifest>(value).is_err());
}

#[test]
fn manifest_rejects_inconsistent_typed_sizing() {
	let mut inconsistent = traits();
	inconsistent
		.sizing
		.as_mut()
		.expect("fixture sizing")
		.estimated_residency_bytes = Some(3);
	let result = ModelManifest::new(
		ModelRef::Hub(HubModelId::parse("owner/model").expect("valid Hub ID")),
		ModelSource::Hub,
		Some(ResolvedRevision::parse("a".repeat(40)).expect("valid revision")),
		files(),
		inconsistent,
		VerificationStatus::Estimated,
		None,
	);
	assert!(matches!(result, Err(ManifestError::SizingMismatch)));
}

#[test]
fn construction_rejects_invalid_generation_defaults() {
	for defaults in [
		ModelGenerationDefaults {
			temperature: Some(f32::NAN),
			..ModelGenerationDefaults::default()
		},
		ModelGenerationDefaults {
			top_p: Some(1.5),
			..ModelGenerationDefaults::default()
		},
		ModelGenerationDefaults {
			max_new_tokens: Some(0),
			..ModelGenerationDefaults::default()
		},
	] {
		let mut invalid = traits();
		invalid.generation_defaults = defaults;
		let result = ModelManifest::new(
			ModelRef::Hub(HubModelId::parse("owner/model").expect("valid Hub ID")),
			ModelSource::Hub,
			Some(ResolvedRevision::parse("a".repeat(40)).expect("valid revision")),
			files(),
			invalid,
			VerificationStatus::Estimated,
			None,
		);
		assert!(matches!(
			result,
			Err(ManifestError::InvalidGenerationDefaults(_))
		));
	}
}

#[test]
fn deserialization_rejects_invalid_generation_defaults() {
	let manifest = ModelManifest::new(
		ModelRef::Hub(HubModelId::parse("owner/model").expect("valid Hub ID")),
		ModelSource::Hub,
		Some(ResolvedRevision::parse("a".repeat(40)).expect("valid revision")),
		files(),
		traits(),
		VerificationStatus::Estimated,
		None,
	)
	.expect("valid manifest");
	let mut value = serde_json::to_value(manifest).expect("serialize manifest");
	value["traits"]["generation_defaults"]["temperature"] = serde_json::json!(3.0);

	assert!(serde_json::from_value::<ModelManifest>(value).is_err());
}
