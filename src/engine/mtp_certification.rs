//! Runtime binding for the checked-in MTP parity certificate.
//!
//! Layout validation proves that tensors can be loaded. This module separately
//! proves that the exact checkpoint bytes are the ones exercised by the
//! three-step reference parity gate.

use std::{
	collections::{BTreeMap, BTreeSet},
	ffi::OsStr,
	path::{Component, Path},
	sync::OnceLock,
};

use serde::Deserialize;

use super::error::{Error, Result};
use crate::model::layout::CheckpointSnapshot;

pub(crate) const IMPLEMENTATION_ID: &str = "emelex-qwen3.5-mtp-dense-bf16-v1";
const CERTIFICATION_SCHEMA: u32 = 2;
const REQUIRED_STEPS: usize = 3;
const CERTIFICATION_JSON: &str = include_str!("../../tests/fixtures/mtp_certification.json");
const EXPECTED_FILES: [&str; 3] = [
	"config.json",
	"model-00001-of-00002.safetensors",
	"model-00002-of-00002.safetensors",
];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Certification {
	schema_version: u32,
	implementation_id: String,
	required_steps: usize,
	model: CertifiedModel,
	reference: serde_json::Value,
	goldens: serde_json::Value,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CertifiedModel {
	source: serde_json::Value,
	equivalence_reference: serde_json::Value,
	files: Vec<CertifiedFile>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CertifiedFile {
	path: String,
	sha256: String,
}

fn certified_files() -> Result<&'static BTreeMap<String, String>> {
	static FILES: OnceLock<std::result::Result<BTreeMap<String, String>, String>> = OnceLock::new();
	let parsed = FILES.get_or_init(|| {
		let certification: Certification =
			serde_json::from_str(CERTIFICATION_JSON).map_err(|error| error.to_string())?;
		if certification.schema_version != CERTIFICATION_SCHEMA
			|| certification.implementation_id != IMPLEMENTATION_ID
			|| certification.required_steps != REQUIRED_STEPS
			|| !certification.reference.is_object()
			|| !certification.goldens.is_object()
			|| !certification.model.source.is_object()
			|| !certification.model.equivalence_reference.is_object()
		{
			return Err("checked-in MTP certification identity is invalid".to_string());
		}
		let mut files = BTreeMap::new();
		for file in certification.model.files {
			let path = Path::new(&file.path);
			let mut components = path.components();
			if !matches!(components.next(), Some(Component::Normal(_)))
				|| components.next().is_some()
				|| file.sha256.len() != 64
				|| !file
					.sha256
					.bytes()
					.all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
				|| files.insert(file.path, file.sha256).is_some()
			{
				return Err("checked-in MTP certification file plan is invalid".to_string());
			}
		}
		if files.len() != EXPECTED_FILES.len()
			|| !EXPECTED_FILES.iter().all(|path| files.contains_key(*path))
		{
			return Err("checked-in MTP certification model file set drifted".to_string());
		}
		Ok(files)
	});
	parsed
		.as_ref()
		.map_err(|message| Error::Model(format!("invalid embedded MTP certification: {message}")))
}

/// Whether the exact descriptor-backed snapshot loaded by the engine matches
/// the model bytes exercised by the checked-in parity certificate.
pub(crate) fn model_is_certified(snapshot: &CheckpointSnapshot) -> Result<bool> {
	let files = certified_files()?;
	// Config is tiny and uniquely identifies the certified layout. Check it
	// first so ordinary checkpoints never pay to hash multi-gigabyte shards.
	if files.get("config.json").map(String::as_str) != Some(snapshot.config_sha256()) {
		return Ok(false);
	}
	if !has_exact_certified_shards(snapshot) {
		return Ok(false);
	}
	for path in EXPECTED_FILES.into_iter().skip(1) {
		if files.get(path).map(String::as_str) != snapshot.shard_sha256(path) {
			return Ok(false);
		}
	}
	Ok(true)
}

fn has_exact_certified_shards(snapshot: &CheckpointSnapshot) -> bool {
	let expected_shards = EXPECTED_FILES
		.into_iter()
		.skip(1)
		.map(OsStr::new)
		.collect::<BTreeSet<_>>();
	snapshot.shard_names() == expected_shards
}

#[cfg(test)]
mod tests {
	#![allow(clippy::expect_used)]

	use std::io::Write as _;

	use super::*;

	fn write_shard(path: &Path, tensor: &str) {
		let header = serde_json::json!({
			tensor: {
				"dtype": "F32",
				"shape": [1],
				"data_offsets": [0, 4],
			}
		});
		let encoded = serde_json::to_vec(&header).expect("encode shard");
		let mut file = std::fs::File::create(path).expect("create shard");
		file.write_all(&(encoded.len() as u64).to_le_bytes())
			.expect("write header length");
		file.write_all(&encoded).expect("write header");
		file.write_all(&[0_u8; 4]).expect("write payload");
	}

	#[test]
	fn embedded_certification_is_strict_and_complete() {
		let files = certified_files().expect("certification");
		assert_eq!(files.len(), 3);
		assert!(EXPECTED_FILES.iter().all(|path| files.contains_key(*path)));
	}

	#[test]
	fn unrelated_config_fails_without_requiring_shards() {
		let directory = tempfile::tempdir().expect("tempdir");
		std::fs::write(directory.path().join("config.json"), b"{}").expect("write");
		let snapshot =
			CheckpointSnapshot::open(directory.path()).expect("descriptor-backed snapshot");
		assert!(!model_is_certified(&snapshot).expect("certificate result"));
	}

	#[test]
	fn indexed_extra_shard_fails_exact_certified_set() {
		let directory = tempfile::tempdir().expect("tempdir");
		std::fs::write(directory.path().join("config.json"), b"{}").expect("write config");
		for (name, tensor) in [
			("model-00001-of-00002.safetensors", "a"),
			("model-00002-of-00002.safetensors", "b"),
			("model-00003-of-00002.safetensors", "extra"),
		] {
			write_shard(&directory.path().join(name), tensor);
		}
		std::fs::write(
			directory.path().join("model.safetensors.index.json"),
			r#"{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors","extra":"model-00003-of-00002.safetensors"}}"#,
		)
		.expect("write index");

		let snapshot =
			CheckpointSnapshot::open(directory.path()).expect("descriptor-backed snapshot");
		assert!(!has_exact_certified_shards(&snapshot));
	}
}
