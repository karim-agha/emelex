//! Shared checkpoint path and safetensors-plan validation.

use std::{
	collections::{BTreeMap, BTreeSet},
	ffi::{CStr, CString, OsStr, OsString},
	fs::{self, File, OpenOptions},
	io::{Read as _, Seek as _, SeekFrom},
	os::{
		fd::{AsRawFd as _, FromRawFd as _, RawFd},
		unix::{
			ffi::{OsStrExt as _, OsStringExt as _},
			fs::{MetadataExt as _, OpenOptionsExt as _},
		},
	},
	path::{Component, Path, PathBuf},
};

use serde_json::Value;
use sha2::Digest as _;

use super::ModelFile;

const MAX_INDEX_BYTES: u64 = 64 << 20;
const MAX_SAFETENSORS_HEADER_BYTES: u64 = 64 << 20;
const MAX_AGGREGATE_HEADER_BYTES: u64 = 256 << 20;
const MAX_CHECKPOINT_SHARDS: usize = 1_024;
const MAX_CHECKPOINT_TENSORS: usize = 1_000_000;
const MAX_TENSOR_NAME_BYTES: usize = 4 << 10;
const MAX_TENSOR_RANK: usize = 32;

/// Immutable, preflighted set of checkpoint files the engine may load.
#[derive(Debug, Clone)]
pub struct CheckpointPlan {
	files: Vec<PathBuf>,
	snapshots: BTreeMap<PathBuf, PlannedShard>,
	index_content: Option<FileContent>,
	weights_bytes: u64,
	mtp_weights_present: bool,
	vision_weights_present: bool,
	audio_weights_present: bool,
}

impl CheckpointPlan {
	pub(crate) fn files(&self) -> &[PathBuf] {
		&self.files
	}

	pub(crate) const fn weights_bytes(&self) -> u64 {
		self.weights_bytes
	}

	pub(crate) const fn mtp_weights_present(&self) -> bool {
		self.mtp_weights_present
	}

	pub(crate) const fn vision_weights_present(&self) -> bool {
		self.vision_weights_present
	}

	pub(crate) const fn audio_weights_present(&self) -> bool {
		self.audio_weights_present
	}

	#[cfg(test)]
	fn validate_opened_shard(
		&self,
		path: &Path,
		file: &mut File,
	) -> Result<(), CheckpointLayoutError> {
		let expected = self
			.snapshots
			.get(path)
			.ok_or_else(|| invalid(path, "checkpoint shard is absent from immutable plan"))?;
		validate_opened_plan(path, file, expected)
	}
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PlannedShard {
	names: BTreeSet<String>,
	bytes: u64,
	header_bytes: u64,
	header_sha256: String,
	device: u64,
	inode: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CapturedShard {
	layout: PlannedShard,
	file_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FileContent {
	bytes: u64,
	sha256: String,
}

/// Owned identity used from checkpoint preflight through model loading and
/// optional MTP certification.
///
/// Emelex pins the source directory, copies bounded runtime metadata, and
/// atomically clones every selected shard into a private unlinked descriptor.
/// Each private clone is fully hashed once before manifest validation, loading,
/// or MTP certification. Later phases never reopen model-owned paths.
#[derive(Debug)]
pub struct CheckpointSnapshot {
	_directory: File,
	config_bytes: Vec<u8>,
	config_sha256: String,
	runtime_metadata: BTreeMap<&'static str, Vec<u8>>,
	shards: Vec<OpenedShard>,
}

impl CheckpointSnapshot {
	pub(crate) fn open_in(
		model_dir: &Path,
		temp_dir: &Path,
	) -> Result<Self, CheckpointLayoutError> {
		Self::open_inner(model_dir, temp_dir, None, || {}, || {})
	}

	pub(crate) fn open_verified_in(
		model_dir: &Path,
		temp_dir: &Path,
		expected_files: &[ModelFile],
	) -> Result<Self, CheckpointLayoutError> {
		Self::open_inner(model_dir, temp_dir, Some(expected_files), || {}, || {})
	}

	#[cfg(test)]
	pub(crate) fn open(model_dir: &Path) -> Result<Self, CheckpointLayoutError> {
		let temp_dir = model_dir
			.parent()
			.unwrap_or(model_dir)
			.join(".emelex-test-temp");
		fs::create_dir_all(&temp_dir).map_err(|error| {
			invalid(
				&temp_dir,
				format!("cannot create test checkpoint temp directory: {error}"),
			)
		})?;
		Self::open_in(model_dir, &temp_dir)
	}

	#[cfg(test)]
	fn open_with_after_config(
		model_dir: &Path,
		after_config: impl FnOnce(),
	) -> Result<Self, CheckpointLayoutError> {
		let temp_dir = model_dir
			.parent()
			.unwrap_or(model_dir)
			.join(".emelex-test-temp");
		fs::create_dir_all(&temp_dir).map_err(|error| {
			invalid(
				&temp_dir,
				format!("cannot create test checkpoint temp directory: {error}"),
			)
		})?;
		Self::open_inner(model_dir, &temp_dir, None, after_config, || {})
	}

	#[cfg(test)]
	fn open_verified_with_metadata_seam(
		model_dir: &Path,
		temp_dir: &Path,
		expected_files: &[ModelFile],
		before_metadata: impl FnOnce(),
		after_metadata: impl FnOnce(),
	) -> Result<Self, CheckpointLayoutError> {
		Self::open_inner(
			model_dir,
			temp_dir,
			Some(expected_files),
			before_metadata,
			after_metadata,
		)
	}

	fn open_inner(
		model_dir: &Path,
		temp_dir: &Path,
		expected_files: Option<&[ModelFile]>,
		after_config: impl FnOnce(),
		after_metadata: impl FnOnce(),
	) -> Result<Self, CheckpointLayoutError> {
		// emelex patch: pin the root directory before planning. Enumeration
		// and every consumed file are relative to this descriptor.
		let directory = open_directory_no_follow(model_dir)?;
		let plan = checkpoint_plan_from_directory(&directory, model_dir)?;
		let config_path = model_dir.join("config.json");
		let config_file = open_no_follow_at(&directory, OsStr::new("config.json"), &config_path)?;
		let config_bytes = read_stable_bounded_file(
			config_file,
			&config_path,
			crate::artifact::MAX_MODEL_CONFIG_BYTES,
		)?;
		let config_sha256 = hex::encode(sha2::Sha256::digest(&config_bytes));
		after_config();
		let runtime_metadata = capture_runtime_metadata(&directory, model_dir)?;
		after_metadata();
		let mut shards = Vec::with_capacity(plan.files.len());
		for path in &plan.files {
			let expected =
				plan.snapshots.get(path).cloned().ok_or_else(|| {
					invalid(path, "checkpoint shard is absent from immutable plan")
				})?;
			let name = path
				.file_name()
				.ok_or_else(|| invalid(path, "checkpoint shard has no filename"))?;
			let mut file = open_no_follow_at(&directory, name, path)?;
			validate_opened_plan(path, &mut file, &expected)?;
			let (file, expected) = private_checkpoint_copy(&mut file, path, &expected, temp_dir)?;
			shards.push(OpenedShard {
				path: path.clone(),
				file,
				expected,
			});
		}
		if let Some(expected_files) = expected_files {
			validate_expected_snapshot(
				&directory,
				model_dir,
				&plan,
				&config_bytes,
				&runtime_metadata,
				&shards,
				expected_files,
			)?;
		}
		Ok(Self {
			_directory: directory,
			config_bytes,
			config_sha256,
			runtime_metadata,
			shards,
		})
	}

	pub(crate) fn config_bytes(&self) -> &[u8] {
		&self.config_bytes
	}

	pub(crate) fn config_sha256(&self) -> &str {
		&self.config_sha256
	}

	pub(crate) fn runtime_metadata(&self, name: &str) -> Option<&[u8]> {
		self.runtime_metadata.get(name).map(Vec::as_slice)
	}

	pub(crate) fn shard_sha256(&self, name: &str) -> Option<&str> {
		self.shards
			.iter()
			.find(|shard| shard.path.file_name() == Some(OsStr::new(name)))
			.map(|shard| shard.expected.file_sha256.as_str())
	}

	pub(crate) fn shard_names(&self) -> BTreeSet<&OsStr> {
		self.shards
			.iter()
			.filter_map(|shard| shard.path.file_name())
			.collect()
	}

	pub(crate) const fn has_shards(&self) -> bool {
		!self.shards.is_empty()
	}

	pub(crate) fn shards_mut(&mut self) -> &mut [OpenedShard] {
		&mut self.shards
	}
}

#[derive(Debug)]
pub struct OpenedShard {
	path: PathBuf,
	file: File,
	expected: CapturedShard,
}

impl OpenedShard {
	pub(crate) fn path(&self) -> &Path {
		&self.path
	}

	pub(crate) fn descriptor(&self) -> RawFd {
		self.file.as_raw_fd()
	}

	pub(crate) fn validate(&mut self) -> Result<(), CheckpointLayoutError> {
		validate_captured_descriptor(&self.path, &mut self.file, &self.expected)
	}
}

/// Rejected checkpoint structure.
#[derive(Debug, thiserror::Error)]
#[error("invalid checkpoint layout {path:?}: {message}")]
pub struct CheckpointLayoutError {
	path: PathBuf,
	message: String,
}

impl CheckpointLayoutError {
	pub(crate) fn path(&self) -> &Path {
		&self.path
	}

	pub(crate) fn message(&self) -> &str {
		&self.message
	}
}

/// Build the sole file plan used by compatibility inspection and MLX loading.
pub fn checkpoint_plan(model_dir: &Path) -> Result<CheckpointPlan, CheckpointLayoutError> {
	let directory = open_directory_no_follow(model_dir)?;
	checkpoint_plan_from_directory(&directory, model_dir)
}

#[expect(
	clippy::too_many_lines,
	reason = "one linear checkpoint-plan validation keeps selection and descriptor snapshots atomic"
)]
fn checkpoint_plan_from_directory(
	directory: &File,
	model_dir: &Path,
) -> Result<CheckpointPlan, CheckpointLayoutError> {
	let mut headers = BTreeMap::<String, BTreeSet<String>>::new();
	let mut sizes = BTreeMap::<String, u64>::new();
	let mut snapshots = BTreeMap::<PathBuf, PlannedShard>::new();
	let mut tensor_owner = BTreeMap::<String, String>::new();
	let mut aggregate_header_bytes = 0_u64;
	let mut alternate_indexes = Vec::new();
	for name in directory_entry_names(directory, model_dir)? {
		let entry_path = model_dir.join(&name);
		if name.as_bytes().ends_with(b".safetensors.index.json")
			&& name != OsStr::new("model.safetensors.index.json")
		{
			alternate_indexes.push(entry_path.clone());
		}
		if !has_safetensors_suffix(&name) {
			continue;
		}
		if headers.len() == MAX_CHECKPOINT_SHARDS {
			return Err(invalid(
				model_dir,
				format!("checkpoint exceeds {MAX_CHECKPOINT_SHARDS} shards"),
			));
		}
		let name = name
			.into_string()
			.map_err(|_| invalid(&entry_path, "checkpoint filename is not UTF-8"))?;
		if !safe_single_component(&name) {
			return Err(invalid(&entry_path, "unsafe checkpoint filename"));
		}
		let mut file = open_no_follow_at(directory, OsStr::new(&name), &entry_path)?;
		let snapshot = inspect_open_safetensors_layout(&mut file, &entry_path)?;
		let names = snapshot.names.clone();
		let bytes = snapshot.bytes;
		let header_bytes = snapshot.header_bytes;
		aggregate_header_bytes = aggregate_header_bytes
			.checked_add(header_bytes)
			.ok_or_else(|| invalid(model_dir, "aggregate header byte count overflow"))?;
		if aggregate_header_bytes > MAX_AGGREGATE_HEADER_BYTES {
			return Err(invalid(
				model_dir,
				format!("aggregate safetensors headers exceed {MAX_AGGREGATE_HEADER_BYTES} bytes"),
			));
		}
		if tensor_owner.len().saturating_add(names.len()) > MAX_CHECKPOINT_TENSORS {
			return Err(invalid(
				model_dir,
				format!("checkpoint exceeds {MAX_CHECKPOINT_TENSORS} tensors"),
			));
		}
		for tensor in &names {
			if let Some(previous) = tensor_owner.insert(tensor.clone(), name.clone()) {
				return Err(invalid(
					&entry_path,
					format!("tensor {tensor:?} appears in both {previous:?} and {name:?}"),
				));
			}
		}
		headers.insert(name.clone(), names);
		sizes.insert(name, bytes);
		snapshots.insert(entry_path, snapshot);
	}
	if !alternate_indexes.is_empty() {
		alternate_indexes.sort();
		return Err(invalid(
			&alternate_indexes[0],
			"multiple or variant safetensors indexes make the runnable checkpoint ambiguous",
		));
	}

	let index_path = model_dir.join("model.safetensors.index.json");
	let mut index_content = None;
	let selected_shards =
		match open_no_follow_at_io(directory, OsStr::new("model.safetensors.index.json")) {
			Ok(index_file) => {
				let index_bytes =
					read_stable_bounded_file(index_file, &index_path, MAX_INDEX_BYTES)?;
				index_content = Some(content_of(&index_bytes));
				let index: Value = serde_json::from_slice(&index_bytes)
					.map_err(|error| invalid(&index_path, format!("invalid JSON: {error}")))?;
				let weight_map = index
					.get("weight_map")
					.and_then(Value::as_object)
					.ok_or_else(|| invalid(&index_path, "missing object weight_map"))?;
				if weight_map.is_empty() {
					return Err(invalid(&index_path, "weight_map is empty"));
				}
				if weight_map.len() > MAX_CHECKPOINT_TENSORS {
					return Err(invalid(
						&index_path,
						format!("weight_map exceeds {MAX_CHECKPOINT_TENSORS} tensors"),
					));
				}
				let mut selected = BTreeSet::new();
				let mut indexed_tensors = BTreeSet::new();
				for (tensor, shard) in weight_map {
					validate_tensor_name(&index_path, tensor)?;
					let shard = shard
						.as_str()
						.ok_or_else(|| invalid(&index_path, "weight_map values must be strings"))?;
					if !safe_single_component(shard) || !shard.ends_with(".safetensors") {
						return Err(invalid(&index_path, format!("unsafe shard path {shard:?}")));
					}
					let names = headers.get(shard).ok_or_else(|| {
						invalid(
							&index_path,
							format!("index references missing shard {shard:?}"),
						)
					})?;
					if !names.contains(tensor) {
						return Err(invalid(
							&index_path,
							format!("tensor {tensor:?} is absent from shard {shard:?}"),
						));
					}
					selected.insert(shard.to_string());
					indexed_tensors.insert(tensor.clone());
				}
				for shard in &selected {
					let names = headers.get(shard).ok_or_else(|| {
						invalid(
							&index_path,
							format!("index references missing shard {shard:?}"),
						)
					})?;
					for tensor in names {
						if !indexed_tensors.contains(tensor) {
							return Err(invalid(
								&index_path,
								format!(
									"selected shard {shard:?} contains unindexed tensor {tensor:?}"
								),
							));
						}
					}
				}
				let unselected = headers
					.keys()
					.filter(|name| !selected.contains(*name))
					.cloned()
					.collect::<Vec<_>>();
				if let Some(name) = unselected.first() {
					return Err(invalid(
						&model_dir.join(name),
						"unindexed safetensors file makes the runnable checkpoint ambiguous",
					));
				}
				selected
			}
			Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
				if headers.len() == 1 && headers.contains_key("model.safetensors") {
					BTreeSet::from(["model.safetensors".to_string()])
				} else if headers.is_empty() {
					BTreeSet::new()
				} else {
					return Err(invalid(
						model_dir,
						"checkpoint without an index must contain only model.safetensors",
					));
				}
			}
			Err(error) => {
				return Err(invalid(
					&index_path,
					format!("cannot inspect checkpoint index: {error}"),
				));
			}
		};

	let runnable = !selected_shards.is_empty();
	let files = if runnable {
		selected_shards
			.iter()
			.map(|name| model_dir.join(name))
			.collect::<Vec<_>>()
	} else {
		Vec::new()
	};
	let weights_bytes = if runnable {
		selected_shards.iter().try_fold(0_u64, |total, name| {
			let bytes = sizes
				.get(name)
				.ok_or_else(|| invalid(model_dir, format!("missing shard size for {name:?}")))?;
			total
				.checked_add(*bytes)
				.ok_or_else(|| invalid(model_dir, "weight byte count overflow"))
		})?
	} else {
		0
	};
	let mtp_weights_present = runnable
		&& tensor_owner
			.keys()
			.any(|name| name.starts_with("language_model.mtp."));
	let vision_weights_present = runnable
		&& [
			"vision_tower.patch_embed.proj.weight",
			"vision_tower.patch_embedder.input_proj.weight",
			"vision_tower.vision_model.embeddings.patch_embedding.weight",
			"vision_embedder.patch_dense.weight",
		]
		.iter()
		.any(|name| tensor_owner.contains_key(*name));
	let audio_weights_present = runnable
		&& [
			"audio_tower.subsample_conv_projection.input_proj_linear.weight",
			"embed_audio.embedding_projection.weight",
		]
		.iter()
		.any(|name| tensor_owner.contains_key(*name));
	snapshots.retain(|path, _| {
		path.file_name()
			.and_then(OsStr::to_str)
			.is_some_and(|name| selected_shards.contains(name))
	});
	Ok(CheckpointPlan {
		files,
		snapshots,
		index_content,
		weights_bytes,
		mtp_weights_present,
		vision_weights_present,
		audio_weights_present,
	})
}

/// Whether a model-owned path is non-empty, relative, and traversal-free.
pub fn safe_relative_path(value: &str) -> bool {
	if value.is_empty() || value.contains('\\') {
		return false;
	}
	let path = Path::new(value);
	!path.is_absolute()
		&& path
			.components()
			.all(|component| matches!(component, Component::Normal(_)))
}

fn safe_single_component(value: &str) -> bool {
	safe_relative_path(value) && Path::new(value).components().count() == 1
}

fn has_safetensors_suffix(name: &OsStr) -> bool {
	name.as_bytes().ends_with(b".safetensors")
}

#[allow(
	clippy::too_many_lines,
	reason = "linear safetensors validation keeps all descriptor and range checks together"
)]
fn inspect_open_safetensors_layout(
	file: &mut File,
	path: &Path,
) -> Result<PlannedShard, CheckpointLayoutError> {
	file.seek(SeekFrom::Start(0))
		.map_err(|error| invalid(path, format!("cannot rewind checkpoint shard: {error}")))?;
	let metadata = file
		.metadata()
		.map_err(|error| invalid(path, format!("cannot inspect shard: {error}")))?;
	if !metadata.file_type().is_file() {
		return Err(invalid(path, "checkpoint shard is not a regular file"));
	}
	let file_len = metadata.len();
	let mut length = [0_u8; 8];
	file.read_exact(&mut length)
		.map_err(|error| invalid(path, format!("cannot read header length: {error}")))?;
	let header_len = u64::from_le_bytes(length);
	if header_len == 0
		|| header_len > MAX_SAFETENSORS_HEADER_BYTES
		|| header_len > file_len.saturating_sub(8)
	{
		return Err(invalid(path, "invalid safetensors header length"));
	}
	let header_len = usize::try_from(header_len)
		.map_err(|_| invalid(path, "safetensors header is too large"))?;
	let mut header = vec![0_u8; header_len];
	file.read_exact(&mut header)
		.map_err(|error| invalid(path, format!("cannot read safetensors header: {error}")))?;
	let header_sha256 = hex::encode(sha2::Sha256::digest(&header));
	let value: Value = serde_json::from_slice(&header)
		.map_err(|error| invalid(path, format!("invalid safetensors header JSON: {error}")))?;
	let object = value
		.as_object()
		.ok_or_else(|| invalid(path, "safetensors header must be an object"))?;
	if object.len() > MAX_CHECKPOINT_TENSORS.saturating_add(1) {
		return Err(invalid(
			path,
			format!("safetensors header exceeds {MAX_CHECKPOINT_TENSORS} tensors"),
		));
	}
	let payload_len = file_len - 8 - u64::try_from(header_len).unwrap_or(u64::MAX);
	let mut names = BTreeSet::new();
	let mut ranges = Vec::new();
	for (name, descriptor) in object {
		if name == "__metadata__" {
			let metadata = descriptor
				.as_object()
				.ok_or_else(|| invalid(path, "__metadata__ must be an object"))?;
			if metadata.values().any(|value| !value.is_string()) {
				return Err(invalid(path, "__metadata__ values must all be strings"));
			}
			continue;
		}
		validate_tensor_name(path, name)?;
		let descriptor = descriptor
			.as_object()
			.ok_or_else(|| invalid(path, format!("tensor {name:?} descriptor is not an object")))?;
		let offsets = descriptor
			.get("data_offsets")
			.and_then(Value::as_array)
			.ok_or_else(|| invalid(path, format!("tensor {name:?} has no data_offsets")))?;
		if offsets.len() != 2 {
			return Err(invalid(
				path,
				format!("tensor {name:?} has invalid offsets"),
			));
		}
		let start = offsets.first().and_then(Value::as_u64);
		let end = offsets.get(1).and_then(Value::as_u64);
		let Some((start, end)) = start.zip(end) else {
			return Err(invalid(
				path,
				format!("tensor {name:?} has invalid offsets"),
			));
		};
		if start > end || end > payload_len {
			return Err(invalid(
				path,
				format!("tensor {name:?} has invalid offsets"),
			));
		}
		let dtype = descriptor
			.get("dtype")
			.and_then(Value::as_str)
			.ok_or_else(|| invalid(path, format!("tensor {name:?} lacks dtype")))?;
		let width = tensor_byte_width(dtype).ok_or_else(|| {
			invalid(
				path,
				format!("tensor {name:?} has unsupported dtype {dtype:?}"),
			)
		})?;
		let shape = descriptor
			.get("shape")
			.and_then(Value::as_array)
			.ok_or_else(|| invalid(path, format!("tensor {name:?} lacks shape")))?;
		if shape.len() > MAX_TENSOR_RANK {
			return Err(invalid(
				path,
				format!("tensor {name:?} rank exceeds {MAX_TENSOR_RANK}"),
			));
		}
		let elements = shape.iter().try_fold(1_u64, |product, dimension| {
			let dimension = dimension.as_u64().ok_or_else(|| {
				invalid(
					path,
					format!("tensor {name:?} shape contains a non-integer dimension"),
				)
			})?;
			if dimension > i32::MAX as u64 {
				return Err(invalid(
					path,
					format!("tensor {name:?} dimension exceeds MLX i32 shape range"),
				));
			}
			product
				.checked_mul(dimension)
				.ok_or_else(|| invalid(path, format!("tensor {name:?} element count overflows")))
		})?;
		let expected_bytes = elements
			.checked_mul(width)
			.ok_or_else(|| invalid(path, format!("tensor {name:?} byte count overflows")))?;
		if end - start != expected_bytes {
			return Err(invalid(
				path,
				format!(
					"tensor {name:?} payload has {} bytes but dtype/shape require {expected_bytes}",
					end - start
				),
			));
		}
		names.insert(name.clone());
		ranges.push((start, end, name));
	}
	if names.is_empty() {
		return Err(invalid(path, "safetensors file contains no tensors"));
	}
	ranges.sort_by_key(|(start, _, _)| *start);
	for pair in ranges.windows(2) {
		let (_, previous_end, previous_name) = pair[0];
		let (next_start, _, next_name) = pair[1];
		if previous_end > next_start {
			return Err(invalid(
				path,
				format!("tensor payloads {previous_name:?} and {next_name:?} overlap"),
			));
		}
	}
	Ok(PlannedShard {
		names,
		bytes: file_len,
		header_bytes: u64::try_from(header_len).unwrap_or(u64::MAX),
		header_sha256,
		device: metadata.dev(),
		inode: metadata.ino(),
	})
}

fn validate_opened_plan(
	path: &Path,
	file: &mut File,
	expected: &PlannedShard,
) -> Result<(), CheckpointLayoutError> {
	let actual = inspect_open_safetensors_layout(file, path)?;
	file.seek(SeekFrom::Start(0))
		.map_err(|error| invalid(path, format!("cannot rewind checkpoint shard: {error}")))?;
	if &actual != expected {
		return Err(invalid(
			path,
			"checkpoint shard changed after immutable plan was created",
		));
	}
	Ok(())
}

fn validate_captured_descriptor(
	path: &Path,
	file: &mut File,
	expected: &CapturedShard,
) -> Result<(), CheckpointLayoutError> {
	let actual = inspect_open_safetensors_layout(file, path)?;
	file.seek(SeekFrom::Start(0))
		.map_err(|error| invalid(path, format!("cannot rewind checkpoint shard: {error}")))?;
	if actual != expected.layout {
		return Err(invalid(
			path,
			"private checkpoint descriptor changed after capture",
		));
	}
	Ok(())
}

fn read_bounded_file(
	mut file: File,
	path: &Path,
	limit: u64,
) -> Result<Vec<u8>, CheckpointLayoutError> {
	file.seek(SeekFrom::Start(0))
		.map_err(|error| invalid(path, format!("cannot rewind file: {error}")))?;
	let metadata = file
		.metadata()
		.map_err(|error| invalid(path, format!("cannot inspect file: {error}")))?;
	if !metadata.file_type().is_file() || metadata.len() > limit {
		return Err(invalid(path, format!("file exceeds {limit} bytes")));
	}
	let capacity = usize::try_from(metadata.len())
		.map_err(|_| invalid(path, "file is too large for this process"))?;
	let mut bytes = Vec::with_capacity(capacity);
	std::io::Read::by_ref(&mut file)
		.take(limit.saturating_add(1))
		.read_to_end(&mut bytes)
		.map_err(|error| invalid(path, format!("cannot read file: {error}")))?;
	if bytes.len() as u64 > limit {
		return Err(invalid(path, format!("file exceeds {limit} bytes")));
	}
	Ok(bytes)
}

fn read_stable_bounded_file(
	file: File,
	path: &Path,
	limit: u64,
) -> Result<Vec<u8>, CheckpointLayoutError> {
	let duplicate = file
		.try_clone()
		.map_err(|error| invalid(path, format!("cannot duplicate file descriptor: {error}")))?;
	let first = read_bounded_file(duplicate, path, limit)?;
	let second = read_bounded_file(file, path, limit)?;
	if first != second {
		return Err(invalid(
			path,
			"runtime metadata changed while being captured",
		));
	}
	Ok(first)
}

fn capture_runtime_metadata(
	directory: &File,
	model_dir: &Path,
) -> Result<BTreeMap<&'static str, Vec<u8>>, CheckpointLayoutError> {
	let mut captured = BTreeMap::new();
	for name in [
		"tokenizer.json",
		"tokenizer_config.json",
		"processor_config.json",
		"chat_template.json",
		"chat_template.jinja",
		"chat_template_tool_use.jinja",
		"generation_config.json",
	] {
		let limit = captured_runtime_metadata_limit(name).unwrap_or(0);
		let path = model_dir.join(name);
		match open_no_follow_at_io(directory, OsStr::new(name)) {
			Ok(file) => {
				captured.insert(name, read_stable_bounded_file(file, &path, limit)?);
			}
			Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
			Err(error) => {
				return Err(invalid(
					&path,
					format!("cannot capture runtime metadata: {error}"),
				));
			}
		}
	}
	let mut named_defaults = Vec::new();
	let mut named_tools = Vec::new();
	for directory_name in [
		crate::engine::tokenizer::CURRENT_CHAT_TEMPLATE_DIR,
		crate::engine::tokenizer::LEGACY_CHAT_TEMPLATE_DIR,
	] {
		if let Some(bytes) =
			capture_nested_runtime_metadata(directory, model_dir, directory_name, "default.jinja")?
		{
			named_defaults.push(bytes);
		}
		if let Some(bytes) =
			capture_nested_runtime_metadata(directory, model_dir, directory_name, "tool_use.jinja")?
		{
			named_tools.push(bytes);
		}
	}
	if captured.contains_key(crate::engine::tokenizer::LEGACY_CHAT_TEMPLATE_FILE)
		&& (!named_defaults.is_empty() || !named_tools.is_empty())
	{
		return Err(invalid(
			model_dir,
			"chat_template.json conflicts with named chat template files",
		));
	}
	capture_normalized_template(
		&mut captured,
		model_dir,
		"chat_template.jinja",
		named_defaults,
		"default",
	)?;
	capture_normalized_template(
		&mut captured,
		model_dir,
		"chat_template_tool_use.jinja",
		named_tools,
		"tool-use",
	)?;
	Ok(captured)
}

fn capture_nested_runtime_metadata(
	directory: &File,
	model_dir: &Path,
	directory_name: &str,
	file_name: &str,
) -> Result<Option<Vec<u8>>, CheckpointLayoutError> {
	let directory_path = model_dir.join(directory_name);
	let nested = match open_no_follow_at_io(directory, OsStr::new(directory_name)) {
		Ok(file) => file,
		Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
		Err(error) => {
			return Err(invalid(
				&directory_path,
				format!("cannot capture named chat templates: {error}"),
			));
		}
	};
	let metadata = nested.metadata().map_err(|error| {
		invalid(
			&directory_path,
			format!("cannot inspect directory: {error}"),
		)
	})?;
	if !metadata.file_type().is_dir() {
		return Err(invalid(
			&directory_path,
			"named chat template path is not a directory",
		));
	}
	let path = directory_path.join(file_name);
	let file = match open_no_follow_at_io(&nested, OsStr::new(file_name)) {
		Ok(file) => file,
		Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
		Err(error) => {
			return Err(invalid(
				&path,
				format!("cannot capture named chat template: {error}"),
			));
		}
	};
	read_stable_bounded_file(file, &path, crate::artifact::MAX_CHAT_TEMPLATE_BYTES).map(Some)
}

fn capture_normalized_template(
	captured: &mut BTreeMap<&'static str, Vec<u8>>,
	model_dir: &Path,
	target: &'static str,
	mut named: Vec<Vec<u8>>,
	label: &str,
) -> Result<(), CheckpointLayoutError> {
	if captured.contains_key(target) && !named.is_empty() {
		return Err(invalid(
			model_dir,
			format!("root and named {label} chat templates map to the same runtime file"),
		));
	}
	match named.len() {
		0 => Ok(()),
		1 => {
			captured.insert(target, named.pop().unwrap_or_default());
			Ok(())
		}
		_ => Err(invalid(
			model_dir,
			format!("multiple named {label} chat templates are present"),
		)),
	}
}

const fn captured_runtime_metadata_limit(name: &str) -> Option<u64> {
	match name.as_bytes() {
		b"tokenizer.json" => Some(crate::artifact::MAX_TOKENIZER_BYTES),
		b"tokenizer_config.json" | b"processor_config.json" => {
			Some(crate::artifact::MAX_TOKENIZER_CONFIG_BYTES)
		}
		b"chat_template.jinja" | b"chat_template_tool_use.jinja" => {
			Some(crate::artifact::MAX_CHAT_TEMPLATE_BYTES)
		}
		b"chat_template.json" => Some(crate::artifact::MAX_TOKENIZER_CONFIG_BYTES),
		b"generation_config.json" => Some(crate::artifact::MAX_MODEL_CONFIG_BYTES),
		_ => None,
	}
}

fn validate_expected_snapshot(
	directory: &File,
	model_dir: &Path,
	plan: &CheckpointPlan,
	config_bytes: &[u8],
	runtime_metadata: &BTreeMap<&'static str, Vec<u8>>,
	shards: &[OpenedShard],
	expected_files: &[ModelFile],
) -> Result<(), CheckpointLayoutError> {
	let mut expected_by_name = BTreeMap::new();
	for expected in expected_files {
		if !safe_single_component(expected.path()) {
			return Err(invalid(
				&model_dir.join(expected.path()),
				"installed manifest file must be a root-level runtime file",
			));
		}
		if expected_by_name.insert(expected.path(), expected).is_some() {
			return Err(invalid(
				&model_dir.join(expected.path()),
				"installed manifest contains a duplicate runtime file",
			));
		}
	}

	let mut required = BTreeSet::from(["config.json"]);
	required.extend(runtime_metadata.keys().copied());
	required.extend(
		plan.files
			.iter()
			.filter_map(|path| path.file_name().and_then(OsStr::to_str)),
	);
	if plan.index_content.is_some() {
		required.insert("model.safetensors.index.json");
	}
	for name in required {
		if !expected_by_name.contains_key(name) {
			return Err(invalid(
				&model_dir.join(name),
				"runtime-influential file is absent from the installed manifest",
			));
		}
	}

	for expected in expected_files {
		let path = model_dir.join(expected.path());
		let actual = if expected.path() == "config.json" {
			content_of(config_bytes)
		} else if let Some(bytes) = runtime_metadata.get(expected.path()) {
			content_of(bytes)
		} else if captured_runtime_metadata_limit(expected.path()).is_some() {
			return Err(invalid(
				&path,
				"installed manifest runtime metadata was absent from the captured snapshot",
			));
		} else if expected.path() == "model.safetensors.index.json" {
			plan.index_content.clone().ok_or_else(|| {
				invalid(
					&path,
					"installed manifest records an absent checkpoint index",
				)
			})?
		} else if expected.path().ends_with(".safetensors") {
			let shard = shards
				.iter()
				.find(|shard| shard.path.file_name() == Some(OsStr::new(expected.path())))
				.ok_or_else(|| {
					invalid(
						&path,
						"installed manifest shard is absent from the captured checkpoint",
					)
				})?;
			FileContent {
				bytes: shard.expected.layout.bytes,
				sha256: shard.expected.file_sha256.clone(),
			}
		} else {
			let file = open_no_follow_at(directory, OsStr::new(expected.path()), &path)?;
			hash_stable_file(file, &path)?
		};
		validate_expected_content(&path, &actual, expected)?;
	}
	Ok(())
}

fn content_of(bytes: &[u8]) -> FileContent {
	FileContent {
		bytes: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
		sha256: hex::encode(sha2::Sha256::digest(bytes)),
	}
}

fn validate_expected_content(
	path: &Path,
	actual: &FileContent,
	expected: &ModelFile,
) -> Result<(), CheckpointLayoutError> {
	if actual.bytes != expected.size() || actual.sha256 != expected.sha256() {
		return Err(invalid(
			path,
			format!(
				"captured file differs from installed manifest: expected {} bytes / {}, got {} bytes / {}",
				expected.size(),
				expected.sha256(),
				actual.bytes,
				actual.sha256
			),
		));
	}
	Ok(())
}

fn hash_stable_file(file: File, path: &Path) -> Result<FileContent, CheckpointLayoutError> {
	let duplicate = file
		.try_clone()
		.map_err(|error| invalid(path, format!("cannot duplicate file descriptor: {error}")))?;
	let first = hash_open_file(duplicate, path)?;
	let second = hash_open_file(file, path)?;
	if first != second {
		return Err(invalid(
			path,
			"installed runtime file changed while its manifest identity was captured",
		));
	}
	Ok(first)
}

fn hash_open_file(mut file: File, path: &Path) -> Result<FileContent, CheckpointLayoutError> {
	let before = file
		.metadata()
		.map_err(|error| invalid(path, format!("cannot inspect runtime file: {error}")))?;
	if !before.file_type().is_file() {
		return Err(invalid(path, "installed runtime file is not regular"));
	}
	file.seek(SeekFrom::Start(0))
		.map_err(|error| invalid(path, format!("cannot rewind runtime file: {error}")))?;
	let mut digest = sha2::Sha256::new();
	let mut bytes = 0_u64;
	let mut buffer = vec![0_u8; 1 << 20];
	loop {
		let read = file
			.read(&mut buffer)
			.map_err(|error| invalid(path, format!("cannot hash runtime file: {error}")))?;
		if read == 0 {
			break;
		}
		digest.update(&buffer[..read]);
		bytes = bytes
			.checked_add(u64::try_from(read).unwrap_or(u64::MAX))
			.ok_or_else(|| invalid(path, "runtime file byte count overflow"))?;
	}
	let after = file
		.metadata()
		.map_err(|error| invalid(path, format!("cannot reinspect runtime file: {error}")))?;
	if !same_file_metadata(&before, &after) || bytes != after.len() {
		return Err(invalid(
			path,
			"installed runtime file changed while being hashed",
		));
	}
	Ok(FileContent {
		bytes,
		sha256: hex::encode(digest.finalize()),
	})
}

fn same_file_metadata(left: &fs::Metadata, right: &fs::Metadata) -> bool {
	left.dev() == right.dev()
		&& left.ino() == right.ino()
		&& left.len() == right.len()
		&& left.mtime() == right.mtime()
		&& left.mtime_nsec() == right.mtime_nsec()
		&& left.ctime() == right.ctime()
		&& left.ctime_nsec() == right.ctime_nsec()
}

fn validate_tensor_name(path: &Path, name: &str) -> Result<(), CheckpointLayoutError> {
	if name.is_empty() || name.len() > MAX_TENSOR_NAME_BYTES || name.contains('\0') {
		return Err(invalid(
			path,
			format!("tensor name must contain 1..={MAX_TENSOR_NAME_BYTES} bytes and no NUL"),
		));
	}
	Ok(())
}

const fn tensor_byte_width(dtype: &str) -> Option<u64> {
	match dtype.as_bytes() {
		b"BOOL" | b"U8" | b"I8" | b"F8_E4M3" => Some(1),
		b"U16" | b"I16" | b"F16" | b"BF16" => Some(2),
		b"U32" | b"I32" | b"F32" => Some(4),
		b"U64" | b"I64" | b"C64" => Some(8),
		_ => None,
	}
}

fn open_no_follow(path: &Path) -> Result<File, CheckpointLayoutError> {
	OpenOptions::new()
		.read(true)
		.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK)
		.open(path)
		.map_err(|error| {
			invalid(
				path,
				format!("cannot open without following symlinks: {error}"),
			)
		})
}

fn open_directory_no_follow(path: &Path) -> Result<File, CheckpointLayoutError> {
	OpenOptions::new()
		.read(true)
		.custom_flags(libc::O_CLOEXEC | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_NONBLOCK)
		.open(path)
		.map_err(|error| {
			invalid(
				path,
				format!("cannot open model directory without following symlinks: {error}"),
			)
		})
}

fn open_no_follow_at(
	directory: &File,
	name: &OsStr,
	path: &Path,
) -> Result<File, CheckpointLayoutError> {
	open_no_follow_at_io(directory, name).map_err(|error| {
		invalid(
			path,
			format!("cannot open relative to immutable model directory: {error}"),
		)
	})
}

fn open_no_follow_at_io(directory: &File, name: &OsStr) -> std::io::Result<File> {
	let name = CString::new(name.as_bytes()).map_err(|_| {
		std::io::Error::new(
			std::io::ErrorKind::InvalidInput,
			"model filename contains NUL",
		)
	})?;
	// SAFETY: `directory` stays open, `name` is NUL-terminated, and the
	// returned descriptor is immediately transferred into `File` ownership.
	let descriptor = unsafe {
		libc::openat(
			directory.as_raw_fd(),
			name.as_ptr(),
			libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK,
		)
	};
	if descriptor < 0 {
		return Err(std::io::Error::last_os_error());
	}
	// SAFETY: `openat` returned a fresh owned descriptor.
	Ok(unsafe { File::from_raw_fd(descriptor) })
}

struct DirectoryStream(*mut libc::DIR);

impl Drop for DirectoryStream {
	fn drop(&mut self) {
		// SAFETY: `fdopendir` returned this owned, non-null stream and no
		// other call closes it.
		unsafe {
			let _ = libc::closedir(self.0);
		}
	}
}

fn directory_entry_names(
	directory: &File,
	path: &Path,
) -> Result<Vec<OsString>, CheckpointLayoutError> {
	// SAFETY: `directory` is a live descriptor. `F_DUPFD_CLOEXEC` returns
	// one new owned descriptor or a negative error indicator.
	let duplicate = unsafe { libc::fcntl(directory.as_raw_fd(), libc::F_DUPFD_CLOEXEC, 0) };
	if duplicate < 0 {
		return Err(invalid(
			path,
			format!(
				"cannot duplicate model directory descriptor: {}",
				std::io::Error::last_os_error()
			),
		));
	}
	// SAFETY: `duplicate` is a fresh directory descriptor. On success,
	// `fdopendir` assumes ownership; on failure, this function closes it.
	let raw_stream = unsafe { libc::fdopendir(duplicate) };
	if raw_stream.is_null() {
		let error = std::io::Error::last_os_error();
		// SAFETY: `fdopendir` failed and did not assume ownership.
		unsafe {
			let _ = libc::close(duplicate);
		}
		return Err(invalid(
			path,
			format!("cannot enumerate model directory descriptor: {error}"),
		));
	}
	let stream = DirectoryStream(raw_stream);
	let mut names = Vec::new();
	loop {
		// SAFETY: Apple exposes the calling thread's errno cell through
		// `__error`; clearing it distinguishes readdir EOF from failure.
		unsafe {
			*libc::__error() = 0;
		}
		// SAFETY: `stream` remains live and is used on this thread only.
		let entry = unsafe { libc::readdir(stream.0) };
		if entry.is_null() {
			// SAFETY: same thread-local errno cell set by `readdir`.
			let errno = unsafe { *libc::__error() };
			if errno == 0 {
				break;
			}
			return Err(invalid(
				path,
				format!(
					"cannot read model directory entry: {}",
					std::io::Error::from_raw_os_error(errno)
				),
			));
		}
		// SAFETY: `readdir` returned a live `dirent` whose d_name field is
		// NUL-terminated until the next call on this stream.
		let bytes = unsafe { CStr::from_ptr((*entry).d_name.as_ptr()) }.to_bytes();
		if bytes != b"." && bytes != b".." {
			names.push(OsString::from_vec(bytes.to_vec()));
		}
	}
	names.sort();
	Ok(names)
}

fn private_checkpoint_copy(
	source: &mut File,
	path: &Path,
	expected: &PlannedShard,
	temp_dir: &Path,
) -> Result<(File, CapturedShard), CheckpointLayoutError> {
	let temporary = tempfile::Builder::new()
		.prefix("emelex-checkpoint-")
		.tempdir_in(temp_dir)
		.map_err(|error| {
			invalid(
				path,
				format!(
					"cannot create private checkpoint area under {}: {error}",
					temp_dir.display()
				),
			)
		})?;
	let private_path = temporary.path().join("shard.safetensors");
	let destination = CString::new(private_path.as_os_str().as_bytes())
		.map_err(|_| invalid(path, "private checkpoint path contains NUL"))?;
	// SAFETY: `source` is a live regular-file descriptor and `destination`
	// is a NUL-terminated absent path inside a mode-0700 temporary directory.
	let clone_status =
		unsafe { libc::fclonefileat(source.as_raw_fd(), libc::AT_FDCWD, destination.as_ptr(), 0) };
	if clone_status != 0 {
		return Err(invalid(
			path,
			format!(
				"cannot create a private copy-on-write checkpoint clone under {}: {}; \
				 import the model into Emelex home or place EMELEX_HOME on the same \
				 clone-capable APFS volume",
				temp_dir.display(),
				std::io::Error::last_os_error()
			),
		));
	}
	let mut private = open_no_follow(&private_path)?;
	fs::remove_file(&private_path)
		.map_err(|error| invalid(path, format!("cannot unlink cloned checkpoint: {error}")))?;

	let private_layout = inspect_open_safetensors_layout(&mut private, path)?;
	if !same_checkpoint_layout(&private_layout, expected) {
		return Err(invalid(
			path,
			"private checkpoint layout differs from the source plan",
		));
	}
	let file_sha256 = hash_open_file(
		private
			.try_clone()
			.map_err(|error| invalid(path, format!("cannot duplicate private shard: {error}")))?,
		path,
	)?
	.sha256;
	private
		.seek(SeekFrom::Start(0))
		.map_err(|error| invalid(path, format!("cannot rewind private checkpoint: {error}")))?;
	Ok((
		private,
		CapturedShard {
			layout: private_layout,
			file_sha256,
		},
	))
}

fn same_checkpoint_layout(left: &PlannedShard, right: &PlannedShard) -> bool {
	left.names == right.names
		&& left.bytes == right.bytes
		&& left.header_bytes == right.header_bytes
		&& left.header_sha256 == right.header_sha256
}

fn invalid(path: &Path, message: impl Into<String>) -> CheckpointLayoutError {
	CheckpointLayoutError {
		path: path.to_path_buf(),
		message: message.into(),
	}
}

#[cfg(test)]
mod tests {
	use std::io::Write as _;

	use super::*;

	fn write_shard(path: &Path, tensors: &[&str]) {
		let mut offset = 0_u64;
		let mut header = serde_json::Map::new();
		for tensor in tensors {
			header.insert(
				(*tensor).to_string(),
				serde_json::json!({
					"dtype": "F32",
					"shape": [1],
					"data_offsets": [offset, offset + 4],
				}),
			);
			offset += 4;
		}
		let encoded = serde_json::to_vec(&Value::Object(header)).unwrap();
		let mut file = File::create(path).unwrap();
		file.write_all(&(encoded.len() as u64).to_le_bytes())
			.unwrap();
		file.write_all(&encoded).unwrap();
		file.write_all(&vec![0_u8; offset as usize]).unwrap();
	}

	fn manifest_files(path: &Path) -> Vec<ModelFile> {
		let mut names = fs::read_dir(path)
			.unwrap()
			.map(|entry| entry.unwrap().file_name().into_string().unwrap())
			.collect::<Vec<_>>();
		names.sort();
		names
			.into_iter()
			.map(|name| {
				let bytes = fs::read(path.join(&name)).unwrap();
				ModelFile::new(
					name,
					u64::try_from(bytes.len()).unwrap(),
					hex::encode(sha2::Sha256::digest(bytes)),
				)
				.unwrap()
			})
			.collect()
	}

	fn write_snapshot_fixture(path: &Path) {
		fs::write(path.join("config.json"), br#"{"identity":"a"}"#).unwrap();
		fs::write(
			path.join("tokenizer.json"),
			br#"{"identity":"tokenizer-a"}"#,
		)
		.unwrap();
		write_shard(&path.join("model.safetensors"), &["a"]);
	}

	#[test]
	fn rejects_index_traversal() {
		let directory = tempfile::tempdir().unwrap();
		write_shard(&directory.path().join("model.safetensors"), &["x"]);
		fs::write(
			directory.path().join("model.safetensors.index.json"),
			r#"{"weight_map":{"x":"../model.safetensors"}}"#,
		)
		.unwrap();
		let error = checkpoint_plan(directory.path()).unwrap_err();
		assert!(error.message().contains("unsafe shard path"));
	}

	#[test]
	fn rejects_duplicate_tensor_ownership() {
		let directory = tempfile::tempdir().unwrap();
		write_shard(&directory.path().join("model.safetensors"), &["x"]);
		write_shard(&directory.path().join("vision.safetensors"), &["x"]);
		let error = checkpoint_plan(directory.path()).unwrap_err();
		assert!(error.message().contains("appears in both"));
	}

	#[test]
	fn rejects_unindexed_sidecar_as_ambiguous() {
		let directory = tempfile::tempdir().unwrap();
		write_shard(&directory.path().join("model.safetensors"), &["text"]);
		write_shard(&directory.path().join("vision.safetensors"), &["vision"]);
		let error = checkpoint_plan(directory.path()).unwrap_err();
		assert!(error.message().contains("only model.safetensors"));
	}

	#[test]
	fn index_selects_exact_shards_and_rejects_extra_weights() {
		let directory = tempfile::tempdir().unwrap();
		write_shard(&directory.path().join("model-00001.safetensors"), &["x"]);
		write_shard(&directory.path().join("adapter.safetensors"), &["adapter"]);
		fs::write(
			directory.path().join("model.safetensors.index.json"),
			r#"{"weight_map":{"x":"model-00001.safetensors"}}"#,
		)
		.unwrap();
		let error = checkpoint_plan(directory.path()).unwrap_err();
		assert!(error.message().contains("unindexed safetensors"));
	}

	#[test]
	fn index_selects_complete_exact_shard_set() {
		let directory = tempfile::tempdir().unwrap();
		write_shard(
			&directory.path().join("model-00001.safetensors"),
			&["layer.0", "layer.1"],
		);
		write_shard(
			&directory.path().join("model-00002.safetensors"),
			&["layer.2"],
		);
		fs::write(
			directory.path().join("model.safetensors.index.json"),
			r#"{"weight_map":{"layer.0":"model-00001.safetensors","layer.1":"model-00001.safetensors","layer.2":"model-00002.safetensors"}}"#,
		)
		.unwrap();
		let plan = checkpoint_plan(directory.path()).expect("unambiguous indexed checkpoint");
		let names = plan
			.files()
			.iter()
			.filter_map(|path| path.file_name())
			.collect::<BTreeSet<_>>();
		assert_eq!(
			names,
			BTreeSet::from([
				OsStr::new("model-00001.safetensors"),
				OsStr::new("model-00002.safetensors"),
			])
		);
	}

	#[test]
	fn index_rejects_tensor_present_in_selected_shard_but_absent_from_map() {
		let directory = tempfile::tempdir().unwrap();
		write_shard(
			&directory.path().join("model-00001.safetensors"),
			&["indexed", "hidden"],
		);
		fs::write(
			directory.path().join("model.safetensors.index.json"),
			r#"{"weight_map":{"indexed":"model-00001.safetensors"}}"#,
		)
		.unwrap();
		let error = checkpoint_plan(directory.path()).unwrap_err();
		assert!(error.message().contains("unindexed tensor"));
	}

	#[test]
	fn variant_index_is_rejected_even_with_canonical_weights() {
		let directory = tempfile::tempdir().unwrap();
		write_shard(&directory.path().join("model.safetensors"), &["x"]);
		fs::write(
			directory.path().join("adapter.safetensors.index.json"),
			r#"{"weight_map":{"x":"model.safetensors"}}"#,
		)
		.unwrap();
		let error = checkpoint_plan(directory.path()).unwrap_err();
		assert!(error.message().contains("ambiguous"));
	}

	#[test]
	fn rejects_tensor_payload_length_that_disagrees_with_dtype_and_shape() {
		let directory = tempfile::tempdir().unwrap();
		let header = serde_json::json!({
			"x": {
				"dtype": "F32",
				"shape": [2],
				"data_offsets": [0, 4],
			}
		});
		let encoded = serde_json::to_vec(&header).unwrap();
		let path = directory.path().join("model.safetensors");
		let mut file = File::create(&path).unwrap();
		file.write_all(&(encoded.len() as u64).to_le_bytes())
			.unwrap();
		file.write_all(&encoded).unwrap();
		file.write_all(&[0_u8; 4]).unwrap();
		let error = checkpoint_plan(directory.path()).unwrap_err();
		assert!(error.message().contains("dtype/shape require"));
	}

	#[test]
	fn rejects_nul_in_tensor_name() {
		let directory = tempfile::tempdir().unwrap();
		write_shard(
			&directory.path().join("model.safetensors"),
			&["unsafe\0name"],
		);
		let error = checkpoint_plan(directory.path()).unwrap_err();
		assert!(error.message().contains("no NUL"));
	}

	#[test]
	fn opened_shard_must_match_planned_inode_and_header() {
		let directory = tempfile::tempdir().unwrap();
		let path = directory.path().join("model.safetensors");
		write_shard(&path, &["first"]);
		let plan = checkpoint_plan(directory.path()).unwrap();
		fs::remove_file(&path).unwrap();
		write_shard(&path, &["other"]);
		let mut replaced = open_no_follow(&path).unwrap();
		let error = plan
			.validate_opened_shard(&path, &mut replaced)
			.unwrap_err();
		assert!(error.message().contains("changed"));
	}

	#[test]
	fn same_inode_header_rewrite_is_detected() {
		let directory = tempfile::tempdir().unwrap();
		let path = directory.path().join("model.safetensors");
		write_shard(&path, &["tensor"]);
		let plan = checkpoint_plan(directory.path()).unwrap();
		let header = serde_json::json!({
			"tensor": {
				"dtype": "I32",
				"shape": [1],
				"data_offsets": [0, 4],
			}
		});
		let encoded = serde_json::to_vec(&header).unwrap();
		let mut file = File::create(&path).unwrap();
		file.write_all(&(encoded.len() as u64).to_le_bytes())
			.unwrap();
		file.write_all(&encoded).unwrap();
		file.write_all(&[0_u8; 4]).unwrap();
		drop(file);
		let mut changed = open_no_follow(&path).unwrap();
		assert!(plan.validate_opened_shard(&path, &mut changed).is_err());
	}

	#[test]
	fn source_plan_defers_payload_identity_to_private_capture() {
		let directory = tempfile::tempdir().unwrap();
		let path = directory.path().join("model.safetensors");
		write_shard(&path, &["tensor"]);
		let plan = checkpoint_plan(directory.path()).unwrap();
		let mut file = OpenOptions::new().write(true).open(&path).unwrap();
		file.seek(SeekFrom::End(-1)).unwrap();
		file.write_all(&[1]).unwrap();
		drop(file);
		let mut changed = open_no_follow(&path).unwrap();
		assert!(plan.validate_opened_shard(&path, &mut changed).is_ok());
	}

	#[test]
	fn held_directory_descriptor_prevents_mid_snapshot_root_mix() {
		let parent = tempfile::tempdir().unwrap();
		let live = parent.path().join("model");
		let parked_a = parent.path().join("model-a");
		let parked_b = parent.path().join("model-b");
		fs::create_dir(&live).unwrap();
		fs::write(live.join("config.json"), br#"{"identity":"a"}"#).unwrap();
		fs::write(
			live.join("tokenizer.json"),
			br#"{"identity":"tokenizer-a"}"#,
		)
		.unwrap();
		fs::write(
			live.join("tokenizer_config.json"),
			br#"{"identity":"tokenizer-config-a"}"#,
		)
		.unwrap();
		fs::write(live.join("chat_template.jinja"), b"template-a").unwrap();
		fs::write(
			live.join("generation_config.json"),
			br#"{"identity":"generation-a"}"#,
		)
		.unwrap();
		write_shard(&live.join("model.safetensors"), &["a"]);
		fs::create_dir(&parked_b).unwrap();
		fs::write(parked_b.join("config.json"), br#"{"identity":"b"}"#).unwrap();
		fs::write(
			parked_b.join("tokenizer.json"),
			br#"{"identity":"tokenizer-b"}"#,
		)
		.unwrap();
		fs::write(
			parked_b.join("tokenizer_config.json"),
			br#"{"identity":"tokenizer-config-b"}"#,
		)
		.unwrap();
		fs::write(parked_b.join("chat_template.jinja"), b"template-b").unwrap();
		fs::write(
			parked_b.join("generation_config.json"),
			br#"{"identity":"generation-b"}"#,
		)
		.unwrap();
		write_shard(&parked_b.join("model.safetensors"), &["b"]);

		let snapshot = CheckpointSnapshot::open_with_after_config(&live, || {
			fs::rename(&live, &parked_a).unwrap();
			fs::rename(&parked_b, &live).unwrap();
		})
		.unwrap();

		assert_eq!(snapshot.config_bytes(), br#"{"identity":"a"}"#);
		assert_eq!(
			snapshot.runtime_metadata("tokenizer.json"),
			Some(br#"{"identity":"tokenizer-a"}"#.as_slice())
		);
		assert_eq!(
			snapshot.runtime_metadata("tokenizer_config.json"),
			Some(br#"{"identity":"tokenizer-config-a"}"#.as_slice())
		);
		assert_eq!(
			snapshot.runtime_metadata("chat_template.jinja"),
			Some(b"template-a".as_slice())
		);
		assert_eq!(
			snapshot.runtime_metadata("generation_config.json"),
			Some(br#"{"identity":"generation-a"}"#.as_slice())
		);
		assert!(snapshot.shards[0].expected.layout.names.contains("a"));
		assert!(!snapshot.shards[0].expected.layout.names.contains("b"));
	}

	#[test]
	fn verified_snapshot_binds_every_captured_file_to_manifest() {
		let model = tempfile::tempdir().unwrap();
		let temp = tempfile::tempdir().unwrap();
		write_snapshot_fixture(model.path());
		let expected = manifest_files(model.path());

		let snapshot =
			CheckpointSnapshot::open_verified_in(model.path(), temp.path(), &expected).unwrap();
		assert_eq!(snapshot.config_bytes(), br#"{"identity":"a"}"#);
		assert!(temp.path().read_dir().unwrap().next().is_none());
		drop(snapshot);

		fs::write(
			model.path().join("tokenizer.json"),
			br#"{"identity":"tokenizer-b"}"#,
		)
		.unwrap();
		let error =
			CheckpointSnapshot::open_verified_in(model.path(), temp.path(), &expected).unwrap_err();
		assert!(error.message().contains("differs from installed manifest"));
	}

	#[test]
	fn snapshot_normalizes_current_named_chat_templates() {
		let model = tempfile::tempdir().unwrap();
		let temp = tempfile::tempdir().unwrap();
		write_snapshot_fixture(model.path());
		let mut expected = manifest_files(model.path());
		let templates = model.path().join("additional_chat_templates");
		fs::create_dir(&templates).unwrap();
		let default = b"default-template";
		let tool_use = b"tool-template";
		fs::write(templates.join("default.jinja"), default).unwrap();
		fs::write(templates.join("tool_use.jinja"), tool_use).unwrap();
		for (name, bytes) in [
			("chat_template.jinja", default.as_slice()),
			("chat_template_tool_use.jinja", tool_use.as_slice()),
		] {
			expected.push(
				ModelFile::new(
					name.to_string(),
					u64::try_from(bytes.len()).unwrap(),
					hex::encode(sha2::Sha256::digest(bytes)),
				)
				.unwrap(),
			);
		}

		let snapshot =
			CheckpointSnapshot::open_verified_in(model.path(), temp.path(), &expected).unwrap();
		assert_eq!(
			snapshot.runtime_metadata("chat_template.jinja"),
			Some(default.as_slice())
		);
		assert_eq!(
			snapshot.runtime_metadata("chat_template_tool_use.jinja"),
			Some(tool_use.as_slice())
		);
	}

	#[test]
	fn snapshot_preserves_legacy_json_and_rejects_named_conflict() {
		let model = tempfile::tempdir().unwrap();
		write_snapshot_fixture(model.path());
		let legacy = br#"{"chat_template":"legacy"}"#;
		fs::write(model.path().join("chat_template.json"), legacy).unwrap();
		let snapshot = CheckpointSnapshot::open(model.path()).unwrap();
		assert_eq!(
			snapshot.runtime_metadata("chat_template.json"),
			Some(legacy.as_slice())
		);
		drop(snapshot);

		let templates = model.path().join("additional_chat_templates");
		fs::create_dir(&templates).unwrap();
		fs::write(templates.join("default.jinja"), b"named").unwrap();
		let error = CheckpointSnapshot::open(model.path()).unwrap_err();
		assert!(error.message().contains("conflicts"));
	}

	#[test]
	fn verified_snapshot_rejects_runtime_metadata_restored_after_capture() {
		let model = tempfile::tempdir().unwrap();
		let temp = tempfile::tempdir().unwrap();
		write_snapshot_fixture(model.path());
		let expected = manifest_files(model.path());
		let path = model.path().join("tokenizer.json");
		let restore_path = path.clone();
		let bytes = fs::read(&path).unwrap();

		let error = CheckpointSnapshot::open_verified_with_metadata_seam(
			model.path(),
			temp.path(),
			&expected,
			|| fs::remove_file(&path).unwrap(),
			|| fs::write(&restore_path, bytes).unwrap(),
		)
		.unwrap_err();

		assert!(
			error
				.message()
				.contains("was absent from the captured snapshot")
		);
	}

	#[test]
	fn verified_snapshot_rejects_changed_shard_and_index() {
		let model = tempfile::tempdir().unwrap();
		let temp = tempfile::tempdir().unwrap();
		write_snapshot_fixture(model.path());
		fs::write(
			model.path().join("model.safetensors.index.json"),
			r#"{"weight_map":{"a":"model.safetensors"}}"#,
		)
		.unwrap();
		let expected = manifest_files(model.path());

		let shard_path = model.path().join("model.safetensors");
		let mut shard = OpenOptions::new().write(true).open(&shard_path).unwrap();
		shard.seek(SeekFrom::End(-1)).unwrap();
		shard.write_all(&[1]).unwrap();
		drop(shard);
		let shard_error =
			CheckpointSnapshot::open_verified_in(model.path(), temp.path(), &expected).unwrap_err();
		assert!(
			shard_error
				.message()
				.contains("differs from installed manifest")
		);

		write_shard(&shard_path, &["a"]);
		fs::write(
			model.path().join("model.safetensors.index.json"),
			"{ \"weight_map\": { \"a\": \"model.safetensors\" } }",
		)
		.unwrap();
		let index_error =
			CheckpointSnapshot::open_verified_in(model.path(), temp.path(), &expected).unwrap_err();
		assert!(
			index_error
				.message()
				.contains("differs from installed manifest")
		);
	}

	#[test]
	fn verified_snapshot_rejects_unmanifested_runtime_metadata() {
		let model = tempfile::tempdir().unwrap();
		let temp = tempfile::tempdir().unwrap();
		write_snapshot_fixture(model.path());
		let expected = manifest_files(model.path());
		fs::write(model.path().join("chat_template.jinja"), "added later").unwrap();

		let error =
			CheckpointSnapshot::open_verified_in(model.path(), temp.path(), &expected).unwrap_err();
		assert!(
			error
				.message()
				.contains("absent from the installed manifest")
		);
	}

	#[test]
	fn descriptor_relative_plan_survives_root_swap_before_enumeration() {
		let parent = tempfile::tempdir().unwrap();
		let live = parent.path().join("model");
		let parked_a = parent.path().join("model-a");
		fs::create_dir(&live).unwrap();
		write_shard(&live.join("model.safetensors"), &["a"]);
		let directory = open_directory_no_follow(&live).unwrap();

		fs::rename(&live, &parked_a).unwrap();
		fs::create_dir(&live).unwrap();
		write_shard(&live.join("model.safetensors"), &["b"]);

		let plan = checkpoint_plan_from_directory(&directory, &live).unwrap();
		let snapshot = plan.snapshots.get(&live.join("model.safetensors")).unwrap();
		assert!(snapshot.names.contains("a"));
		assert!(!snapshot.names.contains("b"));
	}

	#[test]
	fn owned_snapshot_survives_directory_swap_and_a_b_a_path_cycle() {
		let parent = tempfile::tempdir().unwrap();
		let live = parent.path().join("model");
		let parked_a = parent.path().join("model-a");
		let parked_b = parent.path().join("model-b");
		fs::create_dir(&live).unwrap();
		fs::write(live.join("config.json"), br#"{"identity":"a"}"#).unwrap();
		write_shard(&live.join("model.safetensors"), &["a"]);
		let mut snapshot = CheckpointSnapshot::open(&live).unwrap();
		let shard_digest = snapshot
			.shard_sha256("model.safetensors")
			.unwrap()
			.to_string();

		fs::rename(&live, &parked_a).unwrap();
		fs::create_dir(&live).unwrap();
		fs::write(live.join("config.json"), br#"{"identity":"b"}"#).unwrap();
		write_shard(&live.join("model.safetensors"), &["b"]);
		snapshot.shards_mut()[0].validate().unwrap();
		assert_eq!(snapshot.config_bytes(), br#"{"identity":"a"}"#);
		assert_eq!(
			snapshot.shard_sha256("model.safetensors"),
			Some(shard_digest.as_str())
		);

		fs::rename(&live, &parked_b).unwrap();
		fs::rename(&parked_a, &live).unwrap();
		snapshot.shards_mut()[0].validate().unwrap();
		assert_eq!(snapshot.config_bytes(), br#"{"identity":"a"}"#);
	}

	#[test]
	fn private_snapshot_isolated_from_in_place_source_mutation() {
		let directory = tempfile::tempdir().unwrap();
		fs::write(directory.path().join("config.json"), br#"{"identity":"a"}"#).unwrap();
		let path = directory.path().join("model.safetensors");
		write_shard(&path, &["a"]);
		let mut snapshot = CheckpointSnapshot::open(directory.path()).unwrap();
		let expected_digest = snapshot
			.shard_sha256("model.safetensors")
			.unwrap()
			.to_string();

		let mut source = OpenOptions::new().write(true).open(&path).unwrap();
		source.seek(SeekFrom::End(-1)).unwrap();
		source.write_all(&[7]).unwrap();
		drop(source);

		snapshot.shards_mut()[0].validate().unwrap();
		assert_eq!(
			snapshot.shard_sha256("model.safetensors"),
			Some(expected_digest.as_str())
		);
	}

	#[test]
	fn metadata_values_must_be_strings() {
		let directory = tempfile::tempdir().unwrap();
		let path = directory.path().join("model.safetensors");
		let header = serde_json::json!({
			"__metadata__": {"bad": 1},
			"tensor": {
				"dtype": "F32",
				"shape": [1],
				"data_offsets": [0, 4],
			}
		});
		let encoded = serde_json::to_vec(&header).unwrap();
		let mut file = File::create(&path).unwrap();
		file.write_all(&(encoded.len() as u64).to_le_bytes())
			.unwrap();
		file.write_all(&encoded).unwrap();
		file.write_all(&[0_u8; 4]).unwrap();
		let error = checkpoint_plan(directory.path()).unwrap_err();
		assert!(error.message().contains("values must all be strings"));
	}
}
