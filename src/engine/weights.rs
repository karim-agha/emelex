//! Checkpoint loading: single-file and sharded safetensors.

use std::{
	collections::{HashMap, HashSet},
	ffi::{CStr, CString},
	path::Path,
};

use crate::{
	engine::{
		array::Array,
		error::{Error, Result, check, install_error_handler},
		stream::cpu_stream,
		sys,
	},
	model::layout::{CheckpointSnapshot, OpenedShard},
};

struct ArrayMap(sys::mlx_map_string_to_array);

impl ArrayMap {
	fn new() -> Result<Self> {
		let raw = unsafe { sys::mlx_map_string_to_array_new() };
		if raw.ctx.is_null() {
			check(1)?;
		}
		Ok(Self(raw))
	}
}

impl Drop for ArrayMap {
	fn drop(&mut self) {
		if !self.0.ctx.is_null() {
			unsafe {
				let _ = sys::mlx_map_string_to_array_free(self.0);
			}
		}
	}
}

struct StringMap(sys::mlx_map_string_to_string);

impl StringMap {
	fn new() -> Result<Self> {
		let raw = unsafe { sys::mlx_map_string_to_string_new() };
		if raw.ctx.is_null() {
			check(1)?;
		}
		Ok(Self(raw))
	}
}

impl Drop for StringMap {
	fn drop(&mut self) {
		if !self.0.ctx.is_null() {
			unsafe {
				let _ = sys::mlx_map_string_to_string_free(self.0);
			}
		}
	}
}

struct ArrayMapIterator(sys::mlx_map_string_to_array_iterator);

impl ArrayMapIterator {
	fn new(map: &ArrayMap) -> Result<Self> {
		let raw = unsafe { sys::mlx_map_string_to_array_iterator_new(map.0) };
		if raw.ctx.is_null() || raw.map_ctx.is_null() {
			check(1)?;
		}
		Ok(Self(raw))
	}
}

impl Drop for ArrayMapIterator {
	fn drop(&mut self) {
		if !self.0.ctx.is_null() {
			unsafe {
				let _ = sys::mlx_map_string_to_array_iterator_free(self.0);
			}
		}
	}
}

struct RawArray(sys::mlx_array);

impl RawArray {
	/// Allocate the intentionally empty mlx-c out-handle filled by
	/// `mlx_map_string_to_array_iterator_next`.
	fn new() -> Self {
		Self(Array::new_handle())
	}

	fn into_array(mut self) -> Result<Array> {
		let raw = self.0;
		self.0.ctx = std::ptr::null_mut();
		Array::from_raw(raw)
	}
}

impl Drop for RawArray {
	fn drop(&mut self) {
		if !self.0.ctx.is_null() {
			unsafe {
				let _ = sys::mlx_array_free(self.0);
			}
		}
	}
}

/// Load every tensor from the checkpoint's immutable shard snapshot.
pub fn load_all(model_dir: &Path) -> Result<HashMap<String, Array>> {
	let runtime = crate::runtime::initialize_default_if_needed()
		.map_err(|error| Error::Mlx(error.to_string()))?;
	let mut snapshot = CheckpointSnapshot::open_in(model_dir, &runtime.home().join("temp"))
		.map_err(|error| Error::Config(error.to_string()))?;
	load_snapshot(&mut snapshot, model_dir, |_| true)
}

/// Whether one raw checkpoint key belongs to an MTP tensor namespace.
pub(crate) fn is_mtp_tensor_name(name: &str) -> bool {
	name.split('.').any(|segment| segment == "mtp")
}

pub(crate) fn load_snapshot(
	snapshot: &mut CheckpointSnapshot,
	model_dir: &Path,
	mut include: impl FnMut(&str) -> bool,
) -> Result<HashMap<String, Array>> {
	load_snapshot_with_hook(snapshot, model_dir, &mut include, &mut |_| {})
}

fn load_snapshot_with_hook(
	snapshot: &mut CheckpointSnapshot,
	model_dir: &Path,
	include: &mut impl FnMut(&str) -> bool,
	before_evaluate: &mut impl FnMut(&str),
) -> Result<HashMap<String, Array>> {
	if !snapshot.has_shards() {
		return Err(Error::Model(format!(
			"no runnable safetensors checkpoint in {}",
			model_dir.display()
		)));
	}
	let mut all = HashMap::new();
	let mut seen = HashSet::new();
	for shard in snapshot.shards_mut() {
		load_file(shard, &mut all, &mut seen, include, before_evaluate)?;
	}
	Ok(all)
}

/// Load selected tensors from one safetensors file.
///
/// Every name still participates in duplicate detection. Excluded iterator
/// handles are freed before conversion to [`Array`] or evaluation, so their
/// lazy `Load` payloads never materialize.
fn load_file(
	shard: &mut OpenedShard,
	out: &mut HashMap<String, Array>,
	seen: &mut HashSet<String>,
	include: &mut impl FnMut(&str) -> bool,
	before_evaluate: &mut impl FnMut(&str),
) -> Result<()> {
	install_error_handler();
	let stream = cpu_stream()?;
	shard
		.validate()
		.map_err(|error| Error::Config(error.to_string()))?;
	let path = shard.path().to_path_buf();
	let descriptor_path = format!("/dev/fd/{}", shard.descriptor());
	let c_path = CString::new(descriptor_path)
		.map_err(|_| Error::Model("checkpoint descriptor path contains NUL".to_string()))?;

	unsafe {
		let mut tensors = ArrayMap::new()?;
		let mut metadata = StringMap::new()?;
		check(sys::mlx_load_safetensors(
			&mut tensors.0,
			&mut metadata.0,
			c_path.as_ptr(),
			stream,
		))?;

		let it = ArrayMapIterator::new(&tensors)?;
		let mut duplicate = None;
		let mut load_error = None;
		loop {
			let mut key: *const std::ffi::c_char = std::ptr::null();
			let mut value = RawArray::new();
			let status = sys::mlx_map_string_to_array_iterator_next(&mut key, &mut value.0, it.0);
			match status {
				0 => {}
				2 => break,
				error => {
					load_error = Some(match check(error) {
						Err(error) => error,
						Ok(()) => Error::Mlx(format!(
							"MLX checkpoint iterator returned unexpected status {error}"
						)),
					});
					break;
				}
			}
			if key.is_null() {
				load_error = Some(Error::Mlx(String::from(
					"MLX checkpoint iterator returned a null tensor name",
				)));
				break;
			}
			let name = CStr::from_ptr(key).to_string_lossy().into_owned();
			if !seen.insert(name.clone()) {
				duplicate = Some(name);
				break;
			}
			if !include(&name) {
				continue;
			}
			before_evaluate(&name);
			let arr = value.into_array()?;
			// `mlx_load_safetensors` produces arrays backed by the `Load`
			// primitive, which only has a CPU eval kernel. Materialize each
			// tensor immediately so later GPU ops never have to evaluate a
			// graph with a dangling `Load` node.
			if let Err(error) = arr.eval() {
				load_error = Some(error);
				break;
			}
			out.insert(name, arr);
		}
		if let Some(error) = load_error {
			return Err(error);
		}
		if let Some(name) = duplicate {
			return Err(Error::Config(format!(
				"duplicate tensor {name:?} while loading {}",
				path.display()
			)));
		}
	}
	// The descriptor is an unlinked private clone. Revalidate its cheap
	// header/layout identity after iterating every entry and materializing every
	// selected lazy Load primitive; no model-owned path or external writer can
	// reach its payload.
	shard
		.validate()
		.map_err(|error| Error::Config(error.to_string()))?;
	Ok(())
}

#[cfg(test)]
mod tests {
	#![allow(clippy::expect_used)]

	use std::cell::Cell;

	use super::*;

	#[test]
	fn load_all_fills_empty_array_iterator_handles() {
		crate::runtime::initialize_default_if_needed().expect("runtime initializes");
		let directory = tempfile::tempdir().expect("temporary model directory");
		crate::engine::test_support::write_safetensors(
			&directory.path().join("model.safetensors"),
			&[("probe".to_string(), vec![2], vec![1.25, -2.5])],
		)
		.expect("write tiny safetensors");
		std::fs::write(directory.path().join("config.json"), b"{}")
			.expect("write minimal model config");

		let loaded = load_all(directory.path());
		if let Err(Error::Mlx(message)) = &loaded
			&& (message.contains("No Metal device") || message.contains("no Metal device"))
		{
			// Headless macOS cannot complete MLX evaluation, but reaching that
			// environmental boundary still proves the iterator accepted and
			// filled the initially empty array out-handle.
			return;
		}
		let tensors = loaded.expect("load nonempty safetensors");

		assert_eq!(tensors.len(), 1);
		assert_eq!(
			tensors["probe"].to_vec_f32().expect("materialized values"),
			vec![1.25, -2.5]
		);
	}

	#[test]
	fn excluded_mtp_iterator_handles_are_not_evaluated_or_returned() {
		crate::runtime::initialize_default_if_needed().expect("runtime initializes");
		let directory = tempfile::tempdir().expect("temporary model directory");
		crate::engine::test_support::write_safetensors(
			&directory.path().join("model.safetensors"),
			&[(
				"language_model.mtp.probe".to_string(),
				vec![2],
				vec![1.25, -2.5],
			)],
		)
		.expect("write tiny safetensors");
		std::fs::write(directory.path().join("config.json"), b"{}")
			.expect("write minimal model config");
		let runtime = crate::runtime::initialize_default_if_needed().expect("runtime");
		let mut snapshot =
			CheckpointSnapshot::open_in(directory.path(), &runtime.home().join("temp"))
				.expect("snapshot");

		let reached_evaluation_boundary = Cell::new(false);
		let tensors = load_snapshot_with_hook(
			&mut snapshot,
			directory.path(),
			&mut |name| !is_mtp_tensor_name(name),
			&mut |_| {
				reached_evaluation_boundary.set(true);
			},
		)
		.expect("excluded lazy tensor must not require evaluation");

		assert!(tensors.is_empty());
		assert!(!reached_evaluation_boundary.get());
	}
}
