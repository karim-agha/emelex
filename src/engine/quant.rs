//! Quantization configuration parsing and resolution.
//!
//! Follows mlx-lm semantics: `config.json` carries a `quantization` object
//! with global defaults (`group_size`, `bits`, optional `mode`) plus
//! per-layer overrides keyed by module path. A layer is quantized when
//! either an override exists for its path or a `{path}.scales` tensor is
//! present in the checkpoint.

use std::collections::HashMap;

use serde_json::Value;

use crate::engine::{
	array::{Array, Dtype},
	error::{Error, Result},
	ops::{self, QuantMode},
};

const MAX_LAYER_OVERRIDES: usize = 65_536;
const MAX_LAYER_PATH_BYTES: usize = 1_024;

/// Quantization parameters for one layer (or the global default).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantParams {
	pub group_size: i32,
	pub bits: i32,
	pub mode: QuantMode,
}

/// Per-layer override: quantize with specific params, or skip quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerOverride {
	Params(QuantParams),
	Skip,
}

/// Parsed `quantization` section of a model config.
#[derive(Debug, Clone, Default)]
pub struct Quantization {
	pub default: Option<QuantParams>,
	pub per_layer: HashMap<String, LayerOverride>,
}

impl Quantization {
	/// Parse from the root config JSON. Checks `quantization` first, then
	/// `quantization_config` (kept in sync by mlx-lm on save).
	pub fn from_config(config: &Value) -> Result<Self> {
		// emelex patch: checkpoint config is untrusted. Parse without lossy
		// casts and admit only mode/group/bit tuples backed by vendored MLX.
		let section = config
			.get("quantization")
			.or_else(|| config.get("quantization_config"));
		let Some(section) = section else {
			return Ok(Quantization::default());
		};
		let section = section.as_object().ok_or_else(|| {
			Error::Config("quantization configuration must be a JSON object".to_string())
		})?;
		if section.len() > MAX_LAYER_OVERRIDES + 3 {
			return Err(Error::Config(format!(
				"quantization configuration has too many layer overrides (maximum \
				 {MAX_LAYER_OVERRIDES})"
			)));
		}

		let default_mode = section
			.get("mode")
			.and_then(|m| m.as_str())
			.map(QuantMode::parse)
			.transpose()?
			.unwrap_or(QuantMode::Affine);
		if section.get("mode").is_some_and(|value| !value.is_string()) {
			return Err(Error::Config(
				"quantization mode must be a string".to_string(),
			));
		}
		let default_mode_params = mode_defaults(default_mode);
		let default_group = optional_i32(section.get("group_size"), "quantization.group_size")?
			.unwrap_or(default_mode_params.0);
		let default_bits = optional_i32(section.get("bits"), "quantization.bits")?
			.unwrap_or(default_mode_params.1);
		let default = Some(validate_params(
			QuantParams {
				group_size: default_group,
				bits: default_bits,
				mode: default_mode,
			},
			"quantization",
		)?);

		let mut per_layer = HashMap::new();
		for (key, value) in section {
			if matches!(key.as_str(), "group_size" | "bits" | "mode") {
				continue;
			}
			if key.is_empty() || key.len() > MAX_LAYER_PATH_BYTES || key.contains('\0') {
				return Err(Error::Config(format!(
					"invalid quantization layer path {key:?}"
				)));
			}
			match value {
				Value::Object(obj) => {
					for field in obj.keys() {
						if !matches!(field.as_str(), "group_size" | "bits" | "mode") {
							return Err(Error::Config(format!(
								"layer '{key}' quant override has unsupported field \
								 {field:?}"
							)));
						}
					}
					let mode = obj
						.get("mode")
						.and_then(|m| m.as_str())
						.map(QuantMode::parse)
						.transpose()?
						.unwrap_or(default_mode);
					if obj.get("mode").is_some_and(|value| !value.is_string()) {
						return Err(Error::Config(format!(
							"layer '{key}' quantization mode must be a string"
						)));
					}
					let inherited = if mode == default_mode {
						(default_group, default_bits)
					} else {
						mode_defaults(mode)
					};
					let group_size = optional_i32(
						obj.get("group_size"),
						&format!("quantization.{key}.group_size"),
					)?
					.unwrap_or(inherited.0);
					let bits = optional_i32(obj.get("bits"), &format!("quantization.{key}.bits"))?
						.unwrap_or(inherited.1);
					let params = validate_params(
						QuantParams {
							group_size,
							bits,
							mode,
						},
						&format!("quantization.{key}"),
					)?;
					per_layer.insert(key.clone(), LayerOverride::Params(params));
				}
				Value::Bool(false) => {
					per_layer.insert(key.clone(), LayerOverride::Skip);
				}
				_ => {
					return Err(Error::Config(format!(
						"layer '{key}' quant override must be an object or false"
					)));
				}
			}
		}

		Ok(Quantization { default, per_layer })
	}

	/// Does `weight_map` carry per-tensor dynamic-range int8 weights at
	/// `path` (`{path}.input_min`/`input_max`/`output_min`/`output_max`
	/// sitting alongside `{path}.weight`, as opposed to `.scales`)? Used to
	/// pick between [`crate::engine::nn::WeightMap::linear`]'s group-affine path
	/// and [`dequantize_dynamic_int8`] when loading multimodal tower
	/// weights (Gemma4 vision/audio).
	pub fn is_dynamic_range_int8(weight_map: &crate::engine::nn::WeightMap, path: &str) -> bool {
		weight_map.contains(&format!("{path}.output_min"))
			&& weight_map.contains(&format!("{path}.output_max"))
	}

	/// Whether any quantization is configured at all.
	pub fn is_quantized(&self) -> bool {
		self.default.is_some() || !self.per_layer.is_empty()
	}

	/// Resolve quantization for the module at `path`, mirroring the mlx-lm
	/// `class_predicate`:
	/// - a per-layer override wins (params or skip),
	/// - otherwise quantize with defaults iff `{path}.scales` exists,
	/// - `path` may be probed with alternative prefixes by the caller.
	pub fn resolve(&self, path: &str, has_scales: bool) -> Option<QuantParams> {
		match self.per_layer.get(path) {
			Some(LayerOverride::Params(p)) => Some(*p),
			Some(LayerOverride::Skip) => None,
			None => {
				if has_scales {
					self.default
				} else {
					None
				}
			}
		}
	}
}

const fn mode_defaults(mode: QuantMode) -> (i32, i32) {
	match mode {
		QuantMode::Affine => (64, 4),
		QuantMode::Mxfp4 => (32, 4),
		QuantMode::Mxfp8 => (32, 8),
		QuantMode::Nvfp4 => (16, 4),
	}
}

fn optional_i32(value: Option<&Value>, path: &str) -> Result<Option<i32>> {
	let Some(value) = value else {
		return Ok(None);
	};
	let integer = value
		.as_i64()
		.ok_or_else(|| Error::Config(format!("{path} must be an integer")))?;
	i32::try_from(integer)
		.map(Some)
		.map_err(|_| Error::Config(format!("{path} is outside the supported i32 range")))
}

fn validate_params(params: QuantParams, path: &str) -> Result<QuantParams> {
	let valid = match params.mode {
		QuantMode::Affine => {
			matches!(params.group_size, 32 | 64 | 128)
				&& matches!(params.bits, 2 | 3 | 4 | 5 | 6 | 8)
		}
		QuantMode::Mxfp4 => params.group_size == 32 && params.bits == 4,
		QuantMode::Mxfp8 => params.group_size == 32 && params.bits == 8,
		QuantMode::Nvfp4 => params.group_size == 16 && params.bits == 4,
	};
	if !valid {
		return Err(Error::Config(format!(
			"{path} has unsupported parameters for mode '{}': group_size={}, bits={}",
			params.mode.as_str(),
			params.group_size,
			params.bits
		)));
	}
	Ok(params)
}

/// Per-tensor dynamic-range int8 quantization parameters (Gemma4
/// vision/audio tower weights). Distinct from the group-affine scheme
/// above: one scale/zero-point pair for the *whole* weight tensor rather
/// than per-group, plus separately-recorded input-activation range
/// (unused for pure weight-only dequantization, kept for completeness /
/// potential future activation-quantization support).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynamicRangeParams {
	pub input_min: f32,
	pub input_max: f32,
	pub output_min: f32,
	pub output_max: f32,
}

/// Dequantize a per-tensor dynamic-range int8 weight: maps the full signed
/// int8 range `[-128, 127]` affinely onto `[output_min, output_max]`.
pub fn dequantize_dynamic_int8(weight_i8: &Array, params: DynamicRangeParams) -> Result<Array> {
	let scale = (params.output_max - params.output_min) / 255.0;
	let shifted = ops::add(
		&ops::astype(weight_i8, Dtype::Float32)?,
		&Array::scalar_f32(128.0)?,
	)?;
	let scaled = ops::scale_by(&shifted, scale)?;
	ops::add(&scaled, &Array::scalar_f32(params.output_min)?)
}

/// Quantize a dense f32 weight tensor into the same per-tensor
/// dynamic-range int8 scheme [`dequantize_dynamic_int8`] reads back, using
/// the tensor's own min/max as the output range. Used by tests to check
/// the encode/decode round-trip; real checkpoints ship pre-quantized.
pub fn quantize_dynamic_int8(weight: &[f32]) -> (Vec<i8>, DynamicRangeParams) {
	let min = weight.iter().cloned().fold(f32::INFINITY, f32::min);
	let max = weight.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
	let scale = ((max - min) / 255.0).max(f32::EPSILON);
	let q: Vec<i8> = weight
		.iter()
		.map(|&w| (((w - min) / scale).round() - 128.0).clamp(-128.0, 127.0) as i8)
		.collect();
	(
		q,
		DynamicRangeParams {
			input_min: min,
			input_max: max,
			output_min: min,
			output_max: max,
		},
	)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn dynamic_int8_round_trip_is_close() {
		match crate::runtime::recommended_max_working_set_size() {
			Ok(_) => {}
			Err(crate::runtime::RuntimeError::Mlx(message))
				if message.contains("no Metal device is available") =>
			{
				return;
			}
			Err(error) => panic!("unexpected Metal budget failure: {error}"),
		}
		let original = vec![-2.0f32, -1.0, 0.0, 0.5, 1.0, 2.0, 3.5];
		let (q, params) = quantize_dynamic_int8(&original);
		let q_arr = Array::from_slice(
			&q.iter().map(|&v| v as i32).collect::<Vec<_>>(),
			&[original.len() as i32],
		)
		.unwrap();
		let q_arr = ops::astype(&q_arr, Dtype::Int8).unwrap();
		let dequantized = dequantize_dynamic_int8(&q_arr, params).unwrap();
		let out = dequantized.to_vec_f32().unwrap();
		let tolerance = (params.output_max - params.output_min) / 255.0 + 1e-4;
		for (a, b) in original.iter().zip(out.iter()) {
			assert!((a - b).abs() <= tolerance, "{a} vs {b} (tol {tolerance})");
		}
	}

	#[test]
	fn quantization_from_config_defaults_when_absent() {
		let cfg = serde_json::json!({});
		let q = Quantization::from_config(&cfg).unwrap();
		assert!(!q.is_quantized());
		assert!(q.default.is_none());
	}

	#[test]
	fn quantization_from_config_parses_global_defaults() {
		let cfg = serde_json::json!({"quantization": {"group_size": 64, "bits": 4}});
		let q = Quantization::from_config(&cfg).unwrap();
		assert!(q.is_quantized());
		let params = q.default.unwrap();
		assert_eq!(params.group_size, 64);
		assert_eq!(params.bits, 4);
		assert_eq!(params.mode, QuantMode::Affine);
	}

	#[test]
	fn quantization_accepts_exact_vendored_mlx_matrix() {
		for group_size in [32, 64, 128] {
			for bits in [2, 3, 4, 5, 6, 8] {
				let cfg = serde_json::json!({
					"quantization": {"mode": "affine", "group_size": group_size, "bits": bits}
				});
				assert!(
					Quantization::from_config(&cfg).is_ok(),
					"{group_size}x{bits}"
				);
			}
		}
		for (mode, group_size, bits) in [("mxfp4", 32, 4), ("mxfp8", 32, 8), ("nvfp4", 16, 4)] {
			let cfg = serde_json::json!({
				"quantization": {"mode": mode, "group_size": group_size, "bits": bits}
			});
			assert!(Quantization::from_config(&cfg).is_ok(), "{mode}");
		}
	}

	#[test]
	fn quantization_rejects_tuples_without_vendored_kernel_support() {
		for (mode, group_size, bits) in [
			("affine", 16, 4),
			("affine", 64, 0),
			("affine", 64, 1),
			("affine", 64, 9),
			("mxfp8", 16, 8),
			("mxfp8", 32, 4),
		] {
			let cfg = serde_json::json!({
				"quantization": {"mode": mode, "group_size": group_size, "bits": bits}
			});
			assert!(
				Quantization::from_config(&cfg).is_err(),
				"{mode}:{group_size}x{bits}"
			);
		}
	}

	#[test]
	fn quantization_per_layer_skip_override() {
		let cfg = serde_json::json!({
				"quantization": {"group_size": 64, "bits": 4, "model.layers.0.mlp": false}
		});
		let q = Quantization::from_config(&cfg).unwrap();
		assert_eq!(q.resolve("model.layers.0.mlp", true), None);
		assert!(q.resolve("model.layers.1.mlp", true).is_some());
	}

	#[test]
	fn quantization_per_layer_params_override() {
		let cfg = serde_json::json!({
				"quantization": {
						"group_size": 64, "bits": 4,
						"model.layers.0.mlp": {"group_size": 32, "bits": 8}
				}
		});
		let q = Quantization::from_config(&cfg).unwrap();
		let params = q.resolve("model.layers.0.mlp", true).unwrap();
		assert_eq!(params.group_size, 32);
		assert_eq!(params.bits, 8);
	}

	#[test]
	fn resolve_without_scales_and_without_override_is_none() {
		let cfg = serde_json::json!({"quantization": {"group_size": 64, "bits": 4}});
		let q = Quantization::from_config(&cfg).unwrap();
		assert_eq!(q.resolve("model.layers.0.mlp", false), None);
	}

	#[test]
	fn quantization_rejects_out_of_range_or_native_unsupported_parameters() {
		for cfg in [
			serde_json::json!({"quantization": {"group_size": -64, "bits": 4}}),
			serde_json::json!({"quantization": {"group_size": 64, "bits": 7}}),
			serde_json::json!({"quantization": {"group_size": 1_i64 << 40, "bits": 4}}),
			serde_json::json!({"quantization": {"mode": "mxfp4", "group_size": 64, "bits": 4}}),
			serde_json::json!({"quantization": {"mode": "nvfp4", "group_size": 16, "bits": 8}}),
		] {
			assert!(Quantization::from_config(&cfg).is_err(), "{cfg}");
		}
	}

	#[test]
	fn quantization_rejects_malformed_or_unknown_entries() {
		for cfg in [
			serde_json::json!({"quantization": []}),
			serde_json::json!({"quantization": {"group_size": "64"}}),
			serde_json::json!({"quantization": {"quant_method": "gptq"}}),
			serde_json::json!({"quantization": {"model.layers.0": true}}),
			serde_json::json!({
				"quantization": {"model.layers.0": {"bits": 4, "surprise": 1}}
			}),
		] {
			assert!(Quantization::from_config(&cfg).is_err(), "{cfg}");
		}
	}

	#[test]
	fn floating_point_modes_use_their_mlx_defaults() {
		for (mode, expected) in [
			(
				"mxfp4",
				QuantParams {
					group_size: 32,
					bits: 4,
					mode: QuantMode::Mxfp4,
				},
			),
			(
				"mxfp8",
				QuantParams {
					group_size: 32,
					bits: 8,
					mode: QuantMode::Mxfp8,
				},
			),
			(
				"nvfp4",
				QuantParams {
					group_size: 16,
					bits: 4,
					mode: QuantMode::Nvfp4,
				},
			),
		] {
			let cfg = serde_json::json!({"quantization": {"mode": mode}});
			assert_eq!(
				Quantization::from_config(&cfg).unwrap().default,
				Some(expected)
			);
		}
	}
}
