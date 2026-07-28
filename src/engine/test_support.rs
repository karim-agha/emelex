// emelex patch (not upstream): this entire module is a #[cfg(test)]
// emelex addition — on-disk tiny-model fixtures for engine-level tests.

//! Test support: a hand-rolled safetensors writer plus a builder that
//! materializes the committed `tests/fixtures/tiny-model` fixture into a
//! loadable on-disk checkpoint (deterministic tiny random-ish weights,
//! optionally including the 15-tensor dense MTP module), so
//! `Session::load` and the decode loop run end-to-end without a real
//! checkpoint.

use std::{
	io::Write as _,
	path::{Path, PathBuf},
	sync::atomic::{AtomicUsize, Ordering},
};

use crate::engine::error::Result;

/// A uniquely-named scratch directory removed on drop.
pub struct TempModelDir {
	dir: PathBuf,
}

impl TempModelDir {
	pub fn path(&self) -> &Path {
		&self.dir
	}
}

impl Drop for TempModelDir {
	fn drop(&mut self) {
		let _ = std::fs::remove_dir_all(&self.dir);
	}
}

/// On-disk element type for [`write_safetensors_typed`] tensors. Values
/// are supplied as f32 either way; `Bf16` rounds them to bfloat16
/// (round-to-nearest-even) at write time — the dtype the v1 MTP loader
/// contract requires for every MTP tensor.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SavedDtype {
	F32,
	Bf16,
}

/// Round an f32 to bfloat16 bits (round-to-nearest-even).
fn f32_to_bf16_bits(v: f32) -> u16 {
	let bits = v.to_bits();
	let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
	(bits.wrapping_add(rounding_bias) >> 16) as u16
}

/// Write one safetensors file: 8-byte little-endian header length, JSON
/// header (`{"name": {"dtype": "F32", "shape": [...], "data_offsets":
/// [s, e]}, ...}`), then the raw little-endian f32 tensor data.
pub fn write_safetensors(path: &Path, tensors: &[(String, Vec<i32>, Vec<f32>)]) -> Result<()> {
	let typed: Vec<(String, Vec<i32>, Vec<f32>, SavedDtype)> = tensors
		.iter()
		.map(|(n, s, d)| (n.clone(), s.clone(), d.clone(), SavedDtype::F32))
		.collect();
	write_safetensors_typed(path, &typed)
}

/// [`write_safetensors`] with a per-tensor on-disk dtype (`F32` or
/// `BF16`).
pub fn write_safetensors_typed(
	path: &Path,
	tensors: &[(String, Vec<i32>, Vec<f32>, SavedDtype)],
) -> Result<()> {
	let mut header = serde_json::Map::new();
	let mut offset = 0usize;
	for (name, shape, data, dtype) in tensors {
		let (tag, elem) = match dtype {
			SavedDtype::F32 => ("F32", 4),
			SavedDtype::Bf16 => ("BF16", 2),
		};
		let bytes = data.len() * elem;
		header.insert(
			name.clone(),
			serde_json::json!({
				"dtype": tag,
				"shape": shape,
				"data_offsets": [offset, offset + bytes],
			}),
		);
		offset += bytes;
	}
	let header_json = serde_json::Value::Object(header).to_string();
	let mut file = std::fs::File::create(path)?;
	file.write_all(&(header_json.len() as u64).to_le_bytes())?;
	file.write_all(header_json.as_bytes())?;
	for (_, _, data, dtype) in tensors {
		for value in data {
			match dtype {
				SavedDtype::F32 => file.write_all(&value.to_le_bytes())?,
				SavedDtype::Bf16 => {
					file.write_all(&f32_to_bf16_bits(*value).to_le_bytes())?;
				}
			}
		}
	}
	Ok(())
}

/// Deterministic, varied filler values (same scheme as the qwen3_5
/// synthetic-WeightMap tests, whose forwards are proven finite at these
/// dims): `sin(i * 0.7311 + seed) * 0.05`.
fn filler(seed: f32, shape: &[i32]) -> Vec<f32> {
	let len: usize = shape.iter().map(|&d| d as usize).product();
	(0..len)
		.map(|i| ((i as f32) * 0.7311 + seed).sin() * 0.05)
		.collect()
}

/// The complete tiny converted-orientation text-only backbone tensor set
/// — exactly the post-sanitize key names `Qwen35Model::load` consumes for
/// the fixture config (hidden 32, 2 layers: layer 0 gated-delta, layer 1
/// gated full attention, vocab 16, tied embeddings → no `lm_head`).
fn backbone_specs() -> Vec<(&'static str, Vec<i32>)> {
	vec![
		("language_model.model.embed_tokens.weight", vec![16, 32]),
		// Layer 0: gated-delta. key_dim 16, value_dim 32, conv_dim 64.
		(
			"language_model.model.layers.0.linear_attn.conv1d.weight",
			vec![64, 4, 1],
		),
		(
			"language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
			vec![64, 32],
		),
		(
			"language_model.model.layers.0.linear_attn.in_proj_z.weight",
			vec![32, 32],
		),
		(
			"language_model.model.layers.0.linear_attn.in_proj_b.weight",
			vec![4, 32],
		),
		(
			"language_model.model.layers.0.linear_attn.in_proj_a.weight",
			vec![4, 32],
		),
		("language_model.model.layers.0.linear_attn.dt_bias", vec![4]),
		("language_model.model.layers.0.linear_attn.A_log", vec![4]),
		(
			"language_model.model.layers.0.linear_attn.norm.weight",
			vec![8],
		),
		(
			"language_model.model.layers.0.linear_attn.out_proj.weight",
			vec![32, 32],
		),
		(
			"language_model.model.layers.0.mlp.gate_proj.weight",
			vec![64, 32],
		),
		(
			"language_model.model.layers.0.mlp.up_proj.weight",
			vec![64, 32],
		),
		(
			"language_model.model.layers.0.mlp.down_proj.weight",
			vec![32, 64],
		),
		(
			"language_model.model.layers.0.input_layernorm.weight",
			vec![32],
		),
		(
			"language_model.model.layers.0.post_attention_layernorm.weight",
			vec![32],
		),
		// Layer 1: gated full attention (q_proj rows = 2 * heads * head_dim).
		(
			"language_model.model.layers.1.self_attn.q_proj.weight",
			vec![64, 32],
		),
		(
			"language_model.model.layers.1.self_attn.k_proj.weight",
			vec![16, 32],
		),
		(
			"language_model.model.layers.1.self_attn.v_proj.weight",
			vec![16, 32],
		),
		(
			"language_model.model.layers.1.self_attn.o_proj.weight",
			vec![32, 32],
		),
		(
			"language_model.model.layers.1.self_attn.q_norm.weight",
			vec![16],
		),
		(
			"language_model.model.layers.1.self_attn.k_norm.weight",
			vec![16],
		),
		(
			"language_model.model.layers.1.mlp.gate_proj.weight",
			vec![64, 32],
		),
		(
			"language_model.model.layers.1.mlp.up_proj.weight",
			vec![64, 32],
		),
		(
			"language_model.model.layers.1.mlp.down_proj.weight",
			vec![32, 64],
		),
		(
			"language_model.model.layers.1.input_layernorm.weight",
			vec![32],
		),
		(
			"language_model.model.layers.1.post_attention_layernorm.weight",
			vec![32],
		),
		("language_model.model.norm.weight", vec![32]),
	]
}

/// The 15 dense MTP module tensors under the SOLE supported on-disk
/// namespace `language_model.mtp.*` at the tiny
/// dims — the loader canonicalizes them to in-memory `mtp.*`. Written as
/// BF16 (the only dtype the v1 MTP dtype guard accepts).
fn mtp_specs() -> Vec<(&'static str, Vec<i32>)> {
	vec![
		("language_model.mtp.fc.weight", vec![32, 64]),
		("language_model.mtp.pre_fc_norm_embedding.weight", vec![32]),
		("language_model.mtp.pre_fc_norm_hidden.weight", vec![32]),
		("language_model.mtp.norm.weight", vec![32]),
		(
			"language_model.mtp.layers.0.input_layernorm.weight",
			vec![32],
		),
		(
			"language_model.mtp.layers.0.post_attention_layernorm.weight",
			vec![32],
		),
		(
			"language_model.mtp.layers.0.self_attn.q_proj.weight",
			vec![64, 32],
		),
		(
			"language_model.mtp.layers.0.self_attn.k_proj.weight",
			vec![16, 32],
		),
		(
			"language_model.mtp.layers.0.self_attn.v_proj.weight",
			vec![16, 32],
		),
		(
			"language_model.mtp.layers.0.self_attn.o_proj.weight",
			vec![32, 32],
		),
		(
			"language_model.mtp.layers.0.self_attn.q_norm.weight",
			vec![16],
		),
		(
			"language_model.mtp.layers.0.self_attn.k_norm.weight",
			vec![16],
		),
		(
			"language_model.mtp.layers.0.mlp.gate_proj.weight",
			vec![64, 32],
		),
		(
			"language_model.mtp.layers.0.mlp.up_proj.weight",
			vec![64, 32],
		),
		(
			"language_model.mtp.layers.0.mlp.down_proj.weight",
			vec![32, 64],
		),
	]
}

/// Materialize a loadable tiny model directory: the committed fixture's
/// config/tokenizer/chat-template files plus a deterministic
/// `model.safetensors` (optionally including the dense MTP module).
pub fn write_tiny_model(with_mtp: bool) -> Result<TempModelDir> {
	static COUNTER: AtomicUsize = AtomicUsize::new(0);
	let dir = std::env::temp_dir().join(format!(
		"emelex-tiny-model-{}-{}",
		std::process::id(),
		COUNTER.fetch_add(1, Ordering::Relaxed)
	));
	std::fs::create_dir_all(&dir)?;
	let guard = TempModelDir { dir };

	let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-model");
	for name in [
		"config.json",
		"tokenizer.json",
		"tokenizer_config.json",
		"chat_template.jinja",
	] {
		std::fs::copy(fixture.join(name), guard.path().join(name))?;
	}

	let backbone = backbone_specs();
	let backbone_len = backbone.len();
	let mut specs = backbone;
	if with_mtp {
		specs.extend(mtp_specs());
	}
	let tensors: Vec<(String, Vec<i32>, Vec<f32>, SavedDtype)> = specs
		.into_iter()
		.enumerate()
		.map(|(i, (name, shape))| {
			let data = filler(i as f32, &shape);
			// MTP tensors ship BF16 per the v1 dense-BF16 loader contract;
			// the backbone stays F32 (the loader's dtype guard is MTP-only).
			let dtype = if i >= backbone_len {
				SavedDtype::Bf16
			} else {
				SavedDtype::F32
			};
			(name.to_string(), shape, data, dtype)
		})
		.collect();
	write_safetensors_typed(&guard.path().join("model.safetensors"), &tensors)?;
	Ok(guard)
}
