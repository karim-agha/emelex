//! Gemma 3 text-only architecture (`model_type: gemma3` / `gemma3_text`,
//! e.g. TranslateGemma): a Gemma-style decoder with a 5:1
//! sliding/full-attention interleave, per-layer-type RoPE (local base on
//! sliding layers; global base with optional linear position scaling on
//! full layers), q/k RMSNorm, four norms per block, GeLU-tanh MLP,
//! `query_pre_attn_scalar`-derived attention scale, and tied (possibly
//! quantized) embeddings as the LM head.
//!
//! Unlike [`super::gemma4`] (the Gemma 3n family) there are no per-layer
//! input embeddings, no KV sharing, no per-block scalars, no value-path
//! norm, and one uniform `head_dim` for both layer types.
//!
//! Multimodal `Gemma3ForConditionalGeneration` checkpoints degrade to
//! text-only: `vision_tower.*` / `multi_modal_projector.*` weights are
//! dropped by [`sanitize`] (and skipped before materialization by the
//! loader's include predicate). Weight paths are canonicalized to the
//! `language_model.model.*` prefix multimodal conversions already use;
//! bare-prefix `Gemma3ForCausalLM` checkpoints (`model.*`) are renamed,
//! with the identical mapping applied to quantization-override keys so
//! checkpoint-verbatim per-layer overrides keep resolving.

use serde_json::Value;

use super::{
	cache::{KvCache, LayerCache},
	config::{get_bool, get_f32, get_i32, get_str, optional_i32, require_i32, text_config},
};
use crate::engine::{
	array::Array,
	error::{Error, Result},
	nn::{Embedding, Linear, RmsNorm, WeightMap},
	ops::{self, AttentionMask},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
	Sliding,
	Full,
}

/// Read a config value, treating JSON `null` as absent. HF Gemma 3
/// configs ship literal nulls for retired fields (`rope_scaling`,
/// `attn_logit_softcapping`, `final_logit_softcapping`).
fn non_null<'a>(cfg: &'a Value, key: &str) -> Option<&'a Value> {
	cfg.get(key).filter(|value| !value.is_null())
}

/// Null-tolerant [`get_f32`].
fn get_f32_nullable(cfg: &Value, key: &str, default: f32) -> Result<f32> {
	match cfg.get(key) {
		None | Some(Value::Null) => Ok(default),
		Some(_) => get_f32(cfg, key, default),
	}
}

/// Null-tolerant optional float (`None` for absent or `null`).
fn optional_f32_nullable(cfg: &Value, key: &str) -> Result<Option<f32>> {
	match cfg.get(key) {
		None | Some(Value::Null) => Ok(None),
		Some(value) => super::config::finite_f32(value, key).map(Some),
	}
}

#[derive(Debug, Clone)]
pub struct Gemma3Config {
	pub hidden_size: i32,
	pub num_hidden_layers: i32,
	pub num_attention_heads: i32,
	pub num_key_value_heads: i32,
	pub head_dim: i32,
	pub rms_norm_eps: f32,
	pub vocab_size: i32,
	/// `query_pre_attn_scalar^-0.5` — Gemma scales queries by this instead
	/// of the usual `head_dim^-0.5`.
	pub attention_scale: f32,
	pub sliding_window: i32,
	pub final_logit_softcapping: Option<f32>,
	pub tie_word_embeddings: bool,
	pub layer_types: Vec<LayerType>,
	/// Global (full-attention) RoPE base.
	pub rope_theta: f32,
	/// Local (sliding-attention) RoPE base.
	pub rope_local_base_freq: f32,
	/// `1 / factor` from linear rope scaling; 1.0 when unscaled. Applies
	/// to full-attention layers only.
	pub rope_global_scale: f32,
}

impl Gemma3Config {
	pub fn from_json(root: &Value) -> Result<Self> {
		let cfg = text_config(root);
		let hidden_size = require_i32(cfg, "hidden_size")?;
		let num_hidden_layers = require_i32(cfg, "num_hidden_layers")?;
		// HF Gemma 3 configs spell the fallback pattern with a leading
		// underscore (`_sliding_window_pattern`); accept both spellings.
		let sliding_window_pattern = match optional_i32(cfg, "_sliding_window_pattern")? {
			Some(value) => value,
			None => get_i32(cfg, "sliding_window_pattern", 6)?,
		};
		if sliding_window_pattern <= 0 {
			return Err(Error::Config(
				"Gemma3 sliding_window_pattern must be positive".to_string(),
			));
		}

		let layer_types = match cfg.get("layer_types") {
			Some(Value::Array(arr)) => arr
				.iter()
				.map(|v| match v.as_str() {
					Some("full_attention") => Ok(LayerType::Full),
					Some("sliding_attention") => Ok(LayerType::Sliding),
					Some(value) => Err(Error::Config(format!(
						"unsupported Gemma3 layer type '{value}'"
					))),
					None => Err(Error::Config(
						"Gemma3 layer_types entries must be strings".to_string(),
					)),
				})
				.collect::<Result<Vec<_>>>()?,
			Some(_) => {
				return Err(Error::Config(
					"Gemma3 layer_types must be an array".to_string(),
				));
			}
			None => (0..num_hidden_layers)
				.map(|i| {
					if (i + 1) % sliding_window_pattern == 0 {
						LayerType::Full
					} else {
						LayerType::Sliding
					}
				})
				.collect(),
		};
		if layer_types.len() != num_hidden_layers as usize {
			return Err(Error::Config(
				"Gemma3 layer_types length must equal num_hidden_layers".to_string(),
			));
		}

		let query_pre_attn_scalar = get_f32_nullable(cfg, "query_pre_attn_scalar", 256.0)?;
		if query_pre_attn_scalar <= 0.0 {
			return Err(Error::Config(
				"Gemma3 query_pre_attn_scalar must be positive".to_string(),
			));
		}

		// Linear position scaling on the global (full-attention) rope.
		// Newer configs carry it under `rope_parameters.full_attention`,
		// raw-HF configs under `rope_scaling`; both are frequently a
		// literal `null`. Thetas always come from the flat keys.
		let rope_params = non_null(cfg, "rope_parameters");
		if let Some(value) = rope_params
			&& !value.is_object()
		{
			return Err(Error::Config(
				"field 'rope_parameters' must be an object".to_string(),
			));
		}
		let scaling = rope_params
			.and_then(|params| non_null(params, "full_attention"))
			.or_else(|| non_null(cfg, "rope_scaling"));
		let rope_global_scale = match scaling {
			None => 1.0,
			Some(value) => {
				if !value.is_object() {
					return Err(Error::Config(
						"Gemma3 rope scaling parameters must be an object".to_string(),
					));
				}
				let rope_type = match get_str(value, "rope_type")? {
					Some(kind) => Some(kind),
					None => get_str(value, "type")?,
				};
				match rope_type {
					Some("linear") => {
						let factor = get_f32(value, "factor", 1.0)?;
						if factor <= 0.0 {
							return Err(Error::Config(
								"Gemma3 rope scaling factor must be positive".to_string(),
							));
						}
						1.0 / factor
					}
					// An unscaled entry ("default") carries no factor.
					None | Some("default") => 1.0,
					Some(other) => {
						return Err(Error::Config(format!(
							"unsupported Gemma3 rope_type '{other}'"
						)));
					}
				}
			}
		};

		Ok(Gemma3Config {
			hidden_size,
			num_hidden_layers,
			num_attention_heads: require_i32(cfg, "num_attention_heads")?,
			num_key_value_heads: get_i32(cfg, "num_key_value_heads", 4)?,
			head_dim: get_i32(cfg, "head_dim", 256)?,
			rms_norm_eps: get_f32(cfg, "rms_norm_eps", 1e-6)?,
			vocab_size: require_i32(cfg, "vocab_size")?,
			attention_scale: query_pre_attn_scalar.powf(-0.5),
			sliding_window: get_i32(cfg, "sliding_window", 1024)?,
			final_logit_softcapping: optional_f32_nullable(cfg, "final_logit_softcapping")?,
			tie_word_embeddings: get_bool(cfg, "tie_word_embeddings", true)?,
			layer_types,
			rope_theta: get_f32(cfg, "rope_theta", 1_000_000.0)?,
			rope_local_base_freq: get_f32(cfg, "rope_local_base_freq", 10_000.0)?,
			rope_global_scale,
		})
	}
}

/// Gemma RMSNorm stores its scale as a zero-centered delta: the
/// effective scale is `1 + weight` (HF `Gemma3RMSNorm`, mlx-lm
/// `rms_norm(x, 1.0 + w, eps)`). Fold the offset once at load so the
/// fused rms_norm kernel applies the correct scale; the fold preserves
/// the checkpoint dtype.
fn gemma_rms_norm(w: &mut WeightMap, path: &str, eps: f32) -> Result<RmsNorm> {
	let norm = w.rms_norm(path, eps)?;
	let one = ops::astype(&Array::from_slice(&[1.0_f32], &[1])?, norm.weight.dtype())?;
	Ok(RmsNorm {
		weight: ops::add(&norm.weight, &one)?,
		eps,
	})
}

/// Full-dimension rotary embedding with an optional linear position
/// scale (`scale = 1/factor`; 1.0 leaves positions untouched).
struct Rope {
	dims: i32,
	theta: f32,
	scale: f32,
}

impl Rope {
	fn apply(&self, x: &Array, offset: i32) -> Result<Array> {
		ops::rope(
			x,
			self.dims,
			false,
			Some(self.theta),
			self.scale,
			offset,
			None,
		)
	}
}

struct Attention {
	q_proj: Linear,
	k_proj: Linear,
	v_proj: Linear,
	o_proj: Linear,
	q_norm: RmsNorm,
	k_norm: RmsNorm,
	rope: Rope,
	n_heads: i32,
	n_kv_heads: i32,
	head_dim: i32,
	is_sliding: bool,
	sliding_window: i32,
	scale: f32,
}

impl Attention {
	fn load(w: &mut WeightMap, prefix: &str, cfg: &Gemma3Config, layer_idx: i32) -> Result<Self> {
		let attn = format!("{prefix}.self_attn");
		let layer_type = cfg.layer_types[layer_idx as usize];
		let is_sliding = layer_type == LayerType::Sliding;

		let rope = if layer_type == LayerType::Full {
			Rope {
				dims: cfg.head_dim,
				theta: cfg.rope_theta,
				scale: cfg.rope_global_scale,
			}
		} else {
			Rope {
				dims: cfg.head_dim,
				theta: cfg.rope_local_base_freq,
				scale: 1.0,
			}
		};

		Ok(Attention {
			q_proj: w.linear(&format!("{attn}.q_proj"))?,
			k_proj: w.linear(&format!("{attn}.k_proj"))?,
			v_proj: w.linear(&format!("{attn}.v_proj"))?,
			o_proj: w.linear(&format!("{attn}.o_proj"))?,
			q_norm: gemma_rms_norm(w, &format!("{attn}.q_norm"), cfg.rms_norm_eps)?,
			k_norm: gemma_rms_norm(w, &format!("{attn}.k_norm"), cfg.rms_norm_eps)?,
			rope,
			n_heads: cfg.num_attention_heads,
			n_kv_heads: cfg.num_key_value_heads,
			head_dim: cfg.head_dim,
			is_sliding,
			sliding_window: cfg.sliding_window,
			scale: cfg.attention_scale,
		})
	}

	fn forward(&self, x: &Array, cache: &mut KvCache) -> Result<Array> {
		let shape = x.shape();
		let (b, l) = (shape[0], shape[1]);

		let q = self.q_proj.forward(x)?;
		let q = ops::reshape(&q, &[b, l, self.n_heads, self.head_dim])?;
		let q = self.q_norm.forward(&q)?;
		let q = ops::transpose_axes(&q, &[0, 2, 1, 3])?;

		let offset = cache.offset();
		let k = self.k_proj.forward(x)?;
		let k = ops::reshape(&k, &[b, l, self.n_kv_heads, self.head_dim])?;
		let k = self.k_norm.forward(&k)?;
		let k = ops::transpose_axes(&k, &[0, 2, 1, 3])?;
		let k = self.rope.apply(&k, offset)?;
		let v = self.v_proj.forward(x)?;
		let v = ops::reshape(&v, &[b, l, self.n_kv_heads, self.head_dim])?;
		let v = ops::transpose_axes(&v, &[0, 2, 1, 3])?;
		let (keys, values) = cache.update_and_fetch(k, v)?;
		let key_start = cache.start();

		let q = self.rope.apply(&q, offset)?;

		let kv_len = keys.dim(-2);
		let out = if self.is_sliding {
			// `key_start` is nonzero once the windowed cache trimmed.
			let mask = ops::sliding_window_mask(
				l,
				kv_len,
				offset,
				key_start,
				self.sliding_window,
				q.dtype(),
			)?;
			ops::scaled_dot_product_attention_masked(&q, &keys, &values, self.scale, &mask)?
		} else {
			let mask = if l == 1 {
				AttentionMask::None
			} else {
				AttentionMask::Causal
			};
			ops::scaled_dot_product_attention(&q, &keys, &values, self.scale, mask)?
		};
		let out = ops::transpose_axes(&out, &[0, 2, 1, 3])?;
		let out = ops::reshape(&out, &[b, l, -1])?;
		self.o_proj.forward(&out)
	}
}

struct Mlp {
	gate_proj: Linear,
	up_proj: Linear,
	down_proj: Linear,
}

impl Mlp {
	fn load(w: &mut WeightMap, prefix: &str) -> Result<Self> {
		let mlp = format!("{prefix}.mlp");
		Ok(Mlp {
			gate_proj: w.linear(&format!("{mlp}.gate_proj"))?,
			up_proj: w.linear(&format!("{mlp}.up_proj"))?,
			down_proj: w.linear(&format!("{mlp}.down_proj"))?,
		})
	}

	fn forward(&self, x: &Array) -> Result<Array> {
		let gate = ops::gelu_tanh(&self.gate_proj.forward(x)?)?;
		let up = self.up_proj.forward(x)?;
		self.down_proj.forward(&ops::multiply(&gate, &up)?)
	}
}

struct Block {
	self_attn: Attention,
	mlp: Mlp,
	input_layernorm: RmsNorm,
	post_attention_layernorm: RmsNorm,
	pre_feedforward_layernorm: RmsNorm,
	post_feedforward_layernorm: RmsNorm,
}

impl Block {
	fn load(w: &mut WeightMap, prefix: &str, cfg: &Gemma3Config, layer_idx: i32) -> Result<Self> {
		Ok(Block {
			self_attn: Attention::load(w, prefix, cfg, layer_idx)?,
			mlp: Mlp::load(w, prefix)?,
			input_layernorm: gemma_rms_norm(
				w,
				&format!("{prefix}.input_layernorm"),
				cfg.rms_norm_eps,
			)?,
			post_attention_layernorm: gemma_rms_norm(
				w,
				&format!("{prefix}.post_attention_layernorm"),
				cfg.rms_norm_eps,
			)?,
			pre_feedforward_layernorm: gemma_rms_norm(
				w,
				&format!("{prefix}.pre_feedforward_layernorm"),
				cfg.rms_norm_eps,
			)?,
			post_feedforward_layernorm: gemma_rms_norm(
				w,
				&format!("{prefix}.post_feedforward_layernorm"),
				cfg.rms_norm_eps,
			)?,
		})
	}

	fn forward(&self, x: &Array, cache: &mut KvCache) -> Result<Array> {
		let residual = x.clone();
		let h = self.input_layernorm.forward(x)?;
		let h = self.self_attn.forward(&h, cache)?;
		let h = self.post_attention_layernorm.forward(&h)?;
		let h = ops::add(&residual, &h)?;

		let residual = h.clone();
		let m = self.pre_feedforward_layernorm.forward(&h)?;
		let m = self.mlp.forward(&m)?;
		let m = self.post_feedforward_layernorm.forward(&m)?;
		ops::add(&residual, &m)
	}
}

/// Round to the nearest bfloat16-representable value (ties to even).
/// The reference implementation computes the embedding normalizer
/// `sqrt(hidden_size)` in bf16, so matching it exactly avoids a small
/// systematic scale divergence on every token.
fn bf16_round(value: f32) -> f32 {
	let bits = value.to_bits();
	let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
	f32::from_bits(rounded & 0xFFFF_0000)
}

/// Gemma 3 causal language model (text decoder only).
pub struct Gemma3Model {
	pub config: Gemma3Config,
	embed_tokens: Embedding,
	layers: Vec<Block>,
	norm: RmsNorm,
	lm_head: Option<Linear>,
	embed_scale: f32,
}

impl Gemma3Model {
	pub fn load(mut weights: WeightMap, config_json: &Value) -> Result<Self> {
		let cfg = Gemma3Config::from_json(config_json)?;
		let prefix = "language_model.model";

		let embed_tokens = weights.embedding(&format!("{prefix}.embed_tokens"))?;
		let mut layers = Vec::with_capacity(cfg.num_hidden_layers as usize);
		for i in 0..cfg.num_hidden_layers {
			layers.push(Block::load(
				&mut weights,
				&format!("{prefix}.layers.{i}"),
				&cfg,
				i,
			)?);
		}
		let norm = gemma_rms_norm(&mut weights, &format!("{prefix}.norm"), cfg.rms_norm_eps)?;
		let lm_head = if cfg.tie_word_embeddings {
			None
		} else {
			Some(weights.linear("language_model.lm_head")?)
		};
		let embed_scale = bf16_round((cfg.hidden_size as f32).sqrt());

		Ok(Gemma3Model {
			config: cfg,
			embed_tokens,
			layers,
			norm,
			lm_head,
			embed_scale,
		})
	}

	pub fn new_caches(&self) -> Vec<LayerCache> {
		// Sliding-window layers retain only their window (plus growth
		// slack) — their mask never attends further back (see
		// models/cache.rs).
		self.config
			.layer_types
			.iter()
			.map(|layer_type| match layer_type {
				LayerType::Sliding => {
					LayerCache::new_attention_windowed(self.config.sliding_window)
				}
				LayerType::Full => LayerCache::new_attention(),
			})
			.collect()
	}

	pub fn forward(&self, input_ids: &Array, caches: &mut [LayerCache]) -> Result<Array> {
		if caches.len() != self.layers.len() {
			return Err(Error::Model(format!(
				"gemma3: expected {} layer caches, got {}",
				self.layers.len(),
				caches.len()
			)));
		}
		let embeddings = self.embed_tokens.forward(input_ids)?;
		let mut h = ops::scale_by(&embeddings, self.embed_scale)?;
		for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
			h = layer.forward(&h, cache.as_attention()?)?;
		}
		h = self.norm.forward(&h)?;

		let mut logits = match &self.lm_head {
			Some(head) => head.forward(&h)?,
			None => self.embed_tokens.as_linear(&h)?,
		};
		if let Some(cap) = self.config.final_logit_softcapping {
			logits = ops::scale_by(&ops::tanh(&ops::scale_by(&logits, 1.0 / cap)?)?, cap)?;
		}
		Ok(logits)
	}
}

/// Canonicalize weight keys to the `language_model.model.*` prefix
/// multimodal conversions ship, and drop everything this text-only port
/// does not load (`vision_tower.*`, `multi_modal_projector.*`).
///
/// Bare-prefix `Gemma3ForCausalLM` checkpoints (`model.*` /
/// `lm_head.*`) are renamed under `language_model.`; the identical
/// mapping is applied to per-layer quantization-override keys
/// (`normalize_quant_keys`) so overrides keyed by checkpoint-verbatim
/// paths keep resolving after the rename.
pub fn sanitize(weights: &mut WeightMap) {
	let bare = weights.contains("model.embed_tokens.weight");
	let map = move |k: &str| -> Option<String> {
		if bare {
			if k.starts_with("model.") || k.starts_with("lm_head.") {
				return Some(format!("language_model.{k}"));
			}
		} else if k.starts_with("language_model.") {
			return Some(k.to_string());
		}
		None
	};
	weights.rename_keys(&map);
	weights.normalize_quant_keys(&map);
}

#[cfg(test)]
mod tests {
	use std::collections::HashMap;

	use serde_json::json;

	use super::*;
	use crate::engine::quant::Quantization;

	/// Key subset of the real `mlx-community/translategemma-27b-it-8bit`
	/// `config.json` (verified August 2026), truncated from 62 to 12
	/// layers. Pins the exact spellings the parser depends on: nested
	/// `text_config`, flat rope thetas, `rope_parameters` carrying only
	/// the linear factor, literal-null retired fields, and the
	/// underscored pattern key.
	#[test]
	fn config_parses_translategemma_27b_text_config_subset() {
		let config = json!({
			"architectures": ["Gemma3ForConditionalGeneration"],
			"model_type": "gemma3",
			"boi_token_index": 255_999,
			"eoi_token_index": 256_000,
			"image_token_index": 262_144,
			"text_config": {
				"_sliding_window_pattern": 6,
				"attn_logit_softcapping": null,
				"final_logit_softcapping": null,
				"head_dim": 128,
				"hidden_size": 5376,
				"intermediate_size": 21_504,
				"max_position_embeddings": 131_072,
				"model_type": "gemma3_text",
				"num_attention_heads": 32,
				"num_hidden_layers": 12,
				"num_key_value_heads": 16,
				"query_pre_attn_scalar": 168,
				"rms_norm_eps": 1e-6,
				"rope_local_base_freq": 10_000,
				"rope_parameters": {
					"full_attention": {"factor": 8.0, "rope_type": "linear"},
					"sliding_attention": {"rope_type": "default"}
				},
				"rope_scaling": null,
				"rope_theta": 1_000_000,
				"sliding_window": 1024,
				"vocab_size": 262_208,
				"layer_types": [
					"sliding_attention", "sliding_attention", "sliding_attention",
					"sliding_attention", "sliding_attention", "full_attention",
					"sliding_attention", "sliding_attention", "sliding_attention",
					"sliding_attention", "sliding_attention", "full_attention"
				]
			}
		});
		let cfg = Gemma3Config::from_json(&config).unwrap();
		assert_eq!(cfg.hidden_size, 5376);
		assert_eq!(cfg.num_hidden_layers, 12);
		assert_eq!(cfg.num_attention_heads, 32);
		assert_eq!(cfg.num_key_value_heads, 16);
		assert_eq!(cfg.head_dim, 128);
		assert_eq!(cfg.vocab_size, 262_208);
		assert_eq!(cfg.sliding_window, 1024);
		assert_eq!(cfg.attention_scale, 168f32.powf(-0.5));
		assert_eq!(cfg.rope_theta, 1_000_000.0);
		assert_eq!(cfg.rope_local_base_freq, 10_000.0);
		assert_eq!(cfg.rope_global_scale, 0.125);
		assert_eq!(cfg.final_logit_softcapping, None);
		assert!(cfg.tie_word_embeddings);
		for (index, layer_type) in cfg.layer_types.iter().enumerate() {
			let expected = if (index + 1) % 6 == 0 {
				LayerType::Full
			} else {
				LayerType::Sliding
			};
			assert_eq!(*layer_type, expected, "layer {index}");
		}
	}

	/// gemma-3-1b-style flat text config: no `rope_parameters`, no
	/// `rope_scaling`, no `layer_types` — the factor must default to 1.0
	/// and the layer mix must derive from the bare pattern key.
	#[test]
	fn config_parses_1b_style_without_rope_scaling() {
		let config = json!({
			"model_type": "gemma3_text",
			"hidden_size": 1152,
			"num_hidden_layers": 12,
			"num_attention_heads": 4,
			"num_key_value_heads": 1,
			"head_dim": 256,
			"query_pre_attn_scalar": 256,
			"rope_local_base_freq": 10_000,
			"rope_theta": 1_000_000,
			"sliding_window": 512,
			"sliding_window_pattern": 6,
			"vocab_size": 262_144
		});
		let cfg = Gemma3Config::from_json(&config).unwrap();
		assert_eq!(cfg.rope_global_scale, 1.0);
		assert_eq!(cfg.attention_scale, 256f32.powf(-0.5));
		for (index, layer_type) in cfg.layer_types.iter().enumerate() {
			let expected = if (index + 1) % 6 == 0 {
				LayerType::Full
			} else {
				LayerType::Sliding
			};
			assert_eq!(*layer_type, expected, "layer {index}");
		}
	}

	#[test]
	fn layer_type_fallback_honors_underscored_pattern_key() {
		let config = json!({
			"model_type": "gemma3_text",
			"hidden_size": 32,
			"num_hidden_layers": 4,
			"num_attention_heads": 2,
			"num_key_value_heads": 1,
			"head_dim": 16,
			"vocab_size": 16,
			"_sliding_window_pattern": 2
		});
		let cfg = Gemma3Config::from_json(&config).unwrap();
		assert_eq!(
			cfg.layer_types,
			vec![
				LayerType::Sliding,
				LayerType::Full,
				LayerType::Sliding,
				LayerType::Full
			]
		);
	}

	#[test]
	fn config_rejects_invalid_inputs() {
		let base = || {
			json!({
				"model_type": "gemma3_text",
				"hidden_size": 32,
				"num_hidden_layers": 2,
				"num_attention_heads": 2,
				"num_key_value_heads": 1,
				"head_dim": 16,
				"vocab_size": 16
			})
		};

		let mut config = base();
		config["layer_types"] = json!(["sliding_attention", "linear_attention"]);
		assert!(Gemma3Config::from_json(&config).is_err());

		let mut config = base();
		config["layer_types"] = json!("sliding_attention");
		assert!(Gemma3Config::from_json(&config).is_err());

		let mut config = base();
		config["layer_types"] = json!(["sliding_attention"]);
		assert!(Gemma3Config::from_json(&config).is_err());

		let mut config = base();
		config["rope_scaling"] = json!({"rope_type": "yarn", "factor": 8.0});
		assert!(Gemma3Config::from_json(&config).is_err());

		let mut config = base();
		config["rope_scaling"] = json!({"rope_type": "linear", "factor": 0.0});
		assert!(Gemma3Config::from_json(&config).is_err());

		let mut config = base();
		config["rope_parameters"] = json!("invalid");
		assert!(Gemma3Config::from_json(&config).is_err());

		let mut config = base();
		config["query_pre_attn_scalar"] = json!(0.0);
		assert!(Gemma3Config::from_json(&config).is_err());

		let mut config = base();
		config["_sliding_window_pattern"] = json!(0);
		assert!(Gemma3Config::from_json(&config).is_err());
	}

	#[test]
	fn gemma_rms_norm_folds_the_unit_offset() {
		// Gemma checkpoints store norm scales as zero-centered deltas; the
		// effective scale is `1 + weight`. An unfolded weight collapses
		// every activation toward zero and garbles generation.
		let tensors = HashMap::from([(
			"norm.weight".to_string(),
			Array::from_slice(&[0.0_f32, -0.5, 0.5], &[3]).unwrap(),
		)]);
		let mut weights = WeightMap::new(tensors, Quantization::default());
		let norm = gemma_rms_norm(&mut weights, "norm", 1e-6).unwrap();
		assert_eq!(norm.weight.to_vec_f32().unwrap(), vec![1.0, 0.5, 1.5]);
	}

	#[test]
	fn embed_scale_rounds_through_bf16() {
		// sqrt(5376) = 73.3212...; the nearest bf16 value is 73.5.
		assert_eq!(bf16_round(5376f32.sqrt()), 73.5);
		// Exactly representable values pass through unchanged.
		assert_eq!(bf16_round(64.0), 64.0);
		assert_eq!(bf16_round(73.5), 73.5);
		// Round-to-nearest-even at the midpoint between 73.0 and 73.5.
		assert_eq!(bf16_round(73.25), 73.0);
	}

	// ------------------------------------------------------------------
	// Tiny synthetic end-to-end fixtures: 2 layers (one sliding + one
	// full), hidden 8, 2 heads / 1 KV head, head_dim 4, vocab 16 — the
	// only coverage that exercises sanitize + both checkpoint prefixes +
	// the tied head against real kernels.
	// ------------------------------------------------------------------

	/// Deterministic, varied filler values so op-order comparisons are
	/// meaningful (all-equal weights would mask permutation bugs).
	fn arr(seed: f32, shape: &[i32]) -> Array {
		let len: usize = shape.iter().map(|&d| d as usize).product();
		let data: Vec<f32> = (0..len)
			.map(|i| ((i as f32) * 0.7311 + seed).sin() * 0.05)
			.collect();
		Array::from_slice(&data, shape).unwrap()
	}

	fn tiny_tensor_specs() -> Vec<(String, Vec<i32>)> {
		let mut specs = vec![
			("model.embed_tokens.weight".to_string(), vec![16, 8]),
			("model.norm.weight".to_string(), vec![8]),
		];
		for layer in 0..2 {
			let prefix = format!("model.layers.{layer}");
			for (name, shape) in [
				("input_layernorm.weight", vec![8]),
				("post_attention_layernorm.weight", vec![8]),
				("pre_feedforward_layernorm.weight", vec![8]),
				("post_feedforward_layernorm.weight", vec![8]),
				("self_attn.q_proj.weight", vec![8, 8]),
				("self_attn.k_proj.weight", vec![4, 8]),
				("self_attn.v_proj.weight", vec![4, 8]),
				("self_attn.o_proj.weight", vec![8, 8]),
				("self_attn.q_norm.weight", vec![4]),
				("self_attn.k_norm.weight", vec![4]),
				("mlp.gate_proj.weight", vec![16, 8]),
				("mlp.up_proj.weight", vec![16, 8]),
				("mlp.down_proj.weight", vec![8, 16]),
			] {
				specs.push((format!("{prefix}.{name}"), shape));
			}
		}
		specs
	}

	fn tiny_tensors(prefix: &str) -> HashMap<String, Array> {
		tiny_tensor_specs()
			.into_iter()
			.enumerate()
			.map(|(i, (key, shape))| (format!("{prefix}{key}"), arr(i as f32, &shape)))
			.collect()
	}

	fn tiny_config() -> Value {
		json!({
			"model_type": "gemma3_text",
			"hidden_size": 8,
			"num_hidden_layers": 2,
			"num_attention_heads": 2,
			"num_key_value_heads": 1,
			"head_dim": 4,
			"query_pre_attn_scalar": 4.0,
			"rms_norm_eps": 1e-6,
			"sliding_window": 4,
			"vocab_size": 16,
			"layer_types": ["sliding_attention", "full_attention"]
		})
	}

	fn load_tiny(tensors: HashMap<String, Array>) -> Gemma3Model {
		let mut weights = WeightMap::new(tensors, Quantization::default());
		sanitize(&mut weights);
		Gemma3Model::load(weights, &tiny_config()).expect("load")
	}

	fn forward_logits(model: &Gemma3Model) -> Vec<f32> {
		let mut caches = model.new_caches();
		let prompt = Array::from_slice(&[1i32, 2, 3], &[1, 3]).unwrap();
		let logits = model.forward(&prompt, &mut caches).expect("forward");
		assert_eq!(logits.shape(), vec![1, 3, 16]);
		// A second, cached single-token step must also work.
		let next = Array::from_slice(&[4i32], &[1, 1]).unwrap();
		let step = model.forward(&next, &mut caches).expect("cached step");
		assert_eq!(step.shape(), vec![1, 1, 16]);
		let mut values = logits.to_vec_f32().expect("prompt logits");
		values.extend(step.to_vec_f32().expect("step logits"));
		assert!(values.iter().all(|v| v.is_finite()));
		values
	}

	#[test]
	fn loads_and_forwards_tiny_language_model_prefix() {
		let mut tensors = tiny_tensors("language_model.");
		// Multimodal leftovers must be dropped by sanitize, not loaded.
		tensors.insert(
			"vision_tower.encoder.weight".to_string(),
			arr(99.0, &[4, 4]),
		);
		tensors.insert(
			"multi_modal_projector.weight".to_string(),
			arr(98.0, &[4, 4]),
		);
		let model = load_tiny(tensors);
		assert_eq!(model.new_caches().len(), 2);
		forward_logits(&model);
	}

	#[test]
	fn loads_and_forwards_tiny_bare_prefix() {
		// The same tensors under the bare `Gemma3ForCausalLM` prefix must
		// produce bit-identical logits — the rename is semantics-preserving.
		let canonical = forward_logits(&load_tiny(tiny_tensors("language_model.")));
		let renamed = forward_logits(&load_tiny(tiny_tensors("")));
		assert_eq!(canonical, renamed);
	}

	#[test]
	fn new_caches_matches_layer_types() {
		let model = load_tiny(tiny_tensors("language_model."));
		let mut caches = model.new_caches();
		assert_eq!(caches.len(), 2);
		let windows: Vec<Option<i32>> = caches
			.iter_mut()
			.map(|cache| cache.as_attention().expect("attention cache").window())
			.collect();
		assert_eq!(windows, vec![Some(4), None]);
	}
}
