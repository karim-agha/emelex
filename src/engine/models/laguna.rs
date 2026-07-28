//! Laguna (Poolside Laguna M.1 / S-2.1) architecture: MoE with sigmoid
//! top-k routing plus an always-on shared expert, GQA with per-head q/k
//! RMSNorm, softplus attention output gating, and YaRN-scaled RoPE.
//!
//! One module serves the whole family; every difference is config-driven:
//! - M.1: 70 layers (3 dense + 67 sparse), all full attention, uniform 64 query
//!   heads, per-element gating, full-rotary YaRN.
//! - S-2.1: 48 layers (1 dense + 47 sparse), 1:3 full/sliding-window interleave
//!   with a separate rope per layer type (partial-rotary YaRN with a non-unit
//!   attention factor on full layers, plain rope on sliding layers), per-layer
//!   query-head counts, per-head gating.
//!
//! Forward math mirrors the reference `mlx_lm/models/laguna.py` and
//! Poolside's `modeling_laguna.py`.

use serde_json::Value;

use super::{
	base::{RopeConfig, YarnRope, attention_mask_for, merge_heads, split_heads},
	cache::{KvCache, LayerCache},
	config::{get_bool, get_f32, get_i32, get_str, require_i32},
	moe::{SharedExpert, SwitchGlu},
};
use crate::engine::{
	array::{Array, Dtype},
	error::{Error, Result},
	nn::{Embedding, Linear, RmsNorm, WeightMap},
	ops,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
	Full,
	Sliding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatingKind {
	/// Gate has one value per output element (`n_heads * head_dim`).
	PerElement,
	/// Gate has one value per head, broadcast over `head_dim`.
	PerHead,
}

/// One layer type's rope parameters as declared in the config.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeParams {
	pub yarn: bool,
	pub theta: f32,
	pub partial_rotary_factor: f32,
	pub factor: f32,
	pub original_max_position_embeddings: i32,
	pub beta_fast: f32,
	pub beta_slow: f32,
	pub attention_factor: f32,
}

impl RopeParams {
	/// Parse one rope dict (`rope_parameters.full_attention`-style or the
	/// legacy top-level `rope_scaling`). `attention_factor` defaults to
	/// the HF yarn convention `0.1 * ln(factor) + 1` when unspecified.
	fn from_dict(d: &Value, default_theta: f32, default_partial: f32) -> Result<Self> {
		if !d.is_object() {
			return Err(Error::Config(
				"Laguna RoPE parameters must be an object".to_string(),
			));
		}
		let rope_type = match get_str(d, "rope_type")? {
			Some(value) => value,
			None => get_str(d, "type")?.unwrap_or("default"),
		};
		let yarn = match rope_type {
			"yarn" => true,
			"default" => false,
			other => {
				return Err(Error::Config(format!(
					"unsupported laguna rope_type '{other}' (expected 'yarn' or \
					 'default')"
				)));
			}
		};
		let factor = get_f32(d, "factor", 1.0)?;
		Ok(RopeParams {
			yarn,
			theta: get_f32(d, "rope_theta", default_theta)?,
			partial_rotary_factor: get_f32(d, "partial_rotary_factor", default_partial)?,
			factor,
			original_max_position_embeddings: get_i32(d, "original_max_position_embeddings", 4096)?,
			beta_fast: get_f32(d, "beta_fast", 32.0)?,
			beta_slow: get_f32(d, "beta_slow", 1.0)?,
			attention_factor: get_f32(d, "attention_factor", 0.1 * factor.ln() + 1.0)?,
		})
	}

	fn build(&self, head_dim: i32) -> Result<Rope> {
		let dims = ((head_dim as f32) * self.partial_rotary_factor) as i32;
		if self.yarn {
			Ok(Rope::Yarn(YarnRope::new(
				dims,
				head_dim,
				self.theta,
				self.factor,
				self.original_max_position_embeddings,
				self.beta_fast,
				self.beta_slow,
				self.attention_factor,
			)?))
		} else {
			Ok(Rope::Standard(RopeConfig::new(dims, self.theta)))
		}
	}
}

#[derive(Clone)]
enum Rope {
	Standard(RopeConfig),
	Yarn(YarnRope),
}

impl Rope {
	fn apply(&self, x: &Array, offset: i32) -> Result<Array> {
		match self {
			Rope::Standard(r) => r.apply(x, offset),
			Rope::Yarn(r) => r.apply(x, offset),
		}
	}
}

#[derive(Debug, Clone)]
pub struct LagunaConfig {
	pub hidden_size: i32,
	pub num_hidden_layers: i32,
	pub intermediate_size: i32,
	pub num_key_value_heads: i32,
	pub head_dim: i32,
	pub rms_norm_eps: f32,
	pub vocab_size: i32,
	pub tie_word_embeddings: bool,
	/// Query-head count per layer (uniform unless the config carries
	/// `num_attention_heads_per_layer`).
	pub heads_per_layer: Vec<i32>,
	/// Full vs sliding attention per layer (`layer_types`; all-full when
	/// absent, as in M.1).
	pub layer_kinds: Vec<LayerKind>,
	pub sliding_window: i32,
	/// Attention output gating kind per layer (`gating_types`, falling
	/// back to the global `gating` key).
	pub gating_per_layer: Vec<GatingKind>,
	pub full_rope: RopeParams,
	pub sliding_rope: RopeParams,
	// MoE
	pub num_experts: i32,
	pub num_experts_per_tok: i32,
	pub moe_intermediate_size: i32,
	pub shared_expert_intermediate_size: i32,
	pub norm_topk_prob: bool,
	pub moe_routed_scaling_factor: f32,
	/// `true` = sparse (MoE) layer, from `mlp_layer_types`.
	pub sparse_layers: Vec<bool>,
}

fn parse_gating_kind(s: &str) -> Result<GatingKind> {
	match s {
		"per-element" | "per_element" => Ok(GatingKind::PerElement),
		"per-head" | "per_head" => Ok(GatingKind::PerHead),
		other => Err(Error::Config(format!(
			"unsupported laguna gating kind '{other}'"
		))),
	}
}

impl LagunaConfig {
	pub fn from_json(cfg: &Value) -> Result<Self> {
		let hidden_size = require_i32(cfg, "hidden_size")?;
		let num_hidden_layers = require_i32(cfg, "num_hidden_layers")?;
		let num_attention_heads = require_i32(cfg, "num_attention_heads")?;
		let head_dim = get_i32(cfg, "head_dim", hidden_size / num_attention_heads)?;
		let layers = num_hidden_layers as usize;

		let heads_per_layer = match cfg.get("num_attention_heads_per_layer") {
			Some(Value::Array(v)) => {
				let heads: Vec<i32> = v
					.iter()
					.enumerate()
					.map(|(index, head)| {
						let value = head.as_i64().ok_or_else(|| {
							Error::Config(format!(
								"num_attention_heads_per_layer[{index}] must be an integer"
							))
						})?;
						i32::try_from(value).map_err(|_| {
							Error::Config(format!(
								"num_attention_heads_per_layer[{index}] is outside i32"
							))
						})
					})
					.collect::<Result<_>>()?;
				if heads.len() != layers {
					return Err(Error::Config(
						"num_attention_heads_per_layer length does not match \
						 num_hidden_layers"
							.to_string(),
					));
				}
				heads
			}
			Some(_) => {
				return Err(Error::Config(
					"num_attention_heads_per_layer must be an array".to_string(),
				));
			}
			None => vec![num_attention_heads; layers],
		};

		let layer_kinds = match cfg.get("layer_types") {
			Some(Value::Array(v)) => {
				let kinds: Vec<LayerKind> = v
					.iter()
					.map(|t| match t.as_str() {
						Some("full_attention") => Ok(LayerKind::Full),
						Some("sliding_attention") => Ok(LayerKind::Sliding),
						other => Err(Error::Config(format!(
							"unsupported laguna layer type {other:?}"
						))),
					})
					.collect::<Result<_>>()?;
				if kinds.len() != layers {
					return Err(Error::Config(
						"layer_types length does not match num_hidden_layers".to_string(),
					));
				}
				kinds
			}
			Some(_) => {
				return Err(Error::Config("layer_types must be an array".to_string()));
			}
			None => vec![LayerKind::Full; layers],
		};

		let gating_per_layer = match cfg.get("gating_types") {
			Some(Value::Array(v)) => {
				let kinds: Vec<GatingKind> = v
					.iter()
					.map(|t| parse_gating_kind(t.as_str().unwrap_or_default()))
					.collect::<Result<_>>()?;
				if kinds.len() != layers {
					return Err(Error::Config(
						"gating_types length does not match num_hidden_layers".to_string(),
					));
				}
				kinds
			}
			Some(_) => {
				return Err(Error::Config("gating_types must be an array".to_string()));
			}
			None => {
				let kind = parse_gating_kind(get_str(cfg, "gating")?.unwrap_or("per-element"))?;
				vec![kind; layers]
			}
		};

		// `mlp_layer_types` drives which weights exist per layer; a wrong
		// guess would fail the load with a confusing missing-tensor error,
		// so require it explicitly.
		let sparse_layers: Vec<bool> = match cfg.get("mlp_layer_types") {
			Some(Value::Array(v)) => v
				.iter()
				.map(|t| match t.as_str() {
					Some("sparse") => Ok(true),
					Some("dense") => Ok(false),
					other => Err(Error::Config(format!(
						"unsupported laguna mlp layer type {other:?}"
					))),
				})
				.collect::<Result<_>>()?,
			Some(_) => {
				return Err(Error::Config(
					"laguna 'mlp_layer_types' must be an array".to_string(),
				));
			}
			None => {
				return Err(Error::Config(
					"laguna config is missing 'mlp_layer_types'".to_string(),
				));
			}
		};
		if sparse_layers.len() != layers {
			return Err(Error::Config(
				"mlp_layer_types length does not match num_hidden_layers".to_string(),
			));
		}

		// Rope config comes either as per-layer-type dicts under
		// `rope_parameters` (S-2.1) or as a single top-level `rope_scaling`
		// dict plus `rope_theta` (M.1; its `rope_parameters.full_attention`
		// duplicates `rope_scaling`, so both spellings resolve identically).
		let default_theta = get_f32(cfg, "rope_theta", 500_000.0)?;
		let default_partial = get_f32(cfg, "partial_rotary_factor", 1.0)?;
		let rope_parameters = cfg
			.get("rope_parameters")
			.map(|value| {
				value.as_object().ok_or_else(|| {
					Error::Config("field 'rope_parameters' must be an object".to_string())
				})?;
				Ok::<&Value, Error>(value)
			})
			.transpose()?;
		let full_dict = rope_parameters
			.and_then(|r| r.get("full_attention"))
			.or_else(|| cfg.get("rope_scaling"));
		let full_rope = match full_dict {
			Some(d) => RopeParams::from_dict(d, default_theta, default_partial)?,
			None => RopeParams::from_dict(
				&Value::Object(Default::default()),
				default_theta,
				default_partial,
			)?,
		};
		let sliding_rope = match rope_parameters.and_then(|r| r.get("sliding_attention")) {
			Some(d) => RopeParams::from_dict(d, default_theta, 1.0)?,
			None => full_rope,
		};

		Ok(LagunaConfig {
			hidden_size,
			num_hidden_layers,
			intermediate_size: require_i32(cfg, "intermediate_size")?,
			num_key_value_heads: get_i32(cfg, "num_key_value_heads", num_attention_heads)?,
			head_dim,
			rms_norm_eps: get_f32(cfg, "rms_norm_eps", 1e-6)?,
			vocab_size: require_i32(cfg, "vocab_size")?,
			tie_word_embeddings: get_bool(cfg, "tie_word_embeddings", false)?,
			heads_per_layer,
			layer_kinds,
			sliding_window: get_i32(cfg, "sliding_window", 512)?,
			gating_per_layer,
			full_rope,
			sliding_rope,
			num_experts: require_i32(cfg, "num_experts")?,
			num_experts_per_tok: require_i32(cfg, "num_experts_per_tok")?,
			moe_intermediate_size: require_i32(cfg, "moe_intermediate_size")?,
			shared_expert_intermediate_size: require_i32(cfg, "shared_expert_intermediate_size")?,
			norm_topk_prob: get_bool(cfg, "norm_topk_prob", true)?,
			moe_routed_scaling_factor: get_f32(cfg, "moe_routed_scaling_factor", 1.0)?,
			sparse_layers,
		})
	}
}

struct Attention {
	q_proj: Linear,
	k_proj: Linear,
	v_proj: Linear,
	o_proj: Linear,
	/// Softplus output-gate projection (Laguna-specific).
	g_proj: Linear,
	q_norm: RmsNorm,
	k_norm: RmsNorm,
	rope: Rope,
	n_heads: i32,
	n_kv_heads: i32,
	head_dim: i32,
	scale: f32,
	gating: GatingKind,
	kind: LayerKind,
	sliding_window: i32,
}

impl Attention {
	fn load(
		w: &mut WeightMap,
		prefix: &str,
		cfg: &LagunaConfig,
		layer_idx: usize,
		full_rope: &Rope,
		sliding_rope: &Rope,
	) -> Result<Self> {
		let attn = format!("{prefix}.self_attn");
		let kind = cfg.layer_kinds[layer_idx];
		let rope = match kind {
			LayerKind::Full => full_rope.clone(),
			LayerKind::Sliding => sliding_rope.clone(),
		};
		Ok(Attention {
			q_proj: w.linear(&format!("{attn}.q_proj"))?,
			k_proj: w.linear(&format!("{attn}.k_proj"))?,
			v_proj: w.linear(&format!("{attn}.v_proj"))?,
			o_proj: w.linear(&format!("{attn}.o_proj"))?,
			g_proj: w.linear(&format!("{attn}.g_proj"))?,
			q_norm: w.rms_norm(&format!("{attn}.q_norm"), cfg.rms_norm_eps)?,
			k_norm: w.rms_norm(&format!("{attn}.k_norm"), cfg.rms_norm_eps)?,
			rope,
			n_heads: cfg.heads_per_layer[layer_idx],
			n_kv_heads: cfg.num_key_value_heads,
			head_dim: cfg.head_dim,
			scale: (cfg.head_dim as f32).powf(-0.5),
			gating: cfg.gating_per_layer[layer_idx],
			kind,
			sliding_window: cfg.sliding_window,
		})
	}

	fn forward(&self, x: &Array, cache: &mut KvCache) -> Result<Array> {
		let shape = x.shape();
		let (b, l) = (shape[0], shape[1]);

		let q = self.q_proj.forward(x)?;
		let k = self.k_proj.forward(x)?;
		let v = self.v_proj.forward(x)?;

		let q = split_heads(&q, b, l, self.n_heads)?;
		let k = split_heads(&k, b, l, self.n_kv_heads)?;
		let v = split_heads(&v, b, l, self.n_kv_heads)?;

		// Per-head RMSNorm on q/k before rope, per the reference.
		let q = self.q_norm.forward(&q)?;
		let k = self.k_norm.forward(&k)?;

		let offset = cache.offset();
		let q = self.rope.apply(&q, offset)?;
		let k = self.rope.apply(&k, offset)?;
		let (k, v) = cache.update_and_fetch(k, v)?;

		let out = match self.kind {
			LayerKind::Sliding => {
				let kv_len = k.dim(-2);
				let mask = ops::sliding_window_mask(
					l,
					kv_len,
					offset,
					cache.start(),
					self.sliding_window,
					q.dtype(),
				)?;
				ops::scaled_dot_product_attention_masked(&q, &k, &v, self.scale, &mask)?
			}
			LayerKind::Full => {
				ops::scaled_dot_product_attention(&q, &k, &v, self.scale, attention_mask_for(l))?
			}
		};
		let out = merge_heads(&out, b, l)?;

		// Softplus output gating (in f32 for precision, per the
		// reference), applied before o_proj.
		let gate = self.g_proj.forward(x)?;
		let gate = ops::softplus(&ops::astype(&gate, Dtype::Float32)?)?;
		let gate = ops::astype(&gate, out.dtype())?;
		let out = match self.gating {
			GatingKind::PerElement => ops::multiply(&out, &gate)?,
			GatingKind::PerHead => {
				let out4 = ops::reshape(&out, &[b, l, self.n_heads, self.head_dim])?;
				let gate4 = ops::expand_dims(&gate, -1)?;
				let gated = ops::multiply(&out4, &gate4)?;
				ops::reshape(&gated, &[b, l, -1])?
			}
		};

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
		Ok(Mlp {
			gate_proj: w.linear(&format!("{prefix}.gate_proj"))?,
			up_proj: w.linear(&format!("{prefix}.up_proj"))?,
			down_proj: w.linear(&format!("{prefix}.down_proj"))?,
		})
	}

	fn forward(&self, x: &Array) -> Result<Array> {
		let gate = ops::silu(&self.gate_proj.forward(x)?)?;
		let up = self.up_proj.forward(x)?;
		self.down_proj.forward(&ops::multiply(&gate, &up)?)
	}
}

/// Sigmoid top-k router. The correction bias participates in expert
/// *selection* only; the routing weights are the uncorrected sigmoid
/// scores gathered at the selected experts (then renormalized) — the
/// subtlest rule in the architecture, mirrored from the reference.
struct LagunaRouter {
	gate: Linear,
	/// `[num_experts]`, kept in f32 alongside the sigmoid scores.
	/// `None` when the checkpoint carries no bias — the reference
	/// initializes it to zeros, so absence means "no correction".
	e_score_correction_bias: Option<Array>,
	top_k: i32,
	norm_topk_prob: bool,
	routed_scaling_factor: f32,
}

impl LagunaRouter {
	fn load(w: &mut WeightMap, prefix: &str, cfg: &LagunaConfig) -> Result<Self> {
		// Conversions disagree on where the bias lives: ox-ox's M.1
		// stores `mlp.gate.e_score_correction_bias`, pipenetwork's
		// S-2.1 `mlp.e_score_correction_bias` — and it may be absent
		// entirely (zero-initialized in the reference).
		let bias = w
			.take_optional(&format!("{prefix}.gate.e_score_correction_bias"))
			.or_else(|| w.take_optional(&format!("{prefix}.e_score_correction_bias")));
		let bias = match bias {
			Some(bias) => Some(ops::astype(&bias, Dtype::Float32)?),
			None => None,
		};
		Ok(LagunaRouter {
			gate: w.linear(&format!("{prefix}.gate"))?,
			e_score_correction_bias: bias,
			top_k: cfg.num_experts_per_tok,
			norm_topk_prob: cfg.norm_topk_prob,
			routed_scaling_factor: cfg.moe_routed_scaling_factor,
		})
	}

	/// Returns `(indices [B, S, K], weights [B, S, K] in x's dtype)`.
	fn route(&self, x: &Array) -> Result<(Array, Array)> {
		let xg = ops::astype(x, self.gate.weight_dtype(x.dtype()))?;
		let logits = self.gate.forward(&xg)?;
		// Sigmoid scoring in f32: bf16 logits would perturb the top-k
		// margin across 256 experts.
		let scores = ops::sigmoid(&ops::astype(&logits, Dtype::Float32)?)?;
		let corrected = match &self.e_score_correction_bias {
			Some(bias) => ops::add(&scores, bias)?,
			None => scores.clone(),
		};

		let k = self.top_k;
		let n_experts = corrected.dim(-1);
		let part = ops::argpartition_axis(&corrected, n_experts - k, -1)?;
		let shape = part.shape();
		let inds = ops::slice(&part, &[0, 0, n_experts - k], &shape)?;

		let weights = ops::take_along_axis(&scores, &inds, -1)?;
		let weights = if self.norm_topk_prob {
			let sum = ops::sum_axes(&weights, &[-1], true)?;
			ops::divide(&weights, &sum)?
		} else {
			weights
		};
		let weights = if self.routed_scaling_factor != 1.0 {
			ops::multiply(&weights, &Array::scalar_f32(self.routed_scaling_factor)?)?
		} else {
			weights
		};
		Ok((inds, ops::astype(&weights, x.dtype())?))
	}
}

/// Routed experts plus the always-on shared expert (no gate on the
/// shared expert, unlike the Qwen MoE block).
struct LagunaSparseMoeBlock {
	router: LagunaRouter,
	switch_mlp: SwitchGlu,
	shared_expert: SharedExpert,
}

impl LagunaSparseMoeBlock {
	fn load(w: &mut WeightMap, prefix: &str, cfg: &LagunaConfig) -> Result<Self> {
		Ok(LagunaSparseMoeBlock {
			router: LagunaRouter::load(w, prefix, cfg)?,
			switch_mlp: SwitchGlu::load(w, &format!("{prefix}.switch_mlp"))?,
			shared_expert: SharedExpert::load(w, &format!("{prefix}.shared_expert"))?,
		})
	}

	fn forward(&self, x: &Array) -> Result<Array> {
		let (inds, weights) = self.router.route(x)?;
		let y = self.switch_mlp.forward(x, &inds)?; // [B, S, K, H]
		let weighted = ops::multiply(&y, &ops::expand_dims(&weights, -1)?)?;
		let y = ops::sum_axes(&weighted, &[-2], false)?; // [B, S, H]
		ops::add(&y, &self.shared_expert.forward(x)?)
	}
}

enum FeedForward {
	Dense(Mlp),
	Moe(LagunaSparseMoeBlock),
}

impl FeedForward {
	fn forward(&self, x: &Array) -> Result<Array> {
		match self {
			FeedForward::Dense(mlp) => mlp.forward(x),
			FeedForward::Moe(moe) => moe.forward(x),
		}
	}
}

struct Block {
	self_attn: Attention,
	ff: FeedForward,
	input_layernorm: RmsNorm,
	post_attention_layernorm: RmsNorm,
}

impl Block {
	fn load(
		w: &mut WeightMap,
		prefix: &str,
		cfg: &LagunaConfig,
		layer_idx: usize,
		full_rope: &Rope,
		sliding_rope: &Rope,
	) -> Result<Self> {
		let mlp = format!("{prefix}.mlp");
		let ff = if cfg.sparse_layers[layer_idx] {
			FeedForward::Moe(LagunaSparseMoeBlock::load(w, &mlp, cfg)?)
		} else {
			FeedForward::Dense(Mlp::load(w, &mlp)?)
		};
		Ok(Block {
			self_attn: Attention::load(w, prefix, cfg, layer_idx, full_rope, sliding_rope)?,
			ff,
			input_layernorm: w.rms_norm(&format!("{prefix}.input_layernorm"), cfg.rms_norm_eps)?,
			post_attention_layernorm: w.rms_norm(
				&format!("{prefix}.post_attention_layernorm"),
				cfg.rms_norm_eps,
			)?,
		})
	}

	fn forward(&self, x: &Array, cache: &mut KvCache) -> Result<Array> {
		let h = ops::add(
			x,
			&self
				.self_attn
				.forward(&self.input_layernorm.forward(x)?, cache)?,
		)?;
		ops::add(
			&h,
			&self
				.ff
				.forward(&self.post_attention_layernorm.forward(&h)?)?,
		)
	}
}

/// Laguna causal language model.
pub struct LagunaModel {
	pub config: LagunaConfig,
	embed_tokens: Embedding,
	layers: Vec<Block>,
	norm: RmsNorm,
	lm_head: Option<Linear>,
}

impl LagunaModel {
	pub fn load(mut weights: WeightMap, config_json: &Value) -> Result<Self> {
		let cfg = LagunaConfig::from_json(config_json)?;

		let full_rope = cfg.full_rope.build(cfg.head_dim)?;
		let sliding_rope = cfg.sliding_rope.build(cfg.head_dim)?;

		let embed_tokens = weights.embedding("model.embed_tokens")?;
		let mut layers = Vec::with_capacity(cfg.num_hidden_layers as usize);
		for i in 0..cfg.num_hidden_layers as usize {
			layers.push(Block::load(
				&mut weights,
				&format!("model.layers.{i}"),
				&cfg,
				i,
				&full_rope,
				&sliding_rope,
			)?);
		}
		let norm = weights.rms_norm("model.norm", cfg.rms_norm_eps)?;
		let lm_head = if cfg.tie_word_embeddings {
			None
		} else {
			Some(weights.linear("lm_head")?)
		};

		Ok(LagunaModel {
			config: cfg,
			embed_tokens,
			layers,
			norm,
			lm_head,
		})
	}

	pub fn num_layers(&self) -> usize {
		self.layers.len()
	}

	pub fn new_caches(&self) -> Vec<LayerCache> {
		// Sliding-window layers retain only their window (plus growth
		// slack): their mask can never attend further back, and keeping
		// full history grew KV memory ~4x on S-2.1 (36 of 48 layers are
		// sliding) — enough to walk a long chat into the Metal wired
		// limit and crash.
		self.config
			.layer_kinds
			.iter()
			.map(|kind| match kind {
				LayerKind::Sliding => {
					LayerCache::new_attention_windowed(self.config.sliding_window)
				}
				LayerKind::Full => LayerCache::new_attention(),
			})
			.collect()
	}

	/// Run one forward pass. `input_ids` has shape `[B, L]`.
	pub fn forward(&self, input_ids: &Array, caches: &mut [LayerCache]) -> Result<Array> {
		let mut h = self.embed_tokens.forward(input_ids)?;

		for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
			h = layer.forward(&h, cache.as_attention()?)?;
		}
		h = self.norm.forward(&h)?;

		match &self.lm_head {
			Some(head) => head.forward(&h),
			None => self.embed_tokens.as_linear(&h),
		}
	}
}

#[cfg(test)]
mod tests {
	use serde_json::json;

	use super::*;
	use crate::engine::nn::DenseLinear;

	/// Subset of the real `ox-ox/Laguna-M.1-MLX-Q3` config.json, pinning
	/// the key spellings the parser depends on.
	#[test]
	fn laguna_m1_config_parses() {
		let mut mlp_layer_types = vec!["dense"; 3];
		mlp_layer_types.extend(vec!["sparse"; 67]);
		let cfg = json!({
			"model_type": "laguna",
			"hidden_size": 4096,
			"intermediate_size": 16384,
			"num_hidden_layers": 70,
			"num_attention_heads": 64,
			"num_key_value_heads": 8,
			"head_dim": 128,
			"vocab_size": 100352,
			"rms_norm_eps": 1e-6,
			"tie_word_embeddings": false,
			"rope_theta": 500000.0,
			"gating": "per-element",
			"num_experts": 256,
			"num_experts_per_tok": 16,
			"moe_intermediate_size": 1024,
			"shared_expert_intermediate_size": 1024,
			"moe_routed_scaling_factor": 1.0,
			"partial_rotary_factor": 1.0,
			"mlp_layer_types": mlp_layer_types,
			"rope_scaling": {
				"rope_type": "yarn",
				"factor": 32.0,
				"original_max_position_embeddings": 4096,
				"beta_slow": 1.0,
				"beta_fast": 64.0,
				"attention_factor": 1.0
			},
		});
		let c = LagunaConfig::from_json(&cfg).unwrap();
		assert_eq!(c.num_hidden_layers, 70);
		assert_eq!(c.heads_per_layer, vec![64; 70]);
		assert_eq!(c.layer_kinds, vec![LayerKind::Full; 70]);
		assert_eq!(c.gating_per_layer, vec![GatingKind::PerElement; 70]);
		assert!(!c.sparse_layers[0]);
		assert!(!c.sparse_layers[2]);
		assert!(c.sparse_layers[3]);
		assert!(c.sparse_layers[69]);
		assert_eq!(c.num_experts_per_tok, 16);
		assert!(c.full_rope.yarn);
		assert_eq!(c.full_rope.factor, 32.0);
		assert_eq!(c.full_rope.original_max_position_embeddings, 4096);
		assert_eq!(c.full_rope.beta_fast, 64.0);
		assert_eq!(c.full_rope.attention_factor, 1.0);
		assert_eq!(c.full_rope.partial_rotary_factor, 1.0);
		// No sliding layers: the sliding rope mirrors the full rope.
		assert_eq!(c.sliding_rope, c.full_rope);
	}

	/// Subset of the real `poolside/Laguna-S-2.1` config.json.
	#[test]
	fn laguna_s21_config_parses() {
		let mut layer_types = Vec::new();
		let mut heads = Vec::new();
		for _ in 0..12 {
			layer_types.push("full_attention");
			layer_types.extend(vec!["sliding_attention"; 3]);
			heads.push(48);
			heads.extend(vec![72; 3]);
		}
		let mut mlp_layer_types = vec!["dense"];
		mlp_layer_types.extend(vec!["sparse"; 47]);
		let cfg = json!({
			"model_type": "laguna",
			"hidden_size": 3072,
			"intermediate_size": 12288,
			"num_hidden_layers": 48,
			"num_attention_heads": 48,
			"num_key_value_heads": 8,
			"head_dim": 128,
			"vocab_size": 100352,
			"tie_word_embeddings": false,
			"gating": "per-head",
			"gating_types": vec!["per_head"; 48],
			"sliding_window": 512,
			"layer_types": layer_types,
			"num_attention_heads_per_layer": heads,
			"num_experts": 256,
			"num_experts_per_tok": 10,
			"moe_intermediate_size": 1024,
			"shared_expert_intermediate_size": 1024,
			"norm_topk_prob": true,
			"moe_routed_scaling_factor": 2.5,
			"mlp_layer_types": mlp_layer_types,
			"rope_theta": 500000.0,
			"rope_parameters": {
				"full_attention": {
					"rope_theta": 500000.0,
					"rope_type": "yarn",
					"factor": 128.0,
					"original_max_position_embeddings": 8192,
					"beta_slow": 1.0,
					"beta_fast": 32.0,
					"attention_factor": 1.4852030263919618,
					"partial_rotary_factor": 0.5
				},
				"sliding_attention": {
					"rope_type": "default",
					"rope_theta": 10000.0,
					"partial_rotary_factor": 1.0
				}
			},
		});
		let c = LagunaConfig::from_json(&cfg).unwrap();
		assert_eq!(c.layer_kinds[0], LayerKind::Full);
		assert_eq!(c.layer_kinds[1], LayerKind::Sliding);
		assert_eq!(c.layer_kinds[4], LayerKind::Full);
		assert_eq!(c.heads_per_layer[0], 48);
		assert_eq!(c.heads_per_layer[1], 72);
		assert_eq!(c.gating_per_layer, vec![GatingKind::PerHead; 48]);
		assert!(!c.sparse_layers[0]);
		assert!(c.sparse_layers[1]);
		assert_eq!(c.num_experts_per_tok, 10);
		assert_eq!(c.moe_routed_scaling_factor, 2.5);
		assert!(c.full_rope.yarn);
		assert_eq!(c.full_rope.factor, 128.0);
		assert_eq!(c.full_rope.partial_rotary_factor, 0.5);
		assert!((c.full_rope.attention_factor - 1.485_203).abs() < 1e-5);
		assert!(!c.sliding_rope.yarn);
		assert_eq!(c.sliding_rope.theta, 10000.0);
		assert_eq!(c.sliding_rope.partial_rotary_factor, 1.0);
	}

	#[test]
	fn missing_mlp_layer_types_is_an_error() {
		let cfg = json!({
			"hidden_size": 64,
			"intermediate_size": 128,
			"num_hidden_layers": 2,
			"num_attention_heads": 4,
			"head_dim": 16,
			"vocab_size": 100,
			"num_experts": 4,
			"num_experts_per_tok": 2,
			"moe_intermediate_size": 32,
			"shared_expert_intermediate_size": 32,
		});
		assert!(LagunaConfig::from_json(&cfg).is_err());
	}

	/// The correction bias must steer which experts are *selected* while
	/// the routing *weights* stay the renormalized uncorrected sigmoids.
	#[test]
	fn laguna_router_bias_selects_but_does_not_weight() {
		// 4 experts over a 1-dim hidden state; x = [1] so logits equal
		// the gate weights: [0.0, 0.5, 1.0, 2.0].
		let gate_weight = Array::from_slice(&[0.0f32, 0.5, 1.0, 2.0], &[4, 1]).unwrap();
		// Without bias, top-2 selection would pick experts {2, 3}. A
		// large bias on expert 0 flips the selection to {0, 3}.
		let bias = Array::from_slice(&[10.0f32, 0.0, 0.0, 0.0], &[4]).unwrap();
		let router = LagunaRouter {
			gate: Linear::Dense(DenseLinear {
				weight: gate_weight,
				bias: None,
			}),
			e_score_correction_bias: Some(bias),
			top_k: 2,
			norm_topk_prob: true,
			routed_scaling_factor: 1.0,
		};

		let x = Array::from_slice(&[1.0f32], &[1, 1, 1]).unwrap();
		let (inds, weights) = router.route(&x).unwrap();
		let mut inds = inds.to_vec_u32().unwrap();
		inds.sort_unstable();
		assert_eq!(inds, vec![0, 3]);

		// Weights are the *uncorrected* sigmoids at the selected experts,
		// renormalized to sum to one: sigmoid(0)=0.5, sigmoid(2)=0.8808.
		let w = weights.to_vec_f32().unwrap();
		let s0 = 0.5f32;
		let s3 = 1.0 / (1.0 + (-2.0f32).exp());
		let sum = s0 + s3;
		let mut expected = [s0 / sum, s3 / sum];
		// argpartition returns an unordered top-k; align by sorting on
		// the weight value.
		let mut got = [w[0], w[1]];
		got.sort_by(f32::total_cmp);
		expected.sort_by(f32::total_cmp);
		assert!((got[0] - expected[0]).abs() < 1e-5);
		assert!((got[1] - expected[1]).abs() < 1e-5);
		assert!((w[0] + w[1] - 1.0).abs() < 1e-5);
	}

	/// Without a correction bias (some conversions omit the tensor),
	/// selection is plain top-k on the sigmoid scores.
	#[test]
	fn laguna_router_works_without_correction_bias() {
		let gate_weight = Array::from_slice(&[0.0f32, 0.5, 1.0, 2.0], &[4, 1]).unwrap();
		let router = LagunaRouter {
			gate: Linear::Dense(DenseLinear {
				weight: gate_weight,
				bias: None,
			}),
			e_score_correction_bias: None,
			top_k: 2,
			norm_topk_prob: true,
			routed_scaling_factor: 1.0,
		};
		let x = Array::from_slice(&[1.0f32], &[1, 1, 1]).unwrap();
		let (inds, weights) = router.route(&x).unwrap();
		let mut inds = inds.to_vec_u32().unwrap();
		inds.sort_unstable();
		assert_eq!(inds, vec![2, 3]);
		let w = weights.to_vec_f32().unwrap();
		assert!((w[0] + w[1] - 1.0).abs() < 1e-5);
	}

	/// `moe_routed_scaling_factor` scales the final routing weights.
	#[test]
	fn laguna_router_applies_routed_scaling_factor() {
		let gate_weight = Array::from_slice(&[0.0f32, 0.5, 1.0, 2.0], &[4, 1]).unwrap();
		let router = LagunaRouter {
			gate: Linear::Dense(DenseLinear {
				weight: gate_weight,
				bias: None,
			}),
			e_score_correction_bias: None,
			top_k: 2,
			norm_topk_prob: true,
			routed_scaling_factor: 2.5,
		};
		let x = Array::from_slice(&[1.0f32], &[1, 1, 1]).unwrap();
		let (_, weights) = router.route(&x).unwrap();
		let w = weights.to_vec_f32().unwrap();
		assert!((w[0] + w[1] - 2.5).abs() < 1e-5);
	}
}
