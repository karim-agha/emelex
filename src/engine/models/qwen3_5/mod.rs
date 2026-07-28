//! Qwen3.5 / Qwen3.6 hybrid architecture: most decoder layers use the
//! linear-time `GatedDeltaNet` recurrence (see `gated_delta.rs`), with a
//! plain GQA self-attention layer every `full_attention_interval` layers.
//! Also covers the MoE variant (Qwen3.6-A3B) via `SparseMoeBlock`.
//!
//! Checkpoints whose `config.json` carries a `vision_config` sub-dict (and
//! ship the matching `vision_tower.*` weights - see [`vision`]) additionally
//! accept image input: [`Qwen35Model::forward_with_media`] splices the
//! vision tower's projected patch features into the text embedding stream
//! at `image_token_id` placeholder positions before the decoder stack runs
//! (video frames arrive as ordinary entries of `images`, one vision-tower
//! pass per frame). Text-only checkpoints are unaffected: the tower stays
//! `None` and `forward` behaves exactly as before.

pub mod vision;

use serde_json::Value;
use vision::{Qwen35VisionConfig, Qwen35VisionTower};

use super::{
	base::{
		RopeConfig, attention_mask_for, forward_embeds_chunked, merge_heads, splice_media_features,
	},
	cache::{GatedDeltaCache, LayerCache},
	config::{get_bool, get_f32, get_i32, optional_i32, require_i32, text_config},
	gated_delta::{GatedDeltaConfig, GatedDeltaNet},
	moe::SparseMoeBlock,
	mtp::{BackboneOutput, MtpCaches, MtpDetection, MtpStepOutput},
};
use crate::engine::{
	Cancellation,
	array::{Array, Dtype},
	error::{Error, Result},
	media::image::ProcessedImage,
	nn::{Embedding, Linear, RmsNorm, WeightMap},
	ops::{self, AttentionMask},
	quant::Quantization,
};

#[derive(Debug, Clone)]
pub struct Qwen35Config {
	pub hidden_size: i32,
	pub num_hidden_layers: i32,
	pub intermediate_size: i32,
	pub num_attention_heads: i32,
	pub num_key_value_heads: i32,
	pub head_dim: i32,
	pub rms_norm_eps: f32,
	pub vocab_size: i32,
	pub tie_word_embeddings: bool,
	pub attention_bias: bool,
	pub rope_theta: f32,
	pub partial_rotary_factor: f32,
	pub full_attention_interval: i32,

	pub linear_num_value_heads: i32,
	pub linear_num_key_heads: i32,
	pub linear_key_head_dim: i32,
	pub linear_value_head_dim: i32,
	pub linear_conv_kernel_dim: i32,

	pub num_experts: i32,
	pub num_experts_per_tok: i32,
	pub decoder_sparse_step: i32,
	pub moe_intermediate_size: i32,
	pub shared_expert_intermediate_size: i32,
	pub norm_topk_prob: bool,

	// emelex patch (not upstream): MTP-related config surface. The
	// backbone forward already assumes the gated q_proj layout
	// unconditionally, hence `attn_output_gate` defaults to true and is
	// only consulted by the MTP load guards.
	pub attn_output_gate: bool,
	pub mtp_num_hidden_layers: Option<i32>,
	pub mtp_use_dedicated_embeddings: bool,
}

impl Qwen35Config {
	pub fn from_json(cfg: &Value) -> Result<Self> {
		let root = cfg;
		let cfg = text_config(cfg);
		let hidden_size = require_i32(cfg, "hidden_size")?;
		let num_attention_heads = require_i32(cfg, "num_attention_heads")?;
		let head_dim = get_i32(cfg, "head_dim", hidden_size / num_attention_heads)?;
		let rope = cfg
			.get("rope_parameters")
			.map(|value| {
				value.as_object().ok_or_else(|| {
					Error::Config("field 'rope_parameters' must be an object".to_string())
				})?;
				Ok::<&Value, Error>(value)
			})
			.transpose()?;
		let rope_theta = match rope {
			Some(rope) => get_f32(rope, "rope_theta", 100_000.0)?,
			None => 100_000.0,
		};
		let partial_rotary_factor = match rope {
			Some(rope) => get_f32(rope, "partial_rotary_factor", 0.25)?,
			None => 0.25,
		};
		let mtp_num_hidden_layers = match optional_i32(cfg, "mtp_num_hidden_layers")? {
			Some(value) => Some(value),
			None => optional_i32(root, "mtp_num_hidden_layers")?,
		};

		Ok(Qwen35Config {
			hidden_size,
			num_hidden_layers: require_i32(cfg, "num_hidden_layers")?,
			// Absent on checkpoints where every layer is MoE (e.g. Qwen3.6-A3B),
			// since `Mlp::load` (the dense fallback) is then never reached.
			intermediate_size: get_i32(cfg, "intermediate_size", 0)?,
			num_attention_heads,
			num_key_value_heads: get_i32(cfg, "num_key_value_heads", num_attention_heads)?,
			head_dim,
			rms_norm_eps: get_f32(cfg, "rms_norm_eps", 1e-6)?,
			vocab_size: require_i32(cfg, "vocab_size")?,
			tie_word_embeddings: get_bool(cfg, "tie_word_embeddings", false)?,
			attention_bias: get_bool(cfg, "attention_bias", false)?,
			rope_theta,
			partial_rotary_factor,
			full_attention_interval: get_i32(cfg, "full_attention_interval", 4)?,
			linear_num_value_heads: get_i32(cfg, "linear_num_value_heads", 64)?,
			linear_num_key_heads: get_i32(cfg, "linear_num_key_heads", 16)?,
			linear_key_head_dim: get_i32(cfg, "linear_key_head_dim", 192)?,
			linear_value_head_dim: get_i32(cfg, "linear_value_head_dim", 128)?,
			linear_conv_kernel_dim: get_i32(cfg, "linear_conv_kernel_dim", 4)?,
			num_experts: get_i32(cfg, "num_experts", 0)?,
			num_experts_per_tok: get_i32(cfg, "num_experts_per_tok", 0)?,
			decoder_sparse_step: get_i32(cfg, "decoder_sparse_step", 1)?,
			moe_intermediate_size: get_i32(cfg, "moe_intermediate_size", 0)?,
			shared_expert_intermediate_size: get_i32(cfg, "shared_expert_intermediate_size", 0)?,
			norm_topk_prob: get_bool(cfg, "norm_topk_prob", true)?,
			// emelex patch (not upstream): MTP fields. The pinned dense
			// target ships `mtp_num_hidden_layers` inside text_config (the
			// converted fixture carries that config through); the root
			// fallback tolerates checkpoints that hoist it to the config
			// root.
			attn_output_gate: get_bool(cfg, "attn_output_gate", true)?,
			mtp_num_hidden_layers,
			mtp_use_dedicated_embeddings: get_bool(cfg, "mtp_use_dedicated_embeddings", false)?,
		})
	}

	fn is_linear_layer(&self, layer_idx: i32) -> bool {
		(layer_idx + 1) % self.full_attention_interval != 0
	}
}

struct Attention {
	q_proj: Linear,
	k_proj: Linear,
	v_proj: Linear,
	o_proj: Linear,
	q_norm: RmsNorm,
	k_norm: RmsNorm,
	rope: RopeConfig,
	n_heads: i32,
	n_kv_heads: i32,
	head_dim: i32,
	scale: f32,
}

impl Attention {
	fn load(w: &mut WeightMap, prefix: &str, cfg: &Qwen35Config) -> Result<Self> {
		let attn = format!("{prefix}.self_attn");
		let rope_dims = ((cfg.head_dim as f32) * cfg.partial_rotary_factor) as i32;
		Ok(Attention {
			q_proj: w.linear(&format!("{attn}.q_proj"))?,
			k_proj: w.linear(&format!("{attn}.k_proj"))?,
			v_proj: w.linear(&format!("{attn}.v_proj"))?,
			o_proj: w.linear(&format!("{attn}.o_proj"))?,
			q_norm: w.rms_norm(&format!("{attn}.q_norm"), cfg.rms_norm_eps)?,
			k_norm: w.rms_norm(&format!("{attn}.k_norm"), cfg.rms_norm_eps)?,
			rope: RopeConfig::new(rope_dims, cfg.rope_theta),
			n_heads: cfg.num_attention_heads,
			n_kv_heads: cfg.num_key_value_heads,
			head_dim: cfg.head_dim,
			scale: (cfg.head_dim as f32).powf(-0.5),
		})
	}

	fn forward(
		&self,
		x: &Array,
		mask: AttentionMask,
		cache: &mut super::cache::KvCache,
	) -> Result<Array> {
		let shape = x.shape();
		let (b, l) = (shape[0], shape[1]);

		let q_out = self.q_proj.forward(x)?;
		let q_out = ops::reshape(&q_out, &[b, l, self.n_heads, 2 * self.head_dim])?;
		let parts = ops::split(&q_out, 2, -1)?;
		let (queries, gate) = (&parts[0], &parts[1]);
		let gate = ops::reshape(gate, &[b, l, self.n_heads * self.head_dim])?;

		let k = self.k_proj.forward(x)?;
		let v = self.v_proj.forward(x)?;
		let k = ops::reshape(&k, &[b, l, self.n_kv_heads, self.head_dim])?;
		let v = ops::reshape(&v, &[b, l, self.n_kv_heads, self.head_dim])?;

		let queries = self.q_norm.forward(queries)?;
		let queries = ops::transpose_axes(&queries, &[0, 2, 1, 3])?;
		let k = self.k_norm.forward(&k)?;
		let k = ops::transpose_axes(&k, &[0, 2, 1, 3])?;
		let v = ops::transpose_axes(&v, &[0, 2, 1, 3])?;

		let offset = cache.offset();
		let queries = self.rope.apply(&queries, offset)?;
		let k = self.rope.apply(&k, offset)?;
		let (k, v) = cache.update_and_fetch(k, v)?;

		let out = ops::scaled_dot_product_attention(&queries, &k, &v, self.scale, mask)?;
		let out = merge_heads(&out, b, l)?;
		let gated = ops::multiply(&out, &ops::sigmoid(&gate)?)?;
		self.o_proj.forward(&gated)
	}
}

struct Mlp {
	gate_proj: Linear,
	up_proj: Linear,
	down_proj: Linear,
}

impl Mlp {
	fn load(w: &mut WeightMap, prefix: &str, hidden: i32, intermediate: i32) -> Result<Self> {
		let _ = (hidden, intermediate);
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

enum Mixer {
	Linear(GatedDeltaNet),
	Attention(Attention),
}

enum FeedForward {
	Dense(Mlp),
	Moe(SparseMoeBlock),
}

struct Block {
	mixer: Mixer,
	ff: FeedForward,
	input_layernorm: RmsNorm,
	post_attention_layernorm: RmsNorm,
}

impl Block {
	fn load(w: &mut WeightMap, prefix: &str, cfg: &Qwen35Config, layer_idx: i32) -> Result<Self> {
		let is_linear = cfg.is_linear_layer(layer_idx);
		let mixer = if is_linear {
			let gd_cfg = GatedDeltaConfig {
				num_v_heads: cfg.linear_num_value_heads,
				num_k_heads: cfg.linear_num_key_heads,
				head_k_dim: cfg.linear_key_head_dim,
				head_v_dim: cfg.linear_value_head_dim,
				conv_kernel_size: cfg.linear_conv_kernel_dim,
				rms_norm_eps: cfg.rms_norm_eps,
			};
			Mixer::Linear(GatedDeltaNet::load(
				w,
				&format!("{prefix}.linear_attn"),
				&gd_cfg,
			)?)
		} else {
			Mixer::Attention(Attention::load(w, prefix, cfg)?)
		};

		let use_moe = cfg.num_experts > 0 && (layer_idx + 1) % cfg.decoder_sparse_step == 0;
		let ff = if use_moe {
			FeedForward::Moe(SparseMoeBlock::load(w, &format!("{prefix}.mlp"), cfg)?)
		} else {
			FeedForward::Dense(Mlp::load(
				w,
				&format!("{prefix}.mlp"),
				cfg.hidden_size,
				cfg.intermediate_size,
			)?)
		};

		Ok(Block {
			mixer,
			ff,
			input_layernorm: w.rms_norm(&format!("{prefix}.input_layernorm"), cfg.rms_norm_eps)?,
			post_attention_layernorm: w.rms_norm(
				&format!("{prefix}.post_attention_layernorm"),
				cfg.rms_norm_eps,
			)?,
		})
	}

	fn forward(&self, x: &Array, mask: AttentionMask, cache: &mut LayerCache) -> Result<Array> {
		let normed = self.input_layernorm.forward(x)?;
		let r = match &self.mixer {
			Mixer::Linear(m) => m.forward(&normed, cache.as_gated_delta()?)?,
			Mixer::Attention(m) => m.forward(&normed, mask, cache.as_attention()?)?,
		};
		let h = ops::add(x, &r)?;
		let ff_in = self.post_attention_layernorm.forward(&h)?;
		let ff_out = match &self.ff {
			FeedForward::Dense(m) => m.forward(&ff_in)?,
			FeedForward::Moe(m) => m.forward(&ff_in)?,
		};
		ops::add(&h, &ff_out)
	}

	fn is_linear(&self) -> bool {
		matches!(self.mixer, Mixer::Linear(_))
	}
}

/// Optional image support (`Some` only on checkpoints whose `config.json`
/// carries a `vision_config` sub-dict with a matching `vision_tower.*`
/// weight set).
struct VisionSupport {
	tower: Qwen35VisionTower,
	image_token_id: i32,
	vision_start_token_id: i32,
	vision_end_token_id: i32,
	video_token_id: i32,
}

/// emelex patch (not upstream): the Qwen3.5 MTP (multi-token-prediction)
/// module — fusion norms, a 2H→H fc, one full-attention decoder [`Block`]
/// with its own rope + KV cache, and a final norm. Embeddings and the LM
/// head are shared with the backbone.
pub struct Qwen35Mtp {
	pre_fc_norm_embedding: RmsNorm,
	pre_fc_norm_hidden: RmsNorm,
	fc: Linear,
	layer: Block,
	norm: RmsNorm,
}

impl Qwen35Mtp {
	fn load(w: &mut WeightMap, cfg: &Qwen35Config) -> Result<Self> {
		// Per-tensor dense-BF16 assertion at load time:
		// each MTP tensor is re-checked via `peek` before any `take`, so a
		// failure names the tensor and leaves the map (and the backbone,
		// already loaded) untouched — the caller warns and skips.
		for key in MTP_SENTINELS {
			if let Some(t) = w.peek(key) {
				let dt = t.dtype();
				if dt != Dtype::BFloat16 {
					return Err(Error::Model(format!(
						"MTP tensor '{key}' has dtype {dt:?} (v1 loads dense BF16 only)"
					)));
				}
			}
		}
		// The single MTP layer is a full-attention dense block regardless
		// of the backbone's own layer schedule / MoE settings: neutralize
		// both so `Block::load` builds `Attention` + dense `Mlp` (MoE MTP
		// is rejected by the dense-BF16 guards before this runs).
		let mut mtp_cfg = cfg.clone();
		mtp_cfg.num_experts = 0;
		mtp_cfg.full_attention_interval = 1;
		Ok(Qwen35Mtp {
			pre_fc_norm_embedding: w.rms_norm("mtp.pre_fc_norm_embedding", cfg.rms_norm_eps)?,
			pre_fc_norm_hidden: w.rms_norm("mtp.pre_fc_norm_hidden", cfg.rms_norm_eps)?,
			fc: w.linear("mtp.fc")?,
			layer: Block::load(w, "mtp.layers.0", &mtp_cfg, 0)?,
			norm: w.rms_norm("mtp.norm", cfg.rms_norm_eps)?,
		})
	}
}

pub struct Qwen35Model {
	pub config: Qwen35Config,
	embed_tokens: Embedding,
	layers: Vec<Block>,
	norm: RmsNorm,
	lm_head: Option<Linear>,
	vision: Option<VisionSupport>,
	// emelex patch (not upstream): optional MTP module.
	mtp: Option<Qwen35Mtp>,
}

impl Qwen35Model {
	pub fn load(mut weights: WeightMap, config_json: &Value) -> Result<Self> {
		let cfg = Qwen35Config::from_json(config_json)?;

		// emelex patch (not upstream): post-sanitize MTP sentinel
		// validation over the canonical `mtp.*` keys, via non-mutating
		// `peek` only. Every failure warns and skips MTP; leftover
		// canonical `mtp.*` keys are then dropped (not loaded) so the
		// backbone load is byte-identical to a no-MTP load.
		let mtp_detected = weights.keys().any(|k| k.starts_with("mtp."));
		let load_mtp = mtp_detected
			&& match validate_mtp(config_json, &weights, &cfg) {
				Ok(()) => true,
				Err(reason) => {
					tracing::warn!("skipping MTP module ({reason}); backbone loads unchanged");
					false
				}
			};
		if mtp_detected && !load_mtp {
			weights.rename_keys(|k| (!k.starts_with("mtp.")).then(|| k.to_string()));
		}

		let embed_tokens = weights.embedding("language_model.model.embed_tokens")?;
		let mut layers = Vec::with_capacity(cfg.num_hidden_layers as usize);
		for i in 0..cfg.num_hidden_layers {
			layers.push(Block::load(
				&mut weights,
				&format!("language_model.model.layers.{i}"),
				&cfg,
				i,
			)?);
		}
		let norm = weights.rms_norm("language_model.model.norm", cfg.rms_norm_eps)?;
		let lm_head = if cfg.tie_word_embeddings {
			None
		} else {
			Some(weights.linear("language_model.lm_head")?)
		};

		let vision = Self::load_vision(&mut weights, config_json)?;

		// emelex patch (not upstream): optional MTP module load, after the
		// backbone (disjoint `mtp.*` keys — a partial MTP load failure
		// leaves the backbone untouched and warns-and-skips).
		let mtp = if load_mtp {
			match Qwen35Mtp::load(&mut weights, &cfg) {
				Ok(m) => Some(m),
				Err(e) => {
					tracing::warn!("MTP module load failed ({e}); continuing without MTP");
					None
				}
			}
		} else {
			None
		};

		Ok(Qwen35Model {
			config: cfg,
			embed_tokens,
			layers,
			norm,
			lm_head,
			vision,
			mtp,
		})
	}

	/// Build the vision tower iff `config.json` carries a `vision_config`
	/// sub-dict AND the checkpoint actually shipped the matching
	/// `vision_tower.*` weights (a checkpoint could declare `vision_config`
	/// while being distributed text-only); returns `None` otherwise so
	/// text-only checkpoints load exactly as before.
	fn load_vision(weights: &mut WeightMap, config_json: &Value) -> Result<Option<VisionSupport>> {
		let Some(vision_cfg_json) = config_json.get("vision_config") else {
			return Ok(None);
		};
		if !vision::has_vision_weights(weights) {
			return Ok(None);
		}

		let vision_cfg = Qwen35VisionConfig::from_json(vision_cfg_json)?;
		let tower = Qwen35VisionTower::load(weights, vision_cfg)?;

		Ok(Some(VisionSupport {
			tower,
			image_token_id: get_i32(config_json, "image_token_id", 151655)?,
			vision_start_token_id: get_i32(config_json, "vision_start_token_id", 151652)?,
			vision_end_token_id: get_i32(config_json, "vision_end_token_id", 151653)?,
			video_token_id: get_i32(config_json, "video_token_id", 151656)?,
		}))
	}

	/// Whether this checkpoint's vision tower was loaded (i.e. it can
	/// accept image input via [`Qwen35Model::forward_with_media`]).
	pub fn supports_images(&self) -> bool {
		self.vision.is_some()
	}

	/// `(patch_size, max_soft_tokens, spatial_merge_size)` for
	/// [`crate::engine::media::image::preprocess_image_bytes`], or `None` if this
	/// checkpoint has no vision support.
	pub fn image_processing_params(&self) -> Option<(i32, i32, i32)> {
		self.vision.as_ref().map(|v| {
			let cfg = v.tower.config();
			(cfg.patch_size, 1280, cfg.spatial_merge_size)
		})
	}

	/// `(image_token_id, vision_start_token_id, vision_end_token_id)`, or
	/// `None` if this checkpoint has no vision support. Reuses the
	/// `(image_token_id, boi_token_id, eoi_token_id)` shape every
	/// multimodal architecture in this crate exposes: Qwen3.5's
	/// `vision_start`/`vision_end` tokens play the same "wrap the expanded
	/// placeholder span" role as Gemma4's `boi`/`eoi`.
	pub fn image_token_ids(&self) -> Option<(u32, u32, u32)> {
		self.vision.as_ref().map(|v| {
			(
				v.image_token_id as u32,
				v.vision_start_token_id as u32,
				v.vision_end_token_id as u32,
			)
		})
	}

	/// `video_token_id` from `config.json`, or `None` when the checkpoint
	/// has no vision tower (video reuses the vision path per frame).
	pub fn video_token_id(&self) -> Option<u32> {
		self.vision.as_ref().map(|v| v.video_token_id as u32)
	}

	pub fn new_caches(&self) -> Vec<LayerCache> {
		self.layers
			.iter()
			.map(|l| {
				if l.is_linear() {
					LayerCache::GatedDelta(GatedDeltaCache::new())
				} else {
					LayerCache::new_attention()
				}
			})
			.collect()
	}

	pub fn num_layers(&self) -> usize {
		self.layers.len()
	}

	pub fn forward(&self, input_ids: &Array, caches: &mut [LayerCache]) -> Result<Array> {
		let h = self.embed_tokens.forward(input_ids)?;
		self.forward_from_embeds(input_ids, h, caches)
	}

	/// Same as [`Qwen35Model::forward`], but splices `images`' projected
	/// vision-tower features into the text embedding stream at
	/// `image_token_id` placeholder positions before running the decoder
	/// stack (order preserving across multiple images; each image's
	/// soft-token count must exactly match the number of `image_token_id`
	/// placeholders `input_ids` holds for it, produced by expanding the
	/// chat template's single-placeholder-per-image convention beforehand -
	/// see `crate::engine::generate::Session::encode_chat_with_media`).
	pub fn forward_with_images(
		&self,
		input_ids: &Array,
		images: &[ProcessedImage],
		caches: &mut [LayerCache],
	) -> Result<Array> {
		self.forward_with_media(input_ids, images, caches)
	}

	/// Same as [`Qwen35Model::forward_with_images`]; Qwen3.5-VL has no
	/// audio tower, so this simply ignores audio (kept as a distinct name
	/// to mirror the generic [`crate::engine::models::Model::forward_with_media`]
	/// dispatch signature other multimodal architectures use).
	pub fn forward_with_media(
		&self,
		input_ids: &Array,
		images: &[ProcessedImage],
		caches: &mut [LayerCache],
	) -> Result<Array> {
		let h = self.media_embeddings(input_ids, images, Cancellation::disabled())?;
		self.forward_from_embeds(input_ids, h, caches)
	}

	pub(crate) fn forward_with_media_cancellable(
		&self,
		input_ids: &Array,
		images: &[ProcessedImage],
		caches: &mut [LayerCache],
		chunk_tokens: usize,
		cancellation: Cancellation<'_>,
	) -> Result<Array> {
		let h = self.media_embeddings(input_ids, images, cancellation)?;
		forward_embeds_chunked(input_ids, &h, chunk_tokens, cancellation, |ids, chunk| {
			self.forward_from_embeds(ids, chunk, caches)
		})
	}

	fn media_embeddings(
		&self,
		input_ids: &Array,
		images: &[ProcessedImage],
		cancellation: Cancellation<'_>,
	) -> Result<Array> {
		cancellation.checkpoint()?;
		let mut h = self.embed_tokens.forward(input_ids)?;

		if !images.is_empty() {
			let vision = self.vision.as_ref().ok_or_else(|| {
				Error::Model("qwen3.5: model has no vision support (no vision_config)".into())
			})?;
			let mut all_features = Vec::with_capacity(images.len());
			for image in images {
				cancellation.checkpoint()?;
				let features =
					vision
						.tower
						.forward(&image.pixel_values, image.patch_h, image.patch_w)?;
				if cancellation.is_cooperative() {
					features.eval()?;
					cancellation.checkpoint()?;
				}
				all_features.push(features);
			}
			h = splice_media_features(&h, input_ids, all_features, vision.image_token_id, "image")?;
		}

		cancellation.checkpoint()?;
		Ok(h)
	}

	fn forward_from_embeds(
		&self,
		input_ids: &Array,
		h: Array,
		caches: &mut [LayerCache],
	) -> Result<Array> {
		// emelex patch (not upstream): thin wrapper over the
		// hidden-returning split — arithmetically identical op order,
		// discarding the pre-norm hidden.
		self.forward_hidden_from_embeds(input_ids, h, caches)
			.map(|(_h_raw, logits)| logits)
	}

	/// emelex patch (not upstream): the backbone forward split at the
	/// layer-loop → final-norm → head boundary, returning both the
	/// pre-final-norm decoder-stack output (`h_raw`, what the MTP module
	/// consumes) and the logits (`head(norm(h_raw))`) of one pass.
	fn forward_hidden_from_embeds(
		&self,
		input_ids: &Array,
		mut h: Array,
		caches: &mut [LayerCache],
	) -> Result<(Array, Array)> {
		let seq_len = input_ids.dim(1);
		let mask = attention_mask_for(seq_len);

		for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
			h = layer.forward(&h, mask, cache)?;
		}
		let h_raw = h;
		let normed = self.norm.forward(&h_raw)?;
		let logits = self.head(&normed)?;
		Ok((h_raw, logits))
	}

	/// Project hidden states through the LM head (tied embeddings or a
	/// dedicated `lm_head`).
	fn head(&self, h: &Array) -> Result<Array> {
		match &self.lm_head {
			Some(head) => head.forward(h),
			None => self.embed_tokens.as_linear(h),
		}
	}

	/// emelex patch (not upstream): one backbone pass returning the
	/// pre-final-norm hidden states alongside the logits (see
	/// [`BackboneOutput`]).
	pub fn forward_hidden(
		&self,
		input_ids: &Array,
		caches: &mut [LayerCache],
	) -> Result<BackboneOutput> {
		let h = self.embed_tokens.forward(input_ids)?;
		let (hidden_pre_norm, logits) = self.forward_hidden_from_embeds(input_ids, h, caches)?;
		Ok(BackboneOutput {
			hidden_pre_norm,
			logits,
		})
	}

	/// emelex patch (not upstream): whether the checkpoint's MTP module
	/// was loaded.
	pub fn has_mtp(&self) -> bool {
		self.mtp.is_some()
	}

	/// The cache kind of the (single) v1 MTP layer: one full-attention,
	/// non-windowed KV cache.
	///
	/// emelex patch (not upstream).
	fn mtp_cache_kind() -> LayerCache {
		LayerCache::new_attention()
	}

	/// emelex patch (not upstream): fresh working caches for the MTP
	/// module (one non-windowed attention [`super::cache::KvCache`]).
	pub fn new_mtp_caches(&self) -> MtpCaches {
		MtpCaches(vec![Self::mtp_cache_kind()])
	}

	/// emelex patch (not upstream): one MTP step over `input_ids`
	/// (`[1, L]`) and `prev_hidden` (`[1, L, H]` — backbone pre-norm
	/// hiddens, or the previous step's `recycle_hidden` when drafting
	/// recursively).
	///
	/// `fused = fc(concat([pre_fc_norm_embedding(embed(ids)),
	/// pre_fc_norm_hidden(prev_hidden)], -1))` — concat order
	/// `[embedding, hidden]` per both pinned references — then one
	/// full-attention [`Block`] with the rope offset taken from the KV
	/// cache exactly as backbone full-attention layers do (no rope
	/// adjustment), then `recycle_hidden = mtp.norm(stack_out)` and the
	/// shared head over `recycle_hidden`.
	pub fn forward_mtp(
		&self,
		input_ids: &Array,
		prev_hidden: &Array,
		caches: &mut MtpCaches,
	) -> Result<MtpStepOutput> {
		let mtp = self.mtp.as_ref().ok_or_else(|| {
			Error::Model("forward_mtp: this checkpoint has no MTP module loaded".into())
		})?;
		let embeds = self.embed_tokens.forward(input_ids)?;
		let normed_embeds = mtp.pre_fc_norm_embedding.forward(&embeds)?;
		let normed_hidden = mtp.pre_fc_norm_hidden.forward(prev_hidden)?;
		let fused = mtp
			.fc
			.forward(&ops::concatenate(&[&normed_embeds, &normed_hidden], -1)?)?;

		let seq_len = input_ids.dim(1);
		let mask = attention_mask_for(seq_len);
		let cache = caches.0.first_mut().ok_or_else(|| {
			Error::Model("forward_mtp: empty MtpCaches (build them with new_mtp_caches)".into())
		})?;
		let x = mtp.layer.forward(&fused, mask, cache)?;
		let recycle_hidden = mtp.norm.forward(&x)?;
		let logits = self.head(&recycle_hidden)?;
		Ok(MtpStepOutput {
			recycle_hidden,
			logits,
		})
	}
}

/// Normalize checkpoint weight keys. Qwen3.5/3.6 checkpoints (dense or MoE,
/// with or without a vision tower) always ship the language model under a
/// `language_model.*` prefix and, when present, the vision tower under a
/// bare `vision_tower.*` prefix (an `optiq_vision.safetensors` sidecar on
/// OptiQ checkpoints); keep both, and fold fused MoE expert weights into
/// the `switch_mlp.*` layout `SparseMoeBlock` expects.
///
/// emelex patch (not upstream): sanitize historically deleted every MTP
/// key. It now takes the pre-sanitize [`detect_mtp`] outcome: with
/// [`MtpDetection::Prefix`], keys under the detected prefix are preserved
/// and canonicalized to the internal bare `mtp.*` prefix (the identical
/// mapping is applied to quantization-override keys); with
/// [`MtpDetection::None`] the behavior is byte-identical to before. The
/// MoE fold below stays backbone-only — no MTP-prefix folding in v1.
pub fn sanitize(
	weights: &mut WeightMap,
	num_hidden_layers: i32,
	num_experts: i32,
	mtp: MtpDetection,
) {
	let keep = |k: &str| -> Option<String> {
		if let MtpDetection::Prefix(prefix) = &mtp
			&& let Some(canonical) = canonical_mtp_key(prefix, k)
		{
			return Some(canonical);
		}
		if k.starts_with("vision_tower.") {
			Some(k.to_string())
		} else if k.starts_with("language_model.") && !k.contains("mtp.") {
			Some(k.to_string())
		} else {
			None
		}
	};
	weights.rename_keys(&keep);
	if matches!(mtp, MtpDetection::Prefix(_)) {
		// emelex patch (not upstream): identical mapping for the per-layer
		// quantization-override keys (`normalize_quant_keys` precedent),
		// so a quant override naming an MTP tensor is visible to the
		// dense-BF16 guards under its canonical name.
		weights.normalize_quant_keys(&keep);
	}

	if num_experts <= 0 {
		return;
	}

	for l in 0..num_hidden_layers {
		let prefix = format!("language_model.model.layers.{l}.mlp");
		// Fused gate_up_proj (stacked experts): [E, 2*I, H] -> split in half.
		if let Some(gate_up) = weights.take_optional(&format!("{prefix}.experts.gate_up_proj")) {
			let mid = gate_up.dim(-2) / 2;
			let shape = gate_up.shape();
			if let Ok(gate) = ops::slice(&gate_up, &[0, 0, 0], &[shape[0], mid, shape[2]]) {
				weights.insert(format!("{prefix}.switch_mlp.gate_proj.weight"), gate);
			}
			if let Ok(up) = ops::slice(&gate_up, &[0, mid, 0], &[shape[0], shape[1], shape[2]]) {
				weights.insert(format!("{prefix}.switch_mlp.up_proj.weight"), up);
			}
		}
		if let Some(down) = weights.take_optional(&format!("{prefix}.experts.down_proj")) {
			weights.insert(format!("{prefix}.switch_mlp.down_proj.weight"), down);
		}
		// Per-expert separate weights: stack into [E, out, in].
		for name in ["gate_proj", "up_proj", "down_proj"] {
			if weights.contains(&format!("{prefix}.experts.0.{name}.weight")) {
				let mut expert_weights = Vec::new();
				let mut e = 0;
				while let Some(w) =
					weights.take_optional(&format!("{prefix}.experts.{e}.{name}.weight"))
				{
					expert_weights.push(w);
					e += 1;
				}
				let refs: Vec<&Array> = expert_weights.iter().collect();
				if let Ok(stacked) = ops::stack_axis(&refs, 0) {
					weights.insert(format!("{prefix}.switch_mlp.{name}.weight"), stacked);
				}
			}
		}
	}
}

// emelex patch (not upstream): MTP detection + validation. Detection runs
// BEFORE sanitize on the raw key set (sanitize deletes unpreserved MTP
// keys); probing is contractually non-mutating (`WeightMap::peek` only).

/// Canonicalize `{prefix}.rest` to the internal `mtp.rest` (identity when
/// `prefix` is already `mtp`); `None` when `key` is not under `prefix`.
fn canonical_mtp_key(prefix: &str, key: &str) -> Option<String> {
	let rest = key.strip_prefix(prefix)?.strip_prefix('.')?;
	Some(format!("mtp.{rest}"))
}

/// Exact raw-HF (unconverted-checkpoint) predicate, pre-canonicalization:
/// raw iff any key starts with `model.language_model.` OR any key ending
/// in `conv1d.weight` has `shape[-1] != 1` (the pinned MLX detector —
/// converted orientation has last dim 1). v1 is converted-only, so a raw
/// orientation skips MTP entirely; the backbone path is unchanged.
fn raw_hf_orientation(weights: &WeightMap) -> bool {
	weights
		.keys()
		.any(|k| k.starts_with("model.language_model."))
		|| weights.keys().any(|k| {
			k.ends_with("conv1d.weight")
				&& weights
					.peek(k)
					.is_some_and(|a| a.shape().last().copied() != Some(1))
		})
}

/// The sole supported on-disk MTP namespace: the
/// pinned converter's sanitize prepends `language_model.` to root keys, so
/// a converted dense checkpoint carries its MTP tensors as
/// `language_model.mtp.*`. Only this prefix can enable v1 MTP.
const MTP_SUPPORTED_PREFIX: &str = "language_model.mtp";

/// Forbidden-detection on-disk namespaces: probed
/// on the raw key set purely to fail closed — never canonicalization
/// sources. Any population of either (alone or mixed with the supported
/// namespace) warns and skips MTP with the backbone unchanged.
const MTP_FORBIDDEN_PREFIXES: [&str; 2] = ["language_model.model.mtp", "mtp"];

/// Whether any key sits under `{prefix}.`.
fn mtp_prefix_populated(weights: &WeightMap, prefix: &str) -> bool {
	let dotted = format!("{prefix}.");
	weights.keys().any(|k| k.starts_with(&dotted))
}

/// The populated forbidden namespaces, for the guard diagnostic.
fn forbidden_mtp_namespaces(weights: &WeightMap) -> Vec<&'static str> {
	MTP_FORBIDDEN_PREFIXES
		.into_iter()
		.filter(|p| mtp_prefix_populated(weights, p))
		.collect()
}

/// Namespace resolution over the raw key set:
/// only an exclusively-populated `language_model.mtp.*` resolves to
/// [`MtpDetection::Prefix`]; any population of a forbidden namespace —
/// alone or mixed — warns (naming the forbidden namespace) and fails
/// closed; nothing populated means no MTP.
fn resolve_mtp_namespace(weights: &WeightMap) -> MtpDetection {
	let forbidden = forbidden_mtp_namespaces(weights);
	if !forbidden.is_empty() {
		tracing::warn!(
			"MTP weights populate forbidden on-disk namespace(s) {forbidden:?} \
			 (sole supported namespace: '{MTP_SUPPORTED_PREFIX}.*'); skipping MTP — \
			 backbone loads unchanged"
		);
		return MtpDetection::None;
	}
	if mtp_prefix_populated(weights, MTP_SUPPORTED_PREFIX) {
		MtpDetection::Prefix(MTP_SUPPORTED_PREFIX.to_string())
	} else {
		MtpDetection::None
	}
}

/// Pre-sanitize MTP detection for the `Model::load` qwen3_5 arm: raw-HF
/// predicate first, then namespace resolution on the main weight map (the
/// sole MTP weight source — sidecar injection is a rejected alternative).
/// Any rejection warns and returns [`MtpDetection::None`] — the backbone
/// load is unaffected. Detection is contractually non-mutating.
pub fn detect_mtp(weights: &WeightMap) -> MtpDetection {
	if raw_hf_orientation(weights) {
		tracing::warn!(
			"raw-HF Qwen3.5 checkpoint orientation detected; skipping MTP \
			 (converted-checkpoint-only contract)"
		);
		return MtpDetection::None;
	}
	resolve_mtp_namespace(weights)
}

/// The complete dense v1 layer-0 sentinel set (canonical keys): fusion
/// norms + fc + one gated full-attention layer with q/k norms + final
/// norm. Embeddings/head are shared with the backbone.
const MTP_SENTINELS: [&str; 15] = [
	"mtp.fc.weight",
	"mtp.pre_fc_norm_embedding.weight",
	"mtp.pre_fc_norm_hidden.weight",
	"mtp.norm.weight",
	"mtp.layers.0.input_layernorm.weight",
	"mtp.layers.0.post_attention_layernorm.weight",
	"mtp.layers.0.self_attn.q_proj.weight",
	"mtp.layers.0.self_attn.k_proj.weight",
	"mtp.layers.0.self_attn.v_proj.weight",
	"mtp.layers.0.self_attn.o_proj.weight",
	"mtp.layers.0.self_attn.q_norm.weight",
	"mtp.layers.0.self_attn.k_norm.weight",
	"mtp.layers.0.mlp.gate_proj.weight",
	"mtp.layers.0.mlp.up_proj.weight",
	"mtp.layers.0.mlp.down_proj.weight",
];

/// Post-sanitize MTP validation over the canonical `mtp.*` keys, via
/// non-mutating `peek` only. `Err` carries the skip reason; the caller
/// warns, drops the leftover `mtp.*` keys, and loads the backbone
/// byte-identically to a no-MTP load.
fn validate_mtp(
	config_json: &Value,
	weights: &WeightMap,
	cfg: &Qwen35Config,
) -> std::result::Result<(), String> {
	validate_mtp_layout_class(config_json, cfg)?;
	for sentinel in MTP_SENTINELS {
		if weights.peek(sentinel).is_none() {
			return Err(format!(
				"incomplete MTP layer-0 tensor set (missing {sentinel})"
			));
		}
	}
	match cfg.mtp_num_hidden_layers {
		None => tracing::info!(
			"mtp_num_hidden_layers absent with a complete MTP sentinel set; \
			 treating as 1"
		),
		Some(1) => {}
		Some(n) => {
			return Err(format!(
				"mtp_num_hidden_layers = {n} (v1 supports exactly 1)"
			));
		}
	}
	if cfg.mtp_use_dedicated_embeddings {
		return Err("mtp_use_dedicated_embeddings = true (v1 requires shared embeddings)".into());
	}
	if !cfg.attn_output_gate {
		return Err(
			"attn_output_gate = false (the attention path assumes the gated q_proj \
			 layout)"
				.into(),
		);
	}
	validate_mtp_tensor_shapes(weights, cfg)?;
	// Dense-BF16 guards (v1 scope): quantized companions, quantization
	// overrides naming an MTP tensor, MoE keys under the MTP prefix, and
	// any unexpected extra key under the MTP prefix all warn-and-skip.
	for key in weights.keys() {
		if !key.starts_with("mtp.") {
			continue;
		}
		if key.ends_with(".scales") || key.ends_with(".biases") {
			return Err(format!(
				"quantized MTP tensor companion '{key}' (v1 is dense BF16 only)"
			));
		}
		if key.contains("switch_mlp") || key.contains("router") || key.contains("shared_expert") {
			return Err(format!("MoE MTP tensor '{key}' (v1 is dense only)"));
		}
		if !MTP_SENTINELS.contains(&key.as_str()) {
			return Err(format!(
				"unexpected key '{key}' under the MTP prefix (v1 expects exactly the \
				 15-tensor dense layer-0 set)"
			));
		}
	}
	for key in weights.quantization().per_layer.keys() {
		if key.starts_with("mtp.") {
			return Err(format!(
				"quantization override targets MTP tensor '{key}' (v1 is dense BF16 \
				 only)"
			));
		}
	}
	// BF16 dtype guard: v1 is dense BF16 — every required
	// MTP parameter tensor must have dtype BF16, verified via `peek` before
	// any consumption. F16/F32 (or any other) substitution at any tensor
	// warns and skips, naming the offending key and observed dtype.
	for sentinel in MTP_SENTINELS {
		if let Some(t) = weights.peek(sentinel) {
			let dt = t.dtype();
			if dt != Dtype::BFloat16 {
				return Err(format!(
					"MTP tensor '{sentinel}' has dtype {dt:?} (v1 requires dense BF16)"
				));
			}
		}
	}
	Ok(())
}

/// The only production layout enabled by the v1 parity certificate.
///
/// A complete-looking MTP tensor set on another Qwen variant remains an
/// advertised capability, but cannot become actionable until that exact layout
/// class has its own checked-in parity certificate.
fn validate_mtp_layout_class(
	config_json: &Value,
	cfg: &Qwen35Config,
) -> std::result::Result<(), String> {
	#[cfg(test)]
	if cfg.hidden_size == 32
		&& cfg.num_hidden_layers == 2
		&& cfg.intermediate_size == 64
		&& cfg.num_attention_heads == 2
		&& cfg.num_key_value_heads == 1
		&& cfg.head_dim == 16
		&& cfg.vocab_size == 16
	{
		return Ok(());
	}

	let root_model_type = config_json.get("model_type").and_then(Value::as_str);
	let text = text_config(config_json);
	let text_model_type = text.get("model_type").and_then(Value::as_str);
	let declared_dtype = text.get("dtype").and_then(Value::as_str);
	let exact = root_model_type == Some("qwen3_5")
		&& text_model_type == Some("qwen3_5_text")
		&& config_json.get("vision_config").is_none()
		&& declared_dtype == Some("bfloat16")
		&& cfg.hidden_size == 2_560
		&& cfg.num_hidden_layers == 32
		&& cfg.intermediate_size == 9_216
		&& cfg.num_attention_heads == 16
		&& cfg.num_key_value_heads == 4
		&& cfg.head_dim == 256
		&& cfg.vocab_size == 248_320
		&& cfg.full_attention_interval == 4
		&& cfg.linear_num_value_heads == 32
		&& cfg.linear_num_key_heads == 16
		&& cfg.linear_key_head_dim == 128
		&& cfg.linear_value_head_dim == 128
		&& cfg.linear_conv_kernel_dim == 4
		&& cfg.num_experts == 0
		&& cfg.tie_word_embeddings;
	if exact {
		Ok(())
	} else {
		Err(
			"MTP layout is not the parity-certified dense BF16 Qwen3.5-4B \
			 text layout (implementation emelex-qwen3.5-mtp-dense-bf16-v1)"
				.to_string(),
		)
	}
}

fn validate_mtp_tensor_shapes(
	weights: &WeightMap,
	cfg: &Qwen35Config,
) -> std::result::Result<(), String> {
	let hidden = cfg.hidden_size;
	let two_hidden = hidden
		.checked_mul(2)
		.ok_or_else(|| "MTP fusion shape arithmetic overflow".to_string())?;
	let q_rows = cfg
		.num_attention_heads
		.checked_mul(cfg.head_dim)
		.and_then(|value| value.checked_mul(2))
		.ok_or_else(|| "MTP q_proj shape arithmetic overflow".to_string())?;
	let kv_rows = cfg
		.num_key_value_heads
		.checked_mul(cfg.head_dim)
		.ok_or_else(|| "MTP kv projection shape arithmetic overflow".to_string())?;
	let q_width = cfg
		.num_attention_heads
		.checked_mul(cfg.head_dim)
		.ok_or_else(|| "MTP attention output shape arithmetic overflow".to_string())?;
	let expected: [(&str, &[i32]); 15] = [
		("mtp.fc.weight", &[hidden, two_hidden]),
		("mtp.pre_fc_norm_embedding.weight", &[hidden]),
		("mtp.pre_fc_norm_hidden.weight", &[hidden]),
		("mtp.norm.weight", &[hidden]),
		("mtp.layers.0.input_layernorm.weight", &[hidden]),
		("mtp.layers.0.post_attention_layernorm.weight", &[hidden]),
		("mtp.layers.0.self_attn.q_proj.weight", &[q_rows, hidden]),
		("mtp.layers.0.self_attn.k_proj.weight", &[kv_rows, hidden]),
		("mtp.layers.0.self_attn.v_proj.weight", &[kv_rows, hidden]),
		("mtp.layers.0.self_attn.o_proj.weight", &[hidden, q_width]),
		("mtp.layers.0.self_attn.q_norm.weight", &[cfg.head_dim]),
		("mtp.layers.0.self_attn.k_norm.weight", &[cfg.head_dim]),
		(
			"mtp.layers.0.mlp.gate_proj.weight",
			&[cfg.intermediate_size, hidden],
		),
		(
			"mtp.layers.0.mlp.up_proj.weight",
			&[cfg.intermediate_size, hidden],
		),
		(
			"mtp.layers.0.mlp.down_proj.weight",
			&[hidden, cfg.intermediate_size],
		),
	];
	for (key, shape) in expected {
		let observed = weights
			.peek(key)
			.ok_or_else(|| format!("incomplete MTP tensor set (missing {key})"))?
			.shape();
		if observed != shape {
			return Err(format!(
				"MTP tensor '{key}' shape {observed:?} != certified {shape:?}"
			));
		}
	}
	Ok(())
}

pub fn parse_quantization(config_json: &Value) -> Result<Quantization> {
	Quantization::from_config(config_json)
}

pub fn model_error(model_type: &str) -> Error {
	Error::Model(format!("unsupported qwen3.5 variant '{model_type}'"))
}

#[cfg(test)]
mod tests {
	use super::*;

	// Key subsets of the real `config.json` files shipped by
	// `Qwen/Qwen3.6-27B` and `Qwen/Qwen3.6-35B-A3B` (verified July
	// 2026). Qwen3.6 reuses `model_type: qwen3_5[_moe]` — these tests
	// pin the exact keys that generation of checkpoints depends on, so
	// parser drift that would break Qwen3.6 fails here first.

	// emelex patch (not upstream): "fixtures load" gate for the committed
	// tiny-model fixture backing the non-live MTP suites. The fixture is
	// deliberately hybrid (layer 0 gated-delta, layer 1 full attention)
	// so both reconciliation paths are exercisable at toy scale.
	#[test]
	fn tiny_fixture_config_parses() {
		let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
			.join("tests/fixtures/tiny-model/config.json");
		let config: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
		let cfg = Qwen35Config::from_json(&config).unwrap();
		assert_eq!(cfg.hidden_size, 32);
		assert_eq!(cfg.num_hidden_layers, 2);
		assert_eq!(cfg.vocab_size, 16);
		assert_eq!(cfg.head_dim, 16);
		assert_eq!(cfg.full_attention_interval, 2);
		assert!(cfg.is_linear_layer(0));
		assert!(!cfg.is_linear_layer(1));
		assert_eq!(cfg.rope_theta, 100_000.0);
		assert_eq!(cfg.partial_rotary_factor, 0.25);
		assert!(cfg.tie_word_embeddings);
	}

	#[test]
	fn qwen3_6_dense_config_parses() {
		let config: Value = serde_json::json!({
			"architectures": ["Qwen3_5ForConditionalGeneration"],
			"model_type": "qwen3_5",
			"image_token_id": 248056,
			"text_config": {
				"attn_output_gate": true,
				"full_attention_interval": 4,
				"head_dim": 256,
				"hidden_size": 5120,
				"intermediate_size": 17408,
				"linear_conv_kernel_dim": 4,
				"linear_key_head_dim": 128,
				"linear_num_key_heads": 16,
				"linear_num_value_heads": 48,
				"linear_value_head_dim": 128,
				"model_type": "qwen3_5_text",
				"num_attention_heads": 24,
				"num_hidden_layers": 64,
				"num_key_value_heads": 4,
				"output_gate_type": "swish",
				"partial_rotary_factor": 0.25,
				"rms_norm_eps": 1e-6,
				"rope_parameters": {
					"mrope_interleaved": true,
					"mrope_section": [11, 11, 10],
					"partial_rotary_factor": 0.25,
					"rope_theta": 10_000_000,
					"rope_type": "default"
				},
				"tie_word_embeddings": false,
				"vocab_size": 248320
			}
		});
		let parsed = Qwen35Config::from_json(&config).expect("parse");
		assert_eq!(parsed.hidden_size, 5120);
		assert_eq!(parsed.num_hidden_layers, 64);
		assert_eq!(parsed.num_attention_heads, 24);
		assert_eq!(parsed.num_key_value_heads, 4);
		assert_eq!(parsed.head_dim, 256);
		assert_eq!(parsed.full_attention_interval, 4);
		assert!((parsed.rope_theta - 10_000_000.0).abs() < f32::EPSILON);
		assert!((parsed.partial_rotary_factor - 0.25).abs() < f32::EPSILON);
		assert_eq!(parsed.linear_num_value_heads, 48);
		assert_eq!(parsed.linear_num_key_heads, 16);
		assert_eq!(parsed.linear_key_head_dim, 128);
		assert_eq!(parsed.linear_value_head_dim, 128);
		assert_eq!(parsed.linear_conv_kernel_dim, 4);
		assert_eq!(parsed.num_experts, 0, "dense variant has no experts");
		assert!(!parsed.tie_word_embeddings);

		// The published layer_types array is exactly this rule: three
		// linear-attention layers, then one full-attention layer.
		for layer in 0..parsed.num_hidden_layers {
			assert_eq!(
				parsed.is_linear_layer(layer),
				(layer + 1) % 4 != 0,
				"layer {layer} classification"
			);
		}
	}

	// ------------------------------------------------------------------
	// emelex patch (not upstream): MTP loader unit tests — synthetic
	// WeightMaps at the tiny-fixture dims (hidden 32, heads 2, kv 1,
	// head_dim 16 → gated q_proj rows 64, vocab 16, 2 layers: layer 0
	// gated-delta, layer 1 full attention).
	// ------------------------------------------------------------------

	use std::collections::HashMap;

	use super::super::{Model, mtp::MtpDetection};
	use crate::engine::{
		ops::QuantMode,
		quant::{LayerOverride, QuantParams, Quantization},
	};

	/// Deterministic, varied filler values so op-order comparisons are
	/// meaningful (all-equal weights would mask permutation bugs).
	fn arr(seed: f32, shape: &[i32]) -> Array {
		let len: usize = shape.iter().map(|&d| d as usize).product();
		let data: Vec<f32> = (0..len)
			.map(|i| ((i as f32) * 0.7311 + seed).sin() * 0.05)
			.collect();
		Array::from_slice(&data, shape).unwrap()
	}

	/// [`arr`] cast to BF16 — the checkpoint dtype the v1 contract covers
	/// (dense BF16 only) and the only one the MTP dtype guard accepts.
	fn arr_bf16(seed: f32, shape: &[i32]) -> Array {
		ops::astype(&arr(seed, shape), Dtype::BFloat16).unwrap()
	}

	/// A complete tiny converted-orientation backbone
	/// (`language_model.model.*`, conv1d last dim == 1, tied embeddings),
	/// BF16 like a real converted dense checkpoint.
	fn backbone_tensors() -> HashMap<String, Array> {
		let specs: &[(&str, &[i32])] = &[
			("language_model.model.embed_tokens.weight", &[16, 32]),
			// Layer 0: gated-delta (linear attention). key_dim 16,
			// value_dim 32, conv_dim 64.
			(
				"language_model.model.layers.0.linear_attn.conv1d.weight",
				&[64, 4, 1],
			),
			(
				"language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
				&[64, 32],
			),
			(
				"language_model.model.layers.0.linear_attn.in_proj_z.weight",
				&[32, 32],
			),
			(
				"language_model.model.layers.0.linear_attn.in_proj_b.weight",
				&[4, 32],
			),
			(
				"language_model.model.layers.0.linear_attn.in_proj_a.weight",
				&[4, 32],
			),
			("language_model.model.layers.0.linear_attn.dt_bias", &[4]),
			("language_model.model.layers.0.linear_attn.A_log", &[4]),
			(
				"language_model.model.layers.0.linear_attn.norm.weight",
				&[8],
			),
			(
				"language_model.model.layers.0.linear_attn.out_proj.weight",
				&[32, 32],
			),
			(
				"language_model.model.layers.0.mlp.gate_proj.weight",
				&[64, 32],
			),
			(
				"language_model.model.layers.0.mlp.up_proj.weight",
				&[64, 32],
			),
			(
				"language_model.model.layers.0.mlp.down_proj.weight",
				&[32, 64],
			),
			(
				"language_model.model.layers.0.input_layernorm.weight",
				&[32],
			),
			(
				"language_model.model.layers.0.post_attention_layernorm.weight",
				&[32],
			),
			// Layer 1: gated full attention.
			(
				"language_model.model.layers.1.self_attn.q_proj.weight",
				&[64, 32],
			),
			(
				"language_model.model.layers.1.self_attn.k_proj.weight",
				&[16, 32],
			),
			(
				"language_model.model.layers.1.self_attn.v_proj.weight",
				&[16, 32],
			),
			(
				"language_model.model.layers.1.self_attn.o_proj.weight",
				&[32, 32],
			),
			(
				"language_model.model.layers.1.self_attn.q_norm.weight",
				&[16],
			),
			(
				"language_model.model.layers.1.self_attn.k_norm.weight",
				&[16],
			),
			(
				"language_model.model.layers.1.mlp.gate_proj.weight",
				&[64, 32],
			),
			(
				"language_model.model.layers.1.mlp.up_proj.weight",
				&[64, 32],
			),
			(
				"language_model.model.layers.1.mlp.down_proj.weight",
				&[32, 64],
			),
			(
				"language_model.model.layers.1.input_layernorm.weight",
				&[32],
			),
			(
				"language_model.model.layers.1.post_attention_layernorm.weight",
				&[32],
			),
			("language_model.model.norm.weight", &[32]),
		];
		specs
			.iter()
			.enumerate()
			.map(|(i, (key, shape))| (key.to_string(), arr_bf16(i as f32, shape)))
			.collect()
	}

	/// The 15 dense MTP module tensor names relative to their namespace
	/// prefix, shapes scaled to the tiny config. On disk the sole supported
	/// namespace is `language_model.mtp.{name}`; the
	/// same names under bare-root `mtp.` or `language_model.model.mtp.` form
	/// the forbidden layouts.
	fn mtp_tensor_specs() -> Vec<(&'static str, &'static [i32])> {
		vec![
			("fc.weight", &[32, 64]),
			("pre_fc_norm_embedding.weight", &[32]),
			("pre_fc_norm_hidden.weight", &[32]),
			("norm.weight", &[32]),
			("layers.0.input_layernorm.weight", &[32]),
			("layers.0.post_attention_layernorm.weight", &[32]),
			("layers.0.self_attn.q_proj.weight", &[64, 32]),
			("layers.0.self_attn.k_proj.weight", &[16, 32]),
			("layers.0.self_attn.v_proj.weight", &[16, 32]),
			("layers.0.self_attn.o_proj.weight", &[32, 32]),
			("layers.0.self_attn.q_norm.weight", &[16]),
			("layers.0.self_attn.k_norm.weight", &[16]),
			("layers.0.mlp.gate_proj.weight", &[64, 32]),
			("layers.0.mlp.up_proj.weight", &[64, 32]),
			("layers.0.mlp.down_proj.weight", &[32, 64]),
		]
	}

	fn insert_mtp_tensors(tensors: &mut HashMap<String, Array>, prefix: &str) {
		for (i, (name, shape)) in mtp_tensor_specs().into_iter().enumerate() {
			tensors.insert(
				format!("{prefix}.{name}"),
				arr_bf16(100.0 + i as f32, shape),
			);
		}
	}

	/// The tiny fixture config with per-test extra root fields.
	fn tiny_config(extras: &[(&str, Value)]) -> Value {
		let mut config = serde_json::json!({
			"model_type": "qwen3_5_text",
			"hidden_size": 32,
			"num_hidden_layers": 2,
			"intermediate_size": 64,
			"num_attention_heads": 2,
			"num_key_value_heads": 1,
			"head_dim": 16,
			"rms_norm_eps": 1e-6,
			"vocab_size": 16,
			"tie_word_embeddings": true,
			"attention_bias": false,
			"full_attention_interval": 2,
			"linear_num_value_heads": 4,
			"linear_num_key_heads": 2,
			"linear_key_head_dim": 8,
			"linear_value_head_dim": 8,
			"linear_conv_kernel_dim": 4,
			"rope_parameters": {
				"rope_theta": 100000.0,
				"partial_rotary_factor": 0.25
			}
		});
		for (key, value) in extras {
			config[key] = value.clone();
		}
		config
	}

	/// Mirror of the `Model::load` qwen3_5 arm minus the file I/O and the
	/// config-level standalone-sidecar guard: detection (raw-HF predicate +
	/// namespace resolution) → sanitize → `Qwen35Model::load`.
	fn load_via_pipeline(
		tensors: HashMap<String, Array>,
		quant: Quantization,
		config: &Value,
	) -> Model {
		let mut weights = WeightMap::new(tensors, quant);
		let detection = detect_mtp(&weights);
		sanitize(&mut weights, 2, 0, detection);
		Model::Qwen35(Qwen35Model::load(weights, config).expect("backbone load"))
	}

	fn test_ids() -> Array {
		Array::from_slice(&[1i32, 2, 3], &[1, 3]).unwrap()
	}

	/// Every skip case must leave a working backbone and no MTP.
	fn assert_backbone_only(model: &Model) -> Vec<f32> {
		assert!(!model.has_mtp());
		let mut caches = model.new_caches();
		let logits = model.forward(&test_ids(), &mut caches).expect("forward");
		assert_eq!(logits.shape(), vec![1, 3, 16]);
		let mut mtp_caches = model.new_mtp_caches();
		let hidden = arr(0.0, &[1, 3, 32]);
		assert!(
			model
				.forward_mtp(&test_ids(), &hidden, &mut mtp_caches)
				.is_err(),
			"forward_mtp must be Err when no MTP module is loaded"
		);
		logits.to_vec_f32().unwrap()
	}

	/// emelex patch: detection non-mutation snapshot — key set and per-key
	/// shapes AND dtypes are byte-identical before/after the full probe
	/// (raw-HF predicate + namespace resolution, then post-sanitize
	/// sentinel/dtype validation on the canonical keys).
	#[test]
	fn mtp_detection_probe_is_non_mutating() {
		let snapshot = |w: &WeightMap| -> std::collections::BTreeMap<String, (Vec<i32>, Dtype)> {
			w.keys()
				.map(|k| {
					let a = w.peek(k).unwrap();
					(k.clone(), (a.shape(), a.dtype()))
				})
				.collect()
		};

		// Pre-sanitize phase on the on-disk `language_model.mtp.*` layout.
		let mut tensors = backbone_tensors();
		insert_mtp_tensors(&mut tensors, "language_model.mtp");
		let weights = WeightMap::new(tensors, Quantization::default());
		let before = snapshot(&weights);
		assert!(!raw_hf_orientation(&weights));
		assert_eq!(
			detect_mtp(&weights),
			MtpDetection::Prefix("language_model.mtp".to_string())
		);
		assert_eq!(before, snapshot(&weights));

		// Post-sanitize phase: sentinel/gate/dtype validation over the
		// canonical in-memory `mtp.*` keys is equally non-mutating.
		let mut tensors = backbone_tensors();
		insert_mtp_tensors(&mut tensors, "language_model.mtp");
		let mut weights = WeightMap::new(tensors, Quantization::default());
		sanitize(
			&mut weights,
			2,
			0,
			MtpDetection::Prefix("language_model.mtp".to_string()),
		);
		let config = tiny_config(&[]);
		let cfg = Qwen35Config::from_json(&config).unwrap();
		let before = snapshot(&weights);
		assert!(validate_mtp(&config, &weights, &cfg).is_ok());
		assert_eq!(before, snapshot(&weights));
	}

	/// emelex patch: raw-HF orientation classification — the exact
	/// pre-canonicalization predicate.
	#[test]
	fn mtp_raw_hf_predicate_classifies() {
		// A `model.language_model.` key marks the raw orientation.
		let raw = WeightMap::new(
			HashMap::from([(
				"model.language_model.model.embed_tokens.weight".to_string(),
				arr(0.0, &[4, 8]),
			)]),
			Quantization::default(),
		);
		assert!(raw_hf_orientation(&raw));

		// Converted orientation: conv1d.weight last dim == 1.
		let converted = WeightMap::new(backbone_tensors(), Quantization::default());
		assert!(!raw_hf_orientation(&converted));

		// Any conv1d.weight with last dim != 1 marks the raw orientation.
		let mut tensors = backbone_tensors();
		tensors.insert(
			"language_model.extra.conv1d.weight".to_string(),
			arr(0.0, &[64, 1, 4]),
		);
		let raw_conv = WeightMap::new(tensors, Quantization::default());
		assert!(raw_hf_orientation(&raw_conv));
	}

	/// emelex patch: a raw-orientation marker skips MTP through the full
	/// pipeline; the backbone still loads and has_mtp() is false.
	#[test]
	fn mtp_raw_hf_orientation_skips_but_backbone_loads() {
		let mut tensors = backbone_tensors();
		insert_mtp_tensors(&mut tensors, "language_model.mtp");
		tensors.insert(
			"model.language_model.rotary_emb.inv_freq".to_string(),
			arr(0.0, &[4]),
		);
		let model = load_via_pipeline(tensors, Quantization::default(), &tiny_config(&[]));
		assert_backbone_only(&model);
	}

	/// A converted backbone plus the
	/// complete BF16 bare-root 15-key `mtp.*` set is a FORBIDDEN on-disk
	/// namespace → MTP disabled, backbone byte-identical to a clean
	/// no-MTP load.
	#[test]
	fn mtp_forbidden_bare_root_namespace_skips() {
		let mut tensors = backbone_tensors();
		insert_mtp_tensors(&mut tensors, "mtp");
		let probe = WeightMap::new(tensors.clone(), Quantization::default());
		assert_eq!(forbidden_mtp_namespaces(&probe), vec!["mtp"]);
		assert_eq!(detect_mtp(&probe), MtpDetection::None);

		let model = load_via_pipeline(tensors, Quantization::default(), &tiny_config(&[]));
		let skip_logits = assert_backbone_only(&model);

		let clean = load_via_pipeline(
			backbone_tensors(),
			Quantization::default(),
			&tiny_config(&[]),
		);
		assert_eq!(skip_logits, assert_backbone_only(&clean));
	}

	/// A complete
	/// `language_model.model.mtp.*` set is a FORBIDDEN on-disk namespace →
	/// MTP disabled, backbone loads.
	#[test]
	fn mtp_forbidden_nested_namespace_skips() {
		let mut tensors = backbone_tensors();
		insert_mtp_tensors(&mut tensors, "language_model.model.mtp");
		let probe = WeightMap::new(tensors.clone(), Quantization::default());
		assert_eq!(
			forbidden_mtp_namespaces(&probe),
			vec!["language_model.model.mtp"]
		);
		assert_eq!(detect_mtp(&probe), MtpDetection::None);

		let model = load_via_pipeline(tensors, Quantization::default(), &tiny_config(&[]));
		assert_backbone_only(&model);
	}

	/// The supported namespace mixed with
	/// a forbidden one fails closed — the guard diagnostic names the
	/// forbidden namespace and the backbone is byte-identical to a clean
	/// no-MTP load.
	#[test]
	fn mtp_mixed_namespaces_fail_closed() {
		// Supported + bare-root.
		let mut tensors = backbone_tensors();
		insert_mtp_tensors(&mut tensors, "language_model.mtp");
		tensors.insert("mtp.fc.weight".to_string(), arr_bf16(200.0, &[32, 64]));
		let probe = WeightMap::new(tensors.clone(), Quantization::default());
		assert_eq!(
			forbidden_mtp_namespaces(&probe),
			vec!["mtp"],
			"guard diagnostic must name the forbidden namespace"
		);
		assert_eq!(detect_mtp(&probe), MtpDetection::None);

		let model = load_via_pipeline(tensors, Quantization::default(), &tiny_config(&[]));
		let skip_logits = assert_backbone_only(&model);
		let clean = load_via_pipeline(
			backbone_tensors(),
			Quantization::default(),
			&tiny_config(&[]),
		);
		assert_eq!(skip_logits, assert_backbone_only(&clean));

		// Supported + nested-forbidden.
		let mut tensors = backbone_tensors();
		insert_mtp_tensors(&mut tensors, "language_model.mtp");
		tensors.insert(
			"language_model.model.mtp.fc.weight".to_string(),
			arr_bf16(201.0, &[32, 64]),
		);
		let probe = WeightMap::new(tensors.clone(), Quantization::default());
		assert_eq!(
			forbidden_mtp_namespaces(&probe),
			vec!["language_model.model.mtp"]
		);
		assert_eq!(detect_mtp(&probe), MtpDetection::None);
		let model = load_via_pipeline(tensors, Quantization::default(), &tiny_config(&[]));
		assert_backbone_only(&model);
	}

	/// The exact complete
	/// `language_model.mtp.*` set ALONE — the sole supported on-disk
	/// namespace — canonicalizes to in-memory `mtp.*` and enables MTP.
	#[test]
	fn mtp_supported_namespace_canonicalizes_and_loads() {
		let mut tensors = backbone_tensors();
		insert_mtp_tensors(&mut tensors, "language_model.mtp");
		let probe = WeightMap::new(tensors.clone(), Quantization::default());
		assert!(forbidden_mtp_namespaces(&probe).is_empty());
		assert_eq!(
			detect_mtp(&probe),
			MtpDetection::Prefix("language_model.mtp".to_string())
		);

		let model = load_via_pipeline(tensors, Quantization::default(), &tiny_config(&[]));
		assert!(model.has_mtp());
	}

	/// emelex patch: `mtp_use_dedicated_embeddings: true` → warn + skip.
	#[test]
	fn mtp_dedicated_embeddings_skips() {
		let mut tensors = backbone_tensors();
		insert_mtp_tensors(&mut tensors, "language_model.mtp");
		let config = tiny_config(&[("mtp_use_dedicated_embeddings", serde_json::json!(true))]);
		let model = load_via_pipeline(tensors, Quantization::default(), &config);
		assert_backbone_only(&model);
	}

	/// emelex patch: `mtp_num_hidden_layers != 1` → warn + skip.
	#[test]
	fn mtp_multi_layer_config_skips() {
		let mut tensors = backbone_tensors();
		insert_mtp_tensors(&mut tensors, "language_model.mtp");
		let config = tiny_config(&[("mtp_num_hidden_layers", serde_json::json!(2))]);
		let model = load_via_pipeline(tensors, Quantization::default(), &config);
		assert_backbone_only(&model);
	}

	/// emelex patch: an incomplete layer-0 sentinel set → warn + skip.
	#[test]
	fn mtp_missing_sentinel_skips() {
		let mut tensors = backbone_tensors();
		insert_mtp_tensors(&mut tensors, "language_model.mtp");
		tensors
			.remove("language_model.mtp.layers.0.self_attn.k_norm.weight")
			.unwrap();
		let model = load_via_pipeline(tensors, Quantization::default(), &tiny_config(&[]));
		assert_backbone_only(&model);
	}

	/// emelex patch: an ungated q_proj (rows == n_heads * head_dim, not
	/// 2x) fails the gate-shape check → warn + skip.
	#[test]
	fn mtp_gate_shape_mismatch_skips() {
		let mut tensors = backbone_tensors();
		insert_mtp_tensors(&mut tensors, "language_model.mtp");
		// BF16 so the gate-shape check (not the dtype guard) is what fires.
		tensors.insert(
			"language_model.mtp.layers.0.self_attn.q_proj.weight".to_string(),
			arr_bf16(300.0, &[32, 32]),
		);
		let model = load_via_pipeline(tensors, Quantization::default(), &tiny_config(&[]));
		assert_backbone_only(&model);
	}

	/// emelex patch: `attn_output_gate: false` → warn + skip.
	#[test]
	fn mtp_attn_output_gate_false_skips() {
		let mut tensors = backbone_tensors();
		insert_mtp_tensors(&mut tensors, "language_model.mtp");
		let config = tiny_config(&[("attn_output_gate", serde_json::json!(false))]);
		let model = load_via_pipeline(tensors, Quantization::default(), &config);
		assert_backbone_only(&model);
	}

	/// emelex patch: dense-BF16 guards — a `.scales` companion under the
	/// MTP prefix, and a quantization override naming an MTP tensor
	/// (canonicalized through sanitize's identical quant-key mapping),
	/// each warn + skip.
	#[test]
	fn mtp_quantized_guard_skips() {
		// Case a: scales companion tensor.
		let mut tensors = backbone_tensors();
		insert_mtp_tensors(&mut tensors, "language_model.mtp");
		tensors.insert(
			"language_model.mtp.layers.0.self_attn.q_proj.scales".to_string(),
			arr_bf16(400.0, &[64, 4]),
		);
		let model = load_via_pipeline(tensors, Quantization::default(), &tiny_config(&[]));
		assert_backbone_only(&model);

		// Case b: quantization override naming an MTP tensor, shipped
		// under an alternate prefix so the override key must ride the
		// same canonicalization as the tensor keys.
		let mut tensors = backbone_tensors();
		insert_mtp_tensors(&mut tensors, "language_model.mtp");
		let quant = Quantization {
			default: None,
			per_layer: HashMap::from([(
				"language_model.mtp.layers.0.self_attn.q_proj".to_string(),
				LayerOverride::Params(QuantParams {
					group_size: 64,
					bits: 4,
					mode: QuantMode::Affine,
				}),
			)]),
		};
		let model = load_via_pipeline(tensors, quant, &tiny_config(&[]));
		assert_backbone_only(&model);
	}

	/// emelex patch: a MoE key under the MTP prefix → warn + skip.
	#[test]
	fn mtp_moe_guard_skips() {
		let mut tensors = backbone_tensors();
		insert_mtp_tensors(&mut tensors, "language_model.mtp");
		tensors.insert(
			"language_model.mtp.layers.0.mlp.switch_mlp.gate_proj.weight".to_string(),
			arr_bf16(500.0, &[1, 64, 32]),
		);
		let model = load_via_pipeline(tensors, Quantization::default(), &tiny_config(&[]));
		assert_backbone_only(&model);
	}

	/// emelex patch: happy path — a complete tiny dense MTP set loads
	/// (mtp_num_hidden_layers absent → treated as 1); forward_hidden's
	/// logits are element-identical to forward's; forward_mtp returns
	/// `[1, L, H]` recycle + `[1, L, V]` logits and advances pairs_fed by
	/// L per call; truncate_to rolls back and rejects overshoot.
	#[test]
	fn mtp_happy_path_loads_and_runs() {
		let mut tensors = backbone_tensors();
		insert_mtp_tensors(&mut tensors, "language_model.mtp");
		let model = load_via_pipeline(tensors, Quantization::default(), &tiny_config(&[]));
		assert!(model.has_mtp());

		let ids = test_ids();
		let mut plain_caches = model.new_caches();
		let mut hidden_caches = model.new_caches();
		let plain = model.forward(&ids, &mut plain_caches).unwrap();
		let out = model.forward_hidden(&ids, &mut hidden_caches).unwrap();
		assert_eq!(out.hidden_pre_norm.shape(), vec![1, 3, 32]);
		assert_eq!(
			plain.to_vec_f32().unwrap(),
			out.logits.to_vec_f32().unwrap(),
			"forward and forward_hidden must be element-identical"
		);

		let mut mtp_caches = model.new_mtp_caches();
		assert_eq!(mtp_caches.pairs_fed(), 0);
		let step = model
			.forward_mtp(&ids, &out.hidden_pre_norm, &mut mtp_caches)
			.unwrap();
		assert_eq!(step.recycle_hidden.shape(), vec![1, 3, 32]);
		assert_eq!(step.logits.shape(), vec![1, 3, 16]);
		assert_eq!(mtp_caches.pairs_fed(), 3);
		assert!(
			step.recycle_hidden
				.to_vec_f32()
				.unwrap()
				.iter()
				.chain(step.logits.to_vec_f32().unwrap().iter())
				.all(|v| v.is_finite())
		);

		// Recursive one-token step: consume the last recycle-hidden row.
		let one = Array::from_slice(&[4i32], &[1, 1]).unwrap();
		let last_hidden = ops::slice(&step.recycle_hidden, &[0, 2, 0], &[1, 3, 32]).unwrap();
		let step2 = model
			.forward_mtp(&one, &last_hidden, &mut mtp_caches)
			.unwrap();
		assert_eq!(step2.recycle_hidden.shape(), vec![1, 1, 32]);
		assert_eq!(step2.logits.shape(), vec![1, 1, 16]);
		assert_eq!(mtp_caches.pairs_fed(), 4);

		let mut snapshot = mtp_caches.clone();
		snapshot.truncate_to(3).unwrap();
		assert_eq!(snapshot.pairs_fed(), 3);
		assert!(snapshot.truncate_to(10).is_err());
	}

	/// BF16 dtype guard: an F16 or F32
	/// substitution at ANY of the 15 MTP tensor classes warns and skips
	/// MTP while the backbone loads.
	#[test]
	fn mtp_dtype_substitution_skips_per_tensor_class() {
		for (name, shape) in mtp_tensor_specs() {
			for dtype in [Dtype::Float16, Dtype::Float32] {
				let mut tensors = backbone_tensors();
				insert_mtp_tensors(&mut tensors, "language_model.mtp");
				let substituted = ops::astype(&arr(600.0, shape), dtype).unwrap();
				tensors.insert(format!("language_model.mtp.{name}"), substituted);
				let model = load_via_pipeline(tensors, Quantization::default(), &tiny_config(&[]));
				assert!(
					!model.has_mtp(),
					"{name} substituted as {dtype:?} must warn-and-skip MTP"
				);
			}
		}

		// One representative substitution with the full backbone-unchanged
		// assertion: byte-identical logits vs a clean no-MTP load.
		let mut tensors = backbone_tensors();
		insert_mtp_tensors(&mut tensors, "language_model.mtp");
		tensors.insert(
			"language_model.mtp.fc.weight".to_string(),
			arr(601.0, &[32, 64]),
		);
		let model = load_via_pipeline(tensors, Quantization::default(), &tiny_config(&[]));
		let skip_logits = assert_backbone_only(&model);
		let clean = load_via_pipeline(
			backbone_tensors(),
			Quantization::default(),
			&tiny_config(&[]),
		);
		assert_eq!(skip_logits, assert_backbone_only(&clean));
	}

	/// emelex patch: any unexpected extra key under the MTP prefix → warn
	/// + skip (v1 expects exactly the 15-tensor dense layer-0 set).
	#[test]
	fn mtp_unexpected_extra_key_skips() {
		let mut tensors = backbone_tensors();
		insert_mtp_tensors(&mut tensors, "language_model.mtp");
		tensors.insert(
			"language_model.mtp.layers.1.input_layernorm.weight".to_string(),
			arr_bf16(700.0, &[32]),
		);
		let model = load_via_pipeline(tensors, Quantization::default(), &tiny_config(&[]));
		assert_backbone_only(&model);
	}

	/// Standalone-sidecar guard: a directory whose
	/// config carries `model_type = "qwen3_5_mtp"` (the pinned standalone
	/// MTP artifact layout) is NOT a loadable model: `Model::load` returns
	/// a clear error naming the condition, never a partial load.
	#[test]
	fn standalone_sidecar_dir_is_not_loadable() {
		let dir = std::env::temp_dir().join(format!(
			"emelex-mtp-standalone-sidecar-guard-{}",
			std::process::id()
		));
		std::fs::create_dir_all(&dir).unwrap();
		std::fs::write(
			dir.join("config.json"),
			r#"{"model_type": "qwen3_5_mtp", "mtp_num_hidden_layers": 1}"#,
		)
		.unwrap();
		let result = Model::load(&dir);
		std::fs::remove_dir_all(&dir).ok();
		let err = match result {
			Err(e) => e.to_string(),
			Ok(_) => panic!("standalone MTP sidecar directory must not load"),
		};
		assert!(
			err.contains("qwen3_5_mtp"),
			"error must name the condition: {err}"
		);
		assert!(
			err.contains("not a loadable model"),
			"error must state non-loadability: {err}"
		);
	}

	/// emelex patch: the new MTP config fields parse with the documented
	/// defaults, from text_config, and (for mtp_num_hidden_layers) via the
	/// tolerated config-root fallback.
	#[test]
	fn mtp_config_fields_parse() {
		let plain = Qwen35Config::from_json(&tiny_config(&[])).unwrap();
		assert!(plain.attn_output_gate);
		assert_eq!(plain.mtp_num_hidden_layers, None);
		assert!(!plain.mtp_use_dedicated_embeddings);

		let nested: Value = serde_json::json!({
			"model_type": "qwen3_5",
			"mtp_num_hidden_layers": 1,
			"text_config": {
				"hidden_size": 32,
				"num_hidden_layers": 2,
				"num_attention_heads": 2,
				"vocab_size": 16,
				"attn_output_gate": false,
				"mtp_use_dedicated_embeddings": true
			}
		});
		let parsed = Qwen35Config::from_json(&nested).unwrap();
		assert!(!parsed.attn_output_gate);
		assert_eq!(parsed.mtp_num_hidden_layers, Some(1));
		assert!(parsed.mtp_use_dedicated_embeddings);
	}

	#[test]
	fn qwen3_6_moe_config_parses() {
		let config: Value = serde_json::json!({
			"architectures": ["Qwen3_5MoeForConditionalGeneration"],
			"model_type": "qwen3_5_moe",
			"text_config": {
				"attn_output_gate": true,
				"full_attention_interval": 4,
				"head_dim": 256,
				"hidden_size": 2048,
				"linear_conv_kernel_dim": 4,
				"linear_key_head_dim": 128,
				"linear_num_key_heads": 16,
				"linear_num_value_heads": 32,
				"linear_value_head_dim": 128,
				"model_type": "qwen3_5_moe_text",
				"moe_intermediate_size": 512,
				"num_attention_heads": 16,
				"num_experts": 256,
				"num_experts_per_tok": 8,
				"num_hidden_layers": 40,
				"num_key_value_heads": 2,
				"rms_norm_eps": 1e-6,
				"rope_parameters": {
					"mrope_interleaved": true,
					"mrope_section": [11, 11, 10],
					"partial_rotary_factor": 0.25,
					"rope_theta": 10_000_000,
					"rope_type": "default"
				},
				"shared_expert_intermediate_size": 512,
				"tie_word_embeddings": false,
				"vocab_size": 248320
			}
		});
		let parsed = Qwen35Config::from_json(&config).expect("parse");
		assert_eq!(parsed.hidden_size, 2048);
		assert_eq!(parsed.num_hidden_layers, 40);
		assert_eq!(parsed.num_experts, 256);
		assert_eq!(parsed.num_experts_per_tok, 8);
		assert_eq!(parsed.moe_intermediate_size, 512);
		assert_eq!(parsed.shared_expert_intermediate_size, 512);
		// Every layer is MoE in Qwen3.6-A3B: decoder_sparse_step is
		// absent from the config and must default to 1, and the absent
		// intermediate_size must not be required (no dense MLP layer
		// exists to consume it).
		assert_eq!(parsed.decoder_sparse_step, 1);
		assert_eq!(parsed.intermediate_size, 0);
		// norm_topk_prob is absent from published configs; the default
		// must stay `true` (Qwen3-Next lineage renormalizes top-k).
		assert!(parsed.norm_topk_prob);
	}
}
