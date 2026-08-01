//! Concrete model architectures and the loader that dispatches between them
//! based on `config.json`'s `model_type`.

pub mod base;
pub mod cache;
pub mod config;
pub mod dhara;
pub mod gated_delta;
pub mod gemma3;
pub mod gemma4;
pub mod laguna;
pub mod mamba2;
pub mod moe;
// emelex patch (not upstream): generic MTP (multi-token-prediction) types.
pub mod mtp;
pub mod nemotron;
pub mod qwen2;
pub mod qwen3;
pub mod qwen3_5;

use std::path::Path;

use cache::LayerCache;
use mtp::{BackboneOutput, MtpCaches, MtpStepOutput};
use serde_json::Value;

use crate::engine::{
	Cancellation,
	array::Array,
	error::{Error, Result},
	media::{audio::ProcessedAudio, image::ProcessedImage},
	nn::WeightMap,
	quant::Quantization,
	weights,
};

/// A loaded, ready-to-run causal language model.
///
/// New architectures are added as variants here; `forward` dispatches to
/// the concrete implementation. Keeping one enum (rather than a trait
/// object) avoids `dyn`-safety friction around the differing per-layer
/// cache types each architecture needs.
pub enum Model {
	Qwen2(qwen2::Qwen2Model),
	Qwen3(qwen3::Qwen3Model),
	Qwen35(qwen3_5::Qwen35Model),
	Gemma3(gemma3::Gemma3Model),
	Gemma4(gemma4::Gemma4Model),
	NemotronH(nemotron::NemotronModel),
	Dhara(dhara::DharaModel),
	Laguna(laguna::LagunaModel),
}

impl Model {
	/// Load a model directory (expects `config.json` + safetensors shards).
	pub fn load(model_dir: &Path) -> Result<Self> {
		let runtime = crate::runtime::initialize_default_if_needed()
			.map_err(|error| Error::Mlx(error.to_string()))?;
		let mut snapshot = crate::model::layout::CheckpointSnapshot::open_in(
			model_dir,
			&runtime.home().join("temp"),
		)
		.map_err(|error| Error::Config(error.to_string()))?;
		let allow_mtp = crate::engine::mtp_certification::model_is_certified(&snapshot)?;
		Self::load_snapshot(&mut snapshot, model_dir, allow_mtp)
	}

	/// Load from the descriptor-backed snapshot shared with runtime
	/// certification. No model-owned path is reopened after snapshot creation.
	/// MTP tensor payloads are selected for evaluation only when `allow_mtp` is
	/// already backed by the caller's exact certificate policy.
	pub(crate) fn load_snapshot(
		snapshot: &mut crate::model::layout::CheckpointSnapshot,
		model_dir: &Path,
		allow_mtp: bool,
	) -> Result<Self> {
		let config_json: Value = serde_json::from_slice(snapshot.config_bytes())
			.map_err(|e| Error::Config(format!("bad config.json: {e}")))?;

		let model_type = config_json
			.get("model_type")
			.and_then(|v| v.as_str())
			.ok_or_else(|| {
				Error::Config(
					"config.json has no string model_type; refusing to guess an \
					 architecture"
						.to_string(),
				)
			})?;

		// emelex patch (not upstream): standalone-sidecar guard.
		// `model_type = "qwen3_5_mtp"` marks the pinned standalone
		// MTP artifact layout — bare-root MTP tensor keys, no backbone, no
		// embeddings/head. It is not a loadable model; error before any
		// tensor I/O so a partial load is impossible.
		if model_type == "qwen3_5_mtp" {
			return Err(Error::Model(
				"model_type 'qwen3_5_mtp' is a standalone Qwen3.5 MTP sidecar \
				 artifact (no backbone, no embeddings/head) and is not a loadable \
				 model; load a converted dense checkpoint that ships its MTP weights \
				 under 'language_model.mtp.*' instead"
					.into(),
			));
		}

		config::validate_checkpoint_config(&config_json)?;
		let quant = Quantization::from_config(&config_json)?;
		// The text-only gemma3 port never loads multimodal towers — skip
		// their tensors before materialization (excluded lazy handles are
		// freed) rather than paying peak memory for weights `sanitize`
		// would immediately drop.
		let skip_multimodal = matches!(model_type, "gemma3" | "gemma3_text");
		let tensors = weights::load_snapshot(snapshot, model_dir, |name| {
			(allow_mtp || !weights::is_mtp_tensor_name(name))
				&& !(skip_multimodal
					&& (name.starts_with("vision_tower.")
						|| name.starts_with("multi_modal_projector.")))
		})?;
		let mut weight_map = WeightMap::new(tensors, quant);

		match model_type {
			"qwen2" => {
				let tie = config_json
					.get("tie_word_embeddings")
					.and_then(|v| v.as_bool())
					.unwrap_or(true);
				qwen2::sanitize(&mut weight_map, tie);
				let model = qwen2::Qwen2Model::load(weight_map, &config_json)?;
				Ok(Model::Qwen2(model))
			}
			"qwen3" => {
				let tie = config_json
					.get("tie_word_embeddings")
					.and_then(|v| v.as_bool())
					.unwrap_or(true);
				qwen3::sanitize(&mut weight_map, tie);
				let model = qwen3::Qwen3Model::load(weight_map, &config_json)?;
				Ok(Model::Qwen3(model))
			}
			"gemma3" | "gemma3_text" => {
				gemma3::sanitize(&mut weight_map);
				let model = gemma3::Gemma3Model::load(weight_map, &config_json)?;
				Ok(Model::Gemma3(model))
			}
			"gemma4" | "gemma4_text" | "gemma4_unified" | "gemma4_unified_text" => {
				gemma4::sanitize(&mut weight_map);
				let model = gemma4::Gemma4Model::load(weight_map, &config_json)?;
				Ok(Model::Gemma4(model))
			}
			"qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text" => {
				let text_cfg = config::text_config(&config_json);
				let num_hidden_layers = config::require_i32(text_cfg, "num_hidden_layers")?;
				let num_experts = config::get_i32(text_cfg, "num_experts", 0)?;
				// emelex patch (not upstream): MTP detection (raw-HF
				// predicate, then namespace resolution — sole supported
				// on-disk prefix `language_model.mtp`) must run BEFORE
				// sanitize on the raw key set — sanitize deletes every
				// unpreserved MTP key, so this order is load-bearing.
				let mtp = if allow_mtp {
					qwen3_5::detect_mtp(&weight_map)
				} else {
					mtp::MtpDetection::None
				};
				qwen3_5::sanitize(&mut weight_map, num_hidden_layers, num_experts, mtp);
				let model = qwen3_5::Qwen35Model::load(weight_map, &config_json)?;
				Ok(Model::Qwen35(model))
			}
			"nemotron_h" => {
				nemotron::sanitize(&mut weight_map);
				let model = nemotron::NemotronModel::load(weight_map, &config_json)?;
				Ok(Model::NemotronH(model))
			}
			"llama" => {
				// Vanilla Llama-style GQA checkpoints (e.g. MiniCPM5) are
				// structurally identical to our Qwen2 implementation
				// (RoPE, SwiGLU MLP, RMSNorm, optional qkv bias, optional
				// tied lm_head) - reuse it rather than duplicating code.
				let tie = config_json
					.get("tie_word_embeddings")
					.and_then(|v| v.as_bool())
					.unwrap_or(true);
				qwen2::sanitize(&mut weight_map, tie);
				let model = qwen2::Qwen2Model::load(weight_map, &config_json)?;
				Ok(Model::Qwen2(model))
			}
			"dhara_ar" => {
				let tie = config_json
					.get("tie_word_embeddings")
					.and_then(|v| v.as_bool())
					.unwrap_or(true);
				dhara::sanitize(&mut weight_map, tie);
				let model = dhara::DharaModel::load(weight_map, &config_json)?;
				Ok(Model::Dhara(model))
			}
			"laguna" => {
				// Laguna MLX checkpoints ship weights already in the
				// engine's layout (stacked `switch_mlp.*` expert tensors,
				// `gate.e_score_correction_bias`) - no sanitize needed.
				let model = laguna::LagunaModel::load(weight_map, &config_json)?;
				Ok(Model::Laguna(model))
			}
			other => Err(Error::Model(format!(
				"unsupported model_type '{other}' (supported: qwen2, qwen3, qwen3_5, \
				 qwen3_5_moe, gemma3, gemma4, gemma4_unified, nemotron_h, llama, \
				 dhara_ar, laguna)"
			))),
		}
	}

	pub fn new_caches(&self) -> Vec<LayerCache> {
		match self {
			Model::Qwen2(m) => m.new_caches(),
			Model::Qwen3(m) => m.new_caches(),
			Model::Qwen35(m) => m.new_caches(),
			Model::Gemma3(m) => m.new_caches(),
			Model::Gemma4(m) => m.new_caches(),
			Model::NemotronH(m) => m.new_caches(),
			Model::Dhara(m) => m.new_caches(),
			Model::Laguna(m) => m.new_caches(),
		}
	}

	/// Run one forward pass over `input_ids` (`[B, L]`), returning logits
	/// (`[B, L, vocab]`).
	pub fn forward(&self, input_ids: &Array, caches: &mut [LayerCache]) -> Result<Array> {
		match self {
			Model::Qwen2(m) => m.forward(input_ids, caches),
			Model::Qwen3(m) => m.forward(input_ids, caches),
			Model::Qwen35(m) => m.forward(input_ids, caches),
			Model::Gemma3(m) => m.forward(input_ids, caches),
			Model::Gemma4(m) => m.forward(input_ids, caches),
			Model::NemotronH(m) => m.forward(input_ids, caches),
			Model::Dhara(m) => m.forward(input_ids, caches),
			Model::Laguna(m) => m.forward(input_ids, caches),
		}
	}

	/// Whether this checkpoint's MTP (multi-token-prediction) module was
	/// loaded (Qwen3.5-only in v1).
	///
	/// emelex patch (not upstream).
	pub fn has_mtp(&self) -> bool {
		match self {
			Model::Qwen35(m) => m.has_mtp(),
			_ => false,
		}
	}

	/// Fresh working caches for the MTP module (empty for architectures
	/// without MTP support).
	///
	/// emelex patch (not upstream).
	pub fn new_mtp_caches(&self) -> MtpCaches {
		match self {
			Model::Qwen35(m) => m.new_mtp_caches(),
			_ => MtpCaches(Vec::new()),
		}
	}

	/// Run one forward pass over `input_ids` (`[B, L]`), returning both the
	/// pre-final-norm decoder-stack hidden states and the logits (see
	/// [`BackboneOutput`]). Errors on architectures without MTP support.
	///
	/// emelex patch (not upstream).
	pub fn forward_hidden(
		&self,
		input_ids: &Array,
		caches: &mut [LayerCache],
	) -> Result<BackboneOutput> {
		match self {
			Model::Qwen35(m) => m.forward_hidden(input_ids, caches),
			_ => Err(Error::Model(
				"forward_hidden: this architecture has no MTP support".into(),
			)),
		}
	}

	/// Run one MTP step over `input_ids` (`[1, L]`) and `prev_hidden`
	/// (`[1, L, H]`), returning the recycle hidden + logits (see
	/// [`MtpStepOutput`]). Errors on architectures without MTP support or
	/// checkpoints without a loaded MTP module.
	///
	/// emelex patch (not upstream).
	pub fn forward_mtp(
		&self,
		input_ids: &Array,
		prev_hidden: &Array,
		caches: &mut MtpCaches,
	) -> Result<MtpStepOutput> {
		match self {
			Model::Qwen35(m) => m.forward_mtp(input_ids, prev_hidden, caches),
			_ => Err(Error::Model(
				"forward_mtp: this architecture has no MTP support".into(),
			)),
		}
	}

	/// Debug-only helper (see `NemotronModel::debug_layer_stats`).
	pub fn debug_nemotron_layer_stats(&self, input_ids: &Array) -> Result<Vec<(f32, f32)>> {
		match self {
			Model::NemotronH(m) => m.debug_layer_stats(input_ids),
			_ => Err(Error::Model(
				"debug_nemotron_layer_stats: not a NemotronH model".into(),
			)),
		}
	}

	/// Whether this model was loaded with image support (a `vision_config`
	/// in `config.json` plus matching vision tower weights).
	pub fn supports_images(&self) -> bool {
		match self {
			Model::Gemma4(m) => m.supports_images(),
			Model::Qwen35(m) => m.supports_images(),
			_ => false,
		}
	}

	/// `(patch_size, max_soft_tokens, pooling_kernel_size)` for
	/// [`crate::engine::media::image::preprocess_image_bytes`], or `None` if this
	/// model has no image support.
	pub fn image_processing_params(&self) -> Option<(i32, i32, i32)> {
		match self {
			Model::Gemma4(m) => m.image_processing_params(),
			Model::Qwen35(m) => m.image_processing_params(),
			_ => None,
		}
	}

	/// `(image_token_id, boi_token_id, eoi_token_id)`, or `None` if this
	/// model has no image support.
	pub fn image_token_ids(&self) -> Option<(u32, u32, u32)> {
		match self {
			Model::Gemma4(m) => m.image_token_ids(),
			Model::Qwen35(m) => m.image_token_ids(),
			_ => None,
		}
	}

	/// Whether this model was loaded with audio support (an `audio_config`
	/// in `config.json` plus matching audio tower weights).
	pub fn supports_audio(&self) -> bool {
		match self {
			Model::Gemma4(m) => m.supports_audio(),
			_ => false,
		}
	}

	/// `(audio_token_id, boa_token_id, eoa_token_id)`, or `None` if this
	/// model has no audio support.
	pub fn audio_token_ids(&self) -> Option<(u32, u32, u32)> {
		match self {
			Model::Gemma4(m) => m.audio_token_ids(),
			_ => None,
		}
	}

	/// Raw PCM samples per audio token for the encoder-free "unified"
	/// audio path (see
	/// `crate::engine::media::audio::preprocess_audio_bytes_raw`), or `None` if
	/// this model has no audio support or uses the classic mel-spectrogram tower
	/// instead.
	pub fn audio_samples_per_token(&self) -> Option<i32> {
		match self {
			Model::Gemma4(m) => m.audio_samples_per_token(),
			_ => None,
		}
	}

	/// The chat template's video placeholder token id, or `None` if this
	/// model has no vision support (video frames reuse the vision tower).
	pub fn video_token_id(&self) -> Option<u32> {
		match self {
			Model::Gemma4(m) => m.video_token_id(),
			Model::Qwen35(m) => m.video_token_id(),
			_ => None,
		}
	}

	/// Run one forward pass over `input_ids` (`[B, L]`), splicing `images`'
	/// projected vision features in at `image_token_id` placeholder
	/// positions before the decoder stack. Errors if this model has no
	/// image support.
	pub fn forward_with_images(
		&self,
		input_ids: &Array,
		images: &[ProcessedImage],
		caches: &mut [LayerCache],
	) -> Result<Array> {
		self.forward_with_media(input_ids, images, &[], caches)
	}

	/// Run one forward pass over `input_ids` (`[B, L]`), splicing image
	/// and/or audio features in at their placeholder positions before the
	/// decoder stack (video frames arrive as ordinary `images` entries).
	/// Errors if a modality is supplied that this model doesn't support.
	pub fn forward_with_media(
		&self,
		input_ids: &Array,
		images: &[ProcessedImage],
		audios: &[ProcessedAudio],
		caches: &mut [LayerCache],
	) -> Result<Array> {
		match self {
			Model::Gemma4(m) => m.forward_with_media(input_ids, images, audios, caches),
			Model::Qwen35(m) => {
				if !audios.is_empty() {
					return Err(Error::Model(
						"qwen3.5: model has no audio support (no audio_config)".into(),
					));
				}
				m.forward_with_media(input_ids, images, caches)
			}
			_ => Err(Error::Model(
				"forward_with_media: model has no multimodal support".into(),
			)),
		}
	}

	/// Cooperative multimodal prefill. Media projection work is evaluated
	/// at per-item boundaries and the fused decoder stream is evaluated in
	/// bounded token chunks. Disabled cancellation preserves the public
	/// [`Model::forward_with_media`] one-pass behavior.
	pub(crate) fn forward_with_media_cancellable(
		&self,
		input_ids: &Array,
		images: &[ProcessedImage],
		audios: &[ProcessedAudio],
		caches: &mut [LayerCache],
		chunk_tokens: usize,
		cancellation: Cancellation<'_>,
	) -> Result<Array> {
		if !cancellation.is_cooperative() {
			return self.forward_with_media(input_ids, images, audios, caches);
		}
		match self {
			Model::Gemma4(model) => model.forward_with_media_cancellable(
				input_ids,
				images,
				audios,
				caches,
				chunk_tokens,
				cancellation,
			),
			Model::Qwen35(model) => {
				if !audios.is_empty() {
					return Err(Error::Model(
						"qwen3.5: model has no audio support (no audio_config)".into(),
					));
				}
				model.forward_with_media_cancellable(
					input_ids,
					images,
					caches,
					chunk_tokens,
					cancellation,
				)
			}
			_ => Err(Error::Model(
				"forward_with_media: model has no multimodal support".into(),
			)),
		}
	}
}
