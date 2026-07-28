//! Common config parsing helpers shared across architectures.

use serde_json::Value;

use crate::engine::error::{Error, Result};

const MAX_CONFIG_DEPTH: usize = 16;
const MAX_CONFIG_NODES: usize = 262_144;
const MAX_CONFIG_COLLECTION_ITEMS: usize = 65_536;
const MAX_CONFIG_KEY_BYTES: usize = 1_024;
const MAX_LAYERS: i32 = 1_024;
const MAX_HEADS: i32 = 16_384;
const MAX_DIMENSION: i32 = 16_777_216;
const MAX_VOCABULARY: i32 = 16_777_216;
const MAX_EXPERTS: i32 = 4_096;
const MAX_KERNEL: i32 = 65_536;
const MAX_IMAGE_PATCH: i32 = 256;
const MAX_IMAGE_POOL: i32 = 16;
const MIN_SOFT_TOKENS: i32 = 40;
const MAX_SOFT_TOKENS: i32 = 16_384;

pub fn get_str<'a>(cfg: &'a Value, key: &str) -> Result<Option<&'a str>> {
	cfg.get(key)
		.map(|value| {
			value
				.as_str()
				.ok_or_else(|| Error::Config(format!("field '{key}' must be a string")))
		})
		.transpose()
}

pub fn get_i32(cfg: &Value, key: &str, default: i32) -> Result<i32> {
	optional_i32(cfg, key).map(|value| value.unwrap_or(default))
}

pub fn require_i32(cfg: &Value, key: &str) -> Result<i32> {
	let value = cfg
		.get(key)
		.and_then(|v| v.as_i64())
		.ok_or_else(|| Error::Config(format!("missing required integer field '{key}'")))?;
	let value = i32::try_from(value)
		.map_err(|_| Error::Config(format!("integer field '{key}' is outside the i32 range")))?;
	if value <= 0 {
		return Err(Error::Config(format!(
			"integer field '{key}' must be positive"
		)));
	}
	Ok(value)
}

pub fn optional_i32(cfg: &Value, key: &str) -> Result<Option<i32>> {
	cfg.get(key)
		.map(|value| {
			let value = value
				.as_i64()
				.ok_or_else(|| Error::Config(format!("field '{key}' must be an integer")))?;
			i32::try_from(value).map_err(|_| {
				Error::Config(format!("integer field '{key}' is outside the i32 range"))
			})
		})
		.transpose()
}

pub fn get_f32(cfg: &Value, key: &str, default: f32) -> Result<f32> {
	optional_f32(cfg, key).map(|value| value.unwrap_or(default))
}

pub fn optional_f32(cfg: &Value, key: &str) -> Result<Option<f32>> {
	cfg.get(key).map(|value| finite_f32(value, key)).transpose()
}

pub fn get_bool(cfg: &Value, key: &str, default: bool) -> Result<bool> {
	cfg.get(key)
		.map(|value| {
			value
				.as_bool()
				.ok_or_else(|| Error::Config(format!("field '{key}' must be a boolean")))
		})
		.transpose()
		.map(|value| value.unwrap_or(default))
}

/// Read `config.json`, following the `text_config` nesting used by
/// multimodal checkpoints (Gemma4, Qwen3.5-VL, ...) when the requested key
/// is not present at the top level.
pub fn text_config(cfg: &Value) -> &Value {
	cfg.get("text_config").unwrap_or(cfg)
}

/// Validate all configuration values used to construct native MLX shapes.
///
/// This must run before checkpoint files are opened. Architecture parsers
/// intentionally stay small and assume this preflight has established
/// bounded, positive dimensions and safe cross-field geometry.
pub fn validate_checkpoint_config(root: &Value) -> Result<()> {
	let mut nodes = 0;
	validate_json_value(root, 0, &mut nodes)?;
	let root_object = root
		.as_object()
		.ok_or_else(|| Error::Config("config.json root must be an object".to_string()))?;
	let model_type = root_object
		.get("model_type")
		.and_then(Value::as_str)
		.ok_or_else(|| Error::Config("config.json has no string model_type".to_string()))?;
	let text = text_config(root);
	if !text.is_object() {
		return Err(Error::Config("text_config must be an object".to_string()));
	}

	let geometry = validate_text_geometry(text)?;
	validate_experts(text, model_type)?;
	validate_token_ids(root)?;

	match model_type {
		"qwen2" | "llama" => {
			required_bounded(text, "intermediate_size", 1, MAX_DIMENSION)?;
			super::qwen2::Qwen2Config::from_json(text)?;
		}
		"qwen3" => {
			required_bounded(text, "intermediate_size", 1, MAX_DIMENSION)?;
			super::qwen3::Qwen3Config::from_json(text)?;
		}
		"qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text" => {
			validate_qwen35(text, geometry)?;
			validate_qwen35_rope(text, geometry.head_dim)?;
			super::qwen3_5::Qwen35Config::from_json(root)?;
		}
		"gemma4" | "gemma4_text" | "gemma4_unified" | "gemma4_unified_text" => {
			validate_gemma4(text, geometry)?;
			super::gemma4::Gemma4Config::from_json(root)?;
		}
		"nemotron_h" => {
			validate_nemotron(text, geometry)?;
			super::nemotron::NemotronConfig::from_json(text)?;
		}
		"dhara_ar" => {
			required_bounded(text, "intermediate_size", 1, MAX_DIMENSION)?;
			optional_bounded(text, "canon_kernel", 1, MAX_KERNEL)?;
			super::dhara::DharaConfig::from_json(text)?;
		}
		"laguna" => {
			validate_laguna(text, geometry)?;
			validate_laguna_rope(text, geometry.head_dim)?;
			super::laguna::LagunaConfig::from_json(text)?;
		}
		"qwen3_5_mtp" => {
			return Err(Error::Config(
				"standalone qwen3_5_mtp sidecars have no loadable backbone".to_string(),
			));
		}
		other => {
			return Err(Error::Config(format!("unsupported model_type '{other}'")));
		}
	}

	validate_optional_media(root, model_type)?;
	crate::engine::quant::Quantization::from_config(root)?;
	Ok(())
}

#[derive(Clone, Copy)]
struct TextGeometry {
	hidden_size: i32,
	layers: i32,
	attention_heads: i32,
	kv_heads: i32,
	head_dim: i32,
}

fn validate_text_geometry(config: &Value) -> Result<TextGeometry> {
	let hidden_size = required_bounded(config, "hidden_size", 1, MAX_DIMENSION)?;
	let layers = required_bounded(config, "num_hidden_layers", 1, MAX_LAYERS)?;
	let attention_heads = required_bounded(config, "num_attention_heads", 1, MAX_HEADS)?;
	let head_dim = match optional_bounded(config, "head_dim", 1, MAX_DIMENSION)? {
		Some(value) => value,
		None => {
			if hidden_size % attention_heads != 0 {
				return Err(Error::Config(
					"hidden_size must be divisible by num_attention_heads when \
					 head_dim is absent"
						.to_string(),
				));
			}
			hidden_size / attention_heads
		}
	};
	if head_dim % 2 != 0 {
		return Err(Error::Config("head_dim must be even for RoPE".to_string()));
	}
	let kv_heads =
		optional_bounded(config, "num_key_value_heads", 1, MAX_HEADS)?.unwrap_or(attention_heads);
	if kv_heads > attention_heads || attention_heads % kv_heads != 0 {
		return Err(Error::Config(
			"num_key_value_heads must divide num_attention_heads".to_string(),
		));
	}
	required_bounded(config, "vocab_size", 1, MAX_VOCABULARY)?;
	for key in [
		"max_position_embeddings",
		"max_sequence_length",
		"seq_length",
		"model_max_length",
	] {
		optional_bounded(config, key, 1, MAX_DIMENSION)?;
	}
	checked_i32_product("attention projection width", &[attention_heads, head_dim])?;
	validate_positive_float(config, "rms_norm_eps")?;
	validate_positive_float(config, "layer_norm_epsilon")?;
	validate_positive_float(config, "rope_theta")?;
	validate_fraction(config, "partial_rotary_factor")?;
	Ok(TextGeometry {
		hidden_size,
		layers,
		attention_heads,
		kv_heads,
		head_dim,
	})
}

fn validate_experts(config: &Value, model_type: &str) -> Result<()> {
	let experts = optional_bounded(config, "num_experts", 0, MAX_EXPERTS)?.unwrap_or(0);
	let experts_per_token =
		optional_bounded(config, "num_experts_per_tok", 0, MAX_EXPERTS)?.unwrap_or(0);
	if experts == 0 {
		if experts_per_token != 0 {
			return Err(Error::Config(
				"num_experts_per_tok must be zero when num_experts is zero".to_string(),
			));
		}
		if matches!(
			model_type,
			"qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text"
		) {
			required_bounded(config, "intermediate_size", 1, MAX_DIMENSION)?;
		}
		return Ok(());
	}
	if experts_per_token == 0 || experts_per_token > experts {
		return Err(Error::Config(
			"num_experts_per_tok must be positive and no greater than num_experts".to_string(),
		));
	}
	required_bounded(config, "moe_intermediate_size", 1, MAX_DIMENSION)?;
	optional_bounded(config, "shared_expert_intermediate_size", 1, MAX_DIMENSION)?;
	validate_positive_float(config, "moe_routed_scaling_factor")
}

fn validate_qwen35(config: &Value, geometry: TextGeometry) -> Result<()> {
	optional_bounded(config, "full_attention_interval", 1, MAX_LAYERS)?;
	let value_heads =
		optional_bounded(config, "linear_num_value_heads", 1, MAX_HEADS)?.unwrap_or(64);
	let key_heads = optional_bounded(config, "linear_num_key_heads", 1, MAX_HEADS)?.unwrap_or(16);
	if value_heads % key_heads != 0 {
		return Err(Error::Config(
			"linear_num_key_heads must divide linear_num_value_heads".to_string(),
		));
	}
	let key_dim = optional_bounded(config, "linear_key_head_dim", 1, MAX_DIMENSION)?.unwrap_or(192);
	let value_dim =
		optional_bounded(config, "linear_value_head_dim", 1, MAX_DIMENSION)?.unwrap_or(128);
	optional_bounded(config, "linear_conv_kernel_dim", 1, MAX_KERNEL)?;
	optional_bounded(config, "decoder_sparse_step", 1, MAX_LAYERS)?;
	optional_bounded(config, "mtp_num_hidden_layers", 0, geometry.layers)?;
	let key_width = checked_i32_product("Qwen linear key width", &[key_heads, key_dim])?;
	let doubled_key_width = key_width
		.checked_mul(2)
		.ok_or_else(|| Error::Config("Qwen doubled linear key width exceeds i32".to_string()))?;
	let value_width = checked_i32_product("Qwen linear value width", &[value_heads, value_dim])?;
	doubled_key_width
		.checked_add(value_width)
		.ok_or_else(|| Error::Config("Qwen convolution width exceeds i32".to_string()))?;
	Ok(())
}

fn validate_gemma4(config: &Value, geometry: TextGeometry) -> Result<()> {
	let pattern = optional_bounded(config, "sliding_window_pattern", 1, MAX_LAYERS)?.unwrap_or(5);
	optional_bounded(config, "sliding_window", 1, i32::MAX)?;
	let global_head_dim =
		optional_bounded(config, "global_head_dim", 2, MAX_DIMENSION)?.unwrap_or(512);
	if global_head_dim % 2 != 0 {
		return Err(Error::Config(
			"global_head_dim must be even for RoPE".to_string(),
		));
	}
	let global_kv = optional_bounded(config, "num_global_key_value_heads", 1, MAX_HEADS)?
		.unwrap_or(geometry.kv_heads);
	if global_kv > geometry.attention_heads || geometry.attention_heads % global_kv != 0 {
		return Err(Error::Config(
			"num_global_key_value_heads must divide num_attention_heads".to_string(),
		));
	}
	checked_i32_product(
		"Gemma global attention projection width",
		&[geometry.attention_heads, global_head_dim],
	)?;
	let shared = optional_bounded(config, "num_kv_shared_layers", 0, geometry.layers)?.unwrap_or(0);
	optional_bounded(config, "vocab_size_per_layer_input", 0, MAX_VOCABULARY)?;
	optional_bounded(config, "hidden_size_per_layer_input", 0, MAX_DIMENSION)?;
	validate_positive_float(config, "final_logit_softcapping")?;
	validate_string_array(
		config,
		"layer_types",
		Some(geometry.layers),
		&["full_attention", "sliding_attention"],
	)?;
	let layer_types = gemma_layer_types(config, geometry.layers, pattern)?;
	let owner_layers = usize::try_from(geometry.layers - shared)
		.map_err(|_| Error::Config("invalid Gemma KV-sharing boundary".to_string()))?;
	if shared > 0 && owner_layers == 0 {
		return Err(Error::Config(
			"Gemma KV sharing requires at least one owning layer".to_string(),
		));
	}
	for (index, layer_type) in layer_types.iter().enumerate().skip(owner_layers) {
		if !layer_types[..owner_layers].contains(layer_type) {
			return Err(Error::Config(format!(
				"Gemma shared layer {index} has no earlier KV owner of type {layer_type}"
			)));
		}
	}

	let rope = optional_object(config, "rope_parameters", "rope_parameters")?;
	let full = optional_nested_object(rope, "full_attention", "rope_parameters.full_attention")?;
	let sliding = optional_nested_object(
		rope,
		"sliding_attention",
		"rope_parameters.sliding_attention",
	)?;
	validate_rope_values(
		full,
		global_head_dim,
		"rope_parameters.full_attention",
		0.25,
	)?;
	validate_rope_values(
		sliding,
		geometry.head_dim,
		"rope_parameters.sliding_attention",
		1.0,
	)
}

fn validate_nemotron(config: &Value, geometry: TextGeometry) -> Result<()> {
	required_bounded(config, "intermediate_size", 1, MAX_DIMENSION)?;
	let mamba_heads = required_bounded(config, "mamba_num_heads", 1, MAX_HEADS)?;
	let mamba_head_dim = required_bounded(config, "mamba_head_dim", 1, MAX_DIMENSION)?;
	let groups = optional_bounded(config, "n_groups", 1, MAX_HEADS)?.unwrap_or(1);
	let state_size = required_bounded(config, "ssm_state_size", 1, MAX_DIMENSION)?;
	optional_bounded(config, "conv_kernel", 1, MAX_KERNEL)?;
	if mamba_heads % groups != 0 {
		return Err(Error::Config(
			"n_groups must divide mamba_num_heads".to_string(),
		));
	}
	let intermediate =
		checked_i32_product("Mamba intermediate width", &[mamba_heads, mamba_head_dim])?;
	let grouped_state = checked_i32_product("Mamba grouped state", &[2, groups, state_size])?;
	intermediate
		.checked_add(grouped_state)
		.ok_or_else(|| Error::Config("Mamba convolution width exceeds i32".to_string()))?;

	let pattern = config
		.get("hybrid_override_pattern")
		.and_then(Value::as_str)
		.ok_or_else(|| Error::Config("nemotron_h lacks hybrid_override_pattern".to_string()))?;
	if pattern.chars().count()
		!= usize::try_from(geometry.layers)
			.map_err(|_| Error::Config("invalid layer count".to_string()))?
	{
		return Err(Error::Config(
			"hybrid_override_pattern length must equal num_hidden_layers".to_string(),
		));
	}
	if pattern.chars().any(|kind| !matches!(kind, 'M' | '*' | '-')) {
		return Err(Error::Config(
			"hybrid_override_pattern contains an unsupported layer type".to_string(),
		));
	}
	if let Some(limit) = config.get("time_step_limit") {
		let values = limit.as_array().ok_or_else(|| {
			Error::Config("time_step_limit must be a two-number array".to_string())
		})?;
		if values.len() != 2 {
			return Err(Error::Config(
				"time_step_limit must contain exactly two numbers".to_string(),
			));
		}
		let minimum = finite_f32(&values[0], "time_step_limit[0]")?;
		let maximum = finite_f32(&values[1], "time_step_limit[1]")?;
		if minimum < 0.0 || maximum < minimum {
			return Err(Error::Config(
				"time_step_limit must satisfy 0 <= min <= max".to_string(),
			));
		}
	}
	Ok(())
}

fn validate_laguna(config: &Value, geometry: TextGeometry) -> Result<()> {
	required_bounded(config, "intermediate_size", 1, MAX_DIMENSION)?;
	required_bounded(config, "num_experts", 1, MAX_EXPERTS)?;
	required_bounded(config, "num_experts_per_tok", 1, MAX_EXPERTS)?;
	required_bounded(config, "shared_expert_intermediate_size", 1, MAX_DIMENSION)?;
	optional_bounded(config, "sliding_window", 1, i32::MAX)?;

	if let Some(heads) = config.get("num_attention_heads_per_layer") {
		let heads = heads.as_array().ok_or_else(|| {
			Error::Config("num_attention_heads_per_layer must be an array".to_string())
		})?;
		if heads.len()
			!= usize::try_from(geometry.layers)
				.map_err(|_| Error::Config("invalid layer count".to_string()))?
		{
			return Err(Error::Config(
				"num_attention_heads_per_layer length must equal num_hidden_layers".to_string(),
			));
		}
		for (index, value) in heads.iter().enumerate() {
			let heads = bounded_value(
				value,
				&format!("num_attention_heads_per_layer[{index}]"),
				1,
				MAX_HEADS,
			)?;
			if heads % geometry.kv_heads != 0 {
				return Err(Error::Config(format!(
					"num_attention_heads_per_layer[{index}] must be divisible by \
					 num_key_value_heads"
				)));
			}
			checked_i32_product(
				"Laguna attention projection width",
				&[heads, geometry.head_dim],
			)?;
		}
	}
	validate_string_array(
		config,
		"layer_types",
		Some(geometry.layers),
		&["full_attention", "sliding_attention"],
	)?;
	validate_string_array(
		config,
		"gating_types",
		Some(geometry.layers),
		&["per-element", "per_element", "per-head", "per_head"],
	)?;
	validate_string_array(
		config,
		"mlp_layer_types",
		Some(geometry.layers),
		&["dense", "sparse"],
	)?;
	validate_positive_float(config, "moe_routed_scaling_factor")
}

fn validate_qwen35_rope(config: &Value, head_dim: i32) -> Result<()> {
	let rope = optional_object(config, "rope_parameters", "rope_parameters")?;
	validate_rope_values(rope, head_dim, "rope_parameters", 0.25)
}

fn validate_laguna_rope(config: &Value, head_dim: i32) -> Result<()> {
	let parameters = optional_object(config, "rope_parameters", "rope_parameters")?;
	let scaling = optional_object(config, "rope_scaling", "rope_scaling")?;
	let full = optional_nested_object(
		parameters,
		"full_attention",
		"rope_parameters.full_attention",
	)?
	.or(scaling);
	let sliding = optional_nested_object(
		parameters,
		"sliding_attention",
		"rope_parameters.sliding_attention",
	)?;
	validate_rope_values(full, head_dim, "Laguna full-attention RoPE", 1.0)?;
	validate_rope_values(sliding, head_dim, "Laguna sliding-attention RoPE", 1.0)
}

fn validate_optional_media(root: &Value, model_type: &str) -> Result<()> {
	if let Some(vision) = root.get("vision_config") {
		if !vision.is_object() {
			return Err(Error::Config("vision_config must be an object".to_string()));
		}
		let hidden = optional_bounded(vision, "hidden_size", 1, MAX_DIMENSION)?
			.or(optional_bounded(vision, "mm_embed_dim", 1, MAX_DIMENSION)?)
			.unwrap_or(768);
		let heads = optional_bounded(vision, "num_attention_heads", 1, MAX_HEADS)?
			.or(optional_bounded(vision, "num_heads", 1, MAX_HEADS)?)
			.unwrap_or(12);
		if hidden % heads != 0 {
			return Err(Error::Config(
				"vision hidden size must be divisible by its attention-head count".to_string(),
			));
		}
		optional_bounded(vision, "intermediate_size", 1, MAX_DIMENSION)?;
		optional_bounded(vision, "num_hidden_layers", 1, MAX_LAYERS)?;
		optional_bounded(vision, "depth", 1, MAX_LAYERS)?;
		let head_dim =
			optional_bounded(vision, "head_dim", 2, MAX_DIMENSION)?.unwrap_or(hidden / heads);
		let patch = optional_bounded(vision, "patch_size", 1, MAX_IMAGE_PATCH)?.unwrap_or(16);
		let pooling_key = if model_type.starts_with("qwen3_5") {
			"spatial_merge_size"
		} else {
			"pooling_kernel_size"
		};
		let pooling = optional_bounded(vision, pooling_key, 1, MAX_IMAGE_POOL)?.unwrap_or(
			if model_type.starts_with("qwen3_5") {
				2
			} else {
				3
			},
		);
		let aligned_patch = checked_i32_product("vision model patch size", &[patch, pooling])?;
		checked_i32_product("vision patchify width", &[patch, patch, 3])?;
		checked_i32_product(
			"unified vision patchify width",
			&[aligned_patch, aligned_patch, 3],
		)?;
		checked_i32_product("vision spatial-merge width", &[pooling, pooling, hidden])?;
		if model_type.starts_with("qwen3_5") && head_dim % 4 != 0 {
			return Err(Error::Config(
				"Qwen vision head width must be divisible by four".to_string(),
			));
		}
		let vision_kv =
			optional_bounded(vision, "num_key_value_heads", 1, MAX_HEADS)?.unwrap_or(heads);
		if vision_kv > heads || heads % vision_kv != 0 {
			return Err(Error::Config(
				"vision num_key_value_heads must divide its attention-head count".to_string(),
			));
		}
		checked_i32_product("vision attention projection width", &[heads, head_dim])?;
		for key in [
			"position_embedding_size",
			"out_hidden_size",
			"mm_posemb_size",
		] {
			optional_bounded(vision, key, 1, MAX_DIMENSION)?;
		}
		for key in ["default_output_length", "num_soft_tokens"] {
			optional_bounded(vision, key, MIN_SOFT_TOKENS, MAX_SOFT_TOKENS)?;
		}
		if let Some(positions) =
			optional_bounded(vision, "num_position_embeddings", 1, MAX_DIMENSION)?
			&& !is_perfect_square(positions)
		{
			return Err(Error::Config(
				"num_position_embeddings must be a perfect square".to_string(),
			));
		}
		validate_positive_float(vision, "rms_norm_eps")?;
		validate_positive_float(vision, "rope_theta")?;
		let vision_rope =
			optional_object(vision, "rope_parameters", "vision_config.rope_parameters")?;
		validate_rope_values(vision_rope, head_dim, "vision_config.rope_parameters", 1.0)?;
		get_bool(vision, "use_clipped_linears", false)?;
	}

	if let Some(audio) = root.get("audio_config") {
		if !audio.is_object() {
			return Err(Error::Config("audio_config must be an object".to_string()));
		}
		let hidden = optional_bounded(audio, "hidden_size", 1, MAX_DIMENSION)?.unwrap_or(1_024);
		let heads = optional_bounded(audio, "num_attention_heads", 1, MAX_HEADS)?.unwrap_or(8);
		if hidden % heads != 0 {
			return Err(Error::Config(
				"audio hidden_size must be divisible by num_attention_heads".to_string(),
			));
		}
		optional_bounded(audio, "num_hidden_layers", 1, MAX_LAYERS)?;
		optional_bounded(audio, "conv_kernel_size", 1, MAX_KERNEL)?;
		optional_bounded(audio, "attention_context_left", 2, i32::MAX)?;
		optional_bounded(audio, "audio_samples_per_token", 1, MAX_DIMENSION)?;
		validate_positive_float(audio, "rms_norm_eps")?;
		validate_positive_float(audio, "attention_logit_cap")?;
		validate_positive_float(audio, "residual_weight")?;
		optional_f32(audio, "attention_invalid_logits_value")?;
		get_bool(audio, "use_clipped_linears", false)?;
		if let Some(residual) = optional_f32(audio, "residual_weight")?
			&& residual > 1.0
		{
			return Err(Error::Config(
				"audio residual_weight must be no greater than one".to_string(),
			));
		}
	}
	Ok(())
}

fn validate_token_ids(root: &Value) -> Result<()> {
	for key in [
		"image_token_id",
		"vision_start_token_id",
		"vision_end_token_id",
		"video_token_id",
		"audio_token_id",
		"boa_token_id",
		"eoa_token_id",
		"boi_token_id",
		"eoi_token_id",
	] {
		optional_bounded(root, key, 0, i32::MAX)?;
	}
	Ok(())
}

fn gemma_layer_types(
	config: &Value,
	layers: i32,
	sliding_window_pattern: i32,
) -> Result<Vec<&'static str>> {
	let layer_count = usize::try_from(layers)
		.map_err(|_| Error::Config("invalid Gemma layer count".to_string()))?;
	match config.get("layer_types") {
		Some(Value::Array(values)) => values
			.iter()
			.enumerate()
			.map(|(index, value)| match value.as_str() {
				Some("full_attention") => Ok("full_attention"),
				Some("sliding_attention") => Ok("sliding_attention"),
				_ => Err(Error::Config(format!(
					"layer_types[{index}] is unsupported"
				))),
			})
			.collect(),
		Some(_) => Err(Error::Config("layer_types must be an array".to_string())),
		None => (0..layer_count)
			.map(|index| {
				let one_based = i32::try_from(index)
					.ok()
					.and_then(|value| value.checked_add(1))
					.unwrap_or(i32::MAX);
				if one_based % sliding_window_pattern == 0 {
					"full_attention"
				} else {
					"sliding_attention"
				}
			})
			.map(Ok)
			.collect(),
	}
}

fn optional_object<'a>(config: &'a Value, key: &str, path: &str) -> Result<Option<&'a Value>> {
	config
		.get(key)
		.map(|value| {
			value
				.as_object()
				.ok_or_else(|| Error::Config(format!("{path} must be an object")))?;
			Ok(value)
		})
		.transpose()
}

fn optional_nested_object<'a>(
	parent: Option<&'a Value>,
	key: &str,
	path: &str,
) -> Result<Option<&'a Value>> {
	match parent.and_then(|value| value.get(key)) {
		Some(value) => {
			value
				.as_object()
				.ok_or_else(|| Error::Config(format!("{path} must be an object")))?;
			Ok(Some(value))
		}
		None => Ok(None),
	}
}

fn validate_rope_values(
	rope: Option<&Value>,
	head_dim: i32,
	path: &str,
	default_partial: f32,
) -> Result<()> {
	let Some(rope) = rope else {
		return Ok(());
	};
	if let Some(theta) = optional_f32(rope, "rope_theta")?
		&& theta <= 0.0
	{
		return Err(Error::Config(format!("{path}.rope_theta must be positive")));
	}
	let partial = optional_f32(rope, "partial_rotary_factor")?.unwrap_or(default_partial);
	if !(0.0..=1.0).contains(&partial) || partial == 0.0 {
		return Err(Error::Config(format!(
			"{path}.partial_rotary_factor must be greater than zero and no greater than one"
		)));
	}
	let rotated_dims = ((head_dim as f32) * partial) as i32;
	if rotated_dims == 0 || rotated_dims % 2 != 0 {
		return Err(Error::Config(format!(
			"{path} produces a zero or odd RoPE dimension"
		)));
	}
	if let Some(factor) = optional_f32(rope, "factor")?
		&& factor <= 0.0
	{
		return Err(Error::Config(format!("{path}.factor must be positive")));
	}
	if let Some(value) = optional_i32(rope, "original_max_position_embeddings")?
		&& value <= 0
	{
		return Err(Error::Config(format!(
			"{path}.original_max_position_embeddings must be positive"
		)));
	}
	for key in ["beta_fast", "beta_slow", "attention_factor"] {
		if let Some(value) = optional_f32(rope, key)?
			&& value <= 0.0
		{
			return Err(Error::Config(format!("{path}.{key} must be positive")));
		}
	}
	Ok(())
}

fn is_perfect_square(value: i32) -> bool {
	let root = f64::from(value).sqrt() as i32;
	root.checked_mul(root) == Some(value)
}

fn validate_json_value(value: &Value, depth: usize, nodes: &mut usize) -> Result<()> {
	if depth > MAX_CONFIG_DEPTH {
		return Err(Error::Config(format!(
			"config nesting exceeds {MAX_CONFIG_DEPTH} levels"
		)));
	}
	*nodes = nodes
		.checked_add(1)
		.ok_or_else(|| Error::Config("config node count overflow".to_string()))?;
	if *nodes > MAX_CONFIG_NODES {
		return Err(Error::Config(format!(
			"config has more than {MAX_CONFIG_NODES} values"
		)));
	}
	match value {
		Value::Number(_) => {}
		Value::Array(values) => {
			if values.len() > MAX_CONFIG_COLLECTION_ITEMS {
				return Err(Error::Config(format!(
					"config array has more than {MAX_CONFIG_COLLECTION_ITEMS} entries"
				)));
			}
			for value in values {
				validate_json_value(value, depth + 1, nodes)?;
			}
		}
		Value::Object(values) => {
			if values.len() > MAX_CONFIG_COLLECTION_ITEMS {
				return Err(Error::Config(format!(
					"config object has more than {MAX_CONFIG_COLLECTION_ITEMS} fields"
				)));
			}
			for (key, value) in values {
				if key.len() > MAX_CONFIG_KEY_BYTES || key.contains('\0') {
					return Err(Error::Config("config contains an invalid key".to_string()));
				}
				validate_json_value(value, depth + 1, nodes)?;
			}
		}
		Value::Null | Value::Bool(_) | Value::String(_) => {}
	}
	Ok(())
}

fn required_bounded(config: &Value, key: &str, minimum: i32, maximum: i32) -> Result<i32> {
	let value = config
		.get(key)
		.ok_or_else(|| Error::Config(format!("missing required integer field '{key}'")))?;
	bounded_value(value, key, minimum, maximum)
}

fn optional_bounded(config: &Value, key: &str, minimum: i32, maximum: i32) -> Result<Option<i32>> {
	config
		.get(key)
		.map(|value| bounded_value(value, key, minimum, maximum))
		.transpose()
}

fn bounded_value(value: &Value, path: &str, minimum: i32, maximum: i32) -> Result<i32> {
	let integer = value
		.as_i64()
		.ok_or_else(|| Error::Config(format!("{path} must be an integer")))?;
	let integer = i32::try_from(integer)
		.map_err(|_| Error::Config(format!("{path} is outside the supported i32 range")))?;
	if !(minimum..=maximum).contains(&integer) {
		return Err(Error::Config(format!(
			"{path} must be between {minimum} and {maximum}"
		)));
	}
	Ok(integer)
}

pub(super) fn finite_f32(value: &Value, path: &str) -> Result<f32> {
	let value = value
		.as_f64()
		.ok_or_else(|| Error::Config(format!("{path} must be numeric")))?;
	let narrowed = value as f32;
	if !value.is_finite() || !narrowed.is_finite() {
		return Err(Error::Config(format!(
			"{path} is outside the finite f32 range"
		)));
	}
	Ok(narrowed)
}

fn validate_positive_float(config: &Value, key: &str) -> Result<()> {
	if let Some(value) = config.get(key) {
		if finite_f32(value, key)? <= 0.0 {
			return Err(Error::Config(format!("{key} must be positive")));
		}
	}
	Ok(())
}

fn validate_fraction(config: &Value, key: &str) -> Result<()> {
	if let Some(value) = config.get(key) {
		let value = finite_f32(value, key)?;
		if !(0.0..=1.0).contains(&value) || value == 0.0 {
			return Err(Error::Config(format!(
				"{key} must be greater than zero and no greater than one"
			)));
		}
	}
	Ok(())
}

fn checked_i32_product(label: &str, values: &[i32]) -> Result<i32> {
	values.iter().try_fold(1_i32, |product, value| {
		product
			.checked_mul(*value)
			.ok_or_else(|| Error::Config(format!("{label} exceeds i32")))
	})
}

fn validate_string_array(
	config: &Value,
	key: &str,
	expected_len: Option<i32>,
	allowed: &[&str],
) -> Result<()> {
	let Some(value) = config.get(key) else {
		return Ok(());
	};
	let values = value
		.as_array()
		.ok_or_else(|| Error::Config(format!("{key} must be an array")))?;
	if let Some(expected_len) = expected_len {
		let expected_len = usize::try_from(expected_len)
			.map_err(|_| Error::Config(format!("{key} has an invalid expected length")))?;
		if values.len() != expected_len {
			return Err(Error::Config(format!(
				"{key} length must equal num_hidden_layers"
			)));
		}
	}
	for (index, value) in values.iter().enumerate() {
		let value = value
			.as_str()
			.ok_or_else(|| Error::Config(format!("{key}[{index}] must be a string")))?;
		if !allowed.contains(&value) {
			return Err(Error::Config(format!(
				"{key}[{index}] has unsupported value {value:?}"
			)));
		}
	}
	Ok(())
}

#[cfg(test)]
mod tests {
	use super::*;

	fn tiny_qwen() -> Value {
		serde_json::json!({
			"model_type": "qwen3_5_text",
			"hidden_size": 32,
			"num_hidden_layers": 2,
			"intermediate_size": 64,
			"num_attention_heads": 2,
			"num_key_value_heads": 1,
			"head_dim": 16,
			"vocab_size": 16,
			"full_attention_interval": 2,
			"linear_num_value_heads": 4,
			"linear_num_key_heads": 2,
			"linear_key_head_dim": 8,
			"linear_value_head_dim": 8,
			"linear_conv_kernel_dim": 4
		})
	}

	fn tiny_gemma() -> Value {
		serde_json::json!({
			"model_type": "gemma4_text",
			"hidden_size": 32,
			"num_hidden_layers": 2,
			"num_attention_heads": 2,
			"num_key_value_heads": 1,
			"head_dim": 16,
			"global_head_dim": 16,
			"vocab_size": 16,
			"layer_types": ["full_attention", "sliding_attention"]
		})
	}

	#[test]
	fn valid_tiny_checkpoint_config_passes_preflight() {
		validate_checkpoint_config(&tiny_qwen()).unwrap();
	}

	#[test]
	fn preflight_rejects_zero_negative_and_huge_dimensions() {
		for (key, value) in [
			("num_attention_heads", serde_json::json!(0)),
			("num_hidden_layers", serde_json::json!(-1)),
			("hidden_size", serde_json::json!(1_i64 << 40)),
			("linear_conv_kernel_dim", serde_json::json!(0)),
		] {
			let mut config = tiny_qwen();
			config[key] = value;
			assert!(validate_checkpoint_config(&config).is_err(), "{key}");
		}
	}

	#[test]
	fn preflight_rejects_invalid_attention_and_linear_geometry() {
		let mut config = tiny_qwen();
		config["num_key_value_heads"] = serde_json::json!(3);
		assert!(validate_checkpoint_config(&config).is_err());

		let mut config = tiny_qwen();
		config["linear_num_value_heads"] = serde_json::json!(3);
		config["linear_num_key_heads"] = serde_json::json!(2);
		assert!(validate_checkpoint_config(&config).is_err());
	}

	#[test]
	fn preflight_rejects_float_overflow_before_f32_cast() {
		let mut config = tiny_qwen();
		config["rope_parameters"] = serde_json::json!({"rope_theta": 1e300});
		assert!(validate_checkpoint_config(&config).is_err());
	}

	#[test]
	fn preflight_rejects_malformed_nested_media_geometry() {
		let mut config = tiny_qwen();
		config["vision_config"] = serde_json::json!({
			"hidden_size": 10,
			"num_heads": 3,
			"patch_size": 16
		});
		assert!(validate_checkpoint_config(&config).is_err());

		let mut config = tiny_qwen();
		config["audio_config"] = serde_json::json!({
			"hidden_size": 32,
			"num_attention_heads": 4,
			"attention_context_left": 1
		});
		assert!(validate_checkpoint_config(&config).is_err());
	}

	#[test]
	fn preflight_rejects_wrong_optional_types_but_ignores_unused_wide_numbers() {
		let mut config = tiny_qwen();
		config["tie_word_embeddings"] = serde_json::json!("yes");
		assert!(validate_checkpoint_config(&config).is_err());

		let mut config = tiny_qwen();
		config["rope_parameters"] = serde_json::json!("invalid");
		assert!(validate_checkpoint_config(&config).is_err());

		let mut config = tiny_qwen();
		config["unused_training_metric"] = serde_json::json!(1e100);
		assert!(validate_checkpoint_config(&config).is_ok());
	}

	#[test]
	fn preflight_rejects_rope_and_token_id_narrowing_hazards() {
		let mut config = tiny_qwen();
		config["rope_parameters"] =
			serde_json::json!({"rope_theta": 0.0, "partial_rotary_factor": 0.25});
		assert!(validate_checkpoint_config(&config).is_err());

		let mut config = tiny_qwen();
		config["head_dim"] = serde_json::json!(14);
		config["rope_parameters"] = serde_json::json!({"partial_rotary_factor": 0.25});
		assert!(validate_checkpoint_config(&config).is_err());

		let mut config = tiny_qwen();
		config["image_token_id"] = serde_json::json!(-1);
		assert!(validate_checkpoint_config(&config).is_err());
	}

	#[test]
	fn preflight_rejects_combined_qwen_convolution_overflow() {
		let mut config = tiny_qwen();
		config["linear_num_key_heads"] = serde_json::json!(8192);
		config["linear_num_value_heads"] = serde_json::json!(8192);
		config["linear_key_head_dim"] = serde_json::json!(65536);
		config["linear_value_head_dim"] = serde_json::json!(131072);
		assert!(validate_checkpoint_config(&config).is_err());
	}

	#[test]
	fn preflight_matches_runtime_image_geometry_bounds() {
		for (key, value) in [
			("patch_size", serde_json::json!(257)),
			("spatial_merge_size", serde_json::json!(17)),
			("num_soft_tokens", serde_json::json!(16_385)),
		] {
			let mut config = tiny_qwen();
			config["vision_config"] = serde_json::json!({
				"hidden_size": 32,
				"num_heads": 2,
				"patch_size": 16,
				"spatial_merge_size": 2,
				"num_position_embeddings": 1024,
				"num_soft_tokens": 1280
			});
			config["vision_config"][key] = value;
			assert!(validate_checkpoint_config(&config).is_err(), "{key}");
		}

		let mut config = tiny_qwen();
		config["vision_config"] = serde_json::json!({
			"hidden_size": 30,
			"num_heads": 3,
			"patch_size": 16,
			"spatial_merge_size": 2,
			"num_position_embeddings": 1024
		});
		assert!(validate_checkpoint_config(&config).is_err());

		let mut config = tiny_qwen();
		config["vision_config"] = serde_json::json!({
			"hidden_size": 32,
			"num_heads": 2,
			"patch_size": 16,
			"spatial_merge_size": 2,
			"num_position_embeddings": 1000
		});
		assert!(validate_checkpoint_config(&config).is_err());
	}

	#[test]
	fn preflight_rejects_gemma_global_and_shared_kv_hazards() {
		assert!(validate_checkpoint_config(&tiny_gemma()).is_ok());

		let mut config = tiny_gemma();
		config["global_head_dim"] = serde_json::json!(15);
		assert!(validate_checkpoint_config(&config).is_err());

		let mut config = tiny_gemma();
		config["num_global_key_value_heads"] = serde_json::json!(3);
		assert!(validate_checkpoint_config(&config).is_err());

		let mut config = tiny_gemma();
		config["num_kv_shared_layers"] = serde_json::json!(1);
		assert!(validate_checkpoint_config(&config).is_err());

		let mut config = tiny_gemma();
		config["num_kv_shared_layers"] = serde_json::json!(2);
		assert!(validate_checkpoint_config(&config).is_err());
	}
}
