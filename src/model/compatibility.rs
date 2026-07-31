//! Static compatibility, machine-fit, and runtime-verification reports.

use std::{
	collections::BTreeSet,
	num::NonZeroUsize,
	path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{
	EvidenceSource, Modality, ModelGenerationDefaults, ModelRef, ModelSizing, ModelTraits,
	MtpSupport, Task, TraitConfidence, TraitEvidence, VerificationStatus,
	layout::{CheckpointLayoutError, checkpoint_plan},
};

const SUPPORTED_MODEL_TYPES: &[&str] = &[
	"dhara_ar",
	"gemma4",
	"gemma4_text",
	"gemma4_unified",
	"gemma4_unified_text",
	"laguna",
	"llama",
	"nemotron_h",
	"qwen2",
	"qwen3",
	"qwen3_5",
	"qwen3_5_moe",
	"qwen3_5_moe_text",
	"qwen3_5_text",
];

pub fn supported_model_type(value: &str) -> bool {
	SUPPORTED_MODEL_TYPES.contains(&value)
}
/// Workload assumptions used for memory fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkloadProfile {
	batch_size: NonZeroUsize,
	context_tokens: NonZeroUsize,
}

impl WorkloadProfile {
	/// Construct a non-zero workload.
	///
	/// # Errors
	///
	/// Returns [`WorkloadError`] when either value is zero.
	pub fn new(batch_size: usize, context_tokens: usize) -> Result<Self, WorkloadError> {
		Ok(Self {
			batch_size: NonZeroUsize::new(batch_size).ok_or(WorkloadError::ZeroBatch)?,
			context_tokens: NonZeroUsize::new(context_tokens).ok_or(WorkloadError::ZeroContext)?,
		})
	}

	/// Concurrent sequences.
	pub const fn batch_size(self) -> usize {
		self.batch_size.get()
	}

	/// Total context tokens.
	pub const fn context_tokens(self) -> usize {
		self.context_tokens.get()
	}
}

impl Default for WorkloadProfile {
	fn default() -> Self {
		Self {
			batch_size: NonZeroUsize::MIN,
			context_tokens: NonZeroUsize::new(16_384).unwrap_or(NonZeroUsize::MIN),
		}
	}
}

/// Invalid workload input.
#[derive(Debug, thiserror::Error, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WorkloadError {
	/// Batch size was zero.
	#[error("workload batch size must be positive")]
	ZeroBatch,
	/// Context length was zero.
	#[error("workload context tokens must be positive")]
	ZeroContext,
}

/// Memory-fit estimate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct FitReport {
	/// Metal's recommended working-set maximum.
	pub budget_bytes: u64,
	/// Exact selected weight bytes.
	pub weights_bytes: u64,
	/// Architecture-derived KV cache bytes.
	pub kv_cache_bytes: u64,
	/// Additional weighted prompt-cache residency allowance.
	pub prompt_cache_bytes: u64,
	/// MLX freed-buffer cache ceiling.
	pub runtime_cache_bytes: u64,
	/// Persistent runtime and recurrent-state estimate.
	pub persistent_bytes: u64,
	/// Safety margin.
	pub margin_bytes: u64,
	/// Total required bytes.
	pub required_bytes: u64,
	/// Workload assumption.
	pub workload: WorkloadProfile,
	/// Whether required bytes fit the budget.
	pub fits: bool,
}

/// Static compatibility report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CompatibilityReport {
	/// Inspected reference.
	pub reference: ModelRef,
	/// Estimated or runtime-verified.
	pub status: VerificationStatus,
	/// Engine/layout/fit all passed.
	pub compatible: bool,
	/// Engine architecture.
	pub model_type: Option<String>,
	/// Derived capability facts.
	pub traits: ModelTraits,
	/// Machine fit.
	pub fit: FitReport,
	/// Rejection explanations.
	pub reasons: Vec<String>,
}

impl CompatibilityReport {
	/// Record a successful backbone load and any byte-bound MTP
	/// certification established by the loaded client.
	pub(crate) fn mark_runtime_loaded(
		&mut self,
		supports_mtp: bool,
		supports_images: bool,
		supports_audio: bool,
	) {
		if !self.compatible {
			return;
		}
		self.status = VerificationStatus::Verified;
		if supports_mtp && self.traits.mtp == MtpSupport::Advertised {
			self.traits.mtp = MtpSupport::RuntimeVerified;
			self.traits.evidence.push(TraitEvidence {
				trait_key: "acceleration:mtp".to_string(),
				source: EvidenceSource::Runtime,
				detail: format!(
					"exact checkpoint bytes covered by {}",
					crate::engine::mtp_certification::IMPLEMENTATION_ID
				),
			});
			self.traits.confidence.insert(
				"acceleration:mtp".to_string(),
				TraitConfidence::RuntimeVerified,
			);
		}
		self.traits.evidence.push(TraitEvidence {
			trait_key: "compatibility:runtime_load".to_string(),
			source: EvidenceSource::Runtime,
			detail: "checkpoint loaded and completed deterministic one-token generation"
				.to_string(),
		});
		self.traits.confidence.insert(
			"acceleration:mlx".to_string(),
			TraitConfidence::RuntimeVerified,
		);
		for (supported, modality, key) in [
			(supports_images, Modality::Image, "input:image"),
			(supports_audio, Modality::Audio, "input:audio"),
		] {
			if supported {
				self.traits.input.insert(modality);
				self.traits
					.confidence
					.insert(key.to_string(), TraitConfidence::RuntimeVerified);
				self.traits.evidence.push(TraitEvidence {
					trait_key: key.to_string(),
					source: EvidenceSource::Runtime,
					detail: "loaded model configuration, template placeholder, and tokenizer \
					 binding matched exactly"
						.to_string(),
				});
			}
		}
		self.traits.confidence.insert(
			"task:text_generation".to_string(),
			TraitConfidence::RuntimeVerified,
		);
	}
}

/// Static inspection failure before a report can be formed.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum InspectionError {
	/// Required metadata could not be read.
	#[error("cannot read {path:?}: {source}")]
	Read {
		/// Metadata path.
		path: PathBuf,
		/// Underlying I/O error.
		#[source]
		source: std::io::Error,
	},
	/// Required JSON is malformed.
	#[error("invalid JSON {path:?}: {message}")]
	Json {
		/// Metadata path.
		path: PathBuf,
		/// Parser explanation.
		message: String,
	},
	/// JSON parsed, but model configuration is semantically unsupported.
	#[error("unsupported model configuration {path:?}: {message}")]
	Config {
		/// Metadata path.
		path: PathBuf,
		/// Semantic validation explanation.
		message: String,
	},
	/// Checkpoint file plan or safetensors structure is unsafe.
	#[error("invalid checkpoint layout {path:?}: {message}")]
	Layout {
		/// Affected path.
		path: PathBuf,
		/// Validation failure.
		message: String,
	},
}

/// Inspect an on-disk Hub snapshot or local import without initializing MLX.
///
/// # Errors
///
/// Returns an error when required metadata or safetensors headers cannot be
/// read, parsed, or safely matched to the shard index.
pub fn inspect_directory(
	reference: ModelRef,
	path: &Path,
	workload: WorkloadProfile,
	budget_bytes: u64,
) -> Result<CompatibilityReport, InspectionError> {
	inspect_directory_with_prompt_cache_tokens(
		reference,
		path,
		workload,
		budget_bytes,
		workload.context_tokens(),
	)
}

/// Inspect a checkpoint using the effective aggregate prompt-cache token
/// ceiling for a load. Zero omits prompt-cache residency from the estimate.
#[allow(
	clippy::too_many_lines,
	reason = "inspection is one fail-closed pipeline whose ordered diagnostics form the report"
)]
pub fn inspect_directory_with_prompt_cache_tokens(
	reference: ModelRef,
	path: &Path,
	workload: WorkloadProfile,
	budget_bytes: u64,
	prompt_cache_tokens: usize,
) -> Result<CompatibilityReport, InspectionError> {
	let config_path = path.join("config.json");
	let config = read_model_config(&config_path)?;
	let model_type = config
		.get("model_type")
		.and_then(Value::as_str)
		.map(str::to_string);
	let mut reasons = Vec::new();
	let supported_architecture = match model_type.as_deref() {
		None => {
			reasons.push("config.json has no string model_type".to_string());
			false
		}
		Some("qwen3_5_mtp") => {
			reasons.push("standalone qwen3_5_mtp sidecars have no loadable backbone".to_string());
			false
		}
		Some(value) if !SUPPORTED_MODEL_TYPES.contains(&value) => {
			reasons.push(format!("unsupported model_type {value:?}"));
			false
		}
		Some(_) => true,
	};
	if supported_architecture {
		crate::engine::models::config::validate_checkpoint_config(&config).map_err(|error| {
			InspectionError::Config {
				path: config_path.clone(),
				message: error.to_string(),
			}
		})?;
	}
	let tokenizer_path = path.join("tokenizer.json");
	let tokenizer_valid =
		match crate::artifact::read_bytes(&tokenizer_path, crate::artifact::MAX_TOKENIZER_BYTES) {
			Ok(bytes) => match tokenizers::Tokenizer::from_bytes(&bytes) {
				Ok(_) => true,
				Err(error) => {
					reasons.push(format!("invalid tokenizer.json: {error}"));
					false
				}
			},
			Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
				reasons.push("missing tokenizer.json".to_string());
				false
			}
			Err(source) => {
				return Err(InspectionError::Read {
					path: tokenizer_path,
					source,
				});
			}
		};
	let inventory = checkpoint_plan(path).map_err(|error| checkpoint_error(&error))?;
	if inventory.files().is_empty() {
		reasons.push("no runnable safetensors weights".to_string());
	}
	let quantization_valid = match crate::engine::quant::Quantization::from_config(&config) {
		Ok(_) => true,
		Err(error) => {
			reasons.push(format!("unsupported quantization: {error}"));
			false
		}
	};
	let text = config.get("text_config").unwrap_or(&config);
	let fit = match estimate_fit_from_config_with_prompt_cache_tokens(
		text,
		model_type.as_deref(),
		inventory.weights_bytes(),
		workload,
		budget_bytes,
		prompt_cache_tokens,
	) {
		Ok(fit) => fit,
		Err(reason) => {
			reasons.push(reason);
			FitReport {
				budget_bytes,
				weights_bytes: inventory.weights_bytes(),
				kv_cache_bytes: 0,
				prompt_cache_bytes: 0,
				runtime_cache_bytes: crate::engine::generate::MLX_FREED_BUFFER_CACHE_BYTES,
				persistent_bytes: 0,
				margin_bytes: 0,
				required_bytes: u64::MAX,
				workload,
				fits: false,
			}
		}
	};
	if !fit.fits {
		reasons.push(format!(
			"estimated residency {} exceeds Metal budget {budget_bytes}",
			fit.required_bytes
		));
	}
	let layout_valid = supported_architecture
		&& tokenizer_valid
		&& !inventory.files().is_empty()
		&& quantization_valid;
	let traits = derive_traits(
		path,
		&config,
		&DerivedTraitInputs {
			weights_bytes: inventory.weights_bytes(),
			residency: fit.required_bytes,
			workload,
			features: [
				layout_valid.then_some(StaticFeature::LayoutValid),
				inventory
					.mtp_weights_present()
					.then_some(StaticFeature::MtpWeights),
				inventory
					.vision_weights_present()
					.then_some(StaticFeature::VisionWeights),
				inventory
					.audio_weights_present()
					.then_some(StaticFeature::AudioWeights),
			]
			.into_iter()
			.flatten()
			.collect(),
		},
	)?;
	if !traits.tasks.contains(&Task::Chat) {
		reasons.push(
			"no supported chat template in chat_template.jinja, chat_templates/default.jinja, \
			 tokenizer_config.json, or processor_config.json"
				.to_string(),
		);
	}
	Ok(CompatibilityReport {
		reference,
		status: VerificationStatus::Estimated,
		compatible: reasons.is_empty(),
		model_type,
		traits,
		fit,
		reasons,
	})
}

/// Select the largest positive context under one model and Metal ceiling.
///
/// # Errors
///
/// Returns an inspection error when bounded model configuration cannot be read
/// or its architecture cannot be sized.
pub fn maximum_fitting_context(
	path: &Path,
	weights_bytes: u64,
	maximum_context_tokens: usize,
	budget_bytes: u64,
	prompt_cache_token_ceiling: usize,
) -> Result<Option<usize>, InspectionError> {
	let config_path = path.join("config.json");
	let config = read_model_config(&config_path)?;
	let model_type = config.get("model_type").and_then(Value::as_str);
	let text = config.get("text_config").unwrap_or(&config);
	maximum_fitting_context_from_config(
		text,
		model_type,
		weights_bytes,
		maximum_context_tokens,
		budget_bytes,
		prompt_cache_token_ceiling,
	)
	.map_err(|message| InspectionError::Config {
		path: config_path,
		message,
	})
}

fn maximum_fitting_context_from_config(
	config: &Value,
	model_type: Option<&str>,
	weights_bytes: u64,
	maximum_context_tokens: usize,
	budget_bytes: u64,
	prompt_cache_token_ceiling: usize,
) -> Result<Option<usize>, String> {
	if maximum_context_tokens == 0 {
		return Ok(None);
	}
	let fits = |context_tokens| {
		let workload =
			WorkloadProfile::new(1, context_tokens).map_err(|error| error.to_string())?;
		estimate_fit_from_config_with_prompt_cache_tokens(
			config,
			model_type,
			weights_bytes,
			workload,
			budget_bytes,
			context_tokens.min(prompt_cache_token_ceiling),
		)
		.map(|fit| fit.fits)
	};
	if !fits(1)? {
		return Ok(None);
	}
	if fits(maximum_context_tokens)? {
		return Ok(Some(maximum_context_tokens));
	}
	let mut fitting = 1_usize;
	let mut not_fitting = maximum_context_tokens;
	while fitting.saturating_add(1) < not_fitting {
		let candidate = fitting + (not_fitting - fitting) / 2;
		if fits(candidate)? {
			fitting = candidate;
		} else {
			not_fitting = candidate;
		}
	}
	Ok(Some(fitting))
}

fn read_model_config(path: &Path) -> Result<Value, InspectionError> {
	let bytes = crate::artifact::read_bytes(path, crate::artifact::MAX_MODEL_CONFIG_BYTES)
		.map_err(|source| InspectionError::Read {
			path: path.to_path_buf(),
			source,
		})?;
	serde_json::from_slice(&bytes).map_err(|error| InspectionError::Json {
		path: path.to_path_buf(),
		message: error.to_string(),
	})
}

pub fn estimate_fit_from_config(
	config: &Value,
	model_type: Option<&str>,
	weights_bytes: u64,
	workload: WorkloadProfile,
	budget_bytes: u64,
) -> Result<FitReport, String> {
	estimate_fit_from_config_with_prompt_cache_tokens(
		config,
		model_type,
		weights_bytes,
		workload,
		budget_bytes,
		workload.context_tokens(),
	)
}

fn estimate_fit_from_config_with_prompt_cache_tokens(
	config: &Value,
	model_type: Option<&str>,
	weights_bytes: u64,
	workload: WorkloadProfile,
	budget_bytes: u64,
	prompt_cache_tokens: usize,
) -> Result<FitReport, String> {
	let state = estimate_runtime_state(model_type, config, workload)?;
	let context = u64::try_from(workload.context_tokens())
		.map_err(|_| "workload context is too large".to_string())?;
	let batch = u64::try_from(workload.batch_size())
		.map_err(|_| "workload batch is too large".to_string())?;
	let persistent_bytes = checked_add(
		64_u64 << 20,
		checked_add(
			state.recurrent_bytes,
			checked_product(&[
				context,
				integer(config, "hidden_size").unwrap_or(0),
				batch,
				2,
			])
			.ok_or_else(|| "activation-memory estimate overflow".to_string())?,
		)
		.ok_or_else(|| "persistent-memory estimate overflow".to_string())?,
	)
	.ok_or_else(|| "persistent-memory estimate overflow".to_string())?;
	let prompt_cache_bytes = if prompt_cache_tokens > 0 {
		let cached_workload = WorkloadProfile::new(workload.batch_size(), prompt_cache_tokens)
			.map_err(|error| error.to_string())?;
		let cached_state = estimate_runtime_state(model_type, config, cached_workload)?;
		let cached_recurrent = checked_product(&[
			cached_state.recurrent_bytes,
			u64::try_from(crate::engine::prompt_cache::DEFAULT_MAX_ENTRIES)
				.map_err(|_| "prompt-cache entry count overflow".to_string())?,
		])
		.ok_or_else(|| "prompt-cache state estimate overflow".to_string())?;
		checked_add(cached_state.kv_cache_bytes, cached_recurrent)
			.ok_or_else(|| "prompt-cache estimate overflow".to_string())?
	} else {
		0
	};
	let runtime_cache_bytes = crate::engine::generate::MLX_FREED_BUFFER_CACHE_BYTES;
	let margin_bytes = (512_u64 << 20).max(weights_bytes / 10);
	let required_bytes = checked_add(
		weights_bytes,
		checked_add(
			state.kv_cache_bytes,
			checked_add(
				prompt_cache_bytes,
				checked_add(
					runtime_cache_bytes,
					checked_add(persistent_bytes, margin_bytes)
						.ok_or_else(|| "fit estimate overflow".to_string())?,
				)
				.ok_or_else(|| "fit estimate overflow".to_string())?,
			)
			.ok_or_else(|| "fit estimate overflow".to_string())?,
		)
		.ok_or_else(|| "fit estimate overflow".to_string())?,
	)
	.ok_or_else(|| "fit estimate overflow".to_string())?;
	Ok(FitReport {
		budget_bytes,
		weights_bytes,
		kv_cache_bytes: state.kv_cache_bytes,
		prompt_cache_bytes,
		runtime_cache_bytes,
		persistent_bytes,
		margin_bytes,
		required_bytes,
		workload,
		fits: required_bytes <= budget_bytes,
	})
}

#[derive(Debug, Default)]
struct RuntimeState {
	kv_cache_bytes: u64,
	recurrent_bytes: u64,
}

#[allow(
	clippy::too_many_lines,
	reason = "architecture-specific state formulas are kept adjacent for auditability"
)]
fn estimate_runtime_state(
	model_type: Option<&str>,
	config: &Value,
	workload: WorkloadProfile,
) -> Result<RuntimeState, String> {
	let layers = required_integer(config, "num_hidden_layers")?;
	let hidden = required_integer(config, "hidden_size")?;
	let heads = required_integer(config, "num_attention_heads")?;
	let kv_heads = integer(config, "num_key_value_heads").unwrap_or(heads);
	let head_dim = integer(config, "head_dim").unwrap_or(hidden / heads);
	if kv_heads == 0 || head_dim == 0 {
		return Err("incomplete attention geometry for fit estimation".to_string());
	}
	let batch = u64::try_from(workload.batch_size())
		.map_err(|_| "workload batch is too large".to_string())?;
	let context = u64::try_from(workload.context_tokens())
		.map_err(|_| "workload context is too large".to_string())?;
	match model_type {
		Some("qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text") => {
			let interval = integer(config, "full_attention_interval").unwrap_or(4);
			if interval == 0 {
				return Err("full_attention_interval must be positive".to_string());
			}
			let attention_layers = layers / interval;
			let linear_layers = layers - attention_layers;
			let kv_cache_bytes =
				attention_cache_bytes(attention_layers, kv_heads, head_dim, context, batch)?;
			let value_heads = integer(config, "linear_num_value_heads").unwrap_or(64);
			let key_heads = integer(config, "linear_num_key_heads").unwrap_or(16);
			let key_dim = integer(config, "linear_key_head_dim").unwrap_or(192);
			let value_dim = integer(config, "linear_value_head_dim").unwrap_or(128);
			let kernel = integer(config, "linear_conv_kernel_dim").unwrap_or(4);
			let conv_dim = checked_add(
				checked_product(&[key_heads, key_dim, 2])
					.ok_or_else(|| "Qwen linear-cache estimate overflow".to_string())?,
				checked_product(&[value_heads, value_dim])
					.ok_or_else(|| "Qwen linear-cache estimate overflow".to_string())?,
			)
			.ok_or_else(|| "Qwen linear-cache estimate overflow".to_string())?;
			let per_layer = checked_add(
				checked_product(&[batch, kernel.saturating_sub(1), conv_dim, 2])
					.ok_or_else(|| "Qwen convolution-state estimate overflow".to_string())?,
				checked_product(&[batch, value_heads, value_dim, key_dim, 4])
					.ok_or_else(|| "Qwen recurrent-state estimate overflow".to_string())?,
			)
			.ok_or_else(|| "Qwen recurrent-state estimate overflow".to_string())?;
			Ok(RuntimeState {
				kv_cache_bytes,
				recurrent_bytes: checked_product(&[linear_layers, per_layer])
					.ok_or_else(|| "Qwen recurrent-state estimate overflow".to_string())?,
			})
		}
		Some("nemotron_h") => {
			let pattern = config
				.get("hybrid_override_pattern")
				.and_then(Value::as_str)
				.ok_or_else(|| "nemotron_h lacks hybrid_override_pattern".to_string())?;
			let pattern_len = u64::try_from(pattern.chars().count())
				.map_err(|_| "Nemotron pattern is too long".to_string())?;
			if pattern_len != layers || pattern.chars().any(|kind| !matches!(kind, 'M' | '*' | '-'))
			{
				return Err("invalid nemotron_h hybrid_override_pattern".to_string());
			}
			let attention_layers = pattern.chars().filter(|kind| *kind == '*').count();
			let mamba_layers = pattern.chars().filter(|kind| *kind == 'M').count();
			let kv_cache_bytes = attention_cache_bytes(
				u64::try_from(attention_layers)
					.map_err(|_| "Nemotron attention count overflow".to_string())?,
				kv_heads,
				head_dim,
				context,
				batch,
			)?;
			let mamba_heads = required_integer(config, "mamba_num_heads")?;
			let mamba_head_dim = required_integer(config, "mamba_head_dim")?;
			let state_size = required_integer(config, "ssm_state_size")?;
			let groups = integer(config, "n_groups").unwrap_or(1);
			if groups == 0 {
				return Err("nemotron_h n_groups must be positive".to_string());
			}
			let kernel = integer(config, "conv_kernel").unwrap_or(4);
			let conv_dim = checked_add(
				checked_product(&[mamba_heads, mamba_head_dim])
					.ok_or_else(|| "Nemotron convolution-state estimate overflow".to_string())?,
				checked_product(&[2, groups, state_size])
					.ok_or_else(|| "Nemotron convolution-state estimate overflow".to_string())?,
			)
			.ok_or_else(|| "Nemotron convolution-state estimate overflow".to_string())?;
			let per_layer = checked_add(
				checked_product(&[batch, mamba_heads, mamba_head_dim, state_size, 4])
					.ok_or_else(|| "Nemotron recurrent-state estimate overflow".to_string())?,
				checked_product(&[batch, kernel.saturating_sub(1), conv_dim, 2])
					.ok_or_else(|| "Nemotron convolution-state estimate overflow".to_string())?,
			)
			.ok_or_else(|| "Nemotron state estimate overflow".to_string())?;
			Ok(RuntimeState {
				kv_cache_bytes,
				recurrent_bytes: checked_product(&[
					u64::try_from(mamba_layers)
						.map_err(|_| "Nemotron layer count overflow".to_string())?,
					per_layer,
				])
				.ok_or_else(|| "Nemotron state estimate overflow".to_string())?,
			})
		}
		Some("laguna") => {
			let sliding_window = integer(config, "sliding_window").unwrap_or(512);
			let layer_types = config.get("layer_types").and_then(Value::as_array);
			let mut token_layers = 0_u64;
			if let Some(layer_types) = layer_types {
				if u64::try_from(layer_types.len())
					.map_err(|_| "Laguna layer count overflow".to_string())?
					!= layers
				{
					return Err("laguna layer_types length mismatch".to_string());
				}
				for kind in layer_types {
					match kind.as_str() {
						Some("full_attention") => {
							token_layers = checked_add(token_layers, context)
								.ok_or_else(|| "Laguna fit estimate overflow".to_string())?;
						}
						Some("sliding_attention") => {
							token_layers =
								checked_add(token_layers, context.min(sliding_window))
									.ok_or_else(|| "Laguna fit estimate overflow".to_string())?;
						}
						_ => return Err("unsupported laguna layer type".to_string()),
					}
				}
			} else {
				token_layers = checked_product(&[layers, context])
					.ok_or_else(|| "Laguna fit estimate overflow".to_string())?;
			}
			let kv_cache_bytes = checked_product(&[2, token_layers, kv_heads, head_dim, batch, 2])
				.ok_or_else(|| "Laguna KV estimate overflow".to_string())?;
			Ok(RuntimeState {
				kv_cache_bytes,
				recurrent_bytes: 0,
			})
		}
		_ => Ok(RuntimeState {
			kv_cache_bytes: attention_cache_bytes(layers, kv_heads, head_dim, context, batch)?,
			recurrent_bytes: 0,
		}),
	}
}

fn attention_cache_bytes(
	layers: u64,
	kv_heads: u64,
	head_dim: u64,
	context: u64,
	batch: u64,
) -> Result<u64, String> {
	checked_product(&[2, layers, kv_heads, head_dim, context, batch, 2])
		.ok_or_else(|| "KV-cache estimate overflow".to_string())
}

struct DerivedTraitInputs {
	weights_bytes: u64,
	residency: u64,
	workload: WorkloadProfile,
	features: BTreeSet<StaticFeature>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum StaticFeature {
	LayoutValid,
	MtpWeights,
	VisionWeights,
	AudioWeights,
}

#[allow(
	clippy::too_many_lines,
	reason = "trait evidence and confidence are derived together to prevent unsupported claims"
)]
fn derive_traits(
	path: &Path,
	config: &Value,
	inputs: &DerivedTraitInputs,
) -> Result<ModelTraits, InspectionError> {
	let text = config.get("text_config").unwrap_or(config);
	let max_context_tokens = declared_max_context(text);
	let mut traits = ModelTraits {
		input: BTreeSet::from([Modality::Text]),
		output: BTreeSet::from([Modality::Text]),
		tasks: BTreeSet::from([Task::TextGeneration]),
		sizing: Some(ModelSizing {
			weights_bytes: Some(inputs.weights_bytes),
			estimated_residency_bytes: Some(inputs.residency),
			evaluated_context_tokens: Some(inputs.workload.context_tokens()),
			max_context_tokens,
		}),
		generation_defaults: generation_defaults(path)?,
		mlx: inputs.features.contains(&StaticFeature::LayoutValid),
		..ModelTraits::default()
	};
	for key in ["input:text", "output:text", "task:text_generation"] {
		traits
			.confidence
			.insert(key.to_string(), TraitConfidence::Inferred);
	}
	traits.evidence.push(TraitEvidence {
		trait_key: "task:text_generation".to_string(),
		source: EvidenceSource::Config,
		detail: "supported autoregressive architecture and runnable weight layout".to_string(),
	});
	if let Some(max_context_tokens) = max_context_tokens {
		traits.evidence.push(TraitEvidence {
			trait_key: "context:max_tokens".to_string(),
			source: EvidenceSource::Config,
			detail: format!("architecture declares at most {max_context_tokens} tokens"),
		});
		traits.confidence.insert(
			"context:max_tokens".to_string(),
			TraitConfidence::Advertised,
		);
	}
	if traits.generation_defaults != ModelGenerationDefaults::default() {
		traits.evidence.push(TraitEvidence {
			trait_key: "generation:defaults".to_string(),
			source: EvidenceSource::Config,
			detail: "generation_config.json recorded below Emelex and per-load policy precedence"
				.to_string(),
		});
		traits.confidence.insert(
			"generation:defaults".to_string(),
			TraitConfidence::Advertised,
		);
	}
	if inputs.features.contains(&StaticFeature::LayoutValid) {
		traits.evidence.push(TraitEvidence {
			trait_key: "acceleration:mlx".to_string(),
			source: EvidenceSource::Weights,
			detail: "supported architecture with validated safetensors file plan".to_string(),
		});
		traits
			.confidence
			.insert("acceleration:mlx".to_string(), TraitConfidence::Inferred);
		traits.confidence.insert(
			"task:text_generation".to_string(),
			TraitConfidence::Inferred,
		);
	}
	let (templates, bos_token, eos_token) = chat_templates(path)?;
	if let Some(templates) = templates {
		let (capabilities, _tool_format) =
			crate::engine::tokenizer::resolve_chat_templates_capabilities(
				&templates,
				(&bos_token, &eos_token),
			)
			.map_err(|error| {
				layout_error(
					path,
					&format!("chat template cannot be compiled safely: {error}"),
				)
			})?;
		traits.tasks.insert(Task::Chat);
		traits
			.confidence
			.insert("task:chat".to_string(), TraitConfidence::Inferred);
		traits.evidence.push(TraitEvidence {
			trait_key: "task:chat".to_string(),
			source: EvidenceSource::Tokenizer,
			detail: "chat template completed a bounded baseline render".to_string(),
		});
		if capabilities.system_prompt {
			traits.extras.insert(
				"interaction:system_prompt".to_string(),
				serde_json::Value::Bool(true),
			);
			traits.confidence.insert(
				"interaction:system_prompt".to_string(),
				TraitConfidence::Inferred,
			);
			traits.evidence.push(TraitEvidence {
				trait_key: "interaction:system_prompt".to_string(),
				source: EvidenceSource::Tokenizer,
				detail: "bounded semantic render preserved distinct system and user messages"
					.to_string(),
			});
		}
		if capabilities.tools {
			traits.tasks.insert(Task::ToolUse);
			traits
				.confidence
				.insert("interaction:tools".to_string(), TraitConfidence::Inferred);
			traits.evidence.push(TraitEvidence {
				trait_key: "interaction:tools".to_string(),
				source: EvidenceSource::Tokenizer,
				detail: "bounded semantic render preserved declaration structure plus ordered \
				 assistant-call arguments and matching results"
					.to_string(),
			});
		}
		if capabilities.reasoning_history {
			traits.extras.insert(
				"interaction:reasoning_history".to_string(),
				serde_json::Value::Bool(true),
			);
			traits.confidence.insert(
				"interaction:reasoning_history".to_string(),
				TraitConfidence::Inferred,
			);
			traits.evidence.push(TraitEvidence {
				trait_key: "interaction:reasoning_history".to_string(),
				source: EvidenceSource::Tokenizer,
				detail: "bounded semantic render preserved an explicit reasoning span across a \
				 follow-up turn"
					.to_string(),
			});
		}
		if capabilities.thinking_toggle {
			traits.extras.insert(
				"interaction:thinking_toggle".to_string(),
				serde_json::Value::Bool(true),
			);
			traits.confidence.insert(
				"interaction:thinking_toggle".to_string(),
				TraitConfidence::Inferred,
			);
			traits.evidence.push(TraitEvidence {
				trait_key: "interaction:thinking_toggle".to_string(),
				source: EvidenceSource::Tokenizer,
				detail: "bounded semantic renders distinguished enabled and disabled thinking"
					.to_string(),
			});
		}
		if capabilities.reasoning_history || capabilities.thinking_toggle {
			traits.tasks.insert(Task::Reasoning);
			traits.confidence.insert(
				"interaction:reasoning".to_string(),
				TraitConfidence::Inferred,
			);
			traits.evidence.push(TraitEvidence {
				trait_key: "interaction:reasoning".to_string(),
				source: EvidenceSource::Tokenizer,
				detail: "template supports reasoning history, an explicit thinking toggle, or both"
					.to_string(),
			});
		}
	}
	let advertised_mtp = integer(text, "num_nextn_predict_layers").unwrap_or(0) > 0
		|| integer(text, "mtp_num_hidden_layers").unwrap_or(0) > 0
		|| inputs.features.contains(&StaticFeature::MtpWeights);
	traits.mtp = if advertised_mtp {
		MtpSupport::Advertised
	} else {
		MtpSupport::Absent
	};
	if advertised_mtp {
		traits.evidence.push(TraitEvidence {
			trait_key: "acceleration:mtp_advertised".to_string(),
			source: if inputs.features.contains(&StaticFeature::MtpWeights) {
				EvidenceSource::Weights
			} else {
				EvidenceSource::Config
			},
			detail: "MTP metadata or tensor namespace present; runtime validation required"
				.to_string(),
		});
		traits.confidence.insert(
			"acceleration:mtp_advertised".to_string(),
			TraitConfidence::Advertised,
		);
	}
	Ok(traits)
}

fn generation_defaults(path: &Path) -> Result<ModelGenerationDefaults, InspectionError> {
	let generation_path = path.join("generation_config.json");
	let Some(text) = read_optional_bounded(&generation_path, 4 << 20)? else {
		return Ok(ModelGenerationDefaults::default());
	};
	let value: Value = serde_json::from_str(&text).map_err(|error| InspectionError::Json {
		path: generation_path.clone(),
		message: error.to_string(),
	})?;
	let object = value.as_object().ok_or_else(|| InspectionError::Config {
		path: generation_path.clone(),
		message: "generation_config.json must contain an object".to_string(),
	})?;
	let temperature = optional_f32(object.get("temperature"), "temperature", &generation_path)?;
	if temperature.is_some_and(|value| !(0.0..=2.0).contains(&value)) {
		return Err(InspectionError::Config {
			path: generation_path,
			message: "generation temperature must be in 0..=2".to_string(),
		});
	}
	let top_p = optional_f32(object.get("top_p"), "top_p", &generation_path)?;
	if top_p.is_some_and(|value| !(0.0..=1.0).contains(&value)) {
		return Err(InspectionError::Config {
			path: generation_path,
			message: "generation top_p must be in 0..=1".to_string(),
		});
	}
	let top_k = object
		.get("top_k")
		.map(|value| {
			value
				.as_u64()
				.and_then(|value| u32::try_from(value).ok())
				.ok_or_else(|| InspectionError::Config {
					path: generation_path.clone(),
					message: "generation top_k must be an unsigned 32-bit integer".to_string(),
				})
		})
		.transpose()?;
	let max_new_tokens = object
		.get("max_new_tokens")
		.map(|value| {
			value
				.as_u64()
				.and_then(|value| usize::try_from(value).ok())
				.filter(|value| *value > 0)
				.ok_or_else(|| InspectionError::Config {
					path: generation_path.clone(),
					message: "generation max_new_tokens must be positive".to_string(),
				})
		})
		.transpose()?;
	let do_sample = object
		.get("do_sample")
		.map(|value| {
			value.as_bool().ok_or_else(|| InspectionError::Config {
				path: generation_path.clone(),
				message: "generation do_sample must be boolean".to_string(),
			})
		})
		.transpose()?;
	Ok(ModelGenerationDefaults {
		do_sample,
		temperature,
		top_p,
		top_k,
		max_new_tokens,
	})
}

fn optional_f32(
	value: Option<&Value>,
	field: &str,
	path: &Path,
) -> Result<Option<f32>, InspectionError> {
	value
		.map(|value| {
			let value = value.as_f64().ok_or_else(|| InspectionError::Config {
				path: path.to_path_buf(),
				message: format!("generation {field} must be numeric"),
			})?;
			let converted =
				value
					.to_string()
					.parse::<f32>()
					.map_err(|error| InspectionError::Config {
						path: path.to_path_buf(),
						message: format!("generation {field} is not representable: {error}"),
					})?;
			if !converted.is_finite() {
				return Err(InspectionError::Config {
					path: path.to_path_buf(),
					message: format!("generation {field} must be finite"),
				});
			}
			Ok(converted)
		})
		.transpose()
}

fn chat_templates(
	path: &Path,
) -> Result<
	(
		Option<crate::engine::tokenizer::ChatTemplates>,
		String,
		String,
	),
	InspectionError,
> {
	let external = external_chat_templates(path)?;
	let tokenizer = path.join("tokenizer_config.json");
	let tokenizer_config = read_optional_bounded(&tokenizer, 16 << 20)?
		.map(|text| {
			serde_json::from_str::<Value>(&text).map_err(|error| InspectionError::Json {
				path: tokenizer,
				message: error.to_string(),
			})
		})
		.transpose()?
		.unwrap_or(Value::Null);
	let processor = path.join("processor_config.json");
	let processor_config = read_optional_bounded(&processor, 16 << 20)?
		.map(|text| {
			serde_json::from_str::<Value>(&text).map_err(|error| InspectionError::Json {
				path: processor,
				message: error.to_string(),
			})
		})
		.transpose()?
		.unwrap_or(Value::Null);
	let templates = crate::engine::tokenizer::resolve_chat_template_artifacts(
		processor_config
			.get("chat_template")
			.unwrap_or(&Value::Null),
		external.legacy.as_ref(),
		external.default,
		external.tool_use,
		tokenizer_config
			.get("chat_template")
			.unwrap_or(&Value::Null),
	)
	.map_err(|error| layout_error(path, &format!("invalid chat template artifacts: {error}")))?;
	let special = |key: &str| {
		tokenizer_config
			.get(key)
			.and_then(|value| match value {
				Value::String(value) => Some(value.as_str()),
				Value::Object(value) => value.get("content").and_then(Value::as_str),
				_ => None,
			})
			.unwrap_or_default()
			.to_string()
	};
	Ok((templates, special("bos_token"), special("eos_token")))
}

struct ExternalChatTemplates {
	legacy: Option<Value>,
	default: Option<String>,
	tool_use: Option<String>,
}

fn external_chat_templates(path: &Path) -> Result<ExternalChatTemplates, InspectionError> {
	let standalone = path.join("chat_template.jinja");
	let standalone_root = read_optional_bounded(
		&standalone,
		crate::engine::tokenizer::MAX_CHAT_TEMPLATE_BYTES as u64,
	)?;
	let root_tool_path = path.join("chat_template_tool_use.jinja");
	let root_tool = read_optional_bounded(
		&root_tool_path,
		crate::engine::tokenizer::MAX_CHAT_TEMPLATE_BYTES as u64,
	)?;
	let mut named_defaults = Vec::new();
	let mut named_tools = Vec::new();
	for directory in [
		crate::engine::tokenizer::CURRENT_CHAT_TEMPLATE_DIR,
		crate::engine::tokenizer::LEGACY_CHAT_TEMPLATE_DIR,
	] {
		let named_default_path = path.join(directory).join("default.jinja");
		if let Some(template) = read_optional_bounded(
			&named_default_path,
			crate::engine::tokenizer::MAX_CHAT_TEMPLATE_BYTES as u64,
		)? {
			named_defaults.push((named_default_path, template));
		}
		let named_tool_path = path.join(directory).join("tool_use.jinja");
		if let Some(template) = read_optional_bounded(
			&named_tool_path,
			crate::engine::tokenizer::MAX_CHAT_TEMPLATE_BYTES as u64,
		)? {
			named_tools.push((named_tool_path, template));
		}
	}
	if standalone_root.is_some() && !named_defaults.is_empty() {
		return Err(layout_error(
			path,
			"root and named default chat templates map to the same runtime file",
		));
	}
	if root_tool.is_some() && !named_tools.is_empty() {
		return Err(layout_error(
			path,
			"root and named tool-use chat templates map to the same runtime file",
		));
	}
	let named_default = match named_defaults.as_slice() {
		[] => None,
		[(_, template)] => Some(template.clone()),
		_ => {
			return Err(layout_error(
				path,
				"multiple named default chat templates are present",
			));
		}
	};
	let named_tool = match named_tools.as_slice() {
		[] => None,
		[(_, template)] => Some(template.clone()),
		_ => {
			return Err(layout_error(
				path,
				"multiple named tool-use chat templates are present",
			));
		}
	};
	let legacy_path = path.join(crate::engine::tokenizer::LEGACY_CHAT_TEMPLATE_FILE);
	let legacy = read_optional_bounded(&legacy_path, crate::artifact::MAX_TOKENIZER_CONFIG_BYTES)?
		.map(|text| {
			serde_json::from_str::<Value>(&text).map_err(|error| InspectionError::Json {
				path: legacy_path,
				message: error.to_string(),
			})
		})
		.transpose()?;
	if legacy.is_some() && (!named_defaults.is_empty() || !named_tools.is_empty()) {
		return Err(layout_error(
			path,
			"chat_template.json conflicts with named chat template files",
		));
	}
	Ok(ExternalChatTemplates {
		legacy,
		default: standalone_root.or(named_default),
		tool_use: root_tool.or(named_tool),
	})
}

fn read_optional_bounded(path: &Path, limit: u64) -> Result<Option<String>, InspectionError> {
	crate::artifact::read_optional_utf8(path, limit).map_err(|source| InspectionError::Read {
		path: path.to_path_buf(),
		source,
	})
}

fn required_integer(config: &Value, key: &str) -> Result<u64, String> {
	integer(config, key)
		.filter(|value| *value > 0)
		.ok_or_else(|| format!("missing positive {key} for fit estimation"))
}

fn integer(config: &Value, key: &str) -> Option<u64> {
	config.get(key).and_then(Value::as_u64)
}

fn declared_max_context(config: &Value) -> Option<usize> {
	[
		"max_position_embeddings",
		"max_sequence_length",
		"seq_length",
		"model_max_length",
	]
	.into_iter()
	.filter_map(|key| integer(config, key))
	.min()
	.and_then(|value| usize::try_from(value).ok())
	.filter(|value| *value > 0)
}

fn checked_product(values: &[u64]) -> Option<u64> {
	values
		.iter()
		.try_fold(1_u64, |product, value| product.checked_mul(*value))
}

const fn checked_add(left: u64, right: u64) -> Option<u64> {
	left.checked_add(right)
}

fn layout_error(path: &Path, message: &str) -> InspectionError {
	InspectionError::Layout {
		path: path.to_path_buf(),
		message: message.to_string(),
	}
}

fn checkpoint_error(error: &CheckpointLayoutError) -> InspectionError {
	InspectionError::Layout {
		path: error.path().to_path_buf(),
		message: error.message().to_string(),
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn attention_config() -> Value {
		serde_json::json!({
			"hidden_size": 64,
			"num_attention_heads": 4,
			"num_hidden_layers": 4,
			"num_key_value_heads": 2,
			"head_dim": 16
		})
	}

	fn fit(config: &Value, context_tokens: usize, prompt_cache_tokens: usize) -> FitReport {
		estimate_fit_from_config_with_prompt_cache_tokens(
			config,
			Some("llama"),
			1 << 20,
			WorkloadProfile::new(1, context_tokens).unwrap(),
			u64::MAX,
			prompt_cache_tokens,
		)
		.unwrap()
	}

	#[test]
	fn maximum_fitting_context_selects_declared_ceiling_when_it_fits() {
		let config = attention_config();
		let selected = maximum_fitting_context_from_config(
			&config,
			Some("llama"),
			1 << 20,
			2_097_152,
			u64::MAX,
			crate::engine::prompt_cache::DEFAULT_MAX_TOTAL_TOKENS,
		)
		.unwrap();

		assert_eq!(selected, Some(2_097_152));
	}

	#[test]
	fn maximum_fitting_context_selects_exact_machine_boundary() {
		let config = attention_config();
		let budget = fit(&config, 63, 63).required_bytes;
		let selected = maximum_fitting_context_from_config(
			&config,
			Some("llama"),
			1 << 20,
			128,
			budget,
			crate::engine::prompt_cache::DEFAULT_MAX_TOTAL_TOKENS,
		)
		.unwrap();

		assert_eq!(selected, Some(63));
	}

	#[test]
	fn maximum_fitting_context_reports_no_safe_positive_context() {
		let config = attention_config();
		let budget = fit(&config, 1, 1).required_bytes.saturating_sub(1);
		let selected = maximum_fitting_context_from_config(
			&config,
			Some("llama"),
			1 << 20,
			128,
			budget,
			crate::engine::prompt_cache::DEFAULT_MAX_TOTAL_TOKENS,
		)
		.unwrap();

		assert_eq!(selected, None);
	}

	#[test]
	fn explicit_prompt_cache_ceiling_bounds_residency_above_16k() {
		let config = attention_config();
		let cache_limit = crate::engine::prompt_cache::DEFAULT_MAX_TOTAL_TOKENS;
		let at_limit = fit(&config, cache_limit, cache_limit);
		let bounded_above_limit = fit(&config, cache_limit * 2, cache_limit);
		let full_context_cache = fit(&config, cache_limit * 2, cache_limit * 2);

		assert_eq!(
			at_limit.prompt_cache_bytes,
			bounded_above_limit.prompt_cache_bytes
		);
		assert!(full_context_cache.prompt_cache_bytes > bounded_above_limit.prompt_cache_bytes);
	}

	#[test]
	fn maximum_fitting_context_reserves_cache_for_request_reenable() {
		let config = attention_config();
		let maximum = 32_768;
		let no_cache_budget = fit(&config, maximum, 0).required_bytes;
		let selected = maximum_fitting_context_from_config(
			&config,
			Some("llama"),
			1 << 20,
			maximum,
			no_cache_budget,
			crate::engine::prompt_cache::DEFAULT_MAX_TOTAL_TOKENS,
		)
		.unwrap();

		assert!(selected.is_some_and(|context_tokens| context_tokens < maximum));
	}

	fn derived_tasks(template: &str) -> BTreeSet<Task> {
		let directory = tempfile::tempdir().unwrap();
		std::fs::write(directory.path().join("chat_template.jinja"), template).unwrap();
		derive_traits(
			directory.path(),
			&serde_json::json!({"model_type": "qwen3"}),
			&DerivedTraitInputs {
				weights_bytes: 1,
				residency: 1,
				workload: WorkloadProfile::default(),
				features: BTreeSet::from([StaticFeature::LayoutValid]),
			},
		)
		.unwrap()
		.tasks
	}

	#[test]
	fn installed_capability_probe_ignores_inert_template_keywords() {
		let tasks = derived_tasks(
			r"
{# tools tool_calls function reasoning content enable_thinking #}
{% if false %}
	{{ tools|tojson }}
	{{ messages[0].tool_calls|tojson }}
	{{ messages[0].reasoning_content }}
{% endif %}
{% for message in messages %}{{ message.content }}{% endfor %}
",
		);
		assert_eq!(
			(
				tasks.contains(&Task::ToolUse),
				tasks.contains(&Task::Reasoning),
			),
			(false, false)
		);
	}

	#[test]
	fn installed_capability_probe_accepts_semantic_template_fixture() {
		let tasks = derived_tasks(
			r#"
{% if tools %}<tools>{{ tools|tojson }}</tools>{% endif %}
{% for message in messages %}
	{% if message.tool_calls %}
			{% for call in message.tool_calls %}
			<tool_call>{"name":{{ call.function.name|tojson }},"arguments":{{ call.function.arguments|tojson }}}</tool_call>
		{% endfor %}
	{% elif message.role == "tool" %}
		<result>{{ message.content }}</result>
	{% else %}
		{% if message.reasoning_content %}
			<think>{{ message.reasoning_content }}</think>
		{% endif %}
		{{ message.content }}
	{% endif %}
{% endfor %}
"#,
		);
		assert_eq!(
			(
				tasks.contains(&Task::ToolUse),
				tasks.contains(&Task::Reasoning),
			),
			(true, true)
		);
	}

	#[test]
	fn installed_capability_probe_reads_normalized_tool_template() {
		let directory = tempfile::tempdir().unwrap();
		std::fs::write(
			directory.path().join("chat_template.jinja"),
			"{% for message in messages %}{{ message.content }}{% endfor %}",
		)
		.unwrap();
		std::fs::write(
			directory.path().join("chat_template_tool_use.jinja"),
			r#"
{% if tools %}<tools>{{ tools|tojson }}</tools>{% endif %}
{% for message in messages %}
	{% if message.tool_calls %}
		{% for call in message.tool_calls %}
			<tool_call>{"name":{{ call.function.name|tojson }},"arguments":{{ call.function.arguments|tojson }}}</tool_call>
		{% endfor %}
	{% elif message.role == "tool" %}
		<tool_result>{{ message.content }}</tool_result>
	{% else %}
		{{ message.content }}
	{% endif %}
{% endfor %}
"#,
		)
		.unwrap();
		let traits = derive_traits(
			directory.path(),
			&serde_json::json!({"model_type": "qwen3"}),
			&DerivedTraitInputs {
				weights_bytes: 1,
				residency: 1,
				workload: WorkloadProfile::default(),
				features: BTreeSet::from([StaticFeature::LayoutValid]),
			},
		)
		.unwrap();
		assert!(traits.tasks.contains(&Task::ToolUse));
	}
}
