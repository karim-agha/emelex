//! Model capability facts and evidence.

use std::{
	collections::{BTreeMap, BTreeSet},
	fmt,
	str::FromStr,
};

use serde::{Deserialize, Serialize};

/// Input or output modality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Modality {
	/// Text.
	Text,
	/// Images.
	Image,
	/// Audio.
	Audio,
}

/// Model task family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Task {
	/// Autoregressive text generation.
	TextGeneration,
	/// Conversational generation through a chat template.
	Chat,
	/// Tool/function invocation.
	ToolUse,
	/// Structured JSON output by instruction.
	StructuredOutput,
	/// Explicit reasoning/thinking spans.
	Reasoning,
	/// Structured translation through a translation-shaped chat template
	/// (TranslateGemma-style per-message language pairs).
	Translation,
}

/// Confidence state for a capability or compatibility fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum VerificationStatus {
	/// Derived from static repository metadata.
	Estimated,
	/// Confirmed by a successful local runtime probe.
	Verified,
}

/// MTP capability progression.
#[derive(
	Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum MtpSupport {
	/// No MTP claim or layout.
	#[default]
	Absent,
	/// Repository metadata advertises MTP.
	Advertised,
	/// Runtime load and parity gate verified this exact layout.
	RuntimeVerified,
}

/// Origin of one discovered fact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum EvidenceSource {
	/// Hugging Face search metadata or tags.
	HubMetadata,
	/// Repository file tree.
	RepositoryTree,
	/// `config.json`.
	Config,
	/// Tokenizer configuration or chat template.
	Tokenizer,
	/// Safetensors index or headers.
	Weights,
	/// Successful local runtime probe.
	Runtime,
}

/// Evidence supporting a trait.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct TraitEvidence {
	/// Namespaced trait key.
	pub trait_key: String,
	/// Evidence origin.
	pub source: EvidenceSource,
	/// Human-readable observation.
	pub detail: String,
}

impl TraitEvidence {
	/// Construct one capability-evidence record.
	pub fn new(
		trait_key: impl Into<String>,
		source: EvidenceSource,
		detail: impl Into<String>,
	) -> Self {
		Self {
			trait_key: trait_key.into(),
			source,
			detail: detail.into(),
		}
	}
}

/// Strength of one capability claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum TraitConfidence {
	/// Repository metadata makes an explicit claim.
	Advertised,
	/// Emelex inferred the capability from static artifacts.
	Inferred,
	/// A local runtime probe confirmed the capability.
	RuntimeVerified,
}

/// Optional model sizing facts.
///
/// `None` means the relevant artifact was not inspected. It never means zero.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[non_exhaustive]
pub struct ModelSizing {
	/// Exact selected runnable weight bytes.
	pub weights_bytes: Option<u64>,
	/// Estimated peak residency for [`Self::evaluated_context_tokens`].
	pub estimated_residency_bytes: Option<u64>,
	/// Context used for the residency estimate.
	pub evaluated_context_tokens: Option<usize>,
	/// Architecture-declared maximum context.
	pub max_context_tokens: Option<usize>,
}

/// Defaults advertised by a checkpoint's `generation_config.json`.
///
/// These are evidence, not an override of explicit Emelex configuration or
/// per-load policy.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[non_exhaustive]
pub struct ModelGenerationDefaults {
	/// Sampling mode advertised by the model.
	pub do_sample: Option<bool>,
	/// Sampling temperature.
	pub temperature: Option<f32>,
	/// Nucleus-sampling threshold.
	pub top_p: Option<f32>,
	/// Top-k cutoff.
	pub top_k: Option<u32>,
	/// Suggested generation ceiling.
	pub max_new_tokens: Option<usize>,
}

/// Static and verified model capabilities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct ModelTraits {
	/// Accepted input modalities.
	pub input: BTreeSet<Modality>,
	/// Produced output modalities.
	pub output: BTreeSet<Modality>,
	/// Supported task families.
	pub tasks: BTreeSet<Task>,
	/// MLX-optimized repository/layout.
	pub mlx: bool,
	/// MTP progression.
	pub mtp: MtpSupport,
	/// Sizing facts whose absence remains distinguishable from zero.
	#[serde(default)]
	pub sizing: Option<ModelSizing>,
	/// Checkpoint-advertised generation defaults.
	#[serde(default)]
	pub generation_defaults: ModelGenerationDefaults,
	/// Namespaced forward-compatible extension facts.
	pub extras: BTreeMap<String, serde_json::Value>,
	/// Confidence for each namespaced trait key.
	pub confidence: BTreeMap<String, TraitConfidence>,
	/// Evidence trail for derived facts.
	pub evidence: Vec<TraitEvidence>,
}

impl Default for ModelTraits {
	fn default() -> Self {
		Self {
			input: BTreeSet::new(),
			output: BTreeSet::new(),
			tasks: BTreeSet::new(),
			mlx: false,
			mtp: MtpSupport::Absent,
			sizing: None,
			generation_defaults: ModelGenerationDefaults::default(),
			extras: BTreeMap::new(),
			confidence: BTreeMap::new(),
			evidence: Vec::new(),
		}
	}
}

impl ModelTraits {
	/// Test one validated capability filter.
	pub fn satisfies(&self, filter: &TraitFilter) -> bool {
		self.satisfies_predicate(filter.predicate())
	}

	fn satisfies_predicate(&self, predicate: &TraitPredicate) -> bool {
		match predicate {
			TraitPredicate::Capability(key) => self.satisfies_key(key),
			TraitPredicate::MinimumConfidence { key, confidence } => {
				self.satisfies_key(key)
					&& self
						.confidence
						.get(key)
						.is_some_and(|actual| actual >= confidence)
			}
			TraitPredicate::AtMost { metric, value } => {
				self.metric(*metric).is_some_and(|actual| actual <= *value)
			}
			TraitPredicate::AtLeast { metric, value } => {
				self.metric(*metric).is_some_and(|actual| actual >= *value)
			}
			TraitPredicate::MinimumMtp(stage) => self.mtp >= *stage,
		}
	}

	fn satisfies_key(&self, key: &str) -> bool {
		match key {
			"input:text" => self.input.contains(&Modality::Text),
			"input:image" => self.input.contains(&Modality::Image),
			"input:audio" => self.input.contains(&Modality::Audio),
			"output:text" => self.output.contains(&Modality::Text),
			"output:image" => self.output.contains(&Modality::Image),
			"output:audio" => self.output.contains(&Modality::Audio),
			"task:text_generation" => self.tasks.contains(&Task::TextGeneration),
			"task:chat" => self.tasks.contains(&Task::Chat),
			"task:translation" => self.tasks.contains(&Task::Translation),
			"interaction:tools" => self.tasks.contains(&Task::ToolUse),
			"interaction:system_prompt" => self
				.extras
				.get("interaction:system_prompt")
				.and_then(serde_json::Value::as_bool)
				.unwrap_or(false),
			"interaction:reasoning" => self.tasks.contains(&Task::Reasoning),
			"interaction:reasoning_history" | "interaction:thinking_toggle" => self
				.extras
				.get(key)
				.and_then(serde_json::Value::as_bool)
				.unwrap_or(false),
			"interaction:structured_output" => self.tasks.contains(&Task::StructuredOutput),
			"acceleration:mlx" => self.mlx,
			"acceleration:mtp" => self.mtp == MtpSupport::RuntimeVerified,
			"acceleration:mtp_advertised" => matches!(
				self.mtp,
				MtpSupport::Advertised | MtpSupport::RuntimeVerified
			),
			other => self
				.extras
				.get(other)
				.and_then(serde_json::Value::as_bool)
				.unwrap_or(false),
		}
	}

	fn metric(&self, metric: TraitMetric) -> Option<u64> {
		let sizing = self.sizing.as_ref()?;
		match metric {
			TraitMetric::WeightsBytes => sizing.weights_bytes,
			TraitMetric::EstimatedResidencyBytes => sizing.estimated_residency_bytes,
			TraitMetric::EvaluatedContextTokens => sizing
				.evaluated_context_tokens
				.and_then(|value| u64::try_from(value).ok()),
			TraitMetric::MaximumContextTokens => sizing
				.max_context_tokens
				.and_then(|value| u64::try_from(value).ok()),
		}
	}

	/// Evidence confidence for a capability key.
	pub fn confidence(&self, filter: &TraitFilter) -> Option<TraitConfidence> {
		match filter.predicate() {
			TraitPredicate::Capability(key) | TraitPredicate::MinimumConfidence { key, .. } => {
				self.confidence.get(key).copied()
			}
			TraitPredicate::AtMost { .. }
			| TraitPredicate::AtLeast { .. }
			| TraitPredicate::MinimumMtp(_) => None,
		}
	}
}

/// Numeric capability metric used by range predicates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum TraitMetric {
	/// Exact selected runnable weights.
	WeightsBytes,
	/// Estimated peak residency.
	EstimatedResidencyBytes,
	/// Context used by the estimate.
	EvaluatedContextTokens,
	/// Architecture-declared maximum context.
	MaximumContextTokens,
}

/// Typed capability predicate.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum TraitPredicate {
	/// Boolean capability or extension fact.
	Capability(String),
	/// Capability with a minimum evidence confidence.
	MinimumConfidence {
		/// Namespaced capability key.
		key: String,
		/// Required confidence floor.
		confidence: TraitConfidence,
	},
	/// Numeric upper bound.
	AtMost {
		/// Metric being compared.
		metric: TraitMetric,
		/// Inclusive maximum.
		value: u64,
	},
	/// Numeric lower bound.
	AtLeast {
		/// Metric being compared.
		metric: TraitMetric,
		/// Inclusive minimum.
		value: u64,
	},
	/// Minimum MTP verification stage.
	MinimumMtp(MtpSupport),
}

/// Validated namespaced capability filter.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TraitFilter {
	source: String,
	predicate: TraitPredicate,
}

impl TraitFilter {
	/// Parse a known filter or an explicit `extension:<name>` filter.
	///
	/// # Errors
	///
	/// Returns [`TraitFilterError`] for typos and unsafe extension keys.
	pub fn parse(value: impl Into<String>) -> Result<Self, TraitFilterError> {
		let value = value.into();
		let extension = value.strip_prefix("extension:");
		let predicate = if known_capability(&value)
			|| extension.is_some_and(|name| {
				!name.is_empty()
					&& name.len() <= 128
					&& name.bytes().all(|byte| {
						byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.' | b':')
					})
			}) {
			TraitPredicate::Capability(value.clone())
		} else if let Some(rest) = value.strip_prefix("confidence:") {
			let (confidence, key) = rest
				.split_once(':')
				.ok_or_else(|| TraitFilterError(value.clone()))?;
			let confidence =
				parse_confidence(confidence).ok_or_else(|| TraitFilterError(value.clone()))?;
			if !known_capability(key) && !valid_extension_key(key) {
				return Err(TraitFilterError(value));
			}
			TraitPredicate::MinimumConfidence {
				key: key.to_string(),
				confidence,
			}
		} else if let Some(stage) = value.strip_prefix("mtp_stage>=") {
			TraitPredicate::MinimumMtp(
				parse_mtp_stage(stage).ok_or_else(|| TraitFilterError(value.clone()))?,
			)
		} else if let Some((metric, amount)) = parse_metric_predicate(&value, "<=") {
			TraitPredicate::AtMost {
				metric,
				value: amount?,
			}
		} else if let Some((metric, amount)) = parse_metric_predicate(&value, ">=") {
			TraitPredicate::AtLeast {
				metric,
				value: amount?,
			}
		} else {
			return Err(TraitFilterError(value));
		};
		Ok(Self {
			source: value,
			predicate,
		})
	}

	/// Canonical namespaced key.
	pub fn as_str(&self) -> &str {
		&self.source
	}

	/// Parsed typed predicate.
	pub const fn predicate(&self) -> &TraitPredicate {
		&self.predicate
	}
}

impl fmt::Display for TraitFilter {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter.write_str(self.as_str())
	}
}

const fn known_capability(value: &str) -> bool {
	matches!(
		value.as_bytes(),
		b"input:text"
			| b"input:image"
			| b"input:audio"
			| b"output:text"
			| b"output:image"
			| b"output:audio"
			| b"task:text_generation"
			| b"task:chat"
			| b"task:translation"
			| b"interaction:tools"
			| b"interaction:system_prompt"
			| b"interaction:reasoning"
			| b"interaction:reasoning_history"
			| b"interaction:thinking_toggle"
			| b"interaction:structured_output"
			| b"acceleration:mlx"
			| b"acceleration:mtp"
			| b"acceleration:mtp_advertised"
	)
}

fn valid_extension_key(value: &str) -> bool {
	value.strip_prefix("extension:").is_some_and(|name| {
		!name.is_empty()
			&& name.len() <= 128
			&& name.bytes().all(|byte| {
				byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.' | b':')
			})
	})
}

fn parse_confidence(value: &str) -> Option<TraitConfidence> {
	match value {
		"advertised" => Some(TraitConfidence::Advertised),
		"inferred" => Some(TraitConfidence::Inferred),
		"runtime_verified" => Some(TraitConfidence::RuntimeVerified),
		_ => None,
	}
}

fn parse_mtp_stage(value: &str) -> Option<MtpSupport> {
	match value {
		"absent" => Some(MtpSupport::Absent),
		"advertised" => Some(MtpSupport::Advertised),
		"runtime_verified" => Some(MtpSupport::RuntimeVerified),
		_ => None,
	}
}

fn parse_metric_predicate(
	value: &str,
	operator: &str,
) -> Option<(TraitMetric, Result<u64, TraitFilterError>)> {
	let (name, amount) = value.split_once(operator)?;
	let metric = match name {
		"weights_bytes" => TraitMetric::WeightsBytes,
		"residency_bytes" => TraitMetric::EstimatedResidencyBytes,
		"context_tokens" => TraitMetric::EvaluatedContextTokens,
		"max_context_tokens" => TraitMetric::MaximumContextTokens,
		_ => return None,
	};
	Some((
		metric,
		amount
			.parse()
			.map_err(|_| TraitFilterError(value.to_string())),
	))
}

impl FromStr for TraitFilter {
	type Err = TraitFilterError;

	fn from_str(value: &str) -> Result<Self, Self::Err> {
		Self::parse(value)
	}
}

impl Serialize for TraitFilter {
	fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
	where
		S: serde::Serializer,
	{
		serializer.serialize_str(self.as_str())
	}
}

impl<'de> Deserialize<'de> for TraitFilter {
	fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
	where
		D: serde::Deserializer<'de>,
	{
		let value = String::deserialize(deserializer)?;
		Self::parse(value).map_err(serde::de::Error::custom)
	}
}

/// Invalid capability filter.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
#[error("unknown model trait filter {0:?}")]
pub struct TraitFilterError(String);

#[cfg(test)]
#[path = "traits_tests.rs"]
mod tests;
