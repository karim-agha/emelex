//! Hugging Face discovery and bounded immutable downloads.

use std::{
	collections::{BTreeMap, BTreeSet},
	ffi::CString,
	fmt,
	fs::OpenOptions,
	os::unix::{
		ffi::OsStrExt as _,
		fs::{MetadataExt as _, OpenOptionsExt as _, PermissionsExt as _},
	},
	path::{Path, PathBuf},
	sync::{
		Arc,
		atomic::{AtomicBool, Ordering},
	},
	time::Duration,
};

use base64::Engine as _;
use futures::{StreamExt as _, future::join_all};
use reqwest::{StatusCode, Url, header};
use serde::{Deserialize, Deserializer, Serialize, de::DeserializeOwned};
use sha2::{Digest as _, Sha256};
use tokio::io::{AsyncRead, AsyncReadExt as _, AsyncWriteExt as _};

use crate::{
	config::HubConfig,
	model::{
		EvidenceSource, FitReport, HubModelId, Modality, ModelFile, ModelGenerationDefaults,
		ModelSizing, ModelTraits, MtpSupport, ResolvedRevision, Task, TraitConfidence,
		TraitEvidence, TraitFilter, TraitPredicate, WorkloadProfile,
	},
};

const MAX_API_BODY_BYTES: usize = 32 << 20;
const MAX_ERROR_BODY_BYTES: usize = 8 << 10;
const MAX_TREE_ENTRIES: usize = 50_000;
const MAX_TREE_PAGES: usize = 20;
const MAX_RUNTIME_COMPONENT_BYTES: usize = 255;
const MAX_SEARCH_QUERY_BYTES: usize = 1 << 10;
const MAX_SEARCH_CURSOR_BYTES: usize = 2 << 10;
const MAX_UPSTREAM_CURSOR_BYTES: usize = 1 << 10;
const HUB_MLX_FILTER: &str = "mlx";
const SEARCH_CURSOR_VERSION: u8 = 3;

/// One catalog query.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct HubSearch {
	/// Search text. Empty searches preserve the Hub's global ranking.
	pub query: Option<String>,
	/// Required typed capability filters.
	pub require: Vec<TraitFilter>,
	/// Opaque cursor returned by a previous page.
	pub cursor: Option<String>,
	mlx_library: bool,
}

/// One remotely evaluable catalog-filter shape shown by capability discovery.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct RemoteFilterHelp {
	/// User-facing filter or filter-family syntax.
	pub filter: &'static str,
	/// Strongest evidence source available before installation.
	pub evidence: &'static str,
	/// Semantic meaning and important limits.
	pub meaning: &'static str,
	/// Concrete validator-checked member of this filter family.
	#[doc(hidden)]
	#[serde(skip)]
	pub example: &'static str,
}

/// Complete remotely evaluable filter catalog.
pub const REMOTE_FILTERS: &[RemoteFilterHelp] = &[
	RemoteFilterHelp {
		filter: "input:text",
		evidence: "inferred",
		meaning: "supported autoregressive architecture consumes tokenizer text",
		example: "input:text",
	},
	RemoteFilterHelp {
		filter: "output:text",
		evidence: "inferred",
		meaning: "supported autoregressive architecture decodes tokenizer text",
		example: "output:text",
	},
	RemoteFilterHelp {
		filter: "acceleration:mlx",
		evidence: "inferred",
		meaning: "supported MLX architecture, configuration, quantization, and file layout",
		example: "acceleration:mlx",
	},
	RemoteFilterHelp {
		filter: "acceleration:mtp_advertised",
		evidence: "advertised",
		meaning: "repository configuration advertises MTP; runtime certification requires install",
		example: "acceleration:mtp_advertised",
	},
	RemoteFilterHelp {
		filter: "extension:huggingface.advertised_input_image",
		evidence: "advertised",
		meaning: "Hub metadata advertises image input; not a runnable compatibility claim",
		example: "extension:huggingface.advertised_input_image",
	},
	RemoteFilterHelp {
		filter: "extension:huggingface.advertised_input_audio",
		evidence: "advertised",
		meaning: "Hub metadata advertises audio input; not a runnable compatibility claim",
		example: "extension:huggingface.advertised_input_audio",
	},
	RemoteFilterHelp {
		filter: "task:text_generation",
		evidence: "inferred",
		meaning: "exact revision uses a supported autoregressive architecture",
		example: "task:text_generation",
	},
	RemoteFilterHelp {
		filter: "task:chat",
		evidence: "inferred",
		meaning: "runtime-selected template preserves conversational user turns",
		example: "task:chat",
	},
	RemoteFilterHelp {
		filter: "interaction:system_prompt",
		evidence: "inferred",
		meaning: "runtime-selected template independently preserves system instructions",
		example: "interaction:system_prompt",
	},
	RemoteFilterHelp {
		filter: "interaction:tools",
		evidence: "inferred",
		meaning: "selected template tool history round-trips through exact runtime parser",
		example: "interaction:tools",
	},
	RemoteFilterHelp {
		filter: "interaction:reasoning",
		evidence: "inferred",
		meaning: "chat template exposes reasoning history, a thinking toggle, or both",
		example: "interaction:reasoning",
	},
	RemoteFilterHelp {
		filter: "interaction:reasoning_history",
		evidence: "inferred",
		meaning: "explicit reasoning spans survive a follow-up turn",
		example: "interaction:reasoning_history",
	},
	RemoteFilterHelp {
		filter: "interaction:thinking_toggle",
		evidence: "inferred",
		meaning: "thinking-on and thinking-off renders differ semantically",
		example: "interaction:thinking_toggle",
	},
	RemoteFilterHelp {
		filter: "weights_bytes{<=|>=}N",
		evidence: "measured",
		meaning: "selected safetensors byte total satisfies an inclusive bound",
		example: "weights_bytes<=1",
	},
	RemoteFilterHelp {
		filter: "residency_bytes{<=|>=}N",
		evidence: "estimated",
		meaning: "profiled peak residency satisfies an inclusive bound",
		example: "residency_bytes<=1",
	},
	RemoteFilterHelp {
		filter: "context_tokens{<=|>=}N",
		evidence: "evaluated",
		meaning: "fit profile context used for residency evaluation satisfies an inclusive bound",
		example: "context_tokens>=1",
	},
	RemoteFilterHelp {
		filter: "max_context_tokens{<=|>=}N",
		evidence: "advertised",
		meaning: "validated architecture limit satisfies an inclusive token bound",
		example: "max_context_tokens>=1",
	},
	RemoteFilterHelp {
		filter: "mtp_stage>=absent|advertised",
		evidence: "advertised",
		meaning: "require a remotely knowable MTP stage; runtime_verified requires installation",
		example: "mtp_stage>=advertised",
	},
	RemoteFilterHelp {
		filter: "confidence:{advertised|inferred}:TRAIT",
		evidence: "typed",
		meaning: "require a remotely available capability at a minimum evidence confidence",
		example: "confidence:advertised:input:text",
	},
];

impl HubSearch {
	/// Set non-empty Hub search text.
	#[must_use]
	pub fn query(mut self, query: impl Into<String>) -> Self {
		self.query = Some(query.into());
		self
	}

	/// Replace required trait conjunction.
	#[must_use]
	pub fn requirements(mut self, require: Vec<TraitFilter>) -> Self {
		self.require = require;
		self
	}

	/// Resume from an opaque cursor returned by [`HubSearchPage::next_cursor`].
	#[must_use]
	pub fn cursor(mut self, cursor: impl Into<String>) -> Self {
		self.cursor = Some(cursor.into());
		self
	}

	/// Restrict discovery to Hugging Face's MLX library catalog.
	#[must_use]
	pub const fn mlx_library(mut self) -> Self {
		self.mlx_library = true;
		self
	}
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchCursor {
	version: u8,
	upstream: Option<String>,
	offset: usize,
	scope: String,
}

/// Quantization mode reported by an exact Hub revision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum HubQuantizationMode {
	/// Grouped affine quantization.
	Affine,
	/// Microscaling four-bit floating point.
	Mxfp4,
	/// Microscaling eight-bit floating point.
	Mxfp8,
	/// NVIDIA four-bit floating point.
	Nvfp4,
}

/// Exact-revision quantization configuration reported by the Hub.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum HubQuantization {
	/// Quantization was not inspected, including older serialized records.
	#[default]
	Unknown,
	/// No quantization section is configured.
	NotConfigured,
	/// Quantization defaults are configured for layers carrying quantization tensors.
	Configured(HubQuantizationConfig),
}

/// Validated exact-revision quantization defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[non_exhaustive]
pub struct HubQuantizationConfig {
	mode: HubQuantizationMode,
	bits: u8,
	group_size: u16,
	has_layer_overrides: bool,
}

impl HubQuantizationConfig {
	fn new(
		mode: HubQuantizationMode,
		bits: u8,
		group_size: u16,
		has_layer_overrides: bool,
	) -> crate::engine::error::Result<Self> {
		crate::engine::quant::validate_params(
			crate::engine::quant::QuantParams {
				group_size: i32::from(group_size),
				bits: i32::from(bits),
				mode: match mode {
					HubQuantizationMode::Affine => crate::engine::ops::QuantMode::Affine,
					HubQuantizationMode::Mxfp4 => crate::engine::ops::QuantMode::Mxfp4,
					HubQuantizationMode::Mxfp8 => crate::engine::ops::QuantMode::Mxfp8,
					HubQuantizationMode::Nvfp4 => crate::engine::ops::QuantMode::Nvfp4,
				},
			},
			"Hub quantization",
		)?;
		Ok(Self {
			mode,
			bits,
			group_size,
			has_layer_overrides,
		})
	}

	/// Default quantization mode.
	pub const fn mode(self) -> HubQuantizationMode {
		self.mode
	}

	/// Default bits per quantized weight.
	pub const fn bits(self) -> u8 {
		self.bits
	}

	/// Default quantization group size.
	pub const fn group_size(self) -> u16 {
		self.group_size
	}

	/// Whether any per-layer overrides are configured.
	pub const fn has_layer_overrides(self) -> bool {
		self.has_layer_overrides
	}
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum HubQuantizationWire {
	Unknown,
	NotConfigured,
	Configured {
		mode: HubQuantizationMode,
		bits: u8,
		group_size: u16,
		has_layer_overrides: bool,
	},
}

impl<'de> Deserialize<'de> for HubQuantization {
	fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
	where
		D: Deserializer<'de>,
	{
		match HubQuantizationWire::deserialize(deserializer)? {
			HubQuantizationWire::Unknown => Ok(Self::Unknown),
			HubQuantizationWire::NotConfigured => Ok(Self::NotConfigured),
			HubQuantizationWire::Configured {
				mode,
				bits,
				group_size,
				has_layer_overrides,
			} => HubQuantizationConfig::new(mode, bits, group_size, has_layer_overrides)
				.map(Self::Configured)
				.map_err(serde::de::Error::custom),
		}
	}
}

/// One Hub model returned by discovery or inspection.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct HubModel {
	/// Validated Hub identity.
	pub id: HubModelId,
	/// Immutable current repository commit.
	pub revision: ResolvedRevision,
	/// Hub download count.
	pub downloads: u64,
	/// Hub like count.
	pub likes: u64,
	/// Repository tags.
	pub tags: Vec<String>,
	/// Library metadata.
	pub library: Option<String>,
	/// Model-card license tag.
	pub license: Option<String>,
	/// Generic capabilities with evidence.
	pub traits: ModelTraits,
	/// Exact-revision quantization configuration.
	#[serde(default)]
	pub quantization: HubQuantization,
	/// Whether remote static compatibility and any configured machine fit passed.
	pub compatible: bool,
	/// Root file names advertised by the repository.
	pub files: Vec<String>,
	/// Static rejection explanations. Empty means remote preflight passed.
	pub diagnostics: Vec<String>,
	/// Machine fit when the Hub client was constructed with a fit profile.
	pub fit: Option<FitReport>,
	#[serde(skip)]
	download_bytes: Option<u64>,
}

/// Compatible search page.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct HubSearchPage {
	/// Compatible items in original Hub rank order.
	pub items: Vec<HubModel>,
	/// Opaque next-page cursor.
	pub next_cursor: Option<String>,
	/// Number of Hub candidates examined.
	pub scanned: usize,
	/// Candidate-local failures that did not abort the page.
	pub diagnostics: Vec<HubDiagnostic>,
}

/// One candidate-local discovery failure.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct HubDiagnostic {
	/// Candidate identity when it was valid.
	pub id: Option<HubModelId>,
	/// Bounded diagnostic.
	pub message: String,
}

/// One immutable remote file selected for installation.
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct RemoteFile {
	/// Safe repository-relative path.
	path: String,
	/// Exact expected bytes.
	size: u64,
	/// LFS SHA-256 when the Hub supplied one.
	expected_sha256: Option<String>,
}

impl RemoteFile {
	/// Safe repository-relative runtime path.
	pub fn path(&self) -> &str {
		&self.path
	}

	/// Exact expected byte length.
	pub const fn size(&self) -> u64 {
		self.size
	}

	/// Hub-provided LFS SHA-256, when available.
	pub fn expected_sha256(&self) -> Option<&str> {
		self.expected_sha256.as_deref()
	}

	fn new(path: String, size: u64, expected_sha256: Option<String>) -> Result<Self, HubError> {
		if !safe_runtime_file(&path) {
			return Err(HubError::Protocol(format!(
				"unsafe Hub runtime path {path:?}"
			)));
		}
		if path.len() > MAX_RUNTIME_COMPONENT_BYTES {
			return Err(HubError::Protocol(format!(
				"Hub runtime path exceeds {MAX_RUNTIME_COMPONENT_BYTES} bytes"
			)));
		}
		if size == 0 && path.ends_with(".safetensors") {
			return Err(HubError::Incompatible(format!(
				"zero-byte weight file {path:?}"
			)));
		}
		let expected_sha256 = expected_sha256.map(|digest| digest.to_ascii_lowercase());
		if expected_sha256.as_ref().is_some_and(|digest| {
			digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit())
		}) {
			return Err(HubError::Protocol(format!(
				"invalid LFS digest for {path:?}"
			)));
		}
		Ok(Self {
			path,
			size,
			expected_sha256,
		})
	}
}

impl<'de> Deserialize<'de> for RemoteFile {
	fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
	where
		D: Deserializer<'de>,
	{
		#[derive(Deserialize)]
		#[serde(deny_unknown_fields)]
		struct Wire {
			path: String,
			size: u64,
			expected_sha256: Option<String>,
		}

		let wire = Wire::deserialize(deserializer)?;
		Self::new(wire.path, wire.size, wire.expected_sha256).map_err(serde::de::Error::custom)
	}
}

/// Immutable runnable file plan for one Hub revision.
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct DownloadPlan {
	/// Hub repository.
	model: HubModel,
	/// Selected root-level runtime files.
	files: Vec<RemoteFile>,
	/// Total transfer bytes.
	total_bytes: u64,
}

impl DownloadPlan {
	/// Inspected Hub model and immutable revision.
	pub const fn model(&self) -> &HubModel {
		&self.model
	}

	/// Validated runtime files in stable path order.
	pub fn files(&self) -> &[RemoteFile] {
		&self.files
	}

	/// Exact aggregate transfer bytes.
	pub const fn total_bytes(&self) -> u64 {
		self.total_bytes
	}

	fn new(model: HubModel, files: Vec<RemoteFile>) -> Result<Self, HubError> {
		let total_bytes = validate_download_files(&files)?;
		Ok(Self {
			model,
			files,
			total_bytes,
		})
	}
}

/// Download lifecycle event.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum DownloadEvent {
	/// Planned transfer began after staging validation.
	TransferStarted {
		/// Number of planned files.
		files: usize,
		/// Aggregate planned bytes, including resumable prefixes.
		total: u64,
	},
	/// File transfer started or resumed.
	FileStarted {
		/// Relative file.
		path: String,
		/// Bytes already present.
		resumed: u64,
		/// Expected total.
		total: u64,
	},
	/// New bytes persisted.
	Progress {
		/// Relative file.
		path: String,
		/// Current bytes.
		received: u64,
		/// Expected total.
		total: u64,
	},
	/// Transient transfer failure will retry.
	Retrying {
		/// Relative file.
		path: String,
		/// One-based failed attempt.
		attempt: usize,
		/// Rendered reason.
		reason: String,
	},
	/// File hash verified.
	FileVerified {
		/// Relative file.
		path: String,
		/// Computed SHA-256.
		sha256: String,
	},
	/// Every planned file was transferred, verified, and staged.
	TransferCompleted {
		/// Number of transferred files.
		files: usize,
		/// Aggregate transferred bytes, including resumable prefixes.
		total: u64,
	},
}

/// Shareable progress callback.
pub type DownloadReporter = Arc<dyn Fn(&DownloadEvent) + Send + Sync>;

/// Observer decision after one download event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DownloadControl {
	/// Continue transfer.
	Continue,
	/// Cooperatively cancel before another network or file operation.
	Cancel,
}

/// Shareable fallible and cancellable progress observer.
pub type DownloadObserver =
	Arc<dyn Fn(&DownloadEvent) -> Result<DownloadControl, DownloadObserverError> + Send + Sync>;

/// Reporter-side failure.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
#[error("{message}")]
pub struct DownloadObserverError {
	message: String,
}

impl DownloadObserverError {
	/// Construct a bounded reporter failure.
	///
	/// # Errors
	///
	/// Returns [`HubError::Protocol`] when the message is empty or exceeds
	/// 8 KiB.
	pub fn new(message: impl Into<String>) -> Result<Self, HubError> {
		let message = message.into();
		if message.is_empty() || message.len() > MAX_ERROR_BODY_BYTES {
			return Err(HubError::Protocol(
				"download observer error must contain 1..=8192 bytes".to_string(),
			));
		}
		Ok(Self { message })
	}
}

// Linked state lets an operation observe caller cancellation without gaining
// authority to cancel the caller's other operations.
#[derive(Debug)]
struct DownloadCancellationState {
	cancelled: AtomicBool,
	parent: Option<Arc<Self>>,
}

/// Cooperative cancellation handle for long transfers and retry waits.
#[derive(Debug, Clone)]
pub struct DownloadCancellation(Arc<DownloadCancellationState>);

impl Default for DownloadCancellation {
	fn default() -> Self {
		Self(Arc::new(DownloadCancellationState {
			cancelled: AtomicBool::new(false),
			parent: None,
		}))
	}
}

impl DownloadCancellation {
	pub(crate) fn linked(parent: Option<&Self>) -> Self {
		Self(Arc::new(DownloadCancellationState {
			cancelled: AtomicBool::new(false),
			parent: parent.map(|parent| Arc::clone(&parent.0)),
		}))
	}

	/// Request cancellation.
	pub fn cancel(&self) {
		self.0.cancelled.store(true, Ordering::Release);
	}

	/// Whether cancellation was requested.
	pub fn is_cancelled(&self) -> bool {
		let mut state = Some(self.0.as_ref());
		while let Some(current) = state {
			if current.cancelled.load(Ordering::Acquire) {
				return true;
			}
			state = current.parent.as_deref();
		}
		false
	}
}

pub(crate) struct DownloadOperationGuard {
	cancellation: DownloadCancellation,
	active: bool,
}

impl DownloadOperationGuard {
	pub(crate) fn new(parent: Option<&DownloadCancellation>) -> Self {
		Self {
			cancellation: DownloadCancellation::linked(parent),
			active: true,
		}
	}

	pub(crate) const fn cancellation(&self) -> &DownloadCancellation {
		&self.cancellation
	}

	pub(crate) const fn finish(&mut self) {
		self.active = false;
	}
}

impl Drop for DownloadOperationGuard {
	fn drop(&mut self) {
		if self.active {
			self.cancellation.cancel();
		}
	}
}

#[derive(Clone, Copy)]
struct DownloadCallbacks<'a> {
	reporter: Option<&'a DownloadReporter>,
	observer: Option<&'a DownloadObserver>,
	cancellation: Option<&'a DownloadCancellation>,
}

/// Explicit Hugging Face bearer credentials.
///
/// Debug output is always redacted. The contained authorization header is
/// marked sensitive so HTTP middleware does not render it.
#[derive(Clone)]
pub struct HubCredentials {
	authorization: header::HeaderValue,
	scope: [u8; 32],
}

impl HubCredentials {
	/// Construct credentials from one Hugging Face access token.
	///
	/// # Errors
	///
	/// Returns [`HubError`] when the token is empty, too large, contains
	/// controls, or cannot be represented as an HTTP header.
	pub fn bearer_token(token: &str) -> Result<Self, HubError> {
		if token.is_empty()
			|| token.len() > 4_096
			|| !token.bytes().all(|byte| byte.is_ascii_graphic())
		{
			return Err(HubError::Configuration(
				"Hugging Face token must contain 1..=4096 visible ASCII bytes".to_string(),
			));
		}
		let mut authorization =
			header::HeaderValue::from_str(&format!("Bearer {token}")).map_err(|_| {
				HubError::Configuration(
					"Hugging Face token cannot be represented as an HTTP header".to_string(),
				)
			})?;
		authorization.set_sensitive(true);
		let mut hasher = Sha256::new();
		hasher.update(b"emelex:hub-credential-scope:v1\0");
		hasher.update(token.as_bytes());
		Ok(Self {
			authorization,
			scope: hasher.finalize().into(),
		})
	}
}

impl fmt::Debug for HubCredentials {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter
			.debug_struct("HubCredentials")
			.field("authorization", &"[REDACTED]")
			.finish()
	}
}

/// Hugging Face client with anonymous-by-default authentication.
#[derive(Clone)]
pub struct HubClient {
	endpoint: Url,
	client: reqwest::Client,
	config: HubConfig,
	fit_profile: Option<(WorkloadProfile, u64)>,
	search_storage_path: Option<PathBuf>,
	credential_scope: Option<[u8; 32]>,
	#[cfg(test)]
	publish_gate: Option<Arc<TestPublishGate>>,
}

#[cfg(test)]
#[derive(Debug, Default)]
struct TestPublishGate {
	entered: AtomicBool,
	release: AtomicBool,
	completed: AtomicBool,
}

impl HubClient {
	/// Construct a static-only client for `https://huggingface.co`.
	///
	/// Static clients inspect exact repository metadata but make no claim
	/// about fit on the current machine. Use [`Self::with_fit_profile`] when
	/// fit filtering is required.
	///
	/// # Errors
	///
	/// Returns [`HubError`] when the HTTP client cannot be configured.
	pub fn new(config: HubConfig) -> Result<Self, HubError> {
		Self::with_endpoint_and_fit(config, "https://huggingface.co", None, None)
	}

	/// Construct an explicitly authenticated static-only client.
	///
	/// # Errors
	///
	/// Returns [`HubError`] when the HTTP client cannot be configured.
	pub fn with_credentials(
		config: HubConfig,
		credentials: HubCredentials,
	) -> Result<Self, HubError> {
		Self::with_endpoint_and_fit(config, "https://huggingface.co", None, Some(credentials))
	}

	/// Workload used by discovery residency estimates, when configured.
	pub const fn fit_workload(&self) -> Option<WorkloadProfile> {
		match self.fit_profile {
			Some((workload, _)) => Some(workload),
			None => None,
		}
	}

	/// Metal budget used by discovery fit reports, when configured.
	pub const fn metal_budget_bytes(&self) -> Option<u64> {
		match self.fit_profile {
			Some((_, budget)) => Some(budget),
			None => None,
		}
	}

	/// Whether this client sends explicit Hugging Face authorization.
	pub const fn is_authenticated(&self) -> bool {
		self.credential_scope.is_some()
	}

	/// Construct a profiled client whose discovery reports use one invocation's
	/// exact workload and Metal working-set budget.
	///
	/// # Errors
	///
	/// Returns [`HubError`] for invalid Hub configuration, endpoint, or a
	/// zero Metal budget.
	pub fn with_fit_profile(
		config: HubConfig,
		workload: WorkloadProfile,
		metal_budget_bytes: u64,
	) -> Result<Self, HubError> {
		Self::with_endpoint_and_fit(
			config,
			"https://huggingface.co",
			Some((workload, metal_budget_bytes)),
			None,
		)
	}

	/// Construct an explicitly authenticated client with fit reporting.
	///
	/// # Errors
	///
	/// Returns [`HubError`] for invalid Hub configuration, endpoint, or a
	/// zero Metal budget.
	pub fn with_fit_profile_and_credentials(
		config: HubConfig,
		workload: WorkloadProfile,
		metal_budget_bytes: u64,
		credentials: HubCredentials,
	) -> Result<Self, HubError> {
		Self::with_endpoint_and_fit(
			config,
			"https://huggingface.co",
			Some((workload, metal_budget_bytes)),
			Some(credentials),
		)
	}

	/// Construct the model-manager client with local search storage filtering.
	pub(crate) fn with_local_search_profile(
		config: HubConfig,
		workload: WorkloadProfile,
		metal_budget_bytes: u64,
		search_storage_path: PathBuf,
		credentials: Option<HubCredentials>,
	) -> Result<Self, HubError> {
		let mut client = Self::with_endpoint_and_fit(
			config,
			"https://huggingface.co",
			Some((workload, metal_budget_bytes)),
			credentials,
		)?;
		client.search_storage_path = Some(search_storage_path);
		Ok(client)
	}

	fn with_endpoint_and_fit(
		config: HubConfig,
		endpoint: &str,
		fit_profile: Option<(WorkloadProfile, u64)>,
		credentials: Option<HubCredentials>,
	) -> Result<Self, HubError> {
		config
			.validate()
			.map_err(|error| HubError::Configuration(error.to_string()))?;
		let endpoint = Url::parse(endpoint).map_err(|error| HubError::Url(error.to_string()))?;
		if !matches!(endpoint.scheme(), "https" | "http") {
			return Err(HubError::Url(
				"Hub endpoint must use HTTP or HTTPS".to_string(),
			));
		}
		if fit_profile.is_some_and(|(_, budget)| budget == 0) {
			return Err(HubError::Configuration(
				"Metal budget must be positive".to_string(),
			));
		}
		let credential_scope = credentials.as_ref().map(|credentials| credentials.scope);
		let mut default_headers = header::HeaderMap::new();
		if let Some(credentials) = credentials {
			default_headers.insert(header::AUTHORIZATION, credentials.authorization);
		}
		let client = reqwest::Client::builder()
			.user_agent(concat!("emelex/", env!("CARGO_PKG_VERSION")))
			.default_headers(default_headers)
			.connect_timeout(config.request_timeout())
			.read_timeout(config.request_timeout())
			.referer(false)
			.https_only(endpoint.scheme() == "https")
			.redirect(reqwest::redirect::Policy::limited(5))
			.build()
			.map_err(request_error)?;
		Ok(Self {
			endpoint,
			client,
			config,
			fit_profile,
			search_storage_path: None,
			credential_scope,
			#[cfg(test)]
			publish_gate: None,
		})
	}

	/// Search the accessible catalog, preserving Hub ranking while filtering
	/// statically compatible candidates and, when configured, machine fit.
	///
	/// # Errors
	///
	/// Returns network, protocol, identity, or metadata errors.
	#[allow(
		clippy::too_many_lines,
		reason = "rank-preserving concurrent candidate handling is kept in one ordered pipeline"
	)]
	pub async fn search(&self, search: &HubSearch) -> Result<HubSearchPage, HubError> {
		validate_remote_filters(&search.require)?;
		let normalized_query = normalized_search_query(search)?;
		let search_storage_budget_bytes = self.current_search_storage_budget_bytes()?;
		let position =
			decode_search_cursor(search, self.credential_scope.as_ref(), self.fit_profile)?;
		let mut url = self.api_url(&["api", "models"])?;
		{
			let mut query = url.query_pairs_mut();
			query.append_pair("full", "true");
			query.append_pair("config", "true");
			if search.mlx_library {
				query.append_pair("filter", HUB_MLX_FILTER);
			}
			query.append_pair("limit", &self.config.scan_limit.to_string());
			if let Some(value) = normalized_query {
				query.append_pair("search", value);
			}
			if let Some(cursor) = &position.upstream {
				query.append_pair("cursor", cursor);
			}
		}
		let response = self.send(self.client.get(url)).await?;
		let next_upstream = next_upstream_cursor(response.headers())?;
		let candidates: Vec<ModelWire> =
			decode_response(response, MAX_API_BODY_BYTES, "model search").await?;
		if candidates.len() > self.config.scan_limit {
			return Err(HubError::Protocol(format!(
				"Hub search returned {} candidates above configured scan limit {}",
				candidates.len(),
				self.config.scan_limit
			)));
		}
		if position.offset > candidates.len() {
			return Err(HubError::Protocol(
				"Hub search cursor offset exceeds its candidate page".to_string(),
			));
		}
		let candidate_count = candidates.len();
		let mut immediately_scanned = 0;
		let mut metadata_scanned = 0;
		let mut diagnostics = Vec::new();
		let mut ranked = Vec::new();
		for (rank, wire) in candidates.into_iter().enumerate().skip(position.offset) {
			if !self.is_authenticated() && (wire.private || gated(&wire.gated)) {
				immediately_scanned += 1;
				diagnostics.push(HubDiagnostic {
					id: HubModelId::parse(wire.id).ok(),
					message: "candidate is private or gated".to_string(),
				});
				continue;
			}
			match HubModelId::parse(wire.id) {
				Ok(id) => ranked.push((rank, id)),
				Err(error) => {
					immediately_scanned += 1;
					diagnostics.push(HubDiagnostic {
						id: None,
						message: error.to_string(),
					});
				}
			}
		}
		let mut compatible = Vec::new();
		for chunk in ranked.chunks(self.config.metadata_concurrency) {
			let futures = chunk.iter().map(|(rank, id)| {
				let client = self.clone();
				let id = id.clone();
				async move {
					let result = client.inspect(&id).await;
					(*rank, id, result)
				}
			});
			let mut results = join_all(futures).await;
			results.sort_by_key(|(rank, _, _)| *rank);
			let mut accepted = Vec::new();
			for (rank, id, result) in results {
				metadata_scanned += 1;
				match result {
					Ok(model)
						if model.compatible
							&& search
								.require
								.iter()
								.all(|filter| model.traits.satisfies(filter)) =>
					{
						accepted.push((rank, model));
					}
					Ok(model) => {
						let model_id = model.id;
						diagnostics.extend(model.diagnostics.into_iter().map(|message| {
							HubDiagnostic {
								id: Some(model_id.clone()),
								message,
							}
						}));
					}
					Err(error) if candidate_inspection_diagnostic(&error) => {
						diagnostics.push(HubDiagnostic {
							id: Some(id),
							message: error.to_string(),
						});
					}
					Err(error) => return Err(error),
				}
			}
			let accepted =
				filter_search_storage(accepted, search_storage_budget_bytes, &mut diagnostics);
			if let Some(offset) =
				append_compatible_page(&mut compatible, accepted, self.config.results)
			{
				return Ok(HubSearchPage {
					items: compatible,
					next_cursor: continuation_cursor(
						search,
						self.credential_scope.as_ref(),
						self.fit_profile,
						position.upstream.clone(),
						offset,
						candidate_count,
						next_upstream,
					)?,
					scanned: search_scanned(immediately_scanned, metadata_scanned),
					diagnostics,
				});
			}
		}
		Ok(HubSearchPage {
			items: compatible,
			next_cursor: next_upstream
				.map(|upstream| {
					encode_search_cursor(
						search,
						self.credential_scope.as_ref(),
						self.fit_profile,
						Some(upstream),
						0,
					)
				})
				.transpose()?,
			scanned: search_scanned(immediately_scanned, metadata_scanned),
			diagnostics,
		})
	}

	fn current_search_storage_budget_bytes(&self) -> Result<Option<u64>, HubError> {
		let Some(path) = &self.search_storage_path else {
			return Ok(None);
		};
		crate::home::available_disk_bytes(path)
			.map(Some)
			.map_err(|source| {
				if matches!(
					source.kind(),
					std::io::ErrorKind::InvalidInput | std::io::ErrorKind::InvalidData
				) {
					HubError::Configuration(source.to_string())
				} else {
					HubError::Io {
						path: path.clone(),
						source,
					}
				}
			})
	}

	/// Inspect one repository visible to this client at its current revision.
	///
	/// # Errors
	///
	/// Returns an error for inaccessible, unsupported, or malformed models.
	pub async fn inspect(&self, id: &HubModelId) -> Result<HubModel, HubError> {
		let (model, _) = self.inspect_and_plan(id).await?;
		Ok(model)
	}

	async fn inspect_metadata(&self, id: &HubModelId) -> Result<(HubModel, ModelWire), HubError> {
		let segments = model_api_segments(id);
		let mut url = self.api_url(&segments)?;
		url.query_pairs_mut()
			.append_pair("blobs", "true")
			.append_pair("securityStatus", "true");
		let response = self.send(self.client.get(url)).await?;
		let wire: ModelWire =
			decode_response(response, MAX_API_BODY_BYTES, "model metadata").await?;
		let model = model_from_wire(&wire, self.is_authenticated())?;
		Ok((model, wire))
	}

	/// Resolve the immutable runnable subset for a model.
	///
	/// # Errors
	///
	/// Returns an error for unsafe, missing, oversized, or unsupported plans.
	pub async fn plan(&self, id: &HubModelId) -> Result<DownloadPlan, HubError> {
		let (model, files) = self.inspect_and_plan(id).await?;
		if !model.compatible {
			return Err(HubError::Incompatible(model.diagnostics.join("; ")));
		}
		DownloadPlan::new(model, files)
	}

	async fn inspect_and_plan(
		&self,
		id: &HubModelId,
	) -> Result<(HubModel, Vec<RemoteFile>), HubError> {
		let (mut model, wire) = self.inspect_metadata(id).await?;
		let tree = self.repository_tree(id, &model.revision).await?;
		let artifacts = self.revision_artifacts(&model, &tree).await?;
		let files = exact_runtime_plan(
			&tree,
			artifacts.index.as_ref(),
			processor_has_chat_template(artifacts.processor_config.as_ref()),
		)?;
		enrich_remote_model(&mut model, &wire, &artifacts, &files, self.fit_profile)?;
		Ok((model, files))
	}

	async fn repository_tree(
		&self,
		id: &HubModelId,
		revision: &ResolvedRevision,
	) -> Result<Vec<TreeWire>, HubError> {
		let mut segments = model_api_segments(id);
		segments.extend(["tree", revision.as_str()]);
		let mut url = self.api_url(&segments)?;
		url.query_pairs_mut()
			.append_pair("recursive", "true")
			.append_pair("expand", "true");
		let mut tree = Vec::new();
		let mut pages = 0_usize;
		loop {
			pages += 1;
			if pages > MAX_TREE_PAGES {
				return Err(HubError::Protocol(
					"Hub tree pagination exceeded 20 pages".to_string(),
				));
			}
			let response = self.send(self.client.get(url.clone())).await?;
			let next = next_link(response.headers(), &self.endpoint)?;
			let mut page: Vec<TreeWire> =
				decode_response(response, MAX_API_BODY_BYTES, "repository tree").await?;
			if page.len() > MAX_TREE_ENTRIES.saturating_sub(tree.len()) {
				return Err(HubError::Protocol(format!(
					"Hub tree exceeds {MAX_TREE_ENTRIES} entries"
				)));
			}
			tree.append(&mut page);
			match next {
				Some(next) => url = next,
				None => break,
			}
		}
		Ok(tree)
	}

	async fn revision_artifacts(
		&self,
		model: &HubModel,
		tree: &[TreeWire],
	) -> Result<RevisionArtifacts, HubError> {
		let names = tree
			.iter()
			.filter(|file| file.kind == "file")
			.map(|file| file.path.as_str())
			.collect::<BTreeSet<_>>();
		let config = self
			.fetch_revision_json(
				&model.id,
				&model.revision,
				"config.json",
				crate::artifact::MAX_MODEL_CONFIG_BYTES as usize,
			)
			.await?
			.ok_or_else(|| HubError::Incompatible("repository lacks config.json".to_string()))?;
		let tokenizer_config = if names.contains("tokenizer_config.json") {
			self.fetch_revision_json(
				&model.id,
				&model.revision,
				"tokenizer_config.json",
				16 << 20,
			)
			.await?
		} else {
			None
		};
		let generation_config = if names.contains("generation_config.json") {
			self.fetch_revision_json(
				&model.id,
				&model.revision,
				"generation_config.json",
				4 << 20,
			)
			.await?
		} else {
			None
		};
		let processor_config = if names.contains("processor_config.json") {
			self.fetch_revision_json(
				&model.id,
				&model.revision,
				"processor_config.json",
				crate::artifact::MAX_TOKENIZER_CONFIG_BYTES as usize,
			)
			.await?
		} else {
			None
		};
		if names.contains(crate::engine::tokenizer::LEGACY_CHAT_TEMPLATE_FILE)
			&& [
				"additional_chat_templates/default.jinja",
				"additional_chat_templates/tool_use.jinja",
				"chat_templates/default.jinja",
				"chat_templates/tool_use.jinja",
			]
			.into_iter()
			.any(|path| names.contains(path))
		{
			return Err(HubError::Incompatible(
				"chat_template.json conflicts with named chat template files".to_string(),
			));
		}
		let (legacy_chat_template, chat_template, tool_chat_template) =
			if processor_has_chat_template(processor_config.as_ref()) {
				(None, None, None)
			} else {
				self.lower_revision_chat_artifacts(model, &names).await?
			};
		let index = if names.contains("model.safetensors.index.json") {
			self.fetch_revision_json(
				&model.id,
				&model.revision,
				"model.safetensors.index.json",
				64 << 20,
			)
			.await?
		} else {
			None
		};
		Ok(RevisionArtifacts {
			config,
			tokenizer_config,
			processor_config,
			generation_config,
			legacy_chat_template,
			chat_template,
			tool_chat_template,
			index,
		})
	}

	async fn lower_revision_chat_artifacts(
		&self,
		model: &HubModel,
		names: &BTreeSet<&str>,
	) -> Result<(Option<serde_json::Value>, Option<String>, Option<String>), HubError> {
		const CURRENT_DEFAULT: &str = "additional_chat_templates/default.jinja";
		const CURRENT_TOOL: &str = "additional_chat_templates/tool_use.jinja";
		const LEGACY_DEFAULT: &str = "chat_templates/default.jinja";
		const LEGACY_TOOL: &str = "chat_templates/tool_use.jinja";
		let named_defaults = [CURRENT_DEFAULT, LEGACY_DEFAULT]
			.into_iter()
			.filter(|path| names.contains(path))
			.collect::<Vec<_>>();
		let named_tools = [CURRENT_TOOL, LEGACY_TOOL]
			.into_iter()
			.filter(|path| names.contains(path))
			.collect::<Vec<_>>();
		if names.contains(crate::engine::tokenizer::LEGACY_CHAT_TEMPLATE_FILE) {
			if !named_defaults.is_empty() || !named_tools.is_empty() {
				return Err(HubError::Incompatible(
					"chat_template.json conflicts with named chat template files".to_string(),
				));
			}
			let legacy = self
				.fetch_revision_json(
					&model.id,
					&model.revision,
					crate::engine::tokenizer::LEGACY_CHAT_TEMPLATE_FILE,
					crate::artifact::MAX_TOKENIZER_CONFIG_BYTES as usize,
				)
				.await?;
			return Ok((legacy, None, None));
		}
		let default_path =
			select_external_template(names, "chat_template.jinja", &named_defaults, "default")?;
		let tool_path = select_external_template(
			names,
			"chat_template_tool_use.jinja",
			&named_tools,
			"tool-use",
		)?;
		let chat_template = match default_path {
			Some(path) => {
				self.fetch_revision_text(
					&model.id,
					&model.revision,
					path,
					crate::engine::tokenizer::MAX_CHAT_TEMPLATE_BYTES,
				)
				.await?
			}
			None => None,
		};
		let tool_chat_template = match tool_path {
			Some(path) => {
				self.fetch_revision_text(
					&model.id,
					&model.revision,
					path,
					crate::engine::tokenizer::MAX_CHAT_TEMPLATE_BYTES,
				)
				.await?
			}
			None => None,
		};
		Ok((None, chat_template, tool_chat_template))
	}

	async fn fetch_revision_json(
		&self,
		id: &HubModelId,
		revision: &ResolvedRevision,
		path: &str,
		limit: usize,
	) -> Result<Option<serde_json::Value>, HubError> {
		let Some(bytes) = self.fetch_revision_bytes(id, revision, path, limit).await? else {
			return Ok(None);
		};
		serde_json::from_slice(&bytes)
			.map(Some)
			.map_err(|error| HubError::Protocol(format!("invalid {path} JSON: {error}")))
	}

	async fn fetch_revision_text(
		&self,
		id: &HubModelId,
		revision: &ResolvedRevision,
		path: &str,
		limit: usize,
	) -> Result<Option<String>, HubError> {
		let Some(bytes) = self.fetch_revision_bytes(id, revision, path, limit).await? else {
			return Ok(None);
		};
		String::from_utf8(bytes)
			.map(Some)
			.map_err(|error| HubError::Protocol(format!("{path} is not UTF-8: {error}")))
	}

	async fn fetch_revision_bytes(
		&self,
		id: &HubModelId,
		revision: &ResolvedRevision,
		path: &str,
		limit: usize,
	) -> Result<Option<Vec<u8>>, HubError> {
		let url = self.resolve_url(id, revision, path)?;
		let response = self
			.client
			.get(url)
			.header(header::ACCEPT_ENCODING, "identity")
			.timeout(self.config.request_timeout())
			.send()
			.await
			.map_err(request_error)?;
		if response.status() == StatusCode::NOT_FOUND {
			return Ok(None);
		}
		if !response.status().is_success() {
			return Err(self.http_error(response).await);
		}
		read_body(response, limit, path).await.map(Some)
	}

	/// Download a complete plan into an empty Emelex-owned staging directory.
	///
	/// Returns validated manifest file records in plan order.
	///
	/// # Errors
	///
	/// Returns on unsafe staging, I/O, retry exhaustion, size, or hash failure.
	pub async fn download(
		&self,
		plan: &DownloadPlan,
		staging: &Path,
		reporter: Option<&DownloadReporter>,
	) -> Result<Vec<ModelFile>, HubError> {
		let mut operation = DownloadOperationGuard::new(None);
		let result = self
			.download_with_controls(
				plan,
				staging,
				reporter,
				None,
				Some(operation.cancellation()),
			)
			.await;
		operation.finish();
		result
	}

	/// Download with a fallible observer and cooperative cancellation.
	///
	/// # Errors
	///
	/// Returns on unsafe staging, observer failure/cancellation, I/O, retry
	/// exhaustion, size, or hash failure.
	pub async fn download_controlled(
		&self,
		plan: &DownloadPlan,
		staging: &Path,
		observer: Option<&DownloadObserver>,
		cancellation: Option<&DownloadCancellation>,
	) -> Result<Vec<ModelFile>, HubError> {
		let mut operation = DownloadOperationGuard::new(cancellation);
		let result = self
			.download_with_controls(
				plan,
				staging,
				None,
				observer,
				Some(operation.cancellation()),
			)
			.await;
		operation.finish();
		result
	}

	pub(crate) async fn download_with_controls(
		&self,
		plan: &DownloadPlan,
		staging: &Path,
		reporter: Option<&DownloadReporter>,
		observer: Option<&DownloadObserver>,
		cancellation: Option<&DownloadCancellation>,
	) -> Result<Vec<ModelFile>, HubError> {
		self.download_with_callbacks(
			plan,
			staging,
			DownloadCallbacks {
				reporter,
				observer,
				cancellation,
			},
		)
		.await
	}

	async fn download_with_callbacks(
		&self,
		plan: &DownloadPlan,
		staging: &Path,
		callbacks: DownloadCallbacks<'_>,
	) -> Result<Vec<ModelFile>, HubError> {
		validate_download_plan(plan)?;
		check_cancelled(callbacks.cancellation)?;
		let staging_path = staging.to_path_buf();
		tokio::task::spawn_blocking(move || validate_download_staging(&staging_path))
			.await
			.map_err(blocking_hub_task_error)??;
		emit(
			callbacks,
			&DownloadEvent::TransferStarted {
				files: plan.files().len(),
				total: plan.total_bytes(),
			},
		)?;
		let mut installed = Vec::with_capacity(plan.files().len());
		for file in plan.files() {
			check_cancelled(callbacks.cancellation)?;
			installed.push(self.download_file(plan, file, staging, callbacks).await?);
		}
		emit(
			callbacks,
			&DownloadEvent::TransferCompleted {
				files: plan.files().len(),
				total: plan.total_bytes(),
			},
		)?;
		Ok(installed)
	}

	#[allow(
		clippy::too_many_lines,
		reason = "bounded transfer loop keeps retry, identity, cancellation, and publication state together"
	)]
	async fn download_file(
		&self,
		plan: &DownloadPlan,
		remote: &RemoteFile,
		staging: &Path,
		callbacks: DownloadCallbacks<'_>,
	) -> Result<ModelFile, HubError> {
		let local_path = local_runtime_path(&remote.path);
		let destination = staging.join(local_path);
		let part = staging.join(format!("{local_path}.part"));
		let mut part_identity = None;
		let mut attempt = 0_usize;
		loop {
			check_cancelled(callbacks.cancellation)?;
			attempt += 1;
			let result = self
				.download_attempt(plan, remote, &part, &mut part_identity, callbacks)
				.await;
			match result {
				Ok(()) => break,
				Err(error) if error.transient() && attempt <= self.config.retries => {
					emit(
						callbacks,
						&DownloadEvent::Retrying {
							path: remote.path.clone(),
							attempt,
							reason: error.to_string(),
						},
					)?;
					let shift = u32::try_from(attempt.saturating_sub(1))
						.unwrap_or(31)
						.min(5);
					let delay = tokio::time::sleep(Duration::from_secs(1_u64 << shift));
					tokio::pin!(delay);
					if let Some(cancellation) = callbacks.cancellation {
						tokio::select! {
							biased;
							() = cancelled(cancellation) => return Err(HubError::Cancelled),
							() = &mut delay => {}
						}
					} else {
						delay.await;
					}
				}
				Err(error) => return Err(error),
			}
		}
		let (size, digest, snapshot) =
			hash_file(part.clone(), callbacks.cancellation.cloned()).await?;
		if size != remote.size {
			return Err(HubError::Size {
				path: remote.path.clone(),
				expected: remote.size,
				actual: size,
			});
		}
		if remote
			.expected_sha256
			.as_ref()
			.is_some_and(|expected| expected != &digest)
		{
			return Err(HubError::Hash {
				path: remote.path.clone(),
				expected: remote.expected_sha256.clone().unwrap_or_default(),
				actual: digest,
			});
		}
		emit(
			callbacks,
			&DownloadEvent::FileVerified {
				path: remote.path.clone(),
				sha256: digest.clone(),
			},
		)?;
		check_cancelled(callbacks.cancellation)?;
		let publish_part = part.clone();
		let publish_destination = destination.clone();
		let publish_staging = staging.to_path_buf();
		let cancellation = callbacks.cancellation.cloned();
		#[cfg(test)]
		let publish_gate = self.publish_gate.clone();
		tokio::task::spawn_blocking(move || {
			#[cfg(test)]
			if let Some(gate) = &publish_gate {
				gate.entered.store(true, Ordering::Release);
				while !gate.release.load(Ordering::Acquire) {
					std::thread::sleep(Duration::from_millis(1));
				}
			}
			let result = (|| {
				check_cancelled(cancellation.as_ref())?;
				validate_file_snapshot(&publish_part, &snapshot)?;
				check_cancelled(cancellation.as_ref())?;
				rename_exclusive(&publish_part, &publish_destination)?;
				sync_directory(&publish_staging)
			})();
			#[cfg(test)]
			if let Some(gate) = publish_gate {
				gate.completed.store(true, Ordering::Release);
			}
			result
		})
		.await
		.map_err(blocking_hub_task_error)??;
		ModelFile::new(local_path.to_string(), remote.size, digest)
			.map_err(|error| HubError::Protocol(error.to_string()))
	}

	#[allow(
		clippy::too_many_lines,
		reason = "range validation, bounded transfer, persistence, and progress form one attempt"
	)]
	async fn download_attempt(
		&self,
		plan: &DownloadPlan,
		remote: &RemoteFile,
		part: &Path,
		part_identity: &mut Option<FileIdentity>,
		callbacks: DownloadCallbacks<'_>,
	) -> Result<(), HubError> {
		check_cancelled(callbacks.cancellation)?;
		let inspect_part = part.to_path_buf();
		let expected_identity = *part_identity;
		let (existing, identity) =
			tokio::task::spawn_blocking(move || inspect_partial(&inspect_part, expected_identity))
				.await
				.map_err(blocking_hub_task_error)??;
		*part_identity = identity;
		if existing > remote.size {
			return Err(HubError::Size {
				path: remote.path.clone(),
				expected: remote.size,
				actual: existing,
			});
		}
		emit(
			callbacks,
			&DownloadEvent::FileStarted {
				path: remote.path.clone(),
				resumed: existing,
				total: remote.size,
			},
		)?;
		if existing == remote.size {
			return Ok(());
		}
		let url = self.resolve_url(&plan.model().id, &plan.model().revision, remote.path())?;
		let mut request = self
			.client
			.get(url)
			.header(header::ACCEPT_ENCODING, "identity");
		if existing > 0 {
			request = request.header(header::RANGE, format!("bytes={existing}-"));
		}
		let response = self
			.send_download(request, remote.path(), callbacks.cancellation)
			.await?;
		let resumed = response.status() == StatusCode::PARTIAL_CONTENT;
		if resumed {
			validate_content_range(
				response.headers(),
				if existing > 0 { existing } else { 0 },
				remote.size,
			)?;
		} else if existing > 0 && response.status() != StatusCode::OK {
			return Err(HubError::Protocol(format!(
				"Hub ignored range request with unexpected status {}",
				response.status()
			)));
		}
		check_cancelled(callbacks.cancellation)?;
		let mut persisted = if existing > 0 && resumed { existing } else { 0 };
		let open_path = part.to_path_buf();
		let append = persisted > 0;
		let expected_identity = *part_identity;
		let cancellation = callbacks.cancellation.cloned();
		let (file, identity) = tokio::task::spawn_blocking(move || {
			check_cancelled(cancellation.as_ref())?;
			open_partial(&open_path, append, expected_identity)
		})
		.await
		.map_err(blocking_hub_task_error)??;
		*part_identity = Some(identity);
		let mut file = tokio::fs::File::from_std(file);
		let mut body = response.bytes_stream();
		loop {
			let next = tokio::time::timeout(self.config.request_timeout(), body.next());
			tokio::pin!(next);
			let next = if let Some(cancellation) = callbacks.cancellation {
				tokio::select! {
					biased;
					() = cancelled(cancellation) => return Err(HubError::Cancelled),
					result = &mut next => result,
				}
			} else {
				next.await
			}
			.map_err(|_| HubError::DownloadIdleTimeout {
				path: remote.path.clone(),
				seconds: self.config.request_timeout_seconds,
			})?;
			let Some(chunk) = next else {
				break;
			};
			check_cancelled(callbacks.cancellation)?;
			let chunk = chunk.map_err(|error| {
				download_request_error(error, remote.path(), self.config.request_timeout_seconds)
			})?;
			persisted = persisted
				.checked_add(u64::try_from(chunk.len()).map_err(|_| {
					HubError::Protocol("download chunk length overflow".to_string())
				})?)
				.ok_or_else(|| HubError::Protocol("download byte count overflow".to_string()))?;
			if persisted > remote.size {
				return Err(HubError::Size {
					path: remote.path.clone(),
					expected: remote.size,
					actual: persisted,
				});
			}
			file.write_all(&chunk)
				.await
				.map_err(|source| HubError::Io {
					path: part.to_path_buf(),
					source,
				})?;
			// Tokio files may retain an async write until flush. Progress is
			// fallible: an observer can stop the attempt immediately, so make
			// the reported prefix available for resumable inspection first.
			file.flush().await.map_err(|source| HubError::Io {
				path: part.to_path_buf(),
				source,
			})?;
			emit(
				callbacks,
				&DownloadEvent::Progress {
					path: remote.path.clone(),
					received: persisted,
					total: remote.size,
				},
			)?;
		}
		file.flush().await.map_err(|source| HubError::Io {
			path: part.to_path_buf(),
			source,
		})?;
		file.sync_all().await.map_err(|source| HubError::Io {
			path: part.to_path_buf(),
			source,
		})?;
		if persisted < remote.size {
			return Err(HubError::Size {
				path: remote.path.clone(),
				expected: remote.size,
				actual: persisted,
			});
		}
		Ok(())
	}

	async fn send(&self, request: reqwest::RequestBuilder) -> Result<reqwest::Response, HubError> {
		let response = request
			.timeout(self.config.request_timeout())
			.send()
			.await
			.map_err(request_error)?;
		self.validate_response(response).await
	}

	async fn send_download(
		&self,
		request: reqwest::RequestBuilder,
		path: &str,
		cancellation: Option<&DownloadCancellation>,
	) -> Result<reqwest::Response, HubError> {
		let send = tokio::time::timeout(self.config.request_timeout(), request.send());
		tokio::pin!(send);
		let response = if let Some(cancellation) = cancellation {
			tokio::select! {
				biased;
				() = cancelled(cancellation) => return Err(HubError::Cancelled),
				response = &mut send => response,
			}
		} else {
			send.await
		}
		.map_err(|_| HubError::DownloadIdleTimeout {
			path: path.to_string(),
			seconds: self.config.request_timeout_seconds,
		})?
		.map_err(|error| {
			download_request_error(error, path, self.config.request_timeout_seconds)
		})?;
		let validate = tokio::time::timeout(
			self.config.request_timeout(),
			self.validate_response(response),
		);
		tokio::pin!(validate);
		let validated = if let Some(cancellation) = cancellation {
			tokio::select! {
				biased;
				() = cancelled(cancellation) => return Err(HubError::Cancelled),
				response = &mut validate => response,
			}
		} else {
			validate.await
		}
		.map_err(|_| HubError::DownloadIdleTimeout {
			path: path.to_string(),
			seconds: self.config.request_timeout_seconds,
		})?;
		validated.map_err(|error| {
			download_response_error(error, path, self.config.request_timeout_seconds)
		})
	}

	async fn validate_response(
		&self,
		response: reqwest::Response,
	) -> Result<reqwest::Response, HubError> {
		let status = response.status();
		if status.is_success() {
			return Ok(response);
		}
		Err(self.http_error(response).await)
	}

	async fn http_error(&self, response: reqwest::Response) -> HubError {
		let status = response.status();
		let potentially_credentialed = self.credential_scope.is_some()
			|| response.url().origin() != self.endpoint.origin()
			|| response.url().query().is_some();
		let body = if potentially_credentialed {
			"response body suppressed by credential-safety policy".to_string()
		} else {
			match read_error_body(response, MAX_ERROR_BODY_BYTES).await {
				ErrorBodyRead::Text(body) => body,
				ErrorBodyRead::TimedOut(error) => return request_error(error),
			}
		};
		HubError::Http { status, body }
	}

	fn api_url(&self, segments: &[&str]) -> Result<Url, HubError> {
		let mut url = self.endpoint.clone();
		{
			let mut path = url
				.path_segments_mut()
				.map_err(|()| HubError::Url("Hub endpoint cannot be a base URL".to_string()))?;
			path.clear();
			for segment in segments {
				path.push(segment);
			}
		}
		Ok(url)
	}

	fn resolve_url(
		&self,
		id: &HubModelId,
		revision: &ResolvedRevision,
		path: &str,
	) -> Result<Url, HubError> {
		let components = Path::new(path)
			.components()
			.map(|component| component.as_os_str().to_str())
			.collect::<Option<Vec<_>>>()
			.ok_or_else(|| HubError::Protocol("non-UTF-8 remote path".to_string()))?;
		let mut segments = id.as_str().split('/').collect::<Vec<_>>();
		segments.extend(["resolve", revision.as_str()]);
		segments.extend(components);
		self.api_url(&segments)
	}
}

async fn decode_response<T>(
	response: reqwest::Response,
	limit: usize,
	label: &str,
) -> Result<T, HubError>
where
	T: DeserializeOwned,
{
	let bytes = read_body(response, limit, label).await?;
	serde_json::from_slice(&bytes)
		.map_err(|error| HubError::Protocol(format!("invalid {label} JSON: {error}")))
}

async fn read_body(
	response: reqwest::Response,
	limit: usize,
	label: &str,
) -> Result<Vec<u8>, HubError> {
	let mut body = Vec::new();
	let mut stream = response.bytes_stream();
	while let Some(chunk) = stream.next().await {
		let chunk = chunk.map_err(request_error)?;
		let next = body
			.len()
			.checked_add(chunk.len())
			.ok_or_else(|| HubError::Protocol(format!("{label} response size overflow")))?;
		if next > limit {
			return Err(HubError::Protocol(format!(
				"{label} response exceeds {limit} byte limit"
			)));
		}
		body.extend_from_slice(&chunk);
	}
	Ok(body)
}

enum ErrorBodyRead {
	Text(String),
	TimedOut(reqwest::Error),
}

async fn read_error_body(response: reqwest::Response, limit: usize) -> ErrorBodyRead {
	let mut body = Vec::with_capacity(limit.min(8 << 10));
	let mut truncated = false;
	let mut stream = response.bytes_stream();
	while let Some(chunk) = stream.next().await {
		match chunk {
			Ok(chunk) => {
				let remaining = limit.saturating_sub(body.len());
				if chunk.len() > remaining {
					body.extend_from_slice(&chunk[..remaining]);
					truncated = true;
					break;
				}
				body.extend_from_slice(&chunk);
			}
			Err(error) if error.is_timeout() => return ErrorBodyRead::TimedOut(error),
			Err(error) => {
				return ErrorBodyRead::Text(format!(
					"cannot read error body: {}",
					error.without_url()
				));
			}
		}
	}
	let mut rendered = String::from_utf8_lossy(&body).into_owned();
	if truncated {
		rendered.push_str(" [truncated]");
	}
	ErrorBodyRead::Text(rendered)
}

fn request_error(error: reqwest::Error) -> HubError {
	HubError::Request(error.without_url())
}

fn download_request_error(error: reqwest::Error, path: &str, seconds: u64) -> HubError {
	if error.is_timeout() {
		HubError::DownloadIdleTimeout {
			path: path.to_string(),
			seconds,
		}
	} else {
		request_error(error)
	}
}

fn download_response_error(error: HubError, path: &str, seconds: u64) -> HubError {
	match error {
		HubError::Request(source) => download_request_error(source, path, seconds),
		other => other,
	}
}

fn validate_content_range(
	headers: &header::HeaderMap,
	expected_start: u64,
	expected_total: u64,
) -> Result<(), HubError> {
	let value = headers
		.get(header::CONTENT_RANGE)
		.ok_or_else(|| HubError::Protocol("partial response lacks Content-Range".to_string()))?
		.to_str()
		.map_err(|error| HubError::Protocol(format!("invalid Content-Range header: {error}")))?;
	let range = value
		.strip_prefix("bytes ")
		.ok_or_else(|| HubError::Protocol(format!("invalid Content-Range {value:?}")))?;
	let (bounds, total) = range
		.split_once('/')
		.ok_or_else(|| HubError::Protocol(format!("invalid Content-Range {value:?}")))?;
	let (start, end) = bounds
		.split_once('-')
		.ok_or_else(|| HubError::Protocol(format!("invalid Content-Range {value:?}")))?;
	let start = parse_header_u64(start, "Content-Range start")?;
	let end = parse_header_u64(end, "Content-Range end")?;
	let total = parse_header_u64(total, "Content-Range total")?;
	if start != expected_start || total != expected_total || end < start || end >= total {
		return Err(HubError::Protocol(format!(
			"Content-Range {value:?} does not match requested bytes {expected_start}-/{expected_total}"
		)));
	}
	if let Some(length) = headers.get(header::CONTENT_LENGTH) {
		let length = length
			.to_str()
			.map_err(|error| HubError::Protocol(format!("invalid Content-Length: {error}")))?;
		let length = parse_header_u64(length, "Content-Length")?;
		let expected_length = end
			.checked_sub(start)
			.and_then(|value| value.checked_add(1))
			.ok_or_else(|| HubError::Protocol("Content-Range length overflow".to_string()))?;
		if length != expected_length {
			return Err(HubError::Protocol(format!(
				"Content-Length {length} does not match Content-Range length {expected_length}"
			)));
		}
	}
	Ok(())
}

fn parse_header_u64(value: &str, field: &str) -> Result<u64, HubError> {
	if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
		return Err(HubError::Protocol(format!("{field} is not an integer")));
	}
	value
		.parse()
		.map_err(|error| HubError::Protocol(format!("invalid {field}: {error}")))
}

#[derive(Debug, Deserialize)]
struct ModelWire {
	id: String,
	#[serde(default)]
	private: bool,
	#[serde(default)]
	gated: serde_json::Value,
	sha: Option<String>,
	#[serde(default)]
	downloads: u64,
	#[serde(default)]
	likes: u64,
	#[serde(default)]
	tags: Vec<String>,
	#[serde(default)]
	library_name: Option<String>,
	#[serde(default)]
	pipeline_tag: Option<String>,
	#[serde(default)]
	siblings: Vec<SiblingWire>,
}

#[derive(Debug, Deserialize)]
struct SiblingWire {
	rfilename: String,
}

#[derive(Debug, Deserialize)]
struct TreeWire {
	#[serde(rename = "type")]
	kind: String,
	path: String,
	#[serde(default)]
	size: u64,
	#[serde(default)]
	lfs: Option<LfsWire>,
	#[serde(default, rename = "securityFileStatus")]
	security: Option<SecurityWire>,
}

#[derive(Debug, Deserialize)]
struct LfsWire {
	oid: String,
}

#[derive(Debug, Deserialize)]
struct SecurityWire {
	status: String,
}

struct RevisionArtifacts {
	config: serde_json::Value,
	tokenizer_config: Option<serde_json::Value>,
	processor_config: Option<serde_json::Value>,
	generation_config: Option<serde_json::Value>,
	legacy_chat_template: Option<serde_json::Value>,
	chat_template: Option<String>,
	tool_chat_template: Option<String>,
	index: Option<serde_json::Value>,
}

fn model_from_wire(wire: &ModelWire, authenticated: bool) -> Result<HubModel, HubError> {
	if !authenticated && (wire.private || gated(&wire.gated)) {
		return Err(HubError::NotPublic(wire.id.clone()));
	}
	let id = HubModelId::parse(wire.id.clone())
		.map_err(|error| HubError::Protocol(error.to_string()))?;
	let revision = wire
		.sha
		.as_deref()
		.ok_or_else(|| HubError::Protocol("Hub model has no immutable revision".to_string()))
		.and_then(|sha| {
			ResolvedRevision::parse(sha).map_err(|error| HubError::Protocol(error.to_string()))
		})?;
	let files = wire
		.siblings
		.iter()
		.map(|sibling| sibling.rfilename.clone())
		.collect::<Vec<_>>();
	let traits = traits_from_wire(wire, &files);
	let license = wire
		.tags
		.iter()
		.find_map(|tag| tag.strip_prefix("license:").map(str::to_string));
	Ok(HubModel {
		id,
		revision,
		downloads: wire.downloads,
		likes: wire.likes,
		tags: wire.tags.clone(),
		library: wire.library_name.clone(),
		license,
		traits,
		quantization: HubQuantization::Unknown,
		files,
		compatible: false,
		diagnostics: Vec::new(),
		fit: None,
		download_bytes: None,
	})
}

#[allow(
	clippy::too_many_lines,
	reason = "exact checkpoint selection is one fail-closed ambiguity gate"
)]
fn exact_runtime_plan(
	tree: &[TreeWire],
	index: Option<&serde_json::Value>,
	processor_has_template: bool,
) -> Result<Vec<RemoteFile>, HubError> {
	let mut root_files = BTreeMap::new();
	for file in tree
		.iter()
		.filter(|file| file.kind == "file" && Path::new(&file.path).components().count() == 1)
	{
		if root_files.insert(file.path.as_str(), file).is_some() {
			return Err(HubError::Protocol(format!(
				"duplicate root file in Hub tree: {:?}",
				file.path
			)));
		}
	}
	if root_files.contains_key("adapter_config.json")
		|| root_files.contains_key("adapter_model.safetensors")
	{
		return Err(HubError::Incompatible(
			"adapter-only or mixed adapter repositories are not standalone checkpoints".to_string(),
		));
	}
	let alternate_index = root_files
		.keys()
		.find(|name| {
			name.ends_with(".safetensors.index.json") && **name != "model.safetensors.index.json"
		})
		.copied();
	if let Some(path) = alternate_index {
		return Err(HubError::Incompatible(format!(
			"variant safetensors index {path:?} makes checkpoint selection ambiguous"
		)));
	}
	let all_weights = root_files
		.keys()
		.filter(|name| name.ends_with(".safetensors"))
		.map(|name| (*name).to_string())
		.collect::<BTreeSet<_>>();
	let selected_weights = if let Some(index) = index {
		let weight_map = index
			.get("weight_map")
			.and_then(serde_json::Value::as_object)
			.filter(|map| !map.is_empty())
			.ok_or_else(|| {
				HubError::Incompatible(
					"model.safetensors.index.json lacks a non-empty weight_map".to_string(),
				)
			})?;
		if weight_map.len() > 1_000_000 {
			return Err(HubError::Protocol(
				"safetensors index exceeds 1000000 tensors".to_string(),
			));
		}
		let mut selected = BTreeSet::new();
		for shard in weight_map.values() {
			let shard = shard.as_str().ok_or_else(|| {
				HubError::Protocol("safetensors weight_map values must be strings".to_string())
			})?;
			if !crate::model::layout::safe_relative_path(shard)
				|| Path::new(shard).components().count() != 1
				|| !shard.ends_with(".safetensors")
			{
				return Err(HubError::Protocol(format!(
					"unsafe indexed shard path {shard:?}"
				)));
			}
			if !root_files.contains_key(shard) {
				return Err(HubError::Incompatible(format!(
					"index references missing shard {shard:?}"
				)));
			}
			selected.insert(shard.to_string());
		}
		selected
	} else if all_weights == BTreeSet::from(["model.safetensors".to_string()]) {
		all_weights.clone()
	} else {
		return Err(HubError::Incompatible(
			"checkpoint without an index must contain only model.safetensors".to_string(),
		));
	};
	if all_weights != selected_weights {
		let extras = all_weights
			.difference(&selected_weights)
			.cloned()
			.collect::<Vec<_>>();
		return Err(HubError::Incompatible(format!(
			"unindexed safetensors variants or adapters are ambiguous: {extras:?}"
		)));
	}
	let mut selected_names = runtime_metadata_names(&root_files);
	selected_names.extend(selected_weights);
	if index.is_some() {
		selected_names.insert("model.safetensors.index.json".to_string());
	}
	if !selected_names.contains("config.json") || !selected_names.contains("tokenizer.json") {
		return Err(HubError::Incompatible(
			"repository lacks config.json or tokenizer.json".to_string(),
		));
	}
	let mut files = selected_names
		.into_iter()
		.map(|name| {
			let wire = root_files.get(name.as_str()).ok_or_else(|| {
				HubError::Protocol(format!(
					"selected runtime file {name:?} is absent from tree"
				))
			})?;
			remote_file(wire)
		})
		.collect::<Result<Vec<_>, _>>()?;
	if processor_has_template {
		files.retain(|file| {
			!matches!(
				file.path.as_str(),
				"chat_template.json" | "chat_template.jinja" | "chat_template_tool_use.jinja"
			)
		});
		files.sort_by(|left, right| left.path.cmp(&right.path));
		validate_download_files(&files)?;
		return Ok(files);
	}
	let mut nested = BTreeMap::new();
	for file in tree.iter().filter(|file| {
		file.kind == "file"
			&& matches!(
				file.path.as_str(),
				"additional_chat_templates/default.jinja"
					| "additional_chat_templates/tool_use.jinja"
					| "chat_templates/default.jinja"
					| "chat_templates/tool_use.jinja"
					| "chat_template.json"
			)
	}) {
		if nested.insert(file.path.as_str(), file).is_some() {
			return Err(HubError::Protocol(format!(
				"duplicate chat template file in Hub tree: {:?}",
				file.path
			)));
		}
	}
	if let Some(legacy) = nested.get("chat_template.json") {
		if nested.keys().any(|path| *path != "chat_template.json") {
			return Err(HubError::Incompatible(
				"chat_template.json conflicts with named chat template files".to_string(),
			));
		}
		files.retain(|file| {
			!matches!(
				file.path.as_str(),
				"chat_template.jinja" | "chat_template_tool_use.jinja"
			)
		});
		debug_assert!(files.iter().any(|file| file.path == legacy.path));
	} else {
		let defaults = [
			"additional_chat_templates/default.jinja",
			"chat_templates/default.jinja",
		]
		.into_iter()
		.filter_map(|path| nested.get(path))
		.collect::<Vec<_>>();
		if root_files.contains_key("chat_template.jinja") && !defaults.is_empty() {
			return Err(HubError::Incompatible(
				"root and named default chat templates map to the same runtime file".to_string(),
			));
		}
		match defaults.as_slice() {
			[] => {}
			[default] => files.push(remote_file(default)?),
			_ => {
				return Err(HubError::Incompatible(
					"multiple named default chat templates are present".to_string(),
				));
			}
		}
		let tools = [
			"additional_chat_templates/tool_use.jinja",
			"chat_templates/tool_use.jinja",
		]
		.into_iter()
		.filter_map(|path| nested.get(path))
		.collect::<Vec<_>>();
		if root_files.contains_key("chat_template_tool_use.jinja") && !tools.is_empty() {
			return Err(HubError::Incompatible(
				"root and named tool-use chat templates map to the same runtime file".to_string(),
			));
		}
		match tools.as_slice() {
			[] => {}
			[tool_use] => files.push(remote_file(tool_use)?),
			_ => {
				return Err(HubError::Incompatible(
					"multiple named tool-use chat templates are present".to_string(),
				));
			}
		}
	}
	files.sort_by(|left, right| left.path.cmp(&right.path));
	validate_download_files(&files)?;
	Ok(files)
}

fn runtime_metadata_names(tree: &BTreeMap<&str, &TreeWire>) -> BTreeSet<String> {
	tree.keys()
		.filter(|name| runtime_metadata_file_name(name))
		.map(|name| (*name).to_string())
		.collect()
}

pub(crate) fn runtime_metadata_file_name(name: &str) -> bool {
	const NAMES: &[&str] = &[
		"added_tokens.json",
		"chat_template.json",
		"chat_template.jinja",
		"chat_template_tool_use.jinja",
		"config.json",
		"generation_config.json",
		"merges.txt",
		"preprocessor_config.json",
		"processor_config.json",
		"special_tokens_map.json",
		"tokenizer.json",
		"tokenizer.model",
		"tokenizer_config.json",
		"video_preprocessor_config.json",
		"vocab.json",
		"vocab.txt",
	];
	NAMES.contains(&name)
}

#[allow(
	clippy::too_many_lines,
	reason = "capabilities, evidence, confidence, compatibility, and fit must update atomically"
)]
fn enrich_remote_model(
	model: &mut HubModel,
	wire: &ModelWire,
	artifacts: &RevisionArtifacts,
	files: &[RemoteFile],
	fit_profile: Option<(WorkloadProfile, u64)>,
) -> Result<(), HubError> {
	let config = artifacts
		.config
		.as_object()
		.ok_or_else(|| HubError::Incompatible("config.json must contain an object".to_string()))?;
	let model_type = config.get("model_type").and_then(serde_json::Value::as_str);
	let mut diagnostics = Vec::new();
	let supported = model_type.is_some_and(crate::model::supported_model_type);
	if !supported {
		diagnostics.push(model_type.map_or_else(
			|| "config.json lacks string model_type".to_string(),
			|value| format!("unsupported model_type {value:?}"),
		));
	} else if let Err(error) =
		crate::engine::models::config::validate_checkpoint_config(&artifacts.config)
	{
		diagnostics.push(format!("unsupported model configuration: {error}"));
	}
	let quantization = match hub_quantization_from_config(&artifacts.config) {
		Ok(quantization) => quantization,
		Err(error) => {
			diagnostics.push(format!("unsupported quantization: {error}"));
			HubQuantization::Unknown
		}
	};
	let statically_compatible = diagnostics.is_empty();
	let download_bytes = validate_download_files(files)?;
	let weights_bytes = files
		.iter()
		.filter(|file| file.path.ends_with(".safetensors"))
		.try_fold(0_u64, |total, file| total.checked_add(file.size))
		.ok_or_else(|| HubError::Protocol("weight byte count overflow".to_string()))?;
	let text = artifacts
		.config
		.get("text_config")
		.unwrap_or(&artifacts.config);
	let fit = fit_profile.and_then(|(workload, metal_budget_bytes)| {
		match crate::model::estimate_fit_from_config(
			text,
			model_type,
			weights_bytes,
			workload,
			metal_budget_bytes,
		) {
			Ok(fit) => {
				if !fit.fits {
					diagnostics.push(format!(
						"estimated residency {} exceeds Metal budget {metal_budget_bytes}",
						fit.required_bytes
					));
				}
				Some(fit)
			}
			Err(reason) => {
				diagnostics.push(reason);
				None
			}
		}
	});
	let templates = resolved_chat_templates(artifacts)?;
	let generation_defaults = generation_defaults_from_value(artifacts.generation_config.as_ref())?;
	let max_context_tokens = declared_remote_max_context(text);
	let mut traits = traits_from_wire(wire, &model.files);
	traits.mlx = statically_compatible;
	traits.sizing = Some(ModelSizing {
		weights_bytes: Some(weights_bytes),
		estimated_residency_bytes: fit.as_ref().map(|report| report.required_bytes),
		evaluated_context_tokens: fit_profile.map(|(workload, _)| workload.context_tokens()),
		max_context_tokens,
	});
	traits.generation_defaults = generation_defaults;
	if traits.mlx {
		traits.input.insert(Modality::Text);
		traits.output.insert(Modality::Text);
		traits.tasks.insert(Task::TextGeneration);
		for key in ["input:text", "output:text"] {
			traits
				.confidence
				.insert(key.to_string(), TraitConfidence::Inferred);
		}
		traits
			.confidence
			.insert("acceleration:mlx".to_string(), TraitConfidence::Inferred);
		traits.confidence.insert(
			"task:text_generation".to_string(),
			TraitConfidence::Inferred,
		);
		traits.evidence.push(TraitEvidence {
			trait_key: "input:text".to_string(),
			source: EvidenceSource::Config,
			detail: "supported autoregressive architecture consumes tokenizer text".to_string(),
		});
		traits.evidence.push(TraitEvidence {
			trait_key: "output:text".to_string(),
			source: EvidenceSource::Config,
			detail: "supported autoregressive architecture decodes tokenizer text".to_string(),
		});
		traits.evidence.push(TraitEvidence {
			trait_key: "acceleration:mlx".to_string(),
			source: EvidenceSource::RepositoryTree,
			detail: "supported exact-revision config and unambiguous indexed shard plan"
				.to_string(),
		});
		traits.evidence.push(TraitEvidence {
			trait_key: "task:text_generation".to_string(),
			source: EvidenceSource::Config,
			detail: "supported autoregressive architecture at the exact revision".to_string(),
		});
	}
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
	if let Some(templates) = templates {
		let special = |key: &str| {
			artifacts
				.tokenizer_config
				.as_ref()
				.and_then(|config| config.get(key))
				.and_then(|value| match value {
					serde_json::Value::String(value) => Some(value.as_str()),
					serde_json::Value::Object(value) => {
						value.get("content").and_then(serde_json::Value::as_str)
					}
					_ => None,
				})
				.unwrap_or_default()
		};
		let (capabilities, _tool_format) =
			crate::engine::tokenizer::resolve_chat_templates_capabilities(
				&templates,
				(special("bos_token"), special("eos_token")),
			)
			.map_err(|error| {
				HubError::Incompatible(format!("chat template cannot be compiled safely: {error}"))
			})?;
		traits.tasks.insert(Task::Chat);
		traits
			.confidence
			.insert("task:chat".to_string(), TraitConfidence::Inferred);
		traits.evidence.push(TraitEvidence {
			trait_key: "task:chat".to_string(),
			source: EvidenceSource::Tokenizer,
			detail: "exact-revision chat template completed a bounded baseline render".to_string(),
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
	if !traits.tasks.contains(&Task::Chat) {
		diagnostics.push(
			"no supported chat template in chat_template.jinja, chat_templates/default.jinja, \
			 tokenizer_config.json, or processor_config.json"
				.to_string(),
		);
	}
	let mtp_layers = text
		.get("num_nextn_predict_layers")
		.or_else(|| text.get("mtp_num_hidden_layers"))
		.and_then(serde_json::Value::as_u64)
		.unwrap_or(0);
	if mtp_layers > 0 {
		traits.mtp = MtpSupport::Advertised;
		traits.confidence.insert(
			"acceleration:mtp_advertised".to_string(),
			TraitConfidence::Advertised,
		);
		traits.evidence.push(TraitEvidence {
			trait_key: "acceleration:mtp_advertised".to_string(),
			source: EvidenceSource::Config,
			detail: format!("exact-revision config advertises {mtp_layers} MTP layer(s)"),
		});
	}
	model.compatible = diagnostics.is_empty()
		&& match fit_profile {
			Some(_) => fit.as_ref().is_some_and(|report| report.fits),
			None => true,
		};
	model.traits = traits;
	model.quantization = quantization;
	model.fit = fit;
	model.diagnostics = diagnostics;
	model.files = files.iter().map(|file| file.path.clone()).collect();
	model.download_bytes = Some(download_bytes);
	Ok(())
}

fn hub_quantization_from_config(
	config: &serde_json::Value,
) -> crate::engine::error::Result<HubQuantization> {
	let quantization = crate::engine::quant::Quantization::from_config(config)?;
	let Some(default) = quantization.default else {
		return Ok(HubQuantization::NotConfigured);
	};
	let bits = u8::try_from(default.bits).map_err(|_| {
		crate::engine::error::Error::Config(format!(
			"quantization bits {} cannot be represented in Hub metadata",
			default.bits
		))
	})?;
	let group_size = u16::try_from(default.group_size).map_err(|_| {
		crate::engine::error::Error::Config(format!(
			"quantization group size {} cannot be represented in Hub metadata",
			default.group_size
		))
	})?;
	let mode = match default.mode {
		crate::engine::ops::QuantMode::Affine => HubQuantizationMode::Affine,
		crate::engine::ops::QuantMode::Mxfp4 => HubQuantizationMode::Mxfp4,
		crate::engine::ops::QuantMode::Mxfp8 => HubQuantizationMode::Mxfp8,
		crate::engine::ops::QuantMode::Nvfp4 => HubQuantizationMode::Nvfp4,
	};
	HubQuantizationConfig::new(mode, bits, group_size, !quantization.per_layer.is_empty())
		.map(HubQuantization::Configured)
}

fn resolved_chat_templates(
	artifacts: &RevisionArtifacts,
) -> Result<Option<crate::engine::tokenizer::ChatTemplates>, HubError> {
	let processor_embedded = artifacts
		.processor_config
		.as_ref()
		.and_then(|value| value.get("chat_template"))
		.unwrap_or(&serde_json::Value::Null);
	let tokenizer_embedded = artifacts
		.tokenizer_config
		.as_ref()
		.and_then(|value| value.get("chat_template"))
		.unwrap_or(&serde_json::Value::Null);
	crate::engine::tokenizer::resolve_chat_template_artifacts(
		processor_embedded,
		artifacts.legacy_chat_template.as_ref(),
		artifacts.chat_template.clone(),
		artifacts.tool_chat_template.clone(),
		tokenizer_embedded,
	)
	.map_err(|error| HubError::Incompatible(format!("invalid chat template artifacts: {error}")))
}

fn processor_has_chat_template(config: Option<&serde_json::Value>) -> bool {
	config
		.and_then(|value| value.get("chat_template"))
		.is_some_and(|template| !template.is_null())
}

fn select_external_template<'a>(
	names: &BTreeSet<&str>,
	root: &'a str,
	named: &[&'a str],
	label: &str,
) -> Result<Option<&'a str>, HubError> {
	if names.contains(root) && !named.is_empty() {
		return Err(HubError::Incompatible(format!(
			"root and named {label} chat templates map to the same runtime file"
		)));
	}
	match named {
		[] if names.contains(root) => Ok(Some(root)),
		[] => Ok(None),
		[path] => Ok(Some(path)),
		_ => Err(HubError::Incompatible(format!(
			"multiple named {label} chat templates are present"
		))),
	}
}

fn generation_defaults_from_value(
	value: Option<&serde_json::Value>,
) -> Result<ModelGenerationDefaults, HubError> {
	let Some(value) = value else {
		return Ok(ModelGenerationDefaults::default());
	};
	let object = value.as_object().ok_or_else(|| {
		HubError::Incompatible("generation_config.json must contain an object".to_string())
	})?;
	let temperature = optional_generation_f32(object.get("temperature"), "temperature", 0.0..=2.0)?;
	let top_p = optional_generation_f32(object.get("top_p"), "top_p", 0.0..=1.0)?;
	let top_k = object
		.get("top_k")
		.map(|value| {
			value
				.as_u64()
				.and_then(|value| u32::try_from(value).ok())
				.ok_or_else(|| {
					HubError::Incompatible(
						"generation_config top_k must be an unsigned 32-bit integer".to_string(),
					)
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
				.ok_or_else(|| {
					HubError::Incompatible(
						"generation_config max_new_tokens must be positive".to_string(),
					)
				})
		})
		.transpose()?;
	let do_sample = object
		.get("do_sample")
		.map(|value| {
			value.as_bool().ok_or_else(|| {
				HubError::Incompatible("generation_config do_sample must be boolean".to_string())
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

fn optional_generation_f32(
	value: Option<&serde_json::Value>,
	field: &str,
	range: std::ops::RangeInclusive<f32>,
) -> Result<Option<f32>, HubError> {
	value
		.map(|value| {
			let value = value.as_f64().ok_or_else(|| {
				HubError::Incompatible(format!("generation_config {field} must be numeric"))
			})?;
			let value = value.to_string().parse::<f32>().map_err(|error| {
				HubError::Incompatible(format!(
					"generation_config {field} is not representable: {error}"
				))
			})?;
			if !value.is_finite() || !range.contains(&value) {
				return Err(HubError::Incompatible(format!(
					"generation_config {field} is outside the supported range"
				)));
			}
			Ok(value)
		})
		.transpose()
}

fn declared_remote_max_context(config: &serde_json::Value) -> Option<usize> {
	[
		"max_position_embeddings",
		"max_sequence_length",
		"seq_length",
		"model_max_length",
	]
	.into_iter()
	.filter_map(|key| config.get(key).and_then(serde_json::Value::as_u64))
	.min()
	.and_then(|value| usize::try_from(value).ok())
	.filter(|value| *value > 0)
}

fn traits_from_wire(wire: &ModelWire, files: &[String]) -> ModelTraits {
	let mut traits = ModelTraits::default();
	let advertised_image_input = wire.pipeline_tag.as_deref() == Some("image-text-to-text")
		|| wire
			.tags
			.iter()
			.any(|tag| matches!(tag.as_str(), "vision-language-model" | "image-text-to-text"));
	if advertised_image_input {
		const KEY: &str = "extension:huggingface.advertised_input_image";
		traits
			.extras
			.insert(KEY.to_string(), serde_json::Value::Bool(true));
		traits
			.confidence
			.insert(KEY.to_string(), TraitConfidence::Advertised);
		traits.evidence.push(TraitEvidence {
			trait_key: KEY.to_string(),
			source: EvidenceSource::HubMetadata,
			detail: "Hub pipeline metadata or tag explicitly advertises image input".to_string(),
		});
	}
	let advertised_audio_input = wire
		.pipeline_tag
		.as_deref()
		.is_some_and(advertises_audio_input)
		|| wire.tags.iter().any(|tag| advertises_audio_input(tag));
	if advertised_audio_input {
		const KEY: &str = "extension:huggingface.advertised_input_audio";
		traits
			.extras
			.insert(KEY.to_string(), serde_json::Value::Bool(true));
		traits
			.confidence
			.insert(KEY.to_string(), TraitConfidence::Advertised);
		traits.evidence.push(TraitEvidence {
			trait_key: KEY.to_string(),
			source: EvidenceSource::HubMetadata,
			detail: "Hub pipeline metadata or tag explicitly advertises audio input".to_string(),
		});
	}
	if files.iter().any(|file| file == "tokenizer.json") {
		traits.evidence.push(TraitEvidence {
			trait_key: "repository:tokenizer".to_string(),
			source: EvidenceSource::HubMetadata,
			detail: "repository metadata advertises tokenizer.json".to_string(),
		});
	}
	traits
}

fn advertises_audio_input(value: &str) -> bool {
	matches!(
		value,
		"audio-classification"
			| "audio-to-audio"
			| "audio-to-text"
			| "audio-text-to-text"
			| "automatic-speech-recognition"
			| "speech-recognition"
			| "zero-shot-audio-classification"
	)
}

const fn gated(value: &serde_json::Value) -> bool {
	!matches!(
		value,
		serde_json::Value::Null | serde_json::Value::Bool(false)
	)
}

fn safe_runtime_file(path: &str) -> bool {
	crate::model::layout::safe_relative_path(path)
		&& ((Path::new(path).components().count() == 1
			&& (Path::new(path).extension() == Some(std::ffi::OsStr::new("safetensors"))
				|| path == "model.safetensors.index.json"
				|| runtime_metadata_file_name(path)))
			|| matches!(
				path,
				"additional_chat_templates/default.jinja"
					| "additional_chat_templates/tool_use.jinja"
					| "chat_templates/default.jinja"
					| "chat_templates/tool_use.jinja"
			))
}

pub(crate) fn local_runtime_path(remote: &str) -> &str {
	match remote {
		"additional_chat_templates/default.jinja" | "chat_templates/default.jinja" => {
			"chat_template.jinja"
		}
		"additional_chat_templates/tool_use.jinja" | "chat_templates/tool_use.jinja" => {
			"chat_template_tool_use.jinja"
		}
		other => other,
	}
}

fn remote_file(wire: &TreeWire) -> Result<RemoteFile, HubError> {
	if !crate::model::layout::safe_relative_path(&wire.path) {
		return Err(HubError::Protocol(format!(
			"unsafe Hub file path {:?}",
			wire.path
		)));
	}
	if wire.size == 0 && wire.path.ends_with(".safetensors") {
		return Err(HubError::Incompatible(format!(
			"zero-byte weight file {:?}",
			wire.path
		)));
	}
	if wire
		.security
		.as_ref()
		.is_some_and(|security| security.status == "unsafe")
	{
		return Err(HubError::Incompatible(format!(
			"Hub security scan marked {:?} unsafe",
			wire.path
		)));
	}
	RemoteFile::new(
		wire.path.clone(),
		wire.size,
		wire.lfs.as_ref().map(|lfs| lfs.oid.clone()),
	)
}

fn validate_download_plan(plan: &DownloadPlan) -> Result<(), HubError> {
	if !plan.model.traits.mlx {
		return Err(HubError::Incompatible(
			"download plan model is not MLX-compatible".to_string(),
		));
	}
	let total = validate_download_files(&plan.files)?;
	if total != plan.total_bytes {
		return Err(HubError::Protocol(
			"download plan total_bytes does not match files".to_string(),
		));
	}
	Ok(())
}

fn validate_download_files(files: &[RemoteFile]) -> Result<u64, HubError> {
	if files.is_empty() || files.len() > 10_000 {
		return Err(HubError::Protocol(
			"download plan must contain between 1 and 10000 files".to_string(),
		));
	}
	let mut names = BTreeSet::new();
	let mut local_names = BTreeSet::new();
	let mut total = 0_u64;
	let mut non_weight = 0_u64;
	for file in files {
		RemoteFile::new(file.path.clone(), file.size, file.expected_sha256.clone())?;
		if !names.insert(file.path.as_str()) {
			return Err(HubError::Protocol(format!(
				"duplicate download path {:?}",
				file.path
			)));
		}
		if !local_names.insert(local_runtime_path(&file.path)) {
			return Err(HubError::Protocol(format!(
				"download paths collide after runtime normalization: {:?}",
				file.path
			)));
		}
		total = total
			.checked_add(file.size)
			.ok_or_else(|| HubError::Protocol("file-plan byte count overflow".to_string()))?;
		if !file.path.ends_with(".safetensors") {
			non_weight = non_weight
				.checked_add(file.size)
				.ok_or_else(|| HubError::Protocol("file-plan byte count overflow".to_string()))?;
		}
	}
	if !names.contains("config.json")
		|| !names.contains("tokenizer.json")
		|| !files.iter().any(|file| file.path.ends_with(".safetensors"))
	{
		return Err(HubError::Incompatible(
			"download plan lacks config.json, tokenizer.json, or safetensors".to_string(),
		));
	}
	if non_weight > 256 << 20 {
		return Err(HubError::Incompatible(
			"non-weight runtime files exceed 256 MiB".to_string(),
		));
	}
	Ok(total)
}

/// Transfer bytes plus the safety margin retained after a model download.
pub(crate) fn required_download_storage_bytes(transfer_bytes: u64) -> Option<u64> {
	let margin = (64_u64 << 20).max(transfer_bytes / 20);
	transfer_bytes.checked_add(margin)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FileIdentity {
	device: u64,
	inode: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FileSnapshot {
	identity: FileIdentity,
	bytes: u64,
	modified_seconds: i64,
	modified_nanoseconds: i64,
	changed_seconds: i64,
	changed_nanoseconds: i64,
}

fn validate_download_staging(staging: &Path) -> Result<(), HubError> {
	let staging_directory = OpenOptions::new()
		.read(true)
		.custom_flags(libc::O_CLOEXEC | libc::O_DIRECTORY | libc::O_NOFOLLOW)
		.open(staging)
		.map_err(|source| HubError::Io {
			path: staging.to_path_buf(),
			source,
		})?;
	let staging_metadata = staging_directory
		.metadata()
		.map_err(|source| HubError::Io {
			path: staging.to_path_buf(),
			source,
		})?;
	let staging_has_acl =
		crate::home::has_extended_acl(&staging_directory).map_err(|source| HubError::Io {
			path: staging.to_path_buf(),
			source,
		})?;
	if !staging_metadata.file_type().is_dir()
		|| staging_metadata.uid() != crate::home::effective_user_id()
		|| staging_metadata.permissions().mode() & 0o777 != 0o700
		|| staging_has_acl
	{
		return Err(HubError::Io {
			path: staging.to_path_buf(),
			source: std::io::Error::new(
				std::io::ErrorKind::PermissionDenied,
				"staging must be an owner-only 0700 directory without an extended ACL",
			),
		});
	}
	let mut entries = std::fs::read_dir(staging).map_err(|source| HubError::Io {
		path: staging.to_path_buf(),
		source,
	})?;
	if entries
		.next()
		.transpose()
		.map_err(|source| HubError::Io {
			path: staging.to_path_buf(),
			source,
		})?
		.is_some()
	{
		return Err(HubError::Protocol(
			"download staging must be empty".to_string(),
		));
	}
	Ok(())
}

fn inspect_partial(
	path: &Path,
	expected: Option<FileIdentity>,
) -> Result<(u64, Option<FileIdentity>), HubError> {
	match std::fs::symlink_metadata(path) {
		Ok(metadata) => {
			let identity = validate_partial_metadata(path, &metadata)?;
			if expected.is_some_and(|expected| expected != identity) {
				return Err(HubError::Protocol(format!(
					"partial download identity changed: {}",
					path.display()
				)));
			}
			Ok((metadata.len(), Some(identity)))
		}
		Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok((0, None)),
		Err(source) => Err(HubError::Io {
			path: path.to_path_buf(),
			source,
		}),
	}
}

fn validate_partial_metadata(
	path: &Path,
	metadata: &std::fs::Metadata,
) -> Result<FileIdentity, HubError> {
	if !metadata.file_type().is_file()
		|| metadata.nlink() != 1
		|| metadata.uid() != crate::home::effective_user_id()
		|| metadata.permissions().mode() & 0o777 != 0o600
	{
		return Err(HubError::Protocol(format!(
			"partial download must be a current-user 0600 regular file with one link: {}",
			path.display()
		)));
	}
	Ok(FileIdentity {
		device: metadata.dev(),
		inode: metadata.ino(),
	})
}

fn file_snapshot(path: &Path, metadata: &std::fs::Metadata) -> Result<FileSnapshot, HubError> {
	Ok(FileSnapshot {
		identity: validate_partial_metadata(path, metadata)?,
		bytes: metadata.len(),
		modified_seconds: metadata.mtime(),
		modified_nanoseconds: metadata.mtime_nsec(),
		changed_seconds: metadata.ctime(),
		changed_nanoseconds: metadata.ctime_nsec(),
	})
}

fn validate_partial_acl(path: &Path, file: &std::fs::File) -> Result<(), HubError> {
	if crate::home::has_extended_acl(file).map_err(|source| HubError::Io {
		path: path.to_path_buf(),
		source,
	})? {
		return Err(HubError::Protocol(format!(
			"partial download must not have an extended ACL: {}",
			path.display()
		)));
	}
	Ok(())
}

fn validate_file_snapshot(path: &Path, expected: &FileSnapshot) -> Result<(), HubError> {
	let file = OpenOptions::new()
		.read(true)
		.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK)
		.open(path)
		.map_err(|source| HubError::Io {
			path: path.to_path_buf(),
			source,
		})?;
	let metadata = file.metadata().map_err(|source| HubError::Io {
		path: path.to_path_buf(),
		source,
	})?;
	let actual = file_snapshot(path, &metadata)?;
	validate_partial_acl(path, &file)?;
	if &actual != expected {
		return Err(HubError::Protocol(format!(
			"verified download changed before publication: {}",
			path.display()
		)));
	}
	Ok(())
}

fn open_partial(
	path: &Path,
	append: bool,
	expected: Option<FileIdentity>,
) -> Result<(std::fs::File, FileIdentity), HubError> {
	let mut options = OpenOptions::new();
	options
		.write(true)
		.mode(0o600)
		.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
	if expected.is_none() {
		options.create_new(true);
	} else if append {
		options.append(true);
	} else {
		options.truncate(true);
	}
	let file = options.open(path).map_err(|source| HubError::Io {
		path: path.to_path_buf(),
		source,
	})?;
	let metadata = file.metadata().map_err(|source| HubError::Io {
		path: path.to_path_buf(),
		source,
	})?;
	let identity = validate_partial_metadata(path, &metadata)?;
	validate_partial_acl(path, &file)?;
	if expected.is_some_and(|expected| expected != identity) {
		return Err(HubError::Protocol(format!(
			"partial download identity changed while opening: {}",
			path.display()
		)));
	}
	Ok((file, identity))
}

async fn hash_file(
	path: PathBuf,
	cancellation: Option<DownloadCancellation>,
) -> Result<(u64, String, FileSnapshot), HubError> {
	check_cancelled(cancellation.as_ref())?;
	let open_path = path.clone();
	let (file, snapshot) = tokio::task::spawn_blocking(move || {
		let file = OpenOptions::new()
			.read(true)
			.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK)
			.open(&open_path)
			.map_err(|source| HubError::Io {
				path: open_path.clone(),
				source,
			})?;
		let metadata = file.metadata().map_err(|source| HubError::Io {
			path: open_path.clone(),
			source,
		})?;
		let snapshot = file_snapshot(&open_path, &metadata)?;
		validate_partial_acl(&open_path, &file)?;
		Ok::<_, HubError>((file, snapshot))
	})
	.await
	.map_err(blocking_hub_task_error)??;
	let (bytes, digest) = hash_reader(
		tokio::fs::File::from_std(file),
		&path,
		cancellation.as_ref(),
	)
	.await?;
	check_cancelled(cancellation.as_ref())?;
	let validate_path = path.clone();
	tokio::task::spawn_blocking(move || validate_file_snapshot(&validate_path, &snapshot))
		.await
		.map_err(blocking_hub_task_error)??;
	Ok((bytes, digest, snapshot))
}

fn rename_exclusive(source: &Path, destination: &Path) -> Result<(), HubError> {
	let source_c = CString::new(source.as_os_str().as_bytes())
		.map_err(|_| HubError::Protocol("source path contains NUL".to_string()))?;
	let destination_c = CString::new(destination.as_os_str().as_bytes())
		.map_err(|_| HubError::Protocol("destination path contains NUL".to_string()))?;
	// SAFETY: both C strings are NUL-terminated for the duration of the call;
	// AT_FDCWD selects their absolute/relative path semantics; RENAME_EXCL is
	// a valid macOS flag and neither pointer aliases writable memory.
	let status = unsafe {
		libc::renameatx_np(
			libc::AT_FDCWD,
			source_c.as_ptr(),
			libc::AT_FDCWD,
			destination_c.as_ptr(),
			libc::RENAME_EXCL,
		)
	};
	if status == 0 {
		Ok(())
	} else {
		Err(HubError::Io {
			path: destination.to_path_buf(),
			source: std::io::Error::last_os_error(),
		})
	}
}

async fn hash_reader<R>(
	mut reader: R,
	path: &Path,
	cancellation: Option<&DownloadCancellation>,
) -> Result<(u64, String), HubError>
where
	R: AsyncRead + Unpin,
{
	let mut hash = sha2::Sha256::new();
	let mut bytes = 0_u64;
	let mut buffer = vec![0_u8; 1024 * 1024];
	loop {
		check_cancelled(cancellation)?;
		let read = reader
			.read(&mut buffer)
			.await
			.map_err(|source| HubError::Io {
				path: path.to_path_buf(),
				source,
			})?;
		if read == 0 {
			break;
		}
		bytes = bytes
			.checked_add(
				u64::try_from(read)
					.map_err(|_| HubError::Protocol("hash byte count overflow".to_string()))?,
			)
			.ok_or_else(|| HubError::Protocol("hash byte count overflow".to_string()))?;
		hash.update(&buffer[..read]);
	}
	Ok((bytes, hex::encode(hash.finalize())))
}

fn emit(callbacks: DownloadCallbacks<'_>, event: &DownloadEvent) -> Result<(), HubError> {
	check_cancelled(callbacks.cancellation)?;
	if let Some(reporter) = callbacks.reporter {
		reporter(event);
	}
	if let Some(observer) = callbacks.observer {
		match observer(event).map_err(|error| HubError::Observer(error.to_string()))? {
			DownloadControl::Continue => {}
			DownloadControl::Cancel => return Err(HubError::Cancelled),
		}
	}
	check_cancelled(callbacks.cancellation)
}

#[allow(
	clippy::needless_pass_by_value,
	reason = "Result::map_err supplies an owned JoinError directly"
)]
fn blocking_hub_task_error(error: tokio::task::JoinError) -> HubError {
	HubError::Protocol(format!("blocking Hub filesystem task failed: {error}"))
}

fn check_cancelled(cancellation: Option<&DownloadCancellation>) -> Result<(), HubError> {
	if cancellation.is_some_and(DownloadCancellation::is_cancelled) {
		Err(HubError::Cancelled)
	} else {
		Ok(())
	}
}

async fn cancelled(cancellation: &DownloadCancellation) {
	while !cancellation.is_cancelled() {
		tokio::time::sleep(Duration::from_millis(25)).await;
	}
}

fn sync_directory(path: &Path) -> Result<(), HubError> {
	let directory = OpenOptions::new()
		.read(true)
		.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
		.open(path)
		.map_err(|source| HubError::Io {
			path: path.to_path_buf(),
			source,
		})?;
	directory.sync_all().map_err(|source| HubError::Io {
		path: path.to_path_buf(),
		source,
	})
}

fn model_api_segments(id: &HubModelId) -> Vec<&str> {
	let mut segments = Vec::with_capacity(4);
	segments.extend(["api", "models"]);
	segments.extend(id.as_str().split('/'));
	segments
}

fn append_compatible_page<T>(
	items: &mut Vec<T>,
	ranked: impl IntoIterator<Item = (usize, T)>,
	limit: usize,
) -> Option<usize> {
	for (rank, item) in ranked {
		items.push(item);
		if items.len() == limit {
			return rank.checked_add(1);
		}
	}
	None
}

fn filter_search_storage(
	ranked: Vec<(usize, HubModel)>,
	available: Option<u64>,
	diagnostics: &mut Vec<HubDiagnostic>,
) -> Vec<(usize, HubModel)> {
	let Some(available) = available else {
		return ranked;
	};
	let mut accepted = Vec::with_capacity(ranked.len());
	for (rank, model) in ranked {
		let Some(required) = model
			.download_bytes
			.and_then(required_download_storage_bytes)
		else {
			diagnostics.push(HubDiagnostic {
				id: Some(model.id),
				message: "candidate download storage requirement is unavailable".to_string(),
			});
			continue;
		};
		if required > available {
			diagnostics.push(HubDiagnostic {
				id: Some(model.id),
				message: format!(
					"download requires {required} bytes including safety margin, exceeds available \
					 storage {available} bytes"
				),
			});
			continue;
		}
		accepted.push((rank, model));
	}
	accepted
}

const fn search_scanned(immediately_scanned: usize, metadata_scanned: usize) -> usize {
	immediately_scanned.saturating_add(metadata_scanned)
}

fn validate_remote_filters(filters: &[TraitFilter]) -> Result<(), HubError> {
	for filter in filters {
		let available = match filter.predicate() {
			TraitPredicate::Capability(key) => remote_capability_available(key),
			TraitPredicate::MinimumConfidence { key, confidence } => {
				*confidence != TraitConfidence::RuntimeVerified && remote_confidence_available(key)
			}
			TraitPredicate::MinimumMtp(stage) => {
				matches!(stage, MtpSupport::Absent | MtpSupport::Advertised)
			}
			TraitPredicate::AtMost { .. } | TraitPredicate::AtLeast { .. } => true,
		};
		if !available {
			return Err(HubError::Configuration(format!(
				"trait filter {:?} has no remote catalog evidence; use `emelex hub capabilities`",
				filter.as_str()
			)));
		}
	}
	Ok(())
}

fn remote_capability_available(key: &str) -> bool {
	REMOTE_FILTERS.iter().any(|help| help.filter == key)
}

fn remote_confidence_available(key: &str) -> bool {
	remote_capability_available(key)
}

fn continuation_cursor(
	search: &HubSearch,
	credential_scope: Option<&[u8; 32]>,
	fit_profile: Option<(WorkloadProfile, u64)>,
	upstream: Option<String>,
	offset: usize,
	candidate_count: usize,
	next_upstream: Option<String>,
) -> Result<Option<String>, HubError> {
	if offset < candidate_count {
		encode_search_cursor(search, credential_scope, fit_profile, upstream, offset).map(Some)
	} else {
		next_upstream
			.map(|cursor| {
				encode_search_cursor(search, credential_scope, fit_profile, Some(cursor), 0)
			})
			.transpose()
	}
}

fn search_scope(
	search: &HubSearch,
	credential_scope: Option<&[u8; 32]>,
	fit_profile: Option<(WorkloadProfile, u64)>,
) -> Result<String, HubError> {
	let query = normalized_search_query(search)?;
	let mut require = search
		.require
		.iter()
		.map(TraitFilter::as_str)
		.collect::<Vec<_>>();
	require.sort_unstable();
	require.dedup();
	let scope = serde_json::to_vec(&(
		query,
		require,
		search.mlx_library,
		credential_scope,
		fit_profile,
	))
	.map_err(|error| HubError::Protocol(format!("cannot encode Hub search scope: {error}")))?;
	Ok(hex::encode(Sha256::digest(scope)))
}

fn normalized_search_query(search: &HubSearch) -> Result<Option<&str>, HubError> {
	let query = search.query.as_deref().map(str::trim).unwrap_or_default();
	if query.len() > MAX_SEARCH_QUERY_BYTES {
		return Err(HubError::Configuration(format!(
			"Hub search query exceeds {MAX_SEARCH_QUERY_BYTES} bytes"
		)));
	}
	if query.chars().any(char::is_control) {
		return Err(HubError::Configuration(
			"Hub search query contains control characters".to_string(),
		));
	}
	Ok((!query.is_empty()).then_some(query))
}

fn encode_search_cursor(
	search: &HubSearch,
	credential_scope: Option<&[u8; 32]>,
	fit_profile: Option<(WorkloadProfile, u64)>,
	upstream: Option<String>,
	offset: usize,
) -> Result<String, HubError> {
	if let Some(cursor) = &upstream {
		validate_upstream_cursor(cursor)?;
	}
	if offset > 1_000 {
		return Err(HubError::Protocol(
			"Hub search cursor offset exceeds the configured maximum".to_string(),
		));
	}
	let bytes = serde_json::to_vec(&SearchCursor {
		version: SEARCH_CURSOR_VERSION,
		upstream,
		offset,
		scope: search_scope(search, credential_scope, fit_profile)?,
	})
	.map_err(|error| HubError::Protocol(format!("cannot encode Hub search cursor: {error}")))?;
	if bytes.len() > MAX_SEARCH_CURSOR_BYTES {
		return Err(HubError::Protocol(
			"Hub search cursor exceeds its encoded-state bound".to_string(),
		));
	}
	Ok(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes))
}

fn decode_search_cursor(
	search: &HubSearch,
	credential_scope: Option<&[u8; 32]>,
	fit_profile: Option<(WorkloadProfile, u64)>,
) -> Result<SearchCursor, HubError> {
	let Some(cursor) = &search.cursor else {
		return Ok(SearchCursor {
			version: SEARCH_CURSOR_VERSION,
			upstream: None,
			offset: 0,
			scope: search_scope(search, credential_scope, fit_profile)?,
		});
	};
	if cursor.len() > MAX_SEARCH_CURSOR_BYTES.saturating_mul(2) {
		return Err(HubError::Protocol("invalid Hub search cursor".to_string()));
	}
	let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
		.decode(cursor)
		.map_err(|_| HubError::Protocol("invalid Hub search cursor".to_string()))?;
	if bytes.len() > MAX_SEARCH_CURSOR_BYTES {
		return Err(HubError::Protocol("invalid Hub search cursor".to_string()));
	}
	let decoded: SearchCursor = serde_json::from_slice(&bytes)
		.map_err(|_| HubError::Protocol("invalid Hub search cursor".to_string()))?;
	if decoded.version != SEARCH_CURSOR_VERSION
		|| decoded.offset > 1_000
		|| decoded.scope != search_scope(search, credential_scope, fit_profile)?
	{
		return Err(HubError::Protocol(
			"Hub search cursor does not match this query, catalog, credential, or fit scope"
				.to_string(),
		));
	}
	if let Some(cursor) = &decoded.upstream {
		validate_upstream_cursor(cursor)?;
	}
	Ok(decoded)
}

fn validate_upstream_cursor(cursor: &str) -> Result<(), HubError> {
	let bytes = cursor.as_bytes();
	if bytes.is_empty()
		|| bytes.len() > MAX_UPSTREAM_CURSOR_BYTES
		|| !bytes.iter().all(|byte| byte.is_ascii_graphic())
	{
		return Err(HubError::Protocol("invalid Hub cursor".to_string()));
	}
	Ok(())
}

fn candidate_inspection_diagnostic(error: &HubError) -> bool {
	match error {
		HubError::NotPublic(_) | HubError::Incompatible(_) | HubError::Protocol(_) => true,
		HubError::Http { status, .. } => {
			matches!(*status, StatusCode::NOT_FOUND | StatusCode::GONE)
		}
		_ => false,
	}
}

fn next_upstream_cursor(headers: &header::HeaderMap) -> Result<Option<String>, HubError> {
	let Some(url) = next_link_value(headers)? else {
		return Ok(None);
	};
	let url = Url::parse(&url).map_err(|error| HubError::Protocol(error.to_string()))?;
	let cursor = url
		.query_pairs()
		.find_map(|(key, value)| (key == "cursor").then(|| value.into_owned()));
	if let Some(cursor) = &cursor {
		validate_upstream_cursor(cursor)?;
	}
	Ok(cursor)
}

fn next_link(headers: &header::HeaderMap, endpoint: &Url) -> Result<Option<Url>, HubError> {
	let Some(value) = next_link_value(headers)? else {
		return Ok(None);
	};
	let url = Url::parse(&value).map_err(|error| HubError::Protocol(error.to_string()))?;
	if url.scheme() != endpoint.scheme()
		|| url.host_str() != endpoint.host_str()
		|| url.port_or_known_default() != endpoint.port_or_known_default()
	{
		return Err(HubError::Protocol(
			"Hub pagination crossed endpoint origin".to_string(),
		));
	}
	Ok(Some(url))
}

fn next_link_value(headers: &header::HeaderMap) -> Result<Option<String>, HubError> {
	let Some(value) = headers.get(header::LINK) else {
		return Ok(None);
	};
	let value = value
		.to_str()
		.map_err(|error| HubError::Protocol(error.to_string()))?;
	for link in value.split(',') {
		if link.contains("rel=\"next\"")
			&& let Some(url) = link
				.split(';')
				.next()
				.map(str::trim)
				.and_then(|part| part.strip_prefix('<'))
				.and_then(|part| part.strip_suffix('>'))
		{
			return Ok(Some(url.to_string()));
		}
	}
	Ok(None)
}

/// Hugging Face discovery/download failure.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HubError {
	/// Client bounds were unsafe.
	#[error("invalid Hub configuration: {0}")]
	Configuration(String),
	/// Endpoint or generated URL is invalid.
	#[error("invalid Hub URL: {0}")]
	Url(String),
	/// HTTP transport failed.
	#[error("Hub request failed: {0}")]
	Request(#[source] reqwest::Error),
	/// Hub returned an unsuccessful response.
	#[error("Hub returned HTTP {status}: {body}")]
	Http {
		/// HTTP status.
		status: StatusCode,
		/// Capped response body.
		body: String,
	},
	/// Public-only policy rejected the model.
	#[error("model is private or gated: {0}")]
	NotPublic(String),
	/// Static MLX compatibility rejected the model.
	#[error("incompatible Hub model: {0}")]
	Incompatible(String),
	/// Hub response violated the expected protocol.
	#[error("invalid Hub response: {0}")]
	Protocol(String),
	/// Progress observer failed.
	#[error("download observer failed: {0}")]
	Observer(String),
	/// Transfer was cooperatively cancelled.
	#[error("download cancelled")]
	Cancelled,
	/// A file transfer stopped yielding headers, error text, or body bytes.
	#[error("download for {path:?} was idle for {seconds} seconds")]
	DownloadIdleTimeout {
		/// Relative remote path.
		path: String,
		/// Configured idle ceiling.
		seconds: u64,
	},
	/// Emelex-owned staging I/O failed.
	#[error("I/O failed for {path:?}: {source}")]
	Io {
		/// Affected path.
		path: PathBuf,
		/// Underlying error.
		#[source]
		source: std::io::Error,
	},
	/// Downloaded size differs from the plan.
	#[error("size mismatch for {path:?}: expected {expected}, got {actual}")]
	Size {
		/// Relative file.
		path: String,
		/// Planned bytes.
		expected: u64,
		/// Actual bytes.
		actual: u64,
	},
	/// Downloaded SHA-256 differs from Hub LFS metadata.
	#[error("SHA-256 mismatch for {path:?}: expected {expected}, got {actual}")]
	Hash {
		/// Relative file.
		path: String,
		/// Planned digest.
		expected: String,
		/// Actual digest.
		actual: String,
	},
}

impl HubError {
	fn transient(&self) -> bool {
		match self {
			Self::Request(error) => error.is_timeout() || error.is_connect() || error.is_body(),
			Self::Http { status, .. } => {
				*status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
			}
			Self::Size {
				expected, actual, ..
			} => actual < expected,
			Self::DownloadIdleTimeout { .. } => true,
			_ => false,
		}
	}
}

#[cfg(test)]
mod tests {
	use std::{
		os::unix::fs::{OpenOptionsExt as _, PermissionsExt as _},
		process::Command,
		sync::Mutex,
	};

	use sha2::Digest as _;
	use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

	use super::*;

	enum LoopbackBehavior {
		Response {
			status: &'static str,
			headers: Vec<String>,
			body: Vec<u8>,
		},
		Disconnect,
		StallHeaders,
		StallBody {
			status: &'static str,
			headers: Vec<String>,
			prefix: Vec<u8>,
		},
	}

	struct LoopbackServer {
		endpoint: String,
		task: tokio::task::JoinHandle<Vec<u8>>,
	}

	struct LoopbackSequence {
		endpoint: String,
		task: tokio::task::JoinHandle<Vec<Vec<u8>>>,
	}

	async fn loopback_server(behavior: LoopbackBehavior) -> LoopbackServer {
		let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
			.await
			.expect("bind loopback");
		let endpoint = format!(
			"http://{}",
			listener.local_addr().expect("loopback address")
		);
		let task = tokio::spawn(async move { serve_loopback(&listener, behavior).await });
		LoopbackServer { endpoint, task }
	}

	async fn loopback_sequence_server(behaviors: Vec<LoopbackBehavior>) -> LoopbackSequence {
		let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
			.await
			.expect("bind loopback sequence");
		let endpoint = format!(
			"http://{}",
			listener.local_addr().expect("loopback sequence address")
		);
		let task = tokio::spawn(async move {
			let mut requests = Vec::with_capacity(behaviors.len());
			for behavior in behaviors {
				requests.push(serve_loopback(&listener, behavior).await);
			}
			requests
		});
		LoopbackSequence { endpoint, task }
	}

	async fn serve_loopback(
		listener: &tokio::net::TcpListener,
		behavior: LoopbackBehavior,
	) -> Vec<u8> {
		let (mut socket, _) = listener.accept().await.expect("accept loopback");
		let mut request = Vec::new();
		let mut buffer = [0_u8; 1024];
		while !request.windows(4).any(|window| window == b"\r\n\r\n") {
			let read = socket.read(&mut buffer).await.expect("read request");
			if read == 0 {
				break;
			}
			request.extend_from_slice(&buffer[..read]);
			assert!(request.len() <= 16 << 10, "test request headers too large");
		}
		match behavior {
			LoopbackBehavior::Response {
				status,
				headers,
				body,
			} => {
				let response = format!(
					"HTTP/1.1 {status}\r\nConnection: close\r\n{}\r\n\r\n",
					headers.join("\r\n")
				);
				socket
					.write_all(response.as_bytes())
					.await
					.expect("write headers");
				socket.write_all(&body).await.expect("write body");
				socket.shutdown().await.expect("close response");
			}
			LoopbackBehavior::Disconnect => {}
			LoopbackBehavior::StallHeaders => {
				std::future::pending::<()>().await;
			}
			LoopbackBehavior::StallBody {
				status,
				headers,
				prefix,
			} => {
				let response = format!(
					"HTTP/1.1 {status}\r\nConnection: close\r\n{}\r\n\r\n",
					headers.join("\r\n")
				);
				socket
					.write_all(response.as_bytes())
					.await
					.expect("write headers");
				socket.write_all(&prefix).await.expect("write prefix");
				socket.flush().await.expect("flush prefix");
				std::future::pending::<()>().await;
			}
		}
		request
	}

	fn loopback_client(endpoint: &str) -> HubClient {
		let config = HubConfig {
			results: 1,
			scan_limit: 1,
			metadata_concurrency: 1,
			request_timeout_seconds: 1,
			retries: 0,
		};
		HubClient::with_endpoint_and_fit(config, endpoint, None, None).expect("loopback client")
	}

	fn json_response(status: &'static str, value: &serde_json::Value) -> LoopbackBehavior {
		let body = serde_json::to_vec(value).expect("serialize loopback JSON");
		LoopbackBehavior::Response {
			status,
			headers: vec![
				"Content-Type: application/json".to_string(),
				format!("Content-Length: {}", body.len()),
			],
			body,
		}
	}

	fn candidate_wire_value() -> serde_json::Value {
		serde_json::json!({
			"id": "owner/model",
			"private": false,
			"gated": false,
			"sha": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
			"downloads": 7,
			"likes": 3,
			"tags": ["mlx"],
			"library_name": "mlx",
			"pipeline_tag": "text-generation",
			"siblings": [
				{"rfilename": "config.json"},
				{"rfilename": "model.safetensors"}
			]
		})
	}

	fn test_download_plan(path: &str, bytes: &[u8]) -> (DownloadPlan, RemoteFile) {
		let digest = hex::encode(sha2::Sha256::digest(bytes));
		let remote = RemoteFile::new(
			path.to_string(),
			u64::try_from(bytes.len()).expect("fixture size"),
			Some(digest),
		)
		.expect("remote fixture");
		let plan = DownloadPlan {
			model: model(),
			files: vec![remote.clone()],
			total_bytes: remote.size(),
		};
		(plan, remote)
	}

	fn download_staging() -> tempfile::TempDir {
		let directory = tempfile::tempdir().expect("download staging");
		std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o700))
			.expect("staging mode");
		directory
	}

	fn write_partial(path: &Path, bytes: &[u8]) {
		let mut file = OpenOptions::new()
			.write(true)
			.create_new(true)
			.mode(0o600)
			.open(path)
			.expect("create partial");
		std::io::Write::write_all(&mut file, bytes).expect("write partial");
	}

	fn callbacks<'a>(
		observer: Option<&'a DownloadObserver>,
		cancellation: Option<&'a DownloadCancellation>,
	) -> DownloadCallbacks<'a> {
		DownloadCallbacks {
			reporter: None,
			observer,
			cancellation,
		}
	}

	#[test]
	fn hub_urls_preserve_unnamespaced_and_namespaced_repository_ids() {
		let client = loopback_client("http://127.0.0.1:1");
		let revision = ResolvedRevision::parse("a".repeat(40)).expect("revision");
		for (id, api_path, resolve_path) in [
			(
				"gpt2",
				"/api/models/gpt2",
				format!("/gpt2/resolve/{revision}/config.json"),
			),
			(
				"owner/model",
				"/api/models/owner/model",
				format!("/owner/model/resolve/{revision}/config.json"),
			),
		] {
			let id = HubModelId::parse(id).expect("repository ID");
			let api = client
				.api_url(&model_api_segments(&id))
				.expect("model API URL");
			let resolve = client
				.resolve_url(&id, &revision, "config.json")
				.expect("resolve URL");

			assert_eq!(api.path(), api_path);
			assert_eq!(resolve.path(), resolve_path);
		}
	}

	struct DropAwareReader {
		dropped: Arc<AtomicBool>,
		yielded: bool,
	}

	impl tokio::io::AsyncRead for DropAwareReader {
		fn poll_read(
			mut self: std::pin::Pin<&mut Self>,
			_context: &mut std::task::Context<'_>,
			buffer: &mut tokio::io::ReadBuf<'_>,
		) -> std::task::Poll<std::io::Result<()>> {
			if self.yielded {
				return std::task::Poll::Pending;
			}
			self.yielded = true;
			buffer.put_slice(b"x");
			std::task::Poll::Ready(Ok(()))
		}
	}

	impl Drop for DropAwareReader {
		fn drop(&mut self) {
			self.dropped.store(true, Ordering::Release);
		}
	}

	#[tokio::test]
	async fn dropping_hash_future_drops_its_reader_without_detached_work() {
		let dropped = Arc::new(AtomicBool::new(false));
		{
			let future = hash_reader(
				DropAwareReader {
					dropped: Arc::clone(&dropped),
					yielded: false,
				},
				Path::new("drop-aware"),
				None,
			);
			tokio::pin!(future);
			assert!(
				tokio::time::timeout(Duration::from_millis(10), &mut future)
					.await
					.is_err()
			);
		}
		assert!(dropped.load(Ordering::Acquire));
	}

	#[tokio::test]
	async fn range_resume_uses_206_and_preserves_existing_prefix() {
		let bytes = b"abcdef";
		let server = loopback_server(LoopbackBehavior::Response {
			status: "206 Partial Content",
			headers: vec![
				"Content-Length: 3".to_string(),
				"Content-Range: bytes 3-5/6".to_string(),
			],
			body: b"def".to_vec(),
		})
		.await;
		let client = loopback_client(&server.endpoint);
		let staging = download_staging();
		write_partial(&staging.path().join("config.json.part"), b"abc");
		let (plan, remote) = test_download_plan("config.json", bytes);

		client
			.download_file(&plan, &remote, staging.path(), callbacks(None, None))
			.await
			.expect("resumed download");

		let request = server.task.await.expect("server task");
		let request = String::from_utf8_lossy(&request).to_ascii_lowercase();
		assert!(request.contains("range: bytes=3-\r\n"));
		assert_eq!(
			std::fs::read(staging.path().join("config.json")).expect("published file"),
			bytes
		);
		assert!(!staging.path().join("config.json.part").exists());
	}

	#[tokio::test]
	async fn download_events_bracket_files_with_exact_transfer_totals() {
		let bytes = b"abcdef";
		let server = loopback_sequence_server(vec![
			LoopbackBehavior::Response {
				status: "200 OK",
				headers: vec!["Content-Length: 6".to_string()],
				body: bytes.to_vec(),
			},
			LoopbackBehavior::Response {
				status: "200 OK",
				headers: vec!["Content-Length: 1".to_string()],
				body: b"t".to_vec(),
			},
			LoopbackBehavior::Response {
				status: "200 OK",
				headers: vec!["Content-Length: 1".to_string()],
				body: b"w".to_vec(),
			},
		])
		.await;
		let client = loopback_client(&server.endpoint);
		let staging = download_staging();
		let (mut plan, _) = test_download_plan("config.json", bytes);
		let (_, tokenizer) = test_download_plan("tokenizer.json", b"t");
		let (_, weights) = test_download_plan("model.safetensors", b"w");
		plan.files.extend([tokenizer, weights]);
		plan.total_bytes = 8;
		let events = Arc::new(Mutex::new(Vec::new()));
		let observed = Arc::clone(&events);
		let observer: DownloadObserver = Arc::new(move |event| {
			let label = match event {
				DownloadEvent::TransferStarted { files, total } => {
					format!("started:{files}:{total}")
				}
				DownloadEvent::FileStarted { path, .. } => format!("file:{path}"),
				DownloadEvent::Progress { .. } => "progress".to_string(),
				DownloadEvent::FileVerified { path, .. } => format!("verified:{path}"),
				DownloadEvent::TransferCompleted { files, total } => {
					format!("completed:{files}:{total}")
				}
				_ => return Ok(DownloadControl::Continue),
			};
			observed
				.lock()
				.unwrap_or_else(std::sync::PoisonError::into_inner)
				.push(label);
			Ok(DownloadControl::Continue)
		});

		client
			.download_controlled(&plan, staging.path(), Some(&observer), None)
			.await
			.expect("download");

		let events = events
			.lock()
			.unwrap_or_else(std::sync::PoisonError::into_inner)
			.clone();
		assert_eq!(events.first().map(String::as_str), Some("started:3:8"));
		assert_eq!(events.last().map(String::as_str), Some("completed:3:8"));
		assert!(
			events
				.windows(2)
				.any(|events| events == ["progress", "verified:config.json"])
		);
		server.task.await.expect("server task");
	}

	#[tokio::test]
	async fn dropping_public_download_before_commit_cancels_only_its_linked_child() {
		let bytes = b"verified";
		let server = loopback_server(LoopbackBehavior::Response {
			status: "200 OK",
			headers: vec![format!("Content-Length: {}", bytes.len())],
			body: bytes.to_vec(),
		})
		.await;
		let gate = Arc::new(TestPublishGate::default());
		let mut client = loopback_client(&server.endpoint);
		client.publish_gate = Some(Arc::clone(&gate));
		let staging = download_staging();
		let staging_path = staging.path().to_path_buf();
		let destination = staging_path.join("config.json");
		let partial = staging_path.join("config.json.part");
		let (mut plan, _) = test_download_plan("config.json", bytes);
		for (path, body) in [
			("tokenizer.json", b"{}".as_slice()),
			("model.safetensors", b"weights".as_slice()),
		] {
			let remote = RemoteFile::new(
				path.to_string(),
				u64::try_from(body.len()).expect("fixture size"),
				Some(hex::encode(sha2::Sha256::digest(body))),
			)
			.expect("complete-plan file");
			plan.total_bytes = plan
				.total_bytes
				.checked_add(remote.size())
				.expect("complete-plan bytes");
			plan.files.push(remote);
		}
		let caller = DownloadCancellation::default();
		let task_caller = caller.clone();
		let task = tokio::spawn(async move {
			client
				.download_controlled(&plan, &staging_path, None, Some(&task_caller))
				.await
		});
		tokio::time::timeout(Duration::from_secs(3), async {
			while !gate.entered.load(Ordering::Acquire) {
				tokio::task::yield_now().await;
			}
		})
		.await
		.expect("final publication gate");

		task.abort();
		assert!(
			task.await
				.expect_err("download task aborted")
				.is_cancelled()
		);
		assert!(!caller.is_cancelled());
		gate.release.store(true, Ordering::Release);
		tokio::time::timeout(Duration::from_secs(3), async {
			while !gate.completed.load(Ordering::Acquire) {
				tokio::task::yield_now().await;
			}
		})
		.await
		.expect("detached publication phase exits");

		assert!(!destination.exists());
		assert!(partial.exists());
		server.task.await.expect("server task");
	}

	#[tokio::test]
	async fn ignored_range_200_truncates_partial_before_full_body() {
		let bytes = b"abcdef";
		let server = loopback_server(LoopbackBehavior::Response {
			status: "200 OK",
			headers: vec!["Content-Length: 6".to_string()],
			body: bytes.to_vec(),
		})
		.await;
		let client = loopback_client(&server.endpoint);
		let staging = download_staging();
		write_partial(&staging.path().join("config.json.part"), b"abc");
		let (plan, remote) = test_download_plan("config.json", bytes);

		client
			.download_file(&plan, &remote, staging.path(), callbacks(None, None))
			.await
			.expect("full restart");

		let request = server.task.await.expect("server task");
		assert!(
			String::from_utf8_lossy(&request)
				.to_ascii_lowercase()
				.contains("range: bytes=3-\r\n")
		);
		assert_eq!(
			std::fs::read(staging.path().join("config.json")).expect("published file"),
			bytes
		);
	}

	#[tokio::test]
	async fn header_idle_timeout_is_classified_as_download_idle() {
		let server = loopback_server(LoopbackBehavior::StallHeaders).await;
		let client = loopback_client(&server.endpoint);
		let staging = download_staging();
		let (plan, remote) = test_download_plan("config.json", b"abcdef");

		let error = client
			.download_file(&plan, &remote, staging.path(), callbacks(None, None))
			.await
			.expect_err("idle headers");

		assert!(matches!(error, HubError::DownloadIdleTimeout { .. }));
		assert!(!staging.path().join("config.json").exists());
		assert!(!staging.path().join("config.json.part").exists());
		server.task.abort();
	}

	#[tokio::test]
	async fn cancellation_interrupts_stalled_response_headers() {
		let server = loopback_server(LoopbackBehavior::StallHeaders).await;
		let client = loopback_client(&server.endpoint);
		let staging = download_staging();
		let (plan, remote) = test_download_plan("config.json", b"abcdef");
		let cancellation = DownloadCancellation::default();
		let cancel = cancellation.clone();
		let cancel_task = tokio::spawn(async move {
			tokio::time::sleep(Duration::from_millis(50)).await;
			cancel.cancel();
		});

		let error = client
			.download_file(
				&plan,
				&remote,
				staging.path(),
				callbacks(None, Some(&cancellation)),
			)
			.await
			.expect_err("stalled headers cancelled");

		assert!(matches!(error, HubError::Cancelled));
		assert!(!staging.path().join("config.json").exists());
		cancel_task.await.expect("cancellation task");
		server.task.abort();
	}

	#[tokio::test]
	async fn per_chunk_idle_timeout_preserves_only_partial_state() {
		let server = loopback_server(LoopbackBehavior::StallBody {
			status: "200 OK",
			headers: vec!["Content-Length: 6".to_string()],
			prefix: b"abc".to_vec(),
		})
		.await;
		let client = loopback_client(&server.endpoint);
		let staging = download_staging();
		let (plan, remote) = test_download_plan("config.json", b"abcdef");

		let error = client
			.download_file(&plan, &remote, staging.path(), callbacks(None, None))
			.await
			.expect_err("idle body");

		assert!(matches!(error, HubError::DownloadIdleTimeout { .. }));
		assert!(!staging.path().join("config.json").exists());
		assert_eq!(
			std::fs::read(staging.path().join("config.json.part")).expect("partial bytes"),
			b"abc"
		);
		server.task.abort();
	}

	#[tokio::test]
	async fn error_body_idle_timeout_is_classified_as_download_idle() {
		let server = loopback_server(LoopbackBehavior::StallBody {
			status: "500 Internal Server Error",
			headers: vec!["Content-Length: 6".to_string()],
			prefix: b"abc".to_vec(),
		})
		.await;
		let client = loopback_client(&server.endpoint);
		let staging = download_staging();
		let (plan, remote) = test_download_plan("config.json", b"abcdef");

		let error = client
			.download_file(&plan, &remote, staging.path(), callbacks(None, None))
			.await
			.expect_err("idle error body");

		assert!(matches!(error, HubError::DownloadIdleTimeout { .. }));
		assert!(!staging.path().join("config.json").exists());
		assert!(!staging.path().join("config.json.part").exists());
		server.task.abort();
	}

	#[tokio::test]
	async fn observer_error_stops_before_verified_publication() {
		let bytes = b"abcdef";
		let server = loopback_server(LoopbackBehavior::Response {
			status: "200 OK",
			headers: vec!["Content-Length: 6".to_string()],
			body: bytes.to_vec(),
		})
		.await;
		let client = loopback_client(&server.endpoint);
		let staging = download_staging();
		let (plan, remote) = test_download_plan("config.json", bytes);
		let observer: DownloadObserver = Arc::new(|event| {
			if matches!(event, DownloadEvent::Progress { .. }) {
				Err(DownloadObserverError::new("observer stopped").expect("observer error"))
			} else {
				Ok(DownloadControl::Continue)
			}
		});

		let error = client
			.download_file(
				&plan,
				&remote,
				staging.path(),
				callbacks(Some(&observer), None),
			)
			.await
			.expect_err("observer failure");

		assert!(
			matches!(error, HubError::Observer(message) if message.contains("observer stopped"))
		);
		assert!(!staging.path().join("config.json").exists());
		assert_eq!(
			std::fs::read(staging.path().join("config.json.part")).expect("partial bytes"),
			bytes
		);
		server.task.await.expect("server task");
	}

	#[tokio::test]
	async fn verified_partial_mutation_is_detected_before_publication() {
		let bytes = b"abcdef";
		let server = loopback_server(LoopbackBehavior::Response {
			status: "200 OK",
			headers: vec!["Content-Length: 6".to_string()],
			body: bytes.to_vec(),
		})
		.await;
		let client = loopback_client(&server.endpoint);
		let staging = download_staging();
		let part = staging.path().join("config.json.part");
		let mutate = part.clone();
		let observer: DownloadObserver = Arc::new(move |event| {
			if matches!(event, DownloadEvent::FileVerified { .. }) {
				std::fs::write(&mutate, b"ABCDEF").expect("mutate verified partial");
			}
			Ok(DownloadControl::Continue)
		});
		let (plan, remote) = test_download_plan("config.json", bytes);

		let error = client
			.download_file(
				&plan,
				&remote,
				staging.path(),
				callbacks(Some(&observer), None),
			)
			.await
			.expect_err("mutation detected");

		assert!(matches!(
			error,
			HubError::Protocol(message) if message.contains("changed before publication")
		));
		assert!(!staging.path().join("config.json").exists());
		assert_eq!(std::fs::read(part).expect("mutated partial"), b"ABCDEF");
		server.task.await.expect("server task");
	}

	#[tokio::test]
	async fn oversized_transfer_does_not_clobber_existing_destination() {
		let server = loopback_server(LoopbackBehavior::Response {
			status: "200 OK",
			headers: vec!["Content-Length: 7".to_string()],
			body: b"abcdefg".to_vec(),
		})
		.await;
		let client = loopback_client(&server.endpoint);
		let staging = download_staging();
		let destination = staging.path().join("config.json");
		std::fs::write(&destination, b"keep").expect("existing destination");
		let (plan, remote) = test_download_plan("config.json", b"abcdef");

		let error = client
			.download_file(&plan, &remote, staging.path(), callbacks(None, None))
			.await
			.expect_err("oversized transfer");

		assert!(matches!(
			error,
			HubError::Size {
				expected: 6,
				actual: 7,
				..
			}
		));
		assert_eq!(std::fs::read(destination).expect("existing file"), b"keep");
		server.task.await.expect("server task");
	}

	#[tokio::test]
	async fn verified_rename_never_clobbers_existing_destination() {
		let bytes = b"abcdef";
		let server = loopback_server(LoopbackBehavior::Response {
			status: "200 OK",
			headers: vec!["Content-Length: 6".to_string()],
			body: bytes.to_vec(),
		})
		.await;
		let client = loopback_client(&server.endpoint);
		let staging = download_staging();
		let destination = staging.path().join("config.json");
		std::fs::write(&destination, b"keep").expect("existing destination");
		let (plan, remote) = test_download_plan("config.json", bytes);

		assert!(
			client
				.download_file(&plan, &remote, staging.path(), callbacks(None, None))
				.await
				.is_err()
		);
		assert_eq!(std::fs::read(destination).expect("existing file"), b"keep");
		assert_eq!(
			std::fs::read(staging.path().join("config.json.part")).expect("verified partial"),
			bytes
		);
		server.task.await.expect("server task");
	}

	#[test]
	fn staging_validation_rejects_nonempty_and_acl_directories() {
		let empty = download_staging();
		validate_download_staging(empty.path()).expect("empty secure staging");
		std::fs::write(empty.path().join("unexpected"), b"x").expect("unexpected file");
		assert!(matches!(
			validate_download_staging(empty.path()),
			Err(HubError::Protocol(message)) if message.contains("must be empty")
		));

		let acl = download_staging();
		let output = Command::new("/bin/chmod")
			.args([
				"+a",
				"everyone allow list,search,readattr,readextattr,readsecurity",
			])
			.arg(acl.path())
			.output()
			.expect("set staging ACL");
		assert!(
			output.status.success(),
			"chmod ACL failed: {}",
			String::from_utf8_lossy(&output.stderr)
		);
		assert!(matches!(
			validate_download_staging(acl.path()),
			Err(HubError::Io { source, .. })
				if source.kind() == std::io::ErrorKind::PermissionDenied
		));
	}

	fn model() -> HubModel {
		let traits = ModelTraits {
			mlx: true,
			..ModelTraits::default()
		};
		HubModel {
			id: HubModelId::parse("mlx-community/example").expect("valid model ID"),
			revision: ResolvedRevision::parse("a".repeat(40)).expect("valid revision"),
			downloads: 0,
			likes: 0,
			tags: vec!["mlx".to_string()],
			library: Some("mlx".to_string()),
			license: None,
			traits,
			quantization: HubQuantization::Unknown,
			compatible: true,
			files: vec![
				"config.json".to_string(),
				"tokenizer.json".to_string(),
				"model.safetensors".to_string(),
			],
			diagnostics: Vec::new(),
			fit: None,
			download_bytes: None,
		}
	}

	fn files() -> Vec<RemoteFile> {
		vec![
			RemoteFile::new("config.json".to_string(), 1, None).expect("config"),
			RemoteFile::new("tokenizer.json".to_string(), 1, None).expect("tokenizer"),
			RemoteFile::new("model.safetensors".to_string(), 1, None).expect("weights"),
		]
	}

	fn wire() -> ModelWire {
		ModelWire {
			id: "mlx-community/example".to_string(),
			private: false,
			gated: serde_json::Value::Null,
			sha: Some("a".repeat(40)),
			downloads: 7,
			likes: 3,
			tags: vec!["mlx".to_string()],
			library_name: Some("mlx".to_string()),
			pipeline_tag: Some("text-generation".to_string()),
			siblings: vec![
				SiblingWire {
					rfilename: "config.json".to_string(),
				},
				SiblingWire {
					rfilename: "generation_config.json".to_string(),
				},
				SiblingWire {
					rfilename: "tokenizer.json".to_string(),
				},
				SiblingWire {
					rfilename: "model.safetensors".to_string(),
				},
			],
		}
	}

	fn artifacts() -> RevisionArtifacts {
		RevisionArtifacts {
			config: serde_json::json!({
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
				"linear_conv_kernel_dim": 4,
				"max_position_embeddings": 32_768
			}),
			tokenizer_config: None,
			processor_config: None,
			generation_config: Some(serde_json::json!({
				"do_sample": true,
				"temperature": 0.7,
				"top_p": 0.9,
				"top_k": 40,
				"max_new_tokens": 512
			})),
			legacy_chat_template: None,
			chat_template: Some(
				"{% for message in messages %}{{ message.content }}{% endfor %}".to_string(),
			),
			tool_chat_template: None,
			index: None,
		}
	}

	fn enrichment_files() -> Vec<RemoteFile> {
		vec![
			RemoteFile::new("config.json".to_string(), 2_048, None).expect("config"),
			RemoteFile::new("generation_config.json".to_string(), 128, None)
				.expect("generation config"),
			RemoteFile::new("model.safetensors".to_string(), 1 << 20, None).expect("weights"),
			RemoteFile::new("tokenizer.json".to_string(), 4_096, None).expect("tokenizer"),
		]
	}

	fn enriched_with_template(template: &str) -> HubModel {
		let mut model = model();
		let mut artifacts = artifacts();
		artifacts.chat_template = Some(template.to_string());
		enrich_remote_model(&mut model, &wire(), &artifacts, &enrichment_files(), None)
			.expect("remote model enrichment");
		model
	}

	#[test]
	fn remote_capability_probe_ignores_inert_template_keywords() {
		let model = enriched_with_template(
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
				model.traits.tasks.contains(&Task::ToolUse),
				model.traits.tasks.contains(&Task::Reasoning),
			),
			(false, false)
		);
	}

	#[test]
	fn remote_capability_probe_accepts_semantic_template_fixture() {
		let model = enriched_with_template(
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
				model.traits.tasks.contains(&Task::ToolUse),
				model.traits.tasks.contains(&Task::Reasoning),
			),
			(true, true)
		);
	}

	#[test]
	fn remote_file_deserialization_rejects_traversal() {
		let json = r#"{"path":"../escape.safetensors","size":1,"expected_sha256":null}"#;
		assert!(serde_json::from_str::<RemoteFile>(json).is_err());
	}

	#[test]
	fn download_plan_rejects_duplicate_paths() {
		let mut duplicate = files();
		duplicate
			.push(RemoteFile::new("config.json".to_string(), 1, None).expect("duplicate config"));
		assert!(DownloadPlan::new(model(), duplicate).is_err());
	}

	#[test]
	fn exact_plan_uses_only_indexed_shards() {
		let tree = vec![
			tree_file("config.json", 10),
			tree_file("tokenizer.json", 10),
			tree_file("model.safetensors.index.json", 10),
			tree_file("model-00001.safetensors", 100),
		];
		let index = serde_json::json!({"weight_map": {"model.weight": "model-00001.safetensors"}});
		let plan = exact_runtime_plan(&tree, Some(&index), false).expect("exact indexed plan");
		let names = plan.iter().map(RemoteFile::path).collect::<BTreeSet<_>>();
		assert_eq!(
			names,
			BTreeSet::from([
				"config.json",
				"model-00001.safetensors",
				"model.safetensors.index.json",
				"tokenizer.json",
			])
		);
	}

	#[test]
	fn exact_plan_rejects_unindexed_adapter() {
		let tree = vec![
			tree_file("config.json", 10),
			tree_file("tokenizer.json", 10),
			tree_file("model.safetensors.index.json", 10),
			tree_file("model-00001.safetensors", 100),
			tree_file("adapter.safetensors", 20),
		];
		let index = serde_json::json!({"weight_map": {"model.weight": "model-00001.safetensors"}});
		assert!(matches!(
			exact_runtime_plan(&tree, Some(&index), false),
			Err(HubError::Incompatible(_))
		));
	}

	#[test]
	fn exact_plan_rejects_duplicate_root_paths() {
		let tree = vec![
			tree_file("config.json", 10),
			tree_file("config.json", 10),
			tree_file("tokenizer.json", 10),
			tree_file("model.safetensors", 100),
		];
		assert!(matches!(
			exact_runtime_plan(&tree, None, false),
			Err(HubError::Protocol(_))
		));
	}

	#[test]
	fn exact_plan_preserves_legacy_chat_template_json() {
		let mut tree = basic_runtime_tree();
		tree.push(tree_file("chat_template.json", 100));
		let plan = exact_runtime_plan(&tree, None, false).expect("legacy template plan");
		assert!(plan.iter().any(|file| file.path() == "chat_template.json"));
		assert_eq!(
			local_runtime_path("chat_template.json"),
			"chat_template.json"
		);
	}

	#[test]
	fn exact_plan_normalizes_current_named_templates_without_collision() {
		let mut tree = basic_runtime_tree();
		tree.extend([
			tree_file("additional_chat_templates/default.jinja", 100),
			tree_file("additional_chat_templates/tool_use.jinja", 100),
		]);
		let plan = exact_runtime_plan(&tree, None, false).expect("named templates");
		let normalized = plan
			.iter()
			.map(|file| local_runtime_path(file.path()))
			.collect::<BTreeSet<_>>();
		assert!(normalized.contains("chat_template.jinja"));
		assert!(normalized.contains("chat_template_tool_use.jinja"));
	}

	#[test]
	fn exact_plan_rejects_ambiguous_named_template_layouts() {
		let mut root_and_named = basic_runtime_tree();
		root_and_named.extend([
			tree_file("chat_template.jinja", 100),
			tree_file("additional_chat_templates/default.jinja", 100),
		]);
		assert!(matches!(
			exact_runtime_plan(&root_and_named, None, false),
			Err(HubError::Incompatible(_))
		));

		let mut current_and_legacy = basic_runtime_tree();
		current_and_legacy.extend([
			tree_file("additional_chat_templates/default.jinja", 100),
			tree_file("chat_templates/default.jinja", 100),
		]);
		assert!(matches!(
			exact_runtime_plan(&current_and_legacy, None, false),
			Err(HubError::Incompatible(_))
		));
	}

	#[test]
	fn processor_template_excludes_external_template_files() {
		let mut tree = basic_runtime_tree();
		tree.extend([
			tree_file("processor_config.json", 100),
			tree_file("chat_template.jinja", 100),
			tree_file("additional_chat_templates/tool_use.jinja", 100),
		]);
		let plan = exact_runtime_plan(&tree, None, true).expect("processor template plan");
		assert!(
			plan.iter()
				.any(|file| file.path() == "processor_config.json")
		);
		assert!(
			plan.iter()
				.all(|file| !file.path().contains("chat_template"))
		);
	}

	fn basic_runtime_tree() -> Vec<TreeWire> {
		vec![
			tree_file("config.json", 10),
			tree_file("tokenizer.json", 10),
			tree_file("model.safetensors", 100),
		]
	}

	fn tree_file(path: &str, size: u64) -> TreeWire {
		TreeWire {
			kind: "file".to_string(),
			path: path.to_string(),
			size,
			lfs: None,
			security: None,
		}
	}

	#[test]
	fn partial_response_range_must_match_request_and_total() {
		let mut headers = header::HeaderMap::new();
		headers.insert(
			header::CONTENT_RANGE,
			header::HeaderValue::from_static("bytes 5-9/10"),
		);
		headers.insert(
			header::CONTENT_LENGTH,
			header::HeaderValue::from_static("5"),
		);
		assert!(validate_content_range(&headers, 5, 10).is_ok());
		assert!(validate_content_range(&headers, 4, 10).is_err());
		headers.insert(
			header::CONTENT_RANGE,
			header::HeaderValue::from_static("bytes 5-9/*"),
		);
		assert!(validate_content_range(&headers, 5, 10).is_err());
	}

	#[test]
	fn direct_client_rejects_invalid_bounds() {
		let config = HubConfig {
			results: 0,
			..HubConfig::default()
		};
		assert!(HubClient::new(config).is_err());
	}

	#[test]
	fn upstream_cursor_is_bounded_printable_opaque_data() {
		for cursor in ["eyJwayI6IjEifQ==", "+/8=", "future:~.$&?"] {
			validate_upstream_cursor(cursor).expect("supported opaque cursor");
		}
		for cursor in ["", "cursor with space", "line\nbreak", "\u{7f}"] {
			assert!(validate_upstream_cursor(cursor).is_err(), "{cursor:?}");
		}
		assert!(validate_upstream_cursor(&"a".repeat(MAX_UPSTREAM_CURSOR_BYTES + 1)).is_err());

		let mut headers = header::HeaderMap::new();
		headers.insert(
			header::LINK,
			header::HeaderValue::from_static(
				"<https://huggingface.co/api/models?limit=1&cursor=eyJwayI6IjEifQ%3D%3D>; \
					 rel=\"next\"",
			),
		);
		assert_eq!(
			next_upstream_cursor(&headers).expect("decode Link cursor"),
			Some("eyJwayI6IjEifQ==".to_string())
		);

		headers.insert(
			header::LINK,
			header::HeaderValue::from_static(
				"<https://huggingface.co/api/models?cursor=%2B%2F8%3D>; rel=\"next\"",
			),
		);
		assert_eq!(
			next_upstream_cursor(&headers).expect("decode standard Base64 cursor"),
			Some("+/8=".to_string())
		);
	}

	#[tokio::test]
	async fn search_round_trips_padded_upstream_cursor() {
		let body = b"[]".to_vec();
		let first = LoopbackBehavior::Response {
			status: "200 OK",
			headers: vec![
				"Content-Type: application/json".to_string(),
				format!("Content-Length: {}", body.len()),
				"Link: <https://huggingface.co/api/models?limit=1&cursor=eyJwayI6IjEifQ%3D%3D>; \
					 rel=\"next\""
					.to_string(),
			],
			body: body.clone(),
		};
		let second = LoopbackBehavior::Response {
			status: "200 OK",
			headers: vec![
				"Content-Type: application/json".to_string(),
				format!("Content-Length: {}", body.len()),
			],
			body,
		};
		let server = loopback_sequence_server(vec![first, second]).await;
		let client = loopback_client(&server.endpoint);
		let first_page = client
			.search(&HubSearch::default())
			.await
			.expect("first search page");
		let resumed = HubSearch {
			cursor: first_page.next_cursor,
			..HubSearch::default()
		};
		assert!(resumed.cursor.is_some());
		client.search(&resumed).await.expect("resumed search page");

		let requests = server.task.await.expect("pagination server");
		assert_eq!(requests.len(), 2);
		let request = String::from_utf8(requests[1].clone()).expect("UTF-8 request");
		let target = request.split_whitespace().nth(1).expect("request target");
		let url = Url::parse(&format!("http://loopback{target}")).expect("request URL");
		let cursor = url
			.query_pairs()
			.find_map(|(key, value)| (key == "cursor").then(|| value.into_owned()));
		assert_eq!(cursor.as_deref(), Some("eyJwayI6IjEifQ=="));
	}

	#[tokio::test]
	async fn mlx_catalog_filter_composes_with_optional_search_text() {
		for expected_search in [None, Some("qwen")] {
			let server = loopback_server(json_response("200 OK", &serde_json::json!([]))).await;
			let client = loopback_client(&server.endpoint);
			let search = expected_search.map_or_else(
				|| HubSearch::default().mlx_library(),
				|query| HubSearch::default().mlx_library().query(query),
			);
			client.search(&search).await.expect("MLX catalog search");

			let request = String::from_utf8(server.task.await.expect("search server"))
				.expect("UTF-8 request");
			let target = request.split_whitespace().nth(1).expect("request target");
			let url = Url::parse(&format!("http://loopback{target}")).expect("request URL");
			let query = url.query_pairs().collect::<BTreeMap<_, _>>();
			assert_eq!(query.get("filter").map(AsRef::as_ref), Some("mlx"));
			assert_eq!(query.get("search").map(AsRef::as_ref), expected_search);
		}
	}

	#[tokio::test]
	async fn search_propagates_metadata_preflight_service_failures() {
		for (status, expected) in [
			("429 Too Many Requests", StatusCode::TOO_MANY_REQUESTS),
			(
				"500 Internal Server Error",
				StatusCode::INTERNAL_SERVER_ERROR,
			),
		] {
			let candidate = candidate_wire_value();
			let server = loopback_sequence_server(vec![
				json_response("200 OK", &serde_json::json!([candidate])),
				json_response(status, &serde_json::json!({"error": "unavailable"})),
			])
			.await;
			let client = loopback_client(&server.endpoint);
			let error = client
				.search(&HubSearch::default())
				.await
				.expect_err("metadata service failure must abort search");
			assert!(matches!(
				error,
				HubError::Http { status, .. } if status == expected
			));
			assert_eq!(server.task.await.expect("service failure server").len(), 2);
		}

		let candidate = candidate_wire_value();
		let server = loopback_sequence_server(vec![
			json_response("200 OK", &serde_json::json!([candidate])),
			LoopbackBehavior::Disconnect,
		])
		.await;
		let client = loopback_client(&server.endpoint);
		assert!(matches!(
			client
				.search(&HubSearch::default())
				.await
				.expect_err("metadata disconnect must abort search"),
			HubError::Request(_)
		));
		assert_eq!(server.task.await.expect("disconnect server").len(), 2);
	}

	#[tokio::test]
	async fn search_retains_candidate_incompatibility_as_diagnostic() {
		let candidate = candidate_wire_value();
		let tree = serde_json::json!([
			{"type": "file", "path": "config.json", "size": 2},
			{"type": "file", "path": "model.safetensors", "size": 1}
		]);
		let server = loopback_sequence_server(vec![
			json_response("200 OK", &serde_json::json!([candidate.clone()])),
			json_response("200 OK", &candidate),
			json_response("200 OK", &tree),
			json_response("200 OK", &serde_json::json!({})),
		])
		.await;
		let client = loopback_client(&server.endpoint);
		let page = client
			.search(&HubSearch::default())
			.await
			.expect("candidate incompatibility must not abort search");
		assert!(page.items.is_empty());
		assert_eq!(page.scanned, 1);
		assert!(page.diagnostics.iter().any(|diagnostic| {
			diagnostic
				.message
				.contains("repository lacks config.json or tokenizer.json")
		}));
		assert_eq!(server.task.await.expect("candidate server").len(), 4);
	}

	#[test]
	fn composite_cursor_reaches_compatible_results_after_first_twenty() {
		let search = HubSearch::default();
		let candidates = (0..25)
			.map(|rank| {
				let mut candidate = model();
				candidate.id =
					HubModelId::parse(format!("owner/model-{rank:02}")).expect("candidate ID");
				(rank, candidate)
			})
			.collect::<Vec<_>>();
		let mut first = Vec::new();
		let offset = append_compatible_page(&mut first, candidates.iter().cloned(), 20)
			.expect("continuation");
		let cursor = continuation_cursor(&search, None, None, None, offset, candidates.len(), None)
			.expect("encode cursor")
			.expect("next page");
		assert_eq!(first.len(), 20);

		let resumed = HubSearch {
			cursor: Some(cursor),
			..HubSearch::default()
		};
		let position = decode_search_cursor(&resumed, None, None).expect("decode cursor");
		assert_eq!(position.offset, 20);
		let mut second = Vec::new();
		assert_eq!(
			append_compatible_page(
				&mut second,
				candidates.into_iter().skip(position.offset),
				20
			),
			None
		);
		assert_eq!(second.len(), 5);
	}

	#[test]
	fn scanned_count_excludes_metadata_candidates_not_yet_processed() {
		let candidate_count = 200;
		let immediately_scanned = 3;
		let metadata_scanned = 20;
		assert_eq!(search_scanned(immediately_scanned, metadata_scanned), 23);
		assert_ne!(
			search_scanned(immediately_scanned, metadata_scanned),
			candidate_count
		);
	}

	#[test]
	fn composite_cursor_is_scoped_to_query_and_filters() {
		let first = HubSearch {
			query: Some("qwen".to_string()),
			..HubSearch::default()
		};
		let cursor = encode_search_cursor(&first, None, None, Some("upstream".to_string()), 7)
			.expect("encode cursor");
		let changed = HubSearch {
			query: Some("gemma".to_string()),
			cursor: Some(cursor),
			..HubSearch::default()
		};
		assert!(decode_search_cursor(&changed, None, None).is_err());
	}

	#[test]
	fn composite_cursor_is_scoped_to_mlx_catalog_filter() {
		let first = HubSearch::default().mlx_library();
		let cursor = encode_search_cursor(&first, None, None, Some("upstream".to_string()), 7)
			.expect("encode cursor");

		assert!(
			decode_search_cursor(&HubSearch::default().cursor(cursor.clone()), None, None).is_err()
		);
		assert!(
			decode_search_cursor(
				&HubSearch::default().mlx_library().cursor(cursor),
				None,
				None
			)
			.is_ok()
		);
	}

	#[test]
	fn composite_cursor_uses_the_normalized_upstream_query() {
		let first = HubSearch::default().query("  gemma  ");
		let cursor =
			encode_search_cursor(&first, None, None, None, 7).expect("encode normalized cursor");
		let resumed = HubSearch::default().query("gemma").cursor(cursor);
		let position =
			decode_search_cursor(&resumed, None, None).expect("decode normalized cursor");
		assert_eq!(position.offset, 7);
		assert_eq!(
			normalized_search_query(&first).expect("normalize first"),
			normalized_search_query(&resumed).expect("normalize resumed")
		);
	}

	#[test]
	fn search_query_rejects_controls_and_oversize_input() {
		assert!(search_scope(&HubSearch::default().query("gemma\nmodel"), None, None).is_err());
		assert!(
			search_scope(
				&HubSearch::default().query("x".repeat(MAX_SEARCH_QUERY_BYTES + 1)),
				None,
				None
			)
			.is_err()
		);
	}

	#[test]
	fn composite_cursor_is_scoped_to_credential_identity() {
		let search = HubSearch::default();
		let first = HubCredentials::bearer_token("hf_first").expect("first credential");
		let second = HubCredentials::bearer_token("hf_second").expect("second credential");
		let cursor = encode_search_cursor(
			&search,
			Some(&first.scope),
			None,
			Some("upstream".to_string()),
			7,
		)
		.expect("encode cursor");
		let resumed = HubSearch::default().cursor(cursor);
		assert!(decode_search_cursor(&resumed, Some(&second.scope), None).is_err());
		assert!(decode_search_cursor(&resumed, None, None).is_err());
	}

	#[test]
	fn composite_cursor_is_scoped_to_workload_and_metal_budget() {
		let search = HubSearch::default();
		let workload = WorkloadProfile::new(1, 4_096).expect("first workload");
		let profile = Some((workload, 8_u64 << 30));
		let cursor =
			encode_search_cursor(&search, None, profile, None, 7).expect("encode profiled cursor");
		let resumed = HubSearch::default().cursor(cursor);

		assert!(decode_search_cursor(&resumed, None, profile).is_ok());
		assert!(
			decode_search_cursor(
				&resumed,
				None,
				Some((
					WorkloadProfile::new(2, 4_096).expect("second workload"),
					8_u64 << 30,
				)),
			)
			.is_err()
		);
		assert!(decode_search_cursor(&resumed, None, Some((workload, 4_u64 << 30))).is_err());
		assert!(decode_search_cursor(&resumed, None, None).is_err());
	}

	#[test]
	fn remote_search_rejects_filters_without_remote_evidence() {
		for value in [
			"input:video",
			"confidence:advertised:input:video",
			"output:video",
			"confidence:advertised:output:video",
			"mtp_stage>=layout_validated",
		] {
			assert!(TraitFilter::parse(value).is_err(), "{value}");
		}
		for value in [
			"interaction:structured_output",
			"confidence:advertised:interaction:structured_output",
			"acceleration:mtp",
			"confidence:advertised:acceleration:mtp",
			"output:image",
			"confidence:advertised:output:image",
			"output:audio",
			"confidence:advertised:output:audio",
			"extension:future-capability",
			"confidence:advertised:extension:future-capability",
			"mtp_stage>=runtime_verified",
			"confidence:runtime_verified:acceleration:mlx",
		] {
			let filter = TraitFilter::parse(value).expect("known trait filter");
			assert!(validate_remote_filters(&[filter]).is_err(), "{value}");
		}
		let advertised =
			TraitFilter::parse("acceleration:mtp_advertised").expect("advertised MTP filter");
		assert!(validate_remote_filters(&[advertised]).is_ok());
		let advertised_stage =
			TraitFilter::parse("mtp_stage>=advertised").expect("advertised MTP stage");
		assert!(validate_remote_filters(&[advertised_stage]).is_ok());
	}

	#[test]
	fn remote_filter_help_examples_match_validator_contract() {
		for help in REMOTE_FILTERS {
			let filter = TraitFilter::parse(help.example).expect("documented remote filter");
			assert!(
				validate_remote_filters(&[filter]).is_ok(),
				"{}",
				help.filter
			);
		}
		for key in [
			"input:text",
			"output:text",
			"task:text_generation",
			"task:chat",
			"interaction:system_prompt",
			"interaction:tools",
			"interaction:reasoning",
			"interaction:reasoning_history",
			"interaction:thinking_toggle",
			"acceleration:mlx",
			"acceleration:mtp_advertised",
			"extension:huggingface.advertised_input_image",
			"extension:huggingface.advertised_input_audio",
		] {
			assert!(
				REMOTE_FILTERS.iter().any(|help| help.filter == key),
				"missing help for {key}"
			);
			let confidence = TraitFilter::parse(format!("confidence:advertised:{key}"))
				.expect("confidence filter");
			assert!(validate_remote_filters(&[confidence]).is_ok(), "{key}");
		}
	}

	#[test]
	fn advertised_input_modalities_have_evidence_without_directional_overclaim() {
		let mut image = wire();
		image.pipeline_tag = Some("image-text-to-text".to_string());
		let image_traits = traits_from_wire(&image, &[]);
		let image_key = "extension:huggingface.advertised_input_image";
		assert!(image_traits.extras.contains_key(image_key));
		assert_eq!(
			image_traits.confidence.get(image_key),
			Some(&TraitConfidence::Advertised)
		);
		assert!(
			image_traits
				.evidence
				.iter()
				.any(|evidence| evidence.trait_key == image_key)
		);

		let audio_key = "extension:huggingface.advertised_input_audio";
		for pipeline in [
			"automatic-speech-recognition",
			"zero-shot-audio-classification",
		] {
			let mut audio = wire();
			audio.pipeline_tag = Some(pipeline.to_string());
			let audio_traits = traits_from_wire(&audio, &[]);
			assert!(audio_traits.extras.contains_key(audio_key), "{pipeline}");
			assert_eq!(
				audio_traits.confidence.get(audio_key),
				Some(&TraitConfidence::Advertised),
				"{pipeline}"
			);
			assert!(
				audio_traits
					.evidence
					.iter()
					.any(|evidence| evidence.trait_key == audio_key),
				"{pipeline}"
			);
		}

		let mut text_to_audio = wire();
		text_to_audio.pipeline_tag = Some("text-to-audio".to_string());
		text_to_audio.tags = vec!["text-to-audio".to_string(), "audio".to_string()];
		let output_only = traits_from_wire(&text_to_audio, &[]);
		assert!(!output_only.extras.contains_key(audio_key));
		assert!(!output_only.confidence.contains_key(audio_key));
		assert!(
			output_only
				.evidence
				.iter()
				.all(|evidence| evidence.trait_key != audio_key)
		);
	}

	#[test]
	fn direct_client_is_static_only() {
		let client = HubClient::new(HubConfig::default()).expect("valid static Hub client");
		assert_eq!(client.fit_workload(), None);
		assert_eq!(client.metal_budget_bytes(), None);
		assert!(!client.is_authenticated());
	}

	#[test]
	fn credentials_are_secret_sensitive_redacted_and_reject_controls() {
		let credentials = HubCredentials::bearer_token("hf_example").expect("valid token");
		assert!(credentials.authorization.is_sensitive());
		assert_eq!(
			credentials.authorization.to_str().expect("header text"),
			"Bearer hf_example"
		);
		assert!(!format!("{credentials:?}").contains("hf_example"));
		assert!(HubCredentials::bearer_token("secret\nvalue").is_err());
		assert!(HubCredentials::bearer_token("").is_err());
	}

	#[tokio::test]
	async fn transport_errors_strip_signed_urls() {
		let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
			.await
			.expect("bind loopback");
		let address = listener.local_addr().expect("loopback address");
		let close = tokio::spawn(async move {
			let (socket, _) = listener.accept().await.expect("accept loopback");
			drop(socket);
		});
		let secret = "signed-query-must-not-escape";
		let url = format!("http://{address}/model?X-Amz-Signature={secret}");
		let raw = reqwest::Client::builder()
			.no_proxy()
			.timeout(Duration::from_secs(1))
			.build()
			.expect("test client")
			.get(&url)
			.send()
			.await
			.expect_err("closed connection");
		close.await.expect("close task");
		assert!(
			raw.url()
				.is_some_and(|request_url| request_url.as_str().contains(secret))
		);

		let error = request_error(raw);
		assert!(!error.to_string().contains(secret));
		assert!(!format!("{error:?}").contains(secret));
		assert!(matches!(error, HubError::Request(source) if source.url().is_none()));
	}

	#[tokio::test]
	async fn http_error_bodies_cannot_echo_redirect_or_bearer_credentials() {
		let signed_secret = "signed-query-must-not-escape";
		let signed_body = format!("canonical request contains {signed_secret}").into_bytes();
		let destination = loopback_server(LoopbackBehavior::Response {
			status: "403 Forbidden",
			headers: vec![format!("Content-Length: {}", signed_body.len())],
			body: signed_body,
		})
		.await;
		let redirect_url = format!(
			"{}/blob?X-Amz-Signature={signed_secret}",
			destination.endpoint
		);
		let redirect = loopback_server(LoopbackBehavior::Response {
			status: "302 Found",
			headers: vec![
				"Content-Length: 0".to_string(),
				format!("Location: {redirect_url}"),
			],
			body: Vec::new(),
		})
		.await;
		let client = loopback_client(&redirect.endpoint);
		let id = HubModelId::parse("owner/model").expect("model ID");
		let revision = ResolvedRevision::parse("a".repeat(40)).expect("revision");

		let error = client
			.fetch_revision_bytes(&id, &revision, "config.json", 1024)
			.await
			.expect_err("redirected HTTP error");
		assert!(!error.to_string().contains(signed_secret));
		assert!(!format!("{error:?}").contains(signed_secret));
		assert!(matches!(&error, HubError::Http { body, .. } if body.contains("suppressed")));
		redirect.task.await.expect("redirect task");
		let destination_request = destination.task.await.expect("destination task");
		assert!(String::from_utf8_lossy(&destination_request).contains(signed_secret));

		let bearer_secret = "bearer-test-secret-must-not-escape";
		let bearer_body = format!("authorization was Bearer {bearer_secret}").into_bytes();
		let authenticated_server = loopback_server(LoopbackBehavior::Response {
			status: "403 Forbidden",
			headers: vec![format!("Content-Length: {}", bearer_body.len())],
			body: bearer_body,
		})
		.await;
		let config = HubConfig {
			results: 1,
			scan_limit: 1,
			metadata_concurrency: 1,
			request_timeout_seconds: 1,
			retries: 0,
		};
		let credentials = HubCredentials::bearer_token(bearer_secret).expect("credentials");
		let authenticated = HubClient::with_endpoint_and_fit(
			config,
			&authenticated_server.endpoint,
			None,
			Some(credentials),
		)
		.expect("authenticated client");
		let error = authenticated
			.fetch_revision_bytes(&id, &revision, "config.json", 1024)
			.await
			.expect_err("authenticated HTTP error");
		assert!(!error.to_string().contains(bearer_secret));
		assert!(!format!("{error:?}").contains(bearer_secret));
		assert!(matches!(&error, HubError::Http { body, .. } if body.contains("suppressed")));
		let authenticated_request = authenticated_server.task.await.expect("authenticated task");
		assert!(
			String::from_utf8_lossy(&authenticated_request)
				.contains(&format!("Bearer {bearer_secret}"))
		);
	}

	#[tokio::test]
	async fn redirect_chain_never_forwards_signed_url_as_referer() {
		let secret = "signed-referer-must-not-escape";
		let final_server = loopback_server(LoopbackBehavior::Response {
			status: "403 Forbidden",
			headers: vec!["Content-Length: 0".to_string()],
			body: Vec::new(),
		})
		.await;
		let middle = loopback_server(LoopbackBehavior::Response {
			status: "302 Found",
			headers: vec![
				"Content-Length: 0".to_string(),
				format!("Location: {}/final", final_server.endpoint),
			],
			body: Vec::new(),
		})
		.await;
		let initial = loopback_server(LoopbackBehavior::Response {
			status: "302 Found",
			headers: vec![
				"Content-Length: 0".to_string(),
				format!(
					"Location: {}/signed?X-Amz-Signature={secret}",
					middle.endpoint
				),
			],
			body: Vec::new(),
		})
		.await;
		let client = loopback_client(&initial.endpoint);
		let id = HubModelId::parse("owner/model").expect("model ID");
		let revision = ResolvedRevision::parse("a".repeat(40)).expect("revision");

		let error = client
			.fetch_revision_bytes(&id, &revision, "config.json", 1024)
			.await
			.expect_err("final HTTP error");
		assert!(!error.to_string().contains(secret));
		assert!(!format!("{error:?}").contains(secret));
		initial.task.await.expect("initial redirect");
		let middle_request = middle.task.await.expect("signed redirect");
		assert!(String::from_utf8_lossy(&middle_request).contains(secret));
		let final_request = final_server.task.await.expect("final request");
		let final_request = String::from_utf8_lossy(&final_request).to_ascii_lowercase();
		assert!(!final_request.contains("\r\nreferer:"));
		assert!(!final_request.contains(secret));
	}

	#[test]
	fn credentials_are_explicit_per_client() {
		let anonymous = HubClient::new(HubConfig::default()).expect("anonymous client");
		let authenticated = HubClient::with_credentials(
			HubConfig::default(),
			HubCredentials::bearer_token("hf_example").expect("valid token"),
		)
		.expect("authenticated client");
		assert!(!anonymous.is_authenticated());
		assert!(authenticated.is_authenticated());
	}

	#[test]
	fn authenticated_metadata_may_describe_restricted_model() {
		let mut wire = wire();
		wire.private = true;
		assert!(matches!(
			model_from_wire(&wire, false),
			Err(HubError::NotPublic(_))
		));
		assert!(model_from_wire(&wire, true).is_ok());
	}

	#[test]
	fn profiled_client_exposes_exact_fit_inputs() {
		let workload = WorkloadProfile::new(2, 4_096).expect("valid workload");
		let client = HubClient::with_fit_profile(HubConfig::default(), workload, 8_u64 << 30)
			.expect("valid profiled Hub client");
		assert_eq!(client.fit_workload(), Some(workload));
		assert_eq!(client.metal_budget_bytes(), Some(8_u64 << 30));
	}

	#[test]
	fn search_storage_filter_rejects_before_filling_the_ranked_page() {
		let mut fits = model();
		fits.download_bytes = Some(3);
		let mut too_large = model();
		too_large.id = HubModelId::parse("mlx-community/too-large").expect("model ID");
		too_large.download_bytes = Some(4);
		let mut diagnostics = Vec::new();
		let accepted = filter_search_storage(
			vec![(0, too_large), (1, fits)],
			Some((64_u64 << 20) + 3),
			&mut diagnostics,
		);
		let mut page = Vec::new();
		let offset = append_compatible_page(&mut page, accepted, 1);

		assert_eq!(offset, Some(2));
		assert_eq!(page.len(), 1);
		assert_eq!(page[0].id.as_str(), "mlx-community/example");
		assert_eq!(diagnostics.len(), 1);
		assert!(diagnostics[0].message.contains("exceeds available storage"));
	}

	#[test]
	fn download_storage_requirement_preserves_margin_policy() {
		assert_eq!(required_download_storage_bytes(1), Some((64_u64 << 20) + 1));
		assert_eq!(
			required_download_storage_bytes(2_u64 << 30),
			Some((2_u64 << 30) + ((2_u64 << 30) / 20))
		);
		assert_eq!(required_download_storage_bytes(u64::MAX), None);
	}

	#[test]
	fn static_enrichment_reports_compatibility_without_claiming_fit() {
		let wire = wire();
		let mut model = model_from_wire(&wire, false).expect("valid wire model");
		enrich_remote_model(&mut model, &wire, &artifacts(), &enrichment_files(), None)
			.expect("valid static enrichment");

		assert!(model.compatible);
		assert!(model.traits.mlx);
		assert_eq!(model.quantization, HubQuantization::NotConfigured);
		assert_eq!(model.fit, None);
		let sizing = model.traits.sizing.as_ref().expect("static sizing");
		assert_eq!(sizing.weights_bytes, Some(1 << 20));
		assert_eq!(model.download_bytes, Some(1_054_848));
		assert_eq!(sizing.estimated_residency_bytes, None);
		assert_eq!(sizing.evaluated_context_tokens, None);
		assert_eq!(sizing.max_context_tokens, Some(32_768));
		assert_eq!(model.traits.generation_defaults.do_sample, Some(true));
		assert_eq!(model.traits.generation_defaults.temperature, Some(0.7));
		assert_eq!(model.traits.generation_defaults.top_p, Some(0.9));
		assert_eq!(model.traits.generation_defaults.top_k, Some(40));
		assert_eq!(model.traits.generation_defaults.max_new_tokens, Some(512));
		assert!(model.traits.evidence.iter().any(|evidence| {
			evidence.trait_key == "generation:defaults" && evidence.source == EvidenceSource::Config
		}));
		for key in ["input:text", "output:text"] {
			assert_eq!(
				model.traits.confidence.get(key),
				Some(&TraitConfidence::Inferred)
			);
			assert!(
				model
					.traits
					.evidence
					.iter()
					.any(|evidence| evidence.trait_key == key)
			);
		}
	}

	#[test]
	fn static_enrichment_reports_validated_quantization_defaults_and_overrides() {
		let wire = wire();
		let mut artifacts = artifacts();
		artifacts
			.config
			.as_object_mut()
			.expect("config object")
			.insert(
				"quantization".to_string(),
				serde_json::json!({
					"bits": 4,
					"group_size": 64,
					"mode": "affine",
					"model.layers.0.self_attn.q_proj": {
						"bits": 8,
						"group_size": 64,
						"mode": "affine"
					}
				}),
			);
		let mut model = model_from_wire(&wire, false).expect("valid wire model");

		enrich_remote_model(&mut model, &wire, &artifacts, &enrichment_files(), None)
			.expect("valid static enrichment");

		assert_eq!(
			model.quantization,
			HubQuantization::Configured(
				HubQuantizationConfig::new(HubQuantizationMode::Affine, 4, 64, true)
					.expect("valid quantization"),
			)
		);
	}

	#[test]
	fn hub_quantization_deserialization_rejects_unsupported_parameters() {
		let valid = serde_json::json!({
			"kind": "configured",
			"mode": "mxfp4",
			"bits": 4,
			"group_size": 32,
			"has_layer_overrides": true
		});
		let parsed = serde_json::from_value::<HubQuantization>(valid.clone())
			.expect("valid quantization must deserialize");
		assert_eq!(
			serde_json::to_value(parsed).expect("valid quantization must serialize"),
			valid
		);

		let invalid = serde_json::json!({
			"kind": "configured",
			"mode": "affine",
			"bits": 7,
			"group_size": 64,
			"has_layer_overrides": false
		});

		let error = serde_json::from_value::<HubQuantization>(invalid)
			.expect_err("unsupported quantization must not deserialize");

		assert!(error.to_string().contains("unsupported parameters"));
	}

	#[test]
	fn profiled_nonfit_remains_statically_mlx_compatible() {
		let wire = wire();
		let workload = WorkloadProfile::new(1, 4_096).expect("valid workload");
		let mut model = model_from_wire(&wire, false).expect("valid wire model");
		enrich_remote_model(
			&mut model,
			&wire,
			&artifacts(),
			&enrichment_files(),
			Some((workload, 1)),
		)
		.expect("valid profiled enrichment");

		assert!(model.traits.mlx);
		assert!(!model.compatible);
		assert!(model.fit.as_ref().is_some_and(|fit| !fit.fits));
		assert_eq!(
			model
				.traits
				.sizing
				.as_ref()
				.and_then(|sizing| sizing.evaluated_context_tokens),
			Some(workload.context_tokens())
		);
		assert!(
			model
				.diagnostics
				.iter()
				.any(|diagnostic| { diagnostic.contains("exceeds Metal budget") })
		);
	}
}
