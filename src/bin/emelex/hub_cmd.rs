//! Hugging Face discovery and download presentation.

use std::{
	collections::{BTreeMap, BTreeSet},
	future::Future,
	io::{self, IsTerminal as _},
	sync::{Arc, Mutex},
	time::Duration,
};

use anyhow::Context as _;
use emelex::{
	Emelex,
	hub::{
		DownloadCancellation, DownloadControl, DownloadEvent, DownloadObserver, HubDiagnostic,
		HubModel, HubQuantization, HubQuantizationMode, HubSearch, REMOTE_FILTERS,
	},
	model::{
		HubModelId, InstalledModel, Modality, ModelSnapshotId, ModelTraits, MtpSupport,
		ResolvedRevision, Task,
	},
};

use super::{
	args::HubCommand,
	hub_auth_cmd, output,
	style::{Palette, bytes, tokens},
};

const EMPTY_SEARCH_MESSAGE: &str = "No compatible MLX models matched this search on this machine.";
const EMPTY_SEARCH_PAGE_MESSAGE: &str =
	"No compatible MLX models on this ranked page; use the next cursor to continue.";
const SEARCH_CARD_WIDTH: usize = 64;

pub(crate) async fn run(
	emelex: &Emelex,
	command: HubCommand,
	json: bool,
	stdout_palette: Palette,
	stderr_palette: Palette,
) -> anyhow::Result<()> {
	match command {
		HubCommand::Auth { command } => {
			hub_auth_cmd::run(emelex.home(), command, json, stdout_palette, stderr_palette)
		}
		HubCommand::Capabilities => {
			if json {
				output::json_line(&REMOTE_FILTERS)
			} else {
				for capability in REMOTE_FILTERS {
					output::stdout_line(&format!(
						"{:<42} {:<10} {}",
						capability.filter, capability.evidence, capability.meaning
					))?;
				}
				Ok(())
			}
		}
		HubCommand::Search {
			query,
			require,
			cursor,
			verbose,
		} => {
			search(
				emelex,
				query,
				require,
				cursor,
				verbose,
				json,
				stdout_palette,
				stderr_palette,
			)
			.await
		}
		HubCommand::Inspect { model, verbose } => {
			inspect(emelex, &model, verbose, json, stdout_palette).await
		}
		HubCommand::Download { model } => {
			download(emelex, &model, json, stdout_palette, stderr_palette)
				.await
				.map(drop)
		}
	}
}

#[allow(
	clippy::too_many_arguments,
	reason = "CLI presentation inputs remain explicit at the command boundary"
)]
async fn search(
	emelex: &Emelex,
	query: Option<String>,
	require: Vec<emelex::model::TraitFilter>,
	cursor: Option<String>,
	verbose: bool,
	json: bool,
	stdout_palette: Palette,
	stderr_palette: Palette,
) -> anyhow::Result<()> {
	let mut search = HubSearch::default().mlx_library().requirements(require);
	if let Some(query) = query {
		search = search.query(query);
	}
	if let Some(cursor) = cursor {
		search = search.cursor(cursor);
	}
	let models = emelex
		.models()
		.context("initialize fit-aware model catalog")?;
	let page = wait_for_hub("searching Hugging Face", json, async {
		models
			.hub()
			.search(&search)
			.await
			.context("search Hugging Face")
	})
	.await?;
	if json {
		return output::json_line(&page);
	}
	let installed_hub = if page.items.is_empty() {
		InstalledHubIndex::new()
	} else {
		let snapshots = wait_for_hub("checking downloaded models", false, async {
			models
				.installed_hub_snapshots()
				.await
				.context("list installed Hub snapshots")
		})
		.await?;
		installed_hub_index(&snapshots)
	};
	if page.items.is_empty() {
		output::stdout_line(empty_search_message(page.next_cursor.is_some()))?;
	} else {
		for (index, model) in page.items.iter().enumerate() {
			if index > 0 {
				output::stdout_line("")?;
			}
			let status = search_install_status(&model.id, &model.revision, &installed_hub);
			output::stdout_line(&render_search_model(model, status, stdout_palette))?;
		}
	}
	if verbose && !page.diagnostics.is_empty() {
		output::stderr_line(&stderr_palette.yellow("Skipped candidates:"))?;
		for (candidate, messages) in grouped_diagnostics(&page.diagnostics) {
			output::stderr_line(&stderr_palette.yellow(&format!("  {candidate}")))?;
			for message in messages {
				output::stderr_line(&stderr_palette.dim(&format!("    {message}")))?;
			}
		}
	} else if !page.diagnostics.is_empty() {
		output::stderr_line(&stderr_palette.dim(&hidden_diagnostics_line(page.diagnostics.len())))?;
	}
	output::stderr_line(&stderr_palette.dim(&search_summary_line(page.items.len(), page.scanned)))?;
	if let Some(cursor) = &page.next_cursor {
		output::stderr_line(&stderr_palette.dim(&next_cursor_line(cursor)))?;
	}
	if search_selection_enabled(
		json,
		[
			io::stdin().is_terminal(),
			io::stdout().is_terminal(),
			io::stderr().is_terminal(),
		],
	) && !page.items.is_empty()
	{
		select_search_result(
			emelex,
			&page.items,
			&installed_hub,
			stdout_palette,
			stderr_palette,
		)
		.await?;
	}
	Ok(())
}

async fn select_search_result(
	emelex: &Emelex,
	items: &[HubModel],
	installed: &InstalledHubIndex,
	stdout_palette: Palette,
	stderr_palette: Palette,
) -> anyhow::Result<()> {
	let (_, terminal_columns) = dialoguer::console::Term::stderr().size();
	let label_width = usize::from(terminal_columns).saturating_sub(4).max(1);
	let labels = items
		.iter()
		.map(|model| {
			search_selection_label(
				&model.id,
				model.quantization,
				search_install_status(&model.id, &model.revision, installed),
				label_width,
			)
		})
		.collect::<Vec<_>>();
	let selected = dialoguer::Select::new()
		.with_prompt("Choose a model to download")
		.items(&labels)
		.default(0)
		.report(false)
		.interact_opt()
		.context("choose Hub model")?;
	if let Some(selected) = selected {
		let model = items
			.get(selected)
			.context("Hub selection returned an invalid result index")?;
		download_revision(
			emelex,
			&model.id,
			&model.revision,
			false,
			stdout_palette,
			stderr_palette,
		)
		.await?;
	}
	Ok(())
}

type InstalledHubIndex = BTreeMap<HubModelId, BTreeSet<ResolvedRevision>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SearchInstallStatus {
	Downloaded,
	DifferentRevision,
	NotDownloaded,
}

impl SearchInstallStatus {
	const fn label(self) -> &'static str {
		match self {
			Self::Downloaded => "downloaded",
			Self::DifferentRevision => "different revision downloaded",
			Self::NotDownloaded => "not downloaded",
		}
	}

	const fn selector_label(self) -> &'static str {
		match self {
			Self::Downloaded => "downloaded",
			Self::DifferentRevision => "other revision",
			Self::NotDownloaded => "not downloaded",
		}
	}

	const fn selector_code(self) -> &'static str {
		match self {
			Self::Downloaded => "D",
			Self::DifferentRevision => "R",
			Self::NotDownloaded => "N",
		}
	}
}

fn installed_hub_index(installed: &[ModelSnapshotId]) -> InstalledHubIndex {
	let mut index = InstalledHubIndex::new();
	for installed in installed {
		let ModelSnapshotId::Hub { id, revision } = installed else {
			continue;
		};
		index
			.entry(id.clone())
			.or_default()
			.insert(revision.clone());
	}
	index
}

fn search_install_status(
	id: &HubModelId,
	revision: &ResolvedRevision,
	installed: &InstalledHubIndex,
) -> SearchInstallStatus {
	match installed.get(id) {
		Some(revisions) if revisions.contains(revision) => SearchInstallStatus::Downloaded,
		Some(_) => SearchInstallStatus::DifferentRevision,
		None => SearchInstallStatus::NotDownloaded,
	}
}

const fn search_selection_enabled(
	json: bool,
	[stdin_is_terminal, stdout_is_terminal, stderr_is_terminal]: [bool; 3],
) -> bool {
	!json && stdin_is_terminal && stdout_is_terminal && stderr_is_terminal
}

fn search_selection_label(
	id: &HubModelId,
	quantization: HubQuantization,
	status: SearchInstallStatus,
	max_width: usize,
) -> String {
	const MIN_ID_WIDTH: usize = 12;
	let full_prefix = format!(
		"[{:<14}] [{:<14}] ",
		status.selector_label(),
		compact_quantization(quantization),
	);
	let status_prefix = format!("[{}] ", status.selector_label());
	let code_prefix = format!("[{}] ", status.selector_code());
	let prefix = [
		full_prefix.as_str(),
		status_prefix.as_str(),
		code_prefix.as_str(),
	]
	.into_iter()
	.find(|prefix| dialoguer::console::measure_text_width(prefix) + MIN_ID_WIDTH <= max_width)
	.unwrap_or("");
	let id_width = max_width.saturating_sub(dialoguer::console::measure_text_width(prefix));
	let id = truncate_middle_ascii(id.as_str(), id_width);
	format!("{prefix}{id}")
}

fn truncate_middle_ascii(value: &str, width: usize) -> String {
	if value.len() <= width {
		return value.to_string();
	}
	if width == 0 {
		return String::new();
	}
	if width == 1 {
		return "…".to_string();
	}
	let retained = width - 1;
	let head = retained.div_ceil(2);
	let tail = retained / 2;
	format!("{}…{}", &value[..head], &value[value.len() - tail..])
}

fn render_search_model(model: &HubModel, status: SearchInstallStatus, palette: Palette) -> String {
	let sizing = model.traits.sizing.as_ref();
	render_search_card(
		&SearchCardData {
			id: model.id.as_str(),
			status,
			quantization: model.quantization,
			weights_bytes: sizing.and_then(|sizing| sizing.weights_bytes),
			memory: model.fit.as_ref().map(|fit| {
				(
					fit.required_bytes,
					fit.budget_bytes,
					fit.workload.batch_size(),
					fit.workload.context_tokens(),
				)
			}),
			max_context_tokens: sizing.and_then(|sizing| sizing.max_context_tokens),
			traits: &model.traits,
		},
		palette,
	)
}

struct SearchCardData<'a> {
	id: &'a str,
	status: SearchInstallStatus,
	quantization: HubQuantization,
	weights_bytes: Option<u64>,
	memory: Option<(u64, u64, usize, usize)>,
	max_context_tokens: Option<usize>,
	traits: &'a ModelTraits,
}

fn render_search_card(card: &SearchCardData<'_>, palette: Palette) -> String {
	let id = output::terminal_safe_inline(card.id);
	let status = match card.status {
		SearchInstallStatus::Downloaded => palette.green(card.status.label()),
		SearchInstallStatus::DifferentRevision => palette.yellow(card.status.label()),
		SearchInstallStatus::NotDownloaded => palette.dim(card.status.label()),
	};
	let mut lines = vec![
		palette.cyan(&id),
		search_field("Status", &status),
		search_field("Quant", &quantization_summary(card.quantization)),
		search_field("Weights", &optional_bytes(card.weights_bytes)),
	];
	if let Some((required, budget, batch_size, context_tokens)) = card.memory {
		lines.push(search_field(
			"Memory",
			&format!("{} required", bytes(required)),
		));
		lines.push(format!(
			"          at batch {batch_size} · {} tokens",
			optional_tokens(Some(context_tokens))
		));
		lines.push(search_field("Budget", &format!("{} Metal", bytes(budget))));
	} else {
		lines.push(search_field("Memory", "requirement unknown"));
		lines.push(search_field("Budget", "unknown"));
	}
	lines.push(search_field(
		"Context",
		&format!("{} max", optional_tokens(card.max_context_tokens)),
	));
	lines.extend(search_capability_lines(card.traits));
	lines.join("\n")
}

fn search_field(label: &str, value: &str) -> String {
	format!("  {label:<7} {value}")
}

fn quantization_summary(quantization: HubQuantization) -> String {
	let (mut summary, has_layer_overrides) = match quantization {
		HubQuantization::NotConfigured => return "not configured".to_string(),
		HubQuantization::Configured(config) => {
			let summary = match config.mode() {
				HubQuantizationMode::Affine => {
					format!(
						"{}-bit affine · group {}",
						config.bits(),
						config.group_size()
					)
				}
				HubQuantizationMode::Mxfp4 => {
					format!("MXFP4 · group {}", config.group_size())
				}
				HubQuantizationMode::Mxfp8 => {
					format!("MXFP8 · group {}", config.group_size())
				}
				HubQuantizationMode::Nvfp4 => {
					format!("NVFP4 · group {}", config.group_size())
				}
				_ => "unknown".to_string(),
			};
			(summary, config.has_layer_overrides())
		}
		_ => return "unknown".to_string(),
	};
	if has_layer_overrides {
		summary.push_str(" · layer overrides");
	}
	summary
}

fn compact_quantization(quantization: HubQuantization) -> String {
	match quantization {
		HubQuantization::NotConfigured => "not configured".to_string(),
		HubQuantization::Configured(config) => match config.mode() {
			HubQuantizationMode::Affine => format!("{}-bit", config.bits()),
			HubQuantizationMode::Mxfp4 => "MXFP4".to_string(),
			HubQuantizationMode::Mxfp8 => "MXFP8".to_string(),
			HubQuantizationMode::Nvfp4 => "NVFP4".to_string(),
			_ => "unknown".to_string(),
		},
		_ => "unknown".to_string(),
	}
}

fn search_capability_lines(traits: &ModelTraits) -> Vec<String> {
	let mut tasks = Vec::new();
	for task in &traits.tasks {
		let label = match task {
			Task::TextGeneration => "text generation",
			Task::Chat => "chat",
			Task::ToolUse => "tools",
			Task::StructuredOutput => "structured output",
			Task::Reasoning => "reasoning",
			_ => "other task",
		};
		if !tasks.contains(&label) {
			tasks.push(label);
		}
	}
	let mut inputs = Vec::new();
	for input in &traits.input {
		let label = match input {
			Modality::Text => None,
			Modality::Image => Some("image"),
			Modality::Audio => Some("audio"),
			_ => Some("other"),
		};
		if let Some(label) = label
			&& !inputs.contains(&label)
		{
			inputs.push(label);
		}
	}
	let mut lines = Vec::new();
	match traits.mtp {
		MtpSupport::Absent => {}
		MtpSupport::Advertised => lines.push(search_field("MTP", "advertised")),
		MtpSupport::RuntimeVerified => lines.push(search_field("MTP", "verified")),
		_ => lines.push(search_field("MTP", "unknown")),
	}
	for (label, values) in [("Tasks", tasks.as_slice()), ("Inputs", inputs.as_slice())] {
		if !values.is_empty() {
			lines.extend(wrap_search_values(label, values));
		}
	}
	if lines.is_empty() {
		lines.push(search_field("Details", "capabilities unknown"));
	}
	lines
}

fn wrap_search_values(label: &str, values: &[&str]) -> Vec<String> {
	let prefix = format!("  {label:<7} ");
	let continuation = "          ";
	let mut lines = Vec::new();
	let mut line = prefix;
	let mut has_value = false;
	for value in values {
		let separator = if has_value { " · " } else { "" };
		let projected = line.chars().count() + separator.chars().count() + value.chars().count();
		if has_value && projected > SEARCH_CARD_WIDTH {
			lines.push(line);
			line = continuation.to_string();
			has_value = false;
		}
		if has_value {
			line.push_str(" · ");
		}
		line.push_str(value);
		has_value = true;
	}
	lines.push(line);
	lines
}

fn grouped_diagnostics(diagnostics: &[HubDiagnostic]) -> BTreeMap<String, Vec<String>> {
	let mut grouped = BTreeMap::<String, Vec<String>>::new();
	for diagnostic in diagnostics {
		let candidate = diagnostic
			.id
			.as_ref()
			.map_or_else(|| "unidentified candidate".to_string(), ToString::to_string);
		let candidate = output::terminal_safe_inline(&candidate).into_owned();
		let message = output::terminal_safe_inline(&diagnostic.message).into_owned();
		grouped.entry(candidate).or_default().push(message);
	}
	grouped
}

fn hidden_diagnostics_line(count: usize) -> String {
	format!(
		"{count} search {} hidden; rerun with --verbose or --json",
		if count == 1 {
			"diagnostic"
		} else {
			"diagnostics"
		}
	)
}

const fn empty_search_message(has_more: bool) -> &'static str {
	if has_more {
		EMPTY_SEARCH_PAGE_MESSAGE
	} else {
		EMPTY_SEARCH_MESSAGE
	}
}

fn search_summary_line(results: usize, scanned: usize) -> String {
	format!(
		"{results} compatible MLX {} · {scanned} {} scanned",
		if results == 1 { "model" } else { "models" },
		if scanned == 1 {
			"candidate"
		} else {
			"candidates"
		}
	)
}

fn next_cursor_line(cursor: &str) -> String {
	let cursor = output::terminal_safe_inline(cursor);
	format!("next cursor:\n  {cursor}")
}

async fn inspect(
	emelex: &Emelex,
	id: &HubModelId,
	verbose: bool,
	json: bool,
	stdout_palette: Palette,
) -> anyhow::Result<()> {
	let models = emelex
		.models()
		.context("initialize fit-aware model catalog")?;
	let model = wait_for_hub("inspecting Hugging Face", json, async {
		models
			.hub()
			.inspect(id)
			.await
			.context("inspect Hugging Face model")
	})
	.await?;
	if json {
		return output::json_line(&model);
	}
	output::stdout_line(&stdout_palette.bold(model.id.as_str()))?;
	output::stdout_line(&format!("revision  {}", model.revision))?;
	output::stdout_line(&format!("downloads {}", model.downloads))?;
	output::stdout_line(&format!("likes     {}", model.likes))?;
	output::stdout_line(&format!(
		"license   {}",
		output::terminal_safe_inline(model.license.as_deref().unwrap_or("not declared"))
	))?;
	output::stdout_line(&format!("traits    {}", trait_summary(&model.traits)))?;
	output::stdout_line(&format!(
		"weights   {}",
		optional_bytes(
			model
				.traits
				.sizing
				.as_ref()
				.and_then(|sizing| sizing.weights_bytes)
		)
	))?;
	output::stdout_line(&format!(
		"residency {}",
		optional_bytes(
			model
				.traits
				.sizing
				.as_ref()
				.and_then(|sizing| sizing.estimated_residency_bytes)
		)
	))?;
	output::stdout_line(&format!(
		"context   {}",
		model
			.traits
			.sizing
			.as_ref()
			.and_then(|sizing| sizing.max_context_tokens)
			.map_or_else(|| "unknown".to_string(), |tokens| tokens.to_string())
	))?;
	output::stdout_line(&format!(
		"compatible {}",
		if model.compatible { "yes" } else { "no" }
	))?;
	if let Some(fit) = &model.fit {
		output::stdout_line(&format!(
			"fit       {} ({} required / {} budget)",
			if fit.fits { "yes" } else { "no" },
			bytes(fit.required_bytes),
			bytes(fit.budget_bytes)
		))?;
	} else {
		output::stdout_line("fit       unknown")?;
	}
	let visible_diagnostics = if verbose {
		model.diagnostics.len()
	} else {
		model.diagnostics.len().min(10)
	};
	for diagnostic in model.diagnostics.iter().take(visible_diagnostics) {
		output::stdout_line(&format!(
			"diagnostic {}",
			output::terminal_safe_inline(diagnostic)
		))?;
	}
	if visible_diagnostics < model.diagnostics.len() {
		output::stdout_line(&format!(
			"diagnostic {} more omitted; rerun with --verbose or --json",
			model.diagnostics.len() - visible_diagnostics
		))?;
	}
	Ok(())
}

pub(crate) async fn wait_for_hub<T>(
	label: &'static str,
	json: bool,
	operation: impl Future<Output = anyhow::Result<T>>,
) -> anyhow::Result<T> {
	let mut progress = TtyProgress::new(
		label,
		show_hub_progress(json, std::io::stderr().is_terminal()),
	);
	let mut interval = tokio::time::interval(Duration::from_millis(120));
	interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
	let cancellation = tokio::signal::ctrl_c();
	tokio::pin!(operation);
	tokio::pin!(cancellation);
	loop {
		tokio::select! {
			result = &mut operation => {
				progress.clear()?;
				return result;
			}
			signal = &mut cancellation => {
				progress.clear()?;
				signal.context("listen for Hub operation cancellation")?;
				anyhow::bail!("{label} cancelled");
			}
			_ = interval.tick(), if progress.enabled => progress.tick()?,
		}
	}
}

const fn show_hub_progress(json: bool, stderr_is_terminal: bool) -> bool {
	!json && stderr_is_terminal
}

struct TtyProgress {
	label: &'static str,
	enabled: bool,
	frame: usize,
	rendered: bool,
}

impl TtyProgress {
	const FRAMES: [&'static str; 4] = ["⠋", "⠙", "⠹", "⠸"];

	const fn new(label: &'static str, enabled: bool) -> Self {
		Self {
			label,
			enabled,
			frame: 0,
			rendered: false,
		}
	}

	fn tick(&mut self) -> anyhow::Result<()> {
		output::stderr(&format!(
			"\r{} {}",
			Self::FRAMES[self.frame % Self::FRAMES.len()],
			self.label
		))?;
		self.frame = self.frame.wrapping_add(1);
		self.rendered = true;
		Ok(())
	}

	fn clear(&mut self) -> anyhow::Result<()> {
		if self.enabled && self.rendered {
			output::stderr(&format!("\r{}\r", " ".repeat(self.label.len() + 2)))?;
			self.rendered = false;
		}
		Ok(())
	}
}

pub(crate) async fn download(
	emelex: &Emelex,
	model: &HubModelId,
	json: bool,
	stdout_palette: Palette,
	stderr_palette: Palette,
) -> anyhow::Result<InstalledModel> {
	download_selected(emelex, model, None, json, stdout_palette, stderr_palette).await
}

pub(crate) async fn download_revision(
	emelex: &Emelex,
	model: &HubModelId,
	revision: &ResolvedRevision,
	json: bool,
	stdout_palette: Palette,
	stderr_palette: Palette,
) -> anyhow::Result<InstalledModel> {
	download_selected(
		emelex,
		model,
		Some(revision),
		json,
		stdout_palette,
		stderr_palette,
	)
	.await
}

async fn download_selected(
	emelex: &Emelex,
	model: &HubModelId,
	revision: Option<&ResolvedRevision>,
	json: bool,
	stdout_palette: Palette,
	stderr_palette: Palette,
) -> anyhow::Result<InstalledModel> {
	let (observer, output_error) = download_observer(json, stderr_palette);
	let cancellation = DownloadCancellation::default();
	let models = emelex.models().context("initialize model manager")?;
	let watcher = DownloadCancellationWatcher::spawn(cancellation.clone(), tokio::signal::ctrl_c());
	let result = match revision {
		Some(revision) => {
			models
				.download_revision_controlled(model, revision, Some(&observer), Some(&cancellation))
				.await
		}
		None => {
			models
				.download_controlled(model, Some(&observer), Some(&cancellation))
				.await
		}
	};
	watcher.finish().await?;
	let observer_error = output_error
		.lock()
		.unwrap_or_else(std::sync::PoisonError::into_inner)
		.take();
	if let Some(error) = observer_error {
		return Err(error);
	}
	let installed = result.with_context(|| format!("download {model}"))?;
	if json {
		output::json_line(&installed_json(&installed))
	} else {
		output::stdout_line(&format!(
			"{} {}",
			stdout_palette.green("installed"),
			output::terminal_safe_inline(&installed.path().display().to_string())
		))
	}?;
	Ok(installed)
}

struct DownloadCancellationWatcher {
	task: Option<tokio::task::JoinHandle<()>>,
	error: Arc<Mutex<Option<io::Error>>>,
}

impl DownloadCancellationWatcher {
	fn spawn<F>(cancellation: DownloadCancellation, signal: F) -> Self
	where
		F: Future<Output = io::Result<()>> + Send + 'static,
	{
		let error = Arc::new(Mutex::new(None));
		let task_error = Arc::clone(&error);
		let task = tokio::spawn(async move {
			if let Err(signal_error) = signal.await {
				*task_error
					.lock()
					.unwrap_or_else(std::sync::PoisonError::into_inner) = Some(signal_error);
			}
			cancellation.cancel();
		});
		Self {
			task: Some(task),
			error,
		}
	}

	async fn finish(mut self) -> anyhow::Result<()> {
		let Some(task) = self.task.take() else {
			return Err(anyhow::anyhow!(
				"download cancellation watcher task is absent"
			));
		};
		task.abort();
		if let Err(error) = task.await
			&& !error.is_cancelled()
		{
			return Err(anyhow::Error::new(error).context("join download cancellation watcher"));
		}
		let signal_error = self
			.error
			.lock()
			.unwrap_or_else(std::sync::PoisonError::into_inner)
			.take();
		if let Some(error) = signal_error {
			return Err(anyhow::Error::new(error).context("listen for download cancellation"));
		}
		Ok(())
	}
}

impl Drop for DownloadCancellationWatcher {
	fn drop(&mut self) {
		if let Some(task) = &self.task {
			task.abort();
		}
	}
}

fn download_observer(
	json: bool,
	stderr_palette: Palette,
) -> (DownloadObserver, Arc<Mutex<Option<anyhow::Error>>>) {
	let progress = Arc::new(Mutex::new(BTreeMap::<String, u64>::new()));
	let output_error = Arc::new(Mutex::new(None));
	let observer_error = Arc::clone(&output_error);
	let observer = Arc::new(move |event: &DownloadEvent| {
		let rendered = if json {
			let value = match event {
				DownloadEvent::FileStarted {
					path,
					resumed,
					total,
				} => serde_json::json!({
					"type": "download_file_started",
					"path": path,
					"resumed": resumed,
					"total": total,
				}),
				DownloadEvent::Progress {
					path,
					received,
					total,
				} => serde_json::json!({
					"type": "download_progress",
					"path": path,
					"received": received,
					"total": total,
				}),
				DownloadEvent::Retrying {
					path,
					attempt,
					reason,
				} => serde_json::json!({
					"type": "download_retrying",
					"path": path,
					"attempt": attempt,
					"reason": reason,
				}),
				DownloadEvent::FileVerified { path, sha256 } => serde_json::json!({
					"type": "download_file_verified",
					"path": path,
					"sha256": sha256,
				}),
				_ => return Ok(DownloadControl::Continue),
			};
			output::json_line(&value)
		} else {
			render_human_download_event(event, stderr_palette, &progress)
		};
		if let Err(error) = rendered {
			*observer_error
				.lock()
				.unwrap_or_else(std::sync::PoisonError::into_inner) = Some(error);
			return Ok(DownloadControl::Cancel);
		}
		Ok(DownloadControl::Continue)
	});
	(observer, output_error)
}

fn render_human_download_event(
	event: &DownloadEvent,
	palette: Palette,
	progress: &Mutex<BTreeMap<String, u64>>,
) -> anyhow::Result<()> {
	let should_render = match event {
		DownloadEvent::Progress {
			path,
			received,
			total,
		} => {
			let step = (*total / 20).max(1);
			{
				let mut observed = progress
					.lock()
					.unwrap_or_else(std::sync::PoisonError::into_inner);
				let previous = observed.entry(path.clone()).or_default();
				let should_render = received.saturating_sub(*previous) >= step || received == total;
				if should_render {
					*previous = *received;
				}
				drop(observed);
				should_render
			}
		}
		_ => true,
	};
	let Some(line) = human_download_event_line(event).filter(|_| should_render) else {
		return Ok(());
	};
	if matches!(event, DownloadEvent::Retrying { .. }) {
		output::stderr_line(&palette.yellow(&line))
	} else {
		output::stderr_line(&palette.dim(&line))
	}
}

fn human_download_event_line(event: &DownloadEvent) -> Option<String> {
	match event {
		DownloadEvent::FileStarted {
			path,
			resumed,
			total,
		} => {
			let path = output::terminal_safe_inline(path);
			Some(format!(
				"downloading {path} ({} / {})",
				bytes(*resumed),
				bytes(*total)
			))
		}
		DownloadEvent::Progress {
			path,
			received,
			total,
		} => {
			let path = output::terminal_safe_inline(path);
			Some(format!("{path}: {} / {}", bytes(*received), bytes(*total)))
		}
		DownloadEvent::Retrying {
			path,
			attempt,
			reason,
		} => {
			let path = output::terminal_safe_inline(path);
			let reason = output::terminal_safe_inline(reason);
			Some(format!("retrying {path} after attempt {attempt}: {reason}"))
		}
		DownloadEvent::FileVerified { path, .. } => {
			let path = output::terminal_safe_inline(path);
			Some(format!("verified {path}"))
		}
		_ => None,
	}
}

pub(crate) fn installed_json(installed: &emelex::model::InstalledModel) -> serde_json::Value {
	let manifest = installed.manifest();
	serde_json::json!({
		"reference": installed.reference(),
		"snapshot": installed.snapshot_id(),
		"path": installed.path(),
		"revision": manifest.resolved_revision(),
		"source": manifest.source(),
		"installed_at": manifest.installed_at(),
		"verification": manifest.verification(),
		"traits": manifest.traits(),
	})
}

pub(crate) fn trait_summary(traits: &emelex::model::ModelTraits) -> String {
	let mtp = match traits.mtp {
		emelex::model::MtpSupport::Absent => "none",
		emelex::model::MtpSupport::Advertised => "advertised (metadata only)",
		emelex::model::MtpSupport::RuntimeVerified => "runtime verified",
		_ => "unknown",
	};
	format!(
		"input={:?} tasks={:?} mlx={} mtp={mtp}",
		traits.input, traits.tasks, traits.mlx
	)
}

fn optional_bytes(value: Option<u64>) -> String {
	value.map_or_else(|| "unknown".to_string(), bytes)
}

fn optional_tokens(value: Option<usize>) -> String {
	value.map_or_else(
		|| "unknown".to_string(),
		|value| u64::try_from(value).map_or_else(|_| value.to_string(), tokens),
	)
}

#[cfg(test)]
mod tests {
	use super::*;

	fn configured_quantization(
		mode: &str,
		bits: u8,
		group_size: u16,
		has_layer_overrides: bool,
	) -> HubQuantization {
		serde_json::from_value(serde_json::json!({
			"kind": "configured",
			"mode": mode,
			"bits": bits,
			"group_size": group_size,
			"has_layer_overrides": has_layer_overrides
		}))
		.expect("valid Hub quantization")
	}

	#[test]
	fn search_card_is_compact_human_readable_and_complete() {
		let mut traits = ModelTraits::default();
		traits.mlx = true;
		traits.input.extend([Modality::Text, Modality::Image]);
		traits
			.tasks
			.extend([Task::TextGeneration, Task::Chat, Task::ToolUse]);
		traits.mtp = MtpSupport::Advertised;

		let rendered = render_search_card(
			&SearchCardData {
				id: "mlx-community/Qwen3.5-4B-4bit",
				status: SearchInstallStatus::NotDownloaded,
				quantization: configured_quantization("affine", 4, 64, false),
				weights_bytes: Some(4_u64 << 30),
				memory: Some((6_u64 << 30, 16_u64 << 30, 1, 16_384)),
				max_context_tokens: Some(32_768),
				traits: &traits,
			},
			Palette::stdout(crate::style::ColorMode::Never),
		);

		assert_eq!(
			rendered,
			"mlx-community/Qwen3.5-4B-4bit\n  Status  not downloaded\n  Quant   4-bit affine · \
			 group 64\n  Weights 4.0 GiB\n  Memory  6.0 GiB required\n          at batch 1 · 16.4k \
			 tokens\n  Budget  16 GiB Metal\n  Context 32.8k max\n  MTP     advertised\n  Tasks   text \
			 generation · chat · tools\n  Inputs  image"
		);
		assert!(!rendered.contains("Runtime"));
		assert!(
			rendered
				.lines()
				.all(|line| line.chars().count() <= SEARCH_CARD_WIDTH)
		);
	}

	#[test]
	fn search_card_handles_unknowns_and_neutralizes_untrusted_id() {
		let rendered = render_search_card(
			&SearchCardData {
				id: "owner/model\nforged\trow\u{202e}",
				status: SearchInstallStatus::Downloaded,
				quantization: HubQuantization::Unknown,
				weights_bytes: None,
				memory: None,
				max_context_tokens: None,
				traits: &ModelTraits::default(),
			},
			Palette::stdout(crate::style::ColorMode::Never),
		);

		assert_eq!(
			rendered,
			"owner/model\u{240a}forged\u{2409}row\u{fffd}\n  Status  downloaded\n  Quant   \
			 unknown\n  Weights unknown\n  Memory  requirement unknown\n  Budget  unknown\n  Context unknown \
			 max\n  Details capabilities unknown"
		);
		assert_eq!(rendered.lines().count(), 8);
	}

	#[test]
	fn quantization_summary_preserves_named_mode_and_layer_overrides() {
		let quantization = configured_quantization("mxfp4", 4, 32, true);

		assert_eq!(
			quantization_summary(quantization),
			"MXFP4 · group 32 · layer overrides"
		);
		assert_eq!(compact_quantization(quantization), "MXFP4");
	}

	#[test]
	fn installed_status_distinguishes_current_different_and_missing_snapshots() {
		let id = HubModelId::parse("owner/model").expect("valid Hub ID");
		let current = ResolvedRevision::parse("a".repeat(40)).expect("valid revision");
		let different = ResolvedRevision::parse("b".repeat(40)).expect("valid revision");
		let mut installed = InstalledHubIndex::new();

		assert_eq!(
			search_install_status(&id, &current, &installed),
			SearchInstallStatus::NotDownloaded
		);
		installed.entry(id.clone()).or_default().insert(different);
		assert_eq!(
			search_install_status(&id, &current, &installed),
			SearchInstallStatus::DifferentRevision
		);
		installed
			.entry(id.clone())
			.or_default()
			.insert(current.clone());
		assert_eq!(
			search_install_status(&id, &current, &installed),
			SearchInstallStatus::Downloaded
		);
	}

	#[test]
	fn search_selection_label_stays_on_one_narrow_terminal_row() {
		let id = HubModelId::parse("a".repeat(96)).expect("maximum-length Hub ID");

		let label = search_selection_label(
			&id,
			HubQuantization::NotConfigured,
			SearchInstallStatus::DifferentRevision,
			40,
		);

		assert!(dialoguer::console::measure_text_width(&label) <= 40);
		assert!(label.starts_with("[other revision"));
		assert!(label.contains('…'));
	}

	#[test]
	fn narrow_selection_labels_preserve_distinguishing_id_suffixes() {
		let first = HubModelId::parse(format!("org/{}first", "a".repeat(80)))
			.expect("maximum-length first Hub ID");
		let second = HubModelId::parse(format!("org/{}second", "a".repeat(80)))
			.expect("maximum-length second Hub ID");

		let first_label = search_selection_label(
			&first,
			HubQuantization::NotConfigured,
			SearchInstallStatus::DifferentRevision,
			24,
		);
		let second_label = search_selection_label(
			&second,
			HubQuantization::NotConfigured,
			SearchInstallStatus::DifferentRevision,
			24,
		);

		assert_ne!(first_label, second_label);
		assert!(first_label.ends_with("first"));
		assert!(second_label.ends_with("second"));
		assert!(dialoguer::console::measure_text_width(&first_label) <= 24);
		assert!(dialoguer::console::measure_text_width(&second_label) <= 24);
	}

	#[test]
	fn search_selection_requires_human_terminal_streams() {
		assert!(search_selection_enabled(false, [true, true, true]));
		assert!(!search_selection_enabled(true, [true, true, true]));
		assert!(!search_selection_enabled(false, [false, true, true]));
		assert!(!search_selection_enabled(false, [true, false, true]));
		assert!(!search_selection_enabled(false, [true, true, false]));
	}

	#[test]
	fn diagnostics_group_by_candidate_and_cannot_forge_rows() {
		let diagnostics = serde_json::from_value::<Vec<HubDiagnostic>>(serde_json::json!([
			{"id": "owner/model", "message": "unsupported layout"},
			{"id": "owner/model", "message": "bad\nrow\tvalue\u{202e}"},
			{"id": null, "message": "invalid identity"},
		]))
		.expect("diagnostic fixture");

		assert_eq!(
			grouped_diagnostics(&diagnostics),
			BTreeMap::from([
				(
					"owner/model".to_string(),
					vec![
						"unsupported layout".to_string(),
						"bad\u{240a}row\u{2409}value\u{fffd}".to_string(),
					],
				),
				(
					"unidentified candidate".to_string(),
					vec!["invalid identity".to_string()],
				),
			])
		);
	}

	#[test]
	fn search_summaries_have_explicit_empty_and_singular_states() {
		assert_eq!(
			empty_search_message(false),
			"No compatible MLX models matched this search on this machine."
		);
		assert_eq!(
			empty_search_message(true),
			"No compatible MLX models on this ranked page; use the next cursor to continue."
		);
		assert_eq!(
			search_summary_line(0, 200),
			"0 compatible MLX models · 200 candidates scanned"
		);
		assert_eq!(
			search_summary_line(1, 1),
			"1 compatible MLX model · 1 candidate scanned"
		);
		assert_eq!(
			hidden_diagnostics_line(1),
			"1 search diagnostic hidden; rerun with --verbose or --json"
		);
		assert_eq!(
			next_cursor_line("page\nforged\tvalue\u{202e}"),
			"next cursor:\n  page\u{240a}forged\u{2409}value\u{fffd}"
		);
	}

	#[test]
	fn progress_is_tty_human_only() {
		assert!(show_hub_progress(false, true));
		assert!(!show_hub_progress(true, true));
		assert!(!show_hub_progress(false, false));
	}

	#[test]
	fn human_download_fields_cannot_forge_rows() {
		for event in [
			DownloadEvent::FileStarted {
				path: "model\nforged".to_string(),
				resumed: 1,
				total: 2,
			},
			DownloadEvent::Progress {
				path: "model\tforged".to_string(),
				received: 1,
				total: 2,
			},
			DownloadEvent::Retrying {
				path: "model\u{202e}".to_string(),
				attempt: 1,
				reason: "failed\nforged".to_string(),
			},
			DownloadEvent::FileVerified {
				path: "model\nforged".to_string(),
				sha256: "a".repeat(64),
			},
		] {
			let line = human_download_event_line(&event).expect("human event");
			assert!(!line.contains('\n'));
			assert!(!line.contains('\t'));
			assert!(!line.contains('\u{202e}'));
		}
	}

	#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
	async fn cancellation_watcher_runs_during_a_blocking_local_phase() {
		let cancellation = DownloadCancellation::default();
		let watcher = DownloadCancellationWatcher::spawn(cancellation.clone(), async { Ok(()) });
		for _ in 0..1_000 {
			if cancellation.is_cancelled() {
				break;
			}
			std::thread::sleep(Duration::from_millis(1));
		}
		assert!(cancellation.is_cancelled());
		watcher.finish().await.expect("watcher");
	}
}
