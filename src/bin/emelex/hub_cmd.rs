//! Hugging Face discovery and download presentation.

use std::{
	collections::{BTreeMap, BTreeSet, VecDeque},
	future::Future,
	io::{self, IsTerminal as _},
	sync::{Arc, Mutex},
	time::{Duration, Instant},
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
		ResolvedRevision, Task, TraitFilter,
	},
};

use super::{
	args::HubCommand,
	hub_auth_cmd, output,
	style::{Palette, bytes, tokens},
	terminal_ui::{LiveRegion, fit_line},
};

const EMPTY_SEARCH_MESSAGE: &str = "No compatible MLX models matched this search on this machine.";
const EMPTY_SEARCH_PAGE_MESSAGE: &str =
	"No compatible MLX models on this ranked page; use the next cursor to continue.";
const SEARCH_CARD_WIDTH: usize = 64;
const CLI_REQUIRED_SEARCH_TRAIT: &str = "interaction:tools";

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
				output::stdout_line(&stdout_palette.bold("Remote model filters"))?;
				output::stdout_line(
					&stdout_palette.dim("Use with `emelex hub search --require <filter>`."),
				)?;
				for capability in REMOTE_FILTERS {
					output::stdout_line("")?;
					output::stdout_line(&format!(
						"  {}  {}",
						stdout_palette.cyan(capability.filter),
						stdout_palette.dim(capability.evidence)
					))?;
					output::stdout_line(&format!("    {}", capability.meaning))?;
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
	let requirements = cli_search_requirements(require)?;
	let mut search = HubSearch::default()
		.mlx_library()
		.requirements(requirements);
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
	let selection_enabled = search_selection_enabled(
		json,
		[
			io::stdin().is_terminal(),
			io::stdout().is_terminal(),
			io::stderr().is_terminal(),
		],
	) && !page.items.is_empty();
	let selected = if selection_enabled {
		select_search_result(&page.items, &installed_hub, page.scanned, stdout_palette)?
	} else {
		None
	};
	if page.items.is_empty() {
		output::stdout_line(empty_search_message(page.next_cursor.is_some()))?;
	} else if let Some(selected) = selected {
		let model = page
			.items
			.get(selected)
			.context("Hub selection returned an invalid result index")?;
		let status = search_install_status(&model.id, &model.revision, &installed_hub);
		output::stdout_line(&render_search_model(model, status, stdout_palette))?;
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
	if let Some(selected) = selected {
		let model = page
			.items
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

fn cli_search_requirements(require: Vec<TraitFilter>) -> anyhow::Result<Vec<TraitFilter>> {
	let mut requirements = require.into_iter().collect::<BTreeSet<_>>();
	requirements.insert(
		TraitFilter::parse(CLI_REQUIRED_SEARCH_TRAIT)
			.context("build implicit CLI Hub search requirement")?,
	);
	Ok(requirements.into_iter().collect())
}

fn select_search_result(
	items: &[HubModel],
	installed: &InstalledHubIndex,
	scanned: usize,
	palette: Palette,
) -> anyhow::Result<Option<usize>> {
	let mut region = LiveRegion::stdout();
	let result = run_search_selector(&mut region, items, installed, scanned, palette);
	let cleanup = region.clear();
	match (result, cleanup) {
		(Ok(selected), Ok(())) => Ok(selected),
		(Err(error), Ok(())) => Err(error),
		(Ok(_), Err(error)) => Err(error.context("restore Hub search results terminal")),
		(Err(error), Err(cleanup)) => {
			Err(error.context(format!("also failed to restore terminal: {cleanup:#}")))
		}
	}
}

fn run_search_selector(
	region: &mut LiveRegion,
	items: &[HubModel],
	installed: &InstalledHubIndex,
	scanned: usize,
	palette: Palette,
) -> anyhow::Result<Option<usize>> {
	if items.is_empty() {
		return Ok(None);
	}
	if region.size().0 < 2 {
		return Ok(None);
	}
	let mut selected = 0;
	loop {
		let frame = render_search_selector_frame(
			items,
			installed,
			selected,
			scanned,
			region.size(),
			palette,
		);
		region.draw(&frame)?;
		match search_selector_action(&region.read_key()?, selected, items.len()) {
			SearchSelectorAction::Move(next) => selected = next,
			SearchSelectorAction::Select => return Ok(Some(selected)),
			SearchSelectorAction::Cancel => return Ok(None),
			SearchSelectorAction::Interrupt => anyhow::bail!("Hub model selection interrupted"),
			SearchSelectorAction::Ignore => {}
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SearchSelectorAction {
	Move(usize),
	Select,
	Cancel,
	Interrupt,
	Ignore,
}

fn search_selector_action(
	key: &dialoguer::console::Key,
	selected: usize,
	item_count: usize,
) -> SearchSelectorAction {
	use dialoguer::console::Key;

	if item_count == 0 {
		return SearchSelectorAction::Cancel;
	}
	match key {
		Key::ArrowDown | Key::Tab | Key::Char('j') => {
			SearchSelectorAction::Move((selected + 1) % item_count)
		}
		Key::ArrowUp | Key::BackTab | Key::Char('k') => {
			SearchSelectorAction::Move((selected + item_count - 1) % item_count)
		}
		Key::PageDown => SearchSelectorAction::Move(selected.saturating_add(5).min(item_count - 1)),
		Key::PageUp => SearchSelectorAction::Move(selected.saturating_sub(5)),
		Key::Home => SearchSelectorAction::Move(0),
		Key::End => SearchSelectorAction::Move(item_count - 1),
		Key::Enter | Key::Char(' ') => SearchSelectorAction::Select,
		Key::Escape | Key::Char('q') => SearchSelectorAction::Cancel,
		Key::CtrlC => SearchSelectorAction::Interrupt,
		_ => SearchSelectorAction::Ignore,
	}
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

fn render_search_selector_frame(
	items: &[HubModel],
	installed: &InstalledHubIndex,
	selected: usize,
	scanned: usize,
	(rows, columns): (u16, u16),
	palette: Palette,
) -> String {
	let rows = usize::from(rows).saturating_sub(1);
	let columns = usize::from(columns).max(1);
	if rows == 0 {
		return String::new();
	}
	let selected = selected.min(items.len().saturating_sub(1));
	if rows <= 2 {
		return render_tiny_search_selector(items, installed, selected, rows, columns, palette);
	}
	let compact = columns < 48 || rows < 12;
	let cards = items
		.iter()
		.enumerate()
		.map(|(index, model)| {
			search_selector_card(
				model,
				search_install_status(&model.id, &model.revision, installed),
				index == selected,
				compact,
				columns,
				palette,
			)
		})
		.collect::<Vec<_>>();
	let mut frame = Vec::new();
	if rows >= 4 {
		frame.push(fit_line(&palette.bold("Compatible MLX models"), columns));
	}
	if rows >= 6 {
		frame.push(fit_line(
			&palette.dim(&search_summary_line(items.len(), scanned)),
			columns,
		));
	}
	if rows >= 9 {
		frame.push(String::new());
	}
	let footer_rows = 1;
	let card_budget = rows
		.saturating_sub(frame.len())
		.saturating_sub(footer_rows)
		.max(1);
	let (start, end) = search_selector_window(&cards, selected, card_budget);
	let mut used = 0;
	for (offset, card) in cards[start..end].iter().enumerate() {
		if offset > 0 && used < card_budget {
			frame.push(String::new());
			used += 1;
		}
		for line in card {
			if used == card_budget {
				break;
			}
			frame.push(line.clone());
			used += 1;
		}
	}
	let position = format!("{}/{}", selected + 1, items.len());
	let footer = if columns < 52 {
		format!("↑↓ move · enter · esc · {position}")
	} else {
		format!("↑↓ move  ·  enter download  ·  esc close  ·  {position}")
	};
	frame.push(fit_line(&palette.dim(&footer), columns));
	frame.truncate(rows);
	frame.join("\n")
}

fn render_tiny_search_selector(
	items: &[HubModel],
	installed: &InstalledHubIndex,
	selected: usize,
	rows: usize,
	columns: usize,
	palette: Palette,
) -> String {
	let model = &items[selected];
	let quantization = compact_quantization(model.quantization);
	if rows == 1 {
		return fit_line(
			&format!("{} {quantization} · {}", palette.cyan("❯"), model.id),
			columns,
		);
	}
	let status = search_install_status(&model.id, &model.revision, installed);
	let status = match status {
		SearchInstallStatus::Downloaded => palette.green(status.label()),
		SearchInstallStatus::DifferentRevision => palette.yellow(status.label()),
		SearchInstallStatus::NotDownloaded => palette.dim(status.label()),
	};
	[
		fit_line(&format!("{} {}", palette.cyan("❯"), model.id), columns),
		fit_line(
			&format!(
				"  {quantization} · {status} · {}/{} · ↑↓ enter esc",
				selected + 1,
				items.len()
			),
			columns,
		),
	]
	.join("\n")
}

fn search_selector_card(
	model: &HubModel,
	status: SearchInstallStatus,
	selected: bool,
	compact: bool,
	columns: usize,
	palette: Palette,
) -> Vec<String> {
	let rendered = if compact {
		let status = match status {
			SearchInstallStatus::Downloaded => palette.green(status.label()),
			SearchInstallStatus::DifferentRevision => palette.yellow(status.label()),
			SearchInstallStatus::NotDownloaded => palette.dim(status.label()),
		};
		let weights = model
			.traits
			.sizing
			.as_ref()
			.and_then(|sizing| sizing.weights_bytes);
		format!(
			"{}\n  {} · {} · {} weights",
			palette.cyan(model.id.as_str()),
			status,
			compact_quantization(model.quantization),
			optional_bytes(weights),
		)
	} else {
		render_search_model(model, status, palette)
	};
	rendered
		.lines()
		.enumerate()
		.map(|(line_index, line)| {
			let rail = match (selected, line_index) {
				(true, 0) => palette.cyan("❯"),
				(true, _) => palette.cyan("┃"),
				(false, _) => " ".to_string(),
			};
			fit_line(&format!("{rail} {line}"), columns)
		})
		.collect()
}

fn search_selector_window(cards: &[Vec<String>], selected: usize, budget: usize) -> (usize, usize) {
	if cards.is_empty() {
		return (0, 0);
	}
	let selected = selected.min(cards.len() - 1);
	let mut start = selected;
	let mut end = selected + 1;
	let mut used = cards[selected].len().min(budget);
	loop {
		let next_cost = cards.get(end).map(|card| card.len().saturating_add(1));
		if let Some(cost) = next_cost
			&& used.saturating_add(cost) <= budget
		{
			used += cost;
			end += 1;
			continue;
		}
		let previous_cost = start
			.checked_sub(1)
			.and_then(|index| cards.get(index))
			.map(|card| card.len().saturating_add(1));
		if let Some(cost) = previous_cost
			&& used.saturating_add(cost) <= budget
		{
			used += cost;
			start -= 1;
			continue;
		}
		break;
	}
	(start, end)
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
	output::stdout_line(&stdout_palette.dim("Hugging Face model"))?;
	output::stdout_line("")?;
	output::stdout_line(&inspect_field("Revision", &model.revision.to_string()))?;
	output::stdout_line(&inspect_field("Downloads", &model.downloads.to_string()))?;
	output::stdout_line(&inspect_field("Likes", &model.likes.to_string()))?;
	output::stdout_line(&inspect_field(
		"License",
		output::terminal_safe_inline(model.license.as_deref().unwrap_or("not declared")).as_ref(),
	))?;
	output::stdout_line(&inspect_field(
		"Quantization",
		&quantization_summary(model.quantization),
	))?;
	output::stdout_line(&inspect_field("Traits", &trait_summary(&model.traits)))?;
	output::stdout_line(&inspect_field(
		"Weights",
		&optional_bytes(
			model
				.traits
				.sizing
				.as_ref()
				.and_then(|sizing| sizing.weights_bytes),
		),
	))?;
	output::stdout_line(&inspect_field(
		"Memory",
		&optional_bytes(
			model
				.traits
				.sizing
				.as_ref()
				.and_then(|sizing| sizing.estimated_residency_bytes),
		),
	))?;
	let context = model
		.traits
		.sizing
		.as_ref()
		.and_then(|sizing| sizing.max_context_tokens)
		.map_or_else(|| "unknown".to_string(), |tokens| tokens.to_string());
	output::stdout_line(&inspect_field("Context", &context))?;
	let compatibility = if model.compatible {
		stdout_palette.green("compatible")
	} else {
		stdout_palette.red("incompatible")
	};
	output::stdout_line(&inspect_field("Status", &compatibility))?;
	let fit = model.fit.as_ref().map_or_else(
		|| "unknown".to_string(),
		|fit| {
			format!(
				"{} · {} required / {} budget",
				if fit.fits { "fits" } else { "does not fit" },
				bytes(fit.required_bytes),
				bytes(fit.budget_bytes)
			)
		},
	);
	output::stdout_line(&inspect_field("Machine fit", &fit))?;
	let visible_diagnostics = if verbose {
		model.diagnostics.len()
	} else {
		model.diagnostics.len().min(10)
	};
	if visible_diagnostics > 0 {
		output::stdout_line("")?;
		output::stdout_line(&stdout_palette.yellow("Compatibility notes"))?;
	}
	for diagnostic in model.diagnostics.iter().take(visible_diagnostics) {
		output::stdout_line(&format!("  ! {}", output::terminal_safe_inline(diagnostic)))?;
	}
	if visible_diagnostics < model.diagnostics.len() {
		output::stdout_line(&stdout_palette.dim(&format!(
			"  {} more omitted · rerun with --verbose or --json",
			model.diagnostics.len() - visible_diagnostics
		)))?;
	}
	Ok(())
}

fn inspect_field(label: &str, value: &str) -> String {
	format!("  {label:<14}{value}")
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
	region: Option<LiveRegion>,
}

impl TtyProgress {
	const FRAMES: [&'static str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

	fn new(label: &'static str, enabled: bool) -> Self {
		Self {
			label,
			enabled,
			frame: 0,
			region: enabled.then(LiveRegion::stderr),
		}
	}

	fn tick(&mut self) -> anyhow::Result<()> {
		let Some(region) = &mut self.region else {
			return Ok(());
		};
		region.draw(&format!(
			"{} {}…",
			Self::FRAMES[self.frame % Self::FRAMES.len()],
			self.label
		))?;
		self.frame = self.frame.wrapping_add(1);
		Ok(())
	}

	fn clear(&mut self) -> anyhow::Result<()> {
		if let Some(region) = &mut self.region {
			region.clear()?;
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
	let models = emelex.models().context("initialize model manager")?;
	let cancellation = DownloadCancellation::default();
	let presentation = DownloadPresentation::new(model, json, stderr_palette, cancellation.clone());
	let watcher = DownloadCancellationWatcher::spawn(cancellation.clone(), tokio::signal::ctrl_c());
	let result = match revision {
		Some(revision) => {
			models
				.download_revision_controlled(
					model,
					revision,
					Some(presentation.observer()),
					Some(&cancellation),
				)
				.await
		}
		None => {
			models
				.download_controlled(model, Some(presentation.observer()), Some(&cancellation))
				.await
		}
	};
	let watcher_result = watcher.finish().await;
	let presentation_result = presentation.finish().await;
	watcher_result?;
	presentation_result?;
	let installed = result.with_context(|| format!("download {model}"))?;
	if json {
		output::json_line(&installed_json(&installed))
	} else {
		output::stdout_line(&format!(
			"{} {}",
			stdout_palette.green("✓ Installed"),
			stdout_palette.bold(&installed.reference().to_string())
		))?;
		output::stdout_line(&format!(
			"  {}",
			stdout_palette.dim(&output::terminal_safe_inline(
				&installed.path().display().to_string()
			))
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

struct DownloadPresentation {
	observer: DownloadObserver,
	output_error: Arc<Mutex<Option<anyhow::Error>>>,
	live: Option<LiveDownloadTask>,
}

impl DownloadPresentation {
	fn new(
		model: &HubModelId,
		json: bool,
		palette: Palette,
		cancellation: DownloadCancellation,
	) -> Self {
		let state = Arc::new(Mutex::new(DownloadUiState::new(model)));
		let output_error = Arc::new(Mutex::new(None));
		let live_enabled = !json && io::stderr().is_terminal();
		let live = live_enabled.then(|| {
			LiveDownloadTask::spawn(
				Arc::clone(&state),
				Arc::clone(&output_error),
				cancellation.clone(),
				palette,
			)
		});
		let observer_error = Arc::clone(&output_error);
		let observer_state = Arc::clone(&state);
		let observer_cancellation = cancellation;
		let observer = Arc::new(move |event: &DownloadEvent| {
			if observer_error
				.lock()
				.unwrap_or_else(std::sync::PoisonError::into_inner)
				.is_some()
			{
				return Ok(DownloadControl::Cancel);
			}
			let rendered = if json {
				json_download_event(event).map_or(Ok(()), |value| output::json_line(&value))
			} else if live_enabled {
				observer_state
					.lock()
					.unwrap_or_else(std::sync::PoisonError::into_inner)
					.observe(event, Instant::now());
				Ok(())
			} else {
				render_human_download_event(event, palette)
			};
			if let Err(error) = rendered {
				record_output_error(&observer_error, error);
				observer_cancellation.cancel();
				return Ok(DownloadControl::Cancel);
			}
			Ok(DownloadControl::Continue)
		});
		Self {
			observer,
			output_error,
			live,
		}
	}

	fn observer(&self) -> &DownloadObserver {
		&self.observer
	}

	async fn finish(mut self) -> anyhow::Result<()> {
		let live_result = if let Some(live) = self.live.take() {
			live.finish().await
		} else {
			Ok(())
		};
		let output_error = self
			.output_error
			.lock()
			.unwrap_or_else(std::sync::PoisonError::into_inner)
			.take();
		if let Some(error) = output_error {
			return Err(error);
		}
		live_result
	}
}

fn json_download_event(event: &DownloadEvent) -> Option<serde_json::Value> {
	match event {
		DownloadEvent::FileStarted {
			path,
			resumed,
			total,
		} => Some(serde_json::json!({
			"type": "download_file_started",
			"path": path,
			"resumed": resumed,
			"total": total,
		})),
		DownloadEvent::Progress {
			path,
			received,
			total,
		} => Some(serde_json::json!({
			"type": "download_progress",
			"path": path,
			"received": received,
			"total": total,
		})),
		DownloadEvent::Retrying {
			path,
			attempt,
			reason,
		} => Some(serde_json::json!({
			"type": "download_retrying",
			"path": path,
			"attempt": attempt,
			"reason": reason,
		})),
		DownloadEvent::FileVerified { path, sha256 } => Some(serde_json::json!({
			"type": "download_file_verified",
			"path": path,
			"sha256": sha256,
		})),
		_ => None,
	}
}

fn render_human_download_event(event: &DownloadEvent, palette: Palette) -> anyhow::Result<()> {
	let Some(line) = human_download_event_line(event) else {
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
		DownloadEvent::TransferStarted { files, total } => Some(format!(
			"downloading {} ({})",
			counted(*files, "file", "files"),
			bytes(*total)
		)),
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
		DownloadEvent::TransferCompleted { files, total } => Some(format!(
			"transferred {} ({})",
			counted(*files, "file", "files"),
			bytes(*total)
		)),
		_ => None,
	}
}

fn counted(count: usize, singular: &str, plural: &str) -> String {
	format!("{count} {}", if count == 1 { singular } else { plural })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DownloadPhase {
	Preparing,
	Transferring,
	Finalizing,
}

#[derive(Debug, Clone, Default)]
struct DownloadFileProgress {
	received: u64,
	total: u64,
}

#[derive(Debug, Clone)]
struct DownloadRetry {
	path: String,
	attempt: usize,
	reason: String,
}

#[derive(Debug, Clone)]
struct DownloadUiState {
	model: String,
	phase: DownloadPhase,
	total_files: usize,
	total_bytes: u64,
	files: BTreeMap<String, DownloadFileProgress>,
	verified: BTreeSet<String>,
	current_path: Option<String>,
	retry: Option<DownloadRetry>,
	network_bytes: u64,
	samples: VecDeque<(Instant, u64)>,
	last_network_at: Option<Instant>,
}

impl DownloadUiState {
	fn new(model: &HubModelId) -> Self {
		Self {
			model: model.to_string(),
			phase: DownloadPhase::Preparing,
			total_files: 0,
			total_bytes: 0,
			files: BTreeMap::new(),
			verified: BTreeSet::new(),
			current_path: None,
			retry: None,
			network_bytes: 0,
			samples: VecDeque::new(),
			last_network_at: None,
		}
	}

	fn observe(&mut self, event: &DownloadEvent, now: Instant) {
		match event {
			DownloadEvent::TransferStarted { files, total } => {
				self.phase = DownloadPhase::Transferring;
				self.total_files = *files;
				self.total_bytes = *total;
				self.samples.clear();
				self.samples.push_back((now, self.network_bytes));
			}
			DownloadEvent::FileStarted {
				path,
				resumed,
				total,
			} => {
				self.phase = DownloadPhase::Transferring;
				self.current_path = Some(path.clone());
				self.retry = None;
				let file = self.files.entry(path.clone()).or_default();
				file.received = (*resumed).min(*total);
				file.total = *total;
			}
			DownloadEvent::Progress {
				path,
				received,
				total,
			} => {
				self.phase = DownloadPhase::Transferring;
				self.current_path = Some(path.clone());
				self.retry = None;
				let file = self.files.entry(path.clone()).or_default();
				let received = (*received).min(*total);
				let delta = if received >= file.received {
					received - file.received
				} else {
					received
				};
				file.received = received;
				file.total = *total;
				if delta > 0 {
					self.network_bytes = self.network_bytes.saturating_add(delta);
					self.last_network_at = Some(now);
					self.samples.push_back((now, self.network_bytes));
					self.prune_samples(now);
				}
			}
			DownloadEvent::Retrying {
				path,
				attempt,
				reason,
			} => {
				self.phase = DownloadPhase::Transferring;
				self.current_path = Some(path.clone());
				self.retry = Some(DownloadRetry {
					path: path.clone(),
					attempt: *attempt,
					reason: reason.clone(),
				});
			}
			DownloadEvent::FileVerified { path, .. } => {
				self.current_path = Some(path.clone());
				self.verified.insert(path.clone());
				if let Some(file) = self.files.get_mut(path) {
					file.received = file.total;
				}
				self.retry = None;
			}
			DownloadEvent::TransferCompleted { files, total } => {
				self.phase = DownloadPhase::Finalizing;
				self.total_files = *files;
				self.total_bytes = *total;
				self.retry = None;
			}
			_ => {}
		}
	}

	fn prune_samples(&mut self, now: Instant) {
		let cutoff = now.checked_sub(Duration::from_secs(3)).unwrap_or(now);
		while self.samples.len() > 2
			&& self
				.samples
				.get(1)
				.is_some_and(|(timestamp, _)| *timestamp <= cutoff)
		{
			self.samples.pop_front();
		}
	}

	fn received_bytes(&self) -> u64 {
		if self.phase == DownloadPhase::Finalizing {
			return self.total_bytes;
		}
		self.files
			.values()
			.map(|file| file.received.min(file.total))
			.fold(0_u64, u64::saturating_add)
			.min(self.total_bytes)
	}

	fn bytes_per_second(&self, now: Instant) -> Option<u64> {
		if self
			.last_network_at
			.is_none_or(|last| now.saturating_duration_since(last) > Duration::from_secs(2))
		{
			return None;
		}
		let (first_at, first_bytes) = self.samples.front()?;
		let (last_at, last_bytes) = self.samples.back()?;
		let elapsed_millis = last_at.saturating_duration_since(*first_at).as_millis();
		if elapsed_millis < 200 || last_bytes <= first_bytes {
			return None;
		}
		let bytes = u128::from(*last_bytes - *first_bytes);
		u64::try_from(bytes.saturating_mul(1_000) / elapsed_millis)
			.ok()
			.filter(|speed| *speed > 0)
	}
}

struct LiveDownloadTask {
	stop: Option<tokio::sync::oneshot::Sender<()>>,
	task: Option<tokio::task::JoinHandle<()>>,
}

impl LiveDownloadTask {
	fn spawn(
		state: Arc<Mutex<DownloadUiState>>,
		output_error: Arc<Mutex<Option<anyhow::Error>>>,
		cancellation: DownloadCancellation,
		palette: Palette,
	) -> Self {
		let (stop, mut stopped) = tokio::sync::oneshot::channel();
		let task = tokio::spawn(async move {
			let mut region = LiveRegion::stderr();
			let mut interval = tokio::time::interval(Duration::from_millis(80));
			interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
			let mut frame = 0_usize;
			loop {
				tokio::select! {
					_ = &mut stopped => break,
					_ = interval.tick() => {
						let snapshot = state
							.lock()
							.unwrap_or_else(std::sync::PoisonError::into_inner)
							.clone();
						let rendered = render_download_frame(
							&snapshot,
							frame,
							Instant::now(),
							usize::from(region.size().1).max(1),
							palette,
						);
						if let Err(error) = region.draw(&rendered) {
							record_output_error(&output_error, error);
							cancellation.cancel();
							break;
						}
						frame = frame.wrapping_add(1);
					}
				}
			}
			if let Err(error) = region.clear() {
				record_output_error(&output_error, error);
				cancellation.cancel();
			}
		});
		Self {
			stop: Some(stop),
			task: Some(task),
		}
	}

	async fn finish(mut self) -> anyhow::Result<()> {
		if let Some(stop) = self.stop.take() {
			let _ = stop.send(());
		}
		let Some(task) = self.task.take() else {
			return Err(anyhow::anyhow!("download progress task is absent"));
		};
		task.await.context("join download progress task")
	}
}

impl Drop for LiveDownloadTask {
	fn drop(&mut self) {
		if let Some(task) = &self.task {
			task.abort();
		}
	}
}

fn record_output_error(
	destination: &Mutex<Option<anyhow::Error>>,
	error: impl Into<anyhow::Error>,
) {
	let mut destination = destination
		.lock()
		.unwrap_or_else(std::sync::PoisonError::into_inner);
	if destination.is_none() {
		*destination = Some(error.into());
	}
}

fn render_download_frame(
	state: &DownloadUiState,
	frame: usize,
	now: Instant,
	columns: usize,
	palette: Palette,
) -> String {
	const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
	let spinner = palette.cyan(FRAMES[frame % FRAMES.len()]);
	let model = output::terminal_safe_inline(&state.model);
	if state.phase == DownloadPhase::Preparing {
		return fit_line(&format!("{spinner} Preparing {model}"), columns);
	}
	if state.phase == DownloadPhase::Finalizing {
		return fit_line(&format!("{spinner} Finalizing {model}"), columns);
	}
	if let Some(retry) = &state.retry {
		let path = output::terminal_safe_inline(&retry.path);
		let reason = output::terminal_safe_inline(&retry.reason);
		let line = format!(
			"{} Retrying {path} · attempt {} · {reason}",
			palette.yellow(FRAMES[frame % FRAMES.len()]),
			retry.attempt
		);
		return fit_line(&line, columns);
	}

	let received = state.received_bytes();
	let percent = received
		.saturating_mul(100)
		.checked_div(state.total_bytes)
		.unwrap_or(0)
		.min(100);
	if columns < 52 {
		return fit_line(
			&format!(
				"{spinner} Downloading {percent:>3}% · {}/{}",
				bytes(received),
				bytes(state.total_bytes)
			),
			columns,
		);
	}

	let bar_width = if columns >= 96 {
		24
	} else if columns >= 72 {
		16
	} else {
		10
	};
	let bar = progress_bar(percent, bar_width, palette);
	let mut lines = vec![fit_line(
		&format!("{spinner} Downloading  {bar}  {percent:>3}%"),
		columns,
	)];
	let mut details = vec![
		format!("{} / {}", bytes(received), bytes(state.total_bytes)),
		format!("{}/{} verified", state.verified.len(), state.total_files),
	];
	if let Some(speed) = state.bytes_per_second(now) {
		details.insert(1, format!("{}/s", bytes(speed)));
		let remaining = state.total_bytes.saturating_sub(received);
		details.insert(2, remaining_time(remaining.div_ceil(speed)));
	}
	lines.push(fit_line(
		&palette.dim(&format!("  {}", details.join(" · "))),
		columns,
	));
	if columns >= 72
		&& let Some(path) = &state.current_path
	{
		lines.push(fit_line(
			&palette.dim(&format!("  {}", output::terminal_safe_inline(path))),
			columns,
		));
	}
	lines.join("\n")
}

fn progress_bar(percent: u64, width: usize, palette: Palette) -> String {
	let filled = usize::try_from(percent)
		.unwrap_or(width)
		.saturating_mul(width)
		/ 100;
	format!(
		"{}{}",
		palette.cyan(&"━".repeat(filled.min(width))),
		palette.dim(&"─".repeat(width.saturating_sub(filled)))
	)
}

fn remaining_time(seconds: u64) -> String {
	if seconds < 60 {
		format!("{seconds}s left")
	} else {
		format!("{}m {:02}s left", seconds / 60, seconds % 60)
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

	fn search_model(id: &str, revision_byte: char) -> HubModel {
		let mut traits = ModelTraits::default();
		traits.mlx = true;
		traits.tasks.extend([Task::TextGeneration, Task::Chat]);
		serde_json::from_value(serde_json::json!({
			"id": id,
			"revision": revision_byte.to_string().repeat(40),
			"downloads": 42,
			"likes": 7,
			"tags": ["mlx"],
			"library": "mlx",
			"license": "apache-2.0",
			"traits": traits,
			"quantization": {
				"kind": "configured",
				"mode": "affine",
				"bits": 4,
				"group_size": 64,
				"has_layer_overrides": false
			},
			"compatible": true,
			"files": ["config.json", "model.safetensors"],
			"diagnostics": [],
			"fit": null
		}))
		.expect("valid Hub model fixture")
	}

	#[test]
	fn cli_search_implicitly_requires_tool_capability() {
		let requirements = cli_search_requirements(Vec::new()).expect("CLI search requirements");

		assert_eq!(
			requirements
				.iter()
				.map(TraitFilter::as_str)
				.collect::<Vec<_>>(),
			[CLI_REQUIRED_SEARCH_TRAIT]
		);
	}

	#[test]
	fn cli_search_deduplicates_explicit_and_implicit_requirements() {
		let tools = TraitFilter::parse(CLI_REQUIRED_SEARCH_TRAIT).expect("tool capability");
		let reasoning = TraitFilter::parse("interaction:reasoning").expect("reasoning capability");
		let requirements = cli_search_requirements(vec![tools.clone(), reasoning, tools])
			.expect("CLI search requirements");

		assert_eq!(
			requirements
				.iter()
				.map(TraitFilter::as_str)
				.collect::<Vec<_>>(),
			["interaction:reasoning", CLI_REQUIRED_SEARCH_TRAIT]
		);
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
	fn inline_search_selector_marks_the_cards_without_a_duplicate_list() {
		let items = vec![
			search_model("mlx-community/first-4bit", 'a'),
			search_model("mlx-community/second-4bit", 'b'),
			search_model("mlx-community/third-4bit", 'c'),
		];
		let mut installed = InstalledHubIndex::new();
		installed
			.entry(items[0].id.clone())
			.or_default()
			.insert(items[0].revision.clone());

		let frame = render_search_selector_frame(
			&items,
			&installed,
			1,
			17,
			(24, 80),
			Palette::stdout(crate::style::ColorMode::Never),
		);

		assert!(frame.contains("❯ mlx-community/second-4bit"));
		assert!(frame.contains("┃   Status  not downloaded"));
		assert!(frame.contains("Quant   4-bit affine · group 64"));
		assert!(frame.contains("↑↓ move  ·  enter download  ·  esc close  ·  2/3"));
		assert!(!frame.contains("Choose a model"));
		assert_eq!(frame.matches('❯').count(), 1);
		assert!(frame.lines().count() <= 24);
		assert!(
			frame
				.lines()
				.all(|line| dialoguer::console::measure_text_width(line) < 80)
		);
	}

	#[test]
	fn inline_search_selector_adapts_to_a_narrow_terminal() {
		let items = vec![
			search_model("mlx-community/a-very-long-first-model-name-4bit", 'a'),
			search_model("mlx-community/second-4bit", 'b'),
		];

		let frame = render_search_selector_frame(
			&items,
			&InstalledHubIndex::new(),
			0,
			200,
			(8, 40),
			Palette::stdout(crate::style::ColorMode::Never),
		);

		assert!(frame.contains('❯'));
		assert!(frame.contains("4-bit"));
		assert!(frame.contains("1/2"));
		assert!(frame.lines().count() <= 8);
		assert!(
			frame
				.lines()
				.all(|line| dialoguer::console::measure_text_width(line) < 40)
		);
	}

	#[test]
	fn inline_search_selector_reserves_a_cursor_row_at_every_height() {
		let items = vec![search_model("mlx-community/tiny-4bit", 'a')];
		let palette = Palette::stdout(crate::style::ColorMode::Never);

		for terminal_rows in 1..=5 {
			let frame = render_search_selector_frame(
				&items,
				&InstalledHubIndex::new(),
				0,
				1,
				(terminal_rows, 80),
				palette,
			);
			assert!(frame.lines().count() < usize::from(terminal_rows));
		}
		let tiny =
			render_search_selector_frame(&items, &InstalledHubIndex::new(), 0, 1, (3, 80), palette);
		assert!(tiny.contains("4-bit"));
		assert!(tiny.contains("not downloaded"));
	}

	#[test]
	fn inline_search_selector_keys_navigate_and_select_exact_results() {
		use dialoguer::console::Key;

		assert_eq!(
			search_selector_action(&Key::ArrowDown, 2, 3),
			SearchSelectorAction::Move(0)
		);
		assert_eq!(
			search_selector_action(&Key::ArrowUp, 0, 3),
			SearchSelectorAction::Move(2)
		);
		assert_eq!(
			search_selector_action(&Key::PageDown, 1, 10),
			SearchSelectorAction::Move(6)
		);
		assert_eq!(
			search_selector_action(&Key::Home, 8, 10),
			SearchSelectorAction::Move(0)
		);
		assert_eq!(
			search_selector_action(&Key::End, 1, 10),
			SearchSelectorAction::Move(9)
		);
		assert_eq!(
			search_selector_action(&Key::Enter, 1, 3),
			SearchSelectorAction::Select
		);
		assert_eq!(
			search_selector_action(&Key::Escape, 1, 3),
			SearchSelectorAction::Cancel
		);
		assert_eq!(
			search_selector_action(&Key::CtrlC, 1, 3),
			SearchSelectorAction::Interrupt
		);
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
		assert!(
			human_download_event_line(&DownloadEvent::Progress {
				path: "model".to_string(),
				received: 1,
				total: 2,
			})
			.is_none()
		);
	}

	#[test]
	fn download_progress_aggregates_resumes_speed_and_verification() {
		let id = HubModelId::parse("owner/model").expect("valid Hub ID");
		let mut state = DownloadUiState::new(&id);
		let started = Instant::now();
		state.observe(
			&DownloadEvent::TransferStarted {
				files: 2,
				total: 100,
			},
			started,
		);
		state.observe(
			&DownloadEvent::FileStarted {
				path: "first.safetensors".to_string(),
				resumed: 20,
				total: 60,
			},
			started,
		);
		state.observe(
			&DownloadEvent::Progress {
				path: "first.safetensors".to_string(),
				received: 40,
				total: 60,
			},
			started + Duration::from_secs(1),
		);
		state.observe(
			&DownloadEvent::FileVerified {
				path: "first.safetensors".to_string(),
				sha256: "a".repeat(64),
			},
			started + Duration::from_secs(1),
		);
		state.observe(
			&DownloadEvent::FileStarted {
				path: "second.safetensors".to_string(),
				resumed: 0,
				total: 40,
			},
			started + Duration::from_secs(1),
		);
		state.observe(
			&DownloadEvent::Progress {
				path: "second.safetensors".to_string(),
				received: 20,
				total: 40,
			},
			started + Duration::from_secs(2),
		);

		let frame = render_download_frame(
			&state,
			3,
			started + Duration::from_secs(2),
			96,
			Palette::stderr(crate::style::ColorMode::Never),
		);
		assert!(frame.contains(" 80%"));
		assert!(frame.contains("1/2 verified"));
		assert!(frame.contains("20 B/s"));
		assert!(frame.contains("second.safetensors"));
		assert!(frame.lines().all(|line| {
			dialoguer::console::measure_text_width(line) < 96 && !line.contains('\r')
		}));
	}

	#[test]
	fn download_progress_has_truthful_retry_and_finalizing_phases() {
		let id = HubModelId::parse("owner/model").expect("valid Hub ID");
		let mut state = DownloadUiState::new(&id);
		let now = Instant::now();
		state.observe(
			&DownloadEvent::Retrying {
				path: "model\nforged.safetensors".to_string(),
				attempt: 2,
				reason: "network\treset\u{202e}".to_string(),
			},
			now,
		);
		let retry = render_download_frame(
			&state,
			0,
			now,
			80,
			Palette::stderr(crate::style::ColorMode::Never),
		);
		assert!(retry.contains("Retrying"));
		assert!(retry.contains("attempt 2"));
		assert!(!retry.contains('\n'));
		assert!(!retry.contains('\t'));
		assert!(!retry.contains('\u{202e}'));

		state.observe(
			&DownloadEvent::TransferCompleted {
				files: 1,
				total: 42,
			},
			now,
		);
		let finalizing = render_download_frame(
			&state,
			1,
			now,
			40,
			Palette::stderr(crate::style::ColorMode::Never),
		);
		assert_eq!(finalizing, "⠙ Finalizing owner/model");
	}

	#[test]
	fn json_download_events_remain_backward_compatible() {
		assert!(
			json_download_event(&DownloadEvent::TransferStarted {
				files: 2,
				total: 42,
			})
			.is_none()
		);
		assert!(
			json_download_event(&DownloadEvent::TransferCompleted {
				files: 2,
				total: 42,
			})
			.is_none()
		);
		assert_eq!(
			json_download_event(&DownloadEvent::Progress {
				path: "model.safetensors".to_string(),
				received: 21,
				total: 42,
			})
			.expect("progress JSON")["type"],
			"download_progress"
		);
	}

	#[test]
	fn count_and_eta_copy_handles_singular_and_long_transfers() {
		assert_eq!(counted(1, "file", "files"), "1 file");
		assert_eq!(counted(2, "file", "files"), "2 files");
		assert_eq!(remaining_time(10), "10s left");
		assert_eq!(remaining_time(125), "2m 05s left");
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
