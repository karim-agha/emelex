//! Hugging Face discovery and download presentation.

use std::{
	collections::BTreeMap,
	future::Future,
	io::{self, IsTerminal as _},
	sync::{Arc, Mutex},
	time::Duration,
};

use anyhow::Context as _;
use emelex::{
	Emelex,
	hub::{
		DownloadCancellation, DownloadControl, DownloadEvent, DownloadObserver, HubSearch,
		REMOTE_FILTERS,
	},
	model::{HubModelId, InstalledModel},
};

use super::{
	args::HubCommand,
	output,
	style::{Palette, bytes},
};

pub(crate) async fn run(
	emelex: &Emelex,
	command: HubCommand,
	json: bool,
	stdout_palette: Palette,
	stderr_palette: Palette,
) -> anyhow::Result<()> {
	match command {
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
	let mut search = HubSearch::default().requirements(require);
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
	for model in &page.items {
		let weights = optional_bytes(
			model
				.traits
				.sizing
				.as_ref()
				.and_then(|sizing| sizing.weights_bytes),
		);
		let fit = model.fit.as_ref().map_or_else(
			|| "fit unknown".to_string(),
			|fit| {
				format!(
					"{} {} / {}",
					if fit.fits { "fits" } else { "does not fit" },
					bytes(fit.required_bytes),
					bytes(fit.budget_bytes)
				)
			},
		);
		output::stdout_line(&format!(
			"{}  {weights}  {fit}  {}",
			stdout_palette.cyan(model.id.as_str()),
			trait_summary(&model.traits)
		))?;
	}
	let visible_diagnostics = if verbose {
		page.diagnostics.len()
	} else {
		page.diagnostics.len().min(10)
	};
	for diagnostic in page.diagnostics.iter().take(visible_diagnostics) {
		let candidate = diagnostic
			.id
			.as_ref()
			.map_or_else(|| "candidate".to_string(), ToString::to_string);
		let message = format!("skipped {candidate}: {}", diagnostic.message);
		output::stderr_line(&stderr_palette.yellow(&output::terminal_safe_inline(&message)))?;
	}
	if visible_diagnostics < page.diagnostics.len() {
		output::stderr_line(&stderr_palette.dim(&format!(
			"{} more diagnostic(s) omitted; rerun with --verbose or --json",
			page.diagnostics.len() - visible_diagnostics
		)))?;
	}
	output::stderr_line(&stderr_palette.dim(&format!(
		"{} compatible result(s); {} candidate(s) scanned{}",
		page.items.len(),
		page.scanned,
		page.next_cursor
			.as_ref()
			.map_or_else(String::new, |cursor| format!("; next cursor {cursor}"))
	)))?;
	Ok(())
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
	let (observer, output_error) = download_observer(json, stderr_palette);
	let cancellation = DownloadCancellation::default();
	let models = emelex.models().context("initialize model manager")?;
	let watcher = DownloadCancellationWatcher::spawn(cancellation.clone(), tokio::signal::ctrl_c());
	let result = models
		.download_controlled(model, Some(&observer), Some(&cancellation))
		.await;
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

#[cfg(test)]
mod tests {
	use super::*;

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
