//! Managed local model lifecycle presentation.

use std::{collections::BTreeSet, time::Duration};

use anyhow::{Context as _, bail};
use emelex::{
	Emelex,
	config::Config,
	model::{InstalledModel, LocalModelName, ModelRef},
	models::{ImportMode, ImportOptions, ImportSourceDisposition, ModelManager},
};

use super::{
	args::{ModelCommand, ModelImportArgs, ModelsCommand},
	hub_cmd::{download, installed_json, trait_summary},
	output,
	style::{Palette, bytes},
};

pub(crate) fn run_model(
	emelex: &Emelex,
	command: ModelCommand,
	json: bool,
	stdout_palette: Palette,
	stderr_palette: Palette,
) -> anyhow::Result<()> {
	match command {
		ModelCommand::Import(args) => {
			import_checkpoint(emelex, args, json, stdout_palette, stderr_palette)
		}
	}
}

pub(crate) async fn run(
	emelex: &Emelex,
	command: ModelsCommand,
	json: bool,
	stdout_palette: Palette,
	stderr_palette: Palette,
) -> anyhow::Result<()> {
	match command {
		ModelsCommand::List => list(emelex, json, stdout_palette, stderr_palette),
		ModelsCommand::Import { name, path } => {
			let name = LocalModelName::parse(name).context("validate local model name")?;
			let installed = emelex
				.models()
				.context("initialize model manager")?
				.import(&name, &path)
				.with_context(|| format!("import checkpoint {}", path.display()))?;
			write_import_result(&installed, None, json, stdout_palette, stderr_palette)
		}
		ModelsCommand::Default { model, clear } => {
			default_model(emelex, model, clear, json, stdout_palette)
		}
		ModelsCommand::Update { model } => {
			update(emelex, model, json, stdout_palette, stderr_palette).await
		}
		ModelsCommand::Remove { model } => {
			let installed = emelex
				.models()
				.context("initialize model manager")?
				.resolve_snapshot(&model)
				.with_context(|| format!("resolve installed snapshot {model}"))?;
			let quarantine = emelex
				.models()
				.context("initialize model manager")?
				.remove(&installed)
				.with_context(|| format!("quarantine snapshot {model}"))?;
			if json {
				output::json_line(&serde_json::json!({
					"reference": installed.reference(),
					"snapshot": installed.snapshot_id(),
					"quarantine": quarantine,
				}))
			} else {
				output::stdout_line(&format!(
					"{} {}",
					stdout_palette.green("✓ Quarantined"),
					stdout_palette.bold(&output::terminal_safe_inline(
						&installed.snapshot_id().to_string()
					))
				))?;
				output::stdout_line(&format!(
					"  {}",
					stdout_palette.dim(&output::terminal_safe_inline(
						&quarantine.display().to_string()
					))
				))
			}
		}
		ModelsCommand::Verify { model } => {
			verify(emelex, model, json, stdout_palette, stderr_palette)
		}
		ModelsCommand::Gc { older_than_days } => {
			let age = Duration::from_secs(older_than_days.saturating_mul(24 * 60 * 60));
			let removed = emelex
				.models()
				.context("initialize model manager")?
				.gc_quarantine(age)
				.context("garbage-collect quarantined models")?;
			if json {
				output::json_line(&serde_json::json!({"removed": removed}))
			} else if removed == 0 {
				output::stdout_line(&format!(
					"No quarantined snapshots older than {older_than_days} days."
				))
			} else {
				output::stdout_line(&format!(
					"{} {}",
					stdout_palette.green("✓ Removed"),
					quarantined_snapshot_count(removed)
				))
			}
		}
		ModelsCommand::Path { model } => {
			let installed = emelex
				.models()
				.context("initialize model manager")?
				.resolve(&model)
				.with_context(|| format!("resolve installed model {model}"))?;
			if json {
				output::json_line(&serde_json::json!({
					"reference": model,
					"snapshot": installed.snapshot_id(),
					"path": installed.path(),
				}))
			} else {
				output::stdout_line(&output::terminal_safe_inline(
					&installed.path().display().to_string(),
				))
			}
		}
	}
}

fn import_checkpoint(
	emelex: &Emelex,
	args: ModelImportArgs,
	json: bool,
	stdout_palette: Palette,
	stderr_palette: Palette,
) -> anyhow::Result<()> {
	let name = import_name(&args.path, args.name)?;
	let mode = if args.move_source {
		ImportMode::Move
	} else if args.symlink {
		ImportMode::Symlink
	} else {
		ImportMode::Copy
	};
	let options = ImportOptions::default().mode(mode);
	let outcome = emelex
		.models()
		.context("initialize model manager")?
		.import_with_options(&name, &args.path, options)
		.with_context(|| format!("import checkpoint {}", args.path.display()))?;
	write_import_result(
		outcome.installed(),
		Some(outcome.disposition()),
		json,
		stdout_palette,
		stderr_palette,
	)
}

fn import_name(path: &std::path::Path, explicit: Option<String>) -> anyhow::Result<LocalModelName> {
	if let Some(name) = explicit {
		return LocalModelName::parse(name).context("validate local model name");
	}
	let canonical = std::fs::canonicalize(path)
		.with_context(|| format!("resolve checkpoint directory {}", path.display()))?;
	let name = canonical
		.file_name()
		.and_then(std::ffi::OsStr::to_str)
		.context(
			"derive local model name from checkpoint directory; pass an ASCII name with --name",
		)?;
	LocalModelName::parse(name)
		.context("derive local model name from checkpoint directory; pass a valid name with --name")
}

fn write_import_result(
	installed: &InstalledModel,
	disposition: Option<&ImportSourceDisposition>,
	json: bool,
	stdout_palette: Palette,
	stderr_palette: Palette,
) -> anyhow::Result<()> {
	if json {
		let mut result = installed_json(installed);
		if let (Some(fields), Some(disposition)) = (result.as_object_mut(), disposition) {
			fields.insert(
				"source_disposition".to_string(),
				serde_json::to_value(disposition).context("encode import source disposition")?,
			);
		}
		return output::json_line(&result);
	}
	output::stdout_line(&format!(
		"{} {}",
		stdout_palette.green("✓ Installed"),
		stdout_palette.bold(&output::terminal_safe_inline(
			&installed.reference().to_string()
		))
	))?;
	output::stdout_line(&format!(
		"  {}",
		stdout_palette.dim(&output::terminal_safe_inline(
			&installed.path().display().to_string()
		))
	))?;
	if let Some(ImportSourceDisposition::Retained {
		message: warning, ..
	}) = disposition
	{
		output::stderr_line(
			&stderr_palette.yellow(&format!("! {}", output::terminal_safe_inline(warning))),
		)?;
	}
	Ok(())
}

fn default_model(
	emelex: &Emelex,
	model: Option<ModelRef>,
	clear: bool,
	json: bool,
	stdout_palette: Palette,
) -> anyhow::Result<()> {
	if clear {
		Config::write_global_default_model(emelex.home(), None)
			.context("clear global default model")?;
		if json {
			output::json_line(&serde_json::json!({"default_model": null}))
		} else {
			output::stdout_line(&stdout_palette.green("✓ Default model cleared"))
		}
	} else if let Some(model) = model {
		emelex
			.models()
			.context("initialize model manager")?
			.resolve(&model)
			.with_context(|| format!("resolve installed model {model}"))?;
		Config::write_global_default_model(emelex.home(), Some(&model))
			.context("set global default model")?;
		if json {
			output::json_line(&serde_json::json!({"default_model": model}))
		} else {
			output::stdout_line(&format!(
				"{} {}",
				stdout_palette.green("✓ Default model set to"),
				stdout_palette.bold(&output::terminal_safe_inline(&model.to_string()))
			))
		}
	} else if json {
		output::json_line(&serde_json::json!({
			"default_model": emelex.config().default_model.as_ref()
		}))
	} else if let Some(model) = emelex.config().default_model.as_ref() {
		output::stdout_line(&format!(
			"Default model  {}",
			stdout_palette.bold(&output::terminal_safe_inline(&model.to_string()))
		))
	} else {
		output::stdout_line("Default model  not set")?;
		output::stdout_line(
			&stdout_palette.dim("  Set one with: emelex models default NAMESPACE/REPO"),
		)
	}
}

fn list(
	emelex: &Emelex,
	json: bool,
	stdout_palette: Palette,
	stderr_palette: Palette,
) -> anyhow::Result<()> {
	let inventory = model_manager(emelex)?
		.inventory()
		.context("inventory installed models")?;
	if json {
		let models = inventory
			.models
			.iter()
			.map(installed_json)
			.collect::<Vec<_>>();
		let diagnostics = inventory
			.diagnostics
			.iter()
			.map(|diagnostic| {
				serde_json::json!({
					"path": diagnostic.path,
					"message": diagnostic.message,
				})
			})
			.collect::<Vec<_>>();
		return output::json_line(&serde_json::json!({
			"models": models,
			"diagnostics": diagnostics,
		}));
	}
	let rows = inventory
		.models
		.iter()
		.map(|model| HumanModelRow {
			snapshot: model.snapshot_id().to_string(),
			weights: model
				.manifest()
				.traits()
				.sizing
				.as_ref()
				.and_then(|sizing| sizing.weights_bytes)
				.map_or_else(|| "unknown".to_string(), bytes),
			traits: trait_summary(model.manifest().traits()),
			is_default: emelex.config().default_model.as_ref() == Some(model.reference()),
		})
		.collect::<Vec<_>>();
	for line in human_inventory_lines(&rows, stdout_palette) {
		output::stdout_line(&line)?;
	}
	for diagnostic in inventory.diagnostics {
		output::stderr_line(&stderr_palette.yellow(&format!(
			"! {}",
			invalid_model_line(&diagnostic.path, &diagnostic.message)
		)))?;
	}
	Ok(())
}

async fn update(
	emelex: &Emelex,
	selected: Option<ModelRef>,
	json: bool,
	stdout_palette: Palette,
	stderr_palette: Palette,
) -> anyhow::Result<()> {
	let references = match selected {
		Some(ModelRef::Hub(id)) => BTreeSet::from([id]),
		Some(ModelRef::Local(_)) => bail!("local imports cannot be updated from Hugging Face"),
		Some(_) => bail!("this model reference kind cannot be updated"),
		None => emelex
			.models()
			.context("initialize model manager")?
			.list()
			.context("list installed models")?
			.into_iter()
			.filter_map(|model| model.reference().as_hub().cloned())
			.collect(),
	};
	if references.is_empty() {
		bail!("no installed Hugging Face models to update");
	}
	for reference in references {
		download(emelex, &reference, json, stdout_palette, stderr_palette).await?;
	}
	Ok(())
}

fn verify(
	emelex: &Emelex,
	selected: Option<ModelRef>,
	json: bool,
	stdout_palette: Palette,
	stderr_palette: Palette,
) -> anyhow::Result<()> {
	let (installed, diagnostics) = if let Some(reference) = selected {
		(
			vec![
				emelex
					.models()
					.context("initialize model manager")?
					.resolve(&reference)
					.with_context(|| format!("resolve installed model {reference}"))?,
			],
			Vec::new(),
		)
	} else {
		let inventory = model_manager(emelex)?
			.inventory()
			.context("inventory installed models")?;
		(inventory.models, inventory.diagnostics)
	};
	if installed.is_empty() {
		bail!("no installed models to verify");
	}
	for diagnostic in diagnostics {
		if json {
			output::json_line(&serde_json::json!({
				"type": "invalid_model",
				"path": diagnostic.path,
				"message": diagnostic.message,
			}))?;
		} else {
			output::stderr_line(
				&stderr_palette.yellow(&invalid_model_line(&diagnostic.path, &diagnostic.message)),
			)?;
		}
	}
	for model in installed {
		let verification = emelex
			.models()
			.context("initialize model manager")?
			.verify(&model)
			.with_context(|| format!("verify {}", model.reference()))?;
		if json {
			output::json_line(&serde_json::json!({
				"reference": model.reference(),
				"snapshot": model.snapshot_id(),
				"path": model.path(),
				"compatibility": verification.compatibility,
			}))?;
		} else {
			output::stdout_line(&format!(
				"{} {}",
				stdout_palette.green("✓ Verified"),
				stdout_palette.bold(&output::terminal_safe_inline(
					&model.snapshot_id().to_string()
				))
			))?;
		}
	}
	Ok(())
}

struct HumanModelRow {
	snapshot: String,
	weights: String,
	traits: String,
	is_default: bool,
}

fn human_inventory_lines(rows: &[HumanModelRow], palette: Palette) -> Vec<String> {
	if rows.is_empty() {
		return vec![
			"No models installed.".to_string(),
			palette.dim("  Find one with: emelex hub search QUERY"),
		];
	}

	let mut lines = vec![format!(
		"{}  {}",
		palette.bold("Installed models"),
		palette.dim(&rows.len().to_string())
	)];
	for row in rows {
		lines.push(String::new());
		let snapshot = palette.cyan(&output::terminal_safe_inline(&row.snapshot));
		if row.is_default {
			lines.push(format!("  {snapshot}  {}", palette.green("default")));
		} else {
			lines.push(format!("  {snapshot}"));
		}
		lines.push(format!(
			"    {}",
			palette.dim(&format!(
				"{} · {}",
				output::terminal_safe_inline(&row.weights),
				output::terminal_safe_inline(&row.traits)
			))
		));
	}
	lines
}

fn quarantined_snapshot_count(count: usize) -> String {
	if count == 1 {
		"1 quarantined snapshot".to_string()
	} else {
		format!("{count} quarantined snapshots")
	}
}

fn invalid_model_line(path: &std::path::Path, message: &str) -> String {
	let path_display = path.display().to_string();
	let path = output::terminal_safe_inline(&path_display);
	let message = output::terminal_safe_inline(message);
	format!("invalid model entry {path}: {message}")
}

fn model_manager(emelex: &Emelex) -> anyhow::Result<&ModelManager> {
	emelex.models().context("initialize model manager")
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn invalid_model_diagnostic_stays_on_one_human_row() {
		let line = invalid_model_line(
			std::path::Path::new("/tmp/model\nforged"),
			"bad\tentry\u{202e}",
		);
		assert!(!line.contains('\n'));
		assert!(!line.contains('\t'));
		assert!(!line.contains('\u{202e}'));
		assert!(line.contains('\u{240a}'));
		assert!(line.contains('\u{2409}'));
	}

	#[test]
	fn human_inventory_uses_explicit_default_label_and_safe_hierarchy() {
		let rows = [
			HumanModelRow {
				snapshot: "mlx/model\nforged".to_string(),
				weights: "4 GiB".to_string(),
				traits: "input=text\u{1b}[2J".to_string(),
				is_default: true,
			},
			HumanModelRow {
				snapshot: "local/other".to_string(),
				weights: "unknown".to_string(),
				traits: "input=text".to_string(),
				is_default: false,
			},
		];
		let lines = human_inventory_lines(
			&rows,
			Palette::stdout(super::super::style::ColorMode::Never),
		);
		assert_eq!(lines[0], "Installed models  2");
		assert_eq!(lines[2], "  mlx/model\u{240a}forged  default");
		assert_eq!(lines[3], "    4 GiB · input=text\u{241b}[2J");
		assert_eq!(lines[5], "  local/other");
		assert!(lines.iter().all(|line| !line.contains('\n')));
		assert!(lines.iter().all(|line| !line.contains('\u{1b}')));
	}

	#[test]
	fn empty_inventory_gives_one_truthful_next_action() {
		let lines =
			human_inventory_lines(&[], Palette::stdout(super::super::style::ColorMode::Never));
		assert_eq!(
			lines,
			[
				"No models installed.",
				"  Find one with: emelex hub search QUERY"
			]
		);
	}

	#[test]
	fn quarantine_count_has_normal_grammar() {
		assert_eq!(quarantined_snapshot_count(1), "1 quarantined snapshot");
		assert_eq!(quarantined_snapshot_count(3), "3 quarantined snapshots");
	}

	#[test]
	fn import_name_defaults_to_canonical_directory_name() {
		let directory = tempfile::tempdir().expect("temporary directory");
		let checkpoint = directory.path().join("mlx-checkpoint");
		std::fs::create_dir(&checkpoint).expect("checkpoint directory");

		let name = import_name(&checkpoint, None).expect("derived model name");
		assert_eq!(name.as_str(), "mlx-checkpoint");
	}

	#[test]
	fn explicit_import_name_does_not_require_an_existing_path() {
		let name = import_name(
			std::path::Path::new("/missing/checkpoint"),
			Some("work".to_string()),
		)
		.expect("explicit model name");
		assert_eq!(name.as_str(), "work");
	}
}
