//! Managed local model lifecycle presentation.

use std::{collections::BTreeSet, time::Duration};

use anyhow::{Context as _, bail};
use emelex::{
	Emelex,
	config::Config,
	model::{LocalModelName, ModelRef},
	models::ModelManager,
};

use super::{
	args::ModelsCommand,
	hub_cmd::{download, installed_json, trait_summary},
	output,
	style::{Palette, bytes},
};

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
			if json {
				output::json_line(&installed_json(&installed))
			} else {
				output::stdout_line(&format!(
					"{} {}",
					stdout_palette.green("installed"),
					output::terminal_safe_inline(&installed.path().display().to_string())
				))
			}
		}
		ModelsCommand::Default { model, clear } => default_model(emelex, model, clear, json),
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
				output::stdout_line(&output::terminal_safe_inline(
					&quarantine.display().to_string(),
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
			} else {
				output::stdout_line(&format!("removed {removed} quarantined snapshot(s)"))
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

fn default_model(
	emelex: &Emelex,
	model: Option<ModelRef>,
	clear: bool,
	json: bool,
) -> anyhow::Result<()> {
	if clear {
		Config::write_global_default_model(emelex.home(), None)
			.context("clear global default model")?;
		if json {
			output::json_line(&serde_json::json!({"default_model": null}))
		} else {
			output::stdout_line("cleared")
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
			output::stdout_line(&model.to_string())
		}
	} else if json {
		output::json_line(&serde_json::json!({
			"default_model": emelex.config().default_model.as_ref()
		}))
	} else {
		output::stdout_line(
			&emelex
				.config()
				.default_model
				.as_ref()
				.map_or_else(|| "not set".to_string(), ToString::to_string),
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
	for model in inventory.models {
		let default = if emelex.config().default_model.as_ref() == Some(model.reference()) {
			"*"
		} else {
			" "
		};
		let weights = model
			.manifest()
			.traits()
			.sizing
			.as_ref()
			.and_then(|sizing| sizing.weights_bytes)
			.map_or_else(|| "unknown".to_string(), bytes);
		output::stdout_line(&format!(
			"{default} {}  {}  {}",
			stdout_palette.cyan(&model.snapshot_id().to_string()),
			weights,
			trait_summary(model.manifest().traits())
		))?;
	}
	for diagnostic in inventory.diagnostics {
		output::stderr_line(
			&stderr_palette.yellow(&invalid_model_line(&diagnostic.path, &diagnostic.message)),
		)?;
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
				stdout_palette.green("verified"),
				stdout_palette.cyan(&model.snapshot_id().to_string())
			))?;
		}
	}
	Ok(())
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
}
