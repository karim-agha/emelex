//! Local platform and installation diagnostics.

use anyhow::{Context as _, bail};
use emelex::{Emelex, model::InstalledModel};
use serde::Serialize;

use super::{args::DoctorArgs, output, style::Palette};

const MAX_DIAGNOSTIC_CHARS: usize = 4_096;

#[derive(Serialize)]
struct DoctorReport {
	ok: bool,
	home: String,
	checks: Vec<DoctorCheck>,
}

#[derive(Serialize)]
struct DoctorCheck {
	name: String,
	ok: bool,
	detail: String,
	#[serde(skip_serializing_if = "Option::is_none")]
	data: Option<serde_json::Value>,
}

/// Validate each independent local facet and render the complete report.
pub(crate) fn run(
	emelex: &Emelex,
	args: DoctorArgs,
	json: bool,
	palette: Palette,
) -> anyhow::Result<()> {
	let mut checks = base_checks(emelex);
	let (inventory_checks, installed) = inventory_checks(emelex);
	checks.extend(inventory_checks);
	if args.models
		&& let Ok(models) = emelex.models()
	{
		for model in &installed {
			let name = format!("model:{}", model.snapshot_id());
			checks.push(check(&name, || {
				let verification = models
					.verify(model)
					.with_context(|| format!("verify {}", model.snapshot_id()))?;
				Ok(serde_json::to_value(verification.compatibility)?)
			}));
		}
	}

	let report = DoctorReport {
		ok: checks.iter().all(|check| check.ok),
		home: emelex.home().root().display().to_string(),
		checks,
	};
	render_report(&report, json, palette)?;
	if !report.ok {
		bail!("one or more doctor checks failed");
	}
	Ok(())
}

fn base_checks(emelex: &Emelex) -> Vec<DoctorCheck> {
	vec![
		check("home", || {
			Ok(serde_json::json!({
				"path": emelex.home().root(),
				"config_sources": {
					"global": emelex.config_sources().global.as_ref(),
					"project": emelex.config_sources().project.as_ref(),
				},
			}))
		}),
		check("metal_budget", || {
			Ok(serde_json::json!({
				"bytes": emelex
					.metal_budget_bytes()
					.context("query recommended Metal working set")?,
			}))
		}),
		check("memory", || {
			Ok(serde_json::to_value(
				emelex
					.memory()
					.context("initialize durable memory")?
					.status()
					.context("inspect durable memory")?,
			)?)
		}),
		check("runtime_asset", || {
			let asset = emelex::runtime::initialize(emelex.home().root())
				.context("initialize embedded MLX runtime")?;
			Ok(serde_json::json!({
				"metallib": asset.metallib(),
				"digest": asset.digest(),
			}))
		}),
		check("mlx_engine", || {
			emelex::runtime::verify_engine().context("evaluate embedded MLX runtime")?;
			Ok(serde_json::json!({"evaluated": true}))
		}),
	]
}

fn inventory_checks(emelex: &Emelex) -> (Vec<DoctorCheck>, Vec<InstalledModel>) {
	let mut checks = Vec::new();
	let mut installed = Vec::new();
	match emelex.models().context("initialize model manager") {
		Ok(models) => match models
			.inventory()
			.context("inspect installed-model inventory")
		{
			Ok(inventory) => {
				for diagnostic in inventory.diagnostics {
					checks.push(DoctorCheck {
						name: "model_inventory_entry".to_string(),
						ok: false,
						detail: bounded(&diagnostic.message),
						data: Some(serde_json::json!({"path": diagnostic.path})),
					});
				}
				checks.push(DoctorCheck {
					name: "model_inventory".to_string(),
					ok: true,
					detail: format!("{} healthy snapshot(s)", inventory.models.len()),
					data: Some(serde_json::json!({"healthy": inventory.models.len()})),
				});
				installed = inventory.models;
			}
			Err(error) => checks.push(failed("model_inventory", &error)),
		},
		Err(error) => checks.push(failed("model_manager", &error)),
	}
	(checks, installed)
}

fn render_report(report: &DoctorReport, json: bool, palette: Palette) -> anyhow::Result<()> {
	if json {
		output::json_line(report)?;
	} else {
		output::stdout_line(&format!(
			"Emelex Home  {}",
			output::terminal_safe_inline(&report.home)
		))?;
		for check in &report.checks {
			let status = if check.ok {
				palette.green("ok")
			} else {
				palette.red("fail")
			};
			output::stdout_line(&format!(
				"{status}  {}  {}",
				output::terminal_safe_inline(&check.name),
				output::terminal_safe_inline(&check.detail)
			))?;
		}
	}
	Ok(())
}

fn check(name: &str, operation: impl FnOnce() -> anyhow::Result<serde_json::Value>) -> DoctorCheck {
	match operation() {
		Ok(data) => DoctorCheck {
			name: name.to_string(),
			ok: true,
			detail: "ready".to_string(),
			data: Some(data),
		},
		Err(error) => failed(name, &error),
	}
}

fn failed(name: &str, error: &anyhow::Error) -> DoctorCheck {
	DoctorCheck {
		name: name.to_string(),
		ok: false,
		detail: bounded(&format!("{error:#}")),
		data: None,
	}
}

fn bounded(value: &str) -> String {
	let mut text = value
		.chars()
		.take(MAX_DIAGNOSTIC_CHARS.saturating_add(1))
		.collect::<String>();
	if text.chars().count() > MAX_DIAGNOSTIC_CHARS {
		text = text.chars().take(MAX_DIAGNOSTIC_CHARS).collect();
		text.push('…');
	}
	text
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn diagnostic_text_is_bounded_by_characters() {
		let text = "🧠".repeat(MAX_DIAGNOSTIC_CHARS + 20);
		let bounded = bounded(&text);
		assert_eq!(bounded.chars().count(), MAX_DIAGNOSTIC_CHARS + 1);
		assert!(bounded.ends_with('…'));
	}
}
