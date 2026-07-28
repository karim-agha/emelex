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
		for line in human_report_lines(report, palette) {
			output::stdout_line(&line)?;
		}
	}
	Ok(())
}

fn human_report_lines(report: &DoctorReport, palette: Palette) -> Vec<String> {
	let labels = report
		.checks
		.iter()
		.map(|check| doctor_check_label(&check.name))
		.collect::<Vec<_>>();
	let label_width = labels
		.iter()
		.map(|label| dialoguer::console::measure_text_width(label))
		.max()
		.unwrap_or(0);
	let mut lines = vec![
		palette.bold("Emelex doctor"),
		format!(
			"  {}  {}",
			palette.dim("Home"),
			output::terminal_safe_inline(&report.home)
		),
		String::new(),
	];

	for (check, label) in report.checks.iter().zip(labels) {
		let status = if check.ok {
			palette.green("✓")
		} else {
			palette.red("×")
		};
		let padding =
			" ".repeat(label_width.saturating_sub(dialoguer::console::measure_text_width(&label)));
		lines.push(format!(
			"  {status} {label}{padding}  {}",
			output::terminal_safe_inline(&check.detail)
		));
	}

	lines.push(String::new());
	let failed = report.checks.iter().filter(|check| !check.ok).count();
	if failed == 0 {
		lines.push(palette.green(&format!("✓ Ready · {}", check_count(report.checks.len()))));
	} else {
		lines.push(palette.red(&format!(
			"× {failed} of {} checks failed",
			report.checks.len()
		)));
	}
	lines
}

fn doctor_check_label(name: &str) -> String {
	let name = output::terminal_safe_inline(name);
	if let Some(snapshot) = name.strip_prefix("model:") {
		return format!("model {snapshot}");
	}
	match name.as_ref() {
		"metal_budget" => "Metal budget".to_string(),
		"runtime_asset" => "runtime asset".to_string(),
		"mlx_engine" => "MLX engine".to_string(),
		"model_inventory_entry" => "model inventory entry".to_string(),
		"model_inventory" => "model inventory".to_string(),
		_ => name.replace('_', " "),
	}
}

fn check_count(count: usize) -> String {
	if count == 1 {
		"1 check passed".to_string()
	} else {
		format!("{count} checks passed")
	}
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

	#[test]
	fn human_report_is_aligned_sanitized_and_summarized() {
		let report = DoctorReport {
			ok: false,
			home: "/tmp/emelex\nforged".to_string(),
			checks: vec![
				DoctorCheck {
					name: "mlx_engine".to_string(),
					ok: true,
					detail: "ready".to_string(),
					data: None,
				},
				DoctorCheck {
					name: "model:abc\u{202e}".to_string(),
					ok: false,
					detail: "bad\nentry\u{1b}[2J".to_string(),
					data: None,
				},
			],
		};
		let lines = human_report_lines(
			&report,
			Palette::stdout(super::super::style::ColorMode::Never),
		);
		assert_eq!(lines[0], "Emelex doctor");
		assert_eq!(lines[1], "  Home  /tmp/emelex\u{240a}forged");
		assert_eq!(lines[3], "  ✓ MLX engine  ready");
		assert_eq!(
			lines[4],
			"  × model abc\u{fffd}  bad\u{240a}entry\u{241b}[2J"
		);
		assert_eq!(lines[6], "× 1 of 2 checks failed");
		assert!(lines.iter().all(|line| !line.contains('\u{1b}')));
	}

	#[test]
	fn successful_human_report_ends_with_ready_summary() {
		let report = DoctorReport {
			ok: true,
			home: "/tmp/emelex".to_string(),
			checks: vec![DoctorCheck {
				name: "home".to_string(),
				ok: true,
				detail: "ready".to_string(),
				data: None,
			}],
		};
		let lines = human_report_lines(
			&report,
			Palette::stdout(super::super::style::ColorMode::Never),
		);
		assert_eq!(
			lines.last().map(String::as_str),
			Some("✓ Ready · 1 check passed")
		);
	}
}
