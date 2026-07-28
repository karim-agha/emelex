//! Stable-reference model selection and zero-model onboarding.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context as _, bail};
use emelex::{
	Emelex,
	config::Config,
	hub::HubSearch,
	model::{InstalledModel, ModelRef, ModelTraits, TraitFilter},
	models::{ModelManager, ModelsError},
};

use super::{
	hub_cmd::{download, trait_summary, wait_for_hub},
	output,
	style::{Palette, bytes},
};

/// Resolve one installed model without hidden strength tiers or aliases.
pub(crate) async fn resolve(
	emelex: &Emelex,
	explicit: Option<&ModelRef>,
	required: &[TraitFilter],
	interactive: bool,
	stdout_palette: Palette,
	stderr_palette: Palette,
) -> anyhow::Result<InstalledModel> {
	let models = model_manager(emelex)?;
	if let Some(reference) = explicit {
		let installed = models
			.resolve(reference)
			.with_context(|| format!("resolve installed model {reference}"))?;
		validate_traits(&installed, required)?;
		return Ok(installed);
	}
	if let Some(reference) = &emelex.config().default_model {
		match models.resolve(reference) {
			Ok(installed) => {
				if let Some(warning) = configured_default_trait_warning(
					reference,
					installed.manifest().traits(),
					required,
					interactive,
				)? {
					output::stderr_line(
						&stderr_palette.yellow(&output::terminal_safe_inline(&warning)),
					)?;
				} else {
					return Ok(installed);
				}
			}
			Err(error) if interactive => {
				let message = format!("configured default {reference} is unavailable: {error}");
				let message = output::terminal_safe_inline(&message);
				output::stderr_line(&stderr_palette.yellow(&message))?;
			}
			Err(error) => {
				return Err(error)
					.with_context(|| format!("resolve configured default model {reference}"));
			}
		}
	}

	let compatible = newest_compatible(emelex, required)?;
	match compatible.as_slice() {
		[only] => Ok(only.clone()),
		[] if interactive => onboard(emelex, required, stdout_palette, stderr_palette).await,
		[] => bail!(
			"no compatible installed model; run `emelex hub search --require \
			 acceleration:mlx`, then `emelex hub download [NAMESPACE/]REPO`, or pass \
			 `--model`"
		),
		_many if !interactive => bail!(
			"multiple compatible models are installed; pass `--model [NAMESPACE/]REPO` \
			 or set one with `emelex models default [NAMESPACE/]REPO`"
		),
		many => {
			let labels = many
				.iter()
				.map(|model| {
					let weights = model
						.manifest()
						.traits()
						.sizing
						.as_ref()
						.and_then(|sizing| sizing.weights_bytes)
						.map_or_else(|| "unknown".to_string(), bytes);
					format!(
						"{} ({}, {})",
						model.reference(),
						weights,
						trait_summary(model.manifest().traits())
					)
				})
				.collect::<Vec<_>>();
			let selected = dialoguer::Select::new()
				.with_prompt("Choose an installed model")
				.items(&labels)
				.default(0)
				.interact_opt()
				.context("choose installed model")?
				.context("model selection cancelled")?;
			Ok(many[selected].clone())
		}
	}
}

fn newest_compatible(
	emelex: &Emelex,
	required: &[TraitFilter],
) -> anyhow::Result<Vec<InstalledModel>> {
	let mut newest = BTreeMap::<ModelRef, InstalledModel>::new();
	for installed in model_manager(emelex)?
		.list()
		.context("list installed models")?
	{
		if !required
			.iter()
			.all(|filter| installed.manifest().traits().satisfies(filter))
		{
			continue;
		}
		match newest.get(installed.reference()) {
			Some(current)
				if current.manifest().installed_at() >= installed.manifest().installed_at() => {}
			_ => {
				newest.insert(installed.reference().clone(), installed);
			}
		}
	}
	Ok(newest.into_values().collect())
}

fn validate_traits(installed: &InstalledModel, required: &[TraitFilter]) -> anyhow::Result<()> {
	let missing = missing_traits(installed.manifest().traits(), required);
	if missing.is_empty() {
		Ok(())
	} else {
		bail!(
			"model {} lacks required trait(s): {}",
			installed.reference(),
			missing.join(", ")
		)
	}
}

fn configured_default_trait_warning(
	reference: &ModelRef,
	traits: &ModelTraits,
	required: &[TraitFilter],
	interactive: bool,
) -> anyhow::Result<Option<String>> {
	let missing = missing_traits(traits, required);
	if missing.is_empty() {
		return Ok(None);
	}
	let message = format!(
		"configured default {reference} lacks required trait(s): {}",
		missing.join(", ")
	);
	if interactive {
		Ok(Some(format!("{message}; choosing another model")))
	} else {
		bail!("{message}")
	}
}

fn missing_traits(traits: &ModelTraits, required: &[TraitFilter]) -> Vec<String> {
	required
		.iter()
		.filter(|filter| !traits.satisfies(filter))
		.map(ToString::to_string)
		.collect()
}

#[allow(
	clippy::too_many_lines,
	reason = "one paginated interactive state machine keeps every selection and retry transition explicit"
)]
async fn onboard(
	emelex: &Emelex,
	required: &[TraitFilter],
	stdout_palette: Palette,
	stderr_palette: Palette,
) -> anyhow::Result<InstalledModel> {
	output::stderr_line(
		&stderr_palette
			.dim("no compatible installed model; exploring visible Hugging Face MLX checkpoints"),
	)?;
	let query = dialoguer::Input::<String>::new()
		.with_prompt("Optional Hugging Face search text")
		.allow_empty(true)
		.interact_text()
		.context("read Hub search text")?;
	let filters = onboarding_filters(required)?;
	explain_media_discovery(required, stderr_palette)?;
	let mut search = HubSearch::default().requirements(filters);
	if !query.trim().is_empty() {
		search = search.query(query);
	}
	let manager = model_manager(emelex)?;
	let mut cursor: Option<String> = None;
	let mut seen_cursors = BTreeSet::new();
	let mut candidate_ids = BTreeSet::new();
	let mut candidates = Vec::new();
	let mut certification_failed = false;
	let mut saw_fitting_candidate = false;
	let (installed, reference) = 'pages: loop {
		let page_search = cursor.as_ref().map_or_else(
			|| search.clone(),
			|cursor| search.clone().cursor(cursor.clone()),
		);
		let page = wait_for_hub("searching Hugging Face", false, async {
			manager
				.hub()
				.search(&page_search)
				.await
				.context("search compatible Hugging Face models")
		})
		.await?;
		let next_cursor = page.next_cursor;
		candidate_ids.clear();
		for model in page
			.items
			.into_iter()
			.filter(|model| model.fit.as_ref().is_some_and(|fit| fit.fits))
		{
			if candidate_ids.insert(model.id.to_string()) {
				candidates.push(model);
			}
		}
		saw_fitting_candidate |= !candidates.is_empty();

		loop {
			if candidates.is_empty() {
				let Some(next_cursor) = next_cursor.clone() else {
					if certification_failed {
						bail!("no explored Hub candidate could be certified for this invocation");
					}
					if saw_fitting_candidate {
						bail!("Hugging Face catalog exhausted without a model selection");
					}
					bail!("{}", empty_onboarding_message(required));
				};
				let prompt = if certification_failed {
					"No candidate remaining from explored pages could be certified. Search the \
					 next Hub page?"
				} else {
					"No fitting model on this Hub page. Search the next page?"
				};
				if !dialoguer::Confirm::new()
					.with_prompt(prompt)
					.default(true)
					.interact()
					.context("confirm next Hub page")?
				{
					bail!("model search stopped after this page; more Hugging Face pages remain");
				}
				cursor = Some(advance_onboarding_page(
					next_cursor,
					&mut seen_cursors,
					&mut candidates,
				)?);
				continue 'pages;
			}

			let mut labels = candidates
				.iter()
				.map(|model| {
					let weights = model
						.traits
						.sizing
						.as_ref()
						.and_then(|sizing| sizing.weights_bytes)
						.map_or_else(|| "unknown".to_string(), bytes);
					let residency = model
						.fit
						.as_ref()
						.map_or_else(|| "unknown".to_string(), |fit| bytes(fit.required_bytes));
					format!(
						"{} (weights {}, residency {}, {})",
						model.id,
						weights,
						residency,
						trait_summary(&model.traits)
					)
				})
				.collect::<Vec<_>>();
			if next_cursor.is_some() {
				labels.push("Search the next Hugging Face page".to_string());
			}
			let selected = dialoguer::Select::new()
				.with_prompt("Download a compatible model")
				.items(&labels)
				.default(0)
				.interact_opt()
				.context("choose Hub model")?
				.context("model download cancelled")?;
			match onboarding_selection(selected, candidates.len(), next_cursor.is_some())? {
				OnboardingSelection::NextPage => {
					cursor = Some(advance_onboarding_page(
						next_cursor
							.clone()
							.context("Hub page selection lost its next cursor")?,
						&mut seen_cursors,
						&mut candidates,
					)?);
					continue 'pages;
				}
				OnboardingSelection::Candidate(selected) => {
					let id = candidates[selected].id.clone();
					let reference = ModelRef::Hub(id.clone());
					let installed =
						match download(emelex, &id, false, stdout_palette, stderr_palette).await {
							Ok(installed) => installed,
							Err(error) if candidate_certification_failed(&error) => {
								certification_failed = true;
								report_candidate_certification_failure(
									&reference,
									&error,
									stderr_palette,
								)?;
								candidates.remove(selected);
								continue;
							}
							Err(error) => return Err(error),
						};
					match validate_traits(&installed, required) {
						Ok(()) => break 'pages (installed, reference),
						Err(error) => {
							certification_failed = true;
							report_candidate_certification_failure(
								&reference,
								&error,
								stderr_palette,
							)?;
							candidates.remove(selected);
						}
					}
				}
			}
		}
	};
	if dialoguer::Confirm::new()
		.with_prompt("Use this as the global default model?")
		.default(true)
		.interact()
		.context("confirm default model")?
	{
		Config::write_global_default_model(emelex.home(), Some(&reference))
			.context("save global default model")?;
	}
	Ok(installed)
}

fn candidate_certification_failed(error: &anyhow::Error) -> bool {
	error
		.downcast_ref::<ModelsError>()
		.is_some_and(|error| matches!(error, ModelsError::Certification(_)))
}

fn report_candidate_certification_failure(
	reference: &ModelRef,
	error: &dyn std::fmt::Display,
	stderr_palette: Palette,
) -> anyhow::Result<()> {
	let message = candidate_certification_failure_message(reference, error);
	output::stderr_line(&stderr_palette.yellow(&output::terminal_safe_inline(&message)))
}

fn candidate_certification_failure_message(
	reference: &ModelRef,
	error: &dyn std::fmt::Display,
) -> String {
	format!(
		"{reference} could not be certified for this invocation: {error:#}; choose another model or \
		 search the next Hub page"
	)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OnboardingSelection {
	Candidate(usize),
	NextPage,
}

fn onboarding_selection(
	selected: usize,
	candidate_count: usize,
	has_next_page: bool,
) -> anyhow::Result<OnboardingSelection> {
	if selected < candidate_count {
		return Ok(OnboardingSelection::Candidate(selected));
	}
	if has_next_page && selected == candidate_count {
		return Ok(OnboardingSelection::NextPage);
	}
	bail!("model selector returned an invalid index")
}

fn advance_onboarding_page<T>(
	cursor: String,
	seen: &mut BTreeSet<String>,
	candidates: &mut Vec<T>,
) -> anyhow::Result<String> {
	if seen.insert(cursor.clone()) {
		candidates.clear();
		Ok(cursor)
	} else {
		bail!("Hub search returned a repeated next-page cursor")
	}
}

fn explain_media_discovery(
	required: &[TraitFilter],
	stderr_palette: Palette,
) -> anyhow::Result<()> {
	if required
		.iter()
		.any(|filter| matches!(filter.as_str(), "input:image" | "input:audio"))
	{
		output::stderr_line(&stderr_palette.dim(
			"media candidates use Hub-advertised input metadata; every download still requires \
			 local runtime certification",
		))?;
	}
	Ok(())
}

fn onboarding_filters(required: &[TraitFilter]) -> anyhow::Result<Vec<TraitFilter>> {
	let capability_labels = [
		("Tool use", "interaction:tools"),
		("Reasoning", "interaction:reasoning"),
		(
			"MTP advertisement (local certification follows)",
			"acceleration:mtp_advertised",
		),
	];
	let capability_names = capability_labels.map(|(label, _)| label);
	let selected = dialoguer::MultiSelect::new()
		.with_prompt("Additional required capabilities")
		.items(capability_names)
		.interact()
		.context("choose model capabilities")?;
	let mut filters = BTreeSet::new();
	for required in required {
		filters.insert(remote_onboarding_requirement(required)?);
	}
	filters.insert(TraitFilter::parse("acceleration:mlx").context("built-in MLX trait")?);
	for index in selected {
		let Some((_, filter)) = capability_labels.get(index) else {
			bail!("capability selector returned an invalid index");
		};
		filters.insert(TraitFilter::parse(*filter).context("built-in capability trait")?);
	}
	Ok(filters.into_iter().collect())
}

fn remote_onboarding_requirement(required: &TraitFilter) -> anyhow::Result<TraitFilter> {
	let remote = match required.as_str() {
		"acceleration:mtp" => "acceleration:mtp_advertised",
		"input:image" => "extension:huggingface.advertised_input_image",
		"input:audio" => "extension:huggingface.advertised_input_audio",
		_ => return Ok(required.clone()),
	};
	TraitFilter::parse(remote).with_context(|| format!("built-in remote trait {remote}"))
}

fn empty_onboarding_message(required: &[TraitFilter]) -> String {
	let advertised = required
		.iter()
		.filter_map(|filter| match filter.as_str() {
			"input:image" => Some("image"),
			"input:audio" => Some("audio"),
			_ => None,
		})
		.collect::<Vec<_>>();
	if advertised.is_empty() {
		return "Hub search found no compatible model fitting this machine; try `emelex hub search \
		        QUERY --require acceleration:mlx`"
			.to_string();
	}
	format!(
		"Hub search found no fitting model whose metadata advertises {} input; advertised metadata \
		 is only a discovery hint and every downloaded model still requires local runtime \
		 certification",
		advertised.join(" and ")
	)
}

fn model_manager(emelex: &Emelex) -> anyhow::Result<&ModelManager> {
	emelex.models().context("initialize model manager")
}

/// Effective capability requirements for one concrete invocation.
#[derive(Debug, Clone, Copy)]
#[expect(
	clippy::struct_excessive_bools,
	reason = "orthogonal capability requirements deliberately compose independently"
)]
pub(crate) struct InvocationRequirements {
	pub chat: bool,
	pub system_prompt: bool,
	pub agent: bool,
	pub image: bool,
	pub audio: bool,
	pub reasoning_history: bool,
	pub thinking_toggle: bool,
	pub mtp: bool,
}

/// Build evidence-backed filters for one concrete invocation.
pub(crate) fn filters(requirements: InvocationRequirements) -> anyhow::Result<Vec<TraitFilter>> {
	let mut names = vec![if requirements.chat {
		"task:chat"
	} else {
		"task:text_generation"
	}];
	if requirements.system_prompt {
		names.push("interaction:system_prompt");
	}
	if requirements.agent {
		names.push("interaction:tools");
	}
	if requirements.image {
		names.push("input:image");
	}
	if requirements.audio {
		names.push("input:audio");
	}
	if requirements.reasoning_history {
		names.push("interaction:reasoning_history");
	}
	if requirements.thinking_toggle {
		names.push("interaction:thinking_toggle");
	}
	if requirements.mtp {
		names.push("acceleration:mtp");
	}
	names
		.into_iter()
		.map(|name| TraitFilter::parse(name).with_context(|| format!("built-in trait {name}")))
		.collect()
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn interactive_default_trait_mismatch_continues_but_noninteractive_fails() {
		let reference = ModelRef::parse("owner/model").expect("model reference");
		let required = [TraitFilter::parse("input:image").expect("trait filter")];
		let warning =
			configured_default_trait_warning(&reference, &ModelTraits::default(), &required, true)
				.expect("interactive policy")
				.expect("warning");
		assert!(warning.contains("choosing another model"));
		assert!(
			configured_default_trait_warning(&reference, &ModelTraits::default(), &required, false)
				.is_err()
		);
	}

	#[test]
	fn onboarding_media_requirements_use_advertised_remote_evidence() {
		for (required, remote) in [
			(
				"input:image",
				"extension:huggingface.advertised_input_image",
			),
			(
				"input:audio",
				"extension:huggingface.advertised_input_audio",
			),
			("acceleration:mtp", "acceleration:mtp_advertised"),
		] {
			let required = TraitFilter::parse(required).expect("required trait");
			let mapped =
				remote_onboarding_requirement(&required).expect("remote onboarding mapping");
			assert_eq!(mapped.as_str(), remote);
		}
	}

	#[test]
	fn empty_media_onboarding_result_explains_advertised_evidence_limit() {
		let required = [
			TraitFilter::parse("input:image").expect("image trait"),
			TraitFilter::parse("input:audio").expect("audio trait"),
		];
		let message = empty_onboarding_message(&required);
		assert!(message.contains("metadata advertises image and audio input"));
		assert!(message.contains("local runtime certification"));
	}

	#[test]
	fn onboarding_page_transition_is_explicit_and_rejects_cursor_cycles() {
		assert_eq!(
			onboarding_selection(2, 2, true).expect("next-page selection"),
			OnboardingSelection::NextPage
		);
		assert!(onboarding_selection(2, 2, false).is_err());

		let mut seen = BTreeSet::new();
		let mut candidates = vec![1, 2];
		assert_eq!(
			advance_onboarding_page("next".to_string(), &mut seen, &mut candidates)
				.expect("first cursor"),
			"next"
		);
		assert!(candidates.is_empty());
		candidates.push(3);
		assert!(advance_onboarding_page("next".to_string(), &mut seen, &mut candidates).is_err());
		assert_eq!(candidates, [3]);
	}

	#[test]
	fn onboarding_retries_only_typed_candidate_certification_failures() {
		let retryable = anyhow::Error::new(ModelsError::Certification(Box::new(
			ModelsError::Incompatible(vec!["unsupported checkpoint".to_string()]),
		)))
		.context("download owner/model");
		assert!(candidate_certification_failed(&retryable));

		let fatal = anyhow::Error::new(ModelsError::Hub(emelex::hub::HubError::Cancelled))
			.context("download owner/model");
		assert!(!candidate_certification_failed(&fatal));

		let reference = ModelRef::parse("owner/model").expect("model reference");
		let message = candidate_certification_failure_message(&reference, &retryable);
		assert!(message.contains("download owner/model"));
		assert!(message.contains("unsupported checkpoint"));

		let private = anyhow::Error::new(ModelsError::Certification(Box::new(ModelsError::Hub(
			emelex::hub::HubError::NotPublic("owner/model".to_string()),
		))));
		let message = candidate_certification_failure_message(&reference, &private);
		assert!(message.contains("could not be certified for this invocation"));
		assert!(!message.contains("local runtime"));
	}
}
