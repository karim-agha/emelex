use super::*;

#[test]
fn unknown_sizing_never_matches_numeric_predicates() {
	let traits = ModelTraits::default();
	for filter in [
		"weights_bytes<=0",
		"residency_bytes<=0",
		"context_tokens>=1",
		"max_context_tokens>=1",
	] {
		let filter = TraitFilter::parse(filter).expect("valid numeric predicate");
		assert!(!traits.satisfies(&filter), "{filter}");
	}
}

#[test]
fn numeric_predicates_use_typed_sizing_facts() {
	let traits = ModelTraits {
		sizing: Some(ModelSizing {
			weights_bytes: Some(2_000),
			estimated_residency_bytes: Some(4_000),
			evaluated_context_tokens: Some(8_192),
			max_context_tokens: Some(32_768),
		}),
		..ModelTraits::default()
	};
	for filter in [
		"weights_bytes<=2000",
		"residency_bytes<=4000",
		"context_tokens>=8192",
		"max_context_tokens>=32768",
	] {
		let filter = TraitFilter::parse(filter).expect("valid numeric predicate");
		assert!(traits.satisfies(&filter), "{filter}");
	}
}

#[test]
fn confidence_predicate_uses_underlying_capability_key() {
	let mut traits = ModelTraits {
		mlx: true,
		..ModelTraits::default()
	};
	traits
		.confidence
		.insert("acceleration:mlx".to_string(), TraitConfidence::Inferred);
	let inferred = TraitFilter::parse("confidence:inferred:acceleration:mlx")
		.expect("valid confidence predicate");
	let verified = TraitFilter::parse("confidence:runtime_verified:acceleration:mlx")
		.expect("valid confidence predicate");
	assert!(traits.satisfies(&inferred));
	assert!(!traits.satisfies(&verified));
	assert_eq!(
		traits.confidence(&inferred),
		Some(TraitConfidence::Inferred)
	);
}

#[test]
fn mtp_stage_predicate_preserves_progression() {
	let traits = ModelTraits {
		mtp: MtpSupport::Advertised,
		..ModelTraits::default()
	};
	let advertised = TraitFilter::parse("mtp_stage>=advertised").expect("valid MTP predicate");
	let verified = TraitFilter::parse("mtp_stage>=runtime_verified").expect("valid MTP predicate");
	assert!(traits.satisfies(&advertised));
	assert!(!traits.satisfies(&verified));
}

#[test]
fn filters_reject_typos_and_unsafe_extensions() {
	assert!(TraitFilter::parse("acceleration:mx").is_err());
	assert!(TraitFilter::parse("extension:../escape").is_err());
	assert!(TraitFilter::parse("weights_bytes<=many").is_err());
}

#[test]
fn unsupported_video_filters_are_rejected() {
	for value in ["input:video", "output:video"] {
		assert!(TraitFilter::parse(value).is_err(), "{value}");
	}
}

#[test]
fn task_translation_filter_matches_translation_task() {
	let filter = TraitFilter::parse("task:translation").expect("known capability");
	let mut traits = ModelTraits::default();
	assert!(!traits.satisfies(&filter));
	traits.tasks.insert(Task::Translation);
	assert!(traits.satisfies(&filter));
}
