use super::*;

#[test]
fn hub_reference_round_trips() {
	let reference = ModelRef::parse("mlx-community/Qwen3.5-4B-4bit").expect("valid");
	assert_eq!(reference.to_string(), "mlx-community/Qwen3.5-4B-4bit");

	let unnamespaced = ModelRef::parse("gpt2").expect("valid unnamespaced Hub ID");
	assert_eq!(unnamespaced.to_string(), "gpt2");
}

#[test]
fn local_reference_round_trips() {
	let reference = ModelRef::parse("local:experiment-1").expect("valid");
	assert_eq!(reference.to_string(), "local:experiment-1");
}

#[test]
fn path_like_local_name_is_rejected() {
	let error = ModelRef::parse("local:../outside").expect_err("unsafe");
	assert!(matches!(error, ModelRefError::InvalidLocal(_)));
}

#[test]
fn partial_revision_is_rejected() {
	let error = ResolvedRevision::parse("deadbeef").expect_err("partial");
	assert!(matches!(error, ModelRefError::InvalidRevision(_)));
}

#[test]
fn overlong_identity_components_are_rejected() {
	let id = format!("{}/repo", "a".repeat(97));
	assert!(matches!(
		HubModelId::parse(id),
		Err(ModelRefError::InvalidHub(_))
	));
	assert!(matches!(
		LocalModelName::parse("a".repeat(129)),
		Err(ModelRefError::InvalidLocal(_))
	));
}

#[test]
fn hub_reference_matches_hugging_face_repository_rules() {
	for valid in [
		"foo",
		"123",
		"_owner/_repo_",
		"mlx-community/Qwen3.5-4B-4bit",
		"owner.with-dots/repo_name",
		"owner/repo.GIT",
	] {
		assert!(HubModelId::parse(valid).is_ok(), "{valid}");
	}
	for invalid in [
		"",
		".repo_id",
		"foo.git",
		".owner/repo",
		"owner-/repo",
		"owner/repo.",
		"owner/foo--bar",
		"owner/foo..bar",
		"owner/repo.git",
		"datasets/foo/bar",
	] {
		assert!(
			matches!(
				HubModelId::parse(invalid),
				Err(ModelRefError::InvalidHub(_))
			),
			"{invalid}"
		);
	}
	let overlong_repo = format!("owner/{}", "r".repeat(97));
	assert!(matches!(
		HubModelId::parse(overlong_repo),
		Err(ModelRefError::InvalidHub(_))
	));
	let overlong_total = format!("{}/{}", "o".repeat(48), "r".repeat(48));
	assert!(matches!(
		HubModelId::parse(overlong_total),
		Err(ModelRefError::InvalidHub(_))
	));

	let namespaced = HubModelId::parse("owner/model").expect("namespaced ID");
	assert_eq!(namespaced.namespace(), Some("owner"));
	assert_eq!(namespaced.repo_name(), "model");
	let unnamespaced = HubModelId::parse("gpt2").expect("unnamespaced ID");
	assert_eq!(unnamespaced.namespace(), None);
	assert_eq!(unnamespaced.repo_name(), "gpt2");
}

#[test]
fn exact_hub_snapshot_round_trips() {
	let value = format!("owner/repo@{}", "a".repeat(40));
	let snapshot = ModelSnapshotId::parse(&value).expect("valid exact Hub snapshot");
	assert_eq!(snapshot.to_string(), value);

	let unnamespaced = format!("gpt2@{}", "b".repeat(40));
	let snapshot = ModelSnapshotId::parse(&unnamespaced).expect("valid unnamespaced Hub snapshot");
	assert_eq!(snapshot.to_string(), unnamespaced);
}

#[test]
fn exact_local_snapshot_round_trips() {
	let value = format!("local:work@{}", "b".repeat(64));
	let snapshot = ModelSnapshotId::parse(&value).expect("valid exact local snapshot");
	assert_eq!(snapshot.to_string(), value);
}
