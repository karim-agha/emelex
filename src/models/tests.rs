use std::{sync::mpsc, thread, time::Duration};

use sha2::Digest as _;

use super::*;
use crate::{
	config::ThinkingMode,
	memory::MemoryStore,
	model::{
		EvidenceSource, ModelSizing, ModelTraits, MtpSupport, ResolvedRevision, Task,
		TraitConfidence, TraitEvidence,
	},
};

fn manager(config: Config) -> (tempfile::TempDir, ModelManager) {
	let directory = tempfile::tempdir().expect("temporary directory");
	let home = EmelexHome::prepare(&directory.path().join("home")).expect("test Emelex home");
	let workload =
		WorkloadProfile::new(1, config.inference.context_tokens).expect("valid workload");
	let budget = 8_u64 << 30;
	let hub = HubClient::with_fit_profile(config.hub.clone(), workload, budget)
		.expect("profiled Hub client");
	let manager = ModelManager::new(home, config, hub, budget).expect("model manager");
	(directory, manager)
}

fn runtime_files(root: &Path) -> Vec<ModelFile> {
	let contents = [
		("config.json", br"{}".as_slice()),
		("model.safetensors", b"weights".as_slice()),
		("tokenizer.json", br"{}".as_slice()),
	];
	contents
		.into_iter()
		.map(|(name, bytes)| {
			fs::write(root.join(name), bytes).expect("write runtime fixture");
			ModelFile::new(
				name,
				u64::try_from(bytes.len()).expect("fixture length fits u64"),
				hex::encode(sha2::Sha256::digest(bytes)),
			)
			.expect("valid file record")
		})
		.collect()
}

fn write_valid_safetensors(path: &Path) {
	let mut header = br#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#.to_vec();
	while !header.len().is_multiple_of(8) {
		header.push(b' ');
	}
	let mut bytes = u64::try_from(header.len())
		.expect("header length fits u64")
		.to_le_bytes()
		.to_vec();
	bytes.extend_from_slice(&header);
	bytes.extend_from_slice(&[0_u8; 4]);
	fs::write(path, bytes).expect("write safetensors fixture");
}

fn runtime_source(root: &Path) {
	fs::write(root.join("config.json"), br#"{"model_type":"llama"}"#).expect("write model config");
	fs::write(root.join("tokenizer.json"), br#"{"model":{"type":"BPE"}}"#)
		.expect("write tokenizer");
	write_valid_safetensors(&root.join("model.safetensors"));
}

fn verified_local_traits(files: &[ModelFile]) -> ModelTraits {
	let weights_bytes = files
		.iter()
		.filter(|file| file.path().ends_with(".safetensors"))
		.map(ModelFile::size)
		.sum();
	let mut traits = ModelTraits {
		mlx: true,
		tasks: BTreeSet::from([Task::TextGeneration]),
		sizing: Some(ModelSizing {
			weights_bytes: Some(weights_bytes),
			estimated_residency_bytes: Some(weights_bytes + 1),
			evaluated_context_tokens: Some(16),
			max_context_tokens: Some(32),
		}),
		..ModelTraits::default()
	};
	traits.evidence.push(TraitEvidence {
		trait_key: "compatibility:runtime_load".to_string(),
		source: EvidenceSource::Runtime,
		detail: "test-only runtime evidence".to_string(),
	});
	for key in ["acceleration:mlx", "task:text_generation"] {
		traits
			.confidence
			.insert(key.to_string(), TraitConfidence::RuntimeVerified);
	}
	traits
}

fn manifest(files: Vec<ModelFile>) -> ModelManifest {
	let weight_bytes = files
		.iter()
		.filter(|file| file.path().ends_with(".safetensors"))
		.map(ModelFile::size)
		.sum();
	let traits = ModelTraits {
		mlx: true,
		tasks: BTreeSet::from([Task::TextGeneration]),
		sizing: Some(ModelSizing {
			weights_bytes: Some(weight_bytes),
			estimated_residency_bytes: Some(weight_bytes + 1),
			evaluated_context_tokens: Some(16),
			max_context_tokens: Some(32),
		}),
		..ModelTraits::default()
	};
	ModelManifest::new(
		ModelRef::Hub(HubModelId::parse("owner/model").expect("valid Hub ID")),
		ModelSource::Hub,
		Some(ResolvedRevision::parse("a".repeat(40)).expect("valid revision")),
		files,
		traits,
		VerificationStatus::Estimated,
		None,
	)
	.expect("valid manifest")
}

#[test]
fn hub_destinations_encode_repository_arity_without_collisions() {
	let (_directory, manager) = manager(Config::default());
	let revision = "a".repeat(40);
	let unnamespaced = manager.hub_destination(
		&HubModelId::parse("gpt2").expect("unnamespaced ID"),
		&revision,
	);
	let namespaced = manager.hub_destination(
		&HubModelId::parse("owner/model").expect("namespaced ID"),
		&revision,
	);

	assert_eq!(
		unnamespaced
			.strip_prefix(manager.home.models_dir())
			.expect("model-relative path"),
		Path::new("hub/unnamespaced/gpt2").join(&revision)
	);
	assert_eq!(
		namespaced
			.strip_prefix(manager.home.models_dir())
			.expect("model-relative path"),
		Path::new("hub/namespaced/owner/model").join(&revision)
	);
	assert_ne!(unnamespaced, namespaced);
}

#[test]
fn load_policy_resolves_set_clear_and_model_limits() {
	let mut config = Config::default();
	config.inference.temperature = 0.7;
	config.inference.top_p = 0.8;
	config.inference.top_k = Some(20);
	config.inference.seed = Some(7);
	let (_directory, manager) = manager(config);
	let traits = ModelTraits {
		sizing: Some(ModelSizing {
			max_context_tokens: Some(4_096),
			..ModelSizing::default()
		}),
		..ModelTraits::default()
	};
	let policy = manager
		.resolve_load_policy(
			&traits,
			&ModelLoadOptions {
				max_tokens: Some(8_192),
				context_tokens: Some(8_192),
				temperature: LoadOverride::Clear,
				top_p: LoadOverride::Set(0.5),
				top_k: LoadOverride::Clear,
				seed: LoadOverride::Set(42),
				thinking: Some(ThinkingMode::On),
				reasoning_budget_tokens: LoadOverride::Set(1_024),
				..ModelLoadOptions::default()
			},
		)
		.expect("valid resolved policy");
	assert_eq!(policy.max_tokens, 4_096);
	assert_eq!(policy.context_tokens, 4_096);
	assert_eq!(policy.temperature, 0.0);
	assert_eq!(policy.top_p, 0.5);
	assert_eq!(policy.top_k, None);
	assert_eq!(policy.seed, Some(42));
	assert_eq!(policy.thinking, ThinkingMode::On);
	assert_eq!(policy.reasoning_budget_tokens, Some(1_024));
}

#[test]
fn load_policy_rejects_speculation_without_runtime_verified_mtp() {
	let (_directory, manager) = manager(Config::default());
	let error = manager
		.resolve_load_policy(
			&ModelTraits {
				mtp: MtpSupport::Advertised,
				..ModelTraits::default()
			},
			&ModelLoadOptions {
				speculative_tokens: Some(1),
				..ModelLoadOptions::default()
			},
		)
		.expect_err("advertised-only MTP must fail closed");
	assert!(error.to_string().contains("runtime-verified MTP"));
}

#[test]
fn verification_baseline_can_disable_global_mtp_for_non_mtp_snapshot() {
	let mut config = Config::default();
	config.inference.mtp = true;
	config.inference.speculative_tokens = 3;
	let (_directory, manager) = manager(config);
	let policy = manager
		.resolve_load_policy(
			&ModelTraits::default(),
			&ModelLoadOptions {
				speculative_tokens: Some(0),
				..ModelLoadOptions::default()
			},
		)
		.expect("verification baseline");
	assert_eq!(policy.speculative_tokens, 0);
}

#[test]
fn verification_stamp_matches_post_chmod_file_metadata() {
	let directory = tempfile::tempdir().expect("temporary directory");
	let root = directory.path().join("snapshot");
	fs::create_dir(&root).expect("snapshot directory");
	let manifest = manifest(runtime_files(&root));
	write_manifest(&root, &manifest).expect("manifest");
	make_read_only_contents(&root).expect("read-only runtime files");
	write_verification_stamp(&root, &manifest).expect("verification stamp");
	set_mode(&root.join(VERIFIED_STAMP_NAME), 0o400).expect("read-only stamp");
	set_mode(&root, 0o500).expect("read-only root");
	assert!(verification_stamp_matches(&root, manifest.files()).expect("stamp check"));
	make_writable(&root).expect("restore fixture permissions");
}

#[test]
fn verification_stamp_rejects_runtime_mutation() {
	let directory = tempfile::tempdir().expect("temporary directory");
	let root = directory.path().join("snapshot");
	fs::create_dir(&root).expect("snapshot directory");
	let manifest = manifest(runtime_files(&root));
	write_manifest(&root, &manifest).expect("manifest");
	make_read_only_contents(&root).expect("read-only runtime files");
	write_verification_stamp(&root, &manifest).expect("verification stamp");
	set_mode(&root.join(VERIFIED_STAMP_NAME), 0o400).expect("read-only stamp");
	set_mode(&root, 0o500).expect("read-only root");
	let weights = root.join("model.safetensors");
	set_mode(&weights, 0o600).expect("make fixture mutable");
	fs::write(&weights, b"changed").expect("mutate fixture");
	assert!(!verification_stamp_matches(&root, manifest.files()).expect("stamp check"));
	make_writable(&root).expect("restore fixture permissions");
}

#[test]
fn local_snapshot_digest_is_order_independent() {
	let directory = tempfile::tempdir().expect("temporary directory");
	let mut files = runtime_files(directory.path());
	let first = snapshot_digest(&files);
	files.reverse();
	assert_eq!(first, snapshot_digest(&files));
}

#[test]
fn move_retirement_removes_only_unchanged_selected_files() {
	let directory = tempfile::tempdir().expect("temporary directory");
	let source = directory.path().join("source");
	fs::create_dir(&source).expect("source directory");
	runtime_source(&source);
	fs::write(source.join("README.md"), "caller-owned notes").expect("extra source file");
	let plan = vec![
		"config.json".to_string(),
		"model.safetensors".to_string(),
		"tokenizer.json".to_string(),
	];
	let snapshots = capture_source_snapshots(&source, &plan).expect("source snapshots");
	let files = snapshot_runtime_files(&source, &plan).expect("runtime file hashes");

	let disposition = retire_source_files(&source, &snapshots, &files);

	assert!(matches!(
		disposition,
		ImportSourceDisposition::Retained { ref path, .. } if path == &source
	));
	assert!(source.join("README.md").is_file());
	for path in plan {
		assert!(!source.join(path).exists());
	}
}

#[test]
fn move_retirement_keeps_a_source_file_changed_after_certification() {
	let directory = tempfile::tempdir().expect("temporary directory");
	let source = directory.path().join("source");
	fs::create_dir(&source).expect("source directory");
	runtime_source(&source);
	let plan = vec![
		"config.json".to_string(),
		"model.safetensors".to_string(),
		"tokenizer.json".to_string(),
	];
	let snapshots = capture_source_snapshots(&source, &plan).expect("source snapshots");
	let files = snapshot_runtime_files(&source, &plan).expect("runtime file hashes");
	fs::write(source.join("tokenizer.json"), br#"{"changed":true}"#).expect("change source");

	let disposition = retire_source_files(&source, &snapshots, &files);

	assert!(matches!(
		disposition,
		ImportSourceDisposition::Retained { ref message, .. }
			if message.contains("changed after import")
	));
	assert!(source.join("tokenizer.json").is_file());
	assert!(!source.join("config.json").exists());
	assert!(!source.join("model.safetensors").exists());
}

#[test]
fn linked_record_rehashes_external_runtime_files_and_detects_mutation() {
	use std::os::unix::fs::symlink;

	let directory = tempfile::tempdir().expect("temporary directory");
	let source = directory.path().join("source");
	let record = directory.path().join("record");
	fs::create_dir(&source).expect("source directory");
	fs::create_dir(&record).expect("record directory");
	runtime_source(&source);
	let source = fs::canonicalize(source).expect("canonical source");
	let plan = local_runtime_plan(&source).expect("runtime plan");
	let files = snapshot_runtime_files(&source, &plan).expect("runtime files");
	let name = LocalModelName::parse("linked").expect("local name");
	let weights_bytes = files
		.iter()
		.filter(|file| file.path().ends_with(".safetensors"))
		.map(ModelFile::size)
		.sum();
	let traits = ModelTraits {
		mlx: true,
		tasks: BTreeSet::from([Task::TextGeneration]),
		sizing: Some(ModelSizing {
			weights_bytes: Some(weights_bytes),
			estimated_residency_bytes: Some(weights_bytes + 1),
			evaluated_context_tokens: Some(16),
			max_context_tokens: Some(32),
		}),
		..ModelTraits::default()
	};
	let manifest = ModelManifest::new(
		ModelRef::Local(name),
		ModelSource::LocalSymlink {
			original_path: source.clone(),
		},
		None,
		files,
		traits,
		VerificationStatus::Estimated,
		None,
	)
	.expect("linked manifest");
	symlink(&source, record.join(LINKED_SOURCE_NAME)).expect("source link");
	write_manifest(&record, &manifest).expect("link manifest");

	let runtime = linked_runtime_directory(&record, &manifest).expect("linked runtime");
	verify_linked_files(&record, &runtime, &manifest).expect("initial linked verification");
	fs::write(source.join("tokenizer.json"), br#"{"changed":true}"#).expect("mutate source");
	assert!(verify_linked_files(&record, &runtime, &manifest).is_err());
}

#[test]
fn inventory_rejects_an_orphaned_control_link() {
	use std::os::unix::fs::symlink;

	let (directory, manager) = manager(Config::default());
	let orphan = manager.home.models_dir().join("orphan");
	fs::create_dir(&orphan).expect("orphan directory");
	symlink(directory.path(), orphan.join(LINKED_SOURCE_NAME)).expect("orphan control link");

	let inventory = manager.inventory().expect("model inventory");
	assert!(inventory.models.is_empty());
	assert!(inventory.diagnostics.iter().any(|diagnostic| {
		diagnostic.path == orphan.join(LINKED_SOURCE_NAME)
			&& diagnostic.message.contains("symlinks are forbidden")
	}));
}

#[test]
fn linked_record_publication_inventory_removal_and_gc_never_touch_source() {
	use std::os::unix::fs::symlink;

	let (directory, manager) = manager(Config::default());
	let source = directory.path().join("external");
	fs::create_dir(&source).expect("external source");
	runtime_source(&source);
	let source = fs::canonicalize(source).expect("canonical source");
	let plan = local_runtime_plan(&source).expect("runtime plan");
	let files = snapshot_runtime_files(&source, &plan).expect("runtime files");
	let name = LocalModelName::parse("linked-lifecycle").expect("local name");
	let manifest = ModelManifest::new(
		ModelRef::Local(name.clone()),
		ModelSource::LocalSymlink {
			original_path: source.clone(),
		},
		None,
		files.clone(),
		verified_local_traits(&files),
		VerificationStatus::Verified,
		None,
	)
	.expect("linked manifest");
	let destination = manager.local_destination(&name, &snapshot_digest(&files));
	let staging = manager.create_staging("linked-test").expect("link staging");
	symlink(&source, staging.path().join(LINKED_SOURCE_NAME)).expect("source link");
	write_manifest(staging.path(), &manifest).expect("link manifest");

	let installed = manager
		.publish_link(staging, &destination, &manifest)
		.expect("publish linked record");
	assert_eq!(manager.list().expect("linked inventory").len(), 1);
	let quarantine = manager.remove(&installed).expect("remove linked record");
	assert!(source.is_dir());
	assert!(!installed.path().exists());
	assert!(quarantine.is_dir());

	assert_eq!(
		manager
			.gc_quarantine(Duration::ZERO)
			.expect("collect linked record"),
		1
	);
	assert!(source.is_dir());
	assert!(source.join("model.safetensors").is_file());
}

#[test]
fn move_and_link_sources_cannot_overlap_emelex_home() {
	let directory = tempfile::tempdir().expect("temporary directory");
	let home = directory.path().join("home");
	let source = home.join("models/source");
	fs::create_dir_all(&source).expect("nested source");
	assert!(reject_home_overlap(&home, &source).is_err());
	assert!(reject_home_overlap(&source, &home).is_err());
}

#[test]
fn import_reuse_requires_matching_ownership_and_link_target() {
	let first = Path::new("/tmp/first");
	let second = Path::new("/tmp/second");
	let owned = ModelSource::LocalImport {
		original_path: first.to_path_buf(),
	};
	let linked = ModelSource::LocalSymlink {
		original_path: first.to_path_buf(),
	};

	assert!(!source_satisfies_import(
		&owned,
		ModelSourceKind::LocalSymlink,
		Some(second),
	));
	assert!(source_satisfies_import(
		&linked,
		ModelSourceKind::LocalSymlink,
		Some(first),
	));
	assert!(!source_satisfies_import(
		&linked,
		ModelSourceKind::LocalSymlink,
		Some(second),
	));
	assert!(!source_satisfies_import(
		&linked,
		ModelSourceKind::LocalOwned,
		None,
	));
}

#[test]
fn ownership_collision_preserves_the_healthy_existing_snapshot() {
	let (_directory, manager) = manager(Config::default());
	let staging = manager.create_staging("owned-collision").expect("staging");
	let files = runtime_files(staging.path());
	let name = LocalModelName::parse("owned-collision").expect("local name");
	let reference = ModelRef::Local(name.clone());
	let manifest = ModelManifest::new(
		reference.clone(),
		ModelSource::LocalImport {
			original_path: PathBuf::from("/tmp/owned-collision"),
		},
		None,
		files.clone(),
		verified_local_traits(&files),
		VerificationStatus::Verified,
		None,
	)
	.expect("owned manifest");
	write_manifest(staging.path(), &manifest).expect("manifest");
	let destination = manager.local_destination(&name, &snapshot_digest(&files));
	let installed = manager
		.publish(staging, &destination, &manifest, None)
		.expect("publish owned snapshot");

	let error = manager
		.reuse_existing_locked(
			&destination,
			&reference,
			None,
			installed.snapshot_id(),
			ModelSourceKind::LocalSymlink,
			Some(Path::new("/tmp/external-link")),
		)
		.expect_err("ownership mismatch must fail");
	assert!(matches!(error, ModelsError::ImportOwnershipConflict(_)));
	assert_eq!(
		manager
			.load_installed_at(&destination)
			.expect("existing snapshot remains")
			.snapshot_id(),
		installed.snapshot_id()
	);
}

#[test]
fn direct_manager_protects_snapshots_bound_to_durable_sessions() {
	let directory = tempfile::tempdir().expect("temporary directory");
	let home = EmelexHome::prepare(&directory.path().join("home")).expect("test Emelex home");
	let installed = install_test_snapshot(&home).expect("test snapshot");
	let store = MemoryStore::open(&home).expect("memory store");
	let workspace = tempfile::tempdir().expect("workspace");
	let session = store
		.start_session(workspace.path(), None)
		.expect("durable session");
	store
		.bind_session_model(session.id, &installed)
		.expect("bind exact snapshot");

	let config = Config::default();
	let workload =
		WorkloadProfile::new(1, config.inference.context_tokens).expect("valid workload");
	let budget = 8_u64 << 30;
	let hub = HubClient::with_fit_profile(config.hub.clone(), workload, budget)
		.expect("profiled Hub client");
	let manager = ModelManager::new(home, config, hub, budget).expect("direct model manager");

	let error = manager
		.remove(&installed)
		.expect_err("default guard must fail closed for a durable reference");
	assert!(
		matches!(error, ModelsError::SnapshotReferenced(snapshot) if snapshot == *installed.snapshot_id())
	);
	assert!(installed.path().is_dir());
}

#[test]
fn inventory_reports_missing_model_store_root() {
	let (_directory, manager) = manager(Config::default());
	fs::remove_dir(manager.home.models_dir()).expect("remove empty model store");

	let error = manager
		.inventory()
		.expect_err("missing model store must fail");
	assert!(matches!(error, ModelsError::Io { path, .. } if path == manager.home.models_dir()));
}

#[test]
fn installed_manifest_tampering_invalidates_verification_stamp() {
	let (_directory, manager) = manager(Config::default());
	let installed = install_test_snapshot(&manager.home).expect("test snapshot");
	let manifest_path = installed.path().join(MANIFEST_NAME);
	set_mode(installed.path(), 0o700).expect("writable snapshot directory");
	set_mode(&manifest_path, 0o600).expect("writable manifest");
	let mut manifest: serde_json::Value =
		serde_json::from_slice(&fs::read(&manifest_path).expect("read manifest"))
			.expect("decode manifest");
	manifest["license"] = serde_json::Value::String("forged-license".to_string());
	fs::write(
		&manifest_path,
		serde_json::to_vec_pretty(&manifest).expect("encode changed manifest"),
	)
	.expect("tamper manifest");
	set_mode(&manifest_path, 0o400).expect("restore manifest mode");
	set_mode(installed.path(), 0o500).expect("restore snapshot mode");

	let error = manager
		.load_installed_at(installed.path())
		.expect_err("tampered manifest must fail closed");
	assert!(matches!(
		error,
		ModelsError::InvalidVerificationStamp(snapshot) if snapshot == *installed.snapshot_id()
	));
}

#[test]
fn invalid_existing_repair_serializes_against_durable_binding() {
	let (_directory, manager) = manager(Config::default());
	let installed = install_test_snapshot(&manager.home).expect("test snapshot");
	let store = MemoryStore::open(&manager.home).expect("memory store");
	let workspace = tempfile::tempdir().expect("workspace");
	let session = store
		.start_session(workspace.path(), None)
		.expect("durable session");
	let snapshot = installed.snapshot_id().clone();
	let reference = installed.reference().clone();
	let mutation_lock = manager
		.snapshot_mutation_lock()
		.expect("snapshot mutation lock");
	set_mode(installed.path(), 0o700).expect("writable snapshot");
	fs::remove_file(installed.path().join(VERIFIED_STAMP_NAME))
		.expect("invalidate verification stamp");
	let (started_sender, started_receiver) = mpsc::channel();
	let binder_store = store.clone();
	let binder_installed = installed.clone();
	let binder = thread::spawn(move || {
		started_sender.send(()).expect("signal binder");
		binder_store.bind_session_model(session.id, &binder_installed)
	});

	started_receiver.recv().expect("binder started");
	assert!(
		manager
			.reuse_existing_locked(
				installed.path(),
				&reference,
				None,
				&snapshot,
				ModelSourceKind::Hub,
				None,
			)
			.expect("repair invalid existing")
			.is_none()
	);
	drop(mutation_lock);

	assert!(
		binder
			.join()
			.expect("binder thread")
			.expect_err("binding removed snapshot must fail")
			.to_string()
			.contains("snapshot")
	);
	assert!(
		store
			.session(session.id)
			.expect("session")
			.model_snapshot
			.is_none()
	);
	assert!(!installed.path().exists());
}

#[tokio::test]
async fn controlled_verification_observes_cancellation() {
	let directory = tempfile::tempdir().expect("temporary directory");
	let files = runtime_files(directory.path());
	let cancellation = DownloadCancellation::default();
	cancellation.cancel();

	let error = verify_files_controlled(directory.path(), &files, Some(&cancellation))
		.await
		.expect_err("cancelled verification must fail");
	assert!(matches!(error, ModelsError::Hub(HubError::Cancelled)));
}

#[tokio::test(flavor = "current_thread")]
async fn controlled_verification_keeps_current_thread_runtime_responsive() {
	const LARGE_BYTES: u64 = 64 << 20;
	let directory = tempfile::tempdir().expect("temporary directory");
	let path = directory.path().join("model.safetensors");
	let file = OpenOptions::new()
		.write(true)
		.create_new(true)
		.open(&path)
		.expect("create sparse weights");
	file.set_len(LARGE_BYTES).expect("size sparse weights");
	drop(file);
	let mut digest = sha2::Sha256::new();
	let zeros = vec![0_u8; 1 << 20];
	for _ in 0..64 {
		digest.update(&zeros);
	}
	let files = vec![
		ModelFile::new(
			"model.safetensors",
			LARGE_BYTES,
			hex::encode(digest.finalize()),
		)
		.expect("large file record"),
	];
	let verification = verify_files_controlled(directory.path(), &files, None);
	tokio::pin!(verification);

	tokio::select! {
		biased;
		() = tokio::time::sleep(Duration::from_millis(1)) => {}
		result = &mut verification => panic!("verification blocked runtime timer: {result:?}"),
	}

	verification.await.expect("large controlled verification");
}

#[tokio::test(flavor = "current_thread")]
async fn dropped_download_operation_cancels_and_defers_staging_cleanup() {
	let (_directory, manager) = manager(Config::default());
	let operation = DownloadOperationGuard::new(None);
	let observed_cancellation = operation.cancellation().clone();
	let mut staging = manager.create_staging("dropped").expect("staging");
	staging.cleanup_delay = Some(Duration::from_secs(1));
	let staging_path = staging.path().to_path_buf();
	let mut operation = Box::pin(async move {
		let _operation = operation;
		let _staging = staging;
		std::future::pending::<()>().await;
	});
	tokio::select! {
		() = &mut operation => panic!("pending operation completed"),
		() = tokio::task::yield_now() => {}
	}

	let started = std::time::Instant::now();
	drop(operation);
	assert!(
		started.elapsed() < Duration::from_millis(250),
		"future drop performed staging cleanup synchronously"
	);
	assert!(observed_cancellation.is_cancelled());
	tokio::time::timeout(Duration::from_secs(3), async {
		while staging_path.exists() {
			tokio::time::sleep(Duration::from_millis(10)).await;
		}
	})
	.await
	.expect("deferred quarantine completes");
}

#[test]
fn dropped_download_child_does_not_cancel_shared_caller_authority() {
	let caller = DownloadCancellation::default();
	let first = DownloadOperationGuard::new(Some(&caller));
	let first_child = first.cancellation().clone();
	let second = DownloadOperationGuard::new(Some(&caller));
	let second_child = second.cancellation().clone();

	drop(first);

	assert!(first_child.is_cancelled());
	assert!(!caller.is_cancelled());
	assert!(!second_child.is_cancelled());
	caller.cancel();
	assert!(second_child.is_cancelled());
}

#[tokio::test(flavor = "current_thread")]
async fn staging_cleanup_drop_storm_uses_one_nonblocking_worker() {
	let (_directory, manager) = manager(Config::default());
	let started_before = STAGING_CLEANUP_TASKS_STARTED.load(Ordering::Relaxed);
	let mut blocker = manager.create_staging("cleanup-blocker").expect("blocker");
	blocker.cleanup_delay = Some(Duration::from_millis(500));
	drop(blocker);
	tokio::time::timeout(Duration::from_secs(3), async {
		while STAGING_CLEANUP_TASKS_STARTED.load(Ordering::Relaxed) == started_before {
			tokio::task::yield_now().await;
		}
	})
	.await
	.expect("cleanup worker starts blocker");

	let guards = (0..64)
		.map(|_| manager.create_staging("cleanup-storm").expect("staging"))
		.collect::<Vec<_>>();
	let started = std::time::Instant::now();
	drop(guards);

	assert!(
		started.elapsed() < Duration::from_millis(250),
		"staging guard drop blocked on cleanup"
	);
	assert_eq!(STAGING_CLEANUP_WORKERS_STARTED.load(Ordering::Relaxed), 1);
}

#[test]
fn cancelled_prepublish_checkpoint_leaves_destination_absent() {
	let (_directory, manager) = manager(Config::default());
	let staging = manager.create_staging("cancelled").expect("staging");
	let expected = manifest(runtime_files(staging.path()));
	let destination = manager.home.models_dir().join("cancelled-destination");
	let cancellation = DownloadCancellation::default();
	cancellation.cancel();

	let error = manager
		.publish(staging, &destination, &expected, Some(&cancellation))
		.expect_err("cancelled install must not publish");
	assert!(matches!(error, ModelsError::Hub(HubError::Cancelled)));
	assert!(!destination.exists());
}

#[test]
fn cancellation_after_publish_preparation_leaves_destination_absent() {
	let (_directory, manager) = manager(Config::default());
	let staging = manager.create_staging("late-cancelled").expect("staging");
	let expected = manifest(runtime_files(staging.path()));
	write_manifest(staging.path(), &expected).expect("staging manifest");
	let destination = manager.home.models_dir().join("late-cancelled-destination");
	let cancellation = DownloadCancellation::default();

	let error = manager
		.publish_inner(
			staging,
			&destination,
			&expected,
			Some(&cancellation),
			|| cancellation.cancel(),
		)
		.expect_err("cancellation immediately before rename must stop publication");
	assert!(matches!(error, ModelsError::Hub(HubError::Cancelled)));
	assert!(!destination.exists());
}

#[test]
fn certification_boundary_wraps_only_candidate_local_failures() {
	let inspection = ModelsError::Inspection(InspectionError::Config {
		path: PathBuf::from("config.json"),
		message: "unsupported model type".to_string(),
	});
	assert!(matches!(
		mark_candidate_certification_error(inspection),
		ModelsError::Certification(inner)
			if matches!(*inner, ModelsError::Inspection(InspectionError::Config { .. }))
	));

	let load = ModelsError::Client(Error::ModelLoad {
		path: PathBuf::from("model"),
		message: "tensor shape mismatch".to_string(),
	});
	assert!(matches!(
		mark_candidate_certification_error(load),
		ModelsError::Certification(inner)
			if matches!(*inner, ModelsError::Client(Error::ModelLoad { .. }))
	));

	let runtime = ModelsError::Client(Error::Runtime(
		crate::runtime::RuntimeError::MetalDeviceUnavailable,
	));
	assert!(matches!(
		mark_candidate_certification_error(runtime),
		ModelsError::Client(Error::Runtime(
			crate::runtime::RuntimeError::MetalDeviceUnavailable
		))
	));
	assert!(matches!(
		mark_candidate_certification_error(ModelsError::Client(Error::InferencePanic)),
		ModelsError::Client(Error::InferencePanic)
	));
	assert!(matches!(
		mark_candidate_certification_error(ModelsError::Client(Error::InvalidRequest(
			"hardcoded probe invariant".to_string()
		))),
		ModelsError::Client(Error::InvalidRequest(_))
	));
	assert!(matches!(
		mark_candidate_certification_error(ModelsError::Client(Error::ModelPath {
			path: PathBuf::from("model"),
			reason: "inference thread died".to_string(),
		})),
		ModelsError::Client(Error::ModelPath { .. })
	));

	let read = ModelsError::Inspection(InspectionError::Read {
		path: PathBuf::from("config.json"),
		source: std::io::Error::other("disk unavailable"),
	});
	assert!(matches!(
		mark_candidate_certification_error(read),
		ModelsError::Inspection(InspectionError::Read { .. })
	));

	assert!(matches!(
		mark_hub_candidate_certification_error(HubError::Incompatible(
			"unsupported repository".to_string()
		)),
		ModelsError::Certification(inner)
			if matches!(*inner, ModelsError::Hub(HubError::Incompatible(_)))
	));
	assert!(matches!(
		mark_hub_candidate_certification_error(HubError::Cancelled),
		ModelsError::Hub(HubError::Cancelled)
	));
}
