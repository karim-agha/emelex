//! emelex patch (not upstream): the MTP logit-parity enablement gate.
//!
//! The Python dump script (`tools/mtp_parity_dump.py`, pinned mlx-lm
//! fork @ 45f53582d64287aa875c1606e479f7f66c0afb58) writes first-step and
//! recursive-step MTP logits as `.npy` goldens; the test here replays the same
//! recipe through this engine and compares. `tools/party.py` runs only this
//! ignored external fixture test and enforces a hard 20-minute process-group
//! deadline. Missing inputs, hash drift, missing MTP, timeout, or parity drift
//! fail that gate. The certified workload is exactly three steps: first plus
//! two recursive.

#[cfg(test)]
mod tests {
	#![allow(clippy::unwrap_used, clippy::expect_used)]

	use std::{
		collections::{BTreeMap, BTreeSet},
		path::{Component, Path, PathBuf},
	};

	use serde::Deserialize;
	use sha2::{Digest as _, Sha256};

	use crate::engine::{array::Array, generate::Session, ops};

	const CERTIFICATION_SCHEMA: u32 = 2;
	const IMPLEMENTATION_ID: &str = "emelex-qwen3.5-mtp-dense-bf16-v1";
	const REQUIRED_STEPS: usize = 3;
	const MAX_CERTIFICATION_BYTES: u64 = 1 << 20;
	const MAX_GOLDEN_BYTES: u64 = 8 << 20;
	const MODEL_FILES: [&str; 3] = [
		"config.json",
		"model-00001-of-00002.safetensors",
		"model-00002-of-00002.safetensors",
	];
	const GOLDEN_FILES: [&str; 4] = ["meta.json", "step0.npy", "step1.npy", "step2.npy"];

	#[derive(Debug, Deserialize)]
	#[serde(deny_unknown_fields)]
	struct Certification {
		schema_version: u32,
		implementation_id: String,
		required_steps: usize,
		model: CertifiedModel,
		reference: ReferencePins,
		goldens: CertifiedGoldens,
	}

	#[derive(Debug, Deserialize)]
	#[serde(deny_unknown_fields)]
	struct CertifiedModel {
		source: RepositoryPin,
		equivalence_reference: RepositoryPin,
		files: Vec<CertifiedFile>,
	}

	#[derive(Debug, Deserialize)]
	#[serde(deny_unknown_fields)]
	struct RepositoryPin {
		repository: String,
		revision: String,
	}

	#[derive(Debug, Deserialize)]
	#[serde(deny_unknown_fields)]
	struct ReferencePins {
		converter: ConverterPin,
		mlx_version: String,
		python_version: String,
	}

	#[derive(Debug, Deserialize)]
	#[serde(deny_unknown_fields)]
	struct ConverterPin {
		repository: String,
		revision: String,
		package_version: String,
		python_tree_sha256: String,
	}

	#[derive(Debug, Deserialize)]
	#[serde(deny_unknown_fields)]
	struct CertifiedGoldens {
		generator: String,
		files: Vec<CertifiedFile>,
	}

	#[derive(Debug, Deserialize)]
	#[serde(deny_unknown_fields)]
	struct CertifiedFile {
		path: String,
		sha256: String,
	}

	#[derive(Debug, Deserialize)]
	#[serde(deny_unknown_fields)]
	struct GoldenMetadata {
		prompt_ids: Vec<u32>,
		greedy_tokens: Vec<u32>,
		python_version: String,
		mlx_version: String,
		mlx_lm_version: String,
		mlx_lm_source: GoldenSource,
		mlx_lm_tree_sha256: String,
		config_sha256: String,
		steps: usize,
	}

	#[derive(Debug, Deserialize)]
	#[serde(deny_unknown_fields)]
	struct GoldenSource {
		kind: String,
		repository: String,
		revision: String,
	}

	fn load_certification() -> Certification {
		let path =
			Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mtp_certification.json");
		let bytes = crate::artifact::read_bytes(&path, MAX_CERTIFICATION_BYTES)
			.expect("checked-in MTP certification readable");
		let certification: Certification =
			serde_json::from_slice(&bytes).expect("checked-in MTP certification parses");
		validate_certification(&certification);
		certification
	}

	fn validate_certification(certification: &Certification) {
		assert_eq!(
			certification.schema_version, CERTIFICATION_SCHEMA,
			"unsupported MTP certification schema"
		);
		assert_eq!(
			certification.implementation_id, IMPLEMENTATION_ID,
			"MTP certification names another implementation"
		);
		assert_eq!(
			certification.required_steps, REQUIRED_STEPS,
			"MTP certification must bind the exact three-step workload"
		);
		validate_repository_pin(&certification.model.source, "weight source");
		validate_repository_pin(
			&certification.model.equivalence_reference,
			"equivalence reference",
		);
		assert!(
			!certification.reference.converter.repository.is_empty(),
			"converter repository pin is empty"
		);
		assert_commit(
			&certification.reference.converter.revision,
			"converter revision",
		);
		assert!(
			!certification.reference.converter.package_version.is_empty(),
			"converter package version is empty"
		);
		assert_sha256(
			&certification.reference.converter.python_tree_sha256,
			"converter Python tree",
		);
		assert!(
			!certification.reference.mlx_version.is_empty(),
			"MLX version pin is empty"
		);
		assert!(
			!certification.reference.python_version.is_empty(),
			"Python version pin is empty"
		);
		assert_eq!(
			certification.goldens.generator, "tools/mtp_parity_dump.py",
			"unexpected golden generator"
		);
		validate_file_set(&certification.model.files, &MODEL_FILES);
		validate_file_set(&certification.goldens.files, &GOLDEN_FILES);
	}

	fn validate_repository_pin(pin: &RepositoryPin, label: &str) {
		assert!(!pin.repository.is_empty(), "{label} repository is empty");
		assert_commit(&pin.revision, label);
	}

	fn assert_commit(value: &str, label: &str) {
		assert!(
			value.len() == 40
				&& value
					.bytes()
					.all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
			"{label} must be a full lowercase Git commit"
		);
	}

	fn assert_sha256(value: &str, label: &str) {
		assert!(
			value.len() == 64
				&& value
					.bytes()
					.all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
			"{label} must be a lowercase SHA-256"
		);
	}

	fn validate_file_set(files: &[CertifiedFile], expected: &[&str]) {
		let mut actual = BTreeSet::new();
		for file in files {
			let path = Path::new(&file.path);
			let mut components = path.components();
			assert!(
				matches!(components.next(), Some(Component::Normal(_)))
					&& components.next().is_none(),
				"certified path must be one plain file name: {:?}",
				file.path
			);
			assert_sha256(&file.sha256, &format!("certified file {}", file.path));
			assert!(
				actual.insert(file.path.as_str()),
				"duplicate certified file {}",
				file.path
			);
		}
		let expected: BTreeSet<&str> = expected.iter().copied().collect();
		assert_eq!(actual, expected, "certified file set drifted");
	}

	fn verify_golden_files(root: &Path, files: &[CertifiedFile]) -> BTreeMap<String, Vec<u8>> {
		let mut verified = BTreeMap::new();
		for file in files {
			let path = root.join(&file.path);
			let bytes = crate::artifact::read_bytes(&path, MAX_GOLDEN_BYTES)
				.expect("certified golden readable");
			assert_eq!(
				hex::encode(Sha256::digest(&bytes)),
				file.sha256,
				"certified golden hash mismatch for {}",
				file.path
			);
			verified.insert(file.path.clone(), bytes);
		}
		verified
	}

	fn certified_hash<'a>(files: &'a [CertifiedFile], path: &str) -> &'a str {
		files
			.iter()
			.find(|file| file.path == path)
			.map(|file| file.sha256.as_str())
			.expect("certified file exists")
	}

	/// Minimal `.npy` v1/v2 reader for little-endian f32 vectors.
	fn read_npy_f32(bytes: &[u8], name: &str) -> Vec<f32> {
		assert!(bytes.len() >= 10, "{name} is too short for an npy header");
		assert_eq!(&bytes[..6], b"\x93NUMPY", "not an npy file");
		let (major, header_len, data_start) = {
			let major = bytes[6];
			if major >= 2 {
				assert!(bytes.len() >= 12, "{name} has a truncated npy header");
				let len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
				(major, len, 12 + len)
			} else {
				let len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
				(major, len, 10 + len)
			}
		};
		assert!(
			data_start <= bytes.len(),
			"{name} has a truncated npy header"
		);
		let header =
			std::str::from_utf8(&bytes[data_start - header_len..data_start]).expect("ascii header");
		assert!(
			header.contains("'descr': '<f4'"),
			"golden must be little-endian f32, got {header} (npy v{major})"
		);
		assert!(
			header.contains("'fortran_order': False"),
			"golden must be C-order"
		);
		assert_eq!(
			(bytes.len() - data_start) % size_of::<f32>(),
			0,
			"{name} has a partial f32"
		);
		bytes[data_start..]
			.chunks_exact(4)
			.map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
			.collect()
	}

	fn top_indices(row: &[f32], n: usize) -> Vec<usize> {
		let mut indexed: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
		indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
		indexed.into_iter().take(n).map(|(i, _)| i).collect()
	}

	/// One-sided top-set containment, applied in both directions: each
	/// side's top-5 must sit inside the other's top-8 (tie-robust, per
	/// the certification gate definition).
	fn top_sets_agree(ours: &[f32], golden: &[f32]) -> bool {
		let ours8: std::collections::HashSet<usize> = top_indices(ours, 8).into_iter().collect();
		let golden8: std::collections::HashSet<usize> =
			top_indices(golden, 8).into_iter().collect();
		top_indices(ours, 5).iter().all(|i| golden8.contains(i))
			&& top_indices(golden, 5).iter().all(|i| ours8.contains(i))
	}

	/// Thresholds revised against the first recorded real deltas
	/// (2026-07-26, Qwen3.5-4B fixture, mlx 0.32.0 vs this engine):
	/// step-0 max|diff| 0.2017, mean|diff| 0.0359 at max|logit| 18.25,
	/// with the top-5 sets IDENTICAL in identical order - i.e. ~1.6
	/// bf16 ulps at that logit scale (ulp(16..32) = 0.125), pure
	/// hardware numerics between two independent kernel stacks. The
	/// original provisional 2e-2 was below one bf16 ulp at this scale
	/// and therefore unsatisfiable by ANY independent bf16
	/// implementation. Loosening recorded in the parity manifest:
	/// max-abs <= 0.5 (~4 ulps), mean-abs <= 0.1, and the semantic
	/// top-5-in-top-8 mutual containment unchanged.
	const MAX_ABS_DIFF: f32 = 0.5;
	const MAX_MEAN_ABS_DIFF: f64 = 0.1;

	#[test]
	fn checked_in_mtp_certification_is_well_formed() {
		load_certification();
	}

	#[test]
	#[ignore = "external certified fixture; run tools/party.py"]
	fn mtp_logit_parity_gate() {
		let model_dir = PathBuf::from(
			std::env::var_os("EMELEX_TEST_MODEL")
				.expect("EMELEX_TEST_MODEL must name the certified dense MTP fixture"),
		);
		let goldens = PathBuf::from(
			std::env::var_os("EMELEX_PARITY_GOLDENS")
				.expect("EMELEX_PARITY_GOLDENS must name the certified golden directory"),
		);
		let certification = load_certification();

		// emelex patch: one descriptor-backed snapshot establishes fixture
		// identity and is then consumed by Session. A root swap cannot make
		// this gate hash A and execute B.
		let runtime = crate::runtime::initialize_default_if_needed().expect("runtime initializes");
		let checkpoint = crate::model::layout::CheckpointSnapshot::open_in(
			&model_dir,
			&runtime.home().join("temp"),
		)
		.expect("certified descriptor-backed checkpoint");
		assert!(
			crate::engine::mtp_certification::model_is_certified(&checkpoint)
				.expect("embedded model certificate"),
			"descriptor-backed checkpoint does not match MTP certificate"
		);
		let golden_files = verify_golden_files(&goldens, &certification.goldens.files);
		let meta: GoldenMetadata =
			serde_json::from_slice(&golden_files["meta.json"]).expect("meta.json parses");
		assert_eq!(
			meta.steps, REQUIRED_STEPS,
			"meta.json must describe exactly three certified steps"
		);
		assert_eq!(
			meta.mlx_version, certification.reference.mlx_version,
			"meta.json MLX pin differs from certification"
		);
		assert_eq!(
			meta.python_version, certification.reference.python_version,
			"meta.json Python pin differs from certification"
		);
		assert_eq!(
			meta.mlx_lm_version, certification.reference.converter.package_version,
			"meta.json mlx-lm package differs from certification"
		);
		assert_eq!(
			meta.mlx_lm_source.kind, "local_git_checkout",
			"meta.json mlx-lm source kind is not the verified local checkout"
		);
		assert_eq!(
			meta.mlx_lm_source.repository, certification.reference.converter.repository,
			"meta.json mlx-lm repository differs from certification"
		);
		assert_eq!(
			meta.mlx_lm_source.revision, certification.reference.converter.revision,
			"meta.json converter fork differs from certification"
		);
		assert_eq!(
			meta.mlx_lm_tree_sha256, certification.reference.converter.python_tree_sha256,
			"meta.json installed/source tree differs from certification"
		);
		assert_eq!(
			meta.config_sha256,
			certified_hash(&certification.model.files, "config.json"),
			"meta.json config digest differs from certification"
		);
		let prompt_ids = meta.prompt_ids;
		let greedy_tokens = meta.greedy_tokens;
		assert!(!prompt_ids.is_empty(), "meta.json prompt_ids is empty");
		assert_eq!(
			greedy_tokens.len(),
			REQUIRED_STEPS + 1,
			"meta.json greedy_tokens must bind exactly three steps"
		);

		let session = Session::load_certified_snapshot_for_parity(&model_dir, checkpoint)
			.expect("certified model loads");
		assert!(
			session.supports_mtp(),
			"parity fixture must carry a loadable MTP module"
		);
		let model = session.model_for_tests();
		let mut caches = model.new_caches();
		let prompt_arr =
			Array::from_slice(&prompt_ids, &[1, i32::try_from(prompt_ids.len()).unwrap()]).unwrap();
		let backbone = model
			.forward_hidden(&prompt_arr, &mut caches)
			.expect("backbone forward");
		let shape = backbone.hidden_pre_norm.shape();
		let last = ops::slice(
			&backbone.hidden_pre_norm,
			&[0, shape[1] - 1, 0],
			&[1, shape[1], shape[2]],
		)
		.unwrap();
		let mut prev_hidden = ops::contiguous(&last).unwrap();
		prev_hidden.eval().unwrap();

		let mut mtp_caches = model.new_mtp_caches();
		let mut worst: f32 = 0.0;
		for step in 0..REQUIRED_STEPS {
			let token = greedy_tokens[step];
			let out = model
				.forward_mtp(
					&Array::from_slice(&[token], &[1, 1]).unwrap(),
					&prev_hidden,
					&mut mtp_caches,
				)
				.expect("mtp step");
			let logits_shape = out.logits.shape();
			let row = ops::reshape(
				&ops::slice(
					&out.logits,
					&[0, logits_shape[1] - 1, 0],
					&[1, logits_shape[1], logits_shape[2]],
				)
				.unwrap(),
				&[logits_shape[2]],
			)
			.unwrap();
			let ours = row.to_vec_f32().expect("logits row to host");
			let name = format!("step{step}.npy");
			let golden = read_npy_f32(
				golden_files.get(&name).expect("certified golden exists"),
				&name,
			);
			assert_eq!(ours.len(), golden.len(), "vocab mismatch at step {step}");
			let max_diff = ours
				.iter()
				.zip(&golden)
				.map(|(a, b)| (a - b).abs())
				.fold(0.0f32, f32::max);
			// Diagnostics ahead of the assert: distinguish bf16-scale
			// numerics (tiny mean, agreeing top sets) from a recipe bug
			// (systematic offsets, disordered tops).
			let mean_diff: f64 =
				ours.iter()
					.zip(&golden)
					.map(|(a, b)| f64::from((a - b).abs()))
					.sum::<f64>() / ours.len() as f64;
			let max_abs_logit = ours.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
			eprintln!(
				"step {step}: max|diff| {max_diff:.4} mean|diff| {mean_diff:.6} \
				 max|logit| {max_abs_logit:.2} ours-top5 {:?} golden-top5 {:?} \
				 top-sets-agree {}",
				top_indices(&ours, 5),
				top_indices(&golden, 5),
				top_sets_agree(&ours, &golden),
			);
			worst = worst.max(max_diff);
			assert!(
				max_diff <= MAX_ABS_DIFF,
				"step {step}: max |diff| {max_diff} exceeds {MAX_ABS_DIFF}"
			);
			assert!(
				mean_diff <= MAX_MEAN_ABS_DIFF,
				"step {step}: mean |diff| {mean_diff} exceeds {MAX_MEAN_ABS_DIFF}"
			);
			assert!(
				top_sets_agree(&ours, &golden),
				"step {step}: top-5/top-8 containment failed"
			);
			prev_hidden = ops::contiguous(&out.recycle_hidden).unwrap();
			prev_hidden.eval().unwrap();
		}
		eprintln!(
			"parity gate PASS over {REQUIRED_STEPS} steps; worst max-abs-diff {worst} \
			 (record in the parity manifest)"
		);
		let sentinel = PathBuf::from(
			std::env::var_os("EMELEX_PARTY_SENTINEL")
				.expect("parity gate must run through tools/party.py"),
		);
		std::fs::write(sentinel, format!("{IMPLEMENTATION_ID}\n"))
			.expect("party completion sentinel writes");
	}
}
