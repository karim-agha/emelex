//! MTP speculative-decoding benchmark harness.
//!
//! Implements the merge-gate methodology from the MTP plan: a fixed
//! workload matrix of {short chat, long context} x {greedy,
//! sampled(T=0.7, top-p 0.9)} x k in {off, 1, 2, 4, 6, 8}, run with
//! warmup runs excluded, >= 5 measured repetitions, and REP-MAJOR
//! (interleaved-across-cells) ordering with varying position and adjacency
//! to reduce fixed-order thermal bias. Per run: TTFT, decode tok/s,
//! acceptance-by-depth from
//! `SpeculationStatsData`, peak memory (watermark reset per run), and
//! the freed-buffer cache size. Prompt caching is disabled per request
//! so every run pays a fresh prefill and cells stay comparable.
//!
//! Usage:
//! ```sh
//! EMELEX_TEST_MODEL=/path/to/dense-mtp-model \
//!   cargo run --release -p emelex --features bench --example mtp_bench
//! ```
//! Knobs: `MTP_BENCH_REPS` (default 5), `MTP_BENCH_WARMUP` (default 2),
//! `MTP_BENCH_TOKENS` (default 512).
//!
//! The decision rule is evaluated for the pre-registered k* = 4 only
//! (other k are reported for information; no post-hoc k shopping):
//! PASS = spec-on median decode tok/s at k* exceeds the spec-off median
//! by > 5% with non-overlapping IQRs, in both greedy and sampled modes.
//! A JSON report lands in `mtp-bench-report.json` for the handoff
//! manifest.

#![allow(clippy::too_many_lines, missing_docs)]

use std::time::Instant;

use emelex::{Client, SpeculationStatsData};
use futures::StreamExt;
use rig_core::completion::CompletionModel as _;

const KS: &[usize] = &[0, 1, 2, 4, 6, 8];
const K_STAR: usize = 4;

const SHORT_PROMPT: &str = "Explain, step by step, why the sky is blue and \
                            how the answer changes at sunset.";

fn long_prompt() -> String {
	// A repetitive but coherent long-context prompt (~6k words) so the
	// prefill is genuinely long without shipping a corpus fixture.
	let mut prompt = String::from(
		"Below is a log of sensor readings. Summarize the trends, then explain \
		 what maintenance you would schedule and why.\n",
	);
	for i in 0..1500 {
		use std::fmt::Write as _;
		let _ = writeln!(
			prompt,
			"reading {i}: temp={}C vibration={} pressure={}kPa status=ok",
			20 + (i % 17),
			(i % 9) as f32 / 10.0,
			100 + (i % 23),
		);
	}
	prompt
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
	Greedy,
	Sampled,
}

struct Cell {
	prompt_name: &'static str,
	mode: Mode,
	k: usize,
}

#[derive(Default, Clone, serde::Serialize)]
struct RunRecord {
	sweep: usize,
	position: usize,
	ttft_ms: f64,
	decode_tok_s: f64,
	completion_tokens: u64,
	drafted: u64,
	rounds: u64,
	/// One-based depth buckets: index i counts rounds accepting exactly
	/// i + 1 drafts; full rejections increment no bucket.
	accepted_by_depth: Vec<u64>,
	peak_memory_bytes: u64,
	cache_memory_bytes: u64,
}

#[derive(serde::Serialize)]
struct CellReport {
	prompt: &'static str,
	mode: &'static str,
	k: usize,
	median_tok_s: f64,
	iqr_lo: f64,
	iqr_hi: f64,
	median_ttft_ms: f64,
	peak_memory_max: u64,
	runs: Vec<RunRecord>,
}

#[derive(serde::Serialize)]
struct BenchmarkReport<'a> {
	measured_repetitions: usize,
	warmup_sweeps: usize,
	max_tokens: usize,
	ordering: &'static str,
	cells: &'a [CellReport],
}

fn env_usize(name: &str, default: usize) -> std::io::Result<usize> {
	match std::env::var(name) {
		Ok(value) => value.parse().map_err(|_| {
			std::io::Error::other(format!("{name} must be an unsigned integer, got {value:?}"))
		}),
		Err(std::env::VarError::NotPresent) => Ok(default),
		Err(error) => Err(std::io::Error::other(format!("read {name}: {error}"))),
	}
}

fn quartiles(sorted: &[f64]) -> std::io::Result<(f64, f64, f64)> {
	if sorted.is_empty() {
		return Err(std::io::Error::other(
			"cannot compute quartiles for an empty benchmark cell",
		));
	}
	let pick = |q: f64| -> f64 {
		let pos = q * (sorted.len() - 1) as f64;
		#[allow(clippy::cast_sign_loss)] // q in [0,1], len >= 1
		let lo = pos.floor() as usize;
		#[allow(clippy::cast_sign_loss)]
		let hi = pos.ceil() as usize;
		let frac = pos - lo as f64;
		sorted[lo].mul_add(1.0 - frac, sorted[hi] * frac)
	};
	Ok((pick(0.25), pick(0.5), pick(0.75)))
}

const fn gcd(mut left: usize, mut right: usize) -> usize {
	while right != 0 {
		let remainder = left % right;
		left = right;
		right = remainder;
	}
	left
}

fn balanced_cell_order(cell_count: usize, sweep: usize) -> Vec<usize> {
	if cell_count <= 1 {
		return (0..cell_count).collect();
	}
	let strides: Vec<usize> = (1..cell_count)
		.filter(|stride| gcd(*stride, cell_count) == 1)
		.collect();
	let stride = strides[sweep % strides.len()];
	let offset = sweep.wrapping_mul(5) % cell_count;
	(0..cell_count)
		.map(|position| (position * stride + offset) % cell_count)
		.collect()
}

fn benchmark_speculation(
	k: usize,
	speculation: Option<SpeculationStatsData>,
) -> std::io::Result<SpeculationStatsData> {
	if k == 0 {
		return match speculation {
			None => Ok(SpeculationStatsData::default()),
			Some(_) => Err(std::io::Error::other(
				"speculation stats appeared in a k=0 baseline",
			)),
		};
	}
	let speculation = speculation
		.ok_or_else(|| std::io::Error::other("speculative run carried no per-call stats"))?;
	if speculation.drafted == 0 || speculation.rounds == 0 {
		return Err(std::io::Error::other(
			"speculative run completed without drafting and deciding a round",
		));
	}
	Ok(speculation)
}

async fn run_once(
	model: &emelex::CompletionModel,
	prompt: &str,
	mode: Mode,
	k: usize,
	max_tokens: usize,
	seed: u64,
) -> Result<RunRecord, Box<dyn std::error::Error>> {
	emelex::diag::reset_peak_memory()?;
	let mut params = serde_json::json!({
		"prompt_cache": false,
		"max_tokens": max_tokens,
		"enable_thinking": false,
		"speculative_tokens": k,
	});
	if mode == Mode::Sampled {
		params["temperature"] = 0.7.into();
		params["top_p"] = 0.9.into();
		params["seed"] = seed.into();
	}
	let request = model
		.completion_request(prompt)
		.additional_params(params)
		.build();
	let start = Instant::now();
	let mut stream = model.stream(request).await?;
	let mut first_token: Option<Instant> = None;
	while let Some(item) = stream.next().await {
		item?;
		if first_token.is_none() {
			first_token = Some(Instant::now());
		}
	}
	let end = Instant::now();
	let first =
		first_token.ok_or_else(|| std::io::Error::other("benchmark stream emitted no token"))?;
	let final_response = stream
		.response
		.clone()
		.ok_or_else(|| std::io::Error::other("drained benchmark stream had no final response"))?;
	let completion = final_response.usage.completion_tokens;
	if completion < 2 {
		return Err(std::io::Error::other(
			"benchmark completion needs at least two tokens for a decode rate",
		)
		.into());
	}
	let decode_secs = end.duration_since(first).as_secs_f64().max(1e-9);
	let speculation = benchmark_speculation(k, final_response.speculation)?;
	Ok(RunRecord {
		sweep: 0,
		position: 0,
		ttft_ms: first.duration_since(start).as_secs_f64() * 1e3,
		decode_tok_s: (completion.saturating_sub(1)) as f64 / decode_secs,
		completion_tokens: completion,
		drafted: speculation.drafted,
		rounds: speculation.rounds,
		accepted_by_depth: speculation.accepted_by_depth,
		peak_memory_bytes: emelex::diag::peak_memory()?,
		cache_memory_bytes: emelex::diag::cache_memory()?,
	})
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
	let Some(path) = std::env::var_os("EMELEX_TEST_MODEL") else {
		eprintln!("set EMELEX_TEST_MODEL to a model directory");
		std::process::exit(2);
	};
	let reps = env_usize("MTP_BENCH_REPS", 5)?;
	if reps < 5 {
		eprintln!(
			"MTP_BENCH_REPS={reps}: the registered gate requires at least 5 measured \
			 repetitions"
		);
		std::process::exit(2);
	}
	let warmup = env_usize("MTP_BENCH_WARMUP", 2)?;
	let total_sweeps = warmup.checked_add(reps).ok_or_else(|| {
		std::io::Error::other("MTP_BENCH_WARMUP + MTP_BENCH_REPS overflows usize")
	})?;
	let max_tokens = env_usize("MTP_BENCH_TOKENS", 512)?;
	if max_tokens < 2 {
		eprintln!("MTP_BENCH_TOKENS={max_tokens}: decode-rate runs require at least 2 tokens");
		std::process::exit(2);
	}

	let client = Client::from_path(path)?;
	if !client.supports_mtp() {
		return Err(std::io::Error::other(
			"benchmark model is not an MTP-certified Emelex checkpoint",
		)
		.into());
	}
	let model = client.model();
	eprintln!("model loaded; MTP certification verified");

	let long = long_prompt();
	let mut cells: Vec<Cell> = Vec::new();
	for &k in KS {
		for mode in [Mode::Greedy, Mode::Sampled] {
			cells.push(Cell {
				prompt_name: "short",
				mode,
				k,
			});
			cells.push(Cell {
				prompt_name: "long",
				mode,
				k,
			});
		}
	}
	let mut records: Vec<Vec<RunRecord>> = vec![Vec::new(); cells.len()];

	// Rep-major ordering runs every cell once per sweep. Sweep-varying
	// coprime strides and offsets reduce fixed position and adjacency bias.
	for rep in 0..total_sweeps {
		let seed = u64::try_from(rep)
			.ok()
			.and_then(|value| value.checked_add(1000))
			.ok_or_else(|| std::io::Error::other("benchmark seed exceeds u64"))?;
		for (position, idx) in balanced_cell_order(cells.len(), rep)
			.into_iter()
			.enumerate()
		{
			let cell = &cells[idx];
			let prompt: &str = match cell.prompt_name {
				"short" => SHORT_PROMPT,
				_ => &long,
			};
			let mut record = run_once(&model, prompt, cell.mode, cell.k, max_tokens, seed).await?;
			record.sweep = rep;
			record.position = position;
			eprintln!(
				"rep {rep} cell {idx} ({} {} k={}): {:.1} tok/s ttft {:.0}ms drafted \
				 {}",
				cell.prompt_name,
				match cell.mode {
					Mode::Greedy => "greedy",
					Mode::Sampled => "sampled",
				},
				cell.k,
				record.decode_tok_s,
				record.ttft_ms,
				record.drafted,
			);
			if rep >= warmup {
				records[idx].push(record);
			}
		}
	}

	let mut reports: Vec<CellReport> = Vec::new();
	for (idx, cell) in cells.iter().enumerate() {
		let runs = records[idx].clone();
		let mut rates: Vec<f64> = runs.iter().map(|r| r.decode_tok_s).collect();
		rates.sort_by(f64::total_cmp);
		let (lo, median, hi) = quartiles(&rates)?;
		let mut ttfts: Vec<f64> = runs.iter().map(|r| r.ttft_ms).collect();
		ttfts.sort_by(f64::total_cmp);
		let (_, ttft_median, _) = quartiles(&ttfts)?;
		reports.push(CellReport {
			prompt: cell.prompt_name,
			mode: match cell.mode {
				Mode::Greedy => "greedy",
				Mode::Sampled => "sampled",
			},
			k: cell.k,
			median_tok_s: median,
			iqr_lo: lo,
			iqr_hi: hi,
			median_ttft_ms: ttft_median,
			peak_memory_max: runs.iter().map(|r| r.peak_memory_bytes).max().unwrap_or(0),
			runs,
		});
	}

	println!("prompt  mode     k  median tok/s   IQR             TTFT ms");
	for report in &reports {
		println!(
			"{:<7} {:<8} {:<2} {:>10.1}   [{:>6.1},{:>6.1}]  {:>8.0}",
			report.prompt,
			report.mode,
			report.k,
			report.median_tok_s,
			report.iqr_lo,
			report.iqr_hi,
			report.median_ttft_ms,
		);
	}

	// Decision rule: pre-registered k* only, dense fixture, both modes.
	let mut pass = true;
	for mode in ["greedy", "sampled"] {
		for prompt in ["short", "long"] {
			let find = |k: usize| -> std::io::Result<&CellReport> {
				reports
					.iter()
					.find(|r| r.k == k && r.mode == mode && r.prompt == prompt)
					.ok_or_else(|| {
						std::io::Error::other(format!(
							"benchmark report is missing {mode}/{prompt}/k={k}"
						))
					})
			};
			let off = find(0)?;
			let starred = find(K_STAR)?;
			let gain = starred.median_tok_s / off.median_tok_s.max(1e-9);
			let disjoint = starred.iqr_lo > off.iqr_hi;
			let ok = gain > 1.05 && disjoint;
			pass &= ok;
			println!(
				"gate {mode}/{prompt}: k*={K_STAR} median {:.1} vs off {:.1} \
				 (x{gain:.3}), IQR disjoint: {disjoint} -> {}",
				starred.median_tok_s,
				off.median_tok_s,
				if ok { "PASS" } else { "FAIL" },
			);
		}
	}
	println!(
		"merge gate: {}",
		if pass {
			"PASS"
		} else {
			"FAIL (see plan outcomes)"
		}
	);

	std::fs::write(
		"mtp-bench-report.json",
		serde_json::to_vec_pretty(&BenchmarkReport {
			measured_repetitions: reps,
			warmup_sweeps: warmup,
			max_tokens,
			ordering: "rep-major, sweep-varying coprime strides and offsets",
			cells: &reports,
		})?,
	)?;
	eprintln!("wrote mtp-bench-report.json");
	if !pass {
		return Err(std::io::Error::other("MTP benchmark merge gate failed").into());
	}
	Ok(())
}

#[cfg(test)]
mod tests {
	#![allow(clippy::unwrap_used)]

	use super::*;

	#[test]
	fn speculation_stats_distinguish_baseline_and_active_cells() {
		assert_eq!(
			benchmark_speculation(0, None).unwrap(),
			SpeculationStatsData::default()
		);
		assert!(benchmark_speculation(0, Some(SpeculationStatsData::default())).is_err());
		assert!(benchmark_speculation(4, None).is_err());
		assert!(benchmark_speculation(4, Some(SpeculationStatsData::default())).is_err());
		let active = SpeculationStatsData {
			drafted: 4,
			rounds: 2,
			accepted_by_depth: vec![1],
		};
		assert_eq!(
			benchmark_speculation(4, Some(active.clone())).unwrap(),
			active
		);
	}

	#[test]
	fn balanced_order_is_a_permutation_and_changes_positions() {
		let first = balanced_cell_order(24, 0);
		let second = balanced_cell_order(24, 1);
		let mut sorted = first.clone();
		sorted.sort_unstable();
		assert_eq!(sorted, (0..24).collect::<Vec<_>>());
		assert_ne!(first, second);
		assert_ne!(
			first.windows(2).collect::<Vec<_>>(),
			second.windows(2).collect::<Vec<_>>()
		);
		for cell in 0..24 {
			assert_ne!(
				first.iter().position(|value| *value == cell),
				second.iter().position(|value| *value == cell)
			);
		}
	}
}
