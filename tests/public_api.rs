//! Downstream-style public-API pins for Rig response DTOs.
//!
//! This file compiles exactly like a downstream crate: it constructs the
//! response DTOs with exhaustive struct literals and verifies that MTP
//! accounting travels with the call it describes. No model is loaded.

// Test code: unwraps and panics are the assertion mechanism.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

use emelex::{Response, SpeculationStatsData, StreamingResponse, ToolCallData, UsageData};

/// Non-streaming MTP counters belong to the same raw response.
#[test]
fn response_carries_per_call_speculation() {
	let response = Response {
		text: "answer".to_string(),
		reasoning: None,
		tool_calls: vec![ToolCallData {
			id: "call-1".to_string(),
			name: "add".to_string(),
			arguments: serde_json::json!({"a": 1}),
		}],
		usage: UsageData {
			prompt_tokens: 3,
			cached_tokens: 0,
			completion_tokens: 2,
		},
		finish_reason: "stop".to_string(),
		speculation: Some(SpeculationStatsData {
			drafted: 3,
			rounds: 2,
			accepted_by_depth: vec![1],
		}),
	};
	assert_eq!(
		response.speculation.as_ref().map(|stats| stats.drafted),
		Some(3)
	);
}

/// The terminal streaming DTO carries counters for that stream.
#[test]
fn streaming_response_carries_per_call_speculation() {
	let response = StreamingResponse {
		usage: UsageData::default(),
		finish_reason: "stop".to_string(),
		speculation: Some(SpeculationStatsData {
			drafted: 2,
			rounds: 1,
			accepted_by_depth: vec![1],
		}),
	};
	assert_eq!(
		response.speculation.as_ref().map(|stats| stats.rounds),
		Some(1)
	);
}

/// `SpeculationStatsData` carries exactly `drafted: u64`, `rounds: u64`,
/// `accepted_by_depth: Vec<u64>`, with one-based depth buckets surviving a
/// serde round trip.
#[test]
fn speculation_stats_data_field_types_are_pinned() {
	let drafted: u64 = 5;
	let rounds: u64 = 3;
	let accepted_by_depth: Vec<u64> = vec![1, 0, 1];
	let stats = SpeculationStatsData {
		drafted,
		rounds,
		accepted_by_depth,
	};
	// rounds - sum(accepted_by_depth) counts full rejections.
	assert_eq!(
		stats.rounds - stats.accepted_by_depth.iter().sum::<u64>(),
		1
	);
	let json = serde_json::to_string(&stats).unwrap();
	let back: SpeculationStatsData = serde_json::from_str(&json).unwrap();
	assert_eq!(back, stats);
}
