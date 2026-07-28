//! Apple Silicon local inference, model discovery, and agent runtime.

#![cfg_attr(
	test,
	allow(
		clippy::expect_used,
		clippy::float_cmp,
		clippy::panic,
		clippy::unwrap_used,
		reason = "unit tests use fail-fast fixture setup and exact-value assertions"
	)
)]

// Vendored mlex v0.1.3 inference runtime (MIT, see ATTRIBUTION.md). The
// engine keeps upstream's coding style and invariants; workspace lint
// policy applies only to hand-written provider code.
#[allow(
	dead_code,
	clippy::complexity,
	clippy::nursery,
	clippy::pedantic,
	clippy::perf,
	clippy::style,
	reason = "vendored engine retains upstream internal diagnostics and alternate model paths"
)]
#[deny(clippy::correctness, clippy::suspicious)]
pub(crate) mod engine;

/// Native agent loop, approvals, and bounded workspace tools.
pub mod agent;
mod artifact;
mod client;
/// Strict global and project configuration.
pub mod config;
#[cfg(feature = "rig")]
mod convert;
mod error;
/// Native generation API.
pub mod generation;
/// Emelex-owned storage root and layout.
pub mod home;
/// Hugging Face discovery and bounded downloads.
pub mod hub;
mod json;
/// Durable Sessions, compaction, and workspace Knowledge.
pub mod memory;
/// Model identities, capabilities, compatibility, and optional Rig adapter.
pub mod model;
/// Immutable installed-model lifecycle.
pub mod models;
/// Embedded MLX runtime initialization.
pub mod runtime;
mod toolkit;

/// emelex patch (not upstream): benchmark-only diagnostics, see `diag`.
#[cfg(feature = "bench")]
pub mod diag;

pub use client::{Client, ClientBuilder};
pub use error::Error;
pub use toolkit::{Emelex, EmelexBuilder, ToolkitError};
#[cfg(feature = "rig")]
pub use {
	client::ReasoningExt,
	model::{
		CompletionModel, Response, SpeculationStatsData, StreamingResponse, ToolCallData, UsageData,
	},
};
