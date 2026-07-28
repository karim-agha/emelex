//! Stable command-line grammar.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};
use emelex::model::{HubModelId, ModelRef, ModelSnapshotId, TraitFilter};
use uuid::Uuid;

use super::style::ColorMode;

/// Emelex command line.
#[derive(Debug, Parser)]
#[command(name = "emelex", version, about)]
pub(crate) struct Cli {
	/// Override Emelex storage root.
	#[arg(long, global = true)]
	pub(crate) home: Option<PathBuf>,
	/// Override invocation root for workspace state and project configuration.
	#[arg(
		short = 'C',
		long = "directory",
		visible_alias = "root",
		value_name = "PATH",
		global = true
	)]
	pub(crate) directory: Option<PathBuf>,
	/// Ignore nearest Git-root `.emelex.toml`.
	#[arg(long, global = true)]
	pub(crate) no_project_config: bool,
	/// Emit machine-readable JSON or newline-delimited events.
	#[arg(long, global = true)]
	pub(crate) json: bool,
	/// Terminal color policy.
	#[arg(long, value_enum, default_value_t, global = true)]
	pub(crate) color: ColorMode,
	#[command(subcommand)]
	pub(crate) command: Command,
}

/// Top-level command.
#[derive(Debug, Subcommand)]
pub(crate) enum Command {
	/// Start a durable interactive agent session in this directory.
	Chat(ChatArgs),
	/// Resume a durable agent session.
	Resume(ResumeArgs),
	/// Run one non-interactive inference request.
	Generate(GenerateArgs),
	/// Explore or download visible Hugging Face models.
	Hub {
		#[command(subcommand)]
		command: HubCommand,
	},
	/// Manage Emelex-owned immutable model snapshots.
	Models {
		#[command(subcommand)]
		command: ModelsCommand,
	},
	/// Inspect and maintain durable sessions and Knowledge.
	Memory {
		#[command(subcommand)]
		command: MemoryCommand,
	},
	/// Validate platform, home, native runtime, and optional models.
	Doctor(DoctorArgs),
}

/// Interactive chat options.
#[derive(Debug, Clone, Args)]
#[allow(
	clippy::struct_excessive_bools,
	reason = "these booleans are independent command-line switches"
)]
pub(crate) struct ChatArgs {
	/// Stable installed model reference.
	#[arg(long)]
	pub(crate) model: Option<ModelRef>,
	/// Resume with `--resume=SESSION`; bare `--resume` chooses a recent session.
	#[arg(
		long,
		value_name = "SESSION",
		num_args = 0..=1,
		require_equals = true,
		default_missing_value = "recent"
	)]
	pub(crate) resume: Option<ResumeTarget>,
	/// Additional generic system instruction for this session.
	#[arg(long)]
	pub(crate) system: Option<String>,
	/// Maximum generated tokens per reply.
	#[arg(long)]
	pub(crate) max_tokens: Option<usize>,
	/// Sampling temperature.
	#[arg(long)]
	pub(crate) temperature: Option<f32>,
	/// Thinking override; auto keeps the resolved configuration.
	#[arg(long, value_enum)]
	pub(crate) thinking: Option<ThinkingArg>,
	/// Disable all file and shell tools.
	#[arg(long)]
	pub(crate) no_tools: bool,
	/// Disable HTTP fetch tools.
	#[arg(long, conflicts_with = "with_web_search")]
	pub(crate) no_web: bool,
	/// Add approval-gated web search through `DuckDuckGo`'s HTML endpoint.
	#[arg(long)]
	pub(crate) with_web_search: bool,
	/// Permit every protected tool invocation without prompting.
	#[arg(long)]
	pub(crate) approve_all: bool,
	/// Optional first turn; non-interactive mode reads UTF-8 stdin when absent.
	pub(crate) prompt: Option<String>,
}

/// Top-level resume alias.
#[derive(Debug, Clone, Args)]
pub(crate) struct ResumeArgs {
	/// Session identity; omit to choose a recent workspace session.
	#[arg(long, value_name = "SESSION")]
	pub(crate) session: Option<Uuid>,
	/// Permit every protected tool invocation without prompting.
	#[arg(long)]
	pub(crate) approve_all: bool,
	/// Optional resumed turn; non-interactive mode reads UTF-8 stdin when absent.
	pub(crate) prompt: Option<String>,
}

/// `chat --resume` target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResumeTarget {
	Recent,
	Session(Uuid),
}

impl std::str::FromStr for ResumeTarget {
	type Err = String;

	fn from_str(value: &str) -> Result<Self, Self::Err> {
		if value.eq_ignore_ascii_case("recent") {
			Ok(Self::Recent)
		} else {
			value
				.parse::<Uuid>()
				.map(Self::Session)
				.map_err(|error| error.to_string())
		}
	}
}

/// One-shot generation options.
#[derive(Debug, Clone, Args)]
pub(crate) struct GenerateArgs {
	/// Prompt text; when absent, read UTF-8 stdin.
	pub(crate) prompt: Option<String>,
	/// Stable installed model reference.
	#[arg(long)]
	pub(crate) model: Option<ModelRef>,
	/// Run the native tool loop instead of raw generation.
	#[arg(long)]
	pub(crate) agent: bool,
	/// Attach image or audio bytes to the user message.
	#[arg(long = "attach", value_name = "PATH")]
	pub(crate) attachments: Vec<PathBuf>,
	/// Maximum generated tokens.
	#[arg(long)]
	pub(crate) max_tokens: Option<usize>,
	/// Sampling temperature.
	#[arg(long)]
	pub(crate) temperature: Option<f32>,
	/// Thinking override: auto uses the client default, which is safely off
	/// unless explicitly enabled.
	#[arg(long, value_enum)]
	pub(crate) thinking: Option<ThinkingArg>,
	/// Permit every protected agent tool invocation.
	#[arg(long, requires = "agent")]
	pub(crate) approve_all: bool,
}

/// Thinking-mode CLI value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum ThinkingArg {
	Auto,
	On,
	Off,
}

/// Hugging Face Hub operation.
#[derive(Debug, Subcommand)]
pub(crate) enum HubCommand {
	/// List filters supported during remote catalog discovery.
	Capabilities,
	/// Search MLX Hub ranking for models compatible with this machine.
	Search {
		/// Optional Hub search text; MLX and local-fit filters always apply.
		query: Option<String>,
		/// Required remote trait; repeat for conjunction. Run `hub capabilities` for choices.
		#[arg(long = "require")]
		require: Vec<TraitFilter>,
		/// Opaque next-page cursor.
		#[arg(long)]
		cursor: Option<String>,
		/// Print grouped candidate diagnostics instead of only their count.
		#[arg(long)]
		verbose: bool,
	},
	/// Inspect one visible repository and its inferred traits.
	Inspect {
		/// Hugging Face repository visible anonymously or to `HF_TOKEN`.
		model: HubModelId,
		/// Print every compatibility diagnostic instead of the bounded summary.
		#[arg(long)]
		verbose: bool,
	},
	/// Download, verify, runtime-probe, and publish one immutable snapshot.
	Download {
		/// Hugging Face repository visible anonymously or to `HF_TOKEN`.
		model: HubModelId,
	},
}

/// Managed local model operation.
#[derive(Debug, Subcommand)]
pub(crate) enum ModelsCommand {
	/// List installed immutable snapshots.
	List,
	/// Copy and verify a local checkpoint.
	Import {
		/// Stable local name, addressed later as `local:<name>`.
		name: String,
		/// Checkpoint directory.
		path: PathBuf,
	},
	/// Show, set, or clear the global default model.
	Default {
		/// Installed stable model reference.
		model: Option<ModelRef>,
		/// Clear the configured default.
		#[arg(long, conflicts_with = "model")]
		clear: bool,
	},
	/// Download the current Hub revision for one or all installed Hub models.
	Update {
		/// Stable Hub reference; omit to update all.
		model: Option<ModelRef>,
	},
	/// Move one installed snapshot to recoverable quarantine.
	Remove {
		/// Exact immutable snapshot ID printed by `emelex models list`.
		model: ModelSnapshotId,
	},
	/// Rehash and runtime-probe one or all installed snapshots.
	Verify {
		/// Stable installed model reference; omit to verify all.
		model: Option<ModelRef>,
	},
	/// Permanently delete old quarantined snapshots.
	Gc {
		/// Minimum quarantine age.
		#[arg(long, default_value_t = 7)]
		older_than_days: u64,
	},
	/// Print the selected immutable snapshot directory.
	Path {
		/// Stable installed model reference.
		model: ModelRef,
	},
}

/// Durable-memory operation.
#[derive(Debug, Subcommand)]
pub(crate) enum MemoryCommand {
	/// Show `SQLite`, Session, event, Knowledge, and queue counts.
	Status,
	/// Stream this workspace's Sessions, events, assets, and active Knowledge.
	Export {
		/// Output file; stdout when absent.
		#[arg(long)]
		output: Option<PathBuf>,
	},
	/// Apply configured retention and compact `SQLite`.
	Gc,
	/// Process bounded pending compaction and Knowledge-distillation jobs.
	Work {
		/// Maximum jobs claimed in this run.
		#[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u16).range(1..=32))]
		max_jobs: u16,
	},
	/// Inspect terminal compaction and distillation failures.
	Failures {
		/// Maximum newest failures to print.
		#[arg(long, default_value_t = 50, value_parser = clap::value_parser!(u16).range(1..=500))]
		limit: u16,
	},
	/// Reset one terminal worker job for immediate retry.
	Retry {
		/// Durable job identity printed by `emelex memory failures`.
		job: Uuid,
	},
	/// Inspect or delete Sessions.
	Sessions {
		#[command(subcommand)]
		command: SessionsCommand,
	},
	/// Inspect or curate workspace Knowledge.
	Knowledge {
		#[command(subcommand)]
		command: KnowledgeCommand,
	},
}

/// Session-memory operation.
#[derive(Debug, Subcommand)]
pub(crate) enum SessionsCommand {
	/// List recent sessions.
	List {
		/// Include sessions from every workspace.
		#[arg(long)]
		all: bool,
		/// Maximum rows.
		#[arg(long, default_value_t = 50)]
		limit: usize,
	},
	/// Stream metadata and complete event history.
	Show {
		session: Uuid,
		/// Allow a Session belonging to another workspace.
		#[arg(long)]
		all: bool,
	},
	/// Stream one Session as JSON.
	Export {
		session: Uuid,
		/// Allow a Session belonging to another workspace.
		#[arg(long)]
		all: bool,
		/// Output file; stdout when absent.
		#[arg(long)]
		output: Option<PathBuf>,
	},
	/// Reconcile an interrupted agent turn without invoking tools again.
	Recover {
		session: Uuid,
		/// Allow a Session belonging to another workspace.
		#[arg(long)]
		all: bool,
		/// Accept recovery when a started tool's side effects are unknown.
		#[arg(long)]
		accept_unknown_effects: bool,
	},
	/// Delete one session and its unreferenced assets.
	Delete {
		session: Uuid,
		/// Allow a Session belonging to another workspace.
		#[arg(long)]
		all: bool,
	},
}

/// Workspace-Knowledge operation.
#[derive(Debug, Subcommand)]
pub(crate) enum KnowledgeCommand {
	/// List active workspace Knowledge.
	List {
		/// Maximum rows.
		#[arg(long, default_value_t = 50)]
		limit: usize,
	},
	/// Search active workspace Knowledge.
	Search {
		query: String,
		/// Maximum rows.
		#[arg(long, default_value_t = 20)]
		limit: usize,
	},
	/// Show one active Knowledge entry.
	Show { knowledge: Uuid },
	/// Show immutable versions of one Knowledge entry.
	History {
		knowledge: Uuid,
		#[arg(long, default_value_t = 100)]
		limit: usize,
	},
	/// Select one immutable Knowledge version.
	Activate { knowledge: Uuid, version: u32 },
	/// Exempt one Knowledge entry from retention.
	Pin { knowledge: Uuid },
	/// Restore normal retention for one Knowledge entry.
	Unpin { knowledge: Uuid },
	/// Hide one Knowledge entry; configured retention later purges its versions.
	Forget { knowledge: Uuid },
}

/// Doctor options.
#[derive(Debug, Clone, Copy, Args)]
pub(crate) struct DoctorArgs {
	/// Rehash and runtime-probe all installed models.
	#[arg(long)]
	pub(crate) models: bool,
}

#[cfg(test)]
mod tests {
	use clap::Parser as _;

	use super::*;

	#[test]
	fn resume_alias_and_optional_chat_target_parse() {
		let top = Cli::try_parse_from(["emelex", "resume"]).expect("interactive recent resume");
		assert!(matches!(
			top.command,
			Command::Resume(ResumeArgs {
				session: None,
				prompt: None,
				..
			})
		));
		let session = Uuid::now_v7();
		let explicit = Cli::try_parse_from(["emelex", "resume", "--session", &session.to_string()])
			.expect("interactive explicit resume");
		assert!(matches!(
			explicit.command,
			Command::Resume(ResumeArgs {
				session: Some(parsed),
				prompt: None,
				..
			}) if parsed == session
		));
		let recent_prompted = Cli::try_parse_from(["emelex", "--json", "resume", "continue"])
			.expect("non-interactive recent resume with positional prompt");
		assert!(matches!(
			recent_prompted.command,
			Command::Resume(ResumeArgs {
				session: None,
				prompt: Some(ref prompt),
				..
			}) if prompt == "continue"
		));
		let recent_stdin = Cli::try_parse_from(["emelex", "--json", "resume"])
			.expect("non-interactive recent resume with stdin resolved after parsing");
		assert!(matches!(
			recent_stdin.command,
			Command::Resume(ResumeArgs {
				session: None,
				prompt: None,
				..
			})
		));
		let explicit_prompted = Cli::try_parse_from([
			"emelex",
			"--json",
			"resume",
			"continue",
			"--session",
			&session.to_string(),
		])
		.expect("non-interactive explicit resume with positional prompt");
		assert!(matches!(
			explicit_prompted.command,
			Command::Resume(ResumeArgs {
				session: Some(parsed),
				prompt: Some(ref prompt),
				..
			}) if parsed == session && prompt == "continue"
		));

		let chat = Cli::try_parse_from(["emelex", "chat", "--resume"]).expect("chat resume");
		assert!(matches!(
			chat.command,
			Command::Chat(ChatArgs {
				resume: Some(ResumeTarget::Recent),
				..
			})
		));

		let explicit_chat = Cli::try_parse_from(["emelex", "chat", &format!("--resume={session}")])
			.expect("chat explicit resume");
		assert!(matches!(
			explicit_chat.command,
			Command::Chat(ChatArgs {
				resume: Some(ResumeTarget::Session(parsed)),
				prompt: None,
				..
			}) if parsed == session
		));

		let chat =
			Cli::try_parse_from(["emelex", "chat", "--resume", "hello"]).expect("resume prompt");
		assert!(matches!(
			chat.command,
			Command::Chat(ChatArgs {
				resume: Some(ResumeTarget::Recent),
				prompt: Some(ref prompt),
				..
			}) if prompt == "hello"
		));
	}

	#[test]
	fn model_commands_keep_hub_and_local_lifecycle_separate() {
		assert!(Cli::try_parse_from(["emelex", "hub", "search", "qwen"]).is_ok());
		assert!(Cli::try_parse_from(["emelex", "hub", "inspect", "gpt2"]).is_ok());
		assert!(
			Cli::try_parse_from(["emelex", "models", "import", "work", "/tmp/checkpoint"]).is_ok()
		);
		assert!(Cli::try_parse_from(["emelex", "models", "download", "owner/repo"]).is_err());
	}

	#[test]
	fn global_options_work_after_subcommands() {
		let cli = Cli::try_parse_from([
			"emelex",
			"hub",
			"search",
			"qwen",
			"--json",
			"--home",
			"/tmp/emelex-test",
		])
		.expect("global flags");
		assert!(cli.json);
		assert_eq!(cli.home, Some(PathBuf::from("/tmp/emelex-test")));
	}

	#[test]
	fn parser_leaves_environment_home_resolution_to_library() {
		let cli = Cli::try_parse_from(["emelex", "doctor"]).expect("doctor command");
		assert_eq!(cli.home, None);
	}

	#[test]
	fn chat_parses_directory_sampling_and_generic_search_controls() {
		let cli = Cli::try_parse_from([
			"emelex",
			"chat",
			"--root",
			"/tmp/work",
			"--max-tokens",
			"512",
			"--temperature",
			"0.7",
			"--thinking",
			"on",
			"--with-web-search",
		])
		.expect("chat compatibility controls");
		assert_eq!(cli.directory, Some(PathBuf::from("/tmp/work")));
		assert!(matches!(
			cli.command,
			Command::Chat(ChatArgs {
				max_tokens: Some(512),
				temperature: Some(temperature),
				thinking: Some(ThinkingArg::On),
				with_web_search: true,
				..
			}) if (temperature - 0.7).abs() < f32::EPSILON
		));
		assert!(Cli::try_parse_from(["emelex", "chat", "--no-web", "--with-web-search",]).is_err());
	}

	#[test]
	fn memory_failure_inspection_and_retry_commands_parse() {
		let job = Uuid::now_v7();
		assert!(Cli::try_parse_from(["emelex", "memory", "failures", "--limit", "10"]).is_ok());
		assert!(Cli::try_parse_from(["emelex", "memory", "retry", &job.to_string()]).is_ok());
	}
}
