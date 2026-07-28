//! Durable Session and workspace-Knowledge commands.

use std::{os::unix::fs::MetadataExt as _, path::Path, time::Duration};

use anyhow::{Context as _, bail};
use emelex::{
	Emelex,
	memory::{Knowledge, MaintenanceOptions, MemoryJobKind, MemoryStore, Session},
};

use super::{
	args::{KnowledgeCommand, MemoryCommand, SessionsCommand},
	output,
	style::{Palette, bytes},
};

/// Execute one durable-memory command.
pub(crate) async fn run(
	emelex: &Emelex,
	command: MemoryCommand,
	json: bool,
	palette: Palette,
	stderr_palette: Palette,
) -> anyhow::Result<()> {
	let store = emelex.memory().context("initialize durable memory")?;
	match command {
		MemoryCommand::Status => status(store, json, palette),
		MemoryCommand::Export {
			output: destination,
		} => output::export_stream(destination.as_deref(), |writer| {
			write_workspace_export(emelex, store, writer)
		}),
		MemoryCommand::Gc => maintain(emelex, store, json),
		MemoryCommand::Work { max_jobs } => {
			super::memory_worker::run(
				emelex,
				store,
				usize::from(max_jobs),
				json,
				palette,
				stderr_palette,
			)
			.await
		}
		MemoryCommand::Failures { limit } => failures(store, usize::from(limit), json, palette),
		MemoryCommand::Retry { job } => {
			store
				.retry_failed_job(job)
				.with_context(|| format!("retry failed memory job {job}"))?;
			if json {
				output::json_line(&serde_json::json!({"retried_memory_job": job}))
			} else {
				output::stdout_line(&format!("queued {job} for retry"))
			}
		}
		MemoryCommand::Sessions { command } => sessions(emelex, store, command, json, palette),
		MemoryCommand::Knowledge { command } => knowledge(emelex, store, command, json, palette),
	}
}

fn maintain(emelex: &Emelex, store: &MemoryStore, json: bool) -> anyhow::Result<()> {
	let days = u64::from(emelex.config().memory.retention_days);
	let age = Duration::from_secs(
		days.checked_mul(24 * 60 * 60)
			.context("memory retention duration overflow")?,
	);
	let mut options = MaintenanceOptions::default();
	options.retention.session_max_age = age;
	options.retention.knowledge_max_age = age;
	options.vacuum = true;
	let report = store.maintain(options).context("maintain durable memory")?;
	if json {
		output::json_line(&report)
	} else {
		output::stdout_line(&format!(
			"recovered {} claim(s); removed {} session(s), {} Knowledge entries, \
			 {} version(s), and {} asset(s){}",
			report
				.session_claims_recovered
				.saturating_add(report.compactions_recovered)
				.saturating_add(report.distillations_recovered),
			report.sessions_removed,
			report.knowledge_removed,
			report.versions_removed,
			report
				.assets
				.cataloged_files
				.saturating_add(report.assets.orphan_files),
			if report.wal_busy {
				"; WAL checkpoint remained busy"
			} else {
				""
			}
		))
	}
}

fn status(store: &MemoryStore, json: bool, palette: Palette) -> anyhow::Result<()> {
	let status = store.status().context("inspect durable memory")?;
	if json {
		return output::json_line(&status);
	}
	for line in [
		format!("database       {}", bytes(status.database_bytes)),
		format!("sessions       {}", status.sessions),
		format!("events         {}", status.events),
		format!("Knowledge      {}", status.knowledge),
		format!("tombstoned     {}", status.tombstoned_knowledge),
		format!(
			"compactions    {}",
			palette.cyan(&status.pending_compactions.to_string())
		),
		format!(
			"failed compact {}",
			palette.red(&status.failed_compactions.to_string())
		),
		format!("distillations  {}", status.pending_distillations),
		format!(
			"failed distill {}",
			palette.red(&status.failed_distillations.to_string())
		),
		format!(
			"assets          {} ({})",
			status.assets,
			bytes(status.asset_bytes)
		),
	] {
		output::stdout_line(&line)?;
	}
	Ok(())
}

fn failures(store: &MemoryStore, limit: usize, json: bool, palette: Palette) -> anyhow::Result<()> {
	let failures = store
		.failed_jobs(limit)
		.context("list failed durable-memory jobs")?;
	if json {
		return output::json_line(&failures);
	}
	for failure in failures {
		let kind = match failure.kind {
			MemoryJobKind::Compaction => "compaction",
			MemoryJobKind::Distillation => "distillation",
			_ => "unknown",
		};
		output::stdout_line(&format!(
			"{}  {kind:<12}  {} failure(s)  {}  {}",
			palette.red(&failure.id.to_string()),
			failure.failures,
			failure.failed_at,
			output::terminal_safe_inline(&failure.error)
		))?;
	}
	Ok(())
}

fn sessions(
	emelex: &Emelex,
	store: &MemoryStore,
	command: SessionsCommand,
	json: bool,
	palette: Palette,
) -> anyhow::Result<()> {
	match command {
		SessionsCommand::List { all, limit } => {
			let workspace = (!all).then_some(emelex.invocation_root());
			let page = store
				.sessions(workspace, None, limit)
				.context("list durable sessions")?;
			if json {
				output::json_line(&page)
			} else {
				for session in page.items {
					output::stdout_line(&format!(
						"{}  {}  {}",
						palette.cyan(&session.id.to_string()),
						output::terminal_safe_inline(
							session.title.as_deref().unwrap_or("untitled"),
						),
						session.updated_at
					))?;
				}
				Ok(())
			}
		}
		SessionsCommand::Show { session, all } => {
			let session_record = scoped_session(emelex, store, session, all)?;
			output::export_stream(None, |writer| {
				write_session_export(store, &session_record, writer)
			})
		}
		SessionsCommand::Export {
			session,
			all,
			output: destination,
		} => {
			let session_record = scoped_session(emelex, store, session, all)?;
			output::export_stream(destination.as_deref(), |writer| {
				write_session_export(store, &session_record, writer)
			})
		}
		SessionsCommand::Recover {
			session,
			all,
			accept_unknown_effects,
		} => {
			let session_record = scoped_session(emelex, store, session, all)?;
			let recovery_workspace = if all {
				session_record.workspace.as_path()
			} else {
				emelex.invocation_root()
			};
			let report = store
				.recover_interrupted_agent_turn(session, recovery_workspace, accept_unknown_effects)
				.with_context(|| format!("recover interrupted agent turn for session {session}"))?;
			if json {
				output::json_line(&serde_json::json!({
					"type": if report.interrupted_turn {
						"interrupted_agent_turn_recovered"
					} else {
						"interrupted_tool_batch_recovered"
					},
					"report": report,
				}))
			} else if report.interrupted_turn {
				output::stdout_line(&format!(
					"recovered interrupted agent turn {}",
					report.session_id
				))
			} else {
				output::stdout_line(&format!(
					"recovered {}: {} exact, {} uncertain, {} not executed; no tool was \
					 re-invoked",
					report.session_id,
					report.exact_results,
					report.uncertain_results,
					report.not_executed_results
				))
			}
		}
		SessionsCommand::Delete { session, all } => {
			scoped_session(emelex, store, session, all)?;
			store
				.delete_session(session)
				.with_context(|| format!("delete session {session}"))?;
			if json {
				output::json_line(&serde_json::json!({"deleted_session": session}))
			} else {
				output::stdout_line(&format!("deleted {session}"))
			}
		}
	}
}

fn knowledge(
	emelex: &Emelex,
	store: &MemoryStore,
	command: KnowledgeCommand,
	json: bool,
	palette: Palette,
) -> anyhow::Result<()> {
	let workspace = emelex.invocation_root();
	match command {
		KnowledgeCommand::List { limit } => {
			let page = store
				.knowledge_for_workspace(workspace, None, limit)
				.context("list workspace Knowledge")?;
			present_knowledge(&page.items, json, palette)
		}
		KnowledgeCommand::Search { query, limit } => {
			let items = store
				.search_knowledge(workspace, &query, limit)
				.context("search workspace Knowledge")?;
			present_knowledge(&items, json, palette)
		}
		KnowledgeCommand::Show { knowledge } => {
			let entry = workspace_knowledge(store, workspace, knowledge)?;
			if json {
				output::json_line(&entry)
			} else {
				output::export_json(&entry, None)
			}
		}
		KnowledgeCommand::History { knowledge, limit } => {
			workspace_knowledge(store, workspace, knowledge)?;
			let versions = store
				.knowledge_history(knowledge, None, limit)
				.context("load Knowledge history")?;
			if json {
				output::json_line(&versions)
			} else {
				output::export_json(&versions, None)
			}
		}
		KnowledgeCommand::Activate { knowledge, version } => {
			store
				.activate_knowledge(workspace, knowledge, version)
				.context("activate Knowledge version")?;
			mutation_result(json, "activated", knowledge)
		}
		KnowledgeCommand::Pin { knowledge } => {
			store
				.set_knowledge_pinned(workspace, knowledge, true)
				.context("pin Knowledge")?;
			mutation_result(json, "pinned", knowledge)
		}
		KnowledgeCommand::Unpin { knowledge } => {
			store
				.set_knowledge_pinned(workspace, knowledge, false)
				.context("unpin Knowledge")?;
			mutation_result(json, "unpinned", knowledge)
		}
		KnowledgeCommand::Forget { knowledge } => {
			store
				.delete_knowledge(workspace, knowledge)
				.context("forget Knowledge")?;
			mutation_result(json, "forgotten", knowledge)
		}
	}
}

fn scoped_session(
	emelex: &Emelex,
	store: &MemoryStore,
	id: uuid::Uuid,
	all_workspaces: bool,
) -> anyhow::Result<Session> {
	let session = store
		.session(id)
		.with_context(|| format!("load session {id}"))?;
	if !all_workspaces {
		let metadata = std::fs::metadata(emelex.invocation_root())
			.context("inspect current workspace identity")?;
		if metadata.dev() != session.workspace_identity.device()
			|| metadata.ino() != session.workspace_identity.inode()
		{
			bail!("Session {id} belongs to another workspace; pass --all to access it explicitly");
		}
	}
	Ok(session)
}

fn workspace_knowledge(
	store: &MemoryStore,
	workspace: &Path,
	id: uuid::Uuid,
) -> anyhow::Result<Knowledge> {
	let entry = store
		.knowledge(id)
		.with_context(|| format!("load Knowledge {id}"))?;
	let metadata = std::fs::metadata(workspace).context("inspect current workspace identity")?;
	if metadata.dev() != entry.workspace_identity.device()
		|| metadata.ino() != entry.workspace_identity.inode()
	{
		bail!("Knowledge {id} belongs to another workspace");
	}
	Ok(entry)
}

fn mutation_result(json: bool, action: &str, id: uuid::Uuid) -> anyhow::Result<()> {
	if json {
		output::json_line(&serde_json::json!({"action": action, "knowledge": id}))
	} else {
		output::stdout_line(&format!("{action} {id}"))
	}
}

fn present_knowledge(items: &[Knowledge], json: bool, palette: Palette) -> anyhow::Result<()> {
	if json {
		return output::json_line(&items);
	}
	for entry in items {
		output::stdout_line(&format!(
			"{}  v{}  {}{}",
			palette.cyan(&entry.id.to_string()),
			entry.active_version,
			if entry.pinned { "pinned  " } else { "" },
			output::terminal_safe_inline(&entry.key)
		))?;
	}
	Ok(())
}

fn write_workspace_export(
	emelex: &Emelex,
	store: &MemoryStore,
	writer: &mut dyn std::io::Write,
) -> anyhow::Result<()> {
	writer.write_all(b"{\"schema_version\":1,\"workspace\":")?;
	write_json(writer, emelex.invocation_root(), "workspace")?;
	writer.write_all(b",\"sessions\":[")?;
	write_export_sessions(emelex, store, writer)?;
	writer.write_all(b"],\"knowledge\":[")?;
	write_export_knowledge(emelex, store, writer)?;
	writer.write_all(b"]}")?;
	Ok(())
}

fn write_export_sessions(
	emelex: &Emelex,
	store: &MemoryStore,
	writer: &mut dyn std::io::Write,
) -> anyhow::Result<()> {
	let mut cursor = None;
	let mut first_session = true;
	loop {
		let page = store
			.sessions(Some(emelex.invocation_root()), cursor.as_ref(), 500)
			.context("page workspace sessions for export")?;
		for session in page.items {
			write_separator(writer, &mut first_session)?;
			write_session_export(store, &session, writer)?;
		}
		cursor = page.next;
		if cursor.is_none() {
			return Ok(());
		}
	}
}

fn write_session_export(
	store: &MemoryStore,
	session: &Session,
	writer: &mut dyn std::io::Write,
) -> anyhow::Result<()> {
	writer.write_all(b"{\"session\":")?;
	write_json(writer, session, "Session")?;
	writer.write_all(b",\"snapshot\":")?;
	write_json(
		writer,
		&store
			.session_snapshot(session.id)
			.with_context(|| format!("load snapshot for session {}", session.id))?,
		"Session snapshot",
	)?;
	writer.write_all(b",\"events\":[")?;
	write_export_events(store, session.id, writer)?;
	writer.write_all(b"],\"assets\":")?;
	write_json(
		writer,
		&store
			.session_assets(session.id)
			.with_context(|| format!("load assets for session {}", session.id))?,
		"Session assets",
	)?;
	writer.write_all(b"}")?;
	Ok(())
}

fn write_export_events(
	store: &MemoryStore,
	session: uuid::Uuid,
	writer: &mut dyn std::io::Write,
) -> anyhow::Result<()> {
	let mut after = 0_u64;
	let mut first = true;
	loop {
		let events = store
			.events(session, after, 100)
			.with_context(|| format!("page events for session {session}"))?;
		let Some(last) = events.last() else {
			return Ok(());
		};
		after = last.sequence;
		for event in events {
			write_separator(writer, &mut first)?;
			write_json(writer, &event, "Session event")?;
		}
	}
}

fn write_export_knowledge(
	emelex: &Emelex,
	store: &MemoryStore,
	writer: &mut dyn std::io::Write,
) -> anyhow::Result<()> {
	let mut cursor = None;
	let mut first = true;
	loop {
		let page = store
			.knowledge_for_workspace(emelex.invocation_root(), cursor.as_ref(), 100)
			.context("page workspace Knowledge for export")?;
		for knowledge in page.items {
			write_separator(writer, &mut first)?;
			writer.write_all(b"{\"knowledge\":")?;
			write_json(writer, &knowledge, "Knowledge")?;
			writer.write_all(b",\"versions\":[")?;
			write_knowledge_versions(store, knowledge.id, writer)?;
			writer.write_all(b"]}")?;
		}
		cursor = page.next;
		if cursor.is_none() {
			return Ok(());
		}
	}
}

fn write_knowledge_versions(
	store: &MemoryStore,
	knowledge: uuid::Uuid,
	writer: &mut dyn std::io::Write,
) -> anyhow::Result<()> {
	let mut before = None;
	let mut first = true;
	loop {
		let versions = store
			.knowledge_history(knowledge, before, 100)
			.with_context(|| format!("page history for Knowledge {knowledge}"))?;
		let Some(last) = versions.last() else {
			return Ok(());
		};
		before = Some(last.version);
		for version in versions {
			write_separator(writer, &mut first)?;
			write_json(writer, &version, "Knowledge version")?;
		}
	}
}

fn write_separator(writer: &mut dyn std::io::Write, first: &mut bool) -> anyhow::Result<()> {
	if *first {
		*first = false;
	} else {
		writer.write_all(b",")?;
	}
	Ok(())
}

fn write_json(
	writer: &mut dyn std::io::Write,
	value: &(impl serde::Serialize + ?Sized),
	label: &str,
) -> anyhow::Result<()> {
	serde_json::to_writer(&mut *writer, value)
		.with_context(|| format!("encode {label} for memory export"))
}
