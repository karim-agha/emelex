//! Descriptor-anchored workspace tools.

use std::{
	collections::VecDeque,
	ffi::{CStr, CString, OsStr, OsString},
	fs::{File, OpenOptions},
	io::{Read, Seek, SeekFrom, Write},
	mem::MaybeUninit,
	os::{
		fd::{AsRawFd, FromRawFd},
		unix::{
			ffi::{OsStrExt, OsStringExt},
			fs::{MetadataExt, OpenOptionsExt},
			process::CommandExt as _,
		},
	},
	path::{Component, Path, PathBuf},
	process::Stdio,
	sync::{
		Arc, Condvar, Mutex,
		atomic::{AtomicU64, Ordering},
	},
	time::Duration,
};

use async_trait::async_trait;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest as _, Sha256};
use tokio::io::{AsyncRead, AsyncReadExt};

use super::{
	AgentTool, ApprovalRequirement, BoundedJsonError, ToolContext, ToolError, ToolOutput,
	serialize_json_pretty_bounded,
};
use crate::generation::ToolDefinition;

const MAX_FILE_BYTES: usize = 1024 * 1024;
const MAX_LIST_ENTRIES: usize = 2_000;
const MAX_FIND_RESULTS: usize = 2_000;
const MAX_GREP_MATCHES: usize = 500;
const MAX_WALK_ENTRIES: usize = 20_000;
const MAX_WALK_DEPTH: usize = 32;
const MAX_GREP_FILES: usize = 4_096;
const MAX_GREP_TOTAL_BYTES: usize = 8 * 1024 * 1024;
const MAX_SENSITIVE_SKIP_PATHS: usize = 32;
const MAX_FIND_PATTERN_CHARS: usize = 256;
const MAX_GREP_QUERY_CHARS: usize = 4_096;
const MAX_SHELL_COMMAND_BYTES: usize = 64 * 1024;
/// Hard safety ceiling for one host-shell command.
pub const MAX_SHELL_TIMEOUT_SECONDS: u64 = 20 * 60;
const DEFAULT_SHELL_TIMEOUT_SECONDS: u64 = 30;
const FALLBACK_SHELL_PATH: &str = "/usr/bin:/bin:/usr/sbin:/sbin";
const MAX_SHELL_PATH_ENTRIES: usize = 128;
const MAX_SHELL_PATH_BYTES: usize = 16 * 1024;
/// Hard ceiling accepted by [`shell_tool`] for combined stdout/stderr capture.
pub const MAX_SHELL_OUTPUT_BYTES: usize = 512 * 1024;
const DEFAULT_SHELL_OUTPUT_BYTES: usize = 128 * 1024;
const IO_DRAIN_TIMEOUT: Duration = Duration::from_secs(2);

/// Descriptor-anchored workspace operation failure.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum WorkspaceError {
	/// A model-provided path violated lexical or containment rules.
	#[error("invalid workspace path {path:?}: {reason}")]
	Path {
		/// Supplied path.
		path: PathBuf,
		/// Rejected invariant.
		reason: String,
	},
	/// One descriptor-based filesystem operation failed.
	#[error("workspace {operation} failed for {path:?}: {source}")]
	Io {
		/// Stable operation name.
		operation: &'static str,
		/// Logical path.
		path: PathBuf,
		/// Underlying OS failure.
		#[source]
		source: std::io::Error,
	},
	/// Bounded traversal or output ceiling was exceeded.
	#[error("workspace {resource} exceeded its limit of {limit}")]
	Limit {
		/// Bounded resource.
		resource: &'static str,
		/// Configured ceiling.
		limit: usize,
	},
	/// Cooperative agent cancellation was requested.
	#[error("workspace operation cancelled")]
	Cancelled,
}

pub(super) struct WorkspaceRoot {
	path: PathBuf,
	directory: File,
	device: u64,
	inode: u64,
}

impl WorkspaceRoot {
	pub(super) fn open(path: &Path) -> Result<Self, WorkspaceError> {
		let canonical = std::fs::canonicalize(path).map_err(|source| WorkspaceError::Io {
			operation: "canonicalize root",
			path: path.to_path_buf(),
			source,
		})?;
		let expected = std::fs::metadata(&canonical).map_err(|source| WorkspaceError::Io {
			operation: "inspect root",
			path: canonical.clone(),
			source,
		})?;
		if !expected.is_dir() {
			return Err(WorkspaceError::Path {
				path: canonical,
				reason: "workspace root is not a directory".to_string(),
			});
		}
		let directory = open_directory_path(&canonical).map_err(|source| WorkspaceError::Io {
			operation: "open root",
			path: canonical.clone(),
			source,
		})?;
		let actual = directory.metadata().map_err(|source| WorkspaceError::Io {
			operation: "inspect root descriptor",
			path: canonical.clone(),
			source,
		})?;
		if expected.dev() != actual.dev() || expected.ino() != actual.ino() {
			return Err(WorkspaceError::Path {
				path: canonical,
				reason: "workspace root changed while it was opened".to_string(),
			});
		}
		Ok(Self {
			path: canonical,
			directory,
			device: actual.dev(),
			inode: actual.ino(),
		})
	}

	pub(super) fn path(&self) -> &Path {
		&self.path
	}

	pub(super) const fn identity(&self) -> (u64, u64) {
		(self.device, self.inode)
	}

	fn anchor(&self, input: &str, approved: bool) -> Result<AnchoredPath, WorkspaceError> {
		if input.as_bytes().contains(&0) {
			return Err(path_error(input, "paths cannot contain NUL"));
		}
		let supplied = Path::new(input);
		let (base, relative, display) = if supplied.is_absolute() {
			if let Ok(relative) = supplied.strip_prefix(&self.path) {
				(
					self.directory
						.try_clone()
						.map_err(|source| WorkspaceError::Io {
							operation: "duplicate root descriptor",
							path: self.path.clone(),
							source,
						})?,
					relative,
					supplied.to_path_buf(),
				)
			} else {
				if !approved {
					return Err(path_error(
						input,
						"absolute paths outside the workspace require approval",
					));
				}
				(
					open_directory_path(Path::new("/")).map_err(|source| WorkspaceError::Io {
						operation: "open filesystem root",
						path: PathBuf::from("/"),
						source,
					})?,
					supplied
						.strip_prefix("/")
						.map_err(|_| path_error(input, "invalid absolute path"))?,
					supplied.to_path_buf(),
				)
			}
		} else {
			(
				self.directory
					.try_clone()
					.map_err(|source| WorkspaceError::Io {
						operation: "duplicate root descriptor",
						path: self.path.clone(),
						source,
					})?,
				supplied,
				self.path.join(supplied),
			)
		};
		let components = lexical_components(relative, supplied)?;
		Ok(AnchoredPath {
			base,
			components,
			display,
		})
	}

	fn open_read(&self, input: &str, approved: bool) -> Result<(File, PathBuf), WorkspaceError> {
		let anchored = self.anchor(input, approved)?;
		let file = open_components(
			&anchored.base,
			&anchored.components,
			libc::O_RDONLY | libc::O_NONBLOCK,
			0,
		)
		.map_err(|source| WorkspaceError::Io {
			operation: "open file without following links",
			path: anchored.display.clone(),
			source,
		})?;
		Ok((file, anchored.display))
	}

	fn open_directory(
		&self,
		input: &str,
		approved: bool,
	) -> Result<(File, PathBuf), WorkspaceError> {
		let anchored = self.anchor(input, approved)?;
		let file = open_components(
			&anchored.base,
			&anchored.components,
			libc::O_RDONLY | libc::O_DIRECTORY,
			0,
		)
		.map_err(|source| WorkspaceError::Io {
			operation: "open directory without following links",
			path: anchored.display.clone(),
			source,
		})?;
		Ok((file, anchored.display))
	}

	fn mutation_target(
		&self,
		input: &str,
		approved: bool,
	) -> Result<MutationTarget, WorkspaceError> {
		let anchored = self.anchor(input, approved)?;
		if anchored.components.is_empty() {
			return Err(path_error(input, "cannot write a directory"));
		}
		let Some((name, parents)) = anchored.components.split_last() else {
			return Err(path_error(input, "mutation target is empty"));
		};
		let directory = open_components(
			&anchored.base,
			parents,
			libc::O_RDONLY | libc::O_DIRECTORY,
			0,
		)
		.map_err(|source| WorkspaceError::Io {
			operation: "open mutation parent without following links",
			path: anchored.display.clone(),
			source,
		})?;
		Ok(MutationTarget {
			directory,
			name: name.clone(),
			display: anchored.display,
		})
	}
}

struct AnchoredPath {
	base: File,
	components: Vec<OsString>,
	display: PathBuf,
}

struct MutationTarget {
	directory: File,
	name: OsString,
	display: PathBuf,
}

fn lexical_components(path: &Path, supplied: &Path) -> Result<Vec<OsString>, WorkspaceError> {
	let mut components = Vec::new();
	for component in path.components() {
		match component {
			Component::Normal(value) => components.push(value.to_os_string()),
			Component::CurDir => {}
			Component::ParentDir => {
				return Err(WorkspaceError::Path {
					path: supplied.to_path_buf(),
					reason: "parent traversal is not permitted".to_string(),
				});
			}
			Component::RootDir | Component::Prefix(_) => {
				return Err(WorkspaceError::Path {
					path: supplied.to_path_buf(),
					reason: "unexpected rooted component".to_string(),
				});
			}
		}
	}
	Ok(components)
}

fn open_directory_path(path: &Path) -> std::io::Result<File> {
	OpenOptions::new()
		.read(true)
		.custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
		.open(path)
}

fn open_components(
	base: &File,
	components: &[OsString],
	final_flags: i32,
	mode: libc::mode_t,
) -> std::io::Result<File> {
	let mut current = base.try_clone()?;
	if components.is_empty() {
		return Ok(current);
	}
	for (index, component) in components.iter().enumerate() {
		let is_last = index + 1 == components.len();
		let flags = if is_last {
			final_flags | libc::O_NOFOLLOW | libc::O_CLOEXEC
		} else {
			libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC
		};
		current = openat(current.as_raw_fd(), component, flags, mode)?;
	}
	Ok(current)
}

fn openat(
	directory: libc::c_int,
	name: &OsStr,
	flags: libc::c_int,
	mode: libc::mode_t,
) -> std::io::Result<File> {
	let name = CString::new(name.as_bytes())
		.map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidInput))?;
	// SAFETY: `directory` is an open directory descriptor, `name` is a
	// NUL-terminated single path component, and ownership of a successful
	// descriptor is transferred exactly once into `File`.
	let descriptor =
		unsafe { libc::openat(directory, name.as_ptr(), flags, libc::c_uint::from(mode)) };
	if descriptor < 0 {
		return Err(std::io::Error::last_os_error());
	}
	// SAFETY: successful `openat` returned a new owned descriptor.
	Ok(unsafe { File::from_raw_fd(descriptor) })
}

fn open_mutation_existing(target: &MutationTarget) -> Result<Option<File>, WorkspaceError> {
	match openat(
		target.directory.as_raw_fd(),
		&target.name,
		libc::O_RDONLY | libc::O_NONBLOCK | libc::O_NOFOLLOW | libc::O_CLOEXEC,
		0,
	) {
		Ok(file) => {
			let metadata = file.metadata().map_err(|source| WorkspaceError::Io {
				operation: "inspect mutation target",
				path: target.display.clone(),
				source,
			})?;
			if !metadata.is_file() {
				return Err(path_error(
					target.display.clone(),
					"mutation target is not a regular file",
				));
			}
			Ok(Some(file))
		}
		Err(source) if source.kind() == std::io::ErrorKind::NotFound => Ok(None),
		Err(source) => Err(WorkspaceError::Io {
			operation: "open mutation target without following links",
			path: target.display.clone(),
			source,
		}),
	}
}

fn atomic_replace(
	target: &MutationTarget,
	expected: Option<&File>,
	content: &[u8],
) -> Result<(), WorkspaceError> {
	let mode = expected
		.map(File::metadata)
		.transpose()
		.map_err(|source| WorkspaceError::Io {
			operation: "inspect existing mutation target",
			path: target.display.clone(),
			source,
		})?
		.map_or(0o600, |metadata| metadata.mode() & 0o777) as libc::mode_t;
	let (mut temporary, mut pending) = create_sibling_temp(target, mode)?;
	temporary
		.write_all(content)
		.and_then(|()| temporary.sync_all())
		.map_err(|source| WorkspaceError::Io {
			operation: "write and sync temporary file",
			path: target.display.clone(),
			source,
		})?;
	drop(temporary);

	match expected {
		Some(expected) => {
			replace_existing_with_swap(target, expected, &mut pending)?;
		}
		None => install_new_exclusively(target, pending.name.as_ref())?,
	}
	pending.disarm();
	target
		.directory
		.sync_all()
		.map_err(|source| WorkspaceError::Io {
			operation: "sync mutation parent directory",
			path: target.display.clone(),
			source,
		})
}

fn create_sibling_temp(
	target: &MutationTarget,
	mode: libc::mode_t,
) -> Result<(File, PendingTemp), WorkspaceError> {
	static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
	for _ in 0..32 {
		let sequence = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
		let name = format!(".emelex-tmp-{}-{sequence:016x}", std::process::id());
		let name = CString::new(name).map_err(|_| {
			path_error(
				target.display.clone(),
				"generated temporary filename contained NUL",
			)
		})?;
		// SAFETY: parent descriptor is live, `name` is one NUL-terminated
		// component, and a successful descriptor is owned exactly once.
		let descriptor = unsafe {
			libc::openat(
				target.directory.as_raw_fd(),
				name.as_ptr(),
				libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW | libc::O_CLOEXEC,
				libc::c_uint::from(0o600_u16),
			)
		};
		if descriptor < 0 {
			let source = std::io::Error::last_os_error();
			if source.kind() == std::io::ErrorKind::AlreadyExists {
				continue;
			}
			return Err(WorkspaceError::Io {
				operation: "create sibling temporary file",
				path: target.display.clone(),
				source,
			});
		}
		// SAFETY: successful `openat` returned a newly owned descriptor.
		let file = unsafe { File::from_raw_fd(descriptor) };
		// SAFETY: `file` is live and `mode` contains permission bits only.
		if unsafe { libc::fchmod(file.as_raw_fd(), mode) } != 0 {
			let source = std::io::Error::last_os_error();
			let _ = unlink_temp(&target.directory, &name);
			return Err(WorkspaceError::Io {
				operation: "set temporary file mode",
				path: target.display.clone(),
				source,
			});
		}
		let directory = match target.directory.try_clone() {
			Ok(directory) => directory,
			Err(source) => {
				let _ = unlink_temp(&target.directory, &name);
				return Err(WorkspaceError::Io {
					operation: "duplicate mutation parent descriptor",
					path: target.display.clone(),
					source,
				});
			}
		};
		return Ok((
			file,
			PendingTemp {
				directory,
				name: Some(name),
			},
		));
	}
	Err(WorkspaceError::Io {
		operation: "allocate unique sibling temporary file",
		path: target.display.clone(),
		source: std::io::Error::new(
			std::io::ErrorKind::AlreadyExists,
			"temporary filename collision limit reached",
		),
	})
}

fn replace_existing_with_swap(
	target: &MutationTarget,
	expected: &File,
	pending: &mut PendingTemp,
) -> Result<(), WorkspaceError> {
	let temporary_name = pending.name.clone().ok_or_else(|| WorkspaceError::Io {
		operation: "locate sibling temporary file",
		path: target.display.clone(),
		source: std::io::Error::new(
			std::io::ErrorKind::NotFound,
			"temporary filename unavailable",
		),
	})?;
	let target_name = c_name(&target.name).map_err(|source| WorkspaceError::Io {
		operation: "encode mutation target name",
		path: target.display.clone(),
		source,
	})?;
	// `RENAME_SWAP` keeps the previous destination under the temporary name,
	// allowing an identity check before it is removed.
	if let Err(source) = swap_entries(&target.directory, &temporary_name, &target_name) {
		return Err(WorkspaceError::Io {
			operation: "atomically swap mutation target",
			path: target.display.clone(),
			source,
		});
	}
	let expected_metadata = match expected.metadata() {
		Ok(metadata) => metadata,
		Err(source) => {
			rollback_or_preserve(target, pending, &temporary_name, &target_name)?;
			return Err(WorkspaceError::Io {
				operation: "inspect expected mutation descriptor",
				path: target.display.clone(),
				source,
			});
		}
	};
	let previous_metadata = match metadata_at(&target.directory, &temporary_name) {
		Ok(metadata) => metadata,
		Err(source) => {
			rollback_or_preserve(target, pending, &temporary_name, &target_name)?;
			return Err(WorkspaceError::Io {
				operation: "inspect atomically replaced target",
				path: target.display.clone(),
				source,
			});
		}
	};
	let Ok(previous_device) = u64::try_from(previous_metadata.st_dev) else {
		rollback_or_preserve(target, pending, &temporary_name, &target_name)?;
		return Err(path_error(
			target.display.clone(),
			"atomically replaced target has an invalid device identity",
		));
	};
	if expected_metadata.dev() != previous_device
		|| expected_metadata.ino() != previous_metadata.st_ino
	{
		rollback_or_preserve(target, pending, &temporary_name, &target_name)?;
		return Err(path_error(
			target.display.clone(),
			"mutation target changed before atomic replacement",
		));
	}
	unlink_temp(&target.directory, &temporary_name).map_err(|source| WorkspaceError::Io {
		operation: "remove replaced file after atomic swap",
		path: target.display.clone(),
		source,
	})
}

fn rollback_or_preserve(
	target: &MutationTarget,
	pending: &mut PendingTemp,
	temporary_name: &CStr,
	target_name: &CStr,
) -> Result<(), WorkspaceError> {
	if let Err(error) = rollback_swap(target, temporary_name, target_name) {
		// A failed rollback may leave the temporary name referring to the
		// original target rather than disposable scratch data. Disarm cleanup
		// so recovery data can never be unlinked by `PendingTemp::drop`.
		pending.disarm();
		return Err(error);
	}
	Ok(())
}

fn rollback_swap(
	target: &MutationTarget,
	temporary_name: &CStr,
	target_name: &CStr,
) -> Result<(), WorkspaceError> {
	swap_entries(&target.directory, temporary_name, target_name)
		.and_then(|()| target.directory.sync_all())
		.map_err(|source| WorkspaceError::Io {
			operation: "roll back raced mutation target; original may remain under sibling temporary name",
			path: target.display.clone(),
			source,
		})
}

fn swap_entries(directory: &File, left: &CStr, right: &CStr) -> std::io::Result<()> {
	#[cfg(test)]
	if injected_swap_failure() {
		return Err(std::io::Error::other("injected swap failure"));
	}
	// SAFETY: directory is live and both names are NUL-terminated components.
	if unsafe {
		libc::renameatx_np(
			directory.as_raw_fd(),
			left.as_ptr(),
			directory.as_raw_fd(),
			right.as_ptr(),
			libc::RENAME_SWAP,
		)
	} == 0
	{
		Ok(())
	} else {
		Err(std::io::Error::last_os_error())
	}
}

fn metadata_at(directory: &File, name: &CStr) -> std::io::Result<libc::stat> {
	#[cfg(test)]
	if injected_metadata_failure() {
		return Err(std::io::Error::other("injected metadata failure"));
	}
	let mut metadata = MaybeUninit::<libc::stat>::uninit();
	// SAFETY: directory and name are live; successful `fstatat` initializes
	// every byte of the output `stat`.
	if unsafe {
		libc::fstatat(
			directory.as_raw_fd(),
			name.as_ptr(),
			metadata.as_mut_ptr(),
			libc::AT_SYMLINK_NOFOLLOW,
		)
	} != 0
	{
		return Err(std::io::Error::last_os_error());
	}
	// SAFETY: successful `fstatat` initialized the value.
	Ok(unsafe { metadata.assume_init() })
}

#[cfg(test)]
std::thread_local! {
	static SWAP_FAILURE_AFTER: std::cell::Cell<Option<usize>> = const {
		std::cell::Cell::new(None)
	};
	static FAIL_METADATA_ONCE: std::cell::Cell<bool> = const {
		std::cell::Cell::new(false)
	};
}

#[cfg(test)]
fn inject_swap_failure_after(successful_calls: usize) {
	SWAP_FAILURE_AFTER.set(Some(successful_calls));
}

#[cfg(test)]
fn injected_swap_failure() -> bool {
	SWAP_FAILURE_AFTER.get().is_some_and(|remaining| {
		if remaining == 0 {
			SWAP_FAILURE_AFTER.set(None);
			true
		} else {
			SWAP_FAILURE_AFTER.set(Some(remaining - 1));
			false
		}
	})
}

#[cfg(test)]
fn inject_metadata_failure_once() {
	FAIL_METADATA_ONCE.set(true);
}

#[cfg(test)]
fn injected_metadata_failure() -> bool {
	FAIL_METADATA_ONCE.replace(false)
}

fn install_new_exclusively(
	target: &MutationTarget,
	temporary_name: Option<&CString>,
) -> Result<(), WorkspaceError> {
	let temporary_name = temporary_name.ok_or_else(|| WorkspaceError::Io {
		operation: "locate sibling temporary file",
		path: target.display.clone(),
		source: std::io::Error::new(
			std::io::ErrorKind::NotFound,
			"temporary filename unavailable",
		),
	})?;
	let target_name = c_name(&target.name).map_err(|source| WorkspaceError::Io {
		operation: "encode mutation target name",
		path: target.display.clone(),
		source,
	})?;
	// SAFETY: the pinned directory descriptor remains live, both names are
	// NUL-terminated safe single components, and `RENAME_EXCL` cannot replace
	// an entry that appeared after target resolution.
	let status = unsafe {
		libc::renameatx_np(
			target.directory.as_raw_fd(),
			temporary_name.as_ptr(),
			target.directory.as_raw_fd(),
			target_name.as_ptr(),
			libc::RENAME_EXCL,
		)
	};
	if status == 0 {
		Ok(())
	} else {
		Err(WorkspaceError::Io {
			operation: "atomically install new file without clobbering",
			path: target.display.clone(),
			source: std::io::Error::last_os_error(),
		})
	}
}

fn c_name(name: &OsStr) -> std::io::Result<CString> {
	CString::new(name.as_bytes())
		.map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidInput))
}

fn unlink_temp(directory: &File, name: &CStr) -> std::io::Result<()> {
	// SAFETY: directory is live and `name` is one NUL-terminated component.
	if unsafe { libc::unlinkat(directory.as_raw_fd(), name.as_ptr(), 0) } == 0 {
		Ok(())
	} else {
		Err(std::io::Error::last_os_error())
	}
}

struct PendingTemp {
	directory: File,
	name: Option<CString>,
}

impl PendingTemp {
	fn disarm(&mut self) {
		self.name = None;
	}
}

impl Drop for PendingTemp {
	fn drop(&mut self) {
		if let Some(name) = &self.name {
			let _ = unlink_temp(&self.directory, name);
		}
	}
}

fn path_error(path: impl Into<PathBuf>, reason: impl Into<String>) -> WorkspaceError {
	WorkspaceError::Path {
		path: path.into(),
		reason: reason.into(),
	}
}

#[derive(Debug, Clone)]
enum WorkspaceTool {
	Read,
	List,
	Find,
	Grep,
	Write,
	Edit,
	Shell {
		timeout_seconds: u64,
		output_bytes: usize,
		path: OsString,
		home: Option<PathBuf>,
	},
}

/// Construct the seven built-in, bounded workspace tools.
pub fn workspace_tools() -> Vec<Arc<dyn AgentTool>> {
	let mut tools = file_tools();
	tools.push(Arc::new(WorkspaceTool::Shell {
		timeout_seconds: DEFAULT_SHELL_TIMEOUT_SECONDS,
		output_bytes: DEFAULT_SHELL_OUTPUT_BYTES,
		path: sanitized_shell_path(std::env::var_os("PATH").as_deref()),
		home: sanitized_shell_home(std::env::var_os("HOME").as_deref()),
	}));
	tools
}

/// Construct the six descriptor-anchored filesystem tools.
pub fn file_tools() -> Vec<Arc<dyn AgentTool>> {
	[
		WorkspaceTool::Read,
		WorkspaceTool::List,
		WorkspaceTool::Find,
		WorkspaceTool::Grep,
		WorkspaceTool::Write,
		WorkspaceTool::Edit,
	]
	.into_iter()
	.map(|tool| Arc::new(tool) as Arc<dyn AgentTool>)
	.collect()
}

/// Construct host `shell` with authoritative timeout and output ceilings.
///
/// # Errors
///
/// Returns a configuration error when either ceiling exceeds the hard safety
/// maximum.
pub fn shell_tool(
	timeout_seconds: u64,
	output_bytes: usize,
) -> Result<Arc<dyn AgentTool>, WorkspaceError> {
	if !(1..=MAX_SHELL_TIMEOUT_SECONDS).contains(&timeout_seconds) {
		return Err(WorkspaceError::Limit {
			resource: "shell timeout seconds",
			limit: MAX_SHELL_TIMEOUT_SECONDS as usize,
		});
	}
	if !(1..=MAX_SHELL_OUTPUT_BYTES).contains(&output_bytes) {
		return Err(WorkspaceError::Limit {
			resource: "shell output bytes",
			limit: MAX_SHELL_OUTPUT_BYTES,
		});
	}
	Ok(Arc::new(WorkspaceTool::Shell {
		timeout_seconds,
		output_bytes,
		path: sanitized_shell_path(std::env::var_os("PATH").as_deref()),
		home: sanitized_shell_home(std::env::var_os("HOME").as_deref()),
	}))
}

#[async_trait]
impl AgentTool for WorkspaceTool {
	fn implementation_identity(&self) -> String {
		match self {
			Self::Read => "emelex.workspace.read@1".to_string(),
			Self::List => "emelex.workspace.list@1".to_string(),
			Self::Find => "emelex.workspace.find@1".to_string(),
			Self::Grep => "emelex.workspace.grep@1".to_string(),
			Self::Write => "emelex.workspace.write@1".to_string(),
			Self::Edit => "emelex.workspace.edit@1".to_string(),
			Self::Shell {
				timeout_seconds,
				output_bytes,
				path,
				home,
			} => format!(
				"emelex.workspace.shell@2;timeout={timeout_seconds};output={output_bytes};path_sha256={};home_sha256={}",
				shell_environment_digest(b"path", path),
				home.as_ref().map_or_else(
					|| "none".to_string(),
					|home| shell_environment_digest(b"home", home.as_os_str())
				)
			),
		}
	}

	#[expect(
		clippy::too_many_lines,
		reason = "one exhaustive match keeps advertised tool schemas beside their stable names"
	)]
	fn definition(&self) -> ToolDefinition {
		match self {
			Self::Read => ToolDefinition::new(
				"read_file",
				"Read one UTF-8 file. Paths are workspace-relative unless an absolute path is approved.",
				serde_json::json!({
					"type": "object",
					"properties": {
						"path": {"type": "string", "minLength": 1},
						"max_bytes": {
							"type": "integer",
							"minimum": 1,
							"maximum": MAX_FILE_BYTES
						}
					},
					"required": ["path"],
					"additionalProperties": false
				}),
			),
			Self::List => ToolDefinition::new(
				"list_directory",
				"List one directory without following symbolic links.",
				serde_json::json!({
					"type": "object",
					"properties": {
						"path": {"type": "string", "minLength": 1},
						"max_entries": {
							"type": "integer",
							"minimum": 1,
							"maximum": MAX_LIST_ENTRIES
						}
					},
					"required": ["path"],
					"additionalProperties": false
				}),
			),
			Self::Find => ToolDefinition::new(
				"find_files",
				"Find regular files by a bounded '*' and '?' wildcard pattern.",
				serde_json::json!({
					"type": "object",
					"properties": {
						"path": {"type": "string", "minLength": 1},
						"pattern": {"type": "string", "minLength": 1, "maxLength": 256},
						"max_results": {
							"type": "integer",
							"minimum": 1,
							"maximum": MAX_FIND_RESULTS
						},
						"max_depth": {
							"type": "integer",
							"minimum": 0,
							"maximum": MAX_WALK_DEPTH
						}
					},
					"required": ["path", "pattern"],
					"additionalProperties": false
				}),
			),
			Self::Grep => ToolDefinition::new(
				"grep",
				"Search bounded UTF-8 workspace files for a literal string.",
				serde_json::json!({
					"type": "object",
					"properties": {
						"path": {"type": "string", "minLength": 1},
						"query": {"type": "string", "minLength": 1, "maxLength": 4096},
						"case_sensitive": {"type": "boolean"},
						"max_matches": {
							"type": "integer",
							"minimum": 1,
							"maximum": MAX_GREP_MATCHES
						}
					},
					"required": ["path", "query"],
					"additionalProperties": false
				}),
			),
			Self::Write => ToolDefinition::new(
				"write_file",
				"Create or replace one bounded UTF-8 file after approval.",
				serde_json::json!({
					"type": "object",
					"properties": {
						"path": {"type": "string", "minLength": 1},
						"content": {"type": "string", "maxLength": MAX_FILE_BYTES}
					},
					"required": ["path", "content"],
					"additionalProperties": false
				}),
			),
			Self::Edit => ToolDefinition::new(
				"edit_file",
				"Replace exact text in one bounded UTF-8 file after approval.",
				serde_json::json!({
					"type": "object",
					"properties": {
						"path": {"type": "string", "minLength": 1},
						"old_text": {
							"type": "string",
							"minLength": 1,
							"maxLength": MAX_FILE_BYTES
						},
						"new_text": {"type": "string", "maxLength": MAX_FILE_BYTES},
						"replace_all": {"type": "boolean"}
					},
					"required": ["path", "old_text", "new_text"],
					"additionalProperties": false
				}),
			),
			Self::Shell {
				timeout_seconds, ..
			} => ToolDefinition::new(
				"shell",
				"Run one host /bin/sh command after approval, with bounded output and timeout.",
				serde_json::json!({
					"type": "object",
					"properties": {
						"command": {
							"type": "string",
							"minLength": 1,
							"maxLength": MAX_SHELL_COMMAND_BYTES
						},
						"cwd": {"type": "string", "minLength": 1},
						"timeout_seconds": {
							"type": "integer",
							"minimum": 1,
							"maximum": timeout_seconds
						}
					},
					"required": ["command"],
					"additionalProperties": false
				}),
			),
		}
	}

	fn approval_requirement(
		&self,
		context: &ToolContext,
		arguments: &serde_json::Value,
	) -> ApprovalRequirement {
		let path = arguments.get("path").and_then(serde_json::Value::as_str);
		match self {
			Self::Read | Self::List | Self::Find | Self::Grep => path
				.and_then(|path| read_approval_reason(context.workspace_root(), path))
				.map_or(ApprovalRequirement::None, |reason| {
					ApprovalRequirement::Required { reason }
				}),
			Self::Write | Self::Edit => {
				let reason = path
					.and_then(|path| path_boundary_reason(context.workspace_root(), path))
					.unwrap_or_else(|| "filesystem mutation".to_string());
				ApprovalRequirement::Required { reason }
			}
			Self::Shell { .. } => ApprovalRequirement::Required {
				reason: "host shell execution can access files, processes, and network".to_string(),
			},
		}
	}

	async fn invoke(
		&self,
		context: &ToolContext,
		arguments: serde_json::Value,
	) -> Result<ToolOutput, ToolError> {
		match self {
			Self::Shell {
				timeout_seconds,
				output_bytes,
				path,
				home,
			} => {
				shell(
					context,
					arguments,
					*timeout_seconds,
					*output_bytes,
					path.as_os_str(),
					home.as_deref(),
				)
				.await
			}
			Self::Write => {
				let context = context.clone();
				blocking_mutation(move || {
					check_tool_cancellation(&context)?;
					write_file(&context, arguments)
				})
				.await
			}
			Self::Edit => {
				let context = context.clone();
				blocking_mutation(move || {
					check_tool_cancellation(&context)?;
					edit_file(&context, arguments)
				})
				.await
			}
			tool @ (Self::Read | Self::List | Self::Find | Self::Grep) => {
				let context = context.clone();
				let tool = tool.clone();
				tokio::task::spawn_blocking(move || {
					check_tool_cancellation(&context)?;
					match tool {
						Self::Read => read_file(&context, arguments),
						Self::List => list_directory(&context, arguments),
						Self::Find => find_files(&context, arguments),
						Self::Grep => grep_files(&context, arguments),
						Self::Write | Self::Edit | Self::Shell { .. } => Err(ToolError::Fatal(
							"workspace tool dispatch violated its read-only boundary".to_string(),
						)),
					}
				})
				.await
				.map_err(|error| {
					ToolError::Fatal(format!("workspace tool worker failed: {error}"))
				})?
			}
		}
	}
}

struct MutationCompletion {
	state: Arc<(Mutex<bool>, Condvar)>,
	armed: bool,
}

impl MutationCompletion {
	fn new() -> (Self, MutationWorkerCompletion) {
		let state = Arc::new((Mutex::new(false), Condvar::new()));
		(
			Self {
				state: Arc::clone(&state),
				armed: true,
			},
			MutationWorkerCompletion { state },
		)
	}

	const fn disarm(&mut self) {
		self.armed = false;
	}
}

impl Drop for MutationCompletion {
	fn drop(&mut self) {
		if !self.armed {
			return;
		}
		let (completed, wake) = &*self.state;
		let mut completed = completed
			.lock()
			.unwrap_or_else(std::sync::PoisonError::into_inner);
		while !*completed {
			completed = wake
				.wait(completed)
				.unwrap_or_else(std::sync::PoisonError::into_inner);
		}
		drop(completed);
	}
}

struct MutationWorkerCompletion {
	state: Arc<(Mutex<bool>, Condvar)>,
}

impl Drop for MutationWorkerCompletion {
	fn drop(&mut self) {
		let (completed, wake) = &*self.state;
		*completed
			.lock()
			.unwrap_or_else(std::sync::PoisonError::into_inner) = true;
		wake.notify_all();
	}
}

async fn blocking_mutation<F>(work: F) -> Result<ToolOutput, ToolError>
where
	F: FnOnce() -> Result<ToolOutput, ToolError> + Send + 'static,
{
	let (mut completion, worker_completion) = MutationCompletion::new();
	let result = tokio::task::spawn_blocking(move || {
		let _completion = worker_completion;
		work()
	})
	.await
	.map_err(|error| ToolError::Fatal(format!("workspace mutation worker failed: {error}")))?;
	completion.disarm();
	result
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReadArgs {
	path: String,
	#[serde(default = "default_file_bytes")]
	max_bytes: usize,
}

const fn default_file_bytes() -> usize {
	MAX_FILE_BYTES
}

fn read_file(context: &ToolContext, arguments: serde_json::Value) -> Result<ToolOutput, ToolError> {
	let args: ReadArgs = parse_args(arguments)?;
	validate_required_text(&args.path, None, "path")?;
	if !(1..=MAX_FILE_BYTES).contains(&args.max_bytes) {
		return Err(respond(format!(
			"max_bytes must be in 1..={MAX_FILE_BYTES}"
		)));
	}
	let (mut file, path) = context
		.workspace
		.open_read(&args.path, context.approved())
		.map_err(workspace_respond)?;
	let bytes = read_bounded(&mut file, args.max_bytes, &path).map_err(workspace_respond)?;
	let text = String::from_utf8(bytes)
		.map_err(|_| respond(format!("file {} is not valid UTF-8", path.display())))?;
	Ok(ToolOutput::success(text))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ListArgs {
	path: String,
	#[serde(default = "default_list_entries")]
	max_entries: usize,
}

const fn default_list_entries() -> usize {
	MAX_LIST_ENTRIES
}

#[derive(Serialize)]
struct ListedEntry {
	name: String,
	kind: &'static str,
}

fn list_directory(
	context: &ToolContext,
	arguments: serde_json::Value,
) -> Result<ToolOutput, ToolError> {
	let args: ListArgs = parse_args(arguments)?;
	validate_required_text(&args.path, None, "path")?;
	if !(1..=MAX_LIST_ENTRIES).contains(&args.max_entries) {
		return Err(respond(format!(
			"max_entries must be in 1..={MAX_LIST_ENTRIES}"
		)));
	}
	let (directory, path) = context
		.workspace
		.open_directory(&args.path, context.approved())
		.map_err(workspace_respond)?;
	let mut entries = read_directory_entries(&directory, args.max_entries, &path)
		.map_err(workspace_respond)?
		.into_iter()
		.map(|entry| ListedEntry {
			name: entry.name.to_string_lossy().into_owned(),
			kind: entry.kind,
		})
		.collect::<Vec<_>>();
	entries.sort_unstable_by(|left, right| left.name.cmp(&right.name));
	json_output(&entries)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FindArgs {
	path: String,
	pattern: String,
	#[serde(default = "default_find_results")]
	max_results: usize,
	#[serde(default = "default_walk_depth")]
	max_depth: usize,
}

const fn default_find_results() -> usize {
	MAX_FIND_RESULTS
}

const fn default_walk_depth() -> usize {
	8
}

fn find_files(
	context: &ToolContext,
	arguments: serde_json::Value,
) -> Result<ToolOutput, ToolError> {
	let args: FindArgs = parse_args(arguments)?;
	validate_required_text(&args.path, None, "path")?;
	validate_required_text(&args.pattern, Some(MAX_FIND_PATTERN_CHARS), "pattern")?;
	if !(1..=MAX_FIND_RESULTS).contains(&args.max_results) {
		return Err(respond(format!(
			"max_results must be in 1..={MAX_FIND_RESULTS}"
		)));
	}
	if args.max_depth > MAX_WALK_DEPTH {
		return Err(respond(format!(
			"max_depth must be at most {MAX_WALK_DEPTH}"
		)));
	}
	let discovered = discover_files(
		&context.workspace,
		&args.path,
		context.approved(),
		args.max_depth,
		MAX_WALK_ENTRIES,
		MAX_WALK_ENTRIES,
		context.cancellation(),
	)
	.map_err(workspace_respond)?;
	let mut matches = Vec::new();
	for path in discovered.files {
		check_tool_cancellation(context)?;
		let Some(name) = path.file_name().and_then(OsStr::to_str) else {
			continue;
		};
		if wildcard_matches(&args.pattern, name) {
			matches.push(path.to_string_lossy().into_owned());
			if matches.len() == args.max_results {
				break;
			}
		}
	}
	json_output(&DiscoveryOutput {
		matches,
		skipped_sensitive_count: discovered.skipped_sensitive_count,
		skipped_sensitive_paths: discovered.skipped_sensitive_paths,
	})
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GrepArgs {
	path: String,
	query: String,
	#[serde(default = "default_true")]
	case_sensitive: bool,
	#[serde(default = "default_grep_matches")]
	max_matches: usize,
}

const fn default_true() -> bool {
	true
}

const fn default_grep_matches() -> usize {
	MAX_GREP_MATCHES
}

#[derive(Serialize)]
struct GrepMatch {
	path: String,
	line: usize,
	text: String,
}

#[derive(Serialize)]
struct DiscoveryOutput<T> {
	matches: Vec<T>,
	skipped_sensitive_count: usize,
	skipped_sensitive_paths: Vec<String>,
}

fn grep_files(
	context: &ToolContext,
	arguments: serde_json::Value,
) -> Result<ToolOutput, ToolError> {
	let args: GrepArgs = parse_args(arguments)?;
	validate_required_text(&args.path, None, "path")?;
	validate_required_text(&args.query, Some(MAX_GREP_QUERY_CHARS), "query")?;
	if !(1..=MAX_GREP_MATCHES).contains(&args.max_matches) {
		return Err(respond(format!(
			"max_matches must be in 1..={MAX_GREP_MATCHES}"
		)));
	}
	let discovered = discover_files(
		&context.workspace,
		&args.path,
		context.approved(),
		MAX_WALK_DEPTH,
		MAX_GREP_FILES,
		MAX_WALK_ENTRIES,
		context.cancellation(),
	)
	.map_err(workspace_respond)?;
	let query = if args.case_sensitive {
		args.query
	} else {
		args.query.to_lowercase()
	};
	let mut searched_bytes = 0_usize;
	let mut matches = Vec::new();
	for path in discovered.files {
		check_tool_cancellation(context)?;
		let path_string = path.to_string_lossy().into_owned();
		let (mut file, stable_path) = context
			.workspace
			.open_read(&path_string, context.approved())
			.map_err(workspace_respond)?;
		let remaining = MAX_GREP_TOTAL_BYTES.saturating_sub(searched_bytes);
		if remaining == 0 {
			break;
		}
		let file_limit = remaining.min(MAX_FILE_BYTES);
		let bytes = match read_bounded(&mut file, file_limit, &stable_path) {
			Ok(bytes) => bytes,
			Err(WorkspaceError::Limit { .. }) => continue,
			Err(error) => return Err(workspace_respond(error)),
		};
		searched_bytes = searched_bytes.saturating_add(bytes.len());
		let Ok(text) = String::from_utf8(bytes) else {
			continue;
		};
		for (index, line) in text.lines().enumerate() {
			let matches_query = if args.case_sensitive {
				line.contains(&query)
			} else {
				line.to_lowercase().contains(&query)
			};
			if matches_query {
				matches.push(GrepMatch {
					path: path_string.clone(),
					line: index + 1,
					text: line.to_string(),
				});
				if matches.len() == args.max_matches {
					return json_output(&DiscoveryOutput {
						matches,
						skipped_sensitive_count: discovered.skipped_sensitive_count,
						skipped_sensitive_paths: discovered.skipped_sensitive_paths,
					});
				}
			}
		}
	}
	json_output(&DiscoveryOutput {
		matches,
		skipped_sensitive_count: discovered.skipped_sensitive_count,
		skipped_sensitive_paths: discovered.skipped_sensitive_paths,
	})
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WriteArgs {
	path: String,
	content: String,
}

fn write_file(
	context: &ToolContext,
	arguments: serde_json::Value,
) -> Result<ToolOutput, ToolError> {
	let args: WriteArgs = parse_args(arguments)?;
	validate_required_text(&args.path, None, "path")?;
	if !context.approved() {
		return Err(ToolError::Fatal(
			"write_file reached execution without approval".to_string(),
		));
	}
	if args.content.len() > MAX_FILE_BYTES {
		return Err(respond(format!("content exceeds {MAX_FILE_BYTES} bytes")));
	}
	let target = context
		.workspace
		.mutation_target(&args.path, true)
		.map_err(workspace_respond)?;
	let existing = open_mutation_existing(&target).map_err(workspace_respond)?;
	atomic_replace(&target, existing.as_ref(), args.content.as_bytes())
		.map_err(workspace_respond)?;
	Ok(ToolOutput::success(format!(
		"wrote {} bytes to {}",
		args.content.len(),
		target.display.display()
	)))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EditArgs {
	path: String,
	old_text: String,
	new_text: String,
	#[serde(default)]
	replace_all: bool,
}

fn edit_file(context: &ToolContext, arguments: serde_json::Value) -> Result<ToolOutput, ToolError> {
	let args: EditArgs = parse_args(arguments)?;
	validate_required_text(&args.path, None, "path")?;
	if !context.approved() {
		return Err(ToolError::Fatal(
			"edit_file reached execution without approval".to_string(),
		));
	}
	if args.old_text.is_empty() {
		return Err(respond("old_text cannot be empty"));
	}
	if args.old_text.len() > MAX_FILE_BYTES || args.new_text.len() > MAX_FILE_BYTES {
		return Err(respond(format!(
			"edit strings cannot exceed {MAX_FILE_BYTES} bytes"
		)));
	}
	let target = context
		.workspace
		.mutation_target(&args.path, true)
		.map_err(workspace_respond)?;
	let mut file = open_mutation_existing(&target)
		.map_err(workspace_respond)?
		.ok_or_else(|| respond(format!("file {} does not exist", target.display.display())))?;
	let path = &target.display;
	let bytes = read_bounded(&mut file, MAX_FILE_BYTES, path).map_err(workspace_respond)?;
	let original = String::from_utf8(bytes)
		.map_err(|_| respond(format!("file {} is not valid UTF-8", path.display())))?;
	let occurrences = original.matches(&args.old_text).count();
	if occurrences == 0 {
		return Err(respond("old_text was not found"));
	}
	if occurrences > 1 && !args.replace_all {
		return Err(respond(format!(
			"old_text matched {occurrences} times; set replace_all to edit all matches"
		)));
	}
	let updated = if args.replace_all {
		original.replace(&args.old_text, &args.new_text)
	} else {
		original.replacen(&args.old_text, &args.new_text, 1)
	};
	if updated.len() > MAX_FILE_BYTES {
		return Err(respond(format!(
			"edited file would exceed {MAX_FILE_BYTES} bytes"
		)));
	}
	atomic_replace(&target, Some(&file), updated.as_bytes()).map_err(workspace_respond)?;
	Ok(ToolOutput::success(format!(
		"replaced {occurrences} occurrence(s) in {}",
		target.display.display()
	)))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ShellArgs {
	command: String,
	#[serde(default = "default_shell_cwd")]
	cwd: String,
	#[serde(default)]
	timeout_seconds: Option<u64>,
}

fn default_shell_cwd() -> String {
	".".to_string()
}

async fn shell(
	context: &ToolContext,
	arguments: serde_json::Value,
	configured_timeout_seconds: u64,
	output_bytes: usize,
	shell_path: &OsStr,
	shell_home: Option<&Path>,
) -> Result<ToolOutput, ToolError> {
	let args: ShellArgs = parse_args(arguments)?;
	if !context.approved() {
		return Err(ToolError::Fatal(
			"shell reached execution without approval".to_string(),
		));
	}
	validate_required_text(&args.cwd, None, "cwd")?;
	if args.command.is_empty() || args.command.len() > MAX_SHELL_COMMAND_BYTES {
		return Err(respond(format!(
			"command must be in 1..={MAX_SHELL_COMMAND_BYTES} bytes"
		)));
	}
	let timeout_seconds = args.timeout_seconds.unwrap_or(configured_timeout_seconds);
	if !(1..=configured_timeout_seconds).contains(&timeout_seconds) {
		return Err(respond(format!(
			"timeout_seconds must be in 1..={configured_timeout_seconds}"
		)));
	}
	let (cwd, cwd_path) = context
		.workspace
		.open_directory(&args.cwd, true)
		.map_err(workspace_respond)?;
	run_shell(
		&args.command,
		&cwd,
		&cwd_path,
		Duration::from_secs(timeout_seconds),
		output_bytes,
		shell_path,
		shell_home,
		context.cancellation(),
	)
	.await
}

async fn run_shell(
	script: &str,
	cwd: &File,
	cwd_path: &Path,
	timeout: Duration,
	output_bytes: usize,
	path: &OsStr,
	home: Option<&Path>,
	cancellation: &super::AgentCancellation,
) -> Result<ToolOutput, ToolError> {
	let mut command = shell_command(script, cwd, path, home);
	let mut child = command.spawn().map_err(|source| {
		workspace_respond(WorkspaceError::Io {
			operation: "spawn shell",
			path: cwd_path.to_path_buf(),
			source,
		})
	})?;
	let process_id = child
		.id()
		.ok_or_else(|| ToolError::Fatal("spawned shell has no process ID".to_string()))?;
	let mut process_group = ProcessGroupGuard::new(process_id);
	let stdout = child
		.stdout
		.take()
		.ok_or_else(|| ToolError::Fatal("shell stdout pipe is unavailable".to_string()))?;
	let stderr = child
		.stderr
		.take()
		.ok_or_else(|| ToolError::Fatal("shell stderr pipe is unavailable".to_string()))?;
	let stdout_bytes = output_bytes / 2;
	let stderr_bytes = output_bytes.saturating_sub(stdout_bytes);
	let mut stdout_task = DrainTask::new(tokio::spawn(drain_capped(stdout, stdout_bytes)));
	let mut stderr_task = DrainTask::new(tokio::spawn(drain_capped(stderr, stderr_bytes)));
	let stop = tokio::select! {
		biased;
		() = cancellation.cancelled() => ShellStop::Cancelled,
		() = tokio::time::sleep(timeout) => ShellStop::TimedOut,
		status = child.wait() => ShellStop::Exited(status),
	};
	let (status, timed_out, cancelled) =
		finish_shell(stop, &mut child, &mut process_group, cwd_path).await?;
	let stdout = join_drain(&mut stdout_task).await;
	let stderr = join_drain(&mut stderr_task).await;
	if cancelled {
		return Err(ToolError::Cancelled);
	}
	let mut output = format_shell_output(status.code(), timed_out, &stdout, &stderr);
	truncate_utf8(&mut output, output_bytes);
	Ok(if timed_out || !status.success() {
		ToolOutput::error(output)
	} else {
		ToolOutput::success(output)
	})
}

fn shell_command(
	script: &str,
	cwd: &File,
	path: &OsStr,
	home: Option<&Path>,
) -> tokio::process::Command {
	let mut command = tokio::process::Command::new("/bin/sh");
	command
		.arg("-c")
		.arg(script)
		.env_clear()
		.env("PATH", path)
		.env("LANG", "en_US.UTF-8")
		.stdin(Stdio::null())
		.stdout(Stdio::piped())
		.stderr(Stdio::piped())
		.kill_on_drop(true);
	if let Some(home) = home {
		command.env("HOME", home);
	}
	command.as_std_mut().process_group(0);
	let cwd_descriptor = cwd.as_raw_fd();
	// SAFETY: `cwd_descriptor` stays open through `spawn`, and `fchdir` is an
	// async-signal-safe syscall. The closure allocates nothing and touches no
	// shared Rust state after fork.
	unsafe {
		command.as_std_mut().pre_exec(move || {
			if libc::fchdir(cwd_descriptor) == 0 {
				Ok(())
			} else {
				Err(std::io::Error::last_os_error())
			}
		});
	}
	command
}

fn shell_environment_digest(domain: &[u8], value: &OsStr) -> String {
	let mut digest = Sha256::new();
	digest.update(b"emelex.workspace.shell.environment@1");
	digest.update([0]);
	digest.update(domain);
	digest.update([0]);
	digest.update(value.as_bytes());
	hex::encode(digest.finalize())
}

fn sanitized_shell_path(inherited: Option<&OsStr>) -> OsString {
	let mut paths = Vec::<PathBuf>::new();
	let mut total_bytes = 0_usize;
	if let Some(inherited) = inherited {
		for path in std::env::split_paths(inherited).take(MAX_SHELL_PATH_ENTRIES) {
			let bytes = path.as_os_str().as_bytes().len();
			let safe = path.is_absolute()
				&& bytes > 0 && path
				.components()
				.all(|component| matches!(component, Component::RootDir | Component::Normal(_)));
			let fits = total_bytes
				.checked_add(bytes.saturating_add(1))
				.is_some_and(|next| next <= MAX_SHELL_PATH_BYTES);
			if safe && fits && !paths.contains(&path) {
				total_bytes = total_bytes.saturating_add(bytes.saturating_add(1));
				paths.push(path);
			}
		}
	}
	for path in std::env::split_paths(OsStr::new(FALLBACK_SHELL_PATH)) {
		if !paths.contains(&path) {
			paths.push(path);
		}
	}
	std::env::join_paths(paths).unwrap_or_else(|_| OsString::from(FALLBACK_SHELL_PATH))
}

fn sanitized_shell_home(inherited: Option<&OsStr>) -> Option<PathBuf> {
	let home = PathBuf::from(inherited?);
	let safe = home.is_absolute()
		&& !home.as_os_str().is_empty()
		&& home.as_os_str().as_bytes().len() <= 4_096
		&& home
			.components()
			.all(|component| matches!(component, Component::RootDir | Component::Normal(_)));
	safe.then_some(home)
}

async fn finish_shell(
	stop: ShellStop,
	child: &mut tokio::process::Child,
	process_group: &mut ProcessGroupGuard,
	cwd_path: &Path,
) -> Result<(std::process::ExitStatus, bool, bool), ToolError> {
	let (status, timed_out, cancelled) = match stop {
		ShellStop::Exited(status) => {
			let status = status.map_err(|source| {
				workspace_respond(WorkspaceError::Io {
					operation: "wait for shell",
					path: cwd_path.to_path_buf(),
					source,
				})
			})?;
			// The shell leader may exit while background descendants retain the
			// process group and inherited pipes. Kill the group even on a normal
			// leader exit so tools cannot leak detached work into the host.
			process_group.kill().map_err(|source| {
				ToolError::Fatal(format!("cannot kill shell process group: {source}"))
			})?;
			process_group.disarm();
			(status, false, false)
		}
		ShellStop::TimedOut | ShellStop::Cancelled => {
			let group_kill = process_group.kill();
			let _ = child.start_kill();
			let status = tokio::time::timeout(IO_DRAIN_TIMEOUT, child.wait())
				.await
				.map_err(|_| {
					ToolError::Fatal("shell did not exit after process-group kill".to_string())
				})?
				.map_err(|source| {
					workspace_respond(WorkspaceError::Io {
						operation: "reap killed shell",
						path: cwd_path.to_path_buf(),
						source,
					})
				})?;
			group_kill.map_err(|source| {
				ToolError::Fatal(format!("cannot kill shell process group: {source}"))
			})?;
			process_group.disarm();
			(
				status,
				matches!(stop, ShellStop::TimedOut),
				matches!(stop, ShellStop::Cancelled),
			)
		}
	};
	Ok((status, timed_out, cancelled))
}

enum ShellStop {
	Exited(std::io::Result<std::process::ExitStatus>),
	TimedOut,
	Cancelled,
}

struct ProcessGroupGuard {
	process_id: Option<u32>,
}

impl ProcessGroupGuard {
	const fn new(process_id: u32) -> Self {
		Self {
			process_id: Some(process_id),
		}
	}

	fn kill(&self) -> std::io::Result<()> {
		let Some(process_id) = self.process_id else {
			return Ok(());
		};
		let process_id = i32::try_from(process_id)
			.map_err(|_| std::io::Error::other("child process ID is outside i32 range"))?;
		// SAFETY: a negative PID targets only the dedicated process group
		// created for this child.
		if unsafe { libc::kill(-process_id, libc::SIGKILL) } == 0 {
			return Ok(());
		}
		let source = std::io::Error::last_os_error();
		if source.raw_os_error() == Some(libc::ESRCH) {
			Ok(())
		} else {
			Err(source)
		}
	}

	const fn disarm(&mut self) {
		self.process_id = None;
	}
}

impl Drop for ProcessGroupGuard {
	fn drop(&mut self) {
		let _ = self.kill();
	}
}

#[derive(Default)]
struct CappedOutput {
	head: Vec<u8>,
	tail: VecDeque<u8>,
	total: usize,
	limit: usize,
}

impl CappedOutput {
	fn new(limit: usize) -> Self {
		Self {
			limit,
			..Self::default()
		}
	}

	fn push(&mut self, bytes: &[u8]) {
		self.total = self.total.saturating_add(bytes.len());
		if self.limit == 0 {
			return;
		}
		let head_limit = self.limit / 2;
		let head_remaining = head_limit.saturating_sub(self.head.len());
		let head_bytes = head_remaining.min(bytes.len());
		self.head.extend_from_slice(&bytes[..head_bytes]);
		let tail_limit = self.limit.saturating_sub(head_limit);
		for byte in &bytes[head_bytes..] {
			if self.tail.len() == tail_limit {
				self.tail.pop_front();
			}
			self.tail.push_back(*byte);
		}
	}

	fn render(self) -> String {
		if self.total <= self.limit {
			let mut bytes = self.head;
			bytes.extend(self.tail);
			return String::from_utf8_lossy(&bytes).into_owned();
		}
		let omitted = self.total.saturating_sub(self.head.len() + self.tail.len());
		let mut text = String::from_utf8_lossy(&self.head).into_owned();
		text.push_str("\n... ");
		text.push_str(&omitted.to_string());
		text.push_str(" bytes omitted ...\n");
		let tail = self.tail.into_iter().collect::<Vec<_>>();
		text.push_str(&String::from_utf8_lossy(&tail));
		text
	}
}

async fn drain_capped<R>(mut reader: R, limit: usize) -> std::io::Result<CappedOutput>
where
	R: AsyncRead + Unpin,
{
	let mut output = CappedOutput::new(limit);
	let mut buffer = [0_u8; 8192];
	loop {
		let read = reader.read(&mut buffer).await?;
		if read == 0 {
			return Ok(output);
		}
		output.push(&buffer[..read]);
	}
}

async fn join_drain(task: &mut DrainTask) -> String {
	match tokio::time::timeout(IO_DRAIN_TIMEOUT, &mut task.handle).await {
		Ok(Ok(Ok(output))) => output.render(),
		Ok(Ok(Err(error))) => format!("<output read failed: {error}>"),
		Ok(Err(error)) => format!("<output task failed: {error}>"),
		Err(_) => {
			task.handle.abort();
			"<output drain timed out>".to_string()
		}
	}
}

struct DrainTask {
	handle: tokio::task::JoinHandle<std::io::Result<CappedOutput>>,
}

impl DrainTask {
	const fn new(handle: tokio::task::JoinHandle<std::io::Result<CappedOutput>>) -> Self {
		Self { handle }
	}
}

impl Drop for DrainTask {
	fn drop(&mut self) {
		self.handle.abort();
	}
}

fn format_shell_output(code: Option<i32>, timed_out: bool, stdout: &str, stderr: &str) -> String {
	let status = if timed_out {
		"timeout".to_string()
	} else {
		code.map_or_else(|| "signal".to_string(), |code| code.to_string())
	};
	format!("status: {status}\nstdout:\n{stdout}\nstderr:\n{stderr}")
}

struct DirectoryEntry {
	name: OsString,
	kind: &'static str,
}

fn read_directory_entries(
	directory: &File,
	limit: usize,
	path: &Path,
) -> Result<Vec<DirectoryEntry>, WorkspaceError> {
	// SAFETY: `fcntl` duplicates a live descriptor. On success ownership moves
	// immediately to `fdopendir`, which closes it through `DirectoryStream`.
	let descriptor = unsafe { libc::fcntl(directory.as_raw_fd(), libc::F_DUPFD_CLOEXEC, 0) };
	if descriptor < 0 {
		return Err(WorkspaceError::Io {
			operation: "duplicate directory descriptor",
			path: path.to_path_buf(),
			source: std::io::Error::last_os_error(),
		});
	}
	// SAFETY: `descriptor` is a newly owned directory descriptor. `fdopendir`
	// takes ownership on success.
	let stream = unsafe { libc::fdopendir(descriptor) };
	if stream.is_null() {
		let source = std::io::Error::last_os_error();
		// SAFETY: `fdopendir` failed, so ownership remains with this function.
		let _ = unsafe { libc::close(descriptor) };
		return Err(WorkspaceError::Io {
			operation: "open directory stream",
			path: path.to_path_buf(),
			source,
		});
	}
	let stream = DirectoryStream(stream);
	let mut output = Vec::new();
	loop {
		// SAFETY: macOS exposes thread-local errno through `__error`. Clearing
		// it distinguishes end-of-stream from a `readdir` failure.
		unsafe {
			*libc::__error() = 0;
		}
		// SAFETY: `stream` remains owned and open for this loop.
		let entry = unsafe { libc::readdir(stream.0) };
		if entry.is_null() {
			// SAFETY: same thread-local errno read immediately after `readdir`.
			let error = unsafe { *libc::__error() };
			if error == 0 {
				break;
			}
			return Err(WorkspaceError::Io {
				operation: "read directory entry",
				path: path.to_path_buf(),
				source: std::io::Error::from_raw_os_error(error),
			});
		}
		// SAFETY: `readdir` returned a live `dirent`; `d_name` is
		// NUL-terminated for the lifetime of the directory stream.
		let name = unsafe { CStr::from_ptr((*entry).d_name.as_ptr()) };
		if matches!(name.to_bytes(), b"." | b"..") {
			continue;
		}
		if output.len() == limit {
			return Err(WorkspaceError::Limit {
				resource: "directory entries",
				limit,
			});
		}
		// SAFETY: `entry` is valid until the next `readdir`, and `d_type` is a
		// plain value copied before then.
		let kind = directory_entry_kind(directory, name, unsafe { (*entry).d_type }, path)?;
		output.push(DirectoryEntry {
			name: OsString::from_vec(name.to_bytes().to_vec()),
			kind,
		});
	}
	Ok(output)
}

fn directory_entry_kind(
	directory: &File,
	name: &CStr,
	d_type: u8,
	path: &Path,
) -> Result<&'static str, WorkspaceError> {
	let known = match d_type {
		libc::DT_LNK => Some("symlink"),
		libc::DT_DIR => Some("directory"),
		libc::DT_REG => Some("file"),
		libc::DT_UNKNOWN => None,
		_ => Some("other"),
	};
	if let Some(kind) = known {
		return Ok(kind);
	}

	let mut metadata = MaybeUninit::<libc::stat>::uninit();
	// SAFETY: `directory` is a live directory descriptor, `name` remains
	// NUL-terminated for this call, and `metadata` points to writable storage.
	let status = unsafe {
		libc::fstatat(
			directory.as_raw_fd(),
			name.as_ptr(),
			metadata.as_mut_ptr(),
			libc::AT_SYMLINK_NOFOLLOW,
		)
	};
	if status != 0 {
		return Err(WorkspaceError::Io {
			operation: "inspect directory entry",
			path: path.join(OsStr::from_bytes(name.to_bytes())),
			source: std::io::Error::last_os_error(),
		});
	}
	// SAFETY: successful `fstatat` initialized the complete `stat` value.
	let mode = unsafe { metadata.assume_init() }.st_mode;
	Ok(match mode & libc::S_IFMT {
		libc::S_IFLNK => "symlink",
		libc::S_IFDIR => "directory",
		libc::S_IFREG => "file",
		_ => "other",
	})
}

struct DirectoryStream(*mut libc::DIR);

impl Drop for DirectoryStream {
	fn drop(&mut self) {
		// SAFETY: this guard uniquely owns the `fdopendir` stream.
		let _ = unsafe { libc::closedir(self.0) };
	}
}

#[derive(Debug)]
struct DiscoveredFiles {
	files: Vec<PathBuf>,
	skipped_sensitive_count: usize,
	skipped_sensitive_paths: Vec<String>,
}

fn discover_files(
	workspace: &WorkspaceRoot,
	start: &str,
	approved: bool,
	max_depth: usize,
	max_files: usize,
	max_walk_entries: usize,
	cancellation: &super::AgentCancellation,
) -> Result<DiscoveredFiles, WorkspaceError> {
	if max_walk_entries == 0 {
		return Err(WorkspaceError::Limit {
			resource: "walk entries",
			limit: 0,
		});
	}
	let (_directory, start_path) = workspace.open_directory(start, approved)?;
	let mut queue = VecDeque::from([(start_path, 0_usize)]);
	let mut examined_entries = 1_usize;
	let mut files = Vec::new();
	let mut skipped_sensitive_count = 0_usize;
	let mut skipped_sensitive_paths = Vec::new();
	while let Some((directory_path, depth)) = queue.pop_front() {
		check_workspace_cancellation(cancellation)?;
		let directory_string = directory_path.to_string_lossy();
		let (directory, stable_path) = workspace.open_directory(&directory_string, approved)?;
		let entries = read_directory_entries(&directory, MAX_LIST_ENTRIES, &stable_path)?;
		for entry in entries {
			check_workspace_cancellation(cancellation)?;
			examined_entries = examined_entries
				.checked_add(1)
				.ok_or(WorkspaceError::Limit {
					resource: "walk entries",
					limit: max_walk_entries,
				})?;
			if examined_entries > max_walk_entries {
				return Err(WorkspaceError::Limit {
					resource: "walk entries",
					limit: max_walk_entries,
				});
			}
			if entry.kind == "symlink" || entry.kind == "other" {
				continue;
			}
			let path = directory_path.join(&entry.name);
			if path.to_str().is_none() {
				continue;
			}
			if !approved && likely_secret(&path) {
				skipped_sensitive_count = skipped_sensitive_count.saturating_add(1);
				if skipped_sensitive_paths.len() < MAX_SENSITIVE_SKIP_PATHS {
					skipped_sensitive_paths.push(path.to_string_lossy().into_owned());
				}
				continue;
			}
			if entry.kind == "directory" {
				if depth < max_depth {
					if queue.len() >= max_walk_entries {
						return Err(WorkspaceError::Limit {
							resource: "walk queue entries",
							limit: max_walk_entries,
						});
					}
					queue.push_back((path, depth + 1));
				}
			} else if entry.kind == "file" {
				files.push(path);
				if files.len() == max_files {
					return Ok(DiscoveredFiles {
						files,
						skipped_sensitive_count,
						skipped_sensitive_paths,
					});
				}
			}
		}
	}
	Ok(DiscoveredFiles {
		files,
		skipped_sensitive_count,
		skipped_sensitive_paths,
	})
}

fn read_bounded(file: &mut File, limit: usize, path: &Path) -> Result<Vec<u8>, WorkspaceError> {
	let metadata = file.metadata().map_err(|source| WorkspaceError::Io {
		operation: "inspect file descriptor",
		path: path.to_path_buf(),
		source,
	})?;
	if !metadata.is_file() {
		return Err(path_error(path, "target is not a regular file"));
	}
	if metadata.len() > limit as u64 {
		return Err(WorkspaceError::Limit {
			resource: "file bytes",
			limit,
		});
	}
	let mut bytes = Vec::with_capacity(metadata.len() as usize);
	file.seek(SeekFrom::Start(0))
		.and_then(|_| file.take((limit + 1) as u64).read_to_end(&mut bytes))
		.map_err(|source| WorkspaceError::Io {
			operation: "read file descriptor",
			path: path.to_path_buf(),
			source,
		})?;
	if bytes.len() > limit {
		return Err(WorkspaceError::Limit {
			resource: "file bytes",
			limit,
		});
	}
	Ok(bytes)
}

fn wildcard_matches(pattern: &str, value: &str) -> bool {
	let pattern = pattern.as_bytes();
	let value = value.as_bytes();
	let mut previous = vec![false; value.len() + 1];
	previous[0] = true;
	for token in pattern {
		let mut next = vec![false; value.len() + 1];
		if *token == b'*' {
			next[0] = previous[0];
		}
		for index in 1..=value.len() {
			next[index] = match *token {
				b'*' => previous[index] || next[index - 1],
				b'?' => previous[index - 1],
				literal => previous[index - 1] && literal == value[index - 1],
			};
		}
		previous = next;
	}
	previous[value.len()]
}

fn read_approval_reason(root: &Path, path: &str) -> Option<String> {
	path_boundary_reason(root, path).or_else(|| {
		likely_secret(Path::new(path)).then(|| "read may expose a likely secret file".to_string())
	})
}

fn path_boundary_reason(root: &Path, path: &str) -> Option<String> {
	let path = Path::new(path);
	if path.is_absolute() && !path.starts_with(root) {
		return Some("absolute path is outside the workspace".to_string());
	}
	path.components()
		.any(|component| component == Component::ParentDir)
		.then(|| "path contains parent traversal".to_string())
}

fn likely_secret(path: &Path) -> bool {
	path.components().any(|component| {
		let Component::Normal(component) = component else {
			return false;
		};
		let name = component.to_string_lossy().to_ascii_lowercase();
		let stem = Path::new(&name)
			.file_stem()
			.and_then(OsStr::to_str)
			.unwrap_or(&name);
		name == ".env"
			|| name.starts_with(".env.")
			|| matches!(
				name.as_str(),
				".ssh"
					| ".aws" | ".gnupg"
					| ".netrc" | ".npmrc"
					| "credentials" | "credentials.json"
					| "credentials.toml"
					| "id_rsa" | "id_ed25519"
					| "auth.json"
			) || Path::new(&name)
			.extension()
			.is_some_and(|extension| matches!(extension.to_str(), Some("pem" | "key")))
			|| matches!(
				stem,
				"secret"
					| "secrets" | "token"
					| "tokens" | "api_key"
					| "apikey" | "access_token"
					| "auth_token" | "client_secret"
			) || stem.ends_with("_secret")
			|| stem.ends_with("-secret")
	})
}

fn validate_required_text(
	value: &str,
	max_chars: Option<usize>,
	name: &str,
) -> Result<(), ToolError> {
	if value.is_empty() {
		return Err(respond(format!("{name} cannot be empty")));
	}
	if let Some(max_chars) = max_chars
		&& value.chars().count() > max_chars
	{
		return Err(respond(format!(
			"{name} cannot exceed {max_chars} characters"
		)));
	}
	Ok(())
}

fn check_tool_cancellation(context: &ToolContext) -> Result<(), ToolError> {
	if context.cancellation().is_cancelled() {
		return Err(ToolError::Cancelled);
	}
	Ok(())
}

fn check_workspace_cancellation(
	cancellation: &super::AgentCancellation,
) -> Result<(), WorkspaceError> {
	if cancellation.is_cancelled() {
		Err(WorkspaceError::Cancelled)
	} else {
		Ok(())
	}
}

fn parse_args<T: DeserializeOwned>(arguments: serde_json::Value) -> Result<T, ToolError> {
	serde_json::from_value(arguments)
		.map_err(|error| respond(format!("invalid tool arguments: {error}")))
}

fn json_output<T: Serialize>(value: &T) -> Result<ToolOutput, ToolError> {
	match serialize_json_pretty_bounded(value, MAX_FILE_BYTES) {
		Ok(output) => Ok(ToolOutput::success(output)),
		Err(BoundedJsonError::Limit { .. }) => Err(respond(format!(
			"structured output exceeds {MAX_FILE_BYTES} bytes"
		))),
		Err(error @ (BoundedJsonError::Serialize(_) | BoundedJsonError::Utf8)) => Err(
			ToolError::Fatal(format!("cannot serialize tool output: {error}")),
		),
	}
}

fn workspace_respond(error: WorkspaceError) -> ToolError {
	let message = match error {
		WorkspaceError::Path { path, reason } => {
			format!("invalid workspace path {}: {reason}", path.display())
		}
		WorkspaceError::Io {
			operation,
			path,
			source,
		} => format!(
			"workspace {operation} failed for {}: {source}",
			path.display()
		),
		WorkspaceError::Limit { resource, limit } => {
			format!("workspace {resource} exceeded its limit of {limit}")
		}
		WorkspaceError::Cancelled => {
			return ToolError::Cancelled;
		}
	};
	respond(message)
}

fn respond(message: impl Into<String>) -> ToolError {
	ToolError::RespondToModel(message.into())
}

fn truncate_utf8(text: &mut String, limit: usize) {
	if text.len() <= limit {
		return;
	}
	let mut boundary = limit;
	while boundary > 0 && !text.is_char_boundary(boundary) {
		boundary -= 1;
	}
	text.truncate(boundary);
}

#[cfg(test)]
mod tests {
	#![allow(clippy::expect_used, clippy::unwrap_used)]

	use std::os::unix::fs::PermissionsExt as _;

	use super::*;

	fn root(directory: &tempfile::TempDir) -> WorkspaceRoot {
		WorkspaceRoot::open(directory.path()).expect("workspace root")
	}

	fn test_shell(timeout_seconds: u64, output_bytes: usize) -> WorkspaceTool {
		WorkspaceTool::Shell {
			timeout_seconds,
			output_bytes,
			path: sanitized_shell_path(None),
			home: None,
		}
	}

	#[tokio::test(flavor = "current_thread")]
	async fn blocking_mutation_keeps_runtime_timer_responsive() {
		let mutation = blocking_mutation(|| {
			std::thread::sleep(Duration::from_millis(75));
			Ok(ToolOutput::success("done"))
		});
		tokio::pin!(mutation);

		tokio::select! {
			biased;
			() = tokio::time::sleep(Duration::from_millis(10)) => {}
			result = &mut mutation => panic!("mutation finished before timer: {result:?}"),
		}

		assert_eq!(mutation.await.expect("mutation").content, "done");
	}

	#[tokio::test(flavor = "current_thread")]
	async fn dropping_mutation_future_waits_for_terminal_host_effect() {
		let directory = tempfile::tempdir().expect("tempdir");
		let marker = directory.path().join("finished");
		let worker_marker = marker.clone();
		let mut mutation = Box::pin(blocking_mutation(move || {
			std::thread::sleep(Duration::from_millis(75));
			std::fs::write(&worker_marker, b"complete").expect("write marker");
			Ok(ToolOutput::success("done"))
		}));

		tokio::select! {
			biased;
			() = tokio::time::sleep(Duration::from_millis(10)) => {}
			result = &mut mutation => panic!("mutation finished before drop: {result:?}"),
		}
		drop(mutation);

		assert_eq!(
			std::fs::read(marker).expect("completed marker"),
			b"complete"
		);
	}

	#[test]
	fn descriptor_read_rejects_parent_and_symlink_escape() {
		let directory = tempfile::tempdir().expect("tempdir");
		let outside = tempfile::NamedTempFile::new().expect("outside");
		std::os::unix::fs::symlink(outside.path(), directory.path().join("link")).expect("symlink");
		let workspace = root(&directory);

		assert!(workspace.open_read("../outside", true).is_err());
		assert!(workspace.open_read("link", false).is_err());
	}

	#[test]
	fn atomic_mutation_rejects_symlink_target() {
		let directory = tempfile::tempdir().expect("tempdir");
		let outside = tempfile::NamedTempFile::new().expect("outside");
		std::fs::write(outside.path(), "outside").expect("seed outside");
		std::os::unix::fs::symlink(outside.path(), directory.path().join("link")).expect("symlink");
		let context = ToolContext {
			call_id: uuid::Uuid::now_v7().to_string(),
			workspace: Arc::new(root(&directory)),
			cancellation: super::super::AgentCancellation::new(),
			approved: true,
		};

		let result = write_file(
			&context,
			serde_json::json!({"path": "link", "content": "replacement"}),
		);

		assert!(result.is_err());
		assert_eq!(
			std::fs::read_to_string(outside.path()).expect("outside"),
			"outside"
		);
	}

	#[test]
	fn write_and_edit_replace_atomically_and_preserve_permissions() {
		let directory = tempfile::tempdir().expect("tempdir");
		let path = directory.path().join("note.txt");
		std::fs::write(&path, "old").expect("seed");
		let mut permissions = std::fs::metadata(&path).expect("metadata").permissions();
		permissions.set_mode(0o640);
		std::fs::set_permissions(&path, permissions).expect("permissions");
		let original_inode = std::fs::metadata(&path).expect("before").ino();
		let context = ToolContext {
			call_id: uuid::Uuid::now_v7().to_string(),
			workspace: Arc::new(root(&directory)),
			cancellation: super::super::AgentCancellation::new(),
			approved: true,
		};

		write_file(
			&context,
			serde_json::json!({"path": "note.txt", "content": "first"}),
		)
		.expect("atomic write");
		let written_inode = std::fs::metadata(&path).expect("written").ino();
		edit_file(
			&context,
			serde_json::json!({
				"path": "note.txt",
				"old_text": "first",
				"new_text": "second"
			}),
		)
		.expect("atomic edit");
		let metadata = std::fs::metadata(&path).expect("edited");

		assert_ne!(original_inode, written_inode);
		assert_ne!(written_inode, metadata.ino());
		assert_eq!(metadata.mode() & 0o777, 0o640);
		assert_eq!(std::fs::read_to_string(path).expect("read"), "second");
		assert!(
			std::fs::read_dir(directory.path())
				.expect("directory")
				.all(|entry| !entry
					.expect("entry")
					.file_name()
					.to_string_lossy()
					.starts_with(".emelex-tmp-"))
		);
	}

	#[test]
	fn failed_swap_rollback_never_unlinks_recoverable_original() {
		let directory = tempfile::tempdir().expect("tempdir");
		let path = directory.path().join("file.txt");
		std::fs::write(&path, "original").expect("seed");
		let workspace = root(&directory);
		let target = workspace
			.mutation_target("file.txt", true)
			.expect("mutation target");
		let expected = open_mutation_existing(&target)
			.expect("open target")
			.expect("existing target");
		inject_swap_failure_after(1);
		inject_metadata_failure_once();

		let error =
			atomic_replace(&target, Some(&expected), b"replacement").expect_err("rollback failure");

		assert!(error.to_string().contains("original may remain"));
		let recovery = std::fs::read_dir(directory.path())
			.expect("directory")
			.filter_map(Result::ok)
			.find(|entry| {
				entry
					.file_name()
					.to_string_lossy()
					.starts_with(".emelex-tmp-")
			})
			.expect("preserved recovery entry");
		assert_eq!(
			std::fs::read_to_string(recovery.path()).expect("recovery content"),
			"original"
		);
	}

	#[test]
	fn directory_descriptor_listing_is_bounded_and_sorted_by_caller() {
		let directory = tempfile::tempdir().expect("tempdir");
		std::fs::write(directory.path().join("b"), "b").expect("write b");
		std::fs::write(directory.path().join("a"), "a").expect("write a");
		let workspace = root(&directory);
		let (descriptor, path) = workspace
			.open_directory(".", false)
			.expect("open directory");
		let entries = read_directory_entries(&descriptor, 2, &path).expect("list");
		let unknown_type_name = CString::new("a").expect("entry name");
		let detected =
			directory_entry_kind(&descriptor, &unknown_type_name, libc::DT_UNKNOWN, &path)
				.expect("fallback type");

		assert_eq!(entries.len(), 2);
		assert_eq!(detected, "file");
	}

	#[test]
	fn wildcard_matching_supports_star_and_question_mark() {
		assert!(wildcard_matches("*.rs", "agent.rs"));
		assert!(wildcard_matches("a?ent.*", "agent.rs"));
		assert!(!wildcard_matches("*.md", "agent.rs"));
	}

	#[test]
	fn likely_secret_paths_require_approval() {
		let root = Path::new("/workspace");
		assert!(read_approval_reason(root, ".env").is_some());
		assert!(read_approval_reason(root, "/tmp/token.txt").is_some());
		assert!(read_approval_reason(root, "config/client_secret.json").is_some());
		assert!(read_approval_reason(root, "src/tokenizer.rs").is_none());
		assert!(read_approval_reason(root, "docs/secretary.md").is_none());
		assert!(read_approval_reason(root, "src/lib.rs").is_none());
	}

	#[tokio::test]
	async fn direct_tool_invocation_enforces_advertised_string_bounds() {
		let directory = tempfile::tempdir().expect("tempdir");
		let context = ToolContext {
			call_id: uuid::Uuid::now_v7().to_string(),
			workspace: Arc::new(root(&directory)),
			cancellation: super::super::AgentCancellation::new(),
			approved: false,
		};
		let cases = [
			(WorkspaceTool::Read, serde_json::json!({"path": ""})),
			(
				WorkspaceTool::List,
				serde_json::json!({"path": "", "max_entries": 1}),
			),
			(
				WorkspaceTool::Find,
				serde_json::json!({"path": ".", "pattern": "x".repeat(257)}),
			),
			(
				WorkspaceTool::Grep,
				serde_json::json!({"path": ".", "query": "x".repeat(4097)}),
			),
		];
		for (tool, arguments) in cases {
			let error = tool
				.invoke(&context, arguments)
				.await
				.expect_err("direct invocation must enforce schema bounds");
			assert!(matches!(error, ToolError::RespondToModel(_)));
		}
	}

	#[tokio::test]
	async fn discovery_reports_sensitive_skips_without_hiding_tokenizer_sources() {
		let directory = tempfile::tempdir().expect("tempdir");
		std::fs::write(directory.path().join("tokenizer.rs"), "safe").expect("safe source");
		std::fs::write(directory.path().join("token.txt"), "credential").expect("secret fixture");
		let context = ToolContext {
			call_id: uuid::Uuid::now_v7().to_string(),
			workspace: Arc::new(root(&directory)),
			cancellation: super::super::AgentCancellation::new(),
			approved: false,
		};

		let output = WorkspaceTool::Find
			.invoke(
				&context,
				serde_json::json!({"path": ".", "pattern": "*", "max_depth": 1}),
			)
			.await
			.expect("find output");
		let value: serde_json::Value =
			serde_json::from_str(&output.content).expect("structured output");

		assert_eq!(value["skipped_sensitive_count"], 1);
		assert!(
			value["matches"]
				.as_array()
				.expect("matches")
				.iter()
				.any(|path| path
					.as_str()
					.is_some_and(|path| path.ends_with("tokenizer.rs")))
		);
		assert!(
			value["skipped_sensitive_paths"]
				.as_array()
				.expect("skipped paths")
				.iter()
				.any(|path| path
					.as_str()
					.is_some_and(|path| path.ends_with("token.txt")))
		);
	}

	#[test]
	fn discovery_caps_every_enumerated_entry_before_queue_growth() {
		let directory = tempfile::tempdir().expect("tempdir");
		std::fs::create_dir(directory.path().join("a")).expect("directory a");
		std::fs::create_dir(directory.path().join("b")).expect("directory b");
		let workspace = root(&directory);
		let cancellation = super::super::AgentCancellation::new();

		let error = discover_files(&workspace, ".", false, 8, 100, 2, &cancellation)
			.expect_err("third enumerated entry must exceed test ceiling");

		assert!(matches!(
			error,
			WorkspaceError::Limit {
				resource: "walk entries",
				limit: 2
			}
		));
	}

	#[test]
	fn fifo_reads_open_nonblocking_before_regular_file_validation() {
		let directory = tempfile::tempdir().expect("tempdir");
		let fifo = directory.path().join("pipe");
		let fifo_name = CString::new(fifo.as_os_str().as_bytes()).expect("FIFO path");
		// SAFETY: path is NUL-terminated and mode contains permission bits.
		assert_eq!(unsafe { libc::mkfifo(fifo_name.as_ptr(), 0o600) }, 0);
		// Keep both FIFO ends open so this test remains nonblocking even if the
		// descriptor under test accidentally drops O_NONBLOCK.
		// SAFETY: path is live and the returned descriptor is checked.
		let guard_descriptor =
			unsafe { libc::open(fifo_name.as_ptr(), libc::O_RDWR | libc::O_NONBLOCK) };
		assert!(guard_descriptor >= 0);
		// SAFETY: successful `open` returned an owned descriptor.
		let _guard = unsafe { File::from_raw_fd(guard_descriptor) };
		let workspace = root(&directory);

		let (file, _) = workspace.open_read("pipe", false).expect("open FIFO");
		// SAFETY: `file` owns a live descriptor and F_GETFL has no third argument.
		let flags = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GETFL) };
		assert!(flags >= 0);
		assert_ne!(flags & libc::O_NONBLOCK, 0);
	}

	#[test]
	fn capped_output_preserves_head_and_tail() {
		let mut output = CappedOutput::new(8);
		output.push(b"0123456789abcdef");
		let rendered = output.render();

		assert!(rendered.starts_with("0123"));
		assert!(rendered.ends_with("cdef"));
		assert!(rendered.contains("8 bytes omitted"));
	}

	#[test]
	fn capped_output_with_zero_limit_retains_no_bytes() {
		let mut output = CappedOutput::new(0);
		output.push(&[b'x'; 16 * 1024]);

		assert!(output.head.is_empty());
		assert!(output.tail.is_empty());
		assert_eq!(output.total, 16 * 1024);
	}

	#[tokio::test]
	async fn shell_one_byte_budget_bounds_zero_limit_stdout_half() {
		let directory = tempfile::tempdir().expect("tempdir");
		let context = ToolContext {
			call_id: uuid::Uuid::now_v7().to_string(),
			workspace: Arc::new(root(&directory)),
			cancellation: super::super::AgentCancellation::new(),
			approved: true,
		};
		let tool = test_shell(5, 1);

		let output = tool
			.invoke(
				&context,
				serde_json::json!({"command": "printf 0123456789"}),
			)
			.await
			.expect("shell output");

		assert!(output.content.len() <= 1);
	}

	#[tokio::test]
	async fn shell_timeout_kills_process_group_and_bounds_output() {
		let directory = tempfile::tempdir().expect("tempdir");
		let context = ToolContext {
			call_id: uuid::Uuid::now_v7().to_string(),
			workspace: Arc::new(root(&directory)),
			cancellation: super::super::AgentCancellation::new(),
			approved: true,
		};
		let tool = test_shell(1, 1024);
		let started = std::time::Instant::now();

		let output = tool
			.invoke(
				&context,
				serde_json::json!({
					"command": "sleep 10 & wait",
					"timeout_seconds": 1
				}),
			)
			.await
			.expect("timeout output");

		assert!(output.is_error);
		assert!(output.content.starts_with("status: timeout"));
		assert!(output.content.len() <= 1024);
		assert!(started.elapsed() < Duration::from_secs(4));
	}

	#[tokio::test]
	async fn normal_shell_leader_exit_kills_background_descendants() {
		let directory = tempfile::tempdir().expect("tempdir");
		let context = ToolContext {
			call_id: uuid::Uuid::now_v7().to_string(),
			workspace: Arc::new(root(&directory)),
			cancellation: super::super::AgentCancellation::new(),
			approved: true,
		};
		let tool = test_shell(5, 1024);

		let output = tool
			.invoke(
				&context,
				serde_json::json!({"command": "sleep 30 & echo $!"}),
			)
			.await
			.expect("shell output");
		let process_id = output
			.content
			.lines()
			.find_map(|line| line.trim().parse::<i32>().ok())
			.expect("background process ID");
		let mut alive = true;
		for _ in 0..40 {
			// SAFETY: signal zero only probes the exact child PID printed by
			// this test; it does not alter the process.
			alive = unsafe { libc::kill(process_id, 0) } == 0;
			if !alive {
				break;
			}
			tokio::time::sleep(Duration::from_millis(25)).await;
		}

		assert!(!alive, "background descendant {process_id} survived");
	}

	#[test]
	fn shell_timeout_hard_limit_accepts_twenty_minutes_only() {
		assert!(shell_tool(MAX_SHELL_TIMEOUT_SECONDS, 1).is_ok());
		assert!(matches!(
			shell_tool(MAX_SHELL_TIMEOUT_SECONDS + 1, 1),
			Err(WorkspaceError::Limit {
				resource: "shell timeout seconds",
				limit: 1_200
			})
		));
	}

	#[tokio::test]
	async fn shell_resolves_executable_from_sanitized_absolute_inherited_path() {
		let directory = tempfile::tempdir().expect("tempdir");
		let bin = directory.path().join("toolchain-bin");
		std::fs::create_dir(&bin).expect("create toolchain bin");
		let executable = bin.join("emelex-path-fixture");
		let mut file = OpenOptions::new()
			.write(true)
			.create_new(true)
			.mode(0o700)
			.open(&executable)
			.expect("create executable");
		file.write_all(b"#!/bin/sh\nprintf inherited-path-ok\n")
			.expect("write executable");
		drop(file);
		let home = directory.path().join("fixture-home");
		std::fs::create_dir(&home).expect("create fixture HOME");
		std::fs::write(
			home.join(".gitconfig"),
			b"[user]\n\tname = Emelex Fixture\n\temail = fixture@example.invalid\n",
		)
		.expect("write fixture Git config");
		let inherited =
			std::env::join_paths([PathBuf::from("."), bin.clone()]).expect("test inherited PATH");
		let path = sanitized_shell_path(Some(&inherited));
		let entries = std::env::split_paths(&path).collect::<Vec<_>>();
		assert!(entries.contains(&bin));
		assert!(entries.iter().all(|entry| entry.is_absolute()));
		assert!(!entries.contains(&PathBuf::from(".")));
		let context = ToolContext {
			call_id: uuid::Uuid::now_v7().to_string(),
			workspace: Arc::new(root(&directory)),
			cancellation: super::super::AgentCancellation::new(),
			approved: true,
		};
		let same_home_different_path = WorkspaceTool::Shell {
			timeout_seconds: 30,
			output_bytes: 1_024,
			path: sanitized_shell_path(None),
			home: Some(home.clone()),
		};
		let same_path_different_home = WorkspaceTool::Shell {
			timeout_seconds: 30,
			output_bytes: 1_024,
			path: path.clone(),
			home: Some(directory.path().to_path_buf()),
		};
		let tool = WorkspaceTool::Shell {
			timeout_seconds: 30,
			output_bytes: 1_024,
			path,
			home: Some(home.clone()),
		};
		let output = tool
			.invoke(
				&context,
				serde_json::json!({
					"command": "git config --global user.name; emelex-path-fixture"
				}),
			)
			.await
			.expect("run inherited executable");

		assert!(!output.is_error);
		assert!(output.content.contains("Emelex Fixture"));
		assert!(output.content.contains("inherited-path-ok"));
		assert!(tool.implementation_identity().contains("path_sha256="));
		assert!(tool.implementation_identity().contains("home_sha256="));
		assert!(
			!tool
				.implementation_identity()
				.contains(bin.to_string_lossy().as_ref())
		);
		assert!(
			!tool
				.implementation_identity()
				.contains(home.to_string_lossy().as_ref())
		);
		let identity = tool.implementation_identity();
		assert_ne!(identity, same_home_different_path.implementation_identity());
		assert_ne!(identity, same_path_different_home.implementation_identity());
	}
}
