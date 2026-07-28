//! Emelex home resolution and owner-only storage layout.

use std::{
	env,
	ffi::{CStr, CString, OsString},
	fs::{self, File, OpenOptions},
	io::{Read as _, Write as _},
	mem::MaybeUninit,
	os::{
		fd::{AsRawFd as _, FromRawFd as _},
		unix::{
			ffi::OsStrExt as _,
			fs::{MetadataExt as _, OpenOptionsExt as _},
		},
	},
	path::{Path, PathBuf},
};

const HOME_MARKER_NAME: &str = ".emelex-root";
const HOME_MARKER_CONTENT: &[u8] = b"emelex-home-v1\n";
const SNAPSHOT_MUTATION_LOCK_NAME: &str = ".snapshot-mutations.lock";

/// Effective user ID used for owner-only storage checks.
pub(crate) fn effective_user_id() -> u32 {
	// SAFETY: `geteuid` takes no arguments, has no preconditions, and returns
	// process credential state without borrowing caller-owned memory.
	unsafe { libc::geteuid() }
}
const ACL_TYPE_EXTENDED: libc::c_int = 0x0000_0100;
const ACL_FIRST_ENTRY: libc::c_int = 0;

type Acl = *mut libc::c_void;
type AclEntry = *mut libc::c_void;

unsafe extern "C" {
	fn acl_free(object: *mut libc::c_void) -> libc::c_int;
	fn acl_delete_entry(acl: Acl, entry: AclEntry) -> libc::c_int;
	fn acl_get_fd_np(descriptor: libc::c_int, acl_type: libc::c_int) -> Acl;
	fn acl_get_entry(acl: Acl, entry_id: libc::c_int, entry: *mut AclEntry) -> libc::c_int;
	fn acl_set_fd_np(descriptor: libc::c_int, acl: Acl, acl_type: libc::c_int) -> libc::c_int;
}

struct OwnedAcl(Acl);

impl Drop for OwnedAcl {
	fn drop(&mut self) {
		// SAFETY: this guard owns the ACL allocated by an acl_* function.
		unsafe {
			acl_free(self.0);
		}
	}
}

/// Root of all Emelex-owned persistent and temporary data.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EmelexHome {
	root: PathBuf,
}

impl EmelexHome {
	/// Resolve and prepare an Emelex home.
	///
	/// Precedence is `explicit`, then `EMELEX_HOME`, then `~/.emelex`.
	///
	/// # Errors
	///
	/// Returns an error when no home directory can be determined or the
	/// owner-only directory layout cannot be created.
	pub fn resolve(explicit: Option<&Path>) -> Result<Self, HomeError> {
		let requested = requested_root(explicit, env::var_os("EMELEX_HOME"), env::var_os("HOME"))?;
		Self::prepare(&requested)
	}

	/// Prepare an explicitly selected home path.
	///
	/// # Errors
	///
	/// Returns an error when the path or its standard subdirectories cannot be
	/// created, secured, and canonicalized.
	pub fn prepare(path: &Path) -> Result<Self, HomeError> {
		let (root, directory) = prepare_root(path)?;
		for relative in ["cache", "memory", "models", "temp"] {
			create_owner_subdir_from(&directory, &root, &[relative]).map_err(|source| {
				HomeError::Io {
					operation: "create owner-only subdirectory",
					path: root.join(relative),
					source,
				}
			})?;
		}
		directory.sync_all().map_err(|source| HomeError::Io {
			operation: "sync home directory",
			path: root.clone(),
			source,
		})?;
		Ok(Self { root })
	}

	/// Canonical home root.
	pub fn root(&self) -> &Path {
		&self.root
	}

	/// Global configuration file.
	pub fn config_file(&self) -> PathBuf {
		self.root.join("config.toml")
	}

	/// Installed model root.
	pub fn models_dir(&self) -> PathBuf {
		self.root.join("models")
	}

	/// Download, quarantine, and runtime caches.
	pub fn cache_dir(&self) -> PathBuf {
		self.root.join("cache")
	}

	/// Durable session and Knowledge database.
	pub fn database_file(&self) -> PathBuf {
		self.root.join("memory/emelex.sqlite3")
	}

	/// Emelex-owned temporary storage.
	pub fn temp_dir(&self) -> PathBuf {
		self.root.join("temp")
	}

	/// Serialize exact-model binding, quarantine, and permanent deletion.
	///
	/// The process-wide and cross-process lock closes the gap between checking
	/// durable Session references and mutating the corresponding model path.
	pub(crate) fn lock_snapshot_mutations(&self) -> Result<SnapshotMutationLock, HomeError> {
		SnapshotMutationLock::acquire(self)
	}

	pub(crate) fn try_lock_snapshot_mutations(
		&self,
	) -> Result<Option<SnapshotMutationLock>, HomeError> {
		SnapshotMutationLock::try_acquire(self)
	}
}

fn requested_root(
	explicit: Option<&Path>,
	emelex_home: Option<OsString>,
	user_home: Option<OsString>,
) -> Result<PathBuf, HomeError> {
	if let Some(path) = explicit {
		return Ok(path.to_path_buf());
	}
	if let Some(path) = emelex_home.filter(|path| !path.is_empty()) {
		return Ok(PathBuf::from(path));
	}
	let user_home = user_home
		.filter(|path| !path.is_empty())
		.ok_or(HomeError::HomeUnavailable)?;
	Ok(PathBuf::from(user_home).join(".emelex"))
}

impl AsRef<Path> for EmelexHome {
	fn as_ref(&self) -> &Path {
		self.root()
	}
}

/// Emelex home resolution failures.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HomeError {
	/// Neither an explicit home nor a usable environment home exists.
	#[error("cannot determine Emelex home; pass an explicit path or set EMELEX_HOME")]
	HomeUnavailable,
	/// The selected path is not safe to adopt as Emelex-owned storage.
	#[error("refusing unsafe Emelex home {path:?}: {reason}")]
	UnsafePath {
		/// Rejected path.
		path: PathBuf,
		/// Reason adoption was unsafe.
		reason: String,
	},
	/// Home storage could not be prepared.
	#[error("{operation} failed for {path:?}: {source}")]
	Io {
		/// Operation being attempted.
		operation: &'static str,
		/// Affected path.
		path: PathBuf,
		/// Underlying I/O failure.
		#[source]
		source: std::io::Error,
	},
}

/// Held cross-process authority to mutate snapshot bindings or storage.
pub(crate) struct SnapshotMutationLock(File);

impl SnapshotMutationLock {
	fn acquire(home: &EmelexHome) -> Result<Self, HomeError> {
		let (file, path) = Self::open(home)?;
		loop {
			// SAFETY: `file` owns a live descriptor and `LOCK_EX` is valid.
			if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } == 0 {
				return Ok(Self(file));
			}
			let source = std::io::Error::last_os_error();
			if source.kind() != std::io::ErrorKind::Interrupted {
				return Err(HomeError::Io {
					operation: "lock snapshot mutations",
					path,
					source,
				});
			}
		}
	}

	fn try_acquire(home: &EmelexHome) -> Result<Option<Self>, HomeError> {
		let (file, path) = Self::open(home)?;
		loop {
			// SAFETY: `file` owns a live descriptor and both flock flags are
			// valid; failure leaves descriptor ownership with `file`.
			if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
				return Ok(Some(Self(file)));
			}
			let source = std::io::Error::last_os_error();
			match source.kind() {
				std::io::ErrorKind::Interrupted => {}
				std::io::ErrorKind::WouldBlock => return Ok(None),
				_ => {
					return Err(HomeError::Io {
						operation: "try lock snapshot mutations",
						path,
						source,
					});
				}
			}
		}
	}

	fn open(home: &EmelexHome) -> Result<(File, PathBuf), HomeError> {
		let path = home.root.join(SNAPSHOT_MUTATION_LOCK_NAME);
		let file = OpenOptions::new()
			.read(true)
			.write(true)
			.create(true)
			.mode(0o600)
			.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
			.open(&path)
			.map_err(|source| HomeError::Io {
				operation: "open snapshot mutation lock",
				path: path.clone(),
				source,
			})?;
		let metadata = file.metadata().map_err(|source| HomeError::Io {
			operation: "inspect snapshot mutation lock",
			path: path.clone(),
			source,
		})?;
		if !metadata.is_file()
			|| metadata.uid() != effective_user_id()
			|| metadata.mode() & 0o777 != 0o600
			|| has_extended_acl(&file).map_err(|source| HomeError::Io {
				operation: "inspect snapshot mutation lock ACL",
				path: path.clone(),
				source,
			})? {
			return Err(HomeError::UnsafePath {
				path,
				reason: "snapshot mutation lock must be an owner-only regular file without an ACL"
					.to_string(),
			});
		}
		Ok((file, path))
	}
}

impl Drop for SnapshotMutationLock {
	fn drop(&mut self) {
		// SAFETY: the descriptor remains live until this guard finishes dropping.
		unsafe {
			libc::flock(self.0.as_raw_fd(), libc::LOCK_UN);
		}
	}
}

#[allow(
	clippy::too_many_lines,
	reason = "security-critical descriptor workflow stays linear for auditability"
)]
fn prepare_root(path: &Path) -> Result<(PathBuf, File), HomeError> {
	if path.as_os_str().is_empty() {
		return Err(unsafe_path(path, "path is empty"));
	}
	let absolute = std::path::absolute(path).map_err(|source| HomeError::Io {
		operation: "make home path absolute",
		path: path.to_path_buf(),
		source,
	})?;
	let name = absolute
		.file_name()
		.filter(|name| !name.is_empty() && *name != "." && *name != "..")
		.ok_or_else(|| unsafe_path(&absolute, "filesystem roots and broad paths are forbidden"))?;
	let parent = absolute
		.parent()
		.ok_or_else(|| unsafe_path(&absolute, "home has no parent directory"))?;
	let parent = fs::canonicalize(parent).map_err(|source| HomeError::Io {
		operation: "canonicalize home parent",
		path: parent.to_path_buf(),
		source,
	})?;
	let parent_directory = open_directory(&parent).map_err(|source| HomeError::Io {
		operation: "open home parent without following symlinks",
		path: parent.clone(),
		source,
	})?;
	let name_c = CString::new(name.as_bytes())
		.map_err(|_| unsafe_path(&absolute, "home name contains an interior NUL byte"))?;

	// SAFETY: parent descriptor and single-component C string are valid.
	let mkdir_result =
		unsafe { libc::mkdirat(parent_directory.as_raw_fd(), name_c.as_ptr(), 0o700) };
	let created = if mkdir_result == 0 {
		true
	} else {
		let source = std::io::Error::last_os_error();
		if source.kind() != std::io::ErrorKind::AlreadyExists {
			return Err(HomeError::Io {
				operation: "create dedicated home directory",
				path: parent.join(name),
				source,
			});
		}
		false
	};
	let root = parent.join(name);
	let directory =
		open_directory_at(&parent_directory, &name_c).map_err(|source| HomeError::Io {
			operation: "open home without following symlinks",
			path: root.clone(),
			source,
		})?;
	let initial_stat = descriptor_stat(&directory).map_err(|source| HomeError::Io {
		operation: "inspect opened home before locking",
		path: root.clone(),
		source,
	})?;
	if initial_stat.st_uid != effective_user_id() {
		return Err(unsafe_path(
			&root,
			"directory is not owned by the current user",
		));
	}
	lock_home_preparation(&directory, &root)?;
	let stat = descriptor_stat(&directory).map_err(|source| HomeError::Io {
		operation: "inspect opened home",
		path: root.clone(),
		source,
	})?;
	if stat.st_uid != effective_user_id() {
		return Err(unsafe_path(
			&root,
			"directory is not owned by the current user",
		));
	}
	let mode = stat.st_mode & 0o777;
	if created {
		clear_extended_acl(&directory).map_err(|source| HomeError::Io {
			operation: "clear inherited home ACL",
			path: root.clone(),
			source,
		})?;
		// SAFETY: directory is a newly-created, owned directory descriptor.
		if unsafe { libc::fchmod(directory.as_raw_fd(), 0o700) } != 0 {
			return Err(HomeError::Io {
				operation: "secure new home directory",
				path: root,
				source: std::io::Error::last_os_error(),
			});
		}
	} else {
		if has_extended_acl(&directory).map_err(|source| HomeError::Io {
			operation: "inspect existing home ACL",
			path: root.clone(),
			source,
		})? {
			return Err(unsafe_path(&root, "existing directory has an extended ACL"));
		}
		if mode != 0o700 {
			return Err(unsafe_path(
				&root,
				&format!("existing directory mode is {mode:#o}, expected 0o700"),
			));
		}
	}

	let canonical = fs::canonicalize(&root).map_err(|source| HomeError::Io {
		operation: "canonicalize opened home",
		path: root.clone(),
		source,
	})?;
	let canonical_metadata = fs::metadata(&canonical).map_err(|source| HomeError::Io {
		operation: "inspect canonical home",
		path: canonical.clone(),
		source,
	})?;
	let opened_device = u64::try_from(stat.st_dev)
		.map_err(|_| unsafe_path(&root, "opened directory has a negative device identifier"))?;
	if canonical_metadata.dev() != opened_device || canonical_metadata.ino() != stat.st_ino {
		return Err(unsafe_path(
			&root,
			"directory identity changed while it was being prepared",
		));
	}

	if created {
		create_home_marker(&directory, &canonical)?;
	} else if !validate_home_marker(&directory, &canonical)? {
		if !directory_is_empty(&directory).map_err(|source| HomeError::Io {
			operation: "inspect existing home contents",
			path: canonical.clone(),
			source,
		})? {
			return Err(unsafe_path(
				&canonical,
				"existing nonempty directory has no valid Emelex ownership marker",
			));
		}
		create_home_marker(&directory, &canonical)?;
	}
	Ok((canonical, directory))
}

fn lock_home_preparation(directory: &File, root: &Path) -> Result<(), HomeError> {
	loop {
		// SAFETY: `directory` owns a live descriptor and `LOCK_EX` is valid.
		// The returned root descriptor retains this lock through standard
		// subdirectory preparation and its final directory sync.
		if unsafe { libc::flock(directory.as_raw_fd(), libc::LOCK_EX) } == 0 {
			return Ok(());
		}
		let source = std::io::Error::last_os_error();
		if source.kind() != std::io::ErrorKind::Interrupted {
			return Err(HomeError::Io {
				operation: "lock home preparation",
				path: root.to_path_buf(),
				source,
			});
		}
	}
}

fn unsafe_path(path: &Path, reason: &str) -> HomeError {
	HomeError::UnsafePath {
		path: path.to_path_buf(),
		reason: reason.to_string(),
	}
}

fn open_directory(path: &Path) -> Result<File, std::io::Error> {
	OpenOptions::new()
		.read(true)
		.custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW)
		.open(path)
}

fn open_directory_at(parent: &File, name: &CString) -> Result<File, std::io::Error> {
	// SAFETY: parent is an open directory and name is a valid component.
	let descriptor = unsafe {
		libc::openat(
			parent.as_raw_fd(),
			name.as_ptr(),
			libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
		)
	};
	if descriptor < 0 {
		return Err(std::io::Error::last_os_error());
	}
	// SAFETY: openat returned a new owned descriptor.
	Ok(unsafe { File::from_raw_fd(descriptor) })
}

fn descriptor_stat(file: &File) -> Result<libc::stat, std::io::Error> {
	let mut stat = MaybeUninit::<libc::stat>::uninit();
	// SAFETY: stat points to writable storage and file owns a valid descriptor.
	if unsafe { libc::fstat(file.as_raw_fd(), stat.as_mut_ptr()) } != 0 {
		return Err(std::io::Error::last_os_error());
	}
	// SAFETY: successful fstat initialized the structure.
	Ok(unsafe { stat.assume_init() })
}

fn create_home_marker(directory: &File, root: &Path) -> Result<(), HomeError> {
	let name = CString::new(HOME_MARKER_NAME).map_err(|_| {
		unsafe_path(
			&root.join(HOME_MARKER_NAME),
			"marker name contains an interior NUL byte",
		)
	})?;
	// SAFETY: directory and marker component are valid.
	let descriptor = unsafe {
		libc::openat(
			directory.as_raw_fd(),
			name.as_ptr(),
			libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_CLOEXEC | libc::O_NOFOLLOW,
			0o600,
		)
	};
	if descriptor < 0 {
		let source = std::io::Error::last_os_error();
		if source.kind() == std::io::ErrorKind::AlreadyExists {
			if validate_home_marker(directory, root)? {
				return Ok(());
			}
			return Err(unsafe_path(
				&root.join(HOME_MARKER_NAME),
				"existing ownership marker is invalid",
			));
		}
		return Err(HomeError::Io {
			operation: "create home ownership marker",
			path: root.join(HOME_MARKER_NAME),
			source,
		});
	}
	// SAFETY: openat returned a new owned descriptor.
	let mut marker = unsafe { File::from_raw_fd(descriptor) };
	clear_extended_acl(&marker).map_err(|source| HomeError::Io {
		operation: "clear inherited marker ACL",
		path: root.join(HOME_MARKER_NAME),
		source,
	})?;
	// SAFETY: marker is a newly-created, owned regular-file descriptor.
	if unsafe { libc::fchmod(marker.as_raw_fd(), 0o600) } != 0 {
		return Err(HomeError::Io {
			operation: "secure home ownership marker",
			path: root.join(HOME_MARKER_NAME),
			source: std::io::Error::last_os_error(),
		});
	}
	marker
		.write_all(HOME_MARKER_CONTENT)
		.and_then(|()| marker.sync_all())
		.map_err(|source| HomeError::Io {
			operation: "write home ownership marker",
			path: root.join(HOME_MARKER_NAME),
			source,
		})?;
	directory.sync_all().map_err(|source| HomeError::Io {
		operation: "sync home ownership marker",
		path: root.to_path_buf(),
		source,
	})
}

fn validate_home_marker(directory: &File, root: &Path) -> Result<bool, HomeError> {
	let name = CString::new(HOME_MARKER_NAME).map_err(|_| {
		unsafe_path(
			&root.join(HOME_MARKER_NAME),
			"marker name contains an interior NUL byte",
		)
	})?;
	// SAFETY: directory and marker component are valid.
	let descriptor = unsafe {
		libc::openat(
			directory.as_raw_fd(),
			name.as_ptr(),
			libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
		)
	};
	if descriptor < 0 {
		let source = std::io::Error::last_os_error();
		if source.kind() == std::io::ErrorKind::NotFound {
			return Ok(false);
		}
		return Err(HomeError::Io {
			operation: "open home ownership marker",
			path: root.join(HOME_MARKER_NAME),
			source,
		});
	}
	// SAFETY: openat returned a new owned descriptor.
	let marker = unsafe { File::from_raw_fd(descriptor) };
	let stat = descriptor_stat(&marker).map_err(|source| HomeError::Io {
		operation: "inspect home ownership marker",
		path: root.join(HOME_MARKER_NAME),
		source,
	})?;
	let expected_type = libc::S_IFREG as libc::mode_t;
	if stat.st_mode & libc::S_IFMT as libc::mode_t != expected_type
		|| stat.st_uid != effective_user_id()
		|| stat.st_mode & 0o777 != 0o600
		|| has_extended_acl(&marker).map_err(|source| HomeError::Io {
			operation: "inspect home ownership marker ACL",
			path: root.join(HOME_MARKER_NAME),
			source,
		})? {
		return Ok(false);
	}
	let mut bytes = Vec::with_capacity(HOME_MARKER_CONTENT.len() + 1);
	marker
		.take((HOME_MARKER_CONTENT.len() + 1) as u64)
		.read_to_end(&mut bytes)
		.map_err(|source| HomeError::Io {
			operation: "read home ownership marker",
			path: root.join(HOME_MARKER_NAME),
			source,
		})?;
	Ok(bytes == HOME_MARKER_CONTENT)
}

fn directory_is_empty(directory: &File) -> Result<bool, std::io::Error> {
	// SAFETY: directory owns a valid descriptor; dup returns a new one.
	let duplicate = unsafe { libc::dup(directory.as_raw_fd()) };
	if duplicate < 0 {
		return Err(std::io::Error::last_os_error());
	}
	// SAFETY: fdopendir takes ownership of duplicate on success.
	let stream = unsafe { libc::fdopendir(duplicate) };
	if stream.is_null() {
		let source = std::io::Error::last_os_error();
		// SAFETY: fdopendir failed and did not take ownership.
		unsafe {
			libc::close(duplicate);
		}
		return Err(source);
	}
	let result = loop {
		// SAFETY: macOS exposes thread-local errno through __error.
		unsafe {
			*libc::__error() = 0;
		}
		// SAFETY: stream is a live DIR pointer until closed below.
		let entry = unsafe { libc::readdir(stream) };
		if entry.is_null() {
			// SAFETY: same thread-local errno set by readdir.
			let code = unsafe { *libc::__error() };
			break if code == 0 {
				Ok(true)
			} else {
				Err(std::io::Error::from_raw_os_error(code))
			};
		}
		// SAFETY: readdir returns a NUL-terminated d_name for a live entry.
		let name = unsafe { CStr::from_ptr((*entry).d_name.as_ptr()) }.to_bytes();
		if name != b"." && name != b".." {
			break Ok(false);
		}
	};
	// SAFETY: stream is live and owns the duplicated descriptor.
	if unsafe { libc::closedir(stream) } != 0 && result.is_ok() {
		return Err(std::io::Error::last_os_error());
	}
	result
}

pub(crate) fn has_extended_acl(file: &File) -> Result<bool, std::io::Error> {
	let Some(acl) = get_extended_acl(file)? else {
		return Ok(false);
	};
	let mut entry: AclEntry = std::ptr::null_mut();
	clear_errno();
	// SAFETY: acl is live and entry points to writable pointer storage.
	match unsafe { acl_get_entry(acl.0, ACL_FIRST_ENTRY, &raw mut entry) } {
		0 if !entry.is_null() => Ok(true),
		0 => Err(std::io::Error::other(
			"acl_get_entry succeeded without returning an entry",
		)),
		-1 if current_errno() == libc::EINVAL => Ok(false),
		_ => Err(std::io::Error::last_os_error()),
	}
}

fn clear_extended_acl(file: &File) -> Result<(), std::io::Error> {
	let Some(acl) = get_extended_acl(file)? else {
		return Ok(());
	};
	loop {
		let mut entry: AclEntry = std::ptr::null_mut();
		clear_errno();
		// SAFETY: acl is live and entry points to writable pointer storage.
		match unsafe { acl_get_entry(acl.0, ACL_FIRST_ENTRY, &raw mut entry) } {
			0 if !entry.is_null() => {
				// SAFETY: entry belongs to this live ACL.
				if unsafe { acl_delete_entry(acl.0, entry) } != 0 {
					return Err(std::io::Error::last_os_error());
				}
			}
			0 => {
				return Err(std::io::Error::other(
					"acl_get_entry succeeded without returning an entry",
				));
			}
			-1 if current_errno() == libc::EINVAL => break,
			_ => return Err(std::io::Error::last_os_error()),
		}
	}
	// SAFETY: descriptor and empty ACL are live; ACL type is valid on macOS.
	if unsafe { acl_set_fd_np(file.as_raw_fd(), acl.0, ACL_TYPE_EXTENDED) } != 0 {
		return Err(std::io::Error::last_os_error());
	}
	if has_extended_acl(file)? {
		return Err(std::io::Error::other(
			"extended ACL remained after clearing it",
		));
	}
	Ok(())
}

fn clear_errno() {
	// SAFETY: macOS exposes thread-local errno through __error.
	unsafe {
		*libc::__error() = 0;
	}
}

fn current_errno() -> libc::c_int {
	// SAFETY: macOS exposes thread-local errno through __error.
	unsafe { *libc::__error() }
}

fn get_extended_acl(file: &File) -> Result<Option<OwnedAcl>, std::io::Error> {
	// SAFETY: descriptor is live and ACL type is valid on macOS.
	let acl = unsafe { acl_get_fd_np(file.as_raw_fd(), ACL_TYPE_EXTENDED) };
	if acl.is_null() {
		let source = std::io::Error::last_os_error();
		if source.raw_os_error() == Some(libc::ENOENT) {
			return Ok(None);
		}
		return Err(source);
	}
	Ok(Some(OwnedAcl(acl)))
}

/// Create real owner-only directory descendants without following symlinks
/// below `root`.
pub(crate) fn create_owner_subdir(
	root: &Path,
	components: &[&str],
) -> Result<PathBuf, std::io::Error> {
	let directory = open_directory(root)?;
	create_owner_subdir_from(&directory, root, components)
}

fn create_owner_subdir_from(
	root_directory: &File,
	root: &Path,
	components: &[&str],
) -> Result<PathBuf, std::io::Error> {
	let mut directory = root_directory.try_clone()?;
	let mut path = root.to_path_buf();
	for component in components {
		if component.is_empty()
			|| *component == "."
			|| *component == ".."
			|| component.contains('/')
		{
			return Err(std::io::Error::new(
				std::io::ErrorKind::InvalidInput,
				format!("unsafe directory component {component:?}"),
			));
		}
		let name = CString::new(*component).map_err(|_| {
			std::io::Error::new(
				std::io::ErrorKind::InvalidInput,
				"directory component contains an interior NUL byte",
			)
		})?;
		// SAFETY: `directory` is an open directory and `name` is a valid,
		// NUL-terminated single path component.
		let status = unsafe { libc::mkdirat(directory.as_raw_fd(), name.as_ptr(), 0o700) };
		let created = status == 0;
		if !created {
			let error = std::io::Error::last_os_error();
			if error.kind() != std::io::ErrorKind::AlreadyExists {
				return Err(error);
			}
		}
		// SAFETY: arguments are valid and the returned descriptor is checked
		// before ownership transfers to `File`.
		let descriptor = unsafe {
			libc::openat(
				directory.as_raw_fd(),
				name.as_ptr(),
				libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
			)
		};
		if descriptor < 0 {
			return Err(std::io::Error::last_os_error());
		}
		// SAFETY: `openat` returned a new owned descriptor.
		let child = unsafe { File::from_raw_fd(descriptor) };
		let stat = descriptor_stat(&child)?;
		if stat.st_uid != effective_user_id() {
			return Err(std::io::Error::new(
				std::io::ErrorKind::PermissionDenied,
				"directory is not owned by the current user",
			));
		}
		if created {
			clear_extended_acl(&child)?;
			// SAFETY: child is a newly-created, owned directory descriptor.
			if unsafe { libc::fchmod(child.as_raw_fd(), 0o700) } != 0 {
				return Err(std::io::Error::last_os_error());
			}
		} else if stat.st_mode & 0o777 != 0o700 || has_extended_acl(&child)? {
			return Err(std::io::Error::new(
				std::io::ErrorKind::PermissionDenied,
				"existing directory is not owner-only mode 0700 without an ACL",
			));
		}
		directory = child;
		path.push(component);
	}
	Ok(path)
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
