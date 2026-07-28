//! Embedded MLX runtime asset initialization.

use std::{
	ffi::CString,
	fs::{self, File, OpenOptions},
	io::{Read as _, Write as _},
	os::{fd::AsRawFd as _, unix::fs::OpenOptionsExt as _},
	path::{Path, PathBuf},
	sync::{Mutex, OnceLock},
};

use sha2::{Digest as _, Sha256};

const COMPRESSED_METALLIB: &[u8] = include_bytes!(env!("EMELEX_METALLIB_ZST_PATH"));
const METALLIB_DIGEST: &str = env!("EMELEX_METALLIB_SHA256");
const METALLIB_SIZE: &str = env!("EMELEX_METALLIB_SIZE");
const MINIMUM_MACOS: &str = env!("EMELEX_MINIMUM_MACOS");

static RUNTIME: OnceLock<Mutex<Option<RuntimeAsset>>> = OnceLock::new();

/// Runtime asset selected for this process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeAsset {
	home: PathBuf,
	metallib: PathBuf,
	digest: String,
}

impl RuntimeAsset {
	/// Canonical Emelex home that owns this process's runtime.
	pub fn home(&self) -> &Path {
		&self.home
	}

	/// Extracted MLX Metal library.
	pub fn metallib(&self) -> &Path {
		&self.metallib
	}

	/// SHA-256 digest naming this runtime asset.
	pub fn digest(&self) -> &str {
		&self.digest
	}
}

/// Runtime initialization failures.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RuntimeError {
	/// Emelex is only available on supported Apple Silicon systems.
	#[error("unsupported platform: {0}")]
	UnsupportedPlatform(String),
	/// A second Emelex home was requested after runtime initialization.
	#[error(
		"Emelex runtime already uses home {active:?}; requested incompatible home {requested:?}"
	)]
	HomeConflict {
		/// First successfully initialized home.
		active: PathBuf,
		/// Later requested home.
		requested: PathBuf,
	},
	/// Runtime storage failed.
	#[error("runtime storage {operation} failed for {path:?}: {source}")]
	Storage {
		/// Operation being attempted.
		operation: &'static str,
		/// Affected path.
		path: PathBuf,
		/// Underlying I/O failure.
		#[source]
		source: std::io::Error,
	},
	/// Runtime asset path cannot cross the mlx-c UTF-8 ABI.
	#[error("runtime path is not valid UTF-8: {0:?}")]
	InvalidPath(PathBuf),
	/// Embedded runtime data was corrupt.
	#[error("embedded metallib verification failed: {0}")]
	CorruptAsset(String),
	/// Emelex home preparation failed.
	#[error(transparent)]
	Home(#[from] crate::home::HomeError),
	/// MLX rejected runtime configuration.
	#[error("MLX runtime initialization failed: {0}")]
	Mlx(String),
	/// No Metal device is present in the current environment.
	#[error("no Metal device is available")]
	MetalDeviceUnavailable,
}

/// Install the embedded metallib for `home` before any MLX object is created.
///
/// The first successful home wins for the life of the process. Repeating the
/// same home is idempotent; requesting another home returns
/// [`RuntimeError::HomeConflict`].
///
/// # Errors
///
/// Returns an error for unsupported macOS versions, unsafe runtime paths,
/// corrupt embedded data, I/O failures, or MLX initialization failure.
pub fn initialize(home: &Path) -> Result<RuntimeAsset, RuntimeError> {
	ensure_supported_platform()?;
	let prepared_home = crate::home::EmelexHome::prepare(home)?;
	let canonical_home = prepared_home.root().to_path_buf();
	let state = RUNTIME.get_or_init(|| Mutex::new(None));
	let mut selected = state
		.lock()
		.unwrap_or_else(std::sync::PoisonError::into_inner);
	if let Some(active) = selected.as_ref() {
		if active.home == canonical_home {
			return Ok(active.clone());
		}
		return Err(RuntimeError::HomeConflict {
			active: active.home.clone(),
			requested: canonical_home,
		});
	}
	let metallib = extract_metallib(&canonical_home)?;
	crate::engine::error::install_error_handler();
	let text = metallib
		.to_str()
		.ok_or_else(|| RuntimeError::InvalidPath(metallib.clone()))?;
	let path = CString::new(text).map_err(|_| RuntimeError::InvalidPath(metallib.clone()))?;
	// SAFETY: the C string remains alive for the call. The patched mlx-c
	// function copies it into process-owned storage and reports failures.
	let status = unsafe { crate::engine::sys::mlx_metal_set_default_library_path(path.as_ptr()) };
	crate::engine::error::check(status).map_err(runtime_mlx_error)?;
	let asset = RuntimeAsset {
		home: canonical_home,
		metallib,
		digest: METALLIB_DIGEST.to_string(),
	};
	*selected = Some(asset.clone());
	drop(selected);
	Ok(asset)
}

/// Ensure internal engine entry points have a stable runtime.
///
/// Public callers should select a home explicitly before model loading. This
/// fallback exists for crate-internal engine tests and always respects an
/// already-selected process home.
pub(crate) fn initialize_default_if_needed() -> Result<RuntimeAsset, RuntimeError> {
	let state = RUNTIME.get_or_init(|| Mutex::new(None));
	let active = {
		let selected = state
			.lock()
			.unwrap_or_else(std::sync::PoisonError::into_inner);
		selected.as_ref().cloned()
	};
	if let Some(active) = active {
		return Ok(active);
	}
	let home = fallback_home()?;
	initialize(home.root())
}

#[cfg(not(test))]
fn fallback_home() -> Result<crate::home::EmelexHome, RuntimeError> {
	Ok(crate::home::EmelexHome::resolve(None)?)
}

#[cfg(test)]
fn fallback_home() -> Result<crate::home::EmelexHome, RuntimeError> {
	struct TestHome {
		_directory: tempfile::TempDir,
		home: crate::home::EmelexHome,
	}

	static TEST_HOME: OnceLock<Result<TestHome, String>> = OnceLock::new();
	let selected = TEST_HOME.get_or_init(|| {
		let directory = tempfile::tempdir()
			.map_err(|error| format!("cannot create process-local Emelex test home: {error}"))?;
		let home = crate::home::EmelexHome::prepare(&directory.path().join("home"))
			.map_err(|error| format!("cannot prepare Emelex test home: {error}"))?;
		Ok(TestHome {
			_directory: directory,
			home,
		})
	});
	selected
		.as_ref()
		.map(|home| home.home.clone())
		.map_err(|message| RuntimeError::CorruptAsset(message.clone()))
}

/// Force one tiny MLX GPU evaluation to verify the embedded metallib loader.
///
/// Call this after [`initialize`] when validating an installation. Successful
/// completion proves the configured runtime asset can create MLX's default
/// Metal stream, load a kernel, and execute it.
///
/// # Errors
///
/// Returns a runtime error when initialization or the MLX evaluation fails.
pub fn verify_engine() -> Result<(), RuntimeError> {
	initialize_default_if_needed()?;
	let one = crate::engine::array::Array::scalar_f32(1.0).map_err(runtime_mlx_error)?;
	let doubled = crate::engine::ops::add(&one, &one).map_err(runtime_mlx_error)?;
	doubled.eval().map_err(runtime_mlx_error)
}

/// Return Metal's recommended maximum working-set size in bytes.
///
/// This query creates a temporary Metal device but does not initialize the MLX
/// singleton or load the metallib.
///
/// # Errors
///
/// Returns an error when no Metal device exists or mlx-c rejects the query.
pub fn recommended_max_working_set_size() -> Result<u64, RuntimeError> {
	crate::engine::error::install_error_handler();
	let mut bytes = 0_u64;
	// SAFETY: `bytes` is a valid writable result pointer for this call.
	let status =
		unsafe { crate::engine::sys::mlx_metal_recommended_max_working_set_size(&raw mut bytes) };
	crate::engine::error::check(status).map_err(runtime_mlx_error)?;
	Ok(bytes)
}

fn runtime_mlx_error(error: impl std::fmt::Display) -> RuntimeError {
	let message = error.to_string();
	if message.to_ascii_lowercase().contains("no metal device") {
		RuntimeError::MetalDeviceUnavailable
	} else {
		RuntimeError::Mlx(message)
	}
}

fn extract_metallib(home: &Path) -> Result<PathBuf, RuntimeError> {
	let intended = home.join("cache/runtime/mlx").join(METALLIB_DIGEST);
	let directory =
		crate::home::create_owner_subdir(home, &["cache", "runtime", "mlx", METALLIB_DIGEST])
			.map_err(|source| storage("create runtime directory", &intended, source))?;
	let destination = directory.join("mlx.metallib");
	if valid_metallib(&destination)? {
		return Ok(destination);
	}
	let temporary = directory.join(format!(
		".mlx.metallib.{}.{}.tmp",
		std::process::id(),
		uuid::Uuid::now_v7()
	));
	let result = write_metallib(&temporary);
	if let Err(error) = result {
		let _ = fs::remove_file(&temporary);
		return Err(error);
	}
	if let Err(source) = fs::rename(&temporary, &destination) {
		let _ = fs::remove_file(&temporary);
		return Err(storage("publish runtime metallib", &destination, source));
	}
	let directory_file = File::open(&directory)
		.map_err(|source| storage("open runtime directory", &directory, source))?;
	directory_file
		.sync_all()
		.map_err(|source| storage("sync runtime directory", &directory, source))?;
	if !valid_metallib(&destination)? {
		return Err(RuntimeError::CorruptAsset(format!(
			"published metallib does not match {METALLIB_DIGEST}"
		)));
	}
	Ok(destination)
}

fn write_metallib(path: &Path) -> Result<(), RuntimeError> {
	let mut output = OpenOptions::new()
		.create_new(true)
		.write(true)
		.mode(0o600)
		.open(path)
		.map_err(|source| storage("create temporary metallib", path, source))?;
	let mut decoder = zstd::stream::read::Decoder::new(COMPRESSED_METALLIB)
		.map_err(|error| RuntimeError::CorruptAsset(error.to_string()))?;
	let mut digest = Sha256::new();
	let mut length = 0_u64;
	let mut buffer = vec![0_u8; 64 * 1024];
	loop {
		let read = decoder
			.read(&mut buffer)
			.map_err(|error| RuntimeError::CorruptAsset(error.to_string()))?;
		if read == 0 {
			break;
		}
		digest.update(&buffer[..read]);
		length = length.saturating_add(read as u64);
		output
			.write_all(&buffer[..read])
			.map_err(|source| storage("write temporary metallib", path, source))?;
	}
	output
		.sync_all()
		.map_err(|source| storage("sync temporary metallib", path, source))?;
	verify_digest(length, &hex::encode(digest.finalize()))
}

fn valid_metallib(path: &Path) -> Result<bool, RuntimeError> {
	let mut file = match OpenOptions::new()
		.read(true)
		.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK)
		.open(path)
	{
		Ok(file) => file,
		Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
		Err(error) if error.raw_os_error() == Some(libc::ELOOP) => return Ok(false),
		Err(source) => return Err(storage("open cached metallib", path, source)),
	};
	let metadata = file
		.metadata()
		.map_err(|source| storage("inspect cached metallib", path, source))?;
	if !metadata.file_type().is_file() {
		return Ok(false);
	}
	let expected = METALLIB_SIZE
		.parse::<u64>()
		.map_err(|_| RuntimeError::CorruptAsset("invalid embedded metallib size".to_string()))?;
	if metadata.len() != expected {
		return Ok(false);
	}
	// SAFETY: `file` owns a valid descriptor for the regular cache file.
	if unsafe { libc::fchmod(file.as_raw_fd(), 0o600) } != 0 {
		return Err(storage(
			"set cached metallib permissions",
			path,
			std::io::Error::last_os_error(),
		));
	}
	let mut digest = Sha256::new();
	let mut copied = 0_u64;
	let mut buffer = vec![0_u8; 64 * 1024];
	loop {
		let read = file
			.read(&mut buffer)
			.map_err(|source| storage("hash cached metallib", path, source))?;
		if read == 0 {
			break;
		}
		digest.update(&buffer[..read]);
		copied = copied.saturating_add(read as u64);
	}
	Ok(copied == expected && hex::encode(digest.finalize()) == METALLIB_DIGEST)
}

fn verify_digest(length: u64, digest: &str) -> Result<(), RuntimeError> {
	let expected = METALLIB_SIZE
		.parse::<u64>()
		.map_err(|_| RuntimeError::CorruptAsset("invalid embedded metallib size".to_string()))?;
	if length != expected || digest != METALLIB_DIGEST {
		return Err(RuntimeError::CorruptAsset(format!(
			"expected {METALLIB_DIGEST}/{expected} bytes, got {digest}/{length} bytes"
		)));
	}
	Ok(())
}

fn storage(operation: &'static str, path: &Path, source: std::io::Error) -> RuntimeError {
	RuntimeError::Storage {
		operation,
		path: path.to_path_buf(),
		source,
	}
}

#[cfg(target_os = "macos")]
fn ensure_supported_platform() -> Result<(), RuntimeError> {
	let version = macos_product_version()?;
	if version < (26, 5) {
		return Err(RuntimeError::UnsupportedPlatform(format!(
			"macOS {}.{} is older than required {MINIMUM_MACOS}",
			version.0, version.1
		)));
	}
	Ok(())
}

#[cfg(not(target_os = "macos"))]
fn ensure_supported_platform() -> Result<(), RuntimeError> {
	Err(RuntimeError::UnsupportedPlatform(
		"Emelex 1.0 requires Apple Silicon macOS".to_string(),
	))
}

#[cfg(target_os = "macos")]
fn macos_product_version() -> Result<(u32, u32), RuntimeError> {
	let name = c"kern.osproductversion";
	let mut size = 0_usize;
	// SAFETY: first sysctl call uses a valid name and asks only for size.
	let status = unsafe {
		libc::sysctlbyname(
			name.as_ptr(),
			std::ptr::null_mut(),
			&raw mut size,
			std::ptr::null_mut(),
			0,
		)
	};
	if status != 0 || size == 0 {
		return Err(RuntimeError::UnsupportedPlatform(
			"could not query macOS product version".to_string(),
		));
	}
	let mut buffer = vec![0_u8; size];
	// SAFETY: buffer has `size` writable bytes and sysctl updates that size.
	let status = unsafe {
		libc::sysctlbyname(
			name.as_ptr(),
			buffer.as_mut_ptr().cast(),
			&raw mut size,
			std::ptr::null_mut(),
			0,
		)
	};
	if status != 0 {
		return Err(RuntimeError::UnsupportedPlatform(
			"could not read macOS product version".to_string(),
		));
	}
	let text = String::from_utf8_lossy(&buffer[..size])
		.trim_end_matches('\0')
		.to_string();
	let mut parts = text.split('.');
	let major = parts
		.next()
		.and_then(|part| part.parse::<u32>().ok())
		.ok_or_else(|| RuntimeError::UnsupportedPlatform(text.clone()))?;
	let minor = parts
		.next()
		.unwrap_or("0")
		.parse::<u32>()
		.map_err(|_| RuntimeError::UnsupportedPlatform(text))?;
	Ok((major, minor))
}
