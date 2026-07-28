//! Bounded, no-follow reads for model-owned runtime artifacts.

use std::{
	fs::{File, OpenOptions},
	io::{Read as _, Take},
	os::unix::fs::OpenOptionsExt as _,
	path::Path,
};

/// Maximum accepted `config.json` or generation-config bytes.
pub const MAX_MODEL_CONFIG_BYTES: u64 = 16 << 20;
/// Maximum accepted tokenizer JSON bytes.
pub const MAX_TOKENIZER_BYTES: u64 = 256 << 20;
/// Maximum accepted tokenizer configuration bytes.
pub const MAX_TOKENIZER_CONFIG_BYTES: u64 = 16 << 20;
/// Maximum accepted standalone chat-template bytes.
pub const MAX_CHAT_TEMPLATE_BYTES: u64 = 1 << 20;

/// Open one existing regular file without following its final symlink.
pub fn open_regular(path: &Path) -> std::io::Result<File> {
	let file = OpenOptions::new()
		.read(true)
		.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK)
		.open(path)?;
	if !file.metadata()?.file_type().is_file() {
		return Err(std::io::Error::new(
			std::io::ErrorKind::InvalidData,
			"artifact is not a regular file",
		));
	}
	Ok(file)
}

/// Read one regular file through one descriptor with an enforced byte cap.
pub fn read_bytes(path: &Path, limit: u64) -> std::io::Result<Vec<u8>> {
	read_open_file(open_regular(path)?, limit)
}

/// Read one optional regular file through one descriptor with a byte cap.
pub fn read_optional_bytes(path: &Path, limit: u64) -> std::io::Result<Option<Vec<u8>>> {
	match open_regular(path) {
		Ok(file) => read_open_file(file, limit).map(Some),
		Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
		Err(error) => Err(error),
	}
}

/// Read optional bounded UTF-8.
pub fn read_optional_utf8(path: &Path, limit: u64) -> std::io::Result<Option<String>> {
	read_optional_bytes(path, limit)?
		.map(|bytes| {
			String::from_utf8(bytes)
				.map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))
		})
		.transpose()
}

fn read_open_file(file: File, limit: u64) -> std::io::Result<Vec<u8>> {
	let capacity = file
		.metadata()?
		.len()
		.min(limit)
		.try_into()
		.unwrap_or(usize::MAX);
	let mut bytes = Vec::with_capacity(capacity);
	let mut bounded: Take<File> = file.take(limit.saturating_add(1));
	bounded.read_to_end(&mut bytes)?;
	if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > limit {
		return Err(std::io::Error::new(
			std::io::ErrorKind::InvalidData,
			format!("artifact exceeds {limit} byte limit"),
		));
	}
	Ok(bytes)
}
