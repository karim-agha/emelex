//! Descriptor-stable, bounded media attachment loading.

use std::{
	fs::OpenOptions,
	io::Read as _,
	os::unix::fs::OpenOptionsExt as _,
	path::{Path, PathBuf},
};

use anyhow::{Context as _, bail};
use emelex::generation::Content;

const MAX_ATTACHMENT_BYTES: u64 = 128 << 20;
pub(crate) const MAX_ATTACHMENTS: usize = 64;
pub(crate) const MAX_TOTAL_ATTACHMENT_BYTES: usize = 256 << 20;

/// Loaded attachment with presentation metadata.
pub(crate) struct Attachment {
	pub(crate) path: PathBuf,
	pub(crate) kind: &'static str,
	pub(crate) content: Content,
}

impl Attachment {
	pub(crate) const fn bytes(&self) -> usize {
		match &self.content {
			Content::Image(bytes) | Content::Audio(bytes) | Content::Video(bytes) => bytes.len(),
			Content::Text(text) => text.len(),
			_ => 0,
		}
	}
}

/// Load one regular non-symlink media file through a single descriptor.
pub(crate) fn load(path: &Path) -> anyhow::Result<Attachment> {
	let mut file = OpenOptions::new()
		.read(true)
		.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK)
		.open(path)
		.with_context(|| format!("open attachment {}", path.display()))?;
	let metadata = file
		.metadata()
		.with_context(|| format!("inspect attachment {}", path.display()))?;
	if !metadata.file_type().is_file() {
		bail!("attachment is not a regular file: {}", path.display());
	}
	if metadata.len() == 0 {
		bail!("attachment is empty: {}", path.display());
	}
	if metadata.len() > MAX_ATTACHMENT_BYTES {
		bail!(
			"attachment exceeds {} bytes: {}",
			MAX_ATTACHMENT_BYTES,
			path.display()
		);
	}
	let capacity =
		usize::try_from(metadata.len()).context("attachment size does not fit memory")?;
	let mut bytes = Vec::with_capacity(capacity);
	file.by_ref()
		.take(MAX_ATTACHMENT_BYTES + 1)
		.read_to_end(&mut bytes)
		.with_context(|| format!("read attachment {}", path.display()))?;
	if bytes.len() as u64 > MAX_ATTACHMENT_BYTES {
		bail!(
			"attachment grew beyond {} bytes while reading: {}",
			MAX_ATTACHMENT_BYTES,
			path.display()
		);
	}
	let (kind, content) = classify(path, bytes)?;
	Ok(Attachment {
		path: path.to_path_buf(),
		kind,
		content,
	})
}

/// Load a bounded batch and reject aggregate amplification.
pub(crate) fn load_all(paths: &[PathBuf]) -> anyhow::Result<Vec<Attachment>> {
	if paths.len() > MAX_ATTACHMENTS {
		bail!("at most {MAX_ATTACHMENTS} attachments are accepted");
	}
	let mut total = 0_usize;
	let mut attachments = Vec::with_capacity(paths.len());
	for path in paths {
		let attachment = load(path)?;
		total = total
			.checked_add(attachment.bytes())
			.context("attachment byte count overflow")?;
		if total > MAX_TOTAL_ATTACHMENT_BYTES {
			bail!("aggregate attachments exceed {MAX_TOTAL_ATTACHMENT_BYTES} bytes");
		}
		attachments.push(attachment);
	}
	Ok(attachments)
}

fn classify(path: &Path, bytes: Vec<u8>) -> anyhow::Result<(&'static str, Content)> {
	if supported_image(&bytes) {
		return Ok(("image", Content::Image(bytes)));
	}
	if wav_magic(&bytes) {
		emelex::generation::validate_audio_bytes(&bytes)
			.with_context(|| format!("validate audio attachment {}", path.display()))?;
		return Ok(("audio", Content::Audio(bytes)));
	}
	if unsupported_audio_magic(&bytes)
		|| mime_guess::from_path(path)
			.first_raw()
			.is_some_and(|mime| mime.starts_with("audio/"))
	{
		bail!(
			"unsupported audio encoding for {}; self-contained Emelex accepts PCM16 or float32 WAV",
			path.display()
		);
	}
	if video_magic(&bytes)
		|| mime_guess::from_path(path)
			.first_raw()
			.is_some_and(|mime| mime.starts_with("video/"))
	{
		bail!(
			"video attachments are unavailable in the self-contained runtime; attach extracted \
			 image frames instead"
		);
	}
	match mime_guess::from_path(path).first_raw() {
		Some(mime) if mime.starts_with("image/") => {
			bail!(
				"unsupported or malformed image encoding for {}",
				path.display()
			)
		}
		_ => bail!("cannot infer supported media type for {}", path.display()),
	}
}

fn supported_image(bytes: &[u8]) -> bool {
	matches!(
		image::guess_format(bytes),
		Ok(image::ImageFormat::Jpeg | image::ImageFormat::Png | image::ImageFormat::WebP)
	)
}

fn wav_magic(bytes: &[u8]) -> bool {
	bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WAVE")
}

fn unsupported_audio_magic(bytes: &[u8]) -> bool {
	bytes.starts_with(b"fLaC")
		|| bytes.starts_with(b"ID3")
		|| bytes.starts_with(b"OggS")
		|| bytes
			.get(..2)
			.is_some_and(|prefix| prefix[0] == 0xff && prefix[1] & 0xe0 == 0xe0)
}

fn video_magic(bytes: &[u8]) -> bool {
	bytes.starts_with(&[0x1a, 0x45, 0xdf, 0xa3])
		|| bytes.get(4..8) == Some(b"ftyp")
		|| bytes.starts_with(&[0x00, 0x00, 0x01, 0xba])
}

#[cfg(test)]
mod tests {
	use super::*;

	fn wav(audio_format: u16, bits_per_sample: u16, data: &[u8]) -> Vec<u8> {
		let bytes_per_sample = bits_per_sample / 8;
		let padding = data.len() & 1;
		let mut wav = Vec::new();
		wav.extend_from_slice(b"RIFF");
		wav.extend_from_slice(
			&(36_u32 + u32::try_from(data.len() + padding).expect("WAV data length")).to_le_bytes(),
		);
		wav.extend_from_slice(b"WAVEfmt ");
		wav.extend_from_slice(&16_u32.to_le_bytes());
		wav.extend_from_slice(&audio_format.to_le_bytes());
		wav.extend_from_slice(&1_u16.to_le_bytes());
		wav.extend_from_slice(&16_000_u32.to_le_bytes());
		wav.extend_from_slice(&(16_000_u32 * u32::from(bytes_per_sample)).to_le_bytes());
		wav.extend_from_slice(&bytes_per_sample.to_le_bytes());
		wav.extend_from_slice(&bits_per_sample.to_le_bytes());
		wav.extend_from_slice(b"data");
		wav.extend_from_slice(&(data.len() as u32).to_le_bytes());
		wav.extend_from_slice(data);
		if padding != 0 {
			wav.push(0);
		}
		wav
	}

	#[test]
	fn magic_takes_precedence_over_misleading_extension() {
		let directory = tempfile::tempdir().expect("directory");
		let path = directory.path().join("voice.png");
		std::fs::write(&path, wav(1, 16, &[0, 0])).expect("wav");
		let attachment = load(&path).expect("attachment");
		assert_eq!(attachment.kind, "audio");
		assert!(matches!(attachment.content, Content::Audio(_)));
	}

	#[test]
	fn wav_preflight_rejects_truncation_and_unsupported_sample_formats() {
		let directory = tempfile::tempdir().expect("directory");
		let truncated = directory.path().join("truncated.wav");
		std::fs::write(&truncated, b"RIFF\x04\x00\x00\x00WAVE").expect("truncated WAV");
		assert!(load(&truncated).is_err());

		for (name, fixture) in [
			("adpcm.wav", wav(2, 4, &[0])),
			("pcm8.wav", wav(1, 8, &[0])),
			("pcm24.wav", wav(1, 24, &[0, 0, 0])),
		] {
			let path = directory.path().join(name);
			std::fs::write(&path, fixture).expect("unsupported WAV");
			let error = load(&path)
				.err()
				.expect("unsupported WAV must fail preflight");
			let detail = format!("{error:#}");
			assert!(detail.contains("PCM16 or float32"), "{name}: {detail}");
		}
	}

	#[test]
	fn wav_preflight_accepts_pcm16_and_float32() {
		let directory = tempfile::tempdir().expect("directory");
		for (name, fixture) in [
			("pcm16.wav", wav(1, 16, &[0, 0])),
			("float32.wav", wav(3, 32, &0.0_f32.to_le_bytes())),
		] {
			let path = directory.path().join(name);
			std::fs::write(&path, fixture).expect("supported WAV");
			assert_eq!(load(&path).expect("supported attachment").kind, "audio");
		}
	}

	#[test]
	fn symlinks_and_unknown_files_are_rejected() {
		use std::os::unix::fs::symlink;

		let directory = tempfile::tempdir().expect("directory");
		let target = directory.path().join("payload.bin");
		std::fs::write(&target, b"not media").expect("payload");
		let link = directory.path().join("payload.mp3");
		symlink(&target, &link).expect("symlink");
		assert!(load(&link).is_err());
		assert!(load(&target).is_err());
	}

	#[test]
	fn ambient_codec_formats_and_video_fail_before_model_loading() {
		let directory = tempfile::tempdir().expect("directory");
		let mp3 = directory.path().join("voice.mp3");
		std::fs::write(&mp3, b"ID3unsupported").expect("mp3 fixture");
		let mp3_error = load(&mp3)
			.err()
			.expect("ambient MP3 decoder must not be assumed");
		assert!(mp3_error.to_string().contains("PCM16 or float32 WAV"));

		let video = directory.path().join("clip.mp4");
		std::fs::write(&video, b"\0\0\0\x18ftypisom").expect("video fixture");
		let video_error = load(&video)
			.err()
			.expect("video decoder must not be assumed");
		assert!(video_error.to_string().contains("self-contained runtime"));
	}
}
