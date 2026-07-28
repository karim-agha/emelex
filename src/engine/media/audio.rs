//! Audio preprocessing matching Gemma4's audio feature extractor:
//! 16 kHz mono PCM in, `[T, 128]` log-mel spectrogram frames out.
//!
//! Parameters (verified against this model family's published
//! configuration): `sample_rate=16000`, `n_fft=512`,
//! `window_len=320` (20 ms periodic Hann, zero-padded to the FFT size),
//! `hop=160` (10 ms), `n_mels=128` (HTK scale, `fmin=0`, `fmax=8000`, no
//! Slaney area normalization), magnitude (not power) spectrum, natural log
//! with a `1e-3` floor, semicausal left padding of `window_len / 2`, split
//! into 30-second chunks (each chunk is run through the audio tower
//! separately, matching the model's attention-context training length).
//!
//! Decoding is self-contained: RIFF/WAVE PCM16 and float32 are parsed,
//! downmixed, and resampled natively. No ambient codec executable is run.

use rustfft::{FftPlanner, num_complex::Complex32};

use crate::engine::{
	Cancellation,
	array::Array,
	error::{Error, Result},
	media::MAX_ENCODED_MEDIA_BYTES,
};

/// Sample rate the Gemma4 audio front-end expects (16 kHz mono).
pub const AUDIO_SAMPLE_RATE: u32 = 16_000;
/// FFT size.
const N_FFT: usize = 512;
/// Hann window length (20 ms @ 16 kHz), zero-padded to [`N_FFT`].
const WINDOW_LEN: usize = 320;
/// Hop length (10 ms @ 16 kHz).
const HOP: usize = 160;
/// Mel filterbank size.
const N_MELS: usize = 128;
/// Log-mel floor.
const MEL_FLOOR: f64 = 1e-3;
/// Chunk length in samples (30 s, the model's per-pass context limit).
const CHUNK_SAMPLES: usize = 30 * AUDIO_SAMPLE_RATE as usize;
const MAX_AUDIO_SECONDS: usize = 10 * 60;
const MAX_AUDIO_SAMPLES: usize = MAX_AUDIO_SECONDS * AUDIO_SAMPLE_RATE as usize;
const MAX_SAMPLES_PER_TOKEN: i32 = CHUNK_SAMPLES as i32;
const MAX_PADDED_AUDIO_SAMPLES: usize = MAX_AUDIO_SAMPLES + MAX_SAMPLES_PER_TOKEN as usize - 1;

/// A preprocessed audio clip ready for a Gemma4 audio encoder. Two shapes,
/// selected by which encoder the checkpoint loaded (see
/// `crate::engine::models::gemma4::AudioEncoder`):
/// - mel-spectrogram Conformer tower: `chunks` holds one `[1, T_i, 128]`
///   log-mel tensor per 30-second chunk, and each chunk's soft-token count is
///   subsampled by the tower's two stride-2 convolutions.
/// - encoder-free "unified" path: `chunks` holds a single `[n, S]` raw PCM
///   window tensor (`S` = samples per audio token), one soft token per row.
#[derive(Debug, Clone)]
pub struct ProcessedAudio {
	pub chunks: Vec<Array>,
	/// Per-chunk frame count (`chunks[i].dim(1)` for mel, `chunks[i].dim(0)`
	/// for raw windows).
	pub frames_per_chunk: Vec<i32>,
	/// Whether `chunks`/`frames_per_chunk` hold raw PCM windows (unified
	/// path, one soft token per row, no subsampling) rather than
	/// mel-spectrogram frames (Conformer tower path).
	pub raw: bool,
}

impl ProcessedAudio {
	/// Total soft tokens this clip expands to: for the mel-spectrogram
	/// path, after the audio tower's two stride-2 subsampling convolutions
	/// (`O = (I - 1) / 2 + 1`, twice); for the raw-window "unified" path,
	/// one soft token per window (no subsampling), summed over chunks.
	pub fn num_soft_tokens(&self) -> i32 {
		if self.raw {
			self.frames_per_chunk.iter().sum()
		} else {
			self.frames_per_chunk
				.iter()
				.map(|&t| subsampled_len(t))
				.sum()
		}
	}

	pub(crate) fn retained_tensor_bytes(&self) -> Result<usize> {
		self.chunks.iter().try_fold(0_usize, |total, chunk| {
			chunk
				.size()
				.checked_mul(std::mem::size_of::<f32>())
				.and_then(|bytes| total.checked_add(bytes))
				.ok_or_else(|| {
					Error::Model("processed audio tensor byte count overflow".to_string())
				})
		})
	}
}

/// Output length of the audio tower's two stride-2 (kernel 3, pad 1)
/// subsampling convolutions for `t` input mel frames.
pub fn subsampled_len(t: i32) -> i32 {
	let mut n = t;
	for _ in 0..2 {
		n = (n - 1) / 2 + 1;
	}
	n
}

/// Decode bounded RIFF/WAVE PCM16 or float32 `data` to 16 kHz mono f32 PCM,
/// then compute per-chunk log-mel spectrograms.
pub fn preprocess_audio_bytes(data: &[u8]) -> Result<ProcessedAudio> {
	preprocess_audio_bytes_cancellable(data, Cancellation::disabled())
}

pub(crate) fn preprocess_audio_bytes_cancellable(
	data: &[u8],
	cancellation: Cancellation<'_>,
) -> Result<ProcessedAudio> {
	cancellation.checkpoint()?;
	let pcm = decode_audio_bytes_cancellable(data, cancellation)?;
	if pcm.is_empty() {
		return Err(Error::Model("audio clip decoded to zero samples".into()));
	}

	let mut chunks = Vec::new();
	let mut frames_per_chunk = Vec::new();
	for chunk in pcm.chunks(CHUNK_SAMPLES) {
		cancellation.checkpoint()?;
		let mel = log_mel_spectrogram_cancellable(chunk, cancellation)?;
		let t = (mel.len() / N_MELS) as i32;
		if t == 0 {
			continue;
		}
		chunks.push(Array::from_slice(&mel, &[1, t, N_MELS as i32])?);
		frames_per_chunk.push(t);
	}
	if chunks.is_empty() {
		return Err(Error::Model(
			"audio clip too short to produce any mel frames".into(),
		));
	}
	Ok(ProcessedAudio {
		chunks,
		frames_per_chunk,
		raw: false,
	})
}

/// Decode bounded RIFF/WAVE PCM16 or float32 `data` to 16 kHz mono f32 PCM,
/// then build the
/// encoder-free "unified" path's raw-window frame tensor: zero-pad right
/// to a multiple of `samples_per_token`, reshape to `[n_frames,
/// samples_per_token]`, no scaling/normalization (samples pass through
/// untouched). One frame = one audio soft token; there is no tower and no
/// subsampling, unlike the mel-spectrogram Conformer path.
pub fn preprocess_audio_bytes_raw(data: &[u8], samples_per_token: i32) -> Result<ProcessedAudio> {
	preprocess_audio_bytes_raw_cancellable(data, samples_per_token, Cancellation::disabled())
}

pub(crate) fn preprocess_audio_bytes_raw_cancellable(
	data: &[u8],
	samples_per_token: i32,
	cancellation: Cancellation<'_>,
) -> Result<ProcessedAudio> {
	cancellation.checkpoint()?;
	// emelex patch: this checkpoint field is untrusted and becomes both an
	// allocation divisor and an MLX shape.
	if !(1..=MAX_SAMPLES_PER_TOKEN).contains(&samples_per_token) {
		return Err(Error::Model(format!(
			"audio samples_per_token must be within 1..={MAX_SAMPLES_PER_TOKEN}"
		)));
	}
	let pcm = decode_audio_bytes_cancellable(data, cancellation)?;
	if pcm.is_empty() {
		return Err(Error::Model("audio clip decoded to zero samples".into()));
	}
	let spt = usize::try_from(samples_per_token)
		.map_err(|_| Error::Model("audio samples_per_token is invalid".to_string()))?;

	let n = pcm.len();
	let pad = (spt - (n % spt)) % spt;
	let padded_len = n
		.checked_add(pad)
		.ok_or_else(|| Error::Model("raw audio padding overflow".to_string()))?;
	if padded_len > MAX_PADDED_AUDIO_SAMPLES {
		return Err(Error::Model(
			"raw audio padding exceeds bounded allocation limit".to_string(),
		));
	}
	let n_frames = padded_len / spt;
	let n_frames_i32 = i32::try_from(n_frames)
		.map_err(|_| Error::Model("raw audio frame count exceeds i32".to_string()))?;

	let mut frames = pcm;
	frames.resize(padded_len, 0.0);
	cancellation.checkpoint()?;

	let tensor = Array::from_slice(&frames, &[n_frames_i32, samples_per_token])?;
	Ok(ProcessedAudio {
		chunks: vec![tensor],
		frames_per_chunk: vec![n_frames_i32],
		raw: true,
	})
}

/// Decode self-contained RIFF/WAVE audio to 16 kHz mono f32 PCM.
pub fn decode_audio_bytes(data: &[u8]) -> Result<Vec<f32>> {
	decode_audio_bytes_cancellable(data, Cancellation::disabled())
}

/// Validate bounded RIFF/WAVE structure and the PCM16/float32 codec surface
/// without allocating decoded samples or initializing MLX.
pub(crate) fn validate_audio_bytes(data: &[u8]) -> Result<()> {
	validate_audio_bytes_cancellable(data, Cancellation::disabled())
}

fn validate_audio_bytes_cancellable(data: &[u8], cancellation: Cancellation<'_>) -> Result<()> {
	cancellation.checkpoint()?;
	validate_encoded_audio_size(data)?;
	let wav = parse_wav(data, cancellation)?;
	let Some(encoding) = wav.encoding else {
		return Err(unsupported_audio());
	};
	if matches!(encoding, WavEncoding::Float32) {
		for (index, sample) in wav
			.data
			.chunks_exact(encoding.bytes_per_sample())
			.enumerate()
		{
			if index % 8_192 == 0 {
				cancellation.checkpoint()?;
			}
			let value = f32::from_le_bytes([sample[0], sample[1], sample[2], sample[3]]);
			if !value.is_finite() {
				return Err(Error::Model(
					"decoded audio contains non-finite samples".to_string(),
				));
			}
		}
	}
	Ok(())
}

fn decode_audio_bytes_cancellable(data: &[u8], cancellation: Cancellation<'_>) -> Result<Vec<f32>> {
	cancellation.checkpoint()?;
	validate_encoded_audio_size(data)?;
	let Some(pcm) = decode_wav_cancellable(data, cancellation)? else {
		return Err(unsupported_audio());
	};
	validate_pcm_cancellable(pcm, cancellation)
}

fn validate_encoded_audio_size(data: &[u8]) -> Result<()> {
	if data.len() > MAX_ENCODED_MEDIA_BYTES {
		return Err(Error::Model(format!(
			"encoded audio exceeds {} byte limit",
			MAX_ENCODED_MEDIA_BYTES
		)));
	}
	Ok(())
}

fn unsupported_audio() -> Error {
	Error::Model(
		"self-contained audio decoding accepts RIFF/WAVE PCM16 or float32 only".to_string(),
	)
}

/// Parse a RIFF/WAVE byte stream. Returns `Ok(None)` when the file is a
/// valid WAV whose codec is outside the self-contained PCM surface.
#[cfg(test)]
fn decode_wav(bytes: &[u8]) -> Result<Option<Vec<f32>>> {
	decode_wav_cancellable(bytes, Cancellation::disabled())
}

fn decode_wav_cancellable(
	bytes: &[u8],
	cancellation: Cancellation<'_>,
) -> Result<Option<Vec<f32>>> {
	let wav = parse_wav(bytes, cancellation)?;
	let Some(encoding) = wav.encoding else {
		return Ok(None);
	};
	let bytes_per_sample = encoding.bytes_per_sample();
	let decode_sample = |sample: &[u8]| match encoding {
		WavEncoding::Pcm16 => i16::from_le_bytes([sample[0], sample[1]]) as f32 / 32768.0,
		WavEncoding::Float32 => f32::from_le_bytes([sample[0], sample[1], sample[2], sample[3]]),
	};

	if wav.channels == 1 {
		let samples = wav.data.len() / bytes_per_sample;
		let mut mono = Vec::with_capacity(samples);
		for (index, sample) in wav.data.chunks_exact(bytes_per_sample).enumerate() {
			if index % 8_192 == 0 {
				cancellation.checkpoint()?;
			}
			mono.push(decode_sample(sample));
		}
		return resample_pcm_cancellable(mono, wav.sample_rate, cancellation).map(Some);
	}

	// emelex patch: decode and downmix one frame at a time. Holding an
	// interleaved f32 copy can otherwise amplify bounded PCM16 by roughly 2x.
	let frames = wav.data.len() / wav.frame_bytes;
	let mut mono = Vec::with_capacity(frames);
	for (index, frame) in wav.data.chunks_exact(wav.frame_bytes).enumerate() {
		if index % 8_192 == 0 {
			cancellation.checkpoint()?;
		}
		let sum = frame
			.chunks_exact(bytes_per_sample)
			.map(&decode_sample)
			.sum::<f32>();
		mono.push(sum / wav.channels as f32);
	}
	resample_pcm_cancellable(mono, wav.sample_rate, cancellation).map(Some)
}

#[derive(Debug, Clone, Copy)]
enum WavEncoding {
	Pcm16,
	Float32,
}

impl WavEncoding {
	const fn bytes_per_sample(self) -> usize {
		match self {
			Self::Pcm16 => 2,
			Self::Float32 => 4,
		}
	}
}

struct Wav<'a> {
	encoding: Option<WavEncoding>,
	channels: usize,
	sample_rate: u32,
	frame_bytes: usize,
	data: &'a [u8],
}

fn parse_wav<'a>(bytes: &'a [u8], cancellation: Cancellation<'_>) -> Result<Wav<'a>> {
	if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
		return Err(unsupported_audio());
	}
	let declared_size = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
	let declared_end = declared_size
		.checked_add(8)
		.ok_or_else(|| Error::Model("corrupt WAV: RIFF size overflow".into()))?;
	if declared_end != bytes.len() {
		return Err(Error::Model(
			"corrupt WAV: RIFF size does not match file length".into(),
		));
	}
	let mut audio_format = None;
	let mut num_channels = 0u16;
	let mut sample_rate = 0u32;
	let mut byte_rate = 0u32;
	let mut block_align = 0u16;
	let mut bits_per_sample = 0u16;
	let mut data: Option<&[u8]> = None;

	let mut pos = 12usize;
	while pos + 8 <= bytes.len() {
		cancellation.checkpoint()?;
		let chunk_id = &bytes[pos..pos + 4];
		let chunk_size = u32::from_le_bytes([
			bytes[pos + 4],
			bytes[pos + 5],
			bytes[pos + 6],
			bytes[pos + 7],
		]) as usize;
		let body_start = pos + 8;
		let body_end = body_start
			.checked_add(chunk_size)
			.ok_or_else(|| Error::Model("corrupt WAV: chunk size overflow".into()))?;
		if body_end > bytes.len() {
			return Err(Error::Model("corrupt WAV: truncated chunk".into()));
		}
		let body = &bytes[body_start..body_end];
		if chunk_id == b"fmt " {
			if audio_format.is_some() {
				return Err(Error::Model("corrupt WAV: duplicate fmt chunk".into()));
			}
			if body.len() < 16 {
				return Err(Error::Model("corrupt WAV: fmt chunk too short".into()));
			}
			audio_format = Some(u16::from_le_bytes([body[0], body[1]]));
			num_channels = u16::from_le_bytes([body[2], body[3]]);
			sample_rate = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
			byte_rate = u32::from_le_bytes([body[8], body[9], body[10], body[11]]);
			block_align = u16::from_le_bytes([body[12], body[13]]);
			bits_per_sample = u16::from_le_bytes([body[14], body[15]]);
		} else if chunk_id == b"data" {
			if data.is_some() {
				return Err(Error::Model("corrupt WAV: duplicate data chunk".into()));
			}
			data = Some(body);
		}
		pos = body_end
			.checked_add(chunk_size & 1)
			.ok_or_else(|| Error::Model("corrupt WAV: padded chunk size overflow".into()))?;
		if pos > bytes.len() {
			return Err(Error::Model("corrupt WAV: truncated chunk padding".into()));
		}
	}
	if pos != bytes.len() {
		return Err(Error::Model("corrupt WAV: truncated chunk header".into()));
	}

	let Some(audio_format) = audio_format else {
		return Err(Error::Model("corrupt WAV: missing fmt chunk".into()));
	};
	let Some(data) = data else {
		return Err(Error::Model("corrupt WAV: missing data chunk".into()));
	};
	if num_channels == 0 || num_channels > 32 || sample_rate == 0 || sample_rate > 384_000 {
		return Err(Error::Model(
			"corrupt WAV: channel count or sample rate is unsupported".to_string(),
		));
	}
	let encoding = match (audio_format, bits_per_sample) {
		(1, 16) => Some(WavEncoding::Pcm16),
		(3, 32) => Some(WavEncoding::Float32),
		_ => None,
	};
	let Some(bytes_per_sample) = encoding.map(WavEncoding::bytes_per_sample) else {
		return Ok(Wav {
			encoding: None,
			channels: usize::from(num_channels),
			sample_rate,
			frame_bytes: usize::from(block_align),
			data,
		});
	};
	let frame_bytes = usize::from(num_channels)
		.checked_mul(bytes_per_sample)
		.ok_or_else(|| Error::Model("corrupt WAV: frame size overflow".into()))?;
	if usize::from(block_align) != frame_bytes {
		return Err(Error::Model(
			"corrupt WAV: block alignment does not match channels and sample width".into(),
		));
	}
	let expected_byte_rate = sample_rate
		.checked_mul(u32::try_from(frame_bytes).map_err(|_| {
			Error::Model("corrupt WAV: frame size does not fit byte rate".to_string())
		})?)
		.ok_or_else(|| Error::Model("corrupt WAV: byte rate overflow".to_string()))?;
	if byte_rate != expected_byte_rate {
		return Err(Error::Model(
			"corrupt WAV: byte rate does not match sample rate and frame size".into(),
		));
	}
	// emelex patch: chunks_exact would silently discard a partial sample or
	// channel frame, accepting a truncated WAV as valid PCM.
	if data.is_empty() || data.len() % frame_bytes != 0 {
		return Err(Error::Model("corrupt WAV: partial audio frame".to_string()));
	}
	let source_sample_limit = MAX_AUDIO_SECONDS
		.checked_mul(
			usize::try_from(sample_rate)
				.map_err(|_| Error::Model("WAV sample rate does not fit memory".to_string()))?,
		)
		.ok_or_else(|| Error::Model("WAV duration limit overflow".to_string()))?;
	if data.len() / frame_bytes > source_sample_limit {
		return Err(Error::Model(format!(
			"decoded audio exceeds {MAX_AUDIO_SECONDS} second limit"
		)));
	}
	Ok(Wav {
		encoding,
		channels: usize::from(num_channels),
		sample_rate,
		frame_bytes,
		data,
	})
}

#[cfg(test)]
fn resample_pcm(input: Vec<f32>, sample_rate: u32) -> Result<Vec<f32>> {
	resample_pcm_cancellable(input, sample_rate, Cancellation::disabled())
}

fn resample_pcm_cancellable(
	input: Vec<f32>,
	sample_rate: u32,
	cancellation: Cancellation<'_>,
) -> Result<Vec<f32>> {
	cancellation.checkpoint()?;
	if sample_rate == AUDIO_SAMPLE_RATE || input.is_empty() {
		return Ok(input);
	}
	let output_len = input
		.len()
		.checked_mul(AUDIO_SAMPLE_RATE as usize)
		.and_then(|scaled| scaled.checked_add(sample_rate as usize - 1))
		.map(|scaled| scaled / sample_rate as usize)
		.ok_or_else(|| Error::Model("resampled audio length overflow".to_string()))?;
	if output_len > MAX_AUDIO_SAMPLES {
		return Err(Error::Model(format!(
			"decoded audio exceeds {MAX_AUDIO_SECONDS} second limit"
		)));
	}
	let mut output = Vec::with_capacity(output_len);
	for index in 0..output_len {
		if index % 8_192 == 0 {
			cancellation.checkpoint()?;
		}
		let numerator = index
			.checked_mul(sample_rate as usize)
			.ok_or_else(|| Error::Model("audio resampling position overflow".to_string()))?;
		let left = numerator / AUDIO_SAMPLE_RATE as usize;
		let remainder = numerator % AUDIO_SAMPLE_RATE as usize;
		let Some(&first) = input.get(left) else {
			break;
		};
		let second = input.get(left + 1).copied().unwrap_or(first);
		let fraction = remainder as f32 / AUDIO_SAMPLE_RATE as f32;
		output.push(first + (second - first) * fraction);
	}
	Ok(output)
}

fn validate_pcm_cancellable(pcm: Vec<f32>, cancellation: Cancellation<'_>) -> Result<Vec<f32>> {
	if pcm.len() > MAX_AUDIO_SAMPLES {
		return Err(Error::Model(format!(
			"decoded audio exceeds {MAX_AUDIO_SECONDS} second limit"
		)));
	}
	for (index, sample) in pcm.iter().enumerate() {
		if index % 8_192 == 0 {
			cancellation.checkpoint()?;
		}
		if !sample.is_finite() {
			return Err(Error::Model(
				"decoded audio contains non-finite samples".to_string(),
			));
		}
	}
	Ok(pcm)
}

/// Gemma4 log-mel spectrogram for one <=30s chunk of 16 kHz mono PCM.
/// Returns `t * N_MELS` values, frame-major (`out[t * N_MELS + m]`).
#[cfg(test)]
fn log_mel_spectrogram(chunk: &[f32]) -> Vec<f32> {
	log_mel_spectrogram_cancellable(chunk, Cancellation::disabled())
		.expect("disabled cancellation cannot interrupt spectrogram generation")
}

fn log_mel_spectrogram_cancellable(
	chunk: &[f32],
	cancellation: Cancellation<'_>,
) -> Result<Vec<f32>> {
	cancellation.checkpoint()?;
	// Semicausal left padding + right padding to match the expected frame
	// count: unfold(size=window_len + 1, step=hop) over the left-padded
	// waveform.
	let pad_left = WINDOW_LEN / 2;
	let n_with_left = chunk.len() + pad_left;
	if n_with_left < WINDOW_LEN + 1 {
		return Ok(Vec::new());
	}
	let pt_frames = (n_with_left - (WINDOW_LEN + 1)) / HOP + 1;
	let n_padded_needed = (pt_frames - 1) * HOP + N_FFT;
	let total_pad = n_padded_needed.saturating_sub(chunk.len()).max(pad_left);
	let mut padded = vec![0f32; total_pad + chunk.len()];
	padded[pad_left..pad_left + chunk.len()].copy_from_slice(chunk);

	// Standard periodic Hann window of WINDOW_LEN, zero-padded to N_FFT.
	let mut hann = vec![0f32; N_FFT];
	for (i, w) in hann.iter_mut().enumerate().take(WINDOW_LEN) {
		*w = 0.5 - 0.5 * ((2.0 * std::f32::consts::PI * i as f32) / WINDOW_LEN as f32).cos();
	}

	let filters = mel_filterbank();
	let n_bins = N_FFT / 2 + 1;

	let mut planner = FftPlanner::<f32>::new();
	let fft = planner.plan_fft_forward(N_FFT);

	let n_frames = ((padded.len() - N_FFT) / HOP + 1).min(pt_frames);
	let mut out = vec![0f32; n_frames * N_MELS];
	let mut buf = vec![Complex32::new(0.0, 0.0); N_FFT];
	let mut magnitude = vec![0f32; n_bins];
	for t in 0..n_frames {
		if t % 8 == 0 {
			cancellation.checkpoint()?;
		}
		let offset = t * HOP;
		for j in 0..N_FFT {
			buf[j] = Complex32::new(hann[j] * padded[offset + j], 0.0);
		}
		fft.process(&mut buf);
		for (j, m) in magnitude.iter_mut().enumerate() {
			*m = buf[j].norm();
		}
		for m in 0..N_MELS {
			let mut sum = 0f64;
			for (j, &mag) in magnitude.iter().enumerate() {
				sum += mag as f64 * filters[m * n_bins + j] as f64;
			}
			out[t * N_MELS + m] = sum.max(MEL_FLOOR).ln() as f32;
		}
	}
	Ok(out)
}

/// Triangular mel filterbank: HTK mel scale, `fmin=0`, `fmax=sr/2`, no
/// Slaney area normalization. `N_MELS x (N_FFT/2 + 1)`, filter-major.
fn mel_filterbank() -> Vec<f32> {
	let n_bins = N_FFT / 2 + 1;
	let fmax = AUDIO_SAMPLE_RATE as f64 / 2.0;
	let hz_to_mel = |f: f64| 2595.0 * (1.0 + f / 700.0).log10();
	let mel_to_hz = |m: f64| 700.0 * (10f64.powf(m / 2595.0) - 1.0);

	let m_lo = hz_to_mel(0.0);
	let m_hi = hz_to_mel(fmax);
	let hz_pts: Vec<f64> = (0..N_MELS + 2)
		.map(|i| mel_to_hz(m_lo + (m_hi - m_lo) * i as f64 / (N_MELS + 1) as f64))
		.collect();

	let bin_hz_step = AUDIO_SAMPLE_RATE as f64 / N_FFT as f64;
	let mut out = vec![0f32; N_MELS * n_bins];
	for m in 0..N_MELS {
		let (f_left, f_center, f_right) = (hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]);
		let denom_l = (f_center - f_left).max(1e-30);
		let denom_r = (f_right - f_center).max(1e-30);
		for (k, o) in out[m * n_bins..(m + 1) * n_bins].iter_mut().enumerate() {
			let f = k as f64 * bin_hz_step;
			let w = if f >= f_left && f <= f_center {
				(f - f_left) / denom_l
			} else if f > f_center && f <= f_right {
				(f_right - f) / denom_r
			} else {
				0.0
			};
			*o = w as f32;
		}
	}
	out
}

#[cfg(test)]
mod tests {
	use std::cell::Cell;

	use super::*;

	#[test]
	fn preprocessing_observes_cancellation_inside_wav_parsing() {
		let checks = Cell::new(0_usize);
		let is_cancelled = || {
			checks.set(checks.get() + 1);
			checks.get() >= 3
		};
		let mut wav = vec![0_u8; 20];
		wav[0..4].copy_from_slice(b"RIFF");
		wav[4..8].copy_from_slice(&12_u32.to_le_bytes());
		wav[8..12].copy_from_slice(b"WAVE");

		let error =
			preprocess_audio_bytes_cancellable(&wav, Cancellation::cooperative(&is_cancelled))
				.unwrap_err();

		assert!(matches!(error, Error::Cancelled));
		assert_eq!(checks.get(), 3);
	}

	#[test]
	fn subsampled_len_matches_two_stride2_convs() {
		// O = (I - 1)/2 + 1, applied twice.
		assert_eq!(subsampled_len(1), 1);
		assert_eq!(subsampled_len(4), 1);
		assert_eq!(subsampled_len(100), 25);
		assert_eq!(subsampled_len(3000), 750);
	}

	#[test]
	fn mel_filterbank_rows_are_valid_triangles() {
		let filters = mel_filterbank();
		let n_bins = N_FFT / 2 + 1;
		let mut nonempty = 0;
		for m in 0..N_MELS {
			let row = &filters[m * n_bins..(m + 1) * n_bins];
			if row.iter().sum::<f32>() > 0.0 {
				nonempty += 1;
			}
			assert!(row.iter().all(|&w| (0.0..=1.0).contains(&w)));
		}
		// The lowest few filters legitimately span less than one FFT bin
		// (triangle narrower than 31.25 Hz) and come out all-zero -
		// identical to the reference filterbank. Everything else must be
		// a real triangle.
		assert!(
			nonempty >= N_MELS - 8,
			"only {nonempty}/{N_MELS} mel filters are nonzero"
		);
	}

	#[test]
	fn spectrogram_frame_count_matches_pytorch_unfold() {
		// 1 second of a 440 Hz tone: pt_frames = (16000 + 160 - 321)/160 + 1 = 99.
		let pcm: Vec<f32> = (0..16000)
			.map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
			.collect();
		let mel = log_mel_spectrogram(&pcm);
		assert_eq!(mel.len() / N_MELS, 99);
		assert!(mel.iter().all(|v| v.is_finite()));
	}

	#[test]
	fn silence_hits_the_mel_floor() {
		let pcm = vec![0f32; 16000];
		let mel = log_mel_spectrogram(&pcm);
		let floor = (MEL_FLOOR as f32).ln();
		assert!(mel.iter().all(|&v| (v - floor).abs() < 1e-4));
	}

	#[test]
	fn decode_wav_pcm16_mono_roundtrip() {
		let samples: [i16; 4] = [0, 16384, -32768, 32767];
		let mut data = Vec::new();
		for s in samples {
			data.extend_from_slice(&s.to_le_bytes());
		}
		let mut wav = Vec::new();
		wav.extend_from_slice(b"RIFF");
		wav.extend_from_slice(&(36u32 + data.len() as u32).to_le_bytes());
		wav.extend_from_slice(b"WAVE");
		wav.extend_from_slice(b"fmt ");
		wav.extend_from_slice(&16u32.to_le_bytes());
		wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
		wav.extend_from_slice(&1u16.to_le_bytes()); // mono
		wav.extend_from_slice(&16000u32.to_le_bytes());
		wav.extend_from_slice(&32000u32.to_le_bytes());
		wav.extend_from_slice(&2u16.to_le_bytes());
		wav.extend_from_slice(&16u16.to_le_bytes());
		wav.extend_from_slice(b"data");
		wav.extend_from_slice(&(data.len() as u32).to_le_bytes());
		wav.extend_from_slice(&data);

		let pcm = decode_audio_bytes(&wav).unwrap();
		assert_eq!(pcm.len(), 4);
		assert_eq!(pcm[0], 0.0);
		assert_eq!(pcm[1], 0.5);
		assert_eq!(pcm[2], -1.0);
	}

	#[test]
	fn decode_wav_pcm16_stereo_downmixes_frame_by_frame() {
		let samples: [i16; 4] = [16384, -16384, 32767, 32767];
		let mut data = Vec::new();
		for sample in samples {
			data.extend_from_slice(&sample.to_le_bytes());
		}
		let mut wav = Vec::new();
		wav.extend_from_slice(b"RIFF");
		wav.extend_from_slice(&(36_u32 + data.len() as u32).to_le_bytes());
		wav.extend_from_slice(b"WAVE");
		wav.extend_from_slice(b"fmt ");
		wav.extend_from_slice(&16_u32.to_le_bytes());
		wav.extend_from_slice(&1_u16.to_le_bytes());
		wav.extend_from_slice(&2_u16.to_le_bytes());
		wav.extend_from_slice(&AUDIO_SAMPLE_RATE.to_le_bytes());
		wav.extend_from_slice(&(AUDIO_SAMPLE_RATE * 4).to_le_bytes());
		wav.extend_from_slice(&4_u16.to_le_bytes());
		wav.extend_from_slice(&16_u16.to_le_bytes());
		wav.extend_from_slice(b"data");
		wav.extend_from_slice(&(data.len() as u32).to_le_bytes());
		wav.extend_from_slice(&data);

		let pcm = decode_wav(&wav).unwrap().unwrap();
		assert_eq!(pcm.len(), 2);
		assert_eq!(pcm[0], 0.0);
		assert!((pcm[1] - (32767.0 / 32768.0)).abs() < f32::EPSILON);
	}

	#[test]
	fn raw_audio_rejects_invalid_samples_per_token_before_decode() {
		assert!(preprocess_audio_bytes_raw(b"", 0).is_err());
		assert!(preprocess_audio_bytes_raw(b"", -1).is_err());
		assert!(preprocess_audio_bytes_raw(b"", MAX_SAMPLES_PER_TOKEN + 1).is_err());
	}

	#[test]
	fn wav_rejects_partial_channel_frame() {
		let mut wav = Vec::new();
		wav.extend_from_slice(b"RIFF");
		wav.extend_from_slice(&39_u32.to_le_bytes());
		wav.extend_from_slice(b"WAVE");
		wav.extend_from_slice(b"fmt ");
		wav.extend_from_slice(&16_u32.to_le_bytes());
		wav.extend_from_slice(&1_u16.to_le_bytes());
		wav.extend_from_slice(&1_u16.to_le_bytes());
		wav.extend_from_slice(&AUDIO_SAMPLE_RATE.to_le_bytes());
		wav.extend_from_slice(&32_000_u32.to_le_bytes());
		wav.extend_from_slice(&2_u16.to_le_bytes());
		wav.extend_from_slice(&16_u16.to_le_bytes());
		wav.extend_from_slice(b"data");
		wav.extend_from_slice(&3_u32.to_le_bytes());
		wav.extend_from_slice(&[0, 0, 1]);
		assert!(decode_wav(&wav).is_err());
	}

	#[test]
	fn wav_rejects_mismatched_riff_size() {
		let mut wav = pcm_wav(1, WavEncoding::Pcm16, &[0, 0]);
		wav[4..8].copy_from_slice(&4_u32.to_le_bytes());
		assert!(validate_audio_bytes(&wav).is_err());
		assert!(decode_audio_bytes(&wav).is_err());
	}

	#[test]
	fn wav_rejects_mismatched_byte_rate() {
		let mut wav = pcm_wav(1, WavEncoding::Pcm16, &[0, 0]);
		wav[28..32].copy_from_slice(&1_u32.to_le_bytes());
		assert!(validate_audio_bytes(&wav).is_err());
		assert!(decode_audio_bytes(&wav).is_err());
	}

	#[test]
	fn preflight_rejects_non_finite_float_samples() {
		for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
			let wav = pcm_wav(1, WavEncoding::Float32, &value.to_le_bytes());
			assert!(validate_audio_bytes(&wav).is_err());
			assert!(decode_audio_bytes(&wav).is_err());
		}
	}

	fn pcm_wav(channels: u16, encoding: WavEncoding, data: &[u8]) -> Vec<u8> {
		let sample_rate = AUDIO_SAMPLE_RATE;
		let sample_bytes = u16::try_from(encoding.bytes_per_sample()).expect("sample width");
		let block_align = channels * sample_bytes;
		let audio_format = match encoding {
			WavEncoding::Pcm16 => 1_u16,
			WavEncoding::Float32 => 3_u16,
		};
		let mut wav = Vec::new();
		wav.extend_from_slice(b"RIFF");
		wav.extend_from_slice(
			&(36_u32 + u32::try_from(data.len()).expect("data length")).to_le_bytes(),
		);
		wav.extend_from_slice(b"WAVEfmt ");
		wav.extend_from_slice(&16_u32.to_le_bytes());
		wav.extend_from_slice(&audio_format.to_le_bytes());
		wav.extend_from_slice(&channels.to_le_bytes());
		wav.extend_from_slice(&sample_rate.to_le_bytes());
		wav.extend_from_slice(&(sample_rate * u32::from(block_align)).to_le_bytes());
		wav.extend_from_slice(&block_align.to_le_bytes());
		wav.extend_from_slice(&(sample_bytes * 8).to_le_bytes());
		wav.extend_from_slice(b"data");
		wav.extend_from_slice(
			&u32::try_from(data.len())
				.expect("data length")
				.to_le_bytes(),
		);
		wav.extend_from_slice(data);
		wav
	}

	#[test]
	fn self_contained_audio_rejects_ambient_codec_formats() {
		assert!(decode_audio_bytes(b"ID3 encoded audio").is_err());
	}

	#[test]
	fn native_resampler_is_bounded_and_interpolates() {
		let output = resample_pcm(vec![0.0, 1.0, 0.0, -1.0], 8_000).unwrap();
		assert_eq!(output.len(), 8);
		assert_eq!(output[0], 0.0);
		assert_eq!(output[1], 0.5);
		assert_eq!(output[2], 1.0);
	}
}
