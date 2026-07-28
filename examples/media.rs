//! Self-contained image and WAV input, plus typed media failures.
//!
//! No ambient codec process is used. Emelex decodes JPEG/PNG/WebP images and
//! PCM16/float32 WAV audio itself. Encoded video is rejected before inference
//! until a decoder ships as part of the runtime.
//!
//! ```sh
//! cargo run -p emelex --release --example media -- \
//!   "$EMELEX_TEST_MODEL"
//! ```

// Example code: unwraps and panics are acceptable here.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::io::Cursor;

use emelex::{
	Client,
	generation::{Content, GenerationRequest, Message},
};
use image::{Rgb, RgbImage};

fn frame_png() -> Vec<u8> {
	let mut image = RgbImage::from_pixel(256, 256, Rgb([255, 255, 255]));
	for y in 88..168 {
		for x in 88..168 {
			image.put_pixel(x, y, Rgb([220, 30, 30]));
		}
	}
	let mut png = Cursor::new(Vec::new());
	image
		.write_to(&mut png, image::ImageFormat::Png)
		.expect("PNG encode");
	png.into_inner()
}

fn silent_pcm16_wav() -> Vec<u8> {
	const SAMPLE_RATE: u32 = 16_000;
	const SAMPLES: usize = 1_600;
	const DATA_BYTES: usize = SAMPLES * std::mem::size_of::<i16>();
	const DATA_BYTES_U32: u32 = 3_200;
	let mut wav = Vec::with_capacity(44 + DATA_BYTES);
	wav.extend_from_slice(b"RIFF");
	wav.extend_from_slice(&(36 + DATA_BYTES_U32).to_le_bytes());
	wav.extend_from_slice(b"WAVEfmt ");
	wav.extend_from_slice(&16_u32.to_le_bytes());
	wav.extend_from_slice(&1_u16.to_le_bytes());
	wav.extend_from_slice(&1_u16.to_le_bytes());
	wav.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
	wav.extend_from_slice(&(SAMPLE_RATE * 2).to_le_bytes());
	wav.extend_from_slice(&2_u16.to_le_bytes());
	wav.extend_from_slice(&16_u16.to_le_bytes());
	wav.extend_from_slice(b"data");
	wav.extend_from_slice(&DATA_BYTES_U32.to_le_bytes());
	wav.resize(44 + DATA_BYTES, 0);
	wav
}

fn request(content: Content, question: &str) -> GenerationRequest {
	let mut message = Message::user(question);
	message.content.insert(0, content);
	let mut request = GenerationRequest::default();
	request.messages.push(message);
	request
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
	let model_dir = std::env::args()
		.nth(1)
		.expect("usage: media <mlx-model-dir>");
	let client = Client::from_path(model_dir)?;

	if client.supports_images() {
		let response = client
			.generate(request(
				Content::Image(frame_png()),
				"What color is the square? Reply with one word.",
			))
			.await?;
		println!("[image] {}", response.text);
	} else {
		println!("[image] skipped: checkpoint has no image input");
	}

	if client.supports_audio() {
		let response = client
			.generate(request(
				Content::Audio(silent_pcm16_wav()),
				"Is this clip silent? Reply yes or no.",
			))
			.await?;
		println!("[audio] {}", response.text);
	} else {
		println!("[audio] skipped: checkpoint has no audio input");
	}

	let video_error = client
		.generate(request(
			Content::Video(b"\0\0\0\x18ftypisom".to_vec()),
			"Describe this video.",
		))
		.await
		.expect_err("self-contained runtime must reject encoded video");
	println!("[video typed error] {video_error}");

	if client.supports_images() {
		let image_error = client
			.generate(request(
				Content::Image(b"definitely not a PNG".to_vec()),
				"Describe this image.",
			))
			.await
			.expect_err("corrupt image bytes must fail cleanly");
		println!("[corrupt image typed error] {image_error}");
	}

	Ok(())
}
