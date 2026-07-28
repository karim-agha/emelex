//! Vision round-trip on a local MLX model: an image goes IN, an image
//! comes OUT.
//!
//! - **In**: a PNG is drawn procedurally (no files needed), attached to a user
//!   message as raw bytes, and the vision tower answers questions about it.
//! - **Out**: a local text decoder cannot emit pixels, so the idiomatic "image
//!   output" is model-authored vector graphics - the model recreates the
//!   analyzed scene as an SVG document, which is saved next to the source PNG.
//!
//! Requires a vision-capable checkpoint whose `config.json` declares an image
//! tower; Emelex rejects image content on text-only models with a clear
//! `UnsupportedContent` error.
//!
//! ```sh
//! cargo run -p emelex --release --example vision -- \
//!   "$EMELEX_TEST_MODEL"
//! ```

// Example code: unwraps and panics are acceptable here.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::io::Cursor;

use image::{Rgb, RgbImage};
use rig_core::{
	OneOrMany,
	completion::{Message, Prompt},
	message::{DocumentSourceKind, Image, ImageMediaType, UserContent},
};

/// Draw the test scene: a red square and a blue circle on white.
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)] // small fixed dims
fn draw_scene_png() -> Vec<u8> {
	let (w, h) = (256u32, 256u32);
	let mut img = RgbImage::from_pixel(w, h, Rgb([255, 255, 255]));
	// Red square, left side.
	for y in 88..168 {
		for x in 32..112 {
			img.put_pixel(x, y, Rgb([220, 30, 30]));
		}
	}
	// Blue circle, right side.
	let (cx, cy, r) = (184i32, 128i32, 44i32);
	for y in 0..h as i32 {
		for x in 0..w as i32 {
			if (x - cx).pow(2) + (y - cy).pow(2) <= r.pow(2) {
				img.put_pixel(x as u32, y as u32, Rgb([30, 60, 220]));
			}
		}
	}
	let mut png = Cursor::new(Vec::new());
	img.write_to(&mut png, image::ImageFormat::Png)
		.expect("png encode");
	png.into_inner()
}

/// A user message combining an image (raw PNG bytes) with a question.
fn image_question(png: Vec<u8>, question: &str) -> Message {
	Message::User {
		content: OneOrMany::many(vec![
			UserContent::Image(Image {
				data: DocumentSourceKind::Raw(png),
				media_type: Some(ImageMediaType::PNG),
				detail: None,
				additional_params: None,
			}),
			UserContent::text(question),
		])
		.expect("two content items"),
	}
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
	let model_dir = std::env::args()
		.nth(1)
		.expect("usage: vision <mlx-model-dir>");

	let png = draw_scene_png();
	std::fs::write("vision_input.png", &png)?;
	println!("wrote vision_input.png ({} bytes)", png.len());

	let agent = emelex::Client::from_path(model_dir)?
		.agent()
		.preamble(
			"You are a precise visual analyst. Answer questions about the attached \
			 image exactly and concisely.",
		)
		.build();

	// --- Image in: vision Q&A over the generated PNG. -----------------
	let description = agent
		.prompt(image_question(
			png.clone(),
			"Describe every shape in this image: its kind, its color, and where it \
			 sits (left/right/center). One line per shape.",
		))
		.await?;
	println!("\n[analysis]\n{description}");

	let count = agent
		.prompt(image_question(
			png,
			"How many distinct shapes are in this image? Reply with just the number.",
		))
		.await?;
	println!("\n[shape count] {count}");

	// --- Image out: recreate the scene as an SVG document. ------------
	// A local decoder generates text, so image *output* means image
	// *code*: the model turns its own analysis back into vector
	// graphics.
	let svg = agent
		.prompt(format!(
			"Recreate this scene as an SVG image, 256x256, white \
			 background:\n{description}\n\nReply with ONLY the SVG markup, starting \
			 with <svg and ending with </svg>, no code fences."
		))
		.await?;
	let svg = svg
		.find("<svg")
		.map_or(svg.as_str(), |at| &svg[at..])
		.trim()
		.to_string();
	std::fs::write("vision_output.svg", &svg)?;
	println!(
		"\n[recreated] wrote vision_output.svg ({} bytes)\n{svg}",
		svg.len()
	);

	Ok(())
}
