//! Structured translation with a translation model (TranslateGemma-style).
//!
//! Translation models expose a per-message contract instead of free-form
//! chat: every user turn carries exactly one `{source_lang, target_lang,
//! text}` mapping, built here with [`Message::translation`]. Plain chat
//! against such a model fails with a typed capability error.
//!
//! ```sh
//! cargo run -p emelex --release --example translate -- \
//!   ~/.emelex/models/hub/namespaced/mlx-community/translategemma-4b-it-4bit/<revision> \
//!   en de "The weather is beautiful today."
//! ```
//!
//! The language pair and text are optional; they default to en→de and a
//! sample sentence.

// Example code: unwraps and panics are acceptable here.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use emelex::{
	Client,
	generation::{GenerationEvent, GenerationRequest, Message},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
	let mut args = std::env::args().skip(1);
	let model_dir = args
		.next()
		.expect("usage: translate <mlx-model-dir> [source-lang] [target-lang] [text]");
	let source = args.next().unwrap_or_else(|| "en".to_string());
	let target = args.next().unwrap_or_else(|| "pl".to_string());

	let text = args
		.next()
		.unwrap_or_else(|| "The weather is beautiful today.".to_string());

	let client = Client::from_path(model_dir)?;

	assert!(
		client.supports_translation(),
		"this checkpoint's chat template does not accept structured translation \
 			 requests; install one with `emelex hub search --require task:translation`"
	);

	// Translation templates usually embed their supported language table;
	// when present it can validate codes before spending any GPU time.
	if let Some(languages) = client.translation_languages() {
		for code in [source.as_str(), target.as_str()] {
			assert!(
				languages.contains_key(code),
				"language code {code:?} is not in this model's table ({} entries)",
				languages.len()
			);
		}
		println!(
			"[languages] {} supported, e.g. {}",
			languages.len(),
			languages
				.iter()
				.take(5)
				.map(|(code, name)| format!("{code}={name}"))
				.collect::<Vec<_>>()
				.join(", ")
		);
	}

	// One-shot: a single translation message is the whole conversation.
	let response = client
		.generate(
			GenerationRequest::default().message(Message::translation(&source, &target, &text)),
		)
		.await?;
	println!("[{source}→{target}] {}", response.text.trim());

	// Streaming works the same way; translate the answer back.
	let mut stream = client.stream(GenerationRequest::default().message(Message::translation(
		&target,
		&source,
		response.text.trim(),
	)))?;
	print!("[{target}→{source}] ");
	while let Some(event) = stream.recv().await {
		match event? {
			GenerationEvent::Text(piece) => print!("{piece}"),
			GenerationEvent::Completed(_) => println!(),
			_ => {}
		}
	}

	// Plain chat is rejected up front with a typed capability error — the
	// template only renders translation mappings.
	let chat_error = client
		.generate(GenerationRequest::text("Hello! How are you today?"))
		.await
		.expect_err("translation-only template must reject plain chat");
	println!("[plain chat typed error] {chat_error}");

	Ok(())
}
