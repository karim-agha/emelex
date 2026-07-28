//! `mlex`: a safe, idiomatic Rust runtime for running MLX language models —
//! including the full range of quantized checkpoints produced by the MLX
//! community (affine 2/3/4/5/6/8-bit, mxfp4, mxfp8, nvfp4, and mixed
//! per-layer precision such as OptiQ or Google QAT exports).
//!
//! Layering:
//! - [`array`] / [`ops`] / [`stream`]: thin safe wrappers over the private
//!   `sys` module (raw `mlx-c` FFI bindings, built from vendored MLX/mlx-c C++
//!   sources via `build.rs`).
//! - [`quant`]: parses the `quantization` section of `config.json` and resolves
//!   per-layer bit-widths, mirroring mlx-lm's loader semantics.
//! - [`nn`] / [`weights`]: generic building blocks (linear, embedding, norm)
//!   that transparently load dense or quantized weights.
//! - [`models`]: concrete architectures (Qwen3, Qwen3.5 (+MoE), Gemma4).
//! - [`tokenizer`] / [`sampling`] / [`generate`]: text I/O and the generation
//!   loop shared by every architecture.

pub mod array;
pub mod error;
pub mod generate;
pub mod media;
pub mod models;
pub(crate) mod mtp_certification;
pub mod nn;
pub mod ops;
pub mod prompt_cache;
pub mod quant;
pub mod reasoning;
pub mod sampling;
// emelex patch (not upstream): the speculative decode round seam.
pub mod parity;
pub(crate) mod spec;
pub mod stream;
pub mod streaming;
pub(crate) mod sys;
// emelex patch (not upstream): #[cfg(test)] fixtures — hand-rolled
// safetensors writer + tiny on-disk model builder.
#[cfg(test)]
pub mod test_support;
pub mod tokenizer;
pub mod tools;
pub mod weights;

pub use error::{Error, Result};

/// Private cooperative-cancellation probe threaded through synchronous
/// inference work. `disabled` preserves the direct engine API's historical
/// execution, while provider calls install a cheap predicate backed by their
/// cancel-on-drop flag.
#[derive(Clone, Copy)]
pub(crate) struct Cancellation<'a>(Option<&'a dyn Fn() -> bool>);

impl<'a> Cancellation<'a> {
	pub(crate) const fn disabled() -> Self {
		Self(None)
	}

	pub(crate) const fn cooperative(predicate: &'a dyn Fn() -> bool) -> Self {
		Self(Some(predicate))
	}

	pub(crate) const fn is_cooperative(self) -> bool {
		self.0.is_some()
	}

	pub(crate) fn checkpoint(self) -> Result<()> {
		if self.0.is_some_and(|predicate| predicate()) {
			return Err(Error::Cancelled);
		}
		Ok(())
	}
}
