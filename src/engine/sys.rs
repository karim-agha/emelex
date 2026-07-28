//! Raw FFI bindings to Apple MLX via the official `mlx-c` C API.
//!
//! Everything in this module is `unsafe` and mirrors the C API exactly.
//! It is private to this crate; the rest of `mlex` builds a safe, idiomatic
//! interface on top of it (see [`crate::engine::array`],
//! [`crate::engine::ops`], ...).

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(improper_ctypes)]
#![allow(dead_code)]
#![allow(clippy::all)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
