//! This crate's only content is the dual-mode kernel at
//! `../../verus-proof/kernel_dual.rs`, included here by path so the exact
//! same file is compiled by both `cargo build` (this crate, using the real
//! `cuda-device` dependency below) and `verus` (invoked directly on the
//! file from `../verus-proof/`). See that file and `../README.md` for how
//! the `verus_keep_ghost` cfg picks which branch each tool sees.

#[path = "../../verus-proof/kernel_dual.rs"]
mod kernel_dual;

pub use kernel_dual::*;
