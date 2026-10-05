//! Plain-Rust half of the prototype. See `../../README.md` for the full
//! picture: this crate holds stage 0 (the unannotated kernel snapshot) and
//! the `sanity_check` binary. The Verus-verified stages 1 and 2 live in
//! `../../verus-proof/` and are checked by the `verus` binary directly --
//! they are not, and cannot be, compiled by plain `cargo build`, because
//! `vstd`/`verus!` only work under Verus's own patched toolchain.

pub mod pipeline;
