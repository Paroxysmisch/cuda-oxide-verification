// Declarative (`macro_rules!`) macros giving kernel code something close to
// the original cuda-oxide access syntax, instead of calling
// `tile_write_at`/`tile_read_at`/`sync_threads_checked` directly.
//
// This is deliberately `macro_rules!`, not a full `proc_macro_attribute`
// that parses and rewrites an arbitrary kernel function body. A real
// `#[kernel]`-style attribute macro would need to be compiled as a
// dynamically-loaded proc-macro, which has to be built with the *same*
// rustc Verus itself embeds (proc macros run inside the compiling
// process) -- buildable, but real cross-toolchain engineering beyond what
// this pass covers. `macro_rules!` macros are expanded by the compiler
// itself as an ordinary language feature, so they work inside a `verus! {
// }` block with no separate toolchain concerns at all. They can't do
// arbitrary static analysis of a kernel body the way a proc macro could,
// but for a fixed, known access pattern they give real syntactic
// convenience: see `stage2_llm_annotated.rs` for what a kernel written
// against them looks like, next to the original `unsafe { TILE[...] }`
// lines they stand in for.

/// `tile_write!(tile, perms, idx, val)` <-> `unsafe { TILE[idx] = val; }`
#[allow(unused_macros)]
macro_rules! tile_write {
    ($tile:expr, $perms:expr, $idx:expr, $val:expr) => {
        tile_write_at($tile, $perms, $idx, $val)
    };
}

/// `tile_read_neighbor!(tile, perms, i, n)` <->
/// `unsafe { TILE[(tid + 1) % N] }`, where `i` is "my" thread index and the
/// macro resolves the `(i+1) % n` neighbor formula itself.
#[allow(unused_macros)]
macro_rules! tile_read_neighbor {
    ($tile:expr, $perms:expr, $i:expr, $n:expr) => {
        tile_read_at($tile, $perms, $i, ($i + 1) % $n)
    };
}

/// `cuda_sync!(tile, perms, data)` <-> `thread::sync_threads()`.
#[allow(unused_macros)]
macro_rules! cuda_sync {
    ($tile:expr, $perms:expr, $data:expr) => {
        sync_threads_checked($tile, $perms, $data)
    };
}

pub(crate) use cuda_sync;
pub(crate) use tile_read_neighbor;
pub(crate) use tile_write;
