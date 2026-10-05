// Entry point `verus` is pointed at directly (see ../scripts/run_proof.sh).
// Not a cargo crate -- vstd/verus! only work under Verus's own patched
// toolchain, not plain stable rustc. See ../README.md.

pub mod stage1_gpu_semantics;
pub mod tile_macros;
pub mod kernel_dual;
