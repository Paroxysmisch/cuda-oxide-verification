// The dual-mode kernel: ONE source file, compiled two different ways by two
// different tools, selected automatically by the `verus_keep_ghost` cfg
// flag Verus sets internally (confirmed empirically: `verus` checks only
// the `#[cfg(verus_keep_ghost)]` branch; plain `rustc`/`cargo` -- with no
// knowledge of Verus at all -- compiles only the `#[cfg(not(verus_keep_ghost))]`
// branch, and never even tries to resolve `vstd`).
//
//   - `#[cfg(not(verus_keep_ghost))]`: the REAL kernel. Not a lookalike --
//     this is `shared_test`, verbatim, using the real `cuda_device` crate
//     from the `cuda-rust` submodule (`#[cuda_module]`, `#[kernel]`,
//     `SharedArray`, `DisjointSlice`, `thread::*`, the real `unsafe`
//     blocks). This is what `cargo build` in `../kernel/` produces, and
//     it's exactly what cuda-oxide's own compiler would be handed.
//   - `#[cfg(verus_keep_ghost)]`: the verified version, built on
//     `stage1_gpu_semantics`'s `Tile`/`TilePerms` and the `tile_macros`.
//     This is what `verus` actually checks.
//
// This is the literal answer to "annotate the kernel with the
// theorem-proving stuff, and if that stuff is removed we're left with the
// final kernel": the removal isn't metaphorical. Compile this file without
// Verus and the verified branch doesn't exist in the output at all -- cfg
// pruning happens before type-checking even starts. What's left is
// byte-for-byte the real kernel, because it was never anything else.
//
// What this does *not* do -- worth being exact about, not just optimistic:
// Verus checks the second branch; nothing automatically checks that the
// two branches compute the same thing. That equivalence is asserted by
// whoever writes this file (here: me, standing in for the LLM), the same
// translation-validation trust gap named a few turns ago -- now visible
// side by side in one place instead of hidden across separate files, which
// is a real improvement for a reviewer, but not a proof of equivalence.

#[cfg(not(verus_keep_ghost))]
mod real_kernel {
    use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

    #[cuda_module]
    mod kernels {
        use super::*;

        #[kernel]
        pub fn shared_test(data: &[f32], mut out: DisjointSlice<f32>) {
            static mut TILE: SharedArray<f32, 256> = SharedArray::UNINIT;

            let tid = thread::threadIdx_x() as usize;
            let gid = thread::index_1d().get();

            // Write to shared memory.
            unsafe {
                TILE[tid] = data[gid];
            }

            thread::sync_threads();

            // Read from shared memory (neighbor).
            unsafe {
                let neighbor_idx = (tid + 1) % 256;
                if let Some(out_elem) = out.get_mut(thread::index_1d()) {
                    *out_elem = TILE[neighbor_idx];
                }
            }
        }
    }

    pub use kernels::*;
}

#[cfg(not(verus_keep_ghost))]
pub use real_kernel::*;

#[cfg(verus_keep_ghost)]
use crate::stage1_gpu_semantics::*;
#[cfg(verus_keep_ghost)]
use crate::tile_macros::{cuda_sync, tile_read_neighbor, tile_write};
#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// The verified counterpart to `real_kernel::shared_test` above -- same
/// claim as the two-phase write/barrier/read structure, generalized to any
/// block size `n`. See `NOTES.md` for what's still simplified (the
/// whole-block loop here versus a per-thread function lifted by a separate
/// rule, and `DisjointSlice` modeled by reusing `Tile` rather than its own
/// axiom).
pub fn shared_test_verified(
    n: usize,
    tile: &Tile,
    tile_perms: TilePerms,
    out_tile: &Tile,
    out_perms: TilePerms,
    data: &[f32],
) -> (r: TilePerms)
    requires
        n > 0,
        tile.cells.len() == n,
        tile_perms.perms.len() == n,
        out_tile.cells.len() == n,
        out_perms.perms.len() == n,
        data.len() == n,
        forall|i: int| 0 <= i < n ==> #[trigger] tile_perms.perms[i]@.id() == tile.cells[i as int].id(),
        forall|i: int| 0 <= i < n ==> #[trigger] out_perms.perms[i]@.id() == out_tile.cells[i as int].id(),
    ensures
        r.perms.len() == n,
        forall|i: int|
            #![trigger r.perms[i]@.id()]
            #![trigger r.perms[i]@.value()]
            0 <= i < n ==> *r.perms[i]@.value() == data[(i + 1) % (n as int)],
{
    let mut tile_perms = tile_perms;

    let mut i: usize = 0;
    while i < n
        invariant
            i <= n,
            tile.cells.len() == n,
            tile_perms.perms.len() == n,
            data.len() == n,
            forall|k: int|
                0 <= k < i ==> #[trigger] tile_perms.perms[k]@.id() == tile.cells[k as int].id()
                    && *tile_perms.perms[k]@.value() == data[k as int],
            forall|k: int|
                i <= k < n ==> #[trigger] tile_perms.perms[k]@.id() == tile.cells[k as int].id(),
        decreases n - i,
    {
        tile_write!(tile, &mut tile_perms, i, data[i]);
        i += 1;
    }

    let tile_perms = cuda_sync!(tile, tile_perms, data);

    let mut out_perms = out_perms;
    let mut j: usize = 0;
    while j < n
        invariant
            j <= n,
            tile.cells.len() == n,
            out_tile.cells.len() == n,
            tile_perms.perms.len() == n,
            out_perms.perms.len() == n,
            data.len() == n,
            forall|k: int|
                #![trigger tile_perms.perms[k]@.id()]
                #![trigger tile_perms.perms[k]@.value()]
                0 <= k < n ==> tile.cells[(k + 1) % (n as int)].id() == tile_perms.perms[k]@.id()
                    && *tile_perms.perms[k]@.value() == data[(k + 1) % (n as int)],
            forall|k: int|
                #![trigger out_perms.perms[k]@.id()]
                #![trigger out_perms.perms[k]@.value()]
                0 <= k < j ==> out_perms.perms[k]@.id() == out_tile.cells[k as int].id()
                    && *out_perms.perms[k]@.value() == data[(k + 1) % (n as int)],
            forall|k: int|
                j <= k < n ==> #[trigger] out_perms.perms[k]@.id() == out_tile.cells[k as int].id(),
        decreases n - j,
    {
        let v = tile_read_neighbor!(tile, &tile_perms, j, n);
        tile_write!(out_tile, &mut out_perms, j, v);
        j += 1;
    }

    out_perms
}

} // verus!
