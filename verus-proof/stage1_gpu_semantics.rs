// Stage 1 -- the reusable GPU-semantics library, generalized to arbitrary
// shared-memory sizes (the N=4 version proved the technique; this is the
// real thing it was always meant to grow into -- see NOTES.md).
//
//   1. A located, linear permission per shared-memory cell (`Tile` /
//      `TilePerms`, now a `Vec<PCell<f32>>` / `Vec<Tracked<PointsTo<f32>>>`
//      pair instead of four named fields).
//   2. The one genuinely trusted GPU fact: hardware thread-ID uniqueness
//      (`thread_index_x`).
//   3. The one genuinely GPU-specific primitive with no Rust analogue at
//      all: the barrier (`sync_threads_checked`), now a `forall`-quantified
//      permission redistribution instead of four ground facts.
//
// `Tile::new` is the first place in this whole prototype that needs an
// honest loop invariant -- building N cells one at a time, carrying "every
// cell built so far has a matching permission" across each iteration.

use vstd::cell::pcell::{PCell, PointsTo};
use vstd::prelude::*;

verus! {

/// Shared memory for an N-thread block: N independent `PCell`s. Matches
/// cuda-oxide's `static mut TILE: SharedArray<f32, N>` for any N, not a
/// fixed size baked into the type.
pub struct Tile {
    pub cells: Vec<PCell<f32>>,
}

/// The permission tokens for a `Tile`, one per cell, indexed the same way.
/// `perms[i]` is the permission *currently held by thread i* -- after a
/// barrier it may name a different cell than `cells[i]`, same role as in
/// the N=4 version, just addressed by index instead of by field name.
pub struct TilePerms {
    pub perms: Vec<Tracked<PointsTo<f32>>>,
}

impl Tile {
    /// Spawn-time construction of an N-cell tile. The loop invariant is the
    /// honest version of the N=4 version's four `ensures` lines: "every
    /// cell built so far (`0..i`) has a permission naming it."
    pub fn new(n: usize) -> (r: (Tile, TilePerms))
        ensures
            r.0.cells.len() == n,
            r.1.perms.len() == n,
            forall|i: int| 0 <= i < n ==> #[trigger] r.1.perms[i]@.id() == r.0.cells[i as int].id(),
    {
        let mut cells: Vec<PCell<f32>> = Vec::new();
        let mut perms: Vec<Tracked<PointsTo<f32>>> = Vec::new();
        let mut i: usize = 0;
        while i < n
            invariant
                i <= n,
                cells.len() == i,
                perms.len() == i,
                forall|k: int| 0 <= k < i ==> #[trigger] perms[k]@.id() == cells[k as int].id(),
            decreases n - i,
        {
            let (c, p) = PCell::new(0.0f32);
            cells.push(c);
            perms.push(p);
            i += 1;
        }
        (Tile { cells }, TilePerms { perms })
    }
}

/// --- The one trusted GPU fact (the "spawn axiom") ---
/// Unchanged in spirit from the N=4 version -- see its comment there. Still
/// not actually called by stage 2: with the thread loop's index `i` playing
/// the role of "thread i" throughout, there's no runtime value to fetch,
/// same as before.
#[verifier::external_body]
pub fn thread_index_x(tid_hint: u32, n: u32) -> (tid: u32)
    requires tid_hint < n,
    ensures tid == tid_hint,
{
    tid_hint
}

/// --- The barrier: `thread::sync_threads()`, generalized ---
///
/// Same role as the N=4 version's `sync_threads_checked`: a permission
/// redistribution along `j = (i+1) % n`, the real kernel's `neighbor_idx`
/// formula -- except now it's one `forall`-quantified fact instead of four
/// ground ones, because `n` is a runtime value, not a literal.
/// `data` is threaded through the barrier's own contract directly, rather
/// than relating `out`'s values back to `perms`'s values abstractly. That's
/// a deliberate simplification: the fully kernel-agnostic version (state
/// only a permission redistribution, let the caller chain it against
/// whatever the pre-barrier values happened to be) needs Z3 to combine two
/// independently-quantified `forall`s across the index shift `j=(i+1)%n`,
/// which needs real triggering help to go through. Since this function is
/// `external_body` either way -- a trusted axiom, not something proved --
/// there's no honesty cost to stating the axiom in the more directly
/// usable form. The cost is reusability: this contract is now shaped
/// around "callers that have already related their permissions to a
/// `data` buffer," not a fully generic barrier.
#[verifier::external_body]
pub fn sync_threads_checked(tile: &Tile, perms: TilePerms, data: &[f32]) -> (out: TilePerms)
    requires
        perms.perms.len() == tile.cells.len(),
        data.len() == tile.cells.len(),
        forall|i: int|
            0 <= i < tile.cells.len() ==> #[trigger] perms.perms[i]@.id() == tile.cells[i].id()
                && *perms.perms[i]@.value() == data[i],
    ensures
        out.perms.len() == tile.cells.len(),
        forall|i: int|
            0 <= i < tile.cells.len() ==> {
                let n = tile.cells.len() as int;
                let j = (i + 1) % n;
                &&& #[trigger] out.perms[i]@.id() == tile.cells[j].id()
                &&& *out.perms[i]@.value() == data[j]
            },
{
    unimplemented!()
}

/// Reusable per-index write, the generalized counterpart to the N=4
/// version's `write_cell` -- now operating on a `Vec` slot instead of a
/// named field. Proved once; instantiated by a loop in stage 2 instead of
/// four hand-copied call sites.
pub fn tile_write_at(tile: &Tile, perms: &mut TilePerms, idx: usize, v: f32)
    requires
        idx < old(perms).perms.len(),
        old(perms).perms.len() == tile.cells.len(),
        old(perms).perms[idx as int]@.id() == tile.cells[idx as int].id(),
    ensures
        final(perms).perms.len() == old(perms).perms.len(),
        final(perms).perms[idx as int]@.id() == tile.cells[idx as int].id(),
        *final(perms).perms[idx as int]@.value() == v,
        forall|k: int|
            #![trigger final(perms).perms[k]@.id()]
            #![trigger final(perms).perms[k]@.value()]
            0 <= k < final(perms).perms.len() && k != idx ==> {
                &&& final(perms).perms[k]@.id() == old(perms).perms[k]@.id()
                &&& final(perms).perms[k]@.value() == old(perms).perms[k]@.value()
            },
{
    let Tracked(mut p) = perms.perms.remove(idx);
    tile.cells[idx].write(Tracked(&mut p), v);
    perms.perms.insert(idx, Tracked(p));
}

/// Reusable per-index read, the generalized counterpart to `read_cell`.
/// `idx` here is the *cell* index (already resolved from whatever thread
/// owns the permission, e.g. `(i + 1) % n` after the barrier) -- the caller
/// supplies both the cell to read and the permission that should match it.
pub fn tile_read_at(tile: &Tile, perms: &TilePerms, perm_idx: usize, cell_idx: usize) -> (v: f32)
    requires
        perm_idx < perms.perms.len(),
        cell_idx < tile.cells.len(),
        perms.perms[perm_idx as int]@.id() == tile.cells[cell_idx as int].id(),
    ensures
        v == *perms.perms[perm_idx as int]@.value(),
{
    let p: &Tracked<PointsTo<f32>> = &perms.perms[perm_idx];
    *tile.cells[cell_idx].borrow(Tracked(p.borrow()))
}

} // verus!
