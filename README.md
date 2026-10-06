# cuda-oxide-verification

Formal verification of a real `cuda-oxide` (now `cuda-rust`) GPU kernel's
`unsafe` shared-memory code — the exact pattern cuda-oxide's own
safety-model docs name as currently unenforced by the type system: many
SIMT threads touching the same `static mut` shared array across
`sync_threads()` barriers, each access `unsafe`.

Two independent tracks, in this order of maturity:

1. **`dialect-mir` + Viper/Silicon** (`dialect-verify-poc/`, `viper-poc/`,
   and the `dialect-verify` crate added to the `cuda-rust` submodule) —
   verification hosted *inside cuda-oxide's own compiler*, at the
   `dialect-mir` IR stage, via thin ghost ops erased before codegen. This
   is the complete, currently-maintained track: real `verify_*!` macros,
   compiled by the real `rustc` → `mir-importer` pipeline, translated by a
   real (if deliberately scoped) `dialect-mir` → Viper translator, checked
   by the real `silicon` verifier — for race-freedom across any number of
   threads *and* the reduction's numeric correctness.
2. **Verus** (`verus-proof/`, `kernel/`, `verify-proto/`) — an earlier,
   separate prototype: one file, `kernel_dual.rs`, compiles to the real
   kernel under plain `cargo build` and to a Verus-checked proof under
   `verus`, selected by a `cfg` Verus itself sets. Proves race-freedom and
   functional correctness for a different (simpler, single-barrier) real
   kernel. Not extended further once the `dialect-mir`/Viper track proved
   out a cleaner architecture — kept as-is, see its own section below.

## Quick start — track 1 (`dialect-mir` + Viper)

```bash
scripts/install_verus.sh      # one-time: builds Verus from source. Needed
                               # here only for its bundled aarch64 Z3 binary
                               # (Viper ships none for this platform) -- the
                               # Verus proof itself is track 2's, unrelated.
scripts/install_viper.sh      # one-time: downloads the Viper/Silicon tools
scripts/run_viper_poc.sh      # Phases 0-5: hand-built IR through the real
                               # translator, pass + deliberately-broken
                               # variants, at every stage
scripts/run_real_compiler_poc.sh   # the real closure: an actual verify_*!-
                               # annotated kernel, compiled by the real
                               # rustc/mir-importer, through the real
                               # translator, checked by Silicon
scripts/run_matmul_poc.sh     # a second, different real kernel through the
                               # same closure -- a tiled matmul, two shared
                               # tiles, no special hardware features
```

Last run: every phase behaves as documented — real passes verify, every
deliberately-broken variant is rejected with a diagnostic naming the exact
problem, not a generic failure. See `NOTES.md` for the full, phase-by-phase
writeup; the summary below is deliberately short.

### What's actually proven, and how

The real kernel (`cuda-rust/cuda-oxide/crates/rustc-codegen-cuda/examples/verify_demo/src/main.rs`)
is a stride-halving block-sum reduction over 8 shared-memory cells,
annotated inline with ghost calls:

```rust
verify_invariant!(stride <= 4);
verify_perm!(core::ptr::addr_of!(TILE[tid]));
if tid < stride {
    verify_acquire!(core::ptr::addr_of!(TILE[tid + stride]));
    TILE[tid] = TILE[tid] + TILE[tid + stride];
    verify_release!(core::ptr::addr_of!(TILE[tid + stride]));
}
```

These compile through the **real** cuda-oxide pipeline — `rustc` →
`mir-importer` (a patched dispatch table lowers these calls to
`dialect-verify` ghost ops, interleaved with the real `dialect-mir` ops the
rest of the kernel produces) → a debug hook that hands the still-annotated
IR to a real `dialect-mir` → Viper translator → `silicon`/Z3. Then, in the
*same* real build, every ghost op is erased before `mem2reg`/loop
unrolling/LLVM export ever run — `verify_demo.ptx` contains the kernel's
real instructions and zero trace of any of this.

Proven, for this kernel, mechanically:
- **Memory safety and race-freedom for one generic thread**, inductively
  over the loop (no unrolling) — `real_compiler_block_reduce.vpr`.
- **The N-thread combination argument's missing hypothesis**: the
  barrier's redistribution never double-grants a cell, checked generically
  over every valid `stride` — `phase5_nthread_injectivity.vpr`.
- **Numeric correctness**: thread 0 ends up holding the actual sum of all
  8 original elements, via a hand-written recursive "strided sum" function
  and an inductive merge lemma (Viper has no builtin summation) —
  `phase5_numeric_correctness.vpr`.

Every one of these has a deliberately-broken sibling file/variant,
confirmed to be rejected with a diagnostic pointing at the actual problem
— not just a pass-only demo. See `NOTES.md`'s "Phase 5" and "Closing the
loop for real" sections for the complete derivation, every real bug hit
and fixed along the way, and what's honestly still not covered (the
N-thread and numeric-correctness proofs aren't yet mechanically linked to
each other in one unified artifact; both cite the same kind of
by-symmetry argument CSL's parallel rule itself is cited, not re-derived).

### A second kernel: tiled matmul

A second, genuinely different real kernel
(`cuda-rust/cuda-oxide/crates/rustc-codegen-cuda/examples/verify_tiled_matmul/src/main.rs`)
goes through the identical closure — a naive 4x4 tiled matrix multiply,
no WGMMA/tensor-core features, **two** distinct shared tiles instead of
one, and a dot-product loop that borrows a different cell of each tile
every iteration:

```rust
verify_invariant!(k <= 4);
verify_invariant!(k >= 0);
verify_invariant!(row < 4);
verify_invariant!(col < 4);
verify_perm!(core::ptr::addr_of!(TILE_A[tid]));
verify_perm!(core::ptr::addr_of!(TILE_B[tid]));

verify_acquire!(core::ptr::addr_of!(TILE_A[row * 4 + k]));
verify_acquire!(core::ptr::addr_of!(TILE_B[k * 4 + col]));
acc = acc + TILE_A[row * 4 + k] * TILE_B[k * 4 + col];
verify_release!(core::ptr::addr_of!(TILE_A[row * 4 + k]));
verify_release!(core::ptr::addr_of!(TILE_B[k * 4 + col]));
```

Run it with `scripts/run_matmul_poc.sh`. Building this exposed one real
translator gap (every shared-memory access hardcoded the literal tile
name `cells`, fine for one tile, wrong for two) and, via a cleanup pass
that deduplicated `dialect-verify-poc`'s own stale copy of the translator
against this same real one, two more real, previously-latent bugs — see
NOTES.md's "A second real kernel: tiled matmul" section for the complete
derivation of all three.

Proven for this kernel, same two layers as the reduction: **permission
safety** (`real_compiler_tiled_matmul.vpr`), and **numeric correctness**
— thread `(row, col)` ends up holding the real dot product
`sum_{j=0}^{3} A[row][j] * B[j][col]`, not just memory accessed safely
(`viper-poc/matmul_numeric_correctness.vpr`, hand-written the same way
`phase5_numeric_correctness.vpr` is, with a deliberately-broken sibling
confirmed rejected). Markedly simpler than the reduction's numeric proof:
this loop only ever *accumulates* one more term each iteration (never
*combines* two halves), so no merge lemma is needed — a single recursive
`dot_sum` function plus one explicit unfolding step per iteration. See
NOTES.md's "Matmul numeric correctness" section for the full derivation,
including why the kernel's all-reads-after-the-load-phase structure makes
its value-level acquire assumption more direct than the reduction's.

**Generalized to arbitrary N** (`viper-poc/matmul_numeric_correctness_arbitrary_n.vpr`
+ `_broken.vpr`): since this proof is hand-written (not derived from real
compiler output), `n` is free to be a genuine symbolic parameter instead
of a baked-in literal — unlike `real_compiler_tiled_matmul.vpr`, which
stays fixed at N=4 forever, since a real compiled kernel's tile size is
permanently baked in by monomorphization. The real difficulty:
`row*n+k < n*n` is *nonlinear* (a product of two symbolic values), which
Z3 doesn't discharge unprompted, and — found empirically, not assumed —
separately-proved nonlinear facts don't always compose via a later
`assert` the way equivalent linear facts do. The fix, `bound_holds`: an
inductive function that *computes* `a*n+b` by repeated addition rather
than ever re-deriving the product, carrying its own bound as a
postcondition, so every index travels with its bound already attached —
see NOTES.md's "Generalizing the matmul numeric proof to arbitrary N"
section for the complete derivation, including the two dead ends hit
first (chaining nonlinear asserts; quantified preconditions whose
triggers didn't reach Viper's auto-generated termination proof) and why
each failed.

## Layout

```
cuda-rust/                        the submodule (NVIDIA/cuda-rust) -- a
                                   real build dependency, not reference
                                   material. New crates added to it:
  cuda-oxide/crates/dialect-verify/   the ghost-op dialect (verify.assert,
                                   verify.invariant, verify.perm,
                                   verify.acquire, verify.release) plus
                                   the dialect-mir -> Viper translator
  cuda-oxide/crates/cuda-device/src/verify.rs
                                   verify_assert!/verify_invariant!/
                                   verify_perm!/verify_acquire!/
                                   verify_release! -- the real macro
                                   front end
  cuda-oxide/crates/mir-importer/  patched: dispatches the above to
                                   dialect-verify ops; erases them again
                                   right after each function's own
                                   dialect-mir verification
  .../examples/verify_demo/       the real, compiling example kernel
  .../examples/verify_demo_broken/    same kernel, one deliberate bug
  .../examples/verify_tiled_matmul/   a second, different real kernel:
                                   tiled matmul, two shared tiles
  .../examples/verify_tiled_matmul_broken/  same kernel, one deliberate bug

dialect-verify-poc/              hand-built-IR demos: the SAME ghost-op
                                   mechanism and translator (a path
                                   dependency on the real dialect-verify
                                   crate above, not its own copy -- see
                                   NOTES.md), run against hand-constructed
                                   dialect-mir (via pliron directly)
                                   rather than real compiler output, so a
                                   quick, dependency-light way to probe
                                   the translator -- phase1-4 binaries,
                                   each runnable and self-explanatory
  src/bin/phase1_erasure.rs         ghost op coexists with a real op,
                                   erasure removes only the ghost one
  src/bin/phase2_translate.rs       a real automated translator run, on a
                                   hand-built if/else kernel, pass + fail
  src/bin/phase3_reduction.rs       the full stride-halving reduction,
                                   hand-built, a genuine loop (no
                                   unrolling); `--broken` for the rejected
                                   variant
  src/bin/phase4_llm_annotations.rs  a fresh, context-isolated agent's
                                   independently-drafted annotations,
                                   run through the same real pipeline

viper-poc/                        the `.vpr` files every phase produces
                                   and checks, plus the hand-written
                                   Phase 0 sanity tests, Phase 5's
                                   N-thread/numeric-correctness proofs,
                                   matmul_numeric_correctness.vpr (N=4),
                                   and matmul_numeric_correctness_arbitrary_n.vpr
                                   (symbolic N)
                                   (pass + deliberately-broken variants
                                   for every claim)

scripts/
  install_viper.sh                 downloads Viper/Silicon
  run_viper_poc.sh                  Phases 0-5, hand-built IR
  run_real_compiler_poc.sh          the real rustc/mir-importer closure
                                   (block-sum reduction)
  run_matmul_poc.sh                 the same closure, a second kernel
                                   (tiled matmul, two shared tiles)
  install_verus.sh / run_proof.sh   track 2 (Verus), below

verus-proof/, kernel/, verify-proto/, docs/
                                   track 2 -- see "Track 2: Verus" below

NOTES.md                          the complete, honest writeup: every
                                   phase, every real bug found and fixed,
                                   what's proven vs. cited vs. still open
```

## Track 2: Verus

A small, real, end-to-end prototype of formally verifying a different
real `cuda-oxide` kernel's `unsafe` shared-memory code — `shared_test`
from
[`cuda-rust/cuda-oxide/crates/rustc-codegen-cuda/examples/sharedmem/src/main.rs`](cuda-rust/cuda-oxide/crates/rustc-codegen-cuda/examples/sharedmem/src/main.rs),
compiled against the actual `cuda-device`/`cuda-macros` crates. It writes
`TILE[tid] = data[gid]`, calls `thread::sync_threads()`, then reads *its
neighbor's* cell: `TILE[(tid + 1) % N]`.

```bash
scripts/install_verus.sh   # one-time; builds Verus from source (no prebuilt
                            # release exists for linux-aarch64)
scripts/run_proof.sh        # runs the Verus proof, builds the real kernel
                            # against the real cuda-device crate, runs the
                            # brute-force sanity check -- one PASS/FAIL summary
```

Last run: `verification results:: 7 verified, 0 errors`, the real kernel
builds clean against `cuda-device`, sanity check PASS for N in {1, 2, 4, 8,
16, 32, 256, 1024}.

**One file, two compilations.** The real kernel and the verified kernel
are the same source file, [`verus-proof/kernel_dual.rs`](verus-proof/kernel_dual.rs).
`cargo build` in [`kernel/`](kernel/) compiles the `#[cfg(not(verus_keep_ghost))]`
branch — the real, untouched kernel; `verus` compiles the
`#[cfg(verus_keep_ghost)]` branch — the verified version, built on
`Tile`/`TilePerms` from `stage1_gpu_semantics.rs`. `verus_keep_ghost` is a
cfg Verus sets internally, confirmed empirically (see `NOTES.md`).

**What's claimed**, for any block size `n > 0`: race-freedom between the
two `unsafe` blocks, and `out[i] == data[(i + 1) % n]` for every thread
`i`, proven by Z3 as a single `forall`-quantified fact — cross-checked
empirically up to N=1024 by `verify-proto`'s `sanity_check` binary.
**What it does not prove**: that the real branch and the verified branch
compute the same thing — nothing mechanically checks that; it's asserted
by whoever writes the file. This is exactly the architectural ceiling that
motivated moving to track 1 (verifying the real, unmodified kernel source
directly, with no separate "verified twin" to keep in sync) — see
`NOTES.md`'s early sections for the full design history.

See [`NOTES.md`](NOTES.md) for the concept-by-concept mapping across both
tracks: located resources, barrier tokens, the `verus_keep_ghost`
mechanism, the full `dialect-mir`/Viper architecture and every phase's
real results, what's trusted vs. proved, and what's still open.
