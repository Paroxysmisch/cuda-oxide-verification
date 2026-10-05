# cuda-oxide-verification

A small, real, end-to-end prototype of formally verifying a `cuda-oxide`
(now `cuda-rust`) GPU kernel's `unsafe` shared-memory code — race-freedom
*and* functional correctness, checked by [Verus](https://github.com/verus-lang/verus),
not asserted in a comment.

The kernel is the real `shared_test` kernel from
[`cuda-rust/cuda-oxide/crates/rustc-codegen-cuda/examples/sharedmem/src/main.rs`](cuda-rust/cuda-oxide/crates/rustc-codegen-cuda/examples/sharedmem/src/main.rs),
and it is compiled from the **actual `cuda-device`/`cuda-macros` crates** in
the `cuda-rust` submodule — not a lookalike. It writes `TILE[tid] =
data[gid]` to `static mut` shared memory, calls `thread::sync_threads()`,
then reads *its neighbor's* cell: `TILE[(tid + 1) % N]`. Two threads
touching the same `static mut` across a barrier, both `unsafe` — exactly
the pattern cuda-oxide's own safety-model docs name as currently
unenforced by the type system.

## Quick start

```bash
scripts/install_verus.sh   # one-time; builds Verus from source (no prebuilt
                            # release exists for linux-aarch64, i.e. this GB10
                            # box) -- takes a few minutes, needs network
scripts/run_proof.sh        # runs the Verus proof, builds the real kernel
                            # against the real cuda-device crate, runs the
                            # brute-force sanity check -- one PASS/FAIL summary
```

Last run: `verification results:: 7 verified, 0 errors`, the real kernel
builds clean against `cuda-device`, sanity check PASS for N in {1, 2, 4, 8,
16, 32, 256, 1024}.

## One file, two compilations

The real kernel and the verified kernel are **the same source file**:
[`verus-proof/kernel_dual.rs`](verus-proof/kernel_dual.rs). Which half is
"live" is decided automatically by which tool compiles it:

- `cargo build` in [`kernel/`](kernel/) — an ordinary crate depending on the
  real `cuda-device` path dependency from the submodule — compiles the
  `#[cfg(not(verus_keep_ghost))]` branch: the real `#[cuda_module]`/
  `#[kernel]` kernel, `static mut TILE: SharedArray<f32, 256>`, the real
  `unsafe` blocks, untouched.
- `verus` compiles the `#[cfg(verus_keep_ghost)]` branch: the verified
  version, built on `Tile`/`TilePerms` from `stage1_gpu_semantics.rs`.
  (`verus_keep_ghost` is a cfg Verus sets internally — confirmed
  empirically, not assumed: see `NOTES.md`.)

This is the literal, not metaphorical, answer to "strip the
theorem-proving stuff and you're left with the final kernel": compile this
file without Verus and the verified branch is pruned by `cfg` before
type-checking even starts. What `cargo build` produces is the real kernel,
because that's the only branch it ever saw.

## What's actually being claimed

For `shared_test`, for **any block size `n > 0`**:

- **Race-freedom**: the two `unsafe` blocks in the real kernel never race,
  despite both operating on the same `static mut TILE`.
- **Functional correctness**: `out[i] == data[(i + 1) % n]` for every
  thread `i`, for any `n` — verified by Z3 as a single `forall`-quantified
  fact, not just checked on sampled inputs.

It's also cross-checked empirically for N up to 1024 by `verify-proto`'s
`sanity_check` binary — the same belt-and-suspenders pairing of a formal
proof with an independent dynamic check that cuda-oxide's own rustlantis
fuzzer plays relative to its compiler.

**What this does *not* prove**: that the real branch and the verified
branch actually compute the same thing. Nothing automatically checks that
— it's asserted by whoever writes the file (here, standing in for an LLM),
the same translation-validation trust gap named earlier in the design
process, now visible side by side in one file rather than hidden across
two, which helps a reviewer but doesn't close the gap. See `NOTES.md`.

The verified branch is written against `tile_macros.rs` —
`tile_write!`, `tile_read_neighbor!`, `cuda_sync!` — small `macro_rules!`
macros, not a full kernel-parsing `proc_macro_attribute`; `NOTES.md`
explains why.

## Layout

```
cuda-rust/              the submodule (NVIDIA/cuda-rust) -- now a REAL build
                         dependency (kernel/'s Cargo.toml path-depends on
                         cuda-rust/cuda-oxide/crates/cuda-device), not just
                         reference material
verus-proof/             the proof source -- fed directly to the `verus`
                         binary (vstd/verus! need Verus's own patched
                         toolchain, which plain cargo can't provide)
  lib.rs                   entry point `verus` is pointed at
  stage1_gpu_semantics.rs  hand-built, reusable infra (NOT "LLM output"):
                           Tile/TilePerms (arbitrary size), the spawn axiom,
                           the barrier, tile_write_at/tile_read_at
  tile_macros.rs           macro_rules! sugar (tile_write!, etc.)
  kernel_dual.rs           THE kernel -- real branch + verified branch,
                           cfg-selected, same file `kernel/` also compiles
kernel/                 ordinary cargo crate, pinned to cuda-rust's own
                         nightly toolchain, depends on the real cuda-device
  src/lib.rs               includes verus-proof/kernel_dual.rs by path
verify-proto/            plain cargo crate (builds with stable rustc)
  src/pipeline/
    stage0_plain_kernel.rs  standalone reference snapshot (superseded as
                             the verification target by kernel_dual.rs,
                             kept as a readable, dependency-free copy)
  src/bin/sanity_check.rs   independent brute-force cross-check, no Verus
scripts/
  install_verus.sh        detects OS/arch, prefers a prebuilt release,
                           falls back to building from source
  run_proof.sh             single entry point: proof + real build + sanity
NOTES.md                 maps every file back to the underlying design
```

See [`NOTES.md`](NOTES.md) for the concept-by-concept mapping (located
resources, barrier tokens, why race-freedom falls out of struct/index
disjointness rather than an explicit check, the `verus_keep_ghost`
mechanism, what's trusted vs. proved, and what's still simplified).
