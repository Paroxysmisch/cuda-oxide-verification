# Design notes: mapping files back to the underlying ideas

This maps every file in the prototype to a specific piece of the design
worked out before any code was written, so the reasoning survives
independently of that conversation.

## The pipeline stages = the LLM-assisted workflow

| File | Role | Who/what produces it |
|---|---|---|
| `verus-proof/kernel_dual.rs` (`#[cfg(not(verus_keep_ghost))]` branch) | The real kernel, compiled from the real `cuda-device`/`cuda-macros` crates | A cuda-oxide author, unchanged |
| `verus-proof/stage1_gpu_semantics.rs` | Reusable GPU-semantics library: located resources, the spawn axiom, the barrier contract | A verification engineer, **once**, reused across kernels |
| `verus-proof/kernel_dual.rs` (`#[cfg(verus_keep_ghost)]` branch) | This specific kernel's `requires`/`ensures`/the rewritten `unsafe`-to-`PCell` wiring | **This is the part an LLM would draft.** Hard-coded here instead of actually calling a model, per the brief — but every comment in it is written as if explaining a draft to be checked, not a fact already established |
| `verify-proto/src/bin/sanity_check.rs` | Independent brute-force cross-check of the same claim | Backstop, same role as cuda-oxide's own differential fuzzer |

`verify-proto/src/pipeline/stage0_plain_kernel.rs` is an earlier,
dependency-free snapshot of the same real kernel, kept as a readable
reference; it is no longer what gets verified — `kernel_dual.rs`'s own
real branch is, since it's now the real crate rather than a lookalike.

The whole point of splitting the reusable library from the per-kernel part
this way: the library is the expensive, one-time intellectual work (same
as building `DisjointSlice` itself was for cuda-oxide); the per-kernel part
is the cheap, repeatable work an LLM is well-suited to draft and a solver
is well-suited to referee.

## One file, two compilers: the `verus_keep_ghost` mechanism

This is the piece that actually answers "annotate the kernel with macros,
and if the theorem-proving stuff is removed we're left with the final
kernel" — literally, not by analogy. `verus` sets a cfg flag,
`verus_keep_ghost`, internally whenever it processes a file (confirmed
empirically, not assumed from documentation — see below); nothing else
ever sets it. `kernel_dual.rs` has two top-level branches gated on it:

```rust
#[cfg(not(verus_keep_ghost))]
mod real_kernel { /* the real #[cuda_module]/#[kernel] shared_test,
                     real cuda_device types, real unsafe blocks */ }

#[cfg(verus_keep_ghost)]
verus! { /* shared_test_verified, built on Tile/TilePerms */ }
```

`cfg` pruning happens before type-checking, so under plain `cargo
build`/`rustc` the *entire* second branch — including every `use
vstd::...` inside it — is deleted before the compiler even tries to
resolve `vstd` as a crate. That was confirmed directly, not assumed: a
minimal two-branch test file was compiled both ways — `verus` on it
reported `1 verified, 0 errors` against a trivial contract on the first
branch; the identical file compiled with plain `rustup run
nightly-2026-08-28 rustc` ran the *second* branch's arithmetic (deliberately
different, to prove it wasn't silently checking the same code) and printed
the expected result, with no mention of `vstd` anywhere. Only then was
`kernel_dual.rs` built the same way, and `kernel/`'s `cargo build` against
the real `cuda-device` crate confirmed the real branch compiles clean
(after declaring `verus_keep_ghost` in `kernel/Cargo.toml`'s
`[lints.rust] unexpected_cfgs` so rustc's check-cfg lint doesn't flag it as
a typo — cosmetic, not load-bearing).

**What this buys**: compiling `kernel_dual.rs` without Verus produces
exactly the real kernel, because the verified branch was never there to
produce anything else. There is no separate "stripped" build step, no risk
of the two drifting apart through a forgotten sync — they're the same
`cfg`-pruned file, always.

**What this does not buy**: proof that the two branches compute the same
thing. `verus` only ever sees one branch; `rustc` only ever sees the
other; no tool in this pipeline checks them against each other. That's the
translation-validation gap named a few turns earlier in the design
process — still open, now structurally visible (a reviewer can read both
branches side by side in one file) rather than hidden across separately
maintained files, which is a real improvement for auditability but not a
closed gap. Closing it for real would mean the differential-testing idea
floated earlier: compile the real branch through cuda-oxide's actual
backend, run it on this machine's GPU, compile the verified branch with
`verus --compile`, and diff outputs across many inputs. Not done here —
flagged, not attempted, since it needs `llc`/`clang` which this box
doesn't currently have installed (see "What this does not cover" below).

## Using the real `cuda-rust` submodule, for real

`kernel/Cargo.toml` path-depends on
`cuda-rust/cuda-oxide/crates/cuda-device`, pinned to the exact nightly
(`nightly-2026-08-28`) the submodule's own `rust-toolchain.toml` specifies
(already installed on this box). `cuda-device` and its own dependency
`cuda-macros` both build cleanly under that pinned nightly alone — neither
needs `rustc_private`/`rustc-dev`, despite the whole `cuda-rust` workspace
pinning those components, because the *codegen backend* needs them and
`cuda-device` (a leaf, user-facing crate: types and intrinsic stubs, no
compiler-internals access) does not. That's a real, useful fact about the
crate graph, not an assumption: `cuda-device`'s own `Cargo.toml` lists only
`cuda-macros` as a dependency, and `grep` across both crates for
`rustc_private`/`extern crate rustc` turned up nothing.

What this does *not* include: the actual codegen backend
(`rustc-codegen-cuda`), which needs `llc` (LLVM 21+) and `clang-21` —
neither is installed on this machine (checked directly, not assumed), and
getting them would be a separate, open-ended undertaking (build-from-source
risk similar to the Verus bootstrap, but for a much larger project). So
`shared_test` in `kernel_dual.rs` compiles as ordinary Rust against the
real types, and type-checks exactly as cuda-oxide's own compiler would
see it — but this prototype does not go all the way to emitting PTX or
running on the GPU that's sitting right here. That's a real, open next
step, not something achieved.

## Located resources (`Tile` / `TilePerms`)

Maps to the "located resources" idea from Kuiper (`on l p`, `is_send_across`)
and to cuda-oxide's own address-space tag on `mir.ptr<T, addrspace: N>`.
Rather than build a general `Loc` hierarchy (cpu / gpu-global / block-shared
/ ...), this prototype only needed one location — one block's shared memory
— so it's realized directly as `Tile`: `Vec<PCell<f32>>`, arbitrary length,
with a matching `TilePerms { perms: Vec<Tracked<PointsTo<f32>>> }`.

**This is also where race-freedom actually comes from**, and it's worth
being precise about the mechanism, because it's not an explicit check
anywhere in the code: two different `Vec` *indices* are two different
tracked resources *by construction*, the same way two different `PCell`s
have two different `CellId`s by construction. `tile_write_at`'s own
contract (see below) only ever grants exclusive access to the one index it
was called with, and its `forall k != idx` clause proves every other index
is untouched — that's the formal version of the sentence already in
cuda-oxide's own safety-model docs about `DisjointSlice`: "the borrow
checker sees a single `&mut T` per thread... the launch proof makes their
linearization disjoint."

## The spawn axiom (`thread_index_x`) and the barrier (`sync_threads_checked`)

Both are `#[verifier::external_body]` — Verus is told to trust the
`requires`/`ensures` without checking a body, because there is no body to
check: these describe hardware behavior, not Rust semantics. This is the
entire, irreducible trust boundary of the whole proof:

- `thread_index_x`: hardware thread-ID uniqueness. Same status as
  cuda-oxide's own trusted index-function table in the safety-model chapter.
  Still not actually called anywhere in kernel_dual.rs's verified branch — with the loop index `i`
  playing the role of "thread i" throughout, there's no runtime value to
  fetch. Kept for documentation of the trust boundary, same as before.
- `sync_threads_checked`: the barrier. This is the one primitive in the
  whole prototype with **no ordinary-Rust analogue at all** — nothing in
  Rust's language semantics describes what happens when hardware runs many
  concurrent copies of a function body over one shared memory region. Its
  contract models the barrier not as a generic broadcast but as the
  specific permission *redistribution* this kernel needs: thread `i` trades
  its own cell's permission for cell `(i+1) % n`'s — the exact
  `neighbor_idx` formula from the real kernel, now a single
  `forall`-quantified fact covering any `n`, not four ground facts.

  **One deliberate simplification worth being explicit about**: the
  contract threads `data: &[f32]` through directly (`out[i] == data[(i+1) %
  n]`) rather than staying fully kernel-agnostic (`out[i] == whatever
  perms[(i+1)%n] held before the call`, let the caller chain it against
  `data` separately). The fully general version needs Z3 to combine two
  independently-quantified `forall`s across the index shift `(i+1)%n`,
  which real experimentation against the actual `verus` binary showed
  needs substantial manual triggering to go through reliably. Since this
  function is `external_body` either way — a trusted axiom, not something
  proved — there's no soundness cost to stating it in the more directly
  usable form, only a reusability cost: this contract is now shaped around
  "callers that have already related their permissions to a `data`
  buffer," not a fully generic barrier usable by any future kernel
  unchanged.

If either axiom were wrong — say, `sync_threads_checked`'s `ensures`
redistributed two threads onto the *same* cell — the verified branch's reads would
fail to find a matching permission and the proof would not go through.
That failure mode is deliberate: it's the thing that would actually catch a
modeling mistake, the same way a race-condition bug would be caught in the
real, hand-worked argument from earlier in the design conversation.

## Arbitrary block sizes, for real

An earlier pass of this prototype fixed the block size at a concrete `N=4`
specifically to avoid `forall`-quantified reasoning, after a hand-derived
`forall`-quantified invariant turned out to be wrong earlier in the design
conversation. That version proved the technique; this version is what it
was always meant to grow into. `Tile`/`TilePerms` are now `Vec`-backed and
generic over a runtime `n`, `Tile::new` builds them with a genuine loop and
loop invariant (`cells.len() == i`, `perms.len() == i`, "every cell built
so far has a matching permission," `decreases n - i`), and both of
`shared_test_verified`'s phases are real loops with real invariants instead
of four unrolled copies. Getting the loop invariants and `forall` triggers
right this time took real iteration against the actual `verus` binary —
several rounds of genuine SMT trigger failures (Verus reporting "invariant
not satisfied," low-confidence auto-chosen triggers, assertions that
needed explicit `#![trigger ...]` annotations to go through at all) before
converging on `verification results:: 7 verified, 0 errors`. That's worth
noting plainly: even with the technique already validated at N=4, scaling
to arbitrary N was real, nontrivial proof engineering, not a mechanical
substitution.

The generalization gap from the earlier version is now closed, not just
covered empirically — `sanity_check.rs` still brute-force-checks the same
claim for N up to 1024 as an independent backstop, but the Verus proof
itself now covers any `n > 0`, including the real kernel's actual 256.

## Arbitrary input shapes: `data: &[f32]` and `DisjointSlice` via a second `Tile`

The kernel now takes `data: &[f32]` of genuinely arbitrary length, matching
the real signature — and this needed *no* new axiom at all. A shared
immutable reference has no concurrent-mutation hazard (many threads reading
the same `&[f32]` is exactly what `&T` already guarantees safe), so
Verus's native slice support covers it directly.

The real kernel's output parameter, `out: DisjointSlice<f32>`, is modeled
by reusing the *same* `Tile`/`TilePerms` abstraction a second time, rather
than building a separate formalization of `DisjointSlice` from scratch:
each thread keeps its own output permission for the whole function (no
barrier/redistribution on that side, since every thread only ever writes
its own output slot). This is a legitimate simplification, not a dodge —
`DisjointSlice`'s actual safety argument *is* exactly this simpler case
(no cross-thread redistribution involved) — but it does mean
`DisjointSlice::get_mut`'s own contract was never independently written
down or axiomatized; it's implicitly asserted to be *at least as safe as*
reusing `Tile`. A fully faithful treatment would give `DisjointSlice` its
own named axiom the way `SharedArray` effectively has one via `Tile`.

## The macros: `tile_write!`, `tile_read_neighbor!`, `cuda_sync!`

`verus-proof/tile_macros.rs` gives the verified branch something closer to
the original kernel's syntax than calling `tile_write_at`/`tile_read_at`/
`sync_threads_checked` directly — compare `kernel_dual.rs`'s
`tile_write!(tile, &mut tile_perms, i, data[i]);` against the real
branch's `unsafe { TILE[tid] = data[gid]; }` a few lines above it in the
same file.

**These are `macro_rules!` macros, not a full `#[kernel]`-style
`proc_macro_attribute`.** That's a deliberate scope decision, named
explicitly rather than quietly substituted: a real attribute macro that
parses an arbitrary kernel body and generates the right `Tile`/permission
wiring automatically would need to be compiled as a dynamically-loaded
proc-macro — which has to be built with the *same* rustc Verus itself
embeds, since proc macros run inside the compiling process. That's real,
buildable cross-toolchain engineering, not attempted here.
`macro_rules!` macros sidestep it entirely (expanded by the compiler as an
ordinary language feature, no separate toolchain concerns), at the cost of
only handling a fixed, known access shape rather than arbitrary kernel
bodies — exactly the "small catalog of recognized patterns, not one macro
per unsafe line" idea from the design conversation, just not yet grown
into the fuller pattern-recognizing version.

## A second, separate verification track: `dialect-mir` + Viper (Phases 0-2)

Everything above is the Verus track (`verus-proof/`, `kernel/`). A later
design conversation concluded Verus's external-type-contract model has a
structural ceiling — `SharedArray::index_mut(&mut self, idx)` has no
channel to attach a permission to, and more fundamentally takes `&mut self`
on the *whole* array, so every thread independently forging its own
`&mut TILE` via `unsafe` is already a conflict before `idx` enters the
picture at all. The fix isn't a cleverer Verus encoding — it's verifying at
a point where that conflict has already been resolved: `cuda-rust`'s own
`dialect-mir`, after `mir-importer` but before `mem2reg`/loop-unrolling,
where `TILE[tid] = v` is already a plain `mir.shared_alloc` +
`mir.ptr_offset` + `mir.store` sequence, not an opaque trait call. That
reduction isn't something to build — `mir-importer` already does it, for
every kernel, to emit correct PTX. Riding on it reuses a trust boundary
cuda-oxide's users already depend on, rather than adding a new one.

This is a **separate, standalone proof-of-concept track**, living in
`dialect-verify-poc/` and `viper-poc/`, not merged into the Verus track.
Three phases actually built and run (not just designed):

**Phase 0 — due diligence, both items confirmed empirically:**
- Viper's `silicon` verifier ships no GitHub releases, but `viper-ide`
  publishes a prebuilt `ViperToolsLinux.zip` (pure JVM bytecode — no
  aarch64 build needed, unlike Verus). `SiliconRunner` runs directly via
  `java -cp viperserver.jar viper.silicon.SiliconRunner`; the bundled Z3 is
  x86-64 and doesn't run here, but the aarch64 Z3 already built for Verus
  works as a drop-in replacement via `--z3Exe`. `scripts/install_viper.sh`.
- Traced (with file/line citations, not inference) exactly how
  `SharedArray` indexing lowers in the real `mir-importer`:
  `translator/facts.rs:254` (`self_ty_is_shared_array`),
  `translator/terminator/mod.rs:971-974,3221-3252` (the dispatch gate),
  `translator/terminator/intrinsics/memory.rs:756-854` (emits
  `mir.ptr_offset`), `translator/statement.rs:166-224` (the generic
  `*ptr = v` path, unaware it came from `SharedArray`, emits `mir.store`).
  Confirmed: no call-like residue, and the importer never reads
  `index_mut`'s `unreachable!()` body — interception happens by matching
  the `Call` terminator's callee path and `Self` type, before the callee's
  own MIR is ever requested.
- `viper-poc/phase0_test.vpr`: hand-written permission sanity checks,
  run against the real binary, not just read. Two positive cases pass; two
  negative cases fail with exactly the right diagnostics — one for using a
  cell with no permission at all, one (closer to our real concern) for a
  *non-injective* per-thread index map, where Viper's own injectivity
  check on quantified permissions catches the collision unprompted.

**Phase 1 — the ghost-op mechanism, against the real `dialect-mir` crate:**
`dialect-verify-poc/` is a standalone crate path-depending on the real
`dialect-mir` crate (confirmed to need no `rustc_private`/`rustc-dev` --
only `mir-importer` and the codegen backend do, not the IR crate itself).
It defines `VerifyAssertOp` (`verify.assert`) the same way a real
`dialect-verify` would — `#[pliron_op(...)]`, same shape as the real
hand-written `nvvm.assertfail` — builds one real `dialect-mir` function
containing both `verify.assert` and a genuine `mir.store`, runs a
standalone erasure pass, and confirms: before erasure, both ops are
present; after, only `mir.store`/`mir.return` remain, and the function
still verifies. Real output, `cargo run`:
```
=== BEFORE erasure ===
  verify.assert
  mir.store
  mir.return
=== AFTER erasure ===
  mir.store
  mir.return
```

**Deliberate scope cut, named plainly**: this does *not* patch
`mir-importer` itself and run it through rustc. Doing that for real means
rebuilding the actual `rustc-codegen-cuda` backend, which needs the
`rustc-dev` toolchain component (not installed) and, for a full build,
`llc`/`clang` (also not installed). Phase 1 tests the mechanical claim —
ghost op and real op coexist in one `dialect-mir` function, erasure removes
only the ghost one — against the real IR types, without that heavier
toolchain. Patching `mir-importer`'s dispatch table to emit `verify.assert`
from source-level `#[requires(...)]`/`verify_assert!(...)` syntax
automatically is the next step, not yet done.

**Phase 2 — a real, automated `dialect-mir` → Viper translator:**
`dialect-verify-poc/src/translate.rs`. Not hand-written per example — it
walks the real op graph (`Operation::get_opid`, successors, operands) and
applies one fixed rule per op: constants, arithmetic/comparison binops,
one level of shared-memory indirection (`mir.shared_alloc` +
`mir.ptr_offset` + `mir.load`/`mir.store` → `cells[idx].val`), a scalar
local backed by `mir.alloca` (the honest pre-`mem2reg` shape: loads of an
unaliased local are aliased to its own Viper name rather than re-declared,
which also turned out to be load-bearing for loop-invariant soundness, see
Phase 3), structured `if`/`if-without-else`, and a genuine `while` loop
recognized from a real CFG back edge (not a hardcoded shape — found via
reachability over the actual successor graph). `verify.assert` and
`verify.invariant` derive their Viper text from a real SSA boolean value;
only `verify.barrier`'s permission clauses are verbatim Viper text (see
`ghost_ops.rs`'s module doc for why: permissions have no `dialect-mir` SSA
representation to derive them from at all).

`dialect-verify-poc/src/bin/phase2_translate.rs` builds a real `if`/`else`
`dialect-mir` function (shared-memory store on both arms, gated by a real
comparison), translates it with this translator (not by hand), generates
two full `.vpr` files from the identical translated body (one with the
ghost assertion's justification given, one without), and runs both
through Silicon. First real run caught a genuine modeling gap on its own:
`cells: Seq[Ref]` doesn't entail its elements are pairwise distinct, so
`cells[i].val`'s quantified permission wasn't well-formed until an
explicit distinctness precondition was added. After that: the justified
version verifies, the unjustified one is rejected with a diagnostic
pointing at the exact assertion.

**Phase 3 — the real stride-halving block-sum reduction, one generic
thread, a real loop, no unrolling:**
`dialect-verify-poc/src/bin/phase3_reduction.rs` builds the actual
`dialect-mir` CFG for
```
let mut stride = N/2;
while stride >= 1 {
    if tid < stride { TILE[tid] += TILE[tid+stride]; }
    sync_threads();
    stride /= 2;
}
```
(N=8) — a genuine back edge, `stride` as a real `mir.alloca`-backed
mutable local, and the guarded partner access bracketed by two
`verify.barrier` ops (acquire the partner's permission, use it, release
it) standing in for `sync_threads()`'s redistribution. Translated by the
same Phase 2 translator, checked by Silicon. **Passes, with zero
unrolling** — the `while` loop is genuinely inductive, not expanded out N
times.

Getting there took three real, substantive fixes, each found by actually
running it and reading what Silicon said (not anticipated in advance):
1. Silicon's loop treatment havocs every local the body writes to at the
   top of each iteration and only re-assumes what's *textually* in the
   invariant. An invariant that's just a bare variable name referencing a
   separately-computed boolean temp proves nothing, because that temp is
   itself havoc'd with no link back to what it used to mean. Fixed by
   having the translator track, for every constant/arithmetic/comparison
   value, its fully-inlined expression (`expr_text` in `translate.rs`), so
   invariants bottom out in the one genuinely loop-carried variable
   (`stride`) instead of a chain of havoc-able copies.
2. A loop header's own ops (the condition, the invariants) really do
   re-execute every time control reaches the header, including via the
   back edge — translating them once, outside the `while`, silently turns
   the loop condition into a constant. Fixed by re-emitting the header's
   ops a second time at the tail of the body, reassigning the same names
   instead of redeclaring (`redeclare: bool` in `translate_ops`).
3. Permissions need to be carried across the loop boundary explicitly too
   — `acc(cells[tid].val)` isn't a value-level fact, so there's no SSA
   boolean to hang a `verify.invariant` on. Added a sibling ghost op,
   `verify.invariant_perm`, for permission-only invariant clauses (same
   verbatim-text exception as `verify.barrier`, for the same reason).

A deliberately broken variant (`--broken`: one-line off-by-one,
`partner = tid + stride + 1`) is correctly rejected — `cells[8]`, one past
the real tile, caught with a diagnostic at the exact inhale.

Run all of Phase 1-3: `scripts/install_viper.sh`, then from
`dialect-verify-poc/`: `cargo run --bin phase1_erasure`,
`cargo run --bin phase2_translate -- --run`,
`cargo run --bin phase3_reduction -- --run` (and `-- --broken --run`).

**Scope, named plainly, not silently assumed:** Phase 3 verifies ONE
generic thread's permission bookkeeping — that thread `tid` never touches
a cell it lacks justified access to, across every iteration. It does
*not* re-derive the separate argument, flagged all the way back in this
project's original Verus-based plan, that combines `n` such per-thread
proofs via separating conjunction into a whole-block race-freedom
guarantee — nor does it prove the reduction's numeric correctness (the
strided/butterfly sum formula derived earlier in this project, by hand, is
not re-proved here). Both are real further work.

**Phase 4 — LLM-drafted annotations, checked for real, not performed
collaboratively:** to make this an actual test rather than a demo rigged
to pass, a *fresh* agent with no visibility into the derivation above was
asked to independently derive the Phase 3 kernel's four annotation pieces
(the loop invariant, the permission invariant, the barrier's
acquire/release clauses) from the kernel's plain description alone. Its
draft was then plugged into the identical real pipeline (the same
`dialect-mir` construction + Phase 2 translator + Silicon, in
`dialect-verify-poc/src/bin/phase4_llm_annotations.rs`) — not read and
judged by a human, run and checked.

## The real Rust macro front end -- actually built, actually compiled

Everything above (Phases 1-4) builds `dialect-mir` directly via `pliron`'s
Rust API -- standing in for `mir-importer`, which Phase 1 judged out of
reach because patching it needs the `rustc-dev` toolchain component.
That turned out to be wrong: `rustc-dev` was already installed (pinned in
`cuda-rust/cuda-oxide/rust-toolchain.toml`), and the one real blocker --
`cuda-bindings`' bindgen step not finding `stddef.h` -- was a missing
system include path, fixed with `BINDGEN_EXTRA_CLANG_ARGS` (no package
install). So the front end got built for real, inside the `cuda-rust`
submodule (all new/changed files are git-diffable there, nothing
committed):

- **`cuda-oxide/crates/dialect-verify/`**: the ghost ops as a real sibling
  dialect crate (added to the real workspace), registered in
  `mir-importer`'s `register_dialects` alongside `dialect-mir`/
  `dialect-nvvm`. Carries the original string-templated
  `verify.invariant_perm`/`verify.barrier` (still used by the
  hand-built demos above) plus three new, simpler ops added for this
  integration: `verify.perm`/`verify.acquire`/`verify.release`, each
  taking a real pointer SSA value instead of verbatim text -- the
  translator can resolve "which cell" from a real pointer the exact same
  way it already resolves a `mir.load`/`mir.store` address, so there's
  nothing left to author by hand for these three.
- **`cuda_device::verify`** (`crates/cuda-device/src/verify.rs`): real
  stub functions (`assert`, `invariant`, `perm`, `acquire`, `release`),
  `#[inline(never)]` + `unreachable!()`, the identical idiom
  `__gpu_assertfail`/`sync_threads` already use -- plus
  `verify_assert!`/`verify_invariant!`/`verify_perm!`/`verify_acquire!`/
  `verify_release!` macros.
- **`mir-importer` patched for real**: `translator/terminator/intrinsics/verify.rs`
  (new) adds `emit_verify_*`, one per function, each the same "translate
  the arg, build the op, insert it, materialize the `()` result, goto the
  target" shape as `intrinsics::debug::emit_assertfail`. Five new match
  arms in the intrinsic dispatch table route
  `cuda_device::verify::{assert,invariant,perm,acquire,release}` to them.
  `pipeline.rs` calls `dialect_verify::erase_ghost_ops` on every function
  right after that function's own `dialect-mir` verification -- before
  `mem2reg`, loop unrolling, or LLVM export ever see it.

**Tested against the real compiler, not just built:**
`crates/rustc-codegen-cuda/examples/verify_demo/` is a real example crate
with the stride-halving reduction kernel, annotated with the macros:
```rust
let mut stride = 4usize;
while stride >= 1 {
    unsafe {
        verify_invariant!(stride <= 4);
        verify_perm!(core::ptr::addr_of!(TILE[tid]));
    }
    if tid < stride {
        unsafe {
            verify_acquire!(core::ptr::addr_of!(TILE[tid + stride]));
            TILE[tid] = TILE[tid] + TILE[tid + stride];
            verify_release!(core::ptr::addr_of!(TILE[tid + stride]));
        }
    }
    thread::sync_threads();
    stride /= 2;
}
```
`CUDA_OXIDE_DUMP_MIR=1 cargo oxide build verify_demo` runs the real
`rustc` → `mir-importer` → `mem2reg` → loop-unroll → LLVM export → `llc`
pipeline, **successfully, producing real PTX** (`verify_demo.ptx`,
`block_reduce` present). The dump shows exactly what was designed:
- `(pre-verify)`: real `verify.invariant`/`verify.perm`/`verify.acquire`/
  `verify.release` ops, interleaved with the real `mir.alloca`/
  `mir.shared_alloc`/`mir.load`/`mir.store` ops this kernel actually
  produces -- each ghost op's pointer operand is a genuine SSA value
  (`mir.ptr <builtin.fp32, ..., kind:RawConst>`), not a placeholder.
- `(post-ghost-erasure)`: zero `verify.*` ops left (confirmed by grep, not
  just read) -- only the real ops survive, into `mem2reg` and beyond.
- The generated PTX contains `block_reduce`'s real instructions and no
  trace of anything verification-related, confirming the whole point:
  these annotations cost nothing at runtime and touch nothing downstream
  of erasure.

## Closing the loop for real: the real compiled kernel, through the real translator, into Silicon

The gap above is now closed. `dialect-verify::translate` (ported from
`dialect-verify-poc/`'s standalone prototype into the real submodule
crate, alongside the real ops) walks **actual compiler-produced
`dialect-mir`** -- not hand-built IR -- and `mir-importer`'s pipeline
gained a debug hook (`CUDA_OXIDE_VERIFY_EMIT_VPR=<dir>`, panic-isolated
so a translator bug can never break a real build) that runs it on any
function containing a `verify.*` op, right before that function's ghost
ops are erased, and writes the translated body to disk.

Getting `verify_demo`'s real compiled `block_reduce` through this
surfaced real bugs no hand-built example had exercised, each found by
actually running it and reading what Silicon (or the translator itself)
said:
- `mir.ptr_offset` on a non-tile base (e.g. indexing the kernel's
  ordinary `&[f32]` parameter) must not panic -- only a tile base
  resolves to a `cells[...]` index; anything else is just an opaque
  pointer, same as `mir.load`/`mir.store`'s existing fallback.
- Real compiled if/else and loop-body-merge detection can't assume one
  hop: rustc emits one basic block per statement boundary, so an arm's
  own goto target is routinely several blocks short of where it
  actually reconverges. Fixed by chasing the whole chain of
  plain-goto-only blocks, not just the first one.
- A real loop's invariant/permission annotations sit as the first
  statement(s) of the loop **body** (mirroring the source --
  `while cond { verify_invariant!(...); ... }`), not in the header block
  that merely tests `cond`. `translate_block` now returns every
  invariant discovered anywhere in a loop's body subtree, propagated up
  to the enclosing `while` -- not silently empty, which Silicon doesn't
  flag as an error, just proves a weaker (often vacuous) claim.
- A real pointer or comparison result gets routed through its own
  alloca'd stack slot (store once, load repeatedly) just as often as
  kept in one SSA value. Both tile-recognition and inlined expression
  text now propagate through that round-trip, and anything learned from
  the function's skipped prefix (e.g. `stride`'s own `= 4` initializer)
  is explicitly cleared before real translation starts -- a stale
  initial-value fact is exactly the already-fixed-once havoc-correlation
  bug, reappearing through a new path.
- A real compiled local can be `bool` (a width-1 integer) as easily as a
  real `Int`; declaring every local `Int` regardless is a genuine Viper
  type error the moment one is used in boolean position.

With all of that fixed: `verify_demo`'s compiled `block_reduce`,
wrapped with its real signature (`viper-poc/real_compiler_block_reduce.vpr`
-- the tile/thread-index parameters are *bridged* in by hand, matching
the established "the caller states the method signature" scope; the
body itself is 100% translator output, untouched), **verifies with
Silicon**. `verify_demo_broken` (one deliberate off-by-one --
`TILE[tid + stride + 1]`) is correctly **rejected**, with a diagnostic
naming the exact out-of-bounds index
(`viper-poc/real_compiler_block_reduce_broken.vpr`).

This is the real, complete loop: Rust source with `verify_*!` macros →
real `rustc` → real `mir-importer` dispatch → real ghost ops → real
translator → real Viper → real Silicon/Z3 → a genuine pass, and a
genuine, correctly-diagnosed rejection.

**What's still not covered**: `mir.extract_field` (hit twice, both on
the kernel's ordinary `&[f32]` parameter, structurally irrelevant to the
tile reasoning) falls through the translator's generic fallback --
sound (an unconstrained value can only make an unrelated proof harder,
never silently paper over one), but a real pointer field extracted this
way would need its own rule if it were ever actually dereferenced in
something this track verifies. The signature-bridging step (naming
which real local is `tid`, which is the tile) is also still manual, not
derived from the function's own debug/provenance attributes -- a
mechanical next step, not a conceptual gap.

The fresh agent, given only the plain kernel description and the ghost-op
mechanism's rules (no access to the derivation above), derived:
```
LOOP_INVARIANT: stride >= 0 && stride <= 4
PERM_INVARIANT: acc(cells[tid].val)
BARRIER_ACQUIRE_EXHALE: true       BARRIER_ACQUIRE_INHALE: acc(cells[tid+stride].val)
BARRIER_RELEASE_EXHALE: acc(cells[tid+stride].val)   BARRIER_RELEASE_INHALE: true
```
— identical, clause for clause, to what Phase 3 needed. Plugged into the
real pipeline unedited: **Silicon reports verification successful.** The
interesting result isn't "an LLM can do algebra" (this is a well-known
textbook pattern; convergence on the same answer isn't surprising) — it's
that the convergence didn't have to be *taken on faith*. The draft went
through the identical mechanical translator and the identical real SMT
check as every other result in this document; a wrong draft would have
been caught the same way the off-by-one variant was in Phase 3, not
rubber-stamped because it "looked right."

## Phase 5: the two gaps named explicitly above, closed

Everything above proves ONE generic thread's permission bookkeeping is
self-consistent. Two honestly-flagged gaps remained: (1) combining N such
per-thread proofs into a whole-block guarantee, and (2) that the reduction
computes the right *value*, not just that it never touches memory it
shouldn't. Both closed, in `viper-poc/phase5_*.vpr`.

### Part 1 — the N-thread combination argument

Concurrent Separation Logic's parallel-composition rule says N per-thread
Hoare triples combine into "safe to run together" provided their resource
claims are pairwise disjoint. That rule itself is standard, cited
background theory here, not re-derived -- what was missing was checking
its one real hypothesis for *this* kernel: that the barrier's
redistribution (every thread keeps its own cell; active threads
additionally, temporarily, read their partner's) never double-grants a
cell to two different claims.

`phase5_nthread_injectivity.vpr` states exactly that, as a single
quantified-permission precondition with `stride` left symbolic (bounded
`1 <= stride <= 4`, matching `real_compiler_block_reduce.vpr`'s own
`invariant v9 <= 4` exactly, so it's checked for every round the real
loop ever examines, not three separate hardcoded cases):
```viper
requires forall t: Int :: 0 <= t && t < stride ==>
           acc(cells[t].val) && acc(cells[t + stride].val)
```
Viper's own well-formedness check on a quantified permission requires its
receiver set to be injective -- this is the exact mechanism Phase 0's
`bad_non_injective_redistribution` test already demonstrated catching a
real violation. **Verification successful**: the real redistribution is
injective, generically. A deliberately wrong one-off variant
(`t + 1` instead of `t + stride`, the same shape of bug as
`verify_demo_broken`) is **correctly rejected**:
`Quantified resource cells[t].val might not be injective` — Silicon finds
the actual collision (at `stride=2`, thread 0's claimed "partner" and
thread 1's own cell coincide).

### Part 2 — numeric correctness

A different kind of claim: thread 0 ends up holding the sum of all 8
original elements. Viper has no built-in summation, so this needed real
new machinery, used nowhere else in this project:

- `strided_sum(o, i, s, c)`: a recursive Viper function defining "the sum
  of `c` elements of `o`, starting at `i`, spaced `s` apart".
- `merge_lemma`: the one genuine piece of new math this needs -- merging
  two interleaved width-`s` strided sums into one double-width strided
  sum, proved by induction on the count (the two sums' terms interleave
  into exactly the merged sum's terms, just reordered; addition doesn't
  care about order).
- The loop invariant, parameterized by the current `stride` exactly like
  the permission proofs: letting `width = (stride == 0 ? 1 : 2*stride)`,
  `cells[tid].val == strided_sum(orig, tid, width, 8/width)` whenever
  `tid < width`. The `stride == 0` case is the loop's *exit* state (after
  the last round): width collapses to 1, giving `cells[0] == sum of all
  8 original elements` as the postcondition -- the same one formula
  covers every checkpoint, including the last, with no separately
  special-cased exit logic.

Getting this to verify took real, substantive debugging, all against
genuine tool behavior:
- **A JVM `StackOverflowError`** (not a Viper-level error) from Silicon
  itself partway through -- a known rough edge with recursive-function
  proofs this deep. Fixed by raising the JVM thread stack
  (`java -Xss256m ...`); `scripts/run_viper_poc.sh` applies this only to
  the two files that need it.
- **An off-by-one in `strided_sum`'s own precondition**: a sum of `c`
  terms starting at `i` accesses up to `o[i + (c-1)*s]`, not
  `o[i + c*s]` -- the exact kind of arithmetic slip the whole point of
  mechanical checking catches.
- **Z3 not auto-chaining several levels of a recursive definition.**
  Standard for this kind of proof, not a sign of a wrong lemma: both
  `merge_lemma`'s own inductive step and the main invariant's
  maintenance needed explicit `assert`s unfolding `strided_sum` one level
  at a time, each immediately dischargeable from the function's own
  defining equation, chained together with the recursive call's
  postcondition.
- **A real missing hypothesis, not a tooling quirk**: `1 <= stride <= 4`
  alone doesn't imply `stride` is a power of two (`stride = 3` satisfies
  it, and `3/2` truncates to `1`, breaking the halving arithmetic the
  proof depends on). Had to add `stride == 4 || stride == 2 || stride ==
  1 || stride == 0` explicitly -- a fact that's true of the real loop but
  wasn't implied by what had been stated so far.

With all of that: **Verification successful.** A deliberately broken
variant (`own - partner` instead of `own + partner` -- same cells, same
indices, same permissions, same injectivity; only the arithmetic is
wrong) is **correctly rejected**, specifically by the numeric machinery
(`invariant.not.preserved`), confirming this isn't vacuous: something
that passes every permission-level check can still be caught here.

**What this still doesn't close**: the value-level analogue of the
acquire step -- "thread `tid`'s partner holds `strided_sum(...)` at this
exact point" -- is *inhaled* (assumed), justified by the same N-thread
symmetry argument as part 1's permission acquire, not mechanically linked
to part 1's injectivity check or to another thread's own run of this same
proof in one unified artifact. That linkage is exactly the same kind of
"cited, not mechanized" step part 1 already named for CSL itself --
consistent with this track's running theme, not a new, hidden gap.

## A second real kernel: tiled matmul, and two translator bugs it exposed

Everything above verifies one kernel (the stride-halving reduction). To
show the real-compiler-closure architecture itself generalizes -- not
just that it was tuned to fit one example -- a second, genuinely
different real kernel was built the same way: `verify_tiled_matmul`
(`cuda-rust/cuda-oxide/crates/rustc-codegen-cuda/examples/verify_tiled_matmul/`),
a naive 4x4 tiled matrix multiply. 16 threads, two *separate* shared
tiles (`TILE_A`, `TILE_B` -- the whole point of tiling: load each operand
cooperatively once, reuse it for every thread's dot product), a single
barrier, then a dot-product accumulation loop where each iteration
borrows a *different* cell of each tile (not a fixed partner, as the
reduction's acquire/release always used) via the same
`verify_acquire!`/`verify_release!` pattern parameterized by the loop
variable `k`. `scripts/run_matmul_poc.sh` builds it (and a deliberately
broken sibling, `verify_tiled_matmul_broken`, with an off-by-one into the
B tile) end to end, the same way `run_real_compiler_poc.sh` does for the
reduction; see `viper-poc/real_compiler_tiled_matmul.vpr` / `_broken.vpr`
for what the real translator actually produced, wrapped.

**The translator change this needed**: every op in `translate.rs` that
renders a shared-memory access hardcoded the literal name `cells` -- fine
when there's exactly one tile, wrong the moment there are two (`TILE_B`'s
cells would silently alias into the same Viper sequence as `TILE_A`'s).
Fixed by naming tiles by order of first appearance (`tile_values` is now
`HashMap<Value, String>`, not `HashSet<Value>`; `tile_index` carries
`(tile_name, idx)` pairs, not just `idx`) -- the first tile keeps the name
every existing single-tile demo already assumes (`cells`), so nothing
built before this needed to change; a second tile gets `cells2`, a third
`cells3`, and so on. Confirmed via the full regression suite
(`run_viper_poc.sh`, `run_real_compiler_poc.sh`) before and after: zero
behavior change for every existing single-tile kernel.

**Two real bugs this new kernel's shape exposed in the translator
itself** (not kernel-specific workarounds -- both are in `translate.rs`,
fixed once, benefiting every kernel):

1. **The `while` condition was a bare, havocked variable, unrelated to the
   loop variable.** The reduction's own bound check (`tid + stride < 8`)
   never actually depended on the loop's *condition* being known inside
   the body -- only on `stride`'s own invariant. The matmul kernel's
   bound check (`row * 4 + k < 16`) genuinely needs `k < 4` (the loop
   condition itself, strict), not just the invariant's `k <= 4` (which
   permits `k == 4`, one past the end). Silicon only assumes the loop's
   *declared invariants* when checking the body, not whatever opaque
   boolean variable happens to sit in the `while (...)` header text --
   and that variable is itself havocked at the loop boundary like any
   other body-written local, carrying no relationship back to `k` unless
   the condition text *says so directly*. Fixed by inlining the condition
   (via the same `atom()` mechanism already used for invariants) instead
   of using the bare SSA name -- found and fixed together with a related,
   previously-latent bug: `mir.not`'s and the arithmetic/comparison ops'
   own inlined text wasn't parenthesized (`!a_atom` renders wrong when
   `a_atom` is itself `a < b`), invisible until something actually nested
   an inlined condition inside a `!`.

2. **Cleaning up `dialect-verify-poc` surfaced two more, deeper bugs.**
   That crate had its own, separately-maintained copy of `ghost_ops.rs`/
   `translate.rs` (predating the real `dialect-verify` crate inside the
   `cuda-rust` submodule, back when the real crate didn't exist yet) --
   frozen at an earlier, less-debugged state, silently diverging from
   every fix made to the real crate since. Deduplicated by pointing
   `dialect-verify-poc` at the real crate instead (one `Cargo.toml` path
   dependency, four one-line import changes, two now-redundant files
   deleted) -- and running the existing Phase 0-5 regression suite
   against the *real* translator for the first time immediately exposed:
   - **A loop variable initialized to a literal constant right before the
     loop got that constant baked into the `while` condition and every
     invariant, verbatim** -- `while (4 >= 1)` and `invariant (4 <= 4)`
     instead of `while (stride >= 1)` / `invariant (stride <= 4)`. Not
     just imprecise: a tautology. `!(4 >= 1)` is a contradiction, making
     the post-loop continuation's assumed state unreachable and any
     `ensures` clause trivially "proven" regardless of its content --
     Phase 3's own correct run still said "Verification successful" with
     this bug present, for the wrong reason. It never showed up on
     `verify_demo`/`verify_tiled_matmul` because those go through
     `translate_function_body_from_first_verify_op`, which already clears
     this exact tracking (`alloca_expr_text`) for an unrelated reason (a
     real compiled kernel's discarded prefix). Hand-built IR calling the
     plain `translate_function_body` entry point has no such prefix to
     discard, so nothing ever cleared it. Fixed by clearing
     `alloca_expr_text`/`expr_text` at every loop header's first visit
     (`is_loop_header`, already existed for a different purpose), not
     just once at the start of the whole function.
   - **`translate_function_body` (the plain entry point) never prepended
     `alloca_prelude` at all** -- every `mir.alloca`'d local's own `var`
     declaration was silently missing from the output the moment the
     kernel had one, a parse-level error, not a verification failure.
     Present since the day `alloca_prelude` was introduced (for the
     *other* entry point's needs) but never caught, because nothing had
     called the plain entry point against the real crate until this
     cleanup. One-line fix: prepend it, matching the sibling entry
     point's own behavior.

   Both bugs are fixed now and confirmed against the full regression
   suite (every phase, every deliberately-broken variant, both real
   compiler closures) -- but they're a concrete illustration of exactly
   why the duplication was worth removing: two copies of the same logic,
   maintained separately, drift, and the drift is invisible until
   something forces both copies to run against the same input.

## Matmul numeric correctness

Same extension the reduction kernel got in Phase 5 part 2, for the
matmul kernel: not just that it touches memory safely, but that thread
`tid` (at `row = tid/4`, `col = tid%4`) actually ends up holding the real
dot product `sum_{j=0}^{3} A[row][j] * B[j][col]`.
`viper-poc/matmul_numeric_correctness.vpr` (+ `_broken.vpr`), same
architecture as `phase5_numeric_correctness.vpr`: a hand-written Viper
method mirroring the real kernel's structure (not derived from the real
translator's output -- the value-level `inhale` at each acquire has no
counterpart in `dialect-mir`, so this is authored the same way the
barrier clauses are), checked against a deliberately-broken sibling.

**Markedly simpler than the reduction's proof.** The reduction's loop
*combines* two halves every round (`cells[tid] := cells[tid] +
cells[tid+stride]`, repeatedly over `log2(8) = 3` rounds, each round
touching a different, shrinking set of active threads) -- proving that
needed a real inductive lemma (`merge_lemma`: two interleaved width-`s`
strided sums merge into one double-width strided sum) because the
*shape* of what's being summed changes every round. The matmul kernel's
loop just *accumulates* one more term each of 4 iterations, to one fixed
set of cells, with no interleaving to reconcile -- so the recursive
function alone (`dot_sum(oa, ob, row, col, k) = sum_{j<k} oa[row*4+j] *
ob[j*4+col]`) plus one explicit one-step unfolding `assert` per iteration
(the same reason `merge_lemma`'s own body needed explicit unfolding
asserts: Z3 won't chain several levels of a recursive definition on its
own) is enough. No merge lemma needed at all.

**The value-level assumption, spelled out explicitly**: each
`verify_acquire!` in the real kernel only ever grants *permission* --
nothing in `dialect-mir` can state "and this cell still holds its
original value" structurally, the same gap the reduction's acquire step
had. Here the justification is actually more direct than the reduction's
own (which argued its acquire via a *recursive, same-formula-at-every-
round* symmetry): between the load phase and this kernel's own final
write-back, **nothing ever writes to any cell of either tile at all** --
the loop only ever reads. So "cell `i` of tile A still holds `oa[i]`" is
no more than the load phase's own claim, unneeded to re-derive from a
by-round formula; it's assumed to hold for every `i` throughout, which is
what the `inhale cells[...] == oa[...]` lines at each acquire state
directly.

**What this still doesn't close**: same caveat as the reduction's, for
the same reason -- this is one generic thread's own proof, not
mechanically linked to another thread's run of the identical argument,
or to the permission-level injectivity check. And as with every other
claim in this track, it's fixed at `N=16` threads / 4x4 matrices;
generalizing the shape (not just the size) is unexamined.

## Still open: per-thread function + lifting rule, not a whole-block loop

A real SIMT block runs `n` threads *concurrently*, each executing the same
body. `shared_test_verified`'s two `while` loops instead model this as `n`
*sequential* iterations of one body — a different claim that happens to
give the same answer here only because every write in a given phase
targets a disjoint cell, so order never mattered for *this* kernel. The
more faithful structure, named explicitly as the target a few turns into
the design process and not yet built: a genuine per-thread function (body
close to one iteration of today's loop, no loop inside it) plus a separate,
reusable "lifting" axiom in `stage1_gpu_semantics.rs` stating that the
per-thread contract holding for an arbitrary `tid` implies the whole-block
property — matching Kuiper's own `kf(i,j)` + launch-rule structure. This
pass's scope was the `verus_keep_ghost` dual-compilation mechanism and the
real `cuda-device` integration; the loop-vs-lifted-per-thread restructuring
is still pending, not forgotten.

## What this does *not* cover

Consistent with the brief ("I don't care about advanced features like
async copies"): no warp-level collectives (`shfl_sync`/`ballot_sync`), no
TMA/tensor-core collective-call constraints, no multi-block/cluster
synchronization, no divergent-barrier reachability checking. All of these
were named in the design discussion as either genuinely open research
problems (shared with Kuiper, the closest academic prior art) or
straightforward-but-unbuilt extensions of the same located-resources
pattern used here. This prototype's job was narrower: show the core
technique actually runs, end to end, against one real `unsafe` cuda-oxide
kernel — not to cover the whole safety model.
