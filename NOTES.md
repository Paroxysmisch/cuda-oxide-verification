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

**What this does not yet close**: feeding this *exact* real dump through
the Phase 2 translator + Silicon. Real `rustc` MIR is far more verbose
than the hand-built IR Phases 1-4 use -- many more blocks (one rustc
local per sub-expression), `mir.ref`, multiple pointer-kind casts
(`RawConst`/`RawMut`/`UniqueRef`/`SharedRef`) the translator doesn't
handle yet. Extending the translator to that real op surface is further
work, not attempted in this pass -- correctly scoping that extension
matters more than rushing it.

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
