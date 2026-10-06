#!/usr/bin/env bash
# Entry point for the Phase 0-4 Viper/dialect-mir POC. Runs, in order:
#   0. Hand-written permission sanity checks (viper-poc/phase0_test.vpr)
#   1. The ghost-op-coexists-then-erases mechanism (dialect-verify-poc/,
#      built against the real dialect-mir crate)
#   2. The real, automated dialect-mir -> Viper translator, on a real
#      if/else kernel, pass+fail
#   3. The real stride-halving reduction kernel (a genuine loop + barrier,
#      no unrolling), correct + a deliberately broken off-by-one variant
#   4. The same kernel, annotated by a fresh agent with no visibility into
#      Phase 3's derivation, run through the identical real pipeline
# See NOTES.md for what each phase does and does not establish.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIPER_PREFIX="${VIPER_PREFIX:-$HOME/.local/opt/viper-tools}"
Z3="$HOME/.local/opt/verus-src/source/target-verus/release/z3"
SILICON_CP="$VIPER_PREFIX/backends/viperserver.jar"

if [[ ! -f "$SILICON_CP" ]]; then
  echo "Viper tools not found -- run scripts/install_viper.sh first." >&2
  exit 1
fi
if [[ ! -x "$Z3" ]]; then
  echo "aarch64 Z3 not found at $Z3 -- run scripts/install_verus.sh first." >&2
  exit 1
fi

run_silicon() {
  java -cp "$SILICON_CP" viper.silicon.SiliconRunner --z3Exe "$Z3" "$1" 2>/dev/null | tail -4
}

# phase5_numeric_correctness*.vpr's recursive strided_sum/merge_lemma
# proof needs a larger JVM thread stack than Silicon's default -- without
# it, Silicon itself crashes with a StackOverflowError (not a Viper-level
# error) partway through. -Xss is harmless for every other file here.
run_silicon_deep() {
  java -Xss256m -cp "$SILICON_CP" viper.silicon.SiliconRunner --z3Exe "$Z3" "$1" 2>/dev/null | tail -4
}

echo "=================================================================="
echo " 1/3  Phase 0: hand-written Viper permission sanity checks"
echo "=================================================================="
run_silicon "$ROOT/viper-poc/phase0_test.vpr"
phase0_status=$?

echo
echo "=================================================================="
echo " 1/5  Phase 1: ghost-op mechanism against the real dialect-mir crate"
echo "=================================================================="
pushd "$ROOT/dialect-verify-poc" >/dev/null
cargo run --quiet --bin phase1_erasure

echo
echo "=================================================================="
echo " 2/5  Phase 2: a real, automated translator (if/else kernel)"
echo "=================================================================="
cargo run --quiet --bin phase2_translate -- --run

echo
echo "=================================================================="
echo " 3/5  Phase 3: real stride-halving reduction, real loop, no unrolling"
echo "=================================================================="
cargo run --quiet --bin phase3_reduction -- --run

echo
echo "=================================================================="
echo " 4/5  Phase 3 (broken): deliberately injected off-by-one, should fail"
echo "=================================================================="
cargo run --quiet --bin phase3_reduction -- --broken --run

echo
echo "=================================================================="
echo " 5/5  Phase 4: fresh-agent-drafted annotations, same real pipeline"
echo "=================================================================="
cargo run --quiet --bin phase4_llm_annotations -- --run
popd >/dev/null

echo
echo "=================================================================="
echo " Phase 5, part 1: N-thread combination argument (injectivity)"
echo "=================================================================="
run_silicon "$ROOT/viper-poc/phase5_nthread_injectivity.vpr"

echo
echo "=================================================================="
echo " Phase 5, part 2: numeric correctness (correct + broken)"
echo "=================================================================="
echo "--- correct (own + partner): expect Verification successful ---"
run_silicon_deep "$ROOT/viper-poc/phase5_numeric_correctness.vpr"
echo "--- broken (own - partner): expect invariant.not.preserved ---"
run_silicon_deep "$ROOT/viper-poc/phase5_numeric_correctness_broken.vpr"

echo
echo "=================================================================="
echo " Matmul numeric correctness (correct + broken)"
echo "=================================================================="
echo "--- correct (own_a * own_b accumulated): expect Verification successful ---"
run_silicon_deep "$ROOT/viper-poc/matmul_numeric_correctness.vpr"
echo "--- broken (subtracted instead of added): expect invariant.not.preserved ---"
run_silicon_deep "$ROOT/viper-poc/matmul_numeric_correctness_broken.vpr"

echo
echo "=================================================================="
echo " Matmul numeric correctness, arbitrary N (correct + broken)"
echo "=================================================================="
echo "--- correct, symbolic N: expect Verification successful ---"
run_silicon_deep "$ROOT/viper-poc/matmul_numeric_correctness_arbitrary_n.vpr"
echo "--- broken, symbolic N: expect invariant.not.preserved ---"
run_silicon_deep "$ROOT/viper-poc/matmul_numeric_correctness_arbitrary_n_broken.vpr"

echo
echo "=================================================================="
echo "Phase 0-5.2 POC complete. Phase 0's test file has one method"
echo "(bad_non_injective_redistribution) that is SUPPOSED to fail; Phase 3's"
echo "--broken run is ALSO supposed to fail; phase5_nthread_injectivity.vpr's"
echo "second method, phase5_numeric_correctness_broken.vpr,"
echo "matmul_numeric_correctness_broken.vpr, and"
echo "matmul_numeric_correctness_arbitrary_n_broken.vpr are ALSO supposed"
echo "to fail -- read NOTES.md, don't just check exit codes."
