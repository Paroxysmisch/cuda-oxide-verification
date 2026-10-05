#!/usr/bin/env bash
# Builds verify_demo / verify_demo_broken through the REAL cuda-oxide
# compiler (rustc -> mir-importer -> mem2reg -> LLVM export -> llc),
# extracts the real translator's output via the CUDA_OXIDE_VERIFY_EMIT_VPR
# debug hook, wraps it with its (hand-bridged) method signature, and runs
# both through Silicon. See NOTES.md's "Closing the loop for real" section.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_OXIDE="$ROOT/cuda-rust/cuda-oxide"
export BINDGEN_EXTRA_CLANG_ARGS="${BINDGEN_EXTRA_CLANG_ARGS:--isystem /usr/lib/gcc/aarch64-linux-gnu/13/include}"
Z3="$HOME/.local/opt/verus-src/source/target-verus/release/z3"
JAR="$HOME/.local/opt/viper-tools/backends/viperserver.jar"
CARGO_OXIDE="$ROOT/cuda-rust/target/debug/cargo-oxide"

if [[ ! -x "$CARGO_OXIDE" ]]; then
  echo "Building cargo-oxide (first run only)..."
  (cd "$CUDA_OXIDE" && cargo build -p cargo-oxide --quiet)
fi

wrap_and_check() {
  local example="$1" method_name="$2" out_vpr="$3"
  local vpr_out
  vpr_out="$(mktemp -d)"

  echo "=== building $example through the real pipeline ==="
  (cd "$CUDA_OXIDE" && CUDA_OXIDE_VERIFY_EMIT_VPR="$vpr_out" "$CARGO_OXIDE" build "$example") \
    2>&1 | grep -E "CUDA_OXIDE_VERIFY_EMIT_VPR|no rule for|error" || true

  local body="$vpr_out/block_reduce.vpr.body"
  if [[ ! -f "$body" ]]; then
    echo "ERROR: $body was not produced -- check the build output above." >&2
    return 1
  fi

  python3 - "$body" "$method_name" "$out_vpr" <<'PYEOF'
import sys
body_path, method_name, out_path = sys.argv[1:4]
with open(body_path) as f:
    lines = f.read().split("\n")
prelude_end = 0
for i, l in enumerate(lines):
    if l.startswith("  var v"):
        prelude_end = i + 1
    else:
        break
prelude = "\n".join(lines[:prelude_end]) + "\n"
rest = "\n".join(lines[prelude_end:])
header = f"""field val: Int

method {method_name}(cells: Seq[Ref], tid_param: Int)
  requires |cells| == 8
  requires forall i: Int, j: Int :: 0 <= i && i < 8 && 0 <= j && j < 8 && i != j ==> cells[i] != cells[j]
  requires 0 <= tid_param && tid_param < 8
  requires acc(cells[tid_param].val)
  ensures acc(cells[tid_param].val)
{{
"""
bridge = ("  v1 := tid_param  // tid's alloca slot, bridged from the method parameter\n"
          "  v9 := 4  // stride's alloca slot: `let mut stride = 4usize;` in the skipped prefix\n")
with open(out_path, "w") as f:
    f.write(header)
    f.write(prelude)
    f.write(bridge)
    f.write(rest)
    f.write("}\n")
PYEOF

  echo "=== running Silicon on $out_vpr ==="
  java -cp "$JAR" viper.silicon.SiliconRunner --z3Exe "$Z3" "$out_vpr" 2>/dev/null | tail -5
  rm -rf "$vpr_out"
}

wrap_and_check verify_demo real_compiler_block_reduce \
  "$ROOT/viper-poc/real_compiler_block_reduce.vpr"
echo
wrap_and_check verify_demo_broken real_compiler_block_reduce_broken \
  "$ROOT/viper-poc/real_compiler_block_reduce_broken.vpr"

echo
echo "Done. The first run above should say 'Verification successful'; the"
echo "second (verify_demo_broken, a deliberate off-by-one) should report"
echo "exactly one error naming the out-of-bounds index."
