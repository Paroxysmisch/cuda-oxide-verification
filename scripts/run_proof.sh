#!/usr/bin/env bash
# Single entry point for the whole prototype: runs Verus against the
# verified kernel, builds the real kernel against the real cuda-device
# crate, runs the independent brute-force sanity check, and prints one
# combined PASS/FAIL summary.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"

echo "=================================================================="
echo " 1/3  Verus proof: verus-proof/lib.rs (stage1_gpu_semantics +"
echo "      tile_macros + kernel_dual's verified branch)"
echo "=================================================================="
if ! command -v verus >/dev/null 2>&1; then
  echo "verus not found on PATH -- run scripts/install_verus.sh first." >&2
  exit 1
fi

pushd "$ROOT/verus-proof" >/dev/null
verus lib.rs --crate-type=lib
verus_status=$?
popd >/dev/null

echo
echo "=================================================================="
echo " 2/3  Real kernel: kernel/ builds against the real cuda-device"
echo "      crate from the cuda-rust submodule (the verus_keep_ghost"
echo "      branch of the same file Verus just checked)"
echo "=================================================================="
pushd "$ROOT/kernel" >/dev/null
cargo build --quiet
kernel_status=$?
popd >/dev/null

echo
echo "=================================================================="
echo " 3/3  Sanity check (plain Rust, no Verus): verify-proto"
echo "=================================================================="
pushd "$ROOT/verify-proto" >/dev/null
cargo run --quiet --bin sanity_check
sanity_status=$?
popd >/dev/null

echo
echo "=================================================================="
if [[ $verus_status -eq 0 && $kernel_status -eq 0 && $sanity_status -eq 0 ]]; then
  echo "RESULT: PASS -- Verus proved race-freedom + functional correctness"
  echo "        for shared_test_verified; the real shared_test kernel (same"
  echo "        file, cfg-selected) builds against the real cuda-device"
  echo "        crate; and the independent brute-force check agrees with"
  echo "        the same claim for N in {1,2,4,8,...,1024}."
  exit 0
else
  echo "RESULT: FAIL -- verus_status=$verus_status kernel_status=$kernel_status sanity_status=$sanity_status"
  exit 1
fi
