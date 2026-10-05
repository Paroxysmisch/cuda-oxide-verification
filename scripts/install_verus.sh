#!/usr/bin/env bash
# Installs the `verus` binary, preferring a prebuilt release when one matches
# this OS/arch, otherwise building from source (verus-lang/verus's own
# documented process, read fresh from BUILD.md rather than hardcoded).
set -euo pipefail

VERUS_PREFIX="${VERUS_PREFIX:-$HOME/.local/opt}"
SRC_DIR="$VERUS_PREFIX/verus-src"
PREBUILT_DIR="$VERUS_PREFIX/verus"
LINK_DIR="$HOME/.local/bin"

os="$(uname -s)"
arch="$(uname -m)"
echo "==> Detected platform: $os / $arch"

asset_pattern=""
case "$os-$arch" in
  Linux-x86_64)  asset_pattern="x86-linux" ;;
  Darwin-arm64)  asset_pattern="arm64-macos" ;;
  Darwin-x86_64) asset_pattern="x86-macos" ;;
  *) asset_pattern="" ;;
esac

download_url=""
if [[ -n "$asset_pattern" ]]; then
  echo "==> Checking for a prebuilt release ($asset_pattern)..."
  download_url=$(curl -s https://api.github.com/repos/verus-lang/verus/releases/latest \
    | grep browser_download_url | grep "$asset_pattern" | head -1 | cut -d '"' -f4 || true)
fi

mkdir -p "$VERUS_PREFIX" "$LINK_DIR"

if [[ -n "$download_url" ]]; then
  echo "==> Found prebuilt release: $download_url"
  tmp=$(mktemp -d)
  curl -sL "$download_url" -o "$tmp/verus.zip"
  unzip -q "$tmp/verus.zip" -d "$tmp"
  extracted_dir=$(find "$tmp" -maxdepth 1 -type d -name 'verus*' | head -1)
  rm -rf "$PREBUILT_DIR"
  mv "$extracted_dir" "$PREBUILT_DIR"
  rm -rf "$tmp"
  VERUS_BIN="$PREBUILT_DIR/verus"
else
  echo "==> No prebuilt release for $os/$arch -- building from source."
  if [[ ! -d "$SRC_DIR" ]]; then
    git clone --depth 1 https://github.com/verus-lang/verus.git "$SRC_DIR"
  fi

  cd "$SRC_DIR/source"
  echo "==> Fetching Z3..."
  ./tools/get-z3.sh

  toolchain=$(grep -m1 '^channel' "$SRC_DIR/rust-toolchain.toml" | sed -E 's/channel *= *"(.*)"/\1/')
  if [[ -n "$toolchain" ]]; then
    echo "==> Ensuring rust toolchain '$toolchain' is installed..."
    rustup toolchain install "$toolchain" || true
  fi

  echo "==> Activating vargo dev environment..."
  # shellcheck disable=SC1091
  source ../tools/activate

  echo "==> Building verus with vargo (this builds + verifies vstd; can take a while)..."
  vargo build --release

  VERUS_BIN="$SRC_DIR/source/target-verus/release/verus"
fi

if [[ ! -x "$VERUS_BIN" ]]; then
  echo "ERROR: expected verus binary at $VERUS_BIN but it's missing/not executable." >&2
  exit 1
fi

ln -sf "$VERUS_BIN" "$LINK_DIR/verus"
echo "==> Installed: $LINK_DIR/verus -> $VERUS_BIN"
echo "==> Make sure $LINK_DIR is on your PATH."
"$VERUS_BIN" --version || true
