#!/usr/bin/env bash
# Installs Viper's Silicon verifier. Unlike Verus, this is pure JVM
# bytecode -- no architecture-specific build needed -- so the prebuilt
# "ViperTools" bundle (published via the viper-ide project's releases,
# since `silicon` itself ships no GitHub releases) works as-is on this
# aarch64 box. The bundled z3 binary is x86-64 and won't run here; this
# script does not fetch a replacement -- it expects the aarch64 Z3 already
# built for Verus (scripts/install_verus.sh) to exist, and points Silicon
# at it via --z3Exe at run time (see run_viper_poc.sh).
set -euo pipefail

VIPER_PREFIX="${VIPER_PREFIX:-$HOME/.local/opt/viper-tools}"
mkdir -p "$VIPER_PREFIX"

if [[ -f "$VIPER_PREFIX/backends/viperserver.jar" ]]; then
  echo "==> Viper tools already present at $VIPER_PREFIX"
  exit 0
fi

echo "==> Finding latest ViperTools Linux bundle..."
url=$(curl -s https://api.github.com/repos/viperproject/viper-ide/releases/latest \
  | grep browser_download_url | grep ViperToolsLinux.zip | head -1 | cut -d '"' -f4)
if [[ -z "$url" ]]; then
  echo "ERROR: could not find a ViperToolsLinux.zip asset on the latest viper-ide release." >&2
  exit 1
fi

echo "==> Downloading $url"
tmp=$(mktemp -d)
curl -sL -o "$tmp/ViperToolsLinux.zip" "$url"
unzip -q "$tmp/ViperToolsLinux.zip" -d "$tmp/extracted"
rm -rf "$VIPER_PREFIX"
mv "$tmp/extracted" "$VIPER_PREFIX"
rm -rf "$tmp"

echo "==> Installed: $VIPER_PREFIX/backends/viperserver.jar"
# SiliconRunner exits 1 on --help (no file argument given) even when it
# printed the help text fine -- check output content, not exit code.
if java -cp "$VIPER_PREFIX/backends/viperserver.jar" viper.silicon.SiliconRunner --help 2>&1 \
     | grep -q "^Usage: Silicon"; then
  echo "==> SiliconRunner invokes successfully."
else
  echo "ERROR: SiliconRunner did not produce expected output." >&2
  exit 1
fi
