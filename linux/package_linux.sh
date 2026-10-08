#!/usr/bin/env bash
# package_linux.sh - portable Linux release package:
#   release/NoidMiner-v<ver>-linux-x64-cuda12.tar.gz (+ .sha256.txt)
# Generic x86-64 code, CUDA runtime / OpenSSL / C++ runtime linked statically.
set -euo pipefail
cd "$(dirname "$0")/.."
VER="$(sed -n 's/^#define NOIDMINER_VERSION "\(.*\)"/\1/p' src/main.cpp)"
NAME="NoidMiner-v$VER-linux-x64-cuda12"
NOID_HOST_ARCH="-march=x86-64 -mtune=generic" ./linux/build_linux.sh
OUT="release/$NAME"
rm -rf "$OUT" "release/$NAME.tar.gz"
mkdir -p "$OUT"
cp build/noidminer build/vectors.txt "$OUT/"
cp linux/dist/noidminer.conf linux/dist/README.txt linux/dist/start.sh linux/dist/test.sh linux/dist/bench.sh "$OUT/"
cp LICENSE "$OUT/LICENSE.txt"; cp NOTICE "$OUT/NOTICE.txt"
chmod 755 "$OUT/noidminer" "$OUT"/*.sh
# CPU part of the self-test (the GPU part needs a GPU, its exit code is ignored here)
TLOG="$("$OUT/noidminer" --test 2>&1 || true)"
echo "$TLOG" | grep -q "TEST CPU reference: 64/64 official vectors OK" || { echo "$TLOG"; echo "CPU self-test FAILED"; exit 1; }
rm -f "$OUT/noidminer.log"
tar -C release --owner=0 --group=0 -czf "release/$NAME.tar.gz" "$NAME"
(cd release && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256.txt" && cat "$NAME.tar.gz.sha256.txt")
