#!/usr/bin/env bash
# build_host.sh - builds <dir>/noidminer.exe (MSYS2 MINGW64 g++ + OpenSSL),
# linked directly against <dir>/noidgpu.dll.   Log: logs/build_host.log
# NOID_HOST_ARCH overrides -march=native (release: "-march=x86-64 -mtune=generic").
set -u
cd "$(dirname "$0")/.."
OUT="${1:-dev}"
mkdir -p logs
LOG=logs/build_host.log
: > "$LOG"
echo "[$(date '+%F %T')] build_host start" >> "$LOG"
g++ --version | head -1 >> "$LOG"
if [ ! -f "$OUT/noidgpu.dll" ]; then
    echo "ERROR: $OUT/noidgpu.dll missing" | tee -a "$LOG"
    exit 1
fi
if [ ! -f /mingw64/include/openssl/ssl.h ]; then
    echo "ERROR: OpenSSL headers missing (/mingw64/include/openssl). Package: mingw-w64-x86_64-openssl" | tee -a "$LOG"
    exit 1
fi
ARCH="${NOID_HOST_ARCH:--march=native}"
FLAGS="-O2 $ARCH -std=gnu++17 -Wall -Wextra -Wno-unused-parameter -Isrc"
SRC="src/main.cpp src/noid_ref.cpp src/tls_conn.cpp"
if g++ $FLAGS -o "$OUT/noidminer.exe" $SRC "$OUT/noidgpu.dll" -static -lssl -lcrypto -lws2_32 -lcrypt32 -lgdi32 -luser32 -ladvapi32 -lbcrypt -pthread >> "$LOG" 2>&1; then
    echo "static link OK" >> "$LOG"
else
    echo "static link failed, trying dynamic link" >> "$LOG"
    if ! g++ $FLAGS -o "$OUT/noidminer.exe" $SRC "$OUT/noidgpu.dll" -lssl -lcrypto -lws2_32 -lcrypt32 -pthread >> "$LOG" 2>&1; then
        echo "[$(date '+%F %T')] HOST BUILD FAILED" | tee -a "$LOG"
        exit 1
    fi
    echo "dynamic link OK (needs C:\\msys64\\mingw64\\bin in PATH)" >> "$LOG"
fi
cp -f tests/vectors.txt "$OUT/"
if [ -f /mingw64/etc/ssl/certs/ca-bundle.crt ]; then cp -f /mingw64/etc/ssl/certs/ca-bundle.crt "$OUT/"; fi
echo "[$(date '+%F %T')] host OK" | tee -a "$LOG"
exit 0
