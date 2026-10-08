#!/usr/bin/env bash
# build_linux.sh - builds NoidMiner for Linux x86-64 (NVIDIA Pascal sm_61 and newer).
#
#   bash linux/build_linux.sh         -> build/noidminer (+ build/vectors.txt)
#
# Requirements (Ubuntu/Debian):
#   * CUDA Toolkit 12.x (12.0 .. 12.9). CUDA 13 dropped Pascal (GTX 10xx,
#     P104/P106): a binary built with it fails with "invalid device function".
#     Override the compiler with NVCC=/usr/local/cuda-12.9/bin/nvcc if needed.
#   * g++ (C++17), libssl-dev:   sudo apt install build-essential libssl-dev
#
# The CUDA runtime and the C++ runtime are linked statically, OpenSSL too when
# libssl.a/libcrypto.a are available; only the NVIDIA driver is needed at run time.
# Environment overrides: NVCC, CUDA_ARCH (default 61), NOID_HOST_ARCH (default
# -march=native; the release uses "-march=x86-64 -mtune=generic").
set -euo pipefail
cd "$(dirname "$0")/.."

NVCC="${NVCC:-}"
if [ -z "$NVCC" ]; then
    if command -v nvcc >/dev/null 2>&1; then NVCC="$(command -v nvcc)"
    elif [ -x /usr/local/cuda/bin/nvcc ]; then NVCC=/usr/local/cuda/bin/nvcc
    else echo "ERROR: nvcc not found (install the CUDA Toolkit 12.x or set NVCC=...)"; exit 1; fi
fi
VER="$("$NVCC" --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1.\2/p' | head -1)"
MAJOR="${VER%%.*}"
echo "nvcc: $NVCC (CUDA $VER)"
ARCH="${CUDA_ARCH:-61}"
if [ "$MAJOR" -ge 13 ] && [ "$ARCH" -lt 75 ]; then
    echo "ERROR: CUDA $VER cannot build for sm_$ARCH (Pascal support was removed in CUDA 13)."
    echo "       Install CUDA 12.x (e.g. 12.9) next to it and run: NVCC=/usr/local/cuda-12.9/bin/nvcc bash $0"
    exit 1
fi
if [ "$MAJOR" -lt 12 ]; then echo "WARNING: CUDA $VER is older than the tested 12.x"; fi

CUDA_HOME="$(cd "$(dirname "$NVCC")/.." && pwd)"
CUDALIB=""
for d in "$CUDA_HOME/lib64" "$CUDA_HOME/targets/x86_64-linux/lib" /usr/lib/x86_64-linux-gnu; do
    if [ -f "$d/libcudart_static.a" ]; then CUDALIB="$d"; break; fi
done
[ -n "$CUDALIB" ] || { echo "ERROR: libcudart_static.a not found under $CUDA_HOME"; exit 1; }

mkdir -p build
echo "== GPU engine (sm_$ARCH SASS + compute_$ARCH PTX for newer GPUs)"
"$NVCC" -O3 -std=c++17 -Wno-deprecated-gpu-targets \
    -gencode "arch=compute_$ARCH,code=[sm_$ARCH,compute_$ARCH]" \
    -Igpu -Isrc -c gpu/noid_gpu.cu -o build/noid_gpu.o

echo "== host program"
HOSTARCH="${NOID_HOST_ARCH:--march=native}"
FLAGS="-O2 $HOSTARCH -std=gnu++17 -Wall -Wextra -Wno-unused-parameter -Isrc"
SSL="-lssl -lcrypto"
for d in /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/lib; do
    if [ -f "$d/libssl.a" ] && [ -f "$d/libcrypto.a" ]; then SSL="$d/libssl.a $d/libcrypto.a"; break; fi
done
# glibc compatibility layer: a binary built on a recent distribution still runs
# on glibc 2.34+ (see linux/glibc_compat.c)
gcc -O2 -std=c11 -c linux/glibc_compat.c -o build/glibc_compat.o
WRAP=""
for s in __isoc23_strtol __isoc23_strtoul __isoc23_strtoll __isoc23_strtoull __isoc23_fscanf __isoc23_sscanf \
         arc4random arc4random_buf arc4random_uniform; do WRAP="$WRAP -Wl,--wrap=$s"; done
g++ $FLAGS -o build/noidminer src/main.cpp src/noid_ref.cpp src/tls_conn.cpp build/noid_gpu.o build/glibc_compat.o \
    -L"$CUDALIB" -lcudart_static $SSL -static-libstdc++ -static-libgcc -lpthread -ldl -lrt $WRAP
MAXGLIBC="$(objdump -T build/noidminer 2>/dev/null | grep -oE 'GLIBC_[0-9]+\.[0-9]+' | sort -Vu | tail -1 || true)"
[ -n "$MAXGLIBC" ] && echo "   needs $MAXGLIBC or newer"
cp -f tests/vectors.txt build/
echo "== OK: build/noidminer"
echo "   first run:  ./build/noidminer --test        (CPU reference + every GPU against the CPU)"
echo "   then:       ./build/noidminer --pool stratum+ssl://eu2.innovlab.cc:19601 --user ADDRESS.rig"
