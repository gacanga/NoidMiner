#!/usr/bin/env bash
# Correctness tests: CPU reference vs official vectors, then every GPU and engine vs the CPU.
cd "$(dirname "$0")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
./noidminer --test
