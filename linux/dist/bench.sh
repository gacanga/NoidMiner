#!/usr/bin/env bash
# 30 s hashrate benchmark (stop mining first). Extra options are passed through,
# e.g.  ./bench.sh --gpus 0 --tpb 384
cd "$(dirname "$0")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
./noidminer --config noidminer.conf --bench 30 "$@"
