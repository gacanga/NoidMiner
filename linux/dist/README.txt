NoidMiner v0.3.1 - GPU miner for ParanO(1)d (NOID), Linux x86-64, NVIDIA CUDA 12
=================================================================================

Open-source (Apache-2.0), NO developer fee. Pool: InnovLab (parano1d-stratum-v1, TLS).
Made for NVIDIA Pascal (GTX 1060/1070/1080, P104-100, P106-100: sm_61); newer
GPUs run it through the driver's PTX JIT.

Requirements
  - Linux x86-64 with glibc 2.35+ (Ubuntu 22.04 or newer, Debian 12 or newer)
  - NVIDIA driver with CUDA 12 support (R525 or newer). Nothing else: the CUDA
    runtime, OpenSSL and the C++ runtime are built in.

QUICK START
  1. tar xzf NoidMiner-v0.3.1-linux-x64-cuda12.tar.gz && cd NoidMiner-v0.3.1-linux-x64-cuda12
  2. Edit noidminer.conf: replace YOUR_NOID_ADDRESS with your NOID payout address
     (you can also change the rig name "rig1").
  3. ./test.sh     (checks every GPU against the CPU reference)
  4. ./start.sh    (mines, restarts automatically, log in noidminer.log)

Other: ./bench.sh (30 s benchmark), ./noidminer --help

Building from source instead: see linux/build_linux.sh in the repository.
Use CUDA 12.x: CUDA 13 removed Pascal, a binary built with it stops with
"invalid device function" on GTX 10xx / P10x cards.

Source code / code source : https://github.com/gacanga/NoidMiner

--------------------------------------------------------------------------------
FRANCAIS
  1. Décompressez l'archive, puis modifiez noidminer.conf : remplacez
     YOUR_NOID_ADDRESS par votre adresse NOID.
  2. ./test.sh pour vérifier les cartes, puis ./start.sh pour miner.
  Pré-requis : Ubuntu 22.04+ / Debian 12+, pilote NVIDIA compatible CUDA 12.
