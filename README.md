# NoidMiner

Open-source GPU miner for **ParanO(1)d (NOID)**, a Poseidon2b proof of work over GF(2^128).
It runs on **Windows x64** with **NVIDIA CUDA** and connects to the **InnovLab** pool.

- **No developer fee.** Apache-2.0.
- Built for **NVIDIA Pascal** (P104-100, GTX 1060/1070/1080, sm_61). The other NOID miners (INVminer, GB Miner, Fl4shMiner, SRBMiner-Multi) only support RTX 30 and newer.
- Every share is re-checked on the CPU against a reference implementation before it is submitted.

*Version française plus bas.*

## Download

Get `NoidMiner-v0.2.0-win64-cuda12.zip` from the [Releases](../../releases) page and check its SHA-256.

## Quick start

1. Unzip the release.
2. In `noidminer.conf`, replace `YOUR_NOID_ADDRESS` with your NOID payout address. Keep `.rig1` or rename the rig.
3. Run `start.bat`. It restarts the miner automatically and logs to `noidminer.log`.

| Script | Purpose |
|---|---|
| `start.bat` | Mine with `noidminer.conf` |
| `test.bat` | Correctness tests: the CPU reference against official test vectors, then every GPU and every engine against the CPU |
| `bench.bat` | 30 s hashrate benchmark |

Command line help: `noidminer.exe --help`. The main options are:
- `--pool` (repeat it for failover pools), `--user`, `--pass`
- `--gpus 0,1|all`
- `--mode table|tower|kop`
- `--tpb 256|384|512`
- `--npt N`, `--target-ms N`
- `--test`, `--bench [s]`

## Performance

| GPU | MH/s | Notes |
|---|---|---|
| P104-100 (stock, ~175 W) | ~4.75 | engine `table`, 512 threads per block |

This proof of work needs no memory bandwidth. Lowering the memory clock or the power limit saves power for little hashrate.

## Requirements

- Windows 10/11 x64.
- NVIDIA driver with CUDA 12 support.
- Pascal (sm_61) runs a native kernel. Newer GPUs work through the driver's PTX JIT, but dedicated miners are much faster on them.

## How it works

- **PoW:** a Poseidon2b sponge (t=4, rate 2, x^7, 8 full + 58 partial rounds) over 16 header fields, with the nonce in field 10. A block is valid when `digest < target`, both read as 256-bit little-endian.
- **Midstate:** fields 0–9 do not depend on the nonce, so the miner precomputes the state after 5 permutations. That leaves 3 permutations per nonce.
- **Carry-less multiply without CLMUL:**
  - GF(2^128) arithmetic runs in the polynomial basis modulo x^128+x^7+x^2+x+1.
  - The 16×16 carry-less products use plain 16-bit integer multiplies (XMAD) on operands split into 3 bit classes (bit index mod 3), so integer carries never collide.
  - Karatsuba goes 16 → 32 → 64 → 128 bits, for 27 products per 128-bit multiply.
- **MDS constants:** the `table` engine multiplies by them with 4-bit tables in shared memory. The alternatives are the `tower` engine (partial rounds in the tower basis) and `kop` (pure ALU).
- **Validation:** all GPU arithmetic also compiles as plain C++. `tests/test_math.cpp` checks it against `src/noid_ref.cpp`, which itself matches 64 vectors produced by the official crates (`tools/noidvec`).

### Pool protocol (InnovLab, `parano1d-stratum-v1`)

- Transport is TLS with one JSON object per line: `mining.subscribe` → `mining.authorize` → `mining.notify` / `mining.pause` → `mining.submit`. Hashrate is reported with `miner.stats`.
- The nonce is 16 bytes, little-endian. Its upper 8 bytes are the raw bytes of `nonce_prefix_hex`.

## Building from source

Prerequisites:
- CUDA Toolkit 12.9 (the last version that targets Pascal);
- Visual Studio 2022 with the C++ x64 tools;
- MSYS2 MINGW64 with `mingw-w64-x86_64-gcc` and `mingw-w64-x86_64-openssl`.

| Command | Result |
|---|---|
| `build_all.bat` | Builds `dev\noidgpu.dll` (nvcc + MSVC) and `dev\noidminer.exe` (g++ + OpenSSL, static), then runs the tests (`logs\test.log`) |
| `run_bench.bat` | Benchmark |
| `deploy.bat` | Copies `dev\` to `bin\`, only if the tests pass |
| `run_pool.bat` | Mines from `bin\` |
| `windows\build_release.bat` | Portable package: static C runtime, generic x86-64, tested without MSYS2, zipped |

On Linux, validate the arithmetic on the CPU with:

```
g++ -O2 -std=c++17 -Isrc -Igpu tests/test_math.cpp src/noid_ref.cpp && ./a.out
```

## Credits and license

- Apache License 2.0, see `LICENSE` and `NOTICE`.
- Constants and test vectors come from [ParanO(1)d](https://github.com/ignotusnemo/parano1d) (Apache-2.0). The PoW is unchanged between v2.0.2 and v2.0.3.
- The software is provided as is, without warranty. Mining consumes electricity, so check that it is profitable for you.

---

## Français

Mineur GPU open source pour **ParanO(1)d (NOID)**, sous **Windows x64** avec **NVIDIA CUDA**, pour la pool **InnovLab**.

- **Sans frais de développeur**, licence Apache-2.0.
- Pensé pour les cartes **Pascal** (P104-100, GTX 10xx), que les autres mineurs NOID ne gèrent pas.

**Démarrage :**
1. Téléchargez le zip dans [Releases](../../releases).
2. Dans `noidminer.conf`, remplacez `YOUR_NOID_ADDRESS` par votre adresse NOID.
3. Lancez `start.bat`.

`test.bat` vérifie les calculs et `bench.bat` mesure le hashrate.

**Performance :** environ 4,75 MH/s par P104-100. L'algorithme n'utilise pas la mémoire : baisser l'horloge mémoire ou la limite de puissance économise du courant sans perte notable de hashrate.
