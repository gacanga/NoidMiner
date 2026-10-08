NoidMiner v0.3.0 - GPU miner for ParanO(1)d (NOID), Windows x64, NVIDIA CUDA 12
================================================================================

Open-source (Apache-2.0), NO developer fee.
Made for NVIDIA Pascal cards (P104-100, GTX 1070/1080...), which the other NOID
miners do not support. Pool: InnovLab (protocol parano1d-stratum-v1 over TLS).

QUICK START
  1. Open noidminer.conf and replace YOUR_NOID_ADDRESS with your NOID payout
     address (you can also change the rig name "rig1").
  2. Double-click start.bat.

Other scripts
  test.bat   correctness tests: CPU vs official test vectors, every GPU vs CPU
  bench.bat  30 s hashrate benchmark (stop mining first)

Requirements
  - Windows 10/11 x64, NVIDIA GPU with a recent driver (CUDA 12)
  - Native kernel for Pascal (sm_61); newer GPUs run it through the driver's JIT
    (works, but dedicated miners are much faster on RTX 30 and newer)

Reference hashrate: P104-100 ~5.2 MH/s (stock, ~175 W).
The algorithm needs no memory bandwidth: lowering the memory clock or the power
limit saves power at little hashrate cost.

Command line: noidminer.exe --help

--------------------------------------------------------------------------------
FRANCAIS

Mineur GPU open source (Apache-2.0), SANS frais de developpeur, pour ParanO(1)d
(NOID). Pensé pour les cartes NVIDIA Pascal (P104-100, GTX 1070/1080...), que les
autres mineurs NOID ne prennent pas en charge. Pool : InnovLab.

Démarrage rapide
  1. Ouvrez noidminer.conf et remplacez YOUR_NOID_ADDRESS par votre adresse NOID.
  2. Double-cliquez sur start.bat.

test.bat : tests de calcul. bench.bat : mesure du hashrate pendant 30 s.
Hashrate de référence : P104-100 ~5,2 MH/s.
L'algorithme n'utilise pas la mémoire : baisser l'horloge mémoire ou la limite
de puissance économise du courant sans perte notable de hashrate.

Source code / code source : https://github.com/gacanga/NoidMiner
