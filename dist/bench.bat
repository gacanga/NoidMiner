@echo off
REM NoidMiner - 30 s hashrate benchmark on all GPUs (stop mining first).
cd /d "%~dp0"
set CUDA_DEVICE_ORDER=PCI_BUS_ID
noidminer.exe --config noidminer.conf --bench 30
pause
