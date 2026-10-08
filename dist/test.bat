@echo off
REM NoidMiner - correctness tests: CPU reference vs official test vectors,
REM then every GPU and every engine vs the CPU.
cd /d "%~dp0"
set CUDA_DEVICE_ORDER=PCI_BUS_ID
noidminer.exe --test
pause
