@echo off
REM exp_build.bat - PROFILING builds into exp1\ and exp2\ (wrong results, timing only)
setlocal EnableExtensions
cd /d "%~dp0"
if not exist logs mkdir logs
set "NOID_NVCC_EXTRA=-DNOID_EXPERIMENT=1"
call windows\build_gpu_dll.bat exp1
set "NOID_NVCC_EXTRA=-DNOID_EXPERIMENT=2"
call windows\build_gpu_dll.bat exp2
set "NOID_NVCC_EXTRA="
for %%d in (exp1 exp2) do copy /y dev\noidminer.exe %%d\ >nul
set "PATH=C:\msys64\mingw64\bin;%PATH%"
set CUDA_DEVICE_ORDER=PCI_BUS_ID
echo [%date% %time%] experiments> logs\exp.log
for %%d in (exp1 exp2) do (
    for %%m in (table tower) do (
        echo ===== %%d %%m>> logs\exp.log
        %%d\noidminer.exe --bench 12 --gpus 0 --mode %%m --tpb 512 >> logs\exp.log 2>&1
    )
)
