@echo off
REM ===========================================================================
REM build_all.bat - development build into dev\ (the running miner uses bin\),
REM then the correctness tests (logs\test.log).
REM     build_all.bat          build dev\ + tests
REM     build_all.bat nobuild  tests only
REM   If run\exp.flag exists: profiling builds instead (exp_build.bat, logs\exp.log)
REM Logs: logs\build_gpu_dll.log, logs\build_host.log, logs\test.log,
REM       summary logs\build_all.log
REM ===========================================================================
setlocal EnableExtensions
title NoidMiner - build
cd /d "%~dp0"
if not exist logs mkdir logs
if not exist dev mkdir dev
set "S=%CD%\logs\build_all.log"
echo [%date% %time%] build_all start %* > "%S%"
if exist run\exp.flag (
    del /q run\exp.flag
    echo [%date% %time%] profiling experiments >> "%S%"
    call exp_build.bat
    goto end
)
if /i "%~1"=="nobuild" goto tests

echo === dev\noidgpu.dll (nvcc) ===
call windows\build_gpu_dll.bat dev
if errorlevel 1 (
    echo [%date% %time%] GPU DLL BUILD FAILED >> "%S%"
    echo BUILD FAILED - see logs\build_gpu_dll.log
    goto end
)
echo [%date% %time%] noidgpu.dll OK >> "%S%"

echo === dev\noidminer.exe (g++) ===
set MSYSTEM=MINGW64
set CHERE_INVOKING=1
C:\msys64\usr\bin\bash.exe -lc "./windows/build_host.sh dev"
if errorlevel 1 (
    echo [%date% %time%] HOST BUILD FAILED >> "%S%"
    echo BUILD FAILED - see logs\build_host.log
    goto end
)
echo [%date% %time%] host OK >> "%S%"

:tests
call run_test.bat
echo [%date% %time%] tests done >> "%S%"

:end
echo [%date% %time%] build_all end >> "%S%"
echo TERMINE - voir logs\build_all.log
