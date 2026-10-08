@echo off
REM run_test.bat - correctness tests of dev\noidminer.exe: CPU vs the official
REM test vectors, every GPU and both engine modes vs the CPU.  Log: logs\test.log
setlocal EnableExtensions
cd /d "%~dp0"
if not exist logs mkdir logs
set "PATH=%CD%\dev;C:\msys64\mingw64\bin;%PATH%"
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set "ARGS="
if exist run\test_args.txt set "ARGS=--args-file run\test_args.txt"
echo [%date% %time%] test start> logs\test.log
dev\noidminer.exe --test %ARGS% >> logs\test.log 2>&1
echo [%date% %time%] test exit=%ERRORLEVEL%>> logs\test.log
