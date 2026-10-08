@echo off
REM ===========================================================================
REM run_pool.bat - NoidMiner on the InnovLab pool (settings: noidminer.conf).
REM   All GPUs, automatic restart 10 s after any exit.
REM   Log: logs\pool.log (appended).  Stop: stop_miner.bat or close the window.
REM ===========================================================================
setlocal EnableExtensions
title NoidMiner - miner
cd /d "%~dp0"
if not exist logs mkdir logs
set "PATH=%CD%\bin;C:\msys64\mingw64\bin;%PATH%"
set CUDA_DEVICE_ORDER=PCI_BUS_ID
:loop
echo [%date% %time%] miner start>> logs\pool.log
bin\noidminer.exe --config noidminer.conf --log logs\pool.log
echo [%date% %time%] miner exited with code %ERRORLEVEL% - restart in 10 s>> logs\pool.log
echo [%date% %time%] arret du mineur (code %ERRORLEVEL%), relance dans 10 s
timeout /t 10 /nobreak >nul
goto loop
