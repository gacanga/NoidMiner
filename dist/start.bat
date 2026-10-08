@echo off
REM NoidMiner - mines NOID with the settings of noidminer.conf.
REM Edit noidminer.conf first: replace YOUR_NOID_ADDRESS with your NOID address.
REM Restarts automatically; close this window to stop.
title NoidMiner
cd /d "%~dp0"
findstr /c:"YOUR_NOID_ADDRESS" noidminer.conf >nul
if not errorlevel 1 (
    echo.
    echo  Edit noidminer.conf: replace YOUR_NOID_ADDRESS with your NOID payout address.
    echo  Modifiez noidminer.conf : remplacez YOUR_NOID_ADDRESS par votre adresse NOID.
    echo.
    pause
    exit /b 1
)
set CUDA_DEVICE_ORDER=PCI_BUS_ID
:loop
noidminer.exe --config noidminer.conf --log noidminer.log
echo NoidMiner exited (code %ERRORLEVEL%), restarting in 10 s - close this window to stop.
timeout /t 10 /nobreak >nul
goto loop
