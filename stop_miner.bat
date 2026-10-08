@echo off
REM stop_miner.bat - stops NoidMiner: the auto-restart window, then any
REM remaining noidminer.exe.   stop_miner.bat nopause  (no key press)
taskkill /fi "WINDOWTITLE eq NoidMiner - miner*" /t /f >nul 2>&1
timeout /t 2 /nobreak >nul
taskkill /im noidminer.exe /t /f >nul 2>&1
cd /d "%~dp0"
if not exist logs mkdir logs
echo [%date% %time%] miner stopped by stop_miner.bat>> logs\pool.log
tasklist /fi "imagename eq noidminer.exe" 2>nul | find /i "noidminer.exe" >nul
if errorlevel 1 (echo Mineur arrete.) else (echo noidminer.exe tourne encore.)
if /i not "%~1"=="nopause" pause
