@echo off
REM stop_miner.bat - stops NoidMiner: the auto-restart loop(s) (run_pool.bat),
REM then any remaining noidminer.exe.   stop_miner.bat nopause  (no key press)
REM The loop is found by its command line, so it works whatever the window
REM title is (an elevated window is titled "Administrateur : NoidMiner - miner").
taskkill /fi "WINDOWTITLE eq NoidMiner - miner*" /t /f >nul 2>&1
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='cmd.exe'\" | Where-Object { $_.CommandLine -match 'run_pool\.bat' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >nul 2>&1
timeout /t 2 /nobreak >nul
taskkill /im noidminer.exe /t /f >nul 2>&1
cd /d "%~dp0"
if not exist logs mkdir logs
echo [%date% %time%] miner stopped by stop_miner.bat>> logs\pool.log
tasklist /fi "imagename eq noidminer.exe" 2>nul | find /i "noidminer.exe" >nul
if errorlevel 1 (echo Mineur arrete.) else (echo noidminer.exe tourne encore.)
if /i not "%~1"=="nopause" pause
