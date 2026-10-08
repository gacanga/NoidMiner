@echo off
REM deploy.bat - copies the tested build dev\ to bin\ (used by run_pool.bat).
REM Refuses if the last logs\test.log does not end with "ALL OK".
setlocal EnableExtensions
cd /d "%~dp0"
if not exist logs mkdir logs
if not exist bin mkdir bin
find "TEST RESULT: ALL OK" logs\test.log >nul 2>&1
if errorlevel 1 (
    echo [%date% %time%] deploy refused: last tests not OK>> logs\build_all.log
    echo Deploiement refuse : les derniers tests ne sont pas OK.
    exit /b 1
)
call stop_miner.bat nopause
timeout /t 2 /nobreak >nul
for %%f in (noidgpu.dll noidminer.exe cudart64_12.dll vectors.txt ca-bundle.crt) do (
    if exist dev\%%f copy /y dev\%%f bin\ >nul
)
echo [%date% %time%] deployed dev to bin>> logs\build_all.log
echo Deploye dans bin\
