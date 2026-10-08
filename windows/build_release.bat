@echo off
REM ===========================================================================
REM build_release.bat - portable Windows release package of NoidMiner
REM   * noidgpu.dll with a static C runtime (/MT): no Visual C++ redistributable
REM   * noidminer.exe for generic x86-64 (no -march=native), statically linked
REM   * the package is tested WITHOUT MSYS2 in PATH, then zipped:
REM       release\NoidMiner-v<ver>-win64-cuda12.zip (+ .sha256.txt)
REM   Log: logs\release.log
REM ===========================================================================
setlocal EnableExtensions
cd /d "%~dp0.."
set "VER=0.2.0"
set "NAME=NoidMiner-v%VER%-win64-cuda12"
set "OUT=release\%NAME%"
if not exist logs mkdir logs
if not exist release mkdir release
set "L=%CD%\logs\release.log"
echo [%date% %time%] release %NAME%> "%L%"
if exist "%OUT%" rmdir /s /q "%OUT%"
if exist "release\%NAME%.zip" del /q "release\%NAME%.zip"
mkdir "%OUT%"

echo === noidgpu.dll (static CRT) ===
set "NOID_CRT=/MT"
call windows\build_gpu_dll.bat "%OUT%"
if errorlevel 1 (
    echo [%date% %time%] GPU DLL BUILD FAILED, see logs\build_gpu_dll.log >> "%L%"
    goto end
)
set "NOID_CRT="
del /q "%OUT%\noidgpu.lib" "%OUT%\noidgpu.exp" >nul 2>&1

echo === noidminer.exe (generic x86-64) ===
set MSYSTEM=MINGW64
set CHERE_INVOKING=1
set "NOID_HOST_ARCH=-march=x86-64 -mtune=generic"
C:\msys64\usr\bin\bash.exe -lc "./windows/build_host.sh release/%NAME%"
if errorlevel 1 (
    echo [%date% %time%] HOST BUILD FAILED, see logs\build_host.log >> "%L%"
    goto end
)
set "NOID_HOST_ARCH="

copy /y dist\noidminer.conf "%OUT%\" >nul
copy /y LICENSE "%OUT%\LICENSE.txt" >nul
copy /y NOTICE "%OUT%\NOTICE.txt" >nul
for %%f in (start.bat test.bat bench.bat README.txt) do copy /y dist\%%f "%OUT%\" >nul

echo === standalone test (no MSYS2 in PATH) ===
setlocal
set "PATH=%SystemRoot%\system32;%SystemRoot%;%SystemRoot%\System32\Wbem"
set CUDA_DEVICE_ORDER=PCI_BUS_ID
pushd "%OUT%"
noidminer.exe --test >> "%L%" 2>&1
popd
endlocal
find "TEST RESULT: ALL OK" "%L%" >nul 2>&1
if errorlevel 1 (
    echo [%date% %time%] RELEASE TEST FAILED >> "%L%"
    goto end
)

echo === zip ===
powershell -NoProfile -Command "Compress-Archive -Path '%OUT%' -DestinationPath 'release\%NAME%.zip' -Force" >> "%L%" 2>&1
powershell -NoProfile -Command "(Get-FileHash 'release\%NAME%.zip' -Algorithm SHA256).Hash.ToLower() + '  %NAME%.zip'" > "release\%NAME%.zip.sha256.txt"
type "release\%NAME%.zip.sha256.txt" >> "%L%"
dir "%OUT%" >> "%L%"
echo [%date% %time%] RELEASE OK >> "%L%"

:end
echo [%date% %time%] build_release end >> "%L%"
