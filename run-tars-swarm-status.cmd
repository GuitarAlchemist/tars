@echo off
echo.
echo ========================================
echo   TARS Swarm Status Check
echo ========================================
echo.

cd /d "%~dp0"

dotnet build TarsSwarmDemo/TarsSwarmDemo.fsproj --verbosity quiet > nul 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Build failed!
    exit /b 1
)

dotnet run --project TarsSwarmDemo/TarsSwarmDemo.fsproj -- status

echo.
pause
