@echo off
echo.
echo ========================================
echo   TARS Swarm Performance Monitor
echo ========================================
echo.

cd /d "%~dp0"

dotnet build TarsSwarmDemo/TarsSwarmDemo.fsproj --verbosity quiet > nul 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Build failed!
    exit /b 1
)

echo Starting performance monitoring...
echo.

dotnet run --project TarsSwarmDemo/TarsSwarmDemo.fsproj -- monitor

echo.
pause
