@echo off
echo.
echo ========================================
echo   TARS Swarm Demo Launcher
echo ========================================
echo.

cd /d "%~dp0"

echo Building TARS Swarm Demo...
dotnet build TarsSwarmDemo/TarsSwarmDemo.fsproj

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ Build failed! Please check the errors above.
    pause
    exit /b 1
)

echo.
echo ✅ Build successful! Starting demo...
echo.

dotnet run --project TarsSwarmDemo/TarsSwarmDemo.fsproj

echo.
echo Demo completed. Press any key to exit...
pause > nul
